"""
腾讯云微搭低代码平台适配器模块

该模块提供腾讯云微搭(WeDa)低代码平台的通道适配器实现，支持：
- 应用管理（创建、查询、发布）
- 数据模型操作（增删改查）
- 工作流触发与管理
- 数据源配置
- 环境管理

API文档: https://cloud.tencent.com/document/product/1369
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import aiohttp
from aiohttp import ClientTimeout

from ...base import (
    ChannelAdapter,
    ChannelCapability,
    ChannelConfig,
    ConnectionState,
    MessagePriority,
    ReceiveResult,
    RetryConfig,
    SendResult,
)
from ...universal_message import (
    Attachment,
    AttachmentType,
    ChannelIdentity,
    MessageContent,
    MessageDirection,
    MessageMetadata,
    MessageType,
    UniversalMessage,
    UserIdentity,
)

logger = logging.getLogger(__name__)


class AppStatus(Enum):
    """应用状态"""
    DRAFT = "DRAFT"             # 草稿
    PUBLISHED = "PUBLISHED"     # 已发布
    OFFLINE = "OFFLINE"         # 已下线


class DataType(Enum):
    """数据类型"""
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    ARRAY = "array"
    OBJECT = "object"
    FILE = "file"
    IMAGE = "image"


@dataclass
class WeDaConfig(ChannelConfig):
    """腾讯云微搭配置类
    
    Attributes:
        secret_id: 腾讯云SecretId
        secret_key: 腾讯云SecretKey
        endpoint: API端点
        region: 地域ID
        connect_timeout: 连接超时时间（秒）
        read_timeout: 读取超时时间（秒）
        max_retry: 最大重试次数
        default_env_id: 默认环境ID
    """
    secret_id: str = ""
    secret_key: str = ""
    endpoint: str = "weda.tencentcloudapi.com"
    region: str = "ap-guangzhou"
    connect_timeout: int = 30
    read_timeout: int = 60
    max_retry: int = 3
    default_env_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.secret_id or not self.secret_key:
            raise WeDaConfigError("SecretId和SecretKey不能为空")


class WeDaConfigError(Exception):
    """微搭配置错误"""
    pass


class WeDaAPIError(Exception):
    """微搭API错误
    
    Attributes:
        code: 错误代码
        message: 错误信息
        request_id: 请求ID
    """
    
    def __init__(
        self,
        code: str,
        message: str,
        request_id: Optional[str] = None,
    ):
        self.code = code
        self.message = message
        self.request_id = request_id
        super().__init__(f"[{code}] {message} (RequestId: {request_id})")
    
    @property
    def is_retryable(self) -> bool:
        """判断错误是否可重试"""
        retryable_codes = ["InternalError", "RequestTimeout"]
        return self.code in retryable_codes


@dataclass
class WeDaApp:
    """微搭应用信息
    
    Attributes:
        app_id: 应用ID
        app_name: 应用名称
        app_type: 应用类型
        status: 应用状态
        description: 描述
        create_time: 创建时间
        update_time: 更新时间
    """
    app_id: str
    app_name: str
    app_type: str = ""
    status: AppStatus = AppStatus.DRAFT
    description: str = ""
    create_time: Optional[str] = None
    update_time: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "app_id": self.app_id,
            "app_name": self.app_name,
            "app_type": self.app_type,
            "status": self.status.value,
            "description": self.description,
            "create_time": self.create_time,
            "update_time": self.update_time,
        }


@dataclass
class DataModel:
    """数据模型定义
    
    Attributes:
        model_id: 模型ID
        model_name: 模型名称
        fields: 字段定义
        description: 描述
    """
    model_id: str
    model_name: str
    fields: List[Dict[str, Any]] = field(default_factory=list)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "fields": self.fields,
            "description": self.description,
        }


@dataclass
class Workflow:
    """工作流定义
    
    Attributes:
        workflow_id: 工作流ID
        workflow_name: 工作流名称
        trigger_type: 触发类型
        status: 状态
        description: 描述
    """
    workflow_id: str
    workflow_name: str
    trigger_type: str = ""
    status: str = "ENABLED"
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "trigger_type": self.trigger_type,
            "status": self.status,
            "description": self.description,
        }


class TencentWeDaAdapter(ChannelAdapter):
    """腾讯云微搭适配器
    
    提供腾讯云微搭低代码平台的统一接口，支持应用管理、数据模型操作、
    工作流触发等功能。
    
    Example:
        config = WeDaConfig(
            secret_id="your-secret-id",
            secret_key="your-secret-key",
            endpoint="weda.tencentcloudapi.com",
            region="ap-guangzhou",
        )
        adapter = TencentWeDaAdapter(config)
        await adapter.connect()
        
        # 列出应用
        apps = await adapter.list_apps()
        
        # 创建数据记录
        record = await adapter.create_record(
            model_id="model-xxx",
            data={"field1": "value1"},
        )
        
        # 触发工作流
        await adapter.trigger_workflow("workflow-xxx", {"key": "value"})
    """
    
    SERVICE = "weda"
    VERSION = "2021-04-20"
    
    def __init__(self, config: WeDaConfig) -> None:
        super().__init__(config)
        self._cfg = config
        self._session: Optional[aiohttp.ClientSession] = None
    
    def _initialize_capabilities(self) -> None:
        """初始化适配器能力"""
        self._capabilities = {
            ChannelCapability.TEXT_MESSAGES,
            ChannelCapability.FORM_DATA,
            ChannelCapability.WORKFLOW,
            ChannelCapability.CHANNEL_INFO,
        }
    
    def _get_base_url(self) -> str:
        """获取基础URL"""
        return f"https://{self._cfg.endpoint}"
    
    def _sign_request(
        self,
        method: str,
        url_path: str,
        headers: Dict[str, str],
        payload: str = "",
    ) -> Dict[str, str]:
        """生成请求签名（腾讯云API V3签名）"""
        timestamp = str(int(time.time()))
        date = datetime.utcnow().strftime("%Y-%m-%d")
        
        http_method = method.upper()
        canonical_uri = url_path
        canonical_querystring = ""
        
        signed_headers = "content-type;host;x-tc-action;x-tc-timestamp;x-tc-version"
        canonical_headers = f"content-type:{headers.get('Content-Type', 'application/json')}\n"
        canonical_headers += f"host:{headers['Host']}\n"
        canonical_headers += f"x-tc-action:{headers['X-TC-Action']}\n"
        canonical_headers += f"x-tc-timestamp:{headers['X-TC-Timestamp']}\n"
        canonical_headers += f"x-tc-version:{headers['X-TC-Version']}\n"
        
        payload_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        
        canonical_request = f"{http_method}\n{canonical_uri}\n{canonical_querystring}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
        
        credential_scope = f"{date}/{self.SERVICE}/tc3_request"
        string_to_sign = f"TC3-HMAC-SHA256\n{timestamp}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
        
        secret_date = hmac.new(
            f"TC3{self._cfg.secret_key}".encode("utf-8"),
            date.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        secret_service = hmac.new(secret_date, self.SERVICE.encode("utf-8"), hashlib.sha256).digest()
        secret_signing = hmac.new(secret_service, "tc3_request".encode("utf-8"), hashlib.sha256).digest()
        signature = hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
        
        authorization = f"TC3-HMAC-SHA256 Credential={self._cfg.secret_id}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}"
        
        headers["Authorization"] = authorization
        return headers
    
    def _get_auth_headers(self, action: str, payload: str = "") -> Dict[str, str]:
        """获取认证头"""
        headers = {
            "Content-Type": "application/json",
            "Host": self._cfg.endpoint,
            "X-TC-Action": action,
            "X-TC-Version": self.VERSION,
            "X-TC-Timestamp": str(int(time.time())),
            "X-TC-Region": self._cfg.region,
        }
        
        return self._sign_request("POST", "/", headers, payload)
    
    async def _make_request(
        self,
        action: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """发送API请求"""
        if not self._session:
            raise WeDaAPIError("NotConnected", "适配器未连接")
        
        payload = json.dumps(params)
        headers = self._get_auth_headers(action, payload)
        
        url = self._get_base_url()
        timeout = ClientTimeout(
            connect=self._cfg.connect_timeout,
            total=self._cfg.read_timeout,
        )
        
        try:
            async with self._session.post(
                url=url,
                headers=headers,
                data=payload,
                timeout=timeout,
            ) as response:
                body = await response.read()
                result = json.loads(body.decode("utf-8"))
                
                if "Response" in result:
                    response_data = result["Response"]
                    if "Error" in response_data:
                        error = response_data["Error"]
                        raise WeDaAPIError(
                            error.get("Code", "UnknownError"),
                            error.get("Message", "Unknown error"),
                            response_data.get("RequestId"),
                        )
                    return response_data
                
                return result
                
        except aiohttp.ClientError as e:
            raise WeDaAPIError("RequestError", str(e))
        except json.JSONDecodeError as e:
            raise WeDaAPIError("ParseError", f"JSON解析失败: {e}")
    
    async def _connect_impl(self) -> bool:
        """实现连接逻辑"""
        try:
            timeout = ClientTimeout(
                connect=self._cfg.connect_timeout,
                total=self._cfg.read_timeout,
            )
            self._session = aiohttp.ClientSession(timeout=timeout)
            
            await self.list_apps()
            
            self._logger.info(f"已连接到腾讯云微搭: {self._cfg.endpoint}")
            return True
        except Exception as e:
            self._logger.error(f"连接腾讯云微搭失败: {e}")
            return False
    
    async def _disconnect_impl(self) -> None:
        """实现断开连接逻辑"""
        if self._session:
            await self._session.close()
            self._session = None
        self._logger.info("已断开与腾讯云微搭的连接")
    
    async def _health_check_impl(self) -> bool:
        """实现健康检查逻辑"""
        try:
            await self.list_apps()
            return True
        except Exception:
            return False
    
    async def _send_impl(self, message: UniversalMessage, priority: MessagePriority) -> SendResult:
        """实现发送逻辑"""
        try:
            operation = message.get_context("operation", "create_record")
            
            if operation == "create_record":
                model_id = message.get_context("model_id")
                data = message.get_context("data", {})
                
                if not model_id:
                    return SendResult(
                        success=False,
                        error="缺少model_id参数",
                        error_code="MISSING_PARAMS",
                    )
                
                result = await self.create_record(model_id, data)
                return SendResult(
                    success=True,
                    message_id=result.get("record_id", ""),
                    timestamp=time.time(),
                )
            
            elif operation == "trigger_workflow":
                workflow_id = message.get_context("workflow_id")
                data = message.get_context("data", {})
                
                if not workflow_id:
                    return SendResult(
                        success=False,
                        error="缺少workflow_id参数",
                        error_code="MISSING_PARAMS",
                    )
                
                await self.trigger_workflow(workflow_id, data)
                return SendResult(success=True, timestamp=time.time())
            
            else:
                return SendResult(
                    success=False,
                    error=f"不支持的操作: {operation}",
                    error_code="UNSUPPORTED_OPERATION",
                )
                
        except WeDaAPIError as e:
            return SendResult(
                success=False,
                error=e.message,
                error_code=e.code,
            )
        except Exception as e:
            return SendResult(
                success=False,
                error=str(e),
                error_code="UNKNOWN_ERROR",
            )
    
    async def _receive_impl(self, payload: Optional[Dict]) -> ReceiveResult:
        """实现接收逻辑"""
        try:
            operation = payload.get("operation", "query") if payload else "query"
            messages = []
            
            if operation == "query_records":
                model_id = payload.get("model_id") if payload else None
                if model_id:
                    records = await self.query_records(model_id)
                    for record in records:
                        content = MessageContent(text=f"[Record: {record.get('id', '')}]")
                        metadata = MessageMetadata(
                            message_id=str(record.get('id', '')),
                            channel_id="tencent_weda",
                            timestamp=time.time(),
                            direction=MessageDirection.INBOUND,
                            message_type=MessageType.DATA,
                        )
                        msg = UniversalMessage(content=content, metadata=metadata)
                        msg.set_context("record", record)
                        messages.append(msg)
            
            return ReceiveResult(success=True, messages=messages)
        except Exception as e:
            return ReceiveResult(success=False, error=str(e))
    
    # ========== 应用管理 ==========
    
    async def list_apps(
        self,
        env_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> List[WeDaApp]:
        """列出应用
        
        Args:
            env_id: 环境ID
            page: 页码
            page_size: 每页数量
            
        Returns:
            应用列表
        """
        env_id = env_id or self._cfg.default_env_id
        
        params = {
            "PageNumber": page,
            "PageSize": page_size,
        }
        
        if env_id:
            params["EnvId"] = env_id
        
        result = await self._make_request("DescribeApplications", params)
        
        apps = []
        for item in result.get("Applications", []):
            app = WeDaApp(
                app_id=item.get("AppId", ""),
                app_name=item.get("AppName", ""),
                app_type=item.get("AppType", ""),
                status=AppStatus(item.get("Status", "DRAFT")),
                description=item.get("Description", ""),
                create_time=item.get("CreateTime"),
                update_time=item.get("UpdateTime"),
            )
            apps.append(app)
        
        return apps
    
    async def get_app(self, app_id: str) -> Optional[WeDaApp]:
        """获取应用详情
        
        Args:
            app_id: 应用ID
            
        Returns:
            应用信息
        """
        params = {"AppId": app_id}
        
        result = await self._make_request("DescribeApplication", params)
        data = result.get("Application", {})
        
        if not data:
            return None
        
        return WeDaApp(
            app_id=data.get("AppId", ""),
            app_name=data.get("AppName", ""),
            app_type=data.get("AppType", ""),
            status=AppStatus(data.get("Status", "DRAFT")),
            description=data.get("Description", ""),
            create_time=data.get("CreateTime"),
            update_time=data.get("UpdateTime"),
        )
    
    async def publish_app(self, app_id: str) -> bool:
        """发布应用
        
        Args:
            app_id: 应用ID
            
        Returns:
            是否成功
        """
        params = {"AppId": app_id}
        
        await self._make_request("PublishApplication", params)
        return True
    
    # ========== 数据模型操作 ==========
    
    async def list_data_models(
        self,
        env_id: Optional[str] = None,
    ) -> List[DataModel]:
        """列出数据模型
        
        Args:
            env_id: 环境ID
            
        Returns:
            数据模型列表
        """
        env_id = env_id or self._cfg.default_env_id
        
        params = {}
        if env_id:
            params["EnvId"] = env_id
        
        result = await self._make_request("DescribeDataModels", params)
        
        models = []
        for item in result.get("DataModels", []):
            model = DataModel(
                model_id=item.get("ModelId", ""),
                model_name=item.get("ModelName", ""),
                fields=item.get("Fields", []),
                description=item.get("Description", ""),
            )
            models.append(model)
        
        return models
    
    async def create_record(
        self,
        model_id: str,
        data: Dict[str, Any],
        env_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """创建数据记录
        
        Args:
            model_id: 模型ID
            data: 记录数据
            env_id: 环境ID
            
        Returns:
            创建结果
        """
        env_id = env_id or self._cfg.default_env_id
        
        params = {
            "ModelId": model_id,
            "Data": data,
        }
        
        if env_id:
            params["EnvId"] = env_id
        
        result = await self._make_request("CreateRecord", params)
        
        return {
            "success": True,
            "record_id": result.get("RecordId", ""),
            "model_id": model_id,
        }
    
    async def get_record(
        self,
        model_id: str,
        record_id: str,
        env_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """获取数据记录
        
        Args:
            model_id: 模型ID
            record_id: 记录ID
            env_id: 环境ID
            
        Returns:
            记录数据
        """
        env_id = env_id or self._cfg.default_env_id
        
        params = {
            "ModelId": model_id,
            "RecordId": record_id,
        }
        
        if env_id:
            params["EnvId"] = env_id
        
        result = await self._make_request("DescribeRecord", params)
        return result.get("Data")
    
    async def update_record(
        self,
        model_id: str,
        record_id: str,
        data: Dict[str, Any],
        env_id: Optional[str] = None,
    ) -> bool:
        """更新数据记录
        
        Args:
            model_id: 模型ID
            record_id: 记录ID
            data: 更新数据
            env_id: 环境ID
            
        Returns:
            是否成功
        """
        env_id = env_id or self._cfg.default_env_id
        
        params = {
            "ModelId": model_id,
            "RecordId": record_id,
            "Data": data,
        }
        
        if env_id:
            params["EnvId"] = env_id
        
        await self._make_request("UpdateRecord", params)
        return True
    
    async def delete_record(
        self,
        model_id: str,
        record_id: str,
        env_id: Optional[str] = None,
    ) -> bool:
        """删除数据记录
        
        Args:
            model_id: 模型ID
            record_id: 记录ID
            env_id: 环境ID
            
        Returns:
            是否成功
        """
        env_id = env_id or self._cfg.default_env_id
        
        params = {
            "ModelId": model_id,
            "RecordId": record_id,
        }
        
        if env_id:
            params["EnvId"] = env_id
        
        await self._make_request("DeleteRecord", params)
        return True
    
    async def query_records(
        self,
        model_id: str,
        env_id: Optional[str] = None,
        filter_expr: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> List[Dict[str, Any]]:
        """查询数据记录
        
        Args:
            model_id: 模型ID
            env_id: 环境ID
            filter_expr: 过滤条件
            page: 页码
            page_size: 每页数量
            
        Returns:
            记录列表
        """
        env_id = env_id or self._cfg.default_env_id
        
        params = {
            "ModelId": model_id,
            "PageNumber": page,
            "PageSize": page_size,
        }
        
        if env_id:
            params["EnvId"] = env_id
        if filter_expr:
            params["Filter"] = filter_expr
        
        result = await self._make_request("DescribeRecords", params)
        return result.get("Records", [])
    
    # ========== 工作流操作 ==========
    
    async def list_workflows(
        self,
        env_id: Optional[str] = None,
    ) -> List[Workflow]:
        """列出工作流
        
        Args:
            env_id: 环境ID
            
        Returns:
            工作流列表
        """
        env_id = env_id or self._cfg.default_env_id
        
        params = {}
        if env_id:
            params["EnvId"] = env_id
        
        result = await self._make_request("DescribeWorkflows", params)
        
        workflows = []
        for item in result.get("Workflows", []):
            workflow = Workflow(
                workflow_id=item.get("WorkflowId", ""),
                workflow_name=item.get("WorkflowName", ""),
                trigger_type=item.get("TriggerType", ""),
                status=item.get("Status", "ENABLED"),
                description=item.get("Description", ""),
            )
            workflows.append(workflow)
        
        return workflows
    
    async def trigger_workflow(
        self,
        workflow_id: str,
        data: Dict[str, Any],
        env_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """触发工作流
        
        Args:
            workflow_id: 工作流ID
            data: 触发数据
            env_id: 环境ID
            
        Returns:
            触发结果
        """
        env_id = env_id or self._cfg.default_env_id
        
        params = {
            "WorkflowId": workflow_id,
            "Data": data,
        }
        
        if env_id:
            params["EnvId"] = env_id
        
        result = await self._make_request("TriggerWorkflow", params)
        
        return {
            "success": True,
            "execution_id": result.get("ExecutionId", ""),
            "workflow_id": workflow_id,
        }
    
    async def get_workflow_executions(
        self,
        workflow_id: str,
        env_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> List[Dict[str, Any]]:
        """获取工作流执行记录
        
        Args:
            workflow_id: 工作流ID
            env_id: 环境ID
            page: 页码
            page_size: 每页数量
            
        Returns:
            执行记录列表
        """
        env_id = env_id or self._cfg.default_env_id
        
        params = {
            "WorkflowId": workflow_id,
            "PageNumber": page,
            "PageSize": page_size,
        }
        
        if env_id:
            params["EnvId"] = env_id
        
        result = await self._make_request("DescribeWorkflowExecutions", params)
        return result.get("Executions", [])
    
    # ========== 工具方法 ==========
    
    def get_capabilities(self):
        """获取适配器能力"""
        return self._capabilities
    
    def supports_capability(self, capability) -> bool:
        """检查是否支持特定能力"""
        return capability in self._capabilities
    
    def __repr__(self) -> str:
        return f"TencentWeDaAdapter(endpoint={self._cfg.endpoint}, region={self._cfg.region})"


# ========== CLI测试代码 ==========

async def test_tencent_weda():
    """测试腾讯云微搭适配器"""
    import os
    
    # 从环境变量获取配置
    secret_id = os.environ.get("TENCENT_SECRET_ID", "test-secret-id")
    secret_key = os.environ.get("TENCENT_SECRET_KEY", "test-secret-key")
    endpoint = os.environ.get("TENCENT_WEDA_ENDPOINT", "weda.tencentcloudapi.com")
    region = os.environ.get("TENCENT_REGION", "ap-guangzhou")
    
    config = WeDaConfig(
        secret_id=secret_id,
        secret_key=secret_key,
        endpoint=endpoint,
        region=region,
    )
    
    adapter = TencentWeDaAdapter(config)
    
    try:
        # 连接
        print("正在连接腾讯云微搭...")
        connected = await adapter.connect()
        print(f"连接结果: {connected}")
        
        if not connected:
            print("连接失败，请检查配置")
            return
        
        # 健康检查
        healthy = await adapter.health_check()
        print(f"健康检查: {healthy}")
        
        print("\n测试完成!")
        
    except Exception as e:
        print(f"测试出错: {e}")
    finally:
        await adapter.disconnect()


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # 运行测试
    asyncio.run(test_tencent_weda())
