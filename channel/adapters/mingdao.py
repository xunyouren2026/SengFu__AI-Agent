"""
明道云零代码平台适配器模块

该模块提供明道云(Mingdao)零代码平台的通道适配器实现，支持：
- 应用管理（创建、查询、更新）
- 数据表操作（增删改查）
- 工作流触发
- 表单提交
- 数据查询与筛选

API文档: https://www.mingdao.com/api-doc
"""

from __future__ import annotations

import asyncio
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


class AppType(Enum):
    """应用类型"""
    CUSTOM = "custom"       # 自定义应用
    TEMPLATE = "template"   # 模板应用
    SYSTEM = "system"       # 系统应用


class FieldType(Enum):
    """字段类型"""
    TEXT = "text"
    NUMBER = "number"
    DATE = "date"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    SELECT = "select"
    MULTISELECT = "multiselect"
    USER = "user"
    DEPARTMENT = "department"
    FILE = "file"
    IMAGE = "image"
    RELATION = "relation"
    FORMULA = "formula"


@dataclass
class MingdaoConfig(ChannelConfig):
    """明道云配置类
    
    Attributes:
        app_key: 应用AppKey
        app_secret: 应用AppSecret
        base_url: API基础URL
        connect_timeout: 连接超时时间（秒）
        read_timeout: 读取超时时间（秒）
        max_retry: 最大重试次数
        default_app_id: 默认应用ID
    """
    app_key: str = ""
    app_secret: str = ""
    base_url: str = "https://api.mingdao.com/v1"
    connect_timeout: int = 30
    read_timeout: int = 60
    max_retry: int = 3
    default_app_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.app_key or not self.app_secret:
            raise MingdaoConfigError("AppKey和AppSecret不能为空")


class MingdaoConfigError(Exception):
    """明道云配置错误"""
    pass


class MingdaoAPIError(Exception):
    """明道云API错误
    
    Attributes:
        code: 错误代码
        message: 错误信息
    """
    
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")
    
    @property
    def is_retryable(self) -> bool:
        """判断错误是否可重试"""
        return self.code >= 500


@dataclass
class MingdaoApp:
    """明道云应用信息
    
    Attributes:
        app_id: 应用ID
        app_name: 应用名称
        app_type: 应用类型
        description: 描述
        create_time: 创建时间
        update_time: 更新时间
    """
    app_id: str
    app_name: str
    app_type: AppType = AppType.CUSTOM
    description: str = ""
    create_time: Optional[str] = None
    update_time: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "app_id": self.app_id,
            "app_name": self.app_name,
            "app_type": self.app_type.value,
            "description": self.description,
            "create_time": self.create_time,
            "update_time": self.update_time,
        }


@dataclass
class DataTable:
    """数据表信息
    
    Attributes:
        table_id: 表ID
        table_name: 表名称
        app_id: 所属应用ID
        fields: 字段定义
        description: 描述
    """
    table_id: str
    table_name: str
    app_id: str = ""
    fields: List[Dict[str, Any]] = field(default_factory=list)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "table_id": self.table_id,
            "table_name": self.table_name,
            "app_id": self.app_id,
            "fields": self.fields,
            "description": self.description,
        }


@dataclass
class TableRecord:
    """数据表记录
    
    Attributes:
        record_id: 记录ID
        table_id: 表ID
        data: 记录数据
        create_time: 创建时间
        update_time: 更新时间
        creator: 创建者
    """
    record_id: str
    table_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    create_time: Optional[str] = None
    update_time: Optional[str] = None
    creator: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "record_id": self.record_id,
            "table_id": self.table_id,
            "data": self.data,
            "create_time": self.create_time,
            "update_time": self.update_time,
            "creator": self.creator,
        }


@dataclass
class Workflow:
    """工作流信息
    
    Attributes:
        workflow_id: 工作流ID
        workflow_name: 工作流名称
        app_id: 所属应用ID
        trigger_type: 触发类型
        status: 状态
    """
    workflow_id: str
    workflow_name: str
    app_id: str = ""
    trigger_type: str = ""
    status: str = "ENABLED"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "app_id": self.app_id,
            "trigger_type": self.trigger_type,
            "status": self.status,
        }


class MingdaoAdapter(ChannelAdapter):
    """明道云适配器
    
    提供明道云零代码平台的统一接口，支持应用管理、数据表操作、
    工作流触发、表单提交等功能。
    
    Example:
        config = MingdaoConfig(
            app_key="your-app-key",
            app_secret="your-app-secret",
        )
        adapter = MingdaoAdapter(config)
        await adapter.connect()
        
        # 列出应用
        apps = await adapter.list_apps()
        
        # 创建记录
        record = await adapter.create_record(
            table_id="table-xxx",
            data={"field1": "value1"},
        )
        
        # 触发工作流
        await adapter.trigger_workflow("workflow-xxx", {"key": "value"})
    """
    
    def __init__(self, config: MingdaoConfig) -> None:
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
        return self._cfg.base_url
    
    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """生成签名
        
        Args:
            params: 请求参数
            
        Returns:
            签名字符串
        """
        # 按key排序
        sorted_params = sorted(params.items())
        # 构建签名字符串
        sign_string = urllib.parse.urlencode(sorted_params)
        sign_string += f"&appSecret={self._cfg.app_secret}"
        # MD5加密
        return hashlib.md5(sign_string.encode("utf-8")).hexdigest().upper()
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """发送API请求
        
        Args:
            method: HTTP方法
            endpoint: API端点
            params: URL参数
            data: 请求数据
            
        Returns:
            API响应
        """
        if not self._session:
            raise MingdaoAPIError(-1, "适配器未连接")
        
        url = f"{self._get_base_url()}{endpoint}"
        
        # 添加公共参数
        if params is None:
            params = {}
        params["appKey"] = self._cfg.app_key
        params["timestamp"] = str(int(time.time() * 1000))
        
        # 生成签名
        params["sign"] = self._generate_signature(params)
        
        headers = {
            "Content-Type": "application/json",
        }
        
        timeout = ClientTimeout(
            connect=self._cfg.connect_timeout,
            total=self._cfg.read_timeout,
        )
        
        try:
            if method.upper() == "GET":
                async with self._session.get(
                    url=url,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                ) as response:
                    result = await response.json()
            else:
                async with self._session.post(
                    url=url,
                    params=params,
                    json=data,
                    headers=headers,
                    timeout=timeout,
                ) as response:
                    result = await response.json()
            
            # 检查错误
            if result.get("error_code") != 1:
                raise MingdaoAPIError(
                    result.get("error_code", -1),
                    result.get("error_msg", "Unknown error"),
                )
            
            return result.get("data", {})
            
        except aiohttp.ClientError as e:
            raise MingdaoAPIError(-1, f"请求错误: {e}")
    
    async def _connect_impl(self) -> bool:
        """实现连接逻辑"""
        try:
            timeout = ClientTimeout(
                connect=self._cfg.connect_timeout,
                total=self._cfg.read_timeout,
            )
            self._session = aiohttp.ClientSession(timeout=timeout)
            
            # 测试连接：获取应用列表
            await self.list_apps()
            
            self._logger.info("已连接到明道云")
            return True
        except Exception as e:
            self._logger.error(f"连接明道云失败: {e}")
            return False
    
    async def _disconnect_impl(self) -> None:
        """实现断开连接逻辑"""
        if self._session:
            await self._session.close()
            self._session = None
        self._logger.info("已断开与明道云的连接")
    
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
                table_id = message.get_context("table_id")
                data = message.get_context("data", {})
                
                if not table_id:
                    return SendResult(
                        success=False,
                        error="缺少table_id参数",
                        error_code="MISSING_PARAMS",
                    )
                
                result = await self.create_record(table_id, data)
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
            
            elif operation == "submit_form":
                form_id = message.get_context("form_id")
                data = message.get_context("data", {})
                
                if not form_id:
                    return SendResult(
                        success=False,
                        error="缺少form_id参数",
                        error_code="MISSING_PARAMS",
                    )
                
                result = await self.submit_form(form_id, data)
                return SendResult(
                    success=True,
                    message_id=result.get("record_id", ""),
                    timestamp=time.time(),
                )
            
            else:
                return SendResult(
                    success=False,
                    error=f"不支持的操作: {operation}",
                    error_code="UNSUPPORTED_OPERATION",
                )
                
        except MingdaoAPIError as e:
            return SendResult(
                success=False,
                error=e.message,
                error_code=str(e.code),
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
                table_id = payload.get("table_id") if payload else None
                if table_id:
                    records = await self.query_records(table_id)
                    for record in records:
                        content = MessageContent(text=f"[Record: {record.record_id}]")
                        metadata = MessageMetadata(
                            message_id=record.record_id,
                            channel_id="mingdao",
                            timestamp=time.time(),
                            direction=MessageDirection.INBOUND,
                            message_type=MessageType.DATA,
                        )
                        msg = UniversalMessage(content=content, metadata=metadata)
                        msg.set_context("record", record.to_dict())
                        messages.append(msg)
            
            return ReceiveResult(success=True, messages=messages)
        except Exception as e:
            return ReceiveResult(success=False, error=str(e))
    
    # ========== 应用管理 ==========
    
    async def list_apps(
        self,
        page: int = 1,
        page_size: int = 20,
    ) -> List[MingdaoApp]:
        """获取应用列表
        
        Args:
            page: 页码
            page_size: 每页数量
            
        Returns:
            应用列表
        """
        params = {
            "page": page,
            "pageSize": page_size,
        }
        
        result = await self._make_request("GET", "/app/list", params=params)
        
        apps = []
        for item in result.get("apps", []):
            app = MingdaoApp(
                app_id=item.get("appId", ""),
                app_name=item.get("appName", ""),
                app_type=AppType(item.get("appType", "custom")),
                description=item.get("description", ""),
                create_time=item.get("createTime"),
                update_time=item.get("updateTime"),
            )
            apps.append(app)
        
        return apps
    
    async def get_app(self, app_id: str) -> Optional[MingdaoApp]:
        """获取应用详情
        
        Args:
            app_id: 应用ID
            
        Returns:
            应用信息
        """
        params = {"appId": app_id}
        
        result = await self._make_request("GET", "/app/get", params=params)
        
        if not result:
            return None
        
        return MingdaoApp(
            app_id=result.get("appId", ""),
            app_name=result.get("appName", ""),
            app_type=AppType(result.get("appType", "custom")),
            description=result.get("description", ""),
            create_time=result.get("createTime"),
            update_time=result.get("updateTime"),
        )
    
    # ========== 数据表操作 ==========
    
    async def list_tables(
        self,
        app_id: Optional[str] = None,
    ) -> List[DataTable]:
        """获取数据表列表
        
        Args:
            app_id: 应用ID
            
        Returns:
            数据表列表
        """
        app_id = app_id or self._cfg.default_app_id
        
        params = {}
        if app_id:
            params["appId"] = app_id
        
        result = await self._make_request("GET", "/table/list", params=params)
        
        tables = []
        for item in result.get("tables", []):
            table = DataTable(
                table_id=item.get("tableId", ""),
                table_name=item.get("tableName", ""),
                app_id=item.get("appId", ""),
                fields=item.get("fields", []),
                description=item.get("description", ""),
            )
            tables.append(table)
        
        return tables
    
    async def get_table(self, table_id: str) -> Optional[DataTable]:
        """获取数据表详情
        
        Args:
            table_id: 表ID
            
        Returns:
            数据表信息
        """
        params = {"tableId": table_id}
        
        result = await self._make_request("GET", "/table/get", params=params)
        
        if not result:
            return None
        
        return DataTable(
            table_id=result.get("tableId", ""),
            table_name=result.get("tableName", ""),
            app_id=result.get("appId", ""),
            fields=result.get("fields", []),
            description=result.get("description", ""),
        )
    
    async def create_record(
        self,
        table_id: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """创建数据记录
        
        Args:
            table_id: 表ID
            data: 记录数据
            
        Returns:
            创建结果
        """
        params = {"tableId": table_id}
        
        result = await self._make_request(
            "POST",
            "/record/create",
            params=params,
            data={"data": data},
        )
        
        return {
            "success": True,
            "record_id": result.get("recordId", ""),
            "table_id": table_id,
        }
    
    async def get_record(
        self,
        table_id: str,
        record_id: str,
    ) -> Optional[TableRecord]:
        """获取数据记录
        
        Args:
            table_id: 表ID
            record_id: 记录ID
            
        Returns:
            记录信息
        """
        params = {
            "tableId": table_id,
            "recordId": record_id,
        }
        
        result = await self._make_request("GET", "/record/get", params=params)
        
        if not result:
            return None
        
        return TableRecord(
            record_id=result.get("recordId", ""),
            table_id=table_id,
            data=result.get("data", {}),
            create_time=result.get("createTime"),
            update_time=result.get("updateTime"),
            creator=result.get("creator"),
        )
    
    async def update_record(
        self,
        table_id: str,
        record_id: str,
        data: Dict[str, Any],
    ) -> bool:
        """更新数据记录
        
        Args:
            table_id: 表ID
            record_id: 记录ID
            data: 更新数据
            
        Returns:
            是否成功
        """
        params = {"tableId": table_id}
        
        await self._make_request(
            "POST",
            "/record/update",
            params=params,
            data={
                "recordId": record_id,
                "data": data,
            },
        )
        
        return True
    
    async def delete_record(
        self,
        table_id: str,
        record_id: str,
    ) -> bool:
        """删除数据记录
        
        Args:
            table_id: 表ID
            record_id: 记录ID
            
        Returns:
            是否成功
        """
        params = {"tableId": table_id}
        
        await self._make_request(
            "POST",
            "/record/delete",
            params=params,
            data={"recordId": record_id},
        )
        
        return True
    
    async def query_records(
        self,
        table_id: str,
        filters: Optional[List[Dict[str, Any]]] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> List[TableRecord]:
        """查询数据记录
        
        Args:
            table_id: 表ID
            filters: 过滤条件
            page: 页码
            page_size: 每页数量
            
        Returns:
            记录列表
        """
        params = {
            "tableId": table_id,
            "page": page,
            "pageSize": page_size,
        }
        
        data = {}
        if filters:
            data["filters"] = filters
        
        result = await self._make_request(
            "POST",
            "/record/query",
            params=params,
            data=data,
        )
        
        records = []
        for item in result.get("records", []):
            record = TableRecord(
                record_id=item.get("recordId", ""),
                table_id=table_id,
                data=item.get("data", {}),
                create_time=item.get("createTime"),
                update_time=item.get("updateTime"),
                creator=item.get("creator"),
            )
            records.append(record)
        
        return records
    
    # ========== 工作流操作 ==========
    
    async def list_workflows(
        self,
        app_id: Optional[str] = None,
    ) -> List[Workflow]:
        """获取工作流列表
        
        Args:
            app_id: 应用ID
            
        Returns:
            工作流列表
        """
        app_id = app_id or self._cfg.default_app_id
        
        params = {}
        if app_id:
            params["appId"] = app_id
        
        result = await self._make_request("GET", "/workflow/list", params=params)
        
        workflows = []
        for item in result.get("workflows", []):
            workflow = Workflow(
                workflow_id=item.get("workflowId", ""),
                workflow_name=item.get("workflowName", ""),
                app_id=item.get("appId", ""),
                trigger_type=item.get("triggerType", ""),
                status=item.get("status", "ENABLED"),
            )
            workflows.append(workflow)
        
        return workflows
    
    async def trigger_workflow(
        self,
        workflow_id: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """触发工作流
        
        Args:
            workflow_id: 工作流ID
            data: 触发数据
            
        Returns:
            触发结果
        """
        params = {"workflowId": workflow_id}
        
        result = await self._make_request(
            "POST",
            "/workflow/trigger",
            params=params,
            data={"data": data},
        )
        
        return {
            "success": True,
            "execution_id": result.get("executionId", ""),
            "workflow_id": workflow_id,
        }
    
    # ========== 表单操作 ==========
    
    async def submit_form(
        self,
        form_id: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """提交表单
        
        Args:
            form_id: 表单ID
            data: 表单数据
            
        Returns:
            提交结果
        """
        params = {"formId": form_id}
        
        result = await self._make_request(
            "POST",
            "/form/submit",
            params=params,
            data={"data": data},
        )
        
        return {
            "success": True,
            "record_id": result.get("recordId", ""),
            "form_id": form_id,
        }
    
    # ========== 工具方法 ==========
    
    def get_capabilities(self):
        """获取适配器能力"""
        return self._capabilities
    
    def supports_capability(self, capability) -> bool:
        """检查是否支持特定能力"""
        return capability in self._capabilities
    
    def __repr__(self) -> str:
        return f"MingdaoAdapter(app_key={self._cfg.app_key[:8]}...)"


# ========== CLI测试代码 ==========

async def test_mingdao():
    """测试明道云适配器"""
    import os
    
    # 从环境变量获取配置
    app_key = os.environ.get("MINGDAO_APP_KEY", "test-app-key")
    app_secret = os.environ.get("MINGDAO_APP_SECRET", "test-app-secret")
    
    config = MingdaoConfig(
        app_key=app_key,
        app_secret=app_secret,
    )
    
    adapter = MingdaoAdapter(config)
    
    try:
        # 连接
        print("正在连接明道云...")
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
    asyncio.run(test_mingdao())
