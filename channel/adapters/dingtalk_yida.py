"""
钉钉宜搭低代码平台适配器模块

该模块提供钉钉宜搭(Yida)低代码平台的通道适配器实现，支持：
- 表单管理（创建、查询、更新）
- 流程实例管理（发起、查询、审批）
- 数据查询（表单数据、实例数据）
- 审批工作流操作
- 宜搭应用集成

API文档: https://open.dingtalk.com/document/isvapp-server/yida-api-overview
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


class FormType(Enum):
    """表单类型"""
    NORMAL = "normal"           # 普通表单
    PROCESS = "process"         # 流程表单
    REPORT = "report"           # 报表


class InstanceStatus(Enum):
    """流程实例状态"""
    RUNNING = "RUNNING"         # 进行中
    COMPLETED = "COMPLETED"     # 已完成
    TERMINATED = "TERMINATED"   # 已终止


class ApprovalAction(Enum):
    """审批动作"""
    AGREE = "agree"             # 同意
    REFUSE = "refuse"           # 拒绝
    TRANSFER = "transfer"       # 转交
    ADD_SIGN = "addSign"        # 加签


@dataclass
class YidaConfig(ChannelConfig):
    """钉钉宜搭配置类
    
    Attributes:
        app_key: 钉钉应用AppKey
        app_secret: 钉钉应用AppSecret
        corp_id: 企业CorpId
        agent_id: 应用AgentId
        base_url: API基础URL
        connect_timeout: 连接超时时间（秒）
        read_timeout: 读取超时时间（秒）
        max_retry: 最大重试次数
        default_app_key: 默认宜搭应用标识
    """
    app_key: str = ""
    app_secret: str = ""
    corp_id: str = ""
    agent_id: str = ""
    base_url: str = "https://oapi.dingtalk.com"
    connect_timeout: int = 30
    read_timeout: int = 60
    max_retry: int = 3
    default_app_key: Optional[str] = None
    
    def __post_init__(self):
        if not self.app_key or not self.app_secret:
            raise YidaConfigError("AppKey和AppSecret不能为空")


class YidaConfigError(Exception):
    """宜搭配置错误"""
    pass


class YidaAPIError(Exception):
    """宜搭API错误
    
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
        retryable_codes = ["-1", "88", "40003", "40005"]
        return self.code in retryable_codes


@dataclass
class FormSchema:
    """表单结构定义
    
    Attributes:
        form_uuid: 表单唯一标识
        form_name: 表单名称
        form_type: 表单类型
        description: 描述
        fields: 字段定义列表
        create_time: 创建时间
        update_time: 更新时间
    """
    form_uuid: str
    form_name: str
    form_type: FormType = FormType.NORMAL
    description: str = ""
    fields: List[Dict[str, Any]] = field(default_factory=list)
    create_time: Optional[str] = None
    update_time: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "form_uuid": self.form_uuid,
            "form_name": self.form_name,
            "form_type": self.form_type.value,
            "description": self.description,
            "fields": self.fields,
            "create_time": self.create_time,
            "update_time": self.update_time,
        }


@dataclass
class FormInstance:
    """表单实例
    
    Attributes:
        instance_id: 实例ID
        form_uuid: 表单UUID
        data: 表单数据
        create_time: 创建时间
        modify_time: 修改时间
        creator: 创建者
    """
    instance_id: str
    form_uuid: str
    data: Dict[str, Any] = field(default_factory=dict)
    create_time: Optional[str] = None
    modify_time: Optional[str] = None
    creator: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "instance_id": self.instance_id,
            "form_uuid": self.form_uuid,
            "data": self.data,
            "create_time": self.create_time,
            "modify_time": self.modify_time,
            "creator": self.creator,
        }


@dataclass
class ProcessInstance:
    """流程实例
    
    Attributes:
        process_instance_id: 流程实例ID
        form_uuid: 表单UUID
        title: 流程标题
        status: 流程状态
        originator: 发起人
        create_time: 创建时间
        finish_time: 完成时间
        tasks: 任务列表
    """
    process_instance_id: str
    form_uuid: str
    title: str = ""
    status: InstanceStatus = InstanceStatus.RUNNING
    originator: Optional[str] = None
    create_time: Optional[str] = None
    finish_time: Optional[str] = None
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "process_instance_id": self.process_instance_id,
            "form_uuid": self.form_uuid,
            "title": self.title,
            "status": self.status.value,
            "originator": self.originator,
            "create_time": self.create_time,
            "finish_time": self.finish_time,
            "tasks": self.tasks,
        }


class DingTalkYidaAdapter(ChannelAdapter):
    """钉钉宜搭适配器
    
    提供钉钉宜搭低代码平台的统一接口，支持表单管理、流程实例管理、
    数据查询、审批工作流等功能。
    
    Example:
        config = YidaConfig(
            app_key="your-app-key",
            app_secret="your-app-secret",
            corp_id="your-corp-id",
            agent_id="your-agent-id",
        )
        adapter = DingTalkYidaAdapter(config)
        await adapter.connect()
        
        # 查询表单列表
        forms = await adapter.list_forms()
        
        # 创建表单实例
        instance = await adapter.create_form_instance(
            form_uuid="form-xxx",
            data={"field1": "value1"},
        )
        
        # 发起流程
        process = await adapter.start_process(
            form_uuid="form-xxx",
            originator="user-xxx",
            data={"field1": "value1"},
        )
    """
    
    API_VERSION = "v1"
    
    def __init__(self, config: YidaConfig) -> None:
        super().__init__(config)
        self._cfg = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0
    
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
    
    async def _get_access_token(self, force: bool = False) -> str:
        """获取访问令牌
        
        Args:
            force: 是否强制刷新
            
        Returns:
            访问令牌
        """
        now = time.time()
        if not force and self._access_token and now < self._token_expires_at - 60:
            return self._access_token
        
        url = f"{self._get_base_url()}/gettoken"
        params = {
            "appkey": self._cfg.app_key,
            "appsecret": self._cfg.app_secret,
        }
        
        async with self._session.get(url, params=params) as response:
            result = await response.json()
        
        if result.get("errcode") != 0:
            raise YidaAPIError(
                str(result.get("errcode")),
                result.get("errmsg", "获取token失败"),
            )
        
        self._access_token = result.get("access_token")
        self._token_expires_at = now + result.get("expires_in", 7200)
        
        return self._access_token
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """发送API请求
        
        Args:
            method: HTTP方法
            endpoint: API端点
            data: 请求数据
            params: URL参数
            
        Returns:
            API响应
        """
        if not self._session:
            raise YidaAPIError("NotConnected", "适配器未连接")
        
        # 获取access_token
        access_token = await self._get_access_token()
        
        # 构建URL
        url = f"{self._get_base_url()}{endpoint}"
        
        # 添加token到参数
        if params is None:
            params = {}
        params["access_token"] = access_token
        
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
            if result.get("errcode") != 0 and result.get("code") != "0":
                raise YidaAPIError(
                    str(result.get("errcode", result.get("code", "Unknown"))),
                    result.get("errmsg", result.get("message", "Unknown error")),
                    result.get("request_id"),
                )
            
            return result
            
        except aiohttp.ClientError as e:
            raise YidaAPIError("RequestError", str(e))
    
    async def _connect_impl(self) -> bool:
        """实现连接逻辑"""
        try:
            timeout = ClientTimeout(
                connect=self._cfg.connect_timeout,
                total=self._cfg.read_timeout,
            )
            self._session = aiohttp.ClientSession(timeout=timeout)
            
            # 测试连接：获取access_token
            await self._get_access_token()
            
            self._logger.info("已连接到钉钉宜搭")
            return True
        except Exception as e:
            self._logger.error(f"连接钉钉宜搭失败: {e}")
            return False
    
    async def _disconnect_impl(self) -> None:
        """实现断开连接逻辑"""
        if self._session:
            await self._session.close()
            self._session = None
        self._access_token = None
        self._token_expires_at = 0.0
        self._logger.info("已断开与钉钉宜搭的连接")
    
    async def _health_check_impl(self) -> bool:
        """实现健康检查逻辑"""
        try:
            await self._get_access_token()
            return True
        except Exception:
            return False
    
    async def _send_impl(self, message: UniversalMessage, priority: MessagePriority) -> SendResult:
        """实现发送逻辑"""
        try:
            operation = message.get_context("operation", "create_instance")
            
            if operation == "create_instance":
                form_uuid = message.get_context("form_uuid")
                data = message.get_context("data", {})
                
                if not form_uuid:
                    return SendResult(
                        success=False,
                        error="缺少form_uuid参数",
                        error_code="MISSING_PARAMS",
                    )
                
                result = await self.create_form_instance(form_uuid, data)
                return SendResult(
                    success=True,
                    message_id=result.get("instance_id", ""),
                    timestamp=time.time(),
                )
            
            elif operation == "start_process":
                form_uuid = message.get_context("form_uuid")
                originator = message.get_context("originator")
                data = message.get_context("data", {})
                
                if not form_uuid or not originator:
                    return SendResult(
                        success=False,
                        error="缺少form_uuid或originator参数",
                        error_code="MISSING_PARAMS",
                    )
                
                result = await self.start_process(form_uuid, originator, data)
                return SendResult(
                    success=True,
                    message_id=result.get("process_instance_id", ""),
                    timestamp=time.time(),
                )
            
            else:
                return SendResult(
                    success=False,
                    error=f"不支持的操作: {operation}",
                    error_code="UNSUPPORTED_OPERATION",
                )
                
        except YidaAPIError as e:
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
            
            if operation == "query_instances":
                form_uuid = payload.get("form_uuid") if payload else None
                if form_uuid:
                    instances = await self.query_form_instances(form_uuid)
                    for inst in instances:
                        content = MessageContent(text=f"[Form Instance: {inst.instance_id}]")
                        metadata = MessageMetadata(
                            message_id=inst.instance_id,
                            channel_id="dingtalk_yida",
                            timestamp=time.time(),
                            direction=MessageDirection.INBOUND,
                            message_type=MessageType.DATA,
                        )
                        msg = UniversalMessage(content=content, metadata=metadata)
                        msg.set_context("instance", inst.to_dict())
                        messages.append(msg)
            
            elif operation == "query_processes":
                form_uuid = payload.get("form_uuid") if payload else None
                if form_uuid:
                    processes = await self.query_process_instances(form_uuid)
                    for proc in processes:
                        content = MessageContent(text=f"[Process: {proc.process_instance_id}]")
                        metadata = MessageMetadata(
                            message_id=proc.process_instance_id,
                            channel_id="dingtalk_yida",
                            timestamp=time.time(),
                            direction=MessageDirection.INBOUND,
                            message_type=MessageType.DATA,
                        )
                        msg = UniversalMessage(content=content, metadata=metadata)
                        msg.set_context("process", proc.to_dict())
                        messages.append(msg)
            
            return ReceiveResult(success=True, messages=messages)
        except Exception as e:
            return ReceiveResult(success=False, error=str(e))
    
    # ========== 表单管理 ==========
    
    async def list_forms(
        self,
        app_key: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> List[FormSchema]:
        """获取表单列表
        
        Args:
            app_key: 宜搭应用标识
            page: 页码
            page_size: 每页数量
            
        Returns:
            表单列表
        """
        app_key = app_key or self._cfg.default_app_key
        
        endpoint = "/yida/v1/forms"
        params = {
            "appKey": app_key,
            "pageNumber": page,
            "pageSize": page_size,
        }
        
        result = await self._make_request("GET", endpoint, params=params)
        
        forms = []
        for item in result.get("data", {}).get("list", []):
            form = FormSchema(
                form_uuid=item.get("formUuid", ""),
                form_name=item.get("formName", ""),
                form_type=FormType(item.get("formType", "normal")),
                description=item.get("description", ""),
                create_time=item.get("createTime"),
                update_time=item.get("updateTime"),
            )
            forms.append(form)
        
        return forms
    
    async def get_form_schema(
        self,
        form_uuid: str,
        app_key: Optional[str] = None,
    ) -> Optional[FormSchema]:
        """获取表单结构
        
        Args:
            form_uuid: 表单UUID
            app_key: 宜搭应用标识
            
        Returns:
            表单结构
        """
        app_key = app_key or self._cfg.default_app_key
        
        endpoint = f"/yida/v1/forms/{form_uuid}"
        params = {"appKey": app_key}
        
        result = await self._make_request("GET", endpoint, params=params)
        data = result.get("data", {})
        
        if not data:
            return None
        
        return FormSchema(
            form_uuid=data.get("formUuid", ""),
            form_name=data.get("formName", ""),
            form_type=FormType(data.get("formType", "normal")),
            description=data.get("description", ""),
            fields=data.get("fields", []),
            create_time=data.get("createTime"),
            update_time=data.get("updateTime"),
        )
    
    # ========== 表单实例操作 ==========
    
    async def create_form_instance(
        self,
        form_uuid: str,
        data: Dict[str, Any],
        app_key: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """创建表单实例
        
        Args:
            form_uuid: 表单UUID
            data: 表单数据
            app_key: 宜搭应用标识
            user_id: 用户ID
            
        Returns:
            创建结果
        """
        app_key = app_key or self._cfg.default_app_key
        
        endpoint = "/yida/v1/instances"
        payload = {
            "appKey": app_key,
            "formUuid": form_uuid,
            "formDataJson": json.dumps(data),
        }
        
        if user_id:
            payload["userId"] = user_id
        
        result = await self._make_request("POST", endpoint, data=payload)
        
        return {
            "success": True,
            "instance_id": result.get("data", {}).get("instanceId", ""),
            "form_uuid": form_uuid,
        }
    
    async def update_form_instance(
        self,
        instance_id: str,
        data: Dict[str, Any],
        app_key: Optional[str] = None,
    ) -> bool:
        """更新表单实例
        
        Args:
            instance_id: 实例ID
            data: 更新的数据
            app_key: 宜搭应用标识
            
        Returns:
            是否成功
        """
        app_key = app_key or self._cfg.default_app_key
        
        endpoint = f"/yida/v1/instances/{instance_id}"
        payload = {
            "appKey": app_key,
            "formDataJson": json.dumps(data),
        }
        
        await self._make_request("PUT", endpoint, data=payload)
        return True
    
    async def delete_form_instance(
        self,
        instance_id: str,
        app_key: Optional[str] = None,
    ) -> bool:
        """删除表单实例
        
        Args:
            instance_id: 实例ID
            app_key: 宜搭应用标识
            
        Returns:
            是否成功
        """
        app_key = app_key or self._cfg.default_app_key
        
        endpoint = f"/yida/v1/instances/{instance_id}"
        params = {"appKey": app_key}
        
        await self._make_request("DELETE", endpoint, params=params)
        return True
    
    async def get_form_instance(
        self,
        instance_id: str,
        app_key: Optional[str] = None,
    ) -> Optional[FormInstance]:
        """获取表单实例
        
        Args:
            instance_id: 实例ID
            app_key: 宜搭应用标识
            
        Returns:
            表单实例
        """
        app_key = app_key or self._cfg.default_app_key
        
        endpoint = f"/yida/v1/instances/{instance_id}"
        params = {"appKey": app_key}
        
        result = await self._make_request("GET", endpoint, params=params)
        data = result.get("data", {})
        
        if not data:
            return None
        
        return FormInstance(
            instance_id=data.get("instanceId", ""),
            form_uuid=data.get("formUuid", ""),
            data=json.loads(data.get("formDataJson", "{}")),
            create_time=data.get("createTime"),
            modify_time=data.get("modifyTime"),
            creator=data.get("creator"),
        )
    
    async def query_form_instances(
        self,
        form_uuid: str,
        app_key: Optional[str] = None,
        conditions: Optional[List[Dict[str, Any]]] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> List[FormInstance]:
        """查询表单实例
        
        Args:
            form_uuid: 表单UUID
            app_key: 宜搭应用标识
            conditions: 查询条件
            page: 页码
            page_size: 每页数量
            
        Returns:
            表单实例列表
        """
        app_key = app_key or self._cfg.default_app_key
        
        endpoint = "/yida/v1/instances/search"
        payload = {
            "appKey": app_key,
            "formUuid": form_uuid,
            "pageNumber": page,
            "pageSize": page_size,
        }
        
        if conditions:
            payload["searchCondition"] = json.dumps(conditions)
        
        result = await self._make_request("POST", endpoint, data=payload)
        
        instances = []
        for item in result.get("data", {}).get("list", []):
            inst = FormInstance(
                instance_id=item.get("instanceId", ""),
                form_uuid=item.get("formUuid", ""),
                data=json.loads(item.get("formDataJson", "{}")),
                create_time=item.get("createTime"),
                modify_time=item.get("modifyTime"),
                creator=item.get("creator"),
            )
            instances.append(inst)
        
        return instances
    
    # ========== 流程实例操作 ==========
    
    async def start_process(
        self,
        form_uuid: str,
        originator: str,
        data: Dict[str, Any],
        app_key: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """发起流程
        
        Args:
            form_uuid: 表单UUID
            originator: 发起人用户ID
            data: 表单数据
            app_key: 宜搭应用标识
            title: 流程标题
            
        Returns:
            流程实例信息
        """
        app_key = app_key or self._cfg.default_app_key
        
        endpoint = "/yida/v1/processes/instances"
        payload = {
            "appKey": app_key,
            "formUuid": form_uuid,
            "originator": originator,
            "formDataJson": json.dumps(data),
        }
        
        if title:
            payload["title"] = title
        
        result = await self._make_request("POST", endpoint, data=payload)
        
        return {
            "success": True,
            "process_instance_id": result.get("data", {}).get("processInstanceId", ""),
            "form_uuid": form_uuid,
        }
    
    async def get_process_instance(
        self,
        process_instance_id: str,
        app_key: Optional[str] = None,
    ) -> Optional[ProcessInstance]:
        """获取流程实例
        
        Args:
            process_instance_id: 流程实例ID
            app_key: 宜搭应用标识
            
        Returns:
            流程实例
        """
        app_key = app_key or self._cfg.default_app_key
        
        endpoint = f"/yida/v1/processes/instances/{process_instance_id}"
        params = {"appKey": app_key}
        
        result = await self._make_request("GET", endpoint, params=params)
        data = result.get("data", {})
        
        if not data:
            return None
        
        return ProcessInstance(
            process_instance_id=data.get("processInstanceId", ""),
            form_uuid=data.get("formUuid", ""),
            title=data.get("title", ""),
            status=InstanceStatus(data.get("status", "RUNNING")),
            originator=data.get("originator"),
            create_time=data.get("createTime"),
            finish_time=data.get("finishTime"),
            tasks=data.get("tasks", []),
        )
    
    async def query_process_instances(
        self,
        form_uuid: str,
        app_key: Optional[str] = None,
        status: Optional[InstanceStatus] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> List[ProcessInstance]:
        """查询流程实例
        
        Args:
            form_uuid: 表单UUID
            app_key: 宜搭应用标识
            status: 流程状态
            page: 页码
            page_size: 每页数量
            
        Returns:
            流程实例列表
        """
        app_key = app_key or self._cfg.default_app_key
        
        endpoint = "/yida/v1/processes/instances/search"
        payload = {
            "appKey": app_key,
            "formUuid": form_uuid,
            "pageNumber": page,
            "pageSize": page_size,
        }
        
        if status:
            payload["status"] = status.value
        
        result = await self._make_request("POST", endpoint, data=payload)
        
        processes = []
        for item in result.get("data", {}).get("list", []):
            proc = ProcessInstance(
                process_instance_id=item.get("processInstanceId", ""),
                form_uuid=item.get("formUuid", ""),
                title=item.get("title", ""),
                status=InstanceStatus(item.get("status", "RUNNING")),
                originator=item.get("originator"),
                create_time=item.get("createTime"),
                finish_time=item.get("finishTime"),
                tasks=item.get("tasks", []),
            )
            processes.append(proc)
        
        return processes
    
    # ========== 审批操作 ==========
    
    async def approve_task(
        self,
        task_id: str,
        action: ApprovalAction,
        comment: Optional[str] = None,
        app_key: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """审批任务
        
        Args:
            task_id: 任务ID
            action: 审批动作
            comment: 审批意见
            app_key: 宜搭应用标识
            user_id: 用户ID
            
        Returns:
            是否成功
        """
        app_key = app_key or self._cfg.default_app_key
        
        endpoint = f"/yida/v1/tasks/{task_id}/approve"
        payload = {
            "appKey": app_key,
            "action": action.value,
        }
        
        if comment:
            payload["comment"] = comment
        if user_id:
            payload["userId"] = user_id
        
        await self._make_request("POST", endpoint, data=payload)
        return True
    
    async def transfer_task(
        self,
        task_id: str,
        transfer_to: str,
        comment: Optional[str] = None,
        app_key: Optional[str] = None,
    ) -> bool:
        """转交任务
        
        Args:
            task_id: 任务ID
            transfer_to: 转交目标用户ID
            comment: 转交说明
            app_key: 宜搭应用标识
            
        Returns:
            是否成功
        """
        app_key = app_key or self._cfg.default_app_key
        
        endpoint = f"/yida/v1/tasks/{task_id}/transfer"
        payload = {
            "appKey": app_key,
            "transferTo": transfer_to,
        }
        
        if comment:
            payload["comment"] = comment
        
        await self._make_request("POST", endpoint, data=payload)
        return True
    
    async def get_user_tasks(
        self,
        user_id: str,
        app_key: Optional[str] = None,
        status: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> List[Dict[str, Any]]:
        """获取用户任务列表
        
        Args:
            user_id: 用户ID
            app_key: 宜搭应用标识
            status: 任务状态
            page: 页码
            page_size: 每页数量
            
        Returns:
            任务列表
        """
        app_key = app_key or self._cfg.default_app_key
        
        endpoint = "/yida/v1/tasks"
        params = {
            "appKey": app_key,
            "userId": user_id,
            "pageNumber": page,
            "pageSize": page_size,
        }
        
        if status:
            params["status"] = status
        
        result = await self._make_request("GET", endpoint, params=params)
        return result.get("data", {}).get("list", [])
    
    # ========== 工具方法 ==========
    
    def get_capabilities(self):
        """获取适配器能力"""
        return self._capabilities
    
    def supports_capability(self, capability) -> bool:
        """检查是否支持特定能力"""
        return capability in self._capabilities
    
    def __repr__(self) -> str:
        return f"DingTalkYidaAdapter(corp_id={self._cfg.corp_id})"


# ========== CLI测试代码 ==========

async def test_dingtalk_yida():
    """测试钉钉宜搭适配器"""
    import os
    
    # 从环境变量获取配置
    app_key = os.environ.get("DINGTALK_APP_KEY", "test-app-key")
    app_secret = os.environ.get("DINGTALK_APP_SECRET", "test-app-secret")
    corp_id = os.environ.get("DINGTALK_CORP_ID", "test-corp-id")
    agent_id = os.environ.get("DINGTALK_AGENT_ID", "test-agent-id")
    
    config = YidaConfig(
        app_key=app_key,
        app_secret=app_secret,
        corp_id=corp_id,
        agent_id=agent_id,
    )
    
    adapter = DingTalkYidaAdapter(config)
    
    try:
        # 连接
        print("正在连接钉钉宜搭...")
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
    asyncio.run(test_dingtalk_yida())
