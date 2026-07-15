"""
轻流业务流程适配器模块

该模块提供轻流(Qingflow)业务流程平台的通道适配器实现，支持：
- 流程管理（创建、查询、更新）
- 任务操作（获取、处理、转交）
- 数据查询（应用数据、流程数据）
- 审批流程操作
- 应用管理

API文档: https://qingflow.com/api-doc
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


class ProcessStatus(Enum):
    """流程状态"""
    PENDING = "PENDING"         # 待处理
    PROCESSING = "PROCESSING"   # 处理中
    COMPLETED = "COMPLETED"     # 已完成
    REJECTED = "REJECTED"       # 已拒绝
    TERMINATED = "TERMINATED"   # 已终止


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "PENDING"         # 待处理
    APPROVED = "APPROVED"       # 已同意
    REJECTED = "REJECTED"       # 已拒绝
    TRANSFERRED = "TRANSFERRED" # 已转交


@dataclass
class QingflowConfig(ChannelConfig):
    """轻流配置类
    
    Attributes:
        api_key: API密钥
        api_secret: API密钥密文
        base_url: API基础URL
        connect_timeout: 连接超时时间（秒）
        read_timeout: 读取超时时间（秒）
        max_retry: 最大重试次数
        default_app_id: 默认应用ID
    """
    api_key: str = ""
    api_secret: str = ""
    base_url: str = "https://api.qingflow.com/v1"
    connect_timeout: int = 30
    read_timeout: int = 60
    max_retry: int = 3
    default_app_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.api_key or not self.api_secret:
            raise QingflowConfigError("API Key和API Secret不能为空")


class QingflowConfigError(Exception):
    """轻流配置错误"""
    pass


class QingflowAPIError(Exception):
    """轻流API错误
    
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
class QingflowApp:
    """轻流应用信息
    
    Attributes:
        app_id: 应用ID
        app_name: 应用名称
        description: 描述
        create_time: 创建时间
        update_time: 更新时间
    """
    app_id: str
    app_name: str
    description: str = ""
    create_time: Optional[str] = None
    update_time: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "app_id": self.app_id,
            "app_name": self.app_name,
            "description": self.description,
            "create_time": self.create_time,
            "update_time": self.update_time,
        }


@dataclass
class Process:
    """流程信息
    
    Attributes:
        process_id: 流程ID
        app_id: 应用ID
        process_name: 流程名称
        status: 流程状态
        creator: 创建者
        create_time: 创建时间
        update_time: 更新时间
    """
    process_id: str
    app_id: str
    process_name: str = ""
    status: ProcessStatus = ProcessStatus.PENDING
    creator: Optional[str] = None
    create_time: Optional[str] = None
    update_time: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "process_id": self.process_id,
            "app_id": self.app_id,
            "process_name": self.process_name,
            "status": self.status.value,
            "creator": self.creator,
            "create_time": self.create_time,
            "update_time": self.update_time,
        }


@dataclass
class Task:
    """任务信息
    
    Attributes:
        task_id: 任务ID
        process_id: 流程ID
        task_name: 任务名称
        status: 任务状态
        assignee: 处理人
        create_time: 创建时间
        update_time: 更新时间
    """
    task_id: str
    process_id: str
    task_name: str = ""
    status: TaskStatus = TaskStatus.PENDING
    assignee: Optional[str] = None
    create_time: Optional[str] = None
    update_time: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "process_id": self.process_id,
            "task_name": self.task_name,
            "status": self.status.value,
            "assignee": self.assignee,
            "create_time": self.create_time,
            "update_time": self.update_time,
        }


@dataclass
class AppData:
    """应用数据
    
    Attributes:
        data_id: 数据ID
        app_id: 应用ID
        data: 数据内容
        create_time: 创建时间
        update_time: 更新时间
    """
    data_id: str
    app_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    create_time: Optional[str] = None
    update_time: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "data_id": self.data_id,
            "app_id": self.app_id,
            "data": self.data,
            "create_time": self.create_time,
            "update_time": self.update_time,
        }


class QingflowAdapter(ChannelAdapter):
    """轻流适配器
    
    提供轻流业务流程平台的统一接口，支持流程管理、任务操作、
    数据查询、审批流程等功能。
    
    Example:
        config = QingflowConfig(
            api_key="your-api-key",
            api_secret="your-api-secret",
        )
        adapter = QingflowAdapter(config)
        await adapter.connect()
        
        # 列出应用
        apps = await adapter.list_apps()
        
        # 发起流程
        process = await adapter.create_process(
            app_id="app-xxx",
            data={"field1": "value1"},
        )
        
        # 获取任务列表
        tasks = await adapter.list_tasks()
    """
    
    def __init__(self, config: QingflowConfig) -> None:
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
        sign_string += f"&apiSecret={self._cfg.api_secret}"
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
            raise QingflowAPIError(-1, "适配器未连接")
        
        url = f"{self._get_base_url()}{endpoint}"
        
        # 添加公共参数
        if params is None:
            params = {}
        params["apiKey"] = self._cfg.api_key
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
            if result.get("code") != 0:
                raise QingflowAPIError(
                    result.get("code", -1),
                    result.get("msg", "Unknown error"),
                )
            
            return result.get("data", {})
            
        except aiohttp.ClientError as e:
            raise QingflowAPIError(-1, f"请求错误: {e}")
    
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
            
            self._logger.info("已连接到轻流")
            return True
        except Exception as e:
            self._logger.error(f"连接轻流失败: {e}")
            return False
    
    async def _disconnect_impl(self) -> None:
        """实现断开连接逻辑"""
        if self._session:
            await self._session.close()
            self._session = None
        self._logger.info("已断开与轻流的连接")
    
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
            operation = message.get_context("operation", "create_process")
            
            if operation == "create_process":
                app_id = message.get_context("app_id") or self._cfg.default_app_id
                data = message.get_context("data", {})
                
                if not app_id:
                    return SendResult(
                        success=False,
                        error="缺少app_id参数",
                        error_code="MISSING_PARAMS",
                    )
                
                result = await self.create_process(app_id, data)
                return SendResult(
                    success=True,
                    message_id=result.get("process_id", ""),
                    timestamp=time.time(),
                )
            
            elif operation == "handle_task":
                task_id = message.get_context("task_id")
                action = message.get_context("action", "approve")
                comment = message.get_context("comment", "")
                
                if not task_id:
                    return SendResult(
                        success=False,
                        error="缺少task_id参数",
                        error_code="MISSING_PARAMS",
                    )
                
                await self.handle_task(task_id, action, comment)
                return SendResult(success=True, timestamp=time.time())
            
            elif operation == "create_app_data":
                app_id = message.get_context("app_id") or self._cfg.default_app_id
                data = message.get_context("data", {})
                
                if not app_id:
                    return SendResult(
                        success=False,
                        error="缺少app_id参数",
                        error_code="MISSING_PARAMS",
                    )
                
                result = await self.create_app_data(app_id, data)
                return SendResult(
                    success=True,
                    message_id=result.get("data_id", ""),
                    timestamp=time.time(),
                )
            
            else:
                return SendResult(
                    success=False,
                    error=f"不支持的操作: {operation}",
                    error_code="UNSUPPORTED_OPERATION",
                )
                
        except QingflowAPIError as e:
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
            
            if operation == "query_processes":
                app_id = payload.get("app_id") if payload else None
                processes = await self.list_processes(app_id)
                for process in processes:
                    content = MessageContent(text=f"[Process: {process.process_id}]")
                    metadata = MessageMetadata(
                        message_id=process.process_id,
                        channel_id="qingflow",
                        timestamp=time.time(),
                        direction=MessageDirection.INBOUND,
                        message_type=MessageType.DATA,
                    )
                    msg = UniversalMessage(content=content, metadata=metadata)
                    msg.set_context("process", process.to_dict())
                    messages.append(msg)
            
            elif operation == "query_tasks":
                tasks = await self.list_tasks()
                for task in tasks:
                    content = MessageContent(text=f"[Task: {task.task_id}]")
                    metadata = MessageMetadata(
                        message_id=task.task_id,
                        channel_id="qingflow",
                        timestamp=time.time(),
                        direction=MessageDirection.INBOUND,
                        message_type=MessageType.DATA,
                    )
                    msg = UniversalMessage(content=content, metadata=metadata)
                    msg.set_context("task", task.to_dict())
                    messages.append(msg)
            
            elif operation == "query_app_data":
                app_id = payload.get("app_id") if payload else self._cfg.default_app_id
                if app_id:
                    data_list = await self.list_app_data(app_id)
                    for data in data_list:
                        content = MessageContent(text=f"[Data: {data.data_id}]")
                        metadata = MessageMetadata(
                            message_id=data.data_id,
                            channel_id="qingflow",
                            timestamp=time.time(),
                            direction=MessageDirection.INBOUND,
                            message_type=MessageType.DATA,
                        )
                        msg = UniversalMessage(content=content, metadata=metadata)
                        msg.set_context("data", data.to_dict())
                        messages.append(msg)
            
            return ReceiveResult(success=True, messages=messages)
        except Exception as e:
            return ReceiveResult(success=False, error=str(e))
    
    # ========== 应用管理 ==========
    
    async def list_apps(
        self,
        page: int = 1,
        page_size: int = 20,
    ) -> List[QingflowApp]:
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
            app = QingflowApp(
                app_id=item.get("appId", ""),
                app_name=item.get("appName", ""),
                description=item.get("description", ""),
                create_time=item.get("createTime"),
                update_time=item.get("updateTime"),
            )
            apps.append(app)
        
        return apps
    
    async def get_app(self, app_id: str) -> Optional[QingflowApp]:
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
        
        return QingflowApp(
            app_id=result.get("appId", ""),
            app_name=result.get("appName", ""),
            description=result.get("description", ""),
            create_time=result.get("createTime"),
            update_time=result.get("updateTime"),
        )
    
    # ========== 流程管理 ==========
    
    async def list_processes(
        self,
        app_id: Optional[str] = None,
        status: Optional[ProcessStatus] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> List[Process]:
        """获取流程列表
        
        Args:
            app_id: 应用ID
            status: 流程状态
            page: 页码
            page_size: 每页数量
            
        Returns:
            流程列表
        """
        params = {
            "page": page,
            "pageSize": page_size,
        }
        
        if app_id:
            params["appId"] = app_id
        if status:
            params["status"] = status.value
        
        result = await self._make_request("GET", "/process/list", params=params)
        
        processes = []
        for item in result.get("processes", []):
            process = Process(
                process_id=item.get("processId", ""),
                app_id=item.get("appId", ""),
                process_name=item.get("processName", ""),
                status=ProcessStatus(item.get("status", "PENDING")),
                creator=item.get("creator"),
                create_time=item.get("createTime"),
                update_time=item.get("updateTime"),
            )
            processes.append(process)
        
        return processes
    
    async def get_process(self, process_id: str) -> Optional[Process]:
        """获取流程详情
        
        Args:
            process_id: 流程ID
            
        Returns:
            流程信息
        """
        params = {"processId": process_id}
        
        result = await self._make_request("GET", "/process/get", params=params)
        
        if not result:
            return None
        
        return Process(
            process_id=result.get("processId", ""),
            app_id=result.get("appId", ""),
            process_name=result.get("processName", ""),
            status=ProcessStatus(result.get("status", "PENDING")),
            creator=result.get("creator"),
            create_time=result.get("createTime"),
            update_time=result.get("updateTime"),
        )
    
    async def create_process(
        self,
        app_id: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """发起流程
        
        Args:
            app_id: 应用ID
            data: 流程数据
            
        Returns:
            创建结果
        """
        params = {"appId": app_id}
        
        result = await self._make_request(
            "POST",
            "/process/create",
            params=params,
            data={"data": data},
        )
        
        return {
            "success": True,
            "process_id": result.get("processId", ""),
            "app_id": app_id,
        }
    
    async def terminate_process(self, process_id: str) -> bool:
        """终止流程
        
        Args:
            process_id: 流程ID
            
        Returns:
            是否成功
        """
        params = {"processId": process_id}
        
        await self._make_request("POST", "/process/terminate", params=params)
        return True
    
    # ========== 任务操作 ==========
    
    async def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> List[Task]:
        """获取任务列表
        
        Args:
            status: 任务状态
            page: 页码
            page_size: 每页数量
            
        Returns:
            任务列表
        """
        params = {
            "page": page,
            "pageSize": page_size,
        }
        
        if status:
            params["status"] = status.value
        
        result = await self._make_request("GET", "/task/list", params=params)
        
        tasks = []
        for item in result.get("tasks", []):
            task = Task(
                task_id=item.get("taskId", ""),
                process_id=item.get("processId", ""),
                task_name=item.get("taskName", ""),
                status=TaskStatus(item.get("status", "PENDING")),
                assignee=item.get("assignee"),
                create_time=item.get("createTime"),
                update_time=item.get("updateTime"),
            )
            tasks.append(task)
        
        return tasks
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务详情
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务信息
        """
        params = {"taskId": task_id}
        
        result = await self._make_request("GET", "/task/get", params=params)
        
        if not result:
            return None
        
        return Task(
            task_id=result.get("taskId", ""),
            process_id=result.get("processId", ""),
            task_name=result.get("taskName", ""),
            status=TaskStatus(result.get("status", "PENDING")),
            assignee=result.get("assignee"),
            create_time=result.get("createTime"),
            update_time=result.get("updateTime"),
        )
    
    async def handle_task(
        self,
        task_id: str,
        action: str,
        comment: str = "",
    ) -> bool:
        """处理任务
        
        Args:
            task_id: 任务ID
            action: 操作类型（approve/reject）
            comment: 审批意见
            
        Returns:
            是否成功
        """
        params = {"taskId": task_id}
        
        data = {
            "action": action,
            "comment": comment,
        }
        
        await self._make_request(
            "POST",
            "/task/handle",
            params=params,
            data=data,
        )
        
        return True
    
    async def transfer_task(
        self,
        task_id: str,
        transfer_to: str,
        comment: str = "",
    ) -> bool:
        """转交任务
        
        Args:
            task_id: 任务ID
            transfer_to: 转交目标用户ID
            comment: 转交说明
            
        Returns:
            是否成功
        """
        params = {"taskId": task_id}
        
        data = {
            "transferTo": transfer_to,
            "comment": comment,
        }
        
        await self._make_request(
            "POST",
            "/task/transfer",
            params=params,
            data=data,
        )
        
        return True
    
    # ========== 数据查询 ==========
    
    async def list_app_data(
        self,
        app_id: str,
        filters: Optional[List[Dict[str, Any]]] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> List[AppData]:
        """获取应用数据列表
        
        Args:
            app_id: 应用ID
            filters: 过滤条件
            page: 页码
            page_size: 每页数量
            
        Returns:
            数据列表
        """
        params = {
            "appId": app_id,
            "page": page,
            "pageSize": page_size,
        }
        
        data = {}
        if filters:
            data["filters"] = filters
        
        result = await self._make_request(
            "POST",
            "/app/data/list",
            params=params,
            data=data,
        )
        
        data_list = []
        for item in result.get("dataList", []):
            app_data = AppData(
                data_id=item.get("dataId", ""),
                app_id=app_id,
                data=item.get("data", {}),
                create_time=item.get("createTime"),
                update_time=item.get("updateTime"),
            )
            data_list.append(app_data)
        
        return data_list
    
    async def get_app_data(
        self,
        app_id: str,
        data_id: str,
    ) -> Optional[AppData]:
        """获取应用数据详情
        
        Args:
            app_id: 应用ID
            data_id: 数据ID
            
        Returns:
            数据信息
        """
        params = {
            "appId": app_id,
            "dataId": data_id,
        }
        
        result = await self._make_request("GET", "/app/data/get", params=params)
        
        if not result:
            return None
        
        return AppData(
            data_id=result.get("dataId", ""),
            app_id=app_id,
            data=result.get("data", {}),
            create_time=result.get("createTime"),
            update_time=result.get("updateTime"),
        )
    
    async def create_app_data(
        self,
        app_id: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """创建应用数据
        
        Args:
            app_id: 应用ID
            data: 数据内容
            
        Returns:
            创建结果
        """
        params = {"appId": app_id}
        
        result = await self._make_request(
            "POST",
            "/app/data/create",
            params=params,
            data={"data": data},
        )
        
        return {
            "success": True,
            "data_id": result.get("dataId", ""),
            "app_id": app_id,
        }
    
    async def update_app_data(
        self,
        app_id: str,
        data_id: str,
        data: Dict[str, Any],
    ) -> bool:
        """更新应用数据
        
        Args:
            app_id: 应用ID
            data_id: 数据ID
            data: 更新内容
            
        Returns:
            是否成功
        """
        params = {"appId": app_id}
        
        await self._make_request(
            "POST",
            "/app/data/update",
            params=params,
            data={
                "dataId": data_id,
                "data": data,
            },
        )
        
        return True
    
    async def delete_app_data(
        self,
        app_id: str,
        data_id: str,
    ) -> bool:
        """删除应用数据
        
        Args:
            app_id: 应用ID
            data_id: 数据ID
            
        Returns:
            是否成功
        """
        params = {"appId": app_id}
        
        await self._make_request(
            "POST",
            "/app/data/delete",
            params=params,
            data={"dataId": data_id},
        )
        
        return True
    
    # ========== 审批流程 ==========
    
    async def get_approval_flow(
        self, process_id: str) -> Dict[str, Any]:
        """获取审批流程
        
        Args:
            process_id: 流程ID
            
        Returns:
            审批流程信息
        """
        params = {"processId": process_id}
        
        result = await self._make_request(
            "GET",
            "/process/approval/flow",
            params=params,
        )
        
        return result
    
    async def get_approval_history(
        self, process_id: str) -> List[Dict[str, Any]]:
        """获取审批历史
        
        Args:
            process_id: 流程ID
            
        Returns:
            审批历史列表
        """
        params = {"processId": process_id}
        
        result = await self._make_request(
            "GET",
            "/process/approval/history",
            params=params,
        )
        
        return result.get("history", [])
    
    # ========== 工具方法 ==========
    
    def get_capabilities(self):
        """获取适配器能力"""
        return self._capabilities
    
    def supports_capability(self, capability) -> bool:
        """检查是否支持特定能力"""
        return capability in self._capabilities
    
    def __repr__(self) -> str:
        return f"QingflowAdapter(api_key={self._cfg.api_key[:8]}...)"


# ========== CLI测试代码 ==========

async def test_qingflow():
    """测试轻流适配器"""
    import os
    
    # 从环境变量获取配置
    api_key = os.environ.get("QINGFLOW_API_KEY", "test-api-key")
    api_secret = os.environ.get("QINGFLOW_API_SECRET", "test-api-secret")
    
    config = QingflowConfig(
        api_key=api_key,
        api_secret=api_secret,
    )
    
    adapter = QingflowAdapter(config)
    
    try:
        # 连接
        print("正在连接轻流...")
        connected = await adapter.connect()
        print(f"连接结果: {connected}")
        
        if not connected:
            print("连接失败，请检查配置")
            return
        
        # 健康检查
        healthy = await adapter.health_check()
        print(f"健康检查: {healthy}")
        
        # 获取应用列表
        print("\n获取应用列表:")
        apps = await adapter.list_apps(page_size=5)
        for app in apps:
            print(f"  - {app.app_name} ({app.app_id})")
        
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
    asyncio.run(test_qingflow())
