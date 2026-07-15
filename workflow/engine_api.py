"""
工作流REST API模块 (Workflow REST API)

提供工作流管理的REST API接口：
- 启动工作流
- 暂停工作流
- 恢复工作流
- 取消工作流
- 获取状态
- 列出工作流
- Webhook回调

类:
    WorkflowAPI: 工作流API主类
    WorkflowStarter: 工作流启动器
    WorkflowPausor: 工作流暂停器
    WorkflowCanceler: 工作流取消器
    WorkflowStatusAPI: 工作流状态API
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Union
from urllib.parse import urlparse
import threading

# 配置日志
logger = logging.getLogger(__name__)

# 类型变量
T = TypeVar('T')


class WorkflowStatus(Enum):
    """工作流状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class WorkflowPriority(Enum):
    """工作流优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class WebhookEvent(Enum):
    """Webhook事件类型"""
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"
    WORKFLOW_PAUSED = "workflow.paused"
    WORKFLOW_RESUMED = "workflow.resumed"
    WORKFLOW_CANCELLED = "workflow.cancelled"
    NODE_STARTED = "node.started"
    NODE_COMPLETED = "node.completed"
    NODE_FAILED = "node.failed"


@dataclass
class WorkflowRequest:
    """工作流请求"""
    workflow_id: Optional[str] = None
    workflow_name: str = ""
    workflow_type: str = "general"
    input_data: Optional[Any] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: WorkflowPriority = WorkflowPriority.NORMAL
    timeout: Optional[float] = None
    retry_on_failure: bool = True
    max_retries: int = 3
    callback_url: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_by: Optional[str] = None
    correlation_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    
    def __post_init__(self) -> None:
        """初始化后处理"""
        if not self.workflow_id:
            self.workflow_id = str(uuid.uuid4())
        if not self.created_by:
            self.created_by = "system"


@dataclass
class WorkflowResponse:
    """工作流响应"""
    workflow_id: str
    status: WorkflowStatus
    message: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output_data: Optional[Any] = None
    error: Optional[str] = None
    progress: float = 0.0
    current_node: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "workflow_id": self.workflow_id,
            "status": self.status.value,
            "message": self.message,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "output_data": self.output_data,
            "error": self.error,
            "progress": self.progress,
            "current_node": self.current_node,
            "metadata": self.metadata,
        }


@dataclass
class WorkflowInfo:
    """工作流信息"""
    workflow_id: str
    workflow_name: str
    workflow_type: str
    status: WorkflowStatus
    priority: WorkflowPriority
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: Optional[str] = None
    progress: float = 0.0
    current_node: Optional[str] = None
    total_nodes: int = 0
    completed_nodes: int = 0
    failed_nodes: int = 0
    retry_count: int = 0
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "workflow_type": self.workflow_type,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_by": self.created_by,
            "progress": self.progress,
            "current_node": self.current_node,
            "total_nodes": self.total_nodes,
            "completed_nodes": self.completed_nodes,
            "failed_nodes": self.failed_nodes,
            "retry_count": self.retry_count,
            "tags": list(self.tags),
            "metadata": self.metadata,
        }


@dataclass
class WebhookConfig:
    """Webhook配置"""
    url: str
    events: List[WebhookEvent] = field(default_factory=list)
    secret: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    retry_count: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    enabled: bool = True


@dataclass
class WebhookPayload:
    """Webhook载荷"""
    workflow_id: str
    event: WebhookEvent
    timestamp: datetime = field(default_factory=datetime.now)
    workflow_name: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    signature: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "event": self.event.value,
            "timestamp": self.timestamp.isoformat(),
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "data": self.data,
        }
    
    def sign(self, secret: str) -> str:
        """签名"""
        message = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(f"{secret}{message}".encode()).hexdigest()


class WorkflowStarter:
    """工作流启动器"""
    
    def __init__(self, api: WorkflowAPI):
        self.api = api
        self._starting_workflows: Set[str] = set()
        self._lock = threading.Lock()
    
    async def start(
        self,
        request: WorkflowRequest
    ) -> WorkflowResponse:
        """启动工作流"""
        workflow_id = request.workflow_id
        
        # 检查幂等性
        if request.idempotency_key:
            existing = self.api._get_by_idempotency_key(request.idempotency_key)
            if existing:
                logger.info(f"Returning existing workflow for idempotency key: {request.idempotency_key}")
                return WorkflowResponse(
                    workflow_id=existing.workflow_id,
                    status=existing.status,
                    message="Workflow already exists with this idempotency key",
                )
        
        # 检查是否正在启动
        with self._lock:
            if workflow_id in self._starting_workflows:
                return WorkflowResponse(
                    workflow_id=workflow_id,
                    status=WorkflowStatus.PENDING,
                    message="Workflow is already starting",
                )
            self._starting_workflows.add(workflow_id)
        
        try:
            # 创建工作流信息
            workflow_info = WorkflowInfo(
                workflow_id=workflow_id,
                workflow_name=request.workflow_name,
                workflow_type=request.workflow_type,
                status=WorkflowStatus.PENDING,
                priority=request.priority,
                created_at=datetime.now(),
                created_by=request.created_by,
                tags=request.tags,
                metadata=request.metadata,
            )
            
            self.api._workflows[workflow_id] = workflow_info
            self.api._idempotency_keys[request.idempotency_key] = workflow_id if request.idempotency_key else None
            
            # 异步启动
            asyncio.create_task(self._start_async(workflow_id, request))
            
            return WorkflowResponse(
                workflow_id=workflow_id,
                status=WorkflowStatus.PENDING,
                message="Workflow started successfully",
            )
        except Exception as e:
            logger.error(f"Failed to start workflow: {e}")
            with self._lock:
                self._starting_workflows.discard(workflow_id)
            raise
    
    async def _start_async(self, workflow_id: str, request: WorkflowRequest) -> None:
        """异步启动工作流"""
        try:
            workflow_info = self.api._workflows.get(workflow_id)
            if not workflow_info:
                return
            
            # 更新状态为运行中
            workflow_info.status = WorkflowStatus.RUNNING
            workflow_info.started_at = datetime.now()
            
            # 触发webhook
            await self.api._trigger_webhook(
                WebhookEvent.WORKFLOW_STARTED,
                workflow_id,
                {"input_data": request.input_data}
            )
            
            # 执行工作流（模拟）
            try:
                output_data = await self._execute_workflow(workflow_id, request)
                
                workflow_info.status = WorkflowStatus.COMPLETED
                workflow_info.completed_at = datetime.now()
                workflow_info.progress = 1.0
                
                await self.api._trigger_webhook(
                    WebhookEvent.WORKFLOW_COMPLETED,
                    workflow_id,
                    {"output_data": output_data}
                )
            except Exception as e:
                workflow_info.status = WorkflowStatus.FAILED
                workflow_info.completed_at = datetime.now()
                workflow_info.metadata["error"] = str(e)
                
                await self.api._trigger_webhook(
                    WebhookEvent.WORKFLOW_FAILED,
                    workflow_id,
                    {"error": str(e)}
                )
        finally:
            with self._lock:
                self._starting_workflows.discard(workflow_id)
    
    async def _execute_workflow(
        self,
        workflow_id: str,
        request: WorkflowRequest
    ) -> Any:
        """执行工作流"""
        logger.info(f"Executing workflow {workflow_id}")
        
        # 模拟工作流执行
        # 实际实现中，这里会调用具体的工作流引擎
        total_steps = request.parameters.get("total_steps", 5)
        
        for i in range(total_steps):
            # 检查是否被取消
            workflow_info = self.api._workflows.get(workflow_id)
            if workflow_info and workflow_info.status == WorkflowStatus.CANCELLED:
                raise Exception("Workflow was cancelled")
            
            # 模拟步骤执行
            await asyncio.sleep(0.1)
            
            # 更新进度
            workflow_info.progress = (i + 1) / total_steps
            workflow_info.current_node = f"step_{i + 1}"
            
            await self.api._trigger_webhook(
                WebhookEvent.NODE_COMPLETED,
                workflow_id,
                {
                    "node_id": f"step_{i + 1}",
                    "progress": workflow_info.progress,
                }
            )
        
        return {"result": "Workflow completed successfully"}


class WorkflowPausor:
    """工作流暂停器"""
    
    def __init__(self, api: WorkflowAPI):
        self.api = api
        self._paused_workflows: Dict[str, asyncio.Event] = {}
        self._lock = threading.Lock()
    
    async def pause(self, workflow_id: str, reason: str = "") -> WorkflowResponse:
        """暂停工作流"""
        workflow_info = self.api._workflows.get(workflow_id)
        
        if not workflow_info:
            return WorkflowResponse(
                workflow_id=workflow_id,
                status=WorkflowStatus.FAILED,
                message=f"Workflow {workflow_id} not found",
                error="Workflow not found",
            )
        
        if workflow_info.status != WorkflowStatus.RUNNING:
            return WorkflowResponse(
                workflow_id=workflow_id,
                status=workflow_info.status,
                message=f"Cannot pause workflow in {workflow_info.status.value} state",
            )
        
        # 创建暂停事件
        pause_event = asyncio.Event()
        with self._lock:
            self._paused_workflows[workflow_id] = pause_event
        
        # 更新状态
        workflow_info.status = WorkflowStatus.PAUSED
        workflow_info.metadata["pause_reason"] = reason
        workflow_info.metadata["paused_at"] = datetime.now().isoformat()
        
        # 触发webhook
        await self.api._trigger_webhook(
            WebhookEvent.WORKFLOW_PAUSED,
            workflow_id,
            {"reason": reason}
        )
        
        return WorkflowResponse(
            workflow_id=workflow_id,
            status=WorkflowStatus.PAUSED,
            message="Workflow paused successfully",
        )
    
    async def resume(self, workflow_id: str) -> WorkflowResponse:
        """恢复工作流"""
        workflow_info = self.api._workflows.get(workflow_id)
        
        if not workflow_info:
            return WorkflowResponse(
                workflow_id=workflow_id,
                status=WorkflowStatus.FAILED,
                message=f"Workflow {workflow_id} not found",
                error="Workflow not found",
            )
        
        if workflow_info.status != WorkflowStatus.PAUSED:
            return WorkflowResponse(
                workflow_id=workflow_id,
                status=workflow_info.status,
                message=f"Cannot resume workflow in {workflow_info.status.value} state",
            )
        
        # 获取暂停事件并触发
        pause_event = self._paused_workflows.get(workflow_id)
        if pause_event:
            pause_event.set()
        
        # 更新状态
        workflow_info.status = WorkflowStatus.RUNNING
        if "paused_at" in workflow_info.metadata:
            del workflow_info.metadata["paused_at"]
        
        # 触发webhook
        await self.api._trigger_webhook(
            WebhookEvent.WORKFLOW_RESUMED,
            workflow_id,
            {}
        )
        
        return WorkflowResponse(
            workflow_id=workflow_id,
            status=WorkflowStatus.RUNNING,
            message="Workflow resumed successfully",
        )
    
    def is_paused(self, workflow_id: str) -> bool:
        """检查工作流是否暂停"""
        return workflow_id in self._paused_workflows
    
    def wait_if_paused(self, workflow_id: str) -> asyncio.Future:
        """等待如果工作流暂停"""
        pause_event = self._paused_workflows.get(workflow_id)
        if pause_event:
            return pause_event.wait()
        return asyncio.Future()  # type: ignore


class WorkflowCanceler:
    """工作流取消器"""
    
    def __init__(self, api: WorkflowAPI):
        self.api = api
        self._cancelled_workflows: Set[str] = set()
    
    async def cancel(
        self,
        workflow_id: str,
        reason: str = "Cancelled by user"
    ) -> WorkflowResponse:
        """取消工作流"""
        workflow_info = self.api._workflows.get(workflow_id)
        
        if not workflow_info:
            return WorkflowResponse(
                workflow_id=workflow_id,
                status=WorkflowStatus.FAILED,
                message=f"Workflow {workflow_id} not found",
                error="Workflow not found",
            )
        
        if workflow_info.status in {WorkflowStatus.COMPLETED, WorkflowStatus.CANCELLED, WorkflowStatus.FAILED}:
            return WorkflowResponse(
                workflow_id=workflow_id,
                status=workflow_info.status,
                message=f"Cannot cancel workflow in {workflow_info.status.value} state",
            )
        
        # 更新状态
        workflow_info.status = WorkflowStatus.CANCELLED
        workflow_info.completed_at = datetime.now()
        workflow_info.metadata["cancel_reason"] = reason
        workflow_info.metadata["cancelled_at"] = datetime.now().isoformat()
        
        self._cancelled_workflows.add(workflow_id)
        
        # 触发webhook
        await self.api._trigger_webhook(
            WebhookEvent.WORKFLOW_CANCELLED,
            workflow_id,
            {"reason": reason}
        )
        
        return WorkflowResponse(
            workflow_id=workflow_id,
            status=WorkflowStatus.CANCELLED,
            message="Workflow cancelled successfully",
        )
    
    def is_cancelled(self, workflow_id: str) -> bool:
        """检查工作流是否已取消"""
        return workflow_id in self._cancelled_workflows


class WorkflowStatusAPI:
    """工作流状态API"""
    
    def __init__(self, api: WorkflowAPI):
        self.api = api
    
    async def get_status(self, workflow_id: str) -> WorkflowResponse:
        """获取工作流状态"""
        workflow_info = self.api._workflows.get(workflow_id)
        
        if not workflow_info:
            return WorkflowResponse(
                workflow_id=workflow_id,
                status=WorkflowStatus.FAILED,
                message=f"Workflow {workflow_id} not found",
                error="Workflow not found",
            )
        
        return WorkflowResponse(
            workflow_id=workflow_id,
            status=workflow_info.status,
            message=f"Workflow status: {workflow_info.status.value}",
            created_at=workflow_info.created_at,
            started_at=workflow_info.started_at,
            completed_at=workflow_info.completed_at,
            progress=workflow_info.progress,
            current_node=workflow_info.current_node,
            error=workflow_info.metadata.get("error"),
        )
    
    async def list_workflows(
        self,
        status: Optional[WorkflowStatus] = None,
        workflow_type: Optional[str] = None,
        created_by: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[WorkflowInfo]:
        """列出工作流"""
        results: List[WorkflowInfo] = []
        
        for workflow_info in self.api._workflows.values():
            # 状态过滤
            if status and workflow_info.status != status:
                continue
            
            # 类型过滤
            if workflow_type and workflow_info.workflow_type != workflow_type:
                continue
            
            # 创建者过滤
            if created_by and workflow_info.created_by != created_by:
                continue
            
            # 标签过滤
            if tags and not tags.issubset(workflow_info.tags):
                continue
            
            # 时间过滤
            if start_time and workflow_info.created_at < start_time:
                continue
            if end_time and workflow_info.created_at > end_time:
                continue
            
            results.append(workflow_info)
        
        # 分页
        results.sort(key=lambda w: w.created_at, reverse=True)
        return results[offset:offset + limit]
    
    async def get_workflow_details(self, workflow_id: str) -> Optional[WorkflowInfo]:
        """获取工作流详细信息"""
        return self.api._workflows.get(workflow_id)
    
    async def get_workflow_history(self, workflow_id: str) -> List[Dict[str, Any]]:
        """获取工作流历史"""
        # 从审计日志获取历史
        audit_search = self.api.audit_logger.search({
            "workflow_id": workflow_id
        })
        
        return [entry.to_dict() for entry in audit_search]
    
    async def get_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """获取统计信息"""
        workflows = await self.list_workflows(
            start_time=start_time,
            end_time=end_time,
            limit=10000,
        )
        
        status_counts: Dict[str, int] = {}
        type_counts: Dict[str, int] = {}
        total_duration = 0.0
        completed_count = 0
        
        for workflow in workflows:
            status_counts[workflow.status.value] = status_counts.get(workflow.status.value, 0) + 1
            type_counts[workflow.workflow_type] = type_counts.get(workflow.workflow_type, 0) + 1
            
            if workflow.started_at and workflow.completed_at:
                duration = (workflow.completed_at - workflow.started_at).total_seconds()
                total_duration += duration
                completed_count += 1
        
        return {
            "total_workflows": len(workflows),
            "status_distribution": status_counts,
            "type_distribution": type_counts,
            "average_duration_seconds": total_duration / completed_count if completed_count > 0 else 0,
            "time_range": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None,
            },
        }


class WorkflowAPI:
    """工作流REST API主类"""
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        enable_webhooks: bool = True,
        max_concurrent_workflows: int = 1000,
        default_timeout: float = 3600.0,
        audit_logger: Optional[Any] = None,
    ):
        self.storage_path = storage_path
        self.enable_webhooks = enable_webhooks
        self.max_concurrent_workflows = max_concurrent_workflows
        self.default_timeout = default_timeout
        self.audit_logger = audit_logger
        
        # 工作流存储
        self._workflows: Dict[str, WorkflowInfo] = {}
        self._webhooks: Dict[str, WebhookConfig] = {}
        self._idempotency_keys: Dict[str, Optional[str]] = {}
        
        # 子组件
        self.starter = WorkflowStarter(self)
        self.pausor = WorkflowPausor(self)
        self.canceler = WorkflowCanceler(self)
        self.status_api = WorkflowStatusAPI(self)
        
        # HTTP客户端（模拟）
        self._http_client: Optional[Any] = None
        
        # 锁
        self._lock = threading.Lock()
    
    def _get_by_idempotency_key(self, idempotency_key: str) -> Optional[WorkflowInfo]:
        """通过幂等键获取工作流"""
        workflow_id = self._idempotency_keys.get(idempotency_key)
        if workflow_id:
            return self._workflows.get(workflow_id)
        return None
    
    async def start_workflow(self, request: WorkflowRequest) -> WorkflowResponse:
        """启动工作流"""
        # 检查并发限制
        with self._lock:
            running_count = sum(
                1 for w in self._workflows.values()
                if w.status == WorkflowStatus.RUNNING
            )
            if running_count >= self.max_concurrent_workflows:
                return WorkflowResponse(
                    workflow_id=request.workflow_id or "",
                    status=WorkflowStatus.PENDING,
                    message="Too many concurrent workflows",
                    error="Concurrency limit reached",
                )
        
        return await self.starter.start(request)
    
    async def pause_workflow(
        self,
        workflow_id: str,
        reason: str = ""
    ) -> WorkflowResponse:
        """暂停工作流"""
        return await self.pausor.pause(workflow_id, reason)
    
    async def resume_workflow(self, workflow_id: str) -> WorkflowResponse:
        """恢复工作流"""
        return await self.pausor.resume(workflow_id)
    
    async def cancel_workflow(
        self,
        workflow_id: str,
        reason: str = "Cancelled by user"
    ) -> WorkflowResponse:
        """取消工作流"""
        return await self.canceler.cancel(workflow_id, reason)
    
    async def get_workflow_status(self, workflow_id: str) -> WorkflowResponse:
        """获取工作流状态"""
        return await self.status_api.get_status(workflow_id)
    
    async def list_workflows(
        self,
        status: Optional[str] = None,
        workflow_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """列出工作流"""
        status_enum = WorkflowStatus(status) if status else None
        
        workflows = await self.status_api.list_workflows(
            status=status_enum,
            workflow_type=workflow_type,
            limit=limit,
            offset=offset,
        )
        
        return [w.to_dict() for w in workflows]
    
    async def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """获取工作流详情"""
        workflow = await self.status_api.get_workflow_details(workflow_id)
        return workflow.to_dict() if workflow else None
    
    def register_webhook(
        self,
        webhook_id: str,
        config: WebhookConfig
    ) -> bool:
        """注册Webhook"""
        self._webhooks[webhook_id] = config
        return True
    
    def unregister_webhook(self, webhook_id: str) -> bool:
        """取消注册Webhook"""
        if webhook_id in self._webhooks:
            del self._webhooks[webhook_id]
            return True
        return False
    
    def list_webhooks(self) -> List[Dict[str, Any]]:
        """列出Webhook"""
        return [
            {
                "webhook_id": webhook_id,
                "url": config.url,
                "events": [e.value for e in config.events],
                "enabled": config.enabled,
            }
            for webhook_id, config in self._webhooks.items()
        ]
    
    async def _trigger_webhook(
        self,
        event: WebhookEvent,
        workflow_id: str,
        data: Dict[str, Any]
    ) -> None:
        """触发Webhook"""
        if not self.enable_webhooks:
            return
        
        # 获取工作流信息
        workflow_info = self._workflows.get(workflow_id)
        
        # 构建载荷
        payload = WebhookPayload(
            event=event,
            workflow_id=workflow_id,
            workflow_name=workflow_info.workflow_name if workflow_info else None,
            data=data,
        )
        
        # 发送webhook（并发）
        tasks = []
        for webhook_id, config in self._webhooks.items():
            if not config.enabled:
                continue
            if event not in config.events:
                continue
            
            tasks.append(self._send_webhook(webhook_id, config, payload))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_webhook(
        self,
        webhook_id: str,
        config: WebhookConfig,
        payload: WebhookPayload
    ) -> bool:
        """发送Webhook"""
        try:
            # 签名
            if config.secret:
                payload.signature = payload.sign(config.secret)
            
            # 准备请求头
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "WorkflowAPI/1.0",
                **config.headers,
            }
            
            if payload.signature:
                headers["X-Webhook-Signature"] = payload.signature
            
            # 模拟HTTP请求
            # 实际实现中，这里会使用httpx或aiohttp发送请求
            logger.info(f"Sending webhook to {config.url}: {payload.event.value}")
            
            # 模拟重试逻辑
            for attempt in range(config.retry_count):
                try:
                    # 实际HTTP请求（模拟）
                    # response = await self._http_client.post(
                    #     config.url,
                    #     json=payload.to_dict(),
                    #     headers=headers,
                    #     timeout=config.timeout,
                    # )
                    # response.raise_for_status()
                    
                    return True
                except Exception as e:
                    logger.warning(f"Webhook attempt {attempt + 1} failed: {e}")
                    if attempt < config.retry_count - 1:
                        await asyncio.sleep(config.retry_delay * (attempt + 1))
            
            return False
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")
            return False
    
    async def handle_webhook_callback(
        self,
        webhook_id: str,
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理Webhook回调"""
        # 验证请求
        if webhook_id not in self._webhooks:
            return {"error": "Webhook not found"}
        
        webhook = self._webhooks[webhook_id]
        
        # 处理回调数据
        event_type = request_data.get("event")
        workflow_id = request_data.get("workflow_id")
        
        if not workflow_id:
            return {"error": "Missing workflow_id"}
        
        # 更新工作流状态
        workflow_info = self._workflows.get(workflow_id)
        if workflow_info:
            workflow_info.metadata["webhook_callback"] = request_data
        
        return {
            "status": "received",
            "webhook_id": webhook_id,
            "event": event_type,
            "workflow_id": workflow_id,
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "workflows": {
                    "total": len(self._workflows),
                    "running": sum(1 for w in self._workflows.values() if w.status == WorkflowStatus.RUNNING),
                    "completed": sum(1 for w in self._workflows.values() if w.status == WorkflowStatus.COMPLETED),
                    "failed": sum(1 for w in self._workflows.values() if w.status == WorkflowStatus.FAILED),
                },
                "webhooks": {
                    "total": len(self._webhooks),
                    "enabled": sum(1 for w in self._webhooks.values() if w.enabled),
                },
            },
        }
    
    def get_api_spec(self) -> Dict[str, Any]:
        """获取API规范"""
        return {
            "openapi": "3.0.0",
            "info": {
                "title": "Workflow API",
                "version": "1.0.0",
                "description": "REST API for workflow management",
            },
            "paths": {
                "/workflows": {
                    "post": {
                        "summary": "Start a workflow",
                        "operationId": "startWorkflow",
                    },
                    "get": {
                        "summary": "List workflows",
                        "operationId": "listWorkflows",
                    },
                },
                "/workflows/{workflow_id}": {
                    "get": {
                        "summary": "Get workflow status",
                        "operationId": "getWorkflowStatus",
                    },
                    "delete": {
                        "summary": "Cancel workflow",
                        "operationId": "cancelWorkflow",
                    },
                },
                "/workflows/{workflow_id}/pause": {
                    "post": {
                        "summary": "Pause workflow",
                        "operationId": "pauseWorkflow",
                    },
                },
                "/workflows/{workflow_id}/resume": {
                    "post": {
                        "summary": "Resume workflow",
                        "operationId": "resumeWorkflow",
                    },
                },
                "/webhooks": {
                    "get": {
                        "summary": "List webhooks",
                        "operationId": "listWebhooks",
                    },
                    "post": {
                        "summary": "Register webhook",
                        "operationId": "registerWebhook",
                    },
                },
                "/webhooks/{webhook_id}": {
                    "delete": {
                        "summary": "Unregister webhook",
                        "operationId": "unregisterWebhook",
                    },
                },
                "/webhooks/callback": {
                    "post": {
                        "summary": "Handle webhook callback",
                        "operationId": "handleWebhookCallback",
                    },
                },
                "/health": {
                    "get": {
                        "summary": "Health check",
                        "operationId": "healthCheck",
                    },
                },
            },
        }


# 全局API实例
_default_api: Optional[WorkflowAPI] = None


def get_workflow_api(**kwargs: Any) -> WorkflowAPI:
    """获取全局工作流API"""
    global _default_api
    
    if _default_api is None:
        _default_api = WorkflowAPI(**kwargs)
    
    return _default_api


__all__ = [
    # 枚举类型
    "WorkflowStatus",
    "WorkflowPriority",
    "WebhookEvent",
    # 数据类
    "WorkflowRequest",
    "WorkflowResponse",
    "WorkflowInfo",
    "WebhookConfig",
    "WebhookPayload",
    # 核心类
    "WorkflowAPI",
    "WorkflowStarter",
    "WorkflowPausor",
    "WorkflowCanceler",
    "WorkflowStatusAPI",
    # 辅助函数
    "get_workflow_api",
]
