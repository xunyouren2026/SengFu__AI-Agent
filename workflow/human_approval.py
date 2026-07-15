"""
人工审批模块

提供人工介入工作流的功能：
- 审批请求生成
- 通知分发
- 超时处理
- 升级策略
- 审计日志

Classes:
    HumanApproval: 人工审批主类
    ApprovalRequest: 审批请求
    NotificationDispatcher: 通知分发器
    TimeoutManager: 超时管理器
    EscalationPolicy: 升级策略
    ApprovalAudit: 审批审计
"""

import json
import smtplib
import time
import uuid
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError


class ApprovalStatus(Enum):
    """审批状态枚举"""
    PENDING = "pending"         # 等待审批
    APPROVED = "approved"       # 已批准
    REJECTED = "rejected"       # 已拒绝
    TIMEOUT = "timeout"         # 超时
    ESCALATED = "escalated"     # 已升级
    CANCELLED = "cancelled"     # 已取消


class NotificationChannel(Enum):
    """通知渠道枚举"""
    EMAIL = "email"             # 邮件
    WEBHOOK = "webhook"         # Webhook
    CONSOLE = "console"         # 控制台
    CALLBACK = "callback"       # 回调函数


class EscalationLevel(Enum):
    """升级级别枚举"""
    LEVEL_1 = 1                 # 第一级
    LEVEL_2 = 2                 # 第二级
    LEVEL_3 = 3                 # 第三级


class ApprovalError(Exception):
    """审批异常"""
    pass


class ApprovalTimeoutError(ApprovalError):
    """审批超时异常"""
    pass


class EscalationError(ApprovalError):
    """升级异常"""
    pass


class NotificationError(ApprovalError):
    """通知异常"""
    pass


@dataclass
class ApprovalRequest:
    """
    审批请求

    Attributes:
        id: 请求ID
        workflow_id: 工作流ID
        node_id: 节点ID
        title: 标题
        description: 描述
        requester: 请求者
        approvers: 审批人列表
        status: 当前状态
        created_at: 创建时间
        expires_at: 过期时间
        metadata: 元数据
        context: 审批上下文
    """
    id: str
    workflow_id: str
    node_id: str
    title: str
    description: str = ""
    requester: str = ""
    approvers: List[str] = dataclass_field(default_factory=list)
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: float = dataclass_field(default_factory=time.time)
    expires_at: Optional[float] = None
    metadata: Dict[str, Any] = dataclass_field(default_factory=dict)
    context: Dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "node_id": self.node_id,
            "title": self.title,
            "description": self.description,
            "requester": self.requester,
            "approvers": list(self.approvers),
            "status": self.status.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "metadata": dict(self.metadata),
            "context": dict(self.context),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApprovalRequest":
        """从字典创建"""
        data = dict(data)
        data["status"] = ApprovalStatus(data.get("status", "pending"))
        return cls(**data)

    def is_expired(self) -> bool:
        """检查是否已过期"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def get_elapsed_time(self) -> float:
        """获取已等待时间（秒）"""
        return time.time() - self.created_at

    def get_remaining_time(self) -> Optional[float]:
        """获取剩余时间（秒）"""
        if self.expires_at is None:
            return None
        remaining = self.expires_at - time.time()
        return max(0, remaining)


@dataclass
class ApprovalResponse:
    """
    审批响应

    Attributes:
        request_id: 请求ID
        approved: 是否批准
        approver: 审批人
        comment: 审批意见
        responded_at: 响应时间
        metadata: 元数据
    """
    request_id: str
    approved: bool
    approver: str = ""
    comment: str = ""
    responded_at: float = dataclass_field(default_factory=time.time)
    metadata: Dict[str, Any] = dataclass_field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "request_id": self.request_id,
            "approved": self.approved,
            "approver": self.approver,
            "comment": self.comment,
            "responded_at": self.responded_at,
            "metadata": dict(self.metadata),
        }


@dataclass
class EscalationRule:
    """
    升级规则

    Attributes:
        level: 升级级别
        timeout_seconds: 触发超时（秒）
        new_approvers: 新审批人列表
        notification_channels: 通知渠道
        action: 升级动作
    """
    level: EscalationLevel
    timeout_seconds: int
    new_approvers: List[str] = dataclass_field(default_factory=list)
    notification_channels: List[NotificationChannel] = dataclass_field(
        default_factory=lambda: [NotificationChannel.EMAIL]
    )
    action: Optional[Callable[[ApprovalRequest], None]] = None


class NotificationDispatcher:
    """
    通知分发器

    支持多种通知渠道：邮件、Webhook、控制台、回调函数。

    Usage:
        dispatcher = NotificationDispatcher()
        dispatcher.configure_email(smtp_host="smtp.example.com", ...)
        dispatcher.send_notification(request, [NotificationChannel.EMAIL])
    """

    def __init__(self):
        self._email_config: Optional[Dict[str, Any]] = None
        self._webhook_url: Optional[str] = None
        self._webhook_headers: Dict[str, str] = {}
        self._callback: Optional[Callable[[ApprovalRequest], None]] = None
        self._notification_handlers: Dict[NotificationChannel, Callable] = {}
        self._setup_default_handlers()

    def _setup_default_handlers(self) -> None:
        """设置默认通知处理器"""
        self._notification_handlers = {
            NotificationChannel.EMAIL: self._send_email,
            NotificationChannel.WEBHOOK: self._send_webhook,
            NotificationChannel.CONSOLE: self._send_console,
            NotificationChannel.CALLBACK: self._send_callback,
        }

    def configure_email(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        use_tls: bool = True,
        from_address: Optional[str] = None,
    ) -> "NotificationDispatcher":
        """
        配置邮件通知

        Args:
            smtp_host: SMTP服务器地址
            smtp_port: SMTP端口
            username: 用户名
            password: 密码
            use_tls: 是否使用TLS
            from_address: 发件人地址

        Returns:
            self
        """
        self._email_config = {
            "host": smtp_host,
            "port": smtp_port,
            "username": username,
            "password": password,
            "use_tls": use_tls,
            "from_address": from_address or username,
        }
        return self

    def configure_webhook(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> "NotificationDispatcher":
        """
        配置Webhook通知

        Args:
            url: Webhook URL
            headers: 请求头

        Returns:
            self
        """
        self._webhook_url = url
        self._webhook_headers = headers or {"Content-Type": "application/json"}
        return self

    def set_callback(self, callback: Callable[[ApprovalRequest], None]) -> "NotificationDispatcher":
        """
        设置回调函数

        Args:
            callback: 回调函数

        Returns:
            self
        """
        self._callback = callback
        return self

    def send_notification(
        self,
        request: ApprovalRequest,
        channels: Optional[List[NotificationChannel]] = None,
    ) -> Dict[NotificationChannel, bool]:
        """
        发送通知

        Args:
            request: 审批请求
            channels: 通知渠道列表

        Returns:
            各渠道发送结果
        """
        channels = channels or [NotificationChannel.CONSOLE]
        results = {}

        for channel in channels:
            handler = self._notification_handlers.get(channel)
            if handler:
                try:
                    handler(request)
                    results[channel] = True
                except Exception as e:
                    results[channel] = False
            else:
                results[channel] = False

        return results

    def _send_email(self, request: ApprovalRequest) -> None:
        """发送邮件通知"""
        if not self._email_config:
            raise NotificationError("邮件未配置")

        config = self._email_config
        msg = MIMEText(self._format_email_body(request), "plain", "utf-8")
        msg["Subject"] = f"[审批请求] {request.title}"
        msg["From"] = config["from_address"]
        msg["To"] = ", ".join(request.approvers)

        try:
            server = smtplib.SMTP(config["host"], config["port"])
            if config["use_tls"]:
                server.starttls()
            server.login(config["username"], config["password"])
            server.sendmail(config["from_address"], request.approvers, msg.as_string())
            server.quit()
        except Exception as e:
            raise NotificationError(f"邮件发送失败: {e}")

    def _format_email_body(self, request: ApprovalRequest) -> str:
        """格式化邮件正文"""
        return f"""
审批请求

标题: {request.title}
描述: {request.description}
请求ID: {request.id}
工作流ID: {request.workflow_id}
节点ID: {request.node_id}
请求者: {request.requester}
创建时间: {datetime.fromtimestamp(request.created_at)}

请尽快处理此审批请求。
"""

    def _send_webhook(self, request: ApprovalRequest) -> None:
        """发送Webhook通知"""
        if not self._webhook_url:
            raise NotificationError("Webhook未配置")

        payload = json.dumps(request.to_dict()).encode("utf-8")
        req = Request(
            self._webhook_url,
            data=payload,
            headers=self._webhook_headers,
            method="POST",
        )

        try:
            with urlopen(req, timeout=30) as response:
                if response.status not in (200, 201, 202):
                    raise NotificationError(f"Webhook返回错误状态: {response.status}")
        except URLError as e:
            raise NotificationError(f"Webhook请求失败: {e}")

    def _send_console(self, request: ApprovalRequest) -> None:
        """发送控制台通知"""
        print(f"[审批请求] {request.title} (ID: {request.id})")
        print(f"  描述: {request.description}")
        print(f"  请求者: {request.requester}")
        print(f"  审批人: {', '.join(request.approvers)}")

    def _send_callback(self, request: ApprovalRequest) -> None:
        """执行回调通知"""
        if self._callback:
            self._callback(request)


class TimeoutManager:
    """
    超时管理器

    管理审批请求的超时检测和处理。

    Usage:
        manager = TimeoutManager()
        manager.register_timeout(request_id, timeout_seconds=3600)
        expired = manager.get_expired_requests()
    """

    def __init__(self):
        self._timeouts: Dict[str, float] = {}
        self._lock = threading.Lock()

    def register_timeout(self, request_id: str, timeout_seconds: float) -> None:
        """
        注册超时

        Args:
            request_id: 请求ID
            timeout_seconds: 超时秒数
        """
        with self._lock:
            self._timeouts[request_id] = time.time() + timeout_seconds

    def cancel_timeout(self, request_id: str) -> bool:
        """
        取消超时

        Args:
            request_id: 请求ID

        Returns:
            是否成功取消
        """
        with self._lock:
            if request_id in self._timeouts:
                del self._timeouts[request_id]
                return True
            return False

    def is_expired(self, request_id: str) -> bool:
        """
        检查是否已过期

        Args:
            request_id: 请求ID

        Returns:
            是否已过期
        """
        with self._lock:
            if request_id not in self._timeouts:
                return False
            return time.time() > self._timeouts[request_id]

    def get_expired_requests(self) -> List[str]:
        """
        获取所有已过期的请求ID

        Returns:
            过期请求ID列表
        """
        with self._lock:
            current_time = time.time()
            return [
                rid for rid, expiry in self._timeouts.items()
                if current_time > expiry
            ]

    def get_remaining_time(self, request_id: str) -> Optional[float]:
        """
        获取剩余时间

        Args:
            request_id: 请求ID

        Returns:
            剩余秒数，None表示未注册
        """
        with self._lock:
            if request_id not in self._timeouts:
                return None
            remaining = self._timeouts[request_id] - time.time()
            return max(0, remaining)

    def clear(self) -> None:
        """清空所有超时"""
        with self._lock:
            self._timeouts.clear()


class EscalationPolicy:
    """
    升级策略

    定义和管理审批请求的升级规则。

    Usage:
        policy = EscalationPolicy()
        policy.add_rule(EscalationRule(
            level=EscalationLevel.LEVEL_1,
            timeout_seconds=3600,
            new_approvers=["manager@example.com"]
        ))
        escalated = policy.check_escalation(request)
    """

    def __init__(self):
        self._rules: List[EscalationRule] = []
        self._escalation_history: Dict[str, List[EscalationLevel]] = {}

    def add_rule(self, rule: EscalationRule) -> "EscalationPolicy":
        """
        添加升级规则

        Args:
            rule: 升级规则

        Returns:
            self
        """
        self._rules.append(rule)
        # 按级别排序
        self._rules.sort(key=lambda r: r.level.value)
        return self

    def remove_rule(self, level: EscalationLevel) -> bool:
        """
        移除升级规则

        Args:
            level: 升级级别

        Returns:
            是否成功移除
        """
        for i, rule in enumerate(self._rules):
            if rule.level == level:
                self._rules.pop(i)
                return True
        return False

    def check_escalation(self, request: ApprovalRequest) -> Optional[EscalationRule]:
        """
        检查是否需要升级

        Args:
            request: 审批请求

        Returns:
            需要应用的升级规则，None表示不需要升级
        """
        elapsed = request.get_elapsed_time()
        history = self._escalation_history.get(request.id, [])

        for rule in self._rules:
            # 检查是否已达到此级别
            if rule.level in history:
                continue
            # 检查是否达到超时
            if elapsed >= rule.timeout_seconds:
                return rule

        return None

    def apply_escalation(
        self,
        request: ApprovalRequest,
        rule: EscalationRule,
    ) -> ApprovalRequest:
        """
        应用升级

        Args:
            request: 审批请求
            rule: 升级规则

        Returns:
            更新后的请求
        """
        # 更新审批人
        if rule.new_approvers:
            request.approvers = rule.new_approvers

        # 更新状态
        request.status = ApprovalStatus.ESCALATED
        request.metadata["escalation_level"] = rule.level.value
        request.metadata["escalated_at"] = time.time()

        # 记录历史
        if request.id not in self._escalation_history:
            self._escalation_history[request.id] = []
        self._escalation_history[request.id].append(rule.level)

        # 执行升级动作
        if rule.action:
            rule.action(request)

        return request

    def get_escalation_history(self, request_id: str) -> List[EscalationLevel]:
        """
        获取升级历史

        Args:
            request_id: 请求ID

        Returns:
            升级级别历史
        """
        return list(self._escalation_history.get(request_id, []))

    def clear_history(self, request_id: Optional[str] = None) -> None:
        """
        清空升级历史

        Args:
            request_id: 请求ID，None表示清空所有
        """
        if request_id:
            self._escalation_history.pop(request_id, None)
        else:
            self._escalation_history.clear()


class ApprovalAudit:
    """
    审批审计

    记录和查询审批历史。

    Usage:
        audit = ApprovalAudit()
        audit.log_request(request)
        audit.log_response(response)
        history = audit.get_request_history(request_id)
    """

    def __init__(self):
        self._request_logs: Dict[str, List[Dict[str, Any]]] = {}
        self._response_logs: Dict[str, List[ApprovalResponse]] = {}
        self._audit_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []

    def add_audit_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> "ApprovalAudit":
        """
        添加审计回调

        Args:
            callback: 回调函数

        Returns:
            self
        """
        self._audit_callbacks.append(callback)
        return self

    def log_request(self, request: ApprovalRequest) -> None:
        """
        记录审批请求

        Args:
            request: 审批请求
        """
        log_entry = {
            "timestamp": time.time(),
            "type": "request",
            "data": request.to_dict(),
        }

        if request.id not in self._request_logs:
            self._request_logs[request.id] = []
        self._request_logs[request.id].append(log_entry)

        # 触发回调
        for callback in self._audit_callbacks:
            try:
                callback("request", log_entry)
            except Exception:
                pass

    def log_response(self, response: ApprovalResponse) -> None:
        """
        记录审批响应

        Args:
            response: 审批响应
        """
        if response.request_id not in self._response_logs:
            self._response_logs[response.request_id] = []
        self._response_logs[response.request_id].append(response)

        log_entry = {
            "timestamp": time.time(),
            "type": "response",
            "data": response.to_dict(),
        }

        # 触发回调
        for callback in self._audit_callbacks:
            try:
                callback("response", log_entry)
            except Exception:
                pass

    def log_event(self, request_id: str, event_type: str, data: Dict[str, Any]) -> None:
        """
        记录自定义事件

        Args:
            request_id: 请求ID
            event_type: 事件类型
            data: 事件数据
        """
        log_entry = {
            "timestamp": time.time(),
            "type": event_type,
            "data": data,
        }

        if request_id not in self._request_logs:
            self._request_logs[request_id] = []
        self._request_logs[request_id].append(log_entry)

        # 触发回调
        for callback in self._audit_callbacks:
            try:
                callback(event_type, log_entry)
            except Exception:
                pass

    def get_request_history(self, request_id: str) -> List[Dict[str, Any]]:
        """
        获取请求历史

        Args:
            request_id: 请求ID

        Returns:
            历史记录列表
        """
        return list(self._request_logs.get(request_id, []))

    def get_response_history(self, request_id: str) -> List[ApprovalResponse]:
        """
        获取响应历史

        Args:
            request_id: 请求ID

        Returns:
            响应列表
        """
        return list(self._response_logs.get(request_id, []))

    def get_audit_report(self, request_id: str) -> Dict[str, Any]:
        """
        获取审计报告

        Args:
            request_id: 请求ID

        Returns:
            审计报告
        """
        request_history = self.get_request_history(request_id)
        response_history = self.get_response_history(request_id)

        # 计算统计信息
        request_count = sum(1 for h in request_history if h["type"] == "request")
        response_count = len(response_history)

        approved_count = sum(1 for r in response_history if r.approved)
        rejected_count = response_count - approved_count

        return {
            "request_id": request_id,
            "request_count": request_count,
            "response_count": response_count,
            "approved_count": approved_count,
            "rejected_count": rejected_count,
            "history": request_history,
            "responses": [r.to_dict() for r in response_history],
        }

    def export_to_file(self, filepath: str, request_id: Optional[str] = None) -> None:
        """
        导出审计日志到文件

        Args:
            filepath: 文件路径
            request_id: 请求ID，None表示导出所有
        """
        if request_id:
            data = self.get_audit_report(request_id)
        else:
            data = {
                "requests": {
                    rid: self.get_audit_report(rid)
                    for rid in self._request_logs
                }
            }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def clear(self, request_id: Optional[str] = None) -> None:
        """
        清空审计日志

        Args:
            request_id: 请求ID，None表示清空所有
        """
        if request_id:
            self._request_logs.pop(request_id, None)
            self._response_logs.pop(request_id, None)
        else:
            self._request_logs.clear()
            self._response_logs.clear()


class HumanApproval:
    """
    人工审批主类

    整合审批请求、通知、超时、升级和审计功能。

    Usage:
        approval = HumanApproval()
        request = approval.create_request(
            workflow_id="wf1",
            node_id="node1",
            title="请审批",
            approvers=["user@example.com"]
        )
        response = approval.wait_for_response(request.id, timeout=3600)
    """

    def __init__(self):
        self._requests: Dict[str, ApprovalRequest] = {}
        self._responses: Dict[str, ApprovalResponse] = {}
        self._pending_callbacks: Dict[str, List[Callable]] = {}
        self._dispatcher = NotificationDispatcher()
        self._timeout_manager = TimeoutManager()
        self._escalation_policy = EscalationPolicy()
        self._audit = ApprovalAudit()
        self._lock = threading.Lock()

    def create_request(
        self,
        workflow_id: str,
        node_id: str,
        title: str,
        description: str = "",
        requester: str = "",
        approvers: Optional[List[str]] = None,
        timeout_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ApprovalRequest:
        """
        创建审批请求

        Args:
            workflow_id: 工作流ID
            node_id: 节点ID
            title: 标题
            description: 描述
            requester: 请求者
            approvers: 审批人列表
            timeout_seconds: 超时秒数
            metadata: 元数据
            context: 上下文

        Returns:
            审批请求
        """
        request = ApprovalRequest(
            id=str(uuid.uuid4())[:8],
            workflow_id=workflow_id,
            node_id=node_id,
            title=title,
            description=description,
            requester=requester,
            approvers=approvers or [],
            metadata=metadata or {},
            context=context or {},
        )

        if timeout_seconds:
            request.expires_at = time.time() + timeout_seconds
            self._timeout_manager.register_timeout(request.id, timeout_seconds)

        with self._lock:
            self._requests[request.id] = request

        # 记录审计日志
        self._audit.log_request(request)

        return request

    def send_notification(
        self,
        request_id: str,
        channels: Optional[List[NotificationChannel]] = None,
    ) -> Dict[NotificationChannel, bool]:
        """
        发送通知

        Args:
            request_id: 请求ID
            channels: 通知渠道

        Returns:
            发送结果
        """
        request = self._requests.get(request_id)
        if not request:
            raise ApprovalError(f"请求不存在: {request_id}")

        return self._dispatcher.send_notification(request, channels)

    def respond(self, request_id: str, approved: bool, approver: str = "", comment: str = "") -> ApprovalResponse:
        """
        响应审批请求

        Args:
            request_id: 请求ID
            approved: 是否批准
            approver: 审批人
            comment: 审批意见

        Returns:
            审批响应
        """
        with self._lock:
            request = self._requests.get(request_id)
            if not request:
                raise ApprovalError(f"请求不存在: {request_id}")

            if request.status != ApprovalStatus.PENDING:
                raise ApprovalError(f"请求状态不是等待中: {request.status}")

            response = ApprovalResponse(
                request_id=request_id,
                approved=approved,
                approver=approver,
                comment=comment,
            )

            self._responses[request_id] = response
            request.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED

            # 取消超时
            self._timeout_manager.cancel_timeout(request_id)

        # 记录审计日志
        self._audit.log_response(response)

        # 触发回调
        callbacks = self._pending_callbacks.pop(request_id, [])
        for callback in callbacks:
            try:
                callback(response)
            except Exception:
                pass

        return response

    def wait_for_response(
        self,
        request_id: str,
        timeout: Optional[float] = None,
        poll_interval: float = 0.1,
    ) -> Optional[ApprovalResponse]:
        """
        等待响应

        Args:
            request_id: 请求ID
            timeout: 超时时间
            poll_interval: 轮询间隔

        Returns:
            审批响应，超时返回None
        """
        start_time = time.time()

        while True:
            # 检查是否有响应
            response = self._responses.get(request_id)
            if response:
                return response

            # 检查超时
            if timeout and (time.time() - start_time) > timeout:
                return None

            # 检查请求是否过期
            request = self._requests.get(request_id)
            if request and request.is_expired():
                request.status = ApprovalStatus.TIMEOUT
                return None

            time.sleep(poll_interval)

    def on_response(self, request_id: str, callback: Callable[[ApprovalResponse], None]) -> None:
        """
        注册响应回调

        Args:
            request_id: 请求ID
            callback: 回调函数
        """
        with self._lock:
            if request_id not in self._pending_callbacks:
                self._pending_callbacks[request_id] = []
            self._pending_callbacks[request_id].append(callback)

    def check_escalation(self, request_id: str) -> Optional[ApprovalRequest]:
        """
        检查并执行升级

        Args:
            request_id: 请求ID

        Returns:
            升级后的请求，None表示不需要升级
        """
        request = self._requests.get(request_id)
        if not request:
            return None

        rule = self._escalation_policy.check_escalation(request)
        if rule:
            self._escalation_policy.apply_escalation(request, rule)
            # 重新发送通知
            self.send_notification(request_id, rule.notification_channels)
            # 记录审计日志
            self._audit.log_event(request_id, "escalation", {
                "level": rule.level.value,
                "new_approvers": rule.new_approvers,
            })
            return request

        return None

    def cancel_request(self, request_id: str) -> bool:
        """
        取消请求

        Args:
            request_id: 请求ID

        Returns:
            是否成功取消
        """
        with self._lock:
            request = self._requests.get(request_id)
            if not request:
                return False

            if request.status != ApprovalStatus.PENDING:
                return False

            request.status = ApprovalStatus.CANCELLED
            self._timeout_manager.cancel_timeout(request_id)

        # 记录审计日志
        self._audit.log_event(request_id, "cancelled", {})

        return True

    def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """获取请求"""
        return self._requests.get(request_id)

    def get_response(self, request_id: str) -> Optional[ApprovalResponse]:
        """获取响应"""
        return self._responses.get(request_id)

    def get_pending_requests(self) -> List[ApprovalRequest]:
        """获取所有待处理请求"""
        return [
            r for r in self._requests.values()
            if r.status == ApprovalStatus.PENDING
        ]

    def get_dispatcher(self) -> NotificationDispatcher:
        """获取通知分发器"""
        return self._dispatcher

    def get_timeout_manager(self) -> TimeoutManager:
        """获取超时管理器"""
        return self._timeout_manager

    def get_escalation_policy(self) -> EscalationPolicy:
        """获取升级策略"""
        return self._escalation_policy

    def get_audit(self) -> ApprovalAudit:
        """获取审计器"""
        return self._audit

    def clear(self) -> None:
        """清空所有数据"""
        with self._lock:
            self._requests.clear()
            self._responses.clear()
            self._pending_callbacks.clear()
        self._timeout_manager.clear()
        self._escalation_policy.clear_history()
        self._audit.clear()


# 导入threading用于TimeoutManager
import threading
