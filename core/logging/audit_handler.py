"""
审计日志处理器模块

提供审计日志记录、存储和查询功能。
支持人类交互事件、工具调用、Agent动作、配置变更等审计场景。
采用内存+文件双写策略，确保审计数据可靠性。
"""

import json
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# 审计事件类型枚举
# ---------------------------------------------------------------------------

class AuditEventType(str, Enum):
    """审计事件类型。"""
    # 人类交互
    HUMAN_INTERACTION = "human_interaction"
    # 工具调用
    TOOL_CALL = "tool_call"
    # Agent动作
    AGENT_ACTION = "agent_action"
    # 配置变更
    CONFIG_CHANGE = "config_change"
    # 系统事件
    SYSTEM_EVENT = "system_event"
    # 安全事件
    SECURITY_EVENT = "security_event"
    # 数据访问
    DATA_ACCESS = "data_access"


# ---------------------------------------------------------------------------
# 审计事件数据类
# ---------------------------------------------------------------------------

@dataclass
class AuditEvent:
    """审计事件数据类。

    Attributes:
        timestamp: 事件时间戳（ISO格式）
        event_type: 事件类型
        actor: 执行者（用户ID、Agent ID等）
        action: 动作描述
        target: 作用目标（资源ID、工具名等）
        result: 执行结果（success/failure/pending）
        metadata: 附加元数据
        event_id: 事件唯一ID
        duration_ms: 执行耗时（毫秒）
        error_message: 错误信息（如果失败）
    """
    timestamp: str = ""
    event_type: str = ""
    actor: str = ""
    action: str = ""
    target: str = ""
    result: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)
    event_id: str = ""
    duration_ms: Optional[float] = None
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        """初始化后处理，确保必要字段有值。"""
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.event_id:
            self.event_id = self._generate_id()

    @staticmethod
    def _generate_id() -> str:
        """生成事件唯一ID。

        Returns:
            基于时间戳和线程ID的唯一ID
        """
        import hashlib
        raw = f"{time.time()}-{threading.get_ident()}-{id(object())}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return asdict(self)

    def to_json(self, indent: Optional[int] = None) -> str:
        """转换为JSON字符串。

        Args:
            indent: JSON缩进

        Returns:
            JSON格式字符串
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# AuditLog: 审计日志存储
# ---------------------------------------------------------------------------

class AuditLog:
    """审计日志存储。

    采用内存+文件双写策略：
    - 内存存储：支持快速查询
    - 文件存储：持久化保存，防止数据丢失

    使用示例::

        audit_log = AuditLog(filepath="/var/log/audit.jsonl")
        event = AuditEvent(
            event_type="tool_call",
            actor="agent-1",
            action="execute",
            target="web_search",
            result="success",
        )
        audit_log.append(event)
    """

    def __init__(
        self,
        filepath: Optional[str] = None,
        max_memory_events: int = 10000,
        auto_flush_interval: float = 5.0,
    ):
        """初始化审计日志存储。

        Args:
            filepath: 审计日志文件路径，为None则仅使用内存存储
            max_memory_events: 内存中最大保存事件数
            auto_flush_interval: 自动刷盘间隔（秒），0表示不自动刷盘
        """
        self._filepath = filepath
        self._max_memory_events = max_memory_events
        self._events: List[AuditEvent] = []
        self._lock = threading.RLock()
        self._file = None
        self._flush_interval = auto_flush_interval
        self._last_flush = time.time()
        self._total_count = 0

        # 初始化文件
        if filepath:
            self._init_file()

    def _init_file(self) -> None:
        """初始化审计日志文件。"""
        if not self._filepath:
            return
        log_dir = os.path.dirname(self._filepath)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        try:
            self._file = open(self._filepath, 'a', encoding='utf-8')
        except OSError as e:
            # 文件打开失败时降级为纯内存模式
            self._file = None
            import sys
            print(f"[AuditLog] Warning: Cannot open audit file {self._filepath}: {e}", file=sys.stderr)

    def append(self, event: AuditEvent) -> None:
        """追加审计事件。

        Args:
            event: 审计事件
        """
        with self._lock:
            self._events.append(event)
            self._total_count += 1

            # 内存淘汰策略
            if len(self._events) > self._max_memory_events:
                self._events = self._events[-self._max_memory_events:]

            # 文件写入
            if self._file:
                try:
                    self._file.write(event.to_json() + "\n")
                    # 自动刷盘
                    if self._flush_interval > 0:
                        now = time.time()
                        if now - self._last_flush >= self._flush_interval:
                            self._file.flush()
                            self._last_flush = now
                except OSError:
                    pass  # 文件写入失败时静默处理

    def query(
        self,
        event_type: Optional[str] = None,
        actor: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        result: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditEvent]:
        """查询审计记录。

        Args:
            event_type: 按事件类型过滤
            actor: 按执行者过滤
            start_time: 起始时间（ISO格式）
            end_time: 结束时间（ISO格式）
            result: 按执行结果过滤
            limit: 返回记录数上限
            offset: 跳过前N条记录

        Returns:
            匹配的审计事件列表
        """
        with self._lock:
            events = list(self._events)

        # 过滤
        filtered = []
        for event in events:
            if event_type and event.event_type != event_type:
                continue
            if actor and event.actor != actor:
                continue
            if result and event.result != result:
                continue
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            filtered.append(event)

        # 分页
        return filtered[offset:offset + limit]

    def query_by_actor(self, actor: str, limit: int = 100) -> List[AuditEvent]:
        """按执行者查询审计记录。

        Args:
            actor: 执行者标识
            limit: 返回记录数上限

        Returns:
            匹配的审计事件列表
        """
        return self.query(actor=actor, limit=limit)

    def query_by_type(self, event_type: str, limit: int = 100) -> List[AuditEvent]:
        """按事件类型查询审计记录。

        Args:
            event_type: 事件类型
            limit: 返回记录数上限

        Returns:
            匹配的审计事件列表
        """
        return self.query(event_type=event_type, limit=limit)

    def count(self, event_type: Optional[str] = None) -> int:
        """统计审计事件数量。

        Args:
            event_type: 按事件类型过滤

        Returns:
            事件数量
        """
        with self._lock:
            if event_type is None:
                return len(self._events)
            return sum(1 for e in self._events if e.event_type == event_type)

    def flush(self) -> None:
        """强制刷盘。"""
        if self._file:
            try:
                self._file.flush()
                self._last_flush = time.time()
            except OSError:
                pass

    def close(self) -> None:
        """关闭审计日志存储，释放资源。"""
        self.flush()
        if self._file:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None

    def clear(self) -> None:
        """清空内存中的审计记录。"""
        with self._lock:
            self._events.clear()

    @property
    def total_count(self) -> int:
        """总事件数（包括已被淘汰的）。"""
        return self._total_count

    def get_statistics(self) -> Dict[str, Any]:
        """获取审计日志统计信息。

        Returns:
            统计字典，包含各类型事件数量、最近事件等
        """
        with self._lock:
            type_counts: Dict[str, int] = defaultdict(int)
            result_counts: Dict[str, int] = defaultdict(int)
            actor_counts: Dict[str, int] = defaultdict(int)

            for event in self._events:
                type_counts[event.event_type] += 1
                result_counts[event.result] += 1
                actor_counts[event.actor] += 1

            return {
                "total_in_memory": len(self._events),
                "total_all_time": self._total_count,
                "by_type": dict(type_counts),
                "by_result": dict(result_counts),
                "top_actors": dict(
                    sorted(actor_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                ),
                "latest_event": self._events[-1].to_dict() if self._events else None,
            }

    def __del__(self) -> None:
        """析构时自动关闭。"""
        self.close()


# ---------------------------------------------------------------------------
# AuditHandler: 审计日志处理器
# ---------------------------------------------------------------------------

class AuditHandler:
    """审计日志处理器。

    提供高级别的审计日志记录接口，支持：
    - 人类交互事件记录
    - 工具调用记录（含参数、结果、耗时）
    - Agent动作记录
    - 配置变更记录
    - 自定义审计事件

    使用示例::

        audit = AuditHandler(filepath="/var/log/audit.jsonl")
        audit.log_tool_call(
            tool_name="web_search",
            params={"query": "Python logging"},
            result={"status": "ok", "results": 10},
            duration_ms=150.5,
            actor="agent-1",
        )
        audit.log_human_interaction(
            actor="user-42",
            action="provided_feedback",
            target="agent-1",
            metadata={"feedback": "Good response"},
        )
    """

    def __init__(
        self,
        filepath: Optional[str] = None,
        max_memory_events: int = 10000,
        auto_flush_interval: float = 5.0,
        audit_log: Optional[AuditLog] = None,
    ):
        """初始化审计日志处理器。

        Args:
            filepath: 审计日志文件路径
            max_memory_events: 内存中最大保存事件数
            auto_flush_interval: 自动刷盘间隔（秒）
            audit_log: 自定义AuditLog实例，为None时自动创建
        """
        if audit_log:
            self._audit_log = audit_log
        else:
            self._audit_log = AuditLog(
                filepath=filepath,
                max_memory_events=max_memory_events,
                auto_flush_interval=auto_flush_interval,
            )

    def log_human_interaction(
        self,
        actor: str,
        action: str,
        target: str = "",
        result: str = "success",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """记录人类交互事件。

        Args:
            actor: 用户标识
            action: 交互动作（如 provided_input, gave_feedback, approved_action）
            target: 交互目标（如 agent_id, task_id）
            result: 交互结果
            metadata: 附加信息

        Returns:
            创建的审计事件
        """
        event = AuditEvent(
            event_type=AuditEventType.HUMAN_INTERACTION.value,
            actor=actor,
            action=action,
            target=target,
            result=result,
            metadata=metadata or {},
        )
        self._audit_log.append(event)
        return event

    def log_tool_call(
        self,
        tool_name: str,
        params: Optional[Dict[str, Any]] = None,
        result: Optional[Any] = None,
        duration_ms: Optional[float] = None,
        actor: str = "system",
        error_message: Optional[str] = None,
    ) -> AuditEvent:
        """记录工具调用。

        Args:
            tool_name: 工具名称
            params: 调用参数
            result: 调用结果
            duration_ms: 调用耗时（毫秒）
            actor: 调用者（Agent ID等）
            error_message: 错误信息

        Returns:
            创建的审计事件
        """
        call_result = "failure" if error_message else "success"
        metadata: Dict[str, Any] = {}
        if params:
            metadata["params"] = params
        if result is not None:
            metadata["result"] = result

        event = AuditEvent(
            event_type=AuditEventType.TOOL_CALL.value,
            actor=actor,
            action=f"call:{tool_name}",
            target=tool_name,
            result=call_result,
            metadata=metadata,
            duration_ms=duration_ms,
            error_message=error_message,
        )
        self._audit_log.append(event)
        return event

    def log_agent_action(
        self,
        agent_id: str,
        action: str,
        target: str = "",
        result: str = "success",
        metadata: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
    ) -> AuditEvent:
        """记录Agent动作。

        Args:
            agent_id: Agent标识
            action: 动作描述（如 reasoning, planning, executing）
            target: 动作目标
            result: 执行结果
            metadata: 附加信息
            duration_ms: 执行耗时（毫秒）

        Returns:
            创建的审计事件
        """
        event = AuditEvent(
            event_type=AuditEventType.AGENT_ACTION.value,
            actor=agent_id,
            action=action,
            target=target,
            result=result,
            metadata=metadata or {},
            duration_ms=duration_ms,
        )
        self._audit_log.append(event)
        return event

    def log_config_change(
        self,
        actor: str,
        config_key: str,
        old_value: Any = None,
        new_value: Any = None,
        result: str = "success",
    ) -> AuditEvent:
        """记录配置变更。

        Args:
            actor: 变更者
            config_key: 配置键名
            old_value: 旧值
            new_value: 新值
            result: 变更结果

        Returns:
            创建的审计事件
        """
        metadata = {
            "config_key": config_key,
            "old_value": old_value,
            "new_value": new_value,
        }
        event = AuditEvent(
            event_type=AuditEventType.CONFIG_CHANGE.value,
            actor=actor,
            action=f"change_config:{config_key}",
            target=config_key,
            result=result,
            metadata=metadata,
        )
        self._audit_log.append(event)
        return event

    def log_event(
        self,
        event_type: str,
        actor: str,
        action: str,
        target: str = "",
        result: str = "success",
        metadata: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        error_message: Optional[str] = None,
    ) -> AuditEvent:
        """记录自定义审计事件。

        Args:
            event_type: 事件类型
            actor: 执行者
            action: 动作描述
            target: 作用目标
            result: 执行结果
            metadata: 附加元数据
            duration_ms: 执行耗时（毫秒）
            error_message: 错误信息

        Returns:
            创建的审计事件
        """
        event = AuditEvent(
            event_type=event_type,
            actor=actor,
            action=action,
            target=target,
            result=result,
            metadata=metadata or {},
            duration_ms=duration_ms,
            error_message=error_message,
        )
        self._audit_log.append(event)
        return event

    def query(
        self,
        event_type: Optional[str] = None,
        actor: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        result: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditEvent]:
        """查询审计记录。

        Args:
            event_type: 按事件类型过滤
            actor: 按执行者过滤
            start_time: 起始时间（ISO格式）
            end_time: 结束时间（ISO格式）
            result: 按执行结果过滤
            limit: 返回记录数上限
            offset: 跳过前N条记录

        Returns:
            匹配的审计事件列表
        """
        return self._audit_log.query(
            event_type=event_type,
            actor=actor,
            start_time=start_time,
            end_time=end_time,
            result=result,
            limit=limit,
            offset=offset,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """获取审计日志统计信息。

        Returns:
            统计字典
        """
        return self._audit_log.get_statistics()

    def flush(self) -> None:
        """强制刷盘。"""
        self._audit_log.flush()

    def close(self) -> None:
        """关闭审计处理器，释放资源。"""
        self._audit_log.close()

    @property
    def audit_log(self) -> AuditLog:
        """获取底层审计日志存储。"""
        return self._audit_log
