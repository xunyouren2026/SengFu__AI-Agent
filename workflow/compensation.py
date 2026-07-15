"""
Saga补偿处理器

实现Saga模式的补偿机制，在工作流执行失败时按逆序执行
已成功节点的补偿动作，保证数据一致性。
"""

import copy
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .graph_engine import DAGEngine, DAGNode, NodeState


class CompensationError(Exception):
    """补偿操作异常"""
    pass


class PartialCompensationError(CompensationError):
    """部分补偿异常"""
    pass


@dataclass
class CompensationLog:
    """
    补偿日志条目

    Attributes:
        log_id: 日志ID
        node_id: 节点ID
        action: 补偿动作名称
        status: 补偿状态
        input: 补偿输入
        output: 补偿输出
        error: 错误信息
        started_at: 开始时间
        completed_at: 完成时间
        duration: 执行时长
        retry_count: 重试次数
    """
    log_id: str = ""
    node_id: str = ""
    action: str = ""
    status: str = "pending"
    input: Any = None
    output: Any = None
    error: Optional[str] = None
    started_at: float = 0.0
    completed_at: float = 0.0
    duration: float = 0.0
    retry_count: int = 0

    def __post_init__(self):
        if not self.log_id:
            self.log_id = str(uuid.uuid4())[:8]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "log_id": self.log_id,
            "node_id": self.node_id,
            "action": self.action,
            "status": self.status,
            "input": self.input,
            "output": self.output,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration": self.duration,
            "retry_count": self.retry_count,
        }


@dataclass
class CompensationAction:
    """
    补偿动作定义

    Attributes:
        node_id: 关联的节点ID
        compensation_fn: 补偿函数
        name: 补偿动作名称
        timeout: 超时时间
        max_retries: 最大重试次数
        required: 是否必须成功（失败则中止后续补偿）
    """
    node_id: str
    compensation_fn: Callable
    name: str = ""
    timeout: Optional[float] = None
    max_retries: int = 1
    required: bool = True

    def __post_init__(self):
        if not self.name:
            self.name = f"compensate_{self.node_id}"


class SagaCompensation:
    """
    Saga补偿处理器

    管理工作流中的补偿动作，在节点执行失败时按逆序执行
    已成功节点的补偿操作。

    Features:
        - 注册/注销补偿动作
        - 按逆序执行补偿
        - 部分补偿支持
        - 补偿重试
        - 补偿日志记录
        - 线程安全

    Usage:
        saga = SagaCompensation()

        # 注册补偿动作
        saga.register_compensation("create_order", rollback_order)
        saga.register_compensation("reserve_stock", release_stock)
        saga.register_compensation("charge_payment", refund_payment)

        # 标记节点执行成功
        saga.mark_completed("create_order", order_data)
        saga.mark_completed("reserve_stock", stock_data)

        # 某个节点失败时执行补偿
        saga.execute_compensation("charge_payment", error_info)
    """

    def __init__(self, saga_id: str = ""):
        self.saga_id = saga_id or str(uuid.uuid4())[:8]
        self._lock = threading.RLock()

        # 补偿动作注册表
        self._actions: Dict[str, CompensationAction] = {}

        # 已完成的节点（按执行顺序）
        self._completed_nodes: OrderedDict[str, Dict[str, Any]] = OrderedDict()

        # 补偿日志
        self._logs: List[CompensationLog] = []

        # 补偿状态
        self._compensating = False
        self._compensated_nodes: List[str] = []

    # ============================================================
    # 补偿动作注册
    # ============================================================

    def register_compensation(
        self,
        node_id: str,
        compensation_fn: Callable,
        name: str = "",
        timeout: Optional[float] = None,
        max_retries: int = 1,
        required: bool = True,
    ) -> None:
        """
        注册补偿动作

        Args:
            node_id: 关联的节点ID
            compensation_fn: 补偿函数，签名: fn(node_output, context) -> result
            name: 补偿动作名称
            timeout: 超时时间（秒）
            max_retries: 最大重试次数
            required: 是否必须成功
        """
        with self._lock:
            action = CompensationAction(
                node_id=node_id,
                compensation_fn=compensation_fn,
                name=name,
                timeout=timeout,
                max_retries=max_retries,
                required=required,
            )
            self._actions[node_id] = action

    def unregister_compensation(self, node_id: str) -> bool:
        """
        注销补偿动作

        Args:
            node_id: 节点ID

        Returns:
            是否成功注销
        """
        with self._lock:
            if node_id in self._actions:
                del self._actions[node_id]
                return True
            return False

    def has_compensation(self, node_id: str) -> bool:
        """检查节点是否注册了补偿动作"""
        with self._lock:
            return node_id in self._actions

    def get_action(self, node_id: str) -> Optional[CompensationAction]:
        """获取补偿动作"""
        with self._lock:
            return self._actions.get(node_id)

    # ============================================================
    # 节点完成跟踪
    # ============================================================

    def mark_completed(
        self,
        node_id: str,
        output: Any = None,
    ) -> None:
        """
        标记节点执行成功

        Args:
            node_id: 节点ID
            output: 节点输出（补偿时使用）
        """
        with self._lock:
            self._completed_nodes[node_id] = {
                "output": copy.deepcopy(output),
                "completed_at": time.time(),
            }

    def mark_failed(self, node_id: str, error: str = "") -> None:
        """
        标记节点执行失败

        Args:
            node_id: 节点ID
            error: 错误信息
        """
        with self._lock:
            # 失败的节点不需要补偿，但记录日志
            log = CompensationLog(
                node_id=node_id,
                action="mark_failed",
                status="skipped",
                error=error,
                started_at=time.time(),
                completed_at=time.time(),
            )
            self._logs.append(log)

    def clear_completed(self) -> None:
        """清除所有已完成的节点记录"""
        with self._lock:
            self._completed_nodes.clear()

    def get_completed_nodes(self) -> List[str]:
        """获取已完成的节点ID列表（按执行顺序）"""
        with self._lock:
            return list(self._completed_nodes.keys())

    # ============================================================
    # 补偿执行
    # ============================================================

    def execute_compensation(
        self,
        failed_node_id: str,
        context: Optional[Dict[str, Any]] = None,
        stop_on_error: bool = False,
    ) -> List[CompensationLog]:
        """
        执行补偿

        按逆序遍历已完成的节点，对每个注册了补偿动作的节点执行补偿。

        Args:
            failed_node_id: 失败的节点ID
            context: 执行上下文
            stop_on_error: 遇到错误是否停止

        Returns:
            补偿日志列表
        """
        with self._lock:
            if self._compensating:
                raise CompensationError("补偿已在进行中")

            self._compensating = True
            context = context or {}
            context["_saga_id"] = self.saga_id
            context["_failed_node"] = failed_node_id

        try:
            # 获取需要补偿的节点（逆序）
            nodes_to_compensate = list(reversed(self._completed_nodes.keys()))

            # 只补偿失败节点之前的已完成节点
            compensate_list = []
            for nid in nodes_to_compensate:
                if nid == failed_node_id:
                    break
                if nid in self._actions:
                    compensate_list.append(nid)

            # 逆序执行补偿
            logs = []
            for nid in compensate_list:
                action = self._actions[nid]
                node_data = self._completed_nodes[nid]

                log = self._execute_single_compensation(
                    action, node_data, context
                )
                logs.append(log)
                self._logs.append(log)

                if log.status == "failed" and (stop_on_error or action.required):
                    # 必须成功的补偿失败，中止后续补偿
                    self._compensated_nodes = [
                        nid for nid in compensate_list
                        if nid in [l.node_id for l in logs and l.status == "success"]
                    ]
                    raise PartialCompensationError(
                        f"节点 {nid} 的补偿失败（required={action.required}），"
                        f"中止后续补偿。已补偿: {self._compensated_nodes}"
                    )

            self._compensated_nodes = compensate_list
            return logs

        finally:
            with self._lock:
                self._compensating = False

    def compensate_node(
        self,
        node_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> CompensationLog:
        """
        补偿单个节点

        Args:
            node_id: 节点ID
            context: 执行上下文

        Returns:
            补偿日志
        """
        with self._lock:
            if node_id not in self._actions:
                raise CompensationError(f"节点 {node_id} 没有注册补偿动作")

            if node_id not in self._completed_nodes:
                raise CompensationError(f"节点 {node_id} 未标记为已完成")

            action = self._actions[node_id]
            node_data = self._completed_nodes[node_id]
            context = context or {}

            log = self._execute_single_compensation(action, node_data, context)
            self._logs.append(log)
            return log

    def _execute_single_compensation(
        self,
        action: CompensationAction,
        node_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> CompensationLog:
        """执行单个补偿动作"""
        log = CompensationLog(
            node_id=action.node_id,
            action=action.name,
            status="running",
            started_at=time.time(),
        )

        last_error = None
        for attempt in range(action.max_retries):
            try:
                output = node_data.get("output")
                result = action.compensation_fn(output, context)

                log.status = "success"
                log.output = result
                log.completed_at = time.time()
                log.duration = log.completed_at - log.started_at
                log.retry_count = attempt
                return log

            except Exception as e:
                last_error = str(e)
                log.retry_count = attempt + 1
                if attempt < action.max_retries - 1:
                    # 指数退避
                    time.sleep(min(2 ** (attempt + 1), 10))

        log.status = "failed"
        log.error = last_error
        log.completed_at = time.time()
        log.duration = log.completed_at - log.started_at
        return log

    # ============================================================
    # 日志与状态
    # ============================================================

    def get_logs(self) -> List[CompensationLog]:
        """获取所有补偿日志"""
        with self._lock:
            return list(self._logs)

    def get_compensated_nodes(self) -> List[str]:
        """获取已补偿的节点列表"""
        with self._lock:
            return list(self._compensated_nodes)

    def is_compensating(self) -> bool:
        """是否正在执行补偿"""
        with self._lock:
            return self._compensating

    def get_summary(self) -> Dict[str, Any]:
        """获取补偿摘要"""
        with self._lock:
            total = len(self._logs)
            success = sum(1 for l in self._logs if l.status == "success")
            failed = sum(1 for l in self._logs if l.status == "failed")
            return {
                "saga_id": self.saga_id,
                "registered_actions": len(self._actions),
                "completed_nodes": len(self._completed_nodes),
                "compensated_nodes": len(self._compensated_nodes),
                "total_compensations": total,
                "successful_compensations": success,
                "failed_compensations": failed,
                "is_compensating": self._compensating,
            }

    def reset(self) -> None:
        """重置补偿处理器状态"""
        with self._lock:
            self._completed_nodes.clear()
            self._compensated_nodes = []
            self._compensating = False

    def clear_logs(self) -> None:
        """清除补偿日志"""
        with self._lock:
            self._logs.clear()

    def __repr__(self) -> str:
        summary = self.get_summary()
        return (
            f"<SagaCompensation id={self.saga_id!r} "
            f"actions={summary['registered_actions']} "
            f"completed={summary['completed_nodes']}>"
        )
