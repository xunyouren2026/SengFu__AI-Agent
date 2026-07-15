"""
工作流状态持久化

管理工作流执行状态的保存、加载、检查点和恢复。
"""

import copy
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .graph_engine import DAGEngine, DAGNode, NodeState


@dataclass
class NodeStateSnapshot:
    """节点状态快照"""
    node_id: str
    state: str
    output: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "state": self.state,
            "output": self.output,
            "error": self.error,
            "retry_count": self.retry_count,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeStateSnapshot":
        return cls(**data)


@dataclass
class WorkflowState:
    """
    工作流状态

    完整记录工作流执行过程中的所有状态信息。

    Attributes:
        workflow_id: 工作流ID
        dag_snapshot: DAG快照
        node_states: 各节点状态
        variables: 工作流变量
        inputs: 工作流输入
        outputs: 工作流输出
        status: 整体状态
        created_at: 创建时间
        updated_at: 更新时间
        started_at: 开始执行时间
        completed_at: 完成时间
        error: 错误信息
        metadata: 元数据
    """
    workflow_id: str = ""
    dag_snapshot: Optional[Dict[str, Any]] = None
    node_states: Dict[str, NodeStateSnapshot] = dataclass_field(default_factory=dict)
    variables: Dict[str, Any] = dataclass_field(default_factory=dict)
    inputs: Dict[str, Any] = dataclass_field(default_factory=dict)
    outputs: Dict[str, Any] = dataclass_field(default_factory=dict)
    status: str = "pending"
    created_at: float = 0.0
    updated_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self):
        now = time.time()
        if not self.workflow_id:
            self.workflow_id = str(uuid.uuid4())[:8]
        if self.created_at == 0.0:
            self.created_at = now
        if self.updated_at == 0.0:
            self.updated_at = now

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "workflow_id": self.workflow_id,
            "dag_snapshot": self.dag_snapshot,
            "node_states": {
                nid: ns.to_dict() for nid, ns in self.node_states.items()
            },
            "variables": self.variables,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowState":
        """从字典创建"""
        data = copy.deepcopy(data)
        node_states_data = data.pop("node_states", {})
        node_states = {
            nid: NodeStateSnapshot.from_dict(ns)
            for nid, ns in node_states_data.items()
        }
        return cls(node_states=node_states, **data)


@dataclass
class Checkpoint:
    """
    检查点

    Attributes:
        checkpoint_id: 检查点ID
        workflow_id: 工作流ID
        state: 工作流状态快照
        created_at: 创建时间
        label: 检查点标签
    """
    checkpoint_id: str = ""
    workflow_id: str = ""
    state: Optional[WorkflowState] = None
    created_at: float = 0.0
    label: str = ""

    def __post_init__(self):
        now = time.time()
        if not self.checkpoint_id:
            self.checkpoint_id = str(uuid.uuid4())[:8]
        if self.created_at == 0.0:
            self.created_at = now

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "workflow_id": self.workflow_id,
            "state": self.state.to_dict() if self.state else None,
            "created_at": self.created_at,
            "label": self.label,
        }


class WorkflowStateManager:
    """
    工作流状态持久化管理器

    提供工作流状态的保存、加载、检查点和恢复功能。
    支持内存存储和文件持久化。

    Args:
        storage_dir: 状态文件存储目录
        auto_persist: 是否自动持久化到文件
        max_checkpoints: 每个工作流最大检查点数
        history_enabled: 是否记录执行历史
    """

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        auto_persist: bool = True,
        max_checkpoints: int = 10,
        history_enabled: bool = True,
    ):
        self._storage_dir = storage_dir
        self._auto_persist = auto_persist
        self._max_checkpoints = max_checkpoints
        self._history_enabled = history_enabled
        self._lock = threading.RLock()

        # 内存存储
        self._states: Dict[str, WorkflowState] = {}
        self._checkpoints: Dict[str, List[Checkpoint]] = {}
        self._history: Dict[str, List[Dict[str, Any]]] = {}

        # 创建存储目录
        if self._storage_dir and self._auto_persist:
            os.makedirs(self._storage_dir, exist_ok=True)

    def save_state(self, workflow_id: str, state: WorkflowState) -> None:
        """
        保存工作流状态

        Args:
            workflow_id: 工作流ID
            state: 工作流状态
        """
        with self._lock:
            state.workflow_id = workflow_id
            state.updated_at = time.time()
            self._states[workflow_id] = copy.deepcopy(state)

            # 记录历史
            if self._history_enabled:
                if workflow_id not in self._history:
                    self._history[workflow_id] = []
                self._history[workflow_id].append({
                    "status": state.status,
                    "timestamp": state.updated_at,
                    "node_states": {
                        nid: ns.state for nid, ns in state.node_states.items()
                    },
                })

            # 持久化到文件
            if self._auto_persist and self._storage_dir:
                self._persist_to_file(workflow_id, state)

    def load_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """
        加载工作流状态

        Args:
            workflow_id: 工作流ID

        Returns:
            工作流状态，不存在返回None
        """
        with self._lock:
            # 先查内存
            if workflow_id in self._states:
                return copy.deepcopy(self._states[workflow_id])

            # 从文件加载
            if self._storage_dir:
                state = self._load_from_file(workflow_id)
                if state:
                    self._states[workflow_id] = state
                    return copy.deepcopy(state)

            return None

    def delete_state(self, workflow_id: str) -> bool:
        """
        删除工作流状态

        Args:
            workflow_id: 工作流ID

        Returns:
            是否成功删除
        """
        with self._lock:
            if workflow_id in self._states:
                del self._states[workflow_id]
            if workflow_id in self._checkpoints:
                del self._checkpoints[workflow_id]
            if workflow_id in self._history:
                del self._history[workflow_id]

            # 删除文件
            if self._storage_dir:
                filepath = os.path.join(self._storage_dir, f"{workflow_id}.json")
                if os.path.exists(filepath):
                    os.remove(filepath)

            return True

    def update_node_state(
        self,
        workflow_id: str,
        node_id: str,
        state: str,
        output: Any = None,
        error: Optional[str] = None,
    ) -> Optional[WorkflowState]:
        """
        更新单个节点的状态

        Args:
            workflow_id: 工作流ID
            node_id: 节点ID
            state: 节点状态
            output: 节点输出
            error: 错误信息

        Returns:
            更新后的工作流状态
        """
        with self._lock:
            wf_state = self.load_state(workflow_id)
            if wf_state is None:
                wf_state = WorkflowState(workflow_id=workflow_id)

            now = time.time()
            snapshot = NodeStateSnapshot(
                node_id=node_id,
                state=state,
                output=output,
                error=error,
                start_time=now,
                end_time=now,
            )
            wf_state.node_states[node_id] = snapshot
            wf_state.updated_at = now

            # 更新整体状态
            self._update_workflow_status(wf_state)

            self.save_state(workflow_id, wf_state)
            return wf_state

    def set_variable(
        self,
        workflow_id: str,
        key: str,
        value: Any,
    ) -> None:
        """
        设置工作流变量

        Args:
            workflow_id: 工作流ID
            key: 变量名
            value: 变量值
        """
        with self._lock:
            wf_state = self.load_state(workflow_id)
            if wf_state is None:
                wf_state = WorkflowState(workflow_id=workflow_id)
            wf_state.variables[key] = value
            self.save_state(workflow_id, wf_state)

    def get_variable(self, workflow_id: str, key: str, default: Any = None) -> Any:
        """
        获取工作流变量

        Args:
            workflow_id: 工作流ID
            key: 变量名
            default: 默认值

        Returns:
            变量值
        """
        wf_state = self.load_state(workflow_id)
        if wf_state is None:
            return default
        return wf_state.variables.get(key, default)

    # ============================================================
    # 检查点
    # ============================================================

    def checkpoint(
        self,
        workflow_id: str,
        label: str = "",
    ) -> Optional[Checkpoint]:
        """
        创建检查点

        Args:
            workflow_id: 工作流ID
            label: 检查点标签

        Returns:
            检查点对象
        """
        with self._lock:
            state = self.load_state(workflow_id)
            if state is None:
                return None

            cp = Checkpoint(
                workflow_id=workflow_id,
                state=copy.deepcopy(state),
                label=label or f"checkpoint_{len(self._checkpoints.get(workflow_id, []))}",
            )

            if workflow_id not in self._checkpoints:
                self._checkpoints[workflow_id] = []
            self._checkpoints[workflow_id].append(cp)

            # 限制检查点数量
            if len(self._checkpoints[workflow_id]) > self._max_checkpoints:
                self._checkpoints[workflow_id] = \
                    self._checkpoints[workflow_id][-self._max_checkpoints:]

            return cp

    def restore(
        self,
        workflow_id: str,
        checkpoint_id: Optional[str] = None,
    ) -> Optional[WorkflowState]:
        """
        从检查点恢复

        Args:
            workflow_id: 工作流ID
            checkpoint_id: 检查点ID，None则恢复最新检查点

        Returns:
            恢复的工作流状态
        """
        with self._lock:
            checkpoints = self._checkpoints.get(workflow_id, [])
            if not checkpoints:
                return None

            if checkpoint_id:
                cp = None
                for c in checkpoints:
                    if c.checkpoint_id == checkpoint_id:
                        cp = c
                        break
                if cp is None:
                    return None
            else:
                cp = checkpoints[-1]

            if cp.state:
                restored = copy.deepcopy(cp.state)
                restored.status = "restored"
                restored.updated_at = time.time()
                self.save_state(workflow_id, restored)
                return restored

            return None

    def list_checkpoints(self, workflow_id: str) -> List[Checkpoint]:
        """
        列出工作流的所有检查点

        Args:
            workflow_id: 工作流ID

        Returns:
            检查点列表
        """
        with self._lock:
            return list(self._checkpoints.get(workflow_id, []))

    def delete_checkpoint(self, workflow_id: str, checkpoint_id: str) -> bool:
        """删除指定检查点"""
        with self._lock:
            checkpoints = self._checkpoints.get(workflow_id, [])
            for i, cp in enumerate(checkpoints):
                if cp.checkpoint_id == checkpoint_id:
                    checkpoints.pop(i)
                    return True
            return False

    # ============================================================
    # 历史记录
    # ============================================================

    def get_history(self, workflow_id: str) -> List[Dict[str, Any]]:
        """
        获取工作流执行历史

        Args:
            workflow_id: 工作流ID

        Returns:
            历史记录列表
        """
        with self._lock:
            return list(self._history.get(workflow_id, []))

    def clear_history(self, workflow_id: str) -> None:
        """清除工作流历史"""
        with self._lock:
            if workflow_id in self._history:
                self._history[workflow_id] = []

    # ============================================================
    # 查询
    # ============================================================

    def list_workflows(
        self,
        status: Optional[str] = None,
    ) -> List[WorkflowState]:
        """
        列出所有工作流状态

        Args:
            status: 按状态过滤

        Returns:
            工作流状态列表
        """
        with self._lock:
            states = list(self._states.values())
            if status:
                states = [s for s in states if s.status == status]
            return states

    def get_active_workflows(self) -> List[WorkflowState]:
        """获取所有活跃的工作流"""
        active_statuses = ("running", "pending", "waiting")
        return self.list_workflows(status=None)
        # 返回状态为活跃的工作流
        with self._lock:
            return [
                s for s in self._states.values()
                if s.status in active_statuses
            ]

    # ============================================================
    # 内部方法
    # ============================================================

    def _update_workflow_status(self, state: WorkflowState) -> None:
        """根据节点状态更新工作流整体状态"""
        if not state.node_states:
            return

        node_statuses = [ns.state for ns in state.node_states.values()]

        if "running" in node_statuses:
            state.status = "running"
        elif "failed" in node_statuses:
            state.status = "failed"
        elif all(s in ("success", "skipped") for s in node_statuses):
            state.status = "completed"
        elif "pending" in node_statuses or "ready" in node_statuses:
            state.status = "pending"
        else:
            state.status = "running"

    def _persist_to_file(self, workflow_id: str, state: WorkflowState) -> None:
        """持久化到文件"""
        if not self._storage_dir:
            return

        filepath = os.path.join(self._storage_dir, f"{workflow_id}.json")
        try:
            data = state.to_dict()
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, default=str, indent=2)
        except (IOError, OSError):
            pass

    def _load_from_file(self, workflow_id: str) -> Optional[WorkflowState]:
        """从文件加载"""
        if not self._storage_dir:
            return None

        filepath = os.path.join(self._storage_dir, f"{workflow_id}.json")
        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return WorkflowState.from_dict(data)
        except (json.JSONDecodeError, IOError):
            return None
