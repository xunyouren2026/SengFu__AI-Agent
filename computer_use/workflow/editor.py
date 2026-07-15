"""
工作流编辑器模块

提供可视化编辑器后端功能，包括工作流和步骤的CRUD操作。
仅使用Python标准库实现。
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class StepType(Enum):
    """步骤类型枚举。"""
    MOUSE_MOVE = "mouse_move"
    MOUSE_CLICK = "mouse_click"
    MOUSE_DOUBLE_CLICK = "mouse_double_click"
    MOUSE_RIGHT_CLICK = "mouse_right_click"
    MOUSE_DRAG = "mouse_drag"
    MOUSE_SCROLL = "mouse_scroll"
    KEY_PRESS = "key_press"
    KEY_TYPE = "key_type"
    KEY_COMBINATION = "key_combination"
    CLIPBOARD_COPY = "clipboard_copy"
    CLIPBOARD_PASTE = "clipboard_paste"
    WINDOW_FOCUS = "window_focus"
    WINDOW_RESIZE = "window_resize"
    WINDOW_MOVE = "window_move"
    WINDOW_MINIMIZE = "window_minimize"
    WINDOW_MAXIMIZE = "window_maximize"
    WINDOW_CLOSE = "window_close"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    CUSTOM = "custom"


@dataclass
class Step:
    """
    步骤数据类。
    
    Attributes:
        id: 步骤唯一标识符
        type: 步骤类型
        params: 步骤参数
        description: 步骤描述
        enabled: 是否启用
        timeout: 超时时间（秒）
        retry: 重试次数
    """
    id: str
    type: str
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    enabled: bool = True
    timeout: float = 30.0
    retry: int = 0
    
    def __post_init__(self):
        """初始化后处理。"""
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Step":
        """从字典创建。"""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=data["type"],
            params=data.get("params", {}),
            description=data.get("description", ""),
            enabled=data.get("enabled", True),
            timeout=data.get("timeout", 30.0),
            retry=data.get("retry", 0),
        )
    
    def copy(self) -> "Step":
        """创建步骤副本（生成新ID）。"""
        new_step = Step.from_dict(self.to_dict())
        new_step.id = str(uuid.uuid4())
        return new_step
    
    def update(self, **kwargs) -> None:
        """更新步骤属性。"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass
class Workflow:
    """
    工作流数据类。
    
    Attributes:
        id: 工作流唯一标识符
        name: 工作流名称
        steps: 步骤列表
        variables: 变量字典
        metadata: 元数据
    """
    id: str
    name: str
    steps: List[Step] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理。"""
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.metadata:
            self.metadata = {
                "created_at": time.time(),
                "updated_at": time.time(),
                "version": "1.0",
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "id": self.id,
            "name": self.name,
            "steps": [step.to_dict() for step in self.steps],
            "variables": self.variables,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Workflow":
        """从字典创建。"""
        steps = [Step.from_dict(s) for s in data.get("steps", [])]
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            steps=steps,
            variables=data.get("variables", {}),
            metadata=data.get("metadata", {}),
        )
    
    def to_json(self) -> str:
        """转换为JSON字符串。"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "Workflow":
        """从JSON字符串创建。"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_step(self, step_id: str) -> Optional[Step]:
        """获取指定步骤。"""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None
    
    def get_step_index(self, step_id: str) -> int:
        """获取步骤索引。"""
        for i, step in enumerate(self.steps):
            if step.id == step_id:
                return i
        return -1
    
    def update_metadata(self, **kwargs) -> None:
        """更新元数据。"""
        self.metadata.update(kwargs)
        self.metadata["updated_at"] = time.time()


class WorkflowEditor:
    """
    可视化编辑器后端。
    
    提供工作流的创建、编辑、删除等操作。
    """
    
    def __init__(self):
        """初始化工作流编辑器。"""
        self._workflows: Dict[str, Workflow] = {}
        self._history: Dict[str, List[Dict[str, Any]]] = {}
        self._max_history = 50
    
    def create_workflow(self, name: str, workflow_id: Optional[str] = None) -> Workflow:
        """
        创建工作流。
        
        Args:
            name: 工作流名称
            workflow_id: 可选的工作流ID，不提供则自动生成
            
        Returns:
            创建的工作流对象
        """
        if workflow_id and workflow_id in self._workflows:
            raise ValueError(f"工作流ID已存在: {workflow_id}")
        
        workflow = Workflow(
            id=workflow_id or str(uuid.uuid4()),
            name=name,
        )
        self._workflows[workflow.id] = workflow
        self._save_history(workflow.id)
        return workflow
    
    def add_step(
        self,
        workflow_id: str,
        step: Union[Step, Dict[str, Any]],
    ) -> Step:
        """
        添加步骤到工作流末尾。
        
        Args:
            workflow_id: 工作流ID
            step: 步骤对象或字典
            
        Returns:
            添加的步骤
        """
        workflow = self._get_workflow(workflow_id)
        
        if isinstance(step, dict):
            step = Step.from_dict(step)
        
        workflow.steps.append(step)
        workflow.update_metadata()
        self._save_history(workflow_id)
        return step
    
    def insert_step(
        self,
        workflow_id: str,
        position: int,
        step: Union[Step, Dict[str, Any]],
    ) -> Step:
        """
        在指定位置插入步骤。
        
        Args:
            workflow_id: 工作流ID
            position: 插入位置（0-based）
            step: 步骤对象或字典
            
        Returns:
            插入的步骤
        """
        workflow = self._get_workflow(workflow_id)
        
        if isinstance(step, dict):
            step = Step.from_dict(step)
        
        position = max(0, min(position, len(workflow.steps)))
        workflow.steps.insert(position, step)
        workflow.update_metadata()
        self._save_history(workflow_id)
        return step
    
    def remove_step(self, workflow_id: str, step_id: str) -> Optional[Step]:
        """
        删除步骤。
        
        Args:
            workflow_id: 工作流ID
            step_id: 步骤ID
            
        Returns:
            被删除的步骤，不存在则返回None
        """
        workflow = self._get_workflow(workflow_id)
        
        for i, step in enumerate(workflow.steps):
            if step.id == step_id:
                removed = workflow.steps.pop(i)
                workflow.update_metadata()
                self._save_history(workflow_id)
                return removed
        
        return None
    
    def update_step(
        self,
        workflow_id: str,
        step_id: str,
        updates: Dict[str, Any],
    ) -> Optional[Step]:
        """
        更新步骤。
        
        Args:
            workflow_id: 工作流ID
            step_id: 步骤ID
            updates: 更新字段字典
            
        Returns:
            更新后的步骤，不存在则返回None
        """
        workflow = self._get_workflow(workflow_id)
        step = workflow.get_step(step_id)
        
        if step is None:
            return None
        
        step.update(**updates)
        workflow.update_metadata()
        self._save_history(workflow_id)
        return step
    
    def reorder_steps(
        self,
        workflow_id: str,
        new_order: List[str],
    ) -> Workflow:
        """
        重排步骤顺序。
        
        Args:
            workflow_id: 工作流ID
            new_order: 新的步骤ID顺序列表
            
        Returns:
            更新后的工作流
        """
        workflow = self._get_workflow(workflow_id)
        
        if len(new_order) != len(workflow.steps):
            raise ValueError("新顺序列表长度必须与步骤数量一致")
        
        step_map = {step.id: step for step in workflow.steps}
        
        if not all(step_id in step_map for step_id in new_order):
            raise ValueError("新顺序列表包含无效的步骤ID")
        
        workflow.steps = [step_map[step_id] for step_id in new_order]
        workflow.update_metadata()
        self._save_history(workflow_id)
        return workflow
    
    def duplicate_step(
        self,
        workflow_id: str,
        step_id: str,
    ) -> Optional[Step]:
        """
        复制步骤。
        
        Args:
            workflow_id: 工作流ID
            step_id: 要复制的步骤ID
            
        Returns:
            新创建的步骤副本，原步骤不存在则返回None
        """
        workflow = self._get_workflow(workflow_id)
        step = workflow.get_step(step_id)
        
        if step is None:
            return None
        
        new_step = step.copy()
        new_step.description = f"{step.description} (副本)" if step.description else "副本"
        
        index = workflow.get_step_index(step_id)
        workflow.steps.insert(index + 1, new_step)
        workflow.update_metadata()
        self._save_history(workflow_id)
        return new_step
    
    def get_workflow(self, workflow_id: str) -> Workflow:
        """
        获取工作流。
        
        Args:
            workflow_id: 工作流ID
            
        Returns:
            工作流对象
        """
        return self._get_workflow(workflow_id)
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """
        列出所有工作流。
        
        Returns:
            工作流基本信息列表
        """
        return [
            {
                "id": wf.id,
                "name": wf.name,
                "step_count": len(wf.steps),
                "created_at": wf.metadata.get("created_at"),
                "updated_at": wf.metadata.get("updated_at"),
            }
            for wf in self._workflows.values()
        ]
    
    def delete_workflow(self, workflow_id: str) -> bool:
        """
        删除工作流。
        
        Args:
            workflow_id: 工作流ID
            
        Returns:
            是否成功删除
        """
        if workflow_id in self._workflows:
            del self._workflows[workflow_id]
            if workflow_id in self._history:
                del self._history[workflow_id]
            return True
        return False
    
    def clone_workflow(
        self,
        workflow_id: str,
        new_name: Optional[str] = None,
    ) -> Workflow:
        """
        克隆工作流。
        
        Args:
            workflow_id: 源工作流ID
            new_name: 新工作流名称，不提供则添加"副本"后缀
            
        Returns:
            克隆的工作流
        """
        source = self._get_workflow(workflow_id)
        
        # 深拷贝步骤
        new_steps = []
        for step in source.steps:
            new_step = step.copy()
            new_steps.append(new_step)
        
        cloned = Workflow(
            id=str(uuid.uuid4()),
            name=new_name or f"{source.name} (副本)",
            steps=new_steps,
            variables=dict(source.variables),
            metadata={
                "created_at": time.time(),
                "updated_at": time.time(),
                "version": source.metadata.get("version", "1.0"),
                "cloned_from": workflow_id,
            },
        )
        
        self._workflows[cloned.id] = cloned
        self._save_history(cloned.id)
        return cloned
    
    def export_workflow(self, workflow_id: str) -> str:
        """
        导出工作流为JSON字符串。
        
        Args:
            workflow_id: 工作流ID
            
        Returns:
            JSON字符串
        """
        workflow = self._get_workflow(workflow_id)
        return workflow.to_json()
    
    def import_workflow(self, json_str: str) -> Workflow:
        """
        从JSON字符串导入工作流。
        
        Args:
            json_str: JSON字符串
            
        Returns:
            导入的工作流
        """
        workflow = Workflow.from_json(json_str)
        
        # 确保ID唯一
        if workflow.id in self._workflows:
            workflow.id = str(uuid.uuid4())
        
        workflow.update_metadata(imported_at=time.time())
        self._workflows[workflow.id] = workflow
        self._save_history(workflow.id)
        return workflow
    
    def undo(self, workflow_id: str) -> Optional[Workflow]:
        """
        撤销操作。
        
        Args:
            workflow_id: 工作流ID
            
        Returns:
            撤销后的工作流，无历史记录则返回None
        """
        if workflow_id not in self._history or len(self._history[workflow_id]) < 2:
            return None
        
        # 移除当前状态
        self._history[workflow_id].pop()
        
        # 恢复上一个状态
        previous_state = self._history[workflow_id][-1]
        workflow = Workflow.from_dict(previous_state)
        self._workflows[workflow_id] = workflow
        return workflow
    
    def get_workflow_stats(self, workflow_id: str) -> Dict[str, Any]:
        """
        获取工作流统计信息。
        
        Args:
            workflow_id: 工作流ID
            
        Returns:
            统计信息字典
        """
        workflow = self._get_workflow(workflow_id)
        
        type_counts = {}
        enabled_count = 0
        total_timeout = 0.0
        
        for step in workflow.steps:
            type_counts[step.type] = type_counts.get(step.type, 0) + 1
            if step.enabled:
                enabled_count += 1
            total_timeout += step.timeout
        
        return {
            "total_steps": len(workflow.steps),
            "enabled_steps": enabled_count,
            "disabled_steps": len(workflow.steps) - enabled_count,
            "step_types": type_counts,
            "total_timeout": total_timeout,
            "variables_count": len(workflow.variables),
        }
    
    def _get_workflow(self, workflow_id: str) -> Workflow:
        """获取工作流（内部方法）。"""
        if workflow_id not in self._workflows:
            raise ValueError(f"工作流不存在: {workflow_id}")
        return self._workflows[workflow_id]
    
    def _save_history(self, workflow_id: str) -> None:
        """保存历史记录（内部方法）。"""
        if workflow_id not in self._history:
            self._history[workflow_id] = []
        
        workflow = self._workflows[workflow_id]
        self._history[workflow_id].append(workflow.to_dict())
        
        # 限制历史记录数量
        if len(self._history[workflow_id]) > self._max_history:
            self._history[workflow_id].pop(0)
