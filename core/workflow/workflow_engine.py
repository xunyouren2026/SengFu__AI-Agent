"""
Workflow Engine
工作流引擎

提供DAG工作流定义、执行、监控等功能
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, AsyncGenerator

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """工作流状态"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeStatus(Enum):
    """节点状态"""
    PENDING = "pending"
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class NodeType(Enum):
    """节点类型"""
    TASK = "task"
    CONDITION = "condition"
    PARALLEL = "parallel"
    LOOP = "loop"
    SUBWORKFLOW = "subworkflow"
    DELAY = "delay"
    WEBHOOK = "webhook"
    SCRIPT = "script"


@dataclass
class WorkflowNode:
    """工作流节点"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    node_type: NodeType = NodeType.TASK
    status: NodeStatus = NodeStatus.PENDING
    config: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, str] = field(default_factory=dict)  # name -> source_node_id.output_name
    outputs: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout: int = 300  # 秒
    retry_count: int = 0
    max_retries: int = 3
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "node_type": self.node_type.value,
            "status": self.status.value,
            "config": self.config,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "dependencies": self.dependencies,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }


@dataclass
class WorkflowEdge:
    """工作流边"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = ""  # source node id
    target: str = ""  # target node id
    condition: Optional[str] = None  # 条件表达式
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "condition": self.condition,
            "label": self.label,
            "metadata": self.metadata,
        }


@dataclass
class WorkflowExecution:
    """工作流执行记录"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = ""
    status: WorkflowStatus = WorkflowStatus.PENDING
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    node_executions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "status": self.status.value,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "node_executions": self.node_executions,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class Workflow:
    """工作流定义"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    version: str = "1.0"
    nodes: Dict[str, WorkflowNode] = field(default_factory=dict)
    edges: List[WorkflowEdge] = field(default_factory=list)
    start_node: Optional[str] = None
    end_nodes: List[str] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    triggers: List[Dict[str, Any]] = field(default_factory=list)
    timeout: int = 3600  # 总超时
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "start_node": self.start_node,
            "end_nodes": self.end_nodes,
            "variables": self.variables,
            "triggers": self.triggers,
            "timeout": self.timeout,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }
    
    def add_node(self, node: WorkflowNode) -> str:
        """添加节点"""
        self.nodes[node.id] = node
        self.updated_at = time.time()
        return node.id
    
    def add_edge(self, edge: WorkflowEdge) -> str:
        """添加边"""
        self.edges.append(edge)
        # 更新依赖
        if edge.target in self.nodes:
            if edge.source not in self.nodes[edge.target].dependencies:
                self.nodes[edge.target].dependencies.append(edge.source)
        self.updated_at = time.time()
        return edge.id
    
    def get_entry_nodes(self) -> List[WorkflowNode]:
        """获取入口节点"""
        return [n for n in self.nodes.values() if not n.dependencies]
    
    def get_next_nodes(self, node_id: str) -> List[WorkflowNode]:
        """获取下一个节点"""
        next_ids = [e.target for e in self.edges if e.source == node_id]
        return [self.nodes[nid] for nid in next_ids if nid in self.nodes]


class NodeExecutor(ABC):
    """节点执行器基类"""
    
    @abstractmethod
    async def execute(
        self,
        node: WorkflowNode,
        inputs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """执行节点"""
        pass


class TaskNodeExecutor(NodeExecutor):
    """任务节点执行器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._llm_client = None
        
    async def execute(
        self,
        node: WorkflowNode,
        inputs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """执行任务节点"""
        task_config = node.config
        task_type = task_config.get("type", "llm")
        
        if task_type == "llm":
            return await self._execute_llm(task_config, inputs, context)
        elif task_type == "http":
            return await self._execute_http(task_config, inputs, context)
        elif task_type == "script":
            return await self._execute_script(task_config, inputs, context)
        else:
            return {"result": f"Unknown task type: {task_type}"}
    
    async def _execute_llm(
        self,
        config: Dict[str, Any],
        inputs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """执行LLM任务"""
        try:
            from openai import AsyncOpenAI
            
            api_key = self.config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                return {"result": "[Mock] LLM response - no API key"}
            
            if not self._llm_client:
                self._llm_client = AsyncOpenAI(api_key=api_key)
            
            prompt = config.get("prompt", "")
            # 替换变量
            for key, value in inputs.items():
                prompt = prompt.replace(f"{{{{{key}}}}}", str(value))
            
            response = await self._llm_client.chat.completions.create(
                model=config.get("model", "gpt-4o"),
                messages=[{"role": "user", "content": prompt}],
                temperature=config.get("temperature", 0.7),
            )
            
            return {"result": response.choices[0].message.content}
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _execute_http(
        self,
        config: Dict[str, Any],
        inputs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """执行HTTP请求"""
        try:
            import httpx
            
            url = config.get("url", "")
            method = config.get("method", "GET").upper()
            headers = config.get("headers", {})
            body = config.get("body")
            
            # 替换变量
            for key, value in inputs.items():
                url = url.replace(f"{{{{{key}}}}}", str(value))
            
            async with httpx.AsyncClient() as client:
                if method == "GET":
                    response = await client.get(url, headers=headers)
                elif method == "POST":
                    response = await client.post(url, json=body, headers=headers)
                elif method == "PUT":
                    response = await client.put(url, json=body, headers=headers)
                elif method == "DELETE":
                    response = await client.delete(url, headers=headers)
                else:
                    return {"error": f"Unsupported method: {method}"}
            
            return {
                "status_code": response.status_code,
                "result": response.text[:1000],
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _execute_script(
        self,
        config: Dict[str, Any],
        inputs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """执行脚本"""
        script = config.get("script", "")
        language = config.get("language", "python")
        
        if language == "python":
            try:
                # 安全执行
                local_vars = {"inputs": inputs, "context": context}
                exec(script, {"__builtins__": {}}, local_vars)
                return {"result": local_vars.get("result", "Script executed")}
            except Exception as e:
                return {"error": str(e)}
        
        return {"error": f"Unsupported language: {language}"}


class ConditionNodeExecutor(NodeExecutor):
    """条件节点执行器"""
    
    async def execute(
        self,
        node: WorkflowNode,
        inputs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """执行条件节点"""
        condition = node.config.get("condition", "true")
        
        # 简单条件评估
        try:
            # 替换变量
            for key, value in inputs.items():
                condition = condition.replace(f"{{{{{key}}}}}", str(value))
            
            # 评估条件
            result = eval(condition, {"__builtins__": {}}, inputs)
            
            return {"condition_result": bool(result)}
            
        except Exception as e:
            return {"condition_result": False, "error": str(e)}


class ParallelNodeExecutor(NodeExecutor):
    """并行节点执行器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
    async def execute(
        self,
        node: WorkflowNode,
        inputs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """执行并行节点"""
        # 并行执行由工作流引擎处理
        return {"parallel": True}


class WorkflowEngine:
    """
    工作流引擎
    
    功能：
    - 工作流定义
    - DAG执行
    - 并行执行
    - 错误处理
    - 状态监控
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._workflows: Dict[str, Workflow] = {}
        self._executions: Dict[str, WorkflowExecution] = {}
        self._executors: Dict[NodeType, NodeExecutor] = {}
        self._initialized = False
        
    async def initialize(self):
        """初始化工作流引擎"""
        if self._initialized:
            return
        
        # 注册执行器
        self._executors[NodeType.TASK] = TaskNodeExecutor(self.config)
        self._executors[NodeType.CONDITION] = ConditionNodeExecutor()
        self._executors[NodeType.PARALLEL] = ParallelNodeExecutor()
        
        self._initialized = True
        logger.info("Workflow engine initialized")
    
    async def create_workflow(
        self,
        name: str,
        description: str = "",
        nodes: Optional[List[WorkflowNode]] = None,
        edges: Optional[List[WorkflowEdge]] = None,
    ) -> Workflow:
        """创建工作流"""
        await self.initialize()
        
        workflow = Workflow(
            name=name,
            description=description,
        )
        
        if nodes:
            for node in nodes:
                workflow.add_node(node)
        
        if edges:
            for edge in edges:
                workflow.add_edge(edge)
        
        self._workflows[workflow.id] = workflow
        
        logger.info(f"Created workflow: {workflow.name} ({workflow.id})")
        
        return workflow
    
    async def execute(
        self,
        workflow_id: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> WorkflowExecution:
        """执行工作流"""
        await self.initialize()
        
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        # 创建执行记录
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            inputs=inputs or {},
        )
        self._executions[execution.id] = execution
        
        # 开始执行
        execution.status = WorkflowStatus.RUNNING
        execution.started_at = time.time()
        
        try:
            # 执行工作流
            await self._execute_workflow(workflow, execution)
            
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = time.time()
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
            execution.completed_at = time.time()
        
        return execution
    
    async def _execute_workflow(
        self,
        workflow: Workflow,
        execution: WorkflowExecution,
    ):
        """执行工作流内部逻辑"""
        # 初始化节点状态
        for node_id, node in workflow.nodes.items():
            node.status = NodeStatus.PENDING
            execution.node_executions[node_id] = {
                "status": NodeStatus.PENDING.value,
                "started_at": None,
                "completed_at": None,
            }
        
        # 获取入口节点
        entry_nodes = workflow.get_entry_nodes()
        
        # 执行DAG
        await self._execute_dag(workflow, execution, entry_nodes)
    
    async def _execute_dag(
        self,
        workflow: Workflow,
        execution: WorkflowExecution,
        nodes: List[WorkflowNode],
    ):
        """执行DAG"""
        completed_nodes = set()
        
        while nodes:
            # 找到可执行的节点
            ready_nodes = []
            for node in nodes:
                if node.id in completed_nodes:
                    continue
                
                # 检查依赖
                deps_completed = all(
                    dep_id in completed_nodes
                    for dep_id in node.dependencies
                )
                
                if deps_completed:
                    ready_nodes.append(node)
            
            if not ready_nodes:
                # 检查是否有未完成的节点
                remaining = [n for n in nodes if n.id not in completed_nodes]
                if remaining:
                    # 可能存在循环依赖
                    raise ValueError("Circular dependency detected")
                break
            
            # 并行执行就绪节点
            tasks = [
                self._execute_node(workflow, execution, node)
                for node in ready_nodes
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 更新状态
            for node, result in zip(ready_nodes, results):
                if isinstance(result, Exception):
                    node.status = NodeStatus.FAILED
                    node.error = str(result)
                else:
                    completed_nodes.add(node.id)
            
            # 获取下一批节点
            next_nodes = []
            for node in ready_nodes:
                if node.status == NodeStatus.COMPLETED:
                    next_nodes.extend(workflow.get_next_nodes(node.id))
            
            nodes = list(set(next_nodes))
    
    async def _execute_node(
        self,
        workflow: Workflow,
        execution: WorkflowExecution,
        node: WorkflowNode,
    ) -> Dict[str, Any]:
        """执行单个节点"""
        node.status = NodeStatus.RUNNING
        node.started_at = time.time()
        
        execution.node_executions[node.id] = {
            "status": NodeStatus.RUNNING.value,
            "started_at": node.started_at,
        }
        
        try:
            # 准备输入
            inputs = {}
            for input_name, source in node.inputs.items():
                source_node_id, output_name = source.split(".", 1)
                if source_node_id in workflow.nodes:
                    source_node = workflow.nodes[source_node_id]
                    inputs[input_name] = source_node.outputs.get(output_name)
            
            # 获取执行器
            executor = self._executors.get(node.node_type)
            if not executor:
                raise ValueError(f"No executor for node type: {node.node_type}")
            
            # 执行
            result = await executor.execute(node, inputs, execution.inputs)
            
            # 更新输出
            node.outputs = result
            node.status = NodeStatus.COMPLETED
            node.completed_at = time.time()
            
            execution.node_executions[node.id].update({
                "status": NodeStatus.COMPLETED.value,
                "completed_at": node.completed_at,
                "outputs": result,
            })
            
            return result
            
        except Exception as e:
            node.status = NodeStatus.FAILED
            node.error = str(e)
            node.completed_at = time.time()
            
            execution.node_executions[node.id].update({
                "status": NodeStatus.FAILED.value,
                "completed_at": node.completed_at,
                "error": str(e),
            })
            
            raise
    
    async def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """获取工作流"""
        return self._workflows.get(workflow_id)
    
    async def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """获取执行记录"""
        return self._executions.get(execution_id)
    
    async def list_workflows(self) -> List[Workflow]:
        """列出工作流"""
        return list(self._workflows.values())
    
    async def list_executions(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[WorkflowStatus] = None,
    ) -> List[WorkflowExecution]:
        """列出执行记录"""
        executions = list(self._executions.values())
        
        if workflow_id:
            executions = [e for e in executions if e.workflow_id == workflow_id]
        
        if status:
            executions = [e for e in executions if e.status == status]
        
        return executions
    
    async def pause_execution(self, execution_id: str) -> bool:
        """暂停执行"""
        execution = self._executions.get(execution_id)
        if execution and execution.status == WorkflowStatus.RUNNING:
            execution.status = WorkflowStatus.PAUSED
            return True
        return False
    
    async def resume_execution(self, execution_id: str) -> bool:
        """恢复执行"""
        execution = self._executions.get(execution_id)
        if execution and execution.status == WorkflowStatus.PAUSED:
            execution.status = WorkflowStatus.RUNNING
            return True
        return False
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """取消执行"""
        execution = self._executions.get(execution_id)
        if execution and execution.status in [WorkflowStatus.RUNNING, WorkflowStatus.PAUSED]:
            execution.status = WorkflowStatus.CANCELLED
            execution.completed_at = time.time()
            return True
        return False
    
    async def delete_workflow(self, workflow_id: str) -> bool:
        """删除工作流"""
        if workflow_id in self._workflows:
            del self._workflows[workflow_id]
            return True
        return False


# 全局实例
_workflow_engine: Optional[WorkflowEngine] = None


def get_workflow_engine() -> WorkflowEngine:
    """获取全局工作流引擎"""
    global _workflow_engine
    if _workflow_engine is None:
        _workflow_engine = WorkflowEngine()
    return _workflow_engine


async def init_workflow_engine(config: Optional[Dict[str, Any]] = None):
    """初始化全局工作流引擎"""
    global _workflow_engine
    _workflow_engine = WorkflowEngine(config)
    await _workflow_engine.initialize()
    return _workflow_engine
