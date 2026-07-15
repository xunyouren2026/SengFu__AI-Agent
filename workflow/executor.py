"""
工作流执行器

执行DAG工作流，支持并行执行、条件分支、循环、
超时控制、错误处理与重试。
"""

import copy
import threading
import time
import traceback
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .graph_engine import DAGEdge, DAGEngine, DAGNode, NodeState


class ExecutionError(Exception):
    """执行异常"""
    pass


class TimeoutError(ExecutionError):
    """超时异常"""
    pass


class RetryExhaustedError(ExecutionError):
    """重试耗尽异常"""
    pass


@dataclass
class NodeResult:
    """
    节点执行结果

    Attributes:
        node_id: 节点ID
        state: 最终状态
        output: 输出数据
        error: 错误信息
        start_time: 开始时间
        end_time: 结束时间
        duration: 执行时长（秒）
        retry_count: 重试次数
    """
    node_id: str
    state: NodeState = NodeState.PENDING
    output: Any = None
    error: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    retry_count: int = 0

    @property
    def is_success(self) -> bool:
        return self.state == NodeState.SUCCESS

    @property
    def is_failed(self) -> bool:
        return self.state == NodeState.FAILED

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "node_id": self.node_id,
            "state": self.state.value,
            "output": self.output,
            "error": self.error,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "retry_count": self.retry_count,
        }


@dataclass
class WorkflowResult:
    """
    工作流执行结果

    Attributes:
        workflow_id: 工作流ID
        state: 最终状态
        node_results: 各节点结果
        outputs: 工作流输出
        error: 错误信息
        start_time: 开始时间
        end_time: 结束时间
        duration: 执行时长（秒）
    """
    workflow_id: str
    state: NodeState = NodeState.PENDING
    node_results: Dict[str, NodeResult] = dataclass_field(default_factory=dict)
    outputs: Dict[str, Any] = dataclass_field(default_factory=dict)
    error: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0

    @property
    def is_success(self) -> bool:
        return self.state == NodeState.SUCCESS

    @property
    def is_failed(self) -> bool:
        return self.state == NodeState.FAILED

    def get_node_result(self, node_id: str) -> Optional[NodeResult]:
        """获取指定节点的结果"""
        return self.node_results.get(node_id)

    def get_failed_nodes(self) -> List[NodeResult]:
        """获取所有失败的节点"""
        return [r for r in self.node_results.values() if r.is_failed]

    def get_success_nodes(self) -> List[NodeResult]:
        """获取所有成功的节点"""
        return [r for r in self.node_results.values() if r.is_success]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "workflow_id": self.workflow_id,
            "state": self.state.value,
            "node_results": {
                nid: r.to_dict() for nid, r in self.node_results.items()
            },
            "outputs": self.outputs,
            "error": self.error,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
        }


class WorkflowExecutor:
    """
    工作流执行器

    执行DAG工作流，支持：
    - 并行执行无依赖节点（ThreadPoolExecutor）
    - 条件分支执行
    - 循环节点执行
    - 人工审批等待（事件驱动）
    - 超时控制
    - 错误处理与自动重试
    - 执行状态跟踪
    - 自定义节点执行函数

    Args:
        max_workers: 最大并行工作线程数
        default_timeout: 默认节点超时时间（秒）
        node_executor: 自定义节点执行函数
    """

    def __init__(
        self,
        max_workers: int = 4,
        default_timeout: Optional[float] = None,
        node_executor: Optional[Callable] = None,
    ):
        self._max_workers = max_workers
        self._default_timeout = default_timeout
        self._node_executor = node_executor
        self._lock = threading.Lock()
        self._approval_events: Dict[str, threading.Event] = {}
        self._approval_results: Dict[str, Any] = {}

    def execute(
        self,
        dag: DAGEngine,
        inputs: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> WorkflowResult:
        """
        执行工作流

        Args:
            dag: DAG引擎实例
            inputs: 工作流输入
            context: 执行上下文

        Returns:
            工作流执行结果
        """
        workflow_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # 初始化
        inputs = inputs or {}
        context = context or {}
        context["_workflow_id"] = workflow_id
        context["_inputs"] = inputs
        context["_node_outputs"] = {}

        result = WorkflowResult(
            workflow_id=workflow_id,
            start_time=start_time,
        )

        # 验证DAG
        errors = dag.validate()
        if errors:
            result.state = NodeState.FAILED
            result.error = f"DAG验证失败: {'; '.join(errors)}"
            result.end_time = time.time()
            result.duration = result.end_time - start_time
            return result

        # 重置节点状态
        dag.reset_states()

        try:
            # 获取分层执行顺序
            execution_layers = dag.get_execution_order()

            # 逐层执行
            for layer in execution_layers:
                # 检查是否需要提前终止
                if self._should_terminate(result):
                    break

                # 筛选可执行的节点
                executable_nodes = []
                for node_id in layer:
                    node = dag.get_node(node_id)
                    if node is None:
                        continue
                    if node.state == NodeState.CANCELLED:
                        continue
                    # 检查依赖是否都成功
                    deps_satisfied = self._check_dependencies(
                        node, dag, result
                    )
                    if deps_satisfied:
                        node.state = NodeState.READY
                        executable_nodes.append(node)
                    else:
                        # 依赖失败，跳过此节点
                        node.state = NodeState.SKIPPED
                        result.node_results[node_id] = NodeResult(
                            node_id=node_id,
                            state=NodeState.SKIPPED,
                            error="依赖节点失败",
                        )

                if not executable_nodes:
                    continue

                # 并行执行当前层的节点
                layer_results = self._execute_layer(
                    executable_nodes, dag, context, result
                )

                # 合并结果
                for node_result in layer_results:
                    result.node_results[node_result.node_id] = node_result
                    context["_node_outputs"][node_result.node_id] = node_result.output

                    # 更新节点状态
                    node = dag.get_node(node_result.node_id)
                    if node:
                        node.state = node_result.state

                # 检查是否有失败
                failed = [r for r in layer_results if r.is_failed]
                if failed and not self._has_error_handler(dag, failed):
                    result.state = NodeState.FAILED
                    result.error = f"节点执行失败: {', '.join(r.node_id for r in failed)}"
                    break

            # 确定最终状态
            if result.state not in (NodeState.FAILED, NodeState.CANCELLED):
                all_results = list(result.node_results.values())
                if all(r.is_success for r in all_results) if all_results else True:
                    result.state = NodeState.SUCCESS
                elif any(r.is_failed for r in all_results):
                    result.state = NodeState.FAILED
                else:
                    result.state = NodeState.SUCCESS

            # 收集输出
            leaf_nodes = dag.get_leaf_nodes()
            for leaf in leaf_nodes:
                leaf_result = result.node_results.get(leaf.id)
                if leaf_result and leaf_result.output:
                    if isinstance(leaf_result.output, dict):
                        result.outputs.update(leaf_result.output)

        except Exception as e:
            result.state = NodeState.FAILED
            result.error = f"工作流执行异常: {str(e)}\n{traceback.format_exc()}"

        result.end_time = time.time()
        result.duration = result.end_time - start_time
        return result

    def execute_node(
        self,
        node: DAGNode,
        context: Dict[str, Any],
    ) -> NodeResult:
        """
        执行单个节点

        Args:
            node: DAG节点
            context: 执行上下文

        Returns:
            节点执行结果
        """
        start_time = time.time()
        node.state = NodeState.RUNNING

        # 准备输入
        node_input = self._prepare_node_input(node, context)

        # 确定超时
        timeout = node.timeout or self._default_timeout

        try:
            if self._node_executor:
                output = self._node_executor(node, node_input, context)
            else:
                output = self._default_node_executor(node, node_input, context)

            end_time = time.time()
            return NodeResult(
                node_id=node.id,
                state=NodeState.SUCCESS,
                output=output,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
            )

        except Exception as e:
            end_time = time.time()
            error_msg = f"{type(e).__name__}: {str(e)}"

            # 重试逻辑
            if node.retry_count < node.max_retries:
                node.retry_count += 1
                time.sleep(min(2 ** node.retry_count, 10))  # 指数退避
                retry_result = self.execute_node(node, context)
                retry_result.retry_count = node.retry_count
                return retry_result

            return NodeResult(
                node_id=node.id,
                state=NodeState.FAILED,
                error=error_msg,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                retry_count=node.retry_count,
            )

    def wait_for_approval(
        self,
        node_id: str,
        timeout: Optional[float] = None,
    ) -> Any:
        """
        等待人工审批

        Args:
            node_id: 等待审批的节点ID
            timeout: 超时时间

        Returns:
            审批结果
        """
        event = threading.Event()
        with self._lock:
            self._approval_events[node_id] = event

        event.wait(timeout=timeout)

        with self._lock:
            result = self._approval_results.pop(node_id, None)
            self._approval_events.pop(node_id, None)

        if result is None and timeout is not None:
            raise TimeoutError(f"审批超时: {node_id}")

        return result

    def approve(self, node_id: str, result: Any = True) -> None:
        """
        批准人工审批节点

        Args:
            node_id: 节点ID
            result: 审批结果
        """
        with self._lock:
            self._approval_results[node_id] = result
            event = self._approval_events.get(node_id)
        if event:
            event.set()

    def reject(self, node_id: str, reason: str = "") -> None:
        """
        拒绝人工审批节点

        Args:
            node_id: 节点ID
            reason: 拒绝原因
        """
        with self._lock:
            self._approval_results[node_id] = {"approved": False, "reason": reason}
            event = self._approval_events.get(node_id)
        if event:
            event.set()

    # ============================================================
    # 内部方法
    # ============================================================

    def _execute_layer(
        self,
        nodes: List[DAGNode],
        dag: DAGEngine,
        context: Dict[str, Any],
        workflow_result: WorkflowResult,
    ) -> List[NodeResult]:
        """并行执行一层的节点"""
        results: List[NodeResult] = []

        if len(nodes) == 1:
            # 单节点直接执行
            result = self._execute_with_timeout(nodes[0], context)
            results.append(result)
        else:
            # 多节点并行执行
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures: Dict[Future, DAGNode] = {}
                for node in nodes:
                    future = executor.submit(
                        self._execute_with_timeout, node, context
                    )
                    futures[future] = node

                for future in as_completed(futures):
                    node = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        results.append(NodeResult(
                            node_id=node.id,
                            state=NodeState.FAILED,
                            error=str(e),
                        ))

        return results

    def _execute_with_timeout(
        self,
        node: DAGNode,
        context: Dict[str, Any],
    ) -> NodeResult:
        """带超时的节点执行"""
        timeout = node.timeout or self._default_timeout

        if timeout is None:
            return self.execute_node(node, context)

        # 使用线程实现超时
        result_holder: List[Optional[NodeResult]] = [None]
        error_holder: List[Optional[Exception]] = [None]

        def run():
            try:
                result_holder[0] = self.execute_node(node, context)
            except Exception as e:
                error_holder[0] = e

        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            return NodeResult(
                node_id=node.id,
                state=NodeState.FAILED,
                error=f"节点执行超时 ({timeout}s)",
                start_time=time.time(),
            )

        if error_holder[0] is not None:
            return NodeResult(
                node_id=node.id,
                state=NodeState.FAILED,
                error=str(error_holder[0]),
            )

        return result_holder[0] or NodeResult(
            node_id=node.id,
            state=NodeState.FAILED,
            error="未知错误",
        )

    def _default_node_executor(
        self,
        node: DAGNode,
        node_input: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Any:
        """默认节点执行逻辑"""
        config = node.config
        node_type = node.node_type

        if node_type == "task":
            fn = config.get("function")
            if fn and callable(fn):
                return fn(node_input, context)
            return {"executed": node.id, "input": node_input}

        elif node_type == "llm":
            return {
                "type": "llm_response",
                "model": config.get("model", "default"),
                "prompt": config.get("prompt", ""),
                "input": node_input,
            }

        elif node_type == "tool":
            tool_fn = config.get("function")
            if tool_fn and callable(tool_fn):
                return tool_fn(node_input, context)
            return {"tool": node.id, "result": "no_function"}

        elif node_type == "condition":
            return self._evaluate_condition(node, node_input, context)

        elif node_type == "loop":
            return self._execute_loop(node, node_input, context)

        elif node_type == "parallel":
            return {"parallel": node.id, "completed": True}

        elif node_type == "human_approval":
            approval = self.wait_for_approval(
                node.id,
                timeout=node.timeout or self._default_timeout,
            )
            return {"approved": approval}

        elif node_type == "delay":
            delay_seconds = config.get("seconds", 0)
            time.sleep(delay_seconds)
            return {"delayed": delay_seconds}

        elif node_type == "sub_workflow":
            return {"sub_workflow": node.id, "executed": True}

        elif node_type == "error_handler":
            return {"error_handled": True}

        else:
            return {"executed": node.id, "type": node_type}

    def _evaluate_condition(
        self,
        node: DAGNode,
        node_input: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """评估条件节点"""
        config = node.config
        conditions = config.get("conditions", [])
        result = {"branch": "default"}

        for cond in conditions:
            field = cond.get("field", "")
            operator = cond.get("operator", "eq")
            value = cond.get("value")

            actual_value = node_input.get(field)
            matched = False

            if operator == "eq":
                matched = actual_value == value
            elif operator == "ne":
                matched = actual_value != value
            elif operator == "gt":
                matched = actual_value is not None and actual_value > value
            elif operator == "gte":
                matched = actual_value is not None and actual_value >= value
            elif operator == "lt":
                matched = actual_value is not None and actual_value < value
            elif operator == "lte":
                matched = actual_value is not None and actual_value <= value
            elif operator == "in":
                matched = actual_value in value
            elif operator == "contains":
                matched = value in str(actual_value) if actual_value else False
            elif operator == "exists":
                matched = field in node_input
            elif operator == "not_exists":
                matched = field not in node_input

            if matched:
                result = {"branch": cond.get("branch", "default"), "matched": True}
                break

        return result

    def _execute_loop(
        self,
        node: DAGNode,
        node_input: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """执行循环节点"""
        config = node.config
        loop_type = config.get("loop_type", "for_each")
        results = []

        if loop_type == "for_each":
            items = config.get("items", [])
            if callable(items):
                items = items(node_input, context)
            if isinstance(items, str) and items in node_input:
                items = node_input[items]

            for i, item in enumerate(items):
                loop_input = copy.deepcopy(node_input)
                loop_input["_loop_index"] = i
                loop_input["_loop_item"] = item
                results.append({"index": i, "item": item})

        elif loop_type == "while":
            max_iterations = config.get("max_iterations", 100)
            condition_fn = config.get("condition")
            if condition_fn and callable(condition_fn):
                iteration = 0
                while iteration < max_iterations:
                    if not condition_fn(node_input, context):
                        break
                    results.append({"iteration": iteration})
                    iteration += 1

        elif loop_type == "count":
            count = config.get("count", 0)
            for i in range(count):
                results.append({"iteration": i})

        return {"loop_results": results, "total": len(results)}

    def _prepare_node_input(
        self,
        node: DAGNode,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """准备节点输入（从上游节点收集）"""
        node_input = copy.deepcopy(context.get("_inputs", {}))
        node_outputs = context.get("_node_outputs", {})

        # 从依赖节点收集输出
        for dep_id in node.dependencies:
            if dep_id in node_outputs:
                dep_output = node_outputs[dep_id]
                if isinstance(dep_output, dict):
                    node_input.update(dep_output)
                else:
                    node_input[dep_id] = dep_output

        # 应用数据映射
        # 查找入边的数据映射
        # 这里简化处理，直接将上游输出合并
        return node_input

    def _check_dependencies(
        self,
        node: DAGNode,
        dag: DAGEngine,
        result: WorkflowResult,
    ) -> bool:
        """检查节点依赖是否满足"""
        for dep_id in node.dependencies:
            dep_result = result.node_results.get(dep_id)
            if dep_result is None:
                return False
            if dep_result.state not in (NodeState.SUCCESS, NodeState.SKIPPED):
                return False
        return True

    def _should_terminate(self, result: WorkflowResult) -> bool:
        """检查是否应该终止执行"""
        return result.state in (NodeState.FAILED, NodeState.CANCELLED)

    def _has_error_handler(
        self,
        dag: DAGEngine,
        failed_results: List[NodeResult],
    ) -> bool:
        """检查是否有错误处理节点"""
        for fr in failed_results:
            downstream = dag.get_downstream(fr.node_id)
            for node_id in downstream:
                node = dag.get_node(node_id)
                if node and node.node_type == "error_handler":
                    return True
        return False
