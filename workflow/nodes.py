"""
工作流节点类型

定义各种工作流节点的实现，包括任务节点、LLM节点、工具节点、
条件分支节点、循环节点、并行节点、人工审批节点等。
"""

import copy
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .graph_engine import DAGNode, NodeState


class NodeExecutionError(Exception):
    """节点执行异常"""
    pass


class NodeConfig:
    """节点配置工具"""

    @staticmethod
    def validate_config(config: Dict[str, Any], required_keys: List[str]) -> List[str]:
        """验证配置，返回缺失的必填字段"""
        return [k for k in required_keys if k not in config]


# ============================================================
# 基础节点
# ============================================================

class BaseNode(ABC):
    """
    节点基类

    所有工作流节点的抽象基类，定义统一的执行接口。
    """

    def __init__(
        self,
        node_id: str,
        name: str = "",
        config: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        max_retries: int = 0,
    ):
        self.node_id = node_id
        self.name = name or node_id
        self.config = config or {}
        self.timeout = timeout
        self.max_retries = max_retries
        self._state = NodeState.PENDING
        self._output: Any = None
        self._error: Optional[str] = None
        self._start_time: float = 0.0
        self._end_time: float = 0.0

    @abstractmethod
    def execute(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """
        执行节点

        Args:
            inputs: 节点输入数据
            context: 执行上下文

        Returns:
            节点输出
        """
        ...

    def validate(self) -> List[str]:
        """
        验证节点配置

        Returns:
            错误消息列表
        """
        return []

    def to_dag_node(self) -> DAGNode:
        """转换为DAGNode"""
        return DAGNode(
            id=self.node_id,
            name=self.name,
            node_type=self.node_type,
            config=self.config,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    @property
    def node_type(self) -> str:
        """节点类型"""
        return self.__class__.__name__.replace("Node", "").lower()

    @property
    def state(self) -> NodeState:
        return self._state

    @state.setter
    def state(self, value: NodeState) -> None:
        self._state = value

    @property
    def output(self) -> Any:
        return self._output

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def duration(self) -> float:
        if self._start_time == 0:
            return 0.0
        end = self._end_time if self._end_time > 0 else time.time()
        return end - self._start_time

    def reset(self) -> None:
        """重置节点状态"""
        self._state = NodeState.PENDING
        self._output = None
        self._error = None
        self._start_time = 0.0
        self._end_time = 0.0

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.node_id!r} state={self._state.value}>"


# ============================================================
# 任务节点
# ============================================================

class TaskNode(BaseNode):
    """
    任务节点

    执行Python函数的通用任务节点。

    Usage:
        def my_task(inputs, context):
            return {"result": inputs["value"] * 2}

        node = TaskNode("double", function=my_task)
    """

    def __init__(
        self,
        node_id: str,
        function: Optional[Callable] = None,
        name: str = "",
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(node_id, name, config, **kwargs)
        self._function = function
        if function:
            self.config["function"] = function

    def execute(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """执行任务函数"""
        self._state = NodeState.RUNNING
        self._start_time = time.time()

        try:
            fn = self._function or self.config.get("function")
            if fn is None:
                raise NodeExecutionError("任务节点没有设置执行函数")

            if not callable(fn):
                raise NodeExecutionError(f"任务函数不可调用: {type(fn)}")

            result = fn(inputs, context)
            self._output = result
            self._state = NodeState.SUCCESS
            return result

        except Exception as e:
            self._error = str(e)
            self._state = NodeState.FAILED
            raise

        finally:
            self._end_time = time.time()

    def validate(self) -> List[str]:
        errors = super().validate()
        fn = self._function or self.config.get("function")
        if fn is not None and not callable(fn):
            errors.append(f"函数不可调用: {self.node_id}")
        return errors

    @property
    def node_type(self) -> str:
        return "task"


# ============================================================
# LLM节点
# ============================================================

class LLMNode(BaseNode):
    """
    LLM调用节点

    调用大语言模型进行推理。

    Usage:
        node = LLMNode(
            "chat",
            model="gpt-4",
            prompt="请回答以下问题: {{question}}",
            temperature=0.7,
        )
    """

    def __init__(
        self,
        node_id: str,
        model: str = "default",
        prompt: str = "",
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        name: str = "",
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(node_id, name, config, **kwargs)
        self.config.update({
            "model": model,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })

    def execute(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """执行LLM调用"""
        self._state = NodeState.RUNNING
        self._start_time = time.time()

        try:
            # 模板变量替换
            prompt = self._render_template(self.config["prompt"], inputs, context)
            system_prompt = self._render_template(
                self.config.get("system_prompt", ""), inputs, context
            )

            # 检查是否有自定义LLM调用函数
            llm_fn = self.config.get("llm_function")
            if llm_fn and callable(llm_fn):
                result = llm_fn(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=self.config["model"],
                    temperature=self.config["temperature"],
                    max_tokens=self.config["max_tokens"],
                    inputs=inputs,
                    context=context,
                )
            else:
                # 默认返回模拟结果
                result = {
                    "type": "llm_response",
                    "model": self.config["model"],
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "response": f"[模拟LLM响应] prompt: {prompt[:100]}...",
                    "temperature": self.config["temperature"],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                }

            self._output = result
            self._state = NodeState.SUCCESS
            return result

        except Exception as e:
            self._error = str(e)
            self._state = NodeState.FAILED
            raise

        finally:
            self._end_time = time.time()

    def _render_template(
        self,
        template: str,
        inputs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> str:
        """渲染模板变量"""
        if not template:
            return template

        result = template
        # 替换 {{variable}} 格式的变量
        import re
        pattern = r"\{\{(\w+(?:\.\w+)*)\}\}"

        def replacer(match: Any) -> str:
            var_path = match.group(1)
            value = self._resolve_variable(var_path, inputs, context)
            return str(value) if value is not None else match.group(0)

        result = re.sub(pattern, replacer, result)
        return result

    def _resolve_variable(
        self,
        path: str,
        inputs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Any:
        """解析变量路径"""
        parts = path.split(".")
        current: Any = inputs

        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
            if current is None:
                # 尝试从context获取
                current = context.get(part)
                if current is None:
                    return None

        return current

    @property
    def node_type(self) -> str:
        return "llm"


# ============================================================
# 工具节点
# ============================================================

class ToolNode(BaseNode):
    """
    工具调用节点

    调用外部工具或函数。

    Usage:
        def search_tool(inputs, context):
            return {"results": ["item1", "item2"]}

        node = ToolNode("search", tool_name="web_search", function=search_tool)
    """

    def __init__(
        self,
        node_id: str,
        tool_name: str = "",
        function: Optional[Callable] = None,
        parameters: Optional[Dict[str, Any]] = None,
        name: str = "",
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(node_id, name, config, **kwargs)
        self.config.update({
            "tool_name": tool_name,
            "function": function,
            "parameters": parameters or {},
        })

    def execute(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """执行工具调用"""
        self._state = NodeState.RUNNING
        self._start_time = time.time()

        try:
            fn = self.config.get("function")
            if fn is None or not callable(fn):
                raise NodeExecutionError(f"工具函数未设置或不可调用: {self.node_id}")

            # 合并参数
            tool_params = copy.deepcopy(self.config.get("parameters", {}))
            tool_params.update(inputs)

            result = fn(tool_params, context)
            self._output = result
            self._state = NodeState.SUCCESS
            return result

        except Exception as e:
            self._error = str(e)
            self._state = NodeState.FAILED
            raise

        finally:
            self._end_time = time.time()

    @property
    def node_type(self) -> str:
        return "tool"


# ============================================================
# 条件分支节点
# ============================================================

class ConditionalNode(BaseNode):
    """
    条件分支节点

    根据条件选择不同的执行路径。

    Usage:
        node = ConditionalNode(
            "check_age",
            conditions=[
                {"field": "age", "operator": "gte", "value": 18, "branch": "adult"},
                {"field": "age", "operator": "lt", "value": 18, "branch": "minor"},
            ],
            default_branch="unknown",
        )
    """

    def __init__(
        self,
        node_id: str,
        conditions: Optional[List[Dict[str, Any]]] = None,
        default_branch: str = "default",
        name: str = "",
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(node_id, name, config, **kwargs)
        self.config.update({
            "conditions": conditions or [],
            "default_branch": default_branch,
        })

    def execute(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """评估条件并选择分支"""
        self._state = NodeState.RUNNING
        self._start_time = time.time()

        try:
            for condition in self.config.get("conditions", []):
                if self._evaluate_condition(condition, inputs):
                    result = {
                        "branch": condition.get("branch", "default"),
                        "matched": True,
                        "condition": condition,
                    }
                    self._output = result
                    self._state = NodeState.SUCCESS
                    return result

            # 默认分支
            result = {
                "branch": self.config.get("default_branch", "default"),
                "matched": False,
            }
            self._output = result
            self._state = NodeState.SUCCESS
            return result

        except Exception as e:
            self._error = str(e)
            self._state = NodeState.FAILED
            raise

        finally:
            self._end_time = time.time()

    def _evaluate_condition(
        self,
        condition: Dict[str, Any],
        inputs: Dict[str, Any],
    ) -> bool:
        """评估单个条件"""
        field = condition.get("field", "")
        operator = condition.get("operator", "eq")
        value = condition.get("value")

        # 支持嵌套字段路径
        parts = field.split(".")
        actual_value = inputs
        for part in parts:
            if isinstance(actual_value, dict):
                actual_value = actual_value.get(part)
            else:
                actual_value = None
                break

        if operator == "eq":
            return actual_value == value
        elif operator == "ne":
            return actual_value != value
        elif operator == "gt":
            return actual_value is not None and actual_value > value
        elif operator == "gte":
            return actual_value is not None and actual_value >= value
        elif operator == "lt":
            return actual_value is not None and actual_value < value
        elif operator == "lte":
            return actual_value is not None and actual_value <= value
        elif operator == "in":
            return actual_value in value if actual_value is not None else False
        elif operator == "not_in":
            return actual_value not in value if actual_value is not None else True
        elif operator == "contains":
            return value in str(actual_value) if actual_value is not None else False
        elif operator == "is_null":
            return actual_value is None
        elif operator == "is_not_null":
            return actual_value is not None
        elif operator == "regex":
            import re
            if actual_value is None:
                return False
            try:
                return bool(re.search(value, str(actual_value)))
            except re.error:
                return False
        elif operator == "custom":
            fn = condition.get("function")
            if fn and callable(fn):
                return fn(actual_value, inputs)
            return False

        return False

    @property
    def node_type(self) -> str:
        return "condition"


# ============================================================
# 循环节点
# ============================================================

class LoopNode(BaseNode):
    """
    循环节点

    支持for_each、while和count三种循环模式。

    Usage:
        # for_each循环
        node = LoopNode("process_items", loop_type="for_each", items_key="items")

        # while循环
        node = LoopNode(
            "retry_loop",
            loop_type="while",
            condition=lambda inputs, ctx: inputs.get("retry", False),
            max_iterations=10,
        )
    """

    def __init__(
        self,
        node_id: str,
        loop_type: str = "for_each",
        items: Optional[Any] = None,
        items_key: str = "",
        condition: Optional[Callable] = None,
        max_iterations: int = 100,
        name: str = "",
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(node_id, name, config, **kwargs)
        self.config.update({
            "loop_type": loop_type,
            "items": items,
            "items_key": items_key,
            "condition": condition,
            "max_iterations": max_iterations,
        })

    def execute(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """执行循环"""
        self._state = NodeState.RUNNING
        self._start_time = time.time()

        try:
            loop_type = self.config.get("loop_type", "for_each")
            results = []

            if loop_type == "for_each":
                results = self._for_each_loop(inputs, context)
            elif loop_type == "while":
                results = self._while_loop(inputs, context)
            elif loop_type == "count":
                results = self._count_loop(inputs, context)
            else:
                raise NodeExecutionError(f"不支持的循环类型: {loop_type}")

            output = {
                "loop_type": loop_type,
                "results": results,
                "total_iterations": len(results),
            }
            self._output = output
            self._state = NodeState.SUCCESS
            return output

        except Exception as e:
            self._error = str(e)
            self._state = NodeState.FAILED
            raise

        finally:
            self._end_time = time.time()

    def _for_each_loop(
        self, inputs: Dict[str, Any], context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """for_each循环"""
        items = self.config.get("items")
        items_key = self.config.get("items_key", "")

        if items is None and items_key:
            items = inputs.get(items_key, [])

        if callable(items):
            items = items(inputs, context)

        if not isinstance(items, (list, tuple)):
            items = [items]

        body_fn = self.config.get("body_function")
        results = []

        for i, item in enumerate(items):
            loop_input = copy.deepcopy(inputs)
            loop_input["_loop_index"] = i
            loop_input["_loop_item"] = item
            loop_input["_loop_total"] = len(items)

            if body_fn and callable(body_fn):
                result = body_fn(loop_input, context)
            else:
                result = {"index": i, "item": item}

            results.append(result)

        return results

    def _while_loop(
        self, inputs: Dict[str, Any], context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """while循环"""
        condition_fn = self.config.get("condition")
        max_iterations = self.config.get("max_iterations", 100)
        body_fn = self.config.get("body_function")

        if not condition_fn or not callable(condition_fn):
            raise NodeExecutionError("while循环需要设置condition函数")

        results = []
        iteration = 0

        while iteration < max_iterations:
            if not condition_fn(inputs, context):
                break

            loop_input = copy.deepcopy(inputs)
            loop_input["_loop_iteration"] = iteration

            if body_fn and callable(body_fn):
                result = body_fn(loop_input, context)
                if isinstance(result, dict):
                    inputs.update(result)

            results.append({"iteration": iteration})
            iteration += 1

        return results

    def _count_loop(
        self, inputs: Dict[str, Any], context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """计数循环"""
        count = self.config.get("count", 0)
        if isinstance(count, str) and count in inputs:
            count = inputs[count]

        body_fn = self.config.get("body_function")
        results = []

        for i in range(int(count)):
            loop_input = copy.deepcopy(inputs)
            loop_input["_loop_iteration"] = i

            if body_fn and callable(body_fn):
                result = body_fn(loop_input, context)
            else:
                result = {"iteration": i}

            results.append(result)

        return results

    @property
    def node_type(self) -> str:
        return "loop"


# ============================================================
# 并行节点
# ============================================================

class ParallelNode(BaseNode):
    """
    并行节点

    同时执行多个子节点。

    Usage:
        node = ParallelNode(
            "parallel_tasks",
            branches=[
                {"id": "task_a", "function": func_a},
                {"id": "task_b", "function": func_b},
            ],
            fail_fast=True,
        )
    """

    def __init__(
        self,
        node_id: str,
        branches: Optional[List[Dict[str, Any]]] = None,
        fail_fast: bool = True,
        max_workers: int = 4,
        name: str = "",
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(node_id, name, config, **kwargs)
        self.config.update({
            "branches": branches or [],
            "fail_fast": fail_fast,
            "max_workers": max_workers,
        })

    def execute(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """并行执行所有分支"""
        self._state = NodeState.RUNNING
        self._start_time = time.time()

        branches = self.config.get("branches", [])
        if not branches:
            self._output = {"results": {}}
            self._state = NodeState.SUCCESS
            return self._output

        results: Dict[str, Any] = {}
        errors: Dict[str, str] = {}
        fail_fast = self.config.get("fail_fast", True)

        def run_branch(branch: Dict[str, Any]) -> Tuple[str, Any, Optional[str]]:
            bid = branch.get("id", str(uuid.uuid4())[:8])
            fn = branch.get("function")
            try:
                if fn and callable(fn):
                    result = fn(inputs, context)
                else:
                    result = {"branch_id": bid, "executed": True}
                return (bid, result, None)
            except Exception as e:
                return (bid, None, str(e))

        threads: List[threading.Thread] = []
        result_holder: Dict[str, Tuple[Any, Optional[str]]] = {}
        lock = threading.Lock()

        for branch in branches:
            def target(b: Dict[str, Any] = branch) -> None:
                bid, result, error = run_branch(b)
                with lock:
                    result_holder[bid] = (result, error)

            t = threading.Thread(target=target)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        for bid, (result, error) in result_holder.items():
            if error:
                errors[bid] = error
                if fail_fast:
                    self._error = f"分支 {bid} 执行失败: {error}"
                    self._state = NodeState.FAILED
                    self._end_time = time.time()
                    raise NodeExecutionError(self._error)
            else:
                results[bid] = result

        self._output = {"results": results, "errors": errors}
        self._state = NodeState.SUCCESS
        return self._output

    @property
    def node_type(self) -> str:
        return "parallel"


# ============================================================
# 人工审批节点
# ============================================================

class HumanApprovalNode(BaseNode):
    """
    人工审批节点

    阻塞等待人工审批结果。

    Usage:
        node = HumanApprovalNode(
            "manager_approval",
            approval_message="请审批此请求",
            timeout=3600,
            approvers=["manager@example.com"],
        )
    """

    def __init__(
        self,
        node_id: str,
        approval_message: str = "",
        timeout: Optional[float] = None,
        approvers: Optional[List[str]] = None,
        name: str = "",
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(node_id, name, config, timeout=timeout, **kwargs)
        self.config.update({
            "approval_message": approval_message,
            "approvers": approvers or [],
        })
        self._approval_event = threading.Event()
        self._approval_result: Any = None

    def execute(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """等待审批"""
        self._state = NodeState.RUNNING
        self._start_time = time.time()
        self._approval_event.clear()
        self._approval_result = None

        try:
            # 注册审批回调到context
            if "_approval_registry" not in context:
                context["_approval_registry"] = {}
            context["_approval_registry"][self.node_id] = self

            # 等待审批
            timeout = self.timeout
            approved = self._approval_event.wait(timeout=timeout)

            if not approved:
                self._error = f"审批超时 ({timeout}s)"
                self._state = NodeState.FAILED
                raise NodeExecutionError(self._error)

            result = {
                "approved": self._approval_result,
                "message": self.config.get("approval_message", ""),
                "approvers": self.config.get("approvers", []),
            }
            self._output = result
            self._state = NodeState.SUCCESS
            return result

        except NodeExecutionError:
            raise
        except Exception as e:
            self._error = str(e)
            self._state = NodeState.FAILED
            raise

        finally:
            self._end_time = time.time()

    def approve(self, result: Any = True) -> None:
        """批准"""
        self._approval_result = result
        self._approval_event.set()

    def reject(self, reason: str = "") -> None:
        """拒绝"""
        self._approval_result = {"approved": False, "reason": reason}
        self._approval_event.set()

    @property
    def node_type(self) -> str:
        return "human_approval"


# ============================================================
# 子工作流节点
# ============================================================

class SubWorkflowNode(BaseNode):
    """
    子工作流节点

    嵌套执行另一个工作流。

    Usage:
        node = SubWorkflowNode(
            "sub_process",
            workflow_id="process_workflow",
            input_mapping={"source_field": "target_field"},
        )
    """

    def __init__(
        self,
        node_id: str,
        workflow_id: str = "",
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
        name: str = "",
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(node_id, name, config, **kwargs)
        self.config.update({
            "workflow_id": workflow_id,
            "input_mapping": input_mapping or {},
            "output_mapping": output_mapping or {},
        })

    def execute(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """执行子工作流"""
        self._state = NodeState.RUNNING
        self._start_time = time.time()

        try:
            # 应用输入映射
            sub_inputs = self._apply_mapping(
                inputs, self.config.get("input_mapping", {})
            )

            # 检查是否有自定义子工作流执行器
            executor_fn = self.config.get("executor")
            if executor_fn and callable(executor_fn):
                result = executor_fn(
                    workflow_id=self.config["workflow_id"],
                    inputs=sub_inputs,
                    context=context,
                )
            else:
                result = {
                    "sub_workflow_id": self.config["workflow_id"],
                    "inputs": sub_inputs,
                    "status": "completed",
                }

            # 应用输出映射
            if isinstance(result, dict):
                mapped_output = self._apply_mapping(
                    result, self.config.get("output_mapping", {})
                )
                result.update(mapped_output)

            self._output = result
            self._state = NodeState.SUCCESS
            return result

        except Exception as e:
            self._error = str(e)
            self._state = NodeState.FAILED
            raise

        finally:
            self._end_time = time.time()

    def _apply_mapping(
        self, data: Dict[str, Any], mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """应用字段映射"""
        if not mapping:
            return {}
        result = {}
        for source_key, target_key in mapping.items():
            if source_key in data:
                result[target_key] = data[source_key]
        return result

    @property
    def node_type(self) -> str:
        return "sub_workflow"


# ============================================================
# 错误处理节点
# ============================================================

class ErrorHandlerNode(BaseNode):
    """
    错误处理节点

    捕获上游节点的错误并执行恢复逻辑。

    Usage:
        node = ErrorHandlerNode(
            "error_recovery",
            handler_function=lambda error, inputs, ctx: {"recovered": True},
            retry_on_error=True,
            max_retries=3,
        )
    """

    def __init__(
        self,
        node_id: str,
        handler_function: Optional[Callable] = None,
        retry_on_error: bool = False,
        max_retries: int = 3,
        error_types: Optional[List[str]] = None,
        name: str = "",
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(node_id, name, config, max_retries=max_retries, **kwargs)
        self.config.update({
            "handler_function": handler_function,
            "retry_on_error": retry_on_error,
            "max_retries": max_retries,
            "error_types": error_types or [],
        })

    def execute(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """执行错误处理"""
        self._state = NodeState.RUNNING
        self._start_time = time.time()

        try:
            # 获取上游错误
            error_info = inputs.get("_error", inputs.get("error", ""))
            handler_fn = self.config.get("handler_function")

            if handler_fn and callable(handler_fn):
                result = handler_fn(error_info, inputs, context)
            else:
                result = {
                    "error_handled": True,
                    "error": error_info,
                    "action": "logged",
                }

            self._output = result
            self._state = NodeState.SUCCESS
            return result

        except Exception as e:
            self._error = f"错误处理失败: {str(e)}"
            self._state = NodeState.FAILED
            raise

        finally:
            self._end_time = time.time()

    @property
    def node_type(self) -> str:
        return "error_handler"


# ============================================================
# 延迟节点
# ============================================================

class DelayNode(BaseNode):
    """
    延迟节点

    暂停执行指定的时间。

    Usage:
        node = DelayNode("wait", seconds=5)
        node = DelayNode("wait_until", until=lambda: some_condition())
    """

    def __init__(
        self,
        node_id: str,
        seconds: float = 0.0,
        until: Optional[Callable[[], bool]] = None,
        check_interval: float = 1.0,
        name: str = "",
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(node_id, name, config, **kwargs)
        self.config.update({
            "seconds": seconds,
            "until": until,
            "check_interval": check_interval,
        })

    def execute(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """执行延迟"""
        self._state = NodeState.RUNNING
        self._start_time = time.time()

        try:
            until_fn = self.config.get("until")
            if until_fn and callable(until_fn):
                # 等待条件满足
                check_interval = self.config.get("check_interval", 1.0)
                waited = 0.0
                while not until_fn():
                    time.sleep(check_interval)
                    waited += check_interval
                    if self.timeout and waited >= self.timeout:
                        raise NodeExecutionError(
                            f"延迟节点超时 ({self.timeout}s)"
                        )
                result = {"delayed": waited, "type": "until"}
            else:
                # 固定延迟
                seconds = self.config.get("seconds", 0)
                time.sleep(seconds)
                result = {"delayed": seconds, "type": "fixed"}

            self._output = result
            self._state = NodeState.SUCCESS
            return result

        except NodeExecutionError:
            self._error = "延迟超时"
            self._state = NodeState.FAILED
            raise

        except Exception as e:
            self._error = str(e)
            self._state = NodeState.FAILED
            raise

        finally:
            self._end_time = time.time()

    @property
    def node_type(self) -> str:
        return "delay"
