"""
LangChain 生态系统适配器模块

提供工具转换、链集成、代理兼容性、回调处理和记忆集成功能。
仅使用 Python 标准库。
"""

import abc
import json
import re
import time
import uuid
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from .base import Tool, ToolResult, ToolParameter


# ---------------------------------------------------------------------------
# 类型定义和协议
# ---------------------------------------------------------------------------
class LangChainToolLike(Protocol):
    """LangChain 工具协议"""
    name: str
    description: str

    def _run(self, *args: Any, **kwargs: Any) -> str:
        ...

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        ...


class LangChainCallback(Protocol):
    """LangChain 回调协议"""

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        ...

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        ...

    def on_tool_error(self, error: Exception, **kwargs: Any) -> Any:
        ...


class LangChainMemory(Protocol):
    """LangChain 记忆协议"""

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def save_context(
        self, inputs: Dict[str, Any], outputs: Dict[str, Any]
    ) -> None:
        ...


@dataclass
class ChainStep:
    """链步骤

    Attributes:
        tool_name: 工具名称
        params: 参数
        result: 结果
        timestamp: 时间戳
        duration_ms: 执行耗时
    """
    tool_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    result: Optional[ToolResult] = None
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0


@dataclass
class AgentAction:
    """代理动作

    Attributes:
        tool: 工具名称
        tool_input: 工具输入
        log: 思考日志
    """
    tool: str
    tool_input: Union[str, Dict[str, Any]]
    log: str = ""


@dataclass
class AgentFinish:
    """代理完成

    Attributes:
        return_values: 返回值
        log: 思考日志
    """
    return_values: Dict[str, Any] = field(default_factory=dict)
    log: str = ""


# ---------------------------------------------------------------------------
# ToolConverter - 工具转换器
# ---------------------------------------------------------------------------
class ToolConverter:
    """工具转换器

    在框架工具和 LangChain 格式之间进行转换。
    """

    def __init__(self):
        self._conversion_cache: Dict[str, Dict[str, Any]] = {}

    def to_langchain_format(self, tool: Tool) -> Dict[str, Any]:
        """将框架工具转换为 LangChain 格式

        Args:
            tool: 框架工具实例

        Returns:
            LangChain 格式的工具定义
        """
        cache_key = f"{tool.name}:{tool.version}"
        if cache_key in self._conversion_cache:
            return self._conversion_cache[cache_key]

        lc_tool = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters_schema,
        }

        # 添加函数模式（OpenAI 函数调用格式）
        lc_tool["function"] = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters_schema,
        }

        self._conversion_cache[cache_key] = lc_tool
        return lc_tool

    def to_langchain_tools_list(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """转换工具列表为 LangChain 格式"""
        return [self.to_langchain_format(t) for t in tools]

    def from_langchain_format(
        self, lc_tool: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """从 LangChain 格式转换为框架格式

        Args:
            lc_tool: LangChain 工具定义

        Returns:
            框架格式的工具定义
        """
        name = lc_tool.get("name", "")
        description = lc_tool.get("description", "")
        parameters = lc_tool.get("parameters", {})

        # 提取参数定义
        params = self._extract_parameters(parameters)

        return {
            "name": name,
            "description": description,
            "parameters": params,
            "source": "langchain",
        }

    def convert_input_to_params(
        self, tool_input: Union[str, Dict[str, Any]], tool: Tool
    ) -> Dict[str, Any]:
        """将工具输入转换为参数字典

        LangChain 可能传递字符串或字典，需要统一处理。
        """
        if isinstance(tool_input, dict):
            return tool_input

        if isinstance(tool_input, str):
            # 尝试解析 JSON
            try:
                parsed = json.loads(tool_input)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

            # 使用自然语言解析
            return self._parse_natural_language(tool_input, tool)

        return {"input": tool_input}

    def convert_result_to_str(self, result: ToolResult) -> str:
        """将工具结果转换为字符串（LangChain 期望）"""
        if not result.success:
            return f"Error: {result.error}"

        if isinstance(result.data, str):
            return result.data

        try:
            return json.dumps(result.data, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(result.data)

    def _extract_parameters(
        self, schema: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """从 JSON Schema 提取参数定义"""
        params: List[Dict[str, Any]] = []
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        for name, prop in properties.items():
            param = {
                "name": name,
                "type": prop.get("type", "string"),
                "required": name in required,
                "description": prop.get("description", ""),
                "default": prop.get("default"),
            }

            # 复制其他约束
            for key in ["enum", "minimum", "maximum", "minLength", "maxLength", "pattern"]:
                if key in prop:
                    param[key.lower()] = prop[key]

            params.append(param)

        return params

    def _parse_natural_language(
        self, text: str, tool: Tool
    ) -> Dict[str, Any]:
        """从自然语言文本解析参数"""
        params: Dict[str, Any] = {}
        schema = tool.parameters_schema
        properties = schema.get("properties", {})

        # 简单的键值对提取
        for param_name in properties.keys():
            # 查找 "param_name: value" 或 "param_name is value" 模式
            patterns = [
                rf"{param_name}[：:]\s*([^,;\n]+)",
                rf"{param_name}\s+is\s+([^,;\n]+)",
                rf"{param_name}\s*=\s*([^,;\n]+)",
            ]

            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    # 类型转换
                    param_type = properties[param_name].get("type", "string")
                    params[param_name] = self._convert_value(value, param_type)
                    break

        # 如果没有提取到任何参数，将整个文本作为 "input"
        if not params:
            params["input"] = text

        return params

    def _convert_value(self, value: str, target_type: str) -> Any:
        """转换值为目标类型"""
        if target_type == "integer":
            try:
                return int(value)
            except ValueError:
                return value
        elif target_type == "number":
            try:
                return float(value)
            except ValueError:
                return value
        elif target_type == "boolean":
            return value.lower() in ("true", "yes", "1", "on")
        elif target_type == "array":
            # 尝试解析 JSON 数组
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
            # 按逗号分割
            return [v.strip() for v in value.split(",")]
        elif target_type == "object":
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {"value": value}

        return value


# ---------------------------------------------------------------------------
# CallbackHandler - 回调处理器
# ---------------------------------------------------------------------------
class CallbackHandler:
    """回调处理器

    管理 LangChain 风格的回调，支持同步和异步回调。
    """

    def __init__(self):
        self._callbacks: List[LangChainCallback] = []
        self._parent_ids: Dict[str, str] = {}

    def add_callback(self, callback: LangChainCallback) -> None:
        """添加回调"""
        self._callbacks.append(callback)

    def remove_callback(self, callback: LangChainCallback) -> None:
        """移除回调"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def on_tool_start(
        self,
        tool: Tool,
        params: Dict[str, Any],
        run_id: Optional[str] = None,
        parent_run_id: Optional[str] = None,
    ) -> str:
        """工具开始回调"""
        run_id = run_id or uuid.uuid4().hex

        if parent_run_id:
            self._parent_ids[run_id] = parent_run_id

        serialized = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters_schema,
        }
        input_str = json.dumps(params, ensure_ascii=False)

        for callback in self._callbacks:
            try:
                callback.on_tool_start(
                    serialized=serialized,
                    input_str=input_str,
                    run_id=run_id,
                    parent_run_id=parent_run_id,
                )
            except Exception as e:
                warnings.warn(f"回调错误: {e}")

        return run_id

    def on_tool_end(
        self,
        result: ToolResult,
        run_id: str,
    ) -> None:
        """工具结束回调"""
        output = result.data if result.success else f"Error: {result.error}"
        if not isinstance(output, str):
            try:
                output = json.dumps(output, ensure_ascii=False)
            except (TypeError, ValueError):
                output = str(output)

        for callback in self._callbacks:
            try:
                callback.on_tool_end(output=output, run_id=run_id)
            except Exception as e:
                warnings.warn(f"回调错误: {e}")

        self._parent_ids.pop(run_id, None)

    def on_tool_error(
        self,
        error: Exception,
        run_id: str,
    ) -> None:
        """工具错误回调"""
        for callback in self._callbacks:
            try:
                callback.on_tool_error(error=error, run_id=run_id)
            except Exception as e:
                warnings.warn(f"回调错误: {e}")

        self._parent_ids.pop(run_id, None)


# ---------------------------------------------------------------------------
# MemoryIntegrator - 记忆集成器
# ---------------------------------------------------------------------------
class MemoryIntegrator:
    """记忆集成器

    集成 LangChain 风格的记忆功能。
    """

    def __init__(self):
        self._memories: Dict[str, LangChainMemory] = {}
        self._session_memories: Dict[str, Dict[str, Any]] = {}

    def register_memory(self, name: str, memory: LangChainMemory) -> None:
        """注册记忆组件"""
        self._memories[name] = memory

    def get_memory(self, name: str) -> Optional[LangChainMemory]:
        """获取记忆组件"""
        return self._memories.get(name)

    def load_context(
        self, inputs: Dict[str, Any], memory_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """加载记忆上下文"""
        context: Dict[str, Any] = {}

        names = memory_names or list(self._memories.keys())
        for name in names:
            if name in self._memories:
                try:
                    memory_vars = self._memories[name].load_memory_variables(inputs)
                    context.update(memory_vars)
                except Exception as e:
                    warnings.warn(f"加载记忆失败 {name}: {e}")

        return context

    def save_context(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        memory_names: Optional[List[str]] = None,
    ) -> None:
        """保存上下文到记忆"""
        names = memory_names or list(self._memories.keys())
        for name in names:
            if name in self._memories:
                try:
                    self._memories[name].save_context(inputs, outputs)
                except Exception as e:
                    warnings.warn(f"保存记忆失败 {name}: {e}")

    def create_session_memory(self, session_id: str) -> Dict[str, Any]:
        """创建会话记忆"""
        memory = {
            "session_id": session_id,
            "history": [],
            "variables": {},
        }
        self._session_memories[session_id] = memory
        return memory

    def get_session_memory(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话记忆"""
        return self._session_memories.get(session_id)

    def add_to_session_history(
        self, session_id: str, role: str, content: str
    ) -> None:
        """添加到会话历史"""
        if session_id not in self._session_memories:
            self.create_session_memory(session_id)

        self._session_memories[session_id]["history"].append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
        })

    def clear_session(self, session_id: str) -> None:
        """清除会话记忆"""
        self._session_memories.pop(session_id, None)


# ---------------------------------------------------------------------------
# ChainIntegrator - 链集成器
# ---------------------------------------------------------------------------
class ChainIntegrator:
    """链集成器

    支持顺序链、分支链和条件链的执行。
    """

    def __init__(
        self,
        converter: Optional[ToolConverter] = None,
        callback_handler: Optional[CallbackHandler] = None,
    ):
        self.converter = converter or ToolConverter()
        self.callback_handler = callback_handler or CallbackHandler()
        self._chains: Dict[str, List[Tool]] = {}
        self._execution_history: List[ChainStep] = []

    def define_chain(self, name: str, tools: List[Tool]) -> None:
        """定义工具链"""
        self._chains[name] = tools

    def run_chain(
        self,
        chain_name: str,
        initial_input: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Iterator[ChainStep]:
        """运行工具链

        Args:
            chain_name: 链名称
            initial_input: 初始输入
            context: 上下文变量

        Yields:
            链步骤
        """
        if chain_name not in self._chains:
            raise ValueError(f"链未定义: {chain_name}")

        tools = self._chains[chain_name]
        current_input = dict(initial_input)
        ctx = dict(context or {})

        for tool in tools:
            # 合并上下文到输入
            merged_input = {**ctx, **current_input}

            # 执行工具
            start_time = time.time()
            run_id = self.callback_handler.on_tool_start(tool, merged_input)

            try:
                result = tool.execute(merged_input)
                self.callback_handler.on_tool_end(result, run_id)
            except Exception as e:
                self.callback_handler.on_tool_error(e, run_id)
                result = ToolResult.fail(str(e), tool_name=tool.name)

            duration = (time.time() - start_time) * 1000

            step = ChainStep(
                tool_name=tool.name,
                params=merged_input,
                result=result,
                duration_ms=duration,
            )
            self._execution_history.append(step)
            yield step

            # 更新输入用于下一步
            if result.success:
                if isinstance(result.data, dict):
                    current_input.update(result.data)
                ctx[f"{tool.name}_output"] = result.data

    def run_sequential(
        self, tools: List[Tool], initial_input: Dict[str, Any]
    ) -> List[ChainStep]:
        """顺序执行工具列表"""
        steps: List[ChainStep] = []
        current_input = dict(initial_input)

        for tool in tools:
            merged_input = {**current_input}

            start_time = time.time()
            run_id = self.callback_handler.on_tool_start(tool, merged_input)

            try:
                result = tool.execute(merged_input)
                self.callback_handler.on_tool_end(result, run_id)
            except Exception as e:
                self.callback_handler.on_tool_error(e, run_id)
                result = ToolResult.fail(str(e), tool_name=tool.name)

            duration = (time.time() - start_time) * 1000

            step = ChainStep(
                tool_name=tool.name,
                params=merged_input,
                result=result,
                duration_ms=duration,
            )
            steps.append(step)
            self._execution_history.append(step)

            if result.success and isinstance(result.data, dict):
                current_input.update(result.data)

        return steps

    def get_execution_history(self) -> List[ChainStep]:
        """获取执行历史"""
        return list(self._execution_history)

    def clear_history(self) -> None:
        """清除执行历史"""
        self._execution_history.clear()


# ---------------------------------------------------------------------------
# AgentCompatibility - 代理兼容性
# ---------------------------------------------------------------------------
class AgentCompatibility:
    """代理兼容性

    提供与 LangChain 代理系统的兼容层。
    """

    def __init__(
        self,
        converter: Optional[ToolConverter] = None,
        callback_handler: Optional[CallbackHandler] = None,
    ):
        self.converter = converter or ToolConverter()
        self.callback_handler = callback_handler or CallbackHandler()
        self._tools: Dict[str, Tool] = {}
        self._action_history: List[AgentAction] = []
        self._max_iterations: int = 10

    def register_tools(self, tools: List[Tool]) -> None:
        """注册可用工具"""
        for tool in tools:
            self._tools[tool.name] = tool

    def get_tool_descriptions(self) -> List[Dict[str, Any]]:
        """获取工具描述列表（用于提示词）"""
        return [
            self.converter.to_langchain_format(tool)
            for tool in self._tools.values()
        ]

    def format_tools_for_prompt(self) -> str:
        """格式化工具为提示词文本"""
        lines = ["Available tools:"]
        for tool in self._tools.values():
            lines.append(f"\n{tool.name}:")
            lines.append(f"  Description: {tool.description}")
            lines.append(f"  Parameters: {json.dumps(tool.parameters_schema, indent=2)}")
        return "\n".join(lines)

    def execute_action(self, action: AgentAction) -> ToolResult:
        """执行代理动作"""
        if action.tool not in self._tools:
            return ToolResult.fail(f"未知工具: {action.tool}")

        tool = self._tools[action.tool]
        params = self.converter.convert_input_to_params(action.tool_input, tool)

        run_id = self.callback_handler.on_tool_start(tool, params)

        try:
            result = tool.execute(params)
            self.callback_handler.on_tool_end(result, run_id)
        except Exception as e:
            self.callback_handler.on_tool_error(e, run_id)
            result = ToolResult.fail(str(e), tool_name=tool.name)

        self._action_history.append(action)
        return result

    def parse_action(self, text: str) -> Union[AgentAction, AgentFinish]:
        """从文本解析代理动作

        支持多种动作格式：
        - Action: tool_name\nAction Input: {...}
        - ```json\n{"action": "...", "action_input": ...}\n```
        """
        # 尝试 JSON 格式
        json_match = re.search(
            r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text
        )
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if "action" in data:
                    if data["action"].lower() == "final_answer":
                        return AgentFinish(
                            return_values={"output": data.get("action_input", "")},
                            log=text,
                        )
                    return AgentAction(
                        tool=data["action"],
                        tool_input=data.get("action_input", {}),
                        log=text,
                    )
            except json.JSONDecodeError:
                pass

        # 尝试 Action/Action Input 格式
        action_match = re.search(
            r'Action:\s*(\w+)\s*\n+Action Input:\s*([\s\S]*?)(?=\n+Observation:|$)',
            text,
            re.IGNORECASE,
        )
        if action_match:
            tool_name = action_match.group(1).strip()
            tool_input = action_match.group(2).strip()

            # 尝试解析 JSON 输入
            try:
                parsed_input = json.loads(tool_input)
            except json.JSONDecodeError:
                parsed_input = tool_input

            return AgentAction(
                tool=tool_name,
                tool_input=parsed_input,
                log=text,
            )

        # 检查是否是最终答案
        final_match = re.search(
            r'Final Answer:\s*([\s\S]+)$', text, re.IGNORECASE
        )
        if final_match:
            return AgentFinish(
                return_values={"output": final_match.group(1).strip()},
                log=text,
            )

        # 默认返回完成
        return AgentFinish(return_values={"output": text}, log=text)

    def format_observation(self, result: ToolResult) -> str:
        """格式化观察结果"""
        if result.success:
            output = self.converter.convert_result_to_str(result)
            return f"Observation: {output}"
        else:
            return f"Observation: Error - {result.error}"

    def should_continue(self, iteration: int) -> bool:
        """检查是否应该继续迭代"""
        return iteration < self._max_iterations

    def get_action_history(self) -> List[AgentAction]:
        """获取动作历史"""
        return list(self._action_history)

    def clear_history(self) -> None:
        """清除历史"""
        self._action_history.clear()


# ---------------------------------------------------------------------------
# LangChainAdapter - 主适配器类
# ---------------------------------------------------------------------------
class LangChainAdapter:
    """LangChain 主适配器

    整合所有 LangChain 兼容功能。
    """

    def __init__(self):
        self.converter = ToolConverter()
        self.callback_handler = CallbackHandler()
        self.memory_integrator = MemoryIntegrator()
        self.chain_integrator = ChainIntegrator(
            self.converter, self.callback_handler
        )
        self.agent_compat = AgentCompatibility(
            self.converter, self.callback_handler
        )

    def adapt_tool(self, tool: Tool) -> Dict[str, Any]:
        """适配单个工具"""
        return self.converter.to_langchain_format(tool)

    def adapt_tools(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """适配工具列表"""
        return self.converter.to_langchain_tools_list(tools)

    def create_agent_interface(
        self, tools: List[Tool]
    ) -> AgentCompatibility:
        """创建代理接口"""
        self.agent_compat.register_tools(tools)
        return self.agent_compat

    def create_chain_interface(self) -> ChainIntegrator:
        """创建链接口"""
        return self.chain_integrator

    def create_memory_interface(self) -> MemoryIntegrator:
        """创建记忆接口"""
        return self.memory_integrator

    def wrap_for_langchain(self, tool: Tool) -> Callable[..., str]:
        """包装工具为 LangChain 可调用格式

        Returns:
            符合 LangChain 工具签名的可调用对象
        """
        def _run(*args: Any, **kwargs: Any) -> str:
            # 处理位置参数
            if args:
                if len(args) == 1 and isinstance(args[0], str):
                    params = self.converter.convert_input_to_params(args[0], tool)
                else:
                    params = {"args": list(args)}
            else:
                params = kwargs

            run_id = self.callback_handler.on_tool_start(tool, params)

            try:
                result = tool.execute(params)
                self.callback_handler.on_tool_end(result, run_id)
                return self.converter.convert_result_to_str(result)
            except Exception as e:
                self.callback_handler.on_tool_error(e, run_id)
                return f"Error: {e}"

        # 附加元数据
        _run.name = tool.name  # type: ignore
        _run.description = tool.description  # type: ignore
        _run.args_schema = tool.parameters_schema  # type: ignore

        return _run


__all__ = [
    "LangChainAdapter",
    "ToolConverter",
    "ChainIntegrator",
    "AgentCompatibility",
    "CallbackHandler",
    "MemoryIntegrator",
    "ChainStep",
    "AgentAction",
    "AgentFinish",
    "LangChainToolLike",
    "LangChainCallback",
    "LangChainMemory",
]
