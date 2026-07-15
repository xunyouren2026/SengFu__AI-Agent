"""
LLM 推理节点模块

提供高级 LLM 推理功能，包括提示模板渲染、模型选择、
流式响应处理、Token 计数、成本跟踪、回退模型支持和输出解析。

Classes:
    LLMNode: LLM 推理节点主类
    PromptRenderer: 提示模板渲染器
    ModelSelector: 模型选择器
    StreamingHandler: 流式响应处理器
    TokenCounter: Token 计数器
    CostTracker: 成本跟踪器
    FallbackHandler: 回退处理器
    OutputParser: 输出解析器
    LLMNodeConfig: LLM 节点配置
"""

import json
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple, Union


# ============================================================
# 异常类
# ============================================================

class LLMError(Exception):
    """LLM 推理异常"""
    def __init__(
        self,
        model: str = "",
        message: str = "",
        cause: Optional[Exception] = None,
    ) -> None:
        self.model = model
        self.cause = cause
        super().__init__(f"LLM '{model}' 错误: {message}")


class LLMTimeoutError(LLMError):
    """LLM 超时异常"""
    def __init__(
        self,
        model: str = "",
        timeout_seconds: float = 0.0,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        super().__init__(model, f"推理超时 ({timeout_seconds}s)")


class LLMRateLimitError(LLMError):
    """LLM 速率限制异常"""
    def __init__(
        self,
        model: str = "",
        retry_after: float = 0.0,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(
            model,
            f"速率限制，建议 {retry_after}s 后重试",
        )


class PromptRenderingError(LLMError):
    """提示渲染异常"""
    def __init__(self, message: str = "") -> None:
        super().__init__("", f"提示渲染失败: {message}")


# ============================================================
# LLM 节点配置
# ============================================================

@dataclass
class ModelPricing:
    """模型定价"""
    input_price_per_1k: float = 0.0    # 每 1000 输入 token 的价格
    output_price_per_1k: float = 0.0   # 每 1000 输出 token 的价格
    currency: str = "USD"


@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    provider: str = ""
    max_input_tokens: int = 4096
    max_output_tokens: int = 2048
    pricing: ModelPricing = dataclass_field(default_factory=ModelPricing)
    supports_streaming: bool = True
    priority: int = 0  # 用于回退排序，数值越高优先级越高


class LLMNodeConfig:
    """
    LLM 节点配置

    Attributes:
        default_model: 默认模型名称
        temperature: 采样温度
        top_p: 核采样概率
        max_tokens: 最大输出 token 数
        timeout: 请求超时时间
        enable_streaming: 是否启用流式响应
        enable_cost_tracking: 是否启用成本跟踪
        enable_token_counting: 是否启用 token 计数
        fallback_models: 回退模型列表
        max_retries: 最大重试次数
        retry_delay: 重试延迟
    """

    def __init__(
        self,
        default_model: str = "default",
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        timeout: float = 60.0,
        enable_streaming: bool = False,
        enable_cost_tracking: bool = True,
        enable_token_counting: bool = True,
        fallback_models: Optional[List[str]] = None,
        max_retries: int = 2,
        retry_delay: float = 1.0,
    ) -> None:
        self.default_model = default_model
        self.temperature = max(0.0, min(2.0, temperature))
        self.top_p = max(0.0, min(1.0, top_p))
        self.max_tokens = max(1, max_tokens)
        self.timeout = max(1.0, timeout)
        self.enable_streaming = enable_streaming
        self.enable_cost_tracking = enable_cost_tracking
        self.enable_token_counting = enable_token_counting
        self.fallback_models = fallback_models or []
        self.max_retries = max(0, max_retries)
        self.retry_delay = max(0.0, retry_delay)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "default_model": self.default_model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "enable_streaming": self.enable_streaming,
            "enable_cost_tracking": self.enable_cost_tracking,
            "enable_token_counting": self.enable_token_counting,
            "fallback_models": self.fallback_models,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
        }


# ============================================================
# 提示模板渲染器
# ============================================================

class PromptRenderer:
    """
    提示模板渲染器

    支持 Jinja2 风格的变量替换和多种模板语法。

    Usage:
        renderer = PromptRenderer()
        prompt = renderer.render(
            "你好 {{name}}，今天是 {{day}}",
            {"name": "张三", "day": "周一"},
        )
    """

    # 匹配 {{variable}} 和 {{variable.path}} 格式
    _VARIABLE_PATTERN = re.compile(r"\{\{(\w+(?:\.\w+)*)\}\}")

    # 匹配 {% if condition %}...{% endif %} 块
    _IF_PATTERN = re.compile(
        r"\{%\s*if\s+(\w+(?:\.\w+)*)\s*%\}(.*?)\{%\s*endif\s*%\}",
        re.DOTALL,
    )

    # 匹配 {% for item in list %}...{% endfor %} 块
    _FOR_PATTERN = re.compile(
        r"\{%\s*for\s+(\w+)\s+in\s+(\w+(?:\.\w+)*)\s*%\}(.*?)\{%\s*endfor\s*%\}",
        re.DOTALL,
    )

    def render(
        self,
        template: str,
        variables: Dict[str, Any],
    ) -> str:
        """
        渲染模板

        Args:
            template: 模板字符串
            variables: 变量字典

        Returns:
            渲染后的字符串
        """
        if not template:
            return template

        result = template

        # 处理 for 循环
        result = self._render_for_blocks(result, variables)

        # 处理 if 条件
        result = self._render_if_blocks(result, variables)

        # 处理变量替换
        result = self._render_variables(result, variables)

        return result

    def _render_variables(
        self,
        template: str,
        variables: Dict[str, Any],
    ) -> str:
        """替换 {{variable}} 格式的变量"""
        def replacer(match: re.Match) -> str:
            var_path = match.group(1)
            value = self._resolve_path(var_path, variables)
            return str(value) if value is not None else match.group(0)

        return self._VARIABLE_PATTERN.sub(replacer, template)

    def _render_if_blocks(
        self,
        template: str,
        variables: Dict[str, Any],
    ) -> str:
        """处理 {% if %} 条件块"""
        def if_replacer(match: re.Match) -> str:
            var_path = match.group(1)
            content = match.group(2)
            value = self._resolve_path(var_path, variables)
            if value:
                # 递归渲染内容
                return self.render(content, variables)
            return ""

        return self._IF_PATTERN.sub(if_replacer, template)

    def _render_for_blocks(
        self,
        template: str,
        variables: Dict[str, Any],
    ) -> str:
        """处理 {% for %} 循环块"""
        def for_replacer(match: re.Match) -> str:
            item_var = match.group(1)
            list_path = match.group(2)
            content = match.group(3)

            items = self._resolve_path(list_path, variables)
            if not isinstance(items, (list, tuple)):
                return ""

            parts: List[str] = []
            for i, item in enumerate(items):
                loop_vars = dict(variables)
                loop_vars[item_var] = item
                loop_vars[f"{item_var}_index"] = i
                loop_vars[f"{item_var}_first"] = i == 0
                loop_vars[f"{item_var}_last"] = i == len(items) - 1
                parts.append(self.render(content, loop_vars))

            return "".join(parts)

        return self._FOR_PATTERN.sub(for_replacer, template)

    def _resolve_path(
        self,
        path: str,
        variables: Dict[str, Any],
    ) -> Any:
        """解析点分隔的变量路径"""
        parts = path.split(".")
        current: Any = variables
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return None
            if current is None:
                return None
        return current

    def build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        variables: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """
        构建消息列表

        Args:
            system_prompt: 系统提示模板
            user_prompt: 用户提示模板
            variables: 变量字典
            conversation_history: 对话历史

        Returns:
            消息列表
        """
        messages: List[Dict[str, str]] = []

        if system_prompt:
            rendered_system = self.render(system_prompt, variables)
            messages.append({"role": "system", "content": rendered_system})

        if conversation_history:
            messages.extend(conversation_history)

        rendered_user = self.render(user_prompt, variables)
        messages.append({"role": "user", "content": rendered_user})

        return messages


# ============================================================
# 模型选择器
# ============================================================

class ModelSelector:
    """
    模型选择器

    根据需求选择合适的 LLM 模型，支持优先级排序和约束过滤。

    Usage:
        selector = ModelSelector()
        selector.register(ModelInfo(name="gpt-4", priority=10))
        selector.register(ModelInfo(name="gpt-3.5", priority=5))
        model = selector.select(preferred="gpt-4")
    """

    def __init__(self) -> None:
        self._models: Dict[str, ModelInfo] = {}
        self._lock = threading.Lock()

    def register(self, model_info: ModelInfo) -> None:
        """注册模型"""
        with self._lock:
            self._models[model_info.name] = model_info

    def unregister(self, name: str) -> bool:
        """注销模型"""
        with self._lock:
            return self._models.pop(name, None) is not None

    def select(
        self,
        preferred: str = "",
        min_output_tokens: int = 0,
        require_streaming: bool = False,
        exclude: Optional[List[str]] = None,
    ) -> Optional[ModelInfo]:
        """
        选择模型

        Args:
            preferred: 首选模型名称
            min_output_tokens: 最小输出 token 需求
            require_streaming: 是否需要流式支持
            exclude: 排除的模型列表

        Returns:
            选中的模型信息，无可用模型返回 None
        """
        exclude_set = set(exclude or [])

        # 首先尝试首选模型
        if preferred and preferred not in exclude_set:
            model = self._models.get(preferred)
            if model is not None:
                if self._meets_requirements(
                    model, min_output_tokens, require_streaming
                ):
                    return model

        # 按优先级排序选择
        candidates = [
            m for m in self._models.values()
            if m.name not in exclude_set
            and self._meets_requirements(m, min_output_tokens, require_streaming)
        ]
        candidates.sort(key=lambda m: -m.priority)

        return candidates[0] if candidates else None

    def _meets_requirements(
        self,
        model: ModelInfo,
        min_output_tokens: int,
        require_streaming: bool,
    ) -> bool:
        """检查模型是否满足需求"""
        if min_output_tokens > 0 and model.max_output_tokens < min_output_tokens:
            return False
        if require_streaming and not model.supports_streaming:
            return False
        return True

    def list_models(self) -> List[ModelInfo]:
        """列出所有已注册模型"""
        with self._lock:
            return list(self._models.values())

    def get_model(self, name: str) -> Optional[ModelInfo]:
        """获取指定模型"""
        with self._lock:
            return self._models.get(name)


# ============================================================
# Token 计数器
# ============================================================

class TokenCounter:
    """
    Token 计数器

    估算文本的 token 数量。使用基于字符的启发式算法，
    不依赖外部 tokenizer。

    Usage:
        counter = TokenCounter()
        count = counter.count("Hello, world!")
    """

    # 不同语言的大致字符/token 比率
    _CHARS_PER_TOKEN: Dict[str, float] = {
        "en": 4.0,     # 英语约 4 字符/token
        "zh": 1.5,     # 中文约 1.5 字符/token
        "ja": 1.5,     # 日语约 1.5 字符/token
        "ko": 2.0,     # 韩语约 2 字符/token
        "code": 3.5,   # 代码约 3.5 字符/token
        "mixed": 2.5,  # 混合内容
    }

    def __init__(self, language: str = "mixed") -> None:
        self.language = language
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._lock = threading.Lock()

    def count(self, text: str) -> int:
        """
        估算文本的 token 数量

        Args:
            text: 输入文本

        Returns:
            估算的 token 数
        """
        if not text:
            return 0

        # 检测是否主要是代码
        if self._is_code(text):
            chars_per_token = self._CHARS_PER_TOKEN.get("code", 3.5)
        else:
            chars_per_token = self._CHARS_PER_TOKEN.get(
                self.language, 2.5
            )

        # 基础估算
        token_count = max(1, int(len(text) / chars_per_token))

        # 对特殊字符进行微调
        token_count += self._count_special_tokens(text)

        return token_count

    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """
        计算消息列表的总 token 数

        Args:
            messages: 消息列表

        Returns:
            总 token 数
        """
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")
            # 每条消息有约 4 token 的格式开销
            total += 4
            total += self.count(content)
            total += self.count(role)
        # 对话整体有约 3 token 的格式开销
        total += 3
        return total

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """记录 token 使用量"""
        with self._lock:
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens

    @property
    def total_input_tokens(self) -> int:
        """总输入 token 数"""
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        """总输出 token 数"""
        return self._total_output_tokens

    @property
    def total_tokens(self) -> int:
        """总 token 数"""
        return self._total_input_tokens + self._total_output_tokens

    def reset(self) -> None:
        """重置计数器"""
        with self._lock:
            self._total_input_tokens = 0
            self._total_output_tokens = 0

    def _is_code(self, text: str) -> bool:
        """启发式判断文本是否为代码"""
        code_indicators = [
            "def ", "class ", "import ", "function ",
            "=>", "->", "{", "}", "const ", "let ", "var ",
            "public ", "private ", "if __name__",
        ]
        code_count = sum(1 for ind in code_indicators if ind in text)
        return code_count >= 2

    def _count_special_tokens(self, text: str) -> int:
        """计算特殊 token 的额外开销"""
        extra = 0
        # 每个换行约 1 token
        extra += text.count("\n")
        # 数字序列
        numbers = re.findall(r"\d+", text)
        for num in numbers:
            extra += max(1, len(num) // 3)
        return extra


# ============================================================
# 成本跟踪器
# ============================================================

class CostTracker:
    """
    成本跟踪器

    跟踪 LLM 调用的成本。

    Usage:
        tracker = CostTracker()
        tracker.register_model("gpt-4", ModelPricing(0.03, 0.06))
        tracker.record("gpt-4", input_tokens=100, output_tokens=50)
        print(tracker.total_cost)
    """

    def __init__(self) -> None:
        self._model_pricing: Dict[str, ModelPricing] = {}
        self._cost_records: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def register_model(
        self,
        model_name: str,
        pricing: ModelPricing,
    ) -> None:
        """注册模型定价"""
        self._model_pricing[model_name] = pricing

    def record(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        记录一次调用成本

        Args:
            model_name: 模型名称
            input_tokens: 输入 token 数
            output_tokens: 输出 token 数

        Returns:
            本次调用成本
        """
        pricing = self._model_pricing.get(model_name, ModelPricing())
        input_cost = (input_tokens / 1000.0) * pricing.input_price_per_1k
        output_cost = (output_tokens / 1000.0) * pricing.output_price_per_1k
        total = input_cost + output_cost

        with self._lock:
            self._cost_records.append({
                "model": model_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total,
                "currency": pricing.currency,
                "timestamp": time.time(),
            })

        return total

    @property
    def total_cost(self) -> float:
        """总成本"""
        with self._lock:
            return sum(r["total_cost"] for r in self._cost_records)

    @property
    def total_input_tokens(self) -> int:
        """总输入 token 数"""
        with self._lock:
            return sum(r["input_tokens"] for r in self._cost_records)

    @property
    def total_output_tokens(self) -> int:
        """总输出 token 数"""
        with self._lock:
            return sum(r["output_tokens"] for r in self._cost_records)

    def get_cost_by_model(self) -> Dict[str, float]:
        """按模型分组统计成本"""
        costs: Dict[str, float] = defaultdict(float)
        with self._lock:
            for record in self._cost_records:
                costs[record["model"]] += record["total_cost"]
        return dict(costs)

    def get_summary(self) -> Dict[str, Any]:
        """获取成本摘要"""
        with self._lock:
            if not self._cost_records:
                return {
                    "total_cost": 0.0,
                    "total_calls": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                }

            return {
                "total_cost": round(self.total_cost, 6),
                "total_calls": len(self._cost_records),
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "cost_by_model": {
                    k: round(v, 6)
                    for k, v in self.get_cost_by_model().items()
                },
            }

    def reset(self) -> None:
        """重置"""
        with self._lock:
            self._cost_records.clear()


# ============================================================
# 流式响应处理器
# ============================================================

class StreamingHandler:
    """
    流式响应处理器

    处理 LLM 的流式响应，支持逐块处理和缓冲。

    Usage:
        handler = StreamingHandler()
        handler.on_chunk(lambda chunk: print(chunk, end=""))
        handler.on_complete(lambda full: process(full))
        # 模拟流式接收
        for chunk in ["Hello", ", ", "world", "!"]:
            handler.receive_chunk(chunk)
        handler.complete()
    """

    def __init__(self) -> None:
        self._chunks: List[str] = []
        self._chunk_callbacks: List[Callable[[str], None]] = []
        self._complete_callbacks: List[Callable[[str], None]] = []
        self._error_callbacks: List[Callable[[Exception], None]] = []
        self._buffer: List[str] = []
        self._started: bool = False
        self._completed: bool = False
        self._lock = threading.Lock()

    def on_chunk(self, callback: Callable[[str], None]) -> None:
        """注册块接收回调"""
        self._chunk_callbacks.append(callback)

    def on_complete(self, callback: Callable[[str], None]) -> None:
        """注册完成回调"""
        self._complete_callbacks.append(callback)

    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """注册错误回调"""
        self._error_callbacks.append(callback)

    def receive_chunk(self, chunk: str) -> None:
        """
        接收一个流式块

        Args:
            chunk: 文本块
        """
        with self._lock:
            self._started = True
            self._chunks.append(chunk)
            self._buffer.append(chunk)

        for callback in self._chunk_callbacks:
            try:
                callback(chunk)
            except Exception:
                pass

    def receive_delta(self, delta: Dict[str, Any]) -> None:
        """
        接收结构化增量数据

        Args:
            delta: 增量数据字典，需包含 'content' 或 'text' 字段
        """
        content = delta.get("content") or delta.get("text") or ""
        if content:
            self.receive_chunk(str(content))

    def complete(self) -> str:
        """
        标记流式响应完成

        Returns:
            完整的响应文本
        """
        with self._lock:
            self._completed = True
            full_text = "".join(self._chunks)

        for callback in self._complete_callbacks:
            try:
                callback(full_text)
            except Exception:
                pass

        return full_text

    def error(self, exc: Exception) -> None:
        """报告错误"""
        with self._lock:
            self._completed = True
        for callback in self._error_callbacks:
            try:
                callback(exc)
            except Exception:
                pass

    @property
    def full_text(self) -> str:
        """获取当前已接收的完整文本"""
        with self._lock:
            return "".join(self._chunks)

    @property
    def chunk_count(self) -> int:
        """已接收的块数"""
        return len(self._chunks)

    @property
    def is_complete(self) -> bool:
        """是否已完成"""
        return self._completed

    def reset(self) -> None:
        """重置处理器"""
        with self._lock:
            self._chunks.clear()
            self._buffer.clear()
            self._started = False
            self._completed = False


# ============================================================
# 回退处理器
# ============================================================

class FallbackHandler:
    """
    回退处理器

    当主模型失败时自动切换到回退模型。

    Usage:
        fallback = FallbackHandler(
            models=["gpt-4", "gpt-3.5-turbo", "claude-3-haiku"],
            selector=model_selector,
        )
        result = fallback.execute_with_fallback(
            call_fn, messages, config
        )
    """

    def __init__(
        self,
        models: Optional[List[str]] = None,
        selector: Optional[ModelSelector] = None,
        max_attempts: int = 3,
    ) -> None:
        self.models = models or []
        self.selector = selector
        self.max_attempts = max(1, max_attempts)
        self._attempt_history: List[Dict[str, Any]] = []

    def execute_with_fallback(
        self,
        call_fn: Callable[[str, Dict[str, Any]], Any],
        messages: List[Dict[str, str]],
        config: LLMNodeConfig,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        带回退执行 LLM 调用

        Args:
            call_fn: LLM 调用函数 (model_name, params) -> result
            messages: 消息列表
            config: LLM 配置
            context: 执行上下文

        Returns:
            LLM 调用结果
        """
        ctx = context or {}
        models_to_try = [config.default_model] + [
            m for m in self.fallback_models
            if m != config.default_model
        ]

        last_error: Optional[Exception] = None

        for i, model_name in enumerate(models_to_try[:self.max_attempts]):
            attempt_info = {
                "attempt": i + 1,
                "model": model_name,
                "start_time": time.time(),
                "success": False,
            }

            try:
                params = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "max_tokens": config.max_tokens,
                    "context": ctx,
                }

                result = call_fn(model_name, params)

                attempt_info["success"] = True
                attempt_info["end_time"] = time.time()
                attempt_info["duration"] = (
                    attempt_info["end_time"] - attempt_info["start_time"]
                )
                self._attempt_history.append(attempt_info)

                result["model_used"] = model_name
                result["fallback_used"] = i > 0
                result["attempt"] = i + 1
                return result

            except Exception as e:
                last_error = e
                attempt_info["error"] = str(e)
                attempt_info["error_type"] = type(e).__name__
                attempt_info["end_time"] = time.time()
                attempt_info["duration"] = (
                    attempt_info["end_time"] - attempt_info["start_time"]
                )
                self._attempt_history.append(attempt_info)

                # 速率限制时等待
                if isinstance(e, LLMRateLimitError) and e.retry_after > 0:
                    time.sleep(min(e.retry_after, config.retry_delay * (i + 1)))

                continue

        raise LLMError(
            config.default_model,
            f"所有模型尝试均失败 (共 {min(len(models_to_try), self.max_attempts)} 次)",
            cause=last_error,
        )

    @property
    def fallback_models(self) -> List[str]:
        """回退模型列表"""
        return self.models

    def get_attempt_history(self) -> List[Dict[str, Any]]:
        """获取尝试历史"""
        return list(self._attempt_history)

    def reset_history(self) -> None:
        """重置历史"""
        self._attempt_history.clear()


# ============================================================
# 输出解析器
# ============================================================

class OutputParser:
    """
    LLM 输出解析器

    解析 LLM 的原始输出为结构化数据。

    Usage:
        parser = OutputParser()
        result = parser.parse_json(llm_output)
        result = parser.parse_code_block(llm_output)
    """

    def parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        从文本中提取 JSON

        Args:
            text: LLM 输出文本

        Returns:
            解析后的字典，解析失败返回 None
        """
        # 尝试直接解析
        try:
            return json.loads(text.strip())
        except (json.JSONDecodeError, ValueError):
            pass

        # 尝试从 markdown 代码块中提取
        code_block_match = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?\s*```",
            text,
            re.DOTALL,
        )
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1).strip())
            except (json.JSONDecodeError, ValueError):
                pass

        # 尝试找到第一个 { 和最后一个 } 之间的内容
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            try:
                return json.loads(text[first_brace:last_brace + 1])
            except (json.JSONDecodeError, ValueError):
                pass

        return None

    def parse_code_block(
        self,
        text: str,
        language: str = "",
    ) -> Optional[str]:
        """
        从文本中提取代码块

        Args:
            text: LLM 输出文本
            language: 期望的编程语言

        Returns:
            代码块内容，未找到返回 None
        """
        if language:
            pattern = rf"```{re.escape(language)}\s*\n?(.*?)\n?\s*```"
        else:
            pattern = r"```\w*\s*\n?(.*?)\n?\s*```"

        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def parse_list(self, text: str) -> List[str]:
        """
        从文本中提取列表

        Args:
            text: LLM 输出文本

        Returns:
            列表项
        """
        items: List[str] = []

        # 尝试解析 JSON 数组
        try:
            parsed = json.loads(text.strip())
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except (json.JSONDecodeError, ValueError):
            pass

        # 尝试匹配编号列表
        numbered = re.findall(r"^\s*\d+[\.\)]\s*(.+)$", text, re.MULTILINE)
        if numbered:
            return numbered

        # 尝试匹配无序列表
        unordered = re.findall(r"^\s*[-\*]\s*(.+)$", text, re.MULTILINE)
        if unordered:
            return unordered

        # 按换行分割
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return lines

    def extract_thinking(self, text: str) -> Tuple[str, str]:
        """
        分离思考过程和最终答案

        Args:
            text: LLM 输出文本

        Returns:
            (thinking, answer) 元组
        """
        # 常见的思考/答案分隔模式
        patterns = [
            r"(.*?)<answer>(.*?)</answer>",
            r"(.*?)最终答案[：:]\s*(.*)",
            r"(.*?)Answer[：:]\s*(.*)",
            r"<think>(.*?)</think>(.*)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                thinking = match.group(1).strip()
                answer = match.group(2).strip()
                return thinking, answer

        return "", text.strip()

    def clean_output(self, text: str) -> str:
        """
        清理 LLM 输出

        Args:
            text: 原始输出

        Returns:
            清理后的文本
        """
        # 移除多余的空白
        cleaned = re.sub(r"\n{3,}", "\n\n", text)
        cleaned = re.sub(r"  +", " ", cleaned)
        # 移除首尾空白
        cleaned = cleaned.strip()
        return cleaned


# ============================================================
# LLM 推理节点
# ============================================================

class LLMNode:
    """
    LLM 推理节点

    高级 LLM 调用节点，整合模板渲染、模型选择、流式处理、
    Token 计数、成本跟踪和回退支持。

    Usage:
        node = LLMNode(
            node_id="chat_node",
            config=LLMNodeConfig(default_model="gpt-4"),
            model_selector=selector,
        )
        result = node.execute(
            inputs={"question": "什么是机器学习？"},
            context={},
        )
    """

    def __init__(
        self,
        node_id: str,
        system_prompt: str = "",
        user_prompt: str = "",
        config: Optional[LLMNodeConfig] = None,
        model_selector: Optional[ModelSelector] = None,
        prompt_renderer: Optional[PromptRenderer] = None,
        token_counter: Optional[TokenCounter] = None,
        cost_tracker: Optional[CostTracker] = None,
        output_parser: Optional[OutputParser] = None,
        fallback_handler: Optional[FallbackHandler] = None,
        llm_function: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.node_id = node_id
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.config = config or LLMNodeConfig()
        self.model_selector = model_selector or ModelSelector()
        self.prompt_renderer = prompt_renderer or PromptRenderer()
        self.token_counter = token_counter or TokenCounter()
        self.cost_tracker = cost_tracker or CostTracker()
        self.output_parser = output_parser or OutputParser()
        self.fallback_handler = fallback_handler or FallbackHandler(
            models=self.config.fallback_models,
        )
        self.llm_function = llm_function

        self._execution_count: int = 0
        self._streaming_handler: Optional[StreamingHandler] = None

    def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        执行 LLM 推理

        Args:
            inputs: 输入数据（模板变量）
            context: 执行上下文

        Returns:
            推理结果字典
        """
        ctx = context or {}
        start_time = time.time()

        # 合并 inputs 和 context 作为模板变量
        variables = dict(inputs)
        variables.update({k: v for k, v in ctx.items() if not k.startswith("_")})

        # 1. 渲染提示
        system_prompt = self.prompt_renderer.render(
            self.system_prompt, variables
        )
        user_prompt = self.prompt_renderer.render(
            self.user_prompt or inputs.get("prompt", ""), variables
        )

        # 2. 构建消息
        history = inputs.get("conversation_history")
        messages = self.prompt_renderer.build_messages(
            system_prompt, user_prompt, variables, history
        )

        # 3. Token 计数
        input_tokens = 0
        if self.config.enable_token_counting:
            input_tokens = self.token_counter.count_messages(messages)

        # 4. 执行 LLM 调用
        if self.llm_function is not None:
            result = self._call_with_fallback(messages, ctx)
        else:
            # 模拟响应
            result = {
                "content": f"[模拟LLM响应] prompt: {user_prompt[:100]}...",
                "model": self.config.default_model,
                "input_tokens": input_tokens,
                "output_tokens": 0,
            }

        # 5. 处理输出
        output_tokens = result.get("output_tokens", 0)
        if self.config.enable_token_counting and output_tokens > 0:
            self.token_counter.record_usage(input_tokens, output_tokens)

        if self.config.enable_cost_tracking and output_tokens > 0:
            model_name = result.get("model", self.config.default_model)
            self.cost_tracker.record(model_name, input_tokens, output_tokens)

        # 6. 构建返回结果
        duration = time.time() - start_time
        self._execution_count += 1

        return {
            "node_id": self.node_id,
            "content": result.get("content", ""),
            "model": result.get("model", self.config.default_model),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "duration": round(duration, 4),
            "fallback_used": result.get("fallback_used", False),
            "attempt": result.get("attempt", 1),
            "success": True,
        }

    def _call_with_fallback(
        self,
        messages: List[Dict[str, str]],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """带回退的 LLM 调用"""

        def call_fn(model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
            return self.llm_function(  # type: ignore[misc]
                model=model_name,
                messages=params["messages"],
                temperature=params["temperature"],
                top_p=params["top_p"],
                max_tokens=params["max_tokens"],
                context=params["context"],
            )

        return self.fallback_handler.execute_with_fallback(
            call_fn, messages, self.config, context
        )

    def execute_streaming(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> StreamingHandler:
        """
        执行流式 LLM 推理

        Args:
            inputs: 输入数据
            context: 执行上下文

        Returns:
            StreamingHandler 用于接收流式响应
        """
        self._streaming_handler = StreamingHandler()
        # 流式调用由 llm_function 内部处理
        # 这里返回 handler 供外部使用
        return self._streaming_handler

    @property
    def stats(self) -> Dict[str, Any]:
        """获取节点统计"""
        return {
            "node_id": self.node_id,
            "execution_count": self._execution_count,
            "token_stats": {
                "total_input": self.token_counter.total_input_tokens,
                "total_output": self.token_counter.total_output_tokens,
                "total": self.token_counter.total_tokens,
            },
            "cost_stats": self.cost_tracker.get_summary(),
        }
