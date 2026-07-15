"""
阿里云百炼 DashScope LLM Provider

基于阿里云百炼平台 (DashScope / 通义千问) 的大语言模型客户端，
支持对话补全、视觉理解、流式输出、函数调用、文本嵌入和图像生成。

支持的模型:
- qwen-turbo: 通义千问 Turbo（高性价比）
- qwen-plus: 通义千问 Plus（均衡型）
- qwen-max: 通义千问 Max（旗舰型）
- qwen-vl-max: 通义千问 VL Max（视觉理解）

官方文档: https://help.aliyun.com/zh/dashscope/
API 参考: https://help.aliyun.com/zh/dashscope/developer-reference/api-details

Author: AGI Framework Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, Union

import aiohttp

from ..computer_use.agent_brain import LLMClient

logger = logging.getLogger(__name__)


# ============================================================
# 异常定义
# ============================================================

class DashScopeError(Exception):
    """DashScope 平台异常"""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        self.request_id = request_id
        super().__init__(f"[DashScope {code}] {message} (request_id={request_id})")

    @classmethod
    def from_response(cls, body: Dict[str, Any], status_code: int = 0) -> DashScopeError:
        """从 API 响应构造异常"""
        return cls(
            code=body.get("code", "Unknown"),
            message=body.get("message", body.get("msg", "未知错误")),
            status_code=status_code,
            request_id=body.get("request_id"),
        )

    @property
    def is_auth_error(self) -> bool:
        """是否为鉴权错误"""
        return self.code in ("InvalidApiKey", "InvalidParameter.ApiKey", "Unauthorized")

    @property
    def is_rate_limited(self) -> bool:
        """是否被限流"""
        return self.code == "Throttling" or "quota" in self.code.lower()

    @property
    def is_retryable(self) -> bool:
        """是否可重试"""
        return self.is_rate_limited or self.code in (
            "ServiceUnavailable",
            "InternalError",
            "Timeout",
        )


# ============================================================
# 数据模型
# ============================================================

class DashScopeModel(str, Enum):
    """DashScope 支持的模型"""
    QWEN_TURBO = "qwen-turbo"
    QWEN_PLUS = "qwen-plus"
    QWEN_MAX = "qwen-max"
    QWEN_VL_MAX = "qwen-vl-max"
    QWEN_VL_PLUS = "qwen-vl-plus"
    QWEN_LONG = "qwen-long"
    QWEN_7B_CHAT = "qwen-7b-chat"
    QWEN_14B_CHAT = "qwen-14b-chat"
    WANX_V1 = "wanx-v1"          # 图像生成
    WANX2_1_T2I_TURBO = "wanx2.1-t2i-turbo"  # 文生图加速版
    TEXT_EMBEDDING_V1 = "text-embedding-v1"
    TEXT_EMBEDDING_V2 = "text-embedding-v2"
    TEXT_EMBEDDING_V3 = "text-embedding-v3"


@dataclass
class FunctionDefinition:
    """函数/工具定义"""
    name: str
    description: str = ""
    parameters: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为 API 格式"""
        result: Dict[str, Any] = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
            },
        }
        if self.parameters:
            result["function"]["parameters"] = self.parameters
        return result


@dataclass
class FunctionCall:
    """函数调用结果"""
    id: str = ""
    name: str = ""
    arguments: str = ""

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> FunctionCall:
        fn = data.get("function", data)
        return cls(
            id=data.get("id", ""),
            name=fn.get("name", ""),
            arguments=fn.get("arguments", ""),
        )

    @property
    def parsed_arguments(self) -> Dict[str, Any]:
        """解析参数为字典"""
        try:
            return json.loads(self.arguments)
        except json.JSONDecodeError:
            return {}


@dataclass
class ChatResponse:
    """对话补全响应"""
    content: str = ""
    model: str = ""
    finish_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    request_id: str = ""
    function_calls: List[FunctionCall] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def has_function_calls(self) -> bool:
        """是否包含函数调用"""
        return len(self.function_calls) > 0


@dataclass
class EmbeddingResult:
    """文本嵌入结果"""
    embeddings: List[List[float]] = field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    total_tokens: int = 0
    request_id: str = ""


@dataclass
class ImageGenerationResult:
    """图像生成结果"""
    prompt: str = ""
    model: str = ""
    image_urls: List[str] = field(default_factory=list)
    image_b64_list: List[str] = field(default_factory=list)
    request_id: str = ""
    latency_ms: float = 0.0


# ============================================================
# 速率限制器
# ============================================================

class RateLimiter:
    """令牌桶速率限制器"""

    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 300000,
    ) -> None:
        self._rpm = requests_per_minute
        self._tpm = tokens_per_minute
        self._request_times: List[float] = []
        self._token_usage: List[Tuple[float, int]] = []
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int = 0) -> None:
        """获取执行许可，必要时等待"""
        async with self._lock:
            now = time.time()
            window = 60.0

            # 清理过期记录
            self._request_times = [t for t in self._request_times if now - t < window]
            self._token_usage = [(t, c) for t, c in self._token_usage if now - t < window]

            # 检查 RPM
            if len(self._request_times) >= self._rpm:
                sleep_time = window - (now - self._request_times[0]) + 0.1
                if sleep_time > 0:
                    logger.debug("触发 RPM 限流，等待 %.2f 秒", sleep_time)
                    await asyncio.sleep(sleep_time)

            # 检查 TPM
            current_tpm = sum(c for _, c in self._token_usage)
            if current_tpm + estimated_tokens > self._tpm:
                if self._token_usage:
                    sleep_time = window - (now - self._token_usage[0][0]) + 0.1
                    if sleep_time > 0:
                        logger.debug("触发 TPM 限流，等待 %.2f 秒", sleep_time)
                        await asyncio.sleep(sleep_time)

            self._request_times.append(time.time())
            self._token_usage.append((time.time(), estimated_tokens))

    def update_limits(self, rpm: Optional[int] = None, tpm: Optional[int] = None) -> None:
        """动态更新限流阈值"""
        if rpm is not None:
            self._rpm = rpm
        if tpm is not None:
            self._tpm = tpm


# ============================================================
# 主客户端
# ============================================================

class DashScopeClient(LLMClient):
    """阿里云百炼 DashScope LLM 客户端

    继承自 LLMClient 抽象基类，提供完整的 DashScope API 集成。

    功能:
        - 对话补全 (chat_completion)
        - 视觉理解 (vision_completion)
        - 流式输出 (stream_chat)
        - 函数调用 / 工具使用 (function calling)
        - 文本嵌入 (embed)
        - 图像生成 (generate_image)
        - Token 计数
        - 速率限制
        - 自动重试

    Example:
        client = DashScopeClient(api_key="sk-xxx")
        response = await client.chat_completion(
            messages=[{"role": "user", "content": "你好"}],
            model="qwen-turbo",
        )
        print(response.content)
    """

    DEFAULT_MODEL = DashScopeModel.QWEN_TURBO.value
    VISION_MODEL = DashScopeModel.QWEN_VL_MAX.value
    EMBEDDING_MODEL = DashScopeModel.TEXT_EMBEDDING_V3.value
    IMAGE_MODEL = DashScopeModel.WANX2_1_T2I_TURBO.value

    # API 端点
    _CHAT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    _EMBEDDING_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
    _IMAGE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        *,
        default_model: Optional[str] = None,
        rpm_limit: int = 60,
        tpm_limit: int = 300000,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 120.0,
        **kwargs,
    ) -> None:
        """初始化 DashScope 客户端

        Args:
            api_key: DashScope API Key (sk-xxx)
            base_url: 自定义 API 基础地址
            default_model: 默认模型
            rpm_limit: 每分钟请求数限制
            tpm_limit: 每分钟 Token 数限制
            max_retries: 最大重试次数
            retry_delay: 重试基础延迟（秒）
            timeout: 请求超时时间（秒）
        """
        super().__init__(api_key, base_url, **kwargs)
        self._default_model = default_model or self.DEFAULT_MODEL
        self._base_url = (base_url or "https://dashscope.aliyuncs.com").rstrip("/")
        self._rate_limiter = RateLimiter(rpm=rpm_limit, tpm=tpm_limit)
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._timeout = timeout
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

    @property
    def headers(self) -> Dict[str, str]:
        """请求头"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # ----------------------------------------------------------
    # Token 计数
    # ----------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """估算 Token 数量

        使用近似算法: 中文字符约 1 token/字，英文约 4 字符/token。
        对于精确计数，建议使用 DashScope tokenizer API。

        Args:
            text: 输入文本

        Returns:
            估算的 Token 数量
        """
        if not text:
            return 0
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', text))
        other_chars = len(text) - chinese_chars
        return chinese_chars + max(1, other_chars // 4)

    def count_messages_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """估算消息列表的 Token 数量

        Args:
            messages: 消息列表

        Returns:
            估算的 Token 总量
        """
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count_tokens(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            total += self.count_tokens(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            total += 85  # 图像约 85 tokens (低分辨率)
            total += 4  # 每条消息约 4 tokens 的格式开销
        return total + 3  # 对话起始约 3 tokens

    # ----------------------------------------------------------
    # 对话补全
    # ----------------------------------------------------------

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> str:
        """发送对话补全请求

        Args:
            messages: 消息列表，格式 [{"role": "user", "content": "..."}]
            model: 模型名称
            temperature: 温度参数 (0-2)
            max_tokens: 最大生成 Token 数
            **kwargs: 其他参数 (top_p, stop, tools, etc.)

        Returns:
            模型生成的文本内容
        """
        response = await self._chat_completion_full(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.content

    async def _chat_completion_full(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> ChatResponse:
        """完整的对话补全请求，返回 ChatResponse 对象"""
        model = model or self._default_model
        estimated_tokens = self.count_messages_tokens(messages)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # 可选参数
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]
        if "seed" in kwargs:
            payload["seed"] = kwargs["seed"]
        if "response_format" in kwargs:
            payload["response_format"] = kwargs["response_format"]

        # 函数调用 / 工具使用
        if "tools" in kwargs:
            payload["tools"] = kwargs["tools"]
        if "tool_choice" in kwargs:
            payload["tool_choice"] = kwargs["tool_choice"]

        url = f"{self._base_url}/compatible-mode/v1/chat/completions"

        start_time = time.time()
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                await self._rate_limiter.acquire(estimated_tokens)

                timeout = aiohttp.ClientTimeout(total=self._timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, headers=self.headers, json=payload) as resp:
                        body = await resp.json()

                if resp.status != 200:
                    error = DashScopeError.from_response(body, resp.status)
                    if error.is_retryable and attempt < self._max_retries:
                        delay = self._retry_delay * (2 ** attempt)
                        logger.warning("DashScope 请求失败 (attempt %d/%d): %s, %.1f 秒后重试",
                                       attempt + 1, self._max_retries + 1, error.message, delay)
                        await asyncio.sleep(delay)
                        continue
                    raise error

                return self._parse_chat_response(body, model, start_time)

            except DashScopeError:
                raise
            except asyncio.TimeoutError as exc:
                last_error = exc
                if attempt < self._max_retries:
                    delay = self._retry_delay * (2 ** attempt)
                    logger.warning("DashScope 请求超时 (attempt %d/%d), %.1f 秒后重试",
                                   attempt + 1, self._max_retries + 1, delay)
                    await asyncio.sleep(delay)
                    continue
            except aiohttp.ClientError as exc:
                last_error = exc
                if attempt < self._max_retries:
                    delay = self._retry_delay * (2 ** attempt)
                    logger.warning("DashScope 网络错误 (attempt %d/%d): %s, %.1f 秒后重试",
                                   attempt + 1, self._max_retries + 1, exc, delay)
                    await asyncio.sleep(delay)
                    continue

        raise DashScopeError(
            code="MaxRetriesExceeded",
            message=f"请求失败，已达最大重试次数: {last_error}",
        )

    def _parse_chat_response(
        self,
        body: Dict[str, Any],
        model: str,
        start_time: float,
    ) -> ChatResponse:
        """解析对话补全 API 响应"""
        choice = body.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = body.get("usage", {})

        # 提取函数调用
        function_calls: List[FunctionCall] = []
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            function_calls = [FunctionCall.from_api(tc) for tc in tool_calls]

        # 统计 Token 使用量
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self.request_count += 1
        self.token_count += output_tokens

        return ChatResponse(
            content=message.get("content", ""),
            model=body.get("model", model),
            finish_reason=choice.get("finish_reason", ""),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=usage.get("total_tokens", input_tokens + output_tokens),
            request_id=body.get("id", ""),
            function_calls=function_calls,
            latency_ms=(time.time() - start_time) * 1000,
        )

    # ----------------------------------------------------------
    # 视觉理解
    # ----------------------------------------------------------

    async def vision_completion(
        self,
        messages: List[Dict[str, Any]],
        image_base64: str,
        model: Optional[str] = None,
        **kwargs,
    ) -> str:
        """发送视觉理解请求

        将图片附加到最后一条用户消息中，支持多模态理解。

        Args:
            messages: 消息列表
            image_base64: Base64 编码的图片数据
            model: 模型名称，默认 qwen-vl-max
            **kwargs: 其他参数

        Returns:
            模型生成的文本内容
        """
        model = model or self.VISION_MODEL

        # 构造多模态消息
        vision_messages = self._build_vision_messages(messages, image_base64)

        return await self.chat_completion(vision_messages, model=model, **kwargs)

    async def vision_completion_with_url(
        self,
        messages: List[Dict[str, Any]],
        image_url: str,
        model: Optional[str] = None,
        **kwargs,
    ) -> str:
        """通过 URL 发送视觉理解请求

        Args:
            messages: 消息列表
            image_url: 图片 URL
            model: 模型名称
            **kwargs: 其他参数

        Returns:
            模型生成的文本内容
        """
        model = model or self.VISION_MODEL
        vision_messages = self._build_vision_messages_url(messages, image_url)
        return await self.chat_completion(vision_messages, model=model, **kwargs)

    def _build_vision_messages(
        self,
        messages: List[Dict[str, Any]],
        image_base64: str,
    ) -> List[Dict[str, Any]]:
        """构造包含 Base64 图片的多模态消息"""
        vision_messages = [msg.copy() for msg in messages]

        # 在最后一条用户消息中插入图片
        if vision_messages and vision_messages[-1]["role"] == "user":
            content = vision_messages[-1]["content"]
            if isinstance(content, str):
                vision_messages[-1]["content"] = [
                    {"type": "text", "text": content},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                        },
                    },
                ]
            elif isinstance(content, list):
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                    },
                })
        else:
            vision_messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                        },
                    },
                ],
            })

        return vision_messages

    def _build_vision_messages_url(
        self,
        messages: List[Dict[str, Any]],
        image_url: str,
    ) -> List[Dict[str, Any]]:
        """构造包含图片 URL 的多模态消息"""
        vision_messages = [msg.copy() for msg in messages]

        if vision_messages and vision_messages[-1]["role"] == "user":
            content = vision_messages[-1]["content"]
            if isinstance(content, str):
                vision_messages[-1]["content"] = [
                    {"type": "text", "text": content},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ]
            elif isinstance(content, list):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url},
                })
        else:
            vision_messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            })

        return vision_messages

    # ----------------------------------------------------------
    # 流式输出
    # ----------------------------------------------------------

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """流式对话补全

        逐块返回模型生成的内容，适用于实时展示场景。

        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大生成 Token 数
            **kwargs: 其他参数

        Yields:
            模型生成的文本片段
        """
        model = model or self._default_model
        estimated_tokens = self.count_messages_tokens(messages)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]
        if "tools" in kwargs:
            payload["tools"] = kwargs["tools"]

        url = f"{self._base_url}/compatible-mode/v1/chat/completions"

        await self._rate_limiter.acquire(estimated_tokens)
        timeout = aiohttp.ClientTimeout(total=self._timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=self.headers, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.json()
                    raise DashScopeError.from_response(body, resp.status)

                async for line in resp.content:
                    line_str = line.decode("utf-8").strip()
                    if not line_str or not line_str.startswith("data: "):
                        continue

                    data_str = line_str[6:]
                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                self.token_count += self.count_tokens(content)
                                self.request_count += 1
                                yield content
                    except json.JSONDecodeError:
                        continue

    # ----------------------------------------------------------
    # 函数调用 / 工具使用
    # ----------------------------------------------------------

    async def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Union[FunctionDefinition, Dict[str, Any]]],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        tool_choice: str = "auto",
        **kwargs,
    ) -> ChatResponse:
        """带函数调用的对话补全

        Args:
            messages: 消息列表
            tools: 工具/函数定义列表
            model: 模型名称
            temperature: 温度参数（函数调用建议使用较低温度）
            max_tokens: 最大生成 Token 数
            tool_choice: 工具选择策略 ("auto", "none", "required", 或指定函数名)
            **kwargs: 其他参数

        Returns:
            ChatResponse，可能包含 function_calls
        """
        # 转换工具定义
        tool_dicts = []
        for tool in tools:
            if isinstance(tool, FunctionDefinition):
                tool_dicts.append(tool.to_dict())
            elif isinstance(tool, dict):
                tool_dicts.append(tool)
            else:
                raise ValueError(f"不支持的工具定义类型: {type(tool)}")

        return await self._chat_completion_full(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tool_dicts,
            tool_choice=tool_choice,
            **kwargs,
        )

    @staticmethod
    def build_tool_result_message(
        tool_call_id: str,
        function_name: str,
        result: str,
    ) -> Dict[str, Any]:
        """构造工具调用结果消息

        Args:
            tool_call_id: 工具调用 ID
            function_name: 函数名称
            result: 函数执行结果

        Returns:
            可追加到 messages 中的 tool 角色消息
        """
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result,
        }

    # ----------------------------------------------------------
    # 文本嵌入
    # ----------------------------------------------------------

    async def embed(
        self,
        texts: Union[str, List[str]],
        model: Optional[str] = None,
        input_type: str = "search_query",
    ) -> EmbeddingResult:
        """生成文本嵌入向量

        Args:
            texts: 输入文本或文本列表
            model: 嵌入模型名称
            input_type: 输入类型 ("search_query" 或 "search_document")

        Returns:
            EmbeddingResult 包含嵌入向量和使用量
        """
        model = model or self.EMBEDDING_MODEL
        input_texts = [texts] if isinstance(texts, str) else texts

        payload: Dict[str, Any] = {
            "model": model,
            "input": input_texts,
        }

        # DashScope 嵌入 API 兼容 OpenAI 格式，支持 input_type 参数
        if model.startswith("text-embedding-v"):
            payload["encoding_format"] = "float"

        url = f"{self._base_url}/compatible-mode/v1/embeddings"

        await self._rate_limiter.acquire(len(input_texts) * 50)
        timeout = aiohttp.ClientTimeout(total=self._timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=self.headers, json=payload) as resp:
                body = await resp.json()

        if resp.status != 200:
            raise DashScopeError.from_response(body, resp.status)

        usage = body.get("usage", {})
        embeddings = []
        for item in body.get("data", []):
            embeddings.append(item.get("embedding", []))

        # 按 index 排序
        embeddings.sort(key=lambda x: body["data"][embeddings.index(x)].get("index", 0))

        return EmbeddingResult(
            embeddings=embeddings,
            model=body.get("model", model),
            input_tokens=usage.get("prompt_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            request_id=body.get("id", ""),
        )

    # ----------------------------------------------------------
    # 图像生成
    # ----------------------------------------------------------

    async def generate_image(
        self,
        prompt: str,
        model: Optional[str] = None,
        size: str = "1024*1024",
        n: int = 1,
        style: str = "<auto>",
    ) -> ImageGenerationResult:
        """生成图像

        Args:
            prompt: 图像描述文本
            model: 图像生成模型名称
            size: 图像尺寸 ("512*512", "768*768", "1024*1024", "1024*1536", "1536*1024")
            n: 生成数量 (1-4)
            style: 风格 ("<auto>", "<photography>", "<portrait>", "<3d cartoon>", "<anime>", "<oil painting>")

        Returns:
            ImageGenerationResult 包含图像 URL
        """
        model = model or self.IMAGE_MODEL

        payload = {
            "model": model,
            "input": {
                "prompt": prompt,
            },
            "parameters": {
                "size": size,
                "n": n,
                "style": style,
            },
        }

        url = f"{self._base_url}/api/v1/services/aigc/text2image/image-synthesis"

        start_time = time.time()
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                await self._rate_limiter.acquire(100)

                timeout = aiohttp.ClientTimeout(total=120.0)  # 图像生成超时更长
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        url,
                        headers=self.headers,
                        json=payload,
                    ) as resp:
                        body = await resp.json()

                if resp.status != 200:
                    error = DashScopeError.from_response(body, resp.status)
                    if error.is_retryable and attempt < self._max_retries:
                        delay = self._retry_delay * (2 ** attempt)
                        await asyncio.sleep(delay)
                        continue
                    raise error

                # 解析响应
                output = body.get("output", {})
                results = output.get("results", [])
                image_urls = [r.get("url", "") for r in results if r.get("url")]

                return ImageGenerationResult(
                    prompt=prompt,
                    model=model,
                    image_urls=image_urls,
                    request_id=body.get("request_id", body.get("id", "")),
                    latency_ms=(time.time() - start_time) * 1000,
                )

            except DashScopeError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt < self._max_retries:
                    delay = self._retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue

        raise DashScopeError(
            code="MaxRetriesExceeded",
            message=f"图像生成失败，已达最大重试次数: {last_error}",
        )

    # ----------------------------------------------------------
    # 统计信息
    # ----------------------------------------------------------

    def get_usage_stats(self) -> Dict[str, Any]:
        """获取使用统计信息

        Returns:
            包含请求次数、Token 使用量等信息的字典
        """
        return {
            "provider": "dashscope",
            "default_model": self._default_model,
            "request_count": self.request_count,
            "token_count": self.token_count,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
        }

    def reset_stats(self) -> None:
        """重置统计信息"""
        self.request_count = 0
        self.token_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    # ----------------------------------------------------------
    # 辅助方法
    # ----------------------------------------------------------

    async def list_models(self) -> List[str]:
        """获取可用模型列表

        Returns:
            模型名称列表
        """
        url = f"{self._base_url}/compatible-mode/v1/models"
        timeout = aiohttp.ClientTimeout(total=30.0)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=self.headers) as resp:
                body = await resp.json()

        if resp.status != 200:
            raise DashScopeError.from_response(body, resp.status)

        models = body.get("data", [])
        return [m.get("id", "") for m in models]

    def __repr__(self) -> str:
        return (
            f"DashScopeClient("
            f"model={self._default_model!r}, "
            f"requests={self.request_count}, "
            f"tokens={self.token_count})"
        )
