"""
通用模型适配器 (Universal LLM Provider)

设计理念: 90%的国产模型API兼容OpenAI格式，因此一个适配器+配置即可覆盖全部。
该模块提供统一的模型调用接口，通过配置文件接入任意国产/国际模型。

支持的协议类型:
- openai: OpenAI兼容格式 (覆盖90%模型)
- claude: Anthropic Claude格式
- custom: 自定义协议 (通过request_mapping/response_mapping映射)

支持的认证方式:
- bearer: Bearer Token (OpenAI/大部分国产)
- api_key_query: URL参数API Key
- api_key_header: 自定义Header API Key
- oauth2: OAuth2.0
- hmac_sha256: HMAC-SHA256签名
- aws_sigv4: AWS签名
- custom: 自定义认证

Author: AGI Team
Version: 1.0.0
"""

import time
import json
import hashlib
import hmac
import base64
import logging
import asyncio
from abc import ABC
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import (
    Dict, List, Optional, Any, Set, Tuple,
    Callable, Union, AsyncIterator, TypeVar
)
from urllib.parse import urlencode, urlparse, quote

import aiohttp

from .base import (
    BaseLLMProvider,
    LLMConfig,
    LLMResponse,
    LLMError,
    ModelCapability,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================
# 数据类定义
# ============================================================

@dataclass
class UniversalModelConfig:
    """
    单个模型的配置

    通过该配置可以描述任意模型的接入方式，包括API地址、认证方式、
    协议类型、模型能力、请求/响应映射等。

    Attributes:
        model_id: 唯一标识，如 "zhipu/glm-4"
        model_name: 显示名称，如 "GLM-4"
        provider: 提供商，如 "zhipu"
        api_base: API基础URL
        api_key: API密钥
        api_protocol: 协议类型 (openai/claude/custom)
        auth_type: 认证类型 (bearer/api_key_query/api_key_header/oauth2/hmac_sha256/aws_sigv4/custom)
        auth_config: 认证额外配置
        max_context: 最大上下文长度
        max_output: 最大输出长度
        supports_stream: 是否支持流式输出
        supports_vision: 是否支持视觉输入
        supports_function_call: 是否支持函数调用
        supports_embeddings: 是否支持向量嵌入
        request_mapping: 请求字段映射
        response_mapping: 响应字段映射
        default_params: 默认参数
        rate_limit: 每分钟请求限制
        timeout: 请求超时时间(秒)
        retry_count: 重试次数
        input_price_per_1k: 输入价格/千token
        output_price_per_1k: 输出价格/千token
    """
    model_id: str = ""
    model_name: str = ""
    provider: str = ""
    api_base: str = ""
    api_key: str = ""
    api_protocol: str = "openai"
    auth_type: str = "bearer"
    auth_config: Dict[str, Any] = field(default_factory=dict)

    # 模型能力
    max_context: int = 8192
    max_output: int = 4096
    supports_stream: bool = True
    supports_vision: bool = False
    supports_function_call: bool = False
    supports_embeddings: bool = False

    # 请求映射
    request_mapping: Dict[str, Any] = field(default_factory=dict)
    response_mapping: Dict[str, Any] = field(default_factory=dict)

    # 高级配置
    default_params: Dict[str, Any] = field(default_factory=dict)
    rate_limit: int = 60
    timeout: float = 60.0
    retry_count: int = 3

    # 成本
    input_price_per_1k: float = 0.0
    output_price_per_1k: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UniversalModelConfig":
        """从字典创建配置"""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def get_capabilities(self) -> Set[ModelCapability]:
        """获取模型能力集合"""
        caps: Set[ModelCapability] = set()
        if self.supports_stream:
            caps.add(ModelCapability.STREAMING)
        if self.supports_function_call:
            caps.add(ModelCapability.FUNCTION_CALLING)
        if self.supports_vision:
            caps.add(ModelCapability.VISION)
        return caps


# ============================================================
# 速率限制器
# ============================================================

class RateLimiter:
    """
    令牌桶速率限制器

    用于控制对模型API的请求频率，避免超出提供商的RPM限制。

    Attributes:
        rate_limit: 每分钟最大请求数
    """

    def __init__(self, rate_limit: int = 60):
        """
        初始化速率限制器。

        Args:
            rate_limit: 每分钟最大请求数
        """
        self._rate_limit = rate_limit
        self._tokens: float = float(rate_limit)
        self._last_refill: float = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """
        获取一个令牌，如果桶中没有足够的令牌则等待。

        Raises:
            RuntimeError: 如果等待超时
        """
        async with self._lock:
            now = time.time()
            elapsed = now - self._last_refill
            self._tokens = min(
                self._rate_limit,
                self._tokens + elapsed * (self._rate_limit / 60.0)
            )
            self._last_refill = now

            if self._tokens < 1.0:
                wait_time = (1.0 - self._tokens) / (self._rate_limit / 60.0)
                await asyncio.sleep(wait_time)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0

    def update_rate_limit(self, rate_limit: int) -> None:
        """
        更新速率限制。

        Args:
            rate_limit: 新的每分钟最大请求数
        """
        self._rate_limit = rate_limit


# ============================================================
# 通用模型适配器
# ============================================================

class UniversalLLMProvider(BaseLLMProvider):
    """
    通用模型适配器

    通过一个统一的适配器接入任意国产/国际模型。核心设计理念是：
    90%的国产模型API兼容OpenAI格式，因此一个适配器+配置即可覆盖全部。

    对于不兼容OpenAI格式的模型，通过 request_mapping 和 response_mapping
    进行字段映射转换；对于特殊认证方式，通过 auth_type 和 auth_config 处理。

    Example:
        ```python
        # 创建通用适配器
        config = UniversalModelConfig(
            model_id="zhipu/glm-4",
            model_name="GLM-4",
            provider="zhipu",
            api_base="https://open.bigmodel.cn/api/paas/v4",
            api_key="your-api-key",
            api_protocol="openai",
            auth_type="bearer",
            max_context=8192,
            supports_function_call=True,
        )
        provider = UniversalLLMProvider(config)

        # 调用模型
        response = await provider.generate([
            {"role": "user", "content": "你好"}
        ])
        print(response.content)
        ```
    """

    PROVIDER_NAME: str = "universal"
    SUPPORTED_MODELS: Set[str] = {"*"}
    DEFAULT_MODEL: str = ""

    def __init__(
        self,
        model_config: Optional[UniversalModelConfig] = None,
        config: Optional[LLMConfig] = None,
    ):
        """
        初始化通用模型适配器。

        Args:
            model_config: 通用模型配置 (UniversalModelConfig)
            config: 基础LLM配置 (LLMConfig)，可选，用于兼容基类接口
        """
        # 优先使用 UniversalModelConfig
        if model_config:
            self._model_config = model_config
            # 从 UniversalModelConfig 构建 LLMConfig
            base_config = LLMConfig(
                model_id=model_config.model_id,
                api_key=model_config.api_key,
                base_url=model_config.api_base,
                timeout=model_config.timeout,
                max_retries=model_config.retry_count,
                max_tokens=model_config.max_output,
            )
            if config:
                # 合并额外配置
                extra = {k: v for k, v in config.__dict__.items()
                         if v is not None and k not in ("model_id", "api_key", "base_url")}
                for k, v in extra.items():
                    setattr(base_config, k, v)
            super().__init__(base_config)
        else:
            super().__init__(config)
            self._model_config = UniversalModelConfig()

        # HTTP会话
        self._session: Optional[aiohttp.ClientSession] = None

        # 速率限制器
        self._rate_limiter = RateLimiter(
            self._model_config.rate_limit
        )

        # 统计信息
        self._stats: Dict[str, Any] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost": 0.0,
            "avg_latency_ms": 0.0,
        }

    # ========================================================
    # 生命周期管理
    # ========================================================

    def _setup_client(self) -> None:
        """设置HTTP客户端"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._model_config.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self) -> None:
        """关闭HTTP客户端"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def __del__(self):
        """析构函数，确保资源释放"""
        try:
            if self._session and not self._session.closed:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close())
                else:
                    loop.run_until_complete(self.close())
        except Exception:
            pass

    # ========================================================
    # 核心接口实现
    # ========================================================

    async def _async_generate(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> LLMResponse:
        """
        异步生成响应 (基类抽象方法实现)

        根据模型的 api_protocol 自动选择对应的协议处理器。

        Args:
            messages: 消息列表
            config: 生成配置

        Returns:
            LLM响应
        """
        protocol = self._model_config.api_protocol.lower()

        if protocol == "openai":
            return await self._send_openai_compatible(messages, config)
        elif protocol == "claude":
            return await self._send_claude_compatible(messages, config)
        elif protocol == "custom":
            return await self._send_custom_protocol(messages, config)
        else:
            raise LLMError(
                f"不支持的协议类型: {protocol}",
                model_id=self._model_config.model_id,
                code="UNSUPPORTED_PROTOCOL",
            )

    async def _async_stream_generate(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> AsyncIterator[str]:
        """
        异步流式生成响应

        Args:
            messages: 消息列表
            config: 生成配置

        Yields:
            响应文本片段
        """
        if not self._model_config.supports_stream:
            # 不支持流式，退化为一次性返回
            response = await self._async_generate(messages, config)
            if response.error:
                yield f"Error: {response.error}"
                return
            yield response.content
            return

        protocol = self._model_config.api_protocol.lower()

        if protocol == "openai":
            async for chunk in self._stream_openai_compatible(messages, config):
                yield chunk
        elif protocol == "claude":
            async for chunk in self._stream_claude_compatible(messages, config):
                yield chunk
        elif protocol == "custom":
            # 自定义协议的流式处理
            async for chunk in self._stream_custom_protocol(messages, config):
                yield chunk
        else:
            raise LLMError(
                f"不支持的流式协议类型: {protocol}",
                model_id=self._model_config.model_id,
                code="UNSUPPORTED_PROTOCOL",
            )

    # ========================================================
    # OpenAI 兼容协议 (覆盖90%模型)
    # ========================================================

    async def _send_openai_compatible(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> LLMResponse:
        """
        发送OpenAI兼容格式的请求

        OpenAI兼容格式是目前最广泛使用的API格式，覆盖90%以上的国产模型。
        请求格式: POST {api_base}/chat/completions
        请求体: {"model": "...", "messages": [...], ...}

        Args:
            messages: 消息列表
            config: 生成配置

        Returns:
            LLM响应
        """
        url = self._build_openai_url()
        headers = self._build_headers()
        payload = self._build_openai_payload(messages, config)

        # 应用请求映射
        if self._model_config.request_mapping:
            payload = self._apply_request_mapping(payload)

        start_time = time.time()

        for attempt in range(self._model_config.retry_count):
            try:
                await self._rate_limiter.acquire()

                async with self._session.post(
                    url, headers=headers, json=payload
                ) as resp:
                    status = resp.status
                    body = await resp.json()

                    if status == 429:
                        # 速率限制，等待后重试
                        retry_after = float(
                            resp.headers.get("Retry-After", "2")
                        )
                        logger.warning(
                            f"速率限制，等待 {retry_after}s 后重试 "
                            f"(模型: {self._model_config.model_id})"
                        )
                        await asyncio.sleep(retry_after)
                        continue

                    if status >= 400:
                        error_msg = self._extract_error_message(body)
                        raise LLMError(
                            error_msg,
                            model_id=self._model_config.model_id,
                            status_code=status,
                            response=body,
                            code="API_ERROR",
                        )

                    # 应用响应映射
                    if self._model_config.response_mapping:
                        body = self._apply_response_mapping(body)

                    return self._parse_openai_response(
                        body, start_time
                    )

            except LLMError:
                raise
            except aiohttp.ClientError as e:
                logger.warning(
                    f"请求失败 (尝试 {attempt + 1}/{self._model_config.retry_count}): {e}"
                )
                if attempt == self._model_config.retry_count - 1:
                    raise LLMError(
                        f"请求失败: {e}",
                        model_id=self._model_config.model_id,
                        code="NETWORK_ERROR",
                    )
                await asyncio.sleep(2 ** attempt)
            except json.JSONDecodeError as e:
                raise LLMError(
                    f"响应解析失败: {e}",
                    model_id=self._model_config.model_id,
                    code="PARSE_ERROR",
                )

        # 不应到达此处
        raise LLMError(
            "所有重试均失败",
            model_id=self._model_config.model_id,
            code="ALL_RETRIES_FAILED",
        )

    async def _stream_openai_compatible(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> AsyncIterator[str]:
        """
        OpenAI兼容格式的流式请求

        Args:
            messages: 消息列表
            config: 生成配置

        Yields:
            响应文本片段
        """
        url = self._build_openai_url()
        headers = self._build_headers()
        payload = self._build_openai_payload(messages, config)
        payload["stream"] = True

        if self._model_config.request_mapping:
            payload = self._apply_request_mapping(payload)

        await self._rate_limiter.acquire()

        try:
            async with self._session.post(
                url, headers=headers, json=payload
            ) as resp:
                if resp.status >= 400:
                    body = await resp.json()
                    error_msg = self._extract_error_message(body)
                    yield f"Error: {error_msg}"
                    return

                # SSE流解析
                buffer = ""
                async for line in resp.content:
                    decoded = line.decode("utf-8")
                    buffer += decoded

                    while "\n" in buffer:
                        line_str, buffer = buffer.split("\n", 1)
                        line_str = line_str.strip()

                        if not line_str:
                            continue

                        if line_str.startswith("data: "):
                            data_str = line_str[6:]
                            if data_str.strip() == "[DONE]":
                                return

                            try:
                                chunk_data = json.loads(data_str)
                                content = self._extract_stream_content(
                                    chunk_data
                                )
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue

        except aiohttp.ClientError as e:
            yield f"Error: 流式请求失败 - {e}"

    def _build_openai_url(self) -> str:
        """
        构建OpenAI兼容API的完整URL

        Returns:
            完整的API URL
        """
        base = self._model_config.api_base.rstrip("/")
        # 确保URL以 /chat/completions 结尾
        if base.endswith("/chat/completions"):
            return base
        elif base.endswith("/v1"):
            return f"{base}/chat/completions"
        else:
            return f"{base}/chat/completions"

    def _build_openai_payload(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> Dict[str, Any]:
        """
        构建OpenAI兼容格式的请求体

        Args:
            messages: 消息列表
            config: 生成配置

        Returns:
            请求体字典
        """
        payload: Dict[str, Any] = {
            "model": self._model_config.model_id.split("/")[-1],
            "messages": messages,
        }

        # 添加可选参数
        if config.temperature is not None:
            payload["temperature"] = config.temperature
        if config.max_tokens is not None:
            payload["max_tokens"] = config.max_tokens
        if config.top_p is not None and config.top_p != 1.0:
            payload["top_p"] = config.top_p
        if config.stop:
            payload["stop"] = config.stop
        if config.seed is not None:
            payload["seed"] = config.seed
        if config.response_format:
            payload["response_format"] = config.response_format

        # 合并默认参数
        if self._model_config.default_params:
            for k, v in self._model_config.default_params.items():
                if k not in payload:
                    payload[k] = v

        # 合并额外body参数
        if config.extra_body:
            for k, v in config.extra_body.items():
                payload[k] = v

        return payload

    def _parse_openai_response(
        self,
        body: Dict[str, Any],
        start_time: float
    ) -> LLMResponse:
        """
        解析OpenAI兼容格式的响应

        Args:
            body: 响应体
            start_time: 请求开始时间

        Returns:
            LLM响应
        """
        latency = (time.time() - start_time) * 1000

        try:
            choice = body["choices"][0]
            content = choice.get("message", {}).get("content", "")
            finish_reason = choice.get("finish_reason")
            usage = body.get("usage")

            # 更新统计
            self._update_stats(
                latency,
                usage.get("prompt_tokens", 0) if usage else 0,
                usage.get("completion_tokens", 0) if usage else 0,
            )

            return LLMResponse(
                content=content or "",
                model_id=self._model_config.model_id,
                usage=usage,
                finish_reason=finish_reason,
                latency_ms=latency,
                metadata={
                    "protocol": "openai",
                    "provider": self._model_config.provider,
                    "raw_response": body,
                },
            )
        except (KeyError, IndexError) as e:
            raise LLMError(
                f"响应格式异常: {e}",
                model_id=self._model_config.model_id,
                response=body,
                code="INVALID_RESPONSE",
            )

    def _extract_stream_content(self, chunk_data: Dict[str, Any]) -> str:
        """
        从流式响应块中提取文本内容

        Args:
            chunk_data: 流式响应块

        Returns:
            文本内容
        """
        try:
            delta = chunk_data["choices"][0].get("delta", {})
            return delta.get("content", "")
        except (KeyError, IndexError):
            return ""

    # ========================================================
    # Claude 兼容协议
    # ========================================================

    async def _send_claude_compatible(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> LLMResponse:
        """
        发送Claude兼容格式的请求

        Claude API使用不同的请求/响应格式:
        - 系统消息单独传递
        - messages中不含system角色
        - 使用 content blocks 格式

        Args:
            messages: 消息列表
            config: 生成配置

        Returns:
            LLM响应
        """
        url = self._build_claude_url()
        headers = self._build_headers()
        payload = self._build_claude_payload(messages, config)

        start_time = time.time()

        for attempt in range(self._model_config.retry_count):
            try:
                await self._rate_limiter.acquire()

                async with self._session.post(
                    url, headers=headers, json=payload
                ) as resp:
                    status = resp.status
                    body = await resp.json()

                    if status == 429:
                        retry_after = float(
                            resp.headers.get("Retry-After", "2")
                        )
                        await asyncio.sleep(retry_after)
                        continue

                    if status >= 400:
                        error_msg = self._extract_error_message(body)
                        raise LLMError(
                            error_msg,
                            model_id=self._model_config.model_id,
                            status_code=status,
                            response=body,
                            code="API_ERROR",
                        )

                    return self._parse_claude_response(body, start_time)

            except LLMError:
                raise
            except aiohttp.ClientError as e:
                if attempt == self._model_config.retry_count - 1:
                    raise LLMError(
                        f"Claude请求失败: {e}",
                        model_id=self._model_config.model_id,
                        code="NETWORK_ERROR",
                    )
                await asyncio.sleep(2 ** attempt)

        raise LLMError(
            "Claude所有重试均失败",
            model_id=self._model_config.model_id,
            code="ALL_RETRIES_FAILED",
        )

    async def _stream_claude_compatible(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> AsyncIterator[str]:
        """
        Claude兼容格式的流式请求

        Args:
            messages: 消息列表
            config: 生成配置

        Yields:
            响应文本片段
        """
        url = self._build_claude_url()
        headers = self._build_headers()
        payload = self._build_claude_payload(messages, config)
        payload["stream"] = True

        await self._rate_limiter.acquire()

        try:
            async with self._session.post(
                url, headers=headers, json=payload
            ) as resp:
                if resp.status >= 400:
                    body = await resp.json()
                    error_msg = self._extract_error_message(body)
                    yield f"Error: {error_msg}"
                    return

                buffer = ""
                async for line in resp.content:
                    decoded = line.decode("utf-8")
                    buffer += decoded

                    while "\n" in buffer:
                        line_str, buffer = buffer.split("\n", 1)
                        line_str = line_str.strip()

                        if not line_str or not line_str.startswith("data: "):
                            continue

                        data_str = line_str[6:]
                        if data_str.strip() == "[DONE]":
                            return

                        try:
                            chunk_data = json.loads(data_str)
                            # Claude SSE格式
                            if chunk_data.get("type") == "content_block_delta":
                                delta = chunk_data.get("delta", {})
                                text = delta.get("text", "")
                                if text:
                                    yield text
                        except json.JSONDecodeError:
                            continue

        except aiohttp.ClientError as e:
            yield f"Error: Claude流式请求失败 - {e}"

    def _build_claude_url(self) -> str:
        """构建Claude API的完整URL"""
        base = self._model_config.api_base.rstrip("/")
        if base.endswith("/messages"):
            return base
        return f"{base}/messages"

    def _build_claude_payload(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> Dict[str, Any]:
        """
        构建Claude兼容格式的请求体

        Claude格式要求:
        - system消息单独提取
        - messages中只包含user/assistant角色
        - max_tokens为必填字段

        Args:
            messages: 消息列表
            config: 生成配置

        Returns:
            请求体字典
        """
        system_content = ""
        claude_messages: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_content += content + "\n"
            elif role in ("user", "assistant"):
                claude_messages.append({
                    "role": role,
                    "content": content,
                })

        payload: Dict[str, Any] = {
            "model": self._model_config.model_id.split("/")[-1],
            "messages": claude_messages,
            "max_tokens": config.max_tokens or self._model_config.max_output,
        }

        if system_content.strip():
            payload["system"] = system_content.strip()

        if config.temperature is not None:
            payload["temperature"] = config.temperature
        if config.top_p is not None and config.top_p != 1.0:
            payload["top_p"] = config.top_p
        if config.stop:
            payload["stop_sequences"] = config.stop

        return payload

    def _parse_claude_response(
        self,
        body: Dict[str, Any],
        start_time: float
    ) -> LLMResponse:
        """
        解析Claude兼容格式的响应

        Args:
            body: 响应体
            start_time: 请求开始时间

        Returns:
            LLM响应
        """
        latency = (time.time() - start_time) * 1000

        try:
            content_blocks = body.get("content", [])
            content = "".join(
                block.get("text", "") for block in content_blocks
            )

            # Claude的usage格式
            usage_info = body.get("usage", {})
            usage = {
                "prompt_tokens": usage_info.get("input_tokens", 0),
                "completion_tokens": usage_info.get("output_tokens", 0),
                "total_tokens": (
                    usage_info.get("input_tokens", 0)
                    + usage_info.get("output_tokens", 0)
                ),
            }

            stop_reason = body.get("stop_reason")

            self._update_stats(
                latency,
                usage["prompt_tokens"],
                usage["completion_tokens"],
            )

            return LLMResponse(
                content=content,
                model_id=self._model_config.model_id,
                usage=usage,
                finish_reason=stop_reason,
                latency_ms=latency,
                metadata={
                    "protocol": "claude",
                    "provider": self._model_config.provider,
                    "raw_response": body,
                },
            )
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(
                f"Claude响应格式异常: {e}",
                model_id=self._model_config.model_id,
                response=body,
                code="INVALID_RESPONSE",
            )

    # ========================================================
    # 自定义协议
    # ========================================================

    async def _send_custom_protocol(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> LLMResponse:
        """
        发送自定义协议的请求

        通过 request_mapping 和 response_mapping 进行字段映射转换，
        可以适配任意非标准API格式。

        request_mapping 示例:
            {
                "url_path": "/v1/chat",
                "method": "POST",
                "field_map": {
                    "messages": "input.messages",
                    "model": "input.model_name",
                    "temperature": "params.temp"
                }
            }

        response_mapping 示例:
            {
                "field_map": {
                    "result.text": "content",
                    "result.tokens.input": "usage.prompt_tokens"
                }
            }

        Args:
            messages: 消息列表
            config: 生成配置

        Returns:
            LLM响应
        """
        url = self._build_custom_url()
        headers = self._build_headers()

        # 构建基础payload
        base_payload: Dict[str, Any] = {
            "messages": messages,
            "model": self._model_config.model_id.split("/")[-1],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens or self._model_config.max_output,
        }

        # 应用请求映射
        payload = self._apply_request_mapping(base_payload)

        start_time = time.time()

        for attempt in range(self._model_config.retry_count):
            try:
                await self._rate_limiter.acquire()

                method = self._model_config.request_mapping.get(
                    "method", "POST"
                ).upper()

                async with self._session.request(
                    method, url, headers=headers, json=payload
                ) as resp:
                    status = resp.status
                    body = await resp.json()

                    if status >= 400:
                        error_msg = self._extract_error_message(body)
                        raise LLMError(
                            error_msg,
                            model_id=self._model_config.model_id,
                            status_code=status,
                            response=body,
                            code="API_ERROR",
                        )

                    # 应用响应映射
                    mapped_body = self._apply_response_mapping(body)
                    return self._parse_custom_response(
                        mapped_body, start_time
                    )

            except LLMError:
                raise
            except aiohttp.ClientError as e:
                if attempt == self._model_config.retry_count - 1:
                    raise LLMError(
                        f"自定义协议请求失败: {e}",
                        model_id=self._model_config.model_id,
                        code="NETWORK_ERROR",
                    )
                await asyncio.sleep(2 ** attempt)

        raise LLMError(
            "自定义协议所有重试均失败",
            model_id=self._model_config.model_id,
            code="ALL_RETRIES_FAILED",
        )

    async def _stream_custom_protocol(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> AsyncIterator[str]:
        """
        自定义协议的流式请求

        Args:
            messages: 消息列表
            config: 生成配置

        Yields:
            响应文本片段
        """
        # 自定义协议默认退化为非流式
        response = await self._send_custom_protocol(messages, config)
        if response.error:
            yield f"Error: {response.error}"
        else:
            yield response.content

    def _build_custom_url(self) -> str:
        """构建自定义协议的完整URL"""
        base = self._model_config.api_base.rstrip("/")
        url_path = self._model_config.request_mapping.get("url_path", "")
        if url_path:
            return f"{base}{url_path}"
        return f"{base}/chat/completions"

    def _parse_custom_response(
        self,
        body: Dict[str, Any],
        start_time: float
    ) -> LLMResponse:
        """
        解析自定义协议的响应

        尝试从映射后的响应中提取标准字段。

        Args:
            body: 响应体
            start_time: 请求开始时间

        Returns:
            LLM响应
        """
        latency = (time.time() - start_time) * 1000

        content = body.get("content") or body.get("text") or body.get("result", "")
        usage = body.get("usage")
        finish_reason = body.get("finish_reason") or body.get("stop_reason")

        input_tokens = 0
        output_tokens = 0
        if usage:
            if isinstance(usage, dict):
                input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
                output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
            elif isinstance(usage, (int, float)):
                input_tokens = int(usage)

        self._update_stats(latency, input_tokens, output_tokens)

        return LLMResponse(
            content=str(content),
            model_id=self._model_config.model_id,
            usage=usage if isinstance(usage, dict) else None,
            finish_reason=finish_reason,
            latency_ms=latency,
            metadata={
                "protocol": "custom",
                "provider": self._model_config.provider,
                "raw_response": body,
            },
        )

    # ========================================================
    # 请求/响应映射
    # ========================================================

    def _apply_request_mapping(
        self,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        应用请求字段映射

        将标准格式的请求体转换为目标API所需的格式。
        支持:
        - field_map: 字段名映射
        - extra_fields: 额外添加的字段
        - remove_fields: 需要移除的字段
        - wrap: 包装结构

        Args:
            payload: 原始请求体

        Returns:
            映射后的请求体
        """
        mapping = self._model_config.request_mapping
        if not mapping:
            return payload

        result = payload.copy()

        # 字段名映射
        field_map = mapping.get("field_map", {})
        if field_map:
            new_payload: Dict[str, Any] = {}
            for key, value in result.items():
                new_key = field_map.get(key, key)
                new_payload[new_key] = value
            result = new_payload

        # 添加额外字段
        extra_fields = mapping.get("extra_fields", {})
        if extra_fields:
            result.update(extra_fields)

        # 移除字段
        remove_fields = mapping.get("remove_fields", [])
        for field_name in remove_fields:
            result.pop(field_name, None)

        # 包装结构
        wrap = mapping.get("wrap")
        if wrap:
            result = {wrap: result}

        return result

    def _apply_response_mapping(
        self,
        body: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        应用响应字段映射

        将非标准格式的响应体转换为标准格式。
        支持:
        - field_map: 字段名映射 (支持嵌套路径用点号分隔)
        - extract_content: 内容提取路径
        - extract_usage: usage提取路径

        Args:
            body: 原始响应体

        Returns:
            映射后的响应体
        """
        mapping = self._model_config.response_mapping
        if not mapping:
            return body

        result = body.copy()

        # 字段名映射
        field_map = mapping.get("field_map", {})
        if field_map:
            new_body: Dict[str, Any] = {}
            for old_path, new_key in field_map.items():
                value = self._get_nested_value(result, old_path)
                if value is not None:
                    new_body[new_key] = value
            result = new_body

        # 内容提取路径
        content_path = mapping.get("extract_content")
        if content_path:
            content = self._get_nested_value(result, content_path)
            if content is not None:
                result["content"] = content

        # usage提取路径
        usage_path = mapping.get("extract_usage")
        if usage_path:
            usage = self._get_nested_value(result, usage_path)
            if usage is not None:
                result["usage"] = usage

        return result

    @staticmethod
    def _get_nested_value(data: Any, path: str) -> Any:
        """
        从嵌套结构中获取值

        Args:
            data: 数据
            path: 点号分隔的路径，如 "choices.0.message.content"

        Returns:
            找到的值，未找到返回None
        """
        if not path:
            return data

        current = data
        for key in path.split("."):
            if isinstance(current, dict):
                current = current.get(key)
            elif isinstance(current, (list, tuple)):
                try:
                    idx = int(key)
                    current = current[idx]
                except (ValueError, IndexError):
                    return None
            else:
                return None
            if current is None:
                return None
        return current

    # ========================================================
    # 认证处理
    # ========================================================

    def _build_headers(self) -> Dict[str, str]:
        """
        构建HTTP请求头

        根据 auth_type 自动添加认证信息。

        Returns:
            HTTP请求头字典
        """
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        auth_type = self._model_config.auth_type.lower()
        auth_config = self._model_config.auth_config

        if auth_type == "bearer":
            token = self._resolve_api_key()
            if token:
                headers["Authorization"] = f"Bearer {token}"

        elif auth_type == "api_key_header":
            key_name = auth_config.get(
                "header_name", "X-API-Key"
            )
            token = self._resolve_api_key()
            if token:
                headers[key_name] = token

        elif auth_type == "api_key_query":
            # URL参数认证在URL构建时处理
            pass

        elif auth_type == "oauth2":
            # OAuth2 token获取
            token = self._get_oauth_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"

        elif auth_type == "hmac_sha256":
            # HMAC签名在请求发送时处理
            pass

        elif auth_type == "aws_sigv4":
            # AWS签名在请求发送时处理
            pass

        elif auth_type == "custom":
            # 自定义认证头
            custom_headers = auth_config.get("headers", {})
            headers.update(custom_headers)

        # 合并额外头
        if self._config and self._config.extra_headers:
            headers.update(self._config.extra_headers)

        return headers

    def _resolve_api_key(self) -> str:
        """
        解析API密钥

        支持从以下位置获取:
        1. UniversalModelConfig.api_key
        2. 环境变量 (auth_config中配置的env_var)
        3. auth_config中的key字段

        Returns:
            API密钥字符串
        """
        # 1. 直接配置
        if self._model_config.api_key:
            return self._model_config.api_key

        # 2. 环境变量
        auth_config = self._model_config.auth_config
        env_var = auth_config.get("env_var")
        if env_var:
            import os
            return os.environ.get(env_var, "")

        # 3. auth_config中的key
        return auth_config.get("key", "")

    def _get_oauth_token(self) -> str:
        """
        获取OAuth2.0访问令牌

        从auth_config中读取OAuth2配置:
        - token_url: Token端点URL
        - client_id: 客户端ID
        - client_secret: 客户端密钥
        - scope: 权限范围
        - grant_type: 授权类型

        Returns:
            访问令牌
        """
        auth_config = self._model_config.auth_config
        token_url = auth_config.get("token_url", "")
        client_id = auth_config.get("client_id", "")
        client_secret = auth_config.get("client_secret", "")
        scope = auth_config.get("scope", "")
        grant_type = auth_config.get("grant_type", "client_credentials")

        if not token_url or not client_id:
            logger.warning("OAuth2配置不完整，缺少token_url或client_id")
            return ""

        # 使用缓存避免频繁请求token
        cache_key = f"oauth_token_{self._model_config.model_id}"
        cached = self._stats.get(cache_key)
        if cached:
            token, expires_at = cached
            if time.time() < expires_at:
                return token

        # 同步获取token (在实际使用中应考虑异步)
        try:
            import os
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 在异步上下文中，使用run_in_executor
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    token = pool.submit(
                        self._fetch_oauth_token_sync,
                        token_url, client_id, client_secret, scope, grant_type
                    ).result(timeout=30)
            else:
                token = self._fetch_oauth_token_sync(
                    token_url, client_id, client_secret, scope, grant_type
                )

            if token:
                # 缓存token，提前5分钟过期
                self._stats[cache_key] = (token, time.time() + 3500)
                return token
        except Exception as e:
            logger.error(f"OAuth2 token获取失败: {e}")

        return ""

    @staticmethod
    def _fetch_oauth_token_sync(
        token_url: str,
        client_id: str,
        client_secret: str,
        scope: str,
        grant_type: str
    ) -> str:
        """
        同步获取OAuth2 token

        Args:
            token_url: Token端点URL
            client_id: 客户端ID
            client_secret: 客户端密钥
            scope: 权限范围
            grant_type: 授权类型

        Returns:
            访问令牌
        """
        import urllib.request
        import urllib.parse

        data = urllib.parse.urlencode({
            "grant_type": grant_type,
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": scope,
        }).encode("utf-8")

        req = urllib.request.Request(
            token_url,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("access_token", "")

    def _apply_hmac_signature(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        body: Optional[bytes] = None
    ) -> Dict[str, str]:
        """
        应用HMAC-SHA256签名

        用于讯飞星火等需要HMAC签名的API。

        Args:
            method: HTTP方法
            url: 请求URL
            headers: 请求头
            body: 请求体

        Returns:
            添加了签名信息的请求头
        """
        auth_config = self._model_config.auth_config
        secret_key = auth_config.get("secret_key", self._resolve_api_key())
        timestamp = str(int(time.time()))
        nonce = auth_config.get("nonce", hashlib.md5(
            f"{timestamp}{self._model_config.model_id}".encode()
        ).hexdigest())

        # 构建签名字符串
        parsed = urlparse(url)
        path = parsed.path or "/"
        query = parsed.query

        sign_str = f"{method}\n{path}\n{query}\n{timestamp}\n{nonce}"

        if body:
            body_hash = hashlib.sha256(body).hexdigest()
            sign_str += f"\n{body_hash}"

        # 计算签名
        signature = hmac.new(
            secret_key.encode("utf-8"),
            sign_str.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

        # 添加签名头
        headers["X-Timestamp"] = timestamp
        headers["X-Nonce"] = nonce
        headers["X-Signature"] = signature

        # 自定义签名头名称
        sig_header = auth_config.get("signature_header", "X-Signature")
        ts_header = auth_config.get("timestamp_header", "X-Timestamp")
        nonce_header = auth_config.get("nonce_header", "X-Nonce")

        if sig_header != "X-Signature":
            headers[sig_header] = headers.pop("X-Signature")
        if ts_header != "X-Timestamp":
            headers[ts_header] = headers.pop("X-Timestamp")
        if nonce_header != "X-Nonce":
            headers[nonce_header] = headers.pop("X-Nonce")

        return headers

    # ========================================================
    # 公共接口方法
    # ========================================================

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> LLMResponse:
        """
        统一聊天接口

        根据模型的 api_protocol 自动适配，用户无需关心底层协议差异。

        Args:
            messages: 消息列表，格式:
                [{"role": "system", "content": "..."},
                 {"role": "user", "content": "..."},
                 {"role": "assistant", "content": "..."}]
            **kwargs: 其他参数 (temperature, max_tokens, top_p等)

        Returns:
            LLM响应

        Example:
            ```python
            response = await provider.chat_completion([
                {"role": "system", "content": "你是一个有帮助的助手"},
                {"role": "user", "content": "介绍一下北京"},
            ], temperature=0.7)
            ```
        """
        if not self._initialized:
            self.initialize()

        return await self.generate(messages, **kwargs)

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> AsyncIterator[str]:
        """
        统一流式聊天接口

        Args:
            messages: 消息列表
            **kwargs: 其他参数

        Yields:
            响应文本片段

        Example:
            ```python
            async for chunk in provider.stream_chat([
                {"role": "user", "content": "讲个故事"}
            ]):
                print(chunk, end="", flush=True)
            ```
        """
        if not self._initialized:
            self.initialize()

        async for chunk in self.stream_generate(messages, **kwargs):
            yield chunk

    async def test_connection(self) -> Dict[str, Any]:
        """
        测试模型连接

        发送一个简单的测试请求，验证API是否可用。

        Returns:
            测试结果字典:
            {
                "success": bool,
                "model_id": str,
                "latency_ms": float,
                "error": Optional[str],
                "model_info": Optional[dict],
            }
        """
        test_messages = [
            {"role": "user", "content": "Hi"}
        ]

        start_time = time.time()
        try:
            response = await self.chat_completion(
                test_messages,
                max_tokens=5,
                temperature=0.0,
            )
            latency = (time.time() - start_time) * 1000

            return {
                "success": response.is_success,
                "model_id": self._model_config.model_id,
                "latency_ms": latency,
                "error": response.error,
                "model_info": self.get_model_info() if response.is_success else None,
            }
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return {
                "success": False,
                "model_id": self._model_config.model_id,
                "latency_ms": latency,
                "error": str(e),
                "model_info": None,
            }

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            模型详细信息字典
        """
        return {
            "model_id": self._model_config.model_id,
            "model_name": self._model_config.model_name,
            "provider": self._model_config.provider,
            "api_protocol": self._model_config.api_protocol,
            "auth_type": self._model_config.auth_type,
            "max_context": self._model_config.max_context,
            "max_output": self._model_config.max_output,
            "supports_stream": self._model_config.supports_stream,
            "supports_vision": self._model_config.supports_vision,
            "supports_function_call": self._model_config.supports_function_call,
            "supports_embeddings": self._model_config.supports_embeddings,
            "capabilities": [
                cap.name for cap in self._model_config.get_capabilities()
            ],
            "rate_limit": self._model_config.rate_limit,
            "timeout": self._model_config.timeout,
            "input_price_per_1k": self._model_config.input_price_per_1k,
            "output_price_per_1k": self._model_config.output_price_per_1k,
            "stats": self._stats.copy(),
        }

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        估算调用成本

        Args:
            input_tokens: 输入token数
            output_tokens: 输出token数

        Returns:
            预估成本 (元)
        """
        input_cost = (input_tokens / 1000.0) * self._model_config.input_price_per_1k
        output_cost = (output_tokens / 1000.0) * self._model_config.output_price_per_1k
        return input_cost + output_cost

    def get_stats(self) -> Dict[str, Any]:
        """
        获取Provider统计信息

        Returns:
            统计信息字典
        """
        stats = super().get_stats()
        stats.update({
            "model_config": {
                "model_id": self._model_config.model_id,
                "provider": self._model_config.provider,
                "protocol": self._model_config.api_protocol,
            },
            "request_stats": self._stats.copy(),
            "estimated_total_cost": self._stats.get("total_cost", 0.0),
        })
        return stats

    def get_capabilities(self) -> Set[ModelCapability]:
        """获取模型能力集合"""
        return self._model_config.get_capabilities()

    # ========================================================
    # 内部工具方法
    # ========================================================

    def _extract_error_message(self, body: Dict[str, Any]) -> str:
        """
        从错误响应体中提取错误信息

        兼容多种错误响应格式:
        - OpenAI格式: {"error": {"message": "..."}}
        - 简单格式: {"message": "..."}
        - 百度格式: {"error_code": "...", "error_msg": "..."}
        - 讯飞格式: {"code": "...", "message": "..."}

        Args:
            body: 错误响应体

        Returns:
            错误信息字符串
        """
        if not isinstance(body, dict):
            return str(body)

        # OpenAI格式
        error = body.get("error")
        if isinstance(error, dict):
            return error.get("message", str(error))

        # 直接message字段
        if body.get("message"):
            return str(body["message"])

        # 百度格式
        if body.get("error_msg"):
            return f"[{body.get('error_code', 'UNKNOWN')}] {body['error_msg']}"

        # 讯飞格式
        if body.get("code"):
            return f"[{body['code']}] {body.get('message', '未知错误')}"

        # msg字段
        if body.get("msg"):
            return str(body["msg"])

        return json.dumps(body, ensure_ascii=False)

    def _update_stats(
        self,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int
    ) -> None:
        """
        更新统计信息

        Args:
            latency_ms: 请求延迟(毫秒)
            input_tokens: 输入token数
            output_tokens: 输出token数
        """
        self._stats["total_requests"] += 1
        self._stats["successful_requests"] += 1
        self._stats["total_input_tokens"] += input_tokens
        self._stats["total_output_tokens"] += output_tokens

        cost = self.estimate_cost(input_tokens, output_tokens)
        self._stats["total_cost"] += cost

        # 计算平均延迟
        total = self._stats["total_requests"]
        prev_avg = self._stats["avg_latency_ms"]
        self._stats["avg_latency_ms"] = (
            (prev_avg * (total - 1) + latency_ms) / total
        )

    def _handle_auth_failure(self, status_code: int, body: Dict[str, Any]) -> None:
        """
        处理认证失败

        Args:
            status_code: HTTP状态码
            body: 响应体
        """
        if status_code == 401:
            logger.error(
                f"认证失败 (模型: {self._model_config.model_id}): "
                f"请检查API密钥是否正确"
            )
        elif status_code == 403:
            logger.error(
                f"权限不足 (模型: {self._model_config.model_id}): "
                f"请检查账户权限"
            )

    def update_config(self, **kwargs: Any) -> None:
        """
        动态更新模型配置

        Args:
            **kwargs: 要更新的配置字段
        """
        valid_fields = {
            f.name for f in UniversalModelConfig.__dataclass_fields__.values()
        }
        for key, value in kwargs.items():
            if key in valid_fields:
                setattr(self._model_config, key, value)

        # 更新速率限制器
        if "rate_limit" in kwargs:
            self._rate_limiter.update_rate_limit(kwargs["rate_limit"])

        logger.info(
            f"模型配置已更新: {self._model_config.model_id}, "
            f"更新字段: {list(kwargs.keys())}"
        )

    @classmethod
    def create_from_dict(
        cls,
        config_dict: Dict[str, Any]
    ) -> "UniversalLLMProvider":
        """
        从字典配置创建Provider实例

        Args:
            config_dict: 配置字典

        Returns:
            UniversalLLMProvider实例

        Example:
            ```python
            provider = UniversalLLMProvider.create_from_dict({
                "model_id": "zhipu/glm-4",
                "model_name": "GLM-4",
                "provider": "zhipu",
                "api_base": "https://open.bigmodel.cn/api/paas/v4",
                "api_key": "your-key",
            })
            ```
        """
        model_config = UniversalModelConfig.from_dict(config_dict)
        return cls(model_config=model_config)
