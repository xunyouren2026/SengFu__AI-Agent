"""
百度文心一言 (Wenxin) Provider

百度千帆大模型平台文心一言系列模型的适配器。

支持的模型:
- ERNIE-Bot-4 (ernie-4.0-8k)
- ERNIE-Bot-3.5 (ernie-3.5-8k)
- ERNIE-Bot-turbo (ernie-3.5-turbo)
- ERNIE-Speed (ernie-speed)
- ERNIE-Lite (ernie-lite)

API文档: https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Nlks5zkzu

Author: AGI Team
Version: 1.0.0
"""

import time
import logging
import json
import asyncio
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Any, AsyncIterator, Set
)
from .base import (
    BaseLLMProvider, LLMConfig, LLMResponse,
    ModelCapability, LLMError
)

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class WenxinConfig(LLMConfig):
    """
    百度文心一言配置

    Attributes:
        api_key: 百度千帆平台 API Key
        secret_key: 百度千帆平台 Secret Key (用于获取access_token)
        access_token: 已有的access_token (可选, 若提供则跳过获取)
        enable_search: 是否启用搜索增强
        enable_citation: 是否启用引用标注
        knowledge_base_id: 知识库ID (知识库增强)
        token_refresh_margin: token过期提前刷新时间(秒)
    """
    secret_key: Optional[str] = None
    access_token: Optional[str] = None
    enable_search: bool = False
    enable_citation: bool = False
    knowledge_base_id: Optional[str] = None
    token_refresh_margin: int = 300


class WenxinProvider(BaseLLMProvider):
    """
    百度文心一言系列Provider

    百度千帆大模型平台提供的文心一言系列大语言模型适配器。
    支持ERNIE系列模型的对话生成、流式输出和向量嵌入。

    Features:
        - 支持ERNIE-Bot-4/3.5/turbo/Speed/Lite全系列
        - 支持搜索增强 (enable_search)
        - 支持知识库增强 (knowledge_base_id)
        - 支持引用标注 (enable_citation)
        - access_token自动获取与过期刷新
        - 限流自动重试

    Example:
        ```python
        provider = WenxinProvider(WenxinConfig(
            model_id="ernie-4.0-8k",
            api_key="your_api_key",
            secret_key="your_secret_key",
        ))

        response = await provider.generate([
            {"role": "user", "content": "你好"}
        ])
        print(response.content)
        ```
    """

    PROVIDER_NAME = "wenxin"
    SUPPORTED_MODELS = {
        "ernie-4.0-8k",
        "ernie-3.5-8k",
        "ernie-3.5-turbo",
        "ernie-speed",
        "ernie-lite",
        "ernie-4.0-turbo-8k",
        "ernie-longtext",
        "ernie-character-8k",
        "ernie-novel-8k",
    }
    DEFAULT_MODEL = "ernie-4.0-8k"

    # 模型ID到API端点路径的映射
    _MODEL_ENDPOINT_MAP: Dict[str, str] = {
        "ernie-4.0-8k": "completions_pro",
        "ernie-4.0-turbo-8k": "ernie-4.0-turbo-8k",
        "ernie-3.5-8k": "completions",
        "ernie-3.5-turbo": "ernie-3.5-turbo",
        "ernie-speed": "ernie_speed",
        "ernie-lite": "ernie-lite",
        "ernie-longtext": "ernie-longtext",
        "ernie-character-8k": "ernie-char-8k",
        "ernie-novel-8k": "ernie-novel-8k",
    }

    _TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
    _API_BASE = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat"
    _EMBEDDING_BASE = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings"

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        初始化百度文心一言Provider。

        Args:
            config: 文心一言配置, 推荐使用WenxinConfig
        """
        super().__init__(config)
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._client: Optional[httpx.AsyncClient] = None

    def get_capabilities(self) -> Set[ModelCapability]:
        """获取模型能力"""
        capabilities = {
            ModelCapability.STREAMING,
            ModelCapability.JSON_MODE,
        }
        return capabilities

    def _setup_client(self) -> None:
        """设置HTTP客户端"""
        if HTTPX_AVAILABLE:
            timeout = self._config.timeout if self._config else 60.0
            self._client = httpx.AsyncClient(timeout=timeout)

    def _validate_config(self) -> None:
        """验证配置"""
        super()._validate_config()
        if isinstance(self._config, WenxinConfig):
            if not self._config.secret_key and not self._config.access_token:
                raise ValueError(
                    "WenxinConfig requires either 'secret_key' or 'access_token'"
                )

    async def _get_access_token(self) -> str:
        """
        获取百度API的access_token。

        使用API Key和Secret Key通过OAuth2.0获取access_token。
        如果token尚未过期则复用已有token。

        Returns:
            access_token字符串

        Raises:
            LLMError: 获取token失败
        """
        if isinstance(self._config, WenxinConfig) and self._config.access_token:
            return self._config.access_token

        # 检查缓存的token是否仍然有效
        now = time.time()
        margin = 300
        if isinstance(self._config, WenxinConfig):
            margin = self._config.token_refresh_margin

        if self._access_token and now < self._token_expires_at - margin:
            return self._access_token

        api_key = self._config.api_key
        secret_key = self._config.secret_key if isinstance(self._config, WenxinConfig) else None

        if not api_key or not secret_key:
            raise LLMError(
                "api_key and secret_key are required to obtain access_token",
                model_id=self.model_id
            )

        params = {
            "grant_type": "client_credentials",
            "client_id": api_key,
            "client_secret": secret_key,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self._TOKEN_URL, params=params)

                if response.status_code != 200:
                    raise LLMError(
                        f"Failed to get access_token: HTTP {response.status_code} - {response.text}",
                        model_id=self.model_id,
                        status_code=response.status_code,
                    )

                result = response.json()

                if "error" in result:
                    raise LLMError(
                        f"Failed to get access_token: {result.get('error_description', result['error'])}",
                        code=result.get("error"),
                        model_id=self.model_id,
                    )

                self._access_token = result["access_token"]
                # 提前5分钟刷新
                self._token_expires_at = now + result.get("expires_in", 2592000)
                logger.info("Successfully obtained Baidu access_token")
                return self._access_token

        except httpx.TimeoutException:
            raise LLMError(
                "Timeout while obtaining Baidu access_token",
                model_id=self.model_id,
            )
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(
                f"Error obtaining Baidu access_token: {e}",
                model_id=self.model_id,
            )

    def _get_model_endpoint(self, model_id: str) -> str:
        """
        获取模型对应的API端点路径。

        Args:
            model_id: 模型ID

        Returns:
            API端点路径
        """
        endpoint = self._MODEL_ENDPOINT_MAP.get(model_id)
        if not endpoint:
            # 对于未明确映射的模型, 尝试直接使用model_id作为端点
            logger.warning(f"Unknown model endpoint for {model_id}, using model_id as endpoint")
            endpoint = model_id
        return endpoint

    def _convert_messages(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        转换消息格式为百度API要求的格式。

        百度文心API支持的消息角色: user, assistant
        system消息需要通过单独的system字段传递。

        Args:
            messages: 标准消息列表

        Returns:
            转换后的消息列表和system字符串
        """
        system_content = ""
        chat_messages = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                system_content += content + "\n"
            elif role in ("user", "assistant"):
                chat_messages.append({
                    "role": role,
                    "content": content,
                })
            else:
                logger.warning(f"Unsupported message role: {role}, skipping")

        return chat_messages, system_content.strip()

    async def _async_generate(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> LLMResponse:
        """
        异步生成响应。

        Args:
            messages: 消息列表
            config: 生成配置

        Returns:
            LLM响应
        """
        start_time = time.time()

        if not HTTPX_AVAILABLE:
            return LLMResponse(
                content="",
                model_id=config.model_id,
                error="httpx not available",
                latency_ms=(time.time() - start_time) * 1000,
            )

        try:
            access_token = await self._get_access_token()
        except LLMError as e:
            return LLMResponse(
                content="",
                model_id=config.model_id,
                error=f"Token error: {e.message}",
                latency_ms=(time.time() - start_time) * 1000,
            )

        endpoint = self._get_model_endpoint(config.model_id)
        url = f"{self._API_BASE}/{endpoint}"

        chat_messages, system_content = self._convert_messages(messages)

        payload: Dict[str, Any] = {
            "messages": chat_messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }

        if system_content:
            payload["system"] = system_content

        if config.max_tokens is not None:
            payload["max_output_tokens"] = config.max_tokens

        if config.stop:
            payload["stop"] = config.stop

        # 百度特色功能: 搜索增强
        if isinstance(config, WenxinConfig):
            if config.enable_search:
                payload["enable_search"] = True
            if config.enable_citation:
                payload["enable_citation"] = True

        params = {"access_token": access_token}

        # 带重试的请求
        last_error = None
        for attempt in range(config.max_retries):
            try:
                async with httpx.AsyncClient(timeout=config.timeout) as client:
                    response = await client.post(
                        url,
                        params=params,
                        json=payload,
                    )

                    if response.status_code == 401:
                        # token过期, 刷新后重试
                        logger.info("Access token expired, refreshing...")
                        self._access_token = None
                        try:
                            access_token = await self._get_access_token()
                            params["access_token"] = access_token
                            continue
                        except LLMError:
                            pass

                    if response.status_code == 429:
                        # 限流, 等待后重试
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limited, retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue

                    if response.status_code != 200:
                        error_body = response.text
                        try:
                            error_json = response.json()
                            error_msg = error_json.get("error_msg", error_body)
                        except (json.JSONDecodeError, ValueError):
                            error_msg = error_body

                        return LLMResponse(
                            content="",
                            model_id=config.model_id,
                            error=f"HTTP {response.status_code}: {error_msg}",
                            latency_ms=(time.time() - start_time) * 1000,
                        )

                    result = response.json()

                    content = result.get("result", "")
                    finish_reason = result.get("finish_reason", "unknown")
                    usage = result.get("usage")

                    # 构建标准usage格式
                    normalized_usage = None
                    if usage:
                        normalized_usage = {
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        }

                    metadata = {
                        "provider": "wenxin",
                        "model": config.model_id,
                    }

                    # 搜索增强结果
                    if "search_info" in result:
                        metadata["search_info"] = result["search_info"]

                    return LLMResponse(
                        content=content,
                        model_id=config.model_id,
                        usage=normalized_usage,
                        finish_reason=finish_reason,
                        latency_ms=(time.time() - start_time) * 1000,
                        metadata=metadata,
                    )

            except httpx.TimeoutException as e:
                last_error = f"Request timeout: {e}"
                logger.warning(f"Request timeout (attempt {attempt + 1}/{config.max_retries})")
                if attempt < config.max_retries - 1:
                    await asyncio.sleep(1)
                continue
            except Exception as e:
                last_error = str(e)
                break

        return LLMResponse(
            content="",
            model_id=config.model_id,
            error=last_error or "Unknown error",
            latency_ms=(time.time() - start_time) * 1000,
        )

    async def _async_stream_generate(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> AsyncIterator[str]:
        """
        流式生成响应。

        Args:
            messages: 消息列表
            config: 生成配置

        Yields:
            响应文本片段
        """
        if not HTTPX_AVAILABLE:
            yield "Error: httpx not available"
            return

        try:
            access_token = await self._get_access_token()
        except LLMError as e:
            yield f"Error: Token error: {e.message}"
            return

        endpoint = self._get_model_endpoint(config.model_id)
        url = f"{self._API_BASE}/{endpoint}"

        chat_messages, system_content = self._convert_messages(messages)

        payload: Dict[str, Any] = {
            "messages": chat_messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stream": True,
        }

        if system_content:
            payload["system"] = system_content

        if config.max_tokens is not None:
            payload["max_output_tokens"] = config.max_tokens

        if isinstance(config, WenxinConfig) and config.enable_search:
            payload["enable_search"] = True

        params = {"access_token": access_token}

        try:
            async with httpx.AsyncClient(timeout=config.timeout) as client:
                async with client.stream(
                    "POST",
                    url,
                    params=params,
                    json=payload,
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        yield f"Error: HTTP {response.status_code}: {error_text.decode()}"
                        return

                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue

                        if line.startswith("data: "):
                            data_str = line[6:]
                        else:
                            data_str = line

                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            chunk_data = json.loads(data_str)
                            content = chunk_data.get("result", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue

        except httpx.TimeoutException as e:
            yield f"Error: Stream timeout: {e}"
        except Exception as e:
            yield f"Error: {e}"

    async def embeddings(
        self,
        texts: List[str],
        model: str = "bge-large-zh",
        **kwargs
    ) -> List[List[float]]:
        """
        获取文本向量嵌入。

        Args:
            texts: 待嵌入的文本列表
            model: 嵌入模型名称 (bge-large-zh, bge-large-en等)
            **kwargs: 额外参数

        Returns:
            向量列表

        Raises:
            LLMError: 嵌入请求失败
        """
        if not HTTPX_AVAILABLE:
            raise LLMError("httpx not available", model_id=model)

        try:
            access_token = await self._get_access_token()
        except LLMError:
            raise

        url = f"{self._EMBEDDING_BASE}/{model}"
        params = {"access_token": access_token}

        payload = {
            "input": texts,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                url,
                params=params,
                json=payload,
            )

            if response.status_code != 200:
                raise LLMError(
                    f"Embedding request failed: HTTP {response.status_code} - {response.text}",
                    model_id=model,
                    status_code=response.status_code,
                )

            result = response.json()
            embeddings = [item["embedding"] for item in result.get("data", [])]
            return embeddings

    async def token_count(
        self,
        text: str,
        model: Optional[str] = None
    ) -> int:
        """
        估算文本的token数量。

        百度文心API没有专门的token计数接口,
        这里使用简单的字符级估算 (中文约1.5字符/token, 英文约4字符/token)。

        Args:
            text: 待计数的文本
            model: 模型名称 (未使用, 保留接口一致性)

        Returns:
            估算的token数量
        """
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        estimated_tokens = int(chinese_chars / 1.5 + other_chars / 4)
        return max(1, estimated_tokens)

    def get_stats(self) -> Dict[str, Any]:
        """获取Provider统计信息"""
        return {
            "provider": self.PROVIDER_NAME,
            "model_id": self.model_id,
            "initialized": self._initialized,
            "has_access_token": self._access_token is not None,
            "token_expires_at": self._token_expires_at,
            "supported_models": sorted(self.SUPPORTED_MODELS),
        }
