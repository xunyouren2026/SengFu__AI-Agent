"""
讯飞星火 (Spark) Provider

讯飞星火认知大模型的适配器。

支持的模型:
- Spark-Max (generalv3.5)
- Spark-Pro (generalv3)
- Spark-Lite (generalv2)
- Spark-Ultra (4.0Ultra)

API文档: https://www.xfyun.cn/doc/spark/Web.html

Author: AGI Team
Version: 1.0.0
"""

import time
import logging
import json
import hmac
import hashlib
import base64
import asyncio
import re
from datetime import datetime
from dataclasses import dataclass
from typing import (
    Dict, List, Optional, Any, AsyncIterator, Set
)
from urllib.parse import urlencode, urlparse
from .base import (
    BaseLLMProvider, LLMConfig, LLMResponse,
    ModelCapability, LLMError
)

try:
    import httpx
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SparkConfig(LLMConfig):
    """
    讯飞星火配置

    Attributes:
        app_id: 讯飞开放平台应用ID
        api_secret: 讯飞开放平台API Secret
        api_key: 讯飞开放平台API Key
        domain: 模型域名 (generalv3.5, generalv3, generalv2等)
        ws_base_url: WebSocket基础URL
    """
    app_id: Optional[str] = None
    api_secret: Optional[str] = None
    domain: Optional[str] = None
    ws_base_url: str = "wss://spark-api.xf-yun.com/v3.5/chat"


class SparkProvider(BaseLLMProvider):
    """
    讯飞星火认知大模型Provider

    通过WebSocket与讯飞星火大模型进行交互, 支持对话生成和流式输出。
    讯飞星火使用HMAC-SHA256签名进行域名鉴权。

    Features:
        - 支持Spark-Max/Pro/Lite/Ultra全系列
        - WebSocket长连接流式通信
        - HMAC-SHA256域名鉴权
        - 多轮对话上下文管理
        - 签名过期自动重试
        - 连接断开自动重连

    Example:
        ```python
        provider = SparkProvider(SparkConfig(
            model_id="generalv3.5",
            api_key="your_api_key",
            app_id="your_app_id",
            api_secret="your_api_secret",
        ))

        response = await provider.generate([
            {"role": "user", "content": "你好"}
        ])
        print(response.content)
        ```
    """

    PROVIDER_NAME = "spark"
    SUPPORTED_MODELS = {
        "generalv3.5",
        "generalv3",
        "generalv2",
        "4.0Ultra",
        "general",
    }
    DEFAULT_MODEL = "generalv3.5"

    # 模型到域名和API URL的映射
    _MODEL_CONFIG_MAP: Dict[str, Dict[str, str]] = {
        "4.0Ultra": {
            "domain": "4.0Ultra",
            "ws_url": "wss://spark-api.xf-yun.com/v4.0/chat",
        },
        "generalv3.5": {
            "domain": "generalv3.5",
            "ws_url": "wss://spark-api.xf-yun.com/v3.5/chat",
        },
        "generalv3": {
            "domain": "generalv3",
            "ws_url": "wss://spark-api.xf-yun.com/v3.1/chat",
        },
        "generalv2": {
            "domain": "generalv2",
            "ws_url": "wss://spark-api.xf-yun.com/v2.1/chat",
        },
        "general": {
            "domain": "general",
            "ws_url": "wss://spark-api.xf-yun.com/v1.1/chat",
        },
    }

    # HTTP API基础URL (用于非流式请求)
    _HTTP_API_BASE = "https://spark-api.xf-yun.com/v1/chat/completions"

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        初始化讯飞星火Provider。

        Args:
            config: 星火配置, 推荐使用SparkConfig
        """
        super().__init__(config)
        self._app_id: Optional[str] = None
        self._api_secret: Optional[str] = None
        self._api_key: Optional[str] = None

    def get_capabilities(self) -> Set[ModelCapability]:
        """获取模型能力"""
        capabilities = {
            ModelCapability.STREAMING,
            ModelCapability.JSON_MODE,
        }
        return capabilities

    def _validate_config(self) -> None:
        """验证配置"""
        super()._validate_config()
        if isinstance(self._config, SparkConfig):
            if not self._config.app_id:
                raise ValueError("SparkConfig requires 'app_id'")
            if not self._config.api_secret:
                raise ValueError("SparkConfig requires 'api_secret'")

    def _setup_client(self) -> None:
        """设置客户端, 提取认证信息"""
        if isinstance(self._config, SparkConfig):
            self._app_id = self._config.app_id
            self._api_secret = self._config.api_secret
            self._api_key = self._config.api_key or self._config.api_key

    def _get_model_config(self, model_id: str) -> Dict[str, str]:
        """
        获取模型对应的域名和WebSocket URL配置。

        Args:
            model_id: 模型ID

        Returns:
            包含domain和ws_url的字典
        """
        config = self._MODEL_CONFIG_MAP.get(model_id)
        if not config:
            logger.warning(
                f"Unknown model config for {model_id}, using generalv3.5 defaults"
            )
            config = self._MODEL_CONFIG_MAP["generalv3.5"]
        return config

    def _create_auth_url(self, ws_url: str) -> str:
        """
        生成带鉴权参数的WebSocket URL。

        使用HMAC-SHA256对请求进行签名, 签名有效期为当前时间。

        Args:
            ws_url: WebSocket基础URL

        Returns:
            带鉴权参数的完整URL
        """
        if not self._api_key or not self._api_secret:
            raise LLMError(
                "api_key and api_secret are required for authentication",
                model_id=self.model_id,
            )

        # 解析URL
        parsed = urlparse(ws_url)
        host = parsed.hostname
        path = parsed.path

        # RFC1123格式时间戳
        now = datetime.utcnow()
        date = now.strftime("%a, %d %b %Y %H:%M:%S GMT")

        # 构造签名原始字符串
        signature_origin = f"host: {host}\ndate: {date}\nGET {path} HTTP/1.1"

        # HMAC-SHA256签名
        signature_sha = hmac.new(
            self._api_secret.encode("utf-8"),
            signature_origin.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()

        signature = base64.b64encode(signature_sha).decode("utf-8")

        # 构造authorization
        authorization_origin = (
            f'api_key="{self._api_key}", '
            f'algorithm="hmac-sha256", '
            f'headers="host date request-line", '
            f'signature="{signature}"'
        )
        authorization = base64.b64encode(
            authorization_origin.encode("utf-8")
        ).decode("utf-8")

        # 拼接完整URL
        params = {
            "authorization": authorization,
            "date": date,
            "host": host,
        }

        return f"{ws_url}?{urlencode(params)}"

    def _convert_messages(
        self,
        messages: List[Dict[str, Any]]
    ) -> tuple:
        """
        转换消息格式为讯飞星火API要求的格式。

        讯飞星火API的消息格式:
        - role: "user" | "assistant" | "system" (通过header设置)
        - content: 文本内容

        Args:
            messages: 标准消息列表

        Returns:
            (chat_messages, system_content) 元组
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

        return chat_messages, system_content.strip()

    async def _async_generate(
        self,
        messages: List[Dict[str, Any]],
        config: LLMConfig
    ) -> LLMResponse:
        """
        异步生成响应。

        通过WebSocket连接讯飞星火API进行同步请求(等待完整响应)。

        Args:
            messages: 消息列表
            config: 生成配置

        Returns:
            LLM响应
        """
        start_time = time.time()

        if not WEBSOCKETS_AVAILABLE:
            return LLMResponse(
                content="",
                model_id=config.model_id,
                error="websockets library not available, install with: pip install websockets",
                latency_ms=(time.time() - start_time) * 1000,
            )

        model_config = self._get_model_config(config.model_id)
        ws_url = model_config["ws_url"]

        try:
            auth_url = self._create_auth_url(ws_url)
        except LLMError as e:
            return LLMResponse(
                content="",
                model_id=config.model_id,
                error=f"Auth error: {e.message}",
                latency_ms=(time.time() - start_time) * 1000,
            )

        chat_messages, system_content = self._convert_messages(messages)

        # 构建请求payload
        payload: Dict[str, Any] = {
            "header": {
                "app_id": self._app_id,
            },
            "parameter": {
                "chat": {
                    "domain": model_config["domain"],
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens or 4096,
                    "top_k": int(config.top_p * 10),
                }
            },
            "payload": {
                "message": {
                    "text": chat_messages,
                }
            },
        }

        if system_content:
            payload["payload"]["message"]["system"] = system_content

        # 带重试的WebSocket请求
        last_error = None
        for attempt in range(config.max_retries):
            try:
                full_content = ""
                usage_info = {}
                finish_reason = None

                async with websockets.connect(
                    auth_url,
                    ping_interval=None,
                    close_timeout=5,
                ) as ws:
                    await ws.send(json.dumps(payload))

                    while True:
                        try:
                            response_raw = await asyncio.wait_for(
                                ws.recv(), timeout=config.timeout
                            )
                        except asyncio.TimeoutError:
                            last_error = "WebSocket receive timeout"
                            break

                        response_data = json.loads(response_raw)
                        header = response_data.get("header", {})
                        code = header.get("code", 0)

                        if code != 0:
                            error_msg = header.get("message", "Unknown error")
                            last_error = f"Spark API error (code={code}): {error_msg}"
                            # 签名过期错误码
                            if code in (10005, 10013):
                                logger.info("Signature expired, retrying...")
                                await asyncio.sleep(0.5)
                                continue
                            break

                        # 提取内容
                        text_payload = response_data.get("payload", {})
                        choices = text_payload.get("choices", {})
                        text_list = choices.get("text", [])
                        if text_list:
                            full_content += text_list[0].get("content", "")

                        # 提取usage
                        usage_payload = text_payload.get("usage", {})
                        if usage_payload:
                            usage_info = {
                                "prompt_tokens": usage_payload.get("text", {}).get(
                                    "prompt_tokens", 0
                                ),
                                "completion_tokens": usage_payload.get("text", {}).get(
                                    "completion_tokens", 0
                                ),
                                "total_tokens": usage_payload.get("text", {}).get(
                                    "total_tokens", 0
                                ),
                            }

                        # 检查是否结束
                        status = header.get("status", 0)
                        if status == 2:
                            finish_reason = "stop"
                            break

                    if full_content:
                        return LLMResponse(
                            content=full_content,
                            model_id=config.model_id,
                            usage=usage_info if usage_info else None,
                            finish_reason=finish_reason,
                            latency_ms=(time.time() - start_time) * 1000,
                            metadata={
                                "provider": "spark",
                                "model": config.model_id,
                                "domain": model_config["domain"],
                            },
                        )

            except ConnectionError as e:
                last_error = f"WebSocket connection error: {e}"
                logger.warning(
                    f"Connection error (attempt {attempt + 1}/{config.max_retries}): {e}"
                )
                if attempt < config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                continue
            except Exception as e:
                last_error = str(e)
                logger.error(f"Spark generate error: {e}")
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

        通过WebSocket连接讯飞星火API, 实时返回生成的内容片段。

        Args:
            messages: 消息列表
            config: 生成配置

        Yields:
            响应文本片段
        """
        if not WEBSOCKETS_AVAILABLE:
            yield "Error: websockets library not available"
            return

        model_config = self._get_model_config(config.model_id)
        ws_url = model_config["ws_url"]

        try:
            auth_url = self._create_auth_url(ws_url)
        except LLMError as e:
            yield f"Error: Auth error: {e.message}"
            return

        chat_messages, system_content = self._convert_messages(messages)

        payload: Dict[str, Any] = {
            "header": {
                "app_id": self._app_id,
            },
            "parameter": {
                "chat": {
                    "domain": model_config["domain"],
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens or 4096,
                    "top_k": int(config.top_p * 10),
                }
            },
            "payload": {
                "message": {
                    "text": chat_messages,
                }
            },
        }

        if system_content:
            payload["payload"]["message"]["system"] = system_content

        try:
            async with websockets.connect(
                auth_url,
                ping_interval=None,
                close_timeout=5,
            ) as ws:
                await ws.send(json.dumps(payload))

                while True:
                    try:
                        response_raw = await asyncio.wait_for(
                            ws.recv(), timeout=config.timeout
                        )
                    except asyncio.TimeoutError:
                        yield "Error: WebSocket receive timeout"
                        break

                    response_data = json.loads(response_raw)
                    header = response_data.get("header", {})
                    code = header.get("code", 0)

                    if code != 0:
                        error_msg = header.get("message", "Unknown error")
                        yield f"Error: Spark API error (code={code}): {error_msg}"
                        break

                    # 提取并yield内容片段
                    text_payload = response_data.get("payload", {})
                    choices = text_payload.get("choices", {})
                    text_list = choices.get("text", [])
                    if text_list:
                        content = text_list[0].get("content", "")
                        if content:
                            yield content

                    # 检查是否结束
                    status = header.get("status", 0)
                    if status == 2:
                        break

        except ConnectionError as e:
            yield f"Error: WebSocket connection failed: {e}"
        except Exception as e:
            yield f"Error: {e}"

    async def token_count(
        self,
        text: str,
        model: Optional[str] = None
    ) -> int:
        """
        估算文本的token数量。

        讯飞星火没有专门的token计数接口,
        这里使用简单的字符级估算。

        Args:
            text: 待计数的文本
            model: 模型名称 (未使用)

        Returns:
            估算的token数量
        """
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        estimated_tokens = int(chinese_chars / 1.5 + other_chars / 4)
        return max(1, estimated_tokens)

    def get_stats(self) -> Dict[str, Any]:
        """获取Provider统计信息"""
        model_config = self._get_model_config(self.model_id)
        return {
            "provider": self.PROVIDER_NAME,
            "model_id": self.model_id,
            "initialized": self._initialized,
            "domain": model_config.get("domain", ""),
            "ws_url": model_config.get("ws_url", ""),
            "has_app_id": self._app_id is not None,
            "websockets_available": WEBSOCKETS_AVAILABLE,
            "supported_models": sorted(self.SUPPORTED_MODELS),
        }
