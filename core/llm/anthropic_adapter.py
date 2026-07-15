"""
AGI Unified Framework - Anthropic Adapter
Anthropic Claude API适配器，使用urllib.request发送HTTP请求
"""

import json
import re
import time
import urllib.request
import urllib.error
from typing import Any, Dict, Iterator, List, Optional

from .base import (
    AuthenticationError,
    FinishReason,
    GenerateParams,
    InvalidRequestError,
    LLMBackend,
    LLMChunk,
    LLMError,
    LLMResponse,
    Message,
    ModelInfo,
    ModelNotFoundError,
    RateLimitError,
    ServerError,
    TimeoutError,
    ToolCall,
    Usage,
)


class AnthropicAdapter(LLMBackend):
    """
    Anthropic Claude API适配器

    使用urllib.request发送HTTP请求，支持：
    - /v1/messages API
    - SSE流式响应
    - system prompt分离
    - Claude特有参数（top_k）
    - 完整的错误处理
    """

    DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    DEFAULT_TIMEOUT = 60
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "",
        default_params: Optional[Dict[str, Any]] = None,
        timeout: int = 0,
    ):
        self._api_key = api_key
        self._base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self._model = model or self.DEFAULT_MODEL
        self._default_params = default_params or {}
        self._timeout = timeout or self.DEFAULT_TIMEOUT
        self._total_requests = 0
        self._total_errors = 0
        self._last_request_time = 0.0

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str):
        self._model = value

    def _build_headers(self) -> Dict[str, str]:
        """构建Anthropic API请求头"""
        return {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": self.API_VERSION,
        }

    def _make_request(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        stream: bool = False,
    ) -> Any:
        """发送HTTP请求到Anthropic API"""
        url = f"{self._base_url}{endpoint}"
        headers = self._build_headers()
        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        if stream:
            req.add_header("Accept", "text/event-stream")

        self._total_requests += 1
        self._last_request_time = time.time()

        try:
            response = urllib.request.urlopen(req, timeout=self._timeout)
            return response
        except urllib.error.HTTPError as e:
            self._total_errors += 1
            error_body = ""
            try:
                error_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass

            try:
                error_data = json.loads(error_body)
                error_message = error_data.get("error", {}).get("message", str(e))
                error_type = error_data.get("error", {}).get("type", "unknown")
            except (json.JSONDecodeError, AttributeError):
                error_message = error_body or str(e)
                error_type = "unknown"

            if e.code == 401:
                raise AuthenticationError(error_message)
            elif e.code == 404:
                raise ModelNotFoundError(self._model)
            elif e.code == 429:
                retry_after = 1.0
                try:
                    retry_header = e.headers.get("Retry-After", "")
                    if retry_header:
                        retry_after = float(retry_header)
                except (ValueError, TypeError):
                    pass
                raise RateLimitError(error_message, retry_after=retry_after)
            elif e.code == 400:
                raise InvalidRequestError(error_message)
            elif e.code >= 500:
                raise ServerError(error_message, status_code=e.code)
            else:
                raise LLMError(error_message, error_type=error_type, status_code=e.code)
        except urllib.error.URLError as e:
            self._total_errors += 1
            if "timed out" in str(e).lower():
                raise TimeoutError(f"Request timed out: {e}", timeout=self._timeout)
            raise LLMError(f"Connection error: {e}", error_type="connection_error", retryable=True)

    def _separate_system_messages(
        self, messages: List[Message],
    ) -> tuple:
        """
        分离system消息和对话消息

        Anthropic API要求system prompt单独传递，
        不在messages数组中。

        Returns:
            (system_prompt, conversation_messages)
        """
        system_parts = []
        conversation = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            else:
                conversation.append(msg)

        system_prompt = "\n\n".join(system_parts) if system_parts else ""
        return system_prompt, conversation

    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """将Message列表转换为Anthropic API格式"""
        result = []
        for msg in messages:
            msg_dict: Dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.metadata:
                msg_dict.update(msg.metadata)
            result.append(msg_dict)
        return result

    def _parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        """解析Anthropic API响应为LLMResponse"""
        content_blocks = data.get("content", [])
        text_parts = []
        tool_calls = None

        for block in content_blocks:
            block_type = block.get("type", "")
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(ToolCall(
                    id=block.get("id", ""),
                    type="function",
                    function_name=block.get("name", ""),
                    arguments=json.dumps(block.get("input", {})),
                ))

        content = "\n".join(text_parts)

        # 解析使用量
        usage_data = data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
        )

        # 解析结束原因
        stop_reason = data.get("stop_reason", "end_turn")
        finish_reason_map = {
            "end_turn": FinishReason.STOP,
            "max_tokens": FinishReason.LENGTH,
            "tool_use": FinishReason.TOOL_CALLS,
            "stop_sequence": FinishReason.STOP,
        }
        finish_reason = finish_reason_map.get(stop_reason, FinishReason.STOP)

        return LLMResponse(
            content=content,
            usage=usage,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            model=data.get("model", self._model),
            id=data.get("id", ""),
        )

    def generate(
        self,
        messages: List[Message],
        params: Optional[GenerateParams] = None,
    ) -> LLMResponse:
        """调用/messages API生成回复"""
        if not self.validate_messages(messages):
            raise InvalidRequestError("Invalid message format")

        params = params or GenerateParams()
        system_prompt, conversation = self._separate_system_messages(messages)

        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": self._convert_messages(conversation),
            "max_tokens": params.max_tokens or 4096,
        }

        # Anthropic要求max_tokens必填
        if params.max_tokens:
            payload["max_tokens"] = params.max_tokens

        if system_prompt:
            payload["system"] = system_prompt

        # 添加Claude特有参数
        anthropic_params = params.to_anthropic_dict()
        for key, value in anthropic_params.items():
            if key != "max_tokens":
                payload[key] = value

        # 添加工具定义
        if params.tools:
            anthropic_tools = []
            for tool in params.tools:
                func = tool.get("function", {})
                anthropic_tools.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                })
            payload["tools"] = anthropic_tools

        payload.update(self._default_params)

        response = self._make_request("/messages", payload)
        response_data = json.loads(response.read().decode("utf-8"))
        return self._parse_response(response_data)

    def stream(
        self,
        messages: List[Message],
        params: Optional[GenerateParams] = None,
    ) -> Iterator[LLMChunk]:
        """SSE流式响应"""
        if not self.validate_messages(messages):
            raise InvalidRequestError("Invalid message format")

        params = params or GenerateParams()
        system_prompt, conversation = self._separate_system_messages(messages)

        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": self._convert_messages(conversation),
            "max_tokens": params.max_tokens or 4096,
            "stream": True,
        }

        if system_prompt:
            payload["system"] = system_prompt

        anthropic_params = params.to_anthropic_dict()
        for key, value in anthropic_params.items():
            if key not in ("max_tokens",):
                payload[key] = value

        payload.update(self._default_params)

        response = self._make_request("/messages", payload, stream=True)

        buffer = ""
        while True:
            chunk_data = response.read(1024)
            if not chunk_data:
                break
            buffer += chunk_data.decode("utf-8", errors="replace")

            while "\n\n" in buffer:
                event_str, buffer = buffer.split("\n\n", 1)
                event_type = ""
                event_data = ""
                for line in event_str.split("\n"):
                    if line.startswith("event: "):
                        event_type = line[7:].strip()
                    elif line.startswith("data: "):
                        event_data = line[6:].strip()

                if not event_data:
                    continue

                try:
                    data = json.loads(event_data)
                    chunk = self._parse_stream_event(event_type, data)
                    if chunk:
                        yield chunk
                except json.JSONDecodeError:
                    continue

    def _parse_stream_event(self, event_type: str, data: Dict[str, Any]) -> Optional[LLMChunk]:
        """解析Anthropic SSE流式事件"""
        if event_type == "content_block_delta":
            delta = data.get("delta", {})
            delta_type = delta.get("type", "")
            if delta_type == "text_delta":
                return LLMChunk(delta_content=delta.get("text", ""))
            elif delta_type == "input_json_delta":
                return LLMChunk(
                    delta_content=delta.get("partial_json", ""),
                    metadata={"type": "tool_call_delta"},
                )
            return None

        elif event_type == "message_delta":
            delta = data.get("delta", {})
            stop_reason = delta.get("stop_reason")
            finish_reason = None
            if stop_reason:
                reason_map = {
                    "end_turn": FinishReason.STOP,
                    "max_tokens": FinishReason.LENGTH,
                    "tool_use": FinishReason.TOOL_CALLS,
                }
                finish_reason = reason_map.get(stop_reason, FinishReason.STOP)

            usage = None
            usage_data = data.get("usage", {})
            if usage_data:
                usage = Usage(
                    completion_tokens=usage_data.get("output_tokens", 0),
                )

            return LLMChunk(finish_reason=finish_reason, usage=usage)

        elif event_type == "message_start":
            message = data.get("message", {})
            usage_data = message.get("usage", {})
            usage = Usage(
                prompt_tokens=usage_data.get("input_tokens", 0),
            )
            return LLMChunk(usage=usage, metadata={"type": "message_start"})

        elif event_type == "message_stop":
            return LLMChunk(finish_reason=FinishReason.STOP, metadata={"type": "message_stop"})

        return None

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Anthropic目前不直接提供嵌入API，
        返回零向量作为占位（实际使用中应接入其他嵌入服务）
        """
        dim = 1024
        return [[0.0] * dim for _ in texts]

    def count_tokens(self, text: str) -> int:
        """
        近似Token计数
        Claude使用自己的分词器，这里使用字符级别的近似
        """
        if not text:
            return 0

        # 中文字符每个约1.5 token
        # 英文单词每个约1.3 token
        # 标点符号约0.5 token
        total = 0.0
        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                total += 1.5
            elif char.isalpha():
                total += 0.25  # 每个字母约0.25 token
            elif char.isspace():
                total += 0.1
            else:
                total += 0.3

        return max(1, int(total))

    def get_model_info(self) -> ModelInfo:
        """获取模型信息"""
        model_configs = {
            "claude-sonnet-4-20250514": ModelInfo(
                name="claude-sonnet-4-20250514", max_context=200000, max_output=16384,
                supports_streaming=True, supports_functions=True,
                vendor="Anthropic", version="4.0",
                description="Claude Sonnet 4 - balanced performance and speed",
            ),
            "claude-opus-4-20250514": ModelInfo(
                name="claude-opus-4-20250514", max_context=200000, max_output=16384,
                supports_streaming=True, supports_functions=True,
                vendor="Anthropic", version="4.0",
                description="Claude Opus 4 - highest capability",
            ),
            "claude-3-5-sonnet-20241022": ModelInfo(
                name="claude-3-5-sonnet-20241022", max_context=200000, max_output=8192,
                supports_streaming=True, supports_functions=True,
                vendor="Anthropic", version="3.5",
                description="Claude 3.5 Sonnet - fast and intelligent",
            ),
            "claude-3-5-haiku-20241022": ModelInfo(
                name="claude-3-5-haiku-20241022", max_context=200000, max_output=8192,
                supports_streaming=True, supports_functions=True,
                vendor="Anthropic", version="3.5",
                description="Claude 3.5 Haiku - fastest Claude model",
            ),
            "claude-3-opus-20240229": ModelInfo(
                name="claude-3-opus-20240229", max_context=200000, max_output=4096,
                supports_streaming=True, supports_functions=True,
                vendor="Anthropic", version="3.0",
                description="Claude 3 Opus - powerful model for complex tasks",
            ),
        }
        return model_configs.get(
            self._model,
            ModelInfo(
                name=self._model, max_context=100000, max_output=4096,
                supports_streaming=True, supports_functions=True,
                vendor="Anthropic",
            ),
        )

    def health_check(self) -> bool:
        """健康检查"""
        try:
            payload = {
                "model": self._model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            }
            response = self._make_request("/messages", payload)
            response.read()
            return True
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """获取适配器统计信息"""
        return {
            "total_requests": self._total_requests,
            "total_errors": self._total_errors,
            "error_rate": self._total_errors / max(self._total_requests, 1),
            "last_request_time": self._last_request_time,
            "model": self._model,
            "base_url": self._base_url,
        }
