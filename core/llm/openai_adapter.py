"""
AGI Unified Framework - OpenAI Adapter
OpenAI API适配器，使用urllib.request发送HTTP请求
"""

import json
import re
import time
import urllib.request
import urllib.error
import urllib.parse
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


# 简易BPE词表（用于近似Token计数）
_BPE_MERGES = {
    "th": 0, "the": 1, "in": 2, "er": 3, "an": 4, "re": 5, "on": 6,
    "at": 7, "en": 8, "nd": 9, "ti": 10, "es": 11, "or": 12, "te": 13,
    "of": 14, "ed": 15, "is": 16, "it": 17, "al": 18, "ar": 19, "st": 20,
    "to": 21, "nt": 22, "ng": 23, "se": 24, "ha": 25, "as": 26, "ou": 27,
    "io": 28, "le": 29, "ve": 30, "co": 31, "me": 32, "de": 33, "hi": 34,
    "ri": 35, "ro": 36, "ic": 37, "ne": 38, "ea": 39, "ra": 40, "ce": 41,
}


class _SimpleBPETokenizer:
    """
    简易BPE分词器（近似tiktoken）
    用于在不依赖外部库的情况下估算Token数量
    """

    def __init__(self):
        self._pattern = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""",
            re.UNICODE,
        )
        self._cache: Dict[str, int] = {}

    def _bpe_merge(self, word: str) -> int:
        """对单个词进行BPE合并，返回token数"""
        if word in self._cache:
            return self._cache[word]

        tokens = list(word)
        if len(tokens) <= 1:
            self._cache[word] = 1
            return 1

        # 尝试合并相邻字符
        merged = True
        while merged and len(tokens) > 1:
            merged = False
            best_idx = -1
            best_rank = float("inf")
            for i in range(len(tokens) - 1):
                pair = tokens[i] + tokens[i + 1]
                rank = _BPE_MERGES.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_idx = i
            if best_idx >= 0 and best_rank < float("inf"):
                tokens[best_idx] = tokens[best_idx] + tokens[best_idx + 1]
                tokens.pop(best_idx + 1)
                merged = True

        count = len(tokens)
        self._cache[word] = count
        return count

    def count_tokens(self, text: str) -> int:
        """统计文本的Token数量"""
        if not text:
            return 0

        words = self._pattern.findall(text)
        total = 0
        for word in words:
            # 每个词大约1-1.5个token（英文），中文每个字符约1-2个token
            if any("\u4e00" <= c <= "\u9fff" for c in word):
                total += len(word)  # 中文字符每个约1个token
            else:
                total += self._bpe_merge(word)
        return total


class OpenAIAdapter(LLMBackend):
    """
    OpenAI API适配器

    使用urllib.request发送HTTP请求，支持：
    - /v1/chat/completions（对话补全）
    - /v1/embeddings（文本嵌入）
    - SSE流式响应解析
    - Function Calling格式
    - 完整的参数支持（temperature/top_p/max_tokens/n/stop等）
    """

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODEL = "gpt-4o"
    DEFAULT_TIMEOUT = 60

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "",
        organization: str = "",
        default_params: Optional[Dict[str, Any]] = None,
        timeout: int = 0,
    ):
        self._api_key = api_key
        self._base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self._model = model or self.DEFAULT_MODEL
        self._organization = organization
        self._default_params = default_params or {}
        self._timeout = timeout or self.DEFAULT_TIMEOUT
        self._tokenizer = _SimpleBPETokenizer()
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
        """构建请求头"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        if self._organization:
            headers["OpenAI-Organization"] = self._organization
        return headers

    def _make_request(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        stream: bool = False,
    ) -> Any:
        """
        发送HTTP请求到OpenAI API

        Args:
            endpoint: API端点（如/chat/completions）
            payload: 请求体
            stream: 是否为流式请求

        Returns:
            解析后的JSON响应或原始响应对象（流式时）

        Raises:
            各种LLMError子类
        """
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
            except (json.JSONDecodeError, AttributeError):
                error_message = error_body or str(e)

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
                raise LLMError(error_message, status_code=e.code)
        except urllib.error.URLError as e:
            self._total_errors += 1
            if "timed out" in str(e).lower():
                raise TimeoutError(f"Request timed out: {e}", timeout=self._timeout)
            raise LLMError(f"Connection error: {e}", error_type="connection_error", retryable=True)

    def _parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        """解析OpenAI API响应为LLMResponse"""
        choice = data.get("choices", [{}])[0]
        message_data = choice.get("message", {})

        content = message_data.get("content", "") or ""

        # 解析工具调用
        tool_calls = None
        raw_tool_calls = message_data.get("tool_calls")
        if raw_tool_calls:
            tool_calls = []
            for tc in raw_tool_calls:
                func = tc.get("function", {})
                tool_calls.append(ToolCall(
                    id=tc.get("id", ""),
                    type=tc.get("type", "function"),
                    function_name=func.get("name", ""),
                    arguments=func.get("arguments", ""),
                ))

        # 解析使用量
        usage_data = data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        # 解析结束原因
        finish_reason_str = choice.get("finish_reason", "stop")
        try:
            finish_reason = FinishReason(finish_reason_str)
        except ValueError:
            finish_reason = FinishReason.STOP

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
        """调用/v1/chat/completions生成回复"""
        if not self.validate_messages(messages):
            raise InvalidRequestError("Invalid message format")

        params = params or GenerateParams()
        payload_messages = [msg.to_openai_dict() for msg in messages]

        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": payload_messages,
        }
        payload.update(params.to_openai_dict())
        payload.update(self._default_params)
        payload["stream"] = False

        response = self._make_request("/chat/completions", payload)
        response_data = json.loads(response.read().decode("utf-8"))
        return self._parse_response(response_data)

    def stream(
        self,
        messages: List[Message],
        params: Optional[GenerateParams] = None,
    ) -> Iterator[LLMChunk]:
        """SSE流式响应解析"""
        if not self.validate_messages(messages):
            raise InvalidRequestError("Invalid message format")

        params = params or GenerateParams()
        params.stream = True
        payload_messages = [msg.to_openai_dict() for msg in messages]

        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": payload_messages,
        }
        payload.update(params.to_openai_dict())
        payload.update(self._default_params)
        payload["stream"] = True

        response = self._make_request("/chat/completions", payload, stream=True)

        buffer = ""
        while True:
            chunk_data = response.read(1024)
            if not chunk_data:
                break
            buffer += chunk_data.decode("utf-8", errors="replace")

            while "\n\n" in buffer:
                event_str, buffer = buffer.split("\n\n", 1)
                for line in event_str.split("\n"):
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            return
                        try:
                            data = json.loads(data_str)
                            chunk = self._parse_stream_chunk(data)
                            if chunk:
                                yield chunk
                        except json.JSONDecodeError:
                            continue

    def _parse_stream_chunk(self, data: Dict[str, Any]) -> Optional[LLMChunk]:
        """解析SSE流式数据块"""
        choices = data.get("choices", [])
        if not choices:
            return None

        choice = choices[0]
        delta = choice.get("delta", {})
        delta_content = delta.get("content", "") or ""

        finish_reason = None
        finish_reason_str = choice.get("finish_reason")
        if finish_reason_str:
            try:
                finish_reason = FinishReason(finish_reason_str)
            except ValueError:
                finish_reason = FinishReason.STOP

        # 解析增量工具调用
        tool_calls_delta = None
        raw_delta_tool_calls = delta.get("tool_calls")
        if raw_delta_tool_calls:
            tool_calls_delta = []
            for tc in raw_delta_tool_calls:
                func = tc.get("function", {})
                tool_calls_delta.append({
                    "index": tc.get("index", 0),
                    "id": tc.get("id"),
                    "type": tc.get("type"),
                    "function": {
                        "name": func.get("name"),
                        "arguments": func.get("arguments"),
                    },
                })

        usage = None
        usage_data = data.get("usage")
        if usage_data:
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

        return LLMChunk(
            delta_content=delta_content,
            finish_reason=finish_reason,
            usage=usage,
            tool_calls_delta=tool_calls_delta,
        )

    def embed(self, texts: List[str]) -> List[List[float]]:
        """调用/v1/embeddings获取文本嵌入"""
        if not texts:
            return []

        payload: Dict[str, Any] = {
            "model": self._model,
            "input": texts,
        }

        response = self._make_request("/embeddings", payload)
        response_data = json.loads(response.read().decode("utf-8"))

        embeddings = []
        for item in response_data.get("data", []):
            embeddings.append(item.get("embedding", []))

        # 按index排序
        indexed = sorted(
            zip(
                [item.get("index", i) for i, item in enumerate(response_data.get("data", []))],
                embeddings,
            ),
            key=lambda x: x[0],
        )
        return [emb for _, emb in indexed]

    def count_tokens(self, text: str) -> int:
        """基于BPE的近似Token计数"""
        return self._tokenizer.count_tokens(text)

    def get_model_info(self) -> ModelInfo:
        """获取模型信息"""
        model_configs = {
            "gpt-4o": ModelInfo(
                name="gpt-4o", max_context=128000, max_output=16384,
                supports_streaming=True, supports_functions=True, supports_vision=True,
                vendor="OpenAI", version="2024-05",
            ),
            "gpt-4-turbo": ModelInfo(
                name="gpt-4-turbo", max_context=128000, max_output=4096,
                supports_streaming=True, supports_functions=True, supports_vision=True,
                vendor="OpenAI", version="2024-04",
            ),
            "gpt-4": ModelInfo(
                name="gpt-4", max_context=8192, max_output=8192,
                supports_streaming=True, supports_functions=True,
                vendor="OpenAI", version="1.0",
            ),
            "gpt-3.5-turbo": ModelInfo(
                name="gpt-3.5-turbo", max_context=16385, max_output=4096,
                supports_streaming=True, supports_functions=True,
                vendor="OpenAI", version="0125",
            ),
        }
        return model_configs.get(
            self._model,
            ModelInfo(
                name=self._model, max_context=4096, max_output=2048,
                supports_streaming=True, supports_functions=True,
                vendor="OpenAI",
            ),
        )

    def health_check(self) -> bool:
        """健康检查 - 尝试获取模型列表"""
        try:
            url = f"{self._base_url}/models"
            headers = self._build_headers()
            req = urllib.request.Request(url, headers=headers)
            urllib.request.urlopen(req, timeout=10)
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
