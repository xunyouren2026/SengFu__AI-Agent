"""
AGI Unified Framework - Local Adapter
本地模型适配器（模拟Ollama/vLLM），支持HTTP调用和简易文本生成
"""

import json
import random
import re
import time
import urllib.request
import urllib.error
from typing import Any, Dict, Iterator, List, Optional

from .base import (
    FinishReason,
    GenerateParams,
    LLMBackend,
    LLMChunk,
    LLMError,
    LLMResponse,
    Message,
    ModelInfo,
    Usage,
)


# 简易文本生成用的词库
_SIMPLE_RESPONSES = {
    "greeting": [
        "Hello! How can I help you today?",
        "Hi there! What would you like to know?",
        "Greetings! I'm here to assist you.",
    ],
    "default": [
        "I understand your question. Let me provide a helpful response.",
        "That's an interesting point. Here's what I think about it.",
        "Thank you for your question. Let me address that for you.",
        "I'd be happy to help with that. Here's my perspective.",
    ],
    "code": [
        "Here's a code example that demonstrates this concept:\n\n```python\ndef example():\n    pass\n```\n\nThis shows the basic pattern.",
        "Let me illustrate with some code:\n\n```python\n# Example implementation\nresult = process(data)\nprint(result)\n```\n\nThe key idea is to break down the problem.",
    ],
    "error": [
        "I apologize, but I encountered an issue processing your request.",
        "Something went wrong. Could you please try rephrasing your question?",
    ],
}


class _SimpleTextGenerator:
    """
    简易文本生成器

    当本地服务不可用时，使用基于规则的文本生成作为后备。
    这不是真正的语言模型，仅用于演示和测试。
    """

    def __init__(self):
        self._patterns = [
            (re.compile(r"\b(hi|hello|hey|greetings)\b", re.IGNORECASE), "greeting"),
            (re.compile(r"\b(code|function|class|implement|program)\b", re.IGNORECASE), "code"),
            (re.compile(r"\b(error|bug|fail|wrong|broken)\b", re.IGNORECASE), "error"),
        ]

    def classify(self, text: str) -> str:
        """对输入文本进行简单分类"""
        for pattern, category in self._patterns:
            if pattern.search(text):
                return category
        return "default"

    def generate(self, text: str, max_tokens: int = 100) -> str:
        """生成响应文本"""
        category = self.classify(text)
        responses = _SIMPLE_RESPONSES.get(category, _SIMPLE_RESPONSES["default"])
        base = random.choice(responses)

        # 根据max_tokens截断或扩展
        words = base.split()
        if len(words) > max_tokens:
            return " ".join(words[:max_tokens])
        return base

    def generate_stream(self, text: str, max_tokens: int = 100) -> Iterator[str]:
        """流式生成响应文本"""
        full_response = self.generate(text, max_tokens)
        words = full_response.split(" ")
        for i, word in enumerate(words):
            if i == 0:
                yield word
            else:
                yield " " + word
            time.sleep(0.02)  # 模拟生成延迟


class LocalAdapter(LLMBackend):
    """
    本地模型适配器

    支持两种模式：
    1. HTTP模式：通过HTTP调用本地服务（兼容Ollama/vLLM API格式）
    2. 本地模拟模式：当无本地服务时使用简易文本生成

    特性：
    - 自动检测本地服务可用性
    - 支持模型切换
    - 支持流式响应
    - 完整的错误处理
    """

    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_MODEL = "llama3"
    DEFAULT_TIMEOUT = 120

    def __init__(
        self,
        base_url: str = "",
        model: str = "",
        default_params: Optional[Dict[str, Any]] = None,
        timeout: int = 0,
        fallback_to_simulated: bool = True,
    ):
        self._base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self._model = model or self.DEFAULT_MODEL
        self._default_params = default_params or {}
        self._timeout = timeout or self.DEFAULT_TIMEOUT
        self._fallback_to_simulated = fallback_to_simulated
        self._generator = _SimpleTextGenerator()
        self._service_available: Optional[bool] = None
        self._total_requests = 0
        self._total_errors = 0
        self._last_request_time = 0.0

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str):
        self._model = value
        self._service_available = None  # 重置服务可用性

    def _check_service(self) -> bool:
        """检查本地服务是否可用"""
        if self._service_available is not None:
            return self._service_available

        try:
            url = f"{self._base_url}/api/tags"
            req = urllib.request.Request(url, method="GET")
            urllib.request.urlopen(req, timeout=5)
            self._service_available = True
        except Exception:
            self._service_available = False

        return self._service_available

    def _make_request(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        stream: bool = False,
    ) -> Any:
        """发送HTTP请求到本地服务"""
        url = f"{self._base_url}{endpoint}"
        data = json.dumps(payload).encode("utf-8")

        headers = {"Content-Type": "application/json"}
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
            raise LLMError(
                f"Local service error ({e.code}): {error_body}",
                error_type="local_service_error",
                status_code=e.code,
                retryable=e.code >= 500,
            )
        except urllib.error.URLError as e:
            self._total_errors += 1
            self._service_available = False
            raise LLMError(
                f"Cannot connect to local service: {e}",
                error_type="connection_error",
                retryable=True,
            )

    def _build_ollama_payload(
        self,
        messages: List[Message],
        params: GenerateParams,
    ) -> Dict[str, Any]:
        """构建Ollama API格式的请求体"""
        ollama_messages = []
        for msg in messages:
            ollama_messages.append({
                "role": msg.role,
                "content": msg.content,
            })

        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": ollama_messages,
            "stream": params.stream,
        }

        local_params = params.to_local_dict()
        for key, value in local_params.items():
            if key != "stream":
                payload["options"] = payload.get("options", {})
                payload["options"][key] = value

        payload.update(self._default_params)
        return payload

    def _parse_ollama_response(self, data: Dict[str, Any]) -> LLMResponse:
        """解析Ollama API响应"""
        content = data.get("message", {}).get("content", "")

        # Ollama不直接返回token使用量，进行估算
        prompt_eval_count = data.get("prompt_eval_count", 0)
        eval_count = data.get("eval_count", 0)
        usage = Usage(
            prompt_tokens=prompt_eval_count,
            completion_tokens=eval_count,
            total_tokens=prompt_eval_count + eval_count,
        )

        finish_reason = FinishReason.STOP
        if data.get("done_reason") == "length":
            finish_reason = FinishReason.LENGTH

        return LLMResponse(
            content=content,
            usage=usage,
            finish_reason=finish_reason,
            model=data.get("model", self._model),
        )

    def generate(
        self,
        messages: List[Message],
        params: Optional[GenerateParams] = None,
    ) -> LLMResponse:
        """生成回复"""
        params = params or GenerateParams()

        if self._check_service():
            return self._generate_via_http(messages, params)
        elif self._fallback_to_simulated:
            return self._generate_simulated(messages, params)
        else:
            raise LLMError(
                "Local service is not available and simulated fallback is disabled",
                error_type="service_unavailable",
            )

    def _generate_via_http(
        self, messages: List[Message], params: GenerateParams,
    ) -> LLMResponse:
        """通过HTTP调用本地服务生成"""
        payload = self._build_ollama_payload(messages, params)
        payload["stream"] = False

        response = self._make_request("/api/chat", payload)
        response_data = json.loads(response.read().decode("utf-8"))
        return self._parse_ollama_response(response_data)

    def _generate_simulated(
        self, messages: List[Message], params: GenerateParams,
    ) -> LLMResponse:
        """使用简易文本生成器模拟"""
        # 提取最后一条用户消息
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.role == "user":
                last_user_msg = msg.content
                break

        content = self._generator.generate(last_user_msg, params.max_tokens)

        # 估算token数
        prompt_tokens = sum(len(m.content.split()) for m in messages)
        completion_tokens = len(content.split())

        return LLMResponse(
            content=content,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            finish_reason=FinishReason.STOP,
            model=f"simulated/{self._model}",
            metadata={"simulated": True},
        )

    def stream(
        self,
        messages: List[Message],
        params: Optional[GenerateParams] = None,
    ) -> Iterator[LLMChunk]:
        """流式生成回复"""
        params = params or GenerateParams()

        if self._check_service():
            yield from self._stream_via_http(messages, params)
        elif self._fallback_to_simulated:
            yield from self._stream_simulated(messages, params)
        else:
            raise LLMError(
                "Local service is not available and simulated fallback is disabled",
                error_type="service_unavailable",
            )

    def _stream_via_http(
        self, messages: List[Message], params: GenerateParams,
    ) -> Iterator[LLMChunk]:
        """通过HTTP流式调用本地服务"""
        payload = self._build_ollama_payload(messages, params)
        payload["stream"] = True

        response = self._make_request("/api/chat", payload, stream=True)

        buffer = ""
        while True:
            chunk_data = response.read(1024)
            if not chunk_data:
                break
            buffer += chunk_data.decode("utf-8", errors="replace")

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")

                    finish_reason = None
                    if data.get("done"):
                        finish_reason = FinishReason.STOP

                    if content:
                        yield LLMChunk(delta_content=content, finish_reason=finish_reason)
                    elif finish_reason:
                        yield LLMChunk(finish_reason=finish_reason)
                except json.JSONDecodeError:
                    continue

    def _stream_simulated(
        self, messages: List[Message], params: GenerateParams,
    ) -> Iterator[LLMChunk]:
        """使用简易文本生成器模拟流式输出"""
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.role == "user":
                last_user_msg = msg.content
                break

        token_count = 0
        for word in self._generator.generate_stream(last_user_msg, params.max_tokens):
            token_count += 1
            if token_count >= params.max_tokens:
                yield LLMChunk(delta_content=word, finish_reason=FinishReason.LENGTH)
                break
            yield LLMChunk(delta_content=word)

        yield LLMChunk(finish_reason=FinishReason.STOP)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """获取文本嵌入（本地服务模式）"""
        if not self._check_service():
            # 返回伪嵌入
            dim = 768
            return [[random.gauss(0, 0.1) for _ in range(dim)] for _ in texts]

        embeddings = []
        for text in texts:
            try:
                payload = {
                    "model": self._model,
                    "prompt": text,
                }
                response = self._make_request("/api/embeddings", payload)
                data = json.loads(response.read().decode("utf-8"))
                embeddings.append(data.get("embedding", []))
            except LLMError:
                embeddings.append([0.0] * 768)

        return embeddings

    def count_tokens(self, text: str) -> int:
        """Token计数（近似）"""
        if not text:
            return 0
        # 简单的词级计数
        words = text.split()
        count = 0
        for word in words:
            if any("\u4e00" <= c <= "\u9fff" for c in word):
                count += len(word)
            else:
                count += 1
        return count

    def get_model_info(self) -> ModelInfo:
        """获取模型信息"""
        if self._check_service():
            try:
                url = f"{self._base_url}/api/show"
                payload = {"name": self._model}
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    url, data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                response = urllib.request.urlopen(req, timeout=10)
                info = json.loads(response.read().decode("utf-8"))
                details = info.get("details", {})
                return ModelInfo(
                    name=self._model,
                    max_context=details.get("num_ctx", 4096),
                    max_output=details.get("num_predict", 2048),
                    supports_streaming=True,
                    supports_functions=False,
                    vendor="Local",
                    description=details.get("family", "Unknown"),
                )
            except Exception:
                pass

        return ModelInfo(
            name=f"simulated/{self._model}",
            max_context=4096,
            max_output=2048,
            supports_streaming=True,
            supports_functions=False,
            vendor="Local (Simulated)",
        )

    def list_models(self) -> List[str]:
        """列出本地可用的模型"""
        if not self._check_service():
            return []

        try:
            url = f"{self._base_url}/api/tags"
            req = urllib.request.Request(url, method="GET")
            response = urllib.request.urlopen(req, timeout=10)
            data = json.loads(response.read().decode("utf-8"))
            return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            return []

    def health_check(self) -> bool:
        """健康检查"""
        return self._check_service()

    def get_stats(self) -> Dict[str, Any]:
        """获取适配器统计信息"""
        return {
            "total_requests": self._total_requests,
            "total_errors": self._total_errors,
            "error_rate": self._total_errors / max(self._total_requests, 1),
            "last_request_time": self._last_request_time,
            "model": self._model,
            "service_available": self._service_available,
            "base_url": self._base_url,
        }
