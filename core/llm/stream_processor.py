"""
AGI Unified Framework - Stream Processor
流式响应处理器，支持文本累积、工具调用解析和完成检测
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .base import FinishReason, LLMChunk, ToolCall, Usage


@dataclass
class StreamBuffer:
    """
    流式缓冲区

    管理流式响应的累积状态，包括文本内容、工具调用和元数据。
    """

    content: str = ""
    full_content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_call_buffers: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    finish_reason: Optional[FinishReason] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_count: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    is_complete: bool = False
    model: str = ""
    response_id: str = ""

    def reset(self):
        """重置缓冲区"""
        self.content = ""
        self.full_content = ""
        self.tool_calls.clear()
        self.tool_call_buffers.clear()
        self.usage = Usage()
        self.finish_reason = None
        self.metadata.clear()
        self.chunk_count = 0
        self.start_time = 0.0
        self.end_time = 0.0
        self.is_complete = False
        self.model = ""
        self.response_id = ""

    def get_accumulated_content(self) -> str:
        """获取累积的完整内容"""
        return self.full_content

    def get_tool_calls(self) -> List[ToolCall]:
        """获取已解析的工具调用列表"""
        result = []
        for tc_data in self.tool_calls:
            result.append(ToolCall(
                id=tc_data.get("id", ""),
                type=tc_data.get("type", "function"),
                function_name=tc_data.get("function", {}).get("name", ""),
                arguments=tc_data.get("function", {}).get("arguments", ""),
            ))
        return result

    def get_stats(self) -> Dict[str, Any]:
        """获取缓冲区统计"""
        elapsed = 0.0
        if self.start_time > 0:
            if self.end_time > 0:
                elapsed = self.end_time - self.start_time
            else:
                elapsed = time.time() - self.start_time

        return {
            "chunk_count": self.chunk_count,
            "content_length": len(self.full_content),
            "tool_call_count": len(self.tool_calls),
            "is_complete": self.is_complete,
            "elapsed_time": round(elapsed, 4),
            "finish_reason": self.finish_reason.value if self.finish_reason else None,
            "model": self.model,
        }


class StreamProcessor:
    """
    流式响应处理器

    功能：
    - 处理流式数据块
    - 累积文本内容
    - 解析增量JSON工具调用
    - 检测生成完成
    - 管理流式缓冲区
    - 支持回调通知
    """

    def __init__(
        self,
        on_chunk: Optional[Any] = None,
        on_complete: Optional[Any] = None,
        on_tool_call: Optional[Any] = None,
        on_error: Optional[Any] = None,
    ):
        """
        Args:
            on_chunk: 收到块时的回调函数
            on_complete: 完成时的回调函数
            on_tool_call: 工具调用完成时的回调函数
            on_error: 错误时的回调函数
        """
        self._on_chunk = on_chunk
        self._on_complete = on_complete
        self._on_tool_call = on_tool_call
        self._on_error = on_error
        self._buffer = StreamBuffer()

    @property
    def buffer(self) -> StreamBuffer:
        return self._buffer

    def reset(self) -> None:
        """重置处理器状态"""
        self._buffer.reset()

    def process_chunk(self, chunk: LLMChunk) -> StreamBuffer:
        """
        处理流式数据块

        Args:
            chunk: 流式数据块

        Returns:
            StreamBuffer: 更新后的缓冲区
        """
        if self._buffer.chunk_count == 0:
            self._buffer.start_time = time.time()

        self._buffer.chunk_count += 1

        # 累积文本
        if chunk.delta_content:
            self._buffer.content = chunk.delta_content
            self._buffer.full_content += chunk.delta_content

        # 处理工具调用增量
        if chunk.tool_calls_delta:
            self._parse_tool_calls_delta(chunk.tool_calls_delta)

        # 更新使用量
        if chunk.usage:
            if chunk.usage.prompt_tokens > 0:
                self._buffer.usage.prompt_tokens = chunk.usage.prompt_tokens
            if chunk.usage.completion_tokens > 0:
                self._buffer.usage.completion_tokens = chunk.usage.completion_tokens
            self._buffer.usage.total_tokens = (
                self._buffer.usage.prompt_tokens + self._buffer.usage.completion_tokens
            )

        # 更新元数据
        if chunk.metadata:
            self._buffer.metadata.update(chunk.metadata)

        # 检测完成
        if chunk.finish_reason:
            self._buffer.finish_reason = chunk.finish_reason
            self._buffer.is_complete = True
            self._buffer.end_time = time.time()

            # 检查是否有未完成的工具调用
            self._finalize_tool_calls()

        # 回调
        if self._on_chunk:
            try:
                self._on_chunk(chunk, self._buffer)
            except Exception:
                pass

        if self._buffer.is_complete and self._on_complete:
            try:
                self._on_complete(self._buffer)
            except Exception:
                pass

        return self._buffer

    def accumulate_text(self) -> str:
        """
        获取累积的文本

        Returns:
            str: 当前累积的完整文本
        """
        return self._buffer.full_content

    def _parse_tool_calls_delta(self, deltas: List[Dict[str, Any]]) -> None:
        """
        解析增量工具调用数据

        OpenAI的流式工具调用是增量的，需要逐步拼接：
        - 第一个块包含id和function.name
        - 后续块包含function.arguments的增量

        Args:
            deltas: 增量工具调用数据列表
        """
        for delta in deltas:
            index = delta.get("index", 0)

            if index not in self._buffer.tool_call_buffers:
                self._buffer.tool_call_buffers[index] = {
                    "id": "",
                    "type": "function",
                    "function": {
                        "name": "",
                        "arguments": "",
                    },
                }

            buf = self._buffer.tool_call_buffers[index]

            # 更新ID
            if delta.get("id"):
                buf["id"] = delta["id"]

            # 更新类型
            if delta.get("type"):
                buf["type"] = delta["type"]

            # 更新函数名
            func = delta.get("function", {})
            if func and func.get("name"):
                buf["function"]["name"] += func["name"]

            # 更新参数（增量拼接）
            if func and func.get("arguments"):
                buf["function"]["arguments"] += func["arguments"]

    def _finalize_tool_calls(self) -> None:
        """完成工具调用的解析"""
        if not self._buffer.tool_call_buffers:
            return

        self._buffer.tool_calls = []
        for index in sorted(self._buffer.tool_call_buffers.keys()):
            tc_data = self._buffer.tool_call_buffers[index]

            # 验证JSON参数
            args_str = tc_data["function"]["arguments"]
            try:
                json.loads(args_str)
            except json.JSONDecodeError:
                # 尝试修复不完整的JSON
                args_str = self._try_fix_json(args_str)
                tc_data["function"]["arguments"] = args_str

            self._buffer.tool_calls.append(tc_data)

        # 回调
        if self._on_tool_call and self._buffer.tool_calls:
            try:
                self._on_tool_call(self._buffer.get_tool_calls())
            except Exception:
                pass

    def _try_fix_json(self, json_str: str) -> str:
        """
        尝试修复不完整的JSON字符串

        Args:
            json_str: 可能不完整的JSON字符串

        Returns:
            str: 修复后的JSON字符串
        """
        json_str = json_str.strip()
        if not json_str:
            return "{}"

        # 尝试补全缺失的括号
        open_braces = json_str.count("{") - json_str.count("}")
        open_brackets = json_str.count("[") - json_str.count("]")

        # 移除末尾不完整的键值对
        if json_str.endswith(","):
            json_str = json_str[:-1]
        if json_str.endswith('"') or json_str.endswith(":"):
            json_str = json_str[:-1]

        # 补全括号
        json_str += "]" * max(open_brackets, 0)
        json_str += "}" * max(open_braces, 0)

        # 验证修复后的JSON
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            return "{}"

    def detect_finish(self, chunk: LLMChunk) -> bool:
        """
        检测生成是否完成

        Args:
            chunk: 当前数据块

        Returns:
            bool: 是否已完成
        """
        if chunk.finish_reason is not None:
            return True

        # 检测常见的结束标记
        if chunk.delta_content:
            end_markers = ["\n\n\n", "<|end|>", "</s>", "<|im_end|>"]
            for marker in end_markers:
                if marker in chunk.delta_content:
                    return True

        return False

    def process_stream(
        self,
        stream: Iterator[LLMChunk],
        callback: Optional[Any] = None,
    ) -> StreamBuffer:
        """
        处理完整的流式响应

        Args:
            stream: 流式响应迭代器
            callback: 每个块的回调函数

        Returns:
            StreamBuffer: 包含完整结果的缓冲区
        """
        self.reset()

        try:
            for chunk in stream:
                self.process_chunk(chunk)
                if callback:
                    try:
                        callback(chunk, self._buffer)
                    except Exception:
                        pass

                if self._buffer.is_complete:
                    break

        except Exception as e:
            self._buffer.is_complete = True
            self._buffer.end_time = time.time()
            self._buffer.metadata["error"] = str(e)

            if self._on_error:
                try:
                    self._on_error(e, self._buffer)
                except Exception:
                    pass

        return self._buffer

    def get_function_call(self) -> Optional[Dict[str, Any]]:
        """
        获取解析后的函数调用

        Returns:
            函数调用字典，包含name和arguments
        """
        if not self._buffer.tool_calls:
            return None

        tc = self._buffer.tool_calls[0]
        func = tc.get("function", {})
        args_str = func.get("arguments", "{}")

        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}

        return {
            "name": func.get("name", ""),
            "arguments": args,
            "id": tc.get("id", ""),
        }

    def get_all_function_calls(self) -> List[Dict[str, Any]]:
        """
        获取所有解析后的函数调用

        Returns:
            函数调用字典列表
        """
        result = []
        for tc in self._buffer.tool_calls:
            func = tc.get("function", {})
            args_str = func.get("arguments", "{}")

            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {}

            result.append({
                "name": func.get("name", ""),
                "arguments": args,
                "id": tc.get("id", ""),
            })

        return result

    def is_tool_call_response(self) -> bool:
        """判断响应是否包含工具调用"""
        return len(self._buffer.tool_calls) > 0

    def get_content_without_tool_calls(self) -> str:
        """获取不含工具调用的纯文本内容"""
        return self._buffer.full_content

    def create_stream_filter(
        self,
        include_content: bool = True,
        include_tool_calls: bool = True,
        include_usage: bool = False,
    ) -> Iterator[LLMChunk]:
        """
        创建流式过滤器生成器

        Args:
            include_content: 是否包含内容块
            include_tool_calls: 是否包含工具调用块
            include_usage: 是否包含使用量块

        Returns:
            过滤后的流式块生成器
        """
        # 返回一个工厂函数，用于包装流
        def _filter(stream: Iterator[LLMChunk]) -> Iterator[LLMChunk]:
            for chunk in stream:
                if chunk.delta_content and not include_content:
                    continue
                if chunk.tool_calls_delta and not include_tool_calls:
                    continue
                if chunk.usage and not include_usage:
                    # 保留finish_reason但移除usage
                    chunk = LLMChunk(
                        delta_content=chunk.delta_content,
                        finish_reason=chunk.finish_reason,
                        tool_calls_delta=chunk.tool_calls_delta,
                        metadata=chunk.metadata,
                    )
                yield chunk

        return _filter

    @staticmethod
    def merge_chunks(chunks: List[LLMChunk]) -> StreamBuffer:
        """
        合并多个流式块为一个缓冲区

        Args:
            chunks: 流式块列表

        Returns:
            StreamBuffer: 合并后的缓冲区
        """
        processor = StreamProcessor()
        for chunk in chunks:
            processor.process_chunk(chunk)
        return processor.buffer

    @staticmethod
    def extract_text_from_chunks(chunks: List[LLMChunk]) -> str:
        """
        从流式块列表中提取完整文本

        Args:
            chunks: 流式块列表

        Returns:
            str: 完整文本
        """
        return "".join(chunk.delta_content for chunk in chunks if chunk.delta_content)
