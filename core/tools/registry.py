"""
工具注册中心模块

提供单例工具注册中心，支持注册、查询、搜索、执行、统计和导出功能。
仅使用 Python 标准库。
"""

import copy
import functools
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from .base import Tool, ToolResult


# ---------------------------------------------------------------------------
# 工具调用统计
# ---------------------------------------------------------------------------
@dataclass
class ToolCallStats:
    """单个工具的调用统计"""
    call_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_duration_ms: float = 0.0
    last_called: Optional[float] = None
    last_error: Optional[str] = None

    @property
    def success_rate(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.success_count / self.call_count

    @property
    def avg_duration_ms(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.total_duration_ms / self.call_count

    def record(self, result: ToolResult) -> None:
        self.call_count += 1
        self.total_duration_ms += result.duration_ms
        self.last_called = time.time()
        if result.success:
            self.success_count += 1
        else:
            self.failure_count += 1
            self.last_error = result.error


# ---------------------------------------------------------------------------
# ToolRegistry - 工具注册中心（线程安全单例）
# ---------------------------------------------------------------------------
class ToolRegistry:
    """工具注册中心

    支持工具的注册、注销、查询、搜索、执行和统计。
    线程安全，可通过 get_instance() 获取全局单例。
    """

    _instance: Optional["ToolRegistry"] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._tools: OrderedDict[str, Tool] = OrderedDict()
        self._stats: Dict[str, ToolCallStats] = {}
        self._lock = threading.RLock()
        self._hooks: Dict[str, List[Callable]] = {
            "before_register": [],
            "after_register": [],
            "before_unregister": [],
            "after_unregister": [],
            "before_execute": [],
            "after_execute": [],
        }

    # ----- 单例 -----

    @classmethod
    def get_instance(cls) -> "ToolRegistry":
        """获取全局单例"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """重置单例（主要用于测试）"""
        with cls._instance_lock:
            cls._instance = None

    # ----- 注册 / 注销 -----

    def register(self, tool: Tool) -> None:
        """注册工具"""
        with self._lock:
            self._run_hooks("before_register", tool)
            if tool.name in self._tools:
                raise ValueError(f"工具 '{tool.name}' 已注册")
            self._tools[tool.name] = tool
            self._stats[tool.name] = ToolCallStats()
            self._run_hooks("after_register", tool)

    def unregister(self, name: str) -> bool:
        """注销工具，返回是否成功"""
        with self._lock:
            if name not in self._tools:
                return False
            tool = self._tools[name]
            self._run_hooks("before_unregister", tool)
            del self._tools[name]
            del self._stats[name]
            self._run_hooks("after_unregister", tool)
            return True

    # ----- 查询 -----

    def get(self, name: str) -> Optional[Tool]:
        """按名称获取工具"""
        with self._lock:
            return self._tools.get(name)

    def has(self, name: str) -> bool:
        """检查工具是否已注册"""
        with self._lock:
            return name in self._tools

    def list_tools(self) -> List[str]:
        """列出所有工具名称"""
        with self._lock:
            return list(self._tools.keys())

    def list_tool_objects(self) -> List[Tool]:
        """列出所有工具对象"""
        with self._lock:
            return list(self._tools.values())

    def count(self) -> int:
        """已注册工具数量"""
        with self._lock:
            return len(self._tools)

    # ----- 搜索 / 过滤 -----

    def search(self, query: str) -> List[Tool]:
        """按名称/描述/标签模糊搜索工具"""
        query_lower = query.lower().strip()
        if not query_lower:
            return self.list_tool_objects()

        results: List[Tool] = []
        with self._lock:
            for tool in self._tools.values():
                if self._match_tool(tool, query_lower):
                    results.append(tool)
        return results

    def filter_by_tag(self, tags: Set[str]) -> List[Tool]:
        """按标签过滤工具"""
        if not tags:
            return self.list_tool_objects()
        with self._lock:
            return [
                t for t in self._tools.values() if t.tags & tags
            ]

    def filter_by_category(self, category: str) -> List[Tool]:
        """按类别过滤工具"""
        if not category:
            return self.list_tool_objects()
        with self._lock:
            return [
                t for t in self._tools.values()
                if t.category.lower() == category.lower()
            ]

    # ----- 执行 -----

    def execute_tool(
        self,
        name: str,
        params: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> ToolResult:
        """执行工具（含参数校验、超时、异常捕获和统计）"""
        params = params or {}
        with self._lock:
            tool = self._tools.get(name)
            if tool is None:
                return ToolResult.fail(
                    error=f"工具 '{name}' 未注册",
                    tool_name=name,
                    trace_id=uuid.uuid4().hex[:16],
                )

        self._run_hooks("before_execute", tool, params)

        effective_timeout = timeout if timeout is not None else tool.timeout
        result = self._execute_with_timeout(tool, params, effective_timeout)

        # 记录统计
        with self._lock:
            stats = self._stats.get(name)
            if stats is not None:
                stats.record(result)

        self._run_hooks("after_execute", tool, params, result)
        return result

    def _execute_with_timeout(
        self, tool: Tool, params: dict, timeout: float
    ) -> ToolResult:
        """带超时的执行（使用线程）"""
        result_container: List[Optional[ToolResult]] = [None]
        exception_container: List[Optional[Exception]] = [None]

        def _run():
            try:
                result_container[0] = tool.execute(params)
            except Exception as exc:
                exception_container[0] = exc

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            return ToolResult.fail(
                error=f"工具 '{tool.name}' 执行超时 ({timeout}s)",
                tool_name=tool.name,
                trace_id=uuid.uuid4().hex[:16],
            )

        if exception_container[0] is not None:
            exc = exception_container[0]
            return ToolResult.fail(
                error=f"{type(exc).__name__}: {exc}",
                tool_name=tool.name,
                trace_id=uuid.uuid4().hex[:16],
                metadata={"traceback": traceback.format_exc()},
            )

        return result_container[0] or ToolResult.fail(
            error="工具执行未返回结果",
            tool_name=tool.name,
        )

    # ----- 信息导出 -----

    def get_tools_info(self) -> List[dict]:
        """获取所有工具信息（用于 LLM function calling）"""
        with self._lock:
            return [tool.get_info() for tool in self._tools.values()]

    def export_schema(self) -> List[dict]:
        """导出 OpenAI function calling 格式的工具定义

        格式示例::

            [
                {
                    "type": "function",
                    "function": {
                        "name": "tool_name",
                        "description": "...",
                        "parameters": { ... }
                    }
                }
            ]
        """
        with self._lock:
            result = []
            for tool in self._tools.values():
                func_def = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters_schema,
                    },
                }
                result.append(func_def)
            return result

    # ----- 统计 -----

    def get_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """获取调用统计

        Args:
            name: 工具名称，为 None 时返回所有工具的统计汇总
        """
        with self._lock:
            if name is not None:
                stats = self._stats.get(name)
                if stats is None:
                    return {}
                return {
                    "name": name,
                    "call_count": stats.call_count,
                    "success_count": stats.success_count,
                    "failure_count": stats.failure_count,
                    "success_rate": round(stats.success_rate, 4),
                    "avg_duration_ms": round(stats.avg_duration_ms, 3),
                    "last_called": stats.last_called,
                    "last_error": stats.last_error,
                }

            # 汇总
            total_calls = 0
            total_success = 0
            total_failure = 0
            total_duration = 0.0
            per_tool = {}
            for tname, stats in self._stats.items():
                total_calls += stats.call_count
                total_success += stats.success_count
                total_failure += stats.failure_count
                total_duration += stats.total_duration_ms
                per_tool[tname] = {
                    "call_count": stats.call_count,
                    "success_rate": round(stats.success_rate, 4),
                    "avg_duration_ms": round(stats.avg_duration_ms, 3),
                }

            return {
                "total_calls": total_calls,
                "total_success": total_success,
                "total_failure": total_failure,
                "overall_success_rate": (
                    round(total_success / total_calls, 4)
                    if total_calls > 0
                    else 0.0
                ),
                "overall_avg_duration_ms": (
                    round(total_duration / total_calls, 3)
                    if total_calls > 0
                    else 0.0
                ),
                "tools": per_tool,
            }

    def reset_stats(self, name: Optional[str] = None) -> None:
        """重置统计"""
        with self._lock:
            if name is not None:
                if name in self._stats:
                    self._stats[name] = ToolCallStats()
            else:
                for tname in self._stats:
                    self._stats[tname] = ToolCallStats()

    # ----- 钩子 -----

    def add_hook(self, event: str, callback: Callable) -> None:
        """添加事件钩子

        支持的事件: before_register, after_register, before_unregister,
        after_unregister, before_execute, after_execute
        """
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)

    def remove_hook(self, event: str, callback: Callable) -> bool:
        """移除事件钩子"""
        if event not in self._hooks:
            return False
        try:
            self._hooks[event].remove(callback)
            return True
        except ValueError:
            return False

    def _run_hooks(self, event: str, *args, **kwargs) -> None:
        """执行钩子"""
        for callback in self._hooks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception:
                # 钩子异常不影响主流程
                pass

    # ----- 辅助 -----

    @staticmethod
    def _match_tool(tool: Tool, query_lower: str) -> bool:
        """判断工具是否匹配查询"""
        if query_lower in tool.name.lower():
            return True
        if query_lower in tool.description.lower():
            return True
        for tag in tool.tags:
            if query_lower in tag.lower():
                return True
        if query_lower in tool.category.lower():
            return True
        return False

    def clear(self) -> None:
        """清空所有已注册工具和统计"""
        with self._lock:
            self._tools.clear()
            self._stats.clear()
