"""
工作流循环构造模块

提供高级循环控制结构，包括 for-each 循环、while 循环、do-while 循环，
支持循环变量绑定、迭代跟踪、最大迭代限制、break/continue 控制流。

Classes:
    ForEachLoop: for-each 循环执行器
    WhileLoop: while 循环执行器
    DoWhileLoop: do-while 循环执行器
    LoopVariable: 循环变量绑定
    IterationTracker: 迭代跟踪器
    LoopConfig: 循环配置
    BreakSignal: break 信号
    ContinueSignal: continue 信号
    LoopResult: 循环执行结果
"""

import copy
import time
import threading
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


# ============================================================
# 信号类
# ============================================================

class BreakSignal(Exception):
    """
    break 信号异常

    在循环体中抛出此异常以中断循环，类似 Python 的 break 语句。

    Attributes:
        loop_id: 触发 break 的循环 ID
        iteration: 当前迭代次数
        message: 可选的描述信息
    """

    def __init__(
        self,
        loop_id: str = "",
        iteration: int = 0,
        message: str = "",
    ) -> None:
        self.loop_id = loop_id
        self.iteration = iteration
        self.message = message
        super().__init__(f"BreakSignal(loop={loop_id}, iteration={iteration}, msg={message})")


class ContinueSignal(Exception):
    """
    continue 信号异常

    在循环体中抛出此异常以跳过当前迭代，类似 Python 的 continue 语句。

    Attributes:
        loop_id: 触发 continue 的循环 ID
        iteration: 当前迭代次数
        message: 可选的描述信息
    """

    def __init__(
        self,
        loop_id: str = "",
        iteration: int = 0,
        message: str = "",
    ) -> None:
        self.loop_id = loop_id
        self.iteration = iteration
        self.message = message
        super().__init__(
            f"ContinueSignal(loop={loop_id}, iteration={iteration}, msg={message})"
        )


# ============================================================
# 循环配置
# ============================================================

class LoopConfig:
    """
    循环配置

    Attributes:
        max_iterations: 最大迭代次数限制
        timeout_seconds: 循环总超时时间（秒）
        collect_results: 是否收集每次迭代的结果
        fail_fast: 遇到错误时是否立即停止
        continue_on_error: 遇到错误时是否继续下一次迭代
        max_concurrent: 最大并发迭代数（仅 for-each）
        iteration_delay: 每次迭代之间的延迟（秒）
    """

    def __init__(
        self,
        max_iterations: int = 1000,
        timeout_seconds: Optional[float] = None,
        collect_results: bool = True,
        fail_fast: bool = True,
        continue_on_error: bool = False,
        max_concurrent: int = 1,
        iteration_delay: float = 0.0,
    ) -> None:
        self.max_iterations = max(1, max_iterations)
        self.timeout_seconds = timeout_seconds
        self.collect_results = collect_results
        self.fail_fast = fail_fast
        self.continue_on_error = continue_on_error
        self.max_concurrent = max(1, max_concurrent)
        self.iteration_delay = max(0.0, iteration_delay)

    def validate(self) -> List[str]:
        """验证配置有效性，返回错误列表"""
        errors: List[str] = []
        if self.max_iterations < 1:
            errors.append("max_iterations 必须 >= 1")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            errors.append("timeout_seconds 必须 > 0")
        if self.max_concurrent < 1:
            errors.append("max_concurrent 必须 >= 1")
        if self.iteration_delay < 0:
            errors.append("iteration_delay 必须 >= 0")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "max_iterations": self.max_iterations,
            "timeout_seconds": self.timeout_seconds,
            "collect_results": self.collect_results,
            "fail_fast": self.fail_fast,
            "continue_on_error": self.continue_on_error,
            "max_concurrent": self.max_concurrent,
            "iteration_delay": self.iteration_delay,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoopConfig":
        """从字典反序列化"""
        return cls(
            max_iterations=data.get("max_iterations", 1000),
            timeout_seconds=data.get("timeout_seconds"),
            collect_results=data.get("collect_results", True),
            fail_fast=data.get("fail_fast", True),
            continue_on_error=data.get("continue_on_error", False),
            max_concurrent=data.get("max_concurrent", 1),
            iteration_delay=data.get("iteration_delay", 0.0),
        )


# ============================================================
# 循环变量
# ============================================================

class LoopVariable:
    """
    循环变量绑定

    管理循环执行过程中的变量作用域和绑定。

    Attributes:
        name: 变量名
        value: 当前值
        index: 当前索引
        total: 总迭代数
        is_first: 是否为第一次迭代
        is_last: 是否为最后一次迭代
        previous_value: 上一次迭代的值
    """

    def __init__(
        self,
        name: str,
        value: Any = None,
        index: int = 0,
        total: int = 0,
    ) -> None:
        self.name = name
        self.value = value
        self.index = index
        self.total = total
        self.is_first: bool = index == 0
        self.is_last: bool = index == max(0, total - 1)
        self.previous_value: Any = None

    def update(
        self,
        value: Any,
        index: int,
        total: int,
    ) -> None:
        """更新变量状态"""
        self.previous_value = self.value
        self.value = value
        self.index = index
        self.total = total
        self.is_first = index == 0
        self.is_last = index == max(0, total - 1)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "name": self.name,
            "value": self.value,
            "index": self.index,
            "total": self.total,
            "is_first": self.is_first,
            "is_last": self.is_last,
            "previous_value": self.previous_value,
        }

    def bind_to_context(self, context: Dict[str, Any]) -> None:
        """将循环变量绑定到执行上下文"""
        context[self.name] = self.value
        context[f"{self.name}_index"] = self.index
        context[f"{self.name}_total"] = self.total
        context[f"{self.name}_is_first"] = self.is_first
        context[f"{self.name}_is_last"] = self.is_last
        context[f"{self.name}_previous"] = self.previous_value


# ============================================================
# 迭代跟踪器
# ============================================================

@dataclass
class IterationRecord:
    """单次迭代记录"""
    iteration: int
    start_time: float
    end_time: float = 0.0
    duration: float = 0.0
    success: bool = True
    error: Optional[str] = None
    output: Any = None
    skipped: bool = False


class IterationTracker:
    """
    迭代跟踪器

    跟踪循环每次迭代的执行状态、时间和结果。

    Attributes:
        loop_id: 循环标识
        records: 迭代记录列表
        start_time: 循环开始时间
        end_time: 循环结束时间
    """

    def __init__(self, loop_id: str = "") -> None:
        self.loop_id = loop_id
        self.records: List[IterationRecord] = []
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self._lock = threading.Lock()

    def start_loop(self) -> None:
        """标记循环开始"""
        self.start_time = time.time()

    def end_loop(self) -> None:
        """标记循环结束"""
        self.end_time = time.time()

    def begin_iteration(self, iteration: int) -> IterationRecord:
        """开始一次迭代，返回记录对象"""
        record = IterationRecord(
            iteration=iteration,
            start_time=time.time(),
        )
        with self._lock:
            self.records.append(record)
        return record

    def complete_iteration(
        self,
        iteration: int,
        success: bool = True,
        error: Optional[str] = None,
        output: Any = None,
    ) -> None:
        """完成一次迭代"""
        now = time.time()
        with self._lock:
            for record in self.records:
                if record.iteration == iteration and record.end_time == 0.0:
                    record.end_time = now
                    record.duration = now - record.start_time
                    record.success = success
                    record.error = error
                    record.output = output
                    break

    def skip_iteration(self, iteration: int) -> None:
        """标记一次迭代为跳过"""
        now = time.time()
        with self._lock:
            for record in self.records:
                if record.iteration == iteration and record.end_time == 0.0:
                    record.end_time = now
                    record.duration = now - record.start_time
                    record.skipped = True
                    break

    @property
    def total_duration(self) -> float:
        """循环总执行时间"""
        if self.start_time == 0:
            return 0.0
        end = self.end_time if self.end_time > 0 else time.time()
        return end - self.start_time

    @property
    def completed_count(self) -> int:
        """成功完成的迭代数"""
        return sum(1 for r in self.records if r.success and not r.skipped)

    @property
    def failed_count(self) -> int:
        """失败的迭代数"""
        return sum(1 for r in self.records if not r.success and not r.skipped)

    @property
    def skipped_count(self) -> int:
        """跳过的迭代数"""
        return sum(1 for r in self.records if r.skipped)

    @property
    def total_iterations(self) -> int:
        """总迭代数"""
        return len(self.records)

    @property
    def average_duration(self) -> float:
        """平均迭代时间"""
        completed = [r for r in self.records if r.duration > 0]
        if not completed:
            return 0.0
        return sum(r.duration for r in completed) / len(completed)

    def get_record(self, iteration: int) -> Optional[IterationRecord]:
        """获取指定迭代的记录"""
        for record in self.records:
            if record.iteration == iteration:
                return record
        return None

    def get_failed_records(self) -> List[IterationRecord]:
        """获取所有失败的迭代记录"""
        return [r for r in self.records if not r.success and not r.skipped]

    def summary(self) -> Dict[str, Any]:
        """生成迭代摘要"""
        return {
            "loop_id": self.loop_id,
            "total_iterations": self.total_iterations,
            "completed": self.completed_count,
            "failed": self.failed_count,
            "skipped": self.skipped_count,
            "total_duration": round(self.total_duration, 4),
            "average_duration": round(self.average_duration, 4),
        }


# ============================================================
# 循环结果
# ============================================================

@dataclass
class LoopResult:
    """
    循环执行结果

    Attributes:
        loop_id: 循环标识
        loop_type: 循环类型
        results: 所有迭代的输出结果列表
        total_iterations: 总迭代次数
        completed_iterations: 成功完成的迭代次数
        failed_iterations: 失败的迭代次数
        skipped_iterations: 跳过的迭代次数
        broken: 是否通过 break 中断
        duration: 总执行时间
        errors: 错误列表
        tracker: 迭代跟踪器
    """
    loop_id: str = ""
    loop_type: str = ""
    results: List[Any] = dataclass_field(default_factory=list)
    total_iterations: int = 0
    completed_iterations: int = 0
    failed_iterations: int = 0
    skipped_iterations: int = 0
    broken: bool = False
    duration: float = 0.0
    errors: List[str] = dataclass_field(default_factory=list)
    tracker: Optional[IterationTracker] = None

    @property
    def success(self) -> bool:
        """是否全部成功"""
        return self.failed_iterations == 0 and self.total_iterations > 0

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "loop_id": self.loop_id,
            "loop_type": self.loop_type,
            "total_iterations": self.total_iterations,
            "completed_iterations": self.completed_iterations,
            "failed_iterations": self.failed_iterations,
            "skipped_iterations": self.skipped_iterations,
            "broken": self.broken,
            "duration": round(self.duration, 4),
            "success": self.success,
            "errors": self.errors,
            "results": self.results,
        }


# ============================================================
# For-Each 循环
# ============================================================

class ForEachLoop:
    """
    for-each 循环执行器

    遍历可迭代对象，对每个元素执行循环体。

    Usage:
        loop = ForEachLoop(
            loop_id="process_items",
            items=[1, 2, 3, 4, 5],
            body=lambda item, index, ctx: item * 2,
            config=LoopConfig(max_iterations=100),
        )
        result = loop.execute(context)
    """

    def __init__(
        self,
        loop_id: str,
        items: Optional[Sequence[Any]] = None,
        items_key: str = "",
        items_provider: Optional[Callable[[Dict[str, Any]], Sequence[Any]]] = None,
        body: Optional[Callable[[Any, int, Dict[str, Any]], Any]] = None,
        variable_name: str = "item",
        config: Optional[LoopConfig] = None,
    ) -> None:
        self.loop_id = loop_id
        self._items = items
        self.items_key = items_key
        self.items_provider = items_provider
        self.body = body
        self.variable_name = variable_name
        self.config = config or LoopConfig()
        self.tracker = IterationTracker(loop_id=loop_id)

    def _resolve_items(self, context: Dict[str, Any]) -> Sequence[Any]:
        """解析循环项"""
        if self.items_provider is not None:
            return self.items_provider(context)
        if self._items is not None:
            return self._items
        if self.items_key and self.items_key in context:
            return context[self.items_key]
        raise ValueError(
            f"ForEachLoop '{self.loop_id}': 未指定循环项，"
            f"请设置 items、items_key 或 items_provider"
        )

    def execute(self, context: Optional[Dict[str, Any]] = None) -> LoopResult:
        """
        执行 for-each 循环

        Args:
            context: 执行上下文

        Returns:
            LoopResult 包含所有迭代结果
        """
        ctx = context or {}
        items = self._resolve_items(ctx)
        total = len(items)

        # 应用最大迭代限制
        effective_total = min(total, self.config.max_iterations)
        if effective_total < total:
            items = list(items[:effective_total])

        self.tracker = IterationTracker(loop_id=self.loop_id)
        self.tracker.start_loop()

        results: List[Any] = []
        errors: List[str] = []
        broken = False
        completed = 0
        failed = 0
        skipped = 0

        loop_var = LoopVariable(name=self.variable_name, total=effective_total)

        for i, item in enumerate(items):
            # 检查超时
            if (self.config.timeout_seconds is not None
                    and self.tracker.total_duration >= self.config.timeout_seconds):
                errors.append(
                    f"循环超时 ({self.config.timeout_seconds}s)，"
                    f"在迭代 {i} 处停止"
                )
                break

            record = self.tracker.begin_iteration(i)
            loop_var.update(value=item, index=i, total=effective_total)
            loop_var.bind_to_context(ctx)

            try:
                # 迭代延迟
                if self.config.iteration_delay > 0 and i > 0:
                    time.sleep(self.config.iteration_delay)

                if self.body is not None:
                    output = self.body(item, i, ctx)
                else:
                    output = {"item": item, "index": i}

                results.append(output)
                completed += 1
                self.tracker.complete_iteration(i, success=True, output=output)

            except BreakSignal as bs:
                broken = True
                self.tracker.complete_iteration(i, success=True)
                break

            except ContinueSignal:
                skipped += 1
                self.tracker.skip_iteration(i)
                continue

            except Exception as e:
                failed += 1
                error_msg = f"迭代 {i} 失败: {str(e)}"
                errors.append(error_msg)
                self.tracker.complete_iteration(i, success=False, error=error_msg)

                if self.config.fail_fast:
                    break
                if not self.config.continue_on_error:
                    break

        self.tracker.end_loop()

        return LoopResult(
            loop_id=self.loop_id,
            loop_type="for_each",
            results=results,
            total_iterations=len(self.tracker.records),
            completed_iterations=completed,
            failed_iterations=failed,
            skipped_iterations=skipped,
            broken=broken,
            duration=self.tracker.total_duration,
            errors=errors,
            tracker=self.tracker,
        )


# ============================================================
# While 循环
# ============================================================

class WhileLoop:
    """
    while 循环执行器

    在条件为真时重复执行循环体。

    Usage:
        loop = WhileLoop(
            loop_id="poll_until_done",
            condition=lambda ctx: ctx.get("status") != "done",
            body=lambda iteration, ctx: poll_status(ctx),
            config=LoopConfig(max_iterations=100),
        )
        result = loop.execute(context)
    """

    def __init__(
        self,
        loop_id: str,
        condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
        body: Optional[Callable[[int, Dict[str, Any]], Any]] = None,
        config: Optional[LoopConfig] = None,
    ) -> None:
        self.loop_id = loop_id
        self.condition = condition
        self.body = body
        self.config = config or LoopConfig()
        self.tracker = IterationTracker(loop_id=loop_id)

    def execute(self, context: Optional[Dict[str, Any]] = None) -> LoopResult:
        """
        执行 while 循环

        Args:
            context: 执行上下文

        Returns:
            LoopResult 包含所有迭代结果
        """
        ctx = context or {}

        if self.condition is None:
            raise ValueError(
                f"WhileLoop '{self.loop_id}': 必须设置 condition 回调"
            )

        self.tracker = IterationTracker(loop_id=self.loop_id)
        self.tracker.start_loop()

        results: List[Any] = []
        errors: List[str] = []
        broken = False
        completed = 0
        failed = 0
        skipped = 0
        iteration = 0

        while iteration < self.config.max_iterations:
            # 检查超时
            if (self.config.timeout_seconds is not None
                    and self.tracker.total_duration >= self.config.timeout_seconds):
                errors.append(
                    f"循环超时 ({self.config.timeout_seconds}s)，"
                    f"在迭代 {iteration} 处停止"
                )
                break

            # 评估条件
            try:
                should_continue = self.condition(ctx)
            except Exception as e:
                errors.append(f"条件评估失败（迭代 {iteration}）: {str(e)}")
                break

            if not should_continue:
                break

            record = self.tracker.begin_iteration(iteration)

            # 绑定循环变量到上下文
            ctx["_while_iteration"] = iteration
            ctx["_while_is_first"] = iteration == 0

            try:
                # 迭代延迟
                if self.config.iteration_delay > 0 and iteration > 0:
                    time.sleep(self.config.iteration_delay)

                if self.body is not None:
                    output = self.body(iteration, ctx)
                else:
                    output = {"iteration": iteration}

                results.append(output)
                completed += 1
                self.tracker.complete_iteration(
                    iteration, success=True, output=output
                )

            except BreakSignal:
                broken = True
                self.tracker.complete_iteration(iteration, success=True)
                break

            except ContinueSignal:
                skipped += 1
                self.tracker.skip_iteration(iteration)
                iteration += 1
                continue

            except Exception as e:
                failed += 1
                error_msg = f"迭代 {iteration} 失败: {str(e)}"
                errors.append(error_msg)
                self.tracker.complete_iteration(
                    iteration, success=False, error=error_msg
                )

                if self.config.fail_fast:
                    break
                if not self.config.continue_on_error:
                    break

            iteration += 1

        self.tracker.end_loop()

        return LoopResult(
            loop_id=self.loop_id,
            loop_type="while",
            results=results,
            total_iterations=len(self.tracker.records),
            completed_iterations=completed,
            failed_iterations=failed,
            skipped_iterations=skipped,
            broken=broken,
            duration=self.tracker.total_duration,
            errors=errors,
            tracker=self.tracker,
        )


# ============================================================
# Do-While 循环
# ============================================================

class DoWhileLoop:
    """
    do-while 循环执行器

    先执行一次循环体，然后在条件为真时继续循环。
    保证循环体至少执行一次。

    Usage:
        loop = DoWhileLoop(
            loop_id="retry_operation",
            condition=lambda ctx: ctx.get("success") is not True,
            body=lambda iteration, ctx: attempt_operation(ctx),
            config=LoopConfig(max_iterations=5),
        )
        result = loop.execute(context)
    """

    def __init__(
        self,
        loop_id: str,
        condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
        body: Optional[Callable[[int, Dict[str, Any]], Any]] = None,
        config: Optional[LoopConfig] = None,
    ) -> None:
        self.loop_id = loop_id
        self.condition = condition
        self.body = body
        self.config = config or LoopConfig()
        self.tracker = IterationTracker(loop_id=loop_id)

    def execute(self, context: Optional[Dict[str, Any]] = None) -> LoopResult:
        """
        执行 do-while 循环

        Args:
            context: 执行上下文

        Returns:
            LoopResult 包含所有迭代结果
        """
        ctx = context or {}

        if self.condition is None:
            raise ValueError(
                f"DoWhileLoop '{self.loop_id}': 必须设置 condition 回调"
            )

        self.tracker = IterationTracker(loop_id=self.loop_id)
        self.tracker.start_loop()

        results: List[Any] = []
        errors: List[str] = []
        broken = False
        completed = 0
        failed = 0
        skipped = 0
        iteration = 0

        while True:
            # 检查最大迭代次数
            if iteration >= self.config.max_iterations:
                errors.append(
                    f"达到最大迭代次数 ({self.config.max_iterations})"
                )
                break

            # 检查超时
            if (self.config.timeout_seconds is not None
                    and self.tracker.total_duration >= self.config.timeout_seconds):
                errors.append(
                    f"循环超时 ({self.config.timeout_seconds}s)，"
                    f"在迭代 {iteration} 处停止"
                )
                break

            record = self.tracker.begin_iteration(iteration)

            # 绑定循环变量到上下文
            ctx["_do_while_iteration"] = iteration
            ctx["_do_while_is_first"] = iteration == 0

            try:
                # 迭代延迟
                if self.config.iteration_delay > 0 and iteration > 0:
                    time.sleep(self.config.iteration_delay)

                if self.body is not None:
                    output = self.body(iteration, ctx)
                else:
                    output = {"iteration": iteration}

                results.append(output)
                completed += 1
                self.tracker.complete_iteration(
                    iteration, success=True, output=output
                )

            except BreakSignal:
                broken = True
                self.tracker.complete_iteration(iteration, success=True)
                break

            except ContinueSignal:
                skipped += 1
                self.tracker.skip_iteration(iteration)
                iteration += 1
                continue

            except Exception as e:
                failed += 1
                error_msg = f"迭代 {iteration} 失败: {str(e)}"
                errors.append(error_msg)
                self.tracker.complete_iteration(
                    iteration, success=False, error=error_msg
                )

                if self.config.fail_fast:
                    break
                if not self.config.continue_on_error:
                    break

            # 评估条件（在循环体执行之后）
            try:
                should_continue = self.condition(ctx)
            except Exception as e:
                errors.append(f"条件评估失败（迭代 {iteration}）: {str(e)}")
                break

            if not should_continue:
                break

            iteration += 1

        self.tracker.end_loop()

        return LoopResult(
            loop_id=self.loop_id,
            loop_type="do_while",
            results=results,
            total_iterations=len(self.tracker.records),
            completed_iterations=completed,
            failed_iterations=failed,
            skipped_iterations=skipped,
            broken=broken,
            duration=self.tracker.total_duration,
            errors=errors,
            tracker=self.tracker,
        )


# ============================================================
# 辅助函数
# ============================================================

def create_foreach(
    loop_id: str,
    items: Sequence[Any],
    body: Callable[[Any, int, Dict[str, Any]], Any],
    max_iterations: int = 1000,
    variable_name: str = "item",
) -> ForEachLoop:
    """
    快速创建 for-each 循环的工厂函数

    Args:
        loop_id: 循环标识
        items: 要遍历的项
        body: 循环体函数 (item, index, context) -> result
        max_iterations: 最大迭代次数
        variable_name: 循环变量名

    Returns:
        ForEachLoop 实例
    """
    return ForEachLoop(
        loop_id=loop_id,
        items=items,
        body=body,
        variable_name=variable_name,
        config=LoopConfig(max_iterations=max_iterations),
    )


def create_while(
    loop_id: str,
    condition: Callable[[Dict[str, Any]], bool],
    body: Callable[[int, Dict[str, Any]], Any],
    max_iterations: int = 1000,
) -> WhileLoop:
    """
    快速创建 while 循环的工厂函数

    Args:
        loop_id: 循环标识
        condition: 条件函数 (context) -> bool
        body: 循环体函数 (iteration, context) -> result
        max_iterations: 最大迭代次数

    Returns:
        WhileLoop 实例
    """
    return WhileLoop(
        loop_id=loop_id,
        condition=condition,
        body=body,
        config=LoopConfig(max_iterations=max_iterations),
    )


def create_do_while(
    loop_id: str,
    condition: Callable[[Dict[str, Any]], bool],
    body: Callable[[int, Dict[str, Any]], Any],
    max_iterations: int = 1000,
) -> DoWhileLoop:
    """
    快速创建 do-while 循环的工厂函数

    Args:
        loop_id: 循环标识
        condition: 条件函数 (context) -> bool
        body: 循环体函数 (iteration, context) -> result
        max_iterations: 最大迭代次数

    Returns:
        DoWhileLoop 实例
    """
    return DoWhileLoop(
        loop_id=loop_id,
        condition=condition,
        body=body,
        config=LoopConfig(max_iterations=max_iterations),
    )
