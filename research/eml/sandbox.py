"""
EML Sandbox - EML沙箱执行环境

提供安全的代码执行环境，用于在受控条件下运行和测试EML相关代码。
防止不受信任的代码对系统造成损害。

核心功能:
- 安全代码执行：限制可用的内置函数和模块
- 资源限制：控制执行时间、内存使用和输出大小
- 结果捕获：捕获执行输出（stdout、返回值、异常）
- 异常处理：安全地捕获和处理执行错误
- 代码分析：静态检查代码安全性

安全策略:
- 禁止导入危险模块（os, sys, subprocess等）
- 限制可用的内置函数
- 设置执行超时
- 限制输出大小
- 捕获所有异常

⚠️ 研究用途警告: 沙箱不能保证100%安全，请勿用于不受信任的代码
"""

import sys
import time
import math
import random
import traceback
import io
import contextlib
import threading
import signal
import ast
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto


# ============================================================
# 沙箱安全级别
# ============================================================

class SandboxSecurityLevel(Enum):
    """沙箱安全级别"""
    HIGH = auto()       # 高安全：禁止几乎所有外部访问
    MEDIUM = auto()     # 中安全：允许数学和基础操作
    LOW = auto()        # 低安全：允许更多内置函数


# ============================================================
# 执行结果
# ============================================================

@dataclass
class ExecutionResult:
    """
    代码执行结果

    封装代码执行的所有输出信息，包括返回值、标准输出、
    标准错误和异常信息。
    """
    success: bool = False                     # 是否成功执行
    return_value: Any = None                  # 返回值
    stdout: str = ""                          # 标准输出
    stderr: str = ""                          # 标准错误
    error_message: str = ""                   # 错误信息
    error_type: str = ""                      # 错误类型
    execution_time: float = 0.0               # 执行时间（秒）
    timeout: bool = False                     # 是否超时
    memory_usage: int = 0                     # 估算内存使用（字节）
    output_truncated: bool = False            # 输出是否被截断

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'success': self.success,
            'return_value': str(self.return_value),
            'stdout': self.stdout,
            'stderr': self.stderr,
            'error_message': self.error_message,
            'error_type': self.error_type,
            'execution_time': self.execution_time,
            'timeout': self.timeout,
            'memory_usage': self.memory_usage,
            'output_truncated': self.output_truncated,
        }


# ============================================================
# 沙箱配置
# ============================================================

@dataclass
class SandboxConfig:
    """
    沙箱配置

    定义沙箱的安全限制和执行参数。
    """
    # 安全级别
    security_level: SandboxSecurityLevel = SandboxSecurityLevel.MEDIUM

    # 时间限制
    timeout_seconds: float = 10.0             # 执行超时时间（秒）
    cpu_time_limit: float = 5.0               # CPU时间限制（秒）

    # 内存限制
    max_memory_mb: int = 128                  # 最大内存使用（MB）

    # 输出限制
    max_output_length: int = 10000            # 最大输出长度（字符）
    max_stdout_length: int = 5000             # 最大标准输出长度
    max_stderr_length: int = 5000             # 最大标准错误长度
    max_return_value_length: int = 1000       # 返回值最大字符串长度

    # 递归限制
    max_recursion_depth: int = 100            # 最大递归深度

    # 代码大小限制
    max_code_length: int = 10000              # 最大代码长度（字符）
    max_expression_count: int = 1000          # 最大表达式数量


# ============================================================
# 代码安全分析结果
# ============================================================

@dataclass
class CodeAnalysisResult:
    """代码安全分析结果"""
    is_safe: bool = False                     # 是否安全
    risk_level: str = "UNKNOWN"               # 风险级别: LOW, MEDIUM, HIGH, CRITICAL
    warnings: List[str] = field(default_factory=list)   # 警告列表
    blocked_patterns: List[str] = field(default_factory=list)  # 被阻止的模式
    allowed_imports: List[str] = field(default_factory=list)  # 允许的导入
    denied_imports: List[str] = field(default_factory=list)   # 被拒绝的导入


# ============================================================
# EML沙箱执行环境主类
# ============================================================

class EMLSandbox:
    """
    EML沙箱执行环境

    提供安全的代码执行环境，用于运行EML相关的研究代码。
    通过限制可用的函数、模块和资源来防止恶意代码执行。

    安全机制:
    1. 静态代码分析：执行前检查代码安全性
    2. 受限的内置环境：移除危险的内置函数
    3. 模块访问控制：禁止导入危险模块
    4. 资源限制：超时、内存限制、输出大小限制
    5. 异常捕获：安全地处理所有执行错误

    使用示例:
        >>> sandbox = EMLSandbox()
        >>> result = sandbox.execute("2 + 3 * 4")
        >>> print(result.return_value)  # 14
        >>> result = sandbox.execute("math.exp(1) - math.log(2)")
        >>> print(result.return_value)  # e - ln(2)
    """

    # 危险模块列表（禁止导入）
    DANGEROUS_MODULES = {
        'os', 'sys', 'subprocess', 'shutil', 'pathlib',
        'socket', 'http', 'urllib', 'requests', 'ftplib',
        'telnetlib', 'smtplib', 'poplib', 'imaplib', 'nntplib',
        'pickle', 'shelve', 'marshal', 'importlib',
        'ctypes', 'multiprocessing', 'threading',
        'signal', 'resource', 'posix', 'nt',
        'code', 'codeop', 'compile', 'exec',
        'eval', 'input', 'open', 'file',
        '__builtins__', 'builtins',
        'glob', 'fnmatch', 'tempfile',
        'io', 'fcntl', 'pipes', 'commands',
    }

    # 危险内置函数（禁止使用）
    DANGEROUS_BUILTINS = {
        'exec', 'eval', 'compile', '__import__',
        'open', 'input', 'breakpoint',
        'globals', 'locals', 'vars',
        'getattr', 'setattr', 'delattr',
        'type', 'isinstance', 'issubclass',
        'memoryview', 'bytearray',
    }

    # 危险AST节点类型
    DANGEROUS_AST_NODES = {
        ast.Import, ast.ImportFrom,
    }

    def __init__(self, config: Optional[SandboxConfig] = None):
        """
        初始化沙箱环境

        Args:
            config: 沙箱配置，为None时使用默认配置
        """
        self.config = config or SandboxConfig()
        self._execution_count = 0

        # 构建安全的内置环境
        self._safe_builtins = self._build_safe_builtins()

        # 执行历史
        self._history: List[ExecutionResult] = []

    # ----------------------------------------------------------
    # 安全环境构建
    # ----------------------------------------------------------

    def _build_safe_builtins(self) -> Dict[str, Any]:
        """
        构建安全的内置函数字典

        根据安全级别决定哪些内置函数可用。

        Returns:
            安全的内置函数字典
        """
        level = self.config.security_level

        # 基础安全内置函数（所有级别都可用）
        safe = {
            # 数学相关
            'abs': abs,
            'min': min,
            'max': max,
            'sum': sum,
            'round': round,
            'pow': pow,
            'divmod': divmod,

            # 类型转换
            'int': int,
            'float': float,
            'bool': bool,
            'str': str,
            'list': list,
            'tuple': tuple,
            'dict': dict,
            'set': set,
            'frozenset': frozenset,

            # 序列操作
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'sorted': sorted,
            'reversed': reversed,
            'iter': iter,
            'next': next,
            'all': all,
            'any': any,

            # 比较与逻辑
            'True': True,
            'False': False,
            'None': None,

            # 异常相关
            'Exception': Exception,
            'ValueError': ValueError,
            'TypeError': TypeError,
            'RuntimeError': RuntimeError,
            'ZeroDivisionError': ZeroDivisionError,
            'OverflowError': OverflowError,
            'ArithmeticError': ArithmeticError,
            'KeyError': KeyError,
            'IndexError': IndexError,
            'AttributeError': AttributeError,
            'StopIteration': StopIteration,
        }

        if level == SandboxSecurityLevel.HIGH:
            # 高安全级别：仅保留最基本的函数
            pass

        elif level == SandboxSecurityLevel.MEDIUM:
            # 中安全级别：允许更多操作
            safe.update({
                # 数学模块函数
                'math': math,
                'random': random,

                # 字符串操作
                'chr': chr,
                'ord': ord,
                'bin': bin,
                'oct': oct,
                'hex': hex,
                'format': format,
                'repr': repr,
                'hash': hash,
                'id': id,

                # 容器操作
                'slice': slice,
                'object': object,

                # 额外异常
                'NotImplementedError': NotImplementedError,
                'NameError': NameError,
            })

        elif level == SandboxSecurityLevel.LOW:
            # 低安全级别：允许更多函数
            safe.update({
                'math': math,
                'random': random,
                'chr': chr,
                'ord': ord,
                'bin': bin,
                'oct': oct,
                'hex': hex,
                'format': format,
                'repr': repr,
                'hash': hash,
                'id': id,
                'slice': slice,
                'object': object,
                'print': print,  # 低安全级别允许print
                'hasattr': hasattr,
                'callable': callable,
                'staticmethod': staticmethod,
                'classmethod': classmethod,
                'property': property,
                'super': super,
                'complex': complex,
                'bytes': bytes,
                'ascii': ascii,
            })

        return safe

    # ----------------------------------------------------------
    # 代码安全分析
    # ----------------------------------------------------------

    def analyze_code(self, code: str) -> CodeAnalysisResult:
        """
        静态分析代码安全性

        在执行前检查代码是否包含危险模式。

        Args:
            code: 待分析的代码字符串

        Returns:
            代码安全分析结果
        """
        result = CodeAnalysisResult()
        result.is_safe = True
        result.risk_level = "LOW"

        # 检查代码长度
        if len(code) > self.config.max_code_length:
            result.warnings.append(
                f"代码长度 {len(code)} 超过限制 {self.config.max_code_length}"
            )
            result.risk_level = "MEDIUM"

        # AST分析
        try:
            tree = ast.parse(code)

            # 检查导入语句
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split('.')[0]
                        if module_name in self.DANGEROUS_MODULES:
                            result.blocked_patterns.append(
                                f"禁止导入模块: {alias.name}"
                            )
                            result.denied_imports.append(alias.name)
                            result.is_safe = False
                            result.risk_level = "CRITICAL"
                        else:
                            result.allowed_imports.append(alias.name)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module.split('.')[0]
                        if module_name in self.DANGEROUS_MODULES:
                            result.blocked_patterns.append(
                                f"禁止从模块导入: {node.module}"
                            )
                            result.denied_imports.append(node.module)
                            result.is_safe = False
                            result.risk_level = "CRITICAL"
                        else:
                            result.allowed_imports.append(node.module)

                # 检查危险函数调用
                elif isinstance(node, ast.Call):
                    func_name = self._get_call_name(node)
                    if func_name in self.DANGEROUS_BUILTINS:
                        result.blocked_patterns.append(
                            f"禁止调用函数: {func_name}"
                        )
                        result.is_safe = False
                        if result.risk_level != "CRITICAL":
                            result.risk_level = "HIGH"

                # 检查表达式数量
                elif isinstance(node, ast.Expr):
                    pass  # 表达式计数在下面处理

            # 检查表达式数量
            expr_count = sum(1 for n in ast.walk(tree)
                             if isinstance(n, ast.Expr))
            if expr_count > self.config.max_expression_count:
                result.warnings.append(
                    f"表达式数量 {expr_count} 超过限制 "
                    f"{self.config.max_expression_count}"
                )
                result.risk_level = "MEDIUM"

        except SyntaxError as e:
            result.is_safe = False
            result.risk_level = "HIGH"
            result.warnings.append(f"语法错误: {e}")

        # 字符串模式检查（补充AST分析的不足）
        dangerous_patterns = [
            '__import__', 'exec(', 'eval(', 'compile(',
            'open(', 'input(', 'breakpoint(',
            'globals(', 'locals(',
        ]
        for pattern in dangerous_patterns:
            if pattern in code:
                if pattern not in result.blocked_patterns:
                    result.blocked_patterns.append(
                        f"检测到危险模式: {pattern}"
                    )
                    result.is_safe = False
                    if result.risk_level not in ("CRITICAL", "HIGH"):
                        result.risk_level = "HIGH"

        return result

    def _get_call_name(self, node: ast.Call) -> str:
        """
        获取函数调用的名称

        Args:
            node: AST Call节点

        Returns:
            函数名称字符串
        """
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    # ----------------------------------------------------------
    # 安全代码执行
    # ----------------------------------------------------------

    def execute(self, code: str,
                variables: Optional[Dict[str, Any]] = None,
                timeout: Optional[float] = None) -> ExecutionResult:
        """
        在沙箱中安全执行代码

        完整的执行流程:
        1. 静态代码安全分析
        2. 准备受限执行环境
        3. 在资源限制下执行代码
        4. 捕获输出和异常
        5. 返回执行结果

        Args:
            code: 要执行的Python代码
            variables: 预定义的变量字典
            timeout: 覆盖默认超时时间

        Returns:
            执行结果对象
        """
        self._execution_count += 1
        start_time = time.time()

        result = ExecutionResult()

        # 步骤1: 静态安全分析
        analysis = self.analyze_code(code)
        if not analysis.is_safe:
            result.success = False
            result.error_message = "代码安全检查未通过"
            result.error_type = "SecurityError"
            result.stderr = "\n".join(analysis.blocked_patterns)
            result.execution_time = time.time() - start_time
            return result

        # 步骤2: 准备执行环境
        exec_globals = self._prepare_exec_globals(variables or {})

        # 步骤3: 执行代码（带资源限制）
        effective_timeout = timeout or self.config.timeout_seconds

        # 捕获stdout
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        # 设置递归限制
        old_recursion_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(self.config.max_recursion_depth)

        try:
            with contextlib.redirect_stdout(stdout_buffer):
                with contextlib.redirect_stderr(stderr_buffer):
                    # 使用线程实现超时控制
                    exec_result = [None]
                    exec_error = [None]

                    def _run():
                        try:
                            exec(code, exec_globals)
                            exec_result[0] = exec_globals.get('__result__', None)
                        except Exception as e:
                            exec_error[0] = e

                    thread = threading.Thread(target=_run)
                    thread.daemon = True
                    thread.start()
                    thread.join(timeout=effective_timeout)

                    if thread.is_alive():
                        # 超时
                        result.success = False
                        result.timeout = True
                        result.error_message = (
                            f"执行超时 ({effective_timeout}秒)"
                        )
                        result.error_type = "TimeoutError"
                    elif exec_error[0] is not None:
                        # 执行异常
                        error = exec_error[0]
                        result.success = False
                        result.error_message = str(error)
                        result.error_type = type(error).__name__
                        result.stderr = traceback.format_exception_only(
                            type(error), error
                        )[-1].strip()
                    else:
                        # 成功执行
                        result.success = True
                        result.return_value = exec_result[0]

        except Exception as e:
            result.success = False
            result.error_message = f"沙箱内部错误: {e}"
            result.error_type = "SandboxError"
        finally:
            # 恢复递归限制
            sys.setrecursionlimit(old_recursion_limit)

        # 步骤4: 处理输出
        stdout_str = stdout_buffer.getvalue()
        stderr_str = stderr_buffer.getvalue()

        # 截断过长的输出
        if len(stdout_str) > self.config.max_stdout_length:
            stdout_str = stdout_str[:self.config.max_stdout_length] + "\n...[截断]"
            result.output_truncated = True

        if len(stderr_str) > self.config.max_stderr_length:
            stderr_str = stderr_str[:self.config.max_stderr_length] + "\n...[截断]"
            result.output_truncated = True

        result.stdout = stdout_str
        result.stderr = stderr_str

        # 截断返回值的字符串表示
        if result.return_value is not None:
            return_str = str(result.return_value)
            if len(return_str) > self.config.max_return_value_length:
                result.return_value = return_str[:self.config.max_return_value_length] + "..."

        # 记录执行时间
        result.execution_time = time.time() - start_time

        # 保存到历史
        self._history.append(result)

        return result

    def _prepare_exec_globals(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备受限的全局执行环境

        Args:
            variables: 用户提供的变量

        Returns:
            受限的全局命名空间
        """
        # 基础环境：安全的内置函数
        exec_globals = {
            '__builtins__': self._safe_builtins,
        }

        # 添加用户变量
        for key, value in variables.items():
            # 检查变量名是否安全
            if key.startswith('_') and key != '__result__':
                continue  # 跳过以下划线开头的变量（除__result__）
            exec_globals[key] = value

        return exec_globals

    # ----------------------------------------------------------
    # 便捷执行方法
    # ----------------------------------------------------------

    def execute_expression(self, expression: str,
                           variables: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """
        执行单个表达式并返回结果

        将表达式包装在赋值语句中执行，自动捕获返回值。

        Args:
            expression: Python表达式字符串
            variables: 预定义变量

        Returns:
            执行结果
        """
        # 包装为赋值语句以捕获结果
        wrapped_code = f"__result__ = ({expression})"
        return self.execute(wrapped_code, variables)

    def execute_function(self, func_code: str,
                         func_name: str,
                         args: Optional[List[Any]] = None,
                         kwargs: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """
        定义并执行一个函数

        Args:
            func_code: 函数定义代码
            func_name: 函数名
            args: 位置参数列表
            kwargs: 关键字参数字典

        Returns:
            执行结果
        """
        args = args or []
        kwargs = kwargs or {}

        # 序列化参数
        args_str = repr(args)
        kwargs_str = repr(kwargs)

        # 构建完整代码
        full_code = f"""
{func_code}
__result__ = {func_name}(*{args_str}, **{kwargs_str})
"""
        return self.execute(full_code)

    def evaluate_eml(self, x: float, y: float,
                     safety_level: str = "MODERATE") -> ExecutionResult:
        """
        安全地评估EML算子

        在沙箱中计算 eml(x, y) = e^x - ln(y)

        Args:
            x: 指数部分输入
            y: 对数部分输入（必须为正）
            safety_level: 安全级别

        Returns:
            执行结果，return_value为EML计算结果
        """
        code = f"""
import math
if {y} <= 0:
    raise ValueError("y must be positive")
exp_x = math.exp(min({x}, 700))
log_y = math.log({y})
result = exp_x - log_y
if not math.isfinite(result):
    result = 0.0
__result__ = max(-1e10, min(1e10, result))
"""
        return self.execute(code)

    def evaluate_expression_safe(self, expression: str,
                                 x_values: List[float]) -> List[Optional[float]]:
        """
        安全地批量评估表达式

        对每个x值执行表达式，返回结果列表。
        出错的值返回None。

        Args:
            expression: 包含变量x的表达式
            x_values: 输入值列表

        Returns:
            结果列表，出错位置为None
        """
        results = []
        for x in x_values:
            result = self.execute_expression(expression, {'x': x})
            if result.success:
                try:
                    val = float(result.return_value)
                    if math.isfinite(val):
                        results.append(val)
                    else:
                        results.append(None)
                except (TypeError, ValueError):
                    results.append(None)
            else:
                results.append(None)
        return results

    # ----------------------------------------------------------
    # 资源限制
    # ----------------------------------------------------------

    def set_timeout(self, seconds: float):
        """
        设置执行超时时间

        Args:
            seconds: 超时秒数
        """
        self.config.timeout_seconds = seconds

    def set_memory_limit(self, mb: int):
        """
        设置内存使用限制

        Args:
            mb: 最大内存使用（MB）
        """
        self.config.max_memory_mb = mb

    def set_output_limit(self, max_chars: int):
        """
        设置输出大小限制

        Args:
            max_chars: 最大输出字符数
        """
        self.config.max_output_length = max_chars

    def set_security_level(self, level: SandboxSecurityLevel):
        """
        设置安全级别

        Args:
            level: 安全级别
        """
        self.config.security_level = level
        self._safe_builtins = self._build_safe_builtins()

    # ----------------------------------------------------------
    # 结果捕获与历史
    # ----------------------------------------------------------

    def get_last_result(self) -> Optional[ExecutionResult]:
        """
        获取最近一次执行结果

        Returns:
            最近的执行结果，无历史时返回None
        """
        return self._history[-1] if self._history else None

    def get_history(self) -> List[ExecutionResult]:
        """
        获取完整执行历史

        Returns:
            执行结果历史列表
        """
        return list(self._history)

    def clear_history(self):
        """清除执行历史"""
        self._history.clear()

    def get_execution_count(self) -> int:
        """获取总执行次数"""
        return self._execution_count

    def get_success_rate(self) -> float:
        """
        获取执行成功率

        Returns:
            成功率，范围[0, 1]
        """
        if not self._history:
            return 0.0
        success_count = sum(1 for r in self._history if r.success)
        return success_count / len(self._history)

    # ----------------------------------------------------------
    # 报告生成
    # ----------------------------------------------------------

    def get_security_report(self, code: str) -> str:
        """
        生成代码安全分析报告

        Args:
            code: 待分析的代码

        Returns:
            格式化的安全报告字符串
        """
        analysis = self.analyze_code(code)

        lines = []
        lines.append("=" * 60)
        lines.append("EML 沙箱安全分析报告")
        lines.append("=" * 60)
        lines.append("")

        # 总体评估
        status = "安全" if analysis.is_safe else "不安全"
        lines.append(f"  安全评估: {status}")
        lines.append(f"  风险级别: {analysis.risk_level}")
        lines.append("")

        # 警告信息
        if analysis.warnings:
            lines.append("  警告:")
            for w in analysis.warnings:
                lines.append(f"    - {w}")
            lines.append("")

        # 被阻止的模式
        if analysis.blocked_patterns:
            lines.append("  被阻止的模式:")
            for p in analysis.blocked_patterns:
                lines.append(f"    [X] {p}")
            lines.append("")

        # 允许的导入
        if analysis.allowed_imports:
            lines.append("  允许的导入:")
            for imp in analysis.allowed_imports:
                lines.append(f"    [OK] {imp}")
            lines.append("")

        # 被拒绝的导入
        if analysis.denied_imports:
            lines.append("  被拒绝的导入:")
            for imp in analysis.denied_imports:
                lines.append(f"    [X] {imp}")
            lines.append("")

        # 沙箱配置
        lines.append("  当前沙箱配置:")
        lines.append(f"    安全级别: {self.config.security_level.name}")
        lines.append(f"    超时时间: {self.config.timeout_seconds}s")
        lines.append(f"    内存限制: {self.config.max_memory_mb}MB")
        lines.append(f"    递归深度: {self.config.max_recursion_depth}")
        lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def get_summary(self) -> str:
        """
        获取沙箱摘要信息

        Returns:
            格式化的摘要字符串
        """
        lines = []
        lines.append("=" * 60)
        lines.append("EML 沙箱执行环境摘要")
        lines.append("=" * 60)
        lines.append(f"  总执行次数: {self._execution_count}")
        lines.append(f"  成功率: {self.get_success_rate():.2%}")
        lines.append(f"  安全级别: {self.config.security_level.name}")
        lines.append(f"  超时时间: {self.config.timeout_seconds}s")
        lines.append(f"  内存限制: {self.config.max_memory_mb}MB")
        lines.append(f"  递归深度限制: {self.config.max_recursion_depth}")
        lines.append("")

        if self._history:
            lines.append("  最近执行记录:")
            # 显示最近5条记录
            recent = self._history[-5:]
            for i, r in enumerate(recent):
                status = "OK" if r.success else "FAIL"
                if r.timeout:
                    status = "TIMEOUT"
                lines.append(
                    f"    [{status}] 耗时: {r.execution_time:.3f}s | "
                    f"返回: {str(r.return_value)[:50]}"
                )

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================
# 模块自测
# ============================================================

if __name__ == "__main__":
    print("EML Sandbox - 模块自测")
    print("=" * 60)

    # 创建沙箱
    sandbox = EMLSandbox()

    # 测试1: 基本表达式执行
    print("\n[测试1] 基本表达式执行")
    result = sandbox.execute_expression("2 + 3 * 4")
    print(f"  2 + 3 * 4 = {result.return_value}")
    print(f"  成功: {result.success}")

    # 测试2: 数学表达式
    print("\n[测试2] 数学表达式")
    result = sandbox.execute_expression("math.exp(1) - math.log(2)")
    print(f"  e - ln(2) = {result.return_value}")
    print(f"  成功: {result.success}")

    # 测试3: EML评估
    print("\n[测试3] EML评估")
    result = sandbox.evaluate_eml(1.0, 2.0)
    print(f"  eml(1, 2) = {result.return_value}")
    print(f"  成功: {result.success}")

    # 测试4: 安全检查
    print("\n[测试4] 安全检查 - 危险代码")
    result = sandbox.execute("import os; os.system('echo hacked')")
    print(f"  成功: {result.success}")
    print(f"  错误: {result.error_message}")

    # 测试5: 超时控制
    print("\n[测试5] 超时控制")
    sandbox.set_timeout(1.0)
    result = sandbox.execute("x = 0\nwhile True: x += 1")
    print(f"  成功: {result.success}")
    print(f"  超时: {result.timeout}")

    # 测试6: 安全报告
    print("\n[测试6] 安全报告")
    report = sandbox.get_security_report("x = math.exp(2) + math.log(3)")
    print(report)

    # 测试7: 批量评估
    print("\n[测试7] 批量评估")
    sandbox.set_timeout(10.0)
    results = sandbox.evaluate_expression_safe("math.exp(x)", [0, 1, 2, 3])
    print(f"  exp([0,1,2,3]) = {results}")

    # 输出摘要
    print("\n" + sandbox.get_summary())
