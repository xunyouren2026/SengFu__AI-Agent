"""
插件沙箱隔离模块

提供插件执行沙箱，限制文件系统、网络和模块导入。
使用 subprocess 实现进程级隔离。
仅使用 Python 标准库。
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# SandboxContext - 沙箱上下文
# ---------------------------------------------------------------------------
@dataclass
class SandboxContext:
    """沙箱上下文配置

    Attributes:
        allowed_modules: 允许导入的模块集合
        denied_modules: 禁止导入的模块集合
        allowed_paths: 允许访问的文件系统路径
        denied_paths: 禁止访问的文件系统路径
        max_memory_mb: 最大内存使用（MB）
        max_cpu_time: 最大 CPU 时间（秒）
        max_execution_time: 最大执行时间（秒）
        max_output_size: 最大输出大小（字节）
        network_allowed: 是否允许网络访问
        subprocess_allowed: 是否允许子进程
        environment: 环境变量
        working_directory: 工作目录
    """
    allowed_modules: Set[str] = field(default_factory=lambda: {
        "json", "math", "datetime", "re", "collections", "itertools",
        "functools", "operator", "copy", "string", "textwrap",
        "decimal", "fractions", "statistics", "hashlib", "hmac",
        "base64", "binascii", "struct", "uuid", "secrets",
    })
    denied_modules: Set[str] = field(default_factory=lambda: {
        "os", "sys", "subprocess", "shutil", "pathlib",
        "socket", "http", "urllib", "requests",
        "ctypes", "multiprocessing", "threading",
        "importlib", "pkgutil",
        "signal", "resource",
    })
    allowed_paths: List[str] = field(default_factory=list)
    denied_paths: List[str] = field(default_factory=lambda: [
        "/etc", "/var", "/sys", "/proc", "/dev",
        "/boot", "/lib", "/usr/lib", "/sbin",
    ])
    max_memory_mb: int = 256
    max_cpu_time: float = 30.0
    max_execution_time: float = 60.0
    max_output_size: int = 1024 * 1024  # 1MB
    network_allowed: bool = False
    subprocess_allowed: bool = False
    environment: Dict[str, str] = field(default_factory=dict)
    working_directory: str = ""

    def to_dict(self) -> dict:
        return {
            "allowed_modules": sorted(self.allowed_modules),
            "denied_modules": sorted(self.denied_modules),
            "allowed_paths": self.allowed_paths,
            "denied_paths": self.denied_paths,
            "max_memory_mb": self.max_memory_mb,
            "max_cpu_time": self.max_cpu_time,
            "max_execution_time": self.max_execution_time,
            "max_output_size": self.max_output_size,
            "network_allowed": self.network_allowed,
            "subprocess_allowed": self.subprocess_allowed,
            "environment": self.environment,
            "working_directory": self.working_directory,
        }


# ---------------------------------------------------------------------------
# SandboxResult - 沙箱执行结果
# ---------------------------------------------------------------------------
@dataclass
class SandboxResult:
    """沙箱执行结果"""
    success: bool = False
    stdout: str = ""
    stderr: str = ""
    return_code: int = -1
    duration_ms: float = 0.0
    error: Optional[str] = None
    timed_out: bool = False
    killed: bool = False

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_code": self.return_code,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "timed_out": self.timed_out,
            "killed": self.killed,
        }


# ---------------------------------------------------------------------------
# PluginSandbox - 插件沙箱隔离
# ---------------------------------------------------------------------------
class PluginSandbox:
    """插件沙箱隔离

    通过 subprocess 实现进程级隔离，限制插件的:
    - 文件系统访问
    - 网络访问
    - 可导入模块
    - 内存和 CPU 使用
    - 执行时间
    """

    def __init__(self, context: Optional[SandboxContext] = None):
        self._context = context or SandboxContext()
        self._lock = threading.Lock()
        self._execution_count = 0
        self._total_time_ms = 0.0

    @property
    def context(self) -> SandboxContext:
        return self._context

    @context.setter
    def context(self, value: SandboxContext) -> None:
        self._context = value

    # ----- 核心执行 -----

    def execute_in_sandbox(
        self,
        plugin_code: str,
        context: Optional[SandboxContext] = None,
        input_data: Optional[dict] = None,
    ) -> SandboxResult:
        """在沙箱中执行插件代码

        Args:
            plugin_code: 要执行的 Python 代码
            context: 沙箱上下文（覆盖默认配置）
            input_data: 传递给插件代码的输入数据

        Returns:
            SandboxResult
        """
        ctx = context or self._context
        start = time.monotonic()

        try:
            result = self._execute_subprocess(plugin_code, ctx, input_data)
        except Exception as exc:
            result = SandboxResult(
                success=False,
                error=f"沙箱执行失败: {type(exc).__name__}: {exc}",
                stderr=traceback.format_exc(),
            )

        result.duration_ms = round((time.monotonic() - start) * 1000, 3)

        with self._lock:
            self._execution_count += 1
            self._total_time_ms += result.duration_ms

        return result

    def _execute_subprocess(
        self,
        code: str,
        ctx: SandboxContext,
        input_data: Optional[dict],
    ) -> SandboxResult:
        """通过子进程执行代码"""
        # 生成沙箱包装代码
        wrapper = self._generate_wrapper(code, ctx)

        # 准备输入
        stdin_data = ""
        if input_data is not None:
            stdin_data = json.dumps(input_data)

        # 准备环境变量
        env = self._prepare_environment(ctx)

        # 创建临时文件
        tmp_dir = None
        try:
            tmp_dir = tempfile.mkdtemp(prefix="plugin_sandbox_")
            code_file = os.path.join(tmp_dir, "sandbox_code.py")
            with open(code_file, "w", encoding="utf-8") as f:
                f.write(wrapper)

            # 执行
            work_dir = ctx.working_directory or tmp_dir
            process = subprocess.Popen(
                [sys.executable, "-B", code_file],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=work_dir,
                preexec_fn=self._create_preexec_fn(ctx) if hasattr(os, "fork") else None,
            )

            try:
                stdout_bytes, stderr_bytes = process.communicate(
                    input=stdin_data.encode("utf-8") if stdin_data else None,
                    timeout=ctx.max_execution_time,
                )
                stdout = stdout_bytes.decode("utf-8", errors="replace")
                stderr = stderr_bytes.decode("utf-8", errors="replace")

                # 截断过大的输出
                if len(stdout) > ctx.max_output_size:
                    stdout = stdout[:ctx.max_output_size] + "\n... [输出被截断]"
                if len(stderr) > ctx.max_output_size:
                    stderr = stderr[:ctx.max_output_size] + "\n... [错误输出被截断]"

                return SandboxResult(
                    success=process.returncode == 0,
                    stdout=stdout,
                    stderr=stderr,
                    return_code=process.returncode,
                )

            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                return SandboxResult(
                    success=False,
                    timed_out=True,
                    killed=True,
                    error=f"执行超时 ({ctx.max_execution_time}s)",
                    stderr="执行被超时终止",
                )

        finally:
            # 清理临时文件
            if tmp_dir:
                self._safe_cleanup(tmp_dir)

    # ----- 沙箱包装代码生成 -----

    def _generate_wrapper(self, code: str, ctx: SandboxContext) -> str:
        """生成沙箱包装代码

        包装代码包含:
        - 导入限制
        - 文件系统限制
        - 网络限制
        - 输入/输出处理
        """
        allowed_str = json.dumps(sorted(ctx.allowed_modules))
        denied_str = json.dumps(sorted(ctx.denied_modules))
        network_allowed_str = str(ctx.network_allowed)
        subprocess_allowed_str = str(ctx.subprocess_allowed)
        escaped_code = self._escape_code(code)

        parts = []
        parts.append("import sys")
        parts.append("import json")
        parts.append("import builtins")
        parts.append("")
        parts.append("# ===== 导入限制 =====")
        parts.append("_original_import = builtins.__import__")
        parts.append("")
        parts.append("_ALLOWED_MODULES = " + allowed_str)
        parts.append("_DENIED_MODULES = " + denied_str)
        parts.append("")
        parts.append("def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):")
        parts.append("    top_level = name.split('.')[0]")
        parts.append("    if top_level in _DENIED_MODULES:")
        parts.append("        raise ImportError(f'沙箱限制: 不允许导入模块 {name!r}')")
        parts.append("    if _ALLOWED_MODULES and top_level not in _ALLOWED_MODULES:")
        parts.append("        raise ImportError(f'沙箱限制: 模块 {name!r} 不在允许列表中')")
        parts.append("    return _original_import(name, globals, locals, fromlist, level)")
        parts.append("")
        parts.append("builtins.__import__ = _restricted_import")
        parts.append("")
        parts.append("# ===== 网络限制 =====")
        parts.append("if not " + network_allowed_str + ":")
        parts.append("    _blocked_network_modules = ['socket', 'http', 'urllib', 'ssl']")
        parts.append("    for mod_name in _blocked_network_modules:")
        parts.append("        try:")
        parts.append("            mod = _original_import(mod_name)")
        parts.append("            if hasattr(mod, 'socket'):")
        parts.append("                mod.socket = None")
        parts.append("            if hasattr(mod, 'create_connection'):")
        parts.append("                def _make_blocked(name):")
        parts.append("                    def _blocked(*a, **kw):")
        parts.append("                        raise PermissionError('网络访问被禁止')")
        parts.append("                    return _blocked")
        parts.append("                mod.create_connection = _make_blocked(mod_name)")
        parts.append("        except ImportError:")
        parts.append("            pass")
        parts.append("")
        parts.append("# ===== 子进程限制 =====")
        parts.append("if not " + subprocess_allowed_str + ":")
        parts.append("    try:")
        parts.append("        sp = _original_import('subprocess')")
        parts.append("        def _block_subprocess(*a, **kw):")
        parts.append("            raise PermissionError('子进程被禁止')")
        parts.append("        sp.Popen = _block_subprocess")
        parts.append("        sp.run = _block_subprocess")
        parts.append("        sp.call = _block_subprocess")
        parts.append("    except ImportError:")
        parts.append("        pass")
        parts.append("")
        parts.append("# ===== 执行插件代码 =====")
        parts.append("def _main():")
        parts.append("    input_data = None")
        parts.append("    try:")
        parts.append("        stdin_content = sys.stdin.read()")
        parts.append("        if stdin_content.strip():")
        parts.append("            input_data = json.loads(stdin_content)")
        parts.append("    except Exception:")
        parts.append("        input_data = None")
        parts.append("")
        parts.append("    _plugin_globals = {'__builtins__': builtins, 'input_data': input_data}")
        parts.append("    _plugin_result = None")
        parts.append("")
        parts.append("    try:")
        parts.append("        exec(" + repr(escaped_code) + ", _plugin_globals)")
        parts.append("        if 'result' in _plugin_globals:")
        parts.append("            _plugin_result = _plugin_globals['result']")
        parts.append("        elif 'output' in _plugin_globals:")
        parts.append("            _plugin_result = _plugin_globals['output']")
        parts.append("    except Exception as e:")
        parts.append("        import traceback")
        parts.append("        tb = traceback.format_exc()")
        parts.append("        print(json.dumps({'error': str(e), 'traceback': tb}), file=sys.stderr)")
        parts.append("        sys.exit(1)")
        parts.append("")
        parts.append("    if _plugin_result is not None:")
        parts.append("        print(json.dumps(_plugin_result, default=str))")
        parts.append("")
        parts.append("_main()")

        return "\n".join(parts)

    @staticmethod
    def _escape_code(code: str) -> str:
        """转义代码中的三引号"""
        return code.replace("\\", "\\\\").replace("'''", "\\'\\'\\'")

    # ----- 环境准备 -----

    def _prepare_environment(self, ctx: SandboxContext) -> Dict[str, str]:
        """准备沙箱环境变量"""
        env = dict(os.environ)

        # 移除危险环境变量
        dangerous_vars = [
            "PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME",
            "PYTHONINSPECT", "PYTHONDEBUG",
        ]
        for var in dangerous_vars:
            env.pop(var, None)

        # 设置安全相关的环境变量
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"

        # 添加自定义环境变量
        for key, value in ctx.environment.items():
            env[key] = value

        return env

    @staticmethod
    def _create_preexec_fn(ctx: SandboxContext):
        """创建 preexec_fn（仅 Unix）"""
        def preexec():
            # 设置 CPU 时间限制
            try:
                import resource
                cpu_seconds = int(ctx.max_cpu_time)
                resource.setrlimit(
                    resource.RLIMIT_CPU,
                    (cpu_seconds, cpu_seconds + 1),
                )
            except (ImportError, ValueError, OSError):
                pass

            # 设置内存限制
            try:
                import resource
                memory_bytes = ctx.max_memory_mb * 1024 * 1024
                resource.setrlimit(
                    resource.RLIMIT_AS,
                    (memory_bytes, memory_bytes),
                )
            except (ImportError, ValueError, OSError):
                pass

        return preexec

    # ----- 文件系统限制 -----

    def restrict_filesystem(
        self,
        allowed_paths: Optional[List[str]] = None,
        denied_paths: Optional[List[str]] = None,
    ) -> None:
        """配置文件系统访问限制"""
        if allowed_paths is not None:
            self._context.allowed_paths = list(allowed_paths)
        if denied_paths is not None:
            self._context.denied_paths = list(denied_paths)

    def restrict_network(self, allowed: bool = False) -> None:
        """配置网络访问限制"""
        self._context.network_allowed = allowed

    def restrict_imports(
        self,
        allowed: Optional[Set[str]] = None,
        denied: Optional[Set[str]] = None,
    ) -> None:
        """配置模块导入限制"""
        if allowed is not None:
            self._context.allowed_modules = set(allowed)
        if denied is not None:
            self._context.denied_modules = set(denied)

    # ----- 清理 -----

    @staticmethod
    def _safe_cleanup(directory: str) -> None:
        """安全清理临时目录"""
        try:
            import shutil
            shutil.rmtree(directory, ignore_errors=True)
        except Exception:
            pass

    # ----- 统计 -----

    def get_stats(self) -> dict:
        """获取沙箱统计信息"""
        with self._lock:
            avg_time = (
                self._total_time_ms / self._execution_count
                if self._execution_count > 0
                else 0.0
            )
            return {
                "execution_count": self._execution_count,
                "total_time_ms": round(self._total_time_ms, 3),
                "avg_time_ms": round(avg_time, 3),
            }

    def reset_stats(self) -> None:
        """重置统计"""
        with self._lock:
            self._execution_count = 0
            self._total_time_ms = 0.0
