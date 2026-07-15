"""
Jupyter Kernel Client Module

Provides kernel lifecycle management, code execution, output capture,
interrupt, restart, completion requests, and inspection for Jupyter kernels.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

try:
    import jupyter_client as _jupyter_client
except ImportError:
    _jupyter_client = None  # type: ignore

try:
    import zmq as _zmq
except ImportError:
    _zmq = None  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------

class KernelStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    STARTING = "starting"
    RESTARTING = "restarting"
    DEAD = "dead"
    ERROR = "error"
    UNKNOWN = "unknown"


class OutputType(Enum):
    STDOUT = "stdout"
    STDERR = "stderr"
    DISPLAY_DATA = "display_data"
    EXECUTE_RESULT = "execute_result"
    STREAM = "stream"
    ERROR = "error"
    CLEAR_OUTPUT = "clear_output"


class MessageStatus(Enum):
    PENDING = "pending"
    DELIVERED = "delivered"
    REPLIED = "replied"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class KernelSpec:
    """Specification for a Jupyter kernel."""
    name: str
    display_name: str
    language: str = "python"
    kernel_cmd: List[str] = field(default_factory=lambda: ["python3", "-m", "ipykernel_launcher", "-f", "{connection_file}"])
    env: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputMessage:
    """Represents a single output message from kernel execution."""
    output_type: OutputType
    content: str = ""
    mime_type: str = "text/plain"
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_count: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    parent_msg_id: Optional[str] = None


@dataclass
class ExecutionResult:
    """Result of a code execution in the kernel."""
    code: str
    execution_count: int = 0
    status: str = "ok"
    outputs: List[OutputMessage] = field(default_factory=list)
    error_name: Optional[str] = None
    error_value: Optional[str] = None
    error_traceback: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: float = 0.0
    user_expressions: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == "ok"

    @property
    def stdout_text(self) -> str:
        parts: List[str] = []
        for out in self.outputs:
            if out.output_type in (OutputType.STDOUT, OutputType.STREAM):
                if out.metadata.get("name") == "stderr":
                    continue
                parts.append(out.content)
        return "\n".join(parts)

    @property
    def stderr_text(self) -> str:
        parts: List[str] = []
        for out in self.outputs:
            if out.output_type == OutputType.STDERR:
                parts.append(out.content)
            elif out.output_type == OutputType.STREAM and out.metadata.get("name") == "stderr":
                parts.append(out.content)
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "execution_count": self.execution_count,
            "status": self.status,
            "success": self.success,
            "outputs": [
                {
                    "type": o.output_type.value,
                    "content": o.content,
                    "mime_type": o.mime_type,
                    "data": o.data,
                }
                for o in self.outputs
            ],
            "error": {
                "name": self.error_name,
                "value": self.error_value,
                "traceback": self.error_traceback,
            } if self.error_name else None,
            "duration": self.duration,
        }


@dataclass
class CompletionMatch:
    """A single completion match."""
    text: str
    start: int = 0
    end: int = 0
    type: str = "unknown"
    documentation: str = ""


@dataclass
class CompletionResult:
    """Result of a completion request."""
    matches: List[CompletionMatch] = field(default_factory=list)
    cursor_start: int = 0
    cursor_end: int = 0
    status: str = "ok"
    matched_text: str = ""


@dataclass
class InspectionResult:
    """Result of an object inspection request."""
    found: bool = False
    data: Dict[str, str] = field(default_factory=dict)
    documentation: str = ""
    source: Optional[str] = None
    status: str = "ok"


@dataclass
class KernelInfo:
    """Information about the running kernel."""
    kernel_id: str
    status: KernelStatus = KernelStatus.UNKNOWN
    language: str = "python"
    language_version: str = ""
    implementation: str = ""
    implementation_version: str = ""
    banner: str = ""
    protocol_version: str = "5.3"
    kernel_spec: Optional[KernelSpec] = None
    started_at: Optional[datetime] = None
    execution_count: int = 0
    pid: Optional[int] = None


# ---------------------------------------------------------------------------
# Built-in Kernel Specs
# ---------------------------------------------------------------------------

BUILTIN_KERNEL_SPECS: Dict[str, KernelSpec] = {
    "python3": KernelSpec(
        name="python3",
        display_name="Python 3",
        language="python",
        kernel_cmd=["python3", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    ),
    "python": KernelSpec(
        name="python",
        display_name="Python",
        language="python",
        kernel_cmd=["python", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    ),
    "bash": KernelSpec(
        name="bash",
        display_name="Bash",
        language="bash",
        kernel_cmd=["bash", "-c", "cat"],
    ),
}


# ---------------------------------------------------------------------------
# Simulated Kernel Environment
# ---------------------------------------------------------------------------

class _SimulatedKernel:
    """Simulated kernel for environments without jupyter_client."""

    def __init__(self, kernel_spec: KernelSpec) -> None:
        self.spec = kernel_spec
        self._variables: Dict[str, Any] = {}
        self._execution_count = 0
        self._status = KernelStatus.IDLE
        self._output_queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._import_history: List[str] = []

    @property
    def status(self) -> KernelStatus:
        return self._status

    def execute(self, code: str, timeout: Optional[float] = None) -> ExecutionResult:
        start = datetime.utcnow()
        self._status = KernelStatus.BUSY
        outputs: List[OutputMessage] = []
        error_name: Optional[str] = None
        error_value: Optional[str] = None
        error_tb: List[str] = []
        status = "ok"

        try:
            lines = code.strip().split("\n")
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                result = self._execute_line(line)
                if result.output_type == OutputType.ERROR:
                    error_name = result.data.get("ename")
                    error_value = result.data.get("evalue")
                    error_tb = result.data.get("traceback", [])
                    status = "error"
                    outputs.append(result)
                    break
                elif result.content:
                    outputs.append(result)

        except Exception as exc:
            error_name = type(exc).__name__
            error_value = str(exc)
            error_tb = [f"  {type(exc).__module__}.{type(exc).__name__}: {exc}"]
            status = "error"

        with self._lock:
            self._execution_count += 1
            exec_count = self._execution_count

        self._status = KernelStatus.IDLE
        end = datetime.utcnow()

        return ExecutionResult(
            code=code,
            execution_count=exec_count,
            status=status,
            outputs=outputs,
            error_name=error_name,
            error_value=error_value,
            error_traceback=error_tb,
            start_time=start,
            end_time=end,
            duration=(end - start).total_seconds(),
        )

    def _execute_line(self, line: str) -> OutputMessage:
        # Handle print
        print_match = re.match(r'print\s*\((.+)\)$', line)
        if print_match:
            expr = print_match.group(1).strip()
            try:
                value = self._eval_expr(expr)
                return OutputMessage(
                    output_type=OutputType.STREAM,
                    content=str(value),
                    metadata={"name": "stdout"},
                )
            except Exception as exc:
                return OutputMessage(
                    output_type=OutputType.ERROR,
                    content=str(exc),
                    data={"ename": type(exc).__name__, "evalue": str(exc), "traceback": []},
                )

        # Handle assignment
        assign_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$', line)
        if assign_match:
            var_name = assign_match.group(1)
            expr = assign_match.group(2).strip()
            try:
                value = self._eval_expr(expr)
                self._variables[var_name] = value
                return OutputMessage(output_type=OutputType.STREAM, content="")
            except Exception as exc:
                return OutputMessage(
                    output_type=OutputType.ERROR,
                    content=str(exc),
                    data={"ename": type(exc).__name__, "evalue": str(exc), "traceback": []},
                )

        # Handle import
        import_match = re.match(r'^(import\s+.+|from\s+.+\s+import\s+.+)$', line)
        if import_match:
            self._import_history.append(line)
            return OutputMessage(output_type=OutputType.STREAM, content="")

        # Handle expression (evaluate and display)
        try:
            value = self._eval_expr(line)
            if value is not None:
                return OutputMessage(
                    output_type=OutputType.EXECUTE_RESULT,
                    content=repr(value),
                    data={"text/plain": repr(value)},
                )
            return OutputMessage(output_type=OutputType.STREAM, content="")
        except Exception as exc:
            return OutputMessage(
                output_type=OutputType.ERROR,
                content=str(exc),
                data={"ename": type(exc).__name__, "evalue": str(exc), "traceback": []},
            )

    def _eval_expr(self, expr: str) -> Any:
        # Safe evaluation of simple expressions
        expr = expr.strip()

        # String literals
        if (expr.startswith('"') and expr.endswith('"')) or \
           (expr.startswith("'") and expr.endswith("'")):
            return expr[1:-1]

        # Numbers
        try:
            if '.' in expr:
                return float(expr)
            return int(expr)
        except ValueError:
            pass

        # Booleans and None
        if expr == "True":
            return True
        if expr == "False":
            return False
        if expr == "None":
            return None

        # List literal
        if expr.startswith("[") and expr.endswith("]"):
            inner = expr[1:-1].strip()
            if not inner:
                return []
            items = self._split_args(inner)
            return [self._eval_expr(item.strip()) for item in items]

        # Dict literal
        if expr.startswith("{") and expr.endswith("}"):
            inner = expr[1:-1].strip()
            if not inner:
                return {}
            result: Dict[str, Any] = {}
            pairs = self._split_args(inner)
            for pair in pairs:
                if ":" in pair:
                    k, v = pair.split(":", 1)
                    result[str(self._eval_expr(k.strip()))] = self._eval_expr(v.strip())
            return result

        # Variable lookup
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', expr):
            if expr in self._variables:
                return self._variables[expr]
            raise NameError(f"name '{expr}' is not defined")

        # Simple binary operations
        ops = [" + ", " - ", " * ", " / ", " // ", " % ", " ** "]
        for op in ops:
            if op in expr:
                parts = expr.split(op, 1)
                left = self._eval_expr(parts[0].strip())
                right = self._eval_expr(parts[1].strip())
                op_stripped = op.strip()
                if op_stripped == "+":
                    return left + right
                elif op_stripped == "-":
                    return left - right
                elif op_stripped == "*":
                    return left * right
                elif op_stripped == "/":
                    return left / right
                elif op_stripped == "//":
                    return left // right
                elif op_stripped == "%":
                    return left % right
                elif op_stripped == "**":
                    return left ** right

        # String operations
        if ".upper()" in expr:
            base = expr.replace(".upper()", "")
            return str(self._eval_expr(base)).upper()
        if ".lower()" in expr:
            base = expr.replace(".lower()", "")
            return str(self._eval_expr(base)).lower()
        if ".strip()" in expr:
            base = expr.replace(".strip()", "")
            return str(self._eval_expr(base)).strip()
        if ".split(" in expr:
            match = re.match(r'(.+?)\.split\((.+?)\)', expr)
            if match:
                base = self._eval_expr(match.group(1).strip())
                sep = self._eval_expr(match.group(2).strip())
                return str(base).split(str(sep))
        if ".join(" in expr:
            match = re.match(r'(.+?)\.join\((.+?)\)', expr)
            if match:
                sep = self._eval_expr(match.group(1).strip())
                iterable = self._eval_expr(match.group(2).strip())
                return str(sep).join(str(x) for x in iterable)
        if ".format(" in expr:
            base = expr[:expr.index(".format(")]
            args_str = expr[expr.index(".format(") + 8:-1]
            args = [self._eval_expr(a.strip()) for a in self._split_args(args_str)]
            return str(self._eval_expr(base)).format(*args)

        if "len(" in expr:
            match = re.match(r'len\((.+)\)', expr)
            if match:
                return len(self._eval_expr(match.group(1).strip()))

        if "type(" in expr:
            match = re.match(r'type\((.+)\)', expr)
            if match:
                val = self._eval_expr(match.group(1).strip())
                return type(val).__name__

        if "range(" in expr:
            match = re.match(r'range\((.+)\)', expr)
            if match:
                args = [self._eval_expr(a.strip()) for a in self._split_args(match.group(1))]
                return list(range(*[int(a) for a in args]))

        # Numeric literal
        try:
            return float(expr)
        except (ValueError, TypeError):
            pass

        # String literal
        if (expr.startswith('"') and expr.endswith('"')) or \
           (expr.startswith("'") and expr.endswith("'")):
            return expr[1:-1]

        # Boolean / None literals
        if expr == "True":
            return True
        if expr == "False":
            return False
        if expr == "None":
            return None

        # List literal
        if expr.startswith("[") and expr.endswith("]"):
            inner = expr[1:-1].strip()
            if not inner:
                return []
            items = self._split_args(inner)
            return [self._eval_expr(item.strip()) for item in items]

        # Dict literal
        if expr.startswith("{") and expr.endswith("}") and ":" in expr:
            inner = expr[1:-1].strip()
            if not inner:
                return {}
            pairs = self._split_args(inner)
            result = {}
            for pair in pairs:
                if ":" in pair:
                    key_str, val_str = pair.split(":", 1)
                    key = self._eval_expr(key_str.strip())
                    val = self._eval_expr(val_str.strip())
                    result[key] = val
            return result

        # Tuple literal
        if expr.startswith("(") and expr.endswith(")"):
            inner = expr[1:-1].strip()
            if not inner:
                return ()
            items = self._split_args(inner)
            return tuple(self._eval_expr(item.strip()) for item in items)

        # Variable lookup from local namespace
        if hasattr(self, '_variables') and isinstance(expr, str):
            var_name = expr.strip()
            if var_name in self._variables:
                return self._variables[var_name]

        # Attribute access (e.g., obj.attr)
        if "." in expr and not any(op in expr for op in ["(", ")", "[", "]"]):
            parts = expr.split(".", 1)
            base = self._eval_expr(parts[0].strip())
            if hasattr(base, parts[1].strip()):
                return getattr(base, parts[1].strip())

        # Index access (e.g., list[0])
        if "[" in expr and expr.endswith("]"):
            match = re.match(r'(.+?)\[(.+)\]', expr)
            if match:
                base = self._eval_expr(match.group(1).strip())
                index = self._eval_expr(match.group(2).strip())
                try:
                    return base[index]
                except (TypeError, KeyError, IndexError):
                    pass

        # max / min builtins
        if expr.startswith("max(") and expr.endswith(")"):
            args = self._split_args(expr[4:-1])
            values = [self._eval_expr(a.strip()) for a in args]
            return max(values)
        if expr.startswith("min(") and expr.endswith(")"):
            args = self._split_args(expr[4:-1])
            values = [self._eval_expr(a.strip()) for a in args]
            return min(values)

        # sum / abs builtins
        if expr.startswith("sum(") and expr.endswith(")"):
            args = self._split_args(expr[4:-1])
            values = [self._eval_expr(a.strip()) for a in args]
            return sum(values)
        if expr.startswith("abs(") and expr.endswith(")"):
            inner = self._eval_expr(expr[4:-1].strip())
            return abs(inner)

        # round builtin
        if expr.startswith("round(") and expr.endswith(")"):
            args = self._split_args(expr[6:-1])
            values = [self._eval_expr(a.strip()) for a in args]
            if len(values) == 1:
                return round(values[0])
            elif len(values) >= 2:
                return round(values[0], int(values[1]))

        # sorted builtin
        if expr.startswith("sorted(") and expr.endswith(")"):
            args = self._split_args(expr[7:-1])
            values = [self._eval_expr(a.strip()) for a in args]
            return sorted(values)

        # int / float / str conversions
        if expr.startswith("int(") and expr.endswith(")"):
            return int(self._eval_expr(expr[4:-1].strip()))
        if expr.startswith("float(") and expr.endswith(")"):
            return float(self._eval_expr(expr[6:-1].strip()))
        if expr.startswith("str(") and expr.endswith(")"):
            return str(self._eval_expr(expr[5:-1].strip()))

        # enumerate / zip
        if expr.startswith("enumerate(") and expr.endswith(")"):
            args = self._split_args(expr[9:-1])
            iterable = self._eval_expr(args[0].strip())
            return list(enumerate(iterable))
        if expr.startswith("zip(") and expr.endswith(")"):
            args = self._split_args(expr[4:-1])
            iterables = [self._eval_expr(a.strip()) for a in args]
            return list(zip(*iterables))

        # list / tuple / set constructors
        if expr.startswith("list(") and expr.endswith(")"):
            return list(self._eval_expr(expr[5:-1].strip()))
        if expr.startswith("tuple(") and expr.endswith(")"):
            return tuple(self._eval_expr(expr[6:-1].strip()))
        if expr.startswith("set(") and expr.endswith(")"):
            return set(self._eval_expr(expr[4:-1].strip()))

        # isinstance check
        if expr.startswith("isinstance(") and expr.endswith(")"):
            args = self._split_args(expr[11:-1])
            obj = self._eval_expr(args[0].strip())
            type_name = self._eval_expr(args[1].strip())
            type_map = {"int": int, "float": float, "str": str, "list": list, "dict": dict, "tuple": tuple, "bool": bool}
            return isinstance(obj, type_map.get(type_name, object))

        # hasattr / getattr
        if expr.startswith("hasattr(") and expr.endswith(")"):
            args = self._split_args(expr[8:-1])
            obj = self._eval_expr(args[0].strip())
            attr = self._eval_expr(args[1].strip())
            return hasattr(obj, attr)
        if expr.startswith("getattr(") and expr.endswith(")"):
            args = self._split_args(expr[8:-1])
            obj = self._eval_expr(args[0].strip())
            attr = self._eval_expr(args[1].strip())
            default = self._eval_expr(args[2].strip()) if len(args) > 2 else None
            return getattr(obj, attr, default)

        # math.sqrt, math.log, math.exp, math.sin, math.cos
        if expr.startswith("math."):
            import math as _math
            func_map = {
                "sqrt": lambda v: _math.sqrt(v),
                "log": lambda v: _math.log(v),
                "exp": lambda v: _math.exp(v),
                "sin": lambda v: _math.sin(v),
                "cos": lambda v: _math.cos(v),
                "tan": lambda v: _math.tan(v),
                "pi": _math.pi,
                "e": _math.e,
                "floor": lambda v: _math.floor(v),
                "ceil": lambda v: _math.ceil(v),
            }
            for name, fn in func_map.items():
                if expr == f"math.{name}":
                    return fn if not callable(fn) else None
                if expr.startswith(f"math.{name}(") and expr.endswith(")"):
                    arg = self._eval_expr(expr[len(f"math.{name}("):-1].strip())
                    return fn(arg)

        # Fallback: return expression as string representation
        logger.warning(f"Cannot evaluate expression '{expr}', returning as string")
        return expr

    @staticmethod
    def _split_args(s: str) -> List[str]:
        """Split comma-separated arguments respecting brackets and quotes."""
        args: List[str] = []
        depth = 0
        current: List[str] = []
        in_string: Optional[str] = None
        for ch in s:
            if in_string:
                current.append(ch)
                if ch == in_string:
                    in_string = None
                continue
            if ch in ('"', "'"):
                in_string = ch
                current.append(ch)
            elif ch in ('(', '[', '{'):
                depth += 1
                current.append(ch)
            elif ch in (')', ']', '}'):
                depth -= 1
                current.append(ch)
            elif ch == ',' and depth == 0:
                args.append("".join(current))
                current = []
            else:
                current.append(ch)
        if current:
            args.append("".join(current))
        return args

    def complete(self, code: str, cursor_pos: int) -> CompletionResult:
        text = code[:cursor_pos]
        matches: List[CompletionMatch] = []

        # Complete variable names
        if re.search(r'[a-zA-Z_][a-zA-Z0-9_]*$', text):
            prefix = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*$', text)[0]
            start = cursor_pos - len(prefix)
            for var in self._variables:
                if var.startswith(prefix) and var != prefix:
                    matches.append(CompletionMatch(
                        text=var, start=start, end=cursor_pos,
                        type="variable",
                    ))

        # Complete built-in names
        builtins = [
            "print", "len", "range", "type", "int", "float", "str", "list",
            "dict", "set", "tuple", "bool", "abs", "max", "min", "sum",
            "sorted", "enumerate", "zip", "map", "filter", "isinstance",
            "hasattr", "getattr", "setattr", "open", "input", "help",
        ]
        if re.search(r'[a-zA-Z_][a-zA-Z0-9_]*$', text):
            prefix = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*$', text)[0]
            start = cursor_pos - len(prefix)
            for b in builtins:
                if b.startswith(prefix) and b != prefix:
                    matches.append(CompletionMatch(
                        text=b, start=start, end=cursor_pos,
                        type="function",
                    ))

        # Complete attribute access
        dot_match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z0-9_]*)$', text)
        if dot_match:
            obj_name = dot_match.group(1)
            attr_prefix = dot_match.group(2)
            start = cursor_pos - len(attr_prefix)
            obj = self._variables.get(obj_name)
            if obj is not None:
                for attr in dir(obj):
                    if attr.startswith(attr_prefix) and not attr.startswith("_"):
                        matches.append(CompletionMatch(
                            text=attr, start=start, end=cursor_pos,
                            type="attribute",
                        ))

        return CompletionResult(
            matches=matches,
            cursor_start=matches[0].start if matches else cursor_pos,
            cursor_end=matches[0].end if matches else cursor_pos,
            matched_text=text,
        )

    def inspect(self, code: str, cursor_pos: int, detail_level: int = 0) -> InspectionResult:
        text = code[:cursor_pos].strip()
        name_match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)$', text)

        if not name_match:
            return InspectionResult(found=False)

        name = name_match.group(1)

        if name in self._variables:
            obj = self._variables[name]
            doc = f"{type(obj).__name__}: {repr(obj)}"
            return InspectionResult(
                found=True,
                data={
                    "text/plain": repr(obj),
                    "text/html": f"<pre>{repr(obj)}</pre>",
                },
                documentation=doc,
                source=None,
            )

        builtins_doc = {
            "print": "print(*objects, sep=' ', end='\\n', file=sys.stdout, flush=False)",
            "len": "len(s) -> int\nReturn the number of items in a container.",
            "range": "range(stop) -> range object",
            "type": "type(object) -> the object's type",
            "int": "int(x=0) -> integer",
            "float": "float(x=0) -> floating point number",
            "str": "str(object='') -> str",
            "list": "list() -> new empty list",
            "dict": "dict() -> new empty dictionary",
        }

        if name in builtins_doc:
            return InspectionResult(
                found=True,
                data={"text/plain": builtins_doc[name]},
                documentation=builtins_doc[name],
            )

        return InspectionResult(found=False)

    def interrupt(self) -> bool:
        if self._status == KernelStatus.BUSY:
            self._status = KernelStatus.IDLE
            return True
        return False

    def restart(self) -> bool:
        self._variables.clear()
        self._execution_count = 0
        self._import_history.clear()
        self._status = KernelStatus.IDLE
        return True

    def shutdown(self) -> None:
        self._status = KernelStatus.DEAD


# ---------------------------------------------------------------------------
# Kernel Manager
# ---------------------------------------------------------------------------

class KernelManager:
    """Manages Jupyter kernel lifecycle (start, stop, restart, interrupt)."""

    def __init__(self) -> None:
        self._kernels: Dict[str, _SimulatedKernel] = {}
        self._kernel_specs: Dict[str, KernelSpec] = dict(BUILTIN_KERNEL_SPECS)
        self._lock = threading.Lock()

    def list_kernel_specs(self) -> List[KernelSpec]:
        return list(self._kernel_specs.values())

    def get_kernel_spec(self, name: str) -> Optional[KernelSpec]:
        return self._kernel_specs.get(name)

    def register_kernel_spec(self, spec: KernelSpec) -> None:
        self._kernel_specs[spec.name] = spec

    def start_kernel(
        self,
        kernel_name: str = "python3",
        kernel_id: Optional[str] = None,
    ) -> str:
        spec = self._kernel_specs.get(kernel_name)
        if spec is None:
            raise ValueError(f"Unknown kernel spec: {kernel_name}")

        kid = kernel_id or str(uuid.uuid4())[:8]
        with self._lock:
            self._kernels[kid] = _SimulatedKernel(spec)
        logger.info("Started kernel '%s' with id=%s", kernel_name, kid)
        return kid

    def stop_kernel(self, kernel_id: str) -> bool:
        with self._lock:
            kernel = self._kernels.pop(kernel_id, None)
        if kernel:
            kernel.shutdown()
            logger.info("Stopped kernel id=%s", kernel_id)
            return True
        return False

    def restart_kernel(self, kernel_id: str) -> bool:
        with self._lock:
            kernel = self._kernels.get(kernel_id)
        if kernel:
            kernel.restart()
            logger.info("Restarted kernel id=%s", kernel_id)
            return True
        return False

    def interrupt_kernel(self, kernel_id: str) -> bool:
        with self._lock:
            kernel = self._kernels.get(kernel_id)
        if kernel:
            return kernel.interrupt()
        return False

    def get_kernel(self, kernel_id: str) -> Optional[_SimulatedKernel]:
        return self._kernels.get(kernel_id)

    def list_kernels(self) -> List[str]:
        return list(self._kernels.keys())

    def get_kernel_status(self, kernel_id: str) -> KernelStatus:
        kernel = self._kernels.get(kernel_id)
        if kernel is None:
            return KernelStatus.DEAD
        return kernel.status


# ---------------------------------------------------------------------------
# Output Capture
# ---------------------------------------------------------------------------

class OutputCapture:
    """Captures and buffers output from kernel execution."""

    def __init__(self, max_buffer_size: int = 10000) -> None:
        self.max_buffer_size = max_buffer_size
        self._buffer: List[OutputMessage] = []
        self._handlers: List[Callable[[OutputMessage], None]] = []
        self._lock = threading.Lock()
        self._filter_types: Optional[set] = None

    def add_message(self, message: OutputMessage) -> None:
        if self._filter_types and message.output_type not in self._filter_types:
            return
        with self._lock:
            self._buffer.append(message)
            if len(self._buffer) > self.max_buffer_size:
                self._buffer = self._buffer[-self.max_buffer_size:]
        for handler in self._handlers:
            try:
                handler(message)
            except Exception as exc:
                logger.warning("Output handler error: %s", exc)

    def add_handler(self, handler: Callable[[OutputMessage], None]) -> None:
        self._handlers.append(handler)

    def remove_handler(self, handler: Callable[[OutputMessage], None]) -> None:
        self._handlers = [h for h in self._handlers if h != handler]

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    def get_messages(
        self,
        output_type: Optional[OutputType] = None,
        limit: int = 100,
    ) -> List[OutputMessage]:
        with self._lock:
            messages = list(self._buffer)
        if output_type:
            messages = [m for m in messages if m.output_type == output_type]
        return messages[-limit:]

    def get_text_output(self) -> str:
        parts: List[str] = []
        with self._lock:
            for msg in self._buffer:
                if msg.output_type in (OutputType.STREAM, OutputType.STDOUT):
                    parts.append(msg.content)
        return "\n".join(parts)

    def get_error_output(self) -> str:
        parts: List[str] = []
        with self._lock:
            for msg in self._buffer:
                if msg.output_type in (OutputType.ERROR, OutputType.STDERR):
                    parts.append(msg.content)
        return "\n".join(parts)

    def set_filter(self, types: Optional[set] = None) -> None:
        self._filter_types = types

    @property
    def message_count(self) -> int:
        with self._lock:
            return len(self._buffer)


# ---------------------------------------------------------------------------
# Code Executor
# ---------------------------------------------------------------------------

class CodeExecutor:
    """Executes code on a kernel with output capture and error handling."""

    def __init__(
        self,
        kernel_manager: KernelManager,
        output_capture: Optional[OutputCapture] = None,
    ) -> None:
        self.kernel_manager = kernel_manager
        self.output_capture = output_capture or OutputCapture()
        self._execution_history: List[ExecutionResult] = []
        self._lock = threading.Lock()
        self._timeout: float = 30.0
        self._max_output_size: int = 1024 * 1024  # 1MB

    def execute(
        self,
        kernel_id: str,
        code: str,
        timeout: Optional[float] = None,
        silent: bool = False,
        store_history: bool = True,
        user_expressions: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        kernel = self.kernel_manager.get_kernel(kernel_id)
        if kernel is None:
            return ExecutionResult(
                code=code, status="error",
                error_name="KernelNotFound",
                error_value=f"Kernel '{kernel_id}' not found",
            )

        effective_timeout = timeout or self._timeout
        result = kernel.execute(code, timeout=effective_timeout)

        # Capture outputs
        if not silent:
            for output in result.outputs:
                self.output_capture.add_message(output)

        # Store history
        if store_history:
            with self._lock:
                self._execution_history.append(result)

        return result

    def execute_async(
        self,
        kernel_id: str,
        code: str,
        callback: Optional[Callable[[ExecutionResult], None]] = None,
        timeout: Optional[float] = None,
    ) -> threading.Thread:
        def _run() -> None:
            result = self.execute(kernel_id, code, timeout=timeout)
            if callback:
                callback(result)

        thread = threading.Thread(
            target=_run, daemon=True, name=f"kernel-exec-{kernel_id}"
        )
        thread.start()
        return thread

    def get_history(self, limit: int = 50) -> List[ExecutionResult]:
        with self._lock:
            return list(self._execution_history[-limit:])

    def clear_history(self) -> None:
        with self._lock:
            self._execution_history.clear()


# ---------------------------------------------------------------------------
# Completion Request
# ---------------------------------------------------------------------------

class CompletionRequest:
    """Handles code completion requests to a kernel."""

    def __init__(self, kernel_manager: KernelManager) -> None:
        self.kernel_manager = kernel_manager
        self._cache: Dict[str, CompletionResult] = {}
        self._cache_max = 1000
        self._lock = threading.Lock()

    def complete(
        self,
        kernel_id: str,
        code: str,
        cursor_pos: int,
        timeout: float = 2.0,
    ) -> CompletionResult:
        cache_key = f"{kernel_id}:{code}:{cursor_pos}"
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached:
                return cached

        kernel = self.kernel_manager.get_kernel(kernel_id)
        if kernel is None:
            return CompletionResult(status="error")

        result = kernel.complete(code, cursor_pos)

        with self._lock:
            if len(self._cache) >= self._cache_max:
                keys = list(self._cache.keys())
                for k in keys[:len(keys) // 2]:
                    del self._cache[k]
            self._cache[cache_key] = result

        return result

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()


# ---------------------------------------------------------------------------
# Inspection Request
# ---------------------------------------------------------------------------

class InspectionRequest:
    """Handles object inspection requests to a kernel."""

    def __init__(self, kernel_manager: KernelManager) -> None:
        self.kernel_manager = kernel_manager
        self._cache: Dict[str, InspectionResult] = {}
        self._cache_max = 500

    def inspect(
        self,
        kernel_id: str,
        code: str,
        cursor_pos: int,
        detail_level: int = 0,
        timeout: float = 2.0,
    ) -> InspectionResult:
        cache_key = f"{kernel_id}:{code}:{cursor_pos}:{detail_level}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        kernel = self.kernel_manager.get_kernel(kernel_id)
        if kernel is None:
            return InspectionResult(found=False, status="error")

        result = kernel.inspect(code, cursor_pos, detail_level)

        if len(self._cache) >= self._cache_max:
            keys = list(self._cache.keys())
            for k in keys[:len(keys) // 2]:
                del self._cache[k]
        self._cache[cache_key] = result

        return result

    def clear_cache(self) -> None:
        self._cache.clear()


# ---------------------------------------------------------------------------
# Jupyter Client (Main Facade)
# ---------------------------------------------------------------------------

class JupyterClient:
    """Main facade for Jupyter kernel operations."""

    def __init__(
        self,
        kernel_name: str = "python3",
        timeout: float = 30.0,
        max_history: int = 1000,
    ) -> None:
        self.kernel_manager = KernelManager()
        self.output_capture = OutputCapture()
        self.code_executor = CodeExecutor(self.kernel_manager, self.output_capture)
        self.completion = CompletionRequest(self.kernel_manager)
        self.inspection = InspectionRequest(self.kernel_manager)
        self._kernel_name = kernel_name
        self._timeout = timeout
        self._kernel_id: Optional[str] = None
        self._max_history = max_history

    def start_kernel(self, kernel_name: Optional[str] = None) -> str:
        name = kernel_name or self._kernel_name
        self._kernel_id = self.kernel_manager.start_kernel(name)
        return self._kernel_id

    def stop_kernel(self) -> bool:
        if self._kernel_id is None:
            return False
        result = self.kernel_manager.stop_kernel(self._kernel_id)
        self._kernel_id = None
        return result

    def restart_kernel(self) -> bool:
        if self._kernel_id is None:
            return False
        return self.kernel_manager.restart_kernel(self._kernel_id)

    def interrupt_kernel(self) -> bool:
        if self._kernel_id is None:
            return False
        return self.kernel_manager.interrupt_kernel(self._kernel_id)

    def execute(
        self,
        code: str,
        kernel_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> ExecutionResult:
        kid = kernel_id or self._kernel_id
        if kid is None:
            raise RuntimeError("No kernel running. Call start_kernel() first.")
        return self.code_executor.execute(kid, code, timeout=timeout)

    def execute_async(
        self,
        code: str,
        callback: Optional[Callable[[ExecutionResult], None]] = None,
        kernel_id: Optional[str] = None,
    ) -> threading.Thread:
        kid = kernel_id or self._kernel_id
        if kid is None:
            raise RuntimeError("No kernel running. Call start_kernel() first.")
        return self.code_executor.execute_async(kid, code, callback=callback)

    def complete(
        self,
        code: str,
        cursor_pos: int,
        kernel_id: Optional[str] = None,
    ) -> CompletionResult:
        kid = kernel_id or self._kernel_id
        if kid is None:
            return CompletionResult(status="error")
        return self.completion.complete(kid, code, cursor_pos)

    def inspect(
        self,
        code: str,
        cursor_pos: int,
        kernel_id: Optional[str] = None,
    ) -> InspectionResult:
        kid = kernel_id or self._kernel_id
        if kid is None:
            return InspectionResult(found=False, status="error")
        return self.inspection.inspect(kid, code, cursor_pos)

    def get_kernel_info(self, kernel_id: Optional[str] = None) -> KernelInfo:
        kid = kernel_id or self._kernel_id
        if kid is None:
            return KernelInfo(kernel_id="none", status=KernelStatus.DEAD)
        kernel = self.kernel_manager.get_kernel(kid)
        if kernel is None:
            return KernelInfo(kernel_id=kid, status=KernelStatus.DEAD)
        return KernelInfo(
            kernel_id=kid,
            status=kernel.status,
            language=kernel.spec.language,
            execution_count=kernel._execution_count,
            kernel_spec=kernel.spec,
        )

    def get_outputs(self, limit: int = 100) -> List[OutputMessage]:
        return self.output_capture.get_messages(limit=limit)

    def get_execution_history(self, limit: int = 50) -> List[ExecutionResult]:
        return self.code_executor.get_history(limit=limit)

    @property
    def kernel_id(self) -> Optional[str]:
        return self._kernel_id

    @property
    def is_alive(self) -> bool:
        if self._kernel_id is None:
            return False
        status = self.kernel_manager.get_kernel_status(self._kernel_id)
        return status not in (KernelStatus.DEAD, KernelStatus.ERROR)
