"""
Deep Parameter Extraction Module

AST-based extraction from tool calls, nested parameter resolution,
template variable expansion, type inference, and sanitization marking.
"""

from __future__ import annotations

import ast
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class ParamType(Enum):
    """Inferred parameter types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    NONE = "none"
    PATH = "path"
    URL = "url"
    COMMAND = "command"
    EXPRESSION = "expression"
    UNKNOWN = "unknown"


class SanitizationLevel(Enum):
    """Sanitization levels for parameters."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ExtractedParam:
    """A single extracted parameter with metadata."""
    name: str
    raw_value: Any
    resolved_value: Any
    param_type: ParamType
    sanitization_level: SanitizationLevel
    source: str = "direct"
    nested_path: str = ""
    is_template: bool = False
    template_variables: List[str] = field(default_factory=list)
    sanitization_flags: List[str] = field(default_factory=list)
    depth: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "raw_value": str(self.raw_value)[:200],
            "resolved_value": str(self.resolved_value)[:200],
            "param_type": self.param_type.value,
            "sanitization_level": self.sanitization_level.value,
            "source": self.source,
            "nested_path": self.nested_path,
            "is_template": self.is_template,
            "template_variables": self.template_variables,
            "sanitization_flags": self.sanitization_flags,
            "depth": self.depth,
        }


@dataclass
class ExtractedParams:
    """Collection of extracted parameters from a tool call."""
    call_id: str
    tool_name: str
    parameters: List[ExtractedParam] = field(default_factory=list)
    extraction_time_ms: float = 0.0
    total_params: int = 0
    max_depth: int = 0
    has_templates: bool = False
    has_nested: bool = False
    sanitization_required: bool = False
    raw_input: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "parameters": [p.to_dict() for p in self.parameters],
            "extraction_time_ms": self.extraction_time_ms,
            "total_params": self.total_params,
            "max_depth": self.max_depth,
            "has_templates": self.has_templates,
            "has_nested": self.has_nested,
            "sanitization_required": self.sanitization_required,
        }

    def get_param(self, name: str) -> Optional[ExtractedParam]:
        for p in self.parameters:
            if p.name == name:
                return p
        return None

    def get_params_by_type(self, param_type: ParamType) -> List[ExtractedParam]:
        return [p for p in self.parameters if p.param_type == param_type]

    def get_params_requiring_sanitization(
        self, min_level: SanitizationLevel = SanitizationLevel.LOW
    ) -> List[ExtractedParam]:
        level_order = [
            SanitizationLevel.NONE, SanitizationLevel.LOW,
            SanitizationLevel.MEDIUM, SanitizationLevel.HIGH,
            SanitizationLevel.CRITICAL,
        ]
        min_idx = level_order.index(min_level)
        return [
            p for p in self.parameters
            if level_order.index(p.sanitization_level) >= min_idx
        ]


class ASTParser:
    """Parses tool call expressions using Python's AST module."""

    def __init__(self) -> None:
        self._supported_call_patterns: List[str] = [
            "function_call", "method_call", "keyword_args",
            "positional_args", "nested_calls",
        ]

    def parse_call_string(self, call_string: str) -> Optional[Dict[str, Any]]:
        call_string = call_string.strip()
        if not call_string.endswith(")"):
            return None
        try:
            tree = ast.parse(call_string, mode="eval")
        except SyntaxError:
            try:
                modified = call_string
                if not modified.endswith(")"):
                    modified += ")"
                tree = ast.parse(modified, mode="eval")
            except SyntaxError:
                return None
        if not isinstance(tree.body, ast.Call):
            return None
        return self._extract_call_info(tree.body)

    def parse_json_args(self, json_string: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(json_string)
        except (json.JSONDecodeError, TypeError):
            return None

    def parse_dict_string(self, dict_string: str) -> Optional[Dict[str, Any]]:
        try:
            tree = ast.parse(dict_string.strip(), mode="eval")
            if isinstance(tree.body, ast.Dict):
                return self._eval_dict_node(tree.body)
        except SyntaxError:
            pass
        return None

    def _extract_call_info(self, call_node: ast.Call) -> Dict[str, Any]:
        func_name = self._get_func_name(call_node)
        positional_args = []
        for arg in call_node.args:
            positional_args.append(self._ast_to_value(arg))
        keyword_args = {}
        for kw in call_node.keywords:
            keyword_args[kw.arg or ""] = self._ast_to_value(kw.value)
        return {
            "function": func_name,
            "positional_args": positional_args,
            "keyword_args": keyword_args,
        }

    def _get_func_name(self, node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        return "<unknown>"

    def _ast_to_value(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.NameConstant):
            return node.value
        elif isinstance(node, ast.List):
            return [self._ast_to_value(elt) for elt in node.elts]
        elif isinstance(node, ast.Tuple):
            return tuple(self._ast_to_value(elt) for elt in node.elts)
        elif isinstance(node, ast.Dict):
            return self._eval_dict_node(node)
        elif isinstance(node, ast.Set):
            return {self._ast_to_value(elt) for elt in node.elts}
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            operand = self._ast_to_value(node.operand)
            if isinstance(operand, (int, float)):
                return -operand
        elif isinstance(node, ast.BinOp):
            left = self._ast_to_value(node.left)
            right = self._ast_to_value(node.right)
            if isinstance(node.op, ast.Add) and isinstance(left, str) and isinstance(right, str):
                return left + right
        elif isinstance(node, ast.Call):
            return self._extract_call_info(node)
        return repr(node)

    def _eval_dict_node(self, node: ast.Dict) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key_node, value_node in zip(node.keys, node.values):
            key = self._ast_to_value(key_node) if key_node else ""
            result[str(key)] = self._ast_to_value(value_node)
        return result


class NestedResolver:
    """Resolves nested parameter structures and references."""

    def __init__(self) -> None:
        self._reference_pattern = re.compile(r'\$\{([^}]+)\}')
        self._dot_path_pattern = re.compile(r'^([a-zA-Z_]\w*)(?:\.([a-zA-Z_]\w*))*$')
        self._index_pattern = re.compile(r'^([a-zA-Z_]\w*)\[(\d+)\]$')
        self._context: Dict[str, Any] = {}

    def set_context(self, context: Dict[str, Any]) -> None:
        self._context = context

    def update_context(self, key: str, value: Any) -> None:
        self._context[key] = value

    def resolve(self, value: Any, depth: int = 0, max_depth: int = 10) -> Tuple[Any, int]:
        if depth > max_depth:
            return value, depth
        if isinstance(value, str):
            return self._resolve_string(value, depth, max_depth)
        elif isinstance(value, dict):
            return self._resolve_dict(value, depth, max_depth)
        elif isinstance(value, list):
            return self._resolve_list(value, depth, max_depth)
        return value, depth

    def _resolve_string(
        self, value: str, depth: int, max_depth: int
    ) -> Tuple[Any, int]:
        references = self._reference_pattern.findall(value)
        if not references:
            return value, depth
        if len(references) == 1 and value == f"${{{references[0]}}}":
            resolved, new_depth = self._resolve_reference(references[0], depth, max_depth)
            return resolved, new_depth
        result = value
        for ref in references:
            resolved, _ = self._resolve_reference(ref, depth, max_depth)
            result = result.replace(f"${{{ref}}}", str(resolved))
        return result, depth + 1

    def _resolve_reference(
        self, reference: str, depth: int, max_depth: int
    ) -> Tuple[Any, int]:
        dot_match = self._dot_path_pattern.match(reference)
        if dot_match:
            parts = reference.split(".")
            return self._navigate_path(parts, depth, max_depth)
        index_match = self._index_pattern.match(reference)
        if index_match:
            base_name = index_match.group(1)
            index = int(index_match.group(2))
            base_value = self._context.get(base_name)
            if isinstance(base_value, (list, tuple)) and 0 <= index < len(base_value):
                return base_value[index], depth + 1
        return f"${{{reference}}}", depth

    def _navigate_path(
        self, parts: List[str], depth: int, max_depth: int
    ) -> Tuple[Any, int]:
        current: Any = self._context
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, (list, tuple)):
                try:
                    current = current[int(part)]
                except (ValueError, IndexError):
                    return f"${{'.'.join(parts)}}", depth
            else:
                return f"${{'.'.join(parts)}}", depth
            if current is None:
                return None, depth + 1
        return current, depth + 1

    def _resolve_dict(
        self, value: Dict[str, Any], depth: int, max_depth: int
    ) -> Tuple[Dict[str, Any], int]:
        result: Dict[str, Any] = {}
        max_resolved_depth = depth
        for k, v in value.items():
            resolved, new_depth = self.resolve(v, depth + 1, max_depth)
            result[k] = resolved
            max_resolved_depth = max(max_resolved_depth, new_depth)
        return result, max_resolved_depth

    def _resolve_list(
        self, value: List[Any], depth: int, max_depth: int
    ) -> Tuple[List[Any], int]:
        result: List[Any] = []
        max_resolved_depth = depth
        for item in value:
            resolved, new_depth = self.resolve(item, depth + 1, max_depth)
            result.append(resolved)
            max_resolved_depth = max(max_resolved_depth, new_depth)
        return result, max_resolved_depth

    def get_max_nesting_depth(self, value: Any) -> int:
        if isinstance(value, dict):
            if not value:
                return 1
            return 1 + max(self.get_max_nesting_depth(v) for v in value.values())
        elif isinstance(value, (list, tuple)):
            if not value:
                return 1
            return 1 + max(self.get_max_nesting_depth(item) for item in value)
        return 0


class TemplateExpander:
    """Expands template variables in parameter values."""

    def __init__(self) -> None:
        self._variable_pattern = re.compile(r'\$\{([^}]+)\}')
        self._env_pattern = re.compile(r'\$([A-Z_][A-Z0-9_]*)')
        self._format_pattern = re.compile(r'\{([^}]+)\}')
        self._variables: Dict[str, Any] = {}
        self._functions: Dict[str, Any] = {
            "upper": lambda x: str(x).upper(),
            "lower": lambda x: str(x).lower(),
            "strip": lambda x: str(x).strip(),
            "length": lambda x: len(str(x)),
            "default": lambda x, d="": x if x else d,
            "replace": lambda x, old, new: str(x).replace(old, new),
            "split": lambda x, sep=",": str(x).split(sep),
            "join": lambda x, sep=",": sep.join(str(i) for i in x) if isinstance(x, (list, tuple)) else str(x),
        }

    def set_variable(self, name: str, value: Any) -> None:
        self._variables[name] = value

    def set_variables(self, variables: Dict[str, Any]) -> None:
        self._variables.update(variables)

    def register_function(self, name: str, func: Any) -> None:
        self._functions[name] = func

    def expand(self, value: Any) -> Tuple[Any, List[str]]:
        found_vars: List[str] = []
        if isinstance(value, str):
            expanded, found_vars = self._expand_string(value)
            return expanded, found_vars
        elif isinstance(value, dict):
            return self._expand_dict(value)
        elif isinstance(value, list):
            return self._expand_list(value)
        return value, found_vars

    def _expand_string(self, value: str) -> Tuple[str, List[str]]:
        found_vars: List[str] = []
        def replace_var(match: re.Match) -> str:
            var_expr = match.group(1)
            found_vars.append(var_expr)
            parts = var_expr.split("|")
            var_name = parts[0].strip()
            filters = [p.strip() for p in parts[1:]]
            resolved = self._variables.get(var_name, match.group(0))
            for f in filters:
                resolved = self._apply_filter(resolved, f)
            return str(resolved)
        result = self._variable_pattern.sub(replace_var, value)
        return result, found_vars

    def _apply_filter(self, value: Any, filter_expr: str) -> Any:
        filter_parts = filter_expr.split(":", 1)
        func_name = filter_parts[0].strip()
        func = self._functions.get(func_name)
        if func is None:
            return value
        args = []
        if len(filter_parts) > 1:
            args = [a.strip() for a in filter_parts[1].split(":")]
        try:
            return func(value, *args)
        except (TypeError, ValueError):
            return value

    def _expand_dict(
        self, value: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[str]]:
        result: Dict[str, Any] = {}
        all_vars: List[str] = []
        for k, v in value.items():
            expanded, vars_found = self.expand(v)
            result[k] = expanded
            all_vars.extend(vars_found)
        return result, all_vars

    def _expand_list(
        self, value: List[Any]
    ) -> Tuple[List[Any], List[str]]:
        result: List[Any] = []
        all_vars: List[str] = []
        for item in value:
            expanded, vars_found = self.expand(item)
            result.append(expanded)
            all_vars.extend(vars_found)
        return result, all_vars

    def detect_templates(self, value: Any) -> List[str]:
        if isinstance(value, str):
            return self._variable_pattern.findall(value)
        elif isinstance(value, dict):
            templates: List[str] = []
            for v in value.values():
                templates.extend(self.detect_templates(v))
            return templates
        elif isinstance(value, list):
            templates: List[str] = []
            for item in value:
                templates.extend(self.detect_templates(item))
            return templates
        return []


class TypeInferrer:
    """Infers types of parameter values."""

    def __init__(self) -> None:
        self._path_patterns: List[re.Pattern] = [
            re.compile(r'^(/|~/|\.\.?/|[A-Za-z]:\\).+'),
            re.compile(r'^(/|~?/)\.{0,2}/'),
        ]
        self._url_pattern = re.compile(
            r'^(https?|ftp|sftp|ssh|scp|file|ws|wss)://[^\s<>"]+$',
            re.IGNORECASE,
        )
        self._command_patterns: List[re.Pattern] = [
            re.compile(r'^(ls|cat|rm|cp|mv|mkdir|chmod|chown|grep|find|awk|sed|echo|cd|pwd|kill|ps|top|df|du|tar|curl|wget|ssh|scp|rsync|npm|pip|python|bash|sh|docker|kubectl)\b'),
            re.compile(r'^[a-zA-Z_][\w-]*(\s+-\w+)*\s+'),
        ]
        self._expression_patterns: List[re.Pattern] = [
            re.compile(r'^[a-zA-Z_]\w*\s*[+\-*/%]\s*'),
            re.compile(r'^\s*\(?[a-zA-Z_]\w*\s*\.\s*[a-zA-Z_]\w*'),
            re.compile(r'.*\b(if|else|for|while|lambda|def|class)\b'),
        ]

    def infer_type(self, value: Any) -> ParamType:
        if value is None:
            return ParamType.NONE
        if isinstance(value, bool):
            return ParamType.BOOLEAN
        if isinstance(value, int):
            return ParamType.INTEGER
        if isinstance(value, float):
            return ParamType.FLOAT
        if isinstance(value, str):
            return self._infer_string_type(value)
        if isinstance(value, list):
            return ParamType.LIST
        if isinstance(value, dict):
            return ParamType.DICT
        return ParamType.UNKNOWN

    def _infer_string_type(self, value: str) -> ParamType:
        if not value:
            return ParamType.STRING
        stripped = value.strip()
        if self._is_path(stripped):
            return ParamType.PATH
        if self._is_url(stripped):
            return ParamType.URL
        if self._is_command(stripped):
            return ParamType.COMMAND
        if self._is_expression(stripped):
            return ParamType.EXPRESSION
        if stripped.isdigit():
            return ParamType.INTEGER
        try:
            float(stripped)
            return ParamType.FLOAT
        except ValueError:
            pass
        if stripped.lower() in ("true", "false", "yes", "no", "1", "0"):
            return ParamType.BOOLEAN
        return ParamType.STRING

    def _is_path(self, value: str) -> bool:
        return any(p.match(value) for p in self._path_patterns)

    def _is_url(self, value: str) -> bool:
        return bool(self._url_pattern.match(value))

    def _is_command(self, value: str) -> bool:
        return any(p.match(value) for p in self._command_patterns)

    def _is_expression(self, value: str) -> bool:
        return any(p.search(value) for p in self._expression_patterns)

    def infer_all_types(
        self, params: Dict[str, Any]
    ) -> Dict[str, ParamType]:
        return {name: self.infer_type(value) for name, value in params.items()}

    def validate_type(
        self, value: Any, expected_type: ParamType
    ) -> Tuple[bool, str]:
        actual = self.infer_type(value)
        if actual == expected_type:
            return True, ""
        compatible: Dict[ParamType, Set[ParamType]] = {
            ParamType.INTEGER: {ParamType.FLOAT, ParamType.STRING},
            ParamType.FLOAT: {ParamType.INTEGER, ParamType.STRING},
            ParamType.STRING: {ParamType.PATH, ParamType.URL, ParamType.COMMAND},
            ParamType.PATH: {ParamType.STRING},
            ParamType.URL: {ParamType.STRING},
        }
        if actual in compatible.get(expected_type, set()):
            return True, f"Type coercion possible: {actual.value} -> {expected_type.value}"
        return False, f"Type mismatch: expected {expected_type.value}, got {actual.value}"


class SanitizationMarker:
    """Marks parameters that require sanitization."""

    def __init__(self) -> None:
        self._dangerous_content_patterns: List[Tuple[re.Pattern, SanitizationLevel, str]] = [
            (re.compile(r'__import__\s*\('), SanitizationLevel.CRITICAL, "code_injection"),
            (re.compile(r'\beval\s*\('), SanitizationLevel.CRITICAL, "code_execution"),
            (re.compile(r'\bexec\s*\('), SanitizationLevel.CRITICAL, "code_execution"),
            (re.compile(r'\bos\.system\s*\('), SanitizationLevel.CRITICAL, "system_command"),
            (re.compile(r'\bsubprocess\.'), SanitizationLevel.HIGH, "subprocess_access"),
            (re.compile(r'\brm\s+-rf\b'), SanitizationLevel.CRITICAL, "destructive_command"),
            (re.compile(r'\.\.[/\\]'), SanitizationLevel.HIGH, "path_traversal"),
            (re.compile(r'[<>]'), SanitizationLevel.MEDIUM, "html_special_chars"),
            (re.compile(r'\'|"'), SanitizationLevel.LOW, "quote_chars"),
            (re.compile(r'\b(SELECT|INSERT|UPDATE|DELETE|DROP)\b.*\b(FROM|INTO|TABLE)\b', re.I),
             SanitizationLevel.HIGH, "sql_keywords"),
            (re.compile(r'<script\b', re.I), SanitizationLevel.HIGH, "script_tag"),
            (re.compile(r'\bjavascript:'), SanitizationLevel.HIGH, "javascript_uri"),
            (re.compile(r'\bon\w+\s*='), SanitizationLevel.MEDIUM, "event_handler"),
            (re.compile(r'\$\{.*\}'), SanitizationLevel.MEDIUM, "template_expression"),
            (re.compile(r'(?:password|secret|token|api_key)\s*[:=]\s*\S+', re.I),
             SanitizationLevel.HIGH, "credential_in_param"),
        ]
        self._type_sanitization: Dict[ParamType, SanitizationLevel] = {
            ParamType.COMMAND: SanitizationLevel.HIGH,
            ParamType.EXPRESSION: SanitizationLevel.MEDIUM,
            ParamType.PATH: SanitizationLevel.MEDIUM,
            ParamType.URL: SanitizationLevel.LOW,
        }

    def mark(self, name: str, value: Any, param_type: ParamType) -> Tuple[SanitizationLevel, List[str]]:
        level = SanitizationLevel.NONE
        flags: List[str] = []
        type_level = self._type_sanitization.get(param_type, SanitizationLevel.NONE)
        if type_level.value > level.value:
            level = type_level
            flags.append(f"type_based:{param_type.value}")
        if isinstance(value, str):
            content_level, content_flags = self._check_content(value)
            if content_level.value > level.value:
                level = content_level
            flags.extend(content_flags)
        if isinstance(value, (dict, list)):
            nested_level, nested_flags = self._check_nested(value)
            if nested_level.value > level.value:
                level = nested_level
            flags.extend(nested_flags)
        return level, flags

    def _check_content(
        self, value: str
    ) -> Tuple[SanitizationLevel, List[str]]:
        max_level = SanitizationLevel.NONE
        flags: List[str] = []
        for pattern, level, flag_name in self._dangerous_content_patterns:
            if pattern.search(value):
                if level.value > max_level.value:
                    max_level = level
                flags.append(flag_name)
        return max_level, flags

    def _check_nested(
        self, value: Union[Dict[str, Any], List[Any]]
    ) -> Tuple[SanitizationLevel, List[str]]:
        max_level = SanitizationLevel.NONE
        flags: List[str] = []
        items: List[Any] = []
        if isinstance(value, dict):
            items = list(value.values())
        else:
            items = value
        for item in items:
            if isinstance(item, str):
                level, item_flags = self._check_content(item)
                if level.value > max_level.value:
                    max_level = level
                flags.extend(item_flags)
            elif isinstance(item, (dict, list)):
                level, item_flags = self._check_nested(item)
                if level.value > max_level.value:
                    max_level = level
                flags.extend(item_flags)
        return max_level, flags

    def mark_all(
        self, params: Dict[str, Any], type_hints: Optional[Dict[str, ParamType]] = None
    ) -> Dict[str, Tuple[SanitizationLevel, List[str]]]:
        type_inferrer = TypeInferrer()
        results: Dict[str, Tuple[SanitizationLevel, List[str]]] = {}
        for name, value in params.items():
            ptype = type_hints.get(name, type_inferrer.infer_type(value)) if type_hints else type_inferrer.infer_type(value)
            results[name] = self.mark(name, value, ptype)
        return results


class ParamExtractor:
    """Main parameter extraction orchestrator."""

    def __init__(self) -> None:
        self.ast_parser = ASTParser()
        self.nested_resolver = NestedResolver()
        self.template_expander = TemplateExpander()
        self.type_inferrer = TypeInferrer()
        self.sanitization_marker = SanitizationMarker()

    def extract(
        self,
        tool_name: str,
        params_input: Union[str, Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        max_depth: int = 10,
    ) -> ExtractedParams:
        start_time = time.time()
        call_id = uuid.uuid4().hex[:12]
        if context:
            self.nested_resolver.set_context(context)
            self.template_expander.set_variables(context)
        raw_params: Dict[str, Any] = {}
        raw_input_str = ""
        if isinstance(params_input, str):
            raw_input_str = params_input
            parsed = self._parse_input(params_input)
            if parsed is not None:
                raw_params = parsed
            else:
                raw_params = {"raw_input": params_input}
        elif isinstance(params_input, dict):
            raw_params = params_input
            raw_input_str = json.dumps(params_input, default=str)
        extracted_params: List[ExtractedParam] = []
        max_depth_found = 0
        has_templates = False
        has_nested = False
        sanitization_required = False
        for name, raw_value in raw_params.items():
            resolved_value, depth = self.nested_resolver.resolve(raw_value, max_depth=max_depth)
            expanded_value, template_vars = self.template_expander.expand(resolved_value)
            param_type = self.type_inferrer.infer_type(expanded_value)
            san_level, san_flags = self.sanitization_marker.mark(name, expanded_value, param_type)
            nesting_depth = self.nested_resolver.get_max_nesting_depth(raw_value)
            is_template = len(template_vars) > 0
            ep = ExtractedParam(
                name=name,
                raw_value=raw_value,
                resolved_value=expanded_value,
                param_type=param_type,
                sanitization_level=san_level,
                source="parsed",
                nested_path=name,
                is_template=is_template,
                template_variables=template_vars,
                sanitization_flags=san_flags,
                depth=max(depth, nesting_depth),
            )
            extracted_params.append(ep)
            max_depth_found = max(max_depth_found, ep.depth)
            if is_template:
                has_templates = True
            if nesting_depth > 1:
                has_nested = True
            if san_level.value > SanitizationLevel.NONE.value:
                sanitization_required = True
        elapsed_ms = (time.time() - start_time) * 1000
        return ExtractedParams(
            call_id=call_id,
            tool_name=tool_name,
            parameters=extracted_params,
            extraction_time_ms=elapsed_ms,
            total_params=len(extracted_params),
            max_depth=max_depth_found,
            has_templates=has_templates,
            has_nested=has_nested,
            sanitization_required=sanitization_required,
            raw_input=raw_input_str,
        )

    def _parse_input(self, input_str: str) -> Optional[Dict[str, Any]]:
        result = self.ast_parser.parse_json_args(input_str)
        if result is not None:
            return result
        result = self.ast_parser.parse_dict_string(input_str)
        if result is not None:
            return result
        call_info = self.ast_parser.parse_call_string(input_str)
        if call_info is not None:
            params: Dict[str, Any] = {}
            for i, arg in enumerate(call_info.get("positional_args", [])):
                params[f"arg_{i}"] = arg
            params.update(call_info.get("keyword_args", {}))
            return params
        return None

    def extract_simple(
        self, tool_name: str, params: Dict[str, Any]
    ) -> ExtractedParams:
        return self.extract(tool_name, params)

    def sanitize_params(
        self, extracted: ExtractedParams
    ) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        for param in extracted.parameters:
            value = param.resolved_value
            if param.sanitization_level.value >= SanitizationLevel.HIGH.value:
                value = self._apply_sanitization(value, param.sanitization_flags)
            sanitized[param.name] = value
        return sanitized

    def _apply_sanitization(self, value: Any, flags: List[str]) -> Any:
        if isinstance(value, str):
            result = value
            if "quote_chars" in flags:
                result = result.replace("'", "\\'").replace('"', '\\"')
            if "html_special_chars" in flags:
                result = result.replace("<", "&lt;").replace(">", "&gt;")
            if "path_traversal" in flags:
                result = result.replace("../", "").replace("..\\", "")
            if "code_injection" in flags or "code_execution" in flags:
                result = "[SANITIZED: potentially dangerous code]"
            if "credential_in_param" in flags:
                result = "[SANITIZED: credential detected]"
            return result
        elif isinstance(value, dict):
            return {k: self._apply_sanitization(v, flags) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._apply_sanitization(item, flags) for item in value]
        return value
