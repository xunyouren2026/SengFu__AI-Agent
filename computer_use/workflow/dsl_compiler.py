"""
DSL Compiler for Workflows Module

Compiles YAML/JSON DSL into executable workflows:
- DSL parsing (YAML and JSON formats)
- AST (Abstract Syntax Tree) construction
- Workflow validation
- AST optimization
- Code generation to executable workflow

Pure Python standard library only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Optional, Any, Union


class ASTNodeType(Enum):
    """Types of AST nodes."""
    WORKFLOW = "workflow"
    STEP = "step"
    ACTION = "action"
    CONDITION = "condition"
    LOOP = "loop"
    VARIABLE = "variable"
    ASSIGNMENT = "assignment"
    EXPRESSION = "expression"
    LITERAL = "literal"
    IDENTIFIER = "identifier"
    BINARY_OP = "binary_op"
    UNARY_OP = "unary_op"
    FUNCTION_CALL = "function_call"
    BLOCK = "block"
    BRANCH = "branch"
    TRY_CATCH = "try_catch"
    WAIT = "wait"
    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE_TEXT = "type_text"
    SCROLL = "scroll"
    EXTRACT = "extract"
    ASSERT = "assert"
    SCREENSHOT = "screenshot"
    COMMENT = "comment"
    SEQUENCE = "sequence"
    PARALLEL = "parallel"


class ValidationSeverity(Enum):
    """Validation issue severity."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ASTNode:
    """A node in the Abstract Syntax Tree."""
    node_type: ASTNodeType
    value: Any = None
    children: List[ASTNode] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    source_line: int = 0
    source_col: int = 0

    def add_child(self, child: ASTNode) -> ASTNode:
        """Add a child node."""
        self.children.append(child)
        return self

    def find_children(self, node_type: ASTNodeType) -> List[ASTNode]:
        """Find all children of a specific type."""
        results: List[ASTNode] = []
        for child in self.children:
            if child.node_type == node_type:
                results.append(child)
            results.extend(child.find_children(node_type))
        return results

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result: Dict[str, Any] = {
            "type": self.node_type.value,
            "value": self.value,
            "attributes": self.attributes,
            "children": [c.to_dict() for c in self.children],
        }
        if self.source_line:
            result["line"] = self.source_line
        return result


@dataclass
class ValidationIssue:
    """A validation issue found during compilation."""
    severity: ValidationSeverity
    message: str
    node: Optional[ASTNode] = None
    rule: str = ""
    suggestion: str = ""

    def __repr__(self) -> str:
        return f"[{self.severity.value}] {self.message}"


class DSLParser:
    """
    Parses YAML/JSON DSL into an AST.

    Supports both JSON format and a simplified YAML-like format
    parsed using basic string processing (no PyYAML dependency).
    """

    def parse(self, source: str, format: str = "json") -> ASTNode:
        """Parse DSL source into an AST."""
        if format == "json":
            data = self._parse_json(source)
        else:
            data = self._parse_yaml(source)
        return self._build_ast(data)

    def _parse_json(self, source: str) -> Dict[str, Any]:
        """Parse JSON source."""
        try:
            return json.loads(source)
        except json.JSONDecodeError as e:
            raise SyntaxError(f"JSON parse error: {e}")

    def _parse_yaml(self, source: str) -> Dict[str, Any]:
        """
        Parse a simplified YAML-like format.

        Supports:
        - key: value pairs
        - indentation-based nesting
        - lists with -
        - quoted strings
        - comments with #
        """
        lines = source.split("\n")
        return self._parse_yaml_lines(lines, 0, 0)[0]

    def _parse_yaml_lines(self, lines: List[str], start: int,
                           base_indent: int) -> Tuple[Dict[str, Any], int]:
        """Parse YAML lines at a given indentation level."""
        result: Dict[str, Any] = {}
        current_list: Optional[List[Any]] = None
        current_key: str = ""
        i = start

        while i < len(lines):
            line = lines[i]
            stripped = line.lstrip()

            # Skip empty lines and comments
            if not stripped or stripped.startswith("#"):
                i += 1
                continue

            indent = len(line) - len(stripped)
            if indent < base_indent:
                break

            # List item
            if stripped.startswith("- "):
                if current_list is None:
                    current_list = []
                    if current_key:
                        result[current_key] = current_list
                item_content = stripped[2:].strip()
                if item_content.endswith(":"):
                    # Nested dict in list
                    sub_dict, i = self._parse_yaml_lines(lines, i + 1, indent + 2)
                    current_list.append(sub_dict)
                    continue
                else:
                    current_list.append(self._parse_yaml_value(item_content))
                i += 1
                continue

            # Key-value pair
            colon_idx = stripped.find(":")
            if colon_idx > 0:
                key = stripped[:colon_idx].strip()
                value_str = stripped[colon_idx + 1:].strip()

                current_key = key
                current_list = None

                if not value_str or value_str.startswith("#"):
                    # Nested structure
                    sub_dict, i = self._parse_yaml_lines(lines, i + 1, indent + 2)
                    if sub_dict:
                        result[key] = sub_dict
                    else:
                        result[key] = {}
                elif value_str.startswith("["):
                    # Inline list
                    result[key] = self._parse_inline_list(value_str)
                    i += 1
                elif value_str.startswith("{"):
                    # Inline dict
                    result[key] = self._parse_inline_dict(value_str)
                    i += 1
                else:
                    result[key] = self._parse_yaml_value(value_str)
                    i += 1
            else:
                i += 1

        return result, i

    def _parse_yaml_value(self, value: str) -> Any:
        """Parse a YAML value."""
        value = value.strip()
        if value.startswith('"') and value.endswith('"'):
            return value[1:-1]
        if value.startswith("'") and value.endswith("'"):
            return value[1:-1]
        if value.lower() in ("true", "yes"):
            return True
        if value.lower() in ("false", "no"):
            return False
        if value.lower() == "null" or value == "~":
            return None
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    def _parse_inline_list(self, value: str) -> List[Any]:
        """Parse an inline YAML list [a, b, c]."""
        value = value.strip()[1:-1].strip()
        if not value:
            return []
        items: List[Any] = []
        for item in value.split(","):
            items.append(self._parse_yaml_value(item.strip()))
        return items

    def _parse_inline_dict(self, value: str) -> Dict[str, Any]:
        """Parse an inline YAML dict {a: b, c: d}."""
        value = value.strip()[1:-1].strip()
        if not value:
            return {}
        result: Dict[str, Any] = {}
        for pair in value.split(","):
            colon_idx = pair.find(":")
            if colon_idx > 0:
                key = pair[:colon_idx].strip()
                val = pair[colon_idx + 1:].strip()
                result[key] = self._parse_yaml_value(val)
        return result

    def _build_ast(self, data: Dict[str, Any]) -> ASTNode:
        """Build an AST from parsed data."""
        root = ASTNode(
            node_type=ASTNodeType.WORKFLOW,
            attributes={
                "name": data.get("name", "Untitled"),
                "version": data.get("version", "1.0"),
                "description": data.get("description", ""),
            },
        )

        # Parse variables
        variables = data.get("variables", {})
        if variables:
            var_node = ASTNode(node_type=ASTNodeType.VARIABLE, value=variables)
            root.add_child(var_node)

        # Parse steps
        steps = data.get("steps", [])
        for step_data in steps:
            step_node = self._parse_step(step_data)
            root.add_child(step_node)

        # Parse settings
        settings = data.get("settings", {})
        if settings:
            settings_node = ASTNode(
                node_type=ASTNodeType.ASSIGNMENT,
                value=settings,
                attributes={"scope": "settings"},
            )
            root.add_child(settings_node)

        return root

    def _parse_step(self, data: Dict[str, Any]) -> ASTNode:
        """Parse a single step into an AST node."""
        action = data.get("action", data.get("type", "unknown"))

        action_map = {
            "click": ASTNodeType.CLICK,
            "navigate": ASTNodeType.NAVIGATE,
            "type": ASTNodeType.TYPE_TEXT,
            "scroll": ASTNodeType.SCROLL,
            "wait": ASTNodeType.WAIT,
            "extract": ASTNodeType.EXTRACT,
            "assert": ASTNodeType.ASSERT,
            "screenshot": ASTNodeType.SCREENSHOT,
            "if": ASTNodeType.CONDITION,
            "loop": ASTNodeType.LOOP,
            "try": ASTNodeType.TRY_CATCH,
            "parallel": ASTNodeType.PARALLEL,
            "sequence": ASTNodeType.SEQUENCE,
        }

        node_type = action_map.get(action, ASTNodeType.ACTION)
        step_node = ASTNode(
            node_type=node_type,
            value=action,
            attributes={
                "step_id": data.get("id", ""),
                "name": data.get("name", ""),
                "description": data.get("description", ""),
                "target": data.get("target", data.get("selector", "")),
                "delay_before": data.get("delay_before", 0),
                "delay_after": data.get("delay_after", 0),
                "timeout": data.get("timeout", 10),
                "max_retries": data.get("max_retries", 0),
                "skip_on_failure": data.get("skip_on_failure", False),
            },
        )

        # Parse parameters
        params = {k: v for k, v in data.items()
                  if k not in ("id", "name", "description", "action", "type",
                               "steps", "then", "else", "catch", "finally",
                               "delay_before", "delay_after", "timeout",
                               "max_retries", "skip_on_failure", "condition",
                               "iterations", "variable")}
        if params:
            step_node.attributes["parameters"] = params

        # Handle conditionals
        if action == "if":
            condition = data.get("condition", "")
            cond_node = ASTNode(
                node_type=ASTNodeType.EXPRESSION,
                value=condition,
            )
            step_node.add_child(cond_node)

            then_steps = data.get("then", data.get("steps", []))
            for s in then_steps:
                step_node.add_child(self._parse_step(s))

            else_steps = data.get("else", [])
            if else_steps:
                else_node = ASTNode(node_type=ASTNodeType.BRANCH, value="else")
                for s in else_steps:
                    else_node.add_child(self._parse_step(s))
                step_node.add_child(else_node)

        # Handle loops
        elif action == "loop":
            iterations = data.get("iterations", data.get("count", 1))
            step_node.attributes["iterations"] = iterations
            loop_var = data.get("variable", "i")
            step_node.attributes["loop_variable"] = loop_var

            loop_steps = data.get("steps", [])
            for s in loop_steps:
                step_node.add_child(self._parse_step(s))

        # Handle try-catch
        elif action == "try":
            try_steps = data.get("steps", [])
            for s in try_steps:
                step_node.add_child(self._parse_step(s))

            catch_steps = data.get("catch", [])
            if catch_steps:
                catch_node = ASTNode(node_type=ASTNodeType.BRANCH, value="catch")
                for s in catch_steps:
                    catch_node.add_child(self._parse_step(s))
                step_node.add_child(catch_node)

        # Handle parallel/sequence
        elif action in ("parallel", "sequence"):
            sub_steps = data.get("steps", [])
            for s in sub_steps:
                step_node.add_child(self._parse_step(s))

        return step_node


class WorkflowValidator:
    """
    Validates workflow AST for correctness.

    Checks for structural errors, missing fields, type mismatches,
    and logical issues.
    """

    def __init__(self) -> None:
        self._issues: List[ValidationIssue] = []

    def validate(self, ast: ASTNode) -> List[ValidationIssue]:
        """Validate the AST and return a list of issues."""
        self._issues = []
        self._validate_workflow(ast)
        return self._issues

    def _add_issue(self, severity: ValidationSeverity, message: str,
                   node: Optional[ASTNode] = None, rule: str = "",
                   suggestion: str = "") -> None:
        self._issues.append(ValidationIssue(
            severity=severity, message=message, node=node,
            rule=rule, suggestion=suggestion,
        ))

    def _validate_workflow(self, node: ASTNode) -> None:
        """Validate the workflow root node."""
        if node.node_type != ASTNodeType.WORKFLOW:
            self._add_issue(ValidationSeverity.ERROR,
                            "Root node must be a workflow", node, "root_type")

        if not node.attributes.get("name"):
            self._add_issue(ValidationSeverity.WARNING,
                            "Workflow has no name", node, "workflow_name")

        steps = node.find_children(ASTNodeType.STEP)
        steps.extend(node.find_children(ASTNodeType.CLICK))
        steps.extend(node.find_children(ASTNodeType.NAVIGATE))
        steps.extend(node.find_children(ASTNodeType.TYPE_TEXT))
        steps.extend(node.find_children(ASTNodeType.WAIT))

        if not steps:
            self._add_issue(ValidationSeverity.WARNING,
                            "Workflow has no steps", node, "empty_workflow")

        step_ids: set = set()
        for step in node.children:
            self._validate_node(step, step_ids)

        # Check for duplicate IDs
        if len(step_ids) != len([c for c in node.children if c.attributes.get("step_id")]):
            self._add_issue(ValidationSeverity.ERROR,
                            "Duplicate step IDs found", node, "unique_ids")

    def _validate_node(self, node: ASTNode, step_ids: set) -> None:
        """Validate a single AST node."""
        step_id = node.attributes.get("step_id", "")
        if step_id:
            if step_id in step_ids:
                self._add_issue(ValidationSeverity.ERROR,
                                f"Duplicate step ID: {step_id}", node, "unique_id")
            step_ids.add(step_id)

        if node.node_type == ASTNodeType.NAVIGATE:
            params = node.attributes.get("parameters", {})
            url = params.get("url", node.attributes.get("target", ""))
            if not url:
                self._add_issue(ValidationSeverity.ERROR,
                                "Navigate step has no URL", node, "navigate_url",
                                suggestion="Add 'url' parameter to navigate step")

        elif node.node_type == ASTNodeType.CLICK:
            target = node.attributes.get("target", "")
            if not target:
                self._add_issue(ValidationSeverity.WARNING,
                                "Click step has no target", node, "click_target",
                                suggestion="Add 'target' or 'selector' to click step")

        elif node.node_type == ASTNodeType.TYPE_TEXT:
            params = node.attributes.get("parameters", {})
            text = params.get("text", "")
            if not text:
                self._add_issue(ValidationSeverity.WARNING,
                                "Type step has no text", node, "type_text")

        elif node.node_type == ASTNodeType.CONDITION:
            if not node.children:
                self._add_issue(ValidationSeverity.ERROR,
                                "Condition has no body", node, "condition_body")

        elif node.node_type == ASTNodeType.LOOP:
            iterations = node.attributes.get("iterations", 0)
            if iterations <= 0:
                self._add_issue(ValidationSeverity.WARNING,
                                "Loop has non-positive iterations", node, "loop_iterations")

        elif node.node_type == ASTNodeType.WAIT:
            params = node.attributes.get("parameters", {})
            duration = params.get("duration", 0)
            if duration <= 0:
                self._add_issue(ValidationSeverity.WARNING,
                                "Wait has non-positive duration", node, "wait_duration")

        # Validate children recursively
        for child in node.children:
            self._validate_node(child, step_ids)


class ASTOptimizer:
    """
    Optimizes the workflow AST.

    Performs constant folding, dead code elimination, step merging,
    and other optimizations.
    """

    def optimize(self, ast: ASTNode) -> ASTNode:
        """Apply all optimizations to the AST."""
        ast = self._fold_constants(ast)
        ast = self._eliminate_empty_blocks(ast)
        ast = self._merge_consecutive_waits(ast)
        ast = self._remove_noop_steps(ast)
        return ast

    def _fold_constants(self, node: ASTNode) -> ASTNode:
        """Fold constant expressions."""
        if node.node_type == ASTNodeType.EXPRESSION:
            value = node.value
            if isinstance(value, str):
                # Evaluate simple constant expressions
                try:
                    if re.match(r'^[\d\s\+\-\*\/\(\)\.]+$', value):
                        result = eval(value)  # Only numeric expressions
                        node.value = result
                        node.node_type = ASTNodeType.LITERAL
                except Exception:
                    pass

        for i, child in enumerate(node.children):
            node.children[i] = self._fold_constants(child)
        return node

    def _eliminate_empty_blocks(self, node: ASTNode) -> ASTNode:
        """Remove empty block nodes."""
        new_children: List[ASTNode] = []
        for child in node.children:
            optimized = self._eliminate_empty_blocks(child)
            if optimized.node_type in (ASTNodeType.BLOCK, ASTNodeType.SEQUENCE):
                if not optimized.children:
                    continue
                new_children.extend(optimized.children)
            else:
                new_children.append(optimized)
        node.children = new_children
        return node

    def _merge_consecutive_waits(self, node: ASTNode) -> ASTNode:
        """Merge consecutive wait steps into a single wait."""
        new_children: List[ASTNode] = []
        i = 0
        while i < len(node.children):
            child = node.children[i]
            if (child.node_type == ASTNodeType.WAIT and
                    i + 1 < len(node.children) and
                    node.children[i + 1].node_type == ASTNodeType.WAIT):
                # Merge waits
                total_duration = 0
                while (i < len(node.children) and
                       node.children[i].node_type == ASTNodeType.WAIT):
                    params = node.children[i].attributes.get("parameters", {})
                    total_duration += params.get("duration", 0)
                    i += 1
                merged = ASTNode(
                    node_type=ASTNodeType.WAIT,
                    attributes={"parameters": {"duration": total_duration}},
                )
                new_children.append(merged)
            else:
                new_children.append(self._merge_consecutive_waits(child))
                i += 1
        node.children = new_children
        return node

    def _remove_noop_steps(self, node: ASTNode) -> ASTNode:
        """Remove steps that have no effect."""
        new_children: List[ASTNode] = []
        for child in node.children:
            optimized = self._remove_noop_steps(child)
            # Skip comment-only nodes
            if optimized.node_type == ASTNodeType.COMMENT:
                continue
            # Skip empty waits
            if optimized.node_type == ASTNodeType.WAIT:
                params = optimized.attributes.get("parameters", {})
                if params.get("duration", 0) <= 0:
                    continue
            new_children.append(optimized)
        node.children = new_children
        return node


class CodeGenerator:
    """
    Generates executable workflow data from an AST.

    Converts the optimized AST into a WorkflowData structure
    that can be executed by the PlaybackEngine.
    """

    def __init__(self) -> None:
        self._step_counter = 0

    def generate(self, ast: ASTNode) -> Dict[str, Any]:
        """Generate executable workflow from AST."""
        self._step_counter = 0
        workflow_data: Dict[str, Any] = {
            "workflow_id": f"wf-{int(time.time())}" if hasattr(time, 'time') else "wf-0",
            "name": ast.attributes.get("name", "Generated Workflow"),
            "version": ast.attributes.get("version", "1.0"),
            "description": ast.attributes.get("description", ""),
            "created_at": 0,
            "steps": [],
            "variables": {},
        }

        for child in ast.children:
            if child.node_type == ASTNodeType.VARIABLE:
                workflow_data["variables"] = child.value
            elif child.node_type == ASTNodeType.ASSIGNMENT:
                if child.attributes.get("scope") == "settings":
                    workflow_data["settings"] = child.value
            else:
                steps = self._generate_steps(child)
                workflow_data["steps"].extend(steps)

        return workflow_data

    def _generate_steps(self, node: ASTNode) -> List[Dict[str, Any]]:
        """Generate step dictionaries from an AST node."""
        steps: List[Dict[str, Any]] = []

        if node.node_type in (ASTNodeType.WORKFLOW, ASTNodeType.BLOCK,
                               ASTNodeType.SEQUENCE):
            for child in node.children:
                steps.extend(self._generate_steps(child))
            return steps

        self._step_counter += 1
        step_id = node.attributes.get("step_id", f"step-{self._step_counter}")

        step: Dict[str, Any] = {
            "step_id": step_id,
            "step_type": node.node_type.value,
            "action": node.value or node.node_type.value,
            "name": node.attributes.get("name", ""),
            "parameters": node.attributes.get("parameters", {}),
            "delay_before": node.attributes.get("delay_before", 0),
            "delay_after": node.attributes.get("delay_after", 0),
            "max_retries": node.attributes.get("max_retries", 0),
            "skip_on_failure": node.attributes.get("skip_on_failure", False),
        }

        # Handle special step types
        if node.node_type == ASTNodeType.CLICK:
            step["action"] = "click"
            step["parameters"]["target"] = node.attributes.get("target", "")
        elif node.node_type == ASTNodeType.NAVIGATE:
            step["action"] = "navigate"
            step["parameters"]["url"] = node.attributes.get("target", "")
        elif node.node_type == ASTNodeType.TYPE_TEXT:
            step["action"] = "type"
            params = node.attributes.get("parameters", {})
            step["parameters"]["text"] = params.get("text", "")
            step["parameters"]["target"] = node.attributes.get("target", "")
        elif node.node_type == ASTNodeType.WAIT:
            step["action"] = "wait"
            params = node.attributes.get("parameters", {})
            step["parameters"]["duration"] = params.get("duration", 1.0)
        elif node.node_type == ASTNodeType.CONDITION:
            step["action"] = "condition"
            for child in node.children:
                if child.node_type == ASTNodeType.EXPRESSION:
                    step["parameters"]["condition"] = child.value
                elif child.node_type == ASTNodeType.BRANCH:
                    if child.value == "else":
                        step["else_steps"] = self._generate_steps(child)
                else:
                    step.setdefault("then_steps", [])
                    step["then_steps"].extend(self._generate_steps(child))
        elif node.node_type == ASTNodeType.LOOP:
            step["action"] = "loop"
            step["parameters"]["count"] = node.attributes.get("iterations", 1)
            step["parameters"]["variable"] = node.attributes.get("loop_variable", "i")
            for child in node.children:
                step.setdefault("steps", [])
                step["steps"].extend(self._generate_steps(child))
        elif node.node_type == ASTNodeType.TRY_CATCH:
            step["action"] = "try"
            for child in node.children:
                if child.node_type == ASTNodeType.BRANCH and child.value == "catch":
                    step["catch_steps"] = self._generate_steps(child)
                else:
                    step.setdefault("steps", [])
                    step["steps"].extend(self._generate_steps(child))

        steps.append(step)
        return steps


class DSLCompiler:
    """
    High-level DSL compiler.

    Provides a complete pipeline: parse -> validate -> optimize -> generate.
    """

    def __init__(self) -> None:
        self.parser = DSLParser()
        self.validator = WorkflowValidator()
        self.optimizer = ASTOptimizer()
        self.codegen = CodeGenerator()

    def compile(self, source: str, format: str = "json",
                optimize: bool = True) -> Tuple[Dict[str, Any], List[ValidationIssue]]:
        """
        Compile DSL source into an executable workflow.

        Returns (workflow_data, validation_issues).
        """
        # Parse
        ast = self.parser.parse(source, format)

        # Validate
        issues = self.validator.validate(ast)
        errors = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        if errors:
            return {}, issues

        # Optimize
        if optimize:
            ast = self.optimizer.optimize(ast)

        # Generate
        workflow = self.codegen.generate(ast)

        return workflow, issues

    def compile_file(self, file_path: str,
                     format: str = "json") -> Tuple[Dict[str, Any], List[ValidationIssue]]:
        """Compile a DSL file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            return self.compile(source, format)
        except OSError as e:
            return {}, [ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Failed to read file: {e}",
            )]

    def validate_only(self, source: str,
                      format: str = "json") -> List[ValidationIssue]:
        """Validate DSL source without compiling."""
        ast = self.parser.parse(source, format)
        return self.validator.validate(ast)

    def format_ast(self, source: str, format: str = "json") -> str:
        """Pretty-print the AST for debugging."""
        ast = self.parser.parse(source, format)
        return json.dumps(ast.to_dict(), indent=2, ensure_ascii=False)
