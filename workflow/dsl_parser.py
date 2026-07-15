"""
YAML工作流DSL解析器

解析YAML格式的工作流定义，转换为DAG引擎可执行的图结构。
包含内置的简易YAML解析器（不依赖PyYAML）。
"""

import copy
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .graph_engine import DAGEdge, DAGEngine, DAGNode, NodeState


class DSLError(Exception):
    """DSL解析异常"""
    pass


class DSLValidationError(DSLError):
    """DSL验证异常"""
    pass


class SimpleYAMLParser:
    """
    简易YAML解析器

    支持YAML子集：映射、列表、字符串、数字、布尔值、null。
    不依赖PyYAML，仅使用Python标准库。

    支持的语法：
    - key: value
    - key:
        sub_key: value
    - - item1
      - item2
    - key: |
        多行文本
    - 引号字符串
    - 注释 (#)
    """

    def parse(self, yaml_string: str) -> Any:
        """
        解析YAML字符串

        Args:
            yaml_string: YAML格式字符串

        Returns:
            解析后的Python对象
        """
        if not yaml_string or not yaml_string.strip():
            return {}

        lines = yaml_string.split("\n")
        return self._parse_lines(lines)

    def _parse_lines(self, lines: List[str]) -> Any:
        """解析行列表"""
        # 预处理：去除注释、空行，计算缩进
        processed = []
        for line in lines:
            # 去除注释（但保留引号内的#）
            stripped = self._strip_comment(line)
            if stripped.strip():
                indent = len(stripped) - len(stripped.lstrip())
                processed.append((indent, stripped.rstrip()))

        if not processed:
            return {}

        # 判断根类型
        min_indent = min(indent for indent, _ in processed)

        # 检查是否是列表
        list_items = [
            (indent, content) for indent, content in processed
            if content.lstrip().startswith("- ")
        ]

        if list_items and all(
            indent == min_indent for indent, _ in list_items
        ):
            return self._parse_list(processed, min_indent)

        return self._parse_mapping(processed, min_indent)

    def _parse_mapping(
        self, items: List[Tuple[int, str]], base_indent: int
    ) -> Dict[str, Any]:
        """解析映射"""
        result: Dict[str, Any] = {}
        i = 0

        while i < len(items):
            indent, content = items[i]

            if indent != base_indent:
                i += 1
                continue

            # 解析 key: value
            colon_pos = content.find(":")
            if colon_pos == -1:
                i += 1
                continue

            key = content[:colon_pos].strip().strip("'\"")
            value_str = content[colon_pos + 1:].strip()

            if not value_str:
                # 值在下一行或子行
                if i + 1 < len(items):
                    next_indent, next_content = items[i + 1]
                    if next_indent > base_indent:
                        # 检查是否是列表
                        if next_content.lstrip().startswith("- "):
                            sub_items = self._collect_block(items, i + 1, next_indent)
                            result[key] = self._parse_list(sub_items, next_indent)
                            i = self._skip_block(items, i + 1, next_indent)
                            continue
                        else:
                            sub_items = self._collect_block(items, i + 1, next_indent)
                            result[key] = self._parse_mapping(sub_items, next_indent)
                            i = self._skip_block(items, i + 1, next_indent)
                            continue

                result[key] = None
                i += 1
            elif value_str == "|":
                # 多行文本
                multiline = []
                i += 1
                while i < len(items):
                    _, line_content = items[i]
                    if line_content.strip() and not line_content.startswith(" " * (base_indent + 2)):
                        break
                    multiline.append(line_content.strip())
                    i += 1
                result[key] = "\n".join(multiline)
            elif value_str.startswith("[") and value_str.endswith("]"):
                # 内联列表
                result[key] = self._parse_inline_list(value_str)
                i += 1
            elif value_str.startswith("{") and value_str.endswith("}"):
                # 内联映射
                result[key] = self._parse_inline_mapping(value_str)
                i += 1
            else:
                result[key] = self._parse_value(value_str)
                i += 1

        return result

    def _parse_list(
        self, items: List[Tuple[int, str]], base_indent: int
    ) -> List[Any]:
        """解析列表"""
        result: List[Any] = []
        i = 0

        while i < len(items):
            indent, content = items[i]
            stripped = content.lstrip()

            if not stripped.startswith("- "):
                i += 1
                continue

            value_str = stripped[2:].strip()

            if not value_str:
                # 值在子行
                if i + 1 < len(items):
                    next_indent, next_content = items[i + 1]
                    if next_indent > indent:
                        sub_items = self._collect_block(items, i + 1, next_indent)
                        if next_content.lstrip().startswith("- "):
                            result.append(self._parse_list(sub_items, next_indent))
                        else:
                            result.append(self._parse_mapping(sub_items, next_indent))
                        i = self._skip_block(items, i + 1, next_indent)
                        continue

                result.append(None)
                i += 1
            elif value_str.startswith("[") and value_str.endswith("]"):
                result.append(self._parse_inline_list(value_str))
                i += 1
            elif value_str.startswith("{") and value_str.endswith("}"):
                result.append(self._parse_inline_mapping(value_str))
                i += 1
            else:
                # Check if this list item starts a mapping (key: value or key:)
                colon_pos = value_str.find(":")
                if colon_pos != -1:
                    # This is a mapping item - collect all sub-items at deeper indent
                    # The first key-value is on the same line as the dash
                    first_line_indent = indent + 2  # indent of the key after "- "
                    sub_items = [(first_line_indent, value_str)]

                    # Collect subsequent lines that are indented deeper than the dash
                    j = i + 1
                    while j < len(items):
                        ni, nc = items[j]
                        if ni > indent:
                            sub_items.append((ni, nc))
                            j += 1
                        else:
                            break

                    result.append(self._parse_mapping(sub_items, first_line_indent))
                    i = j
                else:
                    result.append(self._parse_value(value_str))
                    i += 1

        return result

    def _parse_value(self, value_str: str) -> Any:
        """解析值"""
        value_str = value_str.strip().strip("'\"")

        if value_str.lower() in ("true", "yes", "on"):
            return True
        elif value_str.lower() in ("false", "no", "off"):
            return False
        elif value_str.lower() in ("null", "~", "none", ""):
            return None
        else:
            # 尝试解析数字
            try:
                return int(value_str)
            except ValueError:
                pass
            try:
                return float(value_str)
            except ValueError:
                pass
            return value_str

    def _parse_inline_list(self, s: str) -> List[Any]:
        """解析内联列表 [a, b, c]"""
        s = s[1:-1].strip()
        if not s:
            return []
        items = [item.strip().strip("'\"") for item in s.split(",")]
        return [self._parse_value(item) for item in items]

    def _parse_inline_mapping(self, s: str) -> Dict[str, Any]:
        """解析内联映射 {k: v, k2: v2}"""
        s = s[1:-1].strip()
        if not s:
            return {}
        result = {}
        pairs = s.split(",")
        for pair in pairs:
            colon_pos = pair.find(":")
            if colon_pos != -1:
                key = pair[:colon_pos].strip().strip("'\"")
                value = pair[colon_pos + 1:].strip()
                result[key] = self._parse_value(value)
        return result

    def _strip_comment(self, line: str) -> str:
        """去除注释"""
        in_quote = False
        quote_char = ""
        i = 0
        while i < len(line):
            c = line[i]
            if in_quote:
                if c == quote_char:
                    in_quote = False
            else:
                if c in ("'", '"'):
                    in_quote = True
                    quote_char = c
                elif c == "#":
                    return line[:i]
            i += 1
        return line

    def _collect_block(
        self,
        items: List[Tuple[int, str]],
        start: int,
        base_indent: int,
    ) -> List[Tuple[int, str]]:
        """收集缩进块"""
        block = []
        i = start
        while i < len(items):
            indent, content = items[i]
            if indent >= base_indent:
                block.append((indent, content))
                i += 1
            else:
                break
        return block

    def _skip_block(
        self,
        items: List[Tuple[int, str]],
        start: int,
        base_indent: int,
    ) -> int:
        """跳过缩进块"""
        i = start
        while i < len(items):
            if items[i][0] >= base_indent:
                i += 1
            else:
                break
        return i


class WorkflowDSLParser:
    """
    工作流DSL解析器

    将YAML格式的工作流定义解析为DAGEngine实例。

    DSL格式示例:
        workflow:
          name: "my_workflow"
          nodes:
            - id: start
              type: task
              config:
                function: my_func
            - id: process
              type: llm
              config:
                model: gpt-4
                prompt: "Hello {{name}}"
          edges:
            - from: start
              to: process
            - from: process
              to: end
              condition: "{{result}} == 'ok'"
          variables:
            name: "World"
            max_retries: 3

    Args:
        custom_yaml_parser: 自定义YAML解析器，为None使用内置解析器
    """

    def __init__(self, custom_yaml_parser: Optional[SimpleYAMLParser] = None):
        self._yaml = custom_yaml_parser or SimpleYAMLParser()
        self._variables: Dict[str, Any] = {}
        self._node_registry: Dict[str, Callable] = {}

    def parse(self, yaml_string: str) -> DAGEngine:
        """
        解析YAML工作流定义为DAG

        Args:
            yaml_string: YAML格式的工作流定义

        Returns:
            DAGEngine实例

        Raises:
            DSLError: 解析失败
        """
        # 解析YAML
        data = self._yaml.parse(yaml_string)
        if not isinstance(data, dict):
            raise DSLError("工作流定义必须是映射类型")

        # 提取工作流定义
        workflow_def = data.get("workflow", data)

        # 提取变量
        self._variables = workflow_def.get("variables", {})

        # 创建DAG引擎
        name = workflow_def.get("name", "")
        dag = DAGEngine(name=name)

        # 解析节点
        nodes_def = workflow_def.get("nodes", [])
        if not nodes_def:
            raise DSLError("工作流定义中没有节点")

        for node_def in nodes_def:
            node = self._parse_node(node_def)
            dag.add_node(node)

        # 解析边
        edges_def = workflow_def.get("edges", [])
        for edge_def in edges_def:
            self._parse_edge(edge_def, dag)

        # 验证
        errors = dag.validate()
        if errors:
            raise DSLValidationError(
                f"DAG验证失败: {'; '.join(errors)}"
            )

        return dag

    def validate(self, yaml_string: str) -> List[str]:
        """
        校验DSL合法性（不抛出异常）

        Args:
            yaml_string: YAML格式的工作流定义

        Returns:
            错误消息列表，空列表表示合法
        """
        errors: List[str] = []

        try:
            data = self._yaml.parse(yaml_string)
        except Exception as e:
            return [f"YAML解析失败: {e}"]

        if not isinstance(data, dict):
            return ["工作流定义必须是映射类型"]

        workflow_def = data.get("workflow", data)

        # 检查节点
        nodes_def = workflow_def.get("nodes", [])
        if not nodes_def:
            errors.append("工作流定义中没有节点")
            return errors

        node_ids: List[str] = []
        for i, node_def in enumerate(nodes_def):
            if not isinstance(node_def, dict):
                errors.append(f"节点定义 #{i} 不是映射类型")
                continue

            node_id = node_def.get("id")
            if not node_id:
                errors.append(f"节点 #{i} 缺少id字段")
            elif node_id in node_ids:
                errors.append(f"节点ID重复: {node_id}")
            else:
                node_ids.append(node_id)

            node_type = node_def.get("type")
            if not node_type:
                errors.append(f"节点 {node_id or i} 缺少type字段")

        # 检查边
        edges_def = workflow_def.get("edges", [])
        for i, edge_def in enumerate(edges_def):
            if not isinstance(edge_def, dict):
                errors.append(f"边定义 #{i} 不是映射类型")
                continue

            from_node = edge_def.get("from", edge_def.get("from_node"))
            to_node = edge_def.get("to", edge_def.get("to_node"))

            if not from_node:
                errors.append(f"边 #{i} 缺少from字段")
            elif from_node not in node_ids:
                errors.append(f"边 #{i} 引用了不存在的源节点: {from_node}")

            if not to_node:
                errors.append(f"边 #{i} 缺少to字段")
            elif to_node not in node_ids:
                errors.append(f"边 #{i} 引用了不存在的目标节点: {to_node}")

            if from_node and to_node and from_node == to_node:
                errors.append(f"边 #{i} 存在自环: {from_node}")

        return errors

    def set_variable(self, key: str, value: Any) -> None:
        """设置模板变量"""
        self._variables[key] = value

    def register_node_type(
        self,
        node_type: str,
        factory: Callable,
    ) -> None:
        """
        注册自定义节点类型工厂

        Args:
            node_type: 节点类型名称
            factory: 工厂函数，接收node_def返回DAGNode
        """
        self._node_registry[node_type] = factory

    # ============================================================
    # 内部方法
    # ============================================================

    def _parse_node(self, node_def: Dict[str, Any]) -> DAGNode:
        """解析单个节点定义"""
        node_id = node_def.get("id", "")
        if not node_id:
            raise DSLError("节点定义缺少id字段")

        node_type = node_def.get("type", "task")
        name = node_def.get("name", node_id)
        config = node_def.get("config", {})
        timeout = node_def.get("timeout")
        max_retries = node_def.get("max_retries", 0)
        dependencies = node_def.get("dependencies", [])

        # 替换模板变量
        config = self._resolve_variables(config)
        if isinstance(timeout, str):
            timeout = self._resolve_variable(timeout)
            if timeout is not None:
                timeout = float(timeout)

        # 检查自定义节点工厂
        if node_type in self._node_registry:
            return self._node_registry[node_type](node_def)

        node = DAGNode(
            id=node_id,
            name=name,
            node_type=node_type,
            config=config,
            timeout=timeout,
            max_retries=max_retries,
            dependencies=dependencies,
        )

        return node

    def _parse_edge(self, edge_def: Dict[str, Any], dag: DAGEngine) -> None:
        """解析单条边定义"""
        from_node = edge_def.get("from", edge_def.get("from_node", ""))
        to_node = edge_def.get("to", edge_def.get("to_node", ""))

        if not from_node or not to_node:
            raise DSLError(f"边定义缺少from或to字段: {edge_def}")

        condition_expr = edge_def.get("condition")
        data_mapping = edge_def.get("data_mapping", {})

        # 解析条件表达式
        condition_fn = None
        if condition_expr:
            condition_fn = self._parse_condition_expression(condition_expr)

        dag.add_edge(
            from_node=from_node,
            to_node=to_node,
            condition=condition_fn,
            condition_expr=condition_expr,
            data_mapping=data_mapping,
        )

    def _parse_condition_expression(
        self, expr: str
    ) -> Callable[[Dict[str, Any]], bool]:
        """
        解析条件表达式

        支持简单的比较表达式，如：
        - "{{result}} == 'ok'"
        - "{{status}} != 'error'"
        - "{{count}} > 0"
        - "{{name}} contains 'test'"
        """
        def condition(context: Dict[str, Any]) -> bool:
            resolved = self._resolve_template_string(expr, context)

            # 尝试解析简单比较
            for op, op_fn in [
                ("==", lambda a, b: a == b),
                ("!=", lambda a, b: a != b),
                (">=", lambda a, b: a is not None and a >= b),
                ("<=", lambda a, b: a is not None and a <= b),
                (">", lambda a, b: a is not None and a > b),
                ("<", lambda a, b: a is not None and a < b),
                (" contains ", lambda a, b: b in str(a) if a else False),
            ]:
                if op in resolved:
                    parts = resolved.split(op, 1)
                    left = parts[0].strip().strip("'\"")
                    right = parts[1].strip().strip("'\"")

                    # 尝试转换为数字
                    left = self._try_number(left)
                    right = self._try_number(right)

                    return op_fn(left, right)

            # 布尔值
            if resolved.lower() in ("true", "yes"):
                return True
            if resolved.lower() in ("false", "no"):
                return False

            return bool(resolved)

        return condition

    def _resolve_variables(self, data: Any) -> Any:
        """递归解析模板变量"""
        if isinstance(data, dict):
            return {k: self._resolve_variables(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._resolve_variables(item) for item in data]
        elif isinstance(data, str):
            return self._resolve_template_string(data, self._variables)
        return data

    def _resolve_template_string(
        self, template: str, context: Dict[str, Any]
    ) -> str:
        """解析模板字符串中的 {{variable}} 占位符"""
        if not isinstance(template, str):
            return str(template)

        import re
        pattern = r"\{\{(\w+(?:\.\w+)*)\}\}"

        def replacer(match: Any) -> str:
            var_path = match.group(1)
            value = self._resolve_variable(var_path, context)
            if value is None:
                return match.group(0)
            return str(value)

        return re.sub(pattern, replacer, template)

    def _resolve_variable(
        self, path: str, context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """解析变量路径"""
        if context is None:
            context = self._variables

        parts = path.split(".")
        current: Any = context

        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
            if current is None:
                return None

        return current

    def _try_number(self, value: str) -> Any:
        """尝试将字符串转换为数字"""
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value
