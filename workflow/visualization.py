"""
工作流可视化模块

提供工作流可视化功能：
- 图布局（DOT算法）
- Mermaid/Graphviz导出
- 进度渲染
- ASCII艺术可视化
- 交互式HTML生成

Classes:
    WorkflowVisualizer: 工作流可视化主类
    GraphLayoutEngine: 图布局引擎
    MermaidExporter: Mermaid导出器
    GraphvizExporter: Graphviz导出器
    ProgressRenderer: 进度渲染器
    ASCIIVisualizer: ASCII可视化器
    HTMLGenerator: HTML生成器
"""

import html
import json
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .graph_engine import DAGEngine, DAGNode, NodeState


class LayoutAlgorithm(Enum):
    """布局算法枚举"""
    DOT = "dot"                 # 层次布局
    CIRCULAR = "circular"       # 圆形布局
    GRID = "grid"               # 网格布局
    FORCE = "force"             # 力导向布局
    TREE = "tree"               # 树形布局


class ExportFormat(Enum):
    """导出格式枚举"""
    MERMAID = "mermaid"
    GRAPHVIZ = "graphviz"
    JSON = "json"
    HTML = "html"
    SVG = "svg"


class VisualizationError(Exception):
    """可视化异常"""
    pass


@dataclass
class Position:
    """位置坐标"""
    x: float = 0.0
    y: float = 0.0

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class NodeLayout:
    """
    节点布局信息

    Attributes:
        node_id: 节点ID
        position: 位置坐标
        width: 宽度
        height: 高度
        label: 显示标签
        style: 样式属性
    """
    node_id: str
    position: Position = dataclass_field(default_factory=Position)
    width: float = 100.0
    height: float = 50.0
    label: str = ""
    style: Dict[str, Any] = dataclass_field(default_factory=dict)


@dataclass
class EdgeLayout:
    """
    边布局信息

    Attributes:
        from_node: 源节点ID
        to_node: 目标节点ID
        path: 路径点列表
        label: 显示标签
        style: 样式属性
    """
    from_node: str
    to_node: str
    path: List[Position] = dataclass_field(default_factory=list)
    label: str = ""
    style: Dict[str, Any] = dataclass_field(default_factory=dict)


@dataclass
class GraphLayout:
    """
    图布局结果

    Attributes:
        nodes: 节点布局字典
        edges: 边布局列表
        width: 图宽度
        height: 图高度
    """
    nodes: Dict[str, NodeLayout] = dataclass_field(default_factory=dict)
    edges: List[EdgeLayout] = dataclass_field(default_factory=list)
    width: float = 0.0
    height: float = 0.0


class GraphLayoutEngine:
    """
    图布局引擎

    实现多种图布局算法。

    Usage:
        engine = GraphLayoutEngine(algorithm=LayoutAlgorithm.DOT)
        layout = engine.layout(dag)
    """

    def __init__(self, algorithm: LayoutAlgorithm = LayoutAlgorithm.DOT):
        self._algorithm = algorithm
        self._node_width = 120.0
        self._node_height = 60.0
        self._level_spacing = 100.0
        self._node_spacing = 50.0

    def layout(self, dag: DAGEngine) -> GraphLayout:
        """
        计算图布局

        Args:
            dag: DAG引擎实例

        Returns:
            图布局结果
        """
        if self._algorithm == LayoutAlgorithm.DOT:
            return self._layout_dot(dag)
        elif self._algorithm == LayoutAlgorithm.CIRCULAR:
            return self._layout_circular(dag)
        elif self._algorithm == LayoutAlgorithm.GRID:
            return self._layout_grid(dag)
        elif self._algorithm == LayoutAlgorithm.TREE:
            return self._layout_tree(dag)
        else:
            return self._layout_dot(dag)

    def _layout_dot(self, dag: DAGEngine) -> GraphLayout:
        """
        DOT层次布局算法

        将节点按层次排列，同一层的节点水平排列。
        """
        result = GraphLayout()

        # 获取分层执行顺序
        try:
            layers = dag.get_execution_order()
        except Exception:
            # 如果有环，退化为网格布局
            return self._layout_grid(dag)

        # 计算每层的位置
        max_width = 0.0
        for level_idx, layer in enumerate(layers):
            y = level_idx * (self._node_height + self._level_spacing)
            layer_width = len(layer) * self._node_width + (len(layer) - 1) * self._node_spacing
            start_x = -layer_width / 2

            for node_idx, node_id in enumerate(layer):
                x = start_x + node_idx * (self._node_width + self._node_spacing)
                node = dag.get_node(node_id)
                label = node.name if node else node_id

                result.nodes[node_id] = NodeLayout(
                    node_id=node_id,
                    position=Position(x, y),
                    width=self._node_width,
                    height=self._node_height,
                    label=label,
                )

            max_width = max(max_width, layer_width)

        # 计算边布局
        for edge in dag.get_edges():
            from_layout = result.nodes.get(edge.from_node)
            to_layout = result.nodes.get(edge.to_node)

            if from_layout and to_layout:
                # 简单的直线路径
                path = [
                    Position(
                        from_layout.position.x + from_layout.width / 2,
                        from_layout.position.y + from_layout.height / 2,
                    ),
                    Position(
                        to_layout.position.x + to_layout.width / 2,
                        to_layout.position.y + to_layout.height / 2,
                    ),
                ]

                result.edges.append(EdgeLayout(
                    from_node=edge.from_node,
                    to_node=edge.to_node,
                    path=path,
                ))

        result.width = max_width
        result.height = len(layers) * (self._node_height + self._level_spacing)

        return result

    def _layout_circular(self, dag: DAGEngine) -> GraphLayout:
        """圆形布局算法"""
        result = GraphLayout()
        nodes = dag.get_all_nodes()
        n = len(nodes)

        if n == 0:
            return result

        radius = max(n * 30, 200)

        for i, node in enumerate(nodes):
            angle = 2 * math.pi * i / n - math.pi / 2
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)

            result.nodes[node.id] = NodeLayout(
                node_id=node.id,
                position=Position(x, y),
                width=self._node_width,
                height=self._node_height,
                label=node.name,
            )

        # 边布局
        for edge in dag.get_edges():
            from_layout = result.nodes.get(edge.from_node)
            to_layout = result.nodes.get(edge.to_node)

            if from_layout and to_layout:
                result.edges.append(EdgeLayout(
                    from_node=edge.from_node,
                    to_node=edge.to_node,
                    path=[from_layout.position, to_layout.position],
                ))

        result.width = result.height = radius * 2 + self._node_width

        return result

    def _layout_grid(self, dag: DAGEngine) -> GraphLayout:
        """网格布局算法"""
        result = GraphLayout()
        nodes = dag.get_all_nodes()
        n = len(nodes)

        if n == 0:
            return result

        # 计算网格大小
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        for i, node in enumerate(nodes):
            row = i // cols
            col = i % cols

            x = col * (self._node_width + self._node_spacing)
            y = row * (self._node_height + self._level_spacing)

            result.nodes[node.id] = NodeLayout(
                node_id=node.id,
                position=Position(x, y),
                width=self._node_width,
                height=self._node_height,
                label=node.name,
            )

        # 边布局
        for edge in dag.get_edges():
            from_layout = result.nodes.get(edge.from_node)
            to_layout = result.nodes.get(edge.to_node)

            if from_layout and to_layout:
                result.edges.append(EdgeLayout(
                    from_node=edge.from_node,
                    to_node=edge.to_node,
                    path=[from_layout.position, to_layout.position],
                ))

        result.width = cols * (self._node_width + self._node_spacing)
        result.height = rows * (self._node_height + self._level_spacing)

        return result

    def _layout_tree(self, dag: DAGEngine) -> GraphLayout:
        """树形布局算法"""
        # 简化为DOT布局
        return self._layout_dot(dag)


class MermaidExporter:
    """
    Mermaid导出器

    将DAG导出为Mermaid流程图格式。

    Usage:
        exporter = MermaidExporter()
        mermaid_code = exporter.export(dag)
    """

    def __init__(self):
        self._direction = "TD"  # TD (top-down) 或 LR (left-right)
        self._show_state = True

    def set_direction(self, direction: str) -> "MermaidExporter":
        """设置方向 (TD 或 LR)"""
        self._direction = direction
        return self

    def export(self, dag: DAGEngine, highlight_nodes: Optional[List[str]] = None) -> str:
        """
        导出为Mermaid格式

        Args:
            dag: DAG引擎实例
            highlight_nodes: 高亮显示的节点ID列表

        Returns:
            Mermaid代码
        """
        lines = [f"flowchart {self._direction}"]
        highlight_nodes = set(highlight_nodes or [])

        # 定义节点
        for node in dag.get_all_nodes():
            node_style = self._get_node_style(node, node.id in highlight_nodes)
            shape = self._get_node_shape(node.node_type)
            label = node.name or node.id

            if self._show_state:
                label = f"{label}<br/>[{node.state.value}]"

            lines.append(f'    {node.id}{shape[0]}"{label}"{shape[1]}')

            # 样式定义
            if node_style:
                lines.append(f'    style {node.id} {node_style}')

        # 定义边
        for edge in dag.get_edges():
            lines.append(f'    {edge.from_node} --> {edge.to_node}')

        return "\n".join(lines)

    def _get_node_shape(self, node_type: str) -> Tuple[str, str]:
        """获取节点形状"""
        shapes = {
            "task": ("[", "]"),
            "llm": ("[[", "]]"),
            "tool": ("([", "])"),
            "condition": ("{", "}"),
            "loop": ("((", "))"),
            "parallel": ("[/", "/]"),
            "human_approval": ("{", "}"),
            "delay": (">", "]"),
        }
        return shapes.get(node_type, ("[", "]"))

    def _get_node_style(self, node: DAGNode, highlighted: bool) -> str:
        """获取节点样式"""
        if highlighted:
            return "fill:#ff6b6b,stroke:#333,stroke-width:4px"

        state_colors = {
            NodeState.PENDING: "fill:#f0f0f0",
            NodeState.READY: "fill:#fff3cd",
            NodeState.RUNNING: "fill:#cce5ff",
            NodeState.SUCCESS: "fill:#d4edda",
            NodeState.FAILED: "fill:#f8d7da",
            NodeState.SKIPPED: "fill:#e2e3e5",
            NodeState.CANCELLED: "fill:#f5c6cb",
        }

        return state_colors.get(node.state, "")


class GraphvizExporter:
    """
    Graphviz导出器

    将DAG导出为Graphviz DOT格式。

    Usage:
        exporter = GraphvizExporter()
        dot_code = exporter.export(dag)
    """

    def __init__(self):
        self._rankdir = "TB"  # TB, BT, LR, RL
        self._show_state = True

    def set_rankdir(self, rankdir: str) -> "GraphvizExporter":
        """设置排列方向"""
        self._rankdir = rankdir
        return self

    def export(self, dag: DAGEngine, highlight_nodes: Optional[List[str]] = None) -> str:
        """
        导出为Graphviz DOT格式

        Args:
            dag: DAG引擎实例
            highlight_nodes: 高亮显示的节点ID列表

        Returns:
            DOT代码
        """
        lines = [
            "digraph Workflow {",
            f'    rankdir={self._rankdir};',
            '    node [shape=box, style="rounded,filled", fontname="Arial"];',
            '    edge [fontname="Arial"];',
            "",
        ]
        highlight_nodes = set(highlight_nodes or [])

        # 定义节点
        for node in dag.get_all_nodes():
            attrs = self._get_node_attrs(node, node.id in highlight_nodes)
            label = node.name or node.id

            if self._show_state:
                label = f"{label}\\n[{node.state.value}]"

            attr_str = ", ".join(f'{k}="{v}"' for k, v in attrs.items())
            lines.append(f'    "{node.id}" [{attr_str}, label="{label}"];')

        lines.append("")

        # 定义边
        for edge in dag.get_edges():
            lines.append(f'    "{edge.from_node}" -> "{edge.to_node}";')

        lines.append("}")

        return "\n".join(lines)

    def _get_node_attrs(self, node: DAGNode, highlighted: bool) -> Dict[str, str]:
        """获取节点属性"""
        attrs = {}

        if highlighted:
            attrs["fillcolor"] = "#ff6b6b"
            attrs["penwidth"] = "3"
        else:
            state_colors = {
                NodeState.PENDING: "#f0f0f0",
                NodeState.READY: "#fff3cd",
                NodeState.RUNNING: "#cce5ff",
                NodeState.SUCCESS: "#d4edda",
                NodeState.FAILED: "#f8d7da",
                NodeState.SKIPPED: "#e2e3e5",
                NodeState.CANCELLED: "#f5c6cb",
            }
            attrs["fillcolor"] = state_colors.get(node.state, "#ffffff")

        # 根据节点类型设置形状
        shapes = {
            "task": "box",
            "llm": "ellipse",
            "tool": "cylinder",
            "condition": "diamond",
            "loop": "doublecircle",
            "parallel": "component",
            "human_approval": "note",
            "delay": "invhouse",
        }
        attrs["shape"] = shapes.get(node.node_type, "box")

        return attrs


class ProgressRenderer:
    """
    进度渲染器

    渲染工作流执行进度。

    Usage:
        renderer = ProgressRenderer()
        progress_bar = renderer.render_progress(dag)
    """

    def __init__(self, width: int = 50):
        self._width = width

    def render_progress(self, dag: DAGEngine) -> str:
        """
        渲染进度条

        Args:
            dag: DAG引擎实例

        Returns:
            进度条字符串
        """
        nodes = dag.get_all_nodes()
        if not nodes:
            return "[空工作流]"

        total = len(nodes)
        completed = sum(1 for n in nodes if n.state == NodeState.SUCCESS)
        failed = sum(1 for n in nodes if n.state == NodeState.FAILED)
        running = sum(1 for n in nodes if n.state == NodeState.RUNNING)

        filled = int(self._width * completed / total)
        failed_len = int(self._width * failed / total)
        running_len = int(self._width * running / total)

        bar = (
            "=" * filled +
            ">" * running_len +
            "!" * failed_len +
            "-" * (self._width - filled - failed_len - running_len)
        )

        percentage = 100 * completed / total

        return f"[{bar}] {percentage:.1f}% ({completed}/{total})"

    def render_detailed_progress(self, dag: DAGEngine) -> str:
        """
        渲染详细进度

        Args:
            dag: DAG引擎实例

        Returns:
            详细进度字符串
        """
        nodes = dag.get_all_nodes()
        if not nodes:
            return "空工作流"

        lines = [self.render_progress(dag), ""]

        state_counts: Dict[NodeState, int] = {state: 0 for state in NodeState}
        for node in nodes:
            state_counts[node.state] += 1

        lines.append("状态统计:")
        for state, count in state_counts.items():
            if count > 0:
                lines.append(f"  {state.value}: {count}")

        return "\n".join(lines)

    def render_node_status(self, dag: DAGEngine) -> str:
        """
        渲染节点状态列表

        Args:
            dag: DAG引擎实例

        Returns:
            节点状态字符串
        """
        nodes = dag.get_all_nodes()
        if not nodes:
            return "无节点"

        lines = ["节点状态:"]
        for node in sorted(nodes, key=lambda n: n.id):
            icon = self._get_state_icon(node.state)
            lines.append(f"  {icon} {node.name or node.id} [{node.state.value}]")

        return "\n".join(lines)

    def _get_state_icon(self, state: NodeState) -> str:
        """获取状态图标"""
        icons = {
            NodeState.PENDING: "○",
            NodeState.READY: "◐",
            NodeState.RUNNING: "◑",
            NodeState.SUCCESS: "●",
            NodeState.FAILED: "✗",
            NodeState.SKIPPED: "⊘",
            NodeState.CANCELLED: "⊗",
        }
        return icons.get(state, "?")


class ASCIIVisualizer:
    """
    ASCII艺术可视化器

    使用ASCII字符绘制工作流图。

    Usage:
        visualizer = ASCIIVisualizer()
        ascii_art = visualizer.visualize(dag)
    """

    def __init__(self):
        self._char_width = 15
        self._char_height = 3

    def visualize(self, dag: DAGEngine) -> str:
        """
        可视化工作流

        Args:
            dag: DAG引擎实例

        Returns:
            ASCII艺术字符串
        """
        try:
            layers = dag.get_execution_order()
        except Exception:
            return "无法可视化：工作流包含循环"

        if not layers:
            return "空工作流"

        lines = []

        # 绘制每层
        for level_idx, layer in enumerate(layers):
            # 节点行
            node_line = ""
            for node_id in layer:
                node = dag.get_node(node_id)
                label = (node.name or node_id)[:self._char_width - 2]
                padding = (self._char_width - len(label) - 2) // 2
                node_str = " " * padding + f"[{label}]" + " " * padding
                node_line += node_str + "  "
            lines.append(node_line)

            # 如果不是最后一层，绘制连接线
            if level_idx < len(layers) - 1:
                lines.append(self._draw_connections(layer, layers[level_idx + 1], dag))

        return "\n".join(lines)

    def _draw_connections(self, from_layer: List[str], to_layer: List[str], dag: DAGEngine) -> str:
        """绘制层间连接"""
        # 简化的连接线
        line = ""
        for _ in from_layer:
            line += " " * (self._char_width // 2) + "|" + " " * (self._char_width // 2 + 2)
        return line

    def visualize_compact(self, dag: DAGEngine) -> str:
        """
        紧凑可视化

        Args:
            dag: DAG引擎实例

        Returns:
            紧凑ASCII字符串
        """
        lines = ["工作流结构:"]

        for edge in dag.get_edges():
            from_node = dag.get_node(edge.from_node)
            to_node = dag.get_node(edge.to_node)
            from_name = from_node.name if from_node else edge.from_node
            to_name = to_node.name if to_node else edge.to_node
            lines.append(f"  {from_name} --> {to_name}")

        return "\n".join(lines)


class HTMLGenerator:
    """
    HTML生成器

    生成交互式HTML可视化。

    Usage:
        generator = HTMLGenerator()
        html = generator.generate(dag, interactive=True)
    """

    def __init__(self):
        self._width = 800
        self._height = 600
        self._interactive = True

    def generate(self, dag: DAGEngine, interactive: bool = True) -> str:
        """
        生成HTML

        Args:
            dag: DAG引擎实例
            interactive: 是否生成交互式HTML

        Returns:
            HTML字符串
        """
        self._interactive = interactive

        # 计算布局
        engine = GraphLayoutEngine(LayoutAlgorithm.DOT)
        layout = engine.layout(dag)

        if interactive:
            return self._generate_interactive_html(dag, layout)
        else:
            return self._generate_static_html(dag, layout)

    def _generate_static_html(self, dag: DAGEngine, layout: GraphLayout) -> str:
        """生成静态HTML"""
        svg = self._generate_svg(dag, layout)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Workflow Visualization</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .workflow-container {{ border: 1px solid #ccc; padding: 20px; }}
        .node-legend {{ margin-top: 20px; }}
        .legend-item {{ display: inline-block; margin-right: 20px; }}
        .legend-color {{ display: inline-block; width: 20px; height: 20px; margin-right: 5px; vertical-align: middle; }}
    </style>
</head>
<body>
    <h1>工作流可视化</h1>
    <div class="workflow-container">
        {svg}
    </div>
    <div class="node-legend">
        <h3>图例</h3>
        <div class="legend-item"><span class="legend-color" style="background:#d4edda;"></span>成功</div>
        <div class="legend-item"><span class="legend-color" style="background:#f8d7da;"></span>失败</div>
        <div class="legend-item"><span class="legend-color" style="background:#cce5ff;"></span>运行中</div>
        <div class="legend-item"><span class="legend-color" style="background:#fff3cd;"></span>就绪</div>
        <div class="legend-item"><span class="legend-color" style="background:#f0f0f0;"></span>等待中</div>
    </div>
</body>
</html>"""

        return html

    def _generate_interactive_html(self, dag: DAGEngine, layout: GraphLayout) -> str:
        """生成交互式HTML"""
        nodes_data = []
        for node in dag.get_all_nodes():
            node_layout = layout.nodes.get(node.id)
            if node_layout:
                nodes_data.append({
                    "id": node.id,
                    "label": node.name or node.id,
                    "x": node_layout.position.x + layout.width / 2,
                    "y": node_layout.position.y + 50,
                    "state": node.state.value,
                    "type": node.node_type,
                })

        edges_data = []
        for edge in dag.get_edges():
            edges_data.append({
                "from": edge.from_node,
                "to": edge.to_node,
            })

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Workflow Visualization</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; }}
        #workflow-container {{
            width: 100%;
            height: 600px;
            border: 1px solid #ccc;
        }}
        .controls {{
            padding: 10px;
            background: #f5f5f5;
            border-bottom: 1px solid #ccc;
        }}
        .controls button {{
            margin-right: 10px;
            padding: 5px 15px;
            cursor: pointer;
        }}
        .info-panel {{
            position: fixed;
            right: 20px;
            top: 100px;
            width: 250px;
            background: white;
            border: 1px solid #ccc;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="controls">
        <button onclick="fitNetwork()">适应窗口</button>
        <button onclick="resetLayout()">重置布局</button>
        <span id="status">点击节点查看详情</span>
    </div>
    <div id="workflow-container"></div>
    <div class="info-panel" id="info-panel">
        <h3>节点信息</h3>
        <div id="node-info">点击节点查看详情</div>
    </div>

    <script>
        const nodes = new vis.DataSet({json.dumps(nodes_data)});
        const edges = new vis.DataSet({json.dumps(edges_data)});

        const container = document.getElementById('workflow-container');
        const data = {{ nodes: nodes, edges: edges }};

        const options = {{
            nodes: {{
                shape: 'box',
                margin: 10,
                font: {{ size: 14 }},
                borderWidth: 2,
                shadow: true
            }},
            edges: {{
                width: 2,
                arrows: {{ to: {{ enabled: true, scaleFactor: 1 }} }},
                smooth: {{ type: 'cubicBezier' }}
            }},
            physics: {{
                enabled: false
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 200
            }}
        }};

        // 根据状态设置颜色
        const stateColors = {{
            'pending': '#f0f0f0',
            'ready': '#fff3cd',
            'running': '#cce5ff',
            'success': '#d4edda',
            'failed': '#f8d7da',
            'skipped': '#e2e3e5',
            'cancelled': '#f5c6cb'
        }};

        nodes.forEach(node => {{
            const color = stateColors[node.state] || '#ffffff';
            nodes.update({{ id: node.id, color: {{ background: color, border: '#333' }} }});
        }});

        const network = new vis.Network(container, data, options);

        network.on('click', function(params) {{
            if (params.nodes.length > 0) {{
                const nodeId = params.nodes[0];
                const node = nodes.get(nodeId);
                document.getElementById('node-info').innerHTML = `
                    <p><strong>ID:</strong> ${{node.id}}</p>
                    <p><strong>名称:</strong> ${{node.label}}</p>
                    <p><strong>类型:</strong> ${{node.type}}</p>
                    <p><strong>状态:</strong> ${{node.state}}</p>
                `;
            }}
        }});

        function fitNetwork() {{
            network.fit();
        }}

        function resetLayout() {{
            network.setData(data);
        }}
    </script>
</body>
</html>"""

        return html

    def _generate_svg(self, dag: DAGEngine, layout: GraphLayout) -> str:
        """生成SVG"""
        margin = 50
        width = layout.width + 2 * margin
        height = layout.height + 2 * margin

        svg_parts = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
            '<defs>',
            '  <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">',
            '    <polygon points="0 0, 10 3.5, 0 7" fill="#666" />',
            '  </marker>',
            '</defs>',
        ]

        # 绘制边
        for edge in layout.edges:
            if len(edge.path) >= 2:
                start = edge.path[0]
                end = edge.path[-1]
                x1 = start.x + width / 2
                y1 = start.y + margin
                x2 = end.x + width / 2
                y2 = end.y + margin

                svg_parts.append(
                    f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                    f'stroke="#666" stroke-width="2" marker-end="url(#arrowhead)" />'
                )

        # 绘制节点
        for node_id, node_layout in layout.nodes.items():
            node = dag.get_node(node_id)
            x = node_layout.position.x + width / 2
            y = node_layout.position.y + margin
            w = node_layout.width
            h = node_layout.height

            color = self._get_node_color(node.state if node else NodeState.PENDING)

            # 矩形
            svg_parts.append(
                f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" '
                f'fill="{color}" stroke="#333" stroke-width="2" rx="5" />'
            )

            # 文本
            label = html.escape(node_layout.label or node_id)
            svg_parts.append(
                f'  <text x="{x + w/2}" y="{y + h/2}" text-anchor="middle" '
                f'dominant-baseline="middle" font-size="12" fill="#333">{label}</text>'
            )

        svg_parts.append('</svg>')

        return "\n".join(svg_parts)

    def _get_node_color(self, state: NodeState) -> str:
        """获取节点颜色"""
        colors = {
            NodeState.PENDING: "#f0f0f0",
            NodeState.READY: "#fff3cd",
            NodeState.RUNNING: "#cce5ff",
            NodeState.SUCCESS: "#d4edda",
            NodeState.FAILED: "#f8d7da",
            NodeState.SKIPPED: "#e2e3e5",
            NodeState.CANCELLED: "#f5c6cb",
        }
        return colors.get(state, "#ffffff")


class WorkflowVisualizer:
    """
    工作流可视化主类

    整合图布局、导出、渲染和生成功能。

    Usage:
        visualizer = WorkflowVisualizer()
        mermaid = visualizer.to_mermaid(dag)
        html = visualizer.to_html(dag)
    """

    def __init__(self):
        self._layout_engine = GraphLayoutEngine()
        self._mermaid_exporter = MermaidExporter()
        self._graphviz_exporter = GraphvizExporter()
        self._progress_renderer = ProgressRenderer()
        self._ascii_visualizer = ASCIIVisualizer()
        self._html_generator = HTMLGenerator()

    def to_mermaid(self, dag: DAGEngine, **kwargs) -> str:
        """导出为Mermaid格式"""
        if "direction" in kwargs:
            self._mermaid_exporter.set_direction(kwargs["direction"])
        return self._mermaid_exporter.export(dag, kwargs.get("highlight_nodes"))

    def to_graphviz(self, dag: DAGEngine, **kwargs) -> str:
        """导出为Graphviz格式"""
        if "rankdir" in kwargs:
            self._graphviz_exporter.set_rankdir(kwargs["rankdir"])
        return self._graphviz_exporter.export(dag, kwargs.get("highlight_nodes"))

    def to_html(self, dag: DAGEngine, interactive: bool = True) -> str:
        """导出为HTML格式"""
        return self._html_generator.generate(dag, interactive)

    def to_svg(self, dag: DAGEngine) -> str:
        """导出为SVG格式"""
        layout = self._layout_engine.layout(dag)
        return self._html_generator._generate_svg(dag, layout)

    def to_ascii(self, dag: DAGEngine, compact: bool = False) -> str:
        """导出为ASCII格式"""
        if compact:
            return self._ascii_visualizer.visualize_compact(dag)
        return self._ascii_visualizer.visualize(dag)

    def render_progress(self, dag: DAGEngine, detailed: bool = False) -> str:
        """渲染进度"""
        if detailed:
            return self._progress_renderer.render_detailed_progress(dag)
        return self._progress_renderer.render_progress(dag)

    def render_node_status(self, dag: DAGEngine) -> str:
        """渲染节点状态"""
        return self._progress_renderer.render_node_status(dag)

    def get_layout(self, dag: DAGEngine, algorithm: LayoutAlgorithm = LayoutAlgorithm.DOT) -> GraphLayout:
        """获取图布局"""
        engine = GraphLayoutEngine(algorithm)
        return engine.layout(dag)

    def export(
        self,
        dag: DAGEngine,
        format: ExportFormat,
        filepath: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        导出工作流

        Args:
            dag: DAG引擎实例
            format: 导出格式
            filepath: 文件路径，None则返回字符串
            **kwargs: 额外参数

        Returns:
            如果filepath为None，返回导出内容
        """
        if format == ExportFormat.MERMAID:
            content = self.to_mermaid(dag, **kwargs)
        elif format == ExportFormat.GRAPHVIZ:
            content = self.to_graphviz(dag, **kwargs)
        elif format == ExportFormat.HTML:
            content = self.to_html(dag, kwargs.get("interactive", True))
        elif format == ExportFormat.SVG:
            content = self.to_svg(dag)
        elif format == ExportFormat.JSON:
            content = json.dumps(dag.to_dict(), indent=2, ensure_ascii=False)
        else:
            raise VisualizationError(f"不支持的导出格式: {format}")

        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            return None

        return content
