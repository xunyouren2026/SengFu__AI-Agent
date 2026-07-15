"""
联盟结构可视化

生成Agent协作关系图（ASCII/SVG格式）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum, auto


class GraphFormat(Enum):
    """图形格式"""
    ASCII = auto()
    SVG = auto()
    DOT = auto()  # GraphViz格式
    JSON = auto()


@dataclass
class Node:
    """图节点"""
    node_id: str
    label: str = ""
    node_type: str = "default"
    x: float = 0.0
    y: float = 0.0
    color: str = "#3498db"
    size: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """图边"""
    from_node: str
    to_node: str
    label: str = ""
    weight: float = 1.0
    color: str = "#95a5a6"
    directed: bool = True


@dataclass
class Graph:
    """图结构"""
    graph_id: str
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    
    def add_node(self, node: Node) -> None:
        """添加节点"""
        self.nodes[node.node_id] = node
    
    def add_edge(self, edge: Edge) -> None:
        """添加边"""
        self.edges.append(edge)
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """获取邻居节点"""
        neighbors = []
        for edge in self.edges:
            if edge.from_node == node_id:
                neighbors.append(edge.to_node)
            elif not edge.directed and edge.to_node == node_id:
                neighbors.append(edge.from_node)
        return neighbors


class CoalitionGraphBuilder:
    """联盟图构建器"""
    
    def __init__(self):
        self.agent_colors = {
            "leader": "#e74c3c",
            "executor": "#3498db",
            "verifier": "#2ecc71",
            "coordinator": "#f39c12",
            "default": "#95a5a6"
        }
    
    def build_from_coalition(
        self,
        coalition_id: str,
        agents: List[str],
        tasks: Dict[str, str],  # task_id -> agent_id
        roles: Optional[Dict[str, str]] = None  # agent_id -> role
    ) -> Graph:
        """
        从联盟构建图
        
        Args:
            coalition_id: 联盟ID
            agents: Agent列表
            tasks: 任务分配 {task_id: agent_id}
            roles: Agent角色 {agent_id: role}
        """
        graph = Graph(graph_id=f"coalition_{coalition_id}")
        
        # 添加Agent节点
        for agent_id in agents:
            role = roles.get(agent_id, "default") if roles else "default"
            color = self.agent_colors.get(role, self.agent_colors["default"])
            
            node = Node(
                node_id=agent_id,
                label=agent_id,
                node_type="agent",
                color=color
            )
            graph.add_node(node)
        
        # 添加任务节点
        for task_id, agent_id in tasks.items():
            task_node = Node(
                node_id=f"task_{task_id}",
                label=task_id,
                node_type="task",
                color="#9b59b6",
                size=0.8
            )
            graph.add_node(task_node)
            
            # 添加任务-Agent边
            if agent_id in agents:
                edge = Edge(
                    from_node=f"task_{task_id}",
                    to_node=agent_id,
                    label="assigned_to",
                    color="#34495e"
                )
                graph.add_edge(edge)
        
        # 添加Agent间的协作边（基于共同任务）
        agent_tasks: Dict[str, List[str]] = {aid: [] for aid in agents}
        for task_id, agent_id in tasks.items():
            if agent_id in agent_tasks:
                agent_tasks[agent_id].append(task_id)
        
        # 如果有多个Agent，添加协作边
        agent_list = list(agents)
        for i in range(len(agent_list)):
            for j in range(i + 1, len(agent_list)):
                aid1, aid2 = agent_list[i], agent_list[j]
                # 检查是否有共同任务类型
                common_tasks = set(agent_tasks[aid1]) & set(agent_tasks[aid2])
                if common_tasks:
                    edge = Edge(
                        from_node=aid1,
                        to_node=aid2,
                        label="collaborates",
                        weight=len(common_tasks),
                        color="#bdc3c7",
                        directed=False
                    )
                    graph.add_edge(edge)
        
        return graph
    
    def build_from_assignments(
        self,
        assignments: Dict[str, str],  # task_id -> agent_id
        agent_roles: Optional[Dict[str, str]] = None
    ) -> Graph:
        """从分配方案构建图"""
        agents = list(set(assignments.values()))
        return self.build_from_coalition(
            coalition_id="main",
            agents=agents,
            tasks=assignments,
            roles=agent_roles
        )


class ASCIIRenderer:
    """ASCII图形渲染器"""
    
    def render(self, graph: Graph, width: int = 60, height: int = 20) -> str:
        """
        渲染为ASCII图形
        
        简化的网格布局渲染
        """
        lines = []
        lines.append(f"Graph: {graph.graph_id}")
        lines.append("=" * width)
        
        # 简单的节点布局
        nodes = list(graph.nodes.values())
        if not nodes:
            lines.append("(Empty graph)")
            return "\n".join(lines)
        
        # 按类型分组
        agents = [n for n in nodes if n.node_type == "agent"]
        tasks = [n for n in nodes if n.node_type == "task"]
        
        # 绘制Agent
        if agents:
            lines.append("\n[Agents]")
            for agent in agents:
                role = ""
                for color, r in [("#e74c3c", "[L]"), ("#3498db", "[E]"), 
                                 ("#2ecc71", "[V]"), ("#f39c12", "[C]")]:
                    if agent.color == color:
                        role = r
                        break
                lines.append(f"  {role} {agent.label}")
        
        # 绘制任务
        if tasks:
            lines.append("\n[Tasks]")
            for task in tasks:
                # 找到分配的Agent
                assigned_agent = None
                for edge in graph.edges:
                    if edge.from_node == task.node_id:
                        assigned_agent = edge.to_node
                        break
                
                agent_label = f" -> {assigned_agent}" if assigned_agent else ""
                lines.append(f"  [T] {task.label}{agent_label}")
        
        # 绘制协作关系
        collab_edges = [e for e in graph.edges if e.label == "collaborates"]
        if collab_edges:
            lines.append("\n[Collaborations]")
            for edge in collab_edges:
                lines.append(f"  {edge.from_node} <-> {edge.to_node}")
        
        lines.append("=" * width)
        
        return "\n".join(lines)


class SVGRenderer:
    """SVG图形渲染器"""
    
    def render(self, graph: Graph, width: int = 800, height: int = 600) -> str:
        """渲染为SVG"""
        # 计算节点位置（简单的圆形布局）
        self._calculate_layout(graph, width, height)
        
        svg_parts = []
        svg_parts.append(f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">')
        
        # 背景
        svg_parts.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
        
        # 绘制边
        for edge in graph.edges:
            from_node = graph.nodes.get(edge.from_node)
            to_node = graph.nodes.get(edge.to_node)
            
            if from_node and to_node:
                svg_parts.append(
                    f'<line x1="{from_node.x}" y1="{from_node.y}" '
                    f'x2="{to_node.x}" y2="{to_node.y}" '
                    f'stroke="{edge.color}" stroke-width="{edge.weight}"/>'
                )
        
        # 绘制节点
        for node in graph.nodes.values():
            radius = 20 * node.size
            
            # 节点圆形
            svg_parts.append(
                f'<circle cx="{node.x}" cy="{node.y}" r="{radius}" '
                f'fill="{node.color}" stroke="#2c3e50" stroke-width="2"/>'
            )
            
            # 节点标签
            svg_parts.append(
                f'<text x="{node.x}" y="{node.y + 5}" '
                f'text-anchor="middle" font-size="12" fill="#ffffff">'
                f'{node.label[:10]}</text>'
            )
        
        svg_parts.append('</svg>')
        
        return "\n".join(svg_parts)
    
    def _calculate_layout(self, graph: Graph, width: int, height: int) -> None:
        """计算节点布局（圆形布局）"""
        nodes = list(graph.nodes.values())
        if not nodes:
            return
        
        center_x = width / 2
        center_y = height / 2
        radius = min(width, height) * 0.35
        
        # Agent和任务分别布局
        agents = [n for n in nodes if n.node_type == "agent"]
        tasks = [n for n in nodes if n.node_type == "task"]
        
        # Agent在外圈
        for i, node in enumerate(agents):
            angle = 2 * 3.14159 * i / len(agents) if agents else 0
            node.x = center_x + radius * __import__('math').cos(angle)
            node.y = center_y + radius * __import__('math').sin(angle)
        
        # 任务在内圈
        inner_radius = radius * 0.5
        for i, node in enumerate(tasks):
            angle = 2 * 3.14159 * i / len(tasks) if tasks else 0
            node.x = center_x + inner_radius * __import__('math').cos(angle)
            node.y = center_y + inner_radius * __import__('math').sin(angle)


class DOTRenderer:
    """GraphViz DOT格式渲染器"""
    
    def render(self, graph: Graph) -> str:
        """渲染为DOT格式"""
        lines = []
        lines.append(f"digraph {graph.graph_id} {{")
        lines.append("  rankdir=TB;")
        lines.append("  node [shape=circle, style=filled];")
        
        # 定义节点
        for node in graph.nodes.values():
            shape = "box" if node.node_type == "task" else "circle"
            lines.append(
                f'  "{node.node_id}" [label="{node.label}", '
                f'shape={shape}, fillcolor="{node.color}"];'
            )
        
        # 定义边
        for edge in graph.edges:
            attrs = []
            if edge.color:
                attrs.append(f'color="{edge.color}"')
            if edge.weight != 1.0:
                attrs.append(f'penwidth={edge.weight}')
            if edge.label:
                attrs.append(f'label="{edge.label}"')
            
            attr_str = f' [{", ".join(attrs)}]' if attrs else ""
            
            if edge.directed:
                lines.append(f'  "{edge.from_node}" -> "{edge.to_node}"{attr_str};')
            else:
                lines.append(f'  "{edge.from_node}" -- "{edge.to_node}"{attr_str};')
        
        lines.append("}")
        
        return "\n".join(lines)


class CoalitionVisualizer:
    """联盟可视化器"""
    
    def __init__(self):
        self.graph_builder = CoalitionGraphBuilder()
        self.renderers = {
            GraphFormat.ASCII: ASCIIRenderer(),
            GraphFormat.SVG: SVGRenderer(),
            GraphFormat.DOT: DOTRenderer(),
        }
    
    def visualize(
        self,
        graph: Graph,
        format: GraphFormat = GraphFormat.ASCII,
        **kwargs
    ) -> str:
        """
        可视化图
        
        Args:
            graph: 图结构
            format: 输出格式
            **kwargs: 渲染器特定参数
        """
        renderer = self.renderers.get(format)
        if not renderer:
            raise ValueError(f"Unsupported format: {format}")
        
        return renderer.render(graph, **kwargs)
    
    def visualize_coalition(
        self,
        coalition_id: str,
        agents: List[str],
        tasks: Dict[str, str],
        roles: Optional[Dict[str, str]] = None,
        format: GraphFormat = GraphFormat.ASCII
    ) -> str:
        """可视化联盟"""
        graph = self.graph_builder.build_from_coalition(
            coalition_id, agents, tasks, roles
        )
        return self.visualize(graph, format)
