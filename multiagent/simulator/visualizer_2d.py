"""
2D仿真可视化 - 实时展示Agent位置与状态
"""
from __future__ import annotations
import sys
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .world import Agent, World, AgentState


@dataclass
class ViewConfig:
    """可视化配置"""
    width: int = 80
    height: int = 30
    agent_char: str = "@"
    empty_char: str = "."
    boundary_char: str = "#"
    show_grid: bool = False
    show_agents: bool = True
    show_stats: bool = True
    color_enabled: bool = False


class ASCIIVisualizer:
    """ASCII字符可视化器"""

    def __init__(self, world: World, config: Optional[ViewConfig] = None):
        self.world = world
        self.config = config or ViewConfig()
        self.color_codes = {
            "reset": "\033[0m",
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "magenta": "\033[95m",
            "cyan": "\033[96m",
            "white": "\033[97m"
        }

    def _get_agent_char(self, agent: Agent) -> str:
        """根据Agent状态获取显示字符"""
        if agent.state == AgentState.INACTIVE:
            return "x"
        elif agent.state == AgentState.TERMINATED:
            return "X"
        energy = agent.energy
        if energy > 80:
            return self.config.agent_char.upper()
        elif energy > 50:
            return self.config.agent_char
        else:
            return self.config.agent_char.lower()

    def _get_agent_color(self, agent: Agent) -> str:
        """获取Agent颜色"""
        if not self.config.color_enabled:
            return self.color_codes["reset"]
        if agent.state == AgentState.INACTIVE:
            return self.color_codes["yellow"]
        elif agent.state == AgentState.TERMINATED:
            return self.color_codes["red"]
        energy = agent.energy
        if energy > 80:
            return self.color_codes["green"]
        elif energy > 50:
            return self.color_codes["cyan"]
        else:
            return self.color_codes["yellow"]

    def render(self) -> str:
        """渲染当前状态为ASCII字符串"""
        lines = []
        world_w = int(self.world.width)
        world_h = int(self.world.height)
        cell_w = world_w / self.config.width
        cell_h = world_h / self.config.height

        lines.append(self._render_header())
        lines.append(self._render_boundary())

        for y in range(self.config.height - 1, -1, -1):
            row = self.config.boundary_char
            for x in range(self.config.width):
                world_x = x * cell_w + cell_w / 2
                world_y = y * cell_h + cell_h / 2

                # 检查是否有Agent
                agent_at_pos = self._get_agent_at(world_x, world_y, cell_w, cell_h)
                if agent_at_pos:
                    char = self._get_agent_char(agent_at_pos)
                    color = self._get_agent_color(agent_at_pos)
                    if self.config.color_enabled:
                        row += f"{color}{char}{self.color_codes['reset']}"
                    else:
                        row += char
                else:
                    if self.config.show_grid:
                        row += "+" if (x + y) % 2 == 0 else "-"
                    else:
                        row += self.config.empty_char

            row += self.config.boundary_char
            lines.append(row)

        lines.append(self._render_boundary())

        if self.config.show_stats:
            lines.extend(self._render_stats())

        return "\n".join(lines)

    def _get_agent_at(self, x: float, y: float, cell_w: float, cell_h: float) -> Optional[Agent]:
        """获取指定位置的Agent"""
        candidates = self.world.get_agents_in_region(
            x - cell_w/2, x + cell_w/2,
            y - cell_h/2, y + cell_h/2
        )
        if candidates:
            return candidates[0]
        return None

    def _render_header(self) -> str:
        """渲染标题栏"""
        title = f" Multi-Agent Simulation - Step {self.world.current_step} "
        padding = (self.config.width - len(title)) // 2
        return " " * padding + title

    def _render_boundary(self) -> str:
        """渲染边界"""
        return self.config.boundary_char * (self.config.width + 2)

    def _render_stats(self) -> List[str]:
        """渲染统计信息"""
        stats = self.world.get_statistics()
        lines = []
        lines.append("")

        active = stats.get("active_agents", 0)
        terminated = stats.get("terminated_agents", 0)
        lines.append(f" Agents: {active} active, {terminated} terminated ")
        lines.append(f" Time: {self.world.current_time:.2f} | Events: {stats.get('total_events', 0)} ")

        # 显示能量分布
        energies = [a.energy for a in self.world.get_all_agents()]
        if energies:
            avg_energy = sum(energies) / len(energies)
            high_energy = sum(1 for e in energies if e > 80)
            med_energy = sum(1 for e in energies if 50 < e <= 80)
            low_energy = sum(1 for e in energies if e <= 50)
            lines.append(f" Energy: avg={avg_energy:.1f} high={high_energy} med={med_energy} low={low_energy} ")

        return lines

    def get_frame(self) -> str:
        """获取一帧"""
        return self.render()


class SVGVisualizer:
    """SVG矢量可视化"""

    def __init__(self, world: World, width: int = 800, height: int = 600):
        self.world = world
        self.svg_width = width
        self.svg_height = height
        self.scale_x = width / world.width
        self.scale_y = height / world.height

    def _to_svg_coords(self, x: float, y: float) -> Tuple[float, float]:
        """转换为SVG坐标"""
        svg_x = x * self.scale_x
        svg_y = self.svg_height - (y * self.scale_y)
        return (svg_x, svg_y)

    def render(self) -> str:
        """渲染为SVG"""
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.svg_width}" height="{self.svg_height}">',
            f'<rect width="100%" height="100%" fill="#1a1a2e"/>',
            self._render_grid(),
            self._render_agents(),
            self._render_info_panel()
        ]

        # 关闭svg标签
        svg_parts.append('</svg>')
        return '\n'.join(svg_parts)

    def _render_grid(self) -> str:
        """渲染网格"""
        lines = ['<g stroke="#333" stroke-width="0.5">']
        grid_size = 20

        for x in range(0, int(self.world.width) + 1, grid_size):
            svg_x, _ = self._to_svg_coords(x, 0)
            lines.append(f'<line x1="{svg_x}" y1="0" x2="{svg_x}" y2="{self.svg_height}"/>')

        for y in range(0, int(self.world.height) + 1, grid_size):
            _, svg_y = self._to_svg_coords(0, y)
            lines.append(f'<line x1="0" y1="{svg_y}" x2="{self.svg_width}" y2="{svg_y}"/>')

        lines.append('</g>')
        return '\n'.join(lines)

    def _render_agents(self) -> str:
        """渲染Agent"""
        circles = ['<g id="agents">']

        for agent in self.world.get_all_agents():
            svg_x, svg_y = self._to_svg_coords(agent.position.x, agent.position.y)

            # 根据能量确定颜色
            if agent.energy > 80:
                color = "#4ade80"  # 绿色
            elif agent.energy > 50:
                color = "#facc15"  # 黄色
            else:
                color = "#ef4444"  # 红色

            # 大小根据能量
            radius = 3 + (agent.energy / 100) * 3

            circles.append(
                f'<circle cx="{svg_x}" cy="{svg_y}" r="{radius}" fill="{color}" '
                f'stroke="#fff" stroke-width="0.5" opacity="0.9"/>'
            )

            # 添加ID标签（仅在空间足够时）
            if len(self.world.get_all_agents()) < 20:
                circles.append(
                    f'<text x="{svg_x}" y="{svg_y - radius - 2}" fill="#fff" '
                    f'font-size="8" text-anchor="middle">{agent.agent_id[:4]}</text>'
                )

        circles.append('</g>')
        return '\n'.join(circles)

    def _render_info_panel(self) -> str:
        """渲染信息面板"""
        stats = self.world.get_statistics()
        lines = [
            '<g id="info-panel" transform="translate(10, 20)">',
            '<rect x="0" y="0" width="150" height="80" fill="#333" rx="5" opacity="0.8"/>'
        ]

        texts = [
            f'Step: {stats.get("total_steps", 0)}',
            f'Time: {self.world.current_time:.1f}',
            f'Agents: {stats.get("active_agents", 0)}'
        ]

        for i, text in enumerate(texts):
            lines.append(
                f'<text x="10" y="{20 + i * 20}" fill="#fff" font-family="monospace" font-size="12">{text}</text>'
            )

        lines.append('</g>')
        return '\n'.join(lines)

    def save_svg(self, filepath: str) -> None:
        """保存SVG到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.render())


class AnimationController:
    """动画控制器"""

    def __init__(self, world: World, visualizer: Optional[ASCIIVisualizer] = None):
        self.world = world
        self.visualizer = visualizer or ASCIIVisualizer(world)
        self.frames: List[str] = []
        self.max_frames: int = 1000

    def capture_frame(self) -> None:
        """捕获当前帧"""
        frame = self.visualizer.get_frame()
        self.frames.append(frame)
        if len(self.frames) > self.max_frames:
            self.frames.pop(0)

    def play(self, delay: float = 0.1) -> None:
        """播放动画"""
        try:
            import time
            for frame in self.frames:
                print("\033[2J\033[H", end="")  # 清屏
                print(frame)
                time.sleep(delay)
        except KeyboardInterrupt:
            pass

    def save_animation(self, filepath: str, format: str = "text") -> None:
        """保存动画"""
        if format == "text":
            with open(filepath, 'w', encoding='utf-8') as f:
                for i, frame in enumerate(self.frames):
                    f.write(f"=== Frame {i + 1} ===\n")
                    f.write(frame)
                    f.write("\n\n")
        elif format == "html":
            self._save_as_html(filepath)

    def _save_as_html(self, filepath: str) -> None:
        """保存为HTML动画"""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Multi-Agent Simulation</title>
    <style>
        body { background: #1a1a2e; color: #eee; font-family: monospace; padding: 20px; }
        pre { font-size: 10px; line-height: 1.2; }
        #controls { margin: 10px 0; }
        button { padding: 5px 15px; margin-right: 10px; }
    </style>
</head>
<body>
    <h2>Multi-Agent Simulation</h2>
    <div id="controls">
        <button onclick="play()">Play</button>
        <button onclick="pause()">Pause</button>
        <button onclick="reset()">Reset</button>
        Speed: <input type="range" id="speed" min="1" max="100" value="50">
    </div>
    <pre id="display"></pre>
    <script>
        const frames = """
        frames_json = []
        for frame in self.frames:
            frames_json.append(frame.replace('\\', '\\\\').replace('"', '\\"').replace('\\n', '\\\\n'))
        separator = '\\\\n'
        html += f'["{separator.join(frames_json)}"];'
        html += """
        let current = 0;
        let playing = false;
        let speed = 50;

        function show() {
            document.getElementById('display').textContent = frames[current];
        }

        function play() {
            playing = true;
            step();
        }

        function pause() {
            playing = false;
        }

        function reset() {
            current = 0;
            show();
        }

        function step() {
            if (playing && current < frames.length - 1) {
                current++;
                show();
                setTimeout(step, 200 - speed);
            }
        }

        document.getElementById('speed').addEventListener('input', function(e) {
            speed = parseInt(e.target.value);
        });

        show();
    </script>
</body>
</html>"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)


def create_visualizer(world: World, mode: str = "ascii", **kwargs) -> Any:
    """创建可视化器"""
    if mode == "ascii":
        config = ViewConfig(**kwargs) if kwargs else None
        return ASCIIVisualizer(world, config)
    elif mode == "svg":
        width = kwargs.get("width", 800)
        height = kwargs.get("height", 600)
        return SVGVisualizer(world, width, height)
    else:
        raise ValueError(f"Unknown mode: {mode}")
