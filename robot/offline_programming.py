#!/usr/bin/env python3
"""
离线编程模块
支持生成机器人程序（URScript、G代码等），模拟运行验证
"""

import time
from typing import List, Tuple, Dict, Any, Optional, Callable


class OfflineProgrammer:
    """离线编程器"""
    
    def __init__(self, robot=None, output_format: str = "urscript"):
        """
        robot: 机器人控制器
        output_format: 输出格式 ("urscript", "gcode", "python")
        """
        self.robot = robot
        self.format = output_format
        self.instructions = []  # 存储生成的指令
    
    def add_move_joint(self, joint_positions: List[float], velocity: float = 0.5, acceleration: float = 0.5):
        """添加关节运动指令"""
        self.instructions.append(("move_joint", joint_positions, velocity, acceleration))
    
    def add_move_cartesian(self, pose: Tuple[float, ...], velocity: float = 0.2, acceleration: float = 0.2):
        """添加笛卡尔运动指令"""
        self.instructions.append(("move_cartesian", pose, velocity, acceleration))
    
    def add_set_digital_out(self, pin: int, value: bool):
        """添加数字输出指令"""
        self.instructions.append(("set_digital_out", pin, value))
    
    def add_delay(self, seconds: float):
        """添加延时指令"""
        self.instructions.append(("delay", seconds))
    
    def generate(self) -> str:
        """生成机器人程序代码"""
        if self.format == "urscript":
            return self._generate_urscript()
        elif self.format == "gcode":
            return self._generate_gcode()
        elif self.format == "python":
            return self._generate_python()
        else:
            raise ValueError(f"Unknown format: {self.format}")
    
    def _generate_urscript(self) -> str:
        lines = ["def main():", "  textmsg(\"Starting offline program\")"]
        for instr in self.instructions:
            if instr[0] == "move_joint":
                _, joints, vel, acc = instr
                lines.append(f"  movej({joints}, a={acc}, v={vel})")
            elif instr[0] == "move_cartesian":
                _, pose, vel, acc = instr
                lines.append(f"  movel(p{pose}, a={acc}, v={vel})")
            elif instr[0] == "set_digital_out":
                _, pin, value = instr
                lines.append(f"  set_tool_digital_out({pin}, {1 if value else 0})")
            elif instr[0] == "delay":
                _, sec = instr
                lines.append(f"  sleep({sec})")
        lines.append("end")
        return "\n".join(lines)
    
    def _generate_gcode(self) -> str:
        lines = ["G90", "G21", "F500"]
        for instr in self.instructions:
            if instr[0] == "move_cartesian":
                _, pose, vel, _ = instr
                x, y, z, rx, ry, rz = pose[:6]
                lines.append(f"G01 X{x:.3f} Y{y:.3f} Z{z:.3f} A{rx:.3f} B{ry:.3f} C{rz:.3f} F{vel*1000:.0f}")
            elif instr[0] == "delay":
                _, sec = instr
                lines.append(f"G04 P{sec*1000:.0f}")
        lines.append("M30")
        return "\n".join(lines)
    
    def _generate_python(self) -> str:
        lines = ["# Generated offline program for robot control", "import time", "def run(robot):"]
        for instr in self.instructions:
            if instr[0] == "move_joint":
                _, joints, vel, acc = instr
                lines.append(f"    robot.move_joint({joints}, velocity={vel}, acceleration={acc})")
            elif instr[0] == "move_cartesian":
                _, pose, vel, acc = instr
                lines.append(f"    robot.move_cartesian({pose}, velocity={vel}, acceleration={acc})")
            elif instr[0] == "set_digital_out":
                _, pin, value = instr
                lines.append(f"    robot.set_digital_out({pin}, {value})")
            elif instr[0] == "delay":
                _, sec = instr
                lines.append(f"    time.sleep({sec})")
        return "\n".join(lines)
    
    def simulate(self, step_callback: Optional[Callable] = None) -> bool:
        """模拟运行（不实际移动机器人）"""
        print("Starting offline simulation...")
        for instr in self.instructions:
            if step_callback:
                step_callback(instr)
            if instr[0] == "delay":
                time.sleep(min(instr[1], 0.1))  # 模拟时不等待太久
            print(f"Sim: {instr}")
        print("Simulation completed")
        return True
    
    def save_to_file(self, filepath: str):
        """保存生成的程序到文件"""
        with open(filepath, 'w') as f:
            f.write(self.generate())
        print(f"Program saved to {filepath}")
    
    def load_from_file(self, filepath: str):
        """从文件加载程序（解析）"""
        # 简化实现，实际需解析URScript/G代码
        print(f"Loading program from {filepath}")
