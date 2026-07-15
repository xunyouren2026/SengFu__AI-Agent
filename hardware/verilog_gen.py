"""
Verilog代码生成器模块

提供自动生成Verilog HDL代码的功能，支持模块定义、组合逻辑、
时序逻辑、有限状态机(FSM)和测试平台生成。
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import textwrap
import re


class SignalDirection(Enum):
    """信号方向枚举"""
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"


class SignalType(Enum):
    """信号类型枚举"""
    WIRE = "wire"
    REG = "reg"
    LOGIC = "logic"  # SystemVerilog
    INTEGER = "integer"


class DataWidth(Enum):
    """数据宽度枚举"""
    BIT = 1
    BYTE = 8
    HALF_WORD = 16
    WORD = 32
    DOUBLE_WORD = 64


@dataclass
class Signal:
    """
    Verilog信号定义
    
    表示模块的端口或内部信号。
    """
    name: str
    direction: Optional[SignalDirection] = None  # None表示内部信号
    width: int = 1
    signal_type: SignalType = SignalType.WIRE
    signed: bool = False
    default_value: Optional[Union[int, str]] = None
    description: str = ""
    
    def __post_init__(self):
        if self.width < 1:
            raise ValueError("信号宽度必须大于0")
    
    def to_verilog(self, system_verilog: bool = False) -> str:
        """生成Verilog信号声明"""
        lines = []
        
        # 方向声明
        if self.direction:
            dir_str = self.direction.value
        else:
            dir_str = ""
        
        # 类型声明
        if system_verilog and self.signal_type == SignalType.REG:
            type_str = "logic"
        else:
            type_str = self.signal_type.value
        
        # 宽度声明
        if self.width > 1:
            width_str = f"[{self.width-1}:0]"
        else:
            width_str = ""
        
        # 有符号声明
        signed_str = "signed" if self.signed else ""
        
        # 组合声明
        parts = [p for p in [dir_str, type_str, signed_str, width_str, self.name] if p]
        declaration = " ".join(parts)
        
        # 默认值
        if self.default_value is not None:
            declaration += f" = {self.default_value}"
        
        declaration += ";"
        
        # 添加注释
        if self.description:
            declaration += f" // {self.description}"
        
        return declaration
    
    def get_slice(self, high: int, low: int) -> str:
        """获取信号切片"""
        return f"{self.name}[{high}:{low}]"
    
    def get_bit(self, index: int) -> str:
        """获取单个位"""
        return f"{self.name}[{index}]"


@dataclass
class Parameter:
    """
    Verilog参数定义
    
    用于模块的参数化设计。
    """
    name: str
    value: Union[int, str]
    description: str = ""
    
    def to_verilog(self) -> str:
        """生成参数声明"""
        if isinstance(self.value, str) and not self.value.isdigit():
            value_str = f'"{self.value}"'
        else:
            value_str = str(self.value)
        
        result = f"parameter {self.name} = {value_str};"
        if self.description:
            result += f" // {self.description}"
        return result


@dataclass
class LocalParam:
    """本地参数定义"""
    name: str
    value: Union[int, str]
    description: str = ""
    
    def to_verilog(self) -> str:
        """生成本地参数声明"""
        if isinstance(self.value, str) and not self.value.isdigit():
            value_str = f'"{self.value}"'
        else:
            value_str = str(self.value)
        
        result = f"localparam {self.name} = {value_str};"
        if self.description:
            result += f" // {self.description}"
        return result


class VerilogExpression:
    """Verilog表达式生成器"""
    
    @staticmethod
    def and_op(*signals: str) -> str:
        """与操作"""
        return " & ".join(signals)
    
    @staticmethod
    def or_op(*signals: str) -> str:
        """或操作"""
        return " | ".join(signals)
    
    @staticmethod
    def xor_op(*signals: str) -> str:
        """异或操作"""
        return " ^ ".join(signals)
    
    @staticmethod
    def not_op(signal: str) -> str:
        """非操作"""
        return f"~{signal}"
    
    @staticmethod
    def concat(*signals: str) -> str:
        """位拼接"""
        return "{" + ", ".join(signals) + "}"
    
    @staticmethod
    def replicate(count: int, signal: str) -> str:
        """位复制"""
        return f"{{{count}{{{signal}}}}}"
    
    @staticmethod
    def ternary(condition: str, true_val: str, false_val: str) -> str:
        """三目运算符"""
        return f"{condition} ? {true_val} : {false_val}"
    
    @staticmethod
    def assign(target: str, expression: str) -> str:
        """连续赋值"""
        return f"assign {target} = {expression};"


class CombinationalLogic:
    """组合逻辑生成器"""
    
    def __init__(self):
        self.assignments: List[Tuple[str, str]] = []
        self.always_blocks: List[str] = []
    
    def add_assignment(self, target: str, expression: str) -> None:
        """添加连续赋值"""
        self.assignments.append((target, expression))
    
    def add_always_comb(self, statements: List[str], sensitivity: Optional[List[str]] = None) -> None:
        """添加always组合块"""
        if sensitivity:
            sens_str = " or ".join(sensitivity)
            block = f"always @({sens_str}) begin"
        else:
            block = "always @(*) begin"
        
        for stmt in statements:
            block += f"\n    {stmt}"
        block += "\nend"
        
        self.always_blocks.append(block)
    
    def to_verilog(self) -> str:
        """生成Verilog代码"""
        lines = []
        
        # 连续赋值
        for target, expr in self.assignments:
            lines.append(VerilogExpression.assign(target, expr))
        
        if self.assignments:
            lines.append("")
        
        # always块
        for block in self.always_blocks:
            lines.append(block)
            lines.append("")
        
        return "\n".join(lines)


class SequentialLogic:
    """时序逻辑生成器"""
    
    def __init__(self, clock: str, reset: Optional[str] = None, 
                 reset_active_low: bool = True):
        self.clock = clock
        self.reset = reset
        self.reset_active_low = reset_active_low
        self.registers: Dict[str, Dict[str, Any]] = {}
        self.always_blocks: List[str] = []
    
    def add_register(self, name: str, width: int = 1,
                     reset_value: Union[int, str] = 0,
                     description: str = "") -> None:
        """添加寄存器"""
        self.registers[name] = {
            'width': width,
            'reset_value': reset_value,
            'description': description
        }
    
    def add_always_ff(self, statements: List[str]) -> None:
        """添加always时序块"""
        if self.reset:
            if self.reset_active_low:
                reset_cond = f"negedge {self.reset}"
            else:
                reset_cond = f"posedge {self.reset}"
            block = f"always @(posedge {self.clock} or {reset_cond}) begin"
        else:
            block = f"always @(posedge {self.clock}) begin"
        
        # 添加复位逻辑
        if self.reset:
            if self.reset_active_low:
                block += f"\n    if (!{self.reset}) begin"
            else:
                block += f"\n    if ({self.reset}) begin"
            
            for name, info in self.registers.items():
                block += f"\n        {name} <= {info['reset_value']};"
            block += f"\n    end else begin"
            
            for stmt in statements:
                block += f"\n        {stmt}"
            block += f"\n    end"
        else:
            for stmt in statements:
                block += f"\n    {stmt}"
        
        block += "\nend"
        self.always_blocks.append(block)
    
    def to_verilog(self) -> str:
        """生成Verilog代码"""
        return "\n\n".join(self.always_blocks)


@dataclass
class FSMState:
    """FSM状态定义"""
    name: str
    encoding: Optional[int] = None
    description: str = ""
    
    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FSMState):
            return False
        return self.name == other.name


@dataclass
class FSMTransition:
    """FSM状态转换"""
    from_state: FSMState
    to_state: FSMState
    condition: str
    actions: List[str] = field(default_factory=list)


class FSMGenerator:
    """
    有限状态机生成器
    
    生成可综合的Verilog FSM代码。
    """
    
    def __init__(self, name: str, clock: str, reset: str,
                 reset_active_low: bool = True,
                 encoding: str = "binary"):
        self.name = name
        self.clock = clock
        self.reset = reset
        self.reset_active_low = reset_active_low
        self.encoding = encoding  # binary, one_hot, gray
        
        self.states: Dict[str, FSMState] = {}
        self.transitions: List[FSMTransition] = []
        self.state_outputs: Dict[str, List[str]] = {}
        self.default_outputs: List[str] = []
    
    def add_state(self, name: str, encoding: Optional[int] = None,
                  description: str = "") -> FSMState:
        """添加状态"""
        state = FSMState(name, encoding, description)
        self.states[name] = state
        return state
    
    def add_transition(self, from_state: str, to_state: str,
                       condition: str, actions: Optional[List[str]] = None) -> None:
        """添加状态转换"""
        if from_state not in self.states or to_state not in self.states:
            raise ValueError("状态不存在")
        
        transition = FSMTransition(
            from_state=self.states[from_state],
            to_state=self.states[to_state],
            condition=condition,
            actions=actions or []
        )
        self.transitions.append(transition)
    
    def add_state_output(self, state_name: str, outputs: List[str]) -> None:
        """添加状态输出"""
        if state_name not in self.states:
            raise ValueError(f"状态不存在: {state_name}")
        self.state_outputs[state_name] = outputs
    
    def set_default_outputs(self, outputs: List[str]) -> None:
        """设置默认输出"""
        self.default_outputs = outputs
    
    def _generate_state_encoding(self) -> str:
        """生成状态编码"""
        lines = [f"// State encoding: {self.encoding}"]
        
        if self.encoding == "one_hot":
            for i, (name, state) in enumerate(self.states.items()):
                lines.append(f"localparam {name} = {len(self.states)}'b" + 
                           "0" * (len(self.states) - i - 1) + "1" + "0" * i + ";")
        elif self.encoding == "gray":
            # 格雷码编码
            for i, (name, state) in enumerate(self.states.items()):
                gray = i ^ (i >> 1)
                lines.append(f"localparam {name} = {len(self.states)}'d{gray};")
        else:  # binary
            for name, state in self.states.items():
                if state.encoding is not None:
                    lines.append(f"localparam {name} = {len(self.states)}'d{state.encoding};")
                else:
                    lines.append(f"localparam {name} = {len(self.states)}'d{list(self.states.keys()).index(name)};")
        
        return "\n".join(lines)
    
    def to_verilog(self, state_var: str = "state", next_state_var: str = "next_state") -> str:
        """生成FSM的Verilog代码"""
        lines = []
        
        # 状态编码
        lines.append(self._generate_state_encoding())
        lines.append("")
        
        # 状态寄存器声明
        state_width = len(self.states) if self.encoding == "one_hot" else (len(self.states) - 1).bit_length()
        lines.append(f"reg [{state_width-1}:0] {state_var}, {next_state_var};")
        lines.append("")
        
        # 状态转移逻辑 (组合逻辑)
        lines.append("// State transition logic")
        lines.append(f"always @(*) begin")
        lines.append(f"    {next_state_var} = {state_var};  // Default: hold state")
        lines.append(f"    case ({state_var})")
        
        # 按状态组织转换
        state_transitions: Dict[str, List[FSMTransition]] = {name: [] for name in self.states}
        for trans in self.transitions:
            state_transitions[trans.from_state.name].append(trans)
        
        for state_name, transitions in state_transitions.items():
            lines.append(f"        {state_name}: begin")
            for trans in transitions:
                lines.append(f"            if ({trans.condition}) begin")
                lines.append(f"                {next_state_var} = {trans.to_state.name};")
                for action in trans.actions:
                    lines.append(f"                {action};")
                lines.append(f"            end")
            lines.append(f"        end")
        
        lines.append(f"        default: {next_state_var} = {list(self.states.keys())[0]};")
        lines.append(f"    endcase")
        lines.append(f"end")
        lines.append("")
        
        # 状态寄存器 (时序逻辑)
        lines.append("// State register")
        if self.reset_active_low:
            lines.append(f"always @(posedge {self.clock} or negedge {self.reset}) begin")
            lines.append(f"    if (!{self.reset}) begin")
        else:
            lines.append(f"always @(posedge {self.clock} or posedge {self.reset}) begin")
            lines.append(f"    if ({self.reset}) begin")
        
        lines.append(f"        {state_var} <= {list(self.states.keys())[0]};")
        lines.append(f"    end else begin")
        lines.append(f"        {state_var} <= {next_state_var};")
        lines.append(f"    end")
        lines.append(f"end")
        lines.append("")
        
        # 输出逻辑
        lines.append("// Output logic")
        lines.append(f"always @(*) begin")
        for output in self.default_outputs:
            lines.append(f"    {output};")
        lines.append(f"    case ({state_var})")
        
        for state_name, outputs in self.state_outputs.items():
            lines.append(f"        {state_name}: begin")
            for output in outputs:
                lines.append(f"            {output};")
            lines.append(f"        end")
        
        lines.append(f"    endcase")
        lines.append(f"end")
        
        return "\n".join(lines)


class VerilogModule:
    """
    Verilog模块生成器
    
    用于生成完整的Verilog模块。
    """
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.parameters: List[Parameter] = []
        self.localparams: List[LocalParam] = []
        self.ports: List[Signal] = []
        self.internal_signals: List[Signal] = []
        self.combinational: Optional[CombinationalLogic] = None
        self.sequential: Optional[SequentialLogic] = None
        self.fsms: List[FSMGenerator] = []
        self.submodules: List[Tuple[str, str, Dict[str, str]]] = []  # (module_name, instance_name, port_map)
        self.custom_code: List[str] = []
    
    def add_parameter(self, name: str, value: Union[int, str],
                     description: str = "") -> None:
        """添加参数"""
        self.parameters.append(Parameter(name, value, description))
    
    def add_localparam(self, name: str, value: Union[int, str],
                      description: str = "") -> None:
        """添加本地参数"""
        self.localparams.append(LocalParam(name, value, description))
    
    def add_port(self, name: str, direction: SignalDirection,
                width: int = 1, signal_type: SignalType = SignalType.WIRE,
                signed: bool = False, description: str = "") -> Signal:
        """添加端口"""
        signal = Signal(name, direction, width, signal_type, signed, description=description)
        self.ports.append(signal)
        return signal
    
    def add_internal_signal(self, name: str, width: int = 1,
                           signal_type: SignalType = SignalType.WIRE,
                           description: str = "") -> Signal:
        """添加内部信号"""
        signal = Signal(name, None, width, signal_type, description=description)
        self.internal_signals.append(signal)
        return signal
    
    def set_combinational(self, comb: CombinationalLogic) -> None:
        """设置组合逻辑"""
        self.combinational = comb
    
    def set_sequential(self, seq: SequentialLogic) -> None:
        """设置时序逻辑"""
        self.sequential = seq
    
    def add_fsm(self, fsm: FSMGenerator) -> None:
        """添加FSM"""
        self.fsms.append(fsm)
    
    def add_submodule(self, module_name: str, instance_name: str,
                     port_map: Dict[str, str]) -> None:
        """添加子模块实例"""
        self.submodules.append((module_name, instance_name, port_map))
    
    def add_custom_code(self, code: str) -> None:
        """添加自定义代码"""
        self.custom_code.append(code)
    
    def to_verilog(self, system_verilog: bool = False) -> str:
        """生成完整的Verilog模块代码"""
        lines = []
        
        # 文件头注释
        lines.append(f"// Module: {self.name}")
        if self.description:
            lines.append(f"// Description: {self.description}")
        lines.append(f"// Auto-generated by Verilog Generator")
        lines.append("")
        
        # 模块声明
        if self.parameters:
            lines.append(f"module {self.name} #(")
            for i, param in enumerate(self.parameters):
                suffix = "," if i < len(self.parameters) - 1 else ""
                lines.append(f"    {param.to_verilog()}{suffix}")
            lines.append(f") (")
        else:
            lines.append(f"module {self.name} (")
        
        # 端口列表
        for i, port in enumerate(self.ports):
            suffix = "," if i < len(self.ports) - 1 else ""
            lines.append(f"    {port.direction.value} {port.name}{suffix}")
        lines.append(f");")
        lines.append("")
        
        # 本地参数
        if self.localparams:
            for lp in self.localparams:
                lines.append(f"    {lp.to_verilog()}")
            lines.append("")
        
        # 内部信号声明
        if self.internal_signals:
            for sig in self.internal_signals:
                lines.append(f"    {sig.to_verilog(system_verilog)}")
            lines.append("")
        
        # 端口详细声明（如果是传统Verilog）
        if not system_verilog:
            for port in self.ports:
                if port.width > 1:
                    lines.append(f"    {port.to_verilog(system_verilog)}")
            if any(p.width > 1 for p in self.ports):
                lines.append("")
        
        # 子模块实例
        if self.submodules:
            for mod_name, inst_name, port_map in self.submodules:
                lines.append(f"    // Submodule: {mod_name}")
                lines.append(f"    {mod_name} {inst_name} (")
                port_items = list(port_map.items())
                for i, (port, conn) in enumerate(port_items):
                    suffix = "," if i < len(port_items) - 1 else ""
                    lines.append(f"        .{port}({conn}){suffix}")
                lines.append(f"    );")
                lines.append("")
        
        # 组合逻辑
        if self.combinational:
            lines.append(textwrap.indent(self.combinational.to_verilog(), "    "))
            lines.append("")
        
        # 时序逻辑
        if self.sequential:
            lines.append(textwrap.indent(self.sequential.to_verilog(), "    "))
            lines.append("")
        
        # FSM
        for fsm in self.fsms:
            lines.append(f"    // FSM: {fsm.name}")
            lines.append(textwrap.indent(fsm.to_verilog(), "    "))
            lines.append("")
        
        # 自定义代码
        for code in self.custom_code:
            lines.append(textwrap.indent(code, "    "))
            lines.append("")
        
        # 模块结束
        lines.append(f"endmodule")
        
        return "\n".join(lines)


class TestbenchGenerator:
    """
    测试平台生成器
    
    生成Verilog测试平台代码。
    """
    
    def __init__(self, module_name: str, tb_name: Optional[str] = None):
        self.module_name = module_name
        self.tb_name = tb_name or f"{module_name}_tb"
        self.signals: Dict[str, Dict[str, Any]] = {}
        self.clock_period: float = 10.0  # ns
        self.clock_name: str = "clk"
        self.reset_name: str = "rst_n"
        self.test_cases: List[str] = []
        self.initializations: List[str] = []
    
    def add_signal(self, name: str, width: int = 1, 
                   is_reg: bool = True, description: str = "") -> None:
        """添加测试信号"""
        self.signals[name] = {
            'width': width,
            'is_reg': is_reg,
            'description': description
        }
    
    def set_clock(self, name: str = "clk", period_ns: float = 10.0) -> None:
        """设置时钟"""
        self.clock_name = name
        self.clock_period = period_ns
        self.add_signal(name, 1, True, "Clock signal")
    
    def set_reset(self, name: str = "rst_n", active_low: bool = True) -> None:
        """设置复位"""
        self.reset_name = name
        self.reset_active_low = active_low
        self.add_signal(name, 1, True, "Reset signal")
    
    def add_initialization(self, code: str) -> None:
        """添加初始化代码"""
        self.initializations.append(code)
    
    def add_test_case(self, code: str) -> None:
        """添加测试用例"""
        self.test_cases.append(code)
    
    def generate_clock_generator(self) -> str:
        """生成时钟生成器"""
        half_period = self.clock_period / 2
        return f"""// Clock generation
    always begin
        #{half_period} {self.clock_name} = ~{self.clock_name};
    end
"""
    
    def to_verilog(self) -> str:
        """生成测试平台代码"""
        lines = []
        
        # 文件头
        lines.append(f"// Testbench for {self.module_name}")
        lines.append(f"`timescale 1ns/1ps")
        lines.append("")
        lines.append(f"module {self.tb_name};")
        lines.append("")
        
        # 信号声明
        for name, info in self.signals.items():
            type_str = "reg" if info['is_reg'] else "wire"
            if info['width'] > 1:
                lines.append(f"    {type_str} [{info['width']-1}:0] {name};  // {info['description']}")
            else:
                lines.append(f"    {type_str} {name};  // {info['description']}")
        lines.append("")
        
        # 实例化被测模块
        lines.append(f"    // Instantiate the Unit Under Test (UUT)")
        lines.append(f"    {self.module_name} uut (")
        
        # 假设端口名称与信号名称匹配
        signal_names = list(self.signals.keys())
        for i, name in enumerate(signal_names):
            suffix = "," if i < len(signal_names) - 1 else ""
            lines.append(f"        .{name}({name}){suffix}")
        lines.append(f"    );")
        lines.append("")
        
        # 时钟生成
        lines.append(self.generate_clock_generator())
        
        # 初始化块
        lines.append(f"    // Test stimulus")
        lines.append(f"    initial begin")
        lines.append(f"        // Initialize inputs")
        for name, info in self.signals.items():
            if info['is_reg'] and name != self.clock_name:
                lines.append(f"        {name} = 0;")
        lines.append("")
        
        # 波形转储
        lines.append(f"        // Dump waves")
        lines.append(f'        $dumpfile("{self.tb_name}.vcd");')
        lines.append(f"        $dumpvars(0, {self.tb_name});")
        lines.append("")
        
        # 复位序列
        lines.append(f"        // Reset sequence")
        if self.reset_active_low:
            lines.append(f"        {self.reset_name} = 0;")
            lines.append(f"        #({self.clock_period} * 5);")
            lines.append(f"        {self.reset_name} = 1;")
        else:
            lines.append(f"        {self.reset_name} = 1;")
            lines.append(f"        #({self.clock_period} * 5);")
            lines.append(f"        {self.reset_name} = 0;")
        lines.append("")
        
        # 自定义初始化
        for init in self.initializations:
            lines.append(f"        {init}")
        if self.initializations:
            lines.append("")
        
        # 测试用例
        for i, test in enumerate(self.test_cases):
            lines.append(f"        // Test case {i+1}")
            lines.append(f"        {test}")
            lines.append("")
        
        # 结束仿真
        lines.append(f"        // End simulation")
        lines.append(f"        #({self.clock_period} * 10);")
        lines.append(f'        $display("Simulation completed successfully!");')
        lines.append(f"        $finish;")
        lines.append(f"    end")
        lines.append("")
        
        # 监控块
        lines.append(f"    // Monitor")
        lines.append(f"    initial begin")
        lines.append(f'        $monitor("Time=%0t', end='')
        for name in self.signals.keys():
            lines[-1] += f' {name}=%b'
        lines[-1] += f'", $time'
        for name in self.signals.keys():
            lines[-1] += f', {name}'
        lines[-1] += ");"
        lines.append(f"    end")
        lines.append("")
        
        # 模块结束
        lines.append(f"endmodule")
        
        return "\n".join(lines)


class VerilogLibrary:
    """
    Verilog常用模块库
    
    提供常用硬件模块的生成。
    """
    
    @staticmethod
    def create_counter(name: str = "counter", width: int = 8,
                      enable: bool = True, load: bool = True) -> VerilogModule:
        """创建计数器模块"""
        mod = VerilogModule(name, f"{width}-bit counter with enable and load")
        
        # 端口
        mod.add_port("clk", SignalDirection.INPUT, 1, description="Clock")
        mod.add_port("rst_n", SignalDirection.INPUT, 1, description="Active-low reset")
        if enable:
            mod.add_port("en", SignalDirection.INPUT, 1, description="Enable")
        if load:
            mod.add_port("load", SignalDirection.INPUT, 1, description="Load")
            mod.add_port("load_val", SignalDirection.INPUT, width, description="Load value")
        mod.add_port("count", SignalDirection.OUTPUT, width, SignalType.REG, description="Counter value")
        mod.add_port("overflow", SignalDirection.OUTPUT, 1, SignalType.REG, description="Overflow flag")
        
        # 时序逻辑
        seq = SequentialLogic("clk", "rst_n", True)
        seq.add_register("count", width, 0)
        seq.add_register("overflow", 1, 0)
        
        statements = []
        if enable and load:
            statements.append(f"if (load) begin")
            statements.append(f"    count <= load_val;")
            statements.append(f"    overflow <= 0;")
            statements.append(f"end else if (en) begin")
            statements.append(f"    if (count == {width}'d{(1 << width) - 1}) begin")
            statements.append(f"        count <= 0;")
            statements.append(f"        overflow <= 1;")
            statements.append(f"    end else begin")
            statements.append(f"        count <= count + 1;")
            statements.append(f"        overflow <= 0;")
            statements.append(f"    end")
            statements.append(f"end")
        elif enable:
            statements.append(f"if (en) begin")
            statements.append(f"    if (count == {width}'d{(1 << width) - 1}) begin")
            statements.append(f"        count <= 0;")
            statements.append(f"        overflow <= 1;")
            statements.append(f"    end else begin")
            statements.append(f"        count <= count + 1;")
            statements.append(f"        overflow <= 0;")
            statements.append(f"    end")
            statements.append(f"end")
        else:
            statements.append(f"if (count == {width}'d{(1 << width) - 1}) begin")
            statements.append(f"    count <= 0;")
            statements.append(f"    overflow <= 1;")
            statements.append(f"end else begin")
            statements.append(f"    count <= count + 1;")
            statements.append(f"    overflow <= 0;")
            statements.append(f"end")
        
        seq.add_always_ff(statements)
        mod.set_sequential(seq)
        
        return mod
    
    @staticmethod
    def create_shift_register(name: str = "shift_reg", width: int = 8,
                             direction: str = "left") -> VerilogModule:
        """创建移位寄存器模块"""
        mod = VerilogModule(name, f"{width}-bit shift register")
        
        mod.add_port("clk", SignalDirection.INPUT, 1, description="Clock")
        mod.add_port("rst_n", SignalDirection.INPUT, 1, description="Active-low reset")
        mod.add_port("en", SignalDirection.INPUT, 1, description="Enable")
        mod.add_port("load", SignalDirection.INPUT, 1, description="Load")
        mod.add_port("load_val", SignalDirection.INPUT, width, description="Load value")
        mod.add_port("sin", SignalDirection.INPUT, 1, description="Serial input")
        mod.add_port("sout", SignalDirection.OUTPUT, 1, description="Serial output")
        mod.add_port("pout", SignalDirection.OUTPUT, width, SignalType.REG, description="Parallel output")
        
        seq = SequentialLogic("clk", "rst_n", True)
        seq.add_register("data", width, 0)
        
        statements = [
            "if (load) begin",
            "    data <= load_val;",
            "end else if (en) begin",
        ]
        
        if direction == "left":
            statements.append("    data <= {data[{width}-2:0], sin};")
        else:
            statements.append("    data <= {sin, data[{width}-1:1]};")
        
        statements.extend([
            "end",
            "pout = data;",
            f"sout = data[{width-1 if direction == 'left' else 0}];"
        ])
        
        seq.add_always_ff(statements)
        mod.set_sequential(seq)
        
        return mod
    
    @staticmethod
    def create_fifo(name: str = "fifo", data_width: int = 32,
                   depth: int = 16) -> VerilogModule:
        """创建FIFO模块"""
        addr_width = (depth - 1).bit_length()
        
        mod = VerilogModule(name, f"{depth}x{data_width} FIFO")
        
        mod.add_port("clk", SignalDirection.INPUT, 1, description="Clock")
        mod.add_port("rst_n", SignalDirection.INPUT, 1, description="Active-low reset")
        mod.add_port("wr_en", SignalDirection.INPUT, 1, description="Write enable")
        mod.add_port("rd_en", SignalDirection.INPUT, 1, description="Read enable")
        mod.add_port("din", SignalDirection.INPUT, data_width, description="Data input")
        mod.add_port("dout", SignalDirection.OUTPUT, data_width, SignalType.REG, description="Data output")
        mod.add_port("full", SignalDirection.OUTPUT, 1, SignalType.REG, description="FIFO full")
        mod.add_port("empty", SignalDirection.OUTPUT, 1, SignalType.REG, description="FIFO empty")
        mod.add_port("count", SignalDirection.OUTPUT, addr_width + 1, SignalType.REG, description="FIFO count")
        
        # 内部信号
        mod.add_internal_signal("mem", data_width, SignalType.REG, description="FIFO memory")
        mod.add_internal_signal("wr_ptr", addr_width, SignalType.REG, description="Write pointer")
        mod.add_internal_signal("rd_ptr", addr_width, SignalType.REG, description="Read pointer")
        
        # 时序逻辑
        seq = SequentialLogic("clk", "rst_n", True)
        
        statements = [
            "if (!rst_n) begin",
            "    wr_ptr <= 0;",
            "    rd_ptr <= 0;",
            "    count <= 0;",
            "    full <= 0;",
            "    empty <= 1;",
            "end else begin",
            "    // Write operation",
            "    if (wr_en && !full) begin",
            "        mem[wr_ptr] <= din;",
            "        wr_ptr <= wr_ptr + 1;",
            "        count <= count + 1;",
            "    end",
            "    // Read operation",
            "    if (rd_en && !empty) begin",
            "        dout <= mem[rd_ptr];",
            "        rd_ptr <= rd_ptr + 1;",
            "        count <= count - 1;",
            "    end",
            "    // Update flags",
            "    full <= (count == {depth});",
            "    empty <= (count == 0);",
            "end"
        ]
        
        seq.add_always_ff(statements)
        mod.set_sequential(seq)
        
        return mod


# 便捷函数
def generate_simple_adder(width: int = 32) -> str:
    """生成简单加法器"""
    mod = VerilogModule("adder", f"{width}-bit adder")
    mod.add_parameter("WIDTH", width, "Data width")
    mod.add_port("a", SignalDirection.INPUT, width, description="Input A")
    mod.add_port("b", SignalDirection.INPUT, width, description="Input B")
    mod.add_port("cin", SignalDirection.INPUT, 1, description="Carry in")
    mod.add_port("sum", SignalDirection.OUTPUT, width, description="Sum output")
    mod.add_port("cout", SignalDirection.OUTPUT, 1, description="Carry out")
    
    comb = CombinationalLogic()
    comb.add_assignment("{cout, sum}", "a + b + cin")
    mod.set_combinational(comb)
    
    return mod.to_verilog()


def generate_simple_fsm_example() -> str:
    """生成简单FSM示例"""
    fsm = FSMGenerator("traffic_light", "clk", "rst_n", True, "one_hot")
    
    # 添加状态
    fsm.add_state("IDLE", description="Idle state")
    fsm.add_state("GREEN", description="Green light")
    fsm.add_state("YELLOW", description="Yellow light")
    fsm.add_state("RED", description="Red light")
    
    # 添加转换
    fsm.add_transition("IDLE", "GREEN", "start", ["timer <= 0"])
    fsm.add_transition("GREEN", "YELLOW", "timer >= GREEN_TIME", ["timer <= 0"])
    fsm.add_transition("YELLOW", "RED", "timer >= YELLOW_TIME", ["timer <= 0"])
    fsm.add_transition("RED", "GREEN", "timer >= RED_TIME", ["timer <= 0"])
    
    # 添加输出
    fsm.set_default_outputs(["green = 0", "yellow = 0", "red = 0"])
    fsm.add_state_output("GREEN", ["green = 1"])
    fsm.add_state_output("YELLOW", ["yellow = 1"])
    fsm.add_state_output("RED", ["red = 1"])
    
    # 创建模块
    mod = VerilogModule("traffic_light", "Traffic light controller")
    mod.add_port("clk", SignalDirection.INPUT, 1, description="Clock")
    mod.add_port("rst_n", SignalDirection.INPUT, 1, description="Reset")
    mod.add_port("start", SignalDirection.INPUT, 1, description="Start signal")
    mod.add_port("green", SignalDirection.OUTPUT, 1, SignalType.REG, description="Green light")
    mod.add_port("yellow", SignalDirection.OUTPUT, 1, SignalType.REG, description="Yellow light")
    mod.add_port("red", SignalDirection.OUTPUT, 1, SignalType.REG, description="Red light")
    
    mod.add_localparam("GREEN_TIME", 100, "Green duration")
    mod.add_localparam("YELLOW_TIME", 20, "Yellow duration")
    mod.add_localparam("RED_TIME", 80, "Red duration")
    
    mod.add_internal_signal("timer", 8, SignalType.REG, description="Timer")
    
    mod.add_fsm(fsm)
    
    return mod.to_verilog()


# 示例用法
if __name__ == "__main__":
    # 生成加法器
    print("=== 加法器 ===")
    print(generate_simple_adder(32))
    print("\n")
    
    # 生成计数器
    print("=== 计数器 ===")
    counter = VerilogLibrary.create_counter("counter_8bit", 8)
    print(counter.to_verilog())
    print("\n")
    
    # 生成FSM
    print("=== 交通灯FSM ===")
    print(generate_simple_fsm_example())
    print("\n")
    
    # 生成测试平台
    print("=== 测试平台 ===")
    tb = TestbenchGenerator("counter_8bit")
    tb.set_clock("clk", 10)
    tb.set_reset("rst_n", True)
    tb.add_signal("en", 1, True)
    tb.add_signal("count", 8, False)
    tb.add_signal("overflow", 1, False)
    
    tb.add_initialization("en = 1;")
    tb.add_test_case("#100;")
    tb.add_test_case("en = 0;")
    tb.add_test_case("#50;")
    tb.add_test_case("en = 1;")
    
    print(tb.to_verilog())
