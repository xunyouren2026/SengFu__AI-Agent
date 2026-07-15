"""
Coq定理证明器接口模块

提供与Coq定理证明器的交互功能，支持Coq脚本生成、证明目标解析、
证明状态查询、证明步执行和错误处理。
"""

from __future__ import annotations

import re
import json
import subprocess
import tempfile
import os
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from pathlib import Path


class ProofState(Enum):
    """证明状态枚举"""
    IDLE = "idle"                    # 空闲状态
    PROVING = "proving"              # 证明中
    PROOF_COMPLETE = "complete"      # 证明完成
    ERROR = "error"                  # 错误状态
    INTERRUPTED = "interrupted"      # 中断


class CoqErrorType(Enum):
    """Coq错误类型枚举"""
    SYNTAX_ERROR = "syntax_error"           # 语法错误
    TYPE_ERROR = "type_error"               # 类型错误
    UNKNOWN_IDENTIFIER = "unknown_id"       # 未知标识符
    TACTIC_FAILURE = "tactic_failure"       # 策略失败
    PROOF_INCOMPLETE = "proof_incomplete"   # 证明不完整
    TIMEOUT = "timeout"                     # 超时
    SYSTEM_ERROR = "system_error"           # 系统错误


@dataclass
class CoqError:
    """Coq错误信息"""
    error_type: CoqErrorType
    message: str
    location: Optional[Tuple[int, int]] = None  # (行, 列)
    context: str = ""
    
    def __str__(self) -> str:
        loc_str = f" at line {self.location[0]}, col {self.location[1]}" if self.location else ""
        return f"[{self.error_type.value}]{loc_str}: {self.message}"


@dataclass
class ProofGoal:
    """
    证明目标
    
    表示Coq证明过程中的一个子目标。
    """
    goal_id: int
    conclusion: str
    hypotheses: List[Dict[str, str]] = field(default_factory=list)
    
    def __str__(self) -> str:
        lines = []
        for hyp in self.hypotheses:
            lines.append(f"{hyp['name']} : {hyp['type']}")
        lines.append("=" * 40)
        lines.append(self.conclusion)
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'goal_id': self.goal_id,
            'conclusion': self.conclusion,
            'hypotheses': self.hypotheses
        }


@dataclass
class ProofStateInfo:
    """
    证明状态信息
    
    包含当前证明的完整状态。
    """
    state: ProofState
    current_goals: List[ProofGoal] = field(default_factory=list)
    shelved_goals: List[ProofGoal] = field(default_factory=list)
    given_up_goals: List[ProofGoal] = field(default_factory=list)
    proof_stack: List[str] = field(default_factory=list)
    error: Optional[CoqError] = None
    
    @property
    def num_goals(self) -> int:
        """当前目标数量"""
        return len(self.current_goals)
    
    @property
    def is_complete(self) -> bool:
        """证明是否完成"""
        return self.state == ProofState.PROOF_COMPLETE
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'state': self.state.value,
            'num_goals': self.num_goals,
            'current_goals': [g.to_dict() for g in self.current_goals],
            'shelved_goals': [g.to_dict() for g in self.shelved_goals],
            'given_up_goals': [g.to_dict() for g in self.given_up_goals],
            'proof_stack': self.proof_stack,
            'error': str(self.error) if self.error else None
        }


class CoqScriptBuilder:
    """
    Coq脚本生成器
    
    用于程序化生成Coq证明脚本。
    """
    
    def __init__(self):
        self.lines: List[str] = []
        self.indent_level = 0
        self.indent_str = "  "
    
    def _indent(self) -> str:
        """获取当前缩进"""
        return self.indent_str * self.indent_level
    
    def add_comment(self, comment: str) -> CoqScriptBuilder:
        """添加注释"""
        for line in comment.split('\n'):
            self.lines.append(f"(* {line} *)")
        return self
    
    def add_require(self, module: str, import_all: bool = False) -> CoqScriptBuilder:
        """添加Require导入"""
        self.lines.append(f'Require Import {module}.')
        if import_all:
            self.lines.append(f'Import {module}.')
        return self
    
    def add_definition(self, name: str, def_type: str, body: str,
                       params: Optional[List[Tuple[str, str]]] = None) -> CoqScriptBuilder:
        """
        添加定义
        
        Args:
            name: 定义名称
            def_type: 定义类型 (Definition, Theorem, Lemma, etc.)
            body: 定义体
            params: 参数列表 [(param_name, param_type), ...]
        """
        if params:
            param_str = " ".join([f"({n}:{t})" for n, t in params])
            line = f"{def_type} {name} {param_str} : {body}."
        else:
            line = f"{def_type} {name} : {body}."
        self.lines.append(line)
        return self
    
    def start_proof(self, name: str, statement: str,
                   params: Optional[List[Tuple[str, str]]] = None) -> CoqScriptBuilder:
        """开始证明"""
        self.add_definition(name, "Theorem", statement, params)
        self.lines.append("Proof.")
        self.indent_level += 1
        return self
    
    def add_tactic(self, tactic: str) -> CoqScriptBuilder:
        """添加策略"""
        self.lines.append(f"{self._indent()}{tactic}.")
        return self
    
    def add_tactics(self, *tactics: str) -> CoqScriptBuilder:
        """添加多个策略"""
        for tactic in tactics:
            self.add_tactic(tactic)
        return self
    
    def start_section(self, name: str) -> CoqScriptBuilder:
        """开始Section"""
        self.lines.append(f"Section {name}.")
        self.indent_level += 1
        return self
    
    def end_section(self, name: str) -> CoqScriptBuilder:
        """结束Section"""
        self.indent_level -= 1
        self.lines.append(f"End {name}.")
        return self
    
    def add_variable(self, name: str, var_type: str) -> CoqScriptBuilder:
        """添加Variable声明"""
        self.lines.append(f"{self._indent()}Variable {name} : {var_type}.")
        return self
    
    def add_hypothesis(self, name: str, hyp_type: str) -> CoqScriptBuilder:
        """添加Hypothesis声明"""
        self.lines.append(f"{self._indent()}Hypothesis {name} : {hyp_type}.")
        return self
    
    def add_inductive(self, name: str, constructors: List[Tuple[str, str]],
                     type_params: Optional[List[str]] = None) -> CoqScriptBuilder:
        """
        添加归纳类型定义
        
        Args:
            name: 类型名称
            constructors: 构造子列表 [(name, type), ...]
            type_params: 类型参数列表
        """
        if type_params:
            params_str = " ".join([f"({p})" for p in type_params])
            line = f"Inductive {name} {params_str} : Set :="
        else:
            line = f"Inductive {name} : Set :="
        
        self.lines.append(line)
        
        for i, (cname, ctype) in enumerate(constructors):
            sep = "|" if i == 0 else "|"
            self.lines.append(f"  {sep} {cname} : {ctype}")
        
        self.lines.append(".")
        return self
    
    def add_fixpoint(self, name: str, params: List[Tuple[str, str]],
                    return_type: str, body: str,
                    struct_param: Optional[str] = None) -> CoqScriptBuilder:
        """
        添加Fixpoint定义
        
        Args:
            name: 函数名称
            params: 参数列表
            return_type: 返回类型
            body: 函数体
            struct_param: 结构递归参数
        """
        param_str = " ".join([f"({n}:{t})" for n, t in params])
        
        if struct_param:
            line = f"Fixpoint {name} {param_str} {{{struct_param}}} : {return_type} :="
        else:
            line = f"Fixpoint {name} {param_str} : {return_type} :="
        
        self.lines.append(line)
        self.lines.append(f"  {body}.")
        return self
    
    def end_proof(self, qed: bool = True) -> CoqScriptBuilder:
        """结束证明"""
        self.indent_level -= 1
        if qed:
            self.lines.append("Qed.")
        else:
            self.lines.append("Defined.")
        return self
    
    def add_notation(self, notation: str, meaning: str,
                    precedence: Optional[int] = None) -> CoqScriptBuilder:
        """添加Notation"""
        if precedence:
            self.lines.append(f'Notation "{notation}" := ({meaning}) (at level {precedence}).')
        else:
            self.lines.append(f'Notation "{notation}" := ({meaning}).')
        return self
    
    def to_string(self) -> str:
        """生成完整脚本"""
        return "\n".join(self.lines)
    
    def save_to_file(self, filepath: str) -> None:
        """保存到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_string())


class CoqInterface:
    """
    Coq定理证明器接口
    
    提供与Coq交互的核心功能。
    """
    
    def __init__(self, coq_path: str = "coqtop", 
                 working_dir: Optional[str] = None,
                 timeout: int = 30):
        self.coq_path = coq_path
        self.working_dir = working_dir or os.getcwd()
        self.timeout = timeout
        self.process: Optional[subprocess.Popen] = None
        self.current_state = ProofStateInfo(ProofState.IDLE)
        self.proof_history: List[ProofStateInfo] = []
        self.script_builder = CoqScriptBuilder()
    
    def start(self) -> bool:
        """启动Coq进程"""
        try:
            self.process = subprocess.Popen(
                [self.coq_path, "-emacs"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.working_dir
            )
            self.current_state = ProofStateInfo(ProofState.IDLE)
            return True
        except Exception as e:
            self.current_state = ProofStateInfo(
                ProofState.ERROR,
                error=CoqError(CoqErrorType.SYSTEM_ERROR, str(e))
            )
            return False
    
    def stop(self) -> None:
        """停止Coq进程"""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            self.current_state = ProofStateInfo(ProofState.IDLE)
    
    def _send_command(self, command: str) -> Tuple[str, str, int]:
        """
        发送命令到Coq
        
        Returns:
            (stdout, stderr, returncode)
        """
        if not self.process:
            raise RuntimeError("Coq进程未启动")
        
        try:
            full_command = command + "\n"
            self.process.stdin.write(full_command)
            self.process.stdin.flush()
            
            # 读取输出
            stdout = ""
            stderr = ""
            
            # 简化实现：实际应该解析Coq的emacs协议
            import select
            import time
            
            start_time = time.time()
            while time.time() - start_time < self.timeout:
                if self.process.poll() is not None:
                    break
                
                # 检查是否有输出
                if select.select([self.process.stdout], [], [], 0.1)[0]:
                    line = self.process.stdout.readline()
                    stdout += line
                    if "<prompt>" in line or "Proof completed." in line:
                        break
                
                # 检查错误
                if select.select([self.process.stderr], [], [], 0.1)[0]:
                    line = self.process.stderr.readline()
                    stderr += line
            
            return stdout, stderr, 0
            
        except Exception as e:
            return "", str(e), 1
    
    def execute_tactic(self, tactic: str) -> ProofStateInfo:
        """
        执行证明策略
        
        Args:
            tactic: Coq策略字符串
        
        Returns:
            执行后的证明状态
        """
        # 保存当前状态到历史
        self.proof_history.append(self.current_state)
        
        stdout, stderr, rc = self._send_command(tactic)
        
        if rc != 0 or stderr:
            error = self._parse_error(stderr or stdout)
            self.current_state = ProofStateInfo(
                ProofState.ERROR,
                error=error
            )
        else:
            # 解析新的证明状态
            self.current_state = self._parse_proof_state(stdout)
        
        return self.current_state
    
    def execute_script(self, script: str) -> List[ProofStateInfo]:
        """
        执行完整脚本
        
        Args:
            script: Coq脚本字符串
        
        Returns:
            每步执行后的状态列表
        """
        states = []
        
        # 分割脚本为独立命令
        commands = self._split_script(script)
        
        for cmd in commands:
            state = self.execute_tactic(cmd)
            states.append(state)
            
            if state.state == ProofState.ERROR:
                break
        
        return states
    
    def _split_script(self, script: str) -> List[str]:
        """将脚本分割为独立命令"""
        commands = []
        current = ""
        
        for line in script.split('\n'):
            line = line.strip()
            if not line or line.startswith('(*'):
                continue
            
            current += " " + line
            
            if line.endswith('.'):
                commands.append(current.strip())
                current = ""
        
        if current.strip():
            commands.append(current.strip())
        
        return commands
    
    def _parse_proof_state(self, output: str) -> ProofStateInfo:
        """解析证明状态输出"""
        info = ProofStateInfo(ProofState.PROVING)
        
        # 解析目标
        goals_match = re.findall(r'(\d+) subgoal\(s\)', output)
        if goals_match:
            num_goals = int(goals_match[0])
            
            # 解析每个目标
            for i in range(num_goals):
                goal = self._parse_goal(output, i)
                if goal:
                    info.current_goals.append(goal)
        
        # 检查证明是否完成
        if "Proof completed" in output or "No more subgoals" in output:
            info.state = ProofState.PROOF_COMPLETE
        
        return info
    
    def _parse_goal(self, output: str, goal_idx: int) -> Optional[ProofGoal]:
        """解析单个目标"""
        # 简化实现：实际应该解析Coq的详细输出
        lines = output.split('\n')
        
        hypotheses = []
        conclusion = ""
        in_goal = False
        
        for line in lines:
            if f"subgoal {goal_idx + 1}" in line:
                in_goal = True
                continue
            
            if in_goal:
                if line.strip() == "":
                    continue
                
                # 假设行通常包含 ":"
                if ':' in line and not line.strip().startswith('='):
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        hypotheses.append({
                            'name': parts[0].strip(),
                            'type': parts[1].strip()
                        })
                else:
                    conclusion = line.strip()
        
        if conclusion:
            return ProofGoal(
                goal_id=goal_idx,
                conclusion=conclusion,
                hypotheses=hypotheses
            )
        
        return None
    
    def _parse_error(self, error_output: str) -> CoqError:
        """解析错误输出"""
        # 尝试识别错误类型
        if "Syntax error" in error_output:
            error_type = CoqErrorType.SYNTAX_ERROR
        elif "The command" in error_output and "has not been parsed" in error_output:
            error_type = CoqErrorType.SYNTAX_ERROR
        elif "Type error" in error_output or "type mismatch" in error_output:
            error_type = CoqErrorType.TYPE_ERROR
        elif "The reference" in error_output and "was not found" in error_output:
            error_type = CoqErrorType.UNKNOWN_IDENTIFIER
        elif "Tactic failure" in error_output:
            error_type = CoqErrorType.TACTIC_FAILURE
        elif "Some proof is still incomplete" in error_output:
            error_type = CoqErrorType.PROOF_INCOMPLETE
        elif "Timeout" in error_output:
            error_type = CoqErrorType.TIMEOUT
        else:
            error_type = CoqErrorType.SYSTEM_ERROR
        
        # 尝试提取位置信息
        location = None
        loc_match = re.search(r'at line (\d+), column (\d+)', error_output)
        if loc_match:
            location = (int(loc_match.group(1)), int(loc_match.group(2)))
        
        return CoqError(
            error_type=error_type,
            message=error_output.strip(),
            location=location
        )
    
    def get_current_state(self) -> ProofStateInfo:
        """获取当前证明状态"""
        return self.current_state
    
    def undo(self, steps: int = 1) -> ProofStateInfo:
        """撤销证明步骤"""
        for _ in range(steps):
            if self.proof_history:
                self.current_state = self.proof_history.pop()
                self._send_command("Undo.")
            else:
                break
        
        return self.current_state
    
    def restart_proof(self) -> ProofStateInfo:
        """重新开始当前证明"""
        self._send_command("Restart.")
        self.current_state = ProofStateInfo(ProofState.PROVING)
        self.proof_history = []
        return self.current_state
    
    def admit_goal(self) -> ProofStateInfo:
        """接受当前目标（跳过证明）"""
        return self.execute_tactic("admit.")
    
    def abort_proof(self) -> ProofStateInfo:
        """中止当前证明"""
        self._send_command("Abort.")
        self.current_state = ProofStateInfo(ProofState.IDLE)
        self.proof_history = []
        return self.current_state
    
    def query_type(self, term: str) -> str:
        """
        查询项的类型
        
        Args:
            term: Coq项
        
        Returns:
            类型字符串
        """
        stdout, _, _ = self._send_command(f"Check {term}.")
        return stdout.strip()
    
    def search_identifier(self, pattern: str) -> List[str]:
        """
        搜索标识符
        
        Args:
            pattern: 搜索模式
        
        Returns:
            匹配的标识符列表
        """
        stdout, _, _ = self._send_command(f"Search {pattern}.")
        # 解析搜索结果
        results = []
        for line in stdout.split('\n'):
            if line.strip() and not line.startswith('('):
                results.append(line.strip())
        return results
    
    def print_definitions(self) -> str:
        """打印当前环境中的所有定义"""
        stdout, _, _ = self._send_command("Print All.")
        return stdout


class CoqProofAssistant:
    """
    Coq证明助手
    
    提供更高级的证明辅助功能。
    """
    
    def __init__(self, coq_interface: Optional[CoqInterface] = None):
        self.coq = coq_interface or CoqInterface()
        self.suggested_tactics: List[str] = []
    
    def start_interactive_proof(self, theorem_name: str, statement: str) -> ProofStateInfo:
        """开始交互式证明"""
        if not self.coq.process:
            self.coq.start()
        
        # 声明定理
        self.coq.execute_tactic(f"Theorem {theorem_name} : {statement}.")
        
        # 开始证明
        return self.coq.execute_tactic("Proof.")
    
    def suggest_tactics(self, goal: ProofGoal) -> List[str]:
        """
        根据目标建议策略
        
        这是基于启发式的简单建议系统。
        """
        suggestions = []
        
        conclusion = goal.conclusion.lower()
        
        # 根据结论形式建议策略
        if "->" in conclusion or "implies" in conclusion:
            suggestions.extend(["intros", "intro H"])
        
        if "forall" in conclusion:
            suggestions.append("intros")
        
        if "exists" in conclusion:
            suggestions.append("exists")
        
        if "and" in conclusion or "/\\" in conclusion:
            suggestions.extend(["split", "constructor"])
        
        if "or" in conclusion or "\\/" in conclusion:
            suggestions.extend(["left", "right"])
        
        if conclusion.startswith("False"):
            suggestions.append("contradiction")
        
        if conclusion.startswith("True"):
            suggestions.append("constructor")
        
        # 通用策略
        suggestions.extend(["auto", "trivial", "reflexivity", "simpl"])
        
        self.suggested_tactics = suggestions
        return suggestions
    
    def auto_prove(self, max_depth: int = 5) -> bool:
        """
        尝试自动证明
        
        使用简单的自动策略组合。
        """
        auto_tactics = [
            "auto",
            "auto with *",
            "intuition",
            "tauto",
            "firstorder",
            "eauto",
            "eauto with *"
        ]
        
        for tactic in auto_tactics:
            state = self.coq.execute_tactic(tactic + ".")
            if state.is_complete:
                return True
            self.coq.undo()
        
        return False
    
    def try_tactics(self, tactics: List[str]) -> Dict[str, ProofStateInfo]:
        """
        尝试多个策略并返回结果
        
        Args:
            tactics: 策略列表
        
        Returns:
            策略到状态的映射
        """
        results = {}
        
        for tactic in tactics:
            state = self.coq.execute_tactic(tactic)
            results[tactic] = state
            
            # 撤销以尝试下一个
            if not state.is_complete:
                self.coq.undo()
        
        return results
    
    def generate_proof_script(self, tactics: List[str]) -> str:
        """生成证明脚本"""
        builder = CoqScriptBuilder()
        for tactic in tactics:
            builder.add_tactic(tactic)
        builder.end_proof()
        return builder.to_string()


# 便捷函数
def create_simple_proof(name: str, statement: str, 
                       tactics: List[str]) -> str:
    """
    创建简单证明脚本
    
    Args:
        name: 定理名称
        statement: 定理陈述
        tactics: 证明策略列表
    
    Returns:
        完整Coq脚本
    """
    builder = CoqScriptBuilder()
    
    # 添加标准导入
    builder.add_require("Arith")
    
    # 开始证明
    builder.start_proof(name, statement)
    
    # 添加策略
    builder.add_tactics(*tactics)
    
    # 结束证明
    builder.end_proof()
    
    return builder.to_string()


def create_inductive_type(name: str, 
                         constructors: List[Tuple[str, str]]) -> str:
    """
    创建归纳类型定义
    
    Args:
        name: 类型名称
        constructors: 构造子列表 [(name, type), ...]
    
    Returns:
        Coq脚本
    """
    builder = CoqScriptBuilder()
    builder.add_inductive(name, constructors)
    return builder.to_string()


# 示例用法
if __name__ == "__main__":
    # 演示脚本生成
    print("=== Coq脚本生成示例 ===\n")
    
    # 创建一个简单的自然数定理证明
    script = create_simple_proof(
        "plus_comm",
        "forall n m : nat, n + m = m + n",
        ["intros", "induction n", "simpl", "reflexivity", 
         "simpl", "rewrite IHn", "reflexivity"]
    )
    print(script)
    print("\n")
    
    # 创建归纳类型
    print("=== 归纳类型定义 ===\n")
    list_script = create_inductive_type(
        "mylist",
        [
            ("mynil", "mylist"),
            ("mycons", "nat -> mylist -> mylist")
        ]
    )
    print(list_script)
    print("\n")
    
    # 演示脚本构建器
    print("=== 复杂脚本构建 ===\n")
    builder = CoqScriptBuilder()
    builder.add_require("Arith")
    builder.add_require("List")
    builder.start_section("MySection")
    builder.add_variable("n", "nat")
    builder.add_hypothesis("H", "n > 0")
    builder.start_proof("my_lemma", "n + 0 = n", [("n", "nat")])
    builder.add_tactics("simpl", "reflexivity")
    builder.end_proof()
    builder.end_section("MySection")
    
    print(builder.to_string())
