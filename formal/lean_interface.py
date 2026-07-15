"""
Lean定理证明器接口模块

提供与Lean 4定理证明器的交互功能，支持Lean代码生成、证明状态管理、
策略应用、类型检查和证明导出。
"""

from __future__ import annotations

import re
import json
import subprocess
import os
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from pathlib import Path


class LeanProofState(Enum):
    """Lean证明状态枚举"""
    IDLE = "idle"
    PROVING = "proving"
    PROOF_COMPLETE = "complete"
    ERROR = "error"
    TYPE_ERROR = "type_error"


class LeanErrorType(Enum):
    """Lean错误类型枚举"""
    SYNTAX_ERROR = "syntax_error"
    TYPE_MISMATCH = "type_mismatch"
    UNKNOWN_IDENTIFIER = "unknown_identifier"
    TACTIC_FAILED = "tactic_failed"
    PROOF_INCOMPLETE = "proof_incomplete"
    TIMEOUT = "timeout"
    SYSTEM_ERROR = "system_error"
    PARSER_ERROR = "parser_error"


@dataclass
class LeanError:
    """Lean错误信息"""
    error_type: LeanErrorType
    message: str
    file_path: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None
    
    def __str__(self) -> str:
        loc = ""
        if self.file_path:
            loc = f"{self.file_path}"
            if self.line:
                loc += f":{self.line}"
                if self.column:
                    loc += f":{self.column}"
            loc = f" at {loc}"
        return f"[{self.error_type.value}]{loc}: {self.message}"


@dataclass
class LeanGoal:
    """
    Lean证明目标
    
    表示Lean证明过程中的一个子目标。
    """
    goal_id: int
    conclusion: str
    hypotheses: List[Dict[str, str]] = field(default_factory=list)
    is_conversion: bool = False
    
    def __str__(self) -> str:
        lines = []
        for hyp in self.hypotheses:
            lines.append(f"{hyp['name']} : {hyp['type']}")
        if self.hypotheses:
            lines.append("⊢ " + self.conclusion)
        else:
            lines.append(self.conclusion)
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'goal_id': self.goal_id,
            'conclusion': self.conclusion,
            'hypotheses': self.hypotheses,
            'is_conversion': self.is_conversion
        }


@dataclass
class LeanTacticInfo:
    """策略执行信息"""
    tactic_name: str
    tactic_state_before: str
    tactic_state_after: str
    goals_before: List[LeanGoal]
    goals_after: List[LeanGoal]
    elapsed_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tactic_name': self.tactic_name,
            'tactic_state_before': self.tactic_state_before,
            'tactic_state_after': self.tactic_state_after,
            'goals_before': [g.to_dict() for g in self.goals_before],
            'goals_after': [g.to_dict() for g in self.goals_after],
            'elapsed_time_ms': self.elapsed_time_ms
        }


@dataclass
class LeanProofSnapshot:
    """Lean证明快照"""
    state: LeanProofState
    current_goals: List[LeanGoal] = field(default_factory=list)
    all_goals: List[LeanGoal] = field(default_factory=list)
    tactic_history: List[LeanTacticInfo] = field(default_factory=list)
    error: Optional[LeanError] = None
    environment: Dict[str, str] = field(default_factory=dict)
    
    @property
    def num_goals(self) -> int:
        return len(self.current_goals)
    
    @property
    def is_complete(self) -> bool:
        return self.state == LeanProofState.PROOF_COMPLETE
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'state': self.state.value,
            'num_goals': self.num_goals,
            'current_goals': [g.to_dict() for g in self.current_goals],
            'all_goals': [g.to_dict() for g in self.all_goals],
            'tactic_history': [t.to_dict() for t in self.tactic_history],
            'error': str(self.error) if self.error else None,
            'environment': self.environment
        }


class LeanCodeBuilder:
    """
    Lean代码生成器
    
    用于程序化生成Lean 4代码。
    """
    
    def __init__(self):
        self.lines: List[str] = []
        self.indent_level = 0
        self.indent_str = "  "
        self.imports: set = set()
        self.open_namespaces: set = set()
    
    def _indent(self) -> str:
        return self.indent_str * self.indent_level
    
    def add_import(self, module: str, with_prefix: bool = True) -> LeanCodeBuilder:
        """添加import语句"""
        if with_prefix:
            self.imports.add(f"import {module}")
        else:
            self.imports.add(f"import {module}")
        return self
    
    def add_open(self, namespace: str) -> LeanCodeBuilder:
        """添加open语句"""
        self.open_namespaces.add(f"open {namespace}")
        return self
    
    def add_comment(self, comment: str, doc: bool = False) -> LeanCodeBuilder:
        """添加注释"""
        if doc:
            for line in comment.split('\n'):
                self.lines.append(f"/-- {line} -/")
        else:
            for line in comment.split('\n'):
                self.lines.append(f"-- {line}")
        return self
    
    def add_def(self, name: str, signature: str, body: str,
                is_theorem: bool = False, is_instance: bool = False) -> LeanCodeBuilder:
        """
        添加定义
        
        Args:
            name: 定义名称
            signature: 类型签名
            body: 定义体
            is_theorem: 是否是定理
            is_instance: 是否是类型类实例
        """
        if is_theorem:
            keyword = "theorem"
        elif is_instance:
            keyword = "instance"
        else:
            keyword = "def"
        
        self.lines.append(f"{keyword} {name} : {signature} := by")
        self.indent_level += 1
        
        # 添加证明体
        for line in body.split('\n'):
            self.lines.append(f"{self._indent()}{line}")
        
        self.indent_level -= 1
        return self
    
    def add_inductive(self, name: str, params: List[str],
                     type_signature: str,
                     constructors: List[Tuple[str, str]]) -> LeanCodeBuilder:
        """
        添加归纳类型定义
        
        Args:
            name: 类型名称
            params: 类型参数
            type_signature: 类型签名
            constructors: 构造子列表 [(name, type), ...]
        """
        param_str = " ".join(params) if params else ""
        self.lines.append(f"inductive {name} {param_str} : {type_signature} where")
        self.indent_level += 1
        
        for cname, ctype in constructors:
            self.lines.append(f"{self._indent()}| {cname} : {ctype}")
        
        self.indent_level -= 1
        return self
    
    def add_structure(self, name: str, params: List[str],
                     extends: Optional[str],
                     fields: List[Tuple[str, str]]) -> LeanCodeBuilder:
        """
        添加结构体定义
        
        Args:
            name: 结构体名称
            params: 类型参数
            extends: 继承的父结构
            fields: 字段列表 [(name, type), ...]
        """
        param_str = " ".join(params) if params else ""
        extends_str = f" extends {extends}" if extends else ""
        
        self.lines.append(f"structure {name} {param_str}{extends_str} where")
        self.indent_level += 1
        
        for fname, ftype in fields:
            self.lines.append(f"{self._indent()}{fname} : {ftype}")
        
        self.indent_level -= 1
        return self
    
    def add_tactic_proof(self, name: str, statement: str,
                        tactics: List[str]) -> LeanCodeBuilder:
        """添加策略证明"""
        self.lines.append(f"theorem {name} : {statement} := by")
        self.indent_level += 1
        
        for tactic in tactics:
            self.lines.append(f"{self._indent()}{tactic}")
        
        self.indent_level -= 1
        return self
    
    def add_term_proof(self, name: str, statement: str,
                      proof_term: str) -> LeanCodeBuilder:
        """添加项证明"""
        self.lines.append(f"theorem {name} : {statement} :=")
        self.indent_level += 1
        self.lines.append(f"{self._indent()}{proof_term}")
        self.indent_level -= 1
        return self
    
    def add_function(self, name: str, params: List[Tuple[str, str]],
                    return_type: str, body: str,
                    is_partial: bool = False) -> LeanCodeBuilder:
        """
        添加函数定义
        
        Args:
            name: 函数名称
            params: 参数列表 [(name, type), ...]
            return_type: 返回类型
            body: 函数体
            is_partial: 是否是偏函数
        """
        partial_str = "partial " if is_partial else ""
        param_str = " ".join([f"({n} : {t})" for n, t in params])
        
        self.lines.append(f"{partial_str}def {name} {param_str} : {return_type} :=")
        self.indent_level += 1
        self.lines.append(f"{self._indent()}{body}")
        self.indent_level -= 1
        return self
    
    def add_match(self, expr: str, cases: List[Tuple[str, str]]) -> LeanCodeBuilder:
        """
        添加match表达式
        
        Args:
            expr: 匹配表达式
            cases: 分支列表 [(pattern, body), ...]
        """
        self.lines.append(f"{self._indent()}match {expr} with")
        
        for pattern, body in cases:
            self.lines.append(f"{self._indent()}| {pattern} => {body}")
        
        return self
    
    def add_namespace(self, name: str) -> LeanCodeBuilder:
        """开始namespace"""
        self.lines.append(f"namespace {name}")
        self.indent_level += 1
        return self
    
    def end_namespace(self, name: str) -> LeanCodeBuilder:
        """结束namespace"""
        self.indent_level -= 1
        self.lines.append(f"end {name}")
        return self
    
    def add_variable(self, name: str, var_type: str) -> LeanCodeBuilder:
        """添加变量声明"""
        self.lines.append(f"variable ({name} : {var_type})")
        return self
    
    def add_axiom(self, name: str, statement: str) -> LeanCodeBuilder:
        """添加公理"""
        self.lines.append(f"axiom {name} : {statement}")
        return self
    
    def add_notation(self, notation: str, precedence: int,
                    meaning: str) -> LeanCodeBuilder:
        """添加notation"""
        self.lines.append(f'notation (priority := {precedence}) "{notation}" => {meaning}')
        return self
    
    def add_macro(self, name: str, params: List[str], body: str) -> LeanCodeBuilder:
        """添加宏定义"""
        param_str = " ".join(params)
        self.lines.append(f"macro {name} {param_str} : term => `{body}")
        return self
    
    def add_instance_deriving(self, type_name: str,
                             typeclasses: List[str]) -> LeanCodeBuilder:
        """添加派生实例"""
        tc_str = ", ".join(typeclasses)
        self.lines.append(f"deriving instance {tc_str} for {type_name}")
        return self
    
    def to_string(self) -> str:
        """生成完整代码"""
        result = []
        
        # 添加imports
        for imp in sorted(self.imports):
            result.append(imp)
        if self.imports:
            result.append("")
        
        # 添加open
        for op in sorted(self.open_namespaces):
            result.append(op)
        if self.open_namespaces:
            result.append("")
        
        # 添加主体
        result.extend(self.lines)
        
        return "\n".join(result)
    
    def save_to_file(self, filepath: str) -> None:
        """保存到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_string())


class LeanInterface:
    """
    Lean定理证明器接口
    
    提供与Lean 4交互的核心功能。
    """
    
    def __init__(self, lean_path: str = "lean",
                 lake_path: str = "lake",
                 working_dir: Optional[str] = None,
                 timeout: int = 30):
        self.lean_path = lean_path
        self.lake_path = lake_path
        self.working_dir = working_dir or os.getcwd()
        self.timeout = timeout
        self.current_snapshot = LeanProofSnapshot(LeanProofState.IDLE)
        self.proof_history: List[LeanProofSnapshot] = []
        self.code_builder = LeanCodeBuilder()
    
    def check_file(self, filepath: str) -> Tuple[bool, List[LeanError]]:
        """
        类型检查文件
        
        Args:
            filepath: Lean文件路径
        
        Returns:
            (是否成功, 错误列表)
        """
        try:
            result = subprocess.run(
                [self.lean_path, filepath],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.working_dir
            )
            
            errors = []
            if result.returncode != 0:
                # 解析错误
                for line in result.stderr.split('\n'):
                    if line.strip():
                        error = self._parse_error_line(line, filepath)
                        errors.append(error)
            
            return result.returncode == 0, errors
            
        except subprocess.TimeoutExpired:
            return False, [LeanError(
                LeanErrorType.TIMEOUT,
                f"Type checking timed out after {self.timeout}s"
            )]
        except Exception as e:
            return False, [LeanError(
                LeanErrorType.SYSTEM_ERROR,
                str(e)
            )]
    
    def _parse_error_line(self, line: str, filepath: str) -> LeanError:
        """解析错误行"""
        # Lean错误格式: file.lean:line:col: error: message
        pattern = r'(.+):(\d+):(\d+):\s*(error|warning):\s*(.+)'
        match = re.match(pattern, line)
        
        if match:
            file_path = match.group(1)
            line_num = int(match.group(2))
            col_num = int(match.group(3))
            level = match.group(4)
            message = match.group(5)
            
            # 确定错误类型
            if "type mismatch" in message.lower():
                error_type = LeanErrorType.TYPE_MISMATCH
            elif "unknown identifier" in message.lower():
                error_type = LeanErrorType.UNKNOWN_IDENTIFIER
            elif "tactic" in message.lower() and "failed" in message.lower():
                error_type = LeanErrorType.TACTIC_FAILED
            elif "expected" in message.lower():
                error_type = LeanErrorType.SYNTAX_ERROR
            else:
                error_type = LeanErrorType.SYSTEM_ERROR
            
            return LeanError(
                error_type=error_type,
                message=message,
                file_path=file_path,
                line=line_num,
                column=col_num
            )
        
        return LeanError(
            error_type=LeanErrorType.SYSTEM_ERROR,
            message=line,
            file_path=filepath
        )
    
    def execute_tactic(self, tactic: str,
                      snapshot: Optional[LeanProofSnapshot] = None) -> LeanProofSnapshot:
        """
        执行策略
        
        注意：这是简化实现，实际应该与Lean服务器通信
        """
        if snapshot is None:
            snapshot = self.current_snapshot
        
        # 保存历史
        self.proof_history.append(snapshot)
        
        # 模拟策略执行（实际实现需要与Lean交互）
        new_snapshot = LeanProofSnapshot(
            state=LeanProofState.PROVING,
            current_goals=snapshot.current_goals.copy(),
            all_goals=snapshot.all_goals.copy(),
            tactic_history=snapshot.tactic_history.copy(),
            environment=snapshot.environment.copy()
        )
        
        # 记录策略执行
        tactic_info = LeanTacticInfo(
            tactic_name=tactic,
            tactic_state_before=str(snapshot.current_goals),
            tactic_state_after=str(new_snapshot.current_goals),
            goals_before=snapshot.current_goals,
            goals_after=new_snapshot.current_goals
        )
        new_snapshot.tactic_history.append(tactic_info)
        
        self.current_snapshot = new_snapshot
        return new_snapshot
    
    def undo(self, steps: int = 1) -> LeanProofSnapshot:
        """撤销策略"""
        for _ in range(steps):
            if len(self.proof_history) > 0:
                self.current_snapshot = self.proof_history.pop()
            else:
                break
        return self.current_snapshot
    
    def get_current_snapshot(self) -> LeanProofSnapshot:
        """获取当前证明快照"""
        return self.current_snapshot
    
    def start_proof(self, theorem_name: str, statement: str) -> LeanProofSnapshot:
        """开始新证明"""
        snapshot = LeanProofSnapshot(
            state=LeanProofState.PROVING,
            current_goals=[LeanGoal(
                goal_id=0,
                conclusion=statement,
                hypotheses=[]
            )]
        )
        self.current_snapshot = snapshot
        self.proof_history = []
        return snapshot
    
    def complete_proof(self) -> LeanProofSnapshot:
        """完成证明"""
        self.current_snapshot.state = LeanProofState.PROOF_COMPLETE
        self.current_snapshot.current_goals = []
        return self.current_snapshot
    
    def export_proof(self, format: str = "lean") -> str:
        """
        导出证明
        
        Args:
            format: 导出格式 (lean, json, text)
        
        Returns:
            导出的证明字符串
        """
        if format == "json":
            return json.dumps(self.current_snapshot.to_dict(), indent=2)
        elif format == "text":
            lines = ["Proof Summary:"]
            lines.append(f"State: {self.current_snapshot.state.value}")
            lines.append(f"Goals remaining: {self.current_snapshot.num_goals}")
            lines.append("\nTactics applied:")
            for i, tactic in enumerate(self.current_snapshot.tactic_history, 1):
                lines.append(f"  {i}. {tactic.tactic_name}")
            return "\n".join(lines)
        else:  # lean
            # 生成Lean代码
            builder = LeanCodeBuilder()
            for tactic_info in self.current_snapshot.tactic_history:
                builder.lines.append(tactic_info.tactic_name)
            return builder.to_string()


class LeanProofManager:
    """
    Lean证明管理器
    
    提供更高级的证明管理功能。
    """
    
    def __init__(self, lean_interface: Optional[LeanInterface] = None):
        self.lean = lean_interface or LeanInterface()
        self.suggested_tactics: List[str] = []
    
    def suggest_tactics(self, goal: LeanGoal) -> List[str]:
        """
        根据目标建议策略
        
        基于启发式规则的策略建议。
        """
        suggestions = []
        conclusion = goal.conclusion.lower()
        
        # 根据结论形式建议
        if "->" in conclusion or "→" in conclusion:
            suggestions.extend(["intro", "intro h", "intros"])
        
        if "forall" in conclusion or "∀" in conclusion:
            suggestions.append("intro")
        
        if "exists" in conclusion or "∃" in conclusion:
            suggestions.append("use")
        
        if "and" in conclusion or "∧" in conclusion:
            suggestions.extend(["constructor", "apply And.intro"])
        
        if "or" in conclusion or "∨" in conclusion:
            suggestions.extend(["left", "right", "apply Or.inl", "apply Or.inr"])
        
        if conclusion.startswith("False") or conclusion.startswith("⊥"):
            suggestions.extend(["contradiction", "exfalso"])
        
        if conclusion.startswith("True") or conclusion.startswith("⊤"):
            suggestions.append("trivial")
        
        if "=" in conclusion:
            suggestions.extend(["rfl", "simp", "rw"])
        
        # 通用策略
        suggestions.extend(["simp", "simp_all", "aesop", "tauto", "omega"])
        
        self.suggested_tactics = suggestions
        return suggestions
    
    def try_auto_prove(self) -> bool:
        """尝试自动证明"""
        auto_tactics = [
            "simp",
            "simp_all",
            "aesop",
            "tauto",
            "omega",
            "linarith",
            "nlinarith",
            "decide"
        ]
        
        for tactic in auto_tactics:
            snapshot = self.lean.execute_tactic(tactic)
            if snapshot.is_complete:
                return True
            self.lean.undo()
        
        return False
    
    def search_lemmas(self, pattern: str) -> List[str]:
        """
        搜索相关引理
        
        注意：这是简化实现
        """
        # 实际实现应该查询Lean环境
        common_lemmas = {
            "add": ["add_assoc", "add_comm", "add_zero", "zero_add"],
            "mul": ["mul_assoc", "mul_comm", "mul_one", "one_mul"],
            "eq": ["Eq.symm", "Eq.trans", "Eq.refl"],
            "le": ["le_refl", "le_trans", "le_antisymm"],
            "lt": ["lt_trans", "lt_of_le_of_lt", "lt_of_lt_of_le"]
        }
        
        results = []
        for key, lemmas in common_lemmas.items():
            if key in pattern.lower():
                results.extend(lemmas)
        
        return results
    
    def generate_proof_template(self, theorem_name: str,
                               statement: str) -> str:
        """生成证明模板"""
        builder = LeanCodeBuilder()
        
        builder.add_comment(f"Theorem: {theorem_name}", doc=True)
        builder.lines.append(f"theorem {theorem_name} : {statement} := by")
        builder.indent_level += 1
        
        # 根据陈述添加初始策略
        if "forall" in statement or "∀" in statement:
            builder.lines.append(f"{builder._indent()}intro x")
        
        if "->" in statement or "→" in statement:
            builder.lines.append(f"{builder._indent()}intro h")
        
        builder.lines.append(f"{builder._indent()}-- TODO: complete proof")
        builder.lines.append(f"{builder._indent()}sorry")
        builder.indent_level -= 1
        
        return builder.to_string()


# 便捷函数
def create_simple_theorem(name: str, statement: str,
                         tactics: List[str]) -> str:
    """创建简单定理证明"""
    builder = LeanCodeBuilder()
    builder.add_import("Mathlib")
    builder.add_tactic_proof(name, statement, tactics)
    return builder.to_string()


def create_inductive_type(name: str, constructors: List[Tuple[str, str]]) -> str:
    """创建归纳类型"""
    builder = LeanCodeBuilder()
    builder.add_inductive(name, [], "Type", constructors)
    return builder.to_string()


def create_structure(name: str, fields: List[Tuple[str, str]]) -> str:
    """创建结构体"""
    builder = LeanCodeBuilder()
    builder.add_structure(name, [], None, fields)
    return builder.to_string()


# 示例用法
if __name__ == "__main__":
    print("=== Lean代码生成示例 ===\n")
    
    # 简单定理证明
    script = create_simple_theorem(
        "add_comm",
        "∀ n m : Nat, n + m = m + n",
        ["intro n m", "induction n with", "| zero => simp",
         "| succ n ih => simp [ih, Nat.add_succ]"]
    )
    print(script)
    print("\n")
    
    # 归纳类型
    print("=== 归纳类型 ===\n")
    list_script = create_inductive_type(
        "MyList",
        [("nil", "MyList"), ("cons", "Nat → MyList → MyList")]
    )
    print(list_script)
    print("\n")
    
    # 结构体
    print("=== 结构体 ===\n")
    struct_script = create_structure(
        "Point",
        [("x", "Float"), ("y", "Float")]
    )
    print(struct_script)
    print("\n")
    
    # 复杂示例
    print("=== 复杂示例 ===\n")
    builder = LeanCodeBuilder()
    builder.add_import("Mathlib")
    builder.add_open("Nat")
    builder.add_namespace("MyNamespace")
    builder.add_variable("n", "Nat")
    builder.add_tactic_proof(
        "my_theorem",
        "n + 0 = n",
        ["induction n with", "| zero => rfl", "| succ n ih => simp [ih]"]
    )
    builder.end_namespace("MyNamespace")
    print(builder.to_string())
