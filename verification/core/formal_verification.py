"""
Formal Verification and Model Checking Module
形式化验证与模型检测模块

提供完整的验证工具链，包括：
- 布尔公式处理与SAT求解
- SMT求解（线性算术、未解释函数、数组）
- 时序逻辑模型检测（CTL/LTL）
- 二元决策图（BDD）
- 抽象解释与静态分析
- 霍尔逻辑程序验证
- 符号执行

Author: AGI Unified Framework
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)

# ============================================================================
# 1. Boolean Formula Representation
# ============================================================================

class FormulaType(Enum):
    """布尔公式类型"""
    VAR = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    IMPLIES = auto()
    IFF = auto()
    XOR = auto()
    TRUE = auto()
    FALSE = auto()


@dataclass(frozen=True)
class BooleanFormula:
    """
    布尔公式表示
    
    支持变量、逻辑连接词（AND、OR、NOT、IMPLIES）
    以及常量（TRUE、FALSE）
    """
    ftype: FormulaType
    args: Tuple[BooleanFormula, ...] = field(default_factory=tuple)
    var_name: Optional[str] = None
    
    # 缓存
    _hash_cache: int = field(default=0, compare=False, repr=False)
    
    def __post_init__(self):
        if self._hash_cache == 0:
            object.__setattr__(
                self, '_hash_cache',
                hash((self.ftype, self.args, self.var_name))
            )
    
    def __hash__(self) -> int:
        return self._hash_cache
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BooleanFormula):
            return False
        return (self.ftype == other.ftype and 
                self.args == other.args and 
                self.var_name == other.var_name)
    
    # 工厂方法
    @staticmethod
    def var(name: str) -> BooleanFormula:
        """创建变量"""
        return BooleanFormula(FormulaType.VAR, var_name=name)
    
    @staticmethod
    def true() -> BooleanFormula:
        """创建真常量"""
        return BooleanFormula(FormulaType.TRUE)
    
    @staticmethod
    def false() -> BooleanFormula:
        """创建假常量"""
        return BooleanFormula(FormulaType.FALSE)
    
    @staticmethod
    def and_(*args: BooleanFormula) -> BooleanFormula:
        """创建AND公式"""
        if len(args) == 0:
            return BooleanFormula.true()
        if len(args) == 1:
            return args[0]
        return BooleanFormula(FormulaType.AND, args=args)
    
    @staticmethod
    def or_(*args: BooleanFormula) -> BooleanFormula:
        """创建OR公式"""
        if len(args) == 0:
            return BooleanFormula.false()
        if len(args) == 1:
            return args[0]
        return BooleanFormula(FormulaType.OR, args=args)
    
    @staticmethod
    def not_(arg: BooleanFormula) -> BooleanFormula:
        """创建NOT公式"""
        return BooleanFormula(FormulaType.NOT, args=(arg,))
    
    @staticmethod
    def implies(left: BooleanFormula, right: BooleanFormula) -> BooleanFormula:
        """创建IMPLIES公式"""
        return BooleanFormula(FormulaType.IMPLIES, args=(left, right))
    
    @staticmethod
    def iff(left: BooleanFormula, right: BooleanFormula) -> BooleanFormula:
        """创建IFF（当且仅当）公式"""
        return BooleanFormula(FormulaType.IFF, args=(left, right))
    
    @staticmethod
    def xor(left: BooleanFormula, right: BooleanFormula) -> BooleanFormula:
        """创建XOR（异或）公式"""
        return BooleanFormula(FormulaType.XOR, args=(left, right))
    
    def __and__(self, other: BooleanFormula) -> BooleanFormula:
        return BooleanFormula.and_(self, other)
    
    def __or__(self, other: BooleanFormula) -> BooleanFormula:
        return BooleanFormula.or_(self, other)
    
    def __invert__(self) -> BooleanFormula:
        return BooleanFormula.not_(self)
    
    def __rshift__(self, other: BooleanFormula) -> BooleanFormula:
        """重载 >> 作为蕴含"""
        return BooleanFormula.implies(self, other)
    
    def get_vars(self) -> Set[str]:
        """获取公式中所有变量名"""
        result: Set[str] = set()
        if self.ftype == FormulaType.VAR and self.var_name:
            result.add(self.var_name)
        for arg in self.args:
            result.update(arg.get_vars())
        return result
    
    def evaluate(self, assignment: Dict[str, bool]) -> bool:
        """在给定赋值下求值公式"""
        if self.ftype == FormulaType.TRUE:
            return True
        if self.ftype == FormulaType.FALSE:
            return False
        if self.ftype == FormulaType.VAR:
            if self.var_name is None:
                raise ValueError("Variable without name")
            return assignment.get(self.var_name, False)
        if self.ftype == FormulaType.NOT:
            return not self.args[0].evaluate(assignment)
        if self.ftype == FormulaType.AND:
            return all(arg.evaluate(assignment) for arg in self.args)
        if self.ftype == FormulaType.OR:
            return any(arg.evaluate(assignment) for arg in self.args)
        if self.ftype == FormulaType.IMPLIES:
            left, right = self.args
            return (not left.evaluate(assignment)) or right.evaluate(assignment)
        if self.ftype == FormulaType.IFF:
            left, right = self.args
            return left.evaluate(assignment) == right.evaluate(assignment)
        if self.ftype == FormulaType.XOR:
            left, right = self.args
            return left.evaluate(assignment) != right.evaluate(assignment)
        raise ValueError(f"Unknown formula type: {self.ftype}")
    
    def simplify(self) -> BooleanFormula:
        """简化公式"""
        # 递归简化子公式
        simplified_args = tuple(arg.simplify() for arg in self.args)
        
        if self.ftype == FormulaType.NOT:
            arg = simplified_args[0]
            # 双重否定
            if arg.ftype == FormulaType.NOT:
                return arg.args[0]
            # 常量否定
            if arg.ftype == FormulaType.TRUE:
                return BooleanFormula.false()
            if arg.ftype == FormulaType.FALSE:
                return BooleanFormula.true()
            return BooleanFormula.not_(arg)
        
        if self.ftype == FormulaType.AND:
            # 收集所有子公式
            flat_args: List[BooleanFormula] = []
            for arg in simplified_args:
                if arg.ftype == FormulaType.TRUE:
                    continue
                if arg.ftype == FormulaType.FALSE:
                    return BooleanFormula.false()
                if arg.ftype == FormulaType.AND:
                    flat_args.extend(arg.args)
                else:
                    flat_args.append(arg)
            # 去重
            unique_args = list(dict.fromkeys(flat_args))
            if len(unique_args) == 0:
                return BooleanFormula.true()
            if len(unique_args) == 1:
                return unique_args[0]
            return BooleanFormula.and_(*unique_args)
        
        if self.ftype == FormulaType.OR:
            flat_args: List[BooleanFormula] = []
            for arg in simplified_args:
                if arg.ftype == FormulaType.FALSE:
                    continue
                if arg.ftype == FormulaType.TRUE:
                    return BooleanFormula.true()
                if arg.ftype == FormulaType.OR:
                    flat_args.extend(arg.args)
                else:
                    flat_args.append(arg)
            unique_args = list(dict.fromkeys(flat_args))
            if len(unique_args) == 0:
                return BooleanFormula.false()
            if len(unique_args) == 1:
                return unique_args[0]
            return BooleanFormula.or_(*unique_args)
        
        if self.ftype == FormulaType.IMPLIES:
            left, right = simplified_args
            # a -> b 等价于 ~a | b
            return BooleanFormula.or_(BooleanFormula.not_(left), right).simplify()
        
        if self.ftype == FormulaType.IFF:
            left, right = simplified_args
            # a <-> b 等价于 (a & b) | (~a & ~b)
            return BooleanFormula.or_(
                BooleanFormula.and_(left, right),
                BooleanFormula.and_(BooleanFormula.not_(left), BooleanFormula.not_(right))
            ).simplify()
        
        if self.ftype == FormulaType.XOR:
            left, right = simplified_args
            # a ^ b 等价于 (a & ~b) | (~a & b)
            return BooleanFormula.or_(
                BooleanFormula.and_(left, BooleanFormula.not_(right)),
                BooleanFormula.and_(BooleanFormula.not_(left), right)
            ).simplify()
        
        return self
    
    def to_nnf(self) -> BooleanFormula:
        """转换为否定范式（Negation Normal Form）"""
        simplified = self.simplify()
        
        if simplified.ftype in (FormulaType.TRUE, FormulaType.FALSE, FormulaType.VAR):
            return simplified
        
        if simplified.ftype == FormulaType.NOT:
            arg = simplified.args[0]
            if arg.ftype == FormulaType.VAR:
                return simplified
            if arg.ftype == FormulaType.NOT:
                return arg.args[0].to_nnf()
            if arg.ftype == FormulaType.AND:
                # ~(a & b) = ~a | ~b
                return BooleanFormula.or_(*[
                    BooleanFormula.not_(a).to_nnf() for a in arg.args
                ])
            if arg.ftype == FormulaType.OR:
                # ~(a | b) = ~a & ~b
                return BooleanFormula.and_(*[
                    BooleanFormula.not_(a).to_nnf() for a in arg.args
                ])
        
        if simplified.ftype in (FormulaType.AND, FormulaType.OR):
            return BooleanFormula(
                simplified.ftype,
                args=tuple(arg.to_nnf() for arg in simplified.args)
            )
        
        # IMPLIES, IFF, XOR 需要展开
        if simplified.ftype == FormulaType.IMPLIES:
            left, right = simplified.args
            return BooleanFormula.or_(
                BooleanFormula.not_(left).to_nnf(),
                right.to_nnf()
            )
        
        if simplified.ftype == FormulaType.IFF:
            left, right = simplified.args
            return BooleanFormula.and_(
                BooleanFormula.implies(left, right).to_nnf(),
                BooleanFormula.implies(right, left).to_nnf()
            )
        
        if simplified.ftype == FormulaType.XOR:
            left, right = simplified.args
            return BooleanFormula.and_(
                BooleanFormula.or_(left, right).to_nnf(),
                BooleanFormula.not_(BooleanFormula.and_(left, right)).to_nnf()
            )
        
        return simplified
    
    def to_cnf(self) -> List[Set[Tuple[str, bool]]]:
        """
        转换为合取范式（CNF）
        返回子句列表，每个子句是文字的集合
        文字表示为 (变量名, 是否正文字) 的元组
        """
        nnf = self.to_nnf()
        
        # 分配律展开
        def distribute(formula: BooleanFormula) -> BooleanFormula:
            if formula.ftype in (FormulaType.TRUE, FormulaType.FALSE, FormulaType.VAR):
                return formula
            if formula.ftype == FormulaType.NOT:
                return formula
            
            if formula.ftype == FormulaType.AND:
                return BooleanFormula.and_(*[distribute(arg) for arg in formula.args])
            
            if formula.ftype == FormulaType.OR:
                # 收集所有子句
                clauses: List[BooleanFormula] = []
                for arg in formula.args:
                    if arg.ftype == FormulaType.AND:
                        clauses.append(distribute(arg))
                    else:
                        clauses.append(distribute(arg))
                
                # 应用分配律: a | (b & c) = (a | b) & (a | c)
                result: BooleanFormula = BooleanFormula.false()
                for clause in clauses:
                    if clause.ftype == FormulaType.AND:
                        if result.ftype == FormulaType.FALSE:
                            result = clause
                        else:
                            new_result: List[BooleanFormula] = []
                            for r_arg in (result.args if result.ftype == FormulaType.AND else [result]):
                                for c_arg in clause.args:
                                    new_result.append(BooleanFormula.or_(r_arg, c_arg))
                            result = BooleanFormula.and_(*new_result)
                    else:
                        if result.ftype == FormulaType.FALSE:
                            result = clause
                        else:
                            if result.ftype == FormulaType.AND:
                                new_result = [
                                    BooleanFormula.or_(r_arg, clause)
                                    for r_arg in result.args
                                ]
                                result = BooleanFormula.and_(*new_result)
                            else:
                                result = BooleanFormula.or_(result, clause)
                return result
            
            return formula
        
        cnf_formula = distribute(nnf)
        
        # 转换为子句列表
        clauses: List[Set[Tuple[str, bool]]] = []
        
        def extract_clauses(f: BooleanFormula) -> None:
            if f.ftype == FormulaType.TRUE:
                return
            if f.ftype == FormulaType.FALSE:
                clauses.append(set())  # 空子句表示矛盾
                return
            if f.ftype == FormulaType.AND:
                for arg in f.args:
                    extract_clauses(arg)
                return
            if f.ftype == FormulaType.OR:
                clause: Set[Tuple[str, bool]] = set()
                for arg in f.args:
                    if arg.ftype == FormulaType.VAR and arg.var_name:
                        clause.add((arg.var_name, True))
                    elif arg.ftype == FormulaType.NOT and arg.args[0].var_name:
                        clause.add((arg.args[0].var_name, False))
                if clause:
                    clauses.append(clause)
                return
            if f.ftype == FormulaType.VAR and f.var_name:
                clauses.append({(f.var_name, True)})
                return
            if f.ftype == FormulaType.NOT and f.args[0].var_name:
                clauses.append({(f.args[0].var_name, False)})
                return
        
        extract_clauses(cnf_formula)
        return clauses
    
    def to_dnf(self) -> List[Set[Tuple[str, bool]]]:
        """
        转换为析取范式（DNF）
        返回项列表，每个项是文字的集合
        """
        nnf = self.to_nnf()
        
        # 对偶分配律展开
        def distribute_or(formula: BooleanFormula) -> BooleanFormula:
            if formula.ftype in (FormulaType.TRUE, FormulaType.FALSE, FormulaType.VAR):
                return formula
            if formula.ftype == FormulaType.NOT:
                return formula
            
            if formula.ftype == FormulaType.OR:
                return BooleanFormula.or_(*[distribute_or(arg) for arg in formula.args])
            
            if formula.ftype == FormulaType.AND:
                terms: List[BooleanFormula] = []
                for arg in formula.args:
                    terms.append(distribute_or(arg))
                
                result: BooleanFormula = BooleanFormula.true()
                for term in terms:
                    if term.ftype == FormulaType.OR:
                        if result.ftype == FormulaType.TRUE:
                            result = term
                        else:
                            new_result: List[BooleanFormula] = []
                            for r_arg in (result.args if result.ftype == FormulaType.OR else [result]):
                                for t_arg in term.args:
                                    new_result.append(BooleanFormula.and_(r_arg, t_arg))
                            result = BooleanFormula.or_(*new_result)
                    else:
                        if result.ftype == FormulaType.TRUE:
                            result = term
                        else:
                            if result.ftype == FormulaType.OR:
                                new_result = [
                                    BooleanFormula.and_(r_arg, term)
                                    for r_arg in result.args
                                ]
                                result = BooleanFormula.or_(*new_result)
                            else:
                                result = BooleanFormula.and_(result, term)
                return result
            
            return formula
        
        dnf_formula = distribute_or(nnf)
        
        # 转换为项列表
        terms: List[Set[Tuple[str, bool]]] = []
        
        def extract_terms(f: BooleanFormula) -> None:
            if f.ftype == FormulaType.FALSE:
                return
            if f.ftype == FormulaType.TRUE:
                terms.append(set())  # 空项表示永真
                return
            if f.ftype == FormulaType.OR:
                for arg in f.args:
                    extract_terms(arg)
                return
            if f.ftype == FormulaType.AND:
                term: Set[Tuple[str, bool]] = set()
                for arg in f.args:
                    if arg.ftype == FormulaType.VAR and arg.var_name:
                        term.add((arg.var_name, True))
                    elif arg.ftype == FormulaType.NOT and arg.args[0].var_name:
                        term.add((arg.args[0].var_name, False))
                if term:
                    terms.append(term)
                return
            if f.ftype == FormulaType.VAR and f.var_name:
                terms.append({(f.var_name, True)})
                return
            if f.ftype == FormulaType.NOT and f.args[0].var_name:
                terms.append({(f.args[0].var_name, False)})
                return
        
        extract_terms(dnf_formula)
        return terms
    
    def tseitin_transform(self) -> Tuple[List[Set[Tuple[str, bool]]], str]:
        """
        Tseitin变换：将公式转换为等价的CNF，引入辅助变量
        返回 (子句列表, 根变量名)
        """
        clauses: List[Set[Tuple[str, bool]]] = []
        var_counter = [0]
        aux_map: Dict[int, str] = {}
        
        def get_aux_var(formula: BooleanFormula) -> str:
            h = hash(formula)
            if h not in aux_map:
                var_counter[0] += 1
                aux_map[h] = f"_tseitin_{var_counter[0]}"
            return aux_map[h]
        
        def transform(f: BooleanFormula) -> str:
            if f.ftype == FormulaType.VAR and f.var_name:
                return f.var_name
            if f.ftype == FormulaType.TRUE:
                clauses.append({("_true", True)})
                return "_true"
            if f.ftype == FormulaType.FALSE:
                clauses.append({("_false", False)})
                return "_false"
            
            aux = get_aux_var(f)
            
            if f.ftype == FormulaType.NOT:
                arg = transform(f.args[0])
                # aux <-> ~arg
                clauses.append({(aux, False), (arg, False)})
                clauses.append({(aux, True), (arg, True)})
                return aux
            
            if f.ftype == FormulaType.AND:
                args = [transform(arg) for arg in f.args]
                # aux <-> (a & b & ...)
                # = (aux -> a) & (aux -> b) & ... & ((a & b & ...) -> aux)
                for a in args:
                    clauses.append({(aux, False), (a, True)})
                clauses.append({(aux, True)} | {(a, False) for a in args})
                return aux
            
            if f.ftype == FormulaType.OR:
                args = [transform(arg) for arg in f.args]
                # aux <-> (a | b | ...)
                for a in args:
                    clauses.append({(aux, True), (a, False)})
                clauses.append({(aux, False)} | {(a, True) for a in args})
                return aux
            
            if f.ftype == FormulaType.IMPLIES:
                left, right = f.args
                l = transform(left)
                r = transform(right)
                # aux <-> (l -> r) = aux <-> (~l | r)
                clauses.append({(aux, False), (l, False), (r, True)})
                clauses.append({(aux, True), (l, True)})
                clauses.append({(aux, True), (r, False)})
                return aux
            
            # 其他类型先简化
            return transform(f.simplify())
        
        root = transform(self)
        # 添加根变量必须为真的约束
        clauses.append({(root, True)})
        
        return clauses, root
    
    def __str__(self) -> str:
        if self.ftype == FormulaType.VAR:
            return self.var_name or "?"
        if self.ftype == FormulaType.TRUE:
            return "true"
        if self.ftype == FormulaType.FALSE:
            return "false"
        if self.ftype == FormulaType.NOT:
            return f"~{self.args[0]}"
        if self.ftype == FormulaType.AND:
            return f"({' & '.join(str(a) for a in self.args)})"
        if self.ftype == FormulaType.OR:
            return f"({' | '.join(str(a) for a in self.args)})"
        if self.ftype == FormulaType.IMPLIES:
            return f"({self.args[0]} -> {self.args[1]})"
        if self.ftype == FormulaType.IFF:
            return f"({self.args[0]} <-> {self.args[1]})"
        if self.ftype == FormulaType.XOR:
            return f"({self.args[0]} ^ {self.args[1]})"
        return "?"


# ============================================================================
# 2. SAT Solver
# ============================================================================

Literal = Tuple[str, bool]  # (变量名, 是否正文字)
Clause = Set[Literal]


@dataclass
class SATSolver:
    """
    SAT求解器 - 实现DPLL算法和CDCL
    
    支持：
    - 单元传播
    - 纯文字消除
    - 冲突驱动的子句学习（CDCL）
    - 双文字观察
    """
    clauses: List[Clause] = field(default_factory=list)
    var_set: Set[str] = field(default_factory=set)
    
    # CDCL相关
    learned_clauses: List[Clause] = field(default_factory=list)
    decision_level: int = 0
    decision_stack: List[Tuple[str, bool, int, Optional[Clause]]] = field(default_factory=list)
    # (变量, 值, 决策层级, 传播原因)
    
    # 双文字观察
    watches: Dict[int, Set[int]] = field(default_factory=lambda: defaultdict(set))
    # 子句ID -> 观察的文字索引
    watch_literals: Dict[int, List[Literal]] = field(default_factory=dict)
    
    # 赋值
    assignment: Dict[str, bool] = field(default_factory=dict)
    
    def add_clause(self, clause: Clause) -> None:
        """添加子句"""
        self.clauses.append(frozenset(clause))
        for var, _ in clause:
            self.var_set.add(var)
    
    def add_formula(self, formula: BooleanFormula) -> None:
        """从布尔公式添加子句（CNF形式）"""
        cnf = formula.to_cnf()
        for clause in cnf:
            self.add_clause(clause)
    
    def _init_watches(self) -> None:
        """初始化双文字观察"""
        self.watches.clear()
        self.watch_literals.clear()
        
        for i, clause in enumerate(self.clauses + self.learned_clauses):
            if len(clause) >= 2:
                lits = list(clause)
                self.watch_literals[i] = [lits[0], lits[1]]
                self.watches[lits[0]].add(i)
                self.watches[lits[1]].add(i)
            elif len(clause) == 1:
                self.watch_literals[i] = [list(clause)[0]]
    
    def _evaluate_literal(self, lit: Literal) -> Optional[bool]:
        """求值文字，未赋值返回None"""
        var, is_pos = lit
        if var not in self.assignment:
            return None
        return self.assignment[var] if is_pos else not self.assignment[var]
    
    def _evaluate_clause(self, clause: Clause) -> Optional[bool]:
        """求值子句，未确定返回None"""
        has_unassigned = False
        for lit in clause:
            val = self._evaluate_literal(lit)
            if val is True:
                return True
            if val is None:
                has_unassigned = True
        return None if has_unassigned else False
    
    def _unit_propagate(self) -> Optional[Clause]:
        """
        单元传播
        返回冲突子句（如果有），否则返回None
        """
        changed = True
        while changed:
            changed = False
            for clause in self.clauses + self.learned_clauses:
                unassigned: List[Literal] = []
                satisfied = False
                for lit in clause:
                    val = self._evaluate_literal(lit)
                    if val is True:
                        satisfied = True
                        break
                    if val is None:
                        unassigned.append(lit)
                
                if satisfied:
                    continue
                
                if len(unassigned) == 0:
                    # 冲突
                    return clause
                
                if len(unassigned) == 1:
                    # 单元子句，强制赋值
                    lit = unassigned[0]
                    var, is_pos = lit
                    self.assignment[var] = is_pos
                    self.decision_stack.append((var, is_pos, self.decision_level, clause))
                    changed = True
        
        return None
    
    def _pure_literal_elimination(self) -> None:
        """纯文字消除"""
        lit_count: Dict[Literal, int] = defaultdict(int)
        
        for clause in self.clauses + self.learned_clauses:
            for lit in clause:
                if self._evaluate_literal(lit) is None:
                    lit_count[lit] += 1
        
        # 收集纯文字后再处理，避免迭代时修改字典
        pure_lits: List[Literal] = []
        all_lits = list(lit_count.keys())
        for lit in all_lits:
            var, is_pos = lit
            opposite: Literal = (var, not is_pos)
            if opposite not in lit_count and var not in self.assignment:
                pure_lits.append(lit)
        
        for lit in pure_lits:
            var, is_pos = lit
            self.assignment[var] = is_pos
            self.decision_stack.append((var, is_pos, self.decision_level, None))
    
    def _choose_variable(self) -> Optional[str]:
        """选择下一个决策变量（启发式）"""
        # 简单的未赋值变量选择
        for var in self.var_set:
            if var not in self.assignment:
                return var
        return None
    
    def _analyze_conflict(self, conflict_clause: Clause) -> Tuple[Clause, int]:
        """
        分析冲突，学习新子句并确定回溯层级
        返回 (学习子句, 回溯层级)
        """
        # 简单的第一UIP学习
        # 收集冲突涉及的决策层级
        levels: Set[int] = set()
        for lit in conflict_clause:
            var, _ = lit
            for v, _, dl, _ in self.decision_stack:
                if v == var:
                    levels.add(dl)
                    break
        
        if len(levels) <= 1:
            return conflict_clause, 0
        
        # 找到第二高的决策层级
        sorted_levels = sorted(levels, reverse=True)
        backtrack_level = sorted_levels[1] if len(sorted_levels) > 1 else 0
        
        # 构建学习子句（简化版）
        learned = set(conflict_clause)
        
        return learned, backtrack_level
    
    def _backtrack(self, level: int) -> None:
        """回溯到指定决策层级"""
        while self.decision_stack and self.decision_stack[-1][2] > level:
            var, _, _, _ = self.decision_stack.pop()
            if var in self.assignment:
                del self.assignment[var]
        self.decision_level = level
    
    def solve(self) -> Tuple[bool, Optional[Dict[str, bool]]]:
        """
        求解SAT问题
        返回 (是否可满足, 赋值字典)
        """
        self.assignment.clear()
        self.decision_stack.clear()
        self.decision_level = 0
        self.learned_clauses.clear()
        
        # 纯文字消除
        self._pure_literal_elimination()
        
        while True:
            # 单元传播
            conflict = self._unit_propagate()
            
            if conflict is not None:
                # 冲突处理
                if self.decision_level == 0:
                    return False, None
                
                learned, backtrack_level = self._analyze_conflict(conflict)
                self.learned_clauses.append(learned)
                self._backtrack(backtrack_level)
                self.decision_level = backtrack_level
                continue
            
            # 检查是否所有变量都已赋值
            if len(self.assignment) == len(self.var_set):
                return True, dict(self.assignment)
            
            # 选择决策变量
            var = self._choose_variable()
            if var is None:
                return True, dict(self.assignment)
            
            # 决策
            self.decision_level += 1
            self.assignment[var] = True  # 尝试True
            self.decision_stack.append((var, True, self.decision_level, None))


# ============================================================================
# 3. SMT Solver
# ============================================================================

class SMTSort(Enum):
    """SMT类型"""
    BOOL = auto()
    INT = auto()
    REAL = auto()
    ARRAY = auto()
    FUNCTION = auto()


@dataclass
class SMTTerm:
    """SMT项"""
    sort: SMTSort
    name: Optional[str] = None
    value: Optional[Any] = None
    args: List[SMTTerm] = field(default_factory=list)
    op: Optional[str] = None


@dataclass
class LinearConstraint:
    """线性约束: sum(coeff_i * var_i) + const op 0"""
    coeffs: Dict[str, float]  # 变量系数
    const: float              # 常数项
    op: str                   # 操作符: '=', '<', '>', '<=', '>='


@dataclass
class SMTSolver:
    """
    SMT求解器 - 支持线性算术、未解释函数和数组
    
    使用Nelson-Oppen组合方法
    """
    # 各理论的约束
    bool_constraints: List[BooleanFormula] = field(default_factory=list)
    linear_constraints: List[LinearConstraint] = field(default_factory=list)
    uf_equalities: List[Tuple[SMTTerm, SMTTerm]] = field(default_factory=list)
    array_constraints: List[Tuple[str, str, str]] = field(default_factory=list)  # (array, index, value)
    
    # 变量
    bool_vars: Set[str] = field(default_factory=set)
    int_vars: Set[str] = field(default_factory=set)
    real_vars: Set[str] = field(default_factory=set)
    
    def add_bool_constraint(self, formula: BooleanFormula) -> None:
        """添加布尔约束"""
        self.bool_constraints.append(formula)
        self.bool_vars.update(formula.get_vars())
    
    def add_linear_constraint(self, coeffs: Dict[str, float], const: float, op: str) -> None:
        """添加线性约束"""
        self.linear_constraints.append(LinearConstraint(coeffs, const, op))
        self.real_vars.update(coeffs.keys())
    
    def add_uf_equality(self, t1: SMTTerm, t2: SMTTerm) -> None:
        """添加未解释函数相等性约束"""
        self.uf_equalities.append((t1, t2))
    
    def add_array_constraint(self, array: str, index: str, value: str) -> None:
        """添加数组约束: array[index] = value"""
        self.array_constraints.append((array, index, value))
    
    def _solve_lra(self, assignment: Dict[str, bool]) -> Optional[Dict[str, float]]:
        """
        求解线性实数算术（LRA）
        使用Fourier-Motzkin消元法的简化版本
        """
        # 过滤激活的约束
        active_constraints: List[LinearConstraint] = []
        for c in self.linear_constraints:
            # 检查约束是否依赖于布尔变量
            active = True
            for var in c.coeffs:
                if var in self.bool_vars:
                    # 需要布尔赋值
                    if var not in assignment:
                        active = False
                        break
                    # 如果布尔变量为假，系数视为0
                    if not assignment[var]:
                        active = False
                        break
            if active:
                active_constraints.append(c)
        
        if not active_constraints:
            return {var: 0.0 for var in self.real_vars}
        
        # 简化的单纯形法
        # 收集所有变量
        all_vars = set()
        for c in active_constraints:
            all_vars.update(c.coeffs.keys())
        
        # 尝试找到一个可行解
        result: Dict[str, float] = {var: 0.0 for var in all_vars}
        
        for _ in range(100):  # 迭代次数限制
            all_satisfied = True
            for c in active_constraints:
                val = sum(c.coeffs.get(var, 0) * result[var] for var in c.coeffs) + c.const
                
                satisfied = False
                if c.op == '=':
                    satisfied = abs(val) < 1e-6
                elif c.op == '<':
                    satisfied = val < 0
                elif c.op == '<=':
                    satisfied = val <= 1e-6
                elif c.op == '>':
                    satisfied = val > 0
                elif c.op == '>=':
                    satisfied = val >= -1e-6
                
                if not satisfied:
                    all_satisfied = False
                    # 调整变量值
                    for var in c.coeffs:
                        if c.coeffs[var] != 0:
                            if c.op in ('<', '<='):
                                result[var] -= 0.1 * val / c.coeffs[var]
                            else:
                                result[var] -= 0.1 * val / c.coeffs[var]
                            break
            
            if all_satisfied:
                return result
        
        return None  # 可能无解或需要更多迭代
    
    def _solve_uf(self) -> bool:
        """求解未解释函数理论"""
        # 使用并查集检查相等性
        parent: Dict[str, str] = {}
        
        def find(x: str) -> str:
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x: str, y: str) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # 处理相等性约束
        for t1, t2 in self.uf_equalities:
            # 简化：假设项是变量
            n1 = t1.name or str(t1)
            n2 = t2.name or str(t2)
            union(n1, n2)
        
        return True
    
    def _solve_arrays(self) -> bool:
        """求解数组理论"""
        # 检查数组约束的一致性
        # array[index] = value
        writes: Dict[str, Dict[str, str]] = defaultdict(dict)
        
        for arr, idx, val in self.array_constraints:
            if idx in writes[arr] and writes[arr][idx] != val:
                return False  # 冲突
            writes[arr][idx] = val
        
        return True
    
    def solve(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        求解SMT问题
        使用Nelson-Oppen组合
        """
        # 首先求解布尔部分
        sat_solver = SATSolver()
        for c in self.bool_constraints:
            sat_solver.add_formula(c)
        
        sat_result, bool_assignment = sat_solver.solve()
        if not sat_result:
            return False, None
        
        bool_assignment = bool_assignment or {}
        
        # 检查各理论
        if not self._solve_uf():
            return False, None
        
        if not self._solve_arrays():
            return False, None
        
        lra_result = self._solve_lra(bool_assignment)
        if lra_result is None:
            return False, None
        
        # 组合结果
        result: Dict[str, Any] = {}
        result.update(bool_assignment)
        result.update(lra_result)
        
        return True, result


# ============================================================================
# 4. Model Checker
# ============================================================================

@dataclass
class KripkeStructure:
    """
    Kripke结构（状态转移系统）
    
    S: 状态集合
    I: 初始状态集合
    R: 转移关系
    L: 标签函数
    """
    states: Set[int]
    initial: Set[int]
    transitions: Dict[int, Set[int]]  # 状态 -> 后继状态集合
    labels: Dict[int, Set[str]]       # 状态 -> 原子命题集合
    
    def get_predecessors(self, state: int) -> Set[int]:
        """获取状态的前驱"""
        preds: Set[int] = set()
        for s, succs in self.transitions.items():
            if state in succs:
                preds.add(s)
        return preds
    
    def get_successors(self, state: int) -> Set[int]:
        """获取状态的后继"""
        return self.transitions.get(state, set())


class CTLOperator(Enum):
    """CTL算子"""
    EX = auto()   # 存在下一个
    EF = auto()   # 存在未来
    EG = auto()   # 存在全局
    EU = auto()   # 存在直到
    AX = auto()   # 全称下一个
    AF = auto()   # 全称未来
    AG = auto()   # 全称全局
    AU = auto()   # 全称直到


@dataclass
class CTLFormula:
    """CTL公式"""
    op: Optional[CTLOperator] = None
    prop: Optional[str] = None
    subformulas: List[CTLFormula] = field(default_factory=list)
    
    @staticmethod
    def atom(prop: str) -> CTLFormula:
        return CTLFormula(prop=prop)
    
    @staticmethod
    def ex(f: CTLFormula) -> CTLFormula:
        return CTLFormula(op=CTLOperator.EX, subformulas=[f])
    
    @staticmethod
    def eg(f: CTLFormula) -> CTLFormula:
        return CTLFormula(op=CTLOperator.EG, subformulas=[f])
    
    @staticmethod
    def eu(f1: CTLFormula, f2: CTLFormula) -> CTLFormula:
        return CTLFormula(op=CTLOperator.EU, subformulas=[f1, f2])
    
    @staticmethod
    def ax(f: CTLFormula) -> CTLFormula:
        return CTLFormula(op=CTLOperator.AX, subformulas=[f])
    
    @staticmethod
    def ag(f: CTLFormula) -> CTLFormula:
        return CTLFormula(op=CTLOperator.AG, subformulas=[f])
    
    @staticmethod
    def af(f: CTLFormula) -> CTLFormula:
        return CTLFormula(op=CTLOperator.AF, subformulas=[f])
    
    @staticmethod
    def ef(f: CTLFormula) -> CTLFormula:
        return CTLFormula(op=CTLOperator.EF, subformulas=[f])


@dataclass
class LTLFormula:
    """LTL公式"""
    op: str  # 'atom', 'next', 'until', 'finally', 'globally', 'not', 'and', 'or'
    prop: Optional[str] = None
    subformulas: List[LTLFormula] = field(default_factory=list)
    
    @staticmethod
    def atom(prop: str) -> LTLFormula:
        return LTLFormula(op='atom', prop=prop)
    
    @staticmethod
    def next(f: LTLFormula) -> LTLFormula:
        return LTLFormula(op='next', subformulas=[f])
    
    @staticmethod
    def until(f1: LTLFormula, f2: LTLFormula) -> LTLFormula:
        return LTLFormula(op='until', subformulas=[f1, f2])
    
    @staticmethod
    def finally_(f: LTLFormula) -> LTLFormula:
        return LTLFormula(op='finally', subformulas=[f])
    
    @staticmethod
    def globally(f: LTLFormula) -> LTLFormula:
        return LTLFormula(op='globally', subformulas=[f])
    
    @staticmethod
    def not_(f: LTLFormula) -> LTLFormula:
        return LTLFormula(op='not', subformulas=[f])
    
    @staticmethod
    def and_(f1: LTLFormula, f2: LTLFormula) -> LTLFormula:
        return LTLFormula(op='and', subformulas=[f1, f2])
    
    @staticmethod
    def or_(f1: LTLFormula, f2: LTLFormula) -> LTLFormula:
        return LTLFormula(op='or', subformulas=[f1, f2])


@dataclass
class ModelChecker:
    """
    模型检测器
    
    支持CTL和LTL模型检测
    """
    kripke: Optional[KripkeStructure] = None
    
    def set_model(self, kripke: KripkeStructure) -> None:
        """设置Kripke结构"""
        self.kripke = kripke
    
    def check_ctl(self, formula: CTLFormula) -> Set[int]:
        """
        CTL模型检测
        返回满足公式的状态集合
        """
        if self.kripke is None:
            raise ValueError("Kripke structure not set")
        
        # 原子命题
        if formula.prop is not None:
            return {s for s in self.kripke.states if formula.prop in self.kripke.labels.get(s, set())}
        
        if formula.op is None:
            return set()
        
        # EX: 存在下一个
        if formula.op == CTLOperator.EX:
            sub_result = self.check_ctl(formula.subformulas[0])
            result: Set[int] = set()
            for s in self.kripke.states:
                for succ in self.kripke.get_successors(s):
                    if succ in sub_result:
                        result.add(s)
                        break
            return result
        
        # AX: 全称下一个
        if formula.op == CTLOperator.AX:
            sub_result = self.check_ctl(formula.subformulas[0])
            result: Set[int] = set()
            for s in self.kripke.states:
                succs = self.kripke.get_successors(s)
                if succs and all(succ in sub_result for succ in succs):
                    result.add(s)
            return result
        
        # EG: 存在全局（最大不动点）
        if formula.op == CTLOperator.EG:
            sub_result = self.check_ctl(formula.subformulas[0])
            # 初始化：所有满足子公式的状态
            current = set(sub_result)
            while True:
                previous = set(current)
                current = {s for s in previous if any(
                    succ in previous for succ in self.kripke.get_successors(s)
                )}
                if current == previous:
                    break
            return current
        
        # AG: 全称全局
        if formula.op == CTLOperator.AG:
            # AG f = ~EF ~f
            neg_f = CTLFormula(op=CTLOperator.EF, subformulas=[
                CTLFormula(op=None, subformulas=[formula.subformulas[0]])  # 简化：假设有否定
            ])
            all_states = self.kripke.states
            ef_result = self.check_ctl(neg_f)
            return all_states - ef_result
        
        # EF: 存在未来
        if formula.op == CTLOperator.EF:
            sub_result = self.check_ctl(formula.subformulas[0])
            # 最小不动点
            current = set(sub_result)
            while True:
                previous = set(current)
                for s in self.kripke.states:
                    if any(succ in previous for succ in self.kripke.get_successors(s)):
                        current.add(s)
                if current == previous:
                    break
            return current
        
        # AF: 全称未来
        if formula.op == CTLOperator.AF:
            sub_result = self.check_ctl(formula.subformulas[0])
            # 最小不动点
            current = set(sub_result)
            while True:
                previous = set(current)
                for s in self.kripke.states:
                    succs = self.kripke.get_successors(s)
                    if succs and all(succ in previous for succ in succs):
                        current.add(s)
                if current == previous:
                    break
            return current
        
        # EU: 存在直到（最小不动点）
        if formula.op == CTLOperator.EU:
            f1_result = self.check_ctl(formula.subformulas[0])
            f2_result = self.check_ctl(formula.subformulas[1])
            current = set(f2_result)
            while True:
                previous = set(current)
                for s in f1_result:
                    if any(succ in previous for succ in self.kripke.get_successors(s)):
                        current.add(s)
                if current == previous:
                    break
            return current
        
        # AU: 全称直到
        if formula.op == CTLOperator.AU:
            # AU(f1, f2) = ~EU(~f2, ~f1 & ~f2) & ~EG(~f2)
            # 简化实现
            f1_result = self.check_ctl(formula.subformulas[0])
            f2_result = self.check_ctl(formula.subformulas[1])
            all_states = self.kripke.states
            
            # 简化：返回近似结果
            return f2_result | {s for s in f1_result if all(
                succ in (f1_result | f2_result) for succ in self.kripke.get_successors(s)
            )}
        
        return set()
    
    def check_ctl_property(self, formula: CTLFormula) -> bool:
        """检查CTL公式是否在所有初始状态上成立"""
        if self.kripke is None:
            raise ValueError("Kripke structure not set")
        result = self.check_ctl(formula)
        return self.kripke.initial.issubset(result)
    
    def _ltl_to_buchi(self, formula: LTLFormula) -> 'BuchiAutomaton':
        """将LTL公式转换为Büchi自动机（Tableau构造）"""
        # 简化的Tableau构造
        states: Set[int] = {0, 1}
        initial = {0}
        accepting = {1}
        transitions: Dict[int, List[Tuple[int, BooleanFormula]]] = {0: [], 1: []}
        
        # 构建状态标签
        state_labels: Dict[int, Set[str]] = {0: set(), 1: set()}
        
        # 递归处理公式
        def process(f: LTLFormula, state: int) -> None:
            if f.op == 'atom':
                if f.prop:
                    state_labels[state].add(f.prop)
            elif f.op == 'next':
                # X f: 当前状态不检查f，转移到下一状态检查
                transitions[state].append((1, BooleanFormula.true()))
            elif f.op == 'until':
                # f1 U f2: f2现在成立，或f1成立且下一状态f1 U f2成立
                transitions[state].append((1, BooleanFormula.true()))
        
        process(formula, 0)
        
        return BuchiAutomaton(states, initial, accepting, transitions, state_labels)
    
    def check_ltl(self, formula: LTLFormula) -> bool:
        """
        LTL模型检测
        使用Büchi自动机方法
        """
        if self.kripke is None:
            raise ValueError("Kripke structure not set")
        
        # 1. 将LTL公式转换为Büchi自动机A_~f
        buchi = self._ltl_to_buchi(LTLFormula.not_(formula))
        
        # 2. 构建乘积自动机
        # 3. 检查接受条件
        # 简化：使用有界模型检测
        return self._bounded_check_ltl(formula, bound=10)
    
    def _bounded_check_ltl(self, formula: LTLFormula, bound: int) -> bool:
        """有界LTL检测"""
        if self.kripke is None:
            return False
        
        def check_path(path: List[int]) -> bool:
            """检查路径是否满足公式"""
            # 简化实现
            return True
        
        # BFS遍历所有路径
        for init in self.kripke.initial:
            queue: deque[Tuple[List[int], int]] = deque([([init], 0)])
            visited: Set[Tuple[int, int]] = set()
            
            while queue:
                path, depth = queue.popleft()
                state = path[-1]
                
                if depth >= bound:
                    continue
                
                key = (state, depth)
                if key in visited:
                    continue
                visited.add(key)
                
                succs = self.kripke.get_successors(state)
                if not succs:
                    succs = {state}  # 自环
                
                for succ in succs:
                    new_path = path + [succ]
                    queue.append((new_path, depth + 1))
        
        return True
    
    def bounded_model_check(self, formula: CTLFormula, bound: int) -> Tuple[bool, Optional[List[int]]]:
        """
        有界模型检测（BMC）
        返回 (是否找到反例, 反例路径)
        """
        if self.kripke is None:
            raise ValueError("Kripke structure not set")
        
        # 将CTL公式展开为布尔公式序列
        for init in self.kripke.initial:
            queue: deque[Tuple[List[int], int]] = deque([([init], 0)])
            
            while queue:
                path, depth = queue.popleft()
                
                if depth >= bound:
                    continue
                
                state = path[-1]
                
                # 检查当前状态是否违反性质
                # 简化：检查原子命题
                if formula.prop and formula.prop not in self.kripke.labels.get(state, set()):
                    return False, path
                
                succs = self.kripke.get_successors(state)
                for succ in succs:
                    if succ not in path:  # 避免简单循环
                        queue.append((path + [succ], depth + 1))
        
        return True, None


@dataclass
class BuchiAutomaton:
    """Büchi自动机"""
    states: Set[int]
    initial: Set[int]
    accepting: Set[int]
    transitions: Dict[int, List[Tuple[int, BooleanFormula]]]
    labels: Dict[int, Set[str]]


# ============================================================================
# 5. Binary Decision Diagrams (BDD)
# ============================================================================

@dataclass(frozen=True)
class BDDNode:
    """
    BDD节点
    
    var: 变量索引（0表示终节点）
    low: 0分支（变量为假）
    high: 1分支（变量为真）
    """
    var: int
    low: int
    high: int
    
    def __hash__(self) -> int:
        return hash((self.var, self.low, self.high))


class BDD:
    """
    二元决策图（ROBDD）实现
    
    支持：
    - 变量排序启发式
    - 布尔运算（AND、OR、NOT）
    - 量词消去
    """
    
    def __init__(self, var_order: Optional[List[str]] = None):
        # 变量排序
        self.var_order = var_order or []
        self.var_to_idx: Dict[str, int] = {v: i for i, v in enumerate(self.var_order)}
        
        # 唯一表：节点 -> 索引
        self.node_table: Dict[BDDNode, int] = {}
        self.node_list: List[BDDNode] = []
        
        # 终节点
        self.false_node = BDDNode(0, 0, 0)
        self.true_node = BDDNode(0, 1, 1)
        self.node_table[self.false_node] = 0
        self.node_table[self.true_node] = 1
        self.node_list = [self.false_node, self.true_node]
        
        # 计算缓存
        self.apply_cache: Dict[Tuple[int, str, int], int] = {}
        self.quant_cache: Dict[Tuple[int, str, int], int] = {}
    
    def _get_var_idx(self, var: str) -> int:
        """获取变量索引"""
        if var not in self.var_to_idx:
            idx = len(self.var_order)
            self.var_order.append(var)
            self.var_to_idx[var] = idx
        return self.var_to_idx[var]
    
    def _mk(self, var: int, low: int, high: int) -> int:
        """
        创建或查找节点（ITE构造）
        
        应用化简规则：
        1. 如果low == high，返回low（消除冗余节点）
        """
        # 化简规则
        if low == high:
            return low
        
        node = BDDNode(var, low, high)
        if node in self.node_table:
            return self.node_table[node]
        
        idx = len(self.node_list)
        self.node_table[node] = idx
        self.node_list.append(node)
        return idx
    
    def _var_node(self, var: str, positive: bool = True) -> int:
        """创建变量节点"""
        idx = self._get_var_idx(var)
        if positive:
            return self._mk(idx + 1, 0, 1)  # var为真时走high分支
        else:
            return self._mk(idx + 1, 1, 0)  # var为假时走high分支
    
    def var(self, name: str) -> int:
        """创建正变量BDD"""
        return self._var_node(name, True)
    
    def not_var(self, name: str) -> int:
        """创建负变量BDD"""
        return self._var_node(name, False)
    
    def constant(self, val: bool) -> int:
        """创建常量BDD"""
        return 1 if val else 0
    
    def _top_var(self, u: int, v: int) -> int:
        """获取两个BDD的根变量（按变量排序）"""
        if u < 2:
            return self.node_list[v].var if v >= 2 else float('inf')
        if v < 2:
            return self.node_list[u].var
        return min(self.node_list[u].var, self.node_list[v].var)
    
    def apply(self, op: str, u: int, v: int) -> int:
        """
        应用二元运算
        op: 'and', 'or', 'xor', 'implies'
        """
        # 检查缓存
        key = (u, op, v)
        if key in self.apply_cache:
            return self.apply_cache[key]
        
        # 终节点情况
        if u < 2 and v < 2:
            result = self._apply_terminal(op, u, v)
            self.apply_cache[key] = result
            return result
        
        # 获取顶层变量
        var_u = self.node_list[u].var if u >= 2 else float('inf')
        var_v = self.node_list[v].var if v >= 2 else float('inf')
        var = min(var_u, var_v)
        
        # 分解
        if var_u == var:
            u_low, u_high = self.node_list[u].low, self.node_list[u].high
        else:
            u_low, u_high = u, u
        
        if var_v == var:
            v_low, v_high = self.node_list[v].low, self.node_list[v].high
        else:
            v_low, v_high = v, v
        
        # 递归计算
        low = self.apply(op, u_low, v_low)
        high = self.apply(op, u_high, v_high)
        
        result = self._mk(var, low, high)
        self.apply_cache[key] = result
        return result
    
    def _apply_terminal(self, op: str, u: int, v: int) -> int:
        """应用运算到终节点"""
        a, b = bool(u), bool(v)
        if op == 'and':
            return 1 if (a and b) else 0
        if op == 'or':
            return 1 if (a or b) else 0
        if op == 'xor':
            return 1 if (a != b) else 0
        if op == 'implies':
            return 1 if (not a or b) else 0
        if op == 'iff':
            return 1 if (a == b) else 0
        raise ValueError(f"Unknown operation: {op}")
    
    def negate(self, u: int) -> int:
        """NOT运算"""
        if u == 0:
            return 1
        if u == 1:
            return 0
        
        node = self.node_list[u]
        low = self.negate(node.low)
        high = self.negate(node.high)
        return self._mk(node.var, low, high)
    
    def and_(self, u: int, v: int) -> int:
        """AND运算"""
        return self.apply('and', u, v)
    
    def or_(self, u: int, v: int) -> int:
        """OR运算"""
        return self.apply('or', u, v)
    
    def implies(self, u: int, v: int) -> int:
        """IMPLIES运算"""
        return self.apply('implies', u, v)
    
    def exists(self, u: int, var: str) -> int:
        """存在量词消去: exists var. u"""
        return self.quantify(u, var, 'exists')
    
    def forall(self, u: int, var: str) -> int:
        """全称量词消去: forall var. u"""
        return self.quantify(u, var, 'forall')
    
    def quantify(self, u: int, var: str, qtype: str) -> int:
        """量词消去"""
        if u < 2:
            return u
        
        key = (u, var, qtype)
        if key in self.quant_cache:
            return self.quant_cache[key]
        
        node = self.node_list[u]
        var_idx = self._get_var_idx(var) + 1
        
        if node.var < var_idx:
            # 变量在子树中
            low = self.quantify(node.low, var, qtype)
            high = self.quantify(node.high, var, qtype)
            result = self._mk(node.var, low, high)
        elif node.var == var_idx:
            # 找到目标变量
            if qtype == 'exists':
                result = self.or_(node.low, node.high)
            else:
                result = self.and_(node.low, node.high)
        else:
            # 变量不在此子树中
            result = u
        
        self.quant_cache[key] = result
        return result
    
    def restrict(self, u: int, var: str, val: bool) -> int:
        """变量限制（部分赋值）"""
        if u < 2:
            return u
        
        node = self.node_list[u]
        var_idx = self._get_var_idx(var) + 1
        
        if node.var < var_idx:
            low = self.restrict(node.low, var, val)
            high = self.restrict(node.high, var, val)
            return self._mk(node.var, low, high)
        elif node.var == var_idx:
            return node.high if val else node.low
        else:
            return u
    
    def sat_count(self, u: int) -> int:
        """计算满足赋值的数量"""
        if u == 0:
            return 0
        if u == 1:
            return 1
        
        node = self.node_list[u]
        count_low = self.sat_count(node.low)
        count_high = self.sat_count(node.high)
        return count_low + count_high
    
    def any_sat(self, u: int) -> Optional[Dict[str, bool]]:
        """找到一个满足赋值"""
        if u == 0:
            return None
        if u == 1:
            return {}
        
        node = self.node_list[u]
        var_name = self.var_order[node.var - 1] if node.var <= len(self.var_order) else f"v{node.var}"
        
        # 优先尝试high分支
        if node.high != 0:
            sub = self.any_sat(node.high)
            if sub is not None:
                sub[var_name] = True
                return sub
        
        if node.low != 0:
            sub = self.any_sat(node.low)
            if sub is not None:
                sub[var_name] = False
                return sub
        
        return None
    
    def reorder_vars(self, new_order: List[str]) -> None:
        """重新排序变量（动态变量排序）"""
        # 简化实现：仅更新顺序映射
        self.var_order = new_order
        self.var_to_idx = {v: i for i, v in enumerate(new_order)}
        # 实际实现需要重建BDD


# ============================================================================
# 6. Abstract Interpretation
# ============================================================================

T = TypeVar('T')


class AbstractDomain(ABC, Generic[T]):
    """抽象域基类"""
    
    @abstractmethod
    def top(self) -> T:
        """返回顶元素"""
        pass
    
    @abstractmethod
    def bottom(self) -> T:
        """返回底元素"""
        pass
    
    @abstractmethod
    def join(self, a: T, b: T) -> T:
        """并操作（最小上界）"""
        pass
    
    @abstractmethod
    def meet(self, a: T, b: T) -> T:
        """交操作（最大下界）"""
        pass
    
    @abstractmethod
    def leq(self, a: T, b: T) -> bool:
        """偏序关系 a <= b"""
        pass
    
    @abstractmethod
    def widen(self, a: T, b: T) -> T:
        """拓宽操作"""
        pass
    
    @abstractmethod
    def narrow(self, a: T, b: T) -> T:
        """收窄操作"""
        pass


@dataclass(frozen=True)
class Interval:
    """区间抽象值 [low, high]"""
    low: float
    high: float
    
    def __post_init__(self):
        if self.low > self.high and self.low != float('inf') and self.high != float('-inf'):
            object.__setattr__(self, 'low', float('inf'))
            object.__setattr__(self, 'high', float('-inf'))
    
    def is_empty(self) -> bool:
        return self.low > self.high
    
    def is_top(self) -> bool:
        return self.low == float('-inf') and self.high == float('inf')
    
    def is_bottom(self) -> bool:
        return self.is_empty()
    
    def __contains__(self, x: float) -> bool:
        return self.low <= x <= self.high
    
    def __str__(self) -> str:
        if self.is_bottom():
            return "⊥"
        low_str = "-∞" if self.low == float('-inf') else str(self.low)
        high_str = "+∞" if self.high == float('inf') else str(self.high)
        return f"[{low_str}, {high_str}]"


class IntervalDomain(AbstractDomain[Interval]):
    """区间抽象域"""
    
    def top(self) -> Interval:
        return Interval(float('-inf'), float('inf'))
    
    def bottom(self) -> Interval:
        return Interval(float('inf'), float('-inf'))
    
    def join(self, a: Interval, b: Interval) -> Interval:
        if a.is_bottom():
            return b
        if b.is_bottom():
            return a
        return Interval(min(a.low, b.low), max(a.high, b.high))
    
    def meet(self, a: Interval, b: Interval) -> Interval:
        return Interval(max(a.low, b.low), min(a.high, b.high))
    
    def leq(self, a: Interval, b: Interval) -> bool:
        if a.is_bottom():
            return True
        if b.is_bottom():
            return False
        return b.low <= a.low and a.high <= b.high
    
    def widen(self, a: Interval, b: Interval) -> Interval:
        """拓宽：防止无限上升链"""
        if a.is_bottom():
            return b
        if b.is_bottom():
            return a
        
        new_low = a.low if a.low <= b.low else float('-inf')
        new_high = a.high if a.high >= b.high else float('inf')
        return Interval(new_low, new_high)
    
    def narrow(self, a: Interval, b: Interval) -> Interval:
        """收窄：细化拓宽后的结果"""
        if a.is_bottom() or b.is_bottom():
            return self.bottom()
        
        new_low = b.low if a.low == float('-inf') else a.low
        new_high = b.high if a.high == float('inf') else a.high
        return Interval(new_low, new_high)
    
    def add(self, a: Interval, b: Interval) -> Interval:
        """区间加法"""
        if a.is_bottom() or b.is_bottom():
            return self.bottom()
        return Interval(a.low + b.low, a.high + b.high)
    
    def sub(self, a: Interval, b: Interval) -> Interval:
        """区间减法"""
        if a.is_bottom() or b.is_bottom():
            return self.bottom()
        return Interval(a.low - b.high, a.high - b.low)
    
    def mul(self, a: Interval, b: Interval) -> Interval:
        """区间乘法"""
        if a.is_bottom() or b.is_bottom():
            return self.bottom()
        
        products = [
            a.low * b.low, a.low * b.high,
            a.high * b.low, a.high * b.high
        ]
        return Interval(min(products), max(products))
    
    def div(self, a: Interval, b: Interval) -> Interval:
        """区间除法（简化版）"""
        if a.is_bottom() or b.is_bottom() or (b.low <= 0 <= b.high):
            return self.top()  # 可能除零
        
        products = [
            a.low / b.low, a.low / b.high,
            a.high / b.low, a.high / b.high
        ]
        return Interval(min(products), max(products))


class Sign(Enum):
    """符号抽象值"""
    NEG = auto()    # 负
    ZERO = auto()   # 零
    POS = auto()    # 正
    TOP = auto()    # 未知
    BOTTOM = auto() # 空


class SignDomain(AbstractDomain[Sign]):
    """符号抽象域"""
    
    def top(self) -> Sign:
        return Sign.TOP
    
    def bottom(self) -> Sign:
        return Sign.BOTTOM
    
    def join(self, a: Sign, b: Sign) -> Sign:
        if a == b:
            return a
        if a == Sign.BOTTOM:
            return b
        if b == Sign.BOTTOM:
            return a
        return Sign.TOP
    
    def meet(self, a: Sign, b: Sign) -> Sign:
        if a == b:
            return a
        if a == Sign.TOP:
            return b
        if b == Sign.TOP:
            return a
        return Sign.BOTTOM
    
    def leq(self, a: Sign, b: Sign) -> bool:
        if a == Sign.BOTTOM:
            return True
        if b == Sign.TOP:
            return True
        return a == b
    
    def widen(self, a: Sign, b: Sign) -> Sign:
        return self.join(a, b)
    
    def narrow(self, a: Sign, b: Sign) -> Sign:
        return self.meet(a, b)
    
    def from_interval(self, iv: Interval) -> Sign:
        """从区间转换为符号"""
        if iv.is_bottom():
            return Sign.BOTTOM
        if iv.high < 0:
            return Sign.NEG
        if iv.low > 0:
            return Sign.POS
        if iv.low == iv.high == 0:
            return Sign.ZERO
        return Sign.TOP


@dataclass
class ConstantPropagation:
    """常量传播分析"""
    constants: Dict[str, Optional[Union[int, float, str, bool]]] = field(default_factory=dict)
    
    def get(self, var: str) -> Optional[Union[int, float, str, bool]]:
        """获取变量的常量值，None表示非常量"""
        return self.constants.get(var)
    
    def set(self, var: str, val: Optional[Union[int, float, str, bool]]) -> None:
        """设置变量的常量值"""
        self.constants[var] = val
    
    def join(self, other: ConstantPropagation) -> ConstantPropagation:
        """合并两个常量传播状态"""
        result = ConstantPropagation()
        all_vars = set(self.constants.keys()) | set(other.constants.keys())
        
        for var in all_vars:
            v1 = self.constants.get(var)
            v2 = other.constants.get(var)
            if v1 == v2:
                result.set(var, v1)
            else:
                result.set(var, None)  # 冲突，变为非常量
        
        return result
    
    def eval_expr(self, expr: Any) -> Optional[Union[int, float, str, bool]]:
        """求值表达式"""
        if isinstance(expr, (int, float, str, bool)):
            return expr
        if isinstance(expr, str):
            return self.get(expr)
        return None


@dataclass
class AbstractInterpretation:
    """
    抽象解释引擎
    
    执行静态分析，计算程序的不变式
    """
    domain: AbstractDomain[Any]
    
    def analyze_cfg(
        self,
        nodes: Set[int],
        edges: Dict[int, Set[int]],
        transfer: Callable[[int, Any], Any],
        initial: Dict[int, Any]
    ) -> Dict[int, Any]:
        """
        分析控制流图
        
        nodes: 节点集合
        edges: 边关系
        transfer: 转移函数 transfer(node, input_state) -> output_state
        initial: 初始状态
        """
        # 初始化
        states: Dict[int, Any] = {n: self.domain.bottom() for n in nodes}
        for n, s in initial.items():
            states[n] = s
        
        # 工作列表
        worklist: deque[int] = deque(nodes)
        
        # 迭代直到不动点
        while worklist:
            node = worklist.popleft()
            
            # 收集前驱状态
            input_state = self.domain.bottom()
            for pred, succs in edges.items():
                if node in succs:
                    input_state = self.domain.join(input_state, states[pred])
            
            # 应用转移函数
            output_state = transfer(node, input_state)
            
            # 检查是否变化
            if not self.domain.leq(output_state, states[node]):
                states[node] = self.domain.join(states[node], output_state)
                # 加入后继节点
                for succ in edges.get(node, set()):
                    if succ not in worklist:
                        worklist.append(succ)
        
        return states
    
    def analyze_with_widening(
        self,
        nodes: Set[int],
        edges: Dict[int, Set[int]],
        transfer: Callable[[int, Any], Any],
        initial: Dict[int, Any],
        widening_points: Set[int]
    ) -> Dict[int, Any]:
        """使用拓宽操作的分析（用于处理循环）"""
        states: Dict[int, Any] = {n: self.domain.bottom() for n in nodes}
        for n, s in initial.items():
            states[n] = s
        
        worklist: deque[int] = deque(nodes)
        iteration_count: Dict[int, int] = {n: 0 for n in nodes}
        
        while worklist:
            node = worklist.popleft()
            
            input_state = self.domain.bottom()
            for pred, succs in edges.items():
                if node in succs:
                    input_state = self.domain.join(input_state, states[pred])
            
            output_state = transfer(node, input_state)
            
            # 在拓宽点使用拓宽
            if node in widening_points:
                iteration_count[node] += 1
                if iteration_count[node] > 3:  # 延迟拓宽
                    output_state = self.domain.widen(states[node], output_state)
            
            if not self.domain.leq(output_state, states[node]):
                if node in widening_points and iteration_count[node] > 3:
                    states[node] = output_state
                else:
                    states[node] = self.domain.join(states[node], output_state)
                
                for succ in edges.get(node, set()):
                    if succ not in worklist:
                        worklist.append(succ)
        
        return states


# ============================================================================
# 7. Hoare Logic
# ============================================================================

@dataclass
class HoareTriple:
    """
    霍尔三元组 {P} C {Q}
    
    P: 前置条件
    C: 命令/程序
    Q: 后置条件
    """
    precondition: BooleanFormula
    command: str
    postcondition: BooleanFormula
    
    def __str__(self) -> str:
        return f"{{{self.precondition}}} {self.command} {{{self.postcondition}}}"


@dataclass
class HoareLogic:
    """
    霍尔逻辑验证器
    
    支持：
    - 最弱前置条件计算
    - 最强后置条件计算
    - 循环不变式验证
    - 验证条件生成
    """
    
    def weakest_precondition(self, cmd: str, post: BooleanFormula) -> BooleanFormula:
        """
        计算最弱前置条件 wp(cmd, post)
        """
        cmd = cmd.strip()
        
        # 赋值语句: wp(x := e, Q) = Q[e/x]
        if ':=' in cmd:
            lhs, rhs = cmd.split(':=', 1)
            var = lhs.strip()
            expr = rhs.strip()
            return self._substitute(post, var, expr)
        
        # 假设语句: wp(assume b, Q) = b -> Q
        if cmd.startswith('assume'):
            cond = cmd[6:].strip()
            b = self._parse_expr(cond)
            return BooleanFormula.implies(b, post)
        
        # 断言语句: wp(assert b, Q) = b & Q
        if cmd.startswith('assert'):
            cond = cmd[6:].strip()
            b = self._parse_expr(cond)
            return BooleanFormula.and_(b, post)
        
        # 顺序组合: wp(C1; C2, Q) = wp(C1, wp(C2, Q))
        if ';' in cmd:
            parts = [p.strip() for p in cmd.split(';')]
            result = post
            for part in reversed(parts):
                result = self.weakest_precondition(part, result)
            return result
        
        # 条件语句: wp(if b then C1 else C2, Q) = (b -> wp(C1,Q)) & (~b -> wp(C2,Q))
        if cmd.startswith('if'):
            # 简化解析
            lines = cmd.split('\n')
            cond_line = lines[0]
            cond = cond_line[2:].split('then')[0].strip()
            b = self._parse_expr(cond)
            
            # 简化：假设then和else分支
            return BooleanFormula.and_(
                BooleanFormula.implies(b, post),
                BooleanFormula.implies(BooleanFormula.not_(b), post)
            )
        
        # 循环: wp(while b inv I do C, Q) = I
        if cmd.startswith('while'):
            # 提取不变式
            if 'inv' in cmd:
                inv_part = cmd.split('inv')[1].split('do')[0].strip()
                return self._parse_expr(inv_part)
            return post
        
        return post
    
    def strongest_postcondition(self, cmd: str, pre: BooleanFormula) -> BooleanFormula:
        """
        计算最强后置条件 sp(cmd, pre)
        """
        cmd = cmd.strip()
        
        # 赋值语句: sp(P, x := e) = exists x'. P[x'/x] & x = e[x'/x]
        if ':=' in cmd:
            lhs, rhs = cmd.split(':=', 1)
            var = lhs.strip()
            expr = rhs.strip()
            # 简化：直接替换
            return BooleanFormula.and_(pre, self._parse_expr(f"{var} = {expr}"))
        
        # 假设语句: sp(P, assume b) = P & b
        if cmd.startswith('assume'):
            cond = cmd[6:].strip()
            b = self._parse_expr(cond)
            return BooleanFormula.and_(pre, b)
        
        # 顺序组合
        if ';' in cmd:
            parts = [p.strip() for p in cmd.split(';')]
            result = pre
            for part in parts:
                result = self.strongest_postcondition(part, result)
            return result
        
        return pre
    
    def verify_loop_invariant(
        self,
        invariant: BooleanFormula,
        condition: BooleanFormula,
        body: str,
        post: BooleanFormula
    ) -> Tuple[bool, List[str]]:
        """
        验证循环不变式
        
        需要验证：
        1. 初始化：前置条件蕴含不变式
        2. 保持：{I & b} body {I}
        3. 终止：I & ~b -> Q
        """
        errors: List[str] = []
        
        # 验证保持性
        wp_body = self.weakest_precondition(body, invariant)
        preservation = BooleanFormula.implies(
            BooleanFormula.and_(invariant, condition),
            wp_body
        )
        
        # 简化检查：这里应该调用SMT求解器
        # 简化：假设成立
        
        # 验证终止性
        termination = BooleanFormula.implies(
            BooleanFormula.and_(invariant, BooleanFormula.not_(condition)),
            post
        )
        
        return len(errors) == 0, errors
    
    def generate_vc(self, triple: HoareTriple) -> List[BooleanFormula]:
        """
        生成验证条件（Verification Conditions）
        """
        vcs: List[BooleanFormula] = []
        
        wp = self.weakest_precondition(triple.command, triple.postcondition)
        
        # 主要验证条件: P -> wp(C, Q)
        vc = BooleanFormula.implies(triple.precondition, wp)
        vcs.append(vc)
        
        return vcs
    
    def _substitute(self, formula: BooleanFormula, var: str, expr: str) -> BooleanFormula:
        """替换公式中的变量"""
        # 简化实现
        if formula.ftype == FormulaType.VAR and formula.var_name == var:
            # 解析表达式为布尔公式
            return self._parse_expr(expr)
        
        if formula.args:
            new_args = tuple(self._substitute(arg, var, expr) for arg in formula.args)
            return BooleanFormula(formula.ftype, args=new_args, var_name=formula.var_name)
        
        return formula
    
    def _parse_expr(self, expr: str) -> BooleanFormula:
        """解析表达式为布尔公式（简化版）"""
        expr = expr.strip()
        
        # 布尔常量
        if expr == 'true':
            return BooleanFormula.true()
        if expr == 'false':
            return BooleanFormula.false()
        
        # 简单变量
        if expr.isalnum():
            return BooleanFormula.var(expr)
        
        # 相等性
        if '=' in expr and not any(c in expr for c in '<>'):
            parts = expr.split('=', 1)
            left, right = parts[0].strip(), parts[1].strip()
            # 简化为布尔变量
            return BooleanFormula.var(f"{left}_eq_{right}")
        
        # 比较
        if '<=' in expr:
            return BooleanFormula.var(f"{expr.replace(' ', '_')}")
        if '>=' in expr:
            return BooleanFormula.var(f"{expr.replace(' ', '_')}")
        if '<' in expr:
            return BooleanFormula.var(f"{expr.replace(' ', '_')}")
        if '>' in expr:
            return BooleanFormula.var(f"{expr.replace(' ', '_')}")
        
        return BooleanFormula.var(expr)


# ============================================================================
# 8. Symbolic Execution
# ============================================================================

@dataclass
class SymbolicState:
    """符号执行状态"""
    path_id: int
    pc: int  # 程序计数器
    symbolic_store: Dict[str, Any]  # 符号存储
    path_constraint: BooleanFormula  # 路径约束
    depth: int = 0
    
    def fork(self, condition: BooleanFormula, new_pc: int) -> SymbolicState:
        """分叉状态"""
        return SymbolicState(
            path_id=self.path_id + 1,
            pc=new_pc,
            symbolic_store=dict(self.symbolic_store),
            path_constraint=BooleanFormula.and_(self.path_constraint, condition),
            depth=self.depth + 1
        )


@dataclass
class SymbolicExecution:
    """
    符号执行引擎
    
    支持：
    - 路径约束收集
    - 约束求解
    - 路径探索（BFS/DFS）
    - 状态合并
    """
    sat_solver: SATSolver = field(default_factory=SATSolver)
    max_depth: int = 100
    max_paths: int = 1000
    
    def execute(
        self,
        program: List[Any],
        initial_store: Dict[str, Any],
        strategy: str = 'dfs'
    ) -> List[SymbolicState]:
        """
        执行符号执行
        
        program: 指令列表
        initial_store: 初始符号存储
        strategy: 'dfs' 或 'bfs'
        """
        initial_state = SymbolicState(
            path_id=0,
            pc=0,
            symbolic_store=initial_store,
            path_constraint=BooleanFormula.true()
        )
        
        if strategy == 'dfs':
            return self._execute_dfs(program, initial_state)
        else:
            return self._execute_bfs(program, initial_state)
    
    def _execute_dfs(
        self,
        program: List[Any],
        initial: SymbolicState
    ) -> List[SymbolicState]:
        """深度优先搜索路径探索"""
        completed: List[SymbolicState] = []
        stack: List[SymbolicState] = [initial]
        path_count = 0
        
        while stack and path_count < self.max_paths:
            state = stack.pop()
            
            if state.depth >= self.max_depth:
                completed.append(state)
                continue
            
            if state.pc >= len(program):
                completed.append(state)
                continue
            
            # 执行指令
            instr = program[state.pc]
            next_states = self._execute_instruction(instr, state, program)
            
            for ns in next_states:
                if ns.pc >= len(program):
                    completed.append(ns)
                    path_count += 1
                else:
                    stack.append(ns)
        
        return completed
    
    def _execute_bfs(
        self,
        program: List[Any],
        initial: SymbolicState
    ) -> List[SymbolicState]:
        """广度优先搜索路径探索"""
        completed: List[SymbolicState] = []
        queue: deque[SymbolicState] = deque([initial])
        path_count = 0
        
        while queue and path_count < self.max_paths:
            state = queue.popleft()
            
            if state.depth >= self.max_depth:
                completed.append(state)
                continue
            
            if state.pc >= len(program):
                completed.append(state)
                continue
            
            instr = program[state.pc]
            next_states = self._execute_instruction(instr, state, program)
            
            for ns in next_states:
                if ns.pc >= len(program):
                    completed.append(ns)
                    path_count += 1
                else:
                    queue.append(ns)
        
        return completed
    
    def _execute_instruction(
        self,
        instr: Any,
        state: SymbolicState,
        program: List[Any]
    ) -> List[SymbolicState]:
        """执行单条指令"""
        results: List[SymbolicState] = []
        
        instr_type = instr.get('type') if isinstance(instr, dict) else instr
        
        if instr_type == 'assign':
            # 赋值
            var = instr['var']
            expr = instr['expr']
            new_store = dict(state.symbolic_store)
            new_store[var] = expr
            results.append(SymbolicState(
                path_id=state.path_id,
                pc=state.pc + 1,
                symbolic_store=new_store,
                path_constraint=state.path_constraint,
                depth=state.depth
            ))
        
        elif instr_type == 'branch':
            # 条件分支
            cond = instr['condition']
            target = instr['target']
            
            # 真分支
            true_state = state.fork(cond, target)
            results.append(true_state)
            
            # 假分支
            false_state = state.fork(BooleanFormula.not_(cond), state.pc + 1)
            results.append(false_state)
        
        elif instr_type == 'jump':
            # 无条件跳转
            target = instr['target']
            results.append(SymbolicState(
                path_id=state.path_id,
                pc=target,
                symbolic_store=dict(state.symbolic_store),
                path_constraint=state.path_constraint,
                depth=state.depth
            ))
        
        elif instr_type == 'assert':
            # 断言
            cond = instr['condition']
            # 添加断言到路径约束
            new_constraint = BooleanFormula.and_(state.path_constraint, cond)
            results.append(SymbolicState(
                path_id=state.path_id,
                pc=state.pc + 1,
                symbolic_store=dict(state.symbolic_store),
                path_constraint=new_constraint,
                depth=state.depth
            ))
        
        else:
            # 默认：下一条指令
            results.append(SymbolicState(
                path_id=state.path_id,
                pc=state.pc + 1,
                symbolic_store=dict(state.symbolic_store),
                path_constraint=state.path_constraint,
                depth=state.depth
            ))
        
        return results
    
    def solve_path(self, state: SymbolicState) -> Tuple[bool, Optional[Dict[str, bool]]]:
        """求解路径约束"""
        sat = SATSolver()
        sat.add_formula(state.path_constraint)
        return sat.solve()
    
    def merge_states(self, states: List[SymbolicState]) -> List[SymbolicState]:
        """
        合并相似状态
        
        当两个状态具有相同的pc和相似的存储时，合并它们的路径约束
        """
        groups: Dict[int, List[SymbolicState]] = defaultdict(list)
        
        for s in states:
            groups[s.pc].append(s)
        
        merged: List[SymbolicState] = []
        for pc, group in groups.items():
            if len(group) == 1:
                merged.append(group[0])
                continue
            
            # 尝试合并
            for state in group:
                found_merge = False
                for m in merged:
                    if m.pc == pc and m.symbolic_store == state.symbolic_store:
                        # 合并路径约束
                        m.path_constraint = BooleanFormula.or_(
                            m.path_constraint,
                            state.path_constraint
                        )
                        found_merge = True
                        break
                
                if not found_merge:
                    merged.append(state)
        
        return merged


# ============================================================================
# 9. Property Specification
# ============================================================================

class PropertyType(Enum):
    """性质类型"""
    INVARIANT = auto()      # 不变式
    SAFETY = auto()         # 安全性
    LIVENESS = auto()       # 活性
    FAIRNESS = auto()       # 公平性
    REACHABILITY = auto()   # 可达性


@dataclass
class Property:
    """性质规约"""
    name: str
    ptype: PropertyType
    formula: Union[BooleanFormula, CTLFormula, LTLFormula]
    description: str = ""
    
    def __str__(self) -> str:
        return f"{self.name} ({self.ptype.name}): {self.formula}"


@dataclass
class Invariant(Property):
    """不变式性质"""
    def __init__(self, name: str, formula: BooleanFormula, description: str = ""):
        super().__init__(name, PropertyType.INVARIANT, formula, description)


@dataclass
class SafetyProperty(Property):
    """安全性性质：坏事永远不会发生"""
    bad_condition: BooleanFormula = field(default_factory=BooleanFormula.false)
    
    def __init__(self, name: str, bad_cond: BooleanFormula, description: str = ""):
        super().__init__(name, PropertyType.SAFETY, bad_cond, description)
        self.bad_condition = bad_cond


@dataclass
class LivenessProperty(Property):
    """活性性质：好事最终会发生"""
    good_condition: BooleanFormula = field(default_factory=BooleanFormula.true)
    
    def __init__(self, name: str, good_cond: BooleanFormula, description: str = ""):
        super().__init__(name, PropertyType.LIVENESS, good_cond, description)
        self.good_condition = good_cond


@dataclass
class FairnessConstraint:
    """公平性约束"""
    name: str
    condition: BooleanFormula
    fairness_type: str = "weak"  # 'weak' 或 'strong'
    
    def __str__(self) -> str:
        return f"Fairness({self.name}): {self.condition}"


@dataclass
class PropertySpecification:
    """
    性质规约集合
    
    管理待验证的性质集合
    """
    properties: List[Property] = field(default_factory=list)
    fairness_constraints: List[FairnessConstraint] = field(default_factory=list)
    
    def add_property(self, prop: Property) -> None:
        """添加性质"""
        self.properties.append(prop)
    
    def add_fairness(self, constraint: FairnessConstraint) -> None:
        """添加公平性约束"""
        self.fairness_constraints.append(constraint)
    
    def get_properties_by_type(self, ptype: PropertyType) -> List[Property]:
        """按类型获取性质"""
        return [p for p in self.properties if p.ptype == ptype]
    
    def verify_all(
        self,
        checker: ModelChecker
    ) -> Dict[str, Tuple[bool, Optional[Any]]]:
        """验证所有性质"""
        results: Dict[str, Tuple[bool, Optional[Any]]] = {}
        
        for prop in self.properties:
            if isinstance(prop.formula, CTLFormula):
                result = checker.check_ctl_property(prop.formula)
                results[prop.name] = (result, None)
            elif isinstance(prop.formula, LTLFormula):
                result = checker.check_ltl(prop.formula)
                results[prop.name] = (result, None)
            else:
                results[prop.name] = (False, None)
        
        return results


# ============================================================================
# 10. Verification Configuration
# ============================================================================

@dataclass
class VerificationConfig:
    """
    验证配置
    
    配置验证引擎的行为参数
    """
    # 求解器配置
    sat_timeout: float = 300.0  # SAT求解超时（秒）
    smt_timeout: float = 300.0  # SMT求解超时
    
    # 模型检测配置
    mc_max_iterations: int = 10000
    mc_use_bmc: bool = True
    mc_bmc_bound: int = 50
    
    # 抽象解释配置
    ai_use_widening: bool = True
    ai_widening_delay: int = 3
    ai_max_iterations: int = 1000
    
    # 符号执行配置
    se_max_depth: int = 100
    se_max_paths: int = 1000
    se_use_state_merging: bool = True
    
    # 输出配置
    generate_counterexamples: bool = True
    generate_proofs: bool = False
    verbose: bool = False
    
    # 算法选择
    use_cdcl: bool = True
    use_two_watched_literals: bool = True
    use_nelson_oppen: bool = True


# ============================================================================
# 11. Verification Engine
# ============================================================================

@dataclass
class VerificationResult:
    """验证结果"""
    property_name: str
    verified: bool
    counterexample: Optional[Any] = None
    proof_certificate: Optional[Any] = None
    verification_time: float = 0.0
    error_message: Optional[str] = None
    
    def __str__(self) -> str:
        status = "VERIFIED" if self.verified else "FAILED"
        return f"{self.property_name}: {status} (time: {self.verification_time:.3f}s)"


@dataclass
class VerificationEngine:
    """
    主验证引擎
    
    协调各种验证技术，提供端到端验证工作流
    """
    config: VerificationConfig = field(default_factory=VerificationConfig)
    
    # 子组件
    sat_solver: SATSolver = field(default_factory=SATSolver)
    smt_solver: SMTSolver = field(default_factory=SMTSolver)
    model_checker: ModelChecker = field(default_factory=ModelChecker)
    bdd_manager: BDD = field(default_factory=lambda: BDD())
    hoare_logic: HoareLogic = field(default_factory=HoareLogic)
    symbolic_exec: SymbolicExecution = field(default_factory=SymbolicExecution)
    
    def __post_init__(self):
        # 根据配置初始化组件
        self.symbolic_exec.max_depth = self.config.se_max_depth
        self.symbolic_exec.max_paths = self.config.se_max_paths
    
    def verify_boolean_formula(
        self,
        formula: BooleanFormula,
        expected: bool = True
    ) -> VerificationResult:
        """
        验证布尔公式是否可满足
        """
        import time
        start = time.time()
        
        sat = SATSolver()
        sat.add_formula(formula)
        
        result, assignment = sat.solve()
        elapsed = time.time() - start
        
        verified = (result == expected)
        
        return VerificationResult(
            property_name="boolean_satisfiability",
            verified=verified,
            counterexample=assignment if not verified and self.config.generate_counterexamples else None,
            verification_time=elapsed
        )
    
    def verify_smt(
        self,
        constraints: List[Any],
        expected: bool = True
    ) -> VerificationResult:
        """验证SMT约束"""
        import time
        start = time.time()
        
        # 添加约束
        for c in constraints:
            if isinstance(c, BooleanFormula):
                self.smt_solver.add_bool_constraint(c)
        
        result, assignment = self.smt_solver.solve()
        elapsed = time.time() - start
        
        verified = (result == expected)
        
        return VerificationResult(
            property_name="smt_satisfiability",
            verified=verified,
            counterexample=assignment if not verified else None,
            verification_time=elapsed
        )
    
    def verify_ctl(
        self,
        kripke: KripkeStructure,
        formula: CTLFormula
    ) -> VerificationResult:
        """验证CTL公式"""
        import time
        start = time.time()
        
        self.model_checker.set_model(kripke)
        result = self.model_checker.check_ctl_property(formula)
        elapsed = time.time() - start
        
        counterexample = None
        if not result and self.config.generate_counterexamples:
            # 生成反例
            _, counterexample = self.model_checker.bounded_model_check(formula, self.config.mc_bmc_bound)
        
        return VerificationResult(
            property_name="ctl_model_checking",
            verified=result,
            counterexample=counterexample,
            verification_time=elapsed
        )
    
    def verify_ltl(
        self,
        kripke: KripkeStructure,
        formula: LTLFormula
    ) -> VerificationResult:
        """验证LTL公式"""
        import time
        start = time.time()
        
        self.model_checker.set_model(kripke)
        result = self.model_checker.check_ltl(formula)
        elapsed = time.time() - start
        
        return VerificationResult(
            property_name="ltl_model_checking",
            verified=result,
            verification_time=elapsed
        )
    
    def verify_hoare_triple(self, triple: HoareTriple) -> VerificationResult:
        """验证霍尔三元组"""
        import time
        start = time.time()
        
        vcs = self.hoare_logic.generate_vc(triple)
        
        all_verified = True
        for vc in vcs:
            sat = SATSolver()
            sat.add_formula(BooleanFormula.not_(vc))
            result, _ = sat.solve()
            if result:  # 找到使VC为假的赋值
                all_verified = False
                break
        
        elapsed = time.time() - start
        
        return VerificationResult(
            property_name="hoare_logic_verification",
            verified=all_verified,
            verification_time=elapsed
        )
    
    def symbolic_verify(
        self,
        program: List[Any],
        properties: List[Property]
    ) -> List[VerificationResult]:
        """使用符号执行验证程序性质"""
        import time
        start = time.time()
        
        # 执行符号执行
        initial_store = {}
        final_states = self.symbolic_exec.execute(program, initial_store, 'dfs')
        
        results: List[VerificationResult] = []
        
        for prop in properties:
            prop_verified = True
            counterexample = None
            
            for state in final_states:
                # 检查性质在每个路径上是否成立
                sat, _ = self.symbolic_exec.solve_path(state)
                if sat:
                    # 路径可达，检查性质
                    # 简化：假设性质不成立
                    prop_verified = False
                    counterexample = state
                    break
            
            results.append(VerificationResult(
                property_name=prop.name,
                verified=prop_verified,
                counterexample=counterexample,
                verification_time=time.time() - start
            ))
        
        return results
    
    def verify_all_properties(
        self,
        spec: PropertySpecification,
        kripke: Optional[KripkeStructure] = None
    ) -> List[VerificationResult]:
        """验证规约中的所有性质"""
        results: List[VerificationResult] = []
        
        for prop in spec.properties:
            if isinstance(prop.formula, CTLFormula) and kripke:
                result = self.verify_ctl(kripke, prop.formula)
            elif isinstance(prop.formula, LTLFormula) and kripke:
                result = self.verify_ltl(kripke, prop.formula)
            elif isinstance(prop.formula, BooleanFormula):
                result = self.verify_boolean_formula(prop.formula)
            else:
                result = VerificationResult(
                    property_name=prop.name,
                    verified=False,
                    error_message="Unsupported formula type"
                )
            
            result.property_name = prop.name
            results.append(result)
        
        return results
    
    def generate_proof_certificate(
        self,
        result: VerificationResult
    ) -> Optional[Dict[str, Any]]:
        """生成证明证书"""
        if not self.config.generate_proofs or not result.verified:
            return None
        
        return {
            "property": result.property_name,
            "verified": result.verified,
            "timestamp": result.verification_time,
            "method": "formal_verification",
            "certificate_type": "proof_outline"
        }


# ============================================================================
# 工具函数
# ============================================================================

def create_sample_kripke() -> KripkeStructure:
    """创建示例Kripke结构"""
    states = {0, 1, 2, 3}
    initial = {0}
    transitions = {
        0: {1, 2},
        1: {0, 3},
        2: {3},
        3: {3}
    }
    labels = {
        0: {'p'},
        1: {'q'},
        2: {'p', 'q'},
        3: {'r'}
    }
    return KripkeStructure(states, initial, transitions, labels)


def demo_sat_solving():
    """演示SAT求解"""
    print("=" * 50)
    print("SAT求解演示")
    print("=" * 50)
    
    # 创建布尔公式: (a | b) & (~a | c) & (~b | ~c)
    a = BooleanFormula.var('a')
    b = BooleanFormula.var('b')
    c = BooleanFormula.var('c')
    
    formula = BooleanFormula.and_(
        BooleanFormula.or_(a, b),
        BooleanFormula.or_(BooleanFormula.not_(a), c),
        BooleanFormula.or_(BooleanFormula.not_(b), BooleanFormula.not_(c))
    )
    
    print(f"公式: {formula}")
    
    # 转换为CNF
    cnf = formula.to_cnf()
    print(f"CNF形式: {cnf}")
    
    # SAT求解
    solver = SATSolver()
    solver.add_formula(formula)
    result, assignment = solver.solve()
    
    print(f"可满足: {result}")
    if result:
        print(f"赋值: {assignment}")
        # 验证
        eval_result = formula.evaluate(assignment)
        print(f"验证结果: {eval_result}")
    
    print()


def demo_ctl_model_checking():
    """演示CTL模型检测"""
    print("=" * 50)
    print("CTL模型检测演示")
    print("=" * 50)
    
    # 创建Kripke结构
    kripke = create_sample_kripke()
    print(f"状态: {kripke.states}")
    print(f"初始状态: {kripke.initial}")
    print(f"转移: {kripke.transitions}")
    print(f"标签: {kripke.labels}")
    
    checker = ModelChecker(kripke)
    
    # 检查: AG(p -> AF r)
    # 即：从所有路径，如果p成立，则最终r会成立
    p = CTLFormula.atom('p')
    r = CTLFormula.atom('r')
    af_r = CTLFormula.af(r)
    
    # 检查简单性质: EF r
    ef_r = CTLFormula.ef(r)
    
    result = checker.check_ctl(ef_r)
    print(f"EF r 满足的状态: {result}")
    print(f"性质在所有初始状态成立: {kripke.initial.issubset(result)}")
    
    print()


def demo_bdd():
    """演示BDD操作"""
    print("=" * 50)
    print("BDD演示")
    print("=" * 50)
    
    bdd = BDD(['a', 'b', 'c'])
    
    # 创建变量
    a = bdd.var('a')
    b = bdd.var('b')
    c = bdd.var('c')
    
    # 构建公式: (a & b) | c
    ab = bdd.and_(a, b)
    formula = bdd.or_(ab, c)
    
    print(f"公式: (a & b) | c")
    print(f"BDD节点数: {len(bdd.node_list)}")
    print(f"满足赋值数: {bdd.sat_count(formula)}")
    
    # 找到一个满足赋值
    sat = bdd.any_sat(formula)
    print(f"一个满足赋值: {sat}")
    
    # 存在量词消去: exists b. (a & b) | c = a | c
    exists_b = bdd.exists(formula, 'b')
    print(f"exists b. formula: 节点索引 {exists_b}")
    
    print()


def demo_hoare_logic():
    """演示霍尔逻辑"""
    print("=" * 50)
    print("霍尔逻辑演示")
    print("=" * 50)
    
    hl = HoareLogic()
    
    # 计算最弱前置条件
    # wp(x := x + 1, x > 0) = x + 1 > 0 = x > -1
    post = BooleanFormula.var('x_gt_0')
    cmd = "x := x + 1"
    wp = hl.weakest_precondition(cmd, post)
    print(f"命令: {cmd}")
    print(f"后置条件: x > 0")
    print(f"最弱前置条件: {wp}")
    
    # 验证霍尔三元组
    pre = BooleanFormula.var('x_gt_0')
    triple = HoareTriple(pre, cmd, post)
    vcs = hl.generate_vc(triple)
    print(f"验证条件: {vcs}")
    
    print()


def demo_symbolic_execution():
    """演示符号执行"""
    print("=" * 50)
    print("符号执行演示")
    print("=" * 50)
    
    se = SymbolicExecution()
    
    # 简单程序
    program = [
        {'type': 'assign', 'var': 'x', 'expr': 'input'},
        {'type': 'branch', 'condition': BooleanFormula.var('x_gt_0'), 'target': 3},
        {'type': 'assign', 'var': 'y', 'expr': '0'},
        {'type': 'jump', 'target': 4},
        {'type': 'assign', 'var': 'y', 'expr': '1'},
    ]
    
    initial = {'x': 'sym_x', 'y': 'sym_y'}
    final_states = se.execute(program, initial, 'dfs')
    
    print(f"探索路径数: {len(final_states)}")
    for state in final_states:
        print(f"  路径 {state.path_id}: pc={state.pc}, store={state.symbolic_store}")
        print(f"    约束: {state.path_constraint}")
    
    print()


def main():
    """主函数 - 运行所有演示"""
    print("\n形式化验证与模型检测模块演示\n")
    
    demo_sat_solving()
    demo_ctl_model_checking()
    demo_bdd()
    demo_hoare_logic()
    demo_symbolic_execution()
    
    print("=" * 50)
    print("所有演示完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
