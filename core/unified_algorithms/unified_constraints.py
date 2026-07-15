"""
统一约束系统模块

提供通用的约束管理功能，支持AGI和视频生成系统的约束需求。
包括物理约束、逻辑约束和一致性约束等。

核心组件：
- Constraint: 约束基类
- PhysicsConstraint: 物理约束（视频用）
- LogicalConstraint: 逻辑约束（AGI用）
- ConsistencyConstraint: 一致性约束
- ConstraintManager: 约束管理器
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, List, Dict, Any, Callable, Tuple, Set
from enum import Enum, auto
import math

from .unified_config import (
    UnifiedAlgorithmConfig,
    ConstraintPriority,
    T
)


# ============================================================================
# 约束相关的数据结构
# ============================================================================

@dataclass
class ConstraintViolation:
    """
    约束违反信息
    
    记录约束违反的详细信息。
    
    Attributes:
        constraint_name: 约束名称
        constraint_type: 约束类型
        severity: 严重程度 (0.0-1.0)
        message: 违反描述
        location: 违反位置
        suggestion: 修复建议
    """
    constraint_name: str
    constraint_type: str
    severity: float = 0.0
    message: str = ""
    location: Optional[Any] = None
    suggestion: str = ""


@dataclass
class ConstraintCheckResult:
    """
    约束检查结果
    
    Attributes:
        is_satisfied: 是否满足约束
        violations: 违反列表
        score: 约束满足分数 (0.0-1.0)
        details: 详细信息
    """
    is_satisfied: bool = True
    violations: List[ConstraintViolation] = field(default_factory=list)
    score: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __bool__(self) -> bool:
        """布尔值表示是否满足约束"""
        return self.is_satisfied


@dataclass
class ConstraintStats:
    """
    约束统计信息
    
    Attributes:
        total_checks: 总检查次数
        satisfied_count: 满足次数
        violated_count: 违反次数
        avg_violation_severity: 平均违反严重程度
    """
    total_checks: int = 0
    satisfied_count: int = 0
    violated_count: int = 0
    avg_violation_severity: float = 0.0


# ============================================================================
# 约束基类
# ============================================================================

class Constraint(ABC, Generic[T]):
    """
    约束基类
    
    所有约束的抽象基类，定义通用接口。
    支持任意类型的数据约束检查。
    
    Attributes:
        name: 约束名称
        priority: 约束优先级
        tolerance: 容差
    """
    
    def __init__(self,
                 name: str,
                 priority: ConstraintPriority = ConstraintPriority.SOFT,
                 tolerance: float = 0.01):
        """
        初始化约束
        
        Args:
            name: 约束名称
            priority: 约束优先级
            tolerance: 容差
        """
        self.name = name
        self.priority = priority
        self.tolerance = tolerance
        
        self._stats = ConstraintStats()
        self._enabled = True
    
    @abstractmethod
    def check(self, data: T) -> ConstraintCheckResult:
        """
        检查数据是否满足约束
        
        Args:
            data: 要检查的数据
            
        Returns:
            检查结果
        """
        pass
    
    @abstractmethod
    def repair(self, data: T) -> T:
        """
        尝试修复违反约束的数据
        
        Args:
            data: 违反约束的数据
            
        Returns:
            修复后的数据
        """
        pass
    
    def is_enabled(self) -> bool:
        """检查约束是否启用"""
        return self._enabled
    
    def enable(self) -> None:
        """启用约束"""
        self._enabled = True
    
    def disable(self) -> None:
        """禁用约束"""
        self._enabled = False
    
    def get_stats(self) -> ConstraintStats:
        """获取约束统计信息"""
        return self._stats
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self._stats = ConstraintStats()
    
    def _update_stats(self, result: ConstraintCheckResult) -> None:
        """更新统计信息"""
        self._stats.total_checks += 1
        
        if result.is_satisfied:
            self._stats.satisfied_count += 1
        else:
            self._stats.violated_count += 1
            if result.violations:
                avg_severity = sum(v.severity for v in result.violations) / len(result.violations)
                # 更新平均严重程度
                total = self._stats.violated_count
                self._stats.avg_violation_severity = (
                    (self._stats.avg_violation_severity * (total - 1) + avg_severity) / total
                )


# ============================================================================
# 物理约束
# ============================================================================

class PhysicsConstraint(Constraint[T]):
    """
    物理约束
    
    用于视频生成系统，确保生成的内容符合物理规律。
    包括速度限制、碰撞检测、能量守恒等。
    
    Attributes:
        constraint_type: 物理约束类型
        params: 物理参数
    """
    
    def __init__(self,
                 name: str,
                 constraint_type: str = "velocity_limit",
                 priority: ConstraintPriority = ConstraintPriority.HARD,
                 tolerance: float = 0.01,
                 **params):
        """
        初始化物理约束
        
        Args:
            name: 约束名称
            constraint_type: 约束类型
            priority: 约束优先级
            tolerance: 容差
            **params: 物理参数
        """
        super().__init__(name, priority, tolerance)
        self.constraint_type = constraint_type
        self.params = params
    
    def check(self, data: T) -> ConstraintCheckResult:
        """
        检查物理约束
        
        Args:
            data: 物理状态数据
            
        Returns:
            检查结果
        """
        if not self._enabled:
            return ConstraintCheckResult(is_satisfied=True)
        
        violations = []
        
        try:
            if self.constraint_type == "velocity_limit":
                violations = self._check_velocity_limit(data)
            elif self.constraint_type == "collision":
                violations = self._check_collision(data)
            elif self.constraint_type == "energy_conservation":
                violations = self._check_energy_conservation(data)
            elif self.constraint_type == "boundary":
                violations = self._check_boundary(data)
        except Exception as e:
            violations.append(ConstraintViolation(
                constraint_name=self.name,
                constraint_type=self.constraint_type,
                severity=1.0,
                message=f"检查过程出错: {str(e)}"
            ))
        
        is_satisfied = len(violations) == 0
        score = 1.0 if is_satisfied else max(0.0, 1.0 - sum(v.severity for v in violations))
        
        result = ConstraintCheckResult(
            is_satisfied=is_satisfied,
            violations=violations,
            score=score
        )
        
        self._update_stats(result)
        return result
    
    def _check_velocity_limit(self, data: T) -> List[ConstraintViolation]:
        """检查速度限制"""
        violations = []
        max_velocity = self.params.get('max_velocity', 10.0)
        
        try:
            if isinstance(data, dict):
                velocity = data.get('velocity')
                if velocity is not None:
                    v_mag = self._magnitude(velocity)
                    if v_mag > max_velocity + self.tolerance:
                        violations.append(ConstraintViolation(
                            constraint_name=self.name,
                            constraint_type=self.constraint_type,
                            severity=min(1.0, (v_mag - max_velocity) / max_velocity),
                            message=f"速度 {v_mag:.2f} 超过限制 {max_velocity}",
                            location=data.get('position'),
                            suggestion="降低速度或调整时间步长"
                        ))
            elif isinstance(data, (list, tuple)):
                # 假设是速度向量
                v_mag = self._magnitude(data)
                if v_mag > max_velocity + self.tolerance:
                    violations.append(ConstraintViolation(
                        constraint_name=self.name,
                        constraint_type=self.constraint_type,
                        severity=min(1.0, (v_mag - max_velocity) / max_velocity),
                        message=f"速度 {v_mag:.2f} 超过限制 {max_velocity}"
                    ))
        except (ValueError, TypeError):
            pass
        
        return violations
    
    def _check_collision(self, data: T) -> List[ConstraintViolation]:
        """检查碰撞"""
        violations = []
        min_distance = self.params.get('min_distance', 0.1)
        
        try:
            if isinstance(data, dict):
                objects = data.get('objects', [])
                for i, obj1 in enumerate(objects):
                    for obj2 in objects[i+1:]:
                        dist = self._distance(
                            obj1.get('position', [0, 0]),
                            obj2.get('position', [0, 0])
                        )
                        if dist < min_distance - self.tolerance:
                            violations.append(ConstraintViolation(
                                constraint_name=self.name,
                                constraint_type=self.constraint_type,
                                severity=min(1.0, 1.0 - dist / min_distance),
                                message=f"对象 {i} 和 {i+1} 发生碰撞，距离: {dist:.3f}",
                                suggestion="调整位置或添加碰撞响应"
                            ))
        except (ValueError, TypeError):
            pass
        
        return violations
    
    def _check_energy_conservation(self, data: T) -> List[ConstraintViolation]:
        """检查能量守恒"""
        violations = []
        tolerance = self.params.get('energy_tolerance', 0.1)
        
        try:
            if isinstance(data, dict):
                initial_energy = data.get('initial_energy', 0)
                current_energy = data.get('current_energy', 0)
                
                if initial_energy > 0:
                    energy_diff = abs(current_energy - initial_energy) / initial_energy
                    if energy_diff > tolerance:
                        violations.append(ConstraintViolation(
                            constraint_name=self.name,
                            constraint_type=self.constraint_type,
                            severity=min(1.0, energy_diff / tolerance),
                            message=f"能量不守恒: 初始={initial_energy:.2f}, 当前={current_energy:.2f}",
                            suggestion="检查数值积分方法或添加能量补偿"
                        ))
        except (ValueError, TypeError):
            pass
        
        return violations
    
    def _check_boundary(self, data: T) -> List[ConstraintViolation]:
        """检查边界约束"""
        violations = []
        bounds = self.params.get('bounds', [[-10, 10], [-10, 10], [-10, 10]])
        
        try:
            if isinstance(data, dict):
                position = data.get('position')
                if position:
                    for i, (pos, bound) in enumerate(zip(position, bounds)):
                        if pos < bound[0] - self.tolerance or pos > bound[1] + self.tolerance:
                            violations.append(ConstraintViolation(
                                constraint_name=self.name,
                                constraint_type=self.constraint_type,
                                severity=min(1.0, abs(pos - sum(bound)/2) / (bound[1] - bound[0])),
                                message=f"位置维度 {i} ({pos:.2f}) 超出边界 [{bound[0]}, {bound[1]}]",
                                suggestion="限制位置在边界内或调整边界"
                            ))
        except (ValueError, TypeError):
            pass
        
        return violations
    
    def repair(self, data: T) -> T:
        """
        修复物理约束违反
        
        Args:
            data: 违反约束的数据
            
        Returns:
            修复后的数据
        """
        try:
            if isinstance(data, dict):
                result = dict(data)
                
                if self.constraint_type == "velocity_limit":
                    velocity = result.get('velocity')
                    if velocity is not None:
                        max_velocity = self.params.get('max_velocity', 10.0)
                        v_mag = self._magnitude(velocity)
                        if v_mag > max_velocity:
                            # 缩放速度
                            scale = max_velocity / v_mag
                            result['velocity'] = [v * scale for v in velocity]
                
                elif self.constraint_type == "boundary":
                    position = result.get('position')
                    if position:
                        bounds = self.params.get('bounds', [[-10, 10], [-10, 10], [-10, 10]])
                        new_pos = []
                        for pos, bound in zip(position, bounds):
                            new_pos.append(max(bound[0], min(bound[1], pos)))
                        result['position'] = new_pos
                
                return result  # type: ignore
        except (ValueError, TypeError):
            pass
        
        return data
    
    def _magnitude(self, vector: Any) -> float:
        """计算向量模长"""
        try:
            if isinstance(vector, (list, tuple)):
                return math.sqrt(sum(float(x) ** 2 for x in vector))
        except (ValueError, TypeError):
            pass
        return 0.0
    
    def _distance(self, a: Any, b: Any) -> float:
        """计算两点距离"""
        try:
            if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))
        except (ValueError, TypeError):
            pass
        return 0.0


# ============================================================================
# 逻辑约束
# ============================================================================

class LogicalConstraint(Constraint[T]):
    """
    逻辑约束
    
    用于AGI系统，确保推理和决策符合逻辑规则。
    包括一致性、完备性、非矛盾性等。
    
    Attributes:
        constraint_type: 逻辑约束类型
        rules: 逻辑规则
    """
    
    def __init__(self,
                 name: str,
                 constraint_type: str = "consistency",
                 priority: ConstraintPriority = ConstraintPriority.HARD,
                 tolerance: float = 0.01,
                 **rules):
        """
        初始化逻辑约束
        
        Args:
            name: 约束名称
            constraint_type: 约束类型
            priority: 约束优先级
            tolerance: 容差
            **rules: 逻辑规则
        """
        super().__init__(name, priority, tolerance)
        self.constraint_type = constraint_type
        self.rules = rules
    
    def check(self, data: T) -> ConstraintCheckResult:
        """
        检查逻辑约束
        
        Args:
            data: 逻辑数据（如知识库、推理链等）
            
        Returns:
            检查结果
        """
        if not self._enabled:
            return ConstraintCheckResult(is_satisfied=True)
        
        violations = []
        
        try:
            if self.constraint_type == "consistency":
                violations = self._check_consistency(data)
            elif self.constraint_type == "completeness":
                violations = self._check_completeness(data)
            elif self.constraint_type == "non_contradiction":
                violations = self._check_non_contradiction(data)
            elif self.constraint_type == "causal_chain":
                violations = self._check_causal_chain(data)
        except Exception as e:
            violations.append(ConstraintViolation(
                constraint_name=self.name,
                constraint_type=self.constraint_type,
                severity=1.0,
                message=f"检查过程出错: {str(e)}"
            ))
        
        is_satisfied = len(violations) == 0
        score = 1.0 if is_satisfied else max(0.0, 1.0 - sum(v.severity for v in violations))
        
        result = ConstraintCheckResult(
            is_satisfied=is_satisfied,
            violations=violations,
            score=score
        )
        
        self._update_stats(result)
        return result
    
    def _check_consistency(self, data: T) -> List[ConstraintViolation]:
        """检查一致性"""
        violations = []
        
        try:
            if isinstance(data, dict):
                # 检查知识库一致性
                facts = data.get('facts', [])
                for i, fact1 in enumerate(facts):
                    for fact2 in facts[i+1:]:
                        if self._are_contradictory(fact1, fact2):
                            violations.append(ConstraintViolation(
                                constraint_name=self.name,
                                constraint_type=self.constraint_type,
                                severity=1.0,
                                message=f"事实矛盾: {fact1} vs {fact2}",
                                suggestion="检查事实来源或添加条件限定"
                            ))
        except Exception:
            pass
        
        return violations
    
    def _check_completeness(self, data: T) -> List[ConstraintViolation]:
        """检查完备性"""
        violations = []
        
        try:
            if isinstance(data, dict):
                required_fields = self.rules.get('required_fields', [])
                for field in required_fields:
                    if field not in data or data[field] is None:
                        violations.append(ConstraintViolation(
                            constraint_name=self.name,
                            constraint_type=self.constraint_type,
                            severity=0.5,
                            message=f"缺少必需字段: {field}",
                            suggestion=f"提供 {field} 的值"
                        ))
        except Exception:
            pass
        
        return violations
    
    def _check_non_contradiction(self, data: T) -> List[ConstraintViolation]:
        """检查非矛盾性"""
        violations = []
        
        try:
            if isinstance(data, list):
                # 检查推理链中的矛盾
                for i in range(len(data) - 1):
                    current = data[i]
                    next_step = data[i + 1]
                    if not self._is_valid_inference(current, next_step):
                        violations.append(ConstraintViolation(
                            constraint_name=self.name,
                            constraint_type=self.constraint_type,
                            severity=0.8,
                            message=f"推理步骤 {i} 到 {i+1} 存在逻辑跳跃",
                            location=i,
                            suggestion="添加中间推理步骤或检查前提"
                        ))
        except Exception:
            pass
        
        return violations
    
    def _check_causal_chain(self, data: T) -> List[ConstraintViolation]:
        """检查因果链"""
        violations = []
        
        try:
            if isinstance(data, list) and len(data) >= 2:
                # 检查因果关系
                for i in range(len(data) - 1):
                    cause = data[i]
                    effect = data[i + 1]
                    if not self._has_causal_link(cause, effect):
                        violations.append(ConstraintViolation(
                            constraint_name=self.name,
                            constraint_type=self.constraint_type,
                            severity=0.6,
                            message=f"步骤 {i} 和 {i+1} 之间缺少因果联系",
                            suggestion="明确因果关系或添加解释"
                        ))
        except Exception:
            pass
        
        return violations
    
    def _are_contradictory(self, fact1: Any, fact2: Any) -> bool:
        """检查两个事实是否矛盾（简化实现）"""
        try:
            if isinstance(fact1, dict) and isinstance(fact2, dict):
                # 检查是否有矛盾的键值对
                for key in set(fact1.keys()) & set(fact2.keys()):
                    if fact1[key] != fact2[key]:
                        return True
            return fact1 == fact2  # 相同事实不矛盾
        except Exception:
            return False
    
    def _is_valid_inference(self, premise: Any, conclusion: Any) -> bool:
        """检查推理是否有效（简化实现）"""
        # 默认允许所有推理
        return True
    
    def _has_causal_link(self, cause: Any, effect: Any) -> bool:
        """检查是否有因果联系（简化实现）"""
        # 默认假设有联系
        return True
    
    def repair(self, data: T) -> T:
        """
        修复逻辑约束违反
        
        Args:
            data: 违反约束的数据
            
        Returns:
            修复后的数据
        """
        try:
            if isinstance(data, dict):
                result = dict(data)
                
                if self.constraint_type == "completeness":
                    # 填充缺失字段
                    required_fields = self.rules.get('required_fields', [])
                    for field in required_fields:
                        if field not in result or result[field] is None:
                            result[field] = self.rules.get(f'default_{field}', None)
                
                return result  # type: ignore
        except Exception:
            pass
        
        return data


# ============================================================================
# 一致性约束
# ============================================================================

class ConsistencyConstraint(Constraint[T]):
    """
    一致性约束
    
    确保数据在不同时间或不同视角下保持一致。
    适用于时序数据、多视图数据等。
    
    Attributes:
        reference_data: 参考数据
        consistency_type: 一致性类型
    """
    
    def __init__(self,
                 name: str,
                 reference_data: Optional[T] = None,
                 consistency_type: str = "temporal",
                 priority: ConstraintPriority = ConstraintPriority.SOFT,
                 tolerance: float = 0.05):
        """
        初始化一致性约束
        
        Args:
            name: 约束名称
            reference_data: 参考数据
            consistency_type: 一致性类型
            priority: 约束优先级
            tolerance: 容差
        """
        super().__init__(name, priority, tolerance)
        self.reference_data = reference_data
        self.consistency_type = consistency_type
    
    def set_reference(self, data: T) -> None:
        """
        设置参考数据
        
        Args:
            data: 参考数据
        """
        self.reference_data = data
    
    def check(self, data: T) -> ConstraintCheckResult:
        """
        检查一致性
        
        Args:
            data: 要检查的数据
            
        Returns:
            检查结果
        """
        if not self._enabled:
            return ConstraintCheckResult(is_satisfied=True)
        
        if self.reference_data is None:
            # 首次检查，设置参考
            self.reference_data = data
            return ConstraintCheckResult(is_satisfied=True)
        
        violations = []
        
        try:
            if self.consistency_type == "temporal":
                violations = self._check_temporal_consistency(data)
            elif self.consistency_type == "spatial":
                violations = self._check_spatial_consistency(data)
            elif self.consistency_type == "semantic":
                violations = self._check_semantic_consistency(data)
        except Exception as e:
            violations.append(ConstraintViolation(
                constraint_name=self.name,
                constraint_type=self.consistency_type,
                severity=1.0,
                message=f"检查过程出错: {str(e)}"
            ))
        
        is_satisfied = len(violations) == 0
        score = 1.0 if is_satisfied else max(0.0, 1.0 - sum(v.severity for v in violations))
        
        result = ConstraintCheckResult(
            is_satisfied=is_satisfied,
            violations=violations,
            score=score
        )
        
        self._update_stats(result)
        
        # 更新参考数据
        if is_satisfied or self.consistency_type == "temporal":
            self.reference_data = data
        
        return result
    
    def _check_temporal_consistency(self, data: T) -> List[ConstraintViolation]:
        """检查时序一致性"""
        violations = []
        
        try:
            similarity = self._compute_similarity(data, self.reference_data)
            if similarity < (1.0 - self.tolerance):
                violations.append(ConstraintViolation(
                    constraint_name=self.name,
                    constraint_type=self.consistency_type,
                    severity=1.0 - similarity,
                    message=f"时序不一致: 相似度 {similarity:.2f} 低于阈值 {1.0 - self.tolerance:.2f}",
                    suggestion="检查时间步长或平滑过渡"
                ))
        except Exception:
            pass
        
        return violations
    
    def _check_spatial_consistency(self, data: T) -> List[ConstraintViolation]:
        """检查空间一致性"""
        violations = []
        
        try:
            if isinstance(data, dict) and isinstance(self.reference_data, dict):
                # 检查位置变化是否合理
                pos = data.get('position')
                ref_pos = self.reference_data.get('position')
                
                if pos and ref_pos:
                    distance = self._distance(pos, ref_pos)
                    max_movement = self.tolerance * 10  # 假设最大移动距离
                    
                    if distance > max_movement:
                        violations.append(ConstraintViolation(
                            constraint_name=self.name,
                            constraint_type=self.consistency_type,
                            severity=min(1.0, distance / max_movement - 1.0),
                            message=f"空间跳跃: 移动距离 {distance:.2f} 超过最大 {max_movement:.2f}",
                            suggestion="插值中间位置或检查坐标系"
                        ))
        except Exception:
            pass
        
        return violations
    
    def _check_semantic_consistency(self, data: T) -> List[ConstraintViolation]:
        """检查语义一致性"""
        violations = []
        
        try:
            if isinstance(data, str) and isinstance(self.reference_data, str):
                # 文本语义一致性（简化）
                if data != self.reference_data:
                    # 检查是否有矛盾的关键词
                    contradiction_keywords = self._get_contradiction_keywords()
                    for kw1, kw2 in contradiction_keywords:
                        if kw1 in data.lower() and kw2 in self.reference_data.lower():
                            violations.append(ConstraintViolation(
                                constraint_name=self.name,
                                constraint_type=self.consistency_type,
                                severity=0.8,
                                message=f"语义矛盾: '{kw1}' vs '{kw2}'",
                                suggestion="统一表述或添加上下文说明"
                            ))
        except Exception:
            pass
        
        return violations
    
    def _compute_similarity(self, a: T, b: T) -> float:
        """计算相似度"""
        try:
            if a == b:
                return 1.0
            
            if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                vec_a = [float(x) for x in a]
                vec_b = [float(x) for x in b]
                
                if len(vec_a) == len(vec_b) and len(vec_a) > 0:
                    dot = sum(x * y for x, y in zip(vec_a, vec_b))
                    norm_a = math.sqrt(sum(x * x for x in vec_a))
                    norm_b = math.sqrt(sum(x * x for x in vec_b))
                    
                    if norm_a > 0 and norm_b > 0:
                        return dot / (norm_a * norm_b)
            
            return 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def _distance(self, a: Any, b: Any) -> float:
        """计算距离"""
        try:
            if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))
        except (ValueError, TypeError):
            pass
        return 0.0
    
    def _get_contradiction_keywords(self) -> List[Tuple[str, str]]:
        """获取矛盾关键词对"""
        return [
            ('是', '不是'),
            ('有', '没有'),
            ('可以', '不能'),
            ('正确', '错误'),
        ]
    
    def repair(self, data: T) -> T:
        """
        修复一致性违反
        
        Args:
            data: 违反约束的数据
            
        Returns:
            修复后的数据
        """
        try:
            if self.consistency_type == "temporal":
                # 时序平滑：与参考数据混合
                if isinstance(data, (list, tuple)) and isinstance(self.reference_data, (list, tuple)):
                    result = []
                    for d, r in zip(data, self.reference_data):
                        try:
                            blended = 0.7 * float(d) + 0.3 * float(r)
                            result.append(type(d)(blended))
                        except (ValueError, TypeError):
                            result.append(d)
                    return type(data)(result)  # type: ignore
        except Exception:
            pass
        
        return data


# ============================================================================
# 约束管理器
# ============================================================================

class ConstraintManager(Generic[T]):
    """
    约束管理器
    
    管理多个约束，提供统一的检查接口。
    支持约束优先级和批量检查。
    
    Attributes:
        constraints: 约束列表
        strict_mode: 严格模式
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        初始化约束管理器
        
        Args:
            strict_mode: 严格模式（任何违反都返回失败）
        """
        self.constraints: List[Constraint[T]] = []
        self.strict_mode = strict_mode
        self._check_history: List[ConstraintCheckResult] = []
    
    def add_constraint(self, constraint: Constraint[T]) -> None:
        """
        添加约束
        
        Args:
            constraint: 约束实例
        """
        self.constraints.append(constraint)
    
    def remove_constraint(self, name: str) -> bool:
        """
        移除约束
        
        Args:
            name: 约束名称
            
        Returns:
            是否成功移除
        """
        for i, c in enumerate(self.constraints):
            if c.name == name:
                self.constraints.pop(i)
                return True
        return False
    
    def check_all(self, data: T) -> ConstraintCheckResult:
        """
        检查所有约束
        
        Args:
            data: 要检查的数据
            
        Returns:
            综合检查结果
        """
        all_violations = []
        total_score = 0.0
        hard_violations = 0
        
        for constraint in self.constraints:
            if not constraint.is_enabled():
                continue
            
            result = constraint.check(data)
            total_score += result.score
            
            for violation in result.violations:
                all_violations.append(violation)
                if constraint.priority == ConstraintPriority.HARD:
                    hard_violations += 1
        
        # 计算综合分数
        num_constraints = len([c for c in self.constraints if c.is_enabled()])
        avg_score = total_score / num_constraints if num_constraints > 0 else 1.0
        
        # 判断是否满足
        if self.strict_mode:
            is_satisfied = len(all_violations) == 0
        else:
            is_satisfied = hard_violations == 0
        
        result = ConstraintCheckResult(
            is_satisfied=is_satisfied,
            violations=all_violations,
            score=avg_score,
            details={
                'hard_violations': hard_violations,
                'total_violations': len(all_violations),
                'num_constraints_checked': num_constraints
            }
        )
        
        self._check_history.append(result)
        
        # 限制历史大小
        if len(self._check_history) > 100:
            self._check_history.pop(0)
        
        return result
    
    def repair_all(self, data: T) -> T:
        """
        尝试修复所有违反
        
        Args:
            data: 原始数据
            
        Returns:
            修复后的数据
        """
        result = data
        
        # 按优先级排序：先修复硬约束
        sorted_constraints = sorted(
            self.constraints,
            key=lambda c: 0 if c.priority == ConstraintPriority.HARD else 1
        )
        
        for constraint in sorted_constraints:
            if constraint.is_enabled():
                check_result = constraint.check(result)
                if not check_result.is_satisfied:
                    result = constraint.repair(result)
        
        return result
    
    def get_constraints_by_priority(self, priority: ConstraintPriority) -> List[Constraint[T]]:
        """
        获取指定优先级的约束
        
        Args:
            priority: 约束优先级
            
        Returns:
            约束列表
        """
        return [c for c in self.constraints if c.priority == priority]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取管理器统计信息
        
        Returns:
            统计信息
        """
        total_checks = len(self._check_history)
        satisfied_checks = sum(1 for r in self._check_history if r.is_satisfied)
        
        constraint_stats = {}
        for constraint in self.constraints:
            constraint_stats[constraint.name] = constraint.get_stats()
        
        return {
            'total_constraints': len(self.constraints),
            'enabled_constraints': sum(1 for c in self.constraints if c.is_enabled()),
            'hard_constraints': len(self.get_constraints_by_priority(ConstraintPriority.HARD)),
            'total_checks': total_checks,
            'satisfied_checks': satisfied_checks,
            'violation_rate': 1.0 - (satisfied_checks / total_checks) if total_checks > 0 else 0.0,
            'constraint_stats': constraint_stats
        }
    
    def reset_all_stats(self) -> None:
        """重置所有统计信息"""
        self._check_history.clear()
        for constraint in self.constraints:
            constraint.reset_stats()
