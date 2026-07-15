"""
Physics Corrector Module for Video Post-processing

使用统一物理约束进行视频后处理。

主要组件：
- PhysicsCorrector: 物理校正器
- MotionSmoother: 运动平滑器
- ConstraintValidator: 约束验证器
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable
import math

# 导入统一核心算法
from agi_unified_framework.core.unified_algorithms import (
    UnifiedAlgorithmConfig,
    ConstraintManager,
    PhysicsConstraint,
    ConsistencyConstraint,
    ConstraintPriority,
    ConstraintCheckResult,
    ConstraintViolation,
)


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class MotionVector:
    """
    运动向量
    
    Attributes:
        dx: x方向位移
        dy: y方向位移
        dz: z方向位移（可选）
        timestamp: 时间戳
    """
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    timestamp: float = 0.0
    
    def magnitude(self) -> float:
        """计算模长"""
        return math.sqrt(self.dx**2 + self.dy**2 + self.dz**2)
    
    def to_list(self) -> List[float]:
        """转换为列表"""
        return [self.dx, self.dy, self.dz]


@dataclass
class ObjectState:
    """
    物体状态
    
    Attributes:
        position: 位置
        velocity: 速度
        acceleration: 加速度
        mass: 质量
        timestamp: 时间戳
    """
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    acceleration: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    mass: float = 1.0
    timestamp: float = 0.0
    
    def kinetic_energy(self) -> float:
        """计算动能"""
        v_squared = sum(v**2 for v in self.velocity)
        return 0.5 * self.mass * v_squared


@dataclass
class CorrectionResult:
    """
    校正结果
    
    Attributes:
        corrected_data: 校正后的数据
        violations_found: 发现的违反数
        violations_fixed: 修复的违反数
        correction_applied: 是否应用了校正
        details: 详细信息
    """
    corrected_data: Any
    violations_found: int = 0
    violations_fixed: int = 0
    correction_applied: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 物理校正器
# ============================================================================

class PhysicsCorrector:
    """
    物理校正器
    
    使用统一物理约束进行视频后处理。
    
    Attributes:
        config: 配置
        constraint_manager: 约束管理器
        physics_params: 物理参数
    """
    
    def __init__(self,
                 use_unified_core: bool = True,
                 config: Optional[UnifiedAlgorithmConfig] = None):
        """
        初始化物理校正器
        
        Args:
            use_unified_core: 是否使用统一核心
            config: 算法配置
        """
        self.use_unified_core = use_unified_core
        self.config = config or UnifiedAlgorithmConfig.video_optimized_config()
        
        # 初始化约束管理器
        if use_unified_core:
            self.constraint_manager = ConstraintManager(strict_mode=False)
            self._init_physics_constraints()
        else:
            self.constraint_manager = None
        
        # 物理参数
        self.physics_params = {
            'gravity': 9.8,
            'max_velocity': 50.0,
            'max_acceleration': 100.0,
            'friction': 0.1,
            'elasticity': 0.3,
            'time_step': 1.0 / 30.0,  # 假设30fps
        }
    
    def _init_physics_constraints(self) -> None:
        """初始化物理约束"""
        if self.constraint_manager is None:
            return
        
        # 速度限制约束
        velocity_constraint = PhysicsConstraint(
            name="velocity_limit",
            constraint_type="velocity_limit",
            priority=ConstraintPriority.SOFT,
            tolerance=0.1,
            max_velocity=self.physics_params['max_velocity']
        )
        self.constraint_manager.add_constraint(velocity_constraint)
        
        # 边界约束
        boundary_constraint = PhysicsConstraint(
            name="spatial_boundary",
            constraint_type="boundary",
            priority=ConstraintPriority.HARD,
            tolerance=0.01,
            bounds=[[-100, 100], [-100, 100], [-100, 100]]
        )
        self.constraint_manager.add_constraint(boundary_constraint)
        
        # 能量守恒约束
        energy_constraint = PhysicsConstraint(
            name="energy_conservation",
            constraint_type="energy_conservation",
            priority=ConstraintPriority.SOFT,
            tolerance=0.1,
            energy_tolerance=0.05
        )
        self.constraint_manager.add_constraint(energy_constraint)
        
        # 碰撞约束
        collision_constraint = PhysicsConstraint(
            name="collision_avoidance",
            constraint_type="collision",
            priority=ConstraintPriority.HARD,
            tolerance=0.01,
            min_distance=0.5
        )
        self.constraint_manager.add_constraint(collision_constraint)
    
    def correct_frame_sequence(self, 
                               frames: List[Dict[str, Any]]) -> CorrectionResult:
        """
        校正帧序列
        
        Args:
            frames: 帧数据列表，每帧包含位置、速度等信息
            
        Returns:
            校正结果
        """
        if not frames:
            return CorrectionResult(corrected_data=frames)
        
        corrected_frames = list(frames)
        violations_found = 0
        violations_fixed = 0
        
        # 步骤1: 检查约束
        if self.use_unified_core and self.constraint_manager:
            for i, frame in enumerate(corrected_frames):
                result = self.constraint_manager.check_all(frame)
                violations_found += len(result.violations)
                
                if not result.is_satisfied:
                    # 修复违反
                    corrected = self.constraint_manager.repair_all(frame)
                    corrected_frames[i] = corrected
                    violations_fixed += len(result.violations)
        
        # 步骤2: 平滑运动
        corrected_frames = self._smooth_motion(corrected_frames)
        
        # 步骤3: 验证时序一致性
        corrected_frames = self._ensure_temporal_consistency(corrected_frames)
        
        correction_applied = violations_fixed > 0 or corrected_frames != frames
        
        return CorrectionResult(
            corrected_data=corrected_frames,
            violations_found=violations_found,
            violations_fixed=violations_fixed,
            correction_applied=correction_applied,
            details={
                'num_frames': len(frames),
                'physics_params': self.physics_params.copy()
            }
        )
    
    def correct_motion_vectors(self, 
                               motion_vectors: List[MotionVector]) -> List[MotionVector]:
        """
        校正运动向量
        
        Args:
            motion_vectors: 运动向量列表
            
        Returns:
            校正后的运动向量
        """
        if not motion_vectors:
            return motion_vectors
        
        corrected = []
        max_vel = self.physics_params['max_velocity']
        
        for mv in motion_vectors:
            # 检查速度限制
            if mv.magnitude() > max_vel:
                # 缩放速度
                scale = max_vel / mv.magnitude()
                mv = MotionVector(
                    dx=mv.dx * scale,
                    dy=mv.dy * scale,
                    dz=mv.dz * scale,
                    timestamp=mv.timestamp
                )
            
            corrected.append(mv)
        
        # 平滑运动向量
        corrected = self._smooth_vectors(corrected)
        
        return corrected
    
    def correct_object_states(self, 
                              states: List[ObjectState]) -> List[ObjectState]:
        """
        校正物体状态
        
        Args:
            states: 物体状态列表
            
        Returns:
            校正后的状态列表
        """
        if not states:
            return states
        
        corrected = list(states)
        dt = self.physics_params['time_step']
        
        for i in range(1, len(corrected)):
            prev_state = corrected[i-1]
            curr_state = corrected[i]
            
            # 验证物理一致性
            expected_position = [
                prev_state.position[j] + prev_state.velocity[j] * dt + 
                0.5 * prev_state.acceleration[j] * dt**2
                for j in range(3)
            ]
            
            # 计算位置偏差
            position_error = [
                curr_state.position[j] - expected_position[j]
                for j in range(3)
            ]
            
            # 如果偏差过大，进行校正
            error_magnitude = math.sqrt(sum(e**2 for e in position_error))
            if error_magnitude > 1.0:  # 阈值
                # 平滑校正
                correction_factor = 0.5
                corrected[i] = ObjectState(
                    position=[
                        curr_state.position[j] - position_error[j] * correction_factor
                        for j in range(3)
                    ],
                    velocity=curr_state.velocity,
                    acceleration=curr_state.acceleration,
                    mass=curr_state.mass,
                    timestamp=curr_state.timestamp
                )
        
        return corrected
    
    def _smooth_motion(self, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        平滑运动
        
        Args:
            frames: 帧列表
            
        Returns:
            平滑后的帧列表
        """
        if len(frames) < 3:
            return frames
        
        smoothed = list(frames)
        
        # 简单的移动平均平滑
        for i in range(1, len(frames) - 1):
            if 'position' in frames[i]:
                prev_pos = frames[i-1].get('position', frames[i]['position'])
                curr_pos = frames[i]['position']
                next_pos = frames[i+1].get('position', frames[i]['position'])
                
                # 检查是否可平滑
                if (isinstance(prev_pos, (list, tuple)) and 
                    isinstance(curr_pos, (list, tuple)) and
                    isinstance(next_pos, (list, tuple))):
                    
                    smoothed_pos = [
                        (prev_pos[j] + curr_pos[j] + next_pos[j]) / 3
                        for j in range(min(len(prev_pos), len(curr_pos), len(next_pos)))
                    ]
                    
                    smoothed[i] = {**frames[i], 'position': smoothed_pos}
        
        return smoothed
    
    def _smooth_vectors(self, vectors: List[MotionVector]) -> List[MotionVector]:
        """
        平滑向量
        
        Args:
            vectors: 向量列表
            
        Returns:
            平滑后的向量列表
        """
        if len(vectors) < 3:
            return vectors
        
        smoothed = list(vectors)
        
        for i in range(1, len(vectors) - 1):
            smoothed[i] = MotionVector(
                dx=(vectors[i-1].dx + vectors[i].dx + vectors[i+1].dx) / 3,
                dy=(vectors[i-1].dy + vectors[i].dy + vectors[i+1].dy) / 3,
                dz=(vectors[i-1].dz + vectors[i].dz + vectors[i+1].dz) / 3,
                timestamp=vectors[i].timestamp
            )
        
        return smoothed
    
    def _ensure_temporal_consistency(self, 
                                     frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        确保时序一致性
        
        Args:
            frames: 帧列表
            
        Returns:
            一致的帧列表
        """
        if len(frames) < 2:
            return frames
        
        consistent = list(frames)
        
        # 检查相邻帧之间的跳跃
        for i in range(1, len(frames)):
            if 'position' in frames[i] and 'position' in frames[i-1]:
                curr_pos = frames[i]['position']
                prev_pos = frames[i-1]['position']
                
                if (isinstance(curr_pos, (list, tuple)) and 
                    isinstance(prev_pos, (list, tuple))):
                    
                    # 计算位移
                    displacement = [
                        curr_pos[j] - prev_pos[j]
                        for j in range(min(len(curr_pos), len(prev_pos)))
                    ]
                    
                    distance = math.sqrt(sum(d**2 for d in displacement))
                    max_distance = self.physics_params['max_velocity'] * self.physics_params['time_step']
                    
                    # 如果位移过大，进行插值
                    if distance > max_distance * 2:
                        # 线性插值
                        alpha = 0.5
                        interpolated_pos = [
                            prev_pos[j] + displacement[j] * alpha
                            for j in range(len(displacement))
                        ]
                        
                        consistent[i] = {**frames[i], 'position': interpolated_pos}
        
        return consistent
    
    def validate_physics(self, data: Any) -> Dict[str, Any]:
        """
        验证物理正确性
        
        Args:
            data: 要验证的数据
            
        Returns:
            验证结果
        """
        if not self.use_unified_core or self.constraint_manager is None:
            return {'enabled': False}
        
        result = self.constraint_manager.check_all(data)
        
        return {
            'enabled': True,
            'is_valid': result.is_satisfied,
            'score': result.score,
            'violations': [
                {
                    'constraint': v.constraint_name,
                    'type': v.constraint_type,
                    'severity': v.severity,
                    'message': v.message,
                    'suggestion': v.suggestion
                }
                for v in result.violations
            ],
            'total_violations': len(result.violations)
        }
    
    def set_physics_param(self, param: str, value: float) -> None:
        """
        设置物理参数
        
        Args:
            param: 参数名
            value: 参数值
        """
        if param in self.physics_params:
            self.physics_params[param] = value
            
            # 更新约束中的参数
            if self.use_unified_core and self.constraint_manager:
                for constraint in self.constraint_manager.constraints:
                    if isinstance(constraint, PhysicsConstraint):
                        if param == 'max_velocity' and constraint.constraint_type == 'velocity_limit':
                            constraint.params['max_velocity'] = value
                        elif param == 'bounds' and constraint.constraint_type == 'boundary':
                            constraint.params['bounds'] = value
    
    def get_physics_params(self) -> Dict[str, float]:
        """
        获取物理参数
        
        Returns:
            物理参数字典
        """
        return self.physics_params.copy()


# ============================================================================
# 运动平滑器
# ============================================================================

class MotionSmoother:
    """
    运动平滑器
    
    专门用于平滑运动轨迹。
    
    Attributes:
        window_size: 平滑窗口大小
        smoothing_factor: 平滑因子
    """
    
    def __init__(self, 
                 window_size: int = 5,
                 smoothing_factor: float = 0.3):
        """
        初始化运动平滑器
        
        Args:
            window_size: 平滑窗口大小
            smoothing_factor: 平滑因子
        """
        self.window_size = window_size
        self.smoothing_factor = smoothing_factor
    
    def smooth_trajectory(self, 
                          positions: List[List[float]]) -> List[List[float]]:
        """
        平滑轨迹
        
        Args:
            positions: 位置列表
            
        Returns:
            平滑后的位置列表
        """
        if len(positions) < self.window_size:
            return positions
        
        smoothed = []
        half_window = self.window_size // 2
        
        for i in range(len(positions)):
            # 获取窗口内的点
            window_start = max(0, i - half_window)
            window_end = min(len(positions), i + half_window + 1)
            window = positions[window_start:window_end]
            
            # 计算加权平均
            weights = self._compute_weights(len(window))
            
            smoothed_pos = []
            for dim in range(len(positions[i])):
                weighted_sum = sum(
                    window[j][dim] * weights[j] 
                    for j in range(len(window))
                    if dim < len(window[j])
                )
                smoothed_pos.append(weighted_sum)
            
            # 混合原始和平滑结果
            final_pos = [
                positions[i][dim] * (1 - self.smoothing_factor) + 
                smoothed_pos[dim] * self.smoothing_factor
                for dim in range(len(positions[i]))
            ]
            
            smoothed.append(final_pos)
        
        return smoothed
    
    def _compute_weights(self, size: int) -> List[float]:
        """
        计算平滑权重
        
        Args:
            size: 窗口大小
            
        Returns:
            权重列表
        """
        # 高斯权重
        center = size // 2
        weights = [
            math.exp(-0.5 * ((i - center) / (size / 3)) ** 2)
            for i in range(size)
        ]
        
        # 归一化
        total = sum(weights)
        return [w / total for w in weights]


# ============================================================================
# 约束验证器
# ============================================================================

class ConstraintValidator:
    """
    约束验证器
    
    专门用于验证约束满足情况。
    
    Attributes:
        corrector: 物理校正器
    """
    
    def __init__(self, corrector: Optional[PhysicsCorrector] = None):
        """
        初始化约束验证器
        
        Args:
            corrector: 物理校正器
        """
        self.corrector = corrector or PhysicsCorrector()
    
    def validate_sequence(self, 
                          frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        验证序列
        
        Args:
            frames: 帧列表
            
        Returns:
            验证结果
        """
        results = []
        total_violations = 0
        
        for i, frame in enumerate(frames):
            validation = self.corrector.validate_physics(frame)
            
            if validation.get('enabled', False):
                results.append({
                    'frame_idx': i,
                    'is_valid': validation['is_valid'],
                    'score': validation['score'],
                    'violations': validation['violations']
                })
                total_violations += validation['total_violations']
        
        # 计算整体统计
        valid_frames = sum(1 for r in results if r['is_valid'])
        avg_score = sum(r['score'] for r in results) / len(results) if results else 0.0
        
        return {
            'total_frames': len(frames),
            'valid_frames': valid_frames,
            'invalid_frames': len(frames) - valid_frames,
            'total_violations': total_violations,
            'average_score': avg_score,
            'frame_results': results
        }
    
    def generate_report(self, 
                        validation_result: Dict[str, Any]) -> str:
        """
        生成验证报告
        
        Args:
            validation_result: 验证结果
            
        Returns:
            报告字符串
        """
        lines = [
            "=" * 50,
            "物理约束验证报告",
            "=" * 50,
            f"总帧数: {validation_result['total_frames']}",
            f"有效帧: {validation_result['valid_frames']}",
            f"无效帧: {validation_result['invalid_frames']}",
            f"总违反数: {validation_result['total_violations']}",
            f"平均分数: {validation_result['average_score']:.4f}",
            "=" * 50,
        ]
        
        # 添加每帧的详细信息
        for frame_result in validation_result.get('frame_results', []):
            if not frame_result['is_valid']:
                lines.append(f"\n帧 {frame_result['frame_idx']}:")
                lines.append(f"  分数: {frame_result['score']:.4f}")
                for v in frame_result['violations']:
                    lines.append(f"  - {v['constraint']}: {v['message']}")
        
        return "\n".join(lines)


# ============================================================================
# 便捷函数
# ============================================================================

def create_physics_corrector(use_unified_core: bool = True,
                              **kwargs) -> PhysicsCorrector:
    """
    创建物理校正器
    
    Args:
        use_unified_core: 是否使用统一核心
        **kwargs: 其他参数
        
    Returns:
        物理校正器
    """
    return PhysicsCorrector(use_unified_core=use_unified_core, **kwargs)


def correct_video_physics(frames: List[Dict[str, Any]],
                          use_unified_core: bool = True,
                          **kwargs) -> CorrectionResult:
    """
    便捷函数：校正视频物理
    
    Args:
        frames: 帧列表
        use_unified_core: 是否使用统一核心
        **kwargs: 其他参数
        
    Returns:
        校正结果
    """
    corrector = create_physics_corrector(use_unified_core=use_unified_core, **kwargs)
    return corrector.correct_frame_sequence(frames)


def validate_video_physics(frames: List[Dict[str, Any]],
                           use_unified_core: bool = True,
                           **kwargs) -> Dict[str, Any]:
    """
    便捷函数：验证视频物理
    
    Args:
        frames: 帧列表
        use_unified_core: 是否使用统一核心
        **kwargs: 其他参数
        
    Returns:
        验证结果
    """
    corrector = create_physics_corrector(use_unified_core=use_unified_core, **kwargs)
    validator = ConstraintValidator(corrector)
    return validator.validate_sequence(frames)


# ============================================================================
# 导出列表
# ============================================================================

__all__ = [
    # 数据结构
    'MotionVector',
    'ObjectState',
    'CorrectionResult',
    
    # 校正器
    'PhysicsCorrector',
    'MotionSmoother',
    'ConstraintValidator',
    
    # 便捷函数
    'create_physics_corrector',
    'correct_video_physics',
    'validate_video_physics',
]
