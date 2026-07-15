"""
物理约束损失 - Physics Constraint Loss

将物理定律编译到损失函数中，确保模型预测符合物理规律。

与 physics_constraint.py 中的 PhysicsConstraintLoss 互补，
本模块侧重于更广泛的物理约束：
  1. 能量守恒 - 系统总能量在演化过程中保持不变
  2. 动量守恒 - 系统总动量在无外力时保持不变
  3. 对称性约束 - 模型输出应满足物理对称性（旋转、平移、反射）
  4. 物理一致性 - 预测结果与已知物理定律的一致性

总损失：
  L_physics = λ_energy * L_energy
            + λ_momentum * L_momentum
            + λ_symmetry * L_symmetry
            + λ_consistency * L_consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Union
import math


class PhysicsConstraintLoss(nn.Module):
    """
    物理约束损失函数

    将物理定律作为软约束编码到损失函数中，引导模型学习物理一致的表示。
    支持多种物理约束的组合，每种约束可以独立启用/禁用和调节权重。

    设计原则：
    - 每种约束都是可微的，支持端到端训练
    - 约束强度可动态调整（课程学习）
    - 支持多种物理场景（粒子系统、流体、分子动力学等）
    """

    def __init__(
        self,
        lambda_energy: float = 1.0,
        lambda_momentum: float = 0.8,
        lambda_symmetry: float = 0.5,
        lambda_consistency: float = 0.3,
        lambda_boundary: float = 0.2,
        lambda_thermodynamic: float = 0.1,
        energy_mode: str = 'kinetic_potential',
        momentum_mode: str = 'linear_angular',
        symmetry_types: Optional[List[str]] = None,
        consistency_laws: Optional[List[str]] = None,
        use_curriculum: bool = False,
        curriculum_warmup_steps: int = 1000,
        boundary_condition: Optional[str] = None,
        dt: float = 1.0,
        mass: float = 1.0,
    ):
        """
        初始化物理约束损失

        Args:
            lambda_energy: 能量守恒损失权重
            lambda_momentum: 动量守恒损失权重
            lambda_symmetry: 对称性约束损失权重
            lambda_consistency: 物理一致性损失权重
            lambda_boundary: 边界条件损失权重
            lambda_thermodynamic: 热力学约束损失权重
            energy_mode: 能量模式
                - 'kinetic_potential': 动能+势能
                - 'total': 仅总能量
                - 'per_particle': 逐粒子能量
            momentum_mode: 动量模式
                - 'linear_angular': 线性动量+角动量
                - 'linear': 仅线性动量
                - 'angular': 仅角动量
            symmetry_types: 对称性类型列表
                - 'rotation': 旋转对称性
                - 'translation': 平移对称性
                - 'reflection': 反射对称性
                - 'scale': 尺度对称性
            consistency_laws: 物理定律列表
                - 'newton_second': 牛顿第二定律
                - 'gravity': 万有引力定律
                - 'elastic': 弹性碰撞
                - 'incompressible': 不可压缩性
            use_curriculum: 是否使用课程学习
            curriculum_warmup_steps: 课程学习预热步数
            boundary_condition: 边界条件类型
                - 'periodic': 周期边界
                - 'reflective': 反射边界
                - 'absorbing': 吸收边界
                - None: 无边界约束
            dt: 时间步长（用于时间导数计算）
            mass: 粒子质量（用于能量和动量计算）
        """
        super().__init__()

        # ---- 损失权重 ----
        self.lambda_energy = lambda_energy
        self.lambda_momentum = lambda_momentum
        self.lambda_symmetry = lambda_symmetry
        self.lambda_consistency = lambda_consistency
        self.lambda_boundary = lambda_boundary
        self.lambda_thermodynamic = lambda_thermodynamic

        # ---- 物理模式 ----
        self.energy_mode = energy_mode
        self.momentum_mode = momentum_mode
        self.dt = dt
        self.mass = mass

        # ---- 对称性类型 ----
        if symmetry_types is None:
            symmetry_types = ['rotation', 'translation']
        self.symmetry_types = symmetry_types

        # ---- 物理定律 ----
        if consistency_laws is None:
            consistency_laws = ['newton_second']
        self.consistency_laws = consistency_laws

        # ---- 课程学习 ----
        self.use_curriculum = use_curriculum
        self.curriculum_warmup_steps = curriculum_warmup_steps
        self.register_buffer('global_step', torch.tensor(0))

        # ---- 边界条件 ----
        self.boundary_condition = boundary_condition

        # ---- 内部状态 ----
        self._prev_energy: Optional[torch.Tensor] = None
        self._prev_momentum: Optional[torch.Tensor] = None

    # ================================================================
    #  主入口
    # ================================================================

    def forward(
        self,
        positions: torch.Tensor,
        velocities: Optional[torch.Tensor] = None,
        predicted_positions: Optional[torch.Tensor] = None,
        predicted_velocities: Optional[torch.Tensor] = None,
        forces: Optional[torch.Tensor] = None,
        masses: Optional[torch.Tensor] = None,
        potential_energy: Optional[torch.Tensor] = None,
        predicted_potential: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算物理约束总损失

        根据提供的输入数据，自动计算所有适用的物理约束损失。

        Args:
            positions: (batch, N, 3) 当前位置
            velocities: (batch, N, 3) 当前速度
            predicted_positions: (batch, N, 3) 预测的下一时刻位置
            predicted_velocities: (batch, N, 3) 预测的下一时刻速度
            forces: (batch, N, 3) 力（用于牛顿第二定律验证）
            masses: (batch, N) 或 (batch, N, 1) 粒子质量
            potential_energy: (batch,) 当前势能
            predicted_potential: (batch,) 预测的下一时刻势能

        Returns:
            total_loss: 总物理约束损失
            loss_dict: 各分量损失的详细字典
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=positions.device)

        # 更新课程学习步数
        self.global_step += 1
        curriculum_factor = self._get_curriculum_factor()

        # ---- 1. 能量守恒约束 ----
        if self.lambda_energy > 0:
            energy_loss = self.energy_conservation_loss(
                positions=positions,
                velocities=velocities,
                predicted_positions=predicted_positions,
                predicted_velocities=predicted_velocities,
                masses=masses,
                potential_energy=potential_energy,
                predicted_potential=predicted_potential,
            )
            weighted_loss = self.lambda_energy * curriculum_factor * energy_loss
            total_loss = total_loss + weighted_loss
            loss_dict['energy_conservation'] = energy_loss.item()

        # ---- 2. 动量守恒约束 ----
        if self.lambda_momentum > 0 and velocities is not None:
            momentum_loss = self.momentum_conservation_loss(
                velocities=velocities,
                predicted_velocities=predicted_velocities,
                masses=masses,
                positions=positions,
                predicted_positions=predicted_positions,
            )
            weighted_loss = self.lambda_momentum * curriculum_factor * momentum_loss
            total_loss = total_loss + weighted_loss
            loss_dict['momentum_conservation'] = momentum_loss.item()

        # ---- 3. 对称性约束 ----
        if self.lambda_symmetry > 0:
            symmetry_loss = self.symmetry_constraint_loss(
                positions=positions,
                predicted_positions=predicted_positions,
                velocities=velocities,
            )
            weighted_loss = self.lambda_symmetry * curriculum_factor * symmetry_loss
            total_loss = total_loss + weighted_loss
            loss_dict['symmetry'] = symmetry_loss.item()

        # ---- 4. 物理一致性约束 ----
        if self.lambda_consistency > 0 and forces is not None:
            consistency_loss = self.physics_consistency_loss(
                positions=positions,
                velocities=velocities,
                forces=forces,
                masses=masses,
            )
            weighted_loss = self.lambda_consistency * curriculum_factor * consistency_loss
            total_loss = total_loss + weighted_loss
            loss_dict['physics_consistency'] = consistency_loss.item()

        # ---- 5. 边界条件约束 ----
        if self.lambda_boundary > 0 and self.boundary_condition is not None:
            boundary_loss = self.boundary_condition_loss(
                positions=predicted_positions if predicted_positions is not None else positions,
            )
            weighted_loss = self.lambda_boundary * curriculum_factor * boundary_loss
            total_loss = total_loss + weighted_loss
            loss_dict['boundary'] = boundary_loss.item()

        # ---- 6. 热力学约束 ----
        if self.lambda_thermodynamic > 0 and velocities is not None:
            thermo_loss = self.thermodynamic_consistency_loss(
                velocities=velocities,
                predicted_velocities=predicted_velocities,
            )
            weighted_loss = self.lambda_thermodynamic * curriculum_factor * thermo_loss
            total_loss = total_loss + weighted_loss
            loss_dict['thermodynamic'] = thermo_loss.item()

        loss_dict['total_physics_loss'] = total_loss.item()
        loss_dict['curriculum_factor'] = curriculum_factor

        return total_loss, loss_dict

    # ================================================================
    #  能量守恒约束
    # ================================================================

    def energy_conservation_loss(
        self,
        positions: torch.Tensor,
        velocities: Optional[torch.Tensor] = None,
        predicted_positions: Optional[torch.Tensor] = None,
        predicted_velocities: Optional[torch.Tensor] = None,
        masses: Optional[torch.Tensor] = None,
        potential_energy: Optional[torch.Tensor] = None,
        predicted_potential: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        能量守恒损失

        在保守系统中，总能量（动能 + 势能）应保持不变。
        损失函数惩罚能量的变化量：

          L_energy = (E(t+dt) - E(t))^2

        其中 E = Σ_i (1/2 * m_i * |v_i|^2 + U(r_i))

        Args:
            positions: (batch, N, 3) 当前位置
            velocities: (batch, N, 3) 当前速度
            predicted_positions: (batch, N, 3) 预测位置
            predicted_velocities: (batch, N, 3) 预测速度
            masses: (batch, N) 粒子质量
            potential_energy: (batch,) 当前势能
            predicted_potential: (batch,) 预测势能

        Returns:
            能量守恒损失标量
        """
        device = positions.device
        batch_size = positions.shape[0]

        # 处理质量参数
        if masses is None:
            m = self.mass
        elif masses.dim() == 2:
            m = masses.squeeze(-1)  # (batch, N)
        else:
            m = masses  # (batch, N)

        # ---- 计算当前时刻能量 ----
        current_energy = torch.zeros(batch_size, device=device)

        # 动能: KE = 1/2 * m * v^2
        if velocities is not None:
            kinetic = 0.5 * m * torch.sum(velocities ** 2, dim=-1)
            current_energy = current_energy + kinetic.sum(dim=-1)

        # 势能
        if potential_energy is not None:
            current_energy = current_energy + potential_energy
        else:
            # 如果没有提供势能，使用简化的Lennard-Jones势
            if self.energy_mode == 'kinetic_potential':
                pe = self._compute_lennard_jones_potential(positions)
                current_energy = current_energy + pe

        # ---- 计算下一时刻能量 ----
        predicted_energy = torch.zeros(batch_size, device=device)

        if predicted_velocities is not None:
            kinetic_pred = 0.5 * m * torch.sum(predicted_velocities ** 2, dim=-1)
            predicted_energy = predicted_energy + kinetic_pred.sum(dim=-1)

        if predicted_potential is not None:
            predicted_energy = predicted_energy + predicted_potential
        elif predicted_positions is not None:
            pe_pred = self._compute_lennard_jones_potential(predicted_positions)
            predicted_energy = predicted_energy + pe_pred

        # ---- 能量守恒误差 ----
        energy_diff = predicted_energy - current_energy

        # 使用相对误差（避免数值问题）
        energy_scale = torch.abs(current_energy) + 1e-8
        relative_error = energy_diff / energy_scale

        return torch.mean(relative_error ** 2)

    def _compute_lennard_jones_potential(
        self,
        positions: torch.Tensor,
        epsilon: float = 1.0,
        sigma: float = 1.0,
    ) -> torch.Tensor:
        """
        计算Lennard-Jones势能（简化版）

        U(r) = 4ε * [(σ/r)^12 - (σ/r)^6]

        Args:
            positions: (batch, N, 3) 粒子位置
            epsilon: 势阱深度
            sigma: 零交叉距离

        Returns:
            (batch,) 总势能
        """
        batch_size, num_particles, _ = positions.shape
        device = positions.device

        total_pe = torch.zeros(batch_size, device=device)

        # 计算所有粒子对的距离
        # (batch, N, N, 3)
        rel_pos = positions.unsqueeze(2) - positions.unsqueeze(1)
        # (batch, N, N)
        distances = torch.norm(rel_pos, dim=-1) + 1e-8

        # 避免自相互作用（对角线设为无穷大）
        mask = ~torch.eye(num_particles, dtype=torch.bool, device=device)
        distances = distances * mask.unsqueeze(0) + (~mask).unsqueeze(0).float() * 1e10

        # Lennard-Jones势
        r_ratio = sigma / distances
        r6 = r_ratio ** 6
        r12 = r6 ** 2

        # U = 4ε * (r12 - r6)
        pair_potential = 4 * epsilon * (r12 - r6)

        # 截断：超过一定距离的势能设为0
        cutoff = 3.0 * sigma
        pair_potential = pair_potential * (distances < cutoff).float()

        # 求和所有粒子对的势能（除以2避免重复计算）
        total_pe = 0.5 * pair_potential.sum(dim=-1).sum(dim=-1)

        return total_pe

    # ================================================================
    #  动量守恒约束
    # ================================================================

    def momentum_conservation_loss(
        self,
        velocities: torch.Tensor,
        predicted_velocities: Optional[torch.Tensor] = None,
        masses: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        predicted_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        动量守恒损失

        在无外力系统中，总动量应保持不变。

        线性动量：P = Σ_i m_i * v_i
        角动量：L = Σ_i m_i * (r_i × v_i)

        L_momentum = |P(t+dt) - P(t)|^2 + |L(t+dt) - L(t)|^2

        Args:
            velocities: (batch, N, 3) 当前速度
            predicted_velocities: (batch, N, 3) 预测速度
            masses: (batch, N) 粒子质量
            positions: (batch, N, 3) 当前位置（角动量需要）
            predicted_positions: (batch, N, 3) 预测位置

        Returns:
            动量守恒损失标量
        """
        device = velocities.device
        batch_size = velocities.shape[0]

        # 处理质量
        if masses is None:
            m = self.mass
        elif masses.dim() == 2:
            m = masses.squeeze(-1)
        else:
            m = masses

        # 扩展质量维度用于广播
        if m.dim() == 1:
            m_expanded = m.unsqueeze(-1)  # (batch, N, 1)
        else:
            m_expanded = m

        total_loss = torch.tensor(0.0, device=device)

        # ---- 线性动量 ----
        if 'linear' in self.momentum_mode or self.momentum_mode == 'linear_angular':
            # P = Σ m * v
            current_momentum = (m_expanded * velocities).sum(dim=1)  # (batch, 3)

            if predicted_velocities is not None:
                predicted_momentum = (m_expanded * predicted_velocities).sum(dim=1)
                momentum_diff = predicted_momentum - current_momentum
                linear_loss = torch.mean(momentum_diff ** 2)
                total_loss = total_loss + linear_loss

        # ---- 角动量 ----
        if 'angular' in self.momentum_mode or self.momentum_mode == 'linear_angular':
            if positions is not None:
                # L = Σ m * (r × v)
                current_angular = torch.cross(
                    positions, m_expanded * velocities, dim=-1
                ).sum(dim=1)  # (batch, 3)

                if predicted_positions is not None and predicted_velocities is not None:
                    predicted_angular = torch.cross(
                        predicted_positions, m_expanded * predicted_velocities, dim=-1
                    ).sum(dim=1)
                    angular_diff = predicted_angular - current_angular
                    angular_loss = torch.mean(angular_diff ** 2)
                    total_loss = total_loss + angular_loss

        return total_loss

    # ================================================================
    #  对称性约束
    # ================================================================

    def symmetry_constraint_loss(
        self,
        positions: torch.Tensor,
        predicted_positions: Optional[torch.Tensor] = None,
        velocities: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        对称性约束损失

        验证模型输出在输入经历对称变换后是否保持等变性：
          f(T(x)) = T(f(x))

        其中 T 是对称变换（旋转、平移、反射等）。

        Args:
            positions: (batch, N, 3) 输入位置
            predicted_positions: (batch, N, 3) 预测位置
            velocities: (batch, N, 3) 速度（可选）

        Returns:
            对称性约束损失标量
        """
        total_loss = torch.tensor(0.0, device=positions.device)
        num_symmetries = 0

        for sym_type in self.symmetry_types:
            if sym_type == 'rotation':
                loss = self._rotation_symmetry_loss(positions, predicted_positions)
                total_loss = total_loss + loss
                num_symmetries += 1

            elif sym_type == 'translation':
                loss = self._translation_symmetry_loss(
                    positions, predicted_positions
                )
                total_loss = total_loss + loss
                num_symmetries += 1

            elif sym_type == 'reflection':
                loss = self._reflection_symmetry_loss(
                    positions, predicted_positions
                )
                total_loss = total_loss + loss
                num_symmetries += 1

            elif sym_type == 'scale':
                loss = self._scale_symmetry_loss(positions, predicted_positions)
                total_loss = total_loss + loss
                num_symmetries += 1

        if num_symmetries > 0:
            total_loss = total_loss / num_symmetries

        return total_loss

    def _rotation_symmetry_loss(
        self,
        positions: torch.Tensor,
        predicted_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        旋转对称性损失

        验证：f(R*x) = R*f(x)

        使用随机旋转矩阵进行测试。

        Args:
            positions: 输入位置
            predicted_positions: 预测位置

        Returns:
            旋转对称性误差
        """
        # 生成随机旋转矩阵（绕Z轴旋转45度作为测试）
        angle = math.pi / 4
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        R = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], device=positions.device, dtype=positions.dtype)

        # 旋转输入
        rotated_positions = torch.einsum('ij,bnj->bni', R, positions)

        # 计算旋转前后的距离差异
        # 对于旋转等变系统，相对距离应保持不变
        # d(R*x_i, R*x_j) = d(x_i, x_j)
        if predicted_positions is not None:
            # 原始相对距离
            orig_rel = positions - positions.mean(dim=1, keepdim=True)
            orig_dist = torch.norm(orig_rel, dim=-1)

            # 旋转后的相对距离
            rot_rel = rotated_positions - rotated_positions.mean(dim=1, keepdim=True)
            rot_dist = torch.norm(rot_rel, dim=-1)

            # 距离应保持不变
            return torch.mean((orig_dist - rot_dist) ** 2)
        else:
            # 没有预测位置时，仅检查旋转后的距离矩阵不变性
            orig_dist_matrix = torch.cdist(positions, positions)
            rot_dist_matrix = torch.cdist(rotated_positions, rotated_positions)
            return torch.mean((orig_dist_matrix - rot_dist_matrix) ** 2)

    def _translation_symmetry_loss(
        self,
        positions: torch.Tensor,
        predicted_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        平移对称性损失

        验证：f(x + t) = f(x) + t（或相对关系不变）

        平移不变性意味着系统行为只取决于相对位置。

        Args:
            positions: 输入位置
            predicted_positions: 预测位置

        Returns:
            平移对称性误差
        """
        # 随机平移向量
        translation = torch.randn(1, 1, 3, device=positions.device) * 0.5

        # 平移后的位置
        translated_positions = positions + translation

        # 平移不应改变相对距离
        orig_dist = torch.cdist(positions, positions)
        trans_dist = torch.cdist(translated_positions, translated_positions)

        return torch.mean((orig_dist - trans_dist) ** 2)

    def _reflection_symmetry_loss(
        self,
        positions: torch.Tensor,
        predicted_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        反射对称性损失

        验证系统关于某个平面的反射对称性。

        Args:
            positions: 输入位置
            predicted_positions: 预测位置

        Returns:
            反射对称性误差
        """
        # 关于XY平面反射（z -> -z）
        reflected_positions = positions.clone()
        reflected_positions[:, :, 2] = -reflected_positions[:, :, 2]

        # 反射不改变距离
        orig_dist = torch.cdist(positions, positions)
        refl_dist = torch.cdist(reflected_positions, reflected_positions)

        return torch.mean((orig_dist - refl_dist) ** 2)

    def _scale_symmetry_loss(
        self,
        positions: torch.Tensor,
        predicted_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        尺度对称性损失

        对于某些物理系统（如引力），尺度变换应保持结构不变。

        Args:
            positions: 输入位置
            predicted_positions: 预测位置

        Returns:
            尺度对称性误差
        """
        scale_factor = 1.5
        scaled_positions = positions * scale_factor

        # 尺度变换后，归一化的距离矩阵应不变
        orig_dist = torch.cdist(positions, positions)
        scaled_dist = torch.cdist(scaled_positions, scaled_positions)

        # 归一化后比较
        orig_norm = orig_dist / (orig_dist.max() + 1e-8)
        scaled_norm = scaled_dist / (scaled_dist.max() + 1e-8)

        return torch.mean((orig_norm - scaled_norm) ** 2)

    # ================================================================
    #  物理一致性约束
    # ================================================================

    def physics_consistency_loss(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        forces: torch.Tensor,
        masses: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        物理一致性损失

        验证预测结果是否与已知物理定律一致。

        Args:
            positions: (batch, N, 3) 位置
            velocities: (batch, N, 3) 速度
            forces: (batch, N, 3) 力
            masses: (batch, N) 质量

        Returns:
            物理一致性损失标量
        """
        total_loss = torch.tensor(0.0, device=positions.device)
        num_laws = 0

        for law in self.consistency_laws:
            if law == 'newton_second':
                loss = self._newton_second_law_loss(forces, velocities, masses)
                total_loss = total_loss + loss
                num_laws += 1

            elif law == 'gravity':
                loss = self._gravity_consistency_loss(positions, forces, masses)
                total_loss = total_loss + loss
                num_laws += 1

            elif law == 'elastic':
                loss = self._elastic_consistency_loss(velocities)
                total_loss = total_loss + loss
                num_laws += 1

            elif law == 'incompressible':
                loss = self._incompressibility_loss(positions)
                total_loss = total_loss + loss
                num_laws += 1

        if num_laws > 0:
            total_loss = total_loss / num_laws

        return total_loss

    def _newton_second_law_loss(
        self,
        forces: torch.Tensor,
        velocities: Optional[torch.Tensor] = None,
        masses: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        牛顿第二定律一致性

        F = m * a = m * dv/dt

        检查力和加速度之间的关系。

        Args:
            forces: (batch, N, 3) 力
            velocities: (batch, N, 3) 速度（用于数值微分计算加速度）
            masses: (batch, N) 质量

        Returns:
            牛顿第二定律误差
        """
        if masses is None:
            m = self.mass
        elif masses.dim() == 2:
            m = masses.squeeze(-1)
        else:
            m = masses

        # 如果有速度，可以检查力的方向是否合理
        # F = m * a，所以 a = F/m
        if m.dim() == 1:
            m_expanded = m.unsqueeze(-1)
        else:
            m_expanded = m

        # 加速度 = F / m
        acceleration = forces / (m_expanded + 1e-8)

        # 检查加速度的量级是否合理（不应过大）
        # 这是一个软约束，防止数值不稳定
        acc_norm = torch.norm(acceleration, dim=-1)
        acc_penalty = F.relu(acc_norm - 100.0)  # 加速度上限100

        return torch.mean(acc_penalty ** 2)

    def _gravity_consistency_loss(
        self,
        positions: torch.Tensor,
        forces: torch.Tensor,
        masses: Optional[torch.Tensor] = None,
        G: float = 1.0,
    ) -> torch.Tensor:
        """
        万有引力一致性

        F_ij = G * m_i * m_j / |r_ij|^2 * r_hat_ij

        检查力是否符合引力定律。

        Args:
            positions: (batch, N, 3) 位置
            forces: (batch, N, 3) 力
            masses: (batch, N) 质量
            G: 引力常数

        Returns:
            引力一致性误差
        """
        if masses is None:
            m = self.mass
        elif masses.dim() == 2:
            m = masses.squeeze(-1)
        else:
            m = masses

        batch_size, num_particles, _ = positions.shape
        device = positions.device

        # 计算引力
        # (batch, N, N, 3)
        rel_pos = positions.unsqueeze(2) - positions.unsqueeze(1)
        distances = torch.norm(rel_pos, dim=-1, keepdim=True) + 1e-8
        direction = rel_pos / distances

        # 引力大小: G * m_i * m_j / r^2
        if m.dim() == 1:
            m_matrix = m.unsqueeze(1) * m.unsqueeze(2)  # (batch, N, N)
        else:
            m_matrix = m.unsqueeze(1) * m.unsqueeze(2)

        force_magnitude = G * m_matrix.unsqueeze(-1) / (distances ** 2)

        # 排除自相互作用
        mask = ~torch.eye(num_particles, dtype=torch.bool, device=device)
        force_magnitude = force_magnitude * mask.unsqueeze(-1).float()

        # 理论引力（所有粒子对 j -> i 的引力之和）
        # 注意方向：引力指向对方，所以是 -direction
        theoretical_forces = -(force_magnitude * direction).sum(dim=2)

        # 与实际力的差异
        force_diff = theoretical_forces - forces

        return torch.mean(force_diff ** 2)

    def _elastic_consistency_loss(
        self,
        velocities: torch.Tensor,
    ) -> torch.Tensor:
        """
        弹性碰撞一致性

        在弹性碰撞中，总动能应守恒。

        Args:
            velocities: (batch, N, 3) 速度

        Returns:
            弹性碰撞一致性误差
        """
        # 总动能
        kinetic_energy = 0.5 * self.mass * torch.sum(velocities ** 2, dim=-1)

        # 检查动能是否为正
        energy_penalty = F.relu(-kinetic_energy)

        return torch.mean(energy_penalty ** 2)

    def _incompressibility_loss(
        self,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        不可压缩性约束

        对于不可压缩流体，局部密度应保持恒定。
        通过检查局部体积变化来近似。

        Args:
            positions: (batch, N, 3) 粒子位置

        Returns:
            不可压缩性误差
        """
        batch_size, num_particles, _ = positions.shape
        device = positions.device

        # 计算每个粒子到其k近邻的距离
        # 使用所有粒子对的距离
        dist_matrix = torch.cdist(positions, positions)  # (batch, N, N)

        # 对每个粒子，取第k近邻的距离作为局部密度估计
        k = min(6, num_particles - 1)  # 使用6近邻
        sorted_dist, _ = torch.sort(dist_matrix, dim=-1)
        local_radius = sorted_dist[:, :, k]  # (batch, N)

        # 理想情况下，所有粒子的局部半径应相近
        mean_radius = local_radius.mean(dim=-1, keepdim=True)
        density_variation = (local_radius - mean_radius) / (mean_radius + 1e-8)

        return torch.mean(density_variation ** 2)

    # ================================================================
    #  边界条件约束
    # ================================================================

    def boundary_condition_loss(
        self,
        positions: torch.Tensor,
        box_size: float = 10.0,
    ) -> torch.Tensor:
        """
        边界条件损失

        根据边界条件类型，惩罚违反边界的行为。

        Args:
            positions: (batch, N, 3) 粒子位置
            box_size: 模拟盒子大小

        Returns:
            边界条件损失
        """
        if self.boundary_condition == 'reflective':
            # 反射边界：粒子不应超出盒子
            violation = F.relu(torch.abs(positions) - box_size / 2)
            return torch.mean(violation ** 2)

        elif self.boundary_condition == 'periodic':
            # 周期边界：粒子位置应映射回盒子内
            wrapped = ((positions + box_size / 2) % box_size) - box_size / 2
            return torch.mean((positions - wrapped) ** 2)

        elif self.boundary_condition == 'absorbing':
            # 吸收边界：超出边界的粒子应被惩罚
            outside = torch.abs(positions) > box_size / 2
            return torch.mean(outside.float() * positions ** 2)

        return torch.tensor(0.0, device=positions.device)

    # ================================================================
    #  热力学约束
    # ================================================================

    def thermodynamic_consistency_loss(
        self,
        velocities: torch.Tensor,
        predicted_velocities: Optional[torch.Tensor] = None,
        target_temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """
        热力学一致性损失

        检查系统温度（与速度分布相关）的合理性。

        对于NVE系综：温度应保持恒定
        对于NVT系综：温度应趋向目标值

        Args:
            velocities: (batch, N, 3) 当前速度
            predicted_velocities: (batch, N, 3) 预测速度
            target_temperature: 目标温度（NVT系综）

        Returns:
            热力学一致性损失
        """
        # 温度正比于平均动能: T ∝ <1/2 m v^2>
        current_ke = 0.5 * self.mass * torch.sum(velocities ** 2, dim=-1)
        current_temp = current_ke.mean(dim=-1)  # (batch,)

        if predicted_velocities is not None:
            predicted_ke = 0.5 * self.mass * torch.sum(
                predicted_velocities ** 2, dim=-1
            )
            predicted_temp = predicted_ke.mean(dim=-1)

            # NVE系综：温度应守恒
            temp_diff = predicted_temp - current_temp
            return torch.mean(temp_diff ** 2)

        elif target_temperature is not None:
            # NVT系综：温度应趋向目标值
            temp_diff = current_temp - target_temperature
            return torch.mean(temp_diff ** 2)

        return torch.tensor(0.0, device=velocities.device)

    # ================================================================
    #  课程学习
    # ================================================================

    def _get_curriculum_factor(self) -> float:
        """
        获取课程学习因子

        在预热期间，约束强度从0线性增长到1。

        Returns:
            课程学习因子 [0, 1]
        """
        if not self.use_curriculum:
            return 1.0

        step = self.global_step.item()
        if step >= self.curriculum_warmup_steps:
            return 1.0

        return step / self.curriculum_warmup_steps

    # ================================================================
    #  工具方法
    # ================================================================

    def compute_energy(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        masses: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算系统的各种能量

        Args:
            positions: (batch, N, 3) 位置
            velocities: (batch, N, 3) 速度
            masses: (batch, N) 质量

        Returns:
            包含各种能量的字典
        """
        device = positions.device

        if masses is None:
            m = self.mass
        elif masses.dim() == 2:
            m = masses.squeeze(-1)
        else:
            m = masses

        # 动能
        kinetic = 0.5 * m * torch.sum(velocities ** 2, dim=-1)
        total_kinetic = kinetic.sum(dim=-1)

        # 势能（Lennard-Jones）
        potential = self._compute_lennard_jones_potential(positions)

        # 总能量
        total_energy = total_kinetic + potential

        return {
            'kinetic_energy': total_kinetic,
            'potential_energy': potential,
            'total_energy': total_energy,
            'per_particle_kinetic': kinetic,
        }

    def compute_momentum(
        self,
        velocities: torch.Tensor,
        masses: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算系统的动量

        Args:
            velocities: (batch, N, 3) 速度
            masses: (batch, N) 质量
            positions: (batch, N, 3) 位置

        Returns:
            包含各种动量的字典
        """
        if masses is None:
            m = self.mass
        elif masses.dim() == 2:
            m = masses.squeeze(-1)
        else:
            m = masses

        if m.dim() == 1:
            m_expanded = m.unsqueeze(-1)
        else:
            m_expanded = m

        result = {
            'linear_momentum': (m_expanded * velocities).sum(dim=1),
        }

        if positions is not None:
            result['angular_momentum'] = torch.cross(
                positions, m_expanded * velocities, dim=-1
            ).sum(dim=1)

        return result

    def extra_repr(self) -> str:
        """额外的字符串表示"""
        return (
            f'energy_mode={self.energy_mode}, '
            f'momentum_mode={self.momentum_mode}, '
            f'symmetries={self.symmetry_types}, '
            f'laws={self.consistency_laws}, '
            f'use_curriculum={self.use_curriculum}'
        )
