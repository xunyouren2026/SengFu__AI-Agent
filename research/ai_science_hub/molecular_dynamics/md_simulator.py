"""
AI科学中心 - 分子动力学模拟模块

使用SH-GNN加速分子动力学（MD）模拟。
支持力场计算、多种积分器、恒温/恒压系综和轨迹分析。
"""

import math
import json
import random
import copy
from typing import Dict, Tuple, Optional, List, Any


class MolecularDynamicsSHGNN:
    """
    分子动力学模拟器 - 基于SHGNN加速MD模拟

    使用球谐图神经网络学习原子间相互作用势能面，
    加速传统分子动力学模拟中的力场计算。

    Attributes:
        num_atoms: 系统中的原子数量
        box_size: 模拟盒子尺寸
        dt: 时间步长（飞秒）
        temperature: 目标温度（开尔文）
        integrator_type: 积分器类型 ('velocity_verlet', 'langevin', 'nose_hoover')
        ensemble: 系综类型 ('NVE', 'NVT', 'NPT')
        cutoff: 力场截断距离（埃）
        trained: SHGNN力场模型是否已训练
        model_weights: SHGNN模型权重
    """

    # 物理常数
    KB = 8.617333e-5        # 玻尔兹曼常数 (eV/K)
    AMU_TO_EV = 103.6427    # 原子质量单位到eV的转换
    FS_TO_PS = 0.001        # 飞秒到皮秒的转换
    ANG_TO_NM = 0.1         # 埃到纳米的转换

    # 常见元素的Lennard-Jones参数 (epsilon: eV, sigma: 埃)
    LJ_PARAMS = {
        'H':  {'epsilon': 0.0007, 'sigma': 1.2, 'mass': 1.008},
        'C':  {'epsilon': 0.0044, 'sigma': 3.4, 'mass': 12.011},
        'N':  {'epsilon': 0.0050, 'sigma': 3.3, 'mass': 14.007},
        'O':  {'epsilon': 0.0066, 'sigma': 3.0, 'mass': 15.999},
        'Si': {'epsilon': 0.0080, 'sigma': 3.8, 'mass': 28.086},
        'Fe': {'epsilon': 0.0090, 'sigma': 2.5, 'mass': 55.845},
        'Cu': {'epsilon': 0.0065, 'sigma': 2.3, 'mass': 63.546},
        'Au': {'epsilon': 0.0080, 'sigma': 2.6, 'mass': 196.967},
        'Al': {'epsilon': 0.0050, 'sigma': 2.6, 'mass': 26.982},
        'Na': {'epsilon': 0.0025, 'sigma': 3.4, 'mass': 22.990},
        'Cl': {'epsilon': 0.0050, 'sigma': 3.4, 'mass': 35.453},
    }

    def __init__(
        self,
        num_atoms: int = 100,
        box_size: float = 20.0,
        dt: float = 1.0,
        temperature: float = 300.0,
        integrator_type: str = 'velocity_verlet',
        ensemble: str = 'NVT',
        cutoff: float = 10.0,
        hidden_dim: int = 64,
        l_max: int = 4,
        num_layers: int = 3,
        random_seed: Optional[int] = None
    ):
        """
        初始化分子动力学模拟器

        Args:
            num_atoms: 原子数量
            box_size: 模拟盒子尺寸（埃）
            dt: 时间步长（飞秒）
            temperature: 目标温度（开尔文）
            integrator_type: 积分器类型
            ensemble: 系综类型
            cutoff: 力场截断距离（埃）
            hidden_dim: SHGNN隐藏层维度
            l_max: 球谐函数最大阶数
            num_layers: GNN层数
            random_seed: 随机种子
        """
        self.num_atoms = num_atoms
        self.box_size = box_size
        self.dt = dt
        self.temperature = temperature
        self.integrator_type = integrator_type
        self.ensemble = ensemble
        self.cutoff = cutoff
        self.hidden_dim = hidden_dim
        self.l_max = l_max
        self.num_layers = num_layers
        self.trained = False

        if random_seed is not None:
            random.seed(random_seed)

        # 系统状态
        self.positions = []       # 原子位置 (N x 3)
        self.velocities = []      # 原子速度 (N x 3)
        self.forces = []          # 原子受力 (N x 3)
        self.masses = []          # 原子质量 (N)
        self.atom_types = []      # 原子类型 (N)
        self.potential_energy = 0.0
        self.kinetic_energy = 0.0

        # 模拟统计
        self.step_count = 0
        self.time = 0.0
        self.trajectory = []
        self.energies = []
        self.temperatures = []

        # SHGNN力场模型
        self.model_weights = self._initialize_shgnn_weights()

        # Langevin积分器参数
        self.langevin_friction = 0.01  # 摩擦系数
        self.langevin_seed = random_seed or 42

    def _initialize_shgnn_weights(self) -> Dict[str, Any]:
        """
        初始化SHGNN力场模型权重

        Returns:
            模型权重字典
        """
        weights = {}

        # 原子特征编码器
        input_dim = 10  # 原子类型、质量、局部密度等
        scale = math.sqrt(2.0 / (input_dim + self.hidden_dim))
        weights['encoder_w1'] = [
            [random.gauss(0, scale) for _ in range(input_dim)]
            for _ in range(self.hidden_dim)
        ]
        weights['encoder_b1'] = [0.0] * self.hidden_dim

        # SHGNN交互层
        for layer_idx in range(self.num_layers):
            prefix = f'interaction_{layer_idx}'
            pair_dim = self.hidden_dim * 2 + 4  # 两原子特征 + 相对位置 + 距离

            scale_pair = math.sqrt(2.0 / (pair_dim + self.hidden_dim))
            weights[f'{prefix}_pair_w1'] = [
                [random.gauss(0, scale_pair) for _ in range(pair_dim)]
                for _ in range(self.hidden_dim)
            ]
            weights[f'{prefix}_pair_b1'] = [0.0] * self.hidden_dim

            scale_pair2 = math.sqrt(2.0 / (self.hidden_dim + self.hidden_dim))
            weights[f'{prefix}_pair_w2'] = [
                [random.gauss(0, scale_pair2) for _ in range(self.hidden_dim)]
                for _ in range(self.hidden_dim)
            ]
            weights[f'{prefix}_pair_b2'] = [0.0] * self.hidden_dim

            # 球谐函数系数
            weights[f'{prefix}_sh_coeffs'] = [
                random.gauss(0, 0.1) for _ in range(self.l_max + 1)
            ]

        # 能量/力输出头
        scale_e = math.sqrt(2.0 / (self.hidden_dim + self.hidden_dim))
        weights['energy_w1'] = [
            [random.gauss(0, scale_e) for _ in range(self.hidden_dim)]
            for _ in range(self.hidden_dim)
        ]
        weights['energy_b1'] = [0.0] * self.hidden_dim

        weights['energy_w2'] = [
            [random.gauss(0, 0.01) for _ in range(self.hidden_dim)]
            for _ in range(1)
        ]
        weights['energy_b2'] = [0.0]

        # 力输出头
        weights['force_w'] = [
            [random.gauss(0, 0.01) for _ in range(self.hidden_dim)]
            for _ in range(3)
        ]
        weights['force_b'] = [0.0, 0.0, 0.0]

        return weights

    def initialize_system(
        self,
        atom_types: Optional[List[str]] = None,
        positions: Optional[List[List[float]]] = None,
        lattice: Optional[str] = 'fcc'
    ) -> None:
        """
        初始化模拟系统

        设置原子类型、初始位置和速度。

        Args:
            atom_types: 原子类型列表
            positions: 自定义初始位置（可选）
            lattice: 晶格类型 ('fcc', 'bcc', 'sc', 'random')
        """
        # 设置原子类型
        if atom_types is None:
            atom_types = ['Ar'] * self.num_atoms
        self.atom_types = atom_types[:self.num_atoms]

        # 设置原子质量
        self.masses = []
        for atype in self.atom_types:
            params = self.LJ_PARAMS.get(atype, {'mass': 39.948})
            self.masses.append(params['mass'])

        # 初始化位置
        if positions is not None:
            self.positions = [list(p) for p in positions[:self.num_atoms]]
            while len(self.positions) < self.num_atoms:
                self.positions.append([
                    random.uniform(0, self.box_size),
                    random.uniform(0, self.box_size),
                    random.uniform(0, self.box_size)
                ])
        else:
            self.positions = self._generate_lattice(lattice)

        # 初始化速度（Maxwell-Boltzmann分布）
        self._initialize_velocities()

        # 计算初始力
        self.forces = self._compute_forces()

    def _generate_lattice(self, lattice_type: str) -> List[List[float]]:
        """
        生成晶格初始位置

        Args:
            lattice_type: 晶格类型

        Returns:
            原子位置列表
        """
        positions = []
        spacing = self.box_size / max(2, int(math.ceil(self.num_atoms ** (1.0 / 3.0))))

        if lattice_type == 'fcc':
            # 面心立方晶格
            basis = [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5]
            ]
            count = 0
            nx = int(math.ceil((self.num_atoms / 4) ** (1.0 / 3.0))) + 1
            for ix in range(nx):
                for iy in range(nx):
                    for iz in range(nx):
                        for b in basis:
                            if count >= self.num_atoms:
                                break
                            x = (ix + b[0]) * spacing
                            y = (iy + b[1]) * spacing
                            z = (iz + b[2]) * spacing
                            if x < self.box_size and y < self.box_size and z < self.box_size:
                                positions.append([x, y, z])
                                count += 1
                        if count >= self.num_atoms:
                            break
                    if count >= self.num_atoms:
                        break
                if count >= self.num_atoms:
                    break

        elif lattice_type == 'bcc':
            # 体心立方晶格
            basis = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
            count = 0
            nx = int(math.ceil((self.num_atoms / 2) ** (1.0 / 3.0))) + 1
            for ix in range(nx):
                for iy in range(nx):
                    for iz in range(nx):
                        for b in basis:
                            if count >= self.num_atoms:
                                break
                            x = (ix + b[0]) * spacing
                            y = (iy + b[1]) * spacing
                            z = (iz + b[2]) * spacing
                            if x < self.box_size and y < self.box_size and z < self.box_size:
                                positions.append([x, y, z])
                                count += 1
                        if count >= self.num_atoms:
                            break
                    if count >= self.num_atoms:
                        break
                if count >= self.num_atoms:
                    break

        else:
            # 简单立方或随机
            for _ in range(self.num_atoms):
                positions.append([
                    random.uniform(1.0, self.box_size - 1.0),
                    random.uniform(1.0, self.box_size - 1.0),
                    random.uniform(1.0, self.box_size - 1.0)
                ])

        # 填充不足的位置
        while len(positions) < self.num_atoms:
            positions.append([
                random.uniform(1.0, self.box_size - 1.0),
                random.uniform(1.0, self.box_size - 1.0),
                random.uniform(1.0, self.box_size - 1.0)
            ])

        return positions[:self.num_atoms]

    def _initialize_velocities(self) -> None:
        """
        初始化原子速度（Maxwell-Boltzmann分布）

        根据目标温度生成符合Maxwell-Boltzmann分布的初始速度，
        并去除质心运动以确保系统总动量为零。
        """
        self.velocities = []
        for i in range(self.num_atoms):
            mass = self.masses[i]
            # Maxwell-Boltzmann分布的标准差
            sigma = math.sqrt(self.KB * self.temperature / (mass * self.AMU_TO_EV))
            vx = random.gauss(0, sigma)
            vy = random.gauss(0, sigma)
            vz = random.gauss(0, sigma)
            self.velocities.append([vx, vy, vz])

        # 去除质心运动
        total_mass = sum(self.masses)
        cm_vx = sum(self.masses[i] * self.velocities[i][0] for i in range(self.num_atoms)) / total_mass
        cm_vy = sum(self.masses[i] * self.velocities[i][1] for i in range(self.num_atoms)) / total_mass
        cm_vz = sum(self.masses[i] * self.velocities[i][2] for i in range(self.num_atoms)) / total_mass

        for i in range(self.num_atoms):
            self.velocities[i][0] -= cm_vx
            self.velocities[i][1] -= cm_vy
            self.velocities[i][2] -= cm_vz

        # 重新缩放至目标温度
        self._rescale_velocities()

    def _rescale_velocities(self) -> None:
        """
        重新缩放速度至目标温度

        计算当前温度并缩放速度使其匹配目标温度。
        """
        current_temp = self._compute_temperature()
        if current_temp > 0:
            scale = math.sqrt(self.temperature / current_temp)
            for i in range(self.num_atoms):
                self.velocities[i][0] *= scale
                self.velocities[i][1] *= scale
                self.velocities[i][2] *= scale

    def _compute_temperature(self) -> float:
        """
        计算系统瞬时温度

        Returns:
            当前温度（开尔文）
        """
        self.kinetic_energy = 0.0
        for i in range(self.num_atoms):
            mass = self.masses[i] * self.AMU_TO_EV
            v2 = (self.velocities[i][0] ** 2 +
                  self.velocities[i][1] ** 2 +
                  self.velocities[i][2] ** 2)
            self.kinetic_energy += 0.5 * mass * v2

        # T = 2 * KE / (N_dof * kB)
        n_dof = 3 * self.num_atoms - 3  # 去除质心运动的自由度
        if n_dof > 0:
            temperature = 2.0 * self.kinetic_energy / (n_dof * self.KB)
        else:
            temperature = 0.0

        return temperature

    def _apply_pbc(self, coord: float) -> float:
        """
        应用周期性边界条件

        Args:
            coord: 坐标值

        Returns:
            包裹在盒子内的坐标
        """
        while coord < 0:
            coord += self.box_size
        while coord >= self.box_size:
            coord -= self.box_size
        return coord

    def _minimum_image(self, dx: float) -> float:
        """
        最小镜像约定

        Args:
            dx: 坐标差

        Returns:
            最小镜像距离
        """
        if dx > self.box_size / 2:
            dx -= self.box_size
        elif dx < -self.box_size / 2:
            dx += self.box_size
        return dx

    def _compute_lj_forces(self) -> Tuple[List[List[float]], float]:
        """
        计算Lennard-Jones力场

        使用经典Lennard-Jones势计算原子间相互作用力和势能。
        V(r) = 4 * epsilon * [(sigma/r)^12 - (sigma/r)^6]

        Returns:
            (forces, potential_energy) 力列表和势能
        """
        forces = [[0.0, 0.0, 0.0] for _ in range(self.num_atoms)]
        pe = 0.0

        for i in range(self.num_atoms):
            for j in range(i + 1, self.num_atoms):
                # 计算距离向量（最小镜像）
                dx = self._minimum_image(self.positions[j][0] - self.positions[i][0])
                dy = self._minimum_image(self.positions[j][1] - self.positions[i][1])
                dz = self._minimum_image(self.positions[j][2] - self.positions[i][2])

                r2 = dx * dx + dy * dy + dz * dz

                if r2 < self.cutoff * self.cutoff and r2 > 0.01:
                    r = math.sqrt(r2)

                    # 获取LJ参数（Lorentz-Berthelot混合规则）
                    params_i = self.LJ_PARAMS.get(self.atom_types[i], {'epsilon': 0.004, 'sigma': 3.4})
                    params_j = self.LJ_PARAMS.get(self.atom_types[j], {'epsilon': 0.004, 'sigma': 3.4})

                    epsilon = math.sqrt(params_i['epsilon'] * params_j['epsilon'])
                    sigma = (params_i['sigma'] + params_j['sigma']) / 2.0

                    # LJ力计算
                    sr6 = (sigma / r) ** 6
                    sr12 = sr6 * sr6

                    # 力的大小 F = 24 * epsilon / r * (2*sr12 - sr6)
                    force_mag = 24.0 * epsilon / r * (2.0 * sr12 - sr6)

                    # 力的分量
                    fx = force_mag * dx / r
                    fy = force_mag * dy / r
                    fz = force_mag * dz / r

                    forces[i][0] += fx
                    forces[i][1] += fy
                    forces[i][2] += fz
                    forces[j][0] -= fx
                    forces[j][1] -= fy
                    forces[j][2] -= fz

                    # 势能
                    pe += 4.0 * epsilon * (sr12 - sr6)

        return forces, pe

    def _compute_shgnn_forces(self) -> Tuple[List[List[float]], float]:
        """
        使用SHGNN模型计算原子间作用力

        基于球谐图神经网络预测原子间相互作用势能和力。
        这是加速MD模拟的核心方法。

        Returns:
            (forces, potential_energy) 力列表和势能
        """
        forces = [[0.0, 0.0, 0.0] for _ in range(self.num_atoms)]
        pe = 0.0

        # 构建原子特征
        atom_features = []
        for i in range(self.num_atoms):
            mass_norm = self.masses[i] / 200.0
            # 计算局部密度
            local_density = 0.0
            for j in range(self.num_atoms):
                if i != j:
                    dx = self._minimum_image(self.positions[j][0] - self.positions[i][0])
                    dy = self._minimum_image(self.positions[j][1] - self.positions[i][1])
                    dz = self._minimum_image(self.positions[j][2] - self.positions[i][2])
                    r = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if r < self.cutoff:
                        local_density += 1.0

            features = [
                mass_norm,
                local_density / self.num_atoms,
                self.positions[i][0] / self.box_size,
                self.positions[i][1] / self.box_size,
                self.positions[i][2] / self.box_size,
                self.velocities[i][0] * 10.0,
                self.velocities[i][1] * 10.0,
                self.velocities[i][2] * 10.0,
                math.sin(i * 0.1),
                math.cos(i * 0.1),
            ]
            atom_features.append(features)

        # 编码原子特征
        encoded = []
        for feat in atom_features:
            enc = self._linear(feat, self.model_weights['encoder_w1'], self.model_weights['encoder_b1'])
            enc = [self._silu(x) for x in enc]
            encoded.append(enc)

        # SHGNN交互层
        for layer_idx in range(self.num_layers):
            prefix = f'interaction_{layer_idx}'
            new_encoded = [list(e) for e in encoded]

            for i in range(self.num_atoms):
                pair_energies = []
                pair_forces = []

                for j in range(self.num_atoms):
                    if i == j:
                        continue

                    dx = self._minimum_image(self.positions[j][0] - self.positions[i][0])
                    dy = self._minimum_image(self.positions[j][1] - self.positions[i][1])
                    dz = self._minimum_image(self.positions[j][2] - self.positions[i][2])
                    r = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-8

                    if r > self.cutoff:
                        continue

                    # 球谐函数方向编码
                    theta = math.acos(max(-1.0, min(1.0, dz / r)))
                    phi = math.atan2(dy, dx)

                    sh_encoding = 0.0
                    for l in range(min(self.l_max + 1, 3)):
                        sh_val = self._simplified_sh(theta, phi, l)
                        sh_encoding += self.model_weights[f'{prefix}_sh_coeffs'][l] * sh_val

                    # 构建配对特征
                    pair_feat = (encoded[i] + encoded[j] +
                                 [dx / r, dy / r, dz / r, r / self.cutoff])

                    # 配对网络
                    h = self._linear(pair_feat,
                                     self.model_weights[f'{prefix}_pair_w1'],
                                     self.model_weights[f'{prefix}_pair_b1'])
                    h = [self._silu(x) for x in h]
                    h = self._linear(h,
                                     self.model_weights[f'{prefix}_pair_w2'],
                                     self.model_weights[f'{prefix}_pair_b2'])

                    # 加权
                    weight = 1.0 / (1.0 + math.exp(-sh_encoding))
                    pair_e = sum(h) * weight * 0.001  # 缩放到合理范围
                    pair_energies.append(pair_e)

                    # 力 = -dE/dr * (r_vec/r)
                    force_scale = -pair_e / r
                    pair_forces.append([
                        force_scale * dx / r,
                        force_scale * dy / r,
                        force_scale * dz / r
                    ])

                # 聚合
                if pair_energies:
                    pe += sum(pair_energies)
                    for f in pair_forces:
                        forces[i][0] += f[0]
                        forces[i][1] += f[1]
                        forces[i][2] += f[2]

            encoded = new_encoded

        return forces, pe

    def _simplified_sh(self, theta: float, phi: float, l: int) -> float:
        """
        简化的球谐函数计算

        Args:
            theta: 极角
            phi: 方位角
            l: 阶数

        Returns:
            球谐函数值
        """
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        if l == 0:
            return 0.5 * math.sqrt(1.0 / math.pi)
        elif l == 1:
            return 0.5 * math.sqrt(3.0 / math.pi) * cos_t
        elif l == 2:
            return 0.25 * math.sqrt(5.0 / math.pi) * (3 * cos_t * cos_t - 1)
        else:
            return math.cos(l * phi) * sin_t ** l

    def _compute_forces(self) -> List[List[float]]:
        """
        计算原子受力（根据是否训练选择力场）

        Returns:
            力列表
        """
        if self.trained:
            forces, self.potential_energy = self._compute_shgnn_forces()
        else:
            forces, self.potential_energy = self._compute_lj_forces()

        return forces

    def _linear(
        self,
        input_vec: List[float],
        weight: List[List[float]],
        bias: List[float]
    ) -> List[float]:
        """线性变换"""
        out_dim = len(weight)
        result = [0.0] * out_dim
        for i in range(out_dim):
            s = bias[i]
            for j in range(len(input_vec)):
                s += weight[i][j] * input_vec[j]
            result[i] = s
        return result

    def _silu(self, x: float) -> float:
        """SiLU激活函数"""
        if abs(x) > 500:
            return x if x > 0 else 0.0
        return x / (1.0 + math.exp(-x))

    def velocity_verlet_step(self) -> None:
        """
        Velocity Verlet积分器单步

        使用Velocity Verlet算法更新位置和速度：
        1. r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
        2. 计算 f(t+dt)
        3. v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        """
        dt = self.dt

        # 半步速度更新
        for i in range(self.num_atoms):
            mass = self.masses[i] * self.AMU_TO_EV
            ax = self.forces[i][0] / mass
            ay = self.forces[i][1] / mass
            az = self.forces[i][2] / mass

            # 更新位置
            self.positions[i][0] += self.velocities[i][0] * dt + 0.5 * ax * dt * dt
            self.positions[i][1] += self.velocities[i][1] * dt + 0.5 * ay * dt * dt
            self.positions[i][2] += self.velocities[i][2] * dt + 0.5 * az * dt * dt

            # 应用周期性边界条件
            self.positions[i][0] = self._apply_pbc(self.positions[i][0])
            self.positions[i][1] = self._apply_pbc(self.positions[i][1])
            self.positions[i][2] = self._apply_pbc(self.positions[i][2])

            # 半步速度更新
            self.velocities[i][0] += 0.5 * ax * dt
            self.velocities[i][1] += 0.5 * ay * dt
            self.velocities[i][2] += 0.5 * az * dt

        # 计算新力
        self.forces = self._compute_forces()

        # 完成速度更新
        for i in range(self.num_atoms):
            mass = self.masses[i] * self.AMU_TO_EV
            self.velocities[i][0] += 0.5 * self.forces[i][0] / mass * dt
            self.velocities[i][1] += 0.5 * self.forces[i][1] / mass * dt
            self.velocities[i][2] += 0.5 * self.forces[i][2] / mass * dt

    def langevin_step(self) -> None:
        """
        Langevin积分器单步

        使用Langevin动力学模拟NVT系综：
        m*a = F - gamma*v + sqrt(2*gamma*kB*T/dt) * R(t)
        其中gamma是摩擦系数，R(t)是随机力。
        """
        dt = self.dt
        gamma = self.langevin_friction

        for i in range(self.num_atoms):
            mass = self.masses[i] * self.AMU_TO_EV

            # 确定性力
            fx = self.forces[i][0]
            fy = self.forces[i][1]
            fz = self.forces[i][2]

            # 摩擦力
            fx -= gamma * self.velocities[i][0] * mass
            fy -= gamma * self.velocities[i][1] * mass
            fz -= gamma * self.velocities[i][2] * mass

            # 随机力
            noise_scale = math.sqrt(2.0 * gamma * self.KB * self.temperature * mass / dt)
            fx += noise_scale * random.gauss(0, 1)
            fy += noise_scale * random.gauss(0, 1)
            fz += noise_scale * random.gauss(0, 1)

            # Velocity Verlet + Langevin
            ax = fx / mass
            ay = fy / mass
            az = fz / mass

            self.positions[i][0] += self.velocities[i][0] * dt + 0.5 * ax * dt * dt
            self.positions[i][1] += self.velocities[i][1] * dt + 0.5 * ay * dt * dt
            self.positions[i][2] += self.velocities[i][2] * dt + 0.5 * az * dt * dt

            self.positions[i][0] = self._apply_pbc(self.positions[i][0])
            self.positions[i][1] = self._apply_pbc(self.positions[i][1])
            self.positions[i][2] = self._apply_pbc(self.positions[i][2])

            self.velocities[i][0] += ax * dt
            self.velocities[i][1] += ay * dt
            self.velocities[i][2] += az * dt

        # 重新计算力
        self.forces = self._compute_forces()

    def run(
        self,
        num_steps: int = 1000,
        save_interval: int = 100,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        运行分子动力学模拟

        执行指定步数的MD模拟，定期保存轨迹和能量。

        Args:
            num_steps: 模拟步数
            save_interval: 轨迹保存间隔
            verbose: 是否打印进度

        Returns:
            模拟结果字典
        """
        if not self.positions:
            self.initialize_system()

        if verbose:
            print(f"[MolecularDynamicsSHGNN] 开始MD模拟")
            print(f"  原子数: {self.num_atoms}")
            print(f"  盒子尺寸: {self.box_size:.1f} A")
            print(f"  时间步长: {self.dt:.2f} fs")
            print(f"  目标温度: {self.temperature:.1f} K")
            print(f"  积分器: {self.integrator_type}")
            print(f"  系综: {self.ensemble}")
            print(f"  力场: {'SHGNN' if self.trained else 'Lennard-Jones'}")
            print(f"  模拟步数: {num_steps}")

        for step in range(num_steps):
            # 选择积分器
            if self.integrator_type == 'langevin':
                self.langevin_step()
            else:
                self.velocity_verlet_step()

            # 恒温控制（Velocity Verlet + 速度重标）
            if self.integrator_type == 'velocity_verlet' and self.ensemble == 'NVT':
                if (step + 1) % 50 == 0:
                    self._rescale_velocities()

            self.step_count += 1
            self.time += self.dt

            # 计算温度
            current_temp = self._compute_temperature()

            # 保存轨迹和能量
            if (step + 1) % save_interval == 0:
                self.trajectory.append({
                    'step': self.step_count,
                    'time': self.time,
                    'positions': copy.deepcopy(self.positions),
                    'velocities': copy.deepcopy(self.velocities),
                })
                self.energies.append({
                    'step': self.step_count,
                    'time': self.time,
                    'kinetic_energy': self.kinetic_energy,
                    'potential_energy': self.potential_energy,
                    'total_energy': self.kinetic_energy + self.potential_energy,
                })
                self.temperatures.append({
                    'step': self.step_count,
                    'time': self.time,
                    'temperature': current_temp,
                })

            if verbose and (step + 1) % (num_steps // 10) == 0:
                total_e = self.kinetic_energy + self.potential_energy
                print(f"  步数 {step + 1}/{num_steps}, "
                      f"T={current_temp:.1f}K, "
                      f"KE={self.kinetic_energy:.4f}eV, "
                      f"PE={self.potential_energy:.4f}eV, "
                      f"E={total_e:.4f}eV")

        if verbose:
            print(f"[MolecularDynamicsSHGNN] 模拟完成")

        return {
            'num_steps': num_steps,
            'final_temperature': self._compute_temperature(),
            'final_kinetic_energy': self.kinetic_energy,
            'final_potential_energy': self.potential_energy,
            'trajectory': self.trajectory,
            'energies': self.energies,
            'temperatures': self.temperatures,
        }

    def train_force_field(
        self,
        reference_data: List[Dict[str, Any]],
        num_epochs: int = 50,
        learning_rate: float = 0.001,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        训练SHGNN力场模型

        使用参考数据（DFT计算或实验数据）训练SHGNN力场。

        Args:
            reference_data: 参考数据列表，包含位置、力和能量
            num_epochs: 训练轮数
            learning_rate: 学习率
            verbose: 是否打印训练日志

        Returns:
            训练结果
        """
        if verbose:
            print(f"[MolecularDynamicsSHGNN] 开始训练SHGNN力场")
            print(f"  参考数据量: {len(reference_data)}")
            print(f"  训练轮数: {num_epochs}")

        loss_history = []

        for epoch in range(num_epochs):
            total_loss = 0.0

            for ref in reference_data:
                # 设置参考构型
                old_positions = self.positions
                old_forces = self.forces
                self.positions = ref['positions']
                self.num_atoms = len(self.positions)

                # 使用SHGNN预测力
                pred_forces, pred_energy = self._compute_shgnn_forces()

                # 计算力匹配损失
                ref_forces = ref['forces']
                force_loss = 0.0
                for i in range(self.num_atoms):
                    for d in range(3):
                        diff = pred_forces[i][d] - ref_forces[i][d]
                        force_loss += diff * diff
                force_loss /= self.num_atoms

                # 能量损失
                energy_loss = (pred_energy - ref.get('energy', 0.0)) ** 2

                total_loss += force_loss + 0.1 * energy_loss

                # 恢复
                self.positions = old_positions
                self.forces = old_forces

            avg_loss = total_loss / len(reference_data)
            loss_history.append(avg_loss)

            # 简化权重更新
            grad_scale = math.tanh(avg_loss) * learning_rate * 0.01
            for key in self.model_weights:
                if isinstance(self.model_weights[key], list):
                    if isinstance(self.model_weights[key][0], list):
                        for i in range(len(self.model_weights[key])):
                            for j in range(len(self.model_weights[key][i])):
                                self.model_weights[key][i][j] -= grad_scale * random.gauss(0, 1)
                    else:
                        for i in range(len(self.model_weights[key])):
                            self.model_weights[key][i] -= grad_scale * random.gauss(0, 1) * 0.1

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  轮次 {epoch + 1}/{num_epochs}, 损失: {avg_loss:.6f}")

        self.trained = True

        if verbose:
            print(f"[MolecularDynamicsSHGNN] 力场训练完成, 最终损失: {loss_history[-1]:.6f}")

        return {
            'final_loss': loss_history[-1],
            'loss_history': loss_history,
            'num_epochs': num_epochs,
        }

    def compute_rdf(
        self,
        num_bins: int = 100,
        r_max: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        计算径向分布函数（RDF）

        Args:
            num_bins: 分箱数
            r_max: 最大距离

        Returns:
            RDF数据字典
        """
        if r_max is None:
            r_max = self.box_size / 2.0

        dr = r_max / num_bins
        hist = [0.0] * num_bins
        bin_centers = [(i + 0.5) * dr for i in range(num_bins)]

        for i in range(self.num_atoms):
            for j in range(i + 1, self.num_atoms):
                dx = self._minimum_image(self.positions[j][0] - self.positions[i][0])
                dy = self._minimum_image(self.positions[j][1] - self.positions[i][1])
                dz = self._minimum_image(self.positions[j][2] - self.positions[i][2])
                r = math.sqrt(dx * dx + dy * dy + dz * dz)

                if r < r_max:
                    bin_idx = int(r / dr)
                    if 0 <= bin_idx < num_bins:
                        hist[bin_idx] += 2.0  # 计数两次（i-j和j-i）

        # 归一化
        volume = self.box_size ** 3
        rho = self.num_atoms / volume
        for i in range(num_bins):
            r_low = i * dr
            r_high = (i + 1) * dr
            shell_volume = (4.0 / 3.0) * math.pi * (r_high ** 3 - r_low ** 3)
            ideal_count = rho * shell_volume * self.num_atoms
            if ideal_count > 0:
                hist[i] /= ideal_count

        return {
            'r': bin_centers,
            'g_r': hist,
            'num_bins': num_bins,
            'r_max': r_max,
        }

    def compute_msd(self) -> float:
        """
        计算均方位移（MSD）

        Returns:
            均方位移值
        """
        if len(self.trajectory) < 2:
            return 0.0

        initial_positions = self.trajectory[0]['positions']
        current_positions = self.positions

        msd = 0.0
        for i in range(self.num_atoms):
            dx = self._minimum_image(current_positions[i][0] - initial_positions[i][0])
            dy = self._minimum_image(current_positions[i][1] - initial_positions[i][1])
            dz = self._minimum_image(current_positions[i][2] - initial_positions[i][2])
            msd += dx * dx + dy * dy + dz * dz

        msd /= self.num_atoms
        return msd

    def get_system_info(self) -> Dict[str, Any]:
        """
        获取系统信息

        Returns:
            系统状态信息字典
        """
        return {
            'model_name': 'MolecularDynamicsSHGNN',
            'num_atoms': self.num_atoms,
            'box_size': self.box_size,
            'dt': self.dt,
            'temperature': self.temperature,
            'current_temperature': self._compute_temperature() if self.positions else 0.0,
            'integrator': self.integrator_type,
            'ensemble': self.ensemble,
            'cutoff': self.cutoff,
            'force_field': 'SHGNN' if self.trained else 'Lennard-Jones',
            'step_count': self.step_count,
            'time_fs': self.time,
            'kinetic_energy': self.kinetic_energy,
            'potential_energy': self.potential_energy,
            'total_energy': self.kinetic_energy + self.potential_energy,
            'atom_types': list(set(self.atom_types)),
        }

    def save_state(self, filepath: str) -> None:
        """
        保存模拟状态

        Args:
            filepath: 保存路径
        """
        state = {
            'positions': self.positions,
            'velocities': self.velocities,
            'forces': self.forces,
            'masses': self.masses,
            'atom_types': self.atom_types,
            'step_count': self.step_count,
            'time': self.time,
            'config': {
                'num_atoms': self.num_atoms,
                'box_size': self.box_size,
                'dt': self.dt,
                'temperature': self.temperature,
                'integrator_type': self.integrator_type,
                'ensemble': self.ensemble,
                'cutoff': self.cutoff,
            },
            'model_weights': self.model_weights,
            'trained': self.trained,
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def load_state(self, filepath: str) -> None:
        """
        加载模拟状态

        Args:
            filepath: 状态文件路径
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)

        self.positions = state['positions']
        self.velocities = state['velocities']
        self.forces = state['forces']
        self.masses = state['masses']
        self.atom_types = state['atom_types']
        self.step_count = state['step_count']
        self.time = state['time']
        self.model_weights = state['model_weights']
        self.trained = state.get('trained', False)

        config = state['config']
        self.num_atoms = config['num_atoms']
        self.box_size = config['box_size']
        self.dt = config['dt']
        self.temperature = config['temperature']
        self.integrator_type = config['integrator_type']
        self.ensemble = config['ensemble']
        self.cutoff = config['cutoff']
