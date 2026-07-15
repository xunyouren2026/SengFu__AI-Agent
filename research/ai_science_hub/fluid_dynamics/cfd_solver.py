"""
AI科学中心 - 流体动力学模块

使用SH-GNN加速计算流体动力学（CFD）求解。
支持不可压缩/可压缩Navier-Stokes方程求解、
网格生成、湍流建模和后处理分析。
"""

import math
import json
import random
import copy
from typing import Dict, Tuple, Optional, List, Any


class CFDSolverSHGNN:
    """
    CFD求解器 - 基于SHGNN加速流体动力学求解

    使用球谐图神经网络加速Navier-Stokes方程的求解过程，
    通过学习流场特征来加速收敛并提供湍流建模能力。

    Attributes:
        nx: x方向网格数
        ny: y方向网格数
        nz: z方向网格数
        domain_size: 计算域尺寸 [Lx, Ly, Lz]
        reynolds_number: 雷诺数
        fluid_viscosity: 流体动力粘度
        fluid_density: 流体密度
        inlet_velocity: 入口速度
        solver_type: 求解器类型 ('steady', 'transient')
        turbulence_model: 湍流模型 ('none', 'smagorinsky', 'shgnn')
        trained: SHGNN湍流模型是否已训练
    """

    def __init__(
        self,
        nx: int = 32,
        ny: int = 32,
        nz: int = 32,
        domain_size: Optional[List[float]] = None,
        reynolds_number: float = 100.0,
        fluid_viscosity: float = 0.001,
        fluid_density: float = 1.0,
        inlet_velocity: float = 1.0,
        solver_type: str = 'steady',
        turbulence_model: str = 'none',
        hidden_dim: int = 64,
        l_max: int = 4,
        num_layers: int = 3,
        random_seed: Optional[int] = None
    ):
        """
        初始化CFD求解器

        Args:
            nx: x方向网格数
            ny: y方向网格数
            nz: z方向网格数
            domain_size: 计算域尺寸
            reynolds_number: 雷诺数
            fluid_viscosity: 流体动力粘度 (Pa·s)
            fluid_density: 流体密度 (kg/m³)
            inlet_velocity: 入口速度 (m/s)
            solver_type: 求解器类型
            turbulence_model: 湍流模型
            hidden_dim: SHGNN隐藏层维度
            l_max: 球谐函数最大阶数
            num_layers: GNN层数
            random_seed: 随机种子
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.domain_size = domain_size or [1.0, 1.0, 1.0]
        self.reynolds_number = reynolds_number
        self.fluid_viscosity = fluid_viscosity
        self.fluid_density = fluid_density
        self.inlet_velocity = inlet_velocity
        self.solver_type = solver_type
        self.turbulence_model = turbulence_model
        self.hidden_dim = hidden_dim
        self.l_max = l_max
        self.num_layers = num_layers
        self.trained = False

        if random_seed is not None:
            random.seed(random_seed)

        # 网格参数
        self.dx = self.domain_size[0] / (self.nx - 1)
        self.dy = self.domain_size[1] / (self.ny - 1)
        self.dz = self.domain_size[2] / (self.nz - 1) if self.nz > 1 else 0.0

        # 流场变量（速度和压力）
        self.u = [[[0.0] * self.nz for _ in range(self.ny)] for _ in range(self.nx)]  # x速度
        self.v = [[[0.0] * self.nz for _ in range(self.ny)] for _ in range(self.nx)]  # y速度
        self.w = [[[0.0] * self.nz for _ in range(self.ny)] for _ in range(self.nx)]  # z速度
        self.p = [[[0.0] * self.nz for _ in range(self.ny)] for _ in range(self.nx)]  # 压力

        # 收敛历史
        self.residual_history = []
        self.iteration_count = 0

        # SHGNN模型权重
        self.model_weights = self._initialize_shgnn_weights()

        # 边界条件
        self.boundary_conditions = {
            'inlet': 'velocity',     # 入口：速度边界
            'outlet': 'pressure',    # 出口：压力边界
            'walls': 'no_slip',      # 壁面：无滑移
            'top_bottom': 'no_slip', # 上下：无滑移
        }

    def _initialize_shgnn_weights(self) -> Dict[str, Any]:
        """
        初始化SHGNN湍流模型权重

        Returns:
            模型权重字典
        """
        weights = {}

        # 流场特征编码器（输入：速度、压力梯度、涡量等）
        input_dim = 12  # u, v, w, p, du/dx, dv/dy, dw/dz, vorticity_x, vorticity_y, vorticity_z, strain_rate, q_criterion
        scale = math.sqrt(2.0 / (input_dim + self.hidden_dim))
        weights['encoder_w'] = [
            [random.gauss(0, scale) for _ in range(input_dim)]
            for _ in range(self.hidden_dim)
        ]
        weights['encoder_b'] = [0.0] * self.hidden_dim

        # SHGNN空间传播层
        for layer_idx in range(self.num_layers):
            prefix = f'spatial_{layer_idx}'
            neighbor_dim = self.hidden_dim * 6 + 3  # 6邻居特征 + 相对位置

            scale_n = math.sqrt(2.0 / (neighbor_dim + self.hidden_dim))
            weights[f'{prefix}_w1'] = [
                [random.gauss(0, scale_n) for _ in range(neighbor_dim)]
                for _ in range(self.hidden_dim)
            ]
            weights[f'{prefix}_b1'] = [0.0] * self.hidden_dim

            scale_n2 = math.sqrt(2.0 / (self.hidden_dim + self.hidden_dim))
            weights[f'{prefix}_w2'] = [
                [random.gauss(0, scale_n2) for _ in range(self.hidden_dim)]
                for _ in range(self.hidden_dim)
            ]
            weights[f'{prefix}_b2'] = [0.0] * self.hidden_dim

            # 球谐函数方向编码系数
            weights[f'{prefix}_sh_coeffs'] = [
                random.gauss(0, 0.1) for _ in range(self.l_max + 1)
            ]

        # 湍流粘度输出头
        weights['turb_visc_w'] = [
            [random.gauss(0, 0.01) for _ in range(self.hidden_dim)]
            for _ in range(1)
        ]
        weights['turb_visc_b'] = [0.0]

        # 速度修正输出头
        weights['vel_corr_w'] = [
            [random.gauss(0, 0.01) for _ in range(self.hidden_dim)]
            for _ in range(3)
        ]
        weights['vel_corr_b'] = [0.0, 0.0, 0.0]

        # 压力修正输出头
        weights['pres_corr_w'] = [
            [random.gauss(0, 0.01) for _ in range(self.hidden_dim)]
            for _ in range(1)
        ]
        weights['pres_corr_b'] = [0.0]

        return weights

    def generate_mesh(
        self,
        mesh_type: str = 'uniform',
        refinement_regions: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        生成计算网格

        Args:
            mesh_type: 网格类型 ('uniform', 'stretched', 'adaptive')
            refinement_regions: 网格加密区域

        Returns:
            网格信息字典
        """
        mesh_info = {
            'type': mesh_type,
            'nx': self.nx,
            'ny': self.ny,
            'nz': self.nz,
            'domain_size': self.domain_size,
            'dx': self.dx,
            'dy': self.dy,
            'dz': self.dz,
            'total_cells': self.nx * self.ny * max(self.nz, 1),
            'vertices': [],
            'cell_centers': [],
        }

        # 生成网格节点坐标
        if mesh_type == 'uniform':
            self._x_coords = [i * self.dx for i in range(self.nx)]
            self._y_coords = [j * self.dy for j in range(self.ny)]
            self._z_coords = [k * self.dz for k in range(self.nz)] if self.nz > 1 else [0.0]

        elif mesh_type == 'stretched':
            # 双曲正切拉伸网格
            stretch_factor = 1.5
            self._x_coords = self._stretched_coords(self.nx, self.domain_size[0], stretch_factor)
            self._y_coords = self._stretched_coords(self.ny, self.domain_size[1], stretch_factor)
            self._z_coords = self._stretched_coords(self.nz, self.domain_size[2], stretch_factor) if self.nz > 1 else [0.0]

        else:
            self._x_coords = [i * self.dx for i in range(self.nx)]
            self._y_coords = [j * self.dy for j in range(self.ny)]
            self._z_coords = [k * self.dz for k in range(self.nz)] if self.nz > 1 else [0.0]

        # 计算网格质量指标
        mesh_info['quality_metrics'] = self._compute_mesh_quality()

        return mesh_info

    def _stretched_coords(self, n: int, length: float, stretch: float) -> List[float]:
        """
        生成拉伸网格坐标

        Args:
            n: 网格点数
            length: 域长度
            stretch: 拉伸因子

        Returns:
            坐标列表
        """
        coords = []
        for i in range(n):
            s = i / (n - 1) if n > 1 else 0.0
            # 双曲正切拉伸
            eta = 0.5 * (1.0 + math.tanh(stretch * (2.0 * s - 1.0)) / math.tanh(stretch))
            coords.append(eta * length)
        return coords

    def _compute_mesh_quality(self) -> Dict[str, float]:
        """
        计算网格质量指标

        Returns:
            质量指标字典
        """
        if not hasattr(self, '_x_coords'):
            self._x_coords = [i * self.dx for i in range(self.nx)]
            self._y_coords = [j * self.dy for j in range(self.ny)]
            self._z_coords = [k * self.dz for k in range(self.nz)] if self.nz > 1 else [0.0]

        # 计算纵横比
        max_aspect = 1.0
        min_aspect = 1.0

        if len(self._x_coords) > 1:
            dx_min = min(self._x_coords[i+1] - self._x_coords[i] for i in range(len(self._x_coords)-1))
            dx_max = max(self._x_coords[i+1] - self._x_coords[i] for i in range(len(self._x_coords)-1))
            dy_min = min(self._y_coords[i+1] - self._y_coords[i] for i in range(len(self._y_coords)-1))
            dy_max = max(self._y_coords[i+1] - self._y_coords[i] for i in range(len(self._y_coords)-1))

            max_aspect = max(dx_max / (dy_min + 1e-10), dy_max / (dx_min + 1e-10))
            min_aspect = min(dx_min / (dy_max + 1e-10), dy_min / (dx_max + 1e-10))

        return {
            'max_aspect_ratio': round(max_aspect, 4),
            'min_aspect_ratio': round(min_aspect, 4),
            'total_cells': self.nx * self.ny * max(self.nz, 1),
            'orthogonality': 1.0,  # 结构化网格正交性为1
            'skewness': 0.0,        # 结构化网格偏斜度为0
        }

    def set_boundary_conditions(
        self,
        bc_type: str,
        inlet_velocity: Optional[float] = None,
        outlet_pressure: float = 0.0,
        wall_velocity: float = 0.0
    ) -> None:
        """
        设置边界条件

        Args:
            bc_type: 边界条件类型
            inlet_velocity: 入口速度
            outlet_pressure: 出口压力
            wall_velocity: 壁面速度
        """
        if inlet_velocity is not None:
            self.inlet_velocity = inlet_velocity

        self._outlet_pressure = outlet_pressure
        self._wall_velocity = wall_velocity

        self.boundary_conditions = {
            'inlet': bc_type if bc_type != 'custom' else 'velocity',
            'outlet': 'pressure',
            'walls': 'no_slip' if wall_velocity == 0.0 else 'moving_wall',
        }

    def _apply_boundary_conditions(self) -> None:
        """
        应用边界条件到流场变量
        """
        # 入口边界（x=0）：设定速度
        for j in range(self.ny):
            for k in range(self.nz):
                self.u[0][j][k] = self.inlet_velocity
                self.v[0][j][k] = 0.0
                self.w[0][j][k] = 0.0

        # 出口边界（x=nx-1）：零梯度
        for j in range(self.ny):
            for k in range(self.nz):
                self.u[self.nx-1][j][k] = self.u[self.nx-2][j][k]
                self.v[self.nx-1][j][k] = self.v[self.nx-2][j][k]
                self.w[self.nx-1][j][k] = self.w[self.nx-2][j][k]
                self.p[self.nx-1][j][k] = self._outlet_pressure

        # 壁面边界（y=0, y=ny-1）：无滑移
        for i in range(self.nx):
            for k in range(self.nz):
                self.u[i][0][k] = self._wall_velocity
                self.v[i][0][k] = 0.0
                self.w[i][0][k] = 0.0
                self.u[i][self.ny-1][k] = self._wall_velocity
                self.v[i][self.ny-1][k] = 0.0
                self.w[i][self.ny-1][k] = 0.0

        # 上下边界（z=0, z=nz-1）
        if self.nz > 1:
            for i in range(self.nx):
                for j in range(self.ny):
                    self.u[i][j][0] = 0.0
                    self.v[i][j][0] = 0.0
                    self.w[i][j][0] = 0.0
                    self.u[i][j][self.nz-1] = 0.0
                    self.v[i][j][self.nz-1] = 0.0
                    self.w[i][j][self.nz-1] = 0.0

    def _compute_velocity_gradients(
        self,
        i: int,
        j: int,
        k: int
    ) -> Dict[str, float]:
        """
        计算速度梯度张量分量

        Args:
            i, j, k: 网格索引

        Returns:
            梯度分量字典
        """
        # 中心差分
        if 0 < i < self.nx - 1:
            dudx = (self.u[i+1][j][k] - self.u[i-1][j][k]) / (2 * self.dx)
            dvdx = (self.v[i+1][j][k] - self.v[i-1][j][k]) / (2 * self.dx)
            dwdx = (self.w[i+1][j][k] - self.w[i-1][j][k]) / (2 * self.dx)
        else:
            dudx = dvdx = dwdx = 0.0

        if 0 < j < self.ny - 1:
            dudy = (self.u[i][j+1][k] - self.u[i][j-1][k]) / (2 * self.dy)
            dvdy = (self.v[i][j+1][k] - self.v[i][j-1][k]) / (2 * self.dy)
            dwdy = (self.w[i][j+1][k] - self.w[i][j-1][k]) / (2 * self.dy)
        else:
            dudy = dvdy = dwdy = 0.0

        if self.nz > 1 and 0 < k < self.nz - 1:
            dudz = (self.u[i][j][k+1] - self.u[i][j][k-1]) / (2 * self.dz)
            dvdz = (self.v[i][j][k+1] - self.v[i][j][k-1]) / (2 * self.dz)
            dwdz = (self.w[i][j][k+1] - self.w[i][j][k-1]) / (2 * self.dz)
        else:
            dudz = dvdz = dwdz = 0.0

        # 涡量分量
        vorticity_x = dwdy - dvdz
        vorticity_y = dudz - dwdx
        vorticity_z = dvdx - dudy

        # 应变率张量模
        strain_rate = math.sqrt(
            2.0 * (dudx**2 + dvdy**2 + dwdz**2) +
            (dudy + dvdx)**2 + (dudz + dwdx)**2 + (dvdz + dwdy)**2
        )

        # Q准则（涡量模² - 应变率模²）
        q_criterion = 0.5 * (vorticity_x**2 + vorticity_y**2 + vorticity_z**2 - strain_rate**2)

        return {
            'dudx': dudx, 'dudy': dudy, 'dudz': dudz,
            'dvdx': dvdx, 'dvdy': dvdy, 'dvdz': dvdz,
            'dwdx': dwdx, 'dwdy': dwdy, 'dwdz': dwdz,
            'vorticity_x': vorticity_x,
            'vorticity_y': vorticity_y,
            'vorticity_z': vorticity_z,
            'strain_rate': strain_rate,
            'q_criterion': q_criterion,
        }

    def _compute_shgnn_correction(
        self,
        i: int,
        j: int,
        k: int,
        gradients: Dict[str, float]
    ) -> Tuple[float, float, float, float]:
        """
        使用SHGNN模型计算流场修正

        基于局部流场特征预测湍流粘度和速度/压力修正量。

        Args:
            i, j, k: 网格索引
            gradients: 速度梯度

        Returns:
            (du, dv, dw, dp) 速度和压力修正量
        """
        # 构建输入特征
        features = [
            self.u[i][j][k],
            self.v[i][j][k],
            self.w[i][j][k],
            self.p[i][j][k],
            gradients['dudx'],
            gradients['dvdy'],
            gradients['dwdz'],
            gradients['vorticity_x'],
            gradients['vorticity_y'],
            gradients['vorticity_z'],
            gradients['strain_rate'],
            gradients['q_criterion'],
        ]

        # 编码
        h = self._linear(features, self.model_weights['encoder_w'], self.model_weights['encoder_b'])
        h = [self._silu(x) for x in h]

        # 空间传播层
        for layer_idx in range(self.num_layers):
            prefix = f'spatial_{layer_idx}'
            neighbor_features = []

            # 收集邻居特征
            for di, dj, dk in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
                ni, nj, nk = i + di, j + dj, k + dk
                if 0 <= ni < self.nx and 0 <= nj < self.ny and 0 <= nk < self.nz:
                    neighbor_features.append(list(h))  # 简化：使用相同编码
                else:
                    neighbor_features.append([0.0] * self.hidden_dim)

            # 相对位置编码
            rel_pos = [0.0, 0.0, 0.0]  # 简化

            # 拼接邻居特征
            all_neighbors = []
            for nf in neighbor_features:
                all_neighbors.extend(nf)
            all_neighbors.extend(rel_pos)

            # 空间消息传递
            msg = self._linear(all_neighbors,
                               self.model_weights[f'{prefix}_w1'],
                               self.model_weights[f'{prefix}_b1'])
            msg = [self._silu(x) for x in msg]
            msg = self._linear(msg,
                               self.model_weights[f'{prefix}_w2'],
                               self.model_weights[f'{prefix}_b2'])

            # 残差连接
            h = [h[m] + msg[m] * 0.1 for m in range(self.hidden_dim)]

        # 输出修正量
        vel_corr = self._linear(h, self.model_weights['vel_corr_w'], self.model_weights['vel_corr_b'])
        pres_corr = self._linear(h, self.model_weights['pres_corr_w'], self.model_weights['pres_corr_b'])

        return vel_corr[0], vel_corr[1], vel_corr[2], pres_corr[0]

    def solve_navier_stokes(
        self,
        num_iterations: int = 500,
        tolerance: float = 1e-6,
        relaxation: float = 0.7,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        求解Navier-Stokes方程

        使用 SIMPLE-like 算法求解不可压缩Navier-Stokes方程：
        连续性方程: div(u) = 0
        动量方程: du/dt + (u·grad)u = -grad(p)/rho + nu*laplacian(u)

        Args:
            num_iterations: 最大迭代次数
            tolerance: 收敛容差
            relaxation: 松弛因子
            verbose: 是否打印求解日志

        Returns:
            求解结果字典
        """
        if verbose:
            print(f"[CFDSolverSHGNN] 开始求解Navier-Stokes方程")
            print(f"  网格: {self.nx} x {self.ny} x {self.nz}")
            print(f"  雷诺数: {self.reynolds_number}")
            print(f"  粘度: {self.fluid_viscosity}")
            print(f"  密度: {self.fluid_density}")
            print(f"  入口速度: {self.inlet_velocity}")
            print(f"  湍流模型: {self.turbulence_model}")
            print(f"  最大迭代: {num_iterations}")

        # 初始化流场
        self._initialize_flow_field()
        self.residual_history = []

        nu = self.fluid_viscosity / self.fluid_density  # 运动粘度

        for iteration in range(num_iterations):
            # 保存旧值
            u_old = copy.deepcopy(self.u)
            v_old = copy.deepcopy(self.v)
            w_old = copy.deepcopy(self.w)
            p_old = copy.deepcopy(self.p)

            max_residual = 0.0

            # 1. 求解动量方程（内部节点）
            for i in range(1, self.nx - 1):
                for j in range(1, self.ny - 1):
                    for k in range(self.nz if self.nz > 1 else 1):
                        # 对流项（上风格式）
                        u_face = self.u[i][j][k]
                        v_face = self.v[i][j][k]
                        w_face = self.w[i][j][k] if self.nz > 1 else 0.0

                        # u动量方程
                        conv_u = (u_face * (self.u[i][j][k] - self.u[i-1][j][k]) / self.dx +
                                  v_face * (self.u[i][j][k] - self.u[i][j-1][k]) / self.dy)

                        diff_u = nu * (
                            (self.u[i+1][j][k] - 2*self.u[i][j][k] + self.u[i-1][j][k]) / (self.dx**2) +
                            (self.u[i][j+1][k] - 2*self.u[i][j][k] + self.u[i][j-1][k]) / (self.dy**2)
                        )

                        dp_dx = (self.p[i+1][j][k] - self.p[i-1][j][k]) / (2 * self.dx)

                        u_new = u_old[i][j][k] + relaxation * (
                            -conv_u + diff_u - dp_dx / self.fluid_density
                        ) * self.dx

                        # v动量方程
                        conv_v = (u_face * (self.v[i][j][k] - self.v[i-1][j][k]) / self.dx +
                                  v_face * (self.v[i][j][k] - self.v[i][j-1][k]) / self.dy)

                        diff_v = nu * (
                            (self.v[i+1][j][k] - 2*self.v[i][j][k] + self.v[i-1][j][k]) / (self.dx**2) +
                            (self.v[i][j+1][k] - 2*self.v[i][j][k] + self.v[i][j-1][k]) / (self.dy**2)
                        )

                        dp_dy = (self.p[i][j+1][k] - self.p[i][j-1][k]) / (2 * self.dy)

                        v_new = v_old[i][j][k] + relaxation * (
                            -conv_v + diff_v - dp_dy / self.fluid_density
                        ) * self.dy

                        # SHGNN湍流修正
                        if self.trained and self.turbulence_model == 'shgnn':
                            grads = self._compute_velocity_gradients(i, j, k)
                            du_shgnn, dv_shgnn, dw_shgnn, dp_shgnn = self._compute_shgnn_correction(i, j, k, grads)
                            u_new += du_shgnn * 0.01
                            v_new += dv_shgnn * 0.01

                        self.u[i][j][k] = u_new
                        self.v[i][j][k] = v_new
                        if self.nz > 1:
                            self.w[i][j][k] = w_face  # 简化

                        # 计算残差
                        res_u = abs(self.u[i][j][k] - u_old[i][j][k])
                        res_v = abs(self.v[i][j][k] - v_old[i][j][k])
                        max_residual = max(max_residual, res_u, res_v)

            # 2. 压力修正（SIMPLE算法）
            for i in range(1, self.nx - 1):
                for j in range(1, self.ny - 1):
                    for k in range(self.nz if self.nz > 1 else 1):
                        # 连续性方程残差
                        div_u = ((self.u[i+1][j][k] - self.u[i-1][j][k]) / (2*self.dx) +
                                 (self.v[i][j+1][k] - self.v[i][j-1][k]) / (2*self.dy))

                        # 压力修正
                        self.p[i][j][k] = p_old[i][j][k] - relaxation * 0.5 * div_u

            # 3. 应用边界条件
            self._apply_boundary_conditions()

            self.iteration_count += 1
            self.residual_history.append(max_residual)

            if verbose and (iteration + 1) % 50 == 0:
                print(f"  迭代 {iteration + 1}/{num_iterations}, 最大残差: {max_residual:.2e}")

            # 检查收敛
            if max_residual < tolerance:
                if verbose:
                    print(f"  在第 {iteration + 1} 次迭代后收敛")
                break

        # 后处理
        results = self._post_process()

        if verbose:
            print(f"[CFDSolverSHGNN] 求解完成")
            print(f"  总迭代次数: {self.iteration_count}")
            print(f"  最终残差: {self.residual_history[-1]:.2e}")

        return results

    def _initialize_flow_field(self) -> None:
        """
        初始化流场变量

        使用均匀入口速度和零压力场作为初始条件。
        """
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz if self.nz > 1 else 1):
                    # 抛物线型入口速度分布
                    y_frac = j / max(self.ny - 1, 1)
                    profile = 4.0 * y_frac * (1.0 - y_frac)

                    self.u[i][j][k] = self.inlet_velocity * profile
                    self.v[i][j][k] = 0.0
                    self.w[i][j][k] = 0.0
                    self.p[i][j][k] = 0.0

        self._apply_boundary_conditions()

    def _post_process(self) -> Dict[str, Any]:
        """
        后处理：计算流场统计量和导出量

        Returns:
            后处理结果字典
        """
        # 计算全局统计量
        max_velocity = 0.0
        avg_velocity = 0.0
        max_pressure = float('-inf')
        min_pressure = float('inf')
        total_cells = 0

        vorticity_max = 0.0
        strain_max = 0.0

        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz if self.nz > 1 else 1):
                    vel_mag = math.sqrt(
                        self.u[i][j][k]**2 +
                        self.v[i][j][k]**2 +
                        self.w[i][j][k]**2
                    )
                    max_velocity = max(max_velocity, vel_mag)
                    avg_velocity += vel_mag
                    max_pressure = max(max_pressure, self.p[i][j][k])
                    min_pressure = min(min_pressure, self.p[i][j][k])
                    total_cells += 1

                    if 0 < i < self.nx-1 and 0 < j < self.ny-1:
                        grads = self._compute_velocity_gradients(i, j, k)
                        vort_mag = math.sqrt(
                            grads['vorticity_x']**2 +
                            grads['vorticity_y']**2 +
                            grads['vorticity_z']**2
                        )
                        vorticity_max = max(vorticity_max, vort_mag)
                        strain_max = max(strain_max, grads['strain_rate'])

        avg_velocity /= max(total_cells, 1)

        # 计算阻力系数（简化）
        # F_d = integral(p * n + tau_w * t) dA
        drag_force = 0.0
        for j in range(self.ny):
            drag_force += self.p[1][j][0] * self.dy  # 简化的阻力计算

        drag_coeff = 2.0 * drag_force / (self.fluid_density * self.inlet_velocity**2 * self.domain_size[1])

        return {
            'max_velocity': round(max_velocity, 6),
            'avg_velocity': round(avg_velocity, 6),
            'max_pressure': round(max_pressure, 6),
            'min_pressure': round(min_pressure, 6),
            'pressure_drop': round(max_pressure - min_pressure, 6),
            'max_vorticity': round(vorticity_max, 6),
            'max_strain_rate': round(strain_max, 6),
            'drag_coefficient': round(drag_coeff, 6),
            'iterations': self.iteration_count,
            'residual_history': self.residual_history,
            'converged': self.residual_history[-1] < 1e-4 if self.residual_history else False,
        }

    def compute_derived_quantities(self) -> Dict[str, Any]:
        """
        计算流场导出量

        包括涡量场、应变率场、Q准则场和速度模场。

        Returns:
            导出量字典
        """
        vorticity_field = []
        strain_rate_field = []
        q_criterion_field = []
        velocity_magnitude = []

        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz if self.nz > 1 else 1):
                    vel_mag = math.sqrt(
                        self.u[i][j][k]**2 +
                        self.v[i][j][k]**2 +
                        self.w[i][j][k]**2
                    )
                    velocity_magnitude.append(vel_mag)

                    if 0 < i < self.nx-1 and 0 < j < self.ny-1:
                        grads = self._compute_velocity_gradients(i, j, k)
                        vort_mag = math.sqrt(
                            grads['vorticity_x']**2 +
                            grads['vorticity_y']**2 +
                            grads['vorticity_z']**2
                        )
                        vorticity_field.append(vort_mag)
                        strain_rate_field.append(grads['strain_rate'])
                        q_criterion_field.append(grads['q_criterion'])
                    else:
                        vorticity_field.append(0.0)
                        strain_rate_field.append(0.0)
                        q_criterion_field.append(0.0)

        return {
            'velocity_magnitude': {
                'mean': round(sum(velocity_magnitude) / len(velocity_magnitude), 6),
                'max': round(max(velocity_magnitude), 6),
                'min': round(min(velocity_magnitude), 6),
            },
            'vorticity': {
                'mean': round(sum(vorticity_field) / len(vorticity_field), 6),
                'max': round(max(vorticity_field), 6),
            },
            'strain_rate': {
                'mean': round(sum(strain_rate_field) / len(strain_rate_field), 6),
                'max': round(max(strain_rate_field), 6),
            },
            'q_criterion': {
                'mean': round(sum(q_criterion_field) / len(q_criterion_field), 6),
                'max': round(max(q_criterion_field), 6),
                'min': round(min(q_criterion_field), 6),
            },
        }

    def train_turbulence_model(
        self,
        training_data: List[Dict[str, Any]],
        num_epochs: int = 50,
        learning_rate: float = 0.001,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        训练SHGNN湍流模型

        使用高保真DNS/LES数据训练SHGNN湍流闭合模型。

        Args:
            training_data: 训练数据
            num_epochs: 训练轮数
            learning_rate: 学习率
            verbose: 是否打印日志

        Returns:
            训练结果
        """
        if verbose:
            print(f"[CFDSolverSHGNN] 开始训练SHGNN湍流模型")
            print(f"  训练样本数: {len(training_data)}")
            print(f"  训练轮数: {num_epochs}")

        loss_history = []

        for epoch in range(num_epochs):
            total_loss = 0.0

            for sample in training_data:
                features = sample['features']
                target_turb_visc = sample['turbulent_viscosity']

                # 前向传播
                h = self._linear(features, self.model_weights['encoder_w'], self.model_weights['encoder_b'])
                h = [self._silu(x) for x in h]

                pred_turb_visc = self._linear(h, self.model_weights['turb_visc_w'], self.model_weights['turb_visc_b'])

                # 损失
                loss = (pred_turb_visc[0] - target_turb_visc) ** 2
                total_loss += loss

            avg_loss = total_loss / len(training_data)
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
            print(f"[CFDSolverSHGNN] 湍流模型训练完成")

        return {
            'final_loss': loss_history[-1],
            'loss_history': loss_history,
        }

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

    def get_solver_info(self) -> Dict[str, Any]:
        """
        获取求解器信息

        Returns:
            求解器配置和状态信息
        """
        return {
            'model_name': 'CFDSolverSHGNN',
            'grid_size': [self.nx, self.ny, self.nz],
            'domain_size': self.domain_size,
            'reynolds_number': self.reynolds_number,
            'fluid_viscosity': self.fluid_viscosity,
            'fluid_density': self.fluid_density,
            'inlet_velocity': self.inlet_velocity,
            'solver_type': self.solver_type,
            'turbulence_model': self.turbulence_model,
            'boundary_conditions': self.boundary_conditions,
            'iterations': self.iteration_count,
            'converged': self.residual_history[-1] < 1e-4 if self.residual_history else False,
            'final_residual': self.residual_history[-1] if self.residual_history else None,
        }

    def export_results(self, filepath: str) -> None:
        """
        导出求解结果

        Args:
            filepath: 导出路径
        """
        results = {
            'solver_info': self.get_solver_info(),
            'post_process': self._post_process(),
            'derived_quantities': self.compute_derived_quantities(),
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
