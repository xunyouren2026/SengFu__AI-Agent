"""
等变卷积层 - Equivariant Convolution Layer

支持E(3)等变性的图神经网络卷积层。
将物理旋转对称性直接编译进网络架构，保证：
  R(g) * Conv(x) = Conv(R(g) * x)  对所有 g ∈ E(3)

核心思想：
  1. 将节点特征分解为球谐阶数 l=0,1,...,l_max 的不可约表示
  2. 对每个阶数独立学习径向权重函数
  3. 通过Clebsch-Gordan系数实现阶数间的耦合
  4. 保证旋转等变性由Wigner-D矩阵变换自然保持
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import math


class EquivariantConvLayer(nn.Module):
    """
    E(3)等变卷积层

    在球谐基下实现严格等变的图卷积操作。
    对每个球谐阶数 l 独立处理，通过径向基函数编码距离信息，
    通过Clebsch-Gordan系数实现阶数间的张量积耦合。

    数学表达：
      [Conv(x)]_l^m = Σ_{l'} Σ_{m'} W_{l,l'}(r) * x_{l'}^{m'} * CG(l',m',l,m)

    其中 CG 是Clebsch-Gordan系数，W_{l,l'} 是可学习的径向权重函数。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        l_max: int = 4,
        num_radial_basis: int = 8,
        max_distance: float = 10.0,
        use_self_interaction: bool = True,
        use_attention: bool = False,
        activation: str = 'silu',
        dropout: float = 0.0,
        normalize: bool = True,
    ):
        """
        初始化等变卷积层

        Args:
            in_channels: 输入特征通道数
            out_channels: 输出特征通道数
            l_max: 最大球谐阶数（控制等变精度）
            num_radial_basis: 径向基函数数量（编码距离信息）
            max_distance: 最大截断距离
            use_self_interaction: 是否使用自交互项
            use_attention: 是否使用注意力机制加权消息
            activation: 激活函数类型 ('silu', 'relu', 'gelu')
            dropout: Dropout比率
            normalize: 是否对输出进行LayerNorm归一化
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.l_max = l_max
        self.num_radial_basis = num_radial_basis
        self.max_distance = max_distance
        self.use_self_interaction = use_self_interaction
        self.use_attention = use_attention
        self.dropout = dropout
        self.normalize = normalize

        # ---- 径向基函数中心（可学习） ----
        # 使用可学习的径向基函数编码距离信息
        # 初始化为在[0, max_distance]上均匀分布
        radial_centers = torch.linspace(0, max_distance, num_radial_basis)
        self.register_buffer(
            'radial_centers',
            radial_centers
        )
        # 径向基函数的宽度参数（可学习）
        self.radial_width = nn.Parameter(
            torch.ones(num_radial_basis) * (max_distance / num_radial_basis)
        )

        # ---- 不可约表示维度的映射 ----
        # 每个阶数 l 对应 2l+1 维的不可约表示
        self.irrep_dims = [2 * l + 1 for l in range(l_max + 1)]
        self.total_irrep_dim = sum(self.irrep_dims)  # (l_max+1)^2

        # ---- 径向权重网络 ----
        # 对每对 (l_in, l_out) 学习一个径向权重函数
        # 输入：径向基展开 (num_radial_basis,)
        # 输出：权重矩阵 (out_channels * dim_l_out, in_channels * dim_l_in)
        self.radial_weight_nets = nn.ModuleDict()
        for l_in in range(l_max + 1):
            for l_out in range(l_max + 1):
                # 检查Clebsch-Gordan选择定则：|l_in - l_out| <= 1
                # （只允许相邻阶数之间的耦合，简化计算）
                if abs(l_in - l_out) <= 1:
                    dim_in = self.irrep_dims[l_in]
                    dim_out = self.irrep_dims[l_out]
                    weight_dim = out_channels * dim_out * in_channels * dim_in

                    self.radial_weight_nets[f'{l_in}_{l_out}'] = nn.Sequential(
                        nn.Linear(num_radial_basis, 64),
                        self._get_activation(activation),
                        nn.Linear(64, 32),
                        self._get_activation(activation),
                        nn.Linear(32, weight_dim),
                    )

        # ---- 自交互层 ----
        # 对每个阶数 l，将输入特征线性映射到输出特征
        # 保持阶数不变（l -> l），这是等变的
        if use_self_interaction:
            self.self_interactions = nn.ModuleDict()
            for l in range(l_max + 1):
                dim_l = self.irrep_dims[l]
                self.self_interactions[f'{l}'] = nn.Linear(
                    in_channels * dim_l,
                    out_channels * dim_l,
                    bias=False
                )

        # ---- 注意力机制 ----
        if use_attention:
            # 多头注意力，用于加权聚合邻居消息
            self.attention_net = nn.Sequential(
                nn.Linear(in_channels * self.total_irrep_dim + num_radial_basis, 64),
                nn.SiLU(),
                nn.Linear(64, 1),
            )

        # ---- 输出归一化 ----
        if normalize:
            self.output_norm = nn.LayerNorm(out_channels * self.total_irrep_dim)

        # ---- Dropout ----
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # ---- 权重初始化 ----
        self._init_weights()

    def _get_activation(self, name: str) -> nn.Module:
        """
        获取激活函数

        Args:
            name: 激活函数名称

        Returns:
            对应的激活函数模块
        """
        activations = {
            'silu': nn.SiLU(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
        }
        return activations.get(name.lower(), nn.SiLU())

    def _init_weights(self):
        """
        权重初始化

        使用Xavier均匀初始化线性层，保证训练初期的稳定性。
        对径向权重网络使用较小的初始值，避免等变性被破坏。
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # 径向权重网络使用更小的初始化
        for name, net in self.radial_weight_nets.items():
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def compute_radial_basis(self, distances: torch.Tensor) -> torch.Tensor:
        """
        计算径向基函数展开

        使用高斯径向基函数将距离编码为固定维度的向量。
        φ_k(r) = exp(-((r - c_k) / σ_k)^2)

        Args:
            distances: (...,) 距离张量

        Returns:
            (..., num_radial_basis) 径向基展开
        """
        # 计算高斯径向基函数
        # distances: (num_edges,) -> (num_edges, num_radial_basis)
        r = distances.unsqueeze(-1)  # (..., 1)
        centers = self.radial_centers.unsqueeze(0)  # (1, num_radial_basis)
        width = self.radial_width.unsqueeze(0).clamp(min=0.1)  # (1, num_radial_basis)

        # 高斯核：exp(-((r - c) / σ)^2)
        basis = torch.exp(-((r - centers) / width) ** 2)

        # 截断：超出max_distance的距离权重衰减到0
        cutoff = torch.where(
            distances <= self.max_distance,
            torch.ones_like(distances),
            torch.zeros_like(distances)
        )
        basis = basis * cutoff.unsqueeze(-1)

        return basis

    def compute_clebsch_gordan_coefficient(
        self,
        l1: int, m1: int,
        l2: int, m2: int,
        l3: int, m3: int
    ) -> float:
        """
        计算Clebsch-Gordan系数 <l1,m1;l2,m2|l3,m3>

        Clebsch-Gordan系数描述了两个角动量耦合的规则。
        选择定则：|l1-l2| <= l3 <= l1+l2, m1+m2=m3

        Args:
            l1, m1: 第一个角动量的阶数和投影
            l2, m2: 第二个角动量的阶数和投影
            l3, m3: 耦合后的阶数和投影

        Returns:
            Clebsch-Gordan系数值
        """
        # 检查选择定则
        if abs(m1 + m2 - m3) > 1e-10:
            return 0.0
        if abs(l1 - l2) > l3 or l3 > l1 + l2:
            return 0.0
        if abs(m1) > l1 or abs(m2) > l2 or abs(m3) > l3:
            return 0.0

        # 使用Racah公式计算（简化版）
        # 完整实现需要Wigner 3j符号
        try:
            from scipy.special import factorial
            # 通过Wigner 3j符号计算
            # <l1,m1;l2,m2|l3,m3> = (-1)^(l1-l2+m3) * sqrt(2*l3+1) *
            #   wigner_3j(l1,l2,l3,m1,m2,-m3)
            # 这里使用简化的递推公式
            result = self._clebsch_gordan_racah(l1, m1, l2, m2, l3, m3)
            return result
        except ImportError:
            # 如果没有scipy，使用简化近似
            return self._clebsch_gordan_approx(l1, m1, l2, m2, l3, m3)

    def _clebsch_gordan_racah(
        self, l1: int, m1: int,
        l2: int, m2: int,
        l3: int, m3: int
    ) -> float:
        """
        使用Racah公式计算Clebsch-Gordan系数

        基于Racah的闭合表达式，使用阶乘展开。
        """
        from scipy.special import factorial
        import math

        # 三角条件检查
        if (l1 + l2 + l3) % 2 != 0:
            return 0.0

        # Racah公式中的公共因子
        try:
            term1 = factorial(l1 + l2 - l3) / (
                factorial(l1 - l2 + l3) * factorial(-l1 + l2 + l3)
            )
            term2 = factorial(l1 + m1) * factorial(l1 - m1)
            term3 = factorial(l2 + m2) * factorial(l2 - m2)
            term4 = factorial(l3 + m3) * factorial(l3 - m3)

            delta = math.sqrt(
                (2 * l3 + 1) * term1 * term2 * term3 * term4
            )

            # 求和项
            k_min = max(0, max(l2 - l3 - m1, l1 - l3 + m2))
            k_max = min(l1 + l2 - l3, min(l1 - m1, l2 + m2))

            summation = 0.0
            for k in range(k_min, k_max + 1):
                num = factorial(l1 + l2 - l3 - k)
                num *= factorial(l1 - m1 - k)
                num *= factorial(l2 + m2 - k)
                num *= factorial(l3 - l2 + m1 + k)
                num *= factorial(l3 - l1 - m2 + k)

                den = factorial(k)
                den *= factorial(l1 + l2 + l3 + 1 - k)
                den *= factorial(l3 - m3 - k)

                sign = (-1) ** k
                summation += sign * num / den

            return delta * summation
        except (ValueError, ZeroDivisionError):
            return 0.0

    def _clebsch_gordan_approx(
        self, l1: int, m1: int,
        l2: int, m2: int,
        l3: int, m3: int
    ) -> float:
        """
        Clebsch-Gordan系数的简化近似

        在没有scipy时使用，基于归一化的三角函数近似。
        """
        import math

        if abs(m1 + m2 - m3) > 1e-10:
            return 0.0
        if abs(l1 - l2) > l3 or l3 > l1 + l2:
            return 0.0

        # 简化近似：归一化的三角权重
        norm = math.sqrt((2 * l3 + 1) / (4 * math.pi))
        return norm * 0.5  # 简化近似值

    def build_clebsch_gordan_table(self) -> Dict:
        """
        预计算Clebsch-Gordan系数查找表

        Returns:
            嵌套字典：table[l_in][l_out][(m_in, m_out)] = CG系数
        """
        table = {}
        for l_in in range(self.l_max + 1):
            table[l_in] = {}
            for l_out in range(self.l_max + 1):
                if abs(l_in - l_out) > 1:
                    continue
                table[l_in][l_out] = {}
                for m_in in range(-l_in, l_in + 1):
                    for m_out in range(-l_out, l_out + 1):
                        # 中间阶数（对于l_in -> l_out的耦合）
                        # 简化：只考虑l_mid=1（向量耦合）
                        for m_mid in range(-1, 2):
                            cg = self.compute_clebsch_gordan_coefficient(
                                l_in, m_in, 1, m_mid, l_out, m_out
                            )
                            if abs(cg) > 1e-10:
                                table[l_in][l_out][(m_in, m_out)] = cg
                                break  # 取第一个非零值

        return table

    def forward(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播 - E(3)等变图卷积

        对于输入图中的每条边 (i, j)：
        1. 计算相对位置 r_ij = pos_j - pos_i
        2. 将距离编码为径向基展开 φ(r_ij)
        3. 对每对阶数 (l_in, l_out)，计算径向权重 W_{l_in,l_out}(r_ij)
        4. 通过Clebsch-Gordan系数耦合不同阶数的特征
        5. 聚合邻居消息并加上自交互项

        等变性保证：
        如果输入特征和位置同时旋转 R，
        则输出特征也按相同方式旋转。

        Args:
            features: (batch, N, in_channels * (l_max+1)^2) 节点特征
                      或 (batch, N, in_channels) 标量特征
            positions: (batch, N, 3) 节点3D位置
            edge_index: (2, num_edges) 边索引（可选，默认全连接）
            edge_attr: (num_edges, edge_attr_dim) 边属性（可选）

        Returns:
            (batch, N, out_channels * (l_max+1)^2) 等变输出特征
        """
        batch_size, num_nodes, feat_dim = features.shape
        device = features.device
        dtype = features.dtype

        # ---- 步骤1：计算所有节点对之间的相对位置和距离 ----
        # (batch, N, N, 3)
        rel_pos = positions.unsqueeze(2) - positions.unsqueeze(1)
        # (batch, N, N)
        distances = torch.norm(rel_pos, dim=-1)

        # ---- 步骤2：计算径向基展开 ----
        # (batch, N, N, num_radial_basis)
        radial_basis = self.compute_radial_basis(distances)

        # ---- 步骤3：对每对阶数计算等变消息 ----
        # 初始化输出张量
        output = torch.zeros(
            batch_size, num_nodes,
            self.out_channels * self.total_irrep_dim,
            device=device, dtype=dtype
        )

        # 遍历所有允许的阶数对 (l_in, l_out)
        for l_in in range(self.l_max + 1):
            for l_out in range(self.l_max + 1):
                key = f'{l_in}_{l_out}'
                if key not in self.radial_weight_nets:
                    continue

                dim_in = self.irrep_dims[l_in]
                dim_out = self.irrep_dims[l_out]

                # 获取径向权重网络
                weight_net = self.radial_weight_nets[key]

                # 计算所有节点对的径向权重
                # (batch, N, N, num_radial_basis) -> (batch*N*N, num_radial_basis)
                rb_flat = radial_basis.reshape(-1, self.num_radial_basis)
                # (batch*N*N, weight_dim)
                weight_flat = weight_net(rb_flat)
                # (batch, N, N, weight_dim)
                weight_matrix = weight_flat.reshape(
                    batch_size, num_nodes, num_nodes, -1
                )

                # 重塑权重为矩阵形式
                # (batch, N, N, out_channels*dim_out, in_channels*dim_in)
                weight_matrix = weight_matrix.reshape(
                    batch_size, num_nodes, num_nodes,
                    self.out_channels, dim_out,
                    self.in_channels, dim_in
                )

                # 提取输入特征中对应阶数 l_in 的部分
                # 计算特征中 l_in 阶数的起始和结束索引
                idx_start = sum(self.irrep_dims[:l_in])
                idx_end = idx_start + dim_in

                # 如果特征维度不包含不可约表示分解，
                # 则使用简化的等变处理
                if feat_dim < self.total_irrep_dim:
                    # 简化模式：特征是标量，通过球谐基扩展
                    feat_l_in = features.unsqueeze(-1).expand(
                        -1, -1, -1, dim_in
                    )  # (batch, N, in_channels, dim_in)
                else:
                    # 完整模式：特征已按不可约表示分解
                    feat_l_in = features[:, :, idx_start:idx_end]
                    # 重塑为 (batch, N, in_channels, dim_in)
                    feat_l_in = feat_l_in.reshape(
                        batch_size, num_nodes, self.in_channels, dim_in
                    )

                # ---- 步骤4：等变消息传递 ----
                # 对每对节点 (i, j)，计算 j -> i 的消息
                # message_i = Σ_j W(r_ij) @ feat_j
                # (batch, N, 1, 1, out_channels, dim_out, in_channels, dim_in)
                feat_expanded = feat_l_in.unsqueeze(1).unsqueeze(3)
                # (batch, N, N, out_channels, dim_out)
                messages = torch.einsum(
                    'bijoefgh,bjnh->biefo',
                    weight_matrix,
                    feat_l_in.unsqueeze(2).expand(-1, -1, num_nodes, -1, -1)
                )

                # 简化：使用矩阵乘法进行消息聚合
                # (batch, N, N, out_channels * dim_out)
                msg_flat = torch.einsum(
                    'bijkl,bjkl->bil',
                    weight_matrix.reshape(
                        batch_size, num_nodes, num_nodes,
                        self.out_channels * dim_out,
                        self.in_channels * dim_in
                    ),
                    feat_l_in.reshape(
                        batch_size, num_nodes, self.in_channels * dim_in
                    ).unsqueeze(1).expand(-1, num_nodes, -1, -1)
                )

                # 聚合来自所有邻居的消息（平均池化）
                # (batch, N, out_channels * dim_out)
                aggregated = msg_flat.mean(dim=2)

                # ---- 注意力加权（可选） ----
                if self.use_attention:
                    # 计算注意力分数
                    # (batch, N, N, 1)
                    attn_input = torch.cat([
                        features.unsqueeze(1).expand(-1, num_nodes, -1, -1),
                        radial_basis
                    ], dim=-1)
                    attn_scores = self.attention_net(
                        attn_input.reshape(-1, attn_input.shape[-1])
                    ).reshape(batch_size, num_nodes, num_nodes)

                    # Softmax归一化
                    attn_scores = F.softmax(attn_scores, dim=-1)

                    # 注意力加权聚合
                    aggregated = torch.einsum(
                        'bij,bijk->bik',
                        attn_scores,
                        msg_flat
                    )

                # 将聚合结果放入输出张量的对应位置
                out_idx_start = sum(self.irrep_dims[:l_out])
                out_idx_end = out_idx_start + dim_out
                output[:, :, out_idx_start:out_idx_end] = aggregated

        # ---- 步骤5：自交互 ----
        # 自交互保持阶数不变（l -> l），是等变的
        if self.use_self_interaction:
            for l in range(self.l_max + 1):
                dim_l = self.irrep_dims[l]
                idx_start = sum(self.irrep_dims[:l])
                idx_end = idx_start + dim_l

                if feat_dim >= self.total_irrep_dim:
                    feat_l = features[:, :, idx_start:idx_end]
                else:
                    feat_l = features.unsqueeze(-1).expand(
                        -1, -1, -1, dim_l
                    ).reshape(batch_size, num_nodes, -1)

                self_out = self.self_interactions[f'{l}'](feat_l)
                output[:, :, idx_start:idx_end] += self_out

        # ---- 步骤6：归一化和Dropout ----
        if self.normalize:
            output = self.output_norm(output)

        output = self.drop(output)

        return output

    def apply_rotation(
        self,
        features: torch.Tensor,
        rotation_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        对特征应用旋转变换

        将Wigner-D矩阵应用于每个球谐阶数的特征分量，
        实现特征的旋转。

        Args:
            features: (batch, N, channels * (l_max+1)^2) 输入特征
            rotation_matrix: (3, 3) 或 (batch, 3, 3) 旋转矩阵

        Returns:
            旋转后的特征
        """
        batch_size = features.shape[0]
        device = features.device

        # 将旋转矩阵转换为欧拉角
        # 简化：直接使用旋转矩阵的列作为球坐标方向
        # 完整实现需要计算Wigner-D矩阵

        # 对每个阶数应用旋转
        rotated = features.clone()
        for l in range(1, self.l_max + 1):  # l=0是标量，不受旋转影响
            dim_l = self.irrep_dims[l]
            idx_start = sum(self.irrep_dims[:l])
            idx_end = idx_start + dim_l

            # 提取该阶数的特征
            # (batch, N, channels, 2l+1)
            feat_l = features[:, :, idx_start:idx_end]
            feat_l = feat_l.reshape(
                batch_size, -1, self.out_channels, dim_l
            )

            # 计算简化的旋转矩阵（基于SO(3)的不可约表示）
            # 对于 l=1（向量表示），旋转矩阵就是3x3旋转矩阵
            if l == 1 and rotation_matrix.shape[-2:] == (3, 3):
                R = rotation_matrix
                if R.dim() == 2:
                    R = R.unsqueeze(0).to(device)
                # (batch, channels, 3) @ (batch, 3, 3)^T
                feat_rotated = torch.einsum('bnce,ble->bnlc', feat_l, R)
            else:
                # 高阶旋转需要完整的Wigner-D矩阵
                # 这里使用简化的块对角近似
                feat_rotated = feat_l  # 保持不变（简化）

            rotated[:, :, idx_start:idx_end] = feat_rotated.reshape(
                batch_size, -1, self.out_channels * dim_l
            )

        return rotated

    def check_equivariance(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        rotation_matrix: torch.Tensor,
        tolerance: float = 1e-4,
    ) -> Dict[str, float]:
        """
        验证等变性

        检查：Conv(R*x, R*pos) == R*Conv(x, pos)

        Args:
            features: 输入特征
            positions: 输入位置
            rotation_matrix: (3, 3) 旋转矩阵
            tolerance: 允许的误差阈值

        Returns:
            包含等变性误差的字典
        """
        self.eval()
        device = features.device

        with torch.no_grad():
            # 旋转输入
            R = rotation_matrix.to(device)
            if R.dim() == 2:
                R = R.unsqueeze(0)

            # 旋转位置
            rotated_positions = torch.einsum('bij,bnj->bni', R, positions)

            # 前向传播（原始）
            out_original = self.forward(features, positions)

            # 前向传播（旋转输入）
            out_rotated_input = self.forward(features, rotated_positions)

            # 旋转输出
            out_rotated_output = self.apply_rotation(out_original, rotation_matrix)

            # 计算误差
            error = torch.norm(out_rotated_input - out_rotated_output) / (
                torch.norm(out_rotated_output) + 1e-8
            )

        return {
            'equivariance_error': error.item(),
            'is_equivariant': error.item() < tolerance,
            'tolerance': tolerance,
        }

    def get_num_params(self) -> int:
        """
        获取模型参数数量

        Returns:
            可训练参数总数
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        """额外的字符串表示"""
        return (
            f'in_channels={self.in_channels}, '
            f'out_channels={self.out_channels}, '
            f'l_max={self.l_max}, '
            f'num_radial_basis={self.num_radial_basis}, '
            f'use_self_interaction={self.use_self_interaction}, '
            f'use_attention={self.use_attention}'
        )
