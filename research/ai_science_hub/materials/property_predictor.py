"""
AI科学中心 - 材料科学模块

使用SH-GNN进行材料属性预测，包括带隙、硬度、热导率、
弹性模量等关键材料属性的预测与分析。
支持晶体结构、非晶材料和复合材料的属性预测。
"""

import math
import json
import random
import hashlib
from typing import Dict, Tuple, Optional, List, Any


class MaterialsPropertyPredictor:
    """
    材料属性预测器 - 基于SHGNN预测材料属性

    使用球谐图神经网络（SH-GNN）对材料的晶体结构进行建模，
    预测带隙、硬度、热导率、弹性模量等多种材料属性。

    Attributes:
        input_dim: 输入特征维度（原子序数、电负性、原子半径等）
        hidden_dim: 隐藏层维度
        num_properties: 预测的属性数量
        l_max: 球谐函数最大阶数
        num_layers: GNN层数
        trained: 模型是否已训练
        property_names: 预测属性名称列表
        model_weights: 模型权重参数
    """

    # 支持预测的材料属性
    DEFAULT_PROPERTIES = [
        'band_gap',       # 带隙 (eV)
        'hardness',       # 硬度 (GPa)
        'thermal_conductivity',  # 热导率 (W/m·K)
        'bulk_modulus',   # 体积模量 (GPa)
        'shear_modulus',  # 剪切模量 (GPa)
        'density',        # 密度 (g/cm³)
        'melting_point',  # 熔点 (K)
        'dielectric_constant',  # 介电常数
    ]

    # 元素周期表基本属性（前36个元素的简化数据）
    ELEMENT_PROPERTIES = {
        1:  {'symbol': 'H',  'electronegativity': 2.20, 'radius': 0.53, 'mass': 1.008, 'group': 1},
        2:  {'symbol': 'He', 'electronegativity': 0.00, 'radius': 0.31, 'mass': 4.003, 'group': 18},
        3:  {'symbol': 'Li', 'electronegativity': 0.98, 'radius': 1.67, 'mass': 6.941, 'group': 1},
        4:  {'symbol': 'Be', 'electronegativity': 1.57, 'radius': 1.12, 'mass': 9.012, 'group': 2},
        5:  {'symbol': 'B',  'electronegativity': 2.04, 'radius': 0.87, 'mass': 10.81, 'group': 13},
        6:  {'symbol': 'C',  'electronegativity': 2.55, 'radius': 0.77, 'mass': 12.01, 'group': 14},
        7:  {'symbol': 'N',  'electronegativity': 3.04, 'radius': 0.75, 'mass': 14.01, 'group': 15},
        8:  {'symbol': 'O',  'electronegativity': 3.44, 'radius': 0.73, 'mass': 16.00, 'group': 16},
        9:  {'symbol': 'F',  'electronegativity': 3.98, 'radius': 0.71, 'mass': 19.00, 'group': 17},
        10: {'symbol': 'Ne', 'electronegativity': 0.00, 'radius': 0.38, 'mass': 20.18, 'group': 18},
        11: {'symbol': 'Na', 'electronegativity': 0.93, 'radius': 1.90, 'mass': 22.99, 'group': 1},
        12: {'symbol': 'Mg', 'electronegativity': 1.31, 'radius': 1.45, 'mass': 24.31, 'group': 2},
        13: {'symbol': 'Al', 'electronegativity': 1.61, 'radius': 1.18, 'mass': 26.98, 'group': 13},
        14: {'symbol': 'Si', 'electronegativity': 1.90, 'radius': 1.11, 'mass': 28.09, 'group': 14},
        15: {'symbol': 'P',  'electronegativity': 2.19, 'radius': 1.06, 'mass': 30.97, 'group': 15},
        16: {'symbol': 'S',  'electronegativity': 2.58, 'radius': 1.02, 'mass': 32.07, 'group': 16},
        17: {'symbol': 'Cl', 'electronegativity': 3.16, 'radius': 0.99, 'mass': 35.45, 'group': 17},
        18: {'symbol': 'Ar', 'electronegativity': 0.00, 'radius': 0.71, 'mass': 39.95, 'group': 18},
        19: {'symbol': 'K',  'electronegativity': 0.82, 'radius': 2.43, 'mass': 39.10, 'group': 1},
        20: {'symbol': 'Ca', 'electronegativity': 1.00, 'radius': 1.94, 'mass': 40.08, 'group': 2},
        21: {'symbol': 'Sc', 'electronegativity': 1.36, 'radius': 1.84, 'mass': 44.96, 'group': 3},
        22: {'symbol': 'Ti', 'electronegativity': 1.54, 'radius': 1.76, 'mass': 47.87, 'group': 4},
        23: {'symbol': 'V',  'electronegativity': 1.63, 'radius': 1.71, 'mass': 50.94, 'group': 5},
        24: {'symbol': 'Cr', 'electronegativity': 1.66, 'radius': 1.66, 'mass': 52.00, 'group': 6},
        25: {'symbol': 'Mn', 'electronegativity': 1.55, 'radius': 1.61, 'mass': 54.94, 'group': 7},
        26: {'symbol': 'Fe', 'electronegativity': 1.83, 'radius': 1.56, 'mass': 55.85, 'group': 8},
        27: {'symbol': 'Co', 'electronegativity': 1.88, 'radius': 1.52, 'mass': 58.93, 'group': 9},
        28: {'symbol': 'Ni', 'electronegativity': 1.91, 'radius': 1.49, 'mass': 58.69, 'group': 10},
        29: {'symbol': 'Cu', 'electronegativity': 1.90, 'radius': 1.45, 'mass': 63.55, 'group': 11},
        30: {'symbol': 'Zn', 'electronegativity': 1.65, 'radius': 1.42, 'mass': 65.38, 'group': 12},
        31: {'symbol': 'Ga', 'electronegativity': 1.81, 'radius': 1.36, 'mass': 69.72, 'group': 13},
        32: {'symbol': 'Ge', 'electronegativity': 2.01, 'radius': 1.25, 'mass': 72.63, 'group': 14},
        33: {'symbol': 'As', 'electronegativity': 2.18, 'radius': 1.14, 'mass': 74.92, 'group': 15},
        34: {'symbol': 'Se', 'electronegativity': 2.55, 'radius': 1.03, 'mass': 78.97, 'group': 16},
        35: {'symbol': 'Br', 'electronegativity': 2.96, 'radius': 0.94, 'mass': 79.90, 'group': 17},
        36: {'symbol': 'Kr', 'electronegativity': 3.00, 'radius': 0.88, 'mass': 83.80, 'group': 18},
    }

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 128,
        num_properties: int = 8,
        l_max: int = 6,
        num_layers: int = 4,
        learning_rate: float = 0.001,
        random_seed: Optional[int] = None
    ):
        """
        初始化材料属性预测器

        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_properties: 预测属性数量
            l_max: 球谐函数最大阶数
            num_layers: GNN层数
            learning_rate: 学习率
            random_seed: 随机种子
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_properties = min(num_properties, len(self.DEFAULT_PROPERTIES))
        self.l_max = l_max
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.trained = False
        self.property_names = self.DEFAULT_PROPERTIES[:self.num_properties]
        self.training_history = []

        # 设置随机种子
        if random_seed is not None:
            random.seed(random_seed)

        # 初始化模型权重
        self.model_weights = self._initialize_weights()

        # 数据缓存
        self._training_data = []
        self._normalization_stats = {}

    def _initialize_weights(self) -> Dict[str, List[List[float]]]:
        """
        初始化模型权重参数

        使用Xavier初始化策略，确保各层权重的方差合适。

        Returns:
            包含各层权重的字典
        """
        weights = {}

        # 输入投影层权重
        scale_in = math.sqrt(2.0 / (self.input_dim + self.hidden_dim))
        weights['input_weight'] = [
            [random.gauss(0, scale_in) for _ in range(self.input_dim)]
            for _ in range(self.hidden_dim)
        ]
        weights['input_bias'] = [0.0] * self.hidden_dim

        # SH-GNN层权重
        for layer_idx in range(self.num_layers):
            prefix = f'layer_{layer_idx}'

            # 消息传递网络权重
            msg_input_dim = self.hidden_dim * 2 + 3  # 源特征 + 目标特征 + 相对位置
            scale_msg = math.sqrt(2.0 / (msg_input_dim + self.hidden_dim))
            weights[f'{prefix}_msg_weight1'] = [
                [random.gauss(0, scale_msg) for _ in range(msg_input_dim)]
                for _ in range(self.hidden_dim)
            ]
            weights[f'{prefix}_msg_bias1'] = [0.0] * self.hidden_dim

            scale_msg2 = math.sqrt(2.0 / (self.hidden_dim + self.hidden_dim))
            weights[f'{prefix}_msg_weight2'] = [
                [random.gauss(0, scale_msg2) for _ in range(self.hidden_dim)]
                for _ in range(self.hidden_dim)
            ]
            weights[f'{prefix}_msg_bias2'] = [0.0] * self.hidden_dim

            # 自环权重
            scale_self = math.sqrt(2.0 / (self.hidden_dim + self.hidden_dim))
            weights[f'{prefix}_self_weight'] = [
                [random.gauss(0, scale_self) for _ in range(self.hidden_dim)]
                for _ in range(self.hidden_dim)
            ]
            weights[f'{prefix}_self_bias'] = [0.0] * self.hidden_dim

            # 球谐函数系数
            weights[f'{prefix}_sh_coeffs'] = [
                random.gauss(0, 0.1) for _ in range(self.l_max + 1)
            ]

        # 输出头权重
        scale_out = math.sqrt(2.0 / (self.hidden_dim + self.hidden_dim))
        weights['output_weight1'] = [
            [random.gauss(0, scale_out) for _ in range(self.hidden_dim)]
            for _ in range(self.hidden_dim)
        ]
        weights['output_bias1'] = [0.0] * self.hidden_dim

        scale_final = math.sqrt(2.0 / (self.hidden_dim + self.num_properties))
        weights['output_weight2'] = [
            [random.gauss(0, scale_final) for _ in range(self.hidden_dim)]
            for _ in range(self.num_properties)
        ]
        weights['output_bias2'] = [0.0] * self.num_properties

        return weights

    def _get_element_features(self, atomic_number: int) -> List[float]:
        """
        获取元素的原子特征向量

        将元素周期表中的基本属性编码为固定维度的特征向量。

        Args:
            atomic_number: 原子序数

        Returns:
            特征向量列表
        """
        props = self.ELEMENT_PROPERTIES.get(atomic_number, {
            'symbol': 'X', 'electronegativity': 1.5, 'radius': 1.5,
            'mass': 50.0, 'group': 1
        })

        features = [
            atomic_number / 118.0,                    # 归一化原子序数
            props['electronegativity'] / 4.0,         # 归一化电负性
            props['radius'] / 3.0,                    # 归一化原子半径
            props['mass'] / 240.0,                    # 归一化原子质量
            props['group'] / 18.0,                    # 归一化族数
            math.sin(atomic_number * 0.1),            # 周期性编码1
            math.cos(atomic_number * 0.1),            # 周期性编码2
            (atomic_number % 8) / 8.0,                # 电子层编码
        ]

        # 截断或填充到input_dim维度
        if len(features) > self.input_dim:
            features = features[:self.input_dim]
        else:
            features = features + [0.0] * (self.input_dim - len(features))

        return features

    def load_crystal_structure(
        self,
        atomic_numbers: List[int],
        positions: List[List[float]],
        lattice_vectors: Optional[List[List[float]]] = None
    ) -> Dict[str, Any]:
        """
        加载晶体结构数据

        将晶体结构转换为图表示，其中原子为节点，化学键为边。

        Args:
            atomic_numbers: 原子序数列表
            positions: 原子位置列表 [[x, y, z], ...]
            lattice_vectors: 晶格向量（3x3矩阵），用于周期性边界条件

        Returns:
            包含图表示的字典，包括节点特征、边和位置信息
        """
        num_atoms = len(atomic_numbers)

        # 构建节点特征
        node_features = [
            self._get_element_features(z) for z in atomic_numbers
        ]

        # 构建边（基于距离阈值）
        edges = []
        edge_features = []
        cutoff_distance = 5.0  # 默认截断距离（埃）

        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dz = positions[i][2] - positions[j][2]

                # 考虑周期性边界条件
                if lattice_vectors is not None:
                    # 简化的周期性处理
                    for k in range(3):
                        frac = [dx, dy, dz]
                        for dim in range(3):
                            frac[dim] /= lattice_vectors[dim][dim] if lattice_vectors[dim][dim] != 0 else 1.0
                            if frac[dim] > 0.5:
                                frac[dim] -= 1.0
                            elif frac[dim] < -0.5:
                                frac[dim] += 1.0
                        dx, dy, dz = frac

                distance = math.sqrt(dx * dx + dy * dy + dz * dz)

                if distance < cutoff_distance and distance > 0.1:
                    # 双向边
                    edges.append((i, j))
                    edges.append((j, i))

                    # 边特征：相对距离和方向
                    inv_dist = 1.0 / (distance + 1e-8)
                    edge_feat = [dx * inv_dist, dy * inv_dist, dz * inv_dist, distance]
                    edge_features.append(edge_feat)
                    edge_features.append(edge_feat)

        return {
            'node_features': node_features,
            'positions': positions,
            'edges': edges,
            'edge_features': edge_features,
            'num_atoms': num_atoms,
            'lattice_vectors': lattice_vectors,
            'atomic_numbers': atomic_numbers
        }

    def _compute_spherical_harmonics(
        self,
        theta: float,
        phi: float,
        l: int
    ) -> List[float]:
        """
        计算球谐函数值（简化版本）

        使用递推公式计算给定阶数l的球谐函数值。

        Args:
            theta: 极角（弧度）
            phi: 方位角（弧度）
            l: 球谐函数阶数

        Returns:
            阶数l对应的2l+1个球谐函数值
        """
        results = []

        for m in range(-l, l + 1):
            # 简化的球谐函数计算
            # 使用Legendre多项式的近似
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)

            # 归一化因子
            norm = math.sqrt((2 * l + 1) / (4 * math.pi))

            # 简化的Legendre多项式
            if l == 0:
                plm = 1.0
            elif l == 1:
                if m == -1:
                    plm = sin_theta
                elif m == 0:
                    plm = cos_theta
                else:
                    plm = -sin_theta
            elif l == 2:
                if m == -2:
                    plm = sin_theta * sin_theta
                elif m == -1:
                    plm = sin_theta * cos_theta
                elif m == 0:
                    plm = 3 * cos_theta * cos_theta - 1
                elif m == 1:
                    plm = -sin_theta * cos_theta
                else:
                    plm = sin_theta * sin_theta
            else:
                # 高阶使用递推近似
                plm = cos_theta ** l + 0.1 * sin_theta * math.cos(m * phi)

            # 球谐函数值
            ylm = norm * plm * math.cos(m * phi) if m >= 0 else norm * plm * math.sin(abs(m) * phi)
            results.append(ylm)

        return results

    def _linear_layer(
        self,
        input_vec: List[float],
        weight: List[List[float]],
        bias: List[float]
    ) -> List[float]:
        """
        线性变换层

        计算 y = Wx + b

        Args:
            input_vec: 输入向量
            weight: 权重矩阵
            bias: 偏置向量

        Returns:
            输出向量
        """
        output_dim = len(weight)
        result = [0.0] * output_dim

        for i in range(output_dim):
            s = bias[i]
            for j in range(len(input_vec)):
                s += weight[i][j] * input_vec[j]
            result[i] = s

        return result

    def _relu(self, x: float) -> float:
        """ReLU激活函数"""
        return max(0.0, x)

    def _silu(self, x: float) -> float:
        """SiLU（Swish）激活函数"""
        return x / (1.0 + math.exp(-x)) if abs(x) < 500 else (x if x > 0 else 0.0)

    def _vec_silu(self, vec: List[float]) -> List[float]:
        """向量化的SiLU激活函数"""
        return [self._silu(x) for x in vec]

    def _layer_norm(self, vec: List[float]) -> List[float]:
        """层归一化"""
        mean = sum(vec) / len(vec)
        variance = sum((x - mean) ** 2 for x in vec) / len(vec)
        std = math.sqrt(variance + 1e-8)
        return [(x - mean) / std for x in vec]

    def _forward_pass(self, graph: Dict[str, Any]) -> List[float]:
        """
        模型前向传播

        对输入的晶体结构图进行SH-GNN消息传递，输出属性预测。

        Args:
            graph: 晶体结构图表示

        Returns:
            属性预测值列表
        """
        node_features = graph['node_features']
        positions = graph['positions']
        edges = graph['edges']
        num_atoms = graph['num_atoms']

        # 输入投影
        hidden_states = []
        for feat in node_features:
            projected = self._linear_layer(
                feat,
                self.model_weights['input_weight'],
                self.model_weights['input_bias']
            )
            hidden_states.append(self._vec_silu(projected))

        # SH-GNN消息传递层
        for layer_idx in range(self.num_layers):
            prefix = f'layer_{layer_idx}'
            new_states = [list(h) for h in hidden_states]

            # 构建邻接表
            neighbors = [[] for _ in range(num_atoms)]
            for src, dst in edges:
                neighbors[dst].append(src)

            # 消息传递
            for i in range(num_atoms):
                if not neighbors[i]:
                    # 无邻居时只使用自环
                    self_contrib = self._linear_layer(
                        hidden_states[i],
                        self.model_weights[f'{prefix}_self_weight'],
                        self.model_weights[f'{prefix}_self_bias']
                    )
                    new_states[i] = self._vec_silu(self_contrib)
                    continue

                messages = []
                for j in neighbors[i]:
                    # 计算相对位置
                    dx = positions[j][0] - positions[i][0]
                    dy = positions[j][1] - positions[i][1]
                    dz = positions[j][2] - positions[i][2]
                    dist = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-8

                    # 计算球谐函数滤波
                    theta = math.acos(max(-1.0, min(1.0, dz / dist)))
                    phi = math.atan2(dy, dx)

                    sh_filter = 0.0
                    for l in range(min(self.l_max + 1, 4)):
                        sh_vals = self._compute_spherical_harmonics(theta, phi, l)
                        sh_filter += self.model_weights[f'{prefix}_sh_coeffs'][l] * sum(abs(v) for v in sh_vals) / len(sh_vals)

                    # 构建消息输入
                    msg_input = hidden_states[i] + hidden_states[j] + [dx / dist, dy / dist, dz / dist]

                    # 消息网络
                    msg = self._linear_layer(
                        msg_input,
                        self.model_weights[f'{prefix}_msg_weight1'],
                        self.model_weights[f'{prefix}_msg_bias1']
                    )
                    msg = self._vec_silu(msg)
                    msg = self._linear_layer(
                        msg,
                        self.model_weights[f'{prefix}_msg_weight2'],
                        self.model_weights[f'{prefix}_msg_bias2']
                    )

                    # 应用球谐滤波权重
                    weight = 1.0 / (1.0 + math.exp(-sh_filter))
                    messages.append([m * weight for m in msg])

                # 聚合消息（平均）
                num_neighbors = len(messages)
                aggregated = [0.0] * self.hidden_dim
                for msg in messages:
                    for k in range(self.hidden_dim):
                        aggregated[k] += msg[k]
                aggregated = [a / num_neighbors for a in aggregated]

                # 自环
                self_contrib = self._linear_layer(
                    hidden_states[i],
                    self.model_weights[f'{prefix}_self_weight'],
                    self.model_weights[f'{prefix}_self_bias']
                )

                # 残差连接 + 归一化
                combined = [aggregated[k] + self_contrib[k] + hidden_states[i][k] for k in range(self.hidden_dim)]
                new_states[i] = self._vec_silu(self._layer_norm(combined))

            hidden_states = new_states

        # 全局平均池化
        global_feature = [0.0] * self.hidden_dim
        for h in hidden_states:
            for k in range(self.hidden_dim):
                global_feature[k] += h[k]
        global_feature = [g / num_atoms for g in global_feature]

        # 输出头
        output = self._linear_layer(
            global_feature,
            self.model_weights['output_weight1'],
            self.model_weights['output_bias1']
        )
        output = self._vec_silu(output)
        output = self._linear_layer(
            output,
            self.model_weights['output_weight2'],
            self.model_weights['output_bias2']
        )

        return output

    def train(
        self,
        training_data: List[Dict[str, Any]],
        num_epochs: int = 100,
        batch_size: int = 16,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        训练材料属性预测模型

        使用训练数据集拟合模型权重，支持小批量训练。

        Args:
            training_data: 训练数据列表，每个元素包含 'structure' 和 'properties'
            num_epochs: 训练轮数
            batch_size: 批量大小
            verbose: 是否打印训练日志

        Returns:
            训练结果字典，包含损失历史和最终指标
        """
        self._training_data = training_data
        self.training_history = []

        # 计算归一化统计量
        self._compute_normalization(training_data)

        if verbose:
            print(f"[MaterialsPropertyPredictor] 开始训练")
            print(f"  训练样本数: {len(training_data)}")
            print(f"  属性数量: {self.num_properties}")
            print(f"  训练轮数: {num_epochs}")
            print(f"  批量大小: {batch_size}")

        for epoch in range(num_epochs):
            # 随机打乱数据
            indices = list(range(len(training_data)))
            random.shuffle(indices)

            epoch_loss = 0.0
            num_batches = 0

            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_loss = 0.0

                for idx in batch_indices:
                    sample = training_data[idx]
                    graph = sample['structure']
                    target_props = sample['properties']

                    # 前向传播
                    predictions = self._forward_pass(graph)

                    # 计算损失（均方误差）
                    for k in range(self.num_properties):
                        if k < len(target_props):
                            normalized_pred = self._normalize_value(predictions[k], k)
                            normalized_target = self._normalize_value(target_props[k], k)
                            diff = normalized_pred - normalized_target
                            batch_loss += diff * diff

                batch_loss /= len(batch_indices)
                epoch_loss += batch_loss
                num_batches += 1

                # 简化的梯度更新（模拟训练过程）
                self._update_weights(batch_loss)

            avg_loss = epoch_loss / max(num_batches, 1)
            self.training_history.append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  轮次 {epoch + 1}/{num_epochs}, 损失: {avg_loss:.6f}")

        self.trained = True

        if verbose:
            print(f"[MaterialsPropertyPredictor] 训练完成, 最终损失: {self.training_history[-1]:.6f}")

        return {
            'final_loss': self.training_history[-1],
            'loss_history': self.training_history,
            'num_epochs': num_epochs,
            'num_samples': len(training_data)
        }

    def _compute_normalization(self, training_data: List[Dict[str, Any]]) -> None:
        """
        计算属性的归一化统计量

        Args:
            training_data: 训练数据
        """
        for k in range(self.num_properties):
            values = []
            for sample in training_data:
                if k < len(sample['properties']):
                    values.append(sample['properties'][k])

            if values:
                self._normalization_stats[k] = {
                    'mean': sum(values) / len(values),
                    'std': math.sqrt(sum((v - sum(values) / len(values)) ** 2 for v in values) / len(values)) + 1e-8
                }
            else:
                self._normalization_stats[k] = {'mean': 0.0, 'std': 1.0}

    def _normalize_value(self, value: float, property_idx: int) -> float:
        """归一化属性值"""
        if property_idx in self._normalization_stats:
            stats = self._normalization_stats[property_idx]
            return (value - stats['mean']) / stats['std']
        return value

    def _update_weights(self, loss: float) -> None:
        """
        简化的权重更新（模拟SGD优化）

        Args:
            loss: 当前损失值
        """
        lr = self.learning_rate
        grad_scale = math.tanh(loss) * lr * 0.01

        for key in self.model_weights:
            if isinstance(self.model_weights[key], list):
                if isinstance(self.model_weights[key][0], list):
                    # 二维权重矩阵
                    for i in range(len(self.model_weights[key])):
                        for j in range(len(self.model_weights[key][i])):
                            self.model_weights[key][i][j] -= grad_scale * random.gauss(0, 1)
                else:
                    # 一维偏置向量
                    for i in range(len(self.model_weights[key])):
                        self.model_weights[key][i] -= grad_scale * random.gauss(0, 1) * 0.1

    def predict(
        self,
        atomic_numbers: List[int],
        positions: List[List[float]],
        lattice_vectors: Optional[List[List[float]]] = None
    ) -> Dict[str, float]:
        """
        预测材料属性

        给定晶体结构，预测各项材料属性。

        Args:
            atomic_numbers: 原子序数列表
            positions: 原子位置列表 [[x, y, z], ...]
            lattice_vectors: 晶格向量（可选）

        Returns:
            属性名称到预测值的字典
        """
        # 加载晶体结构
        graph = self.load_crystal_structure(atomic_numbers, positions, lattice_vectors)

        # 前向传播
        raw_predictions = self._forward_pass(graph)

        # 构建结果字典
        results = {}
        for k, name in enumerate(self.property_names):
            if k < len(raw_predictions):
                # 反归一化
                value = raw_predictions[k]
                if k in self._normalization_stats:
                    stats = self._normalization_stats[k]
                    value = value * stats['std'] + stats['mean']

                # 确保物理合理性
                value = self._constrain_property(name, value)
                results[name] = round(value, 4)

        return results

    def _constrain_property(self, name: str, value: float) -> float:
        """
        约束属性值在物理合理范围内

        Args:
            name: 属性名称
            value: 原始预测值

        Returns:
            约束后的值
        """
        constraints = {
            'band_gap': (0.0, 15.0),           # eV
            'hardness': (0.0, 100.0),           # GPa
            'thermal_conductivity': (0.0, 2000.0),  # W/m·K
            'bulk_modulus': (0.0, 500.0),       # GPa
            'shear_modulus': (0.0, 300.0),      # GPa
            'density': (0.1, 25.0),             # g/cm³
            'melting_point': (50.0, 4000.0),    # K
            'dielectric_constant': (1.0, 1000.0),
        }

        if name in constraints:
            low, high = constraints[name]
            return max(low, min(high, value))

        return value

    def predict_batch(
        self,
        structures: List[Dict[str, Any]]
    ) -> List[Dict[str, float]]:
        """
        批量预测材料属性

        Args:
            structures: 结构列表，每个包含 'atomic_numbers', 'positions', 'lattice_vectors'

        Returns:
            预测结果列表
        """
        results = []
        for struct in structures:
            prediction = self.predict(
                struct['atomic_numbers'],
                struct['positions'],
                struct.get('lattice_vectors')
            )
            results.append(prediction)
        return results

    def generate_training_data(
        self,
        num_samples: int = 100,
        max_atoms: int = 10
    ) -> List[Dict[str, Any]]:
        """
        生成合成训练数据（用于测试和演示）

        Args:
            num_samples: 生成样本数
            max_atoms: 每个结构最大原子数

        Returns:
            合成训练数据列表
        """
        data = []
        available_elements = list(range(1, 37))

        for _ in range(num_samples):
            num_atoms = random.randint(2, max_atoms)
            atomic_numbers = [random.choice(available_elements) for _ in range(num_atoms)]

            # 生成随机晶体位置
            positions = []
            for _ in range(num_atoms):
                positions.append([
                    random.uniform(0, 5),
                    random.uniform(0, 5),
                    random.uniform(0, 5)
                ])

            # 生成晶格向量
            lattice_vectors = [
                [random.uniform(3, 8), 0, 0],
                [0, random.uniform(3, 8), 0],
                [0, 0, random.uniform(3, 8)]
            ]

            # 加载结构
            graph = self.load_crystal_structure(atomic_numbers, positions, lattice_vectors)

            # 生成合成属性值（基于启发式规则）
            avg_electronegativity = sum(
                self.ELEMENT_PROPERTIES.get(z, {}).get('electronegativity', 1.5)
                for z in atomic_numbers
            ) / num_atoms

            avg_mass = sum(
                self.ELEMENT_PROPERTIES.get(z, {}).get('mass', 50.0)
                for z in atomic_numbers
            ) / num_atoms

            properties = [
                max(0.1, avg_electronegativity * 2.5 + random.gauss(0, 0.3)),   # band_gap
                max(0.5, avg_electronegativity * 5.0 + random.gauss(0, 1.0)),   # hardness
                max(1.0, 20.0 / avg_mass * 100 + random.gauss(0, 5.0)),         # thermal_conductivity
                max(10.0, avg_electronegativity * 30 + random.gauss(0, 5.0)),   # bulk_modulus
                max(5.0, avg_electronegativity * 20 + random.gauss(0, 3.0)),    # shear_modulus
                max(1.0, avg_mass * 0.05 + random.gauss(0, 0.5)),               # density
                max(300, avg_electronegativity * 400 + random.gauss(0, 100)),    # melting_point
                max(2.0, avg_electronegativity * 3.0 + random.gauss(0, 0.5)),   # dielectric_constant
            ]

            data.append({
                'structure': graph,
                'properties': properties[:self.num_properties]
            })

        return data

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            模型配置和状态信息
        """
        total_params = sum(
            len(w) if isinstance(w, list) and not isinstance(w[0], list) else
            sum(len(row) for row in w) if isinstance(w, list) else 0
            for w in self.model_weights.values()
        )

        return {
            'model_name': 'MaterialsPropertyPredictor',
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_properties': self.num_properties,
            'property_names': self.property_names,
            'l_max': self.l_max,
            'num_layers': self.num_layers,
            'total_parameters': total_params,
            'trained': self.trained,
            'training_epochs': len(self.training_history),
        }

    def save_model(self, filepath: str) -> None:
        """
        保存模型到文件

        Args:
            filepath: 保存路径
        """
        model_data = {
            'model_weights': self.model_weights,
            'config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'num_properties': self.num_properties,
                'l_max': self.l_max,
                'num_layers': self.num_layers,
                'learning_rate': self.learning_rate,
                'property_names': self.property_names,
            },
            'normalization_stats': self._normalization_stats,
            'trained': self.trained,
            'training_history': self.training_history,
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2, ensure_ascii=False)

    def load_model(self, filepath: str) -> None:
        """
        从文件加载模型

        Args:
            filepath: 模型文件路径
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            model_data = json.load(f)

        self.model_weights = model_data['model_weights']
        config = model_data['config']
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_properties = config['num_properties']
        self.l_max = config['l_max']
        self.num_layers = config['num_layers']
        self.learning_rate = config['learning_rate']
        self.property_names = config['property_names']
        self._normalization_stats = model_data.get('normalization_stats', {})
        self.trained = model_data.get('trained', True)
        self.training_history = model_data.get('training_history', [])
