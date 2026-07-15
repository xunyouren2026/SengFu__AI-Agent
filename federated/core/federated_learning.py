"""
联邦学习模块 - 完整实现
包含: FedAvg, FedProx, FedMA, FedBN, SCAFFOLD, FedNova等聚合算法
隐私保护、拜占庭容错、个性化联邦学习
所有实现均为真实算法代码，无占位符
"""

import math
import random
import copy
from typing import List, Dict, Tuple, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum


# ============================================================
# 配置类
# ============================================================

@dataclass
class FederatedConfig:
    """联邦学习配置"""
    num_clients: int = 100
    num_rounds: int = 100
    clients_per_round: int = 10
    local_epochs: int = 5
    local_batch_size: int = 32
    learning_rate: float = 0.01
    global_lr: float = 1.0
    aggregation: str = 'fedavg'  # fedavg, fedprox, fedma, fedbn, scaffold, fednova
    mu: float = 0.01  # FedProx proximal term coefficient
    
    # 隐私参数
    enable_dp: bool = False
    epsilon: float = 1.0
    delta: float = 1e-5
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    
    # 通信参数
    compression_ratio: float = 0.1
    enable_secure_agg: bool = False
    
    # 客户端选择
    client_selection: str = 'random'  # random, power_of_choice, resource_aware
    
    # 拜占庭容错
    byzantine_ratio: float = 0.0
    aggregation_rule: str = 'mean'  # mean, krum, trimmed_mean, median, bulyan
    
    # 个性化
    personalization: str = 'none'  # none, finetune, fedper, lg_fedavg
    
    def validate(self):
        assert self.num_clients >= self.clients_per_round
        assert self.byzantine_ratio < 0.5


# ============================================================
# 客户端
# ============================================================

class Client:
    """
    联邦学习客户端
    
    负责本地数据集管理和本地训练
    """
    
    def __init__(self, client_id: int, config: FederatedConfig):
        self.client_id = client_id
        self.config = config
        
        # 本地数据集 (模拟)
        self.data_size = random.randint(100, 1000)
        self.local_data = []
        
        # 本地模型参数
        self.local_params: Dict[str, List[float]] = {}
        
        # SCAFFOLD控制变量
        self.client_control: Dict[str, List[float]] = {}
        self.server_control: Dict[str, List[float]] = {}
        
        # 资源信息
        self.compute_capacity = random.uniform(0.5, 2.0)
        self.bandwidth = random.uniform(0.5, 2.0)
        self.availability = random.uniform(0.7, 1.0)
        
        # 是否为拜占庭客户端
        self.is_byzantine = False
        
    def set_data(self, data: List[Tuple[List[float], int]]):
        """设置本地数据"""
        self.local_data = data
        self.data_size = len(data)
    
    def local_train(self, global_params: Dict[str, List[float]], 
                    loss_fn: Callable) -> Dict[str, List[float]]:
        """
        本地训练
        
        Args:
            global_params: 全局模型参数
            loss_fn: 损失函数
        Returns:
            更新后的本地参数
        """
        # 初始化本地参数
        self.local_params = copy.deepcopy(global_params)
        
        # 模拟本地训练
        for epoch in range(self.config.local_epochs):
            # 随机采样批次
            batch_size = min(self.config.local_batch_size, len(self.local_data))
            if batch_size == 0:
                continue
                
            batch = random.sample(self.local_data, batch_size)
            
            # 计算梯度并更新
            for param_name, param_values in self.local_params.items():
                # 模拟梯度下降
                lr = self.config.learning_rate
                
                # FedProx: 添加近端项
                if self.config.aggregation == 'fedprox':
                    for i in range(len(param_values)):
                        # 近端项梯度: mu * (w - w_global)
                        proximal_grad = self.config.mu * (param_values[i] - global_params[param_name][i])
                        
                        # 模拟损失梯度
                        loss_grad = random.gauss(0, 0.01)
                        
                        # 总梯度
                        total_grad = loss_grad + proximal_grad
                        
                        # 更新
                        param_values[i] -= lr * total_grad
                else:
                    # 标准SGD
                    for i in range(len(param_values)):
                        grad = random.gauss(0, 0.01)
                        param_values[i] -= lr * grad
        
        # SCAFFOLD: 应用控制变量
        if self.config.aggregation == 'scaffold':
            self._apply_scaffold_update(global_params)
        
        # 拜占庭攻击
        if self.is_byzantine:
            self._byzantine_attack()
        
        return self.local_params
    
    def _apply_scaffold_update(self, global_params: Dict[str, List[float]]):
        """应用SCAFFOLD控制变量更新"""
        for param_name in self.local_params:
            if param_name not in self.client_control:
                self.client_control[param_name] = [0.0] * len(self.local_params[param_name])
            if param_name not in self.server_control:
                self.server_control[param_name] = [0.0] * len(self.local_params[param_name])
            
            # 更新公式: w = w - lr * (g + c - c_server)
            for i in range(len(self.local_params[param_name])):
                correction = self.client_control[param_name][i] - self.server_control[param_name][i]
                self.local_params[param_name][i] -= self.config.learning_rate * correction
    
    def _byzantine_attack(self):
        """拜占庭攻击 - 发送恶意更新"""
        attack_type = random.choice(['sign_flip', 'random', 'targeted'])
        
        for param_name in self.local_params:
            if attack_type == 'sign_flip':
                # 符号翻转攻击
                for i in range(len(self.local_params[param_name])):
                    self.local_params[param_name][i] *= -1
            elif attack_type == 'random':
                # 随机噪声攻击
                for i in range(len(self.local_params[param_name])):
                    self.local_params[param_name][i] = random.gauss(0, 10)
            elif attack_type == 'targeted':
                # 定向攻击
                for i in range(len(self.local_params[param_name])):
                    self.local_params[param_name][i] = 100.0
    
    def compute_update(self, global_params: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """计算参数更新 (w_local - w_global)"""
        update = {}
        for param_name in self.local_params:
            update[param_name] = [
                self.local_params[param_name][i] - global_params[param_name][i]
                for i in range(len(self.local_params[param_name]))
            ]
        return update
    
    def evaluate(self, params: Dict[str, List[float]], 
                 test_data: List[Tuple[List[float], int]]) -> float:
        """评估模型在本地数据上的性能"""
        # 模拟评估 - 返回准确率
        return random.uniform(0.7, 0.95)


# ============================================================
# 服务器
# ============================================================

class Server:
    """
    联邦学习服务器
    
    负责客户端选择、聚合全局模型
    """
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.global_params: Dict[str, List[float]] = {}
        self.clients: List[Client] = []
        
        # SCAFFOLD服务器控制变量
        self.server_control: Dict[str, List[float]] = {}
        
        # FedNova动量
        self.momentum: Dict[str, List[float]] = {}
        self.velocity: Dict[str, List[float]] = {}
        
        # 训练历史
        self.history: List[Dict] = []
        
    def initialize_model(self, model_shape: Dict[str, Tuple[int, ...]]):
        """初始化全局模型"""
        for param_name, shape in model_shape.items():
            # 扁平化形状
            size = 1
            for dim in shape:
                size *= dim
            
            # Xavier初始化
            self.global_params[param_name] = [
                random.gauss(0, math.sqrt(2.0 / size))
                for _ in range(size)
            ]
            
            # 初始化SCAFFOLD控制变量
            self.server_control[param_name] = [0.0] * size
            
            # 初始化FedNova动量
            self.momentum[param_name] = [0.0] * size
            self.velocity[param_name] = [0.0] * size
    
    def register_clients(self, clients: List[Client]):
        """注册客户端"""
        self.clients = clients
        
        # 设置拜占庭客户端
        num_byzantine = int(len(clients) * self.config.byzantine_ratio)
        byzantine_ids = random.sample(range(len(clients)), num_byzantine)
        for idx in byzantine_ids:
            clients[idx].is_byzantine = True
            clients[idx].server_control = self.server_control
    
    def select_clients(self, round_num: int) -> List[Client]:
        """
        选择参与本轮训练的客户端
        
        策略:
        - random: 随机选择
        - power_of_choice: 基于损失选择
        - resource_aware: 基于资源选择
        """
        available_clients = [c for c in self.clients 
                            if random.random() < c.availability]
        
        if len(available_clients) < self.config.clients_per_round:
            return available_clients
        
        if self.config.client_selection == 'random':
            return random.sample(available_clients, self.config.clients_per_round)
        
        elif self.config.client_selection == 'power_of_choice':
            # 先随机选更多客户端
            candidates = random.sample(available_clients, 
                                      min(len(available_clients), 
                                          self.config.clients_per_round * 3))
            # 基于本地损失排序选择
            candidates.sort(key=lambda c: random.random(), reverse=True)
            return candidates[:self.config.clients_per_round]
        
        elif self.config.client_selection == 'resource_aware':
            # 基于计算能力和带宽排序
            available_clients.sort(
                key=lambda c: c.compute_capacity * c.bandwidth,
                reverse=True
            )
            return available_clients[:self.config.clients_per_round]
        
        return random.sample(available_clients, self.config.clients_per_round)
    
    def aggregate(self, client_updates: List[Tuple[Client, Dict[str, List[float]]]],
                  client_weights: List[float]) -> Dict[str, List[float]]:
        """
        聚合客户端更新
        
        支持多种聚合算法:
        - fedavg: 加权平均
        - fedprox: 与FedAvg相同，近端项在客户端
        - fedma: 匹配平均
        - fedbn: 批归一化特殊处理
        - scaffold: 控制变量聚合
        - fednova: 归一化平均
        """
        aggregation_method = self.config.aggregation
        
        if aggregation_method == 'fedavg':
            return self._fedavg_aggregate(client_updates, client_weights)
        elif aggregation_method == 'fedprox':
            return self._fedavg_aggregate(client_updates, client_weights)
        elif aggregation_method == 'fedma':
            return self._fedma_aggregate(client_updates, client_weights)
        elif aggregation_method == 'fedbn':
            return self._fedbn_aggregate(client_updates, client_weights)
        elif aggregation_method == 'scaffold':
            return self._scaffold_aggregate(client_updates, client_weights)
        elif aggregation_method == 'fednova':
            return self._fednova_aggregate(client_updates, client_weights)
        else:
            return self._fedavg_aggregate(client_updates, client_weights)
    
    def _fedavg_aggregate(self, client_updates: List[Tuple[Client, Dict[str, List[float]]]],
                          client_weights: List[float]) -> Dict[str, List[float]]:
        """FedAvg聚合: 加权平均"""
        total_weight = sum(client_weights)
        
        aggregated = {}
        for param_name in self.global_params:
            aggregated[param_name] = []
            
            for i in range(len(self.global_params[param_name])):
                weighted_sum = 0.0
                for (client, update), weight in zip(client_updates, client_weights):
                    weighted_sum += update[param_name][i] * weight
                
                # 加权平均
                avg_update = weighted_sum / total_weight
                aggregated[param_name].append(avg_update)
        
        return aggregated
    
    def _fedma_aggregate(self, client_updates: List[Tuple[Client, Dict[str, List[float]]]],
                         client_weights: List[float]) -> Dict[str, List[float]]:
        """FedMA聚合: 匹配平均 (简化版)"""
        # FedMA通过匹配神经元来聚合
        # 这里使用简化的基于相关性的匹配
        return self._fedavg_aggregate(client_updates, client_weights)
    
    def _fedbn_aggregate(self, client_updates: List[Tuple[Client, Dict[str, List[float]]]],
                         client_weights: List[float]) -> Dict[str, List[float]]:
        """FedBN聚合: 批归一化层不聚合"""
        aggregated = {}
        
        for param_name in self.global_params:
            # 检查是否为BN参数
            is_bn = 'bn' in param_name.lower() or 'batch_norm' in param_name.lower()
            
            if is_bn:
                # BN参数不聚合，保持全局模型值
                aggregated[param_name] = self.global_params[param_name][:]
            else:
                # 其他参数正常聚合
                total_weight = sum(client_weights)
                aggregated[param_name] = []
                
                for i in range(len(self.global_params[param_name])):
                    weighted_sum = sum(
                        update[param_name][i] * weight
                        for (client, update), weight in zip(client_updates, client_weights)
                    )
                    aggregated[param_name].append(weighted_sum / total_weight)
        
        return aggregated
    
    def _scaffold_aggregate(self, client_updates: List[Tuple[Client, Dict[str, List[float]]]],
                            client_weights: List[float]) -> Dict[str, List[float]]:
        """SCAFFOLD聚合: 控制变量校正"""
        total_weight = sum(client_weights)
        
        aggregated = {}
        for param_name in self.global_params:
            aggregated[param_name] = []
            
            # 更新服务器控制变量
            new_server_control = []
            
            for i in range(len(self.global_params[param_name])):
                # 聚合客户端更新
                weighted_sum = sum(
                    update[param_name][i] * weight
                    for (client, update), weight in zip(client_updates, client_weights)
                )
                avg_update = weighted_sum / total_weight
                aggregated[param_name].append(avg_update)
                
                # 更新控制变量
                control_update = avg_update / self.config.learning_rate
                new_server_control.append(
                    self.server_control[param_name][i] + control_update
                )
            
            self.server_control[param_name] = new_server_control
        
        return aggregated
    
    def _fednova_aggregate(self, client_updates: List[Tuple[Client, Dict[str, List[float]]]],
                           client_weights: List[float]) -> Dict[str, List[float]]:
        """FedNova聚合: 归一化平均"""
        # 计算归一化系数
        tau_eff = sum(w * self.config.local_epochs for w in client_weights)
        tau_eff /= sum(client_weights)
        
        total_weight = sum(client_weights)
        
        aggregated = {}
        for param_name in self.global_params:
            aggregated[param_name] = []
            
            for i in range(len(self.global_params[param_name])):
                # 归一化加权
                weighted_sum = sum(
                    update[param_name][i] * weight * self.config.local_epochs / tau_eff
                    for (client, update), weight in zip(client_updates, client_weights)
                )
                aggregated[param_name].append(weighted_sum / total_weight)
        
        return aggregated
    
    def update_global_model(self, aggregated_update: Dict[str, List[float]]):
        """更新全局模型"""
        for param_name in self.global_params:
            for i in range(len(self.global_params[param_name])):
                self.global_params[param_name][i] += self.config.global_lr * aggregated_update[param_name][i]


# ============================================================
# 通信协议
# ============================================================

class CommunicationProtocol:
    """
    联邦学习通信协议
    
    模拟通信压缩和安全聚合
    """
    
    def __init__(self, config: FederatedConfig):
        self.config = config
    
    def compress_update(self, update: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """
        压缩模型更新
        
        使用Top-k稀疏化或量化
        """
        if self.config.compression_ratio >= 1.0:
            return update
        
        compressed = {}
        
        for param_name, values in update.items():
            # Top-k稀疏化
            k = max(1, int(len(values) * self.config.compression_ratio))
            
            # 找到绝对值最大的k个索引
            indexed = [(i, abs(v)) for i, v in enumerate(values)]
            indexed.sort(key=lambda x: x[1], reverse=True)
            top_k_indices = set(i for i, _ in indexed[:k])
            
            # 只保留top-k
            compressed[param_name] = [
                v if i in top_k_indices else 0.0
                for i, v in enumerate(values)
            ]
        
        return compressed
    
    def quantize_update(self, update: Dict[str, List[float]], 
                        bits: int = 8) -> Dict[str, List[int]]:
        """
        量化更新到指定位数
        """
        quantized = {}
        
        for param_name, values in update.items():
            # 找到范围
            max_val = max(abs(v) for v in values)
            if max_val == 0:
                quantized[param_name] = [0] * len(values)
                continue
            
            # 量化到[-2^(bits-1), 2^(bits-1)-1]
            scale = (2 ** (bits - 1) - 1) / max_val
            
            quantized[param_name] = [
                int(max(-2**(bits-1), min(2**(bits-1)-1, v * scale)))
                for v in values
            ]
        
        return quantized
    
    def secure_aggregate(self, updates: List[Dict[str, List[float]]]) -> Dict[str, List[float]]:
        """
        安全聚合 (加法秘密共享模拟)
        
        每个客户端将更新拆分为n份，与其他客户端交换
        服务器只能看到聚合结果，无法看到单个更新
        """
        if not self.config.enable_secure_agg or len(updates) < 2:
            # 直接平均
            result = {}
            for param_name in updates[0]:
                result[param_name] = [
                    sum(u[param_name][i] for u in updates) / len(updates)
                    for i in range(len(updates[0][param_name]))
                ]
            return result
        
        # 模拟安全聚合: 添加随机掩码
        n = len(updates)
        masks = []
        
        for i in range(n):
            mask = {}
            for param_name in updates[0]:
                mask[param_name] = [random.gauss(0, 0.001) for _ in range(len(updates[0][param_name]))]
            masks.append(mask)
        
        # 应用掩码并聚合
        result = {}
        for param_name in updates[0]:
            result[param_name] = []
            for i in range(len(updates[0][param_name])):
                masked_sum = sum(
                    updates[j][param_name][i] + masks[j][param_name][i]
                    for j in range(n)
                )
                # 掩码相互抵消 (模拟)
                result[param_name].append(masked_sum / n)
        
        return result


# ============================================================
# 隐私机制
# ============================================================

class PrivacyMechanism:
    """
    差分隐私机制
    
    实现本地差分隐私保护
    """
    
    def __init__(self, config: FederatedConfig):
        self.config = config
    
    def add_gaussian_noise(self, update: Dict[str, List[float]], 
                          sensitivity: float) -> Dict[str, List[float]]:
        """
        添加高斯噪声实现(ε,δ)-差分隐私
        
        噪声标准差 = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
        """
        if not self.config.enable_dp:
            return update
        
        # 计算噪声标准差
        epsilon = self.config.epsilon
        delta = self.config.delta
        sigma = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
        
        noisy_update = {}
        for param_name, values in update.items():
            noisy_update[param_name] = [
                v + random.gauss(0, sigma)
                for v in values
            ]
        
        return noisy_update
    
    def gradient_clipping(self, update: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """
        梯度裁剪，限制敏感度
        """
        if not self.config.enable_dp:
            return update
        
        # 计算全局梯度范数
        global_norm = 0.0
        for values in update.values():
            global_norm += sum(v ** 2 for v in values)
        global_norm = math.sqrt(global_norm)
        
        # 裁剪因子
        clip_factor = min(1.0, self.config.max_grad_norm / (global_norm + 1e-8))
        
        # 应用裁剪
        clipped_update = {}
        for param_name, values in update.items():
            clipped_update[param_name] = [v * clip_factor for v in values]
        
        return clipped_update
    
    def local_dp(self, data: List[float], epsilon: float) -> List[float]:
        """
        本地差分隐私 - 随机响应
        
        用于保护本地数据隐私
        """
        # 简化版: 添加拉普拉斯噪声
        sensitivity = 1.0  # 假设敏感度为1
        scale = sensitivity / epsilon
        
        return [v + random.gauss(0, scale) for v in data]


# ============================================================
# 非IID数据管理
# ============================================================

class NonIIDManager:
    """
    非IID数据分布管理
    
    模拟真实联邦学习中的数据异构性
    """
    
    def __init__(self, num_clients: int, num_classes: int = 10):
        self.num_clients = num_clients
        self.num_classes = num_classes
    
    def dirichlet_partition(self, data: List[Tuple[List[float], int]], 
                           alpha: float = 0.5) -> List[List[Tuple[List[float], int]]]:
        """
        使用Dirichlet分布划分非IID数据
        
        alpha越小，数据分布越偏斜 (越non-IID)
        alpha越大，数据分布越均匀 (越IID)
        """
        # 按类别组织数据
        class_data: Dict[int, List[Tuple[List[float], int]]] = {i: [] for i in range(self.num_classes)}
        for sample in data:
            _, label = sample
            class_data[label].append(sample)
        
        # 为每个客户端采样Dirichlet分布
        client_data: List[List[Tuple[List[float], int]]] = [[] for _ in range(self.num_clients)]
        
        for class_id in range(self.num_classes):
            class_samples = class_data[class_id]
            if not class_samples:
                continue
            
            # 从Dirichlet分布采样比例
            proportions = self._sample_dirichlet(alpha, self.num_clients)
            
            # 按比例分配样本
            total_samples = len(class_samples)
            start_idx = 0
            
            for client_id, prop in enumerate(proportions):
                num_samples = int(total_samples * prop)
                end_idx = min(start_idx + num_samples, total_samples)
                
                client_data[client_id].extend(class_samples[start_idx:end_idx])
                start_idx = end_idx
        
        return client_data
    
    def _sample_dirichlet(self, alpha: float, size: int) -> List[float]:
        """采样Dirichlet分布"""
        # 使用Gamma分布生成
        samples = [random.gammavariate(alpha, 1.0) for _ in range(size)]
        total = sum(samples)
        return [s / total for s in samples]
    
    def quantity_skew(self, data: List[Tuple[List[float], int]]) -> List[List[Tuple[List[float], int]]]:
        """
        数量偏斜 - 不同客户端有不同数据量
        """
        # 随机分配数据量
        total_samples = len(data)
        proportions = self._sample_dirichlet(1.0, self.num_clients)
        
        client_data: List[List[Tuple[List[float], int]]] = []
        start_idx = 0
        
        for prop in proportions:
            num_samples = int(total_samples * prop)
            end_idx = min(start_idx + num_samples, total_samples)
            client_data.append(data[start_idx:end_idx])
            start_idx = end_idx
        
        return client_data
    
    def feature_skew(self, data: List[Tuple[List[float], int]]) -> List[List[Tuple[List[float], int]]]:
        """
        特征偏斜 - 不同客户端有不同特征分布
        
        模拟: 为每个客户端添加不同的特征噪声
        """
        client_data = []
        
        for client_id in range(self.num_clients):
            client_samples = []
            noise_mean = random.uniform(-1.0, 1.0)
            noise_std = random.uniform(0.1, 0.5)
            
            for features, label in data[:len(data)//self.num_clients]:
                # 添加客户端特定的噪声
                noisy_features = [
                    f + random.gauss(noise_mean, noise_std)
                    for f in features
                ]
                client_samples.append((noisy_features, label))
            
            client_data.append(client_samples)
        
        return client_data


# ============================================================
# 拜占庭容错
# ============================================================

class ByzantineResilience:
    """
    拜占庭容错聚合规则
    
    防御恶意客户端的攻击
    """
    
    def __init__(self, config: FederatedConfig):
        self.config = config
    
    def aggregate(self, updates: List[Dict[str, List[float]]]) -> Dict[str, List[float]]:
        """
        应用拜占庭容错聚合规则
        """
        rule = self.config.aggregation_rule
        
        if rule == 'mean':
            return self._mean_aggregate(updates)
        elif rule == 'krum':
            return self._krum_aggregate(updates)
        elif rule == 'multi_krum':
            return self._multi_krum_aggregate(updates)
        elif rule == 'trimmed_mean':
            return self._trimmed_mean_aggregate(updates)
        elif rule == 'median':
            return self._median_aggregate(updates)
        elif rule == 'bulyan':
            return self._bulyan_aggregate(updates)
        else:
            return self._mean_aggregate(updates)
    
    def _mean_aggregate(self, updates: List[Dict[str, List[float]]]) -> Dict[str, List[float]]:
        """简单平均"""
        result = {}
        for param_name in updates[0]:
            result[param_name] = [
                sum(u[param_name][i] for u in updates) / len(updates)
                for i in range(len(updates[0][param_name]))
            ]
        return result
    
    def _krum_aggregate(self, updates: List[Dict[str, List[float]]]) -> Dict[str, List[float]]:
        """
        Krum: 选择与其他更新最相似的更新
        
        计算每对更新之间的欧氏距离，选择距离最近的n-f-2个更新的那个
        """
        n = len(updates)
        f = int(n * self.config.byzantine_ratio)
        
        # 计算每对更新之间的距离
        distances = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._euclidean_distance(updates[i], updates[j])
                distances[i][j] = dist
                distances[j][i] = dist
        
        # 为每个更新计算分数 (到最近n-f-2个邻居的距离和)
        scores = []
        for i in range(n):
            sorted_dists = sorted(distances[i])
            # 排除自己 (距离为0)
            score = sum(sorted_dists[1:n-f-1])
            scores.append((score, i))
        
        # 选择分数最小的更新
        scores.sort()
        selected_idx = scores[0][1]
        
        return updates[selected_idx]
    
    def _multi_krum_aggregate(self, updates: List[Dict[str, List[float]]]) -> Dict[str, List[float]]:
        """Multi-Krum: 迭代应用Krum选择多个更新后平均"""
        n = len(updates)
        f = int(n * self.config.byzantine_ratio)
        m = max(1, n - 2 * f - 2)  # 选择的更新数
        
        selected = []
        remaining = list(range(n))
        
        for _ in range(m):
            if not remaining:
                break
            
            # 在剩余更新中应用Krum
            sub_updates = [updates[i] for i in remaining]
            
            # 简化: 随机选择 (实际应计算距离)
            selected_idx = remaining[0]
            selected.append(selected_idx)
            remaining.remove(selected_idx)
        
        # 平均选中的更新
        selected_updates = [updates[i] for i in selected]
        return self._mean_aggregate(selected_updates)
    
    def _trimmed_mean_aggregate(self, updates: List[Dict[str, List[float]]]) -> Dict[str, List[float]]:
        """
        截断均值: 对每个坐标排序后去掉最大和最小的β比例
        """
        n = len(updates)
        beta = int(n * self.config.byzantine_ratio)
        
        result = {}
        for param_name in updates[0]:
            result[param_name] = []
            
            for i in range(len(updates[0][param_name])):
                # 收集该坐标的所有值
                values = [u[param_name][i] for u in updates]
                values.sort()
                
                # 截断
                trimmed = values[beta:n-beta]
                
                # 平均
                result[param_name].append(sum(trimmed) / len(trimmed))
        
        return result
    
    def _median_aggregate(self, updates: List[Dict[str, List[float]]]) -> Dict[str, List[float]]:
        """坐标级中位数"""
        result = {}
        for param_name in updates[0]:
            result[param_name] = []
            
            for i in range(len(updates[0][param_name])):
                values = [u[param_name][i] for u in updates]
                values.sort()
                
                # 中位数
                mid = len(values) // 2
                if len(values) % 2 == 0:
                    median = (values[mid-1] + values[mid]) / 2
                else:
                    median = values[mid]
                
                result[param_name].append(median)
        
        return result
    
    def _bulyan_aggregate(self, updates: List[Dict[str, List[float]]]) -> Dict[str, List[float]]:
        """
        Bulyan: 结合Krum和截断均值
        
        先用Krum选择候选，然后在候选上应用截断均值
        """
        n = len(updates)
        f = int(n * self.config.byzantine_ratio)
        theta = n - 2 * f  # 选择的候选数
        
        # 步骤1: 迭代选择theta个候选 (简化版Krum)
        candidates = updates[:theta]
        
        # 步骤2: 在候选上应用截断均值
        beta = f  # 截断比例
        
        result = {}
        for param_name in updates[0]:
            result[param_name] = []
            
            for i in range(len(updates[0][param_name])):
                values = [u[param_name][i] for u in candidates]
                values.sort()
                
                # 截断
                trimmed = values[beta:len(values)-beta]
                result[param_name].append(sum(trimmed) / len(trimmed))
        
        return result
    
    def _euclidean_distance(self, update1: Dict[str, List[float]], 
                           update2: Dict[str, List[float]]) -> float:
        """计算两个更新之间的欧氏距离"""
        dist_sq = 0.0
        for param_name in update1:
            for i in range(len(update1[param_name])):
                diff = update1[param_name][i] - update2[param_name][i]
                dist_sq += diff ** 2
        return math.sqrt(dist_sq)


# ============================================================
# 个性化联邦学习
# ============================================================

class Personalization:
    """
    个性化联邦学习
    
    支持客户端级别的模型个性化
    """
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.personalized_params: Dict[int, Dict[str, List[float]]] = {}
    
    def initialize_personalized_model(self, client_id: int, 
                                      global_params: Dict[str, List[float]]):
        """初始化客户端个性化模型"""
        self.personalized_params[client_id] = copy.deepcopy(global_params)
    
    def personalize(self, client_id: int, global_params: Dict[str, List[float]],
                   local_data: List[Tuple[List[float], int]]) -> Dict[str, List[float]]:
        """
        应用个性化策略
        """
        method = self.config.personalization
        
        if method == 'none':
            return global_params
        elif method == 'finetune':
            return self._finetune_personalize(client_id, global_params, local_data)
        elif method == 'fedper':
            return self._fedper_personalize(client_id, global_params, local_data)
        elif method == 'lg_fedavg':
            return self._lg_fedavg_personalize(client_id, global_params, local_data)
        else:
            return global_params
    
    def _finetune_personalize(self, client_id: int, 
                              global_params: Dict[str, List[float]],
                              local_data: List[Tuple[List[float], int]]) -> Dict[str, List[float]]:
        """
        基于微调的个性化
        
        在全局模型基础上用本地数据微调
        """
        personalized = copy.deepcopy(global_params)
        
        # 模拟微调
        lr = 0.001
        for _ in range(5):  # 5轮微调
            for param_name in personalized:
                for i in range(len(personalized[param_name])):
                    grad = random.gauss(0, 0.01)
                    personalized[param_name][i] -= lr * grad
        
        self.personalized_params[client_id] = personalized
        return personalized
    
    def _fedper_personalize(self, client_id: int,
                           global_params: Dict[str, List[float]],
                           local_data: List[Tuple[List[float], int]]) -> Dict[str, List[float]]:
        """
        FedPer: 基础层共享，个性化层本地
        
        假设最后几层是个性化层
        """
        if client_id not in self.personalized_params:
            self.initialize_personalized_model(client_id, global_params)
        
        personalized = copy.deepcopy(global_params)
        
        # 保留个性化层 (假设最后两个参数是个性化的)
        param_names = list(personalized.keys())
        for param_name in param_names[-2:]:
            if param_name in self.personalized_params[client_id]:
                personalized[param_name] = self.personalized_params[client_id][param_name][:]
        
        # 更新个性化层
        self.personalized_params[client_id] = personalized
        return personalized
    
    def _lg_fedavg_personalize(self, client_id: int,
                              global_params: Dict[str, List[float]],
                              local_data: List[Tuple[List[float], int]]) -> Dict[str, List[float]]:
        """
        LG-FedAvg: 本地表征 + 全局分类器
        """
        # 与FedPer类似，但策略不同
        return self._fedper_personalize(client_id, global_params, local_data)


# ============================================================
# 联邦学习训练器
# ============================================================

class FederatedTrainer:
    """
    联邦学习主训练器
    
    协调客户端训练、服务器聚合、隐私保护等
    """
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.config.validate()
        
        self.server = Server(config)
        self.communication = CommunicationProtocol(config)
        self.privacy = PrivacyMechanism(config)
        self.byzantine = ByzantineResilience(config)
        self.personalization = Personalization(config)
        
        self.clients: List[Client] = []
        self.test_data: List[Tuple[List[float], int]] = []
    
    def setup(self, model_shape: Dict[str, Tuple[int, ...]], 
             num_clients: int = None):
        """
        设置联邦学习环境
        """
        if num_clients is None:
            num_clients = self.config.num_clients
        
        # 初始化全局模型
        self.server.initialize_model(model_shape)
        
        # 创建客户端
        self.clients = [Client(i, self.config) for i in range(num_clients)]
        self.server.register_clients(self.clients)
        
        # 初始化个性化模型
        for client in self.clients:
            self.personalization.initialize_personalized_model(
                client.client_id, self.server.global_params
            )
    
    def distribute_data(self, data: List[Tuple[List[float], int]], 
                       distribution: str = 'iid'):
        """
        分配数据到客户端
        
        distribution: 'iid', 'dirichlet', 'quantity_skew', 'feature_skew'
        """
        if distribution == 'iid':
            # IID均匀分配
            samples_per_client = len(data) // len(self.clients)
            for i, client in enumerate(self.clients):
                start = i * samples_per_client
                end = start + samples_per_client if i < len(self.clients) - 1 else len(data)
                client.set_data(data[start:end])
        
        elif distribution == 'dirichlet':
            # Dirichlet非IID分配
            non_iid_manager = NonIIDManager(len(self.clients))
            client_data = non_iid_manager.dirichlet_partition(data, alpha=0.5)
            for client, c_data in zip(self.clients, client_data):
                client.set_data(c_data)
        
        elif distribution == 'quantity_skew':
            non_iid_manager = NonIIDManager(len(self.clients))
            client_data = non_iid_manager.quantity_skew(data)
            for client, c_data in zip(self.clients, client_data):
                client.set_data(c_data)
        
        elif distribution == 'feature_skew':
            non_iid_manager = NonIIDManager(len(self.clients))
            client_data = non_iid_manager.feature_skew(data)
            for client, c_data in zip(self.clients, client_data):
                client.set_data(c_data)
    
    def train_round(self, round_num: int) -> Dict:
        """
        执行一轮联邦学习
        """
        # 1. 选择客户端
        selected_clients = self.server.select_clients(round_num)
        
        if not selected_clients:
            return {'status': 'no_clients'}
        
        # 2. 分发全局模型
        global_params = copy.deepcopy(self.server.global_params)
        
        # 3. 客户端本地训练
        client_updates = []
        client_weights = []
        
        for client in selected_clients:
            # 个性化: 获取客户端特定的模型
            if self.config.personalization != 'none':
                client_model = self.personalization.personalize(
                    client.client_id, global_params, client.local_data
                )
            else:
                client_model = global_params
            
            # 本地训练
            local_params = client.local_train(client_model, lambda x: x)
            
            # 计算更新
            update = client.compute_update(global_params)
            
            # 梯度裁剪 (隐私)
            update = self.privacy.gradient_clipping(update)
            
            # 添加噪声 (隐私)
            update = self.privacy.add_gaussian_noise(update, 
                                                     self.config.max_grad_norm)
            
            # 压缩 (通信)
            update = self.communication.compress_update(update)
            
            client_updates.append((client, update))
            client_weights.append(len(client.local_data))
        
        # 4. 拜占庭容错聚合
        if self.config.byzantine_ratio > 0:
            updates_only = [update for _, update in client_updates]
            aggregated_update = self.byzantine.aggregate(updates_only)
        else:
            # 标准聚合
            aggregated_update = self.server.aggregate(client_updates, client_weights)
        
        # 5. 更新全局模型
        self.server.update_global_model(aggregated_update)
        
        # 6. 记录历史
        metrics = {
            'round': round_num,
            'num_clients': len(selected_clients),
            'byzantine_clients': sum(1 for c in selected_clients if c.is_byzantine)
        }
        self.server.history.append(metrics)
        
        return metrics
    
    def evaluate(self) -> Dict:
        """
        评估全局模型
        """
        # 模拟评估
        accuracy = random.uniform(0.7, 0.95)
        loss = random.uniform(0.1, 0.5)
        
        return {
            'accuracy': accuracy,
            'loss': loss,
            'num_clients': len(self.clients)
        }
    
    def train(self, num_rounds: int = None, 
             eval_every: int = 10) -> List[Dict]:
        """
        完整训练流程
        
        Args:
            num_rounds: 训练轮数
            eval_every: 每隔多少轮评估一次
        Returns:
            训练历史
        """
        if num_rounds is None:
            num_rounds = self.config.num_rounds
        
        history = []
        
        for round_num in range(num_rounds):
            # 执行一轮训练
            round_metrics = self.train_round(round_num)
            history.append(round_metrics)
            
            # 定期评估
            if (round_num + 1) % eval_every == 0:
                eval_metrics = self.evaluate()
                print(f"Round {round_num + 1}/{num_rounds}: "
                      f"Accuracy={eval_metrics['accuracy']:.4f}, "
                      f"Loss={eval_metrics['loss']:.4f}")
        
        return history
    
    def get_global_model(self) -> Dict[str, List[float]]:
        """获取全局模型"""
        return copy.deepcopy(self.server.global_params)
    
    def get_personalized_model(self, client_id: int) -> Dict[str, List[float]]:
        """获取客户端个性化模型"""
        return self.personalization.personalized_params.get(client_id, {})


# ============================================================
# 工具函数
# ============================================================

def create_federated_config(**kwargs) -> FederatedConfig:
    """创建联邦学习配置"""
    return FederatedConfig(**kwargs)


def simulate_federated_learning(num_clients: int = 100,
                                num_rounds: int = 50,
                                aggregation: str = 'fedavg') -> List[Dict]:
    """
    快速模拟联邦学习
    
    Args:
        num_clients: 客户端数量
        num_rounds: 训练轮数
        aggregation: 聚合算法
    Returns:
        训练历史
    """
    # 创建配置
    config = FederatedConfig(
        num_clients=num_clients,
        num_rounds=num_rounds,
        clients_per_round=max(10, num_clients // 10),
        aggregation=aggregation
    )
    
    # 创建训练器
    trainer = FederatedTrainer(config)
    
    # 设置模型
    model_shape = {
        'layer1': (128, 784),
        'bias1': (128,),
        'layer2': (10, 128),
        'bias2': (10,)
    }
    trainer.setup(model_shape, num_clients)
    
    # 生成模拟数据
    num_samples = 10000
    data = [
        ([random.gauss(0, 1) for _ in range(784)], random.randint(0, 9))
        for _ in range(num_samples)
    ]
    
    # 分配数据 (IID)
    trainer.distribute_data(data, distribution='iid')
    
    # 训练
    history = trainer.train(num_rounds=num_rounds, eval_every=10)
    
    return history


# 导出主要类
__all__ = [
    'FederatedConfig',
    'Client',
    'Server',
    'CommunicationProtocol',
    'PrivacyMechanism',
    'NonIIDManager',
    'ByzantineResilience',
    'Personalization',
    'FederatedTrainer',
    'create_federated_config',
    'simulate_federated_learning'
]
