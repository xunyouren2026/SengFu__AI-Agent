#!/usr/bin/env python3
"""
联邦学习MNIST示例
==================

使用AGI统一框架实现的联邦学习示例。
模拟多个客户端的非独立同分布数据，使用FedAvg算法进行训练。

作者: AGI Framework Team
日期: 2025-05-13
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import copy
import time

# 导入框架模块
from core.swing_layer.convolution.convolution_layers import Conv2D, MaxPool2D
from core.swing_layer.neural_layers import Linear, Dropout, Flatten
from core.activations.activations import ReLU, Softmax
from core.initialization.initializers import HeInitializer, XavierInitializer
from core.normalization.normalizations import BatchNormalization2D
from training.optimizers.optimizers import SGD, Adam
from training.losses.losses import CrossEntropyLoss
from evaluation.metrics.metrics import Accuracy
from federated.core.federated_learning import FederatedLearning


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class FederatedConfig:
    """联邦学习配置"""
    # 联邦学习参数
    num_clients: int = 10  # 客户端数量
    num_rounds: int = 50  # 联邦轮数
    client_fraction: float = 1.0  # 每轮参与的客户端比例
    local_epochs: int = 5  # 每轮本地训练epoch数
    local_batch_size: int = 32  # 本地批次大小
    
    # 数据分布参数
    iid: bool = False  # 是否独立同分布
    alpha: float = 0.5  # Dirichlet分布参数（越小越非IID）
    
    # 模型参数
    model_type: str = 'cnn'  # 'cnn' 或 'mlp'
    
    # 优化参数
    global_lr: float = 1.0  # 全局学习率（用于FedAvg聚合）
    local_lr: float = 0.01  # 本地学习率
    local_momentum: float = 0.9
    
    # 评估参数
    eval_interval: int = 5  # 评估间隔
    
    # 其他
    seed: int = 42
    verbose: bool = True


# =============================================================================
# 模拟MNIST数据集（非IID分布）
# =============================================================================

class FederatedMNIST:
    """
    联邦学习MNIST数据集
    支持非IID数据分布
    """
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        np.random.seed(config.seed)
        
        # 生成全局数据
        self.global_data, self.global_labels = self._generate_global_data()
        
        # 为每个客户端分配数据
        self.client_data = self._distribute_to_clients()
        
    def _generate_global_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """生成全局MNIST数据"""
        num_samples = 60000
        num_classes = 10
        
        # 生成平衡的全局数据
        samples_per_class = num_samples // num_classes
        
        all_data = []
        all_labels = []
        
        for c in range(num_classes):
            # 为每个类别生成样本
            class_samples = self._generate_class_samples(c, samples_per_class)
            all_data.append(class_samples)
            all_labels.extend([c] * samples_per_class)
        
        data = np.concatenate(all_data, axis=0)
        labels = np.array(all_labels)
        
        # 打乱
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        
        return data[indices], labels[indices]
    
    def _generate_class_samples(self, digit: int, num_samples: int) -> np.ndarray:
        """生成特定数字的样本"""
        samples = []
        size = 28
        
        for _ in range(num_samples):
            # 创建基础模式
            pattern = self._create_digit_pattern(digit, size)
            
            # 添加噪声和变化
            pattern = self._add_variation(pattern)
            pattern = self._add_noise(pattern)
            
            samples.append(pattern)
        
        return np.array(samples)[:, np.newaxis, :, :].astype(np.float32)
    
    def _create_digit_pattern(self, digit: int, size: int) -> np.ndarray:
        """创建数字的基础模式"""
        pattern = np.zeros((size, size))
        center = size // 2
        
        if digit == 0:
            y, x = np.ogrid[:size, :size]
            mask = ((x - center)**2 + (y - center)**2 <= (size//3)**2) & \
                   ((x - center)**2 + (y - center)**2 >= (size//5)**2)
            pattern[mask] = 1.0
        elif digit == 1:
            pattern[:, center-2:center+2] = 1.0
        elif digit == 2:
            for i in range(size):
                x_pos = int(center + (i - center) * 0.5)
                if 0 <= x_pos < size:
                    pattern[i, max(0, x_pos-2):min(size, x_pos+2)] = 1.0
            pattern[size//3, :] = 1.0
            pattern[2*size//3, :] = 1.0
        elif digit == 3:
            for i in range(size):
                x_pos = int(center + abs(i - center) * 0.3)
                if 0 <= x_pos < size:
                    pattern[i, max(0, x_pos-2):min(size, x_pos+2)] = 1.0
        elif digit == 4:
            pattern[size//2:, center-2:center+2] = 1.0
            pattern[size//2, :] = 1.0
        elif digit == 5:
            pattern[size//5:2*size//5, :] = 1.0
            pattern[2*size//5:3*size//5, center:] = 1.0
            pattern[3*size//5:4*size//5, :] = 1.0
        elif digit == 6:
            pattern[size//3:, center-3:center+3] = 1.0
            y, x = np.ogrid[size//2:size, :size]
            mask = ((x - center)**2 + (y - 3*size//4)**2 <= (size//5)**2)
            pattern[size//2:size, :][mask] = 1.0
        elif digit == 7:
            for i in range(size):
                x_pos = int(size - 1 - i * 0.8)
                if 0 <= x_pos < size:
                    pattern[i, max(0, x_pos-2):min(size, x_pos+2)] = 1.0
            pattern[2, :] = 1.0
        elif digit == 8:
            y, x = np.ogrid[:size, :size]
            mask1 = ((x - center)**2 + (y - size//3)**2 <= (size//6)**2)
            mask2 = ((x - center)**2 + (y - 2*size//3)**2 <= (size//6)**2)
            pattern[mask1 | mask2] = 1.0
        elif digit == 9:
            pattern[:2*size//3, center-3:center+3] = 1.0
            y, x = np.ogrid[:size//2+3, :size]
            mask = ((x - center)**2 + (y - size//3)**2 <= (size//5)**2)
            pattern[:size//2+3, :][mask] = 1.0
        
        return pattern
    
    def _add_variation(self, pattern: np.ndarray) -> np.ndarray:
        """添加变化"""
        # 随机旋转（简化）
        if np.random.rand() > 0.5:
            shift = np.random.randint(-2, 3)
            pattern = np.roll(pattern, shift, axis=1)
        
        # 随机缩放
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            # 简化处理
        
        return pattern
    
    def _add_noise(self, pattern: np.ndarray) -> np.ndarray:
        """添加噪声"""
        noise = np.random.normal(0, 0.1, pattern.shape)
        return np.clip(pattern + noise, 0, 1)
    
    def _distribute_to_clients(self) -> Dict[int, Dict]:
        """将数据分配给客户端"""
        client_data = {}
        num_classes = 10
        
        if self.config.iid:
            # IID分布：每个客户端获得均匀分布的数据
            samples_per_client = len(self.global_labels) // self.config.num_clients
            
            for client_id in range(self.config.num_clients):
                start_idx = client_id * samples_per_client
                end_idx = start_idx + samples_per_client
                
                client_data[client_id] = {
                    'data': self.global_data[start_idx:end_idx],
                    'labels': self.global_labels[start_idx:end_idx]
                }
        else:
            # 非IID分布：使用Dirichlet分布
            # 为每个类别分配样本到客户端
            class_indices = defaultdict(list)
            for idx, label in enumerate(self.global_labels):
                class_indices[label].append(idx)
            
            # 每个客户端的数据索引
            client_indices = [[] for _ in range(self.config.num_clients)]
            
            for c in range(num_classes):
                indices = class_indices[c]
                np.random.shuffle(indices)
                
                # 使用Dirichlet分布分配
                proportions = np.random.dirichlet(
                    [self.config.alpha] * self.config.num_clients
                )
                proportions = proportions / proportions.sum()
                splits = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
                
                split_indices = np.split(indices, splits)
                for client_id, split in enumerate(split_indices):
                    client_indices[client_id].extend(split.tolist())
            
            # 为每个客户端创建数据
            for client_id in range(self.config.num_clients):
                indices = client_indices[client_id]
                np.random.shuffle(indices)
                
                client_data[client_id] = {
                    'data': self.global_data[indices],
                    'labels': self.global_labels[indices]
                }
        
        return client_data
    
    def get_client_data(self, client_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """获取特定客户端的数据"""
        return (self.client_data[client_id]['data'],
                self.client_data[client_id]['labels'])
    
    def get_test_data(self, num_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """获取测试数据"""
        # 生成新的测试数据
        test_data = []
        test_labels = []
        
        samples_per_class = num_samples // 10
        for c in range(10):
            class_samples = self._generate_class_samples(c, samples_per_class)
            test_data.append(class_samples)
            test_labels.extend([c] * samples_per_class)
        
        test_data = np.concatenate(test_data, axis=0)
        test_labels = np.array(test_labels)
        
        indices = np.arange(len(test_labels))
        np.random.shuffle(indices)
        
        return test_data[indices], test_labels[indices]
    
    def get_client_class_distribution(self) -> Dict[int, Dict[int, int]]:
        """获取每个客户端的类别分布"""
        distributions = {}
        
        for client_id, data in self.client_data.items():
            labels = data['labels']
            unique, counts = np.unique(labels, return_counts=True)
            distributions[client_id] = dict(zip(unique, counts))
        
        return distributions


# =============================================================================
# CNN模型
# =============================================================================

class FederatedCNN:
    """联邦学习CNN模型"""
    
    def __init__(self):
        self.layers = []
        self._build_model()
        
    def _build_model(self):
        """构建CNN"""
        # 卷积块1
        self.layers.append(Conv2D(1, 32, kernel_size=3, padding=1, initializer=HeInitializer()))
        self.layers.append(BatchNormalization2D(32))
        self.layers.append(ReLU())
        self.layers.append(MaxPool2D(kernel_size=2, stride=2))
        
        # 卷积块2
        self.layers.append(Conv2D(32, 64, kernel_size=3, padding=1, initializer=HeInitializer()))
        self.layers.append(BatchNormalization2D(64))
        self.layers.append(ReLU())
        self.layers.append(MaxPool2D(kernel_size=2, stride=2))
        
        # 展平
        self.layers.append(Flatten())
        
        # 全连接
        self.layers.append(Linear(64 * 7 * 7, 128, initializer=HeInitializer()))
        self.layers.append(ReLU())
        self.layers.append(Dropout(0.5))
        
        # 输出
        self.layers.append(Linear(128, 10, initializer=XavierInitializer()))
        self.layers.append(Softmax())
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """前向传播"""
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = training
            x = layer.forward(x)
        return x
    
    def get_parameters(self) -> List[np.ndarray]:
        """获取参数"""
        params = []
        for layer in self.layers:
            if hasattr(layer, 'get_parameters'):
                params.extend(layer.get_parameters())
        return params
    
    def set_parameters(self, params: List[np.ndarray]):
        """设置参数"""
        idx = 0
        for layer in self.layers:
            if hasattr(layer, 'get_parameters'):
                layer_params = layer.get_parameters()
                for i in range(len(layer_params)):
                    layer_params[i][:] = params[idx]
                    idx += 1
    
    def copy(self) -> 'FederatedCNN':
        """复制模型"""
        new_model = FederatedCNN()
        new_model.set_parameters(self.get_parameters())
        return new_model


# =============================================================================
# 联邦学习客户端
# =============================================================================

class FederatedClient:
    """联邦学习客户端"""
    
    def __init__(self, client_id: int, config: FederatedConfig):
        self.client_id = client_id
        self.config = config
        self.model = FederatedCNN()
        self.optimizer = SGD(lr=config.local_lr, momentum=config.local_momentum)
        self.criterion = CrossEntropyLoss()
        
        # 训练统计
        self.local_losses = []
        self.local_accuracies = []
        
    def set_model(self, model: FederatedCNN):
        """设置全局模型"""
        self.model.set_parameters(model.get_parameters())
    
    def get_model(self) -> FederatedCNN:
        """获取本地模型"""
        return self.model
    
    def train(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """本地训练"""
        num_samples = len(labels)
        batch_size = self.config.local_batch_size
        num_batches = num_samples // batch_size
        
        epoch_losses = []
        epoch_accs = []
        
        for epoch in range(self.config.local_epochs):
            # 打乱数据
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            
            batch_losses = []
            batch_accs = []
            
            for batch_idx in range(num_batches):
                batch_indices = indices[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                batch_data = data[batch_indices]
                batch_labels = labels[batch_indices]
                
                # 前向传播
                outputs = self.model.forward(batch_data, training=True)
                loss = self.criterion.forward(outputs, batch_labels)
                
                # 计算准确率
                preds = np.argmax(outputs, axis=1)
                acc = np.mean(preds == batch_labels)
                
                batch_losses.append(loss)
                batch_accs.append(acc)
                
                # 反向传播（简化版）
                # 实际实现中需要计算梯度并更新参数
            
            epoch_losses.append(np.mean(batch_losses))
            epoch_accs.append(np.mean(batch_accs))
        
        # 记录
        avg_loss = np.mean(epoch_losses)
        avg_acc = np.mean(epoch_accs)
        
        self.local_losses.append(avg_loss)
        self.local_accuracies.append(avg_acc)
        
        return {
            'loss': avg_loss,
            'accuracy': avg_acc,
            'num_samples': num_samples
        }
    
    def evaluate(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """评估模型"""
        outputs = self.model.forward(data, training=False)
        loss = self.criterion.forward(outputs, labels)
        preds = np.argmax(outputs, axis=1)
        acc = np.mean(preds == labels)
        
        return {'loss': loss, 'accuracy': acc}


# =============================================================================
# FedAvg服务器
# =============================================================================

class FedAvgServer:
    """
    FedAvg联邦学习服务器
    协调多个客户端的训练并聚合模型
    """
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.global_model = FederatedCNN()
        self.clients: List[FederatedClient] = []
        
        # 训练历史
        self.global_accuracies = []
        self.global_losses = []
        self.round_stats = []
        
    def register_clients(self, num_clients: int):
        """注册客户端"""
        for i in range(num_clients):
            client = FederatedClient(i, self.config)
            self.clients.append(client)
        
        print(f"已注册 {num_clients} 个客户端")
    
    def select_clients(self, round_idx: int) -> List[FederatedClient]:
        """选择参与本轮训练的客户端"""
        num_select = max(1, int(self.config.num_clients * self.config.client_fraction))
        
        # 随机选择
        selected_indices = np.random.choice(
            len(self.clients),
            size=num_select,
            replace=False
        )
        
        return [self.clients[i] for i in selected_indices]
    
    def aggregate(self, client_models: List[FederatedCNN], 
                  client_weights: List[int]) -> List[np.ndarray]:
        """
        FedAvg聚合
        按样本数量加权平均模型参数
        """
        total_samples = sum(client_weights)
        
        # 获取所有模型的参数
        all_params = [model.get_parameters() for model in client_models]
        
        # 加权平均
        aggregated_params = []
        for param_idx in range(len(all_params[0])):
            weighted_sum = np.zeros_like(all_params[0][param_idx])
            
            for client_idx, params in enumerate(all_params):
                weight = client_weights[client_idx] / total_samples
                weighted_sum += weight * params[param_idx]
            
            aggregated_params.append(weighted_sum)
        
        return aggregated_params
    
    def train_round(self, dataset: FederatedMNIST, round_idx: int) -> Dict[str, float]:
        """执行一轮联邦训练"""
        # 选择客户端
        selected_clients = self.select_clients(round_idx)
        
        # 分发全局模型
        for client in selected_clients:
            client.set_model(self.global_model)
        
        # 本地训练
        client_models = []
        client_weights = []
        client_stats = []
        
        for client in selected_clients:
            # 获取客户端数据
            data, labels = dataset.get_client_data(client.client_id)
            
            # 本地训练
            stats = client.train(data, labels)
            
            # 收集模型和权重
            client_models.append(client.get_model())
            client_weights.append(stats['num_samples'])
            client_stats.append(stats)
        
        # 聚合模型
        aggregated_params = self.aggregate(client_models, client_weights)
        self.global_model.set_parameters(aggregated_params)
        
        # 计算平均统计
        avg_loss = np.mean([s['loss'] for s in client_stats])
        avg_acc = np.mean([s['accuracy'] for s in client_stats])
        
        return {
            'loss': avg_loss,
            'accuracy': avg_acc,
            'num_clients': len(selected_clients)
        }
    
    def evaluate_global(self, test_data: np.ndarray, test_labels: np.ndarray) -> Dict[str, float]:
        """评估全局模型"""
        outputs = self.global_model.forward(test_data, training=False)
        criterion = CrossEntropyLoss()
        loss = criterion.forward(outputs, test_labels)
        preds = np.argmax(outputs, axis=1)
        acc = np.mean(preds == test_labels)
        
        return {'loss': loss, 'accuracy': acc}


# =============================================================================
# 联邦学习训练器
# =============================================================================

class FederatedTrainer:
    """联邦学习训练器"""
    
    def __init__(self, server: FedAvgServer, dataset: FederatedMNIST, 
                 config: FederatedConfig):
        self.server = server
        self.dataset = dataset
        self.config = config
        
        # 获取测试数据
        self.test_data, self.test_labels = dataset.get_test_data()
        
    def train(self):
        """完整联邦训练流程"""
        print("=" * 60)
        print("开始联邦学习训练")
        print("=" * 60)
        print(f"配置: {self.config.num_clients} 客户端, "
              f"{self.config.num_rounds} 轮, "
              f"IID={self.config.iid}")
        print("-" * 60)
        
        # 注册客户端
        self.server.register_clients(self.config.num_clients)
        
        # 打印数据分布
        self._print_data_distribution()
        
        # 训练循环
        for round_idx in range(self.config.num_rounds):
            start_time = time.time()
            
            # 执行一轮训练
            round_stats = self.server.train_round(self.dataset, round_idx)
            
            # 评估
            if (round_idx + 1) % self.config.eval_interval == 0:
                eval_stats = self.server.evaluate_global(self.test_data, self.test_labels)
                
                self.server.global_accuracies.append(eval_stats['accuracy'])
                self.server.global_losses.append(eval_stats['loss'])
                
                elapsed = time.time() - start_time
                
                print(f"Round {round_idx + 1:2d}/{self.config.num_rounds} | "
                      f"Local Acc: {round_stats['accuracy']:.4f} | "
                      f"Global Acc: {eval_stats['accuracy']:.4f} | "
                      f"Global Loss: {eval_stats['loss']:.4f} | "
                      f"Time: {elapsed:.2f}s")
        
        print("\n" + "=" * 60)
        print("训练完成!")
        print(f"最终全局准确率: {self.server.global_accuracies[-1]:.4f}")
        print("=" * 60)
    
    def _print_data_distribution(self):
        """打印数据分布信息"""
        print("\n客户端数据分布:")
        distributions = self.dataset.get_client_class_distribution()
        
        for client_id in range(min(5, self.config.num_clients)):
            dist = distributions[client_id]
            dist_str = ', '.join([f"{k}:{v}" for k, v in sorted(dist.items())])
            print(f"  Client {client_id}: {dist_str}")
        
        if self.config.num_clients > 5:
            print(f"  ... (还有 {self.config.num_clients - 5} 个客户端)")
        print()


# =============================================================================
# 可视化
# =============================================================================

def plot_federated_results(server: FedAvgServer, config: FederatedConfig,
                           save_path: Optional[str] = None):
    """绘制联邦学习结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    rounds = range(config.eval_interval, 
                   len(server.global_accuracies) * config.eval_interval + 1,
                   config.eval_interval)
    
    # 全局准确率
    ax1 = axes[0, 0]
    ax1.plot(rounds, server.global_accuracies, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Global Model Accuracy Over Rounds')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # 全局损失
    ax2 = axes[0, 1]
    ax2.plot(rounds, server.global_losses, 'r-s', linewidth=2, markersize=6)
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Loss')
    ax2.set_title('Global Model Loss Over Rounds')
    ax2.grid(True, alpha=0.3)
    
    # 客户端本地准确率分布
    ax3 = axes[1, 0]
    client_accs = [client.local_accuracies for client in server.clients]
    if client_accs and client_accs[0]:
        final_accs = [accs[-1] if accs else 0 for accs in client_accs]
        ax3.bar(range(len(final_accs)), final_accs, color='steelblue', edgecolor='black')
        ax3.axhline(np.mean(final_accs), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(final_accs):.3f}')
        ax3.set_xlabel('Client ID')
        ax3.set_ylabel('Final Local Accuracy')
        ax3.set_title('Client Local Accuracies')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 收敛曲线比较
    ax4 = axes[1, 1]
    ax4.plot(rounds, server.global_accuracies, 'b-', linewidth=2, label='Federated')
    # 添加理论上的集中式训练参考线
    ax4.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='Target (0.95)')
    ax4.set_xlabel('Round')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Convergence Curve')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"联邦学习结果图已保存至: {save_path}")
    
    plt.show()


def plot_client_data_distribution(dataset: FederatedMNIST, 
                                  save_path: Optional[str] = None):
    """绘制客户端数据分布热力图"""
    distributions = dataset.get_client_class_distribution()
    
    num_clients = len(distributions)
    num_classes = 10
    
    # 构建分布矩阵
    dist_matrix = np.zeros((num_clients, num_classes))
    for client_id, dist in distributions.items():
        for class_id, count in dist.items():
            dist_matrix[client_id, class_id] = count
    
    # 归一化
    row_sums = dist_matrix.sum(axis=1, keepdims=True)
    dist_matrix = dist_matrix / (row_sums + 1e-10)
    
    # 绘制
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(dist_matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xlabel('Class')
    ax.set_ylabel('Client ID')
    ax.set_title('Client Data Distribution (Normalized)')
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_clients))
    
    # 添加颜色条
    plt.colorbar(im, ax=ax, label='Proportion')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"数据分布图已保存至: {save_path}")
    
    plt.show()


def compare_iid_vs_noniid(save_path: Optional[str] = None):
    """比较IID和非IID设置的收敛性（模拟数据）"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rounds = np.arange(1, 51)
    
    # 模拟IID收敛（更快更好）
    iid_acc = 0.5 + 0.45 * (1 - np.exp(-rounds / 10))
    
    # 模拟非IID收敛（更慢）
    noniid_acc = 0.4 + 0.45 * (1 - np.exp(-rounds / 20))
    
    ax.plot(rounds, iid_acc, 'b-', linewidth=2, label='IID (α=∞)', marker='o', markevery=5)
    ax.plot(rounds, noniid_acc, 'r-', linewidth=2, label='Non-IID (α=0.5)', marker='s', markevery=5)
    
    ax.set_xlabel('Round')
    ax.set_ylabel('Accuracy')
    ax.set_title('IID vs Non-IID Convergence Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"对比图已保存至: {save_path}")
    
    plt.show()


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    # 设置配置
    config = FederatedConfig(
        num_clients=10,
        num_rounds=50,
        client_fraction=1.0,
        local_epochs=5,
        local_batch_size=32,
        iid=False,  # 非IID分布
        alpha=0.5,  # Dirichlet参数
        local_lr=0.01,
        eval_interval=5,
        seed=42
    )
    
    np.random.seed(config.seed)
    
    # 创建数据集
    print("准备联邦学习数据集...")
    dataset = FederatedMNIST(config)
    
    # 创建服务器
    print("初始化FedAvg服务器...")
    server = FedAvgServer(config)
    
    # 创建训练器
    trainer = FederatedTrainer(server, dataset, config)
    
    # 训练
    trainer.train()
    
    # 绘制结果
    plot_federated_results(server, config, 
                          save_path='/workspace/federated_results.png')
    
    # 绘制数据分布
    plot_client_data_distribution(dataset,
                                  save_path='/workspace/federated_data_dist.png')
    
    # 对比图
    compare_iid_vs_noniid(save_path='/workspace/federated_comparison.png')
    
    print("\n示例运行完成!")


if __name__ == '__main__':
    main()
