#!/usr/bin/env python3
"""
GNN节点分类示例
================

使用AGI统一框架实现的图神经网络节点分类示例。
模拟Cora-like引文网络数据集，使用GCN模型进行分类。

作者: AGI Framework Team
日期: 2025-05-13
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import time

# 导入框架模块
from core.swing_layer.gnn.gnn import GCNLayer, GATLayer
from core.swing_layer.neural_layers import Linear, Dropout
from core.activations.activations import ReLU, Softmax
from core.initialization.initializers import XavierInitializer
from training.optimizers.optimizers import Adam
from training.losses.losses import CrossEntropyLoss
from evaluation.metrics.metrics import Accuracy, F1Score
from core.normalization.normalizations import LayerNormalization


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class GNNConfig:
    """GNN训练配置"""
    # 图参数
    num_nodes: int = 2708  # Cora数据集大小
    num_features: int = 1433  # 特征维度
    num_classes: int = 7  # 类别数
    num_edges: int = 5429  # 边数
    
    # 模型参数
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.5
    use_gat: bool = False  # 使用GAT代替GCN
    num_heads: int = 8  # GAT头数
    
    # 训练参数
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 200
    early_stopping_patience: int = 20
    
    # 数据分割
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    
    # 其他
    seed: int = 42
    verbose: bool = True


# =============================================================================
# 模拟Cora数据集
# =============================================================================

class CitationNetwork:
    """
    模拟Cora引文网络数据集
    包含论文节点、引用边和主题标签
    """
    
    def __init__(self, config: GNNConfig):
        self.config = config
        np.random.seed(config.seed)
        
        # 生成图数据
        self.features = self._generate_features()
        self.adj_matrix = self._generate_graph()
        self.labels = self._generate_labels()
        
        # 数据分割
        self.train_mask, self.val_mask, self.test_mask = self._split_data()
        
        # 计算归一化邻接矩阵
        self.norm_adj = self._normalize_adjacency()
        
    def _generate_features(self) -> np.ndarray:
        """生成节点特征（词袋特征）"""
        # 模拟稀疏的词袋特征
        features = np.zeros((self.config.num_nodes, self.config.num_features))
        
        # 每个节点有随机数量的非零特征
        for i in range(self.config.num_nodes):
            num_words = np.random.randint(10, 50)
            word_indices = np.random.choice(self.config.num_features, num_words, replace=False)
            features[i, word_indices] = 1.0
        
        # 归一化
        row_sums = features.sum(axis=1, keepdims=True)
        features = features / (row_sums + 1e-10)
        
        return features.astype(np.float32)
    
    def _generate_graph(self) -> np.ndarray:
        """生成引文网络图结构"""
        # 创建邻接矩阵
        adj = np.zeros((self.config.num_nodes, self.config.num_nodes))
        
        # 基于主题社区生成边
        nodes_per_class = self.config.num_nodes // self.config.num_classes
        
        edges_created = 0
        
        # 1. 类内连接（同主题论文相互引用）
        for c in range(self.config.num_classes):
            start_idx = c * nodes_per_class
            end_idx = start_idx + nodes_per_class if c < self.config.num_classes - 1 else self.config.num_nodes
            
            class_nodes = list(range(start_idx, end_idx))
            num_class_edges = int(self.config.num_edges * 0.6 / self.config.num_classes)
            
            for _ in range(num_class_edges):
                i, j = np.random.choice(class_nodes, 2, replace=False)
                if adj[i, j] == 0:
                    adj[i, j] = adj[j, i] = 1
                    edges_created += 1
        
        # 2. 类间连接（跨主题引用）
        remaining_edges = self.config.num_edges - edges_created
        for _ in range(remaining_edges):
            i = np.random.randint(0, self.config.num_nodes)
            # 有一定概率连接到同主题，一定概率连接到其他主题
            if np.random.rand() < 0.7:
                # 同主题
                class_id = min(i // nodes_per_class, self.config.num_classes - 1)
                start_idx = class_id * nodes_per_class
                end_idx = start_idx + nodes_per_class if class_id < self.config.num_classes - 1 else self.config.num_nodes
                j = np.random.randint(start_idx, end_idx)
            else:
                # 跨主题
                j = np.random.randint(0, self.config.num_nodes)
            
            if i != j and adj[i, j] == 0:
                adj[i, j] = adj[j, i] = 1
                edges_created += 1
        
        # 添加自环
        adj = adj + np.eye(self.config.num_nodes)
        
        return adj.astype(np.float32)
    
    def _generate_labels(self) -> np.ndarray:
        """生成节点标签（论文主题）"""
        labels = np.zeros(self.config.num_nodes, dtype=np.int64)
        nodes_per_class = self.config.num_nodes // self.config.num_classes
        
        for c in range(self.config.num_classes):
            start_idx = c * nodes_per_class
            end_idx = start_idx + nodes_per_class if c < self.config.num_classes - 1 else self.config.num_nodes
            labels[start_idx:end_idx] = c
        
        # 打乱以增加难度
        np.random.shuffle(labels)
        
        return labels
    
    def _split_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """分割训练/验证/测试集"""
        indices = np.arange(self.config.num_nodes)
        np.random.shuffle(indices)
        
        n_train = int(self.config.num_nodes * self.config.train_ratio)
        n_val = int(self.config.num_nodes * self.config.val_ratio)
        
        train_mask = np.zeros(self.config.num_nodes, dtype=bool)
        val_mask = np.zeros(self.config.num_nodes, dtype=bool)
        test_mask = np.zeros(self.config.num_nodes, dtype=bool)
        
        train_mask[indices[:n_train]] = True
        val_mask[indices[n_train:n_train + n_val]] = True
        test_mask[indices[n_train + n_val:]] = True
        
        print(f"数据分割: 训练集 {n_train}, 验证集 {n_val}, "
              f"测试集 {self.config.num_nodes - n_train - n_val}")
        
        return train_mask, val_mask, test_mask
    
    def _normalize_adjacency(self) -> np.ndarray:
        """
        归一化邻接矩阵: D^{-1/2} A D^{-1/2}
        """
        adj = self.adj_matrix
        
        # 计算度矩阵
        degree = np.sum(adj, axis=1)
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.
        
        # 对称归一化
        D_inv_sqrt = np.diag(degree_inv_sqrt)
        norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt
        
        return norm_adj.astype(np.float32)
    
    def get_class_distribution(self) -> Dict[int, int]:
        """获取类别分布"""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))
    
    def get_graph_statistics(self) -> Dict[str, float]:
        """获取图统计信息"""
        num_edges = (np.sum(self.adj_matrix) - self.config.num_nodes) / 2
        avg_degree = 2 * num_edges / self.config.num_nodes
        
        # 计算聚类系数（简化版）
        clustering_coeffs = []
        for i in range(min(100, self.config.num_nodes)):  # 采样计算
            neighbors = np.where(self.adj_matrix[i] > 0)[0]
            if len(neighbors) > 1:
                possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
                actual_edges = 0
                for j in neighbors:
                    for k in neighbors:
                        if j < k and self.adj_matrix[j, k] > 0:
                            actual_edges += 1
                clustering_coeffs.append(actual_edges / possible_edges if possible_edges > 0 else 0)
        
        return {
            'num_nodes': self.config.num_nodes,
            'num_edges': int(num_edges),
            'avg_degree': avg_degree,
            'avg_clustering': np.mean(clustering_coeffs) if clustering_coeffs else 0,
            'density': num_edges / (self.config.num_nodes * (self.config.num_nodes - 1) / 2)
        }


# =============================================================================
# GCN模型
# =============================================================================

class GCN:
    """
    图卷积网络 (Graph Convolutional Network)
    用于节点分类任务
    """
    
    def __init__(self, config: GNNConfig):
        self.config = config
        self.layers = []
        self._build_model()
        
    def _build_model(self):
        """构建GCN模型"""
        dims = [self.config.num_features] + self.config.hidden_dims + [self.config.num_classes]
        
        for i in range(len(dims) - 1):
            # 使用GCN层或GAT层
            if self.config.use_gat and i < len(dims) - 2:
                # 隐藏层使用GAT
                layer = GATLayer(
                    in_features=dims[i],
                    out_features=dims[i + 1],
                    num_heads=self.config.num_heads,
                    dropout=self.config.dropout,
                    concat=True if i < len(dims) - 2 else False
                )
            else:
                # 使用GCN
                layer = GCNLayer(
                    in_features=dims[i],
                    out_features=dims[i + 1],
                    activation='relu' if i < len(dims) - 2 else None,
                    dropout=self.config.dropout if i < len(dims) - 2 else 0
                )
            
            self.layers.append(layer)
            
            # 在隐藏层后添加归一化（除了最后一层）
            if i < len(dims) - 2:
                self.layers.append(LayerNormalization(dims[i + 1]))
    
    def forward(self, features: np.ndarray, adj: np.ndarray, 
                training: bool = True) -> np.ndarray:
        """
        前向传播
        
        Args:
            features: 节点特征 [num_nodes, num_features]
            adj: 归一化邻接矩阵 [num_nodes, num_nodes]
            training: 是否训练模式
        """
        x = features
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (GCNLayer, GATLayer)):
                x = layer.forward(x, adj, training=training)
            else:
                x = layer.forward(x)
        
        # 应用softmax获取概率
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        x = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        return x
    
    def predict(self, features: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """预测类别"""
        logits = self.forward(features, adj, training=False)
        return np.argmax(logits, axis=1)
    
    def get_parameters(self) -> List[np.ndarray]:
        """获取所有参数"""
        params = []
        for layer in self.layers:
            if hasattr(layer, 'get_parameters'):
                params.extend(layer.get_parameters())
        return params


# =============================================================================
# 训练器
# =============================================================================

class GNNTrainer:
    """GNN训练器"""
    
    def __init__(self, model: GCN, config: GNNConfig):
        self.model = model
        self.config = config
        self.optimizer = Adam(
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.criterion = CrossEntropyLoss()
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.test_accs = []
        
        self.best_val_acc = 0
        self.best_model_params = None
        self.patience_counter = 0
        
    def train_epoch(self, data: CitationNetwork) -> Tuple[float, float]:
        """训练一个epoch"""
        # 前向传播
        logits = self.model.forward(data.features, data.norm_adj, training=True)
        
        # 计算损失（只在训练节点上）
        train_logits = logits[data.train_mask]
        train_labels = data.labels[data.train_mask]
        
        loss = self.criterion.forward(train_logits, train_labels)
        
        # 计算训练准确率
        train_preds = np.argmax(train_logits, axis=1)
        train_acc = np.mean(train_preds == train_labels)
        
        # 反向传播和优化（简化版）
        # 实际实现中需要计算梯度并更新参数
        
        return loss, train_acc
    
    def evaluate(self, data: CitationNetwork, mask: np.ndarray) -> Tuple[float, float]:
        """评估模型"""
        logits = self.model.forward(data.features, data.norm_adj, training=False)
        
        masked_logits = logits[mask]
        masked_labels = data.labels[mask]
        
        loss = self.criterion.forward(masked_logits, masked_labels)
        preds = np.argmax(masked_logits, axis=1)
        acc = np.mean(preds == masked_labels)
        
        return loss, acc
    
    def fit(self, data: CitationNetwork):
        """完整训练流程"""
        print("=" * 60)
        print("开始训练GCN节点分类模型")
        print("=" * 60)
        
        for epoch in range(self.config.epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(data)
            
            # 验证
            val_loss, val_acc = self.evaluate(data, data.val_mask)
            
            # 测试
            test_loss, test_acc = self.evaluate(data, data.test_mask)
            
            # 记录
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            self.test_accs.append(test_acc)
            
            # 早停检查
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_params = [p.copy() for p in self.model.get_parameters()]
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # 打印进度
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1:3d}/{self.config.epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Train Acc: {train_acc:.4f} | "
                      f"Val Acc: {val_acc:.4f} | "
                      f"Test Acc: {test_acc:.4f}")
            
            # 早停
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\n早停于 epoch {epoch + 1}")
                break
        
        # 恢复最佳模型
        if self.best_model_params is not None:
            params = self.model.get_parameters()
            for p, best_p in zip(params, self.best_model_params):
                p[:] = best_p
        
        print("\n" + "=" * 60)
        print("训练完成!")
        print(f"最佳验证准确率: {self.best_val_acc:.4f}")
        print("=" * 60)
    
    def get_detailed_metrics(self, data: CitationNetwork) -> Dict[str, float]:
        """获取详细评估指标"""
        logits = self.model.forward(data.features, data.norm_adj, training=False)
        preds = np.argmax(logits, axis=1)
        
        # 测试集指标
        test_preds = preds[data.test_mask]
        test_labels = data.labels[data.test_mask]
        
        accuracy = Accuracy()(test_labels, test_preds)
        f1_macro = F1Score(num_classes=self.config.num_classes, average='macro')(test_labels, test_preds)
        f1_micro = F1Score(num_classes=self.config.num_classes, average='micro')(test_labels, test_preds)
        
        # 每类准确率
        class_accs = {}
        for c in range(self.config.num_classes):
            class_mask = test_labels == c
            if np.sum(class_mask) > 0:
                class_accs[f'class_{c}'] = np.mean(test_preds[class_mask] == test_labels[class_mask])
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            **class_accs
        }


# =============================================================================
# 可视化
# =============================================================================

def plot_training_history(trainer: GNNTrainer, save_path: Optional[str] = None):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(trainer.train_losses) + 1)
    
    # 损失曲线
    axes[0].plot(epochs, trainer.train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, trainer.val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[1].plot(epochs, trainer.train_accs, 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, trainer.val_accs, 'r-', label='Val Acc', linewidth=2)
    axes[1].plot(epochs, trainer.test_accs, 'g-', label='Test Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Over Training')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练历史图已保存至: {save_path}")
    
    plt.show()


def visualize_graph_structure(data: CitationNetwork, save_path: Optional[str] = None):
    """可视化图结构（简化版）"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 类别分布
    ax1 = axes[0]
    class_dist = data.get_class_distribution()
    classes = list(class_dist.keys())
    counts = list(class_dist.values())
    
    bars = ax1.bar(classes, counts, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Number of Nodes')
    ax1.set_title('Class Distribution')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontsize=9)
    
    # 度分布
    ax2 = axes[1]
    degrees = np.sum(data.adj_matrix, axis=1) - 1  # 减去自环
    ax2.hist(degrees, bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Node Degree')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Degree Distribution')
    ax2.axvline(np.mean(degrees), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(degrees):.1f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图结构可视化已保存至: {save_path}")
    
    plt.show()


def plot_confusion_matrix(data: CitationNetwork, model: GCN, 
                          save_path: Optional[str] = None):
    """绘制混淆矩阵"""
    logits = model.forward(data.features, data.norm_adj, training=False)
    preds = np.argmax(logits, axis=1)
    
    test_preds = preds[data.test_mask]
    test_labels = data.labels[data.test_mask]
    
    # 计算混淆矩阵
    num_classes = data.config.num_classes
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    for true, pred in zip(test_labels, test_preds):
        cm[true, pred] += 1
    
    # 绘制
    fig, ax = plt.subplots(figsize=(8, 7))
    
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix (Test Set)')
    
    # 设置刻度
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels([f'C{i}' for i in range(num_classes)])
    ax.set_yticklabels([f'C{i}' for i in range(num_classes)])
    
    # 添加数值
    for i in range(num_classes):
        for j in range(num_classes):
            text = ax.text(j, i, cm[i, j],
                          ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white")
    
    plt.colorbar(im, ax=ax, label='Count')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"混淆矩阵已保存至: {save_path}")
    
    plt.show()


def visualize_embeddings(data: CitationNetwork, model: GCN, 
                         save_path: Optional[str] = None):
    """可视化节点嵌入（使用t-SNE降维的简化版）"""
    # 获取中间层表示（简化：使用输入特征）
    # 实际实现中应该提取中间层输出
    
    # 使用PCA降维到2D进行可视化
    from sklearn.decomposition import PCA
    
    # 随机采样一部分节点进行可视化
    sample_size = min(500, data.config.num_nodes)
    sample_indices = np.random.choice(data.config.num_nodes, sample_size, replace=False)
    
    features_sample = data.features[sample_indices]
    labels_sample = data.labels[sample_indices]
    
    # PCA降维
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(features_sample)
    
    # 绘制
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, data.config.num_classes))
    
    for c in range(data.config.num_classes):
        mask = labels_sample == c
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                  c=[colors[c]], label=f'Class {c}',
                  alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_title('Node Embeddings Visualization (PCA)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"嵌入可视化已保存至: {save_path}")
    
    plt.show()


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    # 设置配置
    config = GNNConfig(
        num_nodes=2708,
        num_features=1433,
        num_classes=7,
        num_edges=5429,
        hidden_dims=[256, 128],
        dropout=0.5,
        learning_rate=0.01,
        weight_decay=5e-4,
        epochs=200,
        seed=42
    )
    
    # 创建数据集
    print("生成引文网络数据集...")
    data = CitationNetwork(config)
    
    # 打印数据集统计
    stats = data.get_graph_statistics()
    print("\n图统计信息:")
    print(f"  节点数: {stats['num_nodes']}")
    print(f"  边数: {stats['num_edges']}")
    print(f"  平均度: {stats['avg_degree']:.2f}")
    print(f"  平均聚类系数: {stats['avg_clustering']:.4f}")
    print(f"  图密度: {stats['density']:.4f}")
    
    # 可视化图结构
    visualize_graph_structure(data, save_path='/workspace/gnn_graph_structure.png')
    
    # 创建模型
    print("\n构建GCN模型...")
    model = GCN(config)
    
    # 统计参数量
    total_params = sum(p.size for p in model.get_parameters())
    print(f"模型总参数量: {total_params:,}")
    
    # 创建训练器
    trainer = GNNTrainer(model, config)
    
    # 训练
    trainer.fit(data)
    
    # 绘制训练历史
    plot_training_history(trainer, save_path='/workspace/gnn_training_history.png')
    
    # 详细评估
    print("\n详细评估指标:")
    metrics = trainer.get_detailed_metrics(data)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 混淆矩阵
    plot_confusion_matrix(data, model, save_path='/workspace/gnn_confusion_matrix.png')
    
    # 嵌入可视化
    print("\n生成嵌入可视化...")
    visualize_embeddings(data, model, save_path='/workspace/gnn_embeddings.png')
    
    print("\n示例运行完成!")


if __name__ == '__main__':
    main()
