#!/usr/bin/env python3
"""
MNIST手写数字分类器示例
============================

使用AGI统一框架实现的完整CNN分类器示例。
包含数据加载、模型定义、训练和评估流程。

作者: AGI Framework Team
日期: 2025-05-13
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import time

# 导入框架模块
from core.swing_layer.convolution.convolution_layers import Conv2D, MaxPool2D
from core.swing_layer.neural_layers import Linear, Dropout, Flatten
from core.activations.activations import ReLU, Softmax
from core.initialization.initializers import HeInitializer, XavierInitializer
from core.normalization.normalizations import BatchNormalization2D
from training.optimizers.optimizers import Adam
from training.losses.losses import CrossEntropyLoss
from evaluation.metrics.metrics import Accuracy, Precision, Recall, F1Score


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class MNISTConfig:
    """MNIST训练配置"""
    # 数据参数
    image_size: int = 28
    num_classes: int = 10
    num_channels: int = 1
    
    # 训练参数
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # 模型参数
    conv_filters: List[int] = None
    dense_units: List[int] = None
    dropout_rate: float = 0.5
    
    # 其他
    seed: int = 42
    verbose: bool = True
    
    def __post_init__(self):
        if self.conv_filters is None:
            self.conv_filters = [32, 64, 128]
        if self.dense_units is None:
            self.dense_units = [256, 128]


# =============================================================================
# 模拟MNIST数据集
# =============================================================================

class SimulatedMNIST:
    """
    模拟MNIST数据集生成器
    生成具有真实MNIST统计特征的手写数字数据
    """
    
    def __init__(self, config: MNISTConfig):
        self.config = config
        np.random.seed(config.seed)
        
        # 预定义的数字模式 (0-9)
        self.digit_patterns = self._create_digit_patterns()
        
    def _create_digit_patterns(self) -> List[np.ndarray]:
        """创建基础数字模式"""
        patterns = []
        size = self.config.image_size
        
        for digit in range(10):
            pattern = np.zeros((size, size))
            
            # 为每个数字创建独特的几何模式
            center = size // 2
            
            if digit == 0:  # 圆形
                y, x = np.ogrid[:size, :size]
                mask = ((x - center)**2 + (y - center)**2 <= (size//3)**2) & \
                       ((x - center)**2 + (y - center)**2 >= (size//5)**2)
                pattern[mask] = 1.0
                
            elif digit == 1:  # 垂直线
                pattern[:, center-2:center+2] = 1.0
                pattern[2:6, center-4:center] = 1.0  # 顶部斜线
                
            elif digit == 2:  # 曲线
                for i in range(size):
                    x_pos = int(center + (i - center) * 0.5)
                    if 0 <= x_pos < size:
                        pattern[i, max(0, x_pos-2):min(size, x_pos+2)] = 1.0
                pattern[size//3, :] = 1.0
                pattern[2*size//3, :] = 1.0
                
            elif digit == 3:  # 双曲线
                for i in range(size):
                    x_pos = int(center + abs(i - center) * 0.3)
                    if 0 <= x_pos < size:
                        pattern[i, max(0, x_pos-2):min(size, x_pos+2)] = 1.0
                pattern[size//3, center-3:center+3] = 1.0
                pattern[2*size//3, center-3:center+3] = 1.0
                
            elif digit == 4:  # 交叉线
                pattern[size//2:, center-2:center+2] = 1.0
                pattern[:size//2+3, center-2:center+2] = 0.5
                pattern[size//2, :] = 1.0
                
            elif digit == 5:  # S形
                pattern[size//5:2*size//5, :] = 1.0
                pattern[2*size//5:3*size//5, center:] = 1.0
                pattern[3*size//5:4*size//5, :] = 1.0
                pattern[2*size//5, :center] = 1.0
                pattern[3*size//5, center:] = 1.0
                
            elif digit == 6:  # 带圈的6
                pattern[size//3:, center-3:center+3] = 1.0
                y, x = np.ogrid[size//2:size, :size]
                mask = ((x - center)**2 + (y - 3*size//4)**2 <= (size//5)**2)
                pattern[size//2:size, :][mask] = 1.0
                
            elif digit == 7:  # 斜线
                for i in range(size):
                    x_pos = int(size - 1 - i * 0.8)
                    if 0 <= x_pos < size:
                        pattern[i, max(0, x_pos-2):min(size, x_pos+2)] = 1.0
                pattern[2, :] = 1.0
                
            elif digit == 8:  # 双圆
                y, x = np.ogrid[:size, :size]
                mask1 = ((x - center)**2 + (y - size//3)**2 <= (size//6)**2)
                mask2 = ((x - center)**2 + (y - 2*size//3)**2 <= (size//6)**2)
                pattern[mask1 | mask2] = 1.0
                
            elif digit == 9:  # 带圈的9
                pattern[:2*size//3, center-3:center+3] = 1.0
                y, x = np.ogrid[:size//2+3, :size]
                mask = ((x - center)**2 + (y - size//3)**2 <= (size//5)**2)
                pattern[:size//2+3, :][mask] = 1.0
            
            patterns.append(pattern)
            
        return patterns
    
    def _add_noise(self, image: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """添加噪声模拟真实手写"""
        noise = np.random.normal(0, noise_level, image.shape)
        noisy = image + noise
        return np.clip(noisy, 0, 1)
    
    def _augment(self, image: np.ndarray) -> np.ndarray:
        """数据增强"""
        # 随机旋转 (简化版)
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)
            # 简化的旋转模拟
            shift = int(angle / 15)
            if shift != 0:
                image = np.roll(image, shift, axis=1)
        
        # 随机缩放
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            new_size = int(self.config.image_size * scale)
            if new_size != self.config.image_size:
                # 简化的缩放
                pad = (self.config.image_size - new_size) // 2
                if pad > 0:
                    image = np.pad(image[pad:-pad, pad:-pad] if pad < new_size//2 else image,
                                 ((pad, pad), (pad, pad)), mode='constant')
        
        return np.clip(image, 0, 1)
    
    def generate_batch(self, batch_size: int, split: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
        """生成一批数据"""
        images = []
        labels = []
        
        for _ in range(batch_size):
            label = np.random.randint(0, 10)
            base_pattern = self.digit_patterns[label].copy()
            
            # 添加变化
            if split == 'train':
                base_pattern = self._augment(base_pattern)
            
            # 添加噪声
            noise_level = 0.15 if split == 'train' else 0.05
            image = self._add_noise(base_pattern, noise_level)
            
            images.append(image)
            labels.append(label)
        
        # 添加通道维度并归一化
        images = np.array(images)[:, np.newaxis, :, :].astype(np.float32)
        labels = np.array(labels).astype(np.int64)
        
        return images, labels
    
    def generate_dataset(self, num_samples: int, split: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
        """生成完整数据集"""
        all_images = []
        all_labels = []
        
        num_batches = num_samples // self.config.batch_size
        for _ in range(num_batches):
            images, labels = self.generate_batch(self.config.batch_size, split)
            all_images.append(images)
            all_labels.append(labels)
        
        return np.concatenate(all_images), np.concatenate(all_labels)


# =============================================================================
# CNN模型
# =============================================================================

class MNISTClassifier:
    """MNIST CNN分类器"""
    
    def __init__(self, config: MNISTConfig):
        self.config = config
        self.layers = []
        self._build_model()
        
    def _build_model(self):
        """构建CNN架构"""
        # 卷积块1
        self.layers.append(Conv2D(
            in_channels=self.config.num_channels,
            out_channels=self.config.conv_filters[0],
            kernel_size=3,
            padding=1,
            initializer=HeInitializer()
        ))
        self.layers.append(BatchNormalization2D(self.config.conv_filters[0]))
        self.layers.append(ReLU())
        self.layers.append(MaxPool2D(kernel_size=2, stride=2))
        
        # 卷积块2
        self.layers.append(Conv2D(
            in_channels=self.config.conv_filters[0],
            out_channels=self.config.conv_filters[1],
            kernel_size=3,
            padding=1,
            initializer=HeInitializer()
        ))
        self.layers.append(BatchNormalization2D(self.config.conv_filters[1]))
        self.layers.append(ReLU())
        self.layers.append(MaxPool2D(kernel_size=2, stride=2))
        
        # 卷积块3
        self.layers.append(Conv2D(
            in_channels=self.config.conv_filters[1],
            out_channels=self.config.conv_filters[2],
            kernel_size=3,
            padding=1,
            initializer=HeInitializer()
        ))
        self.layers.append(BatchNormalization2D(self.config.conv_filters[2]))
        self.layers.append(ReLU())
        self.layers.append(MaxPool2D(kernel_size=2, stride=2))
        
        # 展平
        self.layers.append(Flatten())
        
        # 全连接层
        # 计算展平后的维度: 28 -> 14 -> 7 -> 3 (经过3次pooling)
        flat_dim = self.config.conv_filters[2] * 3 * 3
        
        self.layers.append(Linear(
            in_features=flat_dim,
            out_features=self.config.dense_units[0],
            initializer=HeInitializer()
        ))
        self.layers.append(ReLU())
        self.layers.append(Dropout(self.config.dropout_rate))
        
        self.layers.append(Linear(
            in_features=self.config.dense_units[0],
            out_features=self.config.dense_units[1],
            initializer=HeInitializer()
        ))
        self.layers.append(ReLU())
        self.layers.append(Dropout(self.config.dropout_rate))
        
        # 输出层
        self.layers.append(Linear(
            in_features=self.config.dense_units[1],
            out_features=self.config.num_classes,
            initializer=XavierInitializer()
        ))
        self.layers.append(Softmax())
        
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """前向传播"""
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = training
            x = layer.forward(x)
        return x
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """反向传播"""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def get_parameters(self) -> List[np.ndarray]:
        """获取所有可训练参数"""
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


# =============================================================================
# 训练器
# =============================================================================

class MNISTTrainer:
    """MNIST训练器"""
    
    def __init__(self, model: MNISTClassifier, config: MNISTConfig):
        self.model = model
        self.config = config
        self.optimizer = Adam(
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.criterion = CrossEntropyLoss()
        
        # 指标追踪
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, data_generator: SimulatedMNIST, num_batches: int) -> Dict[str, float]:
        """训练一个epoch"""
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for batch_idx in range(num_batches):
            # 生成数据
            images, labels = data_generator.generate_batch(self.config.batch_size, 'train')
            
            # 前向传播
            outputs = self.model.forward(images, training=True)
            
            # 计算损失
            loss = self.criterion.forward(outputs, labels)
            epoch_loss += loss
            
            # 计算准确率
            predictions = np.argmax(outputs, axis=1)
            epoch_correct += np.sum(predictions == labels)
            epoch_total += len(labels)
            
            # 反向传播
            grad = self.criterion.backward(outputs, labels)
            self.model.backward(grad)
            
            # 更新参数
            params = self.model.get_parameters()
            grads = [p.grad for p in params if hasattr(p, 'grad')]
            if grads:
                self.optimizer.step(params, grads)
            
            if self.config.verbose and batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{num_batches}, Loss: {loss:.4f}")
        
        avg_loss = epoch_loss / num_batches
        accuracy = epoch_correct / epoch_total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(self, data_generator: SimulatedMNIST, num_batches: int) -> Dict[str, float]:
        """验证"""
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        all_preds = []
        all_labels = []
        
        for _ in range(num_batches):
            images, labels = data_generator.generate_batch(self.config.batch_size, 'val')
            
            outputs = self.model.forward(images, training=False)
            loss = self.criterion.forward(outputs, labels)
            
            val_loss += loss
            predictions = np.argmax(outputs, axis=1)
            val_correct += np.sum(predictions == labels)
            val_total += len(labels)
            
            all_preds.extend(predictions)
            all_labels.extend(labels)
        
        # 计算详细指标
        accuracy = Accuracy()(np.array(all_labels), np.array(all_preds))
        precision = Precision(num_classes=10)(np.array(all_labels), np.array(all_preds))
        recall = Recall(num_classes=10)(np.array(all_labels), np.array(all_preds))
        f1 = F1Score(num_classes=10)(np.array(all_labels), np.array(all_preds))
        
        return {
            'loss': val_loss / num_batches,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def fit(self, train_generator: SimulatedMNIST, val_generator: SimulatedMNIST,
            train_batches: int = 100, val_batches: int = 20):
        """完整训练流程"""
        print("=" * 60)
        print("开始训练MNIST分类器")
        print("=" * 60)
        
        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            print("-" * 40)
            
            # 训练
            start_time = time.time()
            train_metrics = self.train_epoch(train_generator, train_batches)
            train_time = time.time() - start_time
            
            # 验证
            val_metrics = self.validate(val_generator, val_batches)
            
            # 记录
            self.train_losses.append(train_metrics['loss'])
            self.train_accuracies.append(train_metrics['accuracy'])
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])
            
            # 打印
            print(f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f} | "
                  f"Time: {train_time:.2f}s")
            print(f"Val   Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}")
        
        print("\n" + "=" * 60)
        print("训练完成!")
        print("=" * 60)
    
    def plot_history(self, save_path: Optional[str] = None):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.val_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # 准确率曲线
        axes[1].plot(self.train_accuracies, label='Train Acc')
        axes[1].plot(self.val_accuracies, label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        
        plt.show()


# =============================================================================
# 评估与可视化
# =============================================================================

def visualize_predictions(model: MNISTClassifier, data_generator: SimulatedMNIST, 
                          num_samples: int = 10, save_path: Optional[str] = None):
    """可视化预测结果"""
    images, labels = data_generator.generate_batch(num_samples, 'val')
    predictions = model.forward(images, training=False)
    pred_labels = np.argmax(predictions, axis=1)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    
    for i in range(num_samples):
        ax = axes[i]
        ax.imshow(images[i, 0], cmap='gray')
        color = 'green' if pred_labels[i] == labels[i] else 'red'
        ax.set_title(f'True: {labels[i]}, Pred: {pred_labels[i]}', color=color)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"预测可视化已保存至: {save_path}")
    
    plt.show()


def print_model_summary(model: MNISTClassifier):
    """打印模型摘要"""
    print("\n模型架构:")
    print("-" * 60)
    total_params = 0
    for i, layer in enumerate(model.layers):
        layer_name = layer.__class__.__name__
        if hasattr(layer, 'get_parameters'):
            params = layer.get_parameters()
            num_params = sum(p.size for p in params)
            total_params += num_params
            print(f"{i+1:2d}. {layer_name:20s} - {num_params:,} 参数")
        else:
            print(f"{i+1:2d}. {layer_name:20s}")
    print("-" * 60)
    print(f"总参数量: {total_params:,}")
    print("=" * 60)


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    # 设置配置
    config = MNISTConfig(
        batch_size=64,
        epochs=10,
        learning_rate=0.001,
        dropout_rate=0.3,
        seed=42
    )
    
    # 创建数据生成器
    print("初始化数据生成器...")
    train_generator = SimulatedMNIST(config)
    val_generator = SimulatedMNIST(config)
    
    # 创建模型
    print("构建模型...")
    model = MNISTClassifier(config)
    print_model_summary(model)
    
    # 创建训练器
    trainer = MNISTTrainer(model, config)
    
    # 训练
    trainer.fit(
        train_generator=train_generator,
        val_generator=val_generator,
        train_batches=50,
        val_batches=10
    )
    
    # 绘制训练历史
    trainer.plot_history(save_path='/workspace/mnist_training_history.png')
    
    # 可视化预测
    print("\n生成预测可视化...")
    visualize_predictions(model, val_generator, num_samples=10, 
                         save_path='/workspace/mnist_predictions.png')
    
    print("\n示例运行完成!")


if __name__ == '__main__':
    main()
