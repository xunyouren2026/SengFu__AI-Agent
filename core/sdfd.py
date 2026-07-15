"""
自蒸馏特征检测模块 (Self-Distilled Feature Detection)

该模块实现了基于教师-学生网络的自蒸馏框架,用于特征学习和表示优化。
通过在线硬负样本挖掘和自适应温度缩放,提升模型的特征提取能力。

主要特性:
- 教师-学生网络架构,支持知识传递
- 特征对齐损失(MSE对齐中间层)
- 在线硬负样本挖掘
- 特征重要性评分
- 自适应温度缩放

作者: AGI Unified Framework Team
版本: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable, Union
import numpy as np
from dataclasses import dataclass
from collections import deque


@dataclass
class SDFDConfig:
    """SDFD配置类
    
    Attributes:
        feature_dims: 特征维度列表,对应不同层的输出维度
        temperature: 初始温度参数
        min_temperature: 最小温度
        max_temperature: 最大温度
        alpha: 蒸馏损失权重
        beta: 特征对齐损失权重
        gamma: 硬负样本挖掘权重
        hard_negative_ratio: 硬负样本比例
        momentum: 教师网络动量更新系数
        importance_threshold: 特征重要性阈值
        max_queue_size: 特征队列最大大小
    """
    feature_dims: List[int] = None
    temperature: float = 4.0
    min_temperature: float = 1.0
    max_temperature: float = 10.0
    alpha: float = 0.5
    beta: float = 0.3
    gamma: float = 0.2
    hard_negative_ratio: float = 0.3
    momentum: float = 0.999
    importance_threshold: float = 0.5
    max_queue_size: int = 1000


class FeatureAlignLoss(nn.Module):
    """特征对齐损失模块
    
    使用MSE损失对齐教师网络和学生网络的中间层特征。
    支持多层特征对齐和自适应权重。
    """
    
    def __init__(self, layer_weights: Optional[List[float]] = None):
        """
        Args:
            layer_weights: 各层特征的权重,默认为等权重
        """
        super().__init__()
        self.layer_weights = layer_weights
        self.mse_loss = nn.MSELoss(reduction='mean')
    
    def forward(
        self,
        teacher_feats: List[torch.Tensor],
        student_feats: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        计算多层特征对齐损失
        
        Args:
            teacher_feats: 教师网络特征列表
            student_feats: 学生网络特征列表
            
        Returns:
            加权特征对齐损失
        """
        if len(teacher_feats) != len(student_feats):
            raise ValueError("教师和学生特征层数不匹配")
        
        num_layers = len(teacher_feats)
        if self.layer_weights is None:
            weights = [1.0 / num_layers] * num_layers
        else:
            weights = self.layer_weights
            # 归一化权重
            total = sum(weights)
            weights = [w / total for w in weights]
        
        total_loss = 0.0
        for i, (t_feat, s_feat, weight) in enumerate(zip(teacher_feats, student_feats, weights)):
            # 确保特征维度匹配
            if t_feat.shape != s_feat.shape:
                # 使用自适应池化统一尺寸
                s_feat = F.adaptive_avg_pool1d(
                    s_feat.flatten(2), t_feat.flatten(2).shape[-1]
                ).reshape(t_feat.shape)
            
            # 归一化特征
            t_feat = F.normalize(t_feat, p=2, dim=1)
            s_feat = F.normalize(s_feat, p=2, dim=1)
            
            layer_loss = self.mse_loss(s_feat, t_feat)
            total_loss += weight * layer_loss
        
        return total_loss


class HardNegativeMiner(nn.Module):
    """在线硬负样本挖掘模块
    
    根据特征相似度和预测差异,动态挖掘难样本,
    提升模型对困难样本的学习能力。
    """
    
    def __init__(self, ratio: float = 0.3, min_margin: float = 0.1):
        """
        Args:
            ratio: 硬负样本比例
            min_margin: 最小间隔阈值
        """
        super().__init__()
        self.ratio = ratio
        self.min_margin = min_margin
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        predictions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        挖掘硬负样本
        
        Args:
            features: 特征张量 [N, D]
            labels: 标签张量 [N]
            predictions: 预测结果 [N, C], 可选
            
        Returns:
            (硬负样本索引, 硬负样本权重)
        """
        batch_size = features.shape[0]
        num_hard = max(1, int(batch_size * self.ratio))
        
        # 计算特征相似度矩阵
        features_norm = F.normalize(features, p=2, dim=1)
        similarity = torch.mm(features_norm, features_norm.t())
        
        # 计算样本难度分数
        difficulty_scores = torch.zeros(batch_size, device=features.device)
        
        for i in range(batch_size):
            # 同类样本相似度
            same_class_mask = (labels == labels[i])
            same_class_mask[i] = False  # 排除自身
            
            if same_class_mask.sum() > 0:
                same_class_sim = similarity[i][same_class_mask].mean()
            else:
                same_class_sim = 1.0
            
            # 不同类样本相似度
            diff_class_mask = ~same_class_mask
            diff_class_mask[i] = False
            
            if diff_class_mask.sum() > 0:
                diff_class_sim = similarity[i][diff_class_mask].mean()
            else:
                diff_class_sim = 0.0
            
            # 难度分数: 不同类相似度高 + 同类相似度低 = 困难
            difficulty_scores[i] = diff_class_sim - same_class_sim + self.min_margin
        
        # 如果有预测结果,加入预测置信度
        if predictions is not None:
            probs = F.softmax(predictions, dim=1)
            max_probs = probs.max(dim=1)[0]
            # 低置信度样本更困难
            difficulty_scores += (1.0 - max_probs)
        
        # 选择最难的样本
        _, hard_indices = torch.topk(difficulty_scores, num_hard, largest=True)
        
        # 计算权重(难度越高权重越大)
        hard_scores = difficulty_scores[hard_indices]
        weights = F.softmax(hard_scores, dim=0)
        
        return hard_indices, weights


class FeatureImportanceScorer(nn.Module):
    """特征重要性评分模块
    
    基于注意力机制和梯度信息,计算特征维度的重要性分数,
    用于特征选择和模型压缩。
    """
    
    def __init__(self, feature_dim: int, num_heads: int = 4):
        """
        Args:
            feature_dim: 特征维度
            num_heads: 注意力头数
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # 自注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 重要性预测头
        self.importance_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 通道注意力
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, feature_dim // 16),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 16, feature_dim),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算特征重要性分数
        
        Args:
            features: 特征张量 [N, D] 或 [N, C, H, W]
            
        Returns:
            (样本重要性分数 [N], 通道重要性分数 [D] 或 [C])
        """
        original_shape = features.shape
        
        # 处理不同维度的特征
        if len(original_shape) == 4:  # [N, C, H, W]
            N, C, H, W = original_shape
            features_flat = features.view(N, C, -1).permute(0, 2, 1)  # [N, HW, C]
        elif len(original_shape) == 3:  # [N, L, D]
            features_flat = features
            N, L, D = original_shape
            C = D
        else:  # [N, D]
            features_flat = features.unsqueeze(1)  # [N, 1, D]
            N, C = original_shape
        
        # 自注意力
        attn_out, _ = self.attention(features_flat, features_flat, features_flat)
        
        # 全局平均池化
        if len(original_shape) == 4:
            pooled = attn_out.mean(dim=1)  # [N, C]
        elif len(original_shape) == 3:
            pooled = attn_out.mean(dim=1)  # [N, D]
        else:
            pooled = attn_out.squeeze(1)  # [N, D]
        
        # 计算样本重要性
        sample_importance = self.importance_head(pooled).squeeze(-1)  # [N]
        
        # 计算通道重要性
        if len(original_shape) == 4:
            channel_importance = self.channel_attn(features.view(N, C, -1))
        else:
            channel_importance = torch.ones(C, device=features.device) / C
        
        return sample_importance, channel_importance


class AdaptiveTemperatureScheduler:
    """自适应温度调度器
    
    根据训练进度和模型表现,动态调整蒸馏温度,
    实现从软标签到硬标签的平滑过渡。
    """
    
    def __init__(
        self,
        initial_temp: float = 4.0,
        min_temp: float = 1.0,
        max_temp: float = 10.0,
        warmup_epochs: int = 5,
        decay_epochs: int = 50,
        strategy: str = 'cosine'
    ):
        """
        Args:
            initial_temp: 初始温度
            min_temp: 最小温度
            max_temp: 最大温度
            warmup_epochs: 预热轮数
            decay_epochs: 衰减轮数
            strategy: 调度策略 ('linear', 'cosine', 'exponential')
        """
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.strategy = strategy
        self.current_epoch = 0
    
    def step(self, epoch: Optional[int] = None) -> float:
        """
        更新并返回当前温度
        
        Args:
            epoch: 当前轮数,None则使用内部计数
            
        Returns:
            当前温度值
        """
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        epoch = self.current_epoch
        
        # 预热阶段
        if epoch < self.warmup_epochs:
            progress = epoch / self.warmup_epochs
            temp = self.initial_temp + (self.max_temp - self.initial_temp) * progress
        # 衰减阶段
        elif epoch < self.warmup_epochs + self.decay_epochs:
            progress = (epoch - self.warmup_epochs) / self.decay_epochs
            
            if self.strategy == 'linear':
                temp = self.max_temp - (self.max_temp - self.min_temp) * progress
            elif self.strategy == 'cosine':
                temp = self.min_temp + (self.max_temp - self.min_temp) * \
                       (1 + np.cos(np.pi * progress)) / 2
            elif self.strategy == 'exponential':
                temp = self.min_temp + (self.max_temp - self.min_temp) * \
                       np.exp(-5 * progress)
            else:
                temp = self.max_temp - (self.max_temp - self.min_temp) * progress
        else:
            temp = self.min_temp
        
        return max(self.min_temp, min(self.max_temp, temp))
    
    def get_temperature(self) -> float:
        """获取当前温度"""
        return self.step(self.current_epoch)


class SDFD(nn.Module):
    """自蒸馏特征检测主类
    
    实现完整的自蒸馏框架,包括教师-学生网络、
    特征对齐、硬负样本挖掘和自适应温度。
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        config: Optional[SDFDConfig] = None
    ):
        """
        Args:
            teacher_model: 教师网络
            student_model: 学生网络
            config: SDFD配置
        """
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.config = config or SDFDConfig()
        
        # 教师网络不参与梯度计算
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # 初始化损失模块
        self.feature_align_loss = FeatureAlignLoss()
        self.hard_negative_miner = HardNegativeMiner(
            ratio=self.config.hard_negative_ratio
        )
        
        # 初始化温度调度器
        self.temp_scheduler = AdaptiveTemperatureScheduler(
            initial_temp=self.config.temperature,
            min_temp=self.config.min_temperature,
            max_temp=self.config.max_temperature
        )
        
        # 特征队列(用于存储历史特征)
        self.feature_queues: Dict[str, deque] = {}
        self._init_feature_queues()
        
        # 重要性评分器(延迟初始化)
        self.importance_scorer: Optional[FeatureImportanceScorer] = None
    
    def _init_feature_queues(self):
        """初始化特征队列"""
        if self.config.feature_dims:
            for i, dim in enumerate(self.config.feature_dims):
                self.feature_queues[f'layer_{i}'] = deque(
                    maxlen=self.config.max_queue_size
                )
    
    def _update_teacher(self, momentum: Optional[float] = None):
        """
        使用动量更新教师网络
        
        Args:
            momentum: 动量系数,None则使用配置值
        """
        m = momentum or self.config.momentum
        
        with torch.no_grad():
            for t_param, s_param in zip(
                self.teacher.parameters(),
                self.student.parameters()
            ):
                t_param.data = m * t_param.data + (1 - m) * s_param.data
    
    def compute_feature_loss(
        self,
        teacher_feats: List[torch.Tensor],
        student_feats: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        计算特征对齐损失
        
        Args:
            teacher_feats: 教师网络特征列表
            student_feats: 学生网络特征列表
            
        Returns:
            特征对齐损失
        """
        return self.feature_align_loss(teacher_feats, student_feats)
    
    def mine_hard_negatives(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        predictions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        挖掘硬负样本
        
        Args:
            features: 特征张量
            labels: 标签张量
            predictions: 预测结果,可选
            
        Returns:
            (硬负样本索引, 硬负样本权重)
        """
        return self.hard_negative_miner(features, labels, predictions)
    
    def compute_importance_scores(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算特征重要性分数
        
        Args:
            features: 特征张量
            
        Returns:
            (样本重要性分数, 通道重要性分数)
        """
        if self.importance_scorer is None:
            feature_dim = features.shape[1] if len(features.shape) > 1 else features.shape[0]
            self.importance_scorer = FeatureImportanceScorer(
                feature_dim=feature_dim
            ).to(features.device)
        
        return self.importance_scorer(features)
    
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """
        计算知识蒸馏损失(KL散度)
        
        Args:
            student_logits: 学生网络输出
            teacher_logits: 教师网络输出
            temperature: 温度参数
            
        Returns:
            蒸馏损失
        """
        # 软标签
        soft_targets = F.softmax(teacher_logits / temperature, dim=1)
        soft_predictions = F.log_softmax(student_logits / temperature, dim=1)
        
        # KL散度损失
        kl_loss = F.kl_div(
            soft_predictions,
            soft_targets,
            reduction='batchmean'
        ) * (temperature ** 2)
        
        return kl_loss
    
    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]
    ]:
        """
        前向传播
        
        Args:
            x: 输入张量
            labels: 标签张量,用于硬负样本挖掘
            return_features: 是否返回特征
            
        Returns:
            (学生输出, 总蒸馏损失) 或 (学生输出, 总蒸馏损失, 特征字典)
        """
        # 获取当前温度
        temperature = self.temp_scheduler.get_temperature()
        
        # 教师网络前向(不计算梯度)
        with torch.no_grad():
            if hasattr(self.teacher, 'forward_features'):
                teacher_out, teacher_feats = self.teacher.forward_features(x)
            else:
                teacher_out = self.teacher(x)
                teacher_feats = [teacher_out]
        
        # 学生网络前向
        if hasattr(self.student, 'forward_features'):
            student_out, student_feats = self.student.forward_features(x)
        else:
            student_out = self.student(x)
            student_feats = [student_out]
        
        # 计算蒸馏损失
        distillation_loss = self.compute_distillation_loss(
            student_out, teacher_out, temperature
        )
        
        # 计算特征对齐损失
        if len(teacher_feats) == len(student_feats):
            feature_loss = self.compute_feature_loss(teacher_feats, student_feats)
        else:
            feature_loss = torch.tensor(0.0, device=x.device)
        
        # 硬负样本挖掘损失
        hard_negative_loss = torch.tensor(0.0, device=x.device)
        if labels is not None and len(student_feats) > 0:
            # 使用最后一层特征
            features = student_feats[-1]
            if len(features.shape) > 2:
                features = features.mean(dim=[2, 3]) if len(features.shape) == 4 \
                          else features.mean(dim=1)
            
            hard_indices, hard_weights = self.mine_hard_negatives(
                features, labels, student_out
            )
            
            # 硬样本的蒸馏损失加权
            hard_student_logits = student_out[hard_indices]
            hard_teacher_logits = teacher_out[hard_indices]
            hard_loss = self.compute_distillation_loss(
                hard_student_logits, hard_teacher_logits, temperature
            )
            hard_negative_loss = (hard_loss * hard_weights).sum()
        
        # 总损失
        total_loss = (
            self.config.alpha * distillation_loss +
            self.config.beta * feature_loss +
            self.config.gamma * hard_negative_loss
        )
        
        # 更新教师网络
        if self.training:
            self._update_teacher()
        
        if return_features:
            features_dict = {
                'teacher_features': teacher_feats,
                'student_features': student_feats,
                'temperature': torch.tensor(temperature),
                'distillation_loss': distillation_loss,
                'feature_loss': feature_loss,
                'hard_negative_loss': hard_negative_loss
            }
            return student_out, total_loss, features_dict
        
        return student_out, total_loss
    
    def get_feature_importance(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        获取特征重要性分析
        
        Args:
            x: 输入样本
            
        Returns:
            包含重要性分数的字典
        """
        with torch.no_grad():
            if hasattr(self.student, 'forward_features'):
                _, student_feats = self.student.forward_features(x)
            else:
                out = self.student(x)
                student_feats = [out]
        
        # 计算各层特征的重要性
        importance_results = {}
        for i, feat in enumerate(student_feats):
            if len(feat.shape) > 2:
                feat_pooled = feat.mean(dim=[2, 3]) if len(feat.shape) == 4 \
                             else feat.mean(dim=1)
            else:
                feat_pooled = feat
            
            sample_imp, channel_imp = self.compute_importance_scores(feat_pooled)
            importance_results[f'layer_{i}'] = {
                'sample_importance': sample_imp,
                'channel_importance': channel_imp
            }
        
        return importance_results
    
    def update_temperature(self, epoch: Optional[int] = None) -> float:
        """
        更新温度参数
        
        Args:
            epoch: 当前轮数
            
        Returns:
            更新后的温度值
        """
        return self.temp_scheduler.step(epoch)


# 辅助函数
def create_sdfd_from_config(
    teacher_model: nn.Module,
    student_model: nn.Module,
    config_dict: Dict
) -> SDFD:
    """
    从配置字典创建SDFD实例
    
    Args:
        teacher_model: 教师网络
        student_model: 学生网络
        config_dict: 配置字典
        
    Returns:
        SDFD实例
    """
    config = SDFDConfig(**config_dict)
    return SDFD(teacher_model, student_model, config)


def build_feature_extractor(
    model: nn.Module,
    layer_names: List[str]
) -> Callable:
    """
    构建特征提取器
    
    Args:
        model: 目标模型
        layer_names: 要提取的层名列表
        
    Returns:
        特征提取函数
    """
    features = {}
    handles = []
    
    def hook_fn(name):
        def hook(module, input, output):
            features[name] = output
        return hook
    
    for name in layer_names:
        layer = dict(model.named_modules())[name]
        handle = layer.register_forward_hook(hook_fn(name))
        handles.append(handle)
    
    def extractor(x):
        features.clear()
        _ = model(x)
        return [features[name] for name in layer_names]
    
    return extractor
