"""
AGI统一框架 - 模型集成与融合
实现模型集成、投票、堆叠、蒸馏融合等方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Callable
from collections import defaultdict
import random


# ==================== 模型集成 ====================

class ModelEnsemble(nn.Module):
    """模型集成基类"""
    
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """默认集成：对所有子模型的输出取平均"""
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs, dim=0).mean(dim=0)


class AverageEnsemble(ModelEnsemble):
    """平均集成"""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__(models)
        if weights is None:
            self.weights = [1.0 / self.num_models] * self.num_models
        else:
            self.weights = weights
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [w * model(x) for w, model in zip(self.weights, self.models)]
        return sum(outputs)


class VotingEnsemble(ModelEnsemble):
    """投票集成"""
    
    def __init__(self, models: List[nn.Module], voting: str = 'hard'):
        super().__init__(models)
        self.voting = voting
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [model(x) for model in self.models]
        
        if self.voting == 'hard':
            # 硬投票：多数表决
            predictions = [output.argmax(dim=-1) for output in outputs]
            predictions = torch.stack(predictions, dim=0)
            
            # 统计每个类别的票数
            num_classes = outputs[0].size(-1)
            votes = torch.zeros(x.size(0), num_classes, device=x.device)
            
            for pred in predictions:
                for i, p in enumerate(pred):
                    votes[i, p] += 1
                    
            return votes
        else:
            # 软投票：概率平均
            probs = [F.softmax(output, dim=-1) for output in outputs]
            return torch.stack(probs, dim=0).mean(dim=0)


class StackingEnsemble(nn.Module):
    """堆叠集成"""
    
    def __init__(self, base_models: List[nn.Module], 
                 meta_model: nn.Module,
                 use_proba: bool = True):
        super().__init__()
        self.base_models = nn.ModuleList(base_models)
        self.meta_model = meta_model
        self.use_proba = use_proba
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 获取基模型输出
        base_outputs = []
        
        for model in self.base_models:
            output = model(x)
            if self.use_proba:
                output = F.softmax(output, dim=-1)
            base_outputs.append(output)
            
        # 拼接作为元模型输入
        meta_input = torch.cat(base_outputs, dim=-1)
        
        return self.meta_model(meta_input)


class BaggingEnsemble(nn.Module):
    """Bagging集成"""
    
    def __init__(self, model_class: type, model_args: Dict[str, Any],
                 num_models: int = 10, sample_ratio: float = 0.8):
        super().__init__()
        
        self.models = nn.ModuleList([
            model_class(**model_args) for _ in range(num_models)
        ])
        self.sample_ratio = sample_ratio
        self.num_models = num_models
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs, dim=0).mean(dim=0)
    
    def get_bootstrap_indices(self, n_samples: int) -> List[torch.Tensor]:
        """获取Bootstrap采样索引"""
        indices = []
        sample_size = int(n_samples * self.sample_ratio)
        
        for _ in range(self.num_models):
            idx = torch.randint(0, n_samples, (sample_size,))
            indices.append(idx)
            
        return indices


class BoostingEnsemble(nn.Module):
    """Boosting集成"""
    
    def __init__(self, model_class: type, model_args: Dict[str, Any],
                 num_models: int = 10, learning_rate: float = 0.1):
        super().__init__()
        
        self.models = nn.ModuleList()
        self.model_class = model_class
        self.model_args = model_args
        self.num_models = num_models
        self.learning_rate = learning_rate
        
        self.model_weights: List[float] = []
        
    def add_model(self, weight: float = 1.0):
        """添加新模型"""
        self.models.append(self.model_class(**self.model_args))
        self.model_weights.append(weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = None
        
        for weight, model in zip(self.model_weights, self.models):
            model_output = model(x)
            
            if output is None:
                output = weight * model_output
            else:
                output = output + self.learning_rate * weight * model_output
                
        return output


class SnapshotEnsemble(nn.Module):
    """快照集成"""
    
    def __init__(self, model: nn.Module, num_snapshots: int = 5):
        super().__init__()
        
        self.model = model
        self.snapshots: List[Dict[str, torch.Tensor]] = []
        self.num_snapshots = num_snapshots
        
    def save_snapshot(self):
        """保存当前模型快照"""
        if len(self.snapshots) >= self.num_snapshots:
            self.snapshots.pop(0)
            
        self.snapshots.append({
            name: param.data.clone() 
            for name, param in self.model.named_parameters()
        })
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.snapshots:
            return self.model(x)
            
        outputs = []
        original_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
        
        for snapshot in self.snapshots:
            # 加载快照
            for name, param in self.model.named_parameters():
                param.data = snapshot[name]
                
            outputs.append(self.model(x))
            
        # 恢复原始参数
        for name, param in self.model.named_parameters():
            param.data = original_params[name]
            
        return torch.stack(outputs, dim=0).mean(dim=0)


# ==================== 知识蒸馏融合 ====================

class KnowledgeDistillation(nn.Module):
    """知识蒸馏"""
    
    def __init__(self, teacher: nn.Module, student: nn.Module,
                 temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        
        # 冻结教师模型
        for param in self.teacher.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.student(x)
    
    def distillation_loss(self, student_output: torch.Tensor,
                         teacher_output: torch.Tensor,
                         targets: torch.Tensor) -> torch.Tensor:
        """计算蒸馏损失"""
        # 软标签损失
        soft_loss = F.kl_div(
            F.log_softmax(student_output / self.temperature, dim=-1),
            F.softmax(teacher_output / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # 硬标签损失
        hard_loss = F.cross_entropy(student_output, targets)
        
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


class MultiTeacherDistillation(nn.Module):
    """多教师蒸馏"""
    
    def __init__(self, teachers: List[nn.Module], student: nn.Module,
                 temperatures: Optional[List[float]] = None,
                 weights: Optional[List[float]] = None):
        super().__init__()
        
        self.teachers = nn.ModuleList(teachers)
        self.student = student
        
        if temperatures is None:
            self.temperatures = [4.0] * len(teachers)
        else:
            self.temperatures = temperatures
            
        if weights is None:
            self.weights = [1.0 / len(teachers)] * len(teachers)
        else:
            self.weights = weights
            
        # 冻结教师
        for teacher in self.teachers:
            for param in teacher.parameters():
                param.requires_grad = False
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.student(x)
    
    def distillation_loss(self, student_output: torch.Tensor,
                         targets: torch.Tensor) -> torch.Tensor:
        """计算多教师蒸馏损失"""
        total_loss = 0.0
        
        for teacher, temp, weight in zip(self.teachers, self.temperatures, self.weights):
            with torch.no_grad():
                teacher_output = teacher(x)
                
            soft_loss = F.kl_div(
                F.log_softmax(student_output / temp, dim=-1),
                F.softmax(teacher_output / temp, dim=-1),
                reduction='batchmean'
            ) * (temp ** 2)
            
            total_loss += weight * soft_loss
            
        hard_loss = F.cross_entropy(student_output, targets)
        
        return 0.5 * total_loss + 0.5 * hard_loss


class FeatureDistillation(nn.Module):
    """特征蒸馏"""
    
    def __init__(self, teacher: nn.Module, student: nn.Module,
                 feature_layers: Dict[str, str],
                 projection_dims: Optional[Dict[str, int]] = None):
        super().__init__()
        
        self.teacher = teacher
        self.student = student
        self.feature_layers = feature_layers
        
        # 投影层
        if projection_dims is None:
            self.projectors = nn.ModuleDict()
        else:
            self.projectors = nn.ModuleDict({
                name: nn.Linear(dim, dim)
                for name, dim in projection_dims.items()
            })
            
        # 冻结教师
        for param in self.teacher.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.student(x)
    
    def feature_loss(self, student_features: Dict[str, torch.Tensor],
                    teacher_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算特征蒸馏损失"""
        total_loss = 0.0
        
        for name in self.feature_layers:
            s_feat = student_features[name]
            t_feat = teacher_features[self.feature_layers[name]]
            
            if name in self.projectors:
                s_feat = self.projectors[name](s_feat)
                
            total_loss += F.mse_loss(s_feat, t_feat)
            
        return total_loss


# ==================== 模型选择 ====================

class DynamicModelSelection(nn.Module):
    """动态模型选择"""
    
    def __init__(self, models: List[nn.Module], selector: nn.Module):
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.selector = selector
        self.num_models = len(models)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 选择器输出权重
        weights = F.softmax(self.selector(x), dim=-1)
        
        # 加权组合
        outputs = torch.stack([model(x) for model in self.models], dim=0)
        
        return torch.einsum('m,mb...->b...', weights.T, outputs)


class MixtureOfExperts(nn.Module):
    """混合专家模型"""
    
    def __init__(self, experts: List[nn.Module], gate: nn.Module,
                 num_experts_per_sample: int = 2):
        super().__init__()
        
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_experts = len(experts)
        self.k = num_experts_per_sample
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # 门控输出
        gate_output = self.gate(x)
        
        # 选择top-k专家
        top_k_weights, top_k_indices = torch.topk(
            F.softmax(gate_output, dim=-1), self.k, dim=-1
        )
        
        # 归一化权重
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # 计算输出
        output = torch.zeros_like(x)
        
        for i in range(self.k):
            expert_indices = top_k_indices[:, i]
            weights = top_k_weights[:, i]
            
            for j, expert in enumerate(self.experts):
                mask = (expert_indices == j)
                if mask.any():
                    expert_output = expert(x[mask])
                    output[mask] += weights[mask].unsqueeze(-1) * expert_output
                    
        return output


class HierarchicalEnsemble(nn.Module):
    """层次集成"""
    
    def __init__(self, model_groups: List[List[nn.Module]],
                 meta_models: List[nn.Module]):
        super().__init__()
        
        self.model_groups = nn.ModuleList([
            nn.ModuleList(group) for group in model_groups
        ])
        self.meta_models = nn.ModuleList(meta_models)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 每组模型输出
        group_outputs = []
        
        for group in self.model_groups:
            outputs = [model(x) for model in group]
            group_outputs.append(torch.stack(outputs, dim=-1))
            
        # 元模型融合
        meta_input = torch.cat(group_outputs, dim=-1)
        
        output = meta_input
        for meta_model in self.meta_models:
            output = meta_model(output)
            
        return output


# ==================== 集成训练 ====================

class EnsembleTrainer:
    """集成训练器"""
    
    def __init__(self, ensemble: ModelEnsemble, 
                 criterion: nn.Module,
                 optimizer_class: type = torch.optim.Adam,
                 lr: float = 0.001):
        self.ensemble = ensemble
        self.criterion = criterion
        
        # 为每个模型创建优化器
        self.optimizers = [
            optimizer_class(model.parameters(), lr=lr)
            for model in ensemble.models
        ]
        
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """训练一步"""
        losses = []
        
        for model, optimizer in zip(self.ensemble.models, self.optimizers):
            optimizer.zero_grad()
            
            output = model(x)
            loss = self.criterion(output, y)
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
        return {
            'individual_losses': losses,
            'mean_loss': np.mean(losses)
        }
    
    def eval_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """评估一步"""
        with torch.no_grad():
            output = self.ensemble(x)
            loss = self.criterion(output, y)
            
        return {'loss': loss.item()}


class DiversityRegularizer:
    """多样性正则化"""
    
    def __init__(self, lambda_div: float = 0.1):
        self.lambda_div = lambda_div
        
    def negative_correlation(self, outputs: List[torch.Tensor],
                            targets: torch.Tensor) -> torch.Tensor:
        """负相关正则化"""
        mean_output = torch.stack(outputs, dim=0).mean(dim=0)
        
        div_loss = 0.0
        for output in outputs:
            div_loss += ((output - mean_output) * (targets - mean_output)).mean()
            
        return -self.lambda_div * div_loss
    
    def variance_regularization(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        """方差正则化"""
        mean_output = torch.stack(outputs, dim=0).mean(dim=0)
        
        var_loss = 0.0
        for output in outputs:
            var_loss += ((output - mean_output) ** 2).mean()
            
        return self.lambda_div * var_loss
