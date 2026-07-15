"""
EWC - 弹性权重巩固 (Elastic Weight Consolidation)

防止神经网络在持续学习中出现灾难性遗忘。
核心思想：根据参数对旧任务的重要性，施加不同程度的正则化约束。

本实现采用基于Fisher信息矩阵的方法，但使用独特的近似计算策略。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import copy
import math


class ParameterImportance:
    """
    参数重要性计算器
    
    使用Fisher信息矩阵的对角线近似来估计每个参数的重要性。
    """
    
    def __init__(self, model: nn.Module, damping: float = 1e-4):
        """
        初始化
        
        Args:
            model: 要分析的模型
            damping: 数值稳定性系数
        """
        self.model = model
        self.damping = damping
        self.fisher_matrix: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
        
    def compute_fisher_information(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算Fisher信息矩阵（对角近似）
        
        使用经验Fisher：E[(∇log p(y|x,θ))^2]
        
        Args:
            dataloader: 数据加载器
            num_samples: 采样数量（None表示使用全部）
        
        Returns:
            Fisher信息矩阵（对角形式）
        """
        self.model.eval()
        fisher_accumulator = {}
        
        # 初始化累积器
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_accumulator[name] = torch.zeros_like(param.data)
        
        sample_count = 0
        max_samples = num_samples or float('inf')
        
        for batch in dataloader:
            if sample_count >= max_samples:
                break
            
            # 前向传播
            inputs, targets = batch[0], batch[1]
            inputs = inputs.to(next(self.model.parameters()).device)
            targets = targets.to(next(self.model.parameters()).device)
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            
            # 计算对数似然的梯度
            # 使用预测概率的平方作为Fisher的近似
            probs = F.softmax(outputs, dim=-1)
            log_probs = F.log_softmax(outputs, dim=-1)
            
            # 对每个样本计算梯度并累积平方
            for i in range(inputs.size(0)):
                self.model.zero_grad()
                # 使用负对数似然
                loss = -log_probs[i, targets[i]]
                loss.backward(retain_graph=True)
                
                # 累积梯度平方
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        fisher_accumulator[name] += param.grad.data ** 2
                
                sample_count += 1
                if sample_count >= max_samples:
                    break
        
        # 平均并添加阻尼
        for name in fisher_accumulator:
            fisher_accumulator[name] = fisher_accumulator[name] / sample_count
            fisher_accumulator[name] = fisher_accumulator[name] + self.damping
        
        self.fisher_matrix = fisher_accumulator
        return fisher_accumulator
    
    def save_optimal_params(self):
        """保存当前参数作为最优参数"""
        self.optimal_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
    
    def get_importance_score(self, param_name: str) -> torch.Tensor:
        """获取指定参数的重要性分数"""
        if param_name not in self.fisher_matrix:
            raise ValueError(f"参数 {param_name} 的Fisher信息未计算")
        return self.fisher_matrix[param_name]


class ElasticWeightConsolidation:
    """
    弹性权重巩固主类
    
    通过结合当前任务损失和EWC正则化损失，实现终身学习。
    """
    
    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = 1e4,
        damping: float = 1e-4,
        normalize: bool = True
    ):
        """
        初始化EWC
        
        Args:
            model: 要保护的模型
            lambda_ewc: EWC正则化强度
            damping: Fisher矩阵阻尼系数
            normalize: 是否归一化Fisher矩阵
        """
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.damping = damping
        self.normalize = normalize
        
        # 存储每个任务的Fisher矩阵和最优参数
        self.task_fishers: List[Dict[str, torch.Tensor]] = []
        self.task_params: List[Dict[str, torch.Tensor]] = []
        self.task_names: List[str] = []
        
        self.importance_calculator = ParameterImportance(model, damping)
        
    def update_task_importance(
        self,
        dataloader: torch.utils.data.DataLoader,
        task_name: str,
        num_samples: Optional[int] = None
    ):
        """
        更新任务重要性
        
        在任务训练完成后调用，保存该任务的Fisher信息和最优参数。
        
        Args:
            dataloader: 任务数据加载器
            task_name: 任务名称
            num_samples: 采样数量
        """
        print(f"计算任务 '{task_name}' 的参数重要性...")
        
        # 计算Fisher信息
        fisher = self.importance_calculator.compute_fisher_information(
            dataloader, num_samples
        )
        
        # 保存最优参数
        optimal_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        
        # 存储
        self.task_fishers.append(fisher)
        self.task_params.append(optimal_params)
        self.task_names.append(task_name)
        
        print(f"任务 '{task_name}' 的重要性已保存")
    
    def compute_ewc_loss(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算EWC正则化损失
        
        Returns:
            ewc_loss: EWC损失
            stats: 统计信息
        """
        if len(self.task_fishers) == 0:
            return torch.tensor(0.0, device=next(self.model.parameters()).device), {}
        
        total_loss = 0.0
        per_task_losses = {}
        
        for task_idx, (fisher, old_params, task_name) in enumerate(
            zip(self.task_fishers, self.task_params, self.task_names)
        ):
            task_loss = 0.0
            param_count = 0
            
            for name, param in self.model.named_parameters():
                if name not in fisher or name not in old_params:
                    continue
                
                # EWC损失：Fisher * (θ - θ*)^2
                fisher_diag = fisher[name]
                param_diff = param - old_params[name]
                
                # 可选：归一化Fisher
                if self.normalize:
                    fisher_diag = fisher_diag / (fisher_diag.mean() + 1e-8)
                
                loss_contribution = (fisher_diag * param_diff ** 2).sum()
                task_loss += loss_contribution
                param_count += param.numel()
            
            total_loss += task_loss
            per_task_losses[f'ewc_loss_{task_name}'] = task_loss.item() / (param_count + 1e-8)
        
        # 应用正则化强度
        ewc_loss = self.lambda_ewc * total_loss
        
        stats = {
            'ewc_total_loss': ewc_loss.item(),
            'num_tasks': len(self.task_fishers),
            **per_task_losses
        }
        
        return ewc_loss, stats
    
    def compute_total_loss(
        self,
        current_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算总损失（当前任务损失 + EWC损失）
        
        Args:
            current_loss: 当前任务的损失
        
        Returns:
            total_loss: 总损失
            stats: 统计信息
        """
        ewc_loss, ewc_stats = self.compute_ewc_loss()
        total_loss = current_loss + ewc_loss
        
        stats = {
            'total_loss': total_loss.item(),
            'current_task_loss': current_loss.item(),
            **ewc_stats
        }
        
        return total_loss, stats
    
    def get_parameter_drift(self, task_name: Optional[str] = None) -> Dict[str, float]:
        """
        计算参数漂移
        
        Args:
            task_name: 指定任务（None表示所有任务）
        
        Returns:
            每个参数的漂移程度
        """
        drift_stats = {}
        
        if task_name is not None:
            if task_name not in self.task_names:
                raise ValueError(f"未知任务: {task_name}")
            task_idx = self.task_names.index(task_name)
            tasks_to_check = [(task_idx, task_name)]
        else:
            tasks_to_check = enumerate(self.task_names)
        
        for task_idx, name in tasks_to_check:
            old_params = self.task_params[task_idx]
            
            for param_name, param in self.model.named_parameters():
                if param_name not in old_params:
                    continue
                
                drift = torch.norm(param - old_params[param_name]).item()
                relative_drift = drift / (torch.norm(old_params[param_name]).item() + 1e-8)
                
                drift_stats[f'{name}_{param_name}_drift'] = relative_drift
        
        return drift_stats
    
    def estimate_forgetting(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        估计遗忘程度
        
        通过在新数据上测试旧任务的性能来估计遗忘。
        
        Args:
            dataloader: 测试数据加载器
        
        Returns:
            各任务的遗忘指标
        """
        # 这里简化处理，实际应该加载各任务的测试集
        forgetting_stats = {}
        
        for task_name in self.task_names:
            # 计算该任务的参数漂移作为遗忘指标
            drift = self.get_parameter_drift(task_name)
            avg_drift = sum(drift.values()) / len(drift) if drift else 0.0
            forgetting_stats[f'forgetting_{task_name}'] = avg_drift
        
        return forgetting_stats


class OnlineEWC(ElasticWeightConsolidation):
    """
    在线EWC
    
    使用指数移动平均更新Fisher信息，适用于在线学习场景。
    """
    
    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = 1e4,
        gamma: float = 0.95,
        damping: float = 1e-4
    ):
        """
        初始化在线EWC
        
        Args:
            model: 模型
            lambda_ewc: EWC正则化强度
            gamma: 指数移动平均系数
            damping: 阻尼系数
        """
        super().__init__(model, lambda_ewc, damping)
        self.gamma = gamma
        self.online_fisher: Optional[Dict[str, torch.Tensor]] = None
        
    def update_online_fisher(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: Optional[int] = None
    ):
        """
        更新在线Fisher信息
        
        使用指数移动平均合并新旧Fisher信息。
        """
        # 计算当前任务的Fisher
        new_fisher = self.importance_calculator.compute_fisher_information(
            dataloader, num_samples
        )
        
        if self.online_fisher is None:
            # 首次更新
            self.online_fisher = new_fisher
        else:
            # 指数移动平均更新
            for name in self.online_fisher:
                if name in new_fisher:
                    self.online_fisher[name] = (
                        self.gamma * self.online_fisher[name] +
                        (1 - self.gamma) * new_fisher[name]
                    )
        
        # 保存当前参数
        self.optimal_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
    
    def compute_ewc_loss(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算在线EWC损失"""
        if self.online_fisher is None or not self.optimal_params:
            return torch.tensor(0.0, device=next(self.model.parameters()).device), {}
        
        total_loss = 0.0
        param_count = 0
        
        for name, param in self.model.named_parameters():
            if name not in self.online_fisher or name not in self.optimal_params:
                continue
            
            fisher_diag = self.online_fisher[name]
            param_diff = param - self.optimal_params[name]
            
            loss_contribution = (fisher_diag * param_diff ** 2).sum()
            total_loss += loss_contribution
            param_count += param.numel()
        
        ewc_loss = self.lambda_ewc * total_loss
        
        stats = {
            'online_ewc_loss': ewc_loss.item(),
            'gamma': self.gamma
        }
        
        return ewc_loss, stats


# 便捷函数
def apply_ewc_regularization(
    model: nn.Module,
    current_loss: torch.Tensor,
    task_dataloaders: List[torch.utils.data.DataLoader],
    lambda_ewc: float = 1e4
) -> Tuple[torch.Tensor, ElasticWeightConsolidation]:
    """
    便捷函数：应用EWC正则化
    
    Args:
        model: 模型
        current_loss: 当前损失
        task_dataloaders: 各任务的数据加载器
        lambda_ewc: EWC强度
    
    Returns:
        total_loss: 总损失
        ewc: EWC实例
    """
    ewc = ElasticWeightConsolidation(model, lambda_ewc)
    
    # 为每个历史任务计算重要性
    for i, dataloader in enumerate(task_dataloaders[:-1]):
        ewc.update_task_importance(dataloader, f'task_{i}')
    
    total_loss, stats = ewc.compute_total_loss(current_loss)
    return total_loss, ewc
