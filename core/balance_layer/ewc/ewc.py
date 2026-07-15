"""
EWC - Elastic Weight Consolidation

防止神经网络在持续学习中遗忘先前任务的关键算法。
通过Fisher信息矩阵估计参数重要性，对重要参数施加约束。

核心功能：
1. Fisher信息矩阵计算（对角近似）
2. 多任务参数重要性累积
3. EWC正则化损失计算
4. 参数重要性可视化

参考论文：
"Overcoming catastrophic forgetting in neural networks" (Kirkpatrick et al., 2017)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import copy
import numpy as np
from dataclasses import dataclass


@dataclass
class EWCConfig:
    """EWC配置类"""
    ewc_lambda: float = 1e4          # EWC正则化强度
    normalize_fisher: bool = True     # 是否归一化Fisher矩阵
    fisher_samples: Optional[int] = None  # Fisher计算采样数
    online_ewc: bool = False          # 是否使用在线EWC
    gamma: float = 1.0                # 在线EWC衰减因子


class EWCLoss(nn.Module):
    """
    EWC损失函数
    
    在标准损失上添加Fisher信息加权的参数漂移惩罚。
    惩罚项阻止模型在学习新任务时大幅改变对旧任务重要的参数。
    
    数学公式：
    L_EWC = L_new + λ/2 * Σ_i F_i * (θ_i - θ*_i)^2
    
    其中：
    - L_new: 新任务的损失
    - λ: EWC正则化强度
    - F_i: 参数i的Fisher信息
    - θ_i: 当前参数值
    - θ*_i: 旧任务的最优参数值
    """
    
    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 1e4,
        normalize_fisher: bool = True,
        online: bool = False,
        gamma: float = 1.0
    ):
        """
        初始化EWC损失
        
        Args:
            model: 要保护的模型
            ewc_lambda: EWC正则化强度（默认1e4）
            normalize_fisher: 是否归一化Fisher矩阵
            online: 是否使用在线EWC（累积所有任务的Fisher）
            gamma: 在线EWC的衰减因子
        """
        super().__init__()
        self.ewc_lambda = ewc_lambda
        self.normalize_fisher = normalize_fisher
        self.online = online
        self.gamma = gamma
        
        # 存储每个任务的Fisher信息和最优参数
        self.task_fishers: List[Dict[str, torch.Tensor]] = []
        self.task_params: List[Dict[str, torch.Tensor]] = []
        self.task_names: List[str] = []
        
        # 在线EWC的累积Fisher
        self.online_fisher: Optional[Dict[str, torch.Tensor]] = None
        
        # 模型参数名称
        self.param_names = [name for name, _ in model.named_parameters() if _.requires_grad]
    
    def compute_fisher_information(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        num_samples: Optional[int] = None,
        task_type: str = "classification"
    ) -> Dict[str, torch.Tensor]:
        """
        计算Fisher信息矩阵（对角近似）
        
        Fisher信息衡量了似然函数对参数的敏感度，
        可以看作参数对任务重要性的估计。
        
        数学上，Fisher信息是损失函数对参数的二阶导数的期望：
        F_ij = E[∂²L/∂θ_i∂θ_j]
        
        我们使用对角近似：只计算F_ii（每个参数的重要性）
        
        Args:
            model: 模型
            dataloader: 数据加载器
            num_samples: 采样数量（None表示使用全部）
            task_type: 任务类型（"classification"或"regression"）
        
        Returns:
            每个参数的Fisher信息字典
        """
        model.eval()
        fisher_information = {}
        
        # 初始化Fisher信息
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_information[name] = torch.zeros_like(param)
        
        sample_count = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if num_samples is not None and total_samples >= num_samples:
                break
            
            # 解包批次数据
            if len(batch) == 2:
                inputs, targets = batch
            elif len(batch) == 3:
                inputs, targets, _ = batch
            else:
                inputs = batch[0]
                targets = None
            
            # 移动到设备
            device = next(model.parameters()).device
            inputs = inputs.to(device)
            if targets is not None:
                targets = targets.to(device)
            
            batch_size = inputs.size(0)
            
            model.zero_grad()
            outputs = model(inputs)
            
            # 根据任务类型计算Fisher信息
            if task_type == "classification":
                fisher_information = self._compute_fisher_classification(
                    model, outputs, fisher_information
                )
            elif task_type == "regression":
                fisher_information = self._compute_fisher_regression(
                    model, outputs, targets, fisher_information
                )
            else:
                raise ValueError(f"不支持的任务类型: {task_type}")
            
            sample_count += batch_size
            total_samples += batch_size
            
            # 限制样本数量
            if num_samples is not None and total_samples >= num_samples:
                break
        
        # 平均Fisher信息
        if sample_count > 0:
            for name in fisher_information:
                fisher_information[name] /= sample_count
        
        # 可选：归一化
        if self.normalize_fisher:
            total_fisher = sum(fisher.sum() for fisher in fisher_information.values())
            if total_fisher > 0:
                for name in fisher_information:
                    fisher_information[name] /= total_fisher
        
        return fisher_information
    
    def _compute_fisher_classification(
        self,
        model: nn.Module,
        outputs: torch.Tensor,
        fisher_information: Dict[str, torch.Tensor],
        num_classes_to_sample: int = 10
    ) -> Dict[str, torch.Tensor]:
        """
        计算分类任务的Fisher信息
        
        对于分类任务，Fisher信息通过对所有类别的预测概率加权计算：
        F = Σ_c p(c|x) * (∂log p(c|x)/∂θ)^2
        
        Args:
            model: 模型
            outputs: 模型输出（logits）
            fisher_information: 当前的Fisher信息字典
            num_classes_to_sample: 采样的类别数量（用于加速）
        
        Returns:
            更新后的Fisher信息字典
        """
        batch_size = outputs.size(0)
        
        # 多分类：使用每个类别的概率
        if outputs.dim() > 1 and outputs.size(-1) > 1:
            probs = F.softmax(outputs, dim=-1)
            num_classes = outputs.size(-1)
            
            # 对每个类别采样计算梯度
            classes_to_sample = min(num_classes_to_sample, num_classes)
            
            for class_idx in range(classes_to_sample):
                model.zero_grad()
                
                # 计算该类别的对数概率
                log_probs = F.log_softmax(outputs, dim=-1)
                class_log_probs = log_probs[:, class_idx].mean()
                
                # 反向传播
                class_log_probs.backward(retain_graph=True)
                
                # 累积梯度平方（Fisher信息）
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        # 加权：p(c) * gradient^2
                        weight = probs[:, class_idx].mean().item()
                        fisher_information[name] += param.grad.pow(2) * weight
        else:
            # 二分类
            model.zero_grad()
            probs = torch.sigmoid(outputs)
            loss = F.binary_cross_entropy_with_logits(
                outputs, probs.detach(), reduction='mean'
            )
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_information[name] += param.grad.pow(2)
        
        return fisher_information
    
    def _compute_fisher_regression(
        self,
        model: nn.Module,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        fisher_information: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        计算回归任务的Fisher信息
        
        对于回归任务，直接使用MSE损失的梯度：
        F = (∂L/∂θ)^2
        
        Args:
            model: 模型
            outputs: 模型输出
            targets: 目标值
            fisher_information: 当前的Fisher信息字典
        
        Returns:
            更新后的Fisher信息字典
        """
        model.zero_grad()
        
        # 回归任务：直接使用MSE损失
        loss = F.mse_loss(outputs, targets.float())
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_information[name] += param.grad.pow(2)
        
        return fisher_information
    
    def update_task(
        self,
        model: nn.Module,
        task_name: str,
        dataloader: torch.utils.data.DataLoader,
        task_type: str = "classification"
    ) -> Dict[str, Any]:
        """
        更新任务（学习完一个任务后调用）
        
        计算当前任务的Fisher信息并保存最优参数。
        
        Args:
            model: 训练好的模型
            task_name: 任务名称
            dataloader: 任务数据加载器
            task_type: 任务类型
        
        Returns:
            统计信息字典
        """
        model.eval()
        
        # 计算当前任务的Fisher信息
        fisher = self.compute_fisher_information(
            model, dataloader, task_type=task_type
        )
        
        # 保存当前任务的最优参数
        optimal_params = {
            name: param.data.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        
        if self.online:
            # 在线EWC：累积Fisher信息
            if self.online_fisher is None:
                self.online_fisher = {name: torch.zeros_like(fisher[name]) 
                                      for name in fisher}
            
            # 衰减旧的Fisher信息并添加新的
            for name in fisher:
                self.online_fisher[name] = (
                    self.gamma * self.online_fisher[name] + fisher[name]
                )
            
            # 在线EWC只保存一个任务
            self.task_fishers = [self.online_fisher]
            self.task_params = [optimal_params]
            self.task_names = ["online_ewc"]
        else:
            # 标准EWC：保存每个任务的Fisher信息
            self.task_fishers.append(fisher)
            self.task_params.append(optimal_params)
            self.task_names.append(task_name)
        
        # 计算统计信息
        total_fisher_norm = sum(f.norm().item() for f in fisher.values())
        num_params = len(fisher)
        
        stats = {
            'task_name': task_name,
            'num_params': num_params,
            'total_fisher_norm': total_fisher_norm,
            'mean_fisher_norm': total_fisher_norm / num_params if num_params > 0 else 0
        }
        
        print(f"EWC: 已学习任务 '{task_name}'，参数数量: {num_params}, "
              f"Fisher范数: {total_fisher_norm:.4f}")
        
        return stats
    
    def forward(self, model: nn.Module, base_loss: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        计算EWC损失
        
        L_total = L_base + λ/2 * Σ_tasks Σ_i F_i * (θ_i - θ*_i)^2
        
        Args:
            model: 当前模型
            base_loss: 基础任务损失
        
        Returns:
            total_loss: 总损失（基础损失 + EWC惩罚）
            stats: 统计信息字典
        """
        if len(self.task_fishers) == 0:
            # 没有先前任务，只返回基础损失
            return base_loss, {
                'ewc_loss': 0.0,
                'base_loss': base_loss.item(),
                'total_loss': base_loss.item()
            }
        
        ewc_loss = 0.0
        device = next(model.parameters()).device
        
        # 对每个先前任务计算惩罚
        for task_idx, (fisher, old_params) in enumerate(zip(self.task_fishers, self.task_params)):
            task_penalty = 0.0
            
            for name, param in model.named_parameters():
                if name in fisher and name in old_params:
                    # EWC惩罚项: F_i * (theta_i - theta_old_i)^2
                    param_diff = param - old_params[name].to(device)
                    fisher_values = fisher[name].to(device)
                    
                    # 防止Fisher信息过小导致数值不稳定
                    fisher_values = torch.clamp(fisher_values, min=1e-8)
                    
                    task_penalty += (fisher_values * param_diff.pow(2)).sum()
            
            ewc_loss += task_penalty
        
        # 应用EWC正则化强度
        ewc_loss = self.ewc_lambda * ewc_loss / 2.0
        total_loss = base_loss + ewc_loss
        
        stats = {
            'total_loss': total_loss.item(),
            'base_loss': base_loss.item(),
            'ewc_loss': ewc_loss.item(),
            'ewc_lambda': self.ewc_lambda,
            'num_tasks': len(self.task_fishers)
        }
        
        return total_loss, stats
    
    def get_importance_scores(self, model: nn.Module) -> Dict[str, float]:
        """
        获取参数重要性分数
        
        返回每个参数的平均Fisher信息作为重要性分数。
        
        Args:
            model: 模型
        
        Returns:
            每个参数的重要性分数字典
        """
        if len(self.task_fishers) == 0:
            return {}
        
        importance_scores = {}
        
        for name, param in model.named_parameters():
            if name in self.task_fishers[0]:
                # 计算所有任务中该参数的平均Fisher信息
                total_fisher = sum(
                    fisher[name].abs().mean().item() 
                    for fisher in self.task_fishers
                )
                importance_scores[name] = total_fisher / len(self.task_fishers)
        
        return importance_scores
    
    def get_top_important_params(self, model: nn.Module, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        获取最重要的top_k参数
        
        Args:
            model: 模型
            top_k: 返回的参数数量
        
        Returns:
            (参数名, 重要性分数)的列表，按重要性降序排列
        """
        importance_scores = self.get_importance_scores(model)
        sorted_params = sorted(
            importance_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_params[:top_k]
    
    def save_state(self, path: str) -> None:
        """
        保存EWC状态
        
        Args:
            path: 保存路径
        """
        state = {
            'task_fishers': self.task_fishers,
            'task_params': self.task_params,
            'task_names': self.task_names,
            'ewc_lambda': self.ewc_lambda,
            'normalize_fisher': self.normalize_fisher,
            'online': self.online,
            'gamma': self.gamma
        }
        torch.save(state, path)
        print(f"EWC状态已保存到: {path}")
    
    def load_state(self, path: str) -> None:
        """
        加载EWC状态
        
        Args:
            path: 加载路径
        """
        state = torch.load(path)
        self.task_fishers = state['task_fishers']
        self.task_params = state['task_params']
        self.task_names = state['task_names']
        self.ewc_lambda = state.get('ewc_lambda', self.ewc_lambda)
        self.normalize_fisher = state.get('normalize_fisher', self.normalize_fisher)
        self.online = state.get('online', self.online)
        self.gamma = state.get('gamma', self.gamma)
        print(f"EWC状态已从 {path} 加载，任务数: {len(self.task_names)}")
    
    def reset(self) -> None:
        """重置EWC状态，清除所有任务"""
        self.task_fishers = []
        self.task_params = []
        self.task_names = []
        self.online_fisher = None
        print("EWC状态已重置")


class EWCPlugin:
    """
    EWC训练插件
    
    集成到训练循环中的EWC插件，自动在训练后更新Fisher信息。
    """
    
    def __init__(
        self,
        ewc_lambda: float = 1e4,
        normalize_fisher: bool = True,
        update_frequency: str = "epoch"  # "epoch" 或 "step"
    ):
        """
        初始化EWC插件
        
        Args:
            ewc_lambda: EWC正则化强度
            normalize_fisher: 是否归一化Fisher矩阵
            update_frequency: 更新频率
        """
        self.ewc_lambda = ewc_lambda
        self.normalize_fisher = normalize_fisher
        self.update_frequency = update_frequency
        self.ewc_loss: Optional[EWCLoss] = None
        self.current_task_name: Optional[str] = None
    
    def on_task_start(self, model: nn.Module, task_name: str) -> None:
        """
        任务开始时的回调
        
        Args:
            model: 模型
            task_name: 任务名称
        """
        self.current_task_name = task_name
        
        if self.ewc_loss is None:
            self.ewc_loss = EWCLoss(
                model,
                ewc_lambda=self.ewc_lambda,
                normalize_fisher=self.normalize_fisher
            )
        
        print(f"EWC插件: 开始任务 '{task_name}'")
    
    def on_task_end(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        task_type: str = "classification"
    ) -> Dict[str, Any]:
        """
        任务结束时的回调
        
        Args:
            model: 模型
            dataloader: 数据加载器
            task_type: 任务类型
        
        Returns:
            统计信息
        """
        if self.ewc_loss is None or self.current_task_name is None:
            raise RuntimeError("必须先调用on_task_start")
        
        stats = self.ewc_loss.update_task(
            model, self.current_task_name, dataloader, task_type
        )
        
        return stats
    
    def compute_loss(self, model: nn.Module, base_loss: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        计算带EWC惩罚的损失
        
        Args:
            model: 模型
            base_loss: 基础损失
        
        Returns:
            总损失和统计信息
        """
        if self.ewc_loss is None:
            return base_loss, {'ewc_loss': 0.0, 'base_loss': base_loss.item()}
        
        return self.ewc_loss(model, base_loss)


# 便捷函数
def create_ewc_loss(
    model: nn.Module,
    ewc_lambda: float = 1e4,
    normalize_fisher: bool = True
) -> EWCLoss:
    """
    创建EWC损失函数
    
    Args:
        model: 要保护的模型
        ewc_lambda: EWC正则化强度
        normalize_fisher: 是否归一化Fisher矩阵
    
    Returns:
        EWC损失函数实例
    """
    return EWCLoss(model, ewc_lambda, normalize_fisher)


def compute_fisher_information(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_samples: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """
    便捷函数：计算Fisher信息
    
    Args:
        model: 模型
        dataloader: 数据加载器
        num_samples: 采样数量
    
    Returns:
        Fisher信息字典
    """
    ewc = EWCLoss(model)
    return ewc.compute_fisher_information(model, dataloader, num_samples)
