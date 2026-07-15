"""
参数固化冷记忆模块 (Cold Memory Module)

该模块实现了神经网络参数的固化存储和管理机制，支持参数快照保存、
参数冻结、渐进式解冻、Fisher信息计算以及知识蒸馏等功能。

核心功能:
- 参数快照保存
- 参数冻结机制
- 渐进式解冻
- 参数重要性估计 (Fisher信息)
- 知识蒸馏接口

作者: AGI Universal Framework Team
版本: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import json
import os
import pickle
import threading
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import copy
import warnings
from datetime import datetime


@dataclass
class ParameterSnapshot:
    """
    参数快照数据类
    
    Attributes:
        name: 快照名称
        timestamp: 创建时间戳
        parameters: 参数字典
        metadata: 元数据
        description: 描述
    """
    name: str
    timestamp: float
    parameters: Dict[str, torch.Tensor]
    metadata: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        return {
            'name': self.name,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'description': self.description,
            'parameter_keys': list(self.parameters.keys())
        }


@dataclass
class ParameterImportance:
    """
    参数重要性数据类
    
    Attributes:
        name: 参数名称
        fisher_information: Fisher信息值
        importance_score: 重要性分数
        frozen: 是否已冻结
    """
    name: str
    fisher_information: torch.Tensor
    importance_score: float = 0.0
    frozen: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'fisher_information': self.fisher_information.cpu().numpy().tolist(),
            'importance_score': self.importance_score,
            'frozen': self.frozen
        }


class ColdMemory:
    """
    参数固化冷记忆
    
    管理神经网络参数的快照、冻结和知识蒸馏。
    支持Fisher信息计算来估计参数重要性。
    
    Attributes:
        snapshots: 参数快照字典
        frozen_parameters: 冻结的参数集合
        importance_scores: 参数重要性分数
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        初始化参数固化冷记忆
        
        Args:
            storage_path: 存储路径
        """
        self.storage_path = storage_path
        self.snapshots: Dict[str, ParameterSnapshot] = {}
        self.frozen_parameters: Dict[str, Set[str]] = defaultdict(set)
        self.importance_scores: Dict[str, Dict[str, ParameterImportance]] = defaultdict(dict)
        self._parameter_masks: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        
        # 统计信息
        self._snapshot_count = 0
        self._freeze_count = 0
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 加载已有数据
        if storage_path and os.path.exists(storage_path):
            self._load_from_disk()
    
    def freeze_parameters(
        self,
        model: nn.Module,
        importance_threshold: Optional[float] = None,
        parameter_names: Optional[List[str]] = None,
        model_name: str = "default"
    ) -> Set[str]:
        """
        冻结模型参数
        
        Args:
            model: 神经网络模型
            importance_threshold: 重要性阈值（如果提供，只冻结重要性高于阈值的参数）
            parameter_names: 要冻结的参数名称列表（如果为None，冻结所有参数）
            model_name: 模型名称
            
        Returns:
            被冻结的参数名称集合
        """
        with self._lock:
            frozen_params = set()
            
            for name, param in model.named_parameters():
                # 检查是否应该冻结此参数
                should_freeze = False
                
                if parameter_names is not None:
                    if name in parameter_names:
                        should_freeze = True
                elif importance_threshold is not None:
                    if model_name in self.importance_scores:
                        if name in self.importance_scores[model_name]:
                            importance = self.importance_scores[model_name][name].importance_score
                            if importance >= importance_threshold:
                                should_freeze = True
                else:
                    should_freeze = True
                
                if should_freeze:
                    param.requires_grad = False
                    self.frozen_parameters[model_name].add(name)
                    frozen_params.add(name)
                    
                    # 创建参数掩码
                    if name not in self._parameter_masks[model_name]:
                        self._parameter_masks[model_name][name] = torch.ones_like(param.data)
            
            self._freeze_count += len(frozen_params)
            return frozen_params
    
    def unfreeze_parameters(
        self,
        model: nn.Module,
        parameter_names: Optional[List[str]] = None,
        model_name: str = "default"
    ) -> Set[str]:
        """
        解冻模型参数
        
        Args:
            model: 神经网络模型
            parameter_names: 要解冻的参数名称列表（如果为None，解冻所有冻结的参数）
            model_name: 模型名称
            
        Returns:
            被解冻的参数名称集合
        """
        with self._lock:
            unfrozen_params = set()
            
            if parameter_names is None:
                parameter_names = list(self.frozen_parameters[model_name])
            
            for name, param in model.named_parameters():
                if name in parameter_names and name in self.frozen_parameters[model_name]:
                    param.requires_grad = True
                    self.frozen_parameters[model_name].discard(name)
                    unfrozen_params.add(name)
            
            return unfrozen_params
    
    def progressive_unfreeze(
        self,
        model: nn.Module,
        schedule: List[Dict[str, Any]],
        model_name: str = "default"
    ) -> Dict[int, Set[str]]:
        """
        渐进式解冻参数
        
        按照指定的时间表逐步解冻参数，常用于迁移学习中的微调。
        
        Args:
            model: 神经网络模型
            schedule: 解冻计划，每个元素是一个字典，包含：
                - epoch: 解冻的epoch
                - layers: 要解冻的层名称列表或模式
                - lr_multiplier: 学习率乘数（可选）
            model_name: 模型名称
            
        Returns:
            每个epoch解冻的参数集合
        """
        with self._lock:
            unfreeze_plan = {}
            
            for item in schedule:
                epoch = item['epoch']
                layers = item.get('layers', [])
                
                params_to_unfreeze = set()
                
                for name, param in model.named_parameters():
                    # 检查参数是否匹配任何指定的层模式
                    for layer_pattern in layers:
                        if layer_pattern in name:
                            if name in self.frozen_parameters[model_name]:
                                params_to_unfreeze.add(name)
                            break
                
                unfreeze_plan[epoch] = params_to_unfreeze
            
            return unfreeze_plan
    
    def apply_unfreeze_schedule(
        self,
        model: nn.Module,
        current_epoch: int,
        schedule: List[Dict[str, Any]],
        model_name: str = "default"
    ) -> Set[str]:
        """
        应用解冻计划
        
        Args:
            model: 神经网络模型
            current_epoch: 当前epoch
            schedule: 解冻计划
            model_name: 模型名称
            
        Returns:
            本次解冻的参数集合
        """
        with self._lock:
            unfrozen_now = set()
            
            for item in schedule:
                if item['epoch'] == current_epoch:
                    layers = item.get('layers', [])
                    
                    for name, param in model.named_parameters():
                        for layer_pattern in layers:
                            if layer_pattern in name:
                                if name in self.frozen_parameters[model_name]:
                                    param.requires_grad = True
                                    self.frozen_parameters[model_name].discard(name)
                                    unfrozen_now.add(name)
                                break
            
            return unfrozen_now
    
    def save_snapshot(
        self,
        name: str,
        model: nn.Module,
        metadata: Optional[Dict[str, Any]] = None,
        description: str = "",
        include_optimizer: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> bool:
        """
        保存参数快照
        
        Args:
            name: 快照名称
            model: 神经网络模型
            metadata: 元数据
            description: 描述
            include_optimizer: 是否包含优化器状态
            optimizer: 优化器（如果include_optimizer为True）
            
        Returns:
            是否成功保存
        """
        with self._lock:
            try:
                # 收集参数
                parameters = {}
                for param_name, param in model.named_parameters():
                    parameters[param_name] = param.data.cpu().clone()
                
                # 收集缓冲区（如BN的running_mean）
                buffer_dict = {}
                for buffer_name, buffer in model.named_buffers():
                    buffer_dict[buffer_name] = buffer.data.cpu().clone()
                
                snapshot_data = {
                    'parameters': parameters,
                    'buffers': buffer_dict
                }
                
                # 保存优化器状态
                if include_optimizer and optimizer is not None:
                    snapshot_data['optimizer_state'] = optimizer.state_dict()
                
                # 创建快照
                snapshot = ParameterSnapshot(
                    name=name,
                    timestamp=datetime.now().timestamp(),
                    parameters=snapshot_data,
                    metadata=metadata or {},
                    description=description
                )
                
                self.snapshots[name] = snapshot
                self._snapshot_count += 1
                
                # 保存到磁盘
                if self.storage_path:
                    self._save_to_disk()
                
                return True
            except Exception as e:
                warnings.warn(f"Failed to save snapshot: {e}")
                return False
    
    def load_snapshot(
        self,
        name: str,
        model: nn.Module,
        load_optimizer: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
        strict: bool = True
    ) -> bool:
        """
        加载参数快照
        
        Args:
            name: 快照名称
            model: 神经网络模型
            load_optimizer: 是否加载优化器状态
            optimizer: 优化器（如果load_optimizer为True）
            strict: 是否严格匹配参数
            
        Returns:
            是否成功加载
        """
        with self._lock:
            if name not in self.snapshots:
                warnings.warn(f"Snapshot '{name}' not found")
                return False
            
            try:
                snapshot = self.snapshots[name]
                snapshot_data = snapshot.parameters
                
                # 加载参数
                if 'parameters' in snapshot_data:
                    param_dict = snapshot_data['parameters']
                    
                    for param_name, param in model.named_parameters():
                        if param_name in param_dict:
                            param.data.copy_(param_dict[param_name])
                        elif strict:
                            warnings.warn(f"Parameter '{param_name}' not found in snapshot")
                
                # 加载缓冲区
                if 'buffers' in snapshot_data:
                    buffer_dict = snapshot_data['buffers']
                    
                    for buffer_name, buffer in model.named_buffers():
                        if buffer_name in buffer_dict:
                            buffer.data.copy_(buffer_dict[buffer_name])
                
                # 加载优化器状态
                if load_optimizer and optimizer is not None and 'optimizer_state' in snapshot_data:
                    optimizer.load_state_dict(snapshot_data['optimizer_state'])
                
                return True
            except Exception as e:
                warnings.warn(f"Failed to load snapshot: {e}")
                return False
    
    def delete_snapshot(self, name: str) -> bool:
        """
        删除快照
        
        Args:
            name: 快照名称
            
        Returns:
            是否成功删除
        """
        with self._lock:
            if name not in self.snapshots:
                return False
            
            del self.snapshots[name]
            self._snapshot_count -= 1
            
            if self.storage_path:
                self._save_to_disk()
            
            return True
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """
        列出所有快照
        
        Returns:
            快照信息列表
        """
        with self._lock:
            return [snapshot.to_dict() for snapshot in self.snapshots.values()]
    
    def compute_fisher_information(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        num_samples: Optional[int] = None,
        model_name: str = "default"
    ) -> Dict[str, torch.Tensor]:
        """
        计算Fisher信息矩阵的对角线
        
        Fisher信息反映了参数对模型输出的敏感度，可用于估计参数重要性。
        
        Args:
            model: 神经网络模型
            data_loader: 数据加载器
            num_samples: 用于计算的样本数（None表示使用所有样本）
            model_name: 模型名称
            
        Returns:
            参数名称到Fisher信息的映射
        """
        with self._lock:
            model.eval()
            fisher_information = defaultdict(lambda: torch.zeros_like)
            
            # 初始化Fisher信息
            for name, param in model.named_parameters():
                fisher_information[name] = torch.zeros_like(param.data)
            
            sample_count = 0
            
            for batch in data_loader:
                if num_samples is not None and sample_count >= num_samples:
                    break
                
                # 准备输入
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                    labels = batch[1] if len(batch) > 1 else None
                else:
                    inputs = batch
                    labels = None
                
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(next(model.parameters()).device)
                
                # 前向传播
                model.zero_grad()
                outputs = model(inputs)
                
                # 计算对数概率
                if isinstance(outputs, torch.Tensor):
                    if outputs.dim() > 1 and outputs.shape[-1] > 1:
                        # 分类任务
                        log_probs = F.log_softmax(outputs, dim=-1)
                        # 采样一个类别
                        sampled_targets = torch.multinomial(
                            F.softmax(outputs, dim=-1), 1
                        ).squeeze()
                        loss = F.nll_loss(log_probs, sampled_targets)
                    else:
                        # 回归任务
                        loss = outputs.sum()
                else:
                    continue
                
                # 反向传播
                loss.backward()
                
                # 累积Fisher信息
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        fisher_information[name] += param.grad.data ** 2
                
                sample_count += inputs.size(0)
            
            # 平均Fisher信息
            for name in fisher_information:
                fisher_information[name] /= sample_count
            
            # 更新重要性分数
            self._update_importance_scores(model_name, fisher_information)
            
            return dict(fisher_information)
    
    def _update_importance_scores(
        self,
        model_name: str,
        fisher_information: Dict[str, torch.Tensor]
    ) -> None:
        """更新参数重要性分数"""
        for name, fisher in fisher_information.items():
            importance_score = fisher.sum().item()
            
            self.importance_scores[model_name][name] = ParameterImportance(
                name=name,
                fisher_information=fisher,
                importance_score=importance_score,
                frozen=name in self.frozen_parameters[model_name]
            )
    
    def get_important_parameters(
        self,
        model_name: str = "default",
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[ParameterImportance]:
        """
        获取重要参数
        
        Args:
            model_name: 模型名称
            top_k: 返回前k个重要参数
            threshold: 重要性阈值
            
        Returns:
            重要参数列表
        """
        with self._lock:
            if model_name not in self.importance_scores:
                return []
            
            importances = list(self.importance_scores[model_name].values())
            
            # 按重要性排序
            importances.sort(key=lambda x: x.importance_score, reverse=True)
            
            # 应用过滤
            if threshold is not None:
                importances = [imp for imp in importances if imp.importance_score >= threshold]
            
            if top_k is not None:
                importances = importances[:top_k]
            
            return importances
    
    def distill_to_student(
        self,
        teacher: nn.Module,
        student: nn.Module,
        data_loader: DataLoader,
        temperature: float = 4.0,
        alpha: float = 0.5,
        num_epochs: int = 10,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        知识蒸馏
        
        将教师模型的知识迁移到学生模型。
        
        Args:
            teacher: 教师模型
            student: 学生模型
            data_loader: 数据加载器
            temperature: 蒸馏温度
            alpha: 软目标损失权重（0-1之间）
            num_epochs: 训练轮数
            optimizer: 优化器（如果为None，使用Adam）
            device: 计算设备
            
        Returns:
            训练历史
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        teacher = teacher.to(device).eval()
        student = student.to(device).train()
        
        if optimizer is None:
            optimizer = torch.optim.Adam(student.parameters())
        
        history = {
            'total_loss': [],
            'soft_loss': [],
            'hard_loss': []
        }
        
        for epoch in range(num_epochs):
            epoch_total_loss = 0.0
            epoch_soft_loss = 0.0
            epoch_hard_loss = 0.0
            num_batches = 0
            
            for batch in data_loader:
                # 准备输入
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                    labels = batch[1] if len(batch) > 1 else None
                else:
                    inputs = batch
                    labels = None
                
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(device)
                if isinstance(labels, torch.Tensor):
                    labels = labels.to(device)
                
                # 教师模型前向传播
                with torch.no_grad():
                    teacher_outputs = teacher(inputs)
                    if isinstance(teacher_outputs, torch.Tensor):
                        teacher_logits = teacher_outputs
                    else:
                        teacher_logits = teacher_outputs.logits if hasattr(teacher_outputs, 'logits') else teacher_outputs[0]
                
                # 学生模型前向传播
                student_outputs = student(inputs)
                if isinstance(student_outputs, torch.Tensor):
                    student_logits = student_outputs
                else:
                    student_logits = student_outputs.logits if hasattr(student_outputs, 'logits') else student_outputs[0]
                
                # 计算软目标损失（KL散度）
                soft_loss = F.kl_div(
                    F.log_softmax(student_logits / temperature, dim=-1),
                    F.softmax(teacher_logits / temperature, dim=-1),
                    reduction='batchmean'
                ) * (temperature ** 2)
                
                # 计算硬目标损失（交叉熵）
                hard_loss = 0.0
                if labels is not None:
                    hard_loss = F.cross_entropy(student_logits, labels)
                
                # 总损失
                loss = alpha * soft_loss + (1 - alpha) * hard_loss
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 记录损失
                epoch_total_loss += loss.item()
                epoch_soft_loss += soft_loss.item()
                if isinstance(hard_loss, torch.Tensor):
                    epoch_hard_loss += hard_loss.item()
                num_batches += 1
            
            # 记录epoch平均损失
            history['total_loss'].append(epoch_total_loss / num_batches)
            history['soft_loss'].append(epoch_soft_loss / num_batches)
            history['hard_loss'].append(epoch_hard_loss / num_batches)
        
        return history
    
    def get_frozen_parameters(self, model_name: str = "default") -> Set[str]:
        """
        获取冻结的参数名称
        
        Args:
            model_name: 模型名称
            
        Returns:
            冻结的参数名称集合
        """
        with self._lock:
            return self.frozen_parameters[model_name].copy()
    
    def is_parameter_frozen(
        self,
        parameter_name: str,
        model_name: str = "default"
    ) -> bool:
        """
        检查参数是否已冻结
        
        Args:
            parameter_name: 参数名称
            model_name: 模型名称
            
        Returns:
            是否已冻结
        """
        with self._lock:
            return parameter_name in self.frozen_parameters[model_name]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            total_frozen = sum(len(params) for params in self.frozen_parameters.values())
            
            return {
                'num_snapshots': len(self.snapshots),
                'snapshot_count': self._snapshot_count,
                'num_models': len(self.frozen_parameters),
                'total_frozen_parameters': total_frozen,
                'freeze_count': self._freeze_count
            }
    
    def _save_to_disk(self) -> None:
        """保存到磁盘"""
        if not self.storage_path:
            return
        
        try:
            os.makedirs(os.path.dirname(self.storage_path) if os.path.dirname(self.storage_path) else '.', exist_ok=True)
            
            # 准备数据
            data = {
                'snapshots': {},
                'frozen_parameters': {k: list(v) for k, v in self.frozen_parameters.items()},
                'importance_scores': {},
                'statistics': {
                    'snapshot_count': self._snapshot_count,
                    'freeze_count': self._freeze_count
                }
            }
            
            # 序列化快照
            for name, snapshot in self.snapshots.items():
                snapshot_dict = snapshot.to_dict()
                # 保存参数到单独的文件
                param_path = f"{self.storage_path}_params_{name}.pt"
                torch.save(snapshot.parameters, param_path)
                data['snapshots'][name] = snapshot_dict
            
            # 序列化重要性分数
            for model_name, importances in self.importance_scores.items():
                data['importance_scores'][model_name] = {
                    name: imp.to_dict() 
                    for name, imp in importances.items()
                }
            
            # 保存元数据
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            warnings.warn(f"Failed to save cold memory to disk: {e}")
    
    def _load_from_disk(self) -> None:
        """从磁盘加载"""
        if not os.path.exists(self.storage_path):
            return
        
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 加载统计信息
            stats = data.get('statistics', {})
            self._snapshot_count = stats.get('snapshot_count', 0)
            self._freeze_count = stats.get('freeze_count', 0)
            
            # 加载快照
            for name, snapshot_dict in data.get('snapshots', {}).items():
                param_path = f"{self.storage_path}_params_{name}.pt"
                if os.path.exists(param_path):
                    parameters = torch.load(param_path, map_location='cpu')
                    
                    snapshot = ParameterSnapshot(
                        name=snapshot_dict['name'],
                        timestamp=snapshot_dict['timestamp'],
                        parameters=parameters,
                        metadata=snapshot_dict.get('metadata', {}),
                        description=snapshot_dict.get('description', '')
                    )
                    self.snapshots[name] = snapshot
            
            # 加载冻结参数
            self.frozen_parameters = defaultdict(set)
            for model_name, params in data.get('frozen_parameters', {}).items():
                self.frozen_parameters[model_name] = set(params)
            
            # 加载重要性分数（Fisher信息需要重新计算）
            self.importance_scores = defaultdict(dict)
            for model_name, importances in data.get('importance_scores', {}).items():
                for name, imp_dict in importances.items():
                    fisher_data = imp_dict.get('fisher_information', [])
                    fisher_tensor = torch.tensor(fisher_data) if fisher_data else torch.tensor(0.0)
                    
                    self.importance_scores[model_name][name] = ParameterImportance(
                        name=name,
                        fisher_information=fisher_tensor,
                        importance_score=imp_dict.get('importance_score', 0.0),
                        frozen=imp_dict.get('frozen', False)
                    )
            
        except Exception as e:
            warnings.warn(f"Failed to load cold memory from disk: {e}")
    
    def clear(self) -> None:
        """清空所有数据"""
        with self._lock:
            self.snapshots.clear()
            self.frozen_parameters.clear()
            self.importance_scores.clear()
            self._parameter_masks.clear()
            self._snapshot_count = 0
            self._freeze_count = 0


# 便捷函数
def create_cold_memory(storage_path: Optional[str] = None) -> ColdMemory:
    """
    创建参数固化冷记忆的便捷函数
    
    Args:
        storage_path: 存储路径
        
    Returns:
        ColdMemory实例
    """
    return ColdMemory(storage_path=storage_path)
