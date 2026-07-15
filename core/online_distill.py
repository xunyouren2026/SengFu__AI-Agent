"""
在线知识蒸馏模块 (Online Knowledge Distillation)

该模块实现了动态在线知识蒸馏框架,支持教师模型的在线更新、
软目标生成、温度退火、在线硬样本挖掘和多教师融合。

主要特性:
- 教师模型动态更新
- 软目标生成
- 温度退火
- 在线硬样本挖掘
- 多教师融合

作者: AGI Unified Framework Team
版本: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass, field
from collections import deque
import numpy as np
from copy import deepcopy
import math


@dataclass
class OnlineDistillationConfig:
    """在线蒸馏配置类
    
    Attributes:
        temperature: 初始温度
        min_temperature: 最小温度
        max_temperature: 最大温度
        teacher_momentum: 教师模型动量更新系数
        num_teachers: 教师模型数量
        fusion_strategy: 教师融合策略 ('mean', 'weighted', 'adaptive')
        hard_mining_ratio: 硬样本挖掘比例
        hard_mining_threshold: 硬样本阈值
        update_interval: 教师更新间隔(步数)
        warmup_steps: 预热步数
        temperature_schedule: 温度调度策略 ('constant', 'linear', 'cosine', 'exponential')
        consistency_weight: 一致性损失权重
        diversity_weight: 多样性损失权重
        use_ema_teachers: 是否使用EMA教师
    """
    temperature: float = 4.0
    min_temperature: float = 1.0
    max_temperature: float = 10.0
    teacher_momentum: float = 0.999
    num_teachers: int = 1
    fusion_strategy: str = 'adaptive'
    hard_mining_ratio: float = 0.3
    hard_mining_threshold: float = 0.7
    update_interval: int = 100
    warmup_steps: int = 1000
    temperature_schedule: str = 'cosine'
    consistency_weight: float = 0.5
    diversity_weight: float = 0.1
    use_ema_teachers: bool = True


class TemperatureScheduler:
    """温度调度器
    
    实现多种温度退火策略,控制蒸馏软目标的平滑度。
    """
    
    def __init__(
        self,
        initial_temp: float = 4.0,
        min_temp: float = 1.0,
        max_temp: float = 10.0,
        strategy: str = 'cosine',
        total_steps: int = 10000,
        warmup_steps: int = 1000
    ):
        """
        Args:
            initial_temp: 初始温度
            min_temp: 最小温度
            max_temp: 最大温度
            strategy: 调度策略
            total_steps: 总步数
            warmup_steps: 预热步数
        """
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.strategy = strategy
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.current_step = 0
    
    def step(self) -> float:
        """
        更新并返回当前温度
        
        Returns:
            当前温度值
        """
        self.current_step += 1
        
        # 预热阶段
        if self.current_step < self.warmup_steps:
            return self.max_temp
        
        # 计算进度
        progress = min(1.0, (self.current_step - self.warmup_steps) / 
                      (self.total_steps - self.warmup_steps))
        
        if self.strategy == 'constant':
            temp = self.initial_temp
        elif self.strategy == 'linear':
            temp = self.max_temp - (self.max_temp - self.min_temp) * progress
        elif self.strategy == 'cosine':
            temp = self.min_temp + (self.max_temp - self.min_temp) * \
                   (1 + math.cos(math.pi * progress)) / 2
        elif self.strategy == 'exponential':
            temp = self.min_temp + (self.max_temp - self.min_temp) * \
                   math.exp(-5 * progress)
        else:
            temp = self.initial_temp
        
        return max(self.min_temp, min(self.max_temp, temp))
    
    def get_temperature(self) -> float:
        """获取当前温度"""
        return self.step()
    
    def reset(self):
        """重置调度器"""
        self.current_step = 0


class TeacherEnsemble:
    """教师模型集成
    
    管理多个教师模型,支持多种融合策略。
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        fusion_strategy: str = 'adaptive',
        momentum: float = 0.999
    ):
        """
        Args:
            models: 教师模型列表
            fusion_strategy: 融合策略
            momentum: EMA动量
        """
        self.models = models
        self.fusion_strategy = fusion_strategy
        self.momentum = momentum
        self.num_teachers = len(models)
        
        # 教师权重(用于加权融合)
        self.teacher_weights = torch.ones(self.num_teachers) / self.num_teachers
        
        # 教师性能历史
        self.performance_history: List[deque] = [deque(maxlen=50) for _ in range(self.num_teachers)]
        
        # EMA教师模型
        self.ema_models: List[Optional[nn.Module]] = [None] * self.num_teachers
        self._init_ema_models()
    
    def _init_ema_models(self):
        """初始化EMA模型"""
        for i, model in enumerate(self.models):
            self.ema_models[i] = deepcopy(model)
            for param in self.ema_models[i].parameters():
                param.requires_grad = False
    
    def update_teachers(self, student_model: nn.Module):
        """
        使用学生模型更新教师模型(EMA更新)
        
        Args:
            student_model: 学生模型
        """
        for i, ema_model in enumerate(self.ema_models):
            with torch.no_grad():
                for ema_param, student_param in zip(
                    ema_model.parameters(),
                    student_model.parameters()
                ):
                    ema_param.data = (
                        self.momentum * ema_param.data +
                        (1 - self.momentum) * student_param.data
                    )
    
    def get_soft_targets(
        self,
        x: torch.Tensor,
        temperature: float,
        use_ema: bool = True
    ) -> torch.Tensor:
        """
        获取软目标
        
        Args:
            x: 输入数据
            temperature: 温度
            use_ema: 是否使用EMA模型
            
        Returns:
            融合的软目标
        """
        models = self.ema_models if use_ema else self.models
        
        with torch.no_grad():
            logits_list = []
            for model in models:
                logits = model(x)
                logits_list.append(logits)
            
            # 融合策略
            if self.fusion_strategy == 'mean':
                fused_logits = torch.stack(logits_list).mean(dim=0)
            elif self.fusion_strategy == 'weighted':
                weights = self.teacher_weights.to(x.device)
                fused_logits = sum(w * l for w, l in zip(weights, logits_list))
            elif self.fusion_strategy == 'adaptive':
                # 基于不确定性自适应加权
                uncertainties = []
                for logits in logits_list:
                    probs = F.softmax(logits / temperature, dim=1)
                    uncertainty = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                    uncertainties.append(uncertainty.item())
                
                # 不确定性越低,权重越高
                inv_uncertainties = [1.0 / (u + 1e-8) for u in uncertainties]
                total = sum(inv_uncertainties)
                adaptive_weights = [u / total for u in inv_uncertainties]
                
                fused_logits = sum(w * l for w, l in zip(adaptive_weights, logits_list))
            else:
                fused_logits = torch.stack(logits_list).mean(dim=0)
            
            # 应用温度
            soft_targets = F.softmax(fused_logits / temperature, dim=1)
        
        return soft_targets
    
    def update_teacher_weights(self, losses: List[float]):
        """
        更新教师权重
        
        Args:
            losses: 各教师的损失列表
        """
        # 损失越低,权重越高
        inv_losses = [1.0 / (l + 1e-8) for l in losses]
        total = sum(inv_losses)
        self.teacher_weights = torch.tensor([l / total for l in inv_losses])
    
    def report_performance(self, teacher_idx: int, performance: float):
        """
        报告教师性能
        
        Args:
            teacher_idx: 教师索引
            performance: 性能指标
        """
        self.performance_history[teacher_idx].append(performance)


class HardSampleMiner:
    """在线硬样本挖掘器
    
    动态识别和挖掘难以学习的样本。
    """
    
    def __init__(
        self,
        ratio: float = 0.3,
        threshold: float = 0.7,
        history_size: int = 1000
    ):
        """
        Args:
            ratio: 硬样本比例
            threshold: 硬样本阈值
            history_size: 历史大小
        """
        self.ratio = ratio
        self.threshold = threshold
        self.history_size = history_size
        self.sample_history: deque = deque(maxlen=history_size)
        self.loss_history: deque = deque(maxlen=history_size)
    
    def mine_hard_samples(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        挖掘硬样本
        
        Args:
            student_logits: 学生输出
            teacher_logits: 教师输出
            labels: 真实标签,可选
            
        Returns:
            (硬样本索引, 硬样本权重)
        """
        batch_size = student_logits.shape[0]
        num_hard = max(1, int(batch_size * self.ratio))
        
        with torch.no_grad():
            # 计算每个样本的KL散度(蒸馏损失)
            student_probs = F.log_softmax(student_logits, dim=1)
            teacher_probs = F.softmax(teacher_logits, dim=1)
            kl_div = F.kl_div(student_probs, teacher_probs, reduction='none').sum(dim=1)
            
            # 如果有真实标签,也考虑分类损失
            if labels is not None:
                ce_loss = F.cross_entropy(student_logits, labels, reduction='none')
                sample_difficulty = kl_div + ce_loss
            else:
                sample_difficulty = kl_div
            
            # 选择最难的样本
            _, hard_indices = torch.topk(sample_difficulty, num_hard, largest=True)
            
            # 计算权重
            hard_scores = sample_difficulty[hard_indices]
            weights = F.softmax(hard_scores, dim=0)
        
        return hard_indices, weights
    
    def update_history(self, sample_ids: List, losses: List[float]):
        """
        更新样本历史
        
        Args:
            sample_ids: 样本ID列表
            losses: 对应损失列表
        """
        for sid, loss in zip(sample_ids, losses):
            self.sample_history.append(sid)
            self.loss_history.append(loss)
    
    def get_hard_sample_statistics(self) -> Dict[str, float]:
        """
        获取硬样本统计
        
        Returns:
            统计信息字典
        """
        if len(self.loss_history) == 0:
            return {'mean_loss': 0.0, 'std_loss': 0.0}
        
        losses = np.array(self.loss_history)
        return {
            'mean_loss': float(np.mean(losses)),
            'std_loss': float(np.std(losses)),
            'max_loss': float(np.max(losses)),
            'min_loss': float(np.min(losses))
        }


class ConsistencyLoss(nn.Module):
    """一致性损失
    
    鼓励学生在不同扰动下保持预测一致。
    """
    
    def __init__(self, loss_type: str = 'mse'):
        """
        Args:
            loss_type: 损失类型 ('mse', 'kl', 'js')
        """
        super().__init__()
        self.loss_type = loss_type
    
    def forward(
        self,
        logits1: torch.Tensor,
        logits2: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        计算一致性损失
        
        Args:
            logits1: 第一个输出
            logits2: 第二个输出
            temperature: 温度
            
        Returns:
            一致性损失
        """
        if self.loss_type == 'mse':
            probs1 = F.softmax(logits1 / temperature, dim=1)
            probs2 = F.softmax(logits2 / temperature, dim=1)
            loss = F.mse_loss(probs1, probs2)
        elif self.loss_type == 'kl':
            log_probs1 = F.log_softmax(logits1 / temperature, dim=1)
            probs2 = F.softmax(logits2 / temperature, dim=1)
            loss = F.kl_div(log_probs1, probs2, reduction='batchmean')
        elif self.loss_type == 'js':
            # Jensen-Shannon散度
            probs1 = F.softmax(logits1 / temperature, dim=1)
            probs2 = F.softmax(logits2 / temperature, dim=1)
            mean_probs = (probs1 + probs2) / 2
            
            kl1 = F.kl_div(torch.log(mean_probs + 1e-8), probs1, reduction='batchmean')
            kl2 = F.kl_div(torch.log(mean_probs + 1e-8), probs2, reduction='batchmean')
            loss = (kl1 + kl2) / 2
        else:
            loss = F.mse_loss(logits1, logits2)
        
        return loss


class DiversityLoss(nn.Module):
    """多样性损失
    
    鼓励多教师模型之间的多样性。
    """
    
    def __init__(self, loss_type: str = 'cosine'):
        """
        Args:
            loss_type: 损失类型 ('cosine', 'kl')
        """
        super().__init__()
        self.loss_type = loss_type
    
    def forward(self, logits_list: List[torch.Tensor]) -> torch.Tensor:
        """
        计算多样性损失
        
        Args:
            logits_list: 多个教师的输出列表
            
        Returns:
            多样性损失(负值表示鼓励多样性)
        """
        num_teachers = len(logits_list)
        if num_teachers < 2:
            return torch.tensor(0.0, device=logits_list[0].device)
        
        diversity = 0.0
        count = 0
        
        for i in range(num_teachers):
            for j in range(i + 1, num_teachers):
                if self.loss_type == 'cosine':
                    # 余弦相似度(越低越多样)
                    probs_i = F.softmax(logits_list[i], dim=1)
                    probs_j = F.softmax(logits_list[j], dim=1)
                    sim = F.cosine_similarity(probs_i, probs_j, dim=1).mean()
                    diversity += sim
                elif self.loss_type == 'kl':
                    # KL散度(越高越多样)
                    log_probs_i = F.log_softmax(logits_list[i], dim=1)
                    probs_j = F.softmax(logits_list[j], dim=1)
                    kl = F.kl_div(log_probs_i, probs_j, reduction='batchmean')
                    diversity -= kl  # 负KL表示鼓励高散度
                
                count += 1
        
        # 返回负值以鼓励多样性
        return -diversity / max(1, count)


class OnlineDistillation:
    """在线知识蒸馏主类
    
    实现完整的在线蒸馏流程,包括教师更新、
    软目标生成、硬样本挖掘和多教师融合。
    """
    
    def __init__(
        self,
        student: nn.Module,
        teachers: Union[nn.Module, List[nn.Module]],
        config: Optional[OnlineDistillationConfig] = None
    ):
        """
        Args:
            student: 学生模型
            teachers: 教师模型或教师列表
            config: 配置
        """
        self.student = student
        self.config = config or OnlineDistillationConfig()
        
        # 初始化教师集成
        if isinstance(teachers, list):
            teacher_list = teachers
        else:
            teacher_list = [teachers]
        
        self.teacher_ensemble = TeacherEnsemble(
            teacher_list,
            fusion_strategy=self.config.fusion_strategy,
            momentum=self.config.teacher_momentum
        )
        
        # 初始化温度调度器
        self.temp_scheduler = TemperatureScheduler(
            initial_temp=self.config.temperature,
            min_temp=self.config.min_temperature,
            max_temp=self.config.max_temperature,
            strategy=self.config.temperature_schedule,
            warmup_steps=self.config.warmup_steps
        )
        
        # 初始化硬样本挖掘器
        self.hard_miner = HardSampleMiner(
            ratio=self.config.hard_mining_ratio,
            threshold=self.config.hard_mining_threshold
        )
        
        # 初始化损失模块
        self.consistency_loss_fn = ConsistencyLoss()
        self.diversity_loss_fn = DiversityLoss()
        
        # 训练状态
        self.global_step = 0
        self.teacher_update_count = 0
        
        # 统计信息
        self.loss_history: deque = deque(maxlen=100)
        self.distill_loss_history: deque = deque(maxlen=100)
    
    def update_teachers(self, batch: Optional[Dict] = None):
        """
        更新教师模型
        
        Args:
            batch: 数据批次,用于计算更新条件
        """
        self.teacher_update_count += 1
        
        # 按间隔更新
        if self.teacher_update_count % self.config.update_interval == 0:
            if self.config.use_ema_teachers:
                self.teacher_ensemble.update_teachers(self.student)
    
    def compute_soft_targets(
        self,
        x: torch.Tensor,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        计算软目标
        
        Args:
            x: 输入数据
            temperature: 温度,None则使用调度器当前值
            
        Returns:
            软目标概率分布
        """
        if temperature is None:
            temperature = self.temp_scheduler.get_temperature()
        
        return self.teacher_ensemble.get_soft_targets(
            x, temperature, use_ema=self.config.use_ema_teachers
        )
    
    def adaptive_temperature(self, epoch: Optional[int] = None) -> float:
        """
        获取自适应温度
        
        Args:
            epoch: 当前轮数
            
        Returns:
            当前温度
        """
        if epoch is not None:
            # 基于epoch计算
            progress = epoch / 100  # 假设100轮
            return self.temp_scheduler.max_temp - \
                   (self.temp_scheduler.max_temp - self.temp_scheduler.min_temp) * progress
        
        return self.temp_scheduler.get_temperature()
    
    def distill_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        执行蒸馏步骤
        
        Args:
            batch: 数据批次,包含 'input' 和可选的 'label'
            
        Returns:
            包含各种损失的字典
        """
        x = batch['input']
        labels = batch.get('label')
        
        # 获取当前温度
        temperature = self.temp_scheduler.step()
        
        # 学生前向
        student_logits = self.student(x)
        
        # 获取软目标
        soft_targets = self.compute_soft_targets(x, temperature)
        
        # 计算蒸馏损失
        student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
        distill_loss = F.kl_div(
            student_log_probs,
            soft_targets,
            reduction='batchmean'
        ) * (temperature ** 2)
        
        # 硬样本挖掘损失
        hard_loss = torch.tensor(0.0, device=x.device)
        if labels is not None:
            # 获取教师logits用于硬样本挖掘
            with torch.no_grad():
                teacher_logits_list = []
                for model in self.teacher_ensemble.ema_models:
                    teacher_logits_list.append(model(x))
                teacher_logits = torch.stack(teacher_logits_list).mean(dim=0)
            
            hard_indices, hard_weights = self.hard_miner.mine_hard_samples(
                student_logits, teacher_logits, labels
            )
            
            # 硬样本的蒸馏损失
            hard_student_logits = student_logits[hard_indices]
            hard_soft_targets = soft_targets[hard_indices]
            
            hard_student_log_probs = F.log_softmax(hard_student_logits / temperature, dim=1)
            hard_distill_loss = F.kl_div(
                hard_student_log_probs,
                hard_soft_targets,
                reduction='none'
            ).sum(dim=1)
            
            hard_loss = (hard_distill_loss * hard_weights).sum()
        
        # 一致性损失(如果使用数据增强)
        consistency_loss = torch.tensor(0.0, device=x.device)
        if 'input_aug' in batch:
            x_aug = batch['input_aug']
            student_logits_aug = self.student(x_aug)
            consistency_loss = self.consistency_loss_fn(
                student_logits, student_logits_aug, temperature
            )
        
        # 多样性损失(多教师)
        diversity_loss = torch.tensor(0.0, device=x.device)
        if self.teacher_ensemble.num_teachers > 1:
            with torch.no_grad():
                teacher_logits_list = [
                    model(x) for model in self.teacher_ensemble.ema_models
                ]
            diversity_loss = self.diversity_loss_fn(teacher_logits_list)
        
        # 总损失
        total_loss = distill_loss + \
                     0.5 * hard_loss + \
                     self.config.consistency_weight * consistency_loss + \
                     self.config.diversity_weight * diversity_loss
        
        # 如果有标签,添加监督损失
        if labels is not None:
            ce_loss = F.cross_entropy(student_logits, labels)
            total_loss = 0.7 * total_loss + 0.3 * ce_loss
        
        # 更新教师
        self.update_teachers(batch)
        
        self.global_step += 1
        
        # 记录历史
        self.loss_history.append(total_loss.item())
        self.distill_loss_history.append(distill_loss.item())
        
        return {
            'total_loss': total_loss,
            'distill_loss': distill_loss,
            'hard_loss': hard_loss,
            'consistency_loss': consistency_loss,
            'diversity_loss': diversity_loss,
            'temperature': torch.tensor(temperature),
            'student_logits': student_logits
        }
    
    def get_student(self) -> nn.Module:
        """获取学生模型"""
        return self.student
    
    def get_teachers(self) -> List[nn.Module]:
        """获取教师模型列表"""
        return self.teacher_ensemble.ema_models if self.config.use_ema_teachers \
               else self.teacher_ensemble.models
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取训练统计
        
        Returns:
            统计信息字典
        """
        return {
            'global_step': self.global_step,
            'teacher_update_count': self.teacher_update_count,
            'current_temperature': self.temp_scheduler.get_temperature(),
            'mean_loss': np.mean(self.loss_history) if self.loss_history else 0.0,
            'mean_distill_loss': np.mean(self.distill_loss_history) if self.distill_loss_history else 0.0,
            'hard_sample_stats': self.hard_miner.get_hard_sample_statistics(),
            'teacher_weights': self.teacher_ensemble.teacher_weights.tolist()
        }
    
    def save_checkpoint(self, path: str):
        """
        保存检查点
        
        Args:
            path: 保存路径
        """
        checkpoint = {
            'student_state': self.student.state_dict(),
            'teacher_states': [
                model.state_dict() for model in self.teacher_ensemble.ema_models
            ],
            'temp_scheduler_state': {
                'current_step': self.temp_scheduler.current_step
            },
            'global_step': self.global_step,
            'config': self.config
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """
        加载检查点
        
        Args:
            path: 加载路径
        """
        checkpoint = torch.load(path)
        
        self.student.load_state_dict(checkpoint['student_state'])
        
        for model, state in zip(
            self.teacher_ensemble.ema_models,
            checkpoint['teacher_states']
        ):
            model.load_state_dict(state)
        
        self.temp_scheduler.current_step = checkpoint['temp_scheduler_state']['current_step']
        self.global_step = checkpoint['global_step']


# 辅助函数
def create_online_distillation(
    student: nn.Module,
    teachers: Union[nn.Module, List[nn.Module]],
    config_dict: Optional[Dict] = None
) -> OnlineDistillation:
    """
    从配置创建在线蒸馏器
    
    Args:
        student: 学生模型
        teachers: 教师模型
        config_dict: 配置字典
        
    Returns:
        OnlineDistillation实例
    """
    if config_dict:
        config = OnlineDistillationConfig(**config_dict)
    else:
        config = None
    
    return OnlineDistillation(student, teachers, config)


def mutual_learning_distillation(
    models: List[nn.Module],
    batch: Dict[str, torch.Tensor],
    temperature: float = 4.0
) -> Dict[str, torch.Tensor]:
    """
    互学习蒸馏
    
    多个模型相互学习,没有固定的教师-学生区分。
    
    Args:
        models: 模型列表
        batch: 数据批次
        temperature: 温度
        
    Returns:
        各模型的损失字典
    """
    x = batch['input']
    labels = batch.get('label')
    
    # 所有模型前向
    logits_list = [model(x) for model in models]
    
    losses = {}
    for i, (model, logits) in enumerate(zip(models, logits_list)):
        # 其他模型作为教师
        other_logits = [l for j, l in enumerate(logits_list) if j != i]
        
        # 平均教师预测
        teacher_logits = torch.stack(other_logits).mean(dim=0)
        
        # 蒸馏损失
        student_probs = F.log_softmax(logits / temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
        distill_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
        
        # 监督损失
        if labels is not None:
            ce_loss = F.cross_entropy(logits, labels)
            total_loss = 0.5 * distill_loss + 0.5 * ce_loss
        else:
            total_loss = distill_loss
        
        losses[f'model_{i}'] = total_loss
    
    return losses
