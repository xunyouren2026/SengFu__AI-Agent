"""
学习率调度器模块 - 包含各种学习率调度策略的真实实现
包括: Step LR, Multi-Step LR, Exponential LR, Cosine Annealing, 
      Linear Warmup, OneCycle, Cyclic LR, Polynomial Decay, 
      Warmup + Cosine, Layer-wise LR Decay
"""

import math
from typing import Optional, Tuple, List, Union, Callable, Dict, Any
from abc import ABC, abstractmethod


class LRScheduler(ABC):
    """学习率调度器基类"""
    
    def __init__(self, optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.base_lrs = [group.get('lr', 0.001) for group in optimizer.param_groups]
        self.last_epoch = last_epoch
        self.step()
    
    def step(self, metrics: Optional[float] = None):
        """更新学习率"""
        self.last_epoch += 1
        values = self.get_lr()
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = values[i]
    
    @abstractmethod
    def get_lr(self) -> List[float]:
        """计算当前学习率"""
        pass
    
    def get_last_lr(self) -> List[float]:
        """获取最后的学习率"""
        return [group['lr'] for group in self.optimizer.param_groups]


class StepLR(LRScheduler):
    """
    阶梯式学习率衰减
    
    每隔step_size个epoch，学习率乘以gamma
    lr = base_lr * gamma^(epoch // step_size)
    """
    
    def __init__(
        self,
        optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1
    ):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        factor = self.gamma ** (self.last_epoch // self.step_size)
        return [base_lr * factor for base_lr in self.base_lrs]


class MultiStepLR(LRScheduler):
    """
    多阶梯学习率衰减
    
    在指定的milestones处，学习率乘以gamma
    """
    
    def __init__(
        self,
        optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        last_epoch: int = -1
    ):
        self.milestones = sorted(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        factor = self.gamma ** sum(1 for m in self.milestones if self.last_epoch >= m)
        return [base_lr * factor for base_lr in self.base_lrs]


class ExponentialLR(LRScheduler):
    """
    指数学习率衰减
    
    lr = base_lr * gamma^epoch
    """
    
    def __init__(
        self,
        optimizer,
        gamma: float,
        last_epoch: int = -1
    ):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        factor = self.gamma ** self.last_epoch
        return [base_lr * factor for base_lr in self.base_lrs]


class CosineAnnealingLR(LRScheduler):
    """
    余弦退火学习率
    
    lr = eta_min + (base_lr - eta_min) * (1 + cos(pi * epoch / T_max)) / 2
    """
    
    def __init__(
        self,
        optimizer,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1
    ):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch == 0:
            return self.base_lrs
        
        factor = (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
        return [self.eta_min + (base_lr - self.eta_min) * factor
                for base_lr in self.base_lrs]


class CosineAnnealingWarmRestarts(LRScheduler):
    """
    带热重启的余弦退火
    
    支持多次重启，每次重启周期可以乘以T_mult
    """
    
    def __init__(
        self,
        optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        last_epoch: int = -1
    ):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = T_0
        self.cycle = 0
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        # 计算当前周期内的位置
        T_cur = self.T_0
        epoch = self.last_epoch
        
        while epoch >= T_cur:
            epoch -= T_cur
            T_cur *= self.T_mult
        
        self.T_cur = T_cur
        
        factor = (1 + math.cos(math.pi * epoch / T_cur)) / 2
        return [self.eta_min + (base_lr - self.eta_min) * factor
                for base_lr in self.base_lrs]


class LinearLR(LRScheduler):
    """
    线性学习率衰减
    
    lr = base_lr * (1 - epoch / total_iters) + end_lr * (epoch / total_iters)
    """
    
    def __init__(
        self,
        optimizer,
        total_iters: int,
        end_factor: float = 0.0,
        last_epoch: int = -1
    ):
        self.total_iters = total_iters
        self.end_factor = end_factor
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        factor = 1 - self.last_epoch / self.total_iters
        factor = max(factor, self.end_factor)
        return [base_lr * factor for base_lr in self.base_lrs]


class PolynomialLR(LRScheduler):
    """
    多项式学习率衰减
    
    lr = base_lr * (1 - epoch / total_iters)^power
    """
    
    def __init__(
        self,
        optimizer,
        total_iters: int,
        power: float = 1.0,
        last_epoch: int = -1
    ):
        self.total_iters = total_iters
        self.power = power
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        factor = (1 - self.last_epoch / self.total_iters) ** self.power
        factor = max(factor, 0.0)
        return [base_lr * factor for base_lr in self.base_lrs]


class OneCycleLR(LRScheduler):
    """
    OneCycle学习率策略
    
    先从initial_lr增加到max_lr，然后减少到final_lr
    同时支持动量调整
    """
    
    def __init__(
        self,
        optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: str = 'cos',
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        three_phase: bool = False,
        last_epoch: int = -1
    ):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.three_phase = three_phase
        
        # 计算各阶段边界
        if three_phase:
            self.step_up = int(total_steps * pct_start)
            self.step_down = int(total_steps * pct_start)
            self.step_end = total_steps - self.step_up - self.step_down
        else:
            self.step_up = int(total_steps * pct_start)
            self.step_down = total_steps - self.step_up
            self.step_end = 0
        
        # 初始和最终学习率
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        step = self.last_epoch
        
        if step <= self.step_up:
            # 上升阶段
            pct = step / self.step_up
            if self.anneal_strategy == 'cos':
                factor = 1 + (self.div_factor - 1) * (1 - math.cos(math.pi * pct)) / 2
            else:
                factor = 1 + (self.div_factor - 1) * pct
            lr = self.max_lr / factor
        
        elif step <= self.step_up + self.step_down:
            # 下降阶段
            pct = (step - self.step_up) / self.step_down
            if self.anneal_strategy == 'cos':
                factor = 1 + (self.div_factor - 1) * (1 + math.cos(math.pi * pct)) / 2
            else:
                factor = 1 + (self.div_factor - 1) * (1 - pct)
            lr = self.max_lr / factor
        
        else:
            # 最终阶段
            pct = (step - self.step_up - self.step_down) / self.step_end if self.step_end > 0 else 0
            if self.anneal_strategy == 'cos':
                factor = self.div_factor + (self.final_div_factor - self.div_factor) * (1 - math.cos(math.pi * pct)) / 2
            else:
                factor = self.div_factor + (self.final_div_factor - self.div_factor) * pct
            lr = self.max_lr / factor
        
        return [lr for _ in self.base_lrs]


class CyclicLR(LRScheduler):
    """
    循环学习率
    
    在base_lr和max_lr之间循环变化
    """
    
    def __init__(
        self,
        optimizer,
        base_lr: float,
        max_lr: float,
        step_size_up: int = 2000,
        step_size_down: Optional[int] = None,
        mode: str = 'triangular',
        gamma: float = 1.0,
        scale_fn: Optional[Callable] = None,
        scale_mode: str = 'cycle',
        last_epoch: int = -1
    ):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down or step_size_up
        self.mode = mode
        self.gamma = gamma
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode
        
        self.total_size = step_size_up + self.step_size_down
        self.cycle = 0
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        cycle = math.floor(1 + self.last_epoch / self.total_size)
        x = 1 + self.last_epoch / self.total_size - cycle
        
        if self.scale_fn is not None:
            scale = self.scale_fn(x)
        elif self.mode == 'triangular':
            scale = 1.0
        elif self.mode == 'triangular2':
            scale = 1.0 / (2 ** (cycle - 1))
        elif self.mode == 'exp_range':
            scale = self.gamma ** cycle
        else:
            scale = 1.0
        
        if x <= self.step_size_up / self.total_size:
            # 上升阶段
            scale *= x * self.total_size / self.step_size_up
        else:
            # 下降阶段
            scale *= (self.total_size - x * self.total_size) / self.step_size_down
        
        lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, scale)
        
        return [lr for _ in self.base_lrs]


class WarmupLR(LRScheduler):
    """
    线性预热学习率
    
    前warmup_iters步线性增加，之后保持不变
    """
    
    def __init__(
        self,
        optimizer,
        warmup_iters: int,
        last_epoch: int = -1
    ):
        self.warmup_iters = warmup_iters
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_iters:
            factor = (self.last_epoch + 1) / self.warmup_iters
        else:
            factor = 1.0
        return [base_lr * factor for base_lr in self.base_lrs]


class WarmupCosineLR(LRScheduler):
    """
    预热 + 余弦退火
    
    前warmup_iters步线性预热，之后余弦退火
    """
    
    def __init__(
        self,
        optimizer,
        warmup_iters: int,
        total_iters: int,
        eta_min: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_iters:
            # 预热阶段
            factor = (self.last_epoch + 1) / self.warmup_iters
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段
            progress = (self.last_epoch - self.warmup_iters) / (self.total_iters - self.warmup_iters)
            factor = (1 + math.cos(math.pi * progress)) / 2
            return [self.eta_min + (base_lr - self.eta_min) * factor
                    for base_lr in self.base_lrs]


class WarmupMultiStepLR(LRScheduler):
    """
    预热 + 多阶梯衰减
    """
    
    def __init__(
        self,
        optimizer,
        warmup_iters: int,
        milestones: List[int],
        gamma: float = 0.1,
        last_epoch: int = -1
    ):
        self.warmup_iters = warmup_iters
        self.milestones = sorted(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_iters:
            factor = (self.last_epoch + 1) / self.warmup_iters
        else:
            factor = self.gamma ** sum(1 for m in self.milestones if self.last_epoch >= m)
        
        return [base_lr * factor for base_lr in self.base_lrs]


class NoamLR(LRScheduler):
    """
    Noam学习率 (Transformer使用)
    
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """
    
    def __init__(
        self,
        optimizer,
        d_model: int,
        warmup_steps: int,
        factor: float = 1.0,
        last_epoch: int = -1
    ):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        step = self.last_epoch + 1
        
        lr = self.factor * (self.d_model ** (-0.5) * 
            min(step ** (-0.5), step * self.warmup_steps ** (-1.5)))
        
        return [lr for _ in self.base_lrs]


class LayerWiseLRDecay(LRScheduler):
    """
    层级学习率衰减
    
    不同层使用不同的学习率，深层使用更小的学习率
    """
    
    def __init__(
        self,
        optimizer,
        num_layers: int,
        decay_rate: float = 0.75,
        last_epoch: int = -1
    ):
        self.num_layers = num_layers
        self.decay_rate = decay_rate
        
        # 计算每层的学习率缩放因子
        self.layer_scales = [decay_rate ** (num_layers - i - 1) for i in range(num_layers)]
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        # 假设param_groups按层排列
        lrs = []
        for i, base_lr in enumerate(self.base_lrs):
            layer_idx = min(i, self.num_layers - 1)
            lrs.append(base_lr * self.layer_scales[layer_idx])
        return lrs


class ReduceLROnPlateau:
    """
    当指标停止改善时降低学习率
    
    这是基于指标的调度器，不是基于epoch的
    """
    
    def __init__(
        self,
        optimizer,
        mode: str = 'min',
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = 'rel',
        cooldown: int = 0,
        min_lr: float = 0.0,
        eps: float = 1e-8
    ):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
        
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.last_epoch = 0
        
        self.base_lrs = [group.get('lr', 0.001) for group in optimizer.param_groups]
    
    def step(self, metrics: float):
        """根据指标更新学习率"""
        self.last_epoch += 1
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
            return
        
        current = metrics
        
        if self._is_improvement(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
    
    def _is_improvement(self, current: float, best: float) -> bool:
        if self.mode == 'min':
            if self.threshold_mode == 'rel':
                return current < best - best * self.threshold
            else:
                return current < best - self.threshold
        else:
            if self.threshold_mode == 'rel':
                return current > best + best * self.threshold
            else:
                return current > best + self.threshold
    
    def _reduce_lr(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr


class ChainedScheduler:
    """
    链式调度器
    
    顺序应用多个调度器
    """
    
    def __init__(self, schedulers: List[LRScheduler]):
        self.schedulers = schedulers
        self.optimizer = schedulers[0].optimizer
        self.last_epoch = -1
    
    def step(self, metrics: Optional[float] = None):
        self.last_epoch += 1
        for scheduler in self.schedulers:
            scheduler.step(metrics)
    
    def get_last_lr(self) -> List[float]:
        return self.schedulers[0].get_last_lr()


class SequentialLR:
    """
    顺序调度器
    
    在不同阶段使用不同的调度器
    """
    
    def __init__(
        self,
        optimizer,
        schedulers: List[LRScheduler],
        milestones: List[int],
        last_epoch: int = -1
    ):
        self.optimizer = optimizer
        self.schedulers = schedulers
        self.milestones = milestones
        self.last_epoch = last_epoch
        self.current_scheduler_idx = 0
    
    def step(self, metrics: Optional[float] = None):
        self.last_epoch += 1
        
        # 检查是否需要切换调度器
        while (self.current_scheduler_idx < len(self.milestones) and
               self.last_epoch >= self.milestones[self.current_scheduler_idx]):
            self.current_scheduler_idx += 1
        
        # 使用当前调度器
        idx = min(self.current_scheduler_idx, len(self.schedulers) - 1)
        self.schedulers[idx].step(metrics)
    
    def get_last_lr(self) -> List[float]:
        idx = min(self.current_scheduler_idx, len(self.schedulers) - 1)
        return self.schedulers[idx].get_last_lr()


# 工厂函数
def get_scheduler(name: str, optimizer, **kwargs) -> Union[LRScheduler, ReduceLROnPlateau]:
    """根据名称获取学习率调度器"""
    schedulers = {
        'step': StepLR,
        'multistep': MultiStepLR,
        'exponential': ExponentialLR,
        'cosine': CosineAnnealingLR,
        'cosine_warmup': CosineAnnealingWarmRestarts,
        'linear': LinearLR,
        'polynomial': PolynomialLR,
        'onecycle': OneCycleLR,
        'cyclic': CyclicLR,
        'warmup': WarmupLR,
        'warmup_cosine': WarmupCosineLR,
        'warmup_multistep': WarmupMultiStepLR,
        'noam': NoamLR,
        'layerwise_decay': LayerWiseLRDecay,
        'reduce_on_plateau': ReduceLROnPlateau
    }
    
    name_lower = name.lower()
    if name_lower not in schedulers:
        raise ValueError(f"Unknown scheduler: {name}. Available: {list(schedulers.keys())}")
    
    return schedulers[name_lower](optimizer, **kwargs)


def create_scheduler_with_warmup(
    optimizer,
    scheduler_type: str = 'cosine',
    warmup_iters: int = 1000,
    total_iters: int = 100000,
    **kwargs
) -> LRScheduler:
    """创建带预热的调度器"""
    if scheduler_type == 'cosine':
        return WarmupCosineLR(optimizer, warmup_iters, total_iters, **kwargs)
    elif scheduler_type == 'multistep':
        milestones = kwargs.pop('milestones', [50000, 80000])
        return WarmupMultiStepLR(optimizer, warmup_iters, milestones, **kwargs)
    else:
        return WarmupLR(optimizer, warmup_iters)
