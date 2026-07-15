"""
AGI统一框架 - 自动微分与优化器
实现高级优化算法、学习率调度、梯度处理等
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
import math
from typing import Optional, Tuple, List, Dict, Any, Callable
import numpy as np


# ==================== 高级优化器 ====================

class AdamW(Optimizer):
    """带权重衰减解耦的Adam"""
    
    def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01, amsgrad: bool = False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                    
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                        
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                step = state['step']
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                # AdamW: weight decay is applied after the gradient update
                p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
        return loss


class LAMB(Optimizer):
    """LAMB: Large Batch Optimization for Deep Learning"""
    
    def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-6, weight_decay: float = 0.01, bias_correction: bool = True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, 
                       bias_correction=bias_correction)
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                step = state['step']
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                if group['bias_correction']:
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                    exp_avg = exp_avg / bias_correction1
                    exp_avg_sq = exp_avg_sq / bias_correction2
                    
                # Adam update
                adam_update = exp_avg / (exp_avg_sq.sqrt() + group['eps'])
                
                # Add weight decay
                if group['weight_decay'] > 0:
                    adam_update.add_(p.data, alpha=group['weight_decay'])
                    
                # LAMB trust ratio
                weight_norm = p.data.norm(p=2)
                update_norm = adam_update.norm(p=2)
                
                if weight_norm > 0 and update_norm > 0:
                    trust_ratio = weight_norm / update_norm
                else:
                    trust_ratio = 1.0
                    
                # Apply update
                p.data.add_(adam_update, alpha=-group['lr'] * trust_ratio)
                
        return loss


class Lookahead(Optimizer):
    """Lookahead优化器包装器"""
    
    def __init__(self, optimizer: Optimizer, k: int = 5, alpha: float = 0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state
        
        # 保存慢权重
        self.slow_weights = [[p.clone().detach() for p in group['params']] 
                            for group in self.param_groups]
        self.step_counter = 0
        
    def step(self, closure: Optional[Callable] = None):
        loss = self.optimizer.step(closure)
        self.step_counter += 1
        
        if self.step_counter % self.k == 0:
            # 更新慢权重
            for group, slow_group in zip(self.param_groups, self.slow_weights):
                for p, slow_p in zip(group['params'], slow_group):
                    if p.grad is None:
                        continue
                    slow_p.add_(p.data - slow_p, alpha=self.alpha)
                    p.data.copy_(slow_p)
                    
        return loss
    
    def zero_grad(self):
        self.optimizer.zero_grad()


class RAdam(Optimizer):
    """Rectified Adam"""
    
    def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                step = state['step']
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Rectification
                beta2_t = beta2 ** step
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * step * beta2_t / (1 - beta2_t)
                
                if N_sma >= 5:
                    r = math.sqrt(
                        (N_sma - 4) * (N_sma - 2) * N_sma_max /
                        ((N_sma_max - 4) * (N_sma_max - 2) * N_sma)
                    )
                    bias_correction1 = 1 - beta1 ** step
                    step_size = r * group['lr'] / bias_correction1
                    
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    step_size = group['lr']
                    p.data.add_(exp_avg, alpha=-step_size)
                    
                if group['weight_decay'] > 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
                    
        return loss


class AdaBelief(Optimizer):
    """AdaBelief优化器"""
    
    def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-16, weight_decay: float = 0.0, amsgrad: bool = False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                        
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                step = state['step']
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # AdaBelief: use (grad - exp_avg)^2 instead of grad^2
                grad_residual = grad - exp_avg
                exp_avg_sq.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)
                
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                if group['weight_decay'] > 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
                    
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
        return loss


class Shampoo(Optimizer):
    """Shampoo优化器"""
    
    def __init__(self, params, lr: float = 1e-3, momentum: float = 0.0,
                 weight_decay: float = 0.0, exponent: float = 0.25):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, exponent=exponent)
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                    # Preconditioner matrices
                    state['L'] = torch.zeros_like(p.data)
                    state['R'] = torch.zeros_like(p.data)
                    
                state['step'] += 1
                
                if group['weight_decay'] > 0:
                    grad.add_(p.data, alpha=group['weight_decay'])
                    
                if group['momentum'] > 0:
                    momentum_buffer = state['momentum_buffer']
                    momentum_buffer.mul_(group['momentum']).add_(grad)
                    grad = momentum_buffer
                    
                # Simplified Shampoo update
                # Full implementation requires matrix power operations
                p.data.add_(grad, alpha=-group['lr'])
                
        return loss


# ==================== 学习率调度器 ====================

class CosineAnnealingWarmRestarts:
    """带热重启的余弦退火"""
    
    def __init__(self, optimizer: Optimizer, T_0: int, T_mult: int = 1,
                 eta_min: float = 0, last_epoch: int = -1):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.T_cur = T_0
        self.T_i = T_0
        
        if last_epoch == -1:
            self.step()
            
    def step(self, epoch: Optional[int] = None):
        if epoch is None:
            epoch = self.last_epoch + 1
            
        self.last_epoch = epoch
        
        if self.last_epoch >= self.T_cur:
            self.T_cur = self.T_cur * self.T_mult
            self.T_i = self.T_cur
            
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.eta_min + (self.base_lrs[i] - self.eta_min) * \
                         (1 + math.cos(math.pi * self.last_epoch / self.T_i)) / 2
                         
    def get_lr(self) -> List[float]:
        return [group['lr'] for group in self.optimizer.param_groups]


class OneCycleLR:
    """1cycle学习率策略"""
    
    def __init__(self, optimizer: Optimizer, max_lr: float, total_steps: int,
                 pct_start: float = 0.3, anneal_strategy: str = 'cos',
                 div_factor: float = 25.0, final_div_factor: float = 1e4):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor
        self.step_up = int(total_steps * pct_start)
        self.step_down = total_steps - self.step_up
        
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        
        if self.current_step <= self.step_up:
            # 上升阶段
            pct = self.current_step / self.step_up
            if self.anneal_strategy == 'cos':
                lr = self.initial_lr + (self.max_lr - self.initial_lr) * \
                     (1 - math.cos(math.pi * pct)) / 2
            else:
                lr = self.initial_lr + (self.max_lr - self.initial_lr) * pct
        else:
            # 下降阶段
            pct = (self.current_step - self.step_up) / self.step_down
            if self.anneal_strategy == 'cos':
                lr = self.max_lr - (self.max_lr - self.final_lr) * \
                     (1 - math.cos(math.pi * pct)) / 2
            else:
                lr = self.max_lr - (self.max_lr - self.final_lr) * pct
                
        for group in self.optimizer.param_groups:
            group['lr'] = lr


class LinearWarmup:
    """线性预热"""
    
    def __init__(self, optimizer: Optimizer, warmup_steps: int, 
                 target_lr: float, start_lr: float = 0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
        self.start_lr = start_lr
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            lr = self.start_lr + (self.target_lr - self.start_lr) * \
                 self.current_step / self.warmup_steps
        else:
            lr = self.target_lr
            
        for group in self.optimizer.param_groups:
            group['lr'] = lr


# ==================== 梯度处理 ====================

class GradientClipping:
    """梯度裁剪"""
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
        
    def __call__(self, model: nn.Module):
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), self.max_norm, self.norm_type
        )


class GradientAccumulator:
    """梯度累积"""
    
    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        
    def should_step(self) -> bool:
        self.current_step += 1
        return self.current_step % self.accumulation_steps == 0
    
    def reset(self):
        self.current_step = 0


class GradientCentralization:
    """梯度中心化"""
    
    def __call__(self, model: nn.Module):
        for param in model.parameters():
            if param.grad is not None and param.dim() > 1:
                # 对梯度进行中心化
                grad = param.grad.data
                if grad.dim() == 2:
                    grad = grad - grad.mean(dim=1, keepdim=True)
                elif grad.dim() == 4:
                    grad = grad - grad.mean(dim=(1, 2, 3), keepdim=True)
                param.grad.data = grad


# ==================== 混合精度训练 ====================

class MixedPrecisionTrainer:
    """混合精度训练器"""
    
    def __init__(self, model: nn.Module, optimizer: Optimizer,
                 loss_scale: float = 2.0 ** 15, min_scale: float = 1.0,
                 growth_factor: float = 2.0, backoff_factor: float = 0.5,
                 growth_interval: int = 2000):
        self.model = model
        self.optimizer = optimizer
        
        # 检查是否支持FP16
        self.device = next(model.parameters()).device
        self.scaler = torch.cuda.amp.GradScaler(
            init_scale=loss_scale,
            min_scale=min_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval
        ) if self.device.type == 'cuda' else None
        
    def step(self, loss: torch.Tensor, closure: Optional[Callable] = None):
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
            
    def zero_grad(self):
        self.optimizer.zero_grad()
