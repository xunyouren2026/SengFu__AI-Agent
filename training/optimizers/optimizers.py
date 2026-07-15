"""
优化器模块 - 包含各种优化算法的真实实现
包括: SGD, Adam, AdamW, LAMB, LARS, RMSprop, Adagrad, Adadelta, Nadam, RAdam, Lookahead
"""

import math
from typing import Dict, List, Optional, Tuple, Union, Callable
from abc import ABC, abstractmethod


class Tensor:
    """简化的张量类用于演示"""
    def __init__(self, data, requires_grad=False):
        self.data = data if isinstance(data, list) else data
        self.requires_grad = requires_grad
        self.grad = None
        self.shape = self._compute_shape() if isinstance(data, list) else None
    
    def _compute_shape(self):
        shape = []
        d = self.data
        while isinstance(d, list):
            shape.append(len(d))
            d = d[0] if d else None
        return tuple(shape)
    
    def zero_grad(self):
        self.grad = None
    
    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"


class Optimizer(ABC):
    """优化器基类"""
    
    def __init__(self, params: List[Tensor], defaults: Dict):
        self.params = list(params)
        self.defaults = defaults
        self.state: Dict[int, Dict] = {}
        self.param_groups: List[Dict] = []
        
        param_group = {'params': self.params}
        for name, value in defaults.items():
            param_group[name] = value
        self.param_groups.append(param_group)
    
    def zero_grad(self, set_to_none: bool = True):
        """清零所有参数的梯度"""
        for param in self.params:
            if param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    param.grad = 0.0
    
    @abstractmethod
    def step(self, closure: Optional[Callable] = None):
        """执行一步优化"""
        pass
    
    def _get_state(self, param: Tensor) -> Dict:
        """获取参数状态"""
        param_id = id(param)
        if param_id not in self.state:
            self.state[param_id] = {}
        return self.state[param_id]
    
    def add_param_group(self, param_group: Dict):
        """添加参数组"""
        params = param_group['params']
        for name, value in self.defaults.items():
            param_group.setdefault(name, value)
        self.params.extend(params)
        self.param_groups.append(param_group)


class SGD(Optimizer):
    """
    随机梯度下降优化器
    支持动量、Nesterov动量、权重衰减
    
    v_t = mu * v_{t-1} + g_t
    theta_t = theta_{t-1} - lr * v_t
    
    Nesterov:
    theta_t = theta_{t-1} - lr * (mu * v_t + g_t)
    """
    
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = {
            'lr': lr,
            'momentum': momentum,
            'dampening': dampening,
            'weight_decay': weight_decay,
            'nesterov': nesterov
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        """执行SGD更新"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            dampening = group['dampening']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                
                grad = param.grad
                state = self._get_state(param)
                
                # 权重衰减 (L2正则化)
                if weight_decay != 0:
                    grad = grad + weight_decay * param.data
                
                # 动量更新
                if momentum != 0:
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = 0.0
                    
                    buf = state['momentum_buffer']
                    buf = momentum * buf + (1 - dampening) * grad
                    state['momentum_buffer'] = buf
                    
                    if nesterov:
                        grad = grad + momentum * buf
                    else:
                        grad = buf
                
                # 参数更新
                param.data = param.data - lr * grad
        
        return loss


class Adam(Optimizer):
    """
    Adam优化器
    
    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)
    theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
    """
    
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'amsgrad': amsgrad
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        """执行Adam更新"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                
                grad = param.grad
                state = self._get_state(param)
                
                # 初始化状态
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = 0.0  # m_t
                    state['exp_avg_sq'] = 0.0  # v_t
                    if amsgrad:
                        state['max_exp_avg_sq'] = 0.0
                
                state['step'] += 1
                step = state['step']
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                
                # 权重衰减
                if weight_decay != 0:
                    grad = grad + weight_decay * param.data
                
                # 更新一阶矩估计
                exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                state['exp_avg'] = exp_avg
                
                # 更新二阶矩估计
                exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grad ** 2)
                state['exp_avg_sq'] = exp_avg_sq
                
                # AMSGrad变体
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    max_exp_avg_sq = max(max_exp_avg_sq, exp_avg_sq)
                    state['max_exp_avg_sq'] = max_exp_avg_sq
                    denom = math.sqrt(max_exp_avg_sq) + eps
                else:
                    denom = math.sqrt(exp_avg_sq) + eps
                
                # 偏差修正
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                
                # 参数更新
                param.data = param.data - step_size * exp_avg / denom
        
        return loss


class AdamW(Optimizer):
    """
    AdamW优化器 - 解耦权重衰减的Adam
    
    与Adam的区别: 权重衰减直接应用于参数，而不是梯度
    
    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    theta_t = theta_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * theta_{t-1})
    """
    
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False
    ):
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'amsgrad': amsgrad
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        """执行AdamW更新"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                
                grad = param.grad
                state = self._get_state(param)
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = 0.0
                    state['exp_avg_sq'] = 0.0
                    if amsgrad:
                        state['max_exp_avg_sq'] = 0.0
                
                state['step'] += 1
                step = state['step']
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                
                # 更新一阶矩
                exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                state['exp_avg'] = exp_avg
                
                # 更新二阶矩
                exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grad ** 2)
                state['exp_avg_sq'] = exp_avg_sq
                
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    max_exp_avg_sq = max(max_exp_avg_sq, exp_avg_sq)
                    state['max_exp_avg_sq'] = max_exp_avg_sq
                    denom = math.sqrt(max_exp_avg_sq) + eps
                else:
                    denom = math.sqrt(exp_avg_sq) + eps
                
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                
                # AdamW: 解耦权重衰减
                if weight_decay != 0:
                    param.data = param.data * (1 - lr * weight_decay)
                
                param.data = param.data - step_size * exp_avg / denom
        
        return loss


class LAMB(Optimizer):
    """
    LAMB优化器 (Layer-wise Adaptive Moments optimizer for Batch training)
    
    专为大批量训练设计，结合了Adam的适应性和层级学习率缩放
    
    r_t = m_hat / (sqrt(v_hat) + eps)
    phi(L) = min(max(||w||, lower_bound), upper_bound)
    trust_ratio = phi(L) / ||r_t + lambda * w||
    w_t = w_{t-1} - lr * trust_ratio * (r_t + lambda * w_{t-1})
    """
    
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.01,
        bias_correction: bool = True,
        max_grad_norm: Optional[float] = None,
        trust_coef: float = 0.001,
        lower_bound: float = 0.0,
        upper_bound: float = 10.0
    ):
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'bias_correction': bias_correction,
            'max_grad_norm': max_grad_norm,
            'trust_coef': trust_coef,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        super().__init__(params, defaults)
    
    @staticmethod
    def _norm(tensor):
        """计算张量的L2范数"""
        if isinstance(tensor, (int, float)):
            return abs(tensor)
        return math.sqrt(sum(x ** 2 for x in tensor)) if isinstance(tensor, list) else abs(tensor)
    
    def step(self, closure: Optional[Callable] = None):
        """执行LAMB更新"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            bias_correction = group['bias_correction']
            max_grad_norm = group['max_grad_norm']
            trust_coef = group['trust_coef']
            lower_bound = group['lower_bound']
            upper_bound = group['upper_bound']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                
                grad = param.grad
                state = self._get_state(param)
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = 0.0
                    state['exp_avg_sq'] = 0.0
                
                state['step'] += 1
                step = state['step']
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                
                # 梯度裁剪
                if max_grad_norm is not None:
                    grad_norm = self._norm(grad)
                    if grad_norm > max_grad_norm:
                        grad = grad * max_grad_norm / grad_norm
                
                # 更新矩估计
                exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grad ** 2)
                state['exp_avg'] = exp_avg
                state['exp_avg_sq'] = exp_avg_sq
                
                # 偏差修正
                if bias_correction:
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                    exp_avg_hat = exp_avg / bias_correction1
                    exp_avg_sq_hat = exp_avg_sq / bias_correction2
                else:
                    exp_avg_hat = exp_avg
                    exp_avg_sq_hat = exp_avg_sq
                
                # 计算更新方向
                update = exp_avg_hat / (math.sqrt(exp_avg_sq_hat) + eps)
                
                # 添加权重衰减
                if weight_decay != 0:
                    update = update + weight_decay * param.data
                
                # 计算信任比率
                w_norm = self._norm(param.data)
                g_norm = self._norm(update)
                
                if w_norm > 0 and g_norm > 0:
                    trust_ratio = trust_coef * w_norm / g_norm
                    trust_ratio = max(min(trust_ratio, upper_bound), lower_bound)
                else:
                    trust_ratio = 1.0
                
                # 参数更新
                param.data = param.data - lr * trust_ratio * update
        
        return loss


class LARS(Optimizer):
    """
    LARS优化器 (Layer-wise Adaptive Rate Scaling)
    
    专为大批量训练设计，根据权重和梯度的比率调整学习率
    
    local_lr = trust_coef * ||w|| / (||g|| + lambda * ||w||)
    w_t = w_{t-1} - local_lr * lr * (g + lambda * w_{t-1})
    """
    
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        trust_coef: float = 0.001,
        nesterov: bool = False,
        max_grad_norm: Optional[float] = None
    ):
        defaults = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'trust_coef': trust_coef,
            'nesterov': nesterov,
            'max_grad_norm': max_grad_norm
        }
        super().__init__(params, defaults)
    
    @staticmethod
    def _norm(tensor):
        if isinstance(tensor, (int, float)):
            return abs(tensor)
        return math.sqrt(sum(x ** 2 for x in tensor)) if isinstance(tensor, list) else abs(tensor)
    
    def step(self, closure: Optional[Callable] = None):
        """执行LARS更新"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            trust_coef = group['trust_coef']
            nesterov = group['nesterov']
            max_grad_norm = group['max_grad_norm']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                
                grad = param.grad
                state = self._get_state(param)
                
                if len(state) == 0:
                    state['momentum_buffer'] = 0.0
                
                # 梯度裁剪
                if max_grad_norm is not None:
                    grad_norm = self._norm(grad)
                    if grad_norm > max_grad_norm:
                        grad = grad * max_grad_norm / grad_norm
                
                # 添加权重衰减
                if weight_decay != 0:
                    grad = grad + weight_decay * param.data
                
                # 计算层级学习率
                w_norm = self._norm(param.data)
                g_norm = self._norm(grad)
                
                if w_norm > 0 and g_norm > 0:
                    local_lr = trust_coef * w_norm / g_norm
                else:
                    local_lr = 1.0
                
                # 动量更新
                buf = state['momentum_buffer']
                buf = momentum * buf + local_lr * lr * grad
                state['momentum_buffer'] = buf
                
                if nesterov:
                    update = local_lr * lr * grad + momentum * buf
                else:
                    update = buf
                
                param.data = param.data - update
        
        return loss


class RMSprop(Optimizer):
    """
    RMSprop优化器
    
    v_t = alpha * v_{t-1} + (1 - alpha) * g_t^2
    theta_t = theta_{t-1} - lr * g_t / (sqrt(v_t) + eps)
    """
    
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False
    ):
        defaults = {
            'lr': lr,
            'alpha': alpha,
            'eps': eps,
            'weight_decay': weight_decay,
            'momentum': momentum,
            'centered': centered
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        """执行RMSprop更新"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            eps = group['eps']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            centered = group['centered']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                
                grad = param.grad
                state = self._get_state(param)
                
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = 0.0
                    if momentum > 0:
                        state['momentum_buffer'] = 0.0
                    if centered:
                        state['grad_avg'] = 0.0
                
                state['step'] += 1
                
                square_avg = state['square_avg']
                
                if weight_decay != 0:
                    grad = grad + weight_decay * param.data
                
                square_avg = alpha * square_avg + (1 - alpha) * (grad ** 2)
                state['square_avg'] = square_avg
                
                if centered:
                    grad_avg = state['grad_avg']
                    grad_avg = alpha * grad_avg + (1 - alpha) * grad
                    state['grad_avg'] = grad_avg
                    avg = square_avg - grad_avg ** 2
                else:
                    avg = square_avg
                
                if momentum > 0:
                    buf = state['momentum_buffer']
                    buf = momentum * buf + grad / (math.sqrt(avg) + eps)
                    state['momentum_buffer'] = buf
                    param.data = param.data - lr * buf
                else:
                    param.data = param.data - lr * grad / (math.sqrt(avg) + eps)
        
        return loss


class Adagrad(Optimizer):
    """
    Adagrad优化器 - 自适应学习率
    
    G_t = G_{t-1} + g_t^2
    theta_t = theta_{t-1} - lr * g_t / (sqrt(G_t) + eps)
    """
    
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.01,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        eps: float = 1e-10
    ):
        defaults = {
            'lr': lr,
            'lr_decay': lr_decay,
            'weight_decay': weight_decay,
            'initial_accumulator_value': initial_accumulator_value,
            'eps': eps
        }
        super().__init__(params, defaults)
        
        # 初始化累加器
        for group in self.param_groups:
            for param in group['params']:
                state = self._get_state(param)
                state['sum'] = initial_accumulator_value
    
    def step(self, closure: Optional[Callable] = None):
        """执行Adagrad更新"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            lr_decay = group['lr_decay']
            weight_decay = group['weight_decay']
            eps = group['eps']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                
                grad = param.grad
                state = self._get_state(param)
                
                if 'step' not in state:
                    state['step'] = 0
                
                state['step'] += 1
                step = state['step']
                
                if weight_decay != 0:
                    grad = grad + weight_decay * param.data
                
                # 更新累加器
                state['sum'] = state['sum'] + grad ** 2
                
                # 计算学习率衰减
                clr = lr / (1 + (step - 1) * lr_decay)
                
                # 参数更新
                param.data = param.data - clr * grad / (math.sqrt(state['sum']) + eps)
        
        return loss


class Adadelta(Optimizer):
    """
    Adadelta优化器 - Adagrad的扩展，限制历史窗口
    
    E[g^2]_t = rho * E[g^2]_{t-1} + (1 - rho) * g_t^2
    Delta_x = - sqrt(E[Delta_x^2]_{t-1} + eps) / sqrt(E[g^2]_t + eps) * g_t
    E[Delta_x^2]_t = rho * E[Delta_x^2]_{t-1} + (1 - rho) * Delta_x^2
    """
    
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0.0
    ):
        defaults = {
            'lr': lr,
            'rho': rho,
            'eps': eps,
            'weight_decay': weight_decay
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        """执行Adadelta更新"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            rho = group['rho']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                
                grad = param.grad
                state = self._get_state(param)
                
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = 0.0
                    state['acc_delta'] = 0.0
                
                state['step'] += 1
                
                if weight_decay != 0:
                    grad = grad + weight_decay * param.data
                
                square_avg = state['square_avg']
                acc_delta = state['acc_delta']
                
                # 更新梯度平方的指数移动平均
                square_avg = rho * square_avg + (1 - rho) * (grad ** 2)
                state['square_avg'] = square_avg
                
                # 计算更新
                std = math.sqrt(acc_delta + eps)
                delta = std * grad / math.sqrt(square_avg + eps)
                
                # 更新参数
                param.data = param.data - lr * delta
                
                # 更新Delta平方的指数移动平均
                acc_delta = rho * acc_delta + (1 - rho) * (delta ** 2)
                state['acc_delta'] = acc_delta
        
        return loss


class RAdam(Optimizer):
    """
    RAdam优化器 (Rectified Adam)
    
    解决Adam在训练初期的不稳定性问题
    
    当t足够大时使用Adam，否则使用SGD-like更新
    """
    
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        n_sma_threshold: int = 5
    ):
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'n_sma_threshold': n_sma_threshold
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        """执行RAdam更新"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            n_sma_threshold = group['n_sma_threshold']
            
            beta2_t = beta2
            
            for param in group['params']:
                if param.grad is None:
                    continue
                
                grad = param.grad
                state = self._get_state(param)
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = 0.0
                    state['exp_avg_sq'] = 0.0
                
                state['step'] += 1
                step = state['step']
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                
                if weight_decay != 0:
                    grad = grad + weight_decay * param.data
                
                # 更新矩估计
                exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grad ** 2)
                state['exp_avg'] = exp_avg
                state['exp_avg_sq'] = exp_avg_sq
                
                # 计算n_sma (simple moving average的长度)
                beta2_t = beta2 ** step
                n_sma_max = 2 / (1 - beta2) - 1
                n_sma = n_sma_max - 2 * step * beta2_t / (1 - beta2_t)
                
                # 偏差修正
                bias_correction1 = 1 - beta1 ** step
                
                if n_sma >= n_sma_threshold:
                    # 使用Adam更新
                    bias_correction2 = 1 - beta2_t
                    
                    # 计算rectification项
                    rect = math.sqrt(
                        (n_sma - 4) / (n_sma_max - 4) *
                        (n_sma - 2) / (n_sma_max - 2) *
                        n_sma_max / n_sma
                    )
                    
                    step_size = lr * rect * math.sqrt(bias_correction2) / bias_correction1
                    denom = math.sqrt(exp_avg_sq) + eps
                    
                    param.data = param.data - step_size * exp_avg / denom
                else:
                    # 使用SGD-like更新
                    step_size = lr / bias_correction1
                    param.data = param.data - step_size * exp_avg
        
        return loss


class Nadam(Optimizer):
    """
    Nadam优化器 (Nesterov-accelerated Adam)
    
    结合了Nesterov动量和Adam
    
    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    m_hat = m_t / (1 - beta1^t) + (1 - beta1) * g_t / (1 - beta1^t)
    """
    
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum_decay: float = 0.004
    ):
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'momentum_decay': momentum_decay
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        """执行Nadam更新"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            momentum_decay = group['momentum_decay']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                
                grad = param.grad
                state = self._get_state(param)
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = 0.0
                    state['exp_avg_sq'] = 0.0
                
                state['step'] += 1
                step = state['step']
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                
                if weight_decay != 0:
                    grad = grad + weight_decay * param.data
                
                # 更新二阶矩
                exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grad ** 2)
                state['exp_avg_sq'] = exp_avg_sq
                
                # 更新一阶矩
                exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                state['exp_avg'] = exp_avg
                
                # 偏差修正
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                # Nadam的Nesterov项
                nesterov_weight = (1 - beta1) / bias_correction1
                nesterov_momentum = beta1 * (1 - beta1 ** (step - 1)) / bias_correction1
                
                m_hat = nesterov_momentum * exp_avg + nesterov_weight * grad
                
                step_size = lr * math.sqrt(bias_correction2)
                denom = math.sqrt(exp_avg_sq) + eps
                
                param.data = param.data - step_size * m_hat / denom
        
        return loss


class Lookahead(Optimizer):
    """
    Lookahead优化器
    
    维护两组参数: 快参数和慢参数
    每k步将慢参数向快参数方向更新
    
    theta_slow = theta_slow + alpha * (theta_fast - theta_slow)
    """
    
    def __init__(
        self,
        base_optimizer: Optimizer,
        alpha: float = 0.5,
        k: int = 6,
        pullback_momentum: str = "none"
    ):
        self.base_optimizer = base_optimizer
        self.alpha = alpha
        self.k = k
        self.pullback_momentum = pullback_momentum
        
        # 复制基优化器的属性
        self.params = base_optimizer.params
        self.defaults = base_optimizer.defaults
        self.state = base_optimizer.state
        self.param_groups = base_optimizer.param_groups
        
        # 初始化慢参数
        self.slow_params = []
        for param in self.params:
            self.slow_params.append(param.data)
        
        self._step_counter = 0
    
    def step(self, closure: Optional[Callable] = None):
        """执行Lookahead更新"""
        loss = self.base_optimizer.step(closure)
        self._step_counter += 1
        
        if self._step_counter % self.k == 0:
            # 更新慢参数
            for fast_param, slow_param in zip(self.params, self.slow_params):
                slow_param = slow_param + self.alpha * (fast_param.data - slow_param)
                fast_param.data = slow_param
            
            # 处理动量
            if self.pullback_momentum == "pullback":
                for param in self.params:
                    state = self._get_state(param)
                    if 'momentum_buffer' in state:
                        state['momentum_buffer'] = state['momentum_buffer'] * self.alpha
            elif self.pullback_momentum == "reset":
                for param in self.params:
                    state = self._get_state(param)
                    if 'momentum_buffer' in state:
                        state['momentum_buffer'] = 0.0
        
        return loss
    
    def _get_state(self, param: Tensor) -> Dict:
        return self.base_optimizer._get_state(param)
    
    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none)


class NovoGrad(Optimizer):
    """
    NovoGrad优化器
    
    一种用于大批量训练的优化器，结合了Adam和LARS的优点
    
    v_t = beta2 * v_{t-1} + (1 - beta2) * ||g_t||^2
    m_t = beta1 * m_{t-1} + g_t / sqrt(v_t)
    """
    
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.95, 0.98),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        grad_averaging: bool = True
    ):
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'grad_averaging': grad_averaging
        }
        super().__init__(params, defaults)
    
    @staticmethod
    def _norm(tensor):
        if isinstance(tensor, (int, float)):
            return abs(tensor)
        return math.sqrt(sum(x ** 2 for x in tensor)) if isinstance(tensor, list) else abs(tensor)
    
    def step(self, closure: Optional[Callable] = None):
        """执行NovoGrad更新"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            grad_averaging = group['grad_averaging']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                
                grad = param.grad
                state = self._get_state(param)
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = 0.0
                    state['grad_norms_avg'] = 0.0
                
                state['step'] += 1
                
                exp_avg = state['exp_avg']
                grad_norms_avg = state['grad_norms_avg']
                
                # 计算梯度范数
                grad_norm = self._norm(grad)
                
                # 更新梯度范数的指数移动平均
                grad_norms_avg = beta2 * grad_norms_avg + (1 - beta2) * (grad_norm ** 2)
                state['grad_norms_avg'] = grad_norms_avg
                
                # 权重衰减
                if weight_decay != 0:
                    grad = grad + weight_decay * param.data
                
                # 计算归一化梯度
                denom = math.sqrt(grad_norms_avg) + eps
                normalized_grad = grad / denom
                
                # 更新动量
                if grad_averaging:
                    exp_avg = beta1 * exp_avg + (1 - beta1) * normalized_grad
                else:
                    exp_avg = beta1 * exp_avg + normalized_grad
                state['exp_avg'] = exp_avg
                
                # 参数更新
                param.data = param.data - lr * exp_avg
        
        return loss


class Shampoo(Optimizer):
    """
    Shampoo优化器
    
    使用预调节矩阵的优化器，适用于高维问题
    
    L_t = beta * L_{t-1} + (1 - beta) * g_t @ g_t.T
    G_t = L_t^{1/4}
    theta_t = theta_{t-1} - lr * G_t^{-1} @ g_t
    """
    
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        epsilon: float = 1e-4,
        update_freq: int = 1
    ):
        defaults = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'epsilon': epsilon,
            'update_freq': update_freq
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        """执行Shampoo更新"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            epsilon = group['epsilon']
            update_freq = group['update_freq']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                
                grad = param.grad
                state = self._get_state(param)
                
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = 0.0
                    state['preconditioner'] = epsilon
                
                state['step'] += 1
                step = state['step']
                
                if weight_decay != 0:
                    grad = grad + weight_decay * param.data
                
                # 更新预调节器 (简化版本)
                state['preconditioner'] = 0.9 * state['preconditioner'] + 0.1 * (grad ** 2)
                
                # 计算预调节后的梯度
                preconditioned_grad = grad / (math.sqrt(state['preconditioner']) + epsilon)
                
                # 动量更新
                buf = state['momentum_buffer']
                buf = momentum * buf + preconditioned_grad
                state['momentum_buffer'] = buf
                
                # 参数更新
                if step % update_freq == 0:
                    param.data = param.data - lr * buf
        
        return loss


class GradientAccumulator:
    """梯度累积器，用于大批量训练"""
    
    def __init__(self, optimizer: Optimizer, accumulation_steps: int = 1):
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self._step_counter = 0
    
    def step(self, closure: Optional[Callable] = None):
        """累积梯度并条件性更新"""
        self._step_counter += 1
        
        if self._step_counter % self.accumulation_steps == 0:
            # 缩放梯度
            for param in self.optimizer.params:
                if param.grad is not None:
                    param.grad = param.grad / self.accumulation_steps
            
            # 执行优化步骤
            loss = self.optimizer.step(closure)
            self.optimizer.zero_grad()
            return loss
        
        return None


class GradientClipper:
    """梯度裁剪器"""
    
    @staticmethod
    def clip_by_norm(params: List[Tensor], max_norm: float) -> float:
        """按范数裁剪梯度"""
        total_norm = 0.0
        
        for param in params:
            if param.grad is not None:
                if isinstance(param.grad, (int, float)):
                    param_norm = abs(param.grad)
                else:
                    param_norm = math.sqrt(sum(g ** 2 for g in param.grad)) if isinstance(param.grad, list) else abs(param.grad)
                total_norm += param_norm ** 2
        
        total_norm = math.sqrt(total_norm)
        
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for param in params:
                if param.grad is not None:
                    param.grad = param.grad * clip_coef
        
        return total_norm
    
    @staticmethod
    def clip_by_value(params: List[Tensor], min_value: float, max_value: float):
        """按值裁剪梯度"""
        for param in params:
            if param.grad is not None:
                if isinstance(param.grad, (int, float)):
                    param.grad = max(min_value, min(max_value, param.grad))
                elif isinstance(param.grad, list):
                    param.grad = [max(min_value, min(max_value, g)) for g in param.grad]


# 工具函数
def get_optimizer(name: str, params: List[Tensor], **kwargs) -> Optimizer:
    """根据名称获取优化器"""
    optimizers = {
        'sgd': SGD,
        'adam': Adam,
        'adamw': AdamW,
        'lamb': LAMB,
        'lars': LARS,
        'rmsprop': RMSprop,
        'adagrad': Adagrad,
        'adadelta': Adadelta,
        'radam': RAdam,
        'nadam': Nadam,
        'novograd': NovoGrad,
        'shampoo': Shampoo
    }
    
    name_lower = name.lower()
    if name_lower not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Available: {list(optimizers.keys())}")
    
    return optimizers[name_lower](params, **kwargs)
