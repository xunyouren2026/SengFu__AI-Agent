"""
GaLore: Gradient Low-Rank Projection
梯度低秩投影优化器实现

基于论文 "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection"
支持8bit量化梯度，显著降低训练内存占用
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Dict, List, Optional, Tuple, Any
import math
from dataclasses import dataclass


@dataclass
class GaLoreConfig:
    """GaLore配置参数"""
    rank: int = 128  # 低秩投影的秩
    update_proj_gap: int = 200  # 投影矩阵更新间隔
    scale: float = 1.0  # 缩放因子
    proj_type: str = "std"  # 投影类型: std, reverse_std, right, left
    quantize: bool = False  # 是否启用8bit量化
    quantize_bits: int = 8  # 量化位数


class GaLoreOptimizer(Optimizer):
    """
    GaLore优化器包装器
    将梯度投影到低秩空间进行优化，减少内存占用
    """
    
    def __init__(
        self,
        params,
        base_optimizer: type = torch.optim.AdamW,
        rank: int = 128,
        update_proj_gap: int = 200,
        scale: float = 1.0,
        proj_type: str = "std",
        quantize: bool = False,
        quantize_bits: int = 8,
        **base_optimizer_kwargs
    ):
        """
        初始化GaLore优化器
        
        Args:
            params: 模型参数
            base_optimizer: 基础优化器类
            rank: 低秩投影的秩
            update_proj_gap: 投影矩阵更新间隔
            scale: 缩放因子
            proj_type: 投影类型
            quantize: 是否启用量化
            quantize_bits: 量化位数
            **base_optimizer_kwargs: 基础优化器的其他参数
        """
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.proj_type = proj_type
        self.quantize = quantize
        self.quantize_bits = quantize_bits
        
        # 存储每个参数的状态
        self.state = {}
        self._init_state()
        
        # 创建基础优化器，但使用投影后的参数
        self.base_optimizer = base_optimizer(params, **base_optimizer_kwargs)
        
        defaults = dict(
            rank=rank,
            update_proj_gap=update_proj_gap,
            scale=scale,
            proj_type=proj_type,
            quantize=quantize,
            quantize_bits=quantize_bits
        )
        super().__init__(params, defaults)
    
    def _init_state(self):
        """初始化状态字典"""
        self.state['step'] = 0
        self.state['projections'] = {}
    
    def _get_projection_matrix(self, param: torch.Tensor, rank: int) -> torch.Tensor:
        """
        获取或创建投影矩阵
        
        Args:
            param: 参数张量
            rank: 投影秩
            
        Returns:
            投影矩阵
        """
        param_id = id(param)
        
        if param_id not in self.state['projections']:
            # 初始化投影矩阵
            if len(param.shape) == 2:
                m, n = param.shape
                if self.proj_type == "std":
                    # 标准投影: 对梯度进行低秩分解
                    projection = torch.randn(m, rank, device=param.device, dtype=param.dtype)
                    projection = torch.linalg.qr(projection)[0]  # 正交化
                elif self.proj_type == "reverse_std":
                    projection = torch.randn(n, rank, device=param.device, dtype=param.dtype)
                    projection = torch.linalg.qr(projection)[0]
                elif self.proj_type == "left":
                    projection = torch.randn(m, rank, device=param.device, dtype=param.dtype)
                    projection = torch.linalg.qr(projection)[0]
                elif self.proj_type == "right":
                    projection = torch.randn(n, rank, device=param.device, dtype=param.dtype)
                    projection = torch.linalg.qr(projection)[0]
                else:
                    raise ValueError(f"Unknown proj_type: {self.proj_type}")
                
                self.state['projections'][param_id] = projection
        
        return self.state['projections'].get(param_id, None)
    
    def _update_projection(self, param: torch.Tensor, grad: torch.Tensor):
        """
        更新投影矩阵
        
        Args:
            param: 参数张量
            grad: 梯度张量
        """
        param_id = id(param)
        
        if len(grad.shape) == 2:
            m, n = grad.shape
            if self.proj_type in ["std", "left"]:
                # 对梯度进行SVD分解获取左奇异向量
                try:
                    u, s, v = torch.svd_lowrank(grad, q=self.rank * 2, niter=2)
                    projection = u[:, :self.rank]
                except:
                    # 如果SVD失败，使用随机投影
                    projection = torch.randn(m, self.rank, device=grad.device, dtype=grad.dtype)
                    projection = torch.linalg.qr(projection)[0]
            else:  # reverse_std, right
                try:
                    u, s, v = torch.svd_lowrank(grad.t(), q=self.rank * 2, niter=2)
                    projection = u[:, :self.rank]
                except:
                    projection = torch.randn(n, self.rank, device=grad.device, dtype=grad.dtype)
                    projection = torch.linalg.qr(projection)[0]
            
            self.state['projections'][param_id] = projection
    
    def _project_gradient(self, grad: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
        """
        将梯度投影到低秩空间
        
        Args:
            grad: 原始梯度
            projection: 投影矩阵
            
        Returns:
            投影后的低秩梯度
        """
        if self.proj_type in ["std", "left"]:
            # G_proj = P^T @ G
            return torch.matmul(projection.t(), grad)
        else:  # reverse_std, right
            # G_proj = G @ P
            return torch.matmul(grad, projection)
    
    def _unproject_gradient(self, grad_proj: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
        """
        将低秩梯度反投影回原始空间
        
        Args:
            grad_proj: 投影后的低秩梯度
            projection: 投影矩阵
            
        Returns:
            反投影后的梯度
        """
        if self.proj_type in ["std", "left"]:
            # G = P @ G_proj
            return torch.matmul(projection, grad_proj)
        else:  # reverse_std, right
            # G = G_proj @ P^T
            return torch.matmul(grad_proj, projection.t())
    
    def _quantize_gradient(self, grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        8bit量化梯度
        
        Args:
            grad: 梯度张量
            
        Returns:
            (量化后的梯度, 量化尺度, 零点)
        """
        if not self.quantize or self.quantize_bits != 8:
            return grad, None, None
        
        # 计算量化参数
        grad_min = grad.min()
        grad_max = grad.max()
        scale = (grad_max - grad_min) / (2 ** self.quantize_bits - 1)
        zero_point = -grad_min / scale
        
        # 量化
        grad_quantized = torch.round((grad - grad_min) / scale).clamp(0, 2 ** self.quantize_bits - 1).to(torch.uint8)
        
        return grad_quantized, scale, zero_point
    
    def _dequantize_gradient(self, grad_quantized: torch.Tensor, scale: float, zero_point: float) -> torch.Tensor:
        """
        反量化梯度
        
        Args:
            grad_quantized: 量化后的梯度
            scale: 量化尺度
            zero_point: 零点
            
        Returns:
            反量化后的梯度
        """
        return grad_quantized.float() * scale + (grad_min := -zero_point * scale)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        执行优化步骤
        
        Args:
            closure: 可选的闭包函数
            
        Returns:
            损失值
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self.state['step'] += 1
        step = self.state['step']
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # 只处理2D权重矩阵
                if len(grad.shape) != 2:
                    # 对于非2D参数，直接使用基础优化器
                    continue
                
                # 定期更新投影矩阵
                if step % self.update_proj_gap == 1:
                    self._update_projection(p, grad)
                
                # 获取投影矩阵
                projection = self._get_projection_matrix(p, self.rank)
                if projection is None:
                    continue
                
                # 投影梯度到低秩空间
                grad_proj = self._project_gradient(grad, projection)
                
                # 可选：量化投影后的梯度
                if self.quantize:
                    grad_proj, scale, zero_point = self._quantize_gradient(grad_proj)
                    # 反量化用于计算（模拟量化效果）
                    if scale is not None:
                        grad_proj = self._dequantize_gradient(grad_proj, scale, zero_point)
                
                # 在低秩空间应用缩放
                grad_proj = grad_proj * self.scale
                
                # 反投影回原始空间
                grad_low_rank = self._unproject_gradient(grad_proj, projection)
                
                # 替换梯度
                p.grad.data = grad_low_rank
        
        # 使用基础优化器更新参数
        self.base_optimizer.step()
        self.base_optimizer.zero_grad()
        
        return loss
    
    def zero_grad(self, set_to_none: bool = False):
        """清零梯度"""
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
    
    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        state = {
            'base_optimizer': self.base_optimizer.state_dict(),
            'galore_state': {
                'step': self.state['step'],
                'projections': {id(k): v for k, v in self.state['projections'].items()}
            }
        }
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """加载状态字典"""
        self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
        self.state['step'] = state_dict['galore_state']['step']
        # 注意：投影矩阵需要在参数上重新建立映射


class GaLoreLayer(nn.Module):
    """
    支持GaLore的线性层
    在训练时自动应用低秩投影
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 128,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """重置参数"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return nn.functional.linear(x, self.weight, self.bias)


def apply_galore_to_model(
    model: nn.Module,
    rank: int = 128,
    update_proj_gap: int = 200,
    scale: float = 1.0,
    target_modules: Optional[List[str]] = None
) -> Dict[str, GaLoreConfig]:
    """
    为模型应用GaLore配置
    
    Args:
        model: 目标模型
        rank: 低秩投影的秩
        update_proj_gap: 投影矩阵更新间隔
        scale: 缩放因子
        target_modules: 目标模块名称列表，None表示所有线性层
        
    Returns:
        模块名称到配置的映射
    """
    configs = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 检查是否是目标模块
            if target_modules is not None:
                if not any(target in name for target in target_modules):
                    continue
            
            # 为模块创建配置
            config = GaLoreConfig(
                rank=min(rank, min(module.weight.shape)),  # 确保秩不超过维度
                update_proj_gap=update_proj_gap,
                scale=scale
            )
            configs[name] = config
    
    return configs


# 辅助函数
def create_galore_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    rank: int = 128,
    update_proj_gap: int = 200,
    quantize: bool = False,
    weight_decay: float = 0.01,
    **kwargs
) -> GaLoreOptimizer:
    """
    创建GaLore优化器的便捷函数
    
    Args:
        model: 模型
        lr: 学习率
        rank: 低秩投影的秩
        update_proj_gap: 投影矩阵更新间隔
        quantize: 是否启用量化
        weight_decay: 权重衰减
        **kwargs: 其他参数
        
    Returns:
        GaLore优化器实例
    """
    return GaLoreOptimizer(
        model.parameters(),
        base_optimizer=torch.optim.AdamW,
        lr=lr,
        rank=rank,
        update_proj_gap=update_proj_gap,
        quantize=quantize,
        weight_decay=weight_decay,
        **kwargs
    )
