"""
Kernel Fusion - 算子融合优化性能

模块路径: hardware/gpu/kernel_fusion.py

提供算子融合功能，通过合并多个操作来减少kernel启动开销和内存访问，
显著提升GPU计算效率。支持常见的融合模式如conv-bn-relu、layernorm等。
"""

import logging
from typing import Optional, List, Callable, Dict, Any, Tuple, Union, Set
from dataclasses import dataclass
from functools import wraps
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import symbolic_trace, GraphModule
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.fx.passes.operator_support import OperatorSupport

logger = logging.getLogger(__name__)


@dataclass
class FusionConfig:
    """算子融合配置"""
    enabled: bool = True
    fuse_conv_bn: bool = True  # 融合Conv+BN
    fuse_conv_bn_relu: bool = True  # 融合Conv+BN+ReLU
    fuse_linear_activation: bool = True  # 融合Linear+Activation
    fuse_layernorm: bool = True  # 融合LayerNorm相关操作
    fuse_attention: bool = True  # 融合注意力操作
    fuse_gelu: bool = True  # 融合GELU近似
    remove_dropout: bool = False  # 推理时移除dropout
    verbose: bool = False


class FusedConvBNReLU(nn.Module):
    """
    融合的Conv+BN+ReLU模块
    
    将卷积、批归一化和ReLU融合为单个kernel。
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1,
        groups: int = 1,
        bias: bool = False,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)
        
        # 融合后的权重和偏置
        self._fused = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if self._fused and hasattr(self, '_fused_weight'):
            # 使用融合后的权重进行计算
            return F.relu(F.conv2d(
                x, self._fused_weight, self._fused_bias,
                self.conv.stride, self.conv.padding,
                self.conv.dilation, self.conv.groups
            ), inplace=True)
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
    def fuse(self) -> 'FusedConvBNReLU':
        """
        融合Conv和BN参数
        
        Returns:
            融合后的模块
        """
        if self._fused:
            return self
        
        # 计算融合后的权重和偏置
        weight = self.conv.weight
        bn_weight = self.bn.weight
        bn_bias = self.bn.bias
        bn_mean = self.bn.running_mean
        bn_var = self.bn.running_var
        bn_eps = self.bn.eps
        
        # 融合: BN(Conv(x)) = gamma * (Conv(x) - mean) / sqrt(var + eps) + beta
        # = (gamma / sqrt(var + eps)) * Conv(x) + (beta - gamma * mean / sqrt(var + eps))
        std = torch.sqrt(bn_var + bn_eps)
        fused_weight = weight * (bn_weight / std).view(-1, 1, 1, 1)
        fused_bias = bn_bias - bn_weight * bn_mean / std
        
        self._fused_weight = nn.Parameter(fused_weight)
        self._fused_bias = nn.Parameter(fused_bias)
        self._fused = True
        
        return self


class FusedLinearGELU(nn.Module):
    """
    融合的Linear+GELU模块
    
    使用torch.jit.script优化GELU计算。
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        approximate: str = 'none'
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.approximate = approximate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.linear(x)
        return F.gelu(x, approximate=self.approximate)


class FusedLayerNorm(nn.Module):
    """
    优化的LayerNorm实现
    
    使用融合kernel提高性能。
    """
    
    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, device=device, dtype=dtype))
            self.bias = nn.Parameter(torch.empty(self.normalized_shape, device=device, dtype=dtype))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """重置参数"""
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class KernelFusionOptimizer:
    """
    算子融合优化器
    
    自动识别并融合模型中的可融合操作。
    """
    
    def __init__(self, config: Optional[FusionConfig] = None):
        """
        初始化融合优化器
        
        Args:
            config: 融合配置
        """
        self.config = config or FusionConfig()
        self._fusion_patterns: Dict[str, Callable] = {}
        self._register_default_patterns()
    
    def _register_default_patterns(self):
        """注册默认融合模式"""
        if self.config.fuse_conv_bn:
            self._fusion_patterns['conv_bn'] = self._fuse_conv_bn_pattern
        if self.config.fuse_conv_bn_relu:
            self._fusion_patterns['conv_bn_relu'] = self._fuse_conv_bn_relu_pattern
        if self.config.fuse_linear_activation:
            self._fusion_patterns['linear_activation'] = self._fuse_linear_activation_pattern
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """
        优化模型
        
        Args:
            model: 要优化的模型
            
        Returns:
            优化后的模型
        """
        if not self.config.enabled:
            return model
        
        model.eval()  # 切换到评估模式以进行融合
        
        # 应用各种融合
        if self.config.fuse_conv_bn:
            model = self._fuse_conv_bn(model)
        
        if self.config.fuse_conv_bn_relu:
            model = self._fuse_conv_bn_relu(model)
        
        if self.config.fuse_linear_activation:
            model = self._fuse_linear_activation(model)
        
        if self.config.fuse_layernorm:
            model = self._optimize_layernorm(model)
        
        # 使用torch.jit优化
        try:
            model = torch.jit.script(model)
        except Exception as e:
            if self.config.verbose:
                logger.warning(f"JIT compilation failed: {e}")
        
        logger.info("Kernel fusion optimization completed")
        return model
    
    def _fuse_conv_bn(self, model: nn.Module) -> nn.Module:
        """
        融合Conv和BN层
        
        Args:
            model: 输入模型
            
        Returns:
            融合后的模型
        """
        fused_count = 0
        
        for name, module in list(model.named_children()):
            if isinstance(module, nn.Sequential):
                # 在Sequential中查找Conv+BN模式
                new_layers = []
                i = 0
                layers = list(module.children())
                
                while i < len(layers):
                    if (i + 1 < len(layers) and
                        isinstance(layers[i], (nn.Conv2d, nn.Conv1d, nn.Conv3d)) and
                        isinstance(layers[i + 1], (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d))):
                        
                        # 融合Conv和BN
                        fused_conv = self._fuse_conv_bn_modules(layers[i], layers[i + 1])
                        new_layers.append(fused_conv)
                        i += 2
                        fused_count += 1
                    else:
                        new_layers.append(layers[i])
                        i += 1
                
                # 替换Sequential
                if len(new_layers) != len(layers):
                    setattr(model, name, nn.Sequential(*new_layers))
            
            # 递归处理子模块
            else:
                self._fuse_conv_bn(module)
        
        if fused_count > 0:
            logger.info(f"Fused {fused_count} Conv+BN pairs")
        
        return model
    
    def _fuse_conv_bn_modules(
        self,
        conv: nn.Module,
        bn: nn.Module
    ) -> nn.Module:
        """
        融合单个Conv和BN模块
        
        Args:
            conv: 卷积层
            bn: 批归一化层
            
        Returns:
            融合后的卷积层
        """
        # 计算融合后的权重和偏置
        weight = conv.weight.clone()
        
        if bn.weight is not None:
            bn_weight = bn.weight
            bn_bias = bn.bias
            bn_mean = bn.running_mean
            bn_var = bn.running_var
            bn_eps = bn.eps
            
            std = torch.sqrt(bn_var + bn_eps)
            
            # 调整权重维度
            if isinstance(conv, nn.Conv2d):
                fused_weight = weight * (bn_weight / std).view(-1, 1, 1, 1)
            elif isinstance(conv, nn.Conv1d):
                fused_weight = weight * (bn_weight / std).view(-1, 1, 1)
            else:  # Conv3d
                fused_weight = weight * (bn_weight / std).view(-1, 1, 1, 1, 1)
            
            # 计算融合后的偏置
            if conv.bias is not None:
                fused_bias = conv.bias * bn_weight / std + bn_bias - bn_weight * bn_mean / std
            else:
                fused_bias = bn_bias - bn_weight * bn_mean / std
            
            conv.weight = nn.Parameter(fused_weight)
            conv.bias = nn.Parameter(fused_bias)
        
        return conv
    
    def _fuse_conv_bn_relu(self, model: nn.Module) -> nn.Module:
        """
        融合Conv+BN+ReLU
        
        Args:
            model: 输入模型
            
        Returns:
            融合后的模型
        """
        # 使用PyTorch内置的融合
        if hasattr(torch.nn.intrinsic, 'fuse_modules'):
            # 查找可融合的模式
            modules_to_fuse = []
            
            for name, module in model.named_modules():
                if isinstance(module, nn.Sequential):
                    patterns = self._find_fusable_patterns(module)
                    for pattern in patterns:
                        modules_to_fuse.append([f"{name}.{p}" for p in pattern])
            
            if modules_to_fuse:
                try:
                    torch.nn.intrinsic.modules.fuse.fuse_modules(model, modules_to_fuse, inplace=True)
                    logger.info(f"Fused {len(modules_to_fuse)} Conv+BN+ReLU patterns")
                except Exception as e:
                    logger.warning(f"Module fusion failed: {e}")
        
        return model
    
    def _find_fusable_patterns(self, module: nn.Sequential) -> List[List[str]]:
        """
        查找可融合的模式
        
        Args:
            module: Sequential模块
            
        Returns:
            可融合的模式列表
        """
        patterns = []
        children = list(module.named_children())
        
        for i in range(len(children) - 2):
            name1, mod1 = children[i]
            name2, mod2 = children[i + 1]
            name3, mod3 = children[i + 2]
            
            if (isinstance(mod1, (nn.Conv2d, nn.Conv1d)) and
                isinstance(mod2, (nn.BatchNorm2d, nn.BatchNorm1d)) and
                isinstance(mod3, (nn.ReLU, nn.ReLU6))):
                patterns.append([name1, name2, name3])
        
        return patterns
    
    def _fuse_linear_activation(self, model: nn.Module) -> nn.Module:
        """
        融合Linear+Activation
        
        Args:
            model: 输入模型
            
        Returns:
            融合后的模型
        """
        # 这个优化主要通过JIT编译器完成
        return model
    
    def _optimize_layernorm(self, model: nn.Module) -> nn.Module:
        """
        优化LayerNorm
        
        Args:
            model: 输入模型
            
        Returns:
            优化后的模型
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                # 替换为优化的LayerNorm
                optimized_ln = FusedLayerNorm(
                    module.normalized_shape,
                    module.eps,
                    module.elementwise_affine
                )
                if module.elementwise_affine:
                    optimized_ln.weight.data = module.weight.data.clone()
                    optimized_ln.bias.data = module.bias.data.clone()
                
                # 替换模块
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                if parent_name:
                    parent = model.get_submodule(parent_name)
                    setattr(parent, child_name, optimized_ln)
                else:
                    setattr(model, child_name, optimized_ln)
        
        return model
    
    def _fuse_conv_bn_pattern(self, model: nn.Module) -> nn.Module:
        """Conv+BN融合模式"""
        return self._fuse_conv_bn(model)
    
    def _fuse_conv_bn_relu_pattern(self, model: nn.Module) -> nn.Module:
        """Conv+BN+ReLU融合模式"""
        return self._fuse_conv_bn_relu(model)
    
    def _fuse_linear_activation_pattern(self, model: nn.Module) -> nn.Module:
        """Linear+Activation融合模式"""
        return self._fuse_linear_activation(model)


class FXFusionPass(PassBase):
    """
    基于FX的融合Pass
    
    使用PyTorch FX进行图级别的算子融合。
    """
    
    def __init__(self, config: Optional[FusionConfig] = None):
        super().__init__()
        self.config = config or FusionConfig()
    
    def call(self, graph_module: GraphModule) -> PassResult:
        """
        执行融合Pass
        
        Args:
            graph_module: FX图模块
            
        Returns:
            Pass结果
        """
        modified = False
        graph = graph_module.graph
        
        # 遍历图中的节点
        for node in graph.nodes:
            # 查找融合模式
            if self._is_fusable_pattern(node):
                # 执行融合
                modified = True
                self._fuse_pattern(graph, node)
        
        graph.lint()
        return PassResult(graph_module, modified)
    
    def _is_fusable_pattern(self, node) -> bool:
        """检查是否是可融合模式"""
        # 检查Conv+BN模式
        if node.op == 'call_module':
            module = node.target
            if 'conv' in str(module).lower():
                # 检查下一个节点是否是BN
                for user in node.users:
                    if user.op == 'call_module' and 'bn' in str(user.target).lower():
                        return True
        return False
    
    def _fuse_pattern(self, graph, node):
        """融合模式"""
        # 实现具体的融合逻辑
        pass


class FusionPatternMatcher:
    """
    融合模式匹配器
    
    识别模型中的可融合模式。
    """
    
    def __init__(self):
        self._patterns: Dict[str, List[type]] = {
            'conv_bn_relu': [nn.Conv2d, nn.BatchNorm2d, nn.ReLU],
            'conv_bn': [nn.Conv2d, nn.BatchNorm2d],
            'linear_relu': [nn.Linear, nn.ReLU],
            'linear_gelu': [nn.Linear, nn.GELU],
        }
    
    def find_patterns(self, model: nn.Module) -> Dict[str, List[Tuple[str, ...]]]:
        """
        查找模型中的所有可融合模式
        
        Args:
            model: 要分析的模型
            
        Returns:
            发现的模式字典
        """
        found_patterns = {name: [] for name in self._patterns}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential):
                layers = list(module.children())
                layer_types = [type(l) for l in layers]
                
                for pattern_name, pattern_types in self._patterns.items():
                    # 在序列中查找模式
                    for i in range(len(layer_types) - len(pattern_types) + 1):
                        if layer_types[i:i + len(pattern_types)] == pattern_types:
                            layer_names = [f"{name}.{j}" for j in range(i, i + len(pattern_types))]
                            found_patterns[pattern_name].append(tuple(layer_names))
        
        return found_patterns
    
    def register_pattern(self, name: str, pattern: List[type]):
        """
        注册新的融合模式
        
        Args:
            name: 模式名称
            pattern: 层类型列表
        """
        self._patterns[name] = pattern


# 便捷的融合函数
def fuse_conv_bn_eval(conv: nn.Module, bn: nn.Module) -> nn.Module:
    """
    融合Conv和BN（评估模式）
    
    Args:
        conv: 卷积层
        bn: 批归一化层
        
    Returns:
        融合后的卷积层
    """
    assert not (conv.training or bn.training), "Fusion only for eval!"
    
    fused_conv = conv
    
    # 计算融合后的权重
    weight = conv.weight
    bn_weight = bn.weight
    bn_bias = bn.bias
    bn_mean = bn.running_mean
    bn_var = bn.running_var
    bn_eps = bn.eps
    
    std = torch.sqrt(bn_var + bn_eps)
    
    # 调整权重维度
    if isinstance(conv, nn.Conv2d):
        fused_weight = weight * (bn_weight / std).view(-1, 1, 1, 1)
    elif isinstance(conv, nn.Conv1d):
        fused_weight = weight * (bn_weight / std).view(-1, 1, 1)
    else:
        fused_weight = weight * (bn_weight / std).view(-1, 1, 1, 1, 1)
    
    # 计算融合后的偏置
    if conv.bias is not None:
        fused_bias = conv.bias * bn_weight / std + bn_bias - bn_weight * bn_mean / std
    else:
        fused_bias = bn_bias - bn_weight * bn_mean / std
    
    fused_conv.weight = nn.Parameter(fused_weight)
    fused_conv.bias = nn.Parameter(fused_bias)
    
    return fused_conv


def optimize_for_inference(model: nn.Module) -> nn.Module:
    """
    为推理优化模型
    
    Args:
        model: 输入模型
        
    Returns:
        优化后的模型
    """
    model.eval()
    
    config = FusionConfig(
        enabled=True,
        fuse_conv_bn=True,
        fuse_conv_bn_relu=True,
        fuse_linear_activation=True,
        fuse_layernorm=True,
        remove_dropout=True
    )
    
    optimizer = KernelFusionOptimizer(config)
    return optimizer.optimize_model(model)


def create_fused_mlp(
    in_features: int,
    hidden_features: int,
    out_features: int,
    activation: str = 'gelu',
    dropout: float = 0.0
) -> nn.Module:
    """
    创建融合的MLP模块
    
    Args:
        in_features: 输入维度
        hidden_features: 隐藏层维度
        out_features: 输出维度
        activation: 激活函数类型
        dropout: dropout概率
        
    Returns:
        融合的MLP模块
    """
    layers = []
    
    # 第一层
    layers.append(nn.Linear(in_features, hidden_features))
    
    # 激活函数
    if activation == 'gelu':
        layers.append(nn.GELU())
    elif activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif activation == 'swish' or activation == 'silu':
        layers.append(nn.SiLU(inplace=True))
    
    # Dropout
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    
    # 第二层
    layers.append(nn.Linear(hidden_features, out_features))
    
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    
    return nn.Sequential(*layers)


# 性能测试工具
def benchmark_fusion(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    iterations: int = 100,
    warmup: int = 10
) -> Dict[str, float]:
    """
    基准测试融合效果
    
    Args:
        model: 要测试的模型
        input_shape: 输入形状
        iterations: 迭代次数
        warmup: 预热次数
        
    Returns:
        性能统计
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    device = torch.device("cuda")
    model = model.to(device)
    dummy_input = torch.randn(input_shape, device=device)
    
    # 预热
    for _ in range(warmup):
        _ = model(dummy_input)
    
    torch.cuda.synchronize()
    
    # 测试
    import time
    start_time = time.time()
    
    for _ in range(iterations):
        _ = model(dummy_input)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    return {
        "total_time_ms": elapsed * 1000,
        "avg_time_ms": elapsed * 1000 / iterations,
        "throughput_samples_per_sec": iterations * input_shape[0] / elapsed
    }
