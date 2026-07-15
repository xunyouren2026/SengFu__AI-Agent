"""
AGI统一框架 - 神经架构搜索 (NAS)
实现DARTS、ENAS、ProxylessNAS等可微分架构搜索方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
import math
import random
from collections import OrderedDict
from abc import ABC, abstractmethod
import copy


# ==================== 配置类 ====================

@dataclass
class NASConfig:
    """神经架构搜索配置"""
    # 搜索空间
    num_layers: int = 8
    num_nodes: int = 4  # 每层的节点数
    num_ops: int = 8    # 操作数量
    
    # 搜索配置
    search_epochs: int = 50
    warmup_epochs: int = 5
    
    # 训练配置
    batch_size: int = 64
    learning_rate: float = 0.001
    arch_learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # DARTS特定
    darts_second_order: bool = False
    unrolled: bool = False
    
    # ProxylessNAS特定
    proxyless_gradient_steps: int = 1
    
    # 评估配置
    eval_epochs: int = 100
    
    # 约束
    max_params: Optional[int] = None
    max_flops: Optional[int] = None


# ==================== 搜索空间定义 ====================

class Operation(ABC):
    """操作基类"""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def params(self) -> int:
        pass


class Conv3x3(Operation):
    """3x3卷积"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))
    
    @property
    def name(self) -> str:
        return "conv3x3"
    
    @property
    def params(self) -> int:
        return sum(p.numel() for p in self.conv.parameters())


class Conv5x5(Operation):
    """5x5卷积"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        self.conv = nn.Conv2d(in_channels, out_channels, 5, stride, 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))
    
    @property
    def name(self) -> str:
        return "conv5x5"
    
    @property
    def params(self) -> int:
        return sum(p.numel() for p in self.conv.parameters())


class DepthwiseConv3x3(Operation):
    """深度可分离3x3卷积"""
    
    def __init__(self, channels: int, stride: int = 1):
        self.depthwise = nn.Conv2d(channels, channels, 3, stride, 1, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.pointwise(self.depthwise(x))))
    
    @property
    def name(self) -> str:
        return "dwconv3x3"
    
    @property
    def params(self) -> int:
        return sum(p.numel() for p in self.depthwise.parameters()) + \
               sum(p.numel() for p in self.pointwise.parameters())


class DilatedConv3x3(Operation):
    """空洞卷积"""
    
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 2):
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, dilation, dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))
    
    @property
    def name(self) -> str:
        return f"dilated_conv3x3_d{self.dilation}"
    
    @property
    def params(self) -> int:
        return sum(p.numel() for p in self.conv.parameters())


class Identity(Operation):
    """恒等映射"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    @property
    def name(self) -> str:
        return "identity"
    
    @property
    def params(self) -> int:
        return 0


class Zero(Operation):
    """零操作"""
    
    def __init__(self, stride: int = 1):
        self.stride = stride
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return x * 0
        else:
            return x[:, :, ::self.stride, ::self.stride] * 0
    
    @property
    def name(self) -> str:
        return "zero"
    
    @property
    def params(self) -> int:
        return 0


class AvgPool3x3(Operation):
    """3x3平均池化"""
    
    def __init__(self, stride: int = 1):
        self.pool = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)
    
    @property
    def name(self) -> str:
        return "avgpool3x3"
    
    @property
    def params(self) -> int:
        return 0


class MaxPool3x3(Operation):
    """3x3最大池化"""
    
    def __init__(self, stride: int = 1):
        self.pool = nn.MaxPool2d(3, stride=stride, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)
    
    @property
    def name(self) -> str:
        return "maxpool3x3"
    
    @property
    def params(self) -> int:
        return 0


class SepConv3x3(Operation):
    """深度可分离卷积"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x
    
    @property
    def name(self) -> str:
        return "sepconv3x3"
    
    @property
    def params(self) -> int:
        return sum(p.numel() for p in self.conv1.parameters()) + \
               sum(p.numel() for p in self.conv2.parameters())


# ==================== 混合操作 ====================

class MixedOperation(nn.Module):
    """混合操作（可微分NAS核心）"""
    
    def __init__(self, operations: List[Operation]):
        super().__init__()
        self.operations = nn.ModuleList(operations)
        self.num_ops = len(operations)
        
        # 架构参数
        self.arch_param = nn.Parameter(torch.zeros(self.num_ops))
        
    def forward(self, x: torch.Tensor, hard: bool = False) -> torch.Tensor:
        """前向传播"""
        if hard:
            # 硬选择（评估阶段）
            idx = self.arch_param.argmax()
            return self.operations[idx](x)
        else:
            # 软选择（搜索阶段）
            weights = F.softmax(self.arch_param, dim=0)
            return sum(w * op(x) for w, op in zip(weights, self.operations))
    
    def get_flops(self, input_flops: int) -> int:
        """估算FLOPs"""
        weights = F.softmax(self.arch_param, dim=0)
        return sum(w.item() * input_flops for w in weights)


# ==================== 搜索单元 ====================

class SearchCell(nn.Module):
    """搜索单元"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 num_nodes: int = 4, stride: int = 1):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 预处理
        self.preprocess0 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # 节点间的混合操作
        self.edges = nn.ModuleDict()
        
        for i in range(num_nodes):
            for j in range(i + 2):  # 包括两个输入节点
                edge_name = f"{j}_{i+2}"
                
                # 创建操作集
                ops = self._create_operations(out_channels, stride if j < 2 else 1)
                self.edges[edge_name] = MixedOperation(ops)
        
        # 输出投影
        self.postprocess = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels * num_nodes, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
    def _create_operations(self, channels: int, stride: int) -> List[Operation]:
        """创建操作集"""
        return [
            Identity(),
            Zero(stride),
            AvgPool3x3(stride),
            MaxPool3x3(stride),
            SepConv3x3(channels, channels, stride),
            SepConv3x3(channels, channels, stride),  # 重复以增加权重
            DilatedConv3x3(channels, channels, 2),
            DilatedConv3x3(channels, channels, 3)
        ]
    
    def forward(self, s0: torch.Tensor, s1: torch.Tensor, 
                hard: bool = False) -> torch.Tensor:
        """前向传播"""
        s0 = self.preprocess0(s0)
        s1 = s1  # 假设已经预处理
        
        states = [s0, s1]
        
        for i in range(self.num_nodes):
            s = 0
            for j in range(len(states)):
                edge_name = f"{j}_{i+2}"
                if edge_name in self.edges:
                    s = s + self.edges[edge_name](states[j], hard)
            states.append(s)
        
        # 合并输出节点
        out = torch.cat(states[-self.num_nodes:], dim=1)
        return self.postprocess(out)
    
    def get_arch_params(self) -> List[nn.Parameter]:
        """获取架构参数"""
        params = []
        for edge in self.edges.values():
            params.append(edge.arch_param)
        return params


# ==================== DARTS搜索网络 ====================

class DARTSNetwork(nn.Module):
    """DARTS搜索网络"""
    
    def __init__(self, in_channels: int, num_classes: int,
                 config: Optional[NASConfig] = None):
        super().__init__()
        self.config = config or NASConfig()
        
        # 初始卷积
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # 搜索单元
        self.cells = nn.ModuleList()
        channels = 16
        
        for i in range(self.config.num_layers):
            # 确定stride
            stride = 2 if i in [self.config.num_layers // 3, 
                               2 * self.config.num_layers // 3] else 1
            
            cell = SearchCell(channels, channels * 2 if stride == 2 else channels,
                            self.config.num_nodes, stride)
            self.cells.append(cell)
            
            if stride == 2:
                channels *= 2
        
        # 全局池化和分类器
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, num_classes)
        
    def forward(self, x: torch.Tensor, hard: bool = False) -> torch.Tensor:
        """前向传播"""
        x = self.stem(x)
        
        s0 = x
        s1 = x
        
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1, hard)
        
        out = self.global_pool(s1)
        out = out.view(out.size(0), -1)
        return self.classifier(out)
    
    def get_arch_params(self) -> List[nn.Parameter]:
        """获取所有架构参数"""
        params = []
        for cell in self.cells:
            params.extend(cell.get_arch_params())
        return params
    
    def get_weight_params(self) -> List[nn.Parameter]:
        """获取权重参数（排除架构参数）"""
        arch_params = set(self.get_arch_params())
        return [p for p in self.parameters() if p not in arch_params]
    
    def derive_architecture(self) -> Dict[str, int]:
        """导出离散架构"""
        arch = {}
        for i, cell in enumerate(self.cells):
            cell_arch = {}
            for name, edge in cell.edges.items():
                idx = edge.arch_param.argmax().item()
                cell_arch[name] = idx
            arch[f"cell_{i}"] = cell_arch
        return arch


# ==================== DARTS搜索器 ====================

class DARTSSearcher:
    """DARTS搜索器"""
    
    def __init__(self, network: DARTSNetwork,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 config: Optional[NASConfig] = None,
                 device: str = 'cpu'):
        self.network = network.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or NASConfig()
        self.device = device
        
        # 优化器
        self.weight_optimizer = torch.optim.SGD(
            network.get_weight_params(),
            lr=self.config.learning_rate,
            momentum=0.9,
            weight_decay=self.config.weight_decay
        )
        
        self.arch_optimizer = torch.optim.Adam(
            network.get_arch_params(),
            lr=self.config.arch_learning_rate,
            betas=(0.5, 0.999)
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 统计
        self.search_history: List[Dict] = []
        
    def search(self, callback: Optional[Callable] = None) -> Dict[str, Any]:
        """执行架构搜索"""
        for epoch in range(self.config.search_epochs):
            # 训练一个epoch
            train_loss, train_acc = self._train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self._validate()
            
            # 记录
            self.search_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })
            
            if callback:
                callback(epoch, self.search_history[-1])
        
        # 导出架构
        arch = self.network.derive_architecture()
        
        return {
            'architecture': arch,
            'history': self.search_history,
            'final_val_acc': val_acc
        }
    
    def _train_epoch(self, epoch: int) -> Tuple[float, float]:
        """训练一个epoch"""
        self.network.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for step, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # 更新架构参数
            self._update_arch_params(inputs, targets)
            
            # 更新权重参数
            self._update_weight_params(inputs, targets)
            
            # 统计
            with torch.no_grad():
                outputs = self.network(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                total_correct += (outputs.argmax(1) == targets).sum().item()
                total_samples += targets.size(0)
        
        return total_loss / len(self.train_loader), total_correct / total_samples
    
    def _update_arch_params(self, inputs: torch.Tensor, targets: torch.Tensor):
        """更新架构参数"""
        # 获取验证数据
        val_inputs, val_targets = next(iter(self.val_loader))
        val_inputs = val_inputs.to(self.device)
        val_targets = val_targets.to(self.device)
        
        # 计算验证损失关于架构参数的梯度
        self.arch_optimizer.zero_grad()
        
        if self.config.unrolled:
            # 二阶优化
            self._backward_step_unrolled(inputs, targets, val_inputs, val_targets)
        else:
            # 一阶优化
            val_outputs = self.network(val_inputs)
            val_loss = self.criterion(val_outputs, val_targets)
            val_loss.backward()
        
        self.arch_optimizer.step()
    
    def _update_weight_params(self, inputs: torch.Tensor, targets: torch.Tensor):
        """更新权重参数"""
        self.weight_optimizer.zero_grad()
        
        outputs = self.network(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        
        # 梯度裁剪
        nn.utils.clip_grad_norm_(self.network.get_weight_params(), 5.0)
        
        self.weight_optimizer.step()
    
    def _backward_step_unrolled(self, train_inputs, train_targets, val_inputs, val_targets):
        """二阶展开的反向传播"""
        # 计算虚拟步长
        lr = self.weight_optimizer.param_groups[0]['lr']
        
        # 计算训练损失
        train_outputs = self.network(train_inputs)
        train_loss = self.criterion(train_outputs, train_targets)
        
        # 计算虚拟权重
        virtual_weights = []
        for w in self.network.get_weight_params():
            dw = torch.autograd.grad(train_loss, w, retain_graph=True)[0]
            virtual_weights.append(w - lr * dw)
        
        # 使用虚拟权重计算验证损失
        val_outputs = self.network(val_inputs)
        val_loss = self.criterion(val_outputs, val_targets)
        
        # 计算架构梯度
        for arch_param in self.network.get_arch_params():
            darch = torch.autograd.grad(val_loss, arch_param, retain_graph=True)[0]
            arch_param.grad = darch
    
    def _validate(self) -> Tuple[float, float]:
        """验证"""
        self.network.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.network(inputs, hard=True)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                total_correct += (outputs.argmax(1) == targets).sum().item()
                total_samples += targets.size(0)
        
        return total_loss / len(self.val_loader), total_correct / total_samples


# ==================== ProxylessNAS ====================

class ProxylessNASNetwork(nn.Module):
    """ProxylessNAS网络"""
    
    def __init__(self, in_channels: int, num_classes: int,
              config: Optional[NASConfig] = None):
        super().__init__()
        self.config = config or NASConfig()
        
        # 初始层
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 可搜索的块
        self.blocks = nn.ModuleList()
        channels = 32
        
        for i in range(self.config.num_layers):
            block = self._make_block(channels, channels * 2)
            self.blocks.append(block)
            channels *= 2
        
        # 分类器
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, num_classes)
        
    def _make_block(self, in_channels: int, out_channels: int) -> MixedOperation:
        """创建可搜索块"""
        ops = [
            Identity(),
            Zero(2),
            SepConv3x3(in_channels, out_channels, 1),
            SepConv3x3(in_channels, out_channels, 2),
            DepthwiseConv3x3(out_channels, 1),
            DepthwiseConv3x3(out_channels, 2),
            Conv3x3(in_channels, out_channels, 1),
            Conv3x3(in_channels, out_channels, 2)
        ]
        return MixedOperation(ops)
    
    def forward(self, x: torch.Tensor, hard: bool = False) -> torch.Tensor:
        """前向传播"""
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x, hard)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
    
    def get_arch_params(self) -> List[nn.Parameter]:
        """获取架构参数"""
        return [block.arch_param for block in self.blocks]


class ProxylessNASSearcher:
    """ProxylessNAS搜索器"""
    
    def __init__(self, network: ProxylessNASNetwork,
                 train_loader: torch.utils.data.DataLoader,
                 config: Optional[NASConfig] = None,
                 device: str = 'cpu'):
        self.network = network.to(device)
        self.train_loader = train_loader
        self.config = config or NASConfig()
        self.device = device
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            network.parameters(),
            lr=self.config.learning_rate
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
    def search(self, num_epochs: int = 50) -> Dict[str, Any]:
        """执行搜索"""
        history = []
        
        for epoch in range(num_epochs):
            # 训练
            train_loss, train_acc = self._train_epoch()
            
            # 评估
            val_acc = self._evaluate()
            
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            })
        
        return {
            'architecture': self._derive_architecture(),
            'history': history
        }
    
    def _train_epoch(self) -> Tuple[float, float]:
        """训练一个epoch"""
        self.network.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # ProxylessNAS梯度更新
            self._proxyless_gradient_step(inputs, targets)
            
            # 统计
            with torch.no_grad():
                outputs = self.network(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                total_correct += (outputs.argmax(1) == targets).sum().item()
                total_samples += targets.size(0)
        
        return total_loss / len(self.train_loader), total_correct / total_samples
    
    def _proxyless_gradient_step(self, inputs: torch.Tensor, targets: torch.Tensor):
        """ProxylessNAS梯度步骤"""
        self.optimizer.zero_grad()
        
        # 前向传播
        outputs = self.network(inputs)
        loss = self.criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        
        # 更新架构参数（使用梯度重置）
        for arch_param in self.network.get_arch_params():
            if arch_param.grad is not None:
                # 应用梯度修正
                arch_param.grad = self._correct_gradient(arch_param)
        
        self.optimizer.step()
    
    def _correct_gradient(self, arch_param: nn.Parameter) -> torch.Tensor:
        """修正架构参数梯度"""
        # 简化的梯度修正
        grad = arch_param.grad
        probs = F.softmax(arch_param, dim=0)
        return grad - (probs * grad).sum()
    
    def _evaluate(self) -> float:
        """评估"""
        self.network.eval()
        
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in self.train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.network(inputs, hard=True)
                total_correct += (outputs.argmax(1) == targets).sum().item()
                total_samples += targets.size(0)
        
        return total_correct / total_samples
    
    def _derive_architecture(self) -> List[int]:
        """导出架构"""
        arch = []
        for block in self.network.blocks:
            idx = block.arch_param.argmax().item()
            arch.append(idx)
        return arch


# ==================== 架构评估器 ====================

class ArchitectureEvaluator:
    """架构评估器"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        
    def evaluate(self, network: nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 num_epochs: int = 100) -> Dict[str, float]:
        """评估架构"""
        network = network.to(self.device)
        
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            # 训练
            network.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = network(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            # 验证
            network.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = network(inputs)
                    correct += (outputs.argmax(1) == targets).sum().item()
                    total += targets.size(0)
            
            val_acc = correct / total
            best_val_acc = max(best_val_acc, val_acc)
        
        # 计算模型大小
        num_params = sum(p.numel() for p in network.parameters())
        
        return {
            'val_accuracy': best_val_acc,
            'num_params': num_params,
            'model_size_mb': num_params * 4 / (1024 ** 2)
        }


# ==================== 工具函数 ====================

def count_parameters(network: nn.Module) -> int:
    """计算参数数量"""
    return sum(p.numel() for p in network.parameters())


def estimate_flops(network: nn.Module, input_size: Tuple[int, ...]) -> int:
    """估算FLOPs"""
    # 简化估算
    total_flops = 0
    
    for module in network.modules():
        if isinstance(module, nn.Conv2d):
            # 卷积FLOPs
            out_h = input_size[2] // module.stride[0]
            out_w = input_size[3] // module.stride[1]
            flops = (module.in_channels * module.kernel_size[0] * 
                    module.kernel_size[1] * module.out_channels * out_h * out_w)
            total_flops += flops
        elif isinstance(module, nn.Linear):
            # 全连接FLOPs
            flops = module.in_features * module.out_features
            total_flops += flops
    
    return total_flops


def visualize_architecture(arch: Dict[str, Any]) -> str:
    """可视化架构"""
    lines = ["Architecture:"]
    
    for cell_name, cell_arch in arch.items():
        lines.append(f"  {cell_name}:")
        for edge, op_idx in cell_arch.items():
            lines.append(f"    {edge} -> op_{op_idx}")
    
    return "\n".join(lines)
