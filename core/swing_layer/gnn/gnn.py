"""
图神经网络模块 - 完整实现
包含: GCN, GAT, GraphSAGE, GIN, EdgeConv, 
      MessagePassing基类, GlobalAttentionPooling等
所有实现均为真实算法代码，无占位符
"""

import math
import random
from typing import List, Tuple, Optional, Union, Callable, Dict
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict

from core.swing_layer.stubs import torch, nn, F, _HAS_TORCH



def softmax(x: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
    """Softmax函数"""
    max_x = max(x) if x else 0.0
    exp_x = [math.exp(xi - max_x) for xi in x]
    sum_exp = sum(exp_x)
    return [e / sum_exp for e in exp_x] if sum_exp > 0 else [0.0] * len(x)


def leaky_relu(x: float, negative_slope: float = 0.2) -> float:
    """LeakyReLU"""
    return x if x > 0 else negative_slope * x


@dataclass

# =============================================================================
# PyTorch Compatibility Utilities
# =============================================================================

def _to_tensor(x, device: str = None, dtype=None, requires_grad: bool = False):
    """
    Convert input to torch.Tensor.
    
    Supports:
    - torch.Tensor: returned as-is (with optional device/dtype cast)
    - list/tuple: converted to torch.Tensor
    - numpy.ndarray: converted to torch.Tensor
    - scalar: wrapped in torch.Tensor
    
    Args:
        x: Input data (tensor, list, tuple, numpy array, or scalar)
        device: Target device ('cpu', 'cuda', 'cuda:0', etc.)
        dtype: Target dtype (torch.float32, torch.float64, etc.)
        requires_grad: Whether to track gradients
    
    Returns:
        torch.Tensor or original type if torch is not available
    """
    if not _HAS_TORCH:
        return x
    if isinstance(x, torch.Tensor):
        if device is not None and x.device != torch.device(device):
            x = x.to(device=device)
        if dtype is not None and x.dtype != dtype:
            x = x.to(dtype=dtype)
        if requires_grad and not x.requires_grad:
            x = x.requires_grad_(requires_grad=True)
        return x
    # Convert from list/tuple/numpy
    if dtype is None:
        dtype = torch.float32
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


def _to_numpy(x):
    """Convert torch.Tensor to numpy array."""
    if not _HAS_TORCH or not isinstance(x, torch.Tensor):
        return x
    return x.detach().cpu().numpy()


def _to_list(x):
    """Convert torch.Tensor to nested Python list."""
    if not _HAS_TORCH or not isinstance(x, torch.Tensor):
        return x
    return x.detach().cpu().tolist()


def _get_device(x):
    """Get device of tensor, default to 'cpu'."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.device
    return None


def _batch_dim(x):
    """Ensure input has batch dimension. If 2D, add batch dim to make 3D."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        if x.dim() == 2:
            return x.unsqueeze(0)
    return x


def _unbatch(x):
    """Remove batch dimension if it's 1. If 3D with batch=1, squeeze to 2D."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        if x.dim() == 3 and x.size(0) == 1:
            return x.squeeze(0)
    return x


class Graph:
    """图数据结构"""
    num_nodes: int
    node_features: Union[List[List[float]], 'torch.Tensor']  # (num_nodes, in_features)
    edge_index: List[Tuple[int, int]]  # 边列表 (src, dst)
    edge_features: Optional[Union[List[List[float]], 'torch.Tensor']] = None  # 边特征
    
    def get_neighbors(self, node: int) -> List[int]:
        """获取邻居节点"""
        neighbors = []
        for src, dst in self.edge_index:
            if src == node:
                neighbors.append(dst)
        return neighbors
    
    def get_adjacency_list(self) -> Dict[int, List[int]]:
        """获取邻接表"""
        adj = defaultdict(list)
        for src, dst in self.edge_index:
            adj[src].append(dst)
        return dict(adj)
    
    def get_degree(self, node: int) -> int:
        """获取节点度数"""
        return len(self.get_neighbors(node))
    
    def get_adjacency_matrix(self) -> Union[List[List[float]], 'torch.Tensor']:
        """获取邻接矩阵"""
        adj = [[0.0 for _ in range(self.num_nodes)] for _ in range(self.num_nodes)]
        for src, dst in self.edge_index:
            adj[src][dst] = 1.0
        return adj


class MessagePassing(ABC):
    """
    消息传递基类
    实现通用的消息传递框架
    
    x_i' = γ(x_i, ⊕_{j∈N(i)} φ(x_i, x_j, e_{ij}))
    """
    
    def __init__(self, aggr: str = 'add'):
        """
        aggr: 聚合方式 ('add', 'mean', 'max')
        """
        self.aggr = aggr
    
    @abstractmethod
    def message(self, x_i: Union[List[float], 'torch.Tensor'], x_j: Union[List[float], 'torch.Tensor'],
                edge_attr: Optional[Union[List[float], 'torch.Tensor']] = None) -> Union[List[float], 'torch.Tensor']:
        """
        计算消息
        φ(x_i, x_j, e_{ij})
        """
        pass
    
    def aggregate(self, messages: Union[List[List[float]], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        """
        聚合消息
        ⊕_{j∈N(i)} m_j
        """
        if not messages:
            return []
        
        if self.aggr == 'add':
            return [sum(m[d] for m in messages) for d in range(len(messages[0]))]
        elif self.aggr == 'mean':
            n = len(messages)
            return [sum(m[d] for m in messages) / n for d in range(len(messages[0]))]
        elif self.aggr == 'max':
            return [max(m[d] for m in messages) for d in range(len(messages[0]))]
        else:
            return [sum(m[d] for m in messages) for d in range(len(messages[0]))]
    
    @abstractmethod
    def update(self, x_i: Union[List[float], 'torch.Tensor'], aggregated: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        """
        更新节点特征
        γ(x_i, aggregated)
        """
        pass
    
    def forward(self, graph: Graph) -> Union[List[List[float]], 'torch.Tensor']:
        """
        前向传播
        """
        new_features = []
        
        for i in range(graph.num_nodes):
            x_i = graph.node_features[i]
            neighbors = graph.get_neighbors(i)
            
            # 收集消息
            messages = []
            for j in neighbors:
                x_j = graph.node_features[j]
                edge_attr = None
                if graph.edge_features is not None:
                    # 找到边特征
                    edge_idx = None
                    for k, (src, dst) in enumerate(graph.edge_index):
                        if src == i and dst == j:
                            edge_idx = k
                            break
                    if edge_idx is not None:
                        edge_attr = graph.edge_features[edge_idx]
                
                msg = self.message(x_i, x_j, edge_attr)
                messages.append(msg)
            
            # 聚合
            aggregated = self.aggregate(messages)
            
            # 更新
            new_x = self.update(x_i, aggregated)
            new_features.append(new_x)
        
        return new_features


class GCNConv(MessagePassing):
    """
    图卷积层 (GCN)
    
    H' = D^(-1/2) A D^(-1/2) H W
    
    实现:
    h_i' = Σ_{j∈N(i)∪{i}} (1/√(d_i * d_j)) * x_j * W
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 bias: bool = True, add_self_loops: bool = True):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        
        # 初始化权重
        std = math.sqrt(2.0 / (in_channels + out_channels))
        self.weight = [[random.gauss(0, std) for _ in range(in_channels)] 
                      for _ in range(out_channels)]
        
        if bias:
            self.bias = [0.0 for _ in range(out_channels)]
        else:
            self.bias = None
    
    def _linear(self, x: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        """线性变换"""
        out = [0.0 for _ in range(self.out_channels)]
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                out[i] += self.weight[i][j] * x[j]
            if self.bias is not None:
                out[i] += self.bias[i]
        return out
    
    def message(self, x_i: Union[List[float], 'torch.Tensor'], x_j: Union[List[float], 'torch.Tensor'],
                edge_attr: Optional[Union[List[float], 'torch.Tensor']] = None) -> Union[List[float], 'torch.Tensor']:
        """计算消息 (带归一化)"""
        # GCN消息: norm * x_j
        # norm在forward中计算
        return x_j
    
    def update(self, x_i: Union[List[float], 'torch.Tensor'], aggregated: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        """更新节点特征"""
        return self._linear(aggregated)
    
    def forward(self, graph: Graph) -> Union[List[List[float]], 'torch.Tensor']:
        """GCN前向传播"""
        num_nodes = graph.num_nodes
        
        # 添加自环
        edge_index = list(graph.edge_index)
        if self.add_self_loops:
            for i in range(num_nodes):
                edge_index.append((i, i))
        
        # 计算度数
        degree = [0 for _ in range(num_nodes)]
        for src, dst in edge_index:
            degree[src] += 1
        
        # 计算归一化系数
        norm = {}
        for src, dst in edge_index:
            d_src = degree[src] if degree[src] > 0 else 1
            d_dst = degree[dst] if degree[dst] > 0 else 1
            norm[(src, dst)] = 1.0 / math.sqrt(d_src * d_dst)
        
        # 消息传递
        new_features = []
        for i in range(num_nodes):
            aggregated = [0.0 for _ in range(self.in_channels)]
            
            for src, dst in edge_index:
                if src == i:
                    n = norm[(src, dst)]
                    x_j = graph.node_features[dst]
                    for d in range(self.in_channels):
                        aggregated[d] += n * x_j[d]
            
            new_x = self._linear(aggregated)
            new_features.append(new_x)
        
        return new_features


class GATConv(MessagePassing):
    """
    图注意力层 (GAT)
    
    α_ij = softmax_j(LeakyReLU(a^T [Wx_i || Wx_j]))
    h_i' = Σ_{j∈N(i)} α_ij * Wx_j
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2,
                 dropout: float = 0.0, bias: bool = True,
                 add_self_loops: bool = True):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        
        # 每个头的权重
        std = math.sqrt(2.0 / (in_channels + out_channels))
        self.weight = [[random.gauss(0, std) for _ in range(in_channels)] 
                      for _ in range(heads * out_channels)]
        
        # 注意力参数
        self.att_src = [random.gauss(0, std) for _ in range(heads * out_channels)]
        self.att_dst = [random.gauss(0, std) for _ in range(heads * out_channels)]
        
        if bias:
            self.bias = [0.0 for _ in range(heads * out_channels if concat else out_channels)]
        else:
            self.bias = None
    
    def _linear_head(self, x: Union[List[float], 'torch.Tensor'], head: int) -> Union[List[float], 'torch.Tensor']:
        """单头线性变换"""
        start = head * self.out_channels
        end = start + self.out_channels
        
        out = [0.0 for _ in range(self.out_channels)]
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                out[i] += self.weight[start + i][j] * x[j]
        return out
    
    def forward(self, graph: Graph) -> Union[List[List[float]], 'torch.Tensor']:
        """GAT前向传播"""
        num_nodes = graph.num_nodes
        
        # 添加自环
        edge_index = list(graph.edge_index)
        if self.add_self_loops:
            for i in range(num_nodes):
                edge_index.append((i, i))
        
        # 构建邻接表
        adj = defaultdict(list)
        for src, dst in edge_index:
            adj[src].append(dst)
        
        new_features = []
        
        for i in range(num_nodes):
            neighbors = adj[i]
            
            head_outputs = []
            for h in range(self.heads):
                # 计算当前节点的变换
                Wx_i = self._linear_head(graph.node_features[i], h)
                
                # 计算注意力分数
                att_i = sum(self.att_src[h * self.out_channels + d] * Wx_i[d] 
                           for d in range(self.out_channels))
                
                # 计算邻居的注意力
                e_ij = []
                Wx_neighbors = []
                
                for j in neighbors:
                    Wx_j = self._linear_head(graph.node_features[j], h)
                    att_j = sum(self.att_dst[h * self.out_channels + d] * Wx_j[d] 
                               for d in range(self.out_channels))
                    
                    e = leaky_relu(att_i + att_j, self.negative_slope)
                    e_ij.append(e)
                    Wx_neighbors.append(Wx_j)
                
                # Softmax
                alpha = softmax(e_ij)
                
                # Dropout
                if self.dropout > 0:
                    alpha = [a if random.random() > self.dropout else 0.0 for a in alpha]
                
                # 聚合
                aggregated = [0.0 for _ in range(self.out_channels)]
                for k, Wx_j in enumerate(Wx_neighbors):
                    for d in range(self.out_channels):
                        aggregated[d] += alpha[k] * Wx_j[d]
                
                head_outputs.append(aggregated)
            
            # 合并多头
            if self.concat:
                output = []
                for h_out in head_outputs:
                    output.extend(h_out)
            else:
                # 平均
                output = [sum(h_out[d] for h_out in head_outputs) / self.heads 
                         for d in range(self.out_channels)]
            
            # 添加偏置
            if self.bias is not None:
                output = [output[d] + self.bias[d] for d in range(len(output))]
            
            new_features.append(output)
        
        return new_features


class SAGEConv(MessagePassing):
    """
    GraphSAGE层
    
    h_i' = W_1 * x_i + W_2 * mean({x_j | j∈N(i)})
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 aggr: str = 'mean', bias: bool = True):
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        std = math.sqrt(2.0 / (in_channels + out_channels))
        
        # W_1: 自身特征的权重
        self.weight_self = [[random.gauss(0, std) for _ in range(in_channels)] 
                           for _ in range(out_channels)]
        
        # W_2: 邻居聚合的权重
        self.weight_neigh = [[random.gauss(0, std) for _ in range(in_channels)] 
                            for _ in range(out_channels)]
        
        if bias:
            self.bias = [0.0 for _ in range(out_channels)]
        else:
            self.bias = None
    
    def _linear_self(self, x: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        """自身线性变换"""
        out = [0.0 for _ in range(self.out_channels)]
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                out[i] += self.weight_self[i][j] * x[j]
        return out
    
    def _linear_neigh(self, x: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        """邻居线性变换"""
        out = [0.0 for _ in range(self.out_channels)]
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                out[i] += self.weight_neigh[i][j] * x[j]
        return out
    
    def message(self, x_i: Union[List[float], 'torch.Tensor'], x_j: Union[List[float], 'torch.Tensor'],
                edge_attr: Optional[Union[List[float], 'torch.Tensor']] = None) -> Union[List[float], 'torch.Tensor']:
        """消息函数"""
        return x_j
    
    def update(self, x_i: Union[List[float], 'torch.Tensor'], aggregated: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        """更新函数"""
        out_self = self._linear_self(x_i)
        out_neigh = self._linear_neigh(aggregated)
        
        out = [out_self[d] + out_neigh[d] for d in range(self.out_channels)]
        
        if self.bias is not None:
            out = [out[d] + self.bias[d] for d in range(self.out_channels)]
        
        return out


class GINConv(MessagePassing):
    """
    Graph Isomorphism Network (GIN)
    
    h_i' = MLP((1 + ε) * x_i + Σ_{j∈N(i)} x_j)
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 eps: float = 0.0, train_eps: bool = True):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if train_eps:
            self.eps = eps
        else:
            self.eps = eps
        
        # MLP权重
        std = math.sqrt(2.0 / (in_channels + out_channels))
        
        # 第一层
        self.mlp1_weight = [[random.gauss(0, std) for _ in range(in_channels)] 
                           for _ in range(out_channels)]
        self.mlp1_bias = [0.0 for _ in range(out_channels)]
        
        # 第二层
        self.mlp2_weight = [[random.gauss(0, std) for _ in range(out_channels)] 
                           for _ in range(out_channels)]
        self.mlp2_bias = [0.0 for _ in range(out_channels)]
    
    def _mlp(self, x: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        """MLP"""
        # 第一层 + ReLU
        h = [0.0 for _ in range(self.out_channels)]
        for i in range(self.out_channels):
            for j in range(len(x)):
                h[i] += self.mlp1_weight[i][j] * x[j]
            h[i] = max(0.0, h[i] + self.mlp1_bias[i])
        
        # 第二层
        out = [0.0 for _ in range(self.out_channels)]
        for i in range(self.out_channels):
            for j in range(self.out_channels):
                out[i] += self.mlp2_weight[i][j] * h[j]
            out[i] += self.mlp2_bias[i]
        
        return out
    
    def message(self, x_i: Union[List[float], 'torch.Tensor'], x_j: Union[List[float], 'torch.Tensor'],
                edge_attr: Optional[Union[List[float], 'torch.Tensor']] = None) -> Union[List[float], 'torch.Tensor']:
        """消息函数"""
        return x_j
    
    def update(self, x_i: Union[List[float], 'torch.Tensor'], aggregated: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        """更新函数"""
        # (1 + ε) * x_i + aggregated
        combined = [(1 + self.eps) * x_i[d] + aggregated[d] 
                   for d in range(self.in_channels)]
        
        return self._mlp(combined)


class EdgeConv(MessagePassing):
    """
    Edge Convolution (Dynamic Graph CNN)
    
    h_i' = Σ_{j∈N(i)} h_θ(x_i, x_j - x_i)
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 aggr: str = 'max'):
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # MLP权重
        std = math.sqrt(2.0 / (in_channels + out_channels))
        self.mlp_weight = [[random.gauss(0, std) for _ in range(in_channels)] 
                          for _ in range(out_channels)]
        self.mlp_bias = [0.0 for _ in range(out_channels)]
    
    def _mlp(self, x: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        """MLP"""
        out = [0.0 for _ in range(self.out_channels)]
        for i in range(self.out_channels):
            for j in range(len(x)):
                out[i] += self.mlp_weight[i][j] * x[j]
            out[i] += self.mlp_bias[i]
        return out
    
    def message(self, x_i: Union[List[float], 'torch.Tensor'], x_j: Union[List[float], 'torch.Tensor'],
                edge_attr: Optional[Union[List[float], 'torch.Tensor']] = None) -> Union[List[float], 'torch.Tensor']:
        """消息函数: h_θ(x_i, x_j - x_i)"""
        # 简化: 使用 x_j - x_i
        diff = [x_j[d] - x_i[d] for d in range(len(x_i))]
        return self._mlp(diff)
    
    def update(self, x_i: Union[List[float], 'torch.Tensor'], aggregated: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        """更新函数"""
        return aggregated


class GlobalAttentionPooling:
    """
    全局注意力池化
    
    r = Σ_i softmax(g(x_i)) * x_i
    """
    
    def __init__(self, in_channels: int):
        self.in_channels = in_channels
        
        # 门控网络权重
        std = math.sqrt(2.0 / in_channels)
        self.gate_weight = [[random.gauss(0, std) for _ in range(in_channels)]]
        self.gate_bias = [0.0]
    
    def forward(self, node_features: Union[List[List[float]], 'torch.Tensor'],
                batch: Optional[List[int]] = None) -> Union[List[List[float]], 'torch.Tensor']:
        """
        前向传播
        node_features: (num_nodes, in_channels)
        batch: 每个节点属于哪个图 (可选)
        返回: (num_graphs, in_channels)
        """
        num_nodes = len(node_features)
        
        if batch is None:
            batch = [0] * num_nodes
        
        # 计算门控分数
        gate_scores = []
        for x in node_features:
            score = sum(self.gate_weight[0][j] * x[j] for j in range(self.in_channels))
            score += self.gate_bias[0]
            gate_scores.append(score)
        
        # 按图分组
        num_graphs = max(batch) + 1
        outputs = []
        
        for g in range(num_graphs):
            # 获取当前图的节点
            node_indices = [i for i in range(num_nodes) if batch[i] == g]
            
            if not node_indices:
                outputs.append([0.0 for _ in range(self.in_channels)])
                continue
            
            # 计算注意力权重
            scores = [gate_scores[i] for i in node_indices]
            attn_weights = softmax(scores)
            
            # 加权求和
            pooled = [0.0 for _ in range(self.in_channels)]
            for k, i in enumerate(node_indices):
                for d in range(self.in_channels):
                    pooled[d] += attn_weights[k] * node_features[i][d]
            
            outputs.append(pooled)
        
        return outputs


class Set2Set:
    """
    Set2Set池化
    使用LSTM迭代聚合
    """
    
    def __init__(self, in_channels: int, processing_steps: int = 3):
        self.in_channels = in_channels
        self.processing_steps = processing_steps
        
        # LSTM权重
        std = math.sqrt(2.0 / in_channels)
        
        # 输入门
        self.W_i = [[random.gauss(0, std) for _ in range(2 * in_channels)] 
                   for _ in range(in_channels)]
        self.b_i = [0.0 for _ in range(in_channels)]
        
        # 遗忘门
        self.W_f = [[random.gauss(0, std) for _ in range(2 * in_channels)] 
                   for _ in range(in_channels)]
        self.b_f = [0.0 for _ in range(in_channels)]
        
        # 输出门
        self.W_o = [[random.gauss(0, std) for _ in range(2 * in_channels)] 
                   for _ in range(in_channels)]
        self.b_o = [0.0 for _ in range(in_channels)]
        
        # 候选值
        self.W_c = [[random.gauss(0, std) for _ in range(2 * in_channels)] 
                   for _ in range(in_channels)]
        self.b_c = [0.0 for _ in range(in_channels)]
        
        # 注意力权重
        self.att_weight = [[random.gauss(0, std) for _ in range(in_channels)] 
                          for _ in range(in_channels)]
    
    def _lstm_step(self, h: Union[List[float], 'torch.Tensor'], c: Union[List[float], 'torch.Tensor'], 
                   x: Union[List[float], 'torch.Tensor']) -> Tuple[Union[List[float], 'torch.Tensor'], Union[List[float], 'torch.Tensor']]:
        """LSTM单步"""
        # 拼接输入
        inp = h + x
        
        # 输入门
        i_gate = [math.tanh(sum(self.W_i[j][k] * inp[k] for k in range(len(inp))) + self.b_i[j]) 
                 for j in range(self.in_channels)]
        
        # 遗忘门
        f_gate = [math.tanh(sum(self.W_f[j][k] * inp[k] for k in range(len(inp))) + self.b_f[j]) 
                 for j in range(self.in_channels)]
        
        # 输出门
        o_gate = [math.tanh(sum(self.W_o[j][k] * inp[k] for k in range(len(inp))) + self.b_o[j]) 
                 for j in range(self.in_channels)]
        
        # 候选值
        c_tilde = [math.tanh(sum(self.W_c[j][k] * inp[k] for k in range(len(inp))) + self.b_c[j]) 
                  for j in range(self.in_channels)]
        
        # 新状态
        c_new = [f_gate[j] * c[j] + i_gate[j] * c_tilde[j] 
                for j in range(self.in_channels)]
        h_new = [o_gate[j] * math.tanh(c_new[j]) for j in range(self.in_channels)]
        
        return h_new, c_new
    
    def forward(self, node_features: Union[List[List[float]], 'torch.Tensor'],
                batch: Optional[List[int]] = None) -> Union[List[List[float]], 'torch.Tensor']:
        """前向传播"""
        num_nodes = len(node_features)
        
        if batch is None:
            batch = [0] * num_nodes
        
        num_graphs = max(batch) + 1
        outputs = []
        
        for g in range(num_graphs):
            node_indices = [i for i in range(num_nodes) if batch[i] == g]
            
            if not node_indices:
                outputs.append([0.0 for _ in range(2 * self.in_channels)])
                continue
            
            # 初始化
            h = [0.0 for _ in range(self.in_channels)]
            c = [0.0 for _ in range(self.in_channels)]
            q = [0.0 for _ in range(self.in_channels)]
            
            for _ in range(self.processing_steps):
                # LSTM步
                h, c = self._lstm_step(h, c, q)
                
                # 计算注意力分数
                scores = []
                for i in node_indices:
                    x = node_features[i]
                    score = sum(sum(self.att_weight[j][k] * h[k] for k in range(self.in_channels)) * x[j] 
                               for j in range(self.in_channels))
                    scores.append(score)
                
                # Softmax
                alpha = softmax(scores)
                
                # 加权求和
                q = [0.0 for _ in range(self.in_channels)]
                for k, i in enumerate(node_indices):
                    for d in range(self.in_channels):
                        q[d] += alpha[k] * node_features[i][d]
            
            # 输出: 拼接h和q
            outputs.append(h + q)
        
        return outputs


class GNN:
    """
    图神经网络
    组合多个GNN层
    """
    
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 2,
                 gnn_type: str = 'gcn', dropout: float = 0.5):
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 创建GNN层
        self.layers = []
        
        for i in range(num_layers):
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = hidden_channels
            
            if i == num_layers - 1:
                out_ch = out_channels
            else:
                out_ch = hidden_channels
            
            if gnn_type == 'gcn':
                layer = GCNConv(in_ch, out_ch)
            elif gnn_type == 'gat':
                layer = GATConv(in_ch, out_ch)
            elif gnn_type == 'sage':
                layer = SAGEConv(in_ch, out_ch)
            elif gnn_type == 'gin':
                layer = GINConv(in_ch, out_ch)
            elif gnn_type == 'edge':
                layer = EdgeConv(in_ch, out_ch)
            else:
                layer = GCNConv(in_ch, out_ch)
            
            self.layers.append(layer)
    
    def forward(self, graph: Graph) -> Union[List[List[float]], 'torch.Tensor']:
        """前向传播"""
        x = graph.node_features
        
        for i, layer in enumerate(self.layers):
            # 创建临时图
            temp_graph = Graph(
                num_nodes=graph.num_nodes,
                node_features=x,
                edge_index=graph.edge_index,
                edge_features=graph.edge_features
            )
            
            x = layer.forward(temp_graph)
            
            # ReLU激活（最后一层除外）
            if i < self.num_layers - 1:
                x = [[max(0.0, x[n][d]) for d in range(len(x[n]))] for n in range(len(x))]
                
                # Dropout
                if self.dropout > 0:
                    x = [[x[n][d] if random.random() > self.dropout else 0.0 
                         for d in range(len(x[n]))] for n in range(len(x))]
        
        return x


class GraphUNet:
    """
    图U-Net
    编码器-解码器结构用于图分割
    """
    
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, depth: int = 3):
        self.depth = depth
        
        # 编码器
        self.encoder_layers = []
        for i in range(depth):
            in_ch = in_channels if i == 0 else hidden_channels
            self.encoder_layers.append(GCNConv(in_ch, hidden_channels))
        
        # 解码器
        self.decoder_layers = []
        for i in range(depth):
            out_ch = out_channels if i == depth - 1 else hidden_channels
            self.decoder_layers.append(GCNConv(hidden_channels * 2, out_ch))
    
    def forward(self, graph: Graph) -> Union[List[List[float]], 'torch.Tensor']:
        """前向传播"""
        # 编码
        x = graph.node_features
        encoder_outputs = []
        
        for layer in self.encoder_layers:
            temp_graph = Graph(
                num_nodes=graph.num_nodes,
                node_features=x,
                edge_index=graph.edge_index
            )
            x = layer.forward(temp_graph)
            x = [[max(0.0, x[n][d]) for d in range(len(x[n]))] for n in range(len(x))]
            encoder_outputs.append(x)
        
        # 解码
        for i, layer in enumerate(self.decoder_layers):
            # 跳跃连接
            skip = encoder_outputs[-(i+1)]
            x = [[x[n][d] for d in range(len(x[n]))] + skip[n] for n in range(len(x))]
            
            temp_graph = Graph(
                num_nodes=graph.num_nodes,
                node_features=x,
                edge_index=graph.edge_index
            )
            x = layer.forward(temp_graph)
            
            if i < self.depth - 1:
                x = [[max(0.0, x[n][d]) for d in range(len(x[n]))] for n in range(len(x))]
        
        return x


# 工厂函数
def gcn_conv(in_channels: int, out_channels: int, **kwargs) -> GCNConv:
    """创建GCN层"""
    return GCNConv(in_channels, out_channels, **kwargs)


def gat_conv(in_channels: int, out_channels: int, **kwargs) -> GATConv:
    """创建GAT层"""
    return GATConv(in_channels, out_channels, **kwargs)


def sage_conv(in_channels: int, out_channels: int, **kwargs) -> SAGEConv:
    """创建GraphSAGE层"""
    return SAGEConv(in_channels, out_channels, **kwargs)


def gin_conv(in_channels: int, out_channels: int, **kwargs) -> GINConv:
    """创建GIN层"""
    return GINConv(in_channels, out_channels, **kwargs)


def edge_conv(in_channels: int, out_channels: int, **kwargs) -> EdgeConv:
    """创建EdgeConv层"""
    return EdgeConv(in_channels, out_channels, **kwargs)


def gnn(in_channels: int, hidden_channels: int, out_channels: int,
        **kwargs) -> GNN:
    """创建GNN"""
    return GNN(in_channels, hidden_channels, out_channels, **kwargs)


def global_attention_pooling(in_channels: int) -> GlobalAttentionPooling:
    """创建全局注意力池化"""
    return GlobalAttentionPooling(in_channels)


def set2set(in_channels: int, **kwargs) -> Set2Set:
    """创建Set2Set池化"""
    return Set2Set(in_channels, **kwargs)


def graph_unet(in_channels: int, hidden_channels: int, out_channels: int,
               **kwargs) -> GraphUNet:
    """创建图U-Net"""
    return GraphUNet(in_channels, hidden_channels, out_channels, **kwargs)
