"""
图神经网络模块 - Graph Neural Networks
实现GCN、GAT、GraphSAGE、GIN、消息传递神经网络等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict

# ==================== 图数据结构 ====================

@dataclass
class GraphData:
    """图数据结构"""
    node_features: torch.Tensor  # [N, D]
    edge_index: torch.Tensor  # [2, E]
    edge_features: Optional[torch.Tensor] = None  # [E, D_e]
    node_labels: Optional[torch.Tensor] = None  # [N]
    graph_labels: Optional[torch.Tensor] = None  # [G]
    num_nodes: int = 0
    num_edges: int = 0
    
    def __post_init__(self):
        self.num_nodes = self.node_features.size(0)
        self.num_edges = self.edge_index.size(1)


class BatchedGraphData:
    """批次图数据"""
    
    def __init__(self, graphs: List[GraphData]):
        self.num_graphs = len(graphs)
        
        # 合并节点特征
        node_features_list = [g.node_features for g in graphs]
        self.node_features = torch.cat(node_features_list, dim=0)
        
        # 合并边索引（需要偏移）
        edge_index_list = []
        node_offset = 0
        for g in graphs:
            edge_index = g.edge_index + node_offset
            edge_index_list.append(edge_index)
            node_offset += g.num_nodes
        self.edge_index = torch.cat(edge_index_list, dim=1)
        
        # 合并边特征
        if graphs[0].edge_features is not None:
            edge_features_list = [g.edge_features for g in graphs]
            self.edge_features = torch.cat(edge_features_list, dim=0)
        else:
            self.edge_features = None
        
        # 批次索引
        self.batch = torch.cat([
            torch.full((g.num_nodes,), i, dtype=torch.long)
            for i, g in enumerate(graphs)
        ])
        
        # 节点数量
        self.num_nodes = self.node_features.size(0)
        self.num_edges = self.edge_index.size(1)


# ==================== 图操作 ====================

def compute_degree(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """计算节点度"""
    row = edge_index[0]
    degree = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
    degree.scatter_add_(0, row, torch.ones(row.size(0), device=edge_index.device))
    return degree


def compute_laplacian(
    edge_index: torch.Tensor,
    num_nodes: int,
    normalized: bool = True,
) -> torch.Tensor:
    """计算拉普拉斯矩阵"""
    degree = compute_degree(edge_index, num_nodes)
    
    if normalized:
        degree_inv_sqrt = degree.pow(-0.5)
        degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0
        
        # L = I - D^{-1/2} A D^{-1/2}
        row, col = edge_index
        edge_weight = degree_inv_sqrt[row] * degree_inv_sqrt[col]
        
        # 构建稀疏矩阵
        indices = torch.stack([row, col], dim=0)
        adj = torch.sparse_coo_tensor(indices, edge_weight, (num_nodes, num_nodes))
        
        identity = torch.eye(num_nodes, device=edge_index.device).to_sparse()
        laplacian = identity - adj
    else:
        # L = D - A
        degree_mat = torch.diag(degree)
        
        row, col = edge_index
        adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        adj[row, col] = 1
        
        laplacian = degree_mat - adj
    
    return laplacian


def add_self_loops(
    edge_index: torch.Tensor,
    num_nodes: int,
    fill_value: float = 1.0,
) -> torch.Tensor:
    """添加自环"""
    loop_index = torch.arange(num_nodes, device=edge_index.device)
    loop_index = torch.stack([loop_index, loop_index], dim=0)
    
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    return edge_index


# ==================== GCN ====================

class GCNConv(nn.Module):
    """图卷积层"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        bias: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self._cached_norm = None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self._cached_norm = None
    
    def compute_norm(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """计算归一化系数"""
        row, col = edge_index
        
        # 计算度
        deg = compute_degree(edge_index, num_nodes)
        
        if self.improved:
            deg = deg + 1
        
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return norm
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播"""
        num_nodes = x.size(0)
        
        # 线性变换
        x = x @ self.weight
        
        # 归一化
        if self.normalize:
            if self._cached_norm is None or not self.cached:
                norm = self.compute_norm(edge_index, num_nodes)
                if edge_weight is not None:
                    norm = norm * edge_weight
                self._cached_norm = norm
            else:
                norm = self._cached_norm
        else:
            norm = edge_weight if edge_weight is not None else torch.ones(edge_index.size(1), device=x.device)
        
        # 消息传递
        row, col = edge_index
        out = torch.zeros_like(x)
        
        # x_j * norm
        messages = x[col] * norm.unsqueeze(-1)
        
        # 聚合
        out.scatter_add_(0, row.unsqueeze(-1).expand_as(messages), messages)
        
        # 偏置
        if self.bias is not None:
            out = out + self.bias
        
        return out


class GCN(nn.Module):
    """图卷积网络"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        normalize: bool = True,
    ):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout
        
        # 第一层
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize=normalize))
        self.norms.append(nn.LayerNorm(hidden_channels))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize=normalize))
            self.norms.append(nn.LayerNorm(hidden_channels))
        
        # 输出层
        self.convs.append(GCNConv(hidden_channels, out_channels, normalize=normalize))
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播"""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


# ==================== GAT ====================

class GATConv(nn.Module):
    """图注意力层"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """前向传播"""
        N = x.size(0)
        H, C = self.heads, self.out_channels
        
        # 线性变换
        x = self.lin(x).view(N, H, C)
        
        # 计算注意力分数
        alpha_src = (x * self.att_src).sum(dim=-1)  # [N, H]
        alpha_dst = (x * self.att_dst).sum(dim=-1)  # [N, H]
        
        row, col = edge_index
        
        # e_ij = LeakyReLU(a_src * h_i + a_dst * h_j)
        alpha = alpha_src[row] + alpha_dst[col]  # [E, H]
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        # Softmax
        alpha = self._softmax(alpha, row, N)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # 消息传递
        out = self._aggregate(x, alpha, edge_index, N)
        
        # 拼接或平均
        if self.concat:
            out = out.view(N, H * C)
        else:
            out = out.mean(dim=1)
        
        if self.bias is not None:
            out = out + self.bias
        
        if return_attention:
            return out, alpha
        return out
    
    def _softmax(self, alpha: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """边softmax"""
        alpha_max = torch.zeros(num_nodes, alpha.size(1), device=alpha.device)
        alpha_max.scatter_reduce_(0, index.unsqueeze(-1).expand_as(alpha), alpha, reduce='amax', include_self=False)
        alpha = (alpha - alpha_max[index]).exp()
        
        alpha_sum = torch.zeros(num_nodes, alpha.size(1), device=alpha.device)
        alpha_sum.scatter_add_(0, index.unsqueeze(-1).expand_as(alpha), alpha)
        alpha_sum = alpha_sum[index].clamp(min=1e-16)
        
        return alpha / alpha_sum
    
    def _aggregate(
        self,
        x: torch.Tensor,
        alpha: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """聚合消息"""
        row, col = edge_index
        
        # x_j * alpha_ij
        messages = x[col] * alpha.unsqueeze(-1)  # [E, H, C]
        
        # 聚合
        out = torch.zeros(num_nodes, x.size(1), x.size(2), device=x.device)
        out.scatter_add_(0, row.unsqueeze(-1).unsqueeze(-1).expand_as(messages), messages)
        
        return out


class GAT(nn.Module):
    """图注意力网络"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        heads: int = 8,
        dropout: float = 0.6,
    ):
        super().__init__()
        
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # 第一层
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        self.norms.append(nn.LayerNorm(hidden_channels * heads))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
            self.norms.append(nn.LayerNorm(hidden_channels * heads))
        
        # 输出层（平均多头）
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False))
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播"""
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


# ==================== GraphSAGE ====================

class SAGEConv(nn.Module):
    """GraphSAGE卷积层"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = 'mean',
        normalize: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        self.normalize = normalize
        
        self.lin_self = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.lin_self.weight)
        nn.init.xavier_uniform_(self.lin_neigh.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播"""
        row, col = edge_index
        num_nodes = x.size(0)
        
        # 聚合邻居
        if self.aggr == 'mean':
            # 均值聚合
            degree = compute_degree(edge_index, num_nodes)
            degree_inv = degree.pow(-1)
            degree_inv[degree_inv == float('inf')] = 0
            
            messages = x[col]
            aggregated = torch.zeros_like(x)
            aggregated.scatter_add_(0, row.unsqueeze(-1).expand_as(messages), messages)
            aggregated = aggregated * degree_inv.unsqueeze(-1)
        
        elif self.aggr == 'max':
            # 最大池化
            messages = x[col]
            aggregated = torch.zeros_like(x) - float('inf')
            aggregated.scatter_reduce_(0, row.unsqueeze(-1).expand_as(messages), messages, reduce='amax')
            aggregated[aggregated == -float('inf')] = 0
        
        elif self.aggr == 'lstm':
            # LSTM聚合（简化实现）
            aggregated = self._lstm_aggregate(x, edge_index, num_nodes)
        
        else:  # sum
            messages = x[col]
            aggregated = torch.zeros_like(x)
            aggregated.scatter_add_(0, row.unsqueeze(-1).expand_as(messages), messages)
        
        # 组合
        out = self.lin_self(x) + self.lin_neigh(aggregated)
        
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def _lstm_aggregate(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """LSTM聚合"""
        row, col = edge_index
        
        # 简化实现：使用均值聚合
        degree = compute_degree(edge_index, num_nodes)
        degree_inv = degree.pow(-1)
        degree_inv[degree_inv == float('inf')] = 0
        
        messages = x[col]
        aggregated = torch.zeros_like(x)
        aggregated.scatter_add_(0, row.unsqueeze(-1).expand_as(messages), messages)
        aggregated = aggregated * degree_inv.unsqueeze(-1)
        
        return aggregated


class GraphSAGE(nn.Module):
    """GraphSAGE网络"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        aggr: str = 'mean',
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.dropout = dropout
        
        # 第一层
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
        
        # 输出层
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播"""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


# ==================== GIN ====================

class GINConv(nn.Module):
    """图同构网络卷积层"""
    
    def __init__(
        self,
        nn: nn.Module,
        eps: float = 0.0,
        train_eps: bool = False,
    ):
        super().__init__()
        self.nn = nn
        
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播"""
        row, col = edge_index
        num_nodes = x.size(0)
        
        # 聚合邻居
        messages = x[col]
        aggregated = torch.zeros_like(x)
        aggregated.scatter_add_(0, row.unsqueeze(-1).expand_as(messages), messages)
        
        # (1 + eps) * x + aggregated
        out = (1 + self.eps) * x + aggregated
        
        # MLP
        out = self.nn(out)
        
        return out


class GIN(nn.Module):
    """图同构网络"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 5,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout
        
        # 第一层
        nn1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.convs.append(GINConv(nn1))
        self.norms.append(nn.LayerNorm(hidden_channels))
        
        # 中间层
        for _ in range(num_layers - 1):
            nni = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(nni))
            self.norms.append(nn.LayerNorm(hidden_channels))
        
        # 输出
        self.lin = nn.Linear(hidden_channels, out_channels)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播"""
        h_list = []
        
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            h_list.append(x)
        
        # 跳跃连接（拼接所有层）
        x = torch.stack(h_list, dim=0).sum(dim=0)
        x = self.lin(x)
        
        return x


# ==================== 消息传递神经网络 ====================

class MessagePassing(nn.Module):
    """消息传递基类"""
    
    def __init__(self, aggr: str = 'add'):
        super().__init__()
        self.aggr = aggr
    
    def message(self, x_j: torch.Tensor, **kwargs) -> torch.Tensor:
        """消息函数"""
        return x_j
    
    def aggregate(
        self,
        messages: torch.Tensor,
        index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """聚合函数"""
        out = torch.zeros(num_nodes, messages.size(-1), device=messages.device)
        
        if self.aggr == 'add':
            out.scatter_add_(0, index.unsqueeze(-1).expand_as(messages), messages)
        elif self.aggr == 'mean':
            out.scatter_add_(0, index.unsqueeze(-1).expand_as(messages), messages)
            degree = compute_degree(torch.stack([index, torch.zeros_like(index)]), num_nodes)
            out = out / degree.unsqueeze(-1).clamp(min=1)
        elif self.aggr == 'max':
            out.scatter_reduce_(0, index.unsqueeze(-1).expand_as(messages), messages, reduce='amax')
        
        return out
    
    def update(self, aggregated: torch.Tensor, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """更新函数"""
        return aggregated
    
    def propagate(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """传播函数"""
        row, col = edge_index
        num_nodes = x.size(0)
        
        # 提取源节点特征
        x_j = x[col]
        
        # 消息
        messages = self.message(x_j, **kwargs)
        
        # 聚合
        aggregated = self.aggregate(messages, row, num_nodes)
        
        # 更新
        out = self.update(aggregated, x, **kwargs)
        
        return out


class MPNNConv(MessagePassing):
    """通用消息传递卷积"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_channels: Optional[int] = None,
        aggr: str = 'add',
    ):
        super().__init__(aggr=aggr)
        
        self.message_mlp = nn.Sequential(
            nn.Linear(in_channels + (edge_channels or 0), out_channels),
            nn.ReLU(),
        )
        
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
        )
    
    def message(
        self,
        x_j: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """消息函数"""
        if edge_attr is not None:
            x_j = torch.cat([x_j, edge_attr], dim=-1)
        return self.message_mlp(x_j)
    
    def update(
        self,
        aggregated: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """更新函数"""
        x = torch.cat([x, aggregated], dim=-1)
        return self.update_mlp(x)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播"""
        return self.propagate(x, edge_index, edge_attr=edge_attr)


# ==================== 图池化 ====================

class GlobalMeanPool(nn.Module):
    """全局均值池化"""
    
    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if batch is None:
            return x.mean(dim=0, keepdim=True)
        
        num_graphs = batch.max().item() + 1
        out = torch.zeros(num_graphs, x.size(-1), device=x.device)
        count = torch.zeros(num_graphs, device=x.device)
        
        out.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x)
        count.scatter_add_(0, batch, torch.ones_like(batch, dtype=torch.float))
        
        return out / count.unsqueeze(-1).clamp(min=1)


class GlobalMaxPool(nn.Module):
    """全局最大池化"""
    
    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if batch is None:
            return x.max(dim=0, keepdim=True)[0]
        
        num_graphs = batch.max().item() + 1
        out = torch.zeros(num_graphs, x.size(-1), device=x.device) - float('inf')
        
        out.scatter_reduce_(0, batch.unsqueeze(-1).expand_as(x), x, reduce='amax')
        out[out == -float('inf')] = 0
        
        return out


class GlobalAttentionPool(nn.Module):
    """全局注意力池化"""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(in_channels, 1),
            nn.Softmax(dim=0),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if batch is None:
            att = self.att(x)
            return (x * att).sum(dim=0, keepdim=True)
        
        num_graphs = batch.max().item() + 1
        out = torch.zeros(num_graphs, x.size(-1), device=x.device)
        
        for i in range(num_graphs):
            mask = batch == i
            x_i = x[mask]
            att_i = self.att(x_i)
            out[i] = (x_i * att_i).sum(dim=0)
        
        return out


class Set2Set(nn.Module):
    """Set2Set池化"""
    
    def __init__(self, in_channels: int, processing_steps: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.processing_steps = processing_steps
        
        self.lstm = nn.LSTM(2 * in_channels, in_channels, batch_first=True)
    
    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        num_graphs = batch.max().item() + 1
        out = torch.zeros(num_graphs, 2 * self.in_channels, device=x.device)
        
        for i in range(num_graphs):
            mask = batch == i
            x_i = x[mask].unsqueeze(0)  # [1, N, D]
            
            h = torch.zeros(1, 1, self.in_channels, device=x.device)
            c = torch.zeros(1, 1, self.in_channels, device=x.device)
            q = torch.zeros(1, self.in_channels, device=x.device)
            
            for _ in range(self.processing_steps):
                # LSTM step
                q_star = torch.cat([q, x_i.mean(dim=1)], dim=-1).unsqueeze(1)
                _, (h, c) = self.lstm(q_star, (h, c))
                q = h.squeeze(0)
                
                # Attention
                alpha = F.softmax((x_i * q.unsqueeze(1)).sum(dim=-1), dim=1)
                readout = (x_i * alpha.unsqueeze(-1)).sum(dim=1)
            
            out[i] = torch.cat([q, readout.squeeze(0)], dim=-1)
        
        return out


# ==================== 图分类网络 ====================

class GraphClassifier(nn.Module):
    """图分类网络"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        gnn_type: str = 'gcn',
        pool_type: str = 'mean',
        dropout: float = 0.5,
    ):
        super().__init__()
        
        # GNN层
        if gnn_type == 'gcn':
            self.gnn = GCN(in_channels, hidden_channels, hidden_channels, num_layers, dropout)
        elif gnn_type == 'gat':
            self.gnn = GAT(in_channels, hidden_channels, hidden_channels, num_layers, dropout=dropout)
        elif gnn_type == 'sage':
            self.gnn = GraphSAGE(in_channels, hidden_channels, hidden_channels, num_layers, dropout=dropout)
        elif gnn_type == 'gin':
            self.gnn = GIN(in_channels, hidden_channels, hidden_channels, num_layers, dropout=dropout)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        # 池化层
        if pool_type == 'mean':
            self.pool = GlobalMeanPool()
        elif pool_type == 'max':
            self.pool = GlobalMaxPool()
        elif pool_type == 'attention':
            self.pool = GlobalAttentionPool(hidden_channels)
        elif pool_type == 'set2set':
            self.pool = Set2Set(hidden_channels)
        else:
            raise ValueError(f"Unknown pool type: {pool_type}")
        
        # 分类头
        pool_out_channels = hidden_channels * 2 if pool_type == 'set2set' else hidden_channels
        self.classifier = nn.Sequential(
            nn.Linear(pool_out_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播"""
        x = self.gnn(x, edge_index)
        x = self.pool(x, batch)
        x = self.classifier(x)
        return x


# ==================== 主函数 ====================

def main():
    """测试图神经网络"""
    print("图神经网络测试")
    
    # 创建测试图
    num_nodes = 100
    num_edges = 500
    in_channels = 16
    out_channels = 8
    
    # 随机节点特征
    x = torch.randn(num_nodes, in_channels)
    
    # 随机边
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # 测试GCN
    print("\n测试GCN...")
    gcn = GCN(in_channels, 32, out_channels, num_layers=3)
    out = gcn(x, edge_index)
    print(f"GCN output shape: {out.shape}")
    
    # 测试GAT
    print("\n测试GAT...")
    gat = GAT(in_channels, 32, out_channels, num_layers=2, heads=4)
    out = gat(x, edge_index)
    print(f"GAT output shape: {out.shape}")
    
    # 测试GraphSAGE
    print("\n测试GraphSAGE...")
    sage = GraphSAGE(in_channels, 32, out_channels, num_layers=3)
    out = sage(x, edge_index)
    print(f"GraphSAGE output shape: {out.shape}")
    
    # 测试GIN
    print("\n测试GIN...")
    gin = GIN(in_channels, 32, out_channels, num_layers=5)
    out = gin(x, edge_index)
    print(f"GIN output shape: {out.shape}")
    
    # 测试池化
    print("\n测试池化...")
    batch = torch.zeros(num_nodes, dtype=torch.long)
    
    mean_pool = GlobalMeanPool()
    out = mean_pool(x, batch)
    print(f"MeanPool output shape: {out.shape}")
    
    max_pool = GlobalMaxPool()
    out = max_pool(x, batch)
    print(f"MaxPool output shape: {out.shape}")
    
    # 测试图分类
    print("\n测试图分类...")
    classifier = GraphClassifier(in_channels, 32, 10, num_layers=3, gnn_type='gcn')
    out = classifier(x, edge_index, batch)
    print(f"GraphClassifier output shape: {out.shape}")
    
    print("\n图神经网络测试完成")


if __name__ == "__main__":
    main()
