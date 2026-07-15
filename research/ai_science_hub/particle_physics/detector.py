"""
AI科学中心 - 粒子物理探测器模块

使用SH-GNN进行粒子轨迹重建、顶点探测和能量测量。
支持高能物理实验中的探测器数据分析。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List


class ParticleDetectorSHGNN(nn.Module):
    """
    粒子探测器SH-GNN模型
    
    用于粒子轨迹重建、顶点探测和能量沉积分析。
    支持多种探测器几何结构（桶型、端盖型）。
    
    Attributes:
        input_dim: 输入特征维度（能量、时间、位置等）
        hidden_dim: 隐藏层维度
        output_dim: 输出维度（粒子类型、能量、动量等）
        l_max: 球谐函数最大阶数
        num_layers: GNN层数
        detector_type: 探测器类型 ('barrel', 'endcap', 'hybrid')
    """
    
    def __init__(
        self,
        input_dim: int = 7,  # x, y, z, t, E, charge, layer
        hidden_dim: int = 256,
        output_dim: int = 8,  # particle_id, px, py, pz, E, vertex_x, vertex_y, vertex_z
        l_max: int = 8,
        num_layers: int = 6,
        detector_type: str = 'hybrid'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.l_max = l_max
        self.detector_type = detector_type
        
        # 探测器几何编码器
        self.geometry_encoder = DetectorGeometryEncoder(input_dim, hidden_dim, detector_type)
        
        # SH-GNN层
        self.sh_gnn_layers = nn.ModuleList([
            ParticleEquivariantLayer(hidden_dim, hidden_dim, l_max)
            for _ in range(num_layers)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # 轨迹重建头
        self.track_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 6)  # px, py, pz, x0, y0, z0
        )
        
        # 粒子识别头
        self.pid_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 5)  # e, mu, pi, K, p
        )
        
        # 能量回归头
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 顶点探测头
        self.vertex_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3)  # vertex_x, vertex_y, vertex_z
        )
        
    def forward(
        self,
        hits: torch.Tensor,
        hit_positions: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            hits: (N, input_dim) 探测器击中信号特征
            hit_positions: (N, 3) 击中位置 (x, y, z)
            batch_indices: (N,) 批次索引，用于区分不同事件
            
        Returns:
            Dict包含:
                - 'track_params': (N, 6) 轨迹参数
                - 'particle_probs': (N, 5) 粒子类型概率
                - 'energy': (N, 1) 能量预测
                - 'vertex': (N, 3) 顶点位置
                - 'embeddings': (N, hidden_dim) 节点嵌入
        """
        N = hits.shape[0]
        if batch_indices is None:
            batch_indices = torch.zeros(N, dtype=torch.long, device=hits.device)
        
        # 编码探测器几何信息
        h = self.geometry_encoder(hits, hit_positions)
        
        # SH-GNN处理
        for layer, norm in zip(self.sh_gnn_layers, self.layer_norms):
            h_new = layer(h, hit_positions, batch_indices)
            h = norm(h + h_new)
            h = F.silu(h)
        
        # 多任务预测
        track_params = self.track_head(h)
        particle_probs = F.softmax(self.pid_head(h), dim=-1)
        energy = F.relu(self.energy_head(h))
        vertex = self.vertex_head(h)
        
        return {
            'track_params': track_params,
            'particle_probs': particle_probs,
            'energy': energy,
            'vertex': vertex,
            'embeddings': h
        }
    
    def reconstruct_tracks(
        self,
        predictions: Dict[str, torch.Tensor],
        hit_positions: torch.Tensor,
        min_prob: float = 0.5
    ) -> List[Dict[str, torch.Tensor]]:
        """
        从预测结果重建粒子轨迹
        
        Args:
            predictions: 模型输出
            hit_positions: 击中位置
            min_prob: 最小概率阈值
            
        Returns:
            轨迹列表，每条轨迹包含位置和动量信息
        """
        tracks = []
        particle_probs = predictions['particle_probs']
        max_probs, particle_types = torch.max(particle_probs, dim=-1)
        
        mask = max_probs > min_prob
        if mask.sum() > 0:
            track_params = predictions['track_params'][mask]
            positions = hit_positions[mask]
            energies = predictions['energy'][mask]
            pids = particle_types[mask]
            
            for i in range(mask.sum()):
                tracks.append({
                    'position': positions[i],
                    'momentum': track_params[i, :3],
                    'origin': track_params[i, 3:6],
                    'energy': energies[i],
                    'particle_type': pids[i]
                })
        
        return tracks


class DetectorGeometryEncoder(nn.Module):
    """探测器几何编码器
    
    编码探测器的几何结构信息，支持桶型和端盖型探测器。
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, detector_type: str):
        super().__init__()
        self.detector_type = detector_type
        
        # 位置编码
        self.position_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        
        # 特征编码
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim - 64),
            nn.SiLU(),
            nn.Linear(hidden_dim - 64, hidden_dim - 64)
        )
        
        # 几何感知融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, hits: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """编码击中信号和位置信息"""
        pos_enc = self.position_encoder(positions)
        feat_enc = self.feature_encoder(hits)
        combined = torch.cat([feat_enc, pos_enc], dim=-1)
        return self.fusion(combined)


class ParticleEquivariantLayer(nn.Module):
    """粒子物理等变层
    
    使用球谐函数保持旋转等变性，适合处理探测器中的粒子轨迹。
    """
    
    def __init__(self, in_dim: int, out_dim: int, l_max: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.l_max = l_max
        
        # 消息网络
        self.message_net = nn.Sequential(
            nn.Linear(in_dim * 2 + 3, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )
        
        # 自环权重
        self.self_weight = nn.Linear(in_dim, out_dim)
        
        # 球谐函数系数
        self.sh_coeffs = nn.Parameter(torch.randn(l_max + 1) * 0.1)
        
    def forward(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        batch_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        等变消息传递
        
        Args:
            features: (N, in_dim) 节点特征
            positions: (N, 3) 节点位置
            batch_indices: (N,) 批次索引
        """
        N = features.shape[0]
        device = features.device
        
        # 计算相对位置和距离
        rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)  # (N, N, 3)
        dist = torch.norm(rel_pos, dim=-1, keepdim=True) + 1e-8
        direction = rel_pos / dist
        
        # 球谐函数滤波
        sh_filter = self._compute_sh_filter(dist.squeeze(-1))
        
        # 按批次进行消息传递
        messages = []
        unique_batches = torch.unique(batch_indices)
        
        for batch_id in unique_batches:
            batch_mask = batch_indices == batch_id
            batch_indices_list = torch.where(batch_mask)[0]
            n_batch = batch_indices_list.shape[0]
            
            if n_batch == 0:
                continue
            
            batch_features = features[batch_mask]
            batch_directions = direction[batch_mask][:, batch_mask]
            batch_dist = dist[batch_mask][:, batch_mask]
            batch_filter = sh_filter[batch_mask][:, batch_mask]
            
            # 限制邻居数量以提高效率
            k = min(32, n_batch)
            _, topk_indices = torch.topk(-batch_dist.squeeze(-1), k, dim=-1)
            
            batch_messages = []
            for i in range(n_batch):
                neighbors = topk_indices[i]
                feat_neighbors = batch_features[neighbors]
                feat_i = batch_features[i:i+1].expand(k, -1)
                dir_i = batch_directions[i, neighbors]
                
                message_input = torch.cat([feat_i, feat_neighbors, dir_i], dim=-1)
                message = self.message_net(message_input)
                
                # 应用球谐滤波权重
                weights = batch_filter[i, neighbors].unsqueeze(-1)
                weighted_message = (message * weights).mean(dim=0)
                batch_messages.append(weighted_message)
            
            messages.extend(batch_messages)
        
        aggregated = torch.stack(messages)
        self_loop = self.self_weight(features)
        
        return aggregated + self_loop
    
    def _compute_sh_filter(self, dist: torch.Tensor) -> torch.Tensor:
        """计算球谐函数滤波器"""
        # 简化的球谐滤波：使用距离的高斯加权和
        filter_vals = torch.zeros_like(dist)
        for l in range(self.l_max + 1):
            sigma = 0.1 * (l + 1)
            filter_vals += self.sh_coeffs[l] * torch.exp(-dist**2 / (2 * sigma**2))
        return torch.sigmoid(filter_vals)


class CalorimeterClusterer(nn.Module):
    """
    量能器聚类器
    
    对量能器中的能量沉积进行聚类，识别电磁和强子簇射。
    """
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.cluster_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)  # cluster_id, EM likelihood, hadronic likelihood
        )
        
    def forward(self, energy_deposits: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        聚类能量沉积
        
        Args:
            energy_deposits: (N, 4) 能量、时间、层数、单元格ID
            positions: (N, 3) 位置
            
        Returns:
            (N, 3) 聚类结果
        """
        features = self.encoder(energy_deposits)
        clusters = self.cluster_head(features)
        return clusters


class VertexFinder(nn.Module):
    """
    顶点探测器
    
    从粒子轨迹重建衰变顶点位置。
    """
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        
        self.track_processor = nn.Sequential(
            nn.Linear(6, hidden_dim),  # 6个轨迹参数
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.vertex_regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 3)  # vertex (x, y, z)
        )
        
    def forward(self, track_params: torch.Tensor) -> torch.Tensor:
        """
        重建顶点位置
        
        Args:
            track_params: (N, 6) 轨迹参数
            
        Returns:
            (3,) 顶点位置
        """
        processed = self.track_processor(track_params)
        # 全局平均池化
        global_feat = processed.mean(dim=0, keepdim=True).expand(processed.shape[0], -1)
        combined = torch.cat([processed, global_feat], dim=-1)
        
        # 对所有轨迹预测顶点，然后平均
        vertices = self.vertex_regressor(combined)
        return vertices.mean(dim=0)
