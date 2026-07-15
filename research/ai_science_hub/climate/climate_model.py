"""
AI科学中心 - 气候科学模块

使用SH-GNN进行气候模式预测和极端天气事件检测。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional


class ClimateSHGNN(nn.Module):
    """
    气候预测SH-GNN模型
    
    预测全球温度、降水、海平面等气候变量。
    """
    
    def __init__(
        self,
        input_dim: int = 10,  # 温度、湿度、气压、风速等
        hidden_dim: int = 128,
        output_dim: int = 5,  # 预测变量数
        l_max: int = 10,
        num_layers: int = 4
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 球面网格编码（气候数据通常在球面上）
        self.spherical_encoder = SphericalGridEncoder(input_dim, hidden_dim, l_max)
        
        # SH-GNN层
        self.sh_gnn_layers = nn.ModuleList([
            ClimateEquivariantLayer(hidden_dim, hidden_dim, l_max)
            for _ in range(num_layers)
        ])
        
        # 输出头
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(
        self,
        climate_data: torch.Tensor,
        lat_lon_grid: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            climate_data: (batch, H, W, input_dim) 气候变量
            lat_lon_grid: (H, W, 2) 经纬度网格
            
        Returns:
            (batch, H, W, output_dim) 预测结果
        """
        batch_size, H, W, _ = climate_data.shape
        
        # 将球面数据转换为点云
        points, features = self._grid_to_pointcloud(climate_data, lat_lon_grid)
        
        # 编码
        h = self.spherical_encoder(points, features)
        
        # SH-GNN处理
        for layer in self.sh_gnn_layers:
            h = layer(h, points) + h
            h = torch.relu(h)
        
        # 输出
        output = self.output_head(h)
        
        # 转换回网格
        output_grid = self._pointcloud_to_grid(output, H, W)
        
        return output_grid
    
    def _grid_to_pointcloud(
        self,
        data: torch.Tensor,
        lat_lon: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """将网格数据转换为点云"""
        batch_size = data.shape[0]
        H, W = lat_lon.shape[:2]
        
        # 经纬度转3D坐标
        lat = lat_lon[:, :, 0]
        lon = lat_lon[:, :, 1]
        
        x = torch.cos(lat) * torch.cos(lon)
        y = torch.cos(lat) * torch.sin(lon)
        z = torch.sin(lat)
        
        points = torch.stack([x, y, z], dim=-1)  # (H, W, 3)
        points = points.reshape(-1, 3).unsqueeze(0).expand(batch_size, -1, -1)
        
        features = data.reshape(batch_size, H * W, -1)
        
        return points, features
    
    def _pointcloud_to_grid(
        self,
        pointcloud: torch.Tensor,
        H: int,
        W: int
    ) -> torch.Tensor:
        """将点云转换回网格"""
        batch_size = pointcloud.shape[0]
        return pointcloud.reshape(batch_size, H, W, -1)


class SphericalGridEncoder(nn.Module):
    """球面网格编码器"""
    
    def __init__(self, input_dim: int, hidden_dim: int, l_max: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, points: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)


class ClimateEquivariantLayer(nn.Module):
    """气候等变层"""
    
    def __init__(self, in_dim: int, out_dim: int, l_max: int):
        super().__init__()
        self.message_net = nn.Sequential(
            nn.Linear(in_dim * 2 + 3, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.self_weight = nn.Linear(in_dim, out_dim)
    
    def forward(self, features: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        # 简化：全局消息传递
        batch_size, num_points, _ = points.shape
        
        # 计算相对位置
        rel_pos = points.unsqueeze(2) - points.unsqueeze(1)
        dist = torch.norm(rel_pos, dim=-1, keepdim=True) + 1e-8
        direction = rel_pos / dist
        
        # 消息传递（使用k近邻简化）
        k = min(16, num_points)
        _, topk_indices = torch.topk(-dist.squeeze(-1), k, dim=-1)
        
        messages = []
        for i in range(num_points):
            neighbors = topk_indices[:, i, :]
            feat_neighbors = torch.gather(
                features, 1,
                neighbors.unsqueeze(-1).expand(-1, -1, features.shape[-1])
            )
            feat_i = features[:, i:i+1, :].expand(-1, k, -1)
            dir_i = direction[:, i:i+1, neighbors[0], :].expand(batch_size, k, -1)
            
            message_input = torch.cat([feat_i, feat_neighbors, dir_i], dim=-1)
            message = self.message_net(message_input).mean(dim=1)
            messages.append(message)
        
        aggregated = torch.stack(messages, dim=1)
        self_loop = self.self_weight(features)
        
        return aggregated + self_loop


class ExtremeWeatherDetector(nn.Module):
    """
    极端天气事件检测器
    
    检测台风、暴雨、干旱等极端事件。
    """
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.detection_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 4, kernel_size=1)  # 4类极端事件
        )
        
    def forward(self, climate_data: torch.Tensor) -> torch.Tensor:
        """
        检测极端天气
        
        Args:
            climate_data: (batch, C, H, W)
            
        Returns:
            (batch, 4, H, W) 每类的概率图
        """
        features = self.feature_extractor(climate_data)
        logits = self.detection_head(features)
        return torch.sigmoid(logits)
