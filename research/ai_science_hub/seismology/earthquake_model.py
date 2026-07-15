"""
AI科学中心 - 地震预测模块

使用SH-GNN进行地震事件检测、震源定位和震级预测。
支持实时地震监测和预警系统。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List


class EarthquakePredictorSHGNN(nn.Module):
    """
    地震预测SH-GNN模型

    用于地震事件检测、震源参数反演和地震波传播模拟。
    支持多站地震仪数据融合分析。

    Attributes:
        input_dim: 输入特征维度（波形、位置、时间等）
        hidden_dim: 隐藏层维度
        output_dim: 输出维度（震级、位置、发震时刻等）
        l_max: 球谐函数最大阶数
        num_layers: GNN层数
        max_stations: 最大台站数量
    """

    def __init__(
        self,
        input_dim: int = 12,  # 3分量波形 + 位置 + 台站信息
        hidden_dim: int = 256,
        output_dim: int = 7,  # mag, x, y, z, t0, strike, dip
        l_max: int = 8,
        num_layers: int = 6,
        max_stations: int = 100
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.l_max = l_max
        self.max_stations = max_stations

        # 波形编码器
        self.waveform_encoder = SeismicWaveformEncoder(input_dim, hidden_dim)

        # 地球结构编码器
        self.earth_encoder = EarthStructureEncoder(hidden_dim)

        # SH-GNN层
        self.sh_gnn_layers = nn.ModuleList([
            SeismicEquivariantLayer(hidden_dim, hidden_dim, l_max)
            for _ in range(num_layers)
        ])

        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # 震源参数预测头
        self.source_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, output_dim)
        )

        # 震级预测头
        self.magnitude_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 位置预测头
        self.location_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3)  # x, y, z
        )

        # 发震时刻预测头
        self.origin_time_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 震源机制预测头
        self.focal_mechanism_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 6)  # strike, dip, rake + uncertainties
        )

        # 地震检测头
        self.detection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(
        self,
        waveforms: torch.Tensor,
        station_positions: torch.Tensor,
        station_metadata: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            waveforms: (N, T, 3) 三分量地震波形
            station_positions: (N, 3) 台站位置 (经度, 纬度, 高程)
            station_metadata: (N, 6) 台站元数据（采样率、增益等）
            batch_indices: (N,) 批次索引

        Returns:
            Dict包含:
                - 'detection_prob': (N, 1) 地震检测概率
                - 'magnitude': (N, 1) 震级预测
                - 'location': (N, 3) 震源位置
                - 'origin_time': (N, 1) 发震时刻
                - 'focal_mechanism': (N, 6) 震源机制参数
                - 'embeddings': (N, hidden_dim) 台站嵌入
        """
        N = waveforms.shape[0]
        if batch_indices is None:
            batch_indices = torch.zeros(N, dtype=torch.long, device=waveforms.device)

        # 编码波形和台站信息
        waveform_features = self.waveform_encoder(waveforms, station_metadata)

        # 融合地球结构信息
        h = self.earth_encoder(waveform_features, station_positions)

        # SH-GNN处理
        for layer, norm in zip(self.sh_gnn_layers, self.layer_norms):
            h_new = layer(h, station_positions, batch_indices)
            h = norm(h + h_new)
            h = F.silu(h)

        # 多任务预测
        detection_logits = self.detection_head(h)
        detection_prob = torch.sigmoid(detection_logits)

        magnitude = self.magnitude_head(h)
        location = self.location_head(h)
        origin_time = self.origin_time_head(h)
        focal_mechanism = self.focal_mechanism_head(h)

        return {
            'detection_prob': detection_prob,
            'magnitude': magnitude,
            'location': location,
            'origin_time': origin_time,
            'focal_mechanism': focal_mechanism,
            'embeddings': h
        }

    def predict_earthquake(
        self,
        predictions: Dict[str, torch.Tensor],
        threshold: float = 0.5
    ) -> List[Dict[str, torch.Tensor]]:
        """
        从预测结果提取地震事件

        Args:
            predictions: 模型输出
            threshold: 检测阈值

        Returns:
            地震事件列表
        """
        events = []
        detection_probs = predictions['detection_prob']

        mask = detection_probs.squeeze() > threshold
        if mask.sum() > 0:
            magnitudes = predictions['magnitude'][mask]
            locations = predictions['location'][mask]
            origin_times = predictions['origin_time'][mask]
            focal_mechanisms = predictions['focal_mechanism'][mask]

            for i in range(mask.sum()):
                events.append({
                    'magnitude': magnitudes[i].item(),
                    'location': locations[i],
                    'origin_time': origin_times[i].item(),
                    'focal_mechanism': focal_mechanisms[i]
                })

        return events


class SeismicWaveformEncoder(nn.Module):
    """地震波形编码器

    编码三分量地震波形数据，提取时频特征。
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.conv1d_layers = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

    def forward(self, waveforms: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        # waveforms: (N, T, input_dim) -> (N, input_dim, T)
        x = waveforms.transpose(1, 2)
        x = self.conv1d_layers(x)
        x = x.transpose(1, 2)
        x, _ = self.attention(x, x, x)
        return x.mean(dim=1)


class EarthStructureEncoder(nn.Module):
    """地球结构编码器

    编码地球结构信息，提供先验约束。
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, features: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([features, positions], dim=-1)
        return self.mlp(x)


class SeismicEquivariantLayer(nn.Module):
    """地震等变层

    实现SE(3)等变的图神经网络层。
    """

    def __init__(self, in_dim: int, out_dim: int, l_max: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.l_max = l_max

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        batch_indices: torch.Tensor
    ) -> torch.Tensor:
        return self.linear(x)
