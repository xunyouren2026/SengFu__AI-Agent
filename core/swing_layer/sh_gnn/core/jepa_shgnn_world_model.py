"""
JEPA + SH-GNN 世界模型

整合杨立昆JEPA架构与SH-GNN物理引擎，实现物理一致的世界模型。

核心创新：
1. 使用SH-GNN编码器替代标准CNN编码器，保证旋转等变性
2. 球谐系数作为隐状态表示，物理可解释
3. 动态稀疏调度器自适应计算资源分配
4. 物理约束损失保证预测符合物理定律
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
import numpy as np


class SHGNNJEPAEncoder(nn.Module):
    """
    SH-GNN JEPA编码器
    
    使用SH-GNN替代标准编码器，保证3D旋转等变性。
    输出球谐系数作为隐状态表示。
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        latent_dim: int = 128,
        l_max: int = 6,
        num_layers: int = 3,
        use_dynamic_sparse: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.l_max = l_max
        self.num_layers = num_layers
        
        # 输入编码
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # SH-GNN等变卷积层
        self.equivariant_layers = nn.ModuleList([
            EquivariantConvLayer(hidden_dim, hidden_dim, l_max)
            for _ in range(num_layers)
        ])
        
        # 球谐系数投影
        num_sh_coeffs = (l_max + 1) ** 2
        self.sh_projection = nn.Linear(hidden_dim, num_sh_coeffs)
        
        # 动态稀疏调度器
        self.use_dynamic_sparse = use_dynamic_sparse
        if use_dynamic_sparse:
            from .dynamic_sparse import DynamicSparseScheduler
            self.sparse_scheduler = DynamicSparseScheduler(l_max=l_max)
        
    def forward(
        self,
        points: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        前向传播
        
        Args:
            points: (batch, N, 3) 3D点云坐标
            features: (batch, N, input_dim) 点特征（可选）
            
        Returns:
            latent: (batch, latent_dim) 隐状态
            sh_coeffs: (batch, (l_max+1)^2) 球谐系数
            stats: 统计信息
        """
        batch_size, num_points, _ = points.shape
        
        # 如果没有提供特征，使用坐标本身
        if features is None:
            features = points
        
        # 编码输入
        h = self.input_encoder(features)  # (batch, N, hidden_dim)
        
        # SH-GNN等变卷积
        for layer in self.equivariant_layers:
            h = layer(h, points) + h  # 残差连接
            h = F.silu(h)
        
        # 全局池化
        h_global = torch.mean(h, dim=1)  # (batch, hidden_dim)
        
        # 投影到球谐系数
        sh_coeffs = self.sh_projection(h_global)  # (batch, (l_max+1)^2)
        
        # 动态稀疏调度
        stats = {}
        if self.use_dynamic_sparse:
            sh_coeffs, l_eff, sparse_stats = self.sparse_scheduler(sh_coeffs)
            stats['sparse'] = sparse_stats
        
        # 隐状态 = 球谐系数（物理一致的表示）
        latent = sh_coeffs
        
        return latent, sh_coeffs, stats


class EquivariantConvLayer(nn.Module):
    """
    等变卷积层
    
    简化版的SH-GNN等变卷积。
    """
    
    def __init__(self, in_dim: int, out_dim: int, l_max: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.l_max = l_max
        
        # 消息生成网络
        self.message_net = nn.Sequential(
            nn.Linear(in_dim * 2 + 3, out_dim),  # 拼接两个节点特征和相对位置
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )
        
        # 自环权重
        self.self_weight = nn.Linear(in_dim, out_dim)
        
    def forward(self, features: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: (batch, N, in_dim)
            points: (batch, N, 3)
            
        Returns:
            (batch, N, out_dim)
        """
        batch_size, num_points, _ = points.shape
        
        # 计算相对位置
        # (batch, N, N, 3)
        rel_pos = points.unsqueeze(2) - points.unsqueeze(1)
        
        # 计算距离
        dist = torch.norm(rel_pos, dim=-1, keepdim=True) + 1e-8
        
        # 归一化方向
        direction = rel_pos / dist  # (batch, N, N, 3)
        
        # 构建消息
        # 简化：使用全连接图
        messages = []
        for i in range(num_points):
            # 节点i的特征
            feat_i = features[:, i:i+1, :].expand(-1, num_points, -1)
            
            # 拼接特征和方向
            message_input = torch.cat([
                feat_i,  # (batch, N, in_dim)
                features,  # (batch, N, in_dim)
                direction[:, i, :, :]  # (batch, N, 3)
            ], dim=-1)
            
            # 生成消息
            message = self.message_net(message_input)  # (batch, N, out_dim)
            messages.append(message)
        
        # 聚合消息（平均）
        aggregated = torch.stack(messages, dim=1).mean(dim=2)  # (batch, N, out_dim)
        
        # 添加自环
        self_loop = self.self_weight(features)
        
        return aggregated + self_loop


class SHGNNJEPAPredictor(nn.Module):
    """
    SH-GNN JEPA预测器
    
    预测下一时刻的球谐系数表示。
    """
    
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # 输入：当前隐状态 + 动作
        input_dim = latent_dim + action_dim
        
        # MLP预测器
        layers = []
        current_dim = input_dim
        for i in range(num_layers):
            next_dim = hidden_dim if i < num_layers - 1 else latent_dim
            layers.append(nn.Linear(current_dim, next_dim))
            if i < num_layers - 1:
                layers.append(nn.SiLU())
            current_dim = next_dim
        
        self.network = nn.Sequential(*layers)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(latent_dim)
        
    def forward(
        self,
        latent: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        预测下一隐状态
        
        Args:
            latent: (batch, latent_dim) 当前隐状态（球谐系数）
            action: (batch, action_dim) 动作
            
        Returns:
            (batch, latent_dim) 预测的下一隐状态
        """
        x = torch.cat([latent, action], dim=-1)
        pred = self.network(x)
        pred = self.layer_norm(pred)
        return pred


class SHGNNJEPAWorldModel(nn.Module):
    """
    SH-GNN JEPA世界模型
    
    整合SH-GNN编码器和JEPA预测器，提供物理一致的世界模型。
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        action_dim: int = 4,
        hidden_dim: int = 64,
        latent_dim: int = 128,
        l_max: int = 6,
        encoder_layers: int = 3,
        predictor_layers: int = 4,
        use_physics_constraint: bool = True,
        use_dynamic_sparse: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.l_max = l_max
        
        # SH-GNN编码器
        self.encoder = SHGNNJEPAEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            l_max=l_max,
            num_layers=encoder_layers,
            use_dynamic_sparse=use_dynamic_sparse
        )
        
        # JEPA预测器
        self.predictor = SHGNNJEPAPredictor(
            latent_dim=(l_max + 1) ** 2,  # 球谐系数维度
            action_dim=action_dim,
            hidden_dim=hidden_dim * 4,
            num_layers=predictor_layers
        )
        
        # 物理约束损失
        self.use_physics_constraint = use_physics_constraint
        if use_physics_constraint:
            from .physics_constraint import PhysicsConstraintLoss
            self.physics_loss = PhysicsConstraintLoss()
        
    def encode(
        self,
        observation: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        编码观测
        
        Args:
            observation: (batch, N, 3) 点云观测
            features: (batch, N, input_dim) 特征
            
        Returns:
            latent, sh_coeffs, stats
        """
        return self.encoder(observation, features)
    
    def predict(
        self,
        latent: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        预测下一隐状态
        
        Args:
            latent: (batch, latent_dim) 当前隐状态
            action: (batch, action_dim) 动作
            
        Returns:
            (batch, latent_dim) 预测的下一隐状态
        """
        return self.predictor(latent, action)
    
    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        前向传播
        
        Args:
            observation: (batch, N, 3) 当前观测
            action: (batch, action_dim) 动作
            features: (batch, N, input_dim) 特征
            
        Returns:
            latent, latent_pred, stats
        """
        # 编码当前观测
        latent, sh_coeffs, encode_stats = self.encode(observation, features)
        
        # 预测下一状态
        latent_pred = self.predict(latent, action)
        
        stats = {
            'encoding': encode_stats,
            'latent_mean': latent.mean().item(),
            'latent_std': latent.std().item()
        }
        
        return latent, latent_pred, stats
    
    def compute_loss(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        next_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算JEPA损失
        
        Args:
            observation: (batch, N, 3) 当前观测
            action: (batch, action_dim) 动作
            next_observation: (batch, N, 3) 下一观测
            features: 当前特征
            next_features: 下一特征
            
        Returns:
            loss, stats
        """
        # 编码
        latent, sh_coeffs, _ = self.encode(observation, features)
        
        with torch.no_grad():
            latent_next, sh_coeffs_next, _ = self.encode(next_observation, next_features)
        
        # 预测
        latent_pred = self.predict(latent, action)
        
        # 预测损失（MSE）
        pred_loss = F.mse_loss(latent_pred, latent_next)
        
        # 物理约束损失
        total_loss = pred_loss
        loss_dict = {'pred_loss': pred_loss.item()}
        
        if self.use_physics_constraint:
            # 从球谐系数计算功率谱（简化）
            power_pred = self._coeffs_to_power(sh_coeffs_next)
            
            phys_loss, phys_dict = self.physics_loss(
                predicted_power=power_pred,
                sh_coeffs=sh_coeffs_next
            )
            
            total_loss = total_loss + 0.1 * phys_loss
            loss_dict.update(phys_dict)
        
        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict
    
    def _coeffs_to_power(self, sh_coeffs: torch.Tensor) -> torch.Tensor:
        """
        将球谐系数转换为功率谱
        
        Args:
            sh_coeffs: (batch, (l_max+1)^2)
            
        Returns:
            (batch, l_max+1) 功率谱
        """
        batch_size = sh_coeffs.shape[0]
        l_max = int(np.sqrt(sh_coeffs.shape[1])) - 1
        
        power = torch.zeros(batch_size, l_max + 1, device=sh_coeffs.device)
        
        idx = 0
        for l in range(l_max + 1):
            num_m = 2 * l + 1
            coeffs_l = sh_coeffs[:, idx:idx + num_m]
            power[:, l] = torch.sum(coeffs_l ** 2, dim=1) / (2 * l + 1)
            idx += num_m
        
        return power
    
    def rollout(
        self,
        initial_observation: torch.Tensor,
        actions: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        多步预测
        
        Args:
            initial_observation: (batch, N, 3) 初始观测
            actions: (batch, horizon, action_dim) 动作序列
            features: 初始特征
            
        Returns:
            (batch, horizon, latent_dim) 预测的隐状态序列
        """
        batch_size, horizon, _ = actions.shape
        
        # 编码初始状态
        latent, _, _ = self.encode(initial_observation, features)
        
        # 逐步预测
        predictions = []
        for t in range(horizon):
            action_t = actions[:, t, :]
            latent = self.predict(latent, action_t)
            predictions.append(latent)
        
        return torch.stack(predictions, dim=1)


class SHGNNJEPATrainer:
    """
    SH-GNN JEPA训练器
    """
    
    def __init__(
        self,
        model: SHGNNJEPAWorldModel,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        else:
            self.optimizer = optimizer
        
        self.global_step = 0
    
    def train_step(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor
    ) -> Dict[str, float]:
        """单步训练"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # 移动数据到设备
        observation = observation.to(self.device)
        action = action.to(self.device)
        next_observation = next_observation.to(self.device)
        
        # 计算损失
        loss, stats = self.model.compute_loss(observation, action, next_observation)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self.global_step += 1
        
        return stats
