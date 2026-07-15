"""
JEPA - 联合嵌入预测架构

LeCun提出的世界模型架构，通过编码器-预测器结构
学习环境的动态模型。

核心约400行。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math


class JEPAEncoder(nn.Module):
    """
    JEPA编码器
    
    将观测编码为隐状态表示。
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_layers: int = 4,
        activation: str = 'swish'
    ):
        """
        初始化编码器
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            latent_dim: 隐状态维度
            num_layers: 层数
            activation: 激活函数类型
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 构建网络
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            next_dim = hidden_dim if i < num_layers - 1 else latent_dim
            layers.append(nn.Linear(current_dim, next_dim))
            
            if i < num_layers - 1:
                if activation == 'swish':
                    layers.append(nn.SiLU())
                elif activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'gelu':
                    layers.append(nn.GELU())
            
            current_dim = next_dim
        
        self.network = nn.Sequential(*layers)
        
        # 层归一化（输出）
        self.layer_norm = nn.LayerNorm(latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (batch_size, input_dim)输入
        
        Returns:
            (batch_size, latent_dim)隐状态
        """
        z = self.network(x)
        z = self.layer_norm(z)
        return z


class JEPAPredictor(nn.Module):
    """
    JEPA预测器
    
    根据当前隐状态和动作预测下一隐状态。
    """
    
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        activation: str = 'swish'
    ):
        """
        初始化预测器
        
        Args:
            latent_dim: 隐状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
            num_layers: 层数
            activation: 激活函数类型
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # 输入：当前隐状态 + 动作
        input_dim = latent_dim + action_dim
        
        # 构建网络
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            next_dim = hidden_dim if i < num_layers - 1 else latent_dim
            layers.append(nn.Linear(current_dim, next_dim))
            
            if i < num_layers - 1:
                if activation == 'swish':
                    layers.append(nn.SiLU())
                elif activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'gelu':
                    layers.append(nn.GELU())
            
            current_dim = next_dim
        
        self.network = nn.Sequential(*layers)
        
        # 层归一化（输出）
        self.layer_norm = nn.LayerNorm(latent_dim)
    
    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            z: (batch_size, latent_dim)当前隐状态
            action: (batch_size, action_dim)动作
        
        Returns:
            (batch_size, latent_dim)预测的下一隐状态
        """
        # 拼接隐状态和动作
        x = torch.cat([z, action], dim=-1)
        
        # 预测
        z_pred = self.network(x)
        z_pred = self.layer_norm(z_pred)
        
        return z_pred


class JEPAWorldModel(nn.Module):
    """
    JEPA世界模型
    
    整合编码器和预测器，提供完整的世界模型功能。
    """
    
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        encoder_layers: int = 4,
        predictor_layers: int = 4,
        use_sigreg: bool = True,
        sigreg_weight: float = 1.0
    ):
        """
        初始化世界模型
        
        Args:
            input_dim: 输入观测维度
            action_dim: 动作维度
            latent_dim: 隐状态维度
            hidden_dim: 隐藏层维度
            encoder_layers: 编码器层数
            predictor_layers: 预测器层数
            use_sigreg: 是否使用SIGReg
            sigreg_weight: SIGReg权重
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.use_sigreg = use_sigreg
        self.sigreg_weight = sigreg_weight
        
        # 编码器
        self.encoder = JEPAEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=encoder_layers
        )
        
        # 预测器
        self.predictor = JEPAPredictor(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=predictor_layers
        )
        
        # SIGReg（可选）
        if use_sigreg:
            from .sigreg import SIGReg
            self.sigreg = SIGReg()
        else:
            self.sigreg = None
    
    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        """
        编码观测
        
        Args:
            observation: (batch_size, input_dim)观测
        
        Returns:
            (batch_size, latent_dim)隐状态
        """
        return self.encoder(observation)
    
    def predict(
        self,
        z: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        预测下一隐状态
        
        Args:
            z: (batch_size, latent_dim)当前隐状态
            action: (batch_size, action_dim)动作
        
        Returns:
            (batch_size, latent_dim)预测的下一隐状态
        """
        return self.predictor(z, action)
    
    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            observation: (batch_size, input_dim)当前观测
            action: (batch_size, action_dim)动作
        
        Returns:
            z: (batch_size, latent_dim)当前隐状态
            z_pred: (batch_size, latent_dim)预测的下一隐状态
        """
        z = self.encode(observation)
        z_pred = self.predict(z, action)
        return z, z_pred
    
    def compute_loss(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算JEPA损失
        
        Args:
            observation: (batch_size, input_dim)当前观测
            action: (batch_size, action_dim)动作
            next_observation: (batch_size, input_dim)下一观测
        
        Returns:
            loss: 总损失
            stats: 统计信息
        """
        # 编码当前和下一观测
        z = self.encode(observation)
        with torch.no_grad():
            z_next = self.encode(next_observation)
        
        # 预测下一隐状态
        z_pred = self.predict(z, action)
        
        # 预测损失（MSE）
        pred_loss = F.mse_loss(z_pred, z_next)
        
        # SIGReg损失（可选）
        if self.use_sigreg and self.sigreg is not None:
            sigreg_loss, sigreg_stats = self.sigreg(z)
            total_loss = pred_loss + self.sigreg_weight * sigreg_loss
            
            stats = {
                'total_loss': total_loss.item(),
                'pred_loss': pred_loss.item(),
                'sigreg_loss': sigreg_loss.item(),
                **sigreg_stats
            }
        else:
            total_loss = pred_loss
            stats = {
                'total_loss': total_loss.item(),
                'pred_loss': pred_loss.item(),
            }
        
        return total_loss, stats
    
    def rollout(
        self,
        initial_observation: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        多步预测（ rollout ）
        
        Args:
            initial_observation: (batch_size, input_dim)初始观测
            actions: (batch_size, horizon, action_dim)动作序列
        
        Returns:
            (batch_size, horizon, latent_dim)预测的隐状态序列
        """
        batch_size, horizon, _ = actions.shape
        
        # 编码初始观测
        z = self.encode(initial_observation)
        
        # 逐步预测
        predictions = []
        for t in range(horizon):
            action_t = actions[:, t, :]
            z = self.predict(z, action_t)
            predictions.append(z)
        
        return torch.stack(predictions, dim=1)


class JEPATrainer:
    """
    JEPA训练器
    
    提供训练循环和优化功能。
    """
    
    def __init__(
        self,
        model: JEPAWorldModel,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        初始化训练器
        
        Args:
            model: JEPA世界模型
            optimizer: 优化器（默认Adam）
            device: 计算设备
        """
        self.model = model.to(device)
        self.device = device
        
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        else:
            self.optimizer = optimizer
        
        self.global_step = 0
        self.epoch = 0
    
    def train_step(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor
    ) -> Dict[str, float]:
        """
        单步训练
        
        Args:
            observation: (batch_size, input_dim)当前观测
            action: (batch_size, action_dim)动作
            next_observation: (batch_size, input_dim)下一观测
        
        Returns:
            统计信息
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 移动到设备
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
    
    def validate(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        验证
        
        Args:
            dataloader: 验证数据加载器
        
        Returns:
            平均统计信息
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                observation, action, next_observation = batch
                
                observation = observation.to(self.device)
                action = action.to(self.device)
                next_observation = next_observation.to(self.device)
                
                loss, _ = self.model.compute_loss(observation, action, next_observation)
                total_loss += loss.item()
                num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches if num_batches > 0 else 0.0
        }
    
    def save_checkpoint(self, path: str) -> None:
        """保存检查点"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
        }, path)
    
    def load_checkpoint(self, path: str) -> None:
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
