"""
好奇心驱动模块 - Curiosity Driven Exploration

实现三种好奇心机制：
1. ICM (Intrinsic Curiosity Module) - 逆动力学模型
2. RND (Random Network Distillation) - 随机网络蒸馏
3. NGU (Never Give Up) - 结合ICM和RND

用于驱动智能体主动探索未知状态空间。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
import math


@dataclass
class CuriosityConfig:
    """好奇心配置"""
    # ICM参数
    icm_feature_dim: int = 256
    icm_hidden_dim: int = 512
    icm_eta: float = 0.01  # 前向/逆动力学损失权重比
    
    # RND参数
    rnd_feature_dim: int = 512
    rnd_hidden_dim: int = 1024
    rnd_learning_rate: float = 1e-4
    
    # NGU参数
    ngu_episode_len: int = 100
    ngu_alpha: float = 0.5  # ICM和RND的混合权重
    
    # 通用参数
    observation_dim: int = 84 * 84  # 假设84x84图像
    action_dim: int = 4
    action_continuous: bool = False
    normalize_features: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class FeatureEncoder(nn.Module):
    """特征编码器"""
    
    def __init__(self, obs_dim: int, feature_dim: int, hidden_dim: int):
        super().__init__()
        
        # 判断是否为图像输入
        self.is_image = obs_dim > 1000
        
        if self.is_image:
            # CNN编码器
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, feature_dim)
            )
        else:
            # MLP编码器
            self.encoder = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, feature_dim)
            )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.is_image and obs.dim() == 3:
            obs = obs.unsqueeze(0)
        if self.is_image and obs.dim() == 4 and obs.shape[-1] == obs.shape[-2]:
            # (B, H, W) -> (B, 1, H, W) -> (B, 3, H, W) for grayscale
            if obs.shape[1] == 1:
                obs = obs.expand(-1, 3, -1, -1)
        return self.encoder(obs)


# ============================================================================
# ICM - Intrinsic Curiosity Module
# ============================================================================

class InverseDynamicsModel(nn.Module):
    """
    逆动力学模型
    
    预测动作：a_t = f(s_t, s_{t+1})
    """
    
    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int, continuous: bool = False):
        super().__init__()
        self.continuous = continuous
        
        self.network = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        if continuous:
            self.action_head = nn.Linear(hidden_dim, action_dim)
        else:
            self.action_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, phi_t: torch.Tensor, phi_t1: torch.Tensor) -> torch.Tensor:
        """预测动作"""
        concat = torch.cat([phi_t, phi_t1], dim=-1)
        features = self.network(concat)
        action_pred = self.action_head(features)
        
        if self.continuous:
            return action_pred  # 连续动作
        else:
            return F.log_softmax(action_pred, dim=-1)  # 离散动作log概率


class ForwardDynamicsModel(nn.Module):
    """
    前向动力学模型
    
    预测下一状态特征：phi_{t+1} = f(phi_t, a_t)
    """
    
    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int, continuous: bool = False):
        super().__init__()
        self.continuous = continuous
        self.feature_dim = feature_dim
        
        if continuous:
            input_dim = feature_dim + action_dim
        else:
            input_dim = feature_dim + action_dim  # one-hot编码
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
    
    def forward(self, phi_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """预测下一状态特征"""
        if not self.continuous:
            # 离散动作转one-hot
            if action.dim() == 1:
                action = F.one_hot(action, num_classes=self.feature_dim).float()
            else:
                action = action.float()
        
        concat = torch.cat([phi_t, action], dim=-1)
        return self.network(concat)


class ICM(nn.Module):
    """
    Intrinsic Curiosity Module
    
    好奇心奖励 = ||phi_{t+1} - phi_hat_{t+1}||^2
    
    同时训练逆动力学模型和前向动力学模型。
    """
    
    def __init__(self, config: CuriosityConfig):
        super().__init__()
        self.config = config
        
        # 特征编码器
        self.encoder = FeatureEncoder(
            config.observation_dim,
            config.icm_feature_dim,
            config.icm_hidden_dim
        )
        
        # 逆动力学模型
        self.inverse_model = InverseDynamicsModel(
            config.icm_feature_dim,
            config.action_dim,
            config.icm_hidden_dim,
            config.action_continuous
        )
        
        # 前向动力学模型
        self.forward_model = ForwardDynamicsModel(
            config.icm_feature_dim,
            config.action_dim,
            config.icm_hidden_dim,
            config.action_continuous
        )
        
        # 归一化统计
        self.register_buffer('running_mean', torch.zeros(config.icm_feature_dim))
        self.register_buffer('running_var', torch.ones(config.icm_feature_dim))
        self.register_buffer('count', torch.tensor(0))
    
    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        """归一化特征"""
        if not self.config.normalize_features:
            return features
        
        mean = self.running_mean.to(features.device)
        var = self.running_var.to(features.device)
        return (features - mean) / torch.sqrt(var + 1e-8)
    
    def update_stats(self, features: torch.Tensor) -> None:
        """更新归一化统计"""
        if not self.config.normalize_features:
            return
        
        batch_mean = features.mean(dim=0)
        batch_var = features.var(dim=0)
        batch_count = features.shape[0]
        
        delta = batch_mean - self.running_mean
        total_count = self.count + batch_count
        
        self.running_mean = self.running_mean + delta * batch_count / total_count
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        self.running_var = M2 / total_count
        self.count = total_count
    
    def forward(
        self,
        obs_t: torch.Tensor,
        obs_t1: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            obs_t: 当前观测
            obs_t1: 下一观测
            action: 执行的动作
            
        Returns:
            intrinsic_reward: 内在好奇心奖励
            inverse_loss: 逆动力学损失
            forward_loss: 前向动力学损失
        """
        # 编码特征
        phi_t = self.encoder(obs_t)
        phi_t1 = self.encoder(obs_t1)
        
        # 更新归一化统计
        if self.training:
            self.update_stats(phi_t1.detach())
        
        # 归一化
        phi_t_norm = self.normalize(phi_t)
        phi_t1_norm = self.normalize(phi_t1)
        
        # 逆动力学预测
        action_pred = self.inverse_model(phi_t_norm, phi_t1_norm)
        
        # 前向动力学预测
        phi_t1_pred = self.forward_model(phi_t_norm, action)
        
        # 前向预测误差作为好奇心奖励
        forward_error = F.mse_loss(phi_t1_pred, phi_t1_norm, reduction='none').sum(dim=-1)
        intrinsic_reward = forward_error.detach()
        
        # 损失
        if self.config.action_continuous:
            inverse_loss = F.mse_loss(action_pred, action)
        else:
            if action.dim() == 1:
                inverse_loss = F.nll_loss(action_pred, action)
            else:
                inverse_loss = F.nll_loss(action_pred, action.argmax(dim=-1))
        
        forward_loss = F.mse_loss(phi_t1_pred, phi_t1_norm)
        
        return intrinsic_reward, inverse_loss, forward_loss
    
    def compute_intrinsic_reward(
        self,
        obs_t: torch.Tensor,
        obs_t1: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """仅计算好奇心奖励"""
        with torch.no_grad():
            reward, _, _ = self.forward(obs_t, obs_t1, action)
        return reward


# ============================================================================
# RND - Random Network Distillation
# ============================================================================

class RandomTargetNetwork(nn.Module):
    """
    随机目标网络
    
    参数随机初始化后固定，不参与训练。
    """
    
    def __init__(self, obs_dim: int, feature_dim: int, hidden_dim: int):
        super().__init__()
        
        self.is_image = obs_dim > 1000
        
        if self.is_image:
            self.network = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, feature_dim)
            )
        else:
            self.network = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, feature_dim)
            )
        
        # 随机初始化后固定
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


class PredictorNetwork(nn.Module):
    """
    预测器网络
    
    学习预测随机目标网络的输出。
    """
    
    def __init__(self, obs_dim: int, feature_dim: int, hidden_dim: int):
        super().__init__()
        
        self.is_image = obs_dim > 1000
        
        if self.is_image:
            self.network = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, feature_dim)
            )
        else:
            self.network = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, feature_dim)
            )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


class RND(nn.Module):
    """
    Random Network Distillation
    
    好奇心奖励 = ||f_target(s) - f_predictor(s)||^2
    
    新颖状态预测误差大，熟悉状态预测误差小。
    """
    
    def __init__(self, config: CuriosityConfig):
        super().__init__()
        self.config = config
        
        # 随机目标网络（固定）
        self.target = RandomTargetNetwork(
            config.observation_dim,
            config.rnd_feature_dim,
            config.rnd_hidden_dim
        )
        
        # 预测器网络（可训练）
        self.predictor = PredictorNetwork(
            config.observation_dim,
            config.rnd_feature_dim,
            config.rnd_hidden_dim
        )
        
        # 归一化统计
        self.register_buffer('running_mean', torch.zeros(config.observation_dim))
        self.register_buffer('running_var', torch.ones(config.observation_dim))
        self.register_buffer('count', torch.tensor(0))
    
    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """归一化观测"""
        if not self.config.normalize_features:
            return obs
        
        if obs.dim() > 2:
            # 图像观测，不归一化
            return obs / 255.0
        
        mean = self.running_mean.to(obs.device)
        var = self.running_var.to(obs.device)
        return (obs - mean) / torch.sqrt(var + 1e-8)
    
    def update_stats(self, obs: torch.Tensor) -> None:
        """更新观测归一化统计"""
        if obs.dim() > 2:
            return  # 图像不更新
        
        batch_mean = obs.mean(dim=0)
        batch_var = obs.var(dim=0)
        batch_count = obs.shape[0]
        
        delta = batch_mean - self.running_mean
        total_count = self.count + batch_count
        
        self.running_mean = self.running_mean + delta * batch_count / total_count
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        self.running_var = M2 / total_count
        self.count = total_count
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            obs: 观测
            
        Returns:
            intrinsic_reward: 内在好奇心奖励
            loss: 预测器损失
        """
        # 归一化
        obs_norm = self.normalize_obs(obs)
        
        # 更新统计
        if self.training:
            self.update_stats(obs.detach())
        
        # 目标网络输出（不计算梯度）
        with torch.no_grad():
            target_output = self.target(obs_norm)
        
        # 预测器输出
        predictor_output = self.predictor(obs_norm)
        
        # 预测误差
        error = F.mse_loss(predictor_output, target_output, reduction='none').sum(dim=-1)
        
        # 好奇心奖励
        intrinsic_reward = error.detach()
        
        # 预测器损失
        loss = F.mse_loss(predictor_output, target_output)
        
        return intrinsic_reward, loss
    
    def compute_intrinsic_reward(self, obs: torch.Tensor) -> torch.Tensor:
        """仅计算好奇心奖励"""
        with torch.no_grad():
            reward, _ = self.forward(obs)
        return reward


# ============================================================================
# NGU - Never Give Up
# ============================================================================

class NGU(nn.Module):
    """
    Never Give Up
    
    结合ICM和RND的好奇心机制：
    r^NGU = r^episodic * r^life-long
    
    其中：
    - r^episodic 来自ICM（短期好奇心）
    - r^life-long 来自RND（长期好奇心）
    """
    
    def __init__(self, config: CuriosityConfig):
        super().__init__()
        self.config = config
        
        # ICM模块
        self.icm = ICM(config)
        
        # RND模块
        self.rnd = RND(config)
        
        # Episode记忆（用于计算episodic奖励）
        self.episode_states: List[torch.Tensor] = []
        self.episode_k = 10  # k近邻数量
        self.episode_epsilon = 1e-3
        self.episode_c = 0.001
        self.episode_alpha = 0.5
    
    def reset_episode(self) -> None:
        """重置episode记忆"""
        self.episode_states.clear()
    
    def compute_episodic_reward(self, state: torch.Tensor) -> torch.Tensor:
        """
        计算episodic奖励
        
        基于当前状态与历史状态的距离。
        """
        if len(self.episode_states) == 0:
            self.episode_states.append(state.detach().cpu())
            return torch.tensor(1.0, device=state.device)
        
        # 计算与历史状态的距离
        history = torch.stack(self.episode_states).to(state.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # L2距离
        distances = torch.norm(history - state, dim=-1)
        
        # 找k近邻
        k = min(self.episode_k, len(self.episode_states))
        knn_distances, _ = torch.topk(distances, k, largest=False)
        
        # 计算episodic奖励
        # r = 1 / (sum of distances to k-NN + epsilon)
        episodic_reward = 1.0 / (knn_distances.sum() + self.episode_epsilon)
        
        # 限制范围
        episodic_reward = min(episodic_reward.item(), 1.0 / self.episode_epsilon)
        
        # 添加到记忆
        self.episode_states.append(state.detach().cpu())
        
        return torch.tensor(episodic_reward, device=state.device)
    
    def forward(
        self,
        obs_t: torch.Tensor,
        obs_t1: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Returns:
            intrinsic_reward: NGU好奇心奖励
            losses: 各模块损失字典
        """
        # ICM奖励（episodic）
        icm_reward, inverse_loss, forward_loss = self.icm(obs_t, obs_t1, action)
        
        # RND奖励（life-long）
        rnd_reward, rnd_loss = self.rnd(obs_t1)
        
        # 计算episodic奖励
        if obs_t1.dim() > 2:
            # 图像观测，使用编码后的特征
            state_feature = self.icm.encoder(obs_t1)
        else:
            state_feature = obs_t1
        
        episodic_reward = self.compute_episodic_reward(state_feature.mean(dim=0) if state_feature.dim() > 1 else state_feature)
        
        # NGU奖励 = episodic * RND
        ngu_reward = episodic_reward * (1 + rnd_reward)
        
        # 混合ICM和RND
        intrinsic_reward = self.config.ngu_alpha * icm_reward + (1 - self.config.ngu_alpha) * ngu_reward
        
        losses = {
            'inverse_loss': inverse_loss,
            'forward_loss': forward_loss,
            'rnd_loss': rnd_loss
        }
        
        return intrinsic_reward, losses
    
    def compute_intrinsic_reward(
        self,
        obs_t: torch.Tensor,
        obs_t1: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """仅计算好奇心奖励"""
        with torch.no_grad():
            reward, _ = self.forward(obs_t, obs_t1, action)
        return reward


# ============================================================================
# 工厂函数
# ============================================================================

def create_curiosity_module(
    module_type: str = "icm",
    config: Optional[CuriosityConfig] = None
) -> nn.Module:
    """
    创建好奇心模块
    
    Args:
        module_type: "icm", "rnd", 或 "ngu"
        config: 配置
        
    Returns:
        好奇心模块
    """
    if config is None:
        config = CuriosityConfig()
    
    module_type = module_type.lower()
    
    if module_type == "icm":
        return ICM(config)
    elif module_type == "rnd":
        return RND(config)
    elif module_type == "ngu":
        return NGU(config)
    else:
        raise ValueError(f"Unknown curiosity module type: {module_type}")
