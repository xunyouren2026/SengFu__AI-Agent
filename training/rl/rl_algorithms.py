"""
AGI统一框架 - 强化学习核心算法
实现PPO、SAC、TD3等主流强化学习算法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
import math
from collections import deque
import random
from abc import ABC, abstractmethod
import copy


# ==================== 配置类 ====================

@dataclass
class RLConfig:
    """强化学习通用配置"""
    # 环境配置
    state_dim: int = 64
    action_dim: int = 4
    action_type: str = "continuous"  # continuous, discrete
    
    # 网络配置
    hidden_dim: int = 256
    num_layers: int = 3
    
    # 训练配置
    batch_size: int = 256
    buffer_size: int = 1000000
    learning_rate: float = 3e-4
    gamma: float = 0.99  # 折扣因子
    tau: float = 0.005  # 软更新系数
    
    # 探索配置
    start_epsilon: float = 1.0
    end_epsilon: float = 0.01
    epsilon_decay: float = 0.995


@dataclass
class PPOConfig(RLConfig):
    """PPO配置"""
    # PPO特定参数
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    gae_lambda: float = 0.95
    num_epochs: int = 10
    num_minibatches: int = 32
    normalize_advantage: bool = True


@dataclass
class SACConfig(RLConfig):
    """SAC配置"""
    # SAC特定参数
    alpha: float = 0.2
    auto_entropy: bool = True
    target_entropy: Optional[float] = None
    alpha_lr: float = 3e-4
    policy_delay: int = 1  # 策略更新延迟


@dataclass
class TD3Config(RLConfig):
    """TD3配置"""
    # TD3特定参数
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2  # 策略更新延迟


# ==================== 网络架构 ====================

class ActorNetwork(nn.Module):
    """策略网络（演员）"""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_dim: int = 256, num_layers: int = 3,
                 action_bound: float = 1.0):
        super().__init__()
        
        self.action_dim = action_dim
        self.action_bound = action_bound
        
        # 构建网络
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        self.trunk = nn.Sequential(*layers)
        
        # 输出层
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播，返回均值和对数标准差"""
        features = self.trunk(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        
        # 限制log_std范围
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mean, log_std
    
    def get_action(self, state: torch.Tensor, 
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取动作"""
        mean, log_std = self.forward(state)
        
        if deterministic:
            action = mean
            log_prob = None
        else:
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            action = normal.rsample()  # 重参数化采样
            log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
            
            # 矫正tanh变换后的概率
            action = torch.tanh(action)
            log_prob = log_prob - torch.log(
                1 - action.pow(2) + 1e-6
            ).sum(dim=-1, keepdim=True)
        
        return action * self.action_bound, log_prob


class CriticNetwork(nn.Module):
    """价值网络（评论家）"""
    
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        
        # 构建网络
        layers = []
        layers.append(nn.Linear(state_dim + action_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        self.trunk = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = torch.cat([state, action], dim=-1)
        features = self.trunk(x)
        return self.output_layer(features)


class ValueNetwork(nn.Module):
    """状态价值网络"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        self.trunk = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.output_layer(self.trunk(state))


# ==================== 经验回放 ====================

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 1000000):
        self.capacity = capacity
        self.buffer: List[Dict] = []
        self.pos = 0
        
    def push(self, state: np.ndarray, action: np.ndarray, reward: float,
             next_state: np.ndarray, done: bool):
        """添加经验"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
            
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """采样"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        return {
            'states': torch.tensor(np.stack([e['state'] for e in batch]), dtype=torch.float32),
            'actions': torch.tensor(np.stack([e['action'] for e in batch]), dtype=torch.float32),
            'rewards': torch.tensor([e['reward'] for e in batch], dtype=torch.float32),
            'next_states': torch.tensor(np.stack([e['next_state'] for e in batch]), dtype=torch.float32),
            'dones': torch.tensor([e['done'] for e in batch], dtype=torch.float32)
        }
    
    def __len__(self) -> int:
        return len(self.buffer)


class RolloutBuffer:
    """轨迹缓冲区（用于PPO）"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[Dict] = []
        
    def push(self, state: np.ndarray, action: np.ndarray, reward: float,
             next_state: np.ndarray, done: bool, 
             log_prob: float, value: float):
        """添加经验"""
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob,
            'value': value
        })
        
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
            
    def compute_gae(self, gamma: float, gae_lambda: float,
                    last_value: float) -> None:
        """计算广义优势估计"""
        advantages = []
        returns = []
        
        gae = 0.0
        
        for i in reversed(range(len(self.buffer))):
            if i == len(self.buffer) - 1:
                next_value = last_value
            else:
                next_value = self.buffer[i + 1]['value']
            
            delta = (self.buffer[i]['reward'] + 
                    gamma * next_value * (1 - self.buffer[i]['done']) -
                    self.buffer[i]['value'])
            
            gae = delta + gamma * gae_lambda * (1 - self.buffer[i]['done']) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + self.buffer[i]['value'])
        
        for i, (adv, ret) in enumerate(zip(advantages, returns)):
            self.buffer[i]['advantage'] = adv
            self.buffer[i]['return'] = ret
            
    def get(self) -> Dict[str, torch.Tensor]:
        """获取所有数据"""
        return {
            'states': torch.tensor(np.stack([e['state'] for e in self.buffer]), dtype=torch.float32),
            'actions': torch.tensor(np.stack([e['action'] for e in self.buffer]), dtype=torch.float32),
            'log_probs': torch.tensor([e['log_prob'] for e in self.buffer], dtype=torch.float32),
            'returns': torch.tensor([e['return'] for e in self.buffer], dtype=torch.float32),
            'advantages': torch.tensor([e['advantage'] for e in self.buffer], dtype=torch.float32)
        }
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
        
    def __len__(self) -> int:
        return len(self.buffer)


# ==================== PPO算法 ====================

class PPO:
    """Proximal Policy Optimization"""
    
    def __init__(self, config: Optional[PPOConfig] = None, device: str = 'cpu'):
        self.config = config or PPOConfig()
        self.device = device
        
        # 网络
        self.actor = ActorNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim,
            self.config.num_layers
        ).to(device)
        
        self.critic = ValueNetwork(
            self.config.state_dim,
            self.config.hidden_dim,
            self.config.num_layers
        ).to(device)
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.learning_rate
        )
        
        # 缓冲区
        self.rollout_buffer = RolloutBuffer()
        
        # 统计
        self.update_count = 0
        
    def select_action(self, state: np.ndarray, 
                      deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            action, log_prob = self.actor.get_action(state_tensor, deterministic)
            value = self.critic(state_tensor)
            
        return action.squeeze(0).cpu().numpy(), log_prob.item(), value.item()
    
    def store_transition(self, state: np.ndarray, action: np.ndarray,
                         reward: float, next_state: np.ndarray, done: bool,
                         log_prob: float, value: float):
        """存储转移"""
        self.rollout_buffer.push(state, action, reward, next_state, done, log_prob, value)
        
    def update(self) -> Dict[str, float]:
        """更新网络"""
        if len(self.rollout_buffer) < self.config.batch_size:
            return {'loss': 0.0}
        
        # 获取数据
        data = self.rollout_buffer.get()
        
        states = data['states'].to(self.device)
        actions = data['actions'].to(self.device)
        old_log_probs = data['log_probs'].to(self.device)
        returns = data['returns'].to(self.device)
        advantages = data['advantages'].to(self.device)
        
        # 标准化优势
        if self.config.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 多轮更新
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
        dataset_size = len(states)
        minibatch_size = max(dataset_size // self.config.num_minibatches, 1)
        
        for _ in range(self.config.num_epochs):
            # 随机打乱
            indices = torch.randperm(dataset_size)
            
            for start in range(0, dataset_size, minibatch_size):
                end = min(start + minibatch_size, dataset_size)
                mb_indices = indices[start:end]
                
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]
                
                # 计算新log_prob
                mean, log_std = self.actor(mb_states)
                std = log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                
                # 熵
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # 策略比率
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
                # PPO裁剪目标
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 
                                   1 + self.config.clip_ratio) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                values = self.critic(mb_states).squeeze()
                value_loss = F.mse_loss(values, mb_returns)
                
                # 总损失
                loss = (policy_loss + 
                       self.config.value_coef * value_loss -
                       self.config.entropy_coef * entropy)
                
                # 反向传播
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.config.max_grad_norm
                )
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.config.max_grad_norm
                )
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        self.update_count += 1
        self.rollout_buffer.clear()
        
        num_updates = self.config.num_epochs * self.config.num_minibatches
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }


# ==================== SAC算法 ====================

class SAC:
    """Soft Actor-Critic"""
    
    def __init__(self, config: Optional[SACConfig] = None, device: str = 'cpu'):
        self.config = config or SACConfig()
        self.device = device
        
        # 网络
        self.actor = ActorNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim,
            self.config.num_layers
        ).to(device)
        
        # 双Q网络
        self.critic1 = CriticNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim,
            self.config.num_layers
        ).to(device)
        
        self.critic2 = CriticNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim,
            self.config.num_layers
        ).to(device)
        
        # 目标网络
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.learning_rate
        )
        self.critic1_optimizer = torch.optim.Adam(
            self.critic1.parameters(), lr=self.config.learning_rate
        )
        self.critic2_optimizer = torch.optim.Adam(
            self.critic2.parameters(), lr=self.config.learning_rate
        )
        
        # 熵系数
        if self.config.auto_entropy:
            if self.config.target_entropy is None:
                self.target_entropy = -self.config.action_dim
            else:
                self.target_entropy = self.config.target_entropy
            
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=self.config.alpha_lr
            )
        else:
            self.log_alpha = torch.log(torch.tensor(self.config.alpha))
            
        # 缓冲区
        self.replay_buffer = ReplayBuffer(self.config.buffer_size)
        
        # 统计
        self.update_count = 0
        
    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()
    
    def select_action(self, state: np.ndarray, 
                      deterministic: bool = False) -> np.ndarray:
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, 
                                       device=self.device).unsqueeze(0)
            action, _ = self.actor.get_action(state_tensor, deterministic)
        return action.squeeze(0).cpu().numpy()
    
    def store_transition(self, state: np.ndarray, action: np.ndarray,
                         reward: float, next_state: np.ndarray, done: bool):
        """存储转移"""
        self.replay_buffer.push(state, action, reward, next_state, done)
        
    def update(self) -> Dict[str, float]:
        """更新网络"""
        if len(self.replay_buffer) < self.config.batch_size:
            return {'loss': 0.0}
        
        # 采样
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # 计算目标Q值
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.get_action(next_states)
            
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            
            target_value = rewards.unsqueeze(1) + \
                          self.config.gamma * (1 - dones.unsqueeze(1)) * target_q
        
        # 更新Q网络
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        
        q1_loss = F.mse_loss(q1, target_value)
        q2_loss = F.mse_loss(q2, target_value)
        
        self.critic1_optimizer.zero_grad()
        q1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        q2_loss.backward()
        self.critic2_optimizer.step()
        
        # 更新策略网络
        new_actions, log_probs = self.actor.get_action(states)
        
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha.detach() * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        # 更新熵系数
        if self.config.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_probs.detach() + 
                         self.target_entropy)).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # 软更新目标网络
        self._soft_update(self.target_critic1, self.critic1)
        self._soft_update(self.target_critic2, self.critic2)
        
        self.update_count += 1
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha': self.alpha.item()
        }
    
    def _soft_update(self, target: nn.Module, source: nn.Module):
        """软更新"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )


# ==================== TD3算法 ====================

class TD3:
    """Twin Delayed DDPG"""
    
    def __init__(self, config: Optional[TD3Config] = None, device: str = 'cpu'):
        self.config = config or TD3Config()
        self.device = device
        
        # 网络
        self.actor = ActorNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim,
            self.config.num_layers
        ).to(device)
        
        self.critic1 = CriticNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim,
            self.config.num_layers
        ).to(device)
        
        self.critic2 = CriticNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim,
            self.config.num_layers
        ).to(device)
        
        # 目标网络
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.learning_rate
        )
        self.critic1_optimizer = torch.optim.Adam(
            self.critic1.parameters(), lr=self.config.learning_rate
        )
        self.critic2_optimizer = torch.optim.Adam(
            self.critic2.parameters(), lr=self.config.learning_rate
        )
        
        # 缓冲区
        self.replay_buffer = ReplayBuffer(self.config.buffer_size)
        
        # 统计
        self.update_count = 0
        
    def select_action(self, state: np.ndarray, 
                      noise_scale: float = 0.1) -> np.ndarray:
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32,
                                       device=self.device).unsqueeze(0)
            action, _ = self.actor.get_action(state_tensor, deterministic=True)
            
            # 添加探索噪声
            if noise_scale > 0:
                noise = torch.randn_like(action) * noise_scale
                action = torch.clamp(action + noise, -1, 1)
                
        return action.squeeze(0).cpu().numpy()
    
    def store_transition(self, state: np.ndarray, action: np.ndarray,
                         reward: float, next_state: np.ndarray, done: bool):
        """存储转移"""
        self.replay_buffer.push(state, action, reward, next_state, done)
        
    def update(self) -> Dict[str, float]:
        """更新网络"""
        if len(self.replay_buffer) < self.config.batch_size:
            return {'loss': 0.0}
        
        # 采样
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # 计算目标Q值（带目标策略平滑）
        with torch.no_grad():
            # 添加噪声
            noise = (torch.randn_like(actions) * self.config.policy_noise
                    ).clamp(-self.config.noise_clip, self.config.noise_clip)
            
            next_actions, _ = self.target_actor.get_action(next_states, deterministic=True)
            next_actions = (next_actions + noise).clamp(-1, 1)
            
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            target_value = rewards.unsqueeze(1) + \
                          self.config.gamma * (1 - dones.unsqueeze(1)) * target_q
        
        # 更新Q网络
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        
        q1_loss = F.mse_loss(q1, target_value)
        q2_loss = F.mse_loss(q2, target_value)
        
        self.critic1_optimizer.zero_grad()
        q1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        q2_loss.backward()
        self.critic2_optimizer.step()
        
        # 延迟更新策略网络
        policy_loss_val = 0.0
        if self.update_count % self.config.policy_delay == 0:
            new_actions, _ = self.actor.get_action(states, deterministic=True)
            q1_new = self.critic1(states, new_actions)
            
            policy_loss = -q1_new.mean()
            
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()
            
            policy_loss_val = policy_loss.item()
            
            # 软更新目标网络
            self._soft_update(self.target_actor, self.actor)
            self._soft_update(self.target_critic1, self.critic1)
            self._soft_update(self.target_critic2, self.critic2)
        
        self.update_count += 1
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss_val
        }
    
    def _soft_update(self, target: nn.Module, source: nn.Module):
        """软更新"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )


# ==================== 统一接口 ====================

class RLAgent:
    """强化学习代理统一接口"""
    
    def __init__(self, algorithm: str = "ppo",
                 config: Optional[RLConfig] = None,
                 device: str = 'cpu'):
        self.algorithm = algorithm.lower()
        self.device = device
        
        if self.algorithm == "ppo":
            self.agent = PPO(config or PPOConfig(), device)
        elif self.algorithm == "sac":
            self.agent = SAC(config or SACConfig(), device)
        elif self.algorithm == "td3":
            self.agent = TD3(config or TD3Config(), device)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
            
    def select_action(self, state: np.ndarray, 
                      deterministic: bool = False) -> np.ndarray:
        """选择动作"""
        if self.algorithm == "ppo":
            action, _, _ = self.agent.select_action(state, deterministic)
            return action
        elif self.algorithm == "sac":
            return self.agent.select_action(state, deterministic)
        elif self.algorithm == "td3":
            noise_scale = 0.0 if deterministic else 0.1
            return self.agent.select_action(state, noise_scale)
            
    def store_transition(self, state: np.ndarray, action: np.ndarray,
                         reward: float, next_state: np.ndarray, done: bool,
                         **kwargs):
        """存储转移"""
        if self.algorithm == "ppo":
            self.agent.store_transition(
                state, action, reward, next_state, done,
                kwargs.get('log_prob', 0.0),
                kwargs.get('value', 0.0)
            )
        else:
            self.agent.store_transition(state, action, reward, next_state, done)
            
    def update(self) -> Dict[str, float]:
        """更新网络"""
        return self.agent.update()
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor': self.agent.actor.state_dict(),
            'critic': self.agent.critic.state_dict() if hasattr(self.agent, 'critic') else None,
            'config': self.agent.config
        }, path)
        
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.actor.load_state_dict(checkpoint['actor'])
        if checkpoint['critic'] is not None and hasattr(self.agent, 'critic'):
            self.agent.critic.load_state_dict(checkpoint['critic'])


# ==================== 训练循环 ====================

def train_rl_agent(env, agent: RLAgent, num_episodes: int = 1000,
                   max_steps: int = 1000, callback: Optional[Callable] = None) -> Dict[str, List]:
    """训练强化学习代理"""
    history = {'reward': [], 'length': [], 'loss': []}
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        for step in range(max_steps):
            # 选择动作
            if agent.algorithm == "ppo":
                action, log_prob, value = agent.agent.select_action(state)
                extra = {'log_prob': log_prob, 'value': value}
            else:
                action = agent.select_action(state)
                extra = {}
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储转移
            agent.store_transition(state, action, reward, next_state, done, **extra)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        # 更新网络
        loss_info = agent.update()
        
        # 记录
        history['reward'].append(episode_reward)
        history['length'].append(episode_length)
        history['loss'].append(loss_info.get('loss', 0))
        
        if callback:
            callback(episode, {
                'reward': episode_reward,
                'length': episode_length,
                **loss_info
            })
    
    return history
