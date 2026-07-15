"""
高级强化学习算法 - Advanced RL Algorithms
实现A3C、IMPALA、Rainbow DQN、R2D2等算法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, RMSprop
import numpy as np
import math
import random
import threading
import multiprocessing as mp
from multiprocessing import Process, Queue, Pipe
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
import time
import warnings

# ==================== 经验回放 ====================

class PrioritizedReplayBuffer:
    """优先级经验回放"""
    
    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0
    
    def push(self, experience: Tuple, priority: float = None):
        """添加经验"""
        if priority is None:
            priority = self.max_priority
        
        priority = priority ** self.alpha
        
        if self.size < self.capacity:
            self.buffer.append(experience)
            self.priorities[self.size] = priority
            self.size += 1
        else:
            self.buffer[self.pos] = experience
            self.priorities[self.pos] = priority
        
        self.pos = (self.pos + 1) % self.capacity
        self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """采样"""
        if self.size < batch_size:
            return [], np.array([]), np.array([])
        
        # 计算采样概率
        priorities = self.priorities[:self.size]
        probs = priorities / priorities.sum()
        
        # 采样索引
        indices = np.random.choice(self.size, batch_size, replace=False, p=probs)
        
        # 计算重要性权重
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        experiences = [self.buffer[i] for i in indices]
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """更新优先级"""
        priorities = priorities ** self.alpha
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())
    
    def __len__(self):
        return self.size


class NStepReplayBuffer:
    """N步经验回放"""
    
    def __init__(
        self,
        capacity: int = 100000,
        n_step: int = 3,
        gamma: float = 0.99,
    ):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        
        self.buffer = deque(maxlen=capacity)
        self.n_step_buffer = deque(maxlen=n_step)
    
    def push(self, state, action, reward, next_state, done):
        """添加经验"""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) == self.n_step:
            # 计算N步回报
            state, action = self.n_step_buffer[0][:2]
            reward = 0.0
            next_state = None
            done = False
            
            for i, (_, _, r, ns, d) in enumerate(self.n_step_buffer):
                reward += (self.gamma ** i) * r
                if d:
                    done = True
                    break
            
            if not done:
                next_state = self.n_step_buffer[-1][3]
            
            self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List:
        """采样"""
        if len(self.buffer) < batch_size:
            return []
        return random.sample(list(self.buffer), batch_size)
    
    def __len__(self):
        return len(self.buffer)


class EpisodeReplayBuffer:
    """回合经验回放 - 用于R2D2"""
    
    def __init__(
        self,
        capacity: int = 1000,
        sequence_length: int = 80,
        burn_in_length: int = 40,
    ):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.burn_in_length = burn_in_length
        
        self.episodes = deque(maxlen=capacity)
        self.total_steps = 0
    
    def add_episode(self, episode: List[Tuple]):
        """添加回合"""
        self.episodes.append(episode)
        self.total_steps += len(episode)
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """采样序列"""
        if len(self.episodes) < batch_size:
            return []
        
        sequences = []
        for _ in range(batch_size):
            episode = random.choice(self.episodes)
            
            if len(episode) < self.sequence_length + self.burn_in_length:
                sequences.append(episode)
            else:
                # 随机选择起始位置
                start = random.randint(
                    0, len(episode) - self.sequence_length - self.burn_in_length
                )
                sequence = episode[start:start + self.sequence_length + self.burn_in_length]
                sequences.append(sequence)
        
        return sequences
    
    def __len__(self):
        return len(self.episodes)


# ==================== 分布式价值估计 ====================

class CategoricalValueNet(nn.Module):
    """分类价值网络 - 用于C51"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        self.register_buffer('support', torch.linspace(v_min, v_max, num_atoms))
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * num_atoms),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """返回每个动作的概率分布"""
        batch_size = state.size(0)
        logits = self.net(state).view(batch_size, -1, self.num_atoms)
        probs = F.softmax(logits, dim=-1)
        return probs
    
    def get_q_values(self, probs: torch.Tensor) -> torch.Tensor:
        """计算Q值"""
        return (probs * self.support).sum(dim=-1)


class QuantileValueNet(nn.Module):
    """分位数值网络 - 用于QR-DQN"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_quantiles: int = 32,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_quantiles = num_quantiles
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * num_quantiles),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """返回每个动作的分位数"""
        batch_size = state.size(0)
        quantiles = self.net(state).view(batch_size, -1, self.num_quantiles)
        return quantiles
    
    def get_q_values(self, quantiles: torch.Tensor) -> torch.Tensor:
        """计算Q值（分位数均值）"""
        return quantiles.mean(dim=-1)


# ==================== Rainbow DQN ====================

class RainbowDQN(nn.Module):
    """Rainbow DQN - 集成多种改进"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        hidden_dim: int = 256,
        noisy_std: float = 0.5,
        use_dueling: bool = True,
        use_noisy: bool = True,
        use_distributional: bool = True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.use_dueling = use_dueling
        self.use_noisy = use_noisy
        self.use_distributional = use_distributional
        
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.register_buffer('support', torch.linspace(v_min, v_max, num_atoms))
        
        # 特征提取
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Noisy线性层
        if use_noisy:
            self.noisy_value = NoisyLinear(hidden_dim, hidden_dim, noisy_std)
            self.noisy_advantage = NoisyLinear(hidden_dim, hidden_dim, noisy_std)
        else:
            self.value = nn.Linear(hidden_dim, hidden_dim)
            self.advantage = nn.Linear(hidden_dim, hidden_dim)
        
        # 输出层
        if use_distributional:
            if use_dueling:
                self.value_out = nn.Linear(hidden_dim, num_atoms)
                self.advantage_out = nn.Linear(hidden_dim, action_dim * num_atoms)
            else:
                self.out = nn.Linear(hidden_dim, action_dim * num_atoms)
        else:
            if use_dueling:
                self.value_out = nn.Linear(hidden_dim, 1)
                self.advantage_out = nn.Linear(hidden_dim, action_dim)
            else:
                self.out = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        features = self.feature(state)
        
        if self.use_noisy:
            value = F.relu(self.noisy_value(features))
            advantage = F.relu(self.noisy_advantage(features))
        else:
            value = F.relu(self.value(features))
            advantage = F.relu(self.advantage(features))
        
        if self.use_distributional:
            if self.use_dueling:
                value_out = self.value_out(value).unsqueeze(-1)  # [B, num_atoms, 1]
                advantage_out = self.advantage_out(advantage).view(-1, self.action_dim, self.num_atoms)
                advantage_mean = advantage_out.mean(dim=1, keepdim=True)
                logits = value_out + advantage_out - advantage_mean
            else:
                logits = self.out(advantage).view(-1, self.action_dim, self.num_atoms)
            
            probs = F.softmax(logits, dim=-1)
            return probs
        else:
            if self.use_dueling:
                value_out = self.value_out(value)
                advantage_out = self.advantage_out(advantage)
                advantage_mean = advantage_out.mean(dim=1, keepdim=True)
                q_values = value_out + advantage_out - advantage_mean
            else:
                q_values = self.out(advantage)
            
            return q_values
    
    def get_q_values(self, output: torch.Tensor) -> torch.Tensor:
        """计算Q值"""
        if self.use_distributional:
            return (output * self.support).sum(dim=-1)
        else:
            return output
    
    def reset_noise(self):
        """重置噪声"""
        if self.use_noisy:
            self.noisy_value.reset_noise()
            self.noisy_advantage.reset_noise()


class NoisyLinear(nn.Module):
    """噪声线性层"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        std_init: float = 0.5,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_sigma = nn.Parameter(torch.zeros(out_features))
        
        self.register_buffer('weight_epsilon', torch.zeros(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.zeros(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """重置参数"""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    
    def reset_noise(self):
        """重置噪声"""
        epsilon_in = self._scale_noise(torch.randn(self.in_features))
        epsilon_out = self._scale_noise(torch.randn(self.out_features))
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, x: torch.Tensor) -> torch.Tensor:
        """缩放噪声"""
        return x.sign() * x.abs().sqrt()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)


class RainbowAgent:
    """Rainbow DQN智能体"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        batch_size: int = 32,
        target_update_freq: int = 1000,
        n_step: int = 3,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        device: str = "cuda",
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.n_step = n_step
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.device = device
        
        # 网络
        self.net = RainbowDQN(
            state_dim, action_dim, num_atoms, v_min, v_max,
        ).to(device)
        self.target_net = RainbowDQN(
            state_dim, action_dim, num_atoms, v_min, v_max,
        ).to(device)
        self.target_net.load_state_dict(self.net.state_dict())
        
        # 优化器
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        
        # 经验回放
        self.replay_buffer = NStepReplayBuffer(
            capacity=100000, n_step=n_step, gamma=gamma
        )
        self.prioritized_buffer = PrioritizedReplayBuffer(capacity=100000)
        
        self.update_count = 0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """选择动作"""
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.net(state)
            q_values = self.net.get_q_values(probs)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def update(self) -> Optional[float]:
        """更新网络"""
        if len(self.prioritized_buffer) < self.batch_size:
            return None
        
        # 采样
        experiences, indices, weights = self.prioritized_buffer.sample(self.batch_size)
        weights = torch.FloatTensor(weights).to(self.device)
        
        states = torch.FloatTensor(np.array([e[0] for e in experiences])).to(self.device)
        actions = torch.LongTensor([e[1] for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in experiences]).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in experiences])).to(self.device)
        dones = torch.FloatTensor([e[4] for e in experiences]).to(self.device)
        
        # 计算当前分布
        probs = self.net(states)
        actions = actions.unsqueeze(1).unsqueeze(1).expand(-1, 1, self.num_atoms)
        current_probs = probs.gather(1, actions).squeeze(1)
        
        # 计算目标分布
        with torch.no_grad():
            next_probs = self.target_net(next_states)
            next_q_values = self.target_net.get_q_values(next_probs)
            next_actions = next_q_values.argmax(dim=1)
            
            next_actions = next_actions.unsqueeze(1).unsqueeze(1).expand(-1, 1, self.num_atoms)
            next_probs = next_probs.gather(1, next_actions).squeeze(1)
            
            # 投影到支持上
            support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)
            n_step_gamma = self.gamma ** self.n_step
            target_support = rewards.unsqueeze(1) + n_step_gamma * support.unsqueeze(0) * (1 - dones.unsqueeze(1))
            
            target_probs = self._project_distribution(target_support, next_probs)
        
        # 计算损失
        loss = -(target_probs * torch.log(current_probs + 1e-8)).sum(dim=1)
        loss = (loss * weights).mean()
        
        # 更新优先级
        priorities = loss.detach().cpu().numpy() + 1e-6
        self.prioritized_buffer.update_priorities(indices, priorities)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 重置噪声
        self.net.reset_noise()
        self.target_net.reset_noise()
        
        # 更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        
        # 衰减epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def _project_distribution(
        self,
        target_support: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        """投影分布到支持上"""
        batch_size = target_support.size(0)
        
        # 裁剪到[v_min, v_max]
        target_support = target_support.clamp(self.v_min, self.v_max)
        
        # 计算投影
        support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)
        
        # 下界和上界索引
        b = (target_support - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()
        
        # 处理边界情况
        l[(u > 0) * (l == u)] -= 1
        u[(l < self.num_atoms - 1) * (l == u)] += 1
        
        # 分配概率
        projected_probs = torch.zeros(batch_size, self.num_atoms).to(self.device)
        
        offset = torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size).long().to(self.device)
        
        projected_probs.view(-1).index_add_(
            0, (l + offset).view(-1), (probs * (u.float() - b)).view(-1)
        )
        projected_probs.view(-1).index_add_(
            0, (u + offset).view(-1), (probs * (b - l.float())).view(-1)
        )
        
        return projected_probs


# ==================== A3C ====================

class A3CNetwork(nn.Module):
    """A3C网络"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        # 共享特征提取
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 策略头
        self.policy = nn.Linear(hidden_dim, action_dim)
        
        # 价值头
        self.value = nn.Linear(hidden_dim, 1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        features = self.feature(state)
        policy_logits = self.policy(features)
        value = self.value(features)
        return policy_logits, value


class A3CWorker:
    """A3C工作进程"""
    
    def __init__(
        self,
        worker_id: int,
        env_fn: Callable,
        net: A3CNetwork,
        gamma: float = 0.99,
        t_max: int = 20,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 40.0,
        device: str = "cpu",
    ):
        self.worker_id = worker_id
        self.env = env_fn()
        self.net = net
        self.gamma = gamma
        self.t_max = t_max
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        self.state = self.env.reset()
        self.total_steps = 0
    
    def select_action(self, state: np.ndarray) -> int:
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, _ = self.net(state)
            probs = F.softmax(policy_logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        return action
    
    def collect_experience(self) -> Tuple[List, float, np.ndarray]:
        """收集经验"""
        experiences = []
        total_reward = 0.0
        
        for _ in range(self.t_max):
            action = self.select_action(self.state)
            next_state, reward, done, info = self.env.step(action)
            
            experiences.append((self.state, action, reward, next_state, done))
            total_reward += reward
            self.total_steps += 1
            
            if done:
                self.state = self.env.reset()
                break
            else:
                self.state = next_state
        
        # 计算R（bootstrap value）
        if not experiences[-1][4]:  # 如果不是终止状态
            state = torch.FloatTensor(experiences[-1][3]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, value = self.net(state)
                R = value.item()
        else:
            R = 0.0
        
        return experiences, total_reward, R
    
    def compute_loss(
        self,
        experiences: List,
        R: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算损失"""
        states = torch.FloatTensor(np.array([e[0] for e in experiences])).to(self.device)
        actions = torch.LongTensor([e[1] for e in experiences]).to(self.device)
        rewards = [e[2] for e in experiences]
        dones = [e[4] for e in experiences]
        
        policy_logits, values = self.net(states)
        
        # 计算returns和advantages
        returns = []
        advantages = []
        
        for i in reversed(range(len(experiences))):
            R = rewards[i] + self.gamma * R * (1 - dones[i])
            returns.insert(0, R)
            advantage = R - values[i].item()
            advantages.insert(0, advantage)
        
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # 策略损失
        log_probs = F.log_softmax(policy_logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        policy_loss = -(action_log_probs * advantages).mean()
        
        # 价值损失
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # 熵奖励
        probs = F.softmax(policy_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        
        # 总损失
        total_loss = (
            policy_loss +
            self.value_loss_coef * value_loss -
            self.entropy_coef * entropy
        )
        
        metrics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
        }
        
        return total_loss, metrics


class A3CAgent:
    """A3C智能体"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        env_fn: Callable,
        num_workers: int = 4,
        lr: float = 1e-4,
        gamma: float = 0.99,
        t_max: int = 20,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 40.0,
        device: str = "cpu",
    ):
        self.num_workers = num_workers
        self.device = device
        
        # 共享网络
        self.net = A3CNetwork(state_dim, action_dim).to(device)
        self.net.share_memory()
        
        # 优化器
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        
        # 创建工作进程
        self.workers = [
            A3CWorker(
                i, env_fn, self.net, gamma, t_max,
                entropy_coef, value_loss_coef, max_grad_norm, device
            )
            for i in range(num_workers)
        ]
        
        self.running = False
    
    def train(self, num_steps: int, callback: Optional[Callable] = None):
        """训练"""
        self.running = True
        total_steps = 0
        
        while total_steps < num_steps and self.running:
            # 收集所有worker的经验
            all_losses = []
            all_metrics = defaultdict(float)
            
            for worker in self.workers:
                experiences, reward, R = worker.collect_experience()
                loss, metrics = worker.compute_loss(experiences, R)
                all_losses.append(loss)
                
                for k, v in metrics.items():
                    all_metrics[k] += v
                
                total_steps += len(experiences)
            
            # 平均损失
            total_loss = torch.stack(all_losses).mean()
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.workers[0].max_grad_norm)
            self.optimizer.step()
            
            # 回调
            if callback:
                callback(total_steps, all_metrics)
    
    def stop(self):
        """停止训练"""
        self.running = False
    
    def select_action(self, state: np.ndarray) -> int:
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, _ = self.net(state)
            action = policy_logits.argmax(dim=1).item()
        return action


# ==================== IMPALA ====================

class IMPALAActor(nn.Module):
    """IMPALA Actor网络"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """返回动作logits"""
        return self.net(state)


class IMPALALearner(nn.Module):
    """IMPALA Learner网络"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        # 共享特征
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 策略头
        self.policy = nn.Linear(hidden_dim, action_dim)
        
        # 价值头
        self.value = nn.Linear(hidden_dim, 1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        features = self.feature(state)
        policy_logits = self.policy(features)
        value = self.value(features)
        return policy_logits, value


class IMPALAAgent:
    """IMPALA智能体"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        env_fn: Callable,
        num_actors: int = 4,
        lr: float = 1e-4,
        gamma: float = 0.99,
        rho: float = 1.0,
        c: float = 1.0,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        batch_size: int = 32,
        device: str = "cuda",
    ):
        self.num_actors = num_actors
        self.gamma = gamma
        self.rho = rho
        self.c = c
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.batch_size = batch_size
        self.device = device
        
        # Actor网络（每个actor一个）
        self.actors = [IMPALAActor(state_dim, action_dim).to(device) for _ in range(num_actors)]
        
        # Learner网络
        self.learner = IMPALALearner(state_dim, action_dim).to(device)
        self.optimizer = Adam(self.learner.parameters(), lr=lr)
        
        # 经验队列
        self.experience_queue = Queue(maxsize=10000)
        
        # 环境
        self.envs = [env_fn() for _ in range(num_actors)]
        self.states = [env.reset() for env in self.envs]
        
        self.running = False
    
    def actor_step(self, actor_id: int):
        """Actor步骤"""
        actor = self.actors[actor_id]
        env = self.envs[actor_id]
        state = self.states[actor_id]
        
        # 选择动作
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = actor(state_tensor)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        
        # 执行动作
        next_state, reward, done, info = env.step(action)
        
        # 存储经验
        self.experience_queue.put((state, action, reward, next_state, done, actor_id))
        
        # 更新状态
        if done:
            self.states[actor_id] = env.reset()
        else:
            self.states[actor_id] = next_state
    
    def vtrace_loss(
        self,
        batch: List[Tuple],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """V-Trace损失"""
        states = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device)
        dones = torch.FloatTensor([e[4] for e in batch]).to(self.device)
        actor_ids = [e[5] for e in batch]
        
        # Learner输出
        policy_logits, values = self.learner(states)
        with torch.no_grad():
            next_policy_logits, next_values = self.learner(next_states)
        
        # Actor输出（用于重要性采样）
        actor_logits = []
        for actor_id in actor_ids:
            with torch.no_grad():
                logits = self.actors[actor_id](states[actor_id:actor_id+1])
                actor_logits.append(logits)
        
        # 计算重要性权重
        learner_probs = F.softmax(policy_logits, dim=-1)
        action_probs = learner_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # V-Trace目标
        rhos = torch.clamp(action_probs / (action_probs + 1e-8), max=self.rho)
        cs = torch.clamp(action_probs / (action_probs + 1e-8), max=self.c)
        
        # 计算returns
        returns = rewards + self.gamma * next_values.squeeze() * (1 - dones)
        vtrace_returns = values.squeeze() + rhos * (returns - values.squeeze())
        
        # 策略损失
        log_probs = F.log_softmax(policy_logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        advantages = vtrace_returns - values.squeeze().detach()
        policy_loss = -(action_log_probs * rhos.detach() * advantages).mean()
        
        # 价值损失
        value_loss = F.mse_loss(values.squeeze(), vtrace_returns.detach())
        
        # 熵奖励
        entropy = -(learner_probs * log_probs).sum(dim=1).mean()
        
        # 总损失
        total_loss = (
            policy_loss +
            self.value_loss_coef * value_loss -
            self.entropy_coef * entropy
        )
        
        metrics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
        }
        
        return total_loss, metrics
    
    def train(self, num_steps: int, callback: Optional[Callable] = None):
        """训练"""
        self.running = True
        total_steps = 0
        
        # 启动actor线程
        actor_threads = []
        for i in range(self.num_actors):
            thread = threading.Thread(target=self._actor_loop, args=(i,))
            thread.daemon = True
            thread.start()
            actor_threads.append(thread)
        
        # Learner循环
        while total_steps < num_steps and self.running:
            # 收集批次
            batch = []
            while len(batch) < self.batch_size:
                try:
                    experience = self.experience_queue.get(timeout=1.0)
                    batch.append(experience)
                except:
                    continue
            
            if not batch:
                continue
            
            # 计算损失并更新
            loss, metrics = self.vtrace_loss(batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 更新actors
            for actor in self.actors:
                actor.load_state_dict(self.learner.feature.state_dict())
                actor.net[-1].weight.data.copy_(self.learner.policy.weight.data)
                actor.net[-1].bias.data.copy_(self.learner.policy.bias.data)
            
            total_steps += len(batch)
            
            if callback:
                callback(total_steps, metrics)
        
        self.running = False
    
    def _actor_loop(self, actor_id: int):
        """Actor循环"""
        while self.running:
            self.actor_step(actor_id)
    
    def stop(self):
        """停止训练"""
        self.running = False


# ==================== R2D2 ====================

class RecurrentDQN(nn.Module):
    """递归DQN网络 - 用于R2D2"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 状态嵌入
        self.embed = nn.Linear(state_dim, hidden_dim)
        
        # LSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        
        # Q值头
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(
        self,
        state: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """前向传播"""
        # 嵌入
        x = self.embed(state)
        
        # LSTM
        if hidden is None:
            h = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            c = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            hidden = (h, c)
        
        lstm_out, new_hidden = self.lstm(x.unsqueeze(1), hidden)
        lstm_out = lstm_out.squeeze(1)
        
        # Q值
        q_values = self.q_head(lstm_out)
        
        return q_values, new_hidden


class R2D2Agent:
    """R2D2智能体"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        batch_size: int = 32,
        sequence_length: int = 80,
        burn_in_length: int = 40,
        target_update_freq: int = 1000,
        device: str = "cuda",
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.burn_in_length = burn_in_length
        self.target_update_freq = target_update_freq
        self.device = device
        
        # 网络
        self.net = RecurrentDQN(state_dim, action_dim).to(device)
        self.target_net = RecurrentDQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.net.state_dict())
        
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        
        # 经验回放
        self.replay_buffer = EpisodeReplayBuffer(
            capacity=1000,
            sequence_length=sequence_length,
            burn_in_length=burn_in_length,
        )
        
        self.update_count = 0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
    
    def select_action(
        self,
        state: np.ndarray,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        evaluate: bool = False,
    ) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor]]:
        """选择动作"""
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1), hidden
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values, new_hidden = self.net(state, hidden)
            action = q_values.argmax(dim=1).item()
        
        return action, new_hidden
    
    def update(self) -> Optional[float]:
        """更新网络"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # 采样序列
        sequences = self.replay_buffer.sample(self.batch_size)
        
        total_loss = 0.0
        
        for sequence in sequences:
            # 准备数据
            states = torch.FloatTensor(np.array([s[0] for s in sequence])).to(self.device)
            actions = torch.LongTensor([s[1] for s in sequence]).to(self.device)
            rewards = torch.FloatTensor([s[2] for s in sequence]).to(self.device)
            dones = torch.FloatTensor([s[3] for s in sequence]).to(self.device)
            
            # Burn-in
            burn_in_states = states[:self.burn_in_length]
            with torch.no_grad():
                _, hidden = self.net(burn_in_states, None)
            
            # 计算Q值
            train_states = states[self.burn_in_length:]
            train_actions = actions[self.burn_in_length:]
            train_rewards = rewards[self.burn_in_length:]
            train_dones = dones[self.burn_in_length:]
            
            q_values, _ = self.net(train_states, hidden)
            
            # 计算目标Q值
            with torch.no_grad():
                target_q_values, _ = self.target_net(train_states, hidden)
                max_target_q = target_q_values.max(dim=1)[0]
                target_q = train_rewards + self.gamma * max_target_q * (1 - train_dones)
            
            # 计算损失
            current_q = q_values.gather(1, train_actions.unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(current_q, target_q)
            total_loss += loss.item()
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # 更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        
        # 衰减epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return total_loss / len(sequences)


# ==================== 主函数 ====================

def main():
    """测试高级RL算法"""
    print("高级RL算法测试")
    
    # 测试优先级回放
    buffer = PrioritizedReplayBuffer(capacity=1000)
    for i in range(100):
        buffer.push((i, i % 5, i * 0.1, i + 1, False), priority=random.random())
    
    experiences, indices, weights = buffer.sample(10)
    print(f"Sampled {len(experiences)} experiences from prioritized buffer")
    
    # 测试Rainbow DQN
    state_dim = 10
    action_dim = 5
    
    net = RainbowDQN(state_dim, action_dim)
    state = torch.randn(2, state_dim)
    probs = net(state)
    q_values = net.get_q_values(probs)
    print(f"Rainbow DQN output shape: {probs.shape}, Q-values shape: {q_values.shape}")
    
    # 测试A3C网络
    a3c_net = A3CNetwork(state_dim, action_dim)
    policy_logits, value = a3c_net(state)
    print(f"A3C policy shape: {policy_logits.shape}, value shape: {value.shape}")
    
    # 测试噪声线性层
    noisy = NoisyLinear(state_dim, action_dim)
    output = noisy(state)
    print(f"Noisy linear output shape: {output.shape}")
    
    print("高级RL算法测试完成")


if __name__ == "__main__":
    main()
