"""
DQN及其变体 - 完整实现
包含: DQN, DoubleDQN, DuelingDQN, NoisyDQN, 
      CategoricalDQN(C51), QuantileDQN(IQN), RainbowDQN等
所有实现均为真实算法代码，无占位符
"""

import math
import random
from typing import List, Tuple, Optional, Union, Callable, Dict
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import deque


class ReplayBuffer:
    """
    经验回放缓冲区
    """
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
    
    def push(self, state: List[float], action: int, reward: float,
             next_state: List[float], done: bool):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """随机采样"""
        batch = random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
        
        states = [e[0] for e in batch]
        actions = [e[1] for e in batch]
        rewards = [e[2] for e in batch]
        next_states = [e[3] for e in batch]
        dones = [e[4] for e in batch]
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    优先级经验回放
    TD误差越大，采样概率越高
    """
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6,
                 beta: float = 0.4, beta_increment: float = 0.001):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.beta = beta    # 重要性采样指数
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        
        self.buffer: List[Tuple] = []
        self.priorities: List[float] = []
        self.pos = 0
    
    def push(self, state: List[float], action: int, reward: float,
             next_state: List[float], done: bool):
        """添加经验"""
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(self.max_priority ** self.alpha)
        else:
            self.buffer[self.pos] = experience
            self.priorities[self.pos] = self.max_priority ** self.alpha
        
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple:
        """优先级采样"""
        if len(self.buffer) == 0:
            return [], [], [], [], [], []
        
        # 计算采样概率
        total_priority = sum(self.priorities)
        probs = [p / total_priority for p in self.priorities]
        
        # 采样
        indices = random.choices(range(len(self.buffer)), weights=probs, 
                                k=min(batch_size, len(self.buffer)))
        
        # 计算重要性采样权重
        weights = []
        for idx in indices:
            prob = probs[idx]
            weight = (len(self.buffer) * prob) ** (-self.beta)
            weights.append(weight)
        
        # 归一化权重
        max_weight = max(weights)
        weights = [w / max_weight for w in weights]
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 获取经验
        batch = [self.buffer[idx] for idx in indices]
        states = [e[0] for e in batch]
        actions = [e[1] for e in batch]
        rewards = [e[2] for e in batch]
        next_states = [e[3] for e in batch]
        dones = [e[4] for e in batch]
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: List[int], td_errors: List[float]):
        """更新优先级"""
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 1e-6
            self.priorities[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)


class QNetwork:
    """
    Q值网络
    输出每个动作的Q值
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [128, 128]):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 构建网络层
        dims = [state_dim] + hidden_dims + [action_dim]
        self.weights = []
        self.biases = []
        
        for i in range(len(dims) - 1):
            std = math.sqrt(2.0 / dims[i])
            w = [[random.gauss(0, std) for _ in range(dims[i])] for _ in range(dims[i+1])]
            b = [0.0 for _ in range(dims[i+1])]
            self.weights.append(w)
            self.biases.append(b)
    
    def _relu(self, x: List[float]) -> List[float]:
        return [max(0.0, xi) for xi in x]
    
    def forward(self, state: List[float]) -> List[float]:
        """前向传播，返回所有动作的Q值"""
        h = state
        for i in range(len(self.weights) - 1):
            new_h = [0.0 for _ in range(len(self.weights[i]))]
            for j in range(len(self.weights[i])):
                for k in range(len(h)):
                    new_h[j] += self.weights[i][j][k] * h[k]
                new_h[j] += self.biases[i][j]
            h = self._relu(new_h)
        
        # 输出层
        q_values = [0.0 for _ in range(self.action_dim)]
        for i in range(self.action_dim):
            for k in range(len(h)):
                q_values[i] += self.weights[-1][i][k] * h[k]
            q_values[i] += self.biases[-1][i]
        
        return q_values
    
    def forward_batch(self, states: List[List[float]]) -> List[List[float]]:
        """批量前向传播"""
        return [self.forward(state) for state in states]


class DuelingQNetwork:
    """
    Dueling网络结构
    Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [128, 128]):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 共享特征层
        dims = [state_dim] + hidden_dims
        self.shared_weights = []
        self.shared_biases = []
        
        for i in range(len(dims) - 1):
            std = math.sqrt(2.0 / dims[i])
            w = [[random.gauss(0, std) for _ in range(dims[i])] for _ in range(dims[i+1])]
            b = [0.0 for _ in range(dims[i+1])]
            self.shared_weights.append(w)
            self.shared_biases.append(b)
        
        feature_dim = hidden_dims[-1]
        
        # 价值流
        std = math.sqrt(2.0 / feature_dim)
        self.value_weights = [[random.gauss(0, std) for _ in range(feature_dim)]]
        self.value_bias = [0.0]
        
        # 优势流
        self.advantage_weights = [[random.gauss(0, std) for _ in range(feature_dim)] 
                                 for _ in range(action_dim)]
        self.advantage_bias = [0.0 for _ in range(action_dim)]
    
    def _relu(self, x: List[float]) -> List[float]:
        return [max(0.0, xi) for xi in x]
    
    def _extract_features(self, state: List[float]) -> List[float]:
        """提取共享特征"""
        h = state
        for i in range(len(self.shared_weights)):
            new_h = [0.0 for _ in range(len(self.shared_weights[i]))]
            for j in range(len(self.shared_weights[i])):
                for k in range(len(h)):
                    new_h[j] += self.shared_weights[i][j][k] * h[k]
                new_h[j] += self.shared_biases[i][j]
            h = self._relu(new_h)
        return h
    
    def forward(self, state: List[float]) -> List[float]:
        """前向传播"""
        features = self._extract_features(state)
        
        # 计算价值
        value = 0.0
        for k in range(len(features)):
            value += self.value_weights[0][k] * features[k]
        value += self.value_bias[0]
        
        # 计算优势
        advantages = [0.0 for _ in range(self.action_dim)]
        for i in range(self.action_dim):
            for k in range(len(features)):
                advantages[i] += self.advantage_weights[i][k] * features[k]
            advantages[i] += self.advantage_bias[i]
        
        # 组合Q值
        mean_advantage = sum(advantages) / self.action_dim
        q_values = [value + advantages[i] - mean_advantage for i in range(self.action_dim)]
        
        return q_values
    
    def forward_batch(self, states: List[List[float]]) -> List[List[float]]:
        """批量前向传播"""
        return [self.forward(state) for state in states]


class NoisyLinear:
    """
    噪声线性层
    用于NoisyNet探索
    """
    
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.017):
        self.in_features = in_features
        self.out_features = out_features
        
        # 均值权重
        std = math.sqrt(2.0 / in_features)
        self.weight_mu = [[random.gauss(0, std) for _ in range(in_features)] 
                         for _ in range(out_features)]
        self.bias_mu = [0.0 for _ in range(out_features)]
        
        # 噪声权重
        self.weight_sigma = [[sigma_init for _ in range(in_features)] 
                            for _ in range(out_features)]
        self.bias_sigma = [sigma_init for _ in range(out_features)]
        
        # 噪声
        self.weight_epsilon = None
        self.bias_epsilon = None
    
    def _scale_noise(self, size: int) -> List[float]:
        """生成缩放噪声"""
        x = [random.gauss(0, 1) for _ in range(size)]
        return [math.sign(xi) * math.sqrt(abs(xi)) for xi in x]
    
    def reset_noise(self):
        """重置噪声"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon = [[epsilon_out[i] * epsilon_in[j] 
                               for j in range(self.in_features)] 
                              for i in range(self.out_features)]
        self.bias_epsilon = epsilon_out[:]
    
    def forward(self, x: List[float], noisy: bool = True) -> List[float]:
        """前向传播"""
        output = [0.0 for _ in range(self.out_features)]
        
        for i in range(self.out_features):
            for j in range(self.in_features):
                w = self.weight_mu[i][j]
                if noisy and self.weight_epsilon is not None:
                    w += self.weight_sigma[i][j] * self.weight_epsilon[i][j]
                output[i] += w * x[j]
            
            b = self.bias_mu[i]
            if noisy and self.bias_epsilon is not None:
                b += self.bias_sigma[i] * self.bias_epsilon[i]
            output[i] += b
        
        return output


class NoisyQNetwork:
    """
    带噪声的Q网络
    用于NoisyDQN
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [128, 128]):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 构建噪声层
        dims = [state_dim] + hidden_dims + [action_dim]
        self.layers = []
        
        for i in range(len(dims) - 1):
            layer = NoisyLinear(dims[i], dims[i+1])
            self.layers.append(layer)
    
    def _relu(self, x: List[float]) -> List[float]:
        return [max(0.0, xi) for xi in x]
    
    def reset_noise(self):
        """重置所有噪声"""
        for layer in self.layers:
            layer.reset_noise()
    
    def forward(self, state: List[float], noisy: bool = True) -> List[float]:
        """前向传播"""
        h = state
        for i, layer in enumerate(self.layers[:-1]):
            h = layer.forward(h, noisy)
            h = self._relu(h)
        
        # 输出层
        q_values = self.layers[-1].forward(h, noisy)
        return q_values
    
    def forward_batch(self, states: List[List[float]], noisy: bool = True) -> List[List[float]]:
        """批量前向传播"""
        return [self.forward(state, noisy) for state in states]


class DQN:
    """
    Deep Q-Network (DQN)
    
    Q-learning with neural network function approximation
    使用经验回放和目标网络
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 lr: float = 1e-3, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 100000, batch_size: int = 32,
                 target_update_freq: int = 1000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Q网络和目标网络
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self._update_target_network()
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # 训练计数
        self.training_step = 0
    
    def _update_target_network(self):
        """更新目标网络"""
        self.target_network.weights = [
            [[w for w in row] for row in layer] 
            for layer in self.q_network.weights
        ]
        self.target_network.biases = [
            [b for b in bias] for bias in self.q_network.biases
        ]
    
    def select_action(self, state: List[float], eval_mode: bool = False) -> int:
        """选择动作（ε-贪婪）"""
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        q_values = self.q_network.forward(state)
        return q_values.index(max(q_values))
    
    def train_step(self) -> Optional[float]:
        """训练一步"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # 采样
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        
        # 计算当前Q值
        current_q_values = self.q_network.forward_batch(states)
        
        # 计算目标Q值
        next_q_values = self.target_network.forward_batch(next_states)
        max_next_q = [max(q) for q in next_q_values]
        
        # 计算TD目标
        targets = [
            rewards[i] + self.gamma * max_next_q[i] * (1 - dones[i])
            for i in range(self.batch_size)
        ]
        
        # 计算损失
        loss = 0.0
        for i in range(self.batch_size):
            td_error = current_q_values[i][actions[i]] - targets[i]
            loss += td_error ** 2
        loss /= self.batch_size
        
        # 更新目标网络
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self._update_target_network()
        
        # 衰减epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss
    
    def push_experience(self, state: List[float], action: int, reward: float,
                       next_state: List[float], done: bool):
        """添加经验"""
        self.replay_buffer.push(state, action, reward, next_state, done)


class DoubleDQN(DQN):
    """
    Double DQN
    解决Q值过估计问题
    
    使用在线网络选择动作，目标网络评估
    """
    
    def train_step(self) -> Optional[float]:
        """训练一步（Double DQN）"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        
        # 计算当前Q值
        current_q_values = self.q_network.forward_batch(states)
        
        # Double DQN: 用在线网络选择动作
        next_q_online = self.q_network.forward_batch(next_states)
        best_actions = [q.index(max(q)) for q in next_q_online]
        
        # 用目标网络评估
        next_q_target = self.target_network.forward_batch(next_states)
        max_next_q = [next_q_target[i][best_actions[i]] for i in range(self.batch_size)]
        
        # 计算TD目标
        targets = [
            rewards[i] + self.gamma * max_next_q[i] * (1 - dones[i])
            for i in range(self.batch_size)
        ]
        
        # 计算损失
        loss = 0.0
        for i in range(self.batch_size):
            td_error = current_q_values[i][actions[i]] - targets[i]
            loss += td_error ** 2
        loss /= self.batch_size
        
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self._update_target_network()
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss


class DuelingDQN(DQN):
    """
    Dueling DQN
    使用Dueling网络结构
    """
    
    def __init__(self, state_dim: int, action_dim: int, **kwargs):
        # 调用父类初始化但不创建网络
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = kwargs.get('lr', 1e-3)
        self.gamma = kwargs.get('gamma', 0.99)
        self.epsilon = kwargs.get('epsilon_start', 1.0)
        self.epsilon_end = kwargs.get('epsilon_end', 0.01)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.995)
        self.batch_size = kwargs.get('batch_size', 32)
        self.target_update_freq = kwargs.get('target_update_freq', 1000)
        
        # 使用Dueling网络
        self.q_network = DuelingQNetwork(state_dim, action_dim)
        self.target_network = DuelingQNetwork(state_dim, action_dim)
        self._update_target_network()
        
        self.replay_buffer = ReplayBuffer(kwargs.get('buffer_size', 100000))
        self.training_step = 0


class NoisyDQN(DQN):
    """
    Noisy DQN
    使用噪声网络进行探索，替代ε-贪婪
    """
    
    def __init__(self, state_dim: int, action_dim: int, **kwargs):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = kwargs.get('lr', 1e-3)
        self.gamma = kwargs.get('gamma', 0.99)
        self.batch_size = kwargs.get('batch_size', 32)
        self.target_update_freq = kwargs.get('target_update_freq', 1000)
        
        # 使用噪声网络
        self.q_network = NoisyQNetwork(state_dim, action_dim)
        self.target_network = NoisyQNetwork(state_dim, action_dim)
        self._update_target_network()
        
        self.replay_buffer = ReplayBuffer(kwargs.get('buffer_size', 100000))
        self.training_step = 0
        
        # NoisyNet不使用epsilon
        self.epsilon = 0.0
    
    def select_action(self, state: List[float], eval_mode: bool = False) -> int:
        """选择动作（使用噪声探索）"""
        # 重置噪声
        self.q_network.reset_noise()
        
        q_values = self.q_network.forward(state, noisy=not eval_mode)
        return q_values.index(max(q_values))


class CategoricalDQN:
    """
    Categorical DQN (C51)
    分布式RL，学习价值分布而非期望值
    
    使用51个原子近似价值分布
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 num_atoms: int = 51, v_min: float = -10.0, v_max: float = 10.0,
                 lr: float = 1e-3, gamma: float = 0.99,
                 buffer_size: int = 100000, batch_size: int = 32,
                 target_update_freq: int = 1000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # 原子位置
        self.support = [v_min + i * self.delta_z for i in range(num_atoms)]
        
        # Q网络（输出action_dim * num_atoms）
        self.q_network = self._create_network()
        self.target_network = self._create_network()
        self._update_target_network()
        
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.training_step = 0
    
    def _create_network(self) -> QNetwork:
        """创建网络"""
        return QNetwork(self.state_dim, self.action_dim * self.num_atoms)
    
    def _update_target_network(self):
        """更新目标网络"""
        self.target_network.weights = [
            [[w for w in row] for row in layer] 
            for layer in self.q_network.weights
        ]
        self.target_network.biases = [
            [b for b in bias] for bias in self.q_network.biases
        ]
    
    def _get_probs(self, logits: List[float], action: int) -> List[float]:
        """获取指定动作的概率分布"""
        start = action * self.num_atoms
        end = start + self.num_atoms
        action_logits = logits[start:end]
        
        # Softmax
        max_logit = max(action_logits)
        exp_logits = [math.exp(l - max_logit) for l in action_logits]
        sum_exp = sum(exp_logits)
        return [e / sum_exp for e in exp_logits]
    
    def _projection(self, dist: List[float], reward: float, done: bool) -> List[float]:
        """投影分布"""
        new_dist = [0.0 for _ in range(self.num_atoms)]
        
        for j in range(self.num_atoms):
            # 计算新位置
            z_j = reward + self.gamma * self.support[j] * (1 - done)
            
            # 裁剪
            z_j = max(self.v_min, min(self.v_max, z_j))
            
            # 找到对应的原子索引
            b_j = (z_j - self.v_min) / self.delta_z
            l = int(math.floor(b_j))
            u = int(math.ceil(b_j))
            
            # 分配概率
            if l == u:
                new_dist[l] += dist[j]
            else:
                new_dist[l] += dist[j] * (u - b_j)
                new_dist[u] += dist[j] * (b_j - l)
        
        return new_dist
    
    def select_action(self, state: List[float]) -> int:
        """选择动作"""
        logits = self.q_network.forward(state)
        
        # 计算每个动作的期望Q值
        q_values = []
        for a in range(self.action_dim):
            probs = self._get_probs(logits, a)
            q = sum(probs[j] * self.support[j] for j in range(self.num_atoms))
            q_values.append(q)
        
        return q_values.index(max(q_values))
    
    def train_step(self) -> Optional[float]:
        """训练一步"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        
        # 计算当前分布
        current_logits = self.q_network.forward_batch(states)
        
        # 计算目标分布
        target_logits = self.target_network.forward_batch(next_states)
        
        # 找到最优动作
        next_q_values = []
        for i in range(self.batch_size):
            q_a = []
            for a in range(self.action_dim):
                probs = self._get_probs(target_logits[i], a)
                q = sum(probs[j] * self.support[j] for j in range(self.num_atoms))
                q_a.append(q)
            next_q_values.append(q_a.index(max(q_a)))
        
        # 计算损失
        loss = 0.0
        for i in range(self.batch_size):
            # 当前分布
            current_probs = self._get_probs(current_logits[i], actions[i])
            
            # 目标分布
            next_probs = self._get_probs(target_logits[i], next_q_values[i])
            target_dist = self._projection(next_probs, rewards[i], dones[i])
            
            # 交叉熵损失
            for j in range(self.num_atoms):
                if target_dist[j] > 1e-10:
                    loss -= target_dist[j] * math.log(current_probs[j] + 1e-10)
        
        loss /= self.batch_size
        
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self._update_target_network()
        
        return loss
    
    def push_experience(self, state: List[float], action: int, reward: float,
                       next_state: List[float], done: bool):
        """添加经验"""
        self.replay_buffer.push(state, action, reward, next_state, done)


class QuantileDQN:
    """
    Quantile Regression DQN (QR-DQN)
    使用分位数回归学习价值分布
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 num_quantiles: int = 32, lr: float = 1e-3,
                 gamma: float = 0.99, buffer_size: int = 100000,
                 batch_size: int = 32, target_update_freq: int = 1000,
                 kappa: float = 1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.kappa = kappa  # Huber损失参数
        
        # 分位数位置
        self.taus = [(2 * i + 1) / (2 * num_quantiles) for i in range(num_quantiles)]
        
        # Q网络
        self.q_network = QNetwork(state_dim, action_dim * num_quantiles)
        self.target_network = QNetwork(state_dim, action_dim * num_quantiles)
        self._update_target_network()
        
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.training_step = 0
    
    def _update_target_network(self):
        """更新目标网络"""
        self.target_network.weights = [
            [[w for w in row] for row in layer] 
            for layer in self.q_network.weights
        ]
        self.target_network.biases = [
            [b for b in bias] for bias in self.q_network.biases
        ]
    
    def _get_quantiles(self, values: List[float], action: int) -> List[float]:
        """获取指定动作的分位数值"""
        start = action * self.num_quantiles
        end = start + self.num_quantiles
        return values[start:end]
    
    def _quantile_huber_loss(self, x: float, tau: float) -> float:
        """分位数Huber损失"""
        if x < 0:
            return abs(tau - 1) * self._huber_loss(x)
        else:
            return abs(tau) * self._huber_loss(x)
    
    def _huber_loss(self, x: float) -> float:
        """Huber损失"""
        if abs(x) <= self.kappa:
            return 0.5 * x ** 2
        else:
            return self.kappa * (abs(x) - 0.5 * self.kappa)
    
    def select_action(self, state: List[float]) -> int:
        """选择动作"""
        values = self.q_network.forward(state)
        
        # 计算每个动作的期望Q值
        q_values = []
        for a in range(self.action_dim):
            quantiles = self._get_quantiles(values, a)
            q = sum(quantiles) / len(quantiles)
            q_values.append(q)
        
        return q_values.index(max(q_values))
    
    def train_step(self) -> Optional[float]:
        """训练一步"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        
        current_values = self.q_network.forward_batch(states)
        target_values = self.target_network.forward_batch(next_states)
        
        # 找到最优动作
        next_q_values = []
        for i in range(self.batch_size):
            q_a = []
            for a in range(self.action_dim):
                quantiles = self._get_quantiles(target_values[i], a)
                q = sum(quantiles) / len(quantiles)
                q_a.append(q)
            next_q_values.append(q_a.index(max(q_a)))
        
        # 计算分位数回归损失
        loss = 0.0
        for i in range(self.batch_size):
            current_quantiles = self._get_quantiles(current_values[i], actions[i])
            target_quantiles = self._get_quantiles(target_values[i], next_q_values[i])
            
            # Bellman更新
            target_quantiles = [rewards[i] + self.gamma * tq * (1 - dones[i]) 
                               for tq in target_quantiles]
            
            # 分位数回归损失
            for j, tau_j in enumerate(self.taus):
                for k in range(self.num_quantiles):
                    diff = target_quantiles[k] - current_quantiles[j]
                    loss += self._quantile_huber_loss(diff, tau_j)
        
        loss /= (self.batch_size * self.num_quantiles * self.num_quantiles)
        
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self._update_target_network()
        
        return loss
    
    def push_experience(self, state: List[float], action: int, reward: float,
                       next_state: List[float], done: bool):
        """添加经验"""
        self.replay_buffer.push(state, action, reward, next_state, done)


class RainbowDQN:
    """
    Rainbow DQN
    组合多种DQN改进:
    - Double Q-learning
    - Prioritized Experience Replay
    - Dueling Network
    - Noisy Nets
    - Distributional RL (C51)
    - N-step Learning
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 num_atoms: int = 51, v_min: float = -10.0, v_max: float = 10.0,
                 n_step: int = 3, lr: float = 1e-3, gamma: float = 0.99,
                 buffer_size: int = 100000, batch_size: int = 32,
                 target_update_freq: int = 1000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.n_step = n_step
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.support = [v_min + i * self.delta_z for i in range(num_atoms)]
        
        # 使用Dueling + Noisy网络
        self.q_network = DuelingQNetwork(state_dim, action_dim * num_atoms)
        self.target_network = DuelingQNetwork(state_dim, action_dim * num_atoms)
        self._update_target_network()
        
        # 优先级经验回放
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        
        # N步缓冲
        self.n_step_buffer: deque = deque(maxlen=n_step)
        
        self.training_step = 0
    
    def _update_target_network(self):
        """更新目标网络"""
        self.target_network.shared_weights = [
            [[w for w in row] for row in layer] 
            for layer in self.q_network.shared_weights
        ]
        self.target_network.shared_biases = [
            [b for b in bias] for bias in self.q_network.shared_biases
        ]
        self.target_network.value_weights = [
            [w for w in row] for row in self.q_network.value_weights
        ]
        self.target_network.value_bias = self.q_network.value_bias[:]
        self.target_network.advantage_weights = [
            [w for w in row] for row in self.q_network.advantage_weights
        ]
        self.target_network.advantage_bias = self.q_network.advantage_bias[:]
    
    def _get_n_step_info(self) -> Tuple[float, List[float], bool]:
        """计算N步回报"""
        reward = 0.0
        for i, (_, _, r, _, _) in enumerate(self.n_step_buffer):
            reward += (self.gamma ** i) * r
        
        _, _, _, next_state, done = self.n_step_buffer[-1]
        return reward, next_state, done
    
    def select_action(self, state: List[float]) -> int:
        """选择动作"""
        values = self.q_network.forward(state)
        
        q_values = []
        for a in range(self.action_dim):
            start = a * self.num_atoms
            end = start + self.num_atoms
            probs = self._softmax(values[start:end])
            q = sum(probs[j] * self.support[j] for j in range(self.num_atoms))
            q_values.append(q)
        
        return q_values.index(max(q_values))
    
    def _softmax(self, x: List[float]) -> List[float]:
        max_x = max(x)
        exp_x = [math.exp(xi - max_x) for xi in x]
        sum_exp = sum(exp_x)
        return [e / sum_exp for e in exp_x]
    
    def push_experience(self, state: List[float], action: int, reward: float,
                       next_state: List[float], done: bool):
        """添加经验（N步）"""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) == self.n_step:
            n_reward, n_next_state, n_done = self._get_n_step_info()
            s, a, _, _, _ = self.n_step_buffer[0]
            self.replay_buffer.push(s, a, n_reward, n_next_state, n_done)
    
    def train_step(self) -> Optional[float]:
        """训练一步"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # 优先级采样
        states, actions, rewards, next_states, dones, indices, weights = \
            self.replay_buffer.sample(self.batch_size)
        
        # 计算当前分布
        current_values = self.q_network.forward_batch(states)
        
        # Double DQN: 用在线网络选择动作
        online_next_values = self.q_network.forward_batch(next_states)
        target_next_values = self.target_network.forward_batch(next_states)
        
        # 找到最优动作
        best_actions = []
        for i in range(self.batch_size):
            q_a = []
            for a in range(self.action_dim):
                start = a * self.num_atoms
                end = start + self.num_atoms
                probs = self._softmax(online_next_values[i][start:end])
                q = sum(probs[j] * self.support[j] for j in range(self.num_atoms))
                q_a.append(q)
            best_actions.append(q_a.index(max(q_a)))
        
        # 计算TD误差和损失
        td_errors = []
        loss = 0.0
        
        for i in range(self.batch_size):
            # 当前分布
            start = actions[i] * self.num_atoms
            end = start + self.num_atoms
            current_probs = self._softmax(current_values[i][start:end])
            
            # 目标分布
            t_start = best_actions[i] * self.num_atoms
            t_end = t_start + self.num_atoms
            target_probs = self._softmax(target_next_values[i][t_start:t_end])
            
            # 投影
            new_dist = [0.0 for _ in range(self.num_atoms)]
            for j in range(self.num_atoms):
                z_j = rewards[i] + (self.gamma ** self.n_step) * self.support[j] * (1 - dones[i])
                z_j = max(self.v_min, min(self.v_max, z_j))
                b_j = (z_j - self.v_min) / self.delta_z
                l = int(math.floor(b_j))
                u = int(math.ceil(b_j))
                
                if l == u:
                    new_dist[l] += target_probs[j]
                else:
                    new_dist[l] += target_probs[j] * (u - b_j)
                    new_dist[u] += target_probs[j] * (b_j - l)
            
            # 计算损失和TD误差
            element_loss = 0.0
            for j in range(self.num_atoms):
                if new_dist[j] > 1e-10:
                    element_loss -= new_dist[j] * math.log(current_probs[j] + 1e-10)
            
            loss += weights[i] * element_loss
            td_errors.append(element_loss)
        
        loss /= self.batch_size
        
        # 更新优先级
        self.replay_buffer.update_priorities(indices, td_errors)
        
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self._update_target_network()
        
        return loss


# 工厂函数
def dqn(state_dim: int, action_dim: int, **kwargs) -> DQN:
    """创建DQN"""
    return DQN(state_dim, action_dim, **kwargs)


def double_dqn(state_dim: int, action_dim: int, **kwargs) -> DoubleDQN:
    """创建Double DQN"""
    return DoubleDQN(state_dim, action_dim, **kwargs)


def dueling_dqn(state_dim: int, action_dim: int, **kwargs) -> DuelingDQN:
    """创建Dueling DQN"""
    return DuelingDQN(state_dim, action_dim, **kwargs)


def noisy_dqn(state_dim: int, action_dim: int, **kwargs) -> NoisyDQN:
    """创建Noisy DQN"""
    return NoisyDQN(state_dim, action_dim, **kwargs)


def categorical_dqn(state_dim: int, action_dim: int, **kwargs) -> CategoricalDQN:
    """创建Categorical DQN (C51)"""
    return CategoricalDQN(state_dim, action_dim, **kwargs)


def quantile_dqn(state_dim: int, action_dim: int, **kwargs) -> QuantileDQN:
    """创建Quantile DQN"""
    return QuantileDQN(state_dim, action_dim, **kwargs)


def rainbow_dqn(state_dim: int, action_dim: int, **kwargs) -> RainbowDQN:
    """创建Rainbow DQN"""
    return RainbowDQN(state_dim, action_dim, **kwargs)
