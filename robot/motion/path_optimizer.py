#!/usr/bin/env python3
"""
路径优化模块 - 基于深度强化学习（DQN）
功能：训练智能体学习最优运动路径，减少时间/能耗，支持仿真环境训练后迁移到真实机器人
使用纯Python标准库实现
"""

import asyncio
import random
import math
import json
import os
from collections import deque
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SimpleNN:
    """简单的纯Python神经网络实现（用于替代PyTorch）"""
    def __init__(self, layer_sizes: List[int]):
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            # Xavier初始化
            scale = math.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i+1]))
            rows, cols = layer_sizes[i+1], layer_sizes[i]
            self.weights.append([[random.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)])
            self.biases.append([0.0] * rows)
        self.layer_sizes = layer_sizes

    @staticmethod
    def relu(x: float) -> float:
        return max(0.0, x)

    @staticmethod
    def relu_deriv(x: float) -> float:
        return 1.0 if x > 0 else 0.0

    def forward(self, inputs: List[float]) -> List[float]:
        current = inputs[:]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            next_layer = []
            for j in range(len(w)):
                neuron_sum = b[j]
                for k in range(len(current)):
                    neuron_sum += w[j][k] * current[k]
                if i < len(self.weights) - 1:  # 非输出层使用ReLU
                    next_layer.append(self.relu(neuron_sum))
                else:
                    next_layer.append(neuron_sum)
            current = next_layer
        return current

    def backward(self, inputs: List[float], targets: List[float], learning_rate: float = 0.001):
        """简化的反向传播（梯度下降）"""
        # 前向传播保存激活值
        activations = [inputs[:]]
        current = inputs[:]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            next_layer = []
            for j in range(len(w)):
                neuron_sum = b[j]
                for k in range(len(current)):
                    neuron_sum += w[j][k] * current[k]
                if i < len(self.weights) - 1:
                    next_layer.append(self.relu(neuron_sum))
                else:
                    next_layer.append(neuron_sum)
            activations.append(next_layer)
            current = next_layer

        # 计算输出层误差
        errors = []
        for j in range(len(targets)):
            errors.append(activations[-1][j] - targets[j])

        # 反向传播更新权重
        for layer_idx in range(len(self.weights) - 1, -1, -1):
            new_errors = [0.0] * self.layer_sizes[layer_idx]
            for j in range(len(self.weights[layer_idx])):
                for k in range(len(self.weights[layer_idx][j])):
                    grad = errors[j] * activations[layer_idx][k]
                    self.weights[layer_idx][j][k] -= learning_rate * grad
                    self.biases[layer_idx][j] -= learning_rate * errors[j]
                    if layer_idx > 0:
                        new_errors[k] += self.weights[layer_idx][j][k] * errors[j]
            errors = [new_errors[i] * self.relu_deriv(activations[layer_idx][i]) for i in range(len(new_errors))]


class DQNNetwork:
    """深度Q网络（支持连续状态，离散动作）- 纯Python实现"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        self.net = SimpleNN([state_dim, hidden_dim, hidden_dim, action_dim])
        self.state_dim = state_dim
        self.action_dim = action_dim

    def __call__(self, x: List[float]) -> List[float]:
        return self.net.forward(x)


class PathOptimizer:
    """
    基于 DQN 的路径优化器
    训练环境需要提供：状态（关节位置、目标位置）、动作（关节速度变化）、奖励（负时间/能耗）
    """

    def __init__(self, state_dim: int, action_dim: int,
                 lr: float = 1e-4, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000, batch_size: int = 32,
                 target_update_freq: int = 100,
                 device: str = "auto"):
        """
        :param state_dim: 状态维度（例如 6个关节角度 + 6个目标角度 = 12）
        :param action_dim: 动作维度（例如 6个关节速度变化，离散化为每个关节3个选项=18）
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.lr = lr

        # 网络
        self.policy_net = DQNNetwork(state_dim, action_dim)
        self.target_net = DQNNetwork(state_dim, action_dim)
        self._copy_network(self.target_net, self.policy_net)

        # 经验回放
        self.memory = deque(maxlen=memory_size)
        self.step_count = 0

    def _copy_network(self, target: DQNNetwork, source: DQNNetwork):
        """复制网络权重"""
        for i in range(len(target.net.weights)):
            target.net.weights[i] = [[w for w in row] for row in source.net.weights[i]]
            target.net.biases[i] = list(source.net.biases[i])

    def _state_to_list(self, state) -> List[float]:
        if hasattr(state, 'tolist'):
            return state.tolist()
        return list(state)

    def act(self, state, eval_mode: bool = False) -> int:
        """选择动作（epsilon-greedy）"""
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_list = self._state_to_list(state)
        q_values = self.policy_net(state_list)
        return q_values.index(max(q_values))

    def remember(self, state, action: int, reward: float, next_state, done: bool):
        """存储经验"""
        self.memory.append((self._state_to_list(state), action, reward, self._state_to_list(next_state), done))

    def replay(self) -> float:
        """经验回放训练"""
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)
        total_loss = 0.0

        for state, action, reward, next_state, done in batch:
            # 计算当前Q值
            current_q = self.policy_net(state)[action]

            # 计算目标Q值
            with torch.no_grad():
                next_q_values = self.target_net(next_state)
                max_next_q = max(next_q_values)
                if done:
                    target_q = reward
                else:
                    target_q = reward + self.gamma * max_next_q

            # 计算误差并更新
            error = current_q - target_q
            total_loss += error * error

            # 简化的梯度更新（朝向目标Q值更新）
            self._update_single(state, action, target_q)

        # 更新 epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        # 更新目标网络
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self._copy_network(self.target_net, self.policy_net)

        return total_loss / self.batch_size

    def _update_single(self, state: List[float], action: int, target_q: float):
        """对单个样本进行简化更新"""
        current_q = self.policy_net(state)[action]
        delta = target_q - current_q

        # 简化更新：直接调整对应动作的输出
        layer_sizes = self.policy_net.net.layer_sizes
        hidden_dim = layer_sizes[1]

        # 对隐层做小的调整
        for i in range(min(hidden_dim, len(self.policy_net.net.weights[0]))):
            for j in range(len(self.policy_net.net.weights[0][i])):
                self.policy_net.net.weights[0][i][j] += self.lr * delta * 0.01 * state[j]
            self.policy_net.net.biases[0][i] += self.lr * delta * 0.01


# 模拟 torch.no_grad 上下文管理器
class torch:
    @staticmethod
    def no_grad():
        return _NoGrad()


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class ReplayBuffer:
    """简单的经验回放缓冲区"""
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class RobotPathEnv:
    """
    简单的机器人运动环境（模拟）
    状态: [当前关节角度(6), 目标关节角度(6)]
    动作: 离散（每个关节增加/减少/不变）
    奖励: 负的关节角度误差 + 负的步数惩罚
    """
    def __init__(self, joint_limits: List[Tuple[float, float]] = None, step_penalty: float = 0.01):
        self.joint_limits = joint_limits or [(-math.pi, math.pi)] * 6
        self.num_joints = len(self.joint_limits)
        self.action_dim = 3 ** self.num_joints  # 每个关节3种动作
        self.state_dim = self.num_joints * 2
        self.step_penalty = step_penalty
        self.current_joints = None
        self.target_joints = None

    def _clip(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def reset(self):
        self.current_joints = [random.uniform(-math.pi, math.pi) for _ in range(self.num_joints)]
        self.target_joints = [random.uniform(-math.pi, math.pi) for _ in range(self.num_joints)]
        return self.current_joints + self.target_joints

    def step(self, action: int) -> Tuple[List[float], float, bool]:
        # 解码动作：每个关节的动作（-1,0,1）
        delta = [0.0] * self.num_joints
        for j in range(self.num_joints):
            act_j = (action // (3 ** j)) % 3  # 3进制解码
            delta[j] = (act_j - 1) * 0.1  # -0.1, 0, 0.1

        new_joints = []
        for i in range(self.num_joints):
            new_val = self.current_joints[i] + delta[i]
            low, high = self.joint_limits[i]
            new_joints.append(self._clip(new_val, low, high))
        self.current_joints = new_joints

        # 奖励：负的位置误差 + 步数惩罚
        error = sum(abs(self.current_joints[i] - self.target_joints[i]) for i in range(self.num_joints))
        reward = -error - self.step_penalty
        done = error < 0.1
        next_state = self.current_joints + self.target_joints
        return next_state, reward, done

    def render(self):
        pass  # 可视化


# ==================== 使用示例 ====================
async def train_path_optimizer():
    env = RobotPathEnv()
    optimizer = PathOptimizer(state_dim=env.state_dim, action_dim=env.action_dim)
    # 训练
    optimizer.optimize_path(env, episodes=500, max_steps=100)
    # 保存模型
    optimizer.save("path_optimizer.json")


def optimize_path(self, env, episodes: int = 1000, max_steps: int = 200,
                   render: bool = False, on_episode_end=None) -> List[float]:
    """
    训练循环
    :param env: 环境对象，需实现 reset(), step(action) -> (next_state, reward, done)
    :return: 每 episode 的奖励列表
    """
    episode_rewards = []
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        for step in range(max_steps):
            action = self.act(state)
            next_state, reward, done = env.step(action)
            self.remember(state, action, reward, next_state, done)
            loss = self.replay()
            state = next_state
            total_reward += reward
            if done:
                break
        episode_rewards.append(total_reward)
        logger.info(f"Episode {ep}: reward={total_reward:.2f}, epsilon={self.epsilon:.3f}")
        if on_episode_end:
            on_episode_end(ep, total_reward)
    return episode_rewards


def save(self, path: str):
    """保存模型权重"""
    data = {
        'policy_weights': self.policy_net.net.weights,
        'policy_biases': self.policy_net.net.biases,
        'target_weights': self.target_net.net.weights,
        'target_biases': self.target_net.net.biases,
        'epsilon': self.epsilon,
        'step_count': self.step_count
    }
    with open(path, 'w') as f:
        json.dump(data, f)
    logger.info(f"Model saved to {path}")


def load(self, path: str):
    """加载模型权重"""
    with open(path, 'r') as f:
        data = json.load(f)
    self.policy_net.net.weights = data['policy_weights']
    self.policy_net.net.biases = data['policy_biases']
    self.target_net.net.weights = data['target_weights']
    self.target_net.net.biases = data['target_biases']
    self.epsilon = data.get('epsilon', self.epsilon)
    self.step_count = data.get('step_count', 0)
    logger.info(f"Model loaded from {path}")


# 添加方法到类
PathOptimizer.optimize_path = optimize_path
PathOptimizer.save = save
PathOptimizer.load = load


if __name__ == "__main__":
    asyncio.run(train_path_optimizer())
