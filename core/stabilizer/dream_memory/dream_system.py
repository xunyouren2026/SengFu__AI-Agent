"""
AGI统一框架 - 梦境模拟与记忆巩固
实现类似生物睡眠时的记忆回放、经验重放、知识蒸馏等机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
import numpy as np
import math
from collections import deque, OrderedDict
import random
from abc import ABC, abstractmethod
import time


# ==================== 配置类 ====================

@dataclass
class DreamConfig:
    """梦境模拟配置"""
    # 经验回放缓冲区
    buffer_size: int = 100000
    batch_size: int = 256
    priority_alpha: float = 0.6  # 优先级指数
    priority_beta: float = 0.4  # 重要性采样指数
    
    # 梦境模拟
    num_dream_episodes: int = 10
    dream_length: int = 100
    imagination_steps: int = 5
    
    # 记忆巩固
    consolidation_steps: int = 100
    replay_ratio: int = 4  # 每个真实样本对应多少回放样本
    
    # 知识蒸馏
    distillation_temperature: float = 4.0
    distillation_alpha: float = 0.5
    
    # 睡眠阶段
    light_sleep_ratio: float = 0.3
    deep_sleep_ratio: float = 0.5
    rem_sleep_ratio: float = 0.2
    
    # 遗忘机制
    forgetting_rate: float = 0.01
    importance_threshold: float = 0.1


# ==================== 经验回放缓冲区 ====================

class ExperienceReplayBuffer:
    """标准经验回放缓冲区"""
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, action: np.ndarray, reward: float, 
             next_state: np.ndarray, done: bool, **kwargs):
        """添加经验"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            **kwargs
        }
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Dict]:
        """随机采样"""
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()


class PrioritizedReplayBuffer:
    """优先级经验回放缓冲区 (PER)"""
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        
        self.buffer: List[Dict] = []
        self.priorities: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0
        
    def push(self, state: np.ndarray, action: np.ndarray, reward: float,
             next_state: np.ndarray, done: bool, **kwargs):
        """添加经验"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            **kwargs
        }
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
            
        self.priorities[self.pos] = self.max_priority ** self.alpha
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """优先级采样"""
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])
        
        # 计算采样概率
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities / priorities.sum()
        
        # 采样索引
        indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=probs)
        
        # 计算重要性采样权重
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # 归一化
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        experiences = [self.buffer[i] for i in indices]
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            
    def __len__(self) -> int:
        return len(self.buffer)


class HindsightExperienceReplay:
    """后见之明经验回放 (HER)"""
    
    def __init__(self, capacity: int = 100000, k_future: int = 4, 
                 reward_strategy: str = 'sparse'):
        self.capacity = capacity
        self.k_future = k_future
        self.reward_strategy = reward_strategy
        self.buffer: deque = deque(maxlen=capacity)
        self.episode_buffer: List[Dict] = []
        
    def push(self, state: np.ndarray, action: np.ndarray, reward: float,
             next_state: np.ndarray, done: bool, goal: np.ndarray, **kwargs):
        """添加经验"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'goal': goal,
            **kwargs
        }
        self.episode_buffer.append(experience)
        
    def end_episode(self, achieved_goals: List[np.ndarray]):
        """结束回合，生成HER样本"""
        # 添加原始经验
        for exp in self.episode_buffer:
            self.buffer.append(exp)
        
        # 生成后见之明样本
        for i, exp in enumerate(self.episode_buffer):
            # 随机选择k个未来目标
            future_indices = random.sample(
                range(i, len(self.episode_buffer)), 
                min(self.k_future, len(self.episode_buffer) - i)
            )
            
            for future_idx in future_indices:
                # 使用未来状态作为目标
                new_goal = achieved_goals[future_idx]
                
                # 重新计算奖励
                achieved = achieved_goals[i]
                if self.reward_strategy == 'sparse':
                    new_reward = 0.0 if np.linalg.norm(achieved - new_goal) < 0.05 else -1.0
                else:
                    new_reward = -np.linalg.norm(achieved - new_goal)
                
                # 检查是否达成
                new_done = np.linalg.norm(achieved - new_goal) < 0.05
                
                her_exp = {
                    'state': exp['state'],
                    'action': exp['action'],
                    'reward': new_reward,
                    'next_state': exp['next_state'],
                    'done': new_done,
                    'goal': new_goal
                }
                self.buffer.append(her_exp)
        
        self.episode_buffer.clear()
        
    def sample(self, batch_size: int) -> List[Dict]:
        """采样"""
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)


# ==================== 梦境模拟器 ====================

class DreamSimulator(nn.Module):
    """梦境模拟器 - 在"睡眠"时进行想象和回放"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 config: Optional[DreamConfig] = None):
        super().__init__()
        self.config = config or DreamConfig()
        
        # 世界模型 (状态转移预测)
        self.world_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # 奖励模型
        self.reward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 想象策略网络
        self.imagination_policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 不确定性估计
        self.uncertainty_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def predict_next_state(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """预测下一状态"""
        x = torch.cat([state, action], dim=-1)
        return self.world_model(x)
    
    def predict_reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """预测奖励"""
        x = torch.cat([state, action], dim=-1)
        return self.reward_model(x)
    
    def imagine_action(self, state: torch.Tensor) -> torch.Tensor:
        """想象动作"""
        return torch.tanh(self.imagination_policy(state))
    
    def estimate_uncertainty(self, state: torch.Tensor) -> torch.Tensor:
        """估计状态不确定性"""
        return self.uncertainty_net(state)
    
    def dream_episode(self, initial_state: torch.Tensor, 
                      steps: int = 100) -> Dict[str, torch.Tensor]:
        """执行一个梦境回合"""
        states = [initial_state]
        actions = []
        rewards = []
        uncertainties = []
        
        current_state = initial_state
        
        for _ in range(steps):
            # 想象动作
            action = self.imagine_action(current_state)
            actions.append(action)
            
            # 预测下一状态和奖励
            next_state = self.predict_next_state(current_state, action)
            reward = self.predict_reward(current_state, action)
            
            # 估计不确定性
            uncertainty = self.estimate_uncertainty(current_state)
            
            states.append(next_state)
            rewards.append(reward)
            uncertainties.append(uncertainty)
            
            current_state = next_state
        
        return {
            'states': torch.stack(states),
            'actions': torch.stack(actions),
            'rewards': torch.stack(rewards),
            'uncertainties': torch.stack(uncertainties)
        }
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        next_state = self.predict_next_state(state, action)
        reward = self.predict_reward(state, action)
        uncertainty = self.estimate_uncertainty(state)
        
        return {
            'next_state': next_state,
            'reward': reward,
            'uncertainty': uncertainty
        }


# ==================== 记忆巩固器 ====================

class MemoryConsolidator:
    """记忆巩固器 - 模拟睡眠时的记忆整合"""
    
    def __init__(self, model: nn.Module, config: Optional[DreamConfig] = None,
                 device: str = 'cpu'):
        self.config = config or DreamConfig()
        self.model = model
        self.device = device
        
        # 经验缓冲区
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.config.buffer_size,
            alpha=self.config.priority_alpha,
            beta=self.config.priority_beta
        )
        
        # 统计信息
        self.consolidation_count = 0
        self.total_loss_history: List[float] = []
        
    def add_experience(self, state: np.ndarray, action: np.ndarray, 
                       reward: float, next_state: np.ndarray, done: bool,
                       td_error: float = 1.0, **kwargs):
        """添加经验"""
        self.replay_buffer.push(
            state, action, reward, next_state, done,
            td_error=td_error, **kwargs
        )
        
    def consolidate(self, optimizer: torch.optim.Optimizer,
                    loss_fn: Callable,
                    num_steps: Optional[int] = None) -> Dict[str, float]:
        """执行记忆巩固"""
        num_steps = num_steps or self.config.consolidation_steps
        
        total_loss = 0.0
        num_updates = 0
        
        for _ in range(num_steps):
            if len(self.replay_buffer) < self.config.batch_size:
                break
            
            # 采样
            experiences, indices, weights = self.replay_buffer.sample(
                self.config.batch_size
            )
            
            # 转换为张量
            states = torch.tensor(
                np.stack([e['state'] for e in experiences]), 
                device=self.device, dtype=torch.float32
            )
            actions = torch.tensor(
                np.stack([e['action'] for e in experiences]),
                device=self.device, dtype=torch.float32
            )
            rewards = torch.tensor(
                [e['reward'] for e in experiences],
                device=self.device, dtype=torch.float32
            )
            next_states = torch.tensor(
                np.stack([e['next_state'] for e in experiences]),
                device=self.device, dtype=torch.float32
            )
            dones = torch.tensor(
                [e['done'] for e in experiences],
                device=self.device, dtype=torch.float32
            )
            weights = torch.tensor(weights, device=self.device, dtype=torch.float32)
            
            # 计算损失
            optimizer.zero_grad()
            loss, td_errors = loss_fn(states, actions, rewards, next_states, dones)
            
            # 加权损失
            weighted_loss = (loss * weights).mean()
            weighted_loss.backward()
            optimizer.step()
            
            # 更新优先级
            new_priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-6
            self.replay_buffer.update_priorities(indices, new_priorities)
            
            total_loss += weighted_loss.item()
            num_updates += 1
        
        self.consolidation_count += 1
        avg_loss = total_loss / max(num_updates, 1)
        self.total_loss_history.append(avg_loss)
        
        return {
            'avg_loss': avg_loss,
            'num_updates': num_updates,
            'buffer_size': len(self.replay_buffer)
        }


# ==================== 知识蒸馏 ====================

class KnowledgeDistiller:
    """知识蒸馏 - 将教师模型知识转移到学生模型"""
    
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module,
                 temperature: float = 4.0, alpha: float = 0.5):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha  # 软标签权重
        
        # 教师模型冻结
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
    def distillation_loss(self, student_logits: torch.Tensor,
                          teacher_logits: torch.Tensor,
                          targets: torch.Tensor) -> torch.Tensor:
        """计算蒸馏损失"""
        # 软标签损失 (KL散度)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # 硬标签损失
        hard_loss = F.cross_entropy(student_logits, targets)
        
        # 组合损失
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
    
    def distill(self, data_loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                num_epochs: int = 10) -> Dict[str, List[float]]:
        """执行知识蒸馏"""
        history = {'loss': [], 'accuracy': []}
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch in data_loader:
                inputs, targets = batch
                inputs = inputs.to(next(self.student.parameters()).device)
                targets = targets.to(next(self.student.parameters()).device)
                
                # 教师预测
                with torch.no_grad():
                    teacher_logits = self.teacher(inputs)
                
                # 学生预测
                student_logits = self.student(inputs)
                
                # 计算损失
                loss = self.distillation_loss(student_logits, teacher_logits, targets)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = student_logits.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
            
            avg_loss = total_loss / len(data_loader)
            accuracy = correct / total
            
            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)
            
        return history


# ==================== 睡眠周期管理 ====================

class SleepCycle:
    """睡眠周期管理 - 模拟生物睡眠的不同阶段"""
    
    def __init__(self, config: Optional[DreamConfig] = None):
        self.config = config or DreamConfig()
        
        # 睡眠阶段
        self.phases = ['light_sleep', 'deep_sleep', 'rem_sleep']
        self.phase_ratios = {
            'light_sleep': self.config.light_sleep_ratio,
            'deep_sleep': self.config.deep_sleep_ratio,
            'rem_sleep': self.config.rem_sleep_ratio
        }
        
        # 当前状态
        self.current_phase = None
        self.phase_progress = 0.0
        self.cycle_count = 0
        
    def start_sleep_cycle(self, total_steps: int) -> Dict[str, int]:
        """开始睡眠周期"""
        phase_steps = {}
        for phase, ratio in self.phase_ratios.items():
            phase_steps[phase] = int(total_steps * ratio)
        
        self.cycle_count += 1
        return phase_steps
    
    def get_current_phase(self, step: int, total_steps: int) -> str:
        """获取当前睡眠阶段"""
        progress = step / total_steps
        
        if progress < self.phase_ratios['light_sleep']:
            self.current_phase = 'light_sleep'
        elif progress < self.phase_ratios['light_sleep'] + self.phase_ratios['deep_sleep']:
            self.current_phase = 'deep_sleep'
        else:
            self.current_phase = 'rem_sleep'
            
        self.phase_progress = progress
        return self.current_phase
    
    def get_phase_parameters(self, phase: str) -> Dict[str, float]:
        """获取阶段特定参数"""
        if phase == 'light_sleep':
            return {
                'learning_rate_scale': 0.5,
                'noise_level': 0.1,
                'replay_ratio': 2
            }
        elif phase == 'deep_sleep':
            return {
                'learning_rate_scale': 0.1,
                'noise_level': 0.01,
                'replay_ratio': 4
            }
        else:  # REM sleep
            return {
                'learning_rate_scale': 1.0,
                'noise_level': 0.3,
                'replay_ratio': 1
            }


# ==================== 主梦境系统 ====================

class DreamSystem:
    """完整的梦境系统"""
    
    def __init__(self, model: nn.Module, state_dim: int, action_dim: int,
                 config: Optional[DreamConfig] = None, device: str = 'cpu'):
        self.config = config or DreamConfig()
        self.device = device
        
        # 组件
        self.dream_simulator = DreamSimulator(
            state_dim, action_dim, 
            hidden_dim=256, 
            config=self.config
        ).to(device)
        
        self.memory_consolidator = MemoryConsolidator(
            model, self.config, device
        )
        
        self.sleep_cycle = SleepCycle(self.config)
        
        # 统计
        self.dream_count = 0
        self.total_dream_rewards = 0.0
        
    def add_experience(self, state: np.ndarray, action: np.ndarray,
                       reward: float, next_state: np.ndarray, done: bool,
                       **kwargs):
        """添加经验"""
        self.memory_consolidator.add_experience(
            state, action, reward, next_state, done, **kwargs
        )
        
    def dream(self, initial_states: torch.Tensor,
              dream_optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """执行梦境模拟"""
        self.dream_count += 1
        
        all_dream_data = []
        total_dream_reward = 0.0
        
        for state in initial_states:
            # 执行梦境回合
            dream_data = self.dream_simulator.dream_episode(
                state, self.config.dream_length
            )
            all_dream_data.append(dream_data)
            total_dream_reward += dream_data['rewards'].sum().item()
        
        # 训练世界模型
        if len(self.memory_consolidator.replay_buffer) >= self.config.batch_size:
            world_loss = self._train_world_model(dream_optimizer)
        else:
            world_loss = 0.0
        
        self.total_dream_rewards += total_dream_reward
        
        return {
            'dream_data': all_dream_data,
            'total_reward': total_dream_reward,
            'world_model_loss': world_loss,
            'dream_count': self.dream_count
        }
    
    def _train_world_model(self, optimizer: torch.optim.Optimizer) -> float:
        """训练世界模型"""
        experiences, _, _ = self.memory_consolidator.replay_buffer.sample(
            self.config.batch_size
        )
        
        states = torch.tensor(
            np.stack([e['state'] for e in experiences]),
            device=self.device, dtype=torch.float32
        )
        actions = torch.tensor(
            np.stack([e['action'] for e in experiences]),
            device=self.device, dtype=torch.float32
        )
        rewards = torch.tensor(
            [e['reward'] for e in experiences],
            device=self.device, dtype=torch.float32
        )
        next_states = torch.tensor(
            np.stack([e['next_state'] for e in experiences]),
            device=self.device, dtype=torch.float32
        )
        
        # 预测
        pred_next_states = self.dream_simulator.predict_next_state(states, actions)
        pred_rewards = self.dream_simulator.predict_reward(states, actions)
        
        # 损失
        state_loss = F.mse_loss(pred_next_states, next_states)
        reward_loss = F.mse_loss(pred_rewards.squeeze(), rewards)
        total_loss = state_loss + reward_loss
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item()
    
    def sleep(self, model_optimizer: torch.optim.Optimizer,
              loss_fn: Callable,
              total_steps: int = 1000) -> Dict[str, Any]:
        """执行完整睡眠周期"""
        phase_steps = self.sleep_cycle.start_sleep_cycle(total_steps)
        
        results = {}
        
        for step in range(total_steps):
            phase = self.sleep_cycle.get_current_phase(step, total_steps)
            params = self.sleep_cycle.get_phase_parameters(phase)
            
            # 调整学习率
            for param_group in model_optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * params['learning_rate_scale']
            
            # 执行记忆巩固
            consolidation_result = self.memory_consolidator.consolidate(
                model_optimizer, loss_fn,
                num_steps=params['replay_ratio']
            )
            
            results[phase] = consolidation_result
        
        return {
            'phases': results,
            'cycle_count': self.sleep_cycle.cycle_count
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'dream_count': self.dream_count,
            'total_dream_rewards': self.total_dream_rewards,
            'avg_dream_reward': self.total_dream_rewards / max(self.dream_count, 1),
            'buffer_size': len(self.memory_consolidator.replay_buffer),
            'consolidation_count': self.memory_consolidator.consolidation_count
        }


# ==================== 遗忘机制 ====================

class ForgettingMechanism:
    """遗忘机制 - 模拟人类记忆的遗忘过程"""
    
    def __init__(self, forgetting_rate: float = 0.01,
                 importance_threshold: float = 0.1):
        self.forgetting_rate = forgetting_rate
        self.importance_threshold = importance_threshold
        
    def apply_forgetting(self, memory_weights: np.ndarray,
                         importance_scores: np.ndarray,
                         time_since_access: np.ndarray) -> np.ndarray:
        """应用遗忘"""
        # 基于时间的衰减
        time_decay = np.exp(-self.forgetting_rate * time_since_access)
        
        # 基于重要性的保留
        importance_mask = importance_scores > self.importance_threshold
        
        # 组合
        new_weights = memory_weights * time_decay
        new_weights[importance_mask] = memory_weights[importance_mask]  # 重要记忆保留
        
        return new_weights
    
    def selective_forgetting(self, memories: List[Dict],
                             importance_key: str = 'importance') -> List[Dict]:
        """选择性遗忘"""
        retained_memories = []
        
        for memory in memories:
            importance = memory.get(importance_key, 0.5)
            
            # 基于重要性和随机性决定是否保留
            retain_prob = importance * (1 - self.forgetting_rate)
            
            if np.random.random() < retain_prob:
                retained_memories.append(memory)
        
        return retained_memories


# ==================== 记忆检索 ====================

class MemoryRetrieval:
    """记忆检索系统"""
    
    def __init__(self, embedding_dim: int = 256, num_memories: int = 10000):
        self.embedding_dim = embedding_dim
        self.num_memories = num_memories
        
        # 记忆存储
        self.memory_keys: np.ndarray = np.zeros((num_memories, embedding_dim))
        self.memory_values: List[Dict] = [None] * num_memories
        self.memory_importance: np.ndarray = np.zeros(num_memories)
        
        self.current_size = 0
        self.next_idx = 0
        
    def store(self, key: np.ndarray, value: Dict, importance: float = 1.0):
        """存储记忆"""
        if self.current_size < self.num_memories:
            self.memory_keys[self.next_idx] = key
            self.memory_values[self.next_idx] = value
            self.memory_importance[self.next_idx] = importance
            self.next_idx = (self.next_idx + 1) % self.num_memories
            self.current_size += 1
        else:
            # 替换最不重要的记忆
            min_idx = np.argmin(self.memory_importance)
            self.memory_keys[min_idx] = key
            self.memory_values[min_idx] = value
            self.memory_importance[min_idx] = importance
            
    def retrieve(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """检索记忆"""
        if self.current_size == 0:
            return []
        
        # 计算相似度
        keys = self.memory_keys[:self.current_size]
        similarities = np.dot(keys, query) / (
            np.linalg.norm(keys, axis=1) * np.linalg.norm(query) + 1e-8
        )
        
        # 加权重要性
        importance = self.memory_importance[:self.current_size]
        scores = similarities * 0.7 + importance * 0.3
        
        # 获取top-k
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((self.memory_values[idx], scores[idx]))
            
        return results
    
    def update_importance(self, idx: int, delta: float):
        """更新记忆重要性"""
        if 0 <= idx < self.current_size:
            self.memory_importance[idx] = np.clip(
                self.memory_importance[idx] + delta, 0, 1
            )
