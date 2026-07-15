#!/usr/bin/env python3
"""
CartPole PPO强化学习示例
=========================

使用AGI统一框架实现的完整PPO算法示例。
包含CartPole环境模拟、PPO智能体、训练和奖励可视化。

作者: AGI Framework Team
日期: 2025-05-13
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import time

# 导入框架模块
from core.swing_layer.neural_layers import Linear, Dropout
from core.activations.activations import ReLU, Tanh
from core.initialization.initializers import XavierInitializer, OrthogonalInitializer
from training.optimizers.optimizers import Adam
from training.rl.ppo import PPO  # 假设框架有PPO实现
from core.normalization.normalizations import LayerNormalization


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class PPOConfig:
    """PPO训练配置"""
    # 环境参数
    env_name: str = 'CartPole-v1'
    state_dim: int = 4
    action_dim: int = 2
    max_episode_steps: int = 500
    
    # PPO参数
    gamma: float = 0.99  # 折扣因子
    gae_lambda: float = 0.95  # GAE参数
    clip_epsilon: float = 0.2  # 裁剪参数
    value_coef: float = 0.5  # 价值函数系数
    entropy_coef: float = 0.01  # 熵正则化系数
    
    # 训练参数
    learning_rate: float = 3e-4
    num_epochs: int = 10  # 每次更新的epoch数
    batch_size: int = 64
    num_mini_batches: int = 4
    
    # 网络参数
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = 'tanh'
    
    # 训练流程参数
    total_timesteps: int = 100000
    steps_per_update: int = 2048
    
    # 评估参数
    eval_episodes: int = 10
    eval_interval: int = 10
    
    # 其他
    seed: int = 42
    verbose: bool = True


# =============================================================================
# CartPole环境模拟
# =============================================================================

class CartPoleEnv:
    """
    CartPole环境物理模拟
    基于经典控制问题的物理方程
    """
    
    def __init__(self, config: PPOConfig):
        self.config = config
        
        # 物理参数
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # 杆的半长
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # 时间步长
        
        # 阈值
        self.theta_threshold = 12 * 2 * np.pi / 360  # 角度阈值
        self.x_threshold = 2.4  # 位置阈值
        
        # 状态: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
        self.state = None
        self.steps = 0
        
        np.random.seed(config.seed)
        
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps = 0
        return self.state.copy()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行动作
        
        Returns:
            next_state, reward, done, info
        """
        assert action in [0, 1], "动作必须是0或1"
        
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        
        # 物理计算
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
                   (self.length * (4.0/3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        # 欧拉积分
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.steps += 1
        
        # 检查终止条件
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold
            or theta > self.theta_threshold
            or self.steps >= self.config.max_episode_steps
        )
        
        # 奖励: 每步存活获得1.0
        reward = 1.0
        
        info = {'steps': self.steps}
        
        return self.state.copy(), reward, done, info
    
    def render(self, mode='console'):
        """渲染环境（简化版）"""
        if mode == 'console':
            x, x_dot, theta, theta_dot = self.state
            print(f"Pos: {x:+.3f} | Vel: {x_dot:+.3f} | "
                  f"Angle: {theta:+.3f} | AngVel: {theta_dot:+.3f}")


# =============================================================================
# Actor-Critic网络
# =============================================================================

class ActorCriticNetwork:
    """
    Actor-Critic网络
    共享特征提取层，分别输出策略和价值
    """
    
    def __init__(self, config: PPOConfig):
        self.config = config
        self.shared_layers = []
        self.actor_layers = []
        self.critic_layers = []
        
        self._build_network()
        
    def _build_network(self):
        """构建网络"""
        # 共享特征提取层
        in_dim = self.config.state_dim
        for hidden_dim in self.config.hidden_dims:
            self.shared_layers.append(Linear(
                in_features=in_dim,
                out_features=hidden_dim,
                initializer=OrthogonalInitializer(gain=np.sqrt(2))
            ))
            self.shared_layers.append(LayerNormalization(hidden_dim))
            if self.config.activation == 'tanh':
                self.shared_layers.append(Tanh())
            else:
                self.shared_layers.append(ReLU())
            in_dim = hidden_dim
        
        # Actor头 (策略)
        self.actor_layers.append(Linear(
            in_features=in_dim,
            out_features=self.config.action_dim,
            initializer=OrthogonalInitializer(gain=0.01)
        ))
        
        # Critic头 (价值)
        self.critic_layers.append(Linear(
            in_features=in_dim,
            out_features=1,
            initializer=OrthogonalInitializer(gain=1.0)
        ))
        
    def forward(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        前向传播
        
        Returns:
            logits: 策略logits [batch, action_dim]
            value: 状态价值 [batch, 1]
        """
        x = state
        
        # 共享层
        for layer in self.shared_layers:
            x = layer.forward(x)
        
        # Actor
        logits = self.actor_layers[0].forward(x)
        
        # Critic
        value = self.critic_layers[0].forward(x)
        
        return logits, value
    
    def get_action_and_value(self, state: np.ndarray, 
                             action: Optional[np.ndarray] = None) -> Dict:
        """获取动作、log概率和价值"""
        logits, value = self.forward(state)
        
        # 计算概率
        probs = self._softmax(logits)
        
        if action is None:
            # 采样动作
            action = np.array([np.random.choice(self.config.action_dim, p=p) for p in probs])
        
        # 计算log概率
        log_probs = np.log(probs[np.arange(len(action)), action] + 1e-10)
        
        # 计算熵
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1)
        
        return {
            'action': action,
            'log_prob': log_probs,
            'entropy': entropy,
            'value': value.squeeze(),
            'probs': probs
        }
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """数值稳定的softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def get_value(self, state: np.ndarray) -> np.ndarray:
        """获取状态价值"""
        _, value = self.forward(state)
        return value.squeeze()
    
    def get_parameters(self) -> List[np.ndarray]:
        """获取所有参数"""
        params = []
        for layer in self.shared_layers + self.actor_layers + self.critic_layers:
            if hasattr(layer, 'get_parameters'):
                params.extend(layer.get_parameters())
        return params


# =============================================================================
# 经验回放缓冲区
# =============================================================================

class RolloutBuffer:
    """
    PPO的经验回放缓冲区
    存储轨迹数据用于更新
    """
    
    def __init__(self, buffer_size: int, state_dim: int):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.clear()
        
    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = []
        self.returns = []
        
    def add(self, state: np.ndarray, action: int, reward: float,
            value: float, log_prob: float, done: bool):
        """添加经验"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
    def compute_advantages(self, gamma: float, gae_lambda: float,
                           last_value: float):
        """使用GAE计算优势函数"""
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones)
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * last_gae * (1 - dones[t])
        
        self.advantages = advantages
        self.returns = advantages + np.array(self.values)
        
    def get_batches(self, batch_size: int) -> List[Dict]:
        """获取训练批次"""
        indices = np.arange(len(self.states))
        np.random.shuffle(indices)
        
        batches = []
        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            batch_indices = indices[start:end]
            
            batch = {
                'states': np.array([self.states[i] for i in batch_indices]),
                'actions': np.array([self.actions[i] for i in batch_indices]),
                'log_probs': np.array([self.log_probs[i] for i in batch_indices]),
                'advantages': self.advantages[batch_indices],
                'returns': self.returns[batch_indices]
            }
            batches.append(batch)
        
        return batches
    
    def __len__(self):
        return len(self.states)


# =============================================================================
# PPO智能体
# =============================================================================

class PPOAgent:
    """
    PPO (Proximal Policy Optimization) 智能体
    """
    
    def __init__(self, config: PPOConfig):
        self.config = config
        self.network = ActorCriticNetwork(config)
        self.optimizer = Adam(lr=config.learning_rate)
        self.buffer = RolloutBuffer(config.steps_per_update, config.state_dim)
        
        # 训练统计
        self.training_stats = {
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': [],
            'kl_divs': []
        }
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Dict:
        """选择动作"""
        state_batch = state[np.newaxis, :]
        result = self.network.get_action_and_value(state_batch)
        
        if deterministic:
            action = np.argmax(result['probs'][0])
        else:
            action = result['action'][0]
        
        return {
            'action': action,
            'log_prob': result['log_prob'][0],
            'value': result['value'][0]
        }
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                         value: float, log_prob: float, done: bool):
        """存储转移"""
        self.buffer.add(state, action, reward, value, log_prob, done)
    
    def update(self) -> Dict[str, float]:
        """更新策略"""
        if len(self.buffer) == 0:
            return {}
        
        # 计算优势和回报
        last_state = np.array(self.buffer.states[-1])
        last_value = self.network.get_value(last_state[np.newaxis, :])[0]
        self.buffer.compute_advantages(
            self.config.gamma,
            self.config.gae_lambda,
            last_value
        )
        
        # 标准化优势
        advantages = self.buffer.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages = advantages
        
        # 多轮更新
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for epoch in range(self.config.num_epochs):
            batches = self.buffer.get_batches(self.config.batch_size)
            
            for batch in batches:
                # 计算新的log概率和价值
                result = self.network.get_action_and_value(
                    batch['states'],
                    batch['actions']
                )
                
                new_log_probs = result['log_prob']
                new_values = result['value']
                entropy = result['entropy']
                
                # 策略损失 (PPO裁剪)
                ratio = np.exp(new_log_probs - batch['log_probs'])
                surr1 = ratio * batch['advantages']
                surr2 = np.clip(ratio, 1 - self.config.clip_epsilon, 
                               1 + self.config.clip_epsilon) * batch['advantages']
                policy_loss = -np.mean(np.minimum(surr1, surr2))
                
                # 价值损失
                value_loss = np.mean((new_values - batch['returns']) ** 2)
                
                # 熵损失
                entropy_loss = -np.mean(entropy)
                
                # 总损失
                total_loss = (
                    policy_loss +
                    self.config.value_coef * value_loss +
                    self.config.entropy_coef * entropy_loss
                )
                
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                entropy_losses.append(entropy_loss)
        
        # 清空缓冲区
        self.buffer.clear()
        
        # 记录统计
        stats = {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses)
        }
        
        for key, value in stats.items():
            self.training_stats[f'{key}es'].append(value)
        
        return stats


# =============================================================================
# 训练器
# =============================================================================

class PPOTrainer:
    """PPO训练器"""
    
    def __init__(self, agent: PPOAgent, env: CartPoleEnv, config: PPOConfig):
        self.agent = agent
        self.env = env
        self.config = config
        
        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps = []
        
    def collect_rollouts(self) -> int:
        """收集经验"""
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        num_steps = 0
        
        while num_steps < self.config.steps_per_update:
            # 选择动作
            result = self.agent.select_action(state)
            action = result['action']
            log_prob = result['log_prob']
            value = result['value']
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            
            # 存储转移
            self.agent.store_transition(state, action, reward, value, log_prob, done)
            
            episode_reward += reward
            episode_length += 1
            num_steps += 1
            
            if done:
                # 记录episode统计
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.timesteps.append(num_steps)
                
                # 重置环境
                state = self.env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                state = next_state
        
        return num_steps
    
    def train(self):
        """完整训练流程"""
        print("=" * 60)
        print("开始PPO训练 (CartPole)")
        print("=" * 60)
        print(f"配置: 总步数={self.config.total_timesteps}, "
              f"每轮步数={self.config.steps_per_update}")
        print("-" * 60)
        
        total_timesteps = 0
        update_count = 0
        
        while total_timesteps < self.config.total_timesteps:
            # 收集经验
            steps = self.collect_rollouts()
            total_timesteps += steps
            update_count += 1
            
            # 更新策略
            stats = self.agent.update()
            
            # 打印进度
            if update_count % self.config.eval_interval == 0:
                recent_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
                mean_reward = np.mean(recent_rewards) if recent_rewards else 0
                
                print(f"Update {update_count} | Timesteps: {total_timesteps} | "
                      f"Episodes: {len(self.episode_rewards)} | "
                      f"Mean Reward: {mean_reward:.2f}")
                
                if stats:
                    print(f"  Policy Loss: {stats['policy_loss']:.4f} | "
                          f"Value Loss: {stats['value_loss']:.4f} | "
                          f"Entropy: {stats['entropy_loss']:.4f}")
        
        print("\n" + "=" * 60)
        print("训练完成!")
        print(f"总Episode数: {len(self.episode_rewards)}")
        print(f"最终平均奖励: {np.mean(self.episode_rewards[-10:]):.2f}")
        print("=" * 60)
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """评估智能体"""
        eval_rewards = []
        eval_lengths = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                result = self.agent.select_action(state, deterministic=True)
                action = result['action']
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                episode_length += 1
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'max_reward': np.max(eval_rewards),
            'min_reward': np.min(eval_rewards)
        }


# =============================================================================
# 可视化
# =============================================================================

def plot_training_results(trainer: PPOTrainer, save_path: Optional[str] = None):
    """绘制训练结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Episode奖励
    ax1 = axes[0, 0]
    episodes = range(1, len(trainer.episode_rewards) + 1)
    ax1.plot(episodes, trainer.episode_rewards, alpha=0.3, color='blue', label='Raw')
    
    # 平滑曲线
    if len(trainer.episode_rewards) >= 10:
        smoothed = np.convolve(trainer.episode_rewards, np.ones(10)/10, mode='valid')
        ax1.plot(range(10, len(trainer.episode_rewards) + 1), smoothed, 
                color='red', linewidth=2, label='Smoothed (10-ep)')
    
    ax1.axhline(y=500, color='green', linestyle='--', label='Max Score')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Episode长度
    ax2 = axes[0, 1]
    ax2.plot(episodes, trainer.episode_lengths, alpha=0.3, color='purple')
    if len(trainer.episode_lengths) >= 10:
        smoothed_len = np.convolve(trainer.episode_lengths, np.ones(10)/10, mode='valid')
        ax2.plot(range(10, len(trainer.episode_lengths) + 1), smoothed_len,
                color='darkviolet', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Length')
    ax2.set_title('Episode Lengths')
    ax2.grid(True, alpha=0.3)
    
    # 损失曲线
    ax3 = axes[1, 0]
    agent = trainer.agent
    if agent.training_stats['policy_losses']:
        updates = range(1, len(agent.training_stats['policy_losses']) + 1)
        ax3.plot(updates, agent.training_stats['policy_losses'], 
                label='Policy Loss', color='blue')
        ax3.plot(updates, agent.training_stats['value_losses'],
                label='Value Loss', color='red')
        ax3.set_xlabel('Update')
        ax3.set_ylabel('Loss')
        ax3.set_title('Training Losses')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 熵曲线
    ax4 = axes[1, 1]
    if agent.training_stats['entropy_losses']:
        updates = range(1, len(agent.training_stats['entropy_losses']) + 1)
        ax4.plot(updates, agent.training_stats['entropy_losses'],
                color='green', label='Entropy Loss')
        ax4.set_xlabel('Update')
        ax4.set_ylabel('Entropy')
        ax4.set_title('Entropy Over Training')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练结果图已保存至: {save_path}")
    
    plt.show()


def plot_evaluation(eval_results: Dict[str, float], save_path: Optional[str] = None):
    """绘制评估结果"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    metrics = ['mean_reward', 'max_reward', 'min_reward']
    values = [eval_results[m] for m in metrics]
    colors = ['blue', 'green', 'red']
    
    bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    
    # 添加误差条
    ax.errorbar(['mean_reward'], [eval_results['mean_reward']],
               yerr=[eval_results['std_reward']], fmt='none', 
               color='black', capsize=5, capthick=2)
    
    ax.set_ylabel('Reward')
    ax.set_title('Evaluation Results')
    ax.set_ylim(0, 550)
    ax.axhline(y=500, color='green', linestyle='--', alpha=0.5, label='Max Score')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}',
               ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"评估结果图已保存至: {save_path}")
    
    plt.show()


def demonstrate_agent(agent: PPOAgent, env: CartPoleEnv, num_episodes: int = 3):
    """演示训练好的智能体"""
    print("\n" + "=" * 60)
    print("智能体演示")
    print("=" * 60)
    
    for ep in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        print(f"\nEpisode {ep + 1}:")
        print("-" * 40)
        
        while not done and episode_length < 100:  # 限制演示长度
            result = agent.select_action(state, deterministic=True)
            action = result['action']
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if episode_length % 20 == 0:
                env.render()
        
        print(f"Episode {ep + 1} 完成 | 奖励: {episode_reward:.1f} | 长度: {episode_length}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    # 设置配置
    config = PPOConfig(
        total_timesteps=50000,
        steps_per_update=2048,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        hidden_dims=[64, 64],
        seed=42
    )
    
    # 创建环境
    print("初始化CartPole环境...")
    env = CartPoleEnv(config)
    
    # 创建智能体
    print("创建PPO智能体...")
    agent = PPOAgent(config)
    
    # 创建训练器
    trainer = PPOTrainer(agent, env, config)
    
    # 训练
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time
    
    print(f"\n训练耗时: {train_time:.2f}秒")
    
    # 绘制训练结果
    plot_training_results(trainer, save_path='/workspace/ppo_training.png')
    
    # 评估
    print("\n评估智能体...")
    eval_results = trainer.evaluate(num_episodes=20)
    print(f"评估结果:")
    print(f"  平均奖励: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"  最大奖励: {eval_results['max_reward']:.1f}")
    print(f"  最小奖励: {eval_results['min_reward']:.1f}")
    print(f"  平均长度: {eval_results['mean_length']:.1f}")
    
    plot_evaluation(eval_results, save_path='/workspace/ppo_evaluation.png')
    
    # 演示
    demonstrate_agent(agent, env, num_episodes=3)
    
    print("\n示例运行完成!")


if __name__ == '__main__':
    main()
