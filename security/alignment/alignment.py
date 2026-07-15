"""
安全对齐模块 - Safety and Alignment

实现AI安全对齐机制：
1. RLHF (Reinforcement Learning from Human Feedback)
2. DPO (Direct Preference Optimization)
3. 价值对齐 (Value Alignment)
4. 约束满足 (Constraint Satisfaction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import numpy as np


@dataclass
class AlignmentConfig:
    """对齐配置"""
    # RLHF参数
    kl_coef: float = 0.1  # KL散度系数
    kl_target: float = 6.0  # 目标KL散度
    gamma: float = 1.0  # 折扣因子
    gae_lambda: float = 0.95  # GAE参数
    
    # DPO参数
    dpo_beta: float = 0.1  # DPO温度参数
    
    # 价值对齐
    value_coef: float = 0.5  # 价值损失系数
    value_clip: float = 0.2  # 价值裁剪
    
    # PPO参数
    ppo_clip: float = 0.2  # PPO裁剪
    ppo_epochs: int = 4  # PPO更新轮数
    ppo_batch_size: int = 64
    
    # 奖励模型
    reward_hidden_dim: int = 256
    reward_dropout: float = 0.1
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class RewardModel(nn.Module):
    """
    奖励模型
    
    从人类反馈学习奖励函数。
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        output_dim: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        计算奖励
        
        Args:
            hidden_states: (batch, hidden_dim) 隐藏状态
            
        Returns:
            (batch, 1) 奖励值
        """
        return self.network(hidden_states)


class ValueModel(nn.Module):
    """
    价值模型
    
    估计状态价值函数V(s)。
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        计算价值
        
        Args:
            hidden_states: (batch, hidden_dim)
            
        Returns:
            (batch, 1) 价值估计
        """
        return self.network(hidden_states)


class RLHFTrainer:
    """
    RLHF训练器
    
    基于人类反馈的强化学习。
    """
    
    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        reward_model: RewardModel,
        value_model: ValueModel,
        config: AlignmentConfig
    ):
        self.policy = policy_model.to(config.device)
        self.ref = ref_model.to(config.device)
        self.reward_model = reward_model.to(config.device)
        self.value_model = value_model.to(config.device)
        self.config = config
        
        # 优化器
        self.policy_optimizer = torch.optim.Adam(policy_model.parameters(), lr=1e-5)
        self.value_optimizer = torch.optim.Adam(value_model.parameters(), lr=1e-4)
        
        # KL自适应
        self.kl_coef = config.kl_coef
    
    def compute_kl_divergence(
        self,
        policy_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        计算KL散度
        
        KL(π || π_ref) = Σ π(x) * (log π(x) - log π_ref(x))
        """
        kl = torch.exp(policy_log_probs) * (policy_log_probs - ref_log_probs)
        return kl.sum(dim=-1)
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算GAE优势估计
        
        A_t = Σ (γλ)^l * δ_{t+l}
        δ_t = r_t + γV(s_{t+1}) - V(s_t)
        """
        gamma = self.config.gamma
        lam = self.config.gae_lambda
        
        batch_size = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0
        next_value = 0
        
        for t in reversed(range(batch_size)):
            if dones[t]:
                next_value = 0
                gae = 0
            
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * lam * gae
            
            advantages[t] = gae
            returns[t] = gae + values[t]
            
            next_value = values[t]
        
        return advantages, returns
    
    def compute_ppo_loss(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算PPO损失
        
        L_policy = -min(r * A, clip(r, 1-ε, 1+ε) * A)
        L_value = (V - R)^2
        """
        # 策略损失
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1 - self.config.ppo_clip,
            1 + self.config.ppo_clip
        ) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 价值损失
        value_pred_clipped = old_values + torch.clamp(
            values - old_values,
            -self.config.value_clip,
            self.config.value_clip
        )
        
        value_loss1 = (values - returns) ** 2
        value_loss2 = (value_pred_clipped - returns) ** 2
        
        value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
        
        # 总损失
        total_loss = policy_loss + self.config.value_coef * value_loss
        
        return total_loss, policy_loss, value_loss
    
    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """
        执行一步RLHF训练
        """
        # 计算参考模型log概率
        with torch.no_grad():
            ref_log_probs = self.ref(states, actions)['log_prob']
        
        # 计算优势
        with torch.no_grad():
            values = self.value_model(states).squeeze(-1)
        advantages, returns = self.compute_advantages(rewards, values, dones)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_kl = 0.0
        
        for _ in range(self.config.ppo_epochs):
            # 策略前向
            policy_output = self.policy(states, actions)
            new_log_probs = policy_output['log_prob']
            
            # KL散度
            kl = self.compute_kl_divergence(new_log_probs, ref_log_probs).mean()
            total_kl += kl.item()
            
            # PPO损失
            new_values = self.value_model(states).squeeze(-1)
            total_loss, policy_loss, value_loss = self.compute_ppo_loss(
                old_log_probs, new_log_probs, advantages,
                new_values, old_values, returns
            )
            
            # 加入KL惩罚
            loss = total_loss + self.kl_coef * kl
            
            # 策略更新
            self.policy_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.policy_optimizer.step()
            
            # 价值更新
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), 1.0)
            self.value_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        # 自适应KL系数
        avg_kl = total_kl / self.config.ppo_epochs
        if avg_kl < self.config.kl_target / 1.5:
            self.kl_coef *= 0.5
        elif avg_kl > self.config.kl_target * 1.5:
            self.kl_coef *= 2.0
        
        return {
            'policy_loss': total_policy_loss / self.config.ppo_epochs,
            'value_loss': total_value_loss / self.config.ppo_epochs,
            'kl_divergence': avg_kl,
            'kl_coef': self.kl_coef
        }


class DPOTrainer:
    """
    DPO训练器
    
    直接偏好优化，无需奖励模型。
    
    L_DPO = -log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))
    """
    
    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        config: AlignmentConfig
    ):
        self.policy = policy_model.to(config.device)
        self.ref = ref_model.to(config.device)
        self.config = config
        
        self.optimizer = torch.optim.Adam(policy_model.parameters(), lr=1e-5)
    
    def compute_dpo_loss(
        self,
        states: torch.Tensor,
        chosen_actions: torch.Tensor,
        rejected_actions: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算DPO损失
        
        Args:
            states: 状态
            chosen_actions: 人类偏好的动作
            rejected_actions: 人类拒绝的动作
            
        Returns:
            loss, metrics
        """
        # 策略模型log概率
        chosen_log_prob = self.policy(states, chosen_actions)['log_prob']
        rejected_log_prob = self.policy(states, rejected_actions)['log_prob']
        
        # 参考模型log概率
        with torch.no_grad():
            ref_chosen_log_prob = self.ref(states, chosen_actions)['log_prob']
            ref_rejected_log_prob = self.ref(states, rejected_actions)['log_prob']
        
        # 对数比率
        chosen_log_ratio = chosen_log_prob - ref_chosen_log_prob
        rejected_log_ratio = rejected_log_prob - ref_rejected_log_prob
        
        # DPO损失
        logits = self.config.dpo_beta * (chosen_log_ratio - rejected_log_ratio)
        loss = -F.logsigmoid(logits).mean()
        
        # 准确率（偏好排序正确）
        accuracy = (logits > 0).float().mean()
        
        return loss, {
            'dpo_loss': loss.item(),
            'accuracy': accuracy.item(),
            'chosen_log_ratio': chosen_log_ratio.mean().item(),
            'rejected_log_ratio': rejected_log_ratio.mean().item()
        }
    
    def train_step(
        self,
        states: torch.Tensor,
        chosen_actions: torch.Tensor,
        rejected_actions: torch.Tensor
    ) -> Dict[str, float]:
        """
        执行一步DPO训练
        """
        loss, metrics = self.compute_dpo_loss(
            states, chosen_actions, rejected_actions
        )
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        return metrics


class ValueAlignment:
    """
    价值对齐
    
    确保AI行为符合人类价值观。
    """
    
    def __init__(
        self,
        value_dimensions: List[str],
        value_weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            value_dimensions: 价值维度列表
                例如：['helpfulness', 'honesty', 'harmlessness', 'fairness']
            value_weights: 各维度权重
        """
        self.dimensions = value_dimensions
        self.weights = value_weights or {d: 1.0 / len(value_dimensions) for d in value_dimensions}
        
        # 价值评估器
        self.evaluators: Dict[str, nn.Module] = {}
    
    def add_evaluator(self, dimension: str, model: nn.Module) -> None:
        """添加价值评估器"""
        self.evaluators[dimension] = model
    
    def compute_alignment_score(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算对齐分数
        
        Returns:
            总分数, 各维度分数
        """
        scores = {}
        total = torch.zeros(states.shape[0], device=states.device)
        
        for dim, evaluator in self.evaluators.items():
            with torch.no_grad():
                dim_score = evaluator(states, actions)
                if isinstance(dim_score, dict):
                    dim_score = dim_score.get('score', dim_score.get('value', 0))
            
            scores[dim] = dim_score.mean().item()
            total = total + self.weights[dim] * dim_score
        
        return total, scores
    
    def compute_alignment_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        target_scores: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        计算对齐损失
        
        Args:
            target_scores: 目标分数（用于训练）
        """
        total_loss = torch.tensor(0.0, device=states.device)
        
        for dim, evaluator in self.evaluators.items():
            output = evaluator(states, actions)
            if isinstance(output, dict):
                score = output.get('score', output.get('value', 0))
            else:
                score = output
            
            if target_scores and dim in target_scores:
                target = target_scores[dim]
                loss = F.mse_loss(score, torch.full_like(score, target))
            else:
                # 鼓励高分
                loss = -score.mean()
            
            total_loss = total_loss + self.weights[dim] * loss
        
        return total_loss


class ConstraintSatisfaction:
    """
    约束满足
    
    确保AI输出满足安全约束。
    """
    
    def __init__(self):
        self.constraints: List[Callable] = []
        self.thresholds: List[float] = []
    
    def add_constraint(
        self,
        constraint_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        threshold: float = 0.0
    ) -> None:
        """
        添加约束
        
        Args:
            constraint_fn: 约束函数，返回违反程度
            threshold: 违反阈值
        """
        self.constraints.append(constraint_fn)
        self.thresholds.append(threshold)
    
    def check_constraints(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[bool, List[float]]:
        """
        检查约束
        
        Returns:
            是否全部满足, 各约束违反程度
        """
        violations = []
        
        for constraint_fn, threshold in zip(self.constraints, self.thresholds):
            violation = constraint_fn(states, actions).item()
            violations.append(violation)
        
        all_satisfied = all(v <= t for v, t in zip(violations, self.thresholds))
        
        return all_satisfied, violations
    
    def compute_barrier_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        margin: float = 0.1
    ) -> torch.Tensor:
        """
        计算屏障损失
        
        当接近约束边界时施加惩罚。
        """
        total_loss = torch.tensor(0.0, device=states.device)
        
        for constraint_fn, threshold in zip(self.constraints, self.thresholds):
            violation = constraint_fn(states, actions)
            
            # 屏障函数：当接近阈值时急剧增加
            barrier = F.relu(violation - threshold + margin) / margin
            total_loss = total_loss + barrier ** 2
        
        return total_loss
    
    def project_to_safe(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        max_iterations: int = 10
    ) -> torch.Tensor:
        """
        投影到安全区域
        
        如果违反约束，找到最近的安全动作。
        """
        safe_actions = actions.clone()
        
        for _ in range(max_iterations):
            all_satisfied, violations = self.check_constraints(states, safe_actions)
            
            if all_satisfied:
                break
            
            # 沿约束梯度方向修正
            for i, (violation, threshold) in enumerate(zip(violations, self.thresholds)):
                if violation > threshold:
                    # 计算约束梯度
                    safe_actions.requires_grad_(True)
                    constraint_val = self.constraints[i](states, safe_actions)
                    grad = torch.autograd.grad(constraint_val, safe_actions)[0]
                    
                    # 沿负梯度方向移动
                    step = (violation - threshold) / (grad.norm() + 1e-8)
                    safe_actions = safe_actions - step * grad
                    safe_actions = safe_actions.detach()
        
        return safe_actions
