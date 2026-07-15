"""
GRPO - Group Relative Policy Optimization

该模块实现了GRPO算法,一种基于组采样的相对策略优化方法。
通过组内相对优势计算和KL惩罚,实现稳定的策略更新。

主要特性:
- 组采样
- 相对优势计算
- 策略更新
- KL惩罚
- 参考模型管理

作者: AGI Unified Framework Team
版本: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import math
from copy import deepcopy


@dataclass
class GRPOConfig:
    """GRPO配置类
    
    Attributes:
        group_size: 每组采样数量
        epsilon: PPO裁剪参数
        kl_coef: KL惩罚系数
        entropy_coef: 熵奖励系数
        value_coef: 价值函数系数
        max_grad_norm: 梯度裁剪范数
        lr: 学习率
        gamma: 折扣因子
        gae_lambda: GAE lambda参数
        use_gae: 是否使用GAE
        normalize_advantages: 是否归一化优势
        clip_value: 是否裁剪价值
        value_clip_range: 价值裁剪范围
        temperature: 采样温度
        top_p: 核采样参数
        top_k: Top-k采样参数
    """
    group_size: int = 8
    epsilon: float = 0.2
    kl_coef: float = 0.01
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    lr: float = 1e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    use_gae: bool = True
    normalize_advantages: bool = True
    clip_value: bool = True
    value_clip_range: float = 0.2
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50


class GroupSampler:
    """组采样器
    
    从策略模型中采样一组响应。
    """
    
    def __init__(
        self,
        group_size: int = 8,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50
    ):
        """
        Args:
            group_size: 组大小
            temperature: 采样温度
            top_p: 核采样参数
            top_k: Top-k采样参数
        """
        self.group_size = group_size
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
    
    def sample_group(
        self,
        model: nn.Module,
        prompt: torch.Tensor,
        max_length: int = 100,
        attention_mask: Optional[torch.Tensor] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """
        采样一组响应
        
        Args:
            model: 策略模型
            prompt: 输入提示
            max_length: 最大生成长度
            attention_mask: 注意力掩码
            
        Returns:
            响应列表,每个包含 'tokens', 'log_probs', 'mask'
        """
        responses = []
        
        for _ in range(self.group_size):
            response = self._generate_response(
                model, prompt, max_length, attention_mask
            )
            responses.append(response)
        
        return responses
    
    def _generate_response(
        self,
        model: nn.Module,
        prompt: torch.Tensor,
        max_length: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        生成单个响应
        
        Args:
            model: 策略模型
            prompt: 输入提示
            max_length: 最大长度
            attention_mask: 注意力掩码
            
        Returns:
            响应字典
        """
        model.eval()
        device = prompt.device
        
        generated = prompt.clone()
        log_probs = []
        
        with torch.no_grad():
            for _ in range(max_length):
                # 前向传播
                outputs = model(generated, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :] / self.temperature
                
                # 应用top-k和top-p过滤
                filtered_logits = self._apply_filtering(logits)
                
                # 采样
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 记录log概率
                token_log_prob = F.log_softmax(logits, dim=-1)
                log_probs.append(token_log_prob.gather(-1, next_token))
                
                # 追加token
                generated = torch.cat([generated, next_token], dim=1)
                
                # 更新attention mask
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((attention_mask.shape[0], 1), device=device)
                    ], dim=1)
                
                # 检查是否生成结束符
                if next_token.item() == model.config.eos_token_id:
                    break
        
        return {
            'tokens': generated,
            'response_tokens': generated[:, prompt.shape[1]:],
            'log_probs': torch.cat(log_probs, dim=1),
            'mask': torch.ones_like(generated) if attention_mask is None else attention_mask
        }
    
    def _apply_filtering(self, logits: torch.Tensor) -> torch.Tensor:
        """
        应用top-k和top-p过滤
        
        Args:
            logits: 原始logits
            
        Returns:
            过滤后的logits
        """
        # Top-k过滤
        if self.top_k > 0:
            indices_to_remove = logits < torch.topk(logits, self.top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Top-p过滤
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        return logits


class AdvantageCalculator:
    """优势计算器
    
    计算组内相对优势和GAE优势。
    """
    
    def __init__(
        self,
        use_gae: bool = True,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize: bool = True
    ):
        """
        Args:
            use_gae: 是否使用GAE
            gamma: 折扣因子
            gae_lambda: GAE lambda
            normalize: 是否归一化优势
        """
        self.use_gae = use_gae
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize = normalize
    
    def compute_relative_advantages(
        self,
        rewards: List[float]
    ) -> torch.Tensor:
        """
        计算组内相对优势
        
        Args:
            rewards: 奖励列表
            
        Returns:
            相对优势张量
        """
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        # 计算相对优势: 个体奖励 - 组平均奖励
        mean_reward = rewards_tensor.mean()
        advantages = rewards_tensor - mean_reward
        
        # 归一化
        if self.normalize and len(advantages) > 1:
            std = advantages.std()
            if std > 0:
                advantages = advantages / (std + 1e-8)
        
        return advantages
    
    def compute_gae_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算GAE优势
        
        Args:
            rewards: 奖励序列
            values: 价值估计序列
            dones: 结束标志序列
            
        Returns:
            (优势序列, 回报序列)
        """
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        
        # 计算回报
        returns = advantages + values
        
        # 归一化优势
        if self.normalize and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns


class ReferenceModelManager:
    """参考模型管理器
    
    管理参考模型(通常是初始策略或冻结的策略),
    用于计算KL散度惩罚。
    """
    
    def __init__(self, ref_model: nn.Module, update_interval: int = -1):
        """
        Args:
            ref_model: 参考模型
            update_interval: 更新间隔,-1表示不更新
        """
        self.ref_model = ref_model
        self.update_interval = update_interval
        self.step_count = 0
        
        # 冻结参考模型
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def get_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        获取参考模型的log概率
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            
        Returns:
            log概率张量
        """
        with torch.no_grad():
            outputs = self.ref_model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)
        
        return log_probs
    
    def update_reference(self, new_model: nn.Module):
        """
        更新参考模型
        
        Args:
            new_model: 新模型
        """
        self.ref_model.load_state_dict(new_model.state_dict())
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def maybe_update(self, current_model: nn.Module):
        """
        根据间隔更新参考模型
        
        Args:
            current_model: 当前模型
        """
        if self.update_interval > 0:
            self.step_count += 1
            if self.step_count % self.update_interval == 0:
                self.update_reference(current_model)


class GRPOTrainer:
    """GRPO训练器主类
    
    实现完整的GRPO训练流程,包括组采样、
    相对优势计算、策略更新和KL惩罚。
    """
    
    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        config: Optional[GRPOConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ):
        """
        Args:
            policy_model: 策略模型
            ref_model: 参考模型
            config: 配置
            optimizer: 优化器,None则自动创建
        """
        self.policy_model = policy_model
        self.config = config or GRPOConfig()
        
        # 初始化组件
        self.group_sampler = GroupSampler(
            group_size=self.config.group_size,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k
        )
        
        self.advantage_calculator = AdvantageCalculator(
            use_gae=self.config.use_gae,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            normalize=self.config.normalize_advantages
        )
        
        self.ref_manager = ReferenceModelManager(ref_model)
        
        # 优化器
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.policy_model.parameters(),
                lr=self.config.lr
            )
        else:
            self.optimizer = optimizer
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        
        # 统计信息
        self.loss_history: deque = deque(maxlen=100)
        self.kl_history: deque = deque(maxlen=100)
        self.reward_history: deque = deque(maxlen=100)
    
    def sample_group(
        self,
        prompt: torch.Tensor,
        max_length: int = 100,
        attention_mask: Optional[torch.Tensor] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """
        采样组响应
        
        Args:
            prompt: 输入提示
            max_length: 最大长度
            attention_mask: 注意力掩码
            
        Returns:
            响应组列表
        """
        return self.group_sampler.sample_group(
            self.policy_model, prompt, max_length, attention_mask
        )
    
    def compute_relative_advantages(self, rewards: List[float]) -> torch.Tensor:
        """
        计算相对优势
        
        Args:
            rewards: 奖励列表
            
        Returns:
            相对优势张量
        """
        return self.advantage_calculator.compute_relative_advantages(rewards)
    
    def compute_grpo_loss(
        self,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        ref_log_probs: torch.Tensor,
        old_log_probs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算GRPO损失
        
        Args:
            log_probs: 当前策略log概率
            advantages: 优势值
            ref_log_probs: 参考模型log概率
            old_log_probs: 旧策略log概率,None则使用ref_log_probs
            
        Returns:
            (损失, 统计信息)
        """
        if old_log_probs is None:
            old_log_probs = ref_log_probs
        
        # 计算概率比
        log_ratio = log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        
        # 计算KL散度
        kl_div = torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1
        kl_penalty = kl_div.mean()
        
        # PPO裁剪损失
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 总损失
        loss = policy_loss + self.config.kl_coef * kl_penalty
        
        # 统计信息
        stats = {
            'policy_loss': policy_loss.item(),
            'kl_penalty': kl_penalty.item(),
            'ratio_mean': ratio.mean().item(),
            'ratio_std': ratio.std().item(),
            'advantage_mean': advantages.mean().item(),
            'advantage_std': advantages.std().item()
        }
        
        return loss, stats
    
    def train_step(
        self,
        batch: Dict[str, Any],
        reward_fn: Callable[[List[str]], List[float]]
    ) -> Dict[str, float]:
        """
        执行训练步骤
        
        Args:
            batch: 数据批次,包含 'prompt', 'attention_mask' 等
            reward_fn: 奖励函数,接收响应列表返回奖励列表
            
        Returns:
            训练指标字典
        """
        self.policy_model.train()
        
        prompt = batch['prompt']
        attention_mask = batch.get('attention_mask')
        max_length = batch.get('max_length', 100)
        
        # 采样组响应
        responses = self.sample_group(prompt, max_length, attention_mask)
        
        # 计算奖励
        response_texts = [self._decode_response(r['response_tokens']) for r in responses]
        rewards = reward_fn(response_texts)
        
        # 计算相对优势
        advantages = self.compute_relative_advantages(rewards)
        
        # 计算当前策略的log概率
        log_probs_list = []
        ref_log_probs_list = []
        
        for i, response in enumerate(responses):
            tokens = response['tokens']
            mask = response['mask']
            
            # 当前策略log概率
            outputs = self.policy_model(tokens, attention_mask=mask)
            logits = outputs.logits[:, :-1, :]  # 去掉最后一个
            target_tokens = tokens[:, 1:]  # 目标为下一个token
            
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
            
            # 只计算生成部分的log概率
            prompt_len = prompt.shape[1]
            response_log_probs = token_log_probs[:, prompt_len-1:]
            response_log_prob = response_log_probs.sum(dim=1)
            
            log_probs_list.append(response_log_prob)
            
            # 参考模型log概率
            ref_log_probs = self.ref_manager.get_log_probs(tokens, mask)
            ref_token_log_probs = ref_log_probs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
            ref_response_log_probs = ref_token_log_probs[:, prompt_len-1:]
            ref_response_log_prob = ref_response_log_probs.sum(dim=1)
            
            ref_log_probs_list.append(ref_response_log_prob)
        
        log_probs_tensor = torch.stack(log_probs_list)
        ref_log_probs_tensor = torch.stack(ref_log_probs_list)
        advantages_tensor = advantages.to(log_probs_tensor.device)
        
        # 计算损失
        loss, stats = self.compute_grpo_loss(
            log_probs_tensor,
            advantages_tensor,
            ref_log_probs_tensor
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(),
                self.config.max_grad_norm
            )
        
        self.optimizer.step()
        
        # 更新参考模型(如果需要)
        self.ref_manager.maybe_update(self.policy_model)
        
        self.global_step += 1
        
        # 记录统计
        self.loss_history.append(loss.item())
        self.kl_history.append(stats['kl_penalty'])
        self.reward_history.append(np.mean(rewards))
        
        # 返回指标
        metrics = {
            'loss': loss.item(),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            **stats
        }
        
        return metrics
    
    def _decode_response(self, tokens: torch.Tensor) -> str:
        """
        解码响应token
        
        Args:
            tokens: token张量
            
        Returns:
            解码后的字符串
        """
        # 这里假设模型有tokenizer,实际使用时需要传入
        # 简化实现
        return str(tokens.cpu().tolist())
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取训练统计
        
        Returns:
            统计信息字典
        """
        return {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'mean_loss': np.mean(self.loss_history) if self.loss_history else 0.0,
            'mean_kl': np.mean(self.kl_history) if self.kl_history else 0.0,
            'mean_reward': np.mean(self.reward_history) if self.reward_history else 0.0
        }
    
    def save_checkpoint(self, path: str):
        """
        保存检查点
        
        Args:
            path: 保存路径
        """
        checkpoint = {
            'policy_model_state': self.policy_model.state_dict(),
            'ref_model_state': self.ref_manager.ref_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': self.config
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """
        加载检查点
        
        Args:
            path: 加载路径
        """
        checkpoint = torch.load(path)
        
        self.policy_model.load_state_dict(checkpoint['policy_model_state'])
        self.ref_manager.ref_model.load_state_dict(checkpoint['ref_model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']


# 辅助函数
def create_grpo_trainer(
    policy_model: nn.Module,
    ref_model: nn.Module,
    config_dict: Optional[Dict] = None,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> GRPOTrainer:
    """
    从配置创建GRPO训练器
    
    Args:
        policy_model: 策略模型
        ref_model: 参考模型
        config_dict: 配置字典
        optimizer: 优化器
        
    Returns:
        GRPOTrainer实例
    """
    if config_dict:
        config = GRPOConfig(**config_dict)
    else:
        config = None
    
    return GRPOTrainer(policy_model, ref_model, config, optimizer)


def compute_rewards_with_rules(
    responses: List[str],
    ground_truth: str,
    rules: Optional[List[Callable]] = None
) -> List[float]:
    """
    使用规则计算奖励
    
    Args:
        responses: 响应列表
        ground_truth: 真实答案
        rules: 规则函数列表
        
    Returns:
        奖励列表
    """
    rewards = []
    
    for response in responses:
        reward = 0.0
        
        # 基础奖励:与真实答案的相似度
        if response.strip() == ground_truth.strip():
            reward += 1.0
        
        # 应用额外规则
        if rules:
            for rule in rules:
                reward += rule(response, ground_truth)
        
        rewards.append(reward)
    
    return rewards


def length_penalty_rule(response: str, ground_truth: str) -> float:
    """
    长度惩罚规则
    
    惩罚过长或过短的响应。
    """
    target_len = len(ground_truth.split())
    actual_len = len(response.split())
    
    if actual_len == 0:
        return -1.0
    
    ratio = actual_len / target_len
    
    if 0.8 <= ratio <= 1.2:
        return 0.1
    elif 0.5 <= ratio <= 2.0:
        return 0.0
    else:
        return -0.2


def format_check_rule(response: str, ground_truth: str) -> float:
    """
    格式检查规则
    
    检查响应是否符合预期格式。
    """
    # 检查是否有合理的句子结构
    if response.strip() and response[0].isupper() and response[-1] in '.!?':
        return 0.05
    return 0.0
