"""
SimPO: Simple Preference Optimization
简单偏好优化算法实现

基于论文 "SimPO: Simple Preference Optimization with a Reference-Free Reward"
简化版DPO，无需参考模型，使用长度归一化奖励
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from transformers import PreTrainedModel, PreTrainedTokenizer
import numpy as np


@dataclass
class SimPOConfig:
    """SimPO配置参数"""
    beta: float = 2.0  # 温度参数（SimPO使用较大的beta值）
    gamma: float = 1.0  # 奖励差距的边际参数
    length_normalization: bool = True  # 是否使用长度归一化
    label_smoothing: float = 0.0  # 标签平滑
    max_length: int = 2048  # 最大序列长度
    use_target_reward: bool = True  # 是否使用目标奖励机制


class SimPOLoss(nn.Module):
    """
    SimPO损失函数
    简化版DPO，无需参考模型，使用长度归一化奖励
    """
    
    def __init__(
        self,
        beta: float = 2.0,
        gamma: float = 1.0,
        length_normalization: bool = True,
        label_smoothing: float = 0.0,
        use_target_reward: bool = True
    ):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.length_normalization = length_normalization
        self.label_smoothing = label_smoothing
        self.use_target_reward = use_target_reward
    
    def compute_sequence_log_probs(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算序列的对数概率
        
        Args:
            logits: 模型输出logits [batch_size, seq_len, vocab_size]
            labels: 标签序列 [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            
        Returns:
            每个序列的对数概率 [batch_size]
        """
        # 移位以对齐预测
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        if attention_mask is not None:
            shift_attention_mask = attention_mask[..., 1:].contiguous()
        else:
            shift_attention_mask = torch.ones_like(shift_labels)
        
        # 计算log softmax
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # 获取真实标签的log概率
        batch_size, seq_len, vocab_size = log_probs.shape
        flat_log_probs = log_probs.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)
        
        # 收集目标token的log概率
        token_log_probs = flat_log_probs.gather(dim=-1, index=flat_labels.unsqueeze(-1)).squeeze(-1)
        token_log_probs = token_log_probs.view(batch_size, seq_len)
        
        # 应用attention mask并求和
        masked_log_probs = token_log_probs * shift_attention_mask
        sequence_log_probs = masked_log_probs.sum(dim=-1)
        
        return sequence_log_probs
    
    def compute_length_normalized_reward(
        self,
        log_probs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算长度归一化的奖励
        
        Args:
            log_probs: 序列对数概率
            attention_mask: 注意力掩码
            
        Returns:
            长度归一化奖励
        """
        if not self.length_normalization or attention_mask is None:
            return log_probs
        
        # 计算有效长度（非padding token数）
        sequence_lengths = attention_mask.sum(dim=-1).float()
        
        # 长度归一化: r = log_prob / length
        normalized_reward = log_probs / (sequence_lengths + 1e-8)
        
        return normalized_reward
    
    def forward(
        self,
        policy_chosen_logits: torch.Tensor,
        policy_rejected_logits: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_labels: torch.Tensor,
        chosen_attention_mask: Optional[torch.Tensor] = None,
        rejected_attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播计算SimPO损失
        
        Args:
            policy_chosen_logits: 策略模型对偏好响应的logits
            policy_rejected_logits: 策略模型对非偏好响应的logits
            chosen_labels: 偏好响应的标签
            rejected_labels: 非偏好响应的标签
            chosen_attention_mask: 偏好响应的注意力掩码
            rejected_attention_mask: 非偏好响应的注意力掩码
            
        Returns:
            包含各项损失的字典
        """
        # 计算对数概率
        chosen_log_probs = self.compute_sequence_log_probs(
            policy_chosen_logits, chosen_labels, chosen_attention_mask
        )
        rejected_log_probs = self.compute_sequence_log_probs(
            policy_rejected_logits, rejected_labels, rejected_attention_mask
        )
        
        # 计算长度归一化奖励
        chosen_rewards = self.compute_length_normalized_reward(
            chosen_log_probs, chosen_attention_mask
        )
        rejected_rewards = self.compute_length_normalized_reward(
            rejected_log_probs, rejected_attention_mask
        )
        
        # SimPO的核心: 使用目标奖励差距
        # L_SimPO = -log(sigmoid(beta * ((r_chosen - r_rejected) - gamma)))
        reward_diff = chosen_rewards - rejected_rewards
        
        if self.use_target_reward:
            # 应用目标奖励差距gamma
            reward_diff = reward_diff - self.gamma
        
        # Bradley-Terry模型
        logits = self.beta * reward_diff
        
        if self.label_smoothing > 0:
            # 标签平滑
            loss = -(
                (1 - self.label_smoothing) * F.logsigmoid(logits) +
                self.label_smoothing * F.logsigmoid(-logits)
            ).mean()
        else:
            loss = -F.logsigmoid(logits).mean()
        
        # 计算准确率
        with torch.no_grad():
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
            reward_margin = reward_diff.mean()
        
        return {
            'loss': loss,
            'chosen_rewards': chosen_rewards.mean(),
            'rejected_rewards': rejected_rewards.mean(),
            'reward_diff': reward_diff.mean(),
            'accuracy': accuracy,
            'reward_margin': reward_margin,
            'chosen_log_probs': chosen_log_probs.mean(),
            'rejected_log_probs': rejected_log_probs.mean()
        }


class SimPOTrainer:
    """
    SimPO训练器
    简化版DPO，无需参考模型
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Optional[SimPOConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = config or SimPOConfig()
        self.device = device
        
        # 初始化损失函数
        self.loss_fn = SimPOLoss(
            beta=self.config.beta,
            gamma=self.config.gamma,
            length_normalization=self.config.length_normalization,
            label_smoothing=self.config.label_smoothing,
            use_target_reward=self.config.use_target_reward
        )
        
        # 初始化优化器
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=1e-6,  # SimPO通常使用较小的学习率
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer
        
        self.global_step = 0
    
    def prepare_batch(
        self,
        batch: Dict[str, List[str]]
    ) -> Dict[str, torch.Tensor]:
        """
        准备批次数据
        
        Args:
            batch: 包含prompt、chosen、rejected的批次
            
        Returns:
            编码后的张量
        """
        prompts = batch['prompt']
        chosen_responses = batch['chosen']
        rejected_responses = batch['rejected']
        
        # 编码chosen序列
        chosen_texts = [p + c for p, c in zip(prompts, chosen_responses)]
        chosen_encoded = self.tokenizer(
            chosen_texts,
            max_length=self.config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # 编码rejected序列
        rejected_texts = [p + r for p, r in zip(prompts, rejected_responses)]
        rejected_encoded = self.tokenizer(
            rejected_texts,
            max_length=self.config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'chosen_input_ids': chosen_encoded['input_ids'].to(self.device),
            'chosen_attention_mask': chosen_encoded['attention_mask'].to(self.device),
            'rejected_input_ids': rejected_encoded['input_ids'].to(self.device),
            'rejected_attention_mask': rejected_encoded['attention_mask'].to(self.device)
        }
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        执行单步训练
        
        Args:
            batch: 批次数据
            
        Returns:
            损失字典
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 前向传播 - chosen
        chosen_outputs = self.model(
            input_ids=batch['chosen_input_ids'],
            attention_mask=batch['chosen_attention_mask']
        )
        chosen_logits = chosen_outputs.logits
        
        # 前向传播 - rejected
        rejected_outputs = self.model(
            input_ids=batch['rejected_input_ids'],
            attention_mask=batch['rejected_attention_mask']
        )
        rejected_logits = rejected_outputs.logits
        
        # 计算损失
        loss_dict = self.loss_fn(
            policy_chosen_logits=chosen_logits,
            policy_rejected_logits=rejected_logits,
            chosen_labels=batch['chosen_input_ids'],
            rejected_labels=batch['rejected_input_ids'],
            chosen_attention_mask=batch['chosen_attention_mask'],
            rejected_attention_mask=batch['rejected_attention_mask']
        )
        
        # 反向传播
        loss_dict['loss'].backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # 更新参数
        self.optimizer.step()
        
        self.global_step += 1
        
        # 返回标量值
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
    
    def train(
        self,
        train_dataset: Dataset,
        num_epochs: int = 3,
        batch_size: int = 4,
        eval_dataset: Optional[Dataset] = None,
        eval_steps: int = 500,
        save_steps: int = 1000,
        logging_steps: int = 10
    ):
        """
        训练模型
        
        Args:
            train_dataset: 训练数据集
            num_epochs: 训练轮数
            batch_size: 批次大小
            eval_dataset: 评估数据集
            eval_steps: 评估间隔
            save_steps: 保存间隔
            logging_steps: 日志间隔
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        total_steps = len(train_loader) * num_epochs
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            
            for step, batch in enumerate(train_loader):
                batch = self.prepare_batch(batch)
                loss_dict = self.train_step(batch)
                
                epoch_loss += loss_dict['loss']
                epoch_accuracy += loss_dict['accuracy']
                
                if self.global_step % logging_steps == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Step {step+1}/{len(train_loader)}, "
                          f"Loss: {loss_dict['loss']:.4f}, Accuracy: {loss_dict['accuracy']:.4f}, "
                          f"Reward Margin: {loss_dict['reward_margin']:.4f}")
                
                if eval_dataset is not None and self.global_step % eval_steps == 0:
                    eval_metrics = self.evaluate(eval_dataset, batch_size)
                    print(f"Evaluation at step {self.global_step}: {eval_metrics}")
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            avg_epoch_accuracy = epoch_accuracy / len(train_loader)
            print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}, "
                  f"Average accuracy: {avg_epoch_accuracy:.4f}")
    
    def evaluate(
        self,
        eval_dataset: Dataset,
        batch_size: int = 4
    ) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            eval_dataset: 评估数据集
            batch_size: 批次大小
            
        Returns:
            评估指标
        """
        self.model.eval()
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            collate_fn=self._collate_fn
        )
        
        total_loss = 0.0
        total_accuracy = 0.0
        total_reward_margin = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                batch = self.prepare_batch(batch)
                
                # 前向传播
                chosen_outputs = self.model(
                    input_ids=batch['chosen_input_ids'],
                    attention_mask=batch['chosen_attention_mask']
                )
                rejected_outputs = self.model(
                    input_ids=batch['rejected_input_ids'],
                    attention_mask=batch['rejected_attention_mask']
                )
                
                # 计算损失
                loss_dict = self.loss_fn(
                    policy_chosen_logits=chosen_outputs.logits,
                    policy_rejected_logits=rejected_outputs.logits,
                    chosen_labels=batch['chosen_input_ids'],
                    rejected_labels=batch['rejected_input_ids'],
                    chosen_attention_mask=batch['chosen_attention_mask'],
                    rejected_attention_mask=batch['rejected_attention_mask']
                )
                
                total_loss += loss_dict['loss'].item()
                total_accuracy += loss_dict['accuracy'].item()
                total_reward_margin += loss_dict['reward_margin'].item()
                num_batches += 1
        
        return {
            'eval_loss': total_loss / num_batches,
            'eval_accuracy': total_accuracy / num_batches,
            'eval_reward_margin': total_reward_margin / num_batches
        }
    
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, List]:
        """自定义collate函数"""
        return {
            'prompt': [item['prompt'] for item in batch],
            'chosen': [item['chosen'] for item in batch],
            'rejected': [item['rejected'] for item in batch]
        }
    
    def save_model(self, save_path: str):
        """保存模型"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)


class SimPODataset(Dataset):
    """
    SimPO数据集
    包含prompt、chosen response、rejected response
    """
    
    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.data[idx]


def create_simpo_dataset_from_dicts(
    data_dicts: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048
) -> SimPODataset:
    """
    从字典列表创建SimPO数据集
    
    Args:
        data_dicts: 数据字典列表，每个包含'prompt', 'chosen', 'rejected'
        tokenizer: 分词器
        max_length: 最大序列长度
        
    Returns:
        SimPODataset实例
    """
    return SimPODataset(data_dicts, tokenizer, max_length)


def compute_simpo_metrics(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
    gamma: float = 1.0
) -> Dict[str, float]:
    """
    计算SimPO相关指标
    
    Args:
        chosen_rewards: 偏好响应的奖励
        rejected_rewards: 非偏好响应的奖励
        gamma: 目标奖励差距
        
    Returns:
        指标字典
    """
    # 计算准确率
    correct = (chosen_rewards > rejected_rewards).float().mean()
    
    # 计算边际
    margin = (chosen_rewards - rejected_rewards).mean()
    
    # 计算满足目标差距的比例
    target_met = ((chosen_rewards - rejected_rewards) > gamma).float().mean()
    
    return {
        'accuracy': correct.item(),
        'margin': margin.item(),
        'target_met_ratio': target_met.item(),
        'chosen_avg_reward': chosen_rewards.mean().item(),
        'rejected_avg_reward': rejected_rewards.mean().item()
    }


# 与DPO的对比函数
def compare_simpo_vs_dpo(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_data: List[Dict[str, str]],
    beta_simpo: float = 2.0,
    beta_dpo: float = 0.1
) -> Dict[str, Any]:
    """
    对比SimPO和DPO的性能
    
    Args:
        model: 模型
        tokenizer: 分词器
        test_data: 测试数据
        beta_simpo: SimPO的beta参数
        beta_dpo: DPO的beta参数
        
    Returns:
        对比结果
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    results = {
        'simpo': [],
        'dpo': []
    }
    
    with torch.no_grad():
        for item in test_data:
            prompt = item['prompt']
            chosen = item['chosen']
            rejected = item['rejected']
            
            # 编码文本
            chosen_text = prompt + chosen
            rejected_text = prompt + rejected
            
            chosen_encoded = tokenizer(
                chosen_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(device)
            
            rejected_encoded = tokenizer(
                rejected_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(device)
            
            # 获取logits
            chosen_outputs = model(**chosen_encoded)
            rejected_outputs = model(**rejected_encoded)
            
            # 计算SimPO奖励（长度归一化）
            chosen_len = chosen_encoded['attention_mask'].sum().item()
            rejected_len = rejected_encoded['attention_mask'].sum().item()
            
            # 简化的奖励计算
            simpo_chosen_reward = chosen_outputs.logits.mean().item() / chosen_len
            simpo_rejected_reward = rejected_outputs.logits.mean().item() / rejected_len
            
            results['simpo'].append({
                'chosen_reward': simpo_chosen_reward,
                'rejected_reward': simpo_rejected_reward,
                'margin': simpo_chosen_reward - simpo_rejected_reward
            })
            
            # DPO奖励（不归一化）
            dpo_chosen_reward = chosen_outputs.logits.mean().item()
            dpo_rejected_reward = rejected_outputs.logits.mean().item()
            
            results['dpo'].append({
                'chosen_reward': dpo_chosen_reward,
                'rejected_reward': dpo_rejected_reward,
                'margin': dpo_chosen_reward - dpo_rejected_reward
            })
    
    # 计算平均指标
    simpo_margins = [r['margin'] for r in results['simpo']]
    dpo_margins = [r['margin'] for r in results['dpo']]
    
    return {
        'simpo_avg_margin': np.mean(simpo_margins),
        'dpo_avg_margin': np.mean(dpo_margins),
        'simpo_accuracy': np.mean([m > 0 for m in simpo_margins]),
        'dpo_accuracy': np.mean([m > 0 for m in dpo_margins]),
        'detailed_results': results
    }
