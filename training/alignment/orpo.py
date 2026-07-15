"""
ORPO: Odds Ratio Preference Optimization
赔率比偏好优化算法实现

基于论文 "ORPO: Monolithic Preference Optimization without Reference Model"
无需参考模型，将SFT和对齐目标结合为单一损失函数
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
class ORPOConfig:
    """ORPO配置参数"""
    lambda_orpo: float = 1.0  # ORPO损失的权重系数
    beta: float = 0.1  # 温度参数
    label_smoothing: float = 0.0  # 标签平滑
    sft_weight: float = 1.0  # SFT损失权重
    orpo_weight: float = 1.0  # ORPO损失权重
    max_length: int = 2048  # 最大序列长度
    padding_side: str = "right"  # 填充方向


class ORPOLoss(nn.Module):
    """
    ORPO损失函数
    结合SFT损失和赔率比偏好优化损失
    """
    
    def __init__(
        self,
        lambda_orpo: float = 1.0,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        sft_weight: float = 1.0,
        orpo_weight: float = 1.0
    ):
        super().__init__()
        self.lambda_orpo = lambda_orpo
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.sft_weight = sft_weight
        self.orpo_weight = orpo_weight
    
    def compute_log_probs(
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
    
    def compute_odds_ratio(
        self,
        chosen_log_probs: torch.Tensor,
        rejected_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        计算赔率比
        
        Args:
            chosen_log_probs: 偏好响应的对数概率
            rejected_log_probs: 非偏好响应的对数概率
            
        Returns:
            赔率比损失
        """
        # 将对数概率转换为概率
        chosen_probs = torch.exp(chosen_log_probs)
        rejected_probs = torch.exp(rejected_log_probs)
        
        # 计算赔率: odds = p / (1 - p)
        chosen_odds = chosen_probs / (1 - chosen_probs + 1e-8)
        rejected_odds = rejected_probs / (1 - rejected_probs + 1e-8)
        
        # 计算赔率比
        odds_ratio = chosen_odds / (rejected_odds + 1e-8)
        
        # 使用log odds ratio作为损失
        log_odds_ratio = torch.log(odds_ratio + 1e-8)
        
        # 我们希望log_odds_ratio > 0，即chosen的odds大于rejected的odds
        # 使用sigmoid损失
        orpo_loss = -F.logsigmoid(self.beta * log_odds_ratio).mean()
        
        return orpo_loss
    
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
        前向传播计算ORPO损失
        
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
        # 计算SFT损失（只在偏好响应上）
        shift_chosen_logits = policy_chosen_logits[..., :-1, :].contiguous()
        shift_chosen_labels = chosen_labels[..., 1:].contiguous()
        
        if chosen_attention_mask is not None:
            shift_chosen_mask = chosen_attention_mask[..., 1:].contiguous()
        else:
            shift_chosen_mask = torch.ones_like(shift_chosen_labels)
        
        sft_loss = F.cross_entropy(
            shift_chosen_logits.view(-1, shift_chosen_logits.size(-1)),
            shift_chosen_labels.view(-1),
            reduction='none'
        )
        sft_loss = (sft_loss * shift_chosen_mask.view(-1)).sum() / shift_chosen_mask.sum()
        
        # 计算对数概率
        chosen_log_probs = self.compute_log_probs(
            policy_chosen_logits, chosen_labels, chosen_attention_mask
        )
        rejected_log_probs = self.compute_log_probs(
            policy_rejected_logits, rejected_labels, rejected_attention_mask
        )
        
        # 计算ORPO损失
        orpo_loss = self.compute_odds_ratio(chosen_log_probs, rejected_log_probs)
        
        # 总损失
        total_loss = self.sft_weight * sft_loss + self.lambda_orpo * self.orpo_weight * orpo_loss
        
        return {
            'loss': total_loss,
            'sft_loss': sft_loss,
            'orpo_loss': orpo_loss,
            'chosen_log_probs': chosen_log_probs.mean(),
            'rejected_log_probs': rejected_log_probs.mean(),
            'log_odds_ratio': (chosen_log_probs - rejected_log_probs).mean()
        }


class ORPOTrainer:
    """
    ORPO训练器
    无需参考模型，直接优化策略模型
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Optional[ORPOConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = config or ORPOConfig()
        self.device = device
        
        # 初始化损失函数
        self.loss_fn = ORPOLoss(
            lambda_orpo=self.config.lambda_orpo,
            beta=self.config.beta,
            label_smoothing=self.config.label_smoothing,
            sft_weight=self.config.sft_weight,
            orpo_weight=self.config.orpo_weight
        )
        
        # 初始化优化器
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=1e-5,
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
            
            for step, batch in enumerate(train_loader):
                batch = self.prepare_batch(batch)
                loss_dict = self.train_step(batch)
                
                epoch_loss += loss_dict['loss']
                
                if self.global_step % logging_steps == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Step {step+1}/{len(train_loader)}, "
                          f"Loss: {loss_dict['loss']:.4f}, SFT: {loss_dict['sft_loss']:.4f}, "
                          f"ORPO: {loss_dict['orpo_loss']:.4f}")
                
                if eval_dataset is not None and self.global_step % eval_steps == 0:
                    eval_metrics = self.evaluate(eval_dataset, batch_size)
                    print(f"Evaluation at step {self.global_step}: {eval_metrics}")
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
    
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
        total_sft_loss = 0.0
        total_orpo_loss = 0.0
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
                total_sft_loss += loss_dict['sft_loss'].item()
                total_orpo_loss += loss_dict['orpo_loss'].item()
                num_batches += 1
        
        return {
            'eval_loss': total_loss / num_batches,
            'eval_sft_loss': total_sft_loss / num_batches,
            'eval_orpo_loss': total_orpo_loss / num_batches
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


class ORPODataset(Dataset):
    """
    ORPO数据集
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


def create_orpo_dataset_from_dicts(
    data_dicts: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048
) -> ORPODataset:
    """
    从字典列表创建ORPO数据集
    
    Args:
        data_dicts: 数据字典列表，每个包含'prompt', 'chosen', 'rejected'
        tokenizer: 分词器
        max_length: 最大序列长度
        
    Returns:
        ORPODataset实例
    """
    return ORPODataset(data_dicts, tokenizer, max_length)


# 辅助函数
def compute_orpo_metrics(
    chosen_log_probs: torch.Tensor,
    rejected_log_probs: torch.Tensor
) -> Dict[str, float]:
    """
    计算ORPO相关指标
    
    Args:
        chosen_log_probs: 偏好响应的对数概率
        rejected_log_probs: 非偏好响应的对数概率
        
    Returns:
        指标字典
    """
    # 计算准确率
    correct = (chosen_log_probs > rejected_log_probs).float().mean()
    
    # 计算边际
    margin = (chosen_log_probs - rejected_log_probs).mean()
    
    return {
        'accuracy': correct.item(),
        'margin': margin.item(),
        'chosen_avg_log_prob': chosen_log_probs.mean().item(),
        'rejected_avg_log_prob': rejected_log_probs.mean().item()
    }
