"""
KTO: Kahneman-Tversky Optimization
卡尼曼-特沃斯基优化算法实现

基于论文 "Human-Centered Loss Functions (HALO) Part 2: KTO"
无需成对偏好数据，使用二元反馈信号（好/坏）
基于前景理论（Prospect Theory）建模人类偏好
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
class KTOConfig:
    """KTO配置参数"""
    beta: float = 0.1  # 温度参数
    desirable_weight: float = 1.0  # 正面样本权重
    undesirable_weight: float = 1.0  # 负面样本权重
    lambda_val: float = 1.0  # 损失平衡系数
    reference_model_present: bool = False  # 是否使用参考模型
    max_length: int = 2048  # 最大序列长度
    kl_penalty: float = 0.0  # KL散度惩罚系数


class KTOLoss(nn.Module):
    """
    KTO损失函数
    基于前景理论，使用二元反馈信号
    """
    
    def __init__(
        self,
        beta: float = 0.1,
        desirable_weight: float = 1.0,
        undesirable_weight: float = 1.0,
        lambda_val: float = 1.0,
        reference_model_present: bool = False,
        kl_penalty: float = 0.0
    ):
        super().__init__()
        self.beta = beta
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight
        self.lambda_val = lambda_val
        self.reference_model_present = reference_model_present
        self.kl_penalty = kl_penalty
    
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
    
    def compute_utility(
        self,
        policy_log_probs: torch.Tensor,
        reference_log_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算效用值（基于前景理论）
        
        Args:
            policy_log_probs: 策略模型的对数概率
            reference_log_probs: 参考模型的对数概率（可选）
            
        Returns:
            效用值
        """
        if self.reference_model_present and reference_log_probs is not None:
            # 使用参考模型计算相对效用
            utility = policy_log_probs - reference_log_probs
        else:
            # 直接使用策略模型的对数概率作为效用
            utility = policy_log_probs
        
        return utility
    
    def compute_kto_loss(
        self,
        policy_log_probs: torch.Tensor,
        is_desirable: torch.Tensor,
        reference_log_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算KTO损失
        
        基于前景理论:
        - 对于正面样本（desirable）：最大化效用
        - 对于负面样本（undesirable）：最小化效用
        
        Args:
            policy_log_probs: 策略模型的对数概率
            is_desirable: 是否为正面样本 [batch_size]
            reference_log_probs: 参考模型的对数概率（可选）
            
        Returns:
            KTO损失
        """
        # 计算效用
        utility = self.compute_utility(policy_log_probs, reference_log_probs)
        
        # 分离正面和负面样本
        desirable_mask = is_desirable.bool()
        undesirable_mask = ~desirable_mask
        
        loss = torch.tensor(0.0, device=policy_log_probs.device)
        
        # 正面样本损失：我们希望utility > 0
        if desirable_mask.any():
            desirable_utility = utility[desirable_mask]
            # 使用sigmoid损失: -log(sigmoid(beta * utility))
            desirable_loss = -F.logsigmoid(self.beta * desirable_utility).mean()
            loss = loss + self.desirable_weight * desirable_loss
        
        # 负面样本损失：我们希望utility < 0
        if undesirable_mask.any():
            undesirable_utility = utility[undesirable_mask]
            # 使用sigmoid损失: -log(sigmoid(-beta * utility))
            undesirable_loss = -F.logsigmoid(-self.beta * undesirable_utility).mean()
            loss = loss + self.lambda_val * self.undesirable_weight * undesirable_loss
        
        return loss
    
    def forward(
        self,
        policy_logits: torch.Tensor,
        labels: torch.Tensor,
        is_desirable: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        reference_logits: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播计算KTO损失
        
        Args:
            policy_logits: 策略模型的logits
            labels: 标签序列
            is_desirable: 是否为正面样本
            attention_mask: 注意力掩码
            reference_logits: 参考模型的logits（可选）
            
        Returns:
            包含各项损失的字典
        """
        # 计算策略模型的对数概率
        policy_log_probs = self.compute_sequence_log_probs(
            policy_logits, labels, attention_mask
        )
        
        # 计算参考模型的对数概率（如果提供）
        reference_log_probs = None
        if self.reference_model_present and reference_logits is not None:
            reference_log_probs = self.compute_sequence_log_probs(
                reference_logits, labels, attention_mask
            )
        
        # 计算KTO损失
        kto_loss = self.compute_kto_loss(
            policy_log_probs, is_desirable, reference_log_probs
        )
        
        # 计算KL散度惩罚（如果使用参考模型）
        kl_loss = torch.tensor(0.0, device=policy_logits.device)
        if self.kl_penalty > 0 and reference_log_probs is not None:
            kl_div = (policy_log_probs - reference_log_probs).pow(2).mean()
            kl_loss = self.kl_penalty * kl_div
        
        # 总损失
        total_loss = kto_loss + kl_loss
        
        # 计算指标
        with torch.no_grad():
            utility = self.compute_utility(policy_log_probs, reference_log_probs)
            desirable_accuracy = (utility[is_desirable.bool()] > 0).float().mean() if is_desirable.any() else torch.tensor(0.0)
            undesirable_accuracy = (utility[~is_desirable.bool()] < 0).float().mean() if (~is_desirable.bool()).any() else torch.tensor(0.0)
            avg_utility = utility.mean()
        
        return {
            'loss': total_loss,
            'kto_loss': kto_loss,
            'kl_loss': kl_loss,
            'avg_utility': avg_utility,
            'desirable_accuracy': desirable_accuracy,
            'undesirable_accuracy': undesirable_accuracy,
            'avg_log_prob': policy_log_probs.mean()
        }


class KTOTrainer:
    """
    KTO训练器
    使用二元反馈信号训练模型
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Optional[KTOConfig] = None,
        reference_model: Optional[PreTrainedModel] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = config or KTOConfig()
        self.device = device
        
        # 参考模型（可选）
        self.reference_model = None
        if reference_model is not None:
            self.reference_model = reference_model.to(device)
            self.reference_model.eval()
            self.config.reference_model_present = True
        
        # 初始化损失函数
        self.loss_fn = KTOLoss(
            beta=self.config.beta,
            desirable_weight=self.config.desirable_weight,
            undesirable_weight=self.config.undesirable_weight,
            lambda_val=self.config.lambda_val,
            reference_model_present=self.config.reference_model_present,
            kl_penalty=self.config.kl_penalty
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
        batch: Dict[str, List]
    ) -> Dict[str, torch.Tensor]:
        """
        准备批次数据
        
        Args:
            batch: 包含text、is_desirable的批次
            
        Returns:
            编码后的张量
        """
        texts = batch['text']
        is_desirable = batch['is_desirable']
        
        # 编码文本
        encoded = self.tokenizer(
            texts,
            max_length=self.config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device),
            'is_desirable': torch.tensor(is_desirable, dtype=torch.float32).to(self.device)
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
        
        # 前向传播 - 策略模型
        policy_outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        policy_logits = policy_outputs.logits
        
        # 前向传播 - 参考模型（如果提供）
        reference_logits = None
        if self.reference_model is not None:
            with torch.no_grad():
                reference_outputs = self.reference_model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                reference_logits = reference_outputs.logits
        
        # 计算损失
        loss_dict = self.loss_fn(
            policy_logits=policy_logits,
            labels=batch['input_ids'],
            is_desirable=batch['is_desirable'],
            attention_mask=batch['attention_mask'],
            reference_logits=reference_logits
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
            epoch_desirable_acc = 0.0
            epoch_undesirable_acc = 0.0
            
            for step, batch in enumerate(train_loader):
                batch = self.prepare_batch(batch)
                loss_dict = self.train_step(batch)
                
                epoch_loss += loss_dict['loss']
                epoch_desirable_acc += loss_dict['desirable_accuracy']
                epoch_undesirable_acc += loss_dict['undesirable_accuracy']
                
                if self.global_step % logging_steps == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Step {step+1}/{len(train_loader)}, "
                          f"Loss: {loss_dict['loss']:.4f}, "
                          f"Desirable Acc: {loss_dict['desirable_accuracy']:.4f}, "
                          f"Undesirable Acc: {loss_dict['undesirable_accuracy']:.4f}, "
                          f"Avg Utility: {loss_dict['avg_utility']:.4f}")
                
                if eval_dataset is not None and self.global_step % eval_steps == 0:
                    eval_metrics = self.evaluate(eval_dataset, batch_size)
                    print(f"Evaluation at step {self.global_step}: {eval_metrics}")
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            avg_desirable_acc = epoch_desirable_acc / len(train_loader)
            avg_undesirable_acc = epoch_undesirable_acc / len(train_loader)
            print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}, "
                  f"Desirable Acc: {avg_desirable_acc:.4f}, "
                  f"Undesirable Acc: {avg_undesirable_acc:.4f}")
    
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
        total_desirable_acc = 0.0
        total_undesirable_acc = 0.0
        total_utility = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                batch = self.prepare_batch(batch)
                
                # 前向传播
                policy_outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                reference_logits = None
                if self.reference_model is not None:
                    reference_outputs = self.reference_model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask']
                    )
                    reference_logits = reference_outputs.logits
                
                # 计算损失
                loss_dict = self.loss_fn(
                    policy_logits=policy_outputs.logits,
                    labels=batch['input_ids'],
                    is_desirable=batch['is_desirable'],
                    attention_mask=batch['attention_mask'],
                    reference_logits=reference_logits
                )
                
                total_loss += loss_dict['loss'].item()
                total_desirable_acc += loss_dict['desirable_accuracy'].item()
                total_undesirable_acc += loss_dict['undesirable_accuracy'].item()
                total_utility += loss_dict['avg_utility'].item()
                num_batches += 1
        
        return {
            'eval_loss': total_loss / num_batches,
            'eval_desirable_accuracy': total_desirable_acc / num_batches,
            'eval_undesirable_accuracy': total_undesirable_acc / num_batches,
            'eval_avg_utility': total_utility / num_batches
        }
    
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, List]:
        """自定义collate函数"""
        return {
            'text': [item['text'] for item in batch],
            'is_desirable': [item['is_desirable'] for item in batch]
        }
    
    def save_model(self, save_path: str):
        """保存模型"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)


class KTODataset(Dataset):
    """
    KTO数据集
    包含文本和二元反馈信号（好/坏）
    """
    
    def __init__(
        self,
        data: List[Dict[str, Union[str, bool]]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[str, bool]]:
        return self.data[idx]


def create_kto_dataset_from_dicts(
    data_dicts: List[Dict[str, Union[str, bool]]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048
) -> KTODataset:
    """
    从字典列表创建KTO数据集
    
    Args:
        data_dicts: 数据字典列表，每个包含'text', 'is_desirable'
        tokenizer: 分词器
        max_length: 最大序列长度
        
    Returns:
        KTODataset实例
    """
    return KTODataset(data_dicts, tokenizer, max_length)


def convert_preference_to_kto(
    preference_data: List[Dict[str, str]]
) -> List[Dict[str, Union[str, bool]]]:
    """
    将成对偏好数据转换为KTO格式
    
    Args:
        preference_data: 成对偏好数据，包含'prompt', 'chosen', 'rejected'
        
    Returns:
        KTO格式数据
    """
    kto_data = []
    
    for item in preference_data:
        prompt = item['prompt']
        
        # 正面样本
        kto_data.append({
            'text': prompt + item['chosen'],
            'is_desirable': True
        })
        
        # 负面样本
        kto_data.append({
            'text': prompt + item['rejected'],
            'is_desirable': False
        })
    
    return kto_data


def compute_kto_metrics(
    utilities: torch.Tensor,
    is_desirable: torch.Tensor
) -> Dict[str, float]:
    """
    计算KTO相关指标
    
    Args:
        utilities: 效用值
        is_desirable: 是否为正面样本
        
    Returns:
        指标字典
    """
    desirable_mask = is_desirable.bool()
    undesirable_mask = ~desirable_mask
    
    metrics = {}
    
    # 正面样本准确率（utility > 0）
    if desirable_mask.any():
        metrics['desirable_accuracy'] = (utilities[desirable_mask] > 0).float().mean().item()
        metrics['desirable_avg_utility'] = utilities[desirable_mask].mean().item()
    
    # 负面样本准确率（utility < 0）
    if undesirable_mask.any():
        metrics['undesirable_accuracy'] = (utilities[undesirable_mask] < 0).float().mean().item()
        metrics['undesirable_avg_utility'] = utilities[undesirable_mask].mean().item()
    
    # 总体指标
    metrics['overall_accuracy'] = (
        (utilities[desirable_mask] > 0).sum() + (utilities[undesirable_mask] < 0).sum()
    ).float() / len(utilities)
    metrics['avg_utility'] = utilities.mean().item()
    
    return metrics


# KTO与DPO的对比分析
def analyze_kto_vs_dpo(
    kto_utilities: torch.Tensor,
    dpo_rewards: torch.Tensor,
    is_desirable: torch.Tensor
) -> Dict[str, Any]:
    """
    分析KTO与DPO的差异
    
    Args:
        kto_utilities: KTO计算的效用值
        dpo_rewards: DPO计算的奖励值
        is_desirable: 是否为正面样本
        
    Returns:
        分析结果
    """
    desirable_mask = is_desirable.bool()
    
    analysis = {
        'kto': {
            'avg_utility': kto_utilities.mean().item(),
            'desirable_avg': kto_utilities[desirable_mask].mean().item(),
            'undesirable_avg': kto_utilities[~desirable_mask].mean().item(),
            'separation': (kto_utilities[desirable_mask].mean() - kto_utilities[~desirable_mask].mean()).item()
        },
        'dpo': {
            'avg_reward': dpo_rewards.mean().item(),
            'desirable_avg': dpo_rewards[desirable_mask].mean().item(),
            'undesirable_avg': dpo_rewards[~desirable_mask].mean().item(),
            'separation': (dpo_rewards[desirable_mask].mean() - dpo_rewards[~desirable_mask].mean()).item()
        }
    }
    
    # 计算相关性
    analysis['correlation'] = torch.corrcoef(
        torch.stack([kto_utilities, dpo_rewards])
    )[0, 1].item()
    
    return analysis
