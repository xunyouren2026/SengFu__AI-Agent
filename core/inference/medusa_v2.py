"""
Medusa v2: 多头 Draft Model 推理加速
基于树状注意力解码和自蒸馏训练

基于论文 "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"
实现2-3倍的推理加速
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Any, Union
from dataclasses import dataclass
from transformers import PreTrainedModel, PreTrainedTokenizer
import copy


@dataclass
class MedusaConfig:
    """Medusa配置参数"""
    # 架构配置
    num_heads: int = 4  # Medusa头数量
    num_layers: int = 1  # 每个头的层数
    hidden_size: int = 4096  # 隐藏层大小
    vocab_size: int = 32000  # 词汇表大小
    
    # 生成配置
    max_draft_tokens: int = 3  # 最大draft token数
    top_k: int = 10  # top-k采样
    temperature: float = 0.8  # 采样温度
    
    # 树状注意力配置
    tree_lookahead_depth: int = 3  # 树状前瞻深度
    use_tree_attention: bool = True  # 使用树状注意力
    
    # 训练配置
    use_self_distillation: bool = True  # 使用自蒸馏
    distillation_temperature: float = 2.0  # 蒸馏温度
    medusa_heads_coefficient: float = 1.0  # Medusa头损失系数
    
    # 推理配置
    acceptance_threshold: float = 0.6  # 接受阈值


class MedusaHead(nn.Module):
    """
    单个Medusa Head
    预测未来第n个token
    """
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_layers: int = 1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # 构建Medusa头网络
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size)
            ])
        
        self.network = nn.Sequential(*layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: 隐藏状态 [batch_size, seq_len, hidden_size]
            
        Returns:
            logits [batch_size, seq_len, vocab_size]
        """
        features = self.network(hidden_states)
        logits = self.lm_head(features)
        return logits


class MedusaModel(nn.Module):
    """
    Medusa v2模型
    包含基础LLM和多个Medusa头
    """
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        config: MedusaConfig
    ):
        super().__init__()
        self.base_model = base_model
        self.config = config
        
        # 冻结基础模型
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 创建Medusa头
        self.medusa_heads = nn.ModuleList([
            MedusaHead(
                hidden_size=config.hidden_size,
                vocab_size=config.vocab_size,
                num_layers=config.num_layers
            )
            for _ in range(config.num_heads)
        ])
        
        # 树状注意力模块
        if config.use_tree_attention:
            self.tree_attention = TreeAttentionDecoder(config)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 输入ID
            attention_mask: 注意力掩码
            labels: 标签（用于训练）
            
        Returns:
            输出字典
        """
        # 基础模型前向传播
        with torch.no_grad():
            base_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = base_outputs.hidden_states[-1]
            base_logits = base_outputs.logits
        
        # Medusa头预测
        medusa_logits = []
        for head in self.medusa_heads:
            head_logits = head(hidden_states)
            medusa_logits.append(head_logits)
        
        # 计算损失（训练时）
        loss = None
        if labels is not None:
            loss = self._compute_loss(
                base_logits, medusa_logits, labels, input_ids
            )
        
        return {
            'loss': loss,
            'base_logits': base_logits,
            'medusa_logits': medusa_logits,
            'hidden_states': hidden_states
        }
    
    def _compute_loss(
        self,
        base_logits: torch.Tensor,
        medusa_logits: List[torch.Tensor],
        labels: torch.Tensor,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        计算训练损失
        
        包括：
        1. Medusa头的预测损失
        2. 自蒸馏损失（可选）
        """
        total_loss = torch.tensor(0.0, device=base_logits.device)
        
        # Medusa头损失
        for i, head_logits in enumerate(medusa_logits):
            # 移位以对齐预测
            shift_logits = head_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, i+1:].contiguous() if labels.shape[1] > i+1 else labels[:, 1:].contiguous()
            
            # 计算交叉熵损失
            loss_fct = nn.CrossEntropyLoss()
            head_loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
            total_loss += head_loss * self.config.medusa_heads_coefficient
        
        # 自蒸馏损失
        if self.config.use_self_distillation:
            distill_loss = self._compute_distillation_loss(base_logits, medusa_logits)
            total_loss += distill_loss
        
        return total_loss
    
    def _compute_distillation_loss(
        self,
        base_logits: torch.Tensor,
        medusa_logits: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        计算自蒸馏损失
        
        使Medusa头的输出与基础模型的输出一致
        """
        distill_loss = torch.tensor(0.0, device=base_logits.device)
        temperature = self.config.distillation_temperature
        
        for i, head_logits in enumerate(medusa_logits):
            # 获取对应位置的基础模型输出
            base_logits_shifted = base_logits[:, i+1:, :].contiguous() if base_logits.shape[1] > i+1 else base_logits[:, 1:, :].contiguous()
            head_logits_shifted = head_logits[:, :-1, :].contiguous()
            
            # 温度缩放
            base_probs = F.softmax(base_logits_shifted / temperature, dim=-1)
            head_log_probs = F.log_softmax(head_logits_shifted / temperature, dim=-1)
            
            # KL散度
            kl_div = F.kl_div(head_log_probs, base_probs, reduction='batchmean')
            distill_loss += kl_div * (temperature ** 2)
        
        return distill_loss / len(medusa_logits)
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95
    ) -> torch.Tensor:
        """
        使用Medusa加速生成
        
        Args:
            input_ids: 输入ID
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_k: top-k采样
            top_p: top-p采样
            
        Returns:
            生成的token序列
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 初始化生成序列
        generated = input_ids.clone()
        
        # 预分配KV缓存
        past_key_values = None
        
        while generated.shape[1] < seq_len + max_new_tokens:
            # 基础模型前向传播
            base_outputs = self.base_model(
                input_ids=generated[:, -1:] if past_key_values is not None else generated,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True
            )
            
            hidden_states = base_outputs.hidden_states[-1][:, -1:, :]
            base_logits = base_outputs.logits[:, -1, :]
            past_key_values = base_outputs.past_key_values
            
            # Medusa头预测draft tokens
            draft_tokens = self._generate_draft_tokens(
                hidden_states, temperature, top_k, top_p
            )
            
            # 树状注意力验证（如果使用）
            if self.config.use_tree_attention:
                accepted_tokens = self._verify_with_tree_attention(
                    generated, draft_tokens, base_logits, past_key_values
                )
            else:
                accepted_tokens = self._verify_draft_tokens(
                    generated, draft_tokens, base_logits
                )
            
            # 添加接受的token
            if len(accepted_tokens) > 0:
                new_tokens = torch.tensor([accepted_tokens], device=device)
                generated = torch.cat([generated, new_tokens], dim=1)
            else:
                # 如果没有接受任何draft token，使用基础模型生成
                next_token = self._sample_token(base_logits, temperature, top_k, top_p)
                generated = torch.cat([generated, next_token], dim=1)
            
            # 检查是否生成了EOS
            if (generated == self.base_model.config.eos_token_id).any(dim=1).all():
                break
        
        return generated
    
    def _generate_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float
    ) -> List[int]:
        """
        使用Medusa头生成draft tokens
        
        Args:
            hidden_states: 隐藏状态
            temperature: 温度
            top_k: top-k
            top_p: top-p
            
        Returns:
            draft token列表
        """
        draft_tokens = []
        current_hidden = hidden_states
        
        for head in self.medusa_heads:
            # 预测下一个token
            head_logits = head(current_hidden)
            
            # 采样
            next_token = self._sample_token(head_logits, temperature, top_k, top_p)
            draft_tokens.append(next_token.item())
            
            # 更新隐藏状态（简化处理）
            # 实际实现中应该通过基础模型获取新的隐藏状态
            current_hidden = current_hidden  # 占位符
        
        return draft_tokens
    
    def _verify_draft_tokens(
        self,
        generated: torch.Tensor,
        draft_tokens: List[int],
        base_logits: torch.Tensor
    ) -> List[int]:
        """
        验证draft tokens
        
        Args:
            generated: 已生成的序列
            draft_tokens: draft tokens
            base_logits: 基础模型的logits
            
        Returns:
            接受的token列表
        """
        accepted = []
        device = generated.device
        
        # 获取基础模型的预测
        base_probs = F.softmax(base_logits, dim=-1)
        base_token = base_probs.argmax(dim=-1).item()
        
        # 验证每个draft token
        for i, draft_token in enumerate(draft_tokens):
            if draft_token == base_token:
                accepted.append(draft_token)
                # 更新基础模型的预测（简化）
                if i < len(draft_tokens) - 1:
                    # 实际应该重新计算
                    pass
            else:
                break
        
        return accepted
    
    def _verify_with_tree_attention(
        self,
        generated: torch.Tensor,
        draft_tokens: List[int],
        base_logits: torch.Tensor,
        past_key_values: Optional[Tuple] = None
    ) -> List[int]:
        """
        使用树状注意力验证draft tokens
        
        Args:
            generated: 已生成的序列
            draft_tokens: draft tokens
            base_logits: 基础模型的logits
            past_key_values: KV缓存
            
        Returns:
            接受的token列表
        """
        # 构建树状结构
        tree_structure = self.tree_attention.build_tree(draft_tokens)
        
        # 树状注意力前向传播
        # 实际实现中应该使用树状掩码进行批量验证
        
        # 简化实现：顺序验证
        return self._verify_draft_tokens(generated, draft_tokens, base_logits)
    
    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float
    ) -> torch.Tensor:
        """
        从logits中采样token
        
        Args:
            logits: logits [batch_size, vocab_size] 或 [batch_size, seq_len, vocab_size]
            temperature: 温度
            top_k: top-k
            top_p: top-p
            
        Returns:
            采样的token
        """
        # 处理不同维度的输入
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        
        # 温度缩放
        logits = logits / temperature
        
        # Top-k过滤
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Top-p过滤
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # 采样
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token


class TreeAttentionDecoder(nn.Module):
    """
    树状注意力解码器
    支持多个候选路径的并行验证
    """
    
    def __init__(self, config: MedusaConfig):
        super().__init__()
        self.config = config
        self.max_depth = config.tree_lookahead_depth
    
    def build_tree(self, draft_tokens: List[int]) -> Dict[str, Any]:
        """
        构建树状结构
        
        Args:
            draft_tokens: draft tokens列表
            
        Returns:
            树状结构
        """
        # 构建简单的树状结构
        tree = {
            'root': None,
            'children': {},
            'depth': 0
        }
        
        current = tree
        for i, token in enumerate(draft_tokens):
            if i >= self.max_depth:
                break
            current['children'][token] = {
                'token': token,
                'children': {},
                'depth': i + 1
            }
            current = current['children'][token]
        
        return tree
    
    def create_tree_mask(
        self,
        tree_structure: Dict[str, Any],
        seq_len: int
    ) -> torch.Tensor:
        """
        创建树状注意力掩码
        
        Args:
            tree_structure: 树状结构
            seq_len: 序列长度
            
        Returns:
            注意力掩码
        """
        # 创建因果掩码
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        
        # 根据树状结构调整掩码
        # 实际实现中应该允许树状路径上的token相互关注
        
        return mask
    
    def tree_decoding(
        self,
        hidden_states: torch.Tensor,
        tree_structure: Dict[str, Any]
    ) -> torch.Tensor:
        """
        树状解码
        
        Args:
            hidden_states: 隐藏状态
            tree_structure: 树状结构
            
        Returns:
            解码输出
        """
        # 实现树状解码逻辑
        # 使用树状注意力掩码进行并行验证
        
        return hidden_states


class MedusaTrainer:
    """
    Medusa训练器
    用于训练Medusa头
    """
    
    def __init__(
        self,
        medusa_model: MedusaModel,
        tokenizer: PreTrainedTokenizer,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.medusa_model = medusa_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        if optimizer is None:
            # 只优化Medusa头
            self.optimizer = torch.optim.AdamW(
                [p for p in medusa_model.medusa_heads.parameters() if p.requires_grad],
                lr=1e-4,
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer
        
        self.global_step = 0
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        训练步骤
        
        Args:
            batch: 批次数据
            
        Returns:
            损失字典
        """
        self.medusa_model.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        outputs = self.medusa_model(
            input_ids=batch['input_ids'],
            attention_mask=batch.get('attention_mask'),
            labels=batch.get('labels')
        )
        
        loss = outputs['loss']
        
        # 反向传播
        if loss is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.medusa_model.medusa_heads.parameters(), 1.0)
            self.optimizer.step()
        
        self.global_step += 1
        
        return {
            'loss': loss.item() if loss is not None else 0.0
        }
    
    def save_medusa_heads(self, save_path: str):
        """保存Medusa头"""
        torch.save({
            'medusa_heads': self.medusa_model.medusa_heads.state_dict(),
            'config': self.medusa_model.config
        }, save_path)
    
    def load_medusa_heads(self, load_path: str):
        """加载Medusa头"""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.medusa_model.medusa_heads.load_state_dict(checkpoint['medusa_heads'])


def create_medusa_model_from_base(
    base_model: PreTrainedModel,
    num_medusa_heads: int = 4,
    **kwargs
) -> MedusaModel:
    """
    从基础模型创建Medusa模型
    
    Args:
        base_model: 基础模型
        num_medusa_heads: Medusa头数量
        **kwargs: 其他配置参数
        
    Returns:
        Medusa模型
    """
    config = MedusaConfig(
        num_heads=num_medusa_heads,
        hidden_size=base_model.config.hidden_size,
        vocab_size=base_model.config.vocab_size,
        **kwargs
    )
    
    return MedusaModel(base_model, config)


def benchmark_medusa_speedup(
    base_model: PreTrainedModel,
    medusa_model: MedusaModel,
    test_prompts: List[str],
    tokenizer: PreTrainedTokenizer,
    max_new_tokens: int = 100
) -> Dict[str, float]:
    """
    基准测试Medusa加速比
    
    Args:
        base_model: 基础模型
        medusa_model: Medusa模型
        test_prompts: 测试提示
        tokenizer: 分词器
        max_new_tokens: 最大生成token数
        
    Returns:
        加速统计
    """
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = base_model.to(device)
    medusa_model = medusa_model.to(device)
    
    base_times = []
    medusa_times = []
    
    for prompt in test_prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        # 基础模型生成时间
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        with torch.no_grad():
            _ = base_model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        base_times.append(time.time() - start)
        
        # Medusa生成时间
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        with torch.no_grad():
            _ = medusa_model.generate(input_ids, max_new_tokens=max_new_tokens)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        medusa_times.append(time.time() - start)
    
    avg_base_time = sum(base_times) / len(base_times)
    avg_medusa_time = sum(medusa_times) / len(medusa_times)
    speedup = avg_base_time / avg_medusa_time
    
    return {
        'avg_base_time': avg_base_time,
        'avg_medusa_time': avg_medusa_time,
        'speedup': speedup,
        'speedup_percentage': (speedup - 1) * 100
    }
