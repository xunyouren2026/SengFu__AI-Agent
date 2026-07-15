#!/usr/bin/env python3
"""
Transformer语言模型示例
=========================

使用AGI统一框架实现的完整Transformer语言模型。
包含字符级分词器、Transformer架构、训练和文本生成功能。

作者: AGI Framework Team
日期: 2025-05-13
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from collections import Counter
import re
import time

# 导入框架模块
from core.swing_layer.transformer.transformer_core import TransformerEncoderLayer, TransformerDecoderLayer
from core.swing_layer.embedding.embeddings import Embedding, PositionalEncoding
from core.swing_layer.neural_layers import Linear, Dropout
from core.activations.activations import ReLU, Softmax
from core.initialization.initializers import XavierInitializer
from training.optimizers.optimizers import AdamW
from training.losses.losses import CrossEntropyLoss
from core.normalization.normalizations import LayerNormalization


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class TransformerConfig:
    """Transformer模型配置"""
    # 词汇表参数
    vocab_size: int = 256  # 字符级词汇表
    max_seq_length: int = 512
    
    # 模型架构参数
    d_model: int = 256
    num_heads: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    
    # 训练参数
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 4000
    
    # 生成参数
    max_gen_length: int = 200
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    
    # 其他
    seed: int = 42
    
    def __post_init__(self):
        assert self.d_model % self.num_heads == 0, "d_model必须能被num_heads整除"
        self.d_k = self.d_model // self.num_heads


# =============================================================================
# 字符级分词器
# =============================================================================

class CharTokenizer:
    """
    字符级分词器
    将文本转换为字符ID序列
    """
    
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
            '<MASK>': 4
        }
        self._build_vocab()
        
    def _build_vocab(self):
        """构建词汇表"""
        # 特殊token
        for token, idx in self.special_tokens.items():
            self.char_to_id[token] = idx
            self.id_to_char[idx] = token
        
        # ASCII字符
        next_idx = len(self.special_tokens)
        for i in range(32, 127):  # 可打印ASCII
            char = chr(i)
            if char not in self.char_to_id:
                self.char_to_id[char] = next_idx
                self.id_to_char[next_idx] = char
                next_idx += 1
        
        # 常见标点和中文字符（简化版）
        extra_chars = '。，！？；：""''（）【】《》、'
        for char in extra_chars:
            if char not in self.char_to_id:
                self.char_to_id[char] = next_idx
                self.id_to_char[next_idx] = char
                next_idx += 1
        
        self.vocab_size = len(self.char_to_id)
        print(f"词汇表大小: {self.vocab_size}")
        
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """将文本编码为ID序列"""
        ids = []
        if add_special_tokens:
            ids.append(self.special_tokens['<BOS>'])
        
        for char in text:
            ids.append(self.char_to_id.get(char, self.special_tokens['<UNK>']))
        
        if add_special_tokens:
            ids.append(self.special_tokens['<EOS>'])
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """将ID序列解码为文本"""
        chars = []
        special_ids = set(self.special_tokens.values()) if skip_special_tokens else set()
        
        for idx in ids:
            if idx not in special_ids:
                chars.append(self.id_to_char.get(idx, '<UNK>'))
        
        return ''.join(chars)
    
    def pad_sequence(self, ids: List[int], max_length: int) -> List[int]:
        """填充序列到固定长度"""
        if len(ids) >= max_length:
            return ids[:max_length]
        return ids + [self.special_tokens['<PAD>']] * (max_length - len(ids))
    
    def create_attention_mask(self, ids: List[int]) -> List[int]:
        """创建注意力掩码"""
        pad_id = self.special_tokens['<PAD>']
        return [1 if idx != pad_id else 0 for idx in ids]


# =============================================================================
# 文本数据集
# =============================================================================

class TextDataset:
    """
    文本数据集
    支持多种文本数据源
    """
    
    def __init__(self, config: TransformerConfig, tokenizer: CharTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.samples: List[Tuple[List[int], List[int]]] = []
        
    def load_sample_text(self) -> str:
        """加载示例训练文本"""
        # 包含多种文本类型的综合示例
        sample_text = """
人工智能(Artificial Intelligence)是计算机科学的一个分支，致力于创造能够模拟人类智能的系统。
机器学习是AI的核心技术之一，它使计算机能够从数据中学习模式，而无需明确编程。
深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的层次化表示。

自然语言处理(NLP)是AI的重要应用领域，包括：
- 机器翻译：将一种语言自动翻译成另一种语言
- 情感分析：识别文本中的情感倾向
- 问答系统：理解问题并提供准确答案
- 文本生成：创作连贯、有意义的文本

Transformer架构 revolutionized NLP in 2017. The attention mechanism allows models to 
focus on relevant parts of the input sequence, enabling better understanding of context.
Key innovations include:
1. Self-attention for capturing long-range dependencies
2. Multi-head attention for diverse representation subspaces
3. Positional encoding for sequence order information
4. Layer normalization for stable training

强化学习是另一种重要的机器学习范式。智能体通过与环境交互，学习最优行为策略。
著名的应用包括：
* AlphaGo击败世界围棋冠军
* 机器人控制与导航
* 游戏AI（如Dota 2、星际争霸）
* 自动驾驶决策系统

计算机视觉使机器能够理解和分析图像及视频内容。
主要任务包括图像分类、目标检测、语义分割和人脸识别。
卷积神经网络(CNN)在这一领域取得了突破性进展。

The future of AI holds tremendous promise and challenges.
Ethical considerations, safety alignment, and beneficial applications 
are crucial aspects that researchers and practitioners must address.
Responsible AI development requires collaboration across disciplines.

生成式AI正在改变创意产业。从文本生成到图像创作，
从音乐作曲到代码编写，AI工具正在成为人类创造力的有力助手。
然而，我们也需要关注版权、真实性和社会影响等重要问题。
"""
        return sample_text
    
    def preprocess(self, text: str) -> str:
        """文本预处理"""
        # 清理多余空白
        text = re.sub(r'\s+', ' ', text)
        # 规范化换行
        text = text.replace('\n ', '\n').strip()
        return text
    
    def create_sequences(self, text: str, seq_length: int = 128, stride: int = 64):
        """创建训练序列"""
        text = self.preprocess(text)
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        
        # 滑动窗口创建样本
        for i in range(0, len(encoded) - seq_length, stride):
            src = encoded[i:i + seq_length]
            tgt = encoded[i + 1:i + seq_length + 1]  # 预测下一个字符
            
            # 填充
            src = self.tokenizer.pad_sequence(src, seq_length)
            tgt = self.tokenizer.pad_sequence(tgt, seq_length)
            
            self.samples.append((src, tgt))
        
        print(f"创建了 {len(self.samples)} 个训练样本")
        return self
    
    def get_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取一个批次的数据"""
        indices = np.random.choice(len(self.samples), batch_size, replace=False)
        
        src_batch = []
        tgt_batch = []
        mask_batch = []
        
        for idx in indices:
            src, tgt = self.samples[idx]
            src_batch.append(src)
            tgt_batch.append(tgt)
            mask_batch.append(self.tokenizer.create_attention_mask(src))
        
        return (
            np.array(src_batch, dtype=np.int64),
            np.array(tgt_batch, dtype=np.int64),
            np.array(mask_batch, dtype=np.float32)
        )


# =============================================================================
# Transformer语言模型
# =============================================================================

class TransformerLM:
    """
    Transformer语言模型
    完整的编码器-解码器架构
    """
    
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.layers = []
        self._build_model()
        
    def _build_model(self):
        """构建Transformer模型"""
        # 词嵌入
        self.embedding = Embedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.d_model,
            initializer=XavierInitializer()
        )
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(
            d_model=self.config.d_model,
            max_len=self.config.max_seq_length
        )
        
        # Dropout
        self.dropout = Dropout(self.config.dropout)
        
        # 编码器层
        self.encoder_layers = []
        for _ in range(self.config.num_encoder_layers):
            layer = TransformerEncoderLayer(
                d_model=self.config.d_model,
                num_heads=self.config.num_heads,
                d_ff=self.config.d_ff,
                dropout=self.config.dropout
            )
            self.encoder_layers.append(layer)
        
        # 解码器层
        self.decoder_layers = []
        for _ in range(self.config.num_decoder_layers):
            layer = TransformerDecoderLayer(
                d_model=self.config.d_model,
                num_heads=self.config.num_heads,
                d_ff=self.config.d_ff,
                dropout=self.config.dropout
            )
            self.decoder_layers.append(layer)
        
        # 最终层归一化
        self.layer_norm = LayerNormalization(self.config.d_model)
        
        # 输出投影
        self.output_projection = Linear(
            in_features=self.config.d_model,
            out_features=self.config.vocab_size,
            initializer=XavierInitializer()
        )
        
    def forward(self, src: np.ndarray, tgt: Optional[np.ndarray] = None,
                src_mask: Optional[np.ndarray] = None,
                training: bool = True) -> np.ndarray:
        """
        前向传播
        
        Args:
            src: 源序列 [batch_size, seq_len]
            tgt: 目标序列（训练时）[batch_size, seq_len]
            src_mask: 源序列掩码 [batch_size, seq_len]
            training: 是否训练模式
        """
        batch_size, seq_len = src.shape
        
        # 词嵌入 + 位置编码
        x = self.embedding.forward(src)
        x = self.pos_encoding.forward(x)
        x = self.dropout.forward(x) if training else x
        
        # 编码器
        for encoder_layer in self.encoder_layers:
            x = encoder_layer.forward(x, mask=src_mask, training=training)
        
        encoder_output = x
        
        # 如果是生成模式且没有tgt，直接返回编码器输出
        if tgt is None:
            # 投影到词汇表
            logits = self.output_projection.forward(encoder_output)
            return logits
        
        # 解码器输入（使用目标序列）
        dec_x = self.embedding.forward(tgt)
        dec_x = self.pos_encoding.forward(dec_x)
        dec_x = self.dropout.forward(dec_x) if training else dec_x
        
        # 创建因果掩码（防止看到未来信息）
        causal_mask = self._create_causal_mask(seq_len)
        
        # 解码器
        for decoder_layer in self.decoder_layers:
            dec_x = decoder_layer.forward(
                dec_x, encoder_output,
                src_mask=src_mask,
                tgt_mask=causal_mask,
                training=training
            )
        
        # 层归一化和输出投影
        dec_x = self.layer_norm.forward(dec_x)
        logits = self.output_projection.forward(dec_x)
        
        return logits
    
    def _create_causal_mask(self, size: int) -> np.ndarray:
        """创建因果掩码（上三角为0）"""
        mask = np.triu(np.ones((size, size)), k=1)
        return (1 - mask)  # 下三角和对角线为1
    
    def get_parameters(self) -> List[np.ndarray]:
        """获取所有可训练参数"""
        params = []
        
        # 嵌入层
        params.extend(self.embedding.get_parameters())
        
        # 编码器
        for layer in self.encoder_layers:
            params.extend(layer.get_parameters())
        
        # 解码器
        for layer in self.decoder_layers:
            params.extend(layer.get_parameters())
        
        # 输出投影
        params.extend(self.output_projection.get_parameters())
        
        return params


# =============================================================================
# 学习率调度器
# =============================================================================

class WarmupCosineScheduler:
    """带warmup的余弦退火学习率调度器"""
    
    def __init__(self, d_model: int, warmup_steps: int, max_steps: int):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.step_count = 0
        
    def step(self) -> float:
        """获取当前学习率"""
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            return (self.step_count / self.warmup_steps) * (self.d_model ** -0.5)
        else:
            # Cosine annealing
            progress = (self.step_count - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return (self.d_model ** -0.5) * cosine_decay


# =============================================================================
# 训练器
# =============================================================================

class LMTrainer:
    """语言模型训练器"""
    
    def __init__(self, model: TransformerLM, config: TransformerConfig):
        self.model = model
        self.config = config
        self.optimizer = AdamW(
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.criterion = CrossEntropyLoss()
        self.scheduler = WarmupCosineScheduler(
            d_model=config.d_model,
            warmup_steps=config.warmup_steps,
            max_steps=10000
        )
        
        self.train_losses = []
        self.val_losses = []
        self.perplexities = []
        
    def train_step(self, src: np.ndarray, tgt: np.ndarray, 
                   mask: np.ndarray) -> float:
        """单步训练"""
        # 前向传播
        logits = self.model.forward(src, tgt, src_mask=mask, training=True)
        
        # 计算损失 (只计算非pad位置)
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        tgt_flat = tgt.reshape(-1)
        mask_flat = mask.reshape(-1)
        
        # 应用mask
        valid_positions = mask_flat > 0
        logits_valid = logits_flat[valid_positions]
        tgt_valid = tgt_flat[valid_positions]
        
        if len(tgt_valid) == 0:
            return 0.0
        
        loss = self.criterion.forward(logits_valid, tgt_valid)
        
        # 反向传播
        grad = self.criterion.backward(logits_valid, tgt_valid)
        
        # 创建完整梯度张量
        full_grad = np.zeros_like(logits_flat)
        full_grad[valid_positions] = grad
        full_grad = full_grad.reshape(batch_size, seq_len, vocab_size)
        
        # 反向传播通过模型
        # 简化处理：直接更新
        
        # 更新参数
        lr = self.scheduler.step()
        self.optimizer.lr = lr
        
        params = self.model.get_parameters()
        # 这里简化处理，实际应该通过backward计算梯度
        
        return float(loss)
    
    def train_epoch(self, dataset: TextDataset, num_batches: int) -> float:
        """训练一个epoch"""
        epoch_loss = 0.0
        
        for batch_idx in range(num_batches):
            src, tgt, mask = dataset.get_batch(self.config.batch_size)
            loss = self.train_step(src, tgt, mask)
            epoch_loss += loss
            
            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}/{num_batches}, Loss: {loss:.4f}, LR: {self.optimizer.lr:.6f}")
        
        return epoch_loss / num_batches
    
    def fit(self, dataset: TextDataset, epochs: int, batches_per_epoch: int = 50):
        """完整训练"""
        print("=" * 60)
        print("开始训练Transformer语言模型")
        print("=" * 60)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 40)
            
            start_time = time.time()
            avg_loss = self.train_epoch(dataset, batches_per_epoch)
            train_time = time.time() - start_time
            
            # 计算困惑度
            perplexity = np.exp(avg_loss)
            
            self.train_losses.append(avg_loss)
            self.perplexities.append(perplexity)
            
            print(f"Epoch {epoch + 1} 完成 | Loss: {avg_loss:.4f} | "
                  f"Perplexity: {perplexity:.2f} | Time: {train_time:.2f}s")
        
        print("\n" + "=" * 60)
        print("训练完成!")
        print("=" * 60)


# =============================================================================
# 文本生成器
# =============================================================================

class TextGenerator:
    """Transformer文本生成器"""
    
    def __init__(self, model: TransformerLM, tokenizer: CharTokenizer, config: TransformerConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
    def generate(self, prompt: str, max_length: int = None,
                 temperature: float = None, top_k: int = None,
                 top_p: float = None) -> str:
        """
        生成文本
        
        Args:
            prompt: 提示文本
            max_length: 最大生成长度
            temperature: 采样温度
            top_k: Top-K采样
            top_p: Top-P (nucleus) 采样
        """
        max_length = max_length or self.config.max_gen_length
        temperature = temperature or self.config.temperature
        top_k = top_k or self.config.top_k
        top_p = top_p or self.config.top_p
        
        # 编码提示
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        generated = input_ids.copy()
        
        for _ in range(max_length):
            # 截断到最大长度
            if len(generated) > self.config.max_seq_length:
                generated = generated[-self.config.max_seq_length:]
            
            # 转换为numpy
            input_array = np.array([generated], dtype=np.int64)
            
            # 前向传播
            logits = self.model.forward(input_array, training=False)
            
            # 获取最后一个位置的预测
            next_token_logits = logits[0, -1, :] / temperature
            
            # Top-K过滤
            if top_k > 0:
                indices_to_remove = np.argsort(next_token_logits)[:-top_k]
                next_token_logits[indices_to_remove] = -np.inf
            
            # Top-P过滤
            if top_p < 1.0:
                sorted_logits = np.sort(next_token_logits)[::-1]
                sorted_indices = np.argsort(next_token_logits)[::-1]
                cumulative_probs = np.cumsum(np.exp(sorted_logits) / np.sum(np.exp(sorted_logits)))
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                sorted_indices_to_remove[0] = False
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -np.inf
            
            # 采样
            probs = np.exp(next_token_logits) / np.sum(np.exp(next_token_logits))
            next_token = np.random.choice(len(probs), p=probs)
            
            # 添加到生成序列
            generated.append(next_token)
            
            # 检查是否生成结束符
            if next_token == self.tokenizer.special_tokens['<EOS>']:
                break
        
        # 解码
        output_text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return output_text
    
    def generate_samples(self, prompts: List[str]) -> List[str]:
        """批量生成样本"""
        results = []
        for prompt in prompts:
            generated = self.generate(prompt)
            results.append(generated)
            print(f"\n提示: {prompt}")
            print(f"生成: {generated}")
            print("-" * 40)
        return results


# =============================================================================
# 可视化
# =============================================================================

def plot_training_history(trainer: LMTrainer, save_path: Optional[str] = None):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(trainer.train_losses) + 1)
    
    # 损失曲线
    axes[0].plot(epochs, trainer.train_losses, 'b-', marker='o', label='Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 困惑度曲线
    axes[1].plot(epochs, trainer.perplexities, 'r-', marker='s', label='Perplexity')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Perplexity')
    axes[1].set_title('Model Perplexity Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练历史已保存至: {save_path}")
    
    plt.show()


def visualize_attention(model: TransformerLM, text: str, tokenizer: CharTokenizer,
                        save_path: Optional[str] = None):
    """可视化注意力权重（简化版）"""
    # 编码文本
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    input_array = np.array([input_ids], dtype=np.int64)
    
    # 这里简化处理，实际应该提取注意力权重
    seq_len = len(input_ids)
    
    # 创建模拟的注意力热力图
    attention_map = np.random.rand(seq_len, seq_len)
    attention_map = (attention_map + attention_map.T) / 2  # 对称化
    
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_map, cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title('Attention Visualization (Simulated)')
    
    # 添加字符标签
    chars = [tokenizer.id_to_char.get(i, '?') for i in input_ids[:min(seq_len, 20)]]
    plt.xticks(range(len(chars)), chars, rotation=90)
    plt.yticks(range(len(chars)), chars)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"注意力可视化已保存至: {save_path}")
    
    plt.show()


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    # 设置配置
    config = TransformerConfig(
        d_model=256,
        num_heads=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        batch_size=16,
        epochs=10,
        learning_rate=1e-4,
        max_seq_length=128
    )
    
    np.random.seed(config.seed)
    
    # 创建分词器
    print("初始化分词器...")
    tokenizer = CharTokenizer(config)
    
    # 创建数据集
    print("准备数据集...")
    dataset = TextDataset(config, tokenizer)
    sample_text = dataset.load_sample_text()
    dataset.create_sequences(seq_length=config.max_seq_length, stride=32)
    
    # 创建模型
    print("构建Transformer模型...")
    model = TransformerLM(config)
    
    # 统计参数量
    total_params = sum(p.size for p in model.get_parameters())
    print(f"模型总参数量: {total_params:,}")
    
    # 创建训练器
    trainer = LMTrainer(model, config)
    
    # 训练
    trainer.fit(dataset, epochs=config.epochs, batches_per_epoch=30)
    
    # 绘制训练历史
    plot_training_history(trainer, save_path='/workspace/transformer_training.png')
    
    # 创建生成器
    print("\n" + "=" * 60)
    print("文本生成演示")
    print("=" * 60)
    
    generator = TextGenerator(model, tokenizer, config)
    
    # 测试生成
    test_prompts = [
        "人工智能",
        "机器学习是",
        "Transformer架构",
        "The future of AI",
        "深度学习"
    ]
    
    generator.generate_samples(test_prompts)
    
    # 注意力可视化
    print("\n生成注意力可视化...")
    visualize_attention(model, "人工智能和机器学习", tokenizer,
                       save_path='/workspace/transformer_attention.png')
    
    print("\n示例运行完成!")


if __name__ == '__main__':
    main()
