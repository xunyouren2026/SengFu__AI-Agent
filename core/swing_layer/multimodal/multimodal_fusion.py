"""
AGI统一框架 - 多模态融合模块
实现CLIP风格的视觉-语言多模态融合，包括对比学习、跨模态注意力、模态对齐等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import math
import numpy as np
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')


# ==================== 配置类 ====================

@dataclass
class MultimodalConfig:
    """多模态融合配置"""
    # 文本编码器配置
    text_vocab_size: int = 30522
    text_hidden_dim: int = 768
    text_num_layers: int = 12
    text_num_heads: int = 12
    text_max_length: int = 77
    
    # 视觉编码器配置
    vision_image_size: int = 224
    vision_patch_size: int = 16
    vision_hidden_dim: int = 1024
    vision_num_layers: int = 24
    vision_num_heads: int = 16
    
    # 音频编码器配置
    audio_sample_rate: int = 16000
    audio_hidden_dim: int = 512
    audio_num_layers: int = 6
    
    # 融合配置
    fusion_dim: int = 512
    fusion_num_layers: int = 6
    fusion_num_heads: int = 8
    fusion_dropout: float = 0.1
    
    # 对比学习配置
    temperature: float = 0.07
    learnable_temperature: bool = True
    
    # 其他配置
    use_projection: bool = True
    normalize_embeddings: bool = True
    modality_dropout: float = 0.1


# ==================== 位置编码 ====================

class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, dim: int, max_length: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_length, dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """添加位置编码"""
        return x + self.pe[:, :x.size(1)]


class LearnedPositionalEncoding(nn.Module):
    """可学习位置编码"""
    
    def __init__(self, dim: int, max_length: int = 512):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_length, dim) * 0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """添加位置编码"""
        return x + self.pos_embedding[:, :x.size(1)]


class RotaryPositionalEncoding(nn.Module):
    """旋转位置编码 (RoPE)"""
    
    def __init__(self, dim: int, max_length: int = 2048, base: int = 10000):
        super().__init__()
        
        self.dim = dim
        self.max_length = max_length
        
        # 计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算位置编码
        self._build_cache(max_length)
        
    def _build_cache(self, seq_len: int):
        """构建位置编码缓存"""
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0))
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """应用旋转位置编码"""
        seq_len = x.size(1)
        
        if seq_len > self.cos_cached.size(1):
            self._build_cache(seq_len)
        
        cos = self.cos_cached[:, :seq_len]
        sin = self.sin_cached[:, :seq_len]
        
        # 应用旋转
        x1, x2 = x[..., :x.size(-1)//2], x[..., x.size(-1)//2:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ==================== 文本编码器 ====================

class TextEncoder(nn.Module):
    """文本编码器 (基于Transformer)"""
    
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config
        
        # 词嵌入
        self.token_embedding = nn.Embedding(config.text_vocab_size, config.text_hidden_dim)
        self.position_embedding = LearnedPositionalEncoding(
            config.text_hidden_dim, config.text_max_length
        )
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.text_hidden_dim,
            nhead=config.text_num_heads,
            dim_feedforward=config.text_hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.text_num_layers)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(config.text_hidden_dim)
        
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """编码文本"""
        # 嵌入
        x = self.token_embedding(input_ids)
        x = self.position_embedding(x)
        
        # 创建注意力掩码
        if attention_mask is not None:
            # 转换为Transformer期望的格式
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None
        
        # 编码
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.layer_norm(x)
        
        return x


# ==================== 视觉编码器 ====================

class PatchEmbedding(nn.Module):
    """图像块嵌入"""
    
    def __init__(self, image_size: int, patch_size: int, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将图像转换为patch嵌入"""
        # x: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class VisionEncoder(nn.Module):
    """视觉编码器 (基于ViT)"""
    
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config
        
        # Patch嵌入
        self.patch_embed = PatchEmbedding(
            config.vision_image_size,
            config.vision_patch_size,
            embed_dim=config.vision_hidden_dim
        )
        
        num_patches = self.patch_embed.num_patches
        
        # 位置嵌入
        self.position_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, config.vision_hidden_dim) * 0.02
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.vision_hidden_dim) * 0.02)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.vision_hidden_dim,
            nhead=config.vision_num_heads,
            dim_feedforward=config.vision_hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.vision_num_layers)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(config.vision_hidden_dim)
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """编码图像"""
        batch_size = pixel_values.size(0)
        
        # Patch嵌入
        x = self.patch_embed(pixel_values)
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 添加位置嵌入
        x = x + self.position_embedding
        
        # 编码
        x = self.encoder(x)
        x = self.layer_norm(x)
        
        return x


# ==================== 音频编码器 ====================

class AudioEncoder(nn.Module):
    """音频编码器"""
    
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config
        
        # 卷积特征提取
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=10, stride=5, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, config.audio_hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(config.audio_hidden_dim),
            nn.GELU()
        )
        
        # 位置编码
        self.position_embedding = SinusoidalPositionalEncoding(config.audio_hidden_dim)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.audio_hidden_dim,
            nhead=8,
            dim_feedforward=config.audio_hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.audio_num_layers)
        
        self.layer_norm = nn.LayerNorm(config.audio_hidden_dim)
        
    def forward(self, audio_values: torch.Tensor) -> torch.Tensor:
        """编码音频"""
        # 卷积特征提取
        x = self.conv_layers(audio_values.unsqueeze(1))
        x = x.transpose(1, 2)  # (B, T, D)
        
        # 位置编码
        x = self.position_embedding(x)
        
        # 编码
        x = self.encoder(x)
        x = self.layer_norm(x)
        
        return x


# ==================== 跨模态注意力 ====================

class CrossModalAttention(nn.Module):
    """跨模态注意力机制"""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V投影
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """跨模态注意力计算"""
        batch_size = query.size(0)
        
        # 投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 重塑为多头
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力分数
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn = attn + attention_mask
            
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        return self.out_proj(out)


class CrossModalFusionLayer(nn.Module):
    """跨模态融合层"""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        # 文本到视觉注意力
        self.text_to_vision_attn = CrossModalAttention(dim, num_heads, dropout)
        # 视觉到文本注意力
        self.vision_to_text_attn = CrossModalAttention(dim, num_heads, dropout)
        
        # 自注意力
        self.text_self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.vision_self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        # FFN
        self.text_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.vision_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.text_norm1 = nn.LayerNorm(dim)
        self.text_norm2 = nn.LayerNorm(dim)
        self.text_norm3 = nn.LayerNorm(dim)
        self.vision_norm1 = nn.LayerNorm(dim)
        self.vision_norm2 = nn.LayerNorm(dim)
        self.vision_norm3 = nn.LayerNorm(dim)
        
    def forward(self, text_features: torch.Tensor, vision_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """融合文本和视觉特征"""
        # 文本自注意力
        text_self, _ = self.text_self_attn(text_features, text_features, text_features)
        text_features = self.text_norm1(text_features + text_self)
        
        # 视觉自注意力
        vision_self, _ = self.vision_self_attn(vision_features, vision_features, vision_features)
        vision_features = self.vision_norm1(vision_features + vision_self)
        
        # 跨模态注意力
        text_cross = self.text_to_vision_attn(text_features, vision_features, vision_features)
        vision_cross = self.vision_to_text_attn(vision_features, text_features, text_features)
        
        text_features = self.text_norm2(text_features + text_cross)
        vision_features = self.vision_norm2(vision_features + vision_cross)
        
        # FFN
        text_features = self.text_norm3(text_features + self.text_ffn(text_features))
        vision_features = self.vision_norm3(vision_features + self.vision_ffn(vision_features))
        
        return text_features, vision_features


# ==================== 模态对齐 ====================

class ModalityAlignment(nn.Module):
    """模态对齐模块"""
    
    def __init__(self, text_dim: int, vision_dim: int, fusion_dim: int):
        super().__init__()
        
        # 投影层
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid()
        )
        
    def forward(self, text_features: torch.Tensor, vision_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """对齐并融合模态"""
        # 投影到共享空间
        text_aligned = self.text_proj(text_features)
        vision_aligned = self.vision_proj(vision_features)
        
        # L2归一化
        text_normalized = F.normalize(text_aligned, p=2, dim=-1)
        vision_normalized = F.normalize(vision_aligned, p=2, dim=-1)
        
        # 门控融合
        concat = torch.cat([text_aligned, vision_aligned], dim=-1)
        gate_weights = self.gate(concat)
        
        fused = gate_weights * text_aligned + (1 - gate_weights) * vision_aligned
        
        return {
            'text_aligned': text_aligned,
            'vision_aligned': vision_aligned,
            'text_normalized': text_normalized,
            'vision_normalized': vision_normalized,
            'fused': fused
        }


# ==================== 对比学习损失 ====================

class ContrastiveLoss(nn.Module):
    """对比学习损失 (InfoNCE)"""
    
    def __init__(self, temperature: float = 0.07, learnable: bool = True):
        super().__init__()
        
        if learnable:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer('log_temperature', torch.log(torch.tensor(temperature)))
            
    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)
        
    def forward(self, text_embeddings: torch.Tensor, vision_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算对比损失"""
        # 归一化
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        vision_embeddings = F.normalize(vision_embeddings, p=2, dim=-1)
        
        # 相似度矩阵
        logits = torch.matmul(text_embeddings, vision_embeddings.T) / self.temperature
        
        batch_size = text_embeddings.size(0)
        labels = torch.arange(batch_size, device=text_embeddings.device)
        
        # 双向损失
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        loss = (loss_i2t + loss_t2i) / 2
        
        # 计算准确率
        with torch.no_grad():
            pred_i2t = logits.argmax(dim=1)
            pred_t2i = logits.T.argmax(dim=1)
            accuracy_i2t = (pred_i2t == labels).float().mean()
            accuracy_t2i = (pred_t2i == labels).float().mean()
        
        return {
            'loss': loss,
            'loss_i2t': loss_i2t,
            'loss_t2i': loss_t2i,
            'accuracy_i2t': accuracy_i2t,
            'accuracy_t2i': accuracy_t2i,
            'temperature': self.temperature
        }


# ==================== 主模型 ====================

class MultimodalFusionModel(nn.Module):
    """多模态融合主模型"""
    
    def __init__(self, config: Optional[MultimodalConfig] = None):
        super().__init__()
        self.config = config or MultimodalConfig()
        
        # 编码器
        self.text_encoder = TextEncoder(self.config)
        self.vision_encoder = VisionEncoder(self.config)
        
        # 模态对齐
        self.alignment = ModalityAlignment(
            self.config.text_hidden_dim,
            self.config.vision_hidden_dim,
            self.config.fusion_dim
        )
        
        # 跨模态融合层
        self.fusion_layers = nn.ModuleList([
            CrossModalFusionLayer(
                self.config.fusion_dim,
                self.config.fusion_num_heads,
                self.config.fusion_dropout
            )
            for _ in range(self.config.fusion_num_layers)
        ])
        
        # 对比学习损失
        self.contrastive_loss = ContrastiveLoss(
            self.config.temperature,
            self.config.learnable_temperature
        )
        
        # 输出头
        self.output_projection = nn.Linear(self.config.fusion_dim, self.config.fusion_dim)
        
    def encode_text(self, input_ids: torch.Tensor, 
                    attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """编码文本"""
        text_features = self.text_encoder(input_ids, attention_mask)
        # 使用[CLS] token或平均池化
        return text_features[:, 0]  # (B, D)
        
    def encode_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """编码图像"""
        vision_features = self.vision_encoder(pixel_values)
        # 使用[CLS] token
        return vision_features[:, 0]  # (B, D)
        
    def forward(self, input_ids: torch.Tensor,
                pixel_values: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                compute_loss: bool = True) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 编码
        text_features = self.text_encoder(input_ids, attention_mask)  # (B, L, D_t)
        vision_features = self.vision_encoder(pixel_values)  # (B, N, D_v)
        
        # 获取全局特征
        text_global = text_features[:, 0]  # (B, D_t)
        vision_global = vision_features[:, 0]  # (B, D_v)
        
        # 模态对齐
        aligned = self.alignment(text_global, vision_global)
        
        # 跨模态融合
        text_aligned = aligned['text_aligned'].unsqueeze(1)  # (B, 1, D)
        vision_aligned = aligned['vision_aligned'].unsqueeze(1)  # (B, 1, D)
        
        for fusion_layer in self.fusion_layers:
            text_aligned, vision_aligned = fusion_layer(text_aligned, vision_aligned)
        
        # 输出
        text_output = self.output_projection(text_aligned.squeeze(1))
        vision_output = self.output_projection(vision_aligned.squeeze(1))
        
        result = {
            'text_features': text_features,
            'vision_features': vision_features,
            'text_global': text_global,
            'vision_global': vision_global,
            'text_aligned': text_output,
            'vision_aligned': vision_output,
            'fused_features': aligned['fused']
        }
        
        # 计算对比损失
        if compute_loss:
            loss_dict = self.contrastive_loss(text_output, vision_output)
            result.update(loss_dict)
            
        return result
    
    def get_similarity(self, text_embeddings: torch.Tensor, 
                       vision_embeddings: torch.Tensor) -> torch.Tensor:
        """计算文本-图像相似度"""
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        vision_embeddings = F.normalize(vision_embeddings, p=2, dim=-1)
        return torch.matmul(text_embeddings, vision_embeddings.T)
        
    def retrieve_images(self, text_embedding: torch.Tensor, 
                        image_embeddings: torch.Tensor,
                        top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """根据文本检索图像"""
        similarities = self.get_similarity(text_embedding.unsqueeze(0), image_embeddings)
        scores, indices = similarities.squeeze(0).topk(top_k)
        return scores, indices
        
    def retrieve_texts(self, image_embedding: torch.Tensor,
                       text_embeddings: torch.Tensor,
                       top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """根据图像检索文本"""
        similarities = self.get_similarity(text_embeddings, image_embedding.unsqueeze(0))
        scores, indices = similarities.squeeze(1).topk(top_k)
        return scores, indices


# ==================== 多模态融合变体 ====================

class LateFusionModel(nn.Module):
    """晚期融合模型"""
    
    def __init__(self, text_dim: int = 768, vision_dim: int = 1024, fusion_dim: int = 512):
        super().__init__()
        
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.vision_proj = nn.Linear(vision_dim, fusion_dim)
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        self.classifier = nn.Linear(fusion_dim, 2)  # 二分类
        
    def forward(self, text_features: torch.Tensor, vision_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """晚期融合"""
        text_proj = self.text_proj(text_features)
        vision_proj = self.vision_proj(vision_features)
        
        # 拼接融合
        fused = self.fusion_mlp(torch.cat([text_proj, vision_proj], dim=-1))
        
        # 分类
        logits = self.classifier(fused)
        
        return {
            'fused': fused,
            'logits': logits
        }


class EarlyFusionModel(nn.Module):
    """早期融合模型"""
    
    def __init__(self, text_dim: int = 768, vision_dim: int = 1024, fusion_dim: int = 512):
        super().__init__()
        
        # 联合嵌入空间
        self.joint_embedding = nn.Linear(text_dim + vision_dim, fusion_dim)
        
        # Transformer处理
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim, nhead=8, dim_feedforward=fusion_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        self.output = nn.Linear(fusion_dim, fusion_dim)
        
    def forward(self, text_features: torch.Tensor, vision_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """早期融合"""
        # 拼接输入
        joint_input = torch.cat([text_features, vision_features], dim=-1)
        
        # 联合嵌入
        embedded = self.joint_embedding(joint_input)
        
        # Transformer处理
        processed = self.transformer(embedded)
        
        # 输出
        output = self.output(processed[:, 0])  # 使用第一个token
        
        return {
            'output': output
        }


class HybridFusionModel(nn.Module):
    """混合融合模型 (早期+晚期)"""
    
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        
        # 早期融合分支
        self.early_fusion = EarlyFusionModel(
            config.text_hidden_dim, config.vision_hidden_dim, config.fusion_dim
        )
        
        # 晚期融合分支
        self.late_fusion = LateFusionModel(
            config.text_hidden_dim, config.vision_hidden_dim, config.fusion_dim
        )
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(config.fusion_dim * 2, config.fusion_dim),
            nn.Sigmoid()
        )
        
        self.output = nn.Linear(config.fusion_dim, config.fusion_dim)
        
    def forward(self, text_features: torch.Tensor, vision_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """混合融合"""
        # 早期融合
        early_output = self.early_fusion(text_features, vision_features)['output']
        
        # 晚期融合
        late_output = self.late_fusion(text_features, vision_features)['fused']
        
        # 门控融合
        gate_input = torch.cat([early_output, late_output], dim=-1)
        gate_weights = self.gate(gate_input)
        
        # 加权融合
        fused = gate_weights * early_output + (1 - gate_weights) * late_output
        
        output = self.output(fused)
        
        return {
            'early_output': early_output,
            'late_output': late_output,
            'fused': fused,
            'output': output
        }


# ==================== 工具函数 ====================

def create_multimodal_model(model_type: str = "clip", config: Optional[MultimodalConfig] = None) -> nn.Module:
    """创建多模态模型"""
    config = config or MultimodalConfig()
    
    if model_type == "clip":
        return MultimodalFusionModel(config)
    elif model_type == "late_fusion":
        return LateFusionModel()
    elif model_type == "early_fusion":
        return EarlyFusionModel()
    elif model_type == "hybrid":
        return HybridFusionModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def compute_retrieval_metrics(similarities: torch.Tensor, k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """计算检索指标"""
    batch_size = similarities.size(0)
    labels = torch.arange(batch_size, device=similarities.device)
    
    metrics = {}
    
    for k in k_values:
        # Top-k预测
        _, top_k_indices = similarities.topk(k, dim=1)
        
        # 计算Recall@k
        correct = 0
        for i in range(batch_size):
            if i in top_k_indices[i]:
                correct += 1
        
        metrics[f'Recall@{k}'] = correct / batch_size
    
    # Mean Reciprocal Rank
    sorted_indices = similarities.argsort(dim=1, descending=True)
    mrr = 0.0
    for i in range(batch_size):
        rank = (sorted_indices[i] == i).nonzero(as_tuple=True)[0][0].item() + 1
        mrr += 1.0 / rank
    metrics['MRR'] = mrr / batch_size
    
    return metrics
