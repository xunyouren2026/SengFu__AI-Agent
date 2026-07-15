"""
多模态处理模块 - Multimodal Processing
实现图像-文本、音频-文本、视频处理等多模态功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict

# ==================== 多模态编码器 ====================

class ImageEncoder(nn.Module):
    """图像编码器"""
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 768,
        num_layers: int = 4,
        image_size: int = 224,
        patch_size: int = 16,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch嵌入
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, hidden_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads=12, mlp_ratio=4.0)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """编码图像"""
        B = x.size(0)
        
        # Patch嵌入
        x = self.patch_embed(x)  # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # Transformer层
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        return x


class TextEncoder(nn.Module):
    """文本编码器"""
    
    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        max_length: int = 512,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Token嵌入
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_length, hidden_dim) * 0.02)
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads=num_heads, mlp_ratio=4.0)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """编码文本"""
        B, L = input_ids.size()
        
        # Token嵌入
        x = self.token_embed(input_ids)
        
        # 添加位置编码
        x = x + self.pos_embed[:, :L, :]
        
        # Transformer层
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        x = self.norm(x)
        
        return x


class AudioEncoder(nn.Module):
    """音频编码器"""
    
    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 768,
        num_layers: int = 4,
        num_mel_bins: int = 80,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 卷积预处理
        self.conv1 = nn.Conv1d(num_mel_bins, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, 3, stride=2, padding=1)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, 1000, hidden_dim) * 0.02)
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads=8, mlp_ratio=4.0)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """编码音频"""
        # x: [B, num_mel_bins, T]
        
        # 卷积预处理
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        
        # 转换为序列
        x = x.transpose(1, 2)  # [B, T, D]
        
        # 添加位置编码
        L = x.size(1)
        x = x + self.pos_embed[:, :L, :]
        
        # Transformer层
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer块"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class MultiHeadAttention(nn.Module):
    """多头注意力"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn = attn + attention_mask
        
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


# ==================== 多模态融合 ====================

class CrossModalAttention(nn.Module):
    """跨模态注意力"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """跨模态注意力"""
        B, N_q, C = query.shape
        N_kv = key_value.size(1)
        
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn = attn + attention_mask
        
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.out_proj(x)
        
        return x


class MultimodalFusion(nn.Module):
    """多模态融合"""
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        fusion_type: str = 'cross_attention',  # 'cross_attention', 'concat', 'add'
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.hidden_dim = hidden_dim
        
        if fusion_type == 'cross_attention':
            self.image_to_text = nn.ModuleList([
                CrossModalAttention(hidden_dim, num_heads)
                for _ in range(num_layers)
            ])
            self.text_to_image = nn.ModuleList([
                CrossModalAttention(hidden_dim, num_heads)
                for _ in range(num_layers)
            ])
            
            self.image_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
            self.text_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        elif fusion_type == 'concat':
            self.fusion_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        elif fusion_type == 'add':
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """融合多模态特征"""
        if self.fusion_type == 'cross_attention':
            for i in range(len(self.image_to_text)):
                # 图像关注文本
                image_features = image_features + self.image_to_text[i](
                    self.image_norm[i](image_features),
                    text_features,
                )
                
                # 文本关注图像
                text_features = text_features + self.text_to_image[i](
                    self.text_norm[i](text_features),
                    image_features,
                )
            
            return image_features, text_features
        
        elif self.fusion_type == 'concat':
            # 拼接并投影
            image_cls = image_features[:, 0, :]
            text_cls = text_features[:, 0, :]
            fused = self.fusion_proj(torch.cat([image_cls, text_cls], dim=-1))
            return fused.unsqueeze(1), fused.unsqueeze(1)
        
        elif self.fusion_type == 'add':
            # 门控加法
            image_cls = image_features[:, 0, :]
            text_cls = text_features[:, 0, :]
            gate = self.gate(torch.cat([image_cls, text_cls], dim=-1))
            fused = gate * image_cls + (1 - gate) * text_cls
            return fused.unsqueeze(1), fused.unsqueeze(1)
        
        return image_features, text_features


# ==================== CLIP风格模型 ====================

class CLIPModel(nn.Module):
    """CLIP风格模型"""
    
    def __init__(
        self,
        image_hidden_dim: int = 768,
        text_hidden_dim: int = 768,
        embed_dim: int = 512,
        temperature: float = 0.07,
    ):
        super().__init__()
        
        # 编码器
        self.image_encoder = ImageEncoder(hidden_dim=image_hidden_dim)
        self.text_encoder = TextEncoder(hidden_dim=text_hidden_dim)
        
        # 投影头
        self.image_proj = nn.Linear(image_hidden_dim, embed_dim)
        self.text_proj = nn.Linear(text_hidden_dim, embed_dim)
        
        # 温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
    
    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 编码
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(input_ids, attention_mask)
        
        # CLS token
        image_cls = image_features[:, 0, :]
        text_cls = text_features[:, 0, :]
        
        # 投影
        image_embed = self.image_proj(image_cls)
        text_embed = self.text_proj(text_cls)
        
        # 归一化
        image_embed = F.normalize(image_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)
        
        # 相似度
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embed @ text_embed.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text, logit_scale
    
    def compute_loss(
        self,
        logits_per_image: torch.Tensor,
        logits_per_text: torch.Tensor,
    ) -> torch.Tensor:
        """计算对比损失"""
        batch_size = logits_per_image.size(0)
        labels = torch.arange(batch_size, device=logits_per_image.device)
        
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        
        return (loss_i + loss_t) / 2


# ==================== 视觉问答 ====================

class VisualQuestionAnswering(nn.Module):
    """视觉问答模型"""
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_answers: int = 1000,
        num_layers: int = 4,
    ):
        super().__init__()
        
        # 编码器
        self.image_encoder = ImageEncoder(hidden_dim=hidden_dim)
        self.text_encoder = TextEncoder(hidden_dim=hidden_dim)
        
        # 多模态融合
        self.fusion = MultimodalFusion(hidden_dim, num_layers)
        
        # 答案预测
        self.answer_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_answers),
        )
    
    def forward(
        self,
        image: torch.Tensor,
        question_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播"""
        # 编码
        image_features = self.image_encoder(image)
        question_features = self.text_encoder(question_ids, attention_mask)
        
        # 融合
        _, fused_features = self.fusion(image_features, question_features)
        
        # 预测答案
        fused_cls = fused_features[:, 0, :]
        logits = self.answer_head(fused_cls)
        
        return logits


# ==================== 图像描述生成 ====================

class ImageCaptioning(nn.Module):
    """图像描述生成模型"""
    
    def __init__(
        self,
        hidden_dim: int = 768,
        vocab_size: int = 50000,
        num_layers: int = 6,
        max_length: int = 100,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # 图像编码器
        self.image_encoder = ImageEncoder(hidden_dim=hidden_dim)
        
        # 文本解码器
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_length, hidden_dim) * 0.02)
        
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(hidden_dim, num_heads=8, mlp_ratio=4.0)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
    
    def forward(
        self,
        image: torch.Tensor,
        caption_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播（训练）"""
        # 编码图像
        image_features = self.image_encoder(image)
        
        # 解码文本
        B, L = caption_ids.size()
        text_embed = self.token_embed(caption_ids)
        text_embed = text_embed + self.pos_embed[:, :L, :]
        
        # 解码器层
        for layer in self.decoder_layers:
            text_embed = layer(text_embed, image_features, attention_mask)
        
        text_embed = self.norm(text_embed)
        logits = self.output_proj(text_embed)
        
        return logits
    
    def generate(
        self,
        image: torch.Tensor,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """生成描述"""
        # 编码图像
        image_features = self.image_encoder(image)
        
        # 自回归生成
        B = image.size(0)
        generated = torch.full((B, 1), bos_token_id, dtype=torch.long, device=image.device)
        
        for _ in range(self.max_length):
            # 前向传播
            text_embed = self.token_embed(generated)
            text_embed = text_embed + self.pos_embed[:, :generated.size(1), :]
            
            for layer in self.decoder_layers:
                text_embed = layer(text_embed, image_features)
            
            text_embed = self.norm(text_embed)
            logits = self.output_proj(text_embed[:, -1, :]) / temperature
            
            # Top-p采样
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # 检查结束
            if (next_token == eos_token_id).all():
                break
        
        return generated


class DecoderBlock(nn.Module):
    """解码器块"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        
        # 自注意力
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads)
        
        # 跨模态注意力
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = CrossModalAttention(hidden_dim, num_heads)
        
        # FFN
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 自注意力（因果）
        x = x + self.self_attn(self.norm1(x), attention_mask)
        
        # 跨模态注意力
        x = x + self.cross_attn(self.norm2(x), memory)
        
        # FFN
        x = x + self.mlp(self.norm3(x))
        
        return x


# ==================== 音频-文本模型 ====================

class AudioTextModel(nn.Module):
    """音频-文本模型"""
    
    def __init__(
        self,
        audio_hidden_dim: int = 768,
        text_hidden_dim: int = 768,
        embed_dim: int = 512,
    ):
        super().__init__()
        
        # 编码器
        self.audio_encoder = AudioEncoder(hidden_dim=audio_hidden_dim)
        self.text_encoder = TextEncoder(hidden_dim=text_hidden_dim)
        
        # 投影头
        self.audio_proj = nn.Linear(audio_hidden_dim, embed_dim)
        self.text_proj = nn.Linear(text_hidden_dim, embed_dim)
        
        # 温度
        self.logit_scale = nn.Parameter(torch.zeros([]))
    
    def forward(
        self,
        audio: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 编码
        audio_features = self.audio_encoder(audio)
        text_features = self.text_encoder(input_ids, attention_mask)
        
        # CLS token
        audio_cls = audio_features[:, 0, :]
        text_cls = text_features[:, 0, :]
        
        # 投影和归一化
        audio_embed = F.normalize(self.audio_proj(audio_cls), dim=-1)
        text_embed = F.normalize(self.text_proj(text_cls), dim=-1)
        
        # 相似度
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * audio_embed @ text_embed.t()
        
        return logits, logits.t()


# ==================== 视觉-音频模型 ====================

class VisualAudioModel(nn.Module):
    """视觉-音频模型"""
    
    def __init__(
        self,
        hidden_dim: int = 768,
        embed_dim: int = 512,
    ):
        super().__init__()
        
        # 编码器
        self.image_encoder = ImageEncoder(hidden_dim=hidden_dim)
        self.audio_encoder = AudioEncoder(hidden_dim=hidden_dim)
        
        # 投影
        self.image_proj = nn.Linear(hidden_dim, embed_dim)
        self.audio_proj = nn.Linear(hidden_dim, embed_dim)
        
        # 融合
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        self.logit_scale = nn.Parameter(torch.zeros([]))
    
    def forward(
        self,
        image: torch.Tensor,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播"""
        # 编码
        image_features = self.image_encoder(image)
        audio_features = self.audio_encoder(audio)
        
        # CLS token
        image_cls = image_features[:, 0, :]
        audio_cls = audio_features[:, 0, :]
        
        # 投影
        image_embed = self.image_proj(image_cls)
        audio_embed = self.audio_proj(audio_cls)
        
        # 融合
        fused = self.fusion(torch.cat([image_embed, audio_embed], dim=-1))
        fused = F.normalize(fused, dim=-1)
        
        # 相似度
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * fused @ fused.t()
        
        return logits


# ==================== 主函数 ====================

def main():
    """测试多模态模块"""
    print("多模态模块测试")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 测试图像编码器
    print("\n测试图像编码器...")
    image_encoder = ImageEncoder(hidden_dim=256, num_layers=2).to(device)
    image = torch.randn(2, 3, 224, 224).to(device)
    image_features = image_encoder(image)
    print(f"Image features shape: {image_features.shape}")
    
    # 测试文本编码器
    print("\n测试文本编码器...")
    text_encoder = TextEncoder(vocab_size=1000, hidden_dim=256, num_layers=2).to(device)
    text = torch.randint(0, 1000, (2, 32)).to(device)
    text_features = text_encoder(text)
    print(f"Text features shape: {text_features.shape}")
    
    # 测试多模态融合
    print("\n测试多模态融合...")
    fusion = MultimodalFusion(hidden_dim=256, num_layers=2).to(device)
    fused_image, fused_text = fusion(image_features, text_features)
    print(f"Fused image shape: {fused_image.shape}, Fused text shape: {fused_text.shape}")
    
    # 测试CLIP模型
    print("\n测试CLIP模型...")
    clip = CLIPModel(image_hidden_dim=256, text_hidden_dim=256, embed_dim=128).to(device)
    logits_i, logits_t, scale = clip(image, text)
    print(f"Logits per image shape: {logits_i.shape}, Scale: {scale.item():.4f}")
    
    print("\n多模态模块测试完成")


if __name__ == "__main__":
    main()
