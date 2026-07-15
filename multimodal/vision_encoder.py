"""
视觉编码器集成 - Vision Encoder Integration

集成 CLIP/SigLIP 视觉编码器，支持多种分辨率输入
实现特征提取和投影

作者: UFO Framework Team
"""

import math
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


class VisionEncoderType(Enum):
    """视觉编码器类型"""
    CLIP = "clip"
    SIGLIP = "siglip"
    DINOv2 = "dinov2"
    CUSTOM = "custom"


@dataclass
class VisionEncoderConfig:
    """视觉编码器配置"""
    # 编码器类型
    encoder_type: VisionEncoderType = VisionEncoderType.CLIP
    model_name: str = "openai/clip-vit-base-patch32"

    # 图像配置
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3

    # 模型配置
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0

    # 投影配置
    projection_dim: int = 512
    num_projection_layers: int = 1

    # 多分辨率支持
    support_multi_resolution: bool = True
    supported_resolutions: Tuple[Tuple[int, int], ...] = (
        (224, 224),
        (336, 336),
        (384, 384),
        (448, 448),
    )

    # 预处理配置
    image_mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    image_std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    # 缓存配置
    use_cache: bool = False


def get_default_config(encoder_type: str = "clip") -> VisionEncoderConfig:
    """获取默认配置"""
    configs = {
        "clip": VisionEncoderConfig(
            encoder_type=VisionEncoderType.CLIP,
            model_name="openai/clip-vit-base-patch32",
            image_size=224,
            patch_size=16,
            hidden_size=768,
            projection_dim=512,
        ),
        "clip_large": VisionEncoderConfig(
            encoder_type=VisionEncoderType.CLIP,
            model_name="openai/clip-vit-large-patch14",
            image_size=224,
            patch_size=14,
            hidden_size=1024,
            num_attention_heads=16,
            projection_dim=768,
        ),
        "siglip": VisionEncoderConfig(
            encoder_type=VisionEncoderType.SIGLIP,
            model_name="google/siglip-base-patch16-224",
            image_size=224,
            patch_size=16,
            hidden_size=768,
            projection_dim=768,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
        ),
        "dinov2": VisionEncoderConfig(
            encoder_type=VisionEncoderType.DINOv2,
            model_name="facebook/dinov2-base",
            image_size=518,
            patch_size=14,
            hidden_size=768,
            projection_dim=768,
            image_mean=(0.485, 0.456, 0.406),
            image_std=(0.229, 0.224, 0.225),
        ),
    }
    return configs.get(encoder_type, configs["clip"])


class PatchEmbedding(nn.Module):
    """
    图像Patch嵌入层

    将图像分割为patches并投影到隐藏维度
    """

    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.hidden_size = config.hidden_size

        self.projection = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

        # 可学习的类别token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02)

        # 位置嵌入
        num_patches = (config.image_size // config.patch_size) ** 2
        self.position_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, config.hidden_size) * 0.02
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [batch_size, num_channels, height, width]

        Returns:
            [batch_size, num_patches + 1, hidden_size]
        """
        batch_size = pixel_values.shape[0]

        # Patch嵌入
        patch_embeds = self.projection(pixel_values)  # [B, hidden, H/P, W/P]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # [B, num_patches, hidden]

        # 添加类别token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, patch_embeds], dim=1)

        # 添加位置嵌入
        embeddings = embeddings + self.position_embedding

        return embeddings


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""

    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 生成Q, K, V
        qkv = self.qkv(hidden_states)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 应用注意力
        out = attn @ v  # [B, num_heads, seq_len, head_dim]
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)

        # 投影
        out = self.proj(out)

        return out


class MLP(nn.Module):
    """前馈神经网络"""

    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TransformerBlock(nn.Module):
    """Transformer编码器块"""

    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = MultiHeadAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.mlp = MLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 自注意力 + 残差连接
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))

        # MLP + 残差连接
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))

        return hidden_states


class VisionTransformer(nn.Module):
    """
    视觉Transformer编码器

    基于ViT架构的视觉编码器
    """

    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.config = config

        # Patch嵌入
        self.embeddings = PatchEmbedding(config)

        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])

        # 后处理层归一化
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Args:
            pixel_values: [batch_size, num_channels, height, width]
            output_hidden_states: 是否输出所有隐藏层状态

        Returns:
            如果output_hidden_states=False: [batch_size, hidden_size] (CLS token)
            如果output_hidden_states=True: (最后一层输出, 所有隐藏层列表)
        """
        # 嵌入
        hidden_states = self.embeddings(pixel_values)

        # 存储所有隐藏状态
        all_hidden_states = [hidden_states] if output_hidden_states else None

        # Transformer层
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        # 后处理层归一化
        hidden_states = self.post_layernorm(hidden_states)

        # 提取CLS token
        pooled_output = hidden_states[:, 0, :]  # [batch_size, hidden_size]

        if output_hidden_states:
            return pooled_output, all_hidden_states

        return pooled_output


class ProjectionHead(nn.Module):
    """
    投影头

    将视觉特征投影到多模态共享空间
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, current_dim),
                nn.LayerNorm(current_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])

        layers.append(nn.Linear(current_dim, output_dim))

        self.projection = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, input_dim]

        Returns:
            [batch_size, output_dim]
        """
        return self.projection(features)


class VisionEncoder(nn.Module):
    """
    统一的视觉编码器

    支持CLIP、SigLIP、DINOv2等多种视觉编码器
    支持多分辨率输入
    """

    def __init__(self, config: Optional[VisionEncoderConfig] = None):
        super().__init__()
        self.config = config or VisionEncoderConfig()

        # 初始化编码器
        self._init_encoder()

        # 初始化投影头
        self.projection = ProjectionHead(
            input_dim=self.config.hidden_size,
            output_dim=self.config.projection_dim,
            num_layers=self.config.num_projection_layers,
        )

        # 预处理变换
        self.transform = self._build_transform()

        # 缓存
        self._cache: Dict[str, Any] = {}

    def _init_encoder(self):
        """初始化编码器"""
        if self.config.encoder_type == VisionEncoderType.CLIP:
            self._init_clip_encoder()
        elif self.config.encoder_type == VisionEncoderType.SIGLIP:
            self._init_siglip_encoder()
        elif self.config.encoder_type == VisionEncoderType.DINOv2:
            self._init_dinov2_encoder()
        else:
            # 自定义编码器
            self.encoder = VisionTransformer(self.config)

    def _init_clip_encoder(self):
        """初始化CLIP编码器（使用transformers库）"""
        try:
            from transformers import CLIPVisionModel, CLIPVisionConfig

            # 尝试加载预训练模型
            self.encoder = CLIPVisionModel.from_pretrained(
                self.config.model_name,
                torch_dtype=self.config.dtype,
            )
            self.config.hidden_size = self.encoder.config.hidden_size
            self.config.projection_dim = self.encoder.config.projection_dim

        except ImportError:
            warnings.warn("transformers not available, using custom ViT")
            self.encoder = VisionTransformer(self.config)
        except Exception as e:
            warnings.warn(f"Failed to load CLIP model: {e}, using custom ViT")
            self.encoder = VisionTransformer(self.config)

    def _init_siglip_encoder(self):
        """初始化SigLIP编码器"""
        try:
            from transformers import SiglipVisionModel

            self.encoder = SiglipVisionModel.from_pretrained(
                self.config.model_name,
                torch_dtype=self.config.dtype,
            )
            self.config.hidden_size = self.encoder.config.hidden_size

        except ImportError:
            warnings.warn("transformers not available, using custom ViT")
            self.encoder = VisionTransformer(self.config)
        except Exception as e:
            warnings.warn(f"Failed to load SigLIP model: {e}, using custom ViT")
            self.encoder = VisionTransformer(self.config)

    def _init_dinov2_encoder(self):
        """初始化DINOv2编码器"""
        try:
            from transformers import Dinov2Model

            self.encoder = Dinov2Model.from_pretrained(
                self.config.model_name,
                torch_dtype=self.config.dtype,
            )
            self.config.hidden_size = self.encoder.config.hidden_size

        except ImportError:
            warnings.warn("transformers not available, using custom ViT")
            self.encoder = VisionTransformer(self.config)
        except Exception as e:
            warnings.warn(f"Failed to load DINOv2 model: {e}, using custom ViT")
            self.encoder = VisionTransformer(self.config)

    def _build_transform(self) -> Callable:
        """构建预处理变换"""
        return transforms.Compose([
            transforms.Resize(self.config.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.image_mean, std=self.config.image_std),
        ])

    def preprocess(self, images: Union[Image.Image, List[Image.Image], torch.Tensor]) -> torch.Tensor:
        """
        预处理图像

        Args:
            images: PIL图像、图像列表或张量

        Returns:
            预处理后的张量 [batch_size, num_channels, height, width]
        """
        if isinstance(images, Image.Image):
            images = [images]

        if isinstance(images, list):
            images = [img.convert('RGB') if isinstance(img, Image.Image) else img for img in images]
            pixel_values = torch.stack([self.transform(img) for img in images])
        else:
            pixel_values = images

        return pixel_values.to(device=self.config.device, dtype=self.config.dtype)

    def encode(
        self,
        pixel_values: torch.Tensor,
        return_projection: bool = True,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        编码图像

        Args:
            pixel_values: [batch_size, num_channels, height, width]
            return_projection: 是否返回投影后的特征
            output_hidden_states: 是否输出所有隐藏层状态

        Returns:
            包含特征的字典
        """
        # 确保输入在正确的设备上
        pixel_values = pixel_values.to(device=self.config.device, dtype=self.config.dtype)

        # 编码
        if hasattr(self.encoder, 'forward'):
            if output_hidden_states:
                outputs = self.encoder(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True,
                )
                image_features = outputs.pooler_output
                hidden_states = outputs.hidden_states
            else:
                outputs = self.encoder(pixel_values=pixel_values, return_dict=True)
                image_features = outputs.pooler_output
                hidden_states = None
        else:
            if output_hidden_states:
                image_features, hidden_states = self.encoder(pixel_values, output_hidden_states=True)
            else:
                image_features = self.encoder(pixel_values)
                hidden_states = None

        result = {
            "image_features": image_features,
            "pooled_output": image_features,
        }

        # 投影
        if return_projection:
            projected_features = self.projection(image_features)
            # L2归一化
            projected_features = F.normalize(projected_features, p=2, dim=-1)
            result["projected_features"] = projected_features

        if hidden_states is not None:
            result["hidden_states"] = hidden_states

        return result

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_projection: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """前向传播"""
        return self.encode(pixel_values, return_projection=return_projection)

    def encode_image(
        self,
        images: Union[Image.Image, List[Image.Image], torch.Tensor],
        return_projection: bool = True,
    ) -> torch.Tensor:
        """
        便捷的图像编码接口

        Args:
            images: 输入图像
            return_projection: 是否返回投影特征

        Returns:
            图像特征
        """
        pixel_values = self.preprocess(images)
        outputs = self.encode(pixel_values, return_projection=return_projection)

        if return_projection:
            return outputs["projected_features"]
        return outputs["image_features"]

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """获取图像特征（兼容transformers接口）"""
        outputs = self.encode(pixel_values, return_projection=False)
        return outputs["image_features"]

    def resize_position_embeddings(self, new_image_size: int):
        """
        调整位置嵌入大小以支持不同分辨率

        Args:
            new_image_size: 新的图像尺寸
        """
        if not hasattr(self.encoder, 'embeddings'):
            warnings.warn("Encoder does not support resizing position embeddings")
            return

        old_size = self.config.image_size
        new_num_patches = (new_image_size // self.config.patch_size) ** 2
        old_num_patches = (old_size // self.config.patch_size) ** 2

        if new_num_patches == old_num_patches:
            return

        # 获取旧的位置嵌入
        old_pos_embed = self.encoder.embeddings.position_embedding

        # 插值位置嵌入
        cls_embed = old_pos_embed[:, 0:1, :]
        old_patch_embed = old_pos_embed[:, 1:, :]

        # 重塑为2D
        old_grid_size = int(math.sqrt(old_num_patches))
        new_grid_size = int(math.sqrt(new_num_patches))

        old_patch_embed = old_patch_embed.reshape(1, old_grid_size, old_grid_size, -1)
        old_patch_embed = old_patch_embed.permute(0, 3, 1, 2)  # [1, hidden, H, W]

        # 插值
        new_patch_embed = F.interpolate(
            old_patch_embed,
            size=(new_grid_size, new_grid_size),
            mode='bilinear',
            align_corners=False,
        )

        new_patch_embed = new_patch_embed.permute(0, 2, 3, 1).reshape(1, new_num_patches, -1)

        # 合并CLS token
        new_pos_embed = torch.cat([cls_embed, new_patch_embed], dim=1)

        # 更新位置嵌入
        self.encoder.embeddings.position_embedding = nn.Parameter(new_pos_embed)
        self.config.image_size = new_image_size

        print(f"Resized position embeddings from {old_size} to {new_image_size}")

    def enable_gradient_checkpointing(self):
        """启用梯度检查点以节省显存"""
        if hasattr(self.encoder, 'gradient_checkpointing_enable'):
            self.encoder.gradient_checkpointing_enable()
        else:
            warnings.warn("Encoder does not support gradient checkpointing")

    def freeze_encoder(self):
        """冻结编码器参数"""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """解冻编码器参数"""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def get_config(self) -> Dict[str, Any]:
        """获取配置信息"""
        return {
            "encoder_type": self.config.encoder_type.value,
            "model_name": self.config.model_name,
            "image_size": self.config.image_size,
            "patch_size": self.config.patch_size,
            "hidden_size": self.config.hidden_size,
            "projection_dim": self.config.projection_dim,
            "num_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


def create_vision_encoder(
    encoder_type: str = "clip",
    model_name: Optional[str] = None,
    image_size: Optional[int] = None,
    projection_dim: Optional[int] = None,
    device: Optional[str] = None,
    **kwargs
) -> VisionEncoder:
    """
    便捷函数：创建视觉编码器

    Args:
        encoder_type: 编码器类型 ("clip", "clip_large", "siglip", "dinov2")
        model_name: 模型名称（覆盖默认）
        image_size: 图像尺寸（覆盖默认）
        projection_dim: 投影维度（覆盖默认）
        device: 设备
        **kwargs: 其他配置参数

    Returns:
        VisionEncoder实例
    """
    config = get_default_config(encoder_type)

    if model_name:
        config.model_name = model_name
    if image_size:
        config.image_size = image_size
    if projection_dim:
        config.projection_dim = projection_dim
    if device:
        config.device = device

    # 更新其他配置
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return VisionEncoder(config)


# ==================== 使用示例 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("视觉编码器测试")
    print("=" * 60)

    # 测试配置
    config = VisionEncoderConfig(
        encoder_type=VisionEncoderType.CUSTOM,
        image_size=224,
        patch_size=16,
        hidden_size=768,
        num_hidden_layers=12,
        projection_dim=512,
    )

    # 创建编码器
    print("\n[1] 创建视觉编码器")
    encoder = VisionEncoder(config)
    print(f"  模型参数: {sum(p.numel() for p in encoder.parameters()):,}")

    # 测试前向传播
    print("\n[2] 测试前向传播")
    dummy_images = torch.randn(2, 3, 224, 224)
    outputs = encoder.encode(dummy_images)
    print(f"  图像特征形状: {outputs['image_features'].shape}")
    print(f"  投影特征形状: {outputs['projected_features'].shape}")

    # 测试多分辨率
    print("\n[3] 测试多分辨率支持")
    for size in [224, 336, 384]:
        encoder.resize_position_embeddings(size)
        dummy_images = torch.randn(1, 3, size, size)
        outputs = encoder.encode(dummy_images)
        print(f"  尺寸 {size}x{size}: 特征形状 {outputs['image_features'].shape}")

    # 测试便捷函数
    print("\n[4] 测试便捷函数")
    encoder2 = create_vision_encoder(
        encoder_type="clip",
        image_size=224,
        projection_dim=512,
    )
    print(f"  创建编码器: {encoder2.config.encoder_type.value}")
    print(f"  配置: {encoder2.get_config()}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
