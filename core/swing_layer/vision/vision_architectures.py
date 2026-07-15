"""
视觉模型架构 - Vision Model Architectures
实现Vision Transformer (ViT)、EfficientNet、ConvNeXt、Swin Transformer等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from collections import OrderedDict
from functools import partial

# ==================== 基础模块 ====================

class DropPath(nn.Module):
    """随机深度 (Stochastic Depth)"""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class PatchEmbed(nn.Module):
    """图像到patch嵌入"""
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[nn.Module] = None,
        flatten: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size
        
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.flatten = flatten
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        x = self.norm(x)
        return x


class MLP(nn.Module):
    """MLP模块"""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ==================== Vision Transformer ====================

class Attention(nn.Module):
    """多头自注意力"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn + mask
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer块"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT)"""
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        global_pool: str = 'token',
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.global_pool = global_pool
        
        # Patch嵌入
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        
        # CLS token和位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # 随机深度衰减
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Transformer块
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            for i in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        
        # 分类头
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module_weights)
    
    def _init_module_weights(self, m: nn.Module):
        """初始化模块权重"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征"""
        x = self.patch_embed(x)
        
        # 添加CLS token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer块
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        
        if self.global_pool == 'token':
            x = x[:, 0]
        else:
            x = x.mean(dim=1)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.forward_features(x)
        x = self.head(x)
        return x


# ==================== Swin Transformer ====================

class WindowAttention(nn.Module):
    """窗口注意力"""
    
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 相对位置偏置
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        
        # 相对位置索引
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        
        # 相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer块"""
    
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if min(input_resolution) <= window_size:
            self.shift_size = 0
            self.window_size = min(input_resolution)
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=(window_size, window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        
        if self.shift_size > 0:
            H, W = input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = img_mask.view(
                1, H // window_size, window_size, W // window_size, window_size, 1
            ).permute(0, 1, 3, 2, 5, 4).contiguous().view(-1, window_size * window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        
        self.register_buffer('attn_mask', attn_mask)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # 循环移位
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # 窗口分割
        x_windows = shifted_x.view(
            B, H // self.window_size, self.window_size,
            W // self.window_size, self.window_size, C
        ).permute(0, 1, 3, 2, 5, 4).contiguous().view(-1, self.window_size * self.window_size, C)
        
        # 窗口注意力
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        
        # 合并窗口
        attn_windows = attn_windows.view(
            -1, self.window_size, self.window_size, C
        )
        shifted_x = attn_windows.view(
            B, H // self.window_size, W // self.window_size,
            self.window_size, self.window_size, C
        ).permute(0, 1, 3, 2, 5, 4).contiguous().view(B, H, W, C)
        
        # 反向循环移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class PatchMerging(nn.Module):
    """Patch合并"""
    
    def __init__(
        self,
        input_resolution: Tuple[int, int],
        dim: int,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        
        x = x.view(B, H, W, C)
        
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4 * C)
        
        x = self.norm(x)
        x = self.reduction(x)
        
        return x


class SwinTransformer(nn.Module):
    """Swin Transformer"""
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        patch_norm: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        
        # Patch嵌入
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )
        
        patches_resolution = self.patch_embed.grid_size
        self.patches_resolution = patches_resolution
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # 随机深度衰减
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # 构建层
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = nn.ModuleDict({
                'blocks': nn.ModuleList([
                    SwinTransformerBlock(
                        dim=embed_dim * (2 ** i_layer),
                        input_resolution=(
                            patches_resolution[0] // (2 ** i_layer),
                            patches_resolution[1] // (2 ** i_layer),
                        ),
                        num_heads=num_heads[i_layer],
                        window_size=window_size,
                        shift_size=0 if (i_block % 2 == 0) else window_size // 2,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:i_layer]) + i_block],
                        norm_layer=norm_layer,
                    )
                    for i_block in range(depths[i_layer])
                ]),
                'downsample': PatchMerging(
                    input_resolution=(
                        patches_resolution[0] // (2 ** i_layer),
                        patches_resolution[1] // (2 ** i_layer),
                    ),
                    dim=embed_dim * (2 ** i_layer),
                    norm_layer=norm_layer,
                ) if (i_layer < self.num_layers - 1) else None,
            })
            self.layers.append(layer)
        
        self.norm = norm_layer(embed_dim * 2 ** (self.num_layers - 1))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dim * 2 ** (self.num_layers - 1), num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        self.apply(self._init_module_weights)
    
    def _init_module_weights(self, m: nn.Module):
        """初始化模块权重"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征"""
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        
        for layer in self.layers:
            for block in layer['blocks']:
                x = block(x)
            if layer['downsample'] is not None:
                x = layer['downsample'](x)
        
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2)).transpose(1, 2).squeeze(-1)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.forward_features(x)
        x = self.head(x)
        return x


# ==================== EfficientNet ====================

class MBConvBlock(nn.Module):
    """MBConv块 - EfficientNet核心模块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: int = 6,
        se_ratio: float = 0.25,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        
        mid_channels = in_channels * expand_ratio
        has_se = se_ratio is not None and se_ratio > 0.0
        
        layers = []
        
        # Expansion phase
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.SiLU(inplace=True),
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(
                mid_channels, mid_channels,
                kernel_size, stride=stride,
                padding=kernel_size // 2, groups=mid_channels, bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
        ])
        
        # Squeeze and Excitation
        if has_se:
            se_channels = max(1, int(in_channels * se_ratio))
            layers.extend([
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(mid_channels, se_channels, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(se_channels, mid_channels, 1),
                nn.Sigmoid(),
            ])
        
        # Output phase
        layers.extend([
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.block = nn.Sequential(*layers)
        
        # 残差连接
        self.use_residual = (stride == 1 and in_channels == out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block[0](x) if hasattr(self.block[0], 'weight') else x
        
        # 手动前向传播以处理SE
        idx = 0
        if len(self.block) > 7:  # 有expansion
            out = self.block[0](x)
            out = self.block[1](out)
            out = self.block[2](out)
            idx = 3
        
        # Depthwise
        out = self.block[idx](out)
        out = self.block[idx + 1](out)
        out = self.block[idx + 2](out)
        
        # SE
        if len(self.block) > idx + 7:
            se = self.block[idx + 3](out)
            se = self.block[idx + 4](se)
            se = self.block[idx + 5](se)
            se = self.block[idx + 6](se)
            out = out * se
            idx += 4
        
        # Output
        out = self.block[-2](out)
        out = self.block[-1](out)
        
        # 残差
        if self.use_residual:
            if self.training and self.drop_rate > 0:
                out = F.dropout(out, self.drop_rate, training=True)
            out = out + x
        
        return out


class EfficientNet(nn.Module):
    """EfficientNet"""
    
    # EfficientNet架构参数
    ARCH_PARAMS = {
        'efficientnet_b0': {
            'width_mult': 1.0, 'depth_mult': 1.0, 'resolution': 224,
            'dropout_rate': 0.2,
        },
        'efficientnet_b1': {
            'width_mult': 1.0, 'depth_mult': 1.1, 'resolution': 240,
            'dropout_rate': 0.2,
        },
        'efficientnet_b2': {
            'width_mult': 1.1, 'depth_mult': 1.2, 'resolution': 260,
            'dropout_rate': 0.3,
        },
        'efficientnet_b3': {
            'width_mult': 1.2, 'depth_mult': 1.4, 'resolution': 300,
            'dropout_rate': 0.3,
        },
        'efficientnet_b4': {
            'width_mult': 1.4, 'depth_mult': 1.8, 'resolution': 380,
            'dropout_rate': 0.4,
        },
        'efficientnet_b5': {
            'width_mult': 1.6, 'depth_mult': 2.2, 'resolution': 456,
            'dropout_rate': 0.4,
        },
        'efficientnet_b6': {
            'width_mult': 1.8, 'depth_mult': 2.6, 'resolution': 528,
            'dropout_rate': 0.5,
        },
        'efficientnet_b7': {
            'width_mult': 2.0, 'depth_mult': 3.1, 'resolution': 600,
            'dropout_rate': 0.5,
        },
    }
    
    # 基础架构
    BASE_ARCH = [
        # [expand_ratio, channels, layers, kernel_size, stride]
        [1, 16, 1, 3, 1],
        [6, 24, 2, 3, 2],
        [6, 40, 2, 5, 2],
        [6, 80, 3, 3, 2],
        [6, 112, 3, 5, 1],
        [6, 192, 4, 5, 2],
        [6, 320, 1, 3, 1],
    ]
    
    def __init__(
        self,
        variant: str = 'efficientnet_b0',
        in_chans: int = 3,
        num_classes: int = 1000,
        drop_connect_rate: float = 0.2,
    ):
        super().__init__()
        
        params = self.ARCH_PARAMS[variant]
        width_mult = params['width_mult']
        depth_mult = params['depth_mult']
        dropout_rate = params['dropout_rate']
        
        # Stem
        out_channels = self._round_filters(32, width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        
        # 构建块
        in_channels = out_channels
        blocks = []
        total_blocks = sum(int(round(d * depth_mult)) for _, _, d, _, _ in self.BASE_ARCH)
        block_idx = 0
        
        for expand_ratio, channels, num_layers, kernel_size, stride in self.BASE_ARCH:
            out_channels = self._round_filters(channels, width_mult)
            num_layers = int(round(num_layers * depth_mult))
            
            for i in range(num_layers):
                drop_rate = drop_connect_rate * block_idx / total_blocks
                blocks.append(
                    MBConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride if i == 0 else 1,
                        expand_ratio=expand_ratio,
                        drop_rate=drop_rate,
                    )
                )
                in_channels = out_channels
                block_idx += 1
        
        self.blocks = nn.Sequential(*blocks)
        
        # Head
        final_channels = self._round_filters(1280, width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, final_channels, 1, bias=False),
            nn.BatchNorm2d(final_channels),
            nn.SiLU(inplace=True),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(final_channels, num_classes)
        
        self._init_weights()
    
    def _round_filters(self, filters: int, width_mult: float) -> int:
        """计算并舍入通道数"""
        if not width_mult:
            return filters
        divisor = 8
        min_depth = divisor
        new_filters = max(divisor, int(filters * width_mult + divisor / 2) // divisor * divisor)
        if new_filters < 0.9 * filters * width_mult:
            new_filters += divisor
        return int(new_filters)
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.avgpool(x).flatten(1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


# ==================== ConvNeXt ====================

class ConvNeXtBlock(nn.Module):
    """ConvNeXt块"""
    
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim)
        ) if layer_scale_init_value > 0 else None
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        
        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    """ConvNeXt"""
    
    ARCH_PARAMS = {
        'convnext_tiny': {'depths': [3, 3, 9, 3], 'dims': [96, 192, 384, 768]},
        'convnext_small': {'depths': [3, 3, 27, 3], 'dims': [96, 192, 384, 768]},
        'convnext_base': {'depths': [3, 3, 27, 3], 'dims': [128, 256, 512, 1024]},
        'convnext_large': {'depths': [3, 3, 27, 3], 'dims': [192, 384, 768, 1536]},
        'convnext_xlarge': {'depths': [3, 3, 27, 3], 'dims': [256, 512, 1024, 2048]},
    }
    
    def __init__(
        self,
        variant: str = 'convnext_tiny',
        in_chans: int = 3,
        num_classes: int = 1000,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        head_init_scale: float = 1.0,
    ):
        super().__init__()
        
        params = self.ARCH_PARAMS[variant]
        depths = params['depths']
        dims = params['dims']
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm(dims[0], eps=1e-6),
        )
        
        # 随机深度衰减
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # 构建层
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[
                    ConvNeXtBlock(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]
        
        # 下采样层
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Identity())  # 第一个不需要下采样
        for i in range(len(dims) - 1):
            downsample = nn.Sequential(
                nn.LayerNorm(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample)
        
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        
        self._init_weights(head_init_scale)
    
    def _init_weights(self, head_init_scale: float):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征"""
        x = self.stem(x)
        
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = self.downsample_layers[i](x)
            x = stage(x)
        
        x = self.norm(x.mean([-2, -1]))
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.forward_features(x)
        x = self.head(x)
        return x


# ==================== 主函数 ====================

def main():
    """测试视觉模型架构"""
    print("视觉模型架构测试")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 测试Vision Transformer
    print("\n测试Vision Transformer...")
    vit = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
    )
    x = torch.randn(2, 3, 224, 224)
    out = vit(x)
    print(f"ViT output shape: {out.shape}")
    print(f"ViT parameters: {sum(p.numel() for p in vit.parameters()) / 1e6:.2f}M")
    
    # 测试EfficientNet
    print("\n测试EfficientNet...")
    effnet = EfficientNet(variant='efficientnet_b0')
    out = effnet(x)
    print(f"EfficientNet output shape: {out.shape}")
    print(f"EfficientNet parameters: {sum(p.numel() for p in effnet.parameters()) / 1e6:.2f}M")
    
    # 测试ConvNeXt
    print("\n测试ConvNeXt...")
    convnext = ConvNeXt(variant='convnext_tiny')
    out = convnext(x)
    print(f"ConvNeXt output shape: {out.shape}")
    print(f"ConvNeXt parameters: {sum(p.numel() for p in convnext.parameters()) / 1e6:.2f}M")
    
    print("\n视觉模型架构测试完成")


if __name__ == "__main__":
    main()
