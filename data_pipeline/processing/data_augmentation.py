"""
AGI统一框架 - 数据处理与增强
实现数据加载、预处理、增强、采样等核心功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Callable, Union
from dataclasses import dataclass
import random
import math
from collections import defaultdict


# ==================== 数据增强 ====================

class Compose:
    """组合多个变换"""
    
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
        
    def __call__(self, x: Any) -> Any:
        for t in self.transforms:
            x = t(x)
        return x


class RandomCrop:
    """随机裁剪"""
    
    def __init__(self, size: Tuple[int, int], padding: int = 0):
        self.size = size
        self.padding = padding
        
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self.padding > 0:
            img = F.pad(img, [self.padding] * 4)
            
        h, w = img.shape[-2:]
        th, tw = self.size
        
        if h < th or w < tw:
            return img
            
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        
        return img[..., i:i+th, j:j+tw]


class RandomHorizontalFlip:
    """随机水平翻转"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
        
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return torch.flip(img, [-1])
        return img


class RandomVerticalFlip:
    """随机垂直翻转"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
        
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return torch.flip(img, [-2])
        return img


class RandomRotation:
    """随机旋转"""
    
    def __init__(self, degrees: float):
        self.degrees = degrees
        
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        angle = random.uniform(-self.degrees, self.degrees)
        return self._rotate(img, angle)
    
    def _rotate(self, img: torch.Tensor, angle: float) -> torch.Tensor:
        angle_rad = math.radians(angle)
        
        h, w = img.shape[-2:]
        center = torch.tensor([w / 2, h / 2])
        
        # 旋转矩阵
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        rotation_matrix = torch.tensor([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        # 创建网格
        grid = F.affine_grid(
            torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0]
            ], dtype=img.dtype, device=img.device).unsqueeze(0),
            img.unsqueeze(0).size()
        )
        
        return F.grid_sample(img.unsqueeze(0), grid, 
                            mode='bilinear', padding_mode='zeros',
                            align_corners=False).squeeze(0)


class ColorJitter:
    """颜色抖动"""
    
    def __init__(self, brightness: float = 0.0, contrast: float = 0.0,
                 saturation: float = 0.0, hue: float = 0.0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # 亮度
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            img = img * factor
            
        # 对比度
        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            mean = img.mean()
            img = (img - mean) * factor + mean
            
        # 饱和度
        if self.saturation > 0:
            factor = 1.0 + random.uniform(-self.saturation, self.saturation)
            gray = img.mean(dim=0, keepdim=True).expand_as(img)
            img = gray + (img - gray) * factor
            
        return img


class RandomErasing:
    """随机擦除"""
    
    def __init__(self, p: float = 0.5, scale: Tuple[float, float] = (0.02, 0.33),
                 ratio: Tuple[float, float] = (0.3, 3.3), value: float = 0.0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img
            
        h, w = img.shape[-2:]
        area = h * w
        
        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)
            
            eh = int(round(math.sqrt(target_area * aspect_ratio)))
            ew = int(round(math.sqrt(target_area / aspect_ratio)))
            
            if eh < h and ew < w:
                i = random.randint(0, h - eh)
                j = random.randint(0, w - ew)
                img[..., i:i+eh, j:j+ew] = self.value
                break
                
        return img


class MixUp:
    """MixUp增强"""
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        
    def __call__(self, x1: torch.Tensor, x2: torch.Tensor,
                 y1: torch.Tensor, y2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        
        x = lam * x1 + (1 - lam) * x2
        y = lam * y1 + (1 - lam) * y2
        
        return x, y


class CutMix:
    """CutMix增强"""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        
    def __call__(self, x1: torch.Tensor, x2: torch.Tensor,
                 y1: torch.Tensor, y2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        
        h, w = x1.shape[-2:]
        
        # 计算裁剪区域
        cut_rat = math.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        
        bbx1 = max(0, cx - cut_w // 2)
        bby1 = max(0, cy - cut_h // 2)
        bbx2 = min(w, cx + cut_w // 2)
        bby2 = min(h, cy + cut_h // 2)
        
        x = x1.clone()
        x[..., bby1:bby2, bbx1:bbx2] = x2[..., bby1:bby2, bbx1:bbx2]
        
        # 调整lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        
        y = lam * y1 + (1 - lam) * y2
        
        return x, y


class AutoAugment:
    """AutoAugment策略"""
    
    def __init__(self):
        self.policies = self._get_policies()
        
    def _get_policies(self) -> List[List[Tuple[Callable, float, float]]]:
        """获取AutoAugment策略"""
        policies = [
            [('shearX', 0.9), ('invert', 0.8)],
            [('shearY', 0.9), ('invert', 0.8)],
            [('translateX', 0.9), ('equalize', 0.8)],
            [('translateY', 0.9), ('equalize', 0.8)],
            [('rotate', 0.9), ('equalize', 0.8)],
            [('solarize', 0.8), ('equalize', 0.8)],
            [('solarize', 0.8), ('invert', 0.8)],
            [('color', 0.9), ('equalize', 0.8)],
            [('posterize', 0.8), ('solarize', 0.8)],
            [('solarize', 0.2), ('solarize', 0.8)],
        ]
        return policies
        
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        policy = random.choice(self.policies)
        
        for name, prob in policy:
            if random.random() < prob:
                img = self._apply_op(img, name)
                
        return img
    
    def _apply_op(self, img: torch.Tensor, name: str) -> torch.Tensor:
        """应用操作"""
        if name == 'invert':
            return 1.0 - img
        elif name == 'equalize':
            return self._equalize(img)
        elif name == 'solarize':
            return torch.where(img > 0.5, 1.0 - img, img)
        elif name == 'posterize':
            bits = random.randint(1, 4)
            return torch.round(img * (2 ** bits)) / (2 ** bits)
        return img
    
    def _equalize(self, img: torch.Tensor) -> torch.Tensor:
        """直方图均衡化"""
        # 简化实现
        return (img - img.min()) / (img.max() - img.min() + 1e-8)


# ==================== 数据采样器 ====================

class StratifiedSampler:
    """分层采样器"""
    
    def __init__(self, labels: List[int], batch_size: int):
        self.labels = labels
        self.batch_size = batch_size
        
        # 按类别分组
        self.label_indices = defaultdict(list)
        for i, label in enumerate(labels):
            self.label_indices[label].append(i)
            
        self.num_classes = len(self.label_indices)
        
    def __iter__(self):
        # 打乱每个类别内的索引
        for label in self.label_indices:
            random.shuffle(self.label_indices[label])
            
        # 生成批次
        iterators = {label: iter(indices) 
                    for label, indices in self.label_indices.items()}
        
        while True:
            batch = []
            for label, it in iterators.items():
                try:
                    idx = next(it)
                    batch.append(idx)
                except StopIteration:
                    pass
                    
            if len(batch) < self.batch_size:
                break
                
            random.shuffle(batch)
            yield batch[:self.batch_size]
            
    def __len__(self):
        return len(self.labels) // self.batch_size


class ImbalancedDatasetSampler:
    """不平衡数据集采样器"""
    
    def __init__(self, labels: List[int]):
        self.labels = labels
        
        # 计算每个类别的数量
        label_counts = defaultdict(int)
        for label in labels:
            label_counts[label] += 1
            
        # 计算权重
        self.weights = [1.0 / label_counts[label] for label in labels]
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
        
    def __iter__(self):
        indices = list(range(len(self.labels)))
        random.shuffle(indices)
        
        # 根据权重采样
        sampled = []
        for idx in indices:
            if random.random() < self.weights[idx] * len(self.labels):
                sampled.append(idx)
                
        yield from sampled
        
    def __len__(self):
        return len(self.labels)


class CurriculumSampler:
    """课程学习采样器"""
    
    def __init__(self, difficulties: List[float], 
                 initial_threshold: float = 0.3,
                 increment: float = 0.1):
        self.difficulties = difficulties
        self.threshold = initial_threshold
        self.increment = increment
        
    def update_threshold(self):
        """更新难度阈值"""
        self.threshold = min(1.0, self.threshold + self.increment)
        
    def __iter__(self):
        # 选择难度低于阈值的样本
        eligible = [i for i, d in enumerate(self.difficulties) 
                   if d <= self.threshold]
        
        random.shuffle(eligible)
        yield from eligible
        
    def __len__(self):
        return sum(1 for d in self.difficulties if d <= self.threshold)


# ==================== 数据预处理 ====================

class Normalizer:
    """数据归一化"""
    
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)


class Denormalizer:
    """数据反归一化"""
    
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)


class StandardScaler:
    """标准化缩放器"""
    
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, data: np.ndarray):
        """拟合"""
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0) + 1e-8
        
    def transform(self, data: np.ndarray) -> np.ndarray:
        """转换"""
        return (data - self.mean) / self.std
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """拟合并转换"""
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """逆转换"""
        return data * self.std + self.mean


class MinMaxScaler:
    """MinMax缩放器"""
    
    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        self.feature_range = feature_range
        self.min = None
        self.max = None
        
    def fit(self, data: np.ndarray):
        """拟合"""
        self.min = data.min(axis=0)
        self.max = data.max(axis=0)
        
    def transform(self, data: np.ndarray) -> np.ndarray:
        """转换"""
        scale = (self.feature_range[1] - self.feature_range[0]) / (self.max - self.min + 1e-8)
        return (data - self.min) * scale + self.feature_range[0]
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """拟合并转换"""
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """逆转换"""
        scale = (self.max - self.min) / (self.feature_range[1] - self.feature_range[0])
        return (data - self.feature_range[0]) * scale + self.min


# ==================== 批处理 ====================

class BatchCollator:
    """批处理整理器"""
    
    def __init__(self, pad_value: float = 0.0, max_length: Optional[int] = None):
        self.pad_value = pad_value
        self.max_length = max_length
        
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        result = {}
        
        for key in batch[0].keys():
            values = [item[key] for item in batch]
            
            if isinstance(values[0], torch.Tensor):
                # 填充序列
                if values[0].dim() > 0:
                    max_len = self.max_length or max(v.size(0) for v in values)
                    padded = []
                    for v in values:
                        if v.size(0) < max_len:
                            padding = torch.full(
                                (max_len - v.size(0),) + v.shape[1:],
                                self.pad_value, dtype=v.dtype, device=v.device
                            )
                            padded.append(torch.cat([v, padding]))
                        else:
                            padded.append(v[:max_len])
                    result[key] = torch.stack(padded)
                else:
                    result[key] = torch.stack(values)
            else:
                result[key] = torch.tensor(values)
                
        return result


class DynamicBatchSampler:
    """动态批大小采样器"""
    
    def __init__(self, data_lengths: List[int], max_tokens: int = 4096,
                 min_batch_size: int = 1, max_batch_size: int = 256):
        self.data_lengths = data_lengths
        self.max_tokens = max_tokens
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        
    def __iter__(self):
        indices = list(range(len(self.data_lengths)))
        # 按长度排序
        indices.sort(key=lambda i: self.data_lengths[i], reverse=True)
        
        batch = []
        current_tokens = 0
        
        for idx in indices:
            length = self.data_lengths[idx]
            
            if batch and (current_tokens + length * (len(batch) + 1) > self.max_tokens
                         or len(batch) >= self.max_batch_size):
                yield batch
                batch = []
                current_tokens = 0
                
            batch.append(idx)
            current_tokens += length * len(batch)
            
        if batch:
            yield batch
            
    def __len__(self):
        return len(self.data_lengths)


# ==================== 数据验证 ====================

class DataValidator:
    """数据验证器"""
    
    def __init__(self, checks: List[Callable]):
        self.checks = checks
        
    def validate(self, data: Any) -> Tuple[bool, List[str]]:
        """验证数据"""
        errors = []
        
        for check in self.checks:
            try:
                if not check(data):
                    errors.append(f"Check {check.__name__} failed")
            except Exception as e:
                errors.append(f"Check {check.__name__} raised: {str(e)}")
                
        return len(errors) == 0, errors


def check_no_nan(data: torch.Tensor) -> bool:
    """检查无NaN"""
    return not torch.isnan(data).any()


def check_no_inf(data: torch.Tensor) -> bool:
    """检查无Inf"""
    return not torch.isinf(data).any()


def check_positive(data: torch.Tensor) -> bool:
    """检查为正"""
    return (data > 0).all()


def check_shape(data: torch.Tensor, expected_shape: Tuple[int, ...]) -> bool:
    """检查形状"""
    return data.shape == expected_shape
