"""
数据加载器和预处理 - Data Loaders and Preprocessing
实现各种数据集加载、预处理流水线、数据增强等功能
"""

import torch
from torch.utils.data import Dataset, DataLoader, Sampler, IterableDataset
import numpy as np
import os
import json
import pickle
import csv
import random
import math
import threading
import queue
import mmap
from typing import Dict, List, Optional, Tuple, Any, Callable, Union, Iterator
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from PIL import Image
import warnings

# ==================== 数据集基类 ====================

class BaseDataset(Dataset):
    """数据集基类"""
    
    def __init__(
        self,
        data_path: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.data_path = data_path
        self.transform = transform
        self.target_transform = target_transform
        
        self.data = []
        self.targets = []
        self._load_data()
    
    def _load_data(self):
        """加载数据
        
        默认实现：尝试根据 data_path 的扩展名自动选择加载方式。
        支持 .json, .csv, .pkl, .npy, .pt 格式。
        子类可覆盖此方法以实现自定义加载逻辑。
        """
        import warnings
        
        path = self.data_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据路径不存在: {path}")
        
        if os.path.isdir(path):
            # 目录模式：加载所有文件名作为数据，目录名作为目标
            self.data = []
            self.targets = []
            classes = sorted(os.listdir(path))
            class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
            for cls_name in classes:
                cls_dir = os.path.join(path, cls_name)
                if os.path.isdir(cls_dir):
                    for fname in os.listdir(cls_dir):
                        fpath = os.path.join(cls_dir, fname)
                        if os.path.isfile(fpath):
                            self.data.append(fpath)
                            self.targets.append(class_to_idx[cls_name])
        elif path.endswith('.json'):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        self.data.append(item[0])
                        self.targets.append(item[1])
                    elif isinstance(item, dict):
                        self.data.append(item.get('data', item.get('input', '')))
                        self.targets.append(item.get('target', item.get('label', 0)))
            elif isinstance(data, dict):
                self.data = data.get('data', data.get('inputs', []))
                self.targets = data.get('targets', data.get('labels', []))
        elif path.endswith('.csv'):
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
            if rows:
                # 第一行作为表头，假设最后一列为目标
                for row in rows[1:]:
                    if len(row) >= 2:
                        self.data.append(row[:-1])
                        try:
                            self.targets.append(float(row[-1]))
                        except ValueError:
                            self.targets.append(row[-1])
        elif path.endswith('.pkl'):
            with open(path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, (list, tuple)) and len(data) == 2:
                self.data, self.targets = data
            else:
                self.data = data
                self.targets = [0] * len(data)
        elif path.endswith('.npy'):
            import numpy as np
            arr = np.load(path)
            self.data = list(arr)
            self.targets = [0] * len(arr)
        elif path.endswith('.pt'):
            import torch
            data = torch.load(path, weights_only=False)
            if isinstance(data, dict):
                self.data = data.get('data', data.get('inputs', []))
                self.targets = data.get('targets', data.get('labels', []))
            else:
                self.data = list(data)
                self.targets = [0] * len(data)
        else:
            # 纯文本文件：每行一条数据
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            self.data = [line.strip() for line in lines if line.strip()]
            self.targets = [0] * len(self.data)
        
        # 确保数据长度一致
        if len(self.data) != len(self.targets):
            warnings.warn(
                f"数据量({len(self.data)})与目标量({len(self.targets)})不一致，"
                f"将截断到较小值"
            )
            min_len = min(len(self.data), len(self.targets))
            self.data = self.data[:min_len]
            self.targets = self.targets[:min_len]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        x = self.data[idx]
        y = self.targets[idx]
        
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        
        return x, y


# ==================== 图像数据集 ====================

class ImageFolderDataset(BaseDataset):
    """图像文件夹数据集"""
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.gif'),
    ):
        self.root_dir = root_dir
        self.extensions = extensions
        self.class_to_idx = {}
        self.samples = []
        
        super().__init__(root_dir, transform, target_transform)
    
    def _load_data(self):
        """加载图像数据"""
        # 扫描文件夹结构
        classes = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])
        
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        
        for cls in classes:
            cls_dir = os.path.join(self.root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(self.extensions):
                    path = os.path.join(cls_dir, fname)
                    self.samples.append((path, self.class_to_idx[cls]))
        
        self.data = [s[0] for s in self.samples]
        self.targets = [s[1] for s in self.samples]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.data[idx]
        target = self.targets[idx]
        
        # 加载图像
        img = Image.open(path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        
        return img, target


class ImageListDataset(Dataset):
    """图像列表数据集"""
    
    def __init__(
        self,
        image_list: List[str],
        labels: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        root_dir: str = "",
    ):
        self.image_list = image_list
        self.labels = labels or [0] * len(image_list)
        self.transform = transform
        self.root_dir = root_dir
    
    def __len__(self) -> int:
        return len(self.image_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.image_list[idx]
        if self.root_dir:
            path = os.path.join(self.root_dir, path)
        
        img = Image.open(path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, self.labels[idx]


# ==================== 文本数据集 ====================

class TextDataset(BaseDataset):
    """文本数据集"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Optional[Callable] = None,
        max_length: int = 512,
        transform: Optional[Callable] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        super().__init__(data_path, transform)
    
    def _load_data(self):
        """加载文本数据"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        self.data = [line.strip() for line in lines if line.strip()]
        self.targets = list(range(len(self.data)))  # 自回归目标
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.data[idx]
        
        if self.tokenizer:
            tokens = self.tokenizer(text, max_length=self.max_length)
            return tokens
        else:
            return {'text': text}


class TextClassificationDataset(Dataset):
    """文本分类数据集"""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: Optional[Callable] = None,
        max_length: int = 512,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        if self.tokenizer:
            tokens = self.tokenizer(text, max_length=self.max_length)
            tokens['labels'] = torch.tensor(label)
            return tokens
        else:
            return {'text': text, 'labels': torch.tensor(label)}


class JSONLDataset(Dataset):
    """JSONL格式数据集"""
    
    def __init__(
        self,
        data_path: str,
        text_key: str = 'text',
        label_key: Optional[str] = 'label',
        transform: Optional[Callable] = None,
    ):
        self.data_path = data_path
        self.text_key = text_key
        self.label_key = label_key
        self.transform = transform
        
        self.data = []
        self._load_data()
    
    def _load_data(self):
        """加载JSONL数据"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        result = {'text': item[self.text_key]}
        
        if self.label_key and self.label_key in item:
            result['label'] = item[self.label_key]
        
        if self.transform:
            result = self.transform(result)
        
        return result


# ==================== 序列数据集 ====================

class SequenceDataset(Dataset):
    """序列数据集"""
    
    def __init__(
        self,
        sequences: List[List[Any]],
        targets: Optional[List[Any]] = None,
        max_length: Optional[int] = None,
        pad_value: int = 0,
    ):
        self.sequences = sequences
        self.targets = targets or [None] * len(sequences)
        self.max_length = max_length or max(len(s) for s in sequences)
        self.pad_value = pad_value
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        seq = self.sequences[idx]
        target = self.targets[idx]
        
        # 填充或截断
        if len(seq) < self.max_length:
            seq = seq + [self.pad_value] * (self.max_length - len(seq))
        else:
            seq = seq[:self.max_length]
        
        return torch.tensor(seq, dtype=torch.long), target


class TimeSeriesDataset(Dataset):
    """时间序列数据集"""
    
    def __init__(
        self,
        data: np.ndarray,
        window_size: int,
        horizon: int = 1,
        stride: int = 1,
        transform: Optional[Callable] = None,
    ):
        self.data = data
        self.window_size = window_size
        self.horizon = horizon
        self.stride = stride
        self.transform = transform
        
        self.num_samples = (len(data) - window_size - horizon) // stride + 1
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        end = start + self.window_size
        
        x = self.data[start:end]
        y = self.data[end:end + self.horizon]
        
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        if self.transform:
            x, y = self.transform(x, y)
        
        return x, y


# ==================== 流式数据集 ====================

class StreamDataset(IterableDataset):
    """流式数据集"""
    
    def __init__(
        self,
        data_source: Any,
        transform: Optional[Callable] = None,
        buffer_size: int = 10000,
    ):
        self.data_source = data_source
        self.transform = transform
        self.buffer_size = buffer_size
    
    def __iter__(self) -> Iterator:
        """迭代数据"""
        buffer = []
        
        for item in self._read_data():
            buffer.append(item)
            
            if len(buffer) >= self.buffer_size:
                random.shuffle(buffer)
                for x in buffer:
                    if self.transform:
                        x = self.transform(x)
                    yield x
                buffer = []
        
        # 处理剩余数据
        if buffer:
            random.shuffle(buffer)
            for x in buffer:
                if self.transform:
                    x = self.transform(x)
                yield x
    
    def _read_data(self) -> Iterator:
        """读取数据
        
        默认实现：根据 data_source 类型自动选择读取方式。
        支持文件路径(str)、文件对象、可迭代对象等。
        子类可覆盖此方法以实现自定义读取逻辑。
        """
        source = self.data_source
        
        if isinstance(source, str):
            if not os.path.exists(source):
                raise FileNotFoundError(f"数据源文件不存在: {source}")
            
            if source.endswith('.json'):
                with open(source, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        yield item
            elif source.endswith('.csv'):
                with open(source, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)  # 跳过表头
                    for row in reader:
                        yield row
            elif source.endswith('.pkl'):
                with open(source, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, (list, tuple)):
                    for item in data:
                        yield item
                else:
                    yield data
            elif source.endswith('.npy'):
                import numpy as np
                arr = np.load(source)
                for item in arr:
                    yield item
            else:
                # 纯文本文件：逐行读取
                with open(source, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield line
        elif hasattr(source, 'read'):
            # 文件对象
            for line in source:
                line = line.strip()
                if line:
                    yield line
        elif isinstance(source, (list, tuple)):
            for item in source:
                yield item
        elif hasattr(source, '__iter__'):
            for item in source:
                yield item
        else:
            raise TypeError(
                f"不支持的数据源类型: {type(source)}，"
                f"请提供文件路径、文件对象或可迭代对象"
            )


class LineStreamDataset(StreamDataset):
    """行流数据集"""
    
    def __init__(
        self,
        file_path: str,
        transform: Optional[Callable] = None,
        buffer_size: int = 10000,
        skip_header: bool = False,
    ):
        self.file_path = file_path
        self.skip_header = skip_header
        super().__init__(file_path, transform, buffer_size)
    
    def _read_data(self) -> Iterator:
        """读取行数据"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            if self.skip_header:
                next(f)
            for line in f:
                if line.strip():
                    yield line.strip()


# ==================== 内存映射数据集 ====================

class MemoryMappedDataset(Dataset):
    """内存映射数据集 - 用于大文件"""
    
    def __init__(
        self,
        file_path: str,
        dtype: np.dtype = np.float32,
        shape: Optional[Tuple[int, ...]] = None,
    ):
        self.file_path = file_path
        self.dtype = dtype
        
        # 获取文件大小
        file_size = os.path.getsize(file_path)
        
        if shape:
            self.shape = shape
        else:
            # 假设是一维数据
            item_size = np.dtype(dtype).itemsize
            self.shape = (file_size // item_size,)
        
        # 内存映射
        self.mmap = np.memmap(
            file_path,
            dtype=dtype,
            mode='r',
            shape=self.shape,
        )
    
    def __len__(self) -> int:
        return self.shape[0]
    
    def __getitem__(self, idx: int) -> np.ndarray:
        return self.mmap[idx]
    
    def __del__(self):
        if hasattr(self, 'mmap'):
            del self.mmap


# ==================== 采样器 ====================

class StratifiedSampler(Sampler):
    """分层采样器"""
    
    def __init__(
        self,
        labels: List[int],
        num_samples: Optional[int] = None,
        replacement: bool = True,
    ):
        self.labels = np.array(labels)
        self.num_samples = num_samples or len(labels)
        self.replacement = replacement
        
        # 计算每个类别的索引
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_indices[label].append(idx)
        
        self.classes = list(self.class_indices.keys())
        self.num_classes = len(self.classes)
    
    def __iter__(self) -> Iterator[int]:
        """迭代采样索引"""
        # 每个类别采样相同数量
        samples_per_class = self.num_samples // self.num_classes
        
        indices = []
        for cls in self.classes:
            cls_indices = self.class_indices[cls]
            
            if self.replacement:
                sampled = np.random.choice(
                    cls_indices,
                    size=samples_per_class,
                    replace=True,
                )
            else:
                sampled = np.random.choice(
                    cls_indices,
                    size=min(samples_per_class, len(cls_indices)),
                    replace=False,
                )
            
            indices.extend(sampled.tolist())
        
        random.shuffle(indices)
        return iter(indices)
    
    def __len__(self) -> int:
        return self.num_samples


class WeightedSampler(Sampler):
    """加权采样器"""
    
    def __init__(
        self,
        weights: List[float],
        num_samples: int,
        replacement: bool = True,
    ):
        self.weights = torch.tensor(weights, dtype=torch.float)
        self.num_samples = num_samples
        self.replacement = replacement
    
    def __iter__(self) -> Iterator[int]:
        """迭代采样索引"""
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=self.replacement,
        )
        return iter(indices.tolist())
    
    def __len__(self) -> int:
        return self.num_samples


class DynamicBatchSampler(Sampler):
    """动态批次采样器 - 根据序列长度动态调整批次大小"""
    
    def __init__(
        self,
        data_source: Dataset,
        max_tokens: int = 8192,
        max_batch_size: int = 128,
        length_fn: Optional[Callable] = None,
        drop_last: bool = False,
    ):
        self.data_source = data_source
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.length_fn = length_fn or (lambda x: len(x) if hasattr(x, '__len__') else 1)
        self.drop_last = drop_last
        
        # 预计算长度
        self.lengths = [
            self.length_fn(data_source[i])
            for i in range(len(data_source))
        ]
    
    def __iter__(self) -> Iterator[List[int]]:
        """迭代批次索引"""
        # 按长度排序
        indices = list(range(len(self.data_source)))
        indices.sort(key=lambda i: self.lengths[i], reverse=True)
        
        batches = []
        current_batch = []
        current_tokens = 0
        
        for idx in indices:
            length = self.lengths[idx]
            
            if len(current_batch) >= self.max_batch_size or current_tokens + length > self.max_tokens:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [idx]
                current_tokens = length
            else:
                current_batch.append(idx)
                current_tokens += length
        
        if current_batch and not self.drop_last:
            batches.append(current_batch)
        
        random.shuffle(batches)
        
        for batch in batches:
            yield batch
    
    def __len__(self) -> int:
        """估计批次数量"""
        return len(self.data_source) // (self.max_tokens // 100)


# ==================== 数据预处理 ====================

class Compose:
    """组合多个变换"""
    
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    
    def __call__(self, x: Any) -> Any:
        for t in self.transforms:
            x = t(x)
        return x


class ToTensor:
    """转换为张量"""
    
    def __call__(self, x: Any) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        elif isinstance(x, Image.Image):
            return torch.tensor(np.array(x), dtype=torch.float32).permute(2, 0, 1) / 255.0
        elif isinstance(x, (list, tuple)):
            return torch.tensor(x)
        else:
            return torch.tensor(x)


class Normalize:
    """标准化"""
    
    def __init__(
        self,
        mean: List[float],
        std: List[float],
        inplace: bool = False,
    ):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.inplace = inplace
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self.inplace:
            x = x.clone()
        
        mean = self.mean.to(x.device).view(-1, 1, 1)
        std = self.std.to(x.device).view(-1, 1, 1)
        
        return (x - mean) / std


class Resize:
    """调整大小"""
    
    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        interpolation: int = Image.BILINEAR,
    ):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.interpolation = interpolation
    
    def __call__(self, img: Image.Image) -> Image.Image:
        return img.resize(self.size, self.interpolation)


class CenterCrop:
    """中心裁剪"""
    
    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
    
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        th, tw = self.size
        
        i = (h - th) // 2
        j = (w - tw) // 2
        
        return img.crop((j, i, j + tw, i + th))


class RandomCrop:
    """随机裁剪"""
    
    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        padding: Optional[int] = None,
    ):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.padding = padding
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if self.padding:
            img = img.pad(img, self.padding)
        
        w, h = img.size
        th, tw = self.size
        
        if w == tw and h == th:
            return img
        
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        
        return img.crop((j, i, j + tw, i + th))


class RandomHorizontalFlip:
    """随机水平翻转"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class RandomRotation:
    """随机旋转"""
    
    def __init__(self, degrees: Union[float, Tuple[float, float]]):
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees
    
    def __call__(self, img: Image.Image) -> Image.Image:
        angle = random.uniform(self.degrees[0], self.degrees[1])
        return img.rotate(angle)


class ColorJitter:
    """颜色抖动"""
    
    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, img: Image.Image) -> Image.Image:
        # 亮度
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            img = Image.eval(img, lambda x: x * factor)
        
        # 对比度
        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            mean = np.mean(np.array(img))
            img = Image.eval(img, lambda x: (x - mean) * factor + mean)
        
        # 饱和度
        if self.saturation > 0:
            factor = 1.0 + random.uniform(-self.saturation, self.saturation)
            img = self._adjust_saturation(img, factor)
        
        return img
    
    def _adjust_saturation(self, img: Image.Image, factor: float) -> Image.Image:
        """调整饱和度"""
        gray = img.convert('L').convert('RGB')
        return Image.blend(gray, img, factor)


# ==================== 文本预处理 ====================

class Tokenizer:
    """简单分词器"""
    
    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        unk_token: str = '<unk>',
        pad_token: str = '<pad>',
        bos_token: str = '<bos>',
        eos_token: str = '<eos>',
    ):
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        
        if vocab:
            self.vocab = vocab
        else:
            self.vocab = {
                pad_token: 0,
                unk_token: 1,
                bos_token: 2,
                eos_token: 3,
            }
        
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
    
    def build_vocab(self, texts: List[str], min_freq: int = 1):
        """构建词汇表"""
        counter = Counter()
        for text in texts:
            tokens = text.split()
            counter.update(tokens)
        
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """编码文本"""
        tokens = text.split()
        
        if add_bos:
            tokens = [self.bos_token] + tokens
        if add_eos:
            tokens = tokens + [self.eos_token]
        
        ids = [self.vocab.get(t, self.vocab[self.unk_token]) for t in tokens]
        
        # 截断
        if max_length:
            ids = ids[:max_length]
        
        # 创建attention mask
        attention_mask = [1] * len(ids)
        
        # 填充
        if max_length:
            pad_length = max_length - len(ids)
            ids = ids + [self.vocab[self.pad_token]] * pad_length
            attention_mask = attention_mask + [0] * pad_length
        
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """解码ID"""
        tokens = []
        for id in ids:
            token = self.inv_vocab.get(id, self.unk_token)
            if skip_special_tokens and token in [self.pad_token, self.bos_token, self.eos_token]:
                continue
            tokens.append(token)
        return ' '.join(tokens)
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


class BPETokenizer:
    """BPE分词器"""
    
    def __init__(self, vocab_path: Optional[str] = None):
        self.vocab = {}
        self.merges = []
        
        if vocab_path:
            self.load(vocab_path)
    
    def load(self, vocab_path: str):
        """加载词汇表"""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '=' in line:
                    token, rank = line.strip().split('=')
                    self.vocab[token] = int(rank)
    
    def train(self, texts: List[str], num_merges: int = 10000):
        """训练BPE"""
        # 初始词汇表（字符级别）
        vocab = set()
        for text in texts:
            vocab.update(text)
        
        self.vocab = {c: i for i, c in enumerate(sorted(vocab))}
        
        # 统计字符对频率
        for _ in range(num_merges):
            pairs = Counter()
            for text in texts:
                tokens = list(text)
                for i in range(len(tokens) - 1):
                    pairs[(tokens[i], tokens[i + 1])] += 1
            
            if not pairs:
                break
            
            # 找到最频繁的字符对
            best_pair = max(pairs, key=pairs.get)
            
            # 合并
            new_token = best_pair[0] + best_pair[1]
            self.vocab[new_token] = len(self.vocab)
            self.merges.append(best_pair)
            
            # 更新文本
            new_texts = []
            for text in texts:
                tokens = list(text)
                i = 0
                while i < len(tokens) - 1:
                    if (tokens[i], tokens[i + 1]) == best_pair:
                        tokens[i] = new_token
                        tokens.pop(i + 1)
                    else:
                        i += 1
                new_texts.append(''.join(tokens))
            texts = new_texts
    
    def encode(self, text: str) -> List[int]:
        """编码文本"""
        tokens = list(text)
        
        for a, b in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == a and tokens[i + 1] == b:
                    tokens[i] = a + b
                    tokens.pop(i + 1)
                else:
                    i += 1
        
        return [self.vocab.get(t, 0) for t in tokens]


# ==================== 数据加载器工具 ====================

class PrefetchLoader:
    """预取数据加载器"""
    
    def __init__(
        self,
        loader: DataLoader,
        device: torch.device,
        prefetch_factor: int = 2,
    ):
        self.loader = loader
        self.device = device
        self.prefetch_factor = prefetch_factor
        
        self.queue = queue.Queue(maxsize=prefetch_factor)
        self.stop_event = threading.Event()
        self.loader_thread = None
    
    def start(self):
        """启动预取"""
        self.stop_event.clear()
        self.loader_thread = threading.Thread(target=self._load_loop)
        self.loader_thread.daemon = True
        self.loader_thread.start()
    
    def stop(self):
        """停止预取"""
        self.stop_event.set()
        if self.loader_thread:
            self.loader_thread.join()
    
    def _load_loop(self):
        """加载循环"""
        for batch in self.loader:
            if self.stop_event.is_set():
                break
            
            # 移动到设备
            batch = self._move_to_device(batch)
            self.queue.put(batch)
        
        self.queue.put(None)  # 结束标记
    
    def _move_to_device(self, batch: Any) -> Any:
        """移动批次到设备"""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        elif isinstance(batch, dict):
            return {k: self._move_to_device(v) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._move_to_device(x) for x in batch)
        else:
            return batch
    
    def __iter__(self) -> Iterator:
        """迭代"""
        self.start()
        try:
            while True:
                batch = self.queue.get()
                if batch is None:
                    break
                yield batch
        finally:
            self.stop()


class InfiniteLoader:
    """无限数据加载器"""
    
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.iterator = None
    
    def __iter__(self) -> Iterator:
        return self
    
    def __next__(self) -> Any:
        if self.iterator is None:
            self.iterator = iter(self.loader)
        
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


# ==================== 数据集工具 ====================

def split_dataset(
    dataset: Dataset,
    ratios: List[float],
    seed: int = 42,
) -> List[Dataset]:
    """分割数据集"""
    assert sum(ratios) == 1.0, "Ratios must sum to 1"
    
    indices = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(indices)
    
    datasets = []
    start = 0
    
    for ratio in ratios:
        size = int(len(indices) * ratio)
        subset_indices = indices[start:start + size]
        
        class Subset(Dataset):
            def __init__(self, data, indices):
                self.data = data
                self.indices = indices
            
            def __len__(self):
                return len(self.indices)
            
            def __getitem__(self, idx):
                return self.data[self.indices[idx]]
        
        datasets.append(Subset(dataset, subset_indices))
        start += size
    
    return datasets


def concat_datasets(datasets: List[Dataset]) -> Dataset:
    """连接数据集"""
    class ConcatDataset(Dataset):
        def __init__(self, datasets):
            self.datasets = datasets
            self.cumulative_sizes = np.cumsum([len(d) for d in datasets])
        
        def __len__(self):
            return self.cumulative_sizes[-1]
        
        def __getitem__(self, idx):
            dataset_idx = np.searchsorted(self.cumulative_sizes, idx, side='right')
            if dataset_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            return self.datasets[dataset_idx][sample_idx]
    
    return ConcatDataset(datasets)


# ==================== 主函数 ====================

def main():
    """测试数据加载器和预处理"""
    print("数据加载器和预处理测试")
    
    # 测试分词器
    print("\n测试分词器...")
    texts = ["hello world", "this is a test", "hello test"]
    tokenizer = Tokenizer()
    tokenizer.build_vocab(texts)
    
    encoded = tokenizer.encode("hello world test", max_length=10, add_bos=True, add_eos=True)
    print(f"Encoded: {encoded['input_ids'].tolist()}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    # 测试变换
    print("\n测试变换...")
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    # 测试序列数据集
    print("\n测试序列数据集...")
    sequences = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
    seq_dataset = SequenceDataset(sequences, max_length=5)
    print(f"Sequence dataset length: {len(seq_dataset)}")
    seq, _ = seq_dataset[0]
    print(f"Sample sequence: {seq.tolist()}")
    
    # 测试时间序列数据集
    print("\n测试时间序列数据集...")
    ts_data = np.random.randn(1000).astype(np.float32)
    ts_dataset = TimeSeriesDataset(ts_data, window_size=10, horizon=5)
    print(f"Time series dataset length: {len(ts_dataset)}")
    x, y = ts_dataset[0]
    print(f"Sample shapes: x={x.shape}, y={y.shape}")
    
    # 测试分层采样器
    print("\n测试分层采样器...")
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    sampler = StratifiedSampler(labels, num_samples=9)
    indices = list(sampler)
    print(f"Stratified sample indices: {indices}")
    
    print("\n数据加载器和预处理测试完成")


if __name__ == "__main__":
    main()
