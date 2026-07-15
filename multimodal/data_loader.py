"""
多模态数据加载器 - Multimodal Data Loader

支持图像-文本、视频-文本、音频-文本配对数据加载
实现数据增强和预处理，基于 PyTorch 实现

作者: UFO Framework Team
"""

import os
import json
import random
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union,
    Iterator, Sequence
)
from dataclasses import dataclass, field
from enum import Enum
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, DistributedSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image


class ModalityPairType(Enum):
    """模态配对类型"""
    IMAGE_TEXT = "image_text"
    VIDEO_TEXT = "video_text"
    AUDIO_TEXT = "audio_text"
    IMAGE_AUDIO_TEXT = "image_audio_text"
    VIDEO_AUDIO_TEXT = "video_audio_text"


@dataclass
class MultimodalDataConfig:
    """多模态数据配置"""
    # 数据路径
    data_root: str = "./data"
    image_dir: Optional[str] = None
    video_dir: Optional[str] = None
    audio_dir: Optional[str] = None
    annotation_file: Optional[str] = None

    # 模态配对类型
    pair_type: ModalityPairType = ModalityPairType.IMAGE_TEXT

    # 图像配置
    image_size: int = 224
    image_mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    image_std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

    # 视频配置
    num_frames: int = 8
    frame_sample_rate: int = 1

    # 音频配置
    sample_rate: int = 16000
    n_mels: int = 128
    max_audio_length: float = 30.0  # 秒

    # 文本配置
    max_text_length: int = 77
    tokenizer_name: str = "bert-base-uncased"

    # 数据增强
    enable_augmentation: bool = True
    augmentation_prob: float = 0.5
    random_crop: bool = True
    random_flip: bool = True
    color_jitter: bool = True
    mixup_alpha: float = 0.0

    # 数据加载
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    drop_last: bool = True
    pin_memory: bool = True

    # 分布式
    distributed: bool = False
    world_size: int = 1
    rank: int = 0

    def __post_init__(self):
        if self.image_dir is None:
            self.image_dir = os.path.join(self.data_root, "images")
        if self.video_dir is None:
            self.video_dir = os.path.join(self.data_root, "videos")
        if self.audio_dir is None:
            self.audio_dir = os.path.join(self.data_root, "audio")


class ImageTextPair(Dataset):
    """
    图像-文本配对数据集

    支持的数据格式:
    - COCO格式: {"images": [...], "annotations": [...]}
    - LAION格式: ["image_path\tcaption", ...]
    - 自定义格式: [{"image": "path", "caption": "text"}, ...]
    """

    def __init__(
        self,
        config: MultimodalDataConfig,
        transform: Optional[Callable] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.config = config
        self.transform = transform or self._build_transform()
        self.tokenizer = tokenizer
        self.samples: List[Dict[str, Any]] = []

        self._load_data()

    def _load_data(self):
        """加载数据"""
        if self.config.annotation_file:
            ann_path = Path(self.config.annotation_file)
            if not ann_path.exists():
                raise FileNotFoundError(f"Annotation file not found: {ann_path}")

            with open(ann_path, 'r', encoding='utf-8') as f:
                if ann_path.suffix == '.json':
                    data = json.load(f)
                    self._parse_json_format(data)
                else:
                    # 假设是TSV格式
                    self._parse_tsv_format(f)
        else:
            # 自动扫描图像目录
            self._scan_directory()

    def _parse_json_format(self, data: Dict):
        """解析JSON格式（COCO风格）"""
        if "images" in data and "annotations" in data:
            # COCO格式
            image_map = {img["id"]: img for img in data["images"]}
            for ann in data["annotations"]:
                image_id = ann.get("image_id")
                if image_id in image_map:
                    image_path = os.path.join(
                        self.config.image_dir,
                        image_map[image_id].get("file_name", "")
                    )
                    self.samples.append({
                        "image": image_path,
                        "text": ann.get("caption", ""),
                    })
        elif isinstance(data, list):
            # 自定义格式列表
            for item in data:
                self.samples.append({
                    "image": os.path.join(self.config.image_dir, item.get("image", "")),
                    "text": item.get("caption", item.get("text", "")),
                })

    def _parse_tsv_format(self, f):
        """解析TSV格式（LAION风格）"""
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                self.samples.append({
                    "image": os.path.join(self.config.image_dir, parts[0]),
                    "text": parts[1],
                })

    def _scan_directory(self):
        """扫描目录自动构建数据集"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_dir = Path(self.config.image_dir)

        if image_dir.exists():
            for ext in image_extensions:
                for img_path in image_dir.rglob(f"*{ext}"):
                    # 尝试查找对应的文本文件
                    txt_path = img_path.with_suffix('.txt')
                    caption = ""
                    if txt_path.exists():
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            caption = f.read().strip()

                    self.samples.append({
                        "image": str(img_path),
                        "text": caption,
                    })

    def _build_transform(self) -> Callable:
        """构建图像变换"""
        transform_list = [
            transforms.Resize(self.config.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.image_mean, std=self.config.image_std),
        ]

        if self.config.enable_augmentation:
            aug_list = []
            if self.config.random_crop:
                aug_list.append(transforms.RandomResizedCrop(
                    self.config.image_size,
                    scale=(0.8, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC
                ))
            if self.config.random_flip:
                aug_list.append(transforms.RandomHorizontalFlip(p=0.5))
            if self.config.color_jitter:
                aug_list.append(transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ))

            # 训练时随机应用增强
            transform_list = [
                transforms.Resize(int(self.config.image_size * 1.14), interpolation=transforms.InterpolationMode.BICUBIC),
            ] + aug_list + [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config.image_mean, std=self.config.image_std),
            ]

        return transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # 加载图像
        image_path = sample["image"]
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            warnings.warn(f"Failed to load image {image_path}: {e}")
            # 返回空白图像
            image = torch.zeros(3, self.config.image_size, self.config.image_size)

        # 处理文本
        text = sample.get("text", "")
        text_tokens = self._tokenize_text(text)

        return {
            "image": image,
            "text": text,
            "text_tokens": text_tokens,
            "image_path": image_path,
        }

    def _tokenize_text(self, text: str) -> torch.Tensor:
        """简单的文本tokenization"""
        if self.tokenizer:
            return self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        # 简单的字符级tokenization作为fallback
        tokens = [ord(c) % 256 for c in text[:self.config.max_text_length]]
        tokens += [0] * (self.config.max_text_length - len(tokens))
        return torch.tensor(tokens[:self.config.max_text_length], dtype=torch.long)


class VideoTextPair(Dataset):
    """
    视频-文本配对数据集

    支持从视频文件提取帧，或与预提取帧目录配合使用
    """

    def __init__(
        self,
        config: MultimodalDataConfig,
        transform: Optional[Callable] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.config = config
        self.transform = transform or self._build_transform()
        self.tokenizer = tokenizer
        self.samples: List[Dict[str, Any]] = []

        self._load_data()

    def _load_data(self):
        """加载视频-文本配对数据"""
        if self.config.annotation_file:
            with open(self.config.annotation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    self.samples.append({
                        "video": os.path.join(self.config.video_dir, item.get("video", "")),
                        "text": item.get("caption", item.get("text", "")),
                        "start": item.get("start", 0),
                        "end": item.get("end", -1),
                    })
        else:
            # 扫描视频目录
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
            video_dir = Path(self.config.video_dir)

            if video_dir.exists():
                for ext in video_extensions:
                    for video_path in video_dir.rglob(f"*{ext}"):
                        txt_path = video_path.with_suffix('.txt')
                        caption = ""
                        if txt_path.exists():
                            with open(txt_path, 'r', encoding='utf-8') as f:
                                caption = f.read().strip()

                        self.samples.append({
                            "video": str(video_path),
                            "text": caption,
                            "start": 0,
                            "end": -1,
                        })

    def _build_transform(self) -> Callable:
        """构建视频帧变换"""
        return transforms.Compose([
            transforms.Resize(self.config.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.image_mean, std=self.config.image_std),
        ])

    def _sample_frames(self, video_path: str, start: int = 0, end: int = -1) -> torch.Tensor:
        """
        从视频中采样帧

        返回: [num_frames, C, H, W]
        """
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if end < 0:
                end = total_frames

            # 计算采样位置
            num_frames = min(self.config.num_frames, (end - start) // self.config.frame_sample_rate)
            if num_frames <= 0:
                num_frames = self.config.num_frames

            frame_indices = np.linspace(start, end - 1, num_frames, dtype=int)

            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    if self.transform:
                        frame = self.transform(frame)
                    frames.append(frame)

            cap.release()

            # 填充不足
            while len(frames) < self.config.num_frames:
                frames.append(torch.zeros(3, self.config.image_size, self.config.image_size))

            return torch.stack(frames[:self.config.num_frames])

        except ImportError:
            warnings.warn("OpenCV not available, returning dummy frames")
            return torch.zeros(self.config.num_frames, 3, self.config.image_size, self.config.image_size)
        except Exception as e:
            warnings.warn(f"Failed to load video {video_path}: {e}")
            return torch.zeros(self.config.num_frames, 3, self.config.image_size, self.config.image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # 加载视频帧
        video_path = sample["video"]
        frames = self._sample_frames(video_path, sample.get("start", 0), sample.get("end", -1))

        # 处理文本
        text = sample.get("text", "")
        text_tokens = self._tokenize_text(text)

        return {
            "video": frames,
            "text": text,
            "text_tokens": text_tokens,
            "video_path": video_path,
        }

    def _tokenize_text(self, text: str) -> torch.Tensor:
        """简单的文本tokenization"""
        if self.tokenizer:
            return self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        tokens = [ord(c) % 256 for c in text[:self.config.max_text_length]]
        tokens += [0] * (self.config.max_text_length - len(tokens))
        return torch.tensor(tokens[:self.config.max_text_length], dtype=torch.long)


class AudioTextPair(Dataset):
    """
    音频-文本配对数据集

    支持从音频文件提取特征（Mel频谱图）
    """

    def __init__(
        self,
        config: MultimodalDataConfig,
        transform: Optional[Callable] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.config = config
        self.transform = transform
        self.tokenizer = tokenizer
        self.samples: List[Dict[str, Any]] = []

        self._load_data()

    def _load_data(self):
        """加载音频-文本配对数据"""
        if self.config.annotation_file:
            with open(self.config.annotation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    self.samples.append({
                        "audio": os.path.join(self.config.audio_dir, item.get("audio", "")),
                        "text": item.get("caption", item.get("text", "")),
                        "transcription": item.get("transcription", ""),
                    })
        else:
            # 扫描音频目录
            audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
            audio_dir = Path(self.config.audio_dir)

            if audio_dir.exists():
                for ext in audio_extensions:
                    for audio_path in audio_dir.rglob(f"*{ext}"):
                        txt_path = audio_path.with_suffix('.txt')
                        caption = ""
                        if txt_path.exists():
                            with open(txt_path, 'r', encoding='utf-8') as f:
                                caption = f.read().strip()

                        self.samples.append({
                            "audio": str(audio_path),
                            "text": caption,
                            "transcription": caption,
                        })

    def _extract_mel_spectrogram(self, audio_path: str) -> torch.Tensor:
        """
        提取Mel频谱图

        返回: [n_mels, time_frames]
        """
        try:
            import librosa

            # 加载音频
            audio, sr = librosa.load(audio_path, sr=self.config.sample_rate, mono=True)

            # 限制长度
            max_length = int(self.config.max_audio_length * self.config.sample_rate)
            if len(audio) > max_length:
                audio = audio[:max_length]
            else:
                # 填充
                audio = np.pad(audio, (0, max_length - len(audio)), mode='constant')

            # 提取Mel频谱图
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.config.sample_rate,
                n_mels=self.config.n_mels,
                n_fft=2048,
                hop_length=512,
            )

            # 转换为对数刻度
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

            # 归一化
            log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-8)

            return torch.from_numpy(log_mel_spec).float()

        except ImportError:
            warnings.warn("Librosa not available, returning dummy spectrogram")
            return torch.zeros(self.config.n_mels, 100)
        except Exception as e:
            warnings.warn(f"Failed to process audio {audio_path}: {e}")
            return torch.zeros(self.config.n_mels, 100)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # 加载音频特征
        audio_path = sample["audio"]
        mel_spec = self._extract_mel_spectrogram(audio_path)

        # 处理文本
        text = sample.get("text", sample.get("transcription", ""))
        text_tokens = self._tokenize_text(text)

        return {
            "audio": mel_spec,
            "text": text,
            "text_tokens": text_tokens,
            "audio_path": audio_path,
        }

    def _tokenize_text(self, text: str) -> torch.Tensor:
        """简单的文本tokenization"""
        if self.tokenizer:
            return self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        tokens = [ord(c) % 256 for c in text[:self.config.max_text_length]]
        tokens += [0] * (self.config.max_text_length - len(tokens))
        return torch.tensor(tokens[:self.config.max_text_length], dtype=torch.long)


class MultimodalCollator:
    """
    多模态数据批处理整理器

    处理不同模态数据的padding和batching
    """

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """整理batch数据"""
        result = {}

        # 处理图像
        if "image" in batch[0]:
            result["images"] = torch.stack([item["image"] for item in batch])

        # 处理视频
        if "video" in batch[0]:
            result["videos"] = torch.stack([item["video"] for item in batch])

        # 处理音频
        if "audio" in batch[0]:
            # 音频长度可能不同，需要padding
            max_time = max(item["audio"].shape[-1] for item in batch)
            padded_audio = []
            for item in batch:
                audio = item["audio"]
                if audio.shape[-1] < max_time:
                    pad_width = max_time - audio.shape[-1]
                    audio = torch.nn.functional.pad(audio, (0, pad_width))
                padded_audio.append(audio)
            result["audios"] = torch.stack(padded_audio)

        # 处理文本
        if "text" in batch[0]:
            result["texts"] = [item["text"] for item in batch]

        # 处理文本tokens
        if "text_tokens" in batch[0]:
            tokens_list = [item["text_tokens"] for item in batch]
            if isinstance(tokens_list[0], torch.Tensor):
                result["text_tokens"] = torch.stack(tokens_list)
            else:
                result["text_tokens"] = tokens_list

        # 元数据
        if "image_path" in batch[0]:
            result["image_paths"] = [item.get("image_path", "") for item in batch]
        if "video_path" in batch[0]:
            result["video_paths"] = [item.get("video_path", "") for item in batch]
        if "audio_path" in batch[0]:
            result["audio_paths"] = [item.get("audio_path", "") for item in batch]

        return result


class MultimodalDataLoader:
    """
    多模态数据加载器主类

    统一接口用于创建各种模态配对的数据加载器
    """

    def __init__(self, config: Optional[MultimodalDataConfig] = None):
        self.config = config or MultimodalDataConfig()
        self.collator = MultimodalCollator()

    def create_dataloader(
        self,
        pair_type: Optional[ModalityPairType] = None,
        dataset: Optional[Dataset] = None,
        **kwargs
    ) -> DataLoader:
        """
        创建数据加载器

        Args:
            pair_type: 模态配对类型，如果为None则使用config中的类型
            dataset: 自定义数据集，如果为None则根据pair_type创建
            **kwargs: 覆盖config的参数

        Returns:
            DataLoader实例
        """
        pair_type = pair_type or self.config.pair_type

        # 更新配置
        config = self._update_config(**kwargs)

        # 创建数据集
        if dataset is None:
            dataset = self._create_dataset(pair_type, config)

        # 创建sampler（分布式训练）
        sampler = None
        if config.distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=config.world_size,
                rank=config.rank,
                shuffle=config.shuffle,
            )

        # 创建DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(config.shuffle and sampler is None),
            sampler=sampler,
            num_workers=config.num_workers,
            collate_fn=self.collator,
            drop_last=config.drop_last,
            pin_memory=config.pin_memory,
            persistent_workers=config.num_workers > 0,
        )

        return dataloader

    def _create_dataset(
        self,
        pair_type: ModalityPairType,
        config: MultimodalDataConfig
    ) -> Dataset:
        """根据类型创建数据集"""
        if pair_type == ModalityPairType.IMAGE_TEXT:
            return ImageTextPair(config)
        elif pair_type == ModalityPairType.VIDEO_TEXT:
            return VideoTextPair(config)
        elif pair_type == ModalityPairType.AUDIO_TEXT:
            return AudioTextPair(config)
        else:
            raise ValueError(f"Unsupported pair type: {pair_type}")

    def _update_config(self, **kwargs) -> MultimodalDataConfig:
        """更新配置"""
        config_dict = {
            'data_root': self.config.data_root,
            'image_dir': self.config.image_dir,
            'video_dir': self.config.video_dir,
            'audio_dir': self.config.audio_dir,
            'annotation_file': self.config.annotation_file,
            'pair_type': self.config.pair_type,
            'image_size': self.config.image_size,
            'num_frames': self.config.num_frames,
            'sample_rate': self.config.sample_rate,
            'n_mels': self.config.n_mels,
            'max_text_length': self.config.max_text_length,
            'batch_size': self.config.batch_size,
            'num_workers': self.config.num_workers,
            'shuffle': self.config.shuffle,
            'drop_last': self.config.drop_last,
            'pin_memory': self.config.pin_memory,
            'distributed': self.config.distributed,
            'world_size': self.config.world_size,
            'rank': self.config.rank,
        }
        config_dict.update(kwargs)
        return MultimodalDataConfig(**config_dict)

    def create_image_text_loader(
        self,
        data_root: Optional[str] = None,
        annotation_file: Optional[str] = None,
        **kwargs
    ) -> DataLoader:
        """创建图像-文本数据加载器"""
        return self.create_dataloader(
            ModalityPairType.IMAGE_TEXT,
            data_root=data_root,
            annotation_file=annotation_file,
            **kwargs
        )

    def create_video_text_loader(
        self,
        data_root: Optional[str] = None,
        annotation_file: Optional[str] = None,
        **kwargs
    ) -> DataLoader:
        """创建视频-文本数据加载器"""
        return self.create_dataloader(
            ModalityPairType.VIDEO_TEXT,
            data_root=data_root,
            annotation_file=annotation_file,
            **kwargs
        )

    def create_audio_text_loader(
        self,
        data_root: Optional[str] = None,
        annotation_file: Optional[str] = None,
        **kwargs
    ) -> DataLoader:
        """创建音频-文本数据加载器"""
        return self.create_dataloader(
            ModalityPairType.AUDIO_TEXT,
            data_root=data_root,
            annotation_file=annotation_file,
            **kwargs
        )


def create_dataloader(
    pair_type: str = "image_text",
    data_root: str = "./data",
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """
    便捷函数：创建多模态数据加载器

    Args:
        pair_type: 配对类型 ("image_text", "video_text", "audio_text")
        data_root: 数据根目录
        batch_size: 批次大小
        num_workers: 数据加载线程数
        **kwargs: 其他配置参数

    Returns:
        DataLoader实例
    """
    type_map = {
        "image_text": ModalityPairType.IMAGE_TEXT,
        "video_text": ModalityPairType.VIDEO_TEXT,
        "audio_text": ModalityPairType.AUDIO_TEXT,
    }

    config = MultimodalDataConfig(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        pair_type=type_map.get(pair_type, ModalityPairType.IMAGE_TEXT),
        **kwargs
    )

    loader = MultimodalDataLoader(config)
    return loader.create_dataloader()


# ==================== 使用示例 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("多模态数据加载器测试")
    print("=" * 60)

    # 测试配置
    config = MultimodalDataConfig(
        data_root="./test_data",
        batch_size=4,
        num_workers=0,
        image_size=224,
        num_frames=8,
    )

    # 创建模拟数据
    import tempfile
    import shutil

    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建模拟图像-文本数据
        img_dir = os.path.join(tmpdir, "images")
        os.makedirs(img_dir, exist_ok=True)

        for i in range(10):
            # 创建空白图像
            img = Image.new('RGB', (256, 256), color=(i*20, i*20, i*20))
            img.save(os.path.join(img_dir, f"image_{i}.jpg"))

            # 创建对应文本
            with open(os.path.join(img_dir, f"image_{i}.txt"), 'w') as f:
                f.write(f"This is a test image number {i}")

        # 测试图像-文本加载器
        print("\n[1] 测试图像-文本数据加载器")
        config.image_dir = img_dir
        loader = MultimodalDataLoader(config)
        dataloader = loader.create_image_text_loader(
            data_root=tmpdir,
            batch_size=2,
        )

        for batch_idx, batch in enumerate(dataloader):
            print(f"  Batch {batch_idx}: images shape={batch['images'].shape}, texts={len(batch['texts'])}")
            if batch_idx >= 2:
                break

        print("\n[2] 测试便捷函数")
        dataloader2 = create_dataloader(
            pair_type="image_text",
            data_root=tmpdir,
            batch_size=2,
            num_workers=0,
        )
        for batch_idx, batch in enumerate(dataloader2):
            print(f"  Batch {batch_idx}: images shape={batch['images'].shape}")
            if batch_idx >= 1:
                break

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
