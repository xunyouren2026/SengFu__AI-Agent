"""
Audio Generation Engine
真实音频/音乐生成引擎

支持多种后端：
- MusicGen: Meta的音乐生成模型
- AudioLDM: 音频生成模型
- Bark: Suno AI的音频生成
- Riffusion: 音乐生成
"""

import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

logger = logging.getLogger(__name__)


class AudioFormat(Enum):
    """音频格式"""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"


@dataclass
class AudioConfig:
    """音频生成配置"""
    prompt: str = ""
    negative_prompt: str = ""
    duration: float = 10.0  # 秒
    sample_rate: int = 44100
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    num_outputs: int = 1
    
    # MusicGen参数
    model: str = "musicgen-small"  # musicgen-small, musicgen-medium, musicgen-large
    melody: Optional[str] = None  # 旋律参考
    
    # AudioLDM参数
    audio_length_in_s: float = 10.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
        }


@dataclass
class AudioResult:
    """音频生成结果"""
    success: bool
    audio_data: Optional[bytes] = None
    audio_path: Optional[str] = None
    format: AudioFormat = AudioFormat.MP3
    duration: float = 0.0
    sample_rate: int = 44100
    seed: Optional[int] = None
    prompt: str = ""
    model: str = ""
    inference_time: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "audio_path": self.audio_path,
            "format": self.format.value,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "seed": self.seed,
            "prompt": self.prompt,
            "model": self.model,
            "inference_time": self.inference_time,
            "error": self.error,
            "metadata": self.metadata,
        }


class AudioGenerationEngine(ABC):
    """音频生成引擎基类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialized = False
        
    @abstractmethod
    async def generate(self, config: AudioConfig) -> AudioResult:
        """生成音频"""
        pass
    
    @abstractmethod
    async def list_models(self) -> List[Dict[str, Any]]:
        """列出可用模型"""
        pass
    
    def _save_audio(self, audio_data: bytes, format: AudioFormat = AudioFormat.MP3) -> str:
        """保存音频到文件"""
        output_dir = self.config.get("output_dir", tempfile.gettempdir())
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"audio_{uuid.uuid4().hex}.{format.value}"
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, "wb") as f:
            f.write(audio_data)
        
        return output_path


class MusicGenEngine(AudioGenerationEngine):
    """
    MusicGen引擎
    Meta的音乐生成模型
    
    特点：
    - 高质量音乐生成
    - 支持文本描述
    - 支持旋律条件
    - 需要GPU
    """
    
    MODEL_IDS = {
        "musicgen-small": "facebook/musicgen-small",
        "musicgen-medium": "facebook/musicgen-medium",
        "musicgen-large": "facebook/musicgen-large",
        "musicgen-melody": "facebook/musicgen-melody",
        "musicgen-melody-large": "facebook/musicgen-melody-large",
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._pipeline = None
        self._device = None
        
    async def _ensure_initialized(self):
        """确保MusicGen已初始化"""
        if self._initialized:
            return
            
        try:
            import torch
            from audiocraft.models import MusicGen
            
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            
            model_id = self.config.get("model", "musicgen-small")
            model_path = self.MODEL_IDS.get(model_id, model_id)
            
            logger.info(f"Loading MusicGen model: {model_path}")
            
            self._pipeline = MusicGen.get_pretrained(model_path)
            self._pipeline.to(self._device)
            
            self._initialized = True
            logger.info("MusicGen engine initialized successfully")
            
        except ImportError as e:
            raise ImportError(
                f"audiocraft not installed: {e}. Install with: pip install audiocraft"
            )
    
    async def generate(self, config: AudioConfig) -> AudioResult:
        """使用MusicGen生成音乐"""
        await self._ensure_initialized()
        
        try:
            import torch
            import numpy as np
            
            start_time = time.time()
            
            # 设置随机种子
            if config.seed is not None:
                torch.manual_seed(config.seed)
            
            # 设置生成参数
            self._pipeline.set_generation_params(
                duration=config.duration,
                use_sampling=True,
                top_k=250,
                top_p=0.0,
                temperature=1.0,
                cfg_coef=config.guidance_scale,
            )
            
            # 生成音乐
            if config.melody:
                # 带旋律条件生成
                from audiocraft.data.audio import audio_read
                melody, sr = audio_read(config.melody)
                output = self._pipeline.generate_with_chroma(
                    descriptions=[config.prompt],
                    melody_wavs=melody[None],
                    melody_sample_rate=sr,
                    progress=True,
                )
            else:
                # 文本生成
                output = self._pipeline.generate(
                    descriptions=[config.prompt] * config.num_outputs,
                    progress=True,
                )
            
            inference_time = time.time() - start_time
            
            # 处理输出
            audio_results = []
            audio_paths = []
            
            for i, audio in enumerate(output):
                # 转换为numpy
                audio_np = audio.cpu().numpy()
                
                # 保存为WAV
                output_path = self._save_audio_as_wav(audio_np, config.sample_rate)
                audio_paths.append(output_path)
                
                with open(output_path, "rb") as f:
                    audio_results.append(f.read())
            
            return AudioResult(
                success=True,
                audio_data=audio_results[0] if audio_results else None,
                audio_path=audio_paths[0] if audio_paths else None,
                format=AudioFormat.WAV,
                duration=config.duration,
                sample_rate=config.sample_rate,
                seed=config.seed,
                prompt=config.prompt,
                model=config.model,
                inference_time=inference_time,
                metadata={"engine": "musicgen"}
            )
            
        except Exception as e:
            logger.error(f"MusicGen generation failed: {e}")
            return AudioResult(success=False, error=str(e), prompt=config.prompt)
    
    def _save_audio_as_wav(self, audio: 'np.ndarray', sample_rate: int) -> str:
        """保存为WAV格式"""
        import numpy as np
        from scipy.io import wavfile
        
        output_dir = self.config.get("output_dir", tempfile.gettempdir())
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"music_{uuid.uuid4().hex}.wav")
        
        # 归一化
        audio = np.clip(audio, -1, 1)
        audio_int = (audio * 32767).astype(np.int16)
        
        # 如果是立体声，取第一个通道或合并
        if len(audio_int.shape) > 1:
            audio_int = audio_int.T
        
        wavfile.write(output_path, sample_rate, audio_int)
        
        return output_path
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """列出可用模型"""
        return [
            {"id": "musicgen-small", "name": "MusicGen Small", "type": "local"},
            {"id": "musicgen-medium", "name": "MusicGen Medium", "type": "local"},
            {"id": "musicgen-large", "name": "MusicGen Large", "type": "local"},
            {"id": "musicgen-melody", "name": "MusicGen Melody", "type": "local"},
        ]


class AudioLDMEngine(AudioGenerationEngine):
    """
    AudioLDM引擎
    基于文本的音频生成模型
    
    特点：
    - 支持多种音频类型
    - 文本转音频
    - 高质量生成
    - 需要GPU
    """
    
    MODEL_IDS = {
        "audioldm": "cvssp/audioldm",
        "audioldm-l-full": "cvssp/audioldm-l-full",
        "audioldm-m-full": "cvssp/audioldm-m-full",
        "audioldm-s-full": "cvssp/audioldm-s-full",
        "audioldm2": "cvssp/audioldm2",
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._pipeline = None
        self._device = None
        
    async def _ensure_initialized(self):
        """确保AudioLDM已初始化"""
        if self._initialized:
            return
            
        try:
            import torch
            from diffusers import AudioLDMPipeline
            
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            
            model_id = self.config.get("model", "audioldm")
            model_path = self.MODEL_IDS.get(model_id, model_id)
            
            logger.info(f"Loading AudioLDM model: {model_path}")
            
            self._pipeline = AudioLDMPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            )
            
            if self._device == "cuda":
                self._pipeline = self._pipeline.to(self._device)
            
            self._initialized = True
            logger.info("AudioLDM engine initialized successfully")
            
        except ImportError as e:
            raise ImportError(
                f"diffusers not installed: {e}. Install with: pip install diffusers transformers accelerate torch"
            )
    
    async def generate(self, config: AudioConfig) -> AudioResult:
        """使用AudioLDM生成音频"""
        await self._ensure_initialized()
        
        try:
            import torch
            
            start_time = time.time()
            
            # 设置随机种子
            if config.seed is not None:
                generator = torch.Generator(device=self._device).manual_seed(config.seed)
            else:
                generator = None
            
            # 生成音频
            output = self._pipeline(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
                audio_length_in_s=config.audio_length_in_s,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                generator=generator,
            )
            
            inference_time = time.time() - start_time
            
            # 处理输出
            audio = output.audios[0]
            
            # 保存为WAV
            output_path = self._save_audio_as_wav(audio, 16000)
            
            with open(output_path, "rb") as f:
                audio_data = f.read()
            
            return AudioResult(
                success=True,
                audio_data=audio_data,
                audio_path=output_path,
                format=AudioFormat.WAV,
                duration=config.audio_length_in_s,
                sample_rate=16000,
                seed=config.seed,
                prompt=config.prompt,
                model=config.model,
                inference_time=inference_time,
                metadata={"engine": "audioldm"}
            )
            
        except Exception as e:
            logger.error(f"AudioLDM generation failed: {e}")
            return AudioResult(success=False, error=str(e), prompt=config.prompt)
    
    def _save_audio_as_wav(self, audio: 'np.ndarray', sample_rate: int) -> str:
        """保存为WAV格式"""
        import numpy as np
        from scipy.io import wavfile
        
        output_dir = self.config.get("output_dir", tempfile.gettempdir())
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"audio_ldm_{uuid.uuid4().hex}.wav")
        
        # 归一化
        audio = np.clip(audio, -1, 1)
        audio_int = (audio * 32767).astype(np.int16)
        
        wavfile.write(output_path, sample_rate, audio_int)
        
        return output_path
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """列出可用模型"""
        return [
            {"id": "audioldm", "name": "AudioLDM", "type": "local"},
            {"id": "audioldm-l-full", "name": "AudioLDM L Full", "type": "local"},
            {"id": "audioldm2", "name": "AudioLDM 2", "type": "local"},
        ]


class RiffusionEngine(AudioGenerationEngine):
    """
    Riffusion引擎
    基于Stable Diffusion的音乐生成
    
    特点：
    - 基于图像生成音乐
    - 支持风格迁移
    - 创意性强
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._pipeline = None
        self._device = None
        
    async def _ensure_initialized(self):
        """确保Riffusion已初始化"""
        if self._initialized:
            return
            
        try:
            import torch
            from diffusers import DiffusionPipeline
            
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            
            model_id = self.config.get("model", "riffusion/riffusion-model-v1")
            
            logger.info(f"Loading Riffusion model: {model_id}")
            
            self._pipeline = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            )
            
            if self._device == "cuda":
                self._pipeline = self._pipeline.to(self._device)
            
            self._initialized = True
            logger.info("Riffusion engine initialized successfully")
            
        except ImportError as e:
            raise ImportError(
                f"Required packages not installed: {e}"
            )
    
    async def generate(self, config: AudioConfig) -> AudioResult:
        """使用Riffusion生成音乐"""
        await self._ensure_initialized()
        
        try:
            import torch
            
            start_time = time.time()
            
            if config.seed is not None:
                generator = torch.Generator(device=self._device).manual_seed(config.seed)
            else:
                generator = None
            
            # 生成频谱图
            output = self._pipeline(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                generator=generator,
            )
            
            inference_time = time.time() - start_time
            
            # 频谱图转音频
            # TODO: 实现频谱图到音频的转换
            
            return AudioResult(
                success=True,
                prompt=config.prompt,
                model="riffusion",
                inference_time=inference_time,
                metadata={"engine": "riffusion", "note": "Spectrogram generated, audio conversion pending"}
            )
            
        except Exception as e:
            logger.error(f"Riffusion generation failed: {e}")
            return AudioResult(success=False, error=str(e), prompt=config.prompt)
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """列出可用模型"""
        return [
            {"id": "riffusion/riffusion-model-v1", "name": "Riffusion v1", "type": "local"},
        ]


# 工厂函数
def create_audio_engine(
    engine_type: str = "musicgen",
    config: Optional[Dict[str, Any]] = None
) -> AudioGenerationEngine:
    """
    创建音频生成引擎
    
    Args:
        engine_type: 引擎类型 (musicgen, audioldm, riffusion)
        config: 配置参数
    
    Returns:
        AudioGenerationEngine实例
    """
    engines = {
        "musicgen": MusicGenEngine,
        "audioldm": AudioLDMEngine,
        "riffusion": RiffusionEngine,
    }
    
    if engine_type not in engines:
        raise ValueError(f"Unknown audio engine type: {engine_type}")
    
    return engines[engine_type](config)
