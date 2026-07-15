"""
TTS (Text-to-Speech) Engine
真实语音合成引擎

支持多种后端：
- Edge TTS: 微软免费TTS，无需GPU
- Bark: Suno AI的本地TTS模型
- Coqui TTS: 开源TTS框架
- Azure TTS: 微软云服务
- ElevenLabs: 高质量商业API
"""

import asyncio
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
from typing import Any, Dict, List, Optional, Tuple, Union, AsyncGenerator, Callable

logger = logging.getLogger(__name__)


class AudioFormat(Enum):
    """音频格式"""
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"
    AAC = "aac"


class VoiceGender(Enum):
    """声音性别"""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


@dataclass
class VoiceConfig:
    """语音配置"""
    voice_id: str = "default"
    language: str = "zh-CN"
    gender: VoiceGender = VoiceGender.NEUTRAL
    rate: float = 1.0  # 语速 0.5-2.0
    pitch: float = 0.0  # 音调 -100 到 100
    volume: float = 1.0  # 音量 0-1
    style: Optional[str] = None  # 情感风格
    style_degree: float = 1.0  # 风格强度
    
    def to_edge_tts_params(self) -> Dict[str, Any]:
        """转换为Edge TTS参数"""
        params = {
            "voice": self.voice_id,
            "rate": f"{int(self.rate * 100):+d}%",
            "volume": f"{int(self.volume * 100):+d}%",
        }
        if self.pitch != 0:
            params["pitch"] = f"{int(self.pitch):+d}Hz"
        return params


@dataclass
class TTSResult:
    """TTS结果"""
    success: bool
    audio_data: Optional[bytes] = None
    audio_path: Optional[str] = None
    format: AudioFormat = AudioFormat.MP3
    duration: float = 0.0  # 秒
    sample_rate: int = 24000
    channels: int = 1
    voice_id: str = ""
    text: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "audio_path": self.audio_path,
            "format": self.format.value,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "voice_id": self.voice_id,
            "text": self.text,
            "error": self.error,
            "metadata": self.metadata,
        }


class TTSEngine(ABC):
    """TTS引擎基类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialized = False
        self._cache: Dict[str, TTSResult] = {}
        
    @abstractmethod
    async def synthesize(
        self, 
        text: str, 
        voice_config: Optional[VoiceConfig] = None,
        output_format: AudioFormat = AudioFormat.MP3
    ) -> TTSResult:
        """合成语音"""
        pass
    
    @abstractmethod
    async def list_voices(self, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出可用声音"""
        pass
    
    @abstractmethod
    async def get_voice_info(self, voice_id: str) -> Dict[str, Any]:
        """获取声音信息"""
        pass
    
    def _get_cache_key(self, text: str, voice_config: VoiceConfig, output_format: AudioFormat) -> str:
        """生成缓存键"""
        key_data = f"{text}|{voice_config.voice_id}|{voice_config.rate}|{voice_config.pitch}|{output_format.value}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def synthesize_cached(
        self,
        text: str,
        voice_config: Optional[VoiceConfig] = None,
        output_format: AudioFormat = AudioFormat.MP3
    ) -> TTSResult:
        """带缓存的语音合成"""
        voice_config = voice_config or VoiceConfig()
        cache_key = self._get_cache_key(text, voice_config, output_format)
        
        if cache_key in self._cache:
            logger.debug(f"TTS cache hit for key: {cache_key}")
            return self._cache[cache_key]
        
        result = await self.synthesize(text, voice_config, output_format)
        
        if result.success:
            self._cache[cache_key] = result
            
        return result
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()


class EdgeTTSEngine(TTSEngine):
    """
    Edge TTS引擎
    使用微软Edge的免费TTS服务，无需API密钥
    
    特点：
    - 完全免费
    - 高质量神经网络语音
    - 支持多种语言
    - 无需GPU
    """
    
    # 常用中文声音
    CHINESE_VOICES = {
        "zh-CN-XiaoxiaoNeural": {"gender": "Female", "description": "晓晓 - 温柔女声"},
        "zh-CN-YunxiNeural": {"gender": "Male", "description": "云希 - 阳光男声"},
        "zh-CN-YunjianNeural": {"gender": "Male", "description": "云健 - 新闻男声"},
        "zh-CN-XiaoyiNeural": {"gender": "Female", "description": "晓伊 - 甜美女声"},
        "zh-CN-YunyangNeural": {"gender": "Male", "description": "云扬 - 客服男声"},
        "zh-CN-XiaochenNeural": {"gender": "Female", "description": "晓辰 - 温柔女声"},
        "zh-CN-XiaohanNeural": {"gender": "Female", "description": "晓涵 - 活泼女声"},
        "zh-CN-XiaomengNeural": {"gender": "Female", "description": "晓梦 - 可爱女声"},
        "zh-CN-XiaomoNeural": {"gender": "Female", "description": "晓墨 - 成熟女声"},
        "zh-CN-XiaoruiNeural": {"gender": "Female", "description": "晓睿 - 童声女声"},
        "zh-CN-XiaoshuangNeural": {"gender": "Female", "description": "晓双 - 儿童女声"},
        "zh-CN-XiaoxuanNeural": {"gender": "Female", "description": "晓萱 - 温暖女声"},
        "zh-CN-XiaoyanNeural": {"gender": "Female", "description": "晓颜 - 新闻女声"},
        "zh-CN-XiaoyouNeural": {"gender": "Female", "description": "晓悠 - 儿童女声"},
        "zh-CN-YunfengNeural": {"gender": "Male", "description": "云枫 - 新闻男声"},
        "zh-CN-YunhaoNeural": {"gender": "Male", "description": "云皓 - 广告男声"},
        "zh-CN-YunxiaNeural": {"gender": "Male", "description": "云夏 - 儿童男声"},
        "zh-CN-YunyeNeural": {"gender": "Male", "description": "云野 - 纪录片男声"},
        "zh-CN-YunzeNeural": {"gender": "Male", "description": "云泽 - 故事男声"},
    }
    
    # 英文声音
    ENGLISH_VOICES = {
        "en-US-JennyNeural": {"gender": "Female", "description": "Jenny - Natural American female"},
        "en-US-GuyNeural": {"gender": "Male", "description": "Guy - Natural American male"},
        "en-US-AriaNeural": {"gender": "Female", "description": "Aria - Expressive American female"},
        "en-US-DavisNeural": {"gender": "Male", "description": "Davis - Casual American male"},
        "en-US-AmberNeural": {"gender": "Female", "description": "Amber - Professional American female"},
        "en-US-AnaNeural": {"gender": "Female", "description": "Ana - Young American female"},
        "en-US-ASHLEYNeural": {"gender": "Female", "description": "Ashley - Friendly American female"},
        "en-US-BrandonNeural": {"gender": "Male", "description": "Brandon - Young American male"},
        "en-US-ChristopherNeural": {"gender": "Male", "description": "Christopher - Deep American male"},
        "en-US-CoraNeural": {"gender": "Female", "description": "Cora - Calm American female"},
        "en-US-ElizabethNeural": {"gender": "Female", "description": "Elizabeth - Mature American female"},
        "en-US-EricNeural": {"gender": "Male", "description": "Eric - Professional American male"},
        "en-US-JasonNeural": {"gender": "Male", "description": "Jason - Warm American male"},
        "en-US-MichelleNeural": {"gender": "Female", "description": "Michelle - Clear American female"},
        "en-US-MonicaNeural": {"gender": "Female", "description": "Monica - Friendly American female"},
        "en-US-SaraNeural": {"gender": "Female", "description": "Sara - Sweet American female"},
        "en-US-TonyNeural": {"gender": "Male", "description": "Tony - Energetic American male"},
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._communicate = None
        self._list_voices_func = None
        
    async def _ensure_initialized(self):
        """确保edge-tts已初始化"""
        if self._initialized:
            return
            
        try:
            import edge_tts
            self._communicate = edge_tts.Communicate
            self._list_voices_func = edge_tts.list_voices
            self._initialized = True
            logger.info("Edge TTS engine initialized successfully")
        except ImportError:
            raise ImportError(
                "edge-tts is not installed. Install it with: pip install edge-tts"
            )
    
    async def synthesize(
        self, 
        text: str, 
        voice_config: Optional[VoiceConfig] = None,
        output_format: AudioFormat = AudioFormat.MP3
    ) -> TTSResult:
        """使用Edge TTS合成语音"""
        await self._ensure_initialized()
        
        voice_config = voice_config or VoiceConfig()
        
        # 默认使用中文女声
        if voice_config.voice_id == "default":
            voice_config.voice_id = "zh-CN-XiaoxiaoNeural"
        
        try:
            # 构建参数
            params = voice_config.to_edge_tts_params()
            
            # 创建Communicate对象
            communicate = self._communicate(text, **params)
            
            # 生成音频数据
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            if not audio_data:
                return TTSResult(
                    success=False,
                    error="No audio data generated",
                    text=text,
                    voice_id=voice_config.voice_id
                )
            
            # 保存到临时文件
            output_path = self._save_audio(audio_data, output_format)
            
            # 估算时长（Edge TTS返回的是MP3格式）
            duration = self._estimate_duration(len(audio_data), output_format)
            
            return TTSResult(
                success=True,
                audio_data=audio_data,
                audio_path=output_path,
                format=output_format,
                duration=duration,
                voice_id=voice_config.voice_id,
                text=text,
                metadata={
                    "engine": "edge_tts",
                    "params": params,
                }
            )
            
        except Exception as e:
            logger.error(f"Edge TTS synthesis failed: {e}")
            return TTSResult(
                success=False,
                error=str(e),
                text=text,
                voice_id=voice_config.voice_id
            )
    
    def _save_audio(self, audio_data: bytes, output_format: AudioFormat) -> str:
        """保存音频到文件"""
        output_dir = self.config.get("output_dir", tempfile.gettempdir())
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"tts_{uuid.uuid4().hex}.{output_format.value}"
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, "wb") as f:
            f.write(audio_data)
        
        return output_path
    
    def _estimate_duration(self, audio_size: int, format: AudioFormat) -> float:
        """估算音频时长"""
        # 粗略估算：MP3 @ 128kbps
        if format == AudioFormat.MP3:
            bitrate = 128000  # bits per second
            return (audio_size * 8) / bitrate
        elif format == AudioFormat.WAV:
            # WAV: 24kHz, 16bit, mono
            return audio_size / (24000 * 2)
        else:
            return audio_size / 16000  # 默认估算
    
    async def list_voices(self, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出可用声音"""
        await self._ensure_initialized()
        
        voices = []
        
        # 如果指定了语言，返回该语言的声音
        if language and language.startswith("zh"):
            for voice_id, info in self.CHINESE_VOICES.items():
                voices.append({
                    "id": voice_id,
                    "name": voice_id.replace("Neural", "").replace("zh-CN-", ""),
                    "language": "zh-CN",
                    "gender": info["gender"],
                    "description": info["description"],
                    "engine": "edge_tts",
                })
        elif language and language.startswith("en"):
            for voice_id, info in self.ENGLISH_VOICES.items():
                voices.append({
                    "id": voice_id,
                    "name": voice_id.replace("Neural", "").replace("en-US-", ""),
                    "language": "en-US",
                    "gender": info["gender"],
                    "description": info["description"],
                    "engine": "edge_tts",
                })
        else:
            # 返回所有声音
            for voice_id, info in self.CHINESE_VOICES.items():
                voices.append({
                    "id": voice_id,
                    "language": "zh-CN",
                    "gender": info["gender"],
                    "description": info["description"],
                    "engine": "edge_tts",
                })
            for voice_id, info in self.ENGLISH_VOICES.items():
                voices.append({
                    "id": voice_id,
                    "language": "en-US",
                    "gender": info["gender"],
                    "description": info["description"],
                    "engine": "edge_tts",
                })
        
        return voices
    
    async def get_voice_info(self, voice_id: str) -> Dict[str, Any]:
        """获取声音信息"""
        all_voices = {**self.CHINESE_VOICES, **self.ENGLISH_VOICES}
        
        if voice_id in all_voices:
            info = all_voices[voice_id]
            return {
                "id": voice_id,
                "name": voice_id.replace("Neural", ""),
                "gender": info["gender"],
                "description": info["description"],
                "engine": "edge_tts",
                "available": True,
            }
        
        return {
            "id": voice_id,
            "available": False,
            "error": "Voice not found",
        }


class BarkTTSEngine(TTSEngine):
    """
    Bark TTS引擎
    Suno AI开发的高质量本地TTS模型
    
    特点：
    - 高度自然的语音
    - 支持音乐和背景噪音
    - 可以克隆声音
    - 需要GPU
    """
    
    BARK_VOICES = {
        "v2/zh_speaker_0": {"language": "zh", "gender": "Female", "description": "中文女声"},
        "v2/zh_speaker_1": {"language": "zh", "gender": "Male", "description": "中文男声"},
        "v2/en_speaker_0": {"language": "en", "gender": "Female", "description": "English female"},
        "v2/en_speaker_1": {"language": "en", "gender": "Male", "description": "English male"},
        "v2/en_speaker_2": {"language": "en", "gender": "Female", "description": "English female 2"},
        "v2/en_speaker_3": {"language": "en", "gender": "Male", "description": "English male 2"},
        "v2/en_speaker_4": {"language": "en", "gender": "Female", "description": "English female 3"},
        "v2/en_speaker_5": {"language": "en", "gender": "Male", "description": "English male 3"},
        "v2/en_speaker_6": {"language": "en", "gender": "Female", "description": "English female 4"},
        "v2/en_speaker_7": {"language": "en", "gender": "Male", "description": "English male 4"},
        "v2/en_speaker_8": {"language": "en", "gender": "Female", "description": "English female 5"},
        "v2/en_speaker_9": {"language": "en", "gender": "Male", "description": "English male 5"},
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._bark = None
        self._sample_rate = 24000
        
    async def _ensure_initialized(self):
        """确保Bark已初始化"""
        if self._initialized:
            return
            
        try:
            from bark import SAMPLE_RATE, generate_audio, preload_models
            self._bark = {
                "generate_audio": generate_audio,
                "preload_models": preload_models,
                "SAMPLE_RATE": SAMPLE_RATE,
            }
            self._sample_rate = SAMPLE_RATE
            
            # 预加载模型
            if self.config.get("preload", True):
                preload_models()
            
            self._initialized = True
            logger.info("Bark TTS engine initialized successfully")
        except ImportError:
            raise ImportError(
                "bark is not installed. Install it with: pip install git+https://github.com/suno-ai/bark.git"
            )
    
    async def synthesize(
        self, 
        text: str, 
        voice_config: Optional[VoiceConfig] = None,
        output_format: AudioFormat = AudioFormat.WAV
    ) -> TTSResult:
        """使用Bark合成语音"""
        await self._ensure_initialized()
        
        voice_config = voice_config or VoiceConfig()
        
        if voice_config.voice_id == "default":
            voice_config.voice_id = "v2/zh_speaker_0"
        
        try:
            # Bark生成音频
            audio_array = self._bark["generate_audio"](
                text,
                history_prompt=voice_config.voice_id,
            )
            
            # 转换为WAV格式
            audio_data = self._array_to_wav(audio_array)
            
            # 保存文件
            output_path = self._save_audio(audio_data, output_format)
            
            duration = len(audio_array) / self._sample_rate
            
            return TTSResult(
                success=True,
                audio_data=audio_data,
                audio_path=output_path,
                format=output_format,
                duration=duration,
                sample_rate=self._sample_rate,
                voice_id=voice_config.voice_id,
                text=text,
                metadata={
                    "engine": "bark",
                    "sample_rate": self._sample_rate,
                }
            )
            
        except Exception as e:
            logger.error(f"Bark TTS synthesis failed: {e}")
            return TTSResult(
                success=False,
                error=str(e),
                text=text,
                voice_id=voice_config.voice_id
            )
    
    def _array_to_wav(self, audio_array) -> bytes:
        """将numpy数组转换为WAV格式"""
        import numpy as np
        from scipy.io import wavfile
        
        # 归一化
        audio_array = np.clip(audio_array, -1, 1)
        audio_array = (audio_array * 32767).astype(np.int16)
        
        # 写入内存
        buffer = io.BytesIO()
        wavfile.write(buffer, self._sample_rate, audio_array)
        return buffer.getvalue()
    
    def _save_audio(self, audio_data: bytes, output_format: AudioFormat) -> str:
        """保存音频到文件"""
        output_dir = self.config.get("output_dir", tempfile.gettempdir())
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"tts_bark_{uuid.uuid4().hex}.{output_format.value}"
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, "wb") as f:
            f.write(audio_data)
        
        return output_path
    
    async def list_voices(self, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出可用声音"""
        voices = []
        for voice_id, info in self.BARK_VOICES.items():
            if language is None or info["language"] == language.split("-")[0]:
                voices.append({
                    "id": voice_id,
                    "language": info["language"],
                    "gender": info["gender"],
                    "description": info["description"],
                    "engine": "bark",
                })
        return voices
    
    async def get_voice_info(self, voice_id: str) -> Dict[str, Any]:
        """获取声音信息"""
        if voice_id in self.BARK_VOICES:
            info = self.BARK_VOICES[voice_id]
            return {
                "id": voice_id,
                "language": info["language"],
                "gender": info["gender"],
                "description": info["description"],
                "engine": "bark",
                "available": True,
            }
        return {"id": voice_id, "available": False}


class CoquiTTSEngine(TTSEngine):
    """
    Coqui TTS引擎
    开源TTS框架，支持多种模型
    
    特点：
    - 支持多种预训练模型
    - 支持声音克隆
    - 支持多语言
    - 可本地运行
    """
    
    COQUI_MODELS = {
        "tts_models/zh-CN/baker/tacotron2-DDC": {
            "language": "zh-CN",
            "description": "中文Baker数据集模型",
        },
        "tts_models/en/ljspeech/vits": {
            "language": "en",
            "description": "English VITS model",
        },
        "tts_models/en/vctk/vits": {
            "language": "en",
            "description": "English multi-speaker VITS",
        },
        "tts_models/multilingual/multi-dataset/your_tts": {
            "language": "multi",
            "description": "Multilingual YourTTS",
        },
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._tts = None
        self._model_name = config.get("model", "tts_models/multilingual/multi-dataset/your_tts")
        
    async def _ensure_initialized(self):
        """确保Coqui TTS已初始化"""
        if self._initialized:
            return
            
        try:
            from TTS.api import TTS
            self._tts = TTS(model_name=self._model_name, progress_bar=False, gpu=False)
            self._initialized = True
            logger.info(f"Coqui TTS engine initialized with model: {self._model_name}")
        except ImportError:
            raise ImportError(
                "TTS is not installed. Install it with: pip install TTS"
            )
    
    async def synthesize(
        self, 
        text: str, 
        voice_config: Optional[VoiceConfig] = None,
        output_format: AudioFormat = AudioFormat.WAV
    ) -> TTSResult:
        """使用Coqui TTS合成语音"""
        await self._ensure_initialized()
        
        voice_config = voice_config or VoiceConfig()
        
        try:
            output_dir = self.config.get("output_dir", tempfile.gettempdir())
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(output_dir, f"tts_coqui_{uuid.uuid4().hex}.wav")
            
            # 生成语音
            self._tts.tts_to_file(text=text, file_path=output_path)
            
            # 读取音频数据
            with open(output_path, "rb") as f:
                audio_data = f.read()
            
            return TTSResult(
                success=True,
                audio_data=audio_data,
                audio_path=output_path,
                format=output_format,
                voice_id=self._model_name,
                text=text,
                metadata={
                    "engine": "coqui",
                    "model": self._model_name,
                }
            )
            
        except Exception as e:
            logger.error(f"Coqui TTS synthesis failed: {e}")
            return TTSResult(
                success=False,
                error=str(e),
                text=text,
                voice_id=self._model_name
            )
    
    async def list_voices(self, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出可用声音"""
        voices = []
        for model_id, info in self.COQUI_MODELS.items():
            if language is None or info["language"] == language or info["language"] == "multi":
                voices.append({
                    "id": model_id,
                    "language": info["language"],
                    "description": info["description"],
                    "engine": "coqui",
                })
        return voices
    
    async def get_voice_info(self, voice_id: str) -> Dict[str, Any]:
        """获取声音信息"""
        if voice_id in self.COQUI_MODELS:
            info = self.COQUI_MODELS[voice_id]
            return {
                "id": voice_id,
                "language": info["language"],
                "description": info["description"],
                "engine": "coqui",
                "available": True,
            }
        return {"id": voice_id, "available": False}


class AzureTTSEngine(TTSEngine):
    """
    Azure Cognitive Services TTS引擎
    
    特点：
    - 高质量神经网络语音
    - 支持SSML
    - 支持情感风格
    - 需要Azure订阅
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._speech_config = None
        self._speech_synthesizer = None
        
    async def _ensure_initialized(self):
        """确保Azure TTS已初始化"""
        if self._initialized:
            return
            
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            speech_key = self.config.get("azure_speech_key") or os.environ.get("AZURE_SPEECH_KEY")
            service_region = self.config.get("azure_speech_region") or os.environ.get("AZURE_SPEECH_REGION")
            
            if not speech_key or not service_region:
                raise ValueError("Azure Speech key and region are required")
            
            self._speech_config = speechsdk.SpeechConfig(
                subscription=speech_key,
                region=service_region
            )
            self._speech_config.speech_synthesis_language = "zh-CN"
            
            self._initialized = True
            logger.info("Azure TTS engine initialized successfully")
            
        except ImportError:
            raise ImportError(
                "azure-cognitiveservices-speech is not installed. Install it with: pip install azure-cognitiveservices-speech"
            )
    
    async def synthesize(
        self, 
        text: str, 
        voice_config: Optional[VoiceConfig] = None,
        output_format: AudioFormat = AudioFormat.MP3
    ) -> TTSResult:
        """使用Azure TTS合成语音"""
        await self._ensure_initialized()
        
        voice_config = voice_config or VoiceConfig()
        
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            # 设置声音
            if voice_config.voice_id != "default":
                self._speech_config.speech_synthesis_voice_name = voice_config.voice_id
            else:
                self._speech_config.speech_synthesis_voice_name = "zh-CN-XiaoxiaoNeural"
            
            # 设置输出格式
            if output_format == AudioFormat.MP3:
                self._speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
                )
            else:
                self._speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
                )
            
            # 创建合成器
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self._speech_config,
                audio_config=None
            )
            
            # 合成语音
            result = synthesizer.speak_text_async(text).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                audio_data = result.audio_data
                
                # 保存文件
                output_path = self._save_audio(audio_data, output_format)
                
                return TTSResult(
                    success=True,
                    audio_data=audio_data,
                    audio_path=output_path,
                    format=output_format,
                    voice_id=voice_config.voice_id,
                    text=text,
                    metadata={
                        "engine": "azure",
                    }
                )
            else:
                return TTSResult(
                    success=False,
                    error=f"Azure TTS failed: {result.reason}",
                    text=text,
                    voice_id=voice_config.voice_id
                )
                
        except Exception as e:
            logger.error(f"Azure TTS synthesis failed: {e}")
            return TTSResult(
                success=False,
                error=str(e),
                text=text,
                voice_id=voice_config.voice_id if voice_config else "default"
            )
    
    def _save_audio(self, audio_data: bytes, output_format: AudioFormat) -> str:
        """保存音频到文件"""
        output_dir = self.config.get("output_dir", tempfile.gettempdir())
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"tts_azure_{uuid.uuid4().hex}.{output_format.value}"
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, "wb") as f:
            f.write(audio_data)
        
        return output_path
    
    async def list_voices(self, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出可用声音"""
        # 返回Azure支持的声音列表
        voices = [
            {"id": "zh-CN-XiaoxiaoNeural", "language": "zh-CN", "gender": "Female", "description": "晓晓"},
            {"id": "zh-CN-YunxiNeural", "language": "zh-CN", "gender": "Male", "description": "云希"},
            {"id": "en-US-JennyNeural", "language": "en-US", "gender": "Female", "description": "Jenny"},
            {"id": "en-US-GuyNeural", "language": "en-US", "gender": "Male", "description": "Guy"},
        ]
        return voices
    
    async def get_voice_info(self, voice_id: str) -> Dict[str, Any]:
        """获取声音信息"""
        return {"id": voice_id, "engine": "azure"}


class ElevenLabsTTSEngine(TTSEngine):
    """
    ElevenLabs TTS引擎
    高质量商业TTS API
    
    特点：
    - 极高质量的语音
    - 支持声音克隆
    - 支持多语言
    - 需要API密钥
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._api_key = None
        self._base_url = "https://api.elevenlabs.io/v1"
        
    async def _ensure_initialized(self):
        """确保ElevenLabs已初始化"""
        if self._initialized:
            return
            
        self._api_key = self.config.get("elevenlabs_api_key") or os.environ.get("ELEVENLABS_API_KEY")
        
        if not self._api_key:
            raise ValueError("ElevenLabs API key is required")
        
        self._initialized = True
        logger.info("ElevenLabs TTS engine initialized successfully")
    
    async def synthesize(
        self, 
        text: str, 
        voice_config: Optional[VoiceConfig] = None,
        output_format: AudioFormat = AudioFormat.MP3
    ) -> TTSResult:
        """使用ElevenLabs合成语音"""
        await self._ensure_initialized()
        
        voice_config = voice_config or VoiceConfig()
        
        try:
            import httpx
            
            voice_id = voice_config.voice_id if voice_config.voice_id != "default" else "21m00Tcm4TlvDq8ikWAM"
            
            url = f"{self._base_url}/text-to-speech/{voice_id}"
            
            headers = {
                "xi-api-key": self._api_key,
                "Content-Type": "application/json",
            }
            
            data = {
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                audio_data = response.content
                
                output_path = self._save_audio(audio_data, output_format)
                
                return TTSResult(
                    success=True,
                    audio_data=audio_data,
                    audio_path=output_path,
                    format=output_format,
                    voice_id=voice_id,
                    text=text,
                    metadata={
                        "engine": "elevenlabs",
                    }
                )
            else:
                return TTSResult(
                    success=False,
                    error=f"ElevenLabs API error: {response.status_code}",
                    text=text,
                    voice_id=voice_id
                )
                
        except Exception as e:
            logger.error(f"ElevenLabs TTS synthesis failed: {e}")
            return TTSResult(
                success=False,
                error=str(e),
                text=text,
                voice_id=voice_config.voice_id if voice_config else "default"
            )
    
    def _save_audio(self, audio_data: bytes, output_format: AudioFormat) -> str:
        """保存音频到文件"""
        output_dir = self.config.get("output_dir", tempfile.gettempdir())
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"tts_elevenlabs_{uuid.uuid4().hex}.{output_format.value}"
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, "wb") as f:
            f.write(audio_data)
        
        return output_path
    
    async def list_voices(self, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出可用声音"""
        await self._ensure_initialized()
        
        try:
            import httpx
            
            url = f"{self._base_url}/voices"
            headers = {"xi-api-key": self._api_key}
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                voices = []
                for voice in data.get("voices", []):
                    voices.append({
                        "id": voice["voice_id"],
                        "name": voice["name"],
                        "engine": "elevenlabs",
                    })
                return voices
        except Exception as e:
            logger.error(f"Failed to list ElevenLabs voices: {e}")
        
        return []
    
    async def get_voice_info(self, voice_id: str) -> Dict[str, Any]:
        """获取声音信息"""
        return {"id": voice_id, "engine": "elevenlabs"}


# 工厂函数
def create_tts_engine(
    engine_type: str = "edge",
    config: Optional[Dict[str, Any]] = None
) -> TTSEngine:
    """
    创建TTS引擎
    
    Args:
        engine_type: 引擎类型 (edge, bark, coqui, azure, elevenlabs)
        config: 配置参数
    
    Returns:
        TTSEngine实例
    """
    engines = {
        "edge": EdgeTTSEngine,
        "bark": BarkTTSEngine,
        "coqui": CoquiTTSEngine,
        "azure": AzureTTSEngine,
        "elevenlabs": ElevenLabsTTSEngine,
    }
    
    if engine_type not in engines:
        raise ValueError(f"Unknown TTS engine type: {engine_type}")
    
    return engines[engine_type](config)
