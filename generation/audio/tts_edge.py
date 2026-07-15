"""
Edge TTS Integration Simulation - Edge TTS集成模拟

本模块模拟了Edge TTS的接口，包含语音选择、SSML处理、韵律控制、
批量合成、音频格式选项和速率限制功能。仅使用标准库，
不依赖外部库。
"""

import math
import random
import struct
import time
import threading
import hashlib
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


# ============================================================================
# 辅助函数
# ============================================================================

def _generate_id() -> str:
    raw = f"{time.time()}-{random.random()}-{threading.get_ident()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t, 0.0, 1.0)


def _gaussian_noise(mean: float, std: float) -> float:
    u1 = random.random()
    u2 = random.random()
    while u1 == 0:
        u1 = random.random()
    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mean + std * z0


# ============================================================================
# 数据结构
# ============================================================================

class AudioFormat(Enum):
    """音频格式"""
    PCM_16BIT = "pcm_16bit"
    PCM_8BIT = "pcm_8bit"
    MULAW = "mulaw"
    ALAW = "alaw"
    RAW_FLOAT = "raw_float"


class AudioContainer(Enum):
    """音频容器"""
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    RAW = "raw"


@dataclass
class VoiceInfo:
    """语音信息"""
    voice_id: str = ""
    name: str = ""
    locale: str = "en-US"
    gender: str = "Female"
    short_name: str = ""
    category: str = "Neural"


@dataclass
class SynthesisRequest:
    """合成请求"""
    request_id: str = ""
    text: str = ""
    voice_id: str = ""
    rate: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    ssml: str = ""
    format: AudioFormat = AudioFormat.PCM_16BIT
    container: AudioContainer = AudioContainer.WAV
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesisResult:
    """合成结果"""
    request_id: str = ""
    audio_data: bytes = b""
    samples: List[float] = field(default_factory=list)
    sample_rate: int = 24000
    duration: float = 0.0
    format: AudioFormat = AudioFormat.PCM_16BIT
    container: AudioContainer = AudioContainer.WAV
    success: bool = True
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# VoiceSelector - 语音选择器
# ============================================================================

class VoiceSelector:
    """
    语音选择器：管理和选择可用的TTS语音。

    模拟Edge TTS的语音列表和选择逻辑。
    """

    def __init__(self):
        self._voices: Dict[str, VoiceInfo] = {}
        self._locale_index: Dict[str, List[str]] = defaultdict(list)
        self._gender_index: Dict[str, List[str]] = defaultdict(list)
        self._init_builtin_voices()

    def _init_builtin_voices(self) -> None:
        """初始化内置语音列表"""
        builtin = [
            ("en-US-JennyNeural", "Jenny", "en-US", "Female", "Neural"),
            ("en-US-GuyNeural", "Guy", "en-US", "Male", "Neural"),
            ("en-US-AriaNeural", "Aria", "en-US", "Female", "Neural"),
            ("en-US-DavisNeural", "Davis", "en-US", "Male", "Neural"),
            ("en-GB-SoniaNeural", "Sonia", "en-GB", "Female", "Neural"),
            ("en-GB-RyanNeural", "Ryan", "en-GB", "Male", "Neural"),
            ("zh-CN-XiaoxiaoNeural", "Xiaoxiao", "zh-CN", "Female", "Neural"),
            ("zh-CN-YunxiNeural", "Yunxi", "zh-CN", "Male", "Neural"),
            ("zh-CN-XiaoyiNeural", "Xiaoyi", "zh-CN", "Female", "Neural"),
            ("zh-CN-YunjianNeural", "Yunjian", "zh-CN", "Male", "Neural"),
            ("ja-JP-NanamiNeural", "Nanami", "ja-JP", "Female", "Neural"),
            ("ja-JP-KeitaNeural", "Keita", "ja-JP", "Male", "Neural"),
            ("ko-KR-SunHiNeural", "SunHi", "ko-KR", "Female", "Neural"),
            ("ko-KR-InJoonNeural", "InJoon", "ko-KR", "Male", "Neural"),
            ("es-ES-ElviraNeural", "Elvira", "es-ES", "Female", "Neural"),
            ("es-ES-AlvaroNeural", "Alvaro", "es-ES", "Male", "Neural"),
            ("fr-FR-DeniseNeural", "Denise", "fr-FR", "Female", "Neural"),
            ("fr-FR-HenriNeural", "Henri", "fr-FR", "Male", "Neural"),
            ("de-DE-KatjaNeural", "Katja", "de-DE", "Female", "Neural"),
            ("de-DE-ConradNeural", "Conrad", "de-DE", "Male", "Neural"),
        ]

        for voice_id, name, locale, gender, category in builtin:
            info = VoiceInfo(
                voice_id=voice_id,
                name=name,
                locale=locale,
                gender=gender,
                short_name=name,
                category=category,
            )
            self._voices[voice_id] = info
            self._locale_index[locale].append(voice_id)
            self._gender_index[gender].append(voice_id)

    def list_voices(
        self,
        locale: Optional[str] = None,
        gender: Optional[str] = None,
    ) -> List[VoiceInfo]:
        """列出可用语音"""
        candidates = list(self._voices.values())

        if locale:
            candidates = [v for v in candidates if v.locale == locale]
        if gender:
            candidates = [v for v in candidates if v.gender == gender]

        return candidates

    def get_voice(self, voice_id: str) -> Optional[VoiceInfo]:
        """获取语音信息"""
        return self._voices.get(voice_id)

    def select_best_voice(
        self,
        locale: str = "en-US",
        gender: str = "Female",
        preferred: Optional[str] = None,
    ) -> VoiceInfo:
        """选择最佳语音"""
        if preferred and preferred in self._voices:
            return self._voices[preferred]

        locale_voices = self._locale_index.get(locale, [])
        gender_matches = [
            v for v in locale_voices
            if self._voices[v].gender == gender
        ]

        if gender_matches:
            return self._voices[gender_matches[0]]
        elif locale_voices:
            return self._voices[locale_voices[0]]

        return list(self._voices.values())[0]

    def list_locales(self) -> List[str]:
        """列出支持的区域"""
        return sorted(set(self._locale_index.keys()))


# ============================================================================
# SSMLProcessor - SSML处理器
# ============================================================================

class SSMLProcessor:
    """
    SSML处理器：解析和处理SSML标记。

    支持的SSML标签:
    - <speak>: 根元素
    - <voice>: 语音选择
    - <prosody>: 韵律控制 (rate, pitch, volume)
    - <break>: 停顿
    - <emphasis>: 强调
    - <say-as>: 朗读方式
    - <phoneme>: 音素发音
    """

    def __init__(self):
        self._tag_patterns = {
            "speak": r'<speak[^>]*>(.*?)</speak>',
            "voice": r'<voice\s+name=["\']([^"\']+)["\'][^>]*>(.*?)</voice>',
            "prosody": r'<prosody\s+([^>]+)>(.*?)</prosody>',
            "break": r'<break\s+([^/]*)/>',
            "emphasis": r'<emphasis\s+level=["\']([^"\']+)["\'][^>]*>(.*?)</emphasis>',
            "say_as": r'<say-as\s+([^>]+)>(.*?)</say-as>',
            "phoneme": r'<phoneme\s+([^>]+)>(.*?)</phoneme>',
        }

    def parse(self, ssml: str) -> Dict[str, Any]:
        """解析SSML文档"""
        import re

        result: Dict[str, Any] = {
            "text": "",
            "voice": None,
            "rate": 1.0,
            "pitch": 1.0,
            "volume": 1.0,
            "segments": [],
        }

        # 提取纯文本
        result["text"] = re.sub(r'<[^>]+>', '', ssml).strip()

        # 解析voice标签
        voice_match = re.search(
            r'<voice\s+name=["\']([^"\']+)["\']', ssml, re.IGNORECASE
        )
        if voice_match:
            result["voice"] = voice_match.group(1)

        # 解析prosody标签
        prosody_match = re.search(
            r'<prosody\s+([^>]+)>', ssml, re.IGNORECASE
        )
        if prosody_match:
            attrs = prosody_match.group(1)
            result["rate"] = self._parse_rate(attrs)
            result["pitch"] = self._parse_pitch(attrs)
            result["volume"] = self._parse_volume(attrs)

        # 解析break标签
        breaks = re.findall(r'<break\s+time=["\']([^"\']+)["\']', ssml, re.IGNORECASE)
        result["breaks"] = breaks

        # 解析emphasis标签
        emphases = re.findall(
            r'<emphasis\s+level=["\']([^"\']+)["\']', ssml, re.IGNORECASE
        )
        result["emphasis_levels"] = emphases

        # 解析segments
        result["segments"] = self._parse_segments(ssml)

        return result

    def _parse_rate(self, attrs: str) -> float:
        """解析语速属性"""
        import re
        match = re.search(r'rate=["\']([^"\']+)["\']', attrs, re.IGNORECASE)
        if not match:
            return 1.0
        val = match.group(1).strip().lower()
        rate_map = {"slow": 0.8, "medium": 1.0, "fast": 1.2, "x-fast": 1.5, "x-slow": 0.6}
        if val in rate_map:
            return rate_map[val]
        try:
            pct = float(val.replace('%', '').replace('+', ''))
            return pct / 100.0
        except ValueError:
            return 1.0

    def _parse_pitch(self, attrs: str) -> float:
        """解析基频属性"""
        import re
        match = re.search(r'pitch=["\']([^"\']+)["\']', attrs, re.IGNORECASE)
        if not match:
            return 1.0
        val = match.group(1).strip().lower()
        pitch_map = {"low": 0.8, "medium": 1.0, "high": 1.2, "x-high": 1.5, "x-low": 0.6}
        if val in pitch_map:
            return pitch_map[val]
        try:
            hz = float(val.replace('Hz', '').replace('hz', '').replace('+', ''))
            return hz / 220.0
        except ValueError:
            return 1.0

    def _parse_volume(self, attrs: str) -> float:
        """解析音量属性"""
        import re
        match = re.search(r'volume=["\']([^"\']+)["\']', attrs, re.IGNORECASE)
        if not match:
            return 1.0
        val = match.group(1).strip().lower()
        vol_map = {"silent": 0.0, "x-soft": 0.3, "soft": 0.6, "medium": 0.8, "loud": 1.0, "x-loud": 1.2}
        if val in vol_map:
            return vol_map[val]
        try:
            pct = float(val.replace('%', '').replace('+', ''))
            return pct / 100.0
        except ValueError:
            return 1.0

    def _parse_segments(self, ssml: str) -> List[Dict[str, Any]]:
        """解析SSML为文本段"""
        import re
        segments: List[Dict[str, Any]] = []

        # 简单分割：按标签边界
        parts = re.split(r'(<[^>]+>)', ssml)
        current_text = ""
        current_attrs: Dict[str, Any] = {}

        for part in parts:
            if part.startswith('<') and not part.startswith('</'):
                if current_text.strip():
                    segments.append({
                        "text": current_text.strip(),
                        "attrs": dict(current_attrs),
                    })
                    current_text = ""
                # 解析标签属性
                tag_match = re.match(r'<(\w+)\s*(.*)', part)
                if tag_match:
                    tag_name = tag_match.group(1)
                    current_attrs["tag"] = tag_name
            elif part.startswith('</'):
                if current_text.strip():
                    segments.append({
                        "text": current_text.strip(),
                        "attrs": dict(current_attrs),
                    })
                    current_text = ""
                current_attrs = {}
            else:
                current_text += part

        if current_text.strip():
            segments.append({
                "text": current_text.strip(),
                "attrs": dict(current_attrs),
            })

        return segments

    def build_ssml(
        self,
        text: str,
        voice: Optional[str] = None,
        rate: Optional[float] = None,
        pitch: Optional[float] = None,
        volume: Optional[float] = None,
    ) -> str:
        """构建SSML文档"""
        parts: List[str] = ['<speak>']

        if voice:
            parts.append(f'<voice name="{voice}">')

        prosody_attrs: List[str] = []
        if rate is not None:
            prosody_attrs.append(f'rate="{int(rate * 100)}%"')
        if pitch is not None:
            prosody_attrs.append(f'pitch="{int(pitch * 100)}%"')
        if volume is not None:
            prosody_attrs.append(f'volume="{int(volume * 100)}%"')

        if prosody_attrs:
            attr_str = " ".join(prosody_attrs)
            parts.append(f'<prosody {attr_str}>{text}</prosody>')
        else:
            parts.append(text)

        if voice:
            parts.append('</voice>')
        parts.append('</speak>')

        return "".join(parts)


# ============================================================================
# ProsodyControl - 韵律控制
# ============================================================================

class ProsodyControl:
    """韵律控制：调整语音的语速、基频和音量"""

    def __init__(self):
        self._rate_range = (0.5, 2.0)
        self._pitch_range = (0.5, 2.0)
        self._volume_range = (0.0, 2.0)

    def adjust_rate(self, samples: List[float], rate: float) -> List[float]:
        """调整语速（重采样）"""
        if rate <= 0 or not samples:
            return samples

        if abs(rate - 1.0) < 0.01:
            return samples

        result: List[float] = []
        src_len = len(samples)
        dst_len = int(src_len / rate)

        for i in range(dst_len):
            src_pos = i * rate
            src_idx = int(src_pos)
            frac = src_pos - src_idx

            if src_idx + 1 < src_len:
                val = _lerp(samples[src_idx], samples[src_idx + 1], frac)
            elif src_idx < src_len:
                val = samples[src_idx]
            else:
                val = 0.0
            result.append(val)

        return result

    def adjust_pitch(self, samples: List[float], pitch: float, sample_rate: int = 24000) -> List[float]:
        """调整基频（相位声码器简化版）"""
        if pitch <= 0 or not samples:
            return samples

        if abs(pitch - 1.0) < 0.01:
            return samples

        # 简化基频调整：通过重采样实现
        # 先拉伸再压缩（保持时长不变）
        stretched = self._resample(samples, 1.0 / pitch)
        result = self._resample(stretched, pitch)

        return result

    def adjust_volume(self, samples: List[float], volume: float) -> List[float]:
        """调整音量"""
        if not samples:
            return samples
        return [_clamp(s * volume, -1.0, 1.0) for s in samples]

    def _resample(self, samples: List[float], factor: float) -> List[float]:
        """重采样"""
        if factor <= 0 or not samples:
            return samples

        result: List[float] = []
        src_len = len(samples)
        dst_len = int(src_len * factor)

        for i in range(dst_len):
            src_pos = i / factor
            src_idx = int(src_pos)
            frac = src_pos - src_idx

            if src_idx + 1 < src_len:
                val = _lerp(samples[src_idx], samples[src_idx + 1], frac)
            elif src_idx < src_len:
                val = samples[src_idx]
            else:
                val = 0.0
            result.append(val)

        return result

    def validate_rate(self, rate: float) -> float:
        return _clamp(rate, self._rate_range[0], self._rate_range[1])

    def validate_pitch(self, pitch: float) -> float:
        return _clamp(pitch, self._pitch_range[0], self._pitch_range[1])

    def validate_volume(self, volume: float) -> float:
        return _clamp(volume, self._volume_range[0], self._volume_range[1])


# ============================================================================
# AudioFormatter - 音频格式器
# ============================================================================

class AudioFormatter:
    """音频格式转换器"""

    def __init__(self):
        self._sample_rate = 24000

    def samples_to_bytes(
        self, samples: List[float], fmt: AudioFormat = AudioFormat.PCM_16BIT
    ) -> bytes:
        """将采样数据转换为字节"""
        if fmt == AudioFormat.PCM_16BIT:
            int_samples = [int(_clamp(s, -1.0, 1.0) * 32767) for s in samples]
            return struct.pack(f'<{len(int_samples)}h', *int_samples)
        elif fmt == AudioFormat.PCM_8BIT:
            int_samples = [int(_clamp((s + 1.0) / 2.0, 0.0, 1.0) * 255) for s in samples]
            return bytes(int_samples)
        elif fmt == AudioFormat.MULAW:
            return self._float_to_mulaw(samples)
        elif fmt == AudioFormat.RAW_FLOAT:
            return struct.pack(f'<{len(samples)}f', *samples)
        else:
            int_samples = [int(_clamp(s, -1.0, 1.0) * 32767) for s in samples]
            return struct.pack(f'<{len(int_samples)}h', *int_samples)

    def bytes_to_samples(
        self, data: bytes, fmt: AudioFormat = AudioFormat.PCM_16BIT
    ) -> List[float]:
        """将字节转换为采样数据"""
        if fmt == AudioFormat.PCM_16BIT:
            n = len(data) // 2
            int_samples = struct.unpack(f'<{n}h', data[:n * 2])
            return [s / 32767.0 for s in int_samples]
        elif fmt == AudioFormat.PCM_8BIT:
            return [(b - 128) / 128.0 for b in data]
        elif fmt == AudioFormat.RAW_FLOAT:
            n = len(data) // 4
            return list(struct.unpack(f'<{n}f', data[:n * 4]))
        else:
            n = len(data) // 2
            int_samples = struct.unpack(f'<{n}h', data[:n * 2])
            return [s / 32767.0 for s in int_samples]

    def _float_to_mulaw(self, samples: List[float]) -> bytes:
        """浮点转mu-law编码"""
        result: List[int] = []
        mu = 255
        for s in samples:
            s = _clamp(s, -1.0, 1.0)
            magnitude = abs(s)
            sign = 0 if s >= 0 else 0x80
            if magnitude < 1.0 / (mu + 1):
                encoded = 0
            else:
                encoded = int(mu * (1.0 + math.log(mu * magnitude) / math.log(1 + mu)))
                encoded = min(encoded, mu)
            result.append(sign | encoded)
        return bytes(result)

    def wrap_wav(self, pcm_data: bytes, sample_rate: int = 24000, bits: int = 16) -> bytes:
        """将PCM数据包装为WAV格式"""
        num_channels = 1
        byte_rate = sample_rate * num_channels * bits // 8
        block_align = num_channels * bits // 8
        data_size = len(pcm_data)
        file_size = 36 + data_size

        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF', file_size, b'WAVE',
            b'fmt ', 16,
            1, num_channels, sample_rate, byte_rate, block_align, bits,
            b'data', data_size,
        )
        return header + pcm_data


# ============================================================================
# RateLimiter - 速率限制器
# ============================================================================

class RateLimiter:
    """速率限制器：控制合成请求频率"""

    def __init__(
        self,
        max_requests_per_minute: int = 60,
        max_concurrent: int = 5,
        max_chars_per_request: int = 5000,
    ):
        self._max_rpm = max_requests_per_minute
        self._max_concurrent = max_concurrent
        self._max_chars = max_chars_per_request
        self._request_times: List[float] = []
        self._active_count = 0
        self._lock = threading.Lock()

    def acquire(self, text_length: int = 0) -> Tuple[bool, float]:
        """
        获取请求许可。

        Returns:
            (allowed, wait_time): 是否允许，需要等待的时间（秒）
        """
        with self._lock:
            now = time.time()

            # 清理过期记录
            self._request_times = [
                t for t in self._request_times if now - t < 60.0
            ]

            # 检查字符限制
            if text_length > self._max_chars:
                return False, 0.0

            # 检查并发限制
            if self._active_count >= self._max_concurrent:
                return False, 0.5

            # 检查频率限制
            if len(self._request_times) >= self._max_rpm:
                oldest = self._request_times[0]
                wait = 60.0 - (now - oldest)
                return False, max(wait, 0.0)

            self._request_times.append(now)
            self._active_count += 1
            return True, 0.0

    def release(self) -> None:
        """释放请求许可"""
        with self._lock:
            self._active_count = max(0, self._active_count - 1)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        now = time.time()
        recent = [t for t in self._request_times if now - t < 60.0]
        return {
            "requests_last_minute": len(recent),
            "max_rpm": self._max_rpm,
            "active_requests": self._active_count,
            "max_concurrent": self._max_concurrent,
        }


# ============================================================================
# BatchSynthesizer - 批量合成器
# ============================================================================

class BatchSynthesizer:
    """批量合成器：高效处理多个合成请求"""

    def __init__(self, edge_tts: Any):
        self._tts = edge_tts
        self._results: List[SynthesisResult] = []
        self._lock = threading.Lock()

    def synthesize_batch(
        self,
        texts: List[str],
        voice_id: str = "en-US-JennyNeural",
        rate: float = 1.0,
        pitch: float = 1.0,
        volume: float = 1.0,
        max_workers: int = 3,
    ) -> List[SynthesisResult]:
        """批量合成文本列表"""
        results: List[SynthesisResult] = []

        for text in texts:
            result = self._tts.synthesize(
                text=text,
                voice_id=voice_id,
                rate=rate,
                pitch=pitch,
                volume=volume,
            )
            results.append(result)

        return results

    def synthesize_ssml_batch(
        self, ssml_texts: List[str], max_workers: int = 3
    ) -> List[SynthesisResult]:
        """批量SSML合成"""
        results: List[SynthesisResult] = []

        for ssml in ssml_texts:
            result = self._tts.synthesize_ssml(ssml)
            results.append(result)

        return results

    def concatenate_results(
        self, results: List[SynthesisResult], gap_seconds: float = 0.5
    ) -> SynthesisResult:
        """拼接多个合成结果"""
        all_samples: List[float] = []
        sr = results[0].sample_rate if results else 24000
        gap_samples = int(gap_seconds * sr)

        for i, result in enumerate(results):
            if i > 0:
                all_samples.extend([0.0] * gap_samples)
            all_samples.extend(result.samples)

        duration = len(all_samples) / sr

        return SynthesisResult(
            request_id=_generate_id(),
            samples=all_samples,
            sample_rate=sr,
            duration=duration,
            success=all(r.success for r in results),
            metadata={"concatenated_count": len(results)},
        )


# ============================================================================
# EdgeTTS - Edge TTS引擎（主入口）
# ============================================================================

class EdgeTTS:
    """
    Edge TTS引擎模拟。

    模拟微软Edge浏览器的TTS服务接口。

    使用方法:
        tts = EdgeTTS()
        result = tts.synthesize("Hello, world!")
    """

    def __init__(
        self,
        default_voice: str = "en-US-JennyNeural",
        sample_rate: int = 24000,
        max_rpm: int = 60,
    ):
        self._default_voice = default_voice
        self._sample_rate = sample_rate
        self._voice_selector = VoiceSelector()
        self._ssml_processor = SSMLProcessor()
        self._prosody_control = ProsodyControl()
        self._audio_formatter = AudioFormatter()
        self._rate_limiter = RateLimiter(max_requests_per_minute=max_rpm)
        self._batch_synthesizer = BatchSynthesizer(self)

    @property
    def voice_selector(self) -> VoiceSelector:
        return self._voice_selector

    @property
    def ssml_processor(self) -> SSMLProcessor:
        return self._ssml_processor

    @property
    def rate_limiter(self) -> RateLimiter:
        return self._rate_limiter

    @property
    def batch_synthesizer(self) -> BatchSynthesizer:
        return self._batch_synthesizer

    def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        rate: float = 1.0,
        pitch: float = 1.0,
        volume: float = 1.0,
        fmt: AudioFormat = AudioFormat.PCM_16BIT,
        container: AudioContainer = AudioContainer.WAV,
    ) -> SynthesisResult:
        """
        文本转语音合成。

        Args:
            text: 输入文本
            voice_id: 语音ID
            rate: 语速 (0.5-2.0)
            pitch: 基频 (0.5-2.0)
            volume: 音量 (0.0-2.0)
            fmt: 音频格式
            container: 音频容器

        Returns:
            SynthesisResult: 合成结果
        """
        request_id = _generate_id()

        # 速率限制检查
        allowed, wait_time = self._rate_limiter.acquire(len(text))
        if not allowed:
            self._rate_limiter.release()
            return SynthesisResult(
                request_id=request_id,
                success=False,
                error=f"Rate limit exceeded. Wait {wait_time:.1f}s.",
            )

        try:
            voice = voice_id or self._default_voice
            voice_info = self._voice_selector.get_voice(voice)

            # 验证参数
            rate = self._prosody_control.validate_rate(rate)
            pitch = self._prosody_control.validate_pitch(pitch)
            volume = self._prosody_control.validate_volume(volume)

            # 生成波形
            samples = self._generate_speech(text, voice_info, rate, pitch, volume)

            # 格式转换
            audio_bytes = self._audio_formatter.samples_to_bytes(samples, fmt)

            if container == AudioContainer.WAV:
                audio_bytes = self._audio_formatter.wrap_wav(
                    audio_bytes, self._sample_rate, 16
                )

            duration = len(samples) / self._sample_rate

            return SynthesisResult(
                request_id=request_id,
                audio_data=audio_bytes,
                samples=samples,
                sample_rate=self._sample_rate,
                duration=duration,
                format=fmt,
                container=container,
                success=True,
                metadata={
                    "voice": voice,
                    "text_length": len(text),
                    "generation_id": _generate_id(),
                },
            )
        finally:
            self._rate_limiter.release()

    def synthesize_ssml(self, ssml: str) -> SynthesisResult:
        """使用SSML合成"""
        parsed = self._ssml_processor.parse(ssml)

        voice_id = parsed.get("voice") or self._default_voice
        rate = parsed.get("rate", 1.0)
        pitch = parsed.get("pitch", 1.0)
        volume = parsed.get("volume", 1.0)
        text = parsed.get("text", "")

        return self.synthesize(
            text=text,
            voice_id=voice_id,
            rate=rate,
            pitch=pitch,
            volume=volume,
        )

    def _generate_speech(
        self,
        text: str,
        voice_info: Optional[VoiceInfo],
        rate: float,
        pitch: float,
        volume: float,
    ) -> List[float]:
        """生成语音波形"""
        base_pitch = 220.0
        if voice_info:
            if voice_info.gender == "Male":
                base_pitch = 140.0
            else:
                base_pitch = 240.0

        adjusted_pitch = base_pitch * pitch
        samples: List[float] = []
        sr = self._sample_rate

        # 简单的文本到波形转换
        words = text.lower().split()
        for word_idx, word in enumerate(words):
            # 每个词的持续时间
            word_duration = (0.15 + len(word) * 0.04) / rate
            num_samples = int(word_duration * sr)

            for s in range(num_samples):
                t = s / sr
                # 基频随词索引微调
                word_pitch = adjusted_pitch + (word_idx % 5 - 2) * 10.0

                phase = 2.0 * math.pi * word_pitch * t
                sample = volume * 0.4 * math.sin(phase)
                sample += volume * 0.2 * math.sin(2.0 * phase)
                sample += volume * 0.1 * math.sin(3.0 * phase)

                # 包络
                env_pos = s / max(num_samples - 1, 1)
                envelope = math.sin(math.pi * env_pos)
                sample *= envelope

                sample += _gaussian_noise(0, volume * 0.015)
                samples.append(_clamp(sample, -1.0, 1.0))

            # 词间停顿
            pause_samples = int(0.03 * sr / rate)
            samples.extend([0.0] * pause_samples)

        # 归一化
        if samples:
            max_val = max(abs(s) for s in samples)
            if max_val > 0:
                samples = [s / max_val * 0.9 for s in samples]

        return samples

    def list_voices(
        self, locale: Optional[str] = None, gender: Optional[str] = None
    ) -> List[VoiceInfo]:
        """列出可用语音"""
        return self._voice_selector.list_voices(locale, gender)

    def get_voice_info(self, voice_id: str) -> Optional[VoiceInfo]:
        """获取语音信息"""
        return self._voice_selector.get_voice(voice_id)
