"""
Bark TTS Engine Simulation - Bark文本转语音引擎

本模块实现了Bark TTS引擎的模拟，包含音素到波形转换、韵律控制、
情感/风格迁移、多语言支持和语音预设功能。仅使用标准库，
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


def _hanning_window(n: int) -> List[float]:
    """汉宁窗"""
    if n <= 1:
        return [1.0]
    return [0.5 * (1.0 - math.cos(2.0 * math.pi * i / (n - 1))) for i in range(n)]


# ============================================================================
# 数据结构
# ============================================================================

class EmotionType(Enum):
    """情感类型"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    CALM = "calm"
    EXCITED = "excited"
    WHISPER = "whisper"


@dataclass
class Phoneme:
    """音素"""
    symbol: str = ""
    duration: float = 0.08
    pitch: float = 220.0
    energy: float = 0.8
    start_time: float = 0.0


@dataclass
class ProsodyParams:
    """韵律参数"""
    pitch_mean: float = 220.0
    pitch_range: float = 40.0
    speed: float = 1.0
    energy: float = 0.8
    pause_duration: float = 0.1
    pitch_contour: List[float] = field(default_factory=list)
    stress_pattern: List[float] = field(default_factory=list)


@dataclass
class BarkConfig:
    """Bark配置"""
    sample_rate: int = 24000
    hop_length: int = 256
    n_fft: int = 1024
    n_mels: int = 128
    semantic_frames_per_second: float = 49.9
    coarse_frames_per_second: float = 75.0
    fine_frames_per_second: float = 75.0
    max_generation_duration: float = 15.0
    temperature: float = 0.7
    min_eos_p: float = 0.05
    codec_downsample: int = 8
    language: str = "en"
    voice_preset: str = "default"


@dataclass
class AudioOutput:
    """音频输出"""
    samples: List[float] = field(default_factory=list)
    sample_rate: int = 24000
    duration: float = 0.0
    phonemes: List[Phoneme] = field(default_factory=list)
    prosody: Optional[ProsodyParams] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# PhonemeConverter - 音素转换器
# ============================================================================

class PhonemeConverter:
    """
    音素转换器：将文本转换为音素序列。

    支持英语和中文的基本音素映射。
    """

    def __init__(self):
        self._grapheme_to_phoneme: Dict[str, List[str]] = {
            'a': ['AH'], 'e': ['EH'], 'i': ['IH'], 'o': ['AO'], 'u': ['UH'],
            'b': ['B'], 'c': ['K'], 'd': ['D'], 'f': ['F'], 'g': ['G'],
            'h': ['HH'], 'j': ['JH'], 'k': ['K'], 'l': ['L'], 'm': ['M'],
            'n': ['N'], 'p': ['P'], 'q': ['K'], 'r': ['R'], 's': ['S'],
            't': ['T'], 'v': ['V'], 'w': ['W'], 'x': ['K', 'S'], 'y': ['Y'],
            'z': ['Z'],
            'th': ['TH'], 'sh': ['SH'], 'ch': ['CH'], 'ph': ['F'],
            'ng': ['NG'], 'ck': ['K'],
            'ee': ['IY'], 'oo': ['UW'], 'ea': ['IY'], 'ou': ['OW'],
            'ai': ['EY'], 'ay': ['EY'], 'oi': ['OY'], 'oy': ['OY'],
            'er': ['ER'], 'ir': ['ER'], 'ur': ['ER'],
            'ar': ['AA', 'R'], 'or': ['AO', 'R'],
            'tion': ['SH', 'AH', 'N'], 'sion': ['ZH', 'AH', 'N'],
            'ed': ['T'], 'es': ['EH', 'Z'], 'ly': ['L', 'IH'],
            'ing': ['IH', 'NG'], 'ight': ['AY', 'T'],
            'ough': ['AO', 'F'], 'augh': ['AO', 'F'],
        }

        self._phoneme_durations: Dict[str, float] = {
            'AH': 0.09, 'EH': 0.07, 'IH': 0.06, 'AO': 0.08, 'UH': 0.07,
            'IY': 0.08, 'UW': 0.09, 'EY': 0.09, 'OW': 0.09, 'OY': 0.10,
            'AA': 0.09, 'ER': 0.08,
            'B': 0.06, 'D': 0.05, 'F': 0.07, 'G': 0.06, 'HH': 0.05,
            'JH': 0.07, 'K': 0.06, 'L': 0.06, 'M': 0.07, 'N': 0.06,
            'NG': 0.08, 'P': 0.06, 'R': 0.06, 'S': 0.06, 'SH': 0.08,
            'T': 0.05, 'TH': 0.07, 'V': 0.06, 'W': 0.05, 'Y': 0.04,
            'Z': 0.06, 'ZH': 0.08, 'CH': 0.08,
        }

        self._vowels = {
            'AH', 'EH', 'IH', 'AO', 'UH', 'IY', 'UW', 'EY', 'OW',
            'OY', 'AA', 'ER',
        }

    def text_to_phonemes(self, text: str, language: str = "en") -> List[Phoneme]:
        """将文本转换为音素序列"""
        if not text.strip():
            return []

        words = text.lower().split()
        phonemes: List[Phoneme] = []
        current_time = 0.0

        for word in words:
            word_phonemes = self._word_to_phonemes(word, language)
            for sym in word_phonemes:
                duration = self._phoneme_durations.get(sym, 0.07)
                phonemes.append(Phoneme(
                    symbol=sym,
                    duration=duration,
                    pitch=220.0,
                    energy=0.8,
                    start_time=current_time,
                ))
                current_time += duration

            # 词间停顿
            phonemes.append(Phoneme(
                symbol="PAUSE",
                duration=0.05,
                pitch=0.0,
                energy=0.0,
                start_time=current_time,
            ))
            current_time += 0.05

        return phonemes

    def _word_to_phonemes(self, word: str, language: str) -> List[str]:
        """将单词转换为音素"""
        if language == "zh":
            return self._chinese_to_phonemes(word)

        # 英语：最长匹配
        result: List[str] = []
        i = 0
        while i < len(word):
            matched = False
            # 尝试从长到短匹配
            for length in range(min(4, len(word) - i), 1, -1):
                substr = word[i:i + length]
                if substr in self._grapheme_to_phoneme:
                    result.extend(self._grapheme_to_phoneme[substr])
                    i += length
                    matched = True
                    break
            if not matched:
                char = word[i]
                if char in self._grapheme_to_phoneme:
                    result.extend(self._grapheme_to_phoneme[char])
                i += 1

        return result

    def _chinese_to_phonemes(self, text: str) -> List[str]:
        """中文文本转音素（简化版）"""
        result: List[str] = []
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                # 使用字符编码生成伪音素
                code = ord(char)
                vowel_idx = code % 12
                consonant_idx = (code // 12) % 24
                vowels = ['AH', 'EH', 'IH', 'AO', 'UH', 'IY', 'UW', 'EY', 'OW', 'OY', 'AA', 'ER']
                consonants = ['B', 'D', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH', 'CH', 'NG', 'S']
                result.append(consonants[consonant_idx])
                result.append(vowels[vowel_idx])
            elif char.isalpha():
                if char in self._grapheme_to_phoneme:
                    result.extend(self._grapheme_to_phoneme[char])
        return result

    def is_vowel(self, phoneme: str) -> bool:
        """判断是否为元音"""
        return phoneme in self._vowels


# ============================================================================
# ProsodyController - 韵律控制器
# ============================================================================

class ProsodyController:
    """
    韵律控制器：控制语音的韵律特征。

    包括基频（F0）轮廓、语速、能量和停顿控制。
    """

    def __init__(self):
        self._default_params = ProsodyParams()

    def generate_pitch_contour(
        self, phonemes: List[Phoneme], params: ProsodyParams
    ) -> List[float]:
        """生成基频轮廓"""
        if not phonemes:
            return []

        contour: List[float] = []
        n = len(phonemes)

        for i, ph in enumerate(phonemes):
            if ph.symbol == "PAUSE":
                contour.append(0.0)
                continue

            # 基础音高
            base_pitch = params.pitch_mean

            # 重音模式
            stress = 1.0
            if params.stress_pattern and i < len(params.stress_pattern):
                stress = params.stress_pattern[i]

            # 语句末尾下降
            if i > n * 0.7:
                declination = (i - n * 0.7) / (n * 0.3) * 20.0
            else:
                declination = 0.0

            # 微小随机变化
            jitter = _gaussian_noise(0, params.pitch_range * 0.05)

            pitch = base_pitch * stress - declination + jitter
            contour.append(_clamp(pitch, 80.0, 500.0))

        return contour

    def apply_speed(
        self, phonemes: List[Phoneme], speed: float
    ) -> List[Phoneme]:
        """应用语速调整"""
        if speed <= 0:
            return phonemes

        result: List[Phoneme] = []
        current_time = 0.0

        for ph in phonemes:
            new_ph = Phoneme(
                symbol=ph.symbol,
                duration=ph.duration / speed,
                pitch=ph.pitch,
                energy=ph.energy,
                start_time=current_time,
            )
            result.append(new_ph)
            current_time += new_ph.duration

        return result

    def apply_energy(
        self, phonemes: List[Phoneme], energy: float
    ) -> List[Phoneme]:
        """应用能量调整"""
        result: List[Phoneme] = []
        for ph in phonemes:
            new_ph = Phoneme(
                symbol=ph.symbol,
                duration=ph.duration,
                pitch=ph.pitch,
                energy=_clamp(ph.energy * energy, 0.0, 1.0),
                start_time=ph.start_time,
            )
            result.append(new_ph)
        return result

    def add_pauses(
        self, phonemes: List[Phoneme], pause_positions: List[int],
        pause_duration: float,
    ) -> List[Phoneme]:
        """在指定位置添加停顿"""
        result: List[Phoneme] = []
        for i, ph in enumerate(phonemes):
            result.append(ph)
            if i in pause_positions:
                pause = Phoneme(
                    symbol="PAUSE",
                    duration=pause_duration,
                    pitch=0.0,
                    energy=0.0,
                    start_time=ph.start_time + ph.duration,
                )
                result.append(pause)

        # 更新时间戳
        current_time = 0.0
        for ph in result:
            ph.start_time = current_time
            current_time += ph.duration

        return result

    def detect_sentence_boundaries(
        self, phonemes: List[Phoneme]
    ) -> List[int]:
        """检测句子边界（停顿位置）"""
        boundaries: List[int] = []
        for i, ph in enumerate(phonemes):
            if ph.symbol == "PAUSE" and ph.duration > 0.04:
                boundaries.append(i)
        return boundaries

    def compute_prosody_params(
        self, phonemes: List[Phoneme]
    ) -> ProsodyParams:
        """从音素序列计算韵律参数"""
        if not phonemes:
            return self._default_params

        pitches = [ph.pitch for ph in phonemes if ph.symbol != "PAUSE" and ph.pitch > 0]
        energies = [ph.energy for ph in phonemes if ph.symbol != "PAUSE"]

        pitch_mean = sum(pitches) / max(len(pitches), 1)
        pitch_range = (
            max(pitches) - min(pitches) if len(pitches) > 1 else 40.0
        )
        energy_mean = sum(energies) / max(len(energies), 1)

        # 生成重音模式（简单启发式）
        stress_pattern: List[float] = []
        for ph in phonemes:
            if ph.symbol == "PAUSE":
                stress_pattern.append(0.0)
            else:
                stress_pattern.append(1.0 + _gaussian_noise(0, 0.1))

        return ProsodyParams(
            pitch_mean=pitch_mean,
            pitch_range=pitch_range,
            speed=1.0,
            energy=energy_mean,
            stress_pattern=stress_pattern,
        )


# ============================================================================
# EmotionTransfer - 情感迁移
# ============================================================================

class EmotionTransfer:
    """
    情感迁移器：将情感特征应用到语音合成中。

    通过调整韵律参数来模拟不同情感。
    """

    def __init__(self):
        self._emotion_profiles: Dict[EmotionType, Dict[str, Any]] = {
            EmotionType.NEUTRAL: {
                "pitch_shift": 0.0,
                "pitch_range_factor": 1.0,
                "speed_factor": 1.0,
                "energy_factor": 0.8,
                "jitter_factor": 0.02,
            },
            EmotionType.HAPPY: {
                "pitch_shift": 30.0,
                "pitch_range_factor": 1.4,
                "speed_factor": 1.1,
                "energy_factor": 0.9,
                "jitter_factor": 0.03,
            },
            EmotionType.SAD: {
                "pitch_shift": -20.0,
                "pitch_range_factor": 0.6,
                "speed_factor": 0.85,
                "energy_factor": 0.5,
                "jitter_factor": 0.01,
            },
            EmotionType.ANGRY: {
                "pitch_shift": 20.0,
                "pitch_range_factor": 1.6,
                "speed_factor": 1.15,
                "energy_factor": 1.0,
                "jitter_factor": 0.04,
            },
            EmotionType.SURPRISED: {
                "pitch_shift": 50.0,
                "pitch_range_factor": 1.8,
                "speed_factor": 1.2,
                "energy_factor": 0.95,
                "jitter_factor": 0.05,
            },
            EmotionType.CALM: {
                "pitch_shift": -10.0,
                "pitch_range_factor": 0.7,
                "speed_factor": 0.9,
                "energy_factor": 0.6,
                "jitter_factor": 0.01,
            },
            EmotionType.EXCITED: {
                "pitch_shift": 40.0,
                "pitch_range_factor": 1.5,
                "speed_factor": 1.2,
                "energy_factor": 1.0,
                "jitter_factor": 0.04,
            },
            EmotionType.WHISPER: {
                "pitch_shift": -30.0,
                "pitch_range_factor": 0.4,
                "speed_factor": 0.95,
                "energy_factor": 0.2,
                "jitter_factor": 0.02,
            },
        }

    def apply_emotion(
        self,
        phonemes: List[Phoneme],
        emotion: EmotionType,
    ) -> List[Phoneme]:
        """将情感应用到音素序列"""
        profile = self._emotion_profiles.get(emotion, self._emotion_profiles[EmotionType.NEUTRAL])

        result: List[Phoneme] = []
        for ph in phonemes:
            if ph.symbol == "PAUSE":
                result.append(ph)
                continue

            new_pitch = ph.pitch + profile["pitch_shift"]
            new_duration = ph.duration / profile["speed_factor"]
            new_energy = _clamp(
                ph.energy * profile["energy_factor"], 0.0, 1.0
            )

            result.append(Phoneme(
                symbol=ph.symbol,
                duration=new_duration,
                pitch=_clamp(new_pitch, 60.0, 600.0),
                energy=new_energy,
                start_time=ph.start_time,
            ))

        # 更新时间戳
        current_time = 0.0
        for ph in result:
            ph.start_time = current_time
            current_time += ph.duration

        return result

    def get_emotion_params(
        self, emotion: EmotionType
    ) -> Dict[str, Any]:
        """获取情感参数"""
        return dict(self._emotion_profiles.get(
            emotion, self._emotion_profiles[EmotionType.NEUTRAL]
        ))

    def blend_emotions(
        self,
        phonemes: List[Phoneme],
        emotions: List[Tuple[EmotionType, float]],
    ) -> List[Phoneme]:
        """混合多种情感"""
        total_weight = sum(w for _, w in emotions)
        if total_weight == 0:
            return phonemes

        # 计算混合参数
        blended_shift = 0.0
        blended_range = 0.0
        blended_speed = 0.0
        blended_energy = 0.0

        for emotion, weight in emotions:
            profile = self._emotion_profiles[emotion]
            norm_w = weight / total_weight
            blended_shift += profile["pitch_shift"] * norm_w
            blended_range += profile["pitch_range_factor"] * norm_w
            blended_speed += profile["speed_factor"] * norm_w
            blended_energy += profile["energy_factor"] * norm_w

        result: List[Phoneme] = []
        for ph in phonemes:
            if ph.symbol == "PAUSE":
                result.append(ph)
                continue

            result.append(Phoneme(
                symbol=ph.symbol,
                duration=ph.duration / blended_speed,
                pitch=_clamp(ph.pitch + blended_shift, 60.0, 600.0),
                energy=_clamp(ph.energy * blended_energy, 0.0, 1.0),
                start_time=ph.start_time,
            ))

        current_time = 0.0
        for ph in result:
            ph.start_time = current_time
            current_time += ph.duration

        return result


# ============================================================================
# LanguageSupport - 多语言支持
# ============================================================================

class LanguageSupport:
    """
    多语言支持：处理不同语言的文本到语音转换。

    支持语言检测和语言特定的处理规则。
    """

    def __init__(self):
        self._supported_languages: Dict[str, Dict[str, Any]] = {
            "en": {
                "name": "English",
                "phonemizer": "english",
                "default_pitch": 220.0,
                "default_speed": 1.0,
            },
            "zh": {
                "name": "Chinese",
                "phonemizer": "chinese",
                "default_pitch": 200.0,
                "default_speed": 0.95,
            },
            "ja": {
                "name": "Japanese",
                "phonemizer": "japanese",
                "default_pitch": 230.0,
                "default_speed": 0.9,
            },
            "ko": {
                "name": "Korean",
                "phonemizer": "korean",
                "default_pitch": 210.0,
                "default_speed": 0.95,
            },
            "es": {
                "name": "Spanish",
                "phonemizer": "spanish",
                "default_pitch": 215.0,
                "default_speed": 1.0,
            },
            "fr": {
                "name": "French",
                "phonemizer": "french",
                "default_pitch": 210.0,
                "default_speed": 0.95,
            },
            "de": {
                "name": "German",
                "phonemizer": "german",
                "default_pitch": 205.0,
                "default_speed": 0.95,
            },
        }

    def detect_language(self, text: str) -> str:
        """检测文本语言"""
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        japanese_chars = sum(1 for c in text if '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff')
        korean_chars = sum(1 for c in text if '\uac00' <= c <= '\ud7af')
        latin_chars = sum(1 for c in text if c.isalpha() and ord(c) < 128)

        total = len(text)
        if total == 0:
            return "en"

        if chinese_chars / total > 0.3:
            return "zh"
        elif japanese_chars / total > 0.2:
            return "ja"
        elif korean_chars / total > 0.2:
            return "ko"
        elif latin_chars / total > 0.3:
            return "en"

        return "en"

    def get_language_config(self, language: str) -> Dict[str, Any]:
        """获取语言配置"""
        return dict(self._supported_languages.get(language, self._supported_languages["en"]))

    def is_supported(self, language: str) -> bool:
        return language in self._supported_languages

    def list_languages(self) -> List[str]:
        return list(self._supported_languages.keys())


# ============================================================================
# VoicePreset - 语音预设
# ============================================================================

class VoicePreset:
    """
    语音预设管理：预定义的语音风格配置。
    """

    def __init__(self):
        self._presets: Dict[str, Dict[str, Any]] = {
            "default": {
                "pitch_mean": 220.0,
                "pitch_range": 40.0,
                "speed": 1.0,
                "energy": 0.8,
                "timbre": "neutral",
            },
            "male_deep": {
                "pitch_mean": 130.0,
                "pitch_range": 30.0,
                "speed": 0.95,
                "energy": 0.85,
                "timbre": "deep",
            },
            "female_bright": {
                "pitch_mean": 280.0,
                "pitch_range": 50.0,
                "speed": 1.05,
                "energy": 0.75,
                "timbre": "bright",
            },
            "child": {
                "pitch_mean": 350.0,
                "pitch_range": 60.0,
                "speed": 1.1,
                "energy": 0.7,
                "timbre": "thin",
            },
            "elderly_male": {
                "pitch_mean": 150.0,
                "pitch_range": 25.0,
                "speed": 0.85,
                "energy": 0.65,
                "timbre": "warm",
            },
            "narrator": {
                "pitch_mean": 200.0,
                "pitch_range": 35.0,
                "speed": 0.9,
                "energy": 0.75,
                "timbre": "warm",
            },
            "robot": {
                "pitch_mean": 180.0,
                "pitch_range": 5.0,
                "speed": 1.0,
                "energy": 0.9,
                "timbre": "metallic",
            },
            "whisper_soft": {
                "pitch_mean": 190.0,
                "pitch_range": 20.0,
                "speed": 0.9,
                "energy": 0.25,
                "timbre": "breathy",
            },
        }

    def get_preset(self, name: str) -> Optional[Dict[str, Any]]:
        return dict(self._presets.get(name)) if name in self._presets else None

    def list_presets(self) -> List[str]:
        return list(self._presets.keys())

    def create_custom_preset(
        self, name: str, params: Dict[str, Any]
    ) -> bool:
        if name in self._presets:
            return False
        self._presets[name] = params
        return True

    def apply_preset(
        self, phonemes: List[Phoneme], preset_name: str
    ) -> List[Phoneme]:
        """应用预设到音素序列"""
        preset = self._presets.get(preset_name)
        if not preset:
            return phonemes

        result: List[Phoneme] = []
        for ph in phonemes:
            if ph.symbol == "PAUSE":
                result.append(ph)
                continue

            new_pitch = ph.pitch * (preset["pitch_mean"] / 220.0)
            new_duration = ph.duration / preset["speed"]
            new_energy = _clamp(ph.energy * (preset["energy"] / 0.8), 0.0, 1.0)

            result.append(Phoneme(
                symbol=ph.symbol,
                duration=new_duration,
                pitch=_clamp(new_pitch, 60.0, 600.0),
                energy=new_energy,
                start_time=ph.start_time,
            ))

        current_time = 0.0
        for ph in result:
            ph.start_time = current_time
            current_time += ph.duration

        return result


# ============================================================================
# BarkTTS - Bark TTS引擎（主入口）
# ============================================================================

class BarkTTS:
    """
    Bark TTS引擎：文本到语音的完整管线。

    流程:
    1. 文本预处理和语言检测
    2. 文本到音素转换
    3. 韵律参数生成
    4. 情感/风格应用
    5. 语音预设应用
    6. 波形生成
    7. 后处理

    使用方法:
        tts = BarkTTS()
        audio = tts.synthesize("Hello, world!")
    """

    def __init__(self, config: Optional[BarkConfig] = None):
        self._config = config or BarkConfig()
        self._phoneme_converter = PhonemeConverter()
        self._prosody_controller = ProsodyController()
        self._emotion_transfer = EmotionTransfer()
        self._language_support = LanguageSupport()
        self._voice_preset = VoicePreset()

    @property
    def config(self) -> BarkConfig:
        return self._config

    def synthesize(
        self,
        text: str,
        emotion: EmotionType = EmotionType.NEUTRAL,
        voice_preset: str = "default",
        speed: Optional[float] = None,
        pitch: Optional[float] = None,
        language: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> AudioOutput:
        """
        文本转语音合成。

        Args:
            text: 输入文本
            emotion: 情感类型
            voice_preset: 语音预设名称
            speed: 语速（覆盖预设）
            pitch: 基频（覆盖预设）
            language: 语言代码（自动检测如果为None）
            seed: 随机种子

        Returns:
            AudioOutput: 音频输出
        """
        if seed is not None:
            random.seed(seed)

        if not text.strip():
            return AudioOutput(sample_rate=self._config.sample_rate)

        # 步骤1: 语言检测
        lang = language or self._language_support.detect_language(text)
        lang_config = self._language_support.get_language_config(lang)

        # 步骤2: 文本到音素
        phonemes = self._phoneme_converter.text_to_phonemes(text, lang)

        # 步骤3: 应用语音预设
        phonemes = self._voice_preset.apply_preset(phonemes, voice_preset)

        # 步骤4: 应用情感
        if emotion != EmotionType.NEUTRAL:
            phonemes = self._emotion_transfer.apply_emotion(phonemes, emotion)

        # 步骤5: 应用语速和基频覆盖
        if speed is not None:
            phonemes = self._prosody_controller.apply_speed(phonemes, speed)

        if pitch is not None:
            for ph in phonemes:
                if ph.symbol != "PAUSE":
                    ph.pitch = _clamp(ph.pitch * (pitch / 220.0), 60.0, 600.0)

        # 步骤6: 生成韵律参数
        prosody = self._prosody_controller.compute_prosody_params(phonemes)
        pitch_contour = self._prosody_controller.generate_pitch_contour(
            phonemes, prosody
        )

        # 步骤7: 波形生成
        samples = self._generate_waveform(phonemes, pitch_contour)

        # 步骤8: 后处理
        samples = self._post_process(samples)

        duration = len(samples) / self._config.sample_rate

        return AudioOutput(
            samples=samples,
            sample_rate=self._config.sample_rate,
            duration=duration,
            phonemes=phonemes,
            prosody=prosody,
            metadata={
                "text": text,
                "language": lang,
                "emotion": emotion.value,
                "voice_preset": voice_preset,
                "generation_id": _generate_id(),
            },
        )

    def _generate_waveform(
        self, phonemes: List[Phoneme], pitch_contour: List[float]
    ) -> List[float]:
        """从音素序列生成波形"""
        samples: List[float] = []
        sr = self._config.sample_rate

        for i, ph in enumerate(phonemes):
            if ph.symbol == "PAUSE":
                # 停顿：生成静音
                num_samples = int(ph.duration * sr)
                samples.extend([0.0] * num_samples)
                continue

            pitch = pitch_contour[i] if i < len(pitch_contour) else ph.pitch
            num_samples = int(ph.duration * sr)
            energy = ph.energy

            for s in range(num_samples):
                t = s / sr
                phase = 2.0 * math.pi * pitch * t

                # 基频 + 谐波
                sample = energy * 0.5 * math.sin(phase)
                sample += energy * 0.25 * math.sin(2.0 * phase)
                sample += energy * 0.12 * math.sin(3.0 * phase)
                sample += energy * 0.06 * math.sin(4.0 * phase)

                # 加窗（淡入淡出）
                window_pos = s / max(num_samples - 1, 1)
                envelope = math.sin(math.pi * window_pos)
                sample *= envelope

                # 轻微噪声（模拟声带振动的不规则性）
                sample += _gaussian_noise(0, energy * 0.02)

                samples.append(_clamp(sample, -1.0, 1.0))

        return samples

    def _post_process(self, samples: List[float]) -> List[float]:
        """音频后处理"""
        if not samples:
            return samples

        # 归一化
        max_val = max(abs(s) for s in samples)
        if max_val > 0:
            samples = [s / max_val * 0.9 for s in samples]

        # 淡入淡出
        fade_len = min(int(self._config.sample_rate * 0.01), len(samples) // 4)
        for i in range(fade_len):
            factor = i / fade_len
            samples[i] *= factor
            samples[-(i + 1)] *= factor

        return samples

    def synthesize_with_ssml(
        self, ssml: str, seed: Optional[int] = None
    ) -> AudioOutput:
        """使用SSML格式合成"""
        text = self._parse_ssml_text(ssml)
        emotion = self._parse_ssml_emotion(ssml)
        preset = self._parse_ssml_voice(ssml)
        speed = self._parse_ssml_rate(ssml)
        pitch = self._parse_ssml_pitch(ssml)

        return self.synthesize(
            text, emotion=emotion, voice_preset=preset,
            speed=speed, pitch=pitch, seed=seed,
        )

    def _parse_ssml_text(self, ssml: str) -> str:
        """从SSML中提取纯文本"""
        import re
        text = re.sub(r'<[^>]+>', '', ssml)
        return text.strip()

    def _parse_ssml_emotion(self, ssml: str) -> EmotionType:
        """从SSML中解析情感"""
        ssml_lower = ssml.lower()
        for emotion in EmotionType:
            if emotion.value in ssml_lower:
                return emotion
        return EmotionType.NEUTRAL

    def _parse_ssml_voice(self, ssml: str) -> str:
        """从SSML中解析语音"""
        import re
        match = re.search(r'voice\s*=\s*["\']([^"\']+)["\']', ssml, re.IGNORECASE)
        return match.group(1) if match else "default"

    def _parse_ssml_rate(self, ssml: str) -> Optional[float]:
        """从SSML中解析语速"""
        import re
        match = re.search(r'rate\s*=\s*["\']([^"\']+)["\']', ssml, re.IGNORECASE)
        if match:
            val = match.group(1).replace('%', '')
            try:
                return float(val) / 100.0
            except ValueError:
                return None
        return None

    def _parse_ssml_pitch(self, ssml: str) -> Optional[float]:
        """从SSML中解析基频"""
        import re
        match = re.search(r'pitch\s*=\s*["\']([^"\']+)["\']', ssml, re.IGNORECASE)
        if match:
            val = match.group(1).replace('Hz', '').replace('hz', '').strip()
            try:
                return float(val)
            except ValueError:
                return None
        return None
