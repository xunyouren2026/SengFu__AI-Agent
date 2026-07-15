"""
Sound Effects Synthesis - 音效合成引擎

本模块实现了程序化音效合成系统，包含文本描述到参数映射、波形合成、
包络整形和效果链功能。仅使用标准库，不依赖外部库。
"""

import math
import random
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

class WaveformType(Enum):
    """波形类型"""
    SINE = "sine"
    SAW = "sawtooth"
    SQUARE = "square"
    TRIANGLE = "triangle"
    NOISE_WHITE = "white_noise"
    NOISE_PINK = "pink_noise"
    NOISE_BROWN = "brown_noise"


class EnvelopeType(Enum):
    """包络类型"""
    ADSR = "adsr"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    PERCUSSIVE = "percussive"
    REVERSE = "reverse"


@dataclass
class SynthParams:
    """合成参数"""
    waveform: WaveformType = WaveformType.SINE
    frequency: float = 440.0
    frequency_end: float = 440.0
    amplitude: float = 0.8
    duration: float = 1.0
    attack: float = 0.01
    decay: float = 0.1
    sustain: float = 0.7
    release: float = 0.2
    detune: float = 0.0
    harmonics: int = 1
    harmonic_decay: float = 0.5
    noise_mix: float = 0.0
    vibrato_rate: float = 0.0
    vibrato_depth: float = 0.0
    envelope_type: EnvelopeType = EnvelopeType.ADSR


@dataclass
class EffectParams:
    """效果参数"""
    effect_type: str = ""
    params: Dict[str, float] = field(default_factory=dict)


@dataclass
class SoundEffect:
    """音效"""
    name: str = ""
    samples: List[float] = field(default_factory=list)
    sample_rate: int = 44100
    duration: float = 0.0
    params: Optional[SynthParams] = None
    category: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# WaveformGenerator - 波形生成器
# ============================================================================

class WaveformGenerator:
    """
    波形生成器：生成各种基本波形。

    支持正弦波、锯齿波、方波、三角波和各种噪声。
    """

    def __init__(self, sample_rate: int = 44100):
        self._sample_rate = sample_rate
        self._pink_noise_state: float = 0.0

    def generate(
        self, waveform: WaveformType, frequency: float,
        duration: float, amplitude: float = 0.8,
        phase_offset: float = 0.0,
    ) -> List[float]:
        """生成波形"""
        num_samples = int(duration * self._sample_rate)
        samples: List[float] = []

        if waveform == WaveformType.SINE:
            samples = self._sine(frequency, num_samples, amplitude, phase_offset)
        elif waveform == WaveformType.SAW:
            samples = self._sawtooth(frequency, num_samples, amplitude)
        elif waveform == WaveformType.SQUARE:
            samples = self._square(frequency, num_samples, amplitude)
        elif waveform == WaveformType.TRIANGLE:
            samples = self._triangle(frequency, num_samples, amplitude)
        elif waveform == WaveformType.NOISE_WHITE:
            samples = self._white_noise(num_samples, amplitude)
        elif waveform == WaveformType.NOISE_PINK:
            samples = self._pink_noise(num_samples, amplitude)
        elif waveform == WaveformType.NOISE_BROWN:
            samples = self._brown_noise(num_samples, amplitude)

        return samples

    def _sine(
        self, freq: float, num_samples: int,
        amplitude: float, phase: float,
    ) -> List[float]:
        samples: List[float] = []
        for i in range(num_samples):
            t = i / self._sample_rate
            val = amplitude * math.sin(2.0 * math.pi * freq * t + phase)
            samples.append(val)
        return samples

    def _sawtooth(
        self, freq: float, num_samples: int, amplitude: float
    ) -> List[float]:
        samples: List[float] = []
        period = self._sample_rate / freq
        for i in range(num_samples):
            t = (i % period) / period
            val = amplitude * (2.0 * t - 1.0)
            samples.append(val)
        return samples

    def _square(
        self, freq: float, num_samples: int, amplitude: float
    ) -> List[float]:
        samples: List[float] = []
        period = self._sample_rate / freq
        for i in range(num_samples):
            t = (i % period) / period
            val = amplitude if t < 0.5 else -amplitude
            samples.append(val)
        return samples

    def _triangle(
        self, freq: float, num_samples: int, amplitude: float
    ) -> List[float]:
        samples: List[float] = []
        period = self._sample_rate / freq
        for i in range(num_samples):
            t = (i % period) / period
            if t < 0.25:
                val = amplitude * (t * 4.0)
            elif t < 0.75:
                val = amplitude * (2.0 - t * 4.0)
            else:
                val = amplitude * (t * 4.0 - 4.0)
            samples.append(val)
        return samples

    def _white_noise(
        self, num_samples: int, amplitude: float
    ) -> List[float]:
        return [amplitude * _gaussian_noise(0, 1.0) for _ in range(num_samples)]

    def _pink_noise(
        self, num_samples: int, amplitude: float
    ) -> List[float]:
        samples: List[float] = []
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = 0.0
        for _ in range(num_samples):
            white = _gaussian_noise(0, 1.0)
            b0 = 0.99886 * b0 + white * 0.0555179
            b1 = 0.99332 * b1 + white * 0.0750759
            b2 = 0.96900 * b2 + white * 0.1538520
            b3 = 0.86650 * b3 + white * 0.3104856
            b4 = 0.55000 * b4 + white * 0.5329522
            b5 = -0.7616 * b5 - white * 0.0168980
            pink = b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362
            b6 = white * 0.115926
            samples.append(_clamp(amplitude * pink * 0.11, -amplitude, amplitude))
        return samples

    def _brown_noise(
        self, num_samples: int, amplitude: float
    ) -> List[float]:
        samples: List[float] = []
        last = 0.0
        for _ in range(num_samples):
            white = _gaussian_noise(0, 1.0)
            brown = (last + 0.02 * white) / 1.02
            last = brown
            samples.append(_clamp(amplitude * brown * 3.5, -amplitude, amplitude))
        return samples

    def generate_with_harmonics(
        self,
        waveform: WaveformType,
        frequency: float,
        duration: float,
        amplitude: float,
        num_harmonics: int,
        harmonic_decay: float,
    ) -> List[float]:
        """生成带谐波叠加的波形"""
        base = self.generate(waveform, frequency, duration, amplitude)
        if num_harmonics <= 1:
            return base

        result = list(base)
        for h in range(2, num_harmonics + 1):
            harmonic_freq = frequency * h
            harmonic_amp = amplitude * (harmonic_decay ** (h - 1))
            harmonic = self.generate(
                WaveformType.SINE, harmonic_freq, duration, harmonic_amp
            )
            for i in range(min(len(result), len(harmonic))):
                result[i] += harmonic[i]

        # 归一化
        max_val = max(abs(s) for s in result) if result else 1.0
        if max_val > 0:
            result = [s / max_val * amplitude for s in result]

        return result


# ============================================================================
# EnvelopeShaper - 包络整形器
# ============================================================================

class EnvelopeShaper:
    """
    包络整形器：控制音效的幅度包络。

    支持ADSR、指数、线性和打击乐包络。
    """

    def __init__(self, sample_rate: int = 44100):
        self._sample_rate = sample_rate

    def apply_adsr(
        self,
        samples: List[float],
        attack: float,
        decay: float,
        sustain_level: float,
        release: float,
    ) -> List[float]:
        """应用ADSR包络"""
        if not samples:
            return samples

        n = len(samples)
        result: List[float] = []
        attack_samples = int(attack * self._sample_rate)
        decay_samples = int(decay * self._sample_rate)
        release_samples = int(release * self._sample_rate)
        sustain_samples = n - attack_samples - decay_samples - release_samples

        for i in range(n):
            if i < attack_samples:
                env = i / max(attack_samples, 1)
            elif i < attack_samples + decay_samples:
                t = (i - attack_samples) / max(decay_samples, 1)
                env = 1.0 - (1.0 - sustain_level) * t
            elif i < attack_samples + decay_samples + sustain_samples:
                env = sustain_level
            else:
                t = (i - attack_samples - decay_samples - sustain_samples) / max(release_samples, 1)
                env = sustain_level * (1.0 - t)

            result.append(samples[i] * env)

        return result

    def apply_exponential_decay(
        self, samples: List[float], decay_rate: float = 5.0
    ) -> List[float]:
        """应用指数衰减"""
        if not samples:
            return samples

        n = len(samples)
        result: List[float] = []
        for i in range(n):
            t = i / self._sample_rate
            env = math.exp(-decay_rate * t)
            result.append(samples[i] * env)
        return result

    def apply_percussive(
        self, samples: List[float], attack: float = 0.001,
        decay: float = 0.3,
    ) -> List[float]:
        """应用打击乐包络"""
        return self.apply_adsr(samples, attack, decay, 0.0, decay * 0.5)

    def apply_reverse(
        self, samples: List[float]
    ) -> List[float]:
        """应用反向包络"""
        if not samples:
            return samples
        reversed_env = list(reversed(samples))
        n = len(samples)
        result: List[float] = []
        for i in range(n):
            result.append(samples[i] * abs(reversed_env[i]))
        return result

    def apply_envelope(
        self, samples: List[float], params: SynthParams
    ) -> List[float]:
        """根据参数应用包络"""
        if params.envelope_type == EnvelopeType.ADSR:
            return self.apply_adsr(
                samples, params.attack, params.decay,
                params.sustain, params.release,
            )
        elif params.envelope_type == EnvelopeType.EXPONENTIAL:
            return self.apply_exponential_decay(samples)
        elif params.envelope_type == EnvelopeType.PERCUSSIVE:
            return self.apply_percussive(samples, params.attack)
        elif params.envelope_type == EnvelopeType.REVERSE:
            return self.apply_reverse(samples)
        else:
            return self.apply_adsr(
                samples, params.attack, params.decay,
                params.sustain, params.release,
            )


# ============================================================================
# EffectChain - 效果链
# ============================================================================

class EffectChain:
    """
    效果链：串联多个音频效果。

    支持的效果:
    - 低通/高通滤波器
    - 延迟
    - 混响
    - 失真
    - 颤音
    - 镶边
    """

    def __init__(self, sample_rate: int = 44100):
        self._sample_rate = sample_rate
        self._effects: List[Tuple[str, Dict[str, float]]] = []

    def add_effect(self, effect_type: str, params: Dict[str, float]) -> None:
        """添加效果"""
        self._effects.append((effect_type, params))

    def clear(self) -> None:
        """清除所有效果"""
        self._effects.clear()

    def process(self, samples: List[float]) -> List[float]:
        """处理效果链"""
        result = list(samples)
        for effect_type, params in self._effects:
            result = self._apply_effect(result, effect_type, params)
        return result

    def _apply_effect(
        self, samples: List[float], effect_type: str, params: Dict[str, float]
    ) -> List[float]:
        """应用单个效果"""
        if effect_type == "lowpass":
            return self._lowpass(samples, params.get("cutoff", 1000.0))
        elif effect_type == "highpass":
            return self._highpass(samples, params.get("cutoff", 200.0))
        elif effect_type == "delay":
            return self._delay(
                samples, params.get("time", 0.3), params.get("feedback", 0.4),
                params.get("mix", 0.3),
            )
        elif effect_type == "reverb":
            return self._reverb(
                samples, params.get("room_size", 0.5),
                params.get("damping", 0.5), params.get("mix", 0.3),
            )
        elif effect_type == "distortion":
            return self._distortion(samples, params.get("drive", 2.0))
        elif effect_type == "tremolo":
            return self._tremolo(
                samples, params.get("rate", 5.0), params.get("depth", 0.5),
            )
        elif effect_type == "flanger":
            return self._flanger(
                samples, params.get("rate", 0.5), params.get("depth", 0.003),
            )
        elif effect_type == "chorus":
            return self._chorus(
                samples, params.get("rate", 1.5), params.get("depth", 0.005),
                params.get("mix", 0.5),
            )
        return samples

    def _lowpass(
        self, samples: List[float], cutoff: float
    ) -> List[float]:
        """简单一阶低通滤波器"""
        if not samples:
            return samples
        rc = 1.0 / (2.0 * math.pi * cutoff)
        dt = 1.0 / self._sample_rate
        alpha = dt / (rc + dt)
        result: List[float] = [samples[0]]
        for i in range(1, len(samples)):
            result.append(alpha * samples[i] + (1 - alpha) * result[-1])
        return result

    def _highpass(
        self, samples: List[float], cutoff: float
    ) -> List[float]:
        """简单一阶高通滤波器"""
        if not samples:
            return samples
        rc = 1.0 / (2.0 * math.pi * cutoff)
        dt = 1.0 / self._sample_rate
        alpha = rc / (rc + dt)
        result: List[float] = [samples[0]]
        for i in range(1, len(samples)):
            result.append(alpha * (result[-1] + samples[i] - samples[i - 1]))
        return result

    def _delay(
        self, samples: List[float], time: float,
        feedback: float, mix: float,
    ) -> List[float]:
        """延迟效果"""
        if not samples:
            return samples
        delay_samples = int(time * self._sample_rate)
        result = list(samples)
        n = len(result)
        for i in range(delay_samples, n):
            delayed = result[i - delay_samples]
            result[i] = result[i] + delayed * feedback * mix
        # 归一化
        max_val = max(abs(s) for s in result) if result else 1.0
        if max_val > 1.0:
            result = [s / max_val for s in result]
        return result

    def _reverb(
        self, samples: List[float], room_size: float,
        damping: float, mix: float,
    ) -> List[float]:
        """混响效果"""
        if not samples:
            return samples
        delays = [
            int(room_size * 3000 * (1 + i * 0.37)),
            int(room_size * 5000 * (1 + i * 0.23)),
            int(room_size * 7000 * (1 + i * 0.17)),
        ]
        feedback = 0.4 * (1.0 - damping * 0.5)
        result = list(samples)
        n = len(result)

        for delay in delays:
            if delay >= n:
                continue
            for i in range(delay, n):
                result[i] += result[i - delay] * feedback

        final = [_lerp(samples[i], result[i], mix) for i in range(n)]
        max_val = max(abs(s) for s in final) if final else 1.0
        if max_val > 1.0:
            final = [s / max_val for s in final]
        return final

    def _distortion(
        self, samples: List[float], drive: float
    ) -> List[float]:
        """失真效果"""
        return [_clamp(math.tanh(s * drive), -1.0, 1.0) for s in samples]

    def _tremolo(
        self, samples: List[float], rate: float, depth: float
    ) -> List[float]:
        """颤音效果"""
        result: List[float] = []
        for i, s in enumerate(samples):
            t = i / self._sample_rate
            mod = 1.0 - depth * 0.5 * (1.0 + math.sin(2.0 * math.pi * rate * t))
            result.append(s * mod)
        return result

    def _flanger(
        self, samples: List[float], rate: float, depth: float
    ) -> List[float]:
        """镶边效果"""
        if not samples:
            return samples
        max_delay = int(depth * self._sample_rate)
        result: List[float] = []
        for i in range(len(samples)):
            t = i / self._sample_rate
            delay = int(max_delay * 0.5 * (1.0 + math.sin(2.0 * math.pi * rate * t)))
            if i >= delay:
                result.append((samples[i] + samples[i - delay]) * 0.5)
            else:
                result.append(samples[i])
        return result

    def _chorus(
        self, samples: List[float], rate: float,
        depth: float, mix: float,
    ) -> List[float]:
        """合唱效果"""
        if not samples:
            return samples
        max_delay = int(depth * self._sample_rate)
        result: List[float] = []
        for i in range(len(samples)):
            t = i / self._sample_rate
            delay = int(max_delay * 0.5 * (1.0 + math.sin(2.0 * math.pi * rate * t)))
            delayed = samples[max(0, i - delay)]
            result.append(_lerp(samples[i], delayed, mix))
        return result


# ============================================================================
# TextToParams - 文本到参数映射
# ============================================================================

class TextToParams:
    """
    文本到参数映射器：将文本描述转换为合成参数。

    使用关键词匹配和启发式规则。
    """

    def __init__(self):
        self._sound_presets: Dict[str, SynthParams] = {
            "explosion": SynthParams(
                waveform=WaveformType.NOISE_WHITE,
                frequency=100.0, frequency_end=20.0,
                amplitude=1.0, duration=1.5,
                attack=0.001, decay=0.5, sustain=0.0, release=1.0,
                envelope_type=EnvelopeType.EXPONENTIAL,
            ),
            "laser": SynthParams(
                waveform=WaveformType.SINE,
                frequency=2000.0, frequency_end=200.0,
                amplitude=0.7, duration=0.5,
                attack=0.001, decay=0.3, sustain=0.0, release=0.2,
                envelope_type=EnvelopeType.EXPONENTIAL,
            ),
            "beep": SynthParams(
                waveform=WaveformType.SINE,
                frequency=880.0, amplitude=0.5, duration=0.2,
                attack=0.005, decay=0.05, sustain=0.8, release=0.1,
            ),
            "click": SynthParams(
                waveform=WaveformType.NOISE_WHITE,
                frequency=1000.0, amplitude=0.6, duration=0.05,
                attack=0.001, decay=0.04, sustain=0.0, release=0.01,
                envelope_type=EnvelopeType.PERCUSSIVE,
            ),
            "whoosh": SynthParams(
                waveform=WaveformType.NOISE_PINK,
                frequency=500.0, frequency_end=2000.0,
                amplitude=0.6, duration=0.8,
                attack=0.05, decay=0.3, sustain=0.0, release=0.4,
                noise_mix=0.8,
            ),
            "thunder": SynthParams(
                waveform=WaveformType.NOISE_BROWN,
                frequency=50.0, amplitude=1.0, duration=3.0,
                attack=0.01, decay=1.0, sustain=0.3, release=1.5,
                noise_mix=0.9,
            ),
            "bell": SynthParams(
                waveform=WaveformType.SINE,
                frequency=800.0, amplitude=0.6, duration=2.0,
                attack=0.001, decay=0.5, sustain=0.2, release=1.0,
                harmonics=6, harmonic_decay=0.4,
            ),
            "siren": SynthParams(
                waveform=WaveformType.SINE,
                frequency=400.0, frequency_end=800.0,
                amplitude=0.5, duration=2.0,
                attack=0.1, decay=0.0, sustain=0.8, release=0.3,
                vibrato_rate=3.0, vibrato_depth=200.0,
            ),
            "rain": SynthParams(
                waveform=WaveformType.NOISE_PINK,
                frequency=2000.0, amplitude=0.3, duration=5.0,
                attack=0.5, decay=0.0, sustain=0.8, release=1.0,
                noise_mix=1.0,
            ),
            "footstep": SynthParams(
                waveform=WaveformType.NOISE_BROWN,
                frequency=200.0, amplitude=0.4, duration=0.15,
                attack=0.001, decay=0.1, sustain=0.0, release=0.05,
                envelope_type=EnvelopeType.PERCUSSIVE,
            ),
            "splash": SynthParams(
                waveform=WaveformType.NOISE_WHITE,
                frequency=3000.0, frequency_end=500.0,
                amplitude=0.7, duration=0.8,
                attack=0.001, decay=0.5, sustain=0.0, release=0.3,
                envelope_type=EnvelopeType.EXPONENTIAL,
            ),
            "alarm": SynthParams(
                waveform=WaveformType.SQUARE,
                frequency=600.0, frequency_end=900.0,
                amplitude=0.5, duration=1.0,
                attack=0.01, decay=0.0, sustain=0.9, release=0.1,
                vibrato_rate=4.0, vibrato_depth=300.0,
            ),
        }

        self._category_keywords: Dict[str, List[str]] = {
            "impact": ["hit", "punch", "slam", "crash", "bang", "smash", "break"],
            "whoosh": ["whoosh", "swish", "swoosh", "wind", "rush"],
            "alarm": ["alarm", "alert", "warning", "siren", "beep"],
            "nature": ["rain", "thunder", "wind", "water", "ocean", "fire"],
            "mechanical": ["engine", "motor", "gear", "click", "switch", "button"],
            "sci-fi": ["laser", "blaster", "phaser", "teleport", "alien"],
            "ui": ["click", "pop", "ding", "notification", "success", "error"],
        }

    def parse(self, text: str) -> SynthParams:
        """将文本描述解析为合成参数"""
        text_lower = text.lower()

        # 检查预设匹配
        for preset_name, params in self._sound_presets.items():
            if preset_name in text_lower:
                return self._adjust_from_text(params, text_lower)

        # 基于关键词推断参数
        return self._infer_params(text_lower)

    def _adjust_from_text(
        self, base: SynthParams, text: str
    ) -> SynthParams:
        """根据文本调整预设参数"""
        params = SynthParams(
            waveform=base.waveform,
            frequency=base.frequency,
            frequency_end=base.frequency_end,
            amplitude=base.amplitude,
            duration=base.duration,
            attack=base.attack,
            decay=base.decay,
            sustain=base.sustain,
            release=base.release,
            envelope_type=base.envelope_type,
            harmonics=base.harmonics,
            harmonic_decay=base.harmonic_decay,
            noise_mix=base.noise_mix,
        )

        # 调整时长
        if "long" in text or "slow" in text:
            params.duration *= 2.0
        elif "short" in text or "quick" in text or "fast" in text:
            params.duration *= 0.5

        # 调整音高
        if "low" in text or "deep" in text or "bass" in text:
            params.frequency *= 0.5
            params.frequency_end *= 0.5
        elif "high" in text or "bright" in text or "sharp" in text:
            params.frequency *= 2.0
            params.frequency_end *= 2.0

        # 调整音量
        if "loud" in text or "strong" in text:
            params.amplitude = min(1.0, params.amplitude * 1.3)
        elif "soft" in text or "quiet" in text or "gentle" in text:
            params.amplitude *= 0.6

        return params

    def _infer_params(self, text: str) -> SynthParams:
        """从文本推断参数"""
        params = SynthParams()

        # 检测类别
        for category, keywords in self._category_keywords.items():
            if any(kw in text for kw in keywords):
                if category == "impact":
                    params.waveform = WaveformType.NOISE_WHITE
                    params.frequency = 100.0
                    params.duration = 0.5
                    params.envelope_type = EnvelopeType.PERCUSSIVE
                elif category == "whoosh":
                    params.waveform = WaveformType.NOISE_PINK
                    params.frequency = 800.0
                    params.frequency_end = 2000.0
                    params.duration = 0.6
                elif category == "alarm":
                    params.waveform = WaveformType.SQUARE
                    params.frequency = 700.0
                    params.duration = 1.0
                elif category == "nature":
                    params.waveform = WaveformType.NOISE_PINK
                    params.duration = 3.0
                    params.noise_mix = 0.8
                elif category == "mechanical":
                    params.waveform = WaveformType.SQUARE
                    params.frequency = 200.0
                    params.duration = 0.2
                elif category == "sci-fi":
                    params.waveform = WaveformType.SINE
                    params.frequency = 1500.0
                    params.frequency_end = 200.0
                    params.duration = 0.8
                elif category == "ui":
                    params.waveform = WaveformType.SINE
                    params.frequency = 1000.0
                    params.duration = 0.15
                break

        return self._adjust_from_text(params, text)


# ============================================================================
# SoundLibrary - 音效库
# ============================================================================

class SoundLibrary:
    """音效库：管理和检索预定义音效"""

    def __init__(self):
        self._sounds: Dict[str, SoundEffect] = {}
        self._categories: Dict[str, List[str]] = defaultdict(list)

    def add(self, sound: SoundEffect) -> None:
        """添加音效"""
        self._sounds[sound.name] = sound
        self._categories[sound.category].append(sound.name)
        for tag in sound.tags:
            if tag not in self._categories:
                self._categories[tag] = []
            self._categories[tag].append(sound.name)

    def get(self, name: str) -> Optional[SoundEffect]:
        """获取音效"""
        return self._sounds.get(name)

    def search(
        self, query: str, category: Optional[str] = None
    ) -> List[SoundEffect]:
        """搜索音效"""
        query_lower = query.lower()
        results: List[SoundEffect] = []

        for name, sound in self._sounds.items():
            if category and sound.category != category:
                continue
            if query_lower in name.lower() or any(
                query_lower in tag for tag in sound.tags
            ):
                results.append(sound)

        return results

    def list_categories(self) -> List[str]:
        return list(self._categories.keys())

    def list_names(self) -> List[str]:
        return list(self._sounds.keys())


# ============================================================================
# SoundEffectSynth - 音效合成器（主入口）
# ============================================================================

class SoundEffectSynth:
    """
    音效合成器：从文本描述生成音效。

    流程:
    1. 文本到参数映射
    2. 波形生成
    3. 频率扫描（如果需要）
    4. 包络整形
    5. 效果链处理
    6. 输出音效

    使用方法:
        synth = SoundEffectSynth()
        effect = synth.generate("explosion")
    """

    def __init__(self, sample_rate: int = 44100):
        self._sample_rate = sample_rate
        self._waveform_gen = WaveformGenerator(sample_rate)
        self._envelope_shaper = EnvelopeShaper(sample_rate)
        self._effect_chain = EffectChain(sample_rate)
        self._text_to_params = TextToParams()
        self._library = SoundLibrary()
        self._init_builtin_library()

    def _init_builtin_library(self) -> None:
        """初始化内置音效库"""
        builtin_sounds = [
            ("explosion", "impact", ["boom", "blast", "bang"]),
            ("laser", "sci-fi", ["beam", "zap", "pew"]),
            ("beep", "ui", ["tone", "ping", "ding"]),
            ("click", "ui", ["button", "tap", "mouse"]),
            ("whoosh", "whoosh", ["swish", "swoosh"]),
            ("thunder", "nature", ["storm", "rain"]),
            ("bell", "ui", ["chime", "ring", "notification"]),
            ("siren", "alarm", ["alarm", "warning", "emergency"]),
            ("rain", "nature", ["drizzle", "shower"]),
            ("footstep", "mechanical", ["step", "walk"]),
        ]

        for name, category, tags in builtin_sounds:
            params = self._text_to_params.parse(name)
            sound = self._synthesize_params(params)
            sound.name = name
            sound.category = category
            sound.tags = tags
            self._library.add(sound)

    def generate(
        self, text: str, duration: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> SoundEffect:
        """从文本描述生成音效"""
        if seed is not None:
            random.seed(seed)

        params = self._text_to_params.parse(text)

        if duration is not None:
            params.duration = duration

        sound = self._synthesize_params(params)
        sound.name = text
        sound.params = params
        sound.metadata["generation_id"] = _generate_id()

        return sound

    def _synthesize_params(self, params: SynthParams) -> SoundEffect:
        """根据参数合成音效"""
        # 步骤1: 波形生成
        if params.harmonics > 1:
            samples = self._waveform_gen.generate_with_harmonics(
                params.waveform, params.frequency, params.duration,
                params.amplitude, params.harmonics, params.harmonic_decay,
            )
        else:
            samples = self._waveform_gen.generate(
                params.waveform, params.frequency, params.duration,
                params.amplitude,
            )

        # 步骤2: 频率扫描（如果起始和结束频率不同）
        if abs(params.frequency - params.frequency_end) > 1.0:
            samples = self._apply_frequency_sweep(
                samples, params.frequency, params.frequency_end,
                params.duration,
            )

        # 步骤3: 颤音
        if params.vibrato_rate > 0:
            samples = self._apply_vibrato(
                samples, params.vibrato_rate, params.vibrato_depth,
            )

        # 步骤4: 噪声混合
        if params.noise_mix > 0:
            noise = self._waveform_gen.generate(
                WaveformType.NOISE_WHITE, 1000.0, params.duration,
                params.amplitude * params.noise_mix,
            )
            for i in range(min(len(samples), len(noise))):
                samples[i] = _lerp(samples[i], noise[i], params.noise_mix)

        # 步骤5: 包络整形
        samples = self._envelope_shaper.apply_envelope(samples, params)

        # 步骤6: 淡入淡出
        fade = min(int(self._sample_rate * 0.005), len(samples) // 4)
        if fade > 0:
            for i in range(fade):
                samples[i] *= i / fade
                samples[-(i + 1)] *= i / fade

        # 归一化
        if samples:
            max_val = max(abs(s) for s in samples)
            if max_val > 0:
                samples = [s / max_val * 0.95 for s in samples]

        return SoundEffect(
            samples=samples,
            sample_rate=self._sample_rate,
            duration=len(samples) / self._sample_rate,
        )

    def _apply_frequency_sweep(
        self, samples: List[float], freq_start: float,
        freq_end: float, duration: float,
    ) -> List[float]:
        """应用频率扫描"""
        result: List[float] = []
        n = len(samples)
        for i in range(n):
            t = i / max(n, 1)
            freq = _lerp(freq_start, freq_end, t)
            phase = 2.0 * math.pi * freq * i / self._sample_rate
            env = abs(samples[i]) / max(abs(s) for s in samples) if any(samples) else 1.0
            result.append(env * math.sin(phase))
        return result

    def _apply_vibrato(
        self, samples: List[float], rate: float, depth: float
    ) -> List[float]:
        """应用颤音"""
        result: List[float] = []
        for i, s in enumerate(samples):
            t = i / self._sample_rate
            mod = 1.0 + depth / 440.0 * math.sin(2.0 * math.pi * rate * t)
            result.append(s * mod)
        return result

    def search_library(self, query: str) -> List[SoundEffect]:
        """搜索音效库"""
        return self._library.search(query)

    def list_library(self) -> List[str]:
        """列出音效库"""
        return self._library.list_names()
