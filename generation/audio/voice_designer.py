"""
Voice Designer - 语音设计器

本模块实现了语音设计系统，包含音高变换、共振峰操控、音色修改、
人声效果处理和语音配置管理功能。仅使用标准库，不依赖外部库。
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


def _hanning_window(n: int) -> List[float]:
    if n <= 1:
        return [1.0]
    return [0.5 * (1.0 - math.cos(2.0 * math.pi * i / (n - 1))) for i in range(n)]


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class VoiceProfile:
    """语音配置"""
    profile_id: str = ""
    name: str = ""
    pitch_shift: float = 0.0
    formant_shift: float = 0.0
    timbre_brightness: float = 0.0
    timbre_warmth: float = 0.0
    timbre_breathiness: float = 0.0
    reverb_mix: float = 0.0
    reverb_room_size: float = 0.3
    chorus_mix: float = 0.0
    chorus_rate: float = 1.5
    echo_delay: float = 0.0
    echo_feedback: float = 0.0
    echo_mix: float = 0.0
    compression_threshold: float = 0.5
    compression_ratio: float = 2.0
    eq_low: float = 0.0
    eq_mid: float = 0.0
    eq_high: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VoiceDesignResult:
    """语音设计结果"""
    result_id: str = ""
    samples: List[float] = field(default_factory=list)
    sample_rate: int = 44100
    duration: float = 0.0
    profile_used: Optional[VoiceProfile] = None
    processing_steps: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# PitchShifter - 音高变换器
# ============================================================================

class PitchShifter:
    """
    音高变换器：改变音频的音高而不改变时长。

    使用相位声码器（Phase Vocoder）算法。
    """

    def __init__(self, sample_rate: int = 44100):
        self._sample_rate = sample_rate
        self._fft_size = 2048
        self._hop_size = 512

    def shift(
        self, samples: List[float], semitones: float
    ) -> List[float]:
        """
        音高变换。

        Args:
            samples: 输入音频
            semitones: 变换半音数（正数=升高，负数=降低）
        """
        if not samples or abs(semitones) < 0.01:
            return list(samples)

        shift_factor = 2.0 ** (semitones / 12.0)

        # 相位声码器方法
        # 步骤1: 拉伸（改变时长）
        stretched = self._time_stretch(samples, shift_factor)

        # 步骤2: 重采样（恢复原始时长）
        result = self._resample(stretched, shift_factor)

        return result

    def _time_stretch(
        self, samples: List[float], factor: float
    ) -> List[float]:
        """时间拉伸（OLA方法）"""
        if factor <= 0 or not samples:
            return list(samples)

        hop_in = self._hop_size
        hop_out = int(hop_in * factor)
        window = _hanning_window(self._fft_size)
        n = len(samples)

        output_len = int(n / hop_in * hop_out) + self._fft_size
        output: List[float] = [0.0] * output_len
        window_sum: List[float] = [0.0] * output_len

        pos = 0
        while pos + self._fft_size <= n:
            # 提取帧
            frame = samples[pos:pos + self._fft_size]
            # 加窗
            windowed = [frame[i] * window[i] for i in range(len(frame))]

            # 叠加
            out_pos = int(pos / hop_in * hop_out)
            for i in range(len(windowed)):
                if out_pos + i < output_len:
                    output[out_pos + i] += windowed[i]
                    window_sum[out_pos + i] += window[i] * window[i]

            pos += hop_in

        # 归一化
        for i in range(output_len):
            if window_sum[i] > 1e-8:
                output[i] /= window_sum[i]

        return output[:int(len(samples) * factor)]

    def _resample(
        self, samples: List[float], factor: float
    ) -> List[float]:
        """重采样（线性插值）"""
        if not samples or factor <= 0:
            return list(samples)

        target_len = int(len(samples) / factor)
        result: List[float] = []

        for i in range(target_len):
            src_pos = i * factor
            src_idx = int(src_pos)
            frac = src_pos - src_idx

            if src_idx + 1 < len(samples):
                val = _lerp(samples[src_idx], samples[src_idx + 1], frac)
            elif src_idx < len(samples):
                val = samples[src_idx]
            else:
                val = 0.0
            result.append(val)

        return result

    def detect_pitch(
        self, samples: List[float], frame_size: int = 1024
    ) -> List[float]:
        """检测基频轮廓"""
        if not samples:
            return []

        pitches: List[float] = []
        for i in range(0, len(samples) - frame_size, frame_size // 2):
            frame = samples[i:i + frame_size]
            pitch = self._autocorrelation_pitch(frame)
            pitches.append(pitch)

        return pitches

    def _autocorrelation_pitch(
        self, frame: List[float]
    ) -> float:
        """自相关基频检测"""
        n = len(frame)
        min_lag = int(self._sample_rate / 500)
        max_lag = int(self._sample_rate / 80)

        if max_lag >= n:
            return 0.0

        best_lag = min_lag
        best_corr = -1.0

        for lag in range(min_lag, min(max_lag, n)):
            corr = sum(frame[i] * frame[i + lag] for i in range(n - lag))
            norm = math.sqrt(
                sum(frame[i] ** 2 for i in range(n - lag))
                * sum(frame[i + lag] ** 2 for i in range(n - lag))
            )
            if norm > 0:
                corr /= norm
            if corr > best_corr:
                best_corr = corr
                best_lag = lag

        if best_corr < 0.3:
            return 0.0

        return self._sample_rate / best_lag


# ============================================================================
# FormantManipulator - 共振峰操控器
# ============================================================================

class FormantManipulator:
    """
    共振峰操控器：改变语音的共振峰频率。

    共振峰是语音频谱中的峰值，决定了元音的音色特征。
    """

    def __init__(self, sample_rate: int = 44100):
        self._sample_rate = sample_rate
        self._vowel_formants: Dict[str, Tuple[float, float, float]] = {
            "ah": (730, 1090, 2440),
            "aa": (570, 840, 2410),
            "eh": (530, 1840, 2480),
            "ee": (270, 2290, 3010),
            "oh": (570, 840, 2410),
            "oo": (300, 870, 2240),
            "er": (490, 1350, 1690),
        }

    def shift_formants(
        self, samples: List[float], shift_factor: float
    ) -> List[float]:
        """
        共振峰变换。

        shift_factor > 1.0: 共振峰上移（声音更细/更年轻）
        shift_factor < 1.0: 共振峰下移（声音更粗/更年长）
        """
        if not samples or abs(shift_factor - 1.0) < 0.01:
            return list(samples)

        # 使用重采样方法实现共振峰变换
        # 步骤1: 重采样（改变共振峰位置）
        resampled = self._resample(samples, shift_factor)

        # 步骤2: 恢复原始时长（保持音高不变）
        result = self._resample(resampled, 1.0 / shift_factor)

        return result

    def _resample(
        self, samples: List[float], factor: float
    ) -> List[float]:
        """重采样"""
        if not samples or factor <= 0:
            return list(samples)

        target_len = int(len(samples) * factor)
        if target_len == 0:
            return []

        result: List[float] = []
        for i in range(target_len):
            src_pos = i / factor
            src_idx = int(src_pos)
            frac = src_pos - src_idx

            if src_idx + 1 < len(samples):
                val = _lerp(samples[src_idx], samples[src_idx + 1], frac)
            elif src_idx < len(samples):
                val = samples[src_idx]
            else:
                val = 0.0
            result.append(val)

        return result

    def set_formant(
        self, samples: List[float], target_formants: Tuple[float, float, float]
    ) -> List[float]:
        """设置目标共振峰频率"""
        # 简化实现：通过带通滤波近似
        result = list(samples)
        f1, f2, f3 = target_formants

        # 增强共振峰频率附近的能量
        for formant_freq in (f1, f2, f3):
            bandwidth = formant_freq * 0.1
            result = self._resonance_boost(result, formant_freq, bandwidth)

        # 归一化
        if result:
            max_val = max(abs(s) for s in result)
            if max_val > 0:
                result = [s / max_val for s in result]

        return result

    def _resonance_boost(
        self, samples: List[float], freq: float, bandwidth: float
    ) -> List[float]:
        """在指定频率增强共振"""
        if not samples:
            return samples

        sr = self._sample_rate
        w = 2.0 * math.pi * freq / sr
        r = math.exp(-math.pi * bandwidth / sr)

        a1 = -2.0 * r * math.cos(w)
        a2 = r * r
        b0 = 1.0 - r

        result: List[float] = []
        x1 = x2 = y1 = y2 = 0.0

        for s in samples:
            y = b0 * s - a1 * y1 - a2 * y2
            result.append(y)
            x2, x1 = x1, s
            y2, y1 = y1, y

        return result

    def detect_formants(
        self, samples: List[float]
    ) -> Tuple[float, float, float]:
        """检测共振峰频率（简化LPC方法）"""
        if not samples:
            return (500.0, 1500.0, 2500.0)

        # 使用频谱峰值检测
        n = len(samples)
        frame_size = min(1024, n)
        frame = samples[:frame_size]

        # 计算功率谱
        spectrum: List[float] = []
        num_bins = 256
        for k in range(num_bins):
            freq = k * self._sample_rate / (2 * num_bins)
            real = sum(frame[i] * math.cos(2.0 * math.pi * k * i / frame_size)
                       for i in range(len(frame)))
            imag = sum(frame[i] * math.sin(2.0 * math.pi * k * i / frame_size)
                       for i in range(len(frame)))
            power = real * real + imag * imag
            spectrum.append(power)

        # 找前3个峰值
        peaks: List[Tuple[int, float]] = []
        for i in range(1, len(spectrum) - 1):
            if spectrum[i] > spectrum[i - 1] and spectrum[i] > spectrum[i + 1]:
                peaks.append((i, spectrum[i]))

        peaks.sort(key=lambda p: p[1], reverse=True)

        formants: List[float] = []
        for idx, _ in peaks[:3]:
            freq = idx * self._sample_rate / (2 * num_bins)
            if freq > 100:
                formants.append(freq)

        while len(formants) < 3:
            formants.append(500.0 + len(formants) * 1000.0)

        return (formants[0], formants[1], formants[2])


# ============================================================================
# TimbreModifier - 音色修改器
# ============================================================================

class TimbreModifier:
    """
    音色修改器：修改语音的音色特征。

    包括亮度、温暖度、气息感和频谱均衡。
    """

    def __init__(self, sample_rate: int = 44100):
        self._sample_rate = sample_rate

    def adjust_brightness(
        self, samples: List[float], amount: float
    ) -> List[float]:
        """
        调整亮度（高频能量）。

        amount > 0: 更亮
        amount < 0: 更暗
        """
        if not samples or abs(amount) < 0.01:
            return list(samples)

        result = list(samples)
        n = len(result)

        if amount > 0:
            # 增强高频
            for i in range(1, n):
                diff = result[i] - result[i - 1]
                result[i] += diff * amount * 0.3
        else:
            # 衰减高频（低通）
            rc = 1.0 / (2.0 * math.pi * (5000.0 + abs(amount) * 5000.0))
            dt = 1.0 / self._sample_rate
            alpha = dt / (rc + dt)
            for i in range(1, n):
                result[i] = alpha * result[i] + (1 - alpha) * result[i - 1]

        # 归一化
        max_val = max(abs(s) for s in result) if result else 1.0
        if max_val > 1.0:
            result = [s / max_val for s in result]

        return result

    def adjust_warmth(
        self, samples: List[float], amount: float
    ) -> List[float]:
        """
        调整温暖度（低频能量）。

        amount > 0: 更温暖
        amount < 0: 更冷
        """
        if not samples or abs(amount) < 0.01:
            return list(samples)

        result = list(samples)

        if amount > 0:
            # 增强低频
            window = 32
            for i in range(len(result)):
                start = max(0, i - window)
                end = min(len(result), i + window)
                avg = sum(result[start:end]) / (end - start)
                result[i] = _lerp(result[i], avg, amount * 0.3)
        else:
            # 衰减低频（高通）
            rc = 1.0 / (2.0 * math.pi * (200.0 + abs(amount) * 300.0))
            dt = 1.0 / self._sample_rate
            alpha = rc / (rc + dt)
            for i in range(1, len(result)):
                result[i] = alpha * (result[i - 1] + result[i] - result[i - 1])

        max_val = max(abs(s) for s in result) if result else 1.0
        if max_val > 1.0:
            result = [s / max_val for s in result]

        return result

    def adjust_breathiness(
        self, samples: List[float], amount: float
    ) -> List[float]:
        """调整气息感"""
        if not samples or amount < 0.01:
            return list(samples)

        result: List[float] = []
        for s in samples:
            noise = _gaussian_noise(0, abs(s) * amount * 0.5 + 0.001)
            result.append(_clamp(s + noise, -1.0, 1.0))

        return result

    def apply_equalizer(
        self, samples: List[float],
        low_gain: float = 0.0,
        mid_gain: float = 0.0,
        high_gain: float = 0.0,
    ) -> List[float]:
        """应用三段均衡器"""
        result = list(samples)

        # 低频
        if low_gain != 0:
            result = self.adjust_warmth(result, low_gain)

        # 高频
        if high_gain != 0:
            result = self.adjust_brightness(result, high_gain)

        # 中频（使用简单的峰值滤波近似）
        if mid_gain != 0:
            center_freq = 1000.0
            w = 2.0 * math.pi * center_freq / self._sample_rate
            r = math.exp(-math.pi * 200.0 / self._sample_rate)
            a1 = -2.0 * r * math.cos(w)
            a2 = r * r
            b0 = (1.0 - r) * (1.0 + mid_gain)

            filtered: List[float] = []
            y1 = y2 = 0.0
            for s in result:
                y = b0 * s - a1 * y1 - a2 * y2
                filtered.append(y)
                y2, y1 = y1, y
            result = filtered

        # 归一化
        if result:
            max_val = max(abs(s) for s in result)
            if max_val > 1.0:
                result = [s / max_val for s in result]

        return result


# ============================================================================
# VocalEffects - 人声效果处理器
# ============================================================================

class VocalEffects:
    """
    人声效果处理器：混响、合唱、回声等人声效果。
    """

    def __init__(self, sample_rate: int = 44100):
        self._sample_rate = sample_rate

    def apply_reverb(
        self,
        samples: List[float],
        room_size: float = 0.3,
        damping: float = 0.5,
        mix: float = 0.2,
    ) -> List[float]:
        """应用混响"""
        if not samples or mix < 0.01:
            return list(samples)

        delays = [
            int(room_size * 4000 * (1 + i * 0.37)),
            int(room_size * 6000 * (1 + i * 0.23)),
            int(room_size * 8000 * (1 + i * 0.17)),
        ]
        feedback = 0.35 * (1.0 - damping * 0.5)
        wet = list(samples)

        for delay in delays:
            if delay >= len(wet):
                continue
            for i in range(delay, len(wet)):
                wet[i] += wet[i - delay] * feedback

        result = [_lerp(samples[i], wet[i], mix) for i in range(len(samples))]

        max_val = max(abs(s) for s in result) if result else 1.0
        if max_val > 1.0:
            result = [s / max_val for s in result]

        return result

    def apply_chorus(
        self,
        samples: List[float],
        rate: float = 1.5,
        depth: float = 0.003,
        mix: float = 0.3,
    ) -> List[float]:
        """应用合唱效果"""
        if not samples or mix < 0.01:
            return list(samples)

        max_delay = int(depth * self._sample_rate)
        result: List[float] = []

        for i in range(len(samples)):
            t = i / self._sample_rate
            delay = int(max_delay * 0.5 * (1.0 + math.sin(2.0 * math.pi * rate * t)))
            delayed = samples[max(0, i - delay)]
            result.append(_lerp(samples[i], delayed, mix))

        return result

    def apply_echo(
        self,
        samples: List[float],
        delay_time: float = 0.3,
        feedback: float = 0.4,
        mix: float = 0.3,
    ) -> List[float]:
        """应用回声"""
        if not samples or mix < 0.01 or delay_time < 0.01:
            return list(samples)

        delay_samples = int(delay_time * self._sample_rate)
        result = list(samples)

        for i in range(delay_samples, len(result)):
            result[i] += result[i - delay_samples] * feedback * mix

        max_val = max(abs(s) for s in result) if result else 1.0
        if max_val > 1.0:
            result = [s / max_val for s in result]

        return result

    def apply_compression(
        self,
        samples: List[float],
        threshold: float = 0.5,
        ratio: float = 2.0,
    ) -> List[float]:
        """应用动态压缩"""
        if not samples:
            return samples

        result: List[float] = []
        envelope = 0.0
        attack = int(0.01 * self._sample_rate)
        release = int(0.1 * self._sample_rate)

        for i, s in enumerate(samples):
            abs_s = abs(s)
            if abs_s > envelope:
                envelope += (abs_s - envelope) / max(attack, 1)
            else:
                envelope += (abs_s - envelope) / max(release, 1)

            if envelope > threshold:
                gain = threshold + (envelope - threshold) / ratio
                gain /= max(envelope, 1e-10)
            else:
                gain = 1.0

            result.append(s * gain)

        return result

    def apply_de_ess(
        self, samples: List[float], threshold: float = 0.3,
        reduction: float = 0.5,
    ) -> List[float]:
        """去齿音"""
        if not samples:
            return samples

        # 高通滤波提取齿音频率
        sibilance: List[float] = []
        rc = 1.0 / (2.0 * math.pi * 6000.0)
        dt = 1.0 / self._sample_rate
        alpha = rc / (rc + dt)
        prev = 0.0
        for s in samples:
            high = s - alpha * prev - (1 - alpha) * s
            sibilance.append(high)
            prev = s

        # 压缩齿音
        result: List[float] = []
        for i in range(len(samples)):
            sib_level = abs(sibilance[i])
            if sib_level > threshold:
                gain = 1.0 - reduction * (sib_level - threshold) / sib_level
            else:
                gain = 1.0
            result.append(samples[i] * gain)

        return result


# ============================================================================
# ProfileManager - 配置管理器
# ============================================================================

class ProfileManager:
    """语音配置管理器"""

    def __init__(self):
        self._profiles: Dict[str, VoiceProfile] = {}
        self._init_presets()

    def _init_presets(self) -> None:
        """初始化预设配置"""
        presets = {
            "default": VoiceProfile(name="Default", profile_id="default"),
            "deep_voice": VoiceProfile(
                name="Deep Voice", profile_id="deep_voice",
                pitch_shift=-5.0, formant_shift=0.8,
                timbre_warmth=0.3, timbre_brightness=-0.2,
            ),
            "bright_voice": VoiceProfile(
                name="Bright Voice", profile_id="bright_voice",
                pitch_shift=3.0, formant_shift=1.2,
                timbre_brightness=0.3, timbre_warmth=-0.1,
            ),
            "robot": VoiceProfile(
                name="Robot", profile_id="robot",
                pitch_shift=0.0, formant_shift=0.0,
                timbre_brightness=0.5, timbre_breathiness=0.0,
                reverb_mix=0.3, reverb_room_size=0.5,
                eq_high=0.3, eq_low=-0.2,
            ),
            "radio": VoiceProfile(
                name="Radio", profile_id="radio",
                pitch_shift=1.0, formant_shift=1.1,
                timbre_brightness=0.2,
                compression_threshold=0.3, compression_ratio=4.0,
                eq_mid=0.2,
            ),
            "telephone": VoiceProfile(
                name="Telephone", profile_id="telephone",
                pitch_shift=0.0, formant_shift=1.0,
                eq_low=-0.5, eq_high=-0.3, eq_mid=0.3,
            ),
            "cave": VoiceProfile(
                name="Cave", profile_id="cave",
                reverb_mix=0.6, reverb_room_size=0.9,
                echo_delay=0.4, echo_feedback=0.5, echo_mix=0.3,
                eq_high=-0.3,
            ),
            "whisper": VoiceProfile(
                name="Whisper", profile_id="whisper",
                pitch_shift=-2.0, formant_shift=0.9,
                timbre_breathiness=0.8, timbre_brightness=-0.2,
            ),
        }
        self._profiles.update(presets)

    def save_profile(self, profile: VoiceProfile) -> str:
        """保存配置"""
        if not profile.profile_id:
            profile.profile_id = _generate_id()
        self._profiles[profile.profile_id] = profile
        return profile.profile_id

    def load_profile(self, profile_id: str) -> Optional[VoiceProfile]:
        """加载配置"""
        profile = self._profiles.get(profile_id)
        if profile:
            # 返回副本
            return VoiceProfile(
                profile_id=profile.profile_id,
                name=profile.name,
                pitch_shift=profile.pitch_shift,
                formant_shift=profile.formant_shift,
                timbre_brightness=profile.timbre_brightness,
                timbre_warmth=profile.timbre_warmth,
                timbre_breathiness=profile.timbre_breathiness,
                reverb_mix=profile.reverb_mix,
                reverb_room_size=profile.reverb_room_size,
                chorus_mix=profile.chorus_mix,
                chorus_rate=profile.chorus_rate,
                echo_delay=profile.echo_delay,
                echo_feedback=profile.echo_feedback,
                echo_mix=profile.echo_mix,
                compression_threshold=profile.compression_threshold,
                compression_ratio=profile.compression_ratio,
                eq_low=profile.eq_low,
                eq_mid=profile.eq_mid,
                eq_high=profile.eq_high,
            )
        return None

    def list_profiles(self) -> List[VoiceProfile]:
        return list(self._profiles.values())

    def delete_profile(self, profile_id: str) -> bool:
        if profile_id in self._profiles:
            del self._profiles[profile_id]
            return True
        return False

    def blend_profiles(
        self, id_a: str, id_b: str, blend: float = 0.5
    ) -> Optional[VoiceProfile]:
        """混合两个配置"""
        pa = self._profiles.get(id_a)
        pb = self._profiles.get(id_b)
        if not pa or not pb:
            return None

        blended = VoiceProfile(
            profile_id=_generate_id(),
            name=f"Blend({pa.name}+{pb.name})",
            pitch_shift=_lerp(pa.pitch_shift, pb.pitch_shift, blend),
            formant_shift=_lerp(pa.formant_shift, pb.formant_shift, blend),
            timbre_brightness=_lerp(pa.timbre_brightness, pb.timbre_brightness, blend),
            timbre_warmth=_lerp(pa.timbre_warmth, pb.timbre_warmth, blend),
            timbre_breathiness=_lerp(pa.timbre_breathiness, pb.timbre_breathiness, blend),
            reverb_mix=_lerp(pa.reverb_mix, pb.reverb_mix, blend),
            reverb_room_size=_lerp(pa.reverb_room_size, pb.reverb_room_size, blend),
            chorus_mix=_lerp(pa.chorus_mix, pb.chorus_mix, blend),
            echo_delay=_lerp(pa.echo_delay, pb.echo_delay, blend),
            echo_feedback=_lerp(pa.echo_feedback, pb.echo_feedback, blend),
            echo_mix=_lerp(pa.echo_mix, pb.echo_mix, blend),
            eq_low=_lerp(pa.eq_low, pb.eq_low, blend),
            eq_mid=_lerp(pa.eq_mid, pb.eq_mid, blend),
            eq_high=_lerp(pa.eq_high, pb.eq_high, blend),
        )
        self._profiles[blended.profile_id] = blended
        return blended


# ============================================================================
# VoiceDesigner - 语音设计器（主入口）
# ============================================================================

class VoiceDesigner:
    """
    语音设计器：完整的语音设计管线。

    流程:
    1. 加载语音配置
    2. 音高变换
    3. 共振峰操控
    4. 音色修改
    5. 效果处理
    6. 输出设计后的语音

    使用方法:
        designer = VoiceDesigner()
        result = designer.design(audio_samples, "deep_voice")
    """

    def __init__(self, sample_rate: int = 44100):
        self._sample_rate = sample_rate
        self._pitch_shifter = PitchShifter(sample_rate)
        self._formant_manip = FormantManipulator(sample_rate)
        self._timbre_mod = TimbreModifier(sample_rate)
        self._vocal_effects = VocalEffects(sample_rate)
        self._profile_manager = ProfileManager()

    @property
    def profile_manager(self) -> ProfileManager:
        return self._profile_manager

    def design(
        self,
        samples: List[float],
        profile_id: str = "default",
        custom_profile: Optional[VoiceProfile] = None,
    ) -> VoiceDesignResult:
        """
        设计语音。

        Args:
            samples: 输入音频
            profile_id: 语音配置ID
            custom_profile: 自定义配置（覆盖profile_id）

        Returns:
            VoiceDesignResult: 设计结果
        """
        profile = custom_profile or self._profile_manager.load_profile(profile_id)
        if not profile:
            profile = VoiceProfile(profile_id="default", name="Default")

        result = list(samples)
        steps: List[str] = []

        # 步骤1: 音高变换
        if abs(profile.pitch_shift) > 0.01:
            result = self._pitch_shifter.shift(result, profile.pitch_shift)
            steps.append(f"pitch_shift({profile.pitch_shift:.1f}st)")

        # 步骤2: 共振峰操控
        if abs(profile.formant_shift - 1.0) > 0.01:
            result = self._formant_manip.shift_formants(result, profile.formant_shift)
            steps.append(f"formant_shift({profile.formant_shift:.2f})")

        # 步骤3: 音色修改
        if abs(profile.timbre_brightness) > 0.01:
            result = self._timbre_mod.adjust_brightness(result, profile.timbre_brightness)
            steps.append(f"brightness({profile.timbre_brightness:.2f})")

        if abs(profile.timbre_warmth) > 0.01:
            result = self._timbre_mod.adjust_warmth(result, profile.timbre_warmth)
            steps.append(f"warmth({profile.timbre_warmth:.2f})")

        if profile.timbre_breathiness > 0.01:
            result = self._timbre_mod.adjust_breathiness(result, profile.timbre_breathiness)
            steps.append(f"breathiness({profile.timbre_breathiness:.2f})")

        # 步骤4: 均衡器
        if any(abs(v) > 0.01 for v in (profile.eq_low, profile.eq_mid, profile.eq_high)):
            result = self._timbre_mod.apply_equalizer(
                result, profile.eq_low, profile.eq_mid, profile.eq_high
            )
            steps.append(f"eq(low={profile.eq_low:.1f},mid={profile.eq_mid:.1f},high={profile.eq_high:.1f})")

        # 步骤5: 压缩
        if profile.compression_ratio > 1.0:
            result = self._vocal_effects.apply_compression(
                result, profile.compression_threshold, profile.compression_ratio
            )
            steps.append(f"compression(thresh={profile.compression_threshold:.1f},ratio={profile.compression_ratio:.1f})")

        # 步骤6: 混响
        if profile.reverb_mix > 0.01:
            result = self._vocal_effects.apply_reverb(
                result, profile.reverb_room_size, 0.5, profile.reverb_mix
            )
            steps.append(f"reverb(size={profile.reverb_room_size:.1f},mix={profile.reverb_mix:.1f})")

        # 步骤7: 合唱
        if profile.chorus_mix > 0.01:
            result = self._vocal_effects.apply_chorus(
                result, profile.chorus_rate, 0.003, profile.chorus_mix
            )
            steps.append(f"chorus(rate={profile.chorus_rate:.1f},mix={profile.chorus_mix:.1f})")

        # 步骤8: 回声
        if profile.echo_mix > 0.01 and profile.echo_delay > 0.01:
            result = self._vocal_effects.apply_echo(
                result, profile.echo_delay, profile.echo_feedback, profile.echo_mix
            )
            steps.append(f"echo(delay={profile.echo_delay:.2f},mix={profile.echo_mix:.1f})")

        # 步骤9: 去齿音
        result = self._vocal_effects.apply_de_ess(result)
        steps.append("de_ess")

        # 最终归一化
        if result:
            max_val = max(abs(s) for s in result)
            if max_val > 0:
                result = [s / max_val * 0.95 for s in result]

        duration = len(result) / self._sample_rate

        return VoiceDesignResult(
            result_id=_generate_id(),
            samples=result,
            sample_rate=self._sample_rate,
            duration=duration,
            profile_used=profile,
            processing_steps=steps,
            metadata={"generation_id": _generate_id()},
        )

    def quick_design(
        self,
        samples: List[float],
        pitch_semitones: float = 0.0,
        formant_shift: float = 1.0,
        brightness: float = 0.0,
        reverb_mix: float = 0.0,
    ) -> VoiceDesignResult:
        """快速设计（直接指定参数）"""
        profile = VoiceProfile(
            profile_id=_generate_id(),
            name="Quick Design",
            pitch_shift=pitch_semitones,
            formant_shift=formant_shift,
            timbre_brightness=brightness,
            reverb_mix=reverb_mix,
        )
        return self.design(samples, custom_profile=profile)
