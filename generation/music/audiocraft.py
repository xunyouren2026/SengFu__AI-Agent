"""
AudioCraft Audio Generation - AudioCraft音频生成框架

本模块实现了AudioCraft音频生成系统，包含MusicGen集成、音频提示编码、
多轨生成、流派/风格控制和混音/母带处理功能。仅使用标准库，
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


def _note_freq(note: str, octave: int = 4) -> float:
    """将音符名转换为频率"""
    note_map = {'C': -9, 'C#': -8, 'Db': -8, 'D': -7, 'D#': -6, 'Eb': -6,
                'E': -5, 'F': -4, 'F#': -3, 'Gb': -3, 'G': -2, 'G#': -1,
                'Ab': -1, 'A': 0, 'A#': 1, 'Bb': 1, 'B': 2}
    semitones = note_map.get(note, 0)
    return 440.0 * (2.0 ** ((semitones + (octave - 4) * 12) / 12.0))


def _hanning_window(n: int) -> List[float]:
    if n <= 1:
        return [1.0]
    return [0.5 * (1.0 - math.cos(2.0 * math.pi * i / (n - 1))) for i in range(n)]


def _fade_in_out(samples: List[float], fade_samples: int) -> List[float]:
    """应用淡入淡出"""
    result = list(samples)
    n = len(result)
    fade_in = min(fade_samples, n // 4)
    fade_out = min(fade_samples, n // 4)
    for i in range(fade_in):
        result[i] *= i / fade_in
    for i in range(fade_out):
        result[n - 1 - i] *= i / fade_out
    return result


# ============================================================================
# 数据结构
# ============================================================================

class MusicModel(Enum):
    """音乐模型类型"""
    MUSICGEN_SMALL = "musicgen_small"
    MUSICGEN_MEDIUM = "musicgen_medium"
    MUSICGEN_LARGE = "musicgen_large"
    MUSICGEN_MELODY = "musicgen_melody"
    AUDIOGEN = "audiogen"


@dataclass
class AudioCraftConfig:
    """AudioCraft配置"""
    sample_rate: int = 32000
    model: MusicModel = MusicModel.MUSICGEN_SMALL
    max_duration: float = 30.0
    temperature: float = 0.8
    top_k: int = 250
    top_p: float = 0.9
    classifier_free_guidance: float = 3.0
    num_output_channels: int = 1
    stereo: bool = False
    use_coarse_to_fine: bool = True
    num_coarse_quantizers: int = 4
    num_fine_quantizers: int = 8
    codebook_size: int = 2048
    num_transformer_layers: int = 6
    hidden_dim: int = 1024
    num_heads: int = 16


@dataclass
class AudioPrompt:
    """音频提示"""
    samples: List[float] = field(default_factory=list)
    sample_rate: int = 32000
    text_description: str = ""
    duration: float = 0.0
    embedding: List[float] = field(default_factory=list)


@dataclass
class Track:
    """音轨"""
    name: str = ""
    samples: List[float] = field(default_factory=list)
    sample_rate: int = 32000
    volume: float = 0.8
    pan: float = 0.0
    muted: bool = False
    solo: bool = False
    instrument: str = ""
    midi_notes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MusicOutput:
    """音乐输出"""
    samples: List[float] = field(default_factory=list)
    sample_rate: int = 32000
    duration: float = 0.0
    tracks: List[Track] = field(default_factory=list)
    bpm: int = 120
    key: str = "C"
    genre: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# AudioPromptEncoder - 音频提示编码器
# ============================================================================

class AudioPromptEncoder:
    """
    音频提示编码器：将文本描述和音频片段编码为生成条件。

    模拟EnCodec音频编码器的行为。
    """

    def __init__(self, config: AudioCraftConfig):
        self._config = config
        self._dim = config.hidden_dim

    def encode_text(self, text: str) -> List[float]:
        """将文本描述编码为条件向量"""
        if not text.strip():
            return [0.0] * self._dim

        # 基于文本特征生成嵌入
        words = text.lower().split()
        embedding = [0.0] * self._dim

        # 词袋特征
        for i, word in enumerate(words):
            idx = hash(word) % self._dim
            embedding[idx] += 1.0 / max(len(words), 1)

        # 位置编码
        for i in range(min(len(words), self._dim)):
            embedding[i] += math.sin(i / self._dim * math.pi * 2) * 0.1

        # 非线性变换
        mean = sum(embedding) / self._dim
        std = math.sqrt(sum((v - mean) ** 2 for v in embedding) / self._dim) or 1.0
        embedding = [(v - mean) / std for v in embedding]

        # GELU激活
        for i in range(len(embedding)):
            v = embedding[i]
            embedding[i] = 0.5 * v * (1.0 + math.tanh(
                math.sqrt(2.0 / math.pi) * (v + 0.044715 * v ** 3)
            ))

        return embedding

    def encode_audio(self, samples: List[float], sample_rate: int) -> List[float]:
        """将音频片段编码为条件向量"""
        if not samples:
            return [0.0] * self._dim

        # 计算音频特征
        n = len(samples)
        features: List[float] = []

        # 统计特征
        mean = sum(samples) / n
        std = math.sqrt(sum((s - mean) ** 2 for s in samples) / n)
        features.extend([mean, std])

        # 能量
        energy = sum(s * s for s in samples) / n
        features.append(math.log(energy + 1e-10))

        # 零交叉率
        zcr = sum(1 for i in range(1, n) if (samples[i] >= 0) != (samples[i - 1] >= 0)) / n
        features.append(zcr)

        # 频谱特征（简化）
        frame_size = min(1024, n)
        num_frames = max(1, n // frame_size)
        for f in range(min(32, num_frames)):
            start = f * frame_size
            frame = samples[start:start + frame_size]
            if frame:
                e = sum(s * s for s in frame) / len(frame)
                features.append(math.log(e + 1e-10))

        # 填充到目标维度
        while len(features) < self._dim:
            features.append(_gaussian_noise(0, 0.01))
        features = features[:self._dim]

        # 归一化
        f_mean = sum(features) / len(features)
        f_std = math.sqrt(sum((v - f_mean) ** 2 for v in features) / len(features)) or 1.0
        return [(v - f_mean) / f_std for v in features]

    def encode_combined(
        self, text: str, samples: Optional[List[float]] = None,
        sample_rate: int = 32000,
    ) -> List[float]:
        """组合文本和音频编码"""
        text_emb = self.encode_text(text)

        if samples:
            audio_emb = self.encode_audio(samples, sample_rate)
            combined = [
                0.6 * text_emb[i] + 0.4 * audio_emb[i]
                for i in range(self._dim)
            ]
        else:
            combined = list(text_emb)

        return combined


# ============================================================================
# MusicGenWrapper - MusicGen包装器
# ============================================================================

class MusicGenWrapper:
    """
    MusicGen包装器：模拟Meta MusicGen模型的推理过程。

    使用EnCodec和Transformer架构生成音乐。
    """

    def __init__(self, config: AudioCraftConfig):
        self._config = config
        self._prompt_encoder = AudioPromptEncoder(config)
        random.seed(42)
        self._transformer_weights = [
            [random.gauss(0, 0.02) for _ in range(config.hidden_dim)]
            for _ in range(config.hidden_dim)
        ]
        random.seed()

    def generate(
        self,
        prompt: AudioPrompt,
        duration: float = 10.0,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> List[float]:
        """生成音乐"""
        if seed is not None:
            random.seed(seed)

        cfg = guidance_scale or self._config.classifier_free_guidance

        # 编码提示
        condition = self._prompt_encoder.encode_combined(
            prompt.text_description, prompt.samples, prompt.sample_rate
        )

        # 模拟Transformer生成过程
        sr = self._config.sample_rate
        total_samples = int(duration * sr)

        # 生成粗粒度token
        coarse_tokens = self._generate_coarse_tokens(
            condition, total_samples, cfg
        )

        # 粗粒度到细粒度
        fine_tokens = self._coarse_to_fine(coarse_tokens, condition)

        # 解码为波形
        samples = self._decode_tokens(fine_tokens, total_samples)

        return samples

    def _generate_coarse_tokens(
        self,
        condition: List[float],
        total_samples: int,
        cfg: float,
    ) -> List[int]:
        """生成粗粒度token"""
        num_tokens = total_samples // 512
        tokens: List[int] = []

        for i in range(num_tokens):
            # 条件引导采样
            cond_score = sum(
                condition[j % len(condition)] * self._transformer_weights[i % len(self._transformer_weights)][j % self._config.hidden_dim]
                for j in range(min(64, len(condition)))
            )

            # 无条件分数
            uncond_score = _gaussian_noise(0, 1.0)

            # CFG
            score = uncond_score + cfg * (cond_score - uncond_score)

            # Top-k采样
            token = int(_clamp(score * 500 + self._config.codebook_size // 2,
                               0, self._config.codebook_size - 1))
            tokens.append(token)

        return tokens

    def _coarse_to_fine(
        self, coarse_tokens: List[int], condition: List[float]
    ) -> List[List[int]]:
        """粗粒度到细粒度转换"""
        fine_tokens: List[List[int]] = []

        for ct in coarse_tokens:
            layer_tokens: List[int] = []
            for q in range(self._config.num_fine_quantizers):
                # 每个量化器添加细节
                detail = (ct * (q + 1) * 7 + q * 13) % self._config.codebook_size
                noise = int(_gaussian_noise(0, 50))
                token = (detail + noise) % self._config.codebook_size
                layer_tokens.append(token)
            fine_tokens.append(layer_tokens)

        return fine_tokens

    def _decode_tokens(
        self, fine_tokens: List[List[int]], total_samples: int
    ) -> List[float]:
        """将token解码为波形"""
        samples: List[float] = []
        sr = self._config.sample_rate
        samples_per_token = total_samples / max(len(fine_tokens), 1)

        for i, token_layers in enumerate(fine_tokens):
            # 从token生成音频特征
            base_freq = 100.0 + (token_layers[0] % 400) * 0.5
            amplitude = 0.3 * (token_layers[1] % 100) / 100.0
            harmonic = 1.0 + (token_layers[2] % 50) / 100.0

            num_s = int(samples_per_token)
            for s in range(num_s):
                t = (i * samples_per_token + s) / sr
                phase = 2.0 * math.pi * base_freq * t
                sample = amplitude * math.sin(phase)
                sample += amplitude * 0.5 * math.sin(phase * harmonic)
                sample += amplitude * 0.25 * math.sin(phase * harmonic * 2)
                sample += _gaussian_noise(0, amplitude * 0.05)
                samples.append(_clamp(sample, -1.0, 1.0))

        # 截断到目标长度
        samples = samples[:total_samples]

        # 归一化
        if samples:
            max_val = max(abs(s) for s in samples)
            if max_val > 0:
                samples = [s / max_val * 0.9 for s in samples]

        return samples


# ============================================================================
# GenreController - 流派/风格控制器
# ============================================================================

class GenreController:
    """
    流派/风格控制器：控制生成音乐的流派和风格特征。

    内置多种音乐流派的特征参数。
    """

    def __init__(self):
        self._genres: Dict[str, Dict[str, Any]] = {
            "pop": {
                "bpm_range": (100, 130),
                "key_preferences": ["C", "G", "D", "A"],
                "instrument_weights": {"drums": 0.3, "bass": 0.25, "keys": 0.25, "vocals": 0.2},
                "energy": 0.7,
                "complexity": 0.4,
                "typical_chords": ["I", "V", "vi", "IV"],
            },
            "rock": {
                "bpm_range": (110, 140),
                "key_preferences": ["E", "A", "D", "G"],
                "instrument_weights": {"guitar": 0.35, "drums": 0.3, "bass": 0.25, "vocals": 0.1},
                "energy": 0.85,
                "complexity": 0.5,
                "typical_chords": ["I", "IV", "V", "I"],
            },
            "jazz": {
                "bpm_range": (80, 140),
                "key_preferences": ["Bb", "Eb", "F", "C"],
                "instrument_weights": {"piano": 0.3, "bass": 0.25, "drums": 0.2, "sax": 0.25},
                "energy": 0.5,
                "complexity": 0.9,
                "typical_chords": ["ii7", "V7", "Imaj7", "vi7"],
            },
            "classical": {
                "bpm_range": (60, 120),
                "key_preferences": ["C", "D", "G", "F"],
                "instrument_weights": {"strings": 0.4, "piano": 0.3, "winds": 0.2, "brass": 0.1},
                "energy": 0.5,
                "complexity": 0.8,
                "typical_chords": ["I", "IV", "V", "I"],
            },
            "electronic": {
                "bpm_range": (120, 150),
                "key_preferences": ["Am", "Em", "Cm", "Dm"],
                "instrument_weights": {"synth": 0.4, "drums": 0.35, "bass": 0.25},
                "energy": 0.8,
                "complexity": 0.6,
                "typical_chords": ["i", "VI", "III", "VII"],
            },
            "hiphop": {
                "bpm_range": (80, 100),
                "key_preferences": ["Cm", "Gm", "Dm", "Am"],
                "instrument_weights": {"drums": 0.35, "bass": 0.35, "synth": 0.2, "vocals": 0.1},
                "energy": 0.75,
                "complexity": 0.4,
                "typical_chords": ["i", "iv", "V", "i"],
            },
            "ambient": {
                "bpm_range": (60, 90),
                "key_preferences": ["C", "D", "E", "G"],
                "instrument_weights": {"pad": 0.5, "texture": 0.3, "bass": 0.2},
                "energy": 0.3,
                "complexity": 0.3,
                "typical_chords": ["I", "ii", "iii", "IV"],
            },
            "blues": {
                "bpm_range": (70, 110),
                "key_preferences": ["E", "A", "Bb", "G"],
                "instrument_weights": {"guitar": 0.4, "bass": 0.25, "drums": 0.2, "harmonica": 0.15},
                "energy": 0.6,
                "complexity": 0.5,
                "typical_chords": ["I7", "IV7", "I7", "V7"],
            },
        }

    def get_genre_params(self, genre: str) -> Dict[str, Any]:
        """获取流派参数"""
        genre_lower = genre.lower()
        for key in self._genres:
            if key in genre_lower or genre_lower in key:
                return dict(self._genres[key])
        return dict(self._genres["pop"])

    def list_genres(self) -> List[str]:
        return list(self._genres.keys())

    def generate_bpm(self, genre: str) -> int:
        """根据流派生成BPM"""
        params = self.get_genre_params(genre)
        lo, hi = params["bpm_range"]
        return random.randint(lo, hi)

    def select_key(self, genre: str) -> str:
        """根据流派选择调性"""
        params = self.get_genre_params(genre)
        prefs = params["key_preferences"]
        return random.choice(prefs)

    def get_instrument_mix(self, genre: str) -> Dict[str, float]:
        """获取乐器混合比例"""
        params = self.get_genre_params(genre)
        return dict(params["instrument_weights"])

    def blend_genres(
        self, genre_a: str, genre_b: str, blend: float = 0.5
    ) -> Dict[str, Any]:
        """混合两种流派"""
        params_a = self.get_genre_params(genre_a)
        params_b = self.get_genre_params(genre_b)

        blended: Dict[str, Any] = {}
        lo_a, hi_a = params_a["bpm_range"]
        lo_b, hi_b = params_b["bpm_range"]
        blended["bpm_range"] = (
            int(_lerp(lo_a, lo_b, blend)),
            int(_lerp(hi_a, hi_b, blend)),
        )
        blended["energy"] = _lerp(params_a["energy"], params_b["energy"], blend)
        blended["complexity"] = _lerp(params_a["complexity"], params_b["complexity"], blend)

        # 混合乐器
        all_instruments = set(params_a["instrument_weights"]) | set(params_b["instrument_weights"])
        inst_mix: Dict[str, float] = {}
        for inst in all_instruments:
            w_a = params_a["instrument_weights"].get(inst, 0.0)
            w_b = params_b["instrument_weights"].get(inst, 0.0)
            inst_mix[inst] = _lerp(w_a, w_b, blend)
        blended["instrument_weights"] = inst_mix

        return blended


# ============================================================================
# MultiTrackGenerator - 多轨生成器
# ============================================================================

class MultiTrackGenerator:
    """
    多轨生成器：分别生成各个音轨并混合。

    支持独立控制每个音轨的乐器、音量和声像。
    """

    def __init__(self, config: AudioCraftConfig):
        self._config = config
        self._genre_controller = GenreController()

    def generate_tracks(
        self,
        prompt: AudioPrompt,
        genre: str = "pop",
        bpm: Optional[int] = None,
        duration: float = 10.0,
        seed: Optional[int] = None,
    ) -> List[Track]:
        """生成多轨音乐"""
        if seed is not None:
            random.seed(seed)

        if bpm is None:
            bpm = self._genre_controller.generate_bpm(genre)

        inst_mix = self._genre_controller.get_instrument_mix(genre)
        tracks: List[Track] = []

        for instrument, weight in inst_mix.items():
            track = self._generate_single_track(
                instrument, bpm, duration, weight, prompt
            )
            tracks.append(track)

        return tracks

    def _generate_single_track(
        self,
        instrument: str,
        bpm: int,
        duration: float,
        volume: float,
        prompt: AudioPrompt,
    ) -> Track:
        """生成单个音轨"""
        sr = self._config.sample_rate
        total_samples = int(duration * sr)
        samples: List[float] = [0.0] * total_samples

        beat_duration = 60.0 / bpm
        num_beats = int(duration / beat_duration)

        if instrument in ("drums",):
            samples = self._generate_drums(num_beats, beat_duration, sr)
        elif instrument in ("bass",):
            samples = self._generate_bass(num_beats, beat_duration, sr)
        elif instrument in ("guitar", "synth", "keys", "piano"):
            samples = self._generate_harmonic(num_beats, beat_duration, sr, instrument)
        elif instrument in ("pad", "strings", "texture"):
            samples = self._generate_pad(num_beats, beat_duration, sr)
        else:
            samples = self._generate_harmonic(num_beats, beat_duration, sr, instrument)

        # 截断到目标长度
        samples = samples[:total_samples]
        while len(samples) < total_samples:
            samples.append(0.0)

        return Track(
            name=f"{instrument}_track",
            samples=samples,
            sample_rate=sr,
            volume=volume,
            pan=0.0,
            instrument=instrument,
        )

    def _generate_drums(
        self, num_beats: int, beat_dur: float, sr: int
    ) -> List[float]:
        """生成鼓轨"""
        samples: List[float] = []
        for beat in range(num_beats):
            beat_samples = int(beat_dur * sr)
            for s in range(beat_samples):
                t = s / sr
                sample = 0.0

                # 底鼓（每拍）
                kick_freq = 60.0 * math.exp(-t * 30)
                kick_env = math.exp(-t * 15)
                sample += 0.5 * kick_env * math.sin(2.0 * math.pi * kick_freq * t)

                # 军鼓（2、4拍）
                if beat % 2 == 1 and t < 0.1:
                    snare_env = math.exp(-t * 30)
                    sample += 0.3 * snare_env * _gaussian_noise(0, 1.0)
                    sample += 0.2 * snare_env * math.sin(2.0 * math.pi * 200.0 * t)

                # 踩镲（每半拍）
                half_t = t % (beat_dur / 2)
                if half_t < 0.02:
                    hh_env = math.exp(-half_t * 200)
                    sample += 0.15 * hh_env * _gaussian_noise(0, 1.0)

                samples.append(_clamp(sample, -1.0, 1.0))

        return samples

    def _generate_bass(
        self, num_beats: int, beat_dur: float, sr: int
    ) -> List[float]:
        """生成贝斯轨"""
        bass_notes = [65.41, 73.42, 82.41, 98.0]  # C2, D2, E2, G2
        samples: List[float] = []

        for beat in range(num_beats):
            freq = bass_notes[beat % len(bass_notes)]
            beat_samples = int(beat_dur * sr)
            for s in range(beat_samples):
                t = s / sr
                env = math.exp(-t * 3) * 0.8 + 0.2
                phase = 2.0 * math.pi * freq * t
                sample = env * 0.4 * math.sin(phase)
                sample += env * 0.2 * math.sin(2.0 * phase)
                samples.append(_clamp(sample, -1.0, 1.0))

        return samples

    def _generate_harmonic(
        self, num_beats: int, beat_dur: float, sr: int, instrument: str
    ) -> List[float]:
        """生成和声乐器轨"""
        chord_progressions = [
            [261.63, 329.63, 392.0],   # C major
            [293.66, 369.99, 440.0],   # D major
            [349.23, 440.0, 523.25],   # F major
            [392.0, 493.88, 587.33],   # G major
        ]
        samples: List[float] = []

        for beat in range(num_beats):
            chord = chord_progressions[(beat // 4) % len(chord_progressions)]
            beat_samples = int(beat_dur * sr)
            for s in range(beat_samples):
                t = s / sr
                env = min(t * 20, 1.0) * math.exp(-t * 2) * 0.8 + 0.1
                sample = 0.0
                for freq in chord:
                    phase = 2.0 * math.pi * freq * t
                    sample += env * 0.15 * math.sin(phase)
                    sample += env * 0.05 * math.sin(2.0 * phase)
                samples.append(_clamp(sample, -1.0, 1.0))

        return samples

    def _generate_pad(
        self, num_beats: int, beat_dur: float, sr: int
    ) -> List[float]:
        """生成铺底音色"""
        base_freqs = [130.81, 164.81, 196.0]  # C3, E3, G3
        samples: List[float] = []
        total_samples = int(num_beats * beat_dur * sr)

        for s in range(total_samples):
            t = s / sr
            env = min(t * 2, 1.0) * min((num_beats * beat_dur - t) * 2, 1.0)
            sample = 0.0
            for freq in base_freqs:
                phase = 2.0 * math.pi * freq * t
                sample += env * 0.1 * math.sin(phase)
                sample += env * 0.03 * math.sin(phase * 1.5)
                sample += env * 0.02 * math.sin(phase * 2.01)
            sample += _gaussian_noise(0, env * 0.01)
            samples.append(_clamp(sample, -1.0, 1.0))

        return samples


# ============================================================================
# MixMaster - 混音/母带处理
# ============================================================================

class MixMaster:
    """
    混音/母带处理器：混合多轨音频并进行母带处理。

    包括音量平衡、声像定位、均衡器和压缩器。
    """

    def __init__(self, config: AudioCraftConfig):
        self._config = config

    def mix_tracks(self, tracks: List[Track]) -> List[float]:
        """混合多个音轨"""
        if not tracks:
            return []

        max_len = max(len(t.samples) for t in tracks if not t.muted) or 0
        if max_len == 0:
            return []

        mixed: List[float] = [0.0] * max_len

        for track in tracks:
            if track.muted:
                continue

            volume = track.volume
            if track.solo:
                volume *= 1.5

            for i in range(min(len(track.samples), max_len)):
                mixed[i] += track.samples[i] * volume

        # 防止削波
        max_val = max(abs(s) for s in mixed) if mixed else 1.0
        if max_val > 1.0:
            mixed = [s / max_val * 0.95 for s in mixed]

        return mixed

    def apply_eq(
        self, samples: List[float], low_gain: float = 0.0,
        mid_gain: float = 0.0, high_gain: float = 0.0,
    ) -> List[float]:
        """应用均衡器（简化三段EQ）"""
        if not samples:
            return samples

        result = list(samples)
        n = len(result)

        # 低频增强/衰减（简化：移动平均）
        if low_gain != 0.0:
            window = 64
            smoothed = [0.0] * n
            for i in range(n):
                start = max(0, i - window)
                end = min(n, i + window)
                smoothed[i] = sum(result[start:end]) / (end - start)
            for i in range(n):
                result[i] += smoothed[i] * low_gain

        # 高频增强/衰减
        if high_gain != 0.0:
            for i in range(1, n):
                diff = result[i] - result[i - 1]
                result[i] += diff * high_gain * 0.5

        # 归一化
        max_val = max(abs(s) for s in result) if result else 1.0
        if max_val > 1.0:
            result = [s / max_val for s in result]

        return result

    def apply_compression(
        self, samples: List[float], threshold: float = 0.5,
        ratio: float = 4.0, attack: float = 0.01, release: float = 0.1,
    ) -> List[float]:
        """应用动态压缩"""
        if not samples:
            return samples

        sr = self._config.sample_rate
        attack_samples = int(attack * sr)
        release_samples = int(release * sr)
        result: List[float] = []
        envelope = 0.0

        for i, s in enumerate(samples):
            abs_s = abs(s)
            if abs_s > envelope:
                envelope += (abs_s - envelope) / max(attack_samples, 1)
            else:
                envelope += (abs_s - envelope) / max(release_samples, 1)

            if envelope > threshold:
                gain = threshold + (envelope - threshold) / ratio
                gain = gain / max(envelope, 1e-10)
            else:
                gain = 1.0

            result.append(s * gain)

        return result

    def apply_reverb(
        self, samples: List[float], room_size: float = 0.5,
        damping: float = 0.5, wet_level: float = 0.3,
    ) -> List[float]:
        """应用混响（简化延迟线模型）"""
        if not samples:
            return samples

        sr = self._config.sample_rate
        delay_ms = int(room_size * 200)
        delay_samples = int(delay_ms * sr / 1000)
        feedback = 0.3 * room_size * (1.0 - damping * 0.5)

        result = list(samples)
        n = len(result)

        # 简单的Schroeder混响
        delays = [delay_samples, int(delay_samples * 1.3), int(delay_samples * 1.7)]
        for delay in delays:
            if delay >= n:
                continue
            for i in range(delay, n):
                result[i] += result[i - delay] * feedback

        # 混合干湿信号
        final = [
            _lerp(samples[i], result[i], wet_level)
            for i in range(n)
        ]

        # 归一化
        max_val = max(abs(s) for s in final) if final else 1.0
        if max_val > 1.0:
            final = [s / max_val for s in final]

        return final

    def master(
        self, samples: List[float], target_lufs: float = -14.0
    ) -> List[float]:
        """母带处理"""
        if not samples:
            return samples

        # 压缩
        result = self.apply_compression(samples, threshold=0.4, ratio=3.0)

        # EQ
        result = self.apply_eq(result, low_gain=-0.1, high_gain=0.05)

        # 混响
        result = self.apply_reverb(result, room_size=0.3, wet_level=0.15)

        # 限幅
        result = [max(-0.98, min(0.98, s)) for s in result]

        # 响度归一化
        rms = math.sqrt(sum(s * s for s in result) / len(result))
        if rms > 0:
            target_rms = 10.0 ** ((target_lufs + 18) / 20.0)
            gain = target_rms / rms
            gain = min(gain, 2.0)
            result = [s * gain for s in result]

        return result


# ============================================================================
# AudioCraft - AudioCraft主入口
# ============================================================================

class AudioCraft:
    """
    AudioCraft音频生成框架。

    流程:
    1. 解析文本提示
    2. 选择流派和参数
    3. 生成多轨音频
    4. 混音和母带处理
    5. 输出最终音频

    使用方法:
        ac = AudioCraft()
        output = ac.generate("A happy pop song with piano and drums")
    """

    def __init__(self, config: Optional[AudioCraftConfig] = None):
        self._config = config or AudioCraftConfig()
        self._prompt_encoder = AudioPromptEncoder(self._config)
        self._musicgen = MusicGenWrapper(self._config)
        self._genre_controller = GenreController()
        self._multi_track = MultiTrackGenerator(self._config)
        self._mix_master = MixMaster(self._config)

    @property
    def config(self) -> AudioCraftConfig:
        return self._config

    def generate(
        self,
        text_prompt: str,
        genre: Optional[str] = None,
        bpm: Optional[int] = None,
        duration: float = 10.0,
        melody_prompt: Optional[AudioPrompt] = None,
        seed: Optional[int] = None,
    ) -> MusicOutput:
        """
        生成音乐。

        Args:
            text_prompt: 文本描述
            genre: 音乐流派（自动检测如果为None）
            bpm: BPM（自动生成如果为None）
            duration: 持续时间（秒）
            melody_prompt: 旋律提示音频
            seed: 随机种子

        Returns:
            MusicOutput: 音乐输出
        """
        if seed is not None:
            random.seed(seed)

        # 检测流派
        if genre is None:
            genre = self._detect_genre(text_prompt)

        # 确定BPM
        if bpm is None:
            bpm = self._genre_controller.generate_bpm(genre)

        key = self._genre_controller.select_key(genre)

        # 创建提示
        prompt = AudioPrompt(
            text_description=text_prompt,
            sample_rate=self._config.sample_rate,
        )
        if melody_prompt:
            prompt.samples = melody_prompt.samples

        # 生成多轨
        tracks = self._multi_track.generate_tracks(
            prompt, genre, bpm, duration, seed
        )

        # 混音
        mixed = self._mix_master.mix_tracks(tracks)

        # 母带处理
        mastered = self._mix_master.master(mixed)

        # 淡入淡出
        fade = int(self._config.sample_rate * 0.5)
        mastered = _fade_in_out(mastered, fade)

        actual_duration = len(mastered) / self._config.sample_rate

        return MusicOutput(
            samples=mastered,
            sample_rate=self._config.sample_rate,
            duration=actual_duration,
            tracks=tracks,
            bpm=bpm,
            key=key,
            genre=genre,
            metadata={
                "text_prompt": text_prompt,
                "generation_id": _generate_id(),
                "model": self._config.model.value,
            },
        )

    def generate_with_musicgen(
        self,
        text_prompt: str,
        duration: float = 10.0,
        seed: Optional[int] = None,
    ) -> MusicOutput:
        """使用MusicGen模型直接生成"""
        prompt = AudioPrompt(
            text_description=text_prompt,
            sample_rate=self._config.sample_rate,
        )

        samples = self._musicgen.generate(prompt, duration, seed=seed)

        fade = int(self._config.sample_rate * 0.3)
        samples = _fade_in_out(samples, fade)

        return MusicOutput(
            samples=samples,
            sample_rate=self._config.sample_rate,
            duration=len(samples) / self._config.sample_rate,
            metadata={"method": "musicgen_direct", "generation_id": _generate_id()},
        )

    def _detect_genre(self, text: str) -> str:
        """从文本描述检测流派"""
        text_lower = text.lower()
        genre_keywords: Dict[str, List[str]] = {
            "pop": ["pop", "popular", "catchy", "radio"],
            "rock": ["rock", "guitar", "heavy", "metal"],
            "jazz": ["jazz", "swing", "bebop", "improvis"],
            "classical": ["classical", "orchestra", "symphony", "sonata"],
            "electronic": ["electronic", "edm", "techno", "house", "trance"],
            "hiphop": ["hip hop", "hiphop", "rap", "beat", "trap"],
            "ambient": ["ambient", "atmospheric", "relaxing", "meditation"],
            "blues": ["blues", "blue", "soulful", "delta"],
        }

        best_genre = "pop"
        best_score = 0
        for genre, keywords in genre_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_score:
                best_score = score
                best_genre = genre

        return best_genre
