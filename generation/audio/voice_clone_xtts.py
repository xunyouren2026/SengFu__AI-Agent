"""
XTTS v2 Voice Cloning - XTTS语音克隆引擎

本模块实现了XTTS v2语音克隆系统，包含说话人嵌入、少样本语音适配、
声音风格迁移、语言无关合成和质量评估功能。仅使用标准库，
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


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-10:
        return vec
    return [x / norm for x in vec]


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class SpeakerEmbedding:
    """说话人嵌入向量"""
    vector: List[float] = field(default_factory=list)
    speaker_id: str = ""
    speaker_name: str = ""
    language: str = "en"
    num_samples: int = 0
    quality_score: float = 0.0
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReferenceAudio:
    """参考音频"""
    samples: List[float] = field(default_factory=list)
    sample_rate: int = 24000
    speaker_id: str = ""
    duration: float = 0.0
    text: str = ""
    quality_score: float = 0.0


@dataclass
class CloneConfig:
    """克隆配置"""
    sample_rate: int = 24000
    embedding_dim: int = 512
    min_reference_seconds: float = 3.0
    max_reference_seconds: float = 30.0
    num_reference_files: int = 3
    temperature: float = 0.7
    length_penalty: float = 1.0
    repetition_penalty: float = 1.2
    language: str = "en"
    enable_gpt_codecs: bool = True
    speaker_encoder_model: str = "xtts_v2"


@dataclass
class CloneResult:
    """克隆结果"""
    clone_id: str = ""
    speaker_embedding: Optional[SpeakerEmbedding] = None
    quality_score: float = 0.0
    similarity_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesisOutput:
    """合成输出"""
    samples: List[float] = field(default_factory=list)
    sample_rate: int = 24000
    duration: float = 0.0
    speaker_id: str = ""
    language: str = "en"
    quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# SpeakerEmbedding - 说话人嵌入提取
# ============================================================================

class SpeakerEmbeddingExtractor:
    """
    说话人嵌入提取器：从音频中提取说话人特征向量。

    模拟GE2E/Speaker Verification模型的嵌入提取过程。
    """

    def __init__(self, config: CloneConfig):
        self._config = config
        self._dim = config.embedding_dim
        random.seed(42)
        self._projection_weights = [
            [random.gauss(0, 0.02) for _ in range(self._dim)]
            for _ in range(self._dim)
        ]
        self._projection_bias = [0.0] * self._dim
        random.seed()

    def extract_embedding(
        self, audio: ReferenceAudio
    ) -> SpeakerEmbedding:
        """从参考音频提取说话人嵌入"""
        if not audio.samples:
            return SpeakerEmbedding(speaker_id=audio.speaker_id)

        # 步骤1: 计算音频特征
        features = self._compute_audio_features(audio.samples, audio.sample_rate)

        # 步骤2: 通过模拟的说话人编码器
        embedding = self._encode_features(features)

        # 步骤3: L2归一化
        embedding = _normalize(embedding)

        # 步骤4: 质量评估
        quality = self._assess_embedding_quality(embedding, audio)

        return SpeakerEmbedding(
            vector=embedding,
            speaker_id=audio.speaker_id,
            language=audio.text[:2] if audio.text else "en",
            num_samples=len(audio.samples),
            quality_score=quality,
            timestamp=time.time(),
        )

    def extract_from_multiple(
        self, audios: List[ReferenceAudio]
    ) -> SpeakerEmbedding:
        """从多个参考音频提取聚合嵌入"""
        if not audios:
            return SpeakerEmbedding()

        embeddings = [self.extract_embedding(a) for a in audios]

        # 加权平均（基于质量分数）
        total_weight = sum(e.quality_score for e in embeddings if e.quality_score > 0)
        if total_weight == 0:
            total_weight = len(embeddings)

        dim = self._config.embedding_dim
        avg_vector = [0.0] * dim
        for emb in embeddings:
            weight = emb.quality_score / total_weight
            for i in range(min(dim, len(emb.vector))):
                avg_vector[i] += emb.vector[i] * weight

        avg_vector = _normalize(avg_vector)

        speaker_id = audios[0].speaker_id
        avg_quality = sum(e.quality_score for e in embeddings) / len(embeddings)

        return SpeakerEmbedding(
            vector=avg_vector,
            speaker_id=speaker_id,
            language=embeddings[0].language,
            num_samples=sum(a.num_samples for a in audios),
            quality_score=avg_quality,
            timestamp=time.time(),
        )

    def _compute_audio_features(
        self, samples: List[float], sample_rate: int
    ) -> List[float]:
        """计算音频特征（模拟MFCC）"""
        if not samples:
            return [0.0] * self._dim

        # 简化特征提取
        features: List[float] = []

        # 1. 统计特征
        n = len(samples)
        mean = sum(samples) / n
        variance = sum((s - mean) ** 2 for s in samples) / n
        std = math.sqrt(variance)
        features.extend([mean, std, variance])

        # 2. 零交叉率
        zero_crossings = sum(
            1 for i in range(1, n)
            if (samples[i] >= 0) != (samples[i - 1] >= 0)
        )
        zcr = zero_crossings / n
        features.append(zcr)

        # 3. 能量特征
        energy = sum(s * s for s in samples) / n
        log_energy = math.log(energy + 1e-10)
        features.extend([energy, log_energy])

        # 4. 频谱质心（简化）
        frame_size = min(1024, n)
        num_frames = max(1, n // frame_size)
        spectral_centroids: List[float] = []
        for f in range(num_frames):
            start = f * frame_size
            frame = samples[start:start + frame_size]
            if not frame:
                continue
            # 简化的DFT能量分布
            n_bins = 32
            energies: List[float] = []
            for k in range(n_bins):
                freq = k / n_bins
                e = sum(
                    frame[i] * math.sin(2.0 * math.pi * freq * i / len(frame))
                    for i in range(len(frame))
                )
                energies.append(e * e)
            total_e = sum(energies)
            if total_e > 0:
                centroid = sum(k * e for k, e in enumerate(energies)) / total_e
                spectral_centroids.append(centroid)

        if spectral_centroids:
            features.append(sum(spectral_centroids) / len(spectral_centroids))
            features.append(max(spectral_centroids) - min(spectral_centroids))
        else:
            features.extend([0.0, 0.0])

        # 5. 填充到目标维度
        while len(features) < self._dim:
            features.append(_gaussian_noise(0, 0.01))
        features = features[:self._dim]

        return features

    def _encode_features(self, features: List[float]) -> List[float]:
        """通过模拟编码器"""
        # 简单的非线性变换
        dim = self._dim
        result: List[float] = []
        for i in range(dim):
            val = features[i] if i < len(features) else 0.0
            # 通过投影层
            proj = sum(
                self._projection_weights[i][j] * features[j]
                for j in range(min(dim, len(features)))
            ) + self._projection_bias[i]
            # 非线性激活
            val = 0.5 * proj * (1.0 + math.tanh(proj))
            result.append(val)
        return result

    def _assess_embedding_quality(
        self, embedding: List[float], audio: ReferenceAudio
    ) -> float:
        """评估嵌入质量"""
        score = 0.0

        # 嵌入范数（归一化后应该接近1）
        norm = math.sqrt(sum(x * x for x in embedding))
        score += 0.3 if 0.9 <= norm <= 1.1 else 0.1

        # 音频时长
        if audio.duration >= 3.0:
            score += 0.3
        elif audio.duration >= 1.0:
            score += 0.2
        else:
            score += 0.1

        # 音频能量
        if audio.samples:
            energy = sum(s * s for s in audio.samples) / len(audio.samples)
            if energy > 0.001:
                score += 0.2
            else:
                score += 0.05

        # 嵌入方差
        mean = sum(embedding) / len(embedding)
        var = sum((x - mean) ** 2 for x in embedding) / len(embedding)
        if var > 0.01:
            score += 0.2
        else:
            score += 0.1

        return _clamp(score, 0.0, 1.0)

    def compute_similarity(
        self, emb_a: SpeakerEmbedding, emb_b: SpeakerEmbedding
    ) -> float:
        """计算两个嵌入的相似度"""
        return _cosine_similarity(emb_a.vector, emb_b.vector)


# ============================================================================
# FewShotAdapter - 少样本语音适配
# ============================================================================

class FewShotAdapter:
    """
    少样本语音适配器：使用少量参考音频快速适配目标说话人。

    支持自适应权重调整和参考音频选择。
    """

    def __init__(self, config: CloneConfig):
        self._config = config
        self._embedding_extractor = SpeakerEmbeddingExtractor(config)
        self._reference_store: Dict[str, List[ReferenceAudio]] = defaultdict(list)
        self._adaptation_cache: Dict[str, SpeakerEmbedding] = {}

    def add_reference(self, audio: ReferenceAudio) -> None:
        """添加参考音频"""
        self._reference_store[audio.speaker_id].append(audio)
        # 清除缓存
        if audio.speaker_id in self._adaptation_cache:
            del self._adaptation_cache[audio.speaker_id]

    def add_references(self, audios: List[ReferenceAudio]) -> None:
        """批量添加参考音频"""
        for audio in audios:
            self.add_reference(audio)

    def get_optimal_references(
        self, speaker_id: str, num_refs: Optional[int] = None
    ) -> List[ReferenceAudio]:
        """获取最优参考音频子集"""
        audios = self._reference_store.get(speaker_id, [])
        if not audios:
            return []

        n = num_refs or self._config.num_reference_files

        # 按质量和时长排序
        scored = []
        for audio in audios:
            duration_score = 1.0
            if audio.duration < self._config.min_reference_seconds:
                duration_score = audio.duration / self._config.min_reference_seconds
            elif audio.duration > self._config.max_reference_seconds:
                duration_score = self._config.max_reference_seconds / audio.duration

            quality = audio.quality_score * duration_score
            scored.append((audio, quality))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [a for a, _ in scored[:n]]

    def adapt(
        self, speaker_id: str, num_shots: Optional[int] = None
    ) -> SpeakerEmbedding:
        """执行少样本适配"""
        if speaker_id in self._adaptation_cache:
            return self._adaptation_cache[speaker_id]

        refs = self.get_optimal_references(speaker_id, num_shots)
        if not refs:
            return SpeakerEmbedding(speaker_id=speaker_id)

        embedding = self._embedding_extractor.extract_from_multiple(refs)
        self._adaptation_cache[speaker_id] = embedding
        return embedding

    def incremental_adapt(
        self, speaker_id: str, new_audio: ReferenceAudio
    ) -> SpeakerEmbedding:
        """增量适配：添加新音频后更新嵌入"""
        self.add_reference(new_audio)
        return self.adapt(speaker_id)

    def get_adaptation_info(self, speaker_id: str) -> Dict[str, Any]:
        """获取适配信息"""
        audios = self._reference_store.get(speaker_id, [])
        return {
            "speaker_id": speaker_id,
            "num_references": len(audios),
            "total_duration": sum(a.duration for a in audios),
            "is_cached": speaker_id in self._adaptation_cache,
            "optimal_subset": len(self.get_optimal_references(speaker_id)),
        }


# ============================================================================
# StyleTransfer - 声音风格迁移
# ============================================================================

class StyleTransfer:
    """
    声音风格迁移：将一个说话人的风格特征迁移到另一个说话人。

    迁移维度:
    - 基频轮廓
    - 能量包络
    - 语速模式
    - 音色特征
    """

    def __init__(self, config: CloneConfig):
        self._config = config

    def extract_style_features(
        self, samples: List[float], sample_rate: int
    ) -> Dict[str, Any]:
        """提取声音风格特征"""
        if not samples:
            return self._empty_style()

        n = len(samples)
        frame_size = min(512, n)

        # 基频统计
        pitch_values = self._estimate_pitch_contour(samples, sample_rate)
        pitch_mean = sum(pitch_values) / max(len(pitch_values), 1)
        pitch_std = math.sqrt(
            sum((p - pitch_mean) ** 2 for p in pitch_values) / max(len(pitch_values), 1)
        ) if len(pitch_values) > 1 else 0.0

        # 能量包络
        energy_values: List[float] = []
        for i in range(0, n - frame_size, frame_size // 2):
            frame = samples[i:i + frame_size]
            energy = sum(s * s for s in frame) / len(frame)
            energy_values.append(energy)

        energy_mean = sum(energy_values) / max(len(energy_values), 1)
        energy_std = math.sqrt(
            sum((e - energy_mean) ** 2 for e in energy_values) / max(len(energy_values), 1)
        ) if len(energy_values) > 1 else 0.0

        # 语速（基于能量变化率）
        if len(energy_values) > 1:
            transitions = sum(
                1 for i in range(1, len(energy_values))
                if abs(energy_values[i] - energy_values[i - 1]) > energy_mean * 0.3
            )
            speech_rate = transitions / (n / sample_rate)
        else:
            speech_rate = 0.0

        return {
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "pitch_range": pitch_std * 2,
            "energy_mean": energy_mean,
            "energy_std": energy_std,
            "speech_rate": speech_rate,
            "duration": n / sample_rate,
        }

    def _empty_style(self) -> Dict[str, Any]:
        return {
            "pitch_mean": 220.0, "pitch_std": 30.0, "pitch_range": 60.0,
            "energy_mean": 0.01, "energy_std": 0.005, "speech_rate": 3.0,
            "duration": 0.0,
        }

    def _estimate_pitch_contour(
        self, samples: List[float], sample_rate: int
    ) -> List[float]:
        """估计基频轮廓（自相关法简化版）"""
        frame_size = min(1024, len(samples))
        pitches: List[float] = []

        for i in range(0, len(samples) - frame_size, frame_size // 2):
            frame = samples[i:i + frame_size]
            pitch = self._detect_pitch(frame, sample_rate)
            pitches.append(pitch)

        return pitches

    def _detect_pitch(self, frame: List[float], sample_rate: int) -> float:
        """基频检测（自相关法）"""
        n = len(frame)
        min_lag = int(sample_rate / 500)  # 500Hz上限
        max_lag = int(sample_rate / 80)   # 80Hz下限

        if max_lag >= n:
            return 220.0

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
            return 220.0

        return sample_rate / best_lag

    def transfer_style(
        self,
        source_samples: List[float],
        target_style: Dict[str, Any],
        sample_rate: int,
        blend_factor: float = 1.0,
    ) -> List[float]:
        """将目标风格迁移到源音频"""
        if not source_samples:
            return source_samples

        source_style = self.extract_style_features(source_samples, sample_rate)

        # 计算风格比例
        pitch_ratio = 1.0
        if source_style["pitch_mean"] > 0:
            pitch_ratio = target_style["pitch_mean"] / source_style["pitch_mean"]

        energy_ratio = 1.0
        if source_style["energy_mean"] > 1e-10:
            energy_ratio = target_style["energy_mean"] / source_style["energy_mean"]

        # 应用混合因子
        pitch_ratio = _lerp(1.0, pitch_ratio, blend_factor)
        energy_ratio = _lerp(1.0, energy_ratio, blend_factor)

        # 应用变换
        result: List[float] = []
        for s in source_samples:
            new_s = s * energy_ratio
            result.append(_clamp(new_s, -1.0, 1.0))

        return result

    def compute_style_distance(
        self, style_a: Dict[str, Any], style_b: Dict[str, Any]
    ) -> float:
        """计算两个风格的距离"""
        pitch_diff = abs(style_a["pitch_mean"] - style_b["pitch_mean"]) / 300.0
        energy_diff = abs(style_a["energy_mean"] - style_b["energy_mean"]) / max(
            style_a["energy_mean"] + style_b["energy_mean"], 1e-10
        )
        rate_diff = abs(style_a["speech_rate"] - style_b["speech_rate"]) / 10.0

        return math.sqrt(pitch_diff ** 2 + energy_diff ** 2 + rate_diff ** 2)


# ============================================================================
# LanguageAgnosticSynth - 语言无关合成
# ============================================================================

class LanguageAgnosticSynth:
    """
    语言无关合成器：支持多语言语音合成。

    使用统一的音素集和语言特定的后处理。
    """

    def __init__(self, config: CloneConfig):
        self._config = config
        self._language_configs: Dict[str, Dict[str, Any]] = {
            "en": {"default_pitch": 220.0, "speed": 1.0, "phoneme_set": "ipa"},
            "zh": {"default_pitch": 200.0, "speed": 0.95, "phoneme_set": "pinyin"},
            "ja": {"default_pitch": 230.0, "speed": 0.9, "phoneme_set": "kana"},
            "ko": {"default_pitch": 210.0, "speed": 0.95, "phoneme_set": "hangul"},
            "es": {"default_pitch": 215.0, "speed": 1.0, "phoneme_set": "ipa"},
            "fr": {"default_pitch": 210.0, "speed": 0.95, "phoneme_set": "ipa"},
            "de": {"default_pitch": 205.0, "speed": 0.95, "phoneme_set": "ipa"},
            "pt": {"default_pitch": 215.0, "speed": 1.0, "phoneme_set": "ipa"},
            "ru": {"default_pitch": 200.0, "speed": 0.9, "phoneme_set": "ipa"},
        }

    def detect_language(self, text: str) -> str:
        """检测文本语言"""
        chinese = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        japanese = sum(1 for c in text if '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff')
        korean = sum(1 for c in text if '\uac00' <= c <= '\ud7af')
        cyrillic = sum(1 for c in text if '\u0400' <= c <= '\u04ff')
        latin = sum(1 for c in text if c.isalpha() and ord(c) < 128)

        total = len(text) or 1
        scores = {
            "zh": chinese / total,
            "ja": japanese / total,
            "ko": korean / total,
            "ru": cyrillic / total,
            "en": latin / total,
        }
        return max(scores, key=scores.get) if any(scores.values()) else "en"

    def synthesize(
        self,
        text: str,
        embedding: SpeakerEmbedding,
        language: Optional[str] = None,
    ) -> SynthesisOutput:
        """语言无关合成"""
        lang = language or self.detect_language(text)
        lang_config = self._language_configs.get(lang, self._language_configs["en"])

        samples = self._generate_multilingual(text, lang, lang_config, embedding)

        return SynthesisOutput(
            samples=samples,
            sample_rate=self._config.sample_rate,
            duration=len(samples) / self._config.sample_rate,
            speaker_id=embedding.speaker_id,
            language=lang,
            quality_score=0.8,
        )

    def _generate_multilingual(
        self,
        text: str,
        language: str,
        lang_config: Dict[str, Any],
        embedding: SpeakerEmbedding,
    ) -> List[float]:
        """生成多语言语音"""
        base_pitch = lang_config["default_pitch"]
        speed = lang_config["speed"]
        sr = self._config.sample_rate

        # 使用嵌入调整基频
        if embedding.vector:
            pitch_mod = sum(embedding.vector[:10]) * 20.0
            base_pitch = _clamp(base_pitch + pitch_mod, 80.0, 400.0)

        samples: List[float] = []
        chars = list(text)

        for char_idx, char in enumerate(chars):
            if not char.strip():
                samples.extend([0.0] * int(0.02 * sr))
                continue

            # 字符持续时间
            char_duration = 0.06 / speed
            num_samples = int(char_duration * sr)

            # 基于字符编码生成音高变化
            char_code = ord(char)
            char_pitch = base_pitch + (char_code % 20 - 10) * 2.0

            for s in range(num_samples):
                t = s / sr
                phase = 2.0 * math.pi * char_pitch * t
                sample = 0.3 * math.sin(phase)
                sample += 0.15 * math.sin(2.0 * phase)
                sample += 0.08 * math.sin(3.0 * phase)

                env = math.sin(math.pi * s / max(num_samples - 1, 1))
                sample *= env
                sample += _gaussian_noise(0, 0.01)
                samples.append(_clamp(sample, -1.0, 1.0))

        # 归一化
        if samples:
            max_val = max(abs(s) for s in samples)
            if max_val > 0:
                samples = [s / max_val * 0.9 for s in samples]

        return samples

    def supported_languages(self) -> List[str]:
        return list(self._language_configs.keys())


# ============================================================================
# QualityAssessor - 质量评估器
# ============================================================================

class QualityAssessor:
    """质量评估器：评估合成语音的质量"""

    def __init__(self, config: CloneConfig):
        self._config = config
        self._embedding_extractor = SpeakerEmbeddingExtractor(config)

    def assess_clone_quality(
        self,
        reference: ReferenceAudio,
        synthesized: SynthesisOutput,
    ) -> Dict[str, float]:
        """评估克隆质量"""
        ref_embedding = self._embedding_extractor.extract_embedding(reference)
        synth_ref = ReferenceAudio(
            samples=synthesized.samples,
            sample_rate=synthesized.sample_rate,
            speaker_id=synthesized.speaker_id,
            duration=synthesized.duration,
        )
        synth_embedding = self._embedding_extractor.extract_embedding(synth_ref)

        # 说话人相似度
        similarity = self._embedding_extractor.compute_similarity(
            ref_embedding, synth_embedding
        )

        # 音频质量指标
        snr = self._estimate_snr(synthesized.samples)
        continuity = self._assess_continuity(synthesized.samples)

        # 综合质量分数
        overall = similarity * 0.5 + snr * 0.3 + continuity * 0.2

        return {
            "speaker_similarity": similarity,
            "signal_to_noise": snr,
            "continuity": continuity,
            "overall_quality": overall,
        }

    def _estimate_snr(self, samples: List[float]) -> float:
        """估计信噪比"""
        if not samples:
            return 0.0
        signal_power = sum(s * s for s in samples) / len(samples)
        # 估计噪声（使用差分信号）
        if len(samples) < 2:
            return 1.0
        diffs = [samples[i] - samples[i - 1] for i in range(1, len(samples))]
        noise_power = sum(d * d for d in diffs) / len(diffs) * 0.5

        if noise_power < 1e-10:
            return 1.0

        snr_linear = signal_power / noise_power
        snr_db = 10.0 * math.log10(max(snr_linear, 1e-10))
        return _clamp(snr_db / 40.0, 0.0, 1.0)

    def _assess_continuity(self, samples: List[float]) -> float:
        """评估连续性（检测不自然的跳变）"""
        if len(samples) < 2:
            return 1.0

        frame_size = 256
        num_frames = len(samples) // frame_size
        if num_frames < 2:
            return 0.8

        discontinuities = 0
        for i in range(num_frames - 1):
            end = samples[i * frame_size + frame_size - 1]
            start = samples[(i + 1) * frame_size]
            diff = abs(end - start)
            if diff > 0.3:
                discontinuities += 1

        continuity = 1.0 - discontinuities / max(num_frames - 1, 1)
        return _clamp(continuity, 0.0, 1.0)


# ============================================================================
# XTTSVoiceClone - XTTS语音克隆（主入口）
# ============================================================================

class XTTSVoiceClone:
    """
    XTTS v2语音克隆引擎。

    流程:
    1. 加载参考音频
    2. 提取说话人嵌入
    3. 少样本适配
    4. 语音合成
    5. 质量评估

    使用方法:
        clone = XTTSVoiceClone()
        clone.register_speaker("speaker1", [ref_audio1, ref_audio2])
        result = clone.synthesize("Hello!", "speaker1")
    """

    def __init__(self, config: Optional[CloneConfig] = None):
        self._config = config or CloneConfig()
        self._embedding_extractor = SpeakerEmbeddingExtractor(self._config)
        self._few_shot_adapter = FewShotAdapter(self._config)
        self._style_transfer = StyleTransfer(self._config)
        self._lang_agnostic = LanguageAgnosticSynth(self._config)
        self._quality_assessor = QualityAssessor(self._config)
        self._speaker_embeddings: Dict[str, SpeakerEmbedding] = {}

    @property
    def config(self) -> CloneConfig:
        return self._config

    def register_speaker(
        self,
        speaker_id: str,
        reference_audios: List[ReferenceAudio],
    ) -> CloneResult:
        """注册说话人（提供参考音频）"""
        for audio in reference_audios:
            audio.speaker_id = speaker_id
            self._few_shot_adapter.add_reference(audio)

        embedding = self._few_shot_adapter.adapt(speaker_id)
        self._speaker_embeddings[speaker_id] = embedding

        quality = embedding.quality_score
        return CloneResult(
            clone_id=_generate_id(),
            speaker_embedding=embedding,
            quality_score=quality,
            similarity_score=quality,
            metadata={
                "speaker_id": speaker_id,
                "num_references": len(reference_audios),
                "total_duration": sum(a.duration for a in reference_audios),
            },
        )

    def synthesize(
        self,
        text: str,
        speaker_id: str,
        language: Optional[str] = None,
        emotion: str = "neutral",
        style_blend: float = 1.0,
        seed: Optional[int] = None,
    ) -> SynthesisOutput:
        """使用克隆的语音合成"""
        if seed is not None:
            random.seed(seed)

        embedding = self._speaker_embeddings.get(speaker_id)
        if not embedding:
            embedding = self._few_shot_adapter.adapt(speaker_id)

        if not embedding or not embedding.vector:
            return SynthesisOutput(
                speaker_id=speaker_id,
                quality_score=0.0,
            )

        # 语言无关合成
        output = self._lang_agnostic.synthesize(text, embedding, language)

        # 风格迁移（如果需要）
        if style_blend < 1.0:
            refs = self._few_shot_adapter.get_optimal_references(speaker_id, 1)
            if refs:
                style = self._style_transfer.extract_style_features(
                    refs[0].samples, refs[0].sample_rate
                )
                output.samples = self._style_transfer.transfer_style(
                    output.samples, style, output.sample_rate, style_blend
                )

        return output

    def clone_and_synthesize(
        self,
        text: str,
        reference_audios: List[ReferenceAudio],
        speaker_id: str = "default",
        language: Optional[str] = None,
    ) -> SynthesisOutput:
        """一步完成克隆和合成"""
        self.register_speaker(speaker_id, reference_audios)
        return self.synthesize(text, speaker_id, language)

    def assess_quality(
        self,
        speaker_id: str,
        reference: ReferenceAudio,
        test_text: str = "The quick brown fox jumps over the lazy dog.",
    ) -> Dict[str, float]:
        """评估克隆质量"""
        synthesized = self.synthesize(test_text, speaker_id)
        return self._quality_assessor.assess_clone_quality(reference, synthesized)

    def list_speakers(self) -> List[str]:
        """列出已注册的说话人"""
        return list(self._speaker_embeddings.keys())

    def get_speaker_info(self, speaker_id: str) -> Optional[Dict[str, Any]]:
        """获取说话人信息"""
        embedding = self._speaker_embeddings.get(speaker_id)
        if not embedding:
            return None
        return {
            "speaker_id": speaker_id,
            "quality_score": embedding.quality_score,
            "num_samples": embedding.num_samples,
            "language": embedding.language,
        }

    def remove_speaker(self, speaker_id: str) -> bool:
        """移除说话人"""
        if speaker_id in self._speaker_embeddings:
            del self._speaker_embeddings[speaker_id]
            return True
        return False
