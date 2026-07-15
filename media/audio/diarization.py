"""
说话人分离模块
区分音频中不同的说话人
"""

import os
import json
import subprocess
import wave
import struct
import math
import tempfile
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple, Generator
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class DiarizationMethod(Enum):
    """说话人分离方法"""
    PYANNOTE = "pyannote"
    SIMPLE = "simple"
    CLUSTERING = "clustering"


@dataclass
class SpeakerSegment:
    """说话人片段"""
    speaker_id: int
    start: float
    end: float
    confidence: float = 1.0
    
    @property
    def duration(self) -> float:
        return self.end - self.start
    
    def __repr__(self) -> str:
        return f"SpeakerSegment(speaker={self.speaker_id}, {self.start:.2f}s-{self.end:.2f}s)"


@dataclass
class SpeakerInfo:
    """说话人信息"""
    speaker_id: int
    total_duration: float = 0.0
    segment_count: int = 0
    average_confidence: float = 0.0
    label: str = ""
    
    def __repr__(self) -> str:
        return (
            f"SpeakerInfo(id={self.speaker_id}, duration={self.total_duration:.2f}s, "
            f"segments={self.segment_count})"
        )


@dataclass
class DiarizationResult:
    """说话人分离结果"""
    segments: List[SpeakerSegment]
    speakers: Dict[int, SpeakerInfo]
    num_speakers: int
    duration: float
    
    def __repr__(self) -> str:
        return (
            f"DiarizationResult(speakers={self.num_speakers}, "
            f"segments={len(self.segments)}, duration={self.duration:.2f}s)"
        )
    
    def get_speaker_segments(self, speaker_id: int) -> List[SpeakerSegment]:
        """获取指定说话人的所有片段"""
        return [s for s in self.segments if s.speaker_id == speaker_id]
    
    def get_segment_at_time(self, time: float) -> Optional[SpeakerSegment]:
        """获取指定时间的说话人片段"""
        for segment in self.segments:
            if segment.start <= time < segment.end:
                return segment
        return None
    
    def to_rttm(self) -> str:
        """转换为 RTTM 格式"""
        lines = []
        for seg in self.segments:
            lines.append(
                f"SPEAKER audio 1 {seg.start:.3f} {seg.duration:.3f} "
                f"<NA> <NA> speaker_{seg.speaker_id} <NA> <NA>"
            )
        return "\n".join(lines)
    
    def merge_with_transcription(
        self,
        transcription_segments: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        与转录结果合并
        
        Args:
            transcription_segments: 转录片段列表
            
        Returns:
            合并后的结果列表
        """
        merged = []
        
        for trans_seg in transcription_segments:
            mid_time = (trans_seg.start + trans_seg.end) / 2
            speaker_segment = self.get_segment_at_time(mid_time)
            
            speaker_id = speaker_segment.speaker_id if speaker_segment else -1
            
            merged.append({
                'speaker': speaker_id,
                'start': trans_seg.start,
                'end': trans_seg.end,
                'text': getattr(trans_seg, 'text', ''),
                'confidence': getattr(trans_seg, 'confidence', 1.0)
            })
        
        return merged


@dataclass
class DiarizationConfig:
    """说话人分离配置"""
    method: DiarizationMethod = DiarizationMethod.SIMPLE
    num_speakers: Optional[int] = None
    min_speakers: int = 2
    max_speakers: int = 10
    min_segment_duration: float = 0.5
    embedding_dim: int = 512
    clustering_threshold: float = 0.5
    vad_aggressiveness: int = 3
    use_gpu: bool = True


class VoiceActivityDetector:
    """语音活动检测器"""
    
    def __init__(self, aggressiveness: int = 3):
        """
        初始化 VAD
        
        Args:
            aggressiveness: 激进程度 (0-3)
        """
        self.aggressiveness = aggressiveness
        self.frame_size = 480
        self.sample_rate = 16000
    
    def detect(
        self,
        audio_path: Union[str, Path]
    ) -> List[Tuple[float, float]]:
        """
        检测语音活动
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            语音活动区间列表
        """
        audio_path = Path(audio_path)
        audio_data = self._read_audio(audio_path)
        if audio_data is None:
            return []
        
        segments = self._energy_based_vad(audio_data)
        return segments
    
    def _read_audio(self, audio_path: Path) -> Optional[bytes]:
        """读取音频数据"""
        if audio_path.suffix.lower() == '.wav':
            try:
                with wave.open(str(audio_path), 'rb') as wf:
                    if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                        return None
                    return wf.readframes(wf.getnframes())
            except Exception:
                pass
        return None
    
    def _energy_based_vad(
        self,
        audio_data: bytes,
        frame_duration_ms: int = 30,
        energy_threshold: float = 0.01
    ) -> List[Tuple[float, float]]:
        """基于能量的 VAD"""
        frame_size = int(self.sample_rate * frame_duration_ms / 1000) * 2
        num_frames = len(audio_data) // frame_size
        
        voiced_frames = []
        
        for i in range(num_frames):
            frame_start = i * frame_size
            frame_data = audio_data[frame_start:frame_start + frame_size]
            
            if len(frame_data) < frame_size:
                break
            
            samples = struct.unpack(f'{len(frame_data)//2}h', frame_data)
            energy = sum(s * s for s in samples) / len(samples)
            normalized_energy = energy / (32768 * 32768)
            
            if normalized_energy > energy_threshold:
                start_time = i * frame_duration_ms / 1000.0
                end_time = (i + 1) * frame_duration_ms / 1000.0
                voiced_frames.append((start_time, end_time, normalized_energy))
        
        if not voiced_frames:
            return []
        
        merged = []
        current_start = voiced_frames[0][0]
        current_end = voiced_frames[0][1]
        
        for start, end, _ in voiced_frames[1:]:
            if start - current_end < 0.3:
                current_end = end
            else:
                merged.append((current_start, current_end))
                current_start = start
                current_end = end
        
        merged.append((current_start, current_end))
        return merged


class SpeakerEmbedding:
    """说话人嵌入向量提取器"""
    
    def __init__(self, embedding_dim: int = 512):
        """
        初始化嵌入提取器
        
        Args:
            embedding_dim: 嵌入维度
        """
        self.embedding_dim = embedding_dim
    
    def extract(
        self,
        audio_path: Union[str, Path],
        start: float,
        end: float
    ) -> Optional[List[float]]:
        """
        提取音频片段的嵌入向量
        
        Args:
            audio_path: 音频文件路径
            start: 开始时间
            end: 结束时间
            
        Returns:
            嵌入向量
        """
        # 简化实现：使用 MFCC 特征作为嵌入
        audio_path = Path(audio_path)
        
        try:
            # 读取音频片段
            audio_data = self._read_segment(audio_path, start, end)
            if audio_data is None:
                return None
            
            # 计算简化的特征向量
            embedding = self._compute_features(audio_data)
            return embedding
            
        except Exception:
            return None
    
    def _read_segment(
        self,
        audio_path: Path,
        start: float,
        end: float
    ) -> Optional[bytes]:
        """读取音频片段"""
        if audio_path.suffix.lower() == '.wav':
            try:
                with wave.open(str(audio_path), 'rb') as wf:
                    sample_rate = wf.getframerate()
                    start_frame = int(start * sample_rate)
                    end_frame = int(end * sample_rate)
                    
                    wf.setpos(start_frame)
                    return wf.readframes(end_frame - start_frame)
            except Exception:
                pass
        return None
    
    def _compute_features(self, audio_data: bytes) -> List[float]:
        """计算特征向量"""
        if len(audio_data) < 2:
            return [0.0] * self.embedding_dim
        
        samples = struct.unpack(f'{len(audio_data)//2}h', audio_data)
        
        # 计算统计特征
        features = []
        
        # 能量
        energy = sum(s * s for s in samples) / len(samples)
        features.append(math.sqrt(energy) / 32768.0)
        
        # 过零率
        zero_crossings = sum(
            1 for i in range(1, len(samples)) if samples[i] * samples[i-1] < 0
        )
        features.append(zero_crossings / len(samples))
        
        # 分段能量
        segment_size = len(samples) // 8
        for i in range(8):
            start = i * segment_size
            end = start + segment_size
            if end <= len(samples):
                seg_energy = sum(samples[start:end]) / segment_size / 32768.0
                features.append(abs(seg_energy))
        
        # 扩展到指定维度
        while len(features) < self.embedding_dim:
            features.append(features[-1] * 0.9)
        
        return features[:self.embedding_dim]


class SpeakerClustering:
    """说话人聚类器"""
    
    def __init__(self, threshold: float = 0.5):
        """
        初始化聚类器
        
        Args:
            threshold: 聚类阈值
        """
        self.threshold = threshold
    
    def cluster(
        self,
        embeddings: List[List[float]],
        min_speakers: int = 2,
        max_speakers: int = 10
    ) -> List[int]:
        """
        对嵌入向量进行聚类
        
        Args:
            embeddings: 嵌入向量列表
            min_speakers: 最小说话人数
            max_speakers: 最大说话人数
            
        Returns:
            每个嵌入对应的说话人标签
        """
        if not embeddings:
            return []
        
        n = len(embeddings)
        
        # 计算距离矩阵
        distances = []
        for i in range(n):
            row = []
            for j in range(n):
                dist = self._cosine_distance(embeddings[i], embeddings[j])
                row.append(dist)
            distances.append(row)
        
        # 简化的层次聚类
        labels = list(range(n))
        
        # 合并最近的簇
        while True:
            # 找到最近的两个不同簇
            min_dist = float('inf')
            merge_i, merge_j = -1, -1
            
            for i in range(n):
                for j in range(i + 1, n):
                    if labels[i] != labels[j]:
                        if distances[i][j] < min_dist:
                            min_dist = distances[i][j]
                            merge_i, merge_j = i, j
            
            # 检查是否应该合并
            num_clusters = len(set(labels))
            
            if min_dist > self.threshold or num_clusters <= min_speakers:
                break
            
            if num_clusters > max_speakers:
                break
            
            # 合并簇
            old_label = labels[merge_j]
            new_label = labels[merge_i]
            labels = [new_label if l == old_label else l for l in labels]
        
        # 重新编号
        unique_labels = sorted(set(labels))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = [label_map[l] for l in labels]
        
        return labels
    
    @staticmethod
    def _cosine_distance(a: List[float], b: List[float]) -> float:
        """计算余弦距离"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        
        if norm_a == 0 or norm_b == 0:
            return 1.0
        
        similarity = dot / (norm_a * norm_b)
        return 1.0 - similarity


class Diarizer:
    """说话人分离主类"""
    
    def __init__(self, config: Optional[DiarizationConfig] = None):
        """
        初始化说话人分离器
        
        Args:
            config: 分离配置
        """
        self.config = config or DiarizationConfig()
        self._vad = VoiceActivityDetector(self.config.vad_aggressiveness)
        self._embedding = SpeakerEmbedding(self.config.embedding_dim)
        self._clustering = SpeakerClustering(self.config.clustering_threshold)
    
    def diarize(
        self,
        audio_path: Union[str, Path],
        config: Optional[DiarizationConfig] = None
    ) -> DiarizationResult:
        """
        执行说话人分离
        
        Args:
            audio_path: 音频文件路径
            config: 分离配置
            
        Returns:
            分离结果
        """
        cfg = config or self.config
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        # 根据方法选择分离策略
        if cfg.method == DiarizationMethod.PYANNOTE:
            return self._diarize_pyannote(audio_path, cfg)
        elif cfg.method == DiarizationMethod.CLUSTERING:
            return self._diarize_clustering(audio_path, cfg)
        else:
            return self._diarize_simple(audio_path, cfg)
    
    def _diarize_simple(
        self,
        audio_path: Path,
        config: DiarizationConfig
    ) -> DiarizationResult:
        """简单说话人分离"""
        # 检测语音活动
        vad_segments = self._vad.detect(audio_path)
        
        if not vad_segments:
            return DiarizationResult(
                segments=[],
                speakers={},
                num_speakers=0,
                duration=0.0
            )
        
        # 获取音频时长
        duration = vad_segments[-1][1] if vad_segments else 0.0
        
        # 简单分配：交替分配给不同说话人
        num_speakers = config.num_speakers or config.min_speakers
        segments = []
        
        for i, (start, end) in enumerate(vad_segments):
            if end - start < config.min_segment_duration:
                continue
            
            speaker_id = i % num_speakers
            segments.append(SpeakerSegment(
                speaker_id=speaker_id,
                start=start,
                end=end,
                confidence=0.8
            ))
        
        # 构建说话人信息
        speakers = self._build_speaker_info(segments)
        
        return DiarizationResult(
            segments=segments,
            speakers=speakers,
            num_speakers=len(speakers),
            duration=duration
        )
    
    def _diarize_clustering(
        self,
        audio_path: Path,
        config: DiarizationConfig
    ) -> DiarizationResult:
        """基于聚类的说话人分离"""
        # 检测语音活动
        vad_segments = self._vad.detect(audio_path)
        
        if not vad_segments:
            return DiarizationResult(
                segments=[],
                speakers={},
                num_speakers=0,
                duration=0.0
            )
        
        duration = vad_segments[-1][1] if vad_segments else 0.0
        
        # 提取嵌入向量
        embeddings = []
        valid_segments = []
        
        for start, end in vad_segments:
            if end - start < config.min_segment_duration:
                continue
            
            embedding = self._embedding.extract(audio_path, start, end)
            if embedding is not None:
                embeddings.append(embedding)
                valid_segments.append((start, end))
        
        if not embeddings:
            return self._diarize_simple(audio_path, config)
        
        # 聚类
        labels = self._clustering.cluster(
            embeddings,
            config.min_speakers,
            config.max_speakers
        )
        
        # 构建结果
        segments = []
        for (start, end), label in zip(valid_segments, labels):
            segments.append(SpeakerSegment(
                speaker_id=label,
                start=start,
                end=end,
                confidence=0.9
            ))
        
        # 按时间排序
        segments.sort(key=lambda s: s.start)
        
        # 合并相邻的同说话人片段
        merged = self._merge_segments(segments)
        
        # 构建说话人信息
        speakers = self._build_speaker_info(merged)
        
        return DiarizationResult(
            segments=merged,
            speakers=speakers,
            num_speakers=len(speakers),
            duration=duration
        )
    
    def _diarize_pyannote(
        self,
        audio_path: Path,
        config: DiarizationConfig
    ) -> DiarizationResult:
        """使用 pyannote.audio 进行说话人分离"""
        try:
            from pyannote.audio import Pipeline
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=os.environ.get("HUGGINGFACE_TOKEN")
            )
            
            if config.use_gpu:
                import torch
                pipeline = pipeline.to(torch.device("cuda"))
            
            # 执行分离
            diarization = pipeline(str(audio_path))
            
            # 解析结果
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_id = int(speaker.split('_')[1]) if '_' in speaker else 0
                segments.append(SpeakerSegment(
                    speaker_id=speaker_id,
                    start=turn.start,
                    end=turn.end,
                    confidence=0.95
                ))
            
            # 获取时长
            duration = segments[-1].end if segments else 0.0
            
            # 构建说话人信息
            speakers = self._build_speaker_info(segments)
            
            return DiarizationResult(
                segments=segments,
                speakers=speakers,
                num_speakers=len(speakers),
                duration=duration
            )
            
        except ImportError:
            # 回退到聚类方法
            return self._diarize_clustering(audio_path, config)
    
    def _merge_segments(
        self,
        segments: List[SpeakerSegment],
        gap_threshold: float = 0.5
    ) -> List[SpeakerSegment]:
        """合并相邻的同说话人片段"""
        if not segments:
            return []
        
        merged = [segments[0]]
        
        for seg in segments[1:]:
            last = merged[-1]
            
            if (seg.speaker_id == last.speaker_id and
                seg.start - last.end < gap_threshold):
                # 合并
                merged[-1] = SpeakerSegment(
                    speaker_id=last.speaker_id,
                    start=last.start,
                    end=seg.end,
                    confidence=max(last.confidence, seg.confidence)
                )
            else:
                merged.append(seg)
        
        return merged
    
    def _build_speaker_info(
        self,
        segments: List[SpeakerSegment]
    ) -> Dict[int, SpeakerInfo]:
        """构建说话人信息"""
        speaker_data = defaultdict(lambda: {
            'total_duration': 0.0,
            'segment_count': 0,
            'confidences': []
        })
        
        for seg in segments:
            data = speaker_data[seg.speaker_id]
            data['total_duration'] += seg.duration
            data['segment_count'] += 1
            data['confidences'].append(seg.confidence)
        
        speakers = {}
        for speaker_id, data in speaker_data.items():
            avg_conf = sum(data['confidences']) / len(data['confidences'])
            speakers[speaker_id] = SpeakerInfo(
                speaker_id=speaker_id,
                total_duration=data['total_duration'],
                segment_count=data['segment_count'],
                average_confidence=avg_conf,
                label=f"Speaker {speaker_id + 1}"
            )
        
        return speakers
    
    def save_rttm(
        self,
        result: DiarizationResult,
        output_path: Union[str, Path]
    ) -> bool:
        """
        保存为 RTTM 格式
        
        Args:
            result: 分离结果
            output_path: 输出路径
            
        Returns:
            是否成功
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(result.to_rttm())
        
        return True
    
    def visualize(
        self,
        result: DiarizationResult,
        output_path: Union[str, Path]
    ) -> bool:
        """
        可视化分离结果
        
        Args:
            result: 分离结果
            output_path: 输出路径
            
        Returns:
            是否成功
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            
            fig, ax = plt.subplots(figsize=(12, 3))
            
            colors = plt.cm.tab10.colors
            
            for seg in result.segments:
                color = colors[seg.speaker_id % len(colors)]
                rect = mpatches.Rectangle(
                    (seg.start, seg.speaker_id - 0.4),
                    seg.duration, 0.8,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=0.5
                )
                ax.add_patch(rect)
            
            ax.set_xlim(0, result.duration)
            ax.set_ylim(-0.5, result.num_speakers - 0.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Speaker')
            ax.set_yticks(range(result.num_speakers))
            ax.set_yticklabels([f'Speaker {i+1}' for i in range(result.num_speakers)])
            ax.set_title('Speaker Diarization')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close()
            
            return True
            
        except ImportError:
            return False
    
    def assign_speaker_labels(
        self,
        result: DiarizationResult,
        labels: Dict[int, str]
    ) -> DiarizationResult:
        """
        分配说话人标签
        
        Args:
            result: 分离结果
            labels: 说话人ID到标签的映射
            
        Returns:
            更新后的结果
        """
        for speaker_id, label in labels.items():
            if speaker_id in result.speakers:
                result.speakers[speaker_id].label = label
        
        return result
