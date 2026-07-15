"""
视频帧采样器
实现均匀采样、关键帧采样和场景检测采样
"""
from typing import Optional, List, Dict, Any, Tuple
import math
import random


class FrameSampler:
    """帧采样器基类"""
    
    def __init__(self, num_frames: int = 16):
        self.num_frames = num_frames
    
    def sample(self, total_frames: int) -> List[int]:
        """均匀采样作为默认实现
        
        Args:
            total_frames: 视频总帧数
            
        Returns:
            采样的帧索引列表
        """
        if total_frames <= 0:
            return []
        
        if total_frames <= self.num_frames:
            return list(range(total_frames))
        
        interval = total_frames / self.num_frames
        indices = [int(i * interval) for i in range(self.num_frames)]
        return indices


class UniformSampler(FrameSampler):
    """均匀采样器
    
    在视频中等间隔采样帧
    """
    
    def __init__(self, num_frames: int = 16, random_shift: bool = False):
        super().__init__(num_frames)
        self.random_shift = random_shift
    
    def sample(self, total_frames: int) -> List[int]:
        """
        均匀采样
        
        Args:
            total_frames: 视频总帧数
        
        Returns:
            采样的帧索引列表
        """
        if total_frames <= 0:
            return []
        
        if total_frames <= self.num_frames:
            return list(range(total_frames))
        
        # 计算采样间隔
        interval = total_frames / self.num_frames
        
        # 随机偏移
        shift = 0
        if self.random_shift:
            shift = random.uniform(0, interval)
        
        # 采样
        indices = []
        for i in range(self.num_frames):
            idx = int(shift + i * interval)
            idx = min(idx, total_frames - 1)
            indices.append(idx)
        
        return indices


class KeyFrameSampler(FrameSampler):
    """关键帧采样器
    
    基于帧间差异选择关键帧
    """
    
    def __init__(self, num_frames: int = 16, threshold: float = 0.3):
        super().__init__(num_frames)
        self.threshold = threshold
    
    def _compute_frame_difference(self, frame1: List[List[List[float]]], 
                                   frame2: List[List[List[float]]]) -> float:
        """计算两帧之间的差异"""
        if not frame1 or not frame2:
            return 0.0
        
        diff = 0.0
        count = 0
        
        for c in range(min(len(frame1), len(frame2))):
            for h in range(min(len(frame1[c]), len(frame2[c]))):
                for w in range(min(len(frame1[c][h]), len(frame2[c][h]))):
                    diff += abs(frame1[c][h][w] - frame2[c][h][w])
                    count += 1
        
        return diff / count if count > 0 else 0.0
    
    def sample_with_scores(self, frames: List[List[List[List[float]]]], 
                           num_frames: Optional[int] = None) -> List[int]:
        """
        基于帧差异采样
        
        Args:
            frames: 视频帧列表
            num_frames: 目标帧数
        
        Returns:
            采样的帧索引列表
        """
        num_frames = num_frames or self.num_frames
        total_frames = len(frames)
        
        if total_frames <= num_frames:
            return list(range(total_frames))
        
        # 计算每帧与前一帧的差异
        scores = [0.0]  # 第一帧分数为0
        for i in range(1, total_frames):
            diff = self._compute_frame_difference(frames[i - 1], frames[i])
            scores.append(diff)
        
        # 选择差异最大的帧作为关键帧
        scored_indices = list(range(total_frames))
        scored_indices.sort(key=lambda x: scores[x], reverse=True)
        
        # 取top-k并排序
        selected = sorted(scored_indices[:num_frames])
        
        return selected
    
    def sample(self, total_frames: int) -> List[int]:
        """均匀采样作为后备"""
        if total_frames <= self.num_frames:
            return list(range(total_frames))
        
        interval = total_frames / self.num_frames
        indices = [int(i * interval) for i in range(self.num_frames)]
        return indices


class SceneDetectionSampler(FrameSampler):
    """场景检测采样器
    
    基于场景边界进行采样
    """
    
    def __init__(self, num_frames: int = 16, scene_threshold: float = 0.5,
                 min_scene_frames: int = 5):
        super().__init__(num_frames)
        self.scene_threshold = scene_threshold
        self.min_scene_frames = min_scene_frames
    
    def _compute_histogram(self, frame: List[List[List[float]]]) -> List[float]:
        """计算帧的直方图"""
        # 简化的直方图计算
        bins = [0.0] * 64
        
        if not frame:
            return bins
        
        for c in range(len(frame)):
            for h in range(len(frame[c])):
                for w in range(len(frame[c][h])):
                    val = frame[c][h][w]
                    bin_idx = int(val * 63)
                    bin_idx = max(0, min(63, bin_idx))
                    bins[bin_idx] += 1
        
        # 归一化
        total = sum(bins)
        if total > 0:
            bins = [b / total for b in bins]
        
        return bins
    
    def _histogram_difference(self, hist1: List[float], hist2: List[float]) -> float:
        """计算直方图差异"""
        return sum(abs(h1 - h2) for h1, h2 in zip(hist1, hist2))
    
    def detect_scenes(self, frames: List[List[List[List[float]]]]) -> List[int]:
        """
        检测场景边界
        
        Args:
            frames: 视频帧列表
        
        Returns:
            场景边界帧索引列表
        """
        if len(frames) < 2:
            return [0]
        
        # 计算每帧的直方图
        histograms = [self._compute_histogram(f) for f in frames]
        
        # 检测场景边界
        boundaries = [0]
        
        for i in range(1, len(histograms)):
            diff = self._histogram_difference(histograms[i - 1], histograms[i])
            
            if diff > self.scene_threshold:
                # 确保场景足够长
                if i - boundaries[-1] >= self.min_scene_frames:
                    boundaries.append(i)
        
        return boundaries
    
    def sample_with_scenes(self, frames: List[List[List[List[float]]]], 
                           num_frames: Optional[int] = None) -> List[int]:
        """
        基于场景采样
        
        Args:
            frames: 视频帧列表
            num_frames: 目标帧数
        
        Returns:
            采样的帧索引列表
        """
        num_frames = num_frames or self.num_frames
        total_frames = len(frames)
        
        if total_frames <= num_frames:
            return list(range(total_frames))
        
        # 检测场景边界
        boundaries = self.detect_scenes(frames)
        boundaries.append(total_frames)
        
        # 计算每个场景的帧数
        scene_lengths = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]
        
        # 按场景长度分配采样帧数
        total_length = sum(scene_lengths)
        frames_per_scene = [max(1, int(num_frames * length / total_length)) for length in scene_lengths]
        
        # 在每个场景内均匀采样
        sampled_indices = []
        for i, (start, n) in enumerate(zip(boundaries[:-1], frames_per_scene)):
            end = boundaries[i + 1]
            scene_length = end - start
            
            if scene_length <= n:
                sampled_indices.extend(range(start, end))
            else:
                interval = scene_length / n
                for j in range(n):
                    idx = start + int(j * interval)
                    sampled_indices.append(min(idx, end - 1))
        
        # 调整到目标帧数
        if len(sampled_indices) > num_frames:
            step = len(sampled_indices) / num_frames
            sampled_indices = [sampled_indices[int(i * step)] for i in range(num_frames)]
        
        return sorted(sampled_indices)
    
    def sample(self, total_frames: int) -> List[int]:
        """均匀采样作为后备"""
        if total_frames <= self.num_frames:
            return list(range(total_frames))
        
        interval = total_frames / self.num_frames
        indices = [int(i * interval) for i in range(self.num_frames)]
        return indices


class AdaptiveSampler(FrameSampler):
    """自适应采样器
    
    根据视频动态特性自适应采样
    """
    
    def __init__(self, num_frames: int = 16, min_interval: int = 1,
                 max_interval: Optional[int] = None):
        super().__init__(num_frames)
        self.min_interval = min_interval
        self.max_interval = max_interval
    
    def _compute_motion_score(self, frames: List[List[List[List[float]]]], 
                               start: int, end: int) -> float:
        """计算帧序列的运动分数"""
        if end <= start + 1:
            return 0.0
        
        total_diff = 0.0
        count = 0
        
        for i in range(start, end - 1):
            for c in range(min(len(frames[i]), len(frames[i + 1]))):
                for h in range(min(len(frames[i][c]), len(frames[i + 1][c]))):
                    for w in range(min(len(frames[i][c][h]), len(frames[i + 1][c][h]))):
                        total_diff += abs(frames[i][c][h][w] - frames[i + 1][c][h][w])
                        count += 1
        
        return total_diff / count if count > 0 else 0.0
    
    def sample_adaptive(self, frames: List[List[List[List[float]]]], 
                        num_frames: Optional[int] = None) -> List[int]:
        """
        自适应采样
        
        Args:
            frames: 视频帧列表
            num_frames: 目标帧数
        
        Returns:
            采样的帧索引列表
        """
        num_frames = num_frames or self.num_frames
        total_frames = len(frames)
        
        if total_frames <= num_frames:
            return list(range(total_frames))
        
        # 将视频分成多个片段
        segment_size = total_frames / num_frames
        
        sampled_indices = []
        for i in range(num_frames):
            start = int(i * segment_size)
            end = int((i + 1) * segment_size)
            end = min(end, total_frames)
            
            if start >= end:
                start = end - 1
            
            # 在片段中选择运动最大的帧
            if end - start <= 2:
                sampled_indices.append(start)
            else:
                # 简化：选择片段中间帧
                mid = (start + end) // 2
                sampled_indices.append(mid)
        
        return sampled_indices
    
    def sample(self, total_frames: int) -> List[int]:
        """均匀采样作为后备"""
        if total_frames <= self.num_frames:
            return list(range(total_frames))
        
        interval = total_frames / self.num_frames
        indices = [int(i * interval) for i in range(self.num_frames)]
        return indices


class MultiScaleSampler(FrameSampler):
    """多尺度采样器
    
    在不同时间尺度上采样
    """
    
    def __init__(self, num_frames: int = 16, scales: Optional[List[int]] = None):
        super().__init__(num_frames)
        self.scales = scales or [1, 2, 4]
    
    def sample(self, total_frames: int) -> List[int]:
        """
        多尺度采样
        
        Args:
            total_frames: 视频总帧数
        
        Returns:
            采样的帧索引列表
        """
        if total_frames <= self.num_frames:
            return list(range(total_frames))
        
        all_indices = set()
        
        frames_per_scale = self.num_frames // len(self.scales)
        
        for scale in self.scales:
            # 在当前尺度上采样
            interval = total_frames / (frames_per_scale * scale)
            for i in range(frames_per_scale):
                idx = int(i * interval * scale) % total_frames
                all_indices.add(idx)
        
        # 补充到目标帧数
        indices = sorted(all_indices)
        
        if len(indices) < self.num_frames:
            # 均匀补充
            remaining = self.num_frames - len(indices)
            uniform_sampler = UniformSampler(remaining)
            additional = uniform_sampler.sample(total_frames)
            indices = sorted(set(indices + additional))
        
        return indices[:self.num_frames]


def create_frame_sampler(sampler_type: str = 'uniform', 
                         num_frames: int = 16) -> FrameSampler:
    """
    创建帧采样器
    
    Args:
        sampler_type: 采样器类型 ('uniform', 'keyframe', 'scene', 'adaptive', 'multiscale')
        num_frames: 目标帧数
    
    Returns:
        帧采样器
    """
    samplers = {
        'uniform': UniformSampler(num_frames),
        'keyframe': KeyFrameSampler(num_frames),
        'scene': SceneDetectionSampler(num_frames),
        'adaptive': AdaptiveSampler(num_frames),
        'multiscale': MultiScaleSampler(num_frames)
    }
    
    return samplers.get(sampler_type, UniformSampler(num_frames))
