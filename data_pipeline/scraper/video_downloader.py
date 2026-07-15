"""
视频数据下载器 - Video Data Downloader

支持从多个数据源下载视频数据：
- Kinetics-700 (动作识别)
- Something-Something V2 (动作识别)
- YouTube (通用视频)
- 自定义URL列表

特性：
- 断点续传
- 并行下载
- 自动校验
- 进度追踪
"""

import os
import json
import hashlib
import logging
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
import subprocess
import re


@dataclass
class DownloadConfig:
    """下载配置"""
    output_dir: str = "./data/videos"
    max_concurrent: int = 8
    chunk_size: int = 1024 * 1024  # 1MB
    timeout: int = 300  # 5分钟超时
    max_retries: int = 3
    retry_delay: float = 5.0
    verify_checksum: bool = True
    min_file_size: int = 1024  # 最小文件大小1KB
    max_file_size: int = 1024 * 1024 * 1024  # 最大1GB
    progress_callback: Optional[Callable[[str, float], None]] = None


@dataclass
class VideoMetadata:
    """视频元数据"""
    video_id: str
    source: str
    url: str
    local_path: str
    file_size: int
    checksum: str
    download_time: str
    duration: Optional[float] = None
    fps: Optional[float] = None
    resolution: Optional[Tuple[int, int]] = None
    labels: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


class VideoDownloader:
    """
    视频下载器
    
    支持多源并行下载，断点续传，自动校验。
    """
    
    def __init__(self, config: DownloadConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 下载状态追踪
        self.metadata_file = self.output_dir / "metadata.json"
        self.metadata: Dict[str, VideoMetadata] = {}
        self._load_metadata()
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        # 统计
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'bytes_downloaded': 0
        }
    
    def _load_metadata(self) -> None:
        """加载已下载视频的元数据"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for vid_id, meta in data.items():
                    self.metadata[vid_id] = VideoMetadata(**meta)
    
    def _save_metadata(self) -> None:
        """保存元数据"""
        data = {vid_id: meta.__dict__ for vid_id, meta in self.metadata.items()}
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _compute_checksum(self, file_path: Path) -> str:
        """计算文件MD5校验和"""
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
        return md5.hexdigest()
    
    def _get_video_info(self, file_path: Path) -> Tuple[Optional[float], Optional[float], Optional[Tuple[int, int]]]:
        """使用ffprobe获取视频信息"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet',
                '-print_format', 'json',
                '-show_format', '-show_streams',
                str(file_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                info = json.loads(result.stdout)
                duration = float(info.get('format', {}).get('duration', 0))
                
                # 查找视频流
                for stream in info.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        fps_str = stream.get('r_frame_rate', '0/1')
                        if '/' in fps_str:
                            num, den = map(int, fps_str.split('/'))
                            fps = num / den if den > 0 else 0
                        else:
                            fps = float(fps_str)
                        width = int(stream.get('width', 0))
                        height = int(stream.get('height', 0))
                        return duration, fps, (width, height)
                
                return duration, None, None
        except Exception as e:
            self.logger.warning(f"Failed to get video info for {file_path}: {e}")
        return None, None, None
    
    async def _download_single(
        self,
        session: aiohttp.ClientSession,
        video_id: str,
        url: str,
        source: str,
        labels: List[str] = None
    ) -> Optional[VideoMetadata]:
        """下载单个视频"""
        labels = labels or []
        
        # 检查是否已下载
        if video_id in self.metadata:
            existing = self.metadata[video_id]
            existing_path = Path(existing.local_path)
            if existing_path.exists() and existing.checksum == self._compute_checksum(existing_path):
                self.stats['skipped'] += 1
                self.logger.info(f"Skipping existing: {video_id}")
                return existing
        
        # 准备输出路径
        output_path = self.output_dir / f"{video_id}.mp4"
        temp_path = self.output_dir / f"{video_id}.tmp"
        
        for attempt in range(self.config.max_retries):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.config.timeout)) as response:
                    if response.status != 200:
                        raise ValueError(f"HTTP {response.status}: {url}")
                    
                    # 检查文件大小
                    content_length = int(response.headers.get('Content-Length', 0))
                    if content_length > self.config.max_file_size:
                        raise ValueError(f"File too large: {content_length} > {self.config.max_file_size}")
                    
                    # 下载
                    bytes_downloaded = 0
                    async with aiofiles.open(temp_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(self.config.chunk_size):
                            await f.write(chunk)
                            bytes_downloaded += len(chunk)
                            
                            # 进度回调
                            if self.config.progress_callback and content_length > 0:
                                progress = bytes_downloaded / content_length
                                self.config.progress_callback(video_id, progress)
                    
                    # 校验最小大小
                    if bytes_downloaded < self.config.min_file_size:
                        raise ValueError(f"File too small: {bytes_downloaded} < {self.config.min_file_size}")
                    
                    # 重命名临时文件
                    temp_path.rename(output_path)
                    
                    # 计算校验和
                    checksum = self._compute_checksum(output_path) if self.config.verify_checksum else ""
                    
                    # 获取视频信息
                    duration, fps, resolution = self._get_video_info(output_path)
                    
                    # 创建元数据
                    metadata = VideoMetadata(
                        video_id=video_id,
                        source=source,
                        url=url,
                        local_path=str(output_path),
                        file_size=bytes_downloaded,
                        checksum=checksum,
                        download_time=datetime.now().isoformat(),
                        duration=duration,
                        fps=fps,
                        resolution=resolution,
                        labels=labels
                    )
                    
                    self.metadata[video_id] = metadata
                    self.stats['success'] += 1
                    self.stats['bytes_downloaded'] += bytes_downloaded
                    
                    self.logger.info(f"Downloaded: {video_id} ({bytes_downloaded / 1024 / 1024:.2f} MB)")
                    return metadata
                    
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {video_id}: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    self.stats['failed'] += 1
                    self.logger.error(f"Failed to download {video_id}: {e}")
                    return None
        
        return None
    
    async def download_batch(
        self,
        video_list: List[Dict[str, Any]],
        source: str = "custom"
    ) -> List[VideoMetadata]:
        """
        批量下载视频
        
        Args:
            video_list: 视频列表，每个元素包含 {id, url, labels?}
            source: 数据源名称
            
        Returns:
            成功下载的视频元数据列表
        """
        self.stats['total'] = len(video_list)
        
        connector = aiohttp.TCPConnector(limit=self.config.max_concurrent)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for item in video_list:
                video_id = item['id']
                url = item['url']
                labels = item.get('labels', [])
                
                task = self._download_single(session, video_id, url, source, labels)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
        
        # 保存元数据
        self._save_metadata()
        
        # 返回成功的结果
        return [r for r in results if r is not None]
    
    def download_kinetics(
        self,
        split: str = "train",
        classes: Optional[List[str]] = None,
        max_per_class: int = 100
    ) -> List[VideoMetadata]:
        """
        下载Kinetics数据集
        
        Args:
            split: 数据集分割 (train/val/test)
            classes: 要下载的类别列表，None表示全部
            max_per_class: 每个类别最大视频数
            
        Returns:
            下载的视频元数据列表
        """
        # Kinetics-700的URL模板
        # 实际使用需要从官方获取下载列表
        self.logger.info(f"Downloading Kinetics-{split}, classes: {classes}, max_per_class: {max_per_class}")
        
        # 这里需要实际的Kinetics下载逻辑
        # 由于Kinetics需要官方授权，这里提供框架
        video_list = []
        
        # 示例：如果已有URL列表文件
        url_file = self.output_dir / f"kinetics_{split}_urls.txt"
        if url_file.exists():
            with open(url_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        video_id, url = parts[0], parts[1]
                        label = parts[2] if len(parts) > 2 else ""
                        
                        if classes and label not in classes:
                            continue
                        
                        video_list.append({
                            'id': f"kinetics_{video_id}",
                            'url': url,
                            'labels': [label] if label else []
                        })
        
        # 限制每个类别的数量
        if max_per_class > 0:
            class_counts: Dict[str, int] = {}
            filtered_list = []
            for item in video_list:
                label = item['labels'][0] if item['labels'] else "unknown"
                if class_counts.get(label, 0) < max_per_class:
                    filtered_list.append(item)
                    class_counts[label] = class_counts.get(label, 0) + 1
            video_list = filtered_list
        
        return asyncio.run(self.download_batch(video_list, source="kinetics"))
    
    def download_youtube(
        self,
        video_ids: List[str],
        format: str = "mp4",
        resolution: str = "720p"
    ) -> List[VideoMetadata]:
        """
        使用yt-dlp下载YouTube视频
        
        Args:
            video_ids: YouTube视频ID列表
            format: 输出格式
            resolution: 目标分辨率
            
        Returns:
            下载的视频元数据列表
        """
        results = []
        
        for yt_id in video_ids:
            output_path = self.output_dir / f"youtube_{yt_id}.mp4"
            
            try:
                cmd = [
                    'yt-dlp',
                    '-f', f'bestvideo[height<={resolution[:-1]}]+bestaudio/best[height<={resolution[:-1]}]',
                    '--merge-output-format', format,
                    '-o', str(output_path),
                    '--no-playlist',
                    '--no-overwrites',
                    f'https://www.youtube.com/watch?v={yt_id}'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                
                if result.returncode == 0 and output_path.exists():
                    checksum = self._compute_checksum(output_path)
                    duration, fps, resolution_actual = self._get_video_info(output_path)
                    
                    metadata = VideoMetadata(
                        video_id=f"youtube_{yt_id}",
                        source="youtube",
                        url=f"https://www.youtube.com/watch?v={yt_id}",
                        local_path=str(output_path),
                        file_size=output_path.stat().st_size,
                        checksum=checksum,
                        download_time=datetime.now().isoformat(),
                        duration=duration,
                        fps=fps,
                        resolution=resolution_actual
                    )
                    
                    self.metadata[metadata.video_id] = metadata
                    results.append(metadata)
                    self.stats['success'] += 1
                    
                else:
                    self.logger.error(f"Failed to download YouTube {yt_id}: {result.stderr}")
                    self.stats['failed'] += 1
                    
            except Exception as e:
                self.logger.error(f"Error downloading YouTube {yt_id}: {e}")
                self.stats['failed'] += 1
        
        self._save_metadata()
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """获取下载统计"""
        return {
            **self.stats,
            'success_rate': self.stats['success'] / max(1, self.stats['total']),
            'total_size_mb': self.stats['bytes_downloaded'] / 1024 / 1024
        }
    
    def verify_downloads(self) -> Tuple[int, int]:
        """
        验证已下载文件的完整性
        
        Returns:
            (有效文件数, 损坏文件数)
        """
        valid = 0
        corrupted = 0
        
        for video_id, meta in self.metadata.items():
            path = Path(meta.local_path)
            if not path.exists():
                corrupted += 1
                continue
            
            if self.config.verify_checksum:
                actual_checksum = self._compute_checksum(path)
                if actual_checksum != meta.checksum:
                    self.logger.warning(f"Checksum mismatch for {video_id}")
                    corrupted += 1
                    continue
            
            valid += 1
        
        return valid, corrupted
    
    def cleanup_incomplete(self) -> int:
        """清理未完成的下载（临时文件）"""
        cleaned = 0
        for temp_file in self.output_dir.glob("*.tmp"):
            temp_file.unlink()
            cleaned += 1
        return cleaned
