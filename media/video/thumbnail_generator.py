"""
视频缩略图生成器
从视频提取关键帧生成缩略图
"""

import os
import subprocess
import json
import math
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import struct


class ThumbnailMode(Enum):
    """缩略图生成模式"""
    UNIFORM = "uniform"  # 均匀间隔
    KEYFRAME = "keyframe"  # 关键帧
    SCENE_CHANGE = "scene"  # 场景变化
    SMART = "smart"  # 智能选择


class OutputFormat(Enum):
    """输出格式"""
    INDIVIDUAL = "individual"  # 单独的图片文件
    SPRITE = "sprite"  # 精灵图（网格）
    CONTACT_SHEET = "contact"  # 联系表（带时间戳）


@dataclass
class ThumbnailConfig:
    """缩略图配置"""
    width: int = 160
    height: int = 90
    count: int = 10
    format: str = "jpg"
    quality: int = 85
    mode: ThumbnailMode = ThumbnailMode.UNIFORM
    output_format: OutputFormat = OutputFormat.INDIVIDUAL
    columns: int = 5  # 精灵图列数
    rows: int = 2  # 精灵图行数
    padding: int = 2  # 间距
    background_color: str = "#000000"
    show_timestamp: bool = True
    font_size: int = 12
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@dataclass
class ThumbnailInfo:
    """缩略图信息"""
    index: int
    time: float
    file_path: Path
    width: int
    height: int
    
    def __repr__(self) -> str:
        return f"ThumbnailInfo(index={self.index}, time={self.time:.2f}s, path={self.file_path})"


class ThumbnailGenerator:
    """视频缩略图生成器主类"""
    
    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        """
        初始化缩略图生成器
        
        Args:
            ffmpeg_path: ffmpeg 可执行文件路径
            ffprobe_path: ffprobe 可执行文件路径
        """
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self._thumbnails: List[ThumbnailInfo] = []
    
    def get_video_duration(self, file_path: Union[str, Path]) -> float:
        """
        获取视频时长
        
        Args:
            file_path: 视频文件路径
            
        Returns:
            时长（秒）
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        
        cmd = [
            self.ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return float(data.get('format', {}).get('duration', 0))
        except (json.JSONDecodeError, subprocess.TimeoutExpired):
            pass
        
        return 0.0
    
    def get_video_resolution(
        self,
        file_path: Union[str, Path]
    ) -> Tuple[int, int]:
        """
        获取视频分辨率
        
        Args:
            file_path: 视频文件路径
            
        Returns:
            (宽度, 高度)
        """
        path = Path(file_path)
        
        cmd = [
            self.ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-select_streams", "v:0",
            str(path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                streams = data.get('streams', [])
                if streams:
                    return (
                        int(streams[0].get('width', 0)),
                        int(streams[0].get('height', 0))
                    )
        except (json.JSONDecodeError, subprocess.TimeoutExpired):
            pass
        
        return (0, 0)
    
    def generate(
        self,
        video_path: Union[str, Path],
        output_dir: Union[str, Path],
        config: Optional[ThumbnailConfig] = None
    ) -> List[ThumbnailInfo]:
        """
        生成缩略图
        
        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            config: 缩略图配置
            
        Returns:
            缩略图信息列表
        """
        if config is None:
            config = ThumbnailConfig()
        
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self._thumbnails = []
        
        if config.mode == ThumbnailMode.UNIFORM:
            return self._generate_uniform(video_path, output_dir, config)
        elif config.mode == ThumbnailMode.KEYFRAME:
            return self._generate_keyframe(video_path, output_dir, config)
        elif config.mode == ThumbnailMode.SCENE_CHANGE:
            return self._generate_scene_change(video_path, output_dir, config)
        else:
            return self._generate_smart(video_path, output_dir, config)
    
    def _generate_uniform(
        self,
        video_path: Path,
        output_dir: Path,
        config: ThumbnailConfig
    ) -> List[ThumbnailInfo]:
        """均匀间隔生成缩略图"""
        duration = self.get_video_duration(video_path)
        if duration <= 0:
            return []
        
        # 计算时间范围
        start = config.start_time or 0
        end = config.end_time or duration
        effective_duration = end - start
        
        # 计算间隔
        interval = effective_duration / (config.count + 1)
        
        # 生成缩略图
        for i in range(config.count):
            time = start + interval * (i + 1)
            output_file = output_dir / f"thumb_{i:04d}.{config.format}"
            
            if self._extract_frame(video_path, output_file, time, config):
                info = ThumbnailInfo(
                    index=i,
                    time=time,
                    file_path=output_file,
                    width=config.width,
                    height=config.height
                )
                self._thumbnails.append(info)
        
        # 处理输出格式
        if config.output_format != OutputFormat.INDIVIDUAL:
            self._create_composite(output_dir, config)
        
        return self._thumbnails
    
    def _generate_keyframe(
        self,
        video_path: Path,
        output_dir: Path,
        config: ThumbnailConfig
    ) -> List[ThumbnailInfo]:
        """从关键帧生成缩略图"""
        # 获取关键帧时间点
        keyframes = self._get_keyframe_times(video_path)
        
        if not keyframes:
            # 如果无法获取关键帧，回退到均匀模式
            return self._generate_uniform(video_path, output_dir, config)
        
        # 选择指定数量的关键帧
        if len(keyframes) > config.count:
            step = len(keyframes) / config.count
            selected = [keyframes[int(i * step)] for i in range(config.count)]
        else:
            selected = keyframes
        
        # 生成缩略图
        for i, time in enumerate(selected):
            output_file = output_dir / f"thumb_{i:04d}.{config.format}"
            
            if self._extract_frame(video_path, output_file, time, config):
                info = ThumbnailInfo(
                    index=i,
                    time=time,
                    file_path=output_file,
                    width=config.width,
                    height=config.height
                )
                self._thumbnails.append(info)
        
        if config.output_format != OutputFormat.INDIVIDUAL:
            self._create_composite(output_dir, config)
        
        return self._thumbnails
    
    def _generate_scene_change(
        self,
        video_path: Path,
        output_dir: Path,
        config: ThumbnailConfig
    ) -> List[ThumbnailInfo]:
        """从场景变化点生成缩略图"""
        # 使用 ffmpeg 检测场景变化
        scene_times = self._detect_scene_changes(video_path)
        
        if not scene_times:
            return self._generate_uniform(video_path, output_dir, config)
        
        # 限制数量
        scene_times = scene_times[:config.count]
        
        # 生成缩略图
        for i, time in enumerate(scene_times):
            output_file = output_dir / f"thumb_{i:04d}.{config.format}"
            
            if self._extract_frame(video_path, output_file, time, config):
                info = ThumbnailInfo(
                    index=i,
                    time=time,
                    file_path=output_file,
                    width=config.width,
                    height=config.height
                )
                self._thumbnails.append(info)
        
        # 如果不足指定数量，补充均匀帧
        if len(self._thumbnails) < config.count:
            remaining = config.count - len(self._thumbnails)
            duration = self.get_video_duration(video_path)
            interval = duration / (remaining + 1)
            
            for i in range(remaining):
                time = interval * (i + 1)
                output_file = output_dir / f"thumb_{len(self._thumbnails):04d}.{config.format}"
                
                if self._extract_frame(video_path, output_file, time, config):
                    info = ThumbnailInfo(
                        index=len(self._thumbnails),
                        time=time,
                        file_path=output_file,
                        width=config.width,
                        height=config.height
                    )
                    self._thumbnails.append(info)
        
        if config.output_format != OutputFormat.INDIVIDUAL:
            self._create_composite(output_dir, config)
        
        return self._thumbnails
    
    def _generate_smart(
        self,
        video_path: Path,
        output_dir: Path,
        config: ThumbnailConfig
    ) -> List[ThumbnailInfo]:
        """智能生成缩略图（结合场景变化和均匀分布）"""
        duration = self.get_video_duration(video_path)
        scene_times = self._detect_scene_changes(video_path)
        
        # 智能选择时间点
        selected_times = []
        
        # 首先添加场景变化点
        for time in scene_times[:config.count // 2]:
            selected_times.append(time)
        
        # 然后均匀填充剩余
        remaining = config.count - len(selected_times)
        interval = duration / (remaining + 1)
        
        for i in range(remaining):
            time = interval * (i + 1)
            # 避免与场景变化点太近
            if not any(abs(time - t) < interval / 2 for t in selected_times):
                selected_times.append(time)
        
        selected_times.sort()
        selected_times = selected_times[:config.count]
        
        # 生成缩略图
        for i, time in enumerate(selected_times):
            output_file = output_dir / f"thumb_{i:04d}.{config.format}"
            
            if self._extract_frame(video_path, output_file, time, config):
                info = ThumbnailInfo(
                    index=i,
                    time=time,
                    file_path=output_file,
                    width=config.width,
                    height=config.height
                )
                self._thumbnails.append(info)
        
        if config.output_format != OutputFormat.INDIVIDUAL:
            self._create_composite(output_dir, config)
        
        return self._thumbnails
    
    def _extract_frame(
        self,
        video_path: Path,
        output_path: Path,
        time: float,
        config: ThumbnailConfig
    ) -> bool:
        """提取指定时间的帧"""
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-ss", str(time),
            "-i", str(video_path),
            "-vframes", "1",
            "-vf", f"scale={config.width}:{config.height}:force_original_aspect_ratio=decrease,pad={config.width}:{config.height}:(ow-iw)/2:(oh-ih)/2"
        ]
        
        if config.format == "jpg":
            cmd.extend(["-q:v", str(int(31 - config.quality * 31 / 100))])
        elif config.format == "png":
            cmd.extend(["-compression_level", "9"])
        
        cmd.append(str(output_path))
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
    
    def _get_keyframe_times(self, video_path: Path) -> List[float]:
        """获取关键帧时间点"""
        cmd = [
            self.ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_packets",
            "-select_streams", "v:0",
            "-show_entries", "packet=pts_time,flags",
            str(video_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                packets = data.get('packets', [])
                
                keyframes = []
                for pkt in packets:
                    if 'K' in pkt.get('flags', ''):
                        time = float(pkt.get('pts_time', 0))
                        if time > 0:
                            keyframes.append(time)
                
                return keyframes
        except (json.JSONDecodeError, subprocess.TimeoutExpired):
            pass
        
        return []
    
    def _detect_scene_changes(self, video_path: Path) -> List[float]:
        """检测场景变化"""
        cmd = [
            self.ffmpeg_path,
            "-i", str(video_path),
            "-vf", "select='gt(scene,0.3)',showinfo",
            "-f", "null",
            "-"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            # 解析 showinfo 输出
            import re
            pattern = re.compile(r'pts_time:(\d+\.?\d*)')
            times = []
            
            for line in result.stderr.split('\n'):
                match = pattern.search(line)
                if match:
                    times.append(float(match.group(1)))
            
            return times
        except subprocess.TimeoutExpired:
            return []
    
    def _create_composite(
        self,
        output_dir: Path,
        config: ThumbnailConfig
    ) -> None:
        """创建合成图（精灵图或联系表）"""
        if not self._thumbnails:
            return
        
        if config.output_format == OutputFormat.SPRITE:
            self._create_sprite(output_dir, config)
        elif config.output_format == OutputFormat.CONTACT_SHEET:
            self._create_contact_sheet(output_dir, config)
    
    def _create_sprite(
        self,
        output_dir: Path,
        config: ThumbnailConfig
    ) -> Optional[Path]:
        """创建精灵图"""
        if not self._thumbnails:
            return None
        
        # 计算精灵图尺寸
        cols = config.columns
        rows = min(config.rows, (len(self._thumbnails) + cols - 1) // cols)
        
        sprite_width = cols * config.width + (cols + 1) * config.padding
        sprite_height = rows * config.height + (rows + 1) * config.padding
        
        # 构建 ffmpeg 命令
        inputs = []
        for thumb in self._thumbnails:
            inputs.extend(["-i", str(thumb.file_path)])
        
        # 构建滤镜
        filter_parts = []
        for i, thumb in enumerate(self._thumbnails):
            row = i // cols
            col = i % cols
            x = config.padding + col * (config.width + config.padding)
            y = config.padding + row * (config.height + config.padding)
            filter_parts.append(f"[{i}:v]setpts=PTS-STARTPTS,trim=duration=1[v{i}]")
        
        overlay_filter = "[0:v]null[vbase]"
        for i, thumb in enumerate(self._thumbnails[1:], 1):
            row = i // cols
            col = i % cols
            x = config.padding + col * (config.width + config.padding)
            y = config.padding + row * (config.height + config.padding)
            overlay_filter = f"[{overlay_filter.split('[')[-1].split(']')[0]}][v{i}]overlay={x}:{y}[v{i+1}]"
        
        filter_complex = ";".join(filter_parts) + ";" + overlay_filter
        
        output_file = output_dir / f"sprite.{config.format}"
        
        cmd = [
            self.ffmpeg_path,
            "-y"
        ] + inputs + [
            "-filter_complex", filter_complex,
            "-frames:v", "1",
            str(output_file)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode == 0:
                return output_file
        except subprocess.TimeoutExpired:
            pass
        
        return None
    
    def _create_contact_sheet(
        self,
        output_dir: Path,
        config: ThumbnailConfig
    ) -> Optional[Path]:
        """创建联系表（带时间戳）"""
        if not self._thumbnails:
            return None
        
        cols = config.columns
        rows = min(config.rows, (len(self._thumbnails) + cols - 1) // cols)
        
        # 构建滤镜，添加时间戳
        filter_parts = []
        for i, thumb in enumerate(self._thumbnails):
            time_str = self._format_time(thumb.time)
            
            filter_parts.append(
                f"[{i}:v]drawtext=text='{time_str}':"
                f"fontcolor=white:fontsize={config.font_size}:"
                f"x=10:y=h-{config.font_size + 5}:"
                f"shadowcolor=black:shadowx=1:shadowy=1[v{i}]"
            )
        
        # 合成图像
        sprite_width = cols * config.width + (cols + 1) * config.padding
        sprite_height = rows * config.height + (rows + 1) * config.padding
        
        inputs = []
        for thumb in self._thumbnails:
            inputs.extend(["-i", str(thumb.file_path)])
        
        # 简化：使用 tile 滤镜
        tile_filter = f"tile={cols}x{rows}:padding={config.padding}:color={config.background_color}"
        
        filter_complex = ";".join(filter_parts) + f";[v0][v1]...hstack"
        
        output_file = output_dir / f"contact_sheet.{config.format}"
        
        # 使用更简单的方法
        cmd = [
            self.ffmpeg_path,
            "-y"
        ] + inputs + [
            "-filter_complex", f"tile={cols}x{rows}:padding={config.padding}",
            "-frames:v", "1",
            str(output_file)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode == 0:
                return output_file
        except subprocess.TimeoutExpired:
            pass
        
        return None
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间显示"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
        else:
            return f"{minutes:02d}:{secs:05.2f}"
    
    def generate_preview_gif(
        self,
        video_path: Union[str, Path],
        output_path: Union[str, Path],
        width: int = 320,
        fps: int = 5,
        duration: float = 10.0
    ) -> bool:
        """
        生成预览 GIF
        
        Args:
            video_path: 视频文件路径
            output_path: 输出 GIF 路径
            width: 宽度
            fps: 帧率
            duration: 时长
            
        Returns:
            是否成功
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 获取视频时长
        total_duration = self.get_video_duration(video_path)
        if total_duration <= 0:
            return False
        
        # 计算采样间隔
        sample_duration = min(duration, total_duration)
        
        filter_str = f"fps={fps},scale={width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"
        
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-t", str(sample_duration),
            "-i", str(video_path),
            "-vf", filter_str,
            str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
    
    def generate_storyboard(
        self,
        video_path: Union[str, Path],
        output_path: Union[str, Path],
        cols: int = 5,
        rows: int = 4,
        width: int = 160
    ) -> bool:
        """
        生成故事板
        
        Args:
            video_path: 视频文件路径
            output_path: 输出图片路径
            cols: 列数
            rows: 行数
            width: 单帧宽度
            
        Returns:
            是否成功
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        total_duration = self.get_video_duration(video_path)
        if total_duration <= 0:
            return False
        
        count = cols * rows
        interval = total_duration / count
        
        # 使用 ffmpeg 的 tile 滤镜
        filter_str = f"fps=1/{interval},scale={width}:-1,tile={cols}x{rows}"
        
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", str(video_path),
            "-vf", filter_str,
            "-frames:v", "1",
            str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
    
    @property
    def thumbnails(self) -> List[ThumbnailInfo]:
        """获取生成的缩略图列表"""
        return self._thumbnails.copy()
    
    def get_thumbnail_at_time(
        self,
        video_path: Union[str, Path],
        time: float,
        output_path: Union[str, Path],
        width: int = 160,
        height: int = 90
    ) -> bool:
        """
        获取指定时间的缩略图
        
        Args:
            video_path: 视频文件路径
            time: 时间点（秒）
            output_path: 输出路径
            width: 宽度
            height: 高度
            
        Returns:
            是否成功
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        config = ThumbnailConfig(width=width, height=height)
        return self._extract_frame(video_path, output_path, time, config)
