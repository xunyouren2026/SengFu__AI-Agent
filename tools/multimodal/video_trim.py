"""
Video Trim Tool - 视频剪辑工具
视频剪辑和处理
"""

import json
import subprocess
import os
import re
import math
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


class VideoFormat(Enum):
    """视频格式枚举"""
    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    MKV = "mkv"
    WEBM = "webm"
    GIF = "gif"


class VideoCodec(Enum):
    """视频编码枚举"""
    H264 = "libx264"
    H265 = "libx265"
    VP9 = "libvpx-vp9"
    AV1 = "libaom-av1"
    COPY = "copy"


class AudioCodec(Enum):
    """音频编码枚举"""
    AAC = "aac"
    MP3 = "libmp3lame"
    OPUS = "libopus"
    COPY = "copy"


@dataclass
class VideoInfo:
    """视频信息"""
    file_path: str
    duration: float  # 秒
    width: int
    height: int
    fps: float
    bitrate: int
    codec: str
    audio_codec: Optional[str] = None
    audio_sample_rate: Optional[int] = None
    audio_channels: Optional[int] = None
    file_size: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "duration": self.duration,
            "duration_formatted": self._format_duration(self.duration),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "bitrate": self.bitrate,
            "codec": self.codec,
            "audio_codec": self.audio_codec,
            "audio_sample_rate": self.audio_sample_rate,
            "audio_channels": self.audio_channels,
            "file_size": self.file_size,
            "resolution": f"{self.width}x{self.height}"
        }
    
    def _format_duration(self, seconds: float) -> str:
        """格式化时长"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
        else:
            return f"{minutes:02d}:{secs:05.2f}"


@dataclass
class TrimConfig:
    """剪辑配置"""
    output_format: VideoFormat = VideoFormat.MP4
    video_codec: VideoCodec = VideoCodec.H264
    audio_codec: AudioCodec = AudioCodec.AAC
    quality: int = 23  # CRF值，越小质量越高
    preset: str = "medium"  # 编码预设
    audio_bitrate: str = "128k"
    keep_audio: bool = True
    copy_codec: bool = False  # 直接复制流，不重新编码
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"


@dataclass
class TrimResult:
    """剪辑结果"""
    success: bool
    output_path: Optional[str] = None
    output_info: Optional[VideoInfo] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output_path": self.output_path,
            "output_info": self.output_info.to_dict() if self.output_info else None,
            "error_message": self.error_message,
            "processing_time": self.processing_time
        }


class VideoTrimmer:
    """视频剪辑工具"""
    
    def __init__(self, config: Optional[TrimConfig] = None):
        self.config = config or TrimConfig()
        self._check_ffmpeg()
    
    def _check_ffmpeg(self) -> bool:
        """检查FFmpeg是否可用"""
        try:
            subprocess.run(
                [self.config.ffmpeg_path, "-version"],
                capture_output=True,
                timeout=10
            )
            return True
        except Exception:
            logger.warning("FFmpeg not found or not executable")
            return False
    
    def get_info(self, video_path: str) -> Optional[VideoInfo]:
        """获取视频信息"""
        try:
            cmd = [
                self.config.ffprobe_path,
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            data = json.loads(result.stdout)
            
            # 查找视频流
            video_stream = None
            audio_stream = None
            
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video" and video_stream is None:
                    video_stream = stream
                elif stream.get("codec_type") == "audio" and audio_stream is None:
                    audio_stream = stream
            
            if not video_stream:
                return None
            
            # 解析帧率
            fps_str = video_stream.get("r_frame_rate", "0/1")
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den) if float(den) > 0 else 0
            else:
                fps = float(fps_str)
            
            format_info = data.get("format", {})
            
            return VideoInfo(
                file_path=video_path,
                duration=float(format_info.get("duration", 0)),
                width=int(video_stream.get("width", 0)),
                height=int(video_stream.get("height", 0)),
                fps=fps,
                bitrate=int(format_info.get("bit_rate", 0)),
                codec=video_stream.get("codec_name", ""),
                audio_codec=audio_stream.get("codec_name") if audio_stream else None,
                audio_sample_rate=int(audio_stream.get("sample_rate", 0)) if audio_stream else None,
                audio_channels=int(audio_stream.get("channels", 0)) if audio_stream else None,
                file_size=int(format_info.get("size", 0))
            )
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            return None
    
    def trim(self, video_path: str, output_path: str,
             start_time: float = 0,
             end_time: Optional[float] = None,
             duration: Optional[float] = None,
             **kwargs) -> TrimResult:
        """剪辑视频"""
        # 获取视频信息
        info = self.get_info(video_path)
        if not info:
            return TrimResult(
                success=False,
                error_message="Failed to read video info"
            )
        
        # 计算结束时间
        if end_time is None:
            if duration is not None:
                end_time = start_time + duration
            else:
                end_time = info.duration
        
        # 验证时间范围
        if start_time < 0:
            start_time = 0
        if end_time > info.duration:
            end_time = info.duration
        
        if start_time >= end_time:
            return TrimResult(
                success=False,
                error_message="Invalid time range: start >= end"
            )
        
        # 构建FFmpeg命令
        cmd = self._build_trim_command(video_path, output_path, start_time, end_time)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                output_info = self.get_info(output_path)
                return TrimResult(
                    success=True,
                    output_path=output_path,
                    output_info=output_info
                )
            else:
                return TrimResult(
                    success=False,
                    error_message=result.stderr or "FFmpeg error"
                )
        except subprocess.TimeoutExpired:
            return TrimResult(
                success=False,
                error_message="Processing timeout"
            )
        except Exception as e:
            return TrimResult(
                success=False,
                error_message=str(e)
            )
    
    def _build_trim_command(self, input_path: str, output_path: str,
                            start_time: float, end_time: float) -> List[str]:
        """构建剪辑命令"""
        cmd = [
            self.config.ffmpeg_path,
            "-y",  # 覆盖输出文件
            "-ss", str(start_time),
            "-i", input_path,
            "-t", str(end_time - start_time)
        ]
        
        if self.config.copy_codec:
            cmd.extend(["-c", "copy"])
        else:
            # 视频编码
            cmd.extend([
                "-c:v", self.config.video_codec.value,
                "-crf", str(self.config.quality),
                "-preset", self.config.preset
            ])
            
            # 音频编码
            if self.config.keep_audio:
                cmd.extend([
                    "-c:a", self.config.audio_codec.value,
                    "-b:a", self.config.audio_bitrate
                ])
            else:
                cmd.extend(["-an"])  # 移除音频
        
        cmd.append(output_path)
        
        return cmd
    
    def extract_clip(self, video_path: str, output_path: str,
                     start_time: float, duration: float,
                     **kwargs) -> TrimResult:
        """提取片段"""
        return self.trim(video_path, output_path, start_time=start_time,
                        duration=duration, **kwargs)
    
    def split(self, video_path: str, output_dir: str,
              segment_duration: float,
              output_prefix: str = "segment_",
              **kwargs) -> List[TrimResult]:
        """分割视频"""
        info = self.get_info(video_path)
        if not info:
            return [TrimResult(success=False, error_message="Failed to read video info")]
        
        results = []
        num_segments = math.ceil(info.duration / segment_duration)
        
        for i in range(num_segments):
            start = i * segment_duration
            output_path = os.path.join(output_dir, f"{output_prefix}{i+1:03d}.{self.config.output_format.value}")
            
            result = self.trim(video_path, output_path, start_time=start,
                             duration=segment_duration, **kwargs)
            results.append(result)
        
        return results
    
    def concat(self, video_paths: List[str], output_path: str,
               **kwargs) -> TrimResult:
        """合并视频"""
        if not video_paths:
            return TrimResult(success=False, error_message="No videos to concatenate")
        
        # 创建临时文件列表
        import tempfile
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                for path in video_paths:
                    f.write(f"file '{os.path.abspath(path)}'\n")
                list_file = f.name
            
            cmd = [
                self.config.ffmpeg_path,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", list_file
            ]
            
            if self.config.copy_codec:
                cmd.extend(["-c", "copy"])
            else:
                cmd.extend([
                    "-c:v", self.config.video_codec.value,
                    "-crf", str(self.config.quality),
                    "-preset", self.config.preset
                ])
            
            cmd.append(output_path)
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            os.unlink(list_file)
            
            if result.returncode == 0:
                output_info = self.get_info(output_path)
                return TrimResult(
                    success=True,
                    output_path=output_path,
                    output_info=output_info
                )
            else:
                return TrimResult(
                    success=False,
                    error_message=result.stderr
                )
        except Exception as e:
            return TrimResult(success=False, error_message=str(e))
    
    def extract_audio(self, video_path: str, output_path: str,
                      audio_codec: AudioCodec = AudioCodec.AAC,
                      audio_bitrate: str = "192k") -> TrimResult:
        """提取音频"""
        cmd = [
            self.config.ffmpeg_path,
            "-y",
            "-i", video_path,
            "-vn",  # 不包含视频
            "-c:a", audio_codec.value,
            "-b:a", audio_bitrate,
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                return TrimResult(
                    success=True,
                    output_path=output_path
                )
            else:
                return TrimResult(
                    success=False,
                    error_message=result.stderr
                )
        except Exception as e:
            return TrimResult(success=False, error_message=str(e))
    
    def extract_frames(self, video_path: str, output_dir: str,
                       fps: Optional[float] = None,
                       start_time: float = 0,
                       end_time: Optional[float] = None,
                       output_pattern: str = "frame_%04d.png") -> Tuple[bool, int, str]:
        """提取帧"""
        cmd = [
            self.config.ffmpeg_path,
            "-y",
            "-ss", str(start_time)
        ]
        
        if end_time is not None:
            cmd.extend(["-t", str(end_time - start_time)])
        
        cmd.extend(["-i", video_path])
        
        if fps is not None:
            cmd.extend(["-vf", f"fps={fps}"])
        
        output_path = os.path.join(output_dir, output_pattern)
        cmd.append(output_path)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                # 统计提取的帧数
                frames = [f for f in os.listdir(output_dir) if f.startswith("frame_")]
                return True, len(frames), output_dir
            else:
                return False, 0, result.stderr
        except Exception as e:
            return False, 0, str(e)
    
    def create_gif(self, video_path: str, output_path: str,
                   start_time: float = 0,
                   duration: float = 5,
                   fps: int = 10,
                   width: int = 480) -> TrimResult:
        """创建GIF"""
        cmd = [
            self.config.ffmpeg_path,
            "-y",
            "-ss", str(start_time),
            "-t", str(duration),
            "-i", video_path,
            "-vf", f"fps={fps},scale={width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                return TrimResult(
                    success=True,
                    output_path=output_path
                )
            else:
                return TrimResult(
                    success=False,
                    error_message=result.stderr
                )
        except Exception as e:
            return TrimResult(success=False, error_message=str(e))
    
    def resize(self, video_path: str, output_path: str,
               width: Optional[int] = None,
               height: Optional[int] = None,
               scale: Optional[float] = None,
               keep_aspect_ratio: bool = True) -> TrimResult:
        """调整视频大小"""
        info = self.get_info(video_path)
        if not info:
            return TrimResult(success=False, error_message="Failed to read video info")
        
        # 计算目标尺寸
        if scale is not None:
            width = int(info.width * scale)
            height = int(info.height * scale)
        elif width is None and height is not None:
            if keep_aspect_ratio:
                width = int(height * info.width / info.height)
        elif height is None and width is not None:
            if keep_aspect_ratio:
                height = int(width * info.height / info.width)
        else:
            width = width or info.width
            height = height or info.height
        
        cmd = [
            self.config.ffmpeg_path,
            "-y",
            "-i", video_path,
            "-vf", f"scale={width}:{height}",
            "-c:v", self.config.video_codec.value,
            "-crf", str(self.config.quality),
            "-preset", self.config.preset,
            "-c:a", "copy",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                output_info = self.get_info(output_path)
                return TrimResult(
                    success=True,
                    output_path=output_path,
                    output_info=output_info
                )
            else:
                return TrimResult(
                    success=False,
                    error_message=result.stderr
                )
        except Exception as e:
            return TrimResult(success=False, error_message=str(e))
    
    def add_watermark(self, video_path: str, output_path: str,
                      watermark_path: str,
                      position: str = "bottom_right",
                      opacity: float = 0.5) -> TrimResult:
        """添加水印"""
        position_map = {
            "top_left": "10:10",
            "top_right": "main_w-overlay_w-10:10",
            "bottom_left": "10:main_h-overlay_h-10",
            "bottom_right": "main_w-overlay_w-10:main_h-overlay_h-10",
            "center": "(main_w-overlay_w)/2:(main_h-overlay_h)/2"
        }
        
        overlay_pos = position_map.get(position, position_map["bottom_right"])
        
        cmd = [
            self.config.ffmpeg_path,
            "-y",
            "-i", video_path,
            "-i", watermark_path,
            "-filter_complex",
            f"[1:v]format=rgba,colorchannelmixer=aa={opacity}[wm];[0:v][wm]overlay={overlay_pos}",
            "-c:v", self.config.video_codec.value,
            "-crf", str(self.config.quality),
            "-preset", self.config.preset,
            "-c:a", "copy",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                return TrimResult(
                    success=True,
                    output_path=output_path
                )
            else:
                return TrimResult(
                    success=False,
                    error_message=result.stderr
                )
        except Exception as e:
            return TrimResult(success=False, error_message=str(e))
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的视频格式"""
        return [f.value for f in VideoFormat]
    
    def get_supported_codecs(self) -> Dict[str, List[str]]:
        """获取支持的编码"""
        return {
            "video": [c.value for c in VideoCodec],
            "audio": [c.value for c in AudioCodec]
        }
