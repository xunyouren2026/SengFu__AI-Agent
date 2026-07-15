"""
FFmpeg 封装器
支持视频转码、剪辑、合并、提取音频等操作
"""

import os
import re
import subprocess
import json
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class VideoCodec(Enum):
    """视频编解码器"""
    H264 = "libx264"
    H265 = "libx265"
    VP8 = "libvpx"
    VP9 = "libvpx-vp9"
    AV1 = "libaom-av1"
    MPEG4 = "mpeg4"
    COPY = "copy"


class AudioCodec(Enum):
    """音频编解码器"""
    AAC = "aac"
    MP3 = "libmp3lame"
    OPUS = "libopus"
    VORBIS = "libvorbis"
    FLAC = "flac"
    PCM = "pcm_s16le"
    COPY = "copy"


class ContainerFormat(Enum):
    """容器格式"""
    MP4 = "mp4"
    MKV = "matroska"
    WEBM = "webm"
    AVI = "avi"
    MOV = "mov"
    TS = "mpegts"
    FLV = "flv"


@dataclass
class MediaInfo:
    """媒体信息"""
    duration: float = 0.0
    bitrate: int = 0
    format_name: str = ""
    format_long_name: str = ""
    
    # 视频流信息
    video_codec: str = ""
    video_width: int = 0
    video_height: int = 0
    video_fps: float = 0.0
    video_bitrate: int = 0
    video_aspect_ratio: str = ""
    
    # 音频流信息
    audio_codec: str = ""
    audio_sample_rate: int = 0
    audio_channels: int = 0
    audio_bitrate: int = 0
    
    def __repr__(self) -> str:
        return (
            f"MediaInfo(duration={self.duration:.2f}s, "
            f"video={self.video_codec} {self.video_width}x{self.video_height}, "
            f"audio={self.audio_codec})"
        )


@dataclass
class TranscodeOptions:
    """转码选项"""
    video_codec: VideoCodec = VideoCodec.H264
    audio_codec: AudioCodec = AudioCodec.AAC
    container: ContainerFormat = ContainerFormat.MP4
    video_bitrate: Optional[int] = None
    audio_bitrate: Optional[int] = None
    crf: int = 23  # 恒定质量因子 (0-51, 越小质量越好)
    preset: str = "medium"  # 编码预设
    resolution: Optional[Tuple[int, int]] = None
    fps: Optional[float] = None
    copy_audio: bool = False
    copy_video: bool = False


@dataclass
class ClipSegment:
    """剪辑片段"""
    start_time: float
    end_time: float
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def __repr__(self) -> str:
        return f"ClipSegment({self.start_time:.2f}s - {self.end_time:.2f}s)"


class FFmpegWrapper:
    """FFmpeg 封装器主类"""
    
    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        """
        初始化 FFmpeg 封装器
        
        Args:
            ffmpeg_path: ffmpeg 可执行文件路径
            ffprobe_path: ffprobe 可执行文件路径
        """
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self._validate_installation()
    
    def _validate_installation(self) -> None:
        """验证 FFmpeg 安装"""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError("FFmpeg 未正确安装")
        except FileNotFoundError:
            raise RuntimeError(f"找不到 ffmpeg: {self.ffmpeg_path}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("FFmpeg 验证超时")
    
    def get_version(self) -> str:
        """
        获取 FFmpeg 版本
        
        Returns:
            版本字符串
        """
        result = subprocess.run(
            [self.ffmpeg_path, "-version"],
            capture_output=True,
            text=True
        )
        
        # 解析版本号
        match = re.search(r'ffmpeg version (\S+)', result.stdout)
        if match:
            return match.group(1)
        return "unknown"
    
    def get_media_info(self, file_path: Union[str, Path]) -> Optional[MediaInfo]:
        """
        获取媒体文件信息
        
        Args:
            file_path: 媒体文件路径
            
        Returns:
            MediaInfo 对象或 None
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        
        cmd = [
            self.ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return None
            
            data = json.loads(result.stdout)
            return self._parse_media_info(data)
            
        except (json.JSONDecodeError, subprocess.TimeoutExpired):
            return None
    
    def _parse_media_info(self, data: Dict[str, Any]) -> MediaInfo:
        """解析媒体信息"""
        info = MediaInfo()
        
        # 格式信息
        if 'format' in data:
            fmt = data['format']
            info.duration = float(fmt.get('duration', 0))
            info.bitrate = int(fmt.get('bit_rate', 0))
            info.format_name = fmt.get('format_name', '')
            info.format_long_name = fmt.get('format_long_name', '')
        
        # 流信息
        for stream in data.get('streams', []):
            codec_type = stream.get('codec_type')
            
            if codec_type == 'video' and not info.video_codec:
                info.video_codec = stream.get('codec_name', '')
                info.video_width = int(stream.get('width', 0))
                info.video_height = int(stream.get('height', 0))
                info.video_bitrate = int(stream.get('bit_rate', 0))
                
                # 解析帧率
                fps_str = stream.get('r_frame_rate', '0/1')
                if '/' in fps_str:
                    num, den = map(int, fps_str.split('/'))
                    info.video_fps = num / den if den != 0 else 0
                
                # 宽高比
                info.video_aspect_ratio = stream.get('display_aspect_ratio', '')
                
            elif codec_type == 'audio' and not info.audio_codec:
                info.audio_codec = stream.get('codec_name', '')
                info.audio_sample_rate = int(stream.get('sample_rate', 0))
                info.audio_channels = int(stream.get('channels', 0))
                info.audio_bitrate = int(stream.get('bit_rate', 0))
        
        return info
    
    def transcode(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        options: Optional[TranscodeOptions] = None,
        progress_callback: Optional[callable] = None
    ) -> bool:
        """
        视频转码
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            options: 转码选项
            progress_callback: 进度回调函数
            
        Returns:
            是否成功
        """
        if options is None:
            options = TranscodeOptions()
        
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 构建命令
        cmd = [self.ffmpeg_path, "-y", "-i", str(input_path)]
        
        # 视频编码选项
        if options.copy_video:
            cmd.extend(["-c:v", "copy"])
        else:
            cmd.extend(["-c:v", options.video_codec.value])
            
            if options.video_codec == VideoCodec.H264:
                cmd.extend(["-preset", options.preset])
                cmd.extend(["-crf", str(options.crf)])
            elif options.video_codec == VideoCodec.H265:
                cmd.extend(["-preset", options.preset])
                cmd.extend(["-crf", str(options.crf)])
            
            if options.video_bitrate:
                cmd.extend(["-b:v", f"{options.video_bitrate}k"])
            
            if options.resolution:
                width, height = options.resolution
                cmd.extend(["-s", f"{width}x{height}"])
            
            if options.fps:
                cmd.extend(["-r", str(options.fps)])
        
        # 音频编码选项
        if options.copy_audio:
            cmd.extend(["-c:a", "copy"])
        else:
            cmd.extend(["-c:a", options.audio_codec.value])
            
            if options.audio_bitrate:
                cmd.extend(["-b:a", f"{options.audio_bitrate}k"])
        
        # 输出格式
        cmd.extend(["-f", options.container.value])
        cmd.append(str(output_path))
        
        # 执行命令
        try:
            process = subprocess.Popen(
                cmd,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True
            )
            
            # 监控进度
            if progress_callback:
                self._monitor_progress(process, input_path, progress_callback)
            else:
                process.wait()
            
            return process.returncode == 0
            
        except subprocess.SubprocessError:
            return False
    
    def _monitor_progress(
        self,
        process: subprocess.Popen,
        input_path: Path,
        callback: callable
    ) -> None:
        """监控转码进度"""
        info = self.get_media_info(input_path)
        total_duration = info.duration if info else 0
        
        pattern = re.compile(r'time=(\d+):(\d+):(\d+\.?\d*)')
        
        while True:
            line = process.stderr.readline()
            if not line:
                break
            
            match = pattern.search(line)
            if match and total_duration > 0:
                hours = int(match.group(1))
                minutes = int(match.group(2))
                seconds = float(match.group(3))
                current = hours * 3600 + minutes * 60 + seconds
                progress = min(current / total_duration, 1.0)
                callback(progress)
        
        process.wait()
    
    def clip(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        start_time: float,
        end_time: float,
        options: Optional[TranscodeOptions] = None
    ) -> bool:
        """
        剪辑视频片段
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            options: 转码选项
            
        Returns:
            是否成功
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-ss", str(start_time),
            "-i", str(input_path),
            "-t", str(end_time - start_time),
            "-c", "copy" if options is None else options.video_codec.value,
            str(output_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=300
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
    
    def clip_segments(
        self,
        input_path: Union[str, Path],
        segments: List[ClipSegment],
        output_dir: Union[str, Path],
        base_name: Optional[str] = None
    ) -> List[Path]:
        """
        剪辑多个片段
        
        Args:
            input_path: 输入文件路径
            segments: 片段列表
            output_dir: 输出目录
            base_name: 输出文件基础名称
            
        Returns:
            输出文件路径列表
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if base_name is None:
            base_name = input_path.stem
        
        output_files = []
        
        for i, segment in enumerate(segments):
            output_file = output_dir / f"{base_name}_clip_{i+1:03d}{input_path.suffix}"
            if self.clip(input_path, output_file, segment.start_time, segment.end_time):
                output_files.append(output_file)
        
        return output_files
    
    def concat(
        self,
        input_files: List[Union[str, Path]],
        output_path: Union[str, Path],
        method: str = "concat_demuxer"
    ) -> bool:
        """
        合并视频文件
        
        Args:
            input_files: 输入文件列表
            output_path: 输出文件路径
            method: 合并方法 (concat_demuxer, concat_filter)
            
        Returns:
            是否成功
        """
        if not input_files:
            raise ValueError("输入文件列表为空")
        
        input_paths = [Path(f) for f in input_files]
        output_path = Path(output_path)
        
        for p in input_paths:
            if not p.exists():
                raise FileNotFoundError(f"文件不存在: {p}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if method == "concat_demuxer":
            return self._concat_demuxer(input_paths, output_path)
        else:
            return self._concat_filter(input_paths, output_path)
    
    def _concat_demuxer(self, input_paths: List[Path], output_path: Path) -> bool:
        """使用 concat demuxer 合并"""
        # 创建临时文件列表
        list_file = output_path.parent / ".concat_list.txt"
        
        with open(list_file, 'w') as f:
            for p in input_paths:
                f.write(f"file '{p.absolute()}'\n")
        
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=600)
            return result.returncode == 0
        finally:
            if list_file.exists():
                list_file.unlink()
    
    def _concat_filter(self, input_paths: List[Path], output_path: Path) -> bool:
        """使用 concat filter 合并"""
        # 构建输入参数
        inputs = []
        for p in input_paths:
            inputs.extend(["-i", str(p)])
        
        # 构建 filter
        filter_str = "concat=n={}:v=1:a=1".format(len(input_paths))
        
        cmd = [
            self.ffmpeg_path,
            "-y"
        ] + inputs + [
            "-filter_complex", filter_str,
            str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=600)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
    
    def extract_audio(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        audio_codec: AudioCodec = AudioCodec.AAC,
        bitrate: Optional[int] = None
    ) -> bool:
        """
        从视频中提取音频
        
        Args:
            input_path: 输入视频文件路径
            output_path: 输出音频文件路径
            audio_codec: 音频编码格式
            bitrate: 音频比特率 (kbps)
            
        Returns:
            是否成功
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", str(input_path),
            "-vn",  # 不包含视频
            "-c:a", audio_codec.value
        ]
        
        if bitrate:
            cmd.extend(["-b:a", f"{bitrate}k"])
        
        cmd.append(str(output_path))
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
    
    def add_audio(
        self,
        video_path: Union[str, Path],
        audio_path: Union[str, Path],
        output_path: Union[str, Path],
        replace: bool = False
    ) -> bool:
        """
        为视频添加音频
        
        Args:
            video_path: 视频文件路径
            audio_path: 音频文件路径
            output_path: 输出文件路径
            replace: 是否替换原有音频
            
        Returns:
            是否成功
        """
        video_path = Path(video_path)
        audio_path = Path(audio_path)
        output_path = Path(output_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", str(video_path),
            "-i", str(audio_path)
        ]
        
        if replace:
            cmd.extend(["-map", "0:v", "-map", "1:a"])
        else:
            cmd.extend(["-map", "0:v", "-map", "0:a", "-map", "1:a"])
        
        cmd.extend(["-c:v", "copy", "-c:a", "aac"])
        cmd.append(str(output_path))
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
    
    def extract_frames(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        fps: float = 1.0,
        format: str = "png",
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Path]:
        """
        提取视频帧
        
        Args:
            input_path: 输入视频文件路径
            output_dir: 输出目录
            fps: 提取帧率
            format: 输出图像格式
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            输出文件路径列表
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_pattern = output_dir / f"frame_%06d.{format}"
        
        cmd = [self.ffmpeg_path, "-y"]
        
        if start_time is not None:
            cmd.extend(["-ss", str(start_time)])
        
        cmd.extend(["-i", str(input_path)])
        
        if end_time is not None and start_time is not None:
            cmd.extend(["-t", str(end_time - start_time)])
        
        cmd.extend([
            "-vf", f"fps={fps}",
            str(output_pattern)
        ])
        
        try:
            subprocess.run(cmd, capture_output=True, timeout=300)
            
            # 返回生成的文件列表
            return sorted(output_dir.glob(f"frame_*.{format}"))
        except subprocess.TimeoutExpired:
            return []
    
    def resize(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        width: Optional[int] = None,
        height: Optional[int] = None,
        keep_aspect: bool = True
    ) -> bool:
        """
        调整视频分辨率
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            width: 目标宽度
            height: 目标高度
            keep_aspect: 是否保持宽高比
            
        Returns:
            是否成功
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 构建缩放过滤器
        if width and height:
            if keep_aspect:
                scale = f"scale={width}:{height}:force_original_aspect_ratio=decrease"
            else:
                scale = f"scale={width}:{height}"
        elif width:
            scale = f"scale={width}:-1"
        elif height:
            scale = f"scale=-1:{height}"
        else:
            raise ValueError("必须指定宽度或高度")
        
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", str(input_path),
            "-vf", scale,
            "-c:a", "copy",
            str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=600)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
    
    def add_watermark(
        self,
        input_path: Union[str, Path],
        watermark_path: Union[str, Path],
        output_path: Union[str, Path],
        position: str = "bottom_right",
        opacity: float = 1.0
    ) -> bool:
        """
        添加水印
        
        Args:
            input_path: 输入视频文件路径
            watermark_path: 水印图片路径
            output_path: 输出文件路径
            position: 水印位置
            opacity: 水印透明度
            
        Returns:
            是否成功
        """
        input_path = Path(input_path)
        watermark_path = Path(watermark_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        if not watermark_path.exists():
            raise FileNotFoundError(f"水印文件不存在: {watermark_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 位置映射
        positions = {
            "top_left": "10:10",
            "top_right": "main_w-overlay_w-10:10",
            "bottom_left": "10:main_h-overlay_h-10",
            "bottom_right": "main_w-overlay_w-10:main_h-overlay_h-10",
            "center": "(main_w-overlay_w)/2:(main_h-overlay_h)/2"
        }
        
        pos = positions.get(position, positions["bottom_right"])
        
        # 构建滤镜
        if opacity < 1.0:
            filter_str = f"[1:v]format=rgba,colorchannelmixer=aa={opacity}[wm];[0:v][wm]overlay={pos}"
        else:
            filter_str = f"overlay={pos}"
        
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", str(input_path),
            "-i", str(watermark_path),
            "-filter_complex", filter_str,
            "-c:a", "copy",
            str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=600)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
    
    def convert_to_gif(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        fps: int = 10,
        width: int = 480,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> bool:
        """
        转换为 GIF
        
        Args:
            input_path: 输入视频文件路径
            output_path: 输出 GIF 文件路径
            fps: 帧率
            width: 宽度
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            是否成功
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [self.ffmpeg_path, "-y"]
        
        if start_time is not None:
            cmd.extend(["-ss", str(start_time)])
        
        cmd.extend(["-i", str(input_path)])
        
        if end_time is not None and start_time is not None:
            cmd.extend(["-t", str(end_time - start_time)])
        
        # 使用 palette 生成高质量 GIF
        filter_str = f"fps={fps},scale={width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"
        
        cmd.extend(["-vf", filter_str, str(output_path)])
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
    
    def get_screenshot(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        time: float = 0.0
    ) -> bool:
        """
        获取指定时间的截图
        
        Args:
            input_path: 输入视频文件路径
            output_path: 输出图片文件路径
            time: 时间点（秒）
            
        Returns:
            是否成功
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-ss", str(time),
            "-i", str(input_path),
            "-vframes", "1",
            str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
    
    def adjust_speed(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        speed: float
    ) -> bool:
        """
        调整播放速度
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            speed: 速度倍率 (0.5 = 半速, 2.0 = 两倍速)
            
        Returns:
            是否成功
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 视频和音频速度调整
        video_filter = f"setpts={1/speed}*PTS"
        audio_filter = f"atempo={speed}" if speed != 1.0 else "anull"
        
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", str(input_path),
            "-filter:v", video_filter,
            "-filter:a", audio_filter,
            str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=600)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
    
    def reverse(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path]
    ) -> bool:
        """
        反转视频
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            
        Returns:
            是否成功
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", str(input_path),
            "-vf", "reverse",
            "-af", "areverse",
            str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=600)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
