"""
Generation Utilities for AGI Unified Framework

Provides utility functions for image processing, audio manipulation,
video operations, seed management, and batch generation.
"""

from typing import List, Tuple, Optional, Dict, Any, Callable
import random
import math
import hashlib
from dataclasses import dataclass
from enum import Enum


class InterpolationMethod(Enum):
    """Image interpolation methods."""
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"


@dataclass
class GenerationResult:
    """Result of a generation operation."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    seed: Optional[int] = None


class ImageUtils:
    """
    Image processing utilities.
    
    Provides basic image manipulation operations using pure Python.
    """
    
    @staticmethod
    def resize(
        image: List[List[Tuple[int, int, int]]],
        width: int,
        height: int,
        method: str = "bilinear"
    ) -> List[List[Tuple[int, int, int]]]:
        """
        Resize an image.
        
        Args:
            image: Input image as 2D list of RGB tuples
            width: Target width
            height: Target height
            method: Interpolation method
            
        Returns:
            Resized image
        """
        if not image or not image[0]:
            return []
        
        src_height = len(image)
        src_width = len(image[0])
        
        if method == "nearest":
            return ImageUtils._resize_nearest(image, src_width, src_height, width, height)
        else:
            return ImageUtils._resize_bilinear(image, src_width, src_height, width, height)
    
    @staticmethod
    def _resize_nearest(
        image: List[List[Tuple[int, int, int]]],
        src_w: int,
        src_h: int,
        dst_w: int,
        dst_h: int
    ) -> List[List[Tuple[int, int, int]]]:
        """Nearest neighbor resize."""
        result = []
        for y in range(dst_h):
            row = []
            src_y = int(y * src_h / dst_h)
            for x in range(dst_w):
                src_x = int(x * src_w / dst_w)
                row.append(image[src_y][src_x])
            result.append(row)
        return result
    
    @staticmethod
    def _resize_bilinear(
        image: List[List[Tuple[int, int, int]]],
        src_w: int,
        src_h: int,
        dst_w: int,
        dst_h: int
    ) -> List[List[Tuple[int, int, int]]]:
        """Bilinear resize."""
        result = []
        for y in range(dst_h):
            row = []
            src_y = y * (src_h - 1) / (dst_h - 1) if dst_h > 1 else 0
            for x in range(dst_w):
                src_x = x * (src_w - 1) / (dst_w - 1) if dst_w > 1 else 0
                pixel = ImageUtils._bilinear_interpolate(image, src_x, src_y)
                row.append(pixel)
            result.append(row)
        return result
    
    @staticmethod
    def crop(
        image: List[List[Tuple[int, int, int]]],
        x: int,
        y: int,
        width: int,
        height: int
    ) -> List[List[Tuple[int, int, int]]]:
        """
        Crop an image.
        
        Args:
            image: Input image
            x: X coordinate (top-left)
            y: Y coordinate (top-left)
            width: Crop width
            height: Crop height
            
        Returns:
            Cropped image
        """
        if not image:
            return []
        
        img_height = len(image)
        img_width = len(image[0]) if image else 0
        
        # Clamp coordinates
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        width = min(width, img_width - x)
        height = min(height, img_height - y)
        
        return [row[x:x+width] for row in image[y:y+height]]
    
    @staticmethod
    def pad(
        image: List[List[Tuple[int, int, int]]],
        padding: int,
        fill: Tuple[int, int, int] = (0, 0, 0)
    ) -> List[List[Tuple[int, int, int]]]:
        """
        Pad an image with border.
        
        Args:
            image: Input image
            padding: Padding size (all sides)
            fill: Fill color (RGB)
            
        Returns:
            Padded image
        """
        if not image:
            return []
        
        width = len(image[0]) if image else 0
        
        # Create padding rows
        pad_row = [fill] * (width + 2 * padding)
        
        result = [pad_row[:] for _ in range(padding)]
        
        for row in image:
            padded_row = [fill] * padding + row + [fill] * padding
            result.append(padded_row)
        
        result.extend([pad_row[:] for _ in range(padding)])
        
        return result
    
    @staticmethod
    def rotate(
        image: List[List[Tuple[int, int, int]]],
        angle: float,
        expand: bool = False
    ) -> List[List[Tuple[int, int, int]]]:
        """
        Rotate an image.
        
        Args:
            image: Input image
            angle: Rotation angle in degrees
            expand: Expand output to fit rotated image
            
        Returns:
            Rotated image
        """
        if not image:
            return []
        
        height = len(image)
        width = len(image[0]) if image else 0
        
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        
        if expand:
            # Calculate new bounds
            corners = [
                (0, 0),
                (width, 0),
                (width, height),
                (0, height)
            ]
            rotated_corners = [
                (x * cos_a - y * sin_a, x * sin_a + y * cos_a)
                for x, y in corners
            ]
            min_x = min(c[0] for c in rotated_corners)
            max_x = max(c[0] for c in rotated_corners)
            min_y = min(c[1] for c in rotated_corners)
            max_y = max(c[1] for c in rotated_corners)
            
            new_width = int(max_x - min_x)
            new_height = int(max_y - min_y)
            cx = new_width / 2
            cy = new_height / 2
        else:
            new_width = width
            new_height = height
            cx = width / 2
            cy = height / 2
        
        # Create output
        result = []
        for y in range(new_height):
            row = []
            for x in range(new_width):
                # Inverse rotation
                dx = x - cx
                dy = y - cy
                src_x = dx * cos_a + dy * sin_a + width / 2
                src_y = -dx * sin_a + dy * cos_a + height / 2
                
                pixel = ImageUtils._bilinear_interpolate(image, src_x, src_y)
                row.append(pixel)
            result.append(row)
        
        return result
    
    @staticmethod
    def flip(
        image: List[List[Tuple[int, int, int]]],
        horizontal: bool = True
    ) -> List[List[Tuple[int, int, int]]]:
        """
        Flip an image.
        
        Args:
            image: Input image
            horizontal: Flip horizontally (True) or vertically (False)
            
        Returns:
            Flipped image
        """
        if not image:
            return []
        
        if horizontal:
            return [row[::-1] for row in image]
        else:
            return image[::-1]
    
    @staticmethod
    def adjust_brightness(
        image: List[List[Tuple[int, int, int]]],
        factor: float
    ) -> List[List[Tuple[int, int, int]]]:
        """
        Adjust image brightness.
        
        Args:
            image: Input image
            factor: Brightness factor (1.0 = no change)
            
        Returns:
            Adjusted image
        """
        result = []
        for row in image:
            new_row = []
            for r, g, b in row:
                new_row.append((
                    ImageUtils._clamp(int(r * factor), 0, 255),
                    ImageUtils._clamp(int(g * factor), 0, 255),
                    ImageUtils._clamp(int(b * factor), 0, 255)
                ))
            result.append(new_row)
        return result
    
    @staticmethod
    def adjust_contrast(
        image: List[List[Tuple[int, int, int]]],
        factor: float
    ) -> List[List[Tuple[int, int, int]]]:
        """
        Adjust image contrast.
        
        Args:
            image: Input image
            factor: Contrast factor (1.0 = no change)
            
        Returns:
            Adjusted image
        """
        result = []
        for row in image:
            new_row = []
            for r, g, b in row:
                new_row.append((
                    ImageUtils._clamp(int((r - 128) * factor + 128), 0, 255),
                    ImageUtils._clamp(int((g - 128) * factor + 128), 0, 255),
                    ImageUtils._clamp(int((b - 128) * factor + 128), 0, 255)
                ))
            result.append(new_row)
        return result
    
    @staticmethod
    def adjust_saturation(
        image: List[List[Tuple[int, int, int]]],
        factor: float
    ) -> List[List[Tuple[int, int, int]]]:
        """
        Adjust image saturation.
        
        Args:
            image: Input image
            factor: Saturation factor (1.0 = no change)
            
        Returns:
            Adjusted image
        """
        result = []
        for row in image:
            new_row = []
            for r, g, b in row:
                gray = 0.299 * r + 0.587 * g + 0.114 * b
                new_row.append((
                    ImageUtils._clamp(int(gray + (r - gray) * factor), 0, 255),
                    ImageUtils._clamp(int(gray + (g - gray) * factor), 0, 255),
                    ImageUtils._clamp(int(gray + (b - gray) * factor), 0, 255)
                ))
            result.append(new_row)
        return result
    
    @staticmethod
    def blend(
        image_a: List[List[Tuple[int, int, int]]],
        image_b: List[List[Tuple[int, int, int]]],
        alpha: float
    ) -> List[List[Tuple[int, int, int]]]:
        """
        Blend two images.
        
        Args:
            image_a: First image
            image_b: Second image
            alpha: Blend factor (0.0 = image_a, 1.0 = image_b)
            
        Returns:
            Blended image
        """
        if not image_a or not image_b:
            return image_a or image_b or []
        
        result = []
        for row_a, row_b in zip(image_a, image_b):
            new_row = []
            for (r1, g1, b1), (r2, g2, b2) in zip(row_a, row_b):
                new_row.append((
                    int(r1 * (1 - alpha) + r2 * alpha),
                    int(g1 * (1 - alpha) + g2 * alpha),
                    int(b1 * (1 - alpha) + b2 * alpha)
                ))
            result.append(new_row)
        return result
    
    @staticmethod
    def composite(
        foreground: List[List[Tuple[int, int, int]]],
        background: List[List[Tuple[int, int, int]]],
        mask: List[List[float]]
    ) -> List[List[Tuple[int, int, int]]]:
        """
        Composite foreground onto background using mask.
        
        Args:
            foreground: Foreground image
            background: Background image
            mask: Alpha mask (0.0-1.0)
            
        Returns:
            Composited image
        """
        if not foreground or not background:
            return foreground or background or []
        
        result = []
        for row_f, row_b, row_m in zip(foreground, background, mask):
            new_row = []
            for (rf, gf, bf), (rb, gb, bb), m in zip(row_f, row_b, row_m):
                new_row.append((
                    int(rf * m + rb * (1 - m)),
                    int(gf * m + gb * (1 - m)),
                    int(bf * m + bb * (1 - m))
                ))
            result.append(new_row)
        return result
    
    @staticmethod
    def _bilinear_interpolate(
        image: List[List[Tuple[int, int, int]]],
        x: float,
        y: float
    ) -> Tuple[int, int, int]:
        """
        Bilinear interpolation at coordinates.
        
        Args:
            image: Source image
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Interpolated pixel value
        """
        if not image:
            return (0, 0, 0)
        
        height = len(image)
        width = len(image[0]) if image else 0
        
        x0 = int(math.floor(x))
        y0 = int(math.floor(y))
        x1 = min(x0 + 1, width - 1)
        y1 = min(y0 + 1, height - 1)
        
        fx = x - x0
        fy = y - y0
        
        # Clamp coordinates
        x0 = max(0, min(x0, width - 1))
        y0 = max(0, min(y0, height - 1))
        
        # Get four corners
        c00 = image[y0][x0]
        c01 = image[y0][x1]
        c10 = image[y1][x0]
        c11 = image[y1][x1]
        
        # Interpolate
        result = []
        for i in range(3):
            val = (c00[i] * (1 - fx) * (1 - fy) +
                   c01[i] * fx * (1 - fy) +
                   c10[i] * (1 - fx) * fy +
                   c11[i] * fx * fy)
            result.append(int(val))
        
        return tuple(result)
    
    @staticmethod
    def _clamp(value: float, min_val: int, max_val: int) -> int:
        """Clamp value to range."""
        return max(min_val, min(int(value), max_val))


class AudioUtils:
    """
    Audio processing utilities.
    
    Provides basic audio manipulation operations.
    """
    
    @staticmethod
    def resample(
        audio: List[float],
        orig_sr: int,
        target_sr: int
    ) -> List[float]:
        """
        Resample audio to new sample rate.
        
        Args:
            audio: Audio samples
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio
        """
        if orig_sr == target_sr:
            return audio[:]
        
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        
        result = []
        for i in range(new_length):
            src_idx = i / ratio
            idx0 = int(src_idx)
            idx1 = min(idx0 + 1, len(audio) - 1)
            frac = src_idx - idx0
            
            val = audio[idx0] * (1 - frac) + audio[idx1] * frac
            result.append(val)
        
        return result
    
    @staticmethod
    def trim_silence(
        audio: List[float],
        threshold: float = 0.01
    ) -> List[float]:
        """
        Remove silence from start and end.
        
        Args:
            audio: Audio samples
            threshold: Silence threshold
            
        Returns:
            Trimmed audio
        """
        if not audio:
            return []
        
        # Find start
        start = 0
        for i, sample in enumerate(audio):
            if abs(sample) > threshold:
                start = i
                break
        
        # Find end
        end = len(audio)
        for i in range(len(audio) - 1, -1, -1):
            if abs(audio[i]) > threshold:
                end = i + 1
                break
        
        return audio[start:end]
    
    @staticmethod
    def normalize(
        audio: List[float],
        target_db: float = -20
    ) -> List[float]:
        """
        Normalize audio to target dB level.
        
        Args:
            audio: Audio samples
            target_db: Target decibel level
            
        Returns:
            Normalized audio
        """
        if not audio:
            return []
        
        # Calculate RMS
        rms = math.sqrt(sum(s ** 2 for s in audio) / len(audio))
        
        if rms == 0:
            return audio[:]
        
        # Convert target dB to linear
        target_linear = 10 ** (target_db / 20)
        
        # Calculate gain
        gain = target_linear / rms
        
        # Apply gain
        return [s * gain for s in audio]
    
    @staticmethod
    def fade_in(audio: List[float], duration_ms: int, sr: int = 44100) -> List[float]:
        """
        Apply fade-in to audio.
        
        Args:
            audio: Audio samples
            duration_ms: Fade duration in milliseconds
            sr: Sample rate
            
        Returns:
            Faded audio
        """
        samples = int(duration_ms * sr / 1000)
        samples = min(samples, len(audio))
        
        result = audio[:]
        for i in range(samples):
            result[i] *= (i / samples) ** 2  # Quadratic fade
        
        return result
    
    @staticmethod
    def fade_out(audio: List[float], duration_ms: int, sr: int = 44100) -> List[float]:
        """
        Apply fade-out to audio.
        
        Args:
            audio: Audio samples
            duration_ms: Fade duration in milliseconds
            sr: Sample rate
            
        Returns:
            Faded audio
        """
        samples = int(duration_ms * sr / 1000)
        samples = min(samples, len(audio))
        
        result = audio[:]
        start = len(audio) - samples
        for i in range(samples):
            result[start + i] *= ((samples - i) / samples) ** 2
        
        return result
    
    @staticmethod
    def mix(audios: List[List[float]], weights: List[float]) -> List[float]:
        """
        Mix multiple audio tracks.
        
        Args:
            audios: List of audio tracks
            weights: Weight for each track
            
        Returns:
            Mixed audio
        """
        if not audios:
            return []
        
        # Find max length
        max_len = max(len(a) for a in audios)
        
        result = []
        for i in range(max_len):
            sample = 0.0
            for audio, weight in zip(audios, weights):
                if i < len(audio):
                    sample += audio[i] * weight
            result.append(sample)
        
        return result
    
    @staticmethod
    def stretch_time(audio: List[float], factor: float) -> List[float]:
        """
        Time-stretch audio (simplified implementation).
        
        Args:
            audio: Audio samples
            factor: Stretch factor (>1 = slower, <1 = faster)
            
        Returns:
            Time-stretched audio
        """
        if factor <= 0:
            return audio[:]
        
        new_length = int(len(audio) * factor)
        result = []
        
        for i in range(new_length):
            src_idx = i / factor
            idx0 = int(src_idx)
            idx1 = min(idx0 + 1, len(audio) - 1)
            frac = src_idx - idx0
            
            val = audio[idx0] * (1 - frac) + audio[idx1] * frac
            result.append(val)
        
        return result
    
    @staticmethod
    def pitch_shift(audio: List[float], semitones: float) -> List[float]:
        """
        Pitch-shift audio (simplified implementation).
        
        Args:
            audio: Audio samples
            semitones: Semitones to shift
            
        Returns:
            Pitch-shifted audio
        """
        # Simplified: just resample (changes both pitch and speed)
        factor = 2 ** (semitones / 12)
        new_length = int(len(audio) / factor)
        
        result = []
        for i in range(new_length):
            src_idx = i * factor
            idx0 = int(src_idx)
            idx1 = min(idx0 + 1, len(audio) - 1)
            frac = src_idx - idx0
            
            val = audio[idx0] * (1 - frac) + audio[idx1] * frac
            result.append(val)
        
        return result


class VideoUtils:
    """
    Video processing utilities.
    
    Provides basic video manipulation operations.
    """
    
    @staticmethod
    def extract_frames(
        video: List[List[List[Tuple[int, int, int]]]],
        fps: int
    ) -> List[List[List[Tuple[int, int, int]]]]:
        """
        Extract frames from video at specified FPS.
        
        Args:
            video: Input video (list of frames)
            fps: Target frames per second
            
        Returns:
            Extracted frames
        """
        # Simplified: assume video is already at some FPS
        # In real implementation, would handle actual video files
        return video[:]
    
    @staticmethod
    def create_video(
        frames: List[List[List[Tuple[int, int, int]]]],
        fps: int,
        output_path: str
    ) -> None:
        """
        Create video from frames.
        
        Args:
            frames: List of frames
            fps: Frames per second
            output_path: Output file path
        """
        if not frames:
            raise ValueError("帧列表不能为空")
        
        # 获取输出格式（小写扩展名）
        ext = output_path.lower().split('.')[-1] if '.' in output_path else 'mp4'
        
        # 尝试使用 cv2（OpenCV）创建视频
        try:
            import cv2
            VideoUtils._create_video_cv2(frames, fps, output_path, ext)
            return
        except ImportError:
            pass
        
        # cv2 不可用，使用纯 Python 方式
        if ext == 'gif':
            VideoUtils._create_video_gif(frames, fps, output_path)
        else:
            # mp4/avi 格式：尝试通过 ffmpeg 命令行工具生成
            VideoUtils._create_video_ffmpeg(frames, fps, output_path, ext)

    @staticmethod
    def _create_video_cv2(
        frames: List[List[List[Tuple[int, int, int]]]],
        fps: int,
        output_path: str,
        ext: str
    ) -> None:
        """
        使用 OpenCV (cv2) 创建视频文件。

        Args:
            frames: 帧列表
            fps: 帧率
            output_path: 输出文件路径
            ext: 文件扩展名
        """
        import cv2
        import numpy as np

        # 根据格式选择编码器
        fourcc_map = {
            'mp4': 'mp4v',
            'avi': 'XVID',
            'gif': 'GIF',
        }
        codec = fourcc_map.get(ext, 'mp4v')
        fourcc = cv2.VideoWriter_fourcc(*codec)

        # 从第一帧获取分辨率
        height = len(frames[0])
        width = len(frames[0][0]) if height > 0 else 0

        if width == 0 or height == 0:
            raise ValueError("帧的分辨率不能为0")

        # 创建 VideoWriter 对象
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"无法创建视频写入器: {output_path}")

        for frame in frames:
            # 将帧数据转换为 numpy 数组 (height, width, 3)，BGR 格式
            img_array = np.zeros((height, width, 3), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    r, g, b = frame[y][x]
                    img_array[y, x] = [b, g, r]  # OpenCV 使用 BGR 顺序
            writer.write(img_array)

        writer.release()

    @staticmethod
    def _create_video_gif(
        frames: List[List[List[Tuple[int, int, int]]]],
        fps: int,
        output_path: str
    ) -> None:
        """
        使用纯 Python 生成 GIF 文件。
        基于 GIF89a 规范手动构建二进制数据。

        Args:
            frames: 帧列表
            fps: 帧率
            output_path: 输出文件路径
        """
        import struct

        # 从第一帧获取分辨率
        height = len(frames[0])
        width = len(frames[0][0]) if height > 0 else 0
        if width == 0 or height == 0:
            raise ValueError("帧的分辨率不能为0")

        # 每帧延迟时间（单位：厘秒，即 1/100 秒）
        delay_cs = max(1, int(100 / fps))

        with open(output_path, 'wb') as f:
            # === GIF 文件头 ===
            f.write(b'GIF89a')  # GIF89a 签名

            # === 逻辑屏幕描述符 ===
            f.write(struct.pack('<HH', width, height))  # 画布宽高（小端序）
            # 全局颜色表标志=1, 颜色分辨率=7(256色), 排序标志=0, 全局颜色表大小=7(256色)
            packed = 0xF7  # 11110111
            f.write(struct.pack('B', packed))
            f.write(struct.pack('B', 0))   # 背景色索引
            f.write(struct.pack('B', 0))   # 像素宽高比

            # === 全局颜色表（256色，每色3字节 RGB） ===
            # 使用简单的 6x6x6 色彩立方体 (216色) + 40个灰度色
            palette = []
            # 生成 6x6x6 色彩立方体
            for r in range(0, 256, 51):
                for g in range(0, 256, 51):
                    for b in range(0, 256, 51):
                        palette.append((r, g, b))
            # 补充灰度色
            for i in range(40):
                gray = int(i * 255 / 39)
                palette.append((gray, gray, gray))
            # 填充到 256 色
            while len(palette) < 256:
                palette.append((0, 0, 0))

            for r, g, b in palette:
                f.write(struct.pack('BBB', r, g, b))

            # === Netscape 扩展（支持动画循环） ===
            f.write(b'\x21\xFF\x0B')  # 应用扩展标记 + 块大小11
            f.write(b'NETSCAPE2.0')
            f.write(b'\x03\x01')       # 子块数据
            f.write(struct.pack('<H', 0))  # 循环次数：0=无限循环
            f.write(b'\x00')           # 块终止符

            # === 逐帧写入 ===
            for frame in frames:
                # 图形控制扩展（设置帧延迟和透明色）
                f.write(b'\x21\xF9\x04')  # 图形控制扩展 + 块大小4
                # 保留位=0, 用户输入=0, 透明色标志=0, 处置方法=0(不处置)
                f.write(struct.pack('B', 0x00))
                f.write(struct.pack('<H', delay_cs))  # 延迟时间（厘秒）
                f.write(struct.pack('B', 0))  # 透明色索引
                f.write(b'\x00')  # 块终止符

                # 图像描述符
                f.write(b'\x2C')  # 图像分隔符
                f.write(struct.pack('<HH', 0, 0))  # 左上角坐标
                f.write(struct.pack('<HH', width, height))  # 图像宽高
                # 局部颜色表标志=0, 交叉标志=0, 排序标志=0, 保留=0, 局部颜色表大小=0
                f.write(struct.pack('B', 0x00))

                # 图像数据（使用 LZW 压缩）
                # 将每个像素映射到调色板中最接近的颜色索引
                indices = bytearray()
                for y in range(height):
                    for x in range(width):
                        r, g, b = frame[y][x]
                        # 在调色板中查找最接近的颜色
                        best_idx = 0
                        best_dist = float('inf')
                        # 只在前216色中搜索（色彩立方体），提高效率
                        for idx in range(216):
                            pr, pg, pb = palette[idx]
                            dist = (r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2
                            if dist < best_dist:
                                best_dist = dist
                                best_idx = idx
                        indices.append(best_idx)

                # LZW 压缩编码
                lzw_min_code_size = 8  # 最小 LZW 代码大小
                f.write(struct.pack('B', lzw_min_code_size))

                compressed = VideoUtils._lzw_compress(indices, lzw_min_code_size)
                # 将压缩数据分成不超过255字节的子块写入
                pos = 0
                while pos < len(compressed):
                    chunk_size = min(255, len(compressed) - pos)
                    f.write(struct.pack('B', chunk_size))
                    f.write(compressed[pos:pos + chunk_size])
                    pos += chunk_size
                f.write(b'\x00')  # 块终止符

            # === GIF 结尾标记 ===
            f.write(b'\x3B')

    @staticmethod
    def _lzw_compress(data: bytearray, min_code_size: int) -> bytes:
        """
        LZW 压缩算法实现，用于 GIF 图像数据压缩。

        Args:
            data: 待压缩的原始字节数据
            min_code_size: 最小代码大小（位）

        Returns:
            压缩后的字节数据
        """
        clear_code = 1 << min_code_size
        eoi_code = clear_code + 1

        # 初始化字典：包含所有单字节条目
        table = {}
        for i in range(clear_code):
            table[(i,)] = i

        next_code = eoi_code + 1
        code_size = min_code_size + 1
        max_code = (1 << code_size) - 1

        result = bytearray()
        buffer = 0
        bits_in_buffer = 0

        def emit_code(code):
            """将一个 LZW 代码按位写入输出缓冲区"""
            nonlocal buffer, bits_in_buffer
            buffer |= (code << bits_in_buffer)
            bits_in_buffer += code_size
            while bits_in_buffer >= 8:
                result.append(buffer & 0xFF)
                buffer >>= 8
                bits_in_buffer -= 8

        # 写入清除码
        emit_code(clear_code)

        # 当前序列
        current = (data[0],)

        for i in range(1, len(data)):
            char = data[i]
            candidate = current + (char,)

            if candidate in table:
                current = candidate
            else:
                # 输出当前序列的代码
                emit_code(table[current])

                # 将新序列加入字典
                if next_code <= 4095:  # GIF 最大支持 12 位代码
                    table[candidate] = next_code
                    next_code += 1
                    # 检查是否需要增加代码大小
                    if next_code > max_code + 1 and code_size < 12:
                        code_size += 1
                        max_code = (1 << code_size) - 1

                current = (char,)

        # 输出最后一个序列的代码
        emit_code(table[current])

        # 写入结束码
        emit_code(eoi_code)

        # 刷新缓冲区中剩余的位
        if bits_in_buffer > 0:
            result.append(buffer & 0xFF)

        return bytes(result)

    @staticmethod
    def _create_video_ffmpeg(
        frames: List[List[List[Tuple[int, int, int]]]],
        fps: int,
        output_path: str,
        ext: str
    ) -> None:
        """
        通过 ffmpeg 命令行工具创建视频文件。
        先将帧保存为临时 PPM 图片，再调用 ffmpeg 合成视频。

        Args:
            frames: 帧列表
            fps: 帧率
            output_path: 输出文件路径
            ext: 文件扩展名
        """
        import subprocess
        import tempfile
        import os
        import struct

        # 从第一帧获取分辨率
        height = len(frames[0])
        width = len(frames[0][0]) if height > 0 else 0
        if width == 0 or height == 0:
            raise ValueError("帧的分辨率不能为0")

        # 创建临时目录存放帧图片
        tmp_dir = tempfile.mkdtemp(prefix='video_frames_')

        try:
            # 将每一帧保存为 PPM 格式（纯文本格式，无需额外依赖）
            for idx, frame in enumerate(frames):
                ppm_path = os.path.join(tmp_dir, f'frame_{idx:06d}.ppm')
                with open(ppm_path, 'wb') as f:
                    # P6 格式：二进制 PPM
                    header = f'P6\n{width} {height}\n255\n'.encode('ascii')
                    f.write(header)
                    for y in range(height):
                        for x in range(width):
                            r, g, b = frame[y][x]
                            f.write(struct.pack('BBB',
                                                max(0, min(255, r)),
                                                max(0, min(255, g)),
                                                max(0, min(255, b))))

            # 调用 ffmpeg 合成视频
            frame_pattern = os.path.join(tmp_dir, 'frame_%06d.ppm')
            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', frame_pattern,
                '-c:v', 'libx264' if ext == 'mp4' else 'mpeg4',
                '-pix_fmt', 'yuv420p',
                '-r', str(fps),
                output_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg 执行失败 (返回码 {result.returncode}): "
                    f"{result.stderr}"
                )
        finally:
            # 清理临时文件
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
    
    @staticmethod
    def resize_video(
        frames: List[List[List[Tuple[int, int, int]]]],
        width: int,
        height: int
    ) -> List[List[List[Tuple[int, int, int]]]]:
        """
        Resize all frames in a video.
        
        Args:
            frames: Input frames
            width: Target width
            height: Target height
            
        Returns:
            Resized frames
        """
        return [ImageUtils.resize(frame, width, height) for frame in frames]
    
    @staticmethod
    def concatenate_videos(
        video_list: List[List[List[List[Tuple[int, int, int]]]]]
    ) -> List[List[List[Tuple[int, int, int]]]]:
        """
        Concatenate multiple videos.
        
        Args:
            video_list: List of videos (each is list of frames)
            
        Returns:
            Concatenated video
        """
        result = []
        for video in video_list:
            result.extend(video)
        return result
    
    @staticmethod
    def add_transition(
        frame_a: List[List[Tuple[int, int, int]]],
        frame_b: List[List[Tuple[int, int, int]]],
        type: str = "fade",
        duration: int = 1,
        fps: int = 30
    ) -> List[List[List[Tuple[int, int, int]]]]:
        """
        Add transition between two frames.
        
        Args:
            frame_a: First frame
            frame_b: Second frame
            type: Transition type (fade, wipe, etc.)
            duration: Duration in seconds
            fps: Frames per second
            
        Returns:
            Transition frames
        """
        num_frames = duration * fps
        result = []
        
        for i in range(num_frames):
            alpha = i / (num_frames - 1) if num_frames > 1 else 1.0
            
            if type == "fade":
                frame = ImageUtils.blend(frame_a, frame_b, alpha)
            else:
                # Default to fade
                frame = ImageUtils.blend(frame_a, frame_b, alpha)
            
            result.append(frame)
        
        return result


class SeedManager:
    """
    Random seed management for reproducible generation.
    
    Provides global seed setting, random seed generation,
    and deterministic generator creation.
    """
    
    def __init__(self):
        self._global_seed: Optional[int] = None
    
    def set_global_seed(self, seed: int) -> None:
        """
        Set the global random seed.
        
        Args:
            seed: Random seed value
        """
        self._global_seed = seed
        random.seed(seed)
    
    def get_random_seed(self) -> int:
        """
        Get a random seed.
        
        Returns:
            Random seed value
        """
        if self._global_seed is not None:
            return self._global_seed
        return random.randint(0, 2**32 - 1)
    
    def create_generator(self, seed: Optional[int] = None) -> random.Random:
        """
        Create a random generator.
        
        Args:
            seed: Seed for generator (uses global if None)
            
        Returns:
            Random generator instance
        """
        if seed is None:
            seed = self.get_random_seed()
        return random.Random(seed)
    
    def generate_seeds(
        self,
        count: int,
        base_seed: Optional[int] = None
    ) -> List[int]:
        """
        Generate a sequence of seeds.
        
        Args:
            count: Number of seeds to generate
            base_seed: Base seed for sequence
            
        Returns:
            List of seeds
        """
        if base_seed is None:
            base_seed = self.get_random_seed()
        
        gen = random.Random(base_seed)
        return [gen.randint(0, 2**32 - 1) for _ in range(count)]
    
    def _hash_string_to_seed(self, s: str) -> int:
        """
        Convert a string to a deterministic seed.
        
        Args:
            s: Input string
            
        Returns:
            Seed value
        """
        hash_obj = hashlib.md5(s.encode())
        return int(hash_obj.hexdigest(), 16) % (2**32)


class BatchGenerator:
    """
    Batch generation utilities.
    
    Provides batch processing, parallel generation simulation,
    and time estimation for generation tasks.
    """
    
    def __init__(self, pipeline: Optional[Any] = None):
        self._pipeline = pipeline
    
    def generate_batch(
        self,
        prompts: List[str],
        batch_size: int = 4
    ) -> List[GenerationResult]:
        """
        Generate outputs for multiple prompts in batches.
        
        Args:
            prompts: List of input prompts
            batch_size: Number of prompts per batch
            
        Returns:
            List of generation results
        """
        results = []
        batches = self._split_batches(prompts, batch_size)
        
        for batch in batches:
            batch_results = self._parallel_generate(batch)
            results.extend(batch_results)
        
        return results
    
    def _split_batches(self, items: List[Any], batch_size: int) -> List[List[Any]]:
        """
        Split items into batches.
        
        Args:
            items: List of items
            batch_size: Batch size
            
        Returns:
            List of batches
        """
        return [
            items[i:i + batch_size]
            for i in range(0, len(items), batch_size)
        ]
    
    def _parallel_generate(self, batch: List[str]) -> List[GenerationResult]:
        """
        Generate for a batch (simulated parallel processing).
        
        Args:
            batch: Batch of prompts
            
        Returns:
            List of results
        """
        results = []
        
        for prompt in batch:
            try:
                # Simulate generation
                result = self._simulate_generate(prompt)
                results.append(GenerationResult(
                    success=True,
                    data=result,
                    metadata={"prompt": prompt}
                ))
            except Exception as e:
                results.append(GenerationResult(
                    success=False,
                    error=str(e),
                    metadata={"prompt": prompt}
                ))
        
        return results
    
    def _simulate_generate(self, prompt: str) -> Any:
        """Simulate generation for demonstration."""
        # Return a placeholder result
        return f"Generated output for: {prompt[:50]}..."
    
    def estimate_time(
        self,
        num_prompts: int,
        avg_time_per_prompt: float
    ) -> float:
        """
        Estimate total generation time.
        
        Args:
            num_prompts: Number of prompts
            avg_time_per_prompt: Average time per prompt
            
        Returns:
            Estimated time in seconds
        """
        return num_prompts * avg_time_per_prompt
    
    def set_pipeline(self, pipeline: Any) -> None:
        """
        Set the generation pipeline.
        
        Args:
            pipeline: Pipeline object
        """
        self._pipeline = pipeline


# Utility functions
def create_test_image(
    width: int,
    height: int,
    color: Tuple[int, int, int] = (128, 128, 128)
) -> List[List[Tuple[int, int, int]]]:
    """
    Create a test image filled with a color.
    
    Args:
        width: Image width
        height: Image height
        color: Fill color
        
    Returns:
        Test image
    """
    return [[color for _ in range(width)] for _ in range(height)]


def create_test_audio(
    duration_sec: float,
    sample_rate: int = 44100,
    frequency: float = 440.0
) -> List[float]:
    """
    Create a test sine wave audio.
    
    Args:
        duration_sec: Duration in seconds
        sample_rate: Sample rate
        frequency: Sine wave frequency
        
    Returns:
        Audio samples
    """
    num_samples = int(duration_sec * sample_rate)
    return [
        math.sin(2 * math.pi * frequency * i / sample_rate)
        for i in range(num_samples)
    ]


def calculate_psnr(
    image_a: List[List[Tuple[int, int, int]]],
    image_b: List[List[Tuple[int, int, int]]]
) -> float:
    """
    Calculate PSNR between two images.
    
    Args:
        image_a: First image
        image_b: Second image
        
    Returns:
        PSNR value in dB
    """
    if not image_a or not image_b:
        return 0.0
    
    mse = 0.0
    count = 0
    
    for row_a, row_b in zip(image_a, image_b):
        for (r1, g1, b1), (r2, g2, b2) in zip(row_a, row_b):
            mse += (r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2
            count += 3
    
    mse = mse / count if count > 0 else 0
    
    if mse == 0:
        return float('inf')
    
    return 10 * math.log10(255 ** 2 / mse)
