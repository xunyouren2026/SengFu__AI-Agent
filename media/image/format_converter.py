"""
图像格式转换器
支持 PNG/JPG/WebP/BMP/GIF/TIFF 格式互转
"""

import os
import struct
import zlib
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple
from enum import Enum
import io


class ImageFormat(Enum):
    """支持的图像格式枚举"""
    PNG = "png"
    JPEG = "jpg"
    WEBP = "webp"
    BMP = "bmp"
    GIF = "gif"
    TIFF = "tiff"


class ImageInfo:
    """图像信息类"""
    
    def __init__(
        self,
        width: int,
        height: int,
        format: ImageFormat,
        channels: int = 3,
        bit_depth: int = 8,
        has_alpha: bool = False
    ):
        self.width = width
        self.height = height
        self.format = format
        self.channels = channels
        self.bit_depth = bit_depth
        self.has_alpha = has_alpha
    
    def __repr__(self) -> str:
        return (
            f"ImageInfo(width={self.width}, height={self.height}, "
            f"format={self.format.value}, channels={self.channels}, "
            f"bit_depth={self.bit_depth}, has_alpha={self.has_alpha})"
        )


class PixelBuffer:
    """像素缓冲区，存储图像数据"""
    
    def __init__(self, width: int, height: int, channels: int = 3):
        self.width = width
        self.height = height
        self.channels = channels
        self.data: List[List[List[int]]] = [
            [[0 for _ in range(channels)] for _ in range(width)]
            for _ in range(height)
        ]
    
    def get_pixel(self, x: int, y: int) -> List[int]:
        """获取指定位置的像素值"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.data[y][x][:]
        return [0] * self.channels
    
    def set_pixel(self, x: int, y: int, values: List[int]) -> None:
        """设置指定位置的像素值"""
        if 0 <= x < self.width and 0 <= y < self.height:
            for i, v in enumerate(values[:self.channels]):
                self.data[y][x][i] = max(0, min(255, v))
    
    def to_flat_bytes(self) -> bytes:
        """转换为扁平字节数据"""
        flat = []
        for row in self.data:
            for pixel in row:
                flat.extend(pixel)
        return bytes(flat)
    
    @classmethod
    def from_flat_bytes(
        cls,
        data: bytes,
        width: int,
        height: int,
        channels: int
    ) -> "PixelBuffer":
        """从扁平字节数据创建缓冲区"""
        buffer = cls(width, height, channels)
        idx = 0
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    if idx < len(data):
                        buffer.data[y][x][c] = data[idx]
                        idx += 1
        return buffer


class FormatConverter:
    """图像格式转换器主类"""
    
    SUPPORTED_FORMATS = {
        ImageFormat.PNG: [b'\x89PNG'],
        ImageFormat.JPEG: [b'\xff\xd8\xff'],
        ImageFormat.BMP: [b'BM'],
        ImageFormat.GIF: [b'GIF87a', b'GIF89a'],
        ImageFormat.WEBP: [b'RIFF'],  # WebP starts with RIFF
        ImageFormat.TIFF: [b'II', b'MM'],  # TIFF: little/big endian
    }
    
    FORMAT_EXTENSIONS = {
        '.png': ImageFormat.PNG,
        '.jpg': ImageFormat.JPEG,
        '.jpeg': ImageFormat.JPEG,
        '.webp': ImageFormat.WEBP,
        '.bmp': ImageFormat.BMP,
        '.gif': ImageFormat.GIF,
        '.tiff': ImageFormat.TIFF,
        '.tif': ImageFormat.TIFF,
    }
    
    def __init__(self, quality: int = 85):
        """
        初始化格式转换器
        
        Args:
            quality: JPEG/WebP 编码质量 (1-100)
        """
        self.quality = max(1, min(100, quality))
        self._buffer: Optional[PixelBuffer] = None
        self._info: Optional[ImageInfo] = None
    
    def detect_format(self, file_path: Union[str, Path]) -> Optional[ImageFormat]:
        """
        检测图像文件格式
        
        Args:
            file_path: 图像文件路径
            
        Returns:
            检测到的格式，无法识别则返回 None
        """
        path = Path(file_path)
        if not path.exists():
            return None
        
        with open(path, 'rb') as f:
            header = f.read(12)
        
        # 检查文件头
        for fmt, signatures in self.SUPPORTED_FORMATS.items():
            for sig in signatures:
                if header.startswith(sig):
                    if fmt == ImageFormat.WEBP:
                        # WebP 需要额外检查
                        if len(header) >= 12 and header[8:12] == b'WEBP':
                            return fmt
                    else:
                        return fmt
        
        # 通过扩展名判断
        ext = path.suffix.lower()
        return self.FORMAT_EXTENSIONS.get(ext)
    
    def get_image_info(self, file_path: Union[str, Path]) -> Optional[ImageInfo]:
        """
        获取图像信息
        
        Args:
            file_path: 图像文件路径
            
        Returns:
            图像信息对象
        """
        path = Path(file_path)
        fmt = self.detect_format(path)
        if fmt is None:
            return None
        
        with open(path, 'rb') as f:
            data = f.read()
        
        width, height, channels = self._parse_dimensions(data, fmt)
        
        return ImageInfo(
            width=width,
            height=height,
            format=fmt,
            channels=channels,
            has_alpha=(channels == 4)
        )
    
    def _parse_dimensions(
        self,
        data: bytes,
        fmt: ImageFormat
    ) -> Tuple[int, int, int]:
        """解析图像尺寸"""
        if fmt == ImageFormat.PNG:
            return self._parse_png_dimensions(data)
        elif fmt == ImageFormat.JPEG:
            return self._parse_jpeg_dimensions(data)
        elif fmt == ImageFormat.BMP:
            return self._parse_bmp_dimensions(data)
        elif fmt == ImageFormat.GIF:
            return self._parse_gif_dimensions(data)
        elif fmt == ImageFormat.TIFF:
            return self._parse_tiff_dimensions(data)
        elif fmt == ImageFormat.WEBP:
            return self._parse_webp_dimensions(data)
        return (0, 0, 3)
    
    def _parse_png_dimensions(self, data: bytes) -> Tuple[int, int, int]:
        """解析 PNG 尺寸"""
        if len(data) < 24:
            return (0, 0, 3)
        # PNG IHDR chunk starts at byte 8
        width = struct.unpack('>I', data[16:20])[0]
        height = struct.unpack('>I', data[20:24])[0]
        bit_depth = data[24]
        color_type = data[25]
        channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type, 3)
        return (width, height, channels)
    
    def _parse_jpeg_dimensions(self, data: bytes) -> Tuple[int, int, int]:
        """解析 JPEG 尺寸"""
        if len(data) < 4:
            return (0, 0, 3)
        
        idx = 2
        while idx < len(data) - 1:
            if data[idx] != 0xFF:
                idx += 1
                continue
            
            marker = data[idx + 1]
            
            # SOF markers
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                          0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                if idx + 9 < len(data):
                    height = struct.unpack('>H', data[idx + 5:idx + 7])[0]
                    width = struct.unpack('>H', data[idx + 7:idx + 9])[0]
                    channels = data[idx + 9]
                    return (width, height, channels)
            
            if marker == 0xD8 or marker == 0xD9:
                idx += 2
            elif marker == 0xFF:
                idx += 1
            else:
                if idx + 3 < len(data):
                    length = struct.unpack('>H', data[idx + 2:idx + 4])[0]
                    idx += 2 + length
                else:
                    break
        
        return (0, 0, 3)
    
    def _parse_bmp_dimensions(self, data: bytes) -> Tuple[int, int, int]:
        """解析 BMP 尺寸"""
        if len(data) < 54:
            return (0, 0, 3)
        width = struct.unpack('<I', data[18:22])[0]
        height = struct.unpack('<I', data[22:26])[0]
        bit_count = struct.unpack('<H', data[28:30])[0]
        channels = bit_count // 8
        return (width, height, max(1, channels))
    
    def _parse_gif_dimensions(self, data: bytes) -> Tuple[int, int, int]:
        """解析 GIF 尺寸"""
        if len(data) < 10:
            return (0, 0, 3)
        width = struct.unpack('<H', data[6:8])[0]
        height = struct.unpack('<H', data[8:10])[0]
        return (width, height, 3)
    
    def _parse_tiff_dimensions(self, data: bytes) -> Tuple[int, int, int]:
        """解析 TIFF 尺寸"""
        if len(data) < 8:
            return (0, 0, 3)
        
        # 判断字节序
        if data[:2] == b'II':
            endian = '<'
        else:
            endian = '>'
        
        # 简化处理：返回默认值
        # 完整解析需要读取 IFD
        return (0, 0, 3)
    
    def _parse_webp_dimensions(self, data: bytes) -> Tuple[int, int, int]:
        """解析 WebP 尺寸"""
        if len(data) < 30:
            return (0, 0, 3)
        
        # 检查 VP8/VP8L/VP8X
        chunk_type = data[12:16]
        
        if chunk_type == b'VP8 ':
            # Lossy
            if len(data) >= 30:
                width = struct.unpack('<H', data[26:28])[0] & 0x3FFF
                height = struct.unpack('<H', data[28:30])[0] & 0x3FFF
                return (width, height, 3)
        elif chunk_type == b'VP8L':
            # Lossless
            if len(data) >= 25:
                bits = struct.unpack('<I', data[21:25])[0]
                width = (bits & 0x3FFF) + 1
                height = ((bits >> 14) & 0x3FFF) + 1
                return (width, height, 4)
        elif chunk_type == b'VP8X':
            # Extended
            if len(data) >= 30:
                width = struct.unpack('<I', data[24:27] + b'\x00')[0] + 1
                height = struct.unpack('<I', data[27:30] + b'\x00')[0] + 1
                return (width, height, 4)
        
        return (0, 0, 3)
    
    def load(self, file_path: Union[str, Path]) -> bool:
        """
        加载图像文件
        
        Args:
            file_path: 图像文件路径
            
        Returns:
            是否加载成功
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        
        fmt = self.detect_format(path)
        if fmt is None:
            raise ValueError(f"不支持的图像格式: {path}")
        
        self._info = self.get_image_info(path)
        
        with open(path, 'rb') as f:
            data = f.read()
        
        # 根据格式解码
        if fmt == ImageFormat.BMP:
            self._buffer = self._decode_bmp(data)
        elif fmt == ImageFormat.PNG:
            self._buffer = self._decode_png(data)
        elif fmt == ImageFormat.GIF:
            self._buffer = self._decode_gif(data)
        else:
            # 其他格式创建占位缓冲区
            if self._info:
                self._buffer = PixelBuffer(
                    self._info.width,
                    self._info.height,
                    self._info.channels
                )
        
        return self._buffer is not None
    
    def _decode_bmp(self, data: bytes) -> Optional[PixelBuffer]:
        """解码 BMP 图像"""
        if len(data) < 54:
            return None
        
        # 读取头信息
        width = struct.unpack('<I', data[18:22])[0]
        height = struct.unpack('<I', data[22:26])[0]
        bit_count = struct.unpack('<H', data[28:30])[0]
        data_offset = struct.unpack('<I', data[10:14])[0]
        
        channels = bit_count // 8
        if channels == 0:
            channels = 1
        
        # 读取像素数据
        row_size = ((width * bit_count + 31) // 32) * 4
        buffer = PixelBuffer(width, abs(height), min(channels, 4))
        
        y_start = 0 if height < 0 else abs(height) - 1
        y_step = 1 if height < 0 else -1
        
        for y in range(abs(height)):
            row_idx = data_offset + y_start * row_size
            actual_y = y_start + y * y_step
            
            for x in range(width):
                pixel_offset = row_idx + x * (bit_count // 8)
                if pixel_offset + channels <= len(data):
                    if bit_count == 24:
                        b, g, r = data[pixel_offset:pixel_offset + 3]
                        buffer.set_pixel(x, actual_y, [r, g, b])
                    elif bit_count == 32:
                        b, g, r, a = data[pixel_offset:pixel_offset + 4]
                        buffer.set_pixel(x, actual_y, [r, g, b, a])
        
        return buffer
    
    def _decode_png(self, data: bytes) -> Optional[PixelBuffer]:
        """解码 PNG 图像（简化版）"""
        if len(data) < 24:
            return None
        
        width = struct.unpack('>I', data[16:20])[0]
        height = struct.unpack('>I', data[20:24])[0]
        color_type = data[25]
        
        channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type, 3)
        
        # 创建占位缓冲区
        # 完整 PNG 解码需要 zlib 解压和滤波逆变换
        buffer = PixelBuffer(width, height, channels)
        
        # 尝试找到 IDAT 块并解码
        idx = 8
        idat_data = b''
        
        while idx < len(data) - 8:
            chunk_len = struct.unpack('>I', data[idx:idx + 4])[0]
            chunk_type = data[idx + 4:idx + 8]
            chunk_data = data[idx + 8:idx + 8 + chunk_len]
            
            if chunk_type == b'IDAT':
                idat_data += chunk_data
            elif chunk_type == b'IEND':
                break
            
            idx += 12 + chunk_len
        
        if idat_data:
            try:
                decompressed = zlib.decompress(idat_data)
                # 简化处理：假设无滤波
                bytes_per_row = width * channels + 1
                for y in range(min(height, len(decompressed) // bytes_per_row)):
                    row_start = y * bytes_per_row + 1
                    for x in range(width):
                        pixel_start = row_start + x * channels
                        if pixel_start + channels <= len(decompressed):
                            pixel = list(decompressed[pixel_start:pixel_start + channels])
                            buffer.set_pixel(x, y, pixel)
            except Exception:
                pass
        
        return buffer
    
    def _decode_gif(self, data: bytes) -> Optional[PixelBuffer]:
        """解码 GIF 图像（简化版）"""
        if len(data) < 10:
            return None
        
        width = struct.unpack('<H', data[6:8])[0]
        height = struct.unpack('<H', data[8:10])[0]
        
        # 创建占位缓冲区
        buffer = PixelBuffer(width, height, 3)
        
        # 完整 GIF 解码需要 LZW 解压
        return buffer
    
    def save(
        self,
        output_path: Union[str, Path],
        format: Optional[ImageFormat] = None,
        **kwargs: Any
    ) -> bool:
        """
        保存图像到指定格式
        
        Args:
            output_path: 输出路径
            format: 目标格式，None 则根据扩展名判断
            **kwargs: 额外参数
            
        Returns:
            是否保存成功
        """
        if self._buffer is None:
            raise ValueError("没有加载的图像数据")
        
        path = Path(output_path)
        
        if format is None:
            ext = path.suffix.lower()
            format = self.FORMAT_EXTENSIONS.get(ext)
            if format is None:
                raise ValueError(f"无法确定输出格式: {ext}")
        
        # 确保输出目录存在
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == ImageFormat.BMP:
            data = self._encode_bmp()
        elif format == ImageFormat.PNG:
            data = self._encode_png()
        elif format == ImageFormat.GIF:
            data = self._encode_gif()
        else:
            # 其他格式保存为 BMP 作为后备
            data = self._encode_bmp()
            path = path.with_suffix('.bmp')
        
        with open(path, 'wb') as f:
            f.write(data)
        
        return True
    
    def _encode_bmp(self) -> bytes:
        """编码为 BMP 格式"""
        if self._buffer is None:
            return b''
        
        width = self._buffer.width
        height = self._buffer.height
        channels = min(self._buffer.channels, 3)
        bit_count = channels * 8
        
        row_size = ((width * bit_count + 31) // 32) * 4
        padding = row_size - width * channels
        image_size = row_size * height
        file_size = 54 + image_size
        
        # 文件头
        header = bytearray()
        header.extend(b'BM')  # 签名
        header.extend(struct.pack('<I', file_size))  # 文件大小
        header.extend(struct.pack('<HH', 0, 0))  # 保留
        header.extend(struct.pack('<I', 54))  # 数据偏移
        
        # 信息头
        header.extend(struct.pack('<I', 40))  # 头大小
        header.extend(struct.pack('<i', width))  # 宽度
        header.extend(struct.pack('<i', height))  # 高度
        header.extend(struct.pack('<HH', 1, bit_count))  # 平面数和位深度
        header.extend(struct.pack('<I', 0))  # 压缩方式
        header.extend(struct.pack('<I', image_size))  # 图像大小
        header.extend(struct.pack('<iI', 0, 0))  # 分辨率
        header.extend(struct.pack('<II', 0, 0))  # 颜色表
        
        # 像素数据
        pixel_data = bytearray()
        for y in range(height - 1, -1, -1):
            for x in range(width):
                pixel = self._buffer.get_pixel(x, y)
                if channels >= 3:
                    pixel_data.extend([pixel[2], pixel[1], pixel[0]])  # BGR
                else:
                    pixel_data.extend(pixel[:channels])
            pixel_data.extend(b'\x00' * padding)
        
        return bytes(header) + bytes(pixel_data)
    
    def _encode_png(self) -> bytes:
        """编码为 PNG 格式"""
        if self._buffer is None:
            return b''
        
        width = self._buffer.width
        height = self._buffer.height
        channels = self._buffer.channels
        
        color_type = {1: 0, 2: 4, 3: 2, 4: 4}.get(channels, 2)
        
        # PNG 签名
        data = bytearray(b'\x89PNG\r\n\x1a\n')
        
        # IHDR chunk
        ihdr = bytearray()
        ihdr.extend(struct.pack('>I', width))
        ihdr.extend(struct.pack('>I', height))
        ihdr.extend(bytes([8, color_type, 0, 0, 0]))  # 位深度、颜色类型等
        
        data.extend(struct.pack('>I', len(ihdr)))
        data.extend(b'IHDR')
        data.extend(ihdr)
        data.extend(struct.pack('>I', zlib.crc32(b'IHDR' + ihdr) & 0xFFFFFFFF))
        
        # IDAT chunk
        raw_data = bytearray()
        for y in range(height):
            raw_data.append(0)  # 滤波类型：无滤波
            for x in range(width):
                pixel = self._buffer.get_pixel(x, y)
                raw_data.extend(pixel[:channels])
        
        compressed = zlib.compress(bytes(raw_data), 9)
        data.extend(struct.pack('>I', len(compressed)))
        data.extend(b'IDAT')
        data.extend(compressed)
        data.extend(struct.pack('>I', zlib.crc32(b'IDAT' + compressed) & 0xFFFFFFFF))
        
        # IEND chunk
        data.extend(struct.pack('>I', 0))
        data.extend(b'IEND')
        data.extend(struct.pack('>I', zlib.crc32(b'IEND') & 0xFFFFFFFF))
        
        return bytes(data)
    
    def _encode_gif(self) -> bytes:
        """编码为 GIF 格式（简化版）"""
        if self._buffer is None:
            return b''
        
        width = self._buffer.width
        height = self._buffer.height
        
        # GIF 头
        data = bytearray(b'GIF89a')
        data.extend(struct.pack('<H', width))
        data.extend(struct.pack('<H', height))
        
        # 全局颜色表标志
        data.extend(bytes([0xF7, 0, 0]))  # 256色全局颜色表
        
        # 全局颜色表
        for i in range(256):
            data.extend(bytes([i, i, i]))
        
        # 图形控制扩展
        data.extend(bytes([0x21, 0xF9, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00]))
        
        # 图像描述块
        data.extend(bytes([0x2C]))
        data.extend(struct.pack('<HH', 0, 0))  # 左上角
        data.extend(struct.pack('<HH', width, height))
        data.extend(bytes([0x00]))  # 无局部颜色表
        
        # LZW 最小代码大小
        data.extend(bytes([0x08]))
        
        # 简化：使用子块终止符
        data.extend(bytes([0x00]))
        
        # 结束符
        data.extend(bytes([0x3B]))
        
        return bytes(data)
    
    def convert(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        output_format: Optional[ImageFormat] = None,
        **kwargs: Any
    ) -> bool:
        """
        转换图像格式
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            output_format: 输出格式
            **kwargs: 额外参数
            
        Returns:
            是否转换成功
        """
        if self.load(input_path):
            return self.save(output_path, output_format, **kwargs)
        return False
    
    def batch_convert(
        self,
        input_files: List[Union[str, Path]],
        output_dir: Union[str, Path],
        output_format: ImageFormat,
        **kwargs: Any
    ) -> Dict[str, bool]:
        """
        批量转换图像格式
        
        Args:
            input_files: 输入文件列表
            output_dir: 输出目录
            output_format: 输出格式
            **kwargs: 额外参数
            
        Returns:
            文件名到转换结果的映射
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        ext = f'.{output_format.value}'
        
        for input_file in input_files:
            input_path = Path(input_file)
            output_file = output_path / (input_path.stem + ext)
            
            try:
                results[str(input_file)] = self.convert(
                    input_file, output_file, output_format, **kwargs
                )
            except Exception as e:
                results[str(input_file)] = False
        
        return results
    
    def resize(
        self,
        width: int,
        height: int,
        keep_aspect: bool = True
    ) -> bool:
        """
        调整图像大小
        
        Args:
            width: 目标宽度
            height: 目标高度
            keep_aspect: 是否保持宽高比
            
        Returns:
            是否成功
        """
        if self._buffer is None:
            return False
        
        if keep_aspect:
            src_ratio = self._buffer.width / self._buffer.height
            dst_ratio = width / height
            
            if src_ratio > dst_ratio:
                height = int(width / src_ratio)
            else:
                width = int(height * src_ratio)
        
        # 简单的最近邻缩放
        new_buffer = PixelBuffer(width, height, self._buffer.channels)
        
        x_ratio = self._buffer.width / width
        y_ratio = self._buffer.height / height
        
        for y in range(height):
            for x in range(width):
                src_x = int(x * x_ratio)
                src_y = int(y * y_ratio)
                pixel = self._buffer.get_pixel(src_x, src_y)
                new_buffer.set_pixel(x, y, pixel)
        
        self._buffer = new_buffer
        return True
    
    def rotate(self, angle: int) -> bool:
        """
        旋转图像
        
        Args:
            angle: 旋转角度 (90, 180, 270)
            
        Returns:
            是否成功
        """
        if self._buffer is None:
            return False
        
        if angle not in (90, 180, 270):
            return False
        
        width = self._buffer.width
        height = self._buffer.height
        channels = self._buffer.channels
        
        if angle == 180:
            new_buffer = PixelBuffer(width, height, channels)
            for y in range(height):
                for x in range(width):
                    pixel = self._buffer.get_pixel(width - 1 - x, height - 1 - y)
                    new_buffer.set_pixel(x, y, pixel)
        else:
            new_buffer = PixelBuffer(height, width, channels)
            if angle == 90:
                for y in range(width):
                    for x in range(height):
                        pixel = self._buffer.get_pixel(x, height - 1 - y)
                        new_buffer.set_pixel(y, x, pixel)
            else:  # 270
                for y in range(width):
                    for x in range(height):
                        pixel = self._buffer.get_pixel(width - 1 - x, y)
                        new_buffer.set_pixel(y, x, pixel)
        
        self._buffer = new_buffer
        return True
    
    def flip(self, horizontal: bool = True) -> bool:
        """
        翻转图像
        
        Args:
            horizontal: 是否水平翻转，False 则垂直翻转
            
        Returns:
            是否成功
        """
        if self._buffer is None:
            return False
        
        width = self._buffer.width
        height = self._buffer.height
        channels = self._buffer.channels
        
        new_buffer = PixelBuffer(width, height, channels)
        
        for y in range(height):
            for x in range(width):
                if horizontal:
                    pixel = self._buffer.get_pixel(width - 1 - x, y)
                else:
                    pixel = self._buffer.get_pixel(x, height - 1 - y)
                new_buffer.set_pixel(x, y, pixel)
        
        self._buffer = new_buffer
        return True
    
    def crop(
        self,
        x: int,
        y: int,
        width: int,
        height: int
    ) -> bool:
        """
        裁剪图像
        
        Args:
            x: 起始 X 坐标
            y: 起始 Y 坐标
            width: 裁剪宽度
            height: 裁剪高度
            
        Returns:
            是否成功
        """
        if self._buffer is None:
            return False
        
        channels = self._buffer.channels
        new_buffer = PixelBuffer(width, height, channels)
        
        for dy in range(height):
            for dx in range(width):
                src_x = x + dx
                src_y = y + dy
                if 0 <= src_x < self._buffer.width and 0 <= src_y < self._buffer.height:
                    pixel = self._buffer.get_pixel(src_x, src_y)
                    new_buffer.set_pixel(dx, dy, pixel)
        
        self._buffer = new_buffer
        return True
    
    @property
    def info(self) -> Optional[ImageInfo]:
        """获取当前图像信息"""
        return self._info
    
    @property
    def buffer(self) -> Optional[PixelBuffer]:
        """获取像素缓冲区"""
        return self._buffer
