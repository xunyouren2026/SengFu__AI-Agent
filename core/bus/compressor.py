"""
消息压缩模块

提供gzip、lz4、zstd、snappy等多种压缩算法支持，包括压缩级别协商和自动算法选择。
"""

from __future__ import annotations

import gzip
import logging
import struct
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    Union,
)

logger = logging.getLogger(__name__)


class CompressionError(Exception):
    """压缩错误"""
    pass


class DecompressionError(Exception):
    """解压缩错误"""
    pass


class UnsupportedAlgorithmError(CompressionError):
    """不支持的算法错误"""
    pass


class CompressionAlgorithm(Enum):
    """压缩算法枚举"""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"
    SNAPPY = "snappy"
    
    @classmethod
    def from_string(cls, value: str) -> CompressionAlgorithm:
        """从字符串获取算法"""
        for algo in cls:
            if algo.value == value.lower():
                return algo
        raise UnsupportedAlgorithmError(f"不支持的算法: {value}")


@dataclass
class CompressionStats:
    """压缩统计信息"""
    original_size: int = 0
    compressed_size: int = 0
    compression_ratio: float = 0.0
    compression_time_ms: float = 0.0
    decompression_time_ms: float = 0.0
    algorithm: CompressionAlgorithm = CompressionAlgorithm.NONE
    
    def calculate_ratio(self) -> float:
        """计算压缩比"""
        if self.original_size > 0:
            self.compression_ratio = 1.0 - (self.compressed_size / self.original_size)
        return self.compression_ratio


@dataclass
class CompressionConfig:
    """压缩配置"""
    default_algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP
    default_level: int = 6
    min_size_threshold: int = 1024  # 小于此大小不压缩
    max_size_threshold: int = 100 * 1024 * 1024  # 100MB
    enable_auto_selection: bool = True
    preferred_algorithms: List[CompressionAlgorithm] = field(
        default_factory=lambda: [
            CompressionAlgorithm.ZSTD,
            CompressionAlgorithm.LZ4,
            CompressionAlgorithm.GZIP,
            CompressionAlgorithm.SNAPPY,
        ]
    )


class Compressor(ABC):
    """压缩器抽象基类"""
    
    algorithm: ClassVar[CompressionAlgorithm] = CompressionAlgorithm.NONE
    
    def __init__(self, level: int = 6) -> None:
        self.level = level
        self._stats = CompressionStats(algorithm=self.algorithm)
    
    @abstractmethod
    def compress(self, data: bytes) -> bytes:
        """压缩数据"""
        pass
    
    @abstractmethod
    def decompress(self, data: bytes) -> bytes:
        """解压缩数据"""
        pass
    
    @property
    def stats(self) -> CompressionStats:
        """获取统计信息"""
        return self._stats
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self._stats = CompressionStats(algorithm=self.algorithm)


class GzipCompressor(Compressor):
    """Gzip压缩器"""
    
    algorithm = CompressionAlgorithm.GZIP
    
    # 压缩级别映射 (1-9)
    LEVEL_FASTEST = 1
    LEVEL_DEFAULT = 6
    LEVEL_BEST = 9
    
    def __init__(self, level: int = 6) -> None:
        super().__init__(level)
        if not 1 <= level <= 9:
            raise ValueError("Gzip压缩级别必须在1-9之间")
    
    def compress(self, data: bytes) -> bytes:
        """使用gzip压缩数据"""
        import time
        
        start_time = time.perf_counter()
        
        try:
            compressed = gzip.compress(data, compresslevel=self.level)
            
            elapsed = (time.perf_counter() - start_time) * 1000
            
            self._stats.original_size += len(data)
            self._stats.compressed_size += len(compressed)
            self._stats.compression_time_ms += elapsed
            self._stats.calculate_ratio()
            
            return compressed
        except Exception as e:
            raise CompressionError(f"Gzip压缩失败: {e}")
    
    def decompress(self, data: bytes) -> bytes:
        """使用gzip解压缩数据"""
        import time
        
        start_time = time.perf_counter()
        
        try:
            decompressed = gzip.decompress(data)
            
            elapsed = (time.perf_counter() - start_time) * 1000
            self._stats.decompression_time_ms += elapsed
            
            return decompressed
        except Exception as e:
            raise DecompressionError(f"Gzip解压缩失败: {e}")
    
    @classmethod
    def get_supported_levels(cls) -> List[int]:
        """获取支持的压缩级别"""
        return list(range(1, 10))
    
    @classmethod
    def estimate_compression_ratio(cls, level: int, data_sample: bytes) -> float:
        """估计压缩比"""
        test_data = data_sample[:min(1024, len(data_sample))]
        compressed = gzip.compress(test_data, compresslevel=level)
        return 1.0 - (len(compressed) / len(test_data)) if test_data else 0.0


class LZ4Compressor(Compressor):
    """LZ4压缩器（纯Python实现）"""
    
    algorithm = CompressionAlgorithm.LZ4
    
    # LZ4压缩级别
    LEVEL_FASTEST = 1
    LEVEL_DEFAULT = 1
    LEVEL_BEST = 12
    
    def __init__(self, level: int = 1) -> None:
        super().__init__(level)
        if not 1 <= level <= 12:
            raise ValueError("LZ4压缩级别必须在1-12之间")
        
        # LZ4压缩参数
        self._acceleration = max(1, 13 - level)  # 级别越高，acceleration越小
    
    def compress(self, data: bytes) -> bytes:
        """使用LZ4算法压缩数据（简化实现）"""
        import time
        
        start_time = time.perf_counter()
        
        try:
            # 使用简单的LZ4帧格式模拟
            # 实际格式: Magic(4) + FLG(1) + BD(1) + [Content Size(8)] + [Dict ID(4)] + HC(1) + Data + EndMark(4) + CS(4)
            
            # 这里使用简化的RLE+字典压缩模拟LZ4
            compressed = self._lz4_compress(data)
            
            elapsed = (time.perf_counter() - start_time) * 1000
            
            self._stats.original_size += len(data)
            self._stats.compressed_size += len(compressed)
            self._stats.compression_time_ms += elapsed
            self._stats.calculate_ratio()
            
            return compressed
        except Exception as e:
            raise CompressionError(f"LZ4压缩失败: {e}")
    
    def _lz4_compress(self, data: bytes) -> bytes:
        """简化的LZ4压缩实现"""
        if len(data) < 16:
            # 小数据不压缩
            return b"\x04\x22\x4d\x18" + struct.pack(">I", len(data)) + data + b"\x00\x00\x00\x00"
        
        # LZ4帧魔数
        magic = b"\x04\x22\x4d\x18"
        
        # 帧描述符
        flg = 0x68  # 版本1, 块独立, 有内容大小
        bd = 0x70   # 最大块大小64KB
        
        # 内容大小
        content_size = struct.pack("<Q", len(data))
        
        # 计算头部校验
        header = bytes([flg, bd]) + content_size
        hc = self._xxh32(header, 0) >> 8
        
        # 压缩数据块（使用简化的LZ4块格式）
        compressed_blocks = self._compress_blocks(data)
        
        # 结束标记
        end_mark = b"\x00\x00\x00\x00"
        
        return magic + header + bytes([hc]) + compressed_blocks + end_mark
    
    def _compress_blocks(self, data: bytes, block_size: int = 64 * 1024) -> bytes:
        """压缩数据块"""
        blocks = []
        
        for i in range(0, len(data), block_size):
            chunk = data[i:i + block_size]
            compressed = self._lz4_block_compress(chunk)
            
            # 块大小（最高位表示是否压缩）
            if len(compressed) < len(chunk):
                block_header = struct.pack("<I", len(compressed) | 0x80000000)
            else:
                block_header = struct.pack("<I", len(chunk))
                compressed = chunk
            
            blocks.append(block_header + compressed)
        
        return b"".join(blocks)
    
    def _lz4_block_compress(self, data: bytes) -> bytes:
        """LZ4块压缩（简化实现）"""
        if not data:
            return b""
        
        # 简化的LZ4压缩：使用基本的RLE和重复检测
        result = bytearray()
        pos = 0
        
        while pos < len(data):
            # 查找匹配
            match_len, match_offset = self._find_match(data, pos)
            
            if match_len >= 4:
                # 编码匹配
                literal_len = 0
                if pos > 0:
                    literal_len = min(pos, 15)
                
                token = (literal_len << 4) | min(match_len - 4, 15)
                result.append(token)
                
                if literal_len > 0:
                    result.extend(data[pos - literal_len:pos])
                
                # 偏移量（小端）
                result.extend(struct.pack("<H", match_offset))
                
                if match_len - 4 >= 15:
                    result.append(match_len - 4 - 15)
                
                pos += match_len
            else:
                # 字面量
                if not result or (result[-1] >> 4) >= 15:
                    result.append(0xF0)
                else:
                    result[-1] += 0x10
                
                # 添加字面量长度扩展
                lit_len = min(255, len(data) - pos)
                if lit_len >= 15:
                    result.append(lit_len - 15)
                
                result.extend(data[pos:pos + lit_len])
                pos += lit_len
        
        return bytes(result)
    
    def _find_match(self, data: bytes, pos: int) -> Tuple[int, int]:
        """查找重复匹配"""
        if pos < 4:
            return 0, 0
        
        # 搜索窗口大小
        window_size = min(65535, pos)
        search_start = max(0, pos - window_size)
        
        best_len = 0
        best_offset = 0
        
        # 简化：只检查最近的位置
        for offset in range(4, min(window_size, pos) + 1):
            match_pos = pos - offset
            match_len = 0
            
            while (match_pos + match_len < pos and 
                   pos + match_len < len(data) and
                   data[match_pos + match_len] == data[pos + match_len] and
                   match_len < 65535):
                match_len += 1
            
            if match_len > best_len:
                best_len = match_len
                best_offset = offset
        
        return best_len, best_offset
    
    def _xxh32(self, data: bytes, seed: int) -> int:
        """简化的xxHash32"""
        h = seed + 0x165667B1
        
        # 简化处理
        for i in range(0, len(data), 4):
            chunk = data[i:i+4]
            if len(chunk) < 4:
                chunk = chunk + b'\x00' * (4 - len(chunk))
            val = struct.unpack("<I", chunk)[0]
            h = (h + val * 0x9E3779B1) & 0xFFFFFFFF
            h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
            h = (h * 5 + 0xE6546B64) & 0xFFFFFFFF
        
        # 终结混合
        h ^= h >> 16
        h = (h * 0x85EBCA6B) & 0xFFFFFFFF
        h ^= h >> 13
        h = (h * 0xC2B2AE35) & 0xFFFFFFFF
        h ^= h >> 16
        
        return h
    
    def decompress(self, data: bytes) -> bytes:
        """使用LZ4算法解压缩数据"""
        import time
        
        start_time = time.perf_counter()
        
        try:
            # 验证魔数
            if not data.startswith(b"\x04\x22\x4d\x18"):
                raise DecompressionError("无效的LZ4帧")
            
            decompressed = self._lz4_decompress(data)
            
            elapsed = (time.perf_counter() - start_time) * 1000
            self._stats.decompression_time_ms += elapsed
            
            return decompressed
        except Exception as e:
            raise DecompressionError(f"LZ4解压缩失败: {e}")
    
    def _lz4_decompress(self, data: bytes) -> bytes:
        """简化的LZ4解压缩"""
        pos = 4  # 跳过魔数
        
        # 读取帧描述符
        flg = data[pos]
        pos += 1
        bd = data[pos]
        pos += 1
        
        # 内容大小（如果存在）
        content_size = 0
        if flg & 0x08:
            content_size = struct.unpack("<Q", data[pos:pos+8])[0]
            pos += 8
        
        # 字典ID（如果存在）
        if flg & 0x01:
            pos += 4
        
        # 跳过头部校验
        pos += 1
        
        # 解压缩块
        result = bytearray()
        
        while pos < len(data) - 4:
            block_size = struct.unpack("<I", data[pos:pos+4])[0]
            pos += 4
            
            if block_size == 0:
                break
            
            is_compressed = block_size & 0x80000000
            block_size = block_size & 0x7FFFFFFF
            
            block_data = data[pos:pos + block_size]
            pos += block_size
            
            if is_compressed:
                result.extend(self._decompress_block(bytes(block_data)))
            else:
                result.extend(block_data)
        
        return bytes(result)
    
    def _decompress_block(self, data: bytes) -> bytes:
        """解压缩单个块"""
        result = bytearray()
        pos = 0
        
        while pos < len(data):
            token = data[pos]
            pos += 1
            
            literal_len = token >> 4
            match_len = token & 0x0F
            
            # 读取额外字面量长度
            if literal_len == 15:
                while pos < len(data) and data[pos] == 255:
                    literal_len += 255
                    pos += 1
                if pos < len(data):
                    literal_len += data[pos]
                    pos += 1
            
            # 复制字面量
            if literal_len > 0:
                result.extend(data[pos:pos + literal_len])
                pos += literal_len
            
            if pos >= len(data):
                break
            
            # 读取偏移量
            if pos + 1 < len(data):
                offset = struct.unpack("<H", data[pos:pos+2])[0]
                pos += 2
                
                # 读取额外匹配长度
                if match_len == 15:
                    while pos < len(data) and data[pos] == 255:
                        match_len += 255
                        pos += 1
                    if pos < len(data):
                        match_len += data[pos]
                        pos += 1
                
                match_len += 4
                
                # 复制匹配
                match_start = len(result) - offset
                for i in range(match_len):
                    if match_start + i >= 0:
                        result.append(result[match_start + i])
        
        return bytes(result)


class ZstdCompressor(Compressor):
    """Zstandard压缩器（纯Python实现）"""
    
    algorithm = CompressionAlgorithm.ZSTD
    
    # Zstd压缩级别
    LEVEL_FASTEST = 1
    LEVEL_DEFAULT = 3
    LEVEL_BEST = 22
    
    def __init__(self, level: int = 3) -> None:
        super().__init__(level)
        if not 1 <= level <= 22:
            raise ValueError("Zstd压缩级别必须在1-22之间")
    
    def compress(self, data: bytes) -> bytes:
        """使用Zstandard压缩数据（简化实现）"""
        import time
        
        start_time = time.perf_counter()
        
        try:
            # 简化的Zstd帧格式实现
            compressed = self._zstd_compress(data)
            
            elapsed = (time.perf_counter() - start_time) * 1000
            
            self._stats.original_size += len(data)
            self._stats.compressed_size += len(compressed)
            self._stats.compression_time_ms += elapsed
            self._stats.calculate_ratio()
            
            return compressed
        except Exception as e:
            raise CompressionError(f"Zstd压缩失败: {e}")
    
    def _zstd_compress(self, data: bytes) -> bytes:
        """简化的Zstd压缩"""
        # Zstd魔数
        magic = b"\x28\xb5\x2f\xfd"
        
        # 帧头部
        # 帧内容大小标志(1) + 单段标志(0) + 内容大小(2字节) + 窗口描述符
        frame_header = self._create_frame_header(len(data))
        
        # 压缩数据块
        compressed_data = self._compress_zstd_blocks(data)
        
        # 帧校验（可选）
        checksum = struct.pack("<I", self._xxh32(compressed_data, 0))
        
        return magic + frame_header + compressed_data + checksum
    
    def _create_frame_header(self, content_size: int) -> bytes:
        """创建Zstd帧头部"""
        # 简化的帧头部
        # FCS = 2 (内容大小字段大小)
        # Single_Segment_Flag = 1
        # Frame_Content_Size = content_size
        
        if content_size < 256:
            fcs_field = bytes([content_size])
            fcs_flag = 0
        elif content_size < 65536:
            fcs_field = struct.pack("<H", content_size)
            fcs_flag = 1
        else:
            fcs_field = struct.pack("<I", content_size)[:3] + b'\x00'
            fcs_flag = 2
        
        # 帧头部描述符
        fhd = (fcs_flag << 6) | 0x20  # 单段标志
        
        # 窗口描述符（单段模式下不需要）
        
        return bytes([fhd]) + fcs_field
    
    def _compress_zstd_blocks(self, data: bytes) -> bytes:
        """压缩Zstd数据块"""
        # 简化的块压缩：使用LZ77变体
        block_size = min(len(data), 128 * 1024)  # 最大128KB
        
        if len(data) <= block_size:
            compressed = self._compress_single_block(data)
            # 块头部：最后块标志(1) + 块类型(2) + 块大小(21)
            if len(compressed) < len(data):
                block_header = self._encode_block_header(len(compressed), 1, 1)  # 压缩块
                return block_header + compressed
            else:
                block_header = self._encode_block_header(len(data), 0, 1)  # 原始块
                return block_header + data
        
        # 多块处理
        blocks = []
        for i in range(0, len(data), block_size):
            chunk = data[i:i + block_size]
            is_last = 1 if i + block_size >= len(data) else 0
            
            compressed = self._compress_single_block(chunk)
            if len(compressed) < len(chunk):
                block_header = self._encode_block_header(len(compressed), 1, is_last)
                blocks.append(block_header + compressed)
            else:
                block_header = self._encode_block_header(len(chunk), 0, is_last)
                blocks.append(block_header + chunk)
        
        return b"".join(blocks)
    
    def _encode_block_header(self, size: int, block_type: int, last_block: int) -> bytes:
        """编码块头部"""
        # 24位：最后块(1) + 块类型(2) + 块大小(21)
        header = (last_block << 23) | (block_type << 21) | size
        return struct.pack("<I", header)[:3]
    
    def _compress_single_block(self, data: bytes) -> bytes:
        """压缩单个块"""
        # 简化的LZ77压缩
        result = bytearray()
        pos = 0
        
        # 使用FSE（有限状态熵）编码的简化版本
        while pos < len(data):
            # 查找最长匹配
            match_len, match_offset = self._find_best_match(data, pos)
            
            if match_len >= 4:
                # 编码匹配
                lit_len = pos - max(0, pos - match_offset)
                result.extend(self._encode_sequence(lit_len, match_len, match_offset))
                pos += match_len
            else:
                # 字面量
                lit_len = min(len(data) - pos, 127)
                result.extend(self._encode_literals(data[pos:pos + lit_len]))
                pos += lit_len
        
        return bytes(result)
    
    def _find_best_match(self, data: bytes, pos: int) -> Tuple[int, int]:
        """查找最佳匹配"""
        if pos < 4:
            return 0, 0
        
        best_len = 0
        best_offset = 0
        
        # 搜索窗口
        search_start = max(0, pos - 32768)  # 32KB窗口
        
        for offset in range(1, min(pos - search_start + 1, 65536)):
            match_pos = pos - offset
            match_len = 0
            
            while (match_pos + match_len < pos and
                   pos + match_len < len(data) and
                   data[match_pos + match_len] == data[pos + match_len] and
                   match_len < 131072):  # 最大匹配长度
                match_len += 1
            
            if match_len > best_len:
                best_len = match_len
                best_offset = offset
        
        return best_len, best_offset
    
    def _encode_sequence(self, lit_len: int, match_len: int, offset: int) -> bytes:
        """编码序列"""
        # 简化的序列编码
        result = bytearray()
        
        # 字面量长度编码
        if lit_len < 63:
            result.append(lit_len)
        else:
            result.append(63)
            result.extend(self._encode_variable_length(lit_len - 63))
        
        # 偏移量编码（小端）
        result.extend(struct.pack("<H", offset))
        
        # 匹配长度编码
        match_code = min(match_len - 3, 31)
        result.append(match_code)
        if match_len >= 34:
            result.extend(self._encode_variable_length(match_len - 34))
        
        return bytes(result)
    
    def _encode_literals(self, literals: bytes) -> bytes:
        """编码字面量"""
        result = bytearray()
        
        # 字面量块类型和大小
        size = len(literals)
        if size < 63:
            result.append(0x40 | size)  # 原始字面量类型
        else:
            result.append(0x40 | 63)
            result.extend(self._encode_variable_length(size - 63))
        
        result.extend(literals)
        return bytes(result)
    
    def _encode_variable_length(self, value: int) -> bytes:
        """编码变长整数"""
        result = bytearray()
        while value >= 255:
            result.append(255)
            value -= 255
        result.append(value)
        return bytes(result)
    
    def _xxh32(self, data: bytes, seed: int) -> int:
        """简化的xxHash32校验"""
        h = seed + 0x165667B1
        
        for i in range(0, len(data), 4):
            chunk = data[i:i+4]
            if len(chunk) < 4:
                chunk = chunk + b'\x00' * (4 - len(chunk))
            val = struct.unpack("<I", chunk)[0]
            h = (h + val * 0x9E3779B1) & 0xFFFFFFFF
            h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
            h = (h * 5 + 0xE6546B64) & 0xFFFFFFFF
        
        h ^= h >> 16
        h = (h * 0x85EBCA6B) & 0xFFFFFFFF
        h ^= h >> 13
        h = (h * 0xC2B2AE35) & 0xFFFFFFFF
        h ^= h >> 16
        
        return h
    
    def decompress(self, data: bytes) -> bytes:
        """使用Zstandard解压缩数据"""
        import time
        
        start_time = time.perf_counter()
        
        try:
            # 验证魔数
            if not data.startswith(b"\x28\xb5\x2f\xfd"):
                raise DecompressionError("无效的Zstd帧")
            
            decompressed = self._zstd_decompress(data)
            
            elapsed = (time.perf_counter() - start_time) * 1000
            self._stats.decompression_time_ms += elapsed
            
            return decompressed
        except Exception as e:
            raise DecompressionError(f"Zstd解压缩失败: {e}")
    
    def _zstd_decompress(self, data: bytes) -> bytes:
        """简化的Zstd解压缩"""
        pos = 4  # 跳过魔数
        
        # 解析帧头部
        fhd = data[pos]
        pos += 1
        
        fcs_flag = (fhd >> 6) & 0x03
        single_segment = (fhd >> 5) & 0x01
        
        # 读取内容大小
        content_size = 0
        if fcs_flag == 0:
            if single_segment:
                content_size = data[pos]
                pos += 1
        elif fcs_flag == 1:
            content_size = struct.unpack("<H", data[pos:pos+2])[0]
            pos += 2
        elif fcs_flag == 2:
            content_size = struct.unpack("<I", data[pos:pos+4])[0]
            pos += 4
        else:
            content_size = struct.unpack("<Q", data[pos:pos+8])[0]
            pos += 8
        
        # 解压缩块
        result = bytearray()
        
        while pos < len(data) - 4:
            # 读取块头部
            if pos + 3 > len(data):
                break
            
            header = struct.unpack("<I", data[pos:pos+3] + b'\x00')[0]
            pos += 3
            
            last_block = (header >> 23) & 0x01
            block_type = (header >> 21) & 0x03
            block_size = header & 0x1FFFFF
            
            if block_size == 0:
                break
            
            block_data = data[pos:pos + block_size]
            pos += block_size
            
            if block_type == 0:  # 原始块
                result.extend(block_data)
            elif block_type == 1:  # RLE块
                if block_data:
                    result.extend(block_data[0:1] * block_size)
            elif block_type == 2:  # 压缩块
                result.extend(self._decompress_zstd_block(block_data))
            
            if last_block:
                break
        
        return bytes(result)
    
    def _decompress_zstd_block(self, data: bytes) -> bytes:
        """解压缩Zstd块"""
        # 简化解压缩
        result = bytearray()
        pos = 0
        
        while pos < len(data):
            # 读取字面量长度
            lit_len_code = data[pos] & 0x3F
            pos += 1
            
            if lit_len_code == 63:
                lit_len = 63 + self._decode_variable_length(data, pos)
                pos += self._variable_length_size(data, pos)
            else:
                lit_len = lit_len_code
            
            # 字面量
            if lit_len > 0:
                result.extend(data[pos:pos + lit_len])
                pos += lit_len
            
            if pos >= len(data):
                break
            
            # 偏移量
            offset = struct.unpack("<H", data[pos:pos+2])[0]
            pos += 2
            
            # 匹配长度
            match_code = data[pos] if pos < len(data) else 0
            pos += 1
            
            if match_code == 31:
                match_len = 34 + self._decode_variable_length(data, pos)
                pos += self._variable_length_size(data, pos)
            else:
                match_len = match_code + 3
            
            # 复制匹配
            match_start = len(result) - offset
            for i in range(match_len):
                if match_start + i >= 0:
                    result.append(result[match_start + i])
        
        return bytes(result)
    
    def _decode_variable_length(self, data: bytes, pos: int) -> int:
        """解码变长整数"""
        value = 0
        while pos < len(data) and data[pos] == 255:
            value += 255
            pos += 1
        if pos < len(data):
            value += data[pos]
        return value
    
    def _variable_length_size(self, data: bytes, pos: int) -> int:
        """获取变长整数大小"""
        size = 0
        while pos + size < len(data) and data[pos + size] == 255:
            size += 1
        return size + 1


class SnappyCompressor(Compressor):
    """Snappy压缩器（纯Python实现）"""
    
    algorithm = CompressionAlgorithm.SNAPPY
    
    # Snappy使用固定压缩级别
    LEVEL_DEFAULT = 0
    
    # 流标识符
    STREAM_IDENTIFIER = b"\xff\x06\x00\x00\x73\x4e\x61\x50\x70\x59"
    
    def __init__(self, level: int = 0) -> None:
        super().__init__(level)
    
    def compress(self, data: bytes) -> bytes:
        """使用Snappy压缩数据"""
        import time
        
        start_time = time.perf_counter()
        
        try:
            compressed = self._snappy_compress(data)
            
            elapsed = (time.perf_counter() - start_time) * 1000
            
            self._stats.original_size += len(data)
            self._stats.compressed_size += len(compressed)
            self._stats.compression_time_ms += elapsed
            self._stats.calculate_ratio()
            
            return compressed
        except Exception as e:
            raise CompressionError(f"Snappy压缩失败: {e}")
    
    def _snappy_compress(self, data: bytes) -> bytes:
        """Snappy压缩实现"""
        # 简化的Snappy流格式
        result = bytearray()
        
        # 流标识符
        result.extend(self.STREAM_IDENTIFIER)
        
        # 压缩数据块
        chunk_size = 65536
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            compressed_chunk = self._compress_chunk(chunk)
            
            # 块类型和大小
            if len(compressed_chunk) < len(chunk) * 0.9:  # 压缩有效
                # 压缩块类型 = 0x00
                result.append(0x00)
                result.extend(struct.pack("<I", len(compressed_chunk)))
                result.extend(compressed_chunk)
            else:
                # 未压缩块类型 = 0x01
                result.append(0x01)
                result.extend(struct.pack("<I", len(chunk)))
                result.extend(chunk)
        
        return bytes(result)
    
    def _compress_chunk(self, data: bytes) -> bytes:
        """压缩数据块"""
        result = bytearray()
        pos = 0
        
        while pos < len(data):
            # 查找匹配
            match_len, match_offset = self._find_snappy_match(data, pos)
            
            if match_len >= 4:
                # 编码复制命令
                literal_len = pos - max(0, pos - match_offset)
                
                if literal_len > 0:
                    # 先输出字面量
                    result.extend(self._encode_literal(data[pos - literal_len:pos]))
                
                # 输出复制命令
                result.extend(self._encode_copy(match_offset, match_len))
                pos += match_len
            else:
                # 字面量
                lit_len = min(60, len(data) - pos)
                result.extend(self._encode_literal(data[pos:pos + lit_len]))
                pos += lit_len
        
        return bytes(result)
    
    def _find_snappy_match(self, data: bytes, pos: int) -> Tuple[int, int]:
        """查找Snappy匹配"""
        if pos < 4:
            return 0, 0
        
        best_len = 0
        best_offset = 0
        
        # 16KB窗口
        window_size = min(16384, pos)
        
        for offset in range(4, min(window_size + 1, 65536)):
            match_pos = pos - offset
            match_len = 0
            
            while (match_pos + match_len < pos and
                   pos + match_len < len(data) and
                   data[match_pos + match_len] == data[pos + match_len] and
                   match_len < 64):
                match_len += 1
            
            if match_len > best_len:
                best_len = match_len
                best_offset = offset
        
        return best_len, best_offset
    
    def _encode_literal(self, data: bytes) -> bytes:
        """编码字面量"""
        length = len(data)
        
        if length < 60:
            tag = (length + 1) << 2
            return bytes([tag]) + data
        elif length < 256:
            tag = 60 << 2
            return bytes([tag, length - 60]) + data
        else:
            tag = 61 << 2
            return bytes([tag]) + struct.pack("<H", length - 60) + data
    
    def _encode_copy(self, offset: int, length: int) -> bytes:
        """编码复制命令"""
        if offset < 256 and length < 12:
            # 1字节偏移量
            tag = 1 | ((length - 4) << 2)
            return bytes([tag, offset])
        elif offset < 65536 and length < 64:
            # 2字节偏移量
            if length < 12:
                tag = 2 | ((length - 4) << 2)
            else:
                tag = 3 | ((length - 1) << 2)
            return bytes([tag]) + struct.pack("<H", offset)
        else:
            # 4字节偏移量
            tag = 3 | ((length - 1) << 2)
            return bytes([tag]) + struct.pack("<I", offset)
    
    def decompress(self, data: bytes) -> bytes:
        """使用Snappy解压缩数据"""
        import time
        
        start_time = time.perf_counter()
        
        try:
            # 验证流标识符
            if data.startswith(self.STREAM_IDENTIFIER):
                pos = len(self.STREAM_IDENTIFIER)
            else:
                pos = 0
            
            decompressed = self._snappy_decompress(data, pos)
            
            elapsed = (time.perf_counter() - start_time) * 1000
            self._stats.decompression_time_ms += elapsed
            
            return decompressed
        except Exception as e:
            raise DecompressionError(f"Snappy解压缩失败: {e}")
    
    def _snappy_decompress(self, data: bytes, pos: int) -> bytes:
        """Snappy解压缩"""
        result = bytearray()
        
        while pos < len(data):
            if pos >= len(data):
                break
            
            chunk_type = data[pos]
            pos += 1
            
            if chunk_type == 0x00:  # 压缩块
                if pos + 4 > len(data):
                    break
                chunk_len = struct.unpack("<I", data[pos:pos+4])[0]
                pos += 4
                
                if pos + chunk_len > len(data):
                    break
                
                compressed = data[pos:pos + chunk_len]
                pos += chunk_len
                
                result.extend(self._decompress_chunk(compressed))
                
            elif chunk_type == 0x01:  # 未压缩块
                if pos + 4 > len(data):
                    break
                chunk_len = struct.unpack("<I", data[pos:pos+4])[0]
                pos += 4
                
                if pos + chunk_len > len(data):
                    break
                
                result.extend(data[pos:pos + chunk_len])
                pos += chunk_len
                
            else:
                break
        
        return bytes(result)
    
    def _decompress_chunk(self, data: bytes) -> bytes:
        """解压缩数据块"""
        result = bytearray()
        pos = 0
        
        while pos < len(data):
            if pos >= len(data):
                break
            
            tag = data[pos]
            pos += 1
            
            tag_type = tag & 0x03
            
            if tag_type == 0x00:  # 字面量
                length = (tag >> 2) + 1
                if length <= 60:
                    pass  # 长度已在tag中
                elif length == 61:
                    if pos < len(data):
                        length = data[pos] + 61
                        pos += 1
                elif length == 62:
                    if pos + 1 < len(data):
                        length = struct.unpack("<H", data[pos:pos+2])[0] + 61
                        pos += 2
                
                if pos + length <= len(data):
                    result.extend(data[pos:pos + length])
                    pos += length
                    
            elif tag_type == 0x01:  # 复制，1字节偏移
                if pos < len(data):
                    length = ((tag >> 2) & 0x07) + 4
                    offset = data[pos]
                    pos += 1
                    
                    match_start = len(result) - offset
                    for i in range(length):
                        if match_start + i >= 0:
                            result.append(result[match_start + i])
                            
            elif tag_type == 0x02:  # 复制，2字节偏移
                if pos + 1 < len(data):
                    length = ((tag >> 2) & 0x3F) + 1
                    offset = struct.unpack("<H", data[pos:pos+2])[0]
                    pos += 2
                    
                    match_start = len(result) - offset
                    for i in range(length):
                        if match_start + i >= 0:
                            result.append(result[match_start + i])
                            
            elif tag_type == 0x03:  # 复制，4字节偏移
                if pos + 3 < len(data):
                    length = ((tag >> 2) & 0x3F) + 1
                    offset = struct.unpack("<I", data[pos:pos+4])[0]
                    pos += 4
                    
                    match_start = len(result) - offset
                    for i in range(length):
                        if match_start + i >= 0:
                            result.append(result[match_start + i])
        
        return bytes(result)


class CompressionNegotiator:
    """压缩协商器"""
    
    def __init__(self) -> None:
        self._supported_algorithms: Dict[CompressionAlgorithm, Compressor] = {
            CompressionAlgorithm.GZIP: GzipCompressor(),
            CompressionAlgorithm.LZ4: LZ4Compressor(),
            CompressionAlgorithm.ZSTD: ZstdCompressor(),
            CompressionAlgorithm.SNAPPY: SnappyCompressor(),
        }
        self._negotiated: Dict[str, CompressionAlgorithm] = {}
        self._levels: Dict[str, int] = {}
    
    def negotiate(
        self,
        peer_id: str,
        supported: List[CompressionAlgorithm],
        preferred: Optional[CompressionAlgorithm] = None,
    ) -> CompressionAlgorithm:
        """协商压缩算法"""
        # 找到双方都支持的算法
        common = set(supported) & set(self._supported_algorithms.keys())
        
        if not common:
            return CompressionAlgorithm.NONE
        
        # 优先使用对方的首选
        if preferred and preferred in common:
            selected = preferred
        else:
            # 按优先级排序
            priority = [
                CompressionAlgorithm.ZSTD,
                CompressionAlgorithm.LZ4,
                CompressionAlgorithm.SNAPPY,
                CompressionAlgorithm.GZIP,
            ]
            selected = CompressionAlgorithm.NONE
            for algo in priority:
                if algo in common:
                    selected = algo
                    break
        
        self._negotiated[peer_id] = selected
        return selected
    
    def set_compression_level(self, peer_id: str, level: int) -> None:
        """设置压缩级别"""
        self._levels[peer_id] = level
    
    def get_compressor(self, peer_id: str) -> Optional[Compressor]:
        """获取协商后的压缩器"""
        algo = self._negotiated.get(peer_id)
        if not algo or algo == CompressionAlgorithm.NONE:
            return None
        
        level = self._levels.get(peer_id, 6)
        
        if algo == CompressionAlgorithm.GZIP:
            return GzipCompressor(level)
        elif algo == CompressionAlgorithm.LZ4:
            return LZ4Compressor(level)
        elif algo == CompressionAlgorithm.ZSTD:
            return ZstdCompressor(level)
        elif algo == CompressionAlgorithm.SNAPPY:
            return SnappyCompressor()
        
        return None
    
    def get_supported_algorithms(self) -> List[CompressionAlgorithm]:
        """获取支持的算法列表"""
        return list(self._supported_algorithms.keys())
    
    def clear_negotiation(self, peer_id: str) -> None:
        """清除协商结果"""
        if peer_id in self._negotiated:
            del self._negotiated[peer_id]
        if peer_id in self._levels:
            del self._levels[peer_id]


class AutoCompressor:
    """自动压缩器"""
    
    def __init__(self, config: Optional[CompressionConfig] = None) -> None:
        self.config = config or CompressionConfig()
        self._compressors: Dict[CompressionAlgorithm, Compressor] = {}
        self._init_compressors()
    
    def _init_compressors(self) -> None:
        """初始化压缩器"""
        self._compressors[CompressionAlgorithm.GZIP] = GzipCompressor(self.config.default_level)
        self._compressors[CompressionAlgorithm.LZ4] = LZ4Compressor(1)
        self._compressors[CompressionAlgorithm.ZSTD] = ZstdCompressor(3)
        self._compressors[CompressionAlgorithm.SNAPPY] = SnappyCompressor()
    
    def compress(self, data: bytes, algorithm: Optional[CompressionAlgorithm] = None) -> Tuple[bytes, CompressionStats]:
        """自动压缩数据"""
        # 检查大小阈值
        if len(data) < self.config.min_size_threshold:
            stats = CompressionStats(
                original_size=len(data),
                compressed_size=len(data),
                algorithm=CompressionAlgorithm.NONE,
            )
            return data, stats
        
        if len(data) > self.config.max_size_threshold:
            raise CompressionError(f"数据大小超过最大阈值: {len(data)} > {self.config.max_size_threshold}")
        
        # 选择算法
        if algorithm is None and self.config.enable_auto_selection:
            algorithm = self._select_algorithm(data)
        elif algorithm is None:
            algorithm = self.config.default_algorithm
        
        if algorithm == CompressionAlgorithm.NONE:
            stats = CompressionStats(
                original_size=len(data),
                compressed_size=len(data),
                algorithm=CompressionAlgorithm.NONE,
            )
            return data, stats
        
        # 执行压缩
        compressor = self._compressors.get(algorithm)
        if not compressor:
            raise UnsupportedAlgorithmError(f"不支持的算法: {algorithm}")
        
        compressed = compressor.compress(data)
        stats = CompressionStats(
            original_size=len(data),
            compressed_size=len(compressed),
            compression_ratio=1.0 - len(compressed) / len(data),
            algorithm=algorithm,
        )
        
        # 如果压缩效果不好，返回原始数据
        if stats.compression_ratio < 0.1:
            stats = CompressionStats(
                original_size=len(data),
                compressed_size=len(data),
                algorithm=CompressionAlgorithm.NONE,
            )
            return data, stats
        
        return compressed, stats
    
    def decompress(self, data: bytes, algorithm: CompressionAlgorithm) -> bytes:
        """解压缩数据"""
        if algorithm == CompressionAlgorithm.NONE:
            return data
        
        compressor = self._compressors.get(algorithm)
        if not compressor:
            raise UnsupportedAlgorithmError(f"不支持的算法: {algorithm}")
        
        return compressor.decompress(data)
    
    def _select_algorithm(self, data: bytes) -> CompressionAlgorithm:
        """根据数据特征选择最佳算法"""
        size = len(data)
        
        # 根据数据大小选择
        if size < 1024:
            # 小数据：使用Snappy（低延迟）
            return CompressionAlgorithm.SNAPPY
        elif size < 64 * 1024:
            # 中等数据：使用LZ4
            return CompressionAlgorithm.LZ4
        elif size < 1024 * 1024:
            # 大数据：使用Zstd
            return CompressionAlgorithm.ZSTD
        else:
            # 超大数据：使用Gzip（高压缩比）
            return CompressionAlgorithm.GZIP
    
    def benchmark(self, data: bytes) -> Dict[CompressionAlgorithm, CompressionStats]:
        """对所有算法进行基准测试"""
        results = {}
        
        for algo, compressor in self._compressors.items():
            try:
                import time
                start = time.perf_counter()
                compressed = compressor.compress(data)
                compress_time = (time.perf_counter() - start) * 1000
                
                start = time.perf_counter()
                compressor.decompress(compressed)
                decompress_time = (time.perf_counter() - start) * 1000
                
                stats = CompressionStats(
                    original_size=len(data),
                    compressed_size=len(compressed),
                    compression_ratio=1.0 - len(compressed) / len(data),
                    compression_time_ms=compress_time,
                    decompression_time_ms=decompress_time,
                    algorithm=algo,
                )
                results[algo] = stats
                
            except Exception as e:
                logger.warning(f"算法 {algo} 基准测试失败: {e}")
        
        return results
    
    def get_stats(self) -> Dict[CompressionAlgorithm, CompressionStats]:
        """获取所有压缩器的统计信息"""
        return {algo: comp.stats for algo, comp in self._compressors.items()}
    
    def reset_stats(self) -> None:
        """重置所有统计信息"""
        for compressor in self._compressors.values():
            compressor.reset_stats()


class MessageCompressor:
    """消息压缩器 - 高层API"""
    
    # 压缩算法标识字节
    ALGO_BYTES = {
        CompressionAlgorithm.NONE: b'\x00',
        CompressionAlgorithm.GZIP: b'\x01',
        CompressionAlgorithm.LZ4: b'\x02',
        CompressionAlgorithm.ZSTD: b'\x03',
        CompressionAlgorithm.SNAPPY: b'\x04',
    }
    
    BYTES_TO_ALGO = {v: k for k, v in ALGO_BYTES.items()}
    
    def __init__(self, config: Optional[CompressionConfig] = None) -> None:
        self.auto = AutoCompressor(config)
    
    def compress(self, data: bytes, algorithm: Optional[CompressionAlgorithm] = None) -> bytes:
        """压缩数据并添加算法标识"""
        compressed, stats = self.auto.compress(data, algorithm)
        
        # 添加算法标识前缀
        algo_byte = self.ALGO_BYTES.get(stats.algorithm, b'\x00')
        return algo_byte + compressed
    
    def decompress(self, data: bytes) -> bytes:
        """解压缩带算法标识的数据"""
        if len(data) < 1:
            return data
        
        algo_byte = data[:1]
        compressed = data[1:]
        
        algorithm = self.BYTES_TO_ALGO.get(algo_byte, CompressionAlgorithm.NONE)
        return self.auto.decompress(compressed, algorithm)
    
    def compress_message(self, message: bytes, headers: Optional[Dict[str, str]] = None) -> Tuple[bytes, Dict[str, str]]:
        """压缩消息并更新头部"""
        compressed = self.compress(message)
        
        new_headers = headers.copy() if headers else {}
        new_headers['x-compression'] = 'enabled'
        new_headers['x-original-size'] = str(len(message))
        new_headers['x-compressed-size'] = str(len(compressed))
        
        return compressed, new_headers
    
    def decompress_message(self, data: bytes, headers: Optional[Dict[str, str]] = None) -> bytes:
        """根据头部信息解压缩消息"""
        if headers and headers.get('x-compression') == 'enabled':
            return self.decompress(data)
        return data
