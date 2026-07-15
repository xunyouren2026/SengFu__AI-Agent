"""
MemoryAlignment - 内存对齐模块

提供内存对齐分配和管理功能，包括：
- 16/32/64/128/256/4096字节对齐的内存分配
- 对齐内存池管理
- 对齐感知的缓冲区操作
- 跨平台对齐内存分配 (posix_memalign/_aligned_malloc/valloc)
- 对齐验证与诊断工具

模块路径: hardware/cpu/memory_alignment.py
"""

import os
import sys
import ctypes
import struct
import logging
import platform
import mmap
import weakref
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AlignmentSize(Enum):
    """标准对齐大小枚举"""
    ALIGN_16 = 16
    ALIGN_32 = 32
    ALIGN_64 = 64
    ALIGN_128 = 128
    ALIGN_256 = 256
    ALIGN_512 = 512
    ALIGN_1024 = 1024
    ALIGN_2048 = 2048
    ALIGN_4096 = 4096
    ALIGN_PAGE = -1  # 系统页面大小


@dataclass
class AlignedBuffer:
    """
    对齐内存缓冲区

    封装一块对齐分配的内存，提供安全的访问和自动释放。
    """
    address: int = 0
    size: int = 0
    alignment: int = 64
    _raw_ptr: Any = field(default=None, repr=False)
    _owns_memory: bool = True
    _cleanup_callback: Any = field(default=None, repr=False)

    @property
    def ptr(self) -> Any:
        """获取底层指针"""
        return self._raw_ptr

    def as_bytes(self) -> bytes:
        """将缓冲区内容读取为bytes"""
        if self._raw_ptr is None:
            return b""
        buf = (ctypes.c_char * self.size).from_address(self.address)
        return bytes(buf)

    def as_ctypes_array(self, dtype: type = ctypes.c_float) -> Any:
        """
        将缓冲区转换为ctypes数组

        Args:
            dtype: ctypes数据类型 (如ctypes.c_float, ctypes.c_int32)

        Returns:
            ctypes数组对象
        """
        if self._raw_ptr is None:
            raise RuntimeError("Buffer is not allocated")
        elem_size = ctypes.sizeof(dtype)
        count = self.size // elem_size
        return (dtype * count).from_address(self.address)

    def write_bytes(self, data: bytes, offset: int = 0) -> None:
        """
        向缓冲区写入字节数据

        Args:
            data: 要写入的数据
            offset: 写入偏移量
        """
        if self._raw_ptr is None:
            raise RuntimeError("Buffer is not allocated")
        if offset + len(data) > self.size:
            raise ValueError("Write would exceed buffer bounds")
        buf = (ctypes.c_char * len(data)).from_buffer_copy(data)
        ctypes.memmove(self.address + offset, buf, len(data))

    def read_bytes(self, length: int, offset: int = 0) -> bytes:
        """
        从缓冲区读取字节数据

        Args:
            length: 读取长度
            offset: 读取偏移量

        Returns:
            bytes: 读取的数据
        """
        if self._raw_ptr is None:
            raise RuntimeError("Buffer is not allocated")
        if offset + length > self.size:
            raise ValueError("Read would exceed buffer bounds")
        buf = (ctypes.c_char * length).from_address(self.address + offset)
        return bytes(buf)

    def is_aligned(self, alignment: Optional[int] = None) -> bool:
        """
        检查缓冲区地址是否满足对齐要求

        Args:
            alignment: 对齐大小，默认使用缓冲区的对齐设置

        Returns:
            bool: 是否对齐
        """
        align = alignment or self.alignment
        return self.address % align == 0

    def memset(self, value: int = 0) -> None:
        """
        将缓冲区填充为指定值

        Args:
            value: 填充值 (0-255)
        """
        if self._raw_ptr is not None:
            ctypes.memset(self.address, value, self.size)

    def copy_to(self, dest: "AlignedBuffer") -> None:
        """
        将数据复制到另一个对齐缓冲区

        Args:
            dest: 目标缓冲区
        """
        if self._raw_ptr is None or dest._raw_ptr is None:
            raise RuntimeError("Buffer is not allocated")
        copy_size = min(self.size, dest.size)
        ctypes.memmove(dest.address, self.address, copy_size)

    def free(self) -> None:
        """释放缓冲区内存"""
        if self._owns_memory and self._raw_ptr is not None:
            if self._cleanup_callback:
                self._cleanup_callback(self._raw_ptr)
            self._raw_ptr = None
            self.address = 0
            self._cleanup_callback = None

    def __del__(self) -> None:
        self.free()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "address": hex(self.address),
            "size": self.size,
            "alignment": self.alignment,
            "is_aligned": self.is_aligned(),
        }

    def __repr__(self) -> str:
        return (
            f"AlignedBuffer(addr={hex(self.address)}, "
            f"size={self.size}, align={self.alignment})"
        )


@dataclass
class AlignmentStats:
    """对齐统计信息"""
    total_allocations: int = 0
    total_bytes_allocated: int = 0
    total_bytes_freed: int = 0
    active_buffers: int = 0
    alignment_violations: int = 0
    largest_allocation: int = 0

    @property
    def current_usage(self) -> int:
        return self.total_bytes_allocated - self.total_bytes_freed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_allocations": self.total_allocations,
            "total_bytes_allocated": self.total_bytes_allocated,
            "total_bytes_freed": self.total_bytes_freed,
            "current_usage_bytes": self.current_usage,
            "active_buffers": self.active_buffers,
            "alignment_violations": self.alignment_violations,
            "largest_allocation": self.largest_allocation,
        }


class MemoryAlignment:
    """
    内存对齐管理器

    提供跨平台的内存对齐分配、对齐验证和内存池管理功能。
    支持Linux (posix_memalign)、macOS (posix_memalign)、Windows (_aligned_malloc)。
    """

    # 默认对齐大小
    DEFAULT_ALIGNMENT = 64
    # 页面大小缓存
    _page_size: Optional[int] = None

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化内存对齐管理器

        Args:
            config: 配置字典，支持：
                - default_alignment: int, 默认对齐大小
                - enable_tracking: bool, 是否启用分配跟踪
                - pool_enabled: bool, 是否启用内存池
                - pool_block_size: int, 池块大小
        """
        self.config = config or {}
        self._default_alignment = self.config.get(
            "default_alignment", self.DEFAULT_ALIGNMENT
        )
        self._tracking_enabled = self.config.get("enable_tracking", True)
        self._stats = AlignmentStats()
        self._active_buffers: List[AlignedBuffer] = []
        self._pool_enabled = self.config.get("pool_enabled", False)
        self._pool: Dict[int, List[AlignedBuffer]] = {}
        self._initialized = False
        self._lock = __import__("threading").Lock()

    def initialize(self) -> bool:
        """
        初始化内存对齐管理器

        Returns:
            bool: 初始化是否成功
        """
        try:
            page_size = self.get_page_size()
            self._initialized = True
            logger.info(
                "MemoryAlignment initialized: default_align=%d, page_size=%d",
                self._default_alignment, page_size,
            )
            return True
        except Exception as e:
            logger.error("Failed to initialize MemoryAlignment: %s", e)
            return False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def stats(self) -> AlignmentStats:
        return self._stats

    # ========================
    # 对齐内存分配
    # ========================

    def allocate(
        self, size: int, alignment: Optional[int] = None
    ) -> AlignedBuffer:
        """
        分配对齐内存

        Args:
            size: 分配大小（字节）
            alignment: 对齐大小，默认使用管理器默认值

        Returns:
            AlignedBuffer: 对齐缓冲区对象

        Raises:
            ValueError: 参数无效
            MemoryError: 分配失败
        """
        if size <= 0:
            raise ValueError(f"Size must be positive, got {size}")

        align = alignment or self._default_alignment
        if align <= 0:
            align = self.get_page_size()

        # 确保对齐是2的幂
        if align & (align - 1) != 0:
            raise ValueError(f"Alignment must be a power of 2, got {align}")

        # 从内存池分配
        if self._pool_enabled:
            pooled = self._allocate_from_pool(size, align)
            if pooled is not None:
                return pooled

        # 系统分配
        buffer = self._system_allocate(size, align)

        if self._tracking_enabled:
            with self._lock:
                self._stats.total_allocations += 1
                self._stats.total_bytes_allocated += size
                self._stats.active_buffers += 1
                self._stats.largest_allocation = max(
                    self._stats.largest_allocation, size
                )
                self._active_buffers.append(buffer)

        # 验证对齐
        if not buffer.is_aligned(align):
            self._stats.alignment_violations += 1
            logger.warning(
                "Alignment violation: addr=%s, expected_align=%d",
                hex(buffer.address), align,
            )

        return buffer

    def _system_allocate(self, size: int, alignment: int) -> AlignedBuffer:
        """
        通过系统API分配对齐内存

        Args:
            size: 分配大小
            alignment: 对齐大小

        Returns:
            AlignedBuffer: 对齐缓冲区
        """
        system = platform.system()
        if system in ("Linux", "Darwin"):
            return self._allocate_posix(size, alignment)
        elif system == "Windows":
            return self._allocate_windows(size, alignment)
        else:
            return self._allocate_fallback(size, alignment)

    def _allocate_posix(self, size: int, alignment: int) -> AlignedBuffer:
        """通过posix_memalign分配对齐内存"""
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        if platform.system() == "Darwin":
            libc = ctypes.CDLL("libSystem.dylib", use_errno=True)

        ptr = ctypes.c_void_p()
        result = libc.posix_memalign(
            ctypes.byref(ptr), alignment, size
        )

        if result != 0:
            errno = ctypes.get_errno()
            raise MemoryError(
                f"posix_memalign failed: errno={errno}, "
                f"size={size}, alignment={alignment}"
            )

        def cleanup(p):
            libc.free(p)

        return AlignedBuffer(
            address=ptr.value,
            size=size,
            alignment=alignment,
            _raw_ptr=ptr,
            _owns_memory=True,
            _cleanup_callback=cleanup,
        )

    def _allocate_windows(self, size: int, alignment: int) -> AlignedBuffer:
        """通过_aligned_malloc分配对齐内存"""
        try:
            msvcrt = ctypes.CDLL("msvcrt.dll")
        except OSError:
            return self._allocate_fallback(size, alignment)

        ptr = msvcrt._aligned_malloc(size, alignment)
        if not ptr:
            raise MemoryError(
                f"_aligned_malloc failed: size={size}, alignment={alignment}"
            )

        def cleanup(p):
            msvcrt._aligned_free(p)

        return AlignedBuffer(
            address=ptr,
            size=size,
            alignment=alignment,
            _raw_ptr=ctypes.c_void_p(ptr),
            _owns_memory=True,
            _cleanup_callback=cleanup,
        )

    def _allocate_fallback(self, size: int, alignment: int) -> AlignedBuffer:
        """回退: 使用ctypes create_string_buffer + 手动对齐"""
        # 分配额外空间用于对齐
        total_size = size + alignment + ctypes.sizeof(ctypes.c_void_p)
        raw = ctypes.create_string_buffer(total_size)
        raw_addr = ctypes.addressof(raw)

        # 计算对齐地址
        aligned_addr = (raw_addr + alignment) & ~(alignment - 1)

        # 在对齐地址前存储原始指针
        ptr_store = aligned_addr - ctypes.sizeof(ctypes.c_void_p)
        ctypes.c_void_p.from_address(ptr_store).value = raw_addr

        def cleanup(p):
            pass  # ctypes会自动管理

        return AlignedBuffer(
            address=aligned_addr,
            size=size,
            alignment=alignment,
            _raw_ptr=raw,
            _owns_memory=True,
            _cleanup_callback=cleanup,
        )

    def free(self, buffer: AlignedBuffer) -> None:
        """
        释放对齐缓冲区

        Args:
            buffer: 要释放的缓冲区
        """
        if buffer is None:
            return

        if self._tracking_enabled:
            with self._lock:
                self._stats.total_bytes_freed += buffer.size
                self._stats.active_buffers -= 1
                if buffer in self._active_buffers:
                    self._active_buffers.remove(buffer)

        # 如果启用了内存池，回收到池中
        if self._pool_enabled and buffer.address != 0:
            self._return_to_pool(buffer)
        else:
            buffer.free()

    def reallocate(
        self, buffer: AlignedBuffer, new_size: int
    ) -> AlignedBuffer:
        """
        重新分配对齐缓冲区

        Args:
            buffer: 原始缓冲区
            new_size: 新的大小

        Returns:
            AlignedBuffer: 新的对齐缓冲区
        """
        new_buffer = self.allocate(new_size, buffer.alignment)
        if buffer.address != 0 and new_buffer.address != 0:
            copy_size = min(buffer.size, new_size)
            ctypes.memmove(new_buffer.address, buffer.address, copy_size)
        self.free(buffer)
        return new_buffer

    # ========================
    # 内存池管理
    # ========================

    def _allocate_from_pool(
        self, size: int, alignment: int
    ) -> Optional[AlignedBuffer]:
        """从内存池中分配"""
        pool_key = (size, alignment)
        if pool_key in self._pool and self._pool[pool_key]:
            buffer = self._pool[pool_key].pop()
            buffer.memset(0)
            return buffer
        return None

    def _return_to_pool(self, buffer: AlignedBuffer) -> None:
        """将缓冲区回收到内存池"""
        pool_key = (buffer.size, buffer.alignment)
        if pool_key not in self._pool:
            self._pool[pool_key] = []
        self._pool[pool_key].append(buffer)

    def clear_pool(self) -> int:
        """
        清空内存池，释放所有缓存的缓冲区

        Returns:
            int: 释放的缓冲区数量
        """
        count = 0
        for pool_list in self._pool.values():
            for buffer in pool_list:
                buffer.free()
                count += 1
        self._pool.clear()
        return count

    def get_pool_stats(self) -> Dict[str, Any]:
        """
        获取内存池统计信息

        Returns:
            Dict: 内存池状态
        """
        total_cached = sum(len(v) for v in self._pool.values())
        total_cached_bytes = sum(
            buf.size for pool in self._pool.values() for buf in pool
        )
        return {
            "enabled": self._pool_enabled,
            "pool_entries": len(self._pool),
            "total_cached_buffers": total_cached,
            "total_cached_bytes": total_cached_bytes,
        }

    # ========================
    # 对齐工具
    # ========================

    @classmethod
    def get_page_size(cls) -> int:
        """
        获取系统页面大小

        Returns:
            int: 页面大小（字节）
        """
        if cls._page_size is not None:
            return cls._page_size

        system = platform.system()
        if system in ("Linux", "Darwin"):
            try:
                libc = ctypes.CDLL(
                    "libc.so.6" if system == "Linux" else "libSystem.dylib",
                    use_errno=True,
                )
                cls._page_size = libc.sysconf(30)  # _SC_PAGESIZE = 30
            except (OSError, AttributeError):
                cls._page_size = 4096
        elif system == "Windows":
            try:
                ctypes.windll.kernel32.GetSystemInfo.restype = None
                class SYSTEM_INFO(ctypes.Structure):
                    _fields_ = [("dwPageSize", ctypes.c_ulong)] + [
                        (f"f{i}", ctypes.c_ulong) for i in range(1, 10)
                    ]
                si = SYSTEM_INFO()
                ctypes.windll.kernel32.GetSystemInfo(ctypes.byref(si))
                cls._page_size = si.dwPageSize
            except (AttributeError, OSError):
                cls._page_size = 4096
        else:
            cls._page_size = 4096

        return cls._page_size

    @staticmethod
    def align_up(value: int, alignment: int) -> int:
        """
        将值向上对齐

        Args:
            value: 原始值
            alignment: 对齐大小

        Returns:
            int: 对齐后的值
        """
        if alignment <= 0:
            return value
        return (value + alignment - 1) & ~(alignment - 1)

    @staticmethod
    def align_down(value: int, alignment: int) -> int:
        """
        将值向下对齐

        Args:
            value: 原始值
            alignment: 对齐大小

        Returns:
            int: 对齐后的值
        """
        if alignment <= 0:
            return value
        return value & ~(alignment - 1)

    @staticmethod
    def is_aligned(address: int, alignment: int) -> bool:
        """
        检查地址是否对齐

        Args:
            address: 内存地址
            alignment: 对齐大小

        Returns:
            bool: 是否对齐
        """
        return address % alignment == 0

    @staticmethod
    def compute_padding(offset: int, alignment: int) -> int:
        """
        计算达到对齐所需的填充字节数

        Args:
            offset: 当前偏移量
            alignment: 对齐大小

        Returns:
            int: 需要填充的字节数
        """
        if alignment <= 0:
            return 0
        remainder = offset % alignment
        return (alignment - remainder) % alignment

    def recommend_alignment(self, data_size: int, access_pattern: str = "sequential") -> int:
        """
        根据数据大小和访问模式推荐对齐大小

        Args:
            data_size: 数据大小
            access_pattern: 访问模式 ("sequential", "random", "vectorized")

        Returns:
            int: 推荐的对齐大小
        """
        page_size = self.get_page_size()

        if access_pattern == "vectorized":
            # SIMD操作推荐64字节对齐 (AVX-512需要64字节)
            if data_size >= 512:
                return 64
            return 32
        elif access_pattern == "random":
            # 随机访问推荐页面大小对齐
            if data_size >= page_size:
                return page_size
            return 64
        else:
            # 顺序访问
            if data_size >= page_size:
                return page_size
            elif data_size >= 256:
                return 64
            elif data_size >= 64:
                return 32
            return 16

    def create_struct_layout(
        self, fields: List[Tuple[str, type, int]]
    ) -> Dict[str, Any]:
        """
        创建对齐优化的结构体布局

        Args:
            fields: 字段列表，每个元素为 (名称, 类型, 数组长度)
                    类型支持: int, float, double, bool

        Returns:
            Dict: 结构体布局信息，包含偏移量、大小和对齐
        """
        type_sizes = {
            "int": ctypes.sizeof(ctypes.c_int32),
            "int32": ctypes.sizeof(ctypes.c_int32),
            "int64": ctypes.sizeof(ctypes.c_int64),
            "float": ctypes.sizeof(ctypes.c_float),
            "float32": ctypes.sizeof(ctypes.c_float),
            "double": ctypes.sizeof(ctypes.c_double),
            "float64": ctypes.sizeof(ctypes.c_double),
            "bool": ctypes.sizeof(ctypes.c_bool),
            "uint8": ctypes.sizeof(ctypes.c_uint8),
            "uint16": ctypes.sizeof(ctypes.c_uint16),
            "uint32": ctypes.sizeof(ctypes.c_uint32),
            "uint64": ctypes.sizeof(ctypes.c_uint64),
        }

        type_alignments = {
            "int": 4, "int32": 4, "int64": 8,
            "float": 4, "float32": 4,
            "double": 8, "float64": 8,
            "bool": 1,
            "uint8": 1, "uint16": 2, "uint32": 4, "uint64": 8,
        }

        layout = {
            "fields": [],
            "total_size": 0,
            "struct_alignment": 1,
        }

        current_offset = 0
        for name, dtype, count in fields:
            dtype_str = dtype.__name__ if hasattr(dtype, "__name__") else str(dtype)
            elem_size = type_sizes.get(dtype_str, ctypes.sizeof(ctypes.c_int32))
            elem_align = type_alignments.get(dtype_str, 4)
            field_size = elem_size * count

            # 对齐偏移量
            padding = self.compute_padding(current_offset, elem_align)
            aligned_offset = current_offset + padding

            layout["fields"].append({
                "name": name,
                "type": dtype_str,
                "count": count,
                "element_size": elem_size,
                "field_size": field_size,
                "alignment": elem_align,
                "offset": aligned_offset,
                "padding_before": padding,
            })

            current_offset = aligned_offset + field_size
            layout["struct_alignment"] = max(
                layout["struct_alignment"], elem_align
            )

        # 结构体末尾填充
        final_padding = self.compute_padding(current_offset, layout["struct_alignment"])
        layout["total_size"] = current_offset + final_padding
        layout["padding_end"] = final_padding

        return layout

    # ========================
    # 对齐验证
    # ========================

    def validate_alignment(
        self, address: int, alignment: int
    ) -> Dict[str, Any]:
        """
        验证地址的对齐状态

        Args:
            address: 内存地址
            alignment: 期望的对齐大小

        Returns:
            Dict: 验证结果
        """
        remainder = address % alignment
        return {
            "address": hex(address),
            "alignment": alignment,
            "is_aligned": remainder == 0,
            "remainder": remainder,
            "bytes_to_align": (alignment - remainder) % alignment if remainder else 0,
        }

    def diagnose_alignment_issues(
        self, addresses: List[int], alignment: int
    ) -> Dict[str, Any]:
        """
        批量诊断对齐问题

        Args:
            addresses: 地址列表
            alignment: 期望的对齐大小

        Returns:
            Dict: 诊断报告
        """
        violations = []
        aligned_count = 0

        for addr in addresses:
            if addr % alignment != 0:
                violations.append({
                    "address": hex(addr),
                    "remainder": addr % alignment,
                })
            else:
                aligned_count += 1

        return {
            "total_addresses": len(addresses),
            "aligned_count": aligned_count,
            "violation_count": len(violations),
            "alignment_rate": aligned_count / len(addresses) if addresses else 0,
            "violations": violations[:20],  # 最多返回20个
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        获取内存对齐管理器的完整摘要

        Returns:
            Dict: 包含统计和配置的摘要
        """
        return {
            "initialized": self._initialized,
            "platform": platform.system(),
            "page_size": self.get_page_size(),
            "default_alignment": self._default_alignment,
            "tracking_enabled": self._tracking_enabled,
            "pool_enabled": self._pool_enabled,
            "stats": self._stats.to_dict(),
            "pool_stats": self.get_pool_stats(),
        }

    def __repr__(self) -> str:
        return (
            f"MemoryAlignment(default_align={self._default_alignment}, "
            f"tracking={self._tracking_enabled})"
        )
