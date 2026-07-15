"""
CacheOptimization - CPU缓存优化模块

提供L1/L2/L3缓存感知的数据布局优化，包括：
- 缓存拓扑探测与信息采集
- 缓存友好的数据结构布局
- 缓存行大小感知的内存访问优化
- 缓存预热与预取策略
- 缓存命中率监控与分析

模块路径: hardware/cpu/cache_optimization.py
"""

import os
import sys
import struct
import ctypes
import logging
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import OrderedDict

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """CPU缓存层级枚举"""
    L1I = auto()   # L1 指令缓存
    L1D = auto()   # L1 数据缓存
    L2 = auto()    # L2 统一缓存
    L3 = auto()    # L3 共享缓存
    L4 = auto()    # L4 eDRAM (部分Intel处理器)


@dataclass
class CacheInfo:
    """单个缓存级别的详细信息"""
    level: CacheLevel
    size_bytes: int = 0
    line_size_bytes: int = 64
    associativity: int = 0
    sets: int = 0
    inclusive: bool = True
    shared_by_cores: int = 1
    write_policy: str = "write-back"

    @property
    def size_kb(self) -> float:
        return self.size_bytes / 1024.0

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024.0 * 1024.0)

    @property
    def total_lines(self) -> int:
        if self.line_size_bytes > 0:
            return self.size_bytes // self.line_size_bytes
        return 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.name,
            "size_bytes": self.size_bytes,
            "size_kb": round(self.size_kb, 2),
            "size_mb": round(self.size_mb, 4),
            "line_size_bytes": self.line_size_bytes,
            "associativity": self.associativity,
            "sets": self.sets,
            "inclusive": self.inclusive,
            "shared_by_cores": self.shared_by_cores,
            "write_policy": self.write_policy,
            "total_lines": self.total_lines,
        }


@dataclass
class CacheTopology:
    """CPU缓存拓扑结构"""
    caches: List[CacheInfo] = field(default_factory=list)
    num_physical_cores: int = 0
    num_logical_cores: int = 0
    numa_nodes: int = 1

    def get_cache(self, level: CacheLevel) -> Optional[CacheInfo]:
        """获取指定层级的缓存信息"""
        for cache in self.caches:
            if cache.level == level:
                return cache
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "caches": [c.to_dict() for c in self.caches],
            "num_physical_cores": self.num_physical_cores,
            "num_logical_cores": self.num_logical_cores,
            "numa_nodes": self.numa_nodes,
        }


@dataclass
class CacheAccessPattern:
    """缓存访问模式分析结果"""
    stride_bytes: int = 0
    spatial_locality_score: float = 0.0
    temporal_locality_score: float = 0.0
    estimated_hit_rate: float = 0.0
    recommended_layout: str = "row-major"
    recommended_block_size: int = 0


class CacheOptimization:
    """
    CPU缓存优化器

    提供缓存感知的数据布局优化、缓存预热、命中率监控等功能。
    支持Linux/macOS/Windows跨平台操作。
    """

    # 常见CPU的缓存行大小默认值
    DEFAULT_LINE_SIZE = 64
    # L1缓存典型大小范围 (bytes)
    L1_SIZE_RANGE = (16384, 65536)       # 16KB - 64KB
    # L2缓存典型大小范围 (bytes)
    L2_SIZE_RANGE = (131072, 2097152)    # 128KB - 2MB
    # L3缓存典型大小范围 (bytes)
    L3_SIZE_RANGE = (2097152, 134217728) # 2MB - 128MB

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化缓存优化器

        Args:
            config: 配置字典，支持以下选项：
                - auto_detect: bool, 是否自动检测缓存拓扑 (默认True)
                - enable_prefetch: bool, 是否启用预取建议 (默认True)
                - l1_size: int, 手动指定L1缓存大小 (0为自动检测)
                - l2_size: int, 手动指定L2缓存大小 (0为自动检测)
                - l3_size: int, 手动指定L3缓存大小 (0为自动检测)
                - line_size: int, 手动指定缓存行大小 (0为自动检测)
        """
        self.config = config or {}
        self._topology: Optional[CacheTopology] = None
        self._access_stats: Dict[str, List[float]] = {}
        self._initialized = False

        if self.config.get("auto_detect", True):
            self._topology = self.detect_cache_topology()

    def initialize(self) -> bool:
        """
        初始化缓存优化器，探测系统缓存信息

        Returns:
            bool: 初始化是否成功
        """
        try:
            if self._topology is None:
                self._topology = self.detect_cache_topology()
            self._access_stats = {
                "hit_rates": [],
                "miss_rates": [],
                "access_times_ns": [],
            }
            self._initialized = True
            logger.info(
                "CacheOptimization initialized: %d cache levels detected",
                len(self._topology.caches) if self._topology else 0,
            )
            return True
        except Exception as e:
            logger.error("Failed to initialize CacheOptimization: %s", e)
            return False

    @property
    def topology(self) -> Optional[CacheTopology]:
        """获取缓存拓扑结构"""
        return self._topology

    @property
    def is_initialized(self) -> bool:
        """是否已初始化"""
        return self._initialized

    # ========================
    # 缓存拓扑探测
    # ========================

    def detect_cache_topology(self) -> CacheTopology:
        """
        探测当前系统的CPU缓存拓扑结构

        通过多种途径获取缓存信息：
        1. Linux: /sys/devices/system/cpu/cpu0/cache/
        2. Linux: lscpu命令
        3. macOS: sysctl hw
        4. Windows: wmic / GetLogicalProcessorInformation
        5. 回退: 基于CPU型号的启发式推断

        Returns:
            CacheTopology: 缓存拓扑结构
        """
        system = platform.system()
        caches: List[CacheInfo] = []
        num_physical = 0
        num_logical = 0

        try:
            num_logical = os.cpu_count() or 1
        except (AttributeError, OSError):
            num_logical = 1

        if system == "Linux":
            caches, num_physical = self._detect_linux()
        elif system == "Darwin":
            caches, num_physical = self._detect_macos(num_logical)
        elif system == "Windows":
            caches, num_physical = self._detect_windows()
        else:
            caches, num_physical = self._detect_fallback(num_logical)

        # 应用手动覆盖配置
        caches = self._apply_config_overrides(caches)

        if num_physical == 0:
            num_physical = num_logical

        return CacheTopology(
            caches=caches,
            num_physical_cores=num_physical,
            num_logical_cores=num_logical,
        )

    def _detect_linux(self) -> Tuple[List[CacheInfo], int]:
        """通过Linux sysfs探测缓存信息"""
        caches: List[CacheInfo] = []
        num_physical = 0

        cache_base = Path("/sys/devices/system/cpu/cpu0/cache")
        if not cache_base.exists():
            return self._detect_linux_lscpu()

        index_dirs = sorted(cache_base.glob("index*"))
        for index_dir in index_dirs:
            try:
                level_str = (index_dir / "level").read_text().strip()
                level_int = int(level_str)
                type_str = (index_dir / "type").read_text().strip()
                size_str = (index_dir / "size").read_text().strip()
                coherency = (index_dir / "coherency_line_size").read_text().strip()
                shared_map = (index_dir / "shared_cpu_map").read_text().strip()

                size_bytes = self._parse_size_string(size_str)
                line_size = int(coherency)
                shared_cores = bin(int(shared_map, 16)).count("1")

                cache_level = self._map_cache_level(level_int, type_str)
                if cache_level is None:
                    continue

                associativity = 0
                ways_file = index_dir / "ways_of_associativity"
                if ways_file.exists():
                    associativity = int(ways_file.read_text().strip())

                num_sets = 0
                sets_file = index_dir / "number_of_sets"
                if sets_file.exists():
                    num_sets = int(sets_file.read_text().strip())

                caches.append(CacheInfo(
                    level=cache_level,
                    size_bytes=size_bytes,
                    line_size_bytes=line_size,
                    associativity=associativity,
                    sets=num_sets,
                    shared_by_cores=shared_cores,
                ))
            except (OSError, ValueError, KeyError) as e:
                logger.debug("Error reading cache index %s: %s", index_dir, e)
                continue

        # 获取物理核心数
        try:
            cpu_info = Path("/proc/cpuinfo").read_text()
            for line in cpu_info.splitlines():
                if line.startswith("cpu cores"):
                    num_physical = int(line.split(":")[1].strip())
                    break
        except (OSError, ValueError, IndexError):
            pass

        return caches, num_physical

    def _detect_linux_lscpu(self) -> Tuple[List[CacheInfo], int]:
        """通过lscpu命令探测缓存信息（回退方案）"""
        caches: List[CacheInfo] = []
        num_physical = 0

        try:
            result = subprocess.run(
                ["lscpu", "-b", "-p=CPU,CORE"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                cores_seen = set()
                for line in result.stdout.splitlines():
                    if line.startswith("#"):
                        continue
                    parts = line.split(",")
                    if len(parts) >= 2:
                        cores_seen.add(parts[1].strip())
                num_physical = len(cores_seen)
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass

        try:
            result = subprocess.run(
                ["lscpu"], capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if line.startswith("L1d cache:"):
                        size = self._parse_size_string(line.split(":")[1].strip())
                        caches.append(CacheInfo(
                            level=CacheLevel.L1D, size_bytes=size,
                            line_size_bytes=self.DEFAULT_LINE_SIZE,
                        ))
                    elif line.startswith("L1i cache:"):
                        size = self._parse_size_string(line.split(":")[1].strip())
                        caches.append(CacheInfo(
                            level=CacheLevel.L1I, size_bytes=size,
                            line_size_bytes=self.DEFAULT_LINE_SIZE,
                        ))
                    elif line.startswith("L2 cache:"):
                        size = self._parse_size_string(line.split(":")[1].strip())
                        caches.append(CacheInfo(
                            level=CacheLevel.L2, size_bytes=size,
                            line_size_bytes=self.DEFAULT_LINE_SIZE,
                        ))
                    elif line.startswith("L3 cache:"):
                        size = self._parse_size_string(line.split(":")[1].strip())
                        caches.append(CacheInfo(
                            level=CacheLevel.L3, size_bytes=size,
                            line_size_bytes=self.DEFAULT_LINE_SIZE,
                        ))
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass

        return caches, num_physical

    def _detect_macos(self, num_logical: int) -> Tuple[List[CacheInfo], int]:
        """通过macOS sysctl探测缓存信息"""
        caches: List[CacheInfo] = []
        num_physical = 0

        sysctl_map = {
            "hw.l1dcachesize": CacheLevel.L1D,
            "hw.l1icachesize": CacheLevel.L1I,
            "hw.l2cachesize": CacheLevel.L2,
            "hw.l3cachesize": CacheLevel.L3,
        }

        for sysctl_key, cache_level in sysctl_map.items():
            try:
                result = subprocess.run(
                    ["sysctl", "-n", sysctl_key],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    size = int(result.stdout.strip())
                    if size > 0:
                        caches.append(CacheInfo(
                            level=cache_level,
                            size_bytes=size,
                            line_size_bytes=self.DEFAULT_LINE_SIZE,
                        ))
            except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
                continue

        # 获取物理核心数
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                num_physical = int(result.stdout.strip())
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass

        # 获取缓存行大小
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.cachelinesize"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                line_size = int(result.stdout.strip())
                for cache in caches:
                    cache.line_size_bytes = line_size
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass

        return caches, num_physical

    def _detect_windows(self) -> Tuple[List[CacheInfo], int]:
        """通过Windows API探测缓存信息"""
        caches: List[CacheInfo] = []
        num_physical = 0

        try:
            import ctypes.wintypes

            class CACHE_RELATIONSHIP(ctypes.Structure):
                _fields_ = [
                    ("Level", ctypes.c_ubyte),
                    ("Associativity", ctypes.c_ubyte),
                    ("LineSize", ctypes.c_ushort),
                    ("CacheSize", ctypes.c_ulong),
                    ("Type", ctypes.c_ulong),
                    ("Reserved", ctypes.c_ulong * 20),
                ]

            class PROCESSOR_RELATIONSHIP(ctypes.Structure):
                _fields_ = [
                    ("Flags", ctypes.c_ubyte),
                    ("EfficiencyClass", ctypes.c_ubyte),
                    ("Reserved", ctypes.c_byte * 20),
                    ("GroupCount", ctypes.c_ushort),
                    ("GroupMask", ctypes.c_ulonglong * 1),
                ]

            class SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX(ctypes.Structure):
                _fields_ = [
                    ("RelationshipType", ctypes.c_ulong),
                    ("Size", ctypes.c_ulong),
                    ("Processor", PROCESSOR_RELATIONSHIP),
                ]

            kernel32 = ctypes.windll.kernel32
            buffer_size = ctypes.c_ulong(0)
            kernel32.GetLogicalProcessorInformationEx(
                3, None, ctypes.byref(buffer_size)
            )
            # 简化: 使用wmic回退
        except (AttributeError, OSError):
            pass

        # 回退到wmic
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "L2CacheSize,L3CacheSize,NumberOfCores,NumberOfLogicalProcessors", "/format:csv"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if not line or line.startswith("Node,") or line.startswith("L2"):
                        continue
                    parts = [p.strip() for p in line.split(",") if p.strip()]
                    if len(parts) >= 4:
                        try:
                            l2 = int(parts[0]) * 1024 if parts[0] else 0
                            l3 = int(parts[1]) * 1024 if parts[1] else 0
                            num_physical = int(parts[2])
                            if l2 > 0:
                                caches.append(CacheInfo(
                                    level=CacheLevel.L2, size_bytes=l2,
                                    line_size_bytes=self.DEFAULT_LINE_SIZE,
                                ))
                            if l3 > 0:
                                caches.append(CacheInfo(
                                    level=CacheLevel.L3, size_bytes=l3,
                                    line_size_bytes=self.DEFAULT_LINE_SIZE,
                                ))
                        except (ValueError, IndexError):
                            continue
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return caches, num_physical

    def _detect_fallback(self, num_logical: int) -> Tuple[List[CacheInfo], int]:
        """基于启发式的缓存信息推断（最终回退）"""
        caches: List[CacheInfo] = []
        processor = platform.processor()

        # 基于CPU型号的启发式默认值
        l1_size = 32768    # 32KB
        l2_size = 262144   # 256KB
        l3_size = 8388608  # 8MB

        proc_lower = processor.lower()
        if "intel" in proc_lower:
            if "xeon" in proc_lower:
                l3_size = 16777216  # 16MB
                l2_size = 1048576   # 1MB
            elif "i9" in proc_lower or "i7" in proc_lower:
                l3_size = 12582912  # 12MB
        elif "amd" in proc_lower:
            if "epyc" in proc_lower:
                l3_size = 33554432  # 32MB
                l2_size = 524288    # 512KB
            elif "ryzen" in proc_lower:
                l3_size = 16777216  # 16MB

        caches.append(CacheInfo(
            level=CacheLevel.L1D, size_bytes=l1_size,
            line_size_bytes=self.DEFAULT_LINE_SIZE,
        ))
        caches.append(CacheInfo(
            level=CacheLevel.L1I, size_bytes=l1_size,
            line_size_bytes=self.DEFAULT_LINE_SIZE,
        ))
        caches.append(CacheInfo(
            level=CacheLevel.L2, size_bytes=l2_size,
            line_size_bytes=self.DEFAULT_LINE_SIZE,
        ))
        caches.append(CacheInfo(
            level=CacheLevel.L3, size_bytes=l3_size,
            line_size_bytes=self.DEFAULT_LINE_SIZE,
            shared_by_cores=num_logical,
        ))

        return caches, num_logical

    def _apply_config_overrides(self, caches: List[CacheInfo]) -> List[CacheInfo]:
        """应用配置文件中的手动覆盖"""
        existing_levels = {c.level for c in caches}

        overrides = {
            CacheLevel.L1D: self.config.get("l1_size", 0),
            CacheLevel.L2: self.config.get("l2_size", 0),
            CacheLevel.L3: self.config.get("l3_size", 0),
        }
        manual_line = self.config.get("line_size", 0)

        for cache in caches:
            if manual_line > 0:
                cache.line_size_bytes = manual_line
            if cache.level in overrides and overrides[cache.level] > 0:
                cache.size_bytes = overrides[cache.level]

        return caches

    # ========================
    # 数据布局优化
    # ========================

    def compute_optimal_block_size(self, element_size: int = 8) -> int:
        """
        计算缓存友好的最优块大小

        根据L1/L2缓存大小和缓存行大小，计算适合矩阵运算等
        分块算法的最优块大小。

        Args:
            element_size: 单个数据元素的字节大小 (如float64=8, float32=4)

        Returns:
            int: 推荐的块大小（元素数量）
        """
        if not self._topology:
            return 64

        l1 = self._topology.get_cache(CacheLevel.L1D)
        l2 = self._topology.get_cache(CacheLevel.L2)

        # 目标: 块数据适合L1缓存的1/4，留出空间给其他数据
        target_size = 0
        if l1 and l1.size_bytes > 0:
            target_size = l1.size_bytes // 4
        elif l2 and l2.size_bytes > 0:
            target_size = l2.size_bytes // 16
        else:
            return 64

        block_size = target_size // element_size
        # 确保块大小是缓存行元素数的整数倍
        if l1 and l1.line_size_bytes > 0:
            line_elements = l1.line_size_bytes // element_size
            if line_elements > 0:
                block_size = (block_size // line_elements) * line_elements

        # 限制在合理范围内
        block_size = max(16, min(block_size, 4096))
        # 对齐到2的幂次
        power = 1
        while power * 2 <= block_size:
            power *= 2
        return power

    def compute_optimal_stride(
        self, data_width: int, element_size: int = 8
    ) -> int:
        """
        计算避免缓存行冲突的最优步长

        当多行数据共享同一组时，步长需要避免映射到同一缓存行。

        Args:
            data_width: 数据宽度（元素数）
            element_size: 单个元素字节大小

        Returns:
            int: 推荐的步长（字节数）
        """
        if not self._topology:
            return data_width * element_size

        l1 = self._topology.get_cache(CacheLevel.L1D)
        if not l1 or l1.associativity <= 1:
            return data_width * element_size

        raw_stride = data_width * element_size
        line_size = l1.line_size_bytes or self.DEFAULT_LINE_SIZE

        # 确保步长不是缓存行大小的整数倍，以避免缓存行冲突
        if raw_stride % line_size == 0:
            raw_stride += line_size

        return raw_stride

    def analyze_access_pattern(
        self, data_shape: Tuple[int, ...], element_size: int = 8,
        access_order: str = "row-major",
    ) -> CacheAccessPattern:
        """
        分析给定数据形状和访问模式的缓存行为

        Args:
            data_shape: 数据维度形状
            element_size: 单个元素字节大小
            access_order: 访问顺序 ("row-major" 或 "column-major")

        Returns:
            CacheAccessPattern: 访问模式分析结果
        """
        result = CacheAccessPattern()

        if not self._topology:
            return result

        l1 = self._topology.get_cache(CacheLevel.L1D)
        l2 = self._topology.get_cache(CacheLevel.L2)
        l3 = self._topology.get_cache(CacheLevel.L3)

        total_elements = 1
        for dim in data_shape:
            total_elements *= dim
        total_bytes = total_elements * element_size

        line_size = (l1.line_size_bytes if l1 else self.DEFAULT_LINE_SIZE)
        elements_per_line = line_size // element_size

        # 分析空间局部性
        if len(data_shape) >= 2:
            if access_order == "row-major":
                result.stride_bytes = data_shape[-1] * element_size
            else:
                result.stride_bytes = data_shape[0] * element_size

            if result.stride_bytes <= line_size:
                result.spatial_locality_score = 1.0
            elif result.stride_bytes <= (line_size * 4):
                result.spatial_locality_score = 0.7
            else:
                result.spatial_locality_score = 0.3

        # 分析时间局部性（基于工作集是否适合缓存）
        working_set = min(total_bytes, data_shape[0] * element_size if data_shape else total_bytes)
        if l1 and working_set <= l1.size_bytes * 0.8:
            result.temporal_locality_score = 1.0
            result.estimated_hit_rate = 0.95
        elif l2 and working_set <= l2.size_bytes * 0.8:
            result.temporal_locality_score = 0.8
            result.estimated_hit_rate = 0.85
        elif l3 and working_set <= l3.size_bytes * 0.8:
            result.temporal_locality_score = 0.6
            result.estimated_hit_rate = 0.7
        else:
            result.temporal_locality_score = 0.2
            result.estimated_hit_rate = 0.3

        # 推荐布局
        if len(data_shape) >= 2:
            if access_order == "row-major" and data_shape[-1] >= elements_per_line:
                result.recommended_layout = "row-major"
            elif access_order == "column-major" and data_shape[0] >= elements_per_line:
                result.recommended_layout = "column-major"
            else:
                result.recommended_layout = access_order

        # 推荐块大小
        result.recommended_block_size = self.compute_optimal_block_size(element_size)

        return result

    def suggest_data_layout(
        self, rows: int, cols: int, element_size: int = 8,
    ) -> Dict[str, Any]:
        """
        为二维数据推荐最优布局策略

        Args:
            rows: 行数
            cols: 列数
            element_size: 元素字节大小

        Returns:
            Dict: 布局建议，包含块大小、步长、分块策略等
        """
        block_size = self.compute_optimal_block_size(element_size)
        stride = self.compute_optimal_stride(cols, element_size)
        row_analysis = self.analyze_access_pattern(
            (rows, cols), element_size, "row-major"
        )
        col_analysis = self.analyze_access_pattern(
            (rows, cols), element_size, "column-major"
        )

        best_layout = "row-major"
        if col_analysis.estimated_hit_rate > row_analysis.estimated_hit_rate:
            best_layout = "column-major"

        tile_rows = min(rows, block_size)
        tile_cols = min(cols, block_size)

        return {
            "recommended_layout": best_layout,
            "row_major_hit_rate": row_analysis.estimated_hit_rate,
            "column_major_hit_rate": col_analysis.estimated_hit_rate,
            "block_size": block_size,
            "tile_shape": (tile_rows, tile_cols),
            "stride_bytes": stride,
            "elements_per_cache_line": stride // element_size if element_size > 0 else 0,
            "total_data_bytes": rows * cols * element_size,
            "fits_in_l1": (rows * cols * element_size) <= (
                self._topology.get_cache(CacheLevel.L1D).size_bytes
                if self._topology and self._topology.get_cache(CacheLevel.L1D) else 0
            ),
        }

    # ========================
    # 缓存预热与预取
    # ========================

    def generate_prefetch_hints(
        self, data_size: int, access_pattern: str = "sequential",
        element_size: int = 8,
    ) -> List[int]:
        """
        生成缓存预取偏移量建议

        Args:
            data_size: 数据总大小（字节）
            access_pattern: 访问模式 ("sequential", "strided", "random")
            element_size: 元素字节大小

        Returns:
            List[int]: 预取偏移量列表（字节）
        """
        if not self._topology:
            return []

        l2 = self._topology.get_cache(CacheLevel.L2)
        l3 = self._topology.get_cache(CacheLevel.L3)

        hints = []
        line_size = self.DEFAULT_LINE_SIZE
        l1 = self._topology.get_cache(CacheLevel.L1D)
        if l1:
            line_size = l1.line_size_bytes

        if access_pattern == "sequential":
            # 顺序访问: 预取后续缓存行
            prefetch_distance = 8
            for i in range(1, prefetch_distance + 1):
                offset = i * line_size
                if offset < data_size:
                    hints.append(offset)

        elif access_pattern == "strided":
            # 步长访问: 预取步长位置的数据
            stride = max(element_size * 16, line_size)
            for i in range(1, 5):
                offset = i * stride
                if offset < data_size:
                    hints.append(offset)

        elif access_pattern == "random":
            # 随机访问: 基于L2/L3容量的批量预取
            prefetch_window = 0
            if l2:
                prefetch_window = l2.size_bytes // 2
            elif l3:
                prefetch_window = l3.size_bytes // 4
            else:
                prefetch_window = 262144

            num_hints = min(prefetch_window // line_size, 32)
            for i in range(1, num_hints + 1):
                offset = i * line_size
                if offset < data_size:
                    hints.append(offset)

        return hints

    def compute_prefetch_distance(self, bandwidth_mbps: float = 10000) -> int:
        """
        计算最优预取距离（缓存行数）

        基于内存带宽和L2缓存延迟，计算预取提前量。

        Args:
            bandwidth_mbps: 内存带宽 (MB/s)

        Returns:
            int: 预取距离（缓存行数）
        """
        if not self._topology:
            return 8

        l2 = self._topology.get_cache(CacheLevel.L2)
        l3 = self._topology.get_cache(CacheLevel.L3)

        # L2延迟约4-10ns, L3约10-40ns, 主存约100ns
        l2_latency_ns = 7
        l3_latency_ns = 25
        dram_latency_ns = 100

        line_size = self.DEFAULT_LINE_SIZE
        l1 = self._topology.get_cache(CacheLevel.L1D)
        if l1:
            line_size = l1.line_size_bytes

        bytes_per_ns = bandwidth_mbps / 1000.0  # MB/s -> MB/ns
        if bytes_per_ns <= 0:
            return 8

        # 预取距离 = 延迟 * 带宽 / 缓存行大小
        distance = int(dram_latency_ns * bytes_per_ns * 1024 * 1024 / line_size)
        return max(4, min(distance, 64))

    # ========================
    # 缓存监控
    # ========================

    def get_cache_utilization_estimate(self) -> Dict[str, float]:
        """
        估算当前缓存利用率

        通过读取系统性能计数器（Linux perf/PMC）或使用启发式方法估算。

        Returns:
            Dict: 各级缓存利用率估算
        """
        utilization = {
            "l1_utilization": 0.0,
            "l2_utilization": 0.0,
            "l3_utilization": 0.0,
        }

        if not self._topology:
            return utilization

        system = platform.system()
        if system == "Linux":
            utilization = self._read_linux_perf_counters()
        elif system == "Darwin":
            utilization = self._read_macos_stats()
        elif system == "Windows":
            utilization = self._read_windows_perf_counters()

        return utilization

    def _read_linux_perf_counters(self) -> Dict[str, float]:
        """读取Linux性能计数器"""
        utilization = {
            "l1_utilization": 0.0,
            "l2_utilization": 0.0,
            "l3_utilization": 0.0,
        }

        # 尝试通过perf stat获取缓存事件
        perf_events = {
            "l1-dcache-loads": "l1_loads",
            "l1-dcache-load-misses": "l1_misses",
            "l2-cache-references": "l2_refs",
            "l2-cache-misses": "l2_misses",
            "cache-references": "l3_refs",
            "cache-misses": "l3_misses",
        }

        try:
            events = ",".join(perf_events.keys())
            result = subprocess.run(
                ["perf", "stat", "-e", events, "sleep", "0.1"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                for line in result.stderr.splitlines():
                    for event_name, key in perf_events.items():
                        if event_name in line:
                            try:
                                value = int(line.strip().split()[0].replace(",", ""))
                                if key == "l1_loads" and value > 0:
                                    pass  # 需要配合misses计算
                                elif key == "l3_refs":
                                    pass
                            except (ValueError, IndexError):
                                pass
        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
            pass

        # 回退: 通过/proc/stat估算CPU利用率作为缓存利用率的代理
        try:
            stat_lines = Path("/proc/stat").read_text().splitlines()
            for line in stat_lines:
                if line.startswith("cpu "):
                    parts = line.split()
                    if len(parts) >= 5:
                        user = int(parts[1])
                        nice = int(parts[2])
                        system = int(parts[3])
                        idle = int(parts[4])
                        total = user + nice + system + idle
                        if total > 0:
                            cpu_busy = (user + nice + system) / total
                            utilization["l1_utilization"] = cpu_busy * 0.8
                            utilization["l2_utilization"] = cpu_busy * 0.6
                            utilization["l3_utilization"] = cpu_busy * 0.4
                    break
        except (OSError, ValueError, IndexError):
            pass

        return utilization

    def _read_macos_stats(self) -> Dict[str, float]:
        """读取macOS系统统计"""
        utilization = {
            "l1_utilization": 0.0,
            "l2_utilization": 0.0,
            "l3_utilization": 0.0,
        }

        try:
            result = subprocess.run(
                ["sysctl", "-n", "vm.loadavg"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split()
                if parts:
                    load = float(parts[0])
                    cpu_count = os.cpu_count() or 1
                    ratio = min(load / cpu_count, 1.0)
                    utilization["l1_utilization"] = ratio * 0.8
                    utilization["l2_utilization"] = ratio * 0.6
                    utilization["l3_utilization"] = ratio * 0.4
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass

        return utilization

    def _read_windows_perf_counters(self) -> Dict[str, float]:
        """读取Windows性能计数器"""
        utilization = {
            "l1_utilization": 0.0,
            "l2_utilization": 0.0,
            "l3_utilization": 0.0,
        }

        try:
            import ctypes
            import ctypes.wintypes

            class SYSTEM_INFO(ctypes.Structure):
                _fields_ = [
                    ("wProcessorArchitecture", ctypes.wintypes.WORD),
                    ("wReserved", ctypes.wintypes.WORD),
                    ("dwPageSize", ctypes.wintypes.DWORD),
                    ("lpMinimumApplicationAddress", ctypes.c_void_p),
                    ("lpMaximumApplicationAddress", ctypes.c_void_p),
                    ("dwActiveProcessorMask", ctypes.wintypes.DWORD),
                    ("dwNumberOfProcessors", ctypes.wintypes.DWORD),
                    ("dwProcessorType", ctypes.wintypes.DWORD),
                    ("dwAllocationGranularity", ctypes.wintypes.DWORD),
                    ("wProcessorLevel", ctypes.wintypes.WORD),
                    ("wProcessorRevision", ctypes.wintypes.WORD),
                ]

            sys_info = SYSTEM_INFO()
            ctypes.windll.kernel32.GetSystemInfo(ctypes.byref(sys_info))
            num_cpus = sys_info.dwNumberOfProcessors

            # 使用typeperf获取CPU利用率
            result = subprocess.run(
                ["typeperf", "\\Processor(_Total)\\% Processor Time", "-sc", "1"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().splitlines()
                if len(lines) >= 2:
                    value_str = lines[-1].strip().strip('"').split(",")[-1].strip('"')
                    cpu_usage = float(value_str) / 100.0
                    utilization["l1_utilization"] = cpu_usage * 0.8
                    utilization["l2_utilization"] = cpu_usage * 0.6
                    utilization["l3_utilization"] = cpu_usage * 0.4
        except (AttributeError, OSError, FileNotFoundError, ValueError, subprocess.TimeoutExpired):
            pass

        return utilization

    # ========================
    # 工具方法
    # ========================

    def _parse_size_string(self, size_str: str) -> int:
        """解析缓存大小字符串 (如 '32K', '256K', '8M')"""
        size_str = size_str.strip().upper()
        multipliers = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}
        for suffix, multiplier in multipliers.items():
            if size_str.endswith(suffix):
                try:
                    return int(float(size_str[:-1]) * multiplier)
                except ValueError:
                    return 0
        try:
            return int(size_str)
        except ValueError:
            return 0

    def _map_cache_level(
        self, level_int: int, type_str: str
    ) -> Optional[CacheLevel]:
        """将sysfs缓存级别和类型映射到CacheLevel枚举"""
        type_str = type_str.strip().upper()
        if level_int == 1:
            if type_str == "INSTRUCTION" or type_str == "I":
                return CacheLevel.L1I
            return CacheLevel.L1D
        elif level_int == 2:
            return CacheLevel.L2
        elif level_int == 3:
            return CacheLevel.L3
        elif level_int == 4:
            return CacheLevel.L4
        return None

    def get_summary(self) -> Dict[str, Any]:
        """
        获取缓存优化器的完整摘要信息

        Returns:
            Dict: 包含拓扑、配置和状态的完整摘要
        """
        return {
            "initialized": self._initialized,
            "platform": platform.system(),
            "topology": self._topology.to_dict() if self._topology else None,
            "config": self.config,
            "optimal_block_size": self.compute_optimal_block_size(),
            "prefetch_distance": self.compute_prefetch_distance(),
        }

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        n_caches = len(self._topology.caches) if self._topology else 0
        return f"CacheOptimization({status}, {n_caches} cache levels)"
