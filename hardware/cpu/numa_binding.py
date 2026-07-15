"""
NumaBinding - NUMA节点绑定模块

提供NUMA (Non-Uniform Memory Access) 架构下的内存和CPU绑定功能，包括：
- NUMA拓扑探测与节点信息采集
- NUMA节点内存分配策略
- CPU亲和性绑定到NUMA节点
- NUMA感知的任务调度
- 跨NUMA节点内存带宽优化

模块路径: hardware/cpu/numa_binding.py
"""

import os
import sys
import ctypes
import logging
import platform
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger(__name__)


class NumaPolicy(Enum):
    """NUMA内存分配策略"""
    DEFAULT = auto()       # 默认策略
    BIND = auto()          # 绑定到指定节点
    INTERLEAVE = auto()    # 交叉分配到多个节点
    PREFERRED = auto()     # 优先使用指定节点
    LOCAL = auto()         # 使用本地节点


@dataclass
class NumaNode:
    """NUMA节点信息"""
    node_id: int
    cpus: List[int] = field(default_factory=list)
    memory_size_mb: int = 0
    free_memory_mb: int = 0
    distance: Dict[int, int] = field(default_factory=dict)
    hugepages_total: int = 0
    hugepages_free: int = 0

    @property
    def cpu_count(self) -> int:
        return len(self.cpus)

    @property
    def memory_usage_pct(self) -> float:
        if self.memory_size_mb <= 0:
            return 0.0
        used = self.memory_size_mb - self.free_memory_mb
        return (used / self.memory_size_mb) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "cpus": self.cpus,
            "cpu_count": self.cpu_count,
            "memory_size_mb": self.memory_size_mb,
            "free_memory_mb": self.free_memory_mb,
            "memory_usage_pct": round(self.memory_usage_pct, 1),
            "distance": self.distance,
            "hugepages_total": self.hugepages_total,
            "hugepages_free": self.hugepages_free,
        }


@dataclass
class NumaTopology:
    """NUMA拓扑结构"""
    nodes: List[NumaNode] = field(default_factory=list)
    num_nodes: int = 0
    is_numa_available: bool = False

    def get_node(self, node_id: int) -> Optional[NumaNode]:
        """获取指定节点"""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def get_all_cpus(self) -> List[int]:
        """获取所有CPU列表"""
        cpus = []
        for node in self.nodes:
            cpus.extend(node.cpus)
        return sorted(set(cpus))

    def find_node_for_cpu(self, cpu_id: int) -> Optional[int]:
        """查找CPU所属的NUMA节点"""
        for node in self.nodes:
            if cpu_id in node.cpus:
                return node.node_id
        return None

    def get_distance(self, src: int, dst: int) -> int:
        """获取两个节点之间的距离"""
        src_node = self.get_node(src)
        if src_node and dst in src_node.distance:
            return src_node.distance[dst]
        return 10  # 默认远距离

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_numa_available": self.is_numa_available,
            "num_nodes": self.num_nodes,
            "nodes": [n.to_dict() for n in self.nodes],
        }


@dataclass
class NumaAllocation:
    """NUMA内存分配记录"""
    address: int = 0
    size: int = 0
    node_id: int = -1
    policy: NumaPolicy = NumaPolicy.DEFAULT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "address": hex(self.address),
            "size": self.size,
            "node_id": self.node_id,
            "policy": self.policy.name,
        }


class NumaBinding:
    """
    NUMA节点绑定管理器

    提供NUMA架构下的内存分配策略、CPU亲和性绑定和任务调度功能。
    支持Linux (libnuma/sysfs)、macOS (单节点)、Windows (NUMA API)。
    """

    # libnuma 常量
    MPOL_DEFAULT = 0
    MPOL_PREFERRED = 1
    MPOL_BIND = 2
    MPOL_INTERLEAVE = 3
    MPOL_LOCAL = 4

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化NUMA绑定管理器

        Args:
            config: 配置字典，支持：
                - preferred_node: int, 优先NUMA节点
                - policy: NumaPolicy, 默认内存分配策略
                - auto_detect: bool, 是否自动检测NUMA拓扑
        """
        self.config = config or {}
        self._system = platform.system()
        self._topology: Optional[NumaTopology] = None
        self._initialized = False
        self._libnuma: Optional[ctypes.CDLL] = None
        self._allocations: List[NumaAllocation] = []
        self._lock = threading.Lock()

        if self.config.get("auto_detect", True):
            self._topology = self.detect_numa_topology()

    def initialize(self) -> bool:
        """
        初始化NUMA绑定管理器

        Returns:
            bool: 初始化是否成功
        """
        try:
            if self._topology is None:
                self._topology = self.detect_numa_topology()

            if self._topology.is_numa_available:
                self._load_libnuma()

            self._initialized = True
            logger.info(
                "NumaBinding initialized: numa_available=%s, nodes=%d",
                self._topology.is_numa_available,
                self._topology.num_nodes,
            )
            return True
        except Exception as e:
            logger.error("Failed to initialize NumaBinding: %s", e)
            return False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def topology(self) -> Optional[NumaTopology]:
        return self._topology

    @property
    def is_numa_available(self) -> bool:
        if self._topology is None:
            return False
        return self._topology.is_numa_available

    # ========================
    # NUMA拓扑探测
    # ========================

    def detect_numa_topology(self) -> NumaTopology:
        """
        探测NUMA拓扑结构

        Returns:
            NumaTopology: NUMA拓扑信息
        """
        if self._system == "Linux":
            return self._detect_linux_numa()
        elif self._system == "Windows":
            return self._detect_windows_numa()
        else:
            return self._detect_single_node()

    def _detect_linux_numa(self) -> NumaTopology:
        """通过Linux sysfs探测NUMA拓扑"""
        numa_base = Path("/sys/devices/system/node")
        if not numa_base.exists():
            return self._detect_single_node()

        nodes: List[NumaNode] = []
        node_dirs = sorted(numa_base.glob("node[0-9]*"))

        if not node_dirs:
            return self._detect_single_node()

        for node_dir in node_dirs:
            try:
                node_id = int(node_dir.name.replace("node", ""))
                node = NumaNode(node_id=node_id)

                # 读取CPU列表
                cpulist_file = node_dir / "cpulist"
                if cpulist_file.exists():
                    node.cpus = self._parse_cpu_list(
                        cpulist_file.read_text().strip()
                    )

                # 读取内存信息
                meminfo_file = node_dir / "meminfo"
                if meminfo_file.exists():
                    for line in meminfo_file.read_text().splitlines():
                        if "MemTotal" in line:
                            node.memory_size_mb = self._parse_meminfo_kb(line) // 1024
                        elif "MemFree" in line:
                            node.free_memory_mb = self._parse_meminfo_kb(line) // 1024

                # 读取距离矩阵
                distance_file = node_dir / "distance"
                if distance_file.exists():
                    distances = distance_file.read_text().strip().split()
                    for other_id, dist_str in enumerate(distances):
                        node.distance[other_id] = int(dist_str)

                # 读取大页信息
                hugepage_dir = node_dir / "hugepages"
                if hugepage_dir.exists():
                    for hp_dir in hugepage_dir.glob("hugepages-*kB"):
                        nr_file = hp_dir / "nr_hugepages"
                        free_file = hp_dir / "free_hugepages"
                        try:
                            if nr_file.exists():
                                node.hugepages_total += int(nr_file.read_text().strip())
                            if free_file.exists():
                                node.hugepages_free += int(free_file.read_text().strip())
                        except ValueError:
                            pass

                nodes.append(node)
            except (OSError, ValueError) as e:
                logger.debug("Error reading NUMA node %s: %s", node_dir, e)

        return NumaTopology(
            nodes=nodes,
            num_nodes=len(nodes),
            is_numa_available=len(nodes) > 1,
        )

    def _detect_windows_numa(self) -> NumaTopology:
        """通过Windows API探测NUMA拓扑"""
        nodes: List[NumaNode] = []

        try:
            result = subprocess.run(
                ["wmic", "numanode", "get",
                 "NodeID,NumberOfProcessors,TotalMemory,AvailableMemory",
                 "/format:csv"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if not line or line.startswith("Node") or line.startswith("Available"):
                        continue
                    parts = [p.strip() for p in line.split(",") if p.strip()]
                    if len(parts) >= 4:
                        try:
                            node_id = int(parts[0])
                            num_procs = int(parts[1])
                            total_mem = int(parts[2])
                            free_mem = int(parts[3])
                            node = NumaNode(
                                node_id=node_id,
                                memory_size_mb=total_mem // 1024,
                                free_memory_mb=free_mem // 1024,
                            )
                            nodes.append(node)
                        except (ValueError, IndexError):
                            continue
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        if len(nodes) <= 1:
            return self._detect_single_node()

        # 获取每个节点的CPU列表
        try:
            result = subprocess.run(
                ["wmic", "numanode", "get", "NodeID", "/format:csv"],
                capture_output=True, text=True, timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return NumaTopology(
            nodes=nodes,
            num_nodes=len(nodes),
            is_numa_available=len(nodes) > 1,
        )

    def _detect_single_node(self) -> NumaTopology:
        """单NUMA节点（UMA架构）"""
        num_cpus = os.cpu_count() or 1
        node = NumaNode(
            node_id=0,
            cpus=list(range(num_cpus)),
        )

        # 尝试获取内存信息
        if self._system == "Linux":
            try:
                meminfo = Path("/proc/meminfo").read_text()
                for line in meminfo.splitlines():
                    if line.startswith("MemTotal:"):
                        node.memory_size_mb = int(line.split()[1]) // 1024
                    elif line.startswith("MemAvailable:"):
                        node.free_memory_mb = int(line.split()[1]) // 1024
                    elif line.startswith("MemFree:"):
                        if node.free_memory_mb == 0:
                            node.free_memory_mb = int(line.split()[1]) // 1024
            except (OSError, ValueError):
                pass
        elif self._system == "Darwin":
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    node.memory_size_mb = int(result.stdout.strip()) // (1024 * 1024)
            except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
                pass

        return NumaTopology(
            nodes=[node],
            num_nodes=1,
            is_numa_available=False,
        )

    def _parse_cpu_list(self, cpu_str: str) -> List[int]:
        """解析CPU列表字符串 (如 '0-3,8-11')"""
        cpus: List[int] = []
        for part in cpu_str.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-", 1)
                cpus.extend(range(int(start), int(end) + 1))
            elif part:
                cpus.append(int(part))
        return cpus

    def _parse_meminfo_kb(self, line: str) -> int:
        """解析meminfo行中的KB值"""
        parts = line.split()
        for part in parts:
            try:
                return int(part)
            except ValueError:
                continue
        return 0

    def _load_libnuma(self) -> None:
        """加载libnuma库"""
        if self._system != "Linux":
            return
        try:
            self._libnuma = ctypes.CDLL("libnuma.so.1", use_errno=True)
            self._libnuma.numa_available.restype = ctypes.c_int
        except OSError:
            logger.debug("libnuma not available, using sysfs fallback")

    # ========================
    # CPU亲和性
    # ========================

    def set_cpu_affinity(self, cpu_ids: List[int], pid: Optional[int] = None) -> bool:
        """
        设置进程/线程的CPU亲和性

        Args:
            cpu_ids: 允许运行的CPU ID列表
            pid: 进程ID，None表示当前进程

        Returns:
            bool: 是否设置成功
        """
        if not cpu_ids:
            return False

        target_pid = pid or os.getpid()

        if self._system == "Linux":
            return self._set_linux_affinity(target_pid, cpu_ids)
        elif self._system == "Windows":
            return self._set_windows_affinity(target_pid, cpu_ids)
        elif self._system == "Darwin":
            logger.warning("macOS does not support CPU affinity setting")
            return False
        return False

    def _set_linux_affinity(self, pid: int, cpu_ids: List[int]) -> bool:
        """设置Linux CPU亲和性"""
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)

            # sched_setaffinity
            cpu_set_size = 128  # 支持最多128个CPU
            CPU_SETSIZE = 128
            CPU_SET = lambda s, cpu: (
                ctypes.c_ulong * (CPU_SETSIZE // (8 * ctypes.sizeof(ctypes.c_ulong)))
            )
            cpu_mask = (ctypes.c_ulong * (cpu_set_size // 8))()

            # 清零
            ctypes.memset(ctypes.addressof(cpu_mask), 0, ctypes.sizeof(cpu_mask))

            # 设置CPU位
            for cpu_id in cpu_ids:
                word_idx = cpu_id // (8 * ctypes.sizeof(ctypes.c_ulong))
                bit_idx = cpu_id % (8 * ctypes.sizeof(ctypes.c_ulong))
                if word_idx < len(cpu_mask):
                    cpu_mask[word_idx] |= (1 << bit_idx)

            result = libc.sched_setaffinity(
                pid, ctypes.sizeof(cpu_mask), ctypes.byref(cpu_mask)
            )

            if result != 0:
                errno = ctypes.get_errno()
                logger.warning(
                    "sched_setaffinity failed: errno=%d, pid=%d, cpus=%s",
                    errno, pid, cpu_ids,
                )
                return False
            return True
        except (OSError, AttributeError) as e:
            logger.warning("Failed to set Linux affinity: %s", e)
            return False

    def _set_windows_affinity(self, pid: int, cpu_ids: List[int]) -> bool:
        """设置Windows CPU亲和性"""
        try:
            import ctypes.wintypes
            handle = ctypes.windll.kernel32.OpenProcess(
                0x0200, False, pid  # PROCESS_SET_INFORMATION
            )
            if not handle:
                return False

            mask = 0
            for cpu_id in cpu_ids:
                mask |= (1 << cpu_id)

            result = ctypes.windll.kernel32.SetProcessAffinityMask(handle, mask)
            ctypes.windll.kernel32.CloseHandle(handle)
            return bool(result)
        except (AttributeError, OSError) as e:
            logger.warning("Failed to set Windows affinity: %s", e)
            return False

    def get_cpu_affinity(self, pid: Optional[int] = None) -> List[int]:
        """
        获取进程/线程的CPU亲和性

        Args:
            pid: 进程ID，None表示当前进程

        Returns:
            List[int]: 允许运行的CPU ID列表
        """
        target_pid = pid or os.getpid()

        if self._system == "Linux":
            return self._get_linux_affinity(target_pid)
        elif self._system == "Windows":
            return self._get_windows_affinity(target_pid)
        return list(range(os.cpu_count() or 1))

    def _get_linux_affinity(self, pid: int) -> List[int]:
        """获取Linux CPU亲和性"""
        cpus: List[int] = []
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            cpu_set_size = 128
            cpu_mask = (ctypes.c_ulong * (cpu_set_size // 8))()
            ctypes.memset(ctypes.addressof(cpu_mask), 0, ctypes.sizeof(cpu_mask))

            result = libc.sched_getaffinity(
                pid, ctypes.sizeof(cpu_mask), ctypes.byref(cpu_mask)
            )

            if result == 0:
                for i in range(os.cpu_count() or 1):
                    word_idx = i // (8 * ctypes.sizeof(ctypes.c_ulong))
                    bit_idx = i % (8 * ctypes.sizeof(ctypes.c_ulong))
                    if word_idx < len(cpu_mask) and (cpu_mask[word_idx] & (1 << bit_idx)):
                        cpus.append(i)
        except (OSError, AttributeError):
            pass
        return cpus

    def _get_windows_affinity(self, pid: int) -> List[int]:
        """获取Windows CPU亲和性"""
        cpus: List[int] = []
        try:
            handle = ctypes.windll.kernel32.OpenProcess(
                0x0400, False, pid  # PROCESS_QUERY_INFORMATION
            )
            if handle:
                mask = ctypes.wintypes.DWORD(0)
                proc_mask = ctypes.wintypes.DWORD(0)
                ctypes.windll.kernel32.GetProcessAffinityMask(
                    handle, ctypes.byref(mask), ctypes.byref(proc_mask)
                )
                ctypes.windll.kernel32.CloseHandle(handle)
                for i in range(64):
                    if mask.value & (1 << i):
                        cpus.append(i)
        except (AttributeError, OSError):
            pass
        return cpus

    def bind_to_node(self, node_id: int, pid: Optional[int] = None) -> bool:
        """
        将进程绑定到指定NUMA节点的CPU上

        Args:
            node_id: NUMA节点ID
            pid: 进程ID

        Returns:
            bool: 是否绑定成功
        """
        if not self._topology or not self._topology.is_numa_available:
            logger.warning("NUMA is not available on this system")
            return False

        node = self._topology.get_node(node_id)
        if not node:
            logger.error("NUMA node %d not found", node_id)
            return False

        if not node.cpus:
            logger.error("No CPUs found on NUMA node %d", node_id)
            return False

        return self.set_cpu_affinity(node.cpus, pid)

    # ========================
    # NUMA内存策略
    # ========================

    def set_numa_policy(
        self, policy: NumaPolicy, node_ids: Optional[List[int]] = None
    ) -> bool:
        """
        设置NUMA内存分配策略

        Args:
            policy: NUMA策略
            node_ids: 相关的节点ID列表

        Returns:
            bool: 是否设置成功
        """
        if not self._topology or not self._topology.is_numa_available:
            return False

        if self._system == "Linux":
            return self._set_linux_numa_policy(policy, node_ids)
        return False

    def _set_linux_numa_policy(
        self, policy: NumaPolicy, node_ids: Optional[List[int]]
    ) -> bool:
        """设置Linux NUMA内存策略"""
        if self._libnuma is None:
            return False

        try:
            mode_map = {
                NumaPolicy.DEFAULT: self.MPOL_DEFAULT,
                NumaPolicy.PREFERRED: self.MPOL_PREFERRED,
                NumaPolicy.BIND: self.MPOL_BIND,
                NumaPolicy.INTERLEAVE: self.MPOL_INTERLEAVE,
                NumaPolicy.LOCAL: self.MPOL_LOCAL,
            }
            mode = mode_map.get(policy, self.MPOL_DEFAULT)

            nodemask = 0
            maxnode = 0
            if node_ids:
                for nid in node_ids:
                    nodemask |= (1 << nid)
                    maxnode = max(maxnode, nid + 1)

            result = self._libnuma.numa_set_preferred(nid if node_ids else -1)
            return result == 0
        except (AttributeError, OSError) as e:
            logger.warning("Failed to set NUMA policy: %s", e)
            return False

    def allocate_on_node(self, size: int, node_id: int) -> Optional[int]:
        """
        在指定NUMA节点上分配内存

        Args:
            size: 分配大小（字节）
            node_id: NUMA节点ID

        Returns:
            Optional[int]: 分配的内存地址，失败返回None
        """
        if not self._topology or not self._topology.is_numa_available:
            return None

        if self._system == "Linux":
            return self._allocate_linux_numa(size, node_id)
        return None

    def _allocate_linux_numa(self, size: int, node_id: int) -> Optional[int]:
        """在Linux指定NUMA节点上分配内存"""
        if self._libnuma is None:
            # 回退: 使用mbind
            try:
                libc = ctypes.CDLL("libc.so.6", use_errno=True)
                ptr = libc.mmap(
                    0, size,
                    0x1 | 0x2,  # PROT_READ | PROT_WRITE
                    0x22,       # MAP_PRIVATE | MAP_ANONYMOUS
                    -1, 0,
                )
                if ptr == -1:
                    return None

                # 设置NUMA策略
                maxnode = node_id + 2
                nodemask = (ctypes.c_ulong * ((maxnode + 63) // 64))()
                idx = node_id // 64
                bit = node_id % 64
                nodemask[idx] = (1 << bit)

                libc.mbind(
                    ptr, size, self.MPOL_BIND,
                    ctypes.byref(nodemask), maxnode, 0,
                )
                return ptr
            except (OSError, AttributeError):
                return None

        try:
            ptr = self._libnuma.numa_alloc_onnode(size, node_id)
            if ptr and ptr != -1:
                with self._lock:
                    self._allocations.append(NumaAllocation(
                        address=ptr, size=size,
                        node_id=node_id, policy=NumaPolicy.BIND,
                    ))
                return ptr
        except (AttributeError, OSError):
            pass
        return None

    def allocate_interleaved(self, size: int, node_ids: Optional[List[int]] = None) -> Optional[int]:
        """
        交叉分配内存到多个NUMA节点

        Args:
            size: 分配大小
            node_ids: 节点ID列表，None表示所有节点

        Returns:
            Optional[int]: 分配的内存地址
        """
        if not self._topology or not self._topology.is_numa_available:
            return None

        if node_ids is None:
            node_ids = [n.node_id for n in self._topology.nodes]

        if self._system == "Linux" and self._libnuma:
            try:
                mask = 0
                for nid in node_ids:
                    mask |= (1 << nid)
                ptr = self._libnuma.numa_alloc_interleaved(size, mask)
                if ptr and ptr != -1:
                    with self._lock:
                        self._allocations.append(NumaAllocation(
                            address=ptr, size=size,
                            node_id=-1, policy=NumaPolicy.INTERLEAVE,
                        ))
                    return ptr
            except (AttributeError, OSError):
                pass
        return None

    def free_numa_memory(self, address: int) -> bool:
        """
        释放NUMA分配的内存

        Args:
            address: 内存地址

        Returns:
            bool: 是否释放成功
        """
        if self._system == "Linux" and self._libnuma:
            try:
                self._libnuma.numa_free(address, 0)
                with self._lock:
                    self._allocations = [
                        a for a in self._allocations if a.address != address
                    ]
                return True
            except (AttributeError, OSError):
                pass
        return False

    # ========================
    # NUMA感知调度
    # ========================

    def get_optimal_node(self, current_cpu: int = -1) -> int:
        """
        获取最优NUMA节点

        基于当前CPU位置和各节点内存可用量选择最优节点。

        Args:
            current_cpu: 当前CPU ID，-1表示自动检测

        Returns:
            int: 推荐的NUMA节点ID
        """
        if not self._topology or not self._topology.is_numa_available:
            return 0

        # 如果知道当前CPU，优先使用本地节点
        if current_cpu >= 0:
            local_node = self._topology.find_node_for_cpu(current_cpu)
            if local_node is not None:
                node = self._topology.get_node(local_node)
                if node and node.free_memory_mb > 0:
                    return local_node

        # 否则选择内存最充足的节点
        best_node = 0
        best_free = 0
        for node in self._topology.nodes:
            if node.free_memory_mb > best_free:
                best_free = node.free_memory_mb
                best_node = node.node_id

        return best_node

    def create_numa_aware_thread_pool(
        self, num_workers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        创建NUMA感知的线程池配置

        根据NUMA拓扑自动分配工作线程到各个节点。

        Args:
            num_workers: 工作线程总数，None表示自动

        Returns:
            Dict: 线程池配置，包含每节点的worker分配
        """
        if not self._topology:
            return {"workers_per_node": {0: num_workers or os.cpu_count() or 1}}

        if not self._topology.is_numa_available:
            total = num_workers or os.cpu_count() or 1
            return {"workers_per_node": {0: total}}

        total_cpus = sum(n.cpu_count for n in self._topology.nodes)
        total = num_workers or total_cpus

        workers_per_node: Dict[int, int] = {}
        remaining = total

        # 按CPU比例分配
        for node in self._topology.nodes:
            if node.cpu_count > 0 and remaining > 0:
                share = max(1, int(total * node.cpu_count / total_cpus))
                share = min(share, remaining)
                workers_per_node[node.node_id] = share
                remaining -= share

        # 分配剩余
        if remaining > 0 and workers_per_node:
            first_node = min(workers_per_node.keys())
            workers_per_node[first_node] += remaining

        return {
            "workers_per_node": workers_per_node,
            "total_workers": total,
            "num_nodes": self._topology.num_nodes,
        }

    def get_numa_bandwidth_estimate(self, src: int, dst: int) -> Dict[str, float]:
        """
        估算两个NUMA节点间的内存带宽

        Args:
            src: 源节点ID
            dst: 目标节点ID

        Returns:
            Dict: 带宽估算 (GB/s)
        """
        if not self._topology:
            return {"local_bandwidth_gbps": 0.0, "remote_bandwidth_gbps": 0.0}

        distance = self._topology.get_distance(src, dst)

        # 基于距离估算带宽
        # 本地访问: ~50-100 GB/s (DDR4/DDR5)
        # 远程访问: ~20-40 GB/s (QPI/UPI)
        local_bw = 80.0  # GB/s
        remote_bw = local_bw / max(distance / 10.0, 1.0)

        if src == dst:
            return {
                "bandwidth_gbps": local_bw,
                "type": "local",
                "latency_ns": 80,
            }
        else:
            return {
                "bandwidth_gbps": remote_bw,
                "type": "remote",
                "latency_ns": 80 + distance * 5,
                "distance": distance,
            }

    def refresh_topology(self) -> None:
        """刷新NUMA拓扑信息（更新内存使用量等）"""
        if self._topology:
            new_topo = self.detect_numa_topology()
            if new_topo.is_numa_available:
                # 更新内存信息但保留其他数据
                for new_node in new_topo.nodes:
                    old_node = self._topology.get_node(new_node.node_id)
                    if old_node:
                        old_node.free_memory_mb = new_node.free_memory_mb
                        old_node.hugepages_free = new_node.hugepages_free

    def get_summary(self) -> Dict[str, Any]:
        """获取NUMA绑定管理器的完整摘要"""
        return {
            "initialized": self._initialized,
            "platform": self._system,
            "numa_available": self.is_numa_available,
            "topology": self._topology.to_dict() if self._topology else None,
            "active_allocations": len(self._allocations),
            "libnuma_loaded": self._libnuma is not None,
        }

    def __repr__(self) -> str:
        numa = "NUMA" if self.is_numa_available else "UMA"
        nodes = self._topology.num_nodes if self._topology else 0
        return f"NumaBinding({numa}, {nodes} nodes)"
