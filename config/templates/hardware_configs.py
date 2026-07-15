"""
HardwareConfigs - 硬件配置模板模块

该模块提供了完整的硬件配置管理功能，包括GPU、CPU、内存、存储和网络配置。
支持配置验证、序列化和反序列化操作。

模块路径: config/templates/hardware_configs.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union


class GPUType(Enum):
    """GPU类型枚举"""
    NVIDIA_A100 = "NVIDIA_A100"
    NVIDIA_A100_80GB = "NVIDIA_A100_80GB"
    NVIDIA_A100_40GB = "NVIDIA_A100_40GB"
    NVIDIA_H100 = "NVIDIA_H100"
    NVIDIA_H100_80GB = "NVIDIA_H100_80GB"
    NVIDIA_H200 = "NVIDIA_H200"
    NVIDIA_L40S = "NVIDIA_L40S"
    NVIDIA_L4 = "NVIDIA_L4"
    NVIDIA_RTX_4090 = "NVIDIA_RTX_4090"
    NVIDIA_RTX_3090 = "NVIDIA_RTX_3090"
    NVIDIA_RTX_A6000 = "NVIDIA_RTX_A6000"
    NVIDIA_T4 = "NVIDIA_T4"
    NVIDIA_V100 = "NVIDIA_V100"
    AMD_MI300X = "AMD_MI300X"
    AMD_MI250X = "AMD_MI250X"
    INTEL_MAX_1550 = "INTEL_MAX_1550"
    APPLE_M1_MAX = "Apple_M1_Max"
    APPLE_M2_MAX = "Apple_M2_Max"
    APPLE_M3_MAX = "Apple_M3_Max"


class CPUType(Enum):
    """CPU类型枚举"""
    INTEL_XEON = "Intel_Xeon"
    INTEL_XEON_PLATINUM = "Intel_Xeon_Platinum"
    INTEL_CORE_I9 = "Intel_Core_i9"
    AMD_EPYC = "AMD_EPYC"
    AMD_EPYC_MILAN = "AMD_EPYC_Milan"
    AMD_EPYC_GENOA = "AMD_EPYC_Genoa"
    AMD_RYZEN_9 = "AMD_Ryzen_9"
    ARM_GRAVITON3 = "ARM_Graviton3"
    ARM_GRAVITON4 = "ARM_Graviton4"
    APPLE_M1_MAX = "Apple_M1_Max"
    APPLE_M2_MAX = "Apple_M2_Max"
    APPLE_M3_MAX = "Apple_M3_Max"


class StorageType(Enum):
    """存储类型枚举"""
    NVME_SSD = "NVMe_SSD"
    SATA_SSD = "SATA_SSD"
    HDD = "HDD"
    NETWORK_STORAGE = "Network_Storage"
    OBJECT_STORAGE = "Object_Storage"


class NetworkType(Enum):
    """网络类型枚举"""
    ETHERNET_1GB = "Ethernet_1Gb"
    ETHERNET_10GB = "Ethernet_10Gb"
    ETHERNET_25GB = "Ethernet_25Gb"
    ETHERNET_100GB = "Ethernet_100Gb"
    INFINIBAND_100GB = "InfiniBand_100Gb"
    INFINIBAND_200GB = "InfiniBand_200Gb"
    INFINIBAND_400GB = "InfiniBand_400Gb"


@dataclass
class GPUConfig:
    """GPU配置类 - 配置GPU相关的硬件参数"""
    gpu_type: Union[GPUType, str] = GPUType.NVIDIA_A100
    gpu_count: int = 1
    vram_gb: int = 80
    cuda_cores: Optional[int] = None
    tensor_cores: Optional[int] = None
    compute_capability: Optional[str] = None
    pcie_bandwidth_gbps: Optional[float] = None
    nvlink_enabled: bool = False
    nvlink_bandwidth_gbps: Optional[float] = None
    mixed_precision: bool = True
    fp16_supported: bool = True
    bf16_supported: bool = True
    fp8_supported: bool = False
    int8_supported: bool = True
    int4_supported: bool = False
    
    def __post_init__(self):
        if isinstance(self.gpu_type, str):
            try:
                self.gpu_type = GPUType(self.gpu_type)
            except ValueError:
                pass
    
    def validate(self) -> bool:
        """验证GPU配置的有效性"""
        if self.gpu_count < 1:
            raise ValueError(f"GPU数量必须大于等于1，当前值: {self.gpu_count}")
        if self.vram_gb < 1:
            raise ValueError(f"显存大小必须大于等于1GB，当前值: {self.vram_gb}")
        if self.cuda_cores is not None and self.cuda_cores < 0:
            raise ValueError(f"CUDA核心数不能为负数")
        if self.tensor_cores is not None and self.tensor_cores < 0:
            raise ValueError(f"Tensor核心数不能为负数")
        if self.pcie_bandwidth_gbps is not None and self.pcie_bandwidth_gbps < 0:
            raise ValueError(f"PCIe带宽不能为负数")
        if self.nvlink_bandwidth_gbps is not None and self.nvlink_bandwidth_gbps < 0:
            raise ValueError(f"NVLink带宽不能为负数")
        return True
    
    def total_vram_gb(self) -> int:
        """计算总显存大小"""
        return self.gpu_count * self.vram_gb
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if isinstance(self.gpu_type, GPUType):
            result['gpu_type'] = self.gpu_type.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GPUConfig:
        return cls(**data)
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> GPUConfig:
        return cls.from_dict(json.loads(json_str))
    
    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(self.to_json(), encoding='utf-8')
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> GPUConfig:
        return cls.from_json(Path(path).read_text(encoding='utf-8'))


@dataclass
class CPUConfig:
    """CPU配置类 - 配置CPU相关的硬件参数"""
    cpu_type: Union[CPUType, str] = CPUType.INTEL_XEON
    cpu_cores: int = 32
    cpu_threads: int = 64
    cpu_frequency_ghz: float = 2.5
    cpu_boost_frequency_ghz: Optional[float] = None
    l1_cache_kb: Optional[int] = None
    l2_cache_kb: Optional[int] = None
    l3_cache_mb: Optional[int] = None
    memory_channels: int = 8
    avx_support: bool = True
    avx2_support: bool = True
    avx512_support: bool = False
    numa_nodes: int = 1
    
    def __post_init__(self):
        if isinstance(self.cpu_type, str):
            try:
                self.cpu_type = CPUType(self.cpu_type)
            except ValueError:
                pass
    
    def validate(self) -> bool:
        """验证CPU配置的有效性"""
        if self.cpu_cores < 1:
            raise ValueError(f"CPU核心数必须大于等于1，当前值: {self.cpu_cores}")
        if self.cpu_threads < self.cpu_cores:
            raise ValueError(f"CPU线程数({self.cpu_threads})不能小于核心数({self.cpu_cores})")
        if self.cpu_frequency_ghz <= 0:
            raise ValueError(f"CPU频率必须大于0")
        if self.cpu_boost_frequency_ghz is not None:
            if self.cpu_boost_frequency_ghz < self.cpu_frequency_ghz:
                raise ValueError(f"睿频不能低于基础频率")
        if self.memory_channels < 1:
            raise ValueError(f"内存通道数必须大于等于1")
        if self.numa_nodes < 1:
            raise ValueError(f"NUMA节点数必须大于等于1")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if isinstance(self.cpu_type, CPUType):
            result['cpu_type'] = self.cpu_type.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CPUConfig:
        return cls(**data)
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> CPUConfig:
        return cls.from_dict(json.loads(json_str))
    
    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(self.to_json(), encoding='utf-8')
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> CPUConfig:
        return cls.from_json(Path(path).read_text(encoding='utf-8'))


@dataclass
class MemoryConfig:
    """内存配置类 - 配置系统内存相关的硬件参数"""
    total_memory_gb: int = 512
    memory_type: str = "DDR4"
    memory_speed_mhz: int = 3200
    memory_channels: int = 8
    ecc_enabled: bool = True
    memory_bandwidth_gbps: Optional[float] = None
    numa_balanced: bool = True
    
    def validate(self) -> bool:
        """验证内存配置的有效性"""
        if self.total_memory_gb < 1:
            raise ValueError(f"内存大小必须大于等于1GB")
        valid_types = ["DDR3", "DDR4", "DDR5", "DDR6", "HBM2", "HBM2e", "HBM3", "LPDDR4", "LPDDR5"]
        if self.memory_type not in valid_types:
            raise ValueError(f"不支持的内存类型: {self.memory_type}")
        if self.memory_speed_mhz < 100:
            raise ValueError(f"内存频率过低")
        if self.memory_channels < 1:
            raise ValueError(f"内存通道数必须大于等于1")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemoryConfig:
        return cls(**data)
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> MemoryConfig:
        return cls.from_dict(json.loads(json_str))
    
    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(self.to_json(), encoding='utf-8')
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> MemoryConfig:
        return cls.from_json(Path(path).read_text(encoding='utf-8'))


@dataclass
class StorageConfig:
    """存储配置类 - 配置存储系统相关的硬件参数"""
    storage_type: Union[StorageType, str] = StorageType.NVME_SSD
    capacity_gb: int = 2000
    read_speed_mbps: int = 3500
    write_speed_mbps: int = 3000
    iops_read: Optional[int] = None
    iops_write: Optional[int] = None
    raid_config: Optional[str] = None
    mount_point: str = "/data"
    
    def __post_init__(self):
        if isinstance(self.storage_type, str):
            try:
                self.storage_type = StorageType(self.storage_type)
            except ValueError:
                pass
    
    def validate(self) -> bool:
        """验证存储配置的有效性"""
        if self.capacity_gb < 1:
            raise ValueError(f"存储容量必须大于等于1GB")
        if self.read_speed_mbps < 0:
            raise ValueError(f"读取速度不能为负数")
        if self.write_speed_mbps < 0:
            raise ValueError(f"写入速度不能为负数")
        valid_raid = [None, "RAID0", "RAID1", "RAID5", "RAID6", "RAID10", "RAID50", "RAID60"]
        if self.raid_config not in valid_raid:
            raise ValueError(f"不支持的RAID配置: {self.raid_config}")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if isinstance(self.storage_type, StorageType):
            result['storage_type'] = self.storage_type.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StorageConfig:
        return cls(**data)
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> StorageConfig:
        return cls.from_dict(json.loads(json_str))
    
    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(self.to_json(), encoding='utf-8')
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> StorageConfig:
        return cls.from_json(Path(path).read_text(encoding='utf-8'))


@dataclass
class NetworkConfig:
    """网络配置类 - 配置网络相关的硬件参数"""
    network_type: Union[NetworkType, str] = NetworkType.ETHERNET_10GB
    bandwidth_gbps: float = 10.0
    latency_ms: float = 1.0
    rdma_enabled: bool = False
    nccl_enabled: bool = True
    tcp_enabled: bool = True
    ib_enabled: bool = False
    network_interface: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.network_type, str):
            try:
                self.network_type = NetworkType(self.network_type)
            except ValueError:
                pass
    
    def validate(self) -> bool:
        """验证网络配置的有效性"""
        if self.bandwidth_gbps <= 0:
            raise ValueError(f"网络带宽必须大于0")
        if self.latency_ms < 0:
            raise ValueError(f"网络延迟不能为负数")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if isinstance(self.network_type, NetworkType):
            result['network_type'] = self.network_type.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NetworkConfig:
        return cls(**data)
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> NetworkConfig:
        return cls.from_dict(json.loads(json_str))
    
    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(self.to_json(), encoding='utf-8')
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> NetworkConfig:
        return cls.from_json(Path(path).read_text(encoding='utf-8'))


@dataclass
class HardwareConfig:
    """综合硬件配置类 - 整合GPU、CPU、内存、存储和网络配置"""
    gpu: GPUConfig = field(default_factory=GPUConfig)
    cpu: CPUConfig = field(default_factory=CPUConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    hostname: Optional[str] = None
    platform: str = "Linux"
    environment: str = "cloud"
    
    def validate(self) -> bool:
        """验证所有硬件配置的有效性"""
        self.gpu.validate()
        self.cpu.validate()
        self.memory.validate()
        self.storage.validate()
        self.network.validate()
        valid_platforms = ["Linux", "Windows", "macOS", "FreeBSD"]
        if self.platform not in valid_platforms:
            raise ValueError(f"不支持的平台: {self.platform}")
        valid_environments = ["cloud", "on-premise", "edge", "hybrid"]
        if self.environment not in valid_environments:
            raise ValueError(f"不支持的环境: {self.environment}")
        return True
    
    def estimate_training_capacity(self) -> Dict[str, Any]:
        """估算硬件的训练容量"""
        total_vram = self.gpu.total_vram_gb()
        estimated_params_billion = total_vram / 16
        return {
            "total_gpu_vram_gb": total_vram,
            "estimated_trainable_params_billion": round(estimated_params_billion, 2),
            "gpu_count": self.gpu.gpu_count,
            "cpu_cores": self.cpu.cpu_cores,
            "total_memory_gb": self.memory.total_memory_gb,
            "storage_capacity_gb": self.storage.capacity_gb,
            "network_bandwidth_gbps": self.network.bandwidth_gbps
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gpu": self.gpu.to_dict(),
            "cpu": self.cpu.to_dict(),
            "memory": self.memory.to_dict(),
            "storage": self.storage.to_dict(),
            "network": self.network.to_dict(),
            "hostname": self.hostname,
            "platform": self.platform,
            "environment": self.environment
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HardwareConfig:
        return cls(
            gpu=GPUConfig.from_dict(data.get("gpu", {})),
            cpu=CPUConfig.from_dict(data.get("cpu", {})),
            memory=MemoryConfig.from_dict(data.get("memory", {})),
            storage=StorageConfig.from_dict(data.get("storage", {})),
            network=NetworkConfig.from_dict(data.get("network", {})),
            hostname=data.get("hostname"),
            platform=data.get("platform", "Linux"),
            environment=data.get("environment", "cloud")
        )
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> HardwareConfig:
        return cls.from_dict(json.loads(json_str))
    
    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(self.to_json(), encoding='utf-8')
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> HardwareConfig:
        return cls.from_json(Path(path).read_text(encoding='utf-8'))


class HardwarePresets:
    """预定义的硬件配置模板"""
    
    @staticmethod
    def single_a100_80gb() -> HardwareConfig:
        """单卡NVIDIA A100 80GB配置"""
        return HardwareConfig(
            gpu=GPUConfig(gpu_type=GPUType.NVIDIA_A100_80GB, gpu_count=1, vram_gb=80),
            cpu=CPUConfig(cpu_cores=32, cpu_threads=64),
            memory=MemoryConfig(total_memory_gb=256),
            storage=StorageConfig(capacity_gb=2000),
            network=NetworkConfig(bandwidth_gbps=10)
        )
    
    @staticmethod
    def dgx_a100_8gpu() -> HardwareConfig:
        """NVIDIA DGX A100 8卡配置"""
        return HardwareConfig(
            gpu=GPUConfig(
                gpu_type=GPUType.NVIDIA_A100_80GB, gpu_count=8, vram_gb=80,
                nvlink_enabled=True, nvlink_bandwidth_gbps=600.0
            ),
            cpu=CPUConfig(cpu_type=CPUType.AMD_EPYC_MILAN, cpu_cores=128, cpu_threads=256),
            memory=MemoryConfig(total_memory_gb=2048, memory_type="DDR4"),
            storage=StorageConfig(capacity_gb=30000, read_speed_mbps=7000),
            network=NetworkConfig(
                network_type=NetworkType.INFINIBAND_200GB, bandwidth_gbps=200, rdma_enabled=True
            )
        )
    
    @staticmethod
    def h100_8gpu_node() -> HardwareConfig:
        """H100 8卡节点配置"""
        return HardwareConfig(
            gpu=GPUConfig(
                gpu_type=GPUType.NVIDIA_H100_80GB, gpu_count=8, vram_gb=80,
                nvlink_enabled=True, nvlink_bandwidth_gbps=900.0, fp8_supported=True
            ),
            cpu=CPUConfig(cpu_type=CPUType.INTEL_XEON_PLATINUM, cpu_cores=96, cpu_threads=192),
            memory=MemoryConfig(total_memory_gb=2048, memory_type="DDR5"),
            storage=StorageConfig(capacity_gb=30000, read_speed_mbps=10000),
            network=NetworkConfig(
                network_type=NetworkType.INFINIBAND_400GB, bandwidth_gbps=400, rdma_enabled=True
            )
        )
    
    @staticmethod
    def rtx_4090_workstation() -> HardwareConfig:
        """RTX 4090工作站配置"""
        return HardwareConfig(
            gpu=GPUConfig(gpu_type=GPUType.NVIDIA_RTX_4090, gpu_count=1, vram_gb=24),
            cpu=CPUConfig(cpu_type=CPUType.INTEL_CORE_I9, cpu_cores=24, cpu_threads=32),
            memory=MemoryConfig(total_memory_gb=128, memory_type="DDR5"),
            storage=StorageConfig(capacity_gb=4000),
            network=NetworkConfig(bandwidth_gbps=1)
        )
    
    @staticmethod
    def apple_m3_max() -> HardwareConfig:
        """Apple M3 Max配置"""
        return HardwareConfig(
            gpu=GPUConfig(gpu_type=GPUType.APPLE_M3_MAX, gpu_count=1, vram_gb=128),
            cpu=CPUConfig(cpu_type=CPUType.APPLE_M3_MAX, cpu_cores=16, cpu_threads=16),
            memory=MemoryConfig(total_memory_gb=128, memory_type="LPDDR5"),
            storage=StorageConfig(capacity_gb=2000),
            network=NetworkConfig(bandwidth_gbps=1),
            platform="macOS"
        )


__all__ = [
    "GPUType", "CPUType", "StorageType", "NetworkType",
    "GPUConfig", "CPUConfig", "MemoryConfig", "StorageConfig", "NetworkConfig",
    "HardwareConfig", "HardwarePresets"
]
