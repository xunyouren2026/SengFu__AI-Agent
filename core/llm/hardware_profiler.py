"""
AGI Unified Framework - Hardware Profiler Module
硬件能力分析器：GPU检测(CUDA/ROCm/MPS)、内存分析、计算能力评分、
最优批大小估计、混合精度推荐

提供全面的硬件能力分析和优化建议，支持多种GPU架构和推理场景。
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class GPUVendor(str, Enum):
    """GPU厂商"""
    NVIDIA = "nvidia"
    AMD = "amd"
    APPLE = "apple"
    INTEL = "intel"
    UNKNOWN = "unknown"


class ComputeCapability(str, Enum):
    """计算能力等级"""
    LOW = "low"           # < 5 TFLOPS
    MEDIUM = "medium"     # 5-20 TFLOPS
    HIGH = "high"         # 20-50 TFLOPS
    VERY_HIGH = "very_high"  # > 50 TFLOPS


class PrecisionType(str, Enum):
    """精度类型"""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


@dataclass
class GPUSpecs:
    """GPU规格"""
    name: str
    vendor: GPUVendor
    device_id: int = 0
    total_memory_gb: float = 0.0
    compute_capability: str = ""
    cuda_cores: int = 0
    tensor_cores: int = 0
    clock_speed_mhz: int = 0
    memory_bandwidth_gbps: float = 0.0
    pcie_bandwidth_gbps: float = 0.0
    supports_fp16: bool = False
    supports_bf16: bool = False
    supports_int8: bool = False
    supports_int4: bool = False
    driver_version: str = ""
    
    @property
    def theoretical_tflops_fp32(self) -> float:
        """理论FP32 TFLOPS"""
        if self.vendor == GPUVendor.NVIDIA:
            # 简化估算：CUDA核心数 * 时钟频率 * 2
            return self.cuda_cores * self.clock_speed_mhz * 2 / 1e6
        return 0.0
    
    @property
    def theoretical_tflops_fp16(self) -> float:
        """理论FP16 TFLOPS (Tensor Core加速)"""
        if self.vendor == GPUVendor.NVIDIA and self.tensor_cores > 0:
            return self.theoretical_tflops_fp32 * 8  # Tensor Core约8倍加速
        return self.theoretical_tflops_fp32 * 2


@dataclass
class MemoryStats:
    """内存统计"""
    total_bytes: int = 0
    used_bytes: int = 0
    free_bytes: int = 0
    reserved_bytes: int = 0
    allocated_bytes: int = 0
    cached_bytes: int = 0
    
    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024 ** 3)
    
    @property
    def used_gb(self) -> float:
        return self.used_bytes / (1024 ** 3)
    
    @property
    def free_gb(self) -> float:
        return self.free_bytes / (1024 ** 3)
    
    @property
    def utilization_percent(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return (self.used_bytes / self.total_bytes) * 100


@dataclass
class ComputeScore:
    """计算能力评分"""
    overall_score: float = 0.0  # 0-100
    compute_score: float = 0.0
    memory_score: float = 0.0
    bandwidth_score: float = 0.0
    efficiency_score: float = 0.0
    capability_level: ComputeCapability = ComputeCapability.LOW
    
    @property
    def is_suitable_for_large_models(self) -> bool:
        return self.overall_score >= 70
    
    @property
    def recommended_max_model_size_gb(self) -> float:
        """推荐的最大模型大小"""
        if self.overall_score >= 90:
            return 70.0
        elif self.overall_score >= 70:
            return 40.0
        elif self.overall_score >= 50:
            return 20.0
        else:
            return 7.0


@dataclass
class BatchSizeEstimate:
    """批大小估计"""
    optimal_batch_size: int = 1
    max_batch_size: int = 1
    memory_limited_batch_size: int = 1
    compute_limited_batch_size: int = 1
    latency_ms_estimate: float = 0.0
    throughput_tok_per_sec: float = 0.0
    memory_usage_gb: float = 0.0


@dataclass
class PrecisionRecommendation:
    """精度推荐"""
    recommended_precision: PrecisionType
    available_precisions: List[PrecisionType]
    expected_speedup: float
    expected_memory_reduction: float
    quality_impact: str
    notes: str


class GPUProfiler:
    """
    GPU分析器
    
    检测和分析GPU硬件，支持NVIDIA CUDA、AMD ROCm、Apple MPS等。
    """
    
    def __init__(self):
        self._gpu_info: List[GPUSpecs] = []
        self._lock = threading.RLock()
        self._detected_vendor: Optional[GPUVendor] = None
    
    def detect_gpus(self) -> List[GPUSpecs]:
        """
        检测所有可用GPU
        
        Returns:
            GPU规格列表
        """
        gpus = []
        
        # 尝试NVIDIA
        nvidia_gpus = self._detect_nvidia_gpus()
        if nvidia_gpus:
            gpus.extend(nvidia_gpus)
            self._detected_vendor = GPUVendor.NVIDIA
        
        # 尝试AMD
        if not gpus:
            amd_gpus = self._detect_amd_gpus()
            if amd_gpus:
                gpus.extend(amd_gpus)
                self._detected_vendor = GPUVendor.AMD
        
        # 尝试Apple Silicon
        if not gpus:
            apple_gpus = self._detect_apple_gpus()
            if apple_gpus:
                gpus.extend(apple_gpus)
                self._detected_vendor = GPUVendor.APPLE
        
        with self._lock:
            self._gpu_info = gpus
        
        return gpus
    
    def _detect_nvidia_gpus(self) -> List[GPUSpecs]:
        """检测NVIDIA GPU"""
        gpus = []
        
        try:
            # 尝试使用nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version,clocks.gr",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                for i, line in enumerate(lines):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 5:
                        gpu = self._parse_nvidia_gpu(i, parts)
                        gpus.append(gpu)
        
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass
        
        # 如果没有nvidia-smi，尝试检测环境变量
        if not gpus and os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
            # 创建模拟GPU信息
            gpus.append(self._create_mock_nvidia_gpu())
        
        return gpus
    
    def _parse_nvidia_gpu(self, device_id: int, parts: List[str]) -> GPUSpecs:
        """解析NVIDIA GPU信息"""
        name = parts[0]
        memory_mb = float(parts[1]) if parts[1] else 0
        compute_cap = parts[2] if len(parts) > 2 else ""
        driver = parts[3] if len(parts) > 3 else ""
        clock_mhz = int(float(parts[4])) if len(parts) > 4 and parts[4] else 1000
        
        # 推断CUDA核心数（近似值）
        cuda_cores = self._estimate_cuda_cores(name)
        tensor_cores = self._estimate_tensor_cores(name, compute_cap)
        
        return GPUSpecs(
            name=name,
            vendor=GPUVendor.NVIDIA,
            device_id=device_id,
            total_memory_gb=memory_mb / 1024,
            compute_capability=compute_cap,
            cuda_cores=cuda_cores,
            tensor_cores=tensor_cores,
            clock_speed_mhz=clock_mhz,
            driver_version=driver,
            supports_fp16=True,
            supports_bf16=self._supports_bf16(compute_cap),
            supports_int8=True,
            supports_int4=True,
        )
    
    def _estimate_cuda_cores(self, gpu_name: str) -> int:
        """根据GPU名称估算CUDA核心数"""
        name_lower = gpu_name.lower()
        
        # RTX 40系列
        if "rtx 4090" in name_lower:
            return 16384
        elif "rtx 4080" in name_lower:
            return 9728
        elif "rtx 4070" in name_lower:
            return 5888
        
        # RTX 30系列
        elif "rtx 3090" in name_lower:
            return 10496
        elif "rtx 3080" in name_lower:
            return 8704
        elif "rtx 3070" in name_lower:
            return 5888
        
        # A100/H100
        elif "a100" in name_lower:
            return 6912
        elif "h100" in name_lower:
            return 16896
        
        # 默认估算
        return 4096
    
    def _estimate_tensor_cores(self, gpu_name: str, compute_cap: str) -> int:
        """估算Tensor Core数量"""
        # Volta及以后架构支持Tensor Core
        if compute_cap and float(compute_cap.replace(".", "")) >= 70:
            return self._estimate_cuda_cores(gpu_name) // 64
        return 0
    
    def _supports_bf16(self, compute_cap: str) -> bool:
        """检查是否支持BF16"""
        if compute_cap:
            cap_val = float(compute_cap.replace(".", ""))
            return cap_val >= 80  # Ampere及以后
        return False
    
    def _create_mock_nvidia_gpu(self) -> GPUSpecs:
        """创建模拟NVIDIA GPU"""
        return GPUSpecs(
            name="NVIDIA GPU (Mock)",
            vendor=GPUVendor.NVIDIA,
            total_memory_gb=16.0,
            compute_capability="8.6",
            cuda_cores=8192,
            tensor_cores=256,
            clock_speed_mhz=1500,
            supports_fp16=True,
            supports_bf16=True,
            supports_int8=True,
            supports_int4=True,
        )
    
    def _detect_amd_gpus(self) -> List[GPUSpecs]:
        """检测AMD GPU"""
        gpus = []
        
        try:
            # 尝试rocm-smi
            result = subprocess.run(
                ["rocm-smi", "--showproductname", "--showmeminfo", "vram"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            if result.returncode == 0:
                # 解析rocm-smi输出
                gpu = GPUSpecs(
                    name="AMD GPU",
                    vendor=GPUVendor.AMD,
                    total_memory_gb=16.0,
                    supports_fp16=True,
                    supports_bf16=True,
                    supports_int8=True,
                )
                gpus.append(gpu)
        
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return gpus
    
    def _detect_apple_gpus(self) -> List[GPUSpecs]:
        """检测Apple Silicon GPU"""
        gpus = []
        
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            try:
                # 检测M系列芯片
                result = subprocess.run(
                    ["system_profiler", "SPHardwareDataType"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                
                if result.returncode == 0:
                    output = result.stdout
                    
                    # 解析芯片型号
                    chip_name = "Apple Silicon"
                    memory_gb = 16.0
                    
                    if "M3 Max" in output:
                        chip_name = "Apple M3 Max"
                        memory_gb = 128.0
                    elif "M3 Pro" in output:
                        chip_name = "Apple M3 Pro"
                        memory_gb = 36.0
                    elif "M3" in output:
                        chip_name = "Apple M3"
                        memory_gb = 24.0
                    elif "M2 Max" in output:
                        chip_name = "Apple M2 Max"
                        memory_gb = 96.0
                    elif "M2" in output:
                        chip_name = "Apple M2"
                        memory_gb = 24.0
                    elif "M1 Max" in output:
                        chip_name = "Apple M1 Max"
                        memory_gb = 64.0
                    elif "M1" in output:
                        chip_name = "Apple M1"
                        memory_gb = 16.0
                    
                    gpu = GPUSpecs(
                        name=chip_name,
                        vendor=GPUVendor.APPLE,
                        total_memory_gb=memory_gb,
                        supports_fp16=True,
                        supports_bf16=False,  # MPS不支持BF16
                        supports_int8=True,
                    )
                    gpus.append(gpu)
            
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        
        return gpus
    
    def get_gpu_memory(self, device_id: int = 0) -> MemoryStats:
        """
        获取GPU内存统计
        
        Args:
            device_id: GPU设备ID
            
        Returns:
            内存统计
        """
        stats = MemoryStats()
        
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
                 "--format=csv,noheader,nounits", "-i", str(device_id)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                if len(parts) >= 3:
                    stats.total_bytes = int(float(parts[0])) * 1024 * 1024
                    stats.used_bytes = int(float(parts[1])) * 1024 * 1024
                    stats.free_bytes = int(float(parts[2])) * 1024 * 1024
        
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return stats
    
    def get_gpu_utilization(self, device_id: int = 0) -> float:
        """获取GPU利用率"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits", "-i", str(device_id)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            if result.returncode == 0:
                return float(result.stdout.strip())
        
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
        
        return 0.0


class MemoryProfiler:
    """
    内存分析器
    
    分析系统内存和GPU内存使用情况，提供内存优化建议。
    """
    
    def __init__(self, gpu_profiler: GPUProfiler):
        self.gpu_profiler = gpu_profiler
        self._memory_history: List[Tuple[float, MemoryStats]] = []
        self._max_history_size = 1000
        self._lock = threading.RLock()
    
    def profile_system_memory(self) -> MemoryStats:
        """
        分析系统内存
        
        Returns:
            系统内存统计
        """
        stats = MemoryStats()
        
        try:
            # 读取/proc/meminfo (Linux)
            with open("/proc/meminfo", "r") as f:
                meminfo = f.read()
            
            total_match = re.search(r"MemTotal:\s+(\d+)\s+kB", meminfo)
            available_match = re.search(r"MemAvailable:\s+(\d+)\s+kB", meminfo)
            
            if total_match:
                stats.total_bytes = int(total_match.group(1)) * 1024
            if available_match:
                stats.free_bytes = int(available_match.group(1)) * 1024
            
            stats.used_bytes = stats.total_bytes - stats.free_bytes
        
        except (FileNotFoundError, Exception):
            # 非Linux系统使用模拟值
            stats.total_bytes = 32 * 1024 ** 3  # 32GB
            stats.free_bytes = 16 * 1024 ** 3   # 16GB
            stats.used_bytes = 16 * 1024 ** 3
        
        self._record_memory(stats)
        return stats
    
    def profile_gpu_memory(self, device_id: int = 0) -> MemoryStats:
        """
        分析GPU内存
        
        Args:
            device_id: GPU设备ID
            
        Returns:
            GPU内存统计
        """
        stats = self.gpu_profiler.get_gpu_memory(device_id)
        self._record_memory(stats)
        return stats
    
    def _record_memory(self, stats: MemoryStats) -> None:
        """记录内存历史"""
        with self._lock:
            self._memory_history.append((time.time(), stats))
            if len(self._memory_history) > self._max_history_size:
                self._memory_history.pop(0)
    
    def estimate_memory_for_model(
        self,
        model_params_billions: float,
        precision: PrecisionType = PrecisionType.FP16,
        batch_size: int = 1,
        sequence_length: int = 2048,
    ) -> float:
        """
        估算模型所需内存
        
        Args:
            model_params_billions: 模型参数量（十亿）
            precision: 精度类型
            batch_size: 批大小
            sequence_length: 序列长度
            
        Returns:
            所需内存（GB）
        """
        # 每个参数的比特数
        bits_per_param = {
            PrecisionType.FP32: 32,
            PrecisionType.FP16: 16,
            PrecisionType.BF16: 16,
            PrecisionType.INT8: 8,
            PrecisionType.INT4: 4,
        }.get(precision, 16)
        
        # 模型权重内存
        model_memory_gb = (model_params_billions * 1e9 * bits_per_param) / (8 * 1024 ** 3)
        
        # KV Cache内存 (简化估算)
        # 假设hidden_size=4096, num_layers=32, num_heads=32
        hidden_size = 4096
        num_layers = 32
        num_heads = 32
        head_dim = hidden_size // num_heads
        
        kv_cache_per_token = 2 * num_layers * num_heads * head_dim * bits_per_param / 8
        kv_cache_gb = batch_size * sequence_length * kv_cache_per_token / (1024 ** 3)
        
        # 激活内存 (粗略估算)
        activation_gb = batch_size * sequence_length * hidden_size * bits_per_param / (8 * 1024 ** 3)
        
        # 额外开销 (20%)
        overhead = 1.2
        
        total_gb = (model_memory_gb + kv_cache_gb + activation_gb) * overhead
        
        return total_gb
    
    def get_memory_trend(self, duration_seconds: float = 60.0) -> Dict[str, float]:
        """
        获取内存使用趋势
        
        Args:
            duration_seconds: 时间窗口（秒）
            
        Returns:
            趋势统计
        """
        with self._lock:
            cutoff_time = time.time() - duration_seconds
            recent = [(t, s) for t, s in self._memory_history if t >= cutoff_time]
            
            if not recent:
                return {"avg": 0.0, "min": 0.0, "max": 0.0, "trend": 0.0}
            
            utilizations = [s.utilization_percent for _, s in recent]
            
            return {
                "avg": sum(utilizations) / len(utilizations),
                "min": min(utilizations),
                "max": max(utilizations),
                "trend": utilizations[-1] - utilizations[0] if len(utilizations) > 1 else 0.0,
            }


class ComputeScorer:
    """
    计算能力评分器
    
    评估硬件的计算能力，给出综合评分和建议。
    """
    
    def __init__(self, gpu_profiler: GPUProfiler):
        self.gpu_profiler = gpu_profiler
    
    def score_gpu(self, gpu: GPUSpecs) -> ComputeScore:
        """
        评分单个GPU
        
        Args:
            gpu: GPU规格
            
        Returns:
            计算能力评分
        """
        # 计算性能评分 (0-100)
        compute_score = min(100, gpu.theoretical_tflops_fp32 / 2)
        
        # 内存评分
        memory_score = min(100, gpu.total_memory_gb * 2)
        
        # 带宽评分
        bandwidth_score = min(100, gpu.memory_bandwidth_gbps / 10)
        
        # 效率评分 (Tensor Core等)
        efficiency_score = 50
        if gpu.tensor_cores > 0:
            efficiency_score += 30
        if gpu.supports_bf16:
            efficiency_score += 10
        if gpu.supports_int8:
            efficiency_score += 10
        
        # 综合评分
        overall = (
            compute_score * 0.35 +
            memory_score * 0.35 +
            bandwidth_score * 0.15 +
            efficiency_score * 0.15
        )
        
        # 确定能力等级
        if overall >= 80:
            capability = ComputeCapability.VERY_HIGH
        elif overall >= 60:
            capability = ComputeCapability.HIGH
        elif overall >= 40:
            capability = ComputeCapability.MEDIUM
        else:
            capability = ComputeCapability.LOW
        
        return ComputeScore(
            overall_score=overall,
            compute_score=compute_score,
            memory_score=memory_score,
            bandwidth_score=bandwidth_score,
            efficiency_score=efficiency_score,
            capability_level=capability,
        )
    
    def score_system(self) -> Dict[int, ComputeScore]:
        """
        评分整个系统
        
        Returns:
            设备ID到评分的映射
        """
        gpus = self.gpu_profiler.detect_gpus()
        scores = {}
        
        for gpu in gpus:
            scores[gpu.device_id] = self.score_gpu(gpu)
        
        return scores


class BatchSizeEstimator:
    """
    批大小估计器
    
    根据硬件能力和模型需求估计最优批大小。
    """
    
    def __init__(
        self,
        gpu_profiler: GPUProfiler,
        memory_profiler: MemoryProfiler,
    ):
        self.gpu_profiler = gpu_profiler
        self.memory_profiler = memory_profiler
    
    def estimate(
        self,
        model_params_billions: float,
        precision: PrecisionType = PrecisionType.FP16,
        sequence_length: int = 2048,
        target_latency_ms: Optional[float] = None,
    ) -> BatchSizeEstimate:
        """
        估计最优批大小
        
        Args:
            model_params_billions: 模型参数量
            precision: 精度类型
            sequence_length: 序列长度
            target_latency_ms: 目标延迟
            
        Returns:
            批大小估计
        """
        gpus = self.gpu_profiler.detect_gpus()
        if not gpus:
            return BatchSizeEstimate()
        
        gpu = gpus[0]
        memory_stats = self.gpu_profiler.get_gpu_memory(gpu.device_id)
        
        # 可用内存 (保留20%安全余量)
        available_memory_gb = memory_stats.free_gb * 0.8
        
        # 单个样本内存需求
        single_sample_memory = self.memory_profiler.estimate_memory_for_model(
            model_params_billions,
            precision,
            batch_size=1,
            sequence_length=sequence_length,
        )
        
        # 内存限制的批大小
        memory_limited = max(1, int(available_memory_gb / single_sample_memory))
        
        # 计算限制的批大小 (基于TFLOPS)
        compute_limited = self._estimate_compute_limited_batch(
            gpu, model_params_billions, sequence_length
        )
        
        # 最优批大小
        optimal = min(memory_limited, compute_limited)
        
        # 如果指定了目标延迟，调整批大小
        if target_latency_ms is not None:
            latency_limited = self._estimate_latency_limited_batch(
                gpu, model_params_billions, sequence_length, target_latency_ms
            )
            optimal = min(optimal, latency_limited)
        
        # 估计延迟和吞吐量
        latency_ms = self._estimate_latency(gpu, model_params_billions, optimal, sequence_length)
        throughput = (optimal * sequence_length) / (latency_ms / 1000)
        
        return BatchSizeEstimate(
            optimal_batch_size=optimal,
            max_batch_size=memory_limited,
            memory_limited_batch_size=memory_limited,
            compute_limited_batch_size=compute_limited,
            latency_ms_estimate=latency_ms,
            throughput_tok_per_sec=throughput,
            memory_usage_gb=single_sample_memory * optimal,
        )
    
    def _estimate_compute_limited_batch(
        self,
        gpu: GPUSpecs,
        model_params_billions: float,
        sequence_length: int,
    ) -> int:
        """估计计算限制的批大小"""
        # 简化的FLOPs估算
        # Transformer: ~2 * params * tokens per forward pass
        flops_per_token = 2 * model_params_billions * 1e9
        total_flops = flops_per_token * sequence_length
        
        # 考虑GPU计算能力
        tflops = gpu.theoretical_tflops_fp16 if gpu.supports_fp16 else gpu.theoretical_tflops_fp32
        
        # 假设50%效率，目标100ms per batch
        target_time_s = 0.1
        achievable_flops = tflops * 1e12 * 0.5 * target_time_s
        
        return max(1, int(achievable_flops / total_flops))
    
    def _estimate_latency_limited_batch(
        self,
        gpu: GPUSpecs,
        model_params_billions: float,
        sequence_length: int,
        target_latency_ms: float,
    ) -> int:
        """估计延迟限制的批大小"""
        # 简化的延迟模型
        base_latency_ms = 10  # 基础开销
        per_sample_ms = self._estimate_per_sample_latency(
            gpu, model_params_billions, sequence_length
        )
        
        if per_sample_ms <= 0:
            return 1
        
        return max(1, int((target_latency_ms - base_latency_ms) / per_sample_ms))
    
    def _estimate_per_sample_latency(
        self,
        gpu: GPUSpecs,
        model_params_billions: float,
        sequence_length: int,
    ) -> float:
        """估计每个样本的延迟"""
        flops = 2 * model_params_billions * 1e9 * sequence_length
        tflops = gpu.theoretical_tflops_fp16 if gpu.supports_fp16 else gpu.theoretical_tflops_fp32
        
        # 假设30%效率
        seconds = flops / (tflops * 1e12 * 0.3)
        return seconds * 1000  # 转换为ms
    
    def _estimate_latency(
        self,
        gpu: GPUSpecs,
        model_params_billions: float,
        batch_size: int,
        sequence_length: int,
    ) -> float:
        """估计延迟"""
        per_sample = self._estimate_per_sample_latency(gpu, model_params_billions, sequence_length)
        # 批处理有一定效率提升
        batch_efficiency = 0.7 + 0.3 / batch_size
        return 10 + per_sample * batch_size * batch_efficiency


class PrecisionAdvisor:
    """
    精度建议器
    
    根据硬件能力和模型需求推荐最佳精度设置。
    """
    
    def __init__(self, gpu_profiler: GPUProfiler):
        self.gpu_profiler = gpu_profiler
    
    def recommend(
        self,
        model_params_billions: float,
        target_quality: str = "high",
        memory_constraint_gb: Optional[float] = None,
    ) -> PrecisionRecommendation:
        """
        推荐精度设置
        
        Args:
            model_params_billions: 模型参数量
            target_quality: 目标质量 (high/medium/low)
            memory_constraint_gb: 内存限制
            
        Returns:
            精度推荐
        """
        gpus = self.gpu_profiler.detect_gpus()
        if not gpus:
            return PrecisionRecommendation(
                recommended_precision=PrecisionType.FP32,
                available_precisions=[PrecisionType.FP32],
                expected_speedup=1.0,
                expected_memory_reduction=1.0,
                quality_impact="none",
                notes="No GPU detected, using FP32 fallback",
            )
        
        gpu = gpus[0]
        available = self._get_available_precisions(gpu)
        
        # 根据约束选择精度
        if target_quality == "high":
            if PrecisionType.BF16 in available:
                recommended = PrecisionType.BF16
            elif PrecisionType.FP16 in available:
                recommended = PrecisionType.FP16
            else:
                recommended = PrecisionType.FP32
        elif target_quality == "medium":
            if PrecisionType.FP16 in available:
                recommended = PrecisionType.FP16
            elif PrecisionType.BF16 in available:
                recommended = PrecisionType.BF16
            else:
                recommended = PrecisionType.FP32
        else:  # low
            if PrecisionType.INT8 in available:
                recommended = PrecisionType.INT8
            elif PrecisionType.FP16 in available:
                recommended = PrecisionType.FP16
            else:
                recommended = PrecisionType.FP32
        
        # 检查内存约束
        if memory_constraint_gb is not None:
            model_memory_fp32 = (model_params_billions * 1e9 * 32) / (8 * 1024 ** 3)
            
            if model_memory_fp32 > memory_constraint_gb and PrecisionType.INT8 in available:
                recommended = PrecisionType.INT8
            elif model_memory_fp32 * 0.5 > memory_constraint_gb and PrecisionType.INT4 in available:
                recommended = PrecisionType.INT4
        
        # 计算预期收益
        speedup = self._estimate_speedup(recommended, gpu)
        memory_reduction = self._estimate_memory_reduction(recommended)
        quality = self._estimate_quality_impact(recommended)
        
        notes = self._generate_notes(recommended, gpu)
        
        return PrecisionRecommendation(
            recommended_precision=recommended,
            available_precisions=available,
            expected_speedup=speedup,
            expected_memory_reduction=memory_reduction,
            quality_impact=quality,
            notes=notes,
        )
    
    def _get_available_precisions(self, gpu: GPUSpecs) -> List[PrecisionType]:
        """获取可用精度列表"""
        available = [PrecisionType.FP32]
        
        if gpu.supports_fp16:
            available.append(PrecisionType.FP16)
        if gpu.supports_bf16:
            available.append(PrecisionType.BF16)
        if gpu.supports_int8:
            available.append(PrecisionType.INT8)
        if gpu.supports_int4:
            available.append(PrecisionType.INT4)
        
        return available
    
    def _estimate_speedup(self, precision: PrecisionType, gpu: GPUSpecs) -> float:
        """估计加速比"""
        speedups = {
            PrecisionType.FP32: 1.0,
            PrecisionType.FP16: 2.0 if gpu.tensor_cores > 0 else 1.5,
            PrecisionType.BF16: 2.0 if gpu.tensor_cores > 0 else 1.5,
            PrecisionType.INT8: 3.0 if gpu.tensor_cores > 0 else 2.0,
            PrecisionType.INT4: 4.0,
        }
        return speedups.get(precision, 1.0)
    
    def _estimate_memory_reduction(self, precision: PrecisionType) -> float:
        """估计内存减少比例"""
        reductions = {
            PrecisionType.FP32: 1.0,
            PrecisionType.FP16: 2.0,
            PrecisionType.BF16: 2.0,
            PrecisionType.INT8: 4.0,
            PrecisionType.INT4: 8.0,
        }
        return reductions.get(precision, 1.0)
    
    def _estimate_quality_impact(self, precision: PrecisionType) -> str:
        """估计质量影响"""
        impacts = {
            PrecisionType.FP32: "none",
            PrecisionType.FP16: "minimal",
            PrecisionType.BF16: "minimal",
            PrecisionType.INT8: "low",
            PrecisionType.INT4: "moderate",
        }
        return impacts.get(precision, "unknown")
    
    def _generate_notes(self, precision: PrecisionType, gpu: GPUSpecs) -> str:
        """生成建议说明"""
        notes = []
        
        if precision == PrecisionType.FP16 and not gpu.supports_fp16:
            notes.append("FP16 not natively supported, may use emulation")
        
        if precision == PrecisionType.BF16 and gpu.vendor == GPUVendor.NVIDIA:
            notes.append("BF16 requires Ampere (SM80+) or newer architecture")
        
        if precision in (PrecisionType.INT8, PrecisionType.INT4):
            notes.append("Quantization may require calibration for best results")
        
        return "; ".join(notes) if notes else "Recommended precision based on hardware capabilities"


class HardwareProfiler:
    """
    硬件能力分析器
    
    整合GPU分析、内存分析、计算评分、批大小估计和精度建议，
    提供全面的硬件能力分析和优化建议。
    """
    
    def __init__(self):
        self.gpu_profiler = GPUProfiler()
        self.memory_profiler = MemoryProfiler(self.gpu_profiler)
        self.compute_scorer = ComputeScorer(self.gpu_profiler)
        self.batch_estimator = BatchSizeEstimator(self.gpu_profiler, self.memory_profiler)
        self.precision_advisor = PrecisionAdvisor(self.gpu_profiler)
        
        self._profile_cache: Dict[str, Any] = {}
        self._cache_lock = threading.RLock()
    
    def profile_all(self) -> Dict[str, Any]:
        """
        全面分析硬件能力
        
        Returns:
            完整的硬件分析报告
        """
        # GPU检测
        gpus = self.gpu_profiler.detect_gpus()
        
        # 系统内存
        system_memory = self.memory_profiler.profile_system_memory()
        
        # GPU内存
        gpu_memories = {}
        for gpu in gpus:
            gpu_memories[gpu.device_id] = self.memory_profiler.profile_gpu_memory(gpu.device_id)
        
        # 计算评分
        compute_scores = self.compute_scorer.score_system()
        
        # 构建报告
        report = {
            "gpus": [
                {
                    "specs": gpu,
                    "memory": gpu_memories.get(gpu.device_id),
                    "score": compute_scores.get(gpu.device_id),
                }
                for gpu in gpus
            ],
            "system_memory": system_memory,
            "detected_vendor": self.gpu_profiler._detected_vendor.value if self.gpu_profiler._detected_vendor else "unknown",
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
        }
        
        with self._cache_lock:
            self._profile_cache = report
        
        return report
    
    def get_optimal_config(
        self,
        model_params_billions: float,
        target_quality: str = "high",
    ) -> Dict[str, Any]:
        """
        获取最优配置建议
        
        Args:
            model_params_billions: 模型参数量
            target_quality: 目标质量
            
        Returns:
            配置建议
        """
        gpus = self.gpu_profiler.detect_gpus()
        if not gpus:
            return {"error": "No GPU detected"}
        
        gpu = gpus[0]
        
        # 精度建议
        precision_rec = self.precision_advisor.recommend(
            model_params_billions,
            target_quality,
            memory_constraint_gb=gpu.total_memory_gb * 0.8,
        )
        
        # 批大小估计
        batch_estimate = self.batch_estimator.estimate(
            model_params_billions,
            precision_rec.recommended_precision,
        )
        
        # 计算评分
        compute_score = self.compute_scorer.score_gpu(gpu)
        
        return {
            "gpu": gpu,
            "precision": precision_rec,
            "batch_size": batch_estimate,
            "compute_score": compute_score,
            "recommendations": self._generate_recommendations(
                gpu, precision_rec, batch_estimate, compute_score
            ),
        }
    
    def _generate_recommendations(
        self,
        gpu: GPUSpecs,
        precision: PrecisionRecommendation,
        batch: BatchSizeEstimate,
        score: ComputeScore,
    ) -> List[str]:
        """生成配置建议"""
        recommendations = []
        
        if score.overall_score < 50:
            recommendations.append("Consider using a smaller model or upgrading hardware")
        
        if precision.recommended_precision == PrecisionType.INT8:
            recommendations.append("INT8 quantization recommended for memory-constrained scenarios")
        
        if batch.optimal_batch_size < 4:
            recommendations.append("Small batch size may limit throughput; consider gradient accumulation")
        
        if gpu.tensor_cores > 0 and precision.recommended_precision == PrecisionType.FP32:
            recommendations.append("Tensor Cores available but not utilized; consider FP16/BF16")
        
        if not recommendations:
            recommendations.append("Hardware configuration is optimal for the selected model")
        
        return recommendations
    
    def get_gpu_count(self) -> int:
        """获取GPU数量"""
        return len(self.gpu_profiler.detect_gpus())
    
    def get_total_gpu_memory_gb(self) -> float:
        """获取总GPU内存"""
        gpus = self.gpu_profiler.detect_gpus()
        return sum(gpu.total_memory_gb for gpu in gpus)
    
    def is_cuda_available(self) -> bool:
        """检查CUDA是否可用"""
        gpus = self.gpu_profiler.detect_gpus()
        return any(gpu.vendor == GPUVendor.NVIDIA for gpu in gpus)
    
    def is_mps_available(self) -> bool:
        """检查MPS是否可用"""
        gpus = self.gpu_profiler.detect_gpus()
        return any(gpu.vendor == GPUVendor.APPLE for gpu in gpus)
    
    def get_hardware_summary(self) -> str:
        """获取硬件摘要"""
        report = self.profile_all()
        
        lines = [
            "Hardware Profile Summary",
            "=" * 50,
            f"Platform: {report['platform']['system']} ({report['platform']['machine']})",
            f"System Memory: {report['system_memory'].total_gb:.1f} GB",
            f"GPUs Detected: {len(report['gpus'])}",
        ]
        
        for gpu_info in report['gpus']:
            gpu = gpu_info['specs']
            score = gpu_info['score']
            lines.extend([
                f"\nGPU {gpu.device_id}: {gpu.name}",
                f"  Memory: {gpu.total_memory_gb:.1f} GB",
                f"  Compute Score: {score.overall_score:.1f}/100",
                f"  Capability: {score.capability_level.value}",
            ])
        
        return "\n".join(lines)
