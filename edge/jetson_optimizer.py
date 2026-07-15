"""
Jetson Platform Optimization Module

Provides GPU clock management, power mode control, shared memory (NVSHMEM),
DeepStream pipeline, thermal management, and performance profiling for
NVIDIA Jetson platforms.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

try:
    import ctypes as _ctypes
except ImportError:
    _ctypes = None  # type: ignore

try:
    import numpy as _np
except ImportError:
    _np = None  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JETSON_CLOCK_PATHS: Dict[str, str] = {
    "gpu_freq": "/sys/devices/gpu.0/devfreq/57000000.gpu/cur_freq",
    "gpu_min_freq": "/sys/devices/gpu.0/devfreq/57000000.gpu/min_freq",
    "gpu_max_freq": "/sys/devices/gpu.0/devfreq/57000000.gpu/max_freq",
    "gpu_available_freqs": "/sys/devices/gpu.0/devfreq/57000000.gpu/available_frequencies",
    "cpu_freq": "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
    "cpu_max_freq": "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq",
    "emc_freq": "/sys/kernel/debug/tegra_fansink/thermal_zone/emc_cur_freq",
    "thermal_zone0": "/sys/class/thermal/thermal_zone0/temp",
    "thermal_zone1": "/sys/class/thermal/thermal_zone1/temp",
    "power_state": "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
}

NVSHMEM_DEFAULT_SIZE = 64 * 1024 * 1024  # 64MB
DEEPSTREAM_DEFAULT_BATCH_SIZE = 16
THERMAL_WARNING_THRESHOLD = 75.0
THERMAL_CRITICAL_THRESHOLD = 85.0
THERMAL_SHUTDOWN_THRESHOLD = 95.0


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PowerMode(IntEnum):
    MAXN_0 = 0
    MAXN_1 = 1
    MAXN_2 = 2
    MAXQ_0 = 3
    MAXQ_1 = 4
    MAXQ_2 = 5
    MINN_0 = 6
    MINN_1 = 7
    MINN_2 = 8


class PowerProfile(Enum):
    MAX_PERFORMANCE = "MAX_PERFORMANCE"
    HIGH_PERFORMANCE = "HIGH_PERFORMANCE"
    MEDIUM_PERFORMANCE = "MEDIUM_PERFORMANCE"
    LOW_POWER = "LOW_POWER"
    MIN_POWER = "MIN_POWER"


class ThermalState(Enum):
    NORMAL = "normal"
    WARM = "warm"
    HOT = "hot"
    CRITICAL = "critical"
    SHUTDOWN = "shutdown"


class PipelineState(Enum):
    CREATED = "created"
    INITIALIZED = "initialized"
    PLAYING = "playing"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class GPUFreqDomain(Enum):
    GPU = "gpu"
    EMC = "emc"
    CPU = "cpu"
    DLA = "dla"
    PVA = "pva"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class GPUFreqInfo:
    """GPU frequency information."""
    current_mhz: float = 0.0
    min_mhz: float = 0.0
    max_mhz: float = 0.0
    available_freqs: List[float] = field(default_factory=list)
    domain: GPUFreqDomain = GPUFreqDomain.GPU


@dataclass
class ThermalReading:
    """Thermal sensor reading."""
    zone: str = ""
    temperature_c: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    state: ThermalState = ThermalState.NORMAL


@dataclass
class PowerConsumption:
    """Power consumption reading."""
    total_mw: float = 0.0
    gpu_mw: float = 0.0
    cpu_mw: float = 0.0
    soc_mw: float = 0.0
    ddr_mw: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MemoryInfo:
    """Shared memory information."""
    total_bytes: int = 0
    used_bytes: int = 0
    free_bytes: int = 0
    heap_count: int = 0
    allocation_count: int = 0
    fragmentation_ratio: float = 0.0


@dataclass
class PipelineConfig:
    """DeepStream pipeline configuration."""
    name: str = "default"
    batch_size: int = DEEPSTREAM_DEFAULT_BATCH_SIZE
    num_inferences: int = 4
    inference_backend: str = "nvinfer"
    input_source: str = "filesrc"
    output_sink: str = "fakesink"
    model_config: str = ""
    stream_width: int = 1920
    stream_height: int = 1080
    fps: int = 30
    tracker: str = "nvdcf"
    preprocess_config: str = ""
    postprocess_config: str = ""


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics."""
    pipeline_name: str = ""
    fps: float = 0.0
    latency_ms: float = 0.0
    source_fps: float = 0.0
    inference_latency_ms: float = 0.0
    preprocess_latency_ms: float = 0.0
    postprocess_latency_ms: float = 0.0
    frames_processed: int = 0
    frames_dropped: int = 0
    gpu_utilization: float = 0.0
    memory_used_mb: float = 0.0
    state: PipelineState = PipelineState.CREATED


@dataclass
class ProfileSample:
    """A single profiling sample."""
    timestamp: float = 0.0
    gpu_freq_mhz: float = 0.0
    cpu_freq_mhz: float = 0.0
    temperature_c: float = 0.0
    power_mw: float = 0.0
    gpu_utilization: float = 0.0
    memory_used_mb: float = 0.0


@dataclass
class PlatformInfo:
    """Jetson platform information."""
    model: str = "Unknown"
    jetpack_version: str = ""
    l4t_version: str = ""
    cuda_version: str = ""
    tensorrt_version: str = ""
    cuda_cores: int = 0
    gpu_arch: str = ""
    total_ram_mb: int = 0
    shared_memory_mb: int = 0
    serial_number: str = ""


# ---------------------------------------------------------------------------
# GPU Clock Manager
# ---------------------------------------------------------------------------

class GPUClockManager:
    """Manages GPU clock frequencies on Jetson platforms."""

    def __init__(self, simulated: bool = True) -> None:
        self.simulated = simulated
        self._freqs: Dict[GPUFreqDomain, GPUFreqInfo] = {
            GPUFreqDomain.GPU: GPUFreqInfo(
                current_mhz=921.0, min_mhz=114.0, max_mhz=1300.0,
                available_freqs=[114, 318, 510, 624, 756, 852, 921, 1053, 1140, 1300],
            ),
            GPUFreqDomain.EMC: GPUFreqInfo(
                current_mhz=1600.0, min_mhz=204.0, max_mhz=3200.0,
                available_freqs=[204, 408, 665, 800, 1066, 1333, 1600, 1866, 2133, 2400, 2666, 2933, 3200],
            ),
            GPUFreqDomain.CPU: GPUFreqInfo(
                current_mhz=1428.0, min_mhz=204.0, max_mhz=1908.0,
                available_freqs=[204, 345, 510, 684, 816, 960, 1200, 1428, 1608, 1908],
            ),
        }
        self._lock = threading.Lock()

    def get_frequency(self, domain: GPUFreqDomain = GPUFreqDomain.GPU) -> GPUFreqInfo:
        if self.simulated:
            with self._lock:
                return self._freqs.get(domain, GPUFreqInfo(domain=domain))
        return self._read_sysfs(domain)

    def set_frequency(self, domain: GPUFreqDomain, freq_mhz: float) -> bool:
        info = self._freqs.get(domain)
        if info is None:
            return False
        if freq_mhz < info.min_mhz or freq_mhz > info.max_mhz:
            return False
        if self.simulated:
            with self._lock:
                closest = min(info.available_freqs, key=lambda f: abs(f - freq_mhz))
                info.current_mhz = float(closest)
            return True
        return self._write_sysfs(domain, freq_mhz)

    def set_min_frequency(self, domain: GPUFreqDomain, freq_mhz: float) -> bool:
        info = self._freqs.get(domain)
        if info:
            info.min_mhz = freq_mhz
            return True
        return False

    def set_max_frequency(self, domain: GPUFreqDomain, freq_mhz: float) -> bool:
        info = self._freqs.get(domain)
        if info:
            info.max_mhz = freq_mhz
            return True
        return False

    def set_range(
        self, domain: GPUFreqDomain, min_mhz: float, max_mhz: float
    ) -> bool:
        info = self._freqs.get(domain)
        if info is None:
            return False
        info.min_mhz = min_mhz
        info.max_mhz = max_mhz
        return True

    def boost_gpu(self, target_mhz: Optional[float] = None) -> bool:
        info = self._freqs.get(GPUFreqDomain.GPU)
        if info is None:
            return False
        target = target_mhz or info.max_mhz
        return self.set_frequency(GPUFreqDomain.GPU, target)

    def throttle_gpu(self, target_mhz: Optional[float] = None) -> bool:
        info = self._freqs.get(GPUFreqDomain.GPU)
        if info is None:
            return False
        target = target_mhz or info.min_mhz
        return self.set_frequency(GPUFreqDomain.GPU, target)

    def _read_sysfs(self, domain: GPUFreqDomain) -> GPUFreqInfo:
        return GPUFreqInfo(domain=domain)

    def _write_sysfs(self, domain: GPUFreqDomain, freq_mhz: float) -> bool:
        return True

    def get_all_frequencies(self) -> Dict[str, GPUFreqInfo]:
        return {d.value: self.get_frequency(d) for d in GPUFreqDomain}


# ---------------------------------------------------------------------------
# Power Mode Controller
# ---------------------------------------------------------------------------

class PowerModeController:
    """Controls Jetson power modes and profiles."""

    POWER_MODE_PROFILES: Dict[PowerProfile, Dict[str, Any]] = {
        PowerProfile.MAX_PERFORMANCE: {
            "power_mode": PowerMode.MAXN_0,
            "gpu_freq": "max",
            "cpu_freq": "max",
            "emc_freq": "max",
            "fan_speed": 100,
        },
        PowerProfile.HIGH_PERFORMANCE: {
            "power_mode": PowerMode.MAXN_1,
            "gpu_freq": "high",
            "cpu_freq": "high",
            "emc_freq": "high",
            "fan_speed": 80,
        },
        PowerProfile.MEDIUM_PERFORMANCE: {
            "power_mode": PowerMode.MAXQ_0,
            "gpu_freq": "medium",
            "cpu_freq": "medium",
            "emc_freq": "medium",
            "fan_speed": 50,
        },
        PowerProfile.LOW_POWER: {
            "power_mode": PowerMode.MAXQ_1,
            "gpu_freq": "low",
            "cpu_freq": "low",
            "emc_freq": "low",
            "fan_speed": 30,
        },
        PowerProfile.MIN_POWER: {
            "power_mode": PowerMode.MINN_0,
            "gpu_freq": "min",
            "cpu_freq": "min",
            "emc_freq": "min",
            "fan_speed": 0,
        },
    }

    def __init__(
        self,
        gpu_manager: GPUClockManager,
        thermal_manager: Optional[Any] = None,
    ) -> None:
        self.gpu_manager = gpu_manager
        self.thermal_manager = thermal_manager
        self._current_mode: PowerMode = PowerMode.MAXN_0
        self._current_profile: PowerProfile = PowerProfile.MAX_PERFORMANCE
        self._fan_speed: int = 50
        self._lock = threading.Lock()

    def set_power_mode(self, mode: PowerMode) -> bool:
        with self._lock:
            self._current_mode = mode
        logger.info("Power mode set to %d", mode)
        return True

    def set_profile(self, profile: PowerProfile) -> bool:
        config = self.POWER_MODE_PROFILES.get(profile)
        if config is None:
            return False

        self.set_power_mode(config["power_mode"])

        gpu_target = config["gpu_freq"]
        if gpu_target == "max":
            self.gpu_manager.boost_gpu()
        elif gpu_target == "min":
            self.gpu_manager.throttle_gpu()
        elif gpu_target == "high":
            info = self.gpu_manager.get_frequency(GPUFreqDomain.GPU)
            target = info.min_mhz + (info.max_mhz - info.min_mhz) * 0.75
            self.gpu_manager.set_frequency(GPUFreqDomain.GPU, target)
        elif gpu_target == "medium":
            info = self.gpu_manager.get_frequency(GPUFreqDomain.GPU)
            target = info.min_mhz + (info.max_mhz - info.min_mhz) * 0.5
            self.gpu_manager.set_frequency(GPUFreqDomain.GPU, target)
        elif gpu_target == "low":
            info = self.gpu_manager.get_frequency(GPUFreqDomain.GPU)
            target = info.min_mhz + (info.max_mhz - info.min_mhz) * 0.25
            self.gpu_manager.set_frequency(GPUFreqDomain.GPU, target)

        self._fan_speed = config["fan_speed"]
        self._current_profile = profile
        logger.info("Power profile set to %s", profile.value)
        return True

    def get_power_mode(self) -> PowerMode:
        return self._current_mode

    def get_profile(self) -> PowerProfile:
        return self._current_profile

    def set_fan_speed(self, speed: int) -> bool:
        self._fan_speed = max(0, min(100, speed))
        return True

    def get_fan_speed(self) -> int:
        return self._fan_speed

    def estimate_power_consumption(self) -> PowerConsumption:
        gpu_info = self.gpu_manager.get_frequency(GPUFreqDomain.GPU)
        gpu_ratio = gpu_info.current_mhz / gpu_info.max_mhz if gpu_info.max_mhz > 0 else 0
        total = 5000 + gpu_ratio * 10000  # 5W base + up to 10W GPU
        return PowerConsumption(
            total_mw=total,
            gpu_mw=gpu_ratio * 10000,
            cpu_mw=2000 + gpu_ratio * 3000,
            soc_mw=1500,
            ddr_mw=500 + gpu_ratio * 2000,
        )


# ---------------------------------------------------------------------------
# Shared Memory Manager
# ---------------------------------------------------------------------------

class SharedMemoryManager:
    """Manages NVSHMEM/shared memory for multi-GPU communication."""

    def __init__(self, default_size: int = NVSHMEM_DEFAULT_SIZE) -> None:
        self.default_size = default_size
        self._heaps: Dict[str, Dict[str, Any]] = {}
        self._allocations: Dict[str, Dict[str, Any]] = {}
        self._total_allocated: int = 0
        self._lock = threading.Lock()

    def create_heap(
        self,
        name: str,
        size: Optional[int] = None,
        alignment: int = 4096,
    ) -> str:
        heap_size = size or self.default_size
        heap_id = f"heap_{name}_{id(self)}"
        with self._lock:
            self._heaps[heap_id] = {
                "name": name,
                "size": heap_size,
                "alignment": alignment,
                "used": 0,
                "created_at": datetime.utcnow().isoformat(),
            }
        logger.info("Created shared memory heap '%s' (size=%d bytes)", name, heap_size)
        return heap_id

    def allocate(
        self,
        heap_id: str,
        size: int,
        alignment: int = 64,
    ) -> Optional[str]:
        with self._lock:
            heap = self._heaps.get(heap_id)
            if heap is None:
                return None
            aligned_size = (size + alignment - 1) // alignment * alignment
            if heap["used"] + aligned_size > heap["size"]:
                return None

            alloc_id = f"alloc_{len(self._allocations)}_{int(time.time() * 1000)}"
            offset = heap["used"]
            heap["used"] += aligned_size
            self._allocations[alloc_id] = {
                "heap_id": heap_id,
                "offset": offset,
                "size": aligned_size,
                "alignment": alignment,
                "heap_name": heap["name"],
            }
            self._total_allocated += aligned_size
            return alloc_id

    def deallocate(self, alloc_id: str) -> bool:
        with self._lock:
            alloc = self._allocations.pop(alloc_id, None)
            if alloc is None:
                return False
            heap = self._heaps.get(alloc["heap_id"])
            if heap:
                heap["used"] -= alloc["size"]
                self._total_allocated -= alloc["size"]
            return True

    def get_heap_info(self, heap_id: str) -> Optional[MemoryInfo]:
        heap = self._heaps.get(heap_id)
        if heap is None:
            return None
        return MemoryInfo(
            total_bytes=heap["size"],
            used_bytes=heap["used"],
            free_bytes=heap["size"] - heap["used"],
            heap_count=len(self._heaps),
            allocation_count=len(self._allocations),
            fragmentation_ratio=self._calculate_fragmentation(heap),
        )

    def _calculate_fragmentation(self, heap: Dict[str, Any]) -> float:
        if heap["size"] == 0:
            return 0.0
        return 1.0 - (heap["used"] / heap["size"])

    def get_total_info(self) -> MemoryInfo:
        total_size = sum(h["size"] for h in self._heaps.values())
        total_used = sum(h["used"] for h in self._heaps.values())
        return MemoryInfo(
            total_bytes=total_size,
            used_bytes=total_used,
            free_bytes=total_size - total_used,
            heap_count=len(self._heaps),
            allocation_count=len(self._allocations),
            fragmentation_ratio=1.0 - (total_used / total_size) if total_size > 0 else 0.0,
        )

    def destroy_heap(self, heap_id: str) -> bool:
        with self._lock:
            allocs_to_remove = [
                aid for aid, a in self._allocations.items()
                if a["heap_id"] == heap_id
            ]
            for aid in allocs_to_remove:
                del self._allocations[aid]
            return self._heaps.pop(heap_id, None) is not None

    def list_heaps(self) -> List[Dict[str, Any]]:
        return [
            {"id": hid, **info}
            for hid, info in self._heaps.items()
        ]


# ---------------------------------------------------------------------------
# DeepStream Pipeline
# ---------------------------------------------------------------------------

class DeepStreamPipeline:
    """Simulates a DeepStream inference pipeline."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self._state: PipelineState = PipelineState.CREATED
        self._metrics: PipelineMetrics = PipelineMetrics(pipeline_name=self.config.name)
        self._frame_count: int = 0
        self._start_time: Optional[float] = None
        self._lock = threading.Lock()
        self._pipeline_thread: Optional[threading.Thread] = None
        self._running = False

    def initialize(self) -> bool:
        self._state = PipelineState.INITIALIZED
        self._metrics.state = PipelineState.INITIALIZED
        logger.info("Pipeline '%s' initialized", self.config.name)
        return True

    def build_pipeline_string(self) -> str:
        """Build the GStreamer pipeline string."""
        parts: List[str] = []

        # Input source
        if self.config.input_source == "filesrc":
            parts.append("filesrc location=input.mp4 ! qtdemux ! h264parse ! nvv4l2decoder")
        elif self.config.input_source == "v4l2src":
            parts.append("nvarguscamerasrc ! nvvidconv")
        else:
            parts.append(f"{self.config.input_source}")

        # Stream muxer
        parts.append(f"nvstreammux batch-size={self.config.batch_size} width={self.config.stream_width} height={self.config.stream_height}")

        # Preprocessing
        parts.append("nvvideoconvert ! nvstreamsync")

        # Primary inference
        parts.append(f"nvdsinfer config-file-path={self.config.model_config} batch-size={self.config.batch_size}")

        # Tracker
        if self.config.tracker:
            parts.append(f"nvtracker tracker-width={self.config.stream_width} tracker-height={self.config.stream_height} ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")

        # Post-processing
        parts.append("nvdsanalytics")

        # Output
        if self.config.output_sink == "fakesink":
            parts.append("fakesink sync=false")
        elif self.config.output_sink == "filesink":
            parts.append("nvvideoconvert ! nvjpegenc ! filesink location=output.jpg")
        elif self.config.output_sink == "nveglglessink":
            parts.append("nveglglessink")
        else:
            parts.append(self.config.output_sink)

        return " ! ".join(parts)

    def play(self) -> bool:
        if self._state != PipelineState.INITIALIZED:
            return False
        self._state = PipelineState.PLAYING
        self._metrics.state = PipelineState.PLAYING
        self._start_time = time.time()
        self._running = True
        self._pipeline_thread = threading.Thread(
            target=self._run_loop, daemon=True, name="deepstream-pipeline"
        )
        self._pipeline_thread.start()
        logger.info("Pipeline '%s' playing", self.config.name)
        return True

    def pause(self) -> bool:
        self._state = PipelineState.PAUSED
        self._metrics.state = PipelineState.PAUSED
        self._running = False
        return True

    def stop(self) -> bool:
        self._running = False
        if self._pipeline_thread:
            self._pipeline_thread.join(timeout=5.0)
        self._state = PipelineState.STOPPED
        self._metrics.state = PipelineState.STOPPED
        return True

    def _run_loop(self) -> None:
        while self._running:
            self._simulate_frame()
            time.sleep(1.0 / max(self.config.fps, 1))

    def _simulate_frame(self) -> None:
        with self._lock:
            self._frame_count += 1
            self._metrics.frames_processed = self._frame_count

            elapsed = time.time() - self._start_time if self._start_time else 1.0
            self._metrics.fps = self._frame_count / elapsed if elapsed > 0 else 0
            self._metrics.source_fps = self._metrics.fps * random.uniform(0.95, 1.0) if _np else self._metrics.fps
            self._metrics.inference_latency_ms = 1000.0 / max(self.config.fps, 1) * self.config.num_inferences * 0.3
            self._metrics.preprocess_latency_ms = self._metrics.inference_latency_ms * 0.2
            self._metrics.postprocess_latency_ms = self._metrics.inference_latency_ms * 0.15
            self._metrics.latency_ms = (
                self._metrics.preprocess_latency_ms +
                self._metrics.inference_latency_ms +
                self._metrics.postprocess_latency_ms
            )
            self._metrics.gpu_utilization = min(100.0, self._metrics.fps * 0.5)
            self._metrics.memory_used_mb = 200 + self._frame_count * 0.01

    def get_metrics(self) -> PipelineMetrics:
        with self._lock:
            return PipelineMetrics(
                pipeline_name=self._metrics.pipeline_name,
                fps=self._metrics.fps,
                latency_ms=self._metrics.latency_ms,
                source_fps=self._metrics.source_fps,
                inference_latency_ms=self._metrics.inference_latency_ms,
                preprocess_latency_ms=self._metrics.preprocess_latency_ms,
                postprocess_latency_ms=self._metrics.postprocess_latency_ms,
                frames_processed=self._metrics.frames_processed,
                frames_dropped=self._metrics.frames_dropped,
                gpu_utilization=self._metrics.gpu_utilization,
                memory_used_mb=self._metrics.memory_used_mb,
                state=self._state,
            )

    @property
    def state(self) -> PipelineState:
        return self._state


# ---------------------------------------------------------------------------
# Thermal Manager
# ---------------------------------------------------------------------------

class ThermalManager:
    """Manages Jetson thermal state and cooling."""

    def __init__(self) -> None:
        self._readings: List[ThermalReading] = []
        self._current_temp: float = 45.0
        self._lock = threading.Lock()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[ThermalState, float], None]] = []
        self._fan_speed: int = 50
        self._throttling_enabled = True

    def get_temperature(self, zone: str = "thermal_zone0") -> float:
        if zone == "thermal_zone0":
            return self._current_temp + random.uniform(-1, 1) if _np else self._current_temp
        return self._current_temp + 5.0

    def get_all_temperatures(self) -> List[ThermalReading]:
        zones = ["thermal_zone0", "thermal_zone1", "GPU", "CPU", "SOC"]
        return [
            ThermalReading(zone=z, temperature_c=self.get_temperature(z))
            for z in zones
        ]

    def get_thermal_state(self) -> ThermalState:
        temp = self.get_temperature()
        if temp >= THERMAL_SHUTDOWN_THRESHOLD:
            return ThermalState.SHUTDOWN
        elif temp >= THERMAL_CRITICAL_THRESHOLD:
            return ThermalState.CRITICAL
        elif temp >= THERMAL_WARNING_THRESHOLD:
            return ThermalState.HOT
        elif temp >= 60.0:
            return ThermalState.WARM
        return ThermalState.NORMAL

    def set_fan_speed(self, speed: int) -> None:
        self._fan_speed = max(0, min(100, speed))

    def get_fan_speed(self) -> int:
        return self._fan_speed

    def start_monitoring(self, interval: float = 2.0) -> None:
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="thermal-monitor",
            args=(interval,),
        )
        self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)

    def _monitor_loop(self, interval: float) -> None:
        while self._monitoring:
            temp = self.get_temperature()
            state = self.get_thermal_state()
            reading = ThermalReading(
                zone="thermal_zone0",
                temperature_c=temp,
                state=state,
            )
            with self._lock:
                self._readings.append(reading)
                if len(self._readings) > 10000:
                    self._readings = self._readings[-5000:]

            for cb in self._callbacks:
                try:
                    cb(state, temp)
                except Exception as exc:
                    logger.warning("Thermal callback error: %s", exc)

            if self._throttling_enabled and state == ThermalState.CRITICAL:
                self._fan_speed = 100
            elif state == ThermalState.HOT:
                self._fan_speed = max(self._fan_speed, 70)

            time.sleep(interval)

    def add_callback(self, callback: Callable[[ThermalState, float], None]) -> None:
        self._callbacks.append(callback)

    def get_history(self, limit: int = 100) -> List[ThermalReading]:
        with self._lock:
            return list(self._readings[-limit:])

    def simulate_load(self, temperature: float) -> None:
        self._current_temp = temperature


# ---------------------------------------------------------------------------
# Performance Profiler
# ---------------------------------------------------------------------------

class PerformanceProfiler:
    """Profiles Jetson platform performance."""

    def __init__(
        self,
        gpu_manager: GPUClockManager,
        thermal_manager: ThermalManager,
        shared_memory: SharedMemoryManager,
    ) -> None:
        self.gpu_manager = gpu_manager
        self.thermal_manager = thermal_manager
        self.shared_memory = shared_memory
        self._samples: List[ProfileSample] = []
        self._profiling = False
        self._profile_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def take_sample(self) -> ProfileSample:
        gpu_info = self.gpu_manager.get_frequency(GPUFreqDomain.GPU)
        cpu_info = self.gpu_manager.get_frequency(GPUFreqDomain.CPU)
        return ProfileSample(
            timestamp=time.time(),
            gpu_freq_mhz=gpu_info.current_mhz,
            cpu_freq_mhz=cpu_info.current_mhz,
            temperature_c=self.thermal_manager.get_temperature(),
            power_mw=0,
            gpu_utilization=0,
            memory_used_mb=self.shared_memory.get_total_info().used_bytes / (1024 * 1024),
        )

    def start_profiling(self, interval: float = 1.0, duration: float = 0) -> None:
        self._profiling = True
        self._profile_thread = threading.Thread(
            target=self._profile_loop, daemon=True, name="perf-profiler",
            args=(interval, duration),
        )
        self._profile_thread.start()

    def stop_profiling(self) -> None:
        self._profiling = False
        if self._profile_thread:
            self._profile_thread.join(timeout=5.0)

    def _profile_loop(self, interval: float, duration: float) -> None:
        start = time.time()
        while self._profiling:
            sample = self.take_sample()
            with self._lock:
                self._samples.append(sample)
            if duration > 0 and (time.time() - start) >= duration:
                break
            time.sleep(interval)

    def get_samples(self, limit: int = 1000) -> List[ProfileSample]:
        with self._lock:
            return list(self._samples[-limit:])

    def get_summary(self) -> Dict[str, Any]:
        samples = self._samples[-100:]
        if not samples:
            return {"sample_count": 0}
        return {
            "sample_count": len(samples),
            "duration_s": samples[-1].timestamp - samples[0].timestamp if len(samples) > 1 else 0,
            "avg_gpu_freq_mhz": sum(s.gpu_freq_mhz for s in samples) / len(samples),
            "avg_cpu_freq_mhz": sum(s.cpu_freq_mhz for s in samples) / len(samples),
            "avg_temp_c": sum(s.temperature_c for s in samples) / len(samples),
            "max_temp_c": max(s.temperature_c for s in samples),
            "min_temp_c": min(s.temperature_c for s in samples),
            "avg_memory_mb": sum(s.memory_used_mb for s in samples) / len(samples),
        }

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()


# ---------------------------------------------------------------------------
# Jetson Optimizer (Main Facade)
# ---------------------------------------------------------------------------

class JetsonOptimizer:
    """Main facade for Jetson platform optimization."""

    def __init__(self, simulated: bool = True) -> None:
        self.gpu_manager = GPUClockManager(simulated=simulated)
        self.thermal_manager = ThermalManager()
        self.shared_memory = SharedMemoryManager()
        self.power_controller = PowerModeController(
            self.gpu_manager, self.thermal_manager
        )
        self.profiler = PerformanceProfiler(
            self.gpu_manager, self.thermal_manager, self.shared_memory
        )
        self._pipelines: Dict[str, DeepStreamPipeline] = {}
        self._lock = threading.Lock()

    def create_pipeline(self, config: Optional[PipelineConfig] = None) -> DeepStreamPipeline:
        pipeline = DeepStreamPipeline(config)
        with self._lock:
            self._pipelines[config.name if config else "default"] = pipeline
        return pipeline

    def get_pipeline(self, name: str) -> Optional[DeepStreamPipeline]:
        return self._pipelines.get(name)

    def list_pipelines(self) -> List[str]:
        return list(self._pipelines.keys())

    def optimize_for_inference(
        self,
        model_size_mb: float = 100,
        target_fps: int = 30,
    ) -> Dict[str, Any]:
        recommendations: Dict[str, Any] = {}

        if target_fps >= 30:
            self.power_controller.set_profile(PowerProfile.MAX_PERFORMANCE)
            recommendations["profile"] = PowerProfile.MAX_PERFORMANCE.value
        elif target_fps >= 15:
            self.power_controller.set_profile(PowerProfile.HIGH_PERFORMANCE)
            recommendations["profile"] = PowerProfile.HIGH_PERFORMANCE.value
        else:
            self.power_controller.set_profile(PowerProfile.MEDIUM_PERFORMANCE)
            recommendations["profile"] = PowerProfile.MEDIUM_PERFORMANCE.value

        heap_size = int(model_size_mb * 3 * 1024 * 1024)
        heap_id = self.shared_memory.create_heap("inference", heap_size)
        recommendations["shared_memory_heap"] = heap_id
        recommendations["shared_memory_size_mb"] = heap_size // (1024 * 1024)

        gpu_info = self.gpu_manager.get_frequency(GPUFreqDomain.GPU)
        recommendations["gpu_freq_mhz"] = gpu_info.current_mhz

        power = self.power_controller.estimate_power_consumption()
        recommendations["estimated_power_w"] = power.total_mw / 1000.0

        return recommendations

    def get_platform_status(self) -> Dict[str, Any]:
        return {
            "gpu": self.gpu_manager.get_all_frequencies(),
            "thermal": {
                "temperature_c": self.thermal_manager.get_temperature(),
                "state": self.thermal_manager.get_thermal_state().value,
                "fan_speed": self.thermal_manager.get_fan_speed(),
            },
            "power": {
                "mode": self.power_controller.get_power_mode(),
                "profile": self.power_controller.get_profile().value,
                "consumption": self.power_controller.estimate_power_consumption().__dict__,
            },
            "memory": self.shared_memory.get_total_info().__dict__,
            "pipelines": {
                name: pipeline.get_metrics().__dict__
                for name, pipeline in self._pipelines.items()
            },
        }
