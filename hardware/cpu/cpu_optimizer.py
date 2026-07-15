"""
CpuOptimizer - CPU性能调优模块

提供CPU频率调节、功耗管理和性能监控功能，包括：
- CPU频率探测与调节策略
- 功耗状态管理 (C-states / P-states)
- 温度监控与热节流保护
- 性能模式切换 (性能/平衡/省电)
- CPU亲和性设置
- 实时性能指标采集

模块路径: hardware/cpu/cpu_optimizer.py
"""

import os
import sys
import time
import logging
import platform
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger(__name__)


class PowerProfile(Enum):
    """电源管理配置文件"""
    PERFORMANCE = "performance"
    BALANCED = "balanced"
    POWERSAVE = "powersave"
    CUSTOM = "custom"


class GovernorType(Enum):
    """CPU调频策略"""
    PERFORMANCE = "performance"
    POWERSAVE = "powersave"
    ONDEMAND = "ondemand"
    CONSERVATIVE = "conservative"
    SCHEDUTIL = "schedutil"
    USERSPACE = "userspace"
    INTERACTIVE = "interactive"


@dataclass
class CpuCoreInfo:
    """单个CPU核心的信息"""
    core_id: int
    physical_id: int = 0
    current_freq_mhz: float = 0.0
    min_freq_mhz: float = 0.0
    max_freq_mhz: float = 0.0
    governor: str = ""
    utilization_pct: float = 0.0
    temperature_celsius: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "core_id": self.core_id,
            "physical_id": self.physical_id,
            "current_freq_mhz": round(self.current_freq_mhz, 1),
            "min_freq_mhz": round(self.min_freq_mhz, 1),
            "max_freq_mhz": round(self.max_freq_mhz, 1),
            "governor": self.governor,
            "utilization_pct": round(self.utilization_pct, 1),
            "temperature_celsius": round(self.temperature_celsius, 1),
        }


@dataclass
class CpuSnapshot:
    """CPU状态快照"""
    timestamp: float = 0.0
    total_utilization_pct: float = 0.0
    avg_frequency_mhz: float = 0.0
    temperature_celsius: float = 0.0
    power_watts: float = 0.0
    cores: List[CpuCoreInfo] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_utilization_pct": round(self.total_utilization_pct, 1),
            "avg_frequency_mhz": round(self.avg_frequency_mhz, 1),
            "temperature_celsius": round(self.temperature_celsius, 1),
            "power_watts": round(self.power_watts, 2),
            "cores": [c.to_dict() for c in self.cores],
        }


class CpuOptimizer:
    """
    CPU性能优化器

    提供CPU频率调节、功耗管理和性能监控功能。
    支持Linux (cpufreq/sysfs)、macOS (powermetrics)、Windows (powercfg)。
    """

    # 温度阈值 (摄氏度)
    TEMP_WARNING = 75.0
    TEMP_CRITICAL = 85.0
    TEMP_SHUTDOWN = 95.0

    # 频率调节步进 (MHz)
    FREQ_STEP_MHZ = 100

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化CPU优化器

        Args:
            config: 配置字典，支持：
                - profile: PowerProfile, 电源配置文件
                - target_temp: float, 目标温度上限
                - max_power_watts: float, 最大功耗限制
                - monitor_interval: float, 监控间隔(秒)
                - enable_turbo: bool, 是否启用睿频
        """
        self.config = config or {}
        self._system = platform.system()
        self._initialized = False
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._snapshots: List[CpuSnapshot] = []
        self._lock = threading.Lock()
        self._prev_idle = 0
        self._prev_total = 0
        self._core_info: Dict[int, CpuCoreInfo] = {}

    def initialize(self) -> bool:
        """
        初始化CPU优化器，探测CPU信息

        Returns:
            bool: 初始化是否成功
        """
        try:
            self._detect_cpu_info()
            self._initialized = True
            logger.info(
                "CpuOptimizer initialized on %s with %d cores",
                self._system, len(self._core_info),
            )
            return True
        except Exception as e:
            logger.error("Failed to initialize CpuOptimizer: %s", e)
            return False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def is_monitoring(self) -> bool:
        return self._monitoring

    # ========================
    # CPU信息探测
    # ========================

    def _detect_cpu_info(self) -> None:
        """探测CPU核心信息"""
        if self._system == "Linux":
            self._detect_linux_cpu_info()
        elif self._system == "Darwin":
            self._detect_macos_cpu_info()
        elif self._system == "Windows":
            self._detect_windows_cpu_info()
        else:
            self._detect_fallback_cpu_info()

    def _detect_linux_cpu_info(self) -> None:
        """通过Linux sysfs探测CPU信息"""
        cpu_base = Path("/sys/devices/system/cpu")
        if not cpu_base.exists():
            self._detect_fallback_cpu_info()
            return

        cpu_dirs = sorted(cpu_base.glob("cpu[0-9]*"))
        for cpu_dir in cpu_dirs:
            try:
                core_id = int(cpu_dir.name.replace("cpu", ""))
                info = CpuCoreInfo(core_id=core_id)

                # 读取频率信息
                freq_base = cpu_dir / "cpufreq"
                if freq_base.exists():
                    cur = (freq_base / "scaling_cur_freq").read_text().strip()
                    min_f = (freq_base / "cpuinfo_min_freq").read_text().strip()
                    max_f = (freq_base / "cpuinfo_max_freq").read_text().strip()
                    gov = (freq_base / "scaling_governor").read_text().strip()

                    info.current_freq_mhz = int(cur) / 1000.0
                    info.min_freq_mhz = int(min_f) / 1000.0
                    info.max_freq_mhz = int(max_f) / 1000.0
                    info.governor = gov

                # 读取物理ID
                topo_file = cpu_dir / "topology" / "physical_package_id"
                if topo_file.exists():
                    info.physical_id = int(topo_file.read_text().strip())

                self._core_info[core_id] = info
            except (OSError, ValueError, KeyError) as e:
                logger.debug("Error reading CPU %s info: %s", cpu_dir, e)

    def _detect_macos_cpu_info(self) -> None:
        """通过macOS系统命令探测CPU信息"""
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.cpufrequency"],
                capture_output=True, text=True, timeout=5,
            )
            freq = int(result.stdout.strip()) / 1_000_000 if result.returncode == 0 else 0
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            freq = 0

        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                capture_output=True, text=True, timeout=5,
            )
            num_cores = int(result.stdout.strip()) if result.returncode == 0 else os.cpu_count() or 1
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            num_cores = os.cpu_count() or 1

        for i in range(num_cores):
            self._core_info[i] = CpuCoreInfo(
                core_id=i,
                current_freq_mhz=freq,
                min_freq_mhz=freq * 0.6,
                max_freq_mhz=freq * 1.0,
                governor="apple",
            )

    def _detect_windows_cpu_info(self) -> None:
        """通过Windows命令探测CPU信息"""
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get",
                 "CurrentClockSpeed,MaxClockSpeed,NumberOfCores,LoadPercentage",
                 "/format:csv"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                core_idx = 0
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if not line or line.startswith("Node") or line.startswith("Current"):
                        continue
                    parts = [p.strip() for p in line.split(",") if p.strip()]
                    if len(parts) >= 3:
                        try:
                            cur_speed = float(parts[0]) if parts[0] else 0
                            max_speed = float(parts[1]) if parts[1] else 0
                            load = float(parts[2]) if len(parts) > 2 else 0
                            self._core_info[core_idx] = CpuCoreInfo(
                                core_id=core_idx,
                                current_freq_mhz=cur_speed,
                                min_freq_mhz=max_speed * 0.5,
                                max_freq_mhz=max_speed,
                                utilization_pct=load,
                            )
                            core_idx += 1
                        except (ValueError, IndexError):
                            continue
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._detect_fallback_cpu_info()

    def _detect_fallback_cpu_info(self) -> None:
        """回退: 基础CPU信息"""
        num_cores = os.cpu_count() or 1
        for i in range(num_cores):
            self._core_info[i] = CpuCoreInfo(core_id=i)

    # ========================
    # 频率管理
    # ========================

    def get_current_frequencies(self) -> Dict[int, float]:
        """
        获取所有核心的当前频率

        Returns:
            Dict[int, float]: 核心ID到频率(MHz)的映射
        """
        frequencies = {}
        if self._system == "Linux":
            frequencies = self._get_linux_frequencies()
        elif self._system == "Darwin":
            frequencies = self._get_macos_frequencies()
        elif self._system == "Windows":
            frequencies = self._get_windows_frequencies()

        # 回退到缓存值
        if not frequencies:
            for core_id, info in self._core_info.items():
                frequencies[core_id] = info.current_freq_mhz

        return frequencies

    def _get_linux_frequencies(self) -> Dict[int, float]:
        """获取Linux CPU频率"""
        frequencies = {}
        cpu_base = Path("/sys/devices/system/cpu")
        for core_id in self._core_info:
            freq_file = cpu_base / f"cpu{core_id}" / "cpufreq" / "scaling_cur_freq"
            try:
                freq = int(freq_file.read_text().strip()) / 1000.0
                frequencies[core_id] = freq
            except (OSError, ValueError):
                pass
        return frequencies

    def _get_macos_frequencies(self) -> Dict[int, float]:
        """获取macOS CPU频率"""
        frequencies = {}
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.cpufrequency"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                freq = int(result.stdout.strip()) / 1_000_000
                for core_id in self._core_info:
                    frequencies[core_id] = freq
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass
        return frequencies

    def _get_windows_frequencies(self) -> Dict[int, float]:
        """获取Windows CPU频率"""
        frequencies = {}
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "CurrentClockSpeed", "/format:csv"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                core_idx = 0
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if not line or line.startswith("Node") or line.startswith("Current"):
                        continue
                    try:
                        freq = float(line.split(",")[-1].strip())
                        if core_idx in self._core_info:
                            frequencies[core_idx] = freq
                        core_idx += 1
                    except (ValueError, IndexError):
                        continue
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return frequencies

    def set_governor(self, governor: str) -> bool:
        """
        设置CPU调频策略 (仅Linux)

        Args:
            governor: 调频策略名称 (performance/powersave/ondemand等)

        Returns:
            bool: 是否设置成功
        """
        if self._system != "Linux":
            logger.warning("Governor setting is only supported on Linux")
            return False

        cpu_base = Path("/sys/devices/system/cpu")
        success = True
        for core_id in self._core_info:
            gov_file = cpu_base / f"cpu{core_id}" / "cpufreq" / "scaling_governor"
            try:
                gov_file.write_text(governor + "\n")
                if core_id in self._core_info:
                    self._core_info[core_id].governor = governor
            except (OSError, PermissionError) as e:
                logger.warning("Failed to set governor for CPU %d: %s", core_id, e)
                success = False
        return success

    def get_available_governors(self) -> List[str]:
        """
        获取可用的调频策略列表 (仅Linux)

        Returns:
            List[str]: 可用的调频策略名称
        """
        if self._system != "Linux":
            return []

        governors: List[str] = []
        cpu_base = Path("/sys/devices/system/cpu")
        for core_id in self._core_info:
            avail_file = cpu_base / f"cpu{core_id}" / "cpufreq" / "scaling_available_governors"
            try:
                content = avail_file.read_text().strip()
                governors = content.split()
                break
            except (OSError, FileNotFoundError):
                continue
        return governors

    def set_frequency(self, core_id: int, freq_mhz: float) -> bool:
        """
        设置指定核心的频率 (需要userspace governor)

        Args:
            core_id: 核心ID
            freq_mhz: 目标频率 (MHz)

        Returns:
            bool: 是否设置成功
        """
        if self._system != "Linux":
            logger.warning("Manual frequency setting is only supported on Linux")
            return False

        freq_khz = int(freq_mhz * 1000)
        cpu_base = Path("/sys/devices/system/cpu")
        freq_file = cpu_base / f"cpu{core_id}" / "cpufreq" / "scaling_setspeed"

        try:
            freq_file.write_text(str(freq_khz) + "\n")
            if core_id in self._core_info:
                self._core_info[core_id].current_freq_mhz = freq_mhz
            return True
        except (OSError, PermissionError) as e:
            logger.warning("Failed to set frequency for CPU %d: %s", core_id, e)
            return False

    # ========================
    # 电源管理
    # ========================

    def set_power_profile(self, profile: PowerProfile) -> bool:
        """
        设置系统电源管理配置

        Args:
            profile: 电源配置文件

        Returns:
            bool: 是否设置成功
        """
        if self._system == "Linux":
            return self._set_linux_power_profile(profile)
        elif self._system == "Windows":
            return self._set_windows_power_profile(profile)
        elif self._system == "Darwin":
            return self._set_macos_power_profile(profile)
        return False

    def _set_linux_power_profile(self, profile: PowerProfile) -> bool:
        """设置Linux电源配置"""
        # 尝试使用powerprofilesctl (systemd)
        try:
            result = subprocess.run(
                ["powerprofilesctl", "set", profile.value],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # 回退: 设置governor
        governor_map = {
            PowerProfile.PERFORMANCE: "performance",
            PowerProfile.POWERSAVE: "powersave",
            PowerProfile.BALANCED: "ondemand",
        }
        governor = governor_map.get(profile, "ondemand")
        return self.set_governor(governor)

    def _set_windows_power_profile(self, profile: PowerProfile) -> bool:
        """设置Windows电源配置"""
        scheme_map = {
            PowerProfile.PERFORMANCE: "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c",
            PowerProfile.POWERSAVE: "a1841308-3541-4fab-bc81-f71556f20b4a",
            PowerProfile.BALANCED: "381b4222-f694-41f0-9685-ff5bb260df2e",
        }
        scheme_guid = scheme_map.get(profile)
        if not scheme_guid:
            return False

        try:
            result = subprocess.run(
                ["powercfg", "/setactive", scheme_guid],
                capture_output=True, text=True, timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _set_macos_power_profile(self, profile: PowerProfile) -> bool:
        """设置macOS电源配置"""
        # macOS没有直接的电源配置切换，使用pmset调整
        try:
            if profile == PowerProfile.PERFORMANCE:
                subprocess.run(
                    ["sudo", "pmset", "-c", "sleep", "0", "hibernatemode", "0"],
                    capture_output=True, text=True, timeout=5,
                )
            elif profile == PowerProfile.POWERSAVE:
                subprocess.run(
                    ["sudo", "pmset", "-c", "sleep", "30", "hibernatemode", "3"],
                    capture_output=True, text=True, timeout=5,
                )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def get_power_profile(self) -> PowerProfile:
        """
        获取当前电源管理配置

        Returns:
            PowerProfile: 当前电源配置
        """
        if self._system == "Linux":
            return self._get_linux_power_profile()
        elif self._system == "Windows":
            return self._get_windows_power_profile()
        return PowerProfile.BALANCED

    def _get_linux_power_profile(self) -> PowerProfile:
        """获取Linux当前电源配置"""
        try:
            result = subprocess.run(
                ["powerprofilesctl", "get"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                profile_str = result.stdout.strip()
                for profile in PowerProfile:
                    if profile.value in profile_str:
                        return profile
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # 回退: 检查governor
        governors = set()
        for info in self._core_info.values():
            if info.governor:
                governors.add(info.governor)

        if "performance" in governors:
            return PowerProfile.PERFORMANCE
        elif "powersave" in governors:
            return PowerProfile.POWERSAVE
        return PowerProfile.BALANCED

    def _get_windows_power_profile(self) -> PowerProfile:
        """获取Windows当前电源配置"""
        try:
            result = subprocess.run(
                ["powercfg", "/getactivescheme"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                output = result.stdout.lower()
                if "performance" in output:
                    return PowerProfile.PERFORMANCE
                elif "saver" in output:
                    return PowerProfile.POWERSAVE
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return PowerProfile.BALANCED

    # ========================
    # 温度监控
    # ========================

    def get_temperature(self) -> float:
        """
        获取CPU温度

        Returns:
            float: 温度 (摄氏度)，获取失败返回0.0
        """
        if self._system == "Linux":
            return self._get_linux_temperature()
        elif self._system == "Darwin":
            return self._get_macos_temperature()
        elif self._system == "Windows":
            return self._get_windows_temperature()
        return 0.0

    def _get_linux_temperature(self) -> float:
        """获取Linux CPU温度"""
        # 方法1: 通过sysfs热区
        thermal_base = Path("/sys/class/thermal")
        if thermal_base.exists():
            for zone in sorted(thermal_base.glob("thermal_zone*")):
                try:
                    temp_file = zone / "temp"
                    type_file = zone / "type"
                    zone_type = type_file.read_text().strip() if type_file.exists() else ""
                    if "cpu" in zone_type.lower() or "core" in zone_type.lower() or "package" in zone_type.lower():
                        temp = int(temp_file.read_text().strip()) / 1000.0
                        if 0 < temp < 150:
                            return temp
                except (OSError, ValueError):
                    continue

        # 方法2: 通过hwmon
        hwmon_base = Path("/sys/class/hwmon")
        if hwmon_base.exists():
            for hwmon in sorted(hwmon_base.glob("hwmon*")):
                try:
                    name_file = hwmon / "name"
                    if name_file.exists() and "coretemp" in name_file.read_text().strip().lower():
                        for temp_input in sorted(hwmon.glob("temp*_input")):
                            temp = int(temp_input.read_text().strip()) / 1000.0
                            if 0 < temp < 150:
                                return temp
                except (OSError, ValueError):
                    continue

        return 0.0

    def _get_macos_temperature(self) -> float:
        """获取macOS CPU温度"""
        try:
            result = subprocess.run(
                ["osx-cpu-temp"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                temp = float(result.stdout.strip().replace("°C", ""))
                return temp
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass
        return 0.0

    def _get_windows_temperature(self) -> float:
        """获取Windows CPU温度"""
        try:
            result = subprocess.run(
                ["wmic", "/namespace:\\\\root\\wmi", "MSAcpi_ThermalZoneTemperature",
                 "get", "CurrentTemperature", "/format:csv"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if not line or line.startswith("Node") or line.startswith("Current"):
                        continue
                    try:
                        temp = float(line.split(",")[-1].strip()) / 10.0 - 273.15
                        if 0 < temp < 150:
                            return temp
                    except (ValueError, IndexError):
                        continue
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return 0.0

    # ========================
    # 性能监控
    # ========================

    def get_cpu_utilization(self) -> float:
        """
        获取CPU总利用率

        Returns:
            float: CPU利用率百分比 (0.0 - 100.0)
        """
        if self._system == "Linux":
            return self._get_linux_utilization()
        elif self._system == "Darwin":
            return self._get_macos_utilization()
        elif self._system == "Windows":
            return self._get_windows_utilization()
        return 0.0

    def _get_linux_utilization(self) -> float:
        """获取Linux CPU利用率"""
        try:
            stat = Path("/proc/stat").read_text()
            for line in stat.splitlines():
                if not line.startswith("cpu "):
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                user = int(parts[1])
                nice = int(parts[2])
                system = int(parts[3])
                idle = int(parts[4])
                iowait = int(parts[5]) if len(parts) > 5 else 0

                total = user + nice + system + idle + iowait
                idle_total = idle + iowait

                diff_idle = idle_total - self._prev_idle
                diff_total = total - self._prev_total

                self._prev_idle = idle_total
                self._prev_total = total

                if diff_total > 0:
                    return (1.0 - diff_idle / diff_total) * 100.0
        except (OSError, ValueError, IndexError):
            pass
        return 0.0

    def _get_macos_utilization(self) -> float:
        """获取macOS CPU利用率"""
        try:
            result = subprocess.run(
                ["top", "-l", "1", "-n", "0", "-s", "0"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "CPU usage" in line:
                        parts = line.split(",")
                        for part in parts:
                            part = part.strip()
                            if "idle" in part:
                                try:
                                    idle_pct = float(part.split()[0])
                                    return 100.0 - idle_pct
                                except (ValueError, IndexError):
                                    pass
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return 0.0

    def _get_windows_utilization(self) -> float:
        """获取Windows CPU利用率"""
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "LoadPercentage", "/format:csv"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if not line or line.startswith("Node") or line.startswith("Load"):
                        continue
                    try:
                        return float(line.split(",")[-1].strip())
                    except (ValueError, IndexError):
                        continue
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return 0.0

    def get_per_core_utilization(self) -> Dict[int, float]:
        """
        获取每个核心的利用率

        Returns:
            Dict[int, float]: 核心ID到利用率百分比的映射
        """
        utilization = {}
        if self._system == "Linux":
            utilization = self._get_linux_per_core_util()
        elif self._system == "Windows":
            utilization = self._get_windows_per_core_util()
        return utilization

    def _get_linux_per_core_util(self) -> Dict[int, float]:
        """获取Linux每核利用率"""
        utilization = {}
        try:
            stat = Path("/proc/stat").read_text()
            for line in stat.splitlines():
                if not line.startswith("cpu") or line.startswith("cpu "):
                    continue
                parts = line.split()
                core_name = parts[0]
                try:
                    core_id = int(core_name[3:])
                except ValueError:
                    continue
                if len(parts) < 5:
                    continue
                user = int(parts[1])
                nice = int(parts[2])
                system = int(parts[3])
                idle = int(parts[4])
                total = user + nice + system + idle
                busy = user + nice + system
                if total > 0:
                    utilization[core_id] = (busy / total) * 100.0
        except (OSError, ValueError):
            pass
        return utilization

    def _get_windows_per_core_util(self) -> Dict[int, float]:
        """获取Windows每核利用率"""
        utilization = {}
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "LoadPercentage", "/format:csv"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                core_idx = 0
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if not line or line.startswith("Node") or line.startswith("Load"):
                        continue
                    try:
                        load = float(line.split(",")[-1].strip())
                        utilization[core_idx] = load
                        core_idx += 1
                    except (ValueError, IndexError):
                        continue
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return utilization

    # ========================
    # 功耗估算
    # ========================

    def estimate_power_consumption(self) -> float:
        """
        估算CPU功耗 (瓦特)

        基于CPU利用率、频率和TDP进行估算。

        Returns:
            float: 估算功耗 (瓦特)
        """
        utilization = self.get_cpu_utilization() / 100.0
        tdp = self._estimate_tdp()

        # 基础功耗 + 动态功耗
        static_power = tdp * 0.1  # 空闲功耗约为TDP的10%
        dynamic_power = tdp * 0.9 * utilization

        # 频率修正
        freqs = self.get_current_frequencies()
        if freqs:
            avg_freq = sum(freqs.values()) / len(freqs)
            max_freq = max(
                (info.max_freq_mhz for info in self._core_info.values() if info.max_freq_mhz > 0),
                default=1.0,
            )
            if max_freq > 0:
                freq_ratio = avg_freq / max_freq
                dynamic_power *= freq_ratio

        return static_power + dynamic_power

    def _estimate_tdp(self) -> float:
        """估算CPU TDP (热设计功耗)"""
        processor = platform.processor().lower()
        num_cores = len(self._core_info)

        # 基于CPU型号的TDP估算
        if "xeon" in processor:
            return 105.0 + num_cores * 2.5
        elif "epyc" in processor:
            return 120.0 + num_cores * 3.0
        elif "i9" in processor or "i7" in processor:
            return 95.0
        elif "i5" in processor:
            return 65.0
        elif "ryzen 9" in processor:
            return 105.0
        elif "ryzen 7" in processor:
            return 65.0
        elif "apple" in processor:
            if num_cores <= 4:
                return 15.0
            elif num_cores <= 8:
                return 30.0
            else:
                return 60.0

        # 默认: 基于核心数估算
        return 15.0 + num_cores * 5.0

    # ========================
    # 热节流保护
    # ========================

    def check_thermal_status(self) -> Dict[str, Any]:
        """
        检查CPU热状态

        Returns:
            Dict: 热状态信息，包含温度、状态和建议操作
        """
        temp = self.get_temperature()
        status = "normal"
        action = None

        if temp >= self.TEMP_SHUTDOWN:
            status = "critical_shutdown"
            action = "emergency_shutdown"
        elif temp >= self.TEMP_CRITICAL:
            status = "critical"
            action = "reduce_frequency"
        elif temp >= self.TEMP_WARNING:
            status = "warning"
            action = "monitor_closely"

        return {
            "temperature_celsius": round(temp, 1),
            "status": status,
            "recommended_action": action,
            "warning_threshold": self.TEMP_WARNING,
            "critical_threshold": self.TEMP_CRITICAL,
            "shutdown_threshold": self.TEMP_SHUTDOWN,
        }

    def apply_thermal_throttling(self) -> bool:
        """
        应用热节流保护措施

        当温度超过阈值时，自动降低CPU频率以降温。

        Returns:
            bool: 是否执行了节流操作
        """
        thermal = self.check_thermal_status()
        if thermal["status"] == "normal":
            return False

        if self._system == "Linux":
            if thermal["status"] in ("critical", "critical_shutdown"):
                return self.set_governor("powersave")
            elif thermal["status"] == "warning":
                return self.set_governor("ondemand")
        return False

    # ========================
    # 性能快照与监控
    # ========================

    def take_snapshot(self) -> CpuSnapshot:
        """
        获取当前CPU状态快照

        Returns:
            CpuSnapshot: CPU状态快照
        """
        snapshot = CpuSnapshot(
            timestamp=time.time(),
            total_utilization_pct=self.get_cpu_utilization(),
            temperature_celsius=self.get_temperature(),
            power_watts=self.estimate_power_consumption(),
        )

        freqs = self.get_current_frequencies()
        per_core_util = self.get_per_core_utilization()

        for core_id, info in self._core_info.items():
            core = CpuCoreInfo(
                core_id=info.core_id,
                physical_id=info.physical_id,
                current_freq_mhz=freqs.get(core_id, info.current_freq_mhz),
                min_freq_mhz=info.min_freq_mhz,
                max_freq_mhz=info.max_freq_mhz,
                governor=info.governor,
                utilization_pct=per_core_util.get(core_id, 0.0),
                temperature_celsius=snapshot.temperature_celsius,
            )
            snapshot.cores.append(core)

        if snapshot.cores:
            snapshot.avg_frequency_mhz = sum(
                c.current_freq_mhz for c in snapshot.cores
            ) / len(snapshot.cores)

        with self._lock:
            self._snapshots.append(snapshot)
            # 保留最近1000个快照
            if len(self._snapshots) > 1000:
                self._snapshots = self._snapshots[-500:]

        return snapshot

    def start_monitoring(self, interval: float = 1.0) -> bool:
        """
        启动后台性能监控

        Args:
            interval: 监控间隔 (秒)

        Returns:
            bool: 是否成功启动
        """
        if self._monitoring:
            logger.warning("Monitoring is already running")
            return False

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True,
        )
        self._monitor_thread.start()
        logger.info("CPU monitoring started (interval=%.1fs)", interval)
        return True

    def stop_monitoring(self) -> None:
        """停止后台性能监控"""
        self._monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        self._monitor_thread = None
        logger.info("CPU monitoring stopped")

    def _monitor_loop(self, interval: float) -> None:
        """监控循环"""
        while self._monitoring:
            try:
                self.take_snapshot()
                thermal = self.check_thermal_status()
                if thermal["status"] != "normal":
                    logger.warning(
                        "Thermal alert: %.1f°C (%s)",
                        thermal["temperature_celsius"], thermal["status"],
                    )
                    if thermal["recommended_action"] == "reduce_frequency":
                        self.apply_thermal_throttling()
            except Exception as e:
                logger.error("Monitor loop error: %s", e)
            time.sleep(interval)

    def get_snapshots(self, count: int = 10) -> List[CpuSnapshot]:
        """
        获取最近的性能快照

        Args:
            count: 返回的快照数量

        Returns:
            List[CpuSnapshot]: 最近的快照列表
        """
        with self._lock:
            return list(self._snapshots[-count:])

    def get_summary(self) -> Dict[str, Any]:
        """
        获取CPU优化器的完整摘要

        Returns:
            Dict: 包含所有CPU状态和配置的摘要
        """
        snapshot = self.take_snapshot()
        return {
            "initialized": self._initialized,
            "platform": self._system,
            "num_cores": len(self._core_info),
            "current_profile": self.get_power_profile().value,
            "available_governors": self.get_available_governors() if self._system == "Linux" else [],
            "thermal_status": self.check_thermal_status(),
            "current_snapshot": snapshot.to_dict(),
        }

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        n_cores = len(self._core_info)
        return f"CpuOptimizer({status}, {n_cores} cores, {self._system})"
