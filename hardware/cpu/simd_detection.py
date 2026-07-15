"""
SimdDetection - SIMD指令集检测模块

提供CPU SIMD指令集检测和能力查询功能，包括：
- x86: SSE/SSE2/SSE3/SSSE3/SSE4.1/SSE4.2/AVX/AVX2/AVX512
- ARM: NEON/FP16/SVE/SVE2
- 通用: SIMD寄存器大小、向量通道数
- 运行时SIMD能力验证
- SIMD优化建议生成

模块路径: hardware/cpu/simd_detection.py
"""

import os
import sys
import struct
import ctypes
import logging
import platform
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger(__name__)


class SimdArchitecture(Enum):
    """SIMD架构类型"""
    X86 = "x86"
    X86_64 = "x86_64"
    ARM = "arm"
    ARM64 = "arm64"
    RISCV = "riscv"
    UNKNOWN = "unknown"


class SimdExtension(Enum):
    """SIMD扩展指令集"""
    # x86 SSE系列
    SSE = "SSE"
    SSE2 = "SSE2"
    SSE3 = "SSE3"
    SSSE3 = "SSSE3"
    SSE4_1 = "SSE4.1"
    SSE4_2 = "SSE4.2"
    # x86 AVX系列
    AVX = "AVX"
    AVX2 = "AVX2"
    FMA = "FMA"
    AVX512F = "AVX512F"
    AVX512DQ = "AVX512DQ"
    AVX512IFMA = "AVX512IFMA"
    AVX512PF = "AVX512PF"
    AVX512ER = "AVX512ER"
    AVX512CD = "AVX512CD"
    AVX512BW = "AVX512BW"
    AVX512VL = "AVX512VL"
    AVX512VNNI = "AVX512VNNI"
    AVX512BF16 = "AVX512BF16"
    # ARM NEON系列
    NEON = "NEON"
    FP16 = "FP16"
    SVE = "SVE"
    SVE2 = "SVE2"
    # 其他
    NONE = "NONE"


@dataclass
class SimdCapability:
    """单个SIMD扩展的能力描述"""
    extension: SimdExtension
    supported: bool = False
    register_size_bits: int = 0
    register_size_bytes: int = 0
    vector_width_32bit: int = 0  # float32/int32通道数
    vector_width_64bit: int = 0  # float64/int64通道数
    vector_width_8bit: int = 0   # int8通道数

    def to_dict(self) -> Dict[str, Any]:
        return {
            "extension": self.extension.value,
            "supported": self.supported,
            "register_size_bits": self.register_size_bits,
            "register_size_bytes": self.register_size_bytes,
            "vector_width_32bit": self.vector_width_32bit,
            "vector_width_64bit": self.vector_width_64bit,
            "vector_width_8bit": self.vector_width_8bit,
        }


@dataclass
class SimdProfile:
    """SIMD能力配置文件"""
    architecture: SimdArchitecture = SimdArchitecture.UNKNOWN
    vendor: str = ""
    model: str = ""
    extensions: List[SimdCapability] = field(default_factory=list)
    max_register_bits: int = 0
    max_vector_width_32bit: int = 0
    max_vector_width_64bit: int = 0
    recommended_extension: str = ""

    def get_supported_extensions(self) -> List[str]:
        """获取所有支持的扩展名称"""
        return [
            cap.extension.value
            for cap in self.extensions
            if cap.supported
        ]

    def supports(self, extension: SimdExtension) -> bool:
        """检查是否支持指定扩展"""
        for cap in self.extensions:
            if cap.extension == extension:
                return cap.supported
        return False

    def get_capability(self, extension: SimdExtension) -> Optional[SimdCapability]:
        """获取指定扩展的能力信息"""
        for cap in self.extensions:
            if cap.extension == extension:
                return cap
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture.value,
            "vendor": self.vendor,
            "model": self.model,
            "supported_extensions": self.get_supported_extensions(),
            "max_register_bits": self.max_register_bits,
            "max_vector_width_32bit": self.max_vector_width_32bit,
            "max_vector_width_64bit": self.max_vector_width_64bit,
            "recommended_extension": self.recommended_extension,
            "extensions": [e.to_dict() for e in self.extensions if e.supported],
        }


class SimdDetection:
    """
    SIMD指令集检测器

    提供跨平台的CPU SIMD能力检测和优化建议。
    支持x86 (CPUID)、ARM (sysctl/proc)、RISC-V。
    """

    # x86 SIMD扩展的CPUID特征位定义
    X86_CPUID_FEATURES = {
        # EBX for leaf 1
        "SSE": (1, 0, 25),
        "SSE2": (1, 0, 26),
        # ECX for leaf 1
        "SSE3": (1, 1, 0),
        "SSSE3": (1, 1, 9),
        "FMA": (1, 1, 12),
        "SSE4_1": (1, 1, 19),
        "SSE4_2": (1, 1, 20),
        "AVX": (1, 1, 28),
        # EBX for leaf 7, subleaf 0
        "AVX2": (7, 1, 5),
        "AVX512F": (7, 1, 16),
        "AVX512DQ": (7, 1, 17),
        "AVX512IFMA": (7, 1, 21),
        "AVX512PF": (7, 1, 26),
        "AVX512ER": (7, 1, 27),
        "AVX512CD": (7, 1, 28),
        "AVX512BW": (7, 1, 30),
        "AVX512VL": (7, 1, 31),
        # ECX for leaf 7, subleaf 0
        "AVX512VNNI": (7, 2, 11),
        "AVX512BF16": (7, 2, 30),
    }

    # SIMD扩展的寄存器信息
    EXTENSION_INFO = {
        SimdExtension.SSE:    {"reg_bits": 128, "w32": 4, "w64": 2, "w8": 16},
        SimdExtension.SSE2:   {"reg_bits": 128, "w32": 4, "w64": 2, "w8": 16},
        SimdExtension.SSE3:   {"reg_bits": 128, "w32": 4, "w64": 2, "w8": 16},
        SimdExtension.SSSE3:  {"reg_bits": 128, "w32": 4, "w64": 2, "w8": 16},
        SimdExtension.SSE4_1: {"reg_bits": 128, "w32": 4, "w64": 2, "w8": 16},
        SimdExtension.SSE4_2: {"reg_bits": 128, "w32": 4, "w64": 2, "w8": 16},
        SimdExtension.AVX:    {"reg_bits": 256, "w32": 8, "w64": 4, "w8": 32},
        SimdExtension.AVX2:   {"reg_bits": 256, "w32": 8, "w64": 4, "w8": 32},
        SimdExtension.FMA:    {"reg_bits": 256, "w32": 8, "w64": 4, "w8": 32},
        SimdExtension.AVX512F:      {"reg_bits": 512, "w32": 16, "w64": 8, "w8": 64},
        SimdExtension.AVX512DQ:     {"reg_bits": 512, "w32": 16, "w64": 8, "w8": 64},
        SimdExtension.AVX512BW:     {"reg_bits": 512, "w32": 16, "w64": 8, "w8": 64},
        SimdExtension.AVX512VNNI:   {"reg_bits": 512, "w32": 16, "w64": 8, "w8": 64},
        SimdExtension.AVX512BF16:   {"reg_bits": 512, "w32": 16, "w64": 8, "w8": 64},
        SimdExtension.NEON:   {"reg_bits": 128, "w32": 4, "w64": 2, "w8": 16},
        SimdExtension.FP16:   {"reg_bits": 128, "w32": 4, "w64": 2, "w8": 16},
        SimdExtension.SVE:    {"reg_bits": 0,   "w32": 0, "w64": 0, "w8": 0},
        SimdExtension.SVE2:   {"reg_bits": 0,   "w32": 0, "w64": 0, "w8": 0},
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化SIMD检测器

        Args:
            config: 配置字典，支持：
                - auto_detect: bool, 是否自动检测
                - prefer_extension: str, 优先使用的扩展
        """
        self.config = config or {}
        self._profile: Optional[SimdProfile] = None
        self._initialized = False

        if self.config.get("auto_detect", True):
            self._profile = self.detect()

    def initialize(self) -> bool:
        """初始化SIMD检测器"""
        try:
            if self._profile is None:
                self._profile = self.detect()
            self._initialized = True
            logger.info(
                "SimdDetection initialized: arch=%s, extensions=%s",
                self._profile.architecture.value,
                self._profile.get_supported_extensions(),
            )
            return True
        except Exception as e:
            logger.error("Failed to initialize SimdDetection: %s", e)
            return False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def profile(self) -> Optional[SimdProfile]:
        return self._profile

    # ========================
    # SIMD检测
    # ========================

    def detect(self) -> SimdProfile:
        """
        检测CPU SIMD能力

        Returns:
            SimdProfile: SIMD能力配置文件
        """
        machine = platform.machine().lower()
        system = platform.system()

        if machine in ("x86_64", "amd64", "i686", "i386"):
            arch = SimdArchitecture.X86_64 if "64" in machine else SimdArchitecture.X86
            return self._detect_x86(arch)
        elif machine.startswith("arm") or machine == "aarch64":
            arch = SimdArchitecture.ARM64 if machine == "aarch64" else SimdArchitecture.ARM
            return self._detect_arm(arch)
        elif machine.startswith("riscv"):
            return self._detect_riscv()
        else:
            return self._detect_unknown()

    def _detect_x86(self, arch: SimdArchitecture) -> SimdProfile:
        """检测x86/x86_64 SIMD扩展"""
        profile = SimdProfile(architecture=arch)

        # 获取CPU厂商和型号
        profile.vendor, profile.model = self._get_x86_cpu_info()

        # 通过CPUID检测
        cpu_id_results = self._run_cpuid()

        # 创建扩展能力列表
        extensions = []
        for ext_name, (leaf, reg_idx, bit) in self.X86_CPUID_FEATURES.items():
            try:
                ext_enum = SimdExtension[ext_name]
            except KeyError:
                continue

            supported = False
            if cpu_id_results:
                key = (leaf, reg_idx)
                if key in cpu_id_results:
                    supported = bool(cpu_id_results[key] & (1 << bit))

            info = self.EXTENSION_INFO.get(ext_enum, {})
            cap = SimdCapability(
                extension=ext_enum,
                supported=supported,
                register_size_bits=info.get("reg_bits", 0),
                register_size_bytes=info.get("reg_bits", 0) // 8,
                vector_width_32bit=info.get("w32", 0),
                vector_width_64bit=info.get("w64", 0),
                vector_width_8bit=info.get("w8", 0),
            )
            extensions.append(cap)

        profile.extensions = extensions

        # 计算最大能力
        for cap in extensions:
            if cap.supported:
                profile.max_register_bits = max(
                    profile.max_register_bits, cap.register_size_bits
                )
                profile.max_vector_width_32bit = max(
                    profile.max_vector_width_32bit, cap.vector_width_32bit
                )
                profile.max_vector_width_64bit = max(
                    profile.max_vector_width_64bit, cap.vector_width_64bit
                )

        # 推荐扩展
        if profile.supports(SimdExtension.AVX512F):
            profile.recommended_extension = "AVX512"
        elif profile.supports(SimdExtension.AVX2):
            profile.recommended_extension = "AVX2"
        elif profile.supports(SimdExtension.AVX):
            profile.recommended_extension = "AVX"
        elif profile.supports(SimdExtension.SSE4_2):
            profile.recommended_extension = "SSE4.2"
        elif profile.supports(SimdExtension.SSE2):
            profile.recommended_extension = "SSE2"
        else:
            profile.recommended_extension = "scalar"

        return profile

    def _run_cpuid(self) -> Dict[Tuple[int, int], int]:
        """
        执行CPUID指令获取CPU特征

        Returns:
            Dict: (leaf, register_index) -> 寄存器值
        """
        results: Dict[Tuple[int, int], int] = {}
        system = platform.system()

        if system == "Linux":
            results = self._cpuid_linux()
        elif system == "Darwin":
            results = self._cpuid_macos()
        elif system == "Windows":
            results = self._cpuid_windows()

        # 回退: 通过/proc/cpuinfo flags
        if not results and system == "Linux":
            results = self._cpuid_from_proc_cpuinfo()

        return results

    def _cpuid_linux(self) -> Dict[Tuple[int, int], int]:
        """通过Linux /proc/cpuinfo flags检测"""
        results: Dict[Tuple[int, int], int] = {}

        # 方法1: 直接执行CPUID (通过汇编)
        try:
            code = """
#include <stdio.h>
#include <cpuid.h>
int main() {
    unsigned int eax, ebx, ecx, edx;
    // Leaf 1
    __cpuid(1, eax, ebx, ecx, edx);
    printf("1_0 %u\\n", ebx);
    printf("1_1 %u\\n", ecx);
    printf("1_2 %u\\n", edx);
    // Leaf 7
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    printf("7_0 %u\\n", ebx);
    printf("7_1 %u\\n", ecx);
    printf("7_2 %u\\n", edx);
    return 0;
}
"""
            src_file = "/tmp/cpuid_probe.c"
            bin_file = "/tmp/cpuid_probe"
            with open(src_file, "w") as f:
                f.write(code)
            compile_result = subprocess.run(
                ["gcc", "-o", bin_file, src_file],
                capture_output=True, text=True, timeout=10,
            )
            if compile_result.returncode == 0:
                run_result = subprocess.run(
                    [bin_file], capture_output=True, text=True, timeout=5,
                )
                if run_result.returncode == 0:
                    for line in run_result.stdout.splitlines():
                        parts = line.strip().split()
                        if len(parts) == 2:
                            leaf_str, val_str = parts
                            leaf_parts = leaf_str.split("_")
                            leaf = int(leaf_parts[0])
                            reg = int(leaf_parts[1])
                            results[(leaf, reg)] = int(val_str)
            # 清理
            for f in [src_file, bin_file]:
                try:
                    os.unlink(f)
                except OSError:
                    pass
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass

        return results

    def _cpuid_macos(self) -> Dict[Tuple[int, int], int]:
        """通过macOS sysctl检测SIMD"""
        results: Dict[Tuple[int, int], int] = {}

        try:
            result = subprocess.run(
                ["sysctl", "-a"], capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                output = result.stdout.lower()
                flags = {
                    "sse": (1, 0, 25),
                    "sse2": (1, 0, 26),
                    "sse3": (1, 1, 0),
                    "ssse3": (1, 1, 9),
                    "sse4_1": (1, 1, 19),
                    "sse4_2": (1, 1, 20),
                    "avx1.0": (1, 1, 28),
                    "avx2.0": (7, 1, 5),
                }
                for flag_name, (leaf, reg, bit) in flags.items():
                    if flag_name in output or flag_name.replace(".", "") in output:
                        key = (leaf, reg)
                        if key not in results:
                            results[key] = 0
                        results[key] |= (1 << bit)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return results

    def _cpuid_windows(self) -> Dict[Tuple[int, int], int]:
        """通过Windows检测SIMD"""
        results: Dict[Tuple[int, int], int] = {}

        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "Caption,Name", "/format:csv"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                output = result.stdout.lower()
                flags = {
                    "sse": (1, 0, 25),
                    "sse2": (1, 0, 26),
                    "sse3": (1, 1, 0),
                    "ssse3": (1, 1, 9),
                    "sse4": (1, 1, 19),
                    "avx": (1, 1, 28),
                    "avx2": (7, 1, 5),
                }
                for flag_name, (leaf, reg, bit) in flags.items():
                    if flag_name in output:
                        key = (leaf, reg)
                        if key not in results:
                            results[key] = 0
                        results[key] |= (1 << bit)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return results

    def _cpuid_from_proc_cpuinfo(self) -> Dict[Tuple[int, int], int]:
        """从/proc/cpuinfo flags解析CPUID特征"""
        results: Dict[Tuple[int, int], int] = {}

        try:
            cpuinfo = Path("/proc/cpuinfo").read_text()
            flags_match = re.search(r"flags\s*:\s*(.+)", cpuinfo)
            if not flags_match:
                return results

            flags = flags_match.group(1).split()

            flag_to_cpuid = {
                "sse": (1, 0, 25), "sse2": (1, 0, 26),
                "sse3": (1, 1, 0), "ssse3": (1, 1, 9),
                "fma": (1, 1, 12),
                "sse4_1": (1, 1, 19), "sse4_2": (1, 1, 20),
                "avx": (1, 1, 28),
                "avx2": (7, 1, 5),
                "avx512f": (7, 1, 16), "avx512dq": (7, 1, 17),
                "avx512ifma": (7, 1, 21), "avx512pf": (7, 1, 26),
                "avx512er": (7, 1, 27), "avx512cd": (7, 1, 28),
                "avx512bw": (7, 1, 30), "avx512vl": (7, 1, 31),
                "avx512vnni": (7, 2, 11), "avx512_bf16": (7, 2, 30),
            }

            for flag in flags:
                if flag in flag_to_cpuid:
                    leaf, reg, bit = flag_to_cpuid[flag]
                    key = (leaf, reg)
                    if key not in results:
                        results[key] = 0
                    results[key] |= (1 << bit)
        except (OSError, re.error):
            pass

        return results

    def _detect_arm(self, arch: SimdArchitecture) -> SimdProfile:
        """检测ARM SIMD扩展"""
        profile = SimdProfile(architecture=arch)

        system = platform.system()
        features: Set[str] = set()

        if system == "Linux":
            features = self._detect_arm_linux()
        elif system == "Darwin":
            features = self._detect_arm_macos()

        # ARM扩展映射
        arm_extensions = {
            "neon": SimdExtension.NEON,
            "fp16": SimdExtension.FP16,
            "sve": SimdExtension.SVE,
            "sve2": SimdExtension.SVE2,
            "asimd": SimdExtension.NEON,  # ASIMD = NEON
        }

        extensions = []
        for feat in features:
            feat_lower = feat.lower().strip()
            ext = arm_extensions.get(feat_lower)
            if ext:
                info = self.EXTENSION_INFO.get(ext, {})
                # SVE寄存器大小可变
                reg_bits = info.get("reg_bits", 128)
                if ext in (SimdExtension.SVE, SimdExtension.SVE2):
                    reg_bits = 128  # 最小值

                cap = SimdCapability(
                    extension=ext,
                    supported=True,
                    register_size_bits=reg_bits,
                    register_size_bytes=reg_bits // 8,
                    vector_width_32bit=info.get("w32", 4),
                    vector_width_64bit=info.get("w64", 2),
                    vector_width_8bit=info.get("w8", 16),
                )
                extensions.append(cap)

        profile.extensions = extensions

        # ARM64默认支持NEON
        if arch == SimdArchitecture.ARM64 and not profile.supports(SimdExtension.NEON):
            info = self.EXTENSION_INFO[SimdExtension.NEON]
            profile.extensions.append(SimdCapability(
                extension=SimdExtension.NEON,
                supported=True,
                register_size_bits=128,
                register_size_bytes=16,
                vector_width_32bit=4,
                vector_width_64bit=2,
                vector_width_8bit=16,
            ))

        for cap in profile.extensions:
            if cap.supported:
                profile.max_register_bits = max(
                    profile.max_register_bits, cap.register_size_bits
                )
                profile.max_vector_width_32bit = max(
                    profile.max_vector_width_32bit, cap.vector_width_32bit
                )

        if profile.supports(SimdExtension.SVE2):
            profile.recommended_extension = "SVE2"
        elif profile.supports(SimdExtension.SVE):
            profile.recommended_extension = "SVE"
        elif profile.supports(SimdExtension.NEON):
            profile.recommended_extension = "NEON"
        else:
            profile.recommended_extension = "scalar"

        return profile

    def _detect_arm_linux(self) -> Set[str]:
        """检测Linux ARM特性"""
        features: Set[str] = set()

        # 方法1: /proc/cpuinfo
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text()
            # 搜索Features或features行
            for match in re.finditer(r"(?:Features|features|ASEs implemented)\s*:\s*(.+)", cpuinfo):
                features.update(match.group(1).split())
        except OSError:
            pass

        # 方法2: lscpu
        try:
            result = subprocess.run(
                ["lscpu"], capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "Flags:" in line or "flags:" in line:
                        flags = line.split(":")[1].strip().split()
                        features.update(flags)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return features

    def _detect_arm_macos(self) -> Set[str]:
        """检测macOS ARM特性"""
        features: Set[str] = set()

        try:
            result = subprocess.run(
                ["sysctl", "-a"], capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                output = result.stdout.lower()
                if "neon" in output or "hw.optional.neon" in output:
                    features.add("neon")
                if "fp16" in output or "hw.optional.armv8_fma" in output:
                    features.add("fp16")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Apple Silicon默认支持NEON
        if platform.machine() == "arm64":
            features.add("neon")
            features.add("fp16")

        return features

    def _detect_riscv(self) -> SimdProfile:
        """检测RISC-V V扩展"""
        profile = SimdProfile(architecture=SimdArchitecture.RISCV)

        try:
            cpuinfo = Path("/proc/cpuinfo").read_text()
            if "v" in cpuinfo.lower() or "vector" in cpuinfo.lower():
                profile.extensions.append(SimdCapability(
                    extension=SimdExtension.NONE,
                    supported=True,
                    register_size_bits=0,
                ))
                profile.recommended_extension = "RVV"
        except OSError:
            pass

        return profile

    def _detect_unknown(self) -> SimdProfile:
        """未知架构回退"""
        return SimdProfile(architecture=SimdArchitecture.UNKNOWN)

    def _get_x86_cpu_info(self) -> Tuple[str, str]:
        """获取x86 CPU厂商和型号"""
        vendor = ""
        model = ""

        if platform.system() == "Linux":
            try:
                cpuinfo = Path("/proc/cpuinfo").read_text()
                for line in cpuinfo.splitlines():
                    if line.startswith("vendor_id"):
                        vendor = line.split(":")[1].strip()
                    elif line.startswith("model name"):
                        model = line.split(":")[1].strip()
            except OSError:
                pass
        elif platform.system() == "Darwin":
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    model = result.stdout.strip()
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.vendor"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    vendor = result.stdout.strip()
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        elif platform.system() == "Windows":
            try:
                result = subprocess.run(
                    ["wmic", "cpu", "get", "Manufacturer,Name", "/format:csv"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        line = line.strip()
                        if not line or line.startswith("Node"):
                            continue
                        parts = [p.strip() for p in line.split(",") if p.strip()]
                        if len(parts) >= 2:
                            vendor = parts[0]
                            model = parts[1]
                        break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

        return vendor, model

    # ========================
    # 优化建议
    # ========================

    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """
        生成SIMD优化建议

        Returns:
            Dict: 优化建议
        """
        if not self._profile:
            return {"error": "SIMD profile not available"}

        recommendations = {
            "architecture": self._profile.architecture.value,
            "recommended_extension": self._profile.recommended_extension,
            "data_type_recommendations": {},
            "alignment_recommendation": 0,
            "loop_unroll_factor": 1,
        }

        ext = self._profile.recommended_extension

        if ext in ("AVX512",):
            recommendations["alignment_recommendation"] = 64
            recommendations["loop_unroll_factor"] = 8
            recommendations["data_type_recommendations"] = {
                "float32": "Use AVX512: 16 elements per vector",
                "float64": "Use AVX512: 8 elements per vector",
                "int8": "Use AVX512 VNNI: 64 elements per vector",
                "int32": "Use AVX512: 16 elements per vector",
            }
        elif ext == "AVX2":
            recommendations["alignment_recommendation"] = 32
            recommendations["loop_unroll_factor"] = 4
            recommendations["data_type_recommendations"] = {
                "float32": "Use AVX2: 8 elements per vector",
                "float64": "Use AVX2: 4 elements per vector",
                "int8": "Use AVX2: 32 elements per vector",
                "int32": "Use AVX2: 8 elements per vector",
            }
        elif ext == "AVX":
            recommendations["alignment_recommendation"] = 32
            recommendations["loop_unroll_factor"] = 4
            recommendations["data_type_recommendations"] = {
                "float32": "Use AVX: 8 elements per vector",
                "float64": "Use AVX: 4 elements per vector",
            }
        elif ext.startswith("SSE"):
            recommendations["alignment_recommendation"] = 16
            recommendations["loop_unroll_factor"] = 2
            recommendations["data_type_recommendations"] = {
                "float32": "Use SSE: 4 elements per vector",
                "float64": "Use SSE2: 2 elements per vector",
            }
        elif ext == "NEON":
            recommendations["alignment_recommendation"] = 16
            recommendations["loop_unroll_factor"] = 2
            recommendations["data_type_recommendations"] = {
                "float32": "Use NEON: 4 elements per vector",
                "int8": "Use NEON: 16 elements per vector",
            }
        elif ext == "SVE":
            recommendations["alignment_recommendation"] = 16
            recommendations["loop_unroll_factor"] = 2
            recommendations["data_type_recommendations"] = {
                "float32": "Use SVE: variable length vector",
                "int8": "Use SVE: variable length vector",
            }

        return recommendations

    def compute_vectorized_length(self, element_size: int) -> int:
        """
        计算向量化后的处理长度

        Args:
            element_size: 元素大小（字节）

        Returns:
            int: 每个向量寄存器可处理的元素数
        """
        if not self._profile:
            return 1

        if self._profile.max_register_bits == 0:
            return 1

        return max(1, self._profile.max_register_bits // (element_size * 8))

    def get_summary(self) -> Dict[str, Any]:
        """获取SIMD检测器的完整摘要"""
        return {
            "initialized": self._initialized,
            "platform": platform.system(),
            "machine": platform.machine(),
            "profile": self._profile.to_dict() if self._profile else None,
            "recommendations": self.get_optimization_recommendations(),
        }

    def __repr__(self) -> str:
        ext = self._profile.recommended_extension if self._profile else "unknown"
        arch = self._profile.architecture.value if self._profile else "unknown"
        return f"SimdDetection(arch={arch}, best={ext})"
