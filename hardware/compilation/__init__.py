"""
模型编译优化模块

模块路径: hardware/compilation/__init__.py

提供统一的模型编译优化接口，支持多种编译后端：
- ONNX Runtime: 图优化、量化、执行提供者选择
- TensorRT: 引擎构建、层融合、精度校准
- torch.compile: 动态编译、后端选择、自动调优
- TVM: 算子调度、自动调优、多目标编译
- XLA: JIT编译、图优化、TPU/GPU加速
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Type
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger(__name__)


class CompilationBackend(Enum):
    """支持的编译后端类型"""
    ONNX_RUNTIME = auto()
    TENSORRT = auto()
    TORCH_COMPILE = auto()
    TVM = auto()
    XLA = auto()


class PrecisionMode(Enum):
    """精度模式"""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    MIXED = "mixed"


class OptimizationLevel(Enum):
    """优化级别"""
    NONE = 0
    BASIC = 1
    EXTENDED = 2
    ALL = 3


@dataclass
class CompilationConfig:
    """编译配置基类"""
    backend: CompilationBackend = CompilationBackend.TORCH_COMPILE
    precision: PrecisionMode = PrecisionMode.FP32
    optimization_level: OptimizationLevel = OptimizationLevel.ALL
    enable_profiling: bool = False
    workspace_size: int = 1 << 30  # 1GB 默认工作空间
    device_id: int = 0
    log_verbose: bool = False
    extra_options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "backend": self.backend.name,
            "precision": self.precision.value,
            "optimization_level": self.optimization_level.value,
            "enable_profiling": self.enable_profiling,
            "workspace_size": self.workspace_size,
            "device_id": self.device_id,
            "log_verbose": self.log_verbose,
            "extra_options": self.extra_options,
        }


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    backend: str
    model_name: str
    latency_ms: float
    throughput_fps: float
    memory_usage_mb: float
    compile_time_s: float
    precision: str
    batch_size: int
    input_shape: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """生成摘要字符串"""
        return (
            f"[{self.backend}] {self.model_name} | "
            f"延迟: {self.latency_ms:.2f}ms | "
            f"吞吐: {self.throughput_fps:.1f} FPS | "
            f"显存: {self.memory_usage_mb:.1f}MB | "
            f"编译: {self.compile_time_s:.2f}s | "
            f"精度: {self.precision}"
        )


class CompilationError(Exception):
    """编译异常基类"""
    pass


class BackendNotAvailableError(CompilationError):
    """后端不可用"""
    pass


class ModelConversionError(CompilationError):
    """模型转换失败"""
    pass


class OptimizationError(CompilationError):
    """优化过程失败"""
    pass


class BaseCompiler:
    """编译器基类，定义统一的编译接口"""

    backend_name: str = "base"

    def __init__(self, config: Optional[CompilationConfig] = None):
        self._config = config or CompilationConfig()
        self._compiled = False
        self._model = None
        self._session = None
        self._logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )

    @property
    def config(self) -> CompilationConfig:
        return self._config

    @property
    def is_compiled(self) -> bool:
        return self._compiled

    @property
    def is_available(self) -> bool:
        """检查后端是否可用 - 默认尝试导入后端依赖"""
        try:
            self._check_dependencies()
            return True
        except (ImportError, OSError):
            return False

    def _check_dependencies(self) -> None:
        """检查后端依赖是否已安装，子类应重写此方法"""
        logger.debug(f"Backend {self._backend_type} dependency check passed (default)")

    def compile(
        self,
        model: Any,
        input_shapes: Optional[Dict[str, List[int]]] = None,
        **kwargs,
    ) -> Any:
        """编译模型

        Args:
            model: 待编译的模型（PyTorch/ONNX/TFLite等）
            input_shapes: 输入张量形状映射
            **kwargs: 额外编译参数

        Returns:
            编译后的模型或会话
        """
        logger.info(f"Compiling model with {self._backend_type} backend")
        self._model = model
        self._compiled = True
        logger.info("Model compilation completed (passthrough mode)")
        return model

    def inference(self, inputs: Any, **kwargs) -> Any:
        """执行推理

        Args:
            inputs: 输入数据
            **kwargs: 额外推理参数

        Returns:
            推理输出
        """
        if not self._compiled:
            logger.warning("Model not compiled, running raw inference")
        if self._model is not None and hasattr(self._model, '__call__'):
            return self._model(inputs, **kwargs)
        logger.warning("No compiled model available, returning inputs as-is")
        return inputs

    def benchmark(
        self,
        inputs: Any,
        num_warmup: int = 10,
        num_iterations: int = 100,
        **kwargs,
    ) -> BenchmarkResult:
        """运行基准测试

        Args:
            inputs: 输入数据
            num_warmup: 预热迭代次数
            num_iterations: 正式迭代次数
            **kwargs: 额外参数

        Returns:
            BenchmarkResult 基准测试结果
        """
        import time
        logger.info(f"Running benchmark: {num_warmup} warmup + {num_iterations} iterations")

        # Warmup
        for _ in range(num_warmup):
            self.inference(inputs, **kwargs)

        # Timed runs
        latencies = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            self.inference(inputs, **kwargs)
            latencies.append(time.perf_counter() - start)

        latencies.sort()
        mean_latency = sum(latencies) / len(latencies)
        p95_idx = int(len(latencies) * 0.95)
        p99_idx = int(len(latencies) * 0.99)

        return BenchmarkResult(
            backend=self._backend_type,
            mean_latency_ms=mean_latency * 1000,
            p95_latency_ms=latencies[p95_idx] * 1000,
            p99_latency_ms=latencies[p99_idx] * 1000,
            throughput=1.0 / mean_latency if mean_latency > 0 else 0.0,
            num_iterations=num_iterations,
        )

    def release(self) -> None:
        """释放编译资源"""
        self._compiled = False
        self._model = None
        self._session = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


def get_compiler(
    backend: Union[str, CompilationBackend],
    config: Optional[CompilationConfig] = None,
) -> BaseCompiler:
    """工厂函数：根据后端类型获取编译器实例

    Args:
        backend: 编译后端名称或枚举
        config: 编译配置

    Returns:
        对应的编译器实例

    Raises:
        BackendNotAvailableError: 后端不可用
        ValueError: 不支持的后端类型
    """
    if isinstance(backend, str):
        try:
            backend = CompilationBackend[backend.upper()]
        except KeyError:
            raise ValueError(
                f"不支持的后端: {backend}，"
                f"可选: {[b.name for b in CompilationBackend]}"
            )

    if config is not None:
        config.backend = backend

    compiler_map: Dict[CompilationBackend, str] = {
        CompilationBackend.ONNX_RUNTIME: ".onnx_runtime.ONNXRuntimeCompiler",
        CompilationBackend.TENSORRT: ".tensorrt.TensorRTCompiler",
        CompilationBackend.TORCH_COMPILE: ".torch_compile.TorchCompileCompiler",
        CompilationBackend.TVM: ".tvm.TVMCompiler",
        CompilationBackend.XLA: ".xla.XLACompiler",
    }

    module_path = compiler_map.get(backend)
    if module_path is None:
        raise ValueError(f"不支持的后端: {backend}")

    package = __name__
    parts = module_path.lstrip(".").split(".")
    module_name = parts[0]
    class_name = parts[1]

    try:
        module = __import__(f"{package}.{module_name}", fromlist=[class_name])
        compiler_cls = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise BackendNotAvailableError(
            f"无法加载后端 {backend.name}: {e}"
        ) from e

    return compiler_cls(config=config)


def list_available_backends() -> List[str]:
    """列出所有可用的编译后端

    Returns:
        可用后端名称列表
    """
    available = []
    for backend in CompilationBackend:
        try:
            compiler = get_compiler(backend)
            if compiler.is_available:
                available.append(backend.name)
        except (BackendNotAvailableError, ValueError):
            continue
    return available


def auto_select_backend(
    model: Any = None,
    hardware: Optional[str] = None,
    precision: Optional[PrecisionMode] = None,
) -> CompilationBackend:
    """根据环境和模型自动选择最佳编译后端

    Args:
        model: 待编译模型
        hardware: 硬件类型（cuda/rocm/cpu/tpu）
        precision: 精度要求

    Returns:
        推荐的编译后端
    """
    import platform

    available = list_available_backends()

    if not available:
        raise BackendNotAvailableError("没有可用的编译后端")

    # 检测硬件
    if hardware is None:
        try:
            import torch
            if torch.cuda.is_available():
                hardware = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                hardware = "mps"
            else:
                hardware = "cpu"
        except ImportError:
            hardware = "cpu"

    # 根据硬件和精度选择
    priority_map = {
        "cuda": [
            CompilationBackend.TENSORRT,
            CompilationBackend.TORCH_COMPILE,
            CompilationBackend.ONNX_RUNTIME,
            CompilationBackend.XLA,
            CompilationBackend.TVM,
        ],
        "rocm": [
            CompilationBackend.TORCH_COMPILE,
            CompilationBackend.TVM,
            CompilationBackend.ONNX_RUNTIME,
        ],
        "tpu": [
            CompilationBackend.XLA,
            CompilationBackend.TORCH_COMPILE,
        ],
        "cpu": [
            CompilationBackend.ONNX_RUNTIME,
            CompilationBackend.TORCH_COMPILE,
            CompilationBackend.TVM,
            CompilationBackend.XLA,
        ],
        "mps": [
            CompilationBackend.TORCH_COMPILE,
            CompilationBackend.ONNX_RUNTIME,
        ],
    }

    priority = priority_map.get(hardware, priority_map["cpu"])

    for backend in priority:
        if backend.name in available:
            return backend

    return CompilationBackend[available[0]]


__all__ = [
    "CompilationBackend",
    "PrecisionMode",
    "OptimizationLevel",
    "CompilationConfig",
    "BenchmarkResult",
    "CompilationError",
    "BackendNotAvailableError",
    "ModelConversionError",
    "OptimizationError",
    "BaseCompiler",
    "get_compiler",
    "list_available_backends",
    "auto_select_backend",
]
