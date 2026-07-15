"""
XLA 编译优化模块

模块路径: hardware/compilation/xla.py

功能：
- PyTorch/XLA JIT 编译（PJRT 后端）
- 图捕获与 XLA 编译
- XLA 图优化（算子融合、内存优化、布局优化）
- TPU/GPU/CPU 多后端支持
- XLA Profiler 集成
- 编译缓存管理
- 动态形状与 Padding 策略
- 性能基准测试
"""

import os
import time
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from . import (
    BaseCompiler,
    CompilationConfig,
    CompilationBackend,
    PrecisionMode,
    OptimizationLevel,
    BenchmarkResult,
    CompilationError,
    BackendNotAvailableError,
    ModelConversionError,
    OptimizationError,
)

logger = logging.getLogger(__name__)

# 检测 PyTorch/XLA 可用性
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.debug.metrics as met
    XLA_AVAILABLE = True
    _XLA_VERSION = getattr(torch_xla, "__version__", "unknown")
except ImportError:
    XLA_AVAILABLE = False
    _XLA_VERSION = None


class XLABackend(Enum):
    """XLA 后端类型"""
    TPU = "TPU"
    CUDA = "CUDA"
    CPU = "CPU"
    NEURON = "NEURON"
    GPU = "GPU"


class XLACompileMode(Enum):
    """XLA 编译模式"""
    DEFAULT = "default"
    LEAN = "lean"
    FORCE_COMPILE = "force_compile"


@dataclass
class XLAConfig(CompilationConfig):
    """XLA 专用配置"""
    backend: CompilationBackend = CompilationBackend.XLA
    xla_device: str = "TPU"  # TPU / CUDA / CPU / GPU
    compile_mode: str = "default"  # default / lean / force_compile
    full_graph: bool = True
    dynamic_shapes: bool = False
    padding: bool = False
    pad_to_multiple: int = 128
    # 编译缓存
    cache_dir: Optional[str] = None
    cache_size_gb: int = 8
    persistent_cache: bool = True
    # 精度
    autocast_enabled: bool = False
    autocast_dtype: str = "bfloat16"
    # 性能
    enable_profiling: bool = False
    profile_dir: Optional[str] = None
    # 图优化
    enable_layout_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_fusion: bool = True
    # 调试
    print_graphs: bool = False
    print_hlo: bool = False
    xla_flags: Dict[str, str] = field(default_factory=dict)


@dataclass
class XLAMetrics:
    """XLA 性能指标"""
    compile_time_ms: float = 0.0
    graph_count: int = 0
    total_op_count: int = 0
    memory_allocated_mb: float = 0.0
    transfer_time_ms: float = 0.0
    execute_time_ms: float = 0.0
    counter_values: Dict[str, float] = field(default_factory=dict)


class XLACompiler(BaseCompiler):
    """XLA 编译优化器

    提供完整的 XLA 编译优化流水线：
    1. XLA 设备初始化与配置
    2. 模型到 XLA 设备的放置
    3. JIT 编译（torch_xla.compile 或自动编译）
    4. 图捕获与 XLA IR 生成
    5. XLA 图优化（算子融合、布局优化、内存规划）
    6. TPU/GPU/CPU 多后端推理
    7. 编译缓存管理
    8. 性能分析与基准测试
    """

    backend_name = "xla"

    def __init__(self, config: Optional[XLAConfig] = None):
        super().__init__(config or XLAConfig())
        self._compiled_model: Optional[Any] = None
        self._original_model: Optional[Any] = None
        self._device: Optional[Any] = None
        self._compile_fn: Optional[Callable] = None
        self._metrics: Optional[XLAMetrics] = None
        self._cache_dir: Optional[str] = None
        self._profile_dir: Optional[str] = None

    @property
    def is_available(self) -> bool:
        """检查 XLA 是否可用"""
        return XLA_AVAILABLE

    @property
    def xla_device(self) -> Any:
        """获取 XLA 设备"""
        if self._device is not None:
            return self._device

        if not XLA_AVAILABLE:
            raise BackendNotAvailableError("torch_xla 未安装")

        cfg = self._config
        device_type = cfg.xla_device if isinstance(cfg, XLAConfig) else "TPU"

        try:
            self._device = xm.xla_device(device_type)
        except RuntimeError as e:
            raise BackendNotAvailableError(
                f"XLA 设备 {device_type} 不可用: {e}"
            ) from e

        return self._device

    def _setup_xla_flags(self) -> None:
        """配置 XLA 环境变量和标志

        设置编译缓存、图优化、性能分析等 XLA 标志。
        """
        cfg = self._config
        if not isinstance(cfg, XLAConfig):
            return

        # 编译缓存
        if cfg.cache_dir:
            self._cache_dir = cfg.cache_dir
            os.makedirs(self._cache_dir, exist_ok=True)
            os.environ["XLA_HLO_CACHE_DIR"] = self._cache_dir

        if cfg.persistent_cache:
            os.environ.setdefault("XLA_PERSISTENT_CACHE_PATH", "/tmp/xla_cache")

        # 图优化标志
        if not cfg.enable_fusion:
            os.environ["XLA_DISABLE_FUSION"] = "1"

        if not cfg.enable_layout_optimization:
            os.environ["XLA_FLAGS"] = (
                os.environ.get("XLA_FLAGS", "")
                + " --xla_disable_hlo_passes=layout_assignment"
            )

        # 性能分析
        if cfg.enable_profiling:
            self._profile_dir = cfg.profile_dir or tempfile.mkdtemp(
                prefix="xla_profile_"
            )
            os.makedirs(self._profile_dir, exist_ok=True)
            os.environ["XLA_HLO_PROFILE"] = "1"

        # 自定义标志
        for key, value in cfg.xla_flags.items():
            os.environ[key] = value

    def _prepare_model(
        self,
        model: Any,
    ) -> Any:
        """准备模型（设备放置、精度转换、eval模式）

        Args:
            model: PyTorch nn.Module

        Returns:
            准备好的模型
        """
        if not TORCH_AVAILABLE:
            raise BackendNotAvailableError("PyTorch 未安装")

        self._original_model = model
        device = self.xla_device

        # 设置为评估模式
        if isinstance(model, nn.Module):
            model.eval()

        # 设备放置
        model = model.to(device)

        # 精度转换
        cfg = self._config
        if isinstance(cfg, XLAConfig) and cfg.autocast_enabled:
            if cfg.autocast_dtype == "bfloat16":
                model = model.to(torch.bfloat16)
            elif cfg.autocast_dtype == "float16":
                model = model.to(torch.float16)

        return model

    def _apply_padding(
        self,
        inputs: Any,
        pad_to: int,
    ) -> Tuple[Any, Any]:
        """对输入应用 Padding 以优化 XLA 编译

        XLA 在处理固定大小的张量时性能更优。
        将动态维度 Pad 到指定倍数。

        Args:
            inputs: 输入张量或张量列表
            pad_to: Pad 到的倍数

        Returns:
            (padded_inputs, original_shapes)
        """
        if isinstance(inputs, torch.Tensor):
            original_shape = inputs.shape
            pad_dims = []
            for dim_size in inputs.shape:
                pad_amount = (pad_to - dim_size % pad_to) % pad_to
                pad_dims.append((0, pad_amount))

            if any(p > 0 for p, _ in pad_dims):
                padded = torch.nn.functional.pad(
                    inputs, [p for pair in reversed(pad_dims[1:]) for p in pair]
                )
                return padded, original_shape
            return inputs, original_shape

        elif isinstance(inputs, (tuple, list)):
            padded_list = []
            shapes = []
            for inp in inputs:
                padded, shape = self._apply_padding(inp, pad_to)
                padded_list.append(padded)
                shapes.append(shape)
            return type(inputs)(padded_list), shapes

        return inputs, None

    def _remove_padding(
        self,
        outputs: Any,
        original_shapes: Any,
    ) -> Any:
        """移除 Padding

        Args:
            outputs: 带有 Padding 的输出
            original_shapes: 原始形状

        Returns:
            去除 Padding 后的输出
        """
        if original_shapes is None:
            return outputs

        if isinstance(outputs, torch.Tensor) and isinstance(original_shapes, torch.Size):
            return outputs[:original_shapes[0]]
        elif isinstance(outputs, (tuple, list)) and isinstance(original_shapes, (tuple, list)):
            result = []
            for out, shape in zip(outputs, original_shapes):
                if isinstance(out, torch.Tensor) and isinstance(shape, torch.Size):
                    slices = tuple(slice(0, s) for s in shape)
                    result.append(out[slices])
                else:
                    result.append(out)
            return type(outputs)(result)

        return outputs

    def compile(
        self,
        model: Any,
        input_shapes: Optional[Dict[str, List[int]]] = None,
        **kwargs,
    ) -> Any:
        """使用 XLA 编译模型

        Args:
            model: PyTorch nn.Module
            input_shapes: 输入形状映射
            **kwargs: 额外参数
                - compile_mode: 编译模式
                - full_graph: 是否全图编译
                - dynamic_shapes: 是否启用动态形状
                - sample_inputs: 示例输入（用于预热）

        Returns:
            编译后的模型
        """
        if not XLA_AVAILABLE:
            raise BackendNotAvailableError("torch_xla 未安装")

        start_time = time.time()

        # 设置 XLA 标志
        self._setup_xla_flags()

        # 准备模型
        prepared_model = self._prepare_model(model)

        # 确定编译模式
        cfg = self._config
        compile_mode = kwargs.get(
            "compile_mode",
            cfg.compile_mode if isinstance(cfg, XLAConfig) else "default",
        )

        # 使用 torch_xla.compile 进行显式编译
        use_explicit_compile = hasattr(torch_xla, "compile")

        if use_explicit_compile:
            full_graph = kwargs.get(
                "full_graph",
                cfg.full_graph if isinstance(cfg, XLAConfig) else True,
            )

            try:
                if compile_mode == "lean":
                    compiled_model = torch_xla.compile(
                        prepared_model,
                        backend="lean",
                    )
                elif compile_mode == "force_compile":
                    compiled_model = torch_xla.compile(
                        prepared_model,
                        full_graph=True,
                    )
                else:
                    compiled_model = torch_xla.compile(
                        prepared_model,
                        full_graph=full_graph,
                    )
            except Exception as e:
                logger.warning(f"torch_xla.compile 失败，回退到自动编译: {e}")
                compiled_model = prepared_model
                use_explicit_compile = False
        else:
            compiled_model = prepared_model

        self._compiled_model = compiled_model

        # 预热编译（触发 XLA 图追踪）
        sample_inputs = kwargs.get("sample_inputs")
        if sample_inputs is None and input_shapes is not None:
            sample_inputs = self._create_sample_inputs(input_shapes)

        if sample_inputs is not None:
            try:
                with torch.no_grad():
                    if isinstance(sample_inputs, (tuple, list)):
                        outputs = compiled_model(*sample_inputs)
                    elif isinstance(sample_inputs, dict):
                        outputs = compiled_model(**sample_inputs)
                    else:
                        outputs = compiled_model(sample_inputs)

                    # 标记步骤完成（触发 XLA 执行）
                    xm.mark_step()

                logger.info("XLA 编译预热完成")
            except Exception as e:
                logger.warning(f"XLA 编译预热失败（非致命）: {e}")

        compile_time = time.time() - start_time

        self._compiled = True
        self._model = model

        # 收集指标
        self._metrics = self._collect_metrics(compile_time)

        logger.info(
            f"XLA 编译完成 | "
            f"设备: {self.xla_device} | "
            f"耗时: {compile_time:.3f}s | "
            f"图数量: {self._metrics.graph_count}"
        )

        return compiled_model

    def _create_sample_inputs(
        self,
        input_shapes: Dict[str, List[int]],
    ) -> Tuple[Any, ...]:
        """根据形状创建示例输入

        Args:
            input_shapes: 输入形状映射

        Returns:
            示例输入元组
        """
        if not TORCH_AVAILABLE:
            raise BackendNotAvailableError("PyTorch 未安装")

        device = self.xla_device
        inputs = []

        cfg = self._config
        dtype = torch.float32
        if isinstance(cfg, XLAConfig) and cfg.autocast_enabled:
            if cfg.autocast_dtype == "bfloat16":
                dtype = torch.bfloat16
            elif cfg.autocast_dtype == "float16":
                dtype = torch.float16

        for name, shape in input_shapes.items():
            tensor = torch.randn(*shape, dtype=dtype, device=device)
            inputs.append(tensor)

        return tuple(inputs)

    def _collect_metrics(self, compile_time: float) -> XLAMetrics:
        """收集 XLA 性能指标

        Args:
            compile_time: 编译耗时

        Returns:
            XLAMetrics
        """
        metrics = XLAMetrics(
            compile_time_ms=compile_time * 1000,
        )

        if XLA_AVAILABLE:
            try:
                counter_values = met.counter_names()
                for name in counter_values:
                    try:
                        metrics.counter_values[name] = met.value(name)
                    except Exception:
                        pass

                metrics.graph_count = metrics.counter_values.get(
                    "GraphCount", 0
                )
                metrics.compile_time_ms = metrics.counter_values.get(
                    "CompileTime", compile_time * 1000
                )
            except Exception:
                pass

        return metrics

    def inference(self, inputs: Any, **kwargs) -> Any:
        """执行推理

        Args:
            inputs: 输入数据
                - torch.Tensor: 单输入
                - tuple/list: 多输入
                - dict: 命名输入
                - numpy.ndarray: 自动转换
            **kwargs: 额外参数

        Returns:
            推理输出
        """
        if not XLA_AVAILABLE:
            raise BackendNotAvailableError("torch_xla 未安装")

        model = self._compiled_model
        if model is None:
            raise CompilationError("模型未编译，请先调用 compile()")

        device = self.xla_device

        # 转换 numpy 到 tensor
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).to(device)
        elif isinstance(inputs, dict):
            inputs = {
                k: (torch.from_numpy(v).to(device)
                    if isinstance(v, np.ndarray) else v)
                for k, v in inputs.items()
            }
        elif isinstance(inputs, (list, tuple)):
            inputs = type(inputs)(
                torch.from_numpy(x).to(device)
                if isinstance(x, np.ndarray) else x
                for x in inputs
            )

        # Padding
        original_shapes = None
        cfg = self._config
        if isinstance(cfg, XLAConfig) and cfg.padding:
            inputs, original_shapes = self._apply_padding(
                inputs, cfg.pad_to_multiple
            )

        # 自动混合精度
        use_amp = False
        if isinstance(cfg, XLAConfig) and cfg.autocast_enabled:
            use_amp = True

        with torch.no_grad():
            if use_amp:
                with torch.autocast(
                    device_type="xla",
                    dtype=(
                        torch.bfloat16
                        if cfg.autocast_dtype == "bfloat16"
                        else torch.float16
                    ),
                ):
                    if isinstance(inputs, dict):
                        output = model(**inputs)
                    elif isinstance(inputs, (tuple, list)):
                        output = model(*inputs)
                    else:
                        output = model(inputs)
            else:
                if isinstance(inputs, dict):
                    output = model(**inputs)
                elif isinstance(inputs, (tuple, list)):
                    output = model(*inputs)
                else:
                    output = model(inputs)

            # 标记步骤完成（触发 XLA 执行）
            xm.mark_step()

        # 移除 Padding
        if original_shapes is not None:
            output = self._remove_padding(output, original_shapes)

        # 转换回 numpy
        if isinstance(output, torch.Tensor):
            return output.cpu().numpy()
        elif isinstance(output, (tuple, list)):
            return type(output)(
                o.cpu().numpy() if isinstance(o, torch.Tensor) else o
                for o in output
            )
        return output

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
            num_warmup: 预热次数
            num_iterations: 迭代次数

        Returns:
            BenchmarkResult
        """
        if not self._compiled:
            raise CompilationError("请先调用 compile() 编译模型")

        if not XLA_AVAILABLE:
            raise BackendNotAvailableError("torch_xla 未安装")

        import gc

        # 预热
        for _ in range(num_warmup):
            self.inference(inputs)
        xm.mark_step()
        xm.wait_device_ops()
        gc.collect()

        # 重置计数器
        try:
            met.clear_counters()
        except Exception:
            pass

        # 正式测试
        latencies = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            self.inference(inputs)
            xm.wait_device_ops()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        latencies.sort()
        avg_latency = sum(latencies) / len(latencies)
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]

        batch_size = 1
        if isinstance(inputs, (torch.Tensor, np.ndarray)):
            batch_size = inputs.shape[0]
        elif isinstance(inputs, dict):
            for v in inputs.values():
                if isinstance(v, (torch.Tensor, np.ndarray)):
                    batch_size = v.shape[0]
                    break

        throughput = batch_size / (avg_latency / 1000)

        # 收集 XLA 指标
        metrics = self._collect_metrics(0.0)

        model_name = "unknown"
        if isinstance(self._original_model, nn.Module):
            model_name = self._original_model.__class__.__name__

        return BenchmarkResult(
            backend=self.backend_name,
            model_name=model_name,
            latency_ms=avg_latency,
            throughput_fps=throughput,
            memory_usage_mb=metrics.memory_allocated_mb,
            compile_time_s=metrics.compile_time_ms / 1000,
            precision=self._config.precision.value,
            batch_size=batch_size,
            metadata={
                "p50_ms": p50,
                "p95_ms": p95,
                "p99_ms": p99,
                "num_iterations": num_iterations,
                "device": str(self.xla_device),
                "graph_count": metrics.graph_count,
                "xla_counters": metrics.counter_values,
            },
        )

    def get_xla_hlo_graph(self) -> Optional[str]:
        """获取 XLA HLO 图表示

        Returns:
            HLO 图文本或 None
        """
        if not XLA_AVAILABLE:
            return None

        try:
            graphs = met.get_xla_hlo_graph()
            return graphs
        except Exception as e:
            logger.debug(f"获取 HLO 图失败: {e}")
            return None

    def get_metrics(self) -> Optional[XLAMetrics]:
        """获取 XLA 性能指标

        Returns:
            XLAMetrics 或 None
        """
        return self._metrics

    def get_device_info(self) -> Dict[str, Any]:
        """获取 XLA 设备信息

        Returns:
            设备信息字典
        """
        if not XLA_AVAILABLE:
            return {}

        info = {
            "device": str(self.xla_device),
            "compiled": self._compiled,
        }

        try:
            info["devices"] = xm.get_xla_supported_devices()
            info["num_devices"] = xm.xrt_world_size()
        except Exception:
            pass

        return info

    def save_profile(self, output_dir: Optional[str] = None) -> Optional[str]:
        """保存 XLA 性能分析数据

        Args:
            output_dir: 输出目录

        Returns:
            分析数据目录路径
        """
        if not XLA_AVAILABLE:
            return None

        profile_dir = output_dir or self._profile_dir
        if profile_dir is None:
            return None

        try:
            os.makedirs(profile_dir, exist_ok=True)

            # 保存计数器
            counter_data = {}
            try:
                for name in met.counter_names():
                    counter_data[name] = met.value(name)
            except Exception:
                pass

            counter_file = os.path.join(profile_dir, "counters.json")
            with open(counter_file, "w") as f:
                json.dump(counter_data, f, indent=2)

            # 保存 HLO 图
            hlo_graph = self.get_xla_hlo_graph()
            if hlo_graph:
                hlo_file = os.path.join(profile_dir, "graph.hlo")
                with open(hlo_file, "w") as f:
                    f.write(hlo_graph)

            logger.info(f"XLA 分析数据已保存: {profile_dir}")
            return profile_dir

        except Exception as e:
            logger.warning(f"保存分析数据失败: {e}")
            return None

    def release(self) -> None:
        """释放资源"""
        self._compiled_model = None
        self._original_model = None
        self._device = None
        self._compile_fn = None
        self._metrics = None
        self._cache_dir = None
        self._profile_dir = None
        self._compiled = False
        self._model = None


__all__ = [
    "XLACompiler",
    "XLAConfig",
    "XLAMetrics",
    "XLABackend",
    "XLACompileMode",
    "XLA_AVAILABLE",
    "TORCH_AVAILABLE",
]
