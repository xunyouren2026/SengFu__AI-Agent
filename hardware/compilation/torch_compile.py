"""
torch.compile 动态编译优化模块

模块路径: hardware/compilation/torch_compile.py

功能：
- torch.compile 动态编译加速
- 后端选择（inductor/cudagraphs/onnxrt/tensorrt）
- 编译模式配置（default/reduce-overhead/max-autotune）
- 动态形状支持
- 自定义编译选项（算子融合、内存规划）
- 编译缓存管理
- 模型导出（导出为TorchScript/ONNX）
- 性能基准测试
"""

import os
import time
import logging
import hashlib
import json
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

# 检测 PyTorch 可用性
try:
    import torch
    import torch.nn as nn
    import torch.utils._pytree as pytree
    TORCH_AVAILABLE = True
    _TORCH_VERSION = torch.__version__

    # 检测 torch.compile 可用性 (PyTorch >= 2.0)
    TORCH_COMPILE_AVAILABLE = hasattr(torch, "compile")
    if TORCH_COMPILE_AVAILABLE:
        from torch._dynamo import reset as dynamo_reset
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_COMPILE_AVAILABLE = False
    _TORCH_VERSION = None


class CompileBackend(Enum):
    """torch.compile 后端"""
    DEFAULT = "inductor"
    INDUCTOR = "inductor"
    CUDAGRAPHS = "cudagraphs"
    AOT_EAGER = "aot_eager"
    AOT_TS = "aot_ts"
    ONNXRT = "onnxrt"
    TENSORRT = "tensorrt"
    TVM = "tvm"
    EAGER = "eager"


class CompileMode(Enum):
    """编译模式"""
    DEFAULT = "default"
    REDUCE_OVERHEAD = "reduce-overhead"
    MAX_AUTOTUNE = "max-autotune"


class DynamicShapeStrategy(Enum):
    """动态形状策略"""
    AUTO = "auto"
    STATIC = "static"
    DYNAMIC = "dynamic"
    CONSECUTIVE = "consecutive"
    SAME = "same"


@dataclass
class TorchCompileConfig(CompilationConfig):
    """torch.compile 专用配置"""
    backend: CompilationBackend = CompilationBackend.TORCH_COMPILE
    compile_backend: str = "inductor"
    compile_mode: str = "default"
    fullgraph: bool = False
    dynamic: Union[bool, str] = False
    guard_export_path: Optional[str] = None
    disable: bool = False
    enable_cpp_wrapper: bool = True
    save_artifacts: bool = False
    artifacts_path: Optional[str] = None
    explain: bool = False
    print_graphs: bool = False
    # Inductor 专用选项
    inductor_options: Dict[str, Any] = field(default_factory=dict)
    # 内存规划
    memory_planning: bool = True
    shape_padding: bool = True
    # 编译缓存
    cache_dir: Optional[str] = None
    cache_key: Optional[str] = None
    # 精度相关
    autocast_enabled: bool = False
    autocast_dtype: str = "float16"
    # 导出配置
    export_format: Optional[str] = None  # torchscript / onnx


@dataclass
class CompileDiagnostics:
    """编译诊断信息"""
    compile_time_s: float
    graph_count: int
    graph_break_count: int
    recompilation_count: int
    backend_used: str
    mode_used: str
    memory_estimation_mb: float = 0.0
    warnings: List[str] = field(default_factory=list)
    graph_break_reasons: List[str] = field(default_factory=list)


class TorchCompileCompiler(BaseCompiler):
    """torch.compile 动态编译优化器

    提供完整的 torch.compile 编译优化流水线：
    1. 模型预处理（精度转换、设备放置）
    2. torch.compile 编译配置
    3. 动态形状策略
    4. 后端选择（inductor/cudagraphs/onnxrt等）
    5. 编译缓存管理
    6. 推理执行
    7. 模型导出（TorchScript/ONNX）
    8. 性能基准测试与诊断
    """

    backend_name = "torch_compile"

    def __init__(self, config: Optional[TorchCompileConfig] = None):
        super().__init__(config or TorchCompileConfig())
        self._compiled_model: Optional[Any] = None
        self._original_model: Optional[Any] = None
        self._device: Optional[str] = None
        self._compile_time: float = 0.0
        self._diagnostics: Optional[CompileDiagnostics] = None
        self._guard_manager: Optional[Any] = None
        self._cache_dir: Optional[str] = None

    @property
    def is_available(self) -> bool:
        """检查 torch.compile 是否可用"""
        return TORCH_COMPILE_AVAILABLE

    @property
    def compiled_model(self) -> Any:
        """获取编译后的模型"""
        if self._compiled_model is None:
            raise CompilationError("模型未编译，请先调用 compile()")
        return self._compiled_model

    @property
    def device(self) -> str:
        """获取计算设备"""
        if self._device is None:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self._device = f"cuda:{self._config.device_id}"
            else:
                self._device = "cpu"
        return self._device

    def _prepare_model(
        self,
        model: Any,
        device: Optional[str] = None,
    ) -> Any:
        """准备模型（设备放置、精度转换、eval模式）

        Args:
            model: PyTorch 模型
            device: 目标设备

        Returns:
            准备好的模型
        """
        if not TORCH_AVAILABLE:
            raise BackendNotAvailableError("PyTorch 未安装")

        self._original_model = model
        target_device = device or self.device

        # 设置为评估模式
        if isinstance(model, nn.Module):
            model.eval()

        # 设备放置
        model = model.to(target_device)

        # 精度转换
        cfg = self._config
        if isinstance(cfg, TorchCompileConfig) and cfg.autocast_enabled:
            if cfg.autocast_dtype == "bfloat16":
                model = model.to(torch.bfloat16)
            elif cfg.autocast_dtype == "float16":
                model = model.to(torch.float16)

        return model

    def _build_compile_options(self) -> Dict[str, Any]:
        """构建 torch.compile 选项

        Returns:
            编译选项字典
        """
        cfg = self._config
        options = {}

        if isinstance(cfg, TorchCompileConfig):
            options["backend"] = cfg.compile_backend
            options["mode"] = cfg.compile_mode
            options["fullgraph"] = cfg.fullgraph
            options["dynamic"] = cfg.dynamic
            options["disable"] = cfg.disable

            if cfg.guard_export_path:
                options["guard_export_path"] = cfg.guard_export_path

            if cfg.save_artifacts:
                artifacts_path = cfg.artifacts_path or tempfile.mkdtemp(
                    prefix="torch_compile_"
                )
                os.makedirs(artifacts_path, exist_ok=True)
                options["save_artifacts"] = artifacts_path
                self._cache_dir = artifacts_path

            # Inductor 选项
            if cfg.compile_backend in ("inductor", "cudagraphs"):
                inductor_opts = {}
                if cfg.memory_planning:
                    inductor_opts["memory_planning"] = True
                if cfg.shape_padding:
                    inductor_opts["shape_padding"] = True
                inductor_opts.update(cfg.inductor_options)
                if inductor_opts:
                    options["options"] = inductor_opts

        return options

    def _setup_cache(self, model: Any) -> None:
        """设置编译缓存

        Args:
            model: PyTorch 模型
        """
        cfg = self._config
        if not isinstance(cfg, TorchCompileConfig):
            return

        if cfg.cache_dir:
            self._cache_dir = cfg.cache_dir
            os.makedirs(self._cache_dir, exist_ok=True)

        # 设置环境变量
        if self._cache_dir:
            os.environ["TORCH_COMPILE_CACHE_DIR"] = self._cache_dir

    def _collect_diagnostics(
        self,
        compile_time: float,
        backend: str,
        mode: str,
    ) -> CompileDiagnostics:
        """收集编译诊断信息

        Args:
            compile_time: 编译耗时
            backend: 使用的后端
            mode: 使用的模式

        Returns:
            CompileDiagnostics
        """
        diagnostics = CompileDiagnostics(
            compile_time_s=compile_time,
            graph_count=0,
            graph_break_count=0,
            recompilation_count=0,
            backend_used=backend,
            mode_used=mode,
        )

        if TORCH_AVAILABLE:
            try:
                from torch._dynamo import compile_counters
                counters = compile_counters()
                if counters:
                    diagnostics.graph_count = counters.get("graph", 0)
                    diagnostics.graph_break_count = counters.get(
                        "graph_break", 0
                    )
            except Exception:
                pass

        return diagnostics

    def compile(
        self,
        model: Any,
        input_shapes: Optional[Dict[str, List[int]]] = None,
        **kwargs,
    ) -> Any:
        """使用 torch.compile 编译模型

        Args:
            model: PyTorch nn.Module
            input_shapes: 输入形状映射（用于预热编译）
            **kwargs: 额外参数
                - backend: 编译后端
                - mode: 编译模式
                - dynamic: 动态形状策略
                - sample_inputs: 示例输入（用于预热）

        Returns:
            编译后的模型
        """
        if not TORCH_COMPILE_AVAILABLE:
            raise BackendNotAvailableError(
                "torch.compile 不可用，需要 PyTorch >= 2.0"
            )

        start_time = time.time()

        # 准备模型
        device = kwargs.get("device")
        prepared_model = self._prepare_model(model, device)

        # 设置缓存
        self._setup_cache(prepared_model)

        # 构建编译选项
        options = self._build_compile_options()

        # 覆盖选项
        for key in ("backend", "mode", "fullgraph", "dynamic"):
            if key in kwargs:
                options[key] = kwargs[key]

        # 重置 Dynamo 缓存
        try:
            dynamo_reset()
        except Exception:
            pass

        # 执行编译
        logger.info(
            f"开始 torch.compile | "
            f"后端: {options.get('backend', 'inductor')} | "
            f"模式: {options.get('mode', 'default')}"
        )

        try:
            compiled_model = torch.compile(prepared_model, **options)
        except Exception as e:
            raise CompilationError(f"torch.compile 编译失败: {e}") from e

        self._compiled_model = compiled_model
        compile_time = time.time() - start_time

        # 预热编译（触发实际编译）
        sample_inputs = kwargs.get("sample_inputs")
        if sample_inputs is None and input_shapes is not None:
            sample_inputs = self._create_sample_inputs(input_shapes)

        if sample_inputs is not None:
            try:
                with torch.no_grad():
                    if isinstance(sample_inputs, (tuple, list)):
                        compiled_model(*sample_inputs)
                    elif isinstance(sample_inputs, dict):
                        compiled_model(**sample_inputs)
                    else:
                        compiled_model(sample_inputs)
                logger.info("编译预热完成")
            except Exception as e:
                logger.warning(f"编译预热失败（非致命）: {e}")

        self._compiled = True
        self._model = model

        # 收集诊断信息
        backend = options.get("backend", "inductor")
        mode = options.get("mode", "default")
        self._diagnostics = self._collect_diagnostics(
            compile_time, backend, mode
        )

        logger.info(
            f"torch.compile 完成 | "
            f"耗时: {compile_time:.3f}s | "
            f"图断点: {self._diagnostics.graph_break_count}"
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

        device = self.device
        inputs = []
        for name, shape in input_shapes.items():
            dtype = torch.float32
            tensor = torch.randn(*shape, dtype=dtype, device=device)
            inputs.append(tensor)

        return tuple(inputs)

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
        if not TORCH_AVAILABLE:
            raise BackendNotAvailableError("PyTorch 未安装")

        model = self.compiled_model

        # 转换 numpy 到 tensor
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).to(self.device)
        elif isinstance(inputs, dict):
            inputs = {
                k: (torch.from_numpy(v).to(self.device)
                    if isinstance(v, np.ndarray) else v)
                for k, v in inputs.items()
            }
        elif isinstance(inputs, (list, tuple)):
            inputs = type(inputs)(
                torch.from_numpy(x).to(self.device)
                if isinstance(x, np.ndarray) else x
                for x in inputs
            )

        # 自动混合精度
        use_amp = False
        cfg = self._config
        if isinstance(cfg, TorchCompileConfig) and cfg.autocast_enabled:
            use_amp = True

        with torch.no_grad():
            if use_amp:
                with torch.autocast(
                    device_type=self.device.replace(":", ""),
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

        # 转换回 numpy（如果输入是 numpy）
        if isinstance(output, torch.Tensor):
            return output.cpu().numpy()
        elif isinstance(output, (tuple, list)):
            return type(output)(
                o.cpu().numpy() if isinstance(o, torch.Tensor) else o
                for o in output
            )
        return output

    def export_model(
        self,
        output_path: str,
        format: str = "torchscript",
        input_shapes: Optional[Dict[str, List[int]]] = None,
        sample_inputs: Optional[Any] = None,
        **kwargs,
    ) -> str:
        """导出编译后的模型

        Args:
            output_path: 输出路径
            format: 导出格式 (torchscript/onnx)
            input_shapes: 输入形状
            sample_inputs: 示例输入
            **kwargs: 额外导出参数

        Returns:
            导出文件路径
        """
        if not TORCH_AVAILABLE:
            raise BackendNotAvailableError("PyTorch 未安装")

        if self._original_model is None:
            raise CompilationError("没有可导出的模型")

        model = self._original_model
        model.eval()

        # 准备示例输入
        if sample_inputs is None and input_shapes is not None:
            sample_inputs = self._create_sample_inputs(input_shapes)

        if sample_inputs is None:
            raise ValueError("导出需要提供 input_shapes 或 sample_inputs")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if format == "torchscript":
            return self._export_torchscript(
                model, output_path, sample_inputs, **kwargs
            )
        elif format == "onnx":
            return self._export_onnx(
                model, output_path, sample_inputs, **kwargs
            )
        else:
            raise ValueError(f"不支持的导出格式: {format}")

    def _export_torchscript(
        self,
        model: nn.Module,
        output_path: str,
        sample_inputs: Any,
        **kwargs,
    ) -> str:
        """导出为 TorchScript

        Args:
            model: PyTorch 模型
            output_path: 输出路径
            sample_inputs: 示例输入

        Returns:
            导出文件路径
        """
        device = self.device

        # 转换输入到 tensor
        if isinstance(sample_inputs, np.ndarray):
            sample_inputs = torch.from_numpy(sample_inputs).to(device)

        with torch.no_grad():
            if isinstance(sample_inputs, (tuple, list)):
                traced = torch.jit.trace(model, sample_inputs)
            else:
                traced = torch.jit.trace(model, (sample_inputs,))

        traced.save(output_path)
        logger.info(f"TorchScript 导出完成: {output_path}")
        return output_path

    def _export_onnx(
        self,
        model: nn.Module,
        output_path: str,
        sample_inputs: Any,
        **kwargs,
    ) -> str:
        """导出为 ONNX

        Args:
            model: PyTorch 模型
            output_path: 输出路径
            sample_inputs: 示例输入

        Returns:
            导出文件路径
        """
        device = self.device

        if isinstance(sample_inputs, np.ndarray):
            sample_inputs = torch.from_numpy(sample_inputs).to(device)

        opset_version = kwargs.get("opset_version", 17)
        dynamic_axes = kwargs.get("dynamic_axes", None)

        if dynamic_axes is None:
            dynamic_axes = {}
            if isinstance(sample_inputs, (tuple, list)):
                for idx, inp in enumerate(sample_inputs):
                    if isinstance(inp, torch.Tensor):
                        dynamic_axes[idx] = {
                            dim: f"dim_{dim}"
                            for dim in range(len(inp.shape))
                        }
            elif isinstance(sample_inputs, torch.Tensor):
                dynamic_axes = {
                    0: {dim: f"dim_{dim}"
                        for dim in range(len(sample_inputs.shape))}
                }

        with torch.no_grad():
            torch.onnx.export(
                model,
                sample_inputs,
                output_path,
                opset_version=opset_version,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                export_params=True,
            )

        logger.info(f"ONNX 导出完成: {output_path}")
        return output_path

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

        if not TORCH_AVAILABLE:
            raise BackendNotAvailableError("PyTorch 未安装")

        import gc

        # 同步 CUDA
        def synchronize():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # 预热
        for _ in range(num_warmup):
            self.inference(inputs)
        synchronize()
        gc.collect()

        # 正式测试
        latencies = []
        for _ in range(num_iterations):
            synchronize()
            start = time.perf_counter()
            self.inference(inputs)
            synchronize()
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

        # 内存估算
        memory_mb = 0.0
        if torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            torch.cuda.reset_peak_memory_stats()

        model_name = "unknown"
        if isinstance(self._original_model, nn.Module):
            model_name = self._original_model.__class__.__name__

        return BenchmarkResult(
            backend=self.backend_name,
            model_name=model_name,
            latency_ms=avg_latency,
            throughput_fps=throughput,
            memory_usage_mb=memory_mb,
            compile_time_s=self._compile_time,
            precision=self._config.precision.value,
            batch_size=batch_size,
            metadata={
                "p50_ms": p50,
                "p95_ms": p95,
                "p99_ms": p99,
                "num_iterations": num_iterations,
                "device": self.device,
                "compile_backend": (
                    self._config.compile_backend
                    if isinstance(self._config, TorchCompileConfig)
                    else "inductor"
                ),
                "graph_breaks": (
                    self._diagnostics.graph_break_count
                    if self._diagnostics
                    else 0
                ),
            },
        )

    def get_diagnostics(self) -> Optional[CompileDiagnostics]:
        """获取编译诊断信息

        Returns:
            CompileDiagnostics 或 None
        """
        return self._diagnostics

    def get_compile_stats(self) -> Dict[str, Any]:
        """获取编译统计信息

        Returns:
            统计信息字典
        """
        stats = {
            "backend": self.backend_name,
            "compiled": self._compiled,
            "device": self.device,
            "compile_time_s": self._compile_time,
        }

        if self._diagnostics:
            stats.update({
                "graph_count": self._diagnostics.graph_count,
                "graph_break_count": self._diagnostics.graph_break_count,
                "recompilation_count": self._diagnostics.recompilation_count,
                "compile_backend": self._diagnostics.backend_used,
                "compile_mode": self._diagnostics.mode_used,
            })

        if isinstance(self._config, TorchCompileConfig):
            stats["config"] = {
                "compile_backend": self._config.compile_backend,
                "compile_mode": self._config.compile_mode,
                "fullgraph": self._config.fullgraph,
                "dynamic": self._config.dynamic,
                "autocast": self._config.autocast_enabled,
            }

        return stats

    def reset(self) -> None:
        """重置编译状态"""
        try:
            dynamo_reset()
        except Exception:
            pass

        self._compiled_model = None
        self._compiled = False
        self._diagnostics = None

        # 清理缓存
        if self._cache_dir and os.path.exists(self._cache_dir):
            import shutil
            try:
                shutil.rmtree(self._cache_dir)
            except Exception as e:
                logger.warning(f"清理缓存失败: {e}")

    def release(self) -> None:
        """释放资源"""
        self.reset()
        self._original_model = None
        self._model = None
        self._device = None
        self._compile_time = 0.0
        self._guard_manager = None
        self._cache_dir = None


__all__ = [
    "TorchCompileCompiler",
    "TorchCompileConfig",
    "CompileDiagnostics",
    "CompileBackend",
    "CompileMode",
    "DynamicShapeStrategy",
    "TORCH_COMPILE_AVAILABLE",
    "TORCH_AVAILABLE",
]
