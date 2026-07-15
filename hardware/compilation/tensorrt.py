"""
TensorRT 引擎构建与推理优化模块

模块路径: hardware/compilation/tensorrt.py

功能：
- ONNX 模型到 TensorRT 引擎的转换
- 引擎构建配置（精度、工作空间、层融合策略）
- INT8/FP16 量化与精度校准
- 动态形状/Profile 配置
- 引擎序列化与反序列化
- 高性能推理执行
- 多流并发推理
"""

import os
import time
import logging
import struct
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Sequence
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

# 检测 TensorRT 可用性
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
    _TRT_VERSION = trt.__version__
    _TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
except ImportError:
    TENSORRT_AVAILABLE = False
    _TRT_VERSION = None
    _TRT_LOGGER = None


class EngineCapability(Enum):
    """引擎能力"""
    DEFAULT = 0
    SAFETY = 1
    DLA_STANDALONE = 2


class LayerPrecision(Enum):
    """层精度"""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    TF32 = "tf32"
    BF16 = "bf16"


@dataclass
class TensorRTConfig(CompilationConfig):
    """TensorRT 专用配置"""
    backend: CompilationBackend = CompilationBackend.TENSORRT
    max_workspace_size: int = 1 << 30  # 1GB
    max_batch_size: int = 0  # 0=不限制
    min_batch_size: int = 1
    opt_batch_size: int = 8
    dla_core: int = -1  # -1=不使用DLA
    enable_fp16: bool = False
    enable_int8: bool = False
    enable_tf32: bool = True
    enable_strict_types: bool = False
    calibration_cache: Optional[str] = None
    engine_cache_path: Optional[str] = None
    timing_cache_path: Optional[str] = None
    tactic_sources: int = 0  # 0=默认
    max_aux_streams: int = -1  # -1=默认
    preview_features: List[str] = field(default_factory=list)
    hardware_compatibility_level: int = 0
    profiler_verbosity: int = 0
    dump_layer_info: bool = False
    avg_timing_iterations: int = 8
    dla_sram_size: int = 1 << 20  # 1MB
    dla_local_dram_size: int = 1 << 24  # 16MB
    dla_global_dram_size: int = 1 << 28  # 256MB
    dynamic_shapes: Optional[Dict[str, Dict[str, List[int]]]] = None


@dataclass
class EngineBuildResult:
    """引擎构建结果"""
    engine_path: str
    build_time_s: float
    engine_size_mb: float
    layer_count: int
    precision: str
    device_memory_mb: float = 0.0
    tensor_memory_mb: float = 0.0
    profile_shapes: Dict[str, Any] = field(default_factory=dict)


class INT8Calibrator:
    """INT8 校准器基类

    用于 INT8 量化时的激活值范围校准。
    子类需实现 get_batch 方法提供校准数据。
    """

    def __init__(
        self,
        batch_size: int = 8,
        cache_file: Optional[str] = None,
    ):
        self._batch_size = batch_size
        self._cache_file = cache_file
        self._batches: List[np.ndarray] = []
        self._current_idx = 0

    def set_calibration_data(self, batches: Sequence[np.ndarray]) -> None:
        """设置校准数据

        Args:
            batches: 校准批次数据列表
        """
        self._batches = list(batches)
        self._current_idx = 0

    def get_batch_size(self) -> int:
        """获取批次大小"""
        return self._batch_size

    def get_batch(self, names: List[str]) -> Optional[Any]:
        """获取下一批校准数据

        Args:
            names: 输入层名称列表

        Returns:
            校准数据或 None（表示结束）
        """
        if self._current_idx >= len(self._batches):
            return None

        batch = self._batches[self._current_idx]
        self._current_idx += 1

        if not TENSORRT_AVAILABLE:
            return None

        import pycuda.driver as cuda
        import pycuda.autoinit

        # 分配 GPU 内存并拷贝数据
        host_input = batch
        device_input = cuda.mem_alloc(host_input.nbytes)
        cuda.memcpy_htod(device_input, host_input)

        return [int(device_input)]

    def read_calibration_cache(self) -> Optional[bytes]:
        """读取校准缓存"""
        if self._cache_file and os.path.exists(self._cache_file):
            with open(self._cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        """写入校准缓存"""
        if self._cache_file:
            os.makedirs(os.path.dirname(self._cache_file) or ".", exist_ok=True)
            with open(self._cache_file, "wb") as f:
                f.write(cache)
            logger.info(f"校准缓存已保存: {self._cache_file}")


class TensorRTCompiler(BaseCompiler):
    """TensorRT 编译优化器

    提供完整的 TensorRT 引擎构建与推理流水线：
    1. ONNX 模型解析
    2. 网络定义构建（含动态形状）
    3. 引擎构建配置（精度、工作空间、优化策略）
    4. INT8 校准
    5. 引擎构建与序列化
    6. 高性能推理执行
    7. 多流并发推理
    8. 性能基准测试
    """

    backend_name = "tensorrt"

    def __init__(self, config: Optional[TensorRTConfig] = None):
        super().__init__(config or TensorRTConfig())
        self._engine: Optional[Any] = None
        self._context: Optional[Any] = None
        self._logger = _TRT_LOGGER
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._input_shapes: Dict[str, List[int]] = {}
        self._output_shapes: Dict[str, List[int]] = {}
        self._bindings: Dict[str, int] = {}
        self._stream: Optional[Any] = None
        self._build_result: Optional[EngineBuildResult] = None

    @property
    def is_available(self) -> bool:
        """检查 TensorRT 是否可用"""
        return TENSORRT_AVAILABLE

    @property
    def engine(self) -> Any:
        """获取当前引擎"""
        if self._engine is None:
            raise CompilationError("引擎未构建，请先调用 compile()")
        return self._engine

    @property
    def context(self) -> Any:
        """获取执行上下文"""
        if self._context is None:
            raise CompilationError("上下文未创建，请先调用 compile()")
        return self._context

    def _create_builder(self) -> Any:
        """创建 TensorRT Builder

        Returns:
            trt.Builder 实例
        """
        if not TENSORRT_AVAILABLE:
            raise BackendNotAvailableError("tensorrt 未安装")

        builder = trt.Builder(self._logger)
        return builder

    def _create_network(
        self,
        builder: Any,
        flags: int = 0,
    ) -> Any:
        """创建网络定义

        Args:
            builder: trt.Builder
            flags: 网络创建标志

        Returns:
            trt.INetworkDefinition
        """
        if hasattr(trt, "NetworkDefinitionCreationFlag"):
            network_flags = 0
            if self._config.optimization_level == OptimizationLevel.ALL:
                network_flags |= int(
                    trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
                )
            network = builder.create_network(network_flags | flags)
        else:
            network = builder.create_network(flags)

        return network

    def _create_parser(self, network: Any) -> Any:
        """创建 ONNX 解析器

        Args:
            network: trt.INetworkDefinition

        Returns:
            trt.OnnxParser
        """
        if not TENSORRT_AVAILABLE:
            raise BackendNotAvailableError("tensorrt 未安装")

        parser = trt.OnnxParser(network, self._logger)
        return parser

    def _create_builder_config(
        self,
        builder: Any,
        calibrator: Optional[Any] = None,
    ) -> Any:
        """创建构建器配置

        Args:
            builder: trt.Builder
            calibrator: INT8 校准器

        Returns:
            trt.IBuilderConfig
        """
        if not TENSORRT_AVAILABLE:
            raise BackendNotAvailableError("tensorrt 未安装")

        cfg = self._config
        config = builder.create_builder_config()

        # 工作空间大小
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, cfg.max_workspace_size
        )

        # DLA 配置
        if isinstance(cfg, TensorRTConfig) and cfg.dla_core >= 0:
            config.set_dla_core(cfg.dla_core)
            config.set_memory_pool_limit(
                trt.MemoryPoolType.DLA_MANAGED_SRAM, cfg.dla_sram_size
            )
            config.set_memory_pool_limit(
                trt.MemoryPoolType.DLA_LOCAL_DRAM, cfg.dla_local_dram_size
            )
            config.set_memory_pool_limit(
                trt.MemoryPoolType.DLA_GLOBAL_DRAM, cfg.dla_global_dram_size
            )

        # 精度配置
        if isinstance(cfg, TensorRTConfig):
            if cfg.enable_fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("已启用 FP16 精度")

            if cfg.enable_int8 and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                if calibrator is not None:
                    config.int8_calibrator = calibrator
                logger.info("已启用 INT8 精度")

            if cfg.enable_tf32:
                config.set_flag(trt.BuilderFlag.TF32)

            if cfg.enable_strict_types:
                config.set_flag(trt.BuilderFlag.STRICT_TYPES)

            # 性能调优
            config.avg_timing_iterations = cfg.avg_timing_iterations
            if cfg.max_aux_streams >= 0:
                config.max_aux_streams = cfg.max_aux_streams

            # 预览特性
            for feature in cfg.preview_features:
                if hasattr(trt, "PreviewFeature"):
                    try:
                        flag = getattr(trt.PreviewFeature, feature.upper())
                        config.set_preview_feature(flag, True)
                    except AttributeError:
                        logger.warning(f"未知预览特性: {feature}")

        # 时间缓存
        if isinstance(cfg, TensorRTConfig) and cfg.timing_cache_path:
            cache_path = Path(cfg.timing_cache_path)
            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        cache = config.create_timing_cache(f.read())
                    config.set_timing_cache(cache, False)
                except Exception as e:
                    logger.warning(f"加载时间缓存失败: {e}")

        return config

    def _configure_dynamic_shapes(
        self,
        profile: Any,
        shapes: Dict[str, Dict[str, List[int]]],
    ) -> None:
        """配置动态形状 Profile

        Args:
            profile: trt.IOptimizationProfile
            shapes: 形状配置
                {
                    "input_name": {
                        "min": [1, 3, 224, 224],
                        "opt": [8, 3, 224, 224],
                        "max": [32, 3, 224, 224],
                    }
                }
        """
        for name, shape_config in shapes.items():
            min_shape = shape_config["min"]
            opt_shape = shape_config["opt"]
            max_shape = shape_config["max"]

            profile.set_shape(
                name,
                min=min_shape,
                opt=opt_shape,
                max=max_shape,
            )
            logger.debug(
                f"动态形状 [{name}]: "
                f"min={min_shape}, opt={opt_shape}, max={max_shape}"
            )

    def parse_onnx(
        self,
        model_path: str,
        network: Any,
    ) -> bool:
        """解析 ONNX 模型到 TensorRT 网络定义

        Args:
            model_path: ONNX 模型文件路径
            network: trt.INetworkDefinition

        Returns:
            是否解析成功
        """
        parser = self._create_parser(network)

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX 模型不存在: {model_path}")

        with open(model_path, "rb") as f:
            if not parser.parse(f.read()):
                errors = ""
                for idx in range(parser.num_errors):
                    errors += f"\n  [{idx}] {parser.get_error(idx)}"
                raise ModelConversionError(
                    f"ONNX 解析失败，共 {parser.num_errors} 个错误:{errors}"
                )

        # 获取输入输出信息
        self._input_names = []
        self._output_names = []
        for idx in range(network.num_inputs):
            inp = network.get_input(idx)
            self._input_names.append(inp.name)
            shape = [inp.shape[d] if inp.shape[d] > 0 else -1
                     for d in range(len(inp.shape))]
            self._input_shapes[inp.name] = shape
        for idx in range(network.num_outputs):
            out = network.get_output(idx)
            self._output_names.append(out.name)
            shape = [out.shape[d] if out.shape[d] > 0 else -1
                     for d in range(len(out.shape))]
            self._output_shapes[out.name] = shape

        logger.info(
            f"ONNX 解析完成 | 输入: {self._input_names} | "
            f"输出: {self._output_names}"
        )
        return True

    def build_engine(
        self,
        model_path: str,
        calibrator: Optional[Any] = None,
        dynamic_shapes: Optional[Dict[str, Dict[str, List[int]]]] = None,
    ) -> EngineBuildResult:
        """构建 TensorRT 引擎

        Args:
            model_path: ONNX 模型路径
            calibrator: INT8 校准器
            dynamic_shapes: 动态形状配置

        Returns:
            EngineBuildResult
        """
        if not TENSORRT_AVAILABLE:
            raise BackendNotAvailableError("tensorrt 未安装")

        start_time = time.time()

        builder = self._create_builder()
        network = self._create_network(builder)
        config = self._create_builder_config(builder, calibrator)

        # 解析 ONNX
        self.parse_onnx(model_path, network)

        # 配置动态形状
        cfg = self._config
        shapes = dynamic_shapes
        if shapes is None and isinstance(cfg, TensorRTConfig):
            shapes = cfg.dynamic_shapes

        if shapes:
            profile = builder.create_optimization_profile()
            self._configure_dynamic_shapes(profile, shapes)
            config.add_optimization_profile(profile)

        # 构建引擎
        logger.info("开始构建 TensorRT 引擎...")
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise CompilationError("TensorRT 引擎构建失败")

        # 反序列化引擎
        runtime = trt.Runtime(self._logger)
        engine = runtime.deserialize_cuda_engine(plan)

        if engine is None:
            raise CompilationError("TensorRT 引擎反序列化失败")

        self._engine = engine
        self._context = engine.create_execution_context()

        build_time = time.time() - start_time
        engine_size = len(plan) / (1024 * 1024)

        # 保存时间缓存
        if isinstance(cfg, TensorRTConfig) and cfg.timing_cache_path:
            try:
                cache = config.get_timing_cache()
                os.makedirs(
                    os.path.dirname(cfg.timing_cache_path) or ".",
                    exist_ok=True,
                )
                with open(cfg.timing_cache_path, "wb") as f:
                    f.write(cache.serialize())
            except Exception as e:
                logger.warning(f"保存时间缓存失败: {e}")

        # 确定精度
        precision = "fp32"
        if isinstance(cfg, TensorRTConfig):
            if cfg.enable_int8:
                precision = "int8"
            elif cfg.enable_fp16:
                precision = "fp16"

        result = EngineBuildResult(
            engine_path=model_path,
            build_time_s=build_time,
            engine_size_mb=engine_size,
            layer_count=engine.num_layers,
            precision=precision,
            profile_shapes=shapes or {},
        )

        self._build_result = result
        logger.info(
            f"引擎构建完成 | 层数: {engine.num_layers} | "
            f"大小: {engine_size:.2f}MB | 耗时: {build_time:.2f}s"
        )

        return result

    def serialize_engine(self, output_path: str) -> None:
        """序列化引擎到文件

        Args:
            output_path: 输出文件路径
        """
        if self._engine is None:
            raise CompilationError("引擎未构建")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plan = self._engine.serialize()
        with open(output_path, "wb") as f:
            f.write(plan)
        logger.info(f"引擎已序列化: {output_path}")

    def load_engine(self, engine_path: str) -> None:
        """从文件加载引擎

        Args:
            engine_path: 引擎文件路径
        """
        if not TENSORRT_AVAILABLE:
            raise BackendNotAvailableError("tensorrt 未安装")

        path = Path(engine_path)
        if not path.exists():
            raise FileNotFoundError(f"引擎文件不存在: {engine_path}")

        runtime = trt.Runtime(self._logger)
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        if engine is None:
            raise CompilationError("引擎加载失败")

        self._engine = engine
        self._context = engine.create_execution_context()
        self._compiled = True

        # 提取输入输出信息
        self._input_names = []
        self._output_names = []
        self._bindings = {}
        for idx in range(engine.num_bindings):
            name = engine.get_binding_name(idx)
            shape = engine.get_binding_shape(idx)
            is_input = engine.binding_is_input(idx)
            self._bindings[name] = idx

            if is_input:
                self._input_names.append(name)
                self._input_shapes[name] = list(shape)
            else:
                self._output_names.append(name)
                self._output_shapes[name] = list(shape)

        logger.info(f"引擎加载完成: {engine_path}")

    def compile(
        self,
        model: Any,
        input_shapes: Optional[Dict[str, List[int]]] = None,
        **kwargs,
    ) -> Any:
        """编译模型为 TensorRT 引擎

        Args:
            model: ONNX 模型路径或引擎路径
            input_shapes: 输入形状映射
            **kwargs: 额外参数
                - calibrator: INT8 校准器
                - dynamic_shapes: 动态形状配置
                - engine_path: 引擎保存路径

        Returns:
            TensorRT 引擎
        """
        if not TENSORRT_AVAILABLE:
            raise BackendNotAvailableError("tensorrt 未安装")

        model_str = str(model)

        # 检查是否为已构建的引擎
        if model_str.endswith(".engine") or model_str.endswith(".plan"):
            self.load_engine(model_str)
            return self._engine

        # 检查引擎缓存
        cfg = self._config
        if isinstance(cfg, TensorRTConfig) and cfg.engine_cache_path:
            cache_dir = Path(cfg.engine_cache_path)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_name = Path(model_str).stem
            engine_cache = cache_dir / f"{cache_name}.engine"
            if engine_cache.exists():
                logger.info(f"从缓存加载引擎: {engine_cache}")
                self.load_engine(str(engine_cache))
                return self._engine

        # 构建引擎
        calibrator = kwargs.get("calibrator")
        dynamic_shapes = kwargs.get("dynamic_shapes")

        # 自动生成动态形状
        if dynamic_shapes is None and input_shapes is not None:
            dynamic_shapes = {}
            for name, shape in input_shapes.items():
                dynamic_shapes[name] = {
                    "min": [shape[0], *shape[1:]],
                    "opt": [max(shape[0], 8), *shape[1:]],
                    "max": [shape[0] * 4, *shape[1:]],
                }

        result = self.build_engine(
            model_str,
            calibrator=calibrator,
            dynamic_shapes=dynamic_shapes,
        )

        # 保存引擎
        engine_path = kwargs.get("engine_path")
        if engine_path:
            self.serialize_engine(engine_path)
        elif isinstance(cfg, TensorRTConfig) and cfg.engine_cache_path:
            cache_dir = Path(cfg.engine_cache_path)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_name = Path(model_str).stem
            engine_cache = cache_dir / f"{cache_name}.engine"
            self.serialize_engine(str(engine_cache))

        self._compiled = True
        return self._engine

    def _allocate_buffers(
        self,
        inputs: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[Any]]:
        """分配输入输出缓冲区

        Args:
            inputs: 输入数据

        Returns:
            (host_inputs, device_outputs, bindings)
        """
        if not TENSORRT_AVAILABLE:
            raise BackendNotAvailableError("tensorrt 未安装")

        import pycuda.driver as cuda
        import pycuda.autoinit

        host_inputs = {}
        device_inputs = {}
        device_outputs = {}
        bindings = [None] * self.engine.num_bindings

        for name in self._input_names:
            if name not in inputs:
                raise ValueError(f"缺少输入: {name}")
            data = inputs[name]
            host_inputs[name] = data

            # 设置优化 Profile
            if self._context.all_shape_inputs_specified:
                pass
            else:
                shape = list(data.shape)
                self._context.set_binding_shape(
                    self._bindings[name], shape
                )

            # 分配设备内存
            size = data.nbytes
            device_mem = cuda.mem_alloc(size)
            cuda.memcpy_htod(device_mem, data)
            device_inputs[name] = device_mem
            bindings[self._bindings[name]] = int(device_mem)

        for name in self._output_names:
            idx = self._bindings[name]
            shape = self._context.get_binding_shape(idx)
            dtype = trt.nptype(self.engine.get_binding_dtype(idx))
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize

            device_mem = cuda.mem_alloc(size)
            device_outputs[name] = device_mem
            bindings[idx] = int(device_mem)

        return host_inputs, device_outputs, bindings

    def inference(self, inputs: Any, **kwargs) -> Any:
        """执行推理

        Args:
            inputs: 输入数据
                - dict: {"input_name": numpy_array}
                - numpy.ndarray: 单输入
            **kwargs: 额外参数

        Returns:
            推理输出
        """
        if not TENSORRT_AVAILABLE:
            raise BackendNotAvailableError("tensorrt 未安装")

        import pycuda.driver as cuda
        import pycuda.autoinit

        ctx = self.context

        # 统一输入格式
        if isinstance(inputs, dict):
            feed_dict = inputs
        elif isinstance(inputs, np.ndarray):
            if len(self._input_names) == 1:
                feed_dict = {self._input_names[0]: inputs}
            else:
                raise ValueError(
                    f"模型有 {len(self._input_names)} 个输入，"
                    f"请使用字典格式"
                )
        else:
            raise TypeError(f"不支持的输入类型: {type(inputs)}")

        # 分配缓冲区
        host_inputs, device_outputs, bindings = self._allocate_buffers(
            feed_dict
        )

        # 创建 CUDA 流
        stream = cuda.Stream()

        # 执行推理
        try:
            ctx.execute_async_v2(
                bindings=bindings,
                stream_handle=stream.handle,
            )
            stream.synchronize()
        except Exception as e:
            raise CompilationError(f"TensorRT 推理失败: {e}") from e

        # 拷贝输出
        outputs = []
        for name in self._output_names:
            idx = self._bindings[name]
            shape = self._context.get_binding_shape(idx)
            dtype = trt.nptype(self.engine.get_binding_dtype(idx))
            host_output = np.empty(shape, dtype=dtype)
            cuda.memcpy_dtoh(host_output, device_outputs[name])
            outputs.append(host_output)

        if len(outputs) == 1:
            return outputs[0]
        return outputs

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

        import gc

        # 预热
        for _ in range(num_warmup):
            self.inference(inputs)
        gc.collect()

        # 正式测试
        latencies = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            self.inference(inputs)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        latencies.sort()
        avg_latency = sum(latencies) / len(latencies)
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]

        batch_size = 1
        if isinstance(inputs, dict):
            for v in inputs.values():
                if isinstance(v, np.ndarray):
                    batch_size = v.shape[0]
                    break
        elif isinstance(inputs, np.ndarray):
            batch_size = inputs.shape[0]

        throughput = batch_size / (avg_latency / 1000)

        model_name = "unknown"
        if self._build_result:
            model_name = Path(self._build_result.engine_path).stem

        return BenchmarkResult(
            backend=self.backend_name,
            model_name=model_name,
            latency_ms=avg_latency,
            throughput_fps=throughput,
            memory_usage_mb=self._build_result.engine_size_mb if self._build_result else 0,
            compile_time_s=self._build_result.build_time_s if self._build_result else 0,
            precision=self._config.precision.value,
            batch_size=batch_size,
            metadata={
                "p50_ms": p50,
                "p95_ms": p95,
                "p99_ms": p99,
                "layer_count": self._engine.num_layers if self._engine else 0,
                "num_iterations": num_iterations,
            },
        )

    def get_engine_info(self) -> Dict[str, Any]:
        """获取引擎信息

        Returns:
            引擎详细信息
        """
        if not self._compiled or self._engine is None:
            return {}

        info = {
            "input_names": self._input_names,
            "output_names": self._output_names,
            "input_shapes": self._input_shapes,
            "output_shapes": self._output_shapes,
            "num_layers": self._engine.num_layers,
            "device_memory": self._engine.device_memory_size,
        }

        if self._build_result:
            info["build_time_s"] = self._build_result.build_time_s
            info["engine_size_mb"] = self._build_result.engine_size_mb
            info["precision"] = self._build_result.precision

        return info

    def release(self) -> None:
        """释放资源"""
        self._context = None
        self._engine = None
        self._stream = None
        self._build_result = None
        self._input_names = []
        self._output_names = []
        self._input_shapes = {}
        self._output_shapes = {}
        self._bindings = {}
        self._compiled = False
        self._model = None


__all__ = [
    "TensorRTCompiler",
    "TensorRTConfig",
    "EngineBuildResult",
    "INT8Calibrator",
    "EngineCapability",
    "LayerPrecision",
    "TENSORRT_AVAILABLE",
]
