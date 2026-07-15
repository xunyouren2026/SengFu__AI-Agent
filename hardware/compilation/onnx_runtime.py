"""
ONNX Runtime 推理优化模块

模块路径: hardware/compilation/onnx_runtime.py

功能：
- ONNX模型加载与会话创建
- 图级别优化（常量折叠、死代码消除、算子融合）
- 静态/动态量化（UINT8/INT8）
- 执行提供者选择（CUDA/TensorRT/DML/CoreML/OpenVINO）
- IO绑定与内存优化
- 多线程并行推理
- 性能基准测试
"""

import os
import io
import time
import logging
import tempfile
import threading
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

# 检测 onnxruntime 可用性
try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
    _ORT_VERSION = ort.__version__
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    _ORT_VERSION = None

# 检测 onnx 可用性
try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class ExecutionProvider(Enum):
    """ONNX Runtime 执行提供者"""
    CPU = "CPUExecutionProvider"
    CUDA = "CUDAExecutionProvider"
    TENSORRT = "TensorrtExecutionProvider"
    DML = "DmlExecutionProvider"
    COREML = "CoreMLExecutionProvider"
    OPENVINO = "OpenVINOExecutionProvider"
    ROCM = "ROCMExecutionProvider"
    VITISAI = "VitisAIExecutionProvider"
    NNAPI = "NNAPIExecutionProvider"


class QuantizationMode(Enum):
    """量化模式"""
    DYNAMIC = "dynamic"
    STATIC = "static"
    QUANT_AWARE = "quant_aware"


class GraphOptLevel(Enum):
    """图优化级别"""
    DISABLE_ALL = ort.graph_optimization_level.ORT_DISABLE_ALL if ONNXRUNTIME_AVAILABLE else 0
    ENABLE_BASIC = ort.graph_optimization_level.ORT_ENABLE_BASIC if ONNXRUNTIME_AVAILABLE else 1
    ENABLE_EXTENDED = ort.graph_optimization_level.ORT_ENABLE_EXTENDED if ONNXRUNTIME_AVAILABLE else 2
    ENABLE_ALL = ort.graph_optimization_level.ORT_ENABLE_ALL if ONNXRUNTIME_AVAILABLE else 99


@dataclass
class ONNXRuntimeConfig(CompilationConfig):
    """ONNX Runtime 专用配置"""
    backend: CompilationBackend = CompilationBackend.ONNX_RUNTIME
    execution_providers: List[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    graph_optimization_level: int = 99  # ORT_ENABLE_ALL
    intra_op_num_threads: int = 0  # 0=自动
    inter_op_num_threads: int = 0
    execution_mode: str = "sequential"  # sequential / parallel
    enable_profiling: bool = False
    profile_file: str = "onnxruntime_profile.json"
    quantization_mode: Optional[str] = None
    quantization_per_channel: bool = False
    quantization_weight_type: str = "uint8"
    quantization_activation_type: str = "uint8"
    enable_io_binding: bool = False
    enable_mem_pattern: bool = True
    enable_mem_reuse: bool = True
    log_severity_level: int = 2  # 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal
    free_dimension_overrides: Dict[str, int] = field(default_factory=dict)


@dataclass
class QuantizationResult:
    """量化结果"""
    original_model_path: str
    quantized_model_path: str
    quantization_mode: str
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    op_types_quantized: List[str] = field(default_factory=list)
    op_types_not_quantized: List[str] = field(default_factory=list)


class ONNXRuntimeCompiler(BaseCompiler):
    """ONNX Runtime 编译优化器

    提供完整的 ONNX Runtime 推理优化流水线：
    1. 模型加载与验证
    2. 图优化（常量折叠、算子融合等）
    3. 量化（动态/静态/量化感知训练）
    4. 执行提供者选择与优先级配置
    5. IO绑定与零拷贝推理
    6. 多线程并行推理
    7. 性能基准测试
    """

    backend_name = "onnx_runtime"

    def __init__(self, config: Optional[ONNXRuntimeConfig] = None):
        super().__init__(config or ONNXRuntimeConfig())
        self._session: Optional[Any] = None
        self._model_path: Optional[str] = None
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._input_shapes: Dict[str, List[int]] = {}
        self._output_shapes: Dict[str, List[int]] = {}
        self._model_metadata: Dict[str, Any] = {}
        self._quant_result: Optional[QuantizationResult] = None
        self._io_binding: Optional[Any] = None
        self._lock = threading.Lock()

    @property
    def is_available(self) -> bool:
        """检查 ONNX Runtime 是否可用"""
        return ONNXRUNTIME_AVAILABLE

    @property
    def available_providers(self) -> List[str]:
        """获取当前可用的执行提供者"""
        if not ONNXRUNTIME_AVAILABLE:
            return ["CPUExecutionProvider"]
        return ort.get_available_providers()

    @property
    def session(self) -> Any:
        """获取当前推理会话"""
        if self._session is None:
            raise CompilationError("会话未初始化，请先调用 compile()")
        return self._session

    def _validate_providers(self, providers: List[str]) -> List[str]:
        """验证并过滤可用的执行提供者

        Args:
            providers: 请求的提供者列表

        Returns:
            经过验证的提供者列表
        """
        if not ONNXRUNTIME_AVAILABLE:
            return ["CPUExecutionProvider"]

        available = set(ort.get_available_providers())
        validated = []
        for p in providers:
            if p in available:
                validated.append(p)
            else:
                logger.warning(
                    f"执行提供者 '{p}' 不可用，已跳过。"
                    f"可用提供者: {available}"
                )
        if not validated:
            validated = ["CPUExecutionProvider"]
            logger.warning("所有请求的提供者均不可用，回退到 CPU")
        return validated

    def _build_session_options(self) -> Any:
        """构建 SessionOptions

        Returns:
            onnxruntime.SessionOptions 实例
        """
        if not ONNXRUNTIME_AVAILABLE:
            raise BackendNotAvailableError("onnxruntime 未安装")

        opts = ort.SessionOptions()
        cfg = self._config

        # 图优化级别
        graph_opt_map = {
            OptimizationLevel.NONE: ort.graph_optimization_level.ORT_DISABLE_ALL,
            OptimizationLevel.BASIC: ort.graph_optimization_level.ORT_ENABLE_BASIC,
            OptimizationLevel.EXTENDED: ort.graph_optimization_level.ORT_ENABLE_EXTENDED,
            OptimizationLevel.ALL: ort.graph_optimization_level.ORT_ENABLE_ALL,
        }
        opt_level = cfg.optimization_level
        opts.graph_optimization_level = graph_opt_map.get(
            opt_level, ort.graph_optimization_level.ORT_ENABLE_ALL
        )

        # 线程配置
        if isinstance(cfg, ONNXRuntimeConfig):
            if cfg.intra_op_num_threads > 0:
                opts.intra_op_num_threads = cfg.intra_op_num_threads
            if cfg.inter_op_num_threads > 0:
                opts.inter_op_num_threads = cfg.inter_op_num_threads
            opts.execution_mode = (
                ort.ExecutionMode.ORT_PARALLEL
                if cfg.execution_mode == "parallel"
                else ort.ExecutionMode.ORT_SEQUENTIAL
            )
            opts.enable_profiling = cfg.enable_profiling
            if cfg.enable_profiling and cfg.profile_file:
                opts.profile_file_prefix = cfg.profile_file
            opts.enable_mem_pattern = cfg.enable_mem_pattern
            opts.enable_mem_reuse = cfg.enable_mem_reuse
            opts.log_severity_level = cfg.log_severity_level

        # 精度相关
        if cfg.precision == PrecisionMode.FP16:
            opts.enable_mem_pattern = False

        return opts

    def load_onnx_model(
        self,
        model_path: Optional[str] = None,
        model_bytes: Optional[bytes] = None,
        onnx_model: Optional[Any] = None,
    ) -> Any:
        """加载 ONNX 模型

        Args:
            model_path: ONNX 模型文件路径
            model_bytes: ONNX 模型字节数据
            onnx_model: 已加载的 onnx.ModelProto

        Returns:
            onnx.ModelProto 实例
        """
        if not ONNX_AVAILABLE:
            raise BackendNotAvailableError("onnx 库未安装")

        if onnx_model is not None:
            return onnx_model

        if model_path is not None:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            self._model_path = str(path)
            model = onnx.load(str(path))
        elif model_bytes is not None:
            model = onnx.load_from_string(model_bytes)
        else:
            raise ValueError("必须提供 model_path、model_bytes 或 onnx_model")

        # 验证模型
        try:
            onnx.checker.check_model(model)
            logger.info(f"模型验证通过: {model.opset_import[0].version if model.opset_import else 'unknown'}")
        except onnx.ValidationError as e:
            raise ModelConversionError(f"ONNX 模型验证失败: {e}") from e

        return model

    def _extract_model_info(self, model: Any) -> None:
        """从 ONNX 模型中提取输入输出信息

        Args:
            model: onnx.ModelProto
        """
        self._input_names = []
        self._output_names = []
        self._input_shapes = {}
        self._output_shapes = {}

        for inp in model.graph.input:
            name = inp.name
            self._input_names.append(name)
            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    shape.append(-1)  # 动态维度
            self._input_shapes[name] = shape

        for out in model.graph.output:
            name = out.name
            self._output_names.append(name)
            shape = []
            for dim in out.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    shape.append(-1)
            self._output_shapes[name] = shape

        # 提取元数据
        self._model_metadata = {}
        for prop in model.metadata_props:
            self._model_metadata[prop.key] = prop.value

        logger.debug(
            f"模型信息 - 输入: {self._input_shapes}, "
            f"输出: {self._output_shapes}"
        )

    def optimize_graph(
        self,
        model: Any,
        optimization_level: Optional[OptimizationLevel] = None,
    ) -> Any:
        """对 ONNX 模型执行图级优化

        包含：常量折叠、公共子表达式消除、死代码消除、
        算子融合（Conv+BN、Conv+Add等）、冗余节点消除等。

        Args:
            model: onnx.ModelProto
            optimization_level: 优化级别

        Returns:
            优化后的 onnx.ModelProto
        """
        if not ONNX_AVAILABLE:
            raise BackendNotAvailableError("onnx 库未安装")

        from onnx import helper, numpy_helper, TensorProto

        level = optimization_level or self._config.optimization_level
        optimized = helper.make_model(
            model.graph,
            opset_imports=model.opset_import,
            ir_version=model.ir_version,
        )

        if level.value >= OptimizationLevel.BASIC.value:
            # 常量折叠
            optimized = self._constant_folding(optimized)

        if level.value >= OptimizationLevel.EXTENDED.value:
            # 死代码消除
            optimized = self._dead_code_elimination(optimized)
            # 公共子表达式消除
            optimized = self._common_subexpression_elimination(optimized)

        if level.value >= OptimizationLevel.ALL.value:
            # 算子融合
            optimized = self._fuse_operators(optimized)

        try:
            onnx.checker.check_model(optimized)
        except onnx.ValidationError:
            logger.warning("优化后模型验证失败，返回原始模型")
            return model

        return optimized

    def _constant_folding(self, model: Any) -> Any:
        """常量折叠优化

        将图中可预先计算的常量表达式替换为计算结果。

        Args:
            model: onnx.ModelProto

        Returns:
            优化后的模型
        """
        from onnx import helper, numpy_helper, TensorProto

        try:
            from onnx.optimizer import get_available_passes, optimize_model
            passes = get_available_passes()
            fold_passes = [p for p in passes if "fold" in p.lower()]
            if fold_passes:
                optimized = optimize_model(model, fold_passes)
                return optimized.model
        except Exception as e:
            logger.debug(f"常量折叠失败（非致命）: {e}")

        return model

    def _dead_code_elimination(self, model: Any) -> Any:
        """死代码消除

        移除图中未被使用的节点和初始化器。

        Args:
            model: onnx.ModelProto

        Returns:
            优化后的模型
        """
        from onnx import helper, numpy_helper

        try:
            from onnx.optimizer import optimize_model
            optimized = optimize_model(model, ["eliminate_deadend"])
            return optimized.model
        except Exception as e:
            logger.debug(f"死代码消除失败（非致命）: {e}")

        return model

    def _common_subexpression_elimination(self, model: Any) -> Any:
        """公共子表达式消除

        Args:
            model: onnx.ModelProto

        Returns:
            优化后的模型
        """
        try:
            from onnx.optimizer import optimize_model
            optimized = optimize_model(model, ["eliminate_common_subexpressions"])
            return optimized.model
        except Exception as e:
            logger.debug(f"公共子表达式消除失败（非致命）: {e}")

        return model

    def _fuse_operators(self, model: Any) -> Any:
        """算子融合

        融合 Conv+BN、Conv+Add、MatMul+Add 等常见模式。

        Args:
            model: onnx.ModelProto

        Returns:
            优化后的模型
        """
        try:
            from onnx.optimizer import optimize_model
            optimized = optimize_model(model, [
                "fuse_add_bias_into_conv",
                "fuse_bn_into_conv",
                "fuse_consecutive_concats",
                "fuse_consecutive_reduce_unsqueeze",
                "fuse_consecutive_squeezes",
                "fuse_consecutive_transposes",
                "fuse_matmul_add_bias_into_gemm",
                "fuse_pad_into_conv",
                "fuse_transpose_into_gemm",
            ])
            return optimized.model
        except Exception as e:
            logger.debug(f"算子融合失败（非致命）: {e}")

        return model

    def quantize_model(
        self,
        model: Any,
        mode: str = "dynamic",
        calibration_data: Optional[Sequence[np.ndarray]] = None,
        op_types_to_quantize: Optional[List[str]] = None,
        per_channel: bool = False,
        weight_type: str = "uint8",
        activation_type: str = "uint8",
        output_path: Optional[str] = None,
    ) -> QuantizationResult:
        """量化 ONNX 模型

        支持动态量化、静态量化和量化感知训练模型转换。

        Args:
            model: onnx.ModelProto 或模型路径
            mode: 量化模式 (dynamic/static/quant_aware)
            calibration_data: 静态量化校准数据
            op_types_to_quantize: 需要量化的算子类型
            per_channel: 是否使用逐通道量化
            weight_type: 权重量化类型 (uint8/int8/uint4/int4)
            activation_type: 激活量化类型
            output_path: 量化模型保存路径

        Returns:
            QuantizationResult 量化结果
        """
        try:
            from onnxruntime.quantization import (
                quantize_dynamic,
                quantize_static,
                quant_pre_process,
                QuantType,
                CalibrationDataReader,
            )
        except ImportError as e:
            raise BackendNotAvailableError(
                f"onnxruntime 量化模块不可用: {e}"
            ) from e

        # 保存原始模型到临时文件
        original_size = 0
        if isinstance(model, str):
            model_path = model
            original_size = os.path.getsize(model_path)
        else:
            tmp_dir = tempfile.mkdtemp()
            model_path = os.path.join(tmp_dir, "original_model.onnx")
            with open(model_path, "wb") as f:
                f.write(model.SerializeToString())
            original_size = len(model.SerializeToString())

        if output_path is None:
            output_path = model_path.replace(".onnx", "_quantized.onnx")

        # 量化类型映射
        type_map = {
            "uint8": QuantType.QUInt8,
            "int8": QuantType.QInt8,
            "uint4": QuantType.QUInt4,
            "int4": QuantType.QInt4,
        }
        w_type = type_map.get(weight_type, QuantType.QUInt8)
        a_type = type_map.get(activation_type, QuantType.QUInt8)

        if op_types_to_quantize is None:
            op_types_to_quantize = [
                "MatMul", "Attention", "Gather", "Embedding",
            ]

        try:
            if mode == "dynamic":
                quantize_dynamic(
                    model_input=model_path,
                    model_output=output_path,
                    weight_type=w_type,
                    op_types_to_quantize=op_types_to_quantize,
                    per_channel=per_channel,
                    reduce_range=True,
                )
            elif mode == "static":
                if calibration_data is None:
                    raise ValueError("静态量化需要提供 calibration_data")

                class _CalibReader(CalibrationDataReader):
                    """校准数据读取器"""
                    def __init__(self, data: Sequence[np.ndarray]):
                        self._data = list(data)
                        self._idx = 0

                    def get_next(self) -> Dict[str, np.ndarray]:
                        if self._idx >= len(self._data):
                            return None
                        batch = self._data[self._idx]
                        self._idx += 1
                        return {"input": batch}

                    def rewind(self):
                        self._idx = 0

                calib_reader = _CalibReader(calibration_data)
                quantize_static(
                    model_input=model_path,
                    model_output=output_path,
                    calibration_data_reader=calib_reader,
                    quant_format=None,
                    per_channel=per_channel,
                    weight_type=w_type,
                    activation_type=a_type,
                    op_types_to_quantize=op_types_to_quantize,
                    extra_options={
                        "WeightSymmetric": False,
                        "ActivationSymmetric": True,
                    },
                )
            elif mode == "quant_aware":
                # 量化感知训练模型不需要额外量化
                import shutil
                shutil.copy2(model_path, output_path)
            else:
                raise ValueError(f"不支持的量化模式: {mode}")

        except Exception as e:
            raise OptimizationError(f"量化失败: {e}") from e

        quant_size = os.path.getsize(output_path)
        compression_ratio = original_size / quant_size if quant_size > 0 else 0.0

        result = QuantizationResult(
            original_model_path=model_path,
            quantized_model_path=output_path,
            quantization_mode=mode,
            original_size_mb=original_size / (1024 * 1024),
            quantized_size_mb=quant_size / (1024 * 1024),
            compression_ratio=compression_ratio,
            op_types_quantized=op_types_to_quantize,
        )

        self._quant_result = result
        logger.info(
            f"量化完成 [{mode}]: "
            f"{result.original_size_mb:.2f}MB -> {result.quantized_size_mb:.2f}MB "
            f"(压缩比: {compression_ratio:.2f}x)"
        )

        return result

    def compile(
        self,
        model: Any,
        input_shapes: Optional[Dict[str, List[int]]] = None,
        **kwargs,
    ) -> Any:
        """编译 ONNX 模型并创建推理会话

        Args:
            model: ONNX 模型（路径/字节数据/ModelProto）
            input_shapes: 输入形状映射
            **kwargs: 额外参数
                - providers: 执行提供者列表
                - quantize: 是否进行量化
                - quant_mode: 量化模式

        Returns:
            onnxruntime.InferenceSession
        """
        if not ONNXRUNTIME_AVAILABLE:
            raise BackendNotAvailableError("onnxruntime 未安装")

        start_time = time.time()

        # 加载模型
        if isinstance(model, (str, Path)):
            onnx_model = self.load_onnx_model(model_path=str(model))
        elif isinstance(model, bytes):
            onnx_model = self.load_onnx_model(model_bytes=model)
        elif ONNX_AVAILABLE and hasattr(model, "graph"):
            onnx_model = model
        else:
            raise ModelConversionError(
                f"不支持的模型类型: {type(model)}，"
                f"请提供 ONNX 模型路径、字节数据或 ModelProto"
            )

        # 提取模型信息
        self._extract_model_info(onnx_model)

        # 图优化
        if self._config.optimization_level != OptimizationLevel.NONE:
            onnx_model = self.optimize_graph(onnx_model)

        # 量化
        quantize = kwargs.get("quantize", False)
        if quantize or (
            isinstance(self._config, ONNXRuntimeConfig)
            and self._config.quantization_mode
        ):
            quant_mode = kwargs.get(
                "quant_mode",
                self._config.quantization_mode or "dynamic",
            )
            calib_data = kwargs.get("calibration_data", None)
            quant_result = self.quantize_model(
                onnx_model,
                mode=quant_mode,
                calibration_data=calib_data,
                per_channel=(
                    self._config.quantization_per_channel
                    if isinstance(self._config, ONNXRuntimeConfig)
                    else False
                ),
            )
            onnx_model = self.load_onnx_model(
                model_path=quant_result.quantized_model_path
            )

        # 序列化模型
        model_bytes = onnx_model.SerializeToString()

        # 构建会话选项
        session_options = self._build_session_options()

        # 确定执行提供者
        providers = kwargs.get("providers")
        if providers is None and isinstance(self._config, ONNXRuntimeConfig):
            providers = self._config.execution_providers
        if providers is None:
            providers = ["CPUExecutionProvider"]
        providers = self._validate_providers(providers)

        # 创建推理会话
        try:
            self._session = ort.InferenceSession(
                model_bytes,
                sess_options=session_options,
                providers=providers,
            )
        except Exception as e:
            raise CompilationError(f"创建 ONNX Runtime 会话失败: {e}") from e

        self._compiled = True
        self._model = onnx_model
        compile_time = time.time() - start_time

        logger.info(
            f"ONNX Runtime 编译完成 | "
            f"提供者: {self._session.get_providers()} | "
            f"耗时: {compile_time:.3f}s"
        )

        return self._session

    def inference(self, inputs: Any, **kwargs) -> Any:
        """执行推理

        Args:
            inputs: 输入数据，支持以下格式：
                - dict: {"input_name": numpy_array, ...}
                - list/tuple: [numpy_array, ...] (按输入顺序)
                - numpy.ndarray: 单输入
            **kwargs: 额外参数
                - run_options: ort.RunOptions

        Returns:
            推理输出列表或单个输出
        """
        session = self.session

        # 统一输入格式
        if isinstance(inputs, dict):
            feed_dict = inputs
        elif isinstance(inputs, (list, tuple)):
            feed_dict = {
                name: data
                for name, data in zip(self._input_names, inputs)
            }
        elif isinstance(inputs, np.ndarray):
            if len(self._input_names) == 1:
                feed_dict = {self._input_names[0]: inputs}
            else:
                raise ValueError(
                    f"模型有 {len(self._input_names)} 个输入，"
                    f"请使用字典格式提供"
                )
        else:
            raise TypeError(f"不支持的输入类型: {type(inputs)}")

        run_options = kwargs.get("run_options")

        with self._lock:
            try:
                outputs = session.run(
                    self._output_names,
                    feed_dict,
                    run_options=run_options,
                )
            except Exception as e:
                raise CompilationError(f"推理失败: {e}") from e

        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def inference_with_io_binding(
        self,
        inputs: Dict[str, np.ndarray],
        output_device: str = "cpu",
    ) -> List[np.ndarray]:
        """使用 IO Binding 进行零拷贝推理

        适用于 GPU 推理场景，避免 CPU-GPU 间不必要的数据拷贝。

        Args:
            inputs: 输入字典
            output_device: 输出设备 (cuda/cpu)

        Returns:
            输出列表
        """
        session = self.session
        binding = session.io_binding()

        # 绑定输入
        for name, data in inputs.items():
            if output_device == "cuda" and ONNXRUNTIME_AVAILABLE:
                import onnxruntime as ort_module
                ort_device = ort_module.OrtDevice.cuda(
                    self._config.device_id
                )
                binding.bind_ortvalue_input(name, ort_module.OrtValue.ortvalue_from_numpy(data, ort_device))
            else:
                binding.bind_cpu_input(name, data)

        # 绑定输出
        for name in self._output_names:
            if output_device == "cuda" and ONNXRUNTIME_AVAILABLE:
                import onnxruntime as ort_module
                ort_device = ort_module.OrtDevice.cuda(
                    self._config.device_id
                )
                binding.bind_output(name, ort_device)
            else:
                binding.bind_output(name)

        # 执行推理
        session.run_with_iobinding(binding)

        # 获取输出
        outputs = []
        for name in self._output_names:
            ort_value = binding.get_outputs()[len(outputs)]
            outputs.append(ort_value.numpy())

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
        p50_latency = latencies[len(latencies) // 2]
        p95_latency = latencies[int(len(latencies) * 0.95)]
        p99_latency = latencies[int(len(latencies) * 0.99)]

        batch_size = 1
        if isinstance(inputs, dict):
            for v in inputs.values():
                if isinstance(v, np.ndarray):
                    batch_size = v.shape[0]
                    break
        elif isinstance(inputs, np.ndarray):
            batch_size = inputs.shape[0]

        throughput = (batch_size / (avg_latency / 1000))

        # 内存估算
        memory_mb = 0.0
        try:
            import onnxruntime as ort_module
            if hasattr(ort_module, "get_device"):
                memory_mb = 0.0  # ORT 不直接暴露内存信息
        except Exception:
            pass

        model_name = "unknown"
        if self._model_path:
            model_name = Path(self._model_path).stem

        return BenchmarkResult(
            backend=self.backend_name,
            model_name=model_name,
            latency_ms=avg_latency,
            throughput_fps=throughput,
            memory_usage_mb=memory_mb,
            compile_time_s=0.0,
            precision=self._config.precision.value,
            batch_size=batch_size,
            metadata={
                "p50_ms": p50_latency,
                "p95_ms": p95_latency,
                "p99_ms": p99_latency,
                "providers": self._session.get_providers(),
                "num_iterations": num_iterations,
            },
        )

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息

        Returns:
            包含模型详细信息的字典
        """
        if not self._compiled:
            return {}

        info = {
            "input_names": self._input_names,
            "output_names": self._output_names,
            "input_shapes": self._input_shapes,
            "output_shapes": self._output_shapes,
            "providers": self._session.get_providers(),
            "metadata": self._model_metadata,
        }

        if self._quant_result:
            info["quantization"] = {
                "mode": self._quant_result.quantization_mode,
                "compression_ratio": self._quant_result.compression_ratio,
                "original_size_mb": self._quant_result.original_size_mb,
                "quantized_size_mb": self._quant_result.quantized_size_mb,
            }

        return info

    def export_profile(self, output_path: Optional[str] = None) -> Optional[str]:
        """导出性能分析结果

        Args:
            output_path: 输出路径

        Returns:
            profile 文件路径
        """
        if not self._compiled or self._session is None:
            return None

        if isinstance(self._config, ONNXRuntimeConfig) and self._config.enable_profiling:
            profile_path = self._session.end_profiling()
            if output_path and profile_path:
                import shutil
                shutil.move(profile_path, output_path)
                return output_path
            return profile_path

        return None

    def release(self) -> None:
        """释放资源"""
        if self._session is not None:
            if isinstance(self._config, ONNXRuntimeConfig) and self._config.enable_profiling:
                try:
                    self._session.end_profiling()
                except Exception:
                    pass
            del self._session
            self._session = None
        self._compiled = False
        self._model = None
        self._model_path = None
        self._input_names = []
        self._output_names = []
        self._input_shapes = {}
        self._output_shapes = {}
        self._model_metadata = {}
        self._quant_result = None
        self._io_binding = None


__all__ = [
    "ONNXRuntimeCompiler",
    "ONNXRuntimeConfig",
    "ExecutionProvider",
    "QuantizationMode",
    "GraphOptLevel",
    "QuantizationResult",
    "ONNXRUNTIME_AVAILABLE",
]
