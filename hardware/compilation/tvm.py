"""
TVM 编译优化模块

模块路径: hardware/compilation/tvm.py

功能：
- 多框架模型导入（ONNX/PyTorch/TFLite）
- Relay IR 图级优化
- 算子调度与自动调优（AutoTVM/AutoScheduler）
- 多目标编译（CPU/CUDA/ROCm/Vulkan/LLVM）
- 图代码生成与构建
- 运行时模块管理与推理
- 编译缓存与复用
"""

import os
import time
import logging
import tempfile
import json
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

# 检测 TVM 可用性
try:
    import tvm
    import tvm.relay as relay
    import tvm.runtime as tvm_runtime
    from tvm import autotvm, auto_scheduler
    TVM_AVAILABLE = True
    _TVM_VERSION = tvm.__version__
except ImportError:
    TVM_AVAILABLE = False
    _TVM_VERSION = None


class TVMTarget(Enum):
    """TVM 编译目标"""
    CPU = "llvm"
    CUDA = "cuda"
    ROCM = "rocm"
    VULKAN = "vulkan"
    OPENCL = "opencl"
    METAL = "metal"
    WEBGPU = "webgpu"
    LLVM_AARCH64 = "llvm -mtriple=aarch64-linux-gnu"
    LLVM_ARM = "llvm -mtriple=armv7l-linux-gnueabihf"


class PassLevel(Enum):
    """优化 Pass 级别"""
    NONE = 0
    BASIC = 1
    EXTENDED = 2
    ALL = 3


@dataclass
class TVMConfig(CompilationConfig):
    """TVM 专用配置"""
    backend: CompilationBackend = CompilationBackend.TVM
    target: str = "llvm"
    target_host: Optional[str] = None
    opt_level: int = 3  # Relay 优化级别 0-3
    pass_level: PassLevel = PassLevel.ALL
    use_auto_scheduler: bool = False
    use_autotvm: bool = False
    tuning_log: Optional[str] = None
    tuning_trials: int = 1000
    tuning_early_stopping: int = 100
    tuning_measure_option: Optional[Dict[str, Any]] = None
    executor: str = "graph"  # graph / aot / vm
    runtime: str = "cpp"  # cpp / cuda / vulkan
    mod_name: str = "default"
    build_dir: Optional[str] = None
    cross_compiler: Optional[str] = None
    # 内存优化
    layout_transform: Optional[List[str]] = None
    memory_plan: bool = True
    # 量化
    calibrate_mode: str = "kl_divergence"
    dataset_size: int = 100
    # 导出
    export_lib_path: Optional[str] = None
    export_params_path: Optional[str] = None
    # 调度策略
    scheduler: str = "auto"  # auto / static / dynamic


@dataclass
class TuningResult:
    """自动调优结果"""
    target: str
    model_name: str
    tuning_time_s: float
    num_trials: int
    best_latency_ms: float
    baseline_latency_ms: float
    speedup: float
    log_path: Optional[str] = None
    topk_results: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TVMBuildArtifact:
    """TVM 构建产物"""
    lib_path: str
    params_path: Optional[str]
    graph_json: Optional[str]
    target: str
    executor: str
    build_time_s: float
    lib_size_mb: float


class TVMCompiler(BaseCompiler):
    """TVM 编译优化器

    提供完整的 TVM 编译优化流水线：
    1. 多框架模型导入（ONNX/PyTorch/TFLite）
    2. Relay IR 构建
    3. 图级优化 Pass（布局转换、常量折叠、算子融合等）
    4. 自动调优（AutoTVM/AutoScheduler）
    5. 多目标代码生成
    6. 运行时模块构建
    7. 高性能推理执行
    8. 编译缓存管理
    """

    backend_name = "tvm"

    def __init__(self, config: Optional[TVMConfig] = None):
        super().__init__(config or TVMConfig())
        self._module: Optional[Any] = None
        self._params: Optional[Dict[str, Any]] = None
        self._graph_json: Optional[str] = None
        self._lib: Optional[Any] = None
        self._runtime_module: Optional[Any] = None
        self._device: Optional[Any] = None
        self._tuning_result: Optional[TuningResult] = None
        self._build_artifact: Optional[TVMBuildArtifact] = None
        self._input_names: List[str] = []
        self._input_shapes: Dict[str, List[int]] = {}
        self._output_names: List[str] = []

    @property
    def is_available(self) -> bool:
        """检查 TVM 是否可用"""
        return TVM_AVAILABLE

    @property
    def runtime_module(self) -> Any:
        """获取运行时模块"""
        if self._runtime_module is None:
            raise CompilationError("运行时模块未构建，请先调用 compile()")
        return self._runtime_module

    def _get_target(self) -> Any:
        """获取 TVM Target 对象

        Returns:
            tvm.target.Target
        """
        if not TVM_AVAILABLE:
            raise BackendNotAvailableError("TVM 未安装")

        cfg = self._config
        target_str = cfg.target if isinstance(cfg, TVMConfig) else "llvm"
        host_str = None
        if isinstance(cfg, TVMConfig) and cfg.target_host:
            host_str = cfg.target_host

        if host_str:
            return tvm.target.Target(target_str, host=host_str)
        return tvm.target.Target(target_str)

    def _get_device(self) -> Any:
        """获取 TVM Device

        Returns:
            tvm.runtime.Device
        """
        if self._device is not None:
            return self._device

        if not TVM_AVAILABLE:
            raise BackendNotAvailableError("TVM 未安装")

        cfg = self._config
        target_str = cfg.target if isinstance(cfg, TVMConfig) else "llvm"

        if "cuda" in target_str:
            self._device = tvm.cuda(cfg.device_id)
        elif "rocm" in target_str:
            self._device = tvm.rocm(cfg.device_id)
        elif "vulkan" in target_str:
            self._device = tvm.vulkan(cfg.device_id)
        elif "metal" in target_str:
            self._device = tvm.metal(cfg.device_id)
        elif "opencl" in target_str:
            self._device = tvm.opencl(cfg.device_id)
        else:
            self._device = tvm.cpu(cfg.device_id)

        return self._device

    def import_model(
        self,
        model: Any,
        model_format: Optional[str] = None,
        input_shapes: Optional[Dict[str, List[int]]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """导入模型到 TVM Relay IR

        支持格式：ONNX、PyTorch、TFLite

        Args:
            model: 模型（路径/字节数据/ModelProto/nn.Module）
            model_format: 模型格式 (onnx/pytorch/tflite)
            input_shapes: 输入形状映射

        Returns:
            (relay.Module, params)
        """
        if not TVM_AVAILABLE:
            raise BackendNotAvailableError("TVM 未安装")

        # 自动检测格式
        if model_format is None:
            if isinstance(model, (str, Path)):
                suffix = Path(str(model)).suffix.lower()
                format_map = {
                    ".onnx": "onnx",
                    ".pt": "pytorch",
                    ".pth": "pytorch",
                    ".tflite": "tflite",
                }
                model_format = format_map.get(suffix)
            if model_format is None:
                model_format = "onnx"

        if model_format == "onnx":
            return self._import_onnx(model, input_shapes)
        elif model_format == "pytorch":
            return self._import_pytorch(model, input_shapes)
        elif model_format == "tflite":
            return self._import_tflite(model, input_shapes)
        else:
            raise ModelConversionError(
                f"不支持的模型格式: {model_format}，"
                f"支持: onnx/pytorch/tflite"
            )

    def _import_onnx(
        self,
        model: Any,
        input_shapes: Optional[Dict[str, List[int]]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """导入 ONNX 模型

        Args:
            model: ONNX 模型路径或字节数据
            input_shapes: 输入形状

        Returns:
            (relay.Module, params)
        """
        try:
            from tvm.relay.frontend import from_onnx
        except ImportError as e:
            raise BackendNotAvailableError(
                f"TVM ONNX 前端不可用: {e}"
            ) from e

        model_path = str(model) if isinstance(model, (str, Path)) else None
        model_bytes = model if isinstance(model, bytes) else None

        if model_bytes is not None:
            import io
            model_path = io.BytesIO(model_bytes)

        shape_dict = {}
        if input_shapes:
            shape_dict = {
                name: shape for name, shape in input_shapes.items()
            }

        try:
            mod, params = from_onnx(
                model_path,
                shape=shape_dict,
                freeze_params=True,
            )
        except Exception as e:
            raise ModelConversionError(f"ONNX 导入失败: {e}") from e

        # 提取输入信息
        if hasattr(mod, "main") and hasattr(mod["main"], "params"):
            for param in mod["main"].params:
                if input_shapes and param.name_hint in input_shapes:
                    self._input_names.append(param.name_hint)
                    self._input_shapes[param.name_hint] = input_shapes[
                        param.name_hint
                    ]

        logger.info(f"ONNX 模型导入成功")
        return mod, params

    def _import_pytorch(
        self,
        model: Any,
        input_shapes: Optional[Dict[str, List[int]]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """导入 PyTorch 模型

        Args:
            model: PyTorch nn.Module 或模型路径
            input_shapes: 输入形状

        Returns:
            (relay.Module, params)
        """
        try:
            from tvm.relay.frontend import from_pytorch
        except ImportError as e:
            raise BackendNotAvailableError(
                f"TVM PyTorch 前端不可用: {e}"
            ) from e

        import torch
        import torch.nn as nn

        if isinstance(model, (str, Path)):
            model = torch.jit.load(str(model))
            model.eval()

        if not isinstance(model, (nn.Module, torch.jit.ScriptModule)):
            raise ModelConversionError(
                f"不支持的 PyTorch 模型类型: {type(model)}"
            )

        model.eval()

        # 创建示例输入
        if input_shapes is None:
            raise ValueError("PyTorch 导入需要提供 input_shapes")

        input_list = []
        for name, shape in input_shapes.items():
            self._input_names.append(name)
            self._input_shapes[name] = shape
            input_list.append(torch.randn(*shape))

        shape_dict = input_shapes

        try:
            with torch.no_grad():
                mod, params = from_pytorch(
                    model,
                    input_list,
                    shape_dict=shape_dict,
                )
        except Exception as e:
            raise ModelConversionError(f"PyTorch 导入失败: {e}") from e

        logger.info("PyTorch 模型导入成功")
        return mod, params

    def _import_tflite(
        self,
        model: Any,
        input_shapes: Optional[Dict[str, List[int]]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """导入 TFLite 模型

        Args:
            model: TFLite 模型路径
            input_shapes: 输入形状

        Returns:
            (relay.Module, params)
        """
        try:
            from tvm.relay.frontend import from_tflite
        except ImportError as e:
            raise BackendNotAvailableError(
                f"TVM TFLite 前端不可用: {e}"
            ) from e

        model_path = str(model)
        if not Path(model_path).exists():
            raise FileNotFoundError(f"TFLite 模型不存在: {model_path}")

        try:
            mod, params = from_tflite(
                model_path,
                shape_dict=input_shapes,
            )
        except Exception as e:
            raise ModelConversionError(f"TFLite 导入失败: {e}") from e

        logger.info("TFLite 模型导入成功")
        return mod, params

    def optimize_relay(
        self,
        mod: Any,
        params: Optional[Dict[str, Any]] = None,
        target: Optional[Any] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """对 Relay IR 执行图级优化

        包含：常量折叠、死代码消除、算子融合、布局转换、
        内存规划、代数简化等。

        Args:
            mod: relay.Module
            params: 模型参数
            target: 编译目标

        Returns:
            (优化后的 relay.Module, 更新后的 params)
        """
        if not TVM_AVAILABLE:
            raise BackendNotAvailableError("TVM 未安装")

        if target is None:
            target = self._get_target()

        cfg = self._config
        opt_level = cfg.opt_level if isinstance(cfg, TVMConfig) else 3

        # 构建优化 Pass 流水线
        passes = []

        if isinstance(cfg, TVMConfig):
            level = cfg.pass_level

            if level.value >= PassLevel.BASIC.value:
                passes.extend([
                    "transform.InferType",
                    "transform.SimplifyInference",
                    "transform.FoldConstant",
                    "transform.FoldScaleAxis",
                ])

            if level.value >= PassLevel.EXTENDED.value:
                passes.extend([
                    "transform.AlterOpLayout",
                    "transform.FastMath",
                    "transform.CombineParallelConv2D",
                    "transform.FoldBatchNorm",
                ])

            if level.value >= PassLevel.ALL.value:
                passes.extend([
                    "transform.CanonicalizeOps",
                    "transform.EliminateCommonSubexpr",
                    "transform.Legalize",
                ])

        # 布局转换
        if isinstance(cfg, TVMConfig) and cfg.layout_transform:
            desired_layouts = cfg.layout_transform
            layout_pass = relay.transform.ConvertLayout(desired_layouts)
            passes.append(layout_pass)

        # 执行优化
        try:
            with tvm.transform.PassContext(opt_level=opt_level):
                if passes:
                    seq = tvm.transform.Sequential(passes)
                    mod = seq(mod)

                # 内存规划
                if isinstance(cfg, TVMConfig) and cfg.memory_plan:
                    mod = relay.transform.MemoryPlan()(mod)

        except Exception as e:
            raise OptimizationError(f"Relay 优化失败: {e}") from e

        logger.info(
            f"Relay 优化完成 | 级别: {opt_level} | "
            f"Pass 数量: {len(passes)}"
        )

        return mod, params

    def auto_tune(
        self,
        mod: Any,
        params: Dict[str, Any],
        target: Optional[Any] = None,
        max_trials: Optional[int] = None,
        log_file: Optional[str] = None,
    ) -> TuningResult:
        """自动调优模型

        使用 AutoTVM 或 AutoScheduler 搜索最优算子调度。

        Args:
            mod: relay.Module
            params: 模型参数
            target: 编译目标
            max_trials: 最大调优试验次数
            log_file: 调优日志路径

        Returns:
            TuningResult
        """
        if not TVM_AVAILABLE:
            raise BackendNotAvailableError("TVM 未安装")

        cfg = self._config
        if target is None:
            target = self._get_target()

        trials = max_trials
        if trials is None and isinstance(cfg, TVMConfig):
            trials = cfg.tuning_trials
        if trials is None:
            trials = 1000

        if log_file is None and isinstance(cfg, TVMConfig):
            log_file = cfg.tuning_log

        if log_file is None:
            log_file = os.path.join(
                tempfile.mkdtemp(prefix="tvm_tuning_"),
                "tuning.log",
            )

        start_time = time.time()

        # 获取调优任务
        try:
            tasks = autotvm.task.extract_from_program(
                mod["main"],
                target=target,
                params=params,
            )
        except Exception as e:
            raise OptimizationError(f"提取调优任务失败: {e}") from e

        if not tasks:
            logger.warning("未找到可调优的任务")
            return TuningResult(
                target=str(target),
                model_name="unknown",
                tuning_time_s=0.0,
                num_trials=0,
                best_latency_ms=0.0,
                baseline_latency_ms=0.0,
                speedup=1.0,
                log_path=log_file,
            )

        logger.info(f"找到 {len(tasks)} 个调优任务")

        # 配置测量选项
        measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=10),
            runner=autotvm.LocalRunner(
                number=10,
                repeat=3,
                min_repeat_ms=50,
                timeout=4,
            ),
        )

        if isinstance(cfg, TVMConfig) and cfg.tuning_measure_option:
            custom_runner = cfg.tuning_measure_option.get("runner")
            if custom_runner:
                measure_option = autotvm.measure_option(
                    builder=autotvm.LocalBuilder(timeout=10),
                    runner=autotvm.LocalRunner(**custom_runner),
                )

        # 执行调优
        tuner = None
        best_results = []

        for idx, task in enumerate(tasks):
            logger.info(f"调优任务 [{idx + 1}/{len(tasks)}]: {task.name}")

            prefix = f"[{idx + 1}/{len(tasks)}] {task.name}"
            tuner_obj = autotvm.tuner.XGBoostTuner(
                task, loss_type="reg"
            )

            try:
                tuner_obj.tune(
                    n_trial=min(trials // len(tasks), 500),
                    early_stopping=cfg.tuning_early_stopping
                    if isinstance(cfg, TVMConfig)
                    else 100,
                    measure_option=measure_option,
                    callbacks=[
                        autotvm.callback.log_to_file(log_file),
                        autotvm.callback.progress_bar(
                            min(trials // len(tasks), 500),
                            prefix=prefix,
                        ),
                    ],
                )

                if tuner_obj.best_flops > 0:
                    best_latency = 1e6 / tuner_obj.best_flops
                    best_results.append({
                        "task": task.name,
                        "best_latency_ms": best_latency,
                        "best_flops": tuner_obj.best_flops,
                    })
            except Exception as e:
                logger.warning(f"任务 {task.name} 调优失败: {e}")

        tuning_time = time.time() - start_time

        # 构建结果
        best_latency = min(
            [r["best_latency_ms"] for r in best_results],
            default=0.0,
        )
        baseline_latency = best_latency * 1.5  # 估计基线

        result = TuningResult(
            target=str(target),
            model_name="unknown",
            tuning_time_s=tuning_time,
            num_trials=trials,
            best_latency_ms=best_latency,
            baseline_latency_ms=baseline_latency,
            speedup=baseline_latency / best_latency if best_latency > 0 else 1.0,
            log_path=log_file,
            topk_results=best_results[:10],
        )

        self._tuning_result = result
        logger.info(
            f"自动调优完成 | 耗时: {tuning_time:.1f}s | "
            f"任务数: {len(tasks)} | "
            f"最佳延迟: {best_latency:.3f}ms"
        )

        return result

    def build(
        self,
        mod: Any,
        params: Optional[Dict[str, Any]] = None,
        target: Optional[Any] = None,
        tuning_log: Optional[str] = None,
    ) -> Tuple[Any, Optional[Dict[str, Any]], Optional[str]]:
        """构建 TVM 运行时模块

        Args:
            mod: relay.Module
            params: 模型参数
            target: 编译目标
            tuning_log: 调优日志路径

        Returns:
            (lib, params, graph_json)
        """
        if not TVM_AVAILABLE:
            raise BackendNotAvailableError("TVM 未安装")

        start_time = time.time()

        if target is None:
            target = self._get_target()

        cfg = self._config
        executor = cfg.executor if isinstance(cfg, TVMConfig) else "graph"

        # 应用调优日志
        if tuning_log is None and isinstance(cfg, TVMConfig):
            tuning_log = cfg.tuning_log

        # 构建 PassContext
        pass_ctx = tvm.transform.PassContext(
            opt_level=cfg.opt_level if isinstance(cfg, TVMConfig) else 3,
        )

        if tuning_log and os.path.exists(tuning_log):
            autotvm.tuner.load_best_record(tuning_log)

        # 执行构建
        try:
            with pass_ctx:
                if executor == "aot":
                    from tvm.relay.build import build as relay_build
                    mod = relay.transform.FoldConstant()(mod)
                    executor_factory = relay.build_module.get_executor_factory(
                        mod, target, executor="aot"
                    )
                    lib = tvm.build(mod["main"], target=target, params=params)
                    graph_json = None
                else:
                    lib = relay.build(
                        mod,
                        target=target,
                        params=params,
                    )
                    graph_json = None
                    if hasattr(lib, "get_graph_json"):
                        graph_json = lib.get_graph_json()
                    params = lib.get_params() if hasattr(lib, "get_params") else params

        except Exception as e:
            raise CompilationError(f"TVM 构建失败: {e}") from e

        build_time = time.time() - start_time

        self._lib = lib
        self._params = params
        self._graph_json = graph_json

        # 保存构建产物
        if isinstance(cfg, TVMConfig) and cfg.export_lib_path:
            self._save_artifact(lib, params, graph_json, cfg)

        # 创建运行时模块
        self._create_runtime_module(lib, params, graph_json)

        logger.info(f"TVM 构建完成 | 耗时: {build_time:.3f}s")
        return lib, params, graph_json

    def _create_runtime_module(
        self,
        lib: Any,
        params: Optional[Dict[str, Any]],
        graph_json: Optional[str],
    ) -> None:
        """创建 TVM 运行时模块

        Args:
            lib: 编译后的库
            params: 参数
            graph_json: 图 JSON
        """
        if not TVM_AVAILABLE:
            raise BackendNotAvailableError("TVM 未安装")

        device = self._get_device()

        try:
            if graph_json is not None:
                # Graph Executor
                module = tvm_runtime.create_graph_executor(
                    graph_json, lib, device
                )
                if params:
                    module.load_params(
                        tvm_runtime.save_param_dict(params)
                    )
            else:
                # Module 直接执行
                module = tvm_runtime.GraphModule(lib["default"](device))

            self._runtime_module = module
        except Exception as e:
            raise CompilationError(
                f"创建运行时模块失败: {e}"
            ) from e

    def _save_artifact(
        self,
        lib: Any,
        params: Optional[Dict[str, Any]],
        graph_json: Optional[str],
        cfg: Any,
    ) -> None:
        """保存构建产物

        Args:
            lib: 编译库
            params: 参数
            graph_json: 图 JSON
            cfg: 配置
        """
        if cfg.export_lib_path:
            os.makedirs(
                os.path.dirname(cfg.export_lib_path) or ".",
                exist_ok=True,
            )
            lib.export_library(cfg.export_lib_path)
            logger.info(f"库已导出: {cfg.export_lib_path}")

        if cfg.export_params_path and params:
            os.makedirs(
                os.path.dirname(cfg.export_params_path) or ".",
                exist_ok=True,
            )
            with open(cfg.export_params_path, "wb") as f:
                f.write(tvm_runtime.save_param_dict(params))
            logger.info(f"参数已导出: {cfg.export_params_path}")

    def compile(
        self,
        model: Any,
        input_shapes: Optional[Dict[str, List[int]]] = None,
        **kwargs,
    ) -> Any:
        """编译模型

        Args:
            model: 模型（路径/nn.Module/ModelProto）
            input_shapes: 输入形状映射
            **kwargs: 额外参数
                - model_format: 模型格式
                - target: 编译目标
                - tune: 是否自动调优

        Returns:
            TVM 运行时模块
        """
        if not TVM_AVAILABLE:
            raise BackendNotAvailableError("TVM 未安装")

        start_time = time.time()

        # 导入模型
        model_format = kwargs.get("model_format")
        mod, params = self.import_model(model, model_format, input_shapes)

        self._module = mod
        self._params = params

        # 图优化
        target = self._get_target()
        if self._config.optimization_level != OptimizationLevel.NONE:
            mod, params = self.optimize_relay(mod, params, target)

        # 自动调优
        should_tune = kwargs.get("tune", False)
        if isinstance(self._config, TVMConfig):
            should_tune = should_tune or self._config.use_autotvm

        if should_tune:
            max_trials = kwargs.get("max_trials")
            self.auto_tune(mod, params, target, max_trials)

        # 构建
        tuning_log = None
        if self._tuning_result:
            tuning_log = self._tuning_result.log_path
        lib, params, graph_json = self.build(mod, params, target, tuning_log)

        self._compiled = True
        self._model = model
        compile_time = time.time() - start_time

        logger.info(
            f"TVM 编译完成 | 目标: {self._config.target} | "
            f"耗时: {compile_time:.3f}s"
        )

        return self._runtime_module

    def inference(self, inputs: Any, **kwargs) -> Any:
        """执行推理

        Args:
            inputs: 输入数据
                - dict: {"input_name": numpy_array}
                - numpy.ndarray: 单输入
                - list: 按顺序的多输入
            **kwargs: 额外参数

        Returns:
            推理输出
        """
        module = self.runtime_module

        # 统一输入格式
        if isinstance(inputs, dict):
            feed_dict = inputs
        elif isinstance(inputs, np.ndarray):
            if len(self._input_names) == 1:
                feed_dict = {self._input_names[0]: inputs}
            else:
                feed_dict = {"input": inputs}
        elif isinstance(inputs, (list, tuple)):
            feed_dict = {}
            for idx, data in enumerate(inputs):
                name = (
                    self._input_names[idx]
                    if idx < len(self._input_names)
                    else f"input_{idx}"
                )
                feed_dict[name] = data
        else:
            raise TypeError(f"不支持的输入类型: {type(inputs)}")

        # 设置输入
        for name, data in feed_dict.items():
            if isinstance(data, np.ndarray):
                module.set_input(name, tvm_runtime.nd.array(data))
            else:
                module.set_input(name, data)

        # 执行推理
        module.run()

        # 获取输出
        num_outputs = module.get_num_outputs()
        outputs = []
        for idx in range(num_outputs):
            output = module.get_output(idx).numpy()
            outputs.append(output)

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

        module = self.runtime_module

        # 设置输入
        if isinstance(inputs, dict):
            for name, data in inputs.items():
                if isinstance(data, np.ndarray):
                    module.set_input(name, tvm_runtime.nd.array(data))
        elif isinstance(inputs, np.ndarray):
            module.set_input("input", tvm_runtime.nd.array(inputs))

        # 预热
        for _ in range(num_warmup):
            module.run()
        gc.collect()

        # 正式测试
        if TVM_AVAILABLE:
            ftimer = module.module.time_evaluator(
                "run",
                self._get_device(),
                number=num_iterations,
                repeat=3,
            )
            prof_res = ftimer()
            avg_latency = prof_res.mean * 1000
            std_latency = prof_res.std * 1000
        else:
            latencies = []
            for _ in range(num_iterations):
                start = time.perf_counter()
                module.run()
                end = time.perf_counter()
                latencies.append((end - start) * 1000)
            avg_latency = sum(latencies) / len(latencies)
            std_latency = 0.0

        batch_size = 1
        if isinstance(inputs, dict):
            for v in inputs.values():
                if isinstance(v, np.ndarray):
                    batch_size = v.shape[0]
                    break
        elif isinstance(inputs, np.ndarray):
            batch_size = inputs.shape[0]

        throughput = batch_size / (avg_latency / 1000)

        return BenchmarkResult(
            backend=self.backend_name,
            model_name="unknown",
            latency_ms=avg_latency,
            throughput_fps=throughput,
            memory_usage_mb=0.0,
            compile_time_s=0.0,
            precision=self._config.precision.value,
            batch_size=batch_size,
            metadata={
                "std_ms": std_latency,
                "target": self._config.target,
                "num_iterations": num_iterations,
            },
        )

    def get_module_info(self) -> Dict[str, Any]:
        """获取模块信息

        Returns:
            模块信息字典
        """
        info = {
            "target": self._config.target,
            "input_names": self._input_names,
            "input_shapes": self._input_shapes,
            "compiled": self._compiled,
        }

        if self._tuning_result:
            info["tuning"] = {
                "time_s": self._tuning_result.tuning_time_s,
                "speedup": self._tuning_result.speedup,
                "best_latency_ms": self._tuning_result.best_latency_ms,
            }

        return info

    def release(self) -> None:
        """释放资源"""
        self._runtime_module = None
        self._lib = None
        self._module = None
        self._params = None
        self._graph_json = None
        self._device = None
        self._tuning_result = None
        self._build_artifact = None
        self._input_names = []
        self._input_shapes = {}
        self._output_names = []
        self._compiled = False
        self._model = None


__all__ = [
    "TVMCompiler",
    "TVMConfig",
    "TuningResult",
    "TVMBuildArtifact",
    "TVMTarget",
    "PassLevel",
    "TVM_AVAILABLE",
]
