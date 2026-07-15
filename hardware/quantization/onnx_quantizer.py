"""
ONNX 量化实现

模块路径: hardware/quantization/onnx_quantizer.py

提供ONNX模型的量化支持，包括:
- 动态量化 (Dynamic Quantization)
- 静态量化 (Static Quantization)
- 量化感知训练 (QAT)
- 权重-only量化
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Set
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import json
import warnings

# 尝试导入ONNX相关库
try:
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper, TensorProto
    from onnxruntime.quantization import (
        quantize_dynamic,
        quantize_static,
        CalibrationDataReader,
        QuantType,
        QuantFormat
    )
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    warnings.warn("ONNX or ONNX Runtime not available. ONNX quantization features will be disabled.")


@dataclass
class ONNXQuantizationConfig:
    """ONNX量化配置"""
    quant_format: str = "QOperator"  # "QOperator" or "QDQ"
    activation_type: str = "QUInt8"  # "QUInt8" or "QInt8"
    weight_type: str = "QInt8"  # "QUInt8" or "QInt8"
    optimize_model: bool = True
    per_channel: bool = True
    reduce_range: bool = False
    calibrate_method: str = "MinMax"  # "MinMax" or "Entropy" or "Percentile"
    percentile: float = 99.99
    optimize_level: Optional[int] = None
    use_external_data_format: bool = False
    extra_options: Dict[str, Any] = field(default_factory=dict)


class ONNXCalibrator(CalibrationDataReader if ONNX_AVAILABLE else object):
    """ONNX校准数据读取器"""

    def __init__(self, calibration_data: List[np.ndarray], input_name: str = "input"):
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime not available")

        self.calibration_data = calibration_data
        self.input_name = input_name
        self.index = 0

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        """获取下一个校准样本"""
        if self.index >= len(self.calibration_data):
            return None

        data = {self.input_name: self.calibration_data[self.index]}
        self.index += 1
        return data

    def rewind(self):
        """重置读取器"""
        self.index = 0


class ONNXQuantizer:
    """ONNX量化器"""

    def __init__(self, config: Optional[ONNXQuantizationConfig] = None):
        self.config = config or ONNXQuantizationConfig()

        if not ONNX_AVAILABLE:
            raise ImportError(
                "ONNX and ONNX Runtime are required for ONNX quantization. "
                "Please install: pip install onnx onnxruntime-gpu"
            )

    def export_to_onnx(
        self,
        model: nn.Module,
        dummy_input: torch.Tensor,
        output_path: str,
        opset_version: int = 17,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    ) -> str:
        """
        将PyTorch模型导出为ONNX

        Args:
            model: PyTorch模型
            dummy_input: 示例输入
            output_path: 输出路径
            opset_version: ONNX opset版本
            input_names: 输入名称
            output_names: 输出名称
            dynamic_axes: 动态轴配置

        Returns:
            输出文件路径
        """
        model.eval()

        if input_names is None:
            input_names = ["input"]

        if output_names is None:
            output_names = ["output"]

        if dynamic_axes is None:
            dynamic_axes = {
                input_names[0]: {0: "batch_size"},
                output_names[0]: {0: "batch_size"}
            }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                opset_version=opset_version,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True
            )

        # 验证模型
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)

        return str(output_path)

    def quantize_dynamic_onnx(
        self,
        input_model_path: str,
        output_model_path: str,
        weight_type: Optional[str] = None,
        optimize_model: bool = True
    ) -> str:
        """
        动态量化ONNX模型

        Args:
            input_model_path: 输入ONNX模型路径
            output_model_path: 输出量化模型路径
            weight_type: 权重量化类型 ("QInt8" or "QUInt8")
            optimize_model: 是否优化模型

        Returns:
            输出文件路径
        """
        if weight_type is None:
            weight_type = self.config.weight_type

        # 映射量化类型
        quant_type_map = {
            "QInt8": QuantType.QInt8,
            "QUInt8": QuantType.QUInt8
        }
        weight_quant_type = quant_type_map.get(weight_type, QuantType.QInt8)

        output_path = Path(output_model_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        quantize_dynamic(
            model_input=input_model_path,
            model_output=output_model_path,
            weight_type=weight_quant_type,
            optimize_model=optimize_model
        )

        return str(output_path)

    def quantize_static_onnx(
        self,
        input_model_path: str,
        output_model_path: str,
        calibration_data: List[np.ndarray],
        calibration_method: Optional[str] = None
    ) -> str:
        """
        静态量化ONNX模型

        Args:
            input_model_path: 输入ONNX模型路径
            output_model_path: 输出量化模型路径
            calibration_data: 校准数据
            calibration_method: 校准方法

        Returns:
            输出文件路径
        """
        if calibration_method is None:
            calibration_method = self.config.calibrate_method

        # 创建校准数据读取器
        calibrator = ONNXCalibrator(calibration_data)

        # 映射量化类型
        quant_type_map = {
            "QInt8": QuantType.QInt8,
            "QUInt8": QuantType.QUInt8
        }
        activation_quant_type = quant_type_map.get(self.config.activation_type, QuantType.QUInt8)
        weight_quant_type = quant_type_map.get(self.config.weight_type, QuantType.QInt8)

        # 映射量化格式
        format_map = {
            "QOperator": QuantFormat.QOperator,
            "QDQ": QuantFormat.QDQ
        }
        quant_format = format_map.get(self.config.quant_format, QuantFormat.QOperator)

        output_path = Path(output_model_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        quantize_static(
            model_input=input_model_path,
            model_output=output_model_path,
            calibration_data_reader=calibrator,
            quant_format=quant_format,
            activation_type=activation_quant_type,
            weight_type=weight_quant_type,
            optimize_model=self.config.optimize_model,
            per_channel=self.config.per_channel,
            reduce_range=self.config.reduce_range,
            calibrate_method=calibration_method
        )

        return str(output_path)

    def quantize_model(
        self,
        model: nn.Module,
        dummy_input: torch.Tensor,
        calibration_data: Optional[List[torch.Tensor]] = None,
        quantization_mode: str = "dynamic",
        output_path: str = "model_quantized.onnx"
    ) -> str:
        """
        端到端量化PyTorch模型到ONNX

        Args:
            model: PyTorch模型
            dummy_input: 示例输入
            calibration_data: 校准数据 (静态量化需要)
            quantization_mode: 量化模式 ("dynamic" or "static")
            output_path: 输出路径

        Returns:
            量化后的ONNX模型路径
        """
        # 首先导出到ONNX
        temp_onnx_path = str(Path(output_path).with_suffix(".temp.onnx"))
        self.export_to_onnx(model, dummy_input, temp_onnx_path)

        # 然后量化
        if quantization_mode == "dynamic":
            result_path = self.quantize_dynamic_onnx(temp_onnx_path, output_path)
        elif quantization_mode == "static":
            if calibration_data is None:
                raise ValueError("Static quantization requires calibration data")

            # 转换校准数据为numpy
            calib_data_np = [x.detach().cpu().numpy() for x in calibration_data]
            result_path = self.quantize_static_onnx(
                temp_onnx_path, output_path, calib_data_np
            )
        else:
            raise ValueError(f"Unknown quantization mode: {quantization_mode}")

        # 清理临时文件
        Path(temp_onnx_path).unlink(missing_ok=True)

        return result_path

    def create_inference_session(
        self,
        model_path: str,
        use_gpu: bool = False,
        providers: Optional[List[str]] = None
    ) -> ort.InferenceSession:
        """
        创建ONNX Runtime推理会话

        Args:
            model_path: ONNX模型路径
            use_gpu: 是否使用GPU
            providers: 执行提供者列表

        Returns:
            InferenceSession
        """
        if providers is None:
            if use_gpu:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )

        return session

    def benchmark_model(
        self,
        model_path: str,
        dummy_input: np.ndarray,
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """
        基准测试ONNX模型

        Args:
            model_path: ONNX模型路径
            dummy_input: 示例输入
            num_runs: 测试运行次数
            warmup_runs: 预热运行次数

        Returns:
            性能统计字典
        """
        session = self.create_inference_session(model_path)
        input_name = session.get_inputs()[0].name

        # 预热
        for _ in range(warmup_runs):
            session.run(None, {input_name: dummy_input})

        # 测试
        import time
        times = []

        for _ in range(num_runs):
            start = time.time()
            session.run(None, {input_name: dummy_input})
            times.append(time.time() - start)

        return {
            'mean_latency_ms': np.mean(times) * 1000,
            'median_latency_ms': np.median(times) * 1000,
            'p95_latency_ms': np.percentile(times, 95) * 1000,
            'p99_latency_ms': np.percentile(times, 99) * 1000,
            'throughput_samples_per_sec': 1.0 / np.mean(times)
        }

    def compare_models(
        self,
        fp32_model_path: str,
        quantized_model_path: str,
        test_input: np.ndarray
    ) -> Dict[str, Any]:
        """
        比较FP32和量化模型

        Args:
            fp32_model_path: FP32模型路径
            quantized_model_path: 量化模型路径
            test_input: 测试输入

        Returns:
            比较结果字典
        """
        # 创建会话
        fp32_session = self.create_inference_session(fp32_model_path)
        quant_session = self.create_inference_session(quantized_model_path)

        input_name_fp32 = fp32_session.get_inputs()[0].name
        input_name_quant = quant_session.get_inputs()[0].name

        # 运行推理
        fp32_output = fp32_session.run(None, {input_name_fp32: test_input})[0]
        quant_output = quant_session.run(None, {input_name_quant: test_input})[0]

        # 计算差异
        diff = np.abs(fp32_output - quant_output)
        relative_diff = diff / (np.abs(fp32_output) + 1e-8)

        return {
            'max_absolute_error': float(np.max(diff)),
            'mean_absolute_error': float(np.mean(diff)),
            'max_relative_error': float(np.max(relative_diff)),
            'mean_relative_error': float(np.mean(relative_diff)),
            'cosine_similarity': float(
                np.dot(fp32_output.flatten(), quant_output.flatten()) /
                (np.linalg.norm(fp32_output) * np.linalg.norm(quant_output) + 1e-8)
            )
        }

    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """
        获取ONNX模型信息

        Args:
            model_path: ONNX模型路径

        Returns:
            模型信息字典
        """
        model = onnx.load(model_path)

        # 计算模型大小
        model_size = Path(model_path).stat().st_size / (1024 * 1024)  # MB

        # 获取输入输出信息
        session = self.create_inference_session(model_path)

        inputs = []
        for inp in session.get_inputs():
            inputs.append({
                'name': inp.name,
                'shape': inp.shape,
                'type': inp.type
            })

        outputs = []
        for out in session.get_outputs():
            outputs.append({
                'name': out.name,
                'shape': out.shape,
                'type': out.type
            })

        # 统计节点和初始化器
        num_nodes = len(model.graph.node)
        num_initializers = len(model.graph.initializer)

        # 检查是否量化
        is_quantized = any(
            op_type in ['QLinearConv', 'QLinearMatMul', 'QuantizeLinear', 'DequantizeLinear']
            for op_type in [node.op_type for node in model.graph.node]
        )

        return {
            'model_size_mb': model_size,
            'num_nodes': num_nodes,
            'num_initializers': num_initializers,
            'inputs': inputs,
            'outputs': outputs,
            'is_quantized': is_quantized,
            'opset_version': model.opset_import[0].version if model.opset_import else None
        }


def export_and_quantize(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: str = "model_quantized.onnx",
    quantization_mode: str = "dynamic",
    calibration_data: Optional[List[torch.Tensor]] = None,
    **kwargs
) -> str:
    """
    导出并量化便捷函数

    Args:
        model: PyTorch模型
        dummy_input: 示例输入
        output_path: 输出路径
        quantization_mode: 量化模式
        calibration_data: 校准数据
        **kwargs: 其他配置参数

    Returns:
        量化后的模型路径
    """
    config = ONNXQuantizationConfig(**kwargs)
    quantizer = ONNXQuantizer(config)
    return quantizer.quantize_model(
        model, dummy_input, calibration_data, quantization_mode, output_path
    )


def quantize_onnx_model(
    input_model_path: str,
    output_model_path: str,
    quantization_mode: str = "dynamic",
    calibration_data: Optional[List[np.ndarray]] = None,
    **kwargs
) -> str:
    """
    量化现有ONNX模型便捷函数

    Args:
        input_model_path: 输入模型路径
        output_model_path: 输出模型路径
        quantization_mode: 量化模式
        calibration_data: 校准数据
        **kwargs: 其他配置参数

    Returns:
        输出模型路径
    """
    config = ONNXQuantizationConfig(**kwargs)
    quantizer = ONNXQuantizer(config)

    if quantization_mode == "dynamic":
        return quantizer.quantize_dynamic_onnx(input_model_path, output_model_path)
    elif quantization_mode == "static":
        if calibration_data is None:
            raise ValueError("Static quantization requires calibration data")
        return quantizer.quantize_static_onnx(
            input_model_path, output_model_path, calibration_data
        )
    else:
        raise ValueError(f"Unknown quantization mode: {quantization_mode}")
