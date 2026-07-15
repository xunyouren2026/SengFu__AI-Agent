"""
TFLite Model Conversion Module

Provides quantization (int8, float16, full integer), model optimization,
operator fusion, flatbuffer management, and metadata embedding for TFLite models.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import struct
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
    Set,
    Tuple,
)

try:
    import numpy as _np
except ImportError:
    _np = None  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TFLITE_HEADER_SIZE = 16
TFLITE_MAGIC = b"TFL3"
FLATBUFFER_MAGIC = 0x464C5442

DEFAULT_INT8_MIN = -127.0
DEFAULT_INT8_MAX = 127.0
DEFAULT_FLOAT16_MIN = -65504.0
DEFAULT_FLOAT16_MAX = 65504.0

QUANTIZATION_PARAMS_SIZE = 12  # scale (4) + zero_point (4) + quantized_dimension (4)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class QuantizationType(Enum):
    NONE = "none"
    DYNAMIC = "dynamic"
    FLOAT16 = "float16"
    INT8 = "int8"
    FULL_INTEGER = "full_integer"
    UINT8 = "uint8"


class OptimizationLevel(Enum):
    NONE = 0
    DEFAULT = 1
    AGGRESSIVE = 2
    EXPERIMENTAL = 3


class TensorType(IntEnum):
    FLOAT32 = 0
    INT32 = 1
    UINT8 = 2
    INT64 = 3
    STRING = 4
    BOOL = 5
    INT16 = 6
    COMPLEX64 = 7
    INT8 = 9
    FLOAT16 = 10
    UINT16 = 17
    COMPLEX128 = 18
    UINT64 = 19
    RESOURCE = 20
    VARIANT = 21
    UINT32 = 22
    FLOAT64 = 23
    COMPLEX32 = 24


class BuiltinOperator(IntEnum):
    ADD = 0
    AVERAGE_POOL_2D = 1
    CONV_2D = 3
    DEPTHWISE_CONV_2D = 25
    DEQUANTIZE = 18
    FULLY_CONNECTED = 9
    L2_NORMALIZATION = 10
    LOGISTIC = 15
    MAX_POOL_2D = 2
    MUL = 16
    QUANTIZE = 19
    RELU = 10
    RELU_N1_TO_1 = 11
    RELU6 = 12
    RESHAPE = 14
    SOFTMAX = 13
    TANH = 17
    CONCATENATION = 22
    SPLIT = 23
    PAD = 28
    TRANSPOSE = 36
    MEAN = 32
    SUB = 21
    DIV = 20
    SQUEEZE = 38
    EXP = 47
    LOG = 48
    POW = 50
    PRELU = 52
    LEAKY_RELU = 34
    BATCH_TO_SPACE_ND = 39
    SPACE_TO_BATCH_ND = 40
    STRIDED_SLICE = 43
    GATHER = 49
    PACK = 29
    UNPACK = 30
    RESIZE_BILINEAR = 44
    FLOOR = 33
    CEIL = 35
    ROUND = 37
    MIRROR_PAD = 41
    ABS = 46
    SIN = 51
    COS = 53
    EXPONENTIATION = 54
    TOPK_V2 = 55
    LOG_SOFTMAX = 56


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class QuantizationParams:
    """Quantization parameters for a tensor."""
    scale: float = 1.0
    zero_point: int = 0
    quantized_dimension: int = 0
    quantization_type: QuantizationType = QuantizationType.NONE
    min_value: float = 0.0
    max_value: float = 0.0

    def quantize(self, value: float) -> int:
        if self.scale == 0:
            return 0
        return int(round(value / self.scale) + self.zero_point)

    def dequantize(self, quantized: int) -> float:
        return (quantized - self.zero_point) * self.scale

    def calculate_scale(self, min_val: float, max_val: float, num_bits: int = 8) -> None:
        qmin = -(2 ** (num_bits - 1))
        qmax = 2 ** (num_bits - 1) - 1
        if max_val == min_val:
            self.scale = 1.0
            self.zero_point = 0
        else:
            self.scale = (max_val - min_val) / (qmax - qmin)
            self.zero_point = int(round(qmin - min_val / self.scale))
            self.zero_point = max(qmin, min(qmax, self.zero_point))
        self.min_value = min_val
        self.max_value = max_val


@dataclass
class TensorInfo:
    """Information about a tensor in the model."""
    name: str = ""
    shape: List[int] = field(default_factory=list)
    dtype: TensorType = TensorType.FLOAT32
    quantization: QuantizationParams = field(default_factory=QuantizationParams)
    buffer_index: int = -1
    is_variable: bool = False
    is_input: bool = False
    is_output: bool = False
    size_bytes: int = 0

    @property
    def element_count(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count

    @property
    def size_in_bytes(self) -> int:
        dtype_sizes = {
            TensorType.FLOAT32: 4, TensorType.INT32: 4, TensorType.UINT8: 1,
            TensorType.INT8: 1, TensorType.FLOAT16: 2, TensorType.INT16: 2,
            TensorType.BOOL: 1, TensorType.UINT16: 2,
        }
        return self.element_count * dtype_sizes.get(self.dtype, 4)


@dataclass
class OperatorInfo:
    """Information about an operator in the model."""
    op_code: int = 0
    builtin_code: BuiltinOperator = BuiltinOperator.ADD
    name: str = ""
    inputs: List[int] = field(default_factory=list)
    outputs: List[int] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)
    fused_activation: Optional[str] = None


@dataclass
class ConversionResult:
    """Result of a model conversion."""
    success: bool = False
    original_size: int = 0
    converted_size: int = 0
    compression_ratio: float = 0.0
    quantization_type: QuantizationType = QuantizationType.NONE
    optimization_level: OptimizationLevel = OptimizationLevel.NONE
    operators_fused: int = 0
    operators_removed: int = 0
    conversion_time: float = 0.0
    accuracy_drop: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def size_reduction_pct(self) -> float:
        if self.original_size == 0:
            return 0.0
        return (1.0 - self.converted_size / self.original_size) * 100


@dataclass
class FlatBufferHeader:
    """FlatBuffer file header."""
    root_offset: int = 0
    file_size: int = 0
    schema_identifier: str = ""
    schema_version: int = 0


@dataclass
class ModelMetadata:
    """Metadata embedded in a TFLite model."""
    name: str = ""
    description: str = ""
    version: str = ""
    author: str = ""
    license: str = ""
    min_version: str = ""
    created_at: str = ""
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    process_name: str = ""
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# FlatBuffer Manager
# ---------------------------------------------------------------------------

class FlatBufferManager:
    """Manages TFLite flatbuffer operations."""

    def __init__(self) -> None:
        self._tensors: Dict[int, TensorInfo] = {}
        self._operators: List[OperatorInfo] = []
        self._buffers: List[bytes] = []
        self._inputs: List[int] = []
        self._outputs: List[int] = []
        self._version: str = "1.0"

    def load_model(self, data: bytes) -> bool:
        if len(data) < TFLITE_HEADER_SIZE:
            return False
        self._version = "1.0"
        self._tensors.clear()
        self._operators.clear()
        self._buffers.clear()
        return True

    def add_tensor(self, tensor: TensorInfo) -> int:
        idx = len(self._tensors)
        tensor.buffer_index = len(self._buffers)
        self._tensors[idx] = tensor
        self._buffers.append(b"\x00" * tensor.size_in_bytes)
        return idx

    def add_operator(self, op: OperatorInfo) -> int:
        idx = len(self._operators)
        self._operators.append(op)
        return idx

    def set_inputs(self, indices: List[int]) -> None:
        self._inputs = indices
        for idx in indices:
            if idx in self._tensors:
                self._tensors[idx].is_input = True

    def set_outputs(self, indices: List[int]) -> None:
        self._outputs = indices
        for idx in indices:
            if idx in self._tensors:
                self._tensors[idx].is_output = True

    def get_tensor(self, index: int) -> Optional[TensorInfo]:
        return self._tensors.get(index)

    def get_operator(self, index: int) -> Optional[OperatorInfo]:
        if 0 <= index < len(self._operators):
            return self._operators[index]
        return None

    def get_model_size(self) -> int:
        total = TFLITE_HEADER_SIZE
        for buf in self._buffers:
            total += len(buf)
        total += len(self._tensors) * 64  # approximate tensor metadata
        total += len(self._operators) * 32  # approximate operator metadata
        return total

    def get_tensor_count(self) -> int:
        return len(self._tensors)

    def get_operator_count(self) -> int:
        return len(self._operators)

    def get_summary(self) -> Dict[str, Any]:
        return {
            "version": self._version,
            "tensor_count": len(self._tensors),
            "operator_count": len(self._operators),
            "input_count": len(self._inputs),
            "output_count": len(self._outputs),
            "buffer_count": len(self._buffers),
            "estimated_size": self.get_model_size(),
        }


# ---------------------------------------------------------------------------
# Quantizer
# ---------------------------------------------------------------------------

class Quantizer:
    """Handles model quantization operations."""

    def __init__(self, flatbuffer: FlatBufferManager) -> None:
        self.flatbuffer = flatbuffer
        self._calibration_data: Dict[int, List[float]] = {}
        self._per_tensor_stats: Dict[int, Dict[str, float]] = {}

    def collect_calibration_data(
        self, tensor_index: int, data: List[float]
    ) -> None:
        if tensor_index not in self._calibration_data:
            self._calibration_data[tensor_index] = []
        self._calibration_data[tensor_index].extend(data)

    def compute_statistics(self, tensor_index: int) -> Dict[str, float]:
        data = self._calibration_data.get(tensor_index, [])
        if not data:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
        n = len(data)
        min_val = min(data)
        max_val = max(data)
        mean = sum(data) / n
        variance = sum((x - mean) ** 2 for x in data) / n
        std = math.sqrt(variance)
        self._per_tensor_stats[tensor_index] = {
            "min": min_val, "max": max_val,
            "mean": mean, "std": std,
        }
        return self._per_tensor_stats[tensor_index]

    def quantize_tensor_int8(
        self,
        tensor_index: int,
        symmetric: bool = True,
    ) -> Tuple[List[int], QuantizationParams]:
        stats = self.compute_statistics(tensor_index)
        params = QuantizationParams(quantization_type=QuantizationType.INT8)

        if symmetric:
            max_abs = max(abs(stats["min"]), abs(stats["max"]))
            if max_abs == 0:
                max_abs = 1.0
            params.scale = max_abs / 127.0
            params.zero_point = 0
        else:
            params.calculate_scale(stats["min"], stats["max"], 8)

        data = self._calibration_data.get(tensor_index, [])
        quantized = [params.quantize(v) for v in data]
        return quantized, params

    def quantize_tensor_float16(
        self, tensor_index: int
    ) -> Tuple[List[int], QuantizationParams]:
        data = self._calibration_data.get(tensor_index, [])
        params = QuantizationParams(quantization_type=QuantizationType.FLOAT16)
        quantized = [self._float_to_float16_bits(v) for v in data]
        return quantized, params

    def quantize_model(
        self,
        quant_type: QuantizationType = QuantizationType.INT8,
        representative_data: Optional[Dict[int, List[float]]] = None,
    ) -> ConversionResult:
        start = time.time()
        result = ConversionResult(
            original_size=self.flatbuffer.get_model_size(),
            quantization_type=quant_type,
        )

        if representative_data:
            for idx, data in representative_data.items():
                self.collect_calibration_data(idx, data)

        tensors_quantized = 0
        for idx in range(self.flatbuffer.get_tensor_count()):
            tensor = self.flatbuffer.get_tensor(idx)
            if tensor is None:
                continue
            if tensor.dtype == TensorType.FLOAT32:
                if quant_type == QuantizationType.INT8:
                    _, params = self.quantize_tensor_int8(idx)
                    tensor.quantization = params
                    tensor.dtype = TensorType.INT8
                elif quant_type == QuantizationType.FLOAT16:
                    _, params = self.quantize_tensor_float16(idx)
                    tensor.quantization = params
                    tensor.dtype = TensorType.FLOAT16
                tensors_quantized += 1

        result.converted_size = self.flatbuffer.get_model_size()
        result.compression_ratio = result.original_size / max(result.converted_size, 1)
        result.conversion_time = time.time() - start
        result.success = True
        return result

    @staticmethod
    def _float_to_float16_bits(value: float) -> int:
        if _np is not None:
            return int(_np.float16(value).view(_np.uint16))
        sign = 1 if value >= 0 else -1
        value = abs(value)
        if value == 0:
            return 0
        if value >= 65504:
            return 0x7BFF if sign > 0 else 0xFBFF
        if value < 5.96e-8:
            return 0x0001 if sign > 0 else 0x8001
        exponent = int(math.floor(math.log2(value)))
        mantissa = value / (2 ** exponent) - 1.0
        biased_exp = exponent + 15
        if biased_exp <= 0:
            biased_exp = 0
            mantissa = value / (2 ** -14)
        sign_bit = 0 if sign > 0 else 1
        mantissa_bits = int(mantissa * 1024) & 0x3FF
        return (sign_bit << 15) | (biased_exp << 10) | mantissa_bits


# ---------------------------------------------------------------------------
# Model Optimizer
# ---------------------------------------------------------------------------

class ModelOptimizer:
    """Optimizes TFLite models with various optimization passes."""

    def __init__(self, flatbuffer: FlatBufferManager) -> None:
        self.flatbuffer = flatbuffer
        self._optimizations_applied: List[str] = []

    def optimize(
        self,
        level: OptimizationLevel = OptimizationLevel.DEFAULT,
    ) -> ConversionResult:
        start = time.time()
        result = ConversionResult(
            original_size=self.flatbuffer.get_model_size(),
            optimization_level=level,
        )

        if level.value >= OptimizationLevel.DEFAULT.value:
            ops_removed = self._remove_identity_operators()
            result.operators_removed += ops_removed
            self._optimizations_applied.append("remove_identity")

        if level.value >= OptimizationLevel.AGGRESSIVE.value:
            ops_fused = self._fuse_activation_functions()
            result.operators_fused += ops_fused
            self._optimizations_applied.append("fuse_activations")
            ops_removed = self._remove_redundant_reshapes()
            result.operators_removed += ops_removed
            self._optimizations_applied.append("remove_redundant_reshapes")
            self._fold_constants()
            self._optimizations_applied.append("constant_folding")

        if level.value >= OptimizationLevel.EXPERIMENTAL.value:
            self._optimize_batch_norm()
            self._optimizations_applied.append("batchnorm_fusion")

        result.converted_size = self.flatbuffer.get_model_size()
        result.compression_ratio = result.original_size / max(result.converted_size, 1)
        result.conversion_time = time.time() - start
        result.success = True
        return result

    def _remove_identity_operators(self) -> int:
        removed = 0
        ops = list(self.flatbuffer._operators)
        new_ops: List[OperatorInfo] = []
        for op in ops:
            if op.builtin_code == BuiltinOperator.RESHAPE:
                tensor = self.flatbuffer.get_tensor(op.inputs[0] if op.inputs else -1)
                if tensor and op.outputs:
                    out_tensor = self.flatbuffer.get_tensor(op.outputs[0])
                    if out_tensor and tensor.shape == out_tensor.shape:
                        removed += 1
                        continue
            new_ops.append(op)
        self.flatbuffer._operators = new_ops
        return removed

    def _fuse_activation_functions(self) -> int:
        fused = 0
        activation_ops = {
            BuiltinOperator.RELU, BuiltinOperator.RELU6,
            BuiltinOperator.LEAKY_RELU, BuiltinOperator.TANH,
            BuiltinOperator.LOGISTIC,
        }
        for op in self.flatbuffer._operators:
            if op.fused_activation is None:
                for i, out_idx in enumerate(op.outputs):
                    for next_op in self.flatbuffer._operators:
                        if next_op.inputs and next_op.inputs[0] == out_idx:
                            if next_op.builtin_code in activation_ops:
                                act_name = next_op.builtin_code.name
                                op.fused_activation = act_name
                                next_op.inputs = next_op.inputs[1:] if len(next_op.inputs) > 1 else []
                                fused += 1
                                break
        return fused

    def _remove_redundant_reshapes(self) -> int:
        removed = 0
        ops = list(self.flatbuffer._operators)
        for i in range(len(ops) - 1):
            if ops[i].builtin_code == BuiltinOperator.RESHAPE and \
               ops[i + 1].builtin_code == BuiltinOperator.RESHAPE:
                removed += 1
        return removed // 2

    def _fold_constants(self) -> int:
        folded = 0
        for op in self.flatbuffer._operators:
            if op.builtin_code in (BuiltinOperator.ADD, BuiltinOperator.MUL,
                                    BuiltinOperator.SUB, BuiltinOperator.DIV):
                if len(op.inputs) >= 2:
                    t0 = self.flatbuffer.get_tensor(op.inputs[0])
                    t1 = self.flatbuffer.get_tensor(op.inputs[1])
                    if t0 and t1 and not t0.is_input and not t1.is_input:
                        folded += 1
        return folded

    def _optimize_batch_norm(self) -> int:
        return 0

    def get_applied_optimizations(self) -> List[str]:
        return list(self._optimizations_applied)


# ---------------------------------------------------------------------------
# Operator Fusion
# ---------------------------------------------------------------------------

class OperatorFusion:
    """Handles operator fusion optimization passes."""

    FUSION_PATTERNS: Dict[str, Tuple[BuiltinOperator, BuiltinOperator]] = {
        "conv_relu": (BuiltinOperator.CONV_2D, BuiltinOperator.RELU),
        "conv_relu6": (BuiltinOperator.CONV_2D, BuiltinOperator.RELU6),
        "conv_batchnorm_relu": (BuiltinOperator.CONV_2D, BuiltinOperator.RELU),
        "dwconv_relu": (BuiltinOperator.DEPTHWISE_CONV_2D, BuiltinOperator.RELU),
        "fc_relu": (BuiltinOperator.FULLY_CONNECTED, BuiltinOperator.RELU),
        "add_relu": (BuiltinOperator.ADD, BuiltinOperator.RELU),
        "mul_relu": (BuiltinOperator.MUL, BuiltinOperator.RELU),
        "matmul_relu": (BuiltinOperator.FULLY_CONNECTED, BuiltinOperator.RELU),
    }

    def __init__(self, flatbuffer: FlatBufferManager) -> None:
        self.flatbuffer = flatbuffer
        self._fusions_performed: List[Dict[str, Any]] = []

    def find_fusion_opportunities(self) -> List[Dict[str, Any]]:
        opportunities: List[Dict[str, Any]] = []
        ops = self.flatbuffer._operators
        for i in range(len(ops) - 1):
            current = ops[i]
            next_op = ops[i + 1]
            for pattern_name, (primary, secondary) in self.FUSION_PATTERNS.items():
                if current.builtin_code == primary and next_op.builtin_code == secondary:
                    opportunities.append({
                        "pattern": pattern_name,
                        "primary_index": i,
                        "secondary_index": i + 1,
                        "primary_op": current.builtin_code.name,
                        "secondary_op": next_op.builtin_code.name,
                    })
        return opportunities

    def apply_fusions(self) -> int:
        opportunities = self.find_fusion_opportunities()
        fused = 0
        indices_to_remove: Set[int] = set()
        for opp in opportunities:
            sec_idx = opp["secondary_index"]
            if sec_idx not in indices_to_remove:
                primary_op = self.flatbuffer._operators[opp["primary_index"]]
                primary_op.fused_activation = opp["secondary_op"].lower()
                indices_to_remove.add(sec_idx)
                fused += 1
                self._fusions_performed.append(opp)

        if indices_to_remove:
            new_ops = [
                op for i, op in enumerate(self.flatbuffer._operators)
                if i not in indices_to_remove
            ]
            self.flatbuffer._operators = new_ops

        return fused

    def get_fusion_history(self) -> List[Dict[str, Any]]:
        return list(self._fusions_performed)


# ---------------------------------------------------------------------------
# Metadata Embedder
# ---------------------------------------------------------------------------

class MetadataEmbedder:
    """Embeds metadata into TFLite models."""

    def __init__(self, flatbuffer: FlatBufferManager) -> None:
        self.flatbuffer = flatbuffer
        self._metadata: ModelMetadata = ModelMetadata()

    def set_metadata(self, metadata: ModelMetadata) -> None:
        self._metadata = metadata

    def set_name(self, name: str) -> None:
        self._metadata.name = name

    def set_description(self, description: str) -> None:
        self._metadata.description = description

    def set_version(self, version: str) -> None:
        self._metadata.version = version

    def set_author(self, author: str) -> None:
        self._metadata.author = author

    def set_process(self, process_name: str) -> None:
        self._metadata.process_name = process_name

    def add_tag(self, tag: str) -> None:
        if tag not in self._metadata.tags:
            self._metadata.tags.append(tag)

    def add_extra(self, key: str, value: str) -> None:
        self._metadata.extra[key] = value

    def get_metadata(self) -> ModelMetadata:
        return self._metadata

    def embed(self) -> Dict[str, str]:
        if not self._metadata.created_at:
            self._metadata.created_at = datetime.utcnow().isoformat()
        if not self._metadata.input_names:
            self._metadata.input_names = [
                self.flatbuffer._tensors[i].name
                for i in self.flatbuffer._inputs
                if i in self.flatbuffer._tensors
            ]
        if not self._metadata.output_names:
            self._metadata.output_names = [
                self.flatbuffer._tensors[i].name
                for i in self.flatbuffer._outputs
                if i in self.flatbuffer._tensors
            ]
        return {
            "name": self._metadata.name,
            "description": self._metadata.description,
            "version": self._metadata.version,
            "author": self._metadata.author,
            "created_at": self._metadata.created_at,
            "inputs": ",".join(self._metadata.input_names),
            "outputs": ",".join(self._metadata.output_names),
            "tags": ",".join(self._metadata.tags),
            "process": self._metadata.process_name,
        }


# ---------------------------------------------------------------------------
# Accuracy Evaluator
# ---------------------------------------------------------------------------

class AccuracyEvaluator:
    """Evaluates model accuracy after quantization."""

    def __init__(self) -> None:
        self._original_outputs: List[List[float]] = []
        self._quantized_outputs: List[List[float]] = []
        self._labels: List[int] = []

    def add_comparison(
        self,
        original_output: List[float],
        quantized_output: List[float],
        label: Optional[int] = None,
    ) -> None:
        self._original_outputs.append(original_output)
        self._quantized_outputs.append(quantized_output)
        if label is not None:
            self._labels.append(label)

    def compute_mse(self) -> float:
        total = 0.0
        count = 0
        for orig, quant in zip(self._original_outputs, self._quantized_outputs):
            for o, q in zip(orig, quant):
                total += (o - q) ** 2
                count += 1
        return total / count if count > 0 else 0.0

    def compute_mae(self) -> float:
        total = 0.0
        count = 0
        for orig, quant in zip(self._original_outputs, self._quantized_outputs):
            for o, q in zip(orig, quant):
                total += abs(o - q)
                count += 1
        return total / count if count > 0 else 0.0

    def compute_top1_accuracy(self) -> float:
        if not self._labels:
            return 0.0
        correct = 0
        for orig, quant, label in zip(self._original_outputs, self._quantized_outputs, self._labels):
            orig_pred = orig.index(max(orig))
            quant_pred = quant.index(max(quant))
            if orig_pred == quant_pred == label:
                correct += 1
        return correct / len(self._labels) if self._labels else 0.0

    def compute_cosine_similarity(self) -> float:
        total = 0.0
        count = 0
        for orig, quant in zip(self._original_outputs, self._quantized_outputs):
            dot = sum(o * q for o, q in zip(orig, quant))
            norm_o = math.sqrt(sum(o ** 2 for o in orig))
            norm_q = math.sqrt(sum(q ** 2 for q in quant))
            if norm_o > 0 and norm_q > 0:
                total += dot / (norm_o * norm_q)
            count += 1
        return total / count if count > 0 else 0.0

    def get_report(self) -> Dict[str, float]:
        return {
            "mse": self.compute_mse(),
            "mae": self.compute_mae(),
            "rmse": math.sqrt(self.compute_mse()),
            "cosine_similarity": self.compute_cosine_similarity(),
            "top1_accuracy": self.compute_top1_accuracy(),
            "sample_count": len(self._original_outputs),
        }

    def clear(self) -> None:
        self._original_outputs.clear()
        self._quantized_outputs.clear()
        self._labels.clear()


# ---------------------------------------------------------------------------
# TFLite Converter (Main Facade)
# ---------------------------------------------------------------------------

class TFLiteConverter:
    """Main facade for TFLite model conversion and optimization."""

    def __init__(self) -> None:
        self.flatbuffer = FlatBufferManager()
        self.quantizer = Quantizer(self.flatbuffer)
        self.optimizer = ModelOptimizer(self.flatbuffer)
        self.operator_fusion = OperatorFusion(self.flatbuffer)
        self.metadata_embedder = MetadataEmbedder(self.flatbuffer)
        self.accuracy_evaluator = AccuracyEvaluator()
        self._conversion_history: List[ConversionResult] = []

    def load_model(self, data: bytes) -> bool:
        return self.flatbuffer.load_model(data)

    def convert(
        self,
        quantization: QuantizationType = QuantizationType.INT8,
        optimization: OptimizationLevel = OptimizationLevel.DEFAULT,
        representative_data: Optional[Dict[int, List[float]]] = None,
        metadata: Optional[ModelMetadata] = None,
    ) -> ConversionResult:
        if metadata:
            self.metadata_embedder.set_metadata(metadata)

        if optimization != OptimizationLevel.NONE:
            opt_result = self.optimizer.optimize(optimization)
            fusion_count = self.operator_fusion.apply_fusions()
            opt_result.operators_fused += fusion_count
        else:
            opt_result = ConversionResult(
                original_size=self.flatbuffer.get_model_size(),
                optimization_level=OptimizationLevel.NONE,
            )

        if quantization != QuantizationType.NONE:
            quant_result = self.quantizer.quantize_model(quantization, representative_data)
        else:
            quant_result = ConversionResult(
                original_size=self.flatbuffer.get_model_size(),
                converted_size=self.flatbuffer.get_model_size(),
            )

        final = ConversionResult(
            success=True,
            original_size=opt_result.original_size,
            converted_size=self.flatbuffer.get_model_size(),
            quantization_type=quantization,
            optimization_level=optimization,
            operators_fused=opt_result.operators_fused,
            operators_removed=opt_result.operators_removed,
            conversion_time=opt_result.conversion_time + quant_result.conversion_time,
            accuracy_drop=self.accuracy_evaluator.compute_mse(),
        )

        self._conversion_history.append(final)
        return final

    def get_model_summary(self) -> Dict[str, Any]:
        return self.flatbuffer.get_summary()

    def get_conversion_history(self) -> List[ConversionResult]:
        return list(self._conversion_history)
