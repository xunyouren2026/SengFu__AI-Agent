"""
ONNX Model Optimization Module

Provides graph optimization passes (constant folding, dead code elimination,
common subexpression elimination), operator fusion, quantization, and
shape inference for ONNX models.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
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
# Enums
# ---------------------------------------------------------------------------

class DataType(IntEnum):
    UNDEFINED = 0
    FLOAT = 1
    UINT8 = 2
    INT8 = 3
    UINT16 = 4
    INT16 = 5
    INT32 = 6
    INT64 = 7
    STRING = 8
    BOOL = 9
    FLOAT16 = 10
    DOUBLE = 11
    UINT32 = 12
    UINT64 = 13
    COMPLEX64 = 14
    COMPLEX128 = 15


class OptimizerPass(Enum):
    CONSTANT_FOLDING = "constant_folding"
    DEAD_CODE_ELIMINATION = "dead_code_elimination"
    COMMON_SUBEXPRESSION_ELIMINATION = "common_subexpression_elimination"
    OPERATOR_FUSION = "operator_fusion"
    SHAPE_INFERENCE = "shape_inference"
    QUANTIZATION = "quantization"
    REDUNDANT_NODE_REMOVAL = "redundant_node_removal"
    IDENTITY_REMOVAL = "identity_removal"


class QuantizationMode(Enum):
    NONE = "none"
    DYNAMIC = "dynamic"
    STATIC = "static"
    QAT = "qat"


class OpsetVersion(IntEnum):
    V7 = 7
    V8 = 8
    V9 = 9
    V10 = 10
    V11 = 11
    V12 = 12
    V13 = 13
    V14 = 14
    V15 = 15
    V16 = 16
    V17 = 17
    V18 = 18
    V19 = 19
    V20 = 20
    V21 = 21


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class TensorShape:
    """Shape of a tensor."""
    dims: List[int] = field(default_factory=list)
    is_dynamic: bool = False

    @property
    def rank(self) -> int:
        return len(self.dims)

    @property
    def size(self) -> int:
        result = 1
        for d in self.dims:
            if d > 0:
                result *= d
        return result

    @property
    def is_scalar(self) -> bool:
        return len(self.dims) == 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TensorShape):
            return NotImplemented
        return self.dims == other.dims

    def __hash__(self) -> int:
        return hash(tuple(self.dims))

    def __str__(self) -> str:
        return str(self.dims)


@dataclass
class ONNXTensor:
    """Represents a tensor in the ONNX graph."""
    name: str = ""
    dtype: DataType = DataType.FLOAT
    shape: TensorShape = field(default_factory=TensorShape)
    is_constant: bool = False
    data: Optional[Any] = None
    doc_string: str = ""
    quantization_scale: Optional[float] = None
    quantization_zero_point: Optional[int] = None
    raw_data: bytes = b""

    @property
    def element_count(self) -> int:
        return self.shape.size

    @property
    def byte_size(self) -> int:
        dtype_sizes = {
            DataType.FLOAT: 4, DataType.DOUBLE: 8, DataType.FLOAT16: 2,
            DataType.INT8: 1, DataType.UINT8: 1, DataType.INT16: 2,
            DataType.UINT16: 2, DataType.INT32: 4, DataType.UINT32: 4,
            DataType.INT64: 8, DataType.UINT64: 8, DataType.BOOL: 1,
        }
        return self.element_count * dtype_sizes.get(self.dtype, 4)


@dataclass
class ONNXAttribute:
    """Attribute of an ONNX operator."""
    name: str = ""
    type: str = "int"
    int_value: int = 0
    float_value: float = 0.0
    string_value: str = ""
    ints_value: List[int] = field(default_factory=list)
    floats_value: List[float] = field(default_factory=list)
    graph_value: Optional[ONNXGraph] = None

    def get_value(self) -> Any:
        if self.type == "int":
            return self.int_value
        elif self.type == "float":
            return self.float_value
        elif self.type == "string":
            return self.string_value
        elif self.type == "ints":
            return self.ints_value
        elif self.type == "floats":
            return self.floats_value
        return self.int_value


@dataclass
class ONNXNode:
    """Represents a node in the ONNX graph."""
    name: str = ""
    op_type: str = ""
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    attributes: Dict[str, ONNXAttribute] = field(default_factory=dict)
    domain: str = ""
    doc_string: str = ""
    opset_version: int = 14

    def get_attribute(self, name: str, default: Any = None) -> Any:
        attr = self.attributes.get(name)
        if attr:
            return attr.get_value()
        return default

    def __hash__(self) -> str:
        key = f"{self.op_type}|{','.join(sorted(self.attributes.keys()))}"
        return hash(key)


@dataclass
class ONNXGraph:
    """Represents an ONNX computation graph."""
    name: str = ""
    nodes: List[ONNXNode] = field(default_factory=list)
    initializers: Dict[str, ONNXTensor] = field(default_factory=dict)
    inputs: List[ONNXTensor] = field(default_factory=list)
    outputs: List[ONNXTensor] = field(default_factory=list)
    value_info: Dict[str, ONNXTensor] = field(default_factory=dict)

    def get_node(self, name: str) -> Optional[ONNXNode]:
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def get_producer(self, tensor_name: str) -> Optional[ONNXNode]:
        for node in self.nodes:
            if tensor_name in node.outputs:
                return node
        return None

    def get_consumers(self, tensor_name: str) -> List[ONNXNode]:
        return [n for n in self.nodes if tensor_name in n.inputs]


@dataclass
class ONNXModel:
    """Represents an ONNX model."""
    ir_version: int = 7
    opset_version: int = 14
    producer_name: str = ""
    producer_version: str = ""
    domain: str = ""
    model_version: int = 0
    doc_string: str = ""
    graph: ONNXGraph = field(default_factory=ONNXGraph)
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Result of an optimization pass."""
    pass_name: str = ""
    nodes_before: int = 0
    nodes_after: int = 0
    nodes_removed: int = 0
    nodes_modified: int = 0
    execution_time: float = 0.0
    success: bool = True
    messages: List[str] = field(default_factory=list)


@dataclass
class ShapeInferenceResult:
    """Result of shape inference."""
    tensor_name: str = ""
    inferred_shape: Optional[TensorShape] = None
    dtype: Optional[DataType] = None
    is_complete: bool = False


# ---------------------------------------------------------------------------
# Constant Folder
# ---------------------------------------------------------------------------

class ConstantFolder:
    """Folds constant expressions in the ONNX graph."""

    def __init__(self, graph: ONNXGraph) -> None:
        self.graph = graph
        self._folded_count: int = 0

    def fold(self) -> OptimizationResult:
        start = time.time()
        nodes_before = len(self.graph.nodes)
        self._folded_count = 0

        changed = True
        max_iterations = 10
        iteration = 0
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            for node in list(self.graph.nodes):
                if self._try_fold_node(node):
                    changed = True

        nodes_after = len(self.graph.nodes)
        return OptimizationResult(
            pass_name="constant_folding",
            nodes_before=nodes_before,
            nodes_after=nodes_after,
            nodes_removed=nodes_before - nodes_after,
            nodes_modified=self._folded_count,
            execution_time=time.time() - start,
        )

    def _try_fold_node(self, node: ONNXNode) -> bool:
        if not self._all_inputs_constant(node):
            return False

        input_tensors = [self.graph.initializers.get(inp) for inp in node.inputs]
        if any(t is None for t in input_tensors):
            return False

        result = self._evaluate_node(node, input_tensors)
        if result is None:
            return False

        for inp_name in node.inputs:
            if inp_name in self.graph.initializers:
                del self.graph.initializers[inp_name]

        output_name = node.outputs[0] if node.outputs else ""
        if output_name:
            self.graph.initializers[output_name] = result
            self.graph.nodes.remove(node)
            self._folded_count += 1
            return True
        return False

    def _all_inputs_constant(self, node: ONNXNode) -> bool:
        for inp in node.inputs:
            if inp in self.graph.initializers:
                continue
            if inp in [i.name for i in self.graph.inputs]:
                return False
            return False
        return len(node.inputs) > 0

    def _evaluate_node(
        self, node: ONNXNode, inputs: List[Optional[ONNXTensor]]
    ) -> Optional[ONNXTensor]:
        op = node.op_type.lower()
        try:
            if op == "add":
                return self._binary_op(inputs, lambda a, b: a + b)
            elif op == "sub":
                return self._binary_op(inputs, lambda a, b: a - b)
            elif op == "mul":
                return self._binary_op(inputs, lambda a, b: a * b)
            elif op == "div":
                return self._binary_op(inputs, lambda a, b: a / b if b != 0 else 0)
            elif op == "pow":
                return self._binary_op(inputs, lambda a, b: a ** b)
            elif op == "sqrt":
                return self._unary_op(inputs, lambda a: math.sqrt(a))
            elif op == "exp":
                return self._unary_op(inputs, lambda a: math.exp(a))
            elif op == "log":
                return self._unary_op(inputs, lambda a: math.log(a) if a > 0 else 0)
            elif op == "abs":
                return self._unary_op(inputs, lambda a: abs(a))
            elif op == "neg":
                return self._unary_op(inputs, lambda a: -a)
            elif op == "transpose":
                return self._transpose(inputs, node)
            elif op == "reshape":
                return self._reshape(inputs, node)
            elif op == "squeeze":
                return self._squeeze(inputs, node)
            elif op == "unsqueeze":
                return self._unsqueeze(inputs, node)
            elif op == "gemm":
                return self._gemm(inputs, node)
            elif op == "matmul":
                return self._matmul(inputs)
            elif op == "clip":
                return self._clip(inputs, node)
            elif op == "relu":
                return self._unary_op(inputs, lambda a: max(0, a))
            elif op == "sigmoid":
                return self._unary_op(inputs, lambda a: 1.0 / (1.0 + math.exp(-a)))
            elif op == "tanh":
                return self._unary_op(inputs, lambda a: math.tanh(a))
            elif op == "leakyrelu":
                alpha = node.get_attribute("alpha", 0.01)
                return self._unary_op(inputs, lambda a: a if a >= 0 else alpha * a)
            elif op == "constant":
                value = node.get_attribute("value", 0)
                return ONNXTensor(name=node.outputs[0], data=value, is_constant=True)
        except Exception as exc:
            logger.debug("Cannot fold node %s: %s", node.name, exc)
        return None

    def _get_scalar(self, tensor: ONNXTensor) -> float:
        if tensor.data is not None:
            val = tensor.data
            if isinstance(val, (list, tuple)):
                return float(val[0]) if val else 0.0
            return float(val)
        return 0.0

    def _binary_op(
        self,
        inputs: List[Optional[ONNXTensor]],
        op: Callable[[float, float], float],
    ) -> Optional[ONNXTensor]:
        if len(inputs) < 2 or inputs[0] is None or inputs[1] is None:
            return None
        a = self._get_scalar(inputs[0])
        b = self._get_scalar(inputs[1])
        result = op(a, b)
        return ONNXTensor(
            name="", dtype=DataType.FLOAT,
            shape=inputs[0].shape,
            data=result, is_constant=True,
        )

    def _unary_op(
        self,
        inputs: List[Optional[ONNXTensor]],
        op: Callable[[float], float],
    ) -> Optional[ONNXTensor]:
        if not inputs or inputs[0] is None:
            return None
        a = self._get_scalar(inputs[0])
        result = op(a)
        return ONNXTensor(
            name="", dtype=inputs[0].dtype,
            shape=inputs[0].shape,
            data=result, is_constant=True,
        )

    def _transpose(self, inputs: List[Optional[ONNXTensor]], node: ONNXNode) -> Optional[ONNXTensor]:
        return None

    def _reshape(self, inputs: List[Optional[ONNXTensor]], node: ONNXNode) -> Optional[ONNXTensor]:
        if len(inputs) < 2 or inputs[1] is None:
            return None
        shape_data = inputs[1].data
        if isinstance(shape_data, (list, tuple)):
            new_shape = TensorShape(dims=[int(d) for d in shape_data])
            return ONNXTensor(
                name="", dtype=inputs[0].dtype if inputs[0] else DataType.FLOAT,
                shape=new_shape, data=inputs[0].data, is_constant=True,
            )
        return None

    def _squeeze(self, inputs: List[Optional[ONNXTensor]], node: ONNXNode) -> Optional[ONNXTensor]:
        return None

    def _unsqueeze(self, inputs: List[Optional[ONNXTensor]], node: ONNXNode) -> Optional[ONNXTensor]:
        return None

    def _gemm(self, inputs: List[Optional[ONNXTensor]], node: ONNXNode) -> Optional[ONNXTensor]:
        return None

    def _matmul(self, inputs: List[Optional[ONNXTensor]]) -> Optional[ONNXTensor]:
        return None

    def _clip(self, inputs: List[Optional[ONNXTensor]], node: ONNXNode) -> Optional[ONNXTensor]:
        if not inputs or inputs[0] is None:
            return None
        val = self._get_scalar(inputs[0])
        min_val = node.get_attribute("min", float("-inf"))
        max_val = node.get_attribute("max", float("inf"))
        if isinstance(min_val, (int, float)):
            val = max(float(min_val), val)
        if isinstance(max_val, (int, float)):
            val = min(float(max_val), val)
        return ONNXTensor(
            name="", dtype=inputs[0].dtype,
            shape=inputs[0].shape, data=val, is_constant=True,
        )


# ---------------------------------------------------------------------------
# Dead Code Eliminator
# ---------------------------------------------------------------------------

class DeadCodeEliminator:
    """Removes dead code (unused nodes) from the ONNX graph."""

    def __init__(self, graph: ONNXGraph) -> None:
        self.graph = graph

    def eliminate(self) -> OptimizationResult:
        start = time.time()
        nodes_before = len(self.graph.nodes)

        output_names = {o.name for o in self.graph.outputs}
        used_tensors: Set[str] = set(output_names)

        for name in list(used_tensors):
            self._trace_inputs(name, used_tensors)

        nodes_to_remove = [
            node for node in self.graph.nodes
            if not any(out in used_tensors for out in node.outputs)
        ]

        for node in nodes_to_remove:
            self.graph.nodes.remove(node)

        nodes_after = len(self.graph.nodes)
        return OptimizationResult(
            pass_name="dead_code_elimination",
            nodes_before=nodes_before,
            nodes_after=nodes_after,
            nodes_removed=len(nodes_to_remove),
            execution_time=time.time() - start,
        )

    def _trace_inputs(self, tensor_name: str, used: Set[str]) -> None:
        for node in self.graph.nodes:
            if tensor_name in node.outputs:
                for inp in node.inputs:
                    if inp not in used:
                        used.add(inp)
                        self._trace_inputs(inp, used)


# ---------------------------------------------------------------------------
# Common Subexpression Eliminator
# ---------------------------------------------------------------------------

class CommonSubexpressionEliminator:
    """Eliminates common subexpressions from the ONNX graph."""

    def __init__(self, graph: ONNXGraph) -> None:
        self.graph = graph

    def eliminate(self) -> OptimizationResult:
        start = time.time()
        nodes_before = len(self.graph.nodes)
        seen: Dict[str, str] = {}
        replacements: Dict[str, str] = {}
        removed = 0

        for node in list(self.graph.nodes):
            sig = self._compute_signature(node)
            if sig in seen:
                original_output = seen[sig]
                new_output = node.outputs[0] if node.outputs else ""
                if new_output and original_output != new_output:
                    replacements[new_output] = original_output
                    self.graph.nodes.remove(node)
                    removed += 1
            else:
                if node.outputs:
                    seen[sig] = node.outputs[0]

        for node in self.graph.nodes:
            node.inputs = [replacements.get(inp, inp) for inp in node.inputs]

        nodes_after = len(self.graph.nodes)
        return OptimizationResult(
            pass_name="common_subexpression_elimination",
            nodes_before=nodes_before,
            nodes_after=nodes_after,
            nodes_removed=removed,
            execution_time=time.time() - start,
        )

    def _compute_signature(self, node: ONNXNode) -> str:
        attr_parts = []
        for key in sorted(node.attributes.keys()):
            attr = node.attributes[key]
            attr_parts.append(f"{key}={attr.get_value()}")
        return f"{node.op_type}({','.join(node.inputs)}|{','.join(attr_parts)})"


# ---------------------------------------------------------------------------
# Operator Fusion (ONNX)
# ---------------------------------------------------------------------------

class OperatorFusion:
    """Fuses compatible operators in the ONNX graph."""

    FUSION_PATTERNS: Dict[str, List[str]] = {
        "conv_bn": ["Conv", "BatchNormalization"],
        "conv_relu": ["Conv", "Relu"],
        "conv_sigmoid": ["Conv", "Sigmoid"],
        "conv_clip": ["Conv", "Clip"],
        "gemm_relu": ["Gemm", "Relu"],
        "matmul_relu": ["MatMul", "Relu"],
        "bn_relu": ["BatchNormalization", "Relu"],
        "add_relu": ["Add", "Relu"],
    }

    def __init__(self, graph: ONNXGraph) -> None:
        self.graph = graph
        self._fusions: List[Dict[str, Any]] = []

    def fuse(self) -> OptimizationResult:
        start = time.time()
        nodes_before = len(self.graph.nodes)
        total_fused = 0

        for pattern_name, pattern_ops in self.FUSION_PATTERNS.items():
            fused = self._try_fuse_pattern(pattern_name, pattern_ops)
            total_fused += fused

        nodes_after = len(self.graph.nodes)
        return OptimizationResult(
            pass_name="operator_fusion",
            nodes_before=nodes_before,
            nodes_after=nodes_after,
            nodes_removed=total_fused,
            execution_time=time.time() - start,
        )

    def _try_fuse_pattern(self, pattern_name: str, pattern_ops: List[str]) -> int:
        fused = 0
        i = 0
        while i < len(self.graph.nodes) - 1:
            current = self.graph.nodes[i]
            next_node = self.graph.nodes[i + 1]
            if current.op_type == pattern_ops[0] and next_node.op_type == pattern_ops[1]:
                if next_node.inputs and current.outputs:
                    if next_node.inputs[0] in current.outputs:
                        current.name = f"{current.name}_{next_node.op_type.lower()}"
                        self.graph.nodes.pop(i + 1)
                        fused += 1
                        self._fusions.append({
                            "pattern": pattern_name,
                            "primary": current.name,
                            "secondary": next_node.name,
                        })
                        continue
            i += 1
        return fused

    def get_fusion_history(self) -> List[Dict[str, Any]]:
        return list(self._fusions)


# ---------------------------------------------------------------------------
# Quantizer (ONNX)
# ---------------------------------------------------------------------------

class Quantizer:
    """Quantizes ONNX model operators."""

    def __init__(self, graph: ONNXGraph) -> None:
        self.graph = graph
        self._mode: QuantizationMode = QuantizationMode.DYNAMIC
        self._calibration_data: Dict[str, List[float]] = {}
        self._quantized_nodes: List[str] = []

    def set_mode(self, mode: QuantizationMode) -> None:
        self._mode = mode

    def add_calibration_data(self, tensor_name: str, data: List[float]) -> None:
        self._calibration_data[tensor_name] = data

    def quantize(self) -> OptimizationResult:
        start = time.time()
        nodes_before = len(self.graph.nodes)
        self._quantized_nodes.clear()

        quantizable_ops = {"Conv", "MatMul", "Gemm", "Add", "Mul", "ConvTranspose"}
        for node in self.graph.nodes:
            if node.op_type in quantizable_ops:
                self._quantize_node(node)
                self._quantized_nodes.append(node.name)

        nodes_after = len(self.graph.nodes)
        return OptimizationResult(
            pass_name="quantization",
            nodes_before=nodes_before,
            nodes_after=nodes_after,
            nodes_modified=len(self._quantized_nodes),
            execution_time=time.time() - start,
        )

    def _quantize_node(self, node: ONNXNode) -> None:
        for inp_name in node.inputs:
            tensor = self.graph.value_info.get(inp_name)
            if tensor and tensor.dtype == DataType.FLOAT:
                tensor.dtype = DataType.UINT8
                if inp_name in self._calibration_data:
                    data = self._calibration_data[inp_name]
                    if data:
                        min_val = min(data)
                        max_val = max(data)
                        if max_val > min_val:
                            tensor.quantization_scale = (max_val - min_val) / 255.0
                            tensor.quantization_zero_point = int(round(-min_val / tensor.quantization_scale))

    def get_quantized_nodes(self) -> List[str]:
        return list(self._quantized_nodes)


# ---------------------------------------------------------------------------
# Shape Inferencer
# ---------------------------------------------------------------------------

class ShapeInferencer:
    """Infers tensor shapes in the ONNX graph."""

    def __init__(self, graph: ONNXGraph) -> None:
        self.graph = graph
        self._inferred: Dict[str, ShapeInferenceResult] = {}

    def infer_all(self) -> List[ShapeInferenceResult]:
        self._inferred.clear()
        results: List[ShapeInferenceResult] = []

        for inp in self.graph.inputs:
            result = ShapeInferenceResult(
                tensor_name=inp.name,
                inferred_shape=inp.shape,
                dtype=inp.dtype,
                is_complete=all(d > 0 for d in inp.shape.dims),
            )
            self._inferred[inp.name] = result
            results.append(result)

        for node in self.graph.nodes:
            node_results = self._infer_node(node)
            results.extend(node_results)

        return results

    def _infer_node(self, node: ONNXNode) -> List[ShapeInferenceResult]:
        results: List[ShapeInferenceResult] = []
        op = node.op_type

        input_shapes = []
        for inp_name in node.inputs:
            if inp_name in self._inferred:
                input_shapes.append(self._inferred[inp_name].inferred_shape)
            elif inp_name in self.graph.initializers:
                init = self.graph.initializers[inp_name]
                input_shapes.append(init.shape)

        if op == "Conv":
            result = self._infer_conv(node, input_shapes)
            if result:
                results.append(result)
        elif op == "MatMul":
            result = self._infer_matmul(input_shapes)
            if result:
                results.append(result)
        elif op == "Gemm":
            result = self._infer_gemm(input_shapes)
            if result:
                results.append(result)
        elif op == "Relu" or op == "Sigmoid" or op == "Tanh":
            if input_shapes and node.outputs:
                result = ShapeInferenceResult(
                    tensor_name=node.outputs[0],
                    inferred_shape=input_shapes[0],
                    dtype=DataType.FLOAT,
                    is_complete=input_shapes[0].is_complete if input_shapes else False,
                )
                results.append(result)
                self._inferred[node.outputs[0]] = result
        elif op == "Add" or op == "Sub" or op == "Mul" or op == "Div":
            if input_shapes and node.outputs:
                result = ShapeInferenceResult(
                    tensor_name=node.outputs[0],
                    inferred_shape=input_shapes[0],
                    dtype=DataType.FLOAT,
                    is_complete=input_shapes[0].is_complete if input_shapes else False,
                )
                results.append(result)
                self._inferred[node.outputs[0]] = result
        elif op == "Reshape":
            if input_shapes and node.outputs:
                result = ShapeInferenceResult(
                    tensor_name=node.outputs[0],
                    inferred_shape=input_shapes[0],
                    dtype=DataType.FLOAT,
                    is_complete=False,
                )
                results.append(result)
                self._inferred[node.outputs[0]] = result
        elif op == "Transpose":
            if input_shapes and node.outputs:
                perm = node.get_attribute("perm", None)
                if perm and input_shapes[0]:
                    new_dims = [input_shapes[0].dims[i] for i in perm]
                    result = ShapeInferenceResult(
                        tensor_name=node.outputs[0],
                        inferred_shape=TensorShape(dims=new_dims),
                        dtype=DataType.FLOAT,
                        is_complete=all(d > 0 for d in new_dims),
                    )
                    results.append(result)
                    self._inferred[node.outputs[0]] = result
        elif op == "Pool":
            if input_shapes and node.outputs:
                result = self._infer_pool(node, input_shapes)
                if result:
                    results.append(result)
                    self._inferred[node.outputs[0]] = result
        elif op == "BatchNormalization":
            if input_shapes and node.outputs:
                result = ShapeInferenceResult(
                    tensor_name=node.outputs[0],
                    inferred_shape=input_shapes[0],
                    dtype=DataType.FLOAT,
                    is_complete=input_shapes[0].is_complete if input_shapes else False,
                )
                results.append(result)
                self._inferred[node.outputs[0]] = result
        elif op == "Softmax":
            if input_shapes and node.outputs:
                result = ShapeInferenceResult(
                    tensor_name=node.outputs[0],
                    inferred_shape=input_shapes[0],
                    dtype=DataType.FLOAT,
                    is_complete=input_shapes[0].is_complete if input_shapes else False,
                )
                results.append(result)
                self._inferred[node.outputs[0]] = result
        elif op == "Flatten":
            if input_shapes and node.outputs:
                axis = node.get_attribute("axis", 1)
                if input_shapes[0]:
                    before = 1
                    for d in input_shapes[0].dims[:axis]:
                        before *= d
                    after = 1
                    for d in input_shapes[0].dims[axis:]:
                        after *= d
                    result = ShapeInferenceResult(
                        tensor_name=node.outputs[0],
                        inferred_shape=TensorShape(dims=[before, after]),
                        dtype=DataType.FLOAT,
                        is_complete=input_shapes[0].is_complete,
                    )
                    results.append(result)
                    self._inferred[node.outputs[0]] = result
        elif op == "Dropout":
            if input_shapes and node.outputs:
                result = ShapeInferenceResult(
                    tensor_name=node.outputs[0],
                    inferred_shape=input_shapes[0],
                    dtype=DataType.FLOAT,
                    is_complete=input_shapes[0].is_complete if input_shapes else False,
                )
                results.append(result)
                self._inferred[node.outputs[0]] = result
        elif op == "Concat":
            if input_shapes and node.outputs:
                axis = node.get_attribute("axis", 0)
                concat_dim = sum(s.dims[axis] if axis < len(s.dims) else 0 for s in input_shapes)
                base_shape = list(input_shapes[0].dims) if input_shapes[0] else []
                if axis < len(base_shape):
                    base_shape[axis] = concat_dim
                result = ShapeInferenceResult(
                    tensor_name=node.outputs[0],
                    inferred_shape=TensorShape(dims=base_shape),
                    dtype=DataType.FLOAT,
                    is_complete=all(d > 0 for d in base_shape),
                )
                results.append(result)
                self._inferred[node.outputs[0]] = result
        elif op == "ReduceMean" or op == "ReduceSum" or op == "ReduceMax" or op == "ReduceMin":
            if input_shapes and node.outputs:
                axes = node.get_attribute("axes", None)
                keepdims = node.get_attribute("keepdims", 1)
                if input_shapes[0] and axes is not None:
                    new_dims = list(input_shapes[0].dims)
                    for ax in sorted(axes, reverse=True):
                        if ax < len(new_dims):
                            if keepdims:
                                new_dims[ax] = 1
                            else:
                                new_dims.pop(ax)
                    result = ShapeInferenceResult(
                        tensor_name=node.outputs[0],
                        inferred_shape=TensorShape(dims=new_dims),
                        dtype=DataType.FLOAT,
                        is_complete=all(d > 0 for d in new_dims),
                    )
                    results.append(result)
                    self._inferred[node.outputs[0]] = result

        return results

    def _infer_conv(self, node: ONNXNode, shapes: List[Optional[TensorShape]]) -> Optional[ShapeInferenceResult]:
        if not shapes or not shapes[0] or not node.outputs:
            return None
        in_shape = shapes[0].dims
        if len(in_shape) < 3:
            return None
        auto_pad = node.get_attribute("auto_pad", "NOTSET")
        kernel_shape = node.get_attribute("kernel_shape", [3, 3])
        strides = node.get_attribute("strides", [1, 1])
        pads = node.get_attribute("pads", [0, 0, 0, 0])
        dilations = node.get_attribute("dilations", [1, 1])
        group = node.get_attribute("group", 1)

        out_channels = node.get_attribute("kernel_shape", None)
        if out_channels is None and len(shapes) > 1 and shapes[1]:
            out_channels = shapes[1].dims[0]

        spatial_dims = len(in_shape) - 2
        output_spatial: List[int] = []
        for i in range(spatial_dims):
            k = kernel_shape[i] if i < len(kernel_shape) else 1
            s = strides[i] if i < len(strides) else 1
            d = dilations[i] if i < len(dilations) else 1
            effective_k = (k - 1) * d + 1
            pad_total = (pads[i] if i < len(pads) else 0) + (pads[i + spatial_dims] if i + spatial_dims < len(pads) else 0)
            out_dim = (in_shape[i + 2] + pad_total - effective_k) // s + 1
            output_spatial.append(max(1, out_dim))

        out_shape = [in_shape[0], out_channels if out_channels else in_shape[1]] + output_spatial
        return ShapeInferenceResult(
            tensor_name=node.outputs[0],
            inferred_shape=TensorShape(dims=out_shape),
            dtype=DataType.FLOAT,
            is_complete=all(d > 0 for d in out_shape),
        )

    def _infer_matmul(self, shapes: List[Optional[TensorShape]]) -> Optional[ShapeInferenceResult]:
        return None

    def _infer_gemm(self, shapes: List[Optional[TensorShape]]) -> Optional[ShapeInferenceResult]:
        return None

    def _infer_pool(self, node: ONNXNode, shapes: List[Optional[TensorShape]]) -> Optional[ShapeInferenceResult]:
        if not shapes or not shapes[0] or not node.outputs:
            return None
        in_shape = shapes[0].dims
        kernel_shape = node.get_attribute("kernel_shape", [2, 2])
        strides = node.get_attribute("strides", [1, 1])
        pads = node.get_attribute("pads", [0, 0, 0, 0])
        spatial_dims = len(in_shape) - 2
        output_spatial: List[int] = []
        for i in range(spatial_dims):
            k = kernel_shape[i] if i < len(kernel_shape) else 1
            s = strides[i] if i < len(strides) else 1
            pad_total = (pads[i] if i < len(pads) else 0) + (pads[i + spatial_dims] if i + spatial_dims < len(pads) else 0)
            out_dim = (in_shape[i + 2] + pad_total - k) // s + 1
            output_spatial.append(max(1, out_dim))
        out_shape = [in_shape[0], in_shape[1]] + output_spatial
        return ShapeInferenceResult(
            tensor_name=node.outputs[0],
            inferred_shape=TensorShape(dims=out_shape),
            dtype=DataType.FLOAT,
            is_complete=all(d > 0 for d in out_shape),
        )

    def get_inferred_shapes(self) -> Dict[str, ShapeInferenceResult]:
        return dict(self._inferred)


# ---------------------------------------------------------------------------
# ONNX Optimizer (Main Facade)
# ---------------------------------------------------------------------------

class ONNXOptimizer:
    """Main facade for ONNX model optimization."""

    def __init__(self, model: Optional[ONNXModel] = None) -> None:
        self.model = model or ONNXModel()
        self._results: List[OptimizationResult] = []
        self._enabled_passes: Set[OptimizerPass] = {
            OptimizerPass.CONSTANT_FOLDING,
            OptimizerPass.DEAD_CODE_ELIMINATION,
            OptimizerPass.COMMON_SUBEXPRESSION_ELIMINATION,
            OptimizerPass.OPERATOR_FUSION,
            OptimizerPass.SHAPE_INFERENCE,
            OptimizerPass.IDENTITY_REMOVAL,
        }

    def enable_pass(self, pass_name: OptimizerPass) -> None:
        self._enabled_passes.add(pass_name)

    def disable_pass(self, pass_name: OptimizerPass) -> None:
        self._enabled_passes.discard(pass_name)

    def optimize(self) -> List[OptimizationResult]:
        self._results.clear()
        graph = self.model.graph

        if OptimizerPass.CONSTANT_FOLDING in self._enabled_passes:
            result = ConstantFolder(graph).fold()
            self._results.append(result)

        if OptimizerPass.COMMON_SUBEXPRESSION_ELIMINATION in self._enabled_passes:
            result = CommonSubexpressionEliminator(graph).eliminate()
            self._results.append(result)

        if OptimizerPass.OPERATOR_FUSION in self._enabled_passes:
            result = OperatorFusion(graph).fuse()
            self._results.append(result)

        if OptimizerPass.IDENTITY_REMOVAL in self._enabled_passes:
            result = self._remove_identities(graph)
            self._results.append(result)

        if OptimizerPass.DEAD_CODE_ELIMINATION in self._enabled_passes:
            result = DeadCodeEliminator(graph).eliminate()
            self._results.append(result)

        if OptimizerPass.SHAPE_INFERENCE in self._enabled_passes:
            ShapeInferencer(graph).infer_all()

        if OptimizerPass.QUANTIZATION in self._enabled_passes:
            result = Quantizer(graph).quantize()
            self._results.append(result)

        return list(self._results)

    def _remove_identities(self, graph: ONNXGraph) -> OptimizationResult:
        start = time.time()
        nodes_before = len(graph.nodes)
        identity_nodes = [n for n in graph.nodes if n.op_type == "Identity"]
        for node in identity_nodes:
            if len(node.inputs) == 1 and len(node.outputs) == 1:
                old_name = node.outputs[0]
                new_name = node.inputs[0]
                for other in graph.nodes:
                    other.inputs = [new_name if inp == old_name else inp for inp in other.inputs]
                graph.nodes.remove(node)
        nodes_after = len(graph.nodes)
        return OptimizationResult(
            pass_name="identity_removal",
            nodes_before=nodes_before,
            nodes_after=nodes_after,
            nodes_removed=len(identity_nodes),
            execution_time=time.time() - start,
        )

    def get_results(self) -> List[OptimizationResult]:
        return list(self._results)

    def get_summary(self) -> Dict[str, Any]:
        total_removed = sum(r.nodes_removed for r in self._results)
        total_modified = sum(r.nodes_modified for r in self._results)
        total_time = sum(r.execution_time for r in self._results)
        return {
            "passes_run": len(self._results),
            "total_nodes_removed": total_removed,
            "total_nodes_modified": total_modified,
            "total_time_seconds": total_time,
            "final_node_count": len(self.model.graph.nodes),
            "results": [
                {
                    "pass": r.pass_name,
                    "removed": r.nodes_removed,
                    "modified": r.nodes_modified,
                    "time": r.execution_time,
                }
                for r in self._results
            ],
        }
