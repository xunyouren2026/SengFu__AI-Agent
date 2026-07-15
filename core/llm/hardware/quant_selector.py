"""
Dynamic quantization strategy selector for LLM optimization.

This module provides:
- GPU model detection and capability analysis
- Quantization strategy recommendation (GPTQ, AWQ, FP8, INT8)
- Accuracy comparison between quantization methods
- Performance estimation
- Hardware compatibility checking

Author: AGI Unified Framework Team
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import time


class QuantStrategy(Enum):
    """
    Supported quantization strategies.
    
    FP16: Standard half-precision (baseline)
    FP32: Full precision (no quantization)
    BF16: BFloat16 (good for training)
    FP8: 8-bit floating point (E4M3/E5M2)
    INT8: 8-bit integer quantization
    INT4: 4-bit integer quantization
    GPTQ: GPTQ quantization method
    AWQ: Activation-Aware Weight quantization
    GGML: GGML/GGUF quantization
    """
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"
    GPTQ = "gptq"
    GPTQ_INT4 = "gptq_int4"
    AWQ = "awq"
    AWQ_INT4 = "awq_int4"
    GGML_Q4 = "ggml_q4"
    GGML_Q5 = "ggml_q5"
    GGML_Q8 = "ggml_q8"


class GPUArchitecture(Enum):
    """GPU architectures for compatibility checking."""
    AMPERE = "ampere"      # A100, RTX 30xx
    HOPPER = "hopper"      # H100
    ADA = "ada"            # RTX 40xx
    VOLTA = "volta"       # V100
    PASCAL = "pascal"      # V100 (older)
    UNKNOWN = "unknown"


@dataclass
class GPUCapability:
    """
    GPU hardware capabilities.
    
    Attributes:
        name: GPU model name
        architecture: GPU architecture family
        compute_capability: CUDA compute capability (e.g., 8.0)
        memory_gb: Total memory in GB
        memory_bandwidth_gbs: Memory bandwidth in GB/s
        tensor_cores: Number of tensor cores
        cuda_cores: Number of CUDA cores
        fp16_tflops: FP16 tensor TFLOPS
        fp32_tflops: FP32 tensor TFLOPS
        supports_bf16: Whether BF16 is supported
        supports_fp8: Whether FP8 is supported
        supports_int8_turing: Whether INT8 Turing tensor ops supported
    """
    name: str
    architecture: GPUArchitecture
    compute_capability: float
    memory_gb: float
    memory_bandwidth_gbs: float
    tensor_cores: int = 0
    cuda_cores: int = 0
    fp16_tflops: float = 0.0
    fp32_tflops: float = 0.0
    supports_bf16: bool = False
    supports_fp8: bool = False
    supports_int8_turing: bool = False
    
    def supports_quantization(self, strategy: QuantStrategy) -> bool:
        """Check if GPU supports a quantization strategy."""
        if strategy in (QuantStrategy.FP32, QuantStrategy.FP16, QuantStrategy.BF16):
            return True
        
        if strategy == QuantStrategy.FP8:
            return self.supports_fp8
        
        if strategy in (QuantStrategy.INT8, QuantStrategy.INT4, QuantStrategy.GPTQ, 
                        QuantStrategy.AWQ, QuantStrategy.GPTQ_INT4, QuantStrategy.AWQ_INT4):
            return True  # INT8 generally supported on modern GPUs
        
        if strategy in (QuantStrategy.GGML_Q4, QuantStrategy.GGML_Q5, QuantStrategy.GGML_Q8):
            return True
        
        return False
    
    def get_memory_bits_per_parameter(self, strategy: QuantStrategy) -> float:
        """Get memory bits per parameter for strategy."""
        bits_map = {
            QuantStrategy.FP32: 32,
            QuantStrategy.FP16: 16,
            QuantStrategy.BF16: 16,
            QuantStrategy.FP8: 8,
            QuantStrategy.INT8: 8,
            QuantStrategy.INT4: 4,
            QuantStrategy.GPTQ: 8,
            QuantStrategy.GPTQ_INT4: 4,
            QuantStrategy.AWQ: 4,
            QuantStrategy.AWQ_INT4: 4,
            QuantStrategy.GGML_Q4: 4,
            QuantStrategy.GGML_Q5: 5,
            QuantStrategy.GGML_Q8: 8,
        }
        return bits_map.get(strategy, 16)


@dataclass
class QuantizationResult:
    """
    Result of quantization strategy analysis.
    
    Attributes:
        strategy: Selected quantization strategy
        recommended: Whether this is the recommended strategy
        memory_reduction: Memory reduction ratio (0-1)
        estimated_accuracy_loss: Estimated accuracy loss (0-1)
        estimated_speedup: Estimated speedup factor
        compatibility_score: Compatibility score (0-1)
        warnings: List of warnings
        requirements: Resource requirements
    """
    strategy: QuantStrategy
    recommended: bool
    memory_reduction: float
    estimated_accuracy_loss: float
    estimated_speedup: float
    compatibility_score: float
    memory_gb: float
    warnings: List[str] = field(default_factory=list)
    requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantizationProfile:
    """
    Complete quantization profile for a model.
    
    Attributes:
        model_name: Name of the model
        base_precision: Original model precision
        target_strategy: Selected quantization strategy
        memory_before: Memory before quantization
        memory_after: Memory after quantization
        estimated_performance: Performance estimates
        config: Quantization configuration
    """
    model_name: str
    base_precision: QuantStrategy
    target_strategy: QuantStrategy
    memory_before_gb: float
    memory_after_gb: float
    compression_ratio: float
    estimated_inference_speedup: float
    estimated_throughput: float
    batch_size_multiplier: float
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccuracyBenchmark:
    """
    Accuracy benchmark result for quantization comparison.
    
    Attributes:
        strategy: Quantization strategy tested
        perplexity: Language modeling perplexity
        accuracy_tasks: Per-task accuracy scores
        overall_score: Overall quality score (0-100)
        compared_to_fp16: Relative to FP16 baseline
        sample_size: Number of samples tested
    """
    strategy: QuantStrategy
    perplexity: float = 0.0
    accuracy_tasks: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    compared_to_fp16: float = 0.0
    sample_size: int = 0


class GPUDetector:
    """
    Detects and analyzes GPU hardware.
    
    Supports:
    - NVIDIA GPU detection via CUDA
    - Architecture identification
    - Capability analysis
    - Fallback for non-CUDA environments
    """
    
    # Known GPU specifications
    KNOWN_GPUS: Dict[str, GPUCapability] = {
        "A100": GPUCapability(
            name="NVIDIA A100",
            architecture=GPUArchitecture.AMPERE,
            compute_capability=8.0,
            memory_gb=40.0,
            memory_bandwidth_gbs=1555.0,
            tensor_cores=432,
            cuda_cores=6912,
            fp16_tflops=312.0,
            fp32_tflops=19.5,
            supports_bf16=True,
            supports_fp8=True,
            supports_int8_turing=True,
        ),
        "A100-80GB": GPUCapability(
            name="NVIDIA A100 80GB",
            architecture=GPUArchitecture.AMPERE,
            compute_capability=8.0,
            memory_gb=80.0,
            memory_bandwidth_gbs=2039.0,
            tensor_cores=432,
            cuda_cores=6912,
            fp16_tflops=312.0,
            fp32_tflops=19.5,
            supports_bf16=True,
            supports_fp8=True,
            supports_int8_turing=True,
        ),
        "H100": GPUCapability(
            name="NVIDIA H100",
            architecture=GPUArchitecture.HOPPER,
            compute_capability=9.0,
            memory_gb=80.0,
            memory_bandwidth_gbs=3350.0,
            tensor_cores=528,
            cuda_cores=16896,
            fp16_tflops=989.0,
            fp32_tflops=67.0,
            supports_bf16=True,
            supports_fp8=True,
            supports_int8_turing=True,
        ),
        "RTX 4090": GPUCapability(
            name="NVIDIA RTX 4090",
            architecture=GPUArchitecture.ADA,
            compute_capability=8.9,
            memory_gb=24.0,
            memory_bandwidth_gbs=1008.0,
            tensor_cores=264,
            cuda_cores=16384,
            fp16_tflops=330.0,
            fp32_tflops=82.0,
            supports_bf16=True,
            supports_fp8=False,
            supports_int8_turing=True,
        ),
        "RTX 3090": GPUCapability(
            name="NVIDIA RTX 3090",
            architecture=GPUArchitecture.AMPERE,
            compute_capability=8.6,
            memory_gb=24.0,
            memory_bandwidth_gbs=936.0,
            tensor_cores=328,
            cuda_cores=10496,
            fp16_tflops=356.0,
            fp32_tflops=35.0,
            supports_bf16=True,
            supports_fp8=False,
            supports_int8_turing=True,
        ),
        "RTX 4080": GPUCapability(
            name="NVIDIA RTX 4080",
            architecture=GPUArchitecture.ADA,
            compute_capability=8.9,
            memory_gb=16.0,
            memory_bandwidth_gbs=716.0,
            tensor_cores=304,
            cuda_cores=9728,
            fp16_tflops=228.0,
            fp32_tflops=49.0,
            supports_bf16=True,
            supports_fp8=False,
            supports_int8_turing=True,
        ),
        "V100": GPUCapability(
            name="NVIDIA V100",
            architecture=GPUArchitecture.VOLTA,
            compute_capability=7.0,
            memory_gb=32.0,
            memory_bandwidth_gbs=900.0,
            tensor_cores=640,
            cuda_cores=5120,
            fp16_tflops=125.0,
            fp32_tflops=14.0,
            supports_bf16=False,
            supports_fp8=False,
            supports_int8_turing=True,
        ),
    }
    
    def __init__(self) -> None:
        """Initialize GPU detector."""
        self._detected_gpu: Optional[GPUCapability] = None
        self._is_cuda_available = False
    
    def detect(self) -> GPUCapability:
        """
        Detect GPU hardware.
        
        Returns:
            GPUCapability with detected GPU info
        """
        if self._detected_gpu is not None:
            return self._detected_gpu
        
        # Try CUDA detection
        gpu_info = self._try_cuda_detection()
        
        if gpu_info is not None:
            self._detected_gpu = gpu_info
            self._is_cuda_available = True
            return gpu_info
        
        # Fallback to CPU/mock detection
        self._detected_gpu = GPUCapability(
            name="Unknown GPU (CUDA unavailable)",
            architecture=GPUArchitecture.UNKNOWN,
            compute_capability=0.0,
            memory_gb=0.0,
            memory_bandwidth_gbs=0.0,
        )
        return self._detected_gpu
    
    def _try_cuda_detection() -> Optional[GPUCapability]:
        """Try to detect GPU using CUDA."""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,compute_capability",
                 "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    parts = lines[0].split(',')
                    name = parts[0].strip()
                    memory = float(parts[1].strip().split()[0])
                    capability = float(parts[2].strip())
                    
                    # Match to known GPU
                    for gpu_name, gpu_cap in GPUDetector.KNOWN_GPUS.items():
                        if gpu_name.lower() in name.lower():
                            return gpu_cap
                    
                    # Unknown GPU
                    return GPUCapability(
                        name=name,
                        architecture=GPUArchitecture.UNKNOWN,
                        compute_capability=capability,
                        memory_gb=memory,
                        memory_bandwidth_gbs=0.0,
                    )
        except Exception:
            pass
        
        return None
    
    def get_capability(self) -> GPUCapability:
        """Get detected GPU capability."""
        return self.detect()
    
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        if not self._is_cuda_available:
            self.detect()
        return self._is_cuda_available
    
    def check_support(self, strategy: QuantStrategy) -> Tuple[bool, str]:
        """
        Check if detected GPU supports a quantization strategy.
        
        Returns:
            Tuple of (is_supported, reason)
        """
        gpu = self.detect()
        
        if not gpu.supports_quantization(strategy):
            return False, f"GPU {gpu.name} does not support {strategy.value}"
        
        # Additional checks
        if strategy == QuantStrategy.FP8 and not gpu.supports_fp8:
            return False, "FP8 requires H100 or newer GPU"
        
        if strategy == QuantStrategy.BF16 and not gpu.supports_bf16:
            return False, "BF16 requires Ampere or newer GPU"
        
        return True, "Supported"


class AccuracyComparator:
    """
    Compares accuracy between quantization strategies.
    
    Provides estimates and benchmarks for:
    - Language modeling perplexity
    - Downstream task accuracy
    - Perplexity comparison
    """
    
    # Known accuracy benchmarks for popular strategies
    BENCHMARKS: Dict[str, Dict[QuantStrategy, float]] = {
        "llama-7b": {
            QuantStrategy.FP16: 5.68,
            QuantStrategy.INT8: 5.75,
            QuantStrategy.INT4: 6.15,
            QuantStrategy.GPTQ: 5.70,
            QuantStrategy.GPTQ_INT4: 6.10,
            QuantStrategy.AWQ: 5.69,
            QuantStrategy.AWQ_INT4: 6.05,
        },
        "llama-13b": {
            QuantStrategy.FP16: 4.74,
            QuantStrategy.INT8: 4.80,
            QuantStrategy.INT4: 5.10,
            QuantStrategy.GPTQ: 4.77,
            QuantStrategy.GPTQ_INT4: 5.05,
            QuantStrategy.AWQ: 4.76,
            QuantStrategy.AWQ_INT4: 5.00,
        },
        "llama-70b": {
            QuantStrategy.FP16: 3.32,
            QuantStrategy.INT8: 3.38,
            QuantStrategy.INT4: 3.65,
            QuantStrategy.GPTQ: 3.35,
            QuantStrategy.GPTQ_INT4: 3.60,
        },
        "mistral-7b": {
            QuantStrategy.FP16: 5.20,
            QuantStrategy.INT8: 5.28,
            QuantStrategy.INT4: 5.65,
            QuantStrategy.GPTQ: 5.22,
            QuantStrategy.GPTQ_INT4: 5.55,
        },
    }
    
    def __init__(self) -> None:
        """Initialize accuracy comparator."""
        self._custom_benchmarks: Dict[str, Dict[QuantStrategy, float]] = {}
    
    def add_benchmark(
        self,
        model_name: str,
        strategy: QuantStrategy,
        perplexity: float
    ) -> None:
        """Add a custom benchmark result."""
        if model_name not in self._custom_benchmarks:
            self._custom_benchmarks[model_name] = {}
        self._custom_benchmarks[model_name][strategy] = perplexity
    
    def get_benchmark(
        self,
        model_name: str,
        strategy: QuantStrategy
    ) -> Optional[AccuracyBenchmark]:
        """
        Get benchmark for a model and strategy.
        
        Args:
            model_name: Name of the model
            strategy: Quantization strategy
        
        Returns:
            AccuracyBenchmark or None if not available
        """
        # Check custom benchmarks
        benchmarks = self._custom_benchmarks.get(model_name, {})
        
        # Check known benchmarks
        if not benchmarks:
            for known_model, known_benchmarks in self.BENCHMARKS.items():
                if known_model.lower() in model_name.lower():
                    benchmarks = known_benchmarks
                    break
        
        perplexity = benchmarks.get(strategy)
        if perplexity is None:
            return None
        
        # Get FP16 baseline
        baseline_perplexity = benchmarks.get(QuantStrategy.FP16, perplexity)
        
        # Calculate relative score
        compared_to_fp16 = (
            (baseline_perplexity / perplexity) * 100 if perplexity > 0 else 0
        )
        
        return AccuracyBenchmark(
            strategy=strategy,
            perplexity=perplexity,
            overall_score=max(0, 100 - (perplexity - baseline_perplexity) * 10),
            compared_to_fp16=compared_to_fp16,
        )
    
    def compare_strategies(
        self,
        model_name: str,
        strategies: List[QuantStrategy]
    ) -> List[AccuracyBenchmark]:
        """
        Compare multiple strategies for a model.
        
        Args:
            model_name: Name of the model
            strategies: List of strategies to compare
        
        Returns:
            List of AccuracyBenchmark sorted by score
        """
        results = []
        
        for strategy in strategies:
            benchmark = self.get_benchmark(model_name, strategy)
            if benchmark:
                results.append(benchmark)
        
        # Sort by overall score
        results.sort(key=lambda b: b.overall_score, reverse=True)
        
        return results
    
    def estimate_accuracy_loss(
        self,
        strategy: QuantStrategy,
        bits_per_param: float
    ) -> float:
        """
        Estimate accuracy loss based on quantization.
        
        Args:
            strategy: Quantization strategy
            bits_per_param: Bits per parameter
        
        Returns:
            Estimated accuracy loss (0-1, higher is worse)
        """
        # Baseline
        base_loss = 0.0
        
        if bits_per_param >= 16:
            return 0.0
        
        # INT8 loss
        if bits_per_param >= 8:
            base_loss = 0.02
        
        # INT4 loss
        elif bits_per_param >= 4:
            base_loss = 0.05
        
        # Lower bits
        else:
            base_loss = 0.15
        
        # Adjust based on strategy
        if strategy == QuantStrategy.GPTQ:
            base_loss *= 0.7  # GPTQ is better
        elif strategy == QuantStrategy.AWQ:
            base_loss *= 0.6  # AWQ is best
        elif strategy == QuantStrategy.GGML_Q4:
            base_loss *= 0.9
        
        return min(base_loss, 1.0)


class PerformanceEstimator:
    """
    Estimates performance characteristics for quantization strategies.
    
    Estimates:
    - Memory requirements
    - Inference speed
    - Throughput
    - Batch size capabilities
    """
    
    def __init__(
        self,
        gpu_capability: GPUCapability,
        model_size_params: int,
        sequence_length: int = 2048
    ) -> None:
        """
        Initialize performance estimator.
        
        Args:
            gpu_capability: GPU capability info
            model_size_params: Model size in parameters
            sequence_length: Maximum sequence length
        """
        self.gpu = gpu_capability
        self.model_size_params = model_size_params
        self.sequence_length = sequence_length
    
    def estimate_memory(
        self,
        strategy: QuantStrategy,
        include_kv_cache: bool = True
    ) -> float:
        """
        Estimate memory usage for a strategy.
        
        Args:
            strategy: Quantization strategy
            include_kv_cache: Include KV cache memory
        
        Returns:
            Memory in GB
        """
        # Base memory from bits per parameter
        bits_per_param = self.gpu.get_memory_bits_per_parameter(strategy)
        bytes_per_param = bits_per_param / 8
        
        model_memory = self.model_size_params * bytes_per_param
        
        # KV cache (rough estimate)
        kv_cache = 0
        if include_kv_cache:
            # Assume 32 layers, 4096 hidden size
            kv_cache = 32 * 4096 * self.sequence_length * 2 * 2  # K and V, FP16
            kv_cache /= (1024 ** 3)  # Convert to GB
        
        return model_memory / (1024 ** 3) + kv_cache
    
    def estimate_throughput(
        self,
        strategy: QuantStrategy,
        batch_size: int = 1
    ) -> float:
        """
        Estimate throughput (tokens/second).
        
        Args:
            strategy: Quantization strategy
            batch_size: Batch size
        
        Returns:
            Estimated tokens per second
        """
        # Get FP16 baseline throughput
        base_tflops = self.gpu.fp16_tflops
        
        # Adjust for quantization
        if strategy == QuantStrategy.FP16:
            speed_factor = 1.0
        elif strategy == QuantStrategy.BF16:
            speed_factor = 0.95
        elif strategy == QuantStrategy.FP8:
            speed_factor = 1.5 if self.gpu.supports_fp8 else 1.0
        elif strategy == QuantStrategy.INT8:
            speed_factor = 2.0
        elif strategy in (QuantStrategy.INT4, QuantStrategy.GPTQ_INT4,
                           QuantStrategy.AWQ_INT4, QuantStrategy.GGML_Q4):
            speed_factor = 3.0
        else:
            speed_factor = 1.5
        
        # Model size factor (larger models are relatively faster)
        size_factor = min(self.model_size_params / 7_000_000_000, 1.0)
        
        # Calculate base throughput
        base_tokens_per_sec = 50  # Baseline for small models
        throughput = base_tokens_per_sec * speed_factor * (1 + size_factor)
        
        return throughput * batch_size
    
    def estimate_speedup(
        self,
        strategy: QuantStrategy,
        baseline: QuantStrategy = QuantStrategy.FP16
    ) -> float:
        """
        Estimate speedup compared to baseline.
        
        Args:
            strategy: Target quantization strategy
            baseline: Baseline strategy
        
        Returns:
            Speedup factor (e.g., 2.0 means 2x faster)
        """
        # Memory-based speedup (larger batches possible)
        strategy_bits = self.gpu.get_memory_bits_per_parameter(strategy)
        baseline_bits = self.gpu.get_memory_bits_per_parameter(baseline)
        
        memory_speedup = baseline_bits / max(strategy_bits, 1)
        
        # Compute-based speedup
        if strategy == QuantStrategy.INT8:
            compute_speedup = 2.0
        elif strategy in (QuantStrategy.INT4, QuantStrategy.GPTQ_INT4,
                          QuantStrategy.AWQ_INT4):
            compute_speedup = 3.0
        elif strategy == QuantStrategy.FP8:
            compute_speedup = 1.5
        else:
            compute_speedup = 1.0
        
        return (memory_speedup + compute_speedup) / 2


class QuantSelector:
    """
    Main quantization strategy selector.
    
    Analyzes GPU capabilities and model requirements
    to recommend optimal quantization strategies.
    """
    
    def __init__(
        self,
        gpu_capability: Optional[GPUCapability] = None,
        model_name: str = "unknown",
        model_size_params: int = 0
    ) -> None:
        """
        Initialize quant selector.
        
        Args:
            gpu_capability: GPU capability (auto-detect if None)
            model_name: Name of the model
            model_size_params: Model size in parameters
        """
        self.gpu = gpu_capability or GPUDetector().detect()
        self.model_name = model_name
        self.model_size_params = model_size_params
        
        self.accuracy_comparator = AccuracyComparator()
        self.performance_estimator = PerformanceEstimator(
            self.gpu, model_size_params
        )
        
        self._custom_strategies: List[QuantStrategy] = []
    
    def add_custom_strategy(self, strategy: QuantStrategy) -> None:
        """Add a custom quantization strategy."""
        if strategy not in self._custom_strategies:
            self._custom_strategies.append(strategy)
    
    def select(
        self,
        model_name: str,
        available_memory_gb: float,
        prioritize_memory: bool = True,
        prioritize_accuracy: bool = False,
        require_speedup: float = 1.0
    ) -> QuantizationResult:
        """
        Select optimal quantization strategy.
        
        Args:
            model_name: Name of the model
            available_memory_gb: Available GPU memory in GB
            prioritize_memory: Prioritize memory savings
            prioritize_accuracy: Prioritize accuracy (less quantization)
            require_speedup: Minimum required speedup factor
        
        Returns:
            QuantizationResult with recommended strategy
        """
        self.model_name = model_name
        
        # Get available strategies
        strategies = self._get_available_strategies()
        
        # Filter by requirements
        filtered = self._filter_strategies(
            strategies,
            available_memory_gb,
            require_speedup
        )
        
        if not filtered:
            # Return FP16 as fallback
            return QuantizationResult(
                strategy=QuantStrategy.FP16,
                recommended=False,
                memory_reduction=0.0,
                estimated_accuracy_loss=0.0,
                estimated_speedup=1.0,
                compatibility_score=1.0,
                memory_gb=self.performance_estimator.estimate_memory(QuantStrategy.FP16),
                warnings=["No suitable quantization found, using FP16"],
            )
        
        # Score and rank strategies
        scored = []
        for strategy in filtered:
            score = self._score_strategy(
                strategy,
                available_memory_gb,
                prioritize_memory,
                prioritize_accuracy
            )
            scored.append((strategy, score))
        
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        best_strategy, best_score = scored[0]
        
        # Get detailed results
        result = self._create_result(
            best_strategy,
            best_score,
            available_memory_gb
        )
        
        return result
    
    def _get_available_strategies(self) -> List[QuantStrategy]:
        """Get all available strategies for this GPU."""
        all_strategies = [
            QuantStrategy.FP32,
            QuantStrategy.FP16,
            QuantStrategy.BF16,
            QuantStrategy.FP8,
            QuantStrategy.INT8,
            QuantStrategy.INT4,
            QuantStrategy.GPTQ,
            QuantStrategy.GPTQ_INT4,
            QuantStrategy.AWQ,
            QuantStrategy.AWQ_INT4,
            QuantStrategy.GGML_Q4,
            QuantStrategy.GGML_Q5,
            QuantStrategy.GGML_Q8,
        ]
        
        return [s for s in all_strategies if self.gpu.supports_quantization(s)]
    
    def _filter_strategies(
        self,
        strategies: List[QuantStrategy],
        available_memory_gb: float,
        require_speedup: float
    ) -> List[QuantStrategy]:
        """Filter strategies by memory and performance requirements."""
        filtered = []
        
        for strategy in strategies:
            estimated_memory = self.performance_estimator.estimate_memory(strategy)
            
            # Check memory
            if estimated_memory > available_memory_gb * 0.95:
                continue
            
            # Check speedup
            speedup = self.performance_estimator.estimate_speedup(strategy)
            if speedup < require_speedup:
                continue
            
            filtered.append(strategy)
        
        return filtered
    
    def _score_strategy(
        self,
        strategy: QuantStrategy,
        available_memory_gb: float,
        prioritize_memory: bool,
        prioritize_accuracy: bool
    ) -> float:
        """Score a quantization strategy."""
        # Memory score (how much we save)
        fp16_memory = self.performance_estimator.estimate_memory(QuantStrategy.FP16)
        strategy_memory = self.performance_estimator.estimate_memory(strategy)
        memory_saved = (fp16_memory - strategy_memory) / fp16_memory
        memory_score = min(memory_saved * 100, 100)
        
        # Accuracy score (lower is better for loss)
        accuracy_loss = self.accuracy_comparator.estimate_accuracy_loss(
            strategy,
            self.gpu.get_memory_bits_per_parameter(strategy)
        )
        accuracy_score = (1 - accuracy_loss) * 100
        
        # Speed score
        speedup = self.performance_estimator.estimate_speedup(strategy)
        speed_score = min(speedup * 30, 100)
        
        # Weight scores based on priorities
        if prioritize_accuracy:
            return accuracy_score * 0.6 + speed_score * 0.3 + memory_score * 0.1
        elif prioritize_memory:
            return memory_score * 0.6 + speed_score * 0.25 + accuracy_score * 0.15
        else:
            # Balanced
            return memory_score * 0.3 + speed_score * 0.35 + accuracy_score * 0.35
    
    def _create_result(
        self,
        strategy: QuantStrategy,
        score: float,
        available_memory_gb: float
    ) -> QuantizationResult:
        """Create detailed result for a strategy."""
        memory = self.performance_estimator.estimate_memory(strategy)
        speedup = self.performance_estimator.estimate_speedup(strategy)
        fp16_memory = self.performance_estimator.estimate_memory(QuantStrategy.FP16)
        accuracy_loss = self.accuracy_comparator.estimate_accuracy_loss(
            strategy,
            self.gpu.get_memory_bits_per_parameter(strategy)
        )
        
        warnings = []
        if accuracy_loss > 0.1:
            warnings.append("High accuracy loss expected, test carefully")
        if memory > available_memory_gb * 0.8:
            warnings.append("Memory usage is high, limited room for KV cache")
        if not self.gpu.supports_fp8 and strategy == QuantStrategy.FP8:
            warnings.append("FP8 not natively supported, may be emulated")
        
        return QuantizationResult(
            strategy=strategy,
            recommended=True,
            memory_reduction=(fp16_memory - memory) / fp16_memory,
            estimated_accuracy_loss=accuracy_loss,
            estimated_speedup=speedup,
            compatibility_score=score / 100,
            memory_gb=memory,
            warnings=warnings,
        )
    
    def get_recommendations(
        self,
        model_name: str,
        available_memory_gb: float
    ) -> List[QuantizationResult]:
        """
        Get multiple recommendations ranked.
        
        Args:
            model_name: Name of the model
            available_memory_gb: Available GPU memory
        
        Returns:
            List of QuantizationResult sorted by recommendation
        """
        strategies = self._get_available_strategies()
        filtered = self._filter_strategies(strategies, available_memory_gb, 1.0)
        
        results = []
        for strategy in filtered:
            score = self._score_strategy(strategy, available_memory_gb, False, False)
            result = self._create_result(strategy, score, available_memory_gb)
            results.append(result)
        
        results.sort(key=lambda r: r.compatibility_score, reverse=True)
        
        return results
    
    def compare_all(
        self,
        model_name: str,
        available_memory_gb: float
    ) -> Dict[QuantStrategy, QuantizationResult]:
        """
        Compare all possible strategies.
        
        Args:
            model_name: Name of the model
            available_memory_gb: Available GPU memory
        
        Returns:
            Dictionary mapping strategies to results
        """
        strategies = self._get_available_strategies()
        results = {}
        
        for strategy in strategies:
            memory = self.performance_estimator.estimate_memory(strategy)
            
            # Skip if doesn't fit
            if memory > available_memory_gb:
                continue
            
            score = self._score_strategy(strategy, available_memory_gb, False, False)
            result = self._create_result(strategy, score, available_memory_gb)
            results[strategy] = result
        
        return results


def detect_gpu() -> GPUCapability:
    """
    Detect GPU and return capability.
    
    Returns:
        GPUCapability for detected GPU
    """
    return GPUDetector().detect()


def select_quantization(
    model_name: str,
    model_params: int,
    gpu_memory_gb: float,
    prioritize: str = "balanced"
) -> QuantizationResult:
    """
    Quick function to select quantization strategy.
    
    Args:
        model_name: Name of the model
        model_params: Number of parameters
        gpu_memory_gb: Available GPU memory
        prioritize: Optimization priority ("memory", "accuracy", "balanced")
    
    Returns:
        QuantizationResult with recommended strategy
    """
    gpu = detect_gpu()
    selector = QuantSelector(gpu, model_name, model_params)
    
    prioritize_memory = prioritize == "memory"
    prioritize_accuracy = prioritize == "accuracy"
    
    return selector.select(
        model_name,
        gpu_memory_gb,
        prioritize_memory,
        prioritize_accuracy
    )


def get_quantization_profile(
    model_name: str,
    model_params: int,
    strategy: QuantStrategy,
    gpu_memory_gb: float
) -> QuantizationProfile:
    """
    Get complete quantization profile.
    
    Args:
        model_name: Name of the model
        model_params: Number of parameters
        strategy: Selected quantization strategy
        gpu_memory_gb: Available GPU memory
    
    Returns:
        QuantizationProfile with complete information
    """
    gpu = detect_gpu()
    estimator = PerformanceEstimator(gpu, model_params)
    
    memory_before = estimator.estimate_memory(QuantStrategy.FP16)
    memory_after = estimator.estimate_memory(strategy)
    speedup = estimator.estimate_speedup(strategy)
    throughput = estimator.estimate_throughput(strategy)
    
    return QuantizationProfile(
        model_name=model_name,
        base_precision=QuantStrategy.FP16,
        target_strategy=strategy,
        memory_before_gb=memory_before,
        memory_after_gb=memory_after,
        compression_ratio=memory_before / max(memory_after, 0.001),
        estimated_inference_speedup=speedup,
        estimated_throughput=throughput,
        batch_size_multiplier=speedup,
    )
