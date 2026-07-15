"""
GPU memory planning and allocation for LLM inference.

This module provides:
- Model memory requirement estimation
- KV cache allocation strategies
- Dynamic batch size planning
- Out-of-memory prevention
- Memory monitoring and cleanup

Author: AGI Unified Framework Team
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import time


class MemoryUnit(Enum):
    """Memory units for measurements."""
    BYTES = 1
    KB = 1024
    MB = 1024 ** 2
    GB = 1024 ** 3
    TB = 1024 ** 4


class AllocationStrategy(Enum):
    """KV cache allocation strategies."""
    STATIC = "static"
    DYNAMIC = "dynamic"
    ELASTIC = "elastic"
    HYBRID = "hybrid"


@dataclass
class GPUInfo:
    """
    Information about a GPU device.
    
    Attributes:
        device_id: GPU device ID
        name: GPU model name
        total_memory: Total memory in bytes
        available_memory: Available memory in bytes
        compute_capability: Compute capability (e.g., 8.0)
        num_multiprocessors: Number of multiprocessors
        memory_bandwidth: Memory bandwidth in GB/s
        max_batch_size: Maximum supported batch size
    """
    device_id: int
    name: str
    total_memory: int
    available_memory: int
    compute_capability: float = 0.0
    num_multiprocessors: int = 0
    memory_bandwidth: float = 0.0
    max_batch_size: int = 1
    utilization: float = 0.0
    temperature: float = 0.0
    
    def get_used_memory(self) -> int:
        """Get used memory in bytes."""
        return self.total_memory - self.available_memory
    
    def get_utilization_percent(self) -> float:
        """Get memory utilization percentage."""
        if self.total_memory == 0:
            return 0.0
        return (self.get_used_memory() / self.total_memory) * 100
    
    def has_sufficient_memory(self, required: int, reserve: int = 1024**3) -> bool:
        """Check if GPU has sufficient memory."""
        return self.available_memory >= (required + reserve)
    
    def format_memory(self, memory_bytes: int, unit: MemoryUnit = MemoryUnit.GB) -> str:
        """Format memory in human-readable form."""
        value = memory_bytes / unit.value
        return f"{value:.2f} {unit.name}"


@dataclass
class ModelMemoryRequirements:
    """
    Memory requirements for a model.
    
    Attributes:
        model_name: Name of the model
        model_weights: Memory for model weights
        activations: Memory for activations
        kv_cache: Memory for KV cache
        gradients: Memory for gradients (training)
        overhead: Memory for framework overhead
        total: Total memory requirement
        per_layer: Memory per transformer layer
        activation_per_token: Activation memory per token
    """
    model_name: str
    model_weights: int = 0
    activations: int = 0
    kv_cache: int = 0
    gradients: int = 0
    overhead: int = 0
    total: int = 0
    per_layer: int = 0
    activation_per_token: int = 0
    num_parameters: int = 0
    num_layers: int = 0
    hidden_size: int = 0
    vocab_size: int = 0
    
    def format_all(self, unit: MemoryUnit = MemoryUnit.GB) -> Dict[str, str]:
        """Format all memory values in human-readable form."""
        return {
            "model_name": self.model_name,
            "model_weights": self.format_value(self.model_weights, unit),
            "activations": self.format_value(self.activations, unit),
            "kv_cache": self.format_value(self.kv_cache, unit),
            "gradients": self.format_value(self.gradients, unit),
            "overhead": self.format_value(self.overhead, unit),
            "total": self.format_value(self.total, unit),
        }
    
    @staticmethod
    def format_value(value: int, unit: MemoryUnit = MemoryUnit.GB) -> str:
        """Format a single memory value."""
        return f"{value / unit.value:.2f} {unit.name}"


@dataclass
class KVCacheConfig:
    """
    Configuration for KV cache allocation.
    
    Attributes:
        max_sequence_length: Maximum sequence length
        num_layers: Number of KV cache layers
        num_heads: Number of attention heads
        head_dim: Dimension per head
        dtype_size: Size of data type in bytes
        allocation_strategy: How to allocate cache
        preallocate_fraction: Fraction of memory to preallocate
        enable_caching: Enable KV cache reuse
    """
    max_sequence_length: int = 2048
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    dtype_size: int = 2  # float16
    allocation_strategy: AllocationStrategy = AllocationStrategy.DYNAMIC
    preallocate_fraction: float = 0.7
    enable_caching: bool = True
    
    def calculate_layer_size(self) -> int:
        """Calculate memory for one layer's KV cache."""
        # Key + Value tensors: 2 * seq_len * num_heads * head_dim * dtype_size
        return 2 * self.max_sequence_length * self.num_heads * self.head_dim * self.dtype_size
    
    def calculate_total_size(self) -> int:
        """Calculate total KV cache memory."""
        return self.num_layers * self.calculate_layer_size()


@dataclass
class BatchSizePlan:
    """
    Planned batch size configuration.
    
    Attributes:
        recommended_batch_size: Recommended batch size
        max_batch_size: Maximum possible batch size
        min_batch_size: Minimum batch size
        memory_per_sample: Memory per sample
        estimated_throughput: Estimated samples per second
        latency_estimate_ms: Estimated latency in milliseconds
        safety_margin: Safety margin for memory
    """
    recommended_batch_size: int
    max_batch_size: int
    min_batch_size: int
    memory_per_sample: int
    estimated_throughput: float
    latency_estimate_ms: float
    safety_margin: float = 0.1
    can_fit_in_memory: bool = True
    reason: str = ""


@dataclass
class MemoryPlan:
    """
    Complete memory plan for model deployment.
    
    Attributes:
        model_requirements: Model memory requirements
        kv_cache_config: KV cache configuration
        batch_plan: Batch size plan
        gpu_info: GPU information
        total_required: Total memory required
        available_after_allocation: Available memory after allocation
        fits_in_memory: Whether plan fits in GPU memory
        warnings: List of warnings
        recommendations: List of recommendations
    """
    model_requirements: ModelMemoryRequirements
    kv_cache_config: KVCacheConfig
    batch_plan: BatchSizePlan
    gpu_info: GPUInfo
    total_required: int
    available_after_allocation: int
    fits_in_memory: bool
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class MemoryEstimator:
    """
    Estimates memory requirements for models.
    
    Supports estimation for:
    - Transformer models (LLM, BERT, etc.)
    - Different precision modes (FP32, FP16, BF16, INT8)
    - Various model architectures
    """
    
    # Memory constants per parameter (in bytes)
    PRECISION_FACTORS: Dict[str, float] = {
        "fp32": 4.0,
        "fp16": 2.0,
        "bf16": 2.0,
        "int8": 1.0,
        "int4": 0.5,
    }
    
    # Known model configurations
    KNOWN_MODELS: Dict[str, Dict[str, Any]] = {
        "gpt2": {
            "num_layers": 12,
            "hidden_size": 768,
            "num_heads": 12,
            "vocab_size": 50257,
            "num_parameters": 124_000_000,
        },
        "gpt2-medium": {
            "num_layers": 24,
            "hidden_size": 1024,
            "num_heads": 16,
            "vocab_size": 50257,
            "num_parameters": 355_000_000,
        },
        "gpt2-large": {
            "num_layers": 36,
            "hidden_size": 1280,
            "num_heads": 20,
            "vocab_size": 50257,
            "num_parameters": 774_000_000,
        },
        "llama-7b": {
            "num_layers": 32,
            "hidden_size": 4096,
            "num_heads": 32,
            "vocab_size": 32000,
            "num_parameters": 6_700_000_000,
        },
        "llama-13b": {
            "num_layers": 40,
            "hidden_size": 5120,
            "num_heads": 40,
            "vocab_size": 32000,
            "num_parameters": 13_000_000_000,
        },
        "llama-70b": {
            "num_layers": 80,
            "hidden_size": 8192,
            "num_heads": 64,
            "vocab_size": 32000,
            "num_parameters": 68_000_000_000,
        },
    }
    
    def __init__(self) -> None:
        """Initialize memory estimator."""
        self._custom_models: Dict[str, Dict[str, Any]] = {}
    
    def register_model(
        self,
        name: str,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        vocab_size: int,
        num_parameters: Optional[int] = None
    ) -> None:
        """
        Register a custom model configuration.
        
        Args:
            name: Model name
            num_layers: Number of transformer layers
            hidden_size: Hidden size dimension
            num_heads: Number of attention heads
            vocab_size: Vocabulary size
            num_parameters: Total number of parameters
        """
        self._custom_models[name] = {
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "vocab_size": vocab_size,
            "num_parameters": num_parameters or self._estimate_params(
                num_layers, hidden_size, vocab_size
            ),
        }
    
    def estimate(
        self,
        model_name: str,
        num_layers: Optional[int] = None,
        hidden_size: Optional[int] = None,
        vocab_size: Optional[int] = None,
        num_heads: Optional[int] = None,
        num_parameters: Optional[int] = None,
        precision: str = "fp16",
        sequence_length: int = 2048,
        batch_size: int = 1,
        include_kv_cache: bool = True,
        include_gradients: bool = False,
    ) -> ModelMemoryRequirements:
        """
        Estimate memory requirements for a model.
        
        Args:
            model_name: Name of the model
            num_layers: Number of layers (if known)
            hidden_size: Hidden size (if known)
            vocab_size: Vocabulary size (if known)
            num_heads: Number of attention heads (if known)
            num_parameters: Total parameters (if known)
            precision: Model precision (fp32, fp16, bf16, int8, int4)
            sequence_length: Maximum sequence length
            batch_size: Batch size for activations
            include_kv_cache: Include KV cache memory
            include_gradients: Include gradient memory
        
        Returns:
            ModelMemoryRequirements with estimates
        """
        # Get model config
        config = self._get_model_config(
            model_name, num_layers, hidden_size, vocab_size, num_heads, num_parameters
        )
        
        # Calculate memory
        precision_factor = self.PRECISION_FACTORS.get(precision, 2.0)
        
        # Model weights
        num_params = config["num_parameters"]
        model_weights = int(num_params * precision_factor)
        
        # Embedding table
        embedding_size = config["vocab_size"] * config["hidden_size"] * precision_factor
        model_weights += int(embedding_size)
        
        # Layer overhead
        per_layer_size = self._estimate_layer_size(config, precision_factor)
        model_weights += per_layer_size * config["num_layers"]
        
        # Activations (rough estimate)
        activations = self._estimate_activations(
            config, sequence_length, batch_size, precision_factor
        )
        
        # KV cache
        kv_cache = 0
        if include_kv_cache:
            kv_cache = self._estimate_kv_cache(
                config, sequence_length, precision_factor, batch_size
            )
        
        # Gradients
        gradients = 0
        if include_gradients:
            gradients = int(num_params * precision_factor)
        
        # Framework overhead
        overhead = int(model_weights * 0.1)  # ~10% overhead
        
        total = model_weights + activations + kv_cache + gradients + overhead
        
        return ModelMemoryRequirements(
            model_name=model_name,
            model_weights=model_weights,
            activations=activations,
            kv_cache=kv_cache,
            gradients=gradients,
            overhead=overhead,
            total=total,
            per_layer=per_layer_size,
            activation_per_token=activations // max(batch_size * sequence_length, 1),
            num_parameters=num_params,
            num_layers=config["num_layers"],
            hidden_size=config["hidden_size"],
            vocab_size=config["vocab_size"],
        )
    
    def _get_model_config(
        self,
        model_name: str,
        num_layers: Optional[int],
        hidden_size: Optional[int],
        vocab_size: Optional[int],
        num_heads: Optional[int],
        num_parameters: Optional[int],
    ) -> Dict[str, Any]:
        """Get model configuration."""
        # Check known models
        name_lower = model_name.lower()
        for known_name, config in self.KNOWN_MODELS.items():
            if known_name in name_lower:
                return config.copy()
        
        # Check custom models
        for custom_name, config in self._custom_models.items():
            if custom_name.lower() in name_lower:
                return config.copy()
        
        # Use provided values or defaults
        num_layers = num_layers or 12
        hidden_size = hidden_size or 768
        vocab_size = vocab_size or 50000
        num_heads = num_heads or (hidden_size // 64)
        num_parameters = num_parameters or self._estimate_params(
            num_layers, hidden_size, vocab_size
        )
        
        return {
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "vocab_size": vocab_size,
            "num_parameters": num_parameters,
        }
    
    def _estimate_params(
        self,
        num_layers: int,
        hidden_size: int,
        vocab_size: int
    ) -> int:
        """Estimate total parameters."""
        # Rough estimation for decoder-only transformer
        # This is approximate - actual models vary
        params_per_layer = (
            4 * hidden_size ** 2 +  # Attention weights
            2 * hidden_size * vocab_size  # Embedding
        )
        return num_layers * params_per_layer
    
    def _estimate_layer_size(
        self,
        config: Dict[str, Any],
        precision_factor: float
    ) -> int:
        """Estimate memory per transformer layer."""
        hidden_size = config["hidden_size"]
        
        # Self-attention: 4 weight matrices
        attn_weights = 4 * hidden_size ** 2 * precision_factor
        
        # Layer norms: 2 * 2 * hidden_size
        layer_norms = 4 * hidden_size * precision_factor
        
        # Feed-forward: 2 matrices (upsample + downsample)
        ff_weights = 6 * hidden_size ** 2 * precision_factor
        
        return int(attn_weights + layer_norms + ff_weights)
    
    def _estimate_activations(
        self,
        config: Dict[str, Any],
        sequence_length: int,
        batch_size: int,
        precision_factor: float
    ) -> int:
        """Estimate activation memory."""
        hidden_size = config["hidden_size"]
        num_layers = config["num_layers"]
        
        # Activation memory per token per layer
        # This includes attention patterns, intermediate activations, etc.
        activation_per_layer = sequence_length * batch_size * hidden_size * precision_factor * 16
        
        return int(activation_per_layer * num_layers)
    
    def _estimate_kv_cache(
        self,
        config: Dict[str, Any],
        sequence_length: int,
        precision_factor: float,
        batch_size: int
    ) -> int:
        """Estimate KV cache memory."""
        hidden_size = config["hidden_size"]
        num_layers = config["num_layers"]
        
        # KV cache: 2 * num_layers * batch * seq_len * hidden
        # Key and Value for each layer
        return int(2 * num_layers * batch_size * sequence_length * hidden_size * precision_factor)


class KVCacheAllocator:
    """
    Allocates and manages KV cache memory.
    
    Supports different allocation strategies:
    - Static: Fixed allocation per sequence
    - Dynamic: Allocate as needed
    - Elastic: Grow/shrink allocation dynamically
    - Hybrid: Combine static and dynamic
    """
    
    def __init__(
        self,
        gpu_info: GPUInfo,
        config: KVCacheConfig,
        total_memory_budget: int
    ) -> None:
        """
        Initialize KV cache allocator.
        
        Args:
            gpu_info: GPU information
            config: KV cache configuration
            total_memory_budget: Total memory budget for KV cache
        """
        self.gpu_info = gpu_info
        self.config = config
        self.total_budget = total_memory_budget
        
        self._allocated_blocks: Dict[str, int] = {}
        self._available_memory = total_memory_budget
        self._num_allocated_sequences = 0
    
    def allocate(self, request_id: str, num_tokens: int) -> bool:
        """
        Allocate KV cache for a sequence.
        
        Args:
            request_id: Unique identifier for the request
            num_tokens: Number of tokens to allocate
        
        Returns:
            True if allocation successful, False otherwise
        """
        required = self._calculate_required_memory(num_tokens)
        
        if required > self._available_memory:
            return False
        
        self._allocated_blocks[request_id] = num_tokens
        self._available_memory -= required
        self._num_allocated_sequences += 1
        
        return True
    
    def deallocate(self, request_id: str) -> int:
        """
        Deallocate KV cache for a sequence.
        
        Args:
            request_id: Unique identifier for the request
        
        Returns:
            Number of bytes freed
        """
        if request_id not in self._allocated_blocks:
            return 0
        
        num_tokens = self._allocated_blocks.pop(request_id)
        freed = self._calculate_required_memory(num_tokens)
        
        self._available_memory += freed
        self._num_allocated_sequences -= 1
        
        return freed
    
    def reallocate(self, request_id: str, new_num_tokens: int) -> bool:
        """
        Reallocate KV cache for a sequence.
        
        Args:
            request_id: Unique identifier for the request
            new_num_tokens: New number of tokens
        
        Returns:
            True if reallocation successful, False otherwise
        """
        if request_id not in self._allocated_blocks:
            return self.allocate(request_id, new_num_tokens)
        
        # Calculate memory delta
        old_tokens = self._allocated_blocks[request_id]
        old_memory = self._calculate_required_memory(old_tokens)
        new_memory = self._calculate_required_memory(new_num_tokens)
        
        delta = new_memory - old_memory
        
        if delta > self._available_memory:
            return False
        
        self._allocated_blocks[request_id] = new_num_tokens
        self._available_memory -= delta
        
        return True
    
    def _calculate_required_memory(self, num_tokens: int) -> int:
        """Calculate memory required for num_tokens."""
        return (
            2 *  # Key + Value
            self.config.num_layers *
            num_tokens *
            self.config.num_heads *
            self.config.head_dim *
            self.config.dtype_size
        )
    
    def get_allocation_info(self) -> Dict[str, Any]:
        """Get current allocation information."""
        return {
            "total_budget": self.total_budget,
            "used": self.total_budget - self._available_memory,
            "available": self._available_memory,
            "num_allocated_sequences": self._num_allocated_sequences,
            "utilization": (
                (self.total_budget - self._available_memory) / self.total_budget
                if self.total_budget > 0 else 0
            ),
        }
    
    def can_allocate(self, num_tokens: int) -> bool:
        """Check if allocation is possible."""
        return self._calculate_required_memory(num_tokens) <= self._available_memory


class BatchSizePlanner:
    """
    Plans optimal batch sizes for memory efficiency.
    
    Analyzes available memory and model requirements
    to determine optimal batch sizes.
    """
    
    def __init__(
        self,
        gpu_info: GPUInfo,
        memory_estimator: MemoryEstimator,
        model_name: str,
        sequence_length: int = 2048,
        precision: str = "fp16"
    ) -> None:
        """
        Initialize batch size planner.
        
        Args:
            gpu_info: GPU information
            memory_estimator: Memory estimator instance
            model_name: Model name
            sequence_length: Maximum sequence length
            precision: Model precision
        """
        self.gpu_info = gpu_info
        self.estimator = memory_estimator
        self.model_name = model_name
        self.sequence_length = sequence_length
        self.precision = precision
        
        self._cached_plan: Optional[BatchSizePlan] = None
    
    def plan(self, safety_margin: float = 0.1) -> BatchSizePlan:
        """
        Create a batch size plan.
        
        Args:
            safety_margin: Safety margin (0-1)
        
        Returns:
            BatchSizePlan with recommended sizes
        """
        # Get model requirements
        model_req = self.estimator.estimate(
            self.model_name,
            sequence_length=self.sequence_length,
            batch_size=1,
            precision=self.precision,
        )
        
        # Calculate available memory for inference
        available = int(
            self.gpu_info.available_memory * (1 - safety_margin)
        )
        
        # Reserve memory for KV cache
        kv_cache_budget = int(available * 0.5)
        
        # Memory for model weights and activations
        inference_memory = available - kv_cache_budget
        
        # Estimate memory per sample
        memory_per_sample = model_req.activation_per_token * self.sequence_length
        memory_per_sample += model_req.kv_cache // max(self.sequence_length, 1)
        
        # Calculate max batch size
        max_batch = max(1, inference_memory // max(memory_per_sample, 1))
        
        # Apply GPU limits
        max_batch = min(max_batch, self.gpu_info.max_batch_size)
        
        # Recommended batch size (leave headroom)
        recommended = max(1, int(max_batch * 0.8))
        
        # Latency estimation (rough)
        latency = 100.0 + (max_batch * 50.0)  # ms
        
        # Throughput estimation
        throughput = max_batch / (latency / 1000.0) if latency > 0 else 0
        
        can_fit = max_batch > 0
        
        plan = BatchSizePlan(
            recommended_batch_size=recommended,
            max_batch_size=max_batch,
            min_batch_size=1,
            memory_per_sample=memory_per_sample,
            estimated_throughput=throughput,
            latency_estimate_ms=latency,
            safety_margin=safety_margin,
            can_fit_in_memory=can_fit,
            reason="OK" if can_fit else "Model too large for available memory",
        )
        
        self._cached_plan = plan
        return plan
    
    def get_recommended_batch_size(self, safety_margin: float = 0.1) -> int:
        """Get recommended batch size."""
        plan = self._cached_plan
        if plan is None:
            plan = self.plan(safety_margin)
        return plan.recommended_batch_size


class OOMPrevention:
    """
    Prevents out-of-memory errors with proactive monitoring.
    
    Features:
    - Memory threshold monitoring
    - Automatic batch size reduction
    - Request queuing
    - Memory cleanup triggers
    """
    
    def __init__(
        self,
        gpu_info: GPUInfo,
        warning_threshold: float = 0.85,
        critical_threshold: float = 0.95,
        emergency_threshold: float = 0.98
    ) -> None:
        """
        Initialize OOM prevention.
        
        Args:
            gpu_info: GPU information
            warning_threshold: Warning threshold (0-1)
            critical_threshold: Critical threshold (0-1)
            emergency_threshold: Emergency threshold (0-1)
        """
        self.gpu_info = gpu_info
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.emergency_threshold = emergency_threshold
        
        self._request_queue: List[str] = []
        self._blocked_requests: Dict[str, int] = {}
        self._last_cleanup = time.time()
        self._cleanup_interval = 60.0  # seconds
    
    def check_memory(
        self,
        required_memory: int,
        reserve_memory: int = 512 * 1024 * 1024  # 512 MB reserve
    ) -> Tuple[bool, str]:
        """
        Check if memory is available for allocation.
        
        Args:
            required_memory: Required memory in bytes
            reserve_memory: Reserve memory to keep free
        
        Returns:
            Tuple of (can_allocate, status_message)
        """
        current_util = self.gpu_info.get_utilization_percent() / 100.0
        available = self.gpu_info.available_memory
        
        # Check thresholds
        if current_util >= self.emergency_threshold:
            return False, "EMERGENCY: Memory critically low"
        
        if current_util >= self.critical_threshold:
            return False, "CRITICAL: Memory threshold exceeded"
        
        if current_util >= self.warning_threshold:
            return True, "WARNING: Memory usage high"
        
        # Check absolute memory
        if available < (required_memory + reserve_memory):
            return False, f"INSUFFICIENT: Need {required_memory}, have {available}"
        
        return True, "OK"
    
    def queue_request(self, request_id: str) -> bool:
        """
        Queue a request for later processing.
        
        Args:
            request_id: Unique request identifier
        
        Returns:
            True if queued successfully
        """
        if request_id not in self._request_queue:
            self._request_queue.append(request_id)
            return True
        return False
    
    def get_queue_length(self) -> int:
        """Get number of queued requests."""
        return len(self._request_queue)
    
    def get_next_queued_request(self) -> Optional[str]:
        """Get and remove next queued request."""
        if self._request_queue:
            return self._request_queue.pop(0)
        return None
    
    def should_cleanup(self) -> bool:
        """Check if cleanup should be performed."""
        return (time.time() - self._last_cleanup) > self._cleanup_interval
    
    def trigger_cleanup(self) -> int:
        """
        Trigger memory cleanup.
        
        Returns:
            Estimated memory freed in bytes
        """
        # In real implementation, would call garbage collection
        # and clear caches
        self._last_cleanup = time.time()
        freed = self.gpu_info.total_memory // 10  # Rough estimate
        return freed
    
    def get_status(self) -> Dict[str, Any]:
        """Get OOM prevention status."""
        return {
            "utilization": self.gpu_info.get_utilization_percent(),
            "warning_threshold": self.warning_threshold * 100,
            "critical_threshold": self.critical_threshold * 100,
            "emergency_threshold": self.emergency_threshold * 100,
            "queued_requests": len(self._request_queue),
            "last_cleanup": self._last_cleanup,
            "should_cleanup": self.should_cleanup(),
        }


class MemoryPlanner:
    """
    Main memory planning coordinator.
    
    Orchestrates:
    - Memory estimation
    - KV cache allocation
    - Batch size planning
    - OOM prevention
    """
    
    def __init__(
        self,
        gpu_info: GPUInfo,
        model_name: str,
        sequence_length: int = 2048,
        precision: str = "fp16"
    ) -> None:
        """
        Initialize memory planner.
        
        Args:
            gpu_info: GPU information
            model_name: Model name
            sequence_length: Maximum sequence length
            precision: Model precision
        """
        self.gpu_info = gpu_info
        self.model_name = model_name
        
        # Initialize components
        self.estimator = MemoryEstimator()
        self.batch_planner = BatchSizePlanner(
            gpu_info, self.estimator, model_name, sequence_length, precision
        )
        self.oom_prevention = OOMPrevention(gpu_info)
        
        # Create initial plan
        self._current_plan: Optional[MemoryPlan] = None
        self._refresh_plan()
    
    def _refresh_plan(self) -> None:
        """Refresh the current memory plan."""
        batch_plan = self.batch_planner.plan()
        
        # Calculate KV cache budget
        kv_cache_budget = int(
            self.gpu_info.available_memory * 0.4
        )
        
        kv_config = KVCacheConfig(
            max_sequence_length=self.batch_planner.sequence_length,
        )
        
        kv_allocator = KVCacheAllocator(
            self.gpu_info, kv_config, kv_cache_budget
        )
        
        model_req = self.estimator.estimate(
            self.model_name,
            sequence_length=self.batch_planner.sequence_length,
            precision=self.batch_planner.precision,
        )
        
        total_required = model_req.total + kv_cache_budget
        available = self.gpu_info.available_memory - total_required
        
        warnings = []
        recommendations = []
        
        if total_required > self.gpu_info.available_memory:
            warnings.append("Model may not fit in GPU memory")
            recommendations.append("Consider using quantization or smaller model")
        
        if batch_plan.recommended_batch_size == 1:
            warnings.append("Only batch size 1 possible")
            recommendations.append("Model is very large, consider optimization")
        
        self._current_plan = MemoryPlan(
            model_requirements=model_req,
            kv_cache_config=kv_config,
            batch_plan=batch_plan,
            gpu_info=self.gpu_info,
            total_required=total_required,
            available_after_allocation=available,
            fits_in_memory=available >= 0,
            warnings=warnings,
            recommendations=recommendations,
        )
    
    def get_plan(self) -> MemoryPlan:
        """Get current memory plan."""
        return self._current_plan
    
    def get_recommended_batch_size(self) -> int:
        """Get recommended batch size."""
        return self.batch_planner.get_recommended_batch_size()
    
    def check_allocation(self, batch_size: int, sequence_length: int) -> bool:
        """Check if batch allocation is possible."""
        model_req = self.estimator.estimate(
            self.model_name,
            sequence_length=sequence_length,
            batch_size=batch_size,
        )
        
        can_allocate, _ = self.oom_prevention.check_memory(
            model_req.total
        )
        
        return can_allocate


def create_memory_planner(
    model_name: str,
    gpu_memory_gb: float = 16.0,
    sequence_length: int = 2048,
    precision: str = "fp16"
) -> MemoryPlanner:
    """
    Create a memory planner with GPU info.
    
    Args:
        model_name: Name of the model
        gpu_memory_gb: GPU memory in GB
        sequence_length: Maximum sequence length
        precision: Model precision
    
    Returns:
        MemoryPlanner instance
    """
    gpu_info = GPUInfo(
        device_id=0,
        name="Detected GPU",
        total_memory=int(gpu_memory_gb * 1024 ** 3),
        available_memory=int(gpu_memory_gb * 1024 ** 3),
    )
    
    return MemoryPlanner(gpu_info, model_name, sequence_length, precision)


def estimate_model_memory(
    model_name: str,
    num_layers: Optional[int] = None,
    hidden_size: Optional[int] = None,
    num_parameters: Optional[int] = None,
    precision: str = "fp16"
) -> ModelMemoryRequirements:
    """
    Quick function to estimate model memory.
    
    Args:
        model_name: Name of the model
        num_layers: Number of layers
        hidden_size: Hidden size
        num_parameters: Total parameters
        precision: Model precision
    
    Returns:
        ModelMemoryRequirements
    """
    estimator = MemoryEstimator()
    return estimator.estimate(
        model_name,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_parameters=num_parameters,
        precision=precision,
    )
