"""
AGI Unified Framework - LLM Core Module
模型网关与推理优化模块

提供统一的LLM抽象接口、多后端适配器、网关路由、
降级策略、速率限制、缓存管理和推理优化等功能。
"""

from .base import (
    LLMBackend,
    LLMChunk,
    LLMError,
    LLMResponse,
    Message,
    MessageRole,
    ModelInfo,
    ToolCall,
    Usage,
    FinishReason,
    GenerateParams,
)
from .openai_adapter import OpenAIAdapter
# 别名兼容
OpenAIBackend = OpenAIAdapter
from .anthropic_adapter import AnthropicAdapter
from .local_adapter import LocalAdapter
from .gateway import ModelGateway
from .fallback import FallbackStrategy, FallbackConfig, FallbackResult
from .rate_limiter import RateLimiter, RateLimitConfig, RateLimitStats
from .prompt_formatter import PromptFormatter, SystemPromptBuilder
from .cost_calculator import CostCalculator, CostReport
from .cache_manager import KVCacheManager, SemanticCache, CacheStats
from .speculative import SpeculativeDecoding, SpeculativeStats
from .stream_processor import StreamProcessor, StreamBuffer

# vLLM适配器
from .vllm_adapter import (
    VLLMAdapter,
    PagedAttentionManager,
    ContinuousBatcher,
    TensorParallelEngine,
    PipelineParallelEngine,
    SpeculativeDecoder,
)

# Ollama适配器
from .ollama_adapter import (
    OllamaAdapter,
    OllamaModelManager,
    OllamaEmbedder,
    OllamaChatClient,
    OllamaStreamHandler,
    ModelManifest,
    PullProgress,
    EmbeddingResult,
    ModelStatus,
    OllamaError,
)

# 硬件分析器
from .hardware_profiler import (
    HardwareProfiler,
    GPUProfiler,
    MemoryProfiler,
    ComputeScorer,
    BatchSizeEstimator,
    PrecisionAdvisor,
    GPUSpecs,
    MemoryStats,
    ComputeScore,
    BatchSizeEstimate,
    PrecisionRecommendation,
    GPUVendor,
    ComputeCapability,
    PrecisionType,
)

# 嵌入引擎
from .embedding_engine import (
    EmbeddingEngine,
    BatchEmbedder,
    EmbeddingCache,
    EmbeddingNormalizer,
    DimensionalityReducer,
    SimilarityComputer,
    EmbeddingVector,
    SimilarityResult,
    CacheEntry,
    SimilarityMetric,
    NormalizationType,
    ReductionMethod,
)

__all__ = [
    # 基础抽象
    "LLMBackend",
    "LLMError",
    "LLMResponse",
    "LLMChunk",
    "Message",
    "MessageRole",
    "ModelInfo",
    "ToolCall",
    "Usage",
    "GenerateParams",
    "FinishReason",
    # 适配器
    "OpenAIAdapter",
    "AnthropicAdapter",
    "LocalAdapter",
    # vLLM适配器
    "VLLMAdapter",
    "PagedAttentionManager",
    "ContinuousBatcher",
    "TensorParallelEngine",
    "PipelineParallelEngine",
    "SpeculativeDecoder",
    # Ollama适配器
    "OllamaAdapter",
    "OllamaModelManager",
    "OllamaEmbedder",
    "OllamaChatClient",
    "OllamaStreamHandler",
    "ModelManifest",
    "PullProgress",
    "EmbeddingResult",
    "ModelStatus",
    "OllamaError",
    # 硬件分析器
    "HardwareProfiler",
    "GPUProfiler",
    "MemoryProfiler",
    "ComputeScorer",
    "BatchSizeEstimator",
    "PrecisionAdvisor",
    "GPUSpecs",
    "MemoryStats",
    "ComputeScore",
    "BatchSizeEstimate",
    "PrecisionRecommendation",
    "GPUVendor",
    "ComputeCapability",
    "PrecisionType",
    # 嵌入引擎
    "EmbeddingEngine",
    "BatchEmbedder",
    "EmbeddingCache",
    "EmbeddingNormalizer",
    "DimensionalityReducer",
    "SimilarityComputer",
    "EmbeddingVector",
    "SimilarityResult",
    "CacheEntry",
    "SimilarityMetric",
    "NormalizationType",
    "ReductionMethod",
    # 网关
    "ModelGateway",
    # 降级
    "FallbackStrategy",
    "FallbackConfig",
    "FallbackResult",
    # 速率限制
    "RateLimiter",
    "RateLimitConfig",
    "RateLimitStats",
    # Prompt格式化
    "PromptFormatter",
    "SystemPromptBuilder",
    # 费用计算
    "CostCalculator",
    "CostReport",
    # 缓存
    "KVCacheManager",
    "SemanticCache",
    "CacheStats",
    # 推理优化
    "SpeculativeDecoding",
    "SpeculativeStats",
    # 流式处理
    "StreamProcessor",
    "StreamBuffer",
    # 别名
    "OpenAIBackend",
]
