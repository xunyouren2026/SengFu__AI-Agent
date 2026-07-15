"""
Unified tokenizer loader with caching and provider-specific adapters.

This module provides:
- Automatic model type detection
- Tokenizer caching for common models
- Provider-specific tokenizer wrappers (OpenAI, Anthropic, HuggingFace)
- Unified interface for token counting and encoding/decoding

Author: AGI Unified Framework Team
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import hashlib
import json
import re


class ModelType(Enum):
    """Supported model types/providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    OLLAMA = "ollama"
    VLLM = "vllm"
    UNKNOWN = "unknown"


@dataclass
class TokenizerInfo:
    """Information about a loaded tokenizer."""
    name: str
    model_type: ModelType
    vocab_size: int
    model_name: str
    max_length: int
    encoding_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenCount:
    """Result of token counting operation."""
    text: str
    token_count: int
    char_count: int
    tokenizer_info: TokenizerInfo
    encoded_ids: List[int]
    encoding_time_ms: float


class TokenizerCache:
    """
    Cache for loaded tokenizers with LRU eviction.
    
    Stores tokenizer instances and their metadata to avoid
    reloading the same tokenizer multiple times.
    """
    
    def __init__(self, max_size: int = 20) -> None:
        """
        Initialize tokenizer cache.
        
        Args:
            max_size: Maximum number of tokenizers to cache
        """
        self._cache: Dict[str, TokenizerWrapper] = {}
        self._access_order: List[str] = []
        self._max_size = max_size
        self._stats: Dict[str, int] = {"hits": 0, "misses": 0, "evictions": 0}
    
    def get(self, key: str) -> Optional[TokenizerWrapper]:
        """
        Get tokenizer from cache.
        
        Args:
            key: Cache key for the tokenizer
        
        Returns:
            TokenizerWrapper if found, None otherwise
        """
        if key in self._cache:
            # Move to end of access order (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            self._stats["hits"] += 1
            return self._cache[key]
        
        self._stats["misses"] += 1
        return None
    
    def put(self, key: str, tokenizer: TokenizerWrapper) -> None:
        """
        Add tokenizer to cache.
        
        Args:
            key: Cache key for the tokenizer
            tokenizer: TokenizerWrapper instance to cache
        """
        # Evict if at capacity
        if len(self._cache) >= self._max_size and key not in self._cache:
            self._evict_lru()
        
        if key not in self._cache:
            self._cache[key] = tokenizer
            self._access_order.append(key)
    
    def _evict_lru(self) -> None:
        """Evict least recently used tokenizer."""
        if self._access_order:
            lru_key = self._access_order.pop(0)
            if lru_key in self._cache:
                del self._cache[lru_key]
                self._stats["evictions"] += 1
    
    def clear(self) -> None:
        """Clear all cached tokenizers."""
        self._cache.clear()
        self._access_order.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return self._stats.copy()
    
    def get_cache_keys(self) -> List[str]:
        """Get all cached tokenizer keys."""
        return list(self._cache.keys())


class ModelTypeDetector:
    """
    Automatic model type detection from model names or configurations.
    
    Supports detection of:
    - OpenAI models (gpt-*, text-*)
    - Anthropic models (claude-*)
    - HuggingFace models (organization/model-name)
    - Local/Ollama models (llama-*, mistral-*, phi-*)
    - vLLM models (various formats)
    """
    
    # Pattern-based detection rules
    DETECTION_PATTERNS: Dict[ModelType, List[Tuple[str, re.Pattern[str]]]] = {
        ModelType.OPENAI: [
            ("gpt", re.compile(r"^(gpt-4|gpt-3\.5-turbo|gpt-4o|gpt-4-turbo|gpt-4o-mini)")),
            ("chat", re.compile(r"^(text-davinci|text-curie|text-babbage|text-ada)")),
        ],
        ModelType.ANTHROPIC: [
            ("claude", re.compile(r"^claude-([0-9]+\.)*[0-9]+(-opus|-sonnet|-haiku)?(-[0-9]+)?")),
        ],
        ModelType.HUGGINGFACE: [
            ("hf", re.compile(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+")),
            ("hf_defaults", re.compile(r"^(bert|gpt-2|t5|roberta|albert|deberta|distilbert)")),
        ],
        ModelType.OLLAMA: [
            ("ollama", re.compile(r"^ollama://")),
            ("llama_family", re.compile(r"^(llama|llama2|llama3|mistral|phi|qwen|gemma|zephyr)")),
        ],
        ModelType.VLLM: [
            ("vllm", re.compile(r"^vllm://")),
        ],
        ModelType.LOCAL: [
            ("local", re.compile(r"^local://")),
            ("file", re.compile(r"^file://")),
        ],
    }
    
    # Known model families and their default tokenizers
    MODEL_FAMILY_DEFAULTS: Dict[str, Dict[str, Any]] = {
        "llama": {
            "tokenizer": "hf:meta-llama/Llama-2-7b-hf",
            "vocab_size": 32000,
            "max_length": 4096,
        },
        "llama3": {
            "tokenizer": "hf:meta-llama/Meta-Llama-3-8B",
            "vocab_size": 128256,
            "max_length": 8192,
        },
        "mistral": {
            "tokenizer": "hf:mistralai/Mistral-7B-v0.1",
            "vocab_size": 32000,
            "max_length": 8192,
        },
        "gpt2": {
            "tokenizer": "hf:gpt2",
            "vocab_size": 50257,
            "max_length": 1024,
        },
        "gpt4": {
            "tokenizer": "cl100k_base",
            "vocab_size": 100277,
            "max_length": 8192,
        },
        "claude": {
            "tokenizer": "anthropic:claude",
            "vocab_size": 65536,
            "max_length": 2048,
        },
    }
    
    def __init__(self) -> None:
        """Initialize model type detector."""
        self._custom_rules: List[Tuple[ModelType, re.Pattern[str]]] = []
    
    def detect(self, model_name: str) -> ModelType:
        """
        Detect model type from model name.
        
        Args:
            model_name: Name or identifier of the model
        
        Returns:
            Detected ModelType
        """
        model_name_lower = model_name.lower()
        
        # Check custom rules first
        for model_type, pattern in self._custom_rules:
            if pattern.search(model_name_lower):
                return model_type
        
        # Check built-in patterns
        for model_type, patterns in self.DETECTION_PATTERNS.items():
            for name, pattern in patterns:
                if pattern.search(model_name_lower):
                    return model_type
        
        return ModelType.UNKNOWN
    
    def add_custom_rule(self, model_type: ModelType, pattern: str) -> bool:
        """
        Add a custom detection rule.
        
        Args:
            model_type: ModelType to match
            pattern: Regex pattern string
        
        Returns:
            True if pattern is valid, False otherwise
        """
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
            self._custom_rules.append((model_type, compiled))
            return True
        except re.error:
            return False
    
    def get_model_family(self, model_name: str) -> Optional[str]:
        """
        Identify the model family for known models.
        
        Args:
            model_name: Name of the model
        
        Returns:
            Model family name if identified, None otherwise
        """
        model_lower = model_name.lower()
        
        for family in self.MODEL_FAMILY_DEFAULTS:
            if family in model_lower:
                return family
        
        return None
    
    def get_default_tokenizer(self, model_name: str) -> Optional[str]:
        """
        Get default tokenizer configuration for a model.
        
        Args:
            model_name: Name of the model
        
        Returns:
            Tokenizer configuration string if available, None otherwise
        """
        family = self.get_model_family(model_name)
        
        if family and family in self.MODEL_FAMILY_DEFAULTS:
            return self.MODEL_FAMILY_DEFAULTS[family].get("tokenizer")
        
        return None
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Args:
            model_name: Name of the model
        
        Returns:
            Dictionary with model type, family, and tokenizer info
        """
        model_type = self.detect(model_name)
        family = self.get_model_family(model_name)
        tokenizer = self.get_default_tokenizer(model_name)
        
        return {
            "model_name": model_name,
            "model_type": model_type.value,
            "model_family": family,
            "default_tokenizer": tokenizer,
            "is_known_model": family is not None,
        }


class TokenizerWrapper(ABC):
    """
    Abstract base class for tokenizer wrappers.
    
    Provides a unified interface for tokenization operations
    across different providers.
    """
    
    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Text to encode
            add_special_tokens: Whether to add special tokens
        
        Returns:
            List of token IDs
        """
        pass
    
    @abstractmethod
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
        
        Returns:
            Decoded text
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count number of tokens in text.
        
        Args:
            text: Text to count
        
        Returns:
            Number of tokens
        """
        pass
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        pass
    
    @abstractmethod
    def get_max_length(self) -> int:
        """Get maximum sequence length."""
        pass
    
    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True
    ) -> List[List[int]]:
        """
        Encode multiple texts.
        
        Args:
            texts: List of texts to encode
            add_special_tokens: Whether to add special tokens
        
        Returns:
            List of token ID lists
        """
        return [self.encode(text, add_special_tokens) for text in texts]
    
    def decode_batch(
        self,
        batch_token_ids: List[List[int]],
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Decode multiple token sequences.
        
        Args:
            batch_token_ids: List of token ID lists
            skip_special_tokens: Whether to skip special tokens
        
        Returns:
            List of decoded texts
        """
        return [
            self.decode(ids, skip_special_tokens)
            for ids in batch_token_ids
        ]
    
    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """
        Count tokens for multiple texts.
        
        Args:
            texts: List of texts to count
        
        Returns:
            List of token counts
        """
        return [self.count_tokens(text) for text in texts]
    
    def get_token_info(self) -> TokenizerInfo:
        """
        Get tokenizer information.
        
        Returns:
            TokenizerInfo with tokenizer metadata
        """
        return TokenizerInfo(
            name=self.__class__.__name__,
            model_type=ModelType.UNKNOWN,
            vocab_size=self.get_vocab_size(),
            model_name="unknown",
            max_length=self.get_max_length(),
        )


class TiktokenTokenizer(TokenizerWrapper):
    """
    Tiktoken-based tokenizer for OpenAI-compatible models.
    
    Supports various encodings including cl100k_base (GPT-4),
    p50k_base (GPT-3), and r50k_base (GPT-2).
    
    Note: Requires tiktoken package. Falls back to simple
    character-based estimation if not available.
    """
    
    ENCODING_MODELS: Dict[str, List[str]] = {
        "gpt-4": ["cl100k_base"],
        "gpt-3.5-turbo": ["cl100k_base"],
        "gpt-2": ["r50k_base"],
        "text-davinci": ["p50k_base"],
    }
    
    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        model_name: str = "unknown"
    ) -> None:
        """
        Initialize Tiktoken tokenizer.
        
        Args:
            encoding_name: Name of the encoding to use
            model_name: Name of the model using this tokenizer
        """
        self.encoding_name = encoding_name
        self.model_name = model_name
        self._tokenizer = None
        self._vocab_size: Optional[int] = None
        self._max_length = 8192
        
        # Try to load tiktoken
        self._try_load_tiktoken()
    
    def _try_load_tiktoken(self) -> None:
        """Attempt to load tiktoken tokenizer."""
        try:
            import tiktoken
            self._tokenizer = tiktoken.get_encoding(self.encoding_name)
            self._vocab_size = self._tokenizer.n_vocab
        except ImportError:
            # Fall back to simple estimation
            self._vocab_size = 100277
            self._tokenizer = None
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        if self._tokenizer is not None:
            return self._tokenizer.encode(text)
        # Fallback: simple character-based estimation
        return self._estimate_tokens(text)
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """Decode token IDs to text."""
        if self._tokenizer is not None:
            return self._tokenizer.decode(token_ids)
        # Fallback: cannot decode
        return f"<tokens: {token_ids[:10]}...>"
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text))
        return self._estimate_tokens(text)
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using simple heuristic.
        
        Approximate: ~4 characters per token for English text.
        """
        if not text:
            return 0
        
        # Average 4 chars per token, with some variation
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        
        # Weighted estimate
        estimate = (char_count / 4) + (word_count * 0.1)
        
        return max(1, int(estimate))
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if self._vocab_size is not None:
            return self._vocab_size
        return 100277  # cl100k_base vocab size
    
    def get_max_length(self) -> int:
        """Get maximum sequence length."""
        return self._max_length
    
    def get_token_info(self) -> TokenizerInfo:
        """Get tokenizer information."""
        return TokenizerInfo(
            name="TiktokenTokenizer",
            model_type=ModelType.OPENAI,
            vocab_size=self.get_vocab_size(),
            model_name=self.model_name,
            max_length=self.get_max_length(),
            encoding_name=self.encoding_name,
        )


class HuggingFaceTokenizer(TokenizerWrapper):
    """
    HuggingFace tokenizer wrapper.
    
    Wraps transformers tokenizer with unified interface.
    
    Note: Requires transformers package. Provides basic
    functionality without the package.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        max_length: int = 1024
    ) -> None:
        """
        Initialize HuggingFace tokenizer.
        
        Args:
            model_name: Name or path of the HuggingFace model
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self._max_length = max_length
        self._tokenizer = None
        self._vocab_size: Optional[int] = None
        
        # Try to load transformers tokenizer
        self._try_load_transformers()
    
    def _try_load_transformers(self) -> None:
        """Attempt to load HuggingFace tokenizer."""
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self._vocab_size = self._tokenizer.vocab_size
            self._max_length = self._tokenizer.model_max_length or self._max_length
        except ImportError:
            # Fall back to basic tokenizer
            self._vocab_size = 50257  # GPT-2 vocab size
            self._tokenizer = None
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        if self._tokenizer is not None:
            return self._tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens
            )
        return self._estimate_tokens(text)
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """Decode token IDs to text."""
        if self._tokenizer is not None:
            return self._tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens
            )
        return f"<tokens: {token_ids[:10]}...>"
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text))
        return self._estimate_tokens(text)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        if not text:
            return 0
        return max(1, len(text) // 4)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if self._vocab_size is not None:
            return self._vocab_size
        return 50257
    
    def get_max_length(self) -> int:
        """Get maximum sequence length."""
        return self._max_length
    
    def get_token_info(self) -> TokenizerInfo:
        """Get tokenizer information."""
        return TokenizerInfo(
            name="HuggingFaceTokenizer",
            model_type=ModelType.HUGGINGFACE,
            vocab_size=self.get_vocab_size(),
            model_name=self.model_name,
            max_length=self.get_max_length(),
        )


class AnthropicTokenizer(TokenizerWrapper):
    """
    Anthropic tokenizer for Claude models.
    
    Uses character-based estimation with Anthropic-specific rules.
    
    Note: Anthropic uses a custom tokenizer with ~65536 vocabulary.
    """
    
    # Anthropic model configurations
    MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
        "claude-3": {"max_tokens": 4096, "context_window": 200000},
        "claude-2": {"max_tokens": 4096, "context_window": 100000},
        "claude-instant": {"max_tokens": 4096, "context_window": 100000},
    }
    
    def __init__(self, model_name: str = "claude-3-opus") -> None:
        """
        Initialize Anthropic tokenizer.
        
        Args:
            model_name: Name of the Claude model
        """
        self.model_name = model_name
        self._vocab_size = 65536
        self._max_length = self._get_max_length(model_name)
    
    def _get_max_length(self, model_name: str) -> int:
        """Get maximum length for model."""
        for prefix, config in self.MODEL_CONFIGS.items():
            if prefix in model_name.lower():
                return config.get("context_window", 200000)
        return 200000
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        # Anthropic uses ~3.5 chars per token on average
        tokens = self._estimate_tokens(text)
        # Return placeholder IDs (real implementation would use actual tokenizer)
        return list(range(tokens))
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """Decode token IDs to text."""
        # Cannot decode without actual tokenizer
        return f"<{len(token_ids)} tokens>"
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return self._estimate_tokens(text)
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using Anthropic-specific heuristics.
        
        Anthropic's tokenizer averages ~3.5 chars per token.
        """
        if not text:
            return 0
        
        # Count words for better estimation
        words = text.split()
        word_count = len(words)
        
        # Anthropic: ~3.5 chars per token
        char_count = len(text)
        estimate = char_count / 3.5
        
        # Adjust for words (usually slightly fewer tokens than words)
        word_adjustment = word_count * 0.5
        
        return max(1, int(estimate + word_adjustment))
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self._vocab_size
    
    def get_max_length(self) -> int:
        """Get maximum sequence length."""
        return self._max_length
    
    def get_token_info(self) -> TokenizerInfo:
        """Get tokenizer information."""
        return TokenizerInfo(
            name="AnthropicTokenizer",
            model_type=ModelType.ANTHROPIC,
            vocab_size=self.get_vocab_size(),
            model_name=self.model_name,
            max_length=self.get_max_length(),
        )


class TokenizerLoader:
    """
    Unified tokenizer loader with caching and auto-detection.
    
    Main entry point for loading tokenizers with support for:
    - Automatic model type detection
    - Tokenizer caching
    - Multiple provider support
    - Custom tokenizer registration
    """
    
    def __init__(self, cache_size: int = 20) -> None:
        """
        Initialize tokenizer loader.
        
        Args:
            cache_size: Maximum number of tokenizers to cache
        """
        self._cache = TokenizerCache(max_size=cache_size)
        self._detector = ModelTypeDetector()
        self._custom_loaders: Dict[str, Callable[[str], TokenizerWrapper]] = {}
        self._tokenizer_classes: Dict[ModelType, type] = {
            ModelType.OPENAI: TiktokenTokenizer,
            ModelType.ANTHROPIC: AnthropicTokenizer,
            ModelType.HUGGINGFACE: HuggingFaceTokenizer,
        }
    
    def load(
        self,
        model_name: str,
        force_reload: bool = False
    ) -> TokenizerWrapper:
        """
        Load a tokenizer for the given model.
        
        Args:
            model_name: Name of the model
            force_reload: Force reload even if cached
        
        Returns:
            TokenizerWrapper instance
        
        Raises:
            ValueError: If model type cannot be determined
        """
        cache_key = self._get_cache_key(model_name)
        
        # Check cache
        if not force_reload:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
        
        # Detect model type
        model_type = self._detector.detect(model_name)
        
        # Get or create tokenizer
        tokenizer = self._create_tokenizer(model_name, model_type)
        
        # Cache the tokenizer
        self._cache.put(cache_key, tokenizer)
        
        return tokenizer
    
    def _create_tokenizer(
        self,
        model_name: str,
        model_type: ModelType
    ) -> TokenizerWrapper:
        """
        Create tokenizer for the given model type.
        
        Args:
            model_name: Name of the model
            model_type: Detected model type
        
        Returns:
            TokenizerWrapper instance
        """
        # Check for custom loader
        for loader_name, loader in self._custom_loaders.items():
            if loader_name.lower() in model_name.lower():
                return loader(model_name)
        
        # Use appropriate tokenizer class
        tokenizer_class = self._tokenizer_classes.get(model_type)
        
        if tokenizer_class == TiktokenTokenizer:
            # Determine encoding
            encoding = self._get_encoding_for_model(model_name)
            return TiktokenTokenizer(encoding, model_name)
        
        elif tokenizer_class == AnthropicTokenizer:
            return AnthropicTokenizer(model_name)
        
        elif tokenizer_class == HuggingFaceTokenizer:
            return HuggingFaceTokenizer(model_name)
        
        # Fallback: use model family defaults
        family = self._detector.get_model_family(model_name)
        if family:
            default_tok = self._detector.get_default_tokenizer(model_name)
            if default_tok:
                return self.load(default_tok)
        
        # Last resort: generic tokenizer
        return TiktokenTokenizer("cl100k_base", model_name)
    
    def _get_encoding_for_model(self, model_name: str) -> str:
        """Get appropriate encoding for OpenAI model."""
        model_lower = model_name.lower()
        
        if "gpt-4" in model_lower:
            return "cl100k_base"
        elif "gpt-3.5" in model_lower:
            return "cl100k_base"
        elif "gpt-2" in model_lower:
            return "r50k_base"
        elif "text-davinci" in model_lower:
            return "p50k_base"
        else:
            return "cl100k_base"
    
    def _get_cache_key(self, model_name: str) -> str:
        """Generate cache key for model."""
        return hashlib.md5(model_name.encode()).hexdigest()
    
    def register_custom_loader(
        self,
        name: str,
        loader: Callable[[str], TokenizerWrapper]
    ) -> None:
        """
        Register a custom tokenizer loader.
        
        Args:
            name: Name identifier for the loader
            loader: Function that creates a TokenizerWrapper
        """
        self._custom_loaders[name] = loader
    
    def register_tokenizer_class(
        self,
        model_type: ModelType,
        tokenizer_class: type
    ) -> None:
        """
        Register a custom tokenizer class for a model type.
        
        Args:
            model_type: ModelType to register
            tokenizer_class: TokenizerWrapper subclass
        """
        self._tokenizer_classes[model_type] = tokenizer_class
    
    def get_tokenizer_info(self, model_name: str) -> TokenizerInfo:
        """
        Get tokenizer information without loading.
        
        Args:
            model_name: Name of the model
        
        Returns:
            TokenizerInfo with basic information
        """
        model_type = self._detector.detect(model_name)
        
        # Try to get from cache
        cache_key = self._get_cache_key(model_name)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached.get_token_info()
        
        # Return basic info
        family = self._detector.get_model_family(model_name)
        vocab_size = 100277  # Default
        max_length = 4096
        
        if family and family in ModelTypeDetector.MODEL_FAMILY_DEFAULTS:
            defaults = ModelTypeDetector.MODEL_FAMILY_DEFAULTS[family]
            vocab_size = defaults.get("vocab_size", vocab_size)
            max_length = defaults.get("max_length", max_length)
        
        return TokenizerInfo(
            name=f"{model_type.value.title()}Tokenizer",
            model_type=model_type,
            vocab_size=vocab_size,
            model_name=model_name,
            max_length=max_length,
        )
    
    def count_tokens(self, text: str, model_name: str) -> int:
        """
        Count tokens for text using appropriate tokenizer.
        
        Args:
            text: Text to count
            model_name: Model name for tokenizer selection
        
        Returns:
            Number of tokens
        """
        tokenizer = self.load(model_name)
        return tokenizer.count_tokens(text)
    
    def count_tokens_batch(
        self,
        texts: List[str],
        model_name: str
    ) -> List[int]:
        """
        Count tokens for multiple texts.
        
        Args:
            texts: List of texts to count
            model_name: Model name for tokenizer selection
        
        Returns:
            List of token counts
        """
        tokenizer = self.load(model_name)
        return tokenizer.count_tokens_batch(texts)
    
    def truncate_text(
        self,
        text: str,
        model_name: str,
        max_tokens: int
    ) -> str:
        """
        Truncate text to fit within token limit.
        
        Args:
            text: Text to truncate
            model_name: Model name for tokenizer selection
            max_tokens: Maximum number of tokens
        
        Returns:
            Truncated text
        """
        tokenizer = self.load(model_name)
        tokens = tokenizer.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        return tokenizer.decode(truncated_tokens)
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get tokenizer cache statistics."""
        return self._cache.get_stats()
    
    def clear_cache(self) -> None:
        """Clear all cached tokenizers."""
        self._cache.clear()


# Global tokenizer loader instance
_default_loader: Optional[TokenizerLoader] = None


def get_tokenizer_loader() -> TokenizerLoader:
    """
    Get the global tokenizer loader instance.
    
    Returns:
        TokenizerLoader singleton
    """
    global _default_loader
    if _default_loader is None:
        _default_loader = TokenizerLoader()
    return _default_loader


def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """
    Quick function to count tokens for text.
    
    Args:
        text: Text to count
        model_name: Model name for tokenizer selection
    
    Returns:
        Number of tokens
    """
    loader = get_tokenizer_loader()
    return loader.count_tokens(text, model_name)


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model_name: str = "gpt-4"
) -> float:
    """
    Estimate API cost based on token counts.
    
    Args:
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        model_name: Model name for pricing
    
    Returns:
        Estimated cost in USD
    """
    # Pricing per 1M tokens (approximate)
    PRICING: Dict[str, Tuple[float, float]] = {
        "gpt-4": (30.0, 60.0),  # prompt, completion
        "gpt-4-turbo": (10.0, 30.0),
        "gpt-3.5-turbo": (0.5, 1.5),
        "claude-3-opus": (15.0, 75.0),
        "claude-3-sonnet": (3.0, 15.0),
        "claude-3-haiku": (0.25, 1.25),
    }
    
    model_lower = model_name.lower()
    
    for model_pattern, (prompt_price, completion_price) in PRICING.items():
        if model_pattern in model_lower:
            prompt_cost = (prompt_tokens / 1_000_000) * prompt_price
            completion_cost = (completion_tokens / 1_000_000) * completion_price
            return prompt_cost + completion_cost
    
    # Default pricing for unknown models
    return (prompt_tokens + completion_tokens) / 1_000_000 * 10.0
