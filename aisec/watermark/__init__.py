"""
Watermark模块 - 水印与签名
"""
from .embedding import (
    WatermarkEmbedder,
    WatermarkConfig,
    WatermarkResult,
    WatermarkType
)
from .extraction import (
    WatermarkExtractor,
    ExtractionResult
)
from .content_signature import (
    ContentSigner,
    ContentSignature,
    VerificationResult,
    SignatureAlgorithm
)

__all__ = [
    # embedding.py
    "WatermarkEmbedder",
    "WatermarkConfig",
    "WatermarkResult",
    "WatermarkType",
    # extraction.py
    "WatermarkExtractor",
    "ExtractionResult",
    # content_signature.py
    "ContentSigner",
    "ContentSignature",
    "VerificationResult",
    "SignatureAlgorithm"
]
