"""
音频编码器模块
提供语音和音频特征提取功能
"""
from .whisper_encoder import (
    WhisperEncoder,
    MelSpectrogram,
    AudioEncoder,
    TextDecoder,
    EncoderLayer,
    DecoderLayer,
    LayerNorm,
    MultiHeadAttention,
    FeedForward,
    create_whisper_encoder
)

from .clap_encoder import (
    CLAPEncoder,
    CLAPAudioEncoder,
    CLAPTextEncoder,
    AudioFeatureExtractor,
    TransformerBlock,
    create_clap_encoder
)

__all__ = [
    # Whisper
    'WhisperEncoder',
    'MelSpectrogram',
    'AudioEncoder',
    'TextDecoder',
    'EncoderLayer',
    'DecoderLayer',
    'LayerNorm',
    'MultiHeadAttention',
    'FeedForward',
    'create_whisper_encoder',
    
    # CLAP
    'CLAPEncoder',
    'CLAPAudioEncoder',
    'CLAPTextEncoder',
    'AudioFeatureExtractor',
    'TransformerBlock',
    'create_clap_encoder'
]
