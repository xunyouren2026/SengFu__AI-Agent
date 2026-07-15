"""
Music Generation - 音乐生成模块
包含MusicGen和AudioCraft音乐生成管线。
"""

from .musicgen import (
    MusicTheory,
    MusicGenerator,
    MelodyGenerator,
    RhythmGenerator,
    InstrumentSynthesizer,
    GenerationResult,
)
from .audiocraft import (
    AudioCraft,
    MusicGenWrapper,
    AudioPromptEncoder,
    MultiTrackGenerator,
    GenreController,
    MixMaster,
    AudioCraftConfig,
    AudioPrompt,
    Track,
    MusicOutput,
    MusicModel,
)

__all__ = [
    "MusicTheory",
    "MusicGenerator",
    "MelodyGenerator",
    "RhythmGenerator",
    "InstrumentSynthesizer",
    "GenerationResult",
    "AudioCraft",
    "MusicGenWrapper",
    "AudioPromptEncoder",
    "MultiTrackGenerator",
    "GenreController",
    "MixMaster",
    "AudioCraftConfig",
    "AudioPrompt",
    "Track",
    "MusicOutput",
    "MusicModel",
]
