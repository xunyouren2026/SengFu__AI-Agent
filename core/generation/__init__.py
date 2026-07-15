"""
AGI Unified Framework - Generation Core Module
真实生成功能模块

提供图像、视频、3D、音频、语音合成的真实实现
支持多种后端：本地模型、API服务、混合模式
"""

from .tts_engine import (
    TTSEngine,
    EdgeTTSEngine,
    BarkTTSEngine,
    CoquiTTSEngine,
    VoiceConfig,
    TTSResult,
    AudioFormat,
)

from .image_engine import (
    ImageGenerationEngine,
    DiffusersEngine,
    DALLEEngine,
    StableDiffusionAPIEngine,
    ComfyUIEngine,
    ImageConfig,
    ImageResult,
    ImageFormat,
    ControlNetConfig,
    IPAdapterConfig,
)

from .video_engine import (
    VideoGenerationEngine,
    CogVideoXEngine,
    SVDEngine,
    AnimateDiffEngine,
    VideoConfig,
    VideoResult,
    VideoFormat,
)

from .audio_engine import (
    AudioGenerationEngine,
    MusicGenEngine,
    AudioLDMEngine,
    AudioConfig,
    AudioResult,
    AudioFormat,
)

from .threed_engine import (
    ThreeDEngine,
    TripoSREngine,
    ShapEEngine,
    ThreeDConfig,
    ThreeDResult,
    ThreeDFormat,
)

from .generation_manager import (
    GenerationManager,
    GenerationTask,
    GenerationStatus,
    GenerationProgress,
    TaskPriority,
    GenerationType,
    get_generation_manager,
)

# 工厂函数
def create_tts_engine(engine_type: str = "edge", **kwargs):
    """创建TTS引擎"""
    from .tts_engine import EdgeTTSEngine, BarkTTSEngine, CoquiTTSEngine
    if engine_type == "edge":
        return EdgeTTSEngine(**kwargs)
    elif engine_type == "bark":
        return BarkTTSEngine(**kwargs)
    elif engine_type == "coqui":
        return CoquiTTSEngine(**kwargs)
    else:
        raise ValueError(f"Unknown TTS engine type: {engine_type}")

def create_image_engine(engine_type: str = "diffusers", **kwargs):
    """创建图像生成引擎"""
    from .image_engine import DiffusersEngine, DALLEEngine, StableDiffusionAPIEngine, ComfyUIEngine
    if engine_type == "diffusers":
        return DiffusersEngine(**kwargs)
    elif engine_type == "dalle":
        return DALLEEngine(**kwargs)
    elif engine_type == "stable_diffusion_api":
        return StableDiffusionAPIEngine(**kwargs)
    elif engine_type == "comfyui":
        return ComfyUIEngine(**kwargs)
    else:
        raise ValueError(f"Unknown image engine type: {engine_type}")

def create_video_engine(engine_type: str = "cogvideox", **kwargs):
    """创建视频生成引擎"""
    from .video_engine import CogVideoXEngine, SVDEngine, AnimateDiffEngine
    if engine_type == "cogvideox":
        return CogVideoXEngine(**kwargs)
    elif engine_type == "svd":
        return SVDEngine(**kwargs)
    elif engine_type == "animatediff":
        return AnimateDiffEngine(**kwargs)
    else:
        raise ValueError(f"Unknown video engine type: {engine_type}")

def create_audio_engine(engine_type: str = "musicgen", **kwargs):
    """创建音频生成引擎"""
    from .audio_engine import MusicGenEngine, AudioLDMEngine
    if engine_type == "musicgen":
        return MusicGenEngine(**kwargs)
    elif engine_type == "audioldm":
        return AudioLDMEngine(**kwargs)
    else:
        raise ValueError(f"Unknown audio engine type: {engine_type}")

def create_3d_engine(engine_type: str = "triposr", **kwargs):
    """创建3D生成引擎"""
    from .threed_engine import TripoSREngine, ShapEEngine
    if engine_type == "triposr":
        return TripoSREngine(**kwargs)
    elif engine_type == "shap_e":
        return ShapEEngine(**kwargs)
    else:
        raise ValueError(f"Unknown 3D engine type: {engine_type}")

__all__ = [
    # TTS
    "TTSEngine",
    "EdgeTTSEngine",
    "BarkTTSEngine",
    "CoquiTTSEngine",
    "VoiceConfig",
    "TTSResult",
    "AudioFormat",
    # Image
    "ImageGenerationEngine",
    "DiffusersEngine",
    "DALLEEngine",
    "StableDiffusionAPIEngine",
    "ComfyUIEngine",
    "ImageConfig",
    "ImageResult",
    "ImageFormat",
    "ControlNetConfig",
    "IPAdapterConfig",
    # Video
    "VideoGenerationEngine",
    "CogVideoXEngine",
    "SVDEngine",
    "AnimateDiffEngine",
    "VideoConfig",
    "VideoResult",
    "VideoFormat",
    # Audio
    "AudioGenerationEngine",
    "MusicGenEngine",
    "AudioLDMEngine",
    "AudioConfig",
    "AudioResult",
    # 3D
    "ThreeDEngine",
    "TripoSREngine",
    "ShapEEngine",
    "ThreeDConfig",
    "ThreeDResult",
    "ThreeDFormat",
    # Manager
    "GenerationManager",
    "GenerationTask",
    "GenerationStatus",
    "GenerationProgress",
    "TaskPriority",
    "GenerationType",
    "get_generation_manager",
    # 工厂函数
    "create_tts_engine",
    "create_image_engine",
    "create_video_engine",
    "create_audio_engine",
    "create_3d_engine",
]
