"""
多媒体文件处理系统
包含图像、视频、音频处理功能
"""

from .image.format_converter import (
    FormatConverter,
    ImageFormat,
    ImageInfo,
    PixelBuffer
)

from .image.exif_editor import (
    ExifEditor,
    ExifTag,
    ExifDataType,
    ExifValue,
    IFDEntry
)

from .video.ffmpeg_wrapper import (
    FFmpegWrapper,
    VideoCodec,
    AudioCodec,
    ContainerFormat,
    MediaInfo,
    TranscodeOptions,
    ClipSegment
)

from .video.thumbnail_generator import (
    ThumbnailGenerator,
    ThumbnailMode,
    ThumbnailConfig,
    ThumbnailInfo,
    OutputFormat as ThumbnailOutputFormat
)

from .audio.transcriber import (
    Transcriber,
    AudioProcessor,
    TranscriptionModel,
    TranscriptionLanguage,
    TranscriptionConfig,
    TranscriptionResult,
    TranscriptionSegment,
    OutputFormat as TranscriptionOutputFormat
)

from .audio.diarization import (
    Diarizer,
    DiarizationMethod,
    DiarizationConfig,
    DiarizationResult,
    SpeakerSegment,
    SpeakerInfo,
    VoiceActivityDetector,
    SpeakerEmbedding,
    SpeakerClustering
)


__all__ = [
    # 图像处理
    'FormatConverter',
    'ImageFormat',
    'ImageInfo',
    'PixelBuffer',
    'ExifEditor',
    'ExifTag',
    'ExifDataType',
    'ExifValue',
    'IFDEntry',
    
    # 视频处理
    'FFmpegWrapper',
    'VideoCodec',
    'AudioCodec',
    'ContainerFormat',
    'MediaInfo',
    'TranscodeOptions',
    'ClipSegment',
    'ThumbnailGenerator',
    'ThumbnailMode',
    'ThumbnailConfig',
    'ThumbnailInfo',
    'ThumbnailOutputFormat',
    
    # 音频处理
    'Transcriber',
    'AudioProcessor',
    'TranscriptionModel',
    'TranscriptionLanguage',
    'TranscriptionConfig',
    'TranscriptionResult',
    'TranscriptionSegment',
    'TranscriptionOutputFormat',
    'Diarizer',
    'DiarizationMethod',
    'DiarizationConfig',
    'DiarizationResult',
    'SpeakerSegment',
    'SpeakerInfo',
    'VoiceActivityDetector',
    'SpeakerEmbedding',
    'SpeakerClustering',
]


__version__ = '1.0.0'
__author__ = 'AGI Unified Framework'
