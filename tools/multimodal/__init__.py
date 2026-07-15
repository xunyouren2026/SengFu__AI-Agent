"""
Multimodal Tools Module - 多模态工具模块
提供视觉问答、图像描述、OCR、物体检测、图表读取、文档解析、图像搜索、视频剪辑等工具
"""

from .vqa_tool import (
    VQATool,
    VQAConfig,
    VQAQuestion,
    VQAAnswer,
    VQATaskType,
    BoundingBox as VQABoundingBox
)

from .caption_tool import (
    CaptionTool,
    CaptionConfig,
    CaptionResult,
    CaptionStyle,
    CaptionLength
)

from .ocr_tool import (
    OCRTool,
    OCRConfig,
    OCRResult,
    OCRLanguage,
    OCRMode,
    TextBox
)

from .object_detection import (
    ObjectDetector,
    DetectionConfig,
    DetectionResult,
    Detection,
    DetectionModel,
    BoundingBox as DetectionBoundingBox
)

from .chart_reader import (
    ChartReader,
    ChartReaderConfig,
    ChartData,
    ChartType,
    DataPoint,
    DataSeries,
    AxisInfo
)

from .document_parser import (
    DocumentParser,
    ParsedDocument,
    DocumentMetadata,
    DocumentSection,
    DocumentType,
    ContentType,
    TableData
)

from .image_search import (
    ImageSearchTool,
    ImageSearchConfig,
    ImageSearchResult,
    SearchResult,
    SearchEngine,
    SearchType,
    ImageFeature
)

from .video_trim import (
    VideoTrimmer,
    TrimConfig,
    TrimResult,
    VideoInfo,
    VideoFormat,
    VideoCodec,
    AudioCodec
)

__all__ = [
    # VQA
    "VQATool",
    "VQAConfig",
    "VQAQuestion",
    "VQAAnswer",
    "VQATaskType",
    # Caption
    "CaptionTool",
    "CaptionConfig",
    "CaptionResult",
    "CaptionStyle",
    "CaptionLength",
    # OCR
    "OCRTool",
    "OCRConfig",
    "OCRResult",
    "OCRLanguage",
    "OCRMode",
    "TextBox",
    # Object Detection
    "ObjectDetector",
    "DetectionConfig",
    "DetectionResult",
    "Detection",
    "DetectionModel",
    # Chart Reader
    "ChartReader",
    "ChartReaderConfig",
    "ChartData",
    "ChartType",
    "DataPoint",
    "DataSeries",
    "AxisInfo",
    # Document Parser
    "DocumentParser",
    "ParsedDocument",
    "DocumentMetadata",
    "DocumentSection",
    "DocumentType",
    "ContentType",
    "TableData",
    # Image Search
    "ImageSearchTool",
    "ImageSearchConfig",
    "ImageSearchResult",
    "SearchResult",
    "SearchEngine",
    "SearchType",
    "ImageFeature",
    # Video Trim
    "VideoTrimmer",
    "TrimConfig",
    "TrimResult",
    "VideoInfo",
    "VideoFormat",
    "VideoCodec",
    "AudioCodec"
]
