"""
内容生成API路由

提供统一的内容生成服务，支持视频、图像、音频、3D模型等多种生成类型。
包含完整的生成生命周期管理、历史记录、作品画廊、实时进度等功能。

端点:
    GET    /types                    - 获取支持的生成类型
    POST   /video                    - 生成视频
    GET    /video/{id}              - 获取视频生成状态
    DELETE /video/{id}              - 取消视频生成
    POST   /image                    - 生成图像
    GET    /image/{id}              - 获取图像生成状态
    DELETE /image/{id}              - 取消图像生成
    POST   /audio                    - 生成音频
    GET    /audio/{id}              - 获取音频生成状态
    DELETE /audio/{id}              - 取消音频生成
    POST   /3d                       - 生成3D模型
    GET    /3d/{id}                 - 获取3D生成状态
    GET    /history                  - 获取生成历史
    GET    /gallery                  - 获取作品画廊
    POST   /{id}/download           - 下载生成结果
    POST   /{id}/share              - 分享作品
    POST   /{id}/delete             - 删除作品
    GET    /templates                - 获取提示词模板
    POST   /batch                    - 批量生成
    WS     /ws/generation/{id}      - 实时生成进度

使用示例:
    >>> # 生成视频
    >>> POST /api/v1/generation/video
    >>> {
    >>>     "prompt": "A futuristic city at sunset",
    >>>     "duration": 10,
    >>>     "resolution": "1080p",
    >>>     "style": "cinematic"
    >>> }
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field, validator

from ..validators.schemas import BaseResponse, ErrorResponse, PaginatedResponse
from ..dependencies.injection import get_current_user, require_permissions, get_db_session

# 配置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter()

# =============================================================================
# 枚举类型定义
# =============================================================================

class GenerationType(str, Enum):
    """生成内容类型"""
    VIDEO = "video"
    IMAGE = "image"
    AUDIO = "audio"
    MODEL_3D = "3d"
    TEXT = "text"
    ANIMATION = "animation"


class GenerationStatus(str, Enum):
    """生成任务状态"""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class VideoResolution(str, Enum):
    """视频分辨率"""
    SD_480P = "480p"
    HD_720P = "720p"
    FHD_1080P = "1080p"
    QHD_1440P = "1440p"
    UHD_4K = "4k"
    UHD_8K = "8k"


class VideoStyle(str, Enum):
    """视频风格"""
    CINEMATIC = "cinematic"
    ANIMATED = "animated"
    REALISTIC = "realistic"
    STYLIZED = "stylized"
    DOCUMENTARY = "documentary"
    MUSIC_VIDEO = "music_video"
    COMMERCIAL = "commercial"


class ImageStyle(str, Enum):
    """图像风格"""
    PHOTOREALISTIC = "photorealistic"
    DIGITAL_ART = "digital_art"
    OIL_PAINTING = "oil_painting"
    WATERCOLOR = "watercolor"
    SKETCH = "sketch"
    ANIME = "anime"
    PIXEL_ART = "pixel_art"
    THREE_D_RENDER = "3d_render"
    CONCEPT_ART = "concept_art"


class ImageAspectRatio(str, Enum):
    """图像宽高比"""
    SQUARE = "1:1"
    PORTRAIT = "9:16"
    LANDSCAPE = "16:9"
    WIDE = "21:9"
    CLASSIC = "4:3"


class AudioType(str, Enum):
    """音频类型"""
    MUSIC = "music"
    SOUND_EFFECT = "sound_effect"
    SPEECH = "speech"
    AMBIENT = "ambient"
    INSTRUMENTAL = "instrumental"


class Model3DFormat(str, Enum):
    """3D模型格式"""
    OBJ = "obj"
    FBX = "fbx"
    GLTF = "gltf"
    GLB = "glb"
    STL = "stl"
    USDZ = "usdz"


class Model3DQuality(str, Enum):
    """3D模型质量"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


class ShareVisibility(str, Enum):
    """分享可见性"""
    PUBLIC = "public"
    PRIVATE = "private"
    UNLISTED = "unlisted"
    COMMUNITY = "community"


class BatchStatus(str, Enum):
    """批量生成状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


# =============================================================================
# Pydantic模型定义
# =============================================================================

class GenerationTypeInfo(BaseModel):
    """生成类型信息"""
    type: GenerationType = Field(..., description="生成类型")
    name: str = Field(..., description="类型名称")
    description: str = Field(..., description="类型描述")
    supported_models: List[str] = Field(default_factory=list, description="支持的模型列表")
    max_duration: Optional[int] = Field(None, description="最大时长(秒)")
    max_resolution: Optional[str] = Field(None, description="最大分辨率")
    requires_credits: int = Field(default=1, description="所需积分")
    is_available: bool = Field(default=True, description="是否可用")
    features: List[str] = Field(default_factory=list, description="功能特性")


class VideoGenerationRequest(BaseModel):
    """视频生成请求"""
    prompt: str = Field(..., min_length=1, max_length=2000, description="生成提示词")
    negative_prompt: Optional[str] = Field(None, max_length=1000, description="负面提示词")
    duration: int = Field(default=5, ge=1, le=60, description="视频时长(秒)")
    resolution: VideoResolution = Field(default=VideoResolution.FHD_1080P, description="分辨率")
    fps: int = Field(default=24, ge=1, le=60, description="帧率")
    style: VideoStyle = Field(default=VideoStyle.CINEMATIC, description="视频风格")
    seed: Optional[int] = Field(None, description="随机种子")
    model_id: Optional[str] = Field(None, description="指定模型ID")
    aspect_ratio: str = Field(default="16:9", description="宽高比")
    motion_strength: float = Field(default=0.5, ge=0.0, le=1.0, description="运动强度")
    camera_motion: Optional[str] = Field(None, description="相机运动")
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v or not v.strip():
            raise ValueError("提示词不能为空")
        return v.strip()


class VideoGenerationResponse(BaseModel):
    """视频生成响应"""
    id: str = Field(..., description="生成任务ID")
    status: GenerationStatus = Field(..., description="生成状态")
    prompt: str = Field(..., description="生成提示词")
    progress: float = Field(default=0.0, ge=0.0, le=100.0, description="生成进度")
    estimated_time: Optional[int] = Field(None, description="预计剩余时间(秒)")
    result_url: Optional[str] = Field(None, description="结果URL")
    thumbnail_url: Optional[str] = Field(None, description="缩略图URL")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    credits_used: int = Field(default=0, description="使用积分")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class ImageGenerationRequest(BaseModel):
    """图像生成请求"""
    prompt: str = Field(..., min_length=1, max_length=2000, description="生成提示词")
    negative_prompt: Optional[str] = Field(None, max_length=1000, description="负面提示词")
    width: int = Field(default=1024, ge=256, le=4096, description="图像宽度")
    height: int = Field(default=1024, ge=256, le=4096, description="图像高度")
    aspect_ratio: ImageAspectRatio = Field(default=ImageAspectRatio.SQUARE, description="宽高比")
    style: ImageStyle = Field(default=ImageStyle.PHOTOREALISTIC, description="图像风格")
    seed: Optional[int] = Field(None, description="随机种子")
    model_id: Optional[str] = Field(None, description="指定模型ID")
    num_images: int = Field(default=1, ge=1, le=10, description="生成数量")
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0, description="引导强度")
    steps: int = Field(default=50, ge=10, le=150, description="推理步数")
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v or not v.strip():
            raise ValueError("提示词不能为空")
        return v.strip()


class ImageGenerationResponse(BaseModel):
    """图像生成响应"""
    id: str = Field(..., description="生成任务ID")
    status: GenerationStatus = Field(..., description="生成状态")
    prompt: str = Field(..., description="生成提示词")
    progress: float = Field(default=0.0, ge=0.0, le=100.0, description="生成进度")
    estimated_time: Optional[int] = Field(None, description="预计剩余时间(秒)")
    result_urls: List[str] = Field(default_factory=list, description="结果URL列表")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    credits_used: int = Field(default=0, description="使用积分")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class AudioGenerationRequest(BaseModel):
    """音频生成请求"""
    prompt: str = Field(..., min_length=1, max_length=1000, description="生成提示词")
    audio_type: AudioType = Field(default=AudioType.MUSIC, description="音频类型")
    duration: int = Field(default=30, ge=1, le=300, description="音频时长(秒)")
    tempo: Optional[int] = Field(None, ge=60, le=200, description="节拍(BPM)")
    key: Optional[str] = Field(None, description="调性")
    genre: Optional[str] = Field(None, description="音乐风格")
    mood: Optional[str] = Field(None, description="情绪")
    seed: Optional[int] = Field(None, description="随机种子")
    model_id: Optional[str] = Field(None, description="指定模型ID")
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v or not v.strip():
            raise ValueError("提示词不能为空")
        return v.strip()


class AudioGenerationResponse(BaseModel):
    """音频生成响应"""
    id: str = Field(..., description="生成任务ID")
    status: GenerationStatus = Field(..., description="生成状态")
    prompt: str = Field(..., description="生成提示词")
    audio_type: AudioType = Field(..., description="音频类型")
    progress: float = Field(default=0.0, ge=0.0, le=100.0, description="生成进度")
    estimated_time: Optional[int] = Field(None, description="预计剩余时间(秒)")
    result_url: Optional[str] = Field(None, description="结果URL")
    waveform_url: Optional[str] = Field(None, description="波形图URL")
    duration: int = Field(..., description="音频时长(秒)")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    credits_used: int = Field(default=0, description="使用积分")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class Model3DGenerationRequest(BaseModel):
    """3D模型生成请求"""
    prompt: str = Field(..., min_length=1, max_length=2000, description="生成提示词")
    negative_prompt: Optional[str] = Field(None, max_length=1000, description="负面提示词")
    format: Model3DFormat = Field(default=Model3DFormat.GLB, description="输出格式")
    quality: Model3DQuality = Field(default=Model3DQuality.HIGH, description="模型质量")
    texture_resolution: int = Field(default=2048, ge=512, le=8192, description="纹理分辨率")
    poly_count: Optional[str] = Field(None, description="多边形数量级别")
    seed: Optional[int] = Field(None, description="随机种子")
    model_id: Optional[str] = Field(None, description="指定模型ID")
    auto_rig: bool = Field(default=False, description="自动绑定")
    generate_textures: bool = Field(default=True, description="生成纹理")
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v or not v.strip():
            raise ValueError("提示词不能为空")
        return v.strip()


class Model3DGenerationResponse(BaseModel):
    """3D模型生成响应"""
    id: str = Field(..., description="生成任务ID")
    status: GenerationStatus = Field(..., description="生成状态")
    prompt: str = Field(..., description="生成提示词")
    format: Model3DFormat = Field(..., description="输出格式")
    progress: float = Field(default=0.0, ge=0.0, le=100.0, description="生成进度")
    estimated_time: Optional[int] = Field(None, description="预计剩余时间(秒)")
    result_url: Optional[str] = Field(None, description="结果URL")
    preview_url: Optional[str] = Field(None, description="预览URL")
    file_size: Optional[int] = Field(None, description="文件大小(字节)")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    credits_used: int = Field(default=0, description="使用积分")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class GenerationHistoryItem(BaseModel):
    """生成历史项"""
    id: str = Field(..., description="生成任务ID")
    type: GenerationType = Field(..., description="生成类型")
    status: GenerationStatus = Field(..., description="生成状态")
    prompt: str = Field(..., description="生成提示词")
    thumbnail_url: Optional[str] = Field(None, description="缩略图URL")
    result_url: Optional[str] = Field(None, description="结果URL")
    created_at: datetime = Field(..., description="创建时间")
    credits_used: int = Field(default=0, description="使用积分")
    is_favorite: bool = Field(default=False, description="是否收藏")
    tags: List[str] = Field(default_factory=list, description="标签")


class GalleryItem(BaseModel):
    """画廊作品项"""
    id: str = Field(..., description="作品ID")
    type: GenerationType = Field(..., description="作品类型")
    title: Optional[str] = Field(None, description="作品标题")
    prompt: str = Field(..., description="生成提示词")
    thumbnail_url: str = Field(..., description="缩略图URL")
    result_url: str = Field(..., description="结果URL")
    author_id: str = Field(..., description="作者ID")
    author_name: str = Field(..., description="作者名称")
    created_at: datetime = Field(..., description="创建时间")
    likes: int = Field(default=0, description="点赞数")
    views: int = Field(default=0, description="浏览数")
    visibility: ShareVisibility = Field(default=ShareVisibility.PUBLIC, description="可见性")
    tags: List[str] = Field(default_factory=list, description="标签")


class DownloadRequest(BaseModel):
    """下载请求"""
    format: Optional[str] = Field(None, description="目标格式")
    quality: Optional[str] = Field(None, description="质量级别")
    include_metadata: bool = Field(default=True, description="包含元数据")


class DownloadResponse(BaseModel):
    """下载响应"""
    download_url: str = Field(..., description="下载URL")
    expires_at: datetime = Field(..., description="过期时间")
    file_name: str = Field(..., description="文件名")
    file_size: int = Field(..., description="文件大小(字节)")
    format: str = Field(..., description="文件格式")


class ShareRequest(BaseModel):
    """分享请求"""
    visibility: ShareVisibility = Field(default=ShareVisibility.PUBLIC, description="可见性")
    title: Optional[str] = Field(None, max_length=200, description="作品标题")
    description: Optional[str] = Field(None, max_length=2000, description="作品描述")
    tags: List[str] = Field(default_factory=list, description="标签")
    allow_download: bool = Field(default=True, description="允许下载")
    allow_remix: bool = Field(default=True, description="允许 remix")


class ShareResponse(BaseModel):
    """分享响应"""
    share_id: str = Field(..., description="分享ID")
    share_url: str = Field(..., description="分享URL")
    visibility: ShareVisibility = Field(..., description="可见性")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")


class PromptTemplate(BaseModel):
    """提示词模板"""
    id: str = Field(..., description="模板ID")
    name: str = Field(..., description="模板名称")
    description: str = Field(..., description="模板描述")
    type: GenerationType = Field(..., description="适用类型")
    template: str = Field(..., description="模板内容")
    variables: List[str] = Field(default_factory=list, description="变量列表")
    example_values: Dict[str, str] = Field(default_factory=dict, description="示例值")
    tags: List[str] = Field(default_factory=list, description="标签")
    popularity: int = Field(default=0, description="使用次数")
    created_by: Optional[str] = Field(None, description="创建者")


class BatchGenerationRequest(BaseModel):
    """批量生成请求"""
    type: GenerationType = Field(..., description="生成类型")
    prompts: List[str] = Field(..., min_items=1, max_items=100, description="提示词列表")
    common_params: Dict[str, Any] = Field(default_factory=dict, description="通用参数")
    priority: int = Field(default=5, ge=1, le=10, description="优先级")
    
    @validator('prompts')
    def validate_prompts(cls, v):
        if not v:
            raise ValueError("提示词列表不能为空")
        for prompt in v:
            if not prompt or not prompt.strip():
                raise ValueError("提示词不能为空")
        return v


class BatchGenerationResponse(BaseModel):
    """批量生成响应"""
    batch_id: str = Field(..., description="批次ID")
    status: BatchStatus = Field(..., description="批次状态")
    total_tasks: int = Field(..., description="总任务数")
    completed_tasks: int = Field(default=0, description="已完成任务数")
    failed_tasks: int = Field(default=0, description="失败任务数")
    task_ids: List[str] = Field(default_factory=list, description="任务ID列表")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    estimated_completion: Optional[datetime] = Field(None, description="预计完成时间")


class GenerationProgress(BaseModel):
    """生成进度"""
    task_id: str = Field(..., description="任务ID")
    status: GenerationStatus = Field(..., description="当前状态")
    progress: float = Field(..., ge=0.0, le=100.0, description="进度百分比")
    current_step: Optional[str] = Field(None, description="当前步骤")
    estimated_time_remaining: Optional[int] = Field(None, description="预计剩余时间(秒)")
    message: Optional[str] = Field(None, description="状态消息")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="时间戳")


# =============================================================================
# 模拟数据存储（实际应用中应使用数据库）
# =============================================================================

# 生成任务存储
_generation_tasks: Dict[str, Dict[str, Any]] = {}

# 画廊作品存储
_gallery_items: Dict[str, Dict[str, Any]] = {}

# 提示词模板存储
_prompt_templates: Dict[str, Dict[str, Any]] = {}

# 批量任务存储
_batch_tasks: Dict[str, Dict[str, Any]] = {}

# WebSocket连接存储
_websocket_connections: Dict[str, List[WebSocket]] = {}


def _init_mock_data():
    """初始化模拟数据"""
    global _prompt_templates
    
    templates = [
        {
            "id": "video-cinematic-001",
            "name": "电影级场景",
            "description": "生成高质量电影级视频场景",
            "type": GenerationType.VIDEO,
            "template": "Cinematic shot of {subject}, {lighting} lighting, {mood} atmosphere, {camera_movement}, 8k resolution, professional color grading",
            "variables": ["subject", "lighting", "mood", "camera_movement"],
            "example_values": {
                "subject": "a futuristic cityscape",
                "lighting": "golden hour",
                "mood": "epic",
                "camera_movement": "slow dolly zoom"
            },
            "tags": ["video", "cinematic", "professional"],
            "popularity": 1250,
            "created_by": "system"
        },
        {
            "id": "image-character-001",
            "name": "角色设计",
            "description": "生成独特的角色设计概念图",
            "type": GenerationType.IMAGE,
            "template": "Character design of {character_type}, {style} style, {features}, {pose}, highly detailed, concept art",
            "variables": ["character_type", "style", "features", "pose"],
            "example_values": {
                "character_type": "a cyberpunk warrior",
                "style": "anime",
                "features": "neon accents, mechanical arm",
                "pose": "dynamic action pose"
            },
            "tags": ["image", "character", "concept"],
            "popularity": 980,
            "created_by": "system"
        },
        {
            "id": "audio-ambient-001",
            "name": "环境音景",
            "description": "生成沉浸式环境音效",
            "type": GenerationType.AUDIO,
            "template": "Ambient soundscape of {environment}, {time_of_day}, {mood}, immersive, high quality",
            "variables": ["environment", "time_of_day", "mood"],
            "example_values": {
                "environment": "a peaceful forest",
                "time_of_day": "dawn",
                "mood": "serene"
            },
            "tags": ["audio", "ambient", "soundscape"],
            "popularity": 650,
            "created_by": "system"
        },
        {
            "id": "3d-product-001",
            "name": "产品模型",
            "description": "生成产品3D模型",
            "type": GenerationType.MODEL_3D,
            "template": "3D model of {product}, {style} style, {material}, highly detailed, ready for rendering",
            "variables": ["product", "style", "material"],
            "example_values": {
                "product": "a modern smartphone",
                "style": "minimalist",
                "material": "glass and aluminum"
            },
            "tags": ["3d", "product", "model"],
            "popularity": 420,
            "created_by": "system"
        }
    ]
    
    for template in templates:
        _prompt_templates[template["id"]] = template


# 初始化模拟数据
_init_mock_data()


# =============================================================================
# API端点定义
# =============================================================================

@router.get(
    "/types",
    response_model=BaseResponse[List[GenerationTypeInfo]],
    summary="获取支持的生成类型",
    description="获取系统支持的所有内容生成类型及其详细信息"
)
async def get_generation_types(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> BaseResponse[List[GenerationTypeInfo]]:
    """
    获取支持的生成类型列表
    
    返回所有可用的内容生成类型，包括视频、图像、音频、3D模型等，
    以及每种类型的详细信息和功能特性。
    """
    types_info = [
        GenerationTypeInfo(
            type=GenerationType.VIDEO,
            name="视频生成",
            description="使用AI生成高质量视频内容，支持多种风格和分辨率",
            supported_models=["runway-gen3", "pika-labs", "stable-video", "luma-dream-machine"],
            max_duration=60,
            max_resolution="4k",
            requires_credits=10,
            is_available=True,
            features=["text-to-video", "image-to-video", "video-to-video", "motion-control", "camera-control"]
        ),
        GenerationTypeInfo(
            type=GenerationType.IMAGE,
            name="图像生成",
            description="使用AI生成高质量图像，支持多种艺术风格",
            supported_models=["dall-e-3", "midjourney-v6", "stable-diffusion-xl", "flux-pro"],
            max_resolution="4096x4096",
            requires_credits=2,
            is_available=True,
            features=["text-to-image", "image-to-image", "inpainting", "outpainting", "style-transfer"]
        ),
        GenerationTypeInfo(
            type=GenerationType.AUDIO,
            name="音频生成",
            description="生成音乐、音效和语音内容",
            supported_models=["suno-v3", "udio", "stable-audio", "elevenlabs"],
            max_duration=300,
            requires_credits=5,
            is_available=True,
            features=["text-to-music", "text-to-sound", "text-to-speech", "voice-cloning", "audio-variation"]
        ),
        GenerationTypeInfo(
            type=GenerationType.MODEL_3D,
            name="3D模型生成",
            description="生成3D模型和场景",
            supported_models=["meshy-ai", "tripo3d", "csm-ai", "rodin"],
            requires_credits=15,
            is_available=True,
            features=["text-to-3d", "image-to-3d", "texture-generation", "auto-rigging", "format-conversion"]
        ),
        GenerationTypeInfo(
            type=GenerationType.TEXT,
            name="文本生成",
            description="生成创意文本内容",
            supported_models=["gpt-4", "claude-3", "gemini-pro"],
            requires_credits=1,
            is_available=True,
            features=["creative-writing", "story-generation", "poetry", "script-writing"]
        ),
        GenerationTypeInfo(
            type=GenerationType.ANIMATION,
            name="动画生成",
            description="生成2D/3D动画内容",
            supported_models=["animated-drawings", "runway-motion"],
            max_duration=30,
            requires_credits=12,
            is_available=False,
            features=["sketch-to-animation", "motion-capture", "character-animation"]
        )
    ]
    
    return BaseResponse(
        success=True,
        message="获取生成类型成功",
        data=types_info
    )


# =============================================================================
# 视频生成端点
# =============================================================================

@router.post(
    "/video",
    response_model=BaseResponse[VideoGenerationResponse],
    status_code=status.HTTP_202_ACCEPTED,
    summary="生成视频",
    description="提交视频生成任务，支持文本到视频的生成"
)
async def generate_video(
    request: VideoGenerationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> BaseResponse[VideoGenerationResponse]:
    """
    生成视频
    
    根据提供的提示词和参数生成视频内容。任务提交后会返回任务ID，
    可用于后续查询生成状态和获取结果。
    
    - **prompt**: 视频描述提示词，越详细效果越好
    - **duration**: 视频时长，1-60秒
    - **resolution**: 输出分辨率
    - **style**: 视频风格
    - **seed**: 随机种子，用于可重复生成
    """
    task_id = str(uuid.uuid4())
    
    # 计算所需积分
    credits = request.duration * 2  # 每秒2积分
    
    # 创建任务记录
    task_data = {
        "id": task_id,
        "type": GenerationType.VIDEO,
        "status": GenerationStatus.QUEUED,
        "prompt": request.prompt,
        "progress": 0.0,
        "estimated_time": request.duration * 10,  # 每秒视频约需10秒生成
        "result_url": None,
        "thumbnail_url": None,
        "created_at": datetime.utcnow(),
        "completed_at": None,
        "credits_used": credits,
        "metadata": {
            "duration": request.duration,
            "resolution": request.resolution,
            "fps": request.fps,
            "style": request.style,
            "seed": request.seed,
            "user_id": current_user.get("id")
        }
    }
    
    _generation_tasks[task_id] = task_data
    
    logger.info(f"视频生成任务已创建: {task_id}, 用户: {current_user.get('id')}")
    
    response = VideoGenerationResponse(
        id=task_id,
        status=GenerationStatus.QUEUED,
        prompt=request.prompt,
        progress=0.0,
        estimated_time=request.duration * 10,
        credits_used=credits
    )
    
    return BaseResponse(
        success=True,
        message="视频生成任务已提交",
        data=response
    )


@router.get(
    "/video/{task_id}",
    response_model=BaseResponse[VideoGenerationResponse],
    summary="获取视频生成状态",
    description="查询指定视频生成任务的当前状态和进度"
)
async def get_video_status(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> BaseResponse[VideoGenerationResponse]:
    """
    获取视频生成状态
    
    查询指定任务ID的视频生成任务状态，包括进度、预计完成时间等信息。
    """
    task = _generation_tasks.get(task_id)
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="任务不存在"
        )
    
    # 模拟进度更新
    if task["status"] == GenerationStatus.QUEUED:
        task["status"] = GenerationStatus.PROCESSING
        task["progress"] = 5.0
    elif task["status"] == GenerationStatus.PROCESSING:
        task["progress"] = min(task["progress"] + 10.0, 95.0)
        if task["progress"] >= 95.0:
            task["status"] = GenerationStatus.COMPLETED
            task["completed_at"] = datetime.utcnow()
            task["result_url"] = f"https://cdn.example.com/videos/{task_id}.mp4"
            task["thumbnail_url"] = f"https://cdn.example.com/thumbnails/{task_id}.jpg"
    
    response = VideoGenerationResponse(
        id=task["id"],
        status=task["status"],
        prompt=task["prompt"],
        progress=task["progress"],
        estimated_time=task["estimated_time"],
        result_url=task["result_url"],
        thumbnail_url=task["thumbnail_url"],
        created_at=task["created_at"],
        completed_at=task["completed_at"],
        credits_used=task["credits_used"],
        metadata=task["metadata"]
    )
    
    return BaseResponse(
        success=True,
        message="获取状态成功",
        data=response
    )


@router.delete(
    "/video/{task_id}",
    response_model=BaseResponse[Dict[str, Any]],
    summary="取消视频生成",
    description="取消正在进行的视频生成任务"
)
async def cancel_video_generation(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> BaseResponse[Dict[str, Any]]:
    """
    取消视频生成
    
    取消指定ID的视频生成任务。只能取消正在排队或处理中的任务。
    """
    task = _generation_tasks.get(task_id)
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="任务不存在"
        )
    
    if task["status"] not in [GenerationStatus.PENDING, GenerationStatus.QUEUED, GenerationStatus.PROCESSING]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="只能取消正在进行的任务"
        )
    
    task["status"] = GenerationStatus.CANCELLED
    task["completed_at"] = datetime.utcnow()
    
    logger.info(f"视频生成任务已取消: {task_id}")
    
    return BaseResponse(
        success=True,
        message="任务已取消",
        data={"task_id": task_id, "status": GenerationStatus.CANCELLED}
    )


# =============================================================================
# 图像生成端点
# =============================================================================

@router.post(
    "/image",
    response_model=BaseResponse[ImageGenerationResponse],
    status_code=status.HTTP_202_ACCEPTED,
    summary="生成图像",
    description="提交图像生成任务，支持文本到图像的生成"
)
async def generate_image(
    request: ImageGenerationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> BaseResponse[ImageGenerationResponse]:
    """
    生成图像
    
    根据提供的提示词和参数生成图像内容。支持批量生成多张图像。
    
    - **prompt**: 图像描述提示词
    - **width/height**: 图像尺寸
    - **style**: 图像艺术风格
    - **num_images**: 生成图像数量(1-10)
    - **guidance_scale**: 引导强度，控制提示词遵循程度
    """
    task_id = str(uuid.uuid4())
    
    # 计算所需积分
    credits = request.num_images * 2
    
    # 创建任务记录
    task_data = {
        "id": task_id,
        "type": GenerationType.IMAGE,
        "status": GenerationStatus.QUEUED,
        "prompt": request.prompt,
        "progress": 0.0,
        "estimated_time": request.num_images * 15,
        "result_urls": [],
        "created_at": datetime.utcnow(),
        "completed_at": None,
        "credits_used": credits,
        "metadata": {
            "width": request.width,
            "height": request.height,
            "style": request.style,
            "num_images": request.num_images,
            "seed": request.seed,
            "user_id": current_user.get("id")
        }
    }
    
    _generation_tasks[task_id] = task_data
    
    logger.info(f"图像生成任务已创建: {task_id}, 用户: {current_user.get('id')}")
    
    response = ImageGenerationResponse(
        id=task_id,
        status=GenerationStatus.QUEUED,
        prompt=request.prompt,
        progress=0.0,
        estimated_time=request.num_images * 15,
        credits_used=credits
    )
    
    return BaseResponse(
        success=True,
        message="图像生成任务已提交",
        data=response
    )


@router.get(
    "/image/{task_id}",
    response_model=BaseResponse[ImageGenerationResponse],
    summary="获取图像生成状态",
    description="查询指定图像生成任务的当前状态和进度"
)
async def get_image_status(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> BaseResponse[ImageGenerationResponse]:
    """
    获取图像生成状态
    
    查询指定任务ID的图像生成任务状态。
    """
    task = _generation_tasks.get(task_id)
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="任务不存在"
        )
    
    # 模拟进度更新
    if task["status"] == GenerationStatus.QUEUED:
        task["status"] = GenerationStatus.PROCESSING
        task["progress"] = 10.0
    elif task["status"] == GenerationStatus.PROCESSING:
        task["progress"] = min(task["progress"] + 20.0, 95.0)
        if task["progress"] >= 95.0:
            task["status"] = GenerationStatus.COMPLETED
            task["completed_at"] = datetime.utcnow()
            num_images = task["metadata"].get("num_images", 1)
            task["result_urls"] = [
                f"https://cdn.example.com/images/{task_id}_{i}.png"
                for i in range(num_images)
            ]
    
    response = ImageGenerationResponse(
        id=task["id"],
        status=task["status"],
        prompt=task["prompt"],
        progress=task["progress"],
        estimated_time=task["estimated_time"],
        result_urls=task.get("result_urls", []),
        created_at=task["created_at"],
        completed_at=task["completed_at"],
        credits_used=task["credits_used"],
        metadata=task["metadata"]
    )
    
    return BaseResponse(
        success=True,
        message="获取状态成功",
        data=response
    )


@router.delete(
    "/image/{task_id}",
    response_model=BaseResponse[Dict[str, Any]],
    summary="取消图像生成",
    description="取消正在进行的图像生成任务"
)
async def cancel_image_generation(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> BaseResponse[Dict[str, Any]]:
    """
    取消图像生成
    
    取消指定ID的图像生成任务。
    """
    task = _generation_tasks.get(task_id)
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="任务不存在"
        )
    
    if task["status"] not in [GenerationStatus.PENDING, GenerationStatus.QUEUED, GenerationStatus.PROCESSING]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="只能取消正在进行的任务"
        )
    
    task["status"] = GenerationStatus.CANCELLED
    task["completed_at"] = datetime.utcnow()
    
    logger.info(f"图像生成任务已取消: {task_id}")
    
    return BaseResponse(
        success=True,
        message="任务已取消",
        data={"task_id": task_id, "status": GenerationStatus.CANCELLED}
    )


# =============================================================================
# 音频生成端点
# =============================================================================

@router.post(
    "/audio",
    response_model=BaseResponse[AudioGenerationResponse],
    status_code=status.HTTP_202_ACCEPTED,
    summary="生成音频",
    description="提交音频生成任务，支持音乐、音效和语音生成"
)
async def generate_audio(
    request: AudioGenerationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> BaseResponse[AudioGenerationResponse]:
    """
    生成音频
    
    根据提供的提示词和参数生成音频内容。
    
    - **prompt**: 音频描述提示词
    - **audio_type**: 音频类型(音乐/音效/语音)
    - **duration**: 音频时长(秒)
    - **tempo**: 节拍(BPM)
    - **genre**: 音乐风格
    """
    task_id = str(uuid.uuid4())
    
    # 计算所需积分
    credits = max(request.duration // 10, 1) * 3
    
    # 创建任务记录
    task_data = {
        "id": task_id,
        "type": GenerationType.AUDIO,
        "status": GenerationStatus.QUEUED,
        "prompt": request.prompt,
        "audio_type": request.audio_type,
        "progress": 0.0,
        "estimated_time": request.duration * 2,
        "result_url": None,
        "waveform_url": None,
        "duration": request.duration,
        "created_at": datetime.utcnow(),
        "completed_at": None,
        "credits_used": credits,
        "metadata": {
            "audio_type": request.audio_type,
            "duration": request.duration,
            "tempo": request.tempo,
            "key": request.key,
            "genre": request.genre,
            "mood": request.mood,
            "seed": request.seed,
            "user_id": current_user.get("id")
        }
    }
    
    _generation_tasks[task_id] = task_data
    
    logger.info(f"音频生成任务已创建: {task_id}, 用户: {current_user.get('id')}")
    
    response = AudioGenerationResponse(
        id=task_id,
        status=GenerationStatus.QUEUED,
        prompt=request.prompt,
        audio_type=request.audio_type,
        progress=0.0,
        estimated_time=request.duration * 2,
        duration=request.duration,
        credits_used=credits
    )
    
    return BaseResponse(
        success=True,
        message="音频生成任务已提交",
        data=response
    )


@router.get(
    "/audio/{task_id}",
    response_model=BaseResponse[AudioGenerationResponse],
    summary="获取音频生成状态",
    description="查询指定音频生成任务的当前状态和进度"
)
async def get_audio_status(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> BaseResponse[AudioGenerationResponse]:
    """
    获取音频生成状态
    
    查询指定任务ID的音频生成任务状态。
    """
    task = _generation_tasks.get(task_id)
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="任务不存在"
        )
    
    # 模拟进度更新
    if task["status"] == GenerationStatus.QUEUED:
        task["status"] = GenerationStatus.PROCESSING
        task["progress"] = 5.0
    elif task["status"] == GenerationStatus.PROCESSING:
        task["progress"] = min(task["progress"] + 15.0, 95.0)
        if task["progress"] >= 95.0:
            task["status"] = GenerationStatus.COMPLETED
            task["completed_at"] = datetime.utcnow()
            task["result_url"] = f"https://cdn.example.com/audio/{task_id}.mp3"
            task["waveform_url"] = f"https://cdn.example.com/waveforms/{task_id}.png"
    
    response = AudioGenerationResponse(
        id=task["id"],
        status=task["status"],
        prompt=task["prompt"],
        audio_type=task["audio_type"],
        progress=task["progress"],
        estimated_time=task["estimated_time"],
        result_url=task.get("result_url"),
        waveform_url=task.get("waveform_url"),
        duration=task["duration"],
        created_at=task["created_at"],
        completed_at=task["completed_at"],
        credits_used=task["credits_used"],
        metadata=task["metadata"]
    )
    
    return BaseResponse(
        success=True,
        message="获取状态成功",
        data=response
    )


@router.delete(
    "/audio/{task_id}",
    response_model=BaseResponse[Dict[str, Any]],
    summary="取消音频生成",
    description="取消正在进行的音频生成任务"
)
async def cancel_audio_generation(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> BaseResponse[Dict[str, Any]]:
    """
    取消音频生成
    
    取消指定ID的音频生成任务。
    """
    task = _generation_tasks.get(task_id)
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="任务不存在"
        )
    
    if task["status"] not in [GenerationStatus.PENDING, GenerationStatus.QUEUED, GenerationStatus.PROCESSING]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="只能取消正在进行的任务"
        )
    
    task["status"] = GenerationStatus.CANCELLED
    task["completed_at"] = datetime.utcnow()
    
    logger.info(f"音频生成任务已取消: {task_id}")
    
    return BaseResponse(
        success=True,
        message="任务已取消",
        data={"task_id": task_id, "status": GenerationStatus.CANCELLED}
    )


# =============================================================================
# 3D模型生成端点
# =============================================================================

@router.post(
    "/3d",
    response_model=BaseResponse[Model3DGenerationResponse],
    status_code=status.HTTP_202_ACCEPTED,
    summary="生成3D模型",
    description="提交3D模型生成任务，支持文本到3D的生成"
)
async def generate_3d_model(
    request: Model3DGenerationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> BaseResponse[Model3DGenerationResponse]:
    """
    生成3D模型
    
    根据提供的提示词和参数生成3D模型。
    
    - **prompt**: 模型描述提示词
    - **format**: 输出格式(obj/fbx/gltf/glb等)
    - **quality**: 模型质量级别
    - **texture_resolution**: 纹理分辨率
    - **auto_rig**: 是否自动绑定骨骼
    """
    task_id = str(uuid.uuid4())
    
    # 计算所需积分
    quality_multiplier = {"low": 1, "medium": 1.5, "high": 2, "ultra": 3}
    credits = int(15 * quality_multiplier.get(request.quality.value, 1))
    
    # 创建任务记录
    task_data = {
        "id": task_id,
        "type": GenerationType.MODEL_3D,
        "status": GenerationStatus.QUEUED,
        "prompt": request.prompt,
        "format": request.format,
        "progress": 0.0,
        "estimated_time": 180,
        "result_url": None,
        "preview_url": None,
        "file_size": None,
        "created_at": datetime.utcnow(),
        "completed_at": None,
        "credits_used": credits,
        "metadata": {
            "format": request.format,
            "quality": request.quality,
            "texture_resolution": request.texture_resolution,
            "poly_count": request.poly_count,
            "auto_rig": request.auto_rig,
            "generate_textures": request.generate_textures,
            "seed": request.seed,
            "user_id": current_user.get("id")
        }
    }
    
    _generation_tasks[task_id] = task_data
    
    logger.info(f"3D模型生成任务已创建: {task_id}, 用户: {current_user.get('id')}")
    
    response = Model3DGenerationResponse(
        id=task_id,
        status=GenerationStatus.QUEUED,
        prompt=request.prompt,
        format=request.format,
        progress=0.0,
        estimated_time=180,
        credits_used=credits
    )
    
    return BaseResponse(
        success=True,
        message="3D模型生成任务已提交",
        data=response
    )


@router.get(
    "/3d/{task_id}",
    response_model=BaseResponse[Model3DGenerationResponse],
    summary="获取3D生成状态",
    description="查询指定3D模型生成任务的当前状态和进度"
)
async def get_3d_status(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> BaseResponse[Model3DGenerationResponse]:
    """
    获取3D生成状态
    
    查询指定任务ID的3D模型生成任务状态。
    """
    task = _generation_tasks.get(task_id)
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="任务不存在"
        )
    
    # 模拟进度更新
    if task["status"] == GenerationStatus.QUEUED:
        task["status"] = GenerationStatus.PROCESSING
        task["progress"] = 5.0
    elif task["status"] == GenerationStatus.PROCESSING:
        task["progress"] = min(task["progress"] + 8.0, 95.0)
        if task["progress"] >= 95.0:
            task["status"] = GenerationStatus.COMPLETED
            task["completed_at"] = datetime.utcnow()
            task["result_url"] = f"https://cdn.example.com/3d/{task_id}.{task['format']}"
            task["preview_url"] = f"https://cdn.example.com/previews/{task_id}.png"
            task["file_size"] = 1024 * 1024 * 5  # 5MB
    
    response = Model3DGenerationResponse(
        id=task["id"],
        status=task["status"],
        prompt=task["prompt"],
        format=task["format"],
        progress=task["progress"],
        estimated_time=task["estimated_time"],
        result_url=task.get("result_url"),
        preview_url=task.get("preview_url"),
        file_size=task.get("file_size"),
        created_at=task["created_at"],
        completed_at=task["completed_at"],
        credits_used=task["credits_used"],
        metadata=task["metadata"]
    )
    
    return BaseResponse(
        success=True,
        message="获取状态成功",
        data=response
    )


# =============================================================================
# 历史记录和画廊端点
# =============================================================================

@router.get(
    "/history",
    response_model=PaginatedResponse[GenerationHistoryItem],
    summary="获取生成历史",
    description="获取当前用户的生成历史记录"
)
async def get_generation_history(
    type: Optional[GenerationType] = Query(None, description="筛选类型"),
    status: Optional[GenerationStatus] = Query(None, description="筛选状态"),
    start_date: Optional[datetime] = Query(None, description="开始日期"),
    end_date: Optional[datetime] = Query(None, description="结束日期"),
    is_favorite: Optional[bool] = Query(None, description="是否收藏"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PaginatedResponse[GenerationHistoryItem]:
    """
    获取生成历史
    
    获取当前用户的所有生成任务历史记录，支持多种筛选条件。
    """
    user_id = current_user.get("id")
    
    # 筛选用户的任务
    user_tasks = [
        task for task in _generation_tasks.values()
        if task["metadata"].get("user_id") == user_id
    ]
    
    # 应用筛选
    if type:
        user_tasks = [t for t in user_tasks if t["type"] == type]
    if status:
        user_tasks = [t for t in user_tasks if t["status"] == status]
    if is_favorite is not None:
        user_tasks = [t for t in user_tasks if t.get("is_favorite", False) == is_favorite]
    
    # 按时间倒序
    user_tasks.sort(key=lambda x: x["created_at"], reverse=True)
    
    # 分页
    total = len(user_tasks)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_tasks = user_tasks[start_idx:end_idx]
    
    items = []
    for task in paginated_tasks:
        thumbnail = None
        result_url = None
        
        if task["type"] == GenerationType.VIDEO:
            thumbnail = task.get("thumbnail_url")
            result_url = task.get("result_url")
        elif task["type"] == GenerationType.IMAGE:
            result_urls = task.get("result_urls", [])
            thumbnail = result_urls[0] if result_urls else None
            result_url = result_urls[0] if result_urls else None
        elif task["type"] == GenerationType.AUDIO:
            thumbnail = task.get("waveform_url")
            result_url = task.get("result_url")
        elif task["type"] == GenerationType.MODEL_3D:
            thumbnail = task.get("preview_url")
            result_url = task.get("result_url")
        
        items.append(GenerationHistoryItem(
            id=task["id"],
            type=task["type"],
            status=task["status"],
            prompt=task["prompt"],
            thumbnail_url=thumbnail,
            result_url=result_url,
            created_at=task["created_at"],
            credits_used=task["credits_used"],
            is_favorite=task.get("is_favorite", False),
            tags=task.get("tags", [])
        ))
    
    return PaginatedResponse(
        success=True,
        message="获取历史记录成功",
        data=items,
        pagination={
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size
        }
    )


@router.get(
    "/gallery",
    response_model=PaginatedResponse[GalleryItem],
    summary="获取作品画廊",
    description="浏览社区公开分享的作品画廊"
)
async def get_gallery(
    type: Optional[GenerationType] = Query(None, description="作品类型"),
    sort_by: str = Query("created_at", description="排序字段"),
    tag: Optional[str] = Query(None, description="标签筛选"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=50, description="每页数量"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PaginatedResponse[GalleryItem]:
    """
    获取作品画廊
    
    浏览社区用户公开分享的作品，支持按类型、标签筛选和排序。
    """
    # 模拟画廊数据
    gallery_items = list(_gallery_items.values())
    
    # 应用筛选
    if type:
        gallery_items = [item for item in gallery_items if item["type"] == type]
    if tag:
        gallery_items = [
            item for item in gallery_items
            if tag in item.get("tags", [])
        ]
    
    # 排序
    if sort_by == "likes":
        gallery_items.sort(key=lambda x: x.get("likes", 0), reverse=True)
    elif sort_by == "views":
        gallery_items.sort(key=lambda x: x.get("views", 0), reverse=True)
    else:
        gallery_items.sort(key=lambda x: x["created_at"], reverse=True)
    
    # 分页
    total = len(gallery_items)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_items = gallery_items[start_idx:end_idx]
    
    items = [
        GalleryItem(
            id=item["id"],
            type=item["type"],
            title=item.get("title"),
            prompt=item["prompt"],
            thumbnail_url=item["thumbnail_url"],
            result_url=item["result_url"],
            author_id=item["author_id"],
            author_name=item["author_name"],
            created_at=item["created_at"],
            likes=item.get("likes", 0),
            views=item.get("views", 0),
            visibility=item.get("visibility", ShareVisibility.PUBLIC),
            tags=item.get("tags", [])
        )
        for item in paginated_items
    ]
    
    return PaginatedResponse(
        success=True,
        message="获取画廊成功",
        data=items,
        pagination={
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size
        }
    )


# =============================================================================
# 下载、分享、删除端点
# =============================================================================

@router.post(
    "/{task_id}/download",
    response_model=BaseResponse[DownloadResponse],
    summary="下载生成结果",
    description="获取生成结果的下载链接"
)
async def download_generation(
    task_id: str,
    request: DownloadRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> BaseResponse[DownloadResponse]:
    """
    下载生成结果
    
    获取指定生成任务的下载链接，支持格式转换和质量选择。
    """
    task = _generation_tasks.get(task_id)
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="任务不存在"
        )
    
    if task["status"] != GenerationStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="任务尚未完成"
        )
    
    # 生成下载链接
    download_url = f"https://cdn.example.com/download/{task_id}?format={request.format or 'original'}"
    
    response = DownloadResponse(
        download_url=download_url,
        expires_at=datetime.utcnow() + timedelta(hours=24),
        file_name=f"generation_{task_id}.{request.format or 'zip'}",
        file_size=task.get("file_size", 1024 * 1024),
        format=request.format or "original"
    )
    
    return BaseResponse(
        success=True,
        message="下载链接已生成",
        data=response
    )


@router.post(
    "/{task_id}/share",
    response_model=BaseResponse[ShareResponse],
    summary="分享作品",
    description="将生成作品分享到社区画廊"
)
async def share_generation(
    task_id: str,
    request: ShareRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> BaseResponse[ShareResponse]:
    """
    分享作品
    
    将生成作品分享到社区画廊，可设置可见性和权限。
    """
    task = _generation_tasks.get(task_id)
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="任务不存在"
        )
    
    if task["status"] != GenerationStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="任务尚未完成，无法分享"
        )
    
    share_id = str(uuid.uuid4())
    
    # 创建画廊项
    gallery_item = {
        "id": share_id,
        "type": task["type"],
        "title": request.title or f"作品 {task_id[:8]}",
        "prompt": task["prompt"],
        "thumbnail_url": task.get("thumbnail_url") or task.get("preview_url") or "",
        "result_url": task.get("result_url") or (task.get("result_urls", [None])[0]),
        "author_id": current_user.get("id"),
        "author_name": current_user.get("name", "Anonymous"),
        "created_at": datetime.utcnow(),
        "likes": 0,
        "views": 0,
        "visibility": request.visibility,
        "tags": request.tags,
        "allow_download": request.allow_download,
        "allow_remix": request.allow_remix
    }
    
    _gallery_items[share_id] = gallery_item
    
    logger.info(f"作品已分享: {share_id}, 用户: {current_user.get('id')}")
    
    response = ShareResponse(
        share_id=share_id,
        share_url=f"https://gallery.example.com/works/{share_id}",
        visibility=request.visibility,
        created_at=gallery_item["created_at"]
    )
    
    return BaseResponse(
        success=True,
        message="作品分享成功",
        data=response
    )


@router.post(
    "/{task_id}/delete",
    response_model=BaseResponse[Dict[str, Any]],
    summary="删除作品",
    description="删除生成任务和关联的作品文件"
)
async def delete_generation(
    task_id: str,
    current_user: Dict[str, Any] = Depends(require_permissions(["generation:delete"]))
) -> BaseResponse[Dict[str, Any]]:
    """
    删除作品
    
    永久删除生成任务及其关联文件。此操作不可恢复。
    """
    task = _generation_tasks.get(task_id)
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="任务不存在"
        )
    
    # 检查权限
    if task["metadata"].get("user_id") != current_user.get("id"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权删除此作品"
        )
    
    # 删除任务
    del _generation_tasks[task_id]
    
    # 删除关联的画廊项
    gallery_ids = [
        gid for gid, item in _gallery_items.items()
        if item.get("task_id") == task_id
    ]
    for gid in gallery_ids:
        del _gallery_items[gid]
    
    logger.info(f"作品已删除: {task_id}, 用户: {current_user.get('id')}")
    
    return BaseResponse(
        success=True,
        message="作品已删除",
        data={"task_id": task_id, "deleted_at": datetime.utcnow().isoformat()}
    )


# =============================================================================
# 提示词模板端点
# =============================================================================

@router.get(
    "/templates",
    response_model=PaginatedResponse[PromptTemplate],
    summary="获取提示词模板",
    description="获取系统预设的提示词模板列表"
)
async def get_prompt_templates(
    type: Optional[GenerationType] = Query(None, description="模板类型"),
    tag: Optional[str] = Query(None, description="标签筛选"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=50, description="每页数量"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PaginatedResponse[PromptTemplate]:
    """
    获取提示词模板
    
    获取系统预设的提示词模板，帮助用户快速创建高质量生成内容。
    """
    templates = list(_prompt_templates.values())
    
    # 应用筛选
    if type:
        templates = [t for t in templates if t["type"] == type]
    if tag:
        templates = [t for t in templates if tag in t.get("tags", [])]
    if search:
        search_lower = search.lower()
        templates = [
            t for t in templates
            if search_lower in t["name"].lower()
            or search_lower in t["description"].lower()
        ]
    
    # 按使用次数排序
    templates.sort(key=lambda x: x.get("popularity", 0), reverse=True)
    
    # 分页
    total = len(templates)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_templates = templates[start_idx:end_idx]
    
    items = [
        PromptTemplate(
            id=t["id"],
            name=t["name"],
            description=t["description"],
            type=t["type"],
            template=t["template"],
            variables=t.get("variables", []),
            example_values=t.get("example_values", {}),
            tags=t.get("tags", []),
            popularity=t.get("popularity", 0),
            created_by=t.get("created_by")
        )
        for t in paginated_templates
    ]
    
    return PaginatedResponse(
        success=True,
        message="获取模板成功",
        data=items,
        pagination={
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size
        }
    )


# =============================================================================
# 批量生成端点
# =============================================================================

@router.post(
    "/batch",
    response_model=BaseResponse[BatchGenerationResponse],
    status_code=status.HTTP_202_ACCEPTED,
    summary="批量生成",
    description="提交批量生成任务，一次生成多个内容"
)
async def batch_generate(
    request: BatchGenerationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> BaseResponse[BatchGenerationResponse]:
    """
    批量生成
    
    一次性提交多个生成任务，支持批量生成视频、图像、音频或3D模型。
    
    - **type**: 生成内容类型
    - **prompts**: 提示词列表，每个提示词生成一个作品
    - **common_params**: 所有任务共用的参数
    - **priority**: 任务优先级(1-10)
    """
    batch_id = str(uuid.uuid4())
    task_ids = []
    
    # 创建子任务
    for prompt in request.prompts:
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        
        task_data = {
            "id": task_id,
            "type": request.type,
            "status": GenerationStatus.QUEUED,
            "prompt": prompt,
            "progress": 0.0,
            "batch_id": batch_id,
            "created_at": datetime.utcnow(),
            "metadata": {
                "user_id": current_user.get("id"),
                **request.common_params
            }
        }
        _generation_tasks[task_id] = task_data
    
    # 创建批次记录
    batch_data = {
        "id": batch_id,
        "status": BatchStatus.PENDING,
        "total_tasks": len(request.prompts),
        "completed_tasks": 0,
        "failed_tasks": 0,
        "task_ids": task_ids,
        "created_at": datetime.utcnow(),
        "estimated_completion": datetime.utcnow() + timedelta(minutes=len(request.prompts) * 2)
    }
    _batch_tasks[batch_id] = batch_data
    
    logger.info(f"批量生成任务已创建: {batch_id}, 任务数: {len(task_ids)}, 用户: {current_user.get('id')}")
    
    response = BatchGenerationResponse(
        batch_id=batch_id,
        status=BatchStatus.PENDING,
        total_tasks=len(request.prompts),
        task_ids=task_ids,
        estimated_completion=batch_data["estimated_completion"]
    )
    
    return BaseResponse(
        success=True,
        message=f"批量生成任务已提交，共{len(task_ids)}个任务",
        data=response
    )


# =============================================================================
# WebSocket实时进度端点
# =============================================================================

@router.websocket("/ws/generation/{task_id}")
async def generation_progress_websocket(
    websocket: WebSocket,
    task_id: str
):
    """
    WebSocket实时生成进度
    
    建立WebSocket连接以实时接收生成任务的进度更新。
    连接后会定期收到进度、状态和预计完成时间等信息。
    
    消息格式:
        {
            "task_id": "uuid",
            "status": "processing",
            "progress": 45.5,
            "current_step": "rendering",
            "estimated_time_remaining": 120,
            "message": "正在渲染帧 45/100"
        }
    """
    await websocket.accept()
    
    # 注册连接
    if task_id not in _websocket_connections:
        _websocket_connections[task_id] = []
    _websocket_connections[task_id].append(websocket)
    
    try:
        # 发送初始状态
        task = _generation_tasks.get(task_id)
        if task:
            progress_msg = GenerationProgress(
                task_id=task_id,
                status=task["status"],
                progress=task["progress"],
                current_step="initializing",
                estimated_time_remaining=task.get("estimated_time"),
                message="连接成功，开始接收进度更新"
            )
            await websocket.send_json(progress_msg.dict())
        
        # 保持连接并发送进度更新
        while True:
            task = _generation_tasks.get(task_id)
            if not task:
                await websocket.send_json({
                    "error": "任务不存在",
                    "task_id": task_id
                })
                break
            
            # 模拟进度更新
            if task["status"] == GenerationStatus.PROCESSING:
                progress_msg = GenerationProgress(
                    task_id=task_id,
                    status=task["status"],
                    progress=task["progress"],
                    current_step="generating",
                    estimated_time_remaining=int((100 - task["progress"]) * 2),
                    message=f"生成中... {task['progress']:.1f}%"
                )
                await websocket.send_json(progress_msg.dict())
            elif task["status"] in [GenerationStatus.COMPLETED, GenerationStatus.FAILED, GenerationStatus.CANCELLED]:
                progress_msg = GenerationProgress(
                    task_id=task_id,
                    status=task["status"],
                    progress=100.0 if task["status"] == GenerationStatus.COMPLETED else task["progress"],
                    current_step="finished",
                    estimated_time_remaining=0,
                    message="生成完成" if task["status"] == GenerationStatus.COMPLETED else f"生成{task['status']}"
                )
                await websocket.send_json(progress_msg.dict())
                break
            
            await asyncio.sleep(2)  # 每2秒更新一次
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket断开连接: {task_id}")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
    finally:
        # 移除连接
        if task_id in _websocket_connections:
            if websocket in _websocket_connections[task_id]:
                _websocket_connections[task_id].remove(websocket)
            if not _websocket_connections[task_id]:
                del _websocket_connections[task_id]


# =============================================================================
# 辅助函数
# =============================================================================

async def _notify_progress_update(task_id: str, progress: GenerationProgress):
    """
    通知所有连接的客户端进度更新
    
    Args:
        task_id: 任务ID
        progress: 进度信息
    """
    if task_id in _websocket_connections:
        disconnected = []
        for ws in _websocket_connections[task_id]:
            try:
                await ws.send_json(progress.dict())
            except Exception:
                disconnected.append(ws)
        
        # 清理断开的连接
        for ws in disconnected:
            _websocket_connections[task_id].remove(ws)


def _calculate_credits(generation_type: GenerationType, params: Dict[str, Any]) -> int:
    """
    计算生成所需积分
    
    Args:
        generation_type: 生成类型
        params: 生成参数
        
    Returns:
        所需积分数量
    """
    base_credits = {
        GenerationType.VIDEO: 10,
        GenerationType.IMAGE: 2,
        GenerationType.AUDIO: 5,
        GenerationType.MODEL_3D: 15,
        GenerationType.TEXT: 1,
        GenerationType.ANIMATION: 12
    }
    
    credits = base_credits.get(generation_type, 1)
    
    # 根据参数调整积分
    if generation_type == GenerationType.VIDEO:
        duration = params.get("duration", 5)
        credits = duration * 2
    elif generation_type == GenerationType.IMAGE:
        num_images = params.get("num_images", 1)
        credits = num_images * 2
    
    return credits


# =============================================================================
# 错误处理
# =============================================================================

# 注意：APIRouter 不支持 exception_handler，需要在主应用中注册
# @router.exception_handler(ValueError)
# async def value_error_handler(request, exc):
#     """处理值错误"""
#     return BaseResponse(
#         success=False,
#         message=str(exc),
#     ).dict()


# 模块导出
__all__ = [
    "router",
    "GenerationType",
    "GenerationStatus",
    "VideoGenerationRequest",
    "VideoGenerationResponse",
    "ImageGenerationRequest",
    "ImageGenerationResponse",
    "AudioGenerationRequest",
    "AudioGenerationResponse",
    "Model3DGenerationRequest",
    "Model3DGenerationResponse",
    "GenerationHistoryItem",
    "GalleryItem",
    "PromptTemplate",
    "BatchGenerationRequest",
    "BatchGenerationResponse",
]