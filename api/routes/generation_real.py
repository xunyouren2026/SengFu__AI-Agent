"""
Generation API Routes - Real Implementation
真实生成功能API路由

连接API层与core/generation模块
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# 导入核心生成模块
from core.generation import (
    # TTS
    create_tts_engine,
    VoiceConfig,
    TTSResult,
    AudioFormat as TTSAudioFormat,
    # Image
    create_image_engine,
    ImageConfig,
    ImageResult,
    ControlNetConfig,
    IPAdapterConfig,
    # Video
    create_video_engine,
    VideoConfig,
    VideoResult,
    # Audio
    create_audio_engine,
    AudioConfig,
    AudioResult,
    # 3D
    create_3d_engine,
    ThreeDConfig,
    ThreeDResult,
    # Manager
    GenerationManager,
    GenerationType,
    TaskPriority,
    get_generation_manager,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/generation", tags=["Generation"])


# ============== 请求模型 ==============

class TTSRequest(BaseModel):
    """TTS请求"""
    text: str
    voice_id: str = "default"
    language: str = "zh-CN"
    rate: float = 1.0
    pitch: float = 0.0
    engine: str = "edge"
    output_format: str = "mp3"


class ImageGenerationRequest(BaseModel):
    """图像生成请求"""
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    num_images: int = 1
    seed: Optional[int] = None
    engine: str = "diffusers"
    model: Optional[str] = None


class VideoGenerationRequest(BaseModel):
    """视频生成请求"""
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    num_frames: int = 16
    fps: int = 8
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    engine: str = "cogvideox"


class AudioGenerationRequest(BaseModel):
    """音频生成请求"""
    prompt: str
    negative_prompt: str = ""
    duration: float = 10.0
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    engine: str = "musicgen"


class ThreeDGenerationRequest(BaseModel):
    """3D生成请求"""
    prompt: str = ""
    num_inference_steps: int = 64
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    engine: str = "tripoSR"
    remove_background: bool = True


# ============== TTS API ==============

@router.post("/tts", summary="Text-to-Speech")
async def text_to_speech(request: TTSRequest):
    """
    文字转语音
    
    支持多种引擎：
    - edge: 微软Edge TTS（免费，无需GPU）
    - bark: Suno AI Bark（本地，需GPU）
    - coqui: Coqui TTS（本地）
    - azure: Azure Cognitive Services
    - elevenlabs: ElevenLabs API
    """
    try:
        # 创建TTS引擎
        engine = create_tts_engine(request.engine, {
            "output_dir": "/tmp/tts_output"
        })
        
        # 配置语音
        voice_config = VoiceConfig(
            voice_id=request.voice_id,
            language=request.language,
            rate=request.rate,
            pitch=request.pitch,
        )
        
        # 合成语音
        result = await engine.synthesize(
            request.text,
            voice_config,
            TTSAudioFormat(request.output_format)
        )
        
        if result.success:
            return {
                "success": True,
                "audio_path": result.audio_path,
                "duration": result.duration,
                "format": result.format.value,
                "voice_id": result.voice_id,
            }
        else:
            raise HTTPException(status_code=500, detail=result.error)
            
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tts/voices", summary="List available voices")
async def list_tts_voices(language: Optional[str] = None, engine: str = "edge"):
    """列出可用的语音"""
    try:
        tts_engine = create_tts_engine(engine)
        voices = await tts_engine.list_voices(language)
        return {"voices": voices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Image Generation API ==============

@router.post("/image", summary="Generate image")
async def generate_image(request: ImageGenerationRequest):
    """
    图像生成
    
    支持多种引擎：
    - diffusers: HuggingFace Diffusers（本地，需GPU）
    - dalle: OpenAI DALL-E（API）
    - stability: Stability AI（API）
    - comfyui: ComfyUI（本地服务）
    """
    try:
        # 创建图像引擎
        config = {"output_dir": "/tmp/image_output"}
        if request.model:
            config["model"] = request.model
            
        engine = create_image_engine(request.engine, config)
        
        # 配置生成参数
        image_config = ImageConfig(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            num_images=request.num_images,
            seed=request.seed,
        )
        
        # 生成图像
        result = await engine.generate(image_config)
        
        if result.success:
            return {
                "success": True,
                "images": result.image_paths,
                "seed": result.seed,
                "inference_time": result.inference_time,
                "model": result.model,
            }
        else:
            raise HTTPException(status_code=500, detail=result.error)
            
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/image/img2img", summary="Image-to-Image")
async def image_to_image(
    file: UploadFile = File(...),
    prompt: str = "",
    strength: float = 0.75,
    negative_prompt: str = "",
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    engine: str = "diffusers",
):
    """图像到图像转换"""
    try:
        # 读取上传的图像
        image_data = await file.read()
        
        # 创建引擎
        engine_instance = create_image_engine(engine, {"output_dir": "/tmp/image_output"})
        
        # 配置
        config = ImageConfig(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        
        # 生成
        result = await engine_instance.img2img(image_data, config, strength)
        
        if result.success:
            return {
                "success": True,
                "images": result.image_paths,
            }
        else:
            raise HTTPException(status_code=500, detail=result.error)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/image/models", summary="List available image models")
async def list_image_models(engine: str = "diffusers"):
    """列出可用的图像模型"""
    try:
        image_engine = create_image_engine(engine)
        models = await image_engine.list_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Video Generation API ==============

@router.post("/video", summary="Generate video")
async def generate_video(request: VideoGenerationRequest):
    """
    视频生成
    
    支持多种引擎：
    - cogvideox: 智谱AI CogVideoX（本地，需GPU）
    - svd: Stable Video Diffusion（本地，需GPU）
    - animatediff: AnimateDiff（本地，需GPU）
    - runway: Runway Gen-2（API）
    """
    try:
        engine = create_video_engine(request.engine, {"output_dir": "/tmp/video_output"})
        
        config = VideoConfig(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_frames=request.num_frames,
            fps=request.fps,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
        )
        
        result = await engine.generate(config)
        
        if result.success:
            return {
                "success": True,
                "video_path": result.video_path,
                "duration": result.duration,
                "num_frames": result.num_frames,
                "fps": result.fps,
                "inference_time": result.inference_time,
            }
        else:
            raise HTTPException(status_code=500, detail=result.error)
            
    except Exception as e:
        logger.error(f"Video generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/video/img2vid", summary="Image-to-Video")
async def image_to_video(
    file: UploadFile = File(...),
    prompt: str = "",
    num_frames: int = 16,
    fps: int = 8,
    num_inference_steps: int = 50,
    seed: Optional[int] = None,
    engine: str = "svd",
):
    """图像到视频转换"""
    try:
        image_data = await file.read()
        
        engine_instance = create_video_engine(engine, {"output_dir": "/tmp/video_output"})
        
        config = VideoConfig(
            prompt=prompt,
            num_frames=num_frames,
            fps=fps,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )
        
        result = await engine_instance.image_to_video(image_data, config)
        
        if result.success:
            return {
                "success": True,
                "video_path": result.video_path,
            }
        else:
            raise HTTPException(status_code=500, detail=result.error)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Audio Generation API ==============

@router.post("/audio", summary="Generate audio/music")
async def generate_audio(request: AudioGenerationRequest):
    """
    音频/音乐生成
    
    支持多种引擎：
    - musicgen: Meta MusicGen（本地，需GPU）
    - audioldm: AudioLDM（本地，需GPU）
    - riffusion: Riffusion（本地，需GPU）
    """
    try:
        engine = create_audio_engine(request.engine, {"output_dir": "/tmp/audio_output"})
        
        config = AudioConfig(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            duration=request.duration,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
        )
        
        result = await engine.generate(config)
        
        if result.success:
            return {
                "success": True,
                "audio_path": result.audio_path,
                "duration": result.duration,
                "sample_rate": result.sample_rate,
                "inference_time": result.inference_time,
            }
        else:
            raise HTTPException(status_code=500, detail=result.error)
            
    except Exception as e:
        logger.error(f"Audio generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== 3D Generation API ==============

@router.post("/3d", summary="Generate 3D model")
async def generate_3d(request: ThreeDGenerationRequest):
    """
    3D模型生成
    
    支持多种引擎：
    - tripoSR: TripoSR（快速，需GPU）
    - shap-e: OpenAI Shap-E（文本/图像转3D）
    - stable-fast-3d: Stability AI SF3D
    """
    try:
        engine = create_3d_engine(request.engine, {"output_dir": "/tmp/3d_output"})
        
        config = ThreeDConfig(
            prompt=request.prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            remove_background=request.remove_background,
        )
        
        result = await engine.generate(config)
        
        if result.success:
            return {
                "success": True,
                "model_path": result.model_path,
                "format": result.format.value,
                "vertices": result.vertices,
                "faces": result.faces,
                "inference_time": result.inference_time,
            }
        else:
            raise HTTPException(status_code=500, detail=result.error)
            
    except Exception as e:
        logger.error(f"3D generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/3d/img2obj", summary="Image-to-3D")
async def image_to_3d(
    file: UploadFile = File(...),
    prompt: str = "",
    num_inference_steps: int = 64,
    seed: Optional[int] = None,
    engine: str = "tripoSR",
    remove_background: bool = True,
):
    """图像转3D模型"""
    try:
        image_data = await file.read()
        
        engine_instance = create_3d_engine(engine, {"output_dir": "/tmp/3d_output"})
        
        config = ThreeDConfig(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            seed=seed,
            remove_background=remove_background,
        )
        
        result = await engine_instance.image_to_3d(image_data, config)
        
        if result.success:
            return {
                "success": True,
                "model_path": result.model_path,
                "format": result.format.value,
            }
        else:
            raise HTTPException(status_code=500, detail=result.error)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Task Management API ==============

@router.post("/task/submit", summary="Submit generation task")
async def submit_task(
    type: str,
    prompt: str,
    config: Optional[Dict[str, Any]] = None,
    priority: int = 1,
):
    """提交生成任务到队列"""
    try:
        manager = get_generation_manager()
        
        task_id = await manager.submit(
            type=GenerationType(type),
            prompt=prompt,
            config=config or {},
            priority=TaskPriority(priority),
        )
        
        return {
            "success": True,
            "task_id": task_id,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/task/{task_id}", summary="Get task status")
async def get_task_status(task_id: str):
    """获取任务状态"""
    try:
        manager = get_generation_manager()
        task = await manager.get_task(task_id)
        
        if task:
            return task.to_dict()
        else:
            raise HTTPException(status_code=404, detail="Task not found")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/task/{task_id}/progress", summary="Get task progress")
async def get_task_progress(task_id: str):
    """获取任务进度"""
    try:
        manager = get_generation_manager()
        progress = await manager.get_progress(task_id)
        
        if progress:
            return progress.to_dict()
        else:
            raise HTTPException(status_code=404, detail="Task not found")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/task/{task_id}", summary="Cancel task")
async def cancel_task(task_id: str):
    """取消任务"""
    try:
        manager = get_generation_manager()
        success = await manager.cancel(task_id)
        
        return {
            "success": success,
            "task_id": task_id,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks", summary="List tasks")
async def list_tasks(
    status: Optional[str] = None,
    type: Optional[str] = None,
    limit: int = 100,
):
    """列出任务"""
    try:
        manager = get_generation_manager()
        
        status_enum = GenerationStatus(status) if status else None
        type_enum = GenerationType(type) if type else None
        
        tasks = await manager.list_tasks(status_enum, type_enum, limit)
        
        return {
            "tasks": [t.to_dict() for t in tasks],
            "total": len(tasks),
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", summary="Get generation stats")
async def get_stats():
    """获取生成统计"""
    try:
        manager = get_generation_manager()
        stats = await manager.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 导出路由
__all__ = ["router"]
