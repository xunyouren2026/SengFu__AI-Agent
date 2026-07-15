"""
Generation Manager
统一生成任务管理器

管理所有生成任务的生命周期、队列、进度追踪
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class GenerationStatus(Enum):
    """生成任务状态"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


class GenerationType(Enum):
    """生成类型"""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    TTS = "tts"
    THREED = "3d"


@dataclass
class GenerationTask:
    """生成任务"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: GenerationType = GenerationType.IMAGE
    status: GenerationStatus = GenerationStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    prompt: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "status": self.status.value,
            "priority": self.priority.value,
            "prompt": self.prompt,
            "config": self.config,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress": self.progress,
            "metadata": self.metadata,
        }


@dataclass
class GenerationProgress:
    """生成进度"""
    task_id: str
    progress: float  # 0.0 - 1.0
    stage: str = ""
    message: str = ""
    eta: Optional[float] = None  # 预计剩余时间（秒）
    current_step: int = 0
    total_steps: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "progress": self.progress,
            "stage": self.stage,
            "message": self.message,
            "eta": self.eta,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
        }


class GenerationManager:
    """
    统一生成任务管理器
    
    功能：
    - 任务队列管理
    - 优先级调度
    - 进度追踪
    - 结果缓存
    - 并发控制
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 任务存储
        self._tasks: Dict[str, GenerationTask] = {}
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        
        # 引擎实例
        self._engines: Dict[str, Any] = {}
        
        # 进度回调
        self._progress_callbacks: List[Callable] = []
        
        # 并发控制
        self._max_concurrent = self.config.get("max_concurrent", 3)
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        
        # 结果缓存
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = self.config.get("cache_ttl", 3600)  # 1小时
        
        # 工作线程
        self._workers: List[asyncio.Task] = []
        self._running = False
        
    async def start(self):
        """启动管理器"""
        if self._running:
            return
        
        self._running = True
        
        # 启动工作线程
        for i in range(self._max_concurrent):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)
        
        logger.info(f"Generation manager started with {self._max_concurrent} workers")
    
    async def stop(self):
        """停止管理器"""
        self._running = False
        
        # 取消所有工作线程
        for worker in self._workers:
            worker.cancel()
        
        self._workers.clear()
        
        logger.info("Generation manager stopped")
    
    async def submit(
        self,
        type: GenerationType,
        prompt: str,
        config: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        提交生成任务
        
        Args:
            type: 生成类型
            prompt: 提示词
            config: 生成配置
            priority: 优先级
            metadata: 元数据
        
        Returns:
            任务ID
        """
        task = GenerationTask(
            type=type,
            prompt=prompt,
            config=config or {},
            priority=priority,
            metadata=metadata or {},
        )
        
        # 检查缓存
        cache_key = self._get_cache_key(type, prompt, config)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if time.time() - cached["timestamp"] < self._cache_ttl:
                task.status = GenerationStatus.COMPLETED
                task.result = cached["result"]
                task.completed_at = time.time()
                self._tasks[task.id] = task
                logger.info(f"Task {task.id} served from cache")
                return task.id
        
        # 存储任务
        self._tasks[task.id] = task
        
        # 加入队列
        await self._queue.put((priority.value, time.time(), task))
        
        logger.info(f"Task {task.id} submitted: {type.value} - {prompt[:50]}...")
        
        return task.id
    
    async def get_task(self, task_id: str) -> Optional[GenerationTask]:
        """获取任务"""
        return self._tasks.get(task_id)
    
    async def get_status(self, task_id: str) -> Optional[GenerationStatus]:
        """获取任务状态"""
        task = self._tasks.get(task_id)
        return task.status if task else None
    
    async def get_progress(self, task_id: str) -> Optional[GenerationProgress]:
        """获取任务进度"""
        task = self._tasks.get(task_id)
        if not task:
            return None
        
        return GenerationProgress(
            task_id=task_id,
            progress=task.progress,
            stage=task.metadata.get("stage", ""),
            message=task.metadata.get("message", ""),
            current_step=task.metadata.get("current_step", 0),
            total_steps=task.metadata.get("total_steps", 0),
        )
    
    async def cancel(self, task_id: str) -> bool:
        """取消任务"""
        task = self._tasks.get(task_id)
        if not task:
            return False
        
        if task.status in [GenerationStatus.COMPLETED, GenerationStatus.FAILED, GenerationStatus.CANCELLED]:
            return False
        
        task.status = GenerationStatus.CANCELLED
        task.completed_at = time.time()
        
        logger.info(f"Task {task_id} cancelled")
        
        return True
    
    async def list_tasks(
        self,
        status: Optional[GenerationStatus] = None,
        type: Optional[GenerationType] = None,
        limit: int = 100,
    ) -> List[GenerationTask]:
        """列出任务"""
        tasks = list(self._tasks.values())
        
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        if type:
            tasks = [t for t in tasks if t.type == type]
        
        # 按创建时间排序
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        
        return tasks[:limit]
    
    def add_progress_callback(self, callback: Callable):
        """添加进度回调"""
        self._progress_callbacks.append(callback)
    
    def remove_progress_callback(self, callback: Callable):
        """移除进度回调"""
        if callback in self._progress_callbacks:
            self._progress_callbacks.remove(callback)
    
    async def _worker(self, worker_id: int):
        """工作线程"""
        logger.info(f"Worker {worker_id} started")
        
        while self._running:
            try:
                # 从队列获取任务
                priority, timestamp, task = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0
                )
                
                # 检查任务是否已取消
                if task.status == GenerationStatus.CANCELLED:
                    continue
                
                # 执行任务
                async with self._semaphore:
                    await self._execute_task(task)
                    
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _execute_task(self, task: GenerationTask):
        """执行生成任务"""
        task.status = GenerationStatus.RUNNING
        task.started_at = time.time()
        
        logger.info(f"Executing task {task.id}: {task.type.value}")
        
        try:
            # 根据类型选择引擎
            if task.type == GenerationType.IMAGE:
                result = await self._generate_image(task)
            elif task.type == GenerationType.VIDEO:
                result = await self._generate_video(task)
            elif task.type == GenerationType.AUDIO:
                result = await self._generate_audio(task)
            elif task.type == GenerationType.TTS:
                result = await self._generate_tts(task)
            elif task.type == GenerationType.THREED:
                result = await self._generate_3d(task)
            else:
                raise ValueError(f"Unknown generation type: {task.type}")
            
            task.result = result
            task.status = GenerationStatus.COMPLETED
            task.progress = 1.0
            task.completed_at = time.time()
            
            # 缓存结果
            cache_key = self._get_cache_key(task.type, task.prompt, task.config)
            self._cache[cache_key] = {
                "result": result,
                "timestamp": time.time(),
            }
            
            logger.info(f"Task {task.id} completed in {task.completed_at - task.started_at:.2f}s")
            
        except Exception as e:
            task.status = GenerationStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            
            logger.error(f"Task {task.id} failed: {e}")
    
    async def _generate_image(self, task: GenerationTask) -> Dict[str, Any]:
        """生成图像"""
        from .image_engine import create_image_engine, ImageConfig
        
        engine_type = task.config.get("engine", "diffusers")
        engine = create_image_engine(engine_type, self.config.get("image", {}))
        
        config = ImageConfig(
            prompt=task.prompt,
            negative_prompt=task.config.get("negative_prompt", ""),
            width=task.config.get("width", 512),
            height=task.config.get("height", 512),
            num_inference_steps=task.config.get("num_inference_steps", 30),
            guidance_scale=task.config.get("guidance_scale", 7.5),
            num_images=task.config.get("num_images", 1),
            seed=task.config.get("seed"),
        )
        
        # 更新进度
        task.metadata["stage"] = "generating"
        task.metadata["total_steps"] = config.num_inference_steps
        
        result = await engine.generate(config)
        
        return result.to_dict()
    
    async def _generate_video(self, task: GenerationTask) -> Dict[str, Any]:
        """生成视频"""
        from .video_engine import create_video_engine, VideoConfig
        
        engine_type = task.config.get("engine", "cogvideox")
        engine = create_video_engine(engine_type, self.config.get("video", {}))
        
        config = VideoConfig(
            prompt=task.prompt,
            negative_prompt=task.config.get("negative_prompt", ""),
            width=task.config.get("width", 512),
            height=task.config.get("height", 512),
            num_frames=task.config.get("num_frames", 16),
            fps=task.config.get("fps", 8),
            num_inference_steps=task.config.get("num_inference_steps", 50),
            guidance_scale=task.config.get("guidance_scale", 7.5),
            seed=task.config.get("seed"),
        )
        
        task.metadata["stage"] = "generating"
        
        result = await engine.generate(config)
        
        return result.to_dict()
    
    async def _generate_audio(self, task: GenerationTask) -> Dict[str, Any]:
        """生成音频"""
        from .audio_engine import create_audio_engine, AudioConfig
        
        engine_type = task.config.get("engine", "musicgen")
        engine = create_audio_engine(engine_type, self.config.get("audio", {}))
        
        config = AudioConfig(
            prompt=task.prompt,
            negative_prompt=task.config.get("negative_prompt", ""),
            duration=task.config.get("duration", 10.0),
            num_inference_steps=task.config.get("num_inference_steps", 50),
            guidance_scale=task.config.get("guidance_scale", 7.5),
            seed=task.config.get("seed"),
        )
        
        task.metadata["stage"] = "generating"
        
        result = await engine.generate(config)
        
        return result.to_dict()
    
    async def _generate_tts(self, task: GenerationTask) -> Dict[str, Any]:
        """生成语音"""
        from .tts_engine import create_tts_engine, VoiceConfig
        
        engine_type = task.config.get("engine", "edge")
        engine = create_tts_engine(engine_type, self.config.get("tts", {}))
        
        voice_config = VoiceConfig(
            voice_id=task.config.get("voice_id", "default"),
            language=task.config.get("language", "zh-CN"),
            rate=task.config.get("rate", 1.0),
            pitch=task.config.get("pitch", 0.0),
        )
        
        task.metadata["stage"] = "synthesizing"
        
        result = await engine.synthesize(task.prompt, voice_config)
        
        return result.to_dict()
    
    async def _generate_3d(self, task: GenerationTask) -> Dict[str, Any]:
        """生成3D模型"""
        from .threed_engine import create_3d_engine, ThreeDConfig
        
        engine_type = task.config.get("engine", "tripoSR")
        engine = create_3d_engine(engine_type, self.config.get("3d", {}))
        
        config = ThreeDConfig(
            prompt=task.prompt,
            num_inference_steps=task.config.get("num_inference_steps", 64),
            guidance_scale=task.config.get("guidance_scale", 7.5),
            seed=task.config.get("seed"),
        )
        
        task.metadata["stage"] = "generating"
        
        result = await engine.generate(config)
        
        return result.to_dict()
    
    def _get_cache_key(self, type: GenerationType, prompt: str, config: Optional[Dict]) -> str:
        """生成缓存键"""
        key_data = f"{type.value}|{prompt}|{json.dumps(config or {}, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        tasks = list(self._tasks.values())
        
        return {
            "total_tasks": len(tasks),
            "pending": len([t for t in tasks if t.status == GenerationStatus.PENDING]),
            "running": len([t for t in tasks if t.status == GenerationStatus.RUNNING]),
            "completed": len([t for t in tasks if t.status == GenerationStatus.COMPLETED]),
            "failed": len([t for t in tasks if t.status == GenerationStatus.FAILED]),
            "cancelled": len([t for t in tasks if t.status == GenerationStatus.CANCELLED]),
            "queue_size": self._queue.qsize(),
            "cache_size": len(self._cache),
        }
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        logger.info("Generation cache cleared")
    
    async def cleanup_old_tasks(self, max_age: float = 86400):
        """清理旧任务"""
        now = time.time()
        to_remove = []
        
        for task_id, task in self._tasks.items():
            if task.completed_at and (now - task.completed_at) > max_age:
                to_remove.append(task_id)
        
        for task_id in to_remove:
            del self._tasks[task_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old tasks")


# 全局实例
_generation_manager: Optional[GenerationManager] = None


def get_generation_manager() -> GenerationManager:
    """获取全局生成管理器"""
    global _generation_manager
    if _generation_manager is None:
        _generation_manager = GenerationManager()
    return _generation_manager


async def init_generation_manager(config: Optional[Dict[str, Any]] = None):
    """初始化全局生成管理器"""
    global _generation_manager
    _generation_manager = GenerationManager(config)
    await _generation_manager.start()
    return _generation_manager
