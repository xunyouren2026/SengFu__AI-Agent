"""
Image Generation Engine
真实图像生成引擎

支持多种后端：
- Diffusers: HuggingFace Diffusers本地模型
- DALL-E: OpenAI DALL-E API
- Stable Diffusion: Stability AI API
- Midjourney: 第三方API
- ComfyUI: 本地ComfyUI服务
"""

import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np

logger = logging.getLogger(__name__)


class ImageFormat(Enum):
    """图像格式"""
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"
    BMP = "bmp"


class ImageSize(Enum):
    """图像尺寸预设"""
    SQUARE_512 = "512x512"
    SQUARE_1024 = "1024x1024"
    PORTRAIT_512_768 = "512x768"
    PORTRAIT_768_1024 = "768x1024"
    LANDSCAPE_768_512 = "768x512"
    LANDSCAPE_1024_768 = "1024x768"
    LANDSCAPE_1920_1080 = "1920x1080"


@dataclass
class ImageConfig:
    """图像生成配置"""
    prompt: str = ""
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    num_images: int = 1
    seed: Optional[int] = None
    scheduler: str = "euler"  # euler, ddim, dpm++, etc.
    clip_skip: int = 0
    sampler: str = "euler_a"
    
    # 高级选项
    enable_hr: bool = False  # 高分辨率修复
    hr_scale: float = 2.0
    hr_steps: int = 20
    denoising_strength: float = 0.7
    
    # 风格选项
    style: Optional[str] = None
    quality: str = "standard"  # standard, hd
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "width": self.width,
            "height": self.height,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "num_images": self.num_images,
            "seed": self.seed,
            "scheduler": self.scheduler,
        }


@dataclass
class ControlNetConfig:
    """ControlNet配置"""
    enabled: bool = False
    model: str = "lllyasviel/sd-controlnet-canny"
    image: Optional[str] = None  # base64或路径
    conditioning_scale: float = 1.0
    control_mode: str = "canny"  # canny, depth, pose, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model": self.model,
            "conditioning_scale": self.conditioning_scale,
            "control_mode": self.control_mode,
        }


@dataclass
class IPAdapterConfig:
    """IP-Adapter配置"""
    enabled: bool = False
    model: str = "h94/IP-Adapter"
    image: Optional[str] = None  # 参考图像
    scale: float = 0.8
    num_tokens: int = 4
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model": self.model,
            "scale": self.scale,
            "num_tokens": self.num_tokens,
        }


@dataclass
class ImageResult:
    """图像生成结果"""
    success: bool
    images: List[bytes] = field(default_factory=list)
    image_paths: List[str] = field(default_factory=list)
    format: ImageFormat = ImageFormat.PNG
    width: int = 512
    height: int = 512
    seed: Optional[int] = None
    prompt: str = ""
    negative_prompt: str = ""
    model: str = ""
    inference_time: float = 0.0
    nsfw_detected: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "image_paths": self.image_paths,
            "format": self.format.value,
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "model": self.model,
            "inference_time": self.inference_time,
            "nsfw_detected": self.nsfw_detected,
            "error": self.error,
            "metadata": self.metadata,
        }


class ImageGenerationEngine(ABC):
    """图像生成引擎基类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialized = False
        self._model_cache: Dict[str, Any] = {}
        
    @abstractmethod
    async def generate(
        self,
        config: ImageConfig,
        controlnet: Optional[ControlNetConfig] = None,
        ip_adapter: Optional[IPAdapterConfig] = None,
    ) -> ImageResult:
        """生成图像"""
        pass
    
    @abstractmethod
    async def img2img(
        self,
        image: Union[str, bytes],
        config: ImageConfig,
        strength: float = 0.75,
    ) -> ImageResult:
        """图像到图像转换"""
        pass
    
    @abstractmethod
    async def inpaint(
        self,
        image: Union[str, bytes],
        mask: Union[str, bytes],
        config: ImageConfig,
    ) -> ImageResult:
        """图像修复"""
        pass
    
    @abstractmethod
    async def list_models(self) -> List[Dict[str, Any]]:
        """列出可用模型"""
        pass
    
    @abstractmethod
    async def load_model(self, model_id: str) -> bool:
        """加载模型"""
        pass
    
    def _save_image(self, image_data: bytes, format: ImageFormat = ImageFormat.PNG) -> str:
        """保存图像到文件"""
        output_dir = self.config.get("output_dir", tempfile.gettempdir())
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"img_{uuid.uuid4().hex}.{format.value}"
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, "wb") as f:
            f.write(image_data)
        
        return output_path


class DiffusersEngine(ImageGenerationEngine):
    """
    HuggingFace Diffusers引擎
    本地运行Stable Diffusion等模型
    
    特点：
    - 支持多种SD模型
    - 支持ControlNet
    - 支持IP-Adapter
    - 支持LoRA
    - 需要GPU
    """
    
    DEFAULT_MODELS = {
        "sd15": "runwayml/stable-diffusion-v1-5",
        "sd21": "stabilityai/stable-diffusion-2-1",
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
        "sdxl-turbo": "stabilityai/sdxl-turbo",
        "sd3": "stabilityai/stable-diffusion-3-medium",
        "flux-dev": "black-forest-labs/FLUX.1-dev",
        "flux-schnell": "black-forest-labs/FLUX.1-schnell",
    }
    
    CONTROLNET_MODELS = {
        "canny": "lllyasviel/sd-controlnet-canny",
        "depth": "lllyasviel/sd-controlnet-depth",
        "pose": "lllyasviel/sd-controlnet-openpose",
        "scribble": "lllyasviel/sd-controlnet-scribble",
        "seg": "lllyasviel/sd-controlnet-seg",
        "normal": "lllyasviel/sd-controlnet-normal",
        "lineart": "lllyasviel/sd-controlnet-lineart",
        "shuffle": "lllyasviel/sd-controlnet-shuffle",
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._pipeline = None
        self._controlnet = None
        self._current_model = None
        self._device = None
        self._torch_dtype = None
        
    async def _ensure_initialized(self):
        """确保Diffusers已初始化"""
        if self._initialized:
            return
            
        try:
            import torch
            from diffusers import DiffusionPipeline, StableDiffusionPipeline
            
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._torch_dtype = torch.float16 if self._device == "cuda" else torch.float32
            
            # 加载默认模型
            default_model = self.config.get("model", "sd21")
            model_id = self.DEFAULT_MODELS.get(default_model, default_model)
            
            await self.load_model(model_id)
            
            self._initialized = True
            logger.info(f"Diffusers engine initialized on {self._device}")
            
        except ImportError as e:
            raise ImportError(
                f"Required packages not installed: {e}. Install with: pip install diffusers transformers accelerate torch"
            )
    
    async def load_model(self, model_id: str) -> bool:
        """加载模型"""
        try:
            import torch
            from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline
            
            logger.info(f"Loading model: {model_id}")
            
            # 根据模型类型选择Pipeline
            if "xl" in model_id.lower():
                pipeline_class = StableDiffusionXLPipeline
            elif "flux" in model_id.lower():
                pipeline_class = DiffusionPipeline
            else:
                pipeline_class = StableDiffusionPipeline
            
            self._pipeline = pipeline_class.from_pretrained(
                model_id,
                torch_dtype=self._torch_dtype,
                safety_checker=None if self.config.get("disable_safety", False) else "stable_diffusion",
            )
            
            if self._device == "cuda":
                self._pipeline = self._pipeline.to(self._device)
                
                # 启用内存优化
                if hasattr(self._pipeline, 'enable_attention_slicing'):
                    self._pipeline.enable_attention_slicing()
                if hasattr(self._pipeline, 'enable_vae_slicing'):
                    self._pipeline.enable_vae_slicing()
            
            self._current_model = model_id
            logger.info(f"Model loaded successfully: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return False
    
    async def generate(
        self,
        config: ImageConfig,
        controlnet: Optional[ControlNetConfig] = None,
        ip_adapter: Optional[IPAdapterConfig] = None,
    ) -> ImageResult:
        """使用Diffusers生成图像"""
        await self._ensure_initialized()
        
        if self._pipeline is None:
            return ImageResult(
                success=False,
                error="No model loaded",
                prompt=config.prompt,
            )
        
        try:
            import torch
            
            start_time = time.time()
            
            # 设置随机种子
            if config.seed is not None:
                torch.manual_seed(config.seed)
                generator = torch.Generator(device=self._device).manual_seed(config.seed)
            else:
                generator = None
            
            # 生成图像
            output = self._pipeline(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
                width=config.width,
                height=config.height,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                num_images_per_prompt=config.num_images,
                generator=generator,
            )
            
            inference_time = time.time() - start_time
            
            # 处理输出
            images = []
            image_paths = []
            
            for img in output.images:
                # 转换为bytes
                buffer = io.BytesIO()
                img.save(buffer, format=config.format.value.upper() if hasattr(config, 'format') else "PNG")
                image_data = buffer.getvalue()
                images.append(image_data)
                
                # 保存文件
                path = self._save_image(image_data)
                image_paths.append(path)
            
            return ImageResult(
                success=True,
                images=images,
                image_paths=image_paths,
                format=ImageFormat.PNG,
                width=config.width,
                height=config.height,
                seed=config.seed,
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
                model=self._current_model,
                inference_time=inference_time,
                metadata={
                    "engine": "diffusers",
                    "num_inference_steps": config.num_inference_steps,
                    "guidance_scale": config.guidance_scale,
                }
            )
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return ImageResult(
                success=False,
                error=str(e),
                prompt=config.prompt,
            )
    
    async def img2img(
        self,
        image: Union[str, bytes],
        config: ImageConfig,
        strength: float = 0.75,
    ) -> ImageResult:
        """图像到图像转换"""
        await self._ensure_initialized()
        
        try:
            import torch
            from PIL import Image
            
            # 加载输入图像
            if isinstance(image, str):
                init_image = Image.open(image).convert("RGB")
            else:
                init_image = Image.open(io.BytesIO(image)).convert("RGB")
            
            # 调整尺寸
            init_image = init_image.resize((config.width, config.height))
            
            start_time = time.time()
            
            # 设置随机种子
            if config.seed is not None:
                generator = torch.Generator(device=self._device).manual_seed(config.seed)
            else:
                generator = None
            
            # 使用img2img
            output = self._pipeline(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
                image=init_image,
                strength=strength,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                generator=generator,
            )
            
            inference_time = time.time() - start_time
            
            # 处理输出
            images = []
            image_paths = []
            
            for img in output.images:
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                image_data = buffer.getvalue()
                images.append(image_data)
                path = self._save_image(image_data)
                image_paths.append(path)
            
            return ImageResult(
                success=True,
                images=images,
                image_paths=image_paths,
                format=ImageFormat.PNG,
                width=config.width,
                height=config.height,
                seed=config.seed,
                prompt=config.prompt,
                model=self._current_model,
                inference_time=inference_time,
                metadata={"engine": "diffusers", "mode": "img2img"}
            )
            
        except Exception as e:
            logger.error(f"img2img failed: {e}")
            return ImageResult(success=False, error=str(e), prompt=config.prompt)
    
    async def inpaint(
        self,
        image: Union[str, bytes],
        mask: Union[str, bytes],
        config: ImageConfig,
    ) -> ImageResult:
        """图像修复"""
        # TODO: 实现inpainting
        return ImageResult(
            success=False,
            error="Inpainting not implemented yet",
            prompt=config.prompt,
        )
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """列出可用模型"""
        models = []
        for name, model_id in self.DEFAULT_MODELS.items():
            models.append({
                "id": model_id,
                "name": name,
                "type": "diffusers",
                "loaded": model_id == self._current_model,
            })
        return models


class DALLEEngine(ImageGenerationEngine):
    """
    OpenAI DALL-E引擎
    
    特点：
    - 高质量图像
    - 无需本地GPU
    - 支持DALL-E 2和DALL-E 3
    - 需要API密钥
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._api_key = None
        self._client = None
        self._model = "dall-e-3"
        
    async def _ensure_initialized(self):
        """确保OpenAI已初始化"""
        if self._initialized:
            return
            
        try:
            from openai import AsyncOpenAI
            
            self._api_key = self.config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
            
            if not self._api_key:
                raise ValueError("OpenAI API key is required")
            
            self._client = AsyncOpenAI(api_key=self._api_key)
            self._model = self.config.get("model", "dall-e-3")
            
            self._initialized = True
            logger.info(f"DALL-E engine initialized with model: {self._model}")
            
        except ImportError:
            raise ImportError(
                "openai is not installed. Install it with: pip install openai"
            )
    
    async def generate(
        self,
        config: ImageConfig,
        controlnet: Optional[ControlNetConfig] = None,
        ip_adapter: Optional[IPAdapterConfig] = None,
    ) -> ImageResult:
        """使用DALL-E生成图像"""
        await self._ensure_initialized()
        
        try:
            start_time = time.time()
            
            # DALL-E支持的尺寸
            size = f"{config.width}x{config.height}"
            if size not in ["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]:
                size = "1024x1024"  # 默认尺寸
            
            # 调用API
            response = await self._client.images.generate(
                model=self._model,
                prompt=config.prompt,
                size=size,
                quality=config.quality if config.quality in ["standard", "hd"] else "standard",
                n=config.num_images,
            )
            
            inference_time = time.time() - start_time
            
            # 下载图像
            images = []
            image_paths = []
            
            import httpx
            async with httpx.AsyncClient() as client:
                for img_data in response.data:
                    img_response = await client.get(img_data.url)
                    image_data = img_response.content
                    images.append(image_data)
                    path = self._save_image(image_data)
                    image_paths.append(path)
            
            return ImageResult(
                success=True,
                images=images,
                image_paths=image_paths,
                format=ImageFormat.PNG,
                prompt=config.prompt,
                model=self._model,
                inference_time=inference_time,
                metadata={
                    "engine": "dalle",
                    "revised_prompt": response.data[0].revised_prompt if hasattr(response.data[0], 'revised_prompt') else None,
                }
            )
            
        except Exception as e:
            logger.error(f"DALL-E generation failed: {e}")
            return ImageResult(
                success=False,
                error=str(e),
                prompt=config.prompt,
            )
    
    async def img2img(
        self,
        image: Union[str, bytes],
        config: ImageConfig,
        strength: float = 0.75,
    ) -> ImageResult:
        """DALL-E不支持img2img"""
        return ImageResult(
            success=False,
            error="DALL-E does not support img2img",
            prompt=config.prompt,
        )
    
    async def inpaint(
        self,
        image: Union[str, bytes],
        mask: Union[str, bytes],
        config: ImageConfig,
    ) -> ImageResult:
        """DALL-E不支持inpainting"""
        return ImageResult(
            success=False,
            error="DALL-E does not support inpainting",
            prompt=config.prompt,
        )
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """列出可用模型"""
        return [
            {"id": "dall-e-2", "name": "DALL-E 2", "type": "api"},
            {"id": "dall-e-3", "name": "DALL-E 3", "type": "api"},
        ]
    
    async def load_model(self, model_id: str) -> bool:
        """设置模型"""
        if model_id in ["dall-e-2", "dall-e-3"]:
            self._model = model_id
            return True
        return False


class StableDiffusionAPIEngine(ImageGenerationEngine):
    """
    Stability AI API引擎
    
    特点：
    - 高质量SD模型
    - 无需本地GPU
    - 支持多种模型
    - 需要API密钥
    """
    
    API_URL = "https://api.stability.ai"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._api_key = None
        self._model = "stable-diffusion-xl-1024-v1-0"
        
    async def _ensure_initialized(self):
        """确保Stability AI已初始化"""
        if self._initialized:
            return
            
        self._api_key = self.config.get("stability_api_key") or os.environ.get("STABILITY_API_KEY")
        
        if not self._api_key:
            raise ValueError("Stability AI API key is required")
        
        self._initialized = True
        logger.info("Stability AI engine initialized")
    
    async def generate(
        self,
        config: ImageConfig,
        controlnet: Optional[ControlNetConfig] = None,
        ip_adapter: Optional[IPAdapterConfig] = None,
    ) -> ImageResult:
        """使用Stability AI生成图像"""
        await self._ensure_initialized()
        
        try:
            import httpx
            
            start_time = time.time()
            
            url = f"{self.API_URL}/v1/generation/{self._model}/text-to-image"
            
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            
            data = {
                "text_prompts": [
                    {"text": config.prompt, "weight": 1.0},
                ],
                "cfg_scale": config.guidance_scale,
                "height": config.height,
                "width": config.width,
                "samples": config.num_images,
                "steps": config.num_inference_steps,
            }
            
            if config.negative_prompt:
                data["text_prompts"].append({"text": config.negative_prompt, "weight": -1.0})
            
            if config.seed is not None:
                data["seed"] = config.seed
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=data, headers=headers)
            
            inference_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                images = []
                image_paths = []
                
                for artifact in result.get("artifacts", []):
                    image_data = base64.b64decode(artifact["base64"])
                    images.append(image_data)
                    path = self._save_image(image_data)
                    image_paths.append(path)
                
                return ImageResult(
                    success=True,
                    images=images,
                    image_paths=image_paths,
                    format=ImageFormat.PNG,
                    width=config.width,
                    height=config.height,
                    seed=result.get("seed", config.seed),
                    prompt=config.prompt,
                    model=self._model,
                    inference_time=inference_time,
                    metadata={"engine": "stability_api"}
                )
            else:
                return ImageResult(
                    success=False,
                    error=f"API error: {response.status_code} - {response.text}",
                    prompt=config.prompt,
                )
                
        except Exception as e:
            logger.error(f"Stability AI generation failed: {e}")
            return ImageResult(
                success=False,
                error=str(e),
                prompt=config.prompt,
            )
    
    async def img2img(
        self,
        image: Union[str, bytes],
        config: ImageConfig,
        strength: float = 0.75,
    ) -> ImageResult:
        """图像到图像转换"""
        await self._ensure_initialized()
        
        try:
            import httpx
            
            # 准备图像
            if isinstance(image, bytes):
                init_image = base64.b64encode(image).decode()
            else:
                with open(image, "rb") as f:
                    init_image = base64.b64encode(f.read()).decode()
            
            url = f"{self.API_URL}/v1/generation/{self._model}/image-to-image"
            
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            
            data = {
                "init_image": init_image,
                "init_image_mode": "IMAGE_STRENGTH",
                "image_strength": 1 - strength,
                "text_prompts": [{"text": config.prompt, "weight": 1.0}],
                "cfg_scale": config.guidance_scale,
                "samples": config.num_images,
                "steps": config.num_inference_steps,
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                
                images = []
                image_paths = []
                
                for artifact in result.get("artifacts", []):
                    image_data = base64.b64decode(artifact["base64"])
                    images.append(image_data)
                    path = self._save_image(image_data)
                    image_paths.append(path)
                
                return ImageResult(
                    success=True,
                    images=images,
                    image_paths=image_paths,
                    format=ImageFormat.PNG,
                    prompt=config.prompt,
                    model=self._model,
                    metadata={"engine": "stability_api", "mode": "img2img"}
                )
            else:
                return ImageResult(
                    success=False,
                    error=f"API error: {response.status_code}",
                    prompt=config.prompt,
                )
                
        except Exception as e:
            logger.error(f"img2img failed: {e}")
            return ImageResult(success=False, error=str(e), prompt=config.prompt)
    
    async def inpaint(
        self,
        image: Union[str, bytes],
        mask: Union[str, bytes],
        config: ImageConfig,
    ) -> ImageResult:
        """图像修复"""
        # TODO: 实现inpainting
        return ImageResult(
            success=False,
            error="Inpainting not implemented",
            prompt=config.prompt,
        )
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """列出可用模型"""
        return [
            {"id": "stable-diffusion-xl-1024-v1-0", "name": "SDXL 1.0", "type": "api"},
            {"id": "stable-diffusion-v1-6", "name": "SD 1.6", "type": "api"},
            {"id": "stable-diffusion-512-v2-1", "name": "SD 2.1 512", "type": "api"},
            {"id": "stable-diffusion-768-v2-1", "name": "SD 2.1 768", "type": "api"},
        ]
    
    async def load_model(self, model_id: str) -> bool:
        """设置模型"""
        self._model = model_id
        return True


class ComfyUIEngine(ImageGenerationEngine):
    """
    ComfyUI引擎
    通过API调用本地ComfyUI服务
    
    特点：
    - 支持复杂工作流
    - 支持所有SD扩展
    - 灵活的节点系统
    - 需要运行ComfyUI服务
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._base_url = config.get("comfyui_url", "http://127.0.0.1:8188")
        self._client_id = str(uuid.uuid4())
        
    async def _ensure_initialized(self):
        """确保ComfyUI服务可用"""
        if self._initialized:
            return
            
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self._base_url}/system_stats")
            
            if response.status_code == 200:
                self._initialized = True
                logger.info(f"ComfyUI engine connected to {self._base_url}")
            else:
                raise ConnectionError(f"ComfyUI not responding: {response.status_code}")
                
        except Exception as e:
            raise ConnectionError(f"Failed to connect to ComfyUI: {e}")
    
    async def generate(
        self,
        config: ImageConfig,
        controlnet: Optional[ControlNetConfig] = None,
        ip_adapter: Optional[IPAdapterConfig] = None,
    ) -> ImageResult:
        """使用ComfyUI生成图像"""
        await self._ensure_initialized()
        
        try:
            import httpx
            
            # 构建工作流
            workflow = self._build_workflow(config, controlnet, ip_adapter)
            
            # 提交任务
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(
                    f"{self._base_url}/prompt",
                    json={"prompt": workflow, "client_id": self._client_id}
                )
            
            if response.status_code != 200:
                return ImageResult(
                    success=False,
                    error=f"ComfyUI error: {response.text}",
                    prompt=config.prompt,
                )
            
            result = response.json()
            prompt_id = result.get("prompt_id")
            
            # 等待完成
            images = await self._wait_for_result(prompt_id)
            
            image_paths = [self._save_image(img) for img in images]
            
            return ImageResult(
                success=True,
                images=images,
                image_paths=image_paths,
                format=ImageFormat.PNG,
                prompt=config.prompt,
                model="comfyui",
                metadata={"engine": "comfyui", "prompt_id": prompt_id}
            )
            
        except Exception as e:
            logger.error(f"ComfyUI generation failed: {e}")
            return ImageResult(
                success=False,
                error=str(e),
                prompt=config.prompt,
            )
    
    def _build_workflow(
        self,
        config: ImageConfig,
        controlnet: Optional[ControlNetConfig] = None,
        ip_adapter: Optional[IPAdapterConfig] = None,
    ) -> Dict[str, Any]:
        """构建ComfyUI工作流"""
        # 基础工作流模板
        workflow = {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": config.seed or 0,
                    "steps": config.num_inference_steps,
                    "cfg": config.guidance_scale,
                    "sampler_name": config.sampler,
                    "scheduler": config.scheduler,
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0],
                }
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": self.config.get("checkpoint", "v2-1_768-ema-pruned.ckpt"),
                }
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": config.width,
                    "height": config.height,
                    "batch_size": config.num_images,
                }
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": config.prompt,
                    "clip": ["4", 1],
                }
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": config.negative_prompt,
                    "clip": ["4", 1],
                }
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2],
                }
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "ComfyUI",
                    "images": ["8", 0],
                }
            }
        }
        
        return workflow
    
    async def _wait_for_result(self, prompt_id: str) -> List[bytes]:
        """等待生成结果"""
        import httpx
        
        max_wait = 300  # 最多等待5分钟
        start_time = time.time()
        
        async with httpx.AsyncClient() as client:
            while time.time() - start_time < max_wait:
                # 检查历史
                response = await client.get(f"{self._base_url}/history/{prompt_id}")
                
                if response.status_code == 200:
                    history = response.json()
                    if prompt_id in history:
                        outputs = history[prompt_id].get("outputs", {})
                        
                        images = []
                        for node_id, node_output in outputs.items():
                            if "images" in node_output:
                                for img_info in node_output["images"]:
                                    img_response = await client.get(
                                        f"{self._base_url}/view",
                                        params={
                                            "filename": img_info["filename"],
                                            "subfolder": img_info.get("subfolder", ""),
                                            "type": img_info.get("type", "output"),
                                        }
                                    )
                                    if img_response.status_code == 200:
                                        images.append(img_response.content)
                        
                        if images:
                            return images
                
                await asyncio.sleep(1)
        
        return []
    
    async def img2img(
        self,
        image: Union[str, bytes],
        config: ImageConfig,
        strength: float = 0.75,
    ) -> ImageResult:
        """图像到图像转换"""
        # TODO: 实现ComfyUI img2img
        return ImageResult(
            success=False,
            error="ComfyUI img2img not implemented",
            prompt=config.prompt,
        )
    
    async def inpaint(
        self,
        image: Union[str, bytes],
        mask: Union[str, bytes],
        config: ImageConfig,
    ) -> ImageResult:
        """图像修复"""
        # TODO: 实现ComfyUI inpainting
        return ImageResult(
            success=False,
            error="ComfyUI inpainting not implemented",
            prompt=config.prompt,
        )
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """列出可用模型"""
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self._base_url}/object_info/CheckpointLoaderSimple")
            
            if response.status_code == 200:
                data = response.json()
                models = []
                for ckpt in data.get("CheckpointLoaderSimple", {}).get("input", {}).get("required", {}).get("ckpt_name", []):
                    models.append({"id": ckpt, "name": ckpt, "type": "checkpoint"})
                return models
        except Exception as e:
            logger.error(f"Failed to list ComfyUI models: {e}")
        
        return []
    
    async def load_model(self, model_id: str) -> bool:
        """ComfyUI通过工作流加载模型"""
        self.config["checkpoint"] = model_id
        return True


# 工厂函数
def create_image_engine(
    engine_type: str = "diffusers",
    config: Optional[Dict[str, Any]] = None
) -> ImageGenerationEngine:
    """
    创建图像生成引擎
    
    Args:
        engine_type: 引擎类型 (diffusers, dalle, stability, comfyui)
        config: 配置参数
    
    Returns:
        ImageGenerationEngine实例
    """
    engines = {
        "diffusers": DiffusersEngine,
        "dalle": DALLEEngine,
        "stability": StableDiffusionAPIEngine,
        "comfyui": ComfyUIEngine,
    }
    
    if engine_type not in engines:
        raise ValueError(f"Unknown image engine type: {engine_type}")
    
    return engines[engine_type](config)
