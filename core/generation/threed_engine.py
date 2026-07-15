"""
3D Generation Engine
真实3D生成引擎

支持多种后端：
- TripoSR: 快速单图转3D
- Shap-E: OpenAI的3D生成模型
- Stable Fast 3D: Stability AI的快速3D生成
- InstantMesh: 快速网格生成
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

logger = logging.getLogger(__name__)


class ThreeDFormat(Enum):
    """3D格式"""
    OBJ = "obj"
    GLB = "glb"
    GLTF = "gltf"
    PLY = "ply"
    STL = "stl"


@dataclass
class ThreeDConfig:
    """3D生成配置"""
    prompt: str = ""
    negative_prompt: str = ""
    image: Optional[str] = None  # 输入图像路径或base64
    num_inference_steps: int = 64
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    num_outputs: int = 1
    
    # 3D特定参数
    texture_resolution: int = 1024
    geometry_resolution: int = 256
    remove_background: bool = True
    
    # Shap-E参数
    model: str = "shape"  # shape, text
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
        }


@dataclass
class ThreeDResult:
    """3D生成结果"""
    success: bool
    model_data: Optional[bytes] = None
    model_path: Optional[str] = None
    format: ThreeDFormat = ThreeDFormat.GLB
    vertices: int = 0
    faces: int = 0
    texture_included: bool = False
    seed: Optional[int] = None
    prompt: str = ""
    model: str = ""
    inference_time: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "model_path": self.model_path,
            "format": self.format.value,
            "vertices": self.vertices,
            "faces": self.faces,
            "texture_included": self.texture_included,
            "seed": self.seed,
            "prompt": self.prompt,
            "model": self.model,
            "inference_time": self.inference_time,
            "error": self.error,
            "metadata": self.metadata,
        }


class ThreeDEngine(ABC):
    """3D生成引擎基类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialized = False
        
    @abstractmethod
    async def generate(self, config: ThreeDConfig) -> ThreeDResult:
        """从文本生成3D模型"""
        pass
    
    @abstractmethod
    async def image_to_3d(
        self,
        image: Union[str, bytes],
        config: ThreeDConfig,
    ) -> ThreeDResult:
        """从图像生成3D模型"""
        pass
    
    @abstractmethod
    async def list_models(self) -> List[Dict[str, Any]]:
        """列出可用模型"""
        pass
    
    def _save_model(self, model_data: bytes, format: ThreeDFormat = ThreeDFormat.GLB) -> str:
        """保存3D模型到文件"""
        output_dir = self.config.get("output_dir", tempfile.gettempdir())
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"3d_{uuid.uuid4().hex}.{format.value}"
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, "wb") as f:
            f.write(model_data)
        
        return output_path


class TripoSREngine(ThreeDEngine):
    """
    TripoSR引擎
    快速单图转3D模型
    
    特点：
    - 极快（几秒内完成）
    - 高质量网格
    - 支持纹理
    - 需要GPU
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._model = None
        self._device = None
        
    async def _ensure_initialized(self):
        """确保TripoSR已初始化"""
        if self._initialized:
            return
            
        try:
            import torch
            from tsr.system import TSR
            
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            
            model_path = self.config.get("model_path", "stabilityai/TripoSR")
            
            logger.info(f"Loading TripoSR model: {model_path}")
            
            self._model = TSR.from_pretrained(
                model_path,
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            
            self._model.to(self._device)
            self._model.renderer.device = self._device
            
            self._initialized = True
            logger.info("TripoSR engine initialized successfully")
            
        except ImportError as e:
            raise ImportError(
                f"TripoSR not installed: {e}. Install with: pip install trimesh einops rembg"
            )
    
    async def generate(self, config: ThreeDConfig) -> ThreeDResult:
        """TripoSR需要输入图像"""
        return ThreeDResult(
            success=False,
            error="TripoSR requires an input image. Use image_to_3d instead.",
            prompt=config.prompt,
        )
    
    async def image_to_3d(
        self,
        image: Union[str, bytes],
        config: ThreeDConfig,
    ) -> ThreeDResult:
        """从图像生成3D模型"""
        await self._ensure_initialized()
        
        try:
            import torch
            import numpy as np
            from PIL import Image
            
            start_time = time.time()
            
            # 加载输入图像
            if isinstance(image, str):
                init_image = Image.open(image).convert("RGB")
            else:
                init_image = Image.open(io.BytesIO(image)).convert("RGB")
            
            # 移除背景（如果需要）
            if config.remove_background:
                try:
                    from rembg import remove
                    init_image = remove(init_image)
                    init_image = init_image.convert("RGB")
                except ImportError:
                    logger.warning("rembg not installed, skipping background removal")
            
            # 预处理图像
            image_array = np.array(init_image)
            
            # 生成3D模型
            with torch.no_grad():
                scenes = self._model([image_array])
            
            inference_time = time.time() - start_time
            
            # 导出为GLB
            scene = scenes[0]
            
            output_dir = self.config.get("output_dir", tempfile.gettempdir())
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"3d_tripo_{uuid.uuid4().hex}.glb")
            
            scene.export(output_path)
            
            with open(output_path, "rb") as f:
                model_data = f.read()
            
            # 获取统计信息
            vertices = len(scene.vertices)
            faces = sum(len(m.faces) for m in scene.mesh_list)
            
            return ThreeDResult(
                success=True,
                model_data=model_data,
                model_path=output_path,
                format=ThreeDFormat.GLB,
                vertices=vertices,
                faces=faces,
                texture_included=True,
                prompt=config.prompt,
                model="tripoSR",
                inference_time=inference_time,
                metadata={"engine": "tripoSR"}
            )
            
        except Exception as e:
            logger.error(f"TripoSR generation failed: {e}")
            return ThreeDResult(success=False, error=str(e), prompt=config.prompt)
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """列出可用模型"""
        return [
            {"id": "stabilityai/TripoSR", "name": "TripoSR", "type": "local"},
        ]


class ShapEEngine(ThreeDEngine):
    """
    Shap-E引擎
    OpenAI的文本/图像转3D模型
    
    特点：
    - 支持文本转3D
    - 支持图像转3D
    - 生成隐式表示
    - 需要GPU
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._xm = None
        self._model = None
        self._device = None
        
    async def _ensure_initialized(self):
        """确保Shap-E已初始化"""
        if self._initialized:
            return
            
        try:
            import torch
            from shap_e.models.download import load_model, load_model_config
            from shap_e.models.transmitter import Transmitter
            
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info("Loading Shap-E model...")
            
            # 加载模型
            self._xm = load_model('transmitter', device=self._device)
            self._model = load_model('text300M', device=self._device)
            
            self._initialized = True
            logger.info("Shap-E engine initialized successfully")
            
        except ImportError as e:
            raise ImportError(
                f"Shap-E not installed: {e}. Install with: pip install git+https://github.com/openai/shap-e.git"
            )
    
    async def generate(self, config: ThreeDConfig) -> ThreeDResult:
        """从文本生成3D模型"""
        await self._ensure_initialized()
        
        try:
            import torch
            from shap_e.diffusion.sample import sample_latents
            from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
            from shap_e.models.download import load_model_config
            
            start_time = time.time()
            
            # 设置随机种子
            if config.seed is not None:
                torch.manual_seed(config.seed)
            
            # 生成隐式表示
            diffusion = diffusion_from_config(load_model_config('base'))
            
            latents = sample_latents(
                batch_size=config.num_outputs,
                model=self._model,
                diffusion=diffusion,
                guidance_scale=config.guidance_scale,
                model_kwargs=dict(texts=[config.prompt] * config.num_outputs),
                progress=True,
                clip_denoised=True,
                use_fp16=True,
                use_karras=True,
                karras_steps=config.num_inference_steps,
                sigma_min=1e-3,
                sigma_max=160,
                s_churn=0,
            )
            
            inference_time = time.time() - start_time
            
            # 导出为GLB
            output_path = await self._export_latent(latents[0], config)
            
            with open(output_path, "rb") as f:
                model_data = f.read()
            
            return ThreeDResult(
                success=True,
                model_data=model_data,
                model_path=output_path,
                format=ThreeDFormat.GLB,
                prompt=config.prompt,
                model="shap-e",
                inference_time=inference_time,
                seed=config.seed,
                metadata={"engine": "shap-e", "mode": "text"}
            )
            
        except Exception as e:
            logger.error(f"Shap-E generation failed: {e}")
            return ThreeDResult(success=False, error=str(e), prompt=config.prompt)
    
    async def image_to_3d(
        self,
        image: Union[str, bytes],
        config: ThreeDConfig,
    ) -> ThreeDResult:
        """从图像生成3D模型"""
        await self._ensure_initialized()
        
        try:
            import torch
            from PIL import Image
            from shap_e.diffusion.sample import sample_latents
            from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
            from shap_e.models.download import load_model_config
            from shap_e.util.image_util import load_image
            
            start_time = time.time()
            
            # 加载输入图像
            if isinstance(image, str):
                init_image = load_image(image)
            else:
                init_image = Image.open(io.BytesIO(image)).convert("RGB")
            
            # 加载图像模型
            image_model = load_model('image300M', device=self._device)
            
            # 设置随机种子
            if config.seed is not None:
                torch.manual_seed(config.seed)
            
            # 生成隐式表示
            diffusion = diffusion_from_config(load_model_config('base'))
            
            latents = sample_latents(
                batch_size=config.num_outputs,
                model=image_model,
                diffusion=diffusion,
                guidance_scale=config.guidance_scale,
                model_kwargs=dict(images=[init_image] * config.num_outputs),
                progress=True,
                clip_denoised=True,
                use_fp16=True,
                use_karras=True,
                karras_steps=config.num_inference_steps,
                sigma_min=1e-3,
                sigma_max=160,
                s_churn=0,
            )
            
            inference_time = time.time() - start_time
            
            # 导出为GLB
            output_path = await self._export_latent(latents[0], config)
            
            with open(output_path, "rb") as f:
                model_data = f.read()
            
            return ThreeDResult(
                success=True,
                model_data=model_data,
                model_path=output_path,
                format=ThreeDFormat.GLB,
                prompt=config.prompt,
                model="shap-e",
                inference_time=inference_time,
                seed=config.seed,
                metadata={"engine": "shap-e", "mode": "image"}
            )
            
        except Exception as e:
            logger.error(f"Shap-E image-to-3D failed: {e}")
            return ThreeDResult(success=False, error=str(e), prompt=config.prompt)
    
    async def _export_latent(self, latent, config: ThreeDConfig) -> str:
        """导出隐式表示为GLB"""
        from shap_e.util.notebooks import decode_latent_mesh
        
        output_dir = self.config.get("output_dir", tempfile.gettempdir())
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"3d_shape_{uuid.uuid4().hex}.glb")
        
        # 解码为网格
        mesh = decode_latent_mesh(self._xm, latent)
        
        # 导出
        mesh.export(output_path)
        
        return output_path
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """列出可用模型"""
        return [
            {"id": "text300M", "name": "Shap-E Text 300M", "type": "local"},
            {"id": "image300M", "name": "Shap-E Image 300M", "type": "local"},
        ]


class StableFast3DEngine(ThreeDEngine):
    """
    Stable Fast 3D引擎
    Stability AI的快速3D生成
    
    特点：
    - 快速生成（约1秒）
    - 高质量纹理
    - 支持PBR材质
    - 需要GPU
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._pipeline = None
        self._device = None
        
    async def _ensure_initialized(self):
        """确保Stable Fast 3D已初始化"""
        if self._initialized:
            return
            
        try:
            import torch
            from sf3d.system import SF3D
            
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            
            model_path = self.config.get("model_path", "stabilityai/stable-fast-3d")
            
            logger.info(f"Loading Stable Fast 3D model: {model_path}")
            
            self._pipeline = SF3D.from_pretrained(
                model_path,
                config_name="config.yaml",
                weight_name="model.safetensors",
            )
            
            self._pipeline.to(self._device)
            
            self._initialized = True
            logger.info("Stable Fast 3D engine initialized successfully")
            
        except ImportError as e:
            raise ImportError(
                f"Stable Fast 3D not installed: {e}"
            )
    
    async def generate(self, config: ThreeDConfig) -> ThreeDResult:
        """Stable Fast 3D需要输入图像"""
        return ThreeDResult(
            success=False,
            error="Stable Fast 3D requires an input image. Use image_to_3d instead.",
            prompt=config.prompt,
        )
    
    async def image_to_3d(
        self,
        image: Union[str, bytes],
        config: ThreeDConfig,
    ) -> ThreeDResult:
        """从图像生成3D模型"""
        await self._ensure_initialized()
        
        try:
            import torch
            from PIL import Image
            
            start_time = time.time()
            
            # 加载输入图像
            if isinstance(image, str):
                init_image = Image.open(image).convert("RGB")
            else:
                init_image = Image.open(io.BytesIO(image)).convert("RGB")
            
            # 生成3D模型
            with torch.no_grad():
                result = self._pipeline(init_image)
            
            inference_time = time.time() - start_time
            
            # 导出为GLB
            output_dir = self.config.get("output_dir", tempfile.gettempdir())
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"3d_sf3d_{uuid.uuid4().hex}.glb")
            
            result.export(output_path)
            
            with open(output_path, "rb") as f:
                model_data = f.read()
            
            return ThreeDResult(
                success=True,
                model_data=model_data,
                model_path=output_path,
                format=ThreeDFormat.GLB,
                prompt=config.prompt,
                model="stable-fast-3d",
                inference_time=inference_time,
                metadata={"engine": "stable_fast_3d"}
            )
            
        except Exception as e:
            logger.error(f"Stable Fast 3D generation failed: {e}")
            return ThreeDResult(success=False, error=str(e), prompt=config.prompt)
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """列出可用模型"""
        return [
            {"id": "stabilityai/stable-fast-3d", "name": "Stable Fast 3D", "type": "local"},
        ]


class InstantMeshEngine(ThreeDEngine):
    """
    InstantMesh引擎
    快速网格生成
    
    特点：
    - 快速生成
    - 高质量网格
    - 支持纹理
    - 需要GPU
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._pipeline = None
        self._device = None
        
    async def _ensure_initialized(self):
        """确保InstantMesh已初始化"""
        if self._initialized:
            return
            
        try:
            import torch
            
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # TODO: 加载InstantMesh模型
            logger.info("InstantMesh engine initialized")
            
            self._initialized = True
            
        except Exception as e:
            raise ImportError(f"InstantMesh not available: {e}")
    
    async def generate(self, config: ThreeDConfig) -> ThreeDResult:
        """InstantMesh需要输入图像"""
        return ThreeDResult(
            success=False,
            error="InstantMesh requires an input image. Use image_to_3d instead.",
            prompt=config.prompt,
        )
    
    async def image_to_3d(
        self,
        image: Union[str, bytes],
        config: ThreeDConfig,
    ) -> ThreeDResult:
        """从图像生成3D模型"""
        await self._ensure_initialized()
        
        # TODO: 实现InstantMesh生成
        return ThreeDResult(
            success=False,
            error="InstantMesh not fully implemented",
            prompt=config.prompt,
        )
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """列出可用模型"""
        return [
            {"id": "instantmesh", "name": "InstantMesh", "type": "local"},
        ]


# 工厂函数
def create_3d_engine(
    engine_type: str = "tripoSR",
    config: Optional[Dict[str, Any]] = None
) -> ThreeDEngine:
    """
    创建3D生成引擎
    
    Args:
        engine_type: 引擎类型 (tripoSR, shap-e, stable-fast-3d, instantmesh)
        config: 配置参数
    
    Returns:
        ThreeDEngine实例
    """
    engines = {
        "tripoSR": TripoSREngine,
        "shap-e": ShapEEngine,
        "stable-fast-3d": StableFast3DEngine,
        "instantmesh": InstantMeshEngine,
    }
    
    if engine_type not in engines:
        raise ValueError(f"Unknown 3D engine type: {engine_type}")
    
    return engines[engine_type](config)
