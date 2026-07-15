"""
Anthropic Vision API 实现模块

提供完整的视觉/图像理解功能支持，包括：
- 图像编码和解码
- 多模态消息构建
- 图像URL和base64处理
- 批量图像处理

参考: https://docs.anthropic.com/claude/docs/vision
"""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from urllib.parse import urlparse

from .exceptions import ValidationError


# 支持的图像格式
SUPPORTED_IMAGE_TYPES = {
    "image/jpeg": "jpeg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
}

# 图像源类型
SourceType = Literal["base64", "url"]
MediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]


@dataclass
class ImageSource:
    """
    图像源
    
    表示图像数据的来源，可以是base64编码或URL
    
    示例:
        >>> # 从base64创建
        >>> source = ImageSource.from_base64(base64_data, "image/jpeg")
        >>>
        >>> # 从URL创建
        >>> source = ImageSource.from_url("https://example.com/image.jpg")
        >>>
        >>> # 从文件创建
        >>> source = ImageSource.from_file("/path/to/image.jpg")
    """
    type: SourceType
    media_type: MediaType
    data: Optional[str] = None  # base64编码的图像数据
    url: Optional[str] = None   # 图像URL
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为API格式
        
        Returns:
            符合Anthropic Vision API的图像源字典
        """
        if self.type == "base64":
            return {
                "type": "base64",
                "media_type": self.media_type,
                "data": self.data,
            }
        else:  # url
            return {
                "type": "url",
                "url": self.url,
            }
    
    @classmethod
    def from_base64(
        cls, 
        data: str, 
        media_type: MediaType = "image/jpeg"
    ) -> ImageSource:
        """
        从base64字符串创建图像源
        
        Args:
            data: base64编码的图像数据
            media_type: 图像MIME类型
        
        Returns:
            ImageSource实例
        """
        return cls(
            type="base64",
            media_type=media_type,
            data=data,
        )
    
    @classmethod
    def from_url(cls, url: str) -> ImageSource:
        """
        从URL创建图像源
        
        Args:
            url: 图像URL
        
        Returns:
            ImageSource实例
        """
        return cls(
            type="url",
            media_type="image/jpeg",  # URL类型不验证media_type
            url=url,
        )
    
    @classmethod
    def from_file(
        cls, 
        file_path: Union[str, Path],
        media_type: Optional[MediaType] = None,
    ) -> ImageSource:
        """
        从文件创建图像源
        
        Args:
            file_path: 图像文件路径
            media_type: 图像MIME类型，如果为None则自动检测
        
        Returns:
            ImageSource实例
        
        Raises:
            FileNotFoundError: 文件不存在时抛出
            ValidationError: 图像格式不支持时抛出
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        # 自动检测MIME类型
        if media_type is None:
            mime_type, _ = mimetypes.guess_type(str(path))
            if mime_type not in SUPPORTED_IMAGE_TYPES:
                raise ValidationError(
                    f"Unsupported image format: {mime_type}. "
                    f"Supported formats: {list(SUPPORTED_IMAGE_TYPES.keys())}"
                )
            media_type = mime_type  # type: ignore
        
        # 读取并编码文件
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        
        return cls(
            type="base64",
            media_type=media_type,
            data=data,
        )
    
    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        media_type: MediaType = "image/jpeg",
    ) -> ImageSource:
        """
        从字节数据创建图像源
        
        Args:
            data: 图像字节数据
            media_type: 图像MIME类型
        
        Returns:
            ImageSource实例
        """
        base64_data = base64.b64encode(data).decode("utf-8")
        return cls(
            type="base64",
            media_type=media_type,
            data=base64_data,
        )


@dataclass
class VisionContent:
    """
    视觉内容块
    
    用于构建包含图像的多模态消息内容
    
    示例:
        >>> # 纯图像
        >>> image_source = ImageSource.from_file("photo.jpg")
        >>> content = VisionContent.image(image_source)
        >>>
        >>> # 图文混合
        >>> content = VisionContent.multimodal(
        ...     text="描述这张图片",
        ...     images=[image_source]
        ... )
    """
    blocks: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> List[Dict[str, Any]]:
        """转换为API格式"""
        return self.blocks
    
    @classmethod
    def text(cls, text: str) -> VisionContent:
        """
        创建纯文本内容
        
        Args:
            text: 文本内容
        
        Returns:
            VisionContent实例
        """
        return cls(blocks=[{"type": "text", "text": text}])
    
    @classmethod
    def image(cls, source: ImageSource) -> VisionContent:
        """
        创建纯图像内容
        
        Args:
            source: 图像源
        
        Returns:
            VisionContent实例
        """
        return cls(blocks=[{
            "type": "image",
            "source": source.to_dict(),
        }])
    
    @classmethod
    def multimodal(
        cls,
        text: str,
        images: List[ImageSource],
    ) -> VisionContent:
        """
        创建图文混合内容
        
        Args:
            text: 文本内容
            images: 图像源列表
        
        Returns:
            VisionContent实例
        """
        blocks: List[Dict[str, Any]] = [{"type": "text", "text": text}]
        
        for image in images:
            blocks.append({
                "type": "image",
                "source": image.to_dict(),
            })
        
        return cls(blocks=blocks)
    
    def add_text(self, text: str) -> VisionContent:
        """
        添加文本块
        
        Args:
            text: 文本内容
        
        Returns:
            self，支持链式调用
        """
        self.blocks.append({"type": "text", "text": text})
        return self
    
    def add_image(self, source: ImageSource) -> VisionContent:
        """
        添加图像块
        
        Args:
            source: 图像源
        
        Returns:
            self，支持链式调用
        """
        self.blocks.append({
            "type": "image",
            "source": source.to_dict(),
        })
        return self


@dataclass
class VisionMessage:
    """
    视觉消息
    
    用于构建包含图像的对话消息
    
    示例:
        >>> message = VisionMessage.user(
        ...     text="这是什么?",
        ...     images=[ImageSource.from_file("photo.jpg")]
        ... )
    """
    role: Literal["user", "assistant"]
    content: Union[str, List[Dict[str, Any]]]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为API格式"""
        return {
            "role": self.role,
            "content": self.content,
        }
    
    @classmethod
    def user(
        cls,
        text: str,
        images: Optional[List[ImageSource]] = None,
    ) -> VisionMessage:
        """
        创建用户视觉消息
        
        Args:
            text: 文本内容
            images: 可选的图像列表
        
        Returns:
            VisionMessage实例
        """
        if images:
            content = VisionContent.multimodal(text, images).to_dict()
        else:
            content = text
        
        return cls(role="user", content=content)
    
    @classmethod
    def assistant(cls, text: str) -> VisionMessage:
        """
        创建助手视觉消息
        
        Args:
            text: 文本内容
        
        Returns:
            VisionMessage实例
        """
        return cls(role="assistant", content=text)


class VisionClient:
    """
    视觉客户端
    
    提供便捷的视觉API调用方法
    
    示例:
        >>> client = VisionClient(anthropic_client)
        >>> response = client.analyze_image(
        ...     image_path="photo.jpg",
        ...     prompt="描述这张图片的内容"
        ... )
    """
    
    def __init__(self, client: Any) -> None:
        """
        初始化视觉客户端
        
        Args:
            client: Anthropic客户端实例
        """
        self._client = client
    
    def analyze_image(
        self,
        image_path: Union[str, Path],
        prompt: str = "描述这张图片",
        model: str = "claude-3-opus-20240229",
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        """
        分析单张图片
        
        Args:
            image_path: 图像文件路径
            prompt: 分析提示词
            model: 模型名称
            max_tokens: 最大生成token数
            **kwargs: 其他参数
        
        Returns:
            分析结果文本
        """
        image_source = ImageSource.from_file(image_path)
        
        message = VisionMessage.user(
            text=prompt,
            images=[image_source],
        )
        
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[message.to_dict()],
            **kwargs,
        )
        
        return response.text
    
    def analyze_images(
        self,
        image_paths: List[Union[str, Path]],
        prompt: str = "描述这些图片",
        model: str = "claude-3-opus-20240229",
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """
        分析多张图片
        
        Args:
            image_paths: 图像文件路径列表
            prompt: 分析提示词
            model: 模型名称
            max_tokens: 最大生成token数
            **kwargs: 其他参数
        
        Returns:
            分析结果文本
        """
        image_sources = [ImageSource.from_file(path) for path in image_paths]
        
        message = VisionMessage.user(
            text=prompt,
            images=image_sources,
        )
        
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[message.to_dict()],
            **kwargs,
        )
        
        return response.text
    
    def compare_images(
        self,
        image_path1: Union[str, Path],
        image_path2: Union[str, Path],
        prompt: str = "比较这两张图片的异同",
        model: str = "claude-3-opus-20240229",
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """
        比较两张图片
        
        Args:
            image_path1: 第一张图片路径
            image_path2: 第二张图片路径
            prompt: 比较提示词
            model: 模型名称
            max_tokens: 最大生成token数
            **kwargs: 其他参数
        
        Returns:
            比较结果文本
        """
        image_sources = [
            ImageSource.from_file(image_path1),
            ImageSource.from_file(image_path2),
        ]
        
        message = VisionMessage.user(
            text=prompt,
            images=image_sources,
        )
        
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[message.to_dict()],
            **kwargs,
        )
        
        return response.text
    
    def extract_text_from_image(
        self,
        image_path: Union[str, Path],
        model: str = "claude-3-opus-20240229",
        max_tokens: int = 4096,
        **kwargs,
    ) -> str:
        """
        从图片中提取文字(OCR)
        
        Args:
            image_path: 图像文件路径
            model: 模型名称
            max_tokens: 最大生成token数
            **kwargs: 其他参数
        
        Returns:
            提取的文本内容
        """
        return self.analyze_image(
            image_path=image_path,
            prompt="提取并返回图片中的所有文字内容，保持原有格式：",
            model=model,
            max_tokens=max_tokens,
            **kwargs,
        )
    
    def analyze_image_url(
        self,
        image_url: str,
        prompt: str = "描述这张图片",
        model: str = "claude-3-opus-20240229",
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        """
        分析图片URL
        
        Args:
            image_url: 图片URL
            prompt: 分析提示词
            model: 模型名称
            max_tokens: 最大生成token数
            **kwargs: 其他参数
        
        Returns:
            分析结果文本
        """
        image_source = ImageSource.from_url(image_url)
        
        message = VisionMessage.user(
            text=prompt,
            images=[image_source],
        )
        
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[message.to_dict()],
            **kwargs,
        )
        
        return response.text


# 便捷函数
def encode_image_to_base64(image_path: Union[str, Path]) -> str:
    """
    将图像文件编码为base64字符串
    
    Args:
        image_path: 图像文件路径
    
    Returns:
        base64编码的字符串
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def decode_base64_to_image(
    base64_data: str,
    output_path: Optional[Union[str, Path]] = None,
) -> bytes:
    """
    将base64字符串解码为图像数据
    
    Args:
        base64_data: base64编码的图像数据
        output_path: 可选的输出文件路径
    
    Returns:
        图像字节数据
    """
    image_data = base64.b64decode(base64_data)
    
    if output_path:
        with open(output_path, "wb") as f:
            f.write(image_data)
    
    return image_data


def get_image_media_type(image_path: Union[str, Path]) -> MediaType:
    """
    获取图像文件的MIME类型
    
    Args:
        image_path: 图像文件路径
    
    Returns:
        MIME类型字符串
    
    Raises:
        ValidationError: 格式不支持时抛出
    """
    mime_type, _ = mimetypes.guess_type(str(image_path))
    
    if mime_type not in SUPPORTED_IMAGE_TYPES:
        raise ValidationError(
            f"Unsupported image format: {mime_type}. "
            f"Supported: {list(SUPPORTED_IMAGE_TYPES.keys())}"
        )
    
    return mime_type  # type: ignore


def create_image_message(
    text: str,
    image_path: Union[str, Path],
) -> Dict[str, Any]:
    """
    快速创建包含图像的消息
    
    Args:
        text: 文本内容
        image_path: 图像文件路径
    
    Returns:
        消息字典
    """
    image_source = ImageSource.from_file(image_path)
    message = VisionMessage.user(text=text, images=[image_source])
    return message.to_dict()


def create_multimodal_content(
    text: str,
    image_paths: List[Union[str, Path]],
) -> List[Dict[str, Any]]:
    """
    快速创建多模态内容
    
    Args:
        text: 文本内容
        image_paths: 图像文件路径列表
    
    Returns:
        内容块列表
    """
    image_sources = [ImageSource.from_file(path) for path in image_paths]
    content = VisionContent.multimodal(text, image_sources)
    return content.to_dict()
