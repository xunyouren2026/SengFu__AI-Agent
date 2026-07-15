"""
Image Caption Tool - 图像描述生成工具
生成图像的文本描述
"""

import json
import base64
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CaptionStyle(Enum):
    """描述风格枚举"""
    CONCISE = "concise"
    DETAILED = "detailed"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    ACCESSIBLE = "accessible"


class CaptionLength(Enum):
    """描述长度枚举"""
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


@dataclass
class CaptionResult:
    """描述结果"""
    caption: str
    style: CaptionStyle
    confidence: float
    alternative_captions: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "caption": self.caption,
            "style": self.style.value,
            "confidence": self.confidence,
            "alternative_captions": self.alternative_captions,
            "tags": self.tags,
            "metadata": self.metadata
        }


@dataclass
class CaptionConfig:
    """描述配置"""
    model_name: str = "default"
    style: CaptionStyle = CaptionStyle.DETAILED
    length: CaptionLength = CaptionLength.MEDIUM
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    num_alternatives: int = 3
    include_tags: bool = True
    language: str = "en"
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    timeout: int = 60


class CaptionTool:
    """图像描述生成工具"""
    
    def __init__(self, config: Optional[CaptionConfig] = None):
        self.config = config or CaptionConfig()
        self._style_prompts = self._init_style_prompts()
    
    def _init_style_prompts(self) -> Dict[CaptionStyle, str]:
        """初始化风格提示"""
        return {
            CaptionStyle.CONCISE: "Provide a brief, one-sentence description of this image.",
            CaptionStyle.DETAILED: "Provide a detailed description of this image, including objects, actions, setting, and atmosphere.",
            CaptionStyle.CREATIVE: "Describe this image in a creative and engaging way, using vivid language.",
            CaptionStyle.TECHNICAL: "Provide a technical description of this image, focusing on composition, lighting, and visual elements.",
            CaptionStyle.ACCESSIBLE: "Describe this image in simple, easy-to-understand language suitable for all audiences."
        }
    
    def load_image(self, image_path: str) -> bytes:
        """加载图像"""
        with open(image_path, 'rb') as f:
            return f.read()
    
    def load_image_base64(self, image_path: str) -> str:
        """加载图像并转换为base64"""
        image_data = self.load_image(image_path)
        return base64.b64encode(image_data).decode('utf-8')
    
    def encode_image(self, image_data: bytes) -> str:
        """编码图像为base64"""
        return base64.b64encode(image_data).decode('utf-8')
    
    def generate(self, image: Union[str, bytes],
                 style: Optional[CaptionStyle] = None,
                 length: Optional[CaptionLength] = None,
                 **kwargs) -> CaptionResult:
        """生成图像描述"""
        # 处理输入
        if isinstance(image, str):
            image_base64 = self.load_image_base64(image)
        else:
            image_base64 = self.encode_image(image)
        
        # 使用配置或参数中的风格
        caption_style = style or self.config.style
        caption_length = length or self.config.length
        
        # 构建提示
        prompt = self._build_prompt(caption_style, caption_length)
        
        # 调用模型
        result = self._call_model(image_base64, prompt, **kwargs)
        
        return result
    
    def generate_batch(self, images: List[Union[str, bytes]],
                       style: Optional[CaptionStyle] = None,
                       **kwargs) -> List[CaptionResult]:
        """批量生成描述"""
        results = []
        for image in images:
            caption = self.generate(image, style, **kwargs)
            results.append(caption)
        return results
    
    def generate_with_context(self, image: Union[str, bytes],
                              context: str,
                              style: Optional[CaptionStyle] = None,
                              **kwargs) -> CaptionResult:
        """带上下文生成描述"""
        if isinstance(image, str):
            image_base64 = self.load_image_base64(image)
        else:
            image_base64 = self.encode_image(image)
        
        caption_style = style or self.config.style
        base_prompt = self._build_prompt(caption_style, self.config.length)
        prompt = f"{base_prompt}\n\nContext: {context}"
        
        return self._call_model(image_base64, prompt, **kwargs)
    
    def _build_prompt(self, style: CaptionStyle, length: CaptionLength) -> str:
        """构建提示"""
        style_prompt = self._style_prompts.get(style, self._style_prompts[CaptionStyle.DETAILED])
        
        length_instruction = {
            CaptionLength.SHORT: "Keep the description under 20 words.",
            CaptionLength.MEDIUM: "Keep the description between 20-50 words.",
            CaptionLength.LONG: "Provide a comprehensive description of 50-100 words."
        }
        
        prompt = f"{style_prompt}\n\n{length_instruction.get(length, '')}"
        
        if self.config.include_tags:
            prompt += "\n\nAlso provide relevant tags for the image."
        
        return prompt
    
    def _call_model(self, image_base64: str, prompt: str,
                    **kwargs) -> CaptionResult:
        """调用模型"""
        if self.config.api_endpoint and self.config.api_key:
            return self._call_api(image_base64, prompt)
        
        return CaptionResult(
            caption="[Caption placeholder]",
            style=self.config.style,
            confidence=0.0,
            metadata={"prompt": prompt}
        )
    
    def _call_api(self, image_base64: str, prompt: str) -> CaptionResult:
        """调用API"""
        payload = {
            "image": image_base64,
            "prompt": prompt,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "num_alternatives": self.config.num_alternatives
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }
        
        try:
            data = json.dumps(payload).encode()
            request = urllib.request.Request(
                self.config.api_endpoint,
                data=data,
                headers=headers,
                method="POST"
            )
            
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                result = json.loads(response.read())
                
                return CaptionResult(
                    caption=result.get("caption", ""),
                    style=self.config.style,
                    confidence=result.get("confidence", 0.0),
                    alternative_captions=result.get("alternatives", []),
                    tags=result.get("tags", []),
                    metadata={"raw_response": result}
                )
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return CaptionResult(
                caption="",
                style=self.config.style,
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    def get_available_styles(self) -> List[str]:
        """获取可用风格"""
        return [s.value for s in CaptionStyle]
    
    def get_available_lengths(self) -> List[str]:
        """获取可用长度"""
        return [l.value for l in CaptionLength]
    
    def compare_captions(self, caption1: str, caption2: str) -> Dict[str, Any]:
        """比较两个描述"""
        words1 = set(caption1.lower().split())
        words2 = set(caption2.lower().split())
        
        common_words = words1 & words2
        unique1 = words1 - words2
        unique2 = words2 - words1
        
        similarity = len(common_words) / max(len(words1), len(words2)) if words1 or words2 else 0
        
        return {
            "similarity": similarity,
            "common_words": list(common_words),
            "unique_to_first": list(unique1),
            "unique_to_second": list(unique2),
            "word_count": {
                "first": len(words1),
                "second": len(words2)
            }
        }
