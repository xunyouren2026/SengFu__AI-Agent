"""
OCR Tool - OCR文字识别工具
从图像中提取文字
"""

import json
import base64
import urllib.request
import urllib.error
import re
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OCRLanguage(Enum):
    """OCR语言枚举"""
    CHINESE = "chi_sim"
    CHINESE_TRADITIONAL = "chi_tra"
    ENGLISH = "eng"
    JAPANESE = "jpn"
    KOREAN = "kor"
    AUTO = "auto"


class OCRMode(Enum):
    """OCR模式枚举"""
    STANDARD = "standard"
    FAST = "fast"
    ACCURATE = "accurate"
    HANDWRITING = "handwriting"


@dataclass
class TextBox:
    """文本框"""
    text: str
    x: float
    y: float
    width: float
    height: float
    confidence: float
    language: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "language": self.language
        }
    
    def contains_point(self, px: float, py: float) -> bool:
        """检查点是否在框内"""
        return (self.x <= px <= self.x + self.width and
                self.y <= py <= self.y + self.height)


@dataclass
class OCRResult:
    """OCR结果"""
    text: str
    text_boxes: List[TextBox]
    full_text: str
    confidence: float
    languages_detected: List[str]
    word_count: int
    char_count: int
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "text_boxes": [tb.to_dict() for tb in self.text_boxes],
            "full_text": self.full_text,
            "confidence": self.confidence,
            "languages_detected": self.languages_detected,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "processing_time": self.processing_time,
            "metadata": self.metadata
        }
    
    def get_text_in_region(self, x: float, y: float,
                           width: float, height: float) -> str:
        """获取指定区域的文本"""
        texts = []
        for tb in self.text_boxes:
            if (tb.x >= x and tb.y >= y and
                tb.x + tb.width <= x + width and
                tb.y + tb.height <= y + height):
                texts.append(tb.text)
        return " ".join(texts)


@dataclass
class OCRConfig:
    """OCR配置"""
    languages: List[OCRLanguage] = field(default_factory=lambda: [OCRLanguage.ENGLISH])
    mode: OCRMode = OCRMode.STANDARD
    detect_language: bool = True
    preserve_formatting: bool = True
    extract_tables: bool = False
    min_confidence: float = 0.5
    dpi: int = 300
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    timeout: int = 60


class OCRTool:
    """OCR文字识别工具"""
    
    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
    
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
    
    def recognize(self, image: Union[str, bytes],
                  languages: Optional[List[OCRLanguage]] = None,
                  **kwargs) -> OCRResult:
        """识别图像中的文字"""
        if isinstance(image, str):
            image_base64 = self.load_image_base64(image)
        else:
            image_base64 = self.encode_image(image)
        
        langs = languages or self.config.languages
        
        result = self._call_model(image_base64, langs, **kwargs)
        
        return result
    
    def recognize_region(self, image: Union[str, bytes],
                         x: float, y: float,
                         width: float, height: float,
                         **kwargs) -> OCRResult:
        """识别图像指定区域的文字"""
        # 先识别整张图，然后过滤
        full_result = self.recognize(image, **kwargs)
        
        filtered_boxes = [
            tb for tb in full_result.text_boxes
            if x <= tb.x <= x + width - tb.width
            and y <= tb.y <= y + height - tb.height
        ]
        
        text = " ".join(tb.text for tb in filtered_boxes)
        
        return OCRResult(
            text=text,
            text_boxes=filtered_boxes,
            full_text=text,
            confidence=sum(tb.confidence for tb in filtered_boxes) / len(filtered_boxes) if filtered_boxes else 0,
            languages_detected=full_result.languages_detected,
            word_count=len(text.split()),
            char_count=len(text),
            metadata={"region": {"x": x, "y": y, "width": width, "height": height}}
        )
    
    def recognize_batch(self, images: List[Union[str, bytes]],
                        **kwargs) -> List[OCRResult]:
        """批量识别"""
        results = []
        for image in images:
            result = self.recognize(image, **kwargs)
            results.append(result)
        return results
    
    def _call_model(self, image_base64: str,
                    languages: List[OCRLanguage],
                    **kwargs) -> OCRResult:
        """调用模型"""
        if self.config.api_endpoint and self.config.api_key:
            return self._call_api(image_base64, languages)
        
        return OCRResult(
            text="[OCR placeholder]",
            text_boxes=[],
            full_text="[OCR placeholder]",
            confidence=0.0,
            languages_detected=[],
            word_count=0,
            char_count=0
        )
    
    def _call_api(self, image_base64: str,
                  languages: List[OCRLanguage]) -> OCRResult:
        """调用API"""
        lang_codes = [lang.value for lang in languages]
        
        payload = {
            "image": image_base64,
            "languages": lang_codes,
            "mode": self.config.mode.value,
            "detect_language": self.config.detect_language,
            "preserve_formatting": self.config.preserve_formatting,
            "min_confidence": self.config.min_confidence,
            "dpi": self.config.dpi
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
                
                text_boxes = []
                for tb_data in result.get("text_boxes", []):
                    text_boxes.append(TextBox(
                        text=tb_data.get("text", ""),
                        x=tb_data.get("x", 0),
                        y=tb_data.get("y", 0),
                        width=tb_data.get("width", 0),
                        height=tb_data.get("height", 0),
                        confidence=tb_data.get("confidence", 0),
                        language=tb_data.get("language", "")
                    ))
                
                full_text = result.get("full_text", "")
                
                return OCRResult(
                    text=full_text,
                    text_boxes=text_boxes,
                    full_text=full_text,
                    confidence=result.get("confidence", 0),
                    languages_detected=result.get("languages", []),
                    word_count=len(full_text.split()),
                    char_count=len(full_text),
                    processing_time=result.get("processing_time", 0),
                    metadata={"raw_response": result}
                )
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return OCRResult(
                text="",
                text_boxes=[],
                full_text="",
                confidence=0.0,
                languages_detected=[],
                word_count=0,
                char_count=0,
                metadata={"error": str(e)}
            )
    
    def extract_text(self, image: Union[str, bytes],
                     clean: bool = True,
                     **kwargs) -> str:
        """提取纯文本"""
        result = self.recognize(image, **kwargs)
        text = result.full_text
        
        if clean:
            text = self._clean_text(text)
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:\-()]', '', text)
        return text.strip()
    
    def detect_language(self, text: str) -> str:
        """检测文本语言"""
        # 简单的语言检测
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.replace(' ', ''))
        
        if total_chars == 0:
            return "unknown"
        
        if chinese_chars / total_chars > 0.3:
            return "chinese"
        
        return "english"
    
    def get_supported_languages(self) -> List[str]:
        """获取支持的语言"""
        return [lang.value for lang in OCRLanguage]
    
    def get_available_modes(self) -> List[str]:
        """获取可用模式"""
        return [mode.value for mode in OCRMode]
    
    def find_text(self, ocr_result: OCRResult,
                  pattern: str,
                  case_sensitive: bool = False) -> List[TextBox]:
        """在OCR结果中查找文本"""
        flags = 0 if case_sensitive else re.IGNORECASE
        
        try:
            regex = re.compile(pattern, flags)
        except re.error:
            # 如果不是有效正则，作为普通文本搜索
            regex = re.compile(re.escape(pattern), flags)
        
        matches = []
        for tb in ocr_result.text_boxes:
            if regex.search(tb.text):
                matches.append(tb)
        
        return matches
    
    def extract_numbers(self, ocr_result: OCRResult) -> List[Tuple[str, float]]:
        """提取数字"""
        numbers = []
        
        for tb in ocr_result.text_boxes:
            # 查找所有数字
            num_pattern = r'[-+]?\d*\.?\d+'
            matches = re.findall(num_pattern, tb.text)
            
            for match in matches:
                try:
                    numbers.append((tb.text, float(match)))
                except ValueError:
                    continue
        
        return numbers
    
    def extract_dates(self, ocr_result: OCRResult) -> List[str]:
        """提取日期"""
        date_patterns = [
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # YYYY-MM-DD
            r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',  # MM-DD-YYYY
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2}',  # MM-DD-YY
        ]
        
        dates = []
        for tb in ocr_result.text_boxes:
            for pattern in date_patterns:
                matches = re.findall(pattern, tb.text)
                dates.extend(matches)
        
        return dates
    
    def extract_emails(self, ocr_result: OCRResult) -> List[str]:
        """提取邮箱"""
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        emails = []
        
        for tb in ocr_result.text_boxes:
            matches = re.findall(email_pattern, tb.text)
            emails.extend(matches)
        
        return emails
    
    def extract_urls(self, ocr_result: OCRResult) -> List[str]:
        """提取URL"""
        url_pattern = r'https?://[\w\.-]+(?:/[\w\.-]*)*'
        urls = []
        
        for tb in ocr_result.text_boxes:
            matches = re.findall(url_pattern, tb.text)
            urls.extend(matches)
        
        return urls
