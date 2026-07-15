"""
OCR预处理与语言检测模块

提供图像预处理、语言检测和Tesseract封装功能。
仅使用Python标准库实现（模拟接口）。
"""

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ImageData:
    """图像数据结构。"""
    width: int
    height: int
    pixels: Dict[Tuple[int, int], Tuple[int, int, int]]
    
    def get_pixel(self, x: int, y: int) -> Tuple[int, int, int]:
        """获取像素值。"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.pixels.get((x, y), (255, 255, 255))
        return (0, 0, 0)
    
    def set_pixel(self, x: int, y: int, color: Tuple[int, int, int]) -> None:
        """设置像素值。"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.pixels[(x, y)] = color
    
    def to_grayscale(self) -> "ImageData":
        """转换为灰度图像。"""
        new_pixels = {}
        for (x, y), (r, g, b) in self.pixels.items():
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            new_pixels[(x, y)] = (gray, gray, gray)
        return ImageData(self.width, self.height, new_pixels)


@dataclass
class OCRResult:
    """OCR识别结果。"""
    text: str
    confidence: float
    language: str = ""
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "language": self.language,
            "bounding_box": self.bounding_box,
        }


@dataclass
class LanguageDetectionResult:
    """语言检测结果。"""
    language: str
    confidence: float
    script: str = ""  # 文字体系（如Latin, Han, Cyrillic等）
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "language": self.language,
            "confidence": self.confidence,
            "script": self.script,
        }


# ============================================================
# OCRPreprocessor: 图像预处理
# ============================================================

class OCRPreprocessor:
    """OCR图像预处理器。
    
    提供二值化、降噪、倾斜校正、DPI调整和对比度增强等功能。
    使用纯Python实现，仅使用标准库。
    """
    
    def __init__(self):
        """初始化预处理器。"""
        pass
    
    def _to_grayscale(self, image: ImageData) -> ImageData:
        """将图像转换为灰度。
        
        Args:
            image: 输入图像
            
        Returns:
            灰度图像
        """
        return image.to_grayscale()
    
    def _get_gray_value(self, image: ImageData, x: int, y: int) -> int:
        """获取像素的灰度值。
        
        Args:
            image: 图像
            x: X坐标
            y: Y坐标
            
        Returns:
            灰度值（0-255）
        """
        r, g, b = image.get_pixel(x, y)
        return int(0.299 * r + 0.587 * g + 0.114 * b)
    
    def binarize(self, image: ImageData, threshold: int = 128) -> ImageData:
        """图像二值化。
        
        使用简单的阈值方法将图像转换为黑白。
        
        Args:
            image: 输入图像
            threshold: 阈值（0-255），默认128
            
        Returns:
            二值化后的图像
        """
        gray_image = self._to_grayscale(image)
        new_pixels = {}
        
        for (x, y), (gray, _, _) in gray_image.pixels.items():
            if gray >= threshold:
                new_pixels[(x, y)] = (255, 255, 255)  # 白色
            else:
                new_pixels[(x, y)] = (0, 0, 0)  # 黑色
        
        return ImageData(image.width, image.height, new_pixels)
    
    def _otsu_threshold(self, image: ImageData) -> int:
        """使用Otsu方法计算最佳阈值。
        
        Args:
            image: 灰度图像
            
        Returns:
            最佳阈值
        """
        gray_image = self._to_grayscale(image)
        
        # 计算灰度直方图
        histogram = [0] * 256
        for (x, y), (gray, _, _) in gray_image.pixels.items():
            histogram[gray] += 1
        
        total_pixels = len(gray_image.pixels)
        if total_pixels == 0:
            return 128
        
        # 计算类间方差
        max_variance = 0
        optimal_threshold = 128
        
        for threshold in range(256):
            # 背景类
            w0 = sum(histogram[:threshold])
            if w0 == 0:
                continue
            
            # 前景类
            w1 = sum(histogram[threshold:])
            if w1 == 0:
                continue
            
            # 计算均值
            mu0 = sum(i * histogram[i] for i in range(threshold)) / w0
            mu1 = sum(i * histogram[i] for i in range(threshold, 256)) / w1
            
            # 计算类间方差
            variance = (w0 * w1 * (mu0 - mu1) ** 2) / (total_pixels ** 2)
            
            if variance > max_variance:
                max_variance = variance
                optimal_threshold = threshold
        
        return optimal_threshold
    
    def binarize_auto(self, image: ImageData) -> ImageData:
        """自动二值化（使用Otsu方法）。
        
        Args:
            image: 输入图像
            
        Returns:
            二值化后的图像
        """
        threshold = self._otsu_threshold(image)
        return self.binarize(image, threshold)
    
    def denoise(self, image: ImageData, kernel_size: int = 3) -> ImageData:
        """图像降噪（使用高斯滤波）。
        
        Args:
            image: 输入图像
            kernel_size: 卷积核大小（奇数），默认3
            
        Returns:
            降噪后的图像
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        gray_image = self._to_grayscale(image)
        new_pixels = {}
        
        half_kernel = kernel_size // 2
        
        # 生成高斯核
        sigma = kernel_size / 3.0
        kernel = []
        for i in range(kernel_size):
            row = []
            for j in range(kernel_size):
                x = i - half_kernel
                y = j - half_kernel
                weight = math.exp(-(x * x + y * y) / (2 * sigma * sigma))
                row.append(weight)
            kernel.append(row)
        
        # 归一化
        total_weight = sum(sum(row) for row in kernel)
        kernel = [[w / total_weight for w in row] for row in kernel]
        
        # 应用卷积
        for y in range(image.height):
            for x in range(image.width):
                weighted_sum = 0.0
                
                for ky in range(kernel_size):
                    for kx in range(kernel_size):
                        px = x + kx - half_kernel
                        py = y + ky - half_kernel
                        
                        if 0 <= px < image.width and 0 <= py < image.height:
                            gray = self._get_gray_value(gray_image, px, py)
                            weighted_sum += gray * kernel[ky][kx]
                
                new_gray = int(round(weighted_sum))
                new_gray = max(0, min(255, new_gray))
                new_pixels[(x, y)] = (new_gray, new_gray, new_gray)
        
        return ImageData(image.width, image.height, new_pixels)
    
    def deskew(self, image: ImageData) -> ImageData:
        """图像倾斜校正（使用霍夫变换）。
        
        检测图像中的文本行角度并进行旋转校正。
        
        Args:
            image: 输入图像
            
        Returns:
            校正后的图像
        """
        # 简化的倾斜检测：检测水平线
        gray_image = self._to_grayscale(image)
        
        # 边缘检测（Sobel简化版）
        edges = set()
        for y in range(1, image.height - 1):
            for x in range(1, image.width - 1):
                # 水平梯度
                gx = (self._get_gray_value(gray_image, x + 1, y) -
                      self._get_gray_value(gray_image, x - 1, y))
                # 垂直梯度
                gy = (self._get_gray_value(gray_image, x, y + 1) -
                      self._get_gray_value(gray_image, x, y - 1))
                
                gradient = math.sqrt(gx * gx + gy * gy)
                if gradient > 50:  # 边缘阈值
                    edges.add((x, y))
        
        # 简化的霍夫变换检测主要角度
        angle_votes = Counter()
        
        for x, y in list(edges)[:1000]:  # 采样前1000个边缘点
            for angle in range(-10, 11):  # 检测-10到10度
                rad = math.radians(angle)
                rho = x * math.cos(rad) + y * math.sin(rad)
                angle_votes[angle] += 1
        
        if angle_votes:
            skew_angle = angle_votes.most_common(1)[0][0]
        else:
            skew_angle = 0
        
        # 如果倾斜角度很小，直接返回原图
        if abs(skew_angle) < 1:
            return image
        
        # 简化的旋转（仅支持90度倍数）
        # 实际实现需要更复杂的插值
        return self._rotate_simple(image, -skew_angle)
    
    def _rotate_simple(self, image: ImageData, angle: float) -> ImageData:
        """简单图像旋转。
        
        Args:
            image: 输入图像
            angle: 旋转角度（度）
            
        Returns:
            旋转后的图像
        """
        # 简化的旋转实现
        # 实际应用中需要使用双线性插值
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        
        cx = image.width / 2
        cy = image.height / 2
        
        new_pixels = {}
        
        for y in range(image.height):
            for x in range(image.width):
                # 反向映射
                dx = x - cx
                dy = y - cy
                
                src_x = int(cx + dx * cos_a - dy * sin_a)
                src_y = int(cy + dx * sin_a + dy * cos_a)
                
                if 0 <= src_x < image.width and 0 <= src_y < image.height:
                    color = image.get_pixel(src_x, src_y)
                    new_pixels[(x, y)] = color
                else:
                    new_pixels[(x, y)] = (255, 255, 255)  # 白色背景
        
        return ImageData(image.width, image.height, new_pixels)
    
    def resize_to_dpi(self, image: ImageData, target_dpi: int = 300) -> ImageData:
        """调整图像DPI。
        
        假设输入图像为96 DPI，缩放到目标DPI。
        
        Args:
            image: 输入图像
            target_dpi: 目标DPI，默认300
            
        Returns:
            调整后的图像
        """
        scale_factor = target_dpi / 96.0
        new_width = int(image.width * scale_factor)
        new_height = int(image.height * scale_factor)
        
        # 简化的最近邻插值
        new_pixels = {}
        for y in range(new_height):
            for x in range(new_width):
                src_x = int(x / scale_factor)
                src_y = int(y / scale_factor)
                
                src_x = min(src_x, image.width - 1)
                src_y = min(src_y, image.height - 1)
                
                new_pixels[(x, y)] = image.get_pixel(src_x, src_y)
        
        return ImageData(new_width, new_height, new_pixels)
    
    def enhance_contrast(self, image: ImageData) -> ImageData:
        """对比度增强（直方图均衡化）。
        
        Args:
            image: 输入图像
            
        Returns:
            增强后的图像
        """
        gray_image = self._to_grayscale(image)
        
        # 计算灰度直方图
        histogram = [0] * 256
        for (x, y), (gray, _, _) in gray_image.pixels.items():
            histogram[gray] += 1
        
        total_pixels = len(gray_image.pixels)
        if total_pixels == 0:
            return image
        
        # 计算累积分布函数（CDF）
        cdf = [0] * 256
        cdf[0] = histogram[0]
        for i in range(1, 256):
            cdf[i] = cdf[i - 1] + histogram[i]
        
        # 归一化CDF
        cdf_min = next((c for c in cdf if c > 0), 0)
        cdf_normalized = [
            int(round((c - cdf_min) / (total_pixels - cdf_min) * 255)) if total_pixels > cdf_min else 0
            for c in cdf
        ]
        
        # 应用均衡化
        new_pixels = {}
        for (x, y), (gray, _, _) in gray_image.pixels.items():
            new_gray = cdf_normalized[gray]
            new_pixels[(x, y)] = (new_gray, new_gray, new_gray)
        
        return ImageData(image.width, image.height, new_pixels)
    
    def preprocess_pipeline(self, image: ImageData) -> ImageData:
        """预处理流水线。
        
        依次执行：灰度化 -> 降噪 -> 对比度增强 -> 二值化
        
        Args:
            image: 输入图像
            
        Returns:
            处理后的图像
        """
        # 1. 转换为灰度
        gray = self._to_grayscale(image)
        
        # 2. 降噪
        denoised = self.denoise(gray)
        
        # 3. 对比度增强
        enhanced = self.enhance_contrast(denoised)
        
        # 4. 自动二值化
        binary = self.binarize_auto(enhanced)
        
        return binary


# ============================================================
# LanguageDetector: 语言检测
# ============================================================

class LanguageDetector:
    """语言检测器。
    
    使用字符频率分析检测文本语言（中/英/日/韩等）。
    """
    
    # Unicode 脚本范围
    SCRIPT_RANGES = {
        "Latin": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x00FF)],
        "Han": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF)],
        "Hiragana": [(0x3040, 0x309F)],
        "Katakana": [(0x30A0, 0x30FF)],
        "Hangul": [(0xAC00, 0xD7AF), (0x1100, 0x11FF)],
        "Cyrillic": [(0x0400, 0x04FF)],
        "Arabic": [(0x0600, 0x06FF)],
        "Devanagari": [(0x0900, 0x097F)],
        "Thai": [(0x0E00, 0x0E7F)],
    }
    
    # 语言特征
    LANGUAGE_PROFILES = {
        "zh": {
            "scripts": ["Han"],
            "char_frequency": {},
            "common_chars": set("的一是不了在人有我他这个们中来上大为和国地到以说时要就出会可也你对生能而子那得于着下自之年过发后作里如进着"),
        },
        "en": {
            "scripts": ["Latin"],
            "char_frequency": {},
            "common_words": set("the be to of and a in that have i it for not on with he as you do at"),
        },
        "ja": {
            "scripts": ["Hiragana", "Katakana", "Han"],
            "char_frequency": {},
            "common_chars": set("のにしてをがでたはとしあなるいかもよ"),
        },
        "ko": {
            "scripts": ["Hangul"],
            "char_frequency": {},
            "common_chars": set("의가이은을로는들과에의로"),
        },
    }
    
    def __init__(self):
        """初始化语言检测器。"""
        self._last_result: Optional[LanguageDetectionResult] = None
    
    def _get_char_script(self, char: str) -> str:
        """获取字符所属的脚本。
        
        Args:
            char: 字符
            
        Returns:
            脚本名称
        """
        code_point = ord(char)
        
        for script, ranges in self.SCRIPT_RANGES.items():
            for start, end in ranges:
                if start <= code_point <= end:
                    return script
        
        return "Other"
    
    def _analyze_scripts(self, text: str) -> Dict[str, int]:
        """分析文本中的脚本分布。
        
        Args:
            text: 输入文本
            
        Returns:
            脚本计数字典
        """
        script_counts = Counter()
        
        for char in text:
            if char.isalpha() or unicodedata.category(char).startswith('L'):
                script = self._get_char_script(char)
                script_counts[script] += 1
        
        return dict(script_counts)
    
    def detect_language(self, text: str) -> LanguageDetectionResult:
        """检测文本语言。
        
        使用字符频率分析确定文本的主要语言。
        
        Args:
            text: 输入文本
            
        Returns:
            语言检测结果
        """
        if not text:
            result = LanguageDetectionResult("unknown", 0.0, "")
            self._last_result = result
            return result
        
        # 分析脚本
        script_counts = self._analyze_scripts(text)
        
        if not script_counts:
            result = LanguageDetectionResult("unknown", 0.0, "")
            self._last_result = result
            return result
        
        # 根据脚本判断语言
        total_chars = sum(script_counts.values())
        dominant_script = max(script_counts, key=script_counts.get)
        script_confidence = script_counts[dominant_script] / total_chars
        
        # 脚本到语言的映射
        if dominant_script == "Han":
            # 检查是否包含日文特征（平假名/片假名）
            if script_counts.get("Hiragana", 0) > 0 or script_counts.get("Katakana", 0) > 0:
                language = "ja"
                script_confidence *= 0.9  # 稍微降低置信度
            else:
                language = "zh"
        elif dominant_script == "Latin":
            language = "en"
        elif dominant_script == "Hiragana" or dominant_script == "Katakana":
            language = "ja"
        elif dominant_script == "Hangul":
            language = "ko"
        elif dominant_script == "Cyrillic":
            language = "ru"
        elif dominant_script == "Arabic":
            language = "ar"
        elif dominant_script == "Devanagari":
            language = "hi"
        elif dominant_script == "Thai":
            language = "th"
        else:
            language = "unknown"
        
        # 计算置信度
        confidence = self._calculate_confidence(text, language, script_counts)
        
        result = LanguageDetectionResult(language, confidence, dominant_script)
        self._last_result = result
        return result
    
    def _calculate_confidence(
        self,
        text: str,
        language: str,
        script_counts: Dict[str, int],
    ) -> float:
        """计算语言检测的置信度。
        
        Args:
            text: 输入文本
            language: 检测到的语言
            script_counts: 脚本计数
            
        Returns:
            置信度（0-1）
        """
        total_chars = sum(script_counts.values())
        if total_chars == 0:
            return 0.0
        
        # 基础置信度：主导脚本的比例
        dominant_script = max(script_counts, key=script_counts.get)
        base_confidence = script_counts[dominant_script] / total_chars
        
        # 根据语言特征调整
        profile = self.LANGUAGE_PROFILES.get(language)
        if profile and "common_chars" in profile:
            common_chars = profile["common_chars"]
            matches = sum(1 for c in text if c in common_chars)
            char_ratio = matches / len(text) if text else 0
            
            # 加权平均
            confidence = base_confidence * 0.7 + char_ratio * 0.3
        else:
            confidence = base_confidence
        
        # 文本长度惩罚（非常短的文本置信度较低）
        if len(text) < 10:
            confidence *= 0.8
        
        return min(1.0, max(0.0, confidence))
    
    def get_language_confidence(self) -> float:
        """获取最后一次检测的置信度。
        
        Returns:
            置信度（0-1），如果没有进行过检测则返回0
        """
        if self._last_result is None:
            return 0.0
        return self._last_result.confidence
    
    def detect_multiple(self, text: str) -> List[LanguageDetectionResult]:
        """检测文本中可能包含的多种语言。
        
        Args:
            text: 输入文本
            
        Returns:
            语言检测结果列表（按置信度排序）
        """
        script_counts = self._analyze_scripts(text)
        
        if not script_counts:
            return []
        
        results = []
        total_chars = sum(script_counts.values())
        
        for script, count in script_counts.items():
            script_confidence = count / total_chars
            
            # 脚本到语言的映射
            if script == "Han":
                # 检查是否包含日文特征
                if script_counts.get("Hiragana", 0) > 0 or script_counts.get("Katakana", 0) > 0:
                    language = "ja"
                else:
                    language = "zh"
            elif script == "Latin":
                language = "en"
            elif script == "Hiragana" or script == "Katakana":
                language = "ja"
            elif script == "Hangul":
                language = "ko"
            elif script == "Cyrillic":
                language = "ru"
            else:
                continue
            
            results.append(LanguageDetectionResult(language, script_confidence, script))
        
        # 按置信度排序
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results


# ============================================================
# TesseractWrapper: Tesseract封装
# ============================================================

class TesseractWrapper:
    """Tesseract OCR封装（模拟实现）。
    
    提供OCR识别和语言管理功能。
    实际Tesseract集成需要安装tesseract-ocr包。
    """
    
    # 模拟的可用语言
    AVAILABLE_LANGUAGES = {
        "eng": "English",
        "chi_sim": "Chinese (Simplified)",
        "chi_tra": "Chinese (Traditional)",
        "jpn": "Japanese",
        "kor": "Korean",
        "fra": "French",
        "deu": "German",
        "spa": "Spanish",
        "rus": "Russian",
        "ara": "Arabic",
    }
    
    # 语言代码映射
    LANG_CODE_MAP = {
        "en": "eng",
        "zh": "chi_sim",
        "ja": "jpn",
        "ko": "kor",
        "fr": "fra",
        "de": "deu",
        "es": "spa",
        "ru": "rus",
        "ar": "ara",
    }
    
    def __init__(self, lang: str = "eng"):
        """初始化Tesseract封装。
        
        Args:
            lang: 默认语言代码
        """
        self._default_lang = self._normalize_lang_code(lang)
        self._installed_languages: Set[str] = set(self.AVAILABLE_LANGUAGES.keys())
        self._last_result: Optional[OCRResult] = None
    
    def _normalize_lang_code(self, lang: str) -> str:
        """标准化语言代码。
        
        Args:
            lang: 语言代码
            
        Returns:
            Tesseract语言代码
        """
        lang = lang.lower()
        
        # 映射到Tesseract代码
        if lang in self.LANG_CODE_MAP:
            return self.LANG_CODE_MAP[lang]
        
        # 已经是Tesseract代码
        if lang in self.AVAILABLE_LANGUAGES:
            return lang
        
        # 默认返回英语
        return "eng"
    
    def recognize(self, image: ImageData, lang: Optional[str] = None) -> OCRResult:
        """OCR识别。
        
        Args:
            image: 输入图像
            lang: 语言代码（None使用默认语言）
            
        Returns:
            OCR识别结果
        """
        use_lang = self._normalize_lang_code(lang) if lang else self._default_lang
        
        # 模拟OCR识别结果
        # 实际实现中需要调用tesseract命令行或API
        
        # 根据图像内容生成模拟文本
        simulated_text = self._generate_simulated_text(image, use_lang)
        
        # 计算模拟置信度
        confidence = self._calculate_simulated_confidence(image)
        
        result = OCRResult(
            text=simulated_text,
            confidence=confidence,
            language=use_lang,
            bounding_box=(0, 0, image.width, image.height),
        )
        
        self._last_result = result
        return result
    
    def _generate_simulated_text(self, image: ImageData, lang: str) -> str:
        """生成模拟OCR文本。
        
        Args:
            image: 输入图像
            lang: 语言代码
            
        Returns:
            模拟文本
        """
        # 根据图像大小和复杂度生成模拟文本
        pixel_count = len(image.pixels)
        
        # 简单的模拟：根据像素数量决定文本长度
        text_length = min(100, max(10, pixel_count // 1000))
        
        # 根据语言返回不同的模拟文本
        if lang in ("chi_sim", "chi_tra"):
            sample_text = "这是一段示例文本用于演示OCR识别功能在实际应用中需要连接真正的Tesseract引擎"
            return sample_text[:text_length]
        elif lang == "jpn":
            sample_text = "これはサンプルテキストですOCR機能をデモンストレーションするために使用されます"
            return sample_text[:text_length]
        elif lang == "kor":
            sample_text = "이것은 샘플 텍스트입니다 OCR 기능을 시연하기 위해 사용됩니다"
            return sample_text[:text_length]
        else:
            sample_text = "This is a sample text used to demonstrate OCR functionality in real applications you need to connect to the actual Tesseract engine"
            return sample_text[:text_length]
    
    def _calculate_simulated_confidence(self, image: ImageData) -> float:
        """计算模拟置信度。
        
        Args:
            image: 输入图像
            
        Returns:
            置信度（0-1）
        """
        # 基于图像复杂度的简单模拟
        pixel_count = len(image.pixels)
        
        # 假设较大的图像有更多文本，置信度更高
        base_confidence = min(0.95, 0.5 + pixel_count / 100000)
        
        return base_confidence
    
    def get_available_languages(self) -> List[str]:
        """获取可用语言列表。
        
        Returns:
            语言代码列表
        """
        return list(self._installed_languages)
    
    def get_available_languages_with_names(self) -> Dict[str, str]:
        """获取可用语言及其名称。
        
        Returns:
            语言代码到名称的映射
        """
        return {
            code: name
            for code, name in self.AVAILABLE_LANGUAGES.items()
            if code in self._installed_languages
        }
    
    def is_language_installed(self, lang: str) -> bool:
        """检查语言是否已安装。
        
        Args:
            lang: 语言代码
            
        Returns:
            是否已安装
        """
        normalized = self._normalize_lang_code(lang)
        return normalized in self._installed_languages
    
    def set_default_language(self, lang: str) -> None:
        """设置默认语言。
        
        Args:
            lang: 语言代码
        """
        self._default_lang = self._normalize_lang_code(lang)
    
    def recognize_with_preprocessing(
        self,
        image: ImageData,
        lang: Optional[str] = None,
        preprocessor: Optional[OCRPreprocessor] = None,
    ) -> OCRResult:
        """带预处理的OCR识别。
        
        Args:
            image: 输入图像
            lang: 语言代码
            preprocessor: 预处理器（None使用默认）
            
        Returns:
            OCR识别结果
        """
        if preprocessor is None:
            preprocessor = OCRPreprocessor()
        
        # 预处理图像
        processed_image = preprocessor.preprocess_pipeline(image)
        
        # 执行OCR
        return self.recognize(processed_image, lang)
    
    def recognize_regions(
        self,
        image: ImageData,
        regions: List[Tuple[int, int, int, int]],
        lang: Optional[str] = None,
    ) -> List[OCRResult]:
        """识别指定区域的文本。
        
        Args:
            image: 输入图像
            regions: 区域列表 [(x, y, width, height), ...]
            lang: 语言代码
            
        Returns:
            OCR结果列表
        """
        results = []
        
        for x, y, w, h in regions:
            # 提取区域图像
            region_pixels = {}
            for py in range(y, min(y + h, image.height)):
                for px in range(x, min(x + w, image.width)):
                    region_pixels[(px - x, py - y)] = image.get_pixel(px, py)
            
            region_image = ImageData(w, h, region_pixels)
            
            # 识别区域
            result = self.recognize(region_image, lang)
            result.bounding_box = (x, y, w, h)
            results.append(result)
        
        return results


# ============================================================
# 导出
# ============================================================

__all__ = [
    "ImageData",
    "OCRResult",
    "LanguageDetectionResult",
    "OCRPreprocessor",
    "LanguageDetector",
    "TesseractWrapper",
]