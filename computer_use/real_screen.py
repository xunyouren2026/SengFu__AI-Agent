"""
真实屏幕截图和OCR - 基于mss和pytesseract
替换原有的模拟实现
"""

import mss
import mss.tools
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import time
import logging
import io
import base64

logger = logging.getLogger(__name__)


@dataclass
class ScreenRegion:
    """屏幕区域"""
    x: int
    y: int
    width: int
    height: int
    
    def to_dict(self) -> Dict:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height
        }
    
    @property
    def center(self) -> Tuple[int, int]:
        """中心点坐标"""
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass
class TextBlock:
    """文本块"""
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence
        }


@dataclass
class OCRResult:
    """OCR结果"""
    text: str
    blocks: List[TextBlock]
    language: str
    confidence: float
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "blocks": [b.to_dict() for b in self.blocks],
            "language": self.language,
            "confidence": self.confidence
        }


class RealScreenCapture:
    """
    真实屏幕截图 - 基于mss
    
    功能：
    - 全屏截图
    - 区域截图
    - 窗口截图
    - 多显示器支持
    """
    
    def __init__(self):
        self.sct = mss.mss()
        self.monitors = self.sct.monitors
        logger.info(f"屏幕捕获初始化 - 检测到 {len(self.monitors) - 1} 个显示器")
        
    def get_screen_size(self, monitor_idx: int = 0) -> Tuple[int, int]:
        """
        获取屏幕尺寸
        
        Args:
            monitor_idx: 显示器索引（0=主屏，1+=扩展屏）
            
        Returns:
            (width, height)
        """
        if monitor_idx >= len(self.monitors):
            monitor_idx = 0
        monitor = self.monitors[monitor_idx]
        return (monitor["width"], monitor["height"])
    
    def capture_screen(self, monitor_idx: int = 0) -> Image.Image:
        """
        全屏截图
        
        Args:
            monitor_idx: 显示器索引
            
        Returns:
            PIL Image对象
        """
        if monitor_idx >= len(self.monitors):
            monitor_idx = 0
            
        screenshot = self.sct.grab(self.monitors[monitor_idx])
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        return img
    
    def capture_region(self, x: int, y: int, width: int, height: int,
                      monitor_idx: int = 0) -> Image.Image:
        """
        区域截图
        
        Args:
            x: 区域左上角X坐标
            y: 区域左上角Y坐标
            width: 区域宽度
            height: 区域高度
            monitor_idx: 显示器索引
            
        Returns:
            PIL Image对象
        """
        region = {
            "left": x,
            "top": y,
            "width": width,
            "height": height,
            "mon": monitor_idx
        }
        screenshot = self.sct.grab(region)
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        return img
    
    def capture_all_monitors(self) -> List[Image.Image]:
        """
        截图所有显示器
        
        Returns:
            Image对象列表
        """
        images = []
        for i in range(1, len(self.monitors)):  # 跳过monitors[0]（所有屏幕的并集）
            img = self.capture_screen(i)
            images.append(img)
        return images
    
    def get_pixel_color(self, x: int, y: int, monitor_idx: int = 0) -> Tuple[int, int, int]:
        """
        获取指定位置像素颜色
        
        Args:
            x: X坐标
            y: Y坐标
            monitor_idx: 显示器索引
            
        Returns:
            (R, G, B)
        """
        img = self.capture_region(x, y, 1, 1, monitor_idx)
        return img.getpixel((0, 0))
    
    def find_image(self, template_path: str, confidence: float = 0.8,
                   region: Optional[ScreenRegion] = None) -> List[ScreenRegion]:
        """
        在屏幕上查找图像（模板匹配）
        
        Args:
            template_path: 模板图像路径
            confidence: 匹配阈值
            region: 搜索区域（None表示全屏）
            
        Returns:
            匹配区域列表
        """
        # 加载模板
        template = Image.open(template_path).convert("RGB")
        template_np = np.array(template)
        template_gray = cv2.cvtColor(template_np, cv2.COLOR_RGB2GRAY)
        
        # 截图
        if region:
            screenshot = self.capture_region(region.x, region.y, region.width, region.height)
            offset_x, offset_y = region.x, region.y
        else:
            screenshot = self.capture_screen()
            offset_x, offset_y = 0, 0
            
        screenshot_np = np.array(screenshot)
        screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2GRAY)
        
        # 模板匹配
        result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= confidence)
        
        # 非极大值抑制
        matches = []
        h, w = template_gray.shape
        for pt in zip(*locations[::-1]):
            matches.append({
                "x": pt[0] + offset_x,
                "y": pt[1] + offset_y,
                "width": w,
                "height": h,
                "confidence": float(result[pt[1], pt[0]])
            })
        
        # 合并重叠区域
        filtered_matches = self._non_max_suppression(matches, overlap_thresh=0.3)
        
        return [ScreenRegion(m["x"], m["y"], m["width"], m["height"]) 
                for m in filtered_matches]
    
    def _non_max_suppression(self, boxes: List[Dict], overlap_thresh: float = 0.3) -> List[Dict]:
        """非极大值抑制"""
        if not boxes:
            return []
        
        # 按置信度排序
        boxes = sorted(boxes, key=lambda x: x["confidence"], reverse=True)
        pick = []
        
        while len(boxes) > 0:
            current = boxes[0]
            pick.append(current)
            
            # 计算与其他框的重叠
            rest = []
            for box in boxes[1:]:
                iou = self._calculate_iou(current, box)
                if iou < overlap_thresh:
                    rest.append(box)
            boxes = rest
        
        return pick
    
    def _calculate_iou(self, box1: Dict, box2: Dict) -> float:
        """计算IoU"""
        x1 = max(box1["x"], box2["x"])
        y1 = max(box1["y"], box2["y"])
        x2 = min(box1["x"] + box1["width"], box2["x"] + box2["width"])
        y2 = min(box1["y"] + box1["height"], box2["y"] + box2["height"])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = box1["width"] * box1["height"]
        area2 = box2["width"] * box2["height"]
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def to_base64(self, image: Image.Image) -> str:
        """将图像转换为base64字符串"""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    def save(self, image: Image.Image, path: str) -> None:
        """保存截图"""
        image.save(path)
        logger.info(f"截图已保存: {path}")


class RealOCR:
    """
    真实OCR - 基于pytesseract
    
    功能：
    - 图像文字识别
    - 多语言支持
    - 表格识别
    """
    
    def __init__(self, lang: str = "chi_sim+eng"):
        """
        初始化OCR
        
        Args:
            lang: 语言包（如 "chi_sim+eng" 表示中文简体+英文）
        """
        self.lang = lang
        self._check_tesseract()
        
    def _check_tesseract(self):
        """检查Tesseract是否安装"""
        try:
            pytesseract.get_tesseract_version()
            logger.info(f"Tesseract版本: {pytesseract.get_tesseract_version()}")
        except Exception as e:
            logger.warning(f"Tesseract可能未正确安装: {e}")
            logger.warning("请安装Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
    
    def recognize(self, image: Union[Image.Image, str, np.ndarray],
                  lang: Optional[str] = None) -> OCRResult:
        """
        识别图像中的文字
        
        Args:
            image: PIL Image、图像路径或numpy数组
            lang: 语言（None使用默认）
            
        Returns:
            OCR结果
        """
        # 加载图像
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 转换为RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 识别语言
        lang = lang or self.lang
        
        # OCR识别
        data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
        
        # 解析结果
        blocks = []
        full_text_parts = []
        total_confidence = []
        
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 0:  # 过滤低置信度
                text = data['text'][i].strip()
                if text:
                    block = TextBlock(
                        text=text,
                        x=data['left'][i],
                        y=data['top'][i],
                        width=data['width'][i],
                        height=data['height'][i],
                        confidence=float(data['conf'][i]) / 100.0
                    )
                    blocks.append(block)
                    full_text_parts.append(text)
                    total_confidence.append(data['conf'][i])
        
        # 计算平均置信度
        avg_confidence = sum(total_confidence) / len(total_confidence) / 100.0 if total_confidence else 0
        
        return OCRResult(
            text=" ".join(full_text_parts),
            blocks=blocks,
            language=lang,
            confidence=avg_confidence
        )
    
    def recognize_text_only(self, image: Union[Image.Image, str, np.ndarray],
                           lang: Optional[str] = None) -> str:
        """
        只返回识别到的文字（简化版）
        
        Args:
            image: 图像
            lang: 语言
            
        Returns:
            识别到的文字
        """
        result = self.recognize(image, lang)
        return result.text
    
    def recognize_table(self, image: Union[Image.Image, str, np.ndarray],
                       lang: Optional[str] = None) -> List[List[str]]:
        """
        识别表格
        
        Args:
            image: 图像
            lang: 语言
            
        Returns:
            表格数据（二维列表）
        """
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        lang = lang or self.lang
        
        # 使用Tesseract的表格识别
        data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
        
        # 按行分组
        rows = {}
        for i, text in enumerate(data['text']):
            if int(data['conf'][i]) > 0 and text.strip():
                row_num = data['block_num'][i]
                if row_num not in rows:
                    rows[row_num] = []
                rows[row_num].append(text.strip())
        
        # 转换为表格格式
        table = []
        for row_num in sorted(rows.keys()):
            table.append(rows[row_num])
        
        return table
    
    def preprocess_and_recognize(self, image: Union[Image.Image, str, np.ndarray],
                                 lang: Optional[str] = None) -> OCRResult:
        """
        预处理+识别（提高准确率）
        
        Args:
            image: 图像
            lang: 语言
            
        Returns:
            OCR结果
        """
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 预处理
        processed = self._preprocess(image)
        
        # 识别
        return self.recognize(processed, lang)
    
    def _preprocess(self, image: Image.Image) -> Image.Image:
        """
        图像预处理（提高OCR准确率）
        
        步骤：
        1. 转为灰度
        2. 增强对比度
        3. 去噪
        4. 二值化
        """
        # 转为灰度
        img = image.convert('L')
        
        # 增强对比度
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        
        # 去噪
        img = img.filter(ImageFilter.MedianFilter(size=3))
        
        # 二值化
        threshold = 128
        img = img.point(lambda x: 0 if x < threshold else 255, '1')
        
        return img.convert('RGB')
    
    def get_available_languages(self) -> List[str]:
        """获取可用的语言包"""
        try:
            langs = pytesseract.get_languages()
            return langs
        except Exception as e:
            logger.error(f"获取语言包失败: {e}")
            return []


class VisionAgent:
    """
    视觉智能体 - 结合截图、OCR和大模型
    
    功能：
    - 截图并理解屏幕内容
    - 识别UI元素
    - 生成操作指令
    """
    
    def __init__(self, llm_client=None):
        self.screen = RealScreenCapture()
        self.ocr = RealOCR()
        self.llm = llm_client
        logger.info("视觉智能体初始化完成")
    
    def see(self, region: Optional[ScreenRegion] = None) -> Dict:
        """
        "看"屏幕 - 截图+OCR
        
        Args:
            region: 区域（None表示全屏）
            
        Returns:
            包含截图和OCR结果的字典
        """
        # 截图
        if region:
            image = self.screen.capture_region(region.x, region.y, region.width, region.height)
        else:
            image = self.screen.capture_screen()
        
        # OCR
        ocr_result = self.ocr.recognize(image)
        
        return {
            "image": image,
            "ocr": ocr_result,
            "base64": self.screen.to_base64(image)
        }
    
    def find_text(self, text: str, region: Optional[ScreenRegion] = None) -> Optional[ScreenRegion]:
        """
        在屏幕上查找文字
        
        Args:
            text: 要查找的文字
            region: 搜索区域
            
        Returns:
            文字所在区域（未找到返回None）
        """
        result = self.see(region)
        
        for block in result["ocr"].blocks:
            if text.lower() in block.text.lower():
                return ScreenRegion(block.x, block.y, block.width, block.height)
        
        return None
    
    def describe_screen(self) -> str:
        """
        描述当前屏幕内容（需要大模型支持）
        
        Returns:
            屏幕内容描述
        """
        if not self.llm:
            return "未配置大模型客户端"
        
        result = self.see()
        
        # 构建提示词
        prompt = f"""
请描述以下屏幕截图的内容：

OCR识别到的文字：
{result['ocr'].text}

请描述：
1. 这是什么应用程序/网页？
2. 当前界面有哪些主要元素？
3. 可以进行哪些操作？
"""
        
        # 调用大模型（这里简化处理，实际需要传递图片）
        return self.llm.generate(prompt)


# 便捷函数
def screenshot(path: Optional[str] = None) -> Image.Image:
    """
    快速截图
    
    Args:
        path: 保存路径（None不保存）
        
    Returns:
        PIL Image
    """
    cap = RealScreenCapture()
    img = cap.capture_screen()
    
    if path:
        cap.save(img, path)
    
    return img


def screenshot_region(x: int, y: int, width: int, height: int,
                     path: Optional[str] = None) -> Image.Image:
    """
    快速区域截图
    
    Args:
        x, y, width, height: 区域坐标
        path: 保存路径
        
    Returns:
        PIL Image
    """
    cap = RealScreenCapture()
    img = cap.capture_region(x, y, width, height)
    
    if path:
        cap.save(img, path)
    
    return img


# 测试代码
if __name__ == "__main__":
    print("测试真实屏幕截图和OCR...")
    
    # 测试截图
    screen = RealScreenCapture()
    print(f"屏幕尺寸: {screen.get_screen_size()}")
    
    # 全屏截图
    img = screen.capture_screen()
    print(f"截图尺寸: {img.size}")
    
    # 测试OCR
    try:
        ocr = RealOCR()
        result = ocr.recognize(img)
        print(f"OCR识别文字: {result.text[:100]}...")
        print(f"置信度: {result.confidence:.2%}")
    except Exception as e:
        print(f"OCR测试失败（可能未安装Tesseract）: {e}")
    
    print("✅ 屏幕捕获模块测试完成！")
