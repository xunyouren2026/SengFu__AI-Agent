"""
屏幕捕获模块

提供屏幕捕获、OCR文本识别和UI元素检测功能。
使用模拟实现，仅使用Python标准库。
"""

import math
import time
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ScreenRegion:
    """屏幕区域。"""

    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        """区域面积。"""
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        """区域中心点。"""
        return (self.x + self.width // 2, self.y + self.height // 2)

    def contains(self, x: int, y: int) -> bool:
        """判断点是否在区域内。"""
        return (self.x <= x <= self.x + self.width and
                self.y <= y <= self.y + self.height)

    def overlaps(self, other: "ScreenRegion") -> bool:
        """判断是否与另一个区域重叠。"""
        return not (self.x + self.width < other.x or
                    other.x + other.width < self.x or
                    self.y + self.height < other.y or
                    other.y + other.height < self.y)

    def to_dict(self) -> Dict[str, int]:
        """转换为字典。"""
        return {
            "x": self.x, "y": self.y,
            "width": self.width, "height": self.height,
        }


@dataclass
class TextBlock:
    """文本块。"""

    text: str
    region: ScreenRegion
    confidence: float = 1.0
    font_size: int = 12
    is_bold: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "text": self.text,
            "region": self.region.to_dict(),
            "confidence": self.confidence,
            "font_size": self.font_size,
            "is_bold": self.is_bold,
        }


@dataclass
class UIElement:
    """UI元素。"""

    element_type: str  # "button", "input", "link", "text", "image", etc.
    text: str = ""
    region: ScreenRegion = field(default_factory=lambda: ScreenRegion(0, 0, 0, 0))
    attributes: Dict[str, Any] = field(default_factory=dict)
    children: List["UIElement"] = field(default_factory=list)
    parent: Optional["UIElement"] = None
    element_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        result = {
            "type": self.element_type,
            "text": self.text,
            "region": self.region.to_dict(),
            "attributes": self.attributes,
            "id": self.element_id,
        }
        if self.children:
            result["children"] = [c.to_dict() for c in self.children]
        return result

    def find_by_text(self, text: str) -> Optional["UIElement"]:
        """递归查找包含指定文本的元素。"""
        if text.lower() in self.text.lower():
            return self

        for child in self.children:
            found = child.find_by_text(text)
            if found:
                return found

        return None

    def find_by_type(self, element_type: str) -> List["UIElement"]:
        """递归查找指定类型的元素。"""
        results = []
        if self.element_type == element_type:
            results.append(self)

        for child in self.children:
            results.extend(child.find_by_type(element_type))

        return results

    def get_element_tree(self, indent: int = 0) -> str:
        """获取元素树的可视化表示。"""
        prefix = "  " * indent
        result = f"{prefix}<{self.element_type}"
        if self.element_id:
            result += f" id='{self.element_id}'"
        if self.text:
            result += f" text='{self.text[:30]}'"
        result += f" region=({self.region.x},{self.region.y},{self.region.width},{self.region.height})"

        if self.children:
            result += ">"
            for child in self.children:
                result += "\n" + child.get_element_tree(indent + 1)
            result += f"\n{prefix}</{self.element_type}>"
        else:
            result += " />"

        return result


# ============================================================
# ScreenCapture: 屏幕捕获
# ============================================================

class ScreenCapture:
    """屏幕捕获（模拟实现）。

    提供屏幕截图、区域截图、像素颜色获取和模板匹配等功能。
    """

    # 默认屏幕尺寸
    DEFAULT_WIDTH = 1920
    DEFAULT_HEIGHT = 1080

    def __init__(self, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT):
        """初始化屏幕捕获。

        Args:
            width: 屏幕宽度
            height: 屏幕高度
        """
        self._screen_width = width
        self._screen_height = height

        # 模拟的屏幕缓冲区（用颜色值表示）
        self._screen_buffer: Dict[Tuple[int, int], Tuple[int, int, int]] = {}

        # 初始化屏幕缓冲区（默认白色背景）
        self._init_screen_buffer()

        # 模拟的窗口区域
        self._window_regions: List[ScreenRegion] = []

    def _init_screen_buffer(self) -> None:
        """初始化屏幕缓冲区。"""
        # 使用稀疏表示，默认白色
        self._screen_buffer = {}

    def get_screen_size(self) -> Tuple[int, int]:
        """获取屏幕尺寸。

        Returns:
            (width, height)
        """
        return (self._screen_width, self._screen_height)

    def capture_screen(self) -> Dict[str, Any]:
        """截取整个屏幕。

        Returns:
            屏幕截图元数据
        """
        return {
            "width": self._screen_width,
            "height": self._screen_height,
            "timestamp": time.time(),
            "format": "rgba",
            "description": f"屏幕截图 ({self._screen_width}x{self._screen_height})",
            "pixel_count": self._screen_width * self._screen_height,
        }

    def capture_region(self, x: int, y: int, w: int, h: int) -> Dict[str, Any]:
        """截取屏幕区域。

        Args:
            x: 起始X坐标
            y: 起始Y坐标
            w: 宽度
            h: 高度

        Returns:
            区域截图元数据
        """
        # 边界检查
        x = max(0, min(x, self._screen_width - 1))
        y = max(0, min(y, self._screen_height - 1))
        w = min(w, self._screen_width - x)
        h = min(h, self._screen_height - y)

        return {
            "x": x, "y": y, "width": w, "height": h,
            "timestamp": time.time(),
            "format": "rgba",
            "description": f"区域截图 ({w}x{h}) at ({x}, {y})",
        }

    def get_pixel_color(self, x: int, y: int) -> Tuple[int, int, int]:
        """获取指定像素的颜色。

        Args:
            x: X坐标
            y: Y坐标

        Returns:
            (R, G, B) 颜色值
        """
        if 0 <= x < self._screen_width and 0 <= y < self._screen_height:
            return self._screen_buffer.get((x, y), (255, 255, 255))
        return (0, 0, 0)

    def set_pixel_color(self, x: int, y: int, color: Tuple[int, int, int]) -> None:
        """设置像素颜色（用于模拟）。

        Args:
            x: X坐标
            y: Y坐标
            color: (R, G, B) 颜色值
        """
        if 0 <= x < self._screen_width and 0 <= y < self._screen_height:
            self._screen_buffer[(x, y)] = color

    def find_image(
        self,
        template: Dict[str, Any],
        threshold: float = 0.8,
        search_region: Optional[ScreenRegion] = None,
    ) -> List[Dict[str, Any]]:
        """在屏幕中查找图像（模板匹配）。

        使用NCC（归一化互相关）算法进行模板匹配。

        Args:
            template: 模板图像信息 {width, height, pixels}
            threshold: 匹配阈值
            search_region: 搜索区域（可选）

        Returns:
            匹配结果列表 [{x, y, score, width, height}, ...]
        """
        template_w = template.get("width", 0)
        template_h = template.get("height", 0)
        template_pixels = template.get("pixels", {})

        if template_w == 0 or template_h == 0:
            return []

        # 确定搜索范围
        if search_region:
            start_x = search_region.x
            start_y = search_region.y
            end_x = min(search_region.x + search_region.width, self._screen_width)
            end_y = min(search_region.y + search_region.height, self._screen_height)
        else:
            start_x = 0
            start_y = 0
            end_x = self._screen_width
            end_y = self._screen_height

        matches = []
        step = max(1, min(template_w, template_h) // 4)  # 搜索步长

        for y in range(start_y, end_y - template_h + 1, step):
            for x in range(start_x, end_x - template_w + 1, step):
                score = self._compute_ncc(
                    x, y, template_w, template_h, template_pixels
                )
                if score >= threshold:
                    matches.append({
                        "x": x,
                        "y": y,
                        "score": score,
                        "width": template_w,
                        "height": template_h,
                    })

        # 按分数降序排列
        matches.sort(key=lambda m: m["score"], reverse=True)

        # 非极大值抑制
        matches = self._non_max_suppression(matches, overlap_threshold=0.5)

        return matches

    def _compute_ncc(
        self,
        offset_x: int,
        offset_y: int,
        template_w: int,
        template_h: int,
        template_pixels: Dict[Tuple[int, int], Tuple[int, int, int]],
    ) -> float:
        """计算NCC（归一化互相关）分数。

        NCC = sum((T - mean_T) * (I - mean_I)) /
              sqrt(sum((T - mean_T)^2) * sum((I - mean_I)^2))

        Args:
            offset_x: 偏移X
            offset_y: 偏移Y
            template_w: 模板宽度
            template_h: 模板高度
            template_pixels: 模板像素

        Returns:
            NCC分数，范围[-1, 1]
        """
        # 收集模板和图像区域的像素值
        t_values = []
        i_values = []

        for ty in range(template_h):
            for tx in range(template_w):
                t_color = template_pixels.get((tx, ty), (0, 0, 0))
                i_color = self.get_pixel_color(offset_x + tx, offset_y + ty)

                # 使用灰度值
                t_gray = 0.299 * t_color[0] + 0.587 * t_color[1] + 0.114 * t_color[2]
                i_gray = 0.299 * i_color[0] + 0.587 * i_color[1] + 0.114 * i_color[2]

                t_values.append(t_gray)
                i_values.append(i_gray)

        if not t_values:
            return 0.0

        # 计算均值
        t_mean = sum(t_values) / len(t_values)
        i_mean = sum(i_values) / len(i_values)

        # 计算NCC
        numerator = 0.0
        t_var = 0.0
        i_var = 0.0

        for t_val, i_val in zip(t_values, i_values):
            t_diff = t_val - t_mean
            i_diff = i_val - i_mean
            numerator += t_diff * i_diff
            t_var += t_diff * t_diff
            i_var += i_diff * i_diff

        denominator = math.sqrt(t_var * i_var)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _non_max_suppression(
        self,
        matches: List[Dict[str, Any]],
        overlap_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """非极大值抑制。

        去除重叠的匹配框，保留分数最高的。

        Args:
            matches: 匹配结果列表
            overlap_threshold: IoU阈值

        Returns:
            过滤后的匹配列表
        """
        if not matches:
            return []

        # 按分数降序排列
        matches.sort(key=lambda m: m["score"], reverse=True)

        kept = []
        for match in matches:
            is_overlapping = False
            for kept_match in kept:
                iou = self._compute_iou(match, kept_match)
                if iou > overlap_threshold:
                    is_overlapping = True
                    break

            if not is_overlapping:
                kept.append(match)

        return kept

    def _compute_iou(self, a: Dict[str, Any], b: Dict[str, Any]) -> float:
        """计算两个矩形的IoU（交并比）。

        Args:
            a: 矩形A
            b: 矩形B

        Returns:
            IoU值
        """
        # 交集
        x1 = max(a["x"], b["x"])
        y1 = max(a["y"], b["y"])
        x2 = min(a["x"] + a["width"], b["x"] + b["width"])
        y2 = min(a["y"] + a["height"], b["y"] + b["height"])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # 并集
        area_a = a["width"] * a["height"]
        area_b = b["width"] * b["height"]
        union = area_a + area_b - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def get_screen_description(self) -> str:
        """获取屏幕描述（模拟）。"""
        return (f"屏幕尺寸: {self._screen_width}x{self._screen_height}, "
                f"像素数: {self._screen_width * self._screen_height}, "
                f"缓冲像素数: {len(self._screen_buffer)}")


# ============================================================
# OCRReader: OCR阅读器
# ============================================================

class OCRReader:
    """OCR阅读器（模拟实现）。

    提供文本识别、表格识别和文本块提取功能。
    """

    def __init__(self, language: str = "zh", confidence_threshold: float = 0.5):
        """初始化OCR阅读器。

        Args:
            language: 识别语言
            confidence_threshold: 置信度阈值
        """
        self._language = language
        self._confidence_threshold = confidence_threshold

        # 模拟的文本区域
        self._text_regions: List[TextBlock] = []

    def set_text_regions(self, regions: List[TextBlock]) -> None:
        """设置模拟的文本区域。"""
        self._text_regions = regions

    def recognize_text(self, image_region: Optional[ScreenRegion] = None) -> List[TextBlock]:
        """识别图像区域中的文本。

        Args:
            image_region: 图像区域（None表示全屏）

        Returns:
            识别到的文本块列表
        """
        results = []

        for block in self._text_regions:
            # 过滤低置信度
            if block.confidence < self._confidence_threshold:
                continue

            # 区域过滤
            if image_region is not None:
                if not image_region.overlaps(ScreenRegion(
                    block.region.x, block.region.y,
                    block.region.width, block.region.height
                )):
                    continue

            results.append(block)

        # 按位置排序（从上到下，从左到右）
        results.sort(key=lambda b: (b.region.y, b.region.x))

        return results

    def recognize_table(
        self,
        image_region: Optional[ScreenRegion] = None,
    ) -> Dict[str, Any]:
        """识别图像区域中的表格。

        Args:
            image_region: 图像区域

        Returns:
            表格数据 {rows: [[cell, ...], ...], headers: [str, ...]}
        """
        # 获取区域内的文本块
        blocks = self.recognize_text(image_region)

        if not blocks:
            return {"rows": [], "headers": [], "num_rows": 0, "num_cols": 0}

        # 按Y坐标分组为行
        rows = []
        current_row = [blocks[0]]
        row_y = blocks[0].region.y

        for block in blocks[1:]:
            # 如果Y坐标接近，认为是同一行
            if abs(block.region.y - row_y) < 20:
                current_row.append(block)
            else:
                # 按X坐标排序
                current_row.sort(key=lambda b: b.region.x)
                rows.append([b.text for b in current_row])
                current_row = [block]
                row_y = block.region.y

        if current_row:
            current_row.sort(key=lambda b: b.region.x)
            rows.append([b.text for b in current_row])

        # 第一行作为表头
        headers = rows[0] if rows else []
        data_rows = rows[1:] if len(rows) > 1 else []

        return {
            "rows": data_rows,
            "headers": headers,
            "num_rows": len(data_rows),
            "num_cols": max(len(r) for r in rows) if rows else 0,
        }

    def extract_text_blocks(self) -> List[TextBlock]:
        """提取所有文本块及其位置。

        Returns:
            文本块列表
        """
        return self.recognize_text()

    def get_full_text(self, image_region: Optional[ScreenRegion] = None) -> str:
        """获取区域内的完整文本。

        Args:
            image_region: 图像区域

        Returns:
            完整文本
        """
        blocks = self.recognize_text(image_region)
        return "\n".join(b.text for b in blocks)


# ============================================================
# UIElementDetector: UI元素检测
# ============================================================

class UIElementDetector:
    """UI元素检测器（模拟实现）。

    检测屏幕上的按钮、输入框、链接等UI元素。
    """

    def __init__(self):
        """初始化UI元素检测器。"""
        self._elements: List[UIElement] = []
        self._root_element: Optional[UIElement] = None

    def set_elements(self, elements: List[UIElement]) -> None:
        """设置模拟的UI元素列表。"""
        self._elements = elements

        # 构建元素树
        self._root_element = UIElement(
            element_type="root",
            text="",
            region=ScreenRegion(0, 0, 1920, 1080),
        )
        for elem in elements:
            elem.parent = self._root_element
            self._root_element.children.append(elem)

    def detect_buttons(self) -> List[UIElement]:
        """检测所有按钮元素。

        Returns:
            按钮元素列表
        """
        buttons = []
        for elem in self._elements:
            if elem.element_type == "button":
                buttons.append(elem)
        return buttons

    def detect_inputs(self) -> List[UIElement]:
        """检测所有输入框元素。

        Returns:
            输入框元素列表
        """
        inputs = []
        for elem in self._elements:
            if elem.element_type == "input":
                inputs.append(elem)
        return inputs

    def detect_links(self) -> List[UIElement]:
        """检测所有链接元素。

        Returns:
            链接元素列表
        """
        links = []
        for elem in self._elements:
            if elem.element_type == "link":
                links.append(elem)
        return links

    def get_element_tree(self) -> str:
        """获取完整的元素树。

        Returns:
            元素树的字符串表示
        """
        if self._root_element is None:
            return "<root />"

        return self._root_element.get_element_tree()

    def find_element_by_text(self, text: str) -> Optional[UIElement]:
        """按文本查找元素。

        Args:
            text: 要查找的文本

        Returns:
            匹配的元素（第一个匹配）
        """
        if self._root_element is None:
            return None

        return self._root_element.find_by_text(text)

    def find_elements_by_type(self, element_type: str) -> List[UIElement]:
        """按类型查找元素。

        Args:
            element_type: 元素类型

        Returns:
            匹配的元素列表
        """
        if self._root_element is None:
            return []

        return self._root_element.find_by_type(element_type)

    def find_element_at_position(self, x: int, y: int) -> Optional[UIElement]:
        """查找指定位置的元素。

        Args:
            x: X坐标
            y: Y坐标

        Returns:
            该位置最内层的元素
        """
        found = None

        def _search(elements: List[UIElement]) -> Optional[UIElement]:
            nonlocal found
            for elem in elements:
                if elem.region.contains(x, y):
                    found = elem
                    if elem.children:
                        _search(elem.children)

        if self._root_element:
            _search(self._root_element.children)

        return found

    def get_all_elements(self) -> List[UIElement]:
        """获取所有UI元素。

        Returns:
            元素列表
        """
        return list(self._elements)

    def get_interactive_elements(self) -> List[UIElement]:
        """获取所有可交互元素（按钮、输入框、链接等）。

        Returns:
            可交互元素列表
        """
        interactive_types = {"button", "input", "link", "checkbox", "radio", "select", "textarea"}
        return [e for e in self._elements if e.element_type in interactive_types]
