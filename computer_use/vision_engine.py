"""
视觉处理模块

提供特征检测、多尺度模板匹配和YOLO目标检测功能。
仅使用Python标准库实现（模拟接口）。
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


# ============================================================
# 数据结构
# ============================================================

@dataclass
class KeyPoint:
    """关键点。"""
    x: float
    y: float
    response: float = 0.0  # 响应强度
    size: float = 1.0
    angle: float = 0.0
    octave: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "x": self.x,
            "y": self.y,
            "response": self.response,
            "size": self.size,
            "angle": self.angle,
            "octave": self.octave,
        }


@dataclass
class FeatureDescriptor:
    """特征描述符。"""
    keypoint: KeyPoint
    descriptor: List[float]  # 特征向量
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "keypoint": self.keypoint.to_dict(),
            "descriptor": self.descriptor,
        }


@dataclass
class FeatureMatch:
    """特征匹配。"""
    query_idx: int  # 查询图像中的索引
    train_idx: int  # 训练图像中的索引
    distance: float  # 距离
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "query_idx": self.query_idx,
            "train_idx": self.train_idx,
            "distance": self.distance,
        }


@dataclass
class TemplateMatchResult:
    """模板匹配结果。"""
    x: int
    y: int
    width: int
    height: int
    scale: float
    score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "scale": self.scale,
            "score": self.score,
        }


@dataclass
class BoundingBox:
    """边界框。"""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    class_id: int
    class_name: str = ""
    
    @property
    def center(self) -> Tuple[float, float]:
        """获取中心点。"""
        return (self.x + self.width / 2, self.y + self.height / 2)
    
    @property
    def area(self) -> int:
        """获取面积。"""
        return self.width * self.height
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "center": self.center,
            "area": self.area,
        }


@dataclass
class DetectionResult:
    """检测结果。"""
    bounding_boxes: List[BoundingBox]
    image_width: int
    image_height: int
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "bounding_boxes": [bb.to_dict() for bb in self.bounding_boxes],
            "image_width": self.image_width,
            "image_height": self.image_height,
            "processing_time": self.processing_time,
        }
    
    def get_boxes_by_class(self, class_id: int) -> List[BoundingBox]:
        """按类别获取边界框。"""
        return [bb for bb in self.bounding_boxes if bb.class_id == class_id]
    
    def get_boxes_above_confidence(self, threshold: float) -> List[BoundingBox]:
        """获取置信度高于阈值的边界框。"""
        return [bb for bb in self.bounding_boxes if bb.confidence >= threshold]


# ============================================================
# FeatureDetector: 特征检测
# ============================================================

class FeatureDetector:
    """特征检测器。
    
    使用简化的Harris角点检测算法检测图像特征点。
    提供特征点检测、描述符计算和特征匹配功能。
    """
    
    def __init__(self, max_features: int = 500):
        """初始化特征检测器。
        
        Args:
            max_features: 最大特征点数量
        """
        self._max_features = max_features
        self._keypoints: List[KeyPoint] = []
        self._descriptors: List[FeatureDescriptor] = []
    
    def _get_pixel_gray(self, image: Dict[Tuple[int, int], Tuple[int, int, int]], 
                        x: int, y: int, default: int = 0) -> int:
        """获取像素灰度值。"""
        if (x, y) in image:
            r, g, b = image[(x, y)]
            return int(0.299 * r + 0.587 * g + 0.114 * b)
        return default
    
    def detect_features(self, image: Dict[Tuple[int, int], Tuple[int, int, int]],
                       width: int, height: int) -> List[KeyPoint]:
        """检测图像关键点（Harris角点检测）。
        
        Args:
            image: 图像像素字典 {(x, y): (r, g, b), ...}
            width: 图像宽度
            height: 图像高度
            
        Returns:
            关键点列表
        """
        keypoints = []
        
        # Harris角点检测参数
        k = 0.04  # Harris响应参数
        threshold = 1000  # 响应阈值
        
        # 计算每个像素的Harris响应
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                # 计算梯度
                Ix = (self._get_pixel_gray(image, x + 1, y) - 
                      self._get_pixel_gray(image, x - 1, y)) / 2.0
                Iy = (self._get_pixel_gray(image, x, y + 1) - 
                      self._get_pixel_gray(image, x, y - 1)) / 2.0
                
                # 计算结构张量元素
                Ixx = Ix * Ix
                Iyy = Iy * Iy
                Ixy = Ix * Iy
                
                # 计算Harris响应（简化版，使用3x3窗口）
                window_size = 1
                sum_Ixx = sum_Iyy = sum_Ixy = 0.0
                
                for wy in range(-window_size, window_size + 1):
                    for wx in range(-window_size, window_size + 1):
                        px, py = x + wx, y + wy
                        if 0 <= px < width and 0 <= py < height:
                            gx = (self._get_pixel_gray(image, px + 1, py) - 
                                  self._get_pixel_gray(image, px - 1, py)) / 2.0
                            gy = (self._get_pixel_gray(image, px, py + 1) - 
                                  self._get_pixel_gray(image, px, py - 1)) / 2.0
                            sum_Ixx += gx * gx
                            sum_Iyy += gy * gy
                            sum_Ixy += gx * gy
                
                # 计算行列式和迹
                det = sum_Ixx * sum_Iyy - sum_Ixy * sum_Ixy
                trace = sum_Ixx + sum_Iyy
                
                # Harris响应
                response = det - k * trace * trace
                
                if response > threshold:
                    # 计算角度
                    angle = math.degrees(math.atan2(Iy, Ix))
                    
                    keypoint = KeyPoint(
                        x=float(x),
                        y=float(y),
                        response=response,
                        size=3.0,
                        angle=angle,
                    )
                    keypoints.append(keypoint)
        
        # 按响应强度排序，取前max_features个
        keypoints.sort(key=lambda kp: kp.response, reverse=True)
        keypoints = keypoints[:self._max_features]
        
        self._keypoints = keypoints
        return keypoints
    
    def compute_descriptors(self, image: Dict[Tuple[int, int], Tuple[int, int, int]],
                           width: int, height: int,
                           keypoints: Optional[List[KeyPoint]] = None) -> List[FeatureDescriptor]:
        """计算特征描述符（简化版SIFT描述符）。
        
        Args:
            image: 图像像素字典
            width: 图像宽度
            height: 图像高度
            keypoints: 关键点列表（None使用上次检测的关键点）
            
        Returns:
            特征描述符列表
        """
        if keypoints is None:
            keypoints = self._keypoints
        
        descriptors = []
        
        for kp in keypoints:
            # 简化的描述符：使用16x16邻域的梯度直方图
            descriptor = []
            
            x, y = int(kp.x), int(kp.y)
            
            # 4x4的子区域
            for sub_y in range(-8, 8, 4):
                for sub_x in range(-8, 8, 4):
                    # 每个子区域的梯度直方图（8个方向）
                    hist = [0.0] * 8
                    
                    for dy in range(4):
                        for dx in range(4):
                            px = x + sub_x + dx
                            py = y + sub_y + dy
                            
                            if 0 < px < width - 1 and 0 < py < height - 1:
                                gx = (self._get_pixel_gray(image, px + 1, py) - 
                                      self._get_pixel_gray(image, px - 1, py)) / 2.0
                                gy = (self._get_pixel_gray(image, px, py + 1) - 
                                      self._get_pixel_gray(image, px, py - 1)) / 2.0
                                
                                magnitude = math.sqrt(gx * gx + gy * gy)
                                orientation = math.degrees(math.atan2(gy, gx))
                                
                                # 归一化到0-360度
                                if orientation < 0:
                                    orientation += 360
                                
                                # 分配到8个bin
                                bin_idx = int(orientation / 45) % 8
                                hist[bin_idx] += magnitude
                    
                    descriptor.extend(hist)
            
            # 归一化描述符
            norm = math.sqrt(sum(d * d for d in descriptor))
            if norm > 0:
                descriptor = [d / norm for d in descriptor]
            
            descriptors.append(FeatureDescriptor(
                keypoint=kp,
                descriptor=descriptor,
            ))
        
        self._descriptors = descriptors
        return descriptors
    
    def match_features(self, descriptors1: List[FeatureDescriptor],
                      descriptors2: List[FeatureDescriptor],
                      ratio_threshold: float = 0.7) -> List[FeatureMatch]:
        """特征匹配（使用最近邻距离比）。
        
        Args:
            descriptors1: 第一组描述符
            descriptors2: 第二组描述符
            ratio_threshold: 距离比阈值
            
        Returns:
            匹配结果列表
        """
        matches = []
        
        for i, desc1 in enumerate(descriptors1):
            # 找到最近邻和次近邻
            distances = []
            for j, desc2 in enumerate(descriptors2):
                # 计算欧氏距离
                dist = math.sqrt(
                    sum((a - b) ** 2 for a, b in zip(desc1.descriptor, desc2.descriptor))
                )
                distances.append((dist, j))
            
            distances.sort()
            
            if len(distances) >= 2:
                best_dist, best_idx = distances[0]
                second_dist, _ = distances[1]
                
                # 应用距离比测试
                if best_dist < ratio_threshold * second_dist:
                    matches.append(FeatureMatch(
                        query_idx=i,
                        train_idx=best_idx,
                        distance=best_dist,
                    ))
        
        # 按距离排序
        matches.sort(key=lambda m: m.distance)
        return matches
    
    def get_keypoints(self) -> List[KeyPoint]:
        """获取上次检测的关键点。"""
        return list(self._keypoints)
    
    def get_descriptors(self) -> List[FeatureDescriptor]:
        """获取上次计算的描述符。"""
        return list(self._descriptors)


# ============================================================
# MultiScaleSearch: 多尺度搜索
# ============================================================

class MultiScaleSearch:
    """多尺度模板匹配。
    
    在不同尺度上搜索模板图像，找到最佳匹配位置和尺度。
    """
    
    def __init__(self, scales: Optional[List[float]] = None):
        """初始化多尺度搜索。
        
        Args:
            scales: 尺度列表（None使用默认尺度）
        """
        if scales is None:
            self._scales = [0.5 + i * 0.1 for i in range(16)]
        else:
            self._scales = scales
        
        self._last_results: List[TemplateMatchResult] = []
    
    def _resize_image(self, image: Dict[Tuple[int, int], Tuple[int, int, int]],
                     width: int, height: int,
                     scale: float) -> Tuple[Dict[Tuple[int, int], Tuple[int, int, int]], int, int]:
        """缩放图像。"""
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        new_image = {}
        for y in range(new_height):
            for x in range(new_width):
                src_x = min(int(x / scale), width - 1)
                src_y = min(int(y / scale), height - 1)
                
                if (src_x, src_y) in image:
                    new_image[(x, y)] = image[(src_x, src_y)]
        
        return new_image, new_width, new_height
    
    def _compute_ncc(self, image: Dict[Tuple[int, int], Tuple[int, int, int]],
                    template: Dict[Tuple[int, int], Tuple[int, int, int]],
                    x: int, y: int, tw: int, th: int) -> float:
        """计算归一化互相关（NCC）。"""
        i_values = []
        t_values = []
        
        for ty in range(th):
            for tx in range(tw):
                if (tx, ty) in template and (x + tx, y + ty) in image:
                    t_color = template[(tx, ty)]
                    i_color = image[(x + tx, y + ty)]
                    
                    t_gray = 0.299 * t_color[0] + 0.587 * t_color[1] + 0.114 * t_color[2]
                    i_gray = 0.299 * i_color[0] + 0.587 * i_color[1] + 0.114 * i_color[2]
                    
                    t_values.append(t_gray)
                    i_values.append(i_gray)
        
        if not t_values:
            return 0.0
        
        t_mean = sum(t_values) / len(t_values)
        i_mean = sum(i_values) / len(i_values)
        
        numerator = sum((t - t_mean) * (i - i_mean) for t, i in zip(t_values, i_values))
        t_var = sum((t - t_mean) ** 2 for t in t_values)
        i_var = sum((i - i_mean) ** 2 for i in i_values)
        
        denominator = math.sqrt(t_var * i_var)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def search_template(self, image: Dict[Tuple[int, int], Tuple[int, int, int]],
                       image_width: int, image_height: int,
                       template: Dict[Tuple[int, int], Tuple[int, int, int]],
                       template_width: int, template_height: int,
                       scales: Optional[List[float]] = None) -> List[TemplateMatchResult]:
        """多尺度模板匹配。"""
        if scales is None:
            scales = self._scales
        
        all_results = []
        
        for scale in scales:
            scaled_template, sw, sh = self._resize_image(
                template, template_width, template_height, scale
            )
            
            if sw > image_width or sh > image_height:
                continue
            
            step = max(1, min(sw, sh) // 4)
            
            for y in range(0, image_height - sh + 1, step):
                for x in range(0, image_width - sw + 1, step):
                    score = self._compute_ncc(image, scaled_template, x, y, sw, sh)
                    
                    if score > 0.7:
                        all_results.append(TemplateMatchResult(
                            x=x, y=y, width=sw, height=sh, scale=scale, score=score,
                        ))
        
        all_results.sort(key=lambda r: r.score, reverse=True)
        filtered_results = self._non_max_suppression(all_results)
        
        self._last_results = filtered_results
        return filtered_results
    
    def _non_max_suppression(self, results: List[TemplateMatchResult],
                            overlap_threshold: float = 0.5) -> List[TemplateMatchResult]:
        """非极大值抑制。"""
        if not results:
            return []
        
        kept = []
        for result in results:
            is_overlapping = False
            for kept_result in kept:
                iou = self._compute_iou(result, kept_result)
                if iou > overlap_threshold:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                kept.append(result)
        
        return kept
    
    def _compute_iou(self, a: TemplateMatchResult, b: TemplateMatchResult) -> float:
        """计算两个矩形的IoU。"""
        x1 = max(a.x, b.x)
        y1 = max(a.y, b.y)
        x2 = min(a.x + a.width, b.x + b.width)
        y2 = min(a.y + a.height, b.y + b.height)
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area_a = a.width * a.height
        area_b = b.width * b.height
        union = area_a + area_b - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def find_best_match(self) -> Optional[TemplateMatchResult]:
        """找最佳匹配位置和尺度。"""
        if not self._last_results:
            return None
        return self._last_results[0]
    
    def get_last_results(self) -> List[TemplateMatchResult]:
        """获取上次搜索结果。"""
        return list(self._last_results)


# ============================================================
# YOLODetector: YOLO检测器（模拟接口）
# ============================================================

class YOLODetector:
    """YOLO目标检测器（模拟实现）。
    
    提供目标检测接口，预留真实YOLO集成能力。
    当前使用模拟数据演示接口。
    """
    
    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush",
    ]
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.5):
        """初始化YOLO检测器。"""
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._classes = self.COCO_CLASSES
        self._is_loaded = False
        
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, model_path: str) -> bool:
        """加载YOLO模型。"""
        self._is_loaded = True
        return True
    
    def detect(self, image: Dict[Tuple[int, int], Tuple[int, int, int]],
              width: int, height: int,
              classes: Optional[List[int]] = None) -> DetectionResult:
        """目标检测。"""
        start_time = time.time()
        
        bounding_boxes = self._generate_simulated_detections(width, height, classes)
        
        processing_time = time.time() - start_time
        
        return DetectionResult(
            bounding_boxes=bounding_boxes,
            image_width=width,
            image_height=height,
            processing_time=processing_time,
        )
    
    def _generate_simulated_detections(self, width: int, height: int,
                                       classes: Optional[List[int]] = None) -> List[BoundingBox]:
        """生成模拟检测结果。"""
        import random
        
        bounding_boxes = []
        
        # 随机生成1-3个检测结果作为示例
        num_detections = random.randint(0, 3)
        
        for _ in range(num_detections):
            # 随机选择类别
            if classes:
                class_id = random.choice(classes)
            else:
                class_id = random.randint(0, len(self._classes) - 1)
            
            # 随机生成边界框
            bb_width = random.randint(width // 8, width // 3)
            bb_height = random.randint(height // 8, height // 3)
            x = random.randint(0, max(0, width - bb_width))
            y = random.randint(0, max(0, height - bb_height))
            
            confidence = random.uniform(0.5, 0.95)
            
            bounding_boxes.append(BoundingBox(
                x=x,
                y=y,
                width=bb_width,
                height=bb_height,
                confidence=confidence,
                class_id=class_id,
                class_name=self._classes[class_id] if class_id < len(self._classes) else "unknown",
            ))
        
        # 按置信度排序
        bounding_boxes.sort(key=lambda bb: bb.confidence, reverse=True)
        
        return bounding_boxes
    
    def get_bounding_boxes(self, result: DetectionResult) -> List[BoundingBox]:
        """获取边界框列表。"""
        return result.bounding_boxes
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """设置置信度阈值。"""
        self._confidence_threshold = threshold
    
    def get_classes(self) -> List[str]:
        """获取类别名称列表。"""
        return list(self._classes)
    
    def is_model_loaded(self) -> bool:
        """检查模型是否已加载。"""
        return self._is_loaded


# ============================================================
# 导出
# ============================================================

__all__ = [
    "KeyPoint",
    "FeatureDescriptor",
    "FeatureMatch",
    "TemplateMatchResult",
    "BoundingBox",
    "DetectionResult",
    "FeatureDetector",
    "MultiScaleSearch",
    "YOLODetector",
]