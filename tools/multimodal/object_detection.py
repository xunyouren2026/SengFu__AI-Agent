"""
Object Detection Tool - 物体检测工具
检测图像中的物体
"""

import json
import base64
import urllib.request
import urllib.error
import math
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DetectionModel(Enum):
    """检测模型枚举"""
    YOLO = "yolo"
    FASTER_RCNN = "faster_rcnn"
    SSD = "ssd"
    DETR = "detr"
    CUSTOM = "custom"


@dataclass
class BoundingBox:
    """边界框"""
    x: float
    y: float
    width: float
    height: float
    
    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}
    
    def area(self) -> float:
        return self.width * self.height
    
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)
    
    def intersection(self, other: "BoundingBox") -> Optional["BoundingBox"]:
        """计算交集"""
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        
        if x2 > x1 and y2 > y1:
            return BoundingBox(x1, y1, x2 - x1, y2 - y1)
        return None
    
    def iou(self, other: "BoundingBox") -> float:
        """计算IoU"""
        intersection = self.intersection(other)
        if intersection is None:
            return 0.0
        
        intersection_area = intersection.area()
        union_area = self.area() + other.area() - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def scale(self, factor: float) -> "BoundingBox":
        """缩放边界框"""
        cx, cy = self.center()
        new_width = self.width * factor
        new_height = self.height * factor
        return BoundingBox(
            cx - new_width / 2,
            cy - new_height / 2,
            new_width,
            new_height
        )


@dataclass
class Detection:
    """检测结果"""
    label: str
    confidence: float
    bbox: BoundingBox
    class_id: int = -1
    attributes: Dict[str, Any] = field(default_factory=dict)
    mask: Optional[List[List[Tuple[float, float]]]] = None  # 分割掩码
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "label": self.label,
            "confidence": self.confidence,
            "bbox": self.bbox.to_dict(),
            "class_id": self.class_id,
            "attributes": self.attributes
        }
        if self.mask:
            result["mask"] = self.mask
        return result


@dataclass
class DetectionResult:
    """检测结果集"""
    detections: List[Detection]
    image_width: int
    image_height: int
    processing_time: float = 0.0
    model_used: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "image_width": self.image_width,
            "image_height": self.image_height,
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "metadata": self.metadata
        }
    
    def filter_by_label(self, label: str) -> List[Detection]:
        """按标签过滤"""
        return [d for d in self.detections if d.label == label]
    
    def filter_by_confidence(self, min_confidence: float) -> List[Detection]:
        """按置信度过滤"""
        return [d for d in self.detections if d.confidence >= min_confidence]
    
    def get_labels(self) -> List[str]:
        """获取所有标签"""
        return list(set(d.label for d in self.detections))
    
    def count_by_label(self) -> Dict[str, int]:
        """按标签计数"""
        counts: Dict[str, int] = {}
        for d in self.detections:
            counts[d.label] = counts.get(d.label, 0) + 1
        return counts


@dataclass
class DetectionConfig:
    """检测配置"""
    model: DetectionModel = DetectionModel.YOLO
    model_path: Optional[str] = None
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 100
    classes: Optional[List[str]] = None
    use_nms: bool = True
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    timeout: int = 60


class ObjectDetector:
    """物体检测工具"""
    
    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()
        self._class_names: Dict[int, str] = {}
    
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
    
    def detect(self, image: Union[str, bytes],
               confidence_threshold: Optional[float] = None,
               classes: Optional[List[str]] = None,
               **kwargs) -> DetectionResult:
        """检测图像中的物体"""
        if isinstance(image, str):
            image_base64 = self.load_image_base64(image)
        else:
            image_base64 = self.encode_image(image)
        
        conf_thresh = confidence_threshold or self.config.confidence_threshold
        target_classes = classes or self.config.classes
        
        result = self._call_model(image_base64, conf_thresh, target_classes, **kwargs)
        
        # 应用NMS
        if self.config.use_nms:
            result.detections = self._nms(result.detections, self.config.iou_threshold)
        
        # 限制检测数量
        if len(result.detections) > self.config.max_detections:
            result.detections.sort(key=lambda d: d.confidence, reverse=True)
            result.detections = result.detections[:self.config.max_detections]
        
        return result
    
    def detect_batch(self, images: List[Union[str, bytes]],
                     **kwargs) -> List[DetectionResult]:
        """批量检测"""
        results = []
        for image in images:
            result = self.detect(image, **kwargs)
            results.append(result)
        return results
    
    def detect_region(self, image: Union[str, bytes],
                      x: float, y: float,
                      width: float, height: float,
                      **kwargs) -> DetectionResult:
        """检测图像指定区域"""
        full_result = self.detect(image, **kwargs)
        
        region_detections = []
        for det in full_result.detections:
            det_cx, det_cy = det.bbox.center()
            if (x <= det_cx <= x + width and
                y <= det_cy <= y + height):
                # 调整坐标
                new_bbox = BoundingBox(
                    det.bbox.x - x,
                    det.bbox.y - y,
                    det.bbox.width,
                    det.bbox.height
                )
                region_detections.append(Detection(
                    label=det.label,
                    confidence=det.confidence,
                    bbox=new_bbox,
                    class_id=det.class_id,
                    attributes=det.attributes
                ))
        
        return DetectionResult(
            detections=region_detections,
            image_width=int(width),
            image_height=int(height),
            model_used=full_result.model_used
        )
    
    def _call_model(self, image_base64: str,
                    confidence_threshold: float,
                    classes: Optional[List[str]],
                    **kwargs) -> DetectionResult:
        """调用模型"""
        if self.config.api_endpoint and self.config.api_key:
            return self._call_api(image_base64, confidence_threshold, classes)
        
        return DetectionResult(
            detections=[],
            image_width=0,
            image_height=0,
            model_used=self.config.model.value
        )
    
    def _call_api(self, image_base64: str,
                  confidence_threshold: float,
                  classes: Optional[List[str]]) -> DetectionResult:
        """调用API"""
        payload = {
            "image": image_base64,
            "model": self.config.model.value,
            "confidence_threshold": confidence_threshold,
            "iou_threshold": self.config.iou_threshold,
            "max_detections": self.config.max_detections
        }
        
        if classes:
            payload["classes"] = classes
        
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
                
                detections = []
                for det_data in result.get("detections", []):
                    bbox_data = det_data.get("bbox", {})
                    bbox = BoundingBox(
                        x=bbox_data.get("x", 0),
                        y=bbox_data.get("y", 0),
                        width=bbox_data.get("width", 0),
                        height=bbox_data.get("height", 0)
                    )
                    
                    detections.append(Detection(
                        label=det_data.get("label", ""),
                        confidence=det_data.get("confidence", 0),
                        bbox=bbox,
                        class_id=det_data.get("class_id", -1),
                        attributes=det_data.get("attributes", {})
                    ))
                
                return DetectionResult(
                    detections=detections,
                    image_width=result.get("image_width", 0),
                    image_height=result.get("image_height", 0),
                    processing_time=result.get("processing_time", 0),
                    model_used=result.get("model", self.config.model.value),
                    metadata={"raw_response": result}
                )
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return DetectionResult(
                detections=[],
                image_width=0,
                image_height=0,
                model_used=self.config.model.value,
                metadata={"error": str(e)}
            )
    
    def _nms(self, detections: List[Detection],
             iou_threshold: float) -> List[Detection]:
        """非极大值抑制"""
        if not detections:
            return []
        
        # 按置信度排序
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            remaining = []
            for det in detections:
                if best.label != det.label or best.bbox.iou(det.bbox) < iou_threshold:
                    remaining.append(det)
            
            detections = remaining
        
        return keep
    
    def set_class_names(self, class_names: Dict[int, str]) -> None:
        """设置类别名称"""
        self._class_names = class_names
    
    def get_available_models(self) -> List[str]:
        """获取可用模型"""
        return [m.value for m in DetectionModel]
    
    def visualize(self, detection_result: DetectionResult,
                  draw_labels: bool = True,
                  draw_confidence: bool = True,
                  color_map: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """生成可视化数据"""
        visualizations = []
        
        for det in detection_result.detections:
            vis_data = {
                "bbox": det.bbox.to_dict(),
                "label": det.label if draw_labels else None,
                "confidence": det.confidence if draw_confidence else None,
                "color": (color_map or {}).get(det.label, "#FF0000")
            }
            visualizations.append(vis_data)
        
        return {
            "image_width": detection_result.image_width,
            "image_height": detection_result.image_height,
            "objects": visualizations
        }
    
    def compute_statistics(self, detection_result: DetectionResult) -> Dict[str, Any]:
        """计算统计信息"""
        if not detection_result.detections:
            return {
                "total_count": 0,
                "avg_confidence": 0,
                "labels": {}
            }
        
        confidences = [d.confidence for d in detection_result.detections]
        label_counts = detection_result.count_by_label()
        
        # 计算边界框分布
        areas = [d.bbox.area() for d in detection_result.detections]
        avg_area = sum(areas) / len(areas) if areas else 0
        
        return {
            "total_count": len(detection_result.detections),
            "avg_confidence": sum(confidences) / len(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "labels": label_counts,
            "unique_labels": len(label_counts),
            "avg_bbox_area": avg_area,
            "coverage": sum(areas) / (detection_result.image_width * detection_result.image_height) if detection_result.image_width > 0 else 0
        }
