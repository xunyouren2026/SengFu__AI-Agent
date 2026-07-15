"""
YOLO-style Object Detector Simulation Module

Simulates YOLO object detection pipeline:
- Anchor box generation (k-means clustering on aspect ratios)
- Non-maximum suppression (NMS)
- IoU (Intersection over Union) calculation
- Feature pyramid simulation
- Grid cell assignment
- Detection post-processing

Simulates inference with mock model outputs for testing.
Pure Python standard library only.
"""

from __future__ import annotations

import math
import random
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Optional, Any, Callable


@dataclass
class BBox:
    """Bounding box representation."""
    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0
    class_id: int = 0
    class_name: str = ""

    @property
    def x1(self) -> float:
        return self.x

    @property
    def y1(self) -> float:
        return self.y

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    @property
    def cx(self) -> float:
        return self.x + self.width / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.height / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.width * self.height)

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_cxcywh(self) -> Tuple[float, float, float, float]:
        return (self.cx, self.cy, self.width, self.height)

    def iou(self, other: BBox) -> float:
        """Compute IoU with another bounding box."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)

        inter_w = max(0.0, ix2 - ix1)
        inter_h = max(0.0, iy2 - iy1)
        inter_area = inter_w * inter_h

        union_area = self.area + other.area - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area


@dataclass
class Detection:
    """A single object detection result."""
    bbox: BBox
    confidence: float
    class_id: int
    class_name: str
    class_scores: Dict[int, float] = field(default_factory=dict)


@dataclass
class GridCell:
    """A grid cell in the YOLO detection grid."""
    row: int
    col: int
    anchors: List[BBox] = field(default_factory=list)
    objectness: List[float] = field(default_factory=list)
    class_probs: List[Dict[int, float]] = field(default_factory=list)


@dataclass
class FeatureMapLevel:
    """A single level in the feature pyramid."""
    stride: int
    width: int
    height: int
    anchors: List[Tuple[float, float]] = field(default_factory=list)
    grid: List[List[GridCell]] = field(default_factory=list)


# COCO class names (80 classes)
COCO_CLASSES: List[str] = [
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


class AnchorGenerator:
    """
    Generate anchor boxes for YOLO detection.

    Supports k-means clustering on aspect ratios and manual specification.
    """

    def __init__(self, image_size: Tuple[int, int] = (640, 640),
                 num_anchors: int = 9) -> None:
        self.image_size = image_size
        self.num_anchors = num_anchors
        self._anchors: List[Tuple[float, float]] = []

    def generate_kmeans(self, boxes: List[BBox], num_clusters: int = 9,
                        max_iterations: int = 100, seed: int = 42) -> List[Tuple[float, float]]:
        """
        Generate anchor boxes using k-means clustering on ground truth boxes.

        Clusters are based on IoU distance (1 - IoU) rather than Euclidean distance.
        """
        rng = random.Random(seed)
        if len(boxes) < num_clusters:
            return [(100.0, 100.0)] * num_clusters

        # Convert boxes to (width, height) format
        dimensions = [(b.width, b.height) for b in boxes]

        # Initialize centroids randomly
        indices = rng.sample(range(len(dimensions)), num_clusters)
        centroids: List[Tuple[float, float]] = [dimensions[i] for i in indices]

        for iteration in range(max_iterations):
            # Assign each box to nearest centroid
            assignments: List[int] = []
            for w, h in dimensions:
                best_idx = 0
                best_iou = -1.0
                for j, (cw, ch) in enumerate(centroids):
                    iou = self._box_iou_wh(w, h, cw, ch)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j
                assignments.append(best_idx)

            # Update centroids
            new_centroids: List[Tuple[float, float]] = []
            for j in range(num_clusters):
                cluster_dims = [dimensions[i] for i in range(len(dimensions)) if assignments[i] == j]
                if cluster_dims:
                    avg_w = sum(d[0] for d in cluster_dims) / len(cluster_dims)
                    avg_h = sum(d[1] for d in cluster_dims) / len(cluster_dims)
                    new_centroids.append((avg_w, avg_h))
                else:
                    new_centroids.append(centroids[j])

            # Check convergence
            converged = True
            for j in range(num_clusters):
                if abs(new_centroids[j][0] - centroids[j][0]) > 0.01 or \
                   abs(new_centroids[j][1] - centroids[j][1]) > 0.01:
                    converged = False
                    break
            centroids = new_centroids
            if converged:
                break

        # Sort by area
        centroids.sort(key=lambda x: x[0] * x[1])
        self._anchors = centroids
        return centroids

    def _box_iou_wh(self, w1: float, h1: float, w2: float, h2: float) -> float:
        """Compute IoU between two boxes specified by (width, height)."""
        inter_w = min(w1, w2)
        inter_h = min(h1, h2)
        inter_area = inter_w * inter_h
        union_area = w1 * h1 + w2 * h2 - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def set_anchors(self, anchors: List[Tuple[float, float]]) -> None:
        """Manually set anchor boxes."""
        self._anchors = list(anchors)

    def get_anchors(self) -> List[Tuple[float, float]]:
        """Get current anchor boxes."""
        return self._anchors

    def get_anchors_for_scale(self, scale_idx: int,
                               scales: int = 3) -> List[Tuple[float, float]]:
        """Get anchors for a specific feature pyramid scale."""
        if not self._anchors:
            self._generate_default_anchors()
        per_scale = len(self._anchors) // scales
        start = scale_idx * per_scale
        end = start + per_scale
        return self._anchors[start:end]

    def _generate_default_anchors(self) -> None:
        """Generate default YOLO anchors (COCO-style)."""
        self._anchors = [
            (10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326),
        ]


class NonMaxSuppression:
    """
    Non-maximum suppression for filtering overlapping detections.

    Supports standard NMS, soft-NMS, and class-aware NMS.
    """

    def __init__(self, iou_threshold: float = 0.45,
                 score_threshold: float = 0.25,
                 max_detections: int = 300,
                 method: str = "standard") -> None:
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.method = method

    def suppress(self, detections: List[Detection]) -> List[Detection]:
        """Apply NMS to a list of detections."""
        if not detections:
            return []

        # Filter by score threshold
        filtered = [d for d in detections if d.confidence >= self.score_threshold]
        if not filtered:
            return []

        # Sort by confidence descending
        filtered.sort(key=lambda d: d.confidence, reverse=True)

        if self.method == "standard":
            return self._standard_nms(filtered)
        elif self.method == "soft":
            return self._soft_nms(filtered)
        elif self.method == "class_aware":
            return self._class_aware_nms(filtered)
        return self._standard_nms(filtered)

    def _standard_nms(self, detections: List[Detection]) -> List[Detection]:
        """Standard greedy NMS."""
        keep: List[Detection] = []

        while detections:
            best = detections.pop(0)
            keep.append(best)

            remaining: List[Detection] = []
            for det in detections:
                iou = best.bbox.iou(det.bbox)
                if iou < self.iou_threshold:
                    remaining.append(det)
            detections = remaining

            if len(keep) >= self.max_detections:
                break

        return keep

    def _soft_nms(self, detections: List[Detection], sigma: float = 0.5) -> List[Detection]:
        """Soft-NMS: decay scores of overlapping boxes instead of removing."""
        keep: List[Detection] = []
        scores = [d.confidence for d in detections]

        for i in range(len(detections)):
            if scores[i] < self.score_threshold:
                continue

            keep.append(Detection(
                bbox=detections[i].bbox,
                confidence=scores[i],
                class_id=detections[i].class_id,
                class_name=detections[i].class_name,
                class_scores=detections[i].class_scores,
            ))

            for j in range(i + 1, len(detections)):
                if scores[j] < self.score_threshold:
                    continue
                iou = detections[i].bbox.iou(detections[j].bbox)
                decay = math.exp(-(iou * iou) / sigma)
                scores[j] *= decay

            if len(keep) >= self.max_detections:
                break

        return keep

    def _class_aware_nms(self, detections: List[Detection]) -> List[Detection]:
        """Class-aware NMS: only suppress boxes of the same class."""
        by_class: Dict[int, List[Detection]] = {}
        for det in detections:
            by_class.setdefault(det.class_id, []).append(det)

        keep: List[Detection] = []
        for class_id, class_dets in by_class.items():
            class_dets.sort(key=lambda d: d.confidence, reverse=True)
            class_keep: List[Detection] = []
            while class_dets:
                best = class_dets.pop(0)
                class_keep.append(best)
                remaining: List[Detection] = []
                for det in class_dets:
                    if best.bbox.iou(det.bbox) < self.iou_threshold:
                        remaining.append(det)
                class_dets = remaining
            keep.extend(class_keep)

        keep.sort(key=lambda d: d.confidence, reverse=True)
        return keep[:self.max_detections]


class BBoxEncoder:
    """
    Encode/decode bounding boxes between different formats.

    Supports: (x1,y1,x2,y2), (cx,cy,w,h), YOLO format (relative to grid cell).
    """

    @staticmethod
    def xyxy_to_cxcywh(xyxy: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Convert (x1,y1,x2,y2) to (cx,cy,w,h)."""
        x1, y1, x2, y2 = xyxy
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        return (cx, cy, w, h)

    @staticmethod
    def cxcywh_to_xyxy(cxcywh: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Convert (cx,cy,w,h) to (x1,y1,x2,y2)."""
        cx, cy, w, h = cxcywh
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return (x1, y1, x2, y2)

    @staticmethod
    def encode_yolo(cx: float, cy: float, w: float, h: float,
                    grid_col: int, grid_row: int, stride: int,
                    anchor_w: float, anchor_h: float) -> Tuple[float, float, float, float]:
        """
        Encode bounding box in YOLO format.

        Returns (tx, ty, tw, th) where:
        - tx, ty are offsets from the grid cell center (sigmoid-activated)
        - tw, th are log-space ratios to anchor dimensions
        """
        cell_size = stride
        tx = (cx / cell_size) - grid_col
        ty = (cy / cell_size) - grid_row
        tw = math.log(w / anchor_w + 1e-16)
        th = math.log(h / anchor_h + 1e-16)
        return (tx, ty, tw, th)

    @staticmethod
    def decode_yolo(tx: float, ty: float, tw: float, th: float,
                    grid_col: int, grid_row: int, stride: int,
                    anchor_w: float, anchor_h: float) -> Tuple[float, float, float, float]:
        """
        Decode YOLO format back to (cx, cy, w, h).

        Applies sigmoid to tx, ty and exp to tw, th.
        """
        def sigmoid(x: float) -> float:
            if x >= 0:
                return 1.0 / (1.0 + math.exp(-x))
            ex = math.exp(x)
            return ex / (1.0 + ex)

        cx = (sigmoid(tx) + grid_col) * stride
        cy = (sigmoid(ty) + grid_row) * stride
        w = anchor_w * math.exp(tw)
        h = anchor_h * math.exp(th)
        return (cx, cy, w, h)

    @staticmethod
    def scale_bbox(bbox: BBox, from_size: Tuple[int, int],
                   to_size: Tuple[int, int]) -> BBox:
        """Scale a bounding box from one image size to another."""
        sx = to_size[0] / from_size[0]
        sy = to_size[1] / from_size[1]
        return BBox(
            x=bbox.x * sx, y=bbox.y * sy,
            width=bbox.width * sx, height=bbox.height * sy,
            confidence=bbox.confidence,
            class_id=bbox.class_id, class_name=bbox.class_name,
        )


class FeaturePyramid:
    """
    Feature pyramid network simulation.

    Generates multi-scale feature maps at different strides for
    detecting objects at various sizes.
    """

    def __init__(self, image_size: Tuple[int, int] = (640, 640),
                 strides: Optional[List[int]] = None) -> None:
        self.image_size = image_size
        self.strides = strides or [8, 16, 32]
        self.levels: List[FeatureMapLevel] = []
        self._build_pyramid()

    def _build_pyramid(self) -> None:
        """Build the feature pyramid levels."""
        self.levels = []
        for stride in self.strides:
            w = self.image_size[0] // stride
            h = self.image_size[1] // stride
            grid: List[List[GridCell]] = []
            for row in range(h):
                grid_row: List[GridCell] = []
                for col in range(w):
                    grid_row.append(GridCell(row=row, col=col))
                grid.append(grid_row)
            self.levels.append(FeatureMapLevel(
                stride=stride, width=w, height=h, grid=grid,
            ))

    def get_level(self, stride: int) -> Optional[FeatureMapLevel]:
        """Get a feature map level by stride."""
        for level in self.levels:
            if level.stride == stride:
                return level
        return None

    def assign_anchors(self, anchors_per_level: Dict[int, List[Tuple[float, float]]]) -> None:
        """Assign anchor boxes to each pyramid level."""
        for level in self.levels:
            level.anchors = anchors_per_level.get(level.stride, [])

    def get_total_cells(self) -> int:
        """Get total number of grid cells across all levels."""
        return sum(level.width * level.height for level in self.levels)

    def get_responsible_cells(self, bbox: BBox) -> List[Tuple[int, int, int]]:
        """
        Find the grid cells responsible for a bounding box.

        Returns list of (level_idx, row, col) tuples.
        """
        results: List[Tuple[int, int, int]] = []
        for level_idx, level in enumerate(self.levels):
            col = int(bbox.cx / level.stride)
            row = int(bbox.cy / level.stride)
            if 0 <= row < level.height and 0 <= col < level.width:
                results.append((level_idx, row, col))
        return results

    def simulate_feature_extraction(self, image_size: Tuple[int, int]) -> Dict[int, List[List[float]]]:
        """
        Simulate feature extraction at each pyramid level.

        Returns a dict mapping stride to a 2D feature map (dummy values).
        """
        features: Dict[int, List[List[float]]] = {}
        for level in self.levels:
            # Simulate feature values using a deterministic hash
            feat_map: List[List[float]] = []
            for row in range(level.height):
                feat_row: List[float] = []
                for col in range(level.width):
                    # Generate pseudo-random but deterministic features
                    seed_val = level.stride * 1000 + row * 100 + col
                    h = hashlib.md5(str(seed_val).encode()).hexdigest()
                    val = int(h[:8], 16) / 0xFFFFFFFF
                    feat_row.append(val)
                feat_map.append(feat_row)
            features[level.stride] = feat_map
        return features


class DetectionPostProcessor:
    """
    Post-process raw YOLO outputs into final detections.

    Handles box decoding, confidence filtering, and NMS.
    """

    def __init__(self, conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 max_detections: int = 300,
                 class_names: Optional[List[str]] = None) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.class_names = class_names or COCO_CLASSES
        self.nms = NonMaxSuppression(
            iou_threshold=iou_threshold,
            score_threshold=conf_threshold,
            max_detections=max_detections,
        )
        self.encoder = BBoxEncoder()

    def process(self, raw_outputs: Dict[int, List[List[List[float]]]],
                anchors_per_level: Dict[int, List[Tuple[float, float]]],
                image_size: Tuple[int, int] = (640, 640)) -> List[Detection]:
        """
        Process raw YOLO outputs into detections.

        Args:
            raw_outputs: Dict mapping stride to (H, W, [tx,ty,tw,th,obj,cls...]) per cell
            anchors_per_level: Dict mapping stride to anchor dimensions
            image_size: Input image size
        """
        all_detections: List[Detection] = []

        for stride, output in raw_outputs.items():
            anchors = anchors_per_level.get(stride, [])
            if not anchors:
                continue

            for row_idx, row in enumerate(output):
                for col_idx, cell_output in enumerate(row):
                    for anchor_idx, anchor in enumerate(anchors):
                        # Each cell has: [tx, ty, tw, th, objectness, class_probs...]
                        num_classes = len(self.class_names)
                        offset = anchor_idx * (5 + num_classes)
                        if offset + 5 + num_classes > len(cell_output):
                            continue

                        tx = cell_output[offset]
                        ty = cell_output[offset + 1]
                        tw = cell_output[offset + 2]
                        th = cell_output[offset + 3]
                        objectness = cell_output[offset + 4]

                        if objectness < self.conf_threshold:
                            continue

                        # Decode bounding box
                        cx, cy, w, h = self.encoder.decode_yolo(
                            tx, ty, tw, th,
                            col_idx, row_idx, stride,
                            anchor[0], anchor[1],
                        )

                        # Get class scores
                        class_scores: Dict[int, float] = {}
                        for c in range(num_classes):
                            class_scores[c] = cell_output[offset + 5 + c]

                        # Find best class
                        best_class = max(class_scores, key=class_scores.get)
                        best_score = class_scores[best_class]
                        confidence = objectness * best_score

                        if confidence < self.conf_threshold:
                            continue

                        bbox = BBox(
                            x=cx - w / 2.0, y=cy - h / 2.0,
                            width=w, height=h,
                        )
                        detection = Detection(
                            bbox=bbox,
                            confidence=confidence,
                            class_id=best_class,
                            class_name=self.class_names[best_class] if best_class < len(self.class_names) else str(best_class),
                            class_scores=class_scores,
                        )
                        all_detections.append(detection)

        # Apply NMS
        final = self.nms.suppress(all_detections)
        return final


class YOLODetector:
    """
    YOLO-style object detector simulation.

    Provides the full detection pipeline: anchor generation, feature pyramid,
    inference simulation, and post-processing.
    """

    def __init__(self, image_size: Tuple[int, int] = (640, 640),
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 max_detections: int = 300,
                 num_classes: int = 80,
                 class_names: Optional[List[str]] = None) -> None:
        self.image_size = image_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.num_classes = num_classes
        self.class_names = class_names or COCO_CLASSES[:num_classes]

        self.anchor_generator = AnchorGenerator(image_size)
        self.anchor_generator._generate_default_anchors()
        self.feature_pyramid = FeaturePyramid(image_size)
        self.post_processor = DetectionPostProcessor(
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            class_names=self.class_names,
        )

        self._assign_anchors()

    def _assign_anchors(self) -> None:
        """Assign anchors to feature pyramid levels."""
        anchors = self.anchor_generator.get_anchors()
        scales = len(self.feature_pyramid.strides)
        anchors_per_level: Dict[int, List[Tuple[float, float]]] = {}
        per_scale = len(anchors) // scales
        for i, stride in enumerate(self.feature_pyramid.strides):
            start = i * per_scale
            end = start + per_scale
            anchors_per_level[stride] = anchors[start:end]
        self.feature_pyramid.assign_anchors(anchors_per_level)
        self._anchors_per_level = anchors_per_level

    def detect(self, image: Optional[Any] = None,
               num_objects: int = 5,
               seed: int = 42) -> List[Detection]:
        """
        Run detection (simulation mode).

        When no real model is available, generates mock detections
        for testing purposes.
        """
        raw_outputs = self._simulate_inference(num_objects, seed)
        detections = self.post_processor.process(
            raw_outputs, self._anchors_per_level, self.image_size
        )
        return detections

    def detect_with_bboxes(self, ground_truth: List[BBox],
                           noise_std: float = 5.0,
                           seed: int = 42) -> List[Detection]:
        """
        Simulate detection based on ground truth bounding boxes.

        Adds noise to ground truth to simulate imperfect detection.
        """
        rng = random.Random(seed)
        raw_outputs = self._simulate_from_ground_truth(ground_truth, rng, noise_std)
        detections = self.post_processor.process(
            raw_outputs, self._anchors_per_level, self.image_size
        )
        return detections

    def _simulate_inference(self, num_objects: int,
                             seed: int) -> Dict[int, List[List[List[float]]]]:
        """Generate simulated raw YOLO outputs."""
        rng = random.Random(seed)
        raw_outputs: Dict[int, List[List[List[float]]]] = {}

        for level in self.feature_pyramid.levels:
            stride = level.stride
            anchors = self._anchors_per_level.get(stride, [])
            num_anchors = len(anchors)
            output: List[List[List[float]]] = []

            for row in range(level.height):
                row_output: List[List[float]] = []
                for col in range(level.width):
                    cell: List[float] = []
                    for anchor_idx in range(num_anchors):
                        # Generate random values for each anchor
                        tx = rng.gauss(0, 0.5)
                        ty = rng.gauss(0, 0.5)
                        tw = rng.gauss(0, 0.3)
                        th = rng.gauss(0, 0.3)
                        objectness = rng.random() * 0.1  # Low background objectness

                        # Occasionally place a high-confidence detection
                        if rng.random() < num_objects / (level.width * level.height * num_anchors + 1):
                            objectness = rng.uniform(0.7, 0.99)
                            tx = rng.gauss(0, 0.1)
                            ty = rng.gauss(0, 0.1)
                            tw = rng.gauss(0, 0.15)
                            th = rng.gauss(0, 0.15)

                        class_probs = [rng.random() * 0.1 for _ in range(self.num_classes)]
                        if objectness > 0.5:
                            best_class = rng.randint(0, self.num_classes - 1)
                            class_probs[best_class] = rng.uniform(0.7, 0.99)

                        cell.extend([tx, ty, tw, th, objectness] + class_probs)
                    row_output.append(cell)
                output.append(row_output)
            raw_outputs[stride] = output

        return raw_outputs

    def _simulate_from_ground_truth(self, ground_truth: List[BBox],
                                     rng: random.Random,
                                     noise_std: float) -> Dict[int, List[List[List[float]]]]:
        """Generate simulated outputs from ground truth boxes."""
        raw_outputs: Dict[int, List[List[List[float]]]] = {}

        # Initialize with background noise
        for level in self.feature_pyramid.levels:
            stride = level.stride
            anchors = self._anchors_per_level.get(stride, [])
            num_anchors = len(anchors)
            output: List[List[List[float]]] = []

            for row in range(level.height):
                row_output: List[List[float]] = []
                for col in range(level.width):
                    cell: List[float] = []
                    for anchor_idx in range(num_anchors):
                        tx = rng.gauss(0, 0.5)
                        ty = rng.gauss(0, 0.5)
                        tw = rng.gauss(0, 0.3)
                        th = rng.gauss(0, 0.3)
                        objectness = rng.random() * 0.05
                        class_probs = [rng.random() * 0.05 for _ in range(self.num_classes)]
                        cell.extend([tx, ty, tw, th, objectness] + class_probs)
                    row_output.append(cell)
                output.append(row_output)
            raw_outputs[stride] = output

        # Place ground truth objects
        for gt in ground_truth:
            responsible = self.feature_pyramid.get_responsible_cells(gt)
            if not responsible:
                continue

            level_idx, row, col = responsible[0]
            stride = self.feature_pyramid.strides[level_idx]
            anchors = self._anchors_per_level.get(stride, [])

            # Find best matching anchor
            best_anchor_idx = 0
            best_iou = -1.0
            for ai, (aw, ah) in enumerate(anchors):
                iou = self.anchor_generator._box_iou_wh(gt.width, gt.height, aw, ah)
                if iou > best_iou:
                    best_iou = iou
                    best_anchor_idx = ai

            anchor = anchors[best_anchor_idx]
            tx, ty, tw, th = self.encoder.encode_yolo(
                gt.cx + rng.gauss(0, noise_std),
                gt.cy + rng.gauss(0, noise_std),
                gt.width + rng.gauss(0, noise_std),
                gt.height + rng.gauss(0, noise_std),
                col, row, stride, anchor[0], anchor[1],
            )

            cell = raw_outputs[stride][row][col]
            offset = best_anchor_idx * (5 + self.num_classes)
            cell[offset] = tx
            cell[offset + 1] = ty
            cell[offset + 2] = tw
            cell[offset + 3] = th
            cell[offset + 4] = rng.uniform(0.8, 0.99)

            class_probs = [rng.random() * 0.05 for _ in range(self.num_classes)]
            class_id = gt.class_id if gt.class_id < self.num_classes else 0
            class_probs[class_id] = rng.uniform(0.8, 0.99)
            for i in range(self.num_classes):
                cell[offset + 5 + i] = class_probs[i]

        return raw_outputs

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the detector configuration."""
        return {
            "image_size": self.image_size,
            "num_classes": self.num_classes,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "max_detections": self.max_detections,
            "strides": self.feature_pyramid.strides,
            "anchors": self.anchor_generator.get_anchors(),
            "total_grid_cells": self.feature_pyramid.get_total_cells(),
            "class_names": self.class_names,
        }
