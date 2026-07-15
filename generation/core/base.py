"""
AGI Unified Framework - Generation Module - Core Base
=====================================================

生成模型基础架构，包含配置、安全检查、模型注册、回调管理、
采样器、提示词模板、模型适配器和自定义模型构建器等核心组件。

仅使用 Python 标准库，无外部依赖。
"""

from __future__ import annotations

import abc
import copy
import hashlib
import json
import logging
import math
import os
import random
import re
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from enum import Enum
from functools import partial
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union, Sequence
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. GenerationConfig
# ---------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    """生成配置，控制生成行为的各项参数。"""

    model_name: str = "default"
    model_type: str = "autoregressive"  # diffusion / gan / vae / autoregressive / flow
    device: str = "cpu"                 # cpu / cuda
    precision: str = "fp32"             # fp32 / fp16 / int8
    max_seq_len: int = 512
    temperature: float = 1.0            # 0.1 ~ 5.0
    top_k: int = 50
    top_p: float = 0.9
    num_inference_steps: int = 20
    guidance_scale: float = 1.0
    seed: Optional[int] = None
    callback_interval: int = 1
    safety_checker: bool = True

    # --- 序列化 -----------------------------------------------------------

    def to_dict(self) -> dict:
        """将配置转换为字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GenerationConfig":
        """从字典创建配置实例，忽略未知字段。"""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    # --- 校验 -------------------------------------------------------------

    def _validate(self) -> List[str]:
        """返回校验错误列表，空列表表示通过。"""
        errors: List[str] = []
        valid_types = {"diffusion", "gan", "vae", "autoregressive", "flow"}
        if self.model_type not in valid_types:
            errors.append(f"model_type 必须为 {valid_types} 之一，当前: {self.model_type}")
        if self.device not in {"cpu", "cuda"}:
            errors.append(f"device 必须为 cpu/cuda，当前: {self.device}")
        if self.precision not in {"fp32", "fp16", "int8"}:
            errors.append(f"precision 必须为 fp32/fp16/int8，当前: {self.precision}")
        if not (0.1 <= self.temperature <= 5.0):
            errors.append(f"temperature 必须在 [0.1, 5.0] 范围内，当前: {self.temperature}")
        if self.top_k < 1:
            errors.append(f"top_k 必须 >= 1，当前: {self.top_k}")
        if not (0.0 < self.top_p <= 1.0):
            errors.append(f"top_p 必须在 (0, 1] 范围内，当前: {self.top_p}")
        if self.max_seq_len < 1:
            errors.append(f"max_seq_len 必须 >= 1，当前: {self.max_seq_len}")
        if self.num_inference_steps < 1:
            errors.append(f"num_inference_steps 必须 >= 1，当前: {self.num_inference_steps}")
        if self.guidance_scale < 0:
            errors.append(f"guidance_scale 必须 >= 0，当前: {self.guidance_scale}")
        return errors


# ---------------------------------------------------------------------------
# 2. GenerationResult
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """生成结果，封装生成的数据及元信息。"""

    data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0       # 0 ~ 1
    generation_time: float = 0.0     # 秒
    tokens_used: int = 0
    is_safe: bool = True

    def to_dict(self) -> dict:
        """转换为字典。"""
        result = asdict(self)
        # data 可能不可序列化，尝试简单处理
        if not isinstance(result["data"], (int, float, str, bool, list, dict, type(None))):
            result["data"] = str(result["data"])
        return result


# ---------------------------------------------------------------------------
# 3 & 4. SafetyResult / SafetyChecker
# ---------------------------------------------------------------------------

class RiskLevel(Enum):
    """风险等级枚举。"""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SafetyResult:
    """安全检查结果。"""
    is_safe: bool = True
    risk_level: str = "safe"
    categories: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class SafetyChecker:
    """内容安全检查器，支持文本和图像矩阵的安全审查。

    文本安全检查基于正则模式匹配。
    图像安全检查基于颜色分布统计和纹理分析。
    """

    # 默认 NSFW 关键词模式（示例，仅用于演示）
    _DEFAULT_NSFW_PATTERNS: List[str] = [
        r"nude\b", r"pornograph", r"explicit\s+content",
        r"adult\s+content", r"x[-]?rated",
    ]
    _DEFAULT_VIOLENCE_PATTERNS: List[str] = [
        r"kill\s+\w+", r"murder", r"bomb\s+making",
        r"how\s+to\s+make\s+a\s+weapon", r"terrorist",
    ]

    def __init__(self) -> None:
        self._nsfw_patterns: List[re.Pattern] = [
            re.compile(p, re.IGNORECASE) for p in self._DEFAULT_NSFW_PATTERNS
        ]
        self._violence_patterns: List[re.Pattern] = [
            re.compile(p, re.IGNORECASE) for p in self._DEFAULT_VIOLENCE_PATTERNS
        ]

    # --- 文本检查 ---------------------------------------------------------

    def check_text(self, text: str) -> SafetyResult:
        """对文本执行安全检查，返回 SafetyResult。"""
        categories: List[str] = []
        details: Dict[str, Any] = {}
        max_risk = RiskLevel.SAFE

        # NSFW 检查
        nsfw_matches: List[str] = []
        for pat in self._nsfw_patterns:
            found = pat.findall(text)
            if found:
                nsfw_matches.extend(found)
        if nsfw_matches:
            categories.append("nsfw")
            details["nsfw_matches"] = list(set(nsfw_matches))
            max_risk = RiskLevel.HIGH

        # 暴力检查
        violence_matches: List[str] = []
        for pat in self._violence_patterns:
            found = pat.findall(text)
            if found:
                violence_matches.extend(found)
        if violence_matches:
            categories.append("violence")
            details["violence_matches"] = list(set(violence_matches))
            if max_risk.value < RiskLevel.HIGH.value:
                max_risk = RiskLevel.HIGH

        # 文本长度异常检测（极短或极长可能有问题）
        if len(text.strip()) == 0:
            categories.append("empty_content")
            max_risk = RiskLevel.LOW

        return SafetyResult(
            is_safe=(max_risk == RiskLevel.SAFE),
            risk_level=max_risk.value,
            categories=categories,
            details=details,
        )

    # --- 图像检查 ---------------------------------------------------------

    def check_image(self, image_matrix: List[List[List[int]]]) -> SafetyResult:
        """对图像矩阵执行安全检查。

        image_matrix 结构: [height][width][channels] (RGB, 0-255)
        基于:
          1. 颜色分布统计
          2. 皮肤色调比例检测
          3. 可疑纹理模式检测
        """
        stats = self._compute_image_statistics(image_matrix)
        skin_ratio = self._detect_skin_tone(image_matrix)
        suspicious_score = self._detect_suspicious_patterns(image_matrix)

        categories: List[str] = []
        details: Dict[str, Any] = {
            "image_stats": stats,
            "skin_tone_ratio": round(skin_ratio, 4),
            "suspicious_pattern_score": round(suspicious_score, 4),
        }

        risk = RiskLevel.SAFE

        # 皮肤色调比例过高可能指示不适当内容
        if skin_ratio > 0.65:
            categories.append("high_skin_tone")
            risk = RiskLevel.MEDIUM
            if skin_ratio > 0.85:
                risk = RiskLevel.HIGH

        # 可疑纹理模式
        if suspicious_score > 0.7:
            categories.append("suspicious_texture")
            if risk.value < RiskLevel.HIGH.value:
                risk = RiskLevel.HIGH

        # 颜色分布异常（极度单一色调可能指示问题）
        if stats.get("color_variance", 1.0) < 0.01:
            categories.append("abnormal_color_distribution")
            if risk.value < RiskLevel.LOW.value:
                risk = RiskLevel.LOW

        return SafetyResult(
            is_safe=(risk == RiskLevel.SAFE),
            risk_level=risk.value,
            categories=categories,
            details=details,
        )

    def _compute_image_statistics(self, matrix: List[List[List[int]]]) -> dict:
        """计算图像的统计特征：均值、方差、颜色直方图等。"""
        if not matrix or not matrix[0]:
            return {"mean_rgb": [0, 0, 0], "std_rgb": [0, 0, 0], "color_variance": 0.0}

        height = len(matrix)
        width = len(matrix[0])
        channels = len(matrix[0][0]) if matrix[0][0] else 3

        # 逐通道累加
        sum_c = [0.0] * channels
        sum_sq = [0.0] * channels
        total_pixels = height * width

        # 简单颜色直方图 (每通道 8 个 bin)
        hist_bins = 8
        histograms = [[0] * hist_bins for _ in range(channels)]

        for row in matrix:
            for pixel in row:
                for c in range(channels):
                    val = pixel[c] if c < len(pixel) else 0
                    sum_c[c] += val
                    sum_sq[c] += val * val
                    bin_idx = min(val * hist_bins // 256, hist_bins - 1)
                    histograms[c][bin_idx] += 1

        mean_c = [s / total_pixels for s in sum_c]
        variance_c = [
            (sq / total_pixels) - (m * m)
            for sq, m in zip(sum_sq, mean_c)
        ]
        std_c = [math.sqrt(max(v, 0)) for v in variance_c]

        # 颜色方差（归一化到 0~1）
        max_var = 255.0 * 255.0
        avg_var = sum(variance_c) / len(variance_c)
        color_variance = min(avg_var / max_var, 1.0)

        return {
            "mean_rgb": [round(m, 2) for m in mean_c],
            "std_rgb": [round(s, 2) for s in std_c],
            "color_variance": round(color_variance, 4),
            "histograms": histograms,
            "dimensions": [height, width, channels],
        }

    def _detect_skin_tone(self, matrix: List[List[List[int]]]) -> float:
        """检测图像中皮肤色调像素的比例。

        使用经典的 RGB 皮肤检测规则:
          R > 95, G > 40, B > 20
          max(R,G,B) - min(R,G,B) > 15
          |R - G| > 15
          R > G, R > B
        """
        if not matrix or not matrix[0]:
            return 0.0

        total = 0
        skin_count = 0

        for row in matrix:
            for pixel in row:
                total += 1
                if len(pixel) < 3:
                    continue
                r, g, b = pixel[0], pixel[1], pixel[2]
                max_c = max(r, g, b)
                min_c = min(r, g, b)
                if (r > 95 and g > 40 and b > 20
                        and (max_c - min_c) > 15
                        and abs(r - g) > 15
                        and r > g and r > b):
                    skin_count += 1

        return skin_count / total if total > 0 else 0.0

    def _detect_suspicious_patterns(self, matrix: List[List[List[int]]]) -> float:
        """检测图像中的可疑纹理模式。

        通过分析相邻像素差异来检测高频纹理/噪声模式，
        这种模式可能出现在不适当内容中。
        返回 0~1 之间的可疑分数。
        """
        if not matrix or len(matrix) < 2 or not matrix[0] or len(matrix[0]) < 2:
            return 0.0

        height = len(matrix)
        width = len(matrix[0])

        total_diff = 0.0
        diff_count = 0
        high_freq_count = 0

        for y in range(height):
            for x in range(width):
                if x + 1 < width:
                    p1 = matrix[y][x]
                    p2 = matrix[y][x + 1]
                    diff = sum(abs(p1[c] - p2[c]) for c in range(min(len(p1), len(p2))))
                    total_diff += diff
                    diff_count += 1
                    if diff > 100:
                        high_freq_count += 1
                if y + 1 < height:
                    p1 = matrix[y][x]
                    p2 = matrix[y + 1][x]
                    diff = sum(abs(p1[c] - p2[c]) for c in range(min(len(p1), len(p2))))
                    total_diff += diff
                    diff_count += 1
                    if diff > 100:
                        high_freq_count += 1

        if diff_count == 0:
            return 0.0

        avg_diff = total_diff / diff_count
        high_freq_ratio = high_freq_count / diff_count

        # 高频纹理比例和平均差异结合
        suspicious = min(high_freq_ratio * 2.0 + avg_diff / 255.0, 1.0)
        return round(suspicious, 4)


# ---------------------------------------------------------------------------
# 5 & 6. ModelInfo / ModelRegistry
# ---------------------------------------------------------------------------

@dataclass
class ModelInfo:
    """模型元信息。"""
    name: str = ""
    version: str = "1.0.0"
    type: str = "autoregressive"
    size_mb: float = 0.0
    description: str = ""
    capabilities: List[str] = field(default_factory=list)
    author: str = ""
    license: str = "mit"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelInfo":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ModelRegistry:
    """模型注册表，管理模型信息的注册、查询和加载。

    使用 OrderedDict 保持注册顺序。
    """

    def __init__(self) -> None:
        self._models: Dict[str, ModelInfo] = OrderedDict()
        self._loaders: Dict[str, Callable] = {}

    def register(
        self,
        name: str,
        model_info: ModelInfo,
        loader: Optional[Callable] = None,
    ) -> None:
        """注册模型信息及可选的加载器。"""
        if not name or not isinstance(name, str):
            raise ValueError(f"模型名称必须为非空字符串，当前: {name!r}")
        if not isinstance(model_info, ModelInfo):
            raise TypeError(f"model_info 必须为 ModelInfo 实例，当前: {type(model_info)}")
        model_info.name = name
        self._models[name] = model_info
        if loader is not None:
            self._loaders[name] = loader
        logger.info("模型已注册: %s (v%s)", name, model_info.version)

    def unregister(self, name: str) -> None:
        """注销模型。"""
        if name not in self._models:
            raise KeyError(f"模型 '{name}' 未注册")
        del self._models[name]
        self._loaders.pop(name, None)
        logger.info("模型已注销: %s", name)

    def get(self, name: str) -> ModelInfo:
        """获取模型信息。"""
        if name not in self._models:
            raise KeyError(f"模型 '{name}' 未注册")
        return self._models[name]

    def list_models(self, type_filter: Optional[str] = None) -> List[ModelInfo]:
        """列出所有已注册模型，可按类型过滤。"""
        models = list(self._models.values())
        if type_filter is not None:
            models = [m for m in models if m.type == type_filter]
        return models

    def search(self, query: str) -> List[ModelInfo]:
        """按关键词搜索模型（匹配名称、描述、能力标签）。"""
        query_lower = query.lower()
        results: List[ModelInfo] = []
        for info in self._models.values():
            searchable = " ".join([
                info.name, info.description,
                " ".join(info.capabilities), info.author,
            ]).lower()
            if query_lower in searchable:
                results.append(info)
        return results

    def load_model(self, name: str, config: Optional[GenerationConfig] = None) -> Any:
        """使用注册的加载器加载模型。"""
        if name not in self._models:
            raise KeyError(f"模型 '{name}' 未注册")
        if name not in self._loaders:
            raise RuntimeError(f"模型 '{name}' 没有注册加载器")
        loader = self._loaders[name]
        return loader(config)

    @property
    def count(self) -> int:
        """已注册模型数量。"""
        return len(self._models)


# ---------------------------------------------------------------------------
# 7. CallbackManager
# ---------------------------------------------------------------------------

class CallbackManager:
    """生成回调管理器，支持按事件类型注册和触发回调。"""

    def __init__(self) -> None:
        self._callbacks: List[Tuple[str, Callable]] = []

    def register(self, callback: Callable, event: str = "step") -> None:
        """注册回调函数，指定事件类型。"""
        if not callable(callback):
            raise TypeError("callback 必须为可调用对象")
        self._callbacks.append((event, callback))

    def unregister(self, callback: Callable) -> None:
        """注销回调函数。"""
        self._callbacks = [
            (evt, cb) for evt, cb in self._callbacks if cb is not callback
        ]

    def trigger(self, event: str, data: Any = None) -> None:
        """触发指定事件的所有回调。"""
        for evt, cb in self._callbacks:
            if evt == event:
                try:
                    cb(data)
                except Exception as exc:
                    logger.warning("回调执行异常 (event=%s): %s", event, exc)

    def _create_progress_callback(self, total_steps: int) -> Callable:
        """创建一个进度回调函数，自动计算百分比并记录日志。"""
        state = {"current": 0}

        def progress_callback(step_data: Any) -> None:
            state["current"] += 1
            pct = min(state["current"] / total_steps * 100.0, 100.0)
            logger.info(
                "生成进度: %d/%d (%.1f%%)", state["current"], total_steps, pct
            )

        return progress_callback


# ---------------------------------------------------------------------------
# 8. GenerationPipeline (ABC)
# ---------------------------------------------------------------------------

class GenerationPipeline(abc.ABC):
    """生成管线抽象基类。

    子类必须实现 generate() 方法。
    提供 preprocess / postprocess / 安全检查 / 配置校验等通用功能。
    """

    def __init__(self, config: Optional[GenerationConfig] = None) -> None:
        self.config = config or GenerationConfig()
        self._model: Any = None
        self._safety_checker = SafetyChecker()
        self._callback_manager = CallbackManager()
        self._validate_config()

    @abc.abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> GenerationResult:
        """根据提示词生成内容，子类必须实现。"""
        ...

    def preprocess(self, prompt: str) -> dict:
        """提示词预处理：清洗、截断、格式化。"""
        cleaned = prompt.strip()
        # 截断到最大序列长度
        if len(cleaned) > self.config.max_seq_len:
            cleaned = cleaned[: self.config.max_seq_len]
            logger.warning("提示词已截断到 %d 字符", self.config.max_seq_len)
        return {
            "prompt": cleaned,
            "length": len(cleaned),
            "hash": hashlib.md5(cleaned.encode("utf-8")).hexdigest(),
        }

    def postprocess(self, result: GenerationResult) -> GenerationResult:
        """生成结果后处理：安全检查、质量评分。"""
        if self.config.safety_checker:
            result = self._apply_safety(result)
        return result

    def _validate_config(self) -> None:
        """验证配置参数，遇到错误抛出 ValueError。"""
        errors = self.config._validate()
        if errors:
            raise ValueError(f"配置校验失败: {'; '.join(errors)}")

    def _apply_safety(self, result: GenerationResult) -> GenerationResult:
        """对生成结果执行安全检查。"""
        data = result.data
        if isinstance(data, str):
            safety = self._safety_checker.check_text(data)
        elif isinstance(data, list) and self._is_image_matrix(data):
            safety = self._safety_checker.check_image(data)
        else:
            safety = SafetyResult(is_safe=True, risk_level="safe")

        result.is_safe = safety.is_safe
        result.metadata["safety"] = {
            "is_safe": safety.is_safe,
            "risk_level": safety.risk_level,
            "categories": safety.categories,
        }
        return result

    @staticmethod
    def _is_image_matrix(data: Any) -> bool:
        """简单判断数据是否为图像矩阵格式 [H][W][C]。"""
        if not isinstance(data, list) or len(data) == 0:
            return False
        if not isinstance(data[0], list) or len(data[0]) == 0:
            return False
        return isinstance(data[0][0], list)

    def warmup(self) -> None:
        """预热模型，执行一次空推理以初始化内部状态。"""
        logger.info("模型预热中: %s", self.config.model_name)
        start = time.time()
        try:
            _ = self.generate("", **{"_warmup": True})
        except Exception:
            pass  # 预热允许失败
        elapsed = time.time() - start
        logger.info("模型预热完成，耗时 %.3f 秒", elapsed)


# ---------------------------------------------------------------------------
# 9 ~ 12. Sampler / TopKSampler / TopPSampler / TemperatureSampler
# ---------------------------------------------------------------------------

class Sampler(abc.ABC):
    """采样器抽象基类。"""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def set_seed(self, seed: int) -> None:
        """设置随机种子。"""
        self._rng = random.Random(seed)

    @abc.abstractmethod
    def sample(self, logits: List[float], **kwargs: Any) -> int:
        """从 logits 中采样一个 token 索引。"""
        ...

    @staticmethod
    def _softmax(logits: List[float]) -> List[float]:
        """计算 softmax 概率分布。"""
        max_val = max(logits) if logits else 0.0
        exps = [math.exp(x - max_val) for x in logits]
        total = sum(exps)
        return [e / total for e in exps]

    def _sample_from_probs(self, probs: List[float]) -> int:
        """按概率分布采样索引。"""
        r = self._rng.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return i
        return len(probs) - 1


class TopKSampler(Sampler):
    """Top-K 采样器：仅从概率最高的 K 个候选中采样。"""

    def __init__(self, k: int = 50, seed: Optional[int] = None) -> None:
        super().__init__(seed)
        self.k = max(k, 1)

    def sample(self, logits: List[float], **kwargs: Any) -> int:
        """Top-K 采样。"""
        k = kwargs.get("top_k", self.k)
        k = min(k, len(logits))
        if k <= 0:
            return 0

        # 获取 top-k 索引
        indexed = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
        top_k = indexed[:k]

        # 仅保留 top-k 的概率
        probs = self._softmax([v for _, v in top_k])
        chosen_local = self._sample_from_probs(probs)
        return top_k[chosen_local][0]


class TopPSampler(Sampler):
    """Nucleus (Top-P) 采样器：从累积概率达到 P 的最小集合中采样。"""

    def __init__(self, p: float = 0.9, seed: Optional[int] = None) -> None:
        super().__init__(seed)
        self.p = max(0.0, min(p, 1.0))

    def sample(self, logits: List[float], **kwargs: Any) -> int:
        """Top-P (Nucleus) 采样。"""
        p = kwargs.get("top_p", self.p)
        p = max(0.0, min(p, 1.0))

        # 按概率降序排列
        indexed = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
        probs_full = self._softmax([v for _, v in indexed])

        # 累积概率，选择最小集合
        cumulative = 0.0
        nucleus: List[Tuple[int, float]] = []
        for (orig_idx, _), prob in zip(indexed, probs_full):
            nucleus.append((orig_idx, prob))
            cumulative += prob
            if cumulative >= p:
                break

        # 重新归一化
        total_p = sum(p for _, p in nucleus)
        if total_p <= 0:
            return indexed[0][0]
        probs = [p / total_p for _, p in nucleus]
        chosen_local = self._sample_from_probs(probs)
        return nucleus[chosen_local][0]


class TemperatureSampler(Sampler):
    """温度采样器：通过温度参数调节概率分布的平滑程度。

    温度 > 1: 分布更均匀（更多样化）
    温度 < 1: 分布更尖锐（更确定）
    温度 = 1: 等同于原始分布
    """

    def __init__(self, temperature: float = 1.0, seed: Optional[int] = None) -> None:
        super().__init__(seed)
        self.temperature = max(temperature, 0.01)

    def sample(self, logits: List[float], **kwargs: Any) -> int:
        """温度采样。"""
        temp = kwargs.get("temperature", self.temperature)
        temp = max(temp, 0.01)

        # 应用温度
        scaled = [x / temp for x in logits]
        probs = self._softmax(scaled)
        return self._sample_from_probs(probs)


# ---------------------------------------------------------------------------
# 13. BeamSearchSampler / BeamState
# ---------------------------------------------------------------------------

@dataclass
class BeamState:
    """束搜索状态。"""
    tokens: List[int] = field(default_factory=list)
    log_prob: float = 0.0
    score: float = 0.0
    is_finished: bool = False


class BeamSearchSampler:
    """束搜索采样器。

    维护 beam_width 个候选序列，每步扩展所有候选并保留最优的 beam_width 个。
    """

    def __init__(self, beam_width: int = 5, length_penalty: float = 1.0) -> None:
        self.beam_width = max(beam_width, 1)
        self.length_penalty = length_penalty

    def sample(self, logits: List[float], beam_state: Optional[BeamState] = None) -> BeamState:
        """对单步 logits 执行束搜索扩展。

        如果提供了 beam_state，则在其基础上扩展；
        否则初始化新的束状态。
        """
        if beam_state is None:
            beam_state = BeamState()

        expanded = self._expand_beams(beam_state, logits)
        if not expanded:
            return beam_state

        # 按得分排序，返回最优
        expanded.sort(key=lambda b: b.score, reverse=True)
        return expanded[0]

    def _expand_beams(
        self, beam_state: BeamState, logits: List[float]
    ) -> List[BeamState]:
        """从当前束状态扩展出所有可能的下一步候选。"""
        # softmax
        max_val = max(logits) if logits else 0.0
        exps = [math.exp(x - max_val) for x in logits]
        total = sum(exps)
        probs = [e / total for e in exps]

        # 取 top beam_width 个候选
        top_indices = sorted(
            range(len(probs)), key=lambda i: probs[i], reverse=True
        )[: self.beam_width]

        expanded: List[BeamState] = []
        for idx in top_indices:
            token_log_prob = math.log(probs[idx] + 1e-30)
            new_tokens = beam_state.tokens + [idx]
            new_log_prob = beam_state.log_prob + token_log_prob
            new_state = BeamState(
                tokens=new_tokens,
                log_prob=new_log_prob,
                score=self._score_beam(new_log_prob, len(new_tokens)),
            )
            expanded.append(new_state)

        return expanded

    def _score_beam(self, log_prob: float, length: int) -> float:
        """计算束得分，应用长度惩罚。

        score = log_prob / ((5 + length) / 6) ^ length_penalty
        """
        lp = ((5.0 + length) / 6.0) ** self.length_penalty
        return log_prob / lp if lp > 0 else log_prob


# ---------------------------------------------------------------------------
# 14. PromptTemplate
# ---------------------------------------------------------------------------

class PromptTemplate:
    """提示词模板管理器。

    支持模板注册、变量替换、提示词解析和优化。
    """

    # 预置模板
    _BUILTIN_TEMPLATES: Dict[str, str] = {
        "chat": "You are a helpful assistant.\n\nUser: {input}\nAssistant:",
        "instruct": "### Instruction:\n{instruction}\n\n### Response:\n",
        "completion": "{prefix}",
        "summarize": "Please summarize the following text concisely:\n\n{text}\n\nSummary:",
        "translate": "Translate the following text to {target_language}:\n\n{text}\n\nTranslation:",
        "code": "Write a {language} function that {description}.\n\nRequirements:\n{requirements}\n\nCode:",
        "image_gen": "Generate an image of: {description}. Style: {style}. Quality: {quality}.",
    }

    def __init__(self) -> None:
        self._templates: Dict[str, str] = dict(self._BUILTIN_TEMPLATES)
        self._variables: Dict[str, str] = {}

    def register_template(self, name: str, template_str: str) -> None:
        """注册新模板。"""
        if not name or not isinstance(name, str):
            raise ValueError("模板名称必须为非空字符串")
        self._templates[name] = template_str
        logger.info("模板已注册: %s", name)

    def apply(self, template_name: str, variables: Dict[str, str]) -> str:
        """应用模板，替换变量占位符。"""
        if template_name not in self._templates:
            raise KeyError(f"模板 '{template_name}' 未注册。可用模板: {list(self._templates.keys())}")

        template = self._templates[template_name]
        # 合并全局变量（局部变量优先）
        merged = {**self._variables, **variables}
        try:
            return template.format(**merged)
        except KeyError as exc:
            raise ValueError(f"模板变量缺失: {exc}") from exc

    def parse(self, prompt: str) -> dict:
        """解析提示词，提取结构化信息。

        检测:
          - 语言（基于常见语言关键词）
          - 长度（短/中/长）
          - 是否包含指令性关键词
          - 是否包含代码相关关键词
        """
        result: dict = {
            "original": prompt,
            "length": len(prompt),
            "length_category": self._categorize_length(len(prompt)),
            "has_instruction": False,
            "has_code_request": False,
            "detected_language": "unknown",
            "estimated_complexity": "low",
            "keywords": [],
        }

        # 指令性关键词
        instruction_keywords = [
            "please", "please ", "can you", "could you", "i need",
            "i want", "help me", "how to", "what is", "explain",
            "write", "create", "generate", "translate", "summarize",
            "analyze", "describe", "define", "compare", "list",
        ]
        prompt_lower = prompt.lower()
        for kw in instruction_keywords:
            if kw in prompt_lower:
                result["has_instruction"] = True
                result["keywords"].append(kw.strip())
                break

        # 代码关键词
        code_keywords = [
            "function", "class", "code", "program", "script",
            "algorithm", "bug", "debug", "compile", "api",
            "python", "java", "javascript", "c++", "rust",
            "html", "css", "sql", "json", "xml",
        ]
        for kw in code_keywords:
            if kw in prompt_lower:
                result["has_code_request"] = True
                result["keywords"].append(kw)
                break

        # 语言检测（简单启发式）
        lang_patterns = {
            "chinese": r"[\u4e00-\u9fff]",
            "japanese": r"[\u3040-\u309f\u30a0-\u30ff]",
            "korean": r"[\uac00-\ud7af]",
            "arabic": r"[\u0600-\u06ff]",
            "russian": r"[\u0400-\u04ff]",
        }
        for lang, pattern in lang_patterns.items():
            if re.search(pattern, prompt):
                result["detected_language"] = lang
                break
        else:
            if re.search(r"[a-zA-Z]", prompt):
                result["detected_language"] = "english"

        # 复杂度估计
        word_count = len(prompt.split())
        keyword_count = len(result["keywords"])
        if word_count > 50 or keyword_count > 3:
            result["estimated_complexity"] = "high"
        elif word_count > 20 or keyword_count > 1:
            result["estimated_complexity"] = "medium"

        return result

    def _optimize_prompt(self, prompt: str, target_type: str = "general") -> str:
        """优化提示词以提高生成质量。

        优化策略:
          - 去除多余空白
          - 添加明确的结束标记
          - 根据目标类型添加引导语
        """
        # 基础清洗
        optimized = re.sub(r"\s+", " ", prompt).strip()

        if target_type == "code":
            if not optimized.endswith(("\n", "```")):
                optimized += "\n```"
        elif target_type == "creative":
            optimized = f"Be creative and imaginative.\n\n{optimized}"
        elif target_type == "factual":
            optimized = f"Provide accurate and factual information.\n\n{optimized}"
        elif target_type == "concise":
            optimized = f"Be brief and to the point.\n\n{optimized}"

        return optimized

    def get_recommended_params(self, prompt: str) -> dict:
        """根据提示词特征推荐生成参数。"""
        parsed = self.parse(prompt)
        params: dict = {
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.9,
            "max_seq_len": 512,
        }

        if parsed["has_code_request"]:
            params["temperature"] = 0.2
            params["top_p"] = 0.95
            params["max_seq_len"] = 2048
        elif parsed["estimated_complexity"] == "high":
            params["temperature"] = 0.7
            params["max_seq_len"] = 1024
        elif parsed["estimated_complexity"] == "low":
            params["temperature"] = 1.2
            params["max_seq_len"] = 256

        if parsed["length_category"] == "short":
            params["max_seq_len"] = min(params["max_seq_len"], 128)

        return params

    @staticmethod
    def _categorize_length(length: int) -> str:
        """根据字符数分类长度。"""
        if length < 50:
            return "short"
        elif length < 200:
            return "medium"
        return "long"


# ---------------------------------------------------------------------------
# 15. ModelAdapter (ABC)
# ---------------------------------------------------------------------------

class ModelAdapter(abc.ABC):
    """模型适配器抽象基类。

    为不同类型的生成模型提供统一接口。
    """

    @abc.abstractmethod
    def load_model(self, path: str, config: Optional[GenerationConfig] = None) -> Any:
        """从指定路径加载模型。"""
        ...

    @abc.abstractmethod
    def generate(self, prompt: str, params: Optional[dict] = None) -> GenerationResult:
        """使用模型生成内容。"""
        ...

    @abc.abstractmethod
    def get_model_info(self) -> ModelInfo:
        """返回当前模型的元信息。"""
        ...


# ---------------------------------------------------------------------------
# 16 & 17. CustomModelBuilder / CustomModel
# ---------------------------------------------------------------------------

class CustomModel:
    """自定义模型，由 CustomModelBuilder 构建。

    支持简单的前向传播模拟，使用纯 Python 实现基本的全连接层运算。
    """

    def __init__(
        self,
        layers: Optional[List[Any]] = None,
        weights: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        self._layers: List[Any] = layers or []
        self._weights: Dict[str, List[float]] = weights or {}

    def forward(self, input_data: Any) -> Any:
        """执行前向传播。

        支持的输入类型:
          - List[float]: 一维向量，逐层通过全连接层
          - float: 标量输入，转为单元素列表处理
          - int: 整数输入，转为单元素列表处理
        """
        if isinstance(input_data, (int, float)):
            data = [float(input_data)]
        elif isinstance(input_data, list):
            data = [float(x) for x in input_data]
        else:
            raise TypeError(f"不支持的输入类型: {type(input_data)}")

        for layer in self._layers:
            data = self._apply_layer(data, layer)

        return data

    def _apply_layer(self, data: List[float], layer: dict) -> List[float]:
        """应用单层运算。

        支持的层类型:
          - dense: 全连接层 (weights + bias + activation)
          - activation: 激活函数层
          - normalization: 归一化层
          - dropout: 随机丢弃层 (训练时)
        """
        layer_type = layer.get("type", "dense")

        if layer_type == "dense":
            return self._apply_dense(data, layer)
        elif layer_type == "activation":
            return self._apply_activation(data, layer.get("activation", "relu"))
        elif layer_type == "normalization":
            return self._apply_normalization(data, layer)
        elif layer_type == "dropout":
            rate = layer.get("rate", 0.1)
            return self._apply_dropout(data, rate)
        else:
            raise ValueError(f"不支持的层类型: {layer_type}")

    def _apply_dense(self, data: List[float], layer: dict) -> List[float]:
        """全连接层: output = input @ weights + bias"""
        weight_key = layer.get("name", "layer_0")
        weights = self._weights.get(weight_key, [])
        input_size = len(data)
        output_size = layer.get("output_size", input_size)

        if not weights:
            # 如果没有权重，使用恒等映射
            return data[:output_size] if output_size <= input_size else data + [0.0] * (output_size - input_size)

        # 矩阵乘法: weights 是 [output_size x input_size]
        output = []
        for i in range(output_size):
            row_start = i * input_size
            row = weights[row_start: row_start + input_size]
            if len(row) < input_size:
                row = row + [0.0] * (input_size - len(row))
            val = sum(d * w for d, w in zip(data, row))
            # 加偏置
            bias_key = f"{weight_key}_bias"
            biases = self._weights.get(bias_key, [])
            if i < len(biases):
                val += biases[i]
            output.append(val)

        return output

    @staticmethod
    def _apply_activation(data: List[float], activation: str) -> List[float]:
        """应用激活函数。"""
        if activation == "relu":
            return [max(0.0, x) for x in data]
        elif activation == "sigmoid":
            return [1.0 / (1.0 + math.exp(-max(-500, min(500, x)))) for x in data]
        elif activation == "tanh":
            return [math.tanh(x) for x in data]
        elif activation == "leaky_relu":
            alpha = 0.01
            return [x if x > 0 else alpha * x for x in data]
        elif activation == "softmax":
            max_val = max(data) if data else 0.0
            exps = [math.exp(x - max_val) for x in data]
            total = sum(exps)
            return [e / total for e in exps]
        elif activation == "gelu":
            # 近似 GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            result = []
            for x in data:
                inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
                result.append(0.5 * x * (1.0 + math.tanh(inner)))
            return result
        elif activation == "silu" or activation == "swish":
            return [x * (1.0 / (1.0 + math.exp(-max(-500, min(500, x))))) for x in data]
        else:
            return data

    @staticmethod
    def _apply_normalization(data: List[float], layer: dict) -> List[float]:
        """归一化层 (Layer Norm 风格)。"""
        if not data:
            return data

        norm_type = layer.get("norm_type", "layer")
        eps = layer.get("epsilon", 1e-5)

        if norm_type == "layer":
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            std = math.sqrt(variance + eps)
            normalized = [(x - mean) / std for x in data]

            # 应用可学习的缩放和偏移
            gamma = layer.get("gamma", [1.0] * len(data))
            beta = layer.get("beta", [0.0] * len(data))
            return [
                n * (gamma[i] if i < len(gamma) else 1.0)
                + (beta[i] if i < len(beta) else 0.0)
                for i, n in enumerate(normalized)
            ]
        elif norm_type == "minmax":
            min_val = min(data)
            max_val = max(data)
            rng = max_val - min_val
            if rng < eps:
                return [0.0] * len(data)
            return [(x - min_val) / rng for x in data]

        return data

    @staticmethod
    def _apply_dropout(data: List[float], rate: float) -> List[float]:
        """Dropout 层（推理模式，仅缩放输出）。"""
        scale = 1.0 / (1.0 - rate) if rate < 1.0 else 1.0
        return [x * scale for x in data]

    def _initialize_weights(self) -> None:
        """初始化模型权重。"""
        rng = random.Random(42)
        self._weights = {}

        for i, layer in enumerate(self._layers):
            name = layer.get("name", f"layer_{i}")
            layer_type = layer.get("type", "dense")

            if layer_type == "dense":
                input_size = layer.get("input_size", 1)
                output_size = layer.get("output_size", 1)
                # Xavier/Glorot 初始化
                limit = math.sqrt(6.0 / (input_size + output_size))
                weight_count = output_size * input_size
                self._weights[name] = [
                    rng.uniform(-limit, limit) for _ in range(weight_count)
                ]
                self._weights[f"{name}_bias"] = [0.0] * output_size

    def get_parameter_count(self) -> int:
        """返回模型总参数量。"""
        count = 0
        for key, weights in self._weights.items():
            count += len(weights)
        return count

    def get_model_size(self) -> int:
        """返回模型大小（字节），假设每个参数为 float64 (8 bytes)。"""
        return self.get_parameter_count() * 8


class CustomModelBuilder:
    """自定义模型构建器，支持逐层添加和构建。

    使用示例:
        builder = CustomModelBuilder()
        builder.add_layer("dense", {"input_size": 10, "output_size": 64})
        builder.add_layer("activation", {"activation": "relu"})
        builder.add_layer("dense", {"input_size": 64, "output_size": 10})
        model = builder.build()
    """

    def __init__(self) -> None:
        self._layers: List[dict] = []
        self._config: dict = {
            "name": "custom_model",
            "version": "1.0.0",
            "description": "",
        }

    def add_layer(self, layer_type: str, params: Optional[dict] = None) -> "CustomModelBuilder":
        """添加一个层，返回 self 以支持链式调用。"""
        params = params or {}
        layer = {
            "type": layer_type,
            "name": params.pop("name", f"layer_{len(self._layers)}"),
            **params,
        }
        self._layers.append(layer)
        return self

    def build(self) -> CustomModel:
        """构建并返回 CustomModel 实例。"""
        errors = self.validate()
        if errors:
            raise ValueError(f"模型结构验证失败: {'; '.join(errors)}")

        model = CustomModel(layers=copy.deepcopy(self._layers))
        model._initialize_weights()
        logger.info(
            "模型构建完成: %d 层, %d 参数",
            len(self._layers), model.get_parameter_count(),
        )
        return model

    def save(self, path: str) -> None:
        """将模型定义保存为 JSON 文件。"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "config": self._config,
            "layers": self._layers,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("模型定义已保存: %s", path)

    @classmethod
    def load(cls, path: str) -> "CustomModelBuilder":
        """从 JSON 文件加载模型定义。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        builder = cls()
        builder._config = data.get("config", builder._config)
        for layer_def in data.get("layers", []):
            layer_type = layer_def.pop("type", "dense")
            builder.add_layer(layer_type, layer_def)
        logger.info("模型定义已加载: %s", path)
        return builder

    def validate(self) -> List[str]:
        """验证模型结构，返回错误列表。空列表表示通过。"""
        errors: List[str] = []
        prev_output_size: Optional[int] = None

        supported_types = {"dense", "activation", "normalization", "dropout"}
        supported_activations = {
            "relu", "sigmoid", "tanh", "leaky_relu", "softmax",
            "gelu", "silu", "swish", "linear", "none",
        }

        if not self._layers:
            errors.append("模型至少需要一个层")
            return errors

        for i, layer in enumerate(self._layers):
            layer_type = layer.get("type", "")

            if layer_type not in supported_types:
                errors.append(f"第 {i} 层: 不支持的类型 '{layer_type}'")
                continue

            if layer_type == "dense":
                input_size = layer.get("input_size")
                output_size = layer.get("output_size")
                if input_size is not None and input_size < 1:
                    errors.append(f"第 {i} 层: input_size 必须 >= 1")
                if output_size is not None and output_size < 1:
                    errors.append(f"第 {i} 层: output_size 必须 >= 1")
                # 检查与前一层的维度匹配
                if prev_output_size is not None and input_size is not None:
                    if input_size != prev_output_size:
                        errors.append(
                            f"第 {i} 层: input_size ({input_size}) "
                            f"与前一层的 output_size ({prev_output_size}) 不匹配"
                        )
                if output_size is not None:
                    prev_output_size = output_size

            elif layer_type == "activation":
                activation = layer.get("activation", "relu")
                if activation not in supported_activations:
                    errors.append(
                        f"第 {i} 层: 不支持的激活函数 '{activation}'"
                    )

            elif layer_type == "normalization":
                norm_type = layer.get("norm_type", "layer")
                if norm_type not in {"layer", "minmax", "batch"}:
                    errors.append(
                        f"第 {i} 层: 不支持的归一化类型 '{norm_type}'"
                    )

            elif layer_type == "dropout":
                rate = layer.get("rate", 0.1)
                if not (0.0 <= rate < 1.0):
                    errors.append(
                        f"第 {i} 层: dropout rate 必须在 [0, 1) 范围内，当前: {rate}"
                    )

        return errors
