"""
Anomaly Detector Module

异常检测器实现，提供自动发现问题、根因分析和趋势预测功能。
"""

from __future__ import annotations

import time
import statistics
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """
    异常检测结果
    
    Attributes:
        is_anomaly: 是否异常
        score: 异常分数
        confidence: 置信度
        description: 描述
        timestamp: 时间戳
    """
    is_anomaly: bool
    score: float
    confidence: float
    description: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "is_anomaly": self.is_anomaly,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "description": self.description,
            "timestamp": self.timestamp
        }


@dataclass
class RootCauseResult:
    """
    根因分析结果
    
    Attributes:
        root_causes: 根因列表
        confidence: 置信度
        analysis_time_ms: 分析时间（毫秒）
    """
    root_causes: List[str] = field(default_factory=list)
    confidence: float = 0.0
    analysis_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "root_causes": self.root_causes,
            "confidence": round(self.confidence, 4),
            "analysis_time_ms": round(self.analysis_time_ms, 2)
        }


@dataclass
class PredictionResult:
    """
    预测结果
    
    Attributes:
        values: 预测值列表
        timestamps: 时间戳列表
        confidence_intervals: 置信区间
        confidence: 置信度
    """
    values: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    confidence_intervals: List[Tuple[float, float]] = field(default_factory=list)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "values": [round(v, 4) for v in self.values],
            "timestamps": self.timestamps,
            "confidence_intervals": [(round(l, 4), round(u, 4)) for l, u in self.confidence_intervals],
            "confidence": round(self.confidence, 4)
        }


class AnomalyDetector:
    """
    异常检测器
    
    自动发现系统中的异常。
    
    Example:
        >>> detector = AnomalyDetector()
        >>> 
        >>> # Add data points
        >>> for value in data_stream:
        ...     result = detector.detect(value)
        ...     if result.is_anomaly:
        ...         print(f"Anomaly detected: {result.description}")
    """
    
    def __init__(
        self,
        window_size: int = 100,
        threshold: float = 3.0,
        method: str = "zscore"
    ):
        """
        初始化异常检测器
        
        Args:
            window_size: 窗口大小
            threshold: 检测阈值
            method: 检测方法 (zscore, iqr, mad)
        """
        self._window_size = window_size
        self._threshold = threshold
        self._method = method
        self._history: List[float] = []
        self._lock = threading.Lock()
    
    def detect(self, value: float) -> AnomalyResult:
        """
        检测异常
        
        Args:
            value: 当前值
            
        Returns:
            检测结果
        """
        with self._lock:
            self._history.append(value)
            if len(self._history) > self._window_size:
                self._history = self._history[-self._window_size:]
            
            if len(self._history) < 10:
                return AnomalyResult(
                    is_anomaly=False,
                    score=0.0,
                    confidence=0.0,
                    description="Insufficient data"
                )
            
            is_anomaly, score = self._detect_internal(value)
        
        confidence = min(1.0, len(self._history) / self._window_size)
        
        if is_anomaly:
            description = f"Anomaly detected: value={value:.4f}, score={score:.4f}"
        else:
            description = "Normal"
        
        return AnomalyResult(
            is_anomaly=is_anomaly,
            score=score,
            confidence=confidence,
            description=description
        )
    
    def _detect_internal(self, value: float) -> Tuple[bool, float]:
        """内部检测逻辑"""
        if self._method == "zscore":
            return self._zscore_detect(value)
        elif self._method == "iqr":
            return self._iqr_detect(value)
        elif self._method == "mad":
            return self._mad_detect(value)
        else:
            return False, 0.0
    
    def _zscore_detect(self, value: float) -> Tuple[bool, float]:
        """Z-Score检测"""
        mean = statistics.mean(self._history)
        std = statistics.stdev(self._history) if len(self._history) > 1 else 0
        
        if std == 0:
            return False, 0.0
        
        zscore = abs((value - mean) / std)
        return zscore > self._threshold, zscore
    
    def _iqr_detect(self, value: float) -> Tuple[bool, float]:
        """IQR检测"""
        sorted_history = sorted(self._history)
        q1_idx = len(sorted_history) // 4
        q3_idx = 3 * len(sorted_history) // 4
        q1 = sorted_history[q1_idx]
        q3 = sorted_history[q3_idx]
        iqr = q3 - q1
        
        lower = q1 - self._threshold * iqr
        upper = q3 + self._threshold * iqr
        
        is_anomaly = value < lower or value > upper
        score = abs(value - (q1 + q3) / 2) / iqr if iqr > 0 else 0
        
        return is_anomaly, score
    
    def _mad_detect(self, value: float) -> Tuple[bool, float]:
        """MAD检测"""
        median = statistics.median(self._history)
        mad = statistics.median([abs(x - median) for x in self._history])
        
        if mad == 0:
            return False, 0.0
        
        modified_z = 0.6745 * (value - median) / mad
        return abs(modified_z) > self._threshold, abs(modified_z)


class RootCauseAnalyzer:
    """
    根因分析器
    
    分析问题的根本原因。
    
    Example:
        >>> analyzer = RootCauseAnalyzer()
        >>> 
        >>> # Analyze an issue
        >>> result = analyzer.analyze(
        ...     symptoms=["high_latency", "error_spike"],
        ...     metrics=metrics_data
        ... )
        >>> print(result.root_causes)
    """
    
    def __init__(self):
        self._knowledge_base: Dict[str, List[str]] = {
            "high_latency": ["database_slowdown", "network_congestion", "cpu_throttling"],
            "error_spike": ["dependency_failure", "rate_limiting", "deployment_issue"],
            "memory_leak": ["unclosed_connections", "caching_issue", "infinite_loop"],
            "cpu_high": ["infinite_loop", "heavy_computation", "resource_contention"]
        }
    
    def analyze(
        self,
        symptoms: List[str],
        metrics: Optional[Dict[str, Any]] = None
    ) -> RootCauseResult:
        """
        分析根因
        
        Args:
            symptoms: 症状列表
            metrics: 相关指标
            
        Returns:
            根因分析结果
        """
        start = time.time()
        
        root_causes: Set[str] = set()
        
        for symptom in symptoms:
            causes = self._knowledge_base.get(symptom, [])
            root_causes.update(causes)
        
        # Simple correlation analysis
        if metrics:
            if metrics.get("db_query_time", 0) > 100:
                root_causes.add("database_slowdown")
            if metrics.get("network_errors", 0) > 0:
                root_causes.add("network_congestion")
            if metrics.get("memory_usage", 0) > 90:
                root_causes.add("memory_leak")
        
        elapsed = (time.time() - start) * 1000
        
        return RootCauseResult(
            root_causes=list(root_causes),
            confidence=0.7 if root_causes else 0.0,
            analysis_time_ms=elapsed
        )
    
    def add_knowledge(self, symptom: str, causes: List[str]) -> None:
        """
        添加知识
        
        Args:
            symptom: 症状
            causes: 可能原因
        """
        self._knowledge_base[symptom] = causes


class TrendPredictor:
    """
    趋势预测器
    
    预测指标的未来趋势。
    
    Example:
        >>> predictor = TrendPredictor()
        >>> 
        >>> # Add historical data
        >>> for value in historical_data:
        ...     predictor.add_point(value)
        >>> 
        >>> # Predict future
        >>> prediction = predictor.predict(steps=10)
    """
    
    def __init__(self, window_size: int = 50):
        """
        初始化趋势预测器
        
        Args:
            window_size: 窗口大小
        """
        self._window_size = window_size
        self._values: List[float] = []
        self._timestamps: List[float] = []
        self._lock = threading.Lock()
    
    def add_point(self, value: float, timestamp: Optional[float] = None) -> None:
        """
        添加数据点
        
        Args:
            value: 值
            timestamp: 时间戳
        """
        with self._lock:
            self._values.append(value)
            self._timestamps.append(timestamp or time.time())
            
            if len(self._values) > self._window_size:
                self._values = self._values[-self._window_size:]
                self._timestamps = self._timestamps[-self._window_size:]
    
    def predict(self, steps: int = 10) -> PredictionResult:
        """
        预测未来值
        
        Args:
            steps: 预测步数
            
        Returns:
            预测结果
        """
        with self._lock:
            values = self._values.copy()
            timestamps = self._timestamps.copy()
        
        if len(values) < 5:
            return PredictionResult(
                confidence=0.0
            )
        
        # Simple linear regression
        n = len(values)
        x_mean = sum(range(n)) / n
        y_mean = sum(values) / n
        
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        
        intercept = y_mean - slope * x_mean
        
        # Predict future values
        predicted_values = []
        predicted_timestamps = []
        confidence_intervals = []
        
        last_timestamp = timestamps[-1] if timestamps else time.time()
        interval = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1) if len(timestamps) > 1 else 60
        
        for i in range(1, steps + 1):
            x = n - 1 + i
            predicted = slope * x + intercept
            predicted_values.append(predicted)
            predicted_timestamps.append(last_timestamp + i * interval)
            
            # Simple confidence interval
            std = statistics.stdev(values) if len(values) > 1 else 0
            margin = 1.96 * std  # 95% confidence
            confidence_intervals.append((predicted - margin, predicted + margin))
        
        return PredictionResult(
            values=predicted_values,
            timestamps=predicted_timestamps,
            confidence_intervals=confidence_intervals,
            confidence=0.8 if len(values) >= 20 else 0.5
        )
