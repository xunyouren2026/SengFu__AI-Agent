"""
郁值检测器模块 - 胜复学架构的瓶颈监控系统

郁值检测器(HaltDetector)负责监控系统的瓶颈程度，通过分析损失平坦度、
梯度异常和注意力熵等指标来计算综合郁值(Halt Value)。

郁值计算公式:
    yu = 0.4 * 损失平坦度 + 0.3 * 梯度异常 + 0.3 * 注意力熵

其中:
- 损失平坦度(Loss Flatness): 损失函数在训练过程中的变化缓慢程度
- 梯度异常(Gradient Anomaly): 梯度消失或爆炸的程度
- 注意力熵(Attention Entropy): 注意力分布的混乱程度

阈值规则:
- 郁值 > 0.7: 触发HALT_WARNING预警事件
- 郁值 > 0.9: 触发HALT_ERUPT爆发事件

示例:
    >>> from agi_unified_framework.core.halt_detector import HaltDetector
    >>> detector = HaltDetector(window_size=100)
    >>> 
    >>> # 模拟训练过程中的更新
    >>> for step in range(1000):
    ...     loss = compute_loss()
    ...     gradients = compute_gradients()
    ...     attention_weights = compute_attention()
    ...     
    ...     detector.update(loss, gradients, attention_weights)
    ...     
    ...     halt_value = detector.get_halt_value()
    ...     event = detector.check_threshold()
    ...     
    ...     if event == "HALT_ERUPT":
    ...         print("郁值爆发！需要系统调节")
    ...     elif event == "HALT_WARNING":
    ...         print("郁值预警，注意监控")
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class GradientStats:
    """
    梯度统计信息
    
    存储梯度分析的相关统计数据。
    
    Attributes:
        mean_magnitude: 梯度平均幅度
        max_magnitude: 梯度最大幅度
        min_magnitude: 梯度最小幅度
        variance: 梯度方差
        vanishing_ratio: 梯度消失比例
        exploding_ratio: 梯度爆炸比例
        timestamp: 时间戳
    """
    mean_magnitude: float = 0.0
    max_magnitude: float = 0.0
    min_magnitude: float = 0.0
    variance: float = 0.0
    vanishing_ratio: float = 0.0
    exploding_ratio: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class LossStats:
    """
    损失统计信息
    
    存储损失函数分析的相关统计数据。
    
    Attributes:
        current_loss: 当前损失值
        mean_loss: 平均损失
        std_loss: 损失标准差
        change_rate: 损失变化率
        plateau_score: 平坦度分数
        trend: 趋势("increasing", "decreasing", "flat")
        timestamp: 时间戳
    """
    current_loss: float = 0.0
    mean_loss: float = 0.0
    std_loss: float = 0.0
    change_rate: float = 0.0
    plateau_score: float = 0.0
    trend: str = "flat"
    timestamp: float = field(default_factory=time.time)


@dataclass
class AttentionStats:
    """
    注意力统计信息
    
    存储注意力熵分析的相关统计数据。
    
    Attributes:
        entropy: 注意力熵
        max_attention: 最大注意力权重
        min_attention: 最小注意力权重
        sparsity: 注意力稀疏度
        concentration: 注意力集中度
        timestamp: 时间戳
    """
    entropy: float = 0.0
    max_attention: float = 0.0
    min_attention: float = 0.0
    sparsity: float = 0.0
    concentration: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class HaltMetrics:
    """
    郁值指标集合
    
    存储郁值检测器的完整指标数据。
    
    Attributes:
        halt_value: 综合郁值
        flatness: 损失平坦度
        gradient_anomaly: 梯度异常
        attention_entropy: 注意力熵
        timestamp: 时间戳
    """
    halt_value: float = 0.0
    flatness: float = 0.0
    gradient_anomaly: float = 0.0
    attention_entropy: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典格式"""
        return {
            "halt_value": self.halt_value,
            "flatness": self.flatness,
            "gradient_anomaly": self.gradient_anomaly,
            "attention_entropy": self.attention_entropy,
            "timestamp": self.timestamp
        }


# ============================================================================
# 郁值检测器核心类
# ============================================================================

class HaltDetector:
    """
    郁值检测器 - 监控系统瓶颈
    
    通过分析损失平坦度、梯度异常和注意力熵来计算综合郁值，
    用于检测系统训练和推理过程中的瓶颈和阻滞。
    
    Attributes:
        window_size: 滑动窗口大小
        loss_window: 损失值滑动窗口
        gradient_window: 梯度统计滑动窗口
        attention_window: 注意力权重滑动窗口
        weights: 郁值计算权重
        warning_threshold: 预警阈值
        erupt_threshold: 爆发阈值
    
    示例:
        >>> detector = HaltDetector(
        ...     window_size=100,
        ...     weights={"flatness": 0.4, "gradient": 0.3, "attention": 0.3}
        ... )
        >>> 
        >>> # 训练循环中更新
        >>> detector.update(loss=2.5, gradients=grads, attention_weights=attn)
        >>> 
        >>> # 获取郁值
        >>> halt = detector.get_halt_value()
        >>> print(f"当前郁值: {halt:.3f}")
        >>> 
        >>> # 检查阈值
        >>> event = detector.check_threshold()
        >>> if event:
        ...     print(f"触发事件: {event}")
    """
    
    def __init__(
        self,
        window_size: int = 100,
        weights: Optional[Dict[str, float]] = None,
        warning_threshold: float = 0.7,
        erupt_threshold: float = 0.9,
        vanishing_threshold: float = 1e-7,
        exploding_threshold: float = 1e3
    ):
        """
        初始化郁值检测器
        
        Args:
            window_size: 滑动窗口大小(默认100步)
            weights: 郁值计算权重字典
                - flatness: 损失平坦度权重(默认0.4)
                - gradient: 梯度异常权重(默认0.3)
                - attention: 注意力熵权重(默认0.3)
            warning_threshold: 预警阈值(默认0.7)
            erupt_threshold: 爆发阈值(默认0.9)
            vanishing_threshold: 梯度消失阈值(默认1e-7)
            exploding_threshold: 梯度爆炸阈值(默认1e3)
        """
        self.window_size = window_size
        self.warning_threshold = warning_threshold
        self.erupt_threshold = erupt_threshold
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold
        
        # 设置权重
        self.weights = weights or {
            "flatness": 0.4,
            "gradient": 0.3,
            "attention": 0.3
        }
        # 归一化权重
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        # 滑动窗口
        self._loss_window: deque = deque(maxlen=window_size)
        self._gradient_window: deque = deque(maxlen=window_size)
        self._attention_window: deque = deque(maxlen=window_size)
        
        # 当前统计
        self._current_metrics = HaltMetrics()
        self._loss_stats = LossStats()
        self._gradient_stats = GradientStats()
        self._attention_stats = AttentionStats()
        
        # 历史记录
        self._halt_history: deque = deque(maxlen=window_size)
        self._event_history: deque = deque(maxlen=window_size)
        
        # 状态标志
        self._last_event: Optional[str] = None
        self._consecutive_warnings: int = 0
        self._step_count: int = 0
    
    def update(
        self,
        loss: float,
        gradients: Optional[Union[np.ndarray, List[float]]] = None,
        attention_weights: Optional[Union[np.ndarray, List[List[float]]]] = None,
        step: Optional[int] = None
    ) -> HaltMetrics:
        """
        更新郁值检测器
        
        接收当前训练/推理步骤的数据，更新郁值计算。
        
        Args:
            loss: 当前损失值
            gradients: 梯度值(可选)
            attention_weights: 注意力权重(可选)
            step: 当前步数(可选，自动递增)
        
        Returns:
            当前郁值指标
        
        示例:
            >>> metrics = detector.update(
            ...     loss=2.5,
            ...     gradients=model.gradients(),
            ...     attention_weights=model.attention_weights()
            ... )
            >>> print(f"郁值: {metrics.halt_value:.3f}")
        """
        self._step_count = step if step is not None else self._step_count + 1
        
        # 更新损失窗口
        self._loss_window.append(loss)
        
        # 更新梯度窗口
        if gradients is not None:
            grad_stats = self._compute_gradient_stats(gradients)
            self._gradient_window.append(grad_stats)
            self._gradient_stats = grad_stats
        
        # 更新注意力窗口
        if attention_weights is not None:
            attn_stats = self._compute_attention_stats(attention_weights)
            self._attention_window.append(attn_stats)
            self._attention_stats = attn_stats
        
        # 计算损失统计
        self._loss_stats = self._compute_loss_stats()
        
        # 计算郁值分量
        flatness = self.compute_flatness()
        gradient_anomaly = self.compute_gradient_anomaly()
        attention_entropy = self.compute_attention_entropy()
        
        # 计算综合郁值
        halt_value = (
            self.weights["flatness"] * flatness +
            self.weights["gradient"] * gradient_anomaly +
            self.weights["attention"] * attention_entropy
        )
        
        # 更新当前指标
        self._current_metrics = HaltMetrics(
            halt_value=halt_value,
            flatness=flatness,
            gradient_anomaly=gradient_anomaly,
            attention_entropy=attention_entropy
        )
        
        # 记录历史
        self._halt_history.append(halt_value)
        
        return self._current_metrics
    
    def compute_flatness(self) -> float:
        """
        计算损失平坦度
        
        损失平坦度反映损失函数在训练过程中的变化缓慢程度。
        高平坦度表示损失变化很小，可能陷入局部最优或平台期。
        
        计算方法:
        1. 计算最近窗口内损失的标准差
        2. 计算损失变化率的绝对值平均值
        3. 综合评估平坦度
        
        Returns:
            平坦度分数 (0.0-1.0)，越高表示越平坦
        
        示例:
            >>> flatness = detector.compute_flatness()
            >>> if flatness > 0.8:
            ...     print("损失函数进入平台期")
        """
        if len(self._loss_window) < 10:
            return 0.0
        
        losses = np.array(list(self._loss_window))
        
        # 计算标准差(标准化)
        std_loss = np.std(losses)
        mean_loss = np.mean(losses)
        
        if mean_loss == 0:
            normalized_std = 0.0
        else:
            normalized_std = std_loss / (mean_loss + 1e-8)
        
        # 计算变化率
        if len(losses) > 1:
            changes = np.abs(np.diff(losses))
            mean_change = np.mean(changes)
            max_change = np.max(changes) + 1e-8
            change_ratio = mean_change / max_change
        else:
            change_ratio = 1.0
        
        # 平坦度 = 低变化率 + 相对标准差小
        # 使用sigmoid函数映射到0-1
        flatness_score = 1.0 - change_ratio
        
        # 考虑标准差因素
        std_factor = math.exp(-normalized_std * 10)  # 标准差越小，因子越接近1
        
        final_flatness = flatness_score * 0.7 + std_factor * 0.3
        
        return max(0.0, min(1.0, final_flatness))
    
    def compute_gradient_anomaly(self) -> float:
        """
        计算梯度异常
        
        检测梯度消失(vanishing)或梯度爆炸(exploding)的程度。
        高异常值表示存在严重的梯度问题。
        
        计算方法:
        1. 统计梯度幅度分布
        2. 计算梯度消失比例(低于阈值)
        3. 计算梯度爆炸比例(高于阈值)
        4. 综合评估异常程度
        
        Returns:
            梯度异常分数 (0.0-1.0)，越高表示异常越严重
        
        示例:
            >>> anomaly = detector.compute_gradient_anomaly()
            >>> if anomaly > 0.8:
            ...     print("检测到严重梯度问题")
        """
        if not self._gradient_window:
            return 0.0
        
        # 获取最近的梯度统计
        recent_stats = list(self._gradient_window)[-10:]
        
        if not recent_stats:
            return 0.0
        
        # 计算平均消失和爆炸比例
        avg_vanishing = np.mean([s.vanishing_ratio for s in recent_stats])
        avg_exploding = np.mean([s.exploding_ratio for s in recent_stats])
        
        # 计算梯度方差异常
        variances = [s.variance for s in recent_stats]
        if variances:
            variance_cv = np.std(variances) / (np.mean(variances) + 1e-8)
            variance_anomaly = min(1.0, variance_cv / 5.0)  # 归一化
        else:
            variance_anomaly = 0.0
        
        # 综合异常分数
        anomaly_score = (
            avg_vanishing * 0.4 +
            avg_exploding * 0.4 +
            variance_anomaly * 0.2
        )
        
        return max(0.0, min(1.0, anomaly_score))
    
    def compute_attention_entropy(self) -> float:
        """
        计算注意力熵
        
        注意力熵反映注意力分布的混乱程度。
        高熵表示注意力分散，低熵表示注意力集中。
        适中的熵值是理想的，过高或过低都可能表示问题。
        
        计算方法:
        1. 计算注意力分布的香农熵
        2. 归一化到0-1范围
        3. 评估极端情况(过高或过低的熵)
        
        Returns:
            注意力熵异常分数 (0.0-1.0)，越高表示越异常
        
        示例:
            >>> entropy = detector.compute_attention_entropy()
            >>> if entropy > 0.8:
            ...     print("注意力分布异常")
        """
        if not self._attention_window:
            return 0.0
        
        # 获取最近的注意力统计
        recent_stats = list(self._attention_window)[-10:]
        
        if not recent_stats:
            return 0.0
        
        # 计算平均熵
        avg_entropy = np.mean([s.entropy for s in recent_stats])
        
        # 熵的理想范围是[0.3, 0.7]
        # 低于0.3表示过度集中，高于0.7表示过度分散
        if avg_entropy < 0.3:
            # 过度集中
            entropy_anomaly = (0.3 - avg_entropy) / 0.3
        elif avg_entropy > 0.7:
            # 过度分散
            entropy_anomaly = (avg_entropy - 0.7) / 0.3
        else:
            # 正常范围
            entropy_anomaly = 0.0
        
        # 考虑稀疏度
        avg_sparsity = np.mean([s.sparsity for s in recent_stats])
        sparsity_anomaly = abs(avg_sparsity - 0.5) * 2  # 偏离0.5越多越异常
        
        # 综合异常分数
        final_entropy = entropy_anomaly * 0.7 + sparsity_anomaly * 0.3
        
        return max(0.0, min(1.0, final_entropy))
    
    def get_halt_value(self) -> float:
        """
        获取综合郁值
        
        返回当前计算的综合郁值。
        
        Returns:
            郁值 (0.0-1.0)
        """
        return self._current_metrics.halt_value
    
    def check_threshold(self) -> Optional[str]:
        """
        检查是否触发事件
        
        根据郁值阈值检查是否触发预警或爆发事件。
        
        Returns:
            事件类型字符串或None:
            - "HALT_ERUPT": 郁值爆发(>0.9)
            - "HALT_WARNING": 郁值预警(>0.7)
            - None: 未触发事件
        
        示例:
            >>> event = detector.check_threshold()
            >>> if event == "HALT_ERUPT":
            ...     # 执行紧急调节
            ...     pass
            >>> elif event == "HALT_WARNING":
            ...     # 增加监控频率
            ...     pass
        """
        halt_value = self._current_metrics.halt_value
        
        event = None
        
        if halt_value >= self.erupt_threshold:
            event = "HALT_ERUPT"
            self._consecutive_warnings += 1
        elif halt_value >= self.warning_threshold:
            event = "HALT_WARNING"
            self._consecutive_warnings += 1
        else:
            self._consecutive_warnings = 0
        
        if event and event != self._last_event:
            self._event_history.append({
                "event": event,
                "halt_value": halt_value,
                "timestamp": time.time(),
                "step": self._step_count
            })
            self._last_event = event
        
        return event
    
    def get_metrics(self) -> HaltMetrics:
        """
        获取完整郁值指标
        
        Returns:
            郁值指标对象
        """
        return self._current_metrics
    
    def get_loss_stats(self) -> LossStats:
        """
        获取损失统计
        
        Returns:
            损失统计对象
        """
        return self._loss_stats
    
    def get_gradient_stats(self) -> GradientStats:
        """
        获取梯度统计
        
        Returns:
            梯度统计对象
        """
        return self._gradient_stats
    
    def get_attention_stats(self) -> AttentionStats:
        """
        获取注意力统计
        
        Returns:
            注意力统计对象
        """
        return self._attention_stats
    
    def get_history(self, window: Optional[int] = None) -> List[float]:
        """
        获取郁值历史
        
        Args:
            window: 窗口大小(可选)
        
        Returns:
            郁值历史列表
        """
        history = list(self._halt_history)
        if window is not None:
            history = history[-window:]
        return history
    
    def get_event_history(self) -> List[Dict[str, Any]]:
        """
        获取事件历史
        
        Returns:
            事件历史列表
        """
        return list(self._event_history)
    
    def is_plateau(self, threshold: float = 0.8) -> bool:
        """
        检查是否处于损失平台期
        
        Args:
            threshold: 平坦度阈值
        
        Returns:
            是否处于平台期
        """
        return self._loss_stats.plateau_score >= threshold
    
    def has_gradient_problem(self, threshold: float = 0.8) -> bool:
        """
        检查是否存在梯度问题
        
        Args:
            threshold: 异常阈值
        
        Returns:
            是否存在梯度问题
        """
        return self._current_metrics.gradient_anomaly >= threshold
    
    def get_recommendations(self) -> List[str]:
        """
        获取调节建议
        
        根据当前郁值状态返回调节建议。
        
        Returns:
            建议列表
        """
        recommendations = []
        
        metrics = self._current_metrics
        
        if metrics.flatness > 0.8:
            recommendations.append(
                "损失平坦度过高: 建议降低学习率或增加学习率调度"
            )
        
        if metrics.gradient_anomaly > 0.7:
            grad_stats = self._gradient_stats
            if grad_stats.vanishing_ratio > 0.5:
                recommendations.append(
                    "梯度消失严重: 建议使用残差连接或梯度裁剪"
                )
            if grad_stats.exploding_ratio > 0.5:
                recommendations.append(
                    "梯度爆炸严重: 建议实施梯度裁剪或降低学习率"
                )
        
        if metrics.attention_entropy > 0.7:
            attn_stats = self._attention_stats
            if attn_stats.entropy < 0.2:
                recommendations.append(
                    "注意力过度集中: 建议增加dropout或温度调节"
                )
            else:
                recommendations.append(
                    "注意力过度分散: 建议减少注意力头数或增加正则化"
                )
        
        if metrics.halt_value > 0.9:
            recommendations.append(
                "郁值爆发: 建议立即暂停训练并检查系统状态"
            )
        elif metrics.halt_value > 0.7:
            recommendations.append(
                "郁值预警: 建议增加监控频率并准备调节措施"
            )
        
        return recommendations
    
    def reset(self) -> None:
        """
        重置检测器
        
        清空所有窗口和历史记录。
        """
        self._loss_window.clear()
        self._gradient_window.clear()
        self._attention_window.clear()
        self._halt_history.clear()
        self._event_history.clear()
        self._current_metrics = HaltMetrics()
        self._loss_stats = LossStats()
        self._gradient_stats = GradientStats()
        self._attention_stats = AttentionStats()
        self._last_event = None
        self._consecutive_warnings = 0
        self._step_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取完整统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "current_metrics": self._current_metrics.to_dict(),
            "loss_stats": {
                "current": self._loss_stats.current_loss,
                "mean": self._loss_stats.mean_loss,
                "std": self._loss_stats.std_loss,
                "trend": self._loss_stats.trend
            },
            "gradient_stats": {
                "mean_magnitude": self._gradient_stats.mean_magnitude,
                "vanishing_ratio": self._gradient_stats.vanishing_ratio,
                "exploding_ratio": self._gradient_stats.exploding_ratio
            },
            "attention_stats": {
                "entropy": self._attention_stats.entropy,
                "sparsity": self._attention_stats.sparsity,
                "concentration": self._attention_stats.concentration
            },
            "window_sizes": {
                "loss": len(self._loss_window),
                "gradient": len(self._gradient_window),
                "attention": len(self._attention_window)
            },
            "event_count": len(self._event_history),
            "consecutive_warnings": self._consecutive_warnings,
            "step_count": self._step_count
        }
    
    # ========================================================================
    # 内部辅助方法
    # ========================================================================
    
    def _compute_loss_stats(self) -> LossStats:
        """计算损失统计"""
        if len(self._loss_window) < 2:
            return LossStats(
                current_loss=self._loss_window[-1] if self._loss_window else 0.0
            )
        
        losses = np.array(list(self._loss_window))
        current = losses[-1]
        mean = np.mean(losses)
        std = np.std(losses)
        
        # 计算变化率
        if len(losses) >= 10:
            recent_mean = np.mean(losses[-10:])
            previous_mean = np.mean(losses[-20:-10]) if len(losses) >= 20 else np.mean(losses[:10])
            change_rate = (recent_mean - previous_mean) / (abs(previous_mean) + 1e-8)
        else:
            change_rate = 0.0
        
        # 判断趋势
        if change_rate < -0.01:
            trend = "decreasing"
        elif change_rate > 0.01:
            trend = "increasing"
        else:
            trend = "flat"
        
        # 计算平台期分数
        if len(losses) >= 10:
            recent_std = np.std(losses[-10:])
            plateau_score = math.exp(-recent_std * 10)  # 标准差越小，分数越高
        else:
            plateau_score = 0.0
        
        return LossStats(
            current_loss=current,
            mean_loss=mean,
            std_loss=std,
            change_rate=change_rate,
            plateau_score=plateau_score,
            trend=trend
        )
    
    def _compute_gradient_stats(
        self,
        gradients: Union[np.ndarray, List[float]]
    ) -> GradientStats:
        """计算梯度统计"""
        if isinstance(gradients, list):
            gradients = np.array(gradients)
        
        # 展平梯度
        flat_grads = gradients.flatten()
        
        # 计算幅度
        magnitudes = np.abs(flat_grads)
        
        mean_mag = np.mean(magnitudes)
        max_mag = np.max(magnitudes)
        min_mag = np.min(magnitudes)
        variance = np.var(flat_grads)
        
        # 计算消失和爆炸比例
        vanishing_count = np.sum(magnitudes < self.vanishing_threshold)
        exploding_count = np.sum(magnitudes > self.exploding_threshold)
        total_count = len(flat_grads)
        
        vanishing_ratio = vanishing_count / total_count
        exploding_ratio = exploding_count / total_count
        
        return GradientStats(
            mean_magnitude=mean_mag,
            max_magnitude=max_mag,
            min_magnitude=min_mag,
            variance=variance,
            vanishing_ratio=vanishing_ratio,
            exploding_ratio=exploding_ratio
        )
    
    def _compute_attention_stats(
        self,
        attention_weights: Union[np.ndarray, List[List[float]]]
    ) -> AttentionStats:
        """计算注意力统计"""
        if isinstance(attention_weights, list):
            attention_weights = np.array(attention_weights)
        
        # 确保是2D数组 [batch, seq_len] 或 [seq_len, seq_len]
        if attention_weights.ndim > 2:
            # 取第一个batch和第一个头
            attention_weights = attention_weights[0, 0] if attention_weights.ndim >= 3 else attention_weights[0]
        
        # 归一化
        attention_weights = np.clip(attention_weights, 1e-10, 1.0)
        
        # 计算熵
        # H = -sum(p * log(p))
        entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-10))
        
        # 归一化熵到[0, 1]
        seq_len = attention_weights.shape[-1]
        max_entropy = np.log(seq_len + 1e-10)
        normalized_entropy = entropy / (max_entropy + 1e-10)
        
        # 计算稀疏度(非零元素比例)
        sparsity = np.mean(attention_weights > 0.01)
        
        # 计算集中度(max attention)
        max_attention = np.max(attention_weights)
        min_attention = np.min(attention_weights)
        concentration = max_attention
        
        return AttentionStats(
            entropy=normalized_entropy,
            max_attention=max_attention,
            min_attention=min_attention,
            sparsity=sparsity,
            concentration=concentration
        )


# ============================================================================
# 便捷函数
# ============================================================================

def create_halt_detector(
    window_size: int = 100,
    **kwargs
) -> HaltDetector:
    """
    创建郁值检测器的便捷函数
    
    Args:
        window_size: 滑动窗口大小
        **kwargs: 其他配置参数
    
    Returns:
        郁值检测器实例
    """
    return HaltDetector(window_size=window_size, **kwargs)


def detect_halt_from_metrics(
    loss_history: List[float],
    gradient_history: Optional[List[np.ndarray]] = None,
    attention_history: Optional[List[np.ndarray]] = None,
    window_size: int = 50
) -> HaltMetrics:
    """
    从历史指标检测郁值
    
    一次性分析历史数据并返回郁值指标。
    
    Args:
        loss_history: 损失历史
        gradient_history: 梯度历史(可选)
        attention_history: 注意力历史(可选)
        window_size: 窗口大小
    
    Returns:
        郁值指标
    
    示例:
        >>> metrics = detect_halt_from_metrics(
        ...     loss_history=losses,
        ...     gradient_history=gradients,
        ...     window_size=100
        ... )
        >>> print(f"郁值: {metrics.halt_value:.3f}")
    """
    detector = HaltDetector(window_size=window_size)
    
    # 批量更新
    min_len = len(loss_history)
    if gradient_history:
        min_len = min(min_len, len(gradient_history))
    if attention_history:
        min_len = min(min_len, len(attention_history))
    
    for i in range(min_len):
        loss = loss_history[i]
        grads = gradient_history[i] if gradient_history else None
        attn = attention_history[i] if attention_history else None
        
        detector.update(loss, grads, attn, step=i)
    
    return detector.get_metrics()
