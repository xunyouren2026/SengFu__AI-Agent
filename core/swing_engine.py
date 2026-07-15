"""
执行层引擎模块 - 胜复学架构的策略决策与置信度输出

执行层引擎(SwingEngine)是胜复学架构中执行层的核心组件，负责策略决策、
置信度计算和动作输出。它集成MoE网络，通过MC Dropout计算置信度，
并实现防过亢机制。

核心功能:
- MoE网络集成: 复用unified_moe模块的混合专家系统
- 置信度计算: 基于MC Dropout方差计算置信度
- 动作选择: 基于置信度进行动作决策
- 胜过亢检测: 检测置信度持续过高的情况
- 防过亢机制: 动态增加探索避免过亢

状态系数关联:
- 胜值(Swing): 由本引擎输出的置信度
- 郁值(Halt): 接收自郁值检测器，影响探索策略
- 复值(Balance): 接收自复层，调节决策平衡
- 发值(Release): 输出动作强度
- 道值(Dao): 安全约束

示例:
    >>> from agi_unified_framework.core.swing_engine import SwingEngine
    >>> from agi_unified_framework.core.unified_algorithms.unified_moe import MixtureOfExperts
    >>> 
    >>> # 初始化引擎
    >>> moe = MixtureOfExperts(num_experts=8, top_k=2)
    >>> config = {
    ...     "confidence_threshold": 0.9,
    ...     "excess_steps": 10,
    ...     "mc_dropout_iterations": 10
    ... }
    >>> engine = SwingEngine(moe_network=moe, config=config)
    >>> 
    >>> # 执行决策
    >>> observation = get_observation()
    >>> action, confidence = engine.act(observation, training=True)
    >>> 
    >>> # 检查过亢
    >>> if engine.check_swing_excess():
    ...     print("检测到胜过亢，增加探索")
    >>> 
    >>> # 获取专家使用统计
    >>> usage = engine.get_expert_usage()
    >>> print(f"专家使用分布: {usage}")
"""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class ActionResult:
    """
    动作结果数据类
    
    存储执行层引擎输出的动作和相关信息。
    
    Attributes:
        action: 动作值
        confidence: 置信度
        expert_id: 选中的专家ID
        expert_weights: 专家权重分布
        exploration_used: 是否使用了探索
        timestamp: 时间戳
    """
    action: Any
    confidence: float
    expert_id: int
    expert_weights: Dict[int, float] = field(default_factory=dict)
    exploration_used: bool = False
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "action": self.action,
            "confidence": self.confidence,
            "expert_id": self.expert_id,
            "expert_weights": self.expert_weights,
            "exploration_used": self.exploration_used,
            "timestamp": self.timestamp
        }


@dataclass
class ConfidenceStats:
    """
    置信度统计数据类
    
    存储置信度计算的统计信息。
    
    Attributes:
        mean: 平均置信度
        variance: 方差
        std: 标准差
        min_conf: 最小置信度
        max_conf: 最大置信度
        entropy: 置信度分布熵
    """
    mean: float = 0.0
    variance: float = 0.0
    std: float = 0.0
    min_conf: float = 0.0
    max_conf: float = 0.0
    entropy: float = 0.0


@dataclass
class SwingConfig:
    """
    执行层引擎配置
    
    Attributes:
        confidence_threshold: 胜值过高阈值(默认0.9)
        excess_steps: 连续高胜值步数阈值(默认10)
        mc_dropout_iterations: MC Dropout迭代次数(默认10)
        exploration_rate: 探索率(默认0.1)
        exploration_boost: 过亢时探索增强倍数(默认2.0)
        temperature: Softmax温度(默认1.0)
        min_confidence: 最小置信度(默认0.1)
        max_confidence: 最大置信度(默认1.0)
    """
    confidence_threshold: float = 0.9
    excess_steps: int = 10
    mc_dropout_iterations: int = 10
    exploration_rate: float = 0.1
    exploration_boost: float = 2.0
    temperature: float = 1.0
    min_confidence: float = 0.1
    max_confidence: float = 1.0
    
    @classmethod
    def default_config(cls) -> "SwingConfig":
        """返回默认配置"""
        return cls()


# ============================================================================
# 执行层引擎核心类
# ============================================================================

class SwingEngine:
    """
    执行层引擎 - 策略决策与置信度输出
    
    执行层引擎是胜复学架构中执行层的核心组件，集成MoE网络进行策略决策，
    通过MC Dropout计算置信度(胜值)，并实现防过亢机制。
    
    Attributes:
        moe_network: MoE网络实例
        config: 引擎配置
        confidence_history: 置信度历史
        action_history: 动作历史
        expert_usage_count: 专家使用计数
        excess_counter: 过亢计数器
    
    示例:
        >>> from agi_unified_framework.core.unified_algorithms.unified_moe import MixtureOfExperts
        >>> 
        >>> # 创建MoE网络和引擎
        >>> moe = MixtureOfExperts(num_experts=8, top_k=2)
        >>> engine = SwingEngine(moe_network=moe)
        >>> 
        >>> # 执行决策循环
        >>> for step in range(100):
        ...     obs = get_observation()
        ...     action, confidence = engine.act(obs, training=True)
        ...     
        ...     # 检查过亢
        ...     if engine.check_swing_excess():
        ...         print(f"Step {step}: 检测到胜过亢")
        ...     
        ...     # 执行动作
        ...     execute_action(action)
        >>> 
        >>> # 获取统计
        >>> usage = engine.get_expert_usage()
        >>> print(f"专家使用: {usage}")
    """
    
    def __init__(
        self,
        moe_network: Any,
        config: Optional[Union[SwingConfig, Dict[str, Any]]] = None
    ):
        """
        初始化执行层引擎
        
        Args:
            moe_network: MoE网络实例(来自unified_moe)
            config: 配置对象或字典
        
        示例:
            >>> from agi_unified_framework.core.unified_algorithms.unified_moe import MixtureOfExperts
            >>> moe = MixtureOfExperts(num_experts=8)
            >>> engine = SwingEngine(moe)
        """
        self.moe_network = moe_network
        
        # 解析配置
        if config is None:
            self.config = SwingConfig.default_config()
        elif isinstance(config, dict):
            self.config = SwingConfig(**config)
        else:
            self.config = config
        
        # 历史记录
        self._confidence_history: deque = deque(maxlen=1000)
        self._action_history: deque = deque(maxlen=1000)
        self._expert_usage_count: Dict[int, int] = {}
        
        # 过亢检测
        self._excess_counter: int = 0
        self._high_confidence_streak: int = 0
        self._in_excess_state: bool = False
        
        # 当前状态
        self._current_confidence: float = 0.5
        self._current_action: Optional[Any] = None
        self._current_expert_weights: Dict[int, float] = {}
        
        # 统计
        self._total_steps: int = 0
        self._exploration_steps: int = 0
        self._excess_events: int = 0
    
    def act(
        self,
        observation: Any,
        training: bool = True,
        force_exploration: bool = False
    ) -> Tuple[Any, float]:
        """
        执行动作决策
        
        基于观察值进行动作决策，返回动作和置信度。
        
        Args:
            observation: 观察值/输入
            training: 是否处于训练模式
            force_exploration: 强制探索
        
        Returns:
            (动作, 置信度)元组
        
        示例:
            >>> action, confidence = engine.act(observation, training=True)
            >>> print(f"动作: {action}, 置信度: {confidence:.3f}")
        """
        self._total_steps += 1
        
        # 计算置信度
        confidence = self.compute_confidence(observation)
        self._current_confidence = confidence
        
        # 检查是否使用探索
        use_exploration = force_exploration or self._should_explore(confidence)
        
        if use_exploration and training:
            # 探索模式
            action = self._explore_action(observation)
            self._exploration_steps += 1
            exploration_used = True
        else:
            # 利用模式 - 使用MoE网络
            action_result = self._moe_forward(observation)
            action = action_result.action
            exploration_used = False
            
            # 更新专家使用统计
            self._update_expert_usage(action_result.expert_weights)
        
        # 记录历史
        self._confidence_history.append(confidence)
        self._action_history.append({
            "action": action,
            "confidence": confidence,
            "exploration": exploration_used,
            "timestamp": time.time()
        })
        
        self._current_action = action
        
        # 更新过亢检测
        self._update_excess_detection(confidence)
        
        return action, confidence
    
    def compute_confidence(self, observation: Any) -> float:
        """
        计算置信度 (MC Dropout)
        
        使用MC Dropout方法计算置信度。通过多次前向传播计算输出的方差，
        方差越小表示置信度越高。
        
        计算方法:
        1. 启用dropout进行多次前向传播
        2. 计算输出分布的均值和方差
        3. 将方差转换为置信度分数
        
        Args:
            observation: 观察值
        
        Returns:
            置信度 (0.0-1.0)
        
        示例:
            >>> confidence = engine.compute_confidence(observation)
            >>> print(f"置信度: {confidence:.3f}")
        """
        if self.moe_network is None:
            return 0.5
        
        # MC Dropout迭代
        outputs = []
        
        for _ in range(self.config.mc_dropout_iterations):
            # 使用MoE网络进行前向传播
            result = self.moe_network.process(observation)
            
            # 提取输出值
            if hasattr(result, 'data'):
                output = result.data
            elif hasattr(result, 'confidence'):
                output = result.confidence
            else:
                output = result
            
            outputs.append(output)
        
        # 计算置信度统计
        stats = self._compute_confidence_stats(outputs)
        
        # 将方差转换为置信度 (方差越小，置信度越高)
        # 使用指数衰减函数
        confidence = self.config.max_confidence * math.exp(-stats.variance * 5)
        confidence = max(
            self.config.min_confidence,
            min(self.config.max_confidence, confidence)
        )
        
        return confidence
    
    def check_swing_excess(self) -> bool:
        """
        检查胜过亢
        
        检测置信度是否持续过高(胜过亢)。当置信度超过阈值并持续一定步数时，
        认为系统进入过亢状态，需要增加探索。
        
        Returns:
            是否处于过亢状态
        
        示例:
            >>> if engine.check_swing_excess():
            ...     print("检测到胜过亢，增加探索")
            ...     engine.enable_exploration_boost()
        """
        return self._in_excess_state
    
    def get_expert_usage(self) -> Dict[str, float]:
        """
        获取专家使用统计
        
        返回各专家的使用频率分布。
        
        Returns:
            专家使用统计字典
        
        示例:
            >>> usage = engine.get_expert_usage()
            >>> for expert_id, freq in usage.items():
            ...     print(f"专家{expert_id}: {freq:.2%}")
        """
        total_usage = sum(self._expert_usage_count.values())
        
        if total_usage == 0:
            return {}
        
        usage_stats = {}
        for expert_id, count in self._expert_usage_count.items():
            usage_stats[f"expert_{expert_id}"] = count / total_usage
        
        return usage_stats
    
    def get_confidence_stats(self) -> ConfidenceStats:
        """
        获取置信度统计
        
        Returns:
            置信度统计对象
        """
        if not self._confidence_history:
            return ConfidenceStats()
        
        confidences = list(self._confidence_history)
        
        mean = np.mean(confidences)
        variance = np.var(confidences)
        std = np.std(confidences)
        min_conf = np.min(confidences)
        max_conf = np.max(confidences)
        
        # 计算熵
        hist, _ = np.histogram(confidences, bins=10, range=(0, 1))
        probs = hist / len(confidences)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        return ConfidenceStats(
            mean=mean,
            variance=variance,
            std=std,
            min_conf=min_conf,
            max_conf=max_conf,
            entropy=entropy
        )
    
    def get_recent_confidence(self, window: int = 10) -> List[float]:
        """
        获取最近置信度
        
        Args:
            window: 窗口大小
        
        Returns:
            最近置信度列表
        """
        history = list(self._confidence_history)
        return history[-window:] if len(history) >= window else history
    
    def get_average_confidence(self, window: Optional[int] = None) -> float:
        """
        获取平均置信度
        
        Args:
            window: 窗口大小(可选)
        
        Returns:
            平均置信度
        """
        if not self._confidence_history:
            return 0.5
        
        confidences = list(self._confidence_history)
        if window is not None:
            confidences = confidences[-window:]
        
        return float(np.mean(confidences))
    
    def enable_exploration_boost(self) -> None:
        """
        启用探索增强
        
        临时增加探索率以缓解过亢状态。
        """
        self._in_excess_state = True
        self._excess_events += 1
    
    def disable_exploration_boost(self) -> None:
        """
        禁用探索增强
        
        恢复正常探索率。
        """
        self._in_excess_state = False
        self._high_confidence_streak = 0
    
    def reset(self) -> None:
        """
        重置引擎
        
        清空所有历史记录和状态。
        """
        self._confidence_history.clear()
        self._action_history.clear()
        self._expert_usage_count.clear()
        self._excess_counter = 0
        self._high_confidence_streak = 0
        self._in_excess_state = False
        self._current_confidence = 0.5
        self._current_action = None
        self._current_expert_weights = {}
        self._total_steps = 0
        self._exploration_steps = 0
        self._excess_events = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取引擎统计信息
        
        Returns:
            统计信息字典
        """
        conf_stats = self.get_confidence_stats()
        
        return {
            "total_steps": self._total_steps,
            "exploration_steps": self._exploration_steps,
            "exploration_rate": (
                self._exploration_steps / self._total_steps
                if self._total_steps > 0 else 0.0
            ),
            "excess_events": self._excess_events,
            "in_excess_state": self._in_excess_state,
            "current_confidence": self._current_confidence,
            "confidence_stats": {
                "mean": conf_stats.mean,
                "variance": conf_stats.variance,
                "std": conf_stats.std,
                "min": conf_stats.min_conf,
                "max": conf_stats.max_conf
            },
            "expert_usage": self.get_expert_usage(),
            "high_confidence_streak": self._high_confidence_streak
        }
    
    # ========================================================================
    # 内部辅助方法
    # ========================================================================
    
    def _should_explore(self, confidence: float) -> bool:
        """
        判断是否使用探索
        
        Args:
            confidence: 当前置信度
        
        Returns:
            是否使用探索
        """
        # 基础探索率
        base_rate = self.config.exploration_rate
        
        # 过亢时增加探索
        if self._in_excess_state:
            base_rate *= self.config.exploration_boost
        
        # 高置信度时增加探索
        if confidence > self.config.confidence_threshold:
            base_rate *= 1.5
        
        return random.random() < base_rate
    
    def _explore_action(self, observation: Any) -> Any:
        """
        探索动作
        
        生成随机或扰动动作进行探索。
        
        Args:
            observation: 观察值
        
        Returns:
            探索动作
        """
        # 先获取MoE输出
        result = self.moe_network.process(observation)
        
        if hasattr(result, 'data'):
            base_action = result.data
        else:
            base_action = result
        
        # 添加随机扰动
        if isinstance(base_action, (int, float)):
            noise = random.gauss(0, 0.1)
            return base_action + noise
        elif isinstance(base_action, np.ndarray):
            noise = np.random.normal(0, 0.1, base_action.shape)
            return base_action + noise
        elif isinstance(base_action, (list, tuple)):
            noise = [random.gauss(0, 0.1) for _ in base_action]
            return [a + n for a, n in zip(base_action, noise)]
        else:
            return base_action
    
    def _moe_forward(self, observation: Any) -> ActionResult:
        """
        MoE网络前向传播
        
        Args:
            observation: 观察值
        
        Returns:
            动作结果
        """
        # 使用MoE网络处理
        result = self.moe_network.process(observation)
        
        # 解析结果
        if hasattr(result, 'data'):
            action = result.data
            expert_id = getattr(result, 'expert_id', -1)
            confidence = getattr(result, 'confidence', 0.5)
        else:
            action = result
            expert_id = -1
            confidence = 0.5
        
        # 获取专家权重
        expert_weights = self._get_expert_weights(observation)
        self._current_expert_weights = expert_weights
        
        return ActionResult(
            action=action,
            confidence=confidence,
            expert_id=expert_id,
            expert_weights=expert_weights
        )
    
    def _get_expert_weights(self, observation: Any) -> Dict[int, float]:
        """
        获取专家权重
        
        Args:
            observation: 观察值
        
        Returns:
            专家权重字典
        """
        # 通过路由获取专家权重
        if hasattr(self.moe_network, 'route'):
            routing_info = self.moe_network.route(observation)
            
            if hasattr(routing_info, 'expert_indices'):
                indices = routing_info.expert_indices
                weights = getattr(routing_info, 'gate_weights', [1.0] * len(indices))
                return dict(zip(indices, weights))
        
        return {}
    
    def _update_expert_usage(self, expert_weights: Dict[int, float]) -> None:
        """
        更新专家使用统计
        
        Args:
            expert_weights: 专家权重
        """
        for expert_id, weight in expert_weights.items():
            if expert_id not in self._expert_usage_count:
                self._expert_usage_count[expert_id] = 0
            self._expert_usage_count[expert_id] += 1
    
    def _update_excess_detection(self, confidence: float) -> None:
        """
        更新过亢检测
        
        Args:
            confidence: 当前置信度
        """
        if confidence >= self.config.confidence_threshold:
            self._high_confidence_streak += 1
        else:
            self._high_confidence_streak = 0
            self._in_excess_state = False
        
        # 检查是否触发过亢
        if self._high_confidence_streak >= self.config.excess_steps:
            if not self._in_excess_state:
                self.enable_exploration_boost()
    
    def _compute_confidence_stats(self, outputs: List[Any]) -> ConfidenceStats:
        """
        计算置信度统计
        
        Args:
            outputs: MC Dropout输出列表
        
        Returns:
            置信度统计
        """
        # 转换为数值数组
        numeric_outputs = []
        for output in outputs:
            if isinstance(output, (int, float)):
                numeric_outputs.append(float(output))
            elif isinstance(output, np.ndarray):
                numeric_outputs.extend(output.flatten().tolist())
            elif isinstance(output, (list, tuple)):
                numeric_outputs.extend([float(x) for x in output])
        
        if not numeric_outputs:
            return ConfidenceStats()
        
        outputs_array = np.array(numeric_outputs)
        
        mean = float(np.mean(outputs_array))
        variance = float(np.var(outputs_array))
        std = float(np.std(outputs_array))
        min_val = float(np.min(outputs_array))
        max_val = float(np.max(outputs_array))
        
        # 计算熵
        hist, _ = np.histogram(outputs_array, bins=10)
        probs = hist / len(outputs_array)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        return ConfidenceStats(
            mean=mean,
            variance=variance,
            std=std,
            min_conf=min_val,
            max_conf=max_val,
            entropy=entropy
        )


# ============================================================================
# 便捷函数
# ============================================================================

def create_swing_engine(
    moe_network: Any,
    **config_kwargs
) -> SwingEngine:
    """
    创建执行层引擎的便捷函数
    
    Args:
        moe_network: MoE网络实例
        **config_kwargs: 配置参数
    
    Returns:
        执行层引擎实例
    
    示例:
        >>> from agi_unified_framework.core.unified_algorithms.unified_moe import MixtureOfExperts
        >>> moe = MixtureOfExperts(num_experts=8)
        >>> engine = create_swing_engine(moe, confidence_threshold=0.85)
    """
    config = SwingConfig(**config_kwargs) if config_kwargs else None
    return SwingEngine(moe_network=moe_network, config=config)


def compute_mc_dropout_confidence(
    model: Any,
    observation: Any,
    iterations: int = 10
) -> float:
    """
    使用MC Dropout计算置信度的便捷函数
    
    Args:
        model: 模型实例
        observation: 观察值
        iterations: MC Dropout迭代次数
    
    Returns:
        置信度
    
    示例:
        >>> confidence = compute_mc_dropout_confidence(model, obs, iterations=20)
        >>> print(f"置信度: {confidence:.3f}")
    """
    outputs = []
    
    for _ in range(iterations):
        if hasattr(model, 'process'):
            result = model.process(observation)
        elif hasattr(model, 'forward'):
            result = model.forward(observation)
        else:
            result = model(observation)
        
        if hasattr(result, 'data'):
            output = result.data
        elif hasattr(result, 'confidence'):
            output = result.confidence
        else:
            output = result
        
        outputs.append(output)
    
    # 计算方差
    if isinstance(outputs[0], (int, float)):
        outputs_array = np.array([float(o) for o in outputs])
    elif isinstance(outputs[0], np.ndarray):
        outputs_array = np.array([o.flatten() for o in outputs])
    else:
        return 0.5
    
    variance = np.var(outputs_array)
    confidence = math.exp(-variance * 5)
    
    return max(0.0, min(1.0, confidence))


# Import math here for use in the module
import math
