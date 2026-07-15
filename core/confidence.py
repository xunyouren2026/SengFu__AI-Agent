"""
MC Dropout置信度估计模块 (MC Dropout Confidence Estimation Module)

该模块实现了基于蒙特卡洛Dropout的预测置信度估计，支持分类和回归任务。
通过多次前向传播获取预测分布，计算预测均值、方差、熵等作为置信度指标。

核心功能:
- MC Dropout多次前向传播
- 预测熵计算: H = -Σ p_i * log(p_i)
- 互信息估计 (BALD): I(y; θ) = H(y) - E[H(y|θ)]
- 分类和回归任务的置信度计算
- 与StateBus集成，支持胜复学架构

关键算法:
- MC Dropout: T次前向传播，dropout保持开启
- 预测熵: H = -Σ p_i * log(p_i)
- 互信息: I(y; θ) = H(y) - E[H(y|θ)]
- 方差估计: 分类为预测方差，回归为输出方差

作者: AGI Universal Framework Team
版本: 1.0.0
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Callable, Any, Literal
from dataclasses import dataclass, field
from enum import Enum, auto
import threading
import time
from collections import deque

# 尝试导入state_bus用于集成
try:
    from .state_bus import StateBus, EventType, get_global_state_bus, StateCoefficient
    _HAS_STATE_BUS = True
except ImportError:
    _HAS_STATE_BUS = False


# ============================================================================
# 类型定义
# ============================================================================

class TaskType(Enum):
    """任务类型枚举"""
    CLASSIFICATION = "classification"  # 分类任务
    REGRESSION = "regression"          # 回归任务
    MULTI_LABEL = "multi_label"        # 多标签分类


class ConfidenceMetric(Enum):
    """置信度指标类型枚举"""
    ENTROPY = "entropy"                # 预测熵
    MUTUAL_INFO = "mutual_info"        # 互信息 (BALD)
    VARIANCE = "variance"              # 预测方差
    MAX_PROB = "max_prob"              # 最大概率
    EXPECTED_ENTROPY = "expected_entropy"  # 期望熵


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class ConfidenceEstimate:
    """
    置信度估计结果数据类
    
    存储MC Dropout置信度估计的完整结果，包括各种不确定性指标。
    
    Attributes:
        mean: 预测均值 [batch_size, num_classes] 或 [batch_size, output_dim]
        variance: 预测方差
        entropy: 预测熵 H(y|x,D)
        expected_entropy: 期望熵 E[H(y|x,θ)]
        mutual_info: 互信息 I(y;θ|x,D) = H(y|x,D) - E[H(y|x,θ)]
        max_prob: 最大预测概率
        confidence_score: 综合置信度分数 (0-1)
        epistemic: 认知不确定性 (模型不确定性)
        aleatoric: 偶然不确定性 (数据不确定性)
        task_type: 任务类型
        samples: 原始MC样本 [n_samples, batch_size, ...]
        timestamp: 时间戳
    
    示例:
        >>> estimate = ConfidenceEstimate(
        ...     mean=np.array([[0.7, 0.2, 0.1]]),
        ...     variance=np.array([[0.01, 0.005, 0.003]]),
        ...     entropy=0.8,
        ...     mutual_info=0.3,
        ...     confidence_score=0.85
        ... )
    """
    mean: np.ndarray
    variance: np.ndarray
    entropy: float
    expected_entropy: float
    mutual_info: float
    max_prob: float
    confidence_score: float
    epistemic: Optional[np.ndarray] = None
    aleatoric: Optional[np.ndarray] = None
    task_type: TaskType = TaskType.CLASSIFICATION
    samples: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            包含所有置信度指标的字典
        """
        result = {
            'mean': self.mean.tolist() if isinstance(self.mean, np.ndarray) else self.mean,
            'variance': self.variance.tolist() if isinstance(self.variance, np.ndarray) else self.variance,
            'entropy': float(self.entropy),
            'expected_entropy': float(self.expected_entropy),
            'mutual_info': float(self.mutual_info),
            'max_prob': float(self.max_prob),
            'confidence_score': float(self.confidence_score),
            'task_type': self.task_type.value,
            'timestamp': self.timestamp
        }
        
        if self.epistemic is not None:
            result['epistemic'] = self.epistemic.tolist() if isinstance(self.epistemic, np.ndarray) else self.epistemic
        if self.aleatoric is not None:
            result['aleatoric'] = self.aleatoric.tolist() if isinstance(self.aleatoric, np.ndarray) else self.aleatoric
        
        return result
    
    def get_prediction(self) -> np.ndarray:
        """
        获取最终预测结果
        
        Returns:
            预测类别索引 (分类) 或预测值 (回归)
        """
        if self.task_type == TaskType.CLASSIFICATION:
            return np.argmax(self.mean, axis=-1)
        else:
            return self.mean.squeeze()
    
    def is_confident(self, threshold: float = 0.8) -> bool:
        """
        判断是否足够置信
        
        Args:
            threshold: 置信度阈值
            
        Returns:
            是否超过阈值
        """
        return self.confidence_score >= threshold
    
    def get_uncertainty_breakdown(self) -> Dict[str, float]:
        """
        获取不确定性分解
        
        Returns:
            包含各类不确定性指标的字典
        """
        return {
            'total_uncertainty': float(self.entropy),
            'epistemic_uncertainty': float(self.mutual_info),
            'aleatoric_uncertainty': float(self.expected_entropy),
            'confidence': float(self.confidence_score)
        }


@dataclass
class MCDropoutConfig:
    """
    MC Dropout配置类
    
    Attributes:
        n_samples: MC采样次数 (默认50)
        dropout_rate: Dropout比率 (默认0.1)
        epsilon: 数值稳定性常数 (默认1e-10)
        temperature: 温度缩放参数 (默认1.0)
        return_samples: 是否返回原始样本
        device: 计算设备
    """
    n_samples: int = 50
    dropout_rate: float = 0.1
    epsilon: float = 1e-10
    temperature: float = 1.0
    return_samples: bool = False
    device: Optional[str] = None


# ============================================================================
# 核心类
# ============================================================================

class MCConfidenceEstimator:
    """
    MC Dropout置信度估计器
    
    使用蒙特卡洛Dropout方法估计神经网络预测的不确定性。
    通过多次前向传播（保持dropout开启）获取预测分布，
    计算预测均值、方差、熵等作为置信度指标。
    
    支持分类和回归任务的不同置信度计算方法。
    
    Attributes:
        config: MC Dropout配置
        device: 计算设备
        _lock: 线程锁
        _history: 估计历史记录
    
    示例:
        >>> import torch.nn as nn
        >>> model = nn.Sequential(
        ...     nn.Linear(10, 64),
        ...     nn.ReLU(),
        ...     nn.Dropout(0.1),
        ...     nn.Linear(64, 3)
        ... )
        >>> 
        >>> estimator = MCConfidenceEstimator(n_samples=100)
        >>> x = torch.randn(5, 10)  # 5个样本
        >>> 
        >>> # 分类任务
        >>> result = estimator.estimate(model, x, task_type='classification')
        >>> print(f"置信度: {result.confidence_score:.3f}")
        >>> print(f"预测: {result.get_prediction()}")
        >>> 
        >>> # 回归任务
        >>> result = estimator.estimate(model, x, task_type='regression')
        >>> print(f"预测均值: {result.mean}")
        >>> print(f"预测方差: {result.variance}")
    """
    
    def __init__(
        self,
        n_samples: int = 50,
        dropout_rate: float = 0.1,
        epsilon: float = 1e-10,
        temperature: float = 1.0,
        device: Optional[str] = None,
        state_bus: Optional[Any] = None,
        history_window: int = 100
    ):
        """
        初始化MC Dropout置信度估计器
        
        Args:
            n_samples: MC采样次数，越多越准确但越慢
            dropout_rate: Dropout比率，用于估计不确定性
            epsilon: 数值稳定性常数
            temperature: 温度缩放参数，用于校准置信度
            device: 计算设备 ('cuda', 'cpu', 或 None自动选择)
            state_bus: 状态总线实例，用于与胜复学架构集成
            history_window: 历史记录窗口大小
        """
        self.config = MCDropoutConfig(
            n_samples=n_samples,
            dropout_rate=dropout_rate,
            epsilon=epsilon,
            temperature=temperature,
            device=device
        )
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._lock = threading.RLock()
        self._history: deque = deque(maxlen=history_window)
        self._state_bus = state_bus
        
        # 统计信息
        self._total_estimates = 0
        self._total_samples = 0
    
    def estimate(
        self,
        model: nn.Module,
        x: Union[np.ndarray, torch.Tensor],
        task_type: Union[str, TaskType] = TaskType.CLASSIFICATION,
        n_samples: Optional[int] = None,
        return_samples: bool = False
    ) -> ConfidenceEstimate:
        """
        估计预测置信度
        
        执行MC Dropout多次前向传播，计算各种置信度指标。
        
        Args:
            model: 神经网络模型（需要包含Dropout层）
            x: 输入数据 [batch_size, ...]
            task_type: 任务类型 ('classification', 'regression', 'multi_label')
            n_samples: MC采样次数（覆盖配置中的值）
            return_samples: 是否返回原始MC样本
            
        Returns:
            ConfidenceEstimate对象，包含所有置信度指标
            
        Raises:
            ValueError: 当输入参数无效时
            RuntimeError: 当模型前向传播失败时
            
        示例:
            >>> result = estimator.estimate(model, x, task_type='classification')
            >>> print(f"置信度: {result.confidence_score:.3f}")
            >>> print(f"预测熵: {result.entropy:.3f}")
            >>> print(f"互信息: {result.mutual_info:.3f}")
        """
        # 参数验证和转换
        if isinstance(task_type, str):
            task_type = TaskType(task_type)
        
        n_samples = n_samples or self.config.n_samples
        if n_samples < 2:
            raise ValueError(f"n_samples必须至少为2，当前值: {n_samples}")
        
        # 转换输入为tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"输入必须是numpy数组或torch张量，当前类型: {type(x)}")
        
        x = x.to(self.device)
        model = model.to(self.device)
        
        # 保持训练模式以启用dropout
        model.train()
        
        # 执行MC Dropout采样
        with torch.no_grad():
            samples = self._mc_dropout_forward(model, x, n_samples)
        
        # 根据任务类型计算置信度指标
        if task_type == TaskType.CLASSIFICATION:
            estimate = self._compute_classification_metrics(samples)
        elif task_type == TaskType.REGRESSION:
            estimate = self._compute_regression_metrics(samples)
        elif task_type == TaskType.MULTI_LABEL:
            estimate = self._compute_multilabel_metrics(samples)
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")
        
        estimate.task_type = task_type
        if return_samples:
            estimate.samples = samples
        
        # 更新历史记录
        with self._lock:
            self._history.append(estimate)
            self._total_estimates += 1
            self._total_samples += x.shape[0]
        
        # 发布到状态总线（如果配置）
        self._publish_to_state_bus(estimate)
        
        return estimate
    
    def _mc_dropout_forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        n_samples: int
    ) -> np.ndarray:
        """
        执行MC Dropout前向传播
        
        Args:
            model: 神经网络模型
            x: 输入张量
            n_samples: 采样次数
            
        Returns:
            预测样本数组 [n_samples, batch_size, num_classes/output_dim]
        """
        predictions = []
        
        for _ in range(n_samples):
            output = model(x)
            
            # 处理不同输出格式
            if isinstance(output, tuple):
                output = output[0]
            
            # 应用温度缩放
            if self.config.temperature != 1.0:
                output = output / self.config.temperature
            
            predictions.append(output.cpu().numpy())
        
        return np.stack(predictions, axis=0)
    
    def _compute_classification_metrics(
        self,
        samples: np.ndarray
    ) -> ConfidenceEstimate:
        """
        计算分类任务的置信度指标
        
        Args:
            samples: MC样本 [n_samples, batch_size, num_classes]
            
        Returns:
            ConfidenceEstimate对象
        """
        n_samples, batch_size, num_classes = samples.shape
        epsilon = self.config.epsilon
        
        # 转换为概率
        if np.any(samples < 0) or np.any(samples > 1):
            # 假设是logits，使用softmax
            exp_samples = np.exp(samples - np.max(samples, axis=-1, keepdims=True))
            probs = exp_samples / np.sum(exp_samples, axis=-1, keepdims=True)
        else:
            # 已经是概率
            probs = samples
        
        # 预测均值 (平均预测分布)
        mean_probs = np.mean(probs, axis=0)  # [batch_size, num_classes]
        
        # 预测方差
        variance = np.var(probs, axis=0)
        
        # 预测熵 H(y|x,D) = -Σ p(y|x,D) * log(p(y|x,D))
        predictive_entropy = -np.sum(mean_probs * np.log(mean_probs + epsilon), axis=-1)
        avg_predictive_entropy = float(np.mean(predictive_entropy))
        
        # 期望熵 E[H(y|x,θ)] = (1/T) Σ H(y|x,θ_t)
        expected_entropy_per_sample = -np.sum(probs * np.log(probs + epsilon), axis=-1)
        avg_expected_entropy = float(np.mean(expected_entropy_per_sample))
        
        # 互信息 I(y;θ|x,D) = H(y|x,D) - E[H(y|x,θ)]
        mutual_info = avg_predictive_entropy - avg_expected_entropy
        mutual_info = max(0.0, mutual_info)  # 确保非负
        
        # 最大概率
        max_prob = float(np.mean(np.max(mean_probs, axis=-1)))
        
        # 认知不确定性 (互信息)
        epistemic = mutual_info
        
        # 偶然不确定性 (期望熵)
        aleatoric = avg_expected_entropy
        
        # 综合置信度分数
        # 基于熵的倒数，归一化到[0,1]
        max_entropy = np.log(num_classes)
        confidence_score = 1.0 - (avg_predictive_entropy / max_entropy)
        confidence_score = max(0.0, min(1.0, confidence_score))
        
        return ConfidenceEstimate(
            mean=mean_probs,
            variance=variance,
            entropy=avg_predictive_entropy,
            expected_entropy=avg_expected_entropy,
            mutual_info=mutual_info,
            max_prob=max_prob,
            confidence_score=confidence_score,
            epistemic=np.array([epistemic]),
            aleatoric=np.array([aleatoric])
        )
    
    def _compute_regression_metrics(
        self,
        samples: np.ndarray
    ) -> ConfidenceEstimate:
        """
        计算回归任务的置信度指标
        
        Args:
            samples: MC样本 [n_samples, batch_size, output_dim]
            
        Returns:
            ConfidenceEstimate对象
        """
        n_samples, batch_size, output_dim = samples.shape if samples.ndim == 3 else (samples.shape[0], samples.shape[1], 1)
        
        # 预测均值
        mean_pred = np.mean(samples, axis=0)
        
        # 预测方差 (总不确定性)
        variance = np.var(samples, axis=0)
        avg_variance = float(np.mean(variance))
        
        # 对于回归任务，使用方差作为不确定性度量
        # 熵使用方差的对数近似
        epsilon = self.config.epsilon
        predictive_entropy = 0.5 * np.log(2 * np.pi * np.e * (variance + epsilon))
        avg_predictive_entropy = float(np.mean(predictive_entropy))
        
        # 期望熵 (对于回归，假设噪声模型)
        # 简化为方差的一部分
        expected_entropy = avg_predictive_entropy * 0.5
        
        # 互信息 (认知不确定性)
        mutual_info = avg_predictive_entropy - expected_entropy
        mutual_info = max(0.0, mutual_info)
        
        # 最大概率 (回归任务使用置信区间概率)
        max_prob = float(np.exp(-avg_variance))
        
        # 综合置信度分数
        # 基于方差的倒数
        confidence_score = 1.0 / (1.0 + avg_variance)
        
        return ConfidenceEstimate(
            mean=mean_pred,
            variance=variance,
            entropy=avg_predictive_entropy,
            expected_entropy=expected_entropy,
            mutual_info=mutual_info,
            max_prob=max_prob,
            confidence_score=confidence_score,
            epistemic=np.array([mutual_info]),
            aleatoric=np.array([expected_entropy]),
            task_type=TaskType.REGRESSION
        )
    
    def _compute_multilabel_metrics(
        self,
        samples: np.ndarray
    ) -> ConfidenceEstimate:
        """
        计算多标签分类任务的置信度指标
        
        Args:
            samples: MC样本 [n_samples, batch_size, num_labels]
            
        Returns:
            ConfidenceEstimate对象
        """
        # 多标签任务使用sigmoid激活
        probs = 1 / (1 + np.exp(-samples))
        
        # 预测均值
        mean_probs = np.mean(probs, axis=0)
        
        # 预测方差
        variance = np.var(probs, axis=0)
        
        # 每个标签的熵
        epsilon = self.config.epsilon
        label_entropy = -mean_probs * np.log(mean_probs + epsilon) - \
                        (1 - mean_probs) * np.log(1 - mean_probs + epsilon)
        predictive_entropy = float(np.mean(np.sum(label_entropy, axis=-1)))
        
        # 期望熵
        expected_entropy_per_label = -probs * np.log(probs + epsilon) - \
                                      (1 - probs) * np.log(1 - probs + epsilon)
        expected_entropy = float(np.mean(expected_entropy_per_label))
        
        # 互信息
        mutual_info = predictive_entropy - expected_entropy
        mutual_info = max(0.0, mutual_info)
        
        # 最大概率
        max_prob = float(np.mean(np.max(mean_probs, axis=-1)))
        
        # 置信度分数
        confidence_score = 1.0 - (predictive_entropy / mean_probs.shape[-1])
        confidence_score = max(0.0, min(1.0, confidence_score))
        
        return ConfidenceEstimate(
            mean=mean_probs,
            variance=variance,
            entropy=predictive_entropy,
            expected_entropy=expected_entropy,
            mutual_info=mutual_info,
            max_prob=max_prob,
            confidence_score=confidence_score,
            task_type=TaskType.MULTI_LABEL
        )
    
    def _publish_to_state_bus(self, estimate: ConfidenceEstimate) -> None:
        """
        将置信度估计结果发布到状态总线
        
        Args:
            estimate: 置信度估计结果
        """
        if self._state_bus is None or not _HAS_STATE_BUS:
            return
        
        try:
            # 根据置信度更新胜值(swing)
            self._state_bus.update_coefficient("swing", estimate.confidence_score)
            
            # 如果置信度低，可能增加郁值
            if estimate.confidence_score < 0.5:
                current_halt = self._state_bus.get_current_state().halt
                new_halt = min(1.0, current_halt + 0.1)
                self._state_bus.update_coefficient("halt", new_halt)
        except Exception:
            # 忽略状态总线错误
            pass
    
    def estimate_batch(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        task_type: Union[str, TaskType] = TaskType.CLASSIFICATION,
        n_samples: Optional[int] = None
    ) -> List[ConfidenceEstimate]:
        """
        批量估计整个数据集的置信度
        
        Args:
            model: 神经网络模型
            dataloader: 数据加载器
            task_type: 任务类型
            n_samples: MC采样次数
            
        Returns:
            ConfidenceEstimate列表
        """
        results = []
        
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch
            
            estimate = self.estimate(model, x, task_type, n_samples)
            results.append(estimate)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取估计器统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            stats = {
                'total_estimates': self._total_estimates,
                'total_samples': self._total_samples,
                'config': {
                    'n_samples': self.config.n_samples,
                    'dropout_rate': self.config.dropout_rate,
                    'temperature': self.config.temperature,
                    'device': self.device
                }
            }
            
            if len(self._history) > 0:
                recent = list(self._history)[-10:]
                stats['recent_confidence'] = {
                    'mean': float(np.mean([e.confidence_score for e in recent])),
                    'std': float(np.std([e.confidence_score for e in recent])),
                    'min': float(np.min([e.confidence_score for e in recent])),
                    'max': float(np.max([e.confidence_score for e in recent]))
                }
            
            return stats
    
    def calibrate_temperature(
        self,
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        task_type: Union[str, TaskType] = TaskType.CLASSIFICATION,
        n_samples: int = 20
    ) -> float:
        """
        使用验证集校准温度参数
        
        Args:
            model: 神经网络模型
            val_loader: 验证数据加载器
            task_type: 任务类型
            n_samples: MC采样次数（校准使用较少样本）
            
        Returns:
            最优温度值
        """
        best_temp = 1.0
        best_nll = float('inf')
        
        for temp in np.linspace(0.5, 2.0, 20):
            self.config.temperature = temp
            nll = self._compute_validation_nll(model, val_loader, task_type, n_samples)
            
            if nll < best_nll:
                best_nll = nll
                best_temp = temp
        
        self.config.temperature = best_temp
        return best_temp
    
    def _compute_validation_nll(
        self,
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        task_type: Union[str, TaskType],
        n_samples: int
    ) -> float:
        """
        计算验证集负对数似然
        
        Args:
            model: 神经网络模型
            val_loader: 验证数据加载器
            task_type: 任务类型
            n_samples: MC采样次数
            
        Returns:
            平均负对数似然
        """
        total_nll = 0.0
        total_samples = 0
        epsilon = self.config.epsilon
        
        for batch in val_loader:
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            else:
                continue
            
            estimate = self.estimate(model, x, task_type, n_samples)
            
            if task_type == TaskType.CLASSIFICATION:
                # 分类任务NLL
                mean_probs = estimate.mean
                y_indices = y.numpy() if isinstance(y, torch.Tensor) else y
                nll = -np.log(mean_probs[np.arange(len(y_indices)), y_indices] + epsilon)
                total_nll += np.sum(nll)
                total_samples += len(y_indices)
            elif task_type == TaskType.REGRESSION:
                # 回归任务使用MSE近似
                total_nll += np.sum(estimate.variance)
                total_samples += estimate.variance.shape[0]
        
        return total_nll / max(total_samples, 1)
    
    def reset_history(self) -> None:
        """重置历史记录"""
        with self._lock:
            self._history.clear()
            self._total_estimates = 0
            self._total_samples = 0


# ============================================================================
# 辅助函数
# ============================================================================

def compute_predictive_entropy(
    predictions: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    计算预测熵 H = -Σ p_i * log(p_i)
    
    Args:
        predictions: 预测概率分布 [batch_size, num_classes] 或 [num_classes]
        epsilon: 数值稳定性常数
        
    Returns:
        预测熵值
        
    示例:
        >>> probs = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        >>> entropy = compute_predictive_entropy(probs)
        >>> print(f"预测熵: {entropy:.3f}")
    """
    # 确保是概率分布
    if np.any(predictions < 0) or np.any(predictions > 1):
        exp_preds = np.exp(predictions - np.max(predictions, axis=-1, keepdims=True))
        predictions = exp_preds / np.sum(exp_preds, axis=-1, keepdims=True)
    
    # 计算熵
    entropy = -np.sum(predictions * np.log(predictions + epsilon), axis=-1)
    return float(np.mean(entropy))


def compute_mutual_information(
    mc_samples: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    计算互信息 (BALD) I(y; θ) = H(y) - E[H(y|θ)]
    
    Args:
        mc_samples: MC Dropout样本 [n_samples, batch_size, num_classes]
        epsilon: 数值稳定性常数
        
    Returns:
        互信息值
        
    示例:
        >>> samples = np.random.randn(50, 10, 5)  # 50次采样，10个样本，5个类别
        >>> mi = compute_mutual_information(samples)
        >>> print(f"互信息: {mi:.3f}")
    """
    # 转换为概率
    if np.any(mc_samples < 0) or np.any(mc_samples > 1):
        exp_samples = np.exp(mc_samples - np.max(mc_samples, axis=-1, keepdims=True))
        probs = exp_samples / np.sum(exp_samples, axis=-1, keepdims=True)
    else:
        probs = mc_samples
    
    # 平均预测分布
    mean_probs = np.mean(probs, axis=0)
    
    # H(y|x,D) - 预测熵
    predictive_entropy = -np.sum(mean_probs * np.log(mean_probs + epsilon), axis=-1)
    avg_predictive_entropy = np.mean(predictive_entropy)
    
    # E[H(y|x,θ)] - 期望熵
    expected_entropy = 0.0
    for i in range(probs.shape[0]):
        p = probs[i]
        h = -np.sum(p * np.log(p + epsilon), axis=-1)
        expected_entropy += np.mean(h)
    expected_entropy /= probs.shape[0]
    
    # 互信息
    mutual_info = avg_predictive_entropy - expected_entropy
    return float(max(0.0, mutual_info))


def compute_bald_score(
    mc_samples: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    计算BALD (Bayesian Active Learning by Disagreement) 分数
    
    BALD分数等同于互信息，用于主动学习选择最有价值的样本。
    
    Args:
        mc_samples: MC Dropout样本
        epsilon: 数值稳定性常数
        
    Returns:
        BALD分数
    """
    return compute_mutual_information(mc_samples, epsilon)


def estimate_confidence_mc_dropout(
    model: nn.Module,
    x: Union[np.ndarray, torch.Tensor],
    n_samples: int = 50,
    task_type: str = "classification",
    device: Optional[str] = None
) -> ConfidenceEstimate:
    """
    使用MC Dropout估计置信度的便捷函数
    
    Args:
        model: 神经网络模型
        x: 输入数据
        n_samples: MC采样次数
        task_type: 任务类型
        device: 计算设备
        
    Returns:
        ConfidenceEstimate对象
        
    示例:
        >>> result = estimate_confidence_mc_dropout(
        ...     model, x, n_samples=100, task_type='classification'
        ... )
        >>> print(f"置信度: {result.confidence_score:.3f}")
    """
    estimator = MCConfidenceEstimator(n_samples=n_samples, device=device)
    return estimator.estimate(model, x, task_type=task_type)


# ============================================================================
# 与StateBus集成的便捷函数
# ============================================================================

def create_confidence_estimator_with_state_bus(
    state_bus: Optional[Any] = None,
    **kwargs
) -> MCConfidenceEstimator:
    """
    创建与StateBus集成的置信度估计器
    
    Args:
        state_bus: 状态总线实例（None则使用全局实例）
        **kwargs: 传递给MCConfidenceEstimator的其他参数
        
    Returns:
        配置好的MCConfidenceEstimator实例
    """
    if state_bus is None and _HAS_STATE_BUS:
        state_bus = get_global_state_bus()
    
    return MCConfidenceEstimator(state_bus=state_bus, **kwargs)


# 导出公共接口
__all__ = [
    'MCConfidenceEstimator',
    'ConfidenceEstimate',
    'MCDropoutConfig',
    'TaskType',
    'ConfidenceMetric',
    'compute_predictive_entropy',
    'compute_mutual_information',
    'compute_bald_score',
    'estimate_confidence_mc_dropout',
    'create_confidence_estimator_with_state_bus'
]
