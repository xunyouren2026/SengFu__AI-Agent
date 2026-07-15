"""
不确定性估计模块 (Uncertainty Estimation Module)

该模块实现了多种不确定性估计方法，包括MC Dropout、集成方法、
预测熵计算、互信息估计等。支持认知不确定性和偶然不确定性的分离。

核心功能:
- MC Dropout不确定性
- 集成方法不确定性
- 预测熵计算
- 互信息估计
- 认知不确定性vs偶然不确定性分离

作者: AGI Universal Framework Team
版本: 1.0.0
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass
from enum import Enum
import threading
from collections import defaultdict


class UncertaintyType(Enum):
    """不确定性类型枚举"""
    EPISTEMIC = "epistemic"      # 认知不确定性（模型不确定性）
    ALEATORIC = "aleatoric"      # 偶然不确定性（数据不确定性）
    TOTAL = "total"              # 总不确定性
    PREDICTIVE = "predictive"    # 预测不确定性


@dataclass
class UncertaintyEstimate:
    """
    不确定性估计结果数据类
    
    Attributes:
        mean: 预测均值
        variance: 预测方差
        entropy: 预测熵
        epistemic: 认知不确定性
        aleatoric: 偶然不确定性
        mutual_info: 互信息
        confidence: 置信度
        samples: 原始预测样本（可选）
    """
    mean: np.ndarray
    variance: np.ndarray
    entropy: float
    epistemic: Optional[np.ndarray] = None
    aleatoric: Optional[np.ndarray] = None
    mutual_info: Optional[float] = None
    confidence: Optional[float] = None
    samples: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'mean': self.mean,
            'variance': self.variance,
            'entropy': float(self.entropy),
        }
        
        if self.epistemic is not None:
            result['epistemic'] = self.epistemic
        if self.aleatoric is not None:
            result['aleatoric'] = self.aleatoric
        if self.mutual_info is not None:
            result['mutual_info'] = float(self.mutual_info)
        if self.confidence is not None:
            result['confidence'] = float(self.confidence)
        
        return result


class MCDropoutLayer(nn.Module):
    """
    MC Dropout层
    
    在训练和推理时都启用dropout的层。
    """
    
    def __init__(self, p: float = 0.5):
        """
        初始化MC Dropout层
        
        Args:
            p: dropout概率
        """
        super().__init__()
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，始终使用dropout"""
        return F.dropout(x, p=self.p, training=True)


class UncertaintyEstimator:
    """
    不确定性估计器
    
    实现多种不确定性估计方法，包括MC Dropout、集成方法等。
    
    Attributes:
        device: 计算设备
        n_samples: MC采样次数
        epsilon: 数值稳定性常数
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        n_samples: int = 50,
        epsilon: float = 1e-10
    ):
        """
        初始化不确定性估计器
        
        Args:
            device: 计算设备
            n_samples: MC采样次数
            epsilon: 数值稳定性常数
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_samples = n_samples
        self.epsilon = epsilon
        self._lock = threading.RLock()
    
    def mc_dropout_uncertainty(
        self,
        model: nn.Module,
        x: Union[np.ndarray, torch.Tensor],
        n_samples: Optional[int] = None,
        return_samples: bool = False
    ) -> UncertaintyEstimate:
        """
        使用MC Dropout估计不确定性
        
        通过多次前向传播（启用dropout）来估计预测分布。
        
        Args:
            model: 神经网络模型（需要包含dropout层）
            x: 输入数据
            n_samples: MC采样次数
            return_samples: 是否返回原始样本
            
        Returns:
            不确定性估计结果
        """
        n_samples = n_samples or self.n_samples
        
        # 转换输入为tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        x = x.to(self.device)
        model = model.to(self.device)
        model.train()  # 保持训练模式以启用dropout
        
        # 多次前向传播
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = model(x)
                if isinstance(pred, torch.Tensor):
                    predictions.append(pred.cpu().numpy())
                else:
                    predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # 计算统计量
        mean = np.mean(predictions, axis=0)
        variance = np.var(predictions, axis=0)
        
        # 计算预测熵
        entropy = self.predictive_entropy(predictions)
        
        # 分离认知和偶然不确定性
        epistemic, aleatoric = self._separate_uncertainties(predictions)
        
        # 计算互信息
        mutual_info = self._compute_mutual_information(predictions, mean)
        
        # 计算置信度
        confidence = self._compute_confidence(predictions)
        
        return UncertaintyEstimate(
            mean=mean,
            variance=variance,
            entropy=entropy,
            epistemic=epistemic,
            aleatoric=aleatoric,
            mutual_info=mutual_info,
            confidence=confidence,
            samples=predictions if return_samples else None
        )
    
    def ensemble_uncertainty(
        self,
        models: List[nn.Module],
        x: Union[np.ndarray, torch.Tensor],
        return_samples: bool = False
    ) -> UncertaintyEstimate:
        """
        使用集成方法估计不确定性
        
        通过多个模型的预测差异来估计不确定性。
        
        Args:
            models: 模型列表
            x: 输入数据
            return_samples: 是否返回原始样本
            
        Returns:
            不确定性估计结果
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        x = x.to(self.device)
        
        # 收集所有模型的预测
        predictions = []
        for model in models:
            model = model.to(self.device)
            model.eval()
            
            with torch.no_grad():
                pred = model(x)
                if isinstance(pred, torch.Tensor):
                    predictions.append(pred.cpu().numpy())
                else:
                    predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # 计算统计量
        mean = np.mean(predictions, axis=0)
        variance = np.var(predictions, axis=0)
        
        # 计算预测熵
        entropy = self.predictive_entropy(predictions)
        
        # 分离认知和偶然不确定性
        epistemic, aleatoric = self._separate_uncertainties(predictions)
        
        # 计算互信息
        mutual_info = self._compute_mutual_information(predictions, mean)
        
        # 计算置信度
        confidence = self._compute_confidence(predictions)
        
        return UncertaintyEstimate(
            mean=mean,
            variance=variance,
            entropy=entropy,
            epistemic=epistemic,
            aleatoric=aleatoric,
            mutual_info=mutual_info,
            confidence=confidence,
            samples=predictions if return_samples else None
        )
    
    def predictive_entropy(self, predictions: np.ndarray) -> float:
        """
        计算预测熵
        
        预测熵反映了模型对预测结果的不确定性。
        
        Args:
            predictions: 预测样本数组 [n_samples, ...]
            
        Returns:
            预测熵值
        """
        # 对于分类任务，计算类别分布的熵
        if predictions.ndim > 1 and predictions.shape[-1] > 1:
            # 假设是分类任务的logits或probabilities
            if np.all(predictions >= 0) and np.all(predictions <= 1):
                # 已经是概率
                probs = predictions
            else:
                # 转换为概率
                exp_preds = np.exp(predictions - np.max(predictions, axis=-1, keepdims=True))
                probs = exp_preds / np.sum(exp_preds, axis=-1, keepdims=True)
            
            # 平均预测分布
            mean_probs = np.mean(probs, axis=0)
            
            # 计算熵
            entropy = -np.sum(mean_probs * np.log(mean_probs + self.epsilon))
            
            return float(entropy)
        else:
            # 对于回归任务，使用方差作为不确定性的度量
            variance = np.var(predictions, axis=0)
            return float(np.mean(variance))
    
    def epistemic_uncertainty(
        self,
        model: nn.Module,
        x: Union[np.ndarray, torch.Tensor],
        n_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        计算认知不确定性（模型不确定性）
        
        认知不确定性反映了模型对数据的不确定性，可以通过更多数据减少。
        
        Args:
            model: 神经网络模型
            x: 输入数据
            n_samples: MC采样次数
            
        Returns:
            认知不确定性数组
        """
        estimate = self.mc_dropout_uncertainty(model, x, n_samples)
        return estimate.epistemic if estimate.epistemic is not None else estimate.variance
    
    def aleatoric_uncertainty(
        self,
        model: nn.Module,
        x: Union[np.ndarray, torch.Tensor],
        n_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        计算偶然不确定性（数据不确定性）
        
        偶然不确定性反映了数据本身的噪声，无法通过更多数据减少。
        
        Args:
            model: 神经网络模型
            x: 输入数据
            n_samples: MC采样次数
            
        Returns:
            偶然不确定性数组
        """
        estimate = self.mc_dropout_uncertainty(model, x, n_samples)
        return estimate.aleatoric if estimate.aleatoric is not None else np.zeros_like(estimate.variance)
    
    def mutual_information(
        self,
        model: nn.Module,
        x: Union[np.ndarray, torch.Tensor],
        n_samples: Optional[int] = None
    ) -> float:
        """
        计算预测分布和模型参数后验之间的互信息
        
        互信息可以反映模型对预测的不确定性。
        
        Args:
            model: 神经网络模型
            x: 输入数据
            n_samples: MC采样次数
            
        Returns:
            互信息值
        """
        estimate = self.mc_dropout_uncertainty(model, x, n_samples)
        return estimate.mutual_info if estimate.mutual_info is not None else 0.0
    
    def confidence_score(
        self,
        predictions: np.ndarray,
        method: str = "max_prob"
    ) -> float:
        """
        计算置信度分数
        
        Args:
            predictions: 预测数组
            method: 置信度计算方法
            
        Returns:
            置信度分数
        """
        if method == "max_prob":
            # 使用最大概率作为置信度
            if predictions.ndim > 1 and predictions.shape[-1] > 1:
                mean_pred = np.mean(predictions, axis=0)
                if np.all(mean_pred >= 0) and np.all(mean_pred <= 1):
                    probs = mean_pred
                else:
                    exp_pred = np.exp(mean_pred - np.max(mean_pred))
                    probs = exp_pred / np.sum(exp_pred)
                return float(np.max(probs))
            else:
                return 1.0 - float(np.std(predictions))
        
        elif method == "entropy":
            # 使用熵的倒数作为置信度
            entropy = self.predictive_entropy(predictions)
            max_entropy = np.log(predictions.shape[-1]) if predictions.ndim > 1 else 1.0
            return 1.0 - (entropy / max_entropy)
        
        elif method == "variance":
            # 使用方差的倒数作为置信度
            variance = np.var(predictions, axis=0)
            return 1.0 / (1.0 + np.mean(variance))
        
        else:
            raise ValueError(f"Unknown confidence method: {method}")
    
    def _separate_uncertainties(
        self,
        predictions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        分离认知不确定性和偶然不确定性
        
        总不确定性 = 认知不确定性 + 偶然不确定性
        
        Args:
            predictions: 预测样本数组 [n_samples, ...]
            
        Returns:
            (认知不确定性, 偶然不确定性)元组
        """
        # 预测均值
        mean_pred = np.mean(predictions, axis=0)
        
        # 总不确定性（预测方差）
        total_variance = np.var(predictions, axis=0)
        
        if predictions.ndim > 1 and predictions.shape[-1] > 1:
            # 对于分类任务
            # 认知不确定性：预测均值的信息增益
            mean_probs = np.mean(predictions, axis=0)
            
            # 偶然不确定性：平均预测熵
            aleatoric = np.zeros_like(mean_pred)
            for i in range(predictions.shape[0]):
                pred = predictions[i]
                if np.all(pred >= 0) and np.all(pred <= 1):
                    probs = pred
                else:
                    exp_pred = np.exp(pred - np.max(pred))
                    probs = exp_pred / np.sum(exp_pred)
                aleatoric -= probs * np.log(probs + self.epsilon)
            aleatoric /= predictions.shape[0]
            
            # 认知不确定性 = 总不确定性 - 偶然不确定性
            epistemic = total_variance - aleatoric
            epistemic = np.maximum(epistemic, 0)  # 确保非负
        else:
            # 对于回归任务，使用方差分解
            # 认知不确定性：模型预测的方差
            epistemic = np.var(predictions, axis=0)
            
            # 偶然不确定性：假设的噪声方差
            # 这里简化为总方差的一小部分
            aleatoric = total_variance * 0.1
        
        return epistemic, aleatoric
    
    def _compute_mutual_information(
        self,
        predictions: np.ndarray,
        mean_pred: np.ndarray
    ) -> float:
        """
        计算互信息
        
        I(y, θ|x, D) = H(y|x, D) - E_θ[H(y|x, θ)]
        
        Args:
            predictions: 预测样本数组
            mean_pred: 预测均值
            
        Returns:
            互信息值
        """
        if predictions.ndim <= 1 or predictions.shape[-1] <= 1:
            return 0.0
        
        # 预测分布的熵
        if np.all(predictions >= 0) and np.all(predictions <= 1):
            probs = predictions
        else:
            exp_preds = np.exp(predictions - np.max(predictions, axis=-1, keepdims=True))
            probs = exp_preds / np.sum(exp_preds, axis=-1, keepdims=True)
        
        # H(y|x, D) - 预测分布的熵
        mean_probs = np.mean(probs, axis=0)
        predictive_entropy = -np.sum(mean_probs * np.log(mean_probs + self.epsilon))
        
        # E_θ[H(y|x, θ)] - 平均预测熵
        expected_entropy = 0.0
        for i in range(probs.shape[0]):
            p = probs[i]
            expected_entropy -= np.sum(p * np.log(p + self.epsilon))
        expected_entropy /= probs.shape[0]
        
        # 互信息
        mutual_info = predictive_entropy - expected_entropy
        
        return float(max(0.0, mutual_info))
    
    def _compute_confidence(self, predictions: np.ndarray) -> float:
        """计算置信度"""
        return self.confidence_score(predictions, method="max_prob")
    
    def calibrate_confidence(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        method: str = "temperature"
    ) -> Dict[str, Any]:
        """
        校准置信度
        
        Args:
            predictions: 预测数组
            labels: 真实标签
            method: 校准方法
            
        Returns:
            校准结果
        """
        if method == "temperature":
            # 温度缩放
            best_temp = 1.0
            best_ece = float('inf')
            
            for temp in np.linspace(0.5, 2.0, 20):
                scaled_preds = predictions / temp
                ece = self._compute_ece(scaled_preds, labels)
                if ece < best_ece:
                    best_ece = ece
                    best_temp = temp
            
            return {
                'method': 'temperature',
                'temperature': float(best_temp),
                'ece': float(best_ece)
            }
        
        else:
            raise ValueError(f"Unknown calibration method: {method}")
    
    def _compute_ece(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        计算期望校准误差 (Expected Calibration Error)
        
        Args:
            predictions: 预测数组
            labels: 真实标签
            n_bins: bin数量
            
        Returns:
            ECE值
        """
        # 转换为概率
        if np.all(predictions >= 0) and np.all(predictions <= 1):
            probs = predictions
        else:
            exp_preds = np.exp(predictions - np.max(predictions, axis=-1, keepdims=True))
            probs = exp_preds / np.sum(exp_preds, axis=-1, keepdims=True)
        
        # 获取预测置信度和预测类别
        confidences = np.max(probs, axis=-1)
        predictions_class = np.argmax(probs, axis=-1)
        accuracies = (predictions_class == labels).astype(float)
        
        # 创建bin
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # 找到落在当前bin中的样本
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                avg_confidence = np.mean(confidences[in_bin])
                avg_accuracy = np.mean(accuracies[in_bin])
                ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin
        
        return float(ece)


class BayesianLinearLayer(nn.Module):
    """
    贝叶斯线性层
    
    使用变分推断近似贝叶斯神经网络。
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_sigma: float = 1.0
    ):
        """
        初始化贝叶斯线性层
        
        Args:
            in_features: 输入特征数
            out_features: 输出特征数
            prior_sigma: 先验标准差
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma
        
        # 权重均值和方差
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # 偏置均值和方差
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        # 初始化
        self.reset_parameters()
    
    def reset_parameters(self):
        """重置参数"""
        nn.init.normal_(self.weight_mu, mean=0, std=0.1)
        nn.init.constant_(self.weight_rho, -3)
        nn.init.normal_(self.bias_mu, mean=0, std=0.1)
        nn.init.constant_(self.bias_rho, -3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 重参数化采样
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """计算KL散度"""
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        # KL(q(w)||p(w))
        kl_weight = torch.sum(
            torch.log(self.prior_sigma / weight_sigma) +
            (weight_sigma ** 2 + self.weight_mu ** 2) / (2 * self.prior_sigma ** 2) -
            0.5
        )
        
        kl_bias = torch.sum(
            torch.log(self.prior_sigma / bias_sigma) +
            (bias_sigma ** 2 + self.bias_mu ** 2) / (2 * self.prior_sigma ** 2) -
            0.5
        )
        
        return kl_weight + kl_bias


# 便捷函数
def estimate_uncertainty_mc_dropout(
    model: nn.Module,
    x: Union[np.ndarray, torch.Tensor],
    n_samples: int = 50
) -> UncertaintyEstimate:
    """
    使用MC Dropout估计不确定性的便捷函数
    
    Args:
        model: 神经网络模型
        x: 输入数据
        n_samples: MC采样次数
        
    Returns:
        不确定性估计结果
    """
    estimator = UncertaintyEstimator(n_samples=n_samples)
    return estimator.mc_dropout_uncertainty(model, x, n_samples)


def estimate_uncertainty_ensemble(
    models: List[nn.Module],
    x: Union[np.ndarray, torch.Tensor]
) -> UncertaintyEstimate:
    """
    使用集成方法估计不确定性的便捷函数
    
    Args:
        models: 模型列表
        x: 输入数据
        
    Returns:
        不确定性估计结果
    """
    estimator = UncertaintyEstimator()
    return estimator.ensemble_uncertainty(models, x)
