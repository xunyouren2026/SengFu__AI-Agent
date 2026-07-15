"""
PGD对抗训练 - 投影梯度下降对抗训练
"""
import math
import random
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum


class AttackType(Enum):
    """攻击类型"""
    L_INF = "l_inf"
    L_2 = "l_2"
    L_1 = "l_1"


@dataclass
class AdversarialExample:
    """对抗样本"""
    original: List[float]
    perturbed: List[float]
    perturbation: List[float]
    original_label: int
    adversarial_label: int
    confidence: float


@dataclass
class PGDConfig:
    """PGD配置"""
    epsilon: float = 0.1          # 最大扰动
    alpha: float = 0.01           # 步长
    num_steps: int = 40           # 迭代次数
    attack_type: AttackType = AttackType.L_INF
    random_start: bool = True     # 随机初始化
    clip_min: float = 0.0         # 裁剪下界
    clip_max: float = 1.0         # 裁剪上界


class PGDTrainer:
    """PGD对抗训练器"""
    
    def __init__(self, config: Optional[PGDConfig] = None):
        self._config = config or PGDConfig()
        self._training_history: List[Dict[str, float]] = []
    
    def generate_perturbation(
        self,
        gradient: List[float],
        epsilon: float
    ) -> List[float]:
        """生成扰动"""
        if self._config.attack_type == AttackType.L_INF:
            # L-inf: 符号梯度
            return [epsilon * math.copysign(1, g) if g != 0 else 0 for g in gradient]
        
        elif self._config.attack_type == AttackType.L_2:
            # L-2: 归一化梯度
            norm = math.sqrt(sum(g * g for g in gradient))
            if norm > 0:
                return [epsilon * g / norm for g in gradient]
            return [0] * len(gradient)
        
        elif self._config.attack_type == AttackType.L_1:
            # L-1: 稀疏扰动
            max_idx = max(range(len(gradient)), key=lambda i: abs(gradient[i]))
            perturbation = [0] * len(gradient)
            perturbation[max_idx] = epsilon * math.copysign(1, gradient[max_idx])
            return perturbation
        
        return [0] * len(gradient)
    
    def project_perturbation(
        self,
        perturbation: List[float],
        epsilon: float
    ) -> List[float]:
        """投影扰动到约束球"""
        if self._config.attack_type == AttackType.L_INF:
            # L-inf: 裁剪到[-epsilon, epsilon]
            return [max(-epsilon, min(epsilon, p)) for p in perturbation]
        
        elif self._config.attack_type == AttackType.L_2:
            # L-2: 归一化到epsilon球
            norm = math.sqrt(sum(p * p for p in perturbation))
            if norm > epsilon:
                return [epsilon * p / norm for p in perturbation]
            return perturbation
        
        elif self._config.attack_type == AttackType.L_1:
            # L-1: 投影到L-1球
            l1_norm = sum(abs(p) for p in perturbation)
            if l1_norm > epsilon:
                return [epsilon * p / l1_norm for p in perturbation]
            return perturbation
        
        return perturbation
    
    def pgd_attack(
        self,
        input_data: List[float],
        gradient_fn: Callable[[List[float]], List[float]],
        predict_fn: Callable[[List[float]], Tuple[int, float]]
    ) -> AdversarialExample:
        """PGD攻击"""
        original_label, original_conf = predict_fn(input_data)
        
        # 初始化扰动
        if self._config.random_start:
            perturbation = self._random_perturbation()
        else:
            perturbation = [0] * len(input_data)
        
        # 迭代优化
        for step in range(self._config.num_steps):
            # 计算当前对抗样本
            current = [
                self._clip(input_data[i] + perturbation[i])
                for i in range(len(input_data))
            ]
            
            # 计算梯度
            gradient = gradient_fn(current)
            
            # 更新扰动
            delta = self.generate_perturbation(gradient, self._config.alpha)
            perturbation = [
                perturbation[i] + delta[i]
                for i in range(len(perturbation))
            ]
            
            # 投影到约束球
            perturbation = self.project_perturbation(perturbation, self._config.epsilon)
        
        # 生成最终对抗样本
        perturbed = [
            self._clip(input_data[i] + perturbation[i])
            for i in range(len(input_data))
        ]
        
        adv_label, adv_conf = predict_fn(perturbed)
        
        return AdversarialExample(
            original=input_data,
            perturbed=perturbed,
            perturbation=perturbation,
            original_label=original_label,
            adversarial_label=adv_label,
            confidence=adv_conf
        )
    
    def _random_perturbation(self) -> List[float]:
        """生成随机初始扰动"""
        if self._config.attack_type == AttackType.L_INF:
            return [
                random.uniform(-self._config.epsilon, self._config.epsilon)
                for _ in range(100)  # 假设维度
            ]
        elif self._config.attack_type == AttackType.L_2:
            # 随机方向，固定长度
            direction = [random.gauss(0, 1) for _ in range(100)]
            norm = math.sqrt(sum(d * d for d in direction))
            return [self._config.epsilon * d / norm for d in direction]
        return [0] * 100
    
    def _clip(self, value: float) -> float:
        """裁剪值"""
        return max(self._config.clip_min, min(self._config.clip_max, value))
    
    def adversarial_training_step(
        self,
        input_data: List[float],
        label: int,
        gradient_fn: Callable[[List[float]], List[float]],
        loss_fn: Callable[[List[float], int], float],
        update_fn: Callable[[List[float]], None]
    ) -> Dict[str, float]:
        """对抗训练步骤"""
        # 生成对抗样本
        def predict(x):
            # 简化的预测函数
            return (label, 0.9)
        
        adv_example = self.pgd_attack(input_data, gradient_fn, predict)
        
        # 计算对抗损失
        adv_loss = loss_fn(adv_example.perturbed, label)
        clean_loss = loss_fn(input_data, label)
        
        # 混合损失
        total_loss = 0.5 * clean_loss + 0.5 * adv_loss
        
        # 更新模型（通过回调）
        adv_gradient = gradient_fn(adv_example.perturbed)
        clean_gradient = gradient_fn(input_data)
        mixed_gradient = [
            0.5 * clean_gradient[i] + 0.5 * adv_gradient[i]
            for i in range(len(clean_gradient))
        ]
        update_fn(mixed_gradient)
        
        # 记录历史
        result = {
            "clean_loss": clean_loss,
            "adversarial_loss": adv_loss,
            "total_loss": total_loss,
            "perturbation_norm": math.sqrt(sum(p * p for p in adv_example.perturbation))
        }
        self._training_history.append(result)
        
        return result
    
    def evaluate_robustness(
        self,
        test_data: List[Tuple[List[float], int]],
        predict_fn: Callable[[List[float]], Tuple[int, float]],
        gradient_fn: Callable[[List[float]], List[float]]
    ) -> Dict[str, float]:
        """评估鲁棒性"""
        clean_correct = 0
        adv_correct = 0
        total = len(test_data)
        
        for input_data, label in test_data:
            # 干净样本准确率
            pred_label, _ = predict_fn(input_data)
            if pred_label == label:
                clean_correct += 1
            
            # 对抗样本准确率
            adv_example = self.pgd_attack(input_data, gradient_fn, predict_fn)
            if adv_example.adversarial_label == label:
                adv_correct += 1
        
        return {
            "clean_accuracy": clean_correct / total,
            "adversarial_accuracy": adv_correct / total,
            "robustness_gap": (clean_correct - adv_correct) / total
        }
    
    def get_training_history(self) -> List[Dict[str, float]]:
        """获取训练历史"""
        return self._training_history.copy()


class AdversarialDetector:
    """对抗样本检测器"""
    
    def __init__(self):
        self._threshold = 0.1
        self._statistics: Dict[str, List[float]] = {}
    
    def detect(
        self,
        input_data: List[float],
        reconstruction_fn: Callable[[List[float]], List[float]]
    ) -> Tuple[bool, float]:
        """检测对抗样本"""
        # 重建输入
        reconstructed = reconstruction_fn(input_data)
        
        # 计算重建误差
        error = math.sqrt(sum(
            (input_data[i] - reconstructed[i]) ** 2
            for i in range(len(input_data))
        ))
        
        # 判断是否为对抗样本
        is_adversarial = error > self._threshold
        
        return is_adversarial, error
    
    def feature_squeezing(
        self,
        input_data: List[float],
        bit_depth: int = 8
    ) -> List[float]:
        """特征压缩"""
        levels = 2 ** bit_depth
        return [
            round(x * levels) / levels
            for x in input_data
        ]
    
    def detect_by_squeezing(
        self,
        input_data: List[float],
        predict_fn: Callable[[List[float]], int],
        bit_depths: List[int] = None
    ) -> Tuple[bool, float]:
        """通过特征压缩检测"""
        if bit_depths is None:
            bit_depths = [4, 8, 16]
        
        original_pred = predict_fn(input_data)
        
        disagreements = 0
        for bd in bit_depths:
            squeezed = self.feature_squeezing(input_data, bd)
            squeezed_pred = predict_fn(squeezed)
            if squeezed_pred != original_pred:
                disagreements += 1
        
        is_adversarial = disagreements > len(bit_depths) / 2
        confidence = disagreements / len(bit_depths)
        
        return is_adversarial, confidence
