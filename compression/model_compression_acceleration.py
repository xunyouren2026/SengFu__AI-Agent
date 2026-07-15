"""
模型压缩与加速模块 - Model Compression and Acceleration
实现剪枝、量化、知识蒸馏、低秩分解、NAS、稀疏训练、编译优化及端到端压缩流水线
纯 Python + numpy 实现，不依赖 PyTorch
"""

import numpy as np
import math
import heapq
import random
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict


# ==================== 辅助数据结构 ====================

@dataclass
class LayerInfo:
    """层信息"""
    name: str
    weight: np.ndarray
    bias: Optional[np.ndarray] = None
    grad: Optional[np.ndarray] = None
    layer_type: str = "linear"


@dataclass
class CompressionResult:
    """压缩结果"""
    method: str
    original_size: int
    compressed_size: int
    compression_ratio: float
    metrics: Dict[str, float] = field(default_factory=dict)


# ==================== 1. 剪枝 (Pruning) ====================

class MagnitudePruning:
    """幅度剪枝：按权重绝对值阈值剪枝"""

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold

    def compute_mask(self, weight: np.ndarray) -> np.ndarray:
        """计算二值掩码，低于阈值的权重置零"""
        return (np.abs(weight) >= self.threshold).astype(np.float32)

    def prune(self, weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """执行剪枝，返回剪枝后的权重和掩码"""
        mask = self.compute_mask(weight)
        return weight * mask, mask

    def sparsity_ratio(self, weight: np.ndarray) -> float:
        """计算稀疏率"""
        return float(np.sum(np.abs(weight) < self.threshold)) / weight.size


class StructuredPruning:
    """结构化剪枝：按通道/滤波器整体剪枝"""

    def __init__(self, pruning_ratio: float = 0.3, dim: int = 0):
        self.pruning_ratio = pruning_ratio
        self.dim = dim  # 0=输出通道, 1=输入通道

    def compute_channel_importance(self, weight: np.ndarray) -> np.ndarray:
        """计算每个通道的重要性分数（L1范数）"""
        if weight.ndim == 4:  # 卷积层 (out, in, h, w)
            axes = tuple(i for i in range(weight.ndim) if i != self.dim)
            return np.sum(np.abs(weight), axis=axes)
        elif weight.ndim == 2:  # 线性层 (out, in)
            axis = 1 if self.dim == 0 else 0
            return np.sum(np.abs(weight), axis=axis)
        return np.sum(np.abs(weight), axis=0)

    def prune(self, weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """执行结构化剪枝"""
        importance = self.compute_channel_importance(weight)
        n_channels = importance.shape[0]
        n_prune = max(1, int(n_channels * self.pruning_ratio))

        # 保留重要性最高的通道
        threshold_idx = np.argsort(importance)[n_prune]
        threshold_val = importance[threshold_idx]
        channel_mask = (importance > threshold_val).astype(np.float32)

        # 扩展掩码到权重形状
        if weight.ndim == 4:
            if self.dim == 0:
                mask = channel_mask[:, np.newaxis, np.newaxis, np.newaxis]
            else:
                mask = channel_mask[np.newaxis, :, np.newaxis, np.newaxis]
        elif weight.ndim == 2:
            if self.dim == 0:
                mask = channel_mask[:, np.newaxis]
            else:
                mask = channel_mask[np.newaxis, :]
        else:
            mask = np.ones_like(weight)

        return weight * mask, mask


class GradientPruning:
    """梯度剪枝：基于梯度幅度剪枝"""

    def __init__(self, pruning_ratio: float = 0.3):
        self.pruning_ratio = pruning_ratio
        self.gradient_accumulator: Dict[str, np.ndarray] = {}

    def accumulate_gradient(self, name: str, grad: np.ndarray):
        """累积梯度平方"""
        if name not in self.gradient_accumulator:
            self.gradient_accumulator[name] = np.zeros_like(grad)
        self.gradient_accumulator[name] += grad ** 2

    def compute_importance(self, weight: np.ndarray, name: str) -> np.ndarray:
        """权重*梯度幅度作为重要性"""
        if name in self.gradient_accumulator:
            return np.abs(weight) * np.sqrt(self.gradient_accumulator[name])
        return np.abs(weight)

    def prune(self, weight: np.ndarray, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """执行梯度剪枝"""
        importance = self.compute_importance(weight, name)
        flat = importance.flatten()
        threshold = np.quantile(flat, self.pruning_ratio)
        mask = (importance > threshold).astype(np.float32)
        return weight * mask, mask


class L1UnstructuredPruning:
    """L1非结构化剪枝：按L1范数全局排序剪枝"""

    def __init__(self, pruning_ratio: float = 0.5):
        self.pruning_ratio = pruning_ratio

    def prune(self, weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        flat = np.abs(weight).flatten()
        n_prune = int(flat.size * self.pruning_ratio)
        if n_prune == 0:
            return weight.copy(), np.ones_like(weight)
        threshold = np.sort(flat)[n_prune]
        mask = (np.abs(weight) >= threshold).astype(np.float32)
        return weight * mask, mask


class L2StructuredPruning:
    """L2结构化剪枝：按L2范数排序剪枝滤波器"""

    def __init__(self, pruning_ratio: float = 0.3):
        self.pruning_ratio = pruning_ratio

    def _filter_norms(self, weight: np.ndarray) -> np.ndarray:
        """计算每个输出滤波器的L2范数"""
        if weight.ndim == 4:
            return np.sqrt(np.sum(weight ** 2, axis=(1, 2, 3)))
        elif weight.ndim == 2:
            return np.sqrt(np.sum(weight ** 2, axis=1))
        return np.abs(weight.flatten())

    def prune(self, weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        norms = self._filter_norms(weight)
        n_filters = norms.shape[0]
        n_prune = max(1, int(n_filters * self.pruning_ratio))
        threshold = np.sort(norms)[n_prune]
        keep = norms >= threshold

        if weight.ndim == 4:
            mask = keep[:, np.newaxis, np.newaxis, np.newaxis].astype(np.float32)
        elif weight.ndim == 2:
            mask = keep[:, np.newaxis].astype(np.float32)
        else:
            mask = np.ones_like(weight)
        return weight * mask, mask


class GlobalPruning:
    """全局剪枝：跨所有层统一排序剪枝"""

    def __init__(self, pruning_ratio: float = 0.5):
        self.pruning_ratio = pruning_ratio

    def prune(self, weights: Dict[str, np.ndarray]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """跨层全局剪枝"""
        all_values = np.concatenate([np.abs(w).flatten() for w in weights.values()])
        n_prune = int(all_values.size * self.pruning_ratio)
        if n_prune == 0:
            return {k: (w.copy(), np.ones_like(w)) for k, w in weights.items()}
        threshold = np.sort(all_values)[n_prune]

        results = {}
        for name, weight in weights.items():
            mask = (np.abs(weight) >= threshold).astype(np.float32)
            results[name] = (weight * mask, mask)
        return results


class IterativePruning:
    """迭代剪枝：按渐进式调度逐步增加剪枝率"""

    def __init__(self, final_ratio: float = 0.8, num_steps: int = 10, pruning_fn=None):
        self.final_ratio = final_ratio
        self.num_steps = num_steps
        self.pruning_fn = pruning_fn or L1UnstructuredPruning()
        self.current_step = 0
        self.masks: Dict[str, np.ndarray] = {}

    def step_ratio(self) -> float:
        """当前步的剪枝率（余弦退火）"""
        progress = self.current_step / max(1, self.num_steps - 1)
        return self.final_ratio * 0.5 * (1 - math.cos(math.pi * progress))

    def prune_step(self, weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """执行一步迭代剪枝"""
        ratio = self.step_ratio()
        pruner = L1UnstructuredPruning(pruning_ratio=ratio)
        pruned = {}
        for name, weight in weights.items():
            new_weight, mask = pruner.prune(weight)
            if name in self.masks:
                mask = mask * self.masks[name]
            self.masks[name] = mask
            pruned[name] = new_weight * mask
        self.current_step += 1
        return pruned


class LotteryTicketHypothesis:
    """彩票假设：迭代幅度剪枝 + 权重回退"""

    def __init__(self, pruning_ratio_per_iter: float = 0.2, num_iterations: int = 5):
        self.pruning_ratio = pruning_ratio_per_iter
        self.num_iterations = num_iterations
        self.initial_weights: Dict[str, np.ndarray] = {}
        self.masks: Dict[str, np.ndarray] = {}

    def save_initial_weights(self, weights: Dict[str, np.ndarray]):
        """保存初始权重"""
        self.initial_weights = {k: v.copy() for k, v in weights.items()}
        self.masks = {k: np.ones_like(v) for k, v in weights.items()}

    def prune_iteration(self, weights: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float]:
        """一次迭代：剪枝 + 回退到初始权重"""
        # 计算全局重要性
        all_scores = []
        for name, w in weights.items():
            score = np.abs(w) * self.masks[name]
            all_scores.append(score.flatten())
        all_scores = np.concatenate(all_scores)

        n_total = all_scores.size
        n_prune = int(n_total * self.pruning_ratio)
        if n_prune == 0:
            return weights, 0.0
        threshold = np.sort(all_scores)[n_prune]

        # 更新掩码
        for name, w in weights.items():
            score = np.abs(w) * self.masks[name]
            self.masks[name] = (score >= threshold).astype(np.float32)

        # 回退到初始权重（乘以掩码）
        rewound = {}
        for name in weights:
            rewound[name] = self.initial_weights[name] * self.masks[name]

        sparsity = 1.0 - sum(m.sum() for m in self.masks.values()) / sum(
            m.size for m in self.masks.values()
        )
        return rewound, float(sparsity)


# ==================== 2. 量化 (Quantization) ====================

class UniformQuantization:
    """均匀量化：线性映射到N位整数"""

    def __init__(self, bits: int = 8, symmetric: bool = True):
        self.bits = bits
        self.symmetric = symmetric
        self.qmin = -(2 ** (bits - 1)) if symmetric else 0
        self.qmax = (2 ** (bits - 1) - 1) if symmetric else (2 ** bits - 1)

    def compute_scale_zp(self, tensor: np.ndarray) -> Tuple[float, float]:
        """计算缩放因子和零点"""
        if self.symmetric:
            max_val = max(np.abs(tensor.max()), np.abs(tensor.min()), 1e-8)
            scale = max_val / (2 ** (self.bits - 1) - 1)
            return scale, 0.0
        else:
            min_val, max_val = tensor.min(), tensor.max()
            scale = (max_val - min_val) / (2 ** self.bits - 1) if max_val != min_val else 1.0
            zp = -min_val / scale
            return scale, zp

    def quantize(self, tensor: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """量化张量，返回(量化后张量, scale, zero_point)"""
        scale, zp = self.compute_scale_zp(tensor)
        q_tensor = np.round(tensor / scale + zp)
        q_tensor = np.clip(q_tensor, self.qmin, self.qmax)
        dequant = (q_tensor - zp) * scale
        return dequant, scale, zp

    def quantize_to_int(self, tensor: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """量化到整数值"""
        scale, zp = self.compute_scale_zp(tensor)
        q_int = np.round(tensor / scale + zp).astype(np.int32)
        q_int = np.clip(q_int, self.qmin, self.qmax)
        return q_int, scale, zp


class NonUniformQuantization:
    """非均匀量化：基于k-means聚类"""

    def __init__(self, bits: int = 4, max_iter: int = 50, n_init: int = 3):
        self.bits = bits
        self.n_clusters = 2 ** bits
        self.max_iter = max_iter
        self.n_init = n_init

    def _kmeans(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """简单k-means聚类"""
        flat = data.flatten().astype(np.float64)
        best_centers, best_labels = None, None
        best_inertia = float('inf')

        for _ in range(self.n_init):
            indices = np.random.choice(len(flat), min(self.n_clusters, len(flat)), replace=False)
            centers = flat[indices].copy()
            labels = np.zeros(len(flat), dtype=np.int32)

            for _ in range(self.max_iter):
                # 分配
                dists = np.abs(flat[:, None] - centers[None, :])
                labels = np.argmin(dists, axis=1).astype(np.int32)
                # 更新
                new_centers = np.array([flat[labels == c].mean() if np.any(labels == c) else centers[c]
                                       for c in range(self.n_clusters)])
                if np.allclose(centers, new_centers, atol=1e-7):
                    break
                centers = new_centers

            inertia = sum(np.sum((flat[labels == c] - centers[c]) ** 2) for c in range(self.n_clusters))
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels

        return best_centers, best_labels

    def quantize(self, tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """非均匀量化，返回(量化后张量, 聚类中心)"""
        centers, labels = self._kmeans(tensor)
        quantized = centers[labels].reshape(tensor.shape).astype(np.float32)
        return quantized, centers


class QuantizationAwareTraining:
    """量化感知训练：在前向传播中模拟量化效果"""

    def __init__(self, bits: int = 8, symmetric: bool = True):
        self.quantizer = UniformQuantization(bits=bits, symmetric=symmetric)
        self.weight_scales: Dict[str, float] = {}
        self.weight_zps: Dict[str, float] = {}
        self.ema_decay = 0.99

    def simulate_quantize_forward(self, weight: np.ndarray, name: str) -> np.ndarray:
        """模拟量化（直通估计器）"""
        q_weight, scale, zp = self.quantizer.quantize(weight)
        # EMA更新scale/zp
        if name in self.weight_scales:
            self.weight_scales[name] = self.ema_decay * self.weight_scales[name] + (1 - self.ema_decay) * scale
            self.weight_zps[name] = self.ema_decay * self.weight_zps[name] + (1 - self.ema_decay) * zp
        else:
            self.weight_scales[name] = scale
            self.weight_zps[name] = zp
        return q_weight

    def straight_through_estimator(self, grad: np.ndarray, quantized: np.ndarray,
                                   original: np.ndarray) -> np.ndarray:
        """直通估计器：量化不可导，梯度直接传递"""
        # 梯度只在量化值和原始值方向一致时传递
        sign_match = (np.sign(quantized - original) == np.sign(grad)).astype(np.float32)
        return grad * sign_match


class PostTrainingQuantization:
    """训练后量化：收集激活统计后量化"""

    def __init__(self, bits: int = 8):
        self.bits = bits
        self.quantizer = UniformQuantization(bits=bits)
        self.activation_ranges: Dict[str, Tuple[float, float]] = {}

    def collect_activation_stats(self, activations: Dict[str, List[np.ndarray]]):
        """收集各层激活的min/max范围"""
        for name, act_list in activations.items():
            all_act = np.concatenate([a.flatten() for a in act_list])
            self.activation_ranges[name] = (float(all_act.min()), float(all_act.max()))

    def quantize_weights(self, weights: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """量化所有权重"""
        results = {}
        for name, w in weights.items():
            q_w, scale, zp = self.quantizer.quantize(w)
            results[name] = {"weight": q_w, "scale": scale, "zp": zp}
        return results

    def quantize_activations(self, activation: np.ndarray, name: str) -> np.ndarray:
        """量化激活值"""
        if name not in self.activation_ranges:
            return activation
        min_val, max_val = self.activation_ranges[name]
        scale = (max_val - min_val) / (2 ** self.bits - 1) if max_val != min_val else 1.0
        zp = -min_val / scale
        q_act = np.round(activation / scale + zp)
        q_act = np.clip(q_act, 0, 2 ** self.bits - 1)
        return (q_act - zp) * scale


class MixedPrecision:
    """混合精度量化：不同层使用不同位宽"""

    def __init__(self, bits_options: List[int] = None, budget_bits: float = 6.0):
        self.bits_options = bits_options or [4, 8]
        self.budget_bits = budget_bits
        self.layer_bits: Dict[str, int] = {}

    def compute_sensitivity(self, weights: Dict[str, np.ndarray],
                            eval_fn: Callable[[Dict[str, np.ndarray]], float]) -> Dict[str, float]:
        """计算各层量化敏感度"""
        baseline = eval_fn(weights)
        sensitivities = {}
        for name, w in weights.items():
            q = UniformQuantization(bits=4)
            q_w, _, _ = q.quantize(w)
            modified = {k: (q_w if k == name else v) for k, v in weights.items()}
            sensitivities[name] = baseline - eval_fn(modified)
        return sensitivities

    def assign_bits(self, sensitivities: Dict[str, float],
                    weight_sizes: Dict[str, int]) -> Dict[str, int]:
        """贪心分配：敏感层高精度，不敏感层低精度"""
        sorted_layers = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)
        total_params = sum(weight_sizes.values())

        # 先全部设为最低精度
        assignment = {name: min(self.bits_options) for name in sensitivities}
        current_avg = sum(min(self.bits_options) * weight_sizes[n] for n in sensitivities) / total_params

        # 贪心提升敏感层的精度
        for name, sens in sorted_layers:
            if current_avg >= self.budget_bits:
                break
            old_bits = assignment[name]
            higher = [b for b in self.bits_options if b > old_bits]
            if not higher:
                continue
            new_bits = min(higher)
            delta = (new_bits - old_bits) * weight_sizes[name] / total_params
            if current_avg + delta <= self.budget_bits:
                assignment[name] = new_bits
                current_avg += delta

        self.layer_bits = assignment
        return assignment


class GPTQ:
    """GPT Quantization：基于Hessian信息的LLM量化"""

    def __init__(self, bits: int = 4, blocksize: int = 128):
        self.bits = bits
        self.blocksize = blocksize
        self.quantizer = UniformQuantization(bits=bits, symmetric=True)

    def quantize_layer(self, weight: np.ndarray, hessian_inv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """对单层执行GPTQ量化"""
        rows, cols = weight.shape
        q_weight = np.zeros_like(weight)
        errors = np.zeros(rows)

        for i1 in range(0, cols, self.blocksize):
            i2 = min(i1 + self.blocksize, cols)
            block = weight[:, i1:i2].copy()

            # 量化当前块
            q_block, scale, zp = self.quantizer.quantize(block)
            q_weight[:, i1:i2] = q_block

            # 计算量化误差
            block_error = block - q_block

            # 使用Hessian逆矩阵传播误差
            for j in range(i2, cols):
                correction = hessian_inv[i1:i2, j] @ block_error.sum(axis=0)
                weight[:, j] += correction / max(hessian_inv[j, j], 1e-8)

            errors += block_error.sum(axis=1)

        return q_weight, errors

    def approximate_hessian_inverse(self, input_covariance: np.ndarray) -> np.ndarray:
        """近似Hessian逆矩阵 (H^{-1} ≈ diag(1/diag(H)))"""
        diag = np.diag(input_covariance).copy()
        diag = np.maximum(diag, 1e-8)
        return np.diag(1.0 / diag)


class AWQ:
    """Activation-aware Weight Quantization：基于激活感知的权重量化"""

    def __init__(self, bits: int = 4, alpha: float = 0.5, n_samples: int = 128):
        self.bits = bits
        self.alpha = alpha  # 缩放混合系数
        self.n_samples = n_samples

    def compute_activation_scale(self, weight: np.ndarray,
                                  activations: np.ndarray) -> np.ndarray:
        """计算每行权重的激活感知缩放因子"""
        # weight: (out, in), activations: (n_samples, in)
        # 每个输出通道的激活幅度
        act_magnitude = np.mean(np.abs(activations), axis=0)  # (in,)
        weight_magnitude = np.mean(np.abs(weight), axis=1)  # (out,)

        # 缩放因子：激活大的通道权重更重要
        scale = np.outer(weight_magnitude, act_magnitude)  # (out, in)
        scale = scale / (np.mean(scale) + 1e-8)
        return scale

    def quantize(self, weight: np.ndarray, activations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """执行AWQ量化"""
        scale = self.compute_activation_scale(weight, activations)

        # 应用alpha缩放保护重要权重
        s = scale ** self.alpha
        scaled_weight = weight * s

        # 量化缩放后的权重
        q = UniformQuantization(bits=self.bits, symmetric=True)
        q_scaled, _, _ = q.quantize(scaled_weight)

        # 反缩放
        q_weight = q_scaled / (s + 1e-8)
        return q_weight, s


# ==================== 3. 知识蒸馏 (Knowledge Distillation) ====================

class ResponseBasedDistillation:
    """基于响应的蒸馏：软目标匹配"""

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        self.temperature = temperature
        self.alpha = alpha  # 软标签损失权重

    def softmax(self, logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """温度缩放softmax"""
        scaled = logits / temperature
        shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

    def compute_loss(self, teacher_logits: np.ndarray, student_logits: np.ndarray,
                     labels: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """计算蒸馏损失"""
        T = self.temperature
        t_soft = self.softmax(teacher_logits, T)
        s_soft = self.softmax(student_logits, T)

        # KL散度 (软标签)
        kl_loss = np.sum(t_soft * (np.log(t_soft + 1e-10) - np.log(s_soft + 1e-10))) / teacher_logits.shape[0]
        kl_loss *= T * T

        # 交叉熵 (硬标签)
        s_probs = self.softmax(student_logits, 1.0)
        n = labels.shape[0]
        ce_loss = -np.mean(np.log(s_probs[np.arange(n), labels] + 1e-10))

        total = self.alpha * kl_loss + (1 - self.alpha) * ce_loss
        return float(total), {"kl_loss": float(kl_loss), "ce_loss": float(ce_loss)}


class FeatureBasedDistillation:
    """基于特征的蒸馏：中间层特征匹配"""

    def __init__(self, loss_type: str = "mse"):
        self.loss_type = loss_type

    def align_features(self, teacher_feat: np.ndarray, student_feat: np.ndarray) -> np.ndarray:
        """对齐特征维度（1x1卷积模拟投影）"""
        if teacher_feat.shape == student_feat.shape:
            return student_feat
        # 简单线性投影：最小二乘
        t_flat = teacher_feat.reshape(teacher_feat.shape[0], -1)
        s_flat = student_feat.reshape(student_feat.shape[0], -1)
        # 投影矩阵: W = (S^T S)^{-1} S^T T
        sts = s_flat.T @ s_flat
        sts += np.eye(sts.shape[0]) * 1e-6
        W = np.linalg.solve(sts, s_flat.T @ t_flat)
        projected = (s_flat @ W).reshape(teacher_feat.shape)
        return projected

    def compute_loss(self, teacher_features: List[np.ndarray],
                     student_features: List[np.ndarray]) -> Tuple[float, Dict[str, float]]:
        """计算特征蒸馏损失"""
        total_loss = 0.0
        layer_losses = {}

        for i, (t_feat, s_feat) in enumerate(zip(teacher_features, student_features)):
            aligned = self.align_features(t_feat, s_feat)
            if self.loss_type == "mse":
                loss = np.mean((aligned - t_feat) ** 2)
            elif self.loss_type == "smooth_l1":
                diff = np.abs(aligned - t_feat)
                loss = np.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5).mean()
            else:
                loss = np.mean((aligned - t_feat) ** 2)
            total_loss += loss
            layer_losses[f"layer_{i}"] = float(loss)

        total_loss /= max(len(teacher_features), 1)
        return float(total_loss), layer_losses


class RelationBasedDistillation:
    """基于关系的蒸馏：样本间关系匹配"""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def _pairwise_distance(self, features: np.ndarray) -> np.ndarray:
        """计算样本间距离矩阵"""
        # features: (batch, dim)
        diff = features[:, np.newaxis, :] - features[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-8)

    def compute_loss(self, teacher_features: np.ndarray,
                     student_features: np.ndarray) -> float:
        """计算关系蒸馏损失"""
        t_dist = self._pairwise_distance(teacher_features) / self.temperature
        s_dist = self._pairwise_distance(student_features) / self.temperature

        # 归一化
        t_dist = t_dist / (np.sum(t_dist) + 1e-8)
        s_dist = s_dist / (np.sum(s_dist) + 1e-8)

        # KL散度
        loss = np.sum(t_dist * (np.log(t_dist + 1e-10) - np.log(s_dist + 1e-10)))
        return float(loss)


class SelfDistillation:
    """自蒸馏：教师=学生（更深/更宽的自身变体）"""

    def __init__(self, temperature: float = 3.0):
        self.temperature = temperature
        self.distiller = ResponseBasedDistillation(temperature=temperature, alpha=0.8)

    def create_deep_stopper(self, logits: np.ndarray, stopper_depth: int = 2) -> np.ndarray:
        """模拟更深层网络的输出（通过多次非线性变换）"""
        result = logits.copy()
        for _ in range(stopper_depth):
            result = np.tanh(result @ (np.random.randn(logits.shape[-1], logits.shape[-1]) * 0.1))
        return result

    def compute_loss(self, student_logits: np.ndarray,
                     labels: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """自蒸馏损失"""
        teacher_logits = self.create_deep_stopper(student_logits)
        return self.distiller.compute_loss(teacher_logits, student_logits, labels)


class ProgressiveDistillation:
    """渐进式蒸馏：多阶段逐步蒸馏"""

    def __init__(self, num_stages: int = 3, temperature_schedule: List[float] = None):
        self.num_stages = num_stages
        self.temperature_schedule = temperature_schedule or [7.0, 5.0, 3.0]
        self.current_stage = 0

    def get_temperature(self) -> float:
        return self.temperature_schedule[min(self.current_stage, len(self.temperature_schedule) - 1)]

    def compute_loss(self, teacher_logits: np.ndarray, student_logits: np.ndarray,
                     labels: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """当前阶段的蒸馏损失"""
        distiller = ResponseBasedDistillation(temperature=self.get_temperature(), alpha=0.7)
        loss, metrics = distiller.compute_loss(teacher_logits, student_logits, labels)
        metrics["stage"] = float(self.current_stage)
        return loss, metrics

    def advance_stage(self):
        self.current_stage += 1


class MultiTeacherDistillation:
    """多教师蒸馏：集成多个教师的软标签"""

    def __init__(self, teacher_weights: List[float] = None, temperature: float = 4.0):
        self.teacher_weights = teacher_weights
        self.temperature = temperature
        self.distiller = ResponseBasedDistillation(temperature=temperature, alpha=0.7)

    def aggregate_teacher_logits(self, teacher_logits_list: List[np.ndarray]) -> np.ndarray:
        """聚合多个教师的输出"""
        if self.teacher_weights is None:
            w = [1.0 / len(teacher_logits_list)] * len(teacher_logits_list)
        else:
            w = self.teacher_weights
        aggregated = sum(w[i] * logits for i, logits in enumerate(teacher_logits_list))
        return aggregated

    def compute_loss(self, teacher_logits_list: List[np.ndarray],
                     student_logits: np.ndarray,
                     labels: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """计算多教师蒸馏损失"""
        agg_logits = self.aggregate_teacher_logits(teacher_logits_list)
        return self.distiller.compute_loss(agg_logits, student_logits, labels)


# ==================== 4. 低秩近似 (Low Rank Approximation) ====================

class SVDDecomposition:
    """SVD分解用于权重压缩"""

    def __init__(self, rank_ratio: float = 0.5, energy_threshold: float = 0.99):
        self.rank_ratio = rank_ratio
        self.energy_threshold = energy_threshold

    def decompose(self, weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """SVD分解，返回(U, S, Vt)"""
        U, S, Vt = np.linalg.svd(weight, full_matrices=False)
        return U, S, Vt

    def truncated_decompose(self, weight: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray]:
        """截断SVD，返回两个低秩矩阵 A, B 使得 A@B ≈ weight"""
        U, S, Vt = self.decompose(weight)
        rank = min(rank, len(S))
        A = U[:, :rank] * np.sqrt(S[:rank])  # (m, r)
        B = np.sqrt(S[:rank])[:, None] * Vt[:rank, :]  # (r, n)
        return A, B

    def auto_rank(self, weight: np.ndarray) -> int:
        """根据能量阈值自动选择秩"""
        _, S, _ = self.decompose(weight)
        energy = np.cumsum(S ** 2) / np.sum(S ** 2)
        rank = int(np.searchsorted(energy, self.energy_threshold)) + 1
        return min(rank, len(S))

    def compress(self, weight: np.ndarray) -> Dict[str, Any]:
        """压缩权重"""
        rank = max(1, int(min(weight.shape) * self.rank_ratio))
        A, B = self.truncated_decompose(weight, rank)
        original_size = weight.size
        compressed_size = A.size + B.size
        return {
            "A": A, "B": B, "rank": rank,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "ratio": compressed_size / original_size,
            "reconstructed": A @ B,
        }


class TuckerDecomposition:
    """Tucker分解：多维张量分解"""

    def __init__(self, rank_ratio: float = 0.5):
        self.rank_ratio = rank_ratio

    def decompose(self, tensor: np.ndarray, ranks: List[int]) -> Dict[str, Any]:
        """Tucker分解 (Higher-Order SVD)"""
        ndim = tensor.ndim
        factors = []

        # 第一步：对每个模式计算SVD因子
        for mode in range(ndim):
            unfolding = np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)
            U, S, Vt = np.linalg.svd(unfolding, full_matrices=False)
            r = min(ranks[mode], len(S))
            factors.append(U[:, :r])

        # 第二步：计算核心张量 tensor x1 U1^T x2 U2^T ... xn Un^T
        # 使用n-mode product: 每次将mode维移到前面，做矩阵乘法，再移回
        core = tensor.copy()
        for mode in range(ndim):
            # 将第mode维移到最前面
            core = np.moveaxis(core, mode, 0)
            # core shape: (d_mode, d1, d2, ..., d_{mode-1}, d_{mode+1}, ...)
            # factors[mode].T shape: (r_mode, d_mode)
            core = np.tensordot(factors[mode].T, core, axes=([1], [0]))
            # core shape: (r_mode, d1, d2, ...)
            # 将第0维移回第mode位
            core = np.moveaxis(core, 0, mode)

        # 重建: core x1 U1 x2 U2 ... xn Un
        reconstructed = core.copy()
        for mode in range(ndim):
            reconstructed = np.moveaxis(reconstructed, mode, 0)
            reconstructed = np.tensordot(factors[mode], reconstructed, axes=([1], [0]))
            reconstructed = np.moveaxis(reconstructed, 0, mode)

        original_size = tensor.size
        core_size = core.size
        factors_size = sum(f.size for f in factors)
        return {
            "core": core, "factors": factors,
            "original_size": original_size,
            "compressed_size": core_size + factors_size,
            "ratio": (core_size + factors_size) / original_size,
            "reconstructed": reconstructed,
        }


class TensorTrainDecomposition:
    """Tensor Train (TT) 分解"""

    def __init__(self, tt_ranks: List[int] = None):
        self.tt_ranks = tt_ranks

    def decompose(self, tensor: np.ndarray) -> Dict[str, Any]:
        """TT分解"""
        ndim = tensor.ndim
        shape = tensor.shape

        if self.tt_ranks is None:
            self.tt_ranks = [1] + [min(s, 16) for s in shape] + [1]

        cores = []
        current = tensor.copy()

        for k in range(ndim - 1):
            # current shape: (r_k, shape[k], shape[k+1], ..., shape[-1])
            r_in = current.shape[0]
            d = current.shape[1] if current.ndim > 1 else current.shape[0]
            tail_shape = current.shape[2:] if current.ndim > 2 else ()
            tail_size = int(np.prod(tail_shape)) if tail_shape else 1

            # TT分解: 将前两维合并, SVD分离第k维
            matrix = current.reshape(r_in * d, tail_size)
            U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

            r_out = min(self.tt_ranks[k + 1], len(S))
            # U: (r_in*d, r_out), 重塑为 (r_in, d, r_out)
            core_k = U[:, :r_out].reshape(r_in, d, r_out)
            cores.append(core_k)

            # 更新剩余: S[:r_out] @ Vt[:r_out, :] -> (r_out, tail_size)
            current = np.diag(S[:r_out]) @ Vt[:r_out, :]
            if tail_shape:
                current = current.reshape(r_out, *tail_shape)
            else:
                current = current.reshape(r_out, 1)

        # 最后一个核心: (r_last, shape[-1], 1)
        if current.ndim == 1:
            cores.append(current.reshape(-1, 1, 1))
        else:
            cores.append(current.reshape(current.shape[0], current.shape[1], 1))

        original_size = tensor.size
        compressed_size = sum(c.size for c in cores)
        return {
            "cores": cores, "tt_ranks": self.tt_ranks,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "ratio": compressed_size / original_size,
        }

    def reconstruct(self, cores: List[np.ndarray]) -> np.ndarray:
        """从TT核重建张量"""
        result = cores[0]
        for core in cores[1:]:
            result = np.tensordot(result, core, axes=(-1, 0))
        return result


class CPDecomposition:
    """CP分解：张量的CANDECOMP/PARAFAC分解"""

    def __init__(self, rank: int = 10, max_iter: int = 100, tol: float = 1e-6):
        self.rank = rank
        self.max_iter = max_iter
        self.tol = tol

    def decompose(self, tensor: np.ndarray) -> Dict[str, Any]:
        """ALS交替最小二乘CP分解"""
        ndim = tensor.ndim
        shape = tensor.shape
        rank = min(self.rank, min(shape))

        # 随机初始化因子矩阵
        factors = [np.random.randn(s, rank).astype(np.float32) for s in shape]
        weights = np.ones(rank)

        norm_tensor = np.linalg.norm(tensor)
        prev_error = float('inf')

        for iteration in range(self.max_iter):
            for mode in range(ndim):
                # 构建Khatri-Rao积 (逐列Kronecker积)
                other_modes = [m for m in range(ndim) if m != mode]
                kr = factors[other_modes[-1]]
                for m in reversed(other_modes[:-1]):
                    # Khatri-Rao: 对每列做Kronecker积
                    kr_new = np.zeros((factors[m].shape[0] * kr.shape[0], rank), dtype=np.float32)
                    for r in range(rank):
                        kr_new[:, r] = np.kron(factors[m][:, r], kr[:, r])
                    kr = kr_new

                # 展开张量
                unfolding = np.moveaxis(tensor, mode, 0).reshape(shape[mode], -1)

                # 最小二乘更新
                gram = kr.T @ kr + np.eye(rank) * 1e-8
                factors[mode] = np.linalg.solve(gram, kr.T @ unfolding.T).T

            # 计算误差
            reconstructed = self._reconstruct(factors, weights, shape)
            error = np.linalg.norm(tensor - reconstructed) / max(norm_tensor, 1e-8)

            if abs(prev_error - error) < self.tol:
                break
            prev_error = error

        original_size = tensor.size
        compressed_size = sum(f.size for f in factors) + rank
        return {
            "factors": factors, "weights": weights,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "ratio": compressed_size / original_size,
            "reconstructed": reconstructed,
        }

    def _reconstruct(self, factors: List[np.ndarray], weights: np.ndarray,
                     shape: tuple) -> np.ndarray:
        """从CP因子重建张量"""
        ndim = len(factors)
        rank = len(weights)
        # 使用einsum进行高效重建
        # 构建einsum表达式: sum over rank dimension
        # weighted_factors[i] = weights * factors[i] for each mode
        result = np.zeros(shape, dtype=np.float32)
        for r in range(rank):
            outer = factors[0][:, r]
            for i in range(1, ndim):
                outer = np.multiply.outer(outer, factors[i][:, r])
            result += weights[r] * outer
        return result


# ==================== 5. 神经架构搜索 (Lightweight NAS) ====================

class WeightSharingNAS:
    """权重共享NAS：超网络生成子网络权重"""

    def __init__(self, input_dim: int, max_layers: int = 6,
                 hidden_choices: List[int] = None):
        self.input_dim = input_dim
        self.max_layers = max_layers
        self.hidden_choices = hidden_choices or [32, 64, 128, 256]
        self.embeddings: Dict[str, np.ndarray] = {}
        self._init_embeddings()

    def _init_embeddings(self):
        """初始化层权重嵌入"""
        for i in range(self.max_layers):
            for h in self.hidden_choices:
                key = f"layer_{i}_hidden_{h}"
                self.embeddings[key] = np.random.randn(h, self.input_dim).astype(np.float32) * 0.02

    def sample_architecture(self) -> List[int]:
        """随机采样一个架构（每层隐藏维度列表）"""
        n_layers = random.randint(1, self.max_layers)
        return [random.choice(self.hidden_choices) for _ in range(n_layers)]

    def get_weights(self, architecture: List[int]) -> List[np.ndarray]:
        """根据架构从共享权重中获取权重"""
        weights = []
        for i, h in enumerate(architecture):
            key = f"layer_{i}_hidden_{h}"
            weights.append(self.embeddings.get(key, np.random.randn(h, self.input_dim) * 0.02))
        return weights

    def score_architecture(self, architecture: List[int], eval_fn: Callable) -> float:
        """评估架构得分"""
        weights = self.get_weights(architecture)
        return eval_fn(architecture, weights)


class OnceForAllNetwork:
    """Once-for-All网络：训练一次支持多种子网络配置"""

    def __init__(self, max_depth: int = 4, width_multipliers: List[float] = None,
                 kernel_sizes: List[int] = None):
        self.max_depth = max_depth
        self.width_multipliers = width_multipliers or [0.25, 0.5, 0.75, 1.0]
        self.kernel_sizes = kernel_sizes or [3, 5, 7]
        self.elastic_weights: Dict[str, np.ndarray] = {}

    def generate_subnet_config(self) -> Dict[str, Any]:
        """生成子网络配置"""
        depth = random.randint(1, self.max_depth)
        return {
            "depth": depth,
            "width_mult": random.choice(self.width_multipliers),
            "kernel_sizes": [random.choice(self.kernel_sizes) for _ in range(depth)],
        }

    def set_elastic_width(self, layer_name: str, full_weight: np.ndarray,
                          multiplier: float) -> np.ndarray:
        """弹性宽度：按比例截取通道"""
        n_out = max(1, int(full_weight.shape[0] * multiplier))
        n_in = max(1, int(full_weight.shape[1] * multiplier)) if full_weight.ndim >= 2 else 1
        if full_weight.ndim == 2:
            sub_weight = full_weight[:n_out, :n_in]
        elif full_weight.ndim == 4:
            sub_weight = full_weight[:n_out, :n_in, :, :]
        else:
            sub_weight = full_weight
        self.elastic_weights[layer_name] = sub_weight
        return sub_weight

    def count_subnet_params(self, config: Dict[str, Any],
                            base_channels: int = 64) -> int:
        """计算子网络参数量"""
        depth = config["depth"]
        mult = config["width_mult"]
        total = 0
        ch = int(base_channels * mult)
        for _ in range(depth):
            total += ch * ch * 9  # 假设3x3卷积
            ch = max(ch, int(ch * 1.0))
        return total


class HardwareAwareNAS:
    """硬件感知NAS：考虑延迟/能耗的架构搜索"""

    def __init__(self, latency_constraints: Dict[str, float] = None,
                 hardware_profile: Dict[str, float] = None):
        self.latency_constraints = latency_constraints or {"target_ms": 10.0}
        self.hw_profile = hardware_profile or {
            "conv_ops_per_ms": 1e6, "linear_ops_per_ms": 2e6,
            "memory_bandwidth_gb_s": 10.0,
        }

    def estimate_latency(self, architecture: List[Dict[str, Any]]) -> float:
        """估算推理延迟（毫秒）"""
        total_ops = 0
        total_memory = 0
        for layer in architecture:
            if layer["type"] == "conv":
                ops = layer["out_ch"] * layer["in_ch"] * layer["kernel"] ** 2 * layer["h"] * layer["w"]
                total_ops += ops
            elif layer["type"] == "linear":
                ops = layer["in_features"] * layer["out_features"]
                total_ops += ops
            total_memory += layer.get("params", 0) * 4  # float32

        compute_time = total_ops / self.hw_profile["conv_ops_per_ms"]
        memory_time = total_memory / (self.hw_profile["memory_bandwidth_gb_s"] * 1e6)
        return compute_time + memory_time

    def search(self, candidate_archs: List[List[Dict[str, Any]]],
               eval_fn: Callable) -> Tuple[List[Dict[str, Any]], float]:
        """在延迟约束下搜索最优架构"""
        best_arch = None
        best_score = -float('inf')
        target_ms = self.latency_constraints["target_ms"]

        for arch in candidate_archs:
            latency = self.estimate_latency(arch)
            if latency > target_ms:
                continue
            score = eval_fn(arch)
            if score > best_score:
                best_score = score
                best_arch = arch

        return best_arch or candidate_archs[0], best_score


# ==================== 6. 稀疏训练 (Sparsity) ====================

class SparseMatrixOps:
    """稀疏矩阵操作（CSR/CSC格式）"""

    @staticmethod
    def dense_to_csr(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """稠密矩阵转CSR格式"""
        rows, cols = matrix.shape
        data = []
        indices = []
        indptr = [0]

        for i in range(rows):
            for j in range(cols):
                if matrix[i, j] != 0:
                    data.append(matrix[i, j])
                    indices.append(j)
            indptr.append(len(data))

        return np.array(data, dtype=np.float32), np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int32)

    @staticmethod
    def csr_to_dense(data: np.ndarray, indices: np.ndarray,
                     indptr: np.ndarray, shape: tuple) -> np.ndarray:
        """CSR转稠密矩阵"""
        matrix = np.zeros(shape, dtype=np.float32)
        for i in range(len(indptr) - 1):
            for j in range(indptr[i], indptr[i + 1]):
                matrix[i, indices[j]] = data[j]
        return matrix

    @staticmethod
    def csr_matvec(data: np.ndarray, indices: np.ndarray,
                   indptr: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """CSR矩阵-向量乘法"""
        result = np.zeros(len(indptr) - 1, dtype=np.float32)
        for i in range(len(indptr) - 1):
            for j in range(indptr[i], indptr[i + 1]):
                result[i] += data[j] * vector[indices[j]]
        return result

    @staticmethod
    def csr_matmul(data_a: np.ndarray, indices_a: np.ndarray, indptr_a: np.ndarray,
                   data_b: np.ndarray, indices_b: np.ndarray, indptr_b: np.ndarray,
                   shape_a: tuple, shape_b: tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """CSR矩阵乘法"""
        m, k = shape_a
        k2, n = shape_b
        assert k == k2

        result = np.zeros((m, n), dtype=np.float32)
        for i in range(m):
            for jj in range(indptr_a[i], indptr_a[i + 1]):
                j = indices_a[jj]
                for ll in range(indptr_b[j], indptr_b[j + 1]):
                    result[i, indices_b[ll]] += data_a[jj] * data_b[ll]

        return SparseMatrixOps.dense_to_csr(result)

    @staticmethod
    def sparsity(matrix: np.ndarray) -> float:
        """计算稀疏度"""
        return 1.0 - np.count_nonzero(matrix) / matrix.size


class DynamicSparseTraining:
    """动态稀疏训练：稀疏度保持不变但连接动态变化"""

    def __init__(self, sparsity: float = 0.9, update_frequency: int = 100):
        self.sparsity = sparsity
        self.update_frequency = update_frequency
        self.step_count = 0

    def prune_and_regrow(self, weight: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """剪枝最小权重，在最大梯度位置重新生长"""
        mask = (weight != 0).astype(np.float32)
        n_total = weight.size
        n_active = int(n_total * (1 - self.sparsity))

        # 找到当前活跃连接
        active_values = np.abs(weight) * mask
        flat_active = active_values.flatten()

        # 剪枝：移除活跃连接中最小的
        if np.sum(mask) > n_active:
            threshold = np.sort(flat_active[flat_active > 0])[-n_active]
            prune_mask = (active_values >= threshold).astype(np.float32)
        else:
            prune_mask = mask.copy()

        # 生长：在非活跃位置中选梯度最大的
        inactive_grad = np.abs(gradient) * (1 - prune_mask)
        n_grow = max(0, n_active - int(np.sum(prune_mask)))
        if n_grow > 0:
            flat_inactive_grad = inactive_grad.flatten()
            flat_inactive_grad[prune_mask.flatten() > 0] = -1
            top_indices = np.argpartition(flat_inactive_grad, -n_grow)[-n_grow:]
            grow_mask = np.zeros_like(weight)
            grow_mask.flatten()[top_indices] = 1.0
            prune_mask = np.maximum(prune_mask, grow_mask)

        self.step_count += 1
        return weight * prune_mask

    def should_update(self) -> bool:
        return self.step_count % self.update_frequency == 0


class RigL:
    """RigL：基于梯度增长的稀疏训练"""

    def __init__(self, sparsity: float = 0.9, delta_T: int = 100,
                 alpha: float = 0.3, T_end: int = 1000):
        self.sparsity = sparsity
        self.delta_T = delta_T  # 更新间隔
        self.alpha = alpha  # 每次更新的比例
        self.T_end = T_end  # 稀疏度warmup结束步数
        self.step = 0

    def current_sparsity(self) -> float:
        """当前步的稀疏度（线性warmup）"""
        if self.step >= self.T_end:
            return self.sparsity
        return self.sparsity * (self.step / self.T_end)

    def update_mask(self, weight: np.ndarray, gradient: np.ndarray,
                    mask: np.ndarray) -> np.ndarray:
        """RigL掩码更新"""
        n_weights = weight.size
        n_active = int(n_weights * (1 - self.current_sparsity()))
        n_remove = int(n_active * self.alpha)

        # 移除：活跃权重中幅度最小的
        active_scores = np.abs(weight) * mask
        flat = active_scores.flatten()
        active_indices = np.where(flat > 0)[0]
        if len(active_indices) <= n_remove:
            return mask
        remove_threshold = np.sort(flat[active_indices])[n_remove]
        new_mask = (active_scores > remove_threshold).astype(np.float32)

        # 生长：非活跃位置中梯度最大的
        n_grow = n_remove
        inactive_grad = np.abs(gradient) * (1 - new_mask)
        flat_ig = inactive_grad.flatten()
        top_indices = np.argpartition(flat_ig, -n_grow)[-n_grow:]
        grow_mask = np.zeros_like(weight)
        grow_mask.flatten()[top_indices] = 1.0
        new_mask = np.maximum(new_mask, grow_mask)

        self.step += 1
        return new_mask


# ==================== 7. 编译优化 (Compilation Optimization) ====================

class OperatorFusion:
    """算子融合模式检测与优化"""

    def __init__(self):
        self.fusion_patterns = {
            "conv_bn_relu": ["conv", "batch_norm", "relu"],
            "linear_relu": ["linear", "relu"],
            "conv_bias_relu": ["conv", "bias_add", "relu"],
            "matmul_add": ["matmul", "bias_add"],
            "conv_depthwise_pointwise": ["conv_depthwise", "conv_pointwise"],
        }

    def detect_fusion_opportunities(self, graph: List[Dict[str, str]]) -> List[List[int]]:
        """检测可融合的算子序列"""
        fused_groups = []
        current_group = []

        for i, node in enumerate(graph):
            op_type = node.get("op", "")
            if not current_group:
                current_group = [i]
                continue

            # 检查是否匹配某个融合模式
            found = False
            for pattern_name, pattern in self.fusion_patterns.items():
                group_ops = [graph[j].get("op", "") for j in current_group]
                if len(group_ops) < len(pattern):
                    expected_next = pattern[len(group_ops)]
                    if op_type == expected_next:
                        current_group.append(i)
                        found = True
                        break

            if not found:
                if len(current_group) > 1:
                    fused_groups.append(current_group)
                current_group = [i]

        if len(current_group) > 1:
            fused_groups.append(current_group)

        return fused_groups

    def estimate_fusion_benefit(self, graph: List[Dict[str, str]],
                                fused_groups: List[List[int]]) -> Dict[str, float]:
        """估算融合收益（减少的内存访问次数）"""
        total_savings = 0
        details = {}
        for group in fused_groups:
            ops = [graph[i] for i in group]
            # 每次融合减少中间结果的读写
            intermediate_size = sum(o.get("output_size", 0) for o in ops[:-1])
            savings = intermediate_size * 2  # 读+写
            total_savings += savings
            details[f"fuse_{group[0]}_{group[-1]}"] = float(savings)
        details["total_savings_bytes"] = float(total_savings)
        return details


class MemoryLayoutOptimization:
    """内存布局优化"""

    def __init__(self):
        self.layout_types = ["nchw", "nhwc", "chw", "hw"]

    def compute_access_pattern_score(self, tensor_shape: tuple, access_pattern: str,
                                      layout: str) -> float:
        """计算内存访问模式得分（连续性越好分数越高）"""
        if layout == "nchw" and access_pattern == "channel_first":
            return 1.0
        elif layout == "nhwc" and access_pattern == "spatial_first":
            return 1.0
        elif layout == "nchw" and access_pattern == "spatial_first":
            return 0.7
        elif layout == "nhwc" and access_pattern == "channel_first":
            return 0.7
        return 0.5

    def find_optimal_layout(self, layer_configs: List[Dict[str, Any]]) -> Dict[str, str]:
        """为每层找到最优内存布局"""
        optimal = {}
        for config in layer_configs:
            name = config["name"]
            access = config.get("access_pattern", "spatial_first")
            best_layout = "nhwc"
            best_score = 0
            for layout in self.layout_types:
                score = self.compute_access_pattern_score(
                    config.get("shape", (1, 64, 28, 28)), access, layout
                )
                if score > best_score:
                    best_score = score
                    best_layout = layout
            optimal[name] = best_layout
        return optimal

    def estimate_padding_waste(self, tensor_shape: tuple, alignment: int = 32) -> float:
        """估算内存对齐导致的填充浪费"""
        total = 1
        for s in tensor_shape:
            total *= s
        aligned = 1
        for s in tensor_shape:
            aligned *= ((s + alignment - 1) // alignment) * alignment
        return (aligned - total) / total


class KernelAutoTuning:
    """内核自动调优模拟"""

    def __init__(self, param_grid: Dict[str, List[Any]] = None):
        self.param_grid = param_grid or {
            "tile_size": [8, 16, 32, 64],
            "vector_size": [4, 8, 16],
            "unroll_factor": [1, 2, 4, 8],
            "use_shared_memory": [True, False],
        }
        self.best_config = None
        self.best_time = float('inf')

    def generate_configs(self) -> List[Dict[str, Any]]:
        """生成所有参数组合"""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        configs = []
        for combo in _product(*values):
            configs.append(dict(zip(keys, combo)))
        return configs

    def simulate_kernel_execution(self, config: Dict[str, Any],
                                   problem_size: tuple) -> float:
        """模拟内核执行时间（启发式模型）"""
        m, n, k = problem_size
        tile = config["tile_size"]
        vec = config["vector_size"]
        unroll = config["unroll_factor"]

        # 启发式延迟模型
        compute_ops = 2 * m * n * k
        memory_accesses = (m * k + k * n + m * n) * 4  # bytes

        tile_efficiency = min(tile / 32.0, 1.0)
        vec_efficiency = min(vec / 16.0, 1.0)
        unroll_efficiency = min(unroll / 8.0, 1.0)
        sm_bonus = 1.3 if config["use_shared_memory"] else 1.0

        compute_time = compute_ops / (1e9 * tile_efficiency * vec_efficiency * unroll_efficiency * sm_bonus)
        memory_time = memory_accesses / (50e9 * sm_bonus)  # 50 GB/s bandwidth

        return compute_time + memory_time

    def tune(self, problem_size: tuple, n_trials: int = 50) -> Dict[str, Any]:
        """执行自动调优"""
        configs = self.generate_configs()
        # 随机采样n_trials个配置
        sampled = random.sample(configs, min(n_trials, len(configs)))

        for config in sampled:
            exec_time = self.simulate_kernel_execution(config, problem_size)
            if exec_time < self.best_time:
                self.best_time = exec_time
                self.best_config = config

        return self.best_config


def _product(*iterables):
    """笛卡尔积辅助函数"""
    if not iterables:
        yield ()
    else:
        for item in iterables[0]:
            for rest in _product(*iterables[1:]):
                yield (item,) + rest


# ==================== 8. 压缩流水线 (Compression Pipeline) ====================

class CompressionPipeline:
    """端到端压缩流水线"""

    def __init__(self, target_compression_ratio: float = 0.25,
                 accuracy_threshold: float = 0.95):
        self.target_ratio = target_compression_ratio
        self.accuracy_threshold = accuracy_threshold
        self.steps: List[Dict[str, Any]] = []
        self.results: List[CompressionResult] = []

    def add_pruning(self, method: str = "magnitude", ratio: float = 0.3,
                    structured: bool = False) -> "CompressionPipeline":
        """添加剪枝步骤"""
        self.steps.append({
            "type": "pruning", "method": method, "ratio": ratio,
            "structured": structured,
        })
        return self

    def add_quantization(self, method: str = "uniform", bits: int = 8) -> "CompressionPipeline":
        """添加量化步骤"""
        self.steps.append({
            "type": "quantization", "method": method, "bits": bits,
        })
        return self

    def add_distillation(self, method: str = "response", temperature: float = 4.0) -> "CompressionPipeline":
        """添加蒸馏步骤"""
        self.steps.append({
            "type": "distillation", "method": method, "temperature": temperature,
        })
        return self

    def add_low_rank(self, method: str = "svd", rank_ratio: float = 0.5) -> "CompressionPipeline":
        """添加低秩分解步骤"""
        self.steps.append({
            "type": "low_rank", "method": method, "rank_ratio": rank_ratio,
        })
        return self

    def execute(self, weights: Dict[str, np.ndarray],
                eval_fn: Callable = None) -> Dict[str, Any]:
        """执行压缩流水线"""
        current_weights = {k: v.copy() for k, v in weights.items()}
        original_size = sum(w.size for w in weights.values())
        current_accuracy = eval_fn(current_weights) if eval_fn else 1.0

        pipeline_log = []

        for step in self.steps:
            step_result = self._execute_step(step, current_weights, current_accuracy, eval_fn)
            current_weights = step_result["weights"]
            current_accuracy = step_result.get("accuracy", current_accuracy)
            pipeline_log.append(step_result)

            # 精度保护：如果低于阈值，回退
            if eval_fn and current_accuracy < self.accuracy_threshold:
                pipeline_log.append({"warning": "accuracy_below_threshold", "step": step["type"]})
                break

        compressed_size = sum(w.size for w in current_weights.values())
        ratio = compressed_size / original_size

        return {
            "weights": current_weights,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": ratio,
            "accuracy": current_accuracy,
            "log": pipeline_log,
        }

    def _execute_step(self, step: Dict[str, Any], weights: Dict[str, np.ndarray],
                      current_acc: float, eval_fn: Callable) -> Dict[str, Any]:
        """执行单个压缩步骤"""
        if step["type"] == "pruning":
            return self._prune_step(step, weights, current_acc, eval_fn)
        elif step["type"] == "quantization":
            return self._quantize_step(step, weights, current_acc, eval_fn)
        elif step["type"] == "low_rank":
            return self._lowrank_step(step, weights, current_acc, eval_fn)
        elif step["type"] == "distillation":
            return {"weights": weights, "accuracy": current_acc, "info": "distillation_applied"}
        return {"weights": weights, "accuracy": current_acc}

    def _prune_step(self, step: Dict, weights: Dict, acc: float,
                    eval_fn: Callable) -> Dict:
        if step["structured"]:
            pruner = L2StructuredPruning(pruning_ratio=step["ratio"])
        else:
            pruner = L1UnstructuredPruning(pruning_ratio=step["ratio"])

        new_weights = {}
        for name, w in weights.items():
            new_w, mask = pruner.prune(w)
            new_weights[name] = new_w

        new_acc = eval_fn(new_weights) if eval_fn else acc
        return {"weights": new_weights, "accuracy": new_acc, "method": step["method"]}

    def _quantize_step(self, step: Dict, weights: Dict, acc: float,
                       eval_fn: Callable) -> Dict:
        q = UniformQuantization(bits=step["bits"])
        new_weights = {}
        for name, w in weights.items():
            q_w, _, _ = q.quantize(w)
            new_weights[name] = q_w

        new_acc = eval_fn(new_weights) if eval_fn else acc
        return {"weights": new_weights, "accuracy": new_acc, "bits": step["bits"]}

    def _lowrank_step(self, step: Dict, weights: Dict, acc: float,
                      eval_fn: Callable) -> Dict:
        svd = SVDDecomposition(rank_ratio=step["rank_ratio"])
        new_weights = {}
        for name, w in weights.items():
            if w.ndim >= 2:
                orig_shape = w.shape
                w_2d = w.reshape(w.shape[0], -1)
                rank = max(1, int(min(w_2d.shape) * step["rank_ratio"]))
                A, B = svd.truncated_decompose(w_2d, rank)
                new_weights[name] = (A @ B).reshape(orig_shape)
            else:
                new_weights[name] = w

        new_acc = eval_fn(new_weights) if eval_fn else acc
        return {"weights": new_weights, "accuracy": new_acc, "rank_ratio": step["rank_ratio"]}


class AccuracyAwareCompression:
    """精度感知压缩：在精度约束下最大化压缩"""

    def __init__(self, min_accuracy: float = 0.95, eval_fn: Callable = None):
        self.min_accuracy = min_accuracy
        self.eval_fn = eval_fn
        self.compression_history: List[Dict[str, float]] = []

    def binary_search_compression(self, weights: Dict[str, np.ndarray],
                                   compress_fn: Callable,
                                   param_name: str = "ratio",
                                   low: float = 0.0, high: float = 0.9,
                                   tol: float = 0.01) -> Dict[str, Any]:
        """二分搜索最大压缩率"""
        best_weights = weights.copy()
        best_param = low

        while high - low > tol:
            mid = (low + high) / 2
            compressed = compress_fn(weights, **{param_name: mid})
            acc = self.eval_fn(compressed) if self.eval_fn else 1.0

            self.compression_history.append({param_name: mid, "accuracy": acc})

            if acc >= self.min_accuracy:
                best_weights = compressed
                best_param = mid
                low = mid  # 尝试更大压缩
            else:
                high = mid  # 压缩太多，回退

        return {"weights": best_weights, param_name: best_param,
                "history": self.compression_history}


class HardwareTargetedOptimization:
    """硬件目标优化：针对特定硬件平台优化"""

    def __init__(self, target_hardware: str = "edge_tpu"):
        self.target = target_hardware
        self.hardware_specs = {
            "edge_tpu": {"preferred_ops": ["int8"], "max_tensor": 8 * 1024 * 1024,
                         "latency_budget_ms": 5.0},
            "gpu": {"preferred_ops": ["float16", "tensor_core"],
                    "max_tensor": 256 * 1024 * 1024, "latency_budget_ms": 2.0},
            "cpu": {"preferred_ops": ["int8", "avx512"],
                    "max_tensor": 64 * 1024 * 1024, "latency_budget_ms": 20.0},
        }

    def optimize_for_hardware(self, weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """针对目标硬件优化"""
        spec = self.hardware_specs.get(self.target, self.hardware_specs["cpu"])
        optimized = {}
        strategy = {}

        for name, w in weights.items():
            if "int8" in spec["preferred_ops"] and w.size <= spec["max_tensor"]:
                q = UniformQuantization(bits=8)
                q_w, scale, zp = q.quantize(w)
                optimized[name] = q_w
                strategy[name] = {"quantization": "int8", "scale": scale, "zp": zp}
            elif "float16" in spec["preferred_ops"]:
                # 模拟float16
                q = UniformQuantization(bits=16, symmetric=True)
                q_w, _, _ = q.quantize(w)
                optimized[name] = q_w
                strategy[name] = {"quantization": "float16"}
            else:
                optimized[name] = w
                strategy[name] = {"quantization": "none"}

        return {"weights": optimized, "strategy": strategy, "hardware": self.target}


# ==================== 主函数 ====================

def main():
    """测试模型压缩框架"""
    np.random.seed(42)
    print("=" * 60)
    print("模型压缩与加速框架 - 综合测试")
    print("=" * 60)

    # 生成测试权重
    weights = {
        "fc1": np.random.randn(128, 256).astype(np.float32) * 0.1,
        "fc2": np.random.randn(256, 128).astype(np.float32) * 0.1,
        "conv1": np.random.randn(64, 32, 3, 3).astype(np.float32) * 0.1,
    }

    # 1. 剪枝测试
    print("\n--- 剪枝测试 ---")
    mp = MagnitudePruning(threshold=0.05)
    pruned, mask = mp.prune(weights["fc1"])
    print(f"幅度剪枝稀疏度: {mp.sparsity_ratio(pruned):.2%}")

    sp = StructuredPruning(pruning_ratio=0.3, dim=0)
    pruned_s, _ = sp.prune(weights["conv1"])
    print(f"结构化剪枝完成, 输出通道保留: {int(pruned_s[:, 0, 0, 0].sum())}/{weights['conv1'].shape[0]}")

    gp = GlobalPruning(pruning_ratio=0.5)
    global_result = gp.prune(weights)
    total_params = sum(w.size for w in weights.values())
    total_nonzero = sum(np.count_nonzero(v[0]) for v in global_result.values())
    print(f"全局剪枝稀疏度: {1 - total_nonzero / total_params:.2%}")

    lth = LotteryTicketHypothesis(pruning_ratio_per_iter=0.2, num_iterations=3)
    lth.save_initial_weights(weights)
    for i in range(3):
        rewound, sparsity = lth.prune_iteration(weights)
        print(f"  彩票假设迭代 {i+1}: 稀疏度={sparsity:.2%}")

    # 2. 量化测试
    print("\n--- 量化测试 ---")
    uq = UniformQuantization(bits=8)
    q_w, scale, zp = uq.quantize(weights["fc1"])
    error = np.mean((weights["fc1"] - q_w) ** 2)
    print(f"均匀8bit量化 MSE: {error:.6f}")

    nuq = NonUniformQuantization(bits=4)
    nq_w, centers = nuq.quantize(weights["fc1"])
    error_nu = np.mean((weights["fc1"] - nq_w) ** 2)
    print(f"非均匀4bit量化 MSE: {error_nu:.6f}")

    # 3. 知识蒸馏测试
    print("\n--- 知识蒸馏测试 ---")
    teacher_logits = np.random.randn(8, 100).astype(np.float32)
    student_logits = np.random.randn(8, 100).astype(np.float32) * 0.5
    labels = np.random.randint(0, 100, size=8)

    rbd = ResponseBasedDistillation(temperature=4.0)
    loss, metrics = rbd.compute_loss(teacher_logits, student_logits, labels)
    print(f"响应蒸馏损失: {loss:.4f} (KL={metrics['kl_loss']:.4f}, CE={metrics['ce_loss']:.4f})")

    fbd = FeatureBasedDistillation()
    t_feats = [np.random.randn(8, 64, 8, 8).astype(np.float32) for _ in range(3)]
    s_feats = [np.random.randn(8, 32, 8, 8).astype(np.float32) for _ in range(3)]
    f_loss, f_metrics = fbd.compute_loss(t_feats, s_feats)
    print(f"特征蒸馏损失: {f_loss:.4f}")

    rld = RelationBasedDistillation()
    t_f = np.random.randn(8, 128).astype(np.float32)
    s_f = np.random.randn(8, 128).astype(np.float32)
    print(f"关系蒸馏损失: {rld.compute_loss(t_f, s_f):.4f}")

    mtd = MultiTeacherDistillation(teacher_weights=[0.6, 0.4])
    t2_logits = np.random.randn(8, 100).astype(np.float32)
    mt_loss, _ = mtd.compute_loss([teacher_logits, t2_logits], student_logits, labels)
    print(f"多教师蒸馏损失: {mt_loss:.4f}")

    # 4. 低秩分解测试
    print("\n--- 低秩分解测试 ---")
    svd = SVDDecomposition(rank_ratio=0.3)
    result = svd.compress(weights["fc1"])
    print(f"SVD压缩: 原始={result['original_size']}, 压缩后={result['compressed_size']}, 比率={result['ratio']:.2%}")
    recon_error = np.mean((weights["fc1"] - result['reconstructed']) ** 2)
    print(f"  重建MSE: {recon_error:.6f}")

    tucker = TuckerDecomposition(rank_ratio=0.5)
    t_result = tucker.decompose(weights["conv1"], [32, 16, 2, 2])
    print(f"Tucker分解: 压缩比率={t_result['ratio']:.2%}")

    tt = TensorTrainDecomposition()
    tensor_4d = np.random.randn(4, 8, 8, 16).astype(np.float32)
    tt_result = tt.decompose(tensor_4d)
    print(f"TT分解: 压缩比率={tt_result['ratio']:.2%}")

    cp = CPDecomposition(rank=8)
    cp_result = cp.decompose(tensor_4d)
    print(f"CP分解: 压缩比率={cp_result['ratio']:.2%}")

    # 5. NAS测试
    print("\n--- NAS测试 ---")
    nas = WeightSharingNAS(input_dim=64)
    arch = nas.sample_architecture()
    print(f"采样架构: {arch}")

    ofa = OnceForAllNetwork()
    config = ofa.generate_subnet_config()
    params = ofa.count_subnet_params(config)
    print(f"OfA子网络参数量: {params:,}")

    # 6. 稀疏训练测试
    print("\n--- 稀疏训练测试 ---")
    dst = DynamicSparseTraining(sparsity=0.8)
    w = np.random.randn(32, 32).astype(np.float32)
    g = np.random.randn(32, 32).astype(np.float32)
    new_w = dst.prune_and_regrow(w, g)
    print(f"动态稀疏训练: 非零元素 {np.count_nonzero(new_w)}/{new_w.size}")

    rigl = RigL(sparsity=0.9)
    mask = (np.abs(w) > np.quantile(np.abs(w), 0.9)).astype(np.float32)
    new_mask = rigl.update_mask(w, g, mask)
    print(f"RigL: 非零元素 {int(new_mask.sum())}/{new_mask.size}")

    # 7. 编译优化测试
    print("\n--- 编译优化测试 ---")
    graph = [
        {"op": "conv", "output_size": 64 * 28 * 28 * 4},
        {"op": "batch_norm", "output_size": 64 * 28 * 28 * 4},
        {"op": "relu", "output_size": 64 * 28 * 28 * 4},
        {"op": "conv", "output_size": 128 * 14 * 14 * 4},
        {"op": "bias_add", "output_size": 128 * 14 * 14 * 4},
        {"op": "relu", "output_size": 128 * 14 * 14 * 4},
    ]
    fusion = OperatorFusion()
    groups = fusion.detect_fusion_opportunities(graph)
    benefits = fusion.estimate_fusion_benefit(graph, groups)
    print(f"检测到 {len(groups)} 个融合组, 节省内存访问: {benefits['total_savings_bytes']:.0f} bytes")

    tuner = KernelAutoTuning()
    best_config = tuner.tune(problem_size=(256, 256, 256))
    print(f"自动调优最优配置: {best_config}")

    # 8. 压缩流水线测试
    print("\n--- 压缩流水线测试 ---")
    pipeline = CompressionPipeline(target_compression_ratio=0.3, accuracy_threshold=0.9)
    pipeline.add_pruning("l1", ratio=0.3).add_quantization("uniform", bits=8).add_low_rank("svd", rank_ratio=0.5)

    def dummy_eval(w):
        return 0.97

    result = pipeline.execute(weights, eval_fn=dummy_eval)
    print(f"流水线结果: 压缩比率={result['compression_ratio']:.2%}, 精度={result['accuracy']:.2%}")

    hw_opt = HardwareTargetedOptimization(target_hardware="edge_tpu")
    hw_result = hw_opt.optimize_for_hardware(weights)
    print(f"硬件优化目标: {hw_result['hardware']}")

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
