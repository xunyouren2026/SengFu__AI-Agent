"""
AGI统一框架 - 评估指标与损失函数
实现各种评估指标、损失函数、评分方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
import math


# ==================== 分类指标 ====================

def accuracy(predictions: torch.Tensor, targets: torch.Tensor,
              top_k: Tuple[int, ...] = (1,)) -> List[float]:
    """Top-K准确率"""
    with torch.no_grad():
        batch_size = targets.size(0)
        _, pred_indices = predictions.topk(max(top_k), dim=1)
        pred_indices = pred_indices.t()
        correct = pred_indices.eq(targets.view(1, -1).expand_as(pred_indices))
        
        results = []
        for k in top_k:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            results.append(correct_k.item() / batch_size)
            
    return results


def precision_recall_f1(predictions: torch.Tensor, targets: torch.Tensor,
                        threshold: float = 0.5) -> Dict[str, float]:
    """精确率、召回率、F1分数"""
    predictions = (predictions > threshold).float()
    
    tp = (predictions * targets).sum().item()
    fp = (predictions * (1 - targets)).sum().item()
    fn = ((1 - predictions) * targets).sum().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {'precision': precision, 'recall': recall, 'f1': f1}


def confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor,
                     num_classes: int) -> np.ndarray:
    """混淆矩阵"""
    predictions = predictions.argmax(dim=1)
    
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, t in zip(predictions.cpu().numpy(), targets.cpu().numpy()):
        cm[t, p] += 1
        
    return cm


def per_class_accuracy(confusion_mat: np.ndarray) -> np.ndarray:
    """每类准确率"""
    return confusion_mat.diagonal() / confusion_mat.sum(axis=1)


def balanced_accuracy(confusion_mat: np.ndarray) -> float:
    """平衡准确率"""
    return per_class_accuracy(confusion_mat).mean()


def cohen_kappa(predictions: torch.Tensor, targets: torch.Tensor,
                num_classes: int) -> float:
    """Cohen's Kappa系数"""
    cm = confusion_matrix(predictions, targets, num_classes)
    
    n = cm.sum()
    po = cm.diagonal().sum() / n
    
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    pe = (row_sum * col_sum).sum() / (n ** 2)
    
    return (po - pe) / (1 - pe + 1e-8)


def matthews_correlation_coefficient(predictions: torch.Tensor, 
                                     targets: torch.Tensor) -> float:
    """Matthews相关系数 (MCC)"""
    predictions = predictions.argmax(dim=1)
    
    tp = ((predictions == 1) & (targets == 1)).sum().item()
    tn = ((predictions == 0) & (targets == 0)).sum().item()
    fp = ((predictions == 1) & (targets == 0)).sum().item()
    fn = ((predictions == 0) & (targets == 1)).sum().item()
    
    numerator = tp * tn - fp * fn
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    return numerator / (denominator + 1e-8)


# ==================== 回归指标 ====================

def mean_squared_error(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """均方误差 (MSE)"""
    return F.mse_loss(predictions, targets).item()


def mean_absolute_error(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """平均绝对误差 (MAE)"""
    return F.l1_loss(predictions, targets).item()


def root_mean_squared_error(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """均方根误差 (RMSE)"""
    return math.sqrt(mean_squared_error(predictions, targets))


def r_squared(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """R²决定系数"""
    ss_res = ((targets - predictions) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def mean_absolute_percentage_error(predictions: torch.Tensor, 
                                   targets: torch.Tensor) -> float:
    """平均绝对百分比误差 (MAPE)"""
    return ((targets - predictions).abs() / (targets.abs() + 1e-8)).mean().item()


def symmetric_mean_absolute_percentage_error(predictions: torch.Tensor,
                                             targets: torch.Tensor) -> float:
    """对称平均绝对百分比误差 (SMAPE)"""
    return (2 * (targets - predictions).abs() / 
            (targets.abs() + predictions.abs() + 1e-8)).mean().item()


def explained_variance(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """解释方差"""
    var_diff = targets.var() - (targets - predictions).var()
    return (var_diff / targets.var()).item()


# ==================== 排序指标 ====================

def auc_roc(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """AUC-ROC"""
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    # 按预测值排序
    indices = np.argsort(predictions)[::-1]
    targets_sorted = targets[indices]
    
    # 计算TPR和FPR
    tp_cumsum = np.cumsum(targets_sorted)
    fp_cumsum = np.cumsum(1 - targets_sorted)
    
    tpr = tp_cumsum / tp_cumsum[-1]
    fpr = fp_cumsum / fp_cumsum[-1]
    
    # 梯形法则计算面积
    auc = np.trapz(tpr, fpr)
    
    return float(auc)


def average_precision(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """平均精度 (AP)"""
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    indices = np.argsort(predictions)[::-1]
    targets_sorted = targets[indices]
    
    tp_cumsum = np.cumsum(targets_sorted)
    
    precision_at_k = tp_cumsum / np.arange(1, len(targets) + 1)
    
    # 只在正样本处计算
    ap = (precision_at_k * targets_sorted).sum() / targets.sum()
    
    return float(ap)


def ndcg_at_k(predictions: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """NDCG@K (Normalized Discounted Cumulative Gain)"""
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    # 按预测排序
    pred_indices = np.argsort(predictions)[::-1][:k]
    ideal_indices = np.argsort(targets)[::-1][:k]
    
    # DCG
    dcg = sum(targets[i] / math.log2(j + 2) for j, i in enumerate(pred_indices))
    idcg = sum(targets[i] / math.log2(j + 2) for j, i in enumerate(ideal_indices))
    
    return dcg / (idcg + 1e-8)


def hit_rate_at_k(predictions: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """Hit Rate@K"""
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    pred_top_k = set(np.argsort(predictions)[::-1][:k])
    relevant = set(np.where(targets > 0)[0])
    
    return 1.0 if pred_top_k & relevant else 0.0


# ==================== 损失函数 ====================

class FocalLoss(nn.Module):
    """Focal Loss"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    """标签平滑损失"""
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = predictions.size(-1)
        
        # 创建平滑标签
        smooth_targets = torch.zeros_like(predictions)
        smooth_targets.fill_(self.smoothing / (num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = -smooth_targets * F.log_softmax(predictions, dim=-1)
        
        if self.reduction == 'mean':
            return loss.sum(dim=-1).mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss.sum(dim=-1)


class DiceLoss(nn.Module):
    """Dice损失"""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = torch.sigmoid(predictions)
        
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice


class IoULoss(nn.Module):
    """IoU损失 (交并比损失)"""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = torch.sigmoid(predictions)
        
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum() - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - iou


class TverskyLoss(nn.Module):
    """Tversky损失"""
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = torch.sigmoid(predictions)
        
        tp = (predictions * targets).sum()
        fp = (predictions * (1 - targets)).sum()
        fn = ((1 - predictions) * targets).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        return 1.0 - tversky


class ContrastiveLoss(nn.Module):
    """对比损失"""
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        return loss.mean()


class TripletLoss(nn.Module):
    """三元组损失"""
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        pos_dist = (anchor - positive).pow(2).sum(dim=1)
        neg_dist = (anchor - negative).pow(2).sum(dim=1)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        return loss.mean()


class CenterLoss(nn.Module):
    """中心损失"""
    
    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        
    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_centers = self.centers[targets]
        loss = (features - batch_centers).pow(2).sum(dim=1).mean()
        return loss


class ArcFaceLoss(nn.Module):
    """ArcFace损失 (Additive Angular Margin Loss)"""
    
    def __init__(self, in_features: int, num_classes: int, 
                 scale: float = 30.0, margin: float = 0.5):
        super().__init__()
        self.scale = scale
        self.margin = margin
        
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        
    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 归一化
        features = F.normalize(features, dim=1)
        weight = F.normalize(self.weight, dim=1)
        
        # 余弦相似度
        cosine = F.linear(features, weight)
        
        # 添加角度margin
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, targets.view(-1, 1), 1)
        
        output = torch.where(one_hot.bool(), 
                            torch.cos(theta + self.margin), cosine)
        
        output = output * self.scale
        
        return F.cross_entropy(output, targets)


class CosFaceLoss(nn.Module):
    """CosFace损失 (Additive Cosine Margin Loss)"""
    
    def __init__(self, in_features: int, num_classes: int,
                 scale: float = 30.0, margin: float = 0.35):
        super().__init__()
        self.scale = scale
        self.margin = margin
        
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        
    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=1)
        weight = F.normalize(self.weight, dim=1)
        
        cosine = F.linear(features, weight)
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, targets.view(-1, 1), 1)
        
        output = cosine + one_hot * self.margin
        output = output * self.scale
        
        return F.cross_entropy(output, targets)


# ==================== 信息论指标 ====================

def entropy(probabilities: torch.Tensor) -> float:
    """熵"""
    probabilities = probabilities + 1e-8
    return (-probabilities * torch.log(probabilities)).sum(dim=-1).mean().item()


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """KL散度"""
    p = p + 1e-8
    q = q + 1e-8
    return (p * torch.log(p / q)).sum(dim=-1).mean().item()


def cross_entropy(p: torch.Tensor, q: torch.Tensor) -> float:
    """交叉熵"""
    q = q + 1e-8
    return -(p * torch.log(q)).sum(dim=-1).mean().item()


def mutual_information(joint: torch.Tensor) -> float:
    """互信息"""
    # 边缘分布
    p_x = joint.sum(dim=1)
    p_y = joint.sum(dim=0)
    
    # I(X;Y) = sum p(x,y) log(p(x,y) / (p(x)p(y)))
    joint = joint + 1e-8
    p_xy = p_x.unsqueeze(1) * p_y.unsqueeze(0)
    
    mi = (joint * torch.log(joint / p_xy)).sum()
    
    return mi.item()


# ==================== 语言模型指标 ====================

def perplexity(loss: float) -> float:
    """困惑度"""
    return math.exp(loss)


def bleu_score(predictions: List[str], references: List[str],
               max_n: int = 4) -> float:
    """BLEU分数 (简化版)"""
    from collections import Counter
    
    def get_ngrams(text: str, n: int) -> Counter:
        tokens = text.split()
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngrams[tuple(tokens[i:i+n])] += 1
        return ngrams
    
    # 计算各阶n-gram精度
    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = get_ngrams(predictions[0], n)
        ref_ngrams = get_ngrams(references[0], n)
        
        overlap = sum(min(pred_ngrams[ng], ref_ngrams[ng]) for ng in pred_ngrams)
        total = sum(pred_ngrams.values())
        
        precisions.append(overlap / (total + 1e-8))
    
    # 几何平均
    if all(p > 0 for p in precisions):
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    else:
        geo_mean = 0.0
    
    # 简短惩罚
    pred_len = len(predictions[0].split())
    ref_len = len(references[0].split())
    
    if pred_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / pred_len)
    
    return bp * geo_mean


def word_error_rate(predictions: List[str], references: List[str]) -> float:
    """词错误率 (WER)"""
    def levenshtein_distance(s1: List[str], s2: List[str]) -> int:
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]
    
    total_distance = 0
    total_length = 0
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        
        total_distance += levenshtein_distance(pred_tokens, ref_tokens)
        total_length += len(ref_tokens)
        
    return total_distance / total_length
