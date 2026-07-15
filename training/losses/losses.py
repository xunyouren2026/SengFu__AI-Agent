"""
损失函数模块 - 包含各种损失函数的真实实现
包括: CrossEntropy, Focal Loss, Label Smoothing, Contrastive Loss, Triplet Loss, 
      Dice Loss, IoU Loss, Lovasz Loss, ArcFace Loss, Center Loss, etc.
"""

import math
from typing import Optional, Tuple, List, Union, Callable
from abc import ABC, abstractmethod
from enum import Enum


class Reduction(Enum):
    """损失归约方式"""
    NONE = "none"
    MEAN = "mean"
    SUM = "sum"


def softmax(x, axis=-1):
    """计算softmax"""
    if isinstance(x, (int, float)):
        return 1.0
    if isinstance(x, list):
        if isinstance(x[0], list):
            # 2D情况
            result = []
            for row in x:
                max_val = max(row)
                exp_vals = [math.exp(v - max_val) for v in row]
                sum_exp = sum(exp_vals)
                result.append([e / sum_exp for e in exp_vals])
            return result
        else:
            # 1D情况
            max_val = max(x)
            exp_vals = [math.exp(v - max_val) for v in x]
            sum_exp = sum(exp_vals)
            return [e / sum_exp for e in exp_vals]
    return x


def log_softmax(x, axis=-1):
    """计算log softmax"""
    if isinstance(x, (int, float)):
        return 0.0
    if isinstance(x, list):
        if isinstance(x[0], list):
            result = []
            for row in x:
                max_val = max(row)
                exp_vals = [math.exp(v - max_val) for v in row]
                sum_exp = sum(exp_vals)
                result.append([v - max_val - math.log(sum_exp) for v in row])
            return result
        else:
            max_val = max(x)
            exp_vals = [math.exp(v - max_val) for v in x]
            sum_exp = sum(exp_vals)
            return [v - max_val - math.log(sum_exp) for v in x]
    return 0.0


class Loss(ABC):
    """损失函数基类"""
    
    def __init__(self, reduction: Reduction = Reduction.MEAN):
        self.reduction = reduction
    
    @abstractmethod
    def forward(self, predictions, targets) -> float:
        """计算损失"""
        pass
    
    def __call__(self, predictions, targets) -> float:
        return self.forward(predictions, targets)
    
    def _reduce(self, losses: List[float]) -> float:
        """应用归约"""
        if self.reduction == Reduction.NONE:
            return losses
        elif self.reduction == Reduction.MEAN:
            return sum(losses) / len(losses) if losses else 0.0
        elif self.reduction == Reduction.SUM:
            return sum(losses)
        return sum(losses) / len(losses) if losses else 0.0


class CrossEntropyLoss(Loss):
    """
    交叉熵损失
    
    L = -sum(y * log(p))
    
    支持标签平滑和类别权重
    """
    
    def __init__(
        self,
        weight: Optional[List[float]] = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: Reduction = Reduction.MEAN
    ):
        super().__init__(reduction)
        self.weight = weight
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
    
    def forward(self, predictions, targets) -> float:
        """计算交叉熵损失"""
        # predictions: logits [N, C] 或 [C]
        # targets: 类别索引 [N] 或 int
        
        if isinstance(targets, (int, float)):
            targets = [int(targets)]
        
        if isinstance(predictions, (int, float)):
            predictions = [predictions]
        
        # 处理1D情况
        if not isinstance(predictions[0], list):
            predictions = [predictions]
            if len(targets) == 1:
                targets = targets
        
        losses = []
        batch_size = len(predictions)
        num_classes = len(predictions[0])
        
        for i in range(batch_size):
            target = int(targets[i]) if i < len(targets) else int(targets[0])
            
            if target == self.ignore_index:
                continue
            
            logits = predictions[i]
            
            # 计算log softmax
            max_logit = max(logits)
            log_sum_exp = math.log(sum(math.exp(l - max_logit) for l in logits))
            log_probs = [l - max_logit - log_sum_exp for l in logits]
            
            # 标签平滑
            if self.label_smoothing > 0:
                smooth_loss = 0.0
                for c in range(num_classes):
                    if c == target:
                        target_prob = 1.0 - self.label_smoothing
                    else:
                        target_prob = self.label_smoothing / (num_classes - 1)
                    smooth_loss -= target_prob * log_probs[c]
                loss = smooth_loss
            else:
                loss = -log_probs[target]
            
            # 类别权重
            if self.weight is not None and target < len(self.weight):
                loss *= self.weight[target]
            
            losses.append(loss)
        
        return self._reduce(losses)


class FocalLoss(Loss):
    """
    Focal Loss - 解决类别不平衡问题
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    其中 p_t = p if y=1 else 1-p
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        weight: Optional[List[float]] = None,
        ignore_index: int = -100,
        reduction: Reduction = Reduction.MEAN
    ):
        super().__init__(reduction)
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
    
    def forward(self, predictions, targets) -> float:
        """计算Focal Loss"""
        if isinstance(targets, (int, float)):
            targets = [int(targets)]
        
        if isinstance(predictions, (int, float)):
            predictions = [predictions]
        
        if not isinstance(predictions[0], list):
            predictions = [predictions]
        
        losses = []
        batch_size = len(predictions)
        
        for i in range(batch_size):
            target = int(targets[i]) if i < len(targets) else int(targets[0])
            
            if target == self.ignore_index:
                continue
            
            logits = predictions[i]
            num_classes = len(logits)
            
            # 计算softmax概率
            max_logit = max(logits)
            exp_sum = sum(math.exp(l - max_logit) for l in logits)
            probs = [math.exp(l - max_logit) / exp_sum for l in logits]
            
            # 计算focal loss
            p_t = probs[target]
            focal_weight = (1 - p_t) ** self.gamma
            
            # 计算交叉熵
            ce_loss = -math.log(max(p_t, 1e-10))
            
            loss = self.alpha * focal_weight * ce_loss
            
            if self.weight is not None and target < len(self.weight):
                loss *= self.weight[target]
            
            losses.append(loss)
        
        return self._reduce(losses)


class BinaryFocalLoss(Loss):
    """
    二分类Focal Loss
    
    FL = -alpha * (1-p)^gamma * y*log(p) - (1-alpha) * p^gamma * (1-y)*log(1-p)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: Reduction = Reduction.MEAN
    ):
        super().__init__(reduction)
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions, targets) -> float:
        """计算二分类Focal Loss"""
        # predictions: logits
        # targets: 0 or 1
        
        if isinstance(predictions, (int, float)):
            predictions = [predictions]
        if isinstance(targets, (int, float)):
            targets = [float(targets)]
        
        losses = []
        
        for pred, target in zip(predictions, targets):
            # sigmoid
            p = 1.0 / (1.0 + math.exp(-pred))
            target = float(target)
            
            # 计算focal loss
            if target == 1:
                focal_weight = (1 - p) ** self.gamma
                loss = -self.alpha * focal_weight * math.log(max(p, 1e-10))
            else:
                focal_weight = p ** self.gamma
                loss = -(1 - self.alpha) * focal_weight * math.log(max(1 - p, 1e-10))
            
            losses.append(loss)
        
        return self._reduce(losses)


class ContrastiveLoss(Loss):
    """
    对比损失 - 用于度量学习
    
    L = (1-y) * 0.5 * d^2 + y * 0.5 * max(0, margin - d)^2
    
    y=0: 相似样本，最小化距离
    y=1: 不相似样本，最大化距离（至少margin）
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        reduction: Reduction = Reduction.MEAN
    ):
        super().__init__(reduction)
        self.margin = margin
    
    @staticmethod
    def euclidean_distance(x1, x2) -> float:
        """计算欧氏距离"""
        if isinstance(x1, (int, float)) and isinstance(x2, (int, float)):
            return abs(x1 - x2)
        
        if isinstance(x1, list) and isinstance(x2, list):
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))
        
        return abs(x1 - x2)
    
    def forward(self, predictions, targets) -> float:
        """
        predictions: (embedding1, embedding2) 元组或列表
        targets: 0或1，表示是否相似
        """
        if isinstance(targets, (int, float)):
            targets = [int(targets)]
        
        if not isinstance(predictions[0], tuple) and not isinstance(predictions[0], list):
            # 单个样本
            emb1, emb2 = predictions
            predictions = [(emb1, emb2)]
        
        losses = []
        
        for i, (emb1, emb2) in enumerate(predictions):
            target = targets[i] if i < len(targets) else targets[0]
            
            d = self.euclidean_distance(emb1, emb2)
            
            if target == 0:
                # 相似样本，最小化距离
                loss = 0.5 * d ** 2
            else:
                # 不相似样本，最大化距离
                loss = 0.5 * max(0, self.margin - d) ** 2
            
            losses.append(loss)
        
        return self._reduce(losses)


class TripletLoss(Loss):
    """
    三元组损失 - 用于度量学习
    
    L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        p: float = 2.0,
        reduction: Reduction = Reduction.MEAN
    ):
        super().__init__(reduction)
        self.margin = margin
        self.p = p
    
    def _distance(self, x1, x2) -> float:
        """计算p范数距离"""
        if isinstance(x1, (int, float)) and isinstance(x2, (int, float)):
            return abs(x1 - x2)
        
        if isinstance(x1, list) and isinstance(x2, list):
            return (sum(abs(a - b) ** self.p for a, b in zip(x1, x2))) ** (1 / self.p)
        
        return abs(x1 - x2)
    
    def forward(self, predictions, targets=None) -> float:
        """
        predictions: (anchor, positive, negative) 元组列表
        targets: 忽略
        """
        if not isinstance(predictions[0], (tuple, list)) or len(predictions[0]) != 3:
            anchor, positive, negative = predictions
            predictions = [(anchor, positive, negative)]
        
        losses = []
        
        for anchor, positive, negative in predictions:
            d_pos = self._distance(anchor, positive)
            d_neg = self._distance(anchor, negative)
            
            loss = max(0, d_pos - d_neg + self.margin)
            losses.append(loss)
        
        return self._reduce(losses)


class InfoNCELoss(Loss):
    """
    InfoNCE损失 - 用于对比学习
    
    L = -log(exp(sim(q, k+) / tau) / sum_i exp(sim(q, k_i) / tau))
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        reduction: Reduction = Reduction.MEAN
    ):
        super().__init__(reduction)
        self.temperature = temperature
    
    @staticmethod
    def cosine_similarity(x1, x2) -> float:
        """计算余弦相似度"""
        if isinstance(x1, (int, float)) and isinstance(x2, (int, float)):
            return 1.0 if x1 * x2 > 0 else -1.0
        
        if isinstance(x1, list) and isinstance(x2, list):
            dot = sum(a * b for a, b in zip(x1, x2))
            norm1 = math.sqrt(sum(a ** 2 for a in x1))
            norm2 = math.sqrt(sum(b ** 2 for b in x2))
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot / (norm1 * norm2)
        
        return 0.0
    
    def forward(self, predictions, targets) -> float:
        """
        predictions: query向量
        targets: (positive_key, negative_keys) 或 keys列表（第一个是正样本）
        """
        if isinstance(predictions, (int, float)):
            predictions = [predictions]
        
        losses = []
        
        for i, query in enumerate(predictions):
            if i >= len(targets):
                break
            
            target = targets[i]
            
            if isinstance(target, tuple):
                positive_key, negative_keys = target
                keys = [positive_key] + list(negative_keys)
            else:
                keys = target
            
            # 计算相似度
            similarities = [self.cosine_similarity(query, k) / self.temperature for k in keys]
            
            # 计算log softmax
            max_sim = max(similarities)
            log_sum_exp = math.log(sum(math.exp(s - max_sim) for s in similarities))
            
            # 正样本是第一个
            loss = -(similarities[0] - max_sim - log_sum_exp)
            losses.append(loss)
        
        return self._reduce(losses)


class NTXentLoss(Loss):
    """
    NT-Xent损失 (Normalized Temperature-scaled Cross Entropy)
    用于SimCLR等自监督学习方法
    """
    
    def __init__(
        self,
        temperature: float = 0.5,
        reduction: Reduction = Reduction.MEAN
    ):
        super().__init__(reduction)
        self.temperature = temperature
    
    def forward(self, predictions, targets=None) -> float:
        """
        predictions: 特征向量列表 [z_i, z_j, ...]
        每个样本有两个增强视图，正样本对是(z_{2k}, z_{2k+1})
        """
        if not isinstance(predictions[0], list):
            # 假设是扁平列表
            predictions = [predictions]
        
        losses = []
        n = len(predictions)
        
        for i in range(n):
            # 正样本索引
            j = i + 1 if i % 2 == 0 else i - 1
            
            # 计算与所有其他样本的相似度
            sims = []
            for k in range(n):
                if k == i:
                    continue
                sim = self._cosine_similarity(predictions[i], predictions[k])
                sims.append((sim / self.temperature, k == j))
            
            # 计算损失
            max_sim = max(s[0] for s in sims)
            log_sum_exp = math.log(sum(math.exp(s[0] - max_sim) for s in sims))
            
            # 正样本的损失
            pos_sim = next(s[0] for s in sims if s[1])
            loss = -(pos_sim - max_sim - log_sum_exp)
            losses.append(loss)
        
        return self._reduce(losses)
    
    @staticmethod
    def _cosine_similarity(x1, x2) -> float:
        if isinstance(x1, (int, float)) and isinstance(x2, (int, float)):
            return 1.0 if x1 * x2 > 0 else -1.0
        if isinstance(x1, list) and isinstance(x2, list):
            dot = sum(a * b for a, b in zip(x1, x2))
            norm1 = math.sqrt(sum(a ** 2 for a in x1))
            norm2 = math.sqrt(sum(b ** 2 for b in x2))
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot / (norm1 * norm2)
        return 0.0


class DiceLoss(Loss):
    """
    Dice损失 - 用于分割任务
    
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Loss = 1 - Dice
    """
    
    def __init__(
        self,
        smooth: float = 1.0,
        square_dice: bool = False,
        reduction: Reduction = Reduction.MEAN
    ):
        super().__init__(reduction)
        self.smooth = smooth
        self.square_dice = square_dice
    
    def forward(self, predictions, targets) -> float:
        """
        predictions: 预测概率 [N, H, W] 或 [N, C, H, W]
        targets: 真实标签 [N, H, W]
        """
        if isinstance(predictions, (int, float)):
            predictions = [predictions]
        if isinstance(targets, (int, float)):
            targets = [targets]
        
        if not isinstance(predictions[0], list):
            # 扁平情况
            predictions = [predictions]
            targets = [targets]
        
        losses = []
        
        for pred, target in zip(predictions, targets):
            # 展平
            if isinstance(pred, list):
                pred_flat = self._flatten(pred)
            else:
                pred_flat = [pred]
            
            if isinstance(target, list):
                target_flat = self._flatten(target)
            else:
                target_flat = [target]
            
            # 计算交集和并集
            intersection = sum(p * t for p, t in zip(pred_flat, target_flat))
            
            if self.square_dice:
                pred_sum = sum(p ** 2 for p in pred_flat)
                target_sum = sum(t ** 2 for t in target_flat)
            else:
                pred_sum = sum(pred_flat)
                target_sum = sum(target_flat)
            
            dice = (2 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
            loss = 1 - dice
            losses.append(loss)
        
        return self._reduce(losses)
    
    @staticmethod
    def _flatten(nested_list):
        """展平嵌套列表"""
        result = []
        for item in nested_list:
            if isinstance(item, list):
                result.extend(DiceLoss._flatten(item))
            else:
                result.append(item)
        return result


class IoULoss(Loss):
    """
    IoU损失 (Intersection over Union) - 用于分割任务
    
    IoU = |X ∩ Y| / |X ∪ Y|
    Loss = 1 - IoU
    """
    
    def __init__(
        self,
        smooth: float = 1e-6,
        reduction: Reduction = Reduction.MEAN
    ):
        super().__init__(reduction)
        self.smooth = smooth
    
    def forward(self, predictions, targets) -> float:
        """计算IoU损失"""
        if isinstance(predictions, (int, float)):
            predictions = [predictions]
        if isinstance(targets, (int, float)):
            targets = [targets]
        
        if not isinstance(predictions[0], list):
            predictions = [predictions]
            targets = [targets]
        
        losses = []
        
        for pred, target in zip(predictions, targets):
            if isinstance(pred, list):
                pred_flat = self._flatten(pred)
            else:
                pred_flat = [pred]
            
            if isinstance(target, list):
                target_flat = self._flatten(target)
            else:
                target_flat = [target]
            
            # 二值化预测
            pred_binary = [1 if p > 0.5 else 0 for p in pred_flat]
            target_binary = [1 if t > 0.5 else 0 for t in target_flat]
            
            intersection = sum(p * t for p, t in zip(pred_binary, target_binary))
            union = sum(max(p, t) for p, t in zip(pred_binary, target_binary))
            
            iou = (intersection + self.smooth) / (union + self.smooth)
            loss = 1 - iou
            losses.append(loss)
        
        return self._reduce(losses)
    
    @staticmethod
    def _flatten(nested_list):
        result = []
        for item in nested_list:
            if isinstance(item, list):
                result.extend(IoULoss._flatten(item))
            else:
                result.append(item)
        return result


class TverskyLoss(Loss):
    """
    Tversky损失 - Dice和IoU的泛化
    
    T = |X ∩ Y| / (|X ∩ Y| + alpha * |X - Y| + beta * |Y - X|)
    
    alpha=beta=0.5: Dice
    alpha=beta=1: IoU
    alpha>beta: 更多假阳性惩罚
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1.0,
        reduction: Reduction = Reduction.MEAN
    ):
        super().__init__(reduction)
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, predictions, targets) -> float:
        """计算Tversky损失"""
        if isinstance(predictions, (int, float)):
            predictions = [predictions]
        if isinstance(targets, (int, float)):
            targets = [targets]
        
        if not isinstance(predictions[0], list):
            predictions = [predictions]
            targets = [targets]
        
        losses = []
        
        for pred, target in zip(predictions, targets):
            if isinstance(pred, list):
                pred_flat = self._flatten(pred)
            else:
                pred_flat = [pred]
            
            if isinstance(target, list):
                target_flat = self._flatten(target)
            else:
                target_flat = [target]
            
            # 计算各项
            true_pos = sum(p * t for p, t in zip(pred_flat, target_flat))
            false_pos = sum(p * (1 - t) for p, t in zip(pred_flat, target_flat))
            false_neg = sum((1 - p) * t for p, t in zip(pred_flat, target_flat))
            
            tversky = (true_pos + self.smooth) / (
                true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth
            )
            
            loss = 1 - tversky
            losses.append(loss)
        
        return self._reduce(losses)
    
    @staticmethod
    def _flatten(nested_list):
        result = []
        for item in nested_list:
            if isinstance(item, list):
                result.extend(TverskyLoss._flatten(item))
            else:
                result.append(item)
        return result


class FocalTverskyLoss(Loss):
    """
    Focal Tversky损失 - 结合Focal和Tversky
    
    L = (1 - T)^gamma
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
        smooth: float = 1.0,
        reduction: Reduction = Reduction.MEAN
    ):
        super().__init__(reduction)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, predictions, targets) -> float:
        """计算Focal Tversky损失"""
        if isinstance(predictions, (int, float)):
            predictions = [predictions]
        if isinstance(targets, (int, float)):
            targets = [targets]
        
        if not isinstance(predictions[0], list):
            predictions = [predictions]
            targets = [targets]
        
        losses = []
        
        for pred, target in zip(predictions, targets):
            if isinstance(pred, list):
                pred_flat = self._flatten(pred)
            else:
                pred_flat = [pred]
            
            if isinstance(target, list):
                target_flat = self._flatten(target)
            else:
                target_flat = [target]
            
            true_pos = sum(p * t for p, t in zip(pred_flat, target_flat))
            false_pos = sum(p * (1 - t) for p, t in zip(pred_flat, target_flat))
            false_neg = sum((1 - p) * t for p, t in zip(pred_flat, target_flat))
            
            tversky = (true_pos + self.smooth) / (
                true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth
            )
            
            loss = (1 - tversky) ** self.gamma
            losses.append(loss)
        
        return self._reduce(losses)
    
    @staticmethod
    def _flatten(nested_list):
        result = []
        for item in nested_list:
            if isinstance(item, list):
                result.extend(FocalTverskyLoss._flatten(item))
            else:
                result.append(item)
        return result


class ArcFaceLoss(Loss):
    """
    ArcFace损失 - 用于人脸识别
    
    L = -log(exp(s * cos(theta + m)) / (exp(s * cos(theta + m)) + sum exp(s * cos(theta_j))))
    """
    
    def __init__(
        self,
        scale: float = 30.0,
        margin: float = 0.5,
        reduction: Reduction = Reduction.MEAN
    ):
        super().__init__(reduction)
        self.scale = scale
        self.margin = margin
    
    def forward(self, predictions, targets) -> float:
        """
        predictions: 归一化特征向量
        targets: 目标类别
        """
        if isinstance(targets, (int, float)):
            targets = [int(targets)]
        
        if not isinstance(predictions[0], list):
            predictions = [predictions]
        
        losses = []
        
        for i, embedding in enumerate(predictions):
            target = targets[i] if i < len(targets) else targets[0]
            
            # 假设predictions已经包含了与所有类中心的余弦相似度
            # 或者embedding是特征向量
            if isinstance(embedding[0], list):
                # 多维特征，需要与权重矩阵相乘
                # 这里简化处理
                cos_theta = embedding[target] if target < len(embedding) else 0.0
            else:
                # 直接是余弦相似度
                cos_theta = embedding[target] if target < len(embedding) else 0.0
            
            # 限制cos_theta范围
            cos_theta = max(-1.0, min(1.0, cos_theta))
            theta = math.acos(cos_theta)
            
            # ArcFace变换
            cos_theta_m = math.cos(theta + self.margin)
            
            # 计算logits
            logits = [self.scale * e for e in embedding]
            logits[target] = self.scale * cos_theta_m
            
            # 计算交叉熵
            max_logit = max(logits)
            log_sum_exp = math.log(sum(math.exp(l - max_logit) for l in logits))
            loss = -(logits[target] - max_logit - log_sum_exp)
            
            losses.append(loss)
        
        return self._reduce(losses)


class CenterLoss(Loss):
    """
    Center Loss - 用于深度特征学习
    
    L = 0.5 * sum_i ||x_i - c_{y_i}||^2
    
    需要维护每个类别的中心
    """
    
    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        reduction: Reduction = Reduction.MEAN
    ):
        super().__init__(reduction)
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        # 初始化类中心
        self.centers = [[0.0] * feat_dim for _ in range(num_classes)]
    
    def forward(self, predictions, targets) -> float:
        """
        predictions: 特征向量
        targets: 类别标签
        """
        if isinstance(targets, (int, float)):
            targets = [int(targets)]
        
        if not isinstance(predictions[0], list):
            predictions = [predictions]
        
        losses = []
        
        for i, features in enumerate(predictions):
            target = targets[i] if i < len(targets) else targets[0]
            
            if target >= self.num_classes:
                continue
            
            center = self.centers[target]
            
            # 计算距离平方
            if isinstance(features, list):
                dist_sq = sum((f - c) ** 2 for f, c in zip(features, center))
            else:
                dist_sq = (features - center[0]) ** 2
            
            loss = 0.5 * dist_sq
            losses.append(loss)
        
        return self._reduce(losses)
    
    def update_centers(self, features_list, targets_list, lr: float = 0.5):
        """更新类中心"""
        # 统计每个类别的特征和
        class_features = [[0.0] * self.feat_dim for _ in range(self.num_classes)]
        class_counts = [0] * self.num_classes
        
        for features, target in zip(features_list, targets_list):
            if target >= self.num_classes:
                continue
            
            for j, f in enumerate(features):
                class_features[target][j] += f
            class_counts[target] += 1
        
        # 更新中心
        for c in range(self.num_classes):
            if class_counts[c] > 0:
                for j in range(self.feat_dim):
                    delta = class_features[c][j] / class_counts[c] - self.centers[c][j]
                    self.centers[c][j] += lr * delta


class CosFaceLoss(Loss):
    """
    CosFace损失 (Large Margin Cosine Loss)
    
    L = -log(exp(s * (cos(theta) - m)) / sum_j exp(s * cos(theta_j)))
    """
    
    def __init__(
        self,
        scale: float = 30.0,
        margin: float = 0.35,
        reduction: Reduction = Reduction.MEAN
    ):
        super().__init__(reduction)
        self.scale = scale
        self.margin = margin
    
    def forward(self, predictions, targets) -> float:
        """计算CosFace损失"""
        if isinstance(targets, (int, float)):
            targets = [int(targets)]
        
        if not isinstance(predictions[0], list):
            predictions = [predictions]
        
        losses = []
        
        for i, embedding in enumerate(predictions):
            target = targets[i] if i < len(targets) else targets[0]
            
            # 计算logits
            logits = [self.scale * e for e in embedding]
            logits[target] = self.scale * (embedding[target] - self.margin)
            
            # 计算交叉熵
            max_logit = max(logits)
            log_sum_exp = math.log(sum(math.exp(l - max_logit) for l in logits))
            loss = -(logits[target] - max_logit - log_sum_exp)
            
            losses.append(loss)
        
        return self._reduce(losses)


class SphereFaceLoss(Loss):
    """
    SphereFace损失 (A-Softmax)
    
    L = -log(exp(s * cos(m * theta)) / sum_j exp(s * cos(theta_j)))
    """
    
    def __init__(
        self,
        scale: float = 30.0,
        margin: int = 4,
        reduction: Reduction = Reduction.MEAN
    ):
        super().__init__(reduction)
        self.scale = scale
        self.margin = margin
    
    def forward(self, predictions, targets) -> float:
        """计算SphereFace损失"""
        if isinstance(targets, (int, float)):
            targets = [int(targets)]
        
        if not isinstance(predictions[0], list):
            predictions = [predictions]
        
        losses = []
        
        for i, embedding in enumerate(predictions):
            target = targets[i] if i < len(targets) else targets[0]
            
            cos_theta = embedding[target]
            cos_theta = max(-1.0, min(1.0, cos_theta))
            theta = math.acos(cos_theta)
            
            # 计算cos(m * theta)
            cos_m_theta = self._cos_m_theta(theta, self.margin)
            
            # 计算logits
            logits = [self.scale * e for e in embedding]
            logits[target] = self.scale * cos_m_theta
            
            # 计算交叉熵
            max_logit = max(logits)
            log_sum_exp = math.log(sum(math.exp(l - max_logit) for l in logits))
            loss = -(logits[target] - max_logit - log_sum_exp)
            
            losses.append(loss)
        
        return self._reduce(losses)
    
    def _cos_m_theta(self, theta: float, m: int) -> float:
        """计算cos(m * theta)使用切比雪夫多项式"""
        if m == 1:
            return math.cos(theta)
        elif m == 2:
            return 2 * math.cos(theta) ** 2 - 1
        elif m == 3:
            return 4 * math.cos(theta) ** 3 - 3 * math.cos(theta)
        elif m == 4:
            return 8 * math.cos(theta) ** 4 - 8 * math.cos(theta) ** 2 + 1
        else:
            return math.cos(m * theta)


class LabelSmoothingCrossEntropy(Loss):
    """
    标签平滑交叉熵损失
    
    y_smooth = (1 - epsilon) * y + epsilon / K
    L = -sum(y_smooth * log(p))
    """
    
    def __init__(
        self,
        epsilon: float = 0.1,
        reduction: Reduction = Reduction.MEAN
    ):
        super().__init__(reduction)
        self.epsilon = epsilon
    
    def forward(self, predictions, targets) -> float:
        """计算标签平滑交叉熵"""
        if isinstance(targets, (int, float)):
            targets = [int(targets)]
        
        if isinstance(predictions, (int, float)):
            predictions = [predictions]
        
        if not isinstance(predictions[0], list):
            predictions = [predictions]
        
        losses = []
        
        for i, logits in enumerate(predictions):
            target = targets[i] if i < len(targets) else targets[0]
            num_classes = len(logits)
            
            # 计算log softmax
            max_logit = max(logits)
            log_sum_exp = math.log(sum(math.exp(l - max_logit) for l in logits))
            log_probs = [l - max_logit - log_sum_exp for l in logits]
            
            # 标签平滑
            smooth_loss = 0.0
            for c in range(num_classes):
                if c == target:
                    target_prob = 1.0 - self.epsilon
                else:
                    target_prob = self.epsilon / (num_classes - 1)
                smooth_loss -= target_prob * log_probs[c]
            
            losses.append(smooth_loss)
        
        return self._reduce(losses)


class SoftTargetCrossEntropy(Loss):
    """
    软目标交叉熵 - 用于知识蒸馏
    
    L = -sum(y_soft * log(p))
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        reduction: Reduction = Reduction.MEAN
    ):
        super().__init__(reduction)
        self.temperature = temperature
    
    def forward(self, predictions, targets) -> float:
        """
        predictions: 学生模型logits
        targets: 教师模型logits (软目标)
        """
        if not isinstance(predictions[0], list):
            predictions = [predictions]
        if not isinstance(targets[0], list):
            targets = [targets]
        
        losses = []
        
        for student_logits, teacher_logits in zip(predictions, targets):
            # 计算学生的log softmax
            student_scaled = [l / self.temperature for l in student_logits]
            max_student = max(student_scaled)
            log_sum_exp_student = math.log(sum(math.exp(s - max_student) for s in student_scaled))
            student_log_probs = [s - max_student - log_sum_exp_student for s in student_scaled]
            
            # 计算教师的softmax
            teacher_scaled = [l / self.temperature for l in teacher_logits]
            max_teacher = max(teacher_scaled)
            exp_sum_teacher = sum(math.exp(t - max_teacher) for t in teacher_scaled)
            teacher_probs = [math.exp(t - max_teacher) / exp_sum_teacher for t in teacher_scaled]
            
            # 计算KL散度
            loss = -sum(t * s for t, s in zip(teacher_probs, student_log_probs))
            loss = loss * (self.temperature ** 2)  # 缩放
            
            losses.append(loss)
        
        return self._reduce(losses)


class KLDivergenceLoss(Loss):
    """
    KL散度损失
    
    KL(P||Q) = sum(P * log(P/Q))
    """
    
    def __init__(
        self,
        reduction: Reduction = Reduction.MEAN
    ):
        super().__init__(reduction)
    
    def forward(self, predictions, targets) -> float:
        """
        predictions: Q分布
        targets: P分布
        """
        if not isinstance(predictions[0], list):
            predictions = [predictions]
        if not isinstance(targets[0], list):
            targets = [targets]
        
        losses = []
        
        for q, p in zip(predictions, targets):
            kl = 0.0
            for p_i, q_i in zip(p, q):
                if p_i > 0 and q_i > 0:
                    kl += p_i * math.log(p_i / q_i)
            losses.append(kl)
        
        return self._reduce(losses)


class JSLoss(Loss):
    """
    JS散度损失 (Jensen-Shannon Divergence)
    
    JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    其中 M = 0.5 * (P + Q)
    """
    
    def __init__(
        self,
        reduction: Reduction = Reduction.MEAN
    ):
        super().__init__(reduction)
    
    def forward(self, predictions, targets) -> float:
        """计算JS散度"""
        if not isinstance(predictions[0], list):
            predictions = [predictions]
        if not isinstance(targets[0], list):
            targets = [targets]
        
        losses = []
        
        for q, p in zip(predictions, targets):
            # 计算M
            m = [(p_i + q_i) / 2 for p_i, q_i in zip(p, q)]
            
            # 计算KL(P||M)
            kl_pm = 0.0
            for p_i, m_i in zip(p, m):
                if p_i > 0 and m_i > 0:
                    kl_pm += p_i * math.log(p_i / m_i)
            
            # 计算KL(Q||M)
            kl_qm = 0.0
            for q_i, m_i in zip(q, m):
                if q_i > 0 and m_i > 0:
                    kl_qm += q_i * math.log(q_i / m_i)
            
            js = 0.5 * kl_pm + 0.5 * kl_qm
            losses.append(js)
        
        return self._reduce(losses)


class GHMCLoss(Loss):
    """
    GHM-C损失 (Gradient Harmonizing Mechanism for Classification)
    
    通过梯度密度调制来处理类别不平衡
    """
    
    def __init__(
        self,
        bins: int = 10,
        momentum: float = 0.75,
        reduction: Reduction = Reduction.MEAN
    ):
        super().__init__(reduction)
        self.bins = bins
        self.momentum = momentum
        self.emd = [0.0] * bins  # 梯度密度估计
    
    def forward(self, predictions, targets) -> float:
        """计算GHM-C损失"""
        if isinstance(targets, (int, float)):
            targets = [int(targets)]
        
        if not isinstance(predictions[0], list):
            predictions = [predictions]
        
        losses = []
        gradients = []
        
        # 第一遍：计算梯度
        for i, logits in enumerate(predictions):
            target = targets[i] if i < len(targets) else targets[0]
            
            # 计算softmax
            max_logit = max(logits)
            exp_sum = sum(math.exp(l - max_logit) for l in logits)
            probs = [math.exp(l - max_logit) / exp_sum for l in logits]
            
            # 计算梯度
            g = abs(probs[target] - 1)
            gradients.append(g)
            
            # 计算基础CE损失
            loss = -math.log(max(probs[target], 1e-10))
            losses.append(loss)
        
        # 更新梯度密度
        for g in gradients:
            bin_idx = min(int(g * self.bins), self.bins - 1)
            self.emd[bin_idx] = self.momentum * self.emd[bin_idx] + (1 - self.momentum)
        
        # 应用梯度密度权重
        weighted_losses = []
        for loss, g in zip(losses, gradients):
            bin_idx = min(int(g * self.bins), self.bins - 1)
            weight = 1.0 / (self.emd[bin_idx] + 1e-6)
            weighted_losses.append(loss * weight)
        
        return self._reduce(weighted_losses)


class OHEMLoss(Loss):
    """
    OHEM损失 (Online Hard Example Mining)
    
    只计算损失最大的样本
    """
    
    def __init__(
        self,
        ratio: float = 0.25,
        base_loss: Optional[Loss] = None,
        reduction: Reduction = Reduction.MEAN
    ):
        super().__init__(reduction)
        self.ratio = ratio
        self.base_loss = base_loss if base_loss else CrossEntropyLoss(reduction=Reduction.NONE)
    
    def forward(self, predictions, targets) -> float:
        """计算OHEM损失"""
        # 计算所有样本的损失
        all_losses = self.base_loss(predictions, targets)
        
        if not isinstance(all_losses, list):
            all_losses = [all_losses]
        
        # 选择损失最大的样本
        num_hard = max(1, int(len(all_losses) * self.ratio))
        sorted_losses = sorted(all_losses, reverse=True)
        hard_losses = sorted_losses[:num_hard]
        
        return self._reduce(hard_losses)


class AsymmetricLoss(Loss):
    """
    非对称损失 - 用于多标签分类
    
    L_pos = (1 - p)^gamma_pos * log(p)
    L_neg = p^gamma_neg * log(1 - p)
    L = -L_pos - L_neg
    """
    
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        epsilon: float = 1e-8,
        reduction: Reduction = Reduction.MEAN
    ):
        super().__init__(reduction)
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.epsilon = epsilon
    
    def forward(self, predictions, targets) -> float:
        """计算非对称损失"""
        if isinstance(targets, (int, float)):
            targets = [targets]
        
        if isinstance(predictions, (int, float)):
            predictions = [predictions]
        
        if not isinstance(predictions[0], list):
            predictions = [predictions]
            targets = [targets]
        
        losses = []
        
        for i, logits in enumerate(predictions):
            target = targets[i] if i < len(targets) else targets[0]
            
            # 计算sigmoid概率
            probs = [1.0 / (1.0 + math.exp(-l)) for l in logits]
            
            # 非对称概率裁剪
            if self.clip is not None and self.clip > 0:
                probs = [max(self.clip, min(1 - self.clip, p)) for p in probs]
            
            # 计算损失
            loss = 0.0
            if isinstance(target, list):
                for j, (p, t) in enumerate(zip(probs, target)):
                    if t == 1:
                        loss -= (1 - p) ** self.gamma_pos * math.log(p + self.epsilon)
                    else:
                        loss -= p ** self.gamma_neg * math.log(1 - p + self.epsilon)
            else:
                for j, p in enumerate(probs):
                    if j == target:
                        loss -= (1 - p) ** self.gamma_pos * math.log(p + self.epsilon)
                    else:
                        loss -= p ** self.gamma_neg * math.log(1 - p + self.epsilon)
            
            losses.append(loss)
        
        return self._reduce(losses)


class BCEWithLogitsLoss(Loss):
    """
    带Logits的二分类交叉熵损失
    
    L = -[y * log(sigmoid(x)) + (1-y) * log(1-sigmoid(x))]
    """
    
    def __init__(
        self,
        weight: Optional[List[float]] = None,
        pos_weight: Optional[float] = None,
        reduction: Reduction = Reduction.MEAN
    ):
        super().__init__(reduction)
        self.weight = weight
        self.pos_weight = pos_weight
    
    def forward(self, predictions, targets) -> float:
        """计算BCE损失"""
        if isinstance(predictions, (int, float)):
            predictions = [predictions]
        if isinstance(targets, (int, float)):
            targets = [float(targets)]
        
        if not isinstance(predictions[0], list):
            predictions = [predictions]
            targets = [targets]
        
        losses = []
        
        for i, logits in enumerate(predictions):
            target = targets[i] if i < len(targets) else targets[0]
            
            if isinstance(logits, list):
                for j, (logit, t) in enumerate(zip(logits, target if isinstance(target, list) else [target] * len(logits))):
                    # 数值稳定的计算
                    if logit >= 0:
                        loss = logit + math.log(1 + math.exp(-logit)) - t * logit
                    else:
                        loss = -t * logit + math.log(1 + math.exp(logit))
                    
                    if self.pos_weight is not None and t == 1:
                        loss *= self.pos_weight
                    
                    if self.weight is not None and j < len(self.weight):
                        loss *= self.weight[j]
                    
                    losses.append(loss)
            else:
                logit = logits
                t = float(target)
                
                if logit >= 0:
                    loss = logit + math.log(1 + math.exp(-logit)) - t * logit
                else:
                    loss = -t * logit + math.log(1 + math.exp(logit))
                
                losses.append(loss)
        
        return self._reduce(losses)


class MultiLabelSoftMarginLoss(Loss):
    """
    多标签软边界损失
    """
    
    def __init__(
        self,
        weight: Optional[List[float]] = None,
        reduction: Reduction = Reduction.MEAN
    ):
        super().__init__(reduction)
        self.weight = weight
    
    def forward(self, predictions, targets) -> float:
        """计算多标签软边界损失"""
        if isinstance(predictions, (int, float)):
            predictions = [predictions]
        if isinstance(targets, (int, float)):
            targets = [float(targets)]
        
        if not isinstance(predictions[0], list):
            predictions = [predictions]
            targets = [targets]
        
        losses = []
        
        for logits, target in zip(predictions, targets):
            if isinstance(logits, list) and isinstance(target, list):
                for j, (logit, t) in enumerate(zip(logits, target)):
                    # log(sigmoid(x)) = -log(1 + exp(-x))
                    # log(1-sigmoid(x)) = -log(1 + exp(x))
                    loss = -t * math.log(1 + math.exp(-logit)) - (1 - t) * math.log(1 + math.exp(logit))
                    
                    if self.weight is not None and j < len(self.weight):
                        loss *= self.weight[j]
                    
                    losses.append(loss)
            else:
                logit = logits
                t = float(target)
                loss = -t * math.log(1 + math.exp(-logit)) - (1 - t) * math.log(1 + math.exp(logit))
                losses.append(loss)
        
        return self._reduce(losses)


# 损失函数工厂
def get_loss(name: str, **kwargs) -> Loss:
    """根据名称获取损失函数"""
    losses = {
        'cross_entropy': CrossEntropyLoss,
        'focal': FocalLoss,
        'binary_focal': BinaryFocalLoss,
        'contrastive': ContrastiveLoss,
        'triplet': TripletLoss,
        'infonce': InfoNCELoss,
        'ntxent': NTXentLoss,
        'dice': DiceLoss,
        'iou': IoULoss,
        'tversky': TverskyLoss,
        'focal_tversky': FocalTverskyLoss,
        'arcface': ArcFaceLoss,
        'center': CenterLoss,
        'cosface': CosFaceLoss,
        'sphereface': SphereFaceLoss,
        'label_smoothing': LabelSmoothingCrossEntropy,
        'soft_target': SoftTargetCrossEntropy,
        'kl': KLDivergenceLoss,
        'js': JSLoss,
        'ghmc': GHMCLoss,
        'ohem': OHEMLoss,
        'asymmetric': AsymmetricLoss,
        'bce': BCEWithLogitsLoss,
        'multilabel_soft_margin': MultiLabelSoftMarginLoss
    }
    
    name_lower = name.lower()
    if name_lower not in losses:
        raise ValueError(f"Unknown loss: {name}. Available: {list(losses.keys())}")
    
    return losses[name_lower](**kwargs)


class CombinedLoss(Loss):
    """组合多个损失函数"""
    
    def __init__(self, losses: List[Tuple[Loss, float]], reduction: Reduction = Reduction.MEAN):
        """
        losses: [(loss_fn, weight), ...]
        """
        super().__init__(reduction)
        self.losses = losses
    
    def forward(self, predictions, targets) -> float:
        """计算组合损失"""
        total_loss = 0.0
        for loss_fn, weight in self.losses:
            total_loss += weight * loss_fn(predictions, targets)
        return total_loss
