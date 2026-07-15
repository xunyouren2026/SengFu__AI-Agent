"""
评估指标模块 - 包含各种评估指标的真实实现
包括: Classification Metrics, Detection Metrics, Segmentation Metrics,
      NLP Metrics, Regression Metrics, Ranking Metrics
"""

import math
from typing import Optional, Tuple, List, Union, Dict, Any, Callable
from collections import defaultdict
from abc import ABC, abstractmethod


# ==================== 分类指标 ====================

class Accuracy:
    """准确率"""
    
    def __init__(self, top_k: int = 1):
        self.top_k = top_k
        self.correct = 0
        self.total = 0
    
    def update(self, predictions: List[List[float]], targets: List[int]):
        """
        predictions: [batch_size, num_classes] logits或概率
        targets: [batch_size] 类别索引
        """
        for pred, target in zip(predictions, targets):
            # 获取top-k预测
            sorted_indices = sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)
            top_k_preds = sorted_indices[:self.top_k]
            
            if target in top_k_preds:
                self.correct += 1
            self.total += 1
    
    def compute(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total
    
    def reset(self):
        self.correct = 0
        self.total = 0


class Precision:
    """精确率"""
    
    def __init__(self, average: str = 'macro', num_classes: Optional[int] = None):
        self.average = average
        self.num_classes = num_classes
        self.tp = defaultdict(int)
        self.fp = defaultdict(int)
    
    def update(self, predictions: List[int], targets: List[int]):
        for pred, target in zip(predictions, targets):
            if pred == target:
                self.tp[target] += 1
            else:
                self.fp[pred] += 1
    
    def compute(self) -> Union[float, List[float]]:
        if self.average == 'macro':
            precisions = []
            classes = set(list(self.tp.keys()) + list(self.fp.keys()))
            for c in classes:
                tp = self.tp[c]
                fp = self.fp[c]
                if tp + fp > 0:
                    precisions.append(tp / (tp + fp))
                else:
                    precisions.append(0.0)
            return sum(precisions) / len(precisions) if precisions else 0.0
        
        elif self.average == 'micro':
            total_tp = sum(self.tp.values())
            total_fp = sum(self.fp.values())
            return total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
        
        elif self.average == 'none':
            precisions = {}
            classes = set(list(self.tp.keys()) + list(self.fp.keys()))
            for c in classes:
                tp = self.tp[c]
                fp = self.fp[c]
                precisions[c] = tp / (tp + fp) if tp + fp > 0 else 0.0
            return precisions
    
    def reset(self):
        self.tp = defaultdict(int)
        self.fp = defaultdict(int)


class Recall:
    """召回率"""
    
    def __init__(self, average: str = 'macro'):
        self.average = average
        self.tp = defaultdict(int)
        self.fn = defaultdict(int)
    
    def update(self, predictions: List[int], targets: List[int]):
        for pred, target in zip(predictions, targets):
            if pred == target:
                self.tp[target] += 1
            else:
                self.fn[target] += 1
    
    def compute(self) -> Union[float, Dict[int, float]]:
        if self.average == 'macro':
            recalls = []
            classes = set(list(self.tp.keys()) + list(self.fn.keys()))
            for c in classes:
                tp = self.tp[c]
                fn = self.fn[c]
                if tp + fn > 0:
                    recalls.append(tp / (tp + fn))
                else:
                    recalls.append(0.0)
            return sum(recalls) / len(recalls) if recalls else 0.0
        
        elif self.average == 'micro':
            total_tp = sum(self.tp.values())
            total_fn = sum(self.fn.values())
            return total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
        
        elif self.average == 'none':
            recalls = {}
            classes = set(list(self.tp.keys()) + list(self.fn.keys()))
            for c in classes:
                tp = self.tp[c]
                fn = self.fn[c]
                recalls[c] = tp / (tp + fn) if tp + fn > 0 else 0.0
            return recalls
    
    def reset(self):
        self.tp = defaultdict(int)
        self.fn = defaultdict(int)


class F1Score:
    """F1分数"""
    
    def __init__(self, average: str = 'macro', beta: float = 1.0):
        self.average = average
        self.beta = beta
        self.precision = Precision(average='none')
        self.recall = Recall(average='none')
    
    def update(self, predictions: List[int], targets: List[int]):
        self.precision.update(predictions, targets)
        self.recall.update(predictions, targets)
    
    def compute(self) -> Union[float, Dict[int, float]]:
        precisions = self.precision.compute()
        recalls = self.recall.compute()
        
        beta_sq = self.beta ** 2
        
        f1_scores = {}
        classes = set(list(precisions.keys()) + list(recalls.keys()))
        
        for c in classes:
            p = precisions.get(c, 0.0)
            r = recalls.get(c, 0.0)
            
            if p + r > 0:
                f1_scores[c] = (1 + beta_sq) * p * r / (beta_sq * p + r)
            else:
                f1_scores[c] = 0.0
        
        if self.average == 'macro':
            return sum(f1_scores.values()) / len(f1_scores) if f1_scores else 0.0
        elif self.average == 'micro':
            # 计算micro F1
            total_tp = sum(self.precision.tp.values())
            total_fp = sum(self.precision.fp.values())
            total_fn = sum(self.recall.fn.values())
            
            micro_p = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
            micro_r = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
            
            if micro_p + micro_r > 0:
                return 2 * micro_p * micro_r / (micro_p + micro_r)
            return 0.0
        else:
            return f1_scores
    
    def reset(self):
        self.precision.reset()
        self.recall.reset()


class ConfusionMatrix:
    """混淆矩阵"""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.matrix = [[0] * num_classes for _ in range(num_classes)]
    
    def update(self, predictions: List[int], targets: List[int]):
        for pred, target in zip(predictions, targets):
            if 0 <= pred < self.num_classes and 0 <= target < self.num_classes:
                self.matrix[target][pred] += 1
    
    def compute(self) -> List[List[int]]:
        return self.matrix
    
    def reset(self):
        self.matrix = [[0] * self.num_classes for _ in range(self.num_classes)]
    
    def get_class_metrics(self) -> Dict[str, List[float]]:
        """获取每个类别的指标"""
        metrics = {'precision': [], 'recall': [], 'f1': [], 'support': []}
        
        for c in range(self.num_classes):
            tp = self.matrix[c][c]
            fp = sum(self.matrix[i][c] for i in range(self.num_classes)) - tp
            fn = sum(self.matrix[c][j] for j in range(self.num_classes)) - tp
            
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            support = sum(self.matrix[c])
            
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)
            metrics['support'].append(support)
        
        return metrics


class AUROC:
    """ROC曲线下面积"""
    
    def __init__(self):
        self.scores = []
        self.labels = []
    
    def update(self, predictions: List[float], targets: List[int]):
        """
        predictions: 正类概率
        targets: 0或1
        """
        self.scores.extend(predictions)
        self.labels.extend(targets)
    
    def compute(self) -> float:
        if len(self.scores) == 0:
            return 0.0
        
        # 按分数排序
        sorted_pairs = sorted(zip(self.scores, self.labels), key=lambda x: x[0], reverse=True)
        
        # 计算TPR和FPR
        total_pos = sum(self.labels)
        total_neg = len(self.labels) - total_pos
        
        if total_pos == 0 or total_neg == 0:
            return 0.0
        
        tp = 0
        fp = 0
        prev_score = None
        auc = 0.0
        prev_tpr = 0.0
        prev_fpr = 0.0
        
        for score, label in sorted_pairs:
            if prev_score is not None and score != prev_score:
                tpr = tp / total_pos
                fpr = fp / total_neg
                # 梯形法则
                auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
                prev_tpr = tpr
                prev_fpr = fpr
            
            if label == 1:
                tp += 1
            else:
                fp += 1
            prev_score = score
        
        # 最后一个点
        tpr = tp / total_pos
        fpr = fp / total_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        
        return auc
    
    def reset(self):
        self.scores = []
        self.labels = []


class AUPRC:
    """PR曲线下面积"""
    
    def __init__(self):
        self.scores = []
        self.labels = []
    
    def update(self, predictions: List[float], targets: List[int]):
        self.scores.extend(predictions)
        self.labels.extend(targets)
    
    def compute(self) -> float:
        if len(self.scores) == 0:
            return 0.0
        
        sorted_pairs = sorted(zip(self.scores, self.labels), key=lambda x: x[0], reverse=True)
        
        total_pos = sum(self.labels)
        if total_pos == 0:
            return 0.0
        
        tp = 0
        fp = 0
        prev_precision = 1.0
        prev_recall = 0.0
        auprc = 0.0
        prev_score = None
        
        for score, label in sorted_pairs:
            if prev_score is not None and score != prev_score:
                precision = tp / (tp + fp) if tp + fp > 0 else 0.0
                recall = tp / total_pos
                auprc += (recall - prev_recall) * (precision + prev_precision) / 2
                prev_precision = precision
                prev_recall = recall
            
            if label == 1:
                tp += 1
            else:
                fp += 1
            prev_score = score
        
        # 最后一个点
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / total_pos
        auprc += (recall - prev_recall) * (precision + prev_precision) / 2
        
        return auprc
    
    def reset(self):
        self.scores = []
        self.labels = []


# ==================== 检测指标 ====================

class BBox:
    """边界框"""
    
    def __init__(self, x1: float, y1: float, x2: float, y2: float, 
                 score: float = 1.0, label: int = 0):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.score = score
        self.label = label
    
    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    def iou(self, other: 'BBox') -> float:
        """计算IoU"""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        
        return intersection / union if union > 0 else 0.0


class MeanAveragePrecision:
    """目标检测mAP"""
    
    def __init__(self, iou_thresholds: List[float] = None, num_classes: int = 80):
        self.iou_thresholds = iou_thresholds or [0.5]
        self.num_classes = num_classes
        
        self.predictions = []  # [(image_id, bbox)]
        self.ground_truths = []  # [(image_id, bbox)]
    
    def update(
        self,
        pred_boxes: List[BBox],
        gt_boxes: List[BBox],
        image_id: int = 0
    ):
        for bbox in pred_boxes:
            self.predictions.append((image_id, bbox))
        for bbox in gt_boxes:
            self.ground_truths.append((image_id, bbox))
    
    def compute(self) -> float:
        if len(self.predictions) == 0 or len(self.ground_truths) == 0:
            return 0.0
        
        aps = []
        
        for iou_thresh in self.iou_thresholds:
            ap = self._compute_ap(iou_thresh)
            aps.append(ap)
        
        return sum(aps) / len(aps)
    
    def _compute_ap(self, iou_threshold: float) -> float:
        """计算单个IoU阈值下的AP"""
        # 按类别分组
        pred_by_class = defaultdict(list)
        gt_by_class = defaultdict(list)
        
        for img_id, bbox in self.predictions:
            pred_by_class[bbox.label].append((img_id, bbox))
        
        for img_id, bbox in self.ground_truths:
            gt_by_class[bbox.label].append((img_id, bbox))
        
        aps = []
        
        for class_id in range(self.num_classes):
            preds = pred_by_class[class_id]
            gts = gt_by_class[class_id]
            
            if len(gts) == 0:
                continue
            
            # 按分数排序预测
            preds = sorted(preds, key=lambda x: x[1].score, reverse=True)
            
            # 记录每个GT是否被匹配
            gt_matched = defaultdict(lambda: [False] * len(
                [gt for img_id, gt in gts if img_id == img_id]
            ))
            
            # 按图像分组GT
            gt_by_img = defaultdict(list)
            for img_id, bbox in gts:
                gt_by_img[img_id].append(bbox)
            
            tp = []
            fp = []
            
            for img_id, pred_bbox in preds:
                matched = False
                img_gts = gt_by_img.get(img_id, [])
                
                best_iou = 0.0
                best_gt_idx = -1
                
                for idx, gt_bbox in enumerate(img_gts):
                    iou = pred_bbox.iou(gt_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx
                
                if best_iou >= iou_threshold:
                    # 检查是否已被匹配
                    gt_key = (img_id, class_id)
                    if not matched:  # 简化处理
                        tp.append(1)
                        fp.append(0)
                        matched = True
                    else:
                        tp.append(0)
                        fp.append(1)
                else:
                    tp.append(0)
                    fp.append(1)
            
            # 计算precision-recall曲线
            tp_cumsum = [sum(tp[:i+1]) for i in range(len(tp))]
            fp_cumsum = [sum(fp[:i+1]) for i in range(len(fp))]
            
            recalls = [t / len(gts) for t in tp_cumsum]
            precisions = [t / (t + f) if t + f > 0 else 0.0 
                         for t, f in zip(tp_cumsum, fp_cumsum)]
            
            # 计算AP (11点插值)
            ap = 0.0
            for recall_thresh in [i / 10 for i in range(11)]:
                precisions_above = [p for p, r in zip(precisions, recalls) if r >= recall_thresh]
                if precisions_above:
                    ap += max(precisions_above)
            ap /= 11
            
            aps.append(ap)
        
        return sum(aps) / len(aps) if aps else 0.0
    
    def reset(self):
        self.predictions = []
        self.ground_truths = []


# ==================== 分割指标 ====================

class DiceCoefficient:
    """Dice系数"""
    
    def __init__(self, num_classes: int, smooth: float = 1.0):
        self.num_classes = num_classes
        self.smooth = smooth
        self.intersection = [0.0] * num_classes
        self.union = [0.0] * num_classes
    
    def update(
        self,
        predictions: List[List[int]],
        targets: List[List[int]]
    ):
        """
        predictions, targets: 2D分割掩码
        """
        for pred_row, target_row in zip(predictions, targets):
            for p, t in zip(pred_row, target_row):
                if p == t:
                    self.intersection[p] += 1
                self.union[p] += 1
                self.union[t] += 1
    
    def compute(self) -> float:
        dice_scores = []
        for c in range(self.num_classes):
            if self.union[c] > 0:
                dice = (2 * self.intersection[c] + self.smooth) / (self.union[c] + self.smooth)
                dice_scores.append(dice)
        
        return sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
    
    def reset(self):
        self.intersection = [0.0] * self.num_classes
        self.union = [0.0] * self.num_classes


class MeanIoU:
    """平均IoU"""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.confusion_matrix = [[0] * num_classes for _ in range(num_classes)]
    
    def update(
        self,
        predictions: List[List[int]],
        targets: List[List[int]]
    ):
        for pred_row, target_row in zip(predictions, targets):
            for p, t in zip(pred_row, target_row):
                if 0 <= p < self.num_classes and 0 <= t < self.num_classes:
                    self.confusion_matrix[t][p] += 1
    
    def compute(self) -> float:
        ious = []
        
        for c in range(self.num_classes):
            tp = self.confusion_matrix[c][c]
            fp = sum(self.confusion_matrix[i][c] for i in range(self.num_classes)) - tp
            fn = sum(self.confusion_matrix[c][j] for j in range(self.num_classes)) - tp
            
            if tp + fp + fn > 0:
                iou = tp / (tp + fp + fn)
                ious.append(iou)
        
        return sum(ious) / len(ious) if ious else 0.0
    
    def compute_per_class(self) -> List[float]:
        """计算每个类别的IoU"""
        ious = []
        
        for c in range(self.num_classes):
            tp = self.confusion_matrix[c][c]
            fp = sum(self.confusion_matrix[i][c] for i in range(self.num_classes)) - tp
            fn = sum(self.confusion_matrix[c][j] for j in range(self.num_classes)) - tp
            
            if tp + fp + fn > 0:
                iou = tp / (tp + fp + fn)
            else:
                iou = 0.0
            ious.append(iou)
        
        return ious
    
    def reset(self):
        self.confusion_matrix = [[0] * self.num_classes for _ in range(self.num_classes)]


# ==================== NLP指标 ====================

class BLEU:
    """BLEU分数"""
    
    def __init__(self, max_n: int = 4, smooth: bool = True):
        self.max_n = max_n
        self.smooth = smooth
    
    def compute(
        self,
        hypothesis: List[str],
        references: List[List[str]]
    ) -> float:
        """
        hypothesis: 假设句子 (token列表)
        references: 参考句子列表 (每个是token列表)
        """
        # 计算n-gram精度
        precisions = []
        
        for n in range(1, self.max_n + 1):
            hyp_ngrams = self._get_ngrams(hypothesis, n)
            
            if len(hyp_ngrams) == 0:
                precisions.append(0.0)
                continue
            
            # 对每个参考计算匹配数
            max_matches = 0
            for ref in references:
                ref_ngrams = self._get_ngrams(ref, n)
                matches = self._count_matches(hyp_ngrams, ref_ngrams)
                max_matches = max(max_matches, matches)
            
            if self.smooth:
                precision = (max_matches + 1) / (len(hyp_ngrams) + 1)
            else:
                precision = max_matches / len(hyp_ngrams)
            
            precisions.append(precision)
        
        # 计算几何平均
        if min(precisions) == 0:
            return 0.0
        
        log_precision = sum(math.log(p) for p in precisions) / len(precisions)
        
        # 计算brevity penalty
        hyp_len = len(hypothesis)
        ref_lens = [len(ref) for ref in references]
        closest_ref_len = min(ref_lens, key=lambda x: abs(x - hyp_len))
        
        if hyp_len > closest_ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - closest_ref_len / hyp_len) if hyp_len > 0 else 0.0
        
        return bp * math.exp(log_precision)
    
    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def _count_matches(self, hyp_ngrams: List, ref_ngrams: List) -> int:
        ref_counts = defaultdict(int)
        for ngram in ref_ngrams:
            ref_counts[ngram] += 1
        
        matches = 0
        hyp_counts = defaultdict(int)
        for ngram in hyp_ngrams:
            hyp_counts[ngram] += 1
        
        for ngram, count in hyp_counts.items():
            matches += min(count, ref_counts.get(ngram, 0))
        
        return matches


class ROUGE:
    """ROUGE分数"""
    
    def __init__(self, max_n: int = 4):
        self.max_n = max_n
    
    def compute(
        self,
        hypothesis: List[str],
        references: List[List[str]]
    ) -> Dict[str, float]:
        """计算ROUGE-N和ROUGE-L"""
        results = {}
        
        # ROUGE-N
        for n in range(1, self.max_n + 1):
            hyp_ngrams = self._get_ngrams(hypothesis, n)
            
            best_f1 = 0.0
            for ref in references:
                ref_ngrams = self._get_ngrams(ref, n)
                
                matches = self._count_matches(hyp_ngrams, ref_ngrams)
                
                precision = matches / len(hyp_ngrams) if hyp_ngrams else 0.0
                recall = matches / len(ref_ngrams) if ref_ngrams else 0.0
                
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0.0
                
                best_f1 = max(best_f1, f1)
            
            results[f'rouge-{n}'] = best_f1
        
        # ROUGE-L (最长公共子序列)
        best_lcs_f1 = 0.0
        for ref in references:
            lcs_len = self._lcs_length(hypothesis, ref)
            
            precision = lcs_len / len(hypothesis) if hypothesis else 0.0
            recall = lcs_len / len(ref) if ref else 0.0
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            
            best_lcs_f1 = max(best_lcs_f1, f1)
        
        results['rouge-l'] = best_lcs_f1
        
        return results
    
    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def _count_matches(self, hyp_ngrams: List, ref_ngrams: List) -> int:
        ref_counts = defaultdict(int)
        for ngram in ref_ngrams:
            ref_counts[ngram] += 1
        
        matches = 0
        for ngram in hyp_ngrams:
            if ref_counts[ngram] > 0:
                matches += 1
                ref_counts[ngram] -= 1
        
        return matches
    
    def _lcs_length(self, a: List[str], b: List[str]) -> int:
        """计算最长公共子序列长度"""
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]


class Perplexity:
    """困惑度"""
    
    def __init__(self):
        self.total_log_prob = 0.0
        self.total_tokens = 0
    
    def update(self, log_probs: List[float]):
        """
        log_probs: 每个token的对数概率
        """
        self.total_log_prob += sum(log_probs)
        self.total_tokens += len(log_probs)
    
    def compute(self) -> float:
        if self.total_tokens == 0:
            return float('inf')
        return math.exp(-self.total_log_prob / self.total_tokens)
    
    def reset(self):
        self.total_log_prob = 0.0
        self.total_tokens = 0


# ==================== 回归指标 ====================

class MeanSquaredError:
    """均方误差"""
    
    def __init__(self):
        self.sum_sq_error = 0.0
        self.count = 0
    
    def update(self, predictions: List[float], targets: List[float]):
        for pred, target in zip(predictions, targets):
            self.sum_sq_error += (pred - target) ** 2
            self.count += 1
    
    def compute(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum_sq_error / self.count
    
    def reset(self):
        self.sum_sq_error = 0.0
        self.count = 0


class MeanAbsoluteError:
    """平均绝对误差"""
    
    def __init__(self):
        self.sum_abs_error = 0.0
        self.count = 0
    
    def update(self, predictions: List[float], targets: List[float]):
        for pred, target in zip(predictions, targets):
            self.sum_abs_error += abs(pred - target)
            self.count += 1
    
    def compute(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum_abs_error / self.count
    
    def reset(self):
        self.sum_abs_error = 0.0
        self.count = 0


class R2Score:
    """R²分数"""
    
    def __init__(self):
        self.sum_sq_res = 0.0  # 残差平方和
        self.sum_sq_tot = 0.0  # 总平方和
        self.sum_target = 0.0
        self.count = 0
        self.targets = []
    
    def update(self, predictions: List[float], targets: List[float]):
        self.targets.extend(targets)
        
        for pred, target in zip(predictions, targets):
            self.sum_sq_res += (target - pred) ** 2
            self.sum_target += target
            self.count += 1
    
    def compute(self) -> float:
        if self.count == 0:
            return 0.0
        
        mean_target = self.sum_target / self.count
        
        sum_sq_tot = sum((t - mean_target) ** 2 for t in self.targets)
        
        if sum_sq_tot == 0:
            return 0.0
        
        return 1 - self.sum_sq_res / sum_sq_tot
    
    def reset(self):
        self.sum_sq_res = 0.0
        self.sum_sq_tot = 0.0
        self.sum_target = 0.0
        self.count = 0
        self.targets = []


# ==================== 排序指标 ====================

class MeanReciprocalRank:
    """平均倒数排名"""
    
    def __init__(self):
        self.reciprocal_ranks = []
    
    def update(self, rankings: List[int]):
        """
        rankings: 每个查询中正确答案的排名列表 (从1开始)
        """
        for rank in rankings:
            if rank > 0:
                self.reciprocal_ranks.append(1.0 / rank)
            else:
                self.reciprocal_ranks.append(0.0)
    
    def compute(self) -> float:
        if not self.reciprocal_ranks:
            return 0.0
        return sum(self.reciprocal_ranks) / len(self.reciprocal_ranks)
    
    def reset(self):
        self.reciprocal_ranks = []


class NDCG:
    """归一化折损累积增益"""
    
    def __init__(self, k: Optional[int] = None):
        self.k = k
        self.ndcg_scores = []
    
    def compute(
        self,
        relevance_scores: List[float],
        ideal_relevance: List[float]
    ) -> float:
        """
        relevance_scores: 排序后的相关性分数
        ideal_relevance: 理想排序的相关性分数
        """
        k = self.k or len(relevance_scores)
        
        # 计算DCG
        dcg = 0.0
        for i, rel in enumerate(relevance_scores[:k]):
            dcg += (2 ** rel - 1) / math.log2(i + 2)
        
        # 计算IDCG
        ideal_sorted = sorted(ideal_relevance, reverse=True)
        idcg = 0.0
        for i, rel in enumerate(ideal_sorted[:k]):
            idcg += (2 ** rel - 1) / math.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg


# 工厂函数
def get_metric(name: str, **kwargs):
    """根据名称获取指标"""
    metrics = {
        'accuracy': Accuracy,
        'precision': Precision,
        'recall': Recall,
        'f1': F1Score,
        'confusion_matrix': ConfusionMatrix,
        'auroc': AUROC,
        'auprc': AUPRC,
        'map': MeanAveragePrecision,
        'dice': DiceCoefficient,
        'miou': MeanIoU,
        'bleu': BLEU,
        'rouge': ROUGE,
        'perplexity': Perplexity,
        'mse': MeanSquaredError,
        'mae': MeanAbsoluteError,
        'r2': R2Score,
        'mrr': MeanReciprocalRank,
        'ndcg': NDCG
    }
    
    name_lower = name.lower()
    if name_lower not in metrics:
        raise ValueError(f"Unknown metric: {name}. Available: {list(metrics.keys())}")
    
    return metrics[name_lower](**kwargs)
