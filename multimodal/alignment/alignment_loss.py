"""
对齐损失函数
实现多种模态对齐损失
"""
from typing import Optional, List, Dict, Any, Tuple
import math


class AlignmentLoss:
    """对齐损失基类"""
    
    def __init__(self):
        pass
    
    def __call__(self, features_a: List[List[float]], 
                 features_b: List[List[float]]) -> Tuple[float, Dict[str, float]]:
        """计算对齐损失
        
        默认实现：使用余弦相似度计算对比损失（InfoNCE风格）。
        
        Args:
            features_a: 第一组特征列表，每个元素是一个特征向量
            features_b: 第二组特征列表，每个元素是一个特征向量
            
        Returns:
            (损失值, 指标字典) 元组
        """
        n = len(features_a)
        if n == 0:
            return 0.0, {"alignment_loss": 0.0}
        
        # 归一化
        def _normalize(x: List[float]) -> List[float]:
            norm = math.sqrt(sum(xi ** 2 for xi in x))
            return [xi / norm for xi in x] if norm > 0 else x
        
        features_a = [_normalize(f) for f in features_a]
        features_b = [_normalize(f) for f in features_b]
        
        # 计算对比损失
        temperature = 0.07
        total_loss = 0.0
        
        for i in range(n):
            # 正样本相似度
            pos_sim = sum(a * b for a, b in zip(features_a[i], features_b[i])) / temperature
            
            # 所有相似度
            all_sims = [sum(a * b for a, b in zip(features_a[i], features_b[j])) / temperature 
                       for j in range(n)]
            
            # softmax
            max_sim = max(all_sims)
            exp_sims = [math.exp(s - max_sim) for s in all_sims]
            sum_exp = sum(exp_sims)
            probs = [e / sum_exp for e in exp_sims]
            
            total_loss -= math.log(probs[i] + 1e-10)
        
        avg_loss = total_loss / n
        return avg_loss, {"alignment_loss": avg_loss}


class ContrastiveLoss(AlignmentLoss):
    """对比损失 (InfoNCE)"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def _normalize(self, x: List[float]) -> List[float]:
        norm = math.sqrt(sum(xi ** 2 for xi in x))
        return [xi / norm for xi in x] if norm > 0 else x
    
    def _softmax(self, x: List[float]) -> List[float]:
        max_val = max(x)
        exp_vals = [math.exp(v - max_val) for v in x]
        sum_exp = sum(exp_vals)
        return [e / sum_exp for e in exp_vals]
    
    def __call__(self, features_a: List[List[float]], 
                 features_b: List[List[float]]) -> Tuple[float, Dict[str, float]]:
        n = len(features_a)
        if n == 0:
            return 0.0, {}
        
        # 归一化
        features_a = [self._normalize(f) for f in features_a]
        features_b = [self._normalize(f) for f in features_b]
        
        # 计算相似度矩阵
        sim_matrix = [[sum(a * b for a, b in zip(fa, fb)) / self.temperature 
                       for fb in features_b] for fa in features_a]
        
        # 计算损失
        loss = 0.0
        for i in range(n):
            # A -> B
            probs = self._softmax(sim_matrix[i])
            loss -= math.log(probs[i] + 1e-10)
            
            # B -> A
            col = [sim_matrix[j][i] for j in range(n)]
            probs = self._softmax(col)
            loss -= math.log(probs[i] + 1e-10)
        
        loss /= (2 * n)
        
        return loss, {'contrastive_loss': loss}


class SigmoidLoss(AlignmentLoss):
    """Sigmoid损失 (SigLIP)"""
    
    def __init__(self, temperature: float = 0.1, bias: float = -10.0):
        super().__init__()
        self.temperature = temperature
        self.bias = bias
    
    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-10, min(10, x))))
    
    def _normalize(self, x: List[float]) -> List[float]:
        norm = math.sqrt(sum(xi ** 2 for xi in x))
        return [xi / norm for xi in x] if norm > 0 else x
    
    def __call__(self, features_a: List[List[float]], 
                 features_b: List[List[float]]) -> Tuple[float, Dict[str, float]]:
        n = len(features_a)
        if n == 0:
            return 0.0, {}
        
        features_a = [self._normalize(f) for f in features_a]
        features_b = [self._normalize(f) for f in features_b]
        
        total_loss = 0.0
        
        for i in range(n):
            for j in range(n):
                sim = sum(a * b for a, b in zip(features_a[i], features_b[j]))
                scaled_sim = sim / self.temperature + self.bias
                
                target = 1 if i == j else -1
                
                if target == 1:
                    prob = self._sigmoid(scaled_sim)
                    total_loss -= math.log(prob + 1e-10)
                else:
                    prob = self._sigmoid(scaled_sim)
                    total_loss -= math.log(1.0 - prob + 1e-10)
        
        avg_loss = total_loss / (n * n)
        
        return avg_loss, {'sigmoid_loss': avg_loss}


class TripletLoss(AlignmentLoss):
    """三元组损失"""
    
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
    
    def _normalize(self, x: List[float]) -> List[float]:
        norm = math.sqrt(sum(xi ** 2 for xi in x))
        return [xi / norm for xi in x] if norm > 0 else x
    
    def __call__(self, features_a: List[List[float]], 
                 features_b: List[List[float]]) -> Tuple[float, Dict[str, float]]:
        n = len(features_a)
        if n < 2:
            return 0.0, {}
        
        features_a = [self._normalize(f) for f in features_a]
        features_b = [self._normalize(f) for f in features_b]
        
        total_loss = 0.0
        count = 0
        
        for i in range(n):
            anchor = features_a[i]
            positive = features_b[i]
            
            # 计算正样本距离
            pos_dist = 1.0 - sum(a * b for a, b in zip(anchor, positive))
            
            # 找到最难负样本
            neg_dists = []
            for j in range(n):
                if j != i:
                    neg_dist = 1.0 - sum(a * b for a, b in zip(anchor, features_b[j]))
                    neg_dists.append(neg_dist)
            
            if neg_dists:
                hardest_neg = min(neg_dists)
                loss = max(0.0, pos_dist - hardest_neg + self.margin)
                total_loss += loss
                count += 1
        
        avg_loss = total_loss / count if count > 0 else 0.0
        
        return avg_loss, {'triplet_loss': avg_loss}


class CosineEmbeddingLoss(AlignmentLoss):
    """余弦嵌入损失"""
    
    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin
    
    def _normalize(self, x: List[float]) -> List[float]:
        norm = math.sqrt(sum(xi ** 2 for xi in x))
        return [xi / norm for xi in x] if norm > 0 else x
    
    def __call__(self, features_a: List[List[float]], 
                 features_b: List[List[float]]) -> Tuple[float, Dict[str, float]]:
        n = len(features_a)
        if n == 0:
            return 0.0, {}
        
        features_a = [self._normalize(f) for f in features_a]
        features_b = [self._normalize(f) for f in features_b]
        
        total_loss = 0.0
        
        for i in range(n):
            cos_sim = sum(a * b for a, b in zip(features_a[i], features_b[i]))
            
            # 正样本: 1 - cos_sim
            # 负样本: max(0, cos_sim - margin)
            loss = 1.0 - cos_sim
            total_loss += loss
        
        avg_loss = total_loss / n
        
        return avg_loss, {'cosine_loss': avg_loss}


class MSELoss(AlignmentLoss):
    """均方误差损失"""
    
    def __call__(self, features_a: List[List[float]], 
                 features_b: List[List[float]]) -> Tuple[float, Dict[str, float]]:
        n = len(features_a)
        if n == 0:
            return 0.0, {}
        
        total_loss = 0.0
        
        for i in range(n):
            for j in range(len(features_a[i])):
                diff = features_a[i][j] - features_b[i][j]
                total_loss += diff ** 2
        
        avg_loss = total_loss / (n * len(features_a[0])) if n > 0 and features_a[0] else 0.0
        
        return avg_loss, {'mse_loss': avg_loss}


class CombinedAlignmentLoss(AlignmentLoss):
    """组合对齐损失"""
    
    def __init__(self, losses: List[Tuple[AlignmentLoss, float]]):
        super().__init__()
        self.losses = losses
    
    def __call__(self, features_a: List[List[float]], 
                 features_b: List[List[float]]) -> Tuple[float, Dict[str, float]]:
        total_loss = 0.0
        metrics = {}
        
        for loss_fn, weight in self.losses:
            loss, loss_metrics = loss_fn(features_a, features_b)
            total_loss += weight * loss
            
            for k, v in loss_metrics.items():
                metrics[k] = v
        
        metrics['total_loss'] = total_loss
        
        return total_loss, metrics


class MultiModalAlignmentLoss:
    """多模态对齐损失"""
    
    def __init__(self, modalities: List[str], loss_fn: Optional[AlignmentLoss] = None):
        self.modalities = modalities
        self.loss_fn = loss_fn or ContrastiveLoss()
        
        # 为每对模态创建损失
        self.pair_losses = {}
        for i, mod_a in enumerate(modalities):
            for mod_b in modalities[i + 1:]:
                key = f"{mod_a}_{mod_b}"
                self.pair_losses[key] = self.loss_fn
    
    def __call__(self, features: Dict[str, List[List[float]]]) -> Tuple[float, Dict[str, float]]:
        """计算多模态对齐损失"""
        total_loss = 0.0
        metrics = {}
        count = 0
        
        for key, loss_fn in self.pair_losses.items():
            mod_a, mod_b = key.split('_')
            
            if mod_a in features and mod_b in features:
                loss, pair_metrics = loss_fn(features[mod_a], features[mod_b])
                total_loss += loss
                count += 1
                
                for k, v in pair_metrics.items():
                    metrics[f"{key}_{k}"] = v
        
        if count > 0:
            total_loss /= count
        
        metrics['total_alignment_loss'] = total_loss
        
        return total_loss, metrics


def create_contrastive_loss(temperature: float = 0.07) -> ContrastiveLoss:
    """创建对比损失"""
    return ContrastiveLoss(temperature)


def create_sigmoid_loss(temperature: float = 0.1) -> SigmoidLoss:
    """创建Sigmoid损失"""
    return SigmoidLoss(temperature)


def create_triplet_loss(margin: float = 0.5) -> TripletLoss:
    """创建三元组损失"""
    return TripletLoss(margin)
