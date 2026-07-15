"""
SigLIP训练器
实现基于Sigmoid Loss的对比学习训练
"""
from typing import Optional, List, Dict, Any, Tuple
import math
import random


class SigLIPTrainer:
    """SigLIP训练器
    
    使用Sigmoid Loss进行对比学习，相比Softmax Loss有更好的训练稳定性
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        self.embed_dim = config.get('embed_dim', 512)
        self.batch_size = config.get('batch_size', 64)
        self.temperature = config.get('temperature', 0.1)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.weight_decay = config.get('weight_decay', 0.01)
        self.max_epochs = config.get('max_epochs', 100)
        self.bias_init = config.get('bias_init', -10.0)
        
        # 可学习的温度和偏置参数
        self.log_temperature = math.log(self.temperature)
        self.bias = self.bias_init
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid函数"""
        return 1.0 / (1.0 + math.exp(-max(-10, min(10, x))))
    
    def _normalize(self, x: List[float]) -> List[float]:
        """L2归一化"""
        norm = math.sqrt(sum(xi ** 2 for xi in x))
        if norm > 0:
            return [xi / norm for xi in x]
        return x
    
    def compute_sigmoid_loss(self, similarity: float, target: int) -> Tuple[float, float]:
        """
        计算Sigmoid损失
        
        Args:
            similarity: 相似度分数
            target: 目标标签 (1表示正样本，-1表示负样本)
        
        Returns:
            loss: 损失值
            grad: 梯度
        """
        temperature = math.exp(self.log_temperature)
        scaled_sim = similarity / temperature + self.bias
        
        if target == 1:
            # 正样本: log(sigmoid(scaled_sim))
            prob = self._sigmoid(scaled_sim)
            loss = -math.log(prob + 1e-10)
            grad = -(1.0 - prob) / temperature
        else:
            # 负样本: log(1 - sigmoid(scaled_sim))
            prob = self._sigmoid(scaled_sim)
            loss = -math.log(1.0 - prob + 1e-10)
            grad = prob / temperature
        
        return loss, grad
    
    def compute_similarity_matrix(self, features_a: List[List[float]], 
                                   features_b: List[List[float]]) -> List[List[float]]:
        """计算相似度矩阵"""
        features_a = [self._normalize(f) for f in features_a]
        features_b = [self._normalize(f) for f in features_b]
        
        sim_matrix = []
        for fa in features_a:
            row = []
            for fb in features_b:
                sim = sum(a * b for a, b in zip(fa, fb))
                row.append(sim)
            sim_matrix.append(row)
        
        return sim_matrix
    
    def compute_loss(self, features_a: List[List[float]], 
                     features_b: List[List[float]]) -> Tuple[float, Dict[str, float]]:
        """
        计算SigLIP损失
        
        Args:
            features_a: 模态A特征 [N, D]
            features_b: 模态B特征 [N, D]
        
        Returns:
            loss: 损失值
            metrics: 度量指标
        """
        n = len(features_a)
        if n == 0:
            return 0.0, {}
        
        # 计算相似度矩阵
        sim_matrix = self.compute_similarity_matrix(features_a, features_b)
        
        total_loss = 0.0
        correct_matches = 0
        total_pairs = 0
        
        for i in range(n):
            for j in range(n):
                # 目标: 对角线为正样本(1)，其他为负样本(-1)
                target = 1 if i == j else -1
                
                loss, _ = self.compute_sigmoid_loss(sim_matrix[i][j], target)
                total_loss += loss
                total_pairs += 1
                
                # 检查正确匹配
                if i == j and sim_matrix[i][j] > 0:
                    correct_matches += 1
        
        # 平均损失
        avg_loss = total_loss / total_pairs if total_pairs > 0 else 0.0
        
        # 度量指标
        metrics = {
            'loss': avg_loss,
            'accuracy': correct_matches / n if n > 0 else 0.0,
            'temperature': math.exp(self.log_temperature),
            'bias': self.bias
        }
        
        return avg_loss, metrics
    
    def compute_gradients(self, features_a: List[List[float]], 
                          features_b: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:
        """计算梯度"""
        n = len(features_a)
        d = len(features_a[0]) if features_a else 0
        
        features_a_norm = [self._normalize(f) for f in features_a]
        features_b_norm = [self._normalize(f) for f in features_b]
        
        sim_matrix = self.compute_similarity_matrix(features_a, features_b)
        
        grad_a = [[0.0] * d for _ in range(n)]
        grad_b = [[0.0] * d for _ in range(n)]
        
        temperature = math.exp(self.log_temperature)
        
        for i in range(n):
            for j in range(n):
                target = 1 if i == j else -1
                _, grad = self.compute_sigmoid_loss(sim_matrix[i][j], target)
                
                for k in range(d):
                    grad_a[i][k] += grad * features_b_norm[j][k]
                    grad_b[j][k] += grad * features_a_norm[i][k]
        
        return grad_a, grad_b
    
    def update_parameters(self, loss: float):
        """更新温度和偏置参数"""
        # 简单的参数调整
        if loss > 2.0:
            self.log_temperature -= 0.01
            self.bias -= 0.1
        elif loss < 0.5:
            self.log_temperature += 0.01
            self.bias += 0.1
        
        # 限制范围
        self.log_temperature = max(-5.0, min(2.0, self.log_temperature))
        self.bias = max(-20.0, min(0.0, self.bias))
    
    def train_step(self, features_a: List[List[float]], 
                   features_b: List[List[float]],
                   lr: Optional[float] = None) -> Tuple[float, Dict[str, float]]:
        """训练步骤"""
        lr = lr or self.learning_rate
        
        loss, metrics = self.compute_loss(features_a, features_b)
        grad_a, grad_b = self.compute_gradients(features_a, features_b)
        
        # 更新特征
        for i in range(len(features_a)):
            for j in range(len(features_a[i])):
                features_a[i][j] -= lr * grad_a[i][j]
        
        for i in range(len(features_b)):
            for j in range(len(features_b[i])):
                features_b[i][j] -= lr * grad_b[i][j]
        
        # 更新参数
        self.update_parameters(loss)
        
        self.global_step += 1
        
        return loss, metrics
    
    def get_state(self) -> Dict[str, Any]:
        """获取训练状态"""
        return {
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'log_temperature': self.log_temperature,
            'bias': self.bias
        }
    
    def set_state(self, state: Dict[str, Any]):
        """设置训练状态"""
        self.current_epoch = state.get('current_epoch', 0)
        self.global_step = state.get('global_step', 0)
        self.best_loss = state.get('best_loss', float('inf'))
        self.log_temperature = state.get('log_temperature', math.log(self.temperature))
        self.bias = state.get('bias', self.bias_init)


class SigLIPWithHardNegatives(SigLIPTrainer):
    """带硬负样本挖掘的SigLIP训练器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.hard_negative_weight = config.get('hard_negative_weight', 0.5) if config else 0.5
    
    def find_hard_negatives(self, sim_matrix: List[List[float]], 
                            top_k: int = 5) -> List[Tuple[int, int]]:
        """找到硬负样本"""
        n = len(sim_matrix)
        hard_negatives = []
        
        for i in range(n):
            # 找到与当前样本最相似的负样本
            row = sim_matrix[i]
            # 排除对角线
            candidates = [(j, row[j]) for j in range(n) if j != i]
            # 按相似度排序
            candidates.sort(key=lambda x: x[1], reverse=True)
            # 取top-k
            for j, _ in candidates[:top_k]:
                hard_negatives.append((i, j))
        
        return hard_negatives
    
    def compute_loss(self, features_a: List[List[float]], 
                     features_b: List[List[float]]) -> Tuple[float, Dict[str, float]]:
        """计算带硬负样本权重的损失"""
        n = len(features_a)
        if n == 0:
            return 0.0, {}
        
        sim_matrix = self.compute_similarity_matrix(features_a, features_b)
        hard_negatives = self.find_hard_negatives(sim_matrix)
        hard_neg_set = set(hard_negatives)
        
        total_loss = 0.0
        correct_matches = 0
        total_pairs = 0
        
        for i in range(n):
            for j in range(n):
                target = 1 if i == j else -1
                loss, _ = self.compute_sigmoid_loss(sim_matrix[i][j], target)
                
                # 硬负样本加权
                weight = 1.0
                if (i, j) in hard_neg_set:
                    weight = self.hard_negative_weight
                
                total_loss += weight * loss
                total_pairs += weight
                
                if i == j and sim_matrix[i][j] > 0:
                    correct_matches += 1
        
        avg_loss = total_loss / total_pairs if total_pairs > 0 else 0.0
        
        metrics = {
            'loss': avg_loss,
            'accuracy': correct_matches / n if n > 0 else 0.0,
            'temperature': math.exp(self.log_temperature),
            'bias': self.bias,
            'num_hard_negatives': len(hard_negatives)
        }
        
        return avg_loss, metrics


def create_siglip_trainer(embed_dim: int = 512, 
                          temperature: float = 0.1) -> SigLIPTrainer:
    """创建SigLIP训练器"""
    return SigLIPTrainer({
        'embed_dim': embed_dim,
        'temperature': temperature
    })
