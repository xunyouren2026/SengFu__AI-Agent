"""
对比学习训练器
实现CLIP风格的对比学习训练
"""
from typing import Optional, List, Dict, Any, Tuple, Callable
import math
import random


class ContrastiveTrainer:
    """对比学习训练器
    
    实现CLIP风格的图像-文本对比学习
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        self.embed_dim = config.get('embed_dim', 512)
        self.batch_size = config.get('batch_size', 64)
        self.temperature = config.get('temperature', 0.07)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.weight_decay = config.get('weight_decay', 0.01)
        self.max_epochs = config.get('max_epochs', 100)
        
        # 可学习的温度参数
        self.log_temperature = math.log(self.temperature)
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
    
    def _softmax(self, x: List[float]) -> List[float]:
        """Softmax函数"""
        max_val = max(x)
        exp_vals = [math.exp(v - max_val) for v in x]
        sum_exp = sum(exp_vals)
        return [e / sum_exp for e in exp_vals]
    
    def _normalize(self, x: List[float]) -> List[float]:
        """L2归一化"""
        norm = math.sqrt(sum(xi ** 2 for xi in x))
        if norm > 0:
            return [xi / norm for xi in x]
        return x
    
    def compute_similarity_matrix(self, features_a: List[List[float]], 
                                   features_b: List[List[float]]) -> List[List[float]]:
        """
        计算相似度矩阵
        
        Args:
            features_a: 模态A特征 [N, D]
            features_b: 模态B特征 [N, D]
        
        Returns:
            相似度矩阵 [N, N]
        """
        # 归一化
        features_a = [self._normalize(f) for f in features_a]
        features_b = [self._normalize(f) for f in features_b]
        
        # 计算余弦相似度
        temperature = math.exp(self.log_temperature)
        sim_matrix = []
        
        for fa in features_a:
            row = []
            for fb in features_b:
                sim = sum(a * b for a, b in zip(fa, fb)) / temperature
                row.append(sim)
            sim_matrix.append(row)
        
        return sim_matrix
    
    def compute_loss(self, features_a: List[List[float]], 
                     features_b: List[List[float]]) -> Tuple[float, Dict[str, float]]:
        """
        计算对比损失
        
        Args:
            features_a: 模态A特征 [N, D]
            features_b: 模态B特征 [N, D]
        
        Returns:
            loss: 对比损失值
            metrics: 额外的度量指标
        """
        n = len(features_a)
        if n == 0:
            return 0.0, {}
        
        # 计算相似度矩阵
        sim_matrix = self.compute_similarity_matrix(features_a, features_b)
        
        # 计算损失 (对称的对比损失)
        loss_a_to_b = 0.0
        loss_b_to_a = 0.0
        
        correct_a_to_b = 0
        correct_b_to_a = 0
        
        for i in range(n):
            # A -> B 方向
            row = sim_matrix[i]
            probs = self._softmax(row)
            loss_a_to_b -= math.log(probs[i] + 1e-10)
            
            # 检查是否正确匹配
            if row[i] == max(row):
                correct_a_to_b += 1
            
            # B -> A 方向
            col = [sim_matrix[j][i] for j in range(n)]
            probs = self._softmax(col)
            loss_b_to_a -= math.log(probs[i] + 1e-10)
            
            if col[i] == max(col):
                correct_b_to_a += 1
        
        # 平均损失
        loss_a_to_b /= n
        loss_b_to_a /= n
        
        # 总损失
        total_loss = (loss_a_to_b + loss_b_to_a) / 2
        
        # 度量指标
        metrics = {
            'loss_a_to_b': loss_a_to_b,
            'loss_b_to_a': loss_b_to_a,
            'accuracy_a_to_b': correct_a_to_b / n,
            'accuracy_b_to_a': correct_b_to_a / n,
            'temperature': math.exp(self.log_temperature)
        }
        
        return total_loss, metrics
    
    def compute_gradients(self, features_a: List[List[float]], 
                          features_b: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:
        """
        计算梯度 (简化版本)
        
        Args:
            features_a: 模态A特征
            features_b: 模态B特征
        
        Returns:
            grad_a: 模态A的梯度
            grad_b: 模态B的梯度
        """
        n = len(features_a)
        d = len(features_a[0]) if features_a else 0
        
        # 归一化
        features_a_norm = [self._normalize(f) for f in features_a]
        features_b_norm = [self._normalize(f) for f in features_b]
        
        # 计算相似度矩阵
        sim_matrix = self.compute_similarity_matrix(features_a, features_b)
        
        # 计算softmax概率
        probs_a_to_b = [self._softmax(row) for row in sim_matrix]
        probs_b_to_a = [[self._softmax([sim_matrix[j][i] for j in range(n)])[i] 
                        for i in range(n)] for _ in range(n)]
        
        # 初始化梯度
        grad_a = [[0.0] * d for _ in range(n)]
        grad_b = [[0.0] * d for _ in range(n)]
        
        temperature = math.exp(self.log_temperature)
        
        for i in range(n):
            for j in range(n):
                # A -> B 方向的梯度
                coef_a = probs_a_to_b[i][j] - (1.0 if i == j else 0.0)
                for k in range(d):
                    grad_a[i][k] += coef_a * features_b_norm[j][k] / temperature
                    grad_b[j][k] += coef_a * features_a_norm[i][k] / temperature
        
        return grad_a, grad_b
    
    def update_temperature(self, loss: float, target_loss: float = 2.0):
        """
        更新温度参数
        
        Args:
            loss: 当前损失
            target_loss: 目标损失
        """
        # 简单的温度调整策略
        if loss > target_loss:
            self.log_temperature -= 0.01
        else:
            self.log_temperature += 0.01
        
        # 限制范围
        self.log_temperature = max(-5.0, min(5.0, self.log_temperature))
    
    def train_step(self, features_a: List[List[float]], 
                   features_b: List[List[float]],
                   lr: Optional[float] = None) -> Tuple[float, Dict[str, float]]:
        """
        训练步骤
        
        Args:
            features_a: 模态A特征
            features_b: 模态B特征
            lr: 学习率
        
        Returns:
            loss: 损失值
            metrics: 度量指标
        """
        lr = lr or self.learning_rate
        
        # 计算损失和梯度
        loss, metrics = self.compute_loss(features_a, features_b)
        grad_a, grad_b = self.compute_gradients(features_a, features_b)
        
        # 更新特征 (模拟参数更新)
        for i in range(len(features_a)):
            for j in range(len(features_a[i])):
                features_a[i][j] -= lr * grad_a[i][j]
        
        for i in range(len(features_b)):
            for j in range(len(features_b[i])):
                features_b[i][j] -= lr * grad_b[i][j]
        
        # 更新温度
        self.update_temperature(loss)
        
        self.global_step += 1
        
        return loss, metrics
    
    def get_state(self) -> Dict[str, Any]:
        """获取训练状态"""
        return {
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'log_temperature': self.log_temperature
        }
    
    def set_state(self, state: Dict[str, Any]):
        """设置训练状态"""
        self.current_epoch = state.get('current_epoch', 0)
        self.global_step = state.get('global_step', 0)
        self.best_loss = state.get('best_loss', float('inf'))
        self.log_temperature = state.get('log_temperature', math.log(self.temperature))


class MultiModalContrastiveTrainer:
    """多模态对比学习训练器"""
    
    def __init__(self, modalities: List[str], embed_dim: int = 512):
        self.modalities = modalities
        self.embed_dim = embed_dim
        
        # 为每对模态创建训练器
        self.trainers = {}
        for i, mod_a in enumerate(modalities):
            for mod_b in modalities[i + 1:]:
                key = f"{mod_a}_{mod_b}"
                self.trainers[key] = ContrastiveTrainer({'embed_dim': embed_dim})
    
    def compute_loss(self, features: Dict[str, List[List[float]]]) -> Tuple[float, Dict[str, float]]:
        """
        计算多模态对比损失
        
        Args:
            features: 各模态特征字典
        
        Returns:
            total_loss: 总损失
            metrics: 度量指标
        """
        total_loss = 0.0
        metrics = {}
        count = 0
        
        for key, trainer in self.trainers.items():
            mod_a, mod_b = key.split('_')
            
            if mod_a in features and mod_b in features:
                loss, pair_metrics = trainer.compute_loss(features[mod_a], features[mod_b])
                total_loss += loss
                count += 1
                
                for k, v in pair_metrics.items():
                    metrics[f"{key}_{k}"] = v
        
        if count > 0:
            total_loss /= count
        
        metrics['total_loss'] = total_loss
        
        return total_loss, metrics


def create_contrastive_trainer(embed_dim: int = 512, 
                               temperature: float = 0.07) -> ContrastiveTrainer:
    """创建对比学习训练器"""
    return ContrastiveTrainer({
        'embed_dim': embed_dim,
        'temperature': temperature
    })
