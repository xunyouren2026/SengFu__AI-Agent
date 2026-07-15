"""
知识集成
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import math


class EnsembleMethod(Enum):
    """集成方法"""
    AVERAGE = "average"  # 平均
    WEIGHTED = "weighted"  # 加权平均
    VOTING = "voting"  # 投票
    STACKING = "stacking"  # 堆叠
    BAGGING = "bagging"  # 袋装


class ModelPrediction:
    """模型预测"""
    
    def __init__(
        self,
        model_id: str,
        predictions: List[Any],
        confidence: Optional[List[float]] = None
    ):
        self.model_id = model_id
        self.predictions = predictions
        self.confidence = confidence or [1.0] * len(predictions)
        self.timestamp = datetime.now().timestamp()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_id': self.model_id,
            'predictions': self.predictions,
            'confidence': self.confidence
        }


class KnowledgeEnsemble:
    """
    知识集成
    
    集成多个客户端的知识
    """
    
    def __init__(
        self,
        method: EnsembleMethod = EnsembleMethod.WEIGHTED,
        min_models: int = 2
    ):
        self.method = method
        self.min_models = min_models
        
        self._predictions: Dict[str, ModelPrediction] = {}
        self._weights: Dict[str, float] = {}
        self._model_scores: Dict[str, float] = {}
    
    def add_prediction(
        self,
        model_id: str,
        predictions: List[Any],
        confidence: Optional[List[float]] = None,
        weight: Optional[float] = None
    ) -> None:
        """添加预测"""
        pred = ModelPrediction(model_id, predictions, confidence)
        self._predictions[model_id] = pred
        
        if weight is not None:
            self._weights[model_id] = weight
    
    def set_weight(self, model_id: str, weight: float) -> None:
        """设置模型权重"""
        self._weights[model_id] = weight
    
    def update_score(self, model_id: str, score: float) -> None:
        """更新模型分数"""
        self._model_scores[model_id] = score
        
        # 自动更新权重
        if self.method == EnsembleMethod.WEIGHTED:
            self._weights[model_id] = score
    
    def ensemble(self) -> List[Any]:
        """
        执行集成
        
        Returns:
            集成后的预测
        """
        if len(self._predictions) < self.min_models:
            # 返回任意一个预测
            if self._predictions:
                return list(self._predictions.values())[0].predictions
            return []
        
        if self.method == EnsembleMethod.AVERAGE:
            return self._ensemble_average()
        elif self.method == EnsembleMethod.WEIGHTED:
            return self._ensemble_weighted()
        elif self.method == EnsembleMethod.VOTING:
            return self._ensemble_voting()
        else:
            return self._ensemble_average()
    
    def _ensemble_average(self) -> List[Any]:
        """平均集成"""
        all_preds = list(self._predictions.values())
        n = len(all_preds)
        
        if n == 0:
            return []
        
        result = []
        for i in range(len(all_preds[0].predictions)):
            total = 0.0
            count = 0
            
            for pred in all_preds:
                if i < len(pred.predictions):
                    val = pred.predictions[i]
                    if isinstance(val, (int, float)):
                        total += val
                        count += 1
            
            if count > 0:
                result.append(total / count)
            else:
                result.append(0)
        
        return result
    
    def _ensemble_weighted(self) -> List[Any]:
        """加权集成"""
        all_preds = list(self._predictions.values())
        
        if not all_preds:
            return []
        
        # 归一化权重
        total_weight = sum(
            self._weights.get(p.model_id, 1.0)
            for p in all_preds
        )
        
        if total_weight == 0:
            return self._ensemble_average()
        
        result = []
        for i in range(len(all_preds[0].predictions)):
            weighted_sum = 0.0
            
            for pred in all_preds:
                if i < len(pred.predictions):
                    val = pred.predictions[i]
                    weight = self._weights.get(pred.model_id, 1.0)
                    
                    if isinstance(val, (int, float)):
                        weighted_sum += weight * val
            
            result.append(weighted_sum / total_weight)
        
        return result
    
    def _ensemble_voting(self) -> List[Any]:
        """投票集成"""
        all_preds = list(self._predictions.values())
        
        if not all_preds:
            return []
        
        result = []
        for i in range(len(all_preds[0].predictions)):
            votes: Dict[Any, float] = {}
            
            for pred in all_preds:
                if i < len(pred.predictions):
                    val = pred.predictions[i]
                    weight = self._weights.get(pred.model_id, 1.0)
                    votes[val] = votes.get(val, 0.0) + weight
            
            # 选择票数最多的
            if votes:
                best = max(votes.items(), key=lambda x: x[1])
                result.append(best[0])
            else:
                result.append(0)
        
        return result
    
    def get_uncertainty(self) -> List[float]:
        """
        计算预测不确定性
        
        使用预测方差作为不确定性度量
        """
        all_preds = list(self._predictions.values())
        
        if len(all_preds) < 2:
            return [0.0] * len(all_preds[0].predictions) if all_preds else []
        
        uncertainties = []
        
        for i in range(len(all_preds[0].predictions)):
            values = []
            for pred in all_preds:
                if i < len(pred.predictions):
                    val = pred.predictions[i]
                    if isinstance(val, (int, float)):
                        values.append(val)
            
            if len(values) >= 2:
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                uncertainties.append(math.sqrt(variance))
            else:
                uncertainties.append(0.0)
        
        return uncertainties
    
    def clear(self) -> None:
        """清空预测"""
        self._predictions.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'method': self.method.value,
            'num_models': len(self._predictions),
            'min_models': self.min_models,
            'model_ids': list(self._predictions.keys()),
            'weights': dict(self._weights)
        }


class AdaptiveEnsemble(KnowledgeEnsemble):
    """
    自适应集成
    
    根据模型性能动态调整权重
    """
    
    def __init__(
        self,
        adaptation_rate: float = 0.1,
        min_weight: float = 0.01
    ):
        super().__init__(method=EnsembleMethod.WEIGHTED)
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight
    
    def adapt_weights(
        self,
        model_performances: Dict[str, float]
    ) -> None:
        """
        根据性能自适应调整权重
        
        Args:
            model_performances: 模型ID -> 性能分数
        """
        for model_id, performance in model_performances.items():
            if model_id in self._weights:
                # 指数移动平均
                old_weight = self._weights[model_id]
                new_weight = (
                    (1 - self.adaptation_rate) * old_weight +
                    self.adaptation_rate * performance
                )
                self._weights[model_id] = max(self.min_weight, new_weight)
            else:
                self._weights[model_id] = max(self.min_weight, performance)
    
    def get_top_models(self, k: int = 3) -> List[str]:
        """获取权重最高的k个模型"""
        sorted_models = sorted(
            self._weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [model_id for model_id, _ in sorted_models[:k]]
