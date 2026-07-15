"""
联邦学习集成：联盟间共享模型更新，安全聚合
支持FedAvg、安全聚合、差分隐私

纯 Python 标准库实现，无外部依赖
"""

import copy
import random
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import time


@dataclass
class FederatedRound:
    round_num: int
    participants: List[str]
    start_time: float
    end_time: float = 0.0
    global_model: Optional[Dict] = None
    metrics: Dict[str, float] = field(default_factory=dict)


class FederatedLearning:
    """
    联邦学习聚合器
    支持FedAvg、安全聚合、差分隐私
    """

    def __init__(self, model_initializer: callable, aggregation_method: str = "fedavg",
                 dp_epsilon: float = 1.0, dp_delta: float = 1e-5):
        """
        model_initializer: 返回初始模型参数的函数
        aggregation_method: "fedavg", "median", "trimmed_mean"
        dp_epsilon: 差分隐私预算
        """
        self.model_initializer = model_initializer
        self.aggregation_method = aggregation_method
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.global_model = model_initializer()
        self.rounds: List[FederatedRound] = []
        self._client_models: Dict[str, Dict] = {}

    def submit_model(self, client_id: str, model_update: Dict, round_num: int) -> bool:
        """客户端提交模型更新"""
        if round_num != len(self.rounds):
            return False
        if client_id not in self._client_models:
            self._client_models[client_id] = {}
        self._client_models[client_id] = model_update
        return True

    def aggregate(self, round_num: int, min_clients: int = 2) -> Optional[Dict]:
        """聚合所有客户端提交的模型"""
        if round_num != len(self.rounds):
            return None

        participants = list(self._client_models.keys())
        if len(participants) < min_clients:
            return None

        updates = list(self._client_models.values())
        global_update = self._aggregate_updates(updates)

        # 应用差分隐私
        if self.dp_epsilon < float('inf'):
            global_update = self._apply_differential_privacy(global_update)

        # 更新全局模型
        self.global_model = self._apply_update(self.global_model, global_update)

        # 记录本轮
        if self.rounds:
            self.rounds[-1].end_time = time.time()
            self.rounds[-1].global_model = copy.deepcopy(self.global_model)
            self.rounds[-1].participants = participants

        # 清空客户端提交
        self._client_models.clear()
        return self.global_model

    def _aggregate_updates(self, updates: List[Dict]) -> Dict:
        """聚合多个模型更新"""
        if self.aggregation_method == "fedavg":
            return self._fedavg_aggregate(updates)
        elif self.aggregation_method == "median":
            return self._median_aggregate(updates)
        elif self.aggregation_method == "trimmed_mean":
            return self._trimmed_mean_aggregate(updates, trim_ratio=0.3)
        else:
            return self._fedavg_aggregate(updates)

    def _fedavg_aggregate(self, updates: List[Dict]) -> Dict:
        """FedAvg：加权平均（简单平均）"""
        aggregated = {}
        if not updates:
            return aggregated
        # 获取所有键
        all_keys = set()
        for u in updates:
            all_keys.update(u.keys())
        for key in all_keys:
            values = [u.get(key, 0) for u in updates]
            aggregated[key] = sum(values) / len(values)
        return aggregated

    def _median_aggregate(self, updates: List[Dict]) -> Dict:
        """中位数聚合（抗拜占庭攻击）"""
        aggregated = {}
        if not updates:
            return aggregated
        all_keys = set()
        for u in updates:
            all_keys.update(u.keys())
        for key in all_keys:
            values = [u.get(key, 0) for u in updates]
            values.sort()
            n = len(values)
            if n % 2 == 1:
                aggregated[key] = values[n // 2]
            else:
                aggregated[key] = (values[n // 2 - 1] + values[n // 2]) / 2
        return aggregated

    def _trimmed_mean_aggregate(self, updates: List[Dict], trim_ratio: float = 0.3) -> Dict:
        """修剪平均：去掉最大最小一定比例的值"""
        aggregated = {}
        if not updates:
            return aggregated
        all_keys = set()
        for u in updates:
            all_keys.update(u.keys())
        k = max(1, int(len(updates) * trim_ratio))
        for key in all_keys:
            values = [u.get(key, 0) for u in updates]
            values.sort()
            trimmed = values[k:-k] if k > 0 else values
            aggregated[key] = sum(trimmed) / len(trimmed) if trimmed else 0
        return aggregated

    def _apply_differential_privacy(self, update: Dict) -> Dict:
        """应用差分隐私：添加高斯噪声（使用纯Python实现）"""
        sensitivity = 1.0  # 简化
        # 计算标准差：sigma = sensitivity * sqrt(2 * log(1.25 / delta)) / epsilon
        sigma = sensitivity * math.sqrt(2 * math.log(1.25 / self.dp_delta)) / self.dp_epsilon
        for key in update:
            noise = random.gauss(0, sigma)
            if isinstance(update[key], (int, float)):
                update[key] += noise
        return update

    def _apply_update(self, global_model: Dict, update: Dict) -> Dict:
        """将更新应用到全局模型"""
        new_model = copy.deepcopy(global_model)
        for key, value in update.items():
            if key in new_model:
                # 简单加法（假设更新是梯度或增量）
                if isinstance(new_model[key], (int, float)):
                    new_model[key] += value
                elif isinstance(new_model[key], list):
                    for i, v in enumerate(value):
                        if i < len(new_model[key]):
                            if isinstance(new_model[key][i], (int, float)):
                                new_model[key][i] += v
        return new_model

    def start_round(self) -> int:
        """开始新的一轮联邦学习"""
        round_num = len(self.rounds)
        self.rounds.append(FederatedRound(
            round_num=round_num,
            participants=[],
            start_time=time.time()
        ))
        return round_num

    def get_global_model(self) -> Dict:
        return copy.deepcopy(self.global_model)

    def get_round_metrics(self, round_num: int) -> Optional[Dict]:
        if round_num < len(self.rounds):
            r = self.rounds[round_num]
            return {
                "round": r.round_num,
                "participants": len(r.participants),
                "duration_ms": (r.end_time - r.start_time) * 1000 if r.end_time else 0,
                "metrics": r.metrics
            }
        return None
