"""
联邦聚合器 - 实现FedAvg/FedProx/FedAdam算法
"""
from typing import Dict, List, Optional, Tuple, Any, Callable
from abc import ABC, abstractmethod
import math
import copy


class ModelParameters:
    """模型参数容器，支持嵌套字典结构"""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        self.version = 0
        self.timestamp = 0.0
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取参数"""
        keys = key.split('.')
        value = self.params
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """设置参数"""
        keys = key.split('.')
        params = self.params
        for k in keys[:-1]:
            if k not in params:
                params[k] = {}
            params = params[k]
        params[keys[-1]] = value
    
    def keys(self) -> List[str]:
        """获取所有键"""
        def _flatten(d: Dict, prefix: str = '') -> List[str]:
            keys = []
            for k, v in d.items():
                full_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    keys.extend(_flatten(v, full_key))
                else:
                    keys.append(full_key)
            return keys
        return _flatten(self.params)
    
    def copy(self) -> 'ModelParameters':
        """深拷贝"""
        new_params = ModelParameters(copy.deepcopy(self.params))
        new_params.version = self.version
        new_params.timestamp = self.timestamp
        return new_params
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return copy.deepcopy(self.params)
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ModelParameters':
        """从字典创建"""
        return ModelParameters(copy.deepcopy(data))


class BaseAggregator(ABC):
    """聚合器基类"""
    
    def __init__(self, name: str = "BaseAggregator"):
        self.name = name
        self.round_num = 0
        self.history: List[Dict[str, Any]] = []
    
    @abstractmethod
    def aggregate(
        self,
        client_updates: List[Tuple[ModelParameters, int, str]],
        global_model: ModelParameters
    ) -> ModelParameters:
        """
        聚合客户端更新
        
        Args:
            client_updates: List of (model_params, num_samples, client_id)
            global_model: 当前全局模型
        Returns:
            新的全局模型参数
        """
        pass
    
    def reset(self) -> None:
        """重置聚合器状态"""
        self.round_num = 0
        self.history = []
    
    def get_history(self) -> List[Dict[str, Any]]:
        """获取聚合历史"""
        return self.history


class FedAvgAggregator(BaseAggregator):
    """
    FedAvg聚合器 - 联邦平均算法
    
    实现加权平均聚合:
    w_{t+1} = sum_{k} (n_k / n) * w_k
    其中 n_k 是客户端k的样本数，n是总样本数
    """
    
    def __init__(self, min_clients: int = 1, weighted: bool = True):
        super().__init__(name="FedAvg")
        self.min_clients = min_clients
        self.weighted = weighted
    
    def aggregate(
        self,
        client_updates: List[Tuple[ModelParameters, int, str]],
        global_model: ModelParameters
    ) -> ModelParameters:
        """FedAvg加权平均聚合"""
        if len(client_updates) < self.min_clients:
            raise ValueError(f"需要至少 {self.min_clients} 个客户端更新，当前只有 {len(client_updates)}")
        
        # 计算总样本数
        total_samples = sum(n for _, n, _ in client_updates)
        
        if total_samples == 0:
            raise ValueError("总样本数为0，无法聚合")
        
        # 初始化聚合参数
        aggregated_params: Dict[str, Any] = {}
        all_keys = set()
        
        # 收集所有参数键
        for params, _, _ in client_updates:
            all_keys.update(params.keys())
        
        # 对每个参数进行加权平均
        for key in all_keys:
            weighted_sum = 0.0
            for params, num_samples, _ in client_updates:
                value = params.get(key)
                if value is not None:
                    if isinstance(value, (int, float)):
                        weight = num_samples / total_samples if self.weighted else 1.0 / len(client_updates)
                        weighted_sum += weight * value
                    elif isinstance(value, list):
                        # 处理列表类型（如权重矩阵）
                        if key not in aggregated_params:
                            aggregated_params[key] = [0.0] * len(value)
                        weight = num_samples / total_samples if self.weighted else 1.0 / len(client_updates)
                        for i, v in enumerate(value):
                            if isinstance(v, (int, float)):
                                aggregated_params[key][i] += weight * v
            
            if key not in aggregated_params and isinstance(weighted_sum, (int, float)):
                aggregated_params[key] = weighted_sum
        
        # 创建新的模型参数
        new_model = ModelParameters(aggregated_params)
        new_model.version = global_model.version + 1
        
        # 记录历史
        self.round_num += 1
        self.history.append({
            'round': self.round_num,
            'num_clients': len(client_updates),
            'total_samples': total_samples,
            'algorithm': 'FedAvg'
        })
        
        return new_model


class FedProxAggregator(BaseAggregator):
    """
    FedProx聚合器 - 带近端项的联邦学习
    
    在客户端目标函数中添加近端项:
    min_w F_k(w) + (mu/2) ||w - w^t||^2
    
    聚合方式与FedAvg相同，但客户端训练时考虑近端正则化
    """
    
    def __init__(
        self,
        mu: float = 0.01,
        min_clients: int = 1,
        weighted: bool = True
    ):
        super().__init__(name="FedProx")
        self.mu = mu  # 近端项系数
        self.min_clients = min_clients
        self.weighted = weighted
    
    def aggregate(
        self,
        client_updates: List[Tuple[ModelParameters, int, str]],
        global_model: ModelParameters
    ) -> ModelParameters:
        """FedProx聚合 - 本质上与FedAvg相同"""
        if len(client_updates) < self.min_clients:
            raise ValueError(f"需要至少 {self.min_clients} 个客户端更新")
        
        total_samples = sum(n for _, n, _ in client_updates)
        
        if total_samples == 0:
            raise ValueError("总样本数为0")
        
        aggregated_params: Dict[str, Any] = {}
        all_keys = set()
        
        for params, _, _ in client_updates:
            all_keys.update(params.keys())
        
        for key in all_keys:
            weighted_sum = 0.0
            for params, num_samples, _ in client_updates:
                value = params.get(key)
                if value is not None:
                    if isinstance(value, (int, float)):
                        weight = num_samples / total_samples if self.weighted else 1.0 / len(client_updates)
                        weighted_sum += weight * value
                    elif isinstance(value, list):
                        if key not in aggregated_params:
                            aggregated_params[key] = [0.0] * len(value)
                        weight = num_samples / total_samples if self.weighted else 1.0 / len(client_updates)
                        for i, v in enumerate(value):
                            if isinstance(v, (int, float)):
                                aggregated_params[key][i] += weight * v
            
            if key not in aggregated_params and isinstance(weighted_sum, (int, float)):
                aggregated_params[key] = weighted_sum
        
        new_model = ModelParameters(aggregated_params)
        new_model.version = global_model.version + 1
        
        self.round_num += 1
        self.history.append({
            'round': self.round_num,
            'num_clients': len(client_updates),
            'total_samples': total_samples,
            'mu': self.mu,
            'algorithm': 'FedProx'
        })
        
        return new_model
    
    def compute_proximal_gradient(
        self,
        local_params: ModelParameters,
        global_params: ModelParameters
    ) -> Dict[str, Any]:
        """计算近端项梯度 mu * (w - w^t)"""
        proximal_grad: Dict[str, Any] = {}
        
        for key in local_params.keys():
            local_val = local_params.get(key)
            global_val = global_params.get(key, 0)
            
            if local_val is not None:
                if isinstance(local_val, (int, float)) and isinstance(global_val, (int, float)):
                    proximal_grad[key] = self.mu * (local_val - global_val)
                elif isinstance(local_val, list) and isinstance(global_val, list):
                    proximal_grad[key] = [
                        self.mu * (lv - gv) 
                        for lv, gv in zip(local_val, global_val)
                    ]
        
        return proximal_grad


class FedAdamAggregator(BaseAggregator):
    """
    FedAdam聚合器 - 基于Adam优化的联邦聚合
    
    使用服务器端Adam优化:
    m_t = beta1 * m_{t-1} + (1 - beta1) * delta_t
    v_t = beta2 * v_{t-1} + (1 - beta2) * delta_t^2
    w_{t+1} = w_t - lr * m_t / (sqrt(v_t) + epsilon)
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        tau: float = 1e-3,
        min_clients: int = 1
    ):
        super().__init__(name="FedAdam")
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.tau = tau  # 用于v_t的偏置校正
        self.min_clients = min_clients
        
        # Adam状态
        self.m: Dict[str, Any] = {}  # 一阶矩估计
        self.v: Dict[str, Any] = {}  # 二阶矩估计
    
    def _compute_delta(
        self,
        client_updates: List[Tuple[ModelParameters, int, str]],
        global_model: ModelParameters
    ) -> Dict[str, Any]:
        """计算平均更新量 delta_t = w_t - avg(w_k)"""
        total_samples = sum(n for _, n, _ in client_updates)
        
        avg_update: Dict[str, Any] = {}
        all_keys = set()
        
        for params, _, _ in client_updates:
            all_keys.update(params.keys())
        
        for key in all_keys:
            weighted_sum = 0.0
            global_val = global_model.get(key, 0)
            
            for params, num_samples, _ in client_updates:
                value = params.get(key)
                if value is not None and isinstance(value, (int, float)):
                    weight = num_samples / total_samples
                    weighted_sum += weight * value
            
            if isinstance(global_val, (int, float)):
                avg_update[key] = global_val - weighted_sum
        
        return avg_update
    
    def aggregate(
        self,
        client_updates: List[Tuple[ModelParameters, int, str]],
        global_model: ModelParameters
    ) -> ModelParameters:
        """FedAdam聚合"""
        if len(client_updates) < self.min_clients:
            raise ValueError(f"需要至少 {self.min_clients} 个客户端更新")
        
        # 计算平均更新量
        delta = self._compute_delta(client_updates, global_model)
        
        # 更新Adam状态
        new_params = global_model.to_dict()
        
        for key, delta_val in delta.items():
            if not isinstance(delta_val, (int, float)):
                continue
            
            # 初始化状态
            if key not in self.m:
                self.m[key] = 0.0
                self.v[key] = 0.0
            
            # Adam更新
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * delta_val
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (delta_val ** 2)
            
            # 偏置校正
            m_hat = self.m[key] / (1 - self.beta1 ** (self.round_num + 1))
            v_hat = self.v[key] / (1 - self.beta2 ** (self.round_num + 1))
            
            # 更新参数
            current_val = global_model.get(key, 0)
            if isinstance(current_val, (int, float)):
                new_params[key] = current_val - self.lr * m_hat / (math.sqrt(v_hat) + self.tau)
        
        new_model = ModelParameters(new_params)
        new_model.version = global_model.version + 1
        
        total_samples = sum(n for _, n, _ in client_updates)
        self.round_num += 1
        self.history.append({
            'round': self.round_num,
            'num_clients': len(client_updates),
            'total_samples': total_samples,
            'lr': self.lr,
            'algorithm': 'FedAdam'
        })
        
        return new_model
    
    def reset(self) -> None:
        """重置聚合器状态"""
        super().reset()
        self.m = {}
        self.v = {}


class AggregationStrategy:
    """聚合策略工厂"""
    
    @staticmethod
    def create(
        algorithm: str = "fedavg",
        **kwargs
    ) -> BaseAggregator:
        """创建聚合器实例"""
        algorithm = algorithm.lower()
        
        if algorithm == "fedavg":
            return FedAvgAggregator(**kwargs)
        elif algorithm == "fedprox":
            return FedProxAggregator(**kwargs)
        elif algorithm == "fedadam":
            return FedAdamAggregator(**kwargs)
        else:
            raise ValueError(f"未知的聚合算法: {algorithm}")


class SecureAggregator(BaseAggregator):
    """
    安全聚合器 - 支持差分隐私的聚合
    
    在聚合结果中添加高斯噪声以实现差分隐私
    """
    
    def __init__(
        self,
        base_aggregator: BaseAggregator,
        noise_scale: float = 0.1,
        clip_norm: float = 1.0
    ):
        super().__init__(name=f"Secure{base_aggregator.name}")
        self.base_aggregator = base_aggregator
        self.noise_scale = noise_scale
        self.clip_norm = clip_norm
    
    def _clip_params(self, params: ModelParameters) -> ModelParameters:
        """裁剪参数以限制敏感度"""
        clipped = params.copy()
        
        for key in clipped.keys():
            value = clipped.get(key)
            if isinstance(value, (int, float)):
                # L2裁剪
                if abs(value) > self.clip_norm:
                    clipped.set(key, self.clip_norm * (1 if value >= 0 else -1))
            elif isinstance(value, list):
                # 计算L2范数
                norm = math.sqrt(sum(v ** 2 for v in value if isinstance(v, (int, float))))
                if norm > self.clip_norm:
                    scale = self.clip_norm / norm
                    clipped.set(key, [v * scale for v in value])
        
        return clipped
    
    def _add_noise(self, params: ModelParameters) -> ModelParameters:
        """添加高斯噪声"""
        import random
        noisy = params.copy()
        
        for key in noisy.keys():
            value = noisy.get(key)
            if isinstance(value, (int, float)):
                noise = random.gauss(0, self.noise_scale * self.clip_norm)
                noisy.set(key, value + noise)
            elif isinstance(value, list):
                noisy_list = [
                    v + random.gauss(0, self.noise_scale * self.clip_norm)
                    for v in value
                ]
                noisy.set(key, noisy_list)
        
        return noisy
    
    def aggregate(
        self,
        client_updates: List[Tuple[ModelParameters, int, str]],
        global_model: ModelParameters
    ) -> ModelParameters:
        """安全聚合"""
        # 裁剪所有客户端更新
        clipped_updates = [
            (self._clip_params(params), n, cid)
            for params, n, cid in client_updates
        ]
        
        # 使用基础聚合器聚合
        aggregated = self.base_aggregator.aggregate(clipped_updates, global_model)
        
        # 添加噪声
        result = self._add_noise(aggregated)
        
        self.round_num += 1
        self.history.append({
            'round': self.round_num,
            'num_clients': len(client_updates),
            'noise_scale': self.noise_scale,
            'clip_norm': self.clip_norm,
            'algorithm': f'Secure{self.base_aggregator.name}'
        })
        
        return result
