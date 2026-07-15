"""
客户端训练器 - 本地训练循环
"""
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime
import math
import random
import copy


class TrainingConfig:
    """训练配置"""
    
    def __init__(
        self,
        local_epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        gradient_clip: Optional[float] = None,
        proximal_mu: float = 0.0  # FedProx参数
    ):
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.proximal_mu = proximal_mu


class LocalDataset:
    """本地数据集抽象"""
    
    def __init__(
        self,
        data: List[Tuple[Any, Any]],
        client_id: str
    ):
        self.data = data
        self.client_id = client_id
        self._indices: List[int] = list(range(len(data)))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def shuffle(self) -> None:
        """打乱数据"""
        random.shuffle(self._indices)
    
    def get_batch(self, batch_size: int, offset: int = 0) -> List[Tuple[Any, Any]]:
        """获取一个批次"""
        end = min(offset + batch_size, len(self._indices))
        indices = self._indices[offset:end]
        return [self.data[i] for i in indices]
    
    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool = True
    ) -> List[List[Tuple[Any, Any]]]:
        """迭代所有批次"""
        if shuffle:
            self.shuffle()
        
        batches = []
        for i in range(0, len(self), batch_size):
            batch = self.get_batch(batch_size, i)
            batches.append(batch)
        
        return batches


class ModelWrapper:
    """
    模型包装器
    
    提供统一的模型接口，支持参数访问和更新
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        self._gradients: Dict[str, Any] = {}
        self._velocity: Dict[str, Any] = {}  # 动量
    
    def get_params(self) -> Dict[str, Any]:
        """获取参数"""
        return copy.deepcopy(self.params)
    
    def set_params(self, params: Dict[str, Any]) -> None:
        """设置参数"""
        self.params = copy.deepcopy(params)
    
    def get_param(self, key: str) -> Any:
        """获取单个参数"""
        return self.params.get(key)
    
    def set_param(self, key: str, value: Any) -> None:
        """设置单个参数"""
        self.params[key] = value
    
    def compute_gradient(
        self,
        data_batch: List[Tuple[Any, Any]],
        loss_fn: Callable
    ) -> Dict[str, Any]:
        """
        计算梯度（简化实现）
        
        实际应用中应使用自动微分
        """
        # 这里是简化的梯度计算示例
        # 实际实现需要根据具体模型和损失函数
        gradients: Dict[str, Any] = {}
        
        for key, param in self.params.items():
            if isinstance(param, (int, float)):
                # 数值梯度（简化）
                gradients[key] = random.gauss(0, 0.1)
            elif isinstance(param, list):
                gradients[key] = [random.gauss(0, 0.1) for _ in param]
        
        self._gradients = gradients
        return gradients
    
    def apply_gradients(
        self,
        gradients: Dict[str, Any],
        config: TrainingConfig
    ) -> None:
        """应用梯度更新参数"""
        lr = config.learning_rate
        
        for key, grad in gradients.items():
            param = self.params.get(key)
            if param is None:
                continue
            
            # 动量更新
            if config.momentum > 0:
                if key not in self._velocity:
                    if isinstance(grad, (int, float)):
                        self._velocity[key] = 0.0
                    elif isinstance(grad, list):
                        self._velocity[key] = [0.0] * len(grad)
                
                if isinstance(grad, (int, float)):
                    self._velocity[key] = config.momentum * self._velocity[key] + grad
                    update = lr * self._velocity[key]
                elif isinstance(grad, list):
                    self._velocity[key] = [
                        config.momentum * v + g
                        for v, g in zip(self._velocity[key], grad)
                    ]
                    update = [lr * v for v in self._velocity[key]]
            else:
                if isinstance(grad, (int, float)):
                    update = lr * grad
                elif isinstance(grad, list):
                    update = [lr * g for g in grad]
                else:
                    continue
            
            # 梯度裁剪
            if config.gradient_clip is not None:
                if isinstance(update, (int, float)):
                    if abs(update) > config.gradient_clip:
                        update = config.gradient_clip * (1 if update >= 0 else -1)
                elif isinstance(update, list):
                    norm = math.sqrt(sum(u ** 2 for u in update))
                    if norm > config.gradient_clip:
                        scale = config.gradient_clip / norm
                        update = [u * scale for u in update]
            
            # 应用更新
            if isinstance(param, (int, float)) and isinstance(update, (int, float)):
                self.params[key] = param - update
            elif isinstance(param, list) and isinstance(update, list):
                self.params[key] = [
                    p - u for p, u in zip(param, update)
                ]
    
    def add_proximal_term(
        self,
        global_params: Dict[str, Any],
        mu: float
    ) -> None:
        """添加近端项（FedProx）"""
        for key, global_val in global_params.items():
            local_val = self.params.get(key)
            if local_val is None:
                continue
            
            if isinstance(local_val, (int, float)) and isinstance(global_val, (int, float)):
                proximal_grad = mu * (local_val - global_val)
                if key in self._gradients:
                    self._gradients[key] += proximal_grad
                else:
                    self._gradients[key] = proximal_grad


class TrainingMetrics:
    """训练指标"""
    
    def __init__(self):
        self.losses: List[float] = []
        self.accuracies: List[float] = []
        self.timestamps: List[float] = []
    
    def add(self, loss: float, accuracy: float) -> None:
        """添加指标"""
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        self.timestamps.append(datetime.now().timestamp())
    
    def get_latest(self) -> Tuple[Optional[float], Optional[float]]:
        """获取最新指标"""
        if not self.losses:
            return None, None
        return self.losses[-1], self.accuracies[-1]
    
    def get_average(self, last_n: Optional[int] = None) -> Tuple[float, float]:
        """获取平均指标"""
        losses = self.losses[-last_n:] if last_n else self.losses
        accs = self.accuracies[-last_n:] if last_n else self.accuracies
        
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        avg_acc = sum(accs) / len(accs) if accs else 0.0
        
        return avg_loss, avg_acc
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'num_records': len(self.losses),
            'latest_loss': self.losses[-1] if self.losses else None,
            'latest_accuracy': self.accuracies[-1] if self.accuracies else None,
            'avg_loss': sum(self.losses) / len(self.losses) if self.losses else None,
            'avg_accuracy': sum(self.accuracies) / len(self.accuracies) if self.accuracies else None
        }


class ClientTrainer:
    """
    客户端训练器
    
    执行本地训练循环，支持:
    - 多轮本地训练
    - 梯度裁剪
    - 动量优化
    - FedProx近端正则化
    """
    
    def __init__(
        self,
        client_id: str,
        config: Optional[TrainingConfig] = None
    ):
        self.client_id = client_id
        self.config = config or TrainingConfig()
        
        self._model = ModelWrapper()
        self._dataset: Optional[LocalDataset] = None
        self._metrics = TrainingMetrics()
        
        self._current_round: int = 0
        self._is_training: bool = False
    
    def set_model(self, params: Dict[str, Any]) -> None:
        """设置模型参数"""
        self._model.set_params(params)
    
    def get_model(self) -> Dict[str, Any]:
        """获取模型参数"""
        return self._model.get_params()
    
    def set_dataset(self, data: List[Tuple[Any, Any]]) -> None:
        """设置本地数据集"""
        self._dataset = LocalDataset(data, self.client_id)
    
    def get_num_samples(self) -> int:
        """获取样本数量"""
        return len(self._dataset) if self._dataset else 0
    
    def train(
        self,
        global_params: Optional[Dict[str, Any]] = None,
        loss_fn: Optional[Callable] = None,
        on_epoch_end: Optional[Callable[[int, float, float], None]] = None
    ) -> Dict[str, Any]:
        """
        执行本地训练
        
        Args:
            global_params: 全局模型参数（用于FedProx）
            loss_fn: 损失函数
            on_epoch_end: 每轮结束回调
        
        Returns:
            训练结果
        """
        if self._dataset is None:
            raise ValueError("未设置数据集")
        
        self._is_training = True
        self._current_round += 1
        
        start_time = datetime.now().timestamp()
        
        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            total_samples = 0
            
            batches = self._dataset.iter_batches(
                self.config.batch_size,
                shuffle=True
            )
            
            for batch in batches:
                # 计算梯度
                gradients = self._model.compute_gradient(batch, loss_fn or self._default_loss)
                
                # 添加近端项（FedProx）
                if global_params is not None and self.config.proximal_mu > 0:
                    self._model.add_proximal_term(
                        global_params,
                        self.config.proximal_mu
                    )
                
                # 应用梯度
                self._model.apply_gradients(gradients, self.config)
                
                # 计算损失和准确率（简化）
                batch_loss = random.uniform(0.1, 1.0)  # 模拟损失
                batch_acc = random.uniform(0.5, 0.99)  # 模拟准确率
                
                epoch_loss += batch_loss * len(batch)
                epoch_correct += int(batch_acc * len(batch))
                total_samples += len(batch)
            
            avg_loss = epoch_loss / total_samples
            avg_acc = epoch_correct / total_samples
            
            self._metrics.add(avg_loss, avg_acc)
            
            if on_epoch_end:
                on_epoch_end(epoch, avg_loss, avg_acc)
        
        end_time = datetime.now().timestamp()
        self._is_training = False
        
        return {
            'client_id': self.client_id,
            'round': self._current_round,
            'num_samples': self.get_num_samples(),
            'local_epochs': self.config.local_epochs,
            'final_loss': avg_loss,
            'final_accuracy': avg_acc,
            'training_time': end_time - start_time,
            'model_params': self.get_model()
        }
    
    def _default_loss(self, predictions: Any, targets: Any) -> float:
        """默认损失函数（MSE）"""
        # 简化实现
        return random.uniform(0.1, 1.0)
    
    def evaluate(
        self,
        eval_data: Optional[List[Tuple[Any, Any]]] = None
    ) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            eval_data: 评估数据，None则使用训练数据
        """
        data = eval_data or (self._dataset.data if self._dataset else [])
        
        if not data:
            return {'loss': 0.0, 'accuracy': 0.0}
        
        # 简化评估
        loss = random.uniform(0.1, 0.5)
        accuracy = random.uniform(0.7, 0.95)
        
        return {'loss': loss, 'accuracy': accuracy}
    
    def get_metrics(self) -> TrainingMetrics:
        """获取训练指标"""
        return self._metrics
    
    def is_training(self) -> bool:
        """是否正在训练"""
        return self._is_training
    
    def get_current_round(self) -> int:
        """获取当前轮次"""
        return self._current_round
    
    def compute_model_difference(
        self,
        global_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """计算与全局模型的差异"""
        diff: Dict[str, Any] = {}
        
        for key in self._model.params:
            local_val = self._model.params.get(key)
            global_val = global_params.get(key)
            
            if local_val is not None and global_val is not None:
                if isinstance(local_val, (int, float)) and isinstance(global_val, (int, float)):
                    diff[key] = local_val - global_val
                elif isinstance(local_val, list) and isinstance(global_val, list):
                    diff[key] = [l - g for l, g in zip(local_val, global_val)]
        
        return diff
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'client_id': self.client_id,
            'current_round': self._current_round,
            'num_samples': self.get_num_samples(),
            'is_training': self._is_training,
            'config': {
                'local_epochs': self.config.local_epochs,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate
            },
            'metrics': self._metrics.to_dict()
        }
