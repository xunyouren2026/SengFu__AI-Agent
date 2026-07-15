"""
个性化层 - 本地微调
"""
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime
from enum import Enum
import copy
import math


class PersonalizationStrategy(Enum):
    """个性化策略"""
    LOCAL_FINE_TUNING = "local_fine_tuning"  # 本地微调
    LAYER_FREEZING = "layer_freezing"  # 层冻结
    ADAPTER = "adapter"  # 适配器层
    MIXTURE_OF_EXPERTS = "mixture_of_experts"  # 混合专家
    META_LEARNING = "meta_learning"  # 元学习


class LayerConfig:
    """层配置"""
    
    def __init__(
        self,
        layer_name: str,
        is_shared: bool = True,
        is_frozen: bool = False,
        learning_rate_scale: float = 1.0
    ):
        self.layer_name = layer_name
        self.is_shared = is_shared  # 是否与全局共享
        self.is_frozen = is_frozen  # 是否冻结
        self.learning_rate_scale = learning_rate_scale


class PersonalizedModel:
    """
    个性化模型
    
    管理共享层和个性化层
    """
    
    def __init__(self, shared_layers: Optional[Set[str]] = None):
        self.shared_layers = shared_layers or set()
        self.personal_layers: Set[str] = set()
        
        self._params: Dict[str, Any] = {}
        self._layer_configs: Dict[str, LayerConfig] = {}
    
    def set_params(self, params: Dict[str, Any]) -> None:
        """设置所有参数"""
        self._params = copy.deepcopy(params)
        
        # 自动识别层
        for key in params:
            if key not in self._layer_configs:
                is_shared = key in self.shared_layers
                self._layer_configs[key] = LayerConfig(
                    layer_name=key,
                    is_shared=is_shared
                )
            
            if key not in self.shared_layers:
                self.personal_layers.add(key)
    
    def get_params(self) -> Dict[str, Any]:
        """获取所有参数"""
        return copy.deepcopy(self._params)
    
    def get_shared_params(self) -> Dict[str, Any]:
        """获取共享参数"""
        return {
            k: copy.deepcopy(v)
            for k, v in self._params.items()
            if k in self.shared_layers
        }
    
    def get_personal_params(self) -> Dict[str, Any]:
        """获取个性化参数"""
        return {
            k: copy.deepcopy(v)
            for k, v in self._params.items()
            if k not in self.shared_layers
        }
    
    def update_shared_params(self, global_params: Dict[str, Any]) -> None:
        """更新共享参数"""
        for key, value in global_params.items():
            if key in self.shared_layers:
                self._params[key] = copy.deepcopy(value)
    
    def freeze_layer(self, layer_name: str) -> None:
        """冻结层"""
        if layer_name in self._layer_configs:
            self._layer_configs[layer_name].is_frozen = True
    
    def unfreeze_layer(self, layer_name: str) -> None:
        """解冻层"""
        if layer_name in self._layer_configs:
            self._layer_configs[layer_name].is_frozen = False
    
    def set_layer_learning_rate_scale(
        self,
        layer_name: str,
        scale: float
    ) -> None:
        """设置层学习率缩放"""
        if layer_name in self._layer_configs:
            self._layer_configs[layer_name].learning_rate_scale = scale
    
    def get_trainable_params(self) -> Dict[str, Any]:
        """获取可训练参数"""
        return {
            k: copy.deepcopy(v)
            for k, v in self._params.items()
            if k in self._layer_configs and not self._layer_configs[k].is_frozen
        }
    
    def get_layer_config(self, layer_name: str) -> Optional[LayerConfig]:
        """获取层配置"""
        return self._layer_configs.get(layer_name)


class AdapterLayer:
    """
    适配器层
    
    在预训练模型中插入小型适配器模块
    """
    
    def __init__(
        self,
        input_dim: int,
        adapter_dim: int,
        output_dim: int,
        layer_name: str = "adapter"
    ):
        self.layer_name = layer_name
        self.input_dim = input_dim
        self.adapter_dim = adapter_dim
        self.output_dim = output_dim
        
        # 初始化适配器参数
        # 降维矩阵
        self.down_weight: List[List[float]] = [
            [self._init_weight() for _ in range(input_dim)]
            for _ in range(adapter_dim)
        ]
        self.down_bias: List[float] = [0.0] * adapter_dim
        
        # 升维矩阵
        self.up_weight: List[List[float]] = [
            [self._init_weight() for _ in range(adapter_dim)]
            for _ in range(output_dim)
        ]
        self.up_bias: List[float] = [0.0] * output_dim
    
    def _init_weight(self) -> float:
        """初始化权重"""
        import random
        return random.gauss(0, 0.01)
    
    def forward(self, x: List[float]) -> List[float]:
        """前向传播"""
        # 降维
        h = [
            sum(w * xi for w, xi in zip(row, x)) + b
            for row, b in zip(self.down_weight, self.down_bias)
        ]
        
        # 激活函数 (ReLU)
        h = [max(0, hi) for hi in h]
        
        # 升维
        out = [
            sum(w * hi for w, hi in zip(row, h)) + b
            for row, b in zip(self.up_weight, self.up_bias)
        ]
        
        return out
    
    def get_params(self) -> Dict[str, Any]:
        """获取参数"""
        return {
            f"{self.layer_name}.down_weight": self.down_weight,
            f"{self.layer_name}.down_bias": self.down_bias,
            f"{self.layer_name}.up_weight": self.up_weight,
            f"{self.layer_name}.up_bias": self.up_bias
        }
    
    def get_param_count(self) -> int:
        """获取参数数量"""
        return (
            self.adapter_dim * self.input_dim +  # down_weight
            self.adapter_dim +  # down_bias
            self.output_dim * self.adapter_dim +  # up_weight
            self.output_dim  # up_bias
        )


class LocalFineTuner:
    """
    本地微调器
    
    在全局模型基础上进行本地个性化微调
    """
    
    def __init__(
        self,
        fine_tune_epochs: int = 5,
        fine_tune_lr: float = 0.001,
        warmup_epochs: int = 1,
        early_stopping_patience: int = 3
    ):
        self.fine_tune_epochs = fine_tune_epochs
        self.fine_tune_lr = fine_tune_lr
        self.warmup_epochs = warmup_epochs
        self.early_stopping_patience = early_stopping_patience
        
        self._best_loss: float = float('inf')
        self._patience_counter: int = 0
        self._fine_tune_history: List[Dict[str, Any]] = []
    
    def fine_tune(
        self,
        model: PersonalizedModel,
        train_data: List[Tuple[Any, Any]],
        val_data: Optional[List[Tuple[Any, Any]]] = None
    ) -> Dict[str, Any]:
        """
        执行本地微调
        
        Args:
            model: 个性化模型
            train_data: 训练数据
            val_data: 验证数据
        
        Returns:
            微调结果
        """
        start_time = datetime.now().timestamp()
        
        for epoch in range(self.fine_tune_epochs):
            # 学习率预热
            if epoch < self.warmup_epochs:
                lr = self.fine_tune_lr * (epoch + 1) / self.warmup_epochs
            else:
                lr = self.fine_tune_lr
            
            # 训练一个epoch（简化）
            train_loss = self._train_epoch(model, train_data, lr)
            
            # 验证
            val_loss = train_loss
            if val_data:
                val_loss = self._evaluate(model, val_data)
            
            # 记录历史
            self._fine_tune_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': lr
            })
            
            # 早停检查
            if val_loss < self._best_loss:
                self._best_loss = val_loss
                self._patience_counter = 0
            else:
                self._patience_counter += 1
            
            if self._patience_counter >= self.early_stopping_patience:
                break
        
        end_time = datetime.now().timestamp()
        
        return {
            'epochs_trained': epoch + 1,
            'best_loss': self._best_loss,
            'final_loss': val_loss,
            'fine_tune_time': end_time - start_time,
            'history': self._fine_tune_history
        }
    
    def _train_epoch(
        self,
        model: PersonalizedModel,
        data: List[Tuple[Any, Any]],
        lr: float
    ) -> float:
        """训练一个epoch"""
        # 简化实现
        import random
        return random.uniform(0.1, 0.5)
    
    def _evaluate(
        self,
        model: PersonalizedModel,
        data: List[Tuple[Any, Any]]
    ) -> float:
        """评估模型"""
        import random
        return random.uniform(0.1, 0.4)


class PersonalizationManager:
    """
    个性化管理器
    
    管理客户端的个性化配置和训练
    """
    
    def __init__(
        self,
        strategy: PersonalizationStrategy = PersonalizationStrategy.LOCAL_FINE_TUNING,
        shared_layers: Optional[Set[str]] = None
    ):
        self.strategy = strategy
        self.shared_layers = shared_layers or set()
        
        self._models: Dict[str, PersonalizedModel] = {}
        self._adapters: Dict[str, Dict[str, AdapterLayer]] = {}
        self._fine_tuner = LocalFineTuner()
    
    def register_client(
        self,
        client_id: str,
        global_params: Dict[str, Any]
    ) -> None:
        """注册客户端"""
        model = PersonalizedModel(self.shared_layers)
        model.set_params(global_params)
        self._models[client_id] = model
    
    def get_model(self, client_id: str) -> Optional[PersonalizedModel]:
        """获取客户端模型"""
        return self._models.get(client_id)
    
    def update_global_params(
        self,
        client_id: str,
        global_params: Dict[str, Any]
    ) -> None:
        """更新客户端的共享参数"""
        if client_id in self._models:
            self._models[client_id].update_shared_params(global_params)
    
    def add_adapter(
        self,
        client_id: str,
        layer_name: str,
        input_dim: int,
        adapter_dim: int,
        output_dim: int
    ) -> AdapterLayer:
        """添加适配器层"""
        adapter = AdapterLayer(
            input_dim=input_dim,
            adapter_dim=adapter_dim,
            output_dim=output_dim,
            layer_name=layer_name
        )
        
        if client_id not in self._adapters:
            self._adapters[client_id] = {}
        
        self._adapters[client_id][layer_name] = adapter
        return adapter
    
    def get_adapters(self, client_id: str) -> Dict[str, AdapterLayer]:
        """获取客户端的适配器"""
        return self._adapters.get(client_id, {})
    
    def personalize(
        self,
        client_id: str,
        train_data: List[Tuple[Any, Any]],
        val_data: Optional[List[Tuple[Any, Any]]] = None
    ) -> Dict[str, Any]:
        """
        执行个性化
        
        Args:
            client_id: 客户端ID
            train_data: 训练数据
            val_data: 验证数据
        
        Returns:
            个性化结果
        """
        model = self._models.get(client_id)
        if model is None:
            return {'error': 'Client not registered'}
        
        if self.strategy == PersonalizationStrategy.LOCAL_FINE_TUNING:
            return self._fine_tuner.fine_tune(model, train_data, val_data)
        
        elif self.strategy == PersonalizationStrategy.ADAPTER:
            # 只训练适配器参数
            return self._train_adapters(client_id, train_data)
        
        else:
            return {'error': f'Unsupported strategy: {self.strategy}'}
    
    def _train_adapters(
        self,
        client_id: str,
        data: List[Tuple[Any, Any]]
    ) -> Dict[str, Any]:
        """训练适配器"""
        adapters = self._adapters.get(client_id, {})
        
        # 简化实现
        return {
            'trained_adapters': list(adapters.keys()),
            'total_params': sum(a.get_param_count() for a in adapters.values())
        }
    
    def get_personal_params(self, client_id: str) -> Dict[str, Any]:
        """获取个性化参数"""
        model = self._models.get(client_id)
        if model:
            return model.get_personal_params()
        return {}
    
    def get_shared_params(self, client_id: str) -> Dict[str, Any]:
        """获取共享参数"""
        model = self._models.get(client_id)
        if model:
            return model.get_shared_params()
        return {}
    
    def compute_personalization_score(
        self,
        client_id: str,
        global_params: Dict[str, Any]
    ) -> float:
        """
        计算个性化分数
        
        衡量本地模型与全局模型的差异程度
        """
        model = self._models.get(client_id)
        if model is None:
            return 0.0
        
        personal_params = model.get_personal_params()
        if not personal_params:
            return 0.0
        
        # 计算参数差异
        total_diff = 0.0
        count = 0
        
        for key, local_val in personal_params.items():
            global_val = global_params.get(key)
            if global_val is None:
                continue
            
            if isinstance(local_val, (int, float)) and isinstance(global_val, (int, float)):
                total_diff += abs(local_val - global_val)
                count += 1
            elif isinstance(local_val, list) and isinstance(global_val, list):
                for lv, gv in zip(local_val, global_val):
                    if isinstance(lv, (int, float)) and isinstance(gv, (int, float)):
                        total_diff += abs(lv - gv)
                        count += 1
        
        return total_diff / count if count > 0 else 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_personal_params = 0
        total_adapter_params = 0
        
        for model in self._models.values():
            params = model.get_personal_params()
            total_personal_params += len(params)
        
        for adapters in self._adapters.values():
            for adapter in adapters.values():
                total_adapter_params += adapter.get_param_count()
        
        return {
            'strategy': self.strategy.value,
            'num_clients': len(self._models),
            'shared_layers': list(self.shared_layers),
            'total_personal_params': total_personal_params,
            'total_adapter_params': total_adapter_params
        }
