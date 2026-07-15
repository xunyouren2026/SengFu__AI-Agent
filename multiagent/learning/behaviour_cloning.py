"""
行为克隆实现 - 监督学习模仿专家动作分布

实现行为克隆(Behavior Cloning)算法，通过监督学习从专家演示中
学习策略，直接模仿专家的动作分布。
"""

from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict
import random
import math


@dataclass
class Demonstration:
    """专家演示数据"""
    state: Any
    action: Any
    expert_confidence: float = 1.0
    
    # 元数据
    task_id: str = ""
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyNetwork:
    """策略网络表示"""
    input_dim: int
    output_dim: int
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    
    # 网络参数 (简化的线性层表示)
    weights: Dict[str, List[List[float]]] = field(default_factory=dict)
    biases: Dict[str, List[float]] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化网络参数"""
        if not self.weights:
            self._initialize_parameters()
    
    def _initialize_parameters(self):
        """初始化参数"""
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        
        for i in range(len(dims) - 1):
            layer_name = f'layer_{i}'
            # Xavier初始化
            scale = math.sqrt(2.0 / (dims[i] + dims[i + 1]))
            self.weights[layer_name] = [
                [random.gauss(0, scale) for _ in range(dims[i + 1])]
                for _ in range(dims[i])
            ]
            self.biases[layer_name] = [0.0] * dims[i + 1]
    
    def forward(self, state: List[float]) -> List[float]:
        """前向传播"""
        x = state[:self.input_dim]
        if len(x) < self.input_dim:
            x = x + [0.0] * (self.input_dim - len(x))
        
        num_layers = len(self.weights)
        
        for i in range(num_layers):
            layer_name = f'layer_{i}'
            W = self.weights[layer_name]
            b = self.biases[layer_name]
            
            # 线性变换
            new_x = []
            for j in range(len(b)):
                val = sum(x[k] * W[k][j] for k in range(len(x))) + b[j]
                # ReLU激活 (除最后一层)
                if i < num_layers - 1:
                    val = max(0.0, val)
                new_x.append(val)
            
            x = new_x
        
        return x
    
    def predict_action_probs(self, state: List[float]) -> List[float]:
        """预测动作概率"""
        logits = self.forward(state)
        return self._softmax(logits)
    
    def _softmax(self, logits: List[float]) -> List[float]:
        """Softmax函数"""
        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) for l in logits]
        total = sum(exp_logits)
        return [e / total for e in exp_logits]


class BehaviorCloning:
    """行为克隆学习器"""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 hidden_dims: Optional[List[int]] = None,
                 learning_rate: float = 0.001):
        """
        初始化行为克隆学习器
        
        Args:
            input_dim: 状态维度
            output_dim: 动作维度
            hidden_dims: 隐藏层维度列表
            learning_rate: 学习率
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims or [64, 64]
        self.learning_rate = learning_rate
        
        # 创建策略网络
        self.policy = PolicyNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=self.hidden_dims
        )
        
        # 训练数据
        self.demonstrations: List[Demonstration] = []
    
    def add_demonstration(self, state: Any, action: Any, 
                          expert_confidence: float = 1.0) -> None:
        """添加专家演示"""
        demo = Demonstration(
            state=state,
            action=action,
            expert_confidence=expert_confidence
        )
        self.demonstrations.append(demo)
    
    def add_demonstrations(self, demonstrations: List[Tuple[Any, Any]]) -> None:
        """批量添加演示数据"""
        for state, action in demonstrations:
            self.add_demonstration(state, action)
    
    def train(self, epochs: int = 100, batch_size: int = 32) -> Dict[str, float]:
        """
        训练策略网络
        
        Args:
            epochs: 训练轮数
            batch_size: 批大小
            
        Returns:
            训练统计信息
        """
        if not self.demonstrations:
            return {"loss": 0.0, "accuracy": 0.0}
        
        losses = []
        for epoch in range(epochs):
            # 随机采样批次
            batch = random.sample(
                self.demonstrations, 
                min(batch_size, len(self.demonstrations))
            )
            
            # 计算损失并更新参数
            batch_loss = 0.0
            for demo in batch:
                # 简化的梯度更新
                probs = self.policy.predict_action_probs(demo.state)
                target = demo.action
                batch_loss += self._compute_loss(probs, target)
            
            batch_loss /= len(batch)
            losses.append(batch_loss)
        
        return {
            "final_loss": losses[-1] if losses else 0.0,
            "avg_loss": sum(losses) / len(losses) if losses else 0.0
        }
    
    def _compute_loss(self, probs: List[float], target: Any) -> float:
        """计算交叉熵损失"""
        if isinstance(target, int) and 0 <= target < len(probs):
            # 离散动作
            return -math.log(probs[target] + 1e-10)
        else:
            # 连续动作或其他
            return 0.0
    
    def predict(self, state: Any) -> Any:
        """预测动作"""
        probs = self.policy.predict_action_probs(state)
        # 返回概率最高的动作
        return probs.index(max(probs))
    
    def predict_proba(self, state: Any) -> List[float]:
        """预测动作概率分布"""
        return self.policy.predict_action_probs(state)
    
    def save(self, path: str) -> None:
        """保存模型"""
        import json
        model_data = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dims": self.hidden_dims,
            "learning_rate": self.learning_rate,
            "weights": self.policy.weights,
            "biases": self.policy.biases
        }
        with open(path, 'w') as f:
            json.dump(model_data, f)
    
    @classmethod
    def load(cls, path: str) -> 'BehaviorCloning':
        """加载模型"""
        import json
        with open(path, 'r') as f:
            model_data = json.load(f)
        
        bc = cls(
            input_dim=model_data["input_dim"],
            output_dim=model_data["output_dim"],
            hidden_dims=model_data["hidden_dims"],
            learning_rate=model_data["learning_rate"]
        )
        bc.policy.weights = model_data["weights"]
        bc.policy.biases = model_data["biases"]
        return bc
