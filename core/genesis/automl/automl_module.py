"""
自动机器学习模块 - AutoML Module
实现超参数优化、神经架构搜索、自动特征工程、模型选择等功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
import json
import os
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from functools import partial
import warnings

# ==================== 搜索空间 ====================

@dataclass
class HyperParameter:
    """超参数定义"""
    name: str
    type: str  # 'int', 'float', 'categorical', 'bool'
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    log: bool = False
    default: Optional[Any] = None


class SearchSpace:
    """搜索空间"""
    
    def __init__(self):
        self.params: Dict[str, HyperParameter] = {}
    
    def add_int(
        self,
        name: str,
        low: int,
        high: int,
        log: bool = False,
        default: Optional[int] = None,
    ):
        """添加整数参数"""
        self.params[name] = HyperParameter(
            name=name,
            type='int',
            low=low,
            high=high,
            log=log,
            default=default,
        )
        return self
    
    def add_float(
        self,
        name: str,
        low: float,
        high: float,
        log: bool = False,
        default: Optional[float] = None,
    ):
        """添加浮点参数"""
        self.params[name] = HyperParameter(
            name=name,
            type='float',
            low=low,
            high=high,
            log=log,
            default=default,
        )
        return self
    
    def add_categorical(
        self,
        name: str,
        choices: List[Any],
        default: Optional[Any] = None,
    ):
        """添加分类参数"""
        self.params[name] = HyperParameter(
            name=name,
            type='categorical',
            choices=choices,
            default=default,
        )
        return self
    
    def add_bool(self, name: str, default: bool = False):
        """添加布尔参数"""
        self.params[name] = HyperParameter(
            name=name,
            type='bool',
            choices=[True, False],
            default=default,
        )
        return self
    
    def sample(self) -> Dict[str, Any]:
        """随机采样"""
        config = {}
        for name, param in self.params.items():
            if param.type == 'int':
                if param.log:
                    log_low = np.log(param.low)
                    log_high = np.log(param.high)
                    value = int(np.exp(random.uniform(log_low, log_high)))
                else:
                    value = random.randint(int(param.low), int(param.high))
            elif param.type == 'float':
                if param.log:
                    log_low = np.log(param.low)
                    log_high = np.log(param.high)
                    value = np.exp(random.uniform(log_low, log_high))
                else:
                    value = random.uniform(param.low, param.high)
            elif param.type == 'categorical':
                value = random.choice(param.choices)
            elif param.type == 'bool':
                value = random.choice([True, False])
            config[name] = value
        return config
    
    def get_default(self) -> Dict[str, Any]:
        """获取默认配置"""
        config = {}
        for name, param in self.params.items():
            if param.default is not None:
                config[name] = param.default
            else:
                if param.type == 'int':
                    config[name] = int((param.low + param.high) / 2)
                elif param.type == 'float':
                    config[name] = (param.low + param.high) / 2
                elif param.type == 'categorical':
                    config[name] = param.choices[0]
                elif param.type == 'bool':
                    config[name] = False
        return config


# ==================== 超参数优化 ====================

@dataclass
class Trial:
    """试验记录"""
    trial_id: int
    config: Dict[str, Any]
    score: Optional[float] = None
    status: str = 'pending'  # 'pending', 'running', 'completed', 'failed'
    info: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class HyperparameterOptimizer:
    """超参数优化器基类"""
    
    def __init__(
        self,
        search_space: SearchSpace,
        objective_fn: Callable,
        direction: str = 'minimize',  # 'minimize' or 'maximize'
        max_trials: int = 100,
    ):
        self.search_space = search_space
        self.objective_fn = objective_fn
        self.direction = direction
        self.max_trials = max_trials
        
        self.trials: List[Trial] = []
        self.best_trial: Optional[Trial] = None
        self.trial_id = 0
    
    def _is_better(self, score1: float, score2: float) -> bool:
        """比较分数"""
        if self.direction == 'minimize':
            return score1 < score2
        else:
            return score1 > score2
    
    def _update_best(self, trial: Trial):
        """更新最佳试验"""
        if trial.score is None:
            return
        
        if self.best_trial is None:
            self.best_trial = trial
        elif self._is_better(trial.score, self.best_trial.score):
            self.best_trial = trial
    
    def run(self) -> Trial:
        """运行优化 - 默认使用随机搜索实现"""
        warnings.warn(
            "HyperparameterOptimizer.run() called directly; "
            "consider using RandomSearch, BayesianOptimizer, or TPEOptimizer instead.",
            UserWarning,
        )
        for _ in range(self.max_trials):
            config = self.search_space.sample()
            trial = Trial(
                trial_id=self.trial_id,
                config=config,
                status='running',
                start_time=time.time(),
            )
            self.trials.append(trial)
            self.trial_id += 1
            try:
                score = self.objective_fn(config)
                trial.score = score
                trial.status = 'completed'
                trial.end_time = time.time()
                self._update_best(trial)
            except Exception as e:
                trial.status = 'failed'
                trial.info['error'] = str(e)
                trial.end_time = time.time()
        return self.best_trial


class RandomSearch(HyperparameterOptimizer):
    """随机搜索"""
    
    def run(self) -> Trial:
        """运行随机搜索"""
        for _ in range(self.max_trials):
            # 采样配置
            config = self.search_space.sample()
            
            # 创建试验
            trial = Trial(
                trial_id=self.trial_id,
                config=config,
                status='running',
                start_time=time.time(),
            )
            self.trials.append(trial)
            self.trial_id += 1
            
            # 评估
            try:
                score = self.objective_fn(config)
                trial.score = score
                trial.status = 'completed'
                trial.end_time = time.time()
                self._update_best(trial)
            except Exception as e:
                trial.status = 'failed'
                trial.info['error'] = str(e)
                trial.end_time = time.time()
        
        return self.best_trial


class GridSearch(HyperparameterOptimizer):
    """网格搜索"""
    
    def __init__(
        self,
        search_space: SearchSpace,
        objective_fn: Callable,
        direction: str = 'minimize',
        num_points: int = 5,
    ):
        super().__init__(search_space, objective_fn, direction)
        self.num_points = num_points
    
    def _generate_grid(self) -> List[Dict[str, Any]]:
        """生成网格点"""
        param_values = {}
        
        for name, param in self.search_space.params.items():
            if param.type == 'int':
                values = np.linspace(param.low, param.high, self.num_points, dtype=int).tolist()
            elif param.type == 'float':
                values = np.linspace(param.low, param.high, self.num_points).tolist()
            elif param.type == 'categorical':
                values = param.choices
            elif param.type == 'bool':
                values = [True, False]
            param_values[name] = values
        
        # 生成所有组合
        configs = [{}]
        for name, values in param_values.items():
            new_configs = []
            for config in configs:
                for value in values:
                    new_config = config.copy()
                    new_config[name] = value
                    new_configs.append(new_config)
            configs = new_configs
        
        return configs
    
    def run(self) -> Trial:
        """运行网格搜索"""
        configs = self._generate_grid()
        
        for config in configs[:self.max_trials]:
            trial = Trial(
                trial_id=self.trial_id,
                config=config,
                status='running',
                start_time=time.time(),
            )
            self.trials.append(trial)
            self.trial_id += 1
            
            try:
                score = self.objective_fn(config)
                trial.score = score
                trial.status = 'completed'
                trial.end_time = time.time()
                self._update_best(trial)
            except Exception as e:
                trial.status = 'failed'
                trial.info['error'] = str(e)
                trial.end_time = time.time()
        
        return self.best_trial


class BayesianOptimizer(HyperparameterOptimizer):
    """贝叶斯优化"""
    
    def __init__(
        self,
        search_space: SearchSpace,
        objective_fn: Callable,
        direction: str = 'minimize',
        max_trials: int = 100,
        n_startup_trials: int = 5,
        acquisition: str = 'ei',  # 'ei', 'pi', 'ucb'
        kappa: float = 2.0,
    ):
        super().__init__(search_space, objective_fn, direction, max_trials)
        self.n_startup_trials = n_startup_trials
        self.acquisition = acquisition
        self.kappa = kappa
        
        # 高斯过程代理模型
        self.gp_x = []
        self.gp_y = []
    
    def _to_vector(self, config: Dict[str, Any]) -> np.ndarray:
        """配置转换为向量"""
        vector = []
        for name, param in self.search_space.params.items():
            value = config[name]
            
            if param.type == 'int':
                if param.log:
                    normalized = (np.log(value) - np.log(param.low)) / (np.log(param.high) - np.log(param.low))
                else:
                    normalized = (value - param.low) / (param.high - param.low)
            elif param.type == 'float':
                if param.log:
                    normalized = (np.log(value) - np.log(param.low)) / (np.log(param.high) - np.log(param.low))
                else:
                    normalized = (value - param.low) / (param.high - param.low)
            elif param.type == 'categorical':
                # One-hot encoding
                for i, choice in enumerate(param.choices):
                    vector.append(1.0 if value == choice else 0.0)
                continue
            elif param.type == 'bool':
                normalized = 1.0 if value else 0.0
            
            vector.append(normalized)
        
        return np.array(vector)
    
    def _from_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        """向量转换为配置"""
        config = {}
        idx = 0
        
        for name, param in self.search_space.params.items():
            if param.type == 'categorical':
                # One-hot decoding
                one_hot = vector[idx:idx + len(param.choices)]
                choice_idx = np.argmax(one_hot)
                config[name] = param.choices[choice_idx]
                idx += len(param.choices)
            else:
                normalized = vector[idx]
                idx += 1
                
                if param.type == 'int':
                    if param.log:
                        log_value = np.log(param.low) + normalized * (np.log(param.high) - np.log(param.low))
                        value = int(np.exp(log_value))
                    else:
                        value = int(param.low + normalized * (param.high - param.low))
                elif param.type == 'float':
                    if param.log:
                        log_value = np.log(param.low) + normalized * (np.log(param.high) - np.log(param.low))
                        value = np.exp(log_value)
                    else:
                        value = param.low + normalized * (param.high - param.low)
                elif param.type == 'bool':
                    value = normalized > 0.5
                
                config[name] = value
        
        return config
    
    def _gp_predict(
        self,
        x: np.ndarray,
        x_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Tuple[float, float]:
        """高斯过程预测"""
        if len(x_train) == 0:
            return 0.0, 1.0
        
        # RBF核
        def kernel(x1, x2, l=1.0, sigma_f=1.0):
            return sigma_f ** 2 * np.exp(-0.5 * np.sum((x1 - x2) ** 2) / l ** 2)
        
        # 计算核矩阵
        K = np.array([[kernel(xi, xj) for xj in x_train] for xi in x_train])
        K += 1e-6 * np.eye(len(K))  # 数值稳定性
        
        K_s = np.array([kernel(x, xi) for xi in x_train])
        K_ss = kernel(x, x)
        
        # 预测
        K_inv = np.linalg.inv(K)
        mu = K_s @ K_inv @ y_train
        sigma2 = K_ss - K_s @ K_inv @ K_s
        
        return mu, np.sqrt(max(sigma2, 1e-6))
    
    def _acquisition_function(
        self,
        x: np.ndarray,
        best_y: float,
    ) -> float:
        """采集函数"""
        mu, sigma = self._gp_predict(x, np.array(self.gp_x), np.array(self.gp_y))
        
        if self.acquisition == 'ei':
            # Expected Improvement
            if sigma == 0:
                return 0.0
            z = (best_y - mu) / sigma
            ei = (best_y - mu) * self._norm_cdf(z) + sigma * self._norm_pdf(z)
            return ei
        
        elif self.acquisition == 'pi':
            # Probability of Improvement
            if sigma == 0:
                return 0.0
            z = (best_y - mu) / sigma
            return self._norm_cdf(z)
        
        elif self.acquisition == 'ucb':
            # Upper Confidence Bound
            return mu - self.kappa * sigma  # 最小化
        
        return 0.0
    
    def _norm_pdf(self, x: float) -> float:
        """标准正态PDF"""
        return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
    
    def _norm_cdf(self, x: float) -> float:
        """标准正态CDF"""
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    
    def _optimize_acquisition(self) -> np.ndarray:
        """优化采集函数"""
        best_x = None
        best_acq = float('-inf')
        
        # 随机搜索
        for _ in range(100):
            x = np.random.rand(len(self._to_vector(self.search_space.sample())))
            acq = self._acquisition_function(x, min(self.gp_y) if self.gp_y else 0)
            
            if acq > best_acq:
                best_acq = acq
                best_x = x
        
        return best_x
    
    def run(self) -> Trial:
        """运行贝叶斯优化"""
        for i in range(self.max_trials):
            if i < self.n_startup_trials or len(self.gp_x) == 0:
                # 随机采样
                config = self.search_space.sample()
            else:
                # 贝叶斯建议
                x = self._optimize_acquisition()
                config = self._from_vector(x)
            
            # 创建试验
            trial = Trial(
                trial_id=self.trial_id,
                config=config,
                status='running',
                start_time=time.time(),
            )
            self.trials.append(trial)
            self.trial_id += 1
            
            # 评估
            try:
                score = self.objective_fn(config)
                trial.score = score
                trial.status = 'completed'
                trial.end_time = time.time()
                
                # 更新GP
                self.gp_x.append(self._to_vector(config).tolist())
                self.gp_y.append(score)
                
                self._update_best(trial)
            except Exception as e:
                trial.status = 'failed'
                trial.info['error'] = str(e)
                trial.end_time = time.time()
        
        return self.best_trial


class TPEOptimizer(HyperparameterOptimizer):
    """TPE (Tree-structured Parzen Estimator) 优化器"""
    
    def __init__(
        self,
        search_space: SearchSpace,
        objective_fn: Callable,
        direction: str = 'minimize',
        max_trials: int = 100,
        n_startup_trials: int = 10,
        gamma: float = 0.25,
    ):
        super().__init__(search_space, objective_fn, direction, max_trials)
        self.n_startup_trials = n_startup_trials
        self.gamma = gamma
    
    def _split_trials(self) -> Tuple[List[Trial], List[Trial]]:
        """分割试验为好和坏两组"""
        completed = [t for t in self.trials if t.status == 'completed' and t.score is not None]
        completed.sort(key=lambda t: t.score, reverse=(self.direction == 'maximize'))
        
        n_good = max(1, int(len(completed) * self.gamma))
        good = completed[:n_good]
        bad = completed[n_good:]
        
        return good, bad
    
    def _estimate_density(
        self,
        values: List[Any],
        param: HyperParameter,
    ) -> Dict:
        """估计密度"""
        if param.type in ['int', 'float']:
            values = [float(v) for v in values]
            mean = np.mean(values)
            std = np.std(values) + 1e-6
            return {'type': 'normal', 'mean': mean, 'std': std}
        
        elif param.type == 'categorical':
            counts = defaultdict(int)
            for v in values:
                counts[v] += 1
            total = len(values)
            probs = {c: counts[c] / total for c in param.choices}
            return {'type': 'categorical', 'probs': probs}
        
        elif param.type == 'bool':
            p_true = sum(1 for v in values if v) / len(values)
            return {'type': 'bool', 'p_true': p_true}
    
    def _sample_from_density(self, density: Dict, param: HyperParameter) -> Any:
        """从密度采样"""
        if density['type'] == 'normal':
            value = np.random.normal(density['mean'], density['std'])
            
            if param.type == 'int':
                value = int(round(value))
                value = max(int(param.low), min(int(param.high), value))
            else:
                value = max(param.low, min(param.high, value))
            
            return value
        
        elif density['type'] == 'categorical':
            probs = density['probs']
            choices = list(probs.keys())
            p = [probs[c] for c in choices]
            return np.random.choice(choices, p=p)
        
        elif density['type'] == 'bool':
            return np.random.random() < density['p_true']
    
    def _compute_ratio(
        self,
        value: Any,
        good_density: Dict,
        bad_density: Dict,
        param: HyperParameter,
    ) -> float:
        """计算密度比"""
        if good_density['type'] == 'normal':
            good_pdf = np.exp(-0.5 * ((value - good_density['mean']) / good_density['std']) ** 2) / (good_density['std'] * np.sqrt(2 * np.pi))
            bad_pdf = np.exp(-0.5 * ((value - bad_density['mean']) / bad_density['std']) ** 2) / (bad_density['std'] * np.sqrt(2 * np.pi))
            return good_pdf / (bad_pdf + 1e-10)
        
        elif good_density['type'] == 'categorical':
            return good_density['probs'].get(value, 1e-10) / (bad_density['probs'].get(value, 1e-10) + 1e-10)
        
        elif good_density['type'] == 'bool':
            p_good = good_density['p_true'] if value else 1 - good_density['p_true']
            p_bad = bad_density['p_true'] if value else 1 - bad_density['p_true']
            return p_good / (p_bad + 1e-10)
        
        return 1.0
    
    def run(self) -> Trial:
        """运行TPE优化"""
        for i in range(self.max_trials):
            if i < self.n_startup_trials:
                config = self.search_space.sample()
            else:
                good, bad = self._split_trials()
                
                if not good:
                    config = self.search_space.sample()
                else:
                    config = {}
                    for name, param in self.search_space.params.items():
                        good_values = [t.config[name] for t in good]
                        bad_values = [t.config[name] for t in bad] if bad else good_values
                        
                        good_density = self._estimate_density(good_values, param)
                        bad_density = self._estimate_density(bad_values, param)
                        
                        # 采样候选并选择最佳
                        candidates = [
                            self._sample_from_density(good_density, param)
                            for _ in range(10)
                        ]
                        
                        best_candidate = max(
                            candidates,
                            key=lambda c: self._compute_ratio(c, good_density, bad_density, param),
                        )
                        config[name] = best_candidate
            
            trial = Trial(
                trial_id=self.trial_id,
                config=config,
                status='running',
                start_time=time.time(),
            )
            self.trials.append(trial)
            self.trial_id += 1
            
            try:
                score = self.objective_fn(config)
                trial.score = score
                trial.status = 'completed'
                trial.end_time = time.time()
                self._update_best(trial)
            except Exception as e:
                trial.status = 'failed'
                trial.info['error'] = str(e)
                trial.end_time = time.time()
        
        return self.best_trial


# ==================== 神经架构搜索 ====================

@dataclass
class ArchitectureConfig:
    """架构配置"""
    layers: List[Dict[str, Any]]
    connections: List[Tuple[int, int]]
    operations: Dict[str, Callable]


class NASSearchSpace:
    """神经架构搜索空间"""
    
    def __init__(self):
        self.operations = {
            'conv3x3': lambda c: nn.Conv2d(c, c, 3, padding=1),
            'conv5x5': lambda c: nn.Conv2d(c, c, 5, padding=2),
            'depthwise_conv': lambda c: nn.Sequential(
                nn.Conv2d(c, c, 3, padding=1, groups=c),
                nn.Conv2d(c, c, 1),
            ),
            'avg_pool': lambda c: nn.AvgPool2d(3, stride=1, padding=1),
            'max_pool': lambda c: nn.MaxPool2d(3, stride=1, padding=1),
            'identity': lambda c: nn.Identity(),
            'zero': lambda c: ZeroLayer(c),
            'relu_conv3x3': lambda c: nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(c, c, 3, padding=1),
            ),
            'batchnorm': lambda c: nn.BatchNorm2d(c),
        }
        self.num_layers = 8
        self.num_operations = len(self.operations)
    
    def sample(self) -> ArchitectureConfig:
        """采样架构"""
        layers = []
        for i in range(self.num_layers):
            op_name = random.choice(list(self.operations.keys()))
            layers.append({'operation': op_name, 'channels': 64})
        
        connections = []
        for i in range(1, self.num_layers):
            # 连接到前面的层
            prev = random.randint(0, i - 1)
            connections.append((prev, i))
        
        return ArchitectureConfig(
            layers=layers,
            connections=connections,
            operations=self.operations,
        )
    
    def build_model(
        self,
        config: ArchitectureConfig,
        in_channels: int = 3,
        num_classes: int = 10,
    ) -> nn.Module:
        """构建模型"""
        return NASModel(config, in_channels, num_classes)


class ZeroLayer(nn.Module):
    """零层"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class NASModel(nn.Module):
    """NAS模型"""
    
    def __init__(
        self,
        config: ArchitectureConfig,
        in_channels: int = 3,
        num_classes: int = 10,
    ):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # 构建层
        self.layers = nn.ModuleList()
        for layer_config in config.layers:
            op_name = layer_config['operation']
            channels = layer_config['channels']
            op = config.operations[op_name](channels)
            self.layers.append(op)
        
        self.connections = config.connections
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        
        # 存储中间结果
        states = [x]
        
        for i, layer in enumerate(self.layers):
            # 收集输入
            inputs = []
            for src, dst in self.connections:
                if dst == i:
                    inputs.append(states[src])
            
            if inputs:
                x = sum(inputs)
            else:
                x = states[-1]
            
            x = layer(x)
            states.append(x)
        
        x = self.head(x)
        return x


class NASEvolver:
    """神经架构搜索进化器"""
    
    def __init__(
        self,
        search_space: NASSearchSpace,
        evaluate_fn: Callable,
        population_size: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.5,
    ):
        self.search_space = search_space
        self.evaluate_fn = evaluate_fn
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.population: List[Tuple[ArchitectureConfig, float]] = []
        self.best_arch: Optional[ArchitectureConfig] = None
        self.best_score: float = float('-inf')
    
    def mutate(self, arch: ArchitectureConfig) -> ArchitectureConfig:
        """变异架构"""
        new_layers = []
        for layer in arch.layers:
            if random.random() < self.mutation_rate:
                op_name = random.choice(list(self.search_space.operations.keys()))
                new_layers.append({'operation': op_name, 'channels': layer['channels']})
            else:
                new_layers.append(layer.copy())
        
        return ArchitectureConfig(
            layers=new_layers,
            connections=arch.connections,
            operations=arch.operations,
        )
    
    def crossover(
        self,
        arch1: ArchitectureConfig,
        arch2: ArchitectureConfig,
    ) -> ArchitectureConfig:
        """交叉架构"""
        new_layers = []
        for i in range(len(arch1.layers)):
            if random.random() < 0.5:
                new_layers.append(arch1.layers[i].copy())
            else:
                new_layers.append(arch2.layers[i].copy())
        
        return ArchitectureConfig(
            layers=new_layers,
            connections=arch1.connections,
            operations=arch1.operations,
        )
    
    def run(self, generations: int = 10) -> ArchitectureConfig:
        """运行进化"""
        # 初始化种群
        if not self.population:
            for _ in range(self.population_size):
                arch = self.search_space.sample()
                score = self.evaluate_fn(arch)
                self.population.append((arch, score))
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_arch = arch
        
        # 进化
        for gen in range(generations):
            # 排序
            self.population.sort(key=lambda x: x[1], reverse=True)
            
            # 选择
            parents = self.population[:self.population_size // 2]
            
            # 生成子代
            offspring = []
            while len(offspring) < self.population_size - len(parents):
                # 选择父代
                p1, p2 = random.sample(parents, 2)
                
                # 交叉
                if random.random() < self.crossover_rate:
                    child = self.crossover(p1[0], p2[0])
                else:
                    child = p1[0]
                
                # 变异
                child = self.mutate(child)
                
                # 评估
                score = self.evaluate_fn(child)
                offspring.append((child, score))
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_arch = child
            
            # 更新种群
            self.population = parents + offspring
        
        return self.best_arch


# ==================== 自动特征工程 ====================

class FeatureEngineer:
    """自动特征工程"""
    
    def __init__(self):
        self.transformers: List[Callable] = []
        self.feature_names: List[str] = []
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'FeatureEngineer':
        """拟合"""
        n_features = X.shape[1]
        
        # 原始特征
        self.feature_names = [f'x{i}' for i in range(n_features)]
        
        # 自动添加变换
        for i in range(n_features):
            # 平方
            self.transformers.append(lambda X, i=i: X[:, i:i+1] ** 2)
            self.feature_names.append(f'x{i}^2')
            
            # 平方根（正值）
            self.transformers.append(lambda X, i=i: np.sqrt(np.abs(X[:, i:i+1])))
            self.feature_names.append(f'sqrt(|x{i}|)')
            
            # 对数（正值）
            self.transformers.append(lambda X, i=i: np.log1p(np.abs(X[:, i:i+1])))
            self.feature_names.append(f'log1p(|x{i}|)')
        
        # 交互特征
        for i in range(min(5, n_features)):
            for j in range(i + 1, min(5, n_features)):
                self.transformers.append(lambda X, i=i, j=j: X[:, i:i+1] * X[:, j:j+1])
                self.feature_names.append(f'x{i}*x{j}')
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """变换"""
        features = [X]
        
        for transformer in self.transformers:
            features.append(transformer(X))
        
        return np.hstack(features)
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """拟合并变换"""
        return self.fit(X, y).transform(X)


class FeatureSelector:
    """特征选择器"""
    
    def __init__(
        self,
        method: str = 'mutual_info',  # 'mutual_info', 'correlation', 'variance'
        k: int = 10,
    ):
        self.method = method
        self.k = k
        self.selected_indices: Optional[List[int]] = None
    
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> 'FeatureSelector':
        """拟合"""
        n_features = X.shape[1]
        
        if self.method == 'variance':
            # 方差选择
            variances = np.var(X, axis=0)
            self.selected_indices = np.argsort(variances)[-self.k:].tolist()
        
        elif self.method == 'correlation' and y is not None:
            # 相关性选择
            correlations = []
            for i in range(n_features):
                corr = np.abs(np.corrcoef(X[:, i], y)[0, 1])
                correlations.append(corr if not np.isnan(corr) else 0)
            self.selected_indices = np.argsort(correlations)[-self.k:].tolist()
        
        elif self.method == 'mutual_info' and y is not None:
            # 互信息选择（简化实现）
            scores = []
            for i in range(n_features):
                score = self._estimate_mi(X[:, i], y)
                scores.append(score)
            self.selected_indices = np.argsort(scores)[-self.k:].tolist()
        
        else:
            self.selected_indices = list(range(min(self.k, n_features)))
        
        return self
    
    def _estimate_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        """估计互信息"""
        # 简化实现：使用相关系数的平方
        corr = np.corrcoef(x, y)[0, 1]
        return corr ** 2 if not np.isnan(corr) else 0
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """变换"""
        if self.selected_indices is None:
            return X
        return X[:, self.selected_indices]
    
    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """拟合并变换"""
        return self.fit(X, y).transform(X)


# ==================== 模型选择 ====================

class ModelSelector:
    """模型选择器"""
    
    def __init__(
        self,
        models: Dict[str, Callable],
        metric: str = 'accuracy',
        cv_folds: int = 5,
    ):
        self.models = models
        self.metric = metric
        self.cv_folds = cv_folds
        
        self.results: Dict[str, Dict] = {}
        self.best_model_name: Optional[str] = None
        self.best_score: float = float('-inf')
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Dict]:
        """评估所有模型"""
        n_samples = len(X)
        fold_size = n_samples // self.cv_folds
        
        for name, model_fn in self.models.items():
            scores = []
            
            for i in range(self.cv_folds):
                # 分割数据
                val_start = i * fold_size
                val_end = (i + 1) * fold_size
                
                X_train = np.concatenate([X[:val_start], X[val_end:]])
                y_train = np.concatenate([y[:val_start], y[val_end:]])
                X_val = X[val_start:val_end]
                y_val = y[val_start:val_end]
                
                # 训练和评估
                model = model_fn()
                model.fit(X_train, y_train)
                score = self._compute_score(model, X_val, y_val)
                scores.append(score)
            
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            self.results[name] = {
                'mean_score': mean_score,
                'std_score': std_score,
                'scores': scores,
            }
            
            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_model_name = name
        
        return self.results
    
    def _compute_score(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """计算分数"""
        y_pred = model.predict(X)
        
        if self.metric == 'accuracy':
            return np.mean(y_pred == y)
        elif self.metric == 'mse':
            return -np.mean((y_pred - y) ** 2)
        elif self.metric == 'mae':
            return -np.mean(np.abs(y_pred - y))
        else:
            return 0.0
    
    def get_best_model(self) -> Any:
        """获取最佳模型"""
        if self.best_model_name is None:
            return None
        return self.models[self.best_model_name]()


# ==================== 主函数 ====================

def main():
    """测试AutoML模块"""
    print("AutoML模块测试")
    
    # 测试搜索空间
    print("\n测试搜索空间...")
    space = SearchSpace()
    space.add_int('num_layers', 1, 5)
    space.add_float('learning_rate', 1e-5, 1e-1, log=True)
    space.add_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])
    space.add_bool('use_batchnorm')
    
    config = space.sample()
    print(f"Sampled config: {config}")
    
    # 测试随机搜索
    print("\n测试随机搜索...")
    def objective(cfg):
        return (cfg['num_layers'] - 3) ** 2 + np.log10(cfg['learning_rate'] + 1e-10) ** 2
    
    rs = RandomSearch(space, objective, max_trials=20)
    best = rs.run()
    print(f"Best config: {best.config}, Score: {best.score:.4f}")
    
    # 测试贝叶斯优化
    print("\n测试贝叶斯优化...")
    bo = BayesianOptimizer(space, objective, max_trials=20)
    best = bo.run()
    print(f"Best config: {best.config}, Score: {best.score:.4f}")
    
    # 测试特征工程
    print("\n测试特征工程...")
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    
    fe = FeatureEngineer()
    X_transformed = fe.fit_transform(X)
    print(f"Original features: {X.shape[1]}, Transformed features: {X_transformed.shape[1]}")
    
    # 测试特征选择
    print("\n测试特征选择...")
    fs = FeatureSelector(method='correlation', k=5)
    X_selected = fs.fit_transform(X, y)
    print(f"Selected features: {X_selected.shape[1]}")
    
    print("\nAutoML模块测试完成")


if __name__ == "__main__":
    main()
