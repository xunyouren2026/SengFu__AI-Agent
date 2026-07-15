"""
Local AI System Module
本地AI系统 - 支持离线运行和联邦协调

This module implements a complete local AI system that can:
1. Load and run AI models locally
2. Manage local datasets
3. Perform local training
4. Participate in federated learning
5. Operate in a P2P network

Inspired by the feudal system:
- LocalNode = Castle (城堡)
- LocalDataset = Granary (粮仓)
- LocalModel = Royal Library (皇家图书馆)
- FederatedClient = Vassal (臣民)

Author: AGI Unified Framework
"""

import random
import hashlib
import time
import json
import math
import threading
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import defaultdict
from enum import Enum


# ============== 配置和枚举 ==============

class QuantizeLevel(Enum):
    """量化等级"""
    NONE = "none"    # 不量化
    INT8 = "int8"    # 8位整数
    INT4 = "int4"    # 4位整数


class ClientStatus(Enum):
    """客户端状态"""
    IDLE = "idle"
    TRAINING = "training"
    SUBMITTING = "submitting"
    WAITING = "waiting"
    ERROR = "error"


@dataclass
class LocalAIConfig:
    """本地AI配置"""
    model_path: str = "models/default"
    quantize_level: QuantizeLevel = QuantizeLevel.NONE
    max_memory_mb: int = 4096
    enable_gpu: bool = False
    federated_mode: bool = True
    peer_port: int = 8000
    
    batch_size: int = 32
    learning_rate: float = 0.001
    max_epochs: int = 10


@dataclass
class TrainingResult:
    """训练结果"""
    epochs: int
    final_loss: float
    metrics: Dict[str, float]
    duration: float  # 训练时长（秒）
    samples_processed: int
    checkpoint_path: Optional[str] = None


# ============== 本地模型管理 ==============

class LocalModel:
    """
    本地模型管理
    
    负责加载、存储和运行AI模型
    模拟实际的模型推理过程
    """
    
    def __init__(self, config: LocalAIConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._quantizer = None
        self._model_weights = None
        self._metadata = {}
        
        self._lock = threading.Lock()
        self._loaded = False
        
    def load_model(self, path: str = None) -> bool:
        """
        加载模型
        
        从磁盘加载模型到内存
        """
        if path is None:
            path = self.config.model_path
            
        with self._lock:
            # 模拟模型加载
            # 实际中会使用transformers等库
            self._model_weights = self._create_mock_weights()
            self._metadata = {
                'name': 'mock-model',
                'version': '1.0',
                'parameters': sum(w.size for w in self._model_weights),
                'loaded_at': time.time()
            }
            
            # 应用量化
            if self.config.quantize_level != QuantizeLevel.NONE:
                self._apply_quantization()
            
            self._loaded = True
            
        return True
    
    def _create_mock_weights(self) -> List:
        """创建模拟权重"""
        # 模拟一个小型神经网络的权重
        return [
            [[random.uniform(-1, 1) for _ in range(128)] for _ in range(64)],
            [random.uniform(-1, 1) for _ in range(128)],
            [[random.uniform(-1, 1) for _ in range(64)] for _ in range(128)],
            [random.uniform(-1, 1) for _ in range(64)]
        ]
    
    def _apply_quantization(self) -> None:
        """应用量化"""
        if self._model_weights is None:
            return
            
        if self.config.quantize_level == QuantizeLevel.INT8:
            # 8位量化
            for i, layer in enumerate(self._model_weights):
                if isinstance(layer, list) and isinstance(layer[0], list):
                    self._model_weights[i] = [
                        [int(x * 127) for x in row]
                        for row in layer
                    ]
                elif isinstance(layer, list):
                    self._model_weights[i] = [int(x * 127) for x in layer]
                    
        elif self.config.quantize_level == QuantizeLevel.INT4:
            # 4位量化（更激进）
            for i, layer in enumerate(self._model_weights):
                if isinstance(layer, list) and isinstance(layer[0], list):
                    self._model_weights[i] = [
                        [int(x * 7) for x in row]
                        for row in layer
                    ]
                elif isinstance(layer, list):
                    self._model_weights[i] = [int(x * 7) for x in layer]
    
    def unload_model(self) -> None:
        """卸载模型释放内存"""
        with self._lock:
            self._model = None
            self._model_weights = None
            self._tokenizer = None
            self._loaded = False
            
    def generate(self, prompt: str, **kwargs) -> str:
        """
        本地推理
        
        模拟模型生成过程
        """
        if not self._loaded:
            return "Error: Model not loaded"
        
        max_length = kwargs.get('max_length', 100)
        temperature = kwargs.get('temperature', 1.0)
        
        # 模拟生成过程
        words = ['token', 'output', 'generated', 'text', 'response']
        result = []
        
        for _ in range(min(max_length, 20)):
            result.append(random.choice(words))
        
        return ' '.join(result)
    
    def predict(self, inputs: List[float]) -> List[float]:
        """
        预测
        
        简单的矩阵运算模拟
        """
        if not self._loaded or self._model_weights is None:
            return []
        
        # 模拟前向传播
        x = inputs[:128]  # 截断到模型输入大小
        
        # 第一层
        w1 = self._model_weights[0]
        h1 = [0.0] * len(w1[0])
        for i in range(len(w1)):
            for j in range(len(w1[0])):
                h1[j] += w1[i][j] * x[i] if i < len(x) else 0
        
        # ReLU激活
        h1 = [max(0, v) for v in h1]
        
        # 添加偏置
        h1 = [v + self._model_weights[1][i] for i, v in enumerate(h1)]
        
        return h1[:10]  # 返回前10个值
    
    def get_weights(self) -> List[List]:
        """获取模型权重"""
        if not self._loaded:
            return []
        return self._model_weights
    
    def set_weights(self, weights: List[List]) -> None:
        """设置模型权重（用于联邦学习聚合）"""
        with self._lock:
            self._model_weights = weights
            self._loaded = True
    
    def _estimate_memory(self) -> int:
        """估算内存占用（MB）"""
        if self._model_weights is None:
            return 0
        
        total_elements = 0
        for layer in self._model_weights:
            if isinstance(layer, list) and isinstance(layer[0], list):
                total_elements += len(layer) * len(layer[0])
            elif isinstance(layer, list):
                total_elements += len(layer)
        
        # 假设每个float64占8字节
        bytes_used = total_elements * 8
        
        if self.config.quantize_level == QuantizeLevel.INT8:
            bytes_used //= 8
        elif self.config.quantize_level == QuantizeLevel.INT4:
            bytes_used //= 16
            
        return bytes_used // (1024 * 1024)
    
    def _offload_to_disk(self) -> bool:
        """部分卸载到磁盘"""
        # 模拟：实际会保存到磁盘
        return True
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._loaded
    
    def get_info(self) -> dict:
        """获取模型信息"""
        return {
            'loaded': self._loaded,
            'memory_mb': self._estimate_memory(),
            'quantize_level': self.config.quantize_level.value,
            'metadata': self._metadata.copy() if self._metadata else {}
        }


# ============== 本地数据集 ==============

class LocalDataset:
    """
    本地数据集管理
    
    模拟数据仓库，类似古代的"粮仓"
    - 存储本地数据
    - 支持隐私保护的查询和采样
    - 差分隐私导出
    """
    
    def __init__(self, privacy_level: int = 3):
        """
        Args:
            privacy_level: 隐私级别 (0-5, 越高越严格)
        """
        self._data: List[Dict] = []
        self._metadata: Dict = {
            'name': 'local-dataset',
            'created_at': time.time(),
            'records': 0
        }
        self._privacy_level = privacy_level
        
        # 统计缓存
        self._statistics_cache: Dict = {}
        self._cache_time: float = 0
        
        self._lock = threading.RLock()
        
    def add_record(self, record: Dict) -> None:
        """添加记录"""
        with self._lock:
            self._data.append(record)
            self._metadata['records'] = len(self._data)
            
            # 使统计缓存失效
            self._statistics_cache = {}
    
    def add_records(self, records: List[Dict]) -> None:
        """批量添加记录"""
        with self._lock:
            self._data.extend(records)
            self._metadata['records'] = len(self._data)
            self._statistics_cache = {}
    
    def query(self, filters: Dict = None) -> List[Dict]:
        """
        查询数据
        
        Args:
            filters: 过滤条件，如 {'field': 'value', 'min_value': 10}
        """
        with self._lock:
            if filters is None:
                return self._data.copy()
            
            results = self._data
            
            for field, condition in filters.items():
                if isinstance(condition, dict):
                    # 范围查询
                    if 'eq' in condition:
                        results = [r for r in results if r.get(field) == condition['eq']]
                    if 'min' in condition:
                        results = [r for r in results if r.get(field, 0) >= condition['min']]
                    if 'max' in condition:
                        results = [r for r in results if r.get(field, float('inf')) <= condition['max']]
                else:
                    # 精确匹配
                    results = [r for r in results if r.get(field) == condition]
            
            return results
    
    def export_sample(self, n: int, epsilon: float = 1.0) -> List[Dict]:
        """
        差分隐私采样
        
        Args:
            n: 采样数量
            epsilon: 隐私预算
        """
        with self._lock:
            if n >= len(self._data):
                return self._data.copy()
            
            # 添加拉普拉斯噪声进行差分隐私
            sensitivity = 1.0
            noise_scale = sensitivity / epsilon
            
            # 随机采样
            sampled = random.sample(self._data, n)
            
            # 可选：添加噪声
            if self._privacy_level > 2:
                for record in sampled:
                    for key, value in record.items():
                        if isinstance(value, (int, float)):
                            noise = random.laplace(0, noise_scale)
                            record[key] = value + noise
            
            return sampled
    
    def _compute_statistics(self) -> Dict:
        """计算统计摘要"""
        current_time = time.time()
        
        # 检查缓存
        if self._statistics_cache and (current_time - self._cache_time) < 60:
            return self._statistics_cache
        
        with self._lock:
            stats = {
                'total_records': len(self._data),
                'timestamp': current_time
            }
            
            if not self._data:
                return stats
            
            # 数值字段统计
            numeric_fields = defaultdict(list)
            for record in self._data:
                for key, value in record.items():
                    if isinstance(value, (int, float)):
                        numeric_fields[key].append(value)
            
            for field, values in numeric_fields.items():
                stats[f'{field}_mean'] = sum(values) / len(values)
                stats[f'{field}_min'] = min(values)
                stats[f'{field}_max'] = max(values)
            
            self._statistics_cache = stats
            self._cache_time = current_time
            
            return stats
    
    def _encrypt(self, data: Any, key: str = None) -> bytes:
        """
        简单加密（XOR模拟）
        
        实际应使用同态加密或更安全的方法
        """
        if key is None:
            key = "default-key"
        
        json_str = json.dumps(data, sort_keys=True)
        key_hash = hashlib.sha256(key.encode()).digest()
        
        encrypted = bytearray()
        for i, char in enumerate(json_str.encode()):
            encrypted.append(char ^ key_hash[i % len(key_hash)])
        
        return bytes(encrypted)
    
    def get_size(self) -> int:
        """获取数据集大小"""
        return len(self._data)
    
    def clear(self) -> None:
        """清空数据集"""
        with self._lock:
            self._data.clear()
            self._metadata['records'] = 0
            self._statistics_cache = {}


# ============== 本地训练器 ==============

class LocalTrainer:
    """
    本地训练器
    
    负责本地模型训练和梯度计算
    """
    
    def __init__(self, model: LocalModel, dataset: LocalDataset, 
                 config: LocalAIConfig):
        self._model = model
        self._dataset = dataset
        self.config = config
        
        self._optimizer = "adam"
        self._gradient_history: List[List[float]] = []
        
        self._lock = threading.Lock()
        
    def train_local(self, epochs: int = None, 
                   batch_size: int = None) -> TrainingResult:
        """
        本地训练
        
        执行本地数据上的训练
        """
        if epochs is None:
            epochs = self.config.max_epochs
        if batch_size is None:
            batch_size = self.config.batch_size
            
        start_time = time.time()
        
        if not self._model.is_loaded():
            self._model.load_model()
        
        total_loss = 0.0
        samples_processed = 0
        metrics = {}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            # 获取训练数据
            data = self._dataset.query()
            random.shuffle(data)
            
            # 分批训练
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                
                # 计算梯度
                grads = self._compute_gradient(batch)
                
                # 应用梯度
                self._apply_gradient(grads)
                
                # 模拟损失
                batch_loss = random.uniform(0.1, 2.0)
                epoch_loss += batch_loss
                batch_count += 1
            
            avg_loss = epoch_loss / max(batch_count, 1)
            total_loss += avg_loss
            samples_processed += len(data)
            
            # 模拟指标计算
            if epoch % 2 == 0:
                metrics[f'epoch_{epoch}_loss'] = avg_loss
                metrics[f'epoch_{epoch}_accuracy'] = random.uniform(0.7, 0.95)
        
        # 计算最终指标
        metrics['final_loss'] = total_loss / max(epochs, 1)
        metrics['avg_accuracy'] = random.uniform(0.8, 0.95)
        
        duration = time.time() - start_time
        
        # 保存检查点
        checkpoint_path = self._save_checkpoint(f"checkpoint_{int(time.time())}")
        
        return TrainingResult(
            epochs=epochs,
            final_loss=metrics['final_loss'],
            metrics=metrics,
            duration=duration,
            samples_processed=samples_processed,
            checkpoint_path=checkpoint_path
        )
    
    def _compute_gradient(self, batch: List[Dict]) -> List[List[float]]:
        """
        计算梯度
        
        模拟随机梯度下降
        """
        # 获取模型权重
        weights = self._model.get_weights()
        
        if not weights:
            return []
        
        # 模拟梯度计算
        gradients = []
        for layer in weights:
            if isinstance(layer, list) and isinstance(layer[0], list):
                grad = [[random.uniform(-0.1, 0.1) for _ in range(len(layer[0]))] 
                       for _ in range(len(layer))]
                gradients.append(grad)
            elif isinstance(layer, list):
                gradients.append([random.uniform(-0.1, 0.1) for _ in range(len(layer))])
        
        with self._lock:
            self._gradient_history.append(gradients)
            # 保持历史在合理范围内
            if len(self._gradient_history) > 100:
                self._gradient_history = self._gradient_history[-100:]
        
        return gradients
    
    def _apply_gradient(self, grads: List[List[float]]) -> None:
        """应用梯度更新模型"""
        weights = self._model.get_weights()
        
        if not weights or not grads:
            return
        
        # 简单的梯度下降更新
        lr = self.config.learning_rate
        
        for i, (weight, grad) in enumerate(zip(weights, grads)):
            if isinstance(weight, list) and isinstance(grad, list):
                if isinstance(weight[0], list) and isinstance(grad[0], list):
                    # 2D层
                    for j in range(min(len(weight), len(grad))):
                        for k in range(min(len(weight[0]), len(grad[0]))):
                            weight[j][k] -= lr * grad[j][k]
                else:
                    # 1D层
                    for j in range(min(len(weight), len(grad))):
                        weight[j] -= lr * grad[j]
    
    def _save_checkpoint(self, path: str) -> str:
        """保存检查点"""
        weights = self._model.get_weights()
        
        checkpoint = {
            'weights': weights,
            'timestamp': time.time(),
            'gradient_history_len': len(self._gradient_history)
        }
        
        # 实际会保存到文件
        return f"/checkpoints/{path}.pt"
    
    def _load_checkpoint(self, path: str) -> bool:
        """加载检查点"""
        # 实际会从文件加载
        return True
    
    def federated_aggregate(self, updates: List[Dict]) -> bool:
        """
        聚合联邦更新
        
        将来自其他节点的更新合并到本地模型
        """
        if not updates:
            return False
        
        current_weights = self._model.get_weights()
        
        if not current_weights:
            return False
        
        # 简单平均聚合
        aggregated = []
        
        for layer_idx in range(len(current_weights)):
            layer = current_weights[layer_idx]
            
            if isinstance(layer, list) and isinstance(layer[0], list):
                # 2D层
                new_layer = [
                    [0.0 for _ in range(len(layer[0]))]
                    for _ in range(len(layer))
                ]
                
                for update in updates:
                    if 'weights' in update and layer_idx < len(update['weights']):
                        update_layer = update['weights'][layer_idx]
                        for i in range(len(layer)):
                            for j in range(len(layer[0])):
                                new_layer[i][j] += update_layer[i][j] / len(updates)
                
                aggregated.append(new_layer)
            elif isinstance(layer, list):
                # 1D层
                new_layer = [0.0] * len(layer)
                
                for update in updates:
                    if 'weights' in update and layer_idx < len(update['weights']):
                        update_layer = update['weights'][layer_idx]
                        for j in range(len(layer)):
                            new_layer[j] += update_layer[j] / len(updates)
                
                aggregated.append(new_layer)
        
        # 应用聚合权重
        self._model.set_weights(aggregated)
        
        return True


# ============== 联邦客户端 ==============

class FederatedClient:
    """
    联邦客户端
    
    模拟联邦学习中的客户端节点（臣民）
    - 与服务器通信
    - 计算本地更新
    - 提交到服务器
    """
    
    def __init__(self, client_id: str, local_model: LocalModel,
                 local_dataset: LocalDataset, config: LocalAIConfig):
        self._client_id = client_id
        self._local_model = local_model
        self._local_dataset = local_dataset
        self.config = config
        
        self._server_url: str = ""
        self._round: int = 0
        self._is_active: bool = True
        self._status: ClientStatus = ClientStatus.IDLE
        
        # 更新历史
        self._update_history: List[Dict] = []
        
        # 锁
        self._lock = threading.Lock()
        
    def join_round(self, server_url: str, config: Dict = None) -> bool:
        """
        加入训练轮次
        
        向服务器注册参与当前轮次
        """
        with self._lock:
            self._server_url = server_url
            self._round += 1
            self._status = ClientStatus.WAITING
            
            # 模拟注册请求
            return True
    
    def compute_local_update(self, epochs: int = 1) -> Dict:
        """
        计算本地更新
        
        在本地数据上训练模型，返回更新
        """
        with self._lock:
            self._status = ClientStatus.TRAINING
        
        # 创建训练器
        trainer = LocalTrainer(
            self._local_model,
            self._local_dataset,
            self.config
        )
        
        # 执行训练
        result = trainer.train_local(epochs=epochs)
        
        # 构建更新
        update = {
            'client_id': self._client_id,
            'round': self._round,
            'weights': self._local_model.get_weights(),
            'num_samples': self._local_dataset.get_size(),
            'metrics': result.metrics,
            'timestamp': time.time()
        }
        
        with self._lock:
            self._update_history.append(update)
            self._status = ClientStatus.SUBMITTING
        
        return update
    
    def submit_update(self, update: Dict) -> bool:
        """
        提交更新到服务器
        
        模拟网络请求
        """
        # 验证更新
        if not self._validate_update(update):
            with self._lock:
                self._status = ClientStatus.ERROR
            return False
        
        # 模拟提交
        time.sleep(random.uniform(0.1, 0.5))
        
        with self._lock:
            self._status = ClientStatus.IDLE
        
        return True
    
    def receive_global_model(self, global_params: Dict) -> bool:
        """
        接收全局模型
        
        服务器聚合后下发的新模型
        """
        if 'weights' not in global_params:
            return False
        
        # 更新本地模型
        self._local_model.set_weights(global_params['weights'])
        
        return True
    
    def _validate_update(self, update: Dict) -> bool:
        """
        验证更新合法性
        
        检查更新是否有异常
        """
        required_fields = ['client_id', 'round', 'weights', 'num_samples']
        
        for field in required_fields:
            if field not in update:
                return False
        
        # 检查权重维度是否合理
        weights = update.get('weights', [])
        if not weights:
            return False
        
        # 检查数值范围
        for layer in weights:
            if isinstance(layer, list):
                for row in layer if isinstance(layer[0], list) else [layer]:
                    for val in (row if isinstance(row, list) else [row]):
                        if isinstance(val, (int, float)):
                            if math.isnan(val) or math.isinf(val):
                                return False
                            if abs(val) > 1000:  # 异常大值
                                return False
        
        return True
    
    def get_status(self) -> ClientStatus:
        """获取客户端状态"""
        return self._status
    
    def get_contribution(self) -> Dict:
        """获取贡献信息"""
        with self._lock:
            return {
                'client_id': self._client_id,
                'rounds_participated': len(self._update_history),
                'total_samples': sum(u.get('num_samples', 0) for u in self._update_history),
                'avg_loss': sum(u.get('metrics', {}).get('final_loss', 0) 
                              for u in self._update_history) / max(len(self._update_history), 1)
            }


# ============== 本地节点 ==============

class LocalNode:
    """
    完整本地节点
    
    整合所有组件，模拟一个完整的联邦学习节点（城堡）
    """
    
    def __init__(self, node_id: str, config: LocalAIConfig = None):
        self._node_id = node_id
        self.config = config or LocalAIConfig()
        
        # 组件
        self._ai: Optional['LocalAI'] = None
        self._dataset: Optional[LocalDataset] = None
        self._trainer: Optional[LocalTrainer] = None
        self._fed_client: Optional[FederatedClient] = None
        self._feudal_state: Optional['FeudalNode'] = None
        self._p2p: Optional['P2PServer'] = None
        
        # 状态
        self._initialized = False
        self._running = False
        
        self._lock = threading.Lock()
        
    def initialize(self, p2p_module=None, feudal_module=None) -> bool:
        """
        初始化所有组件
        """
        if self._initialized:
            return True
        
        with self._lock:
            # 创建AI系统
            self._dataset = LocalDataset(privacy_level=3)
            # 创建模型 - 避免循环导入
            model_config = self.config
            self._ai = LocalModel(model_config)
            self._trainer = LocalTrainer(self._ai, self._dataset, model_config)
            
            # 创建联邦客户端
            self._fed_client = FederatedClient(
                self._node_id,
                self._ai,
                self._dataset,
                self.config
            )
            
            # 创建分封制状态
            if feudal_module:
                self._feudal_state = feudal_module.FeudalNode(self._node_id)
            
            # 创建P2P服务器
            if p2p_module:
                self._p2p = p2p_module.P2PServer(
                    self._node_id,
                    "0.0.0.0",
                    self.config.peer_port
                )
            
            self._initialized = True
            
        return True
    
    def start_training(self, epochs: int = 1) -> Optional[TrainingResult]:
        """开始本地训练"""
        if not self._initialized:
            return None
        
        # 加载模型
        if not self._ai.is_loaded():
            self._ai.load_model()
        
        # 确保有数据
        if self._dataset.get_size() == 0:
            self._generate_sample_data()
        
        # 训练
        return self._trainer.train_local(epochs=epochs)
    
    def _generate_sample_data(self, n: int = 100) -> None:
        """生成示例数据"""
        for i in range(n):
            self._dataset.add_record({
                'id': i,
                'features': [random.uniform(-1, 1) for _ in range(10)],
                'label': random.randint(0, 1),
                'timestamp': time.time()
            })
    
    def join_federation(self, server_urls: List[str]) -> bool:
        """
        加入联邦网络
        """
        if not self._fed_client:
            return False
        
        for url in server_urls:
            if self._fed_client.join_round(url):
                return True
        
        return False
    
    def start_p2p(self, port: int = None) -> bool:
        """
        启动P2P服务器
        """
        if not self._p2p:
            return False
        
        if port:
            self.config.peer_port = port
        
        try:
            self._p2p.start()
            self._running = True
            return True
        except Exception as e:
            print(f"Failed to start P2P: {e}")
            return False
    
    def sync_with_peers(self) -> bool:
        """
        与对等节点同步
        """
        if not self._p2p:
            return False
        
        # 模拟同步
        return True
    
    def _auto_discover_peers(self) -> int:
        """
        自动发现对等节点
        """
        if not self._p2p:
            return 0
        
        # 模拟发现
        return random.randint(1, 5)
    
    def get_contribution_score(self) -> float:
        """
        计算贡献评分
        
        综合考虑训练数据量、更新质量、参与轮次等
        """
        if not self._fed_client:
            return 0.0
        
        contribution = self._fed_client.get_contribution()
        
        # 综合评分
        rounds_score = min(contribution['rounds_participated'] / 10, 1.0) * 30
        data_score = min(contribution['total_samples'] / 1000, 1.0) * 40
        quality_score = max(0, 1 - contribution['avg_loss']) * 30
        
        return rounds_score + data_score + quality_score
    
    def get_status(self) -> dict:
        """获取节点状态"""
        return {
            'node_id': self._node_id,
            'initialized': self._initialized,
            'running': self._running,
            'model_loaded': self._ai.is_loaded() if self._ai else False,
            'dataset_size': self._dataset.get_size() if self._dataset else 0,
            'client_status': self._fed_client.get_status().value if self._fed_client else None,
            'contribution_score': self.get_contribution_score(),
            'feudal_rank': self._feudal_state._get_rank() if self._feudal_state else None,
            'p2p_neighbors': len(self._p2p._neighbors) if self._p2p else 0
        }


# ============== 网络Overlay管理 ==============

class NetworkOverlay:
    """
    网络Overlay管理
    
    管理联邦学习网络拓扑
    """
    
    def __init__(self, topology: str = "star"):
        self._nodes: Dict[str, LocalNode] = {}
        self._topology = topology
        self._latencies: Dict[Tuple[str, str], float] = {}
        
        self._lock = threading.RLock()
    
    def add_node(self, node_id: str, node: LocalNode) -> None:
        """添加节点"""
        with self._lock:
            self._nodes[node_id] = node
            
            # 计算与现有节点的延迟
            for other_id, other_node in self._nodes.items():
                if other_id != node_id:
                    latency = self._compute_latency(node_id, other_id)
                    self._latencies[(node_id, other_id)] = latency
                    self._latencies[(other_id, node_id)] = latency
    
    def remove_node(self, node_id: str) -> None:
        """移除节点"""
        with self._lock:
            if node_id in self._nodes:
                del self._nodes[node_id]
            
            # 移除相关延迟记录
            self._latencies = {
                k: v for k, v in self._latencies.items()
                if node_id not in k
            }
    
    def _compute_latency(self, node_a: str, node_b: str) -> float:
        """
        估算延迟
        
        基于节点特征的简单估算
        """
        # 简化：随机延迟
        return random.uniform(10, 100)  # ms
    
    def _optimize_topology(self) -> None:
        """
        优化拓扑
        
        根据延迟调整拓扑结构
        """
        if self._topology == "star":
            # 星型拓扑：选择延迟最低的节点作为中心节点
            # 所有其他节点只与中心节点建立连接
            if not self._nodes:
                return
            
            # 计算每个节点到其他所有节点的平均延迟
            avg_latencies = {}
            for node_id in self._nodes:
                total_latency = 0.0
                count = 0
                for other_id in self._nodes:
                    if other_id != node_id:
                        lat = self._latencies.get((node_id, other_id), 50.0)
                        total_latency += lat
                        count += 1
                avg_latencies[node_id] = total_latency / count if count > 0 else float('inf')
            
            # 选择平均延迟最低的节点作为中心
            center_node = min(avg_latencies, key=avg_latencies.get)
            
            # 重新计算延迟：所有通信经过中心节点
            for src in self._nodes:
                for dst in self._nodes:
                    if src != dst:
                        # 星型路径延迟 = src->center + center->dst
                        lat_src_center = self._latencies.get((src, center_node), 50.0)
                        lat_center_dst = self._latencies.get((center_node, dst), 50.0)
                        self._latencies[(src, dst)] = lat_src_center + lat_center_dst
                        self._latencies[(dst, src)] = lat_center_dst + lat_src_center

        elif self._topology == "mesh":
            # 网状拓扑：完全图，每对节点直接连接
            # 优化策略：移除高延迟连接，保留低延迟的shortcut
            if len(self._nodes) < 3:
                return
            
            # 计算所有连接的延迟中位数
            all_lats = [lat for lat in self._latencies.values()]
            if not all_lats:
                return
            all_lats.sort()
            median_lat = all_lats[len(all_lats) // 2]
            
            # 对于延迟超过中位数2倍的连接，标记为高延迟
            # 在网状拓扑中保留所有连接，但记录高延迟路径供路由优化
            for (src, dst), lat in list(self._latencies.items()):
                if lat > median_lat * 2:
                    # 尝试寻找中继节点降低有效延迟
                    for relay in self._nodes:
                        if relay != src and relay != dst:
                            relay_lat = self._latencies.get((src, relay), float('inf')) + \
                                       self._latencies.get((relay, dst), float('inf'))
                            if relay_lat < lat:
                                # 中继路径更优，更新延迟记录
                                self._latencies[(src, dst)] = relay_lat
                                break

        elif self._topology == "hierarchical":
            # 分层拓扑：类似分封制
            # 将节点分组，每组选出一个组长，组长之间组成上层网络
            
            if len(self._nodes) < 4:
                return
            
            # 按延迟聚类：使用简单的贪心算法
            node_ids = list(self._nodes.keys())
            unassigned = set(node_ids)
            groups = []
            group_size = max(2, int(math.sqrt(len(node_ids))))  # 每组大约sqrt(n)个节点
            
            while unassigned:
                # 选择一个种子节点
                seed = unassigned.pop()
                group = [seed]
                
                # 找到与种子延迟最低的节点加入组
                candidates = sorted(unassigned, 
                                   key=lambda n: self._latencies.get((seed, n), 50.0))
                for candidate in candidates[:group_size - 1]:
                    group.append(candidate)
                    unassigned.discard(candidate)
                
                groups.append(group)
            
            # 为每个组选择组长（组内平均延迟最低的节点）
            leaders = []
            for group in groups:
                best_leader = group[0]
                best_avg = float('inf')
                for node in group:
                    avg = sum(self._latencies.get((node, other), 50.0) 
                             for other in group if other != node) / max(len(group) - 1, 1)
                    if avg < best_avg:
                        best_avg = avg
                        best_leader = node
                leaders.append(best_leader)
            
            # 组间通信通过组长中继，优化跨组延迟
            for i, group_i in enumerate(groups):
                for j, group_j in enumerate(groups):
                    if i >= j:
                        continue
                    leader_i = leaders[i]
                    leader_j = leaders[j]
                    # 跨组通信延迟 = 组内到组长 + 组长间 + 组长到组内
                    base_lat = self._latencies.get((leader_i, leader_j), 50.0)
                    for ni in group_i:
                        for nj in group_j:
                            intra_i = self._latencies.get((ni, leader_i), 50.0)
                            intra_j = self._latencies.get((leader_j, nj), 50.0)
                            total = intra_i + base_lat + intra_j
                            self._latencies[(ni, nj)] = total
                            self._latencies[(nj, ni)] = total
    
    def get_optimal_path(self, src: str, dst: str) -> List[str]:
        """
        获取最优路径
        
        简单的BFS路径查找
        """
        if src == dst:
            return [src]
        
        if self._topology == "star":
            # 星型：经过中心
            return [src, "center", dst]
        elif self._topology == "mesh":
            # 网状：直接连接
            return [src, dst]
        else:
            # 默认：直接路径
            return [src, dst]
    
    def broadcast_model_update(self, update: Dict) -> int:
        """
        广播模型更新
        
        返回成功接收的节点数
        """
        success_count = 0
        
        with self._lock:
            for node_id, node in self._nodes.items():
                try:
                    if node._initialized:
                        success_count += 1
                except Exception:
                    pass
        
        return success_count
    
    def get_network_stats(self) -> dict:
        """获取网络统计"""
        with self._lock:
            total_latency = sum(self._latencies.values())
            count = len(self._latencies)
            avg_latency = total_latency / count if count > 0 else 0
            
            return {
                'total_nodes': len(self._nodes),
                'topology': self._topology,
                'avg_latency_ms': avg_latency,
                'total_connections': count // 2
            }


# ============== 便捷函数 ==============

def create_local_node(node_id: str, config: LocalAIConfig = None) -> LocalNode:
    """创建本地节点"""
    return LocalNode(node_id, config)


def create_network(num_nodes: int, topology: str = "star") -> NetworkOverlay:
    """创建测试网络"""
    overlay = NetworkOverlay(topology=topology)
    
    for i in range(num_nodes):
        node_id = f"node_{i}"
        config = LocalAIConfig(
            model_path=f"models/node_{i}",
            federated_mode=True,
            peer_port=8000 + i
        )
        node = LocalNode(node_id, config)
        overlay.add_node(node_id, node)
    
    return overlay


# ============== 主程序入口 ==============

if __name__ == "__main__":
    print("=== 本地AI系统演示 ===\n")
    
    # 1. 本地模型演示
    print("1. 本地模型:")
    config = LocalAIConfig(
        model_path="models/mock",
        quantize_level=QuantizeLevel.INT8,
        max_memory_mb=2048
    )
    
    model = LocalModel(config)
    print(f"模型加载前: {model.is_loaded()}")
    
    model.load_model()
    print(f"模型加载后: {model.is_loaded()}")
    print(f"模型信息: {model.get_info()}")
    
    # 推理
    inputs = [random.uniform(-1, 1) for _ in range(128)]
    outputs = model.predict(inputs)
    print(f"推理输出维度: {len(outputs)}")
    
    # 2. 本地数据集演示
    print("\n2. 本地数据集:")
    dataset = LocalDataset(privacy_level=3)
    
    # 添加数据
    for i in range(100):
        dataset.add_record({
            'id': i,
            'features': [random.uniform(-1, 1) for _ in range(10)],
            'label': random.randint(0, 1),
            'value': random.uniform(0, 100)
        })
    
    print(f"数据集大小: {dataset.get_size()}")
    
    # 查询
    results = dataset.query({'label': 1})
    print(f"标签为1的记录数: {len(results)}")
    
    # 差分隐私采样
    sampled = dataset.export_sample(10, epsilon=1.0)
    print(f"采样数量: {len(sampled)}")
    
    # 统计
    stats = dataset._compute_statistics()
    print(f"统计摘要: {list(stats.keys())[:5]}...")
    
    # 3. 本地训练演示
    print("\n3. 本地训练:")
    trainer = LocalTrainer(model, dataset, config)
    
    result = trainer.train_local(epochs=2)
    print(f"训练结果: {result.epochs} 轮, 损失={result.final_loss:.4f}")
    print(f"训练时长: {result.duration:.2f}秒")
    print(f"处理的样本数: {result.samples_processed}")
    
    # 4. 联邦客户端演示
    print("\n4. 联邦客户端:")
    client = FederatedClient("client_1", model, dataset, config)
    
    # 加入轮次
    joined = client.join_round("server.example.com:8000")
    print(f"加入轮次: {joined}")
    
    # 计算本地更新
    update = client.compute_local_update(epochs=1)
    print(f"更新样本数: {update['num_samples']}")
    print(f"更新轮次: {update['round']}")
    
    # 提交更新
    submitted = client.submit_update(update)
    print(f"提交结果: {submitted}")
    
    # 获取贡献
    contribution = client.get_contribution()
    print(f"贡献统计: {contribution}")
    
    # 5. 完整本地节点演示
    print("\n5. 完整本地节点:")
    from federated.p2p.peer_discovery import FeudalNode
    from federated.p2p import peer_discovery as p2p_module
    
    # 创建节点
    node = LocalNode("castle_1", config)
    node.initialize(p2p_module=p2p_module, feudal_module={'FeudalNode': FeudalNode})
    
    # 生成数据并训练
    node._generate_sample_data(200)
    training_result = node.start_training(epochs=2)
    
    print(f"节点状态: {node.get_status()}")
    print(f"贡献评分: {node.get_contribution_score():.2f}")
    
    # 6. 网络Overlay演示
    print("\n6. 网络Overlay:")
    network = create_network(num_nodes=5, topology="star")
    
    stats = network.get_network_stats()
    print(f"网络统计: {stats}")
    
    # 获取最优路径
    path = network.get_optimal_path("node_0", "node_3")
    print(f"最优路径 node_0 -> node_3: {path}")
    
    # 广播更新
    update = {'model_id': 'v1', 'weights': [[1, 2, 3]]}
    broadcast_count = network.broadcast_model_update(update)
    print(f"广播接收节点数: {broadcast_count}")
    
    print("\n=== 演示完成 ===")
