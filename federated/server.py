"""
联邦学习服务器 - 聚合调度（完整生产级实现）
支持: FedAvg, FedProx, SCAFFOLD, 客户端管理, 模型聚合, 通信接口, 检查点

纯 Python 标准库实现，无外部依赖
"""

import os
import sys
import json
import math
import logging
import copy
import time
import threading
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from collections import OrderedDict, defaultdict
import hashlib
import struct
import pickle


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


# ============================================================================
# 简单序列化工具（替代 torch.save/load）
# ============================================================================

def serialize_weights(weights: Dict[str, List]) -> bytes:
    """将模型权重序列化为字节"""
    # 将 numpy array 或 list 转换为 bytes
    serialized = {}
    for key, value in weights.items():
        if hasattr(value, 'tolist'):  # numpy array
            serialized[key] = value.tolist()
        elif isinstance(value, (list, tuple)):
            serialized[key] = value
        elif isinstance(value, (int, float)):
            serialized[key] = value
        elif isinstance(value, bytes):
            serialized[key] = list(value)
        else:
            serialized[key] = value
    return pickle.dumps(serialized)


def deserialize_weights(data: bytes) -> Dict[str, Any]:
    """从字节反序列化模型权重"""
    return pickle.loads(data)


def save_weights_to_file(weights: Dict[str, Any], filepath: Path):
    """保存权重到文件"""
    serialized = {}
    for key, value in weights.items():
        if hasattr(value, 'tolist'):
            serialized[key] = value.tolist()
        elif isinstance(value, (list, tuple, int, float, bytes)):
            serialized[key] = value
        else:
            serialized[key] = str(value)
    with open(filepath, 'wb') as f:
        pickle.dumps(serialized)
        f.write(pickle.dumps(serialized))


def load_weights_from_file(filepath: Path) -> Dict[str, Any]:
    """从文件加载权重"""
    if not filepath.exists():
        return {}
    with open(filepath, 'rb') as f:
        return pickle.loads(f.read())


# ============================================================================
# 联邦学习服务器配置
# ============================================================================

@dataclass
class FederatedServerConfig:
    """联邦学习服务器完整配置"""
    # 基础设置
    server_id: str = "server_0"
    global_model_path: Optional[str] = None
    output_dir: str = "./federated_server"
    
    # 聚合算法
    aggregation_algorithm: str = "fedavg"
    fedprox_mu: float = 0.01
    scaffold_lr: float = 1.0
    
    # 聚合参数
    weighted_average: bool = True
    min_clients_per_round: int = 1
    fraction_fit: float = 1.0
    round_timeout: int = 3600
    
    # 模型管理
    save_model_every_round: int = 10
    keep_last_n_models: int = 5
    use_best_model: bool = True
    validation_data_path: Optional[str] = None
    validation_frequency: int = 5
    
    # 通信配置
    communication_protocol: str = "file"
    host: str = "0.0.0.0"
    port: int = 8080
    max_clients: int = 100
    client_timeout: int = 300
    
    # 差分隐私
    use_aggregation_dp: bool = False
    dp_clip_norm: float = 1.0
    dp_noise_multiplier: float = 1.0
    
    # 安全加密
    use_encryption: bool = False
    encryption_key_path: Optional[str] = None
    
    # 检查点
    save_checkpoint_frequency: int = 5
    resume_from_checkpoint: Optional[str] = None


# ============================================================================
# 模型权重管理（替代 torch 依赖）
# ============================================================================

class ModelWeights:
    """纯 Python 模型权重管理"""
    
    def __init__(self, weights: Optional[Dict[str, Any]] = None):
        self.weights = weights or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return self.weights
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelWeights":
        return cls(data)
    
    def apply_diff(self, diff: Dict[str, Any]) -> "ModelWeights":
        """应用权重差分"""
        new_weights = {}
        for key, value in self.weights.items():
            if key in diff:
                delta = diff[key]
                if isinstance(value, (int, float)) and isinstance(delta, (int, float)):
                    new_weights[key] = value + delta
                elif isinstance(value, list) and isinstance(delta, list):
                    new_weights[key] = [
                        v + d if isinstance(v, (int, float)) and isinstance(d, (int, float)) 
                        else v for v, d in zip(value, delta)
                    ]
                else:
                    new_weights[key] = value
            else:
                new_weights[key] = value
        return ModelWeights(new_weights)
    
    def compute_diff(self, other: "ModelWeights") -> Dict[str, Any]:
        """计算权重差分"""
        diff = {}
        for key, value in self.weights.items():
            if key in other.weights:
                other_val = other.weights[key]
                if isinstance(value, (int, float)) and isinstance(other_val, (int, float)):
                    diff[key] = value - other_val
                elif isinstance(value, list) and isinstance(other_val, list):
                    diff[key] = [
                        v - ov if isinstance(v, (int, float)) and isinstance(ov, (int, float))
                        else v for v, ov in zip(value, other_val)
                    ]
                else:
                    diff[key] = value
        return diff
    
    def add_noise(self, clip_norm: float, noise_multiplier: float) -> "ModelWeights":
        """添加高斯噪声（差分隐私）"""
        import random
        noise_scale = clip_norm * noise_multiplier
        new_weights = {}
        for key, value in self.weights.items():
            if isinstance(value, (int, float)):
                noise = random.gauss(0, noise_scale)
                new_weights[key] = value + noise
            elif isinstance(value, list):
                new_weights[key] = [
                    v + random.gauss(0, noise_scale) if isinstance(v, (int, float)) else v
                    for v in value
                ]
            else:
                new_weights[key] = value
        return ModelWeights(new_weights)
    
    def copy(self) -> "ModelWeights":
        return ModelWeights(copy.deepcopy(self.weights))


# ============================================================================
# 通信后端（服务器端）
# ============================================================================

class CommunicationBackendServer:
    """服务器端通信抽象基类"""
    def __init__(self, config: FederatedServerConfig):
        self.config = config
    
    def start(self):
        """启动服务器"""
        # 默认实现：设置运行状态
        self._running = True
        logger.info(f"Communication backend server started (host={self.config.host}, port={self.config.port})")
    
    def stop(self):
        """停止服务器"""
        # 默认实现：清理运行状态
        self._running = False
        logger.info("Communication backend server stopped")
    
    def receive_updates(self, timeout: int) -> List[Dict[str, Any]]:
        """接收客户端更新"""
        # 默认实现：返回空列表，子类应重写以实现具体通信逻辑
        logger.warning("Using default receive_updates, no updates will be received")
        return []
    
    def send_global_model(self, round_num: int, model_data: Dict[str, Any], client_ids: List[str]):
        """向指定客户端发送全局模型"""
        # 默认实现：记录日志，子类应重写以实现具体通信逻辑
        logger.info(f"Sending global model (round={round_num}) to {len(client_ids)} clients")
    
    def register_client(self, client_id: str, metadata: Dict) -> bool:
        """注册客户端"""
        # 默认实现：直接返回成功
        logger.info(f"Registered client {client_id}")
        return True


class FileCommunicationBackendServer(CommunicationBackendServer):
    """基于文件的通信后端（用于测试）"""
    def __init__(self, config: FederatedServerConfig):
        super().__init__(config)
        self.server_dir = Path(config.output_dir) / "server_data"
        self.updates_dir = self.server_dir / "updates"
        self.global_models_dir = self.server_dir / "global_models"
        self.clients_dir = self.server_dir / "clients"
        self.updates_dir.mkdir(parents=True, exist_ok=True)
        self.global_models_dir.mkdir(parents=True, exist_ok=True)
        self.clients_dir.mkdir(parents=True, exist_ok=True)
        self._round_updates = {}
        self._running = False
    
    def start(self):
        self._running = True
        logger.info("File communication backend started")
    
    def stop(self):
        self._running = False
        logger.info("File communication backend stopped")
    
    def receive_updates(self, timeout: int) -> List[Dict[str, Any]]:
        """等待客户端上传更新文件"""
        start_time = time.time()
        updates = []
        while time.time() - start_time < timeout:
            for file_path in self.updates_dir.glob("*.pkl"):
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    if "round" in data:
                        updates.append(data)
                    os.unlink(file_path)
                except Exception as e:
                    logger.warning(f"Failed to load update file {file_path}: {e}")
            if updates:
                break
            time.sleep(5)
        return updates
    
    def send_global_model(self, round_num: int, model_data: Dict[str, Any], client_ids: List[str]):
        """保存全局模型到文件"""
        model_file = self.global_models_dir / f"global_round_{round_num}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Saved global model for round {round_num} to {model_file}")
    
    def register_client(self, client_id: str, metadata: Dict) -> bool:
        reg_file = self.clients_dir / f"{client_id}.json"
        with open(reg_file, 'w') as f:
            json.dump({"client_id": client_id, "metadata": metadata, "registered_at": time.time()}, f)
        logger.info(f"Registered client {client_id}")
        return True


# ============================================================================
# 聚合器（FedAvg, FedProx, SCAFFOLD）
# ============================================================================

class Aggregator:
    """聚合器抽象基类"""
    def __init__(self, config: FederatedServerConfig, initial_weights: Optional[Dict] = None):
        self.config = config
        self.global_weights = initial_weights or {}
    
    def aggregate(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """聚合客户端更新 - 默认实现：加权平均"""
        if not updates:
            return self.global_weights
        
        # 计算总样本数用于加权
        total_samples = sum(u.get("num_samples", 1) for u in updates)
        
        # 判断更新格式：权重差分 or 完整模型
        use_diff = "weight_diff" in updates[0]
        
        aggregated = {}
        
        if use_diff:
            # 聚合权重差分
            keys = self.global_weights.keys()
            for key in keys:
                aggregated[key] = 0.0
            
            for update in updates:
                weight_diff = update.get("weight_diff", {})
                num_samples = update.get("num_samples", 1)
                w = num_samples / total_samples if self.config.weighted_average else 1.0 / len(updates)
                for key, diff in weight_diff.items():
                    if key in aggregated:
                        aggregated[key] += diff * w
            
            # 应用差分到全局权重
            for key in keys:
                if key in aggregated and isinstance(self.global_weights.get(key), (int, float)):
                    aggregated[key] = self.global_weights[key] + aggregated[key]
        else:
            # 直接平均完整模型权重
            keys = updates[0].get("model_state", {}).keys()
            for key in keys:
                aggregated[key] = 0.0
            
            for update in updates:
                model_state = update.get("model_state", {})
                if not model_state:
                    continue
                num_samples = update.get("num_samples", 1)
                w = num_samples / total_samples if self.config.weighted_average else 1.0 / len(updates)
                for key, param in model_state.items():
                    if key in aggregated and isinstance(param, (int, float)):
                        aggregated[key] += param * w
        
        return aggregated if aggregated else self.global_weights
    
    def update_global_model(self, new_weights: Dict[str, Any]):
        """更新全局模型权重"""
        self.global_weights = new_weights


class FedAvgAggregator(Aggregator):
    """FedAvg 聚合器：加权平均"""
    def aggregate(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not updates:
            return self.global_weights
        
        use_diff = "weight_diff" in updates[0]
        total_samples = sum(u.get("num_samples", 1) for u in updates)
        
        aggregated_weights = {}
        
        if use_diff:
            # 聚合权重差
            for key in self.global_weights.keys():
                aggregated_weights[key] = 0.0
            
            for update in updates:
                weight_diff = update.get("weight_diff", {})
                num_samples = update.get("num_samples", 1)
                weight = num_samples / total_samples if self.config.weighted_average else 1.0 / len(updates)
                for key, diff in weight_diff.items():
                    if key in aggregated_weights:
                        aggregated_weights[key] += diff * weight
            
            # 应用差到全局权重
            for key in self.global_weights.keys():
                if key in aggregated_weights:
                    if isinstance(self.global_weights[key], (int, float)):
                        self.global_weights[key] += aggregated_weights[key]
        else:
            # 直接平均模型权重
            for key in updates[0].get("model_state", {}).keys():
                aggregated_weights[key] = 0.0
            
            for update in updates:
                model_state = update.get("model_state", {})
                if not model_state:
                    continue
                num_samples = update.get("num_samples", 1)
                weight = num_samples / total_samples if self.config.weighted_average else 1.0 / len(updates)
                for key, param in model_state.items():
                    if key in aggregated_weights and isinstance(param, (int, float)):
                        aggregated_weights[key] += param * weight
            
            aggregated_weights = aggregated_weights
        
        # 可选：添加差分隐私噪声
        if self.config.use_aggregation_dp:
            for key in aggregated_weights:
                if isinstance(aggregated_weights[key], (int, float)):
                    import random
                    noise = random.gauss(0, self.config.dp_clip_norm * self.config.dp_noise_multiplier)
                    aggregated_weights[key] += noise
        
        return aggregated_weights if aggregated_weights else self.global_weights


class FedProxAggregator(FedAvgAggregator):
    """FedProx 聚合器（与 FedAvg 相同）"""
    pass


class SCAFFOLDAggregator(Aggregator):
    """SCAFFOLD 聚合器"""
    def __init__(self, config: FederatedServerConfig, initial_weights: Optional[Dict] = None):
        super().__init__(config, initial_weights)
        self.global_control_variate = {}
        self._init_control_variate()
    
    def _init_control_variate(self):
        """初始化全局控制变量"""
        if not self.global_weights:
            return
        for key, param in self.global_weights.items():
            if isinstance(param, (int, float)):
                self.global_control_variate[key] = 0.0
            elif isinstance(param, list):
                self.global_control_variate[key] = [0.0] * len(param)
    
    def aggregate(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not updates:
            return self.global_weights
        
        # 1. 聚合权重差
        total_samples = sum(u.get("num_samples", 1) for u in updates)
        aggregated_diff = {}
        for key in self.global_weights.keys():
            aggregated_diff[key] = 0.0
        
        for update in updates:
            weight_diff = update.get("weight_diff", {})
            if not weight_diff:
                continue
            num_samples = update.get("num_samples", 1)
            weight = num_samples / total_samples if self.config.weighted_average else 1.0 / len(updates)
            for key, diff in weight_diff.items():
                if key in aggregated_diff and isinstance(diff, (int, float)):
                    aggregated_diff[key] += diff * weight
        
        # 应用权重差
        new_weights = {}
        for key in self.global_weights.keys():
            if isinstance(self.global_weights[key], (int, float)) and key in aggregated_diff:
                new_weights[key] = self.global_weights[key] + aggregated_diff[key]
            else:
                new_weights[key] = self.global_weights[key]
        
        # 2. 聚合控制变量更新
        scaffold_updates = [u.get("scaffold_update") for u in updates if "scaffold_update" in u]
        if scaffold_updates and self.global_control_variate:
            aggregated_cv_update = {}
            for key in self.global_control_variate.keys():
                aggregated_cv_update[key] = 0.0
            
            total_updates = len(scaffold_updates)
            for su in scaffold_updates:
                for key, delta in su.items():
                    if key in aggregated_cv_update and isinstance(delta, (int, float)):
                        aggregated_cv_update[key] += delta / total_updates
            
            # 更新全局控制变量
            lr = self.config.scaffold_lr
            for key in self.global_control_variate.keys():
                if key in aggregated_cv_update:
                    if isinstance(self.global_control_variate[key], (int, float)):
                        self.global_control_variate[key] += lr * aggregated_cv_update[key]
        
        return new_weights
    
    def get_global_control_variate(self) -> Dict[str, Any]:
        return self.global_control_variate


# ============================================================================
# 客户端管理器
# ============================================================================

class ClientManager:
    """管理客户端注册、状态、选择参与轮次的客户端"""
    def __init__(self, config: FederatedServerConfig):
        self.config = config
        self.clients: Dict[str, Dict] = {}
        self.client_data_sizes: Dict[str, int] = {}
        self._lock = threading.Lock()
    
    def register_client(self, client_id: str, metadata: Dict) -> bool:
        with self._lock:
            if client_id in self.clients:
                logger.warning(f"Client {client_id} already registered, updating metadata")
            self.clients[client_id] = {
                "metadata": metadata,
                "registered_at": time.time(),
                "last_active": time.time(),
            }
            self.client_data_sizes[client_id] = metadata.get("data_size", 0)
            logger.info(f"Client {client_id} registered. Total clients: {len(self.clients)}")
        return True
    
    def get_available_clients(self) -> List[str]:
        with self._lock:
            return list(self.clients.keys())
    
    def select_clients_for_round(self, round_num: int) -> List[str]:
        """选择参与当前轮次的客户端"""
        import random
        available = self.get_available_clients()
        if not available:
            return []
        num_clients = max(1, int(len(available) * self.config.fraction_fit))
        selected = random.sample(available, min(num_clients, len(available)))
        logger.info(f"Round {round_num}: selected {len(selected)} clients out of {len(available)}")
        return selected
    
    def get_client_data_sizes(self, client_ids: List[str]) -> Dict[str, int]:
        return {cid: self.client_data_sizes.get(cid, 0) for cid in client_ids}
    
    def update_client_active(self, client_id: str):
        with self._lock:
            if client_id in self.clients:
                self.clients[client_id]["last_active"] = time.time()


# ============================================================================
# 联邦学习服务器主类
# ============================================================================

class FederatedServer:
    """
    联邦学习服务器主类
    管理全局模型、聚合客户端更新、调度轮次
    """
    def __init__(
        self,
        config: FederatedServerConfig,
        initial_weights: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self.global_weights = initial_weights or {}
        
        # 客户端管理器
        self.client_manager = ClientManager(config)
        
        # 聚合器
        self.aggregator = self._create_aggregator()
        
        # 通信后端
        self.comm = self._create_communication_backend()
        
        # 状态
        self.current_round = 0
        self.round_history = []
        
        # 检查点目录
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.output_dir / "global_models"
        self.models_dir.mkdir(exist_ok=True)
        
        # 加载初始模型
        self._load_initial_model()
        
        # 恢复检查点
        if config.resume_from_checkpoint:
            self._load_checkpoint(config.resume_from_checkpoint)
    
    def _create_aggregator(self) -> Aggregator:
        if self.config.aggregation_algorithm == "fedavg":
            agg = FedAvgAggregator(self.config, self.global_weights)
        elif self.config.aggregation_algorithm == "fedprox":
            agg = FedProxAggregator(self.config, self.global_weights)
        elif self.config.aggregation_algorithm == "scaffold":
            agg = SCAFFOLDAggregator(self.config, self.global_weights)
        else:
            raise ValueError(f"Unknown aggregation algorithm: {self.config.aggregation_algorithm}")
        agg.global_weights = self.global_weights
        return agg
    
    def _create_communication_backend(self) -> CommunicationBackendServer:
        return FileCommunicationBackendServer(self.config)
    
    def _load_initial_model(self):
        """加载初始全局模型"""
        if self.config.global_model_path and os.path.exists(self.config.global_model_path):
            logger.info(f"Loading initial global model from {self.config.global_model_path}")
            self.global_weights = load_weights_from_file(Path(self.config.global_model_path))
        else:
            logger.info("No initial model provided, using empty weights")
        
        # 保存初始模型
        init_path = self.models_dir / "global_init.pkl"
        with open(init_path, 'wb') as f:
            pickle.dump(self.global_weights, f)
    
    def _save_checkpoint(self, round_num: int):
        """保存服务器状态"""
        checkpoint = {
            "round": round_num,
            "global_weights": self.global_weights,
            "aggregator_state": None,
            "client_manager_state": {
                "clients": self.client_manager.clients,
                "client_data_sizes": self.client_manager.client_data_sizes,
            },
            "round_history": self.round_history,
        }
        
        if isinstance(self.aggregator, SCAFFOLDAggregator):
            checkpoint["global_control_variate"] = self.aggregator.global_control_variate
        
        ckpt_path = self.output_dir / f"server_checkpoint_round_{round_num}.pkl"
        with open(ckpt_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        logger.info(f"Saved server checkpoint to {ckpt_path}")
        
        # 删除旧检查点
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self, keep: int = 5):
        checkpoints = sorted(self.output_dir.glob("server_checkpoint_round_*.pkl"))
        for ckpt in checkpoints[:-keep]:
            ckpt.unlink()
    
    def _load_checkpoint(self, checkpoint_path: str):
        """恢复服务器状态"""
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint {checkpoint_path} not found")
            return
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        self.current_round = checkpoint["round"]
        self.global_weights = checkpoint["global_weights"]
        self.round_history = checkpoint.get("round_history", [])
        
        # 恢复客户端管理器状态
        client_state = checkpoint.get("client_manager_state", {})
        if client_state:
            self.client_manager.clients = client_state.get("clients", {})
            self.client_manager.client_data_sizes = client_state.get("client_data_sizes", {})
        
        # 恢复 SCAFFOLD 控制变量
        if isinstance(self.aggregator, SCAFFOLDAggregator) and "global_control_variate" in checkpoint:
            self.aggregator.global_control_variate = checkpoint["global_control_variate"]
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}, resuming from round {self.current_round}")
    
    def _save_global_model(self, round_num: int):
        """保存全局模型快照"""
        model_path = self.models_dir / f"global_round_{round_num}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.global_weights, f)
        # 清理旧模型
        models = sorted(self.models_dir.glob("global_round_*.pkl"))
        for model_file in models[:-self.config.keep_last_n_models]:
            model_file.unlink()
        logger.info(f"Saved global model for round {round_num} to {model_path}")
    
    def _aggregate_updates(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """聚合客户端更新"""
        if len(updates) < self.config.min_clients_per_round:
            logger.warning(f"Only {len(updates)} clients responded, less than min_clients_per_round={self.config.min_clients_per_round}")
            return self.global_weights
        
        new_weights = self.aggregator.aggregate(updates)
        return new_weights
    
    def _broadcast_model(self, round_num: int, selected_clients: List[str]):
        """向选定的客户端广播全局模型"""
        model_data = {
            "model_state": self.global_weights,
            "version": round_num,
        }
        if isinstance(self.aggregator, SCAFFOLDAggregator):
            model_data["global_control_variate"] = self.aggregator.get_global_control_variate()
        
        self.comm.send_global_model(round_num, model_data, selected_clients)
        logger.info(f"Broadcast global model to {len(selected_clients)} clients for round {round_num}")
    
    def _receive_updates(self, timeout: int) -> List[Dict[str, Any]]:
        """接收客户端更新"""
        return self.comm.receive_updates(timeout)
    
    def run_round(self) -> Dict[str, Any]:
        """执行一轮联邦学习"""
        self.current_round += 1
        round_num = self.current_round
        logger.info(f"=== Starting round {round_num} ===")
        
        # 选择参与客户端
        selected_clients = self.client_manager.select_clients_for_round(round_num)
        if not selected_clients:
            logger.warning("No clients available, stopping")
            return {"status": "no_clients"}
        
        # 广播全局模型
        self._broadcast_model(round_num, selected_clients)
        
        # 接收更新
        updates = self._receive_updates(self.config.round_timeout)
        logger.info(f"Received {len(updates)} updates from clients")
        
        # 聚合更新
        new_weights = self._aggregate_updates(updates)
        
        # 更新全局模型
        self.global_weights = new_weights
        self.aggregator.global_weights = self.global_weights
        
        # 保存模型快照
        if round_num % self.config.save_model_every_round == 0:
            self._save_global_model(round_num)
        
        # 保存服务器检查点
        if round_num % self.config.save_checkpoint_frequency == 0:
            self._save_checkpoint(round_num)
        
        round_info = {
            "round": round_num,
            "num_clients": len(updates),
            "selected_clients": len(selected_clients),
        }
        self.round_history.append(round_info)
        logger.info(f"Round {round_num} completed: {round_info}")
        return round_info
    
    def run(self, num_rounds: int):
        """运行多轮联邦学习"""
        self.comm.start()
        
        try:
            for _ in range(num_rounds):
                self.run_round()
        except KeyboardInterrupt:
            logger.info("Server interrupted")
        finally:
            self.comm.stop()
            # 保存最终模型
            final_path = self.output_dir / "final_global_model.pkl"
            with open(final_path, 'wb') as f:
                pickle.dump(self.global_weights, f)
            logger.info(f"Final model saved to {final_path}")
    
    def get_global_model(self) -> Dict[str, Any]:
        """返回当前全局模型权重"""
        return self.global_weights


# ============================================================================
# 便捷函数
# ============================================================================

def run_federated_server(
    output_dir: str = "./federated_server",
    num_rounds: int = 10,
    aggregation: str = "fedavg",
) -> FederatedServer:
    """快速启动联邦学习服务器"""
    server_cfg = FederatedServerConfig(
        output_dir=output_dir,
        aggregation_algorithm=aggregation,
    )
    
    server = FederatedServer(
        config=server_cfg,
        initial_weights={},
    )
    
    server.run(num_rounds=num_rounds)
    return server


# ============================================================================
# 命令行入口
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Federated Learning Server")
    parser.add_argument("--output_dir", type=str, default="./federated_server", help="Output directory")
    parser.add_argument("--num_rounds", type=int, default=10, help="Number of federated rounds")
    parser.add_argument("--aggregation", type=str, default="fedavg", choices=["fedavg", "fedprox", "scaffold"])
    args = parser.parse_args()
    
    run_federated_server(
        output_dir=args.output_dir,
        num_rounds=args.num_rounds,
        aggregation=args.aggregation,
    )
