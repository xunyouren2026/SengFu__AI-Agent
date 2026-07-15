"""
模型存储 - 版本管理、检查点
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import copy
import os


class ModelVersion:
    """模型版本信息"""
    
    def __init__(
        self,
        version_id: int,
        timestamp: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.version_id = version_id
        self.timestamp = timestamp
        self.metadata = metadata or {}
        self.checksum: Optional[str] = None
        self.size_bytes: int = 0
        self.num_clients: int = 0
        self.round_num: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'version_id': self.version_id,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'checksum': self.checksum,
            'size_bytes': self.size_bytes,
            'num_clients': self.num_clients,
            'round_num': self.round_num
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """从字典创建"""
        version = cls(
            version_id=data['version_id'],
            timestamp=data['timestamp'],
            metadata=data.get('metadata', {})
        )
        version.checksum = data.get('checksum')
        version.size_bytes = data.get('size_bytes', 0)
        version.num_clients = data.get('num_clients', 0)
        version.round_num = data.get('round_num', 0)
        return version


class Checkpoint:
    """检查点"""
    
    def __init__(
        self,
        checkpoint_id: str,
        model_params: Dict[str, Any],
        version: int,
        timestamp: float
    ):
        self.checkpoint_id = checkpoint_id
        self.model_params = model_params
        self.version = version
        self.timestamp = timestamp
        self.is_compressed: bool = False
        self.compression_ratio: float = 1.0
        self.tags: List[str] = []
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'checkpoint_id': self.checkpoint_id,
            'model_params': self.model_params,
            'version': self.version,
            'timestamp': self.timestamp,
            'is_compressed': self.is_compressed,
            'compression_ratio': self.compression_ratio,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Checkpoint':
        """从字典创建"""
        cp = cls(
            checkpoint_id=data['checkpoint_id'],
            model_params=data['model_params'],
            version=data['version'],
            timestamp=data['timestamp']
        )
        cp.is_compressed = data.get('is_compressed', False)
        cp.compression_ratio = data.get('compression_ratio', 1.0)
        cp.tags = data.get('tags', [])
        return cp


class ModelStore:
    """
    模型存储管理器
    
    功能:
    - 版本管理: 跟踪模型版本历史
    - 检查点: 保存和恢复模型状态
    - 压缩存储: 可选的模型压缩
    - 元数据管理: 存储模型相关信息
    """
    
    def __init__(
        self,
        store_path: Optional[str] = None,
        max_versions: int = 100,
        auto_checkpoint: bool = True,
        checkpoint_interval: int = 10
    ):
        self.store_path = store_path or "/tmp/model_store"
        self.max_versions = max_versions
        self.auto_checkpoint = auto_checkpoint
        self.checkpoint_interval = checkpoint_interval
        
        # 存储结构
        self._models: Dict[int, Dict[str, Any]] = {}  # version_id -> params
        self._versions: Dict[int, ModelVersion] = {}
        self._checkpoints: Dict[str, Checkpoint] = {}
        self._current_version: int = 0
        self._latest_params: Optional[Dict[str, Any]] = None
        
        # 统计信息
        self._total_stored: int = 0
        self._total_size: int = 0
    
    def save(
        self,
        model_params: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        create_checkpoint: bool = False
    ) -> int:
        """
        保存模型参数
        
        Args:
            model_params: 模型参数字典
            metadata: 可选的元数据
            create_checkpoint: 是否创建检查点
        
        Returns:
            版本ID
        """
        self._current_version += 1
        version_id = self._current_version
        
        # 深拷贝参数
        params_copy = copy.deepcopy(model_params)
        self._models[version_id] = params_copy
        self._latest_params = params_copy
        
        # 创建版本信息
        timestamp = datetime.now().timestamp()
        version = ModelVersion(version_id, timestamp, metadata)
        
        # 计算大小和校验和
        version.size_bytes = self._estimate_size(params_copy)
        version.checksum = self._compute_checksum(params_copy)
        
        self._versions[version_id] = version
        self._total_stored += 1
        self._total_size += version.size_bytes
        
        # 自动检查点
        if self.auto_checkpoint and version_id % self.checkpoint_interval == 0:
            self.create_checkpoint(
                f"auto_v{version_id}",
                model_params,
                tags=["auto"]
            )
        
        # 手动检查点
        if create_checkpoint:
            self.create_checkpoint(
                f"manual_v{version_id}",
                model_params,
                tags=["manual"]
            )
        
        # 清理旧版本
        self._cleanup_old_versions()
        
        return version_id
    
    def load(self, version_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        加载模型参数
        
        Args:
            version_id: 版本ID，None表示最新版本
        
        Returns:
            模型参数字典
        """
        if version_id is None:
            return copy.deepcopy(self._latest_params) if self._latest_params else None
        
        if version_id in self._models:
            return copy.deepcopy(self._models[version_id])
        
        return None
    
    def get_version_info(self, version_id: int) -> Optional[ModelVersion]:
        """获取版本信息"""
        return self._versions.get(version_id)
    
    def list_versions(
        self,
        limit: Optional[int] = None,
        reverse: bool = True
    ) -> List[ModelVersion]:
        """
        列出版本历史
        
        Args:
            limit: 限制返回数量
            reverse: 是否按时间倒序
        """
        versions = list(self._versions.values())
        versions.sort(key=lambda v: v.version_id, reverse=reverse)
        
        if limit:
            versions = versions[:limit]
        
        return versions
    
    def create_checkpoint(
        self,
        checkpoint_id: str,
        model_params: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> Checkpoint:
        """
        创建检查点
        
        Args:
            checkpoint_id: 检查点ID
            model_params: 模型参数，None则使用最新版本
            tags: 标签列表
        """
        if model_params is None:
            model_params = self._latest_params
        
        if model_params is None:
            raise ValueError("没有可用的模型参数")
        
        timestamp = datetime.now().timestamp()
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            model_params=copy.deepcopy(model_params),
            version=self._current_version,
            timestamp=timestamp
        )
        checkpoint.tags = tags or []
        
        self._checkpoints[checkpoint_id] = checkpoint
        return checkpoint
    
    def restore_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """从检查点恢复"""
        if checkpoint_id in self._checkpoints:
            return copy.deepcopy(self._checkpoints[checkpoint_id].model_params)
        return None
    
    def list_checkpoints(self, tag: Optional[str] = None) -> List[Checkpoint]:
        """
        列出检查点
        
        Args:
            tag: 按标签过滤
        """
        checkpoints = list(self._checkpoints.values())
        
        if tag:
            checkpoints = [cp for cp in checkpoints if tag in cp.tags]
        
        checkpoints.sort(key=lambda cp: cp.timestamp, reverse=True)
        return checkpoints
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """删除检查点"""
        if checkpoint_id in self._checkpoints:
            del self._checkpoints[checkpoint_id]
            return True
        return False
    
    def rollback(self, version_id: int) -> bool:
        """
        回滚到指定版本
        
        Args:
            version_id: 目标版本ID
        """
        if version_id not in self._models:
            return False
        
        self._latest_params = copy.deepcopy(self._models[version_id])
        return True
    
    def diff_versions(
        self,
        version_id1: int,
        version_id2: int
    ) -> Dict[str, Tuple[Any, Any]]:
        """
        比较两个版本的差异
        
        Returns:
            参数名 -> (旧值, 新值) 的差异字典
        """
        params1 = self._models.get(version_id1, {})
        params2 = self._models.get(version_id2, {})
        
        diff: Dict[str, Tuple[Any, Any]] = {}
        all_keys = set(params1.keys()) | set(params2.keys())
        
        for key in all_keys:
            val1 = params1.get(key)
            val2 = params2.get(key)
            
            if val1 != val2:
                diff[key] = (val1, val2)
        
        return diff
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        return {
            'total_versions': len(self._versions),
            'total_checkpoints': len(self._checkpoints),
            'current_version': self._current_version,
            'total_stored': self._total_stored,
            'total_size_bytes': self._total_size,
            'max_versions': self.max_versions
        }
    
    def export_version(
        self,
        version_id: int,
        export_path: str
    ) -> bool:
        """导出版本到文件"""
        if version_id not in self._models:
            return False
        
        data = {
            'params': self._models[version_id],
            'version_info': self._versions[version_id].to_dict()
        }
        
        try:
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            with open(export_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception:
            return False
    
    def import_version(
        self,
        import_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[int]:
        """从文件导入版本"""
        try:
            with open(import_path, 'r') as f:
                data = json.load(f)
            
            return self.save(
                data['params'],
                metadata=metadata or data.get('version_info', {}).get('metadata')
            )
        except Exception:
            return None
    
    def _cleanup_old_versions(self) -> None:
        """清理超出限制的旧版本"""
        if len(self._models) <= self.max_versions:
            return
        
        # 按版本ID排序
        sorted_versions = sorted(self._versions.keys())
        
        # 删除最旧的版本
        while len(self._models) > self.max_versions:
            oldest = sorted_versions.pop(0)
            size = self._versions[oldest].size_bytes
            del self._models[oldest]
            del self._versions[oldest]
            self._total_size -= size
    
    def _estimate_size(self, params: Dict[str, Any]) -> int:
        """估算参数大小"""
        def _count_size(obj: Any) -> int:
            if isinstance(obj, (int, float)):
                return 8
            elif isinstance(obj, str):
                return len(obj)
            elif isinstance(obj, list):
                return sum(_count_size(item) for item in obj)
            elif isinstance(obj, dict):
                return sum(_count_size(v) for v in obj.values())
            return 0
        
        return _count_size(params)
    
    def _compute_checksum(self, params: Dict[str, Any]) -> str:
        """计算参数校验和"""
        import hashlib
        
        # 简化的校验和计算
        content = json.dumps(params, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:16]


class ModelRegistry:
    """
    模型注册表
    
    管理多个模型的存储
    """
    
    def __init__(self):
        self._stores: Dict[str, ModelStore] = {}
    
    def get_store(self, model_name: str) -> ModelStore:
        """获取或创建模型存储"""
        if model_name not in self._stores:
            self._stores[model_name] = ModelStore()
        return self._stores[model_name]
    
    def list_models(self) -> List[str]:
        """列出所有模型"""
        return list(self._stores.keys())
    
    def delete_model(self, model_name: str) -> bool:
        """删除模型存储"""
        if model_name in self._stores:
            del self._stores[model_name]
            return True
        return False
