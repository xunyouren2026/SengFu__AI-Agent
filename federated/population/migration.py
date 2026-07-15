"""
跨节点模型迁移
"""
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime
from enum import Enum
import copy


class MigrationStatus(Enum):
    """迁移状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelMigration:
    """模型迁移"""
    
    def __init__(
        self,
        migration_id: str,
        source_node: str,
        target_node: str,
        model_params: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.migration_id = migration_id
        self.source_node = source_node
        self.target_node = target_node
        self.model_params = model_params
        self.metadata = metadata or {}
        self.status = MigrationStatus.PENDING
        self.created_at = datetime.now().timestamp()
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.bytes_transferred: int = 0
    
    def start(self) -> None:
        """开始迁移"""
        self.status = MigrationStatus.IN_PROGRESS
        self.started_at = datetime.now().timestamp()
    
    def complete(self) -> None:
        """完成迁移"""
        self.status = MigrationStatus.COMPLETED
        self.completed_at = datetime.now().timestamp()
    
    def fail(self) -> None:
        """迁移失败"""
        self.status = MigrationStatus.FAILED
        self.completed_at = datetime.now().timestamp()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'migration_id': self.migration_id,
            'source_node': self.source_node,
            'target_node': self.target_node,
            'status': self.status.value,
            'created_at': self.created_at,
            'bytes_transferred': self.bytes_transferred
        }


class MigrationScheduler:
    """
    迁移调度器
    
    管理跨节点模型迁移
    """
    
    def __init__(
        self,
        max_concurrent: int = 5,
        bandwidth_limit: float = 100.0  # MB/s
    ):
        self.max_concurrent = max_concurrent
        self.bandwidth_limit = bandwidth_limit
        
        self._pending: List[ModelMigration] = []
        self._active: Dict[str, ModelMigration] = {}
        self._completed: List[ModelMigration] = []
        self._failed: List[ModelMigration] = []
        
        self._total_migrations = 0
        self._total_bytes = 0
    
    def submit(
        self,
        source_node: str,
        target_node: str,
        model_params: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """提交迁移请求"""
        migration_id = f"mig_{self._total_migrations}_{int(datetime.now().timestamp())}"
        
        migration = ModelMigration(
            migration_id=migration_id,
            source_node=source_node,
            target_node=target_node,
            model_params=model_params,
            metadata=metadata
        )
        
        self._pending.append(migration)
        self._total_migrations += 1
        
        return migration_id
    
    def schedule(self) -> List[ModelMigration]:
        """调度迁移"""
        started = []
        
        while len(self._active) < self.max_concurrent and self._pending:
            migration = self._pending.pop(0)
            migration.start()
            self._active[migration.migration_id] = migration
            started.append(migration)
        
        return started
    
    def complete(
        self,
        migration_id: str,
        bytes_transferred: int = 0
    ) -> Optional[ModelMigration]:
        """完成迁移"""
        if migration_id not in self._active:
            return None
        
        migration = self._active.pop(migration_id)
        migration.bytes_transferred = bytes_transferred
        migration.complete()
        
        self._completed.append(migration)
        self._total_bytes += bytes_transferred
        
        return migration
    
    def fail(self, migration_id: str) -> Optional[ModelMigration]:
        """迁移失败"""
        if migration_id not in self._active:
            return None
        
        migration = self._active.pop(migration_id)
        migration.fail()
        self._failed.append(migration)
        
        return migration
    
    def get_migration(self, migration_id: str) -> Optional[ModelMigration]:
        """获取迁移"""
        if migration_id in self._active:
            return self._active[migration_id]
        
        for m in self._pending:
            if m.migration_id == migration_id:
                return m
        
        for m in self._completed:
            if m.migration_id == migration_id:
                return m
        
        for m in self._failed:
            if m.migration_id == migration_id:
                return m
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_migrations': self._total_migrations,
            'pending': len(self._pending),
            'active': len(self._active),
            'completed': len(self._completed),
            'failed': len(self._failed),
            'total_bytes': self._total_bytes
        }


class ModelMigrator:
    """
    模型迁移器
    
    处理模型参数的序列化和传输
    """
    
    def __init__(self):
        self._scheduler = MigrationScheduler()
        self._node_models: Dict[str, Dict[str, Any]] = {}  # node_id -> model_params
    
    def register_node(
        self,
        node_id: str,
        model_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """注册节点"""
        self._node_models[node_id] = model_params or {}
    
    def unregister_node(self, node_id: str) -> None:
        """注销节点"""
        self._node_models.pop(node_id, None)
    
    def migrate(
        self,
        source_node: str,
        target_node: str,
        model_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        发起迁移
        
        Args:
            source_node: 源节点
            target_node: 目标节点
            model_params: 模型参数，None则使用源节点当前模型
        
        Returns:
            迁移ID
        """
        if model_params is None:
            model_params = self._node_models.get(source_node, {})
        
        return self._scheduler.submit(
            source_node=source_node,
            target_node=target_node,
            model_params=model_params
        )
    
    def receive(
        self,
        migration_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        接收迁移的模型
        
        Args:
            migration_id: 迁移ID
        
        Returns:
            模型参数
        """
        migration = self._scheduler.get_migration(migration_id)
        
        if migration is None or migration.status != MigrationStatus.COMPLETED:
            return None
        
        # 存储到目标节点
        target_node = migration.target_node
        if target_node in self._node_models:
            self._node_models[target_node] = copy.deepcopy(migration.model_params)
        
        return migration.model_params
    
    def tick(self) -> List[ModelMigration]:
        """时钟滴答"""
        return self._scheduler.schedule()
    
    def get_node_model(self, node_id: str) -> Optional[Dict[str, Any]]:
        """获取节点模型"""
        return self._node_models.get(node_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'num_nodes': len(self._node_models),
            'scheduler_stats': self._scheduler.get_statistics()
        }
