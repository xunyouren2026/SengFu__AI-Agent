"""
分布式种群档案
"""
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime
from enum import Enum
import copy


class ArchiveStatus(Enum):
    """档案状态"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MIGRATING = "migrating"


class Individual:
    """个体"""
    
    def __init__(
        self,
        individual_id: str,
        genome: Dict[str, Any],
        fitness: float = 0.0,
        source_node: Optional[str] = None
    ):
        self.individual_id = individual_id
        self.genome = genome
        self.fitness = fitness
        self.source_node = source_node
        self.created_at = datetime.now().timestamp()
        self.updated_at = self.created_at
        self.generation: int = 0
        self.metadata: Dict[str, Any] = {}
    
    def update_fitness(self, fitness: float) -> None:
        """更新适应度"""
        self.fitness = fitness
        self.updated_at = datetime.now().timestamp()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'individual_id': self.individual_id,
            'fitness': self.fitness,
            'source_node': self.source_node,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'generation': self.generation
        }


class DistributedArchive:
    """
    分布式种群档案
    
    存储和管理分布式种群
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        migration_interval: int = 10
    ):
        self.max_size = max_size
        self.migration_interval = migration_interval
        
        self._individuals: Dict[str, Individual] = {}
        self._node_archives: Dict[str, Set[str]] = {}  # node_id -> individual_ids
        self._generation: int = 0
        
        # 统计
        self._total_added = 0
        self._total_removed = 0
        self._total_migrations = 0
    
    def add(
        self,
        individual: Individual,
        node_id: Optional[str] = None
    ) -> bool:
        """添加个体"""
        if len(self._individuals) >= self.max_size:
            self._evict_worst()
        
        self._individuals[individual.individual_id] = individual
        self._total_added += 1
        
        # 按节点索引
        if node_id:
            if node_id not in self._node_archives:
                self._node_archives[node_id] = set()
            self._node_archives[node_id].add(individual.individual_id)
            individual.source_node = node_id
        
        return True
    
    def remove(self, individual_id: str) -> Optional[Individual]:
        """移除个体"""
        if individual_id not in self._individuals:
            return None
        
        individual = self._individuals.pop(individual_id)
        self._total_removed += 1
        
        # 从节点索引中移除
        for node_id, ids in self._node_archives.items():
            ids.discard(individual_id)
        
        return individual
    
    def get(self, individual_id: str) -> Optional[Individual]:
        """获取个体"""
        return self._individuals.get(individual_id)
    
    def get_all(self) -> List[Individual]:
        """获取所有个体"""
        return list(self._individuals.values())
    
    def get_by_node(self, node_id: str) -> List[Individual]:
        """按节点获取个体"""
        ids = self._node_archives.get(node_id, set())
        return [self._individuals[iid] for iid in ids if iid in self._individuals]
    
    def get_top(self, k: int = 10) -> List[Individual]:
        """获取最优个体"""
        sorted_individuals = sorted(
            self._individuals.values(),
            key=lambda x: x.fitness,
            reverse=True
        )
        return sorted_individuals[:k]
    
    def get_worst(self, k: int = 10) -> List[Individual]:
        """获取最差个体"""
        sorted_individuals = sorted(
            self._individuals.values(),
            key=lambda x: x.fitness
        )
        return sorted_individuals[:k]
    
    def _evict_worst(self) -> None:
        """淘汰最差个体"""
        worst = self.get_worst(1)
        if worst:
            self.remove(worst[0].individual_id)
    
    def migrate(
        self,
        source_node: str,
        target_node: str,
        num_individuals: int = 5
    ) -> List[Individual]:
        """
        迁移个体
        
        Args:
            source_node: 源节点
            target_node: 目标节点
            num_individuals: 迁移数量
        """
        source_individuals = self.get_by_node(source_node)
        
        # 选择最优个体迁移
        source_individuals.sort(key=lambda x: x.fitness, reverse=True)
        to_migrate = source_individuals[:num_individuals]
        
        migrated = []
        for ind in to_migrate:
            # 创建副本
            new_ind = Individual(
                individual_id=f"{ind.individual_id}_mig{self._total_migrations}",
                genome=copy.deepcopy(ind.genome),
                fitness=ind.fitness,
                source_node=target_node
            )
            new_ind.generation = ind.generation
            
            self.add(new_ind, target_node)
            migrated.append(new_ind)
        
        self._total_migrations += len(migrated)
        return migrated
    
    def merge(
        self,
        other_archive: 'DistributedArchive',
        keep_best: bool = True
    ) -> int:
        """
        合并另一个档案
        
        Returns:
            合并的个体数量
        """
        merged = 0
        
        for individual in other_archive.get_all():
            if individual.individual_id not in self._individuals:
                if keep_best and len(self._individuals) >= self.max_size:
                    # 只保留更好的
                    worst = self.get_worst(1)
                    if worst and individual.fitness > worst[0].fitness:
                        self.remove(worst[0].individual_id)
                        self.add(individual, individual.source_node)
                        merged += 1
                else:
                    self.add(individual, individual.source_node)
                    merged += 1
        
        return merged
    
    def advance_generation(self) -> int:
        """推进代数"""
        self._generation += 1
        
        for individual in self._individuals.values():
            individual.generation = self._generation
        
        return self._generation
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        fitnesses = [i.fitness for i in self._individuals.values()]
        
        return {
            'size': len(self._individuals),
            'max_size': self.max_size,
            'generation': self._generation,
            'num_nodes': len(self._node_archives),
            'total_added': self._total_added,
            'total_removed': self._total_removed,
            'total_migrations': self._total_migrations,
            'avg_fitness': sum(fitnesses) / len(fitnesses) if fitnesses else 0,
            'max_fitness': max(fitnesses) if fitnesses else 0,
            'min_fitness': min(fitnesses) if fitnesses else 0
        }


class ArchiveSynchronizer:
    """
    档案同步器
    
    同步分布式档案
    """
    
    def __init__(self, sync_interval: int = 100):
        self.sync_interval = sync_interval
        self._archives: Dict[str, DistributedArchive] = {}
        self._sync_count: int = 0
    
    def register_archive(
        self,
        node_id: str,
        archive: DistributedArchive
    ) -> None:
        """注册档案"""
        self._archives[node_id] = archive
    
    def unregister_archive(self, node_id: str) -> None:
        """注销档案"""
        self._archives.pop(node_id, None)
    
    def sync(self) -> Dict[str, int]:
        """
        同步所有档案
        
        Returns:
            node_id -> 同步的个体数量
        """
        sync_results: Dict[str, int] = {}
        
        # 收集所有最优个体
        all_best: List[Individual] = []
        for archive in self._archives.values():
            all_best.extend(archive.get_top(10))
        
        # 分发到所有档案
        for node_id, archive in self._archives.items():
            count = 0
            for ind in all_best:
                if ind.individual_id not in archive._individuals:
                    new_ind = Individual(
                        individual_id=f"{ind.individual_id}_sync{self._sync_count}",
                        genome=copy.deepcopy(ind.genome),
                        fitness=ind.fitness,
                        source_node=node_id
                    )
                    archive.add(new_ind, node_id)
                    count += 1
            
            sync_results[node_id] = count
        
        self._sync_count += 1
        return sync_results
    
    def should_sync(self) -> bool:
        """是否应该同步"""
        return self._sync_count % self.sync_interval == 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'num_archives': len(self._archives),
            'sync_count': self._sync_count,
            'sync_interval': self.sync_interval
        }
