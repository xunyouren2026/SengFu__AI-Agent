"""
权重交叉
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import random
import copy


class CrossoverType(Enum):
    """交叉类型"""
    UNIFORM = "uniform"  # 均匀交叉
    SINGLE_POINT = "single_point"  # 单点交叉
    TWO_POINT = "two_point"  # 两点交叉
    ARITHMETIC = "arithmetic"  # 算术交叉
    BLEND = "blend"  # 混合交叉


class WeightCrossover:
    """
    权重交叉
    
    在联邦学习中交叉不同模型的权重
    """
    
    def __init__(
        self,
        crossover_type: CrossoverType = CrossoverType.UNIFORM,
        crossover_rate: float = 0.5
    ):
        self.crossover_type = crossover_type
        self.crossover_rate = crossover_rate
    
    def crossover(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        执行交叉
        
        Args:
            parent1: 父代1的权重
            parent2: 父代2的权重
        
        Returns:
            (子代1, 子代2)
        """
        if self.crossover_type == CrossoverType.UNIFORM:
            return self._uniform_crossover(parent1, parent2)
        elif self.crossover_type == CrossoverType.SINGLE_POINT:
            return self._single_point_crossover(parent1, parent2)
        elif self.crossover_type == CrossoverType.TWO_POINT:
            return self._two_point_crossover(parent1, parent2)
        elif self.crossover_type == CrossoverType.ARITHMETIC:
            return self._arithmetic_crossover(parent1, parent2)
        elif self.crossover_type == CrossoverType.BLEND:
            return self._blend_crossover(parent1, parent2)
        else:
            return self._uniform_crossover(parent1, parent2)
    
    def _uniform_crossover(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """均匀交叉"""
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        all_keys = set(parent1.keys()) | set(parent2.keys())
        
        for key in all_keys:
            if random.random() < self.crossover_rate:
                # 交换
                child1[key] = copy.deepcopy(parent2.get(key))
                child2[key] = copy.deepcopy(parent1.get(key))
        
        return child1, child2
    
    def _single_point_crossover(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """单点交叉"""
        keys = sorted(set(parent1.keys()) | set(parent2.keys()))
        
        if len(keys) < 2:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        # 随机选择交叉点
        crossover_point = random.randint(1, len(keys) - 1)
        
        child1: Dict[str, Any] = {}
        child2: Dict[str, Any] = {}
        
        for i, key in enumerate(keys):
            if i < crossover_point:
                child1[key] = copy.deepcopy(parent1.get(key))
                child2[key] = copy.deepcopy(parent2.get(key))
            else:
                child1[key] = copy.deepcopy(parent2.get(key))
                child2[key] = copy.deepcopy(parent1.get(key))
        
        return child1, child2
    
    def _two_point_crossover(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """两点交叉"""
        keys = sorted(set(parent1.keys()) | set(parent2.keys()))
        
        if len(keys) < 3:
            return self._single_point_crossover(parent1, parent2)
        
        # 随机选择两个交叉点
        points = sorted(random.sample(range(1, len(keys)), 2))
        
        child1: Dict[str, Any] = {}
        child2: Dict[str, Any] = {}
        
        for i, key in enumerate(keys):
            if i < points[0] or i >= points[1]:
                child1[key] = copy.deepcopy(parent1.get(key))
                child2[key] = copy.deepcopy(parent2.get(key))
            else:
                child1[key] = copy.deepcopy(parent2.get(key))
                child2[key] = copy.deepcopy(parent1.get(key))
        
        return child1, child2
    
    def _arithmetic_crossover(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """算术交叉"""
        alpha = self.crossover_rate
        
        child1: Dict[str, Any] = {}
        child2: Dict[str, Any] = {}
        
        all_keys = set(parent1.keys()) | set(parent2.keys())
        
        for key in all_keys:
            v1 = parent1.get(key, 0)
            v2 = parent2.get(key, 0)
            
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                child1[key] = alpha * v1 + (1 - alpha) * v2
                child2[key] = (1 - alpha) * v1 + alpha * v2
            elif isinstance(v1, list) and isinstance(v2, list):
                child1[key] = [
                    alpha * a + (1 - alpha) * b
                    for a, b in zip(v1, v2)
                ]
                child2[key] = [
                    (1 - alpha) * a + alpha * b
                    for a, b in zip(v1, v2)
                ]
            else:
                child1[key] = copy.deepcopy(v1)
                child2[key] = copy.deepcopy(v2)
        
        return child1, child2
    
    def _blend_crossover(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """混合交叉 (BLX-alpha)"""
        alpha = 0.5  # BLX参数
        
        child1: Dict[str, Any] = {}
        child2: Dict[str, Any] = {}
        
        all_keys = set(parent1.keys()) | set(parent2.keys())
        
        for key in all_keys:
            v1 = parent1.get(key, 0)
            v2 = parent2.get(key, 0)
            
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                min_val = min(v1, v2)
                max_val = max(v1, v2)
                range_val = max_val - min_val
                
                low = min_val - alpha * range_val
                high = max_val + alpha * range_val
                
                child1[key] = random.uniform(low, high)
                child2[key] = random.uniform(low, high)
            elif isinstance(v1, list) and isinstance(v2, list):
                child1[key] = []
                child2[key] = []
                
                for a, b in zip(v1, v2):
                    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                        min_val = min(a, b)
                        max_val = max(a, b)
                        range_val = max_val - min_val
                        
                        low = min_val - alpha * range_val
                        high = max_val + alpha * range_val
                        
                        child1[key].append(random.uniform(low, high))
                        child2[key].append(random.uniform(low, high))
                    else:
                        child1[key].append(a)
                        child2[key].append(b)
            else:
                child1[key] = copy.deepcopy(v1)
                child2[key] = copy.deepcopy(v2)
        
        return child1, child2


class PopulationCrossover:
    """
    种群交叉
    
    管理多个模型的交叉操作
    """
    
    def __init__(
        self,
        crossover: Optional[WeightCrossover] = None,
        selection_pressure: float = 1.5
    ):
        self.crossover = crossover or WeightCrossover()
        self.selection_pressure = selection_pressure
        
        self._population: Dict[str, Dict[str, Any]] = {}
        self._fitness: Dict[str, float] = {}
        self._offspring: List[Dict[str, Any]] = []
    
    def add_individual(
        self,
        individual_id: str,
        weights: Dict[str, Any],
        fitness: float = 0.0
    ) -> None:
        """添加个体"""
        self._population[individual_id] = weights
        self._fitness[individual_id] = fitness
    
    def remove_individual(self, individual_id: str) -> None:
        """移除个体"""
        self._population.pop(individual_id, None)
        self._fitness.pop(individual_id, None)
    
    def select_parents(self) -> Tuple[str, str]:
        """
        选择父代
        
        使用锦标赛选择
        """
        if len(self._population) < 2:
            raise ValueError("种群大小不足")
        
        # 锦标赛选择
        def tournament():
            candidates = random.sample(
                list(self._population.keys()),
                min(3, len(self._population))
            )
            best = max(candidates, key=lambda x: self._fitness.get(x, 0))
            return best
        
        parent1 = tournament()
        parent2 = tournament()
        
        # 确保不同
        while parent2 == parent1 and len(self._population) > 1:
            parent2 = tournament()
        
        return parent1, parent2
    
    def generate_offspring(
        self,
        num_offspring: int
    ) -> List[Dict[str, Any]]:
        """
        生成后代
        
        Args:
            num_offspring: 后代数量
        """
        self._offspring = []
        
        for _ in range(num_offspring // 2):
            try:
                p1_id, p2_id = self.select_parents()
                parent1 = self._population[p1_id]
                parent2 = self._population[p2_id]
                
                child1, child2 = self.crossover.crossover(parent1, parent2)
                
                self._offspring.append(child1)
                self._offspring.append(child2)
            except ValueError:
                break
        
        return self._offspring
    
    def get_offspring(self) -> List[Dict[str, Any]]:
        """获取后代"""
        return self._offspring.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'population_size': len(self._population),
            'offspring_size': len(self._offspring),
            'crossover_type': self.crossover.crossover_type.value,
            'avg_fitness': sum(self._fitness.values()) / len(self._fitness) if self._fitness else 0
        }
