"""
NSGA-II 多目标优化算法

Non-dominated Sorting Genetic Algorithm II

用于同时优化多个目标（如精度、速度、内存占用）。
"""

import numpy as np
from typing import List, Tuple, Callable, Optional, Dict, Any
from dataclasses import dataclass
import random


@dataclass
class NSGA2Config:
    """NSGA-II配置"""
    population_size: int = 100
    max_generations: int = 100
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1
    eta_c: float = 15.0  # 分布指数（交叉）
    eta_m: float = 20.0  # 分布指数（变异）
    num_objectives: int = 2
    num_variables: int = 10
    lower_bounds: Optional[List[float]] = None
    upper_bounds: Optional[List[float]] = None


class Individual:
    """个体"""
    
    def __init__(self, genes: np.ndarray):
        self.genes = genes
        self.objectives: np.ndarray = np.array([])
        self.rank: int = -1
        self.crowding_distance: float = 0.0
    
    def __lt__(self, other: 'Individual') -> bool:
        """支配比较"""
        # self支配other：所有目标不差，至少一个更好
        return np.all(self.objectives <= other.objectives) and \
               np.any(self.objectives < other.objectives)
    
    def dominates(self, other: 'Individual') -> bool:
        """self是否支配other"""
        return np.all(self.objectives <= other.objectives) and \
               np.any(self.objectives < other.objectives)


def fast_non_dominated_sort(population: List[Individual]) -> List[List[Individual]]:
    """
    快速非支配排序
    
    将种群分成多个前沿层：
    - 第1层：不被任何个体支配
    - 第2层：去掉第1层后不被支配
    - ...
    """
    n = len(population)
    
    # 每个个体被多少个体支配
    domination_count = [0] * n
    
    # 每个个体支配哪些个体
    dominated_set = [[] for _ in range(n)]
    
    # 前沿层
    fronts = [[]]
    
    for i in range(n):
        for j in range(i + 1, n):
            if population[i].dominates(population[j]):
                domination_count[j] += 1
                dominated_set[i].append(j)
            elif population[j].dominates(population[i]):
                domination_count[i] += 1
                dominated_set[j].append(i)
        
        if domination_count[i] == 0:
            population[i].rank = 0
            fronts[0].append(i)
    
    # 构建后续前沿
    i = 0
    while fronts[i]:
        next_front = []
        for p_idx in fronts[i]:
            for q_idx in dominated_set[p_idx]:
                domination_count[q_idx] -= 1
                if domination_count[q_idx] == 0:
                    population[q_idx].rank = i + 1
                    next_front.append(q_idx)
        i += 1
        fronts.append(next_front)
    
    # 转换为个体列表
    result = []
    for front in fronts[:-1]:  # 最后一层是空的
        result.append([population[i] for i in front])
    
    return result


def compute_crowding_distance(front: List[Individual]) -> None:
    """
    计算拥挤距离
    
    用于在同一前沿内区分个体密度。
    拥挤距离大的个体更稀疏，应优先保留。
    """
    n = len(front)
    if n <= 2:
        for ind in front:
            ind.crowding_distance = float('inf')
        return
    
    num_objectives = len(front[0].objectives)
    
    for ind in front:
        ind.crowding_distance = 0.0
    
    for m in range(num_objectives):
        # 按第m个目标排序
        front.sort(key=lambda ind: ind.objectives[m])
        
        # 边界个体距离设为无穷
        front[0].crowding_distance = float('inf')
        front[-1].crowding_distance = float('inf')
        
        # 目标范围
        f_min = front[0].objectives[m]
        f_max = front[-1].objectives[m]
        
        if f_max - f_min < 1e-10:
            continue
        
        # 计算拥挤距离
        for i in range(1, n - 1):
            front[i].crowding_distance += \
                (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / (f_max - f_min)


def crowded_comparison(a: Individual, b: Individual) -> int:
    """
    拥挤比较算子
    
    返回：
    - -1: a优于b
    - 1: b优于a
    - 0: 相等
    """
    if a.rank < b.rank:
        return -1
    elif a.rank > b.rank:
        return 1
    elif a.crowding_distance > b.crowding_distance:
        return -1
    elif a.crowding_distance < b.crowding_distance:
        return 1
    else:
        return 0


def sbx_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    eta: float,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    模拟二进制交叉 (Simulated Binary Crossover)
    
    产生两个子代，保持父代的搜索能力。
    """
    child1 = np.copy(parent1)
    child2 = np.copy(parent2)
    
    for i in range(len(parent1)):
        if random.random() < 0.5:
            if abs(parent1[i] - parent2[i]) > 1e-10:
                if parent1[i] < parent2[i]:
                    y1, y2 = parent1[i], parent2[i]
                else:
                    y1, y2 = parent2[i], parent1[i]
                
                yl, yu = lower_bounds[i], upper_bounds[i]
                
                # 计算beta
                beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1))
                alpha = 2.0 - pow(beta, -(eta + 1.0))
                
                rand = random.random()
                if rand <= (1.0 / alpha):
                    betaq = pow((rand * alpha), (1.0 / (eta + 1.0)))
                else:
                    betaq = pow((1.0 / (2.0 - rand * alpha)), (1.0 / (eta + 1.0)))
                
                c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                
                beta = 1.0 + (2.0 * (yu - y2) / (y2 - y1))
                alpha = 2.0 - pow(beta, -(eta + 1.0))
                
                if rand <= (1.0 / alpha):
                    betaq = pow((rand * alpha), (1.0 / (eta + 1.0)))
                else:
                    betaq = pow((1.0 / (2.0 - rand * alpha)), (1.0 / (eta + 1.0)))
                
                c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
                
                # 边界检查
                c1 = max(yl, min(yu, c1))
                c2 = max(yl, min(yu, c2))
                
                child1[i] = c1
                child2[i] = c2
    
    return child1, child2


def polynomial_mutation(
    genes: np.ndarray,
    eta: float,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    prob: float
) -> np.ndarray:
    """
    多项式变异
    
    在个体附近进行小范围扰动。
    """
    mutant = np.copy(genes)
    
    for i in range(len(genes)):
        if random.random() < prob:
            y = genes[i]
            yl, yu = lower_bounds[i], upper_bounds[i]
            
            delta1 = (y - yl) / (yu - yl)
            delta2 = (yu - y) / (yu - yl)
            
            rand = random.random()
            mut_pow = 1.0 / (eta + 1.0)
            
            if rand < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * pow(xy, (eta + 1.0))
                delta = pow(val, mut_pow) - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * pow(xy, (eta + 1.0))
                delta = 1.0 - pow(val, mut_pow)
            
            y = y + delta * (yu - yl)
            mutant[i] = max(yl, min(yu, y))
    
    return mutant


class NSGA2:
    """NSGA-II算法"""
    
    def __init__(
        self,
        config: NSGA2Config,
        objective_functions: List[Callable[[np.ndarray], float]]
    ):
        self.config = config
        self.objective_functions = objective_functions
        
        # 边界
        self.lower_bounds = np.array(
            config.lower_bounds if config.lower_bounds 
            else [0.0] * config.num_variables
        )
        self.upper_bounds = np.array(
            config.upper_bounds if config.upper_bounds 
            else [1.0] * config.num_variables
        )
        
        # 种群
        self.population: List[Individual] = []
        
        # 统计
        self.history = {
            'best_front_size': [],
            'hypervolume': [],
            'convergence': []
        }
    
    def _evaluate(self, individual: Individual) -> None:
        """评估个体的目标函数值"""
        objectives = []
        for func in self.objective_functions:
            objectives.append(func(individual.genes))
        individual.objectives = np.array(objectives)
    
    def _initialize_population(self) -> None:
        """初始化种群"""
        self.population = []
        
        for _ in range(self.config.population_size):
            genes = np.random.uniform(
                self.lower_bounds,
                self.upper_bounds
            )
            ind = Individual(genes)
            self._evaluate(ind)
            self.population.append(ind)
    
    def _selection(self) -> Individual:
        """锦标赛选择"""
        candidates = random.sample(self.population, 2)
        if crowded_comparison(candidates[0], candidates[1]) <= 0:
            return candidates[0]
        return candidates[1]
    
    def _create_offspring(self) -> List[Individual]:
        """创建子代"""
        offspring = []
        
        while len(offspring) < self.config.population_size:
            parent1 = self._selection()
            parent2 = self._selection()
            
            if random.random() < self.config.crossover_prob:
                child1_genes, child2_genes = sbx_crossover(
                    parent1.genes, parent2.genes,
                    self.config.eta_c,
                    self.lower_bounds, self.upper_bounds
                )
            else:
                child1_genes = parent1.genes.copy()
                child2_genes = parent2.genes.copy()
            
            child1_genes = polynomial_mutation(
                child1_genes, self.config.eta_m,
                self.lower_bounds, self.upper_bounds,
                self.config.mutation_prob / self.config.num_variables
            )
            child2_genes = polynomial_mutation(
                child2_genes, self.config.eta_m,
                self.lower_bounds, self.upper_bounds,
                self.config.mutation_prob / self.config.num_variables
            )
            
            child1 = Individual(child1_genes)
            child2 = Individual(child2_genes)
            
            self._evaluate(child1)
            self._evaluate(child2)
            
            offspring.append(child1)
            offspring.append(child2)
        
        return offspring[:self.config.population_size]
    
    def _select_new_population(
        self,
        combined: List[Individual]
    ) -> List[Individual]:
        """选择新种群"""
        fronts = fast_non_dominated_sort(combined)
        
        new_population = []
        front_idx = 0
        
        while len(new_population) + len(fronts[front_idx]) <= self.config.population_size:
            compute_crowding_distance(fronts[front_idx])
            new_population.extend(fronts[front_idx])
            front_idx += 1
            
            if front_idx >= len(fronts):
                break
        
        # 如果还需要更多个体
        if len(new_population) < self.config.population_size and front_idx < len(fronts):
            compute_crowding_distance(fronts[front_idx])
            fronts[front_idx].sort(
                key=lambda ind: ind.crowding_distance,
                reverse=True
            )
            needed = self.config.population_size - len(new_population)
            new_population.extend(fronts[front_idx][:needed])
        
        return new_population
    
    def _compute_hypervolume(self, front: List[Individual], reference: np.ndarray) -> float:
        """计算超体积（简化版，仅2目标）"""
        if len(front) == 0:
            return 0.0
        
        if self.config.num_objectives == 2:
            # 2目标：排序后计算面积
            points = [(ind.objectives[0], ind.objectives[1]) for ind in front]
            points.sort()
            
            hv = 0.0
            prev_x = reference[0]
            
            for x, y in points:
                if x < reference[0] and y < reference[1]:
                    hv += (reference[0] - x) * (reference[1] - y)
                    prev_x = x
            
            return hv
        
        return 0.0
    
    def run(self) -> Tuple[List[Individual], Dict[str, Any]]:
        """
        运行NSGA-II
        
        Returns:
            Pareto前沿个体列表
            运行历史
        """
        self._initialize_population()
        
        for gen in range(self.config.max_generations):
            # 创建子代
            offspring = self._create_offspring()
            
            # 合并父代和子代
            combined = self.population + offspring
            
            # 选择新种群
            self.population = self._select_new_population(combined)
            
            # 记录统计
            fronts = fast_non_dominated_sort(self.population)
            self.history['best_front_size'].append(len(fronts[0]))
            
            # 计算超体积
            reference = np.array([1e10] * self.config.num_objectives)
            hv = self._compute_hypervolume(fronts[0], reference)
            self.history['hypervolume'].append(hv)
            
            if gen % 10 == 0:
                print(f"Generation {gen}: Front size = {len(fronts[0])}, HV = {hv:.4f}")
        
        # 返回Pareto前沿
        fronts = fast_non_dominated_sort(self.population)
        return fronts[0], self.history
    
    def get_pareto_front(self) -> np.ndarray:
        """获取Pareto前沿的目标值"""
        fronts = fast_non_dominated_sort(self.population)
        if not fronts:
            return np.array([])
        
        return np.array([ind.objectives for ind in fronts[0]])
    
    def get_best_solution(self, weights: Optional[List[float]] = None) -> Individual:
        """
        从Pareto前沿选择最佳解
        
        Args:
            weights: 目标权重（加权和）
            
        Returns:
            最佳个体
        """
        fronts = fast_non_dominated_sort(self.population)
        if not fronts or not fronts[0]:
            return self.population[0]
        
        if weights is None:
            weights = [1.0 / self.config.num_objectives] * self.config.num_objectives
        
        best = None
        best_score = float('inf')
        
        for ind in fronts[0]:
            score = sum(w * obj for w, obj in zip(weights, ind.objectives))
            if score < best_score:
                best_score = score
                best = ind
        
        return best
