"""
Swarm Intelligence and Evolutionary Computation Module
群智能与进化计算模块

A comprehensive implementation of swarm intelligence and evolutionary algorithms
for optimization problems. All algorithms are implemented in pure Python.

包含算法:
1. Particle Swarm Optimization (PSO) - 粒子群优化
2. Ant Colony Optimization (ACO) - 蚁群优化
3. Artificial Bee Colony (ABC) - 人工蜂群
4. Firefly Algorithm (FA) - 萤火虫算法
5. Cuckoo Search (CS) - 布谷鸟搜索
6. Bat Algorithm (BA) - 蝙蝠算法
7. Grey Wolf Optimizer (GWO) - 灰狼优化
8. Whale Optimization Algorithm (WOA) - 鲸鱼优化
9. Differential Evolution (DE) - 差分进化
10. Evolution Strategy (ES) - 进化策略
11. Genetic Algorithm (GA) - 遗传算法
12. Multi-Objective variants - 多目标变体

Author: AGI Unified Framework
Version: 1.0.0
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, Generic, List, Optional, Protocol, Tuple, TypeVar, Union

# Type definitions
T = TypeVar('T')
FitnessFunction = Callable[[List[float]], float]
MultiObjectiveFunction = Callable[[List[float]], List[float]]


# =============================================================================
# Utility Functions
# =============================================================================

def clamp(value: float, min_val: float, max_val: float) -> float:
    """限制值在指定范围内"""
    return max(min_val, min(max_val, value))


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """计算欧几里得距离"""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def levy_flight(beta: float = 1.5, size: int = 1) -> List[float]:
    """
    生成Lévy飞行步长
    Lévy分布: step = u / |v|^(1/beta)
    其中 u ~ N(0, sigma_u^2), v ~ N(0, sigma_v^2)
    """
    # 计算标准差
    sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
               (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    sigma_v = 1.0

    steps = []
    for _ in range(size):
        u = random.gauss(0, sigma_u)
        v = random.gauss(0, sigma_v)
        step = u / (abs(v) ** (1 / beta))
        steps.append(step)
    return steps if size > 1 else steps[0]


def softmax(values: List[float]) -> List[float]:
    """Softmax函数，用于概率计算"""
    exp_vals = [math.exp(v - max(values)) for v in values]
    sum_exp = sum(exp_vals)
    return [v / sum_exp for v in exp_vals]


def dominates(a: List[float], b: List[float]) -> bool:
    """
    Pareto支配关系判断
    返回True如果a支配b（a在所有目标上不差于b，且至少在一个目标上严格优于b）
    """
    better_in_one = False
    for obj_a, obj_b in zip(a, b):
        if obj_a > obj_b:
            return False
        elif obj_a < obj_b:
            better_in_one = True
    return better_in_one


def non_dominated_sort(objectives: List[List[float]]) -> List[List[int]]:
    """
    非支配排序 (Non-dominated Sorting)
    返回 fronts: 每个front包含该层级的解的索引列表
    """
    n = len(objectives)
    domination_count = [0] * n
    dominated_solutions = [[] for _ in range(n)]
    fronts = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if dominates(objectives[i], objectives[j]):
                dominated_solutions[i].append(j)
                domination_count[j] += 1
            elif dominates(objectives[j], objectives[i]):
                dominated_solutions[j].append(i)
                domination_count[i] += 1

        if domination_count[i] == 0:
            fronts[0].append(i)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return fronts[:-1]  # 移除最后一个空front


def crowding_distance(objectives: List[List[float]]) -> List[float]:
    """
    计算拥挤距离 (Crowding Distance)
    用于保持解的多样性
    """
    n = len(objectives)
    if n <= 2:
        return [float('inf')] * n

    num_objectives = len(objectives[0])
    distances = [0.0] * n

    for m in range(num_objectives):
        # 按第m个目标排序
        sorted_indices = sorted(range(n), key=lambda i: objectives[i][m])
        distances[sorted_indices[0]] = distances[sorted_indices[-1]] = float('inf')

        obj_range = objectives[sorted_indices[-1]][m] - objectives[sorted_indices[0]][m]
        if obj_range > 0:
            for i in range(1, n - 1):
                distances[sorted_indices[i]] += (
                    (objectives[sorted_indices[i + 1]][m] - objectives[sorted_indices[i - 1]][m]) / obj_range
                )

    return distances


# =============================================================================
# Configuration Dataclass
# =============================================================================

@dataclass
class SwarmConfig:
    """
    群智能算法配置类
    包含各种算法的通用参数
    """
    # 通用参数
    population_size: int = 50
    max_iterations: int = 1000
    dimensions: int = 10
    bounds: Tuple[float, float] = (-5.0, 5.0)
    minimize: bool = True

    # PSO参数
    w: float = 0.729  # 惯性权重
    c1: float = 1.494  # 认知系数
    c2: float = 1.494  # 社会系数
    w_decay: bool = True  # 是否衰减惯性权重
    w_end: float = 0.4  # 最终惯性权重
    use_constriction: bool = False  # 使用收缩因子
    use_fips: bool = False  # 使用FIPS

    # ACO参数
    alpha: float = 1.0  # 信息素重要性
    beta: float = 2.0  # 启发式信息重要性
    rho: float = 0.1  # 信息素蒸发率
    Q: float = 100.0  # 信息素增量常数
    use_mmas: bool = False  # 使用Max-Min Ant System
    use_elitist: bool = False  # 使用精英策略
    tau_min: float = 0.001  # 最小信息素
    tau_max: float = 1.0  # 最大信息素

    # ABC参数
    limit: int = 100  # 食物源放弃限制

    # Firefly参数
    beta0: float = 1.0  # 初始吸引力
    gamma: float = 1.0  # 光吸收系数
    alpha_firefly: float = 0.2  # 随机项系数

    # Cuckoo参数
    pa: float = 0.25  # 巢穴发现概率
    beta_cuckoo: float = 1.5  # Lévy飞行参数

    # Bat参数
    f_min: float = 0.0  # 最小频率
    f_max: float = 2.0  # 最大频率
    A_init: float = 0.9  # 初始响度
    r_init: float = 0.5  # 初始脉冲率
    alpha_bat: float = 0.9  # 响度衰减系数
    gamma_bat: float = 0.9  # 脉冲率增加系数

    # DE参数
    F: float = 0.5  # 缩放因子
    CR: float = 0.9  # 交叉概率
    de_strategy: str = "rand/1/bin"  # DE策略

    # ES参数
    mu: int = 15  # 父代数量
    lambda_es: int = 100  # 子代数量
    selection_type: str = "plus"  # "plus" (μ+λ) 或 "comma" (μ,λ)
    use_cma: bool = False  # 使用CMA-ES

    # GA参数
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    selection_method: str = "tournament"  # tournament, roulette, rank
    crossover_method: str = "uniform"  # single_point, two_point, uniform, arithmetic
    mutation_method: str = "gaussian"  # bit_flip, gaussian, polynomial
    elitism_count: int = 2
    tournament_size: int = 3
    use_sharing: bool = False  # 使用共享函数
    sharing_sigma: float = 0.1  # 共享距离

    # 随机种子
    seed: Optional[int] = None

    def __post_init__(self):
        if self.seed is not None:
            random.seed(self.seed)


# =============================================================================
# Particle Class
# =============================================================================

class Particle:
    """
    粒子类 - 用于PSO算法

    Attributes:
        position: 当前位置
        velocity: 当前速度
        best_position: 历史最佳位置
        fitness: 当前适应度
        best_fitness: 历史最佳适应度
    """

    def __init__(self, dimensions: int, bounds: Tuple[float, float]):
        self.dimensions = dimensions
        self.bounds = bounds
        self.lower, self.upper = bounds

        # 初始化位置和速度
        self.position = [random.uniform(self.lower, self.upper) for _ in range(dimensions)]
        self.velocity = [random.uniform(-abs(self.upper - self.lower), abs(self.upper - self.lower))
                        for _ in range(dimensions)]

        # 个人最佳
        self.best_position = self.position.copy()
        self.fitness: Optional[float] = None
        self.best_fitness: Optional[float] = None

        # 邻居最佳（用于FIPS）
        self.neighbors: List[Particle] = []

    def update_velocity(self, w: float, c1: float, c2: float, global_best: List[float]) -> None:
        """
        标准PSO速度更新
        v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)
        """
        for i in range(self.dimensions):
            r1, r2 = random.random(), random.random()
            cognitive = c1 * r1 * (self.best_position[i] - self.position[i])
            social = c2 * r2 * (global_best[i] - self.position[i])
            self.velocity[i] = w * self.velocity[i] + cognitive + social

    def update_velocity_fips(self, w: float, phi: float = 4.1) -> None:
        """
        Fully Informed Particle Swarm (FIPS) 速度更新
        使用所有邻居的信息
        """
        if not self.neighbors:
            return

        for i in range(self.dimensions):
            total = w * self.velocity[i]
            for neighbor in self.neighbors:
                r = random.random()
                # FIPS: 每个邻居贡献一部分
                contribution = (phi * r / len(self.neighbors)) * (neighbor.best_position[i] - self.position[i])
                total += contribution
            self.velocity[i] = total

    def update_velocity_constriction(self, c1: float, c2: float, global_best: List[float],
                                     kappa: float = 1.0, phi: float = 4.1) -> None:
        """
        使用收缩因子的速度更新
        Clerc's constriction factor
        """
        phi_total = c1 + c2
        chi = 2 * kappa / abs(2 - phi_total - math.sqrt(phi_total ** 2 - 4 * phi_total))

        for i in range(self.dimensions):
            r1, r2 = random.random(), random.random()
            cognitive = c1 * r1 * (self.best_position[i] - self.position[i])
            social = c2 * r2 * (global_best[i] - self.position[i])
            self.velocity[i] = chi * (self.velocity[i] + cognitive + social)

    def update_position(self) -> None:
        """更新位置: x = x + v"""
        for i in range(self.dimensions):
            self.position[i] += self.velocity[i]
            # 边界处理 - 反弹
            if self.position[i] < self.lower:
                self.position[i] = self.lower
                self.velocity[i] *= -0.5
            elif self.position[i] > self.upper:
                self.position[i] = self.upper
                self.velocity[i] *= -0.5

    def evaluate(self, fitness_func: FitnessFunction) -> float:
        """评估适应度"""
        self.fitness = fitness_func(self.position)
        if self.best_fitness is None or self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position.copy()
        return self.fitness

    def __repr__(self) -> str:
        return f"Particle(pos={self.position[:3]}..., fitness={self.fitness})"


# =============================================================================
# Base Optimizer Class
# =============================================================================

class BaseOptimizer(ABC):
    """优化器基类"""

    def __init__(self, config: SwarmConfig):
        self.config = config
        self.iteration = 0
        self.best_solution: Optional[List[float]] = None
        self.best_fitness: Optional[float] = None
        self.history: List[float] = []

    @abstractmethod
    def optimize(self, fitness_func: FitnessFunction) -> Tuple[List[float], float]:
        """执行优化，返回最优解和适应度"""
        pass

    def _update_best(self, solution: List[float], fitness: float) -> None:
        """更新全局最优"""
        if self.best_fitness is None:
            self.best_fitness = fitness
            self.best_solution = solution.copy()
        elif self.config.minimize and fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_solution = solution.copy()
        elif not self.config.minimize and fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_solution = solution.copy()


# =============================================================================
# Particle Swarm Optimization
# =============================================================================

class ParticleSwarmOptimization(BaseOptimizer):
    """
    粒子群优化算法 (PSO)

    支持以下变体:
    - 标准PSO (带惯性权重)
    - 自适应惯性权重
    - 收缩因子PSO
    - Fully Informed PSO (FIPS)
    """

    def __init__(self, config: SwarmConfig):
        super().__init__(config)
        self.swarm: List[Particle] = []
        self.global_best_position: Optional[List[float]] = None
        self.global_best_fitness: Optional[float] = None

    def _initialize_swarm(self) -> None:
        """初始化粒子群"""
        self.swarm = [
            Particle(self.config.dimensions, self.config.bounds)
            for _ in range(self.config.population_size)
        ]

        # 为FIPS设置邻居拓扑（环形拓扑）
        if self.config.use_fips:
            for i, particle in enumerate(self.swarm):
                # 每个粒子的邻居包括自己和左右各一个
                left = self.swarm[(i - 1) % len(self.swarm)]
                right = self.swarm[(i + 1) % len(self.swarm)]
                particle.neighbors = [particle, left, right]

    def optimize(self, fitness_func: FitnessFunction) -> Tuple[List[float], float]:
        """执行PSO优化"""
        self._initialize_swarm()

        # 初始评估
        for particle in self.swarm:
            fitness = particle.evaluate(fitness_func)
            self._update_global_best(particle, fitness)

        for iteration in range(self.config.max_iterations):
            self.iteration = iteration
            w = self._get_inertia_weight(iteration)

            for particle in self.swarm:
                # 速度更新
                if self.config.use_constriction:
                    particle.update_velocity_constriction(
                        self.config.c1, self.config.c2, self.global_best_position
                    )
                elif self.config.use_fips:
                    particle.update_velocity_fips(w)
                else:
                    particle.update_velocity(
                        w, self.config.c1, self.config.c2, self.global_best_position
                    )

                # 位置更新
                particle.update_position()

                # 评估
                fitness = particle.evaluate(fitness_func)
                self._update_global_best(particle, fitness)

            self.history.append(self.global_best_fitness)

        return self.global_best_position, self.global_best_fitness

    def _get_inertia_weight(self, iteration: int) -> float:
        """获取当前迭代的惯性权重"""
        if not self.config.w_decay:
            return self.config.w

        # 线性递减
        return self.config.w - (self.config.w - self.config.w_end) * iteration / self.config.max_iterations

    def _update_global_best(self, particle: Particle, fitness: float) -> None:
        """更新全局最优"""
        if self.global_best_fitness is None:
            self.global_best_fitness = fitness
            self.global_best_position = particle.position.copy()
        elif self.config.minimize and fitness < self.global_best_fitness:
            self.global_best_fitness = fitness
            self.global_best_position = particle.position.copy()
        elif not self.config.minimize and fitness > self.global_best_fitness:
            self.global_best_fitness = fitness
            self.global_best_position = particle.position.copy()


# =============================================================================
# Ant Colony Optimization
# =============================================================================

class AntColonyOptimization(BaseOptimizer):
    """
    蚁群优化算法 (ACO)

    主要用于解决组合优化问题如TSP
    支持:
    - 基本ACO
    - Max-Min Ant System (MMAS)
    - 精英蚂蚁策略
    """

    def __init__(self, config: SwarmConfig, distance_matrix: Optional[List[List[float]]] = None):
        super().__init__(config)
        self.distance_matrix = distance_matrix
        self.num_cities = config.dimensions
        self.pheromone: List[List[float]] = []
        self.heuristic: List[List[float]] = []
        self.best_tour: List[int] = []
        self.best_tour_length: float = float('inf')

    def _initialize_pheromone(self) -> None:
        """初始化信息素矩阵"""
        tau0 = self.config.tau_max if self.config.use_mmas else 1.0
        self.pheromone = [[tau0 for _ in range(self.num_cities)] for _ in range(self.num_cities)]

        # 初始化启发式信息 (距离的倒数)
        if self.distance_matrix:
            self.heuristic = []
            for i in range(self.num_cities):
                row = []
                for j in range(self.num_cities):
                    if i != j and self.distance_matrix[i][j] > 0:
                        row.append(1.0 / self.distance_matrix[i][j])
                    else:
                        row.append(0.0)
                self.heuristic.append(row)
        else:
            # 如果没有距离矩阵，使用均匀启发式
            self.heuristic = [[1.0 if i != j else 0.0 for j in range(self.num_cities)]
                             for i in range(self.num_cities)]

    def _construct_solution(self) -> Tuple[List[int], float]:
        """构建一个解（蚂蚁的路径）"""
        unvisited = list(range(self.num_cities))
        tour = []
        current = random.choice(unvisited)
        tour.append(current)
        unvisited.remove(current)
        total_distance = 0.0

        while unvisited:
            # 计算转移概率
            probabilities = []
            for city in unvisited:
                tau = self.pheromone[current][city] ** self.config.alpha
                eta = self.heuristic[current][city] ** self.config.beta
                probabilities.append(tau * eta)

            # 轮盘赌选择
            total = sum(probabilities)
            if total > 0:
                probabilities = [p / total for p in probabilities]
                next_city = random.choices(unvisited, weights=probabilities)[0]
            else:
                next_city = random.choice(unvisited)

            if self.distance_matrix:
                total_distance += self.distance_matrix[current][next_city]
            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city

        # 回到起点
        if self.distance_matrix and len(tour) > 1:
            total_distance += self.distance_matrix[tour[-1]][tour[0]]

        return tour, total_distance

    def _update_pheromone(self, all_tours: List[Tuple[List[int], float]]) -> None:
        """更新信息素"""
        # 蒸发
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                self.pheromone[i][j] *= (1 - self.config.rho)

        if self.config.use_mmas:
            # Max-Min Ant System: 只使用全局最优
            if self.best_tour:
                delta = self.config.Q / self.best_tour_length if self.best_tour_length > 0 else self.config.Q
                for i in range(len(self.best_tour)):
                    city1, city2 = self.best_tour[i], self.best_tour[(i + 1) % len(self.best_tour)]
                    self.pheromone[city1][city2] += delta
                    self.pheromone[city2][city1] += delta
        elif self.config.use_elitist:
            # 精英策略: 额外加强全局最优
            for tour, length in all_tours:
                delta = self.config.Q / length if length > 0 else self.config.Q
                for i in range(len(tour)):
                    city1, city2 = tour[i], tour[(i + 1) % len(tour)]
                    self.pheromone[city1][city2] += delta
                    self.pheromone[city2][city1] += delta

            # 额外加强全局最优
            if self.best_tour:
                delta = self.config.Q / self.best_tour_length if self.best_tour_length > 0 else self.config.Q
                for i in range(len(self.best_tour)):
                    city1, city2 = self.best_tour[i], self.best_tour[(i + 1) % len(self.best_tour)]
                    self.pheromone[city1][city2] += delta * self.config.population_size
                    self.pheromone[city2][city1] += delta * self.config.population_size
        else:
            # 标准ACO: 所有蚂蚁都贡献
            for tour, length in all_tours:
                delta = self.config.Q / length if length > 0 else self.config.Q
                for i in range(len(tour)):
                    city1, city2 = tour[i], tour[(i + 1) % len(tour)]
                    self.pheromone[city1][city2] += delta
                    self.pheromone[city2][city1] += delta

        # MMAS: 限制信息素范围
        if self.config.use_mmas:
            for i in range(self.num_cities):
                for j in range(self.num_cities):
                    self.pheromone[i][j] = clamp(
                        self.pheromone[i][j], self.config.tau_min, self.config.tau_max
                    )

    def optimize(self, fitness_func: Optional[FitnessFunction] = None) -> Tuple[List[int], float]:
        """
        执行ACO优化

        对于TSP问题，不需要fitness_func，使用内置的距离矩阵
        """
        self._initialize_pheromone()

        for iteration in range(self.config.max_iterations):
            self.iteration = iteration
            all_tours = []

            # 每只蚂蚁构建解
            for _ in range(self.config.population_size):
                tour, length = self._construct_solution()
                all_tours.append((tour, length))

                if length < self.best_tour_length:
                    self.best_tour_length = length
                    self.best_tour = tour.copy()

            # 更新信息素
            self._update_pheromone(all_tours)
            self.history.append(self.best_tour_length)

        return self.best_tour, self.best_tour_length


# =============================================================================
# Artificial Bee Colony
# =============================================================================

class FoodSource:
    """食物源类 - 用于ABC算法"""

    def __init__(self, dimensions: int, bounds: Tuple[float, float]):
        self.position = [random.uniform(bounds[0], bounds[1]) for _ in range(dimensions)]
        self.fitness: Optional[float] = None
        self.trial = 0  # 尝试次数


class ArtificialBeeColony(BaseOptimizer):
    """
    人工蜂群算法 (ABC)

    三种蜜蜂:
    - 雇佣蜂 (Employed bees): 在食物源附近搜索
    - 观察蜂 (Onlooker bees): 根据适应度选择食物源
    - 侦查蜂 (Scout bees): 当食物源被放弃时随机搜索
    """

    def __init__(self, config: SwarmConfig):
        super().__init__(config)
        self.food_sources: List[FoodSource] = []
        self.probabilities: List[float] = []

    def _initialize_food_sources(self) -> None:
        """初始化食物源"""
        num_sources = self.config.population_size // 2
        self.food_sources = [
            FoodSource(self.config.dimensions, self.config.bounds)
            for _ in range(num_sources)
        ]

    def _employed_bee_phase(self, fitness_func: FitnessFunction) -> None:
        """雇佣蜂阶段 - 在食物源附近搜索"""
        for i, food in enumerate(self.food_sources):
            new_position = self._generate_neighbor(food.position, i)
            new_fitness = fitness_func(new_position)

            # 贪婪选择
            if self._is_better(new_fitness, food.fitness):
                food.position = new_position
                food.fitness = new_fitness
                food.trial = 0
            else:
                food.trial += 1

    def _onlooker_bee_phase(self, fitness_func: FitnessFunction) -> None:
        """观察蜂阶段 - 根据适应度概率选择食物源"""
        # 计算选择概率
        fitnesses = [f.fitness for f in self.food_sources if f.fitness is not None]
        if not fitnesses:
            return

        # 使用适应度值计算概率（最小化问题需要转换）
        if self.config.minimize:
            # 对于最小化，使用倒数或减去最大值
            max_fit = max(fitnesses)
            adjusted = [max_fit - f + 1e-10 for f in fitnesses]
        else:
            adjusted = fitnesses

        total = sum(adjusted)
        if total > 0:
            self.probabilities = [f / total for f in adjusted]
        else:
            self.probabilities = [1.0 / len(self.food_sources)] * len(self.food_sources)

        # 观察蜂选择
        num_onlookers = len(self.food_sources)
        for _ in range(num_onlookers):
            # 轮盘赌选择食物源
            selected_idx = random.choices(range(len(self.food_sources)), weights=self.probabilities)[0]
            food = self.food_sources[selected_idx]

            new_position = self._generate_neighbor(food.position, selected_idx)
            new_fitness = fitness_func(new_position)

            if self._is_better(new_fitness, food.fitness):
                food.position = new_position
                food.fitness = new_fitness
                food.trial = 0
            else:
                food.trial += 1

    def _scout_bee_phase(self, fitness_func: FitnessFunction) -> None:
        """侦查蜂阶段 - 放弃超过限制的食物源"""
        for food in self.food_sources:
            if food.trial >= self.config.limit:
                # 放弃并随机生成新食物源
                food.position = [random.uniform(self.config.bounds[0], self.config.bounds[1])
                               for _ in range(self.config.dimensions)]
                food.fitness = fitness_func(food.position)
                food.trial = 0

    def _generate_neighbor(self, position: List[float], exclude_idx: int) -> List[float]:
        """生成邻域解"""
        new_pos = position.copy()
        dim = random.randint(0, self.config.dimensions - 1)

        # 随机选择另一个食物源
        partner_idx = random.randint(0, len(self.food_sources) - 1)
        while partner_idx == exclude_idx:
            partner_idx = random.randint(0, len(self.food_sources) - 1)

        partner_pos = self.food_sources[partner_idx].position
        phi = random.uniform(-1, 1)

        new_pos[dim] = position[dim] + phi * (position[dim] - partner_pos[dim])
        new_pos[dim] = clamp(new_pos[dim], self.config.bounds[0], self.config.bounds[1])

        return new_pos

    def _is_better(self, new_fitness: float, old_fitness: Optional[float]) -> bool:
        """判断新适应度是否更好"""
        if old_fitness is None:
            return True
        if self.config.minimize:
            return new_fitness < old_fitness
        return new_fitness > old_fitness

    def optimize(self, fitness_func: FitnessFunction) -> Tuple[List[float], float]:
        """执行ABC优化"""
        self._initialize_food_sources()

        # 初始评估
        for food in self.food_sources:
            food.fitness = fitness_func(food.position)
            self._update_best(food.position, food.fitness)

        for iteration in range(self.config.max_iterations):
            self.iteration = iteration

            self._employed_bee_phase(fitness_func)
            self._onlooker_bee_phase(fitness_func)
            self._scout_bee_phase(fitness_func)

            # 更新全局最优
            for food in self.food_sources:
                self._update_best(food.position, food.fitness)

            self.history.append(self.best_fitness)

        return self.best_solution, self.best_fitness


# =============================================================================
# Firefly Algorithm
# =============================================================================

class Firefly:
    """萤火虫类"""

    def __init__(self, dimensions: int, bounds: Tuple[float, float]):
        self.position = [random.uniform(bounds[0], bounds[1]) for _ in range(dimensions)]
        self.fitness: Optional[float] = None
        self.intensity: float = 0.0  # 光强度


class FireflyAlgorithm(BaseOptimizer):
    """
    萤火虫算法 (Firefly Algorithm)

    核心概念:
    - 吸引力与光强度随距离指数衰减
    - 较亮的萤火虫吸引较暗的萤火虫
    - 随机项提供探索能力
    """

    def __init__(self, config: SwarmConfig):
        super().__init__(config)
        self.fireflies: List[Firefly] = []

    def _initialize_fireflies(self) -> None:
        """初始化萤火虫群"""
        self.fireflies = [
            Firefly(self.config.dimensions, self.config.bounds)
            for _ in range(self.config.population_size)
        ]

    def _calculate_intensity(self, fitness: float) -> float:
        """计算光强度"""
        # 对于最小化问题，适应度越小光越强
        if self.config.minimize:
            return 1.0 / (1 + fitness) if fitness >= 0 else 1.0 / (1 - fitness)
        else:
            return fitness if fitness > 0 else 0.0

    def _calculate_attractiveness(self, distance: float) -> float:
        """
        计算吸引力
        β = β0 * exp(-γ * r^2)
        """
        return self.config.beta0 * math.exp(-self.config.gamma * distance ** 2)

    def _move_firefly(self, firefly: Firefly, brighter_firefly: Firefly) -> None:
        """
        移动萤火虫向更亮的萤火虫
        xi = xi + β*(xj-xi) + α*(rand-0.5)
        """
        distance = euclidean_distance(firefly.position, brighter_firefly.position)
        beta = self._calculate_attractiveness(distance)

        for i in range(self.config.dimensions):
            r = random.random()
            movement = (beta * (brighter_firefly.position[i] - firefly.position[i]) +
                       self.config.alpha_firefly * (r - 0.5))
            firefly.position[i] += movement
            firefly.position[i] = clamp(firefly.position[i], self.config.bounds[0], self.config.bounds[1])

    def _random_move(self, firefly: Firefly) -> None:
        """随机移动（用于最亮的萤火虫）"""
        for i in range(self.config.dimensions):
            firefly.position[i] += self.config.alpha_firefly * (random.random() - 0.5)
            firefly.position[i] = clamp(firefly.position[i], self.config.bounds[0], self.config.bounds[1])

    def optimize(self, fitness_func: FitnessFunction) -> Tuple[List[float], float]:
        """执行萤火虫算法优化"""
        self._initialize_fireflies()

        # 初始评估
        for firefly in self.fireflies:
            firefly.fitness = fitness_func(firefly.position)
            firefly.intensity = self._calculate_intensity(firefly.fitness)
            self._update_best(firefly.position, firefly.fitness)

        for iteration in range(self.config.max_iterations):
            self.iteration = iteration

            # 按光强度排序（降序）
            self.fireflies.sort(key=lambda f: f.intensity, reverse=True)

            # 移动萤火虫
            for i in range(len(self.fireflies)):
                moved = False
                for j in range(i):
                    if self.fireflies[j].intensity > self.fireflies[i].intensity:
                        self._move_firefly(self.fireflies[i], self.fireflies[j])
                        moved = True
                        break

                if not moved:
                    # 最亮的萤火虫随机移动
                    self._random_move(self.fireflies[i])

            # 重新评估
            for firefly in self.fireflies:
                firefly.fitness = fitness_func(firefly.position)
                firefly.intensity = self._calculate_intensity(firefly.fitness)
                self._update_best(firefly.position, firefly.fitness)

            self.history.append(self.best_fitness)

        return self.best_solution, self.best_fitness


# =============================================================================
# Cuckoo Search
# =============================================================================

class Nest:
    """巢穴类 - 用于布谷鸟搜索"""

    def __init__(self, dimensions: int, bounds: Tuple[float, float]):
        self.position = [random.uniform(bounds[0], bounds[1]) for _ in range(dimensions)]
        self.fitness: Optional[float] = None


class CuckooSearch(BaseOptimizer):
    """
    布谷鸟搜索算法 (Cuckoo Search)

    核心概念:
    - Lévy飞行进行全局搜索
    - 以概率pa放弃巢穴并随机重建
    - 布谷鸟蛋代表解
    """

    def __init__(self, config: SwarmConfig):
        super().__init__(config)
        self.nests: List[Nest] = []

    def _initialize_nests(self) -> None:
        """初始化巢穴"""
        self.nests = [
            Nest(self.config.dimensions, self.config.bounds)
            for _ in range(self.config.population_size)
        ]

    def _levy_flight_step(self) -> List[float]:
        """生成Lévy飞行步长"""
        return levy_flight(self.config.beta_cuckoo, self.config.dimensions)

    def _generate_cuckoo(self, nest: Nest) -> Nest:
        """通过Lévy飞行生成新的布谷鸟蛋"""
        new_nest = Nest(self.config.dimensions, self.config.bounds)
        step = self._levy_flight_step()

        for i in range(self.config.dimensions):
            new_nest.position[i] = nest.position[i] + 0.01 * step[i] * (nest.position[i] - self.best_solution[i] if self.best_solution else 0)
            new_nest.position[i] = clamp(new_nest.position[i], self.config.bounds[0], self.config.bounds[1])

        return new_nest

    def _abandon_nests(self) -> None:
        """以概率pa放弃巢穴"""
        num_abandon = int(self.config.pa * len(self.nests))

        # 按适应度排序，放弃较差的巢穴
        sorted_nests = sorted(self.nests, key=lambda n: n.fitness if n.fitness is not None else float('inf'))

        for i in range(num_abandon):
            # 随机游走生成新巢穴
            idx = self.nests.index(sorted_nests[-(i+1)])
            for j in range(self.config.dimensions):
                self.nests[idx].position[j] = random.uniform(self.config.bounds[0], self.config.bounds[1])

    def optimize(self, fitness_func: FitnessFunction) -> Tuple[List[float], float]:
        """执行布谷鸟搜索优化"""
        self._initialize_nests()

        # 初始评估
        for nest in self.nests:
            nest.fitness = fitness_func(nest.position)
            self._update_best(nest.position, nest.fitness)

        for iteration in range(self.config.max_iterations):
            self.iteration = iteration

            # 对每个巢穴生成布谷鸟蛋
            for i, nest in enumerate(self.nests):
                cuckoo = self._generate_cuckoo(nest)
                cuckoo.fitness = fitness_func(cuckoo.position)

                # 随机选择一个巢穴进行比较
                j = random.randint(0, len(self.nests) - 1)

                if self.config.minimize:
                    is_better = cuckoo.fitness < self.nests[j].fitness
                else:
                    is_better = cuckoo.fitness > self.nests[j].fitness

                if is_better:
                    self.nests[j].position = cuckoo.position
                    self.nests[j].fitness = cuckoo.fitness
                    self._update_best(self.nests[j].position, self.nests[j].fitness)

            # 放弃巢穴
            self._abandon_nests()

            # 重新评估被放弃的巢穴
            for nest in self.nests:
                if nest.fitness is None:
                    nest.fitness = fitness_func(nest.position)
                    self._update_best(nest.position, nest.fitness)

            self.history.append(self.best_fitness)

        return self.best_solution, self.best_fitness


# =============================================================================
# Bat Algorithm
# =============================================================================

class Bat:
    """蝙蝠类"""

    def __init__(self, dimensions: int, bounds: Tuple[float, float],
                 f_min: float, f_max: float, A_init: float, r_init: float):
        self.position = [random.uniform(bounds[0], bounds[1]) for _ in range(dimensions)]
        self.velocity = [0.0] * dimensions
        self.frequency = f_min
        self.fitness: Optional[float] = None

        # 响度和脉冲率
        self.A = A_init  # 响度
        self.r = r_init  # 脉冲率
        self.r0 = r_init  # 初始脉冲率

        # 个人最佳
        self.best_position = None
        self.best_fitness = None


class BatAlgorithm(BaseOptimizer):
    """
    蝙蝠算法 (Bat Algorithm)

    核心概念:
    - 频率调节控制移动步长
    - 响度(A)和脉冲率(r)的动态调整
    - 响度逐渐减小，脉冲率逐渐增加
    """

    def __init__(self, config: SwarmConfig):
        super().__init__(config)
        self.bats: List[Bat] = []

    def _initialize_bats(self) -> None:
        """初始化蝙蝠群"""
        self.bats = [
            Bat(self.config.dimensions, self.config.bounds,
                self.config.f_min, self.config.f_max,
                self.config.A_init, self.config.r_init)
            for _ in range(self.config.population_size)
        ]

    def _update_bat(self, bat: Bat, global_best: List[float], iteration: int) -> None:
        """
        更新蝙蝠位置和速度
        fi = fmin + (fmax-fmin)*β
        vi = vi + (xi - xbest)*fi
        xi = xi + vi
        """
        # 更新频率
        beta = random.random()
        bat.frequency = self.config.f_min + (self.config.f_max - self.config.f_min) * beta

        # 更新速度
        for i in range(self.config.dimensions):
            bat.velocity[i] += (bat.position[i] - global_best[i]) * bat.frequency

        # 更新位置
        for i in range(self.config.dimensions):
            bat.position[i] += bat.velocity[i]
            bat.position[i] = clamp(bat.position[i], self.config.bounds[0], self.config.bounds[1])

    def _local_search(self, bat: Bat, global_best: List[float]) -> List[float]:
        """局部搜索 - 在当前最佳解附近随机搜索"""
        new_position = bat.position.copy()
        for i in range(self.config.dimensions):
            # 在当前最佳解附近随机游走
            new_position[i] = global_best[i] + random.uniform(-1, 1) * bat.A
            new_position[i] = clamp(new_position[i], self.config.bounds[0], self.config.bounds[1])
        return new_position

    def optimize(self, fitness_func: FitnessFunction) -> Tuple[List[float], float]:
        """执行蝙蝠算法优化"""
        self._initialize_bats()

        # 初始评估
        for bat in self.bats:
            bat.fitness = fitness_func(bat.position)
            bat.best_position = bat.position.copy()
            bat.best_fitness = bat.fitness
            self._update_best(bat.position, bat.fitness)

        for iteration in range(self.config.max_iterations):
            self.iteration = iteration
            avg_A = sum(bat.A for bat in self.bats) / len(self.bats)

            for bat in self.bats:
                self._update_bat(bat, self.best_solution, iteration)

                # 以概率(1-r)进行局部搜索
                if random.random() > bat.r:
                    new_position = self._local_search(bat, self.best_solution)
                else:
                    new_position = bat.position

                new_fitness = fitness_func(new_position)

                # 以概率A接受新解
                if random.random() < bat.A:
                    if self.config.minimize and new_fitness < bat.fitness:
                        accept = True
                    elif not self.config.minimize and new_fitness > bat.fitness:
                        accept = True
                    else:
                        accept = False

                    if accept:
                        bat.position = new_position
                        bat.fitness = new_fitness
                        self._update_best(bat.position, bat.fitness)

                        # 增加脉冲率，减小响度
                        bat.r = bat.r0 * (1 - math.exp(-self.config.gamma_bat * iteration))
                        bat.A *= self.config.alpha_bat

            self.history.append(self.best_fitness)

        return self.best_solution, self.best_fitness


# =============================================================================
# Grey Wolf Optimizer
# =============================================================================

class Wolf:
    """灰狼类"""

    def __init__(self, dimensions: int, bounds: Tuple[float, float]):
        self.position = [random.uniform(bounds[0], bounds[1]) for _ in range(dimensions)]
        self.fitness: Optional[float] = None


class GreyWolfOptimizer(BaseOptimizer):
    """
    灰狼优化算法 (Grey Wolf Optimizer, GWO)

    核心概念:
    - 模拟灰狼的社会等级: Alpha(α), Beta(β), Delta(δ)
    - 包围猎物: D = |C*X_prey - X|
    - 狩猎: X = X_prey - A*D
    - A和C向量随迭代线性递减
    """

    def __init__(self, config: SwarmConfig):
        super().__init__(config)
        self.wolves: List[Wolf] = []
        self.alpha: Optional[Wolf] = None
        self.beta: Optional[Wolf] = None
        self.delta: Optional[Wolf] = None

    def _initialize_wolves(self) -> None:
        """初始化狼群"""
        self.wolves = [
            Wolf(self.config.dimensions, self.config.bounds)
            for _ in range(self.config.population_size)
        ]

    def _update_hierarchy(self) -> None:
        """更新Alpha, Beta, Delta狼"""
        sorted_wolves = sorted(self.wolves, key=lambda w: w.fitness if w.fitness is not None else float('inf'))
        self.alpha = sorted_wolves[0]
        self.beta = sorted_wolves[1] if len(sorted_wolves) > 1 else self.alpha
        self.delta = sorted_wolves[2] if len(sorted_wolves) > 2 else self.beta

    def _get_a_coefficient(self, iteration: int) -> float:
        """
        计算A向量系数
        a从2线性递减到0
        """
        return 2 - 2 * iteration / self.config.max_iterations

    def _update_position(self, wolf: Wolf, a: float) -> None:
        """
        更新灰狼位置
        根据Alpha, Beta, Delta的位置进行更新
        """
        new_position = [0.0] * self.config.dimensions

        for i in range(self.config.dimensions):
            # 计算与三个头狼的距离
            r1, r2 = random.random(), random.random()
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = abs(C1 * self.alpha.position[i] - wolf.position[i])
            X1 = self.alpha.position[i] - A1 * D_alpha

            r1, r2 = random.random(), random.random()
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = abs(C2 * self.beta.position[i] - wolf.position[i])
            X2 = self.beta.position[i] - A2 * D_beta

            r1, r2 = random.random(), random.random()
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = abs(C3 * self.delta.position[i] - wolf.position[i])
            X3 = self.delta.position[i] - A3 * D_delta

            # 平均三个方向
            new_position[i] = (X1 + X2 + X3) / 3.0
            new_position[i] = clamp(new_position[i], self.config.bounds[0], self.config.bounds[1])

        wolf.position = new_position

    def optimize(self, fitness_func: FitnessFunction) -> Tuple[List[float], float]:
        """执行GWO优化"""
        self._initialize_wolves()

        # 初始评估
        for wolf in self.wolves:
            wolf.fitness = fitness_func(wolf.position)
            self._update_best(wolf.position, wolf.fitness)

        self._update_hierarchy()

        for iteration in range(self.config.max_iterations):
            self.iteration = iteration
            a = self._get_a_coefficient(iteration)

            for wolf in self.wolves:
                self._update_position(wolf, a)
                wolf.fitness = fitness_func(wolf.position)
                self._update_best(wolf.position, wolf.fitness)

            self._update_hierarchy()
            self.history.append(self.best_fitness)

        return self.best_solution, self.best_fitness


# =============================================================================
# Whale Optimization Algorithm
# =============================================================================

class Whale:
    """鲸鱼类"""

    def __init__(self, dimensions: int, bounds: Tuple[float, float]):
        self.position = [random.uniform(bounds[0], bounds[1]) for _ in range(dimensions)]
        self.fitness: Optional[float] = None


class WhaleOptimizationAlgorithm(BaseOptimizer):
    """
    鲸鱼优化算法 (Whale Optimization Algorithm, WOA)

    核心概念:
    - 收缩包围机制 (Shrinking encircling)
    - 气泡网攻击 (Bubble-net attacking) - 螺旋运动
    - 搜索猎物 (Search for prey) - 随机探索
    """

    def __init__(self, config: SwarmConfig):
        super().__init__(config)
        self.whales: List[Whale] = []

    def _initialize_whales(self) -> None:
        """初始化鲸群"""
        self.whales = [
            Whale(self.config.dimensions, self.config.bounds)
            for _ in range(self.config.population_size)
        ]

    def _get_a_and_r(self, iteration: int) -> Tuple[float, List[float]]:
        """
        计算a系数和随机向量r
        a从2线性递减到0
        """
        a = 2 - 2 * iteration / self.config.max_iterations
        r = [random.random() for _ in range(self.config.dimensions)]
        return a, r

    def _encircling_prey(self, whale: Whale, best_position: List[float], A: List[float], C: List[float]) -> List[float]:
        """
        收缩包围猎物
        D = |C*X_best - X|
        X_new = X_best - A*D
        """
        new_position = []
        for i in range(self.config.dimensions):
            D = abs(C[i] * best_position[i] - whale.position[i])
            new_pos = best_position[i] - A[i] * D
            new_position.append(clamp(new_pos, self.config.bounds[0], self.config.bounds[1]))
        return new_position

    def _bubble_net_attacking(self, whale: Whale, best_position: List[float]) -> List[float]:
        """
        气泡网攻击 - 螺旋运动
        X_new = D' * exp(b*l) * cos(2*pi*l) + X_best
        其中 D' = |X_best - X|
        """
        b = 1.0  # 螺旋常数
        l = random.uniform(-1, 1)

        new_position = []
        for i in range(self.config.dimensions):
            D_prime = abs(best_position[i] - whale.position[i])
            new_pos = D_prime * math.exp(b * l) * math.cos(2 * math.pi * l) + best_position[i]
            new_position.append(clamp(new_pos, self.config.bounds[0], self.config.bounds[1]))
        return new_position

    def _search_prey(self, whale: Whale, random_whale: Whale, A: List[float], C: List[float]) -> List[float]:
        """
        搜索猎物 - 随机探索
        D = |C*X_rand - X|
        X_new = X_rand - A*D
        """
        new_position = []
        for i in range(self.config.dimensions):
            D = abs(C[i] * random_whale.position[i] - whale.position[i])
            new_pos = random_whale.position[i] - A[i] * D
            new_position.append(clamp(new_pos, self.config.bounds[0], self.config.bounds[1]))
        return new_position

    def optimize(self, fitness_func: FitnessFunction) -> Tuple[List[float], float]:
        """执行WOA优化"""
        self._initialize_whales()

        # 初始评估
        for whale in self.whales:
            whale.fitness = fitness_func(whale.position)
            self._update_best(whale.position, whale.fitness)

        for iteration in range(self.config.max_iterations):
            self.iteration = iteration
            a, r = self._get_a_and_r(iteration)

            for whale in self.whales:
                p = random.random()
                A = [2 * a * random.random() - a for _ in range(self.config.dimensions)]
                C = [2 * random.random() for _ in range(self.config.dimensions)]

                if p < 0.5:
                    # 收缩包围或搜索猎物
                    if all(abs(ai) < 1 for ai in A):
                        # 收缩包围
                        new_position = self._encircling_prey(whale, self.best_solution, A, C)
                    else:
                        # 搜索猎物
                        random_whale = random.choice(self.whales)
                        new_position = self._search_prey(whale, random_whale, A, C)
                else:
                    # 气泡网攻击
                    new_position = self._bubble_net_attacking(whale, self.best_solution)

                whale.position = new_position
                whale.fitness = fitness_func(whale.position)
                self._update_best(whale.position, whale.fitness)

            self.history.append(self.best_fitness)

        return self.best_solution, self.best_fitness


# =============================================================================
# Differential Evolution
# =============================================================================

class Individual:
    """个体类 - 用于DE"""

    def __init__(self, dimensions: int, bounds: Tuple[float, float]):
        self.position = [random.uniform(bounds[0], bounds[1]) for _ in range(dimensions)]
        self.fitness: Optional[float] = None


class DifferentialEvolution(BaseOptimizer):
    """
    差分进化算法 (Differential Evolution, DE)

    支持策略:
    - DE/rand/1: 随机选择基向量
    - DE/best/1: 使用最优个体作为基向量
    - DE/current-to-best/1: 当前个体向最优个体移动

    交叉方式:
    - binomial: 二项式交叉
    - exponential: 指数交叉
    """

    def __init__(self, config: SwarmConfig):
        super().__init__(config)
        self.population: List[Individual] = []

    def _initialize_population(self) -> None:
        """初始化种群"""
        self.population = [
            Individual(self.config.dimensions, self.config.bounds)
            for _ in range(self.config.population_size)
        ]

    def _mutate(self, target_idx: int) -> List[float]:
        """
        变异操作
        根据策略选择不同的变异方式
        """
        strategy = self.config.de_strategy

        if strategy.startswith("rand/1"):
            # DE/rand/1: v = xr1 + F * (xr2 - xr3)
            indices = list(range(len(self.population)))
            indices.remove(target_idx)
            r1, r2, r3 = random.sample(indices, 3)

            mutant = []
            for i in range(self.config.dimensions):
                v = (self.population[r1].position[i] +
                     self.config.F * (self.population[r2].position[i] - self.population[r3].position[i]))
                mutant.append(v)

        elif strategy.startswith("best/1"):
            # DE/best/1: v = xbest + F * (xr1 - xr2)
            best_idx = min(range(len(self.population)),
                          key=lambda i: self.population[i].fitness if self.population[i].fitness is not None else float('inf'))
            indices = list(range(len(self.population)))
            indices.remove(target_idx)
            indices.remove(best_idx)
            r1, r2 = random.sample(indices, 2)

            mutant = []
            for i in range(self.config.dimensions):
                v = (self.population[best_idx].position[i] +
                     self.config.F * (self.population[r1].position[i] - self.population[r2].position[i]))
                mutant.append(v)

        elif strategy.startswith("current-to-best/1"):
            # DE/current-to-best/1: v = xi + F * (xbest - xi) + F * (xr1 - xr2)
            best_idx = min(range(len(self.population)),
                          key=lambda i: self.population[i].fitness if self.population[i].fitness is not None else float('inf'))
            indices = list(range(len(self.population)))
            indices.remove(target_idx)
            indices.remove(best_idx)
            r1, r2 = random.sample(indices, 2)

            mutant = []
            for i in range(self.config.dimensions):
                v = (self.population[target_idx].position[i] +
                     self.config.F * (self.population[best_idx].position[i] - self.population[target_idx].position[i]) +
                     self.config.F * (self.population[r1].position[i] - self.population[r2].position[i]))
                mutant.append(v)
        else:
            # 默认使用rand/1
            return self._mutate_rand1(target_idx)

        return mutant

    def _mutate_rand1(self, target_idx: int) -> List[float]:
        """DE/rand/1变异"""
        indices = list(range(len(self.population)))
        indices.remove(target_idx)
        r1, r2, r3 = random.sample(indices, 3)

        mutant = []
        for i in range(self.config.dimensions):
            v = (self.population[r1].position[i] +
                 self.config.F * (self.population[r2].position[i] - self.population[r3].position[i]))
            mutant.append(v)
        return mutant

    def _crossover(self, target: List[float], mutant: List[float]) -> List[float]:
        """
        交叉操作
        """
        if "bin" in self.config.de_strategy:
            return self._binomial_crossover(target, mutant)
        else:
            return self._exponential_crossover(target, mutant)

    def _binomial_crossover(self, target: List[float], mutant: List[float]) -> List[float]:
        """二项式交叉"""
        trial = []
        j_rand = random.randint(0, self.config.dimensions - 1)

        for j in range(self.config.dimensions):
            if random.random() < self.config.CR or j == j_rand:
                trial.append(mutant[j])
            else:
                trial.append(target[j])

        # 边界处理
        for j in range(self.config.dimensions):
            trial[j] = clamp(trial[j], self.config.bounds[0], self.config.bounds[1])

        return trial

    def _exponential_crossover(self, target: List[float], mutant: List[float]) -> List[float]:
        """指数交叉"""
        trial = target.copy()
        n = random.randint(0, self.config.dimensions - 1)
        L = 0

        while random.random() < self.config.CR and L < self.config.dimensions:
            trial[(n + L) % self.config.dimensions] = mutant[(n + L) % self.config.dimensions]
            L += 1

        # 边界处理
        for j in range(self.config.dimensions):
            trial[j] = clamp(trial[j], self.config.bounds[0], self.config.bounds[1])

        return trial

    def optimize(self, fitness_func: FitnessFunction) -> Tuple[List[float], float]:
        """执行DE优化"""
        self._initialize_population()

        # 初始评估
        for individual in self.population:
            individual.fitness = fitness_func(individual.position)
            self._update_best(individual.position, individual.fitness)

        for iteration in range(self.config.max_iterations):
            self.iteration = iteration

            for i, individual in enumerate(self.population):
                # 变异
                mutant = self._mutate(i)

                # 交叉
                trial_vector = self._crossover(individual.position, mutant)

                # 选择（贪婪策略）
                trial_fitness = fitness_func(trial_vector)

                if self.config.minimize:
                    is_better = trial_fitness < individual.fitness
                else:
                    is_better = trial_fitness > individual.fitness

                if is_better:
                    individual.position = trial_vector
                    individual.fitness = trial_fitness
                    self._update_best(individual.position, individual.fitness)

            self.history.append(self.best_fitness)

        return self.best_solution, self.best_fitness


# =============================================================================
# Evolution Strategy
# =============================================================================

class ESIndividual:
    """ES个体类 - 包含策略参数"""

    def __init__(self, dimensions: int, bounds: Tuple[float, float]):
        self.position = [random.uniform(bounds[0], bounds[1]) for _ in range(dimensions)]
        self.fitness: Optional[float] = None
        # 自适应变异步长
        self.sigmas = [random.uniform(0.1, 1.0) for _ in range(dimensions)]


class EvolutionStrategy(BaseOptimizer):
    """
    进化策略 (Evolution Strategy, ES)

    支持:
    - (μ+λ) 选择: 父母和后代一起竞争
    - (μ,λ) 选择: 只从后代中选择
    - 自适应变异
    - CMA-ES (Covariance Matrix Adaptation)
    """

    def __init__(self, config: SwarmConfig):
        super().__init__(config)
        self.parents: List[ESIndividual] = []
        self.offspring: List[ESIndividual] = []
        self.tau = 1.0 / math.sqrt(2 * config.dimensions)  # 全局学习率
        self.tau_prime = 1.0 / math.sqrt(2 * math.sqrt(config.dimensions))  # 局部学习率

        # CMA-ES相关
        self.cma_mean: Optional[List[float]] = None
        self.cma_cov: Optional[List[List[float]]] = None
        self.cma_sigma: float = 1.0

    def _initialize_parents(self) -> None:
        """初始化父代"""
        self.parents = [
            ESIndividual(self.config.dimensions, self.config.bounds)
            for _ in range(self.config.mu)
        ]

        if self.config.use_cma:
            self.cma_mean = [random.uniform(self.config.bounds[0], self.config.bounds[1])
                            for _ in range(self.config.dimensions)]
            self.cma_cov = [[1.0 if i == j else 0.0 for j in range(self.config.dimensions)]
                           for i in range(self.config.dimensions)]

    def _mutate_self_adaptive(self, parent: ESIndividual) -> ESIndividual:
        """自适应变异"""
        offspring = ESIndividual(self.config.dimensions, self.config.bounds)

        # 更新变异步长
        tau_global = random.gauss(0, 1)
        for i in range(self.config.dimensions):
            tau_local = random.gauss(0, 1)
            offspring.sigmas[i] = parent.sigmas[i] * math.exp(
                self.tau_prime * tau_global + self.tau * tau_local
            )
            offspring.sigmas[i] = max(offspring.sigmas[i], 1e-10)  # 防止过小

        # 变异位置
        for i in range(self.config.dimensions):
            offspring.position[i] = parent.position[i] + offspring.sigmas[i] * random.gauss(0, 1)
            offspring.position[i] = clamp(offspring.position[i], self.config.bounds[0], self.config.bounds[1])

        return offspring

    def _recombine(self, parent1: ESIndividual, parent2: ESIndividual) -> ESIndividual:
        """重组（交叉）"""
        offspring = ESIndividual(self.config.dimensions, self.config.bounds)

        # 离散重组位置
        for i in range(self.config.dimensions):
            if random.random() < 0.5:
                offspring.position[i] = parent1.position[i]
            else:
                offspring.position[i] = parent2.position[i]

        # 中间重组步长
        for i in range(self.config.dimensions):
            offspring.sigmas[i] = (parent1.sigmas[i] + parent2.sigmas[i]) / 2.0

        return offspring

    def _generate_offspring(self) -> None:
        """生成子代"""
        self.offspring = []

        for _ in range(self.config.lambda_es):
            # 随机选择两个父代进行重组
            parent1, parent2 = random.sample(self.parents, 2)
            offspring = self._recombine(parent1, parent2)

            # 变异
            mutated = self._mutate_self_adaptive(offspring)
            self.offspring.append(mutated)

    def _select_plus(self) -> None:
        """ (μ+λ) 选择 - 父母和后代一起竞争 """
        all_individuals = self.parents + self.offspring
        all_individuals.sort(key=lambda x: x.fitness if x.fitness is not None else float('inf'))
        self.parents = all_individuals[:self.config.mu]

    def _select_comma(self) -> None:
        """ (μ,λ) 选择 - 只从后代中选择 """
        self.offspring.sort(key=lambda x: x.fitness if x.fitness is not None else float('inf'))
        self.parents = self.offspring[:self.config.mu]

    def optimize(self, fitness_func: FitnessFunction) -> Tuple[List[float], float]:
        """执行ES优化"""
        self._initialize_parents()

        # 初始评估
        for parent in self.parents:
            parent.fitness = fitness_func(parent.position)
            self._update_best(parent.position, parent.fitness)

        for iteration in range(self.config.max_iterations):
            self.iteration = iteration

            # 生成子代
            self._generate_offspring()

            # 评估子代
            for offspring in self.offspring:
                offspring.fitness = fitness_func(offspring.position)
                self._update_best(offspring.position, offspring.fitness)

            # 选择
            if self.config.selection_type == "plus":
                self._select_plus()
            else:
                self._select_comma()

            self.history.append(self.best_fitness)

        return self.best_solution, self.best_fitness


# =============================================================================
# Genetic Algorithm
# =============================================================================

class GAIndividual:
    """GA个体类"""

    def __init__(self, dimensions: int, bounds: Tuple[float, float]):
        self.position = [random.uniform(bounds[0], bounds[1]) for _ in range(dimensions)]
        self.fitness: Optional[float] = None
        self.shared_fitness: Optional[float] = None  # 共享适应度（用于niching）


class GeneticAlgorithm(BaseOptimizer):
    """
    遗传算法 (Genetic Algorithm, GA)

    支持:
    - 选择: 锦标赛选择、轮盘赌选择、排序选择
    - 交叉: 单点、两点、均匀、算术交叉
    - 变异: 位翻转变异、高斯变异、多项式变异
    - 精英保留
    - 共享函数（niching）
    """

    def __init__(self, config: SwarmConfig):
        super().__init__(config)
        self.population: List[GAIndividual] = []

    def _initialize_population(self) -> None:
        """初始化种群"""
        self.population = [
            GAIndividual(self.config.dimensions, self.config.bounds)
            for _ in range(self.config.population_size)
        ]

    def _tournament_selection(self) -> GAIndividual:
        """锦标赛选择"""
        tournament = random.sample(self.population, self.config.tournament_size)
        if self.config.minimize:
            return min(tournament, key=lambda x: x.fitness if x.fitness is not None else float('inf'))
        else:
            return max(tournament, key=lambda x: x.fitness if x.fitness is not None else float('-inf'))

    def _roulette_selection(self) -> GAIndividual:
        """轮盘赌选择"""
        fitnesses = [ind.fitness for ind in self.population if ind.fitness is not None]
        if not fitnesses:
            return random.choice(self.population)

        if self.config.minimize:
            # 最小化问题需要转换
            max_fit = max(fitnesses)
            adjusted = [max_fit - f + 1e-10 for f in fitnesses]
        else:
            adjusted = fitnesses

        total = sum(adjusted)
        if total > 0:
            probabilities = [f / total for f in adjusted]
            return random.choices(self.population, weights=probabilities)[0]
        else:
            return random.choice(self.population)

    def _rank_selection(self) -> GAIndividual:
        """排序选择"""
        sorted_pop = sorted(self.population,
                           key=lambda x: x.fitness if x.fitness is not None else float('inf'),
                           reverse=not self.config.minimize)

        # 线性排名概率
        n = len(sorted_pop)
        probabilities = [(2 - 1.0/n) + 2*(i-1)*(1.0/n - 1)/(n-1) for i in range(1, n+1)]
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]

        return random.choices(sorted_pop, weights=probabilities)[0]

    def _select_parent(self) -> GAIndividual:
        """选择父代"""
        if self.config.selection_method == "tournament":
            return self._tournament_selection()
        elif self.config.selection_method == "roulette":
            return self._roulette_selection()
        elif self.config.selection_method == "rank":
            return self._rank_selection()
        else:
            return self._tournament_selection()

    def _crossover(self, parent1: GAIndividual, parent2: GAIndividual) -> Tuple[GAIndividual, GAIndividual]:
        """交叉操作"""
        method = self.config.crossover_method

        if method == "single_point":
            return self._single_point_crossover(parent1, parent2)
        elif method == "two_point":
            return self._two_point_crossover(parent1, parent2)
        elif method == "uniform":
            return self._uniform_crossover(parent1, parent2)
        elif method == "arithmetic":
            return self._arithmetic_crossover(parent1, parent2)
        else:
            return self._uniform_crossover(parent1, parent2)

    def _single_point_crossover(self, parent1: GAIndividual, parent2: GAIndividual) -> Tuple[GAIndividual, GAIndividual]:
        """单点交叉"""
        point = random.randint(1, self.config.dimensions - 1)
        offspring1 = GAIndividual(self.config.dimensions, self.config.bounds)
        offspring2 = GAIndividual(self.config.dimensions, self.config.bounds)

        offspring1.position = parent1.position[:point] + parent2.position[point:]
        offspring2.position = parent2.position[:point] + parent1.position[point:]

        return offspring1, offspring2

    def _two_point_crossover(self, parent1: GAIndividual, parent2: GAIndividual) -> Tuple[GAIndividual, GAIndividual]:
        """两点交叉"""
        point1 = random.randint(0, self.config.dimensions - 2)
        point2 = random.randint(point1 + 1, self.config.dimensions - 1)

        offspring1 = GAIndividual(self.config.dimensions, self.config.bounds)
        offspring2 = GAIndividual(self.config.dimensions, self.config.bounds)

        offspring1.position = (parent1.position[:point1] +
                              parent2.position[point1:point2] +
                              parent1.position[point2:])
        offspring2.position = (parent2.position[:point1] +
                              parent1.position[point1:point2] +
                              parent2.position[point2:])

        return offspring1, offspring2

    def _uniform_crossover(self, parent1: GAIndividual, parent2: GAIndividual) -> Tuple[GAIndividual, GAIndividual]:
        """均匀交叉"""
        offspring1 = GAIndividual(self.config.dimensions, self.config.bounds)
        offspring2 = GAIndividual(self.config.dimensions, self.config.bounds)

        for i in range(self.config.dimensions):
            if random.random() < 0.5:
                offspring1.position[i] = parent1.position[i]
                offspring2.position[i] = parent2.position[i]
            else:
                offspring1.position[i] = parent2.position[i]
                offspring2.position[i] = parent1.position[i]

        return offspring1, offspring2

    def _arithmetic_crossover(self, parent1: GAIndividual, parent2: GAIndividual) -> Tuple[GAIndividual, GAIndividual]:
        """算术交叉"""
        alpha = random.random()
        offspring1 = GAIndividual(self.config.dimensions, self.config.bounds)
        offspring2 = GAIndividual(self.config.dimensions, self.config.bounds)

        for i in range(self.config.dimensions):
            offspring1.position[i] = alpha * parent1.position[i] + (1 - alpha) * parent2.position[i]
            offspring2.position[i] = alpha * parent2.position[i] + (1 - alpha) * parent1.position[i]

        return offspring1, offspring2

    def _mutate(self, individual: GAIndividual) -> None:
        """变异操作"""
        method = self.config.mutation_method

        if method == "bit_flip":
            self._bit_flip_mutation(individual)
        elif method == "gaussian":
            self._gaussian_mutation(individual)
        elif method == "polynomial":
            self._polynomial_mutation(individual)
        else:
            self._gaussian_mutation(individual)

    def _bit_flip_mutation(self, individual: GAIndividual) -> None:
        """位翻转变异（适用于二进制编码，这里用于实数编码的随机重置）"""
        for i in range(self.config.dimensions):
            if random.random() < self.config.mutation_rate:
                individual.position[i] = random.uniform(self.config.bounds[0], self.config.bounds[1])

    def _gaussian_mutation(self, individual: GAIndividual) -> None:
        """高斯变异"""
        for i in range(self.config.dimensions):
            if random.random() < self.config.mutation_rate:
                noise = random.gauss(0, 1) * 0.1 * (self.config.bounds[1] - self.config.bounds[0])
                individual.position[i] += noise
                individual.position[i] = clamp(individual.position[i], self.config.bounds[0], self.config.bounds[1])

    def _polynomial_mutation(self, individual: GAIndividual) -> None:
        """多项式变异"""
        eta_m = 20.0  # 分布指数

        for i in range(self.config.dimensions):
            if random.random() < self.config.mutation_rate:
                delta1 = (individual.position[i] - self.config.bounds[0]) / (self.config.bounds[1] - self.config.bounds[0])
                delta2 = (self.config.bounds[1] - individual.position[i]) / (self.config.bounds[1] - self.config.bounds[0])

                rand = random.random()
                mut_pow = 1.0 / (eta_m + 1.0)

                if rand <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1))
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1))
                    delta_q = 1.0 - val ** mut_pow

                individual.position[i] += delta_q * (self.config.bounds[1] - self.config.bounds[0])
                individual.position[i] = clamp(individual.position[i], self.config.bounds[0], self.config.bounds[1])

    def _calculate_sharing_function(self, ind1: GAIndividual, ind2: GAIndividual) -> float:
        """计算共享函数值"""
        distance = euclidean_distance(ind1.position, ind2.position)
        sigma = self.config.sharing_sigma

        if distance >= sigma:
            return 0.0
        else:
            return 1.0 - (distance / sigma) ** 2

    def _apply_sharing(self) -> None:
        """应用共享函数（niching）"""
        for i, ind1 in enumerate(self.population):
            niche_count = sum(self._calculate_sharing_function(ind1, ind2) for ind2 in self.population)
            if niche_count > 0:
                ind1.shared_fitness = ind1.fitness / niche_count
            else:
                ind1.shared_fitness = ind1.fitness

    def optimize(self, fitness_func: FitnessFunction) -> Tuple[List[float], float]:
        """执行GA优化"""
        self._initialize_population()

        # 初始评估
        for individual in self.population:
            individual.fitness = fitness_func(individual.position)
            self._update_best(individual.position, individual.fitness)

        for iteration in range(self.config.max_iterations):
            self.iteration = iteration

            # 精英保留
            sorted_pop = sorted(self.population,
                               key=lambda x: x.fitness if x.fitness is not None else float('inf'))
            elites = sorted_pop[:self.config.elitism_count]

            # 应用共享函数
            if self.config.use_sharing:
                self._apply_sharing()

            # 生成新种群
            new_population = []

            while len(new_population) < self.config.population_size - self.config.elitism_count:
                # 选择
                parent1 = self._select_parent()
                parent2 = self._select_parent()

                # 交叉
                if random.random() < self.config.crossover_rate:
                    offspring1, offspring2 = self._crossover(parent1, parent2)
                else:
                    offspring1 = GAIndividual(self.config.dimensions, self.config.bounds)
                    offspring1.position = parent1.position.copy()
                    offspring2 = GAIndividual(self.config.dimensions, self.config.bounds)
                    offspring2.position = parent2.position.copy()

                # 变异
                self._mutate(offspring1)
                self._mutate(offspring2)

                new_population.extend([offspring1, offspring2])

            # 截断多余个体
            new_population = new_population[:self.config.population_size - self.config.elitism_count]

            # 添加精英
            new_population.extend(elites)
            self.population = new_population

            # 评估
            for individual in self.population:
                if individual.fitness is None:
                    individual.fitness = fitness_func(individual.position)
                    self._update_best(individual.position, individual.fitness)

            self.history.append(self.best_fitness)

        return self.best_solution, self.best_fitness


# =============================================================================
# Multi-Objective Swarm
# =============================================================================

class MOParticle:
    """多目标粒子类"""

    def __init__(self, dimensions: int, bounds: Tuple[float, float], num_objectives: int):
        self.position = [random.uniform(bounds[0], bounds[1]) for _ in range(dimensions)]
        self.velocity = [random.uniform(-1, 1) for _ in range(dimensions)]
        self.best_position = self.position.copy()
        self.objectives: List[float] = [float('inf')] * num_objectives
        self.best_objectives: List[float] = [float('inf')] * num_objectives
        self.crowding_distance: float = 0.0


class MOPSO:
    """
    多目标粒子群优化 (Multi-Objective PSO)

    特点:
    - 外部存档保存非支配解
    - 拥挤距离保持多样性
    - 从存档中选择全局最优
    """

    def __init__(self, config: SwarmConfig, num_objectives: int = 2):
        self.config = config
        self.num_objectives = num_objectives
        self.swarm: List[MOParticle] = []
        self.archive: List[MOParticle] = []
        self.archive_size = config.population_size * 2

    def _initialize_swarm(self) -> None:
        """初始化粒子群"""
        self.swarm = [
            MOParticle(self.config.dimensions, self.config.bounds, self.num_objectives)
            for _ in range(self.config.population_size)
        ]

    def _evaluate(self, particle: MOParticle, fitness_func: MultiObjectiveFunction) -> None:
        """评估多目标适应度"""
        particle.objectives = fitness_func(particle.position)

        # 更新个人最佳（如果新解支配旧解）
        if dominates(particle.objectives, particle.best_objectives):
            particle.best_position = particle.position.copy()
            particle.best_objectives = particle.objectives.copy()

    def _update_archive(self) -> None:
        """更新外部存档"""
        # 合并当前粒子和存档
        candidates = self.swarm + self.archive

        # 非支配排序
        all_objectives = [p.objectives for p in candidates]
        fronts = non_dominated_sort(all_objectives)

        # 选择非支配解进入存档
        new_archive = []
        for front in fronts:
            if len(new_archive) + len(front) <= self.archive_size:
                for idx in front:
                    new_archive.append(candidates[idx])
            else:
                # 使用拥挤距离选择
                front_particles = [candidates[i] for i in front]
                front_objectives = [candidates[i].objectives for i in front]
                distances = crowding_distance(front_objectives)

                for p, d in zip(front_particles, distances):
                    p.crowding_distance = d

                # 按拥挤距离排序，选择最分散的
                sorted_front = sorted(front_particles, key=lambda p: p.crowding_distance, reverse=True)
                remaining = self.archive_size - len(new_archive)
                new_archive.extend(sorted_front[:remaining])
                break

        self.archive = new_archive

    def _select_global_best(self) -> MOParticle:
        """从存档中选择全局最优（使用拥挤距离锦标赛）"""
        if len(self.archive) <= 2:
            return random.choice(self.archive) if self.archive else random.choice(self.swarm)

        # 锦标赛选择
        candidates = random.sample(self.archive, min(3, len(self.archive)))
        return max(candidates, key=lambda p: p.crowding_distance)

    def optimize(self, fitness_func: MultiObjectiveFunction) -> List[Tuple[List[float], List[float]]]:
        """
        执行MOPSO优化

        返回: 存档中的解列表，每个元素为 (position, objectives)
        """
        self._initialize_swarm()

        # 初始评估
        for particle in self.swarm:
            self._evaluate(particle, fitness_func)
            particle.best_position = particle.position.copy()
            particle.best_objectives = particle.objectives.copy()

        self._update_archive()

        for iteration in range(self.config.max_iterations):
            w = self.config.w - (self.config.w - 0.4) * iteration / self.config.max_iterations

            for particle in self.swarm:
                # 从存档选择全局最优
                global_best = self._select_global_best()

                # 速度更新
                for i in range(self.config.dimensions):
                    r1, r2 = random.random(), random.random()
                    cognitive = self.config.c1 * r1 * (particle.best_position[i] - particle.position[i])
                    social = self.config.c2 * r2 * (global_best.position[i] - particle.position[i])
                    particle.velocity[i] = w * particle.velocity[i] + cognitive + social

                # 位置更新
                for i in range(self.config.dimensions):
                    particle.position[i] += particle.velocity[i]
                    particle.position[i] = clamp(particle.position[i],
                                                self.config.bounds[0], self.config.bounds[1])

                # 评估
                self._evaluate(particle, fitness_func)

            # 更新存档
            self._update_archive()

        # 返回存档中的解
        return [(p.position, p.objectives) for p in self.archive]


class MOABC:
    """
    多目标人工蜂群 (Multi-Objective ABC)

    特点:
    - 非支配排序选择食物源
    - 外部存档保存Pareto前沿
    """

    def __init__(self, config: SwarmConfig, num_objectives: int = 2):
        self.config = config
        self.num_objectives = num_objectives
        self.food_sources: List[MOParticle] = []
        self.archive: List[MOParticle] = []
        self.archive_size = config.population_size * 2

    def _initialize_food_sources(self) -> None:
        """初始化食物源"""
        num_sources = self.config.population_size // 2
        self.food_sources = [
            MOParticle(self.config.dimensions, self.config.bounds, self.num_objectives)
            for _ in range(num_sources)
        ]

    def _evaluate(self, particle: MOParticle, fitness_func: MultiObjectiveFunction) -> None:
        """评估多目标适应度"""
        particle.objectives = fitness_func(particle.position)

    def _update_archive(self) -> None:
        """更新外部存档"""
        candidates = self.food_sources + self.archive

        all_objectives = [p.objectives for p in candidates]
        fronts = non_dominated_sort(all_objectives)

        new_archive = []
        for front in fronts:
            if len(new_archive) + len(front) <= self.archive_size:
                for idx in front:
                    new_archive.append(candidates[idx])
            else:
                front_particles = [candidates[i] for i in front]
                front_objectives = [candidates[i].objectives for i in front]
                distances = crowding_distance(front_objectives)

                for p, d in zip(front_particles, distances):
                    p.crowding_distance = d

                sorted_front = sorted(front_particles, key=lambda p: p.crowding_distance, reverse=True)
                remaining = self.archive_size - len(new_archive)
                new_archive.extend(sorted_front[:remaining])
                break

        self.archive = new_archive

    def _select_food_source(self) -> MOParticle:
        """基于拥挤距离选择食物源"""
        if not self.food_sources:
            return None

        # 计算选择概率（基于拥挤距离）
        distances = [f.crowding_distance for f in self.food_sources]
        total = sum(distances) if sum(distances) > 0 else len(self.food_sources)

        if total > 0 and sum(distances) > 0:
            probabilities = [d / total for d in distances]
            return random.choices(self.food_sources, weights=probabilities)[0]
        else:
            return random.choice(self.food_sources)

    def optimize(self, fitness_func: MultiObjectiveFunction) -> List[Tuple[List[float], List[float]]]:
        """执行MOABC优化"""
        self._initialize_food_sources()

        # 初始评估
        for food in self.food_sources:
            self._evaluate(food, fitness_func)
            food.best_position = food.position.copy()
            food.best_objectives = food.objectives.copy()

        self._update_archive()

        for iteration in range(self.config.max_iterations):
            # 雇佣蜂阶段
            for i, food in enumerate(self.food_sources):
                new_pos = self._generate_neighbor(food.position, i)
                new_food = MOParticle(self.config.dimensions, self.config.bounds, self.num_objectives)
                new_food.position = new_pos
                self._evaluate(new_food, fitness_func)

                # 非支配比较
                if dominates(new_food.objectives, food.objectives):
                    food.position = new_pos
                    food.objectives = new_food.objectives
                    food.trial = 0
                elif not dominates(food.objectives, new_food.objectives):
                    # 互不支配，随机选择
                    if random.random() < 0.5:
                        food.position = new_pos
                        food.objectives = new_food.objectives
                    food.trial += 1
                else:
                    food.trial += 1

            # 观察蜂阶段
            num_onlookers = len(self.food_sources)
            for _ in range(num_onlookers):
                selected = self._select_food_source()
                if selected:
                    idx = self.food_sources.index(selected)
                    new_pos = self._generate_neighbor(selected.position, idx)
                    new_food = MOParticle(self.config.dimensions, self.config.bounds, self.num_objectives)
                    new_food.position = new_pos
                    self._evaluate(new_food, fitness_func)

                    if dominates(new_food.objectives, selected.objectives):
                        selected.position = new_pos
                        selected.objectives = new_food.objectives
                        selected.trial = 0

            # 侦查蜂阶段
            for food in self.food_sources:
                if food.trial >= self.config.limit:
                    food.position = [random.uniform(self.config.bounds[0], self.config.bounds[1])
                                   for _ in range(self.config.dimensions)]
                    self._evaluate(food, fitness_func)
                    food.trial = 0

            self._update_archive()

        return [(p.position, p.objectives) for p in self.archive]

    def _generate_neighbor(self, position: List[float], exclude_idx: int) -> List[float]:
        """生成邻域解"""
        new_pos = position.copy()
        dim = random.randint(0, self.config.dimensions - 1)

        partner_idx = random.randint(0, len(self.food_sources) - 1)
        while partner_idx == exclude_idx:
            partner_idx = random.randint(0, len(self.food_sources) - 1)

        partner_pos = self.food_sources[partner_idx].position
        phi = random.uniform(-1, 1)

        new_pos[dim] = position[dim] + phi * (position[dim] - partner_pos[dim])
        new_pos[dim] = clamp(new_pos[dim], self.config.bounds[0], self.config.bounds[1])

        return new_pos


# =============================================================================
# Factory and Utility Functions
# =============================================================================

class OptimizerType(Enum):
    """优化器类型枚举"""
    PSO = auto()
    ACO = auto()
    ABC = auto()
    FIREFLY = auto()
    CUCKOO = auto()
    BAT = auto()
    GWO = auto()
    WOA = auto()
    DE = auto()
    ES = auto()
    GA = auto()
    MOPSO = auto()
    MOABC = auto()


def create_optimizer(optimizer_type: OptimizerType, config: SwarmConfig, **kwargs) -> BaseOptimizer:
    """
    工厂函数 - 创建优化器实例

    Args:
        optimizer_type: 优化器类型
        config: 配置对象
        **kwargs: 额外参数（如ACO的distance_matrix）

    Returns:
        优化器实例
    """
    if optimizer_type == OptimizerType.PSO:
        return ParticleSwarmOptimization(config)
    elif optimizer_type == OptimizerType.ACO:
        return AntColonyOptimization(config, kwargs.get('distance_matrix'))
    elif optimizer_type == OptimizerType.ABC:
        return ArtificialBeeColony(config)
    elif optimizer_type == OptimizerType.FIREFLY:
        return FireflyAlgorithm(config)
    elif optimizer_type == OptimizerType.CUCKOO:
        return CuckooSearch(config)
    elif optimizer_type == OptimizerType.BAT:
        return BatAlgorithm(config)
    elif optimizer_type == OptimizerType.GWO:
        return GreyWolfOptimizer(config)
    elif optimizer_type == OptimizerType.WOA:
        return WhaleOptimizationAlgorithm(config)
    elif optimizer_type == OptimizerType.DE:
        return DifferentialEvolution(config)
    elif optimizer_type == OptimizerType.ES:
        return EvolutionStrategy(config)
    elif optimizer_type == OptimizerType.GA:
        return GeneticAlgorithm(config)
    elif optimizer_type == OptimizerType.MOPSO:
        return MOPSO(config, kwargs.get('num_objectives', 2))
    elif optimizer_type == OptimizerType.MOABC:
        return MOABC(config, kwargs.get('num_objectives', 2))
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


# =============================================================================
# Example Usage and Test Functions
# =============================================================================

def sphere_function(x: List[float]) -> float:
    """Sphere测试函数"""
    return sum(xi ** 2 for xi in x)


def rastrigin_function(x: List[float]) -> float:
    """Rastrigin测试函数"""
    A = 10
    n = len(x)
    return A * n + sum(xi ** 2 - A * math.cos(2 * math.pi * xi) for xi in x)


def rosenbrock_function(x: List[float]) -> float:
    """Rosenbrock测试函数"""
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))


def ackley_function(x: List[float]) -> float:
    """Ackley测试函数"""
    a, b, c = 20, 0.2, 2 * math.pi
    n = len(x)
    sum1 = sum(xi ** 2 for xi in x)
    sum2 = sum(math.cos(c * xi) for xi in x)
    return -a * math.exp(-b * math.sqrt(sum1 / n)) - math.exp(sum2 / n) + a + math.e


def run_demo():
    """运行演示"""
    print("=" * 60)
    print("群智能与进化计算算法演示")
    print("=" * 60)

    config = SwarmConfig(
        population_size=30,
        max_iterations=100,
        dimensions=10,
        bounds=(-5.0, 5.0),
        minimize=True,
        seed=42
    )

    test_functions = {
        "Sphere": sphere_function,
        "Rastrigin": rastrigin_function,
    }

    algorithms = {
        "PSO": OptimizerType.PSO,
        "ABC": OptimizerType.ABC,
        "Firefly": OptimizerType.FIREFLY,
        "Cuckoo": OptimizerType.CUCKOO,
        "Bat": OptimizerType.BAT,
        "GWO": OptimizerType.GWO,
        "WOA": OptimizerType.WOA,
        "DE": OptimizerType.DE,
        "GA": OptimizerType.GA,
    }

    for func_name, func in test_functions.items():
        print(f"\n测试函数: {func_name}")
        print("-" * 40)

        for algo_name, algo_type in algorithms.items():
            optimizer = create_optimizer(algo_type, config)
            solution, fitness = optimizer.optimize(func)
            print(f"  {algo_name:10s}: 最优值 = {fitness:.6e}")

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
