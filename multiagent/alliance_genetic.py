"""
联盟遗传算法优化模块

提供基于遗传算法的多智能体联盟优化功能，包括联盟编码、适应度评估、
选择算子、交叉算子、变异算子和精英保留等完整遗传算法流程。
"""

from __future__ import annotations

import random
import copy
from typing import Dict, List, Tuple, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod


class SelectionMethod(Enum):
    """选择算子方法枚举"""
    TOURNAMENT = "tournament"      # 锦标赛选择
    ROULETTE = "roulette"          # 轮盘赌选择
    RANK = "rank"                  # 排序选择
    STOCHASTIC_UNIVERSAL = "sus"   # 随机通用采样


class CrossoverMethod(Enum):
    """交叉算子方法枚举"""
    SINGLE_POINT = "single_point"  # 单点交叉
    UNIFORM = "uniform"            # 均匀交叉
    TWO_POINT = "two_point"        # 两点交叉
    PMX = "pmx"                    # 部分映射交叉(用于排列)


class MutationMethod(Enum):
    """变异算子方法枚举"""
    BIT_FLIP = "bit_flip"          # 位翻转变异
    SWAP = "swap"                  # 交换变异
    INVERSION = "inversion"        # 逆转变异
    GAUSSIAN = "gaussian"          # 高斯变异


@dataclass
class AgentCapability:
    """智能体能力描述"""
    agent_id: str
    skills: Set[str] = field(default_factory=set)
    efficiency: float = 1.0          # 效率系数
    reliability: float = 1.0         # 可靠性
    cost: float = 1.0                # 成本
    
    def __hash__(self) -> int:
        return hash(self.agent_id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AgentCapability):
            return False
        return self.agent_id == other.agent_id


@dataclass
class TaskRequirement:
    """任务需求描述"""
    task_id: str
    required_skills: Set[str] = field(default_factory=set)
    min_efficiency: float = 0.0
    max_cost: float = float('inf')
    priority: float = 1.0


@dataclass
class AllianceChromosome:
    """
    联盟染色体表示
    
    编码方案: 使用二进制/整数向量表示联盟结构
    - 对于N个智能体和M个任务，使用N×M的矩阵
    - chromosome[i][j] = 1 表示智能体i被分配到任务j
    """
    assignment_matrix: np.ndarray    # 分配矩阵 (agents × tasks)
    agent_ids: List[str] = field(default_factory=list)
    task_ids: List[str] = field(default_factory=list)
    fitness: float = 0.0
    generation: int = 0
    
    def __post_init__(self):
        if len(self.agent_ids) == 0:
            self.agent_ids = [f"agent_{i}" for i in range(self.assignment_matrix.shape[0])]
        if len(self.task_ids) == 0:
            self.task_ids = [f"task_{j}" for j in range(self.assignment_matrix.shape[1])]
    
    def copy(self) -> AllianceChromosome:
        """创建深拷贝"""
        return AllianceChromosome(
            assignment_matrix=self.assignment_matrix.copy(),
            agent_ids=self.agent_ids.copy(),
            task_ids=self.task_ids.copy(),
            fitness=self.fitness,
            generation=self.generation
        )
    
    def get_agent_tasks(self, agent_idx: int) -> List[int]:
        """获取指定智能体被分配的任务索引列表"""
        return [j for j, assigned in enumerate(self.assignment_matrix[agent_idx]) if assigned > 0]
    
    def get_task_agents(self, task_idx: int) -> List[int]:
        """获取被分配到指定任务的智能体索引列表"""
        return [i for i, assigned in enumerate(self.assignment_matrix[:, task_idx]) if assigned > 0]
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证染色体有效性"""
        errors = []
        
        # 检查每个任务至少有一个智能体
        for j in range(len(self.task_ids)):
            agents = self.get_task_agents(j)
            if len(agents) == 0:
                errors.append(f"任务 {self.task_ids[j]} 没有分配智能体")
        
        # 检查每个智能体至少有一个任务（可选约束）
        for i in range(len(self.agent_ids)):
            tasks = self.get_agent_tasks(i)
            if len(tasks) == 0:
                errors.append(f"智能体 {self.agent_ids[i]} 没有分配任务")
        
        return len(errors) == 0, errors


class FitnessEvaluator(ABC):
    """适应度评估器基类"""
    
    @abstractmethod
    def evaluate(self, chromosome: AllianceChromosome, 
                 agents: List[AgentCapability],
                 tasks: List[TaskRequirement]) -> float:
        """评估染色体适应度"""
        pass


class CollaborationEfficiencyEvaluator(FitnessEvaluator):
    """
    协作效率适应度评估器
    
    综合考虑以下因素:
    1. 技能匹配度
    2. 负载均衡
    3. 通信开销
    4. 成本效益
    """
    
    def __init__(self,
                 skill_weight: float = 0.4,
                 balance_weight: float = 0.2,
                 communication_weight: float = 0.2,
                 cost_weight: float = 0.2):
        self.skill_weight = skill_weight
        self.balance_weight = balance_weight
        self.communication_weight = communication_weight
        self.cost_weight = cost_weight
    
    def evaluate(self, chromosome: AllianceChromosome,
                 agents: List[AgentCapability],
                 tasks: List[TaskRequirement]) -> float:
        """评估协作效率"""
        skill_score = self._evaluate_skill_match(chromosome, agents, tasks)
        balance_score = self._evaluate_load_balance(chromosome)
        comm_score = self._evaluate_communication(chromosome, agents, tasks)
        cost_score = self._evaluate_cost(chromosome, agents, tasks)
        
        total_fitness = (
            self.skill_weight * skill_score +
            self.balance_weight * balance_score +
            self.communication_weight * comm_score +
            self.cost_weight * cost_score
        )
        
        return max(0.0, min(1.0, total_fitness))
    
    def _evaluate_skill_match(self, chromosome: AllianceChromosome,
                              agents: List[AgentCapability],
                              tasks: List[TaskRequirement]) -> float:
        """评估技能匹配度"""
        total_score = 0.0
        total_weight = 0.0
        
        for j, task in enumerate(tasks):
            task_agents = chromosome.get_task_agents(j)
            if not task_agents:
                continue
            
            # 计算任务j的技能覆盖
            covered_skills = set()
            for agent_idx in task_agents:
                agent = agents[agent_idx]
                covered_skills.update(agent.skills)
            
            if task.required_skills:
                match_ratio = len(covered_skills & task.required_skills) / len(task.required_skills)
            else:
                match_ratio = 1.0
            
            total_score += match_ratio * task.priority
            total_weight += task.priority
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _evaluate_load_balance(self, chromosome: AllianceChromosome) -> float:
        """评估负载均衡"""
        task_counts = [len(chromosome.get_agent_tasks(i)) 
                      for i in range(len(chromosome.agent_ids))]
        
        if not task_counts or max(task_counts) == 0:
            return 1.0
        
        # 使用变异系数衡量均衡度
        mean_count = np.mean(task_counts)
        std_count = np.std(task_counts)
        
        if mean_count == 0:
            return 1.0
        
        cv = std_count / mean_count
        balance_score = max(0.0, 1.0 - cv)
        
        return balance_score
    
    def _evaluate_communication(self, chromosome: AllianceChromosome,
                                agents: List[AgentCapability],
                                tasks: List[TaskRequirement]) -> float:
        """评估通信开销"""
        total_communication = 0.0
        max_possible = 0.0
        
        for j in range(len(tasks)):
            task_agents = chromosome.get_task_agents(j)
            n_agents = len(task_agents)
            
            # 通信复杂度与团队大小呈平方关系
            communication_cost = n_agents * (n_agents - 1) / 2 if n_agents > 1 else 0
            total_communication += communication_cost
            max_possible += len(agents) * (len(agents) - 1) / 2
        
        if max_possible == 0:
            return 1.0
        
        return 1.0 - (total_communication / max_possible)
    
    def _evaluate_cost(self, chromosome: AllianceChromosome,
                       agents: List[AgentCapability],
                       tasks: List[TaskRequirement]) -> float:
        """评估成本效益"""
        total_cost = 0.0
        max_cost = 0.0
        
        for j, task in enumerate(tasks):
            task_agents = chromosome.get_task_agents(j)
            task_cost = sum(agents[i].cost for i in task_agents)
            total_cost += task_cost
            max_cost += sum(agent.cost for agent in agents)
        
        if max_cost == 0:
            return 1.0
        
        cost_ratio = total_cost / max_cost
        return max(0.0, 1.0 - cost_ratio)


class SelectionOperator:
    """选择算子"""
    
    def __init__(self, method: SelectionMethod = SelectionMethod.TOURNAMENT,
                 tournament_size: int = 3,
                 elitism_count: int = 2):
        self.method = method
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
    
    def select(self, population: List[AllianceChromosome], 
               num_selections: int) -> List[AllianceChromosome]:
        """执行选择操作"""
        if self.method == SelectionMethod.TOURNAMENT:
            return self._tournament_selection(population, num_selections)
        elif self.method == SelectionMethod.ROULETTE:
            return self._roulette_selection(population, num_selections)
        elif self.method == SelectionMethod.RANK:
            return self._rank_selection(population, num_selections)
        elif self.method == SelectionMethod.STOCHASTIC_UNIVERSAL:
            return self._sus_selection(population, num_selections)
        else:
            raise ValueError(f"未知的选择方法: {self.method}")
    
    def _tournament_selection(self, population: List[AllianceChromosome],
                              num_selections: int) -> List[AllianceChromosome]:
        """锦标赛选择"""
        selected = []
        for _ in range(num_selections):
            tournament = random.sample(population, 
                                      min(self.tournament_size, len(population)))
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner.copy())
        return selected
    
    def _roulette_selection(self, population: List[AllianceChromosome],
                            num_selections: int) -> List[AllianceChromosome]:
        """轮盘赌选择"""
        total_fitness = sum(c.fitness for c in population)
        if total_fitness == 0:
            return [random.choice(population).copy() for _ in range(num_selections)]
        
        selected = []
        for _ in range(num_selections):
            pick = random.uniform(0, total_fitness)
            current = 0
            for chromosome in population:
                current += chromosome.fitness
                if current >= pick:
                    selected.append(chromosome.copy())
                    break
            else:
                selected.append(population[-1].copy())
        
        return selected
    
    def _rank_selection(self, population: List[AllianceChromosome],
                        num_selections: int) -> List[AllianceChromosome]:
        """排序选择"""
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        ranks = list(range(len(sorted_pop), 0, -1))
        total_rank = sum(ranks)
        
        selected = []
        for _ in range(num_selections):
            pick = random.uniform(0, total_rank)
            current = 0
            for i, chromosome in enumerate(sorted_pop):
                current += ranks[i]
                if current >= pick:
                    selected.append(chromosome.copy())
                    break
        
        return selected
    
    def _sus_selection(self, population: List[AllianceChromosome],
                       num_selections: int) -> List[AllianceChromosome]:
        """随机通用采样选择"""
        total_fitness = sum(c.fitness for c in population)
        if total_fitness == 0:
            return [random.choice(population).copy() for _ in range(num_selections)]
        
        pointer_distance = total_fitness / num_selections
        start = random.uniform(0, pointer_distance)
        
        selected = []
        current_fitness = 0
        population_idx = 0
        
        for i in range(num_selections):
            pointer = start + i * pointer_distance
            while current_fitness < pointer and population_idx < len(population):
                current_fitness += population[population_idx].fitness
                population_idx += 1
            if population_idx > 0:
                selected.append(population[population_idx - 1].copy())
            else:
                selected.append(population[0].copy())
        
        return selected
    
    def get_elites(self, population: List[AllianceChromosome]) -> List[AllianceChromosome]:
        """获取精英个体"""
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        return [c.copy() for c in sorted_pop[:self.elitism_count]]


class CrossoverOperator:
    """交叉算子"""
    
    def __init__(self, method: CrossoverMethod = CrossoverMethod.UNIFORM,
                 crossover_rate: float = 0.8):
        self.method = method
        self.crossover_rate = crossover_rate
    
    def crossover(self, parent1: AllianceChromosome,
                  parent2: AllianceChromosome) -> Tuple[AllianceChromosome, AllianceChromosome]:
        """执行交叉操作"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        if self.method == CrossoverMethod.SINGLE_POINT:
            return self._single_point_crossover(parent1, parent2)
        elif self.method == CrossoverMethod.UNIFORM:
            return self._uniform_crossover(parent1, parent2)
        elif self.method == CrossoverMethod.TWO_POINT:
            return self._two_point_crossover(parent1, parent2)
        elif self.method == CrossoverMethod.PMX:
            return self._pmx_crossover(parent1, parent2)
        else:
            raise ValueError(f"未知的交叉方法: {self.method}")
    
    def _single_point_crossover(self, parent1: AllianceChromosome,
                                parent2: AllianceChromosome) -> Tuple[AllianceChromosome, AllianceChromosome]:
        """单点交叉"""
        rows, cols = parent1.assignment_matrix.shape
        
        # 随机选择交叉点
        point = random.randint(1, rows * cols - 1)
        row_point = point // cols
        col_point = point % cols
        
        child1_matrix = parent1.assignment_matrix.copy()
        child2_matrix = parent2.assignment_matrix.copy()
        
        # 交换交叉点后的部分
        for i in range(row_point, rows):
            start_col = col_point if i == row_point else 0
            for j in range(start_col, cols):
                child1_matrix[i, j] = parent2.assignment_matrix[i, j]
                child2_matrix[i, j] = parent1.assignment_matrix[i, j]
        
        child1 = AllianceChromosome(
            assignment_matrix=child1_matrix,
            agent_ids=parent1.agent_ids.copy(),
            task_ids=parent1.task_ids.copy()
        )
        child2 = AllianceChromosome(
            assignment_matrix=child2_matrix,
            agent_ids=parent2.agent_ids.copy(),
            task_ids=parent2.task_ids.copy()
        )
        
        return child1, child2
    
    def _uniform_crossover(self, parent1: AllianceChromosome,
                           parent2: AllianceChromosome) -> Tuple[AllianceChromosome, AllianceChromosome]:
        """均匀交叉"""
        rows, cols = parent1.assignment_matrix.shape
        
        mask = np.random.random((rows, cols)) < 0.5
        
        child1_matrix = np.where(mask, parent1.assignment_matrix, parent2.assignment_matrix)
        child2_matrix = np.where(mask, parent2.assignment_matrix, parent1.assignment_matrix)
        
        child1 = AllianceChromosome(
            assignment_matrix=child1_matrix,
            agent_ids=parent1.agent_ids.copy(),
            task_ids=parent1.task_ids.copy()
        )
        child2 = AllianceChromosome(
            assignment_matrix=child2_matrix,
            agent_ids=parent2.agent_ids.copy(),
            task_ids=parent2.task_ids.copy()
        )
        
        return child1, child2
    
    def _two_point_crossover(self, parent1: AllianceChromosome,
                             parent2: AllianceChromosome) -> Tuple[AllianceChromosome, AllianceChromosome]:
        """两点交叉"""
        rows, cols = parent1.assignment_matrix.shape
        total_elements = rows * cols
        
        point1 = random.randint(0, total_elements - 2)
        point2 = random.randint(point1 + 1, total_elements - 1)
        
        child1_matrix = parent1.assignment_matrix.copy()
        child2_matrix = parent2.assignment_matrix.copy()
        
        for idx in range(point1, point2):
            i, j = idx // cols, idx % cols
            child1_matrix[i, j] = parent2.assignment_matrix[i, j]
            child2_matrix[i, j] = parent1.assignment_matrix[i, j]
        
        child1 = AllianceChromosome(
            assignment_matrix=child1_matrix,
            agent_ids=parent1.agent_ids.copy(),
            task_ids=parent1.task_ids.copy()
        )
        child2 = AllianceChromosome(
            assignment_matrix=child2_matrix,
            agent_ids=parent2.agent_ids.copy(),
            task_ids=parent2.task_ids.copy()
        )
        
        return child1, child2
    
    def _pmx_crossover(self, parent1: AllianceChromosome,
                       parent2: AllianceChromosome) -> Tuple[AllianceChromosome, AllianceChromosome]:
        """部分映射交叉(适用于排列编码)"""
        # 简化实现：对每行分别进行PMX
        rows, cols = parent1.assignment_matrix.shape
        
        child1_matrix = np.zeros_like(parent1.assignment_matrix)
        child2_matrix = np.zeros_like(parent2.assignment_matrix)
        
        for i in range(rows):
            if random.random() < 0.5:
                child1_matrix[i] = parent1.assignment_matrix[i]
                child2_matrix[i] = parent2.assignment_matrix[i]
            else:
                child1_matrix[i] = parent2.assignment_matrix[i]
                child2_matrix[i] = parent1.assignment_matrix[i]
        
        child1 = AllianceChromosome(
            assignment_matrix=child1_matrix,
            agent_ids=parent1.agent_ids.copy(),
            task_ids=parent1.task_ids.copy()
        )
        child2 = AllianceChromosome(
            assignment_matrix=child2_matrix,
            agent_ids=parent2.agent_ids.copy(),
            task_ids=parent2.task_ids.copy()
        )
        
        return child1, child2


class MutationOperator:
    """变异算子"""
    
    def __init__(self, method: MutationMethod = MutationMethod.BIT_FLIP,
                 mutation_rate: float = 0.1,
                 gaussian_std: float = 0.1):
        self.method = method
        self.mutation_rate = mutation_rate
        self.gaussian_std = gaussian_std
    
    def mutate(self, chromosome: AllianceChromosome) -> AllianceChromosome:
        """执行变异操作"""
        mutated = chromosome.copy()
        
        if self.method == MutationMethod.BIT_FLIP:
            mutated.assignment_matrix = self._bit_flip_mutate(mutated.assignment_matrix)
        elif self.method == MutationMethod.SWAP:
            mutated.assignment_matrix = self._swap_mutate(mutated.assignment_matrix)
        elif self.method == MutationMethod.INVERSION:
            mutated.assignment_matrix = self._inversion_mutate(mutated.assignment_matrix)
        elif self.method == MutationMethod.GAUSSIAN:
            mutated.assignment_matrix = self._gaussian_mutate(mutated.assignment_matrix)
        else:
            raise ValueError(f"未知的变异方法: {self.method}")
        
        return mutated
    
    def _bit_flip_mutate(self, matrix: np.ndarray) -> np.ndarray:
        """位翻转变异"""
        mutated = matrix.copy()
        rows, cols = mutated.shape
        
        for i in range(rows):
            for j in range(cols):
                if random.random() < self.mutation_rate:
                    mutated[i, j] = 1.0 - mutated[i, j]
        
        return mutated
    
    def _swap_mutate(self, matrix: np.ndarray) -> np.ndarray:
        """交换变异"""
        mutated = matrix.copy()
        rows, cols = mutated.shape
        
        if random.random() < self.mutation_rate and rows > 1:
            # 随机选择两行交换
            i, j = random.sample(range(rows), 2)
            mutated[[i, j]] = mutated[[j, i]]
        
        return mutated
    
    def _inversion_mutate(self, matrix: np.ndarray) -> np.ndarray:
        """逆转变异"""
        mutated = matrix.copy()
        rows, cols = mutated.shape
        
        if random.random() < self.mutation_rate and cols > 1:
            # 随机选择一行并反转部分
            row = random.randint(0, rows - 1)
            start = random.randint(0, cols - 2)
            end = random.randint(start + 1, cols)
            mutated[row, start:end] = mutated[row, start:end][::-1]
        
        return mutated
    
    def _gaussian_mutate(self, matrix: np.ndarray) -> np.ndarray:
        """高斯变异"""
        mutated = matrix.copy()
        rows, cols = mutated.shape
        
        mask = np.random.random((rows, cols)) < self.mutation_rate
        noise = np.random.normal(0, self.gaussian_std, (rows, cols))
        mutated = mutated + mask * noise
        mutated = np.clip(mutated, 0, 1)
        
        return mutated


class AllianceGeneticOptimizer:
    """
    联盟遗传算法优化器
    
    使用遗传算法优化多智能体联盟结构，最大化协作效率。
    """
    
    def __init__(self,
                 population_size: int = 100,
                 max_generations: int = 200,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 elitism_count: int = 2,
                 selection_method: SelectionMethod = SelectionMethod.TOURNAMENT,
                 crossover_method: CrossoverMethod = CrossoverMethod.UNIFORM,
                 mutation_method: MutationMethod = MutationMethod.BIT_FLIP,
                 fitness_evaluator: Optional[FitnessEvaluator] = None,
                 convergence_threshold: float = 0.001,
                 patience: int = 20):
        
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        self.convergence_threshold = convergence_threshold
        self.patience = patience
        
        self.selection_op = SelectionOperator(selection_method, elitism_count=elitism_count)
        self.crossover_op = CrossoverOperator(crossover_method, crossover_rate)
        self.mutation_op = MutationOperator(mutation_method, mutation_rate)
        self.fitness_evaluator = fitness_evaluator or CollaborationEfficiencyEvaluator()
        
        self.population: List[AllianceChromosome] = []
        self.best_chromosome: Optional[AllianceChromosome] = None
        self.generation_stats: List[Dict[str, float]] = []
    
    def initialize_population(self, num_agents: int, num_tasks: int,
                              agent_ids: Optional[List[str]] = None,
                              task_ids: Optional[List[str]] = None) -> None:
        """初始化种群"""
        self.population = []
        
        for i in range(self.population_size):
            # 随机生成分配矩阵
            matrix = np.random.randint(0, 2, (num_agents, num_tasks)).astype(float)
            
            # 确保每个任务至少有一个智能体
            for j in range(num_tasks):
                if matrix[:, j].sum() == 0:
                    matrix[random.randint(0, num_agents - 1), j] = 1.0
            
            chromosome = AllianceChromosome(
                assignment_matrix=matrix,
                agent_ids=agent_ids or [f"agent_{k}" for k in range(num_agents)],
                task_ids=task_ids or [f"task_{k}" for k in range(num_tasks)],
                generation=0
            )
            self.population.append(chromosome)
    
    def evaluate_population(self, agents: List[AgentCapability],
                           tasks: List[TaskRequirement]) -> None:
        """评估种群适应度"""
        for chromosome in self.population:
            chromosome.fitness = self.fitness_evaluator.evaluate(chromosome, agents, tasks)
    
    def evolve_generation(self, agents: List[AgentCapability],
                         tasks: List[TaskRequirement],
                         generation: int) -> None:
        """进化一代"""
        # 评估当前种群
        self.evaluate_population(agents, tasks)
        
        # 记录统计信息
        fitnesses = [c.fitness for c in self.population]
        stats = {
            'generation': generation,
            'best_fitness': max(fitnesses),
            'avg_fitness': sum(fitnesses) / len(fitnesses),
            'worst_fitness': min(fitnesses),
            'std_fitness': np.std(fitnesses)
        }
        self.generation_stats.append(stats)
        
        # 更新最优个体
        current_best = max(self.population, key=lambda x: x.fitness)
        if self.best_chromosome is None or current_best.fitness > self.best_chromosome.fitness:
            self.best_chromosome = current_best.copy()
        
        # 精英保留
        elites = self.selection_op.get_elites(self.population)
        
        # 选择
        selected = self.selection_op.select(self.population, 
                                           self.population_size - self.elitism_count)
        
        # 交叉
        offspring = []
        for i in range(0, len(selected) - 1, 2):
            child1, child2 = self.crossover_op.crossover(selected[i], selected[i + 1])
            offspring.extend([child1, child2])
        
        if len(selected) % 2 == 1:
            offspring.append(selected[-1].copy())
        
        # 变异
        offspring = [self.mutation_op.mutate(child) for child in offspring]
        
        # 合并精英和后代
        self.population = elites + offspring[:self.population_size - self.elitism_count]
        
        # 更新代数
        for chromosome in self.population:
            chromosome.generation = generation
    
    def optimize(self, agents: List[AgentCapability],
                tasks: List[TaskRequirement],
                callback: Optional[Callable[[int, AllianceChromosome], None]] = None
                ) -> AllianceChromosome:
        """
        执行优化
        
        Args:
            agents: 智能体列表
            tasks: 任务列表
            callback: 每代回调函数 (generation, best_chromosome)
        
        Returns:
            最优染色体
        """
        if not self.population:
            self.initialize_population(len(agents), len(tasks),
                                      [a.agent_id for a in agents],
                                      [t.task_id for t in tasks])
        
        best_fitness_history = []
        no_improvement_count = 0
        
        for generation in range(self.max_generations):
            self.evolve_generation(agents, tasks, generation)
            
            if callback:
                callback(generation, self.best_chromosome)
            
            # 检查收敛
            if self.best_chromosome:
                best_fitness_history.append(self.best_chromosome.fitness)
                
                if len(best_fitness_history) > self.patience:
                    recent_improvement = (best_fitness_history[-1] - 
                                        best_fitness_history[-self.patience])
                    if recent_improvement < self.convergence_threshold:
                        no_improvement_count += 1
                        if no_improvement_count >= 5:
                            break
                    else:
                        no_improvement_count = 0
        
        return self.best_chromosome
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """获取优化报告"""
        return {
            'generations': len(self.generation_stats),
            'final_best_fitness': self.best_chromosome.fitness if self.best_chromosome else 0,
            'generation_stats': self.generation_stats,
            'best_solution': {
                'assignment_matrix': self.best_chromosome.assignment_matrix.tolist() 
                                   if self.best_chromosome else None,
                'agent_ids': self.best_chromosome.agent_ids if self.best_chromosome else None,
                'task_ids': self.best_chromosome.task_ids if self.best_chromosome else None
            }
        }


# 便捷函数
def optimize_alliance(agents: List[AgentCapability],
                     tasks: List[TaskRequirement],
                     population_size: int = 100,
                     max_generations: int = 200,
                     **kwargs) -> Tuple[AllianceChromosome, Dict[str, Any]]:
    """
    便捷函数：优化联盟结构
    
    Args:
        agents: 智能体列表
        tasks: 任务列表
        population_size: 种群大小
        max_generations: 最大迭代次数
        **kwargs: 其他优化器参数
    
    Returns:
        (最优染色体, 优化报告)
    """
    optimizer = AllianceGeneticOptimizer(
        population_size=population_size,
        max_generations=max_generations,
        **kwargs
    )
    
    best = optimizer.optimize(agents, tasks)
    report = optimizer.get_optimization_report()
    
    return best, report
