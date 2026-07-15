"""
AGI统一框架 - 进化策略与自然进化策略
实现CMA-ES、OpenAI-ES、Natural ES等进化优化算法
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Callable
from dataclasses import dataclass
import math


# ==================== 配置 ====================

@dataclass
class ESConfig:
    """进化策略配置"""
    population_size: int = 50
    num_generations: int = 100
    sigma: float = 0.1  # 初始标准差
    learning_rate: float = 0.01
    sigma_decay: float = 0.95
    sigma_min: float = 0.01
    elite_ratio: float = 0.2  # 精英比例


# ==================== 基础进化策略 ====================

class EvolutionStrategy:
    """基础进化策略"""
    
    def __init__(self, objective_fn: Callable, dim: int,
                 config: Optional[ESConfig] = None):
        self.objective_fn = objective_fn
        self.dim = dim
        self.config = config or ESConfig()
        
        # 初始化参数
        self.theta = np.zeros(dim)
        self.sigma = self.config.sigma
        
        # 历史记录
        self.history: List[Dict] = []
        
    def optimize(self, callback: Optional[Callable] = None) -> Dict[str, Any]:
        """执行优化"""
        best_fitness = float('-inf')
        best_solution = None
        
        for gen in range(self.config.num_generations):
            # 采样
            solutions = self._sample_population()
            
            # 评估
            fitness = np.array([self.objective_fn(s) for s in solutions])
            
            # 更新
            self._update(fitness, solutions)
            
            # 记录
            gen_best_idx = np.argmax(fitness)
            if fitness[gen_best_idx] > best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_solution = solutions[gen_best_idx].copy()
            
            self.history.append({
                'generation': gen,
                'mean_fitness': fitness.mean(),
                'best_fitness': fitness.max(),
                'sigma': self.sigma
            })
            
            if callback:
                callback(gen, self.history[-1])
            
            # 衰减sigma
            self.sigma = max(self.sigma * self.config.sigma_decay, self.config.sigma_min)
        
        return {
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'history': self.history
        }
    
    def _sample_population(self) -> np.ndarray:
        """采样种群"""
        noise = np.random.randn(self.config.population_size, self.dim)
        return self.theta + self.sigma * noise
    
    def _update(self, fitness: np.ndarray, solutions: np.ndarray):
        """更新参数"""
        # 简单梯度估计
        ranks = np.argsort(fitness)[::-1]
        weights = np.zeros(self.config.population_size)
        weights[ranks[:int(self.config.elite_ratio * self.config.population_size)]] = 1.0
        weights = weights / weights.sum()
        
        # 更新中心
        self.theta = np.sum(weights[:, None] * solutions, axis=0)


# ==================== CMA-ES ====================

class CMAES:
    """协方差矩阵自适应进化策略"""
    
    def __init__(self, objective_fn: Callable, dim: int,
                 population_size: Optional[int] = None,
                 initial_sigma: float = 0.5):
        self.objective_fn = objective_fn
        self.dim = dim
        
        # 种群大小
        self.population_size = population_size or 4 + int(3 * math.log(dim))
        self.mu = self.population_size // 2  # 父代数量
        
        # 权重
        weights_prime = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights_prime / weights_prime.sum()
        
        # 方差有效选择质量
        self.mueff = 1.0 / (self.weights ** 2).sum()
        
        # 步长控制参数
        self.cs = (self.mueff + 2) / (dim + self.mueff + 5)
        self.ds = 1 + 2 * max(0, math.sqrt((self.mueff - 1) / (dim + 1)) - 1) + self.cs
        
        # 协方差矩阵自适应参数
        self.cc = (4 + self.mueff / dim) / (dim + 4 + 2 * self.mueff / dim)
        self.c1 = 2 / ((dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / 
                       ((dim + 2) ** 2 + self.mueff))
        
        # 初始化
        self.mean = np.zeros(dim)
        self.sigma = initial_sigma
        self.C = np.eye(dim)  # 协方差矩阵
        self.B = np.eye(dim)  # 特征向量
        self.D = np.ones(dim)  # 特征值的平方根
        
        # 进化路径
        self.ps = np.zeros(dim)
        self.pc = np.zeros(dim)
        
        # 代数
        self.generation = 0
        self.eigeneval = 0
        
        # 历史记录
        self.history: List[Dict] = []
        
    def optimize(self, max_generations: int = 1000,
                 target_fitness: Optional[float] = None,
                 callback: Optional[Callable] = None) -> Dict[str, Any]:
        """执行优化"""
        best_fitness = float('-inf')
        best_solution = None
        
        for gen in range(max_generations):
            # 采样
            solutions, noises = self._sample_population()
            
            # 评估
            fitness = np.array([self.objective_fn(s) for s in solutions])
            
            # 更新
            self._update(fitness, solutions, noises)
            
            # 记录
            gen_best_idx = np.argmax(fitness)
            if fitness[gen_best_idx] > best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_solution = solutions[gen_best_idx].copy()
            
            self.history.append({
                'generation': gen,
                'mean_fitness': fitness.mean(),
                'best_fitness': fitness.max(),
                'sigma': self.sigma
            })
            
            if callback:
                callback(gen, self.history[-1])
            
            # 检查收敛
            if target_fitness is not None and best_fitness >= target_fitness:
                break
        
        return {
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'history': self.history
        }
    
    def _sample_population(self) -> Tuple[np.ndarray, np.ndarray]:
        """采样种群"""
        # 更新特征分解
        if self.generation - self.eigeneval > self.population_size / 10:
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            D2, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(np.maximum(D2, 1e-10))
            self.eigeneval = self.generation
        
        # 采样
        noises = np.random.randn(self.population_size, self.dim)
        solutions = self.mean + self.sigma * (noises @ (self.B * self.D).T)
        
        return solutions, noises
    
    def _update(self, fitness: np.ndarray, solutions: np.ndarray, noises: np.ndarray):
        """更新参数"""
        # 排序
        idx = np.argsort(fitness)[::-1]
        
        # 选择父代
        selected_noises = noises[idx[:self.mu]]
        selected_solutions = solutions[idx[:self.mu]]
        
        # 更新均值
        old_mean = self.mean.copy()
        self.mean = np.sum(self.weights[:, None] * selected_solutions, axis=0)
        
        # 更新进化路径
        ymean = (self.mean - old_mean) / self.sigma
        
        # ps更新
        Cinv = (self.B / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + \
                  math.sqrt(self.cs * (2 - self.cs) * self.mueff) * (Cinv @ ymean)
        
        # pc更新
        hsig = np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * self.generation + 2)) \
               < 1.4 + 2 / (self.dim + 1)
        self.pc = (1 - self.cc) * self.pc + \
                  hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * ymean
        
        # 更新协方差矩阵
        artmp = selected_noises - ymean / self.sigma
        self.C = (1 - self.c1 - self.cmu) * self.C + \
                 self.c1 * (np.outer(self.pc, self.pc) + 
                           (1 - hsig) * self.cc * (2 - self.cc) * self.C) + \
                 self.cmu * np.sum(self.weights[:, None, None] * 
                                  np.array([np.outer(artmp[i], artmp[i]) 
                                           for i in range(self.mu)]), axis=0)
        
        # 更新步长
        self.sigma = self.sigma * math.exp((self.cs / self.ds) * 
                                           (np.linalg.norm(self.ps) / math.sqrt(self.dim) - 1))
        
        self.generation += 1


# ==================== OpenAI进化策略 ====================

class OpenAI_ES:
    """OpenAI进化策略"""
    
    def __init__(self, objective_fn: Callable, dim: int,
                 population_size: int = 50,
                 sigma: float = 0.1,
                 learning_rate: float = 0.01):
        self.objective_fn = objective_fn
        self.dim = dim
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        
        # 参数
        self.theta = np.zeros(dim)
        
        # Adam参数
        self.m = np.zeros(dim)
        self.v = np.zeros(dim)
        self.t = 0
        
        # 历史记录
        self.history: List[Dict] = []
        
    def optimize(self, num_iterations: int = 1000,
                 callback: Optional[Callable] = None) -> Dict[str, Any]:
        """执行优化"""
        best_fitness = float('-inf')
        best_solution = None
        
        for iteration in range(num_iterations):
            # 采样噪声
            noise = np.random.randn(self.population_size, self.dim)
            
            # 扰动解
            solutions_pos = self.theta + self.sigma * noise
            solutions_neg = self.theta - self.sigma * noise
            
            # 评估
            fitness_pos = np.array([self.objective_fn(s) for s in solutions_pos])
            fitness_neg = np.array([self.objective_fn(s) for s in solutions_neg])
            
            # 梯度估计
            fitness_combined = fitness_pos - fitness_neg
            grad = (fitness_combined @ noise) / (self.population_size * self.sigma)
            
            # Adam更新
            self.t += 1
            self.m = 0.9 * self.m + 0.1 * grad
            self.v = 0.999 * self.v + 0.001 * grad ** 2
            
            m_hat = self.m / (1 - 0.9 ** self.t)
            v_hat = self.v / (1 - 0.999 ** self.t)
            
            self.theta = self.theta + self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            
            # 记录
            current_fitness = self.objective_fn(self.theta)
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_solution = self.theta.copy()
            
            self.history.append({
                'iteration': iteration,
                'fitness': current_fitness,
                'best_fitness': best_fitness
            })
            
            if callback:
                callback(iteration, self.history[-1])
        
        return {
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'history': self.history
        }


# ==================== 自然进化策略 ====================

class NaturalES:
    """自然进化策略"""
    
    def __init__(self, objective_fn: Callable, dim: int,
                 population_size: int = 50,
                 sigma: float = 0.1,
                 learning_rate: float = 0.01):
        self.objective_fn = objective_fn
        self.dim = dim
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        
        # 参数
        self.theta = np.zeros(dim)
        
        # Fisher信息矩阵估计的衰减
        self.fisher_decay = 0.99
        self.fisher_inv = np.eye(dim)
        
        # 历史记录
        self.history: List[Dict] = []
        
    def optimize(self, num_iterations: int = 1000,
                 callback: Optional[Callable] = None) -> Dict[str, Any]:
        """执行优化"""
        best_fitness = float('-inf')
        best_solution = None
        
        for iteration in range(num_iterations):
            # 采样
            noise = np.random.randn(self.population_size, self.dim)
            solutions = self.theta + self.sigma * noise
            
            # 评估
            fitness = np.array([self.objective_fn(s) for s in solutions])
            
            # 标准化适应度
            fitness = (fitness - fitness.mean()) / (fitness.std() + 1e-8)
            
            # 梯度估计
            grad = (fitness @ noise) / (self.population_size * self.sigma)
            
            # Fisher信息矩阵估计
            fisher = np.zeros((self.dim, self.dim))
            for i in range(self.population_size):
                fisher += np.outer(noise[i], noise[i])
            fisher = fisher / (self.population_size * self.sigma ** 2)
            
            # 更新Fisher逆
            self.fisher_inv = self.fisher_decay * self.fisher_inv + \
                             (1 - self.fisher_decay) * np.linalg.inv(fisher + 1e-6 * np.eye(self.dim))
            
            # 自然梯度
            natural_grad = self.fisher_inv @ grad
            
            # 更新参数
            self.theta = self.theta + self.learning_rate * natural_grad
            
            # 记录
            current_fitness = self.objective_fn(self.theta)
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_solution = self.theta.copy()
            
            self.history.append({
                'iteration': iteration,
                'fitness': current_fitness,
                'best_fitness': best_fitness
            })
            
            if callback:
                callback(iteration, self.history[-1])
        
        return {
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'history': self.history
        }


# ==================== 差分进化 ====================

class DifferentialEvolution:
    """差分进化算法"""
    
    def __init__(self, objective_fn: Callable, bounds: List[Tuple[float, float]],
                 population_size: int = 50,
                 mutation_factor: float = 0.8,
                 crossover_prob: float = 0.9,
                 strategy: str = "DE/rand/1"):
        self.objective_fn = objective_fn
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.population_size = population_size
        self.F = mutation_factor
        self.CR = crossover_prob
        self.strategy = strategy
        
        # 初始化种群
        self.population = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1],
            size=(population_size, self.dim)
        )
        
        # 评估初始种群
        self.fitness = np.array([objective_fn(ind) for ind in self.population])
        
        # 历史记录
        self.history: List[Dict] = []
        
    def optimize(self, max_generations: int = 1000,
                 callback: Optional[Callable] = None) -> Dict[str, Any]:
        """执行优化"""
        best_idx = np.argmax(self.fitness)
        best_fitness = self.fitness[best_idx]
        best_solution = self.population[best_idx].copy()
        
        for gen in range(max_generations):
            for i in range(self.population_size):
                # 变异
                mutant = self._mutate(i)
                
                # 交叉
                trial = self._crossover(self.population[i], mutant)
                
                # 选择
                trial_fitness = self.objective_fn(trial)
                
                if trial_fitness > self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    
                    if trial_fitness > best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial.copy()
            
            self.history.append({
                'generation': gen,
                'mean_fitness': self.fitness.mean(),
                'best_fitness': best_fitness
            })
            
            if callback:
                callback(gen, self.history[-1])
        
        return {
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'history': self.history
        }
    
    def _mutate(self, target_idx: int) -> np.ndarray:
        """变异操作"""
        # 选择不同的个体
        candidates = [i for i in range(self.population_size) if i != target_idx]
        selected = np.random.choice(candidates, size=3, replace=False)
        
        if self.strategy == "DE/rand/1":
            mutant = self.population[selected[0]] + \
                    self.F * (self.population[selected[1]] - self.population[selected[2]])
        elif self.strategy == "DE/best/1":
            best_idx = np.argmax(self.fitness)
            mutant = self.population[best_idx] + \
                    self.F * (self.population[selected[0]] - self.population[selected[1]])
        else:
            mutant = self.population[selected[0]] + \
                    self.F * (self.population[selected[1]] - self.population[selected[2]])
        
        # 裁剪到边界
        return np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])
    
    def _crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """交叉操作"""
        trial = target.copy()
        
        # 确保至少一个维度来自变异个体
        j_rand = np.random.randint(self.dim)
        
        for j in range(self.dim):
            if np.random.random() < self.CR or j == j_rand:
                trial[j] = mutant[j]
                
        return trial


# ==================== 粒子群优化 ====================

class ParticleSwarmOptimization:
    """粒子群优化"""
    
    def __init__(self, objective_fn: Callable, bounds: List[Tuple[float, float]],
                 num_particles: int = 30,
                 inertia: float = 0.7,
                 cognitive: float = 1.5,
                 social: float = 1.5):
        self.objective_fn = objective_fn
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.num_particles = num_particles
        self.w = inertia
        self.c1 = cognitive
        self.c2 = social
        
        # 初始化粒子
        self.positions = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1],
            size=(num_particles, self.dim)
        )
        
        self.velocities = np.random.uniform(
            -(self.bounds[:, 1] - self.bounds[:, 0]),
            (self.bounds[:, 1] - self.bounds[:, 0]),
            size=(num_particles, self.dim)
        )
        
        # 评估
        self.fitness = np.array([objective_fn(p) for p in self.positions])
        
        # 个体最优
        self.pbest_positions = self.positions.copy()
        self.pbest_fitness = self.fitness.copy()
        
        # 全局最优
        self.gbest_idx = np.argmax(self.fitness)
        self.gbest_position = self.positions[self.gbest_idx].copy()
        self.gbest_fitness = self.fitness[self.gbest_idx]
        
        # 历史记录
        self.history: List[Dict] = []
        
    def optimize(self, max_iterations: int = 1000,
                 callback: Optional[Callable] = None) -> Dict[str, Any]:
        """执行优化"""
        for iteration in range(max_iterations):
            for i in range(self.num_particles):
                # 更新速度
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                
                self.velocities[i] = (
                    self.w * self.velocities[i] +
                    self.c1 * r1 * (self.pbest_positions[i] - self.positions[i]) +
                    self.c2 * r2 * (self.gbest_position - self.positions[i])
                )
                
                # 更新位置
                self.positions[i] = self.positions[i] + self.velocities[i]
                
                # 裁剪
                self.positions[i] = np.clip(
                    self.positions[i], self.bounds[:, 0], self.bounds[:, 1]
                )
                
                # 评估
                self.fitness[i] = self.objective_fn(self.positions[i])
                
                # 更新个体最优
                if self.fitness[i] > self.pbest_fitness[i]:
                    self.pbest_positions[i] = self.positions[i].copy()
                    self.pbest_fitness[i] = self.fitness[i]
                    
                    # 更新全局最优
                    if self.fitness[i] > self.gbest_fitness:
                        self.gbest_position = self.positions[i].copy()
                        self.gbest_fitness = self.fitness[i]
            
            self.history.append({
                'iteration': iteration,
                'mean_fitness': self.fitness.mean(),
                'best_fitness': self.gbest_fitness
            })
            
            if callback:
                callback(iteration, self.history[-1])
        
        return {
            'best_solution': self.gbest_position,
            'best_fitness': self.gbest_fitness,
            'history': self.history
        }
