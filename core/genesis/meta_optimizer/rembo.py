"""
AGI统一框架 - REMBO元优化器
Random Embedding Bayesian Optimization for High-Dimensional Hyperparameter Optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
import math
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 尝试导入scipy和sklearn
try:
    from scipy.stats import norm
    from scipy.linalg import qr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
    SKLEARN_GP_AVAILABLE = True
except ImportError:
    SKLEARN_GP_AVAILABLE = False


# ==================== 配置类 ====================

@dataclass
class REMBOConfig:
    """REMBO配置"""
    # 原始维度和嵌入维度
    original_dim: int = 100
    embedding_dim: int = 10  # 通常远小于original_dim
    
    # 搜索空间
    lower_bound: float = -1.0
    upper_bound: float = 1.0
    
    # 高斯过程配置
    gp_kernel: str = "matern"  # matern, rbf
    gp_nu: float = 2.5  # Matern核的平滑参数
    gp_noise: float = 1e-6
    
    # 采集函数配置
    acquisition_type: str = "ei"  # ei, ucb, pi
    ucb_beta: float = 2.0
    xi: float = 0.01  # EI的探索参数
    
    # 优化配置
    n_initial: int = 10  # 初始随机采样数
    n_iterations: int = 100
    n_restarts: int = 10  # 采集函数优化的重启次数
    
    # 随机嵌入配置
    embedding_type: str = "gaussian"  # gaussian, sparse, orthogonal
    sparse_ratio: float = 0.1
    
    # 元学习配置
    use_meta_learning: bool = True
    meta_history_size: int = 100


# ==================== 随机嵌入 ====================

class RandomEmbedding:
    """随机嵌入矩阵"""
    
    def __init__(self, original_dim: int, embedding_dim: int,
                 embedding_type: str = "gaussian",
                 sparse_ratio: float = 0.1):
        self.original_dim = original_dim
        self.embedding_dim = embedding_dim
        self.embedding_type = embedding_type
        
        # 生成嵌入矩阵
        if embedding_type == "gaussian":
            self.matrix = np.random.randn(embedding_dim, original_dim)
            # 归一化
            self.matrix = self.matrix / np.linalg.norm(self.matrix, axis=1, keepdims=True)
            
        elif embedding_type == "sparse":
            # 稀疏随机嵌入
            self.matrix = np.zeros((embedding_dim, original_dim))
            for i in range(embedding_dim):
                active_indices = np.random.choice(
                    original_dim, 
                    size=int(original_dim * sparse_ratio),
                    replace=False
                )
                self.matrix[i, active_indices] = np.random.randn(len(active_indices))
                
        elif embedding_type == "orthogonal":
            # 正交嵌入
            if SCIPY_AVAILABLE:
                random_matrix = np.random.randn(original_dim, embedding_dim)
                self.matrix, _ = qr(random_matrix, mode='economic')
                self.matrix = self.matrix.T  # (embedding_dim, original_dim)
            else:
                # 回退到高斯嵌入
                self.matrix = np.random.randn(embedding_dim, original_dim)
                self.matrix = self.matrix / np.linalg.norm(self.matrix, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
            
    def embed(self, x: np.ndarray) -> np.ndarray:
        """将低维点嵌入到高维空间: z = A^T @ y"""
        if x.ndim == 1:
            return self.matrix.T @ x
        else:
            return x @ self.matrix  # (n, embedding_dim) @ (embedding_dim, original_dim)
            
    def project(self, z: np.ndarray) -> np.ndarray:
        """将高维点投影到低维空间: y = A @ z"""
        if z.ndim == 1:
            return self.matrix @ z
        else:
            return z @ self.matrix.T  # (n, original_dim) @ (original_dim, embedding_dim)
            
    def clip_to_embedding(self, y: np.ndarray, 
                          lower: float = -1.0, upper: float = 1.0) -> np.ndarray:
        """将嵌入空间的点裁剪到有效区域"""
        z = self.embed(y)
        z_clipped = np.clip(z, lower, upper)
        # 投影回嵌入空间
        y_clipped = self.project(z_clipped)
        return y_clipped


# ==================== 高斯过程 ====================

class SimpleGaussianProcess:
    """简单高斯过程实现（不依赖sklearn）"""
    
    def __init__(self, kernel: str = "matern", nu: float = 2.5, 
                 length_scale: float = 1.0, noise: float = 1e-6):
        self.kernel_type = kernel
        self.nu = nu
        self.length_scale = length_scale
        self.noise = noise
        
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        self.alpha = None
        
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF核"""
        if X1.ndim == 1:
            X1 = X1.reshape(1, -1)
        if X2.ndim == 1:
            X2 = X2.reshape(1, -1)
            
        dist_sq = np.sum(X1**2, axis=1, keepdims=True) + \
                  np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        return np.exp(-0.5 * dist_sq / self.length_scale**2)
    
    def _matern_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Matern核"""
        if X1.ndim == 1:
            X1 = X1.reshape(1, -1)
        if X2.ndim == 1:
            X2 = X2.reshape(1, -1)
            
        dist = np.sqrt(np.sum(X1**2, axis=1, keepdims=True) + 
                       np.sum(X2**2, axis=1) - 2 * X1 @ X2.T + 1e-12)
        
        scaled_dist = dist / self.length_scale
        
        if self.nu == 0.5:
            return np.exp(-scaled_dist)
        elif self.nu == 1.5:
            return (1 + np.sqrt(3) * scaled_dist) * np.exp(-np.sqrt(3) * scaled_dist)
        elif self.nu == 2.5:
            return (1 + np.sqrt(5) * scaled_dist + 5/3 * scaled_dist**2) * \
                   np.exp(-np.sqrt(5) * scaled_dist)
        else:
            # 回退到RBF
            return np.exp(-0.5 * scaled_dist**2)
            
    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """计算核矩阵"""
        if self.kernel_type == "rbf":
            return self._rbf_kernel(X1, X2)
        else:
            return self._matern_kernel(X1, X2)
            
    def fit(self, X: np.ndarray, y: np.ndarray):
        """拟合高斯过程"""
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        # 计算核矩阵
        K = self._compute_kernel(X, X)
        K += self.noise * np.eye(len(X))
        
        # Cholesky分解
        try:
            L = np.linalg.cholesky(K)
            self.alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            self.K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(len(X))))
        except np.linalg.LinAlgError:
            # 如果Cholesky失败，使用伪逆
            self.alpha = np.linalg.lstsq(K, y, rcond=None)[0]
            self.K_inv = np.linalg.pinv(K)
            
    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """预测"""
        if self.X_train is None:
            mean = np.zeros(len(X))
            std = np.ones(len(X)) if return_std else None
            return mean, std
            
        K_star = self._compute_kernel(X, self.X_train)
        mean = K_star @ self.alpha
        
        if return_std:
            K_star_star = self._compute_kernel(X, X)
            var = np.diag(K_star_star - K_star @ self.K_inv @ K_star.T)
            var = np.maximum(var, 1e-10)  # 确保非负
            std = np.sqrt(var)
            return mean, std
        else:
            return mean, None


# ==================== 采集函数 ====================

class AcquisitionFunction:
    """采集函数"""
    
    def __init__(self, gp: SimpleGaussianProcess, 
                 acquisition_type: str = "ei",
                 y_best: float = 0.0,
                 beta: float = 2.0,
                 xi: float = 0.01):
        self.gp = gp
        self.acquisition_type = acquisition_type
        self.y_best = y_best
        self.beta = beta
        self.xi = xi
        
    def expected_improvement(self, X: np.ndarray) -> np.ndarray:
        """期望提升 (EI)"""
        mu, sigma = self.gp.predict(X, return_std=True)
        
        improvement = self.y_best - mu - self.xi
        z = improvement / (sigma + 1e-10)
        
        if SCIPY_AVAILABLE:
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        else:
            # 近似计算
            ei = improvement * self._approx_cdf(z) + sigma * self._approx_pdf(z)
            
        ei[sigma < 1e-10] = 0.0
        return ei
    
    def _approx_cdf(self, x: np.ndarray) -> np.ndarray:
        """近似标准正态CDF"""
        return 0.5 * (1 + np.tanh(x * 0.7978845608))
    
    def _approx_pdf(self, x: np.ndarray) -> np.ndarray:
        """近似标准正态PDF"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
        
    def upper_confidence_bound(self, X: np.ndarray) -> np.ndarray:
        """上置信界 (UCB)"""
        mu, sigma = self.gp.predict(X, return_std=True)
        return mu + self.beta * sigma
    
    def probability_of_improvement(self, X: np.ndarray) -> np.ndarray:
        """提升概率 (PI)"""
        mu, sigma = self.gp.predict(X, return_std=True)
        
        z = (self.y_best - mu - self.xi) / (sigma + 1e-10)
        
        if SCIPY_AVAILABLE:
            return norm.cdf(z)
        else:
            return self._approx_cdf(z)
            
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """评估采集函数"""
        if self.acquisition_type == "ei":
            return self.expected_improvement(X)
        elif self.acquisition_type == "ucb":
            return self.upper_confidence_bound(X)
        elif self.acquisition_type == "pi":
            return self.probability_of_improvement(X)
        else:
            raise ValueError(f"Unknown acquisition type: {self.acquisition_type}")


# ==================== REMBO优化器 ====================

class REMBOOptimizer:
    """REMBO优化器主类"""
    
    def __init__(self, objective_fn: Callable,
                 config: Optional[REMBOConfig] = None):
        self.config = config or REMBOConfig()
        self.objective_fn = objective_fn
        
        # 创建随机嵌入
        self.embedding = RandomEmbedding(
            self.config.original_dim,
            self.config.embedding_dim,
            self.config.embedding_type,
            self.config.sparse_ratio
        )
        
        # 高斯过程
        self.gp = SimpleGaussianProcess(
            kernel=self.config.gp_kernel,
            nu=self.config.gp_nu,
            noise=self.config.gp_noise
        )
        
        # 历史记录
        self.X_observed: List[np.ndarray] = []
        self.y_observed: List[float] = []
        self.best_value = float('inf')
        self.best_point: Optional[np.ndarray] = None
        
        # 元学习历史
        self.meta_history: List[Dict] = []
        
    def _initialize(self) -> None:
        """初始化采样"""
        for _ in range(self.config.n_initial):
            # 在嵌入空间随机采样
            y = np.random.uniform(
                self.config.lower_bound,
                self.config.upper_bound,
                size=self.config.embedding_dim
            )
            self._evaluate_and_update(y)
            
    def _evaluate_and_update(self, y: np.ndarray) -> float:
        """评估并更新"""
        # 嵌入到高维空间
        z = self.embedding.embed(y)
        z = np.clip(z, self.config.lower_bound, self.config.upper_bound)
        
        # 评估目标函数
        value = self.objective_fn(z)
        
        # 更新历史
        self.X_observed.append(y.copy())
        self.y_observed.append(value)
        
        # 更新最优
        if value < self.best_value:
            self.best_value = value
            self.best_point = z.copy()
            
        # 元学习历史
        if self.config.use_meta_learning:
            self.meta_history.append({
                'y': y.copy(),
                'z': z.copy(),
                'value': value,
                'best_value': self.best_value
            })
            if len(self.meta_history) > self.config.meta_history_size:
                self.meta_history.pop(0)
                
        return value
    
    def _optimize_acquisition(self, acquisition: AcquisitionFunction) -> np.ndarray:
        """优化采集函数"""
        best_y = None
        best_acq = float('-inf')
        
        for _ in range(self.config.n_restarts):
            # 随机初始点
            y0 = np.random.uniform(
                self.config.lower_bound,
                self.config.upper_bound,
                size=self.config.embedding_dim
            )
            
            # 简单的梯度上升
            y = y0.copy()
            lr = 0.01
            
            for _ in range(100):
                # 数值梯度
                grad = np.zeros(self.config.embedding_dim)
                eps = 1e-5
                
                for i in range(self.config.embedding_dim):
                    y_plus = y.copy()
                    y_plus[i] += eps
                    y_minus = y.copy()
                    y_minus[i] -= eps
                    
                    grad[i] = (acquisition.evaluate(y_plus.reshape(1, -1))[0] - 
                              acquisition.evaluate(y_minus.reshape(1, -1))[0]) / (2 * eps)
                
                # 更新
                y = y + lr * grad
                
                # 裁剪
                y = np.clip(y, self.config.lower_bound, self.config.upper_bound)
            
            # 评估
            acq_value = acquisition.evaluate(y.reshape(1, -1))[0]
            
            if acq_value > best_acq:
                best_acq = acq_value
                best_y = y.copy()
                
        return best_y
    
    def step(self) -> Dict[str, Any]:
        """执行一步优化"""
        # 拟合高斯过程
        if len(self.X_observed) > 0:
            X = np.array(self.X_observed)
            y = np.array(self.y_observed)
            self.gp.fit(X, y)
        
        # 创建采集函数
        y_best = min(self.y_observed) if self.y_observed else 0.0
        acquisition = AcquisitionFunction(
            self.gp,
            acquisition_type=self.config.acquisition_type,
            y_best=y_best,
            beta=self.config.ucb_beta,
            xi=self.config.xi
        )
        
        # 优化采集函数
        y_next = self._optimize_acquisition(acquisition)
        
        # 评估新点
        value = self._evaluate_and_update(y_next)
        
        return {
            'y': y_next,
            'value': value,
            'best_value': self.best_value,
            'n_evaluations': len(self.y_observed)
        }
    
    def optimize(self, n_iterations: Optional[int] = None,
                 callback: Optional[Callable] = None) -> Dict[str, Any]:
        """执行完整优化"""
        n_iterations = n_iterations or self.config.n_iterations
        
        # 初始化
        if len(self.X_observed) == 0:
            self._initialize()
        
        # 迭代优化
        for i in range(n_iterations):
            result = self.step()
            
            if callback:
                callback(i, result)
                
        return {
            'best_value': self.best_value,
            'best_point': self.best_point,
            'n_evaluations': len(self.y_observed),
            'all_values': self.y_observed.copy()
        }
    
    def suggest_next(self) -> np.ndarray:
        """建议下一个评估点"""
        if len(self.X_observed) == 0:
            return np.random.uniform(
                self.config.lower_bound,
                self.config.upper_bound,
                size=self.config.embedding_dim
            )
        
        # 拟合GP
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        self.gp.fit(X, y)
        
        # 创建采集函数
        y_best = min(self.y_observed)
        acquisition = AcquisitionFunction(
            self.gp,
            acquisition_type=self.config.acquisition_type,
            y_best=y_best
        )
        
        return self._optimize_acquisition(acquisition)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if len(self.y_observed) == 0:
            return {'n_evaluations': 0}
            
        return {
            'n_evaluations': len(self.y_observed),
            'best_value': self.best_value,
            'mean_value': np.mean(self.y_observed),
            'std_value': np.std(self.y_observed),
            'improvement': self.y_observed[0] - self.best_value if len(self.y_observed) > 0 else 0
        }


# ==================== 元学习优化器 ====================

class MetaOptimizer:
    """元学习优化器 - 跨任务学习优化策略"""
    
    def __init__(self, embedding_dim: int = 10, 
                 hidden_dim: int = 64,
                 learning_rate: float = 0.01):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # LSTM策略网络
        self.policy_network = nn.Sequential(
            nn.Linear(embedding_dim + 1, hidden_dim),  # y + value
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Tanh()
        )
        
        # 价值网络
        self.value_network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.optimizer = torch.optim.Adam(
            list(self.policy_network.parameters()) + 
            list(self.value_network.parameters()),
            lr=learning_rate
        )
        
        # 任务历史
        self.task_histories: List[List[Dict]] = []
        
    def suggest_from_meta(self, current_y: np.ndarray,
                          current_value: float) -> np.ndarray:
        """基于元学习建议下一个点"""
        # 编码当前状态
        state = torch.tensor(
            np.concatenate([current_y, [current_value]]),
            dtype=torch.float32
        ).unsqueeze(0)
        
        # 策略网络建议
        with torch.no_grad():
            suggestion = self.policy_network(state).squeeze(0).numpy()
            
        return suggestion
    
    def update_meta(self, trajectory: List[Dict]):
        """更新元学习模型"""
        if len(trajectory) < 2:
            return
            
        # 计算优势
        states = []
        advantages = []
        
        for i, step in enumerate(trajectory):
            y = step['y']
            value = step['value']
            
            # 计算未来奖励
            future_rewards = [-s['value'] for s in trajectory[i:]]
            advantage = sum(future_rewards) - value
            
            states.append(torch.tensor(
                np.concatenate([y, [value]]),
                dtype=torch.float32
            ))
            advantages.append(advantage)
            
        # 训练
        states = torch.stack(states)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        
        # 策略梯度更新
        self.optimizer.zero_grad()
        
        predictions = self.policy_network(states)
        values = self.value_network(states[:, :-1])
        
        # 策略损失
        policy_loss = -torch.mean(
            advantages * torch.sum(predictions * states[:, :-1], dim=1)
        )
        
        # 价值损失
        value_loss = F.mse_loss(values.squeeze(), advantages)
        
        total_loss = policy_loss + 0.5 * value_loss
        total_loss.backward()
        self.optimizer.step()
        
        # 保存任务历史
        self.task_histories.append(trajectory)
        if len(self.task_histories) > 100:
            self.task_histories.pop(0)


# ==================== 多保真度优化 ====================

class MultiFidelityOptimizer:
    """多保真度优化器"""
    
    def __init__(self, objective_fns: Dict[str, Callable],
                 config: Optional[REMBOConfig] = None,
                 fidelity_costs: Optional[Dict[str, float]] = None):
        """
        objective_fns: 不同保真度的目标函数
        fidelity_costs: 各保真度的计算成本
        """
        self.objective_fns = objective_fns
        self.config = config or REMBOConfig()
        
        # 保真度级别
        self.fidelities = list(objective_fns.keys())
        self.fidelity_costs = fidelity_costs or {f: 1.0 for f in self.fidelities}
        
        # 每个保真度的优化器
        self.optimizers: Dict[str, REMBOOptimizer] = {}
        for fidelity, fn in objective_fns.items():
            self.optimizers[fidelity] = REMBOOptimizer(fn, config)
            
        # 跨保真度历史
        self.cross_fidelity_history: List[Dict] = []
        
    def _select_fidelity(self, y: np.ndarray) -> str:
        """选择保真度级别"""
        # 基于不确定性选择
        uncertainties = {}
        
        for fidelity, opt in self.optimizers.items():
            if len(opt.X_observed) > 0:
                X = np.array(opt.X_observed)
                mu, sigma = opt.gp.predict(y.reshape(1, -1), return_std=True)
                uncertainties[fidelity] = sigma[0]
            else:
                uncertainties[fidelity] = 1.0
                
        # 平衡探索和成本
        scores = {}
        for f in self.fidelities:
            scores[f] = uncertainties[f] / self.fidelity_costs[f]
            
        return max(scores, key=scores.get)
    
    def step(self) -> Dict[str, Any]:
        """执行一步多保真度优化"""
        # 选择保真度
        # 首先在低保真度探索
        if np.random.random() < 0.3:
            fidelity = self.fidelities[0]  # 低保真度
        else:
            fidelity = self.fidelities[-1]  # 高保真度
            
        # 使用对应优化器
        result = self.optimizers[fidelity].step()
        result['fidelity'] = fidelity
        
        # 跨保真度知识迁移
        self._transfer_knowledge(fidelity, result)
        
        self.cross_fidelity_history.append(result)
        
        return result
    
    def _transfer_knowledge(self, source_fidelity: str, result: Dict):
        """跨保真度知识迁移"""
        y = result['y']
        value = result['value']
        
        # 将结果传递给其他保真度
        for target_fidelity, opt in self.optimizers.items():
            if target_fidelity != source_fidelity:
                # 添加为伪观测（带噪声）
                noise_scale = 0.1 * abs(value)
                noisy_value = value + np.random.randn() * noise_scale
                
                opt.X_observed.append(y.copy())
                opt.y_observed.append(noisy_value)
                
    def optimize(self, total_budget: float) -> Dict[str, Any]:
        """在预算约束下优化"""
        spent_budget = 0.0
        results = []
        
        while spent_budget < total_budget:
            result = self.step()
            spent_budget += self.fidelity_costs[result['fidelity']]
            results.append(result)
            
        # 返回高保真度的最优结果
        best_optimizer = self.optimizers[self.fidelities[-1]]
        
        return {
            'best_value': best_optimizer.best_value,
            'best_point': best_optimizer.best_point,
            'total_budget': total_budget,
            'spent_budget': spent_budget,
            'n_evaluations': len(results)
        }


# ==================== 并行REMBO ====================

class ParallelREMBO:
    """并行REMBO优化"""
    
    def __init__(self, objective_fn: Callable,
                 n_workers: int = 4,
                 config: Optional[REMBOConfig] = None):
        self.objective_fn = objective_fn
        self.n_workers = n_workers
        self.config = config or REMBOConfig()
        
        # 主优化器
        self.main_optimizer = REMBOOptimizer(objective_fn, config)
        
        # 批量大小
        self.batch_size = n_workers
        
    def _generate_batch(self) -> List[np.ndarray]:
        """生成批量候选点"""
        candidates = []
        
        # 使用不同的采集函数或随机重启
        for i in range(self.batch_size):
            if i == 0 and len(self.main_optimizer.X_observed) > 0:
                # 第一个使用标准采集函数
                y = self.main_optimizer.suggest_next()
            else:
                # 其他使用随机或扰动
                if len(self.main_optimizer.X_observed) > 0:
                    base = self.main_optimizer.X_observed[-1]
                    y = base + 0.1 * np.random.randn(self.config.embedding_dim)
                    y = np.clip(y, self.config.lower_bound, self.config.upper_bound)
                else:
                    y = np.random.uniform(
                        self.config.lower_bound,
                        self.config.upper_bound,
                        size=self.config.embedding_dim
                    )
            candidates.append(y)
            
        return candidates
    
    def step(self) -> List[Dict]:
        """执行并行步骤"""
        # 生成候选点
        candidates = self._generate_batch()
        
        # 并行评估（这里简化为顺序）
        results = []
        for y in candidates:
            z = self.main_optimizer.embedding.embed(y)
            z = np.clip(z, self.config.lower_bound, self.config.upper_bound)
            value = self.objective_fn(z)
            
            self.main_optimizer.X_observed.append(y.copy())
            self.main_optimizer.y_observed.append(value)
            
            if value < self.main_optimizer.best_value:
                self.main_optimizer.best_value = value
                self.main_optimizer.best_point = z.copy()
                
            results.append({
                'y': y,
                'z': z,
                'value': value
            })
            
        return results
    
    def optimize(self, n_iterations: int = 100) -> Dict[str, Any]:
        """并行优化"""
        # 初始化
        if len(self.main_optimizer.X_observed) == 0:
            self.main_optimizer._initialize()
        
        # 迭代
        for _ in range(n_iterations):
            self.step()
            
        return {
            'best_value': self.main_optimizer.best_value,
            'best_point': self.main_optimizer.best_point,
            'n_evaluations': len(self.main_optimizer.y_observed)
        }
