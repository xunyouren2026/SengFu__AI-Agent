"""
AGI统一框架 - 因果推理模块
实现因果发现、因果推断、反事实推理等核心功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
import math
from collections import defaultdict
from itertools import combinations, permutations
import warnings
warnings.filterwarnings('ignore')


# ==================== 配置类 ====================

@dataclass
class CausalConfig:
    """因果推理配置"""
    # 因果发现配置
    num_variables: int = 10
    max_parents: int = 3
    sparsity_lambda: float = 0.1
    
    # 因果图配置
    edge_threshold: float = 0.3
    use_dag_constraint: bool = True
    
    # 干预配置
    intervention_strength: float = 1.0
    
    # 反事实配置
    num_counterfactual_samples: int = 100
    
    # 学习配置
    learning_rate: float = 0.01
    num_epochs: int = 1000


# ==================== 因果图结构 ====================

class CausalGraph:
    """因果图数据结构"""
    
    def __init__(self, num_variables: int, variable_names: Optional[List[str]] = None):
        self.num_variables = num_variables
        
        if variable_names is None:
            self.variable_names = [f"X{i}" for i in range(num_variables)]
        else:
            self.variable_names = variable_names
        
        # 邻接矩阵 (adj[i][j] = 1 表示 i -> j)
        self.adj_matrix = np.zeros((num_variables, num_variables), dtype=int)
        
        # 边权重（用于软因果图）
        self.edge_weights = np.zeros((num_variables, num_variables), dtype=float)
        
    def add_edge(self, cause: int, effect: int, weight: float = 1.0):
        """添加因果边"""
        self.adj_matrix[cause, effect] = 1
        self.edge_weights[cause, effect] = weight
        
    def remove_edge(self, cause: int, effect: int):
        """移除因果边"""
        self.adj_matrix[cause, effect] = 0
        self.edge_weights[cause, effect] = 0.0
        
    def has_edge(self, cause: int, effect: int) -> bool:
        """检查是否存在边"""
        return self.adj_matrix[cause, effect] == 1
    
    def get_parents(self, node: int) -> List[int]:
        """获取节点的父节点（原因）"""
        return list(np.where(self.adj_matrix[:, node] == 1)[0])
    
    def get_children(self, node: int) -> List[int]:
        """获取节点的子节点（结果）"""
        return list(np.where(self.adj_matrix[node, :] == 1)[0])
    
    def is_dag(self) -> bool:
        """检查是否为DAG（有向无环图）"""
        # 使用拓扑排序检测环
        visited = [False] * self.num_variables
        rec_stack = [False] * self.num_variables
        
        def has_cycle(v):
            visited[v] = True
            rec_stack[v] = True
            
            for u in self.get_children(v):
                if not visited[u]:
                    if has_cycle(u):
                        return True
                elif rec_stack[u]:
                    return True
            
            rec_stack[v] = False
            return False
        
        for i in range(self.num_variables):
            if not visited[i]:
                if has_cycle(i):
                    return False
        
        return True
    
    def topological_sort(self) -> List[int]:
        """拓扑排序"""
        in_degree = np.sum(self.adj_matrix, axis=0)
        queue = [i for i in range(self.num_variables) if in_degree[i] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            for child in self.get_children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        
        return result
    
    def get_ancestors(self, node: int) -> List[int]:
        """获取所有祖先节点"""
        ancestors = set()
        to_visit = self.get_parents(node)
        
        while to_visit:
            parent = to_visit.pop()
            if parent not in ancestors:
                ancestors.add(parent)
                to_visit.extend(self.get_parents(parent))
        
        return list(ancestors)
    
    def get_descendants(self, node: int) -> List[int]:
        """获取所有后代节点"""
        descendants = set()
        to_visit = self.get_children(node)
        
        while to_visit:
            child = to_visit.pop()
            if child not in descendants:
                descendants.add(child)
                to_visit.extend(self.get_children(child))
        
        return list(descendants)
    
    def find_backdoor_adjustment(self, treatment: int, outcome: int) -> Optional[List[int]]:
        """寻找后门调整集"""
        # 获取treatment的后代
        descendants = self.get_descendants(treatment)
        
        # 候选变量（排除treatment、outcome及其后代）
        candidates = [i for i in range(self.num_variables) 
                     if i != treatment and i != outcome and i not in descendants]
        
        # 检查每个候选集
        for r in range(len(candidates) + 1):
            for subset in combinations(candidates, r):
                if self._is_valid_backdoor(treatment, outcome, list(subset)):
                    return list(subset)
        
        return None
    
    def _is_valid_backdoor(self, treatment: int, outcome: int, 
                           adjustment_set: List[int]) -> bool:
        """检查是否为有效的后门调整集"""
        # 简化检查：调整集不应包含treatment的后代
        descendants = self.get_descendants(treatment)
        return not any(adj in descendants for adj in adjustment_set)
    
    def to_adjacency_matrix(self) -> np.ndarray:
        """转换为邻接矩阵"""
        return self.adj_matrix.copy()
    
    @classmethod
    def from_adjacency_matrix(cls, adj_matrix: np.ndarray,
                              variable_names: Optional[List[str]] = None) -> 'CausalGraph':
        """从邻接矩阵创建因果图"""
        num_variables = adj_matrix.shape[0]
        graph = cls(num_variables, variable_names)
        graph.adj_matrix = adj_matrix.astype(int)
        return graph


# ==================== 结构方程模型 ====================

class StructuralEquationModel(nn.Module):
    """结构方程模型 (SEM)"""
    
    def __init__(self, num_variables: int, hidden_dim: int = 64):
        super().__init__()
        self.num_variables = num_variables
        
        # 每个变量的结构方程
        self.equations = nn.ModuleList()
        
        for i in range(num_variables):
            # 输入：所有潜在父变量 + 噪声
            equation = nn.Sequential(
                nn.Linear(num_variables, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            self.equations.append(equation)
        
        # 因果掩码（决定哪些父变量可以影响子变量）
        self.causal_mask = nn.Parameter(
            torch.ones(num_variables, num_variables) * 0.5,
            requires_grad=True
        )
        
    def forward(self, x: torch.Tensor, 
                interventions: Optional[Dict[int, float]] = None) -> torch.Tensor:
        """前向传播，计算变量的值"""
        batch_size = x.size(0)
        output = torch.zeros(batch_size, self.num_variables, device=x.device)
        
        # 应用干预
        if interventions:
            for var, value in interventions.items():
                x[:, var] = value
        
        # 计算每个变量
        for i in range(self.num_variables):
            # 应用因果掩码
            masked_input = x * torch.sigmoid(self.causal_mask[:, i])
            output[:, i] = self.equations[i](masked_input).squeeze(-1)
        
        return output
    
    def get_causal_graph(self, threshold: float = 0.5) -> CausalGraph:
        """从学习到的掩码提取因果图"""
        graph = CausalGraph(self.num_variables)
        
        with torch.no_grad():
            mask = torch.sigmoid(self.causal_mask).numpy()
            
        for i in range(self.num_variables):
            for j in range(self.num_variables):
                if mask[i, j] > threshold:
                    graph.add_edge(i, j, mask[i, j])
        
        return graph


# ==================== 因果发现算法 ====================

class CausalDiscovery:
    """因果发现算法"""
    
    def __init__(self, config: Optional[CausalConfig] = None):
        self.config = config or CausalConfig()
        
    def pc_algorithm(self, data: np.ndarray, 
                     alpha: float = 0.05) -> CausalGraph:
        """PC算法 - 基于条件独立性检验"""
        n_samples, n_vars = data.shape
        graph = CausalGraph(n_vars)
        
        # 初始化完全图
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    graph.add_edge(i, j)
                    graph.add_edge(j, i)
        
        # 阶段1：移除条件独立的边
        for cond_size in range(n_vars):
            for i in range(n_vars):
                for j in graph.get_parents(i) + graph.get_children(i):
                    if not graph.has_edge(i, j) and not graph.has_edge(j, i):
                        continue
                    
                    # 获取候选条件集
                    adj_i = set(graph.get_parents(i) + graph.get_children(i)) - {j}
                    
                    for cond_set in combinations(adj_i, min(cond_size, len(adj_i))):
                        # 条件独立性检验
                        if self._conditional_independent(data, i, j, list(cond_set), alpha):
                            graph.remove_edge(i, j)
                            graph.remove_edge(j, i)
                            break
        
        # 阶段2：定向边（简化版本）
        self._orient_edges(graph)
        
        return graph
    
    def _conditional_independent(self, data: np.ndarray, x: int, y: int,
                                 cond_set: List[int], alpha: float) -> bool:
        """条件独立性检验（基于偏相关）"""
        if len(cond_set) == 0:
            # 简单相关
            corr = np.corrcoef(data[:, x], data[:, y])[0, 1]
            return abs(corr) < alpha
        else:
            # 偏相关（简化计算）
            vars_idx = [x, y] + cond_set
            sub_data = data[:, vars_idx]
            corr_matrix = np.corrcoef(sub_data.T)
            
            try:
                prec_matrix = np.linalg.inv(corr_matrix)
                partial_corr = -prec_matrix[0, 1] / np.sqrt(prec_matrix[0, 0] * prec_matrix[1, 1])
                return abs(partial_corr) < alpha
            except np.linalg.LinAlgError:
                return False
    
    def _orient_edges(self, graph: CausalGraph):
        """定向边（v-结构检测）"""
        n = graph.num_variables
        
        for i in range(n):
            for j in range(i + 1, n):
                # 检查是否为v-结构 i -> k <- j
                for k in range(n):
                    if k == i or k == j:
                        continue
                    
                    if (graph.has_edge(i, k) or graph.has_edge(k, i)) and \
                       (graph.has_edge(j, k) or graph.has_edge(k, j)) and \
                       not (graph.has_edge(i, j) or graph.has_edge(j, i)):
                        # i和j不相邻，都连接到k -> v-结构
                        graph.remove_edge(k, i)
                        graph.remove_edge(k, j)
    
    def notears_algorithm(self, data: np.ndarray, 
                          lambda1: float = 0.1,
                          max_iter: int = 100) -> CausalGraph:
        """NOTEARS算法 - 基于连续优化的因果发现"""
        n_samples, n_vars = data.shape
        
        # 初始化权重矩阵
        W = np.zeros((n_vars, n_vars))
        
        # 优化
        for iteration in range(max_iter):
            # 计算梯度
            grad = self._compute_gradient(data, W, lambda1)
            
            # 更新
            W = W - self.config.learning_rate * grad
            
            # 投影到DAG空间
            W = self._project_to_dag(W)
        
        # 构建因果图
        graph = CausalGraph(n_vars)
        for i in range(n_vars):
            for j in range(n_vars):
                if abs(W[i, j]) > self.config.edge_threshold:
                    graph.add_edge(i, j, abs(W[i, j]))
        
        return graph
    
    def _compute_gradient(self, data: np.ndarray, W: np.ndarray, 
                          lambda1: float) -> np.ndarray:
        """计算损失梯度"""
        n_samples, n_vars = data.shape
        
        # 损失函数: ||X - XW||^2 + lambda1 * ||W||_1 + DAG约束
        XW = data @ W
        residual = data - XW
        
        # 梯度
        grad = -2 * data.T @ residual / n_samples
        
        # L1正则化次梯度
        grad += lambda1 * np.sign(W)
        
        return grad
    
    def _project_to_dag(self, W: np.ndarray) -> np.ndarray:
        """投影到DAG空间（简化版本）"""
        # 使用矩阵指数的迹作为DAG约束
        # h(W) = tr(e^{W \circ W}) - n = 0
        
        n = W.shape[0]
        W_squared = W * W
        
        # 计算h(W)
        try:
            exp_W = np.linalg.matrix_exp(W_squared)
            h = np.trace(exp_W) - n
        except:
            return W
        
        # 如果h(W) > 0，需要调整
        if h > 0:
            # 简化：按权重排序，移除最小的边
            flat_W = np.abs(W).flatten()
            sorted_indices = np.argsort(flat_W)
            
            for idx in sorted_indices:
                i, j = idx // n, idx % n
                if h > 0:
                    W[i, j] = 0
                    try:
                        exp_W = np.linalg.matrix_exp(W * W)
                        h = np.trace(exp_W) - n
                    except:
                        break
        
        return W


# ==================== 因果推断 ====================

class CausalInference:
    """因果推断"""
    
    def __init__(self, graph: CausalGraph, 
                 data: Optional[np.ndarray] = None):
        self.graph = graph
        self.data = data
        
    def compute_ate(self, treatment: int, outcome: int,
                    treatment_value: float = 1.0,
                    control_value: float = 0.0) -> float:
        """计算平均处理效应 (ATE)"""
        # 使用后门调整
        adjustment_set = self.graph.find_backdoor_adjustment(treatment, outcome)
        
        if adjustment_set is None:
            # 无法找到有效调整集
            return 0.0
        
        # E[Y|do(T=1)] - E[Y|do(T=0)]
        # = sum_z E[Y|T=1,Z=z] * P(Z=z) - E[Y|T=0,Z=z] * P(Z=z)
        
        if self.data is None:
            return 0.0
        
        ate = 0.0
        
        # 离散化调整变量
        adjustment_data = self.data[:, adjustment_set]
        unique_values = np.unique(adjustment_data, axis=0)
        
        for z in unique_values:
            # P(Z=z)
            mask_z = np.all(adjustment_data == z, axis=1)
            p_z = mask_z.sum() / len(self.data)
            
            # E[Y|T=1,Z=z]
            mask_t1 = (self.data[:, treatment] == treatment_value) & mask_z
            if mask_t1.sum() > 0:
                e_y_t1 = self.data[mask_t1, outcome].mean()
            else:
                e_y_t1 = 0.0
            
            # E[Y|T=0,Z=z]
            mask_t0 = (self.data[:, treatment] == control_value) & mask_z
            if mask_t0.sum() > 0:
                e_y_t0 = self.data[mask_t0, outcome].mean()
            else:
                e_y_t0 = 0.0
            
            ate += (e_y_t1 - e_y_t0) * p_z
        
        return ate
    
    def compute_cate(self, treatment: int, outcome: int,
                     conditioning_vars: List[int],
                     treatment_value: float = 1.0,
                     control_value: float = 0.0) -> Dict[Tuple, float]:
        """计算条件平均处理效应 (CATE)"""
        if self.data is None:
            return {}
        
        cate = {}
        
        # 获取条件变量的唯一值
        cond_data = self.data[:, conditioning_vars]
        unique_values = np.unique(cond_data, axis=0)
        
        for z in unique_values:
            mask_z = np.all(cond_data == z, axis=1)
            
            # E[Y|T=1,Z=z]
            mask_t1 = (self.data[:, treatment] == treatment_value) & mask_z
            if mask_t1.sum() > 0:
                e_y_t1 = self.data[mask_t1, outcome].mean()
            else:
                e_y_t1 = 0.0
            
            # E[Y|T=0,Z=z]
            mask_t0 = (self.data[:, treatment] == control_value) & mask_z
            if mask_t0.sum() > 0:
                e_y_t0 = self.data[mask_t0, outcome].mean()
            else:
                e_y_t0 = 0.0
            
            cate[tuple(z)] = e_y_t1 - e_y_t0
        
        return cate
    
    def compute_ite(self, treatment: int, outcome: int,
                    individual_idx: int,
                    treatment_value: float = 1.0,
                    control_value: float = 0.0) -> float:
        """计算个体处理效应 (ITE) - 需要反事实"""
        # 这里使用简化的估计
        # 实际需要反事实推理
        
        if self.data is None or individual_idx >= len(self.data):
            return 0.0
        
        # 使用最近邻匹配
        individual = self.data[individual_idx]
        
        # 找到相似的已处理个体
        treated = self.data[self.data[:, treatment] == treatment_value]
        control = self.data[self.data[:, treatment] == control_value]
        
        if len(treated) == 0 or len(control) == 0:
            return 0.0
        
        # 计算距离（排除treatment和outcome）
        feature_idx = [i for i in range(self.data.shape[1]) 
                      if i != treatment and i != outcome]
        
        distances_treated = np.linalg.norm(
            treated[:, feature_idx] - individual[feature_idx], axis=1
        )
        distances_control = np.linalg.norm(
            control[:, feature_idx] - individual[feature_idx], axis=1
        )
        
        # 最近邻
        nn_treated = treated[np.argmin(distances_treated), outcome]
        nn_control = control[np.argmin(distances_control), outcome]
        
        return nn_treated - nn_control


# ==================== 反事实推理 ====================

class CounterfactualReasoning:
    """反事实推理"""
    
    def __init__(self, sem: StructuralEquationModel,
                 config: Optional[CausalConfig] = None):
        self.sem = sem
        self.config = config or CausalConfig()
        
    def compute_counterfactual(self, factual: torch.Tensor,
                               interventions: Dict[int, float],
                               target: int) -> torch.Tensor:
        """计算反事实: Y_x(u) = Y 在干预X=x后的值"""
        # 三步法：
        # 1. Abduction: 从事实推断噪声
        # 2. Action: 执行干预
        # 3. Prediction: 预测目标变量
        
        with torch.no_grad():
            # Step 1: Abduction - 估计噪声
            predicted = self.sem(factual)
            noise = factual - predicted
            
            # Step 2 & 3: Action & Prediction
            # 创建干预后的输入
            counterfactual_input = factual.clone()
            for var, value in interventions.items():
                counterfactual_input[:, var] = value
            
            # 计算反事实值
            counterfactual_output = self.sem(counterfactual_input, interventions)
            
            # 添加噪声（保持个体特征）
            counterfactual_output = counterfactual_output + noise
            
        return counterfactual_output[:, target]
    
    def compute_counterfactual_distribution(self, factual: torch.Tensor,
                                            interventions: Dict[int, float],
                                            target: int,
                                            num_samples: int = 100) -> torch.Tensor:
        """计算反事实分布"""
        samples = []
        
        for _ in range(num_samples):
            # 添加随机噪声
            noisy_factual = factual + torch.randn_like(factual) * 0.1
            cf = self.compute_counterfactual(noisy_factual, interventions, target)
            samples.append(cf)
        
        return torch.stack(samples)
    
    def compute_counterfactual_effect(self, factual: torch.Tensor,
                                      treatment: int,
                                      treatment_value: float,
                                      control_value: float,
                                      outcome: int) -> torch.Tensor:
        """计算反事实处理效应"""
        # Y_{T=1}(u) - Y_{T=0}(u)
        cf_treated = self.compute_counterfactual(
            factual, {treatment: treatment_value}, outcome
        )
        cf_control = self.compute_counterfactual(
            factual, {treatment: control_value}, outcome
        )
        
        return cf_treated - cf_control


# ==================== 因果模型学习 ====================

class CausalModelLearner:
    """因果模型学习器"""
    
    def __init__(self, num_variables: int, 
                 config: Optional[CausalConfig] = None,
                 device: str = 'cpu'):
        self.num_variables = num_variables
        self.config = config or CausalConfig()
        self.device = device
        
        # 结构方程模型
        self.sem = StructuralEquationModel(
            num_variables, hidden_dim=64
        ).to(device)
        
        # 因果发现
        self.discovery = CausalDiscovery(config)
        
    def learn_structure(self, data: np.ndarray,
                        method: str = "pc") -> CausalGraph:
        """学习因果结构"""
        if method == "pc":
            return self.discovery.pc_algorithm(data)
        elif method == "notears":
            return self.discovery.notears_algorithm(data)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def learn_mechanisms(self, data: torch.Tensor,
                         graph: CausalGraph,
                         num_epochs: int = 100) -> Dict[str, float]:
        """学习因果机制"""
        data = data.to(self.device)
        optimizer = torch.optim.Adam(self.sem.parameters(), lr=0.01)
        
        history = {'loss': []}
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # 前向传播
            predicted = self.sem(data)
            
            # 重建损失
            loss = F.mse_loss(predicted, data)
            
            # DAG约束
            if self.config.use_dag_constraint:
                dag_loss = self._compute_dag_loss()
                loss = loss + self.config.sparsity_lambda * dag_loss
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            history['loss'].append(loss.item())
        
        return history
    
    def _compute_dag_loss(self) -> torch.Tensor:
        """计算DAG约束损失"""
        # h(W) = tr(e^{W \circ W}) - n
        W = torch.sigmoid(self.sem.causal_mask)
        W_squared = W * W
        
        # 使用矩阵幂级数近似
        n = self.num_variables
        exp_approx = torch.eye(n, device=self.device)
        term = torch.eye(n, device=self.device)
        
        for k in range(1, 10):
            term = term @ W_squared / k
            exp_approx = exp_approx + term
        
        h = torch.trace(exp_approx) - n
        return h * h  # 平方以确保非负
    
    def fit(self, data: np.ndarray,
            structure_method: str = "pc",
            num_epochs: int = 100) -> CausalGraph:
        """拟合因果模型"""
        # 学习结构
        graph = self.learn_structure(data, structure_method)
        
        # 学习机制
        data_tensor = torch.tensor(data, dtype=torch.float32)
        self.learn_mechanisms(data_tensor, graph, num_epochs)
        
        return graph
    
    def intervene(self, data: torch.Tensor,
                  interventions: Dict[int, float]) -> torch.Tensor:
        """执行干预"""
        return self.sem(data.to(self.device), interventions)
    
    def counterfactual(self, factual: torch.Tensor,
                       interventions: Dict[int, float],
                       target: int) -> torch.Tensor:
        """反事实推理"""
        reasoner = CounterfactualReasoning(self.sem, self.config)
        return reasoner.compute_counterfactual(
            factual.to(self.device), interventions, target
        )


# ==================== 工具函数 ====================

def visualize_causal_graph(graph: CausalGraph) -> str:
    """可视化因果图（文本格式）"""
    lines = ["Causal Graph:"]
    
    for i in range(graph.num_variables):
        parents = graph.get_parents(i)
        if parents:
            parent_names = [graph.variable_names[p] for p in parents]
            lines.append(f"  {graph.variable_names[i]} <- {', '.join(parent_names)}")
    
    return "\n".join(lines)


def compute_causal_effect_strength(data: np.ndarray, 
                                   cause: int, effect: int) -> float:
    """计算因果效应强度（简化版本）"""
    # 使用回归系数作为效应强度的估计
    X = data[:, cause].reshape(-1, 1)
    y = data[:, effect]
    
    # 添加截距
    X = np.column_stack([np.ones(len(X)), X])
    
    # 最小二乘
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        return beta[1]  # 返回斜率
    except:
        return 0.0


def d_separation(graph: CausalGraph, x: int, y: int, 
                 z: List[int]) -> bool:
    """d-分离检验"""
    # 简化的d-分离检验
    # X和Y在给定Z时d-分离，如果所有路径都被Z阻塞
    
    # 获取所有路径（简化：只检查直接路径）
    ancestors_z = set()
    for node in z:
        ancestors_z.update(graph.get_ancestors(node))
        ancestors_z.add(node)
    
    # 检查直接边
    if graph.has_edge(x, y) or graph.has_edge(y, x):
        # 如果x->y或y->x，且中间节点不在Z中
        return False
    
    # 检查共同子节点（v-结构）
    children_x = set(graph.get_children(x))
    children_y = set(graph.get_children(y))
    common_children = children_x & children_y
    
    for child in common_children:
        # 如果共同子节点或其后代在Z中，则路径激活
        if child in ancestors_z:
            return False
    
    return True
