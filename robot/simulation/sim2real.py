#!/usr/bin/env python3
"""
Sim2Real完整工具集
包含域随机化、系统辨识（梯度优化）、残差策略、在线自适应
使用纯Python标准库实现
"""

import random
import pickle
import os
import math
from typing import List, Tuple, Dict, Any, Callable, Optional


class SimpleMatrix:
    """简单的纯Python矩阵实现（替代numpy）"""
    def __init__(self, data: List[List[float]]):
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0]) if data else 0

    @staticmethod
    def eye(n: int) -> 'SimpleMatrix':
        """单位矩阵"""
        data = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        return SimpleMatrix(data)

    @staticmethod
    def zeros(rows: int, cols: int) -> 'SimpleMatrix':
        """零矩阵"""
        return SimpleMatrix([[0.0] * cols for _ in range(rows)])

    def __add__(self, other: 'SimpleMatrix') -> 'SimpleMatrix':
        result = [[self.data[i][j] + other.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
        return SimpleMatrix(result)

    def __sub__(self, other: 'SimpleMatrix') -> 'SimpleMatrix':
        result = [[self.data[i][j] - other.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
        return SimpleMatrix(result)

    def __mul__(self, scalar: float) -> 'SimpleMatrix':
        result = [[self.data[i][j] * scalar for j in range(self.cols)] for i in range(self.rows)]
        return SimpleMatrix(result)

    def __matmul__(self, other: 'SimpleMatrix') -> 'SimpleMatrix':
        """矩阵乘法"""
        if self.cols != other.rows:
            raise ValueError(f"Matrix dimensions mismatch: {self.cols} != {other.rows}")
        result = [[0.0] * other.cols for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    result[i][j] += self.data[i][k] * other.data[k][j]
        return SimpleMatrix(result)

    def __getitem__(self, key: Tuple[int, int]) -> float:
        i, j = key
        return self.data[i][j]

    def __setitem__(self, key: Tuple[int, int], value: float):
        i, j = key
        self.data[i][j] = value

    def T(self) -> 'SimpleMatrix':
        """转置"""
        result = [[self.data[j][i] for j in range(self.rows)] for i in range(self.cols)]
        return SimpleMatrix(result)

    def to_list(self) -> List[List[float]]:
        return self.data

    @staticmethod
    def from_list(arr: List) -> 'SimpleMatrix':
        """从列表创建矩阵"""
        if isinstance(arr[0], (int, float)):
            return SimpleMatrix([arr])  # 向量转单行矩阵
        return SimpleMatrix(arr)

    def sum(self) -> float:
        """所有元素之和"""
        return sum(sum(row) for row in self.data)

    def norm(self) -> float:
        """Frobenius范数"""
        return math.sqrt(sum(x*x for row in self.data for x in row))


def matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """矩阵乘法"""
    if len(a[0]) != len(b):
        raise ValueError(f"Matrix dimensions mismatch: {len(a[0])} != {len(b)}")
    result = [[0.0] * len(b[0]) for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(a[0])):
                result[i][j] += a[i][k] * b[k][j]
    return result


def mat_vec_mul(a: List[List[float]], v: List[float]) -> List[float]:
    """矩阵向量乘法"""
    return [sum(a[i][j] * v[j] for j in range(len(v))) for i in range(len(a))]


def outer_product(a: List[float], b: List[float]) -> List[List[float]]:
    """外积"""
    return [[a[i] * b[j] for j in range(len(b))] for i in range(len(a))]


def random_normal(mean: float = 0.0, std: float = 1.0) -> float:
    """Box-Muller变换生成正态分布随机数"""
    u1 = random.random()
    u2 = random.random()
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mean + std * z


class DomainRandomizer:
    """域随机化：对仿真参数进行随机扰动"""

    def __init__(self, randomization_rules: Dict[str, Tuple[float, float]] = None):
        """
        randomization_rules: 参数名 -> (最小值, 最大值) 乘性因子或加性范围
        """
        self.rules = randomization_rules or {
            "joint_friction": (0.8, 1.2),
            "mass_scale": (0.9, 1.1),
            "control_latency": (0.0, 0.05),
            "sensor_noise_std": (0.0, 0.02),
            "gravity": (9.8, 9.8),
        }

    def randomize(self, sim_params: Dict[str, Any]) -> Dict[str, Any]:
        """对仿真参数进行随机化"""
        randomized = sim_params.copy()
        for key, (low, high) in self.rules.items():
            if key in randomized:
                if key in ["joint_friction", "mass_scale"]:
                    randomized[key] = randomized[key] * random.uniform(low, high)
                elif key == "control_latency":
                    randomized[key] = random.uniform(low, high)
                elif key == "sensor_noise_std":
                    randomized[key] = random.uniform(low, high)
                else:
                    randomized[key] = random.uniform(low, high)
        return randomized

    def add_noise_to_observation(self, obs: List[float], noise_std: float = 0.01) -> List[float]:
        """给观测添加高斯噪声"""
        return [v + random_normal(0, noise_std) for v in obs]

    def generate_randomized_configs(self, base_config: Dict, num_configs: int = 10) -> List[Dict]:
        """生成多个随机化配置"""
        return [self.randomize(base_config) for _ in range(num_configs)]


class SystemIdentification:
    """系统辨识：通过优化仿真参数匹配真实轨迹"""

    def __init__(self, sim_model: Callable, initial_params: Dict[str, float]):
        """
        sim_model: 仿真模型函数，输入(control, params_array) 输出 next_state
        initial_params: 初始参数
        """
        self.sim_model = sim_model
        self.params = initial_params.copy()
        self.param_names = list(initial_params.keys())

    def _param_vector(self) -> List[float]:
        return [self.params[n] for n in self.param_names]

    def _set_params(self, vec: List[float]):
        for i, name in enumerate(self.param_names):
            self.params[name] = vec[i]

    def _loss(self, param_vec: List[float], trajectory: List[Tuple[List[float], List[float]]]) -> float:
        """计算仿真轨迹与真实轨迹的MSE"""
        self._set_params(param_vec)
        total_loss = 0.0
        for control, real_state in trajectory:
            sim_state = self.sim_model(control, param_vec)
            error = [sim_state[i] - real_state[i] for i in range(len(real_state))]
            total_loss += sum(e*e for e in error)
        return total_loss / len(trajectory)

    def _gradient(self, param_vec: List[float], trajectory: List[Tuple[List[float], List[float]]],
                  eps: float = 1e-5) -> List[float]:
        """数值梯度计算"""
        loss0 = self._loss(param_vec, trajectory)
        grad = []
        for i in range(len(param_vec)):
            perturbed = param_vec[:]
            perturbed[i] += eps
            loss1 = self._loss(perturbed, trajectory)
            grad.append((loss1 - loss0) / eps)
        return grad

    def identify(self, trajectory: List[Tuple[List[float], List[float]]],
                 method: str = 'gradient_descent', bounds: List[Tuple[float, float]] = None) -> Dict[str, float]:
        """
        辨识参数（使用梯度下降替代scipy.optimize.minimize）
        trajectory: [(control, observed_state), ...]
        """
        x0 = self._param_vector()
        if bounds is None:
            bounds = [(0.5, 1.5) for _ in range(len(x0))]  # 默认范围

        lr = 0.01
        max_iter = 1000
        tolerance = 1e-6

        x = x0[:]
        for _ in range(max_iter):
            grad = self._gradient(x, trajectory)
            new_x = [x[i] - lr * grad[i] for i in range(len(x))]

            # 应用边界约束
            new_x = [max(bounds[i][0], min(bounds[i][1], new_x[i])) for i in range(len(new_x))]

            # 检查收敛
            diff = math.sqrt(sum((new_x[i] - x[i])**2 for i in range(len(x))))
            x = new_x

            if diff < tolerance:
                break

        self._set_params(x)
        return self.params

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.params, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            self.params = pickle.load(f)


class ResidualPolicy:
    """残差策略：基础策略 + 自适应校正"""

    def __init__(self, base_policy: Callable, adaptation_model: Optional[Callable] = None,
                 learning_rate: float = 0.01):
        """
        base_policy: 基础策略，输入状态输出动作
        adaptation_model: 自适应模型，输入状态输出动作增量（可为None，则使用线性回归）
        """
        self.base_policy = base_policy
        self.adaptation = adaptation_model
        self.lr = learning_rate
        if adaptation_model is None:
            # 使用简单的线性回归作为自适应模型
            self._dim_state = 6
            self._dim_action = 6
            self._linear_weights = [[0.0] * self._dim_state for _ in range(self._dim_action)]
            self._adaptation_func = self._linear_adapt

    def _linear_adapt(self, state: List[float]) -> List[float]:
        """线性自适应"""
        return mat_vec_mul(self._linear_weights, state)

    def act(self, state: List[float]) -> List[float]:
        base_action = self.base_policy(state)
        if self.adaptation:
            delta = self.adaptation(state)
        else:
            delta = self._linear_adapt(state)
        return [base_action[i] + delta[i] for i in range(len(base_action))]

    def update_adaptation(self, state: List[float], desired_action: List[float], actual_action: List[float]):
        """在线更新自适应模型（基于误差）"""
        error = [desired_action[i] - actual_action[i] for i in range(len(desired_action))]
        if self.adaptation is None:
            # 线性回归更新: W += lr * outer(error, state)
            outer = outer_product(error, state)
            for i in range(len(self._linear_weights)):
                for j in range(len(self._linear_weights[i])):
                    self._linear_weights[i][j] += self.lr * outer[i][j]
        else:
            # 对于可训练模型，可以调用其更新方法（需额外实现）
            pass

    def get_adaptation_weights(self) -> List[List[float]]:
        return self._linear_weights


# 辅助：简单弹簧质量阻尼系统模型（用于测试系统辨识）
def spring_mass_damper_model(control: List[float], params: List[float]) -> List[float]:
    """
    状态: [position, velocity], 控制: [force]
    params: [mass, stiffness, damping]
    """
    pos, vel = control[0], control[1]
    force = control[2]
    m, k, b = params
    acc = (force - k*pos - b*vel) / m
    dt = 0.01
    new_pos = pos + vel * dt
    new_vel = vel + acc * dt
    return [new_pos, new_vel]


# 示例基础策略（例如PID）
def example_base_policy(state: List[float]) -> List[float]:
    # 简单的比例控制
    target = [1.0, 0.0]
    error = [target[i] - state[i] for i in range(2)]
    return [error[0] * 10.0, 0.0, 0.0]  # 输出力
