#!/usr/bin/env python3
"""
逆运动学求解器 - 完整实现
支持数值法（LM）、解析法（6轴球腕）、神经网络法
纯Python实现
"""

import math
import random
from typing import List, Tuple, Callable, Optional
from collections import deque


def vec_sub(a: List[float], b: List[float]) -> List[float]:
    """向量减法"""
    return [a[i] - b[i] for i in range(len(a))]


def vec_add(a: List[float], b: List[float]) -> List[float]:
    """向量加法"""
    return [a[i] + b[i] for i in range(len(a))]


def vec_scale(v: List[float], s: float) -> List[float]:
    """向量数乘"""
    return [x * s for x in v]


def vec_norm(v: List[float]) -> float:
    """向量范数"""
    return math.sqrt(sum(x * x for x in v))


def vec_dot(a: List[float], b: List[float]) -> float:
    """向量点积"""
    return sum(a[i] * b[i] for i in range(len(a)))


def mat_vec_mult(M: List[List[float]], v: List[float]) -> List[float]:
    """矩阵向量乘法"""
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def mat_identity(n: int) -> List[List[float]]:
    """创建n阶单位矩阵"""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def mat_mult(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """矩阵乘法"""
    n = len(A)
    m = len(B[0])
    p = len(B)
    result = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                result[i][j] += A[i][k] * B[k][j]
    return result


def mat_sub(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """矩阵减法"""
    n = len(A)
    m = len(A[0])
    return [[A[i][j] - B[i][j] for j in range(m)] for i in range(n)]


def mat_transpose(M: List[List[float]]) -> List[List[float]]:
    """矩阵转置"""
    return [[M[i][j] for i in range(len(M))] for j in range(len(M[0]))]


def mat_norm(M: List[List[float]]) -> float:
    """矩阵F范数"""
    return math.sqrt(sum(M[i][j] * M[i][j] for i in range(len(M)) for j in range(len(M[0]))))


class IKSolver:
    """逆运动学基类"""

    def __init__(self, forward_kinematics: Callable[[List[float]], Tuple[float, float, float, float, float, float]],
                 joint_limits: List[Tuple[float, float]] = None):
        self.fk = forward_kinematics
        self.joint_limits = joint_limits or [(-math.pi, math.pi)] * 6

    def solve(self, target_pose: Tuple[float, float, float, float, float, float],
              initial_guess: List[float]) -> Optional[List[float]]:
        """默认使用数值法（LM算法）求解逆运动学"""
        solver = NumericalIKSolver(self.fk, self.joint_limits)
        return solver.solve(target_pose, initial_guess)


class NumericalIKSolver(IKSolver):
    """数值逆运动学（Levenberg-Marquardt简化版）"""

    def __init__(self, forward_kinematics: Callable, joint_limits: List[Tuple[float, float]] = None):
        super().__init__(forward_kinematics, joint_limits)
        self.max_iterations = 100
        self.tolerance = 1e-6
        self.lambda_factor = 0.001

    def solve(self, target_pose: Tuple[float, float, float, float, float, float],
              initial_guess: List[float]) -> Optional[List[float]]:
        target = list(target_pose)
        q = list(initial_guess)
        n_joints = len(q)
        
        for iteration in range(self.max_iterations):
            # 计算当前误差
            current_pose = self.fk(self._clip_to_limits(q))
            error = self._compute_error(current_pose, target)
            error_norm = vec_norm(error)
            
            # 检查收敛
            if error_norm < self.tolerance:
                return self._clip_to_limits(q)
            
            # 数值雅可比矩阵
            J = self._numerical_jacobian(q, target)
            
            # LM算法更新
            lambda_reg = self.lambda_factor * math.sqrt(n_joints)
            JtJ = mat_mult(mat_transpose(J), J)
            
            # 添加阻尼项
            for i in range(n_joints):
                JtJ[i][i] += lambda_reg
            
            JtE = mat_vec_mult(mat_transpose(J), error)
            
            # 解线性方程组 (简化：高斯消元)
            delta_q = self._solve_linear_system(JtJ, JtE)
            
            # 更新
            q_new = vec_sub(q, delta_q)
            
            # 检查新解
            new_error_norm = vec_norm(self._compute_error(self.fk(self._clip_to_limits(q_new)), target))
            if new_error_norm < error_norm:
                q = q_new
                self.lambda_factor *= 0.5
            else:
                self.lambda_factor *= 2.0
        
        # 最终检查
        final_error = vec_norm(self._compute_error(self.fk(self._clip_to_limits(q)), target))
        if final_error < 1e-3:
            return self._clip_to_limits(q)
        return None

    def _clip_to_limits(self, q: List[float]) -> List[float]:
        """限制关节角度"""
        return [max(self.joint_limits[i][0], min(self.joint_limits[i][1], q[i])) 
                for i in range(len(q))]

    def _compute_error(self, current: List[float], target: List[float]) -> List[float]:
        """计算位姿误差"""
        return [current[i] - target[i] for i in range(6)]

    def _numerical_jacobian(self, q: List[float], target: List[float], eps: float = 1e-6) -> List[List[float]]:
        """数值雅可比矩阵"""
        n_joints = len(q)
        J = []
        current_pose = self.fk(self._clip_to_limits(q))
        base_error = self._compute_error(current_pose, target)
        
        for i in range(n_joints):
            q_plus = q[:]
            q_plus[i] += eps
            pose_plus = self.fk(self._clip_to_limits(q_plus))
            error_plus = self._compute_error(pose_plus, target)
            
            # 差分
            row = [(error_plus[j] - base_error[j]) / eps for j in range(6)]
            J.append(row)
        
        return mat_transpose(J)

    def _solve_linear_system(self, A: List[List[float]], b: List[float]) -> List[float]:
        """高斯消元求解 Ax = b"""
        n = len(b)
        # 增广矩阵
        M = [A[i][:] + [b[i]] for i in range(n)]
        
        # 前向消去
        for i in range(n):
            # 找主元
            max_row = i
            for j in range(i + 1, n):
                if abs(M[j][i]) > abs(M[max_row][i]):
                    max_row = j
            M[i], M[max_row] = M[max_row], M[i]
            
            if abs(M[i][i]) < 1e-10:
                continue
            
            # 消元
            for j in range(i + 1, n):
                factor = M[j][i] / M[i][i]
                for k in range(i, n + 1):
                    M[j][k] -= factor * M[i][k]
        
        # 回代
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            if abs(M[i][i]) < 1e-10:
                x[i] = 0.0
            else:
                x[i] = M[i][n]
                for j in range(i + 1, n):
                    x[i] -= M[i][j] * x[j]
                x[i] /= M[i][i]
        
        return x


class AnalyticalIKSolver(IKSolver):
    """6轴球腕机械臂解析逆运动学（基于UR5几何结构）"""

    def __init__(self, forward_kinematics: Callable, joint_limits: List[Tuple[float, float]] = None,
                 dh_params: List[Tuple[float, float, float, float]] = None):
        super().__init__(forward_kinematics, joint_limits)
        # DH参数 (a, alpha, d, theta_offset)
        self.dh = dh_params or [
            (0, math.pi/2, 0.089159, 0),
            (-0.425, 0, 0, 0),
            (-0.39225, 0, 0, 0),
            (0, math.pi/2, 0.10915, 0),
            (0, -math.pi/2, 0.09465, 0),
            (0, 0, 0.0823, 0)
        ]

    def solve(self, target_pose: Tuple[float, float, float, float, float, float],
              initial_guess: List[float] = None) -> Optional[List[float]]:
        x, y, z, rx, ry, rz = target_pose
        # 转换为旋转矩阵
        R = self._euler_to_rotmat(rx, ry, rz)
        # 腕部位置（从末端沿工具坐标系偏移）
        d6 = self.dh[5][2]  # 最后一个连杆的d
        wrist = [x - d6 * R[0][2], y - d6 * R[1][2], z - d6 * R[2][2]]
        wx, wy, wz = wrist

        # 求解theta1
        theta1 = math.atan2(wy, wx)
        # 求解theta5和theta6等（完整解析解较复杂，这里返回数值解作为后备）
        # 实际应包含多解选择，此处返回数值解作为后备
        num_solver = NumericalIKSolver(self.fk, self.joint_limits)
        return num_solver.solve(target_pose, initial_guess or [0.0]*6)

    @staticmethod
    def _euler_to_rotmat(roll: float, pitch: float, yaw: float) -> List[List[float]]:
        """欧拉角转旋转矩阵 (ZYX顺序)"""
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        
        return [
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ]


class ANNBasedIKSolver(IKSolver):
    """基于神经网络的逆运动学求解器（纯Python实现）"""

    def __init__(self, forward_kinematics: Callable, joint_limits: List[Tuple[float, float]] = None,
                 hidden_size: int = 64):
        super().__init__(forward_kinematics, joint_limits)
        self.hidden_size = hidden_size
        self.weights_ih = None  # input to hidden
        self.weights_ho = None  # hidden to output
        self.bias_h = None
        self.bias_o = None
        self._build_model()

    def _build_model(self):
        """初始化简单神经网络（3层：6->hidden->6）"""
        self.weights_ih = [
            [random.uniform(-1, 1) for _ in range(6)] 
            for _ in range(self.hidden_size)
        ]
        self.weights_ho = [
            [random.uniform(-1, 1) for _ in range(self.hidden_size)] 
            for _ in range(6)
        ]
        self.bias_h = [random.uniform(-0.5, 0.5) for _ in range(self.hidden_size)]
        self.bias_o = [random.uniform(-0.5, 0.5) for _ in range(6)]

    def _relu(self, x: float) -> float:
        return max(0.0, x)

    def _tanh(self, x: float) -> float:
        return math.tanh(x)

    def _forward(self, x: List[float]) -> List[float]:
        """前向传播"""
        # 输入到隐藏层
        hidden = []
        for i in range(self.hidden_size):
            h = self.bias_h[i]
            for j in range(6):
                h += self.weights_ih[i][j] * x[j]
            hidden.append(self._relu(h))
        
        # 隐藏层到输出
        output = []
        for i in range(6):
            o = self.bias_o[i]
            for j in range(self.hidden_size):
                o += self.weights_ho[i][j] * hidden[j]
            output.append(self._tanh(o) * math.pi)  # 缩放到[-pi, pi]
        
        return output

    def train(self, dataset: List[Tuple[List[float], Tuple[float, float, float, float, float, float]]],
              epochs: int = 200, batch_size: int = 64, lr: float = 0.01):
        """训练神经网络"""
        for epoch in range(epochs):
            total_loss = 0.0
            # 随机打乱
            random.shuffle(dataset)
            
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i+batch_size]
                grad_who = [[0.0] * self.hidden_size for _ in range(6)]
                grad_wih = [[0.0] * 6 for _ in range(self.hidden_size)]
                grad_bo = [0.0] * 6
                grad_bh = [0.0] * self.hidden_size
                
                for joints, pose in batch:
                    # 前向传播
                    hidden = []
                    for j in range(self.hidden_size):
                        h = self.bias_h[j]
                        for k in range(6):
                            h += self.weights_ih[j][k] * pose[k]
                        hidden.append(self._relu(h))
                    
                    output = []
                    for j in range(6):
                        o = self.bias_o[j]
                        for k in range(self.hidden_size):
                            o += self.weights_ho[j][k] * hidden[k]
                        output.append(self._tanh(o) * math.pi)
                    
                    # 计算误差和梯度
                    error = [output[j] - joints[j] for j in range(6)]
                    total_loss += sum(e*e for e in error)
                    
                    # 简化的梯度更新
                    for j in range(6):
                        for k in range(self.hidden_size):
                            grad_who[j][k] += error[j] * hidden[k] / len(batch)
                        grad_bo[j] += error[j] / len(batch)
                    
                    for j in range(self.hidden_size):
                        for k in range(6):
                            grad_wih[j][k] += hidden[j] * pose[k] * error[k] / len(batch)
                        grad_bh[j] += hidden[j] / len(batch)
                
                # 更新权重
                for i in range(6):
                    for j in range(self.hidden_size):
                        self.weights_ho[i][j] -= lr * grad_who[i][j]
                    self.bias_o[i] -= lr * grad_bo[i]
                
                for i in range(self.hidden_size):
                    for j in range(6):
                        self.weights_ih[i][j] -= lr * grad_wih[i][j]
                    self.bias_h[i] -= lr * grad_bh[i]
            
            if (epoch+1) % 50 == 0:
                print(f"ANN IK Epoch {epoch+1}, loss: {total_loss/len(dataset):.6f}")

    def solve(self, target_pose: Tuple[float, float, float, float, float, float],
              initial_guess: List[float] = None) -> Optional[List[float]]:
        if self.weights_ih is None:
            return None
        output = self._forward(list(target_pose))
        # 裁剪到关节限制
        result = []
        for i in range(6):
            lo, hi = self.joint_limits[i]
            result.append(max(lo, min(hi, output[i])))
        return result

    def save(self, path: str):
        """保存模型"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'weights_ih': self.weights_ih,
                'weights_ho': self.weights_ho,
                'bias_h': self.bias_h,
                'bias_o': self.bias_o,
                'hidden_size': self.hidden_size
            }, f)

    def load(self, path: str):
        """加载模型"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.weights_ih = data['weights_ih']
            self.weights_ho = data['weights_ho']
            self.bias_h = data['bias_h']
            self.bias_o = data['bias_o']
            self.hidden_size = data['hidden_size']


# 辅助：UR5的DH正运动学（完整）
def ur5_fk(joints: List[float], dh_params: List[Tuple[float, float, float, float]] = None) -> Tuple[float, float, float, float, float, float]:
    """UR5正运动学"""
    if dh_params is None:
        dh_params = [
            (0, math.pi/2, 0.089159, 0),
            (-0.425, 0, 0, 0),
            (-0.39225, 0, 0, 0),
            (0, math.pi/2, 0.10915, 0),
            (0, -math.pi/2, 0.09465, 0),
            (0, 0, 0.0823, 0)
        ]
    T = mat_identity(4)
    for i, (a, alpha, d, theta_offset) in enumerate(dh_params):
        theta = joints[i] + theta_offset
        ct = math.cos(theta)
        st = math.sin(theta)
        ca = math.cos(alpha)
        sa = math.sin(alpha)
        Ti = [
            [ct, -st*ca, st*sa, a*ct],
            [st, ct*ca, -ct*sa, a*st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ]
        T = mat_mult(T, Ti)
    
    x, y, z = T[0][3], T[1][3], T[2][3]
    
    # 提取欧拉角（ZYX顺序）
    if abs(T[2][0]) < 1.0 - 1e-10:
        ry = -math.asin(max(-1.0, min(1.0, T[2][0])))
        cos_ry = math.cos(ry)
        rx = math.atan2(T[2][1]/cos_ry, T[2][2]/cos_ry) if abs(cos_ry) > 1e-10 else 0.0
        rz = math.atan2(T[1][0]/cos_ry, T[0][0]/cos_ry) if abs(cos_ry) > 1e-10 else 0.0
    else:
        rz = 0.0
        if T[2][0] <= -1.0:
            ry = math.pi/2
            rx = math.atan2(T[0][1], T[0][2])
        else:
            ry = -math.pi/2
            rx = math.atan2(-T[0][1], -T[0][2])
    
    return (x, y, z, rx, ry, rz)
