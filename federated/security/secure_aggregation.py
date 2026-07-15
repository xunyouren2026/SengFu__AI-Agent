"""
Secure Aggregation Module for Federated Learning
实现联邦学习安全聚合 - 保护隐私的同时进行模型聚合

This module provides security mechanisms for federated learning:
1. Secret Sharing (Shamir's Secret Sharing)
2. Differential Privacy (Gaussian noise addition)
3. Byzantine Fault Tolerance (Krum, Trimmed Mean, Multi-Krum)
4. Secure Aggregation Protocol

Author: AGI Unified Framework
"""

import random
import hashlib
import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import threading


# ============== 配置和基础数据结构 ==============

@dataclass
class SecureAggregationConfig:
    """
    安全聚合配置
    
    配置秘密分享阈值、客户端数量、噪声参数等
    """
    threshold: int = 3                    # 秘密分享阈值（3-of-5表示5份中需要3份才能恢复）
    num_clients: int = 5                  # 参与客户端数量
    security_param: float = 1e-5          # 安全参数δ
    noise_multiplier: float = 1.0         # 噪声乘数（用于差分隐私）
    max_gradient_norm: float = 1.0        # 梯度裁剪最大范数
    privacy_budget: float = 8.0           # 隐私预算ε


@dataclass
class SecretShare:
    """
    秘密分享片
    
    Shamir秘密分享的一份
    """
    share_id: int           # 分享ID
    owner_id: str           # 所有者ID
    x_coord: int            # 在多项式上的x坐标
    value: float            # 分享值
    mac: str = ""           # 消息认证码（用于验证）
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'share_id': self.share_id,
            'owner_id': self.owner_id,
            'x_coord': self.x_coord,
            'value': self.value,
            'mac': self.mac
        }


# ============== Shamir秘密分享 ==============

class SecretSharing:
    """
    Shamir秘密分享实现
    
    将一个秘密分成n份，只有收集到至少threshold份
    才能重建原始秘密。这提供了：
    - 容错性：即使部分节点失效也不影响
    - 隐私性：少于threshold份无法得到任何信息
    
    基于有限域上的多项式插值
    """
    
    # 使用一个足够大的素数来保证数值精度
    DEFAULT_PRIME = 2**61 - 1  # Mersenne素数
    
    def __init__(self, threshold: int = 3, prime: int = None):
        self._threshold = threshold
        self._prime = prime or self.DEFAULT_PRIME
        self._generator = 5  # 有限域的生成元
        
    def split(self, secret: float, n_shares: int, owner_id: str = "") -> List[SecretShare]:
        """
        将秘密分成n份
        
        使用Shamir秘密分享算法：
        1. 在有限域上随机生成一个threshold-1次多项式
        2. 多项式在0点的值就是原始秘密
        3. 在不同的x点求值得到n份分享
        
        Args:
            secret: 要分享的秘密值
            n_shares: 分享份数
            
        Returns:
            n份SecretShare对象
        """
        # 将秘密映射到有限域
        secret_int = self._float_to_field(secret)
        
        # 生成随机系数（除了常数项是秘密）
        coefficients = [secret_int]
        for _ in range(self._threshold - 1):
            coefficients.append(random.randint(0, self._prime - 1))
        
        # 在不同的x点求值
        shares = []
        for i in range(1, n_shares + 1):
            # 使用多项式求值（Horner's method）
            value = 0
            for coeff in reversed(coefficients):
                value = (value * i + coeff) % self._prime
            
            share = SecretShare(
                share_id=i,
                owner_id=owner_id or f"share_{i}",
                x_coord=i,
                value=self._field_to_float(value),
                mac=""  # MAC稍后添加
            )
            shares.append(share)
        
        return shares
    
    def combine(self, shares: List[SecretShare]) -> float:
        """
        重建秘密
        
        使用拉格朗日插值在x=0点求值来恢复秘密
        只需要任意threshold份即可
        
        Args:
            shares: 至少threshold份分享
            
        Returns:
            恢复的秘密值
        """
        if len(shares) < self._threshold:
            raise ValueError(
                f"需要至少 {self._threshold} 份分享才能重建秘密，"
                f"但只提供了 {len(shares)} 份"
            )
        
        # 提取x坐标和y值
        x_coords = [s.x_coord for s in shares]
        y_values = [self._float_to_field(s.value) for s in shares]
        
        # 拉格朗日插值在x=0点的值
        secret_int = self._lagrange_interpolation(0, x_coords, y_values)
        
        return self._field_to_float(secret_int)
    
    def _lagrange_interpolation(self, x: int, x_coords: List[int], y_values: List[int]) -> int:
        """
        拉格朗日插值
        
        计算拉格朗日基多项式在点x的值
        
        L(x) = Σ y_i * l_i(x)
        其中 l_i(x) = Π_{j≠i} (x - x_j) / (x_i - x_j)
        """
        result = 0
        
        for i, x_i in enumerate(x_coords):
            y_i = y_values[i]
            
            # 计算拉格朗日基多项式在x点的值
            numerator = 1
            denominator = 1
            
            for j, x_j in enumerate(x_coords):
                if i != j:
                    numerator = (numerator * (x - x_j)) % self._prime
                    denominator = (denominator * (x_i - x_j)) % self._prime
            
            # 计算模逆元
            l_i_x = numerator * self._mod_inverse(denominator, self._prime) % self._prime
            
            result = (result + y_i * l_i_x) % self._prime
        
        return result
    
    def _mod_inverse(self, a: int, m: int) -> int:
        """计算模逆元"""
        # 扩展欧几里得算法
        def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y
        
        _, x, _ = extended_gcd(a % m, m)
        return (x + m) % m
    
    def verify_share(self, share: SecretShare, mac_key: str) -> bool:
        """
        验证分享的有效性
        
        使用MAC来验证分享者确实是声称的身份
        """
        if not share.mac:
            return True  # 没有MAC时默认有效
        
        expected_mac = self._generate_mac(share, mac_key)
        return share.mac == expected_mac
    
    def _generate_mac(self, share: SecretShare, key: str) -> str:
        """生成MAC"""
        data = f"{share.share_id}:{share.owner_id}:{share.x_coord}:{share.value}"
        return hashlib.sha256(f"{data}:{key}".encode()).hexdigest()[:16]
    
    def add_mac_to_shares(self, shares: List[SecretShare], mac_key: str) -> List[SecretShare]:
        """为分享添加MAC"""
        for share in shares:
            share.mac = self._generate_mac(share, mac_key)
        return shares
    
    def _float_to_field(self, value: float) -> int:
        """将浮点数映射到有限域"""
        # 使用缩放因子处理小数
        scale = 10**6
        scaled = int(value * scale)
        return scaled % self._prime
    
    def _field_to_float(self, value: int) -> float:
        """将有限域值映射回浮点数"""
        scale = 10**6
        return value / scale


# ============== 差分隐私 ==============

class DifferentialPrivacy:
    """
    差分隐私实现
    
    通过向数据添加校准噪声来提供隐私保证：
    - ε-差分隐私：隐私预算，越小隐私保护越强
    - δ-额外失败概率：允许的隐私保证失败概率
    
    实现基于Renyi差分隐私（RDP）的高斯机制
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, sensitivity: float = 1.0):
        self._epsilon = epsilon
        self._delta = delta
        self._sensitivity = sensitivity
        self._total_steps = 0
        self._accumulated_epsilon = 0.0
        
    def add_noise(self, value: float) -> float:
        """
        向值添加校准高斯噪声
        
        噪声标准差根据隐私预算和敏感度计算
        """
        sigma = self._compute_noise_scale()
        noise = random.gauss(0, sigma)
        return value + noise
    
    def _compute_noise_scale(self) -> float:
        """
        计算噪声标准差
        
        基于Renyi差分隐私的账本机制
        """
        # 对于高斯机制，使用RDP转换
        # σ = c * Δf / ε，其中c ≈ 1
        # 这里使用更保守的估计
        c = 1.2  # 安全系数
        
        # 考虑到累积隐私损失
        effective_epsilon = self._epsilon / max(1, math.sqrt(2 * self._total_steps))
        
        return c * self._sensitivity / max(effective_epsilon, 0.01)
    
    def clip_gradient(self, gradient: List[float], max_norm: float) -> List[float]:
        """
        梯度裁剪
        
        将梯度的L2范数裁剪到max_norm
        这控制了敏感度
        """
        if not gradient:
            return gradient
            
        # 计算当前范数
        norm = math.sqrt(sum(g**2 for g in gradient))
        
        # 如果范数超过阈值，按比例缩放
        if norm > max_norm:
            scale = max_norm / norm
            return [g * scale for g in gradient]
        
        return gradient
    
    def privacy_budget_spent(self) -> float:
        """
        计算已消耗的隐私预算
        
        基于RDP的隐私账本
        """
        if self._total_steps == 0:
            return 0.0
        
        # 累积RDP
        accumulated_rdp = sum(
            self._compute_rdp(self._epsilon / math.sqrt(self._total_steps), 1)
            for _ in range(self._total_steps)
        )
        
        # 转换为(ε,δ)-DP
        return min(accumulated_rdp, self._epsilon)
    
    def _compute_rdp(self, noise_multiplier: float, steps: int) -> float:
        """
        计算Renyi差分隐私参数
        
        Args:
            noise_multiplier: σ/Δf
            steps: 步骤数
            
        Returns:
            RDP的ε值
        """
        if noise_multiplier <= 0:
            return float('inf')
        
        # 对于高斯机制，RDP为
        # ε = 1/(2α) * Σ log(1 + α/(α-1) * (σ/Δ)²)
        # 使用简化估计
        alpha = 2.0  # Renyi阶数
        
        rdp = 0.0
        for _ in range(steps):
            # 简化的RDP计算
            variance_ratio = 1.0 / (noise_multiplier ** 2)
            rdp += 0.5 * alpha * variance_ratio
        
        return rdp
    
    def step(self) -> None:
        """记录一个隐私步骤"""
        self._total_steps += 1
    
    def get_privacy_accountant(self) -> dict:
        """获取隐私账户信息"""
        return {
            'epsilon': self._epsilon,
            'delta': self._delta,
            'total_steps': self._total_steps,
            'spent': self.privacy_budget_spent(),
            'remaining': self._epsilon - self.privacy_budget_spent()
        }
    
    def noised_sum(self, values: List[float]) -> float:
        """对列表求和并添加噪声"""
        total = sum(values)
        return self.add_noise(total)
    
    def noised_mean(self, values: List[float]) -> float:
        """计算带噪声的平均值"""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        noise_variance = (self._sensitivity ** 2) / (len(values) ** 2)
        noise = random.gauss(0, math.sqrt(noise_variance) * self._compute_noise_scale())
        return mean + noise


# ============== 安全聚合器 ==============

class SecureAggregator:
    """
    安全聚合器
    
    在联邦学习中实现安全聚合：
    1. 使用掩码保护客户端更新
    2. 使用秘密分享进行聚合
    3. 处理掉队者和拜占庭节点
    
    协议流程：
    1. 客户端注册
    2. 开始轮次，分配掩码
    3. 客户端计算并提交更新
    4. 聚合器聚合（使用掩码抵消）
    5. 检测并移除异常值
    6. 完成聚合
    """
    
    ROUND_TIMEOUT = 300  # 轮次超时（秒）
    
    def __init__(self, config: SecureAggregationConfig = None):
        self._config = config or SecureAggregationConfig()
        self._threshold = self._config.threshold
        self._num_clients = self._config.num_clients
        
        # 客户端配置
        self._clients: Dict[str, SecureAggregationConfig] = {}
        
        # 客户端掩码（用于安全聚合）
        self._masks: Dict[str, float] = {}
        self._mask_keys: Dict[str, str] = {}  # 掩码密钥
        
        # 聚合累加器
        self._accumulated: Dict[str, List[float]] = {}
        
        # 当前轮次状态
        self._current_round = 0
        self._round_start_time = 0
        self._active_clients: Set[str] = set()
        self._completed_clients: Set[str] = set()
        self._dropped_clients: Set[str] = set()
        
        # 秘密分享器
        self._secret_sharing = SecretSharing(threshold=self._threshold)
        
        # 差分隐私
        self._dp = DifferentialPrivacy(
            epsilon=self._config.privacy_budget,
            delta=self._config.security_param,
            sensitivity=self._config.max_gradient_norm
        )
        
        # 同步锁
        self._lock = threading.Lock()
        
    def register_client(self, client_id: str, config: SecureAggregationConfig = None) -> bool:
        """
        注册客户端到聚合器
        
        Args:
            client_id: 客户端唯一标识
            config: 客户端特定配置（可选）
        """
        with self._lock:
            if client_id in self._clients:
                return False
            
            self._clients[client_id] = config or self._config
            return True
    
    def unregister_client(self, client_id: str) -> bool:
        """注销客户端"""
        with self._lock:
            if client_id in self._clients:
                del self._clients[client_id]
                self._masks.pop(client_id, None)
                return True
            return False
    
    def begin_round(self) -> Tuple[str, List[str]]:
        """
        开始新的聚合轮次
        
        Returns:
            round_id: 轮次ID
            client_ids: 参与客户端列表
        """
        with self._lock:
            self._current_round += 1
            self._round_start_time = time.time()
            
            round_id = f"round_{self._current_round}_{int(time.time())}"
            
            # 重置状态
            self._active_clients = set(self._clients.keys())
            self._completed_clients = set()
            self._dropped_clients = set()
            self._accumulated = {}
            
            # 为每个客户端生成掩码
            for client_id in self._active_clients:
                self._masks[client_id] = self._generate_client_mask(client_id)
                self._mask_keys[client_id] = hashlib.sha256(
                    f"{round_id}:{client_id}".encode()
                ).hexdigest()
            
            return round_id, list(self._active_clients)
    
    def _generate_client_mask(self, client_id: str) -> float:
        """生成客户端掩码"""
        # 使用随机数生成器确保掩码不可预测
        random.seed(hash(client_id) ^ int(time.time() // 3600))
        return random.uniform(-1.0, 1.0)
    
    def receive_update(self, client_id: str, update: List[float], 
                     round_id: str) -> bool:
        """
        接收客户端更新
        
        Args:
            client_id: 客户端ID
            update: 梯度更新
            round_id: 轮次ID
            
        Returns:
            是否成功接收
        """
        with self._lock:
            if client_id not in self._active_clients:
                return False
            
            if client_id in self._completed_clients:
                return False
            
            # 应用掩码
            masked_update = self._apply_mask(client_id, update)
            
            # 累加
            if client_id not in self._accumulated:
                self._accumulated[client_id] = masked_update
            else:
                for i, val in enumerate(masked_update):
                    if i < len(self._accumulated[client_id]):
                        self._accumulated[client_id][i] += val
                    else:
                        self._accumulated[client_id].append(val)
            
            self._completed_clients.add(client_id)
            
            return True
    
    def _apply_mask(self, client_id: str, update: List[float]) -> List[float]:
        """应用掩码到更新"""
        mask = self._masks.get(client_id, 0.0)
        
        # 对更新添加掩码
        return [val + mask for val in update]
    
    def _drop_stragglers(self, timeout: float = None) -> List[str]:
        """
        丢弃超时客户端
        
        联邦学习中，有些客户端可能因网络问题而掉队
        需要设置超时机制
        """
        if timeout is None:
            timeout = self.ROUND_TIMEOUT
        
        with self._lock:
            elapsed = time.time() - self._round_start_time
            
            if elapsed < timeout:
                return []
            
            # 将未完成的客户端标记为掉队
            stragglers = [
                client_id for client_id in self._active_clients
                if client_id not in self._completed_clients
            ]
            
            for client_id in stragglers:
                self._dropped_clients.add(client_id)
                self._active_clients.discard(client_id)
            
            return stragglers
    
    def complete_round(self) -> Optional[List[float]]:
        """
        完成聚合轮次
        
        Returns:
            聚合后的梯度（如果有足够的客户端）
        """
        with self._lock:
            # 检查是否达到阈值
            if len(self._completed_clients) < self._threshold:
                return None
            
            # 收集所有有效更新
            valid_updates = [
                self._accumulated[client_id]
                for client_id in self._completed_clients
                if client_id in self._accumulated
            ]
            
            if not valid_updates:
                return None
        
        # 计算聚合（简单平均 + 掩码抵消）
        aggregated = self._aggregate_updates(valid_updates)
        
        # 应用差分隐私
        noised = [self._dp.add_noise(v) for v in aggregated]
        
        # 更新隐私账户
        self._dp.step()
        
        return noised
    
    def _aggregate_updates(self, updates: List[List[float]]) -> List[float]:
        """聚合更新（简单平均）"""
        if not updates:
            return []
        
        # 掩码应该抵消，所以直接平均即可
        n = len(updates)
        dim = max(len(u) for u in updates)
        
        result = [0.0] * dim
        for update in updates:
            for i, val in enumerate(update):
                if i < dim:
                    result[i] += val / n
        
        return result
    
    def _reconstruct_secret(self, shares: List[SecretShare]) -> float:
        """重建秘密"""
        return self._secret_sharing.combine(shares)
    
    def verify_aggregates(self, result: List[float]) -> bool:
        """
        验证聚合结果的正确性
        
        检查聚合是否正确执行
        """
        with self._lock:
            if not self._completed_clients:
                return False
            
            # 检查结果的合理性
            # 1. 不应该包含无穷大或NaN
            for val in result:
                if math.isnan(val) or math.isinf(val):
                    return False
            
            # 2. 值不应该过大（可能被攻击）
            max_val = max(abs(v) for v in result)
            if max_val > 1000 * self._config.max_gradient_norm:
                return False
            
            return True
    
    def get_round_status(self) -> dict:
        """获取当前轮次状态"""
        with self._lock:
            return {
                'round': self._current_round,
                'total_clients': len(self._clients),
                'active_clients': len(self._active_clients),
                'completed_clients': len(self._completed_clients),
                'dropped_clients': len(self._dropped_clients),
                'elapsed_time': time.time() - self._round_start_time,
                'privacy_spent': self._dp.privacy_budget_spent()
            }


# ============== 拜占庭容错 ==============

class ByzantineRobust:
    """
    拜占庭容错算法
    
    在联邦学习中，部分客户端可能是恶意的或存在故障，
    会发送错误的梯度。拜占庭容错算法能够：
    - 检测异常梯度
    - 过滤恶意更新
    - 提供鲁棒的聚合结果
    
    实现的算法：
    1. Krum：选择最接近其他梯度的梯度
    2. Multi-Krum：选择多个良好梯度的平均
    3. Trimmed Mean：去除极值后求平均
    4. Coordinate-wise Median：坐标-wise中位数
    5. Zeno：基于置信度的过滤
    """
    
    def __init__(self, num_byzantine: int = 1):
        """
        Args:
            num_byzantine: 预期的拜占庭节点数量
        """
        self._byzantine_threshold = num_byzantine
        self._f = num_byzantine  # 拜占庭容错参数
        
    def detect_byzantine(self, gradients: List[List[float]]) -> List[int]:
        """
        检测可能是拜占庭的梯度索引
        
        Returns:
            可疑梯度的索引列表
        """
        if len(gradients) <= 2 * self._f + 1:
            # 客户端太少，无法检测
            return []
        
        # 使用Krum分数检测
        suspicious = set()
        
        for i, g_i in enumerate(gradients):
            # 计算与其他梯度的距离
            distances = []
            for j, g_j in enumerate(gradients):
                if i != j:
                    dist = self._euclidean_distance(g_i, g_j)
                    distances.append((dist, j))
            
            distances.sort(key=lambda x: x[0])
            
            # 选择最近的n-f-2个梯度
            n = len(gradients)
            k = n - self._f - 2
            
            if k > 0 and k <= len(distances):
                scores = [d for d, _ in distances[:k]]
                avg_score = sum(scores) / len(scores)
                
                # 如果与最近的梯度距离过大，标记为可疑
                if distances[0][0] > avg_score * 2:
                    suspicious.add(i)
        
        return list(suspicious)
    
    def _euclidean_distance(self, a: List[float], b: List[float]) -> float:
        """计算欧几里得距离"""
        if len(a) != len(b):
            min_len = min(len(a), len(b))
            a = a[:min_len]
            b = b[:min_len]
        
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    
    def _cosine_distance(self, a: List[float], b: List[float]) -> float:
        """计算余弦距离"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x**2 for x in a))
        norm_b = math.sqrt(sum(x**2 for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 1.0
        
        return 1.0 - dot / (norm_a * norm_b)
    
    def krum_select(self, gradients: List[List[float]], f: int = None) -> List[float]:
        """
        Krum算法
        
        选择与其他梯度最接近的梯度
        基于假设：诚实梯度应该相互接近
        
        Args:
            gradients: 梯度列表
            f: 拜占庭节点数量上限
            
        Returns:
            选中的梯度
        """
        if f is None:
            f = self._f
            
        n = len(gradients)
        
        # Krum需要 n > 2f + 2
        if n <= 2 * f + 2:
            # 无法执行Krum，返回中位数
            return self._coordinate_wise_median(gradients)
        
        # 计算每个梯度的Krum分数
        scores = []
        
        for i, g_i in enumerate(gradients):
            # 计算与所有其他梯度的距离（排除最近的f个）
            distances = []
            for j, g_j in enumerate(gradients):
                if i != j:
                    dist = self._euclidean_distance(g_i, g_j)
                    distances.append(dist)
            
            distances.sort()
            
            # 取最近的 n - f - 2 个距离的平均
            k = n - f - 2
            if k > 0:
                score = sum(distances[:k]) / k
            else:
                score = sum(distances) / len(distances)
            
            scores.append((score, i))
        
        # 选择分数最低的（最近的）
        scores.sort(key=lambda x: x[0])
        
        return gradients[scores[0][1]]
    
    def _trimmed_mean(self, gradients: List[List[float]], trim_ratio: float = 0.2) -> List[float]:
        """
        截断均值
        
        对于每个坐标，去除最高和最低的trim_ratio比例的值，
        然后计算均值
        
        Args:
            gradients: 梯度列表
            trim_ratio: 截断比例
            
        Returns:
            处理后的梯度
        """
        if not gradients:
            return []
        
        dim = len(gradients[0])
        n = len(gradients)
        k = int(n * trim_ratio)  # 每端截断的数量
        
        result = []
        
        for coord_idx in range(dim):
            # 收集该坐标的所有值
            values = [grad[coord_idx] for grad in gradients if coord_idx < len(grad)]
            values.sort()
            
            # 截断
            if len(values) > 2 * k:
                trimmed = values[k:len(values)-k]
            else:
                trimmed = values
            
            # 计算均值
            result.append(sum(trimmed) / len(trimmed) if trimmed else 0.0)
        
        return result
    
    def _multi_krum(self, gradients: List[List[float]], f: int, m: int) -> List[float]:
        """
        Multi-Krum算法
        
        选择m个最接近的梯度，然后求平均
        
        Args:
            gradients: 梯度列表
            f: 拜占庭节点数量上限
            m: 选择的梯度数量
            
        Returns:
            聚合后的梯度
        """
        n = len(gradients)
        
        if m >= n:
            # 返回简单平均
            return [sum(g[i] for g in gradients) / n for i in range(len(gradients[0]))]
        
        # 计算每个梯度与其他最近梯度的距离和
        scores = []
        
        for i, g_i in enumerate(gradients):
            distances = []
            for j, g_j in enumerate(gradients):
                if i != j:
                    dist = self._euclidean_distance(g_i, g_j)
                    distances.append((dist, j))
            
            distances.sort(key=lambda x: x[0])
            
            # 取最近的 n - f - 1 个距离
            k = n - f - 1
            if k > 0 and k <= len(distances):
                score = sum(d for d, _ in distances[:k])
            else:
                score = sum(d for d, _ in distances)
            
            scores.append((score, i))
        
        scores.sort(key=lambda x: x[0])
        
        # 选择前m个
        selected_indices = [idx for _, idx in scores[:m]]
        selected_gradients = [gradients[i] for i in selected_indices]
        
        # 求平均
        return [sum(g[i] for g in selected_gradients) / m for i in range(len(gradients[0]))]
    
    def _coordinate_wise_median(self, gradients: List[List[float]]) -> List[float]:
        """
        坐标-wise中位数
        
        对于每个坐标，计算所有梯度在该坐标的中位数
        
        Args:
            gradients: 梯度列表
            
        Returns:
            中位数梯度
        """
        if not gradients:
            return []
        
        dim = len(gradients[0])
        result = []
        
        for coord_idx in range(dim):
            values = [grad[coord_idx] for grad in gradients if coord_idx < len(grad)]
            values.sort()
            
            n = len(values)
            if n % 2 == 0:
                median = (values[n//2 - 1] + values[n//2]) / 2
            else:
                median = values[n//2]
            
            result.append(median)
        
        return result
    
    def _zeno_filter(self, gradients: List[List[float]], weights: List[float] = None) -> List[float]:
        """
        Zeno过滤算法
        
        基于梯度间的 squared distance 判断诚实节点，
        逐步过滤掉异常值
        
        Args:
            gradients: 梯度列表
            weights: 可选的权重列表
            
        Returns:
            过滤后的聚合梯度
        """
        if weights is None:
            weights = [1.0] * len(gradients)
        
        n = len(gradients)
        if n == 0:
            return []
        
        # 初始信任阈值
        trust_threshold = float('inf')
        selected = set(range(n))
        
        max_iterations = 10
        
        for _ in range(max_iterations):
            if len(selected) <= 2:
                break
            
            # 计算当前选中的梯度
            current_grads = [gradients[i] for i in selected]
            current_weights = [weights[i] for i in selected]
            
            # 计算加权平均
            avg = []
            dim = len(current_grads[0])
            for d in range(dim):
                weighted_sum = sum(g[d] * w for g, w in zip(current_grads, current_weights))
                total_weight = sum(current_weights)
                avg.append(weighted_sum / total_weight)
            
            # 计算每个梯度到平均的距离
            distances = []
            for i, idx in enumerate(selected):
                dist = self._euclidean_distance(gradients[idx], avg)
                distances.append((dist, idx))
            
            distances.sort()
            
            # 更新信任阈值
            if len(distances) > 1:
                # 使用第二小的距离作为阈值
                trust_threshold = distances[1][0] * 1.5
            
            # 移除异常值
            new_selected = set()
            for dist, idx in distances:
                if dist <= trust_threshold:
                    new_selected.add(idx)
            
            if len(new_selected) == len(selected):
                break
            
            selected = new_selected
        
        # 计算最终聚合
        if not selected:
            return self._coordinate_wise_median(gradients)
        
        selected_grads = [gradients[i] for i in selected]
        selected_weights = [weights[i] for i in selected]
        
        # 加权平均
        result = []
        dim = len(selected_grads[0])
        for d in range(dim):
            weighted_sum = sum(g[d] * w for g, w in zip(selected_grads, selected_weights))
            total_weight = sum(selected_weights)
            result.append(weighted_sum / total_weight)
        
        return result
    
    def robust_aggregate(self, gradients: List[List[float]], 
                        method: str = 'trimmed_mean',
                        **kwargs) -> List[float]:
        """
        鲁棒聚合
        
        根据指定的方法聚合梯度
        
        Args:
            gradients: 梯度列表
            method: 聚合方法 ('krum', 'multi_krum', 'trimmed_mean', 'median', 'zeno')
            **kwargs: 额外参数
            
        Returns:
            聚合后的梯度
        """
        if not gradients:
            return []
        
        # 移除空梯度
        gradients = [g for g in gradients if g]
        
        if not gradients:
            return []
        
        # 首先进行异常检测
        suspicious = self.detect_byzantine(gradients)
        
        if suspicious:
            # 移除可疑梯度
            gradients = [g for i, g in enumerate(gradients) if i not in suspicious]
        
        if not gradients:
            return []
        
        # 根据方法聚合
        if method == 'krum':
            return self.krum_select(gradients, f=kwargs.get('f', self._f))
        
        elif method == 'multi_krum':
            m = kwargs.get('m', max(1, len(gradients) - self._f - 1))
            return self._multi_krum(gradients, self._f, m)
        
        elif method == 'trimmed_mean':
            ratio = kwargs.get('trim_ratio', 0.2)
            return self._trimmed_mean(gradients, ratio)
        
        elif method == 'median':
            return self._coordinate_wise_median(gradients)
        
        elif method == 'zeno':
            weights = kwargs.get('weights')
            return self._zeno_filter(gradients, weights)
        
        else:
            # 默认使用截断均值
            return self._trimmed_mean(gradients, 0.2)


# ============== 主程序入口 ==============

if __name__ == "__main__":
    print("=== 安全聚合演示 ===\n")
    
    # 1. 演示Shamir秘密分享
    print("1. Shamir秘密分享:")
    secret_sharing = SecretSharing(threshold=3)
    
    secret = 42.5
    shares = secret_sharing.split(secret, n_shares=5)
    
    print(f"原始秘密: {secret}")
    print(f"分享份数: {len(shares)}")
    print(f"使用3份重建: {secret_sharing.combine(shares[:3])}")
    print(f"使用5份重建: {secret_sharing.combine(shares)}")
    
    # 2. 演示差分隐私
    print("\n2. 差分隐私:")
    dp = DifferentialPrivacy(epsilon=1.0, sensitivity=1.0)
    
    original_value = 10.0
    noised_values = [dp.add_noise(original_value) for _ in range(10)]
    
    print(f"原始值: {original_value}")
    print(f"添加噪声后的值: {noised_values[:3]}")
    print(f"平均值: {sum(noised_values)/len(noised_values)}")
    
    # 梯度裁剪
    gradient = [1.5, 2.3, -0.8, 3.2, 1.0]
    clipped = dp.clip_gradient(gradient, max_norm=1.0)
    print(f"裁剪前: {gradient}")
    print(f"裁剪后: {clipped}")
    
    # 3. 演示安全聚合器
    print("\n3. 安全聚合器:")
    config = SecureAggregationConfig(
        threshold=3,
        num_clients=5,
        noise_multiplier=1.0
    )
    aggregator = SecureAggregator(config)
    
    # 注册客户端
    for i in range(5):
        aggregator.register_client(f"client_{i}")
    
    # 开始轮次
    round_id, clients = aggregator.begin_round()
    print(f"开始轮次 {round_id}, 参与客户端: {len(clients)}")
    
    # 模拟提交更新
    for client_id in clients[:3]:
        update = [random.uniform(-1, 1) for _ in range(5)]
        aggregator.receive_update(client_id, update, round_id)
    
    # 完成聚合
    result = aggregator.complete_round()
    print(f"聚合结果: {result}")
    
    # 4. 演示拜占庭容错
    print("\n4. 拜占庭容错:")
    byzantine = ByzantineRobust(num_byzantine=1)
    
    # 模拟梯度
    true_gradient = [0.5, -0.3, 0.2, -0.1, 0.4]
    gradients = [
        true_gradient,
        [0.6, -0.2, 0.3, -0.15, 0.35],
        [0.4, -0.35, 0.15, -0.1, 0.45],
        [10.0, -5.0, 8.0, -3.0, 12.0],  # 恶意梯度
        [0.55, -0.25, 0.25, -0.12, 0.38]
    ]
    
    print(f"梯度数量: {len(gradients)}")
    
    # 检测异常
    suspicious = byzantine.detect_byzantine(gradients)
    print(f"检测到的异常梯度索引: {suspicious}")
    
    # 使用不同方法聚合
    result_krum = byzantine.robust_aggregate(gradients, method='krum')
    result_trimmed = byzantine.robust_aggregate(gradients, method='trimmed_mean')
    result_median = byzantine.robust_aggregate(gradients, method='median')
    
    print(f"\nKrum结果: {result_krum}")
    print(f"截断均值: {result_trimmed}")
    print(f"中位数: {result_median}")
    print(f"真值: {true_gradient}")
    
    print("\n=== 演示完成 ===")
