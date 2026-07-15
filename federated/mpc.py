"""
安全多方计算模块 (SPDZ协议实现)

提供基于SPDZ协议的安全多方计算功能，支持秘密共享、安全加法/乘法、
比较协议和重构协议等核心MPC操作。
"""

from __future__ import annotations

import random
import secrets
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor


class SecretSharingScheme(Enum):
    """秘密共享方案枚举"""
    SHAMIR = "shamir"          # Shamir秘密共享
    ADDITIVE = "additive"      # 加法秘密共享
    REPLICATED = "replicated"  # 复制秘密共享


@dataclass
class FieldElement:
    """
    有限域元素
    
    在素数域Z_p中进行运算，支持基本的算术操作。
    """
    value: int
    prime: int = field(default=2**61 - 1)  # 默认使用梅森素数
    
    def __post_init__(self):
        self.value = self.value % self.prime
    
    def __add__(self, other: Union[FieldElement, int]) -> FieldElement:
        if isinstance(other, FieldElement):
            if other.prime != self.prime:
                raise ValueError("不同素数的域元素不能相加")
            return FieldElement((self.value + other.value) % self.prime, self.prime)
        return FieldElement((self.value + other) % self.prime, self.prime)
    
    def __sub__(self, other: Union[FieldElement, int]) -> FieldElement:
        if isinstance(other, FieldElement):
            if other.prime != self.prime:
                raise ValueError("不同素数的域元素不能相减")
            return FieldElement((self.value - other.value) % self.prime, self.prime)
        return FieldElement((self.value - other) % self.prime, self.prime)
    
    def __mul__(self, other: Union[FieldElement, int]) -> FieldElement:
        if isinstance(other, FieldElement):
            if other.prime != self.prime:
                raise ValueError("不同素数的域元素不能相乘")
            return FieldElement((self.value * other.value) % self.prime, self.prime)
        return FieldElement((self.value * other) % self.prime, self.prime)
    
    def __truediv__(self, other: Union[FieldElement, int]) -> FieldElement:
        """除法通过乘以模逆元实现"""
        if isinstance(other, FieldElement):
            if other.prime != self.prime:
                raise ValueError("不同素数的域元素不能相除")
            other = other.value
        inv = pow(other, self.prime - 2, self.prime)
        return FieldElement((self.value * inv) % self.prime, self.prime)
    
    def __neg__(self) -> FieldElement:
        return FieldElement((-self.value) % self.prime, self.prime)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FieldElement):
            return False
        return self.value == other.value and self.prime == other.prime
    
    def __repr__(self) -> str:
        return f"FieldElement({self.value}, prime={self.prime})"
    
    def inverse(self) -> FieldElement:
        """计算模逆元"""
        return FieldElement(pow(self.value, self.prime - 2, self.prime), self.prime)


@dataclass
class Share:
    """
    秘密份额
    
    表示秘密共享后的单个份额，包含份额值、持有者ID和元数据。
    """
    value: FieldElement
    holder_id: int
    share_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __add__(self, other: Union[Share, FieldElement, int]) -> Share:
        """份额加法（本地操作）"""
        if isinstance(other, Share):
            if other.holder_id != self.holder_id:
                raise ValueError("不同持有者的份额不能直接相加")
            return Share(
                value=self.value + other.value,
                holder_id=self.holder_id,
                share_index=self.share_index,
                metadata={**self.metadata, **other.metadata}
            )
        return Share(
            value=self.value + other,
            holder_id=self.holder_id,
            share_index=self.share_index,
            metadata=self.metadata.copy()
        )
    
    def __mul__(self, scalar: Union[FieldElement, int]) -> Share:
        """份额与标量乘法（本地操作）"""
        return Share(
            value=self.value * scalar,
            holder_id=self.holder_id,
            share_index=self.share_index,
            metadata=self.metadata.copy()
        )
    
    def __repr__(self) -> str:
        return f"Share(holder={self.holder_id}, value={self.value.value})"


@dataclass
class Triple:
    """
    乘法三元组 (Beaver Triple)
    
    用于安全乘法的预计算三元组 (a, b, c)，其中 c = a * b。
    每个参与方持有三元组的份额。
    """
    a: Share
    b: Share
    c: Share
    
    def validate(self) -> bool:
        """验证三元组一致性"""
        return (self.a.holder_id == self.b.holder_id == self.c.holder_id)


class SecretSharing(ABC):
    """秘密共享基类"""
    
    def __init__(self, num_parties: int, threshold: int, prime: int = 2**61 - 1):
        self.num_parties = num_parties
        self.threshold = threshold
        self.prime = prime
    
    @abstractmethod
    def share(self, secret: Union[int, FieldElement], 
              party_ids: Optional[List[int]] = None) -> List[Share]:
        """将秘密分割为份额"""
        pass
    
    @abstractmethod
    def reconstruct(self, shares: List[Share]) -> FieldElement:
        """从份额重构秘密"""
        pass


class ShamirSecretSharing(SecretSharing):
    """
    Shamir秘密共享实现
    
    基于(t, n)门限方案，任意t个份额可重构秘密，少于t个份额
    无法获得任何信息。
    """
    
    def __init__(self, num_parties: int, threshold: int, prime: int = 2**61 - 1):
        super().__init__(num_parties, threshold, prime)
        if threshold > num_parties:
            raise ValueError("阈值不能大于参与方数量")
    
    def _evaluate_polynomial(self, coefficients: List[int], x: int) -> int:
        """在点x处求多项式的值"""
        result = 0
        power = 1
        for coeff in coefficients:
            result = (result + coeff * power) % self.prime
            power = (power * x) % self.prime
        return result
    
    def share(self, secret: Union[int, FieldElement],
              party_ids: Optional[List[int]] = None) -> List[Share]:
        """
        使用Shamir方案分割秘密
        
        构造t-1次随机多项式，f(0) = secret
        """
        if isinstance(secret, FieldElement):
            secret = secret.value
        
        party_ids = party_ids or list(range(1, self.num_parties + 1))
        
        # 生成随机系数（t-1次多项式）
        coefficients = [secret] + [
            secrets.randbelow(self.prime) 
            for _ in range(self.threshold - 1)
        ]
        
        shares = []
        for party_id in party_ids:
            share_value = self._evaluate_polynomial(coefficients, party_id)
            shares.append(Share(
                value=FieldElement(share_value, self.prime),
                holder_id=party_id,
                share_index=party_id
            ))
        
        return shares
    
    def reconstruct(self, shares: List[Share]) -> FieldElement:
        """
        使用Lagrange插值重构秘密
        
        需要至少threshold个份额。
        """
        if len(shares) < self.threshold:
            raise ValueError(f"需要至少{self.threshold}个份额，但只有{len(shares)}个")
        
        # Lagrange插值在x=0处的值
        secret = 0
        for i, share_i in enumerate(shares[:self.threshold]):
            x_i = share_i.share_index
            y_i = share_i.value.value
            
            # 计算Lagrange基多项式在0处的值
            lagrange_basis = 1
            for j, share_j in enumerate(shares[:self.threshold]):
                if i != j:
                    x_j = share_j.share_index
                    # L_i(0) = x_j / (x_j - x_i)
                    numerator = x_j
                    denominator = (x_j - x_i) % self.prime
                    lagrange_basis = (lagrange_basis * numerator * 
                                    pow(denominator, self.prime - 2, self.prime)) % self.prime
            
            secret = (secret + y_i * lagrange_basis) % self.prime
        
        return FieldElement(secret, self.prime)


class AdditiveSecretSharing(SecretSharing):
    """
    加法秘密共享实现
    
    秘密s被分割为n个随机份额，满足 s = sum(shares)。
    所有参与方必须协作才能重构秘密。
    """
    
    def __init__(self, num_parties: int, prime: int = 2**61 - 1):
        super().__init__(num_parties, num_parties, prime)
    
    def share(self, secret: Union[int, FieldElement],
              party_ids: Optional[List[int]] = None) -> List[Share]:
        """分割秘密为加法份额"""
        if isinstance(secret, FieldElement):
            secret = secret.value
        
        party_ids = party_ids or list(range(self.num_parties))
        
        # 生成n-1个随机份额
        shares = []
        sum_shares = 0
        
        for i in range(self.num_parties - 1):
            share_value = secrets.randbelow(self.prime)
            shares.append(Share(
                value=FieldElement(share_value, self.prime),
                holder_id=party_ids[i],
                share_index=i
            ))
            sum_shares = (sum_shares + share_value) % self.prime
        
        # 最后一个份额使得总和等于秘密
        last_share = (secret - sum_shares) % self.prime
        shares.append(Share(
            value=FieldElement(last_share, self.prime),
            holder_id=party_ids[-1],
            share_index=self.num_parties - 1
        ))
        
        return shares
    
    def reconstruct(self, shares: List[Share]) -> FieldElement:
        """通过求和重构秘密"""
        if len(shares) < self.num_parties:
            raise ValueError(f"需要所有{self.num_parties}个份额")
        
        secret = sum(share.value.value for share in shares) % self.prime
        return FieldElement(secret, self.prime)


class MPCParty:
    """
    MPC参与方
    
    表示MPC协议中的一个参与方，管理其持有的份额和通信。
    """
    
    def __init__(self, party_id: int, prime: int = 2**61 - 1):
        self.party_id = party_id
        self.prime = prime
        self.shares: Dict[str, Share] = {}
        self.triples: List[Triple] = []
        self.message_queue: List[Dict[str, Any]] = []
    
    def store_share(self, name: str, share: Share) -> None:
        """存储份额"""
        self.shares[name] = share
    
    def get_share(self, name: str) -> Share:
        """获取份额"""
        if name not in self.shares:
            raise KeyError(f"未找到份额: {name}")
        return self.shares[name]
    
    def generate_triple(self, ss: SecretSharing) -> Triple:
        """生成乘法三元组份额"""
        a = secrets.randbelow(self.prime)
        b = secrets.randbelow(self.prime)
        c = (a * b) % self.prime
        
        a_shares = ss.share(a, [self.party_id])
        b_shares = ss.share(b, [self.party_id])
        c_shares = ss.share(c, [self.party_id])
        
        triple = Triple(
            a=a_shares[0],
            b=b_shares[0],
            c=c_shares[0]
        )
        self.triples.append(triple)
        return triple


class SPDZProtocol:
    """
    SPDZ协议实现
    
    提供安全多方计算的核心功能，包括：
    - 秘密共享
    - 安全加法/乘法
    - 比较协议
    - 重构协议
    """
    
    def __init__(self, num_parties: int, threshold: Optional[int] = None,
                 scheme: SecretSharingScheme = SecretSharingScheme.SHAMIR,
                 prime: int = 2**61 - 1):
        self.num_parties = num_parties
        self.threshold = threshold or num_parties
        self.scheme_type = scheme
        self.prime = prime
        
        # 初始化秘密共享方案
        if scheme == SecretSharingScheme.SHAMIR:
            self.secret_sharing = ShamirSecretSharing(num_parties, self.threshold, prime)
        elif scheme == SecretSharingScheme.ADDITIVE:
            self.secret_sharing = AdditiveSecretSharing(num_parties, prime)
        else:
            raise ValueError(f"未支持的共享方案: {scheme}")
        
        # 初始化参与方
        self.parties: Dict[int, MPCParty] = {
            i: MPCParty(i, prime) for i in range(num_parties)
        }
        
        # 预计算三元组池
        self.triple_pool: List[List[Triple]] = []
    
    def share_secret(self, secret: Union[int, FieldElement], 
                     name: str) -> List[Share]:
        """
        将秘密共享给所有参与方
        
        Returns:
            所有参与方的份额列表
        """
        shares = self.secret_sharing.share(secret)
        
        # 分发给各参与方
        for share in shares:
            self.parties[share.holder_id].store_share(name, share)
        
        return shares
    
    def secure_add(self, share_a: Share, share_b: Share) -> Share:
        """
        安全加法
        
        本地操作，无需通信。
        [x] + [y] = [x + y]
        """
        if share_a.holder_id != share_b.holder_id:
            raise ValueError("份额持有者不一致")
        
        return Share(
            value=share_a.value + share_b.value,
            holder_id=share_a.holder_id,
            share_index=share_a.share_index,
            metadata={**share_a.metadata, **share_b.metadata, 'op': 'add'}
        )
    
    def secure_sub(self, share_a: Share, share_b: Share) -> Share:
        """
        安全减法
        
        本地操作，无需通信。
        [x] - [y] = [x - y]
        """
        if share_a.holder_id != share_b.holder_id:
            raise ValueError("份额持有者不一致")
        
        return Share(
            value=share_a.value - share_b.value,
            holder_id=share_a.holder_id,
            share_index=share_a.share_index,
            metadata={**share_a.metadata, **share_b.metadata, 'op': 'sub'}
        )
    
    def secure_mul_const(self, share: Share, constant: Union[int, FieldElement]) -> Share:
        """
        安全标量乘法
        
        本地操作，无需通信。
        c * [x] = [c * x]
        """
        return Share(
            value=share.value * constant,
            holder_id=share.holder_id,
            share_index=share.share_index,
            metadata={**share.metadata, 'op': 'mul_const'}
        )
    
    def generate_multiplication_triples(self, count: int = 10) -> None:
        """
        预生成乘法三元组
        
        使用离线阶段生成Beaver三元组，供在线乘法使用。
        """
        self.triple_pool = []
        
        for _ in range(count):
            triple_set = []
            for party_id in range(self.num_parties):
                party = self.parties[party_id]
                triple = party.generate_triple(self.secret_sharing)
                triple_set.append(triple)
            self.triple_pool.append(triple_set)
    
    def secure_mul(self, share_a: Share, share_b: Share, 
                   triple_idx: int = 0) -> Share:
        """
        安全乘法 (使用Beaver三元组)
        
        基于预计算三元组(a, b, c)实现安全乘法：
        1. 计算 [d] = [x] - [a], [e] = [y] - [b]
        2. 重构 d, e
        3. 计算 [z] = [c] + d*[b] + e*[a] + d*e
        
        需要一轮通信重构d和e。
        """
        if share_a.holder_id != share_b.holder_id:
            raise ValueError("份额持有者不一致")
        
        party_id = share_a.holder_id
        party = self.parties[party_id]
        
        # 获取三元组
        if triple_idx >= len(self.triple_pool):
            raise ValueError("三元组池耗尽")
        
        triple = self.triple_pool[triple_idx][party_id]
        
        # 计算[d]和[e]
        d_share = self.secure_sub(share_a, triple.a)
        e_share = self.secure_sub(share_b, triple.b)
        
        # 模拟重构d和e（实际实现需要通信）
        d = d_share.value
        e = e_share.value
        
        # 计算[z] = [c] + d*[b] + e*[a] + d*e
        term1 = triple.c
        term2 = triple.b * d
        term3 = triple.a * e
        term4 = FieldElement((d.value * e.value) % self.prime, self.prime)
        
        result = Share(
            value=term1.value + term2.value + term3.value + term4.value,
            holder_id=party_id,
            share_index=share_a.share_index,
            metadata={'op': 'mul'}
        )
        
        return result
    
    def secure_matmul(self, shares_a: List[List[Share]], 
                      shares_b: List[List[Share]]) -> List[List[Share]]:
        """
        安全矩阵乘法
        
        对共享的矩阵进行安全乘法运算。
        """
        rows_a = len(shares_a)
        cols_a = len(shares_a[0])
        cols_b = len(shares_b[0])
        
        result = []
        triple_idx = 0
        
        for i in range(rows_a):
            row = []
            for j in range(cols_b):
                # 计算C[i,j] = sum(A[i,k] * B[k,j])
                acc = None
                for k in range(cols_a):
                    prod = self.secure_mul(shares_a[i][k], shares_b[k][j], triple_idx)
                    triple_idx += 1
                    
                    if acc is None:
                        acc = prod
                    else:
                        acc = self.secure_add(acc, prod)
                
                row.append(acc)
            result.append(row)
        
        return result
    
    def secure_compare(self, share_a: Share, share_b: Share) -> Share:
        """
        安全比较协议
        
        比较两个共享值的大小，返回[x > y]的共享结果。
        使用位分解和按位比较实现。
        
        注意：这是简化实现，完整实现需要更复杂的协议。
        """
        # 简化实现：计算差值并检查符号位
        diff = self.secure_sub(share_a, share_b)
        
        # 实际实现需要位分解和按位操作
        # 这里返回差值的符号信息（简化）
        return Share(
            value=FieldElement(1 if diff.value.value > self.prime // 2 else 0, self.prime),
            holder_id=share_a.holder_id,
            share_index=share_a.share_index,
            metadata={'op': 'compare'}
        )
    
    def secure_max(self, shares: List[Share]) -> Share:
        """
        安全最大值
        
        在一组共享值中找到最大值。
        使用锦标赛方式比较。
        """
        if not shares:
            raise ValueError("空列表")
        
        current_max = shares[0]
        for share in shares[1:]:
            # 比较并选择较大值
            cmp = self.secure_compare(share, current_max)
            # 简化：实际实现需要条件选择
            # 这里假设cmp=1表示share > current_max
            if cmp.value.value == 1:
                current_max = share
        
        return current_max
    
    def reconstruct(self, shares: List[Share]) -> FieldElement:
        """
        重构协议
        
        从份额中重构原始值。
        """
        return self.secret_sharing.reconstruct(shares)
    
    def batch_reconstruct(self, share_groups: List[List[Share]]) -> List[FieldElement]:
        """
        批量重构
        
        同时重构多个秘密。
        """
        return [self.reconstruct(shares) for shares in share_groups]
    
    def verify_integrity(self, shares: List[Share], expected: FieldElement) -> bool:
        """
        验证份额完整性
        
        检查份额是否能正确重构预期值。
        """
        try:
            reconstructed = self.reconstruct(shares)
            return reconstructed == expected
        except Exception:
            return False
    
    def generate_random_share(self, name: str) -> List[Share]:
        """
        生成随机共享值
        
        各参与方协作生成一个共享的随机数。
        """
        # 每个参与方生成随机份额
        random_shares = []
        for party_id in range(self.num_parties):
            random_value = secrets.randbelow(self.prime)
            share = Share(
                value=FieldElement(random_value, self.prime),
                holder_id=party_id,
                share_index=party_id
            )
            self.parties[party_id].store_share(name, share)
            random_shares.append(share)
        
        return random_shares
    
    def input_masking(self, input_value: Union[int, FieldElement], 
                      party_id: int) -> Tuple[List[Share], FieldElement]:
        """
        输入掩码
        
        对输入值进行掩码处理以保护隐私。
        Returns:
            (掩码后的共享值, 掩码值)
        """
        # 生成随机掩码
        mask = secrets.randbelow(self.prime)
        masked_input = (input_value + mask) % self.prime
        
        # 共享掩码值
        mask_shares = self.secret_sharing.share(mask)
        for share in mask_shares:
            self.parties[share.holder_id].store_share(f"mask_{party_id}", share)
        
        return mask_shares, FieldElement(masked_input, self.prime)
    
    def compute_circuit(self, circuit: List[Dict[str, Any]], 
                       inputs: Dict[str, List[Share]]) -> Dict[str, List[Share]]:
        """
        执行安全计算电路
        
        执行由基本门组成的计算电路。
        
        Args:
            circuit: 电路描述，每个门包含类型和输入
            inputs: 输入变量的份额
        
        Returns:
            输出变量的份额
        """
        outputs = {}
        triple_idx = 0
        
        for gate in circuit:
            gate_type = gate['type']
            input_names = gate['inputs']
            output_name = gate['output']
            
            if gate_type == 'input':
                # 输入门
                outputs[output_name] = inputs[output_name]
            
            elif gate_type == 'add':
                # 加法门
                a_shares = outputs.get(input_names[0]) or inputs[input_names[0]]
                b_shares = outputs.get(input_names[1]) or inputs[input_names[1]]
                result = [self.secure_add(a, b) for a, b in zip(a_shares, b_shares)]
                outputs[output_name] = result
            
            elif gate_type == 'mul':
                # 乘法门
                a_shares = outputs.get(input_names[0]) or inputs[input_names[0]]
                b_shares = outputs.get(input_names[1]) or inputs[input_names[1]]
                result = []
                for a, b in zip(a_shares, b_shares):
                    res = self.secure_mul(a, b, triple_idx)
                    triple_idx += 1
                    result.append(res)
                outputs[output_name] = result
            
            elif gate_type == 'const_mul':
                # 常数乘法
                a_shares = outputs.get(input_names[0]) or inputs[input_names[0]]
                const = gate['constant']
                result = [self.secure_mul_const(a, const) for a in a_shares]
                outputs[output_name] = result
        
        return outputs


class SecureAggregator:
    """
    安全聚合器
    
    用于联邦学习中的安全模型聚合。
    """
    
    def __init__(self, num_parties: int, prime: int = 2**61 - 1):
        self.num_parties = num_parties
        self.prime = prime
        self.spdz = SPDZProtocol(num_parties, scheme=SecretSharingScheme.ADDITIVE, prime=prime)
    
    def aggregate_gradients(self, gradients_list: List[List[np.ndarray]]) -> List[np.ndarray]:
        """
        安全聚合梯度
        
        Args:
            gradients_list: 各参与方的梯度列表
        
        Returns:
            聚合后的梯度
        """
        num_layers = len(gradients_list[0])
        aggregated = []
        
        for layer_idx in range(num_layers):
            # 获取该层的所有梯度
            layer_grads = [g[layer_idx] for g in gradients_list]
            
            # 将梯度转换为整数表示
            flat_grads = [g.flatten() for g in layer_grads]
            
            # 对每个参数进行安全聚合
            aggregated_flat = []
            for param_idx in range(len(flat_grads[0])):
                # 共享参数值
                param_values = [int(g[param_idx] * 10000) % self.prime for g in flat_grads]
                
                # 各参与方共享其值
                all_shares = []
                for party_id, value in enumerate(param_values):
                    shares = self.spdz.share_secret(value, f"grad_{layer_idx}_{param_idx}_{party_id}")
                    all_shares.append(shares)
                
                # 聚合（求和）
                sum_shares = all_shares[0]
                for shares in all_shares[1:]:
                    sum_shares = [self.spdz.secure_add(s1, s2) for s1, s2 in zip(sum_shares, shares)]
                
                # 重构
                aggregated_value = self.spdz.reconstruct(sum_shares)
                aggregated_flat.append(aggregated_value.value / 10000.0)
            
            # 恢复形状
            original_shape = layer_grads[0].shape
            aggregated_layer = np.array(aggregated_flat).reshape(original_shape)
            aggregated.append(aggregated_layer)
        
        return aggregated
    
    def secure_average(self, values: List[int]) -> float:
        """
        安全平均值计算
        
        计算共享值的平均值而不泄露个体值。
        """
        # 共享所有值
        all_shares = []
        for party_id, value in enumerate(values):
            shares = self.spdz.share_secret(value, f"val_{party_id}")
            all_shares.append(shares)
        
        # 求和
        sum_shares = all_shares[0]
        for shares in all_shares[1:]:
            sum_shares = [self.spdz.secure_add(s1, s2) for s1, s2 in zip(sum_shares, shares)]
        
        # 除以数量（乘以模逆元）
        inv_n = FieldElement(pow(len(values), self.prime - 2, self.prime), self.prime)
        avg_shares = [self.spdz.secure_mul_const(s, inv_n) for s in sum_shares]
        
        # 重构
        result = self.spdz.reconstruct(avg_shares)
        return float(result.value)


# 便捷函数
def secure_computation_demo():
    """
    MPC演示函数
    
    演示基本的MPC操作。
    """
    print("=== SPDZ安全多方计算演示 ===\n")
    
    # 初始化3方SPDZ协议
    num_parties = 3
    spdz = SPDZProtocol(num_parties, threshold=2, scheme=SecretSharingScheme.SHAMIR)
    
    # 生成乘法三元组
    print("生成乘法三元组...")
    spdz.generate_multiplication_triples(10)
    print(f"生成了 {len(spdz.triple_pool)} 个三元组\n")
    
    # 秘密共享
    secret_a = 42
    secret_b = 17
    
    print(f"秘密a = {secret_a}")
    print(f"秘密b = {secret_b}\n")
    
    shares_a = spdz.share_secret(secret_a, "a")
    shares_b = spdz.share_secret(secret_b, "b")
    
    print(f"a的份额: {shares_a}")
    print(f"b的份额: {shares_b}\n")
    
    # 安全加法
    print("执行安全加法...")
    add_shares = [spdz.secure_add(shares_a[i], shares_b[i]) for i in range(num_parties)]
    add_result = spdz.reconstruct(add_shares)
    print(f"[a] + [b] = {add_result.value} (期望: {secret_a + secret_b})\n")
    
    # 安全乘法
    print("执行安全乘法...")
    mul_shares = [spdz.secure_mul(shares_a[i], shares_b[i], 0) for i in range(num_parties)]
    mul_result = spdz.reconstruct(mul_shares)
    print(f"[a] * [b] = {mul_result.value} (期望: {secret_a * secret_b})\n")
    
    # 安全比较
    print("执行安全比较...")
    cmp_shares = [spdz.secure_compare(shares_a[i], shares_b[i]) for i in range(num_parties)]
    cmp_result = spdz.reconstruct(cmp_shares)
    print(f"[a > b] = {cmp_result.value} (期望: {1 if secret_a > secret_b else 0})\n")
    
    print("=== 演示完成 ===")
    
    return spdz


if __name__ == "__main__":
    secure_computation_demo()
