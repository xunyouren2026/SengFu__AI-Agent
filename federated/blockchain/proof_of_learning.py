"""
Proof of Learning - 学习证明
联邦学习中的训练证明协议

实现学习证明（Proof of Learning, PoL）机制，确保参与者
确实进行了有效的模型训练。包含挑战生成、证明验证、
难度调整和证明链管理。

Author: AGI Unified Framework
"""

import hashlib
import json
import time
import math
import threading
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from collections import deque


# ============== 学习证明挑战 ==============

@dataclass
class PoLChallenge:
    """
    学习证明挑战

    由验证者生成的训练挑战，包含模型版本、数据样本和难度参数。
    参与者需要在指定时间内完成训练并提交证明。

    Attributes:
        challenge_id: 挑战唯一标识
        model_version: 模型版本标识
        data_samples: 挑战数据样本（特征子集）
        difficulty: 难度级别（1-10）
        target_loss: 目标loss值（需要低于此值）
        max_gradient_norm: 最大允许梯度范数
        time_limit: 时间限制（秒）
        seed: 随机种子（确保可复现）
        created_at: 创建时间
        expires_at: 过期时间
    """
    challenge_id: str
    model_version: str
    data_samples: List[Dict[str, Any]]
    difficulty: int = 5
    target_loss: float = 0.5
    max_gradient_norm: float = 10.0
    time_limit: float = 300.0
    seed: int = 0
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0

    def __post_init__(self):
        if not self.seed:
            self.seed = int(self.created_at * 1000) % (2**32)
        if not self.expires_at:
            self.expires_at = self.created_at + self.time_limit

    @property
    def is_expired(self) -> bool:
        """挑战是否已过期"""
        return time.time() > self.expires_at

    @property
    def difficulty_multiplier(self) -> float:
        """难度乘数（影响验证严格程度）"""
        return 1.0 + (self.difficulty - 1) * 0.15

    def compute_hash(self) -> str:
        """计算挑战哈希"""
        content = json.dumps({
            'challenge_id': self.challenge_id,
            'model_version': self.model_version,
            'difficulty': self.difficulty,
            'target_loss': self.target_loss,
            'seed': self.seed,
            'created_at': self.created_at
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            'challenge_id': self.challenge_id,
            'model_version': self.model_version,
            'data_samples': self.data_samples,
            'difficulty': self.difficulty,
            'target_loss': self.target_loss,
            'max_gradient_norm': self.max_gradient_norm,
            'time_limit': self.time_limit,
            'seed': self.seed,
            'created_at': self.created_at,
            'expires_at': self.expires_at,
            'hash': self.compute_hash()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PoLChallenge':
        """从字典反序列化"""
        return cls(
            challenge_id=data['challenge_id'],
            model_version=data['model_version'],
            data_samples=data['data_samples'],
            difficulty=data.get('difficulty', 5),
            target_loss=data.get('target_loss', 0.5),
            max_gradient_norm=data.get('max_gradient_norm', 10.0),
            time_limit=data.get('time_limit', 300.0),
            seed=data.get('seed', 0),
            created_at=data.get('created_at', time.time()),
            expires_at=data.get('expires_at', 0.0)
        )


# ============== 训练证明 ==============

@dataclass
class TrainingProof:
    """
    训练证明

    参与者提交的训练证明，包含训练结果和签名。
    验证者通过检查loss值、梯度范数、训练时间等指标
    来验证训练是否有效。

    Attributes:
        proof_id: 证明唯一标识
        challenge_id: 对应的挑战ID
        participant: 参与者标识
        model_version: 模型版本
        initial_loss: 初始loss值
        final_loss: 最终loss值
        gradient_norm: 梯度范数
        training_time: 训练耗时（秒）
        iterations: 训练迭代次数
        data_hash: 使用数据的哈希
        model_hash: 训练后模型的哈希
        timestamp: 提交时间
        signature: 参与者签名
    """
    proof_id: str
    challenge_id: str
    participant: str
    model_version: str
    initial_loss: float
    final_loss: float
    gradient_norm: float
    training_time: float
    iterations: int
    data_hash: str = ""
    model_hash: str = ""
    timestamp: float = field(default_factory=time.time)
    signature: str = ""

    @property
    def loss_improvement(self) -> float:
        """loss改善量"""
        return self.initial_loss - self.final_loss

    @property
    def loss_improvement_ratio(self) -> float:
        """loss改善比例"""
        if self.initial_loss <= 0:
            return 0.0
        return self.loss_improvement / self.initial_loss

    @property
    def training_speed(self) -> float:
        """训练速度（迭代/秒）"""
        if self.training_time <= 0:
            return 0.0
        return self.iterations / self.training_time

    def compute_hash(self) -> str:
        """计算证明哈希"""
        content = json.dumps({
            'proof_id': self.proof_id,
            'challenge_id': self.challenge_id,
            'participant': self.participant,
            'model_version': self.model_version,
            'initial_loss': self.initial_loss,
            'final_loss': self.final_loss,
            'gradient_norm': self.gradient_norm,
            'training_time': self.training_time,
            'iterations': self.iterations,
            'timestamp': self.timestamp
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def sign(self, private_key: str) -> str:
        """签名证明"""
        proof_hash = self.compute_hash()
        # 简化的签名：HMAC-SHA256
        import hmac
        signature = hmac.new(
            private_key.encode(),
            proof_hash.encode(),
            hashlib.sha256
        ).hexdigest()
        self.signature = signature
        return signature

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            'proof_id': self.proof_id,
            'challenge_id': self.challenge_id,
            'participant': self.participant,
            'model_version': self.model_version,
            'initial_loss': self.initial_loss,
            'final_loss': self.final_loss,
            'gradient_norm': self.gradient_norm,
            'training_time': self.training_time,
            'iterations': self.iterations,
            'data_hash': self.data_hash,
            'model_hash': self.model_hash,
            'timestamp': self.timestamp,
            'signature': self.signature,
            'hash': self.compute_hash()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingProof':
        """从字典反序列化"""
        return cls(
            proof_id=data['proof_id'],
            challenge_id=data['challenge_id'],
            participant=data['participant'],
            model_version=data['model_version'],
            initial_loss=data['initial_loss'],
            final_loss=data['final_loss'],
            gradient_norm=data['gradient_norm'],
            training_time=data['training_time'],
            iterations=data['iterations'],
            data_hash=data.get('data_hash', ''),
            model_hash=data.get('model_hash', ''),
            timestamp=data.get('timestamp', time.time()),
            signature=data.get('signature', '')
        )


# ============== 证明验证器 ==============

class ProofVerifier:
    """
    证明验证器

    验证训练证明的有效性，包含多个维度的检查：
    1. Loss验证：loss是否合理下降
    2. 梯度验证：梯度范数是否合理
    3. 时间验证：训练时间是否合理
    4. 签名验证：证明是否被篡改

    Author: AGI Unified Framework
    """

    def __init__(self):
        self._verification_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def verify(self, proof: TrainingProof,
               challenge: PoLChallenge) -> Dict[str, Any]:
        """
        完整验证训练证明

        Args:
            proof: 训练证明
            challenge: 对应的挑战

        Returns:
            验证结果字典
        """
        results: Dict[str, Any] = {
            'valid': True,
            'proof_id': proof.proof_id,
            'challenge_id': challenge.challenge_id,
            'participant': proof.participant,
            'checks': {},
            'score': 0.0,
            'timestamp': time.time()
        }

        # 1. 验证loss
        loss_result = self.verify_loss(proof, challenge)
        results['checks']['loss'] = loss_result
        if not loss_result['passed']:
            results['valid'] = False

        # 2. 验证梯度
        gradient_result = self.verify_gradient(proof, challenge)
        results['checks']['gradient'] = gradient_result
        if not gradient_result['passed']:
            results['valid'] = False

        # 3. 验证时间
        time_result = self.verify_time(proof, challenge)
        results['checks']['time'] = time_result
        if not time_result['passed']:
            results['valid'] = False

        # 4. 验证签名
        sig_result = self.verify_signature(proof)
        results['checks']['signature'] = sig_result
        if not sig_result['passed']:
            results['valid'] = False

        # 计算综合得分
        results['score'] = self._compute_score(results['checks'])

        # 记录验证历史
        with self._lock:
            self._verification_history.append(results)

        return results

    def verify_loss(self, proof: TrainingProof,
                    challenge: PoLChallenge) -> Dict[str, Any]:
        """
        验证loss是否合理下降

        检查项：
        - final_loss是否低于target_loss
        - loss是否有合理下降
        - loss值是否在合理范围内

        Args:
            proof: 训练证明
            challenge: 对应的挑战

        Returns:
            验证结果
        """
        passed = True
        reasons: List[str] = []

        # 检查final_loss是否达标
        multiplier = challenge.difficulty_multiplier
        effective_target = challenge.target_loss * multiplier

        if proof.final_loss > effective_target:
            passed = False
            reasons.append(
                f"Final loss {proof.final_loss:.4f} > target {effective_target:.4f}"
            )

        # 检查loss是否有下降
        if proof.final_loss >= proof.initial_loss:
            passed = False
            reasons.append("Loss did not decrease during training")

        # 检查loss改善是否合理（不能太夸张）
        if proof.loss_improvement_ratio > 0.99:
            passed = False
            reasons.append("Loss improvement too extreme (>99%)")

        # 检查loss值范围
        if proof.final_loss < 0 or proof.initial_loss < 0:
            passed = False
            reasons.append("Negative loss values")

        if proof.final_loss > 100 or proof.initial_loss > 100:
            passed = False
            reasons.append("Loss values too large")

        return {
            'passed': passed,
            'reasons': reasons,
            'initial_loss': proof.initial_loss,
            'final_loss': proof.final_loss,
            'improvement': proof.loss_improvement,
            'improvement_ratio': proof.loss_improvement_ratio
        }

    def verify_gradient(self, proof: TrainingProof,
                        challenge: PoLChallenge) -> Dict[str, Any]:
        """
        验证梯度范数是否合理

        检查项：
        - 梯度范数是否在合理范围内
        - 梯度范数是否不超过最大值
        - 梯度范数与loss改善的关系是否合理

        Args:
            proof: 训练证明
            challenge: 对应的挑战

        Returns:
            验证结果
        """
        passed = True
        reasons: List[str] = []

        # 检查梯度范数上限
        if proof.gradient_norm > challenge.max_gradient_norm:
            passed = False
            reasons.append(
                f"Gradient norm {proof.gradient_norm:.4f} > max {challenge.max_gradient_norm:.4f}"
            )

        # 检查梯度范数下限（不能为0，除非完全收敛）
        if proof.gradient_norm < 1e-8 and proof.final_loss > 0.01:
            passed = False
            reasons.append("Gradient norm too small (possible fake)")

        # 检查梯度范数与loss的关系
        # 在正常训练中，梯度范数应与loss正相关
        if proof.final_loss > 0 and proof.gradient_norm > 0:
            ratio = proof.gradient_norm / max(proof.final_loss, 1e-8)
            if ratio > 1000:
                reasons.append(
                    f"Gradient/loss ratio too high: {ratio:.2f}"
                )

        return {
            'passed': passed,
            'reasons': reasons,
            'gradient_norm': proof.gradient_norm,
            'max_allowed': challenge.max_gradient_norm
        }

    def verify_time(self, proof: TrainingProof,
                    challenge: PoLChallenge) -> Dict[str, Any]:
        """
        验证训练时间是否合理

        检查项：
        - 训练时间是否在限制内
        - 训练速度是否合理
        - 训练时间是否过短（可能作弊）

        Args:
            proof: 训练证明
            challenge: 对应的挑战

        Returns:
            验证结果
        """
        passed = True
        reasons: List[str] = []

        # 检查是否超时
        if proof.training_time > challenge.time_limit:
            passed = False
            reasons.append(
                f"Training time {proof.training_time:.1f}s > limit {challenge.time_limit:.1f}s"
            )

        # 检查训练时间是否过短（基于难度）
        min_time = challenge.difficulty * 0.5  # 每难度级别至少0.5秒
        if proof.training_time < min_time:
            passed = False
            reasons.append(
                f"Training time too short: {proof.training_time:.1f}s < min {min_time:.1f}s"
            )

        # 检查训练速度是否合理
        if proof.training_time > 0:
            speed = proof.iterations / proof.training_time
            # 合理的训练速度范围
            if speed > 10000:  # 超过10000 iter/s不太可能
                reasons.append(f"Training speed too fast: {speed:.0f} iter/s")

        return {
            'passed': passed,
            'reasons': reasons,
            'training_time': proof.training_time,
            'time_limit': challenge.time_limit,
            'iterations': proof.iterations,
            'speed': proof.training_speed
        }

    def verify_signature(self, proof: TrainingProof) -> Dict[str, Any]:
        """
        验证签名

        检查证明的签名是否有效，确保证明未被篡改。

        Args:
            proof: 训练证明

        Returns:
            验证结果
        """
        if not proof.signature:
            return {
                'passed': False,
                'reasons': ['No signature provided']
            }

        # 验证哈希一致性
        expected_hash = proof.compute_hash()
        # 简化验证：检查签名长度和格式
        if len(proof.signature) != 64:  # SHA-256 hex
            return {
                'passed': False,
                'reasons': ['Invalid signature format']
            }

        return {
            'passed': True,
            'reasons': [],
            'signature': proof.signature[:16] + '...'
        }

    def _compute_score(self, checks: Dict[str, Dict[str, Any]]) -> float:
        """计算综合验证得分"""
        score = 0.0
        weights = {'loss': 0.4, 'gradient': 0.2, 'time': 0.2, 'signature': 0.2}

        for check_name, weight in weights.items():
            check = checks.get(check_name, {})
            if check.get('passed', False):
                score += weight

        return score

    def get_verification_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取验证历史"""
        with self._lock:
            return self._verification_history[-limit:]


# ============== 证明生成器 ==============

class PoLGenerator:
    """
    证明生成器

    模拟训练过程并生成训练证明。
    用于测试和演示。

    Author: AGI Unified Framework
    """

    def __init__(self):
        self._generated_proofs: List[TrainingProof] = []

    def generate_proof(self, challenge: PoLChallenge,
                       participant: str,
                       private_key: str = "default_key") -> TrainingProof:
        """
        生成训练证明

        模拟训练过程，生成符合挑战要求的证明。

        Args:
            challenge: 学习挑战
            participant: 参与者标识
            private_key: 签名私钥

        Returns:
            训练证明
        """
        # 模拟训练参数
        difficulty = challenge.difficulty
        base_iterations = 100 * difficulty
        iterations = base_iterations + random.randint(-10, 10)

        # 模拟loss下降
        initial_loss = random.uniform(2.0, 5.0)
        decay_rate = 0.01 * (11 - difficulty) / 10.0  # 难度越高衰减越慢
        final_loss = initial_loss * math.exp(-decay_rate * iterations)

        # 确保final_loss低于target
        if final_loss > challenge.target_loss:
            final_loss = challenge.target_loss * random.uniform(0.5, 0.9)

        # 模拟梯度范数
        gradient_norm = final_loss * random.uniform(0.5, 2.0)

        # 模拟训练时间
        training_time = iterations * random.uniform(0.001, 0.01) * difficulty

        # 生成证明
        proof = TrainingProof(
            proof_id=hashlib.sha256(
                f"{challenge.challenge_id}:{participant}:{time.time()}".encode()
            ).hexdigest()[:32],
            challenge_id=challenge.challenge_id,
            participant=participant,
            model_version=challenge.model_version,
            initial_loss=initial_loss,
            final_loss=final_loss,
            gradient_norm=gradient_norm,
            training_time=training_time,
            iterations=iterations,
            data_hash=hashlib.sha256(
                json.dumps(challenge.data_samples, sort_keys=True).encode()
            ).hexdigest()[:16],
            model_hash=hashlib.sha256(
                f"{challenge.model_version}:{final_loss}".encode()
            ).hexdigest()[:16]
        )

        # 签名
        proof.sign(private_key)

        self._generated_proofs.append(proof)
        return proof

    def generate_challenge(self, model_version: str,
                           difficulty: int = 5,
                           num_samples: int = 100) -> PoLChallenge:
        """
        生成学习挑战

        Args:
            model_version: 模型版本
            difficulty: 难度级别
            num_samples: 数据样本数

        Returns:
            学习挑战
        """
        # 生成模拟数据样本
        data_samples = []
        for i in range(num_samples):
            data_samples.append({
                'index': i,
                'features': [random.random() for _ in range(10)],
                'label': random.randint(0, 9)
            })

        # 根据难度设置参数
        target_loss = 0.1 + (difficulty - 1) * 0.05
        max_gradient = 15.0 - difficulty * 0.5
        time_limit = 300.0 - (difficulty - 1) * 20.0

        challenge = PoLChallenge(
            challenge_id=hashlib.sha256(
                f"{model_version}:{difficulty}:{time.time()}".encode()
            ).hexdigest()[:32],
            model_version=model_version,
            data_samples=data_samples,
            difficulty=difficulty,
            target_loss=target_loss,
            max_gradient_norm=max(max_gradient, 2.0),
            time_limit=max(max_time_limit, 60.0)
        )

        return challenge


# ============== 难度调整器 ==============

class DifficultyAdjuster:
    """
    难度调整器

    根据网络整体训练情况动态调整挑战难度。
    目标是维持一个合理的证明通过率（约60-80%）。

    调整策略：
    - 通过率高 -> 提高难度
    - 通过率低 -> 降低难度
    - 使用指数移动平均平滑调整

    Author: AGI Unified Framework
    """

    TARGET_PASS_RATE = 0.7  # 目标通过率
    ADJUSTMENT_SPEED = 0.1  # 调整速度
    MIN_DIFFICULTY = 1
    MAX_DIFFICULTY = 10
    WINDOW_SIZE = 50  # 统计窗口大小

    def __init__(self, initial_difficulty: int = 5):
        self._current_difficulty = initial_difficulty
        self._recent_results: deque = deque(maxlen=self.WINDOW_SIZE)
        self._adjustment_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    @property
    def current_difficulty(self) -> int:
        """获取当前难度"""
        return self._current_difficulty

    def record_result(self, passed: bool, score: float,
                      difficulty: int) -> None:
        """
        记录验证结果

        Args:
            passed: 是否通过
            score: 验证得分
            difficulty: 挑战难度
        """
        with self._lock:
            self._recent_results.append({
                'passed': passed,
                'score': score,
                'difficulty': difficulty,
                'timestamp': time.time()
            })

    def get_pass_rate(self) -> float:
        """获取最近的通过率"""
        with self._lock:
            if not self._recent_results:
                return 0.0
            passed = sum(1 for r in self._recent_results if r['passed'])
            return passed / len(self._recent_results)

    def get_average_score(self) -> float:
        """获取最近平均得分"""
        with self._lock:
            if not self._recent_results:
                return 0.0
            total = sum(r['score'] for r in self._recent_results)
            return total / len(self._recent_results)

    def adjust(self) -> int:
        """
        调整难度

        根据最近的验证结果调整挑战难度。

        Returns:
            调整后的难度
        """
        with self._lock:
            if len(self._recent_results) < 10:
                return self._current_difficulty

            pass_rate = self.get_pass_rate()
            avg_score = self.get_average_score()

            old_difficulty = self._current_difficulty

            # 计算调整量
            delta = (pass_rate - self.TARGET_PASS_RATE) * self.ADJUSTMENT_SPEED * 10

            # 根据得分微调
            if avg_score > 0.9:
                delta += 0.5
            elif avg_score < 0.5:
                delta -= 0.5

            # 应用调整
            new_difficulty = self._current_difficulty + int(round(delta))
            new_difficulty = max(self.MIN_DIFFICULTY,
                                 min(self.MAX_DIFFICULTY, new_difficulty))

            self._current_difficulty = new_difficulty

            # 记录调整历史
            self._adjustment_history.append({
                'old_difficulty': old_difficulty,
                'new_difficulty': new_difficulty,
                'pass_rate': pass_rate,
                'avg_score': avg_score,
                'timestamp': time.time()
            })

            return self._current_difficulty

    def get_stats(self) -> Dict[str, Any]:
        """获取调整器统计"""
        with self._lock:
            return {
                'current_difficulty': self._current_difficulty,
                'pass_rate': self.get_pass_rate(),
                'average_score': self.get_average_score(),
                'total_verifications': len(self._recent_results),
                'adjustment_count': len(self._adjustment_history)
            }


# ============== 证明链 ==============

@dataclass
class ProofBlock:
    """
    证明链区块

    类似区块链的结构，每个区块包含一组训练证明。
    通过哈希链确保证明的不可篡改性。

    Attributes:
        index: 区块索引
        timestamp: 创建时间
        proofs: 包含的证明列表
        prev_hash: 前一区块哈希
        hash: 当前区块哈希
        nonce: 随机数
    """
    index: int
    timestamp: float
    proofs: List[TrainingProof]
    prev_hash: str
    hash: str = ""
    nonce: int = 0

    def __post_init__(self):
        if not self.hash:
            self.hash = self.compute_hash()

    def compute_hash(self) -> str:
        """计算区块哈希"""
        proof_hashes = [p.compute_hash() for p in self.proofs]
        content = json.dumps({
            'index': self.index,
            'timestamp': self.timestamp,
            'proof_hashes': proof_hashes,
            'prev_hash': self.prev_hash,
            'nonce': self.nonce
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def mine(self, difficulty: int = 2) -> None:
        """挖矿（简单PoW）"""
        target = '0' * difficulty
        while not self.hash.startswith(target):
            self.nonce += 1
            self.hash = self.compute_hash()

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'proof_count': len(self.proofs),
            'prev_hash': self.prev_hash,
            'hash': self.hash,
            'nonce': self.nonce
        }


class ProofChain:
    """
    证明链

    链式存储训练证明，确保证明的历史不可篡改。
    类似区块链结构，但专注于存储训练证明。

    Author: AGI Unified Framework
    """

    MAX_PROOFS_PER_BLOCK = 10
    MINING_DIFFICULTY = 2

    def __init__(self):
        self._chain: List[ProofBlock] = []
        self._pending_proofs: List[TrainingProof] = []
        self._lock = threading.Lock()
        self._create_genesis_block()

    def _create_genesis_block(self) -> None:
        """创建创世区块"""
        genesis = ProofBlock(
            index=0,
            timestamp=time.time(),
            proofs=[],
            prev_hash="0" * 64
        )
        genesis.mine(self.MINING_DIFFICULTY)
        self._chain.append(genesis)

    def add_proof(self, proof: TrainingProof) -> bool:
        """
        添加训练证明

        将证明添加到待处理队列，当队列满时自动打包成区块。

        Args:
            proof: 训练证明

        Returns:
            是否添加成功
        """
        with self._lock:
            self._pending_proofs.append(proof)

            if len(self._pending_proofs) >= self.MAX_PROOFS_PER_BLOCK:
                self._mine_new_block()

            return True

    def _mine_new_block(self) -> Optional[ProofBlock]:
        """打包新区块"""
        if not self._pending_proofs:
            return None

        proofs = self._pending_proofs[:self.MAX_PROOFS_PER_BLOCK]
        self._pending_proofs = self._pending_proofs[self.MAX_PROOFS_PER_BLOCK:]

        last_block = self._chain[-1]
        new_block = ProofBlock(
            index=len(self._chain),
            timestamp=time.time(),
            proofs=proofs,
            prev_hash=last_block.hash
        )
        new_block.mine(self.MINING_DIFFICULTY)
        self._chain.append(new_block)

        return new_block

    def force_mine(self) -> Optional[ProofBlock]:
        """强制打包待处理的证明"""
        with self._lock:
            return self._mine_new_block()

    def get_chain_length(self) -> int:
        """获取链长度"""
        return len(self._chain)

    def get_total_proofs(self) -> int:
        """获取总证明数"""
        return sum(len(b.proofs) for b in self._chain)

    def is_chain_valid(self) -> bool:
        """验证链的完整性"""
        for i in range(1, len(self._chain)):
            current = self._chain[i]
            previous = self._chain[i - 1]

            if current.prev_hash != previous.hash:
                return False

            if current.compute_hash() != current.hash:
                return False

        return True

    def get_block(self, index: int) -> Optional[ProofBlock]:
        """获取指定区块"""
        if 0 <= index < len(self._chain):
            return self._chain[index]
        return None

    def get_latest_block(self) -> ProofBlock:
        """获取最新区块"""
        return self._chain[-1]

    def get_chain_info(self) -> Dict[str, Any]:
        """获取链信息"""
        with self._lock:
            return {
                'length': len(self._chain),
                'total_proofs': self.get_total_proofs(),
                'pending_proofs': len(self._pending_proofs),
                'is_valid': self.is_chain_valid(),
                'latest_hash': self._chain[-1].hash[:16] + '...'
            }


# ============== 学习证明主类 ==============

class ProofOfLearning:
    """
    学习证明主类

    整合挑战生成、证明验证、难度调整和证明链管理，
    提供完整的学习证明流程。

    使用方式：
    1. 创建ProofOfLearning实例
    2. 调用create_challenge()生成挑战
    3. 参与者训练后调用submit_proof()提交证明
    4. 系统自动验证并记录到证明链

    Author: AGI Unified Framework
    """

    def __init__(self, initial_difficulty: int = 5):
        self._verifier = ProofVerifier()
        self._generator = PoLGenerator()
        self._adjuster = DifficultyAdjuster(initial_difficulty)
        self._proof_chain = ProofChain()
        self._active_challenges: Dict[str, PoLChallenge] = {}
        self._lock = threading.Lock()
        self._stats = {
            'challenges_created': 0,
            'proofs_submitted': 0,
            'proofs_accepted': 0,
            'proofs_rejected': 0
        }

    def create_challenge(self, model_version: str,
                         participant: Optional[str] = None,
                         difficulty: Optional[int] = None) -> PoLChallenge:
        """
        创建学习挑战

        Args:
            model_version: 模型版本
            participant: 目标参与者（可选）
            difficulty: 自定义难度（可选，默认自动调整）

        Returns:
            学习挑战
        """
        if difficulty is None:
            difficulty = self._adjuster.current_difficulty

        challenge = self._generator.generate_challenge(
            model_version, difficulty
        )

        with self._lock:
            self._active_challenges[challenge.challenge_id] = challenge
            self._stats['challenges_created'] += 1

        return challenge

    def submit_proof(self, proof: TrainingProof) -> Dict[str, Any]:
        """
        提交训练证明

        验证证明并记录到证明链。

        Args:
            proof: 训练证明

        Returns:
            验证结果
        """
        with self._lock:
            self._stats['proofs_submitted'] += 1

            # 获取对应挑战
            challenge = self._active_challenges.get(proof.challenge_id)
            if challenge is None:
                return {
                    'valid': False,
                    'reason': 'Challenge not found or expired'
                }

            # 验证证明
            result = self._verifier.verify(proof, challenge)

            # 记录结果
            self._adjuster.record_result(
                result['valid'], result['score'], challenge.difficulty
            )

            if result['valid']:
                self._stats['proofs_accepted'] += 1
                # 添加到证明链
                self._proof_chain.add_proof(proof)
            else:
                self._stats['proofs_rejected'] += 1

            # 调整难度
            self._adjuster.adjust()

            return result

    def generate_and_submit(self, challenge: PoLChallenge,
                            participant: str,
                            private_key: str = "default_key") -> Dict[str, Any]:
        """
        生成证明并提交（便捷方法）

        Args:
            challenge: 学习挑战
            participant: 参与者
            private_key: 签名私钥

        Returns:
            验证结果
        """
        proof = self._generator.generate_proof(
            challenge, participant, private_key
        )
        return self.submit_proof(proof)

    def get_challenge(self, challenge_id: str) -> Optional[PoLChallenge]:
        """获取活跃挑战"""
        return self._active_challenges.get(challenge_id)

    def cleanup_challenges(self) -> int:
        """清理过期挑战"""
        with self._lock:
            expired = [
                cid for cid, c in self._active_challenges.items()
                if c.is_expired
            ]
            for cid in expired:
                del self._active_challenges[cid]
            return len(expired)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                **self._stats,
                'difficulty': self._adjuster.current_difficulty,
                'pass_rate': self._adjuster.get_pass_rate(),
                'chain_info': self._proof_chain.get_chain_info()
            }


# ============== 主程序入口 ==============

if __name__ == "__main__":
    print("=== Proof of Learning Demo ===\n")

    pol = ProofOfLearning(initial_difficulty=5)

    # 创建挑战
    challenge = pol.create_challenge("model_v1", difficulty=5)
    print(f"Challenge: {challenge.challenge_id[:16]}...")
    print(f"Difficulty: {challenge.difficulty}, Target loss: {challenge.target_loss}")

    # 生成并提交证明
    for i in range(15):
        result = pol.generate_and_submit(challenge, f"node_{i}")
        status = "ACCEPTED" if result['valid'] else "REJECTED"
        print(f"  Node_{i}: {status} (score={result['score']:.2f})")

    # 强制打包
    pol._proof_chain.force_mine()

    # 统计
    stats = pol.get_stats()
    print(f"\nStats:")
    print(f"  Challenges: {stats['challenges_created']}")
    print(f"  Submitted: {stats['proofs_submitted']}")
    print(f"  Accepted: {stats['proofs_accepted']}")
    print(f"  Rejected: {stats['proofs_rejected']}")
    print(f"  Difficulty: {stats['difficulty']}")
    print(f"  Pass rate: {stats['pass_rate']:.2%}")
    print(f"  Chain: {stats['chain_info']['length']} blocks, "
          f"{stats['chain_info']['total_proofs']} proofs")

    print("\n=== Demo Complete ===")
