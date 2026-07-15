"""
区块链激励：代币贡献证明，智能合约模拟
支持贡献积分、任务奖励、惩罚机制
"""

import hashlib
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class TransactionType(Enum):
    REWARD = "reward"
    PENALTY = "penalty"
    TRANSFER = "transfer"
    STAKE = "stake"


@dataclass
class Transaction:
    tx_id: str
    tx_type: TransactionType
    from_agent: str
    to_agent: str
    amount: float
    reason: str
    timestamp: float = field(default_factory=time.time)
    signature: str = ""


@dataclass
class Block:
    block_id: int
    previous_hash: str
    transactions: List[Transaction]
    timestamp: float
    nonce: int
    hash: str = ""


class TokenIncentiveSystem:
    """
    基于区块链的激励系统
    模拟代币经济：贡献获得代币，恶意行为扣除代币
    """

    def __init__(self, genesis_alloc: Dict[str, float] = None):
        self.balances: Dict[str, float] = genesis_alloc or {}
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.difficulty = 4  # PoW难度（前导零个数）
        self._create_genesis_block()

    def _create_genesis_block(self):
        """创世区块"""
        genesis = Block(
            block_id=0,
            previous_hash="0" * 64,
            transactions=[],
            timestamp=time.time(),
            nonce=0,
            hash=self._calculate_hash(0, "0" * 64, [], 0)
        )
        self.chain.append(genesis)

    def _calculate_hash(self, block_id: int, previous_hash: str,
                        transactions: List[Transaction], nonce: int) -> str:
        """计算区块哈希"""
        tx_data = [f"{t.tx_id}{t.tx_type.value}{t.from_agent}{t.to_agent}{t.amount}{t.reason}" 
                   for t in transactions]
        content = f"{block_id}{previous_hash}{json.dumps(tx_data)}{nonce}{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _proof_of_work(self, block_id: int, previous_hash: str,
                       transactions: List[Transaction]) -> tuple:
        """PoW挖矿"""
        nonce = 0
        while True:
            hash_val = self._calculate_hash(block_id, previous_hash, transactions, nonce)
            if hash_val[:self.difficulty] == "0" * self.difficulty:
                return nonce, hash_val
            nonce += 1

    def add_transaction(self, tx_type: TransactionType, from_agent: str,
                        to_agent: str, amount: float, reason: str) -> Optional[str]:
        """添加待处理交易"""
        # 验证余额
        if from_agent != "system" and self.balances.get(from_agent, 0) < amount:
            return None
        tx = Transaction(
            tx_id=hashlib.md5(f"{from_agent}{to_agent}{amount}{time.time()}".encode()).hexdigest()[:16],
            tx_type=tx_type,
            from_agent=from_agent,
            to_agent=to_agent,
            amount=amount,
            reason=reason
        )
        self.pending_transactions.append(tx)
        return tx.tx_id

    def mine_block(self, miner_id: str) -> Optional[Block]:
        """挖矿：打包待处理交易"""
        if not self.pending_transactions:
            return None

        prev_block = self.chain[-1]
        nonce, block_hash = self._proof_of_work(
            len(self.chain),
            prev_block.hash,
            self.pending_transactions
        )

        # 矿工奖励
        self.pending_transactions.append(Transaction(
            tx_id=f"mining_reward_{time.time()}",
            tx_type=TransactionType.REWARD,
            from_agent="system",
            to_agent=miner_id,
            amount=10.0,  # 挖矿奖励
            reason="mining_reward"
        ))

        block = Block(
            block_id=len(self.chain),
            previous_hash=prev_block.hash,
            transactions=self.pending_transactions.copy(),
            timestamp=time.time(),
            nonce=nonce,
            hash=block_hash
        )
        self.chain.append(block)
        self._apply_transactions(self.pending_transactions)
        self.pending_transactions = []
        return block

    def _apply_transactions(self, transactions: List[Transaction]):
        """应用交易到余额"""
        for tx in transactions:
            if tx.from_agent != "system":
                self.balances[tx.from_agent] = self.balances.get(tx.from_agent, 0) - tx.amount
            self.balances[tx.to_agent] = self.balances.get(tx.to_agent, 0) + tx.amount

    def reward_agent(self, agent_id: str, amount: float, reason: str) -> str:
        """奖励Agent代币"""
        return self.add_transaction(TransactionType.REWARD, "system", agent_id, amount, reason)

    def penalize_agent(self, agent_id: str, amount: float, reason: str) -> str:
        """惩罚Agent（扣除代币）"""
        return self.add_transaction(TransactionType.PENALTY, agent_id, "system", amount, reason)

    def transfer(self, from_agent: str, to_agent: str, amount: float, reason: str) -> str:
        """Agent间转账"""
        return self.add_transaction(TransactionType.TRANSFER, from_agent, to_agent, amount, reason)

    def get_balance(self, agent_id: str) -> float:
        return self.balances.get(agent_id, 0)

    def get_leaderboard(self, top_n: int = 10) -> List[tuple]:
        """财富排行榜"""
        sorted_balances = sorted(self.balances.items(), key=lambda x: x[1], reverse=True)
        return sorted_balances[:top_n]

    def verify_chain(self) -> bool:
        """验证区块链完整性"""
        for i in range(1, len(self.chain)):
            block = self.chain[i]
            prev_block = self.chain[i-1]
            # 验证哈希链接
            if block.previous_hash != prev_block.hash:
                return False
            # 验证PoW
            expected_hash = self._calculate_hash(block.block_id, block.previous_hash,
                                                  block.transactions, block.nonce)
            if expected_hash != block.hash:
                return False
            if block.hash[:self.difficulty] != "0" * self.difficulty:
                return False
        return True


