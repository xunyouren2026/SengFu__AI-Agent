"""
加密向量存储
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import hashlib
import base64
import copy


class EncryptionScheme(Enum):
    """加密方案"""
    NONE = "none"
    XOR = "xor"  # 简单XOR（仅用于演示）
    HASH = "hash"  # 哈希承诺
    SHAMIR = "shamir"  # Shamir秘密共享


class EncryptedVector:
    """加密向量"""
    
    def __init__(
        self,
        vector_id: str,
        encrypted_data: bytes,
        scheme: EncryptionScheme,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.vector_id = vector_id
        self.encrypted_data = encrypted_data
        self.scheme = scheme
        self.metadata = metadata or {}
        self.created_at = datetime.now().timestamp()
        self.size = len(encrypted_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'vector_id': self.vector_id,
            'encrypted_data': base64.b64encode(self.encrypted_data).decode(),
            'scheme': self.scheme.value,
            'metadata': self.metadata,
            'created_at': self.created_at,
            'size': self.size
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EncryptedVector':
        """从字典创建"""
        return cls(
            vector_id=data['vector_id'],
            encrypted_data=base64.b64decode(data['encrypted_data']),
            scheme=EncryptionScheme(data['scheme']),
            metadata=data.get('metadata', {})
        )


class SimpleEncryptor:
    """
    简单加密器
    
    注意：这仅用于演示，实际应用应使用专业加密库
    """
    
    def __init__(self, key: Optional[bytes] = None):
        self.key = key or self._generate_key()
    
    def _generate_key(self) -> bytes:
        """生成密钥"""
        import random
        return bytes([random.randint(0, 255) for _ in range(32)])
    
    def encrypt(self, data: bytes) -> bytes:
        """加密数据"""
        # 简单XOR加密
        key_len = len(self.key)
        encrypted = bytearray(len(data))
        
        for i, byte in enumerate(data):
            encrypted[i] = byte ^ self.key[i % key_len]
        
        return bytes(encrypted)
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """解密数据"""
        # XOR解密与加密相同
        return self.encrypt(encrypted_data)


class HashCommitter:
    """
    哈希承诺器
    
    用于验证数据完整性而不暴露内容
    """
    
    def __init__(self, hash_algorithm: str = "sha256"):
        self.hash_algorithm = hash_algorithm
    
    def commit(self, data: bytes) -> Tuple[str, bytes]:
        """
        生成承诺
        
        Returns:
            (承诺值, 开启值)
        """
        # 生成随机nonce
        import random
        nonce = bytes([random.randint(0, 255) for _ in range(16)])
        
        # 计算承诺
        h = hashlib.new(self.hash_algorithm)
        h.update(data + nonce)
        commitment = h.hexdigest()
        
        return commitment, nonce
    
    def verify(
        self,
        data: bytes,
        commitment: str,
        nonce: bytes
    ) -> bool:
        """验证承诺"""
        h = hashlib.new(self.hash_algorithm)
        h.update(data + nonce)
        return h.hexdigest() == commitment


class EncryptedVectorStore:
    """
    加密向量存储
    
    安全存储联邦学习中的敏感向量
    """
    
    def __init__(
        self,
        encryption_scheme: EncryptionScheme = EncryptionScheme.XOR,
        max_vectors: int = 10000
    ):
        self.encryption_scheme = encryption_scheme
        self.max_vectors = max_vectors
        
        self._vectors: Dict[str, EncryptedVector] = {}
        self._encryptor = SimpleEncryptor()
        self._committer = HashCommitter()
        
        # 统计
        self._total_stored = 0
        self._total_retrieved = 0
    
    def store(
        self,
        vector_id: str,
        data: List[float],
        owner: Optional[str] = None
    ) -> EncryptedVector:
        """
        存储向量
        
        Args:
            vector_id: 向量ID
            data: 向量数据
            owner: 所有者
        
        Returns:
            加密向量
        """
        # 序列化
        serialized = self._serialize_vector(data)
        
        # 加密
        if self.encryption_scheme == EncryptionScheme.NONE:
            encrypted_data = serialized
        else:
            encrypted_data = self._encryptor.encrypt(serialized)
        
        # 创建加密向量
        encrypted_vector = EncryptedVector(
            vector_id=vector_id,
            encrypted_data=encrypted_data,
            scheme=self.encryption_scheme,
            metadata={'owner': owner, 'dim': len(data)}
        )
        
        # 存储
        if len(self._vectors) >= self.max_vectors:
            self._evict_old()
        
        self._vectors[vector_id] = encrypted_vector
        self._total_stored += 1
        
        return encrypted_vector
    
    def retrieve(
        self,
        vector_id: str
    ) -> Optional[List[float]]:
        """
        检索向量
        
        Args:
            vector_id: 向量ID
        
        Returns:
            解密后的向量
        """
        if vector_id not in self._vectors:
            return None
        
        encrypted_vector = self._vectors[vector_id]
        
        # 解密
        if encrypted_vector.scheme == EncryptionScheme.NONE:
            decrypted_data = encrypted_vector.encrypted_data
        else:
            decrypted_data = self._encryptor.decrypt(encrypted_vector.encrypted_data)
        
        # 反序列化
        vector = self._deserialize_vector(decrypted_data)
        
        self._total_retrieved += 1
        return vector
    
    def _serialize_vector(self, data: List[float]) -> bytes:
        """序列化向量"""
        import struct
        return b''.join(struct.pack('d', v) for v in data)
    
    def _deserialize_vector(self, data: bytes) -> List[float]:
        """反序列化向量"""
        import struct
        n = len(data) // 8
        return list(struct.unpack(f'{n}d', data))
    
    def compute_commitment(
        self,
        vector_id: str
    ) -> Optional[Tuple[str, bytes]]:
        """计算向量承诺"""
        vector = self.retrieve(vector_id)
        if vector is None:
            return None
        
        serialized = self._serialize_vector(vector)
        return self._committer.commit(serialized)
    
    def verify_commitment(
        self,
        vector_id: str,
        commitment: str,
        nonce: bytes
    ) -> bool:
        """验证向量承诺"""
        vector = self.retrieve(vector_id)
        if vector is None:
            return False
        
        serialized = self._serialize_vector(vector)
        return self._committer.verify(serialized, commitment, nonce)
    
    def aggregate_encrypted(
        self,
        vector_ids: List[str],
        weights: Optional[List[float]] = None
    ) -> Optional[List[float]]:
        """
        聚合加密向量
        
        解密后聚合
        """
        vectors = []
        for vid in vector_ids:
            v = self.retrieve(vid)
            if v is None:
                return None
            vectors.append(v)
        
        if weights is None:
            weights = [1.0 / len(vectors)] * len(vectors)
        
        # 加权聚合
        dim = len(vectors[0])
        result = [0.0] * dim
        
        for v, w in zip(vectors, weights):
            for i in range(dim):
                result[i] += w * v[i]
        
        return result
    
    def delete(self, vector_id: str) -> bool:
        """删除向量"""
        if vector_id in self._vectors:
            del self._vectors[vector_id]
            return True
        return False
    
    def _evict_old(self) -> None:
        """淘汰旧向量"""
        n_remove = len(self._vectors) // 10
        sorted_vectors = sorted(
            self._vectors.values(),
            key=lambda v: v.created_at
        )
        
        for v in sorted_vectors[:n_remove]:
            del self._vectors[v.vector_id]
    
    def get_vector_ids(self) -> List[str]:
        """获取所有向量ID"""
        return list(self._vectors.keys())
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_size = sum(v.size for v in self._vectors.values())
        
        return {
            'total_vectors': len(self._vectors),
            'total_stored': self._total_stored,
            'total_retrieved': self._total_retrieved,
            'total_size_bytes': total_size,
            'encryption_scheme': self.encryption_scheme.value,
            'max_vectors': self.max_vectors
        }


class SecureAggregationProtocol:
    """
    安全聚合协议
    
    实现联邦学习中的安全聚合
    """
    
    def __init__(self, num_clients: int, threshold: int):
        self.num_clients = num_clients
        self.threshold = threshold  # 最少参与客户端数
        
        self._client_shares: Dict[str, List[bytes]] = {}
        self._aggregation_result: Optional[bytes] = None
    
    def submit_share(
        self,
        client_id: str,
        share: bytes
    ) -> bool:
        """提交秘密份额"""
        if client_id not in self._client_shares:
            self._client_shares[client_id] = []
        
        self._client_shares[client_id].append(share)
        return True
    
    def can_aggregate(self) -> bool:
        """是否可以聚合"""
        return len(self._client_shares) >= self.threshold
    
    def aggregate(self) -> Optional[bytes]:
        """执行安全聚合"""
        if not self.can_aggregate():
            return None
        
        # 简化实现：直接XOR所有份额
        all_shares = []
        for shares in self._client_shares.values():
            all_shares.extend(shares)
        
        if not all_shares:
            return None
        
        result = bytearray(len(all_shares[0]))
        for share in all_shares:
            for i, b in enumerate(share):
                if i < len(result):
                    result[i] ^= b
        
        self._aggregation_result = bytes(result)
        return self._aggregation_result
    
    def reset(self) -> None:
        """重置协议"""
        self._client_shares.clear()
        self._aggregation_result = None
