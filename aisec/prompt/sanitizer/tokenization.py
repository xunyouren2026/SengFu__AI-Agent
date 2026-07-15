"""
令牌化 - 敏感数据令牌化处理
"""
import hashlib
import secrets
import time
import json
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum


class TokenType(Enum):
    """令牌类型"""
    RANDOM = "random"           # 随机令牌
    HASH = "hash"               # 哈希令牌
    TIMESTAMP = "timestamp"     # 带时间戳令牌
    PREFIX = "prefix"           # 带前缀令牌
    REVERSIBLE = "reversible"   # 可逆令牌


@dataclass
class TokenEntry:
    """令牌条目"""
    original_value: str
    token: str
    token_type: TokenType
    created_at: float
    expires_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TokenizationEngine:
    """令牌化引擎"""
    
    def __init__(
        self,
        default_token_type: TokenType = TokenType.PREFIX,
        token_prefix: str = "TOK_",
        token_length: int = 16,
        default_ttl: Optional[float] = None  # 令牌过期时间（秒）
    ):
        self._token_type = default_token_type
        self._token_prefix = token_prefix
        self._token_length = token_length
        self._default_ttl = default_ttl
        
        # 令牌存储
        self._token_store: Dict[str, TokenEntry] = {}
        self._reverse_store: Dict[str, str] = {}  # 用于可逆令牌
        
        # 加密密钥（用于可逆令牌）
        self._encryption_key = secrets.token_bytes(32)
    
    def _generate_random_token(self) -> str:
        """生成随机令牌"""
        return secrets.token_urlsafe(self._token_length)
    
    def _generate_hash_token(self, value: str) -> str:
        """生成哈希令牌"""
        salt = secrets.token_bytes(8)
        hash_value = hashlib.sha256(salt + value.encode()).hexdigest()
        return hash_value[:self._token_length]
    
    def _generate_timestamp_token(self) -> str:
        """生成带时间戳令牌"""
        timestamp = int(time.time() * 1000)
        random_part = secrets.token_urlsafe(8)
        return f"{timestamp}_{random_part}"
    
    def _generate_prefix_token(self) -> str:
        """生成带前缀令牌"""
        return self._token_prefix + secrets.token_urlsafe(self._token_length)
    
    def _generate_reversible_token(self, value: str) -> str:
        """生成可逆令牌"""
        # 简单的XOR加密（实际应用中应使用更安全的加密）
        value_bytes = value.encode('utf-8')
        key_repeated = (self._encryption_key * ((len(value_bytes) // 32) + 1))[:len(value_bytes)]
        encrypted = bytes(a ^ b for a, b in zip(value_bytes, key_repeated))
        
        # Base64编码
        import base64
        token = self._token_prefix + base64.urlsafe_b64encode(encrypted).decode()
        
        # 存储反向映射
        self._reverse_store[token] = value
        
        return token
    
    def tokenize(
        self,
        value: str,
        token_type: Optional[TokenType] = None,
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """令牌化值"""
        if not value:
            return value
        
        # 检查是否已令牌化
        if value in self._token_store:
            entry = self._token_store[value]
            # 检查是否过期
            if entry.expires_at and time.time() > entry.expires_at:
                del self._token_store[value]
            else:
                return entry.token
        
        # 确定令牌类型
        actual_type = token_type or self._token_type
        actual_ttl = ttl if ttl is not None else self._default_ttl
        
        # 生成令牌
        if actual_type == TokenType.RANDOM:
            token = self._generate_random_token()
        elif actual_type == TokenType.HASH:
            token = self._generate_hash_token(value)
        elif actual_type == TokenType.TIMESTAMP:
            token = self._generate_timestamp_token()
        elif actual_type == TokenType.PREFIX:
            token = self._generate_prefix_token()
        elif actual_type == TokenType.REVERSIBLE:
            token = self._generate_reversible_token(value)
        else:
            token = self._generate_prefix_token()
        
        # 计算过期时间
        expires_at = time.time() + actual_ttl if actual_ttl else None
        
        # 存储条目
        entry = TokenEntry(
            original_value=value,
            token=token,
            token_type=actual_type,
            created_at=time.time(),
            expires_at=expires_at,
            metadata=metadata or {}
        )
        
        self._token_store[value] = entry
        
        return token
    
    def detokenize(self, token: str) -> Optional[str]:
        """反令牌化"""
        # 检查可逆令牌
        if token in self._reverse_store:
            return self._reverse_store[token]
        
        # 查找原始值
        for value, entry in self._token_store.items():
            if entry.token == token:
                # 检查是否过期
                if entry.expires_at and time.time() > entry.expires_at:
                    return None
                return value
        
        return None
    
    def tokenize_dict(
        self,
        data: Dict[str, Any],
        fields: List[str],
        token_type: Optional[TokenType] = None
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """令牌化字典中的指定字段"""
        result = data.copy()
        token_mapping = {}
        
        def _process(d: Dict[str, Any], path: str = "") -> None:
            for key, value in list(d.items()):
                current_path = f"{path}.{key}" if path else key
                
                if key in fields and isinstance(value, str):
                    token = self.tokenize(value, token_type)
                    d[key] = token
                    token_mapping[current_path] = token
                
                elif isinstance(value, dict):
                    _process(value, current_path)
                
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            _process(item, f"{current_path}[{i}]")
        
        _process(result)
        return result, token_mapping
    
    def detokenize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """反令牌化字典"""
        result = data.copy()
        
        def _process(d: Dict[str, Any]) -> None:
            for key, value in list(d.items()):
                if isinstance(value, str) and value.startswith(self._token_prefix):
                    original = self.detokenize(value)
                    if original:
                        d[key] = original
                elif isinstance(value, dict):
                    _process(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            _process(item)
        
        _process(result)
        return result
    
    def tokenize_text(
        self,
        text: str,
        patterns: List[str],
        token_type: Optional[TokenType] = None
    ) -> Tuple[str, Dict[str, str]]:
        """令牌化文本中匹配模式的值"""
        import re
        token_mapping = {}
        
        for pattern in patterns:
            regex = re.compile(pattern)
            
            for match in regex.finditer(text):
                original = match.group()
                if original not in token_mapping:
                    token = self.tokenize(original, token_type)
                    token_mapping[original] = token
        
        # 替换文本
        result = text
        for original, token in token_mapping.items():
            result = result.replace(original, token)
        
        return result, token_mapping
    
    def revoke_token(self, token: str) -> bool:
        """撤销令牌"""
        # 从反向存储中删除
        if token in self._reverse_store:
            del self._reverse_store[token]
        
        # 从主存储中删除
        for value, entry in list(self._token_store.items()):
            if entry.token == token:
                del self._token_store[value]
                return True
        
        return False
    
    def revoke_value(self, value: str) -> bool:
        """撤销值的令牌"""
        if value in self._token_store:
            entry = self._token_store[value]
            if entry.token in self._reverse_store:
                del self._reverse_store[entry.token]
            del self._token_store[value]
            return True
        return False
    
    def cleanup_expired(self) -> int:
        """清理过期令牌"""
        current_time = time.time()
        expired_count = 0
        
        for value, entry in list(self._token_store.items()):
            if entry.expires_at and current_time > entry.expires_at:
                if entry.token in self._reverse_store:
                    del self._reverse_store[entry.token]
                del self._token_store[value]
                expired_count += 1
        
        return expired_count
    
    def get_token_info(self, token: str) -> Optional[Dict[str, Any]]:
        """获取令牌信息"""
        for value, entry in self._token_store.items():
            if entry.token == token:
                return {
                    "token": token,
                    "token_type": entry.token_type.value,
                    "created_at": entry.created_at,
                    "expires_at": entry.expires_at,
                    "is_expired": entry.expires_at is not None and time.time() > entry.expires_at,
                    "metadata": entry.metadata
                }
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        current_time = time.time()
        active_count = 0
        expired_count = 0
        
        for entry in self._token_store.values():
            if entry.expires_at and current_time > entry.expires_at:
                expired_count += 1
            else:
                active_count += 1
        
        return {
            "total_tokens": len(self._token_store),
            "active_tokens": active_count,
            "expired_tokens": expired_count,
            "reversible_tokens": len(self._reverse_store)
        }
    
    def export_tokens(self) -> str:
        """导出令牌（JSON格式）"""
        data = {
            "tokens": [
                {
                    "original_hash": hashlib.sha256(entry.original_value.encode()).hexdigest()[:16],
                    "token": entry.token,
                    "token_type": entry.token_type.value,
                    "created_at": entry.created_at,
                    "expires_at": entry.expires_at
                }
                for entry in self._token_store.values()
            ]
        }
        return json.dumps(data, indent=2)
