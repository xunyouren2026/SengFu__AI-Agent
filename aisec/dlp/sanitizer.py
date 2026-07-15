"""
Data Sanitizer Module - 数据脱敏引擎

提供多种数据脱敏策略：
- 掩码脱敏（保留前n后m位）
- 哈希脱敏（SHA-256）
- 令牌化脱敏（随机token映射）
- 截断脱敏
- 完全遮蔽
- 差分隐私扰动
"""

import re
import hashlib
import random
import string
import json
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Any, List, Union, Tuple
from pathlib import Path


class MaskingStrategy(Enum):
    """掩码策略"""
    FULL = "full"                   # 完全遮蔽
    PARTIAL = "partial"             # 部分遮蔽（保留前后）
    EMAIL = "email"                 # 邮箱专用
    PHONE = "phone"                 # 手机号专用
    CREDIT_CARD = "credit_card"     # 信用卡专用
    ID_CARD = "id_card"             # 身份证专用


@dataclass
class SanitizationRule:
    """脱敏规则"""
    field_pattern: Union[str, re.Pattern]
    pii_type: str
    strategy: MaskingStrategy
    preserve_chars: Tuple[int, int] = (0, 0)  # (前保留, 后保留)
    mask_char: str = "*"
    description: str = ""
    
    def __post_init__(self):
        if isinstance(self.field_pattern, str):
            self.field_pattern = re.compile(self.field_pattern, re.IGNORECASE)
    
    def matches_field(self, field_name: str) -> bool:
        """检查字段名是否匹配"""
        if isinstance(self.field_pattern, re.Pattern):
            return bool(self.field_pattern.search(field_name))
        return self.field_pattern.lower() in field_name.lower()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "pii_type": self.pii_type,
            "strategy": self.strategy.value,
            "preserve_chars": self.preserve_chars,
            "mask_char": self.mask_char,
            "description": self.description
        }


class DataSanitizer:
    """数据脱敏引擎"""
    
    def __init__(self):
        self.token_map: Dict[str, str] = {}  # 原始值到token的映射
        self.reverse_map: Dict[str, str] = {}  # token到原始值的映射
        self.rules: List[SanitizationRule] = []
        self._init_default_rules()
    
    def _init_default_rules(self):
        """初始化默认脱敏规则"""
        # 邮箱脱敏规则
        self.add_rule(SanitizationRule(
            field_pattern=r'email|mail|邮箱',
            pii_type="email",
            strategy=MaskingStrategy.EMAIL,
            description="邮箱地址脱敏"
        ))
        
        # 手机号脱敏规则
        self.add_rule(SanitizationRule(
            field_pattern=r'phone|mobile|tel|电话|手机',
            pii_type="phone",
            strategy=MaskingStrategy.PHONE,
            description="手机号码脱敏"
        ))
        
        # 身份证号脱敏规则
        self.add_rule(SanitizationRule(
            field_pattern=r'id.?card|identity|身份证',
            pii_type="id_card",
            strategy=MaskingStrategy.ID_CARD,
            description="身份证号脱敏"
        ))
        
        # 信用卡号脱敏规则
        self.add_rule(SanitizationRule(
            field_pattern=r'credit.?card|card.?number|银行卡',
            pii_type="credit_card",
            strategy=MaskingStrategy.CREDIT_CARD,
            description="信用卡号脱敏"
        ))
        
        # 姓名脱敏规则
        self.add_rule(SanitizationRule(
            field_pattern=r'name|姓名',
            pii_type="name",
            strategy=MaskingStrategy.PARTIAL,
            preserve_chars=(1, 0),
            description="姓名脱敏"
        ))
    
    def add_rule(self, rule: SanitizationRule):
        """添加脱敏规则"""
        self.rules.append(rule)
    
    def remove_rule(self, index: int) -> bool:
        """移除脱敏规则"""
        if 0 <= index < len(self.rules):
            del self.rules[index]
            return True
        return False
    
    def mask(self, value: str, mask_char: str = '*', 
             preserve_prefix: int = 0, preserve_suffix: int = 0) -> str:
        """
        掩码脱敏 - 保留前n后m位，其余用掩码字符替换
        
        Args:
            value: 原始值
            mask_char: 掩码字符
            preserve_prefix: 前保留位数
            preserve_suffix: 后保留位数
            
        Returns:
            脱敏后的值
        """
        if not value:
            return value
        
        length = len(value)
        
        if preserve_prefix + preserve_suffix >= length:
            return value
        
        prefix = value[:preserve_prefix]
        suffix = value[-preserve_suffix:] if preserve_suffix > 0 else ""
        middle_length = length - preserve_prefix - preserve_suffix
        
        return prefix + mask_char * middle_length + suffix
    
    def hash(self, value: str, salt: Optional[str] = None) -> str:
        """
        哈希脱敏 - 使用SHA-256
        
        Args:
            value: 原始值
            salt: 盐值
            
        Returns:
            哈希值
        """
        if not value:
            return value
        
        data = value.encode('utf-8')
        if salt:
            data = salt.encode('utf-8') + data
        
        return hashlib.sha256(data).hexdigest()
    
    def tokenize(self, value: str, token_length: int = 16, 
                 prefix: str = "TOK_") -> str:
        """
        令牌化脱敏 - 生成随机token映射
        
        Args:
            value: 原始值
            token_length: token长度
            prefix: token前缀
            
        Returns:
            token
        """
        if not value:
            return value
        
        # 检查是否已有token
        if value in self.token_map:
            return self.token_map[value]
        
        # 生成新token
        chars = string.ascii_letters + string.digits
        token = prefix + ''.join(random.choices(chars, k=token_length))
        
        # 确保唯一性
        while token in self.reverse_map:
            token = prefix + ''.join(random.choices(chars, k=token_length))
        
        # 存储映射
        self.token_map[value] = token
        self.reverse_map[token] = value
        
        return token
    
    def detokenize(self, token: str) -> Optional[str]:
        """
        反令牌化 - 从token获取原始值
        
        Args:
            token: token
            
        Returns:
            原始值，如果不存在则返回None
        """
        return self.reverse_map.get(token)
    
    def truncate(self, value: str, length: int, suffix: str = "...") -> str:
        """
        截断脱敏 - 截断到指定长度
        
        Args:
            value: 原始值
            length: 保留长度
            suffix: 截断后缀
            
        Returns:
            截断后的值
        """
        if not value or len(value) <= length:
            return value
        
        return value[:length - len(suffix)] + suffix
    
    def redact(self, value: str, replacement: str = "[REDACTED]") -> str:
        """
        完全遮蔽
        
        Args:
            value: 原始值
            replacement: 替换文本
            
        Returns:
            遮蔽后的值
        """
        if not value:
            return value
        
        return replacement
    
    def perturb(self, value: Union[int, float], epsilon: float = 1.0) -> Union[int, float]:
        """
        差分隐私扰动 - 添加拉普拉斯噪声
        
        Args:
            value: 原始数值
            epsilon: 隐私预算（越小隐私保护越强）
            
        Returns:
            扰动后的值
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        
        # 拉普拉斯分布参数
        scale = 1.0 / epsilon
        
        # 生成拉普拉斯噪声
        u = random.random() - 0.5
        noise = -scale * (1 if u < 0 else -1) * math.log(1 - 2 * abs(u))
        
        result = value + noise
        
        # 保持原始类型
        if isinstance(value, int):
            return int(round(result))
        return result
    
    def sanitize_email(self, email: str) -> str:
        """邮箱专用脱敏"""
        if not email or '@' not in email:
            return email
        
        local, domain = email.rsplit('@', 1)
        
        if len(local) <= 2:
            masked_local = local[0] + '*' if len(local) > 1 else '*'
        else:
            masked_local = local[0] + '*' * (len(local) - 2) + local[-1]
        
        return f"{masked_local}@{domain}"
    
    def sanitize_phone(self, phone: str) -> str:
        """手机号专用脱敏"""
        digits = ''.join(c for c in phone if c.isdigit())
        
        if len(digits) < 7:
            return self.mask(digits, preserve_prefix=2, preserve_suffix=2)
        
        # 保留前3位和后4位
        return digits[:3] + '*' * (len(digits) - 7) + digits[-4:]
    
    def sanitize_credit_card(self, card: str) -> str:
        """信用卡专用脱敏"""
        digits = ''.join(c for c in card if c.isdigit())
        
        if len(digits) < 4:
            return '*' * len(digits)
        
        # 保留前6位和后4位
        return digits[:6] + '*' * (len(digits) - 10) + digits[-4:]
    
    def sanitize_id_card(self, id_card: str) -> str:
        """身份证专用脱敏"""
        if len(id_card) != 18:
            return self.mask(id_card, preserve_prefix=2, preserve_suffix=2)
        
        # 保留前6位和后4位
        return id_card[:6] + '*' * 8 + id_card[-4:]
    
    def sanitize_with_strategy(self, value: str, strategy: MaskingStrategy,
                                preserve_chars: Tuple[int, int] = (0, 0)) -> str:
        """使用指定策略脱敏"""
        if strategy == MaskingStrategy.FULL:
            return self.redact(value)
        elif strategy == MaskingStrategy.PARTIAL:
            return self.mask(value, preserve_prefix=preserve_chars[0], 
                           preserve_suffix=preserve_chars[1])
        elif strategy == MaskingStrategy.EMAIL:
            return self.sanitize_email(value)
        elif strategy == MaskingStrategy.PHONE:
            return self.sanitize_phone(value)
        elif strategy == MaskingStrategy.CREDIT_CARD:
            return self.sanitize_credit_card(value)
        elif strategy == MaskingStrategy.ID_CARD:
            return self.sanitize_id_card(value)
        else:
            return self.mask(value)
    
    def sanitize_field(self, field_name: str, value: str) -> str:
        """根据字段名自动选择脱敏策略"""
        for rule in self.rules:
            if rule.matches_field(field_name):
                return self.sanitize_with_strategy(value, rule.strategy, 
                                                   rule.preserve_chars)
        
        # 默认部分脱敏
        return self.mask(value, preserve_prefix=2, preserve_suffix=2)
    
    def sanitize_dict(self, data: Dict[str, Any], 
                      recursive: bool = True) -> Dict[str, Any]:
        """脱敏字典中的所有字段"""
        result = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.sanitize_field(key, value)
            elif isinstance(value, dict) and recursive:
                result[key] = self.sanitize_dict(value, recursive)
            elif isinstance(value, list) and recursive:
                result[key] = self.sanitize_list(value)
            else:
                result[key] = value
        
        return result
    
    def sanitize_list(self, data: List[Any]) -> List[Any]:
        """脱敏列表中的字符串项"""
        result = []
        
        for item in data:
            if isinstance(item, str):
                result.append(self.mask(item, preserve_prefix=2, preserve_suffix=2))
            elif isinstance(item, dict):
                result.append(self.sanitize_dict(item))
            elif isinstance(item, list):
                result.append(self.sanitize_list(item))
            else:
                result.append(item)
        
        return result
    
    def clear_token_map(self):
        """清除token映射"""
        self.token_map.clear()
        self.reverse_map.clear()


class SanitizationPipeline:
    """脱敏管道 - 多规则链式处理"""
    
    def __init__(self):
        self.steps: List[Callable[[str], str]] = []
    
    def add_step(self, func: Callable[[str], str]):
        """添加处理步骤"""
        self.steps.append(func)
        return self
    
    def process(self, value: str) -> str:
        """执行管道处理"""
        result = value
        for step in self.steps:
            result = step(result)
        return result
    
    def process_batch(self, values: List[str]) -> List[str]:
        """批量处理"""
        return [self.process(v) for v in values]


class ReversibleSanitizer:
    """可逆脱敏器 - 使用加密存储原始值映射"""
    
    def __init__(self, encryption_key: Optional[str] = None):
        self.sanitizer = DataSanitizer()
        self.encryption_key = encryption_key or self._generate_key()
        self.encrypted_map: Dict[str, str] = {}  # token -> 加密后的原始值
    
    def _generate_key(self) -> str:
        """生成随机密钥"""
        chars = string.ascii_letters + string.digits
        return ''.join(random.choices(chars, k=32))
    
    def _encrypt(self, value: str) -> str:
        """简单XOR加密"""
        key_bytes = self.encryption_key.encode('utf-8')
        value_bytes = value.encode('utf-8')
        
        encrypted = bytearray()
        for i, b in enumerate(value_bytes):
            encrypted.append(b ^ key_bytes[i % len(key_bytes)])
        
        return encrypted.hex()
    
    def _decrypt(self, encrypted_hex: str) -> str:
        """解密"""
        key_bytes = self.encryption_key.encode('utf-8')
        encrypted = bytearray.fromhex(encrypted_hex)
        
        decrypted = bytearray()
        for i, b in enumerate(encrypted):
            decrypted.append(b ^ key_bytes[i % len(key_bytes)])
        
        return decrypted.decode('utf-8')
    
    def sanitize(self, value: str) -> str:
        """脱敏并存储加密映射"""
        token = self.sanitizer.tokenize(value)
        
        if value not in self.encrypted_map:
            self.encrypted_map[token] = self._encrypt(value)
        
        return token
    
    def restore(self, token: str) -> Optional[str]:
        """恢复原始值"""
        encrypted = self.encrypted_map.get(token)
        if encrypted:
            return self._decrypt(encrypted)
        return None


def sanitizer_for_type(pii_type: str) -> MaskingStrategy:
    """
    根据PII类型选择最佳脱敏策略
    
    Args:
        pii_type: PII类型
        
    Returns:
        最佳脱敏策略
    """
    strategy_map = {
        "email": MaskingStrategy.EMAIL,
        "china_mobile": MaskingStrategy.PHONE,
        "credit_card": MaskingStrategy.CREDIT_CARD,
        "china_id_card": MaskingStrategy.ID_CARD,
        "us_ssn": MaskingStrategy.PARTIAL,
        "person_name": MaskingStrategy.PARTIAL,
        "address": MaskingStrategy.PARTIAL,
        "password": MaskingStrategy.FULL,
        "token": MaskingStrategy.FULL,
        "api_key": MaskingStrategy.FULL,
    }
    
    return strategy_map.get(pii_type, MaskingStrategy.PARTIAL)


# 便捷函数
def mask_value(value: str, preserve_prefix: int = 2, 
               preserve_suffix: int = 2) -> str:
    """便捷函数：掩码脱敏"""
    sanitizer = DataSanitizer()
    return sanitizer.mask(value, preserve_prefix=preserve_prefix, 
                         preserve_suffix=preserve_suffix)


def hash_value(value: str, salt: Optional[str] = None) -> str:
    """便捷函数：哈希脱敏"""
    sanitizer = DataSanitizer()
    return sanitizer.hash(value, salt)


def sanitize_text(text: str, pii_type: str) -> str:
    """便捷函数：根据PII类型脱敏"""
    sanitizer = DataSanitizer()
    strategy = sanitizer_for_type(pii_type)
    return sanitizer.sanitize_with_strategy(text, strategy)


# 示例用法
if __name__ == "__main__":
    import math
    
    sanitizer = DataSanitizer()
    
    print("数据脱敏测试：")
    print("=" * 60)
    
    # 测试掩码脱敏
    test_values = [
        ("13800138000", "phone"),
        ("zhangsan@example.com", "email"),
        ("110101199001011234", "id_card"),
        ("4532015112830366", "credit_card"),
        ("张三", "name"),
    ]
    
    for value, pii_type in test_values:
        print(f"\n原始值 ({pii_type}): {value}")
        
        strategy = sanitizer_for_type(pii_type)
        masked = sanitizer.sanitize_with_strategy(value, strategy)
        print(f"策略脱敏 ({strategy.value}): {masked}")
        
        hashed = sanitizer.hash(value, salt="mysalt")
        print(f"哈希脱敏: {hashed[:20]}...")
        
        token = sanitizer.tokenize(value)
        print(f"令牌化: {token}")
    
    # 测试差分隐私
    print("\n差分隐私扰动测试：")
    for _ in range(5):
        original = 100
        perturbed = sanitizer.perturb(original, epsilon=0.5)
        print(f"原始值: {original}, 扰动后: {perturbed:.2f}")
    
    # 测试字典脱敏
    print("\n字典脱敏测试：")
    test_dict = {
        "name": "张三",
        "email": "zhangsan@example.com",
        "phone": "13800138000",
        "address": "北京市朝阳区xxx街道",
        "age": 30
    }
    
    sanitized_dict = sanitizer.sanitize_dict(test_dict)
    print("原始字典:", test_dict)
    print("脱敏字典:", sanitized_dict)
    
    # 测试可逆脱敏
    print("\n可逆脱敏测试：")
    reversible = ReversibleSanitizer()
    original = "sensitive_data_123"
    token = reversible.sanitize(original)
    restored = reversible.restore(token)
    print(f"原始值: {original}")
    print(f"Token: {token}")
    print(f"恢复值: {restored}")
