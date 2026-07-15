"""
字段脱敏 - 敏感字段掩码处理
"""
import re
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum


class MaskStrategy(Enum):
    """掩码策略"""
    FULL = "full"               # 完全掩码
    PARTIAL = "partial"         # 部分掩码
    HASH = "hash"               # 哈希掩码
    REPLACE = "replace"         # 替换掩码
    REMOVE = "remove"           # 移除
    KEEP_FIRST = "keep_first"   # 保留前N位
    KEEP_LAST = "keep_last"     # 保留后N位
    KEEP_MIDDLE = "keep_middle" # 保留中间


@dataclass
class FieldRule:
    """字段规则"""
    field_name: str
    mask_strategy: MaskStrategy
    mask_char: str = "*"
    keep_chars: int = 2         # 保留字符数
    replacement: str = "[REDACTED]"
    case_sensitive: bool = False
    pattern: Optional[re.Pattern] = None
    
    def __post_init__(self):
        if self.pattern is None:
            # 创建字段匹配模式
            flags = 0 if self.case_sensitive else re.IGNORECASE
            self.pattern = re.compile(
                rf'("{self.field_name}"\s*:\s*["\']?)([^"\',\s}}]+)(["\']?)',
                flags
            )


class FieldMasker:
    """字段脱敏器"""
    
    def __init__(self):
        self._rules: Dict[str, FieldRule] = {}
        self._custom_handlers: Dict[str, Callable[[Any], str]] = {}
        self._load_default_rules()
    
    def _load_default_rules(self) -> None:
        """加载默认规则"""
        default_rules = [
            FieldRule("password", MaskStrategy.FULL, mask_char="*"),
            FieldRule("passwd", MaskStrategy.FULL, mask_char="*"),
            FieldRule("pwd", MaskStrategy.FULL, mask_char="*"),
            FieldRule("secret", MaskStrategy.FULL, mask_char="*"),
            FieldRule("secret_key", MaskStrategy.FULL, mask_char="*"),
            FieldRule("api_key", MaskStrategy.KEEP_LAST, keep_chars=4),
            FieldRule("apikey", MaskStrategy.KEEP_LAST, keep_chars=4),
            FieldRule("token", MaskStrategy.KEEP_LAST, keep_chars=4),
            FieldRule("access_token", MaskStrategy.KEEP_LAST, keep_chars=4),
            FieldRule("auth_token", MaskStrategy.KEEP_LAST, keep_chars=4),
            FieldRule("private_key", MaskStrategy.REMOVE),
            FieldRule("credit_card", MaskStrategy.KEEP_LAST, keep_chars=4),
            FieldRule("card_number", MaskStrategy.KEEP_LAST, keep_chars=4),
            FieldRule("ssn", MaskStrategy.KEEP_LAST, keep_chars=4),
            FieldRule("email", MaskStrategy.PARTIAL, keep_chars=2),
            FieldRule("phone", MaskStrategy.KEEP_LAST, keep_chars=4),
            FieldRule("mobile", MaskStrategy.KEEP_LAST, keep_chars=4),
            FieldRule("address", MaskStrategy.REPLACE),
            FieldRule("ip_address", MaskStrategy.KEEP_LAST, keep_chars=2),
        ]
        
        for rule in default_rules:
            self._rules[rule.field_name] = rule
    
    def add_rule(self, rule: FieldRule) -> None:
        """添加规则"""
        self._rules[rule.field_name] = rule
    
    def add_custom_handler(self, field_name: str, handler: Callable[[Any], str]) -> None:
        """添加自定义处理器"""
        self._custom_handlers[field_name] = handler
    
    def mask_value(self, value: Any, rule: FieldRule) -> str:
        """掩码单个值"""
        if value is None:
            return ""
        
        str_value = str(value)
        
        if rule.mask_strategy == MaskStrategy.FULL:
            return rule.mask_char * len(str_value)
        
        elif rule.mask_strategy == MaskStrategy.PARTIAL:
            if len(str_value) <= rule.keep_chars * 2:
                return str_value[:rule.keep_chars] + rule.mask_char * (len(str_value) - rule.keep_chars)
            return str_value[:rule.keep_chars] + rule.mask_char * (len(str_value) - rule.keep_chars * 2) + str_value[-rule.keep_chars:]
        
        elif rule.mask_strategy == MaskStrategy.KEEP_FIRST:
            return str_value[:rule.keep_chars] + rule.mask_char * (len(str_value) - rule.keep_chars)
        
        elif rule.mask_strategy == MaskStrategy.KEEP_LAST:
            return rule.mask_char * (len(str_value) - rule.keep_chars) + str_value[-rule.keep_chars:]
        
        elif rule.mask_strategy == MaskStrategy.KEEP_MIDDLE:
            start = (len(str_value) - rule.keep_chars) // 2
            return rule.mask_char * start + str_value[start:start + rule.keep_chars] + rule.mask_char * (len(str_value) - start - rule.keep_chars)
        
        elif rule.mask_strategy == MaskStrategy.HASH:
            import hashlib
            return hashlib.sha256(str_value.encode()).hexdigest()[:8]
        
        elif rule.mask_strategy == MaskStrategy.REPLACE:
            return rule.replacement
        
        elif rule.mask_strategy == MaskStrategy.REMOVE:
            return ""
        
        return str_value
    
    def mask_text(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """掩码文本中的敏感字段"""
        masked_text = text
        changes = []
        
        for field_name, rule in self._rules.items():
            # 检查自定义处理器
            if field_name in self._custom_handlers:
                continue
            
            for match in rule.pattern.finditer(text):
                original = match.group(2)
                masked = self.mask_value(original, rule)
                
                if original != masked:
                    changes.append({
                        "field": field_name,
                        "original_length": len(original),
                        "strategy": rule.mask_strategy.value,
                        "position": match.start()
                    })
                    
                    # 替换
                    full_match = match.group(0)
                    new_match = match.group(1) + masked + match.group(3)
                    masked_text = masked_text.replace(full_match, new_match, 1)
        
        return masked_text, changes
    
    def mask_dict(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """掩码字典中的敏感字段"""
        result = data.copy()
        changes = []
        
        def _process_dict(d: Dict[str, Any], path: str = "") -> None:
            for key, value in list(d.items()):
                current_path = f"{path}.{key}" if path else key
                
                # 检查是否需要掩码
                key_lower = key.lower()
                matched_rule = None
                
                for rule_name, rule in self._rules.items():
                    if rule_name.lower() == key_lower:
                        matched_rule = rule
                        break
                
                if matched_rule and value is not None:
                    original_value = value
                    
                    # 使用自定义处理器或默认掩码
                    if key_lower in self._custom_handlers:
                        d[key] = self._custom_handlers[key_lower](value)
                    else:
                        d[key] = self.mask_value(value, matched_rule)
                    
                    if str(original_value) != str(d[key]):
                        changes.append({
                            "field": current_path,
                            "strategy": matched_rule.mask_strategy.value,
                            "original_length": len(str(original_value))
                        })
                
                # 递归处理嵌套字典
                elif isinstance(value, dict):
                    _process_dict(value, current_path)
                
                # 处理列表中的字典
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            _process_dict(item, f"{current_path}[{i}]")
        
        _process_dict(result)
        return result, changes
    
    def mask_json(self, json_str: str) -> Tuple[str, List[Dict[str, Any]]]:
        """掩码JSON字符串"""
        import json
        
        try:
            data = json.loads(json_str)
            masked_data, changes = self.mask_dict(data)
            return json.dumps(masked_data), changes
        except json.JSONDecodeError:
            # 如果不是有效JSON，按文本处理
            return self.mask_text(json_str)
    
    def add_field_pattern(
        self,
        field_name: str,
        pattern: str,
        mask_strategy: MaskStrategy,
        **kwargs
    ) -> None:
        """添加带正则模式的字段规则"""
        rule = FieldRule(
            field_name=field_name,
            mask_strategy=mask_strategy,
            pattern=re.compile(pattern, re.IGNORECASE),
            **kwargs
        )
        self._rules[field_name] = rule
    
    def get_rules(self) -> List[Dict[str, Any]]:
        """获取所有规则"""
        return [
            {
                "field_name": rule.field_name,
                "strategy": rule.mask_strategy.value,
                "mask_char": rule.mask_char,
                "keep_chars": rule.keep_chars
            }
            for rule in self._rules.values()
        ]
    
    def remove_rule(self, field_name: str) -> bool:
        """移除规则"""
        if field_name in self._rules:
            del self._rules[field_name]
            return True
        return False
