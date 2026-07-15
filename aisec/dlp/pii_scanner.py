"""
PII Scanner Module - 个人身份信息扫描器

提供全面的PII识别和扫描能力，支持多种PII类型：
- 中国：身份证号、手机号、银行卡号、护照号、车牌号
- 国际：邮箱、信用卡号、SSN、IP地址、MAC地址
- 通用：姓名、地址、公司名称
"""

import re
import json
import hashlib
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, Pattern, Union, Tuple
from pathlib import Path


class Severity(Enum):
    """PII严重级别枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PIICategory(Enum):
    """PII类别枚举"""
    IDENTITY = "identity"           # 身份标识
    FINANCIAL = "financial"         # 财务信息
    CONTACT = "contact"             # 联系方式
    LOCATION = "location"           # 位置信息
    BIOMETRIC = "biometric"         # 生物特征
    HEALTH = "health"               # 健康信息
    LEGAL = "legal"                 # 法律信息


@dataclass
class PIIMatch:
    """PII匹配结果"""
    pii_type: str
    value: str
    position: Tuple[int, int]
    confidence: float
    severity: Severity
    category: PIICategory
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "pii_type": self.pii_type,
            "value": self.value[:20] + "..." if len(self.value) > 20 else self.value,
            "position": self.position,
            "confidence": self.confidence,
            "severity": self.severity.value,
            "category": self.category.value,
            "metadata": self.metadata
        }
    
    def __hash__(self):
        return hash((self.pii_type, self.value, self.position))


@dataclass
class PIIPattern:
    """PII模式定义"""
    name: str
    regex: Pattern
    validator: Optional[Callable[[str], bool]]
    severity: Severity
    category: PIICategory
    description: str = ""
    examples: List[str] = field(default_factory=list)
    
    def match(self, text: str) -> List[PIIMatch]:
        """在文本中匹配此模式"""
        matches = []
        for match in self.regex.finditer(text):
            value = match.group(0)
            
            # 如果提供了验证器，进行验证
            confidence = 1.0
            if self.validator:
                if not self.validator(value):
                    continue
                confidence = 0.95
            
            matches.append(PIIMatch(
                pii_type=self.name,
                value=value,
                position=(match.start(), match.end()),
                confidence=confidence,
                severity=self.severity,
                category=self.category,
                metadata={"pattern": self.name}
            ))
        return matches


class PIIScanner:
    """PII扫描器 - 识别文本中的个人身份信息"""
    
    # 中国手机号段
    CHINA_MOBILE_PREFIXES = [
        '134', '135', '136', '137', '138', '139', '147', '150', '151', '152',
        '157', '158', '159', '178', '182', '183', '184', '187', '188', '198',  # 移动
        '130', '131', '132', '145', '155', '156', '166', '175', '176', '185', '186',  # 联通
        '133', '149', '153', '173', '177', '180', '181', '189', '199',  # 电信
        '170', '171'  # 虚拟运营商
    ]
    
    def __init__(self):
        self.patterns: Dict[str, PIIPattern] = {}
        self.matches: List[PIIMatch] = []
        self._init_default_patterns()
    
    def _init_default_patterns(self):
        """初始化默认PII模式"""
        # 中国身份证号 (18位)
        self.add_pattern(PIIPattern(
            name="china_id_card",
            regex=re.compile(r'\d{17}[\dXx]'),
            validator=self._validate_china_id_card,
            severity=Severity.CRITICAL,
            category=PIICategory.IDENTITY,
            description="中国居民身份证号码",
            examples=["110101199001011234"]
        ))
        
        # 中国手机号
        self.add_pattern(PIIPattern(
            name="china_mobile",
            regex=re.compile(r'1[3-9]\d{9}'),
            validator=self._validate_china_mobile,
            severity=Severity.HIGH,
            category=PIICategory.CONTACT,
            description="中国手机号码",
            examples=["13800138000"]
        ))
        
        # 中国银行卡号 (16-19位)
        self.add_pattern(PIIPattern(
            name="china_bank_card",
            regex=re.compile(r'\d{16,19}'),
            validator=self._validate_bank_card,
            severity=Severity.CRITICAL,
            category=PIICategory.FINANCIAL,
            description="银行卡号",
            examples=["6222021234567890123"]
        ))
        
        # 中国护照号
        self.add_pattern(PIIPattern(
            name="china_passport",
            regex=re.compile(r'[EG]\d{8}|[PS]\d{7}'),
            validator=None,
            severity=Severity.CRITICAL,
            category=PIICategory.IDENTITY,
            description="中国护照号码",
            examples=["E12345678"]
        ))
        
        # 中国车牌号
        self.add_pattern(PIIPattern(
            name="china_license_plate",
            regex=re.compile(r'[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领][A-Z][A-Z0-9]{4,5}[A-Z0-9挂学警港澳]'),
            validator=None,
            severity=Severity.MEDIUM,
            category=PIICategory.LOCATION,
            description="中国车牌号码",
            examples=["京A12345"]
        ))
        
        # 邮箱地址
        self.add_pattern(PIIPattern(
            name="email",
            regex=re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
            validator=None,
            severity=Severity.MEDIUM,
            category=PIICategory.CONTACT,
            description="电子邮件地址",
            examples=["user@example.com"]
        ))
        
        # 信用卡号 (Luhn校验)
        self.add_pattern(PIIPattern(
            name="credit_card",
            regex=re.compile(r'\b(?:4\d{12}(?:\d{3})?|5[1-5]\d{14}|3[47]\d{13}|3(?:0[0-5]|[68]\d)\d{11}|6(?:011|5\d{2})\d{12}|(?:2131|1800|35\d{3})\d{11})\b'),
            validator=self._validate_luhn,
            severity=Severity.CRITICAL,
            category=PIICategory.FINANCIAL,
            description="信用卡号码",
            examples=["4532015112830366"]
        ))
        
        # 美国SSN
        self.add_pattern(PIIPattern(
            name="us_ssn",
            regex=re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            validator=None,
            severity=Severity.CRITICAL,
            category=PIICategory.IDENTITY,
            description="美国社会安全号码",
            examples=["123-45-6789"]
        ))
        
        # IPv4地址
        self.add_pattern(PIIPattern(
            name="ipv4_address",
            regex=re.compile(r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'),
            validator=None,
            severity=Severity.LOW,
            category=PIICategory.LOCATION,
            description="IPv4地址",
            examples=["192.168.1.1"]
        ))
        
        # IPv6地址
        self.add_pattern(PIIPattern(
            name="ipv6_address",
            regex=re.compile(r'(([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:))'),
            validator=None,
            severity=Severity.LOW,
            category=PIICategory.LOCATION,
            description="IPv6地址",
            examples=["2001:0db8:85a3:0000:0000:8a2e:0370:7334"]
        ))
        
        # MAC地址
        self.add_pattern(PIIPattern(
            name="mac_address",
            regex=re.compile(r'([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})'),
            validator=None,
            severity=Severity.LOW,
            category=PIICategory.LOCATION,
            description="MAC地址",
            examples=["00:1B:44:11:3A:B7"]
        ))
        
        # 姓名 (简单模式)
        self.add_pattern(PIIPattern(
            name="person_name",
            regex=re.compile(r'(?:姓名|Name)[:：\s]*([\u4e00-\u9fa5]{2,4}|[A-Z][a-z]+\s[A-Z][a-z]+)'),
            validator=None,
            severity=Severity.MEDIUM,
            category=PIICategory.IDENTITY,
            description="个人姓名",
            examples=["张三", "John Smith"]
        ))
        
        # 地址
        self.add_pattern(PIIPattern(
            name="address",
            regex=re.compile(r'(?:地址|Address)[:：\s]*([\u4e00-\u9fa5]{5,30}|[\w\s,.-]{10,100})'),
            validator=None,
            severity=Severity.HIGH,
            category=PIICategory.LOCATION,
            description="地址信息",
            examples=["北京市朝阳区xxx街道xxx号"]
        ))
        
        # 公司名称
        self.add_pattern(PIIPattern(
            name="company_name",
            regex=re.compile(r'(?:公司|Company)[:：\s]*([\u4e00-\u9fa5]{4,20}|\w+\s+(?:Inc\.?|Ltd\.?|Corp\.?|LLC|Company))'),
            validator=None,
            severity=Severity.LOW,
            category=PIICategory.IDENTITY,
            description="公司名称",
            examples=["某某科技有限公司", "ABC Inc."]
        ))
    
    def add_pattern(self, pattern: PIIPattern):
        """添加自定义PII模式"""
        self.patterns[pattern.name] = pattern
    
    def remove_pattern(self, name: str):
        """移除PII模式"""
        if name in self.patterns:
            del self.patterns[name]
    
    def scan_text(self, text: str, pattern_names: Optional[List[str]] = None) -> List[PIIMatch]:
        """
        扫描文本中的PII
        
        Args:
            text: 要扫描的文本
            pattern_names: 指定要扫描的模式名称列表，None表示扫描所有
            
        Returns:
            匹配到的PII列表
        """
        if not text or not isinstance(text, str):
            return []
        
        matches = []
        patterns_to_scan = []
        
        if pattern_names:
            patterns_to_scan = [self.patterns[name] for name in pattern_names if name in self.patterns]
        else:
            patterns_to_scan = list(self.patterns.values())
        
        for pattern in patterns_to_scan:
            pattern_matches = pattern.match(text)
            matches.extend(pattern_matches)
        
        # 去重（基于位置和类型）
        seen = set()
        unique_matches = []
        for match in matches:
            key = (match.position, match.pii_type)
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)
        
        self.matches = unique_matches
        return unique_matches
    
    def scan_file(self, file_path: Union[str, Path], encoding: str = 'utf-8') -> List[PIIMatch]:
        """
        扫描文件中的PII
        
        Args:
            file_path: 文件路径
            encoding: 文件编码
            
        Returns:
            匹配到的PII列表
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 根据文件扩展名选择处理方式
        suffix = file_path.suffix.lower()
        
        if suffix in ['.json']:
            return self._scan_json_file(file_path, encoding)
        elif suffix in ['.csv']:
            return self._scan_csv_file(file_path, encoding)
        else:
            # 文本文件
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                content = f.read()
            return self.scan_text(content)
    
    def _scan_json_file(self, file_path: Path, encoding: str) -> List[PIIMatch]:
        """扫描JSON文件"""
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            data = json.load(f)
        return self.scan_dict(data)
    
    def _scan_csv_file(self, file_path: Path, encoding: str) -> List[PIIMatch]:
        """扫描CSV文件"""
        import csv
        matches = []
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            reader = csv.reader(f)
            for row in reader:
                for cell in row:
                    cell_matches = self.scan_text(str(cell))
                    matches.extend(cell_matches)
        return matches
    
    def scan_dict(self, data: Dict, max_depth: int = 10) -> List[PIIMatch]:
        """
        递归扫描字典中的PII
        
        Args:
            data: 要扫描的字典
            max_depth: 最大递归深度
            
        Returns:
            匹配到的PII列表
        """
        matches = []
        self._scan_dict_recursive(data, matches, 0, max_depth)
        return matches
    
    def _scan_dict_recursive(self, data: Any, matches: List[PIIMatch], depth: int, max_depth: int):
        """递归扫描字典的内部实现"""
        if depth > max_depth:
            return
        
        if isinstance(data, dict):
            for key, value in data.items():
                # 扫描键
                key_matches = self.scan_text(str(key))
                for match in key_matches:
                    match.metadata['in_key'] = True
                matches.extend(key_matches)
                
                # 递归扫描值
                self._scan_dict_recursive(value, matches, depth + 1, max_depth)
        elif isinstance(data, list):
            for item in data:
                self._scan_dict_recursive(item, matches, depth + 1, max_depth)
        elif isinstance(data, str):
            matches.extend(self.scan_text(data))
        elif isinstance(data, (int, float)):
            matches.extend(self.scan_text(str(data)))
    
    def batch_scan(self, texts: List[str]) -> Dict[int, List[PIIMatch]]:
        """
        批量扫描文本
        
        Args:
            texts: 文本列表
            
        Returns:
            字典，键为索引，值为匹配结果列表
        """
        results = {}
        for i, text in enumerate(texts):
            results[i] = self.scan_text(text)
        return results
    
    def get_matches(self) -> List[PIIMatch]:
        """获取最后一次扫描的匹配结果"""
        return self.matches
    
    def get_matches_by_severity(self, severity: Severity) -> List[PIIMatch]:
        """按严重级别获取匹配结果"""
        return [m for m in self.matches if m.severity == severity]
    
    def get_matches_by_category(self, category: PIICategory) -> List[PIIMatch]:
        """按类别获取匹配结果"""
        return [m for m in self.matches if m.category == category]
    
    def has_pii(self, text: str) -> bool:
        """检查文本是否包含PII"""
        return len(self.scan_text(text)) > 0
    
    def get_pii_summary(self, text: str) -> Dict[str, Any]:
        """获取PII摘要统计"""
        matches = self.scan_text(text)
        
        summary = {
            "total_matches": len(matches),
            "by_type": {},
            "by_severity": {s.value: 0 for s in Severity},
            "by_category": {c.value: 0 for c in PIICategory},
            "highest_severity": None
        }
        
        severity_order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        highest = None
        
        for match in matches:
            # 按类型统计
            if match.pii_type not in summary["by_type"]:
                summary["by_type"][match.pii_type] = 0
            summary["by_type"][match.pii_type] += 1
            
            # 按严重级别统计
            summary["by_severity"][match.severity.value] += 1
            
            # 按类别统计
            summary["by_category"][match.category.value] += 1
            
            # 更新最高严重级别
            if highest is None or severity_order.index(match.severity) > severity_order.index(highest):
                highest = match.severity
        
        summary["highest_severity"] = highest.value if highest else None
        return summary
    
    # ============ 验证器方法 ============
    
    @staticmethod
    def _validate_luhn(card_number: str) -> bool:
        """
        Luhn算法验证信用卡号
        
        Args:
            card_number: 信用卡号
            
        Returns:
            是否有效
        """
        # 移除非数字字符
        digits = ''.join(c for c in card_number if c.isdigit())
        
        if len(digits) < 13 or len(digits) > 19:
            return False
        
        # Luhn算法
        total = 0
        reverse_digits = digits[::-1]
        
        for i, digit in enumerate(reverse_digits):
            n = int(digit)
            if i % 2 == 1:
                n *= 2
                if n > 9:
                    n -= 9
            total += n
        
        return total % 10 == 0
    
    @staticmethod
    def _validate_china_id_card(id_number: str) -> bool:
        """
        验证中国身份证号
        
        Args:
            id_number: 身份证号
            
        Returns:
            是否有效
        """
        if len(id_number) != 18:
            return False
        
        # 前17位必须是数字
        if not id_number[:17].isdigit():
            return False
        
        # 最后一位可以是数字或X
        if not (id_number[17].isdigit() or id_number[17].upper() == 'X'):
            return False
        
        # 加权因子
        weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
        
        # 校验码映射
        check_codes = ['1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2']
        
        # 计算校验码
        total = sum(int(id_number[i]) * weights[i] for i in range(17))
        expected_check = check_codes[total % 11]
        
        return id_number[17].upper() == expected_check
    
    @staticmethod
    def _validate_china_mobile(mobile: str) -> bool:
        """
        验证中国手机号
        
        Args:
            mobile: 手机号
            
        Returns:
            是否有效
        """
        if len(mobile) != 11:
            return False
        
        if not mobile.isdigit():
            return False
        
        # 检查号段
        prefix = mobile[:3]
        valid_prefixes = PIIScanner.CHINA_MOBILE_PREFIXES
        
        return prefix in valid_prefixes
    
    @staticmethod
    def _validate_bank_card(card_number: str) -> bool:
        """
        验证银行卡号
        
        Args:
            card_number: 银行卡号
            
        Returns:
            是否有效
        """
        if len(card_number) < 16 or len(card_number) > 19:
            return False
        
        if not card_number.isdigit():
            return False
        
        # 使用Luhn算法验证
        return PIIScanner._validate_luhn(card_number)


# 便捷函数
def scan_text(text: str) -> List[PIIMatch]:
    """便捷函数：扫描文本中的PII"""
    scanner = PIIScanner()
    return scanner.scan_text(text)


def scan_file(file_path: Union[str, Path]) -> List[PIIMatch]:
    """便捷函数：扫描文件中的PII"""
    scanner = PIIScanner()
    return scanner.scan_file(file_path)


def has_pii(text: str) -> bool:
    """便捷函数：检查文本是否包含PII"""
    scanner = PIIScanner()
    return scanner.has_pii(text)


# 示例用法
if __name__ == "__main__":
    scanner = PIIScanner()
    
    # 测试文本
    test_text = """
    用户信息：
    姓名：张三
    身份证号：110101199001011234
    手机号：13800138000
    邮箱：zhangsan@example.com
    地址：北京市朝阳区xxx街道xxx号
    银行卡：6222021234567890123
    
    公司信息：
    公司名称：某某科技有限公司
    
    其他：
    IP地址：192.168.1.1
    MAC地址：00:1B:44:11:3A:B7
    信用卡：4532015112830366
    护照号：E12345678
    车牌号：京A12345
    """
    
    matches = scanner.scan_text(test_text)
    
    print("扫描结果：")
    print("=" * 50)
    for match in matches:
        print(f"类型: {match.pii_type}")
        print(f"值: {match.value}")
        print(f"位置: {match.position}")
        print(f"置信度: {match.confidence}")
        print(f"严重级别: {match.severity.value}")
        print(f"类别: {match.category.value}")
        print("-" * 50)
    
    print("\nPII摘要：")
    summary = scanner.get_pii_summary(test_text)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
