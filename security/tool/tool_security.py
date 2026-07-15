"""
工具安全模块 - Tool Security

实现工具调用安全防护机制：
1. 工具调用验证 (Tool Call Validation)
2. 权限管理 (Permission Management)
3. 工具输出净化 (Tool Output Sanitization)
4. 工具滥用检测 (Tool Abuse Detection)
"""

import re
import json
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Set, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
from abc import ABC, abstractmethod
import threading


class PermissionLevel(Enum):
    """权限等级"""
    NONE = 0
    READ = 1
    WRITE = 2
    EXECUTE = 3
    ADMIN = 4


class ToolCategory(Enum):
    """工具类别"""
    FILE_SYSTEM = "file_system"
    NETWORK = "network"
    DATABASE = "database"
    SYSTEM = "system"
    CODE_EXECUTION = "code_execution"
    EXTERNAL_API = "external_api"
    DATA_PROCESSING = "data_processing"


@dataclass
class ToolCall:
    """工具调用请求"""
    tool_name: str
    arguments: Dict[str, Any]
    call_id: str
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    allowed: bool
    risk_score: float
    violations: List[Dict[str, Any]]
    sanitized_args: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSecurityConfig:
    """工具安全配置"""
    # 验证选项
    strict_mode: bool = True
    validate_arguments: bool = True
    sanitize_outputs: bool = True
    
    # 限制选项
    max_calls_per_minute: int = 60
    max_concurrent_calls: int = 10
    max_argument_size: int = 1024 * 1024  # 1MB
    max_output_size: int = 10 * 1024 * 1024  # 10MB
    
    # 超时设置
    call_timeout: float = 30.0
    total_timeout: float = 300.0
    
    # 滥用检测
    enable_abuse_detection: bool = True
    abuse_threshold: int = 10
    abuse_window: float = 60.0  # 秒
    
    # 敏感数据检测
    detect_sensitive_data: bool = True
    redact_sensitive_data: bool = True


class ToolSchema:
    """
    工具模式定义
    
    定义工具的结构、参数和约束。
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        category: ToolCategory,
        parameters: Dict[str, Any],
        required_permissions: List[PermissionLevel],
        dangerous: bool = False,
        rate_limit: Optional[int] = None
    ):
        self.name = name
        self.description = description
        self.category = category
        self.parameters = parameters
        self.required_permissions = required_permissions
        self.dangerous = dangerous
        self.rate_limit = rate_limit
    
    def validate_parameters(self, arguments: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        验证参数是否符合模式
        
        Returns:
            (是否有效, 错误列表)
        """
        errors = []
        
        # 检查必需参数
        required = self.parameters.get('required', [])
        for param in required:
            if param not in arguments:
                errors.append(f"Missing required parameter: {param}")
        
        # 检查参数类型
        properties = self.parameters.get('properties', {})
        for param, value in arguments.items():
            if param in properties:
                param_schema = properties[param]
                param_type = param_schema.get('type')
                
                if param_type == 'string' and not isinstance(value, str):
                    errors.append(f"Parameter '{param}' must be a string")
                elif param_type == 'integer' and not isinstance(value, int):
                    errors.append(f"Parameter '{param}' must be an integer")
                elif param_type == 'number' and not isinstance(value, (int, float)):
                    errors.append(f"Parameter '{param}' must be a number")
                elif param_type == 'boolean' and not isinstance(value, bool):
                    errors.append(f"Parameter '{param}' must be a boolean")
                elif param_type == 'array' and not isinstance(value, list):
                    errors.append(f"Parameter '{param}' must be an array")
                elif param_type == 'object' and not isinstance(value, dict):
                    errors.append(f"Parameter '{param}' must be an object")
                
                # 检查枚举值
                if 'enum' in param_schema:
                    if value not in param_schema['enum']:
                        errors.append(f"Parameter '{param}' must be one of {param_schema['enum']}")
                
                # 检查字符串模式
                if param_type == 'string' and 'pattern' in param_schema:
                    if not re.match(param_schema['pattern'], value):
                        errors.append(f"Parameter '{param}' does not match required pattern")
                
                # 检查数值范围
                if param_type in ('integer', 'number'):
                    if 'minimum' in param_schema and value < param_schema['minimum']:
                        errors.append(f"Parameter '{param}' must be >= {param_schema['minimum']}")
                    if 'maximum' in param_schema and value > param_schema['maximum']:
                        errors.append(f"Parameter '{param}' must be <= {param_schema['maximum']}")
        
        return len(errors) == 0, errors


class ToolRegistry:
    """
    工具注册表
    
    管理所有可用工具及其安全元数据。
    """
    
    def __init__(self):
        self._tools: Dict[str, ToolSchema] = {}
        self._categories: Dict[ToolCategory, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()
    
    def register(self, schema: ToolSchema) -> None:
        """注册工具"""
        with self._lock:
            self._tools[schema.name] = schema
            self._categories[schema.category].add(schema.name)
    
    def unregister(self, tool_name: str) -> bool:
        """注销工具"""
        with self._lock:
            if tool_name in self._tools:
                schema = self._tools.pop(tool_name)
                self._categories[schema.category].discard(tool_name)
                return True
            return False
    
    def get(self, tool_name: str) -> Optional[ToolSchema]:
        """获取工具模式"""
        with self._lock:
            return self._tools.get(tool_name)
    
    def list_tools(self, category: Optional[ToolCategory] = None) -> List[str]:
        """列出工具"""
        with self._lock:
            if category:
                return list(self._categories[category])
            return list(self._tools.keys())
    
    def get_by_category(self, category: ToolCategory) -> List[ToolSchema]:
        """按类别获取工具"""
        with self._lock:
            return [
                self._tools[name] 
                for name in self._categories[category]
            ]


class PermissionManager:
    """
    权限管理器
    
    管理用户和工具的权限关系。
    """
    
    def __init__(self):
        # 用户权限: user_id -> {tool_name -> permission_level}
        self._user_permissions: Dict[str, Dict[str, PermissionLevel]] = defaultdict(dict)
        
        # 角色权限: role -> {category -> permission_level}
        self._role_permissions: Dict[str, Dict[ToolCategory, PermissionLevel]] = defaultdict(dict)
        
        # 用户角色: user_id -> set of roles
        self._user_roles: Dict[str, Set[str]] = defaultdict(set)
        
        # 危险工具黑名单
        self._blacklist: Set[str] = set()
        
        # 白名单
        self._whitelist: Set[str] = set()
        
        self._lock = threading.RLock()
    
    def grant_permission(
        self,
        user_id: str,
        tool_name: str,
        level: PermissionLevel
    ) -> None:
        """授予用户工具权限"""
        with self._lock:
            self._user_permissions[user_id][tool_name] = level
    
    def revoke_permission(self, user_id: str, tool_name: str) -> None:
        """撤销用户工具权限"""
        with self._lock:
            if tool_name in self._user_permissions[user_id]:
                del self._user_permissions[user_id][tool_name]
    
    def assign_role(self, user_id: str, role: str) -> None:
        """分配角色给用户"""
        with self._lock:
            self._user_roles[user_id].add(role)
    
    def remove_role(self, user_id: str, role: str) -> None:
        """移除用户角色"""
        with self._lock:
            self._user_roles[user_id].discard(role)
    
    def set_role_permission(
        self,
        role: str,
        category: ToolCategory,
        level: PermissionLevel
    ) -> None:
        """设置角色对某类工具的权限"""
        with self._lock:
            self._role_permissions[role][category] = level
    
    def check_permission(
        self,
        user_id: str,
        tool_name: str,
        required_level: PermissionLevel,
        tool_category: Optional[ToolCategory] = None
    ) -> bool:
        """
        检查用户是否有权限使用工具
        
        Args:
            user_id: 用户ID
            tool_name: 工具名称
            required_level: 所需权限等级
            tool_category: 工具类别
            
        Returns:
            是否有权限
        """
        with self._lock:
            # 检查黑名单
            if tool_name in self._blacklist:
                return False
            
            # 检查白名单
            if self._whitelist and tool_name not in self._whitelist:
                return False
            
            # 检查直接权限
            user_perm = self._user_permissions[user_id].get(tool_name)
            if user_perm is not None and user_perm.value >= required_level.value:
                return True
            
            # 检查角色权限
            for role in self._user_roles[user_id]:
                if tool_category and tool_category in self._role_permissions[role]:
                    role_perm = self._role_permissions[role][tool_category]
                    if role_perm.value >= required_level.value:
                        return True
            
            return False
    
    def add_to_blacklist(self, tool_name: str) -> None:
        """添加到黑名单"""
        with self._lock:
            self._blacklist.add(tool_name)
    
    def remove_from_blacklist(self, tool_name: str) -> None:
        """从黑名单移除"""
        with self._lock:
            self._blacklist.discard(tool_name)
    
    def add_to_whitelist(self, tool_name: str) -> None:
        """添加到白名单"""
        with self._lock:
            self._whitelist.add(tool_name)
    
    def get_user_permissions(self, user_id: str) -> Dict[str, PermissionLevel]:
        """获取用户权限"""
        with self._lock:
            return dict(self._user_permissions[user_id])


class ToolCallValidator:
    """
    工具调用验证器
    
    验证工具调用的合法性：
    - 参数类型检查
    - 参数范围验证
    - 危险参数检测
    - 注入攻击防护
    """
    
    # 危险模式
    DANGEROUS_PATTERNS = {
        'path_traversal': [
            r'\.\./',
            r'\.\.\\',
            r'%2e%2e/',
            r'%2e%2e%2f',
            r'\.\.//',
            r'\\.\.\\',
        ],
        'command_injection': [
            r'[`;|&$\n\r]',
            r'\$\(',
            r'`[^`]+`',
            r'\|\s*\w+',
            r';\s*\w+',
            r'&&\s*\w+',
        ],
        'sql_injection': [
            r'\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION)\b',
            r'\bOR\s+1\s*=\s*1\b',
            r'\bAND\s+1\s*=\s*1\b',
            r"'\s*OR\s+'",
            r'--\s*$',
            r'/\*.*\*/',
        ],
        'code_injection': [
            r'\beval\s*\(',
            r'\bexec\s*\(',
            r'\bsystem\s*\(',
            r'\bos\.system',
            r'\bsubprocess\.call',
            r'\b__import__\s*\(',
            r'\bcompile\s*\(',
        ],
        'ssrf_patterns': [
            r'http://localhost',
            r'http://127\.0\.0\.1',
            r'http://0\.0\.0\.0',
            r'http://\[::1\]',
            r'file://',
            r'dict://',
            r'gopher://',
            r'ftp://',
        ],
    }
    
    # 危险文件路径
    DANGEROUS_PATHS = [
        '/etc/passwd',
        '/etc/shadow',
        '/etc/hosts',
        '/proc/',
        '/sys/',
        'C:\\Windows\\',
        'C:\\System32',
        '.env',
        '.git/',
        '.ssh/',
        'id_rsa',
        'config.json',
        'credentials',
        'secret',
        'password',
    ]
    
    def __init__(self, registry: ToolRegistry, config: Optional[ToolSecurityConfig] = None):
        self.registry = registry
        self.config = config or ToolSecurityConfig()
        self._compile_patterns()
        self.validation_log: deque = deque(maxlen=1000)
    
    def _compile_patterns(self) -> None:
        """编译正则表达式"""
        self.compiled_patterns = {}
        for category, patterns in self.DANGEROUS_PATTERNS.items():
            self.compiled_patterns[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def validate_tool_exists(self, tool_name: str) -> Tuple[bool, Optional[str]]:
        """验证工具是否存在"""
        schema = self.registry.get(tool_name)
        if schema is None:
            return False, f"Tool '{tool_name}' not found"
        return True, None
    
    def validate_parameters(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """验证参数"""
        schema = self.registry.get(tool_name)
        if schema is None:
            return False, [f"Tool '{tool_name}' not found"]
        
        return schema.validate_parameters(arguments)
    
    def detect_path_traversal(self, value: str) -> Tuple[float, List[Dict]]:
        """检测路径遍历攻击"""
        score = 0.0
        matches = []
        
        for pattern in self.compiled_patterns['path_traversal']:
            if pattern.search(value):
                score += 0.3
                matches.append({
                    'type': 'path_traversal',
                    'pattern': pattern.pattern,
                    'severity': 'critical'
                })
        
        # 检查危险路径
        value_lower = value.lower()
        for dangerous_path in self.DANGEROUS_PATHS:
            if dangerous_path.lower() in value_lower:
                score += 0.4
                matches.append({
                    'type': 'dangerous_path',
                    'path': dangerous_path,
                    'severity': 'critical'
                })
        
        return min(score, 1.0), matches
    
    def detect_command_injection(self, value: str) -> Tuple[float, List[Dict]]:
        """检测命令注入"""
        score = 0.0
        matches = []
        
        for pattern in self.compiled_patterns['command_injection']:
            if pattern.search(value):
                score += 0.35
                matches.append({
                    'type': 'command_injection',
                    'pattern': pattern.pattern,
                    'severity': 'critical'
                })
        
        return min(score, 1.0), matches
    
    def detect_sql_injection(self, value: str) -> Tuple[float, List[Dict]]:
        """检测SQL注入"""
        score = 0.0
        matches = []
        
        for pattern in self.compiled_patterns['sql_injection']:
            if pattern.search(value):
                score += 0.3
                matches.append({
                    'type': 'sql_injection',
                    'pattern': pattern.pattern,
                    'severity': 'high'
                })
        
        return min(score, 1.0), matches
    
    def detect_code_injection(self, value: str) -> Tuple[float, List[Dict]]:
        """检测代码注入"""
        score = 0.0
        matches = []
        
        for pattern in self.compiled_patterns['code_injection']:
            if pattern.search(value):
                score += 0.35
                matches.append({
                    'type': 'code_injection',
                    'pattern': pattern.pattern,
                    'severity': 'critical'
                })
        
        return min(score, 1.0), matches
    
    def detect_ssrf(self, value: str) -> Tuple[float, List[Dict]]:
        """检测SSRF攻击"""
        score = 0.0
        matches = []
        
        for pattern in self.compiled_patterns['ssrf_patterns']:
            if pattern.search(value):
                score += 0.3
                matches.append({
                    'type': 'ssrf',
                    'pattern': pattern.pattern,
                    'severity': 'high'
                })
        
        # 检测IP地址
        ip_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
        if ip_pattern.search(value):
            score += 0.1
            matches.append({
                'type': 'ip_address_in_url',
                'severity': 'medium'
            })
        
        return min(score, 1.0), matches
    
    def analyze_argument(
        self,
        arg_name: str,
        value: Any,
        tool_category: Optional[ToolCategory] = None
    ) -> Tuple[float, List[Dict]]:
        """
        分析单个参数的安全性
        
        Returns:
            (风险分数, 检测结果列表)
        """
        if not isinstance(value, str):
            return 0.0, []
        
        total_score = 0.0
        all_matches = []
        
        # 根据工具类别选择检测策略
        if tool_category == ToolCategory.FILE_SYSTEM:
            score, matches = self.detect_path_traversal(value)
            total_score += score
            all_matches.extend(matches)
        
        elif tool_category == ToolCategory.CODE_EXECUTION:
            score, matches = self.detect_code_injection(value)
            total_score += score
            all_matches.extend(matches)
        
        elif tool_category == ToolCategory.DATABASE:
            score, matches = self.detect_sql_injection(value)
            total_score += score
            all_matches.extend(matches)
        
        elif tool_category == ToolCategory.NETWORK:
            score, matches = self.detect_ssrf(value)
            total_score += score
            all_matches.extend(matches)
        
        # 通用检测：命令注入
        score, matches = self.detect_command_injection(value)
        if score > 0:
            total_score += score * 0.5
            all_matches.extend(matches)
        
        return min(total_score, 1.0), all_matches
    
    def sanitize_argument(self, value: Any, arg_type: str = 'string') -> Any:
        """净化参数值"""
        if not isinstance(value, str):
            return value
        
        sanitized = value
        
        # 移除控制字符
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\n\r\t')
        
        # 规范化路径
        if arg_type in ('path', 'filepath', 'filename'):
            sanitized = re.sub(r'\.+[/\\]', '', sanitized)
            sanitized = sanitized.replace('..', '')
        
        # 限制长度
        if len(sanitized) > 10000:
            sanitized = sanitized[:10000]
        
        return sanitized
    
    def validate(
        self,
        tool_call: ToolCall,
        user_permissions: Optional[PermissionLevel] = None
    ) -> ValidationResult:
        """
        执行完整的工具调用验证
        
        Args:
            tool_call: 工具调用请求
            user_permissions: 用户权限等级
            
        Returns:
            ValidationResult: 验证结果
        """
        violations = []
        risk_score = 0.0
        
        # 1. 检查工具是否存在
        exists, error = self.validate_tool_exists(tool_call.tool_name)
        if not exists:
            violations.append({
                'type': 'tool_not_found',
                'message': error
            })
            return ValidationResult(
                is_valid=False,
                allowed=False,
                risk_score=1.0,
                violations=violations
            )
        
        schema = self.registry.get(tool_call.tool_name)
        
        # 2. 检查权限
        if user_permissions is not None:
            min_required = min(schema.required_permissions, key=lambda x: x.value)
            if user_permissions.value < min_required.value:
                violations.append({
                    'type': 'insufficient_permissions',
                    'required': min_required.name,
                    'granted': user_permissions.name
                })
                risk_score += 0.5
        
        # 3. 验证参数
        if self.config.validate_arguments:
            valid, errors = self.validate_parameters(
                tool_call.tool_name,
                tool_call.arguments
            )
            if not valid:
                for error in errors:
                    violations.append({
                        'type': 'parameter_validation_error',
                        'message': error
                    })
                risk_score += 0.3
        
        # 4. 安全分析
        all_matches = []
        for arg_name, value in tool_call.arguments.items():
            score, matches = self.analyze_argument(
                arg_name,
                value,
                schema.category
            )
            if score > 0:
                risk_score += score * 0.2
                all_matches.extend([
                    {**m, 'argument': arg_name} for m in matches
                ])
        
        violations.extend(all_matches)
        
        # 5. 检查参数大小
        args_size = len(json.dumps(tool_call.arguments))
        if args_size > self.config.max_argument_size:
            violations.append({
                'type': 'argument_too_large',
                'size': args_size,
                'max': self.config.max_argument_size
            })
            risk_score += 0.2
        
        # 6. 净化参数
        sanitized_args = None
        if self.config.strict_mode and risk_score < 0.8:
            sanitized_args = {}
            properties = schema.parameters.get('properties', {})
            for arg_name, value in tool_call.arguments.items():
                arg_type = properties.get(arg_name, {}).get('type', 'string')
                sanitized_args[arg_name] = self.sanitize_argument(value, arg_type)
        
        is_valid = len([v for v in violations if v['type'] in (
            'tool_not_found', 'parameter_validation_error'
        )]) == 0
        
        allowed = risk_score < 0.7 and is_valid
        
        result = ValidationResult(
            is_valid=is_valid,
            allowed=allowed,
            risk_score=min(risk_score, 1.0),
            violations=violations,
            sanitized_args=sanitized_args,
            metadata={
                'tool_category': schema.category.value,
                'is_dangerous': schema.dangerous,
                'validation_time': time.time()
            }
        )
        
        self.validation_log.append({
            'tool_name': tool_call.tool_name,
            'risk_score': risk_score,
            'allowed': allowed,
            'timestamp': time.time()
        })
        
        return result


class ToolOutputSanitizer:
    """
    工具输出净化器
    
    清理工具输出，防止：
    - 敏感信息泄露
    - 输出注入攻击
    - 信息过载
    """
    
    # 敏感数据模式
    SENSITIVE_PATTERNS = {
        'api_key': [
            r'\b[a-zA-Z_]+_API_KEY\s*[=:]\s*[\'"]?([\w-]{16,})[\'"]?',
            r'\bapi[_-]?key\s*[=:]\s*[\'"]?([\w-]{16,})[\'"]?',
            r'\b[A-Za-z0-9]{32,64}\b',  # Generic API key pattern
        ],
        'password': [
            r'\bpassword\s*[=:]\s*[\'"]?([^\s\'"]{4,})[\'"]?',
            r'\bpwd\s*[=:]\s*[\'"]?([^\s\'"]{4,})[\'"]?',
            r'\bpasswd\s*[=:]\s*[\'"]?([^\s\'"]{4,})[\'"]?',
        ],
        'token': [
            r'\btoken\s*[=:]\s*[\'"]?([\w-]{20,})[\'"]?',
            r'\baccess[_-]?token\s*[=:]\s*[\'"]?([\w-]{20,})[\'"]?',
            r'\brefresh[_-]?token\s*[=:]\s*[\'"]?([\w-]{20,})[\'"]?',
            r'\bbearer\s+([\w-]{20,})',
        ],
        'secret': [
            r'\bsecret[_-]?key\s*[=:]\s*[\'"]?([\w-]{16,})[\'"]?',
            r'\bclient[_-]?secret\s*[=:]\s*[\'"]?([\w-]{16,})[\'"]?',
            r'\baws[_-]?secret[_-]?access[_-]?key\s*[=:]\s*[\'"]?([\w/+=]{40})[\'"]?',
        ],
        'credential': [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}:[^\s]+',  # email:password
            r'\b\d{16,19}\b',  # Credit card number
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        ],
        'private_key': [
            r'-----BEGIN (RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----',
            r'-----BEGIN (RSA |DSA |EC |OPENSSH )?ENCRYPTED PRIVATE KEY-----',
        ],
    }
    
    # 危险输出模式
    DANGEROUS_OUTPUT_PATTERNS = [
        r'<script[^>]*>.*?</script>',
        r'javascript:\s*',
        r'on\w+\s*=\s*["\']',
        r'\{\{.*?\}\}',  # Template injection
        r'\$\{.*?\}',      # Template injection
        r'<%.*?%>',        # Server-side template
    ]
    
    def __init__(self, config: Optional[ToolSecurityConfig] = None):
        self.config = config or ToolSecurityConfig()
        self._compile_patterns()
        self.sanitization_stats = defaultdict(int)
    
    def _compile_patterns(self) -> None:
        """编译正则表达式"""
        self.compiled_sensitive = {}
        for data_type, patterns in self.SENSITIVE_PATTERNS.items():
            self.compiled_sensitive[data_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
        
        self.compiled_dangerous = [
            re.compile(p, re.IGNORECASE | re.DOTALL) 
            for p in self.DANGEROUS_OUTPUT_PATTERNS
        ]
    
    def redact_sensitive_data(self, output: str) -> Tuple[str, List[Dict]]:
        """
        脱敏敏感数据
        
        Returns:
            (脱敏后的输出, 检测到的敏感信息)
        """
        if not self.config.redact_sensitive_data:
            return output, []
        
        redacted = output
        detected = []
        
        for data_type, patterns in self.compiled_sensitive.items():
            for pattern in patterns:
                for match in pattern.finditer(redacted):
                    detected.append({
                        'type': data_type,
                        'position': (match.start(), match.end()),
                        'placeholder': f'[{data_type.upper()}_REDACTED]'
                    })
                    redacted = redacted[:match.start()] + f'[{data_type.upper()}_REDACTED]' + redacted[match.end():]
        
        self.sanitization_stats['sensitive_redacted'] += len(detected)
        
        return redacted, detected
    
    def detect_dangerous_content(self, output: str) -> Tuple[float, List[Dict]]:
        """检测危险内容"""
        score = 0.0
        matches = []
        
        for pattern in self.compiled_dangerous:
            if pattern.search(output):
                score += 0.3
                matches.append({
                    'type': 'dangerous_content',
                    'pattern': pattern.pattern[:50],
                    'severity': 'high'
                })
        
        # 检测超大输出
        if len(output) > self.config.max_output_size:
            score += 0.2
            matches.append({
                'type': 'output_too_large',
                'size': len(output),
                'max': self.config.max_output_size,
                'severity': 'medium'
            })
        
        return min(score, 1.0), matches
    
    def truncate_output(self, output: str, max_length: Optional[int] = None) -> str:
        """截断输出"""
        max_len = max_length or self.config.max_output_size
        if len(output) > max_len:
            return output[:max_len] + f"\n... [Output truncated, {len(output) - max_len} characters omitted]"
        return output
    
    def sanitize(self, output: Any) -> Tuple[Any, Dict]:
        """
        执行完整的输出净化
        
        Returns:
            (净化后的输出, 净化报告)
        """
        report = {
            'original_type': type(output).__name__,
            'modifications': [],
            'warnings': []
        }
        
        # 处理字符串输出
        if isinstance(output, str):
            # 1. 脱敏
            if self.config.detect_sensitive_data:
                output, detected = self.redact_sensitive_data(output)
                if detected:
                    report['modifications'].append({
                        'type': 'redaction',
                        'count': len(detected),
                        'types': list(set(d['type'] for d in detected))
                    })
            
            # 2. 检测危险内容
            risk_score, matches = self.detect_dangerous_content(output)
            if matches:
                report['warnings'].extend(matches)
            
            # 3. 截断
            original_len = len(output)
            output = self.truncate_output(output)
            if len(output) < original_len:
                report['modifications'].append({
                    'type': 'truncation',
                    'original_length': original_len,
                    'final_length': len(output)
                })
        
        # 处理字典输出
        elif isinstance(output, dict):
            sanitized_dict = {}
            for key, value in output.items():
                if isinstance(value, str):
                    sanitized_value, _ = self.redact_sensitive_data(value)
                    sanitized_dict[key] = sanitized_value
                else:
                    sanitized_dict[key] = value
            output = sanitized_dict
        
        # 处理列表输出
        elif isinstance(output, list):
            sanitized_list = []
            for item in output:
                if isinstance(item, str):
                    sanitized_item, _ = self.redact_sensitive_data(item)
                    sanitized_list.append(sanitized_item)
                else:
                    sanitized_list.append(item)
            output = sanitized_list
        
        return output, report


class ToolAbuseDetector:
    """
    工具滥用检测器
    
    检测和防止工具滥用：
    - 频率滥用
    - 模式滥用
    - 资源耗尽攻击
    - 链式攻击
    """
    
    def __init__(self, config: Optional[ToolSecurityConfig] = None):
        self.config = config or ToolSecurityConfig()
        
        # 调用历史: user_id -> deque of (timestamp, tool_name)
        self._call_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        
        # 工具链历史: user_id -> list of tool_names
        self._chain_history: Dict[str, List[str]] = defaultdict(list)
        
        # 滥用计数器
        self._abuse_counters: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        
        # 已阻止用户
        self._blocked_users: Set[str] = set()
        
        self._lock = threading.RLock()
    
    def record_call(self, user_id: str, tool_name: str) -> None:
        """记录工具调用"""
        with self._lock:
            self._call_history[user_id].append((time.time(), tool_name))
            self._chain_history[user_id].append(tool_name)
            
            # 限制链历史长度
            if len(self._chain_history[user_id]) > 100:
                self._chain_history[user_id] = self._chain_history[user_id][-50:]
    
    def check_rate_limit(self, user_id: str) -> Tuple[bool, Dict]:
        """
        检查速率限制
        
        Returns:
            (是否允许, 详细信息)
        """
        with self._lock:
            if user_id in self._blocked_users:
                return False, {'reason': 'user_blocked', 'action': 'deny'}
            
            now = time.time()
            window = self.config.abuse_window
            max_calls = self.config.max_calls_per_minute
            
            # 清理旧记录
            history = self._call_history[user_id]
            while history and now - history[0][0] > window:
                history.popleft()
            
            call_count = len(history)
            
            if call_count >= max_calls:
                self._abuse_counters[user_id]['rate_limit_violations'] += 1
                return False, {
                    'reason': 'rate_limit_exceeded',
                    'calls_in_window': call_count,
                    'max_allowed': max_calls,
                    'window_seconds': window
                }
            
            return True, {
                'calls_in_window': call_count,
                'remaining': max_calls - call_count
            }
    
    def detect_repetitive_calls(self, user_id: str) -> Tuple[float, Optional[Dict]]:
        """检测重复调用模式"""
        with self._lock:
            history = list(self._call_history[user_id])
            if len(history) < 5:
                return 0.0, None
            
            # 检查最近10次调用
            recent = history[-10:]
            tool_names = [call[1] for call in recent]
            
            # 检测相同工具的重复调用
            from collections import Counter
            tool_counts = Counter(tool_names)
            
            max_repetition = max(tool_counts.values()) if tool_counts else 0
            if max_repetition >= 5:
                score = min(0.3 + 0.1 * (max_repetition - 5), 0.8)
                most_common = tool_counts.most_common(1)[0]
                return score, {
                    'type': 'repetitive_calls',
                    'tool': most_common[0],
                    'count': most_common[1],
                    'severity': 'medium'
                }
            
            return 0.0, None
    
    def detect_chain_abuse(self, user_id: str) -> Tuple[float, Optional[Dict]]:
        """检测工具链滥用"""
        with self._lock:
            chain = self._chain_history[user_id]
            if len(chain) < 10:
                return 0.0, None
            
            # 检测危险工具组合
            dangerous_combinations = [
                (['read_file', 'write_file', 'execute'], 0.6),
                (['query_database', 'write_file'], 0.4),
                (['fetch_url', 'execute'], 0.7),
                (['list_directory', 'read_file', 'send_email'], 0.5),
            ]
            
            recent_chain = chain[-20:]
            recent_set = set(recent_chain)
            
            for combo, base_score in dangerous_combinations:
                if all(tool in recent_set for tool in combo):
                    return base_score, {
                        'type': 'dangerous_tool_chain',
                        'combination': combo,
                        'severity': 'high'
                    }
            
            # 检测长链
            if len(chain) > 50:
                return 0.3, {
                    'type': 'long_tool_chain',
                    'chain_length': len(chain),
                    'severity': 'low'
                }
            
            return 0.0, None
    
    def detect_resource_exhaustion(self, user_id: str) -> Tuple[float, Optional[Dict]]:
        """检测资源耗尽攻击"""
        with self._lock:
            history = list(self._call_history[user_id])
            if len(history) < 3:
                return 0.0, None
            
            # 检查调用频率是否异常高
            now = time.time()
            recent_calls = [t for t, _ in history if now - t < 10]  # 最近10秒
            
            if len(recent_calls) >= 10:
                return 0.5, {
                    'type': 'resource_exhaustion_attempt',
                    'calls_in_10s': len(recent_calls),
                    'severity': 'high'
                }
            
            return 0.0, None
    
    def analyze(self, user_id: str, tool_name: str) -> Tuple[bool, float, List[Dict]]:
        """
        综合分析滥用风险
        
        Returns:
            (是否允许, 风险分数, 检测结果)
        """
        # 记录调用
        self.record_call(user_id, tool_name)
        
        # 检查速率限制
        allowed, rate_info = self.check_rate_limit(user_id)
        if not allowed:
            return False, 1.0, [rate_info]
        
        detections = []
        total_score = 0.0
        
        # 各项检测
        score, detection = self.detect_repetitive_calls(user_id)
        if detection:
            total_score += score
            detections.append(detection)
        
        score, detection = self.detect_chain_abuse(user_id)
        if detection:
            total_score += score
            detections.append(detection)
        
        score, detection = self.detect_resource_exhaustion(user_id)
        if detection:
            total_score += score
            detections.append(detection)
        
        # 检查总滥用计数
        total_violations = sum(self._abuse_counters[user_id].values())
        if total_violations > self.config.abuse_threshold:
            self._blocked_users.add(user_id)
            return False, 1.0, [{
                'type': 'user_blocked',
                'reason': 'excessive_violations',
                'total_violations': total_violations
            }]
        
        final_score = min(total_score, 1.0)
        allowed = final_score < 0.7
        
        return allowed, final_score, detections
    
    def unblock_user(self, user_id: str) -> bool:
        """解除用户阻止"""
        with self._lock:
            if user_id in self._blocked_users:
                self._blocked_users.remove(user_id)
                self._abuse_counters[user_id].clear()
                return True
            return False
    
    def get_user_stats(self, user_id: str) -> Dict:
        """获取用户统计信息"""
        with self._lock:
            return {
                'total_calls': len(self._call_history[user_id]),
                'abuse_violations': dict(self._abuse_counters[user_id]),
                'is_blocked': user_id in self._blocked_users,
                'recent_chain': self._chain_history[user_id][-10:]
            }


class ToolSecurityEngine:
    """
    工具安全引擎
    
    整合所有工具安全机制。
    """
    
    def __init__(self, config: Optional[ToolSecurityConfig] = None):
        self.config = config or ToolSecurityConfig()
        
        # 初始化各模块
        self.registry = ToolRegistry()
        self.permission_manager = PermissionManager()
        self.validator = ToolCallValidator(self.registry, self.config)
        self.output_sanitizer = ToolOutputSanitizer(self.config)
        self.abuse_detector = ToolAbuseDetector(self.config)
        
        # 统计
        self.total_calls = 0
        self.blocked_calls = 0
    
    def register_tool(self, schema: ToolSchema) -> None:
        """注册工具"""
        self.registry.register(schema)
    
    def validate_and_execute(
        self,
        tool_call: ToolCall,
        user_id: str,
        execute_fn: Optional[Callable] = None
    ) -> Dict:
        """
        验证并执行工具调用
        
        Args:
            tool_call: 工具调用请求
            user_id: 用户ID
            execute_fn: 实际执行函数
            
        Returns:
            执行结果
        """
        self.total_calls += 1
        
        # 1. 滥用检测
        if self.config.enable_abuse_detection:
            allowed, abuse_score, abuse_detections = self.abuse_detector.analyze(
                user_id, tool_call.tool_name
            )
            if not allowed:
                self.blocked_calls += 1
                return {
                    'success': False,
                    'error': 'Tool call blocked due to abuse detection',
                    'abuse_detections': abuse_detections,
                    'risk_score': abuse_score
                }
        
        # 2. 权限检查
        schema = self.registry.get(tool_call.tool_name)
        if schema:
            has_permission = self.permission_manager.check_permission(
                user_id,
                tool_call.tool_name,
                min(schema.required_permissions, key=lambda x: x.value),
                schema.category
            )
            if not has_permission:
                self.blocked_calls += 1
                return {
                    'success': False,
                    'error': 'Permission denied',
                    'required_permissions': [p.name for p in schema.required_permissions]
                }
        
        # 3. 调用验证
        validation_result = self.validator.validate(tool_call)
        if not validation_result.allowed:
            self.blocked_calls += 1
            return {
                'success': False,
                'error': 'Validation failed',
                'violations': validation_result.violations,
                'risk_score': validation_result.risk_score
            }
        
        # 4. 执行
        if execute_fn:
            try:
                args = validation_result.sanitized_args or tool_call.arguments
                output = execute_fn(tool_call.tool_name, args)
                
                # 5. 输出净化
                if self.config.sanitize_outputs:
                    sanitized_output, sanitize_report = self.output_sanitizer.sanitize(output)
                else:
                    sanitized_output = output
                    sanitize_report = {}
                
                return {
                    'success': True,
                    'output': sanitized_output,
                    'sanitization_report': sanitize_report,
                    'validation_metadata': validation_result.metadata
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'error_type': type(e).__name__
                }
        
        return {
            'success': True,
            'validated': True,
            'sanitized_args': validation_result.sanitized_args
        }
    
    def get_statistics(self) -> Dict:
        """获取安全统计信息"""
        return {
            'total_calls': self.total_calls,
            'blocked_calls': self.blocked_calls,
            'block_rate': self.blocked_calls / self.total_calls if self.total_calls > 0 else 0,
            'registered_tools': len(self.registry.list_tools()),
            'sanitization_stats': dict(self.output_sanitizer.sanitization_stats)
        }
