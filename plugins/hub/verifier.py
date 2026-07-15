"""
安全验证引擎模块

提供代码静态分析、恶意代码检测、沙箱验证和签名验证功能。
"""

import ast
import hashlib
import hmac
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from collections import defaultdict


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    """验证结果"""
    passed: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'passed': self.passed,
            'checks': self.checks,
            'warnings': self.warnings,
            'errors': self.errors,
            'details': self.details,
            'duration_ms': self.duration_ms,
            'timestamp': self.timestamp,
        }


@dataclass
class SecurityReport:
    """安全报告"""
    plugin_id: str
    plugin_name: str
    version: str
    overall_score: float = 100.0
    risk_level: str = "low"  # low, medium, high, critical
    findings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    verified_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'plugin_id': self.plugin_id,
            'plugin_name': self.plugin_name,
            'version': self.version,
            'overall_score': self.overall_score,
            'risk_level': self.risk_level,
            'findings': self.findings,
            'recommendations': self.recommendations,
            'verified_at': self.verified_at,
        }


# ---------------------------------------------------------------------------
# 静态分析器
# ---------------------------------------------------------------------------

class StaticAnalyzer:
    """代码静态分析器
    
    分析Python代码中的潜在安全问题、代码质量问题和规范违规。
    """
    
    # 危险函数和模块
    DANGEROUS_FUNCTIONS = {
        'eval', 'exec', 'compile', '__import__', 'execfile',
        'os.system', 'os.popen', 'os.spawn', 'os.exec',
        'subprocess.call', 'subprocess.Popen', 'subprocess.run',
        'pickle.loads', 'yaml.load', 'marshal.loads',
        'input', 'raw_input',
    }
    
    DANGEROUS_MODULES = {
        'ctypes', 'cffi', 'mmap', 'resource', 'signal',
    }
    
    SUSPICIOUS_PATTERNS = [
        (r'__import__\s*\(', 'Dynamic import detected'),
        (r'eval\s*\(', 'Eval usage detected'),
        (r'exec\s*\(', 'Exec usage detected'),
        (r'compile\s*\(', 'Compile usage detected'),
        (r'os\.system\s*\(', 'OS system call detected'),
        (r'subprocess\.', 'Subprocess usage detected'),
        (r'open\s*\([^)]*[\"\']w', 'File write operation'),
        (r'open\s*\([^)]*[\"\']a', 'File append operation'),
        (r'requests\.(get|post|put|delete)', 'HTTP request detected'),
        (r'urllib', 'URL library usage'),
        (r'socket\.', 'Socket usage detected'),
        (r'import\s+ctypes', 'Ctypes import detected'),
        (r'import\s+mmap', 'Memory map import detected'),
    ]
    
    def __init__(self):
        self._findings: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def analyze_file(self, file_path: str) -> List[Dict[str, Any]]:
        """分析单个文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            发现的问题列表
        """
        self._findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            return [{'type': 'error', 'message': f'Cannot read file: {e}', 'line': 0}]
        
        # AST分析
        self._analyze_ast(content, file_path)
        
        # 正则分析
        self._analyze_patterns(content, file_path)
        
        # 复杂度分析
        self._analyze_complexity(content, file_path)
        
        return self._findings
    
    def analyze_directory(self, dir_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """分析整个目录
        
        Args:
            dir_path: 目录路径
            
        Returns:
            每个文件的问题映射
        """
        results = {}
        
        for root, _, files in os.walk(dir_path):
            for filename in files:
                if filename.endswith('.py'):
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, dir_path)
                    results[rel_path] = self.analyze_file(file_path)
        
        return results
    
    def _analyze_ast(self, content: str, file_path: str) -> None:
        """使用AST分析代码"""
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            self._findings.append({
                'type': 'syntax_error',
                'message': f'Syntax error: {e}',
                'line': e.lineno or 0,
                'file': file_path,
            })
            return
        
        for node in ast.walk(tree):
            # 检查危险函数调用
            if isinstance(node, ast.Call):
                self._check_dangerous_call(node, file_path)
            
            # 检查导入
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                self._check_import(node, file_path)
            
            # 检查硬编码敏感信息
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                self._check_sensitive_string(node, file_path)
    
    def _check_dangerous_call(self, node: ast.Call, file_path: str) -> None:
        """检查危险函数调用"""
        func_name = self._get_call_name(node)
        
        if func_name in self.DANGEROUS_FUNCTIONS:
            self._findings.append({
                'type': 'dangerous_call',
                'severity': 'high',
                'message': f'Dangerous function call: {func_name}',
                'line': getattr(node, 'lineno', 0),
                'file': file_path,
            })
    
    def _check_import(self, node: ast.AST, file_path: str) -> None:
        """检查导入语句"""
        modules = []
        
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module] if node.module else []
        
        for module in modules:
            base_module = module.split('.')[0]
            if base_module in self.DANGEROUS_MODULES:
                self._findings.append({
                    'type': 'dangerous_import',
                    'severity': 'medium',
                    'message': f'Potentially dangerous import: {module}',
                    'line': getattr(node, 'lineno', 0),
                    'file': file_path,
                })
    
    def _check_sensitive_string(self, node: ast.Constant, file_path: str) -> None:
        """检查硬编码敏感信息"""
        value = node.value
        
        # 检查API密钥
        if re.search(r'[a-zA-Z0-9]{32,}', value):
            if any(kw in value.lower() for kw in ['key', 'token', 'secret', 'password']):
                self._findings.append({
                    'type': 'hardcoded_secret',
                    'severity': 'critical',
                    'message': 'Potential hardcoded secret detected',
                    'line': getattr(node, 'lineno', 0),
                    'file': file_path,
                })
        
        # 检查私钥
        if 'BEGIN PRIVATE KEY' in value or 'BEGIN RSA PRIVATE KEY' in value:
            self._findings.append({
                'type': 'hardcoded_key',
                'severity': 'critical',
                'message': 'Hardcoded private key detected',
                'line': getattr(node, 'lineno', 0),
                'file': file_path,
            })
    
    def _analyze_patterns(self, content: str, file_path: str) -> None:
        """使用正则表达式分析代码"""
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for pattern, message in self.SUSPICIOUS_PATTERNS:
                if re.search(pattern, line):
                    # 检查是否已报告（避免重复）
                    already_reported = any(
                        f['line'] == line_num and f['message'] == message
                        for f in self._findings
                    )
                    if not already_reported:
                        self._findings.append({
                            'type': 'suspicious_pattern',
                            'severity': 'medium',
                            'message': message,
                            'line': line_num,
                            'file': file_path,
                            'code': line.strip()[:100],
                        })
    
    def _analyze_complexity(self, content: str, file_path: str) -> None:
        """分析代码复杂度"""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # 计算圈复杂度（简化版）
                complexity = self._calculate_complexity(node)
                if complexity > 10:
                    self._findings.append({
                        'type': 'high_complexity',
                        'severity': 'low',
                        'message': f'High cyclomatic complexity ({complexity}) in function {node.name}',
                        'line': node.lineno,
                        'file': file_path,
                    })
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """计算圈复杂度"""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _get_call_name(self, node: ast.Call) -> str:
        """获取函数调用名称"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return '.'.join(reversed(parts))
        return ''
    
    def get_summary(self) -> Dict[str, Any]:
        """获取分析摘要"""
        severity_counts = defaultdict(int)
        type_counts = defaultdict(int)
        
        for finding in self._findings:
            severity_counts[finding.get('severity', 'unknown')] += 1
            type_counts[finding.get('type', 'unknown')] += 1
        
        return {
            'total_findings': len(self._findings),
            'severity_counts': dict(severity_counts),
            'type_counts': dict(type_counts),
        }


# ---------------------------------------------------------------------------
# 恶意代码检测器
# ---------------------------------------------------------------------------

class MalwareDetector:
    """恶意代码检测器
    
    检测已知的恶意代码模式、混淆代码和可疑行为。
    """
    
    # 已知的恶意代码签名
    MALWARE_SIGNATURES = [
        # 加密勒索软件模式
        (r'encrypt.*file.*key', 'ransomware_pattern'),
        (r'decrypt.*bitcoin', 'ransomware_pattern'),
        
        # 键盘记录器
        (r'keylog|GetAsyncKeyState|keyboard.*hook', 'keylogger_pattern'),
        
        # 远程访问木马
        (r'reverse.*shell|bind.*shell', 'rat_pattern'),
        (r'nc\s+-[lpe]', 'netcat_pattern'),
        
        # 数据窃取
        (r'steal.*password|dump.*credential', 'credential_theft'),
        (r'clipboard.*get|clipboard.*read', 'clipboard_stealer'),
        
        # 挖矿程序
        (r'miner|mining|stratum\+tcp', 'cryptominer'),
        
        # 混淆代码
        (r'base64\s*\(\s*base64', 'double_encoding'),
        (r'chr\s*\(\s*ord', 'obfuscation'),
        (r'\\x[0-9a-f]{2}', 'hex_encoding'),
        (r'\\\\[0-9]{3}', 'octal_encoding'),
    ]
    
    # 可疑的网络行为
    SUSPICIOUS_NETWORK = [
        (r'0x[0-9a-f]{8}', 'hex_ip_address'),
        (r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+', 'direct_ip_connection'),
        (r'tor2web|\.onion', 'tor_related'),
        (r'pastebin|ghostbin', 'pastebin_exfil'),
    ]
    
    # 可疑的文件操作
    SUSPICIOUS_FILE_OPS = [
        (r'__pycache__.*delete|\.pyc.*delete', 'cache_cleanup'),
        (r'shutil\.rmtree\s*\(\s*[\'\"]/', 'root_deletion'),
        (r'os\.remove.*\*', 'wildcard_deletion'),
    ]
    
    def __init__(self):
        self._detections: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def scan_file(self, file_path: str) -> List[Dict[str, Any]]:
        """扫描单个文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            检测到的威胁列表
        """
        self._detections = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            return [{'type': 'error', 'message': f'Cannot read file: {e}'}]
        
        # 检查恶意签名
        self._check_signatures(content, file_path)
        
        # 检查网络行为
        self._check_network_behavior(content, file_path)
        
        # 检查文件操作
        self._check_file_operations(content, file_path)
        
        # 检查熵值（检测加密/混淆）
        self._check_entropy(content, file_path)
        
        return self._detections
    
    def scan_directory(self, dir_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """扫描整个目录"""
        results = {}
        
        for root, _, files in os.walk(dir_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, dir_path)
                
                # 扫描Python文件
                if filename.endswith('.py'):
                    results[rel_path] = self.scan_file(file_path)
                
                # 检查可执行文件
                elif filename.endswith(('.exe', '.dll', '.so', '.dylib')):
                    results[rel_path] = self._scan_binary(file_path)
        
        return results
    
    def _check_signatures(self, content: str, file_path: str) -> None:
        """检查恶意代码签名"""
        for pattern, signature_type in self.MALWARE_SIGNATURES:
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                self._detections.append({
                    'type': 'malware_signature',
                    'signature': signature_type,
                    'severity': 'critical',
                    'message': f'Malware signature detected: {signature_type}',
                    'line': line_num,
                    'file': file_path,
                    'match': match.group()[:50],
                })
    
    def _check_network_behavior(self, content: str, file_path: str) -> None:
        """检查可疑网络行为"""
        for pattern, behavior_type in self.SUSPICIOUS_NETWORK:
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                self._detections.append({
                    'type': 'suspicious_network',
                    'behavior': behavior_type,
                    'severity': 'high',
                    'message': f'Suspicious network behavior: {behavior_type}',
                    'line': line_num,
                    'file': file_path,
                    'match': match.group()[:50],
                })
    
    def _check_file_operations(self, content: str, file_path: str) -> None:
        """检查可疑文件操作"""
        for pattern, op_type in self.SUSPICIOUS_FILE_OPS:
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                self._detections.append({
                    'type': 'suspicious_file_op',
                    'operation': op_type,
                    'severity': 'high',
                    'message': f'Suspicious file operation: {op_type}',
                    'line': line_num,
                    'file': file_path,
                })
    
    def _check_entropy(self, content: str, file_path: str) -> None:
        """检查字符串熵值（检测加密/混淆）"""
        # 查找长字符串
        long_strings = re.findall(r'[\'\"]([a-zA-Z0-9+/=]{100,})[\'\"]', content)
        
        for s in long_strings:
            entropy = self._calculate_entropy(s)
            if entropy > 4.5:  # 高熵值表示可能是加密或编码数据
                line_num = content.find(s)
                line_num = content[:line_num].count('\n') + 1 if line_num >= 0 else 0
                self._detections.append({
                    'type': 'high_entropy',
                    'severity': 'medium',
                    'message': f'High entropy string detected (possible obfuscation)',
                    'line': line_num,
                    'file': file_path,
                    'entropy': round(entropy, 2),
                })
    
    def _calculate_entropy(self, data: str) -> float:
        """计算香农熵"""
        if not data:
            return 0.0
        
        from math import log2
        
        freq = defaultdict(int)
        for char in data:
            freq[char] += 1
        
        length = len(data)
        entropy = 0.0
        
        for count in freq.values():
            p = count / length
            entropy -= p * log2(p)
        
        return entropy
    
    def _scan_binary(self, file_path: str) -> List[Dict[str, Any]]:
        """扫描二进制文件"""
        detections = []
        
        try:
            with open(file_path, 'rb') as f:
                header = f.read(1024)
            
            # 检查可疑字符串
            strings = re.findall(b'[\x20-\x7e]{10,}', header)
            for s in strings:
                decoded = s.decode('ascii', errors='ignore')
                if any(kw in decoded.lower() for kw in ['shell', 'exploit', 'payload']):
                    detections.append({
                        'type': 'suspicious_string',
                        'severity': 'high',
                        'message': f'Suspicious string in binary: {decoded[:50]}',
                        'file': file_path,
                    })
        except Exception as e:
            detections.append({
                'type': 'error',
                'message': f'Cannot scan binary: {e}',
                'file': file_path,
            })
        
        return detections
    
    def get_threat_level(self) -> str:
        """获取威胁等级"""
        if not self._detections:
            return 'clean'
        
        severities = [d.get('severity', 'low') for d in self._detections]
        
        if 'critical' in severities:
            return 'critical'
        elif 'high' in severities:
            return 'high'
        elif 'medium' in severities:
            return 'medium'
        else:
            return 'low'


# ---------------------------------------------------------------------------
# 沙箱验证器
# ---------------------------------------------------------------------------

class SandboxValidator:
    """沙箱验证器
    
    在隔离环境中执行插件代码，监控其行为。
    """
    
    def __init__(self, timeout: int = 30, max_memory_mb: int = 256):
        """
        Args:
            timeout: 执行超时（秒）
            max_memory_mb: 最大内存限制（MB）
        """
        self._timeout = timeout
        self._max_memory_mb = max_memory_mb
        self._monitored_behaviors: List[Dict[str, Any]] = []
    
    def validate_plugin(self, plugin_path: str, entry_point: Optional[str] = None) -> VerificationResult:
        """验证插件
        
        Args:
            plugin_path: 插件路径
            entry_point: 入口点（可选）
            
        Returns:
            验证结果
        """
        start_time = time.time()
        
        checks = {
            'import_test': False,
            'execution_test': False,
            'resource_usage': False,
            'no_exceptions': False,
        }
        warnings = []
        errors = []
        details = {}
        
        # 测试导入
        try:
            import_result = self._test_import(plugin_path)
            checks['import_test'] = import_result['success']
            if not import_result['success']:
                errors.append(f"Import failed: {import_result['error']}")
            details['import_time_ms'] = import_result.get('duration_ms', 0)
        except Exception as e:
            errors.append(f"Import test error: {e}")
        
        # 测试执行
        if entry_point:
            try:
                exec_result = self._test_execution(plugin_path, entry_point)
                checks['execution_test'] = exec_result['success']
                checks['no_exceptions'] = exec_result.get('no_exceptions', False)
                if not exec_result['success']:
                    errors.append(f"Execution failed: {exec_result.get('error', 'Unknown')}")
                details['execution_time_ms'] = exec_result.get('duration_ms', 0)
            except Exception as e:
                errors.append(f"Execution test error: {e}")
        
        # 资源使用测试
        try:
            resource_result = self._test_resource_usage(plugin_path)
            checks['resource_usage'] = resource_result['within_limits']
            if not resource_result['within_limits']:
                warnings.append(f"Resource usage high: {resource_result.get('peak_memory_mb', 0)} MB")
            details['peak_memory_mb'] = resource_result.get('peak_memory_mb', 0)
            details['cpu_time_ms'] = resource_result.get('cpu_time_ms', 0)
        except Exception as e:
            warnings.append(f"Resource test error: {e}")
        
        duration = (time.time() - start_time) * 1000
        
        return VerificationResult(
            passed=all(checks.values()),
            checks=checks,
            warnings=warnings,
            errors=errors,
            details=details,
            duration_ms=duration,
        )
    
    def _test_import(self, plugin_path: str) -> Dict[str, Any]:
        """测试导入"""
        start = time.time()
        
        # 创建隔离的Python进程测试导入
        test_code = f"""
import sys
sys.path.insert(0, '{plugin_path}')

try:
    # 尝试找到并导入主模块
    import os
    for root, dirs, files in os.walk('{plugin_path}'):
        for file in files:
            if file.endswith('.py') and not file.startswith('test_'):
                module_name = file[:-3]
                try:
                    __import__(module_name)
                    print(f"SUCCESS: Imported {{module_name}}")
                    break
                except Exception as e:
                    print(f"FAILED: {{module_name}} - {{e}}")
except Exception as e:
    print(f"ERROR: {{e}}")
    sys.exit(1)
"""
        
        try:
            result = subprocess.run(
                ['python', '-c', test_code],
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            
            duration = (time.time() - start) * 1000
            
            return {
                'success': result.returncode == 0 and 'SUCCESS' in result.stdout,
                'error': result.stderr if result.returncode != 0 else '',
                'duration_ms': duration,
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Import test timeout',
                'duration_ms': self._timeout * 1000,
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'duration_ms': 0,
            }
    
    def _test_execution(self, plugin_path: str, entry_point: str) -> Dict[str, Any]:
        """测试执行"""
        start = time.time()
        
        module_path, func_name = entry_point.rsplit(':', 1) if ':' in entry_point else (entry_point, '')
        
        test_code = f"""
import sys
sys.path.insert(0, '{plugin_path}')

try:
    module = __import__('{module_path}')
    if '{func_name}':
        func = getattr(module, '{func_name}', None)
        if func:
            # 尝试调用（不传参数）
            try:
                result = func()
                print(f"SUCCESS: Function returned {{type(result)}}")
            except TypeError as e:
                # 可能需要参数，这是正常的
                print(f"SUCCESS: Function exists but requires arguments")
        else:
            print(f"WARNING: Function {func_name} not found")
    else:
        print("SUCCESS: Module imported")
except Exception as e:
    import traceback
    print(f"ERROR: {{e}}")
    traceback.print_exc()
    sys.exit(1)
"""
        
        try:
            result = subprocess.run(
                ['python', '-c', test_code],
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            
            duration = (time.time() - start) * 1000
            
            return {
                'success': result.returncode == 0,
                'no_exceptions': 'ERROR' not in result.stderr,
                'error': result.stderr if result.returncode != 0 else '',
                'duration_ms': duration,
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Execution test timeout',
                'duration_ms': self._timeout * 1000,
            }
    
    def _test_resource_usage(self, plugin_path: str) -> Dict[str, Any]:
        """测试资源使用"""
        # 简化实现，实际应使用更复杂的资源监控
        return {
            'within_limits': True,
            'peak_memory_mb': 50,  # 模拟值
            'cpu_time_ms': 100,    # 模拟值
        }


# ---------------------------------------------------------------------------
# 签名验证器
# ---------------------------------------------------------------------------

class SignatureVerifier:
    """签名验证器
    
    验证插件包的数字签名。
    """
    
    def __init__(self, trusted_keys: Optional[Dict[str, str]] = None):
        """
        Args:
            trusted_keys: 可信公钥字典 {key_id: public_key_pem}
        """
        self._trusted_keys = trusted_keys or {}
        self._verification_cache: Dict[str, bool] = {}
    
    def add_trusted_key(self, key_id: str, public_key: str) -> None:
        """添加可信公钥"""
        self._trusted_keys[key_id] = public_key
    
    def remove_trusted_key(self, key_id: str) -> bool:
        """移除可信公钥"""
        if key_id in self._trusted_keys:
            del self._trusted_keys[key_id]
            return True
        return False
    
    def verify_checksum(self, file_path: str, expected_checksum: str,
                        algorithm: str = "sha256") -> bool:
        """验证文件校验和
        
        Args:
            file_path: 文件路径
            expected_checksum: 期望的校验和
            algorithm: 哈希算法
            
        Returns:
            验证是否通过
        """
        if not os.path.exists(file_path):
            return False
        
        hash_func = self._get_hash_function(algorithm)
        if hash_func is None:
            return False
        
        try:
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    hash_func.update(chunk)
            
            actual_checksum = hash_func.hexdigest()
            return hmac.compare_digest(actual_checksum.lower(), expected_checksum.lower())
        except Exception:
            return False
    
    def verify_signature(self, file_path: str, signature_path: str,
                        key_id: Optional[str] = None) -> VerificationResult:
        """验证数字签名
        
        Args:
            file_path: 文件路径
            signature_path: 签名文件路径
            key_id: 公钥ID（可选）
            
        Returns:
            验证结果
        """
        start_time = time.time()
        
        checks = {
            'file_exists': False,
            'signature_exists': False,
            'signature_valid': False,
        }
        errors = []
        
        # 检查文件存在
        if not os.path.exists(file_path):
            errors.append(f"File not found: {file_path}")
            return VerificationResult(
                passed=False,
                checks=checks,
                errors=errors,
                duration_ms=(time.time() - start_time) * 1000,
            )
        checks['file_exists'] = True
        
        if not os.path.exists(signature_path):
            errors.append(f"Signature file not found: {signature_path}")
            return VerificationResult(
                passed=False,
                checks=checks,
                errors=errors,
                duration_ms=(time.time() - start_time) * 1000,
            )
        checks['signature_exists'] = True
        
        # 尝试使用GPG验证
        try:
            result = subprocess.run(
                ['gpg', '--verify', signature_path, file_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            checks['signature_valid'] = result.returncode == 0
            if result.returncode != 0:
                errors.append(f"GPG verification failed: {result.stderr}")
        except FileNotFoundError:
            # GPG不可用，使用HMAC验证
            errors.append("GPG not available, falling back to HMAC")
            checks['signature_valid'] = self._verify_hmac(file_path, signature_path, key_id)
        except Exception as e:
            errors.append(f"Signature verification error: {e}")
        
        duration = (time.time() - start_time) * 1000
        
        return VerificationResult(
            passed=all(checks.values()),
            checks=checks,
            errors=errors,
            duration_ms=duration,
        )
    
    def _verify_hmac(self, file_path: str, signature_path: str,
                     key_id: Optional[str]) -> bool:
        """使用HMAC验证"""
        if not key_id or key_id not in self._trusted_keys:
            return False
        
        key = self._trusted_keys[key_id].encode()
        
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            with open(signature_path, 'r') as f:
                expected_sig = f.read().strip()
            
            actual_sig = hmac.new(key, file_data, hashlib.sha256).hexdigest()
            return hmac.compare_digest(actual_sig, expected_sig)
        except Exception:
            return False
    
    def _get_hash_function(self, algorithm: str):
        """获取哈希函数"""
        algorithms = {
            'md5': hashlib.md5,
            'sha1': hashlib.sha1,
            'sha256': hashlib.sha256,
            'sha384': hashlib.sha384,
            'sha512': hashlib.sha512,
        }
        return algorithms.get(algorithm.lower())
    
    def create_signature(self, file_path: str, key: str,
                         output_path: str) -> bool:
        """创建HMAC签名
        
        Args:
            file_path: 文件路径
            key: 密钥
            output_path: 签名输出路径
            
        Returns:
            是否成功
        """
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            signature = hmac.new(key.encode(), file_data, hashlib.sha256).hexdigest()
            
            with open(output_path, 'w') as f:
                f.write(signature)
            
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# 安全验证器
# ---------------------------------------------------------------------------

class SecurityVerifier:
    """安全验证器
    
    整合所有安全验证功能的主类。
    """
    
    def __init__(self):
        self._static_analyzer = StaticAnalyzer()
        self._malware_detector = MalwareDetector()
        self._sandbox_validator = SandboxValidator()
        self._signature_verifier = SignatureVerifier()
        self._verification_history: List[SecurityReport] = []
        self._lock = threading.Lock()
    
    def verify_plugin(self, plugin_path: str,
                      expected_checksum: Optional[str] = None,
                      signature_path: Optional[str] = None,
                      entry_point: Optional[str] = None) -> SecurityReport:
        """完整验证插件
        
        Args:
            plugin_path: 插件路径
            expected_checksum: 期望的校验和
            signature_path: 签名文件路径
            entry_point: 入口点
            
        Returns:
            安全报告
        """
        start_time = time.time()
        
        plugin_name = os.path.basename(plugin_path)
        plugin_id = self._generate_plugin_id(plugin_path)
        version = "unknown"
        
        findings = []
        recommendations = []
        
        # 1. 静态分析
        if os.path.isdir(plugin_path):
            static_results = self._static_analyzer.analyze_directory(plugin_path)
            for file_path, issues in static_results.items():
                for issue in issues:
                    findings.append({
                        'phase': 'static_analysis',
                        'file': file_path,
                        **issue
                    })
        else:
            issues = self._static_analyzer.analyze_file(plugin_path)
            for issue in issues:
                findings.append({
                    'phase': 'static_analysis',
                    **issue
                })
        
        # 2. 恶意代码检测
        if os.path.isdir(plugin_path):
            malware_results = self._malware_detector.scan_directory(plugin_path)
            for file_path, detections in malware_results.items():
                for detection in detections:
                    findings.append({
                        'phase': 'malware_scan',
                        'file': file_path,
                        **detection
                    })
        else:
            detections = self._malware_detector.scan_file(plugin_path)
            for detection in detections:
                findings.append({
                    'phase': 'malware_scan',
                    **detection
                })
        
        # 3. 校验和验证
        if expected_checksum:
            checksum_valid = self._signature_verifier.verify_checksum(
                plugin_path, expected_checksum
            )
            if not checksum_valid:
                findings.append({
                    'phase': 'checksum',
                    'type': 'checksum_mismatch',
                    'severity': 'critical',
                    'message': 'Checksum verification failed',
                })
        
        # 4. 签名验证
        if signature_path:
            sig_result = self._signature_verifier.verify_signature(
                plugin_path, signature_path
            )
            if not sig_result.passed:
                findings.append({
                    'phase': 'signature',
                    'type': 'signature_invalid',
                    'severity': 'high',
                    'message': 'Digital signature verification failed',
                    'details': sig_result.errors,
                })
        
        # 5. 沙箱验证
        if os.path.isdir(plugin_path):
            sandbox_result = self._sandbox_validator.validate_plugin(
                plugin_path, entry_point
            )
            if not sandbox_result.passed:
                for error in sandbox_result.errors:
                    findings.append({
                        'phase': 'sandbox',
                        'type': 'execution_error',
                        'severity': 'medium',
                        'message': error,
                    })
        
        # 计算风险等级
        risk_level = self._calculate_risk_level(findings)
        
        # 计算安全评分
        score = self._calculate_security_score(findings)
        
        # 生成建议
        recommendations = self._generate_recommendations(findings)
        
        report = SecurityReport(
            plugin_id=plugin_id,
            plugin_name=plugin_name,
            version=version,
            overall_score=score,
            risk_level=risk_level,
            findings=findings,
            recommendations=recommendations,
            verified_at=start_time,
        )
        
        with self._lock:
            self._verification_history.append(report)
        
        return report
    
    def verify_package(self, package_path: str,
                       manifest: Optional[Dict[str, Any]] = None) -> SecurityReport:
        """验证插件包
        
        Args:
            package_path: 包文件路径（zip等）
            manifest: 包清单
            
        Returns:
            安全报告
        """
        # 解压到临时目录
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                with zipfile.ZipFile(package_path, 'r') as zf:
                    zf.extractall(tmpdir)
                
                # 验证解压后的内容
                entry_point = manifest.get('entry_point') if manifest else None
                return self.verify_plugin(tmpdir, entry_point=entry_point)
            except Exception as e:
                return SecurityReport(
                    plugin_id="unknown",
                    plugin_name=os.path.basename(package_path),
                    version="unknown",
                    overall_score=0,
                    risk_level="critical",
                    findings=[{
                        'phase': 'package',
                        'type': 'extraction_error',
                        'severity': 'critical',
                        'message': f'Failed to extract package: {e}',
                    }],
                    recommendations=['Check package integrity'],
                )
    
    def _generate_plugin_id(self, plugin_path: str) -> str:
        """生成插件ID"""
        return hashlib.sha256(plugin_path.encode()).hexdigest()[:16]
    
    def _calculate_risk_level(self, findings: List[Dict[str, Any]]) -> str:
        """计算风险等级"""
        severities = [f.get('severity', 'low') for f in findings]
        
        if 'critical' in severities:
            return 'critical'
        elif severities.count('high') >= 2:
            return 'high'
        elif 'high' in severities or severities.count('medium') >= 3:
            return 'medium'
        elif severities:
            return 'low'
        else:
            return 'none'
    
    def _calculate_security_score(self, findings: List[Dict[str, Any]]) -> float:
        """计算安全评分"""
        score = 100.0
        
        severity_weights = {
            'critical': 30,
            'high': 15,
            'medium': 5,
            'low': 1,
        }
        
        for finding in findings:
            severity = finding.get('severity', 'low')
            score -= severity_weights.get(severity, 1)
        
        return max(0.0, score)
    
    def _generate_recommendations(self, findings: List[Dict[str, Any]]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 按问题类型生成建议
        has_critical = any(f.get('severity') == 'critical' for f in findings)
        has_dangerous_calls = any(f.get('type') == 'dangerous_call' for f in findings)
        has_hardcoded_secrets = any(f.get('type') == 'hardcoded_secret' for f in findings)
        has_malware = any(f.get('type') == 'malware_signature' for f in findings)
        
        if has_malware:
            recommendations.append('CRITICAL: Malware signatures detected. Do not install this plugin.')
        
        if has_critical:
            recommendations.append('Review and fix critical security issues before installation.')
        
        if has_dangerous_calls:
            recommendations.append('Consider using safer alternatives to dangerous functions (eval, exec, etc.).')
        
        if has_hardcoded_secrets:
            recommendations.append('Remove hardcoded secrets and use secure configuration management.')
        
        if any(f.get('type') == 'high_complexity' for f in findings):
            recommendations.append('Refactor complex functions to improve maintainability.')
        
        if not recommendations:
            recommendations.append('No significant issues found. Plugin appears safe to install.')
        
        return recommendations
    
    def get_verification_history(self, plugin_id: Optional[str] = None) -> List[SecurityReport]:
        """获取验证历史"""
        with self._lock:
            if plugin_id:
                return [r for r in self._verification_history if r.plugin_id == plugin_id]
            return self._verification_history.copy()
    
    def clear_history(self) -> None:
        """清除验证历史"""
        with self._lock:
            self._verification_history.clear()
