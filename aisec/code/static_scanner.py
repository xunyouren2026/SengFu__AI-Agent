"""
静态代码扫描 - Bandit/Semgrep风格的代码安全分析
"""
import ast
import re
import os
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum


class Severity(Enum):
    """严重程度"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IssueType(Enum):
    """问题类型"""
    SQL_INJECTION = "sql_injection"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    XSS = "xss"
    SSRF = "ssrf"
    HARDCODED_SECRET = "hardcoded_secret"
    WEAK_CRYPTO = "weak_crypto"
    INSECURE_DESERIALIZE = "insecure_deserialize"
    CODE_INJECTION = "code_injection"
    DANGEROUS_FUNCTION = "dangerous_function"
    ASSERT_USAGE = "assert_usage"
    DEBUG_INFO = "debug_info"
    EXCEPTION_LEAK = "exception_leak"
    WEAK_RANDOM = "weak_random"
    TEMP_FILE = "temp_file"


@dataclass
class CodeIssue:
    """代码问题"""
    issue_type: IssueType
    severity: Severity
    message: str
    file_path: str
    line_number: int
    column: int = 0
    code_snippet: str = ""
    confidence: float = 1.0
    cwe_id: Optional[str] = None
    owasp_category: Optional[str] = None
    references: List[str] = field(default_factory=list)


@dataclass
class ScanResult:
    """扫描结果"""
    file_path: str
    issues: List[CodeIssue]
    lines_scanned: int
    scan_time: float
    language: str = "python"


class ASTVisitor(ast.NodeVisitor):
    """AST访问器，用于检测安全问题"""
    
    def __init__(self, file_path: str, source_code: str):
        self.file_path = file_path
        self.source_code = source_code
        self.issues: List[CodeIssue] = []
        self._source_lines = source_code.split('\n')
        
        # 危险函数列表
        self._dangerous_functions = {
            # 命令执行
            'os.system': (IssueType.COMMAND_INJECTION, Severity.HIGH, "使用os.system可能导致命令注入"),
            'os.popen': (IssueType.COMMAND_INJECTION, Severity.HIGH, "使用os.popen可能导致命令注入"),
            'subprocess.call': (IssueType.COMMAND_INJECTION, Severity.MEDIUM, "subprocess.call需要检查shell参数"),
            'subprocess.run': (IssueType.COMMAND_INJECTION, Severity.MEDIUM, "subprocess.run需要检查shell参数"),
            'subprocess.Popen': (IssueType.COMMAND_INJECTION, Severity.MEDIUM, "subprocess.Popen需要检查shell参数"),
            'eval': (IssueType.CODE_INJECTION, Severity.CRITICAL, "使用eval可能导致代码注入"),
            'exec': (IssueType.CODE_INJECTION, Severity.CRITICAL, "使用exec可能导致代码注入"),
            'compile': (IssueType.CODE_INJECTION, Severity.HIGH, "使用compile需要确保输入安全"),
            
            # 序列化
            'pickle.loads': (IssueType.INSECURE_DESERIALIZE, Severity.CRITICAL, "pickle反序列化不安全数据"),
            'pickle.load': (IssueType.INSECURE_DESERIALIZE, Severity.CRITICAL, "pickle反序列化不安全数据"),
            'yaml.load': (IssueType.INSECURE_DESERIALIZE, Severity.HIGH, "yaml.load不安全，应使用yaml.safe_load"),
            
            # 弱加密
            'hashlib.md5': (IssueType.WEAK_CRYPTO, Severity.MEDIUM, "MD5哈希不安全"),
            'hashlib.sha1': (IssueType.WEAK_CRYPTO, Severity.MEDIUM, "SHA1哈希不安全"),
            'random.random': (IssueType.WEAK_RANDOM, Severity.LOW, "random模块不适用于安全场景"),
            'random.randint': (IssueType.WEAK_RANDOM, Severity.LOW, "random模块不适用于安全场景"),
            
            # 断言
            'assert': (IssueType.ASSERT_USAGE, Severity.LOW, "assert语句在生产环境可能被禁用"),
        }
        
        # SQL危险函数
        self._sql_functions = {'execute', 'executemany', 'executescript', 'raw'}
        
        # 文件操作危险模式
        self._file_functions = {'open', 'file'}
    
    def _get_line(self, lineno: int) -> str:
        """获取指定行代码"""
        if 0 < lineno <= len(self._source_lines):
            return self._source_lines[lineno - 1]
        return ""
    
    def visit_Call(self, node: ast.Call) -> None:
        """访问函数调用"""
        func_name = self._get_func_name(node)
        
        # 检查危险函数
        for dangerous_func, (issue_type, severity, message) in self._dangerous_functions.items():
            if func_name == dangerous_func or func_name.endswith('.' + dangerous_func):
                self.issues.append(CodeIssue(
                    issue_type=issue_type,
                    severity=severity,
                    message=message,
                    file_path=self.file_path,
                    line_number=node.lineno,
                    code_snippet=self._get_line(node.lineno),
                    confidence=0.8
                ))
        
        # 检查SQL注入
        if func_name in self._sql_functions or func_name.endswith('.' + func_name) and func_name.split('.')[-1] in self._sql_functions:
            self._check_sql_injection(node)
        
        # 检查文件路径
        if func_name in self._file_functions or func_name.endswith('.open'):
            self._check_path_traversal(node)
        
        self.generic_visit(node)
    
    def visit_Assert(self, node: ast.Assert) -> None:
        """访问assert语句"""
        self.issues.append(CodeIssue(
            issue_type=IssueType.ASSERT_USAGE,
            severity=Severity.LOW,
            message="assert语句在生产环境可能被禁用",
            file_path=self.file_path,
            line_number=node.lineno,
            code_snippet=self._get_line(node.lineno),
            confidence=0.7
        ))
        self.generic_visit(node)
    
    def visit_Assign(self, node: ast.Assign) -> None:
        """访问赋值语句，检查硬编码密钥"""
        # 检查是否为密钥变量
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id.lower()
                if any(keyword in var_name for keyword in ['password', 'secret', 'key', 'token', 'api_key']):
                    # 检查是否为硬编码字符串
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        if len(node.value.value) > 4:  # 忽略短字符串
                            self.issues.append(CodeIssue(
                                issue_type=IssueType.HARDCODED_SECRET,
                                severity=Severity.HIGH,
                                message=f"检测到硬编码的敏感信息: {var_name}",
                                file_path=self.file_path,
                                line_number=node.lineno,
                                code_snippet=self._get_line(node.lineno),
                                confidence=0.9,
                                cwe_id="CWE-798"
                            ))
        
        self.generic_visit(node)
    
    def visit_Try(self, node: ast.Try) -> None:
        """访问try语句，检查异常泄露"""
        for handler in node.handlers:
            if handler.name and isinstance(handler.type, ast.Name):
                # 检查是否直接打印异常
                for child in ast.walk(handler):
                    if isinstance(child, ast.Call):
                        func_name = self._get_func_name(child)
                        if func_name in ('print', 'logging.info', 'logging.debug'):
                            self.issues.append(CodeIssue(
                                issue_type=IssueType.EXCEPTION_LEAK,
                                severity=Severity.MEDIUM,
                                message="异常信息可能泄露敏感数据",
                                file_path=self.file_path,
                                line_number=handler.lineno,
                                confidence=0.6
                            ))
                            break
        
        self.generic_visit(node)
    
    def _get_func_name(self, node: ast.Call) -> str:
        """获取函数名"""
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
        return ""
    
    def _check_sql_injection(self, node: ast.Call) -> None:
        """检查SQL注入"""
        if node.args:
            arg = node.args[0]
            # 检查是否使用字符串拼接
            if isinstance(arg, ast.BinOp) and isinstance(arg.op, (ast.Add, ast.Mod)):
                self.issues.append(CodeIssue(
                    issue_type=IssueType.SQL_INJECTION,
                    severity=Severity.HIGH,
                    message="SQL查询使用字符串拼接，可能导致注入",
                    file_path=self.file_path,
                    line_number=node.lineno,
                    code_snippet=self._get_line(node.lineno),
                    confidence=0.8,
                    cwe_id="CWE-89"
                ))
            # 检查是否使用f-string
            elif isinstance(arg, ast.JoinedStr):
                self.issues.append(CodeIssue(
                    issue_type=IssueType.SQL_INJECTION,
                    severity=Severity.HIGH,
                    message="SQL查询使用f-string，可能导致注入",
                    file_path=self.file_path,
                    line_number=node.lineno,
                    code_snippet=self._get_line(node.lineno),
                    confidence=0.9,
                    cwe_id="CWE-89"
                ))
    
    def _check_path_traversal(self, node: ast.Call) -> None:
        """检查路径遍历"""
        if node.args:
            arg = node.args[0]
            # 检查是否使用用户输入拼接路径
            if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Add):
                self.issues.append(CodeIssue(
                    issue_type=IssueType.PATH_TRAVERSAL,
                    severity=Severity.HIGH,
                    message="文件路径拼接可能导致路径遍历",
                    file_path=self.file_path,
                    line_number=node.lineno,
                    code_snippet=self._get_line(node.lineno),
                    confidence=0.7,
                    cwe_id="CWE-22"
                ))


class StaticScanner:
    """静态代码扫描器"""
    
    def __init__(self):
        self._excluded_dirs = {'node_modules', '.git', '__pycache__', 'venv', '.venv', 'build', 'dist'}
        self._excluded_files = {'.pyc', '.pyo', '.pyd'}
        self._custom_rules: List[Dict[str, Any]] = []
    
    def scan_file(self, file_path: str) -> ScanResult:
        """扫描单个文件"""
        import time
        start_time = time.time()
        
        issues = []
        lines_scanned = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source_code = f.read()
            
            lines_scanned = len(source_code.split('\n'))
            
            # 解析AST
            try:
                tree = ast.parse(source_code)
                visitor = ASTVisitor(file_path, source_code)
                visitor.visit(tree)
                issues.extend(visitor.issues)
            except SyntaxError:
                # AST解析失败，使用正则扫描
                issues.extend(self._regex_scan(file_path, source_code))
            
            # 应用自定义规则
            issues.extend(self._apply_custom_rules(file_path, source_code))
            
        except Exception as e:
            issues.append(CodeIssue(
                issue_type=IssueType.DEBUG_INFO,
                severity=Severity.INFO,
                message=f"扫描文件时出错: {str(e)}",
                file_path=file_path,
                line_number=0
            ))
        
        scan_time = time.time() - start_time
        
        return ScanResult(
            file_path=file_path,
            issues=issues,
            lines_scanned=lines_scanned,
            scan_time=scan_time
        )
    
    def scan_directory(self, directory: str, recursive: bool = True) -> List[ScanResult]:
        """扫描目录"""
        results = []
        
        if recursive:
            for root, dirs, files in os.walk(directory):
                # 排除目录
                dirs[:] = [d for d in dirs if d not in self._excluded_dirs]
                
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        results.append(self.scan_file(file_path))
        else:
            for file in os.listdir(directory):
                if file.endswith('.py'):
                    file_path = os.path.join(directory, file)
                    results.append(self.scan_file(file_path))
        
        return results
    
    def scan_code(self, code: str, language: str = "python") -> ScanResult:
        """扫描代码字符串"""
        import time
        start_time = time.time()
        
        issues = []
        lines_scanned = len(code.split('\n'))
        
        if language == "python":
            try:
                tree = ast.parse(code)
                visitor = ASTVisitor("<string>", code)
                visitor.visit(tree)
                issues.extend(visitor.issues)
            except SyntaxError:
                issues.extend(self._regex_scan("<string>", code))
        
        scan_time = time.time() - start_time
        
        return ScanResult(
            file_path="<string>",
            issues=issues,
            lines_scanned=lines_scanned,
            scan_time=scan_time,
            language=language
        )
    
    def _regex_scan(self, file_path: str, source_code: str) -> List[CodeIssue]:
        """正则扫描（当AST解析失败时）"""
        issues = []
        lines = source_code.split('\n')
        
        # 危险模式
        patterns = [
            (r'\beval\s*\(', IssueType.CODE_INJECTION, Severity.CRITICAL, "使用eval函数"),
            (r'\bexec\s*\(', IssueType.CODE_INJECTION, Severity.CRITICAL, "使用exec函数"),
            (r'\bos\.system\s*\(', IssueType.COMMAND_INJECTION, Severity.HIGH, "使用os.system"),
            (r'\bpickle\.loads?\s*\(', IssueType.INSECURE_DESERIALIZE, Severity.CRITICAL, "使用pickle"),
            (r'password\s*=\s*["\'][^"\']+["\']', IssueType.HARDCODED_SECRET, Severity.HIGH, "硬编码密码"),
            (r'api_key\s*=\s*["\'][^"\']+["\']', IssueType.HARDCODED_SECRET, Severity.HIGH, "硬编码API密钥"),
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, issue_type, severity, message in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(CodeIssue(
                        issue_type=issue_type,
                        severity=severity,
                        message=message,
                        file_path=file_path,
                        line_number=i,
                        code_snippet=line.strip(),
                        confidence=0.6
                    ))
        
        return issues
    
    def _apply_custom_rules(self, file_path: str, source_code: str) -> List[CodeIssue]:
        """应用自定义规则"""
        issues = []
        lines = source_code.split('\n')
        
        for rule in self._custom_rules:
            pattern = rule.get('pattern')
            if not pattern:
                continue
            
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    issues.append(CodeIssue(
                        issue_type=IssueType(rule.get('type', 'dangerous_function')),
                        severity=Severity(rule.get('severity', 'medium')),
                        message=rule.get('message', '匹配自定义规则'),
                        file_path=file_path,
                        line_number=i,
                        code_snippet=line.strip(),
                        confidence=rule.get('confidence', 0.7)
                    ))
        
        return issues
    
    def add_custom_rule(self, rule: Dict[str, Any]) -> None:
        """添加自定义规则"""
        self._custom_rules.append(rule)
    
    def add_excluded_dir(self, dir_name: str) -> None:
        """添加排除目录"""
        self._excluded_dirs.add(dir_name)
    
    def get_summary(self, results: List[ScanResult]) -> Dict[str, Any]:
        """获取扫描摘要"""
        all_issues = []
        for result in results:
            all_issues.extend(result.issues)
        
        severity_counts = {}
        type_counts = {}
        
        for issue in all_issues:
            severity_counts[issue.severity.value] = severity_counts.get(issue.severity.value, 0) + 1
            type_counts[issue.issue_type.value] = type_counts.get(issue.issue_type.value, 0) + 1
        
        return {
            "files_scanned": len(results),
            "total_issues": len(all_issues),
            "by_severity": severity_counts,
            "by_type": type_counts,
            "total_lines": sum(r.lines_scanned for r in results)
        }
