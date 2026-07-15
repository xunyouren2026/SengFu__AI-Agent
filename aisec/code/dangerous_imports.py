"""
危险导入检测 - 检测代码中的危险模块导入
"""
import ast
import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    """风险等级"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class DangerousImport:
    """危险导入信息"""
    module: str
    alias: Optional[str]
    line_number: int
    risk_level: RiskLevel
    reason: str
    suggestion: str
    code_snippet: str = ""


class DangerousImportDetector:
    """危险导入检测器"""
    
    def __init__(self):
        # 危险模块配置
        self._dangerous_modules: Dict[str, tuple] = {
            # 代码执行相关
            'os': (RiskLevel.MEDIUM, "操作系统接口，可能被滥用", "使用受限的os功能或封装"),
            'subprocess': (RiskLevel.HIGH, "子进程执行，可能导致命令注入", "使用shell=False并验证输入"),
            'commands': (RiskLevel.HIGH, "已废弃的命令执行模块", "使用subprocess替代"),
            'popen2': (RiskLevel.HIGH, "已废弃的进程模块", "使用subprocess替代"),
            
            # 序列化相关
            'pickle': (RiskLevel.CRITICAL, "不安全的序列化，可能导致RCE", "使用json或限制unpickle"),
            'cPickle': (RiskLevel.CRITICAL, "不安全的序列化，可能导致RCE", "使用json或限制unpickle"),
            'marshal': (RiskLevel.HIGH, "不安全的序列化", "使用json"),
            'shelve': (RiskLevel.HIGH, "基于pickle的持久化", "使用安全的存储方案"),
            
            # 网络相关
            'socket': (RiskLevel.MEDIUM, "底层网络接口", "确保正确绑定和验证"),
            'telnetlib': (RiskLevel.HIGH, "不安全的Telnet协议", "使用SSH替代"),
            'ftplib': (RiskLevel.MEDIUM, "FTP协议可能泄露凭据", "使用SFTP替代"),
            'poplib': (RiskLevel.MEDIUM, "POP3协议不安全", "使用IMAPS替代"),
            'imaplib': (RiskLevel.MEDIUM, "IMAP需要使用SSL", "使用IMAP_SSL"),
            
            # 加密相关
            'hashlib': (RiskLevel.LOW, "检查是否使用弱哈希", "避免MD5/SHA1用于安全场景"),
            'hmac': (RiskLevel.LOW, "检查密钥管理", "确保密钥安全存储"),
            'Crypto': (RiskLevel.MEDIUM, "PyCrypto已停止维护", "使用pycryptodome"),
            
            # 系统相关
            'sys': (RiskLevel.LOW, "系统接口", "注意不要泄露敏感信息"),
            'ctypes': (RiskLevel.HIGH, "C类型接口，可能导致内存问题", "谨慎使用"),
            '_thread': (RiskLevel.MEDIUM, "底层线程接口", "使用threading模块"),
            
            # 调试相关
            'pdb': (RiskLevel.LOW, "调试器", "生产环境应移除"),
            'code': (RiskLevel.MEDIUM, "交互式解释器", "生产环境应禁用"),
            'codeop': (RiskLevel.MEDIUM, "代码编译", "可能被滥用"),
            
            # 动态执行
            'exec': (RiskLevel.CRITICAL, "内置exec函数", "避免动态代码执行"),
            'eval': (RiskLevel.CRITICAL, "内置eval函数", "避免动态代码执行"),
            'compile': (RiskLevel.HIGH, "代码编译函数", "验证输入来源"),
            
            # 其他
            'tempfile': (RiskLevel.LOW, "临时文件", "注意权限设置"),
            'xml.etree': (RiskLevel.MEDIUM, "XML解析可能受XXE攻击", "禁用外部实体"),
            'xml.dom': (RiskLevel.MEDIUM, "XML解析可能受XXE攻击", "禁用外部实体"),
            'lxml': (RiskLevel.MEDIUM, "XML解析可能受XXE攻击", "配置安全解析"),
        }
        
        # 危险函数导入
        self._dangerous_functions: Dict[str, tuple] = {
            'eval': (RiskLevel.CRITICAL, "动态代码执行", "避免使用eval"),
            'exec': (RiskLevel.CRITICAL, "动态代码执行", "避免使用exec"),
            'compile': (RiskLevel.HIGH, "代码编译", "验证输入"),
            'open': (RiskLevel.LOW, "文件操作", "注意路径验证"),
            'input': (RiskLevel.LOW, "用户输入", "Python 2中可能执行代码"),
        }
        
        # 白名单
        self._whitelist: Set[str] = set()
        self._custom_dangerous: Dict[str, tuple] = {}
    
    def detect(self, source_code: str) -> List[DangerousImport]:
        """检测源代码中的危险导入"""
        results = []
        lines = source_code.split('\n')
        
        try:
            tree = ast.parse(source_code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        result = self._check_module(
                            alias.name,
                            alias.asname,
                            node.lineno,
                            lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                        )
                        if result:
                            results.append(result)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        result = self._check_module(
                            node.module,
                            None,
                            node.lineno,
                            lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                        )
                        if result:
                            results.append(result)
                        
                        # 检查导入的具体函数
                        for alias in node.names:
                            full_name = f"{node.module}.{alias.name}"
                            func_result = self._check_function(
                                full_name,
                                alias.asname,
                                node.lineno,
                                lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                            )
                            if func_result:
                                results.append(func_result)
        
        except SyntaxError:
            # AST解析失败，使用正则
            results.extend(self._regex_detect(source_code))
        
        return results
    
    def _check_module(
        self,
        module: str,
        alias: Optional[str],
        line_number: int,
        code_snippet: str
    ) -> Optional[DangerousImport]:
        """检查模块是否危险"""
        # 检查白名单
        if module in self._whitelist:
            return None
        
        # 检查自定义危险模块
        if module in self._custom_dangerous:
            risk_level, reason, suggestion = self._custom_dangerous[module]
            return DangerousImport(
                module=module,
                alias=alias,
                line_number=line_number,
                risk_level=risk_level,
                reason=reason,
                suggestion=suggestion,
                code_snippet=code_snippet.strip()
            )
        
        # 检查内置危险模块
        for dangerous_mod, (risk_level, reason, suggestion) in self._dangerous_modules.items():
            if module == dangerous_mod or module.startswith(dangerous_mod + '.'):
                return DangerousImport(
                    module=module,
                    alias=alias,
                    line_number=line_number,
                    risk_level=risk_level,
                    reason=reason,
                    suggestion=suggestion,
                    code_snippet=code_snippet.strip()
                )
        
        return None
    
    def _check_function(
        self,
        full_name: str,
        alias: Optional[str],
        line_number: int,
        code_snippet: str
    ) -> Optional[DangerousImport]:
        """检查导入的函数是否危险"""
        for func_name, (risk_level, reason, suggestion) in self._dangerous_functions.items():
            if full_name.endswith('.' + func_name) or full_name == func_name:
                return DangerousImport(
                    module=full_name,
                    alias=alias,
                    line_number=line_number,
                    risk_level=risk_level,
                    reason=reason,
                    suggestion=suggestion,
                    code_snippet=code_snippet.strip()
                )
        
        return None
    
    def _regex_detect(self, source_code: str) -> List[DangerousImport]:
        """正则检测（当AST失败时）"""
        results = []
        lines = source_code.split('\n')
        
        import_pattern = re.compile(r'^\s*(?:import|from)\s+([a-zA-Z0-9_.]+)')
        
        for i, line in enumerate(lines, 1):
            match = import_pattern.match(line)
            if match:
                module = match.group(1)
                result = self._check_module(module, None, i, line)
                if result:
                    results.append(result)
        
        return results
    
    def add_dangerous_module(
        self,
        module: str,
        risk_level: RiskLevel,
        reason: str,
        suggestion: str
    ) -> None:
        """添加自定义危险模块"""
        self._custom_dangerous[module] = (risk_level, reason, suggestion)
    
    def add_to_whitelist(self, module: str) -> None:
        """添加到白名单"""
        self._whitelist.add(module)
    
    def remove_from_whitelist(self, module: str) -> None:
        """从白名单移除"""
        self._whitelist.discard(module)
    
    def scan_file(self, file_path: str) -> List[DangerousImport]:
        """扫描文件"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source_code = f.read()
            return self.detect(source_code)
        except Exception:
            return []
    
    def scan_directory(self, directory: str, recursive: bool = True) -> Dict[str, List[DangerousImport]]:
        """扫描目录"""
        import os
        results = {}
        
        if recursive:
            for root, dirs, files in os.walk(directory):
                # 排除常见目录
                dirs[:] = [d for d in dirs if d not in {'node_modules', '.git', '__pycache__', 'venv', '.venv'}]
                
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        imports = self.scan_file(file_path)
                        if imports:
                            results[file_path] = imports
        else:
            for file in os.listdir(directory):
                if file.endswith('.py'):
                    file_path = os.path.join(directory, file)
                    imports = self.scan_file(file_path)
                    if imports:
                        results[file_path] = imports
        
        return results
    
    def get_summary(self, imports: List[DangerousImport]) -> Dict[str, Any]:
        """获取摘要"""
        by_risk = {}
        by_module = {}
        
        for imp in imports:
            risk = imp.risk_level.value
            by_risk[risk] = by_risk.get(risk, 0) + 1
            
            base_module = imp.module.split('.')[0]
            by_module[base_module] = by_module.get(base_module, 0) + 1
        
        return {
            "total_dangerous_imports": len(imports),
            "by_risk_level": by_risk,
            "by_module": by_module,
            "critical_count": by_risk.get("critical", 0),
            "high_count": by_risk.get("high", 0)
        }
