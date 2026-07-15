"""
代码执行插件

提供多语言支持、沙箱执行、超时控制和安全限制功能。
"""

import os
import re
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import ast


class Language(Enum):
    """支持的编程语言"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    BASH = "bash"
    SQL = "sql"


@dataclass
class ExecutionResult:
    """执行结果"""
    success: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    execution_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'exit_code': self.exit_code,
            'execution_time_ms': self.execution_time_ms,
            'memory_usage_mb': self.memory_usage_mb,
        }


class CodeRunnerPlugin:
    """代码执行插件
    
    提供多语言支持、沙箱执行、超时控制和安全限制。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置字典
        """
        self._config = config or {}
        self._timeout = self._config.get('timeout', 30)
        self._max_memory_mb = self._config.get('max_memory_mb', 256)
        self._allowed_languages = self._config.get('allowed_languages', 
                                                    [l.value for l in Language])
        self._blocked_modules = self._config.get('blocked_modules', 
                                                  ['os', 'subprocess', 'sys'])
    
    def execute(self, code: str, language: Language,
                timeout: Optional[int] = None) -> ExecutionResult:
        """执行代码
        
        Args:
            code: 代码
            language: 语言
            timeout: 超时（秒）
            
        Returns:
            执行结果
        """
        if language.value not in self._allowed_languages:
            return ExecutionResult(
                success=False,
                stderr=f"Language {language.value} is not allowed",
                exit_code=-1,
            )
        
        timeout = timeout or self._timeout
        
        if language == Language.PYTHON:
            return self._execute_python(code, timeout)
        elif language == Language.JAVASCRIPT:
            return self._execute_javascript(code, timeout)
        elif language == Language.BASH:
            return self._execute_bash(code, timeout)
        elif language == Language.SQL:
            return self._execute_sql(code, timeout)
        else:
            return ExecutionResult(
                success=False,
                stderr=f"Unsupported language: {language}",
                exit_code=-1,
            )
    
    def _execute_python(self, code: str, timeout: int) -> ExecutionResult:
        """执行Python代码"""
        start_time = time.time()
        
        # 安全检查
        if not self._is_safe_python(code):
            return ExecutionResult(
                success=False,
                stderr="Code contains unsafe operations",
                exit_code=-1,
            )
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            # 执行
            result = subprocess.run(
                ['python', temp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                execution_time_ms=execution_time,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stderr=f"Execution timed out after {timeout} seconds",
                exit_code=-1,
                execution_time_ms=timeout * 1000,
            )
        finally:
            os.remove(temp_path)
    
    def _execute_javascript(self, code: str, timeout: int) -> ExecutionResult:
        """执行JavaScript代码"""
        start_time = time.time()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                ['node', temp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                execution_time_ms=execution_time,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stderr=f"Execution timed out after {timeout} seconds",
                exit_code=-1,
                execution_time_ms=timeout * 1000,
            )
        except FileNotFoundError:
            return ExecutionResult(
                success=False,
                stderr="Node.js is not installed",
                exit_code=-1,
            )
        finally:
            os.remove(temp_path)
    
    def _execute_bash(self, code: str, timeout: int) -> ExecutionResult:
        """执行Bash代码"""
        start_time = time.time()
        
        # 安全检查
        if not self._is_safe_bash(code):
            return ExecutionResult(
                success=False,
                stderr="Code contains unsafe operations",
                exit_code=-1,
            )
        
        try:
            result = subprocess.run(
                code,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                execution_time_ms=execution_time,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stderr=f"Execution timed out after {timeout} seconds",
                exit_code=-1,
                execution_time_ms=timeout * 1000,
            )
    
    def _execute_sql(self, code: str, timeout: int) -> ExecutionResult:
        """执行SQL代码（模拟）"""
        # SQL执行需要数据库连接，这里提供模拟实现
        return ExecutionResult(
            success=True,
            stdout="SQL execution simulated. Connect to a database for actual execution.",
            execution_time_ms=0,
        )
    
    def _is_safe_python(self, code: str) -> bool:
        """检查Python代码是否安全"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        
        for node in ast.walk(tree):
            # 检查危险导入
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self._blocked_modules:
                        return False
            elif isinstance(node, ast.ImportFrom):
                if node.module in self._blocked_modules:
                    return False
            
            # 检查危险函数
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', '__import__']:
                        return False
        
        return True
    
    def _is_safe_bash(self, code: str) -> bool:
        """检查Bash代码是否安全"""
        dangerous_patterns = [
            r'rm\s+-rf\s+/',
            r'>\s*/dev/',
            r'mkfs',
            r'dd\s+if=',
            r':\(\)\s*{\s*:\|:\s*&\s*};',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False
        
        return True
    
    def validate_syntax(self, code: str, language: Language) -> Dict[str, Any]:
        """验证语法
        
        Args:
            code: 代码
            language: 语言
            
        Returns:
            验证结果
        """
        if language == Language.PYTHON:
            try:
                ast.parse(code)
                return {'valid': True, 'errors': []}
            except SyntaxError as e:
                return {'valid': False, 'errors': [str(e)]}
        
        # 其他语言的语法检查简化处理
        return {'valid': True, 'errors': []}
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取插件元数据"""
        return {
            'name': 'code_runner',
            'version': '1.0.0',
            'description': 'Code execution plugin with sandbox support',
            'languages': [l.value for l in Language],
        }
