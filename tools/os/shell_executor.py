"""
受限Shell执行器模块
提供命令白名单、参数过滤、超时控制的安全Shell执行
"""

import os
import subprocess
import shlex
import re
import time
import threading
from typing import Optional, Union, List, Dict, Any, Tuple, Set, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """执行状态枚举"""
    SUCCESS = "success"
    TIMEOUT = "timeout"
    DENIED = "denied"
    ERROR = "error"
    RUNNING = "running"


@dataclass
class ExecutionResult:
    """执行结果数据类"""
    command: str
    status: ExecutionStatus
    exit_code: Optional[int]
    stdout: str
    stderr: str
    execution_time: float
    pid: Optional[int]
    message: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CommandRule:
    """命令规则数据类"""
    command: str
    allowed_args: Optional[List[str]] = None  # None表示允许所有参数
    forbidden_args: List[str] = field(default_factory=list)
    max_args: Optional[int] = None
    timeout: Optional[int] = None
    requires_sudo: bool = False
    description: str = ""


class CommandWhitelist:
    """命令白名单管理器"""
    
    def __init__(self):
        """初始化命令白名单"""
        self._allowed_commands: Dict[str, CommandRule] = {}
        self._forbidden_patterns: List[str] = []
        self._init_default_whitelist()
    
    def _init_default_whitelist(self) -> None:
        """初始化默认白名单"""
        # 常用安全命令
        default_commands = [
            CommandRule("ls", allowed_args=["-l", "-a", "-la", "-al", "-h", "-R"], description="列出目录"),
            CommandRule("cat", max_args=1, description="查看文件"),
            CommandRule("head", allowed_args=["-n"], max_args=2, description="查看文件头部"),
            CommandRule("tail", allowed_args=["-n", "-f"], max_args=2, description="查看文件尾部"),
            CommandRule("grep", allowed_args=["-i", "-v", "-n", "-r", "-l"], max_args=5, description="搜索文本"),
            CommandRule("find", allowed_args=["-name", "-type", "-size", "-mtime"], max_args=10, description="查找文件"),
            CommandRule("wc", allowed_args=["-l", "-w", "-c"], max_args=2, description="统计"),
            CommandRule("sort", allowed_args=["-n", "-r", "-u"], description="排序"),
            CommandRule("uniq", allowed_args=["-c", "-d"], description="去重"),
            CommandRule("echo", max_args=10, description="输出文本"),
            CommandRule("pwd", description="显示当前目录"),
            CommandRule("whoami", description="显示当前用户"),
            CommandRule("date", description="显示日期"),
            CommandRule("uname", allowed_args=["-a", "-r", "-s"], description="系统信息"),
            CommandRule("df", allowed_args=["-h", "-k", "-m"], description="磁盘使用"),
            CommandRule("du", allowed_args=["-h", "-s", "-k", "-m"], max_args=5, description="目录大小"),
            CommandRule("ps", allowed_args=["aux", "-ef", "-e", "-f"], description="进程列表"),
            CommandRule("top", allowed_args=["-b", "-n"], timeout=5, description="进程监控"),
            CommandRule("free", allowed_args=["-h", "-m", "-g"], description="内存信息"),
            CommandRule("uptime", description="系统运行时间"),
            CommandRule("hostname", description="主机名"),
            CommandRule("id", description="用户ID信息"),
            CommandRule("env", description="环境变量"),
            CommandRule("which", max_args=1, description="查找命令路径"),
            CommandRule("file", max_args=1, description="文件类型"),
            CommandRule("stat", max_args=1, description="文件状态"),
            CommandRule("diff", allowed_args=["-u", "-r"], max_args=5, description="比较文件"),
            CommandRule("tar", allowed_args=["-c", "-x", "-v", "-z", "-f"], max_args=10, description="归档"),
            CommandRule("gzip", allowed_args=["-d", "-k"], max_args=2, description="压缩"),
            CommandRule("gunzip", max_args=2, description="解压"),
            CommandRule("mkdir", allowed_args=["-p"], max_args=5, description="创建目录"),
            CommandRule("touch", max_args=5, description="创建文件"),
            CommandRule("cp", allowed_args=["-r", "-p", "-v"], max_args=5, description="复制"),
            CommandRule("mv", max_args=5, description="移动"),
            CommandRule("rm", allowed_args=["-r", "-f", "-i"], max_args=5, description="删除"),
            CommandRule("chmod", allowed_args=["-R"], max_args=5, description="修改权限"),
            CommandRule("chown", allowed_args=["-R"], max_args=5, requires_sudo=True, description="修改所有者"),
            CommandRule("python", allowed_args=["--version", "-m", "-c"], max_args=10, description="Python"),
            CommandRule("python3", allowed_args=["--version", "-m", "-c"], max_args=10, description="Python3"),
            CommandRule("pip", allowed_args=["list", "show", "freeze"], max_args=5, description="Pip"),
            CommandRule("pip3", allowed_args=["list", "show", "freeze"], max_args=5, description="Pip3"),
            CommandRule("git", allowed_args=["status", "log", "diff", "branch", "show", "remote", "clone", "pull", "push"], max_args=10, description="Git"),
            CommandRule("curl", allowed_args=["-I", "-s", "-o", "-L"], max_args=5, timeout=30, description="HTTP请求"),
            CommandRule("wget", allowed_args=["-q", "-O"], max_args=5, timeout=60, description="下载"),
            CommandRule("ping", allowed_args=["-c", "-t"], max_args=3, timeout=10, description="网络测试"),
            CommandRule("netstat", allowed_args=["-an", "-tulpn"], description="网络状态"),
            CommandRule("ss", allowed_args=["-tulpn"], description="Socket统计"),
        ]
        
        for rule in default_commands:
            self._allowed_commands[rule.command] = rule
        
        # 危险模式
        self._forbidden_patterns = [
            r'rm\s+-rf\s+/',  # 删除根目录
            r'rm\s+-rf\s+\*',  # 删除所有
            r'>\s*/etc/',  # 覆盖系统文件
            r'sudo\s+rm',  # sudo删除
            r'chmod\s+777',  # 危险权限
            r'eval\s+',  # eval执行
            r'exec\s+',  # exec执行
            r'\$\([^)]+\)',  # 命令替换
            r'`[^`]+`',  # 反引号命令替换
            r';\s*rm',  # 链式删除
            r'\|\s*rm',  # 管道删除
            r'dd\s+if=',  # dd命令
            r':(){:|:&};:',  # fork炸弹
            r'mkfs',  # 格式化
            r'fdisk',  # 分区
            r'killall\s+-9',  # 强制杀死所有
        ]
    
    def add_command(self, rule: CommandRule) -> None:
        """添加允许的命令"""
        self._allowed_commands[rule.command] = rule
    
    def remove_command(self, command: str) -> None:
        """移除允许的命令"""
        self._allowed_commands.pop(command, None)
    
    def add_forbidden_pattern(self, pattern: str) -> None:
        """添加禁止模式"""
        self._forbidden_patterns.append(pattern)
    
    def is_command_allowed(self, command: str) -> Tuple[bool, str]:
        """
        检查命令是否允许
        
        Args:
            command: 命令字符串
            
        Returns:
            (是否允许, 原因)
        """
        # 解析命令
        try:
            parts = shlex.split(command)
        except ValueError as e:
            return False, f"命令解析失败: {e}"
        
        if not parts:
            return False, "空命令"
        
        cmd_name = os.path.basename(parts[0])
        args = parts[1:]
        
        # 检查危险模式
        for pattern in self._forbidden_patterns:
            if re.search(pattern, command):
                return False, f"命令匹配危险模式: {pattern}"
        
        # 检查命令是否在白名单
        if cmd_name not in self._allowed_commands:
            return False, f"命令不在白名单: {cmd_name}"
        
        rule = self._allowed_commands[cmd_name]
        
        # 检查参数数量
        if rule.max_args is not None and len(args) > rule.max_args:
            return False, f"参数数量超限: {len(args)} > {rule.max_args}"
        
        # 检查禁止参数
        for arg in args:
            if arg in rule.forbidden_args:
                return False, f"禁止的参数: {arg}"
        
        # 检查允许参数
        if rule.allowed_args is not None:
            for arg in args:
                # 允许非选项参数（文件路径等）
                if arg.startswith('-') and arg not in rule.allowed_args:
                    return False, f"不允许的参数: {arg}"
        
        return True, "命令允许"
    
    def get_rule(self, command: str) -> Optional[CommandRule]:
        """获取命令规则"""
        try:
            parts = shlex.split(command)
            if parts:
                cmd_name = os.path.basename(parts[0])
                return self._allowed_commands.get(cmd_name)
        except ValueError:
            pass
        return None
    
    def list_allowed_commands(self) -> List[str]:
        """列出所有允许的命令"""
        return list(self._allowed_commands.keys())


class ArgumentFilter:
    """参数过滤器"""
    
    def __init__(self):
        """初始化参数过滤器"""
        self._dangerous_chars = [';', '|', '&', '$', '`', '(', ')', '{', '}', '<', '>', '\n', '\r']
        self._dangerous_patterns = [
            r'\$\([^)]+\)',  # 命令替换
            r'`[^`]+`',  # 反引号
            r'\$\{[^}]+\}',  # 变量扩展
            r'\.\./',  # 路径遍历
        ]
    
    def filter_argument(self, arg: str, escape: bool = True) -> Tuple[str, bool]:
        """
        过滤单个参数
        
        Args:
            arg: 参数字符串
            escape: 是否转义危险字符
            
        Returns:
            (过滤后的参数, 是否被修改)
        """
        original = arg
        modified = False
        
        # 检查危险模式
        for pattern in self._dangerous_patterns:
            if re.search(pattern, arg):
                arg = re.sub(pattern, '', arg)
                modified = True
        
        # 转义危险字符
        if escape:
            for char in self._dangerous_chars:
                if char in arg:
                    arg = arg.replace(char, f'\\{char}')
                    modified = True
        
        return arg, modified or (arg != original)
    
    def filter_arguments(self, args: List[str], escape: bool = True) -> Tuple[List[str], bool]:
        """
        过滤参数列表
        
        Args:
            args: 参数列表
            escape: 是否转义
            
        Returns:
            (过滤后的参数列表, 是否被修改)
        """
        filtered = []
        modified = False
        
        for arg in args:
            new_arg, was_modified = self.filter_argument(arg, escape)
            filtered.append(new_arg)
            if was_modified:
                modified = True
        
        return filtered, modified
    
    def validate_path(self, path: str, base_dir: Optional[str] = None,
                      allow_absolute: bool = True) -> Tuple[bool, str]:
        """
        验证路径安全性
        
        Args:
            path: 路径字符串
            base_dir: 基础目录
            allow_absolute: 是否允许绝对路径
            
        Returns:
            (是否安全, 原因)
        """
        # 检查路径遍历
        if '..' in path:
            return False, "路径包含遍历字符"
        
        # 检查绝对路径
        if os.path.isabs(path) and not allow_absolute:
            return False, "不允许绝对路径"
        
        # 如果指定了基础目录，检查是否在其中
        if base_dir:
            try:
                real_path = os.path.realpath(os.path.join(base_dir, path))
                real_base = os.path.realpath(base_dir)
                if not real_path.startswith(real_base):
                    return False, "路径超出基础目录范围"
            except Exception as e:
                return False, f"路径验证失败: {e}"
        
        return True, "路径安全"


class ShellExecutor:
    """受限Shell执行器"""
    
    def __init__(self, whitelist: Optional[CommandWhitelist] = None,
                 arg_filter: Optional[ArgumentFilter] = None,
                 default_timeout: int = 30,
                 max_output_size: int = 1024 * 1024,  # 1MB
                 working_dir: Optional[str] = None,
                 env: Optional[Dict[str, str]] = None):
        """
        初始化Shell执行器
        
        Args:
            whitelist: 命令白名单
            arg_filter: 参数过滤器
            default_timeout: 默认超时时间
            max_output_size: 最大输出大小
            working_dir: 工作目录
            env: 环境变量
        """
        self.whitelist = whitelist or CommandWhitelist()
        self.arg_filter = arg_filter or ArgumentFilter()
        self.default_timeout = default_timeout
        self.max_output_size = max_output_size
        self.working_dir = working_dir
        self.env = env or {}
        self._execution_history: List[ExecutionResult] = []
        self._lock = threading.Lock()
    
    def execute(self, command: str, timeout: Optional[int] = None,
                cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None,
                input_data: Optional[str] = None,
                check_whitelist: bool = True) -> ExecutionResult:
        """
        执行命令
        
        Args:
            command: 命令字符串
            timeout: 超时时间
            cwd: 工作目录
            env: 环境变量
            input_data: 标准输入
            check_whitelist: 是否检查白名单
            
        Returns:
            执行结果
        """
        start_time = time.time()
        
        # 检查白名单
        if check_whitelist:
            allowed, reason = self.whitelist.is_command_allowed(command)
            if not allowed:
                result = ExecutionResult(
                    command=command,
                    status=ExecutionStatus.DENIED,
                    exit_code=None,
                    stdout="",
                    stderr="",
                    execution_time=0,
                    pid=None,
                    message=f"命令被拒绝: {reason}"
                )
                self._record_execution(result)
                return result
        
        # 获取命令规则
        rule = self.whitelist.get_rule(command)
        if rule and rule.timeout:
            timeout = rule.timeout
        elif timeout is None:
            timeout = self.default_timeout
        
        try:
            # 准备环境
            exec_env = {**os.environ, **self.env}
            if env:
                exec_env.update(env)
            
            # 准备工作目录
            exec_cwd = cwd or self.working_dir
            
            # 执行命令
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE if input_data else None,
                cwd=exec_cwd,
                env=exec_env
            )
            
            try:
                stdout, stderr = process.communicate(
                    input=input_data.encode() if input_data else None,
                    timeout=timeout
                )
                
                # 限制输出大小
                stdout = stdout[:self.max_output_size]
                stderr = stderr[:self.max_output_size]
                
                execution_time = time.time() - start_time
                
                result = ExecutionResult(
                    command=command,
                    status=ExecutionStatus.SUCCESS if process.returncode == 0 else ExecutionStatus.ERROR,
                    exit_code=process.returncode,
                    stdout=stdout.decode('utf-8', errors='replace'),
                    stderr=stderr.decode('utf-8', errors='replace'),
                    execution_time=execution_time,
                    pid=process.pid,
                    message="执行完成"
                )
                
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                
                execution_time = time.time() - start_time
                
                result = ExecutionResult(
                    command=command,
                    status=ExecutionStatus.TIMEOUT,
                    exit_code=None,
                    stdout="",
                    stderr="",
                    execution_time=execution_time,
                    pid=process.pid,
                    message=f"执行超时 ({timeout}秒)"
                )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            result = ExecutionResult(
                command=command,
                status=ExecutionStatus.ERROR,
                exit_code=None,
                stdout="",
                stderr="",
                execution_time=execution_time,
                pid=None,
                message=f"执行错误: {e}"
            )
        
        self._record_execution(result)
        return result
    
    def execute_pipeline(self, commands: List[str],
                         timeout: Optional[int] = None) -> ExecutionResult:
        """
        执行管道命令
        
        Args:
            commands: 命令列表
            timeout: 总超时时间
            
        Returns:
            执行结果
        """
        # 检查所有命令
        for cmd in commands:
            allowed, reason = self.whitelist.is_command_allowed(cmd)
            if not allowed:
                result = ExecutionResult(
                    command=' | '.join(commands),
                    status=ExecutionStatus.DENIED,
                    exit_code=None,
                    stdout="",
                    stderr="",
                    execution_time=0,
                    pid=None,
                    message=f"命令被拒绝: {cmd} - {reason}"
                )
                return result
        
        # 构建管道命令
        pipeline_cmd = ' | '.join(commands)
        return self.execute(pipeline_cmd, timeout=timeout)
    
    def execute_script(self, script: str, interpreter: str = '/bin/bash',
                       timeout: Optional[int] = None) -> ExecutionResult:
        """
        执行脚本
        
        Args:
            script: 脚本内容
            interpreter: 解释器
            timeout: 超时时间
            
        Returns:
            执行结果
        """
        # 检查脚本中的危险模式
        dangerous_patterns = [
            r'rm\s+-rf\s+/',
            r':\(\)\s*\{.*\}',  # fork炸弹
            r'mkfs',
            r'fdisk',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, script):
                result = ExecutionResult(
                    command=f"{interpreter} (script)",
                    status=ExecutionStatus.DENIED,
                    exit_code=None,
                    stdout="",
                    stderr="",
                    execution_time=0,
                    pid=None,
                    message="脚本包含危险内容"
                )
                return result
        
        # 执行脚本
        command = f'{interpreter} -c {shlex.quote(script)}'
        return self.execute(command, timeout=timeout, check_whitelist=False)
    
    def execute_async(self, command: str,
                      callback: Optional[Callable[[ExecutionResult], None]] = None,
                      **kwargs) -> subprocess.Popen:
        """
        异步执行命令
        
        Args:
            command: 命令字符串
            callback: 完成回调
            **kwargs: 其他参数
            
        Returns:
            进程对象
        """
        # 检查白名单
        allowed, reason = self.whitelist.is_command_allowed(command)
        if not allowed:
            raise PermissionError(f"命令被拒绝: {reason}")
        
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=kwargs.get('cwd', self.working_dir),
            env={**os.environ, **self.env, **kwargs.get('env', {})}
        )
        
        if callback:
            def wait_and_callback():
                stdout, stderr = process.communicate(timeout=kwargs.get('timeout', self.default_timeout))
                result = ExecutionResult(
                    command=command,
                    status=ExecutionStatus.SUCCESS if process.returncode == 0 else ExecutionStatus.ERROR,
                    exit_code=process.returncode,
                    stdout=stdout.decode('utf-8', errors='replace'),
                    stderr=stderr.decode('utf-8', errors='replace'),
                    execution_time=0,
                    pid=process.pid,
                    message="异步执行完成"
                )
                callback(result)
            
            thread = threading.Thread(target=wait_and_callback)
            thread.start()
        
        return process
    
    def _record_execution(self, result: ExecutionResult) -> None:
        """记录执行历史"""
        with self._lock:
            self._execution_history.append(result)
            # 保留最近1000条记录
            if len(self._execution_history) > 1000:
                self._execution_history = self._execution_history[-1000:]
    
    def get_execution_history(self, limit: int = 100) -> List[ExecutionResult]:
        """
        获取执行历史
        
        Args:
            limit: 返回数量限制
            
        Returns:
            执行结果列表
        """
        with self._lock:
            return self._execution_history[-limit:]
    
    def clear_history(self) -> None:
        """清空执行历史"""
        with self._lock:
            self._execution_history.clear()
    
    def add_allowed_command(self, rule: CommandRule) -> None:
        """添加允许的命令"""
        self.whitelist.add_command(rule)
    
    def remove_allowed_command(self, command: str) -> None:
        """移除允许的命令"""
        self.whitelist.remove_command(command)
    
    def list_allowed_commands(self) -> List[str]:
        """列出所有允许的命令"""
        return self.whitelist.list_allowed_commands()
    
    def validate_command(self, command: str) -> Tuple[bool, str]:
        """
        验证命令是否可执行
        
        Args:
            command: 命令字符串
            
        Returns:
            (是否可执行, 原因)
        """
        return self.whitelist.is_command_allowed(command)
    
    def get_command_info(self, command: str) -> Optional[Dict[str, Any]]:
        """
        获取命令信息
        
        Args:
            command: 命令字符串
            
        Returns:
            命令信息字典
        """
        rule = self.whitelist.get_rule(command)
        if rule:
            return {
                'command': rule.command,
                'allowed_args': rule.allowed_args,
                'forbidden_args': rule.forbidden_args,
                'max_args': rule.max_args,
                'timeout': rule.timeout,
                'requires_sudo': rule.requires_sudo,
                'description': rule.description
            }
        return None


class SafeCommandBuilder:
    """安全命令构建器"""
    
    def __init__(self, executor: ShellExecutor):
        """
        初始化命令构建器
        
        Args:
            executor: Shell执行器
        """
        self.executor = executor
        self._command_parts: List[str] = []
    
    def cmd(self, command: str) -> 'SafeCommandBuilder':
        """设置命令"""
        self._command_parts = [command]
        return self
    
    def arg(self, arg: str) -> 'SafeCommandBuilder':
        """添加参数"""
        filtered, _ = self.executor.arg_filter.filter_argument(arg)
        self._command_parts.append(filtered)
        return self
    
    def args(self, *args: str) -> 'SafeCommandBuilder':
        """添加多个参数"""
        for arg in args:
            self.arg(arg)
        return self
    
    def option(self, opt: str, value: Optional[str] = None) -> 'SafeCommandBuilder':
        """添加选项"""
        self._command_parts.append(opt)
        if value:
            filtered, _ = self.executor.arg_filter.filter_argument(value)
            self._command_parts.append(filtered)
        return self
    
    def build(self) -> str:
        """构建命令字符串"""
        return ' '.join(shlex.quote(part) if ' ' in part else part for part in self._command_parts)
    
    def execute(self, **kwargs) -> ExecutionResult:
        """执行构建的命令"""
        command = self.build()
        return self.executor.execute(command, **kwargs)


# 类型提示导入
from typing import Callable
