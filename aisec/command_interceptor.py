"""
命令拦截器 - 拦截和过滤危险命令
"""
import re
import shlex
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum


class CommandAction(Enum):
    """命令动作"""
    ALLOW = "allow"
    BLOCK = "block"
    MODIFY = "modify"
    SANDBOX = "sandbox"
    LOG = "log"


class RiskLevel(Enum):
    """风险等级"""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CommandAnalysis:
    """命令分析结果"""
    original: str
    parsed: List[str]
    risk_level: RiskLevel
    risk_factors: List[str]
    suggested_action: CommandAction
    modified: Optional[str] = None


@dataclass
class InterceptResult:
    """拦截结果"""
    action: CommandAction
    original_command: str
    final_command: Optional[str]
    reason: str
    risk_level: RiskLevel


class CommandInterceptor:
    """命令拦截器"""
    
    def __init__(self):
        self._dangerous_patterns = self._load_dangerous_patterns()
        self._allowed_commands: set = set()
        self._blocked_commands: set = set()
        self._modifiers: List[Callable[[str], str]] = []
        self._hooks: List[Callable[[str], InterceptResult]] = []
    
    def _load_dangerous_patterns(self) -> List[Tuple[re.Pattern, RiskLevel, str]]:
        """加载危险模式"""
        patterns = [
            # 危险命令
            (re.compile(r'\brm\s+-rf\s+/', re.IGNORECASE), RiskLevel.CRITICAL, "递归删除根目录"),
            (re.compile(r'\brm\s+-rf\s+\*', re.IGNORECASE), RiskLevel.CRITICAL, "删除所有文件"),
            (re.compile(r'\bdd\s+if=.*of=/dev/', re.IGNORECASE), RiskLevel.CRITICAL, "写入设备文件"),
            (re.compile(r'\bmkfs\b', re.IGNORECASE), RiskLevel.CRITICAL, "格式化文件系统"),
            (re.compile(r'\bfdisk\b', re.IGNORECASE), RiskLevel.HIGH, "磁盘分区操作"),
            (re.compile(r'\bshutdown\b', re.IGNORECASE), RiskLevel.HIGH, "关机命令"),
            (re.compile(r'\breboot\b', re.IGNORECASE), RiskLevel.HIGH, "重启命令"),
            (re.compile(r'\binit\s+0', re.IGNORECASE), RiskLevel.HIGH, "关机命令"),
            
            # 权限提升
            (re.compile(r'\bsudo\b', re.IGNORECASE), RiskLevel.HIGH, "sudo权限提升"),
            (re.compile(r'\bsu\b', re.IGNORECASE), RiskLevel.HIGH, "su切换用户"),
            (re.compile(r'\bchmod\s+777', re.IGNORECASE), RiskLevel.HIGH, "设置完全开放权限"),
            (re.compile(r'\bchown\b', re.IGNORECASE), RiskLevel.MEDIUM, "修改文件所有者"),
            
            # 网络相关
            (re.compile(r'\bnc\s+-l', re.IGNORECASE), RiskLevel.HIGH, "监听网络端口"),
            (re.compile(r'\bnetcat\b', re.IGNORECASE), RiskLevel.MEDIUM, "netcat网络工具"),
            (re.compile(r'\biptables\b', re.IGNORECASE), RiskLevel.HIGH, "修改防火墙规则"),
            (re.compile(r'\bssh\b', re.IGNORECASE), RiskLevel.MEDIUM, "SSH连接"),
            (re.compile(r'\bscp\b', re.IGNORECASE), RiskLevel.MEDIUM, "SCP文件传输"),
            
            # 代码执行
            (re.compile(r'\beval\b', re.IGNORECASE), RiskLevel.HIGH, "eval执行"),
            (re.compile(r'\bexec\b', re.IGNORECASE), RiskLevel.HIGH, "exec执行"),
            (re.compile(r'\bpython\s+-c', re.IGNORECASE), RiskLevel.MEDIUM, "Python命令行执行"),
            (re.compile(r'\bperl\s+-e', re.IGNORECASE), RiskLevel.MEDIUM, "Perl命令行执行"),
            (re.compile(r'\bruby\s+-e', re.IGNORECASE), RiskLevel.MEDIUM, "Ruby命令行执行"),
            
            # 系统信息收集
            (re.compile(r'\bcat\s+/etc/passwd', re.IGNORECASE), RiskLevel.MEDIUM, "读取密码文件"),
            (re.compile(r'\bcat\s+/etc/shadow', re.IGNORECASE), RiskLevel.HIGH, "读取影子文件"),
            (re.compile(r'\buname\s+-a', re.IGNORECASE), RiskLevel.LOW, "获取系统信息"),
            (re.compile(r'\bps\s+aux', re.IGNORECASE), RiskLevel.LOW, "列出所有进程"),
            
            # 下载执行
            (re.compile(r'\bcurl\s+.*\|\s*bash', re.IGNORECASE), RiskLevel.CRITICAL, "下载并执行脚本"),
            (re.compile(r'\bwget\s+.*\|\s*bash', re.IGNORECASE), RiskLevel.CRITICAL, "下载并执行脚本"),
            (re.compile(r'\bcurl\b', re.IGNORECASE), RiskLevel.MEDIUM, "网络下载"),
            (re.compile(r'\bwget\b', re.IGNORECASE), RiskLevel.MEDIUM, "网络下载"),
        ]
        return patterns
    
    def analyze(self, command: str) -> CommandAnalysis:
        """分析命令"""
        # 解析命令
        try:
            parsed = shlex.split(command)
        except ValueError:
            parsed = command.split()
        
        # 检查危险模式
        risk_factors = []
        max_risk = RiskLevel.SAFE
        
        for pattern, risk, description in self._dangerous_patterns:
            if pattern.search(command):
                risk_factors.append(description)
                if risk.value > max_risk.value:
                    max_risk = risk
        
        # 检查管道和重定向
        if '|' in command:
            risk_factors.append("使用管道")
            if max_risk == RiskLevel.SAFE:
                max_risk = RiskLevel.LOW
        
        if '>' in command or '>>' in command:
            risk_factors.append("输出重定向")
            if max_risk == RiskLevel.SAFE:
                max_risk = RiskLevel.LOW
        
        # 确定建议动作
        if max_risk in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            suggested = CommandAction.BLOCK
        elif max_risk == RiskLevel.MEDIUM:
            suggested = CommandAction.SANDBOX
        elif max_risk == RiskLevel.LOW:
            suggested = CommandAction.LOG
        else:
            suggested = CommandAction.ALLOW
        
        return CommandAnalysis(
            original=command,
            parsed=parsed,
            risk_level=max_risk,
            risk_factors=risk_factors,
            suggested_action=suggested
        )
    
    def intercept(self, command: str) -> InterceptResult:
        """拦截命令"""
        # 检查白名单
        base_cmd = command.split()[0] if command.split() else ""
        if base_cmd in self._allowed_commands:
            return InterceptResult(
                action=CommandAction.ALLOW,
                original_command=command,
                final_command=command,
                reason="命令在白名单中",
                risk_level=RiskLevel.SAFE
            )
        
        # 检查黑名单
        if base_cmd in self._blocked_commands:
            return InterceptResult(
                action=CommandAction.BLOCK,
                original_command=command,
                final_command=None,
                reason="命令在黑名单中",
                risk_level=RiskLevel.HIGH
            )
        
        # 分析命令
        analysis = self.analyze(command)
        
        # 执行钩子
        for hook in self._hooks:
            result = hook(command)
            if result.action != CommandAction.ALLOW:
                return result
        
        # 应用修改器
        modified = command
        for modifier in self._modifiers:
            modified = modifier(modified)
        
        if modified != command:
            return InterceptResult(
                action=CommandAction.MODIFY,
                original_command=command,
                final_command=modified,
                reason="命令已修改",
                risk_level=analysis.risk_level
            )
        
        # 根据分析结果决定动作
        action = analysis.suggested_action
        
        if action == CommandAction.BLOCK:
            return InterceptResult(
                action=action,
                original_command=command,
                final_command=None,
                reason=f"检测到危险模式: {', '.join(analysis.risk_factors)}",
                risk_level=analysis.risk_level
            )
        
        return InterceptResult(
            action=action,
            original_command=command,
            final_command=command,
            reason="命令已分析",
            risk_level=analysis.risk_level
        )
    
    def add_allowed_command(self, command: str) -> None:
        """添加允许的命令"""
        self._allowed_commands.add(command)
    
    def add_blocked_command(self, command: str) -> None:
        """添加阻止的命令"""
        self._blocked_commands.add(command)
    
    def add_modifier(self, modifier: Callable[[str], str]) -> None:
        """添加命令修改器"""
        self._modifiers.append(modifier)
    
    def add_hook(self, hook: Callable[[str], InterceptResult]) -> None:
        """添加拦截钩子"""
        self._hooks.append(hook)
    
    def sanitize_command(self, command: str) -> str:
        """清理命令"""
        # 移除危险字符
        sanitized = command
        
        # 移除反引号命令替换
        sanitized = re.sub(r'`[^`]+`', '', sanitized)
        
        # 移除$()命令替换
        sanitized = re.sub(r'\$\([^)]+\)', '', sanitized)
        
        # 转义分号
        sanitized = sanitized.replace(';', '\\;')
        
        # 移除多余的管道
        sanitized = re.sub(r'\|{2,}', '|', sanitized)
        
        return sanitized.strip()
    
    def is_safe(self, command: str) -> Tuple[bool, str]:
        """检查命令是否安全"""
        result = self.intercept(command)
        is_safe = result.action == CommandAction.ALLOW
        return is_safe, result.reason
