"""
Seccomp系统调用过滤配置
提供默认安全profile，限制容器内进程可用的系统调用
"""

import json
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class SeccompAction(Enum):
    """Seccomp动作枚举"""
    KILL = "SCMP_ACT_KILL"           # 终止进程
    TRAP = "SCMP_ACT_TRAP"           # 发送SIGSYS信号
    ERRNO = "SCMP_ACT_ERRNO"         # 返回错误码
    TRACE = "SCMP_ACT_TRACE"         # 通知追踪器
    ALLOW = "SCMP_ACT_ALLOW"         # 允许执行
    LOG = "SCMP_ACT_LOG"             # 记录并允许
    KILL_PROCESS = "SCMP_ACT_KILL_PROCESS"  # 终止整个进程
    KILL_THREAD = "SCMP_ACT_KILL_THREAD"    # 终止线程
    NOTIFY = "SCMP_ACT_NOTIFY"       # 通知用户空间


class SyscallArch(Enum):
    """系统架构枚举"""
    X86_64 = "SCMP_ARCH_X86_64"
    X86 = "SCMP_ARCH_X86"
    ARM = "SCMP_ARCH_ARM"
    AARCH64 = "SCMP_ARCH_AARCH64"
    MIPS = "SCMP_ARCH_MIPS"
    MIPS64 = "SCMP_ARCH_MIPS64"
    PPC = "SCMP_ARCH_PPC"
    PPC64 = "SCMP_ARCH_PPC64"
    PPC64LE = "SCMP_ARCH_PPC64LE"
    S390 = "SCMP_ARCH_S390"
    S390X = "SCMP_ARCH_S390X"


@dataclass
class SyscallRule:
    """系统调用规则"""
    syscall: str                              # 系统调用名称
    action: SeccompAction = SeccompAction.ALLOW  # 动作
    args: List[Dict[str, Any]] = field(default_factory=list)  # 参数条件
    comment: Optional[str] = None             # 注释
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'names': [self.syscall],
            'action': self.action.value
        }
        if self.args:
            result['args'] = self.args
        return result


class SeccompProfile:
    """
    Seccomp配置文件
    用于限制进程可用的系统调用
    """
    
    # 默认允许的系统调用（基础操作）
    DEFAULT_ALLOWED_SYSCALLS: Set[str] = {
        # 文件操作
        'read', 'write', 'open', 'close', 'stat', 'fstat', 'lstat',
        'openat', 'fstatat', 'newfstatat', 'fstatfs', 'statfs',
        'readv', 'writev', 'preadv', 'pwritev', 'pread64', 'pwrite64',
        'lseek', 'llseek', 'dup', 'dup2', 'dup3', 'fcntl', 'ioctl',
        'mmap', 'mmap2', 'munmap', 'mprotect', 'msync', 'madvise',
        'getdents', 'getdents64', 'getcwd', 'chdir', 'fchdir',
        
        # 内存操作
        'brk', 'sbrk',
        
        # 进程控制
        'exit', 'exit_group', 'getpid', 'getppid', 'gettid',
        'nanosleep', 'clock_nanosleep', 'clock_gettime', 'clock_getres',
        'time', 'gettimeofday', 'settimeofday',
        
        # 线程操作
        'clone', 'clone3', 'set_tid_address', 'set_robust_list',
        'futex', 'get_robust_list',
        
        # 信号处理
        'rt_sigaction', 'rt_sigprocmask', 'rt_sigreturn', 'rt_sigsuspend',
        'rt_sigpending', 'rt_sigqueueinfo', 'rt_tgsigqueueinfo',
        'sigaltstack', 'signalfd', 'signalfd4',
        
        # 文件系统
        'access', 'faccessat', 'faccessat2', 'mkdir', 'mkdirat',
        'rmdir', 'unlink', 'unlinkat', 'rename', 'renameat', 'renameat2',
        'link', 'linkat', 'symlink', 'symlinkat', 'readlink', 'readlinkat',
        'umask', 'chmod', 'fchmod', 'fchmodat', 'chown', 'fchown',
        'lchown', 'fchownat', 'truncate', 'ftruncate',
        
        # 用户/组信息
        'getuid', 'geteuid', 'getgid', 'getegid',
        'getresuid', 'getresgid', 'getgroups', 'setgroups',
        'getpgrp', 'getsid', 'setsid',
        
        # 管道
        'pipe', 'pipe2',
        
        # 事件轮询
        'select', 'pselect6', 'poll', 'ppoll', 'epoll_create',
        'epoll_create1', 'epoll_ctl', 'epoll_wait', 'epoll_pwait',
        
        # 时间相关
        'timer_create', 'timer_delete', 'timer_settime', 'timer_gettime',
        'timer_getoverrun', 'timerfd_create', 'timerfd_settime',
        'timerfd_gettime',
        
        # 随机数
        'getrandom', 'sysinfo',
        
        # 套接字（可选）
        'socket', 'socketpair', 'bind', 'listen', 'accept', 'accept4',
        'connect', 'getsockname', 'getpeername', 'sendto', 'recvfrom',
        'sendmsg', 'recvmsg', 'shutdown', 'getsockopt', 'setsockopt',
        
        # 其他基础
        'uname', 'arch_prctl', 'getrlimit', 'setrlimit', 'prlimit64',
        'getauxval', 'sched_getaffinity', 'sched_setaffinity',
        'sched_yield', 'sched_getparam', 'sched_setparam',
        'sched_getscheduler', 'sched_setscheduler',
    }
    
    # 危险系统调用（应该禁止）
    DANGEROUS_SYSCALLS: Set[str] = {
        # 权限提升
        'setuid', 'seteuid', 'setgid', 'setegid',
        'setreuid', 'setregid', 'setresuid', 'setresgid',
        'capset', 'capget',
        
        # 系统管理
        'reboot', 'kexec_load', 'kexec_file_load',
        'init_module', 'finit_module', 'delete_module',
        'acct', 'swapon', 'swapoff',
        
        # 文件系统危险操作
        'mount', 'umount', 'umount2', 'pivot_root', 'chroot',
        'pivot_root', 'uselib',
        
        # 系统调用劫持
        'ptrace', 'process_vm_readv', 'process_vm_writev',
        
        # 网络配置
        'sethostname', 'setdomainname',
        
        # 内核模块
        'create_module', 'query_module', 'get_kernel_syms',
        
        # 系统控制
        'sysctl', '_sysctl',
        
        # 危险IO
        'iopl', 'ioperm', 'vm86', 'vm86old',
        
        # 时间设置
        'stime', 'adjtimex', 'clock_settime',
        
        # 用户命名空间
        'unshare', 'setns',
        
        # 其他危险
        'personality', 'modify_ldt', 'seccomp',
        'prctl',  # 需要限制参数
    }
    
    def __init__(
        self,
        default_action: SeccompAction = SeccompAction.ERRNO,
        architectures: Optional[List[SyscallArch]] = None
    ):
        """
        初始化Seccomp配置
        
        Args:
            default_action: 默认动作
            architectures: 支持的架构列表
        """
        self.default_action = default_action
        self.architectures = architectures or [SyscallArch.X86_64, SyscallArch.X86]
        self._allowed_syscalls: Set[str] = set(self.DEFAULT_ALLOWED_SYSCALLS)
        self._blocked_syscalls: Set[str] = set(self.DANGEROUS_SYSCALLS)
        self._custom_rules: List[SyscallRule] = []
    
    def allow_syscall(self, syscall: str) -> 'SeccompProfile':
        """
        允许系统调用
        
        Args:
            syscall: 系统调用名称
            
        Returns:
            self，支持链式调用
        """
        self._allowed_syscalls.add(syscall)
        self._blocked_syscalls.discard(syscall)
        return self
    
    def block_syscall(self, syscall: str) -> 'SeccompProfile':
        """
        禁止系统调用
        
        Args:
            syscall: 系统调用名称
            
        Returns:
            self，支持链式调用
        """
        self._blocked_syscalls.add(syscall)
        self._allowed_syscalls.discard(syscall)
        return self
    
    def add_rule(self, rule: SyscallRule) -> 'SeccompProfile':
        """
        添加自定义规则
        
        Args:
            rule: 系统调用规则
            
        Returns:
            self，支持链式调用
        """
        self._custom_rules.append(rule)
        return self
    
    def allow_syscalls(self, syscalls: List[str]) -> 'SeccompProfile':
        """批量允许系统调用"""
        for syscall in syscalls:
            self.allow_syscall(syscall)
        return self
    
    def block_syscalls(self, syscalls: List[str]) -> 'SeccompProfile':
        """批量禁止系统调用"""
        for syscall in syscalls:
            self.block_syscall(syscall)
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式（Docker/Containerd兼容）
        
        Returns:
            配置字典
        """
        # 构建系统调用规则
        syscalls = []
        
        # 添加允许的系统调用
        for syscall in sorted(self._allowed_syscalls):
            syscalls.append({
                'names': [syscall],
                'action': SeccompAction.ALLOW.value
            })
        
        # 添加禁止的系统调用
        for syscall in sorted(self._blocked_syscalls):
            syscalls.append({
                'names': [syscall],
                'action': SeccompAction.KILL.value
            })
        
        # 添加自定义规则
        for rule in self._custom_rules:
            syscalls.append(rule.to_dict())
        
        return {
            'defaultAction': self.default_action.value,
            'defaultErrnoRet': 1,  # EPERM
            'architectures': [arch.value for arch in self.architectures],
            'syscalls': syscalls
        }
    
    def to_json(self, indent: int = 2) -> str:
        """
        转换为JSON字符串
        
        Args:
            indent: 缩进空格数
            
        Returns:
            JSON字符串
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, filepath: str) -> None:
        """
        保存到文件
        
        Args:
            filepath: 文件路径
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, filepath: str) -> 'SeccompProfile':
        """
        从文件加载
        
        Args:
            filepath: 文件路径
            
        Returns:
            SeccompProfile实例
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SeccompProfile':
        """
        从字典创建
        
        Args:
            data: 配置字典
            
        Returns:
            SeccompProfile实例
        """
        default_action = SeccompAction(data.get('defaultAction', SeccompAction.ERRNO.value))
        arch_strs = data.get('architectures', [])
        architectures = [SyscallArch(arch) for arch in arch_strs]
        
        profile = cls(default_action=default_action, architectures=architectures)
        
        # 解析系统调用规则
        for syscall_rule in data.get('syscalls', []):
            names = syscall_rule.get('names', [])
            action = SeccompAction(syscall_rule.get('action', SeccompAction.ALLOW.value))
            args = syscall_rule.get('args', [])
            
            for name in names:
                rule = SyscallRule(syscall=name, action=action, args=args)
                profile.add_rule(rule)
                
                if action == SeccompAction.ALLOW:
                    profile._allowed_syscalls.add(name)
                    profile._blocked_syscalls.discard(name)
                elif action == SeccompAction.KILL:
                    profile._blocked_syscalls.add(name)
                    profile._allowed_syscalls.discard(name)
        
        return profile


class SeccompProfileBuilder:
    """Seccomp配置构建器"""
    
    @staticmethod
    def create_default() -> SeccompProfile:
        """创建默认安全配置"""
        return SeccompProfile(default_action=SeccompAction.ERRNO)
    
    @staticmethod
    def create_strict() -> SeccompProfile:
        """创建严格安全配置（禁止更多系统调用）"""
        profile = SeccompProfile(default_action=SeccompAction.KILL)
        
        # 只允许最基本的系统调用
        basic_syscalls = {
            'read', 'write', 'exit', 'exit_group', 'getpid',
            'nanosleep', 'clock_gettime', 'mmap', 'munmap',
            'brk', 'futex', 'rt_sigaction', 'rt_sigprocmask',
        }
        
        profile._allowed_syscalls = basic_syscalls
        profile._blocked_syscalls = set()
        
        return profile
    
    @staticmethod
    def create_python_runtime() -> SeccompProfile:
        """创建Python运行时安全配置"""
        profile = SeccompProfile(default_action=SeccompAction.ERRNO)
        
        # Python需要的额外系统调用
        python_syscalls = {
            'execve', 'execveat',  # 进程执行
            'wait4', 'waitid', 'waitpid',  # 进程等待
            'vfork', 'fork',  # 进程创建
            'getrandom',  # 随机数
            'sysinfo',  # 系统信息
            'uname',  # 系统名称
        }
        
        profile.allow_syscalls(list(python_syscalls))
        
        # 禁止危险的Python相关系统调用
        profile.block_syscall('ptrace')
        profile.block_syscall('process_vm_readv')
        profile.block_syscall('process_vm_writev')
        
        return profile
    
    @staticmethod
    def create_network_enabled() -> SeccompProfile:
        """创建允许网络的配置"""
        profile = SeccompProfile(default_action=SeccompAction.ERRNO)
        
        # 网络相关系统调用
        network_syscalls = {
            'socket', 'socketpair', 'bind', 'listen', 'accept', 'accept4',
            'connect', 'getsockname', 'getpeername', 'sendto', 'recvfrom',
            'sendmsg', 'recvmsg', 'shutdown', 'getsockopt', 'setsockopt',
            'send', 'recv',
        }
        
        profile.allow_syscalls(list(network_syscalls))
        
        return profile
    
    @staticmethod
    def create_filesystem_restricted() -> SeccompProfile:
        """创建文件系统受限配置"""
        profile = SeccompProfile(default_action=SeccompAction.ERRNO)
        
        # 禁止文件系统修改操作
        restricted_syscalls = {
            'mkdir', 'mkdirat', 'rmdir', 'unlink', 'unlinkat',
            'rename', 'renameat', 'renameat2', 'link', 'linkat',
            'symlink', 'symlinkat', 'chmod', 'fchmod', 'fchmodat',
            'chown', 'fchown', 'lchown', 'fchownat',
            'truncate', 'ftruncate', 'creat',
        }
        
        profile.block_syscalls(list(restricted_syscalls))
        
        return profile


class SeccompValidator:
    """Seccomp配置验证器"""
    
    @staticmethod
    def validate(profile: SeccompProfile) -> List[str]:
        """
        验证配置
        
        Args:
            profile: Seccomp配置
            
        Returns:
            错误消息列表
        """
        errors = []
        
        # 检查是否有冲突
        conflicts = profile._allowed_syscalls & profile._blocked_syscalls
        if conflicts:
            errors.append(f"Conflicting syscalls: {conflicts}")
        
        # 检查架构
        if not profile.architectures:
            errors.append("No architectures specified")
        
        # 检查默认动作
        if profile.default_action not in [
            SeccompAction.KILL, SeccompAction.ERRNO, SeccompAction.TRAP
        ]:
            errors.append(f"Unsafe default action: {profile.default_action}")
        
        return errors
    
    @staticmethod
    def is_safe(profile: SeccompProfile) -> bool:
        """
        检查配置是否安全
        
        Args:
            profile: Seccomp配置
            
        Returns:
            是否安全
        """
        # 检查危险系统调用是否被禁止
        for syscall in SeccompProfile.DANGEROUS_SYSCALLS:
            if syscall in profile._allowed_syscalls:
                return False
        
        return True


# 预定义的配置实例
DEFAULT_PROFILE = SeccompProfileBuilder.create_default()
STRICT_PROFILE = SeccompProfileBuilder.create_strict()
PYTHON_RUNTIME_PROFILE = SeccompProfileBuilder.create_python_runtime()
NETWORK_ENABLED_PROFILE = SeccompProfileBuilder.create_network_enabled()
FILESYSTEM_RESTRICTED_PROFILE = SeccompProfileBuilder.create_filesystem_restricted()
