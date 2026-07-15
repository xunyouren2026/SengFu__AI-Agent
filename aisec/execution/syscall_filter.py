"""
系统调用过滤 - Seccomp/BPF系统调用过滤
"""
import json
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class SyscallAction(Enum):
    """系统调用动作"""
    ALLOW = "SCMP_ACT_ALLOW"
    ERRNO = "SCMP_ACT_ERRNO"
    KILL = "SCMP_ACT_KILL"
    KILL_PROCESS = "SCMP_ACT_KILL_PROCESS"
    TRAP = "SCMP_ACT_TRAP"
    LOG = "SCMP_ACT_LOG"


class Architecture(Enum):
    """架构"""
    X86_64 = "SCMP_ARCH_X86_64"
    X86 = "SCMP_ARCH_X86"
    ARM = "SCMP_ARCH_ARM"
    AARCH64 = "SCMP_ARCH_AARCH64"


@dataclass
class SyscallRule:
    """系统调用规则"""
    syscall_names: List[str]
    action: SyscallAction
    args: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SyscallFilter:
    """系统调用过滤器"""
    default_action: SyscallAction = SyscallAction.ERRNO
    architectures: List[Architecture] = field(default_factory=lambda: [Architecture.X86_64])
    rules: List[SyscallRule] = field(default_factory=list)
    
    def to_seccomp_json(self) -> Dict[str, Any]:
        """转换为seccomp JSON格式"""
        return {
            "defaultAction": self.default_action.value,
            "architectures": [arch.value for arch in self.architectures],
            "syscalls": [
                {
                    "names": rule.syscall_names,
                    "action": rule.action.value,
                    **({"args": rule.args} if rule.args else {})
                }
                for rule in self.rules
            ]
        }


class SyscallFilterBuilder:
    """系统调用过滤器构建器"""
    
    def __init__(self):
        self._default_action = SyscallAction.ERRNO
        self._architectures: List[Architecture] = [Architecture.X86_64]
        self._rules: List[SyscallRule] = []
        self._allowed_syscalls: Set[str] = set()
        self._blocked_syscalls: Set[str] = set()
    
    def with_default_action(self, action: SyscallAction) -> 'SyscallFilterBuilder':
        self._default_action = action
        return self
    
    def with_architecture(self, arch: Architecture) -> 'SyscallFilterBuilder':
        if arch not in self._architectures:
            self._architectures.append(arch)
        return self
    
    def allow_syscalls(self, syscalls: List[str]) -> 'SyscallFilterBuilder':
        """允许系统调用"""
        self._allowed_syscalls.update(syscalls)
        return self
    
    def block_syscalls(self, syscalls: List[str]) -> 'SyscallFilterBuilder':
        """阻止系统调用"""
        self._blocked_syscalls.update(syscalls)
        return self
    
    def add_rule(self, rule: SyscallRule) -> 'SyscallFilterBuilder':
        """添加规则"""
        self._rules.append(rule)
        return self
    
    def build(self) -> SyscallFilter:
        """构建过滤器"""
        rules = list(self._rules)
        
        # 添加允许的系统调用
        if self._allowed_syscalls:
            rules.append(SyscallRule(
                syscall_names=list(self._allowed_syscalls),
                action=SyscallAction.ALLOW
            ))
        
        # 添加阻止的系统调用
        if self._blocked_syscalls:
            rules.append(SyscallRule(
                syscall_names=list(self._blocked_syscalls),
                action=SyscallAction.KILL
            ))
        
        return SyscallFilter(
            default_action=self._default_action,
            architectures=self._architectures,
            rules=rules
        )


class SyscallProfiles:
    """预定义系统调用配置"""
    
    # 基础系统调用 - 几乎所有程序都需要
    BASE_SYSCALLS = [
        "read", "write", "close", "fstat", "mmap", "munmap",
        "brk", "ioctl", "exit_group", "arch_prctl"
    ]
    
    # 文件操作相关
    FILE_SYSCALLS = [
        "open", "openat", "stat", "lstat", "access", "faccessat",
        "readlink", "readlinkat", "getdents", "getdents64",
        "fcntl", "dup", "dup2", "dup3", "lseek"
    ]
    
    # 进程管理相关
    PROCESS_SYSCALLS = [
        "getpid", "getppid", "getuid", "getgid", "geteuid", "getegid",
        "gettid", "clone", "fork", "vfork", "execve",
        "wait4", "waitid", "exit", "set_tid_address"
    ]
    
    # 内存相关
    MEMORY_SYSCALLS = [
        "mprotect", "mremap", "msync", "munmap",
        "madvise", "mincore", "mlock", "munlock"
    ]
    
    # 信号相关
    SIGNAL_SYSCALLS = [
        "rt_sigaction", "rt_sigprocmask", "rt_sigreturn",
        "rt_sigsuspend", "rt_sigtimedwait", "sigaltstack",
        "kill", "tgkill", "tkill"
    ]
    
    # 时间相关
    TIME_SYSCALLS = [
        "clock_gettime", "clock_getres", "nanosleep",
        "gettimeofday", "time", "clock_nanosleep"
    ]
    
    # 网络相关
    NETWORK_SYSCALLS = [
        "socket", "connect", "bind", "listen", "accept",
        "accept4", "sendto", "recvfrom", "sendmsg", "recvmsg",
        "getsockname", "getpeername", "setsockopt", "getsockopt",
        "shutdown", "socketpair"
    ]
    
    # 线程相关
    THREAD_SYSCALLS = [
        "futex", "set_robust_list", "get_robust_list",
        "set_thread_area", "get_thread_area"
    ]
    
    # 危险系统调用 - 应该阻止
    DANGEROUS_SYSCALLS = [
        "ptrace",          # 调试接口
        "kexec_load",      # 内核执行
        "kexec_file_load", # 内核执行
        "init_module",     # 加载内核模块
        "finit_module",    # 加载内核模块
        "delete_module",   # 删除内核模块
        "acct",            # 进程记账
        "swapon",          # 启用交换
        "swapoff",         # 禁用交换
        "reboot",          # 重启
        "setuid",          # 设置用户ID
        "setgid",          # 设置组ID
        "setreuid",        # 设置真实/有效用户ID
        "setregid",        # 设置真实/有效组ID
        "setresuid",       # 设置所有用户ID
        "setresgid",       # 设置所有组ID
        "chroot",          # 改变根目录
        "pivot_root",      # 改变根文件系统
        "mount",           # 挂载文件系统
        "umount",          # 卸载文件系统
        "umount2",         # 卸载文件系统
        "create_module",   # 创建内核模块
        "get_kernel_syms", # 获取内核符号
        "query_module",    # 查询内核模块
        "uselib",          # 加载库
        "sysfs",           # 系统文件系统
        "afs_syscall",     # AFS系统调用
        "iopl",            # 设置I/O权限级别
        "ioperm",          # 设置I/O权限
        "vm86",            # 虚拟8086模式
        "vm86old",         # 虚拟8086模式
        "modify_ldt",      # 修改局部描述符表
    ]
    
    @classmethod
    def minimal_profile(cls) -> SyscallFilter:
        """最小权限配置"""
        builder = SyscallFilterBuilder()
        builder.with_default_action(SyscallAction.ERRNO)
        builder.allow_syscalls(cls.BASE_SYSCALLS)
        builder.allow_syscalls(cls.THREAD_SYSCALLS)
        builder.allow_syscalls(["futex"])
        return builder.build()
    
    @classmethod
    def basic_profile(cls) -> SyscallFilter:
        """基本配置"""
        builder = SyscallFilterBuilder()
        builder.with_default_action(SyscallAction.ERRNO)
        builder.allow_syscalls(cls.BASE_SYSCALLS)
        builder.allow_syscalls(cls.FILE_SYSCALLS)
        builder.allow_syscalls(cls.PROCESS_SYSCALLS)
        builder.allow_syscalls(cls.MEMORY_SYSCALLS)
        builder.allow_syscalls(cls.SIGNAL_SYSCALLS)
        builder.allow_syscalls(cls.TIME_SYSCALLS)
        builder.allow_syscalls(cls.THREAD_SYSCALLS)
        return builder.build()
    
    @classmethod
    def network_profile(cls) -> SyscallFilter:
        """允许网络的配置"""
        filter_obj = cls.basic_profile()
        # 添加网络系统调用
        builder = SyscallFilterBuilder()
        builder.with_default_action(SyscallAction.ERRNO)
        builder.allow_syscalls(cls.BASE_SYSCALLS)
        builder.allow_syscalls(cls.FILE_SYSCALLS)
        builder.allow_syscalls(cls.PROCESS_SYSCALLS)
        builder.allow_syscalls(cls.MEMORY_SYSCALLS)
        builder.allow_syscalls(cls.SIGNAL_SYSCALLS)
        builder.allow_syscalls(cls.TIME_SYSCALLS)
        builder.allow_syscalls(cls.THREAD_SYSCALLS)
        builder.allow_syscalls(cls.NETWORK_SYSCALLS)
        return builder.build()
    
    @classmethod
    def strict_profile(cls) -> SyscallFilter:
        """严格配置 - 阻止危险系统调用"""
        builder = SyscallFilterBuilder()
        builder.with_default_action(SyscallAction.ALLOW)
        builder.block_syscalls(cls.DANGEROUS_SYSCALLS)
        return builder.build()
    
    @classmethod
    def python_profile(cls) -> SyscallFilter:
        """Python运行时配置"""
        builder = SyscallFilterBuilder()
        builder.with_default_action(SyscallAction.ERRNO)
        builder.allow_syscalls(cls.BASE_SYSCALLS)
        builder.allow_syscalls(cls.FILE_SYSCALLS)
        builder.allow_syscalls(cls.PROCESS_SYSCALLS)
        builder.allow_syscalls(cls.MEMORY_SYSCALLS)
        builder.allow_syscalls(cls.SIGNAL_SYSCALLS)
        builder.allow_syscalls(cls.TIME_SYSCALLS)
        builder.allow_syscalls(cls.THREAD_SYSCALLS)
        
        # Python特定的系统调用
        builder.allow_syscalls([
            "getrandom",    # 随机数
            "poll",         # 事件轮询
            "ppoll",        # 事件轮询
            "select",       # I/O多路复用
            "pselect6",     # I/O多路复用
            "epoll_create1",# epoll
            "epoll_ctl",    # epoll控制
            "epoll_wait",   # epoll等待
            "pipe2",        # 管道
            "eventfd2",     # 事件文件描述符
            "timerfd_create",# 定时器
            "timerfd_settime",
            "timerfd_gettime",
            "sched_getaffinity",# CPU亲和性
            "sched_setaffinity",
            "sysinfo",      # 系统信息
            "uname",        # 系统名
        ])
        
        return builder.build()


class SyscallFilterManager:
    """系统调用过滤器管理器"""
    
    def __init__(self):
        self._filters: Dict[str, SyscallFilter] = {}
    
    def register_filter(self, name: str, filter_obj: SyscallFilter) -> None:
        """注册过滤器"""
        self._filters[name] = filter_obj
    
    def get_filter(self, name: str) -> Optional[SyscallFilter]:
        """获取过滤器"""
        return self._filters.get(name)
    
    def get_seccomp_json(self, name: str) -> Optional[Dict[str, Any]]:
        """获取seccomp JSON"""
        filter_obj = self.get_filter(name)
        if filter_obj:
            return filter_obj.to_seccomp_json()
        return None
    
    def list_filters(self) -> List[str]:
        """列出所有过滤器"""
        return list(self._filters.keys())
    
    def merge_filters(
        self,
        name: str,
        filter1: SyscallFilter,
        filter2: SyscallFilter
    ) -> SyscallFilter:
        """合并两个过滤器"""
        builder = SyscallFilterBuilder()
        
        # 使用更严格的默认动作
        if filter1.default_action == SyscallAction.KILL or filter2.default_action == SyscallAction.KILL:
            builder.with_default_action(SyscallAction.KILL)
        elif filter1.default_action == SyscallAction.ERRNO or filter2.default_action == SyscallAction.ERRNO:
            builder.with_default_action(SyscallAction.ERRNO)
        else:
            builder.with_default_action(filter1.default_action)
        
        # 合并规则
        for rule in filter1.rules + filter2.rules:
            builder.add_rule(rule)
        
        merged = builder.build()
        self._filters[name] = merged
        return merged
