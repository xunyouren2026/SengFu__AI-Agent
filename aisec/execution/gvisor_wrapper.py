"""
gVisor沙箱封装 - 更强的隔离级别
"""
import json
import subprocess
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

from .docker_wrapper import ContainerConfig, ExecutionResult, ContainerStatus


class GVisorRuntime(Enum):
    """gVisor运行时"""
    RUNSC = "runsc"
    KRUN = "krun"


@dataclass
class GVisorConfig:
    """gVisor配置"""
    runtime: GVisorRuntime = GVisorRuntime.RUNSC
    platform: str = "ptrace"  # ptrace, kvm, systrap
    rootless: bool = True
    network_stack: str = "none"  # none, sandbox
    debug: bool = False
    strace: bool = False
    # 继承基础容器配置
    container_config: ContainerConfig = field(default_factory=ContainerConfig)


class GVisorSandbox:
    """gVisor沙箱封装"""
    
    def __init__(self, config: Optional[GVisorConfig] = None):
        self._config = config or GVisorConfig()
        self._gvisor_available = self._check_gvisor()
        self._status = ContainerStatus.CREATED
    
    def _check_gvisor(self) -> bool:
        """检查gVisor是否可用"""
        try:
            # 检查runsc是否安装
            result = subprocess.run(
                ["which", "runsc"],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                return True
            
            # 检查Docker是否配置了runsc运行时
            result = subprocess.run(
                ["docker", "info", "--format", "{{.Runtimes}}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return "runsc" in result.stdout
        except Exception:
            return False
    
    def _build_gvisor_args(self) -> List[str]:
        """构建gVisor特定参数"""
        args = []
        
        # 运行时
        args.append(f"--runtime={self._config.runtime.value}")
        
        # 平台特定配置
        if self._config.platform:
            args.append(f"--platform={self._config.platform}")
        
        return args
    
    def _build_runsc_command(
        self,
        image: str,
        command: List[str]
    ) -> List[str]:
        """构建runsc命令"""
        container_config = self._config.container_config
        
        args = [
            "docker", "run",
            "--rm",
            f"--runtime={self._config.runtime.value}",
            f"--memory={container_config.memory_limit}",
            f"--cpus={container_config.cpu_limit}",
        ]
        
        # gVisor特定选项
        if self._config.network_stack == "none":
            args.append("--network=none")
        
        # 安全选项
        args.append("--security-opt=no-new-privileges")
        
        # 只读根文件系统
        if container_config.read_only_root:
            args.append("--read-only")
        
        # 卷挂载
        for host_path, container_path in container_config.volumes.items():
            args.append(f"-v={host_path}:{container_path}:ro")
        
        # 环境变量
        for key, value in container_config.environment.items():
            args.append(f"-e={key}={value}")
        
        # 工作目录
        args.append(f"-w={container_config.working_dir}")
        
        # 用户
        args.append(f"-u={container_config.user}")
        
        # 镜像和命令
        args.append(image)
        args.extend(command)
        
        return args
    
    def execute(
        self,
        code: str,
        language: str = "python"
    ) -> ExecutionResult:
        """在gVisor沙箱中执行代码"""
        start_time = time.time()
        container_config = self._config.container_config
        
        if not self._gvisor_available:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr="gVisor不可用。请安装runsc: https://gvisor.dev/docs/user_guide/quick_start/",
                execution_time=0,
                container_id=""
            )
        
        # 构建执行命令
        if language == "python":
            command = ["python", "-c", code]
        elif language == "bash":
            command = ["bash", "-c", code]
        else:
            command = ["python", "-c", code]
        
        try:
            args = self._build_runsc_command(
                container_config.image,
                command
            )
            
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=container_config.timeout
            )
            
            execution_time = time.time() - start_time
            self._status = ContainerStatus.RUNNING
            
            return ExecutionResult(
                success=result.returncode == 0,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time=execution_time,
                container_id="",
                resource_usage={
                    "execution_time": execution_time,
                    "runtime": self._config.runtime.value,
                    "platform": self._config.platform
                }
            )
        
        except subprocess.TimeoutExpired:
            self._status = ContainerStatus.ERROR
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"执行超时（超过{container_config.timeout}秒）",
                execution_time=container_config.timeout,
                container_id=""
            )
        except Exception as e:
            self._status = ContainerStatus.ERROR
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                execution_time=time.time() - start_time,
                container_id=""
            )
    
    def execute_with_seccomp(
        self,
        code: str,
        seccomp_profile: Dict[str, Any]
    ) -> ExecutionResult:
        """使用seccomp配置执行"""
        import tempfile
        import os
        
        # 写入seccomp配置文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(seccomp_profile, f)
            profile_path = f.name
        
        try:
            container_config = self._config.container_config
            
            args = [
                "docker", "run", "--rm",
                f"--runtime={self._config.runtime.value}",
                f"--security-opt=seccomp={profile_path}",
                container_config.image,
                "python", "-c", code
            ]
            
            start_time = time.time()
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=container_config.timeout
            )
            
            return ExecutionResult(
                success=result.returncode == 0,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time=time.time() - start_time,
                container_id=""
            )
        finally:
            os.unlink(profile_path)
    
    def get_status(self) -> ContainerStatus:
        """获取状态"""
        return self._status
    
    def get_config(self) -> GVisorConfig:
        """获取配置"""
        return self._config
    
    def is_available(self) -> bool:
        """检查是否可用"""
        return self._gvisor_available


class GVisorSandboxBuilder:
    """gVisor沙箱构建器"""
    
    def __init__(self):
        self._config = GVisorConfig()
    
    def with_runtime(self, runtime: GVisorRuntime) -> 'GVisorSandboxBuilder':
        self._config.runtime = runtime
        return self
    
    def with_platform(self, platform: str) -> 'GVisorSandboxBuilder':
        self._config.platform = platform
        return self
    
    def with_network_stack(self, stack: str) -> 'GVisorSandboxBuilder':
        self._config.network_stack = stack
        return self
    
    def with_container_config(self, config: ContainerConfig) -> 'GVisorSandboxBuilder':
        self._config.container_config = config
        return self
    
    def with_image(self, image: str) -> 'GVisorSandboxBuilder':
        self._config.container_config.image = image
        return self
    
    def with_memory(self, memory: str) -> 'GVisorSandboxBuilder':
        self._config.container_config.memory_limit = memory
        return self
    
    def with_timeout(self, timeout: int) -> 'GVisorSandboxBuilder':
        self._config.container_config.timeout = timeout
        return self
    
    def build(self) -> GVisorSandbox:
        return GVisorSandbox(self._config)


# 预定义的seccomp配置
class SeccompProfiles:
    """Seccomp配置模板"""
    
    @staticmethod
    def default_profile() -> Dict[str, Any]:
        """默认配置"""
        return {
            "defaultAction": "SCMP_ACT_ERRNO",
            "architectures": ["SCMP_ARCH_X86_64"],
            "syscalls": [
                {
                    "names": [
                        "read", "write", "open", "close", "stat", "fstat",
                        "mmap", "munmap", "brk", "ioctl", "access", "pipe",
                        "dup", "dup2", "getpid", "getppid", "getuid", "getgid",
                        "exit_group", "arch_prctl", "gettid", "futex"
                    ],
                    "action": "SCMP_ACT_ALLOW"
                }
            ]
        }
    
    @staticmethod
    def strict_profile() -> Dict[str, Any]:
        """严格配置"""
        return {
            "defaultAction": "SCMP_ACT_KILL",
            "architectures": ["SCMP_ARCH_X86_64"],
            "syscalls": [
                {
                    "names": [
                        "read", "write", "close", "fstat", "mmap", "munmap",
                        "brk", "exit_group", "gettid", "futex"
                    ],
                    "action": "SCMP_ACT_ALLOW"
                }
            ]
        }
    
    @staticmethod
    def network_allowed_profile() -> Dict[str, Any]:
        """允许网络的配置"""
        profile = SeccompProfiles.default_profile()
        profile["syscalls"][0]["names"].extend([
            "socket", "connect", "bind", "listen", "accept",
            "sendto", "recvfrom", "sendmsg", "recvmsg",
            "getsockname", "getpeername", "setsockopt", "getsockopt"
        ])
        return profile
