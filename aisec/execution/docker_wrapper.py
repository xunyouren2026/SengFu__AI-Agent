"""
Docker沙箱封装 - 容器化安全执行环境
"""
import json
import subprocess
import time
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ContainerStatus(Enum):
    """容器状态"""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ContainerConfig:
    """容器配置"""
    image: str = "python:3.11-slim"
    memory_limit: str = "512m"
    cpu_limit: float = 1.0
    timeout: int = 60
    network_disabled: bool = True
    read_only_root: bool = True
    no_new_privileges: bool = True
    cap_drop: List[str] = field(default_factory=lambda: ["ALL"])
    cap_add: List[str] = field(default_factory=list)
    security_opt: List[str] = field(default_factory=lambda: ["no-new-privileges"])
    volumes: Dict[str, str] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    working_dir: str = "/sandbox"
    user: str = "nobody"


@dataclass
class ExecutionResult:
    """执行结果"""
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    container_id: str
    resource_usage: Dict[str, Any] = field(default_factory=dict)


class DockerSandbox:
    """Docker沙箱封装"""
    
    def __init__(self, config: Optional[ContainerConfig] = None):
        self._config = config or ContainerConfig()
        self._container_id: Optional[str] = None
        self._status = ContainerStatus.CREATED
        self._docker_available = self._check_docker()
    
    def _check_docker(self) -> bool:
        """检查Docker是否可用"""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def _build_run_args(self) -> List[str]:
        """构建docker run参数"""
        args = [
            "docker", "run",
            "--rm",  # 自动删除容器
            "-d",    # 后台运行
            f"--memory={self._config.memory_limit}",
            f"--cpus={self._config.cpu_limit}",
            f"--timeout={self._config.timeout}",
        ]
        
        # 网络禁用
        if self._config.network_disabled:
            args.append("--network=none")
        
        # 只读根文件系统
        if self._config.read_only_root:
            args.append("--read-only")
        
        # 禁止新权限
        if self._config.no_new_privileges:
            args.append("--security-opt=no-new-privileges")
        
        # 能力控制
        for cap in self._config.cap_drop:
            args.append(f"--cap-drop={cap}")
        for cap in self._config.cap_add:
            args.append(f"--cap-add={cap}")
        
        # 安全选项
        for opt in self._config.security_opt:
            args.append(f"--security-opt={opt}")
        
        # 卷挂载
        for host_path, container_path in self._config.volumes.items():
            args.append(f"-v={host_path}:{container_path}:ro")
        
        # 环境变量
        for key, value in self._config.environment.items():
            args.append(f"-e={key}={value}")
        
        # 工作目录
        args.append(f"-w={self._config.working_dir}")
        
        # 用户
        args.append(f"-u={self._config.user}")
        
        # 镜像
        args.append(self._config.image)
        
        return args
    
    def create_container(self, command: List[str]) -> str:
        """创建容器"""
        if not self._docker_available:
            raise RuntimeError("Docker不可用")
        
        args = self._build_run_args()
        args.extend(command)
        
        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=self._config.timeout + 10
            )
            
            if result.returncode == 0:
                self._container_id = result.stdout.strip()
                self._status = ContainerStatus.RUNNING
                return self._container_id
            else:
                self._status = ContainerStatus.ERROR
                raise RuntimeError(f"创建容器失败: {result.stderr}")
        
        except subprocess.TimeoutExpired:
            self._status = ContainerStatus.ERROR
            raise RuntimeError("创建容器超时")
    
    def execute(
        self,
        code: str,
        language: str = "python"
    ) -> ExecutionResult:
        """在沙箱中执行代码"""
        start_time = time.time()
        
        if not self._docker_available:
            # Docker不可用时，返回模拟结果
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr="Docker不可用，无法执行代码",
                execution_time=0,
                container_id=""
            )
        
        # 根据语言构建执行命令
        if language == "python":
            command = ["python", "-c", code]
        elif language == "bash":
            command = ["bash", "-c", code]
        else:
            command = ["python", "-c", code]
        
        try:
            # 使用docker run直接执行
            args = self._build_run_args()
            args.remove("-d")  # 移除后台运行标志
            args.extend(command)
            
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=self._config.timeout
            )
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=result.returncode == 0,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time=execution_time,
                container_id="",
                resource_usage={"execution_time": execution_time}
            )
        
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"执行超时（超过{self._config.timeout}秒）",
                execution_time=self._config.timeout,
                container_id=""
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                execution_time=time.time() - start_time,
                container_id=""
            )
    
    def execute_file(self, file_path: str, args: List[str] = None) -> ExecutionResult:
        """执行文件"""
        import os
        import tempfile
        
        # 创建临时目录并复制文件
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.basename(file_path)
            dest_path = os.path.join(tmpdir, filename)
            
            # 复制文件
            import shutil
            shutil.copy2(file_path, dest_path)
            
            # 更新卷挂载
            original_volumes = self._config.volumes.copy()
            self._config.volumes[tmpdir] = "/sandbox"
            
            # 构建执行命令
            command = ["python", f"/sandbox/{filename}"]
            if args:
                command.extend(args)
            
            result = self.execute_script(command)
            
            # 恢复卷挂载
            self._config.volumes = original_volumes
            
            return result
    
    def execute_script(self, command: List[str]) -> ExecutionResult:
        """执行脚本命令"""
        start_time = time.time()
        
        if not self._docker_available:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr="Docker不可用",
                execution_time=0,
                container_id=""
            )
        
        try:
            args = self._build_run_args()
            args.remove("-d")
            args.extend(command)
            
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=self._config.timeout
            )
            
            return ExecutionResult(
                success=result.returncode == 0,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time=time.time() - start_time,
                container_id=""
            )
        
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr="执行超时",
                execution_time=self._config.timeout,
                container_id=""
            )
    
    def stop(self) -> bool:
        """停止容器"""
        if self._container_id:
            try:
                subprocess.run(
                    ["docker", "stop", self._container_id],
                    capture_output=True,
                    timeout=10
                )
                self._status = ContainerStatus.STOPPED
                return True
            except Exception:
                return False
        return True
    
    def get_status(self) -> ContainerStatus:
        """获取容器状态"""
        return self._status
    
    def get_config(self) -> ContainerConfig:
        """获取配置"""
        return self._config
    
    def update_config(self, **kwargs) -> None:
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)


class DockerSandboxBuilder:
    """Docker沙箱构建器"""
    
    def __init__(self):
        self._config = ContainerConfig()
    
    def with_image(self, image: str) -> 'DockerSandboxBuilder':
        self._config.image = image
        return self
    
    def with_memory(self, memory: str) -> 'DockerSandboxBuilder':
        self._config.memory_limit = memory
        return self
    
    def with_cpu(self, cpu: float) -> 'DockerSandboxBuilder':
        self._config.cpu_limit = cpu
        return self
    
    def with_timeout(self, timeout: int) -> 'DockerSandboxBuilder':
        self._config.timeout = timeout
        return self
    
    def with_network(self, enabled: bool) -> 'DockerSandboxBuilder':
        self._config.network_disabled = not enabled
        return self
    
    def with_volume(self, host_path: str, container_path: str) -> 'DockerSandboxBuilder':
        self._config.volumes[host_path] = container_path
        return self
    
    def with_env(self, key: str, value: str) -> 'DockerSandboxBuilder':
        self._config.environment[key] = value
        return self
    
    def with_user(self, user: str) -> 'DockerSandboxBuilder':
        self._config.user = user
        return self
    
    def with_capabilities(self, add: List[str] = None, drop: List[str] = None) -> 'DockerSandboxBuilder':
        if add:
            self._config.cap_add.extend(add)
        if drop:
            self._config.cap_drop.extend(drop)
        return self
    
    def build(self) -> DockerSandbox:
        return DockerSandbox(self._config)
