"""
Docker执行器
封装docker-py API实现容器化执行
"""

import subprocess
import json
import os
import time
import uuid
import tempfile
import shutil
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from .interface import (
    SandboxExecutor, SandboxConfig, ExecutionContext, ExecutionResult,
    ExecutionStatus, SandboxState, ResourceLimits
)


class DockerClient:
    """
    Docker客户端封装
    使用subprocess调用docker命令行实现
    """
    
    def __init__(self, docker_path: str = "docker"):
        self.docker_path = docker_path
        self._version: Optional[str] = None
    
    def is_available(self) -> bool:
        """检查Docker是否可用"""
        try:
            result = subprocess.run(
                [self.docker_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def get_version(self) -> Optional[str]:
        """获取Docker版本"""
        if self._version is None:
            try:
                result = subprocess.run(
                    [self.docker_path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    self._version = result.stdout.strip()
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        return self._version
    
    def pull_image(self, image: str) -> Tuple[bool, str]:
        """
        拉取镜像
        
        Args:
            image: 镜像名称
            
        Returns:
            (是否成功, 输出信息)
        """
        try:
            result = subprocess.run(
                [self.docker_path, "pull", image],
                capture_output=True,
                text=True,
                timeout=300
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Pull timeout"
    
    def image_exists(self, image: str) -> bool:
        """检查镜像是否存在"""
        try:
            result = subprocess.run(
                [self.docker_path, "image", "inspect", image],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
    
    def create_container(
        self,
        image: str,
        name: Optional[str] = None,
        command: Optional[List[str]] = None,
        environment: Optional[Dict[str, str]] = None,
        working_dir: Optional[str] = None,
        volumes: Optional[Dict[str, Dict[str, str]]] = None,
        memory_limit: Optional[int] = None,
        cpu_quota: Optional[int] = None,
        cpu_period: Optional[int] = None,
        pids_limit: Optional[int] = None,
        network_disabled: bool = True,
        security_opts: Optional[List[str]] = None,
        cap_drop: Optional[List[str]] = None,
        cap_add: Optional[List[str]] = None,
        read_only_rootfs: bool = False,
        timeout: int = 60
    ) -> Tuple[bool, str, Optional[str]]:
        """
        创建容器
        
        Returns:
            (是否成功, 输出信息, 容器ID)
        """
        cmd = [self.docker_path, "create"]
        
        if name:
            cmd.extend(["--name", name])
        
        if working_dir:
            cmd.extend(["--workdir", working_dir])
        
        if environment:
            for key, value in environment.items():
                cmd.extend(["--env", f"{key}={value}"])
        
        if volumes:
            for host_path, config in volumes.items():
                bind = config.get('bind', host_path)
                mode = config.get('mode', 'rw')
                cmd.extend(["--volume", f"{host_path}:{bind}:{mode}"])
        
        if memory_limit:
            cmd.extend(["--memory", str(memory_limit)])
        
        if cpu_quota and cpu_period:
            cmd.extend(["--cpu-quota", str(cpu_quota)])
            cmd.extend(["--cpu-period", str(cpu_period)])
        
        if pids_limit:
            cmd.extend(["--pids-limit", str(pids_limit)])
        
        if network_disabled:
            cmd.append("--network=none")
        
        if security_opts:
            for opt in security_opts:
                cmd.extend(["--security-opt", opt])
        
        if cap_drop:
            for cap in cap_drop:
                cmd.extend(["--cap-drop", cap])
        
        if cap_add:
            for cap in cap_add:
                cmd.extend(["--cap-add", cap])
        
        if read_only_rootfs:
            cmd.append("--read-only")
        
        cmd.append(image)
        
        if command:
            cmd.extend(command)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            if result.returncode == 0:
                container_id = result.stdout.strip()
                return True, "", container_id
            return False, result.stderr, None
        except subprocess.TimeoutExpired:
            return False, "Create timeout", None
    
    def start_container(self, container_id: str) -> Tuple[bool, str]:
        """启动容器"""
        try:
            result = subprocess.run(
                [self.docker_path, "start", container_id],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0, result.stderr
        except subprocess.TimeoutExpired:
            return False, "Start timeout"
    
    def stop_container(self, container_id: str, timeout: int = 10) -> Tuple[bool, str]:
        """停止容器"""
        try:
            result = subprocess.run(
                [self.docker_path, "stop", "-t", str(timeout), container_id],
                capture_output=True,
                text=True,
                timeout=timeout + 5
            )
            return result.returncode == 0, result.stderr
        except subprocess.TimeoutExpired:
            return False, "Stop timeout"
    
    def remove_container(self, container_id: str, force: bool = False) -> Tuple[bool, str]:
        """删除容器"""
        try:
            cmd = [self.docker_path, "rm"]
            if force:
                cmd.append("-f")
            cmd.append(container_id)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0, result.stderr
        except subprocess.TimeoutExpired:
            return False, "Remove timeout"
    
    def wait_container(self, container_id: str, timeout: int = 60) -> Tuple[bool, int, str]:
        """
        等待容器结束
        
        Returns:
            (是否成功, 退出码, 错误信息)
        """
        try:
            result = subprocess.run(
                [self.docker_path, "wait", container_id],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            if result.returncode == 0:
                exit_code = int(result.stdout.strip())
                return True, exit_code, ""
            return False, -1, result.stderr
        except subprocess.TimeoutExpired:
            return False, -1, "Wait timeout"
    
    def get_logs(
        self,
        container_id: str,
        stdout: bool = True,
        stderr: bool = True
    ) -> Tuple[str, str]:
        """获取容器日志"""
        stdout_data = ""
        stderr_data = ""
        
        try:
            if stdout:
                result = subprocess.run(
                    [self.docker_path, "logs", "--stdout", container_id],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                stdout_data = result.stdout
            
            if stderr:
                result = subprocess.run(
                    [self.docker_path, "logs", "--stderr", container_id],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                stderr_data = result.stderr
        except subprocess.TimeoutExpired:
            pass
        
        return stdout_data, stderr_data
    
    def inspect_container(self, container_id: str) -> Optional[Dict[str, Any]]:
        """检查容器详情"""
        try:
            result = subprocess.run(
                [self.docker_path, "inspect", container_id],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return data[0] if data else None
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            pass
        return None
    
    def get_container_stats(self, container_id: str) -> Optional[Dict[str, Any]]:
        """获取容器资源统计"""
        try:
            result = subprocess.run(
                [self.docker_path, "stats", "--no-stream", "--format", "{{json .}}", container_id],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout.strip())
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            pass
        return None
    
    def exec_in_container(
        self,
        container_id: str,
        command: List[str],
        timeout: int = 60
    ) -> Tuple[int, str, str]:
        """
        在容器中执行命令
        
        Returns:
            (退出码, stdout, stderr)
        """
        try:
            cmd = [self.docker_path, "exec", container_id] + command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Exec timeout"
    
    def copy_to_container(
        self,
        container_id: str,
        src_path: str,
        dest_path: str
    ) -> Tuple[bool, str]:
        """复制文件到容器"""
        try:
            result = subprocess.run(
                [self.docker_path, "cp", src_path, f"{container_id}:{dest_path}"],
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.returncode == 0, result.stderr
        except subprocess.TimeoutExpired:
            return False, "Copy timeout"
    
    def copy_from_container(
        self,
        container_id: str,
        src_path: str,
        dest_path: str
    ) -> Tuple[bool, str]:
        """从容器复制文件"""
        try:
            result = subprocess.run(
                [self.docker_path, "cp", f"{container_id}:{src_path}", dest_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.returncode == 0, result.stderr
        except subprocess.TimeoutExpired:
            return False, "Copy timeout"
    
    def list_containers(
        self,
        all: bool = False,
        filters: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """列出容器"""
        cmd = [self.docker_path, "ps", "--format", "{{json .}}"]
        if all:
            cmd.append("-a")
        if filters:
            for key, value in filters.items():
                cmd.extend(["--filter", f"{key}={value}"])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                containers = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        containers.append(json.loads(line))
                return containers
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            pass
        return []
    
    def prune_containers(self) -> Tuple[bool, str]:
        """清理停止的容器"""
        try:
            result = subprocess.run(
                [self.docker_path, "container", "prune", "-f"],
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.returncode == 0, result.stdout
        except subprocess.TimeoutExpired:
            return False, "Prune timeout"


class DockerExecutor(SandboxExecutor):
    """
    Docker沙箱执行器
    使用Docker容器实现代码隔离执行
    """
    
    def __init__(self, config: SandboxConfig):
        super().__init__(config)
        self.client = DockerClient()
        self._container_id: Optional[str] = None
        self._temp_dir: Optional[str] = None
        self._running_executions: Dict[str, str] = {}  # execution_id -> container_id
    
    def initialize(self) -> bool:
        """初始化Docker执行环境"""
        self._state = SandboxState.INITIALIZING
        
        # 检查Docker是否可用
        if not self.client.is_available():
            self._state = SandboxState.ERROR
            return False
        
        # 确保镜像存在
        if not self.client.image_exists(self.config.image):
            success, msg = self.client.pull_image(self.config.image)
            if not success:
                self._state = SandboxState.ERROR
                return False
        
        # 创建临时目录
        self._temp_dir = tempfile.mkdtemp(prefix=f"sandbox_{self.config.name}_")
        
        self._state = SandboxState.IDLE
        return True
    
    def execute(self, context: ExecutionContext) -> ExecutionResult:
        """在Docker容器中执行代码"""
        start_time = time.time()
        execution_id = context.execution_id
        
        # 验证上下文
        errors = self.validate_context(context)
        if errors:
            return ExecutionResult.create_error(
                execution_id,
                f"Validation failed: {', '.join(errors)}",
                start_time=start_time
            )
        
        self._state = SandboxState.RUNNING
        
        try:
            # 准备执行环境
            work_dir = self._prepare_work_directory(context)
            
            # 创建执行脚本
            script_path = self._create_execution_script(context, work_dir)
            
            # 创建容器
            container_name = f"sandbox_{execution_id}_{uuid.uuid4().hex[:8]}"
            success, msg, container_id = self._create_execution_container(
                container_name, work_dir, context
            )
            
            if not success:
                return ExecutionResult.create_error(
                    execution_id,
                    f"Failed to create container: {msg}",
                    start_time=start_time
                )
            
            self._running_executions[execution_id] = container_id
            
            # 启动容器并等待执行
            success, msg = self.client.start_container(container_id)
            if not success:
                self._cleanup_container(container_id)
                return ExecutionResult.create_error(
                    execution_id,
                    f"Failed to start container: {msg}",
                    start_time=start_time
                )
            
            # 等待执行完成
            timeout = context.timeout or self.config.resource_limits.timeout
            success, exit_code, msg = self.client.wait_container(
                container_id, timeout=timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # 获取输出
            stdout, stderr = self.client.get_logs(container_id)
            
            # 收集输出文件
            output_files = self._collect_output_files(container_id, work_dir)
            
            # 获取资源使用
            resource_usage = self._get_container_resource_usage(container_id)
            
            # 清理容器
            self._cleanup_container(container_id)
            self._running_executions.pop(execution_id, None)
            
            # 构建结果
            if not success:
                status = ExecutionStatus.TIMEOUT
            elif exit_code == 0:
                status = ExecutionStatus.SUCCESS
            else:
                status = ExecutionStatus.FAILED
            
            self._execution_count += 1
            self._last_execution_time = end_time
            self._state = SandboxState.IDLE
            
            return ExecutionResult(
                execution_id=execution_id,
                status=status,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                resource_usage=resource_usage,
                output_files=output_files
            )
            
        except Exception as e:
            self._state = SandboxState.ERROR
            return ExecutionResult.create_error(
                execution_id,
                f"Execution error: {str(e)}",
                start_time=start_time
            )
    
    def execute_batch(self, contexts: List[ExecutionContext]) -> List[ExecutionResult]:
        """批量执行代码"""
        results = []
        for context in contexts:
            result = self.execute(context)
            results.append(result)
        return results
    
    def stop(self, execution_id: str) -> bool:
        """停止执行"""
        container_id = self._running_executions.get(execution_id)
        if container_id:
            success, _ = self.client.stop_container(container_id)
            if success:
                self._cleanup_container(container_id)
                self._running_executions.pop(execution_id, None)
            return success
        return False
    
    def cleanup(self) -> bool:
        """清理执行环境"""
        self._state = SandboxState.CLEANUP
        
        # 停止所有运行中的容器
        for execution_id, container_id in list(self._running_executions.items()):
            self.client.stop_container(container_id, timeout=5)
            self._cleanup_container(container_id)
        self._running_executions.clear()
        
        # 清理临时目录
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
        
        self._state = SandboxState.STOPPED
        return True
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """获取资源使用情况"""
        usage = {
            'total_executions': self._execution_count,
            'running_executions': len(self._running_executions),
            'containers': []
        }
        
        for execution_id, container_id in self._running_executions.items():
            stats = self.client.get_container_stats(container_id)
            if stats:
                usage['containers'].append({
                    'execution_id': execution_id,
                    'container_id': container_id,
                    'stats': stats
                })
        
        return usage
    
    def health_check(self) -> bool:
        """健康检查"""
        return self.client.is_available() and self._state != SandboxState.ERROR
    
    def _prepare_work_directory(self, context: ExecutionContext) -> str:
        """准备工作目录"""
        work_dir = os.path.join(
            self._temp_dir or tempfile.gettempdir(),
            context.execution_id
        )
        os.makedirs(work_dir, exist_ok=True)
        
        # 写入输入文件
        for filename, content in context.files.items():
            file_path = os.path.join(work_dir, filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return work_dir
    
    def _create_execution_script(self, context: ExecutionContext, work_dir: str) -> str:
        """创建执行脚本"""
        script_name = "execute.py" if context.language == "python" else "execute.sh"
        script_path = os.path.join(work_dir, script_name)
        
        if context.language == "python":
            script_content = f'''#!/usr/bin/env python3
import sys
import os

# 设置工作目录
os.chdir('{self.config.workdir}')

# 执行代码
{context.code}
'''
        else:
            script_content = f'''#!/bin/bash
cd {self.config.workdir}
{context.code}
'''
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        return script_path
    
    def _create_execution_container(
        self,
        name: str,
        work_dir: str,
        context: ExecutionContext
    ) -> Tuple[bool, str, Optional[str]]:
        """创建执行容器"""
        limits = self.config.resource_limits
        security = self.config.security
        
        # 构建安全选项
        security_opts = []
        if security.seccomp_profile:
            security_opts.append(f"seccomp={security.seccomp_profile}")
        if security.apparmor_profile:
            security_opts.append(f"apparmor={security.apparmor_profile}")
        if security.no_new_privileges:
            security_opts.append("no-new-privileges")
        
        # 构建命令
        if context.language == "python":
            command = ["python3", f"{self.config.workdir}/execute.py"]
        else:
            command = ["/bin/bash", f"{self.config.workdir}/execute.sh"]
        
        command.extend(context.args)
        
        # 创建容器
        return self.client.create_container(
            image=self.config.image,
            name=name,
            command=command,
            environment=self.config.environment,
            working_dir=self.config.workdir,
            volumes={
                work_dir: {'bind': self.config.workdir, 'mode': 'rw'}
            },
            memory_limit=limits.memory_limit,
            cpu_quota=int(limits.cpu_quota * limits.cpu_period),
            cpu_period=limits.cpu_period,
            pids_limit=limits.pids_limit,
            network_disabled=not self.config.network.enabled,
            security_opts=security_opts if security_opts else None,
            cap_drop=security.drop_capabilities,
            cap_add=security.add_capabilities,
            read_only_rootfs=security.read_only_rootfs,
            timeout=30
        )
    
    def _collect_output_files(
        self,
        container_id: str,
        work_dir: str
    ) -> Dict[str, str]:
        """收集输出文件"""
        output_files = {}
        output_dir = os.path.join(work_dir, "output")
        
        if os.path.exists(output_dir):
            for root, _, files in os.walk(output_dir):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, output_dir)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            output_files[rel_path] = f.read()
                    except (IOError, UnicodeDecodeError):
                        pass
        
        return output_files
    
    def _get_container_resource_usage(self, container_id: str) -> Dict[str, Any]:
        """获取容器资源使用"""
        stats = self.client.get_container_stats(container_id)
        if stats:
            return {
                'cpu_percent': stats.get('CPUPerc', '0%'),
                'memory_usage': stats.get('MemUsage', '0B / 0B'),
                'network_io': stats.get('NetIO', '0B / 0B'),
                'block_io': stats.get('BlockIO', '0B / 0B'),
                'pids': stats.get('PIDs', '0')
            }
        return {}
    
    def _cleanup_container(self, container_id: str) -> None:
        """清理容器"""
        self.client.stop_container(container_id, timeout=5)
        self.client.remove_container(container_id, force=True)
    
    def __del__(self):
        """析构函数"""
        if self.config.auto_cleanup:
            try:
                self.cleanup()
            except Exception:
                pass
