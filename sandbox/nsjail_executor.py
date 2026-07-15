"""
NsJail执行器
Linux命名空间隔离执行器，使用NsJail实现进程隔离
"""

import subprocess
import os
import time
import uuid
import tempfile
import shutil
import signal
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

from .interface import (
    SandboxExecutor, SandboxConfig, ExecutionContext, ExecutionResult,
    ExecutionStatus, SandboxState, ResourceLimits
)


@dataclass
class NsJailConfig:
    """NsJail配置"""
    # 命名空间配置
    clone_newuser: bool = True       # 用户命名空间
    clone_newns: bool = True         # 挂载命名空间
    clone_newpid: bool = True        # PID命名空间
    clone_newnet: bool = True        # 网络命名空间
    clone_newuts: bool = True        # UTS命名空间
    clone_newipc: bool = True        # IPC命名空间
    clone_newcgroup: bool = True     # Cgroup命名空间
    
    # 资源限制
    time_limit: int = 60             # 时间限制（秒）
    memory_limit: int = 512          # 内存限制（MB）
    cpu_limit: float = 1.0           # CPU限制（核心数）
    pids_limit: int = 100            # 进程数限制
    
    # 文件系统
    mount_proc: bool = True          # 挂载proc
    mount_tmp: bool = True           # 挂载tmpfs
    mount_dev: bool = False          # 挂载dev（最小化）
    read_only: bool = False          # 只读根文件系统
    
    # 网络
    disable_network: bool = True     # 禁用网络
    
    # 安全
    keep_caps: bool = False          # 保留能力
    drop_caps: List[str] = field(default_factory=lambda: [
        'CAP_SYS_ADMIN', 'CAP_NET_ADMIN', 'CAP_SYS_PTRACE',
        'CAP_SYS_MODULE', 'CAP_SYS_RAWIO', 'CAP_SYS_CHROOT'
    ])
    
    # 用户映射
    uid_map: Optional[str] = None    # UID映射
    gid_map: Optional[str] = None    # GID映射


class NsJailConfigWriter:
    """NsJail配置文件写入器"""
    
    @staticmethod
    def write_config(config: NsJailConfig, filepath: str) -> None:
        """
        写入NsJail配置文件
        
        Args:
            config: NsJail配置
            filepath: 配置文件路径
        """
        lines = []
        
        # 命名空间配置
        lines.append(f"clone_new_user: {str(config.clone_newuser).lower()}")
        lines.append(f"clone_new_mount: {str(config.clone_newns).lower()}")
        lines.append(f"clone_new_pid: {str(config.clone_newpid).lower()}")
        lines.append(f"clone_new_net: {str(config.clone_newnet).lower()}")
        lines.append(f"clone_new_uts: {str(config.clone_newuts).lower()}")
        lines.append(f"clone_new_ipc: {str(config.clone_newipc).lower()}")
        lines.append(f"clone_new_cgroup: {str(config.clone_newcgroup).lower()}")
        
        # 资源限制
        lines.append(f"time_limit: {config.time_limit}")
        lines.append(f"memory_limit: {config.memory_limit}")
        lines.append(f"cpu_limit: {int(config.cpu_limit * 100)}")  # 百分比
        lines.append(f"max_procs: {config.pids_limit}")
        
        # 文件系统
        lines.append(f"mount_proc: {str(config.mount_proc).lower()}")
        lines.append(f"mount_tmp: {str(config.mount_tmp).lower()}")
        lines.append(f"mount_dev: {str(config.mount_dev).lower()}")
        
        if config.read_only:
            lines.append("read_only: true")
        
        # 网络
        if config.disable_network:
            lines.append("clone_new_net: true")
        
        # 能力
        if not config.keep_caps:
            lines.append("keep_caps: false")
        
        for cap in config.drop_caps:
            lines.append(f"drop_cap: {cap}")
        
        # 用户映射
        if config.uid_map:
            lines.append(f"uid_map: {config.uid_map}")
        if config.gid_map:
            lines.append(f"gid_map: {config.gid_map}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    @staticmethod
    def write_config_xml(config: NsJailConfig, filepath: str) -> None:
        """
        写入NsJail XML配置文件
        
        Args:
            config: NsJail配置
            filepath: 配置文件路径
        """
        root = ET.Element("nsjail")
        
        # 命名空间
        ns = ET.SubElement(root, "namespace")
        ET.SubElement(ns, "clone_new_user").text = str(config.clone_newuser).lower()
        ET.SubElement(ns, "clone_new_mount").text = str(config.clone_newns).lower()
        ET.SubElement(ns, "clone_new_pid").text = str(config.clone_newpid).lower()
        ET.SubElement(ns, "clone_new_net").text = str(config.clone_newnet).lower()
        ET.SubElement(ns, "clone_new_uts").text = str(config.clone_newuts).lower()
        ET.SubElement(ns, "clone_new_ipc").text = str(config.clone_newipc).lower()
        ET.SubElement(ns, "clone_new_cgroup").text = str(config.clone_newcgroup).lower()
        
        # 资源限制
        rlimits = ET.SubElement(root, "rlimit")
        ET.SubElement(rlimits, "time_limit").text = str(config.time_limit)
        ET.SubElement(rlimits, "memory_limit").text = str(config.memory_limit)
        ET.SubElement(rlimits, "cpu_limit").text = str(int(config.cpu_limit * 100))
        ET.SubElement(rlimits, "max_procs").text = str(config.pids_limit)
        
        # 挂载点
        if config.mount_proc:
            mount = ET.SubElement(root, "mount")
            ET.SubElement(mount, "src").text = "proc"
            ET.SubElement(mount, "dst").text = "/proc"
            ET.SubElement(mount, "type").text = "proc"
        
        if config.mount_tmp:
            mount = ET.SubElement(root, "mount")
            ET.SubElement(mount, "src").text = "none"
            ET.SubElement(mount, "dst").text = "/tmp"
            ET.SubElement(mount, "type").text = "tmpfs"
        
        # 能力
        if config.drop_caps:
            caps = ET.SubElement(root, "capabilities")
            for cap in config.drop_caps:
                ET.SubElement(caps, "drop").text = cap
        
        tree = ET.ElementTree(root)
        tree.write(filepath, encoding='utf-8', xml_declaration=True)


class NsJailClient:
    """
    NsJail客户端
    封装NsJail命令行调用
    """
    
    def __init__(self, nsjail_path: str = "nsjail"):
        self.nsjail_path = nsjail_path
        self._version: Optional[str] = None
    
    def is_available(self) -> bool:
        """检查NsJail是否可用"""
        try:
            result = subprocess.run(
                [self.nsjail_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def get_version(self) -> Optional[str]:
        """获取NsJail版本"""
        if self._version is None:
            try:
                result = subprocess.run(
                    [self.nsjail_path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    self._version = result.stdout.strip()
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        return self._version
    
    def execute(
        self,
        config_path: str,
        command: List[str],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        stdin: Optional[str] = None,
        timeout: int = 60
    ) -> Tuple[int, str, str]:
        """
        执行命令
        
        Args:
            config_path: 配置文件路径
            command: 要执行的命令
            cwd: 工作目录
            env: 环境变量
            stdin: 标准输入
            timeout: 超时
            
        Returns:
            (退出码, stdout, stderr)
        """
        cmd = [
            self.nsjail_path,
            "--config", config_path,
            "--", *command
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=cwd,
                env=env or os.environ.copy(),
                input=stdin,
                timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -9, "", "Execution timed out"
    
    def execute_inline(
        self,
        config: NsJailConfig,
        command: List[str],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: int = 60
    ) -> Tuple[int, str, str]:
        """
        使用内联参数执行命令
        
        Args:
            config: NsJail配置
            command: 要执行的命令
            cwd: 工作目录
            env: 环境变量
            timeout: 超时
            
        Returns:
            (退出码, stdout, stderr)
        """
        cmd = [self.nsjail_path]
        
        # 命名空间参数
        if config.clone_newuser:
            cmd.append("--clone_new_user")
        if config.clone_newns:
            cmd.append("--clone_new_mount")
        if config.clone_newpid:
            cmd.append("--clone_new_pid")
        if config.clone_newnet:
            cmd.append("--clone_new_net")
        if config.clone_newuts:
            cmd.append("--clone_new_uts")
        if config.clone_newipc:
            cmd.append("--clone_new_ipc")
        if config.clone_newcgroup:
            cmd.append("--clone_new_cgroup")
        
        # 资源限制
        cmd.extend(["--time_limit", str(config.time_limit)])
        cmd.extend(["--memory_limit", str(config.memory_limit)])
        cmd.extend(["--cpu_limit", str(int(config.cpu_limit * 100))])
        cmd.extend(["--max_procs", str(config.pids_limit)])
        
        # 能力
        for cap in config.drop_caps:
            cmd.extend(["--drop_cap", cap])
        
        # 命令分隔符
        cmd.append("--")
        cmd.extend(command)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=cwd,
                env=env or os.environ.copy(),
                timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -9, "", "Execution timed out"


class NsJailExecutor(SandboxExecutor):
    """
    NsJail沙箱执行器
    使用Linux命名空间实现进程隔离
    """
    
    def __init__(self, config: SandboxConfig):
        super().__init__(config)
        self.client = NsJailClient()
        self._temp_dir: Optional[str] = None
        self._config_dir: Optional[str] = None
        self._running_processes: Dict[str, subprocess.Popen] = {}
    
    def initialize(self) -> bool:
        """初始化NsJail执行环境"""
        self._state = SandboxState.INITIALIZING
        
        # 检查NsJail是否可用
        if not self.client.is_available():
            self._state = SandboxState.ERROR
            return False
        
        # 创建临时目录
        self._temp_dir = tempfile.mkdtemp(prefix=f"nsjail_{self.config.name}_")
        self._config_dir = os.path.join(self._temp_dir, "configs")
        os.makedirs(self._config_dir, exist_ok=True)
        
        self._state = SandboxState.IDLE
        return True
    
    def execute(self, context: ExecutionContext) -> ExecutionResult:
        """在NsJail中执行代码"""
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
            
            # 创建NsJail配置
            nsjail_config = self._create_nsjail_config(context)
            config_path = os.path.join(
                self._config_dir,
                f"{execution_id}.conf"
            )
            NsJailConfigWriter.write_config(nsjail_config, config_path)
            
            # 创建执行脚本
            script_path = self._create_execution_script(context, work_dir)
            
            # 构建命令
            if context.language == "python":
                command = ["python3", script_path]
            else:
                command = ["/bin/bash", script_path]
            
            command.extend(context.args)
            
            # 执行
            timeout = context.timeout or self.config.resource_limits.timeout
            exit_code, stdout, stderr = self.client.execute(
                config_path=config_path,
                command=command,
                cwd=work_dir,
                env=self._build_environment(),
                stdin=context.input_data,
                timeout=timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # 收集输出文件
            output_files = self._collect_output_files(work_dir)
            
            # 确定状态
            if exit_code == -9:
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
        """批量执行"""
        results = []
        for context in contexts:
            results.append(self.execute(context))
        return results
    
    def stop(self, execution_id: str) -> bool:
        """停止执行"""
        proc = self._running_processes.pop(execution_id, None)
        if proc:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            return True
        return False
    
    def cleanup(self) -> bool:
        """清理执行环境"""
        self._state = SandboxState.CLEANUP
        
        # 停止所有运行中的进程
        for execution_id, proc in list(self._running_processes.items()):
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._running_processes.clear()
        
        # 清理临时目录
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
            self._config_dir = None
        
        self._state = SandboxState.STOPPED
        return True
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """获取资源使用情况"""
        return {
            'total_executions': self._execution_count,
            'running_executions': len(self._running_processes),
            'temp_dir': self._temp_dir
        }
    
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
    
    def _create_nsjail_config(self, context: ExecutionContext) -> NsJailConfig:
        """创建NsJail配置"""
        limits = self.config.resource_limits
        security = self.config.security
        
        return NsJailConfig(
            time_limit=context.timeout or limits.timeout,
            memory_limit=limits.memory_limit // (1024 * 1024),  # 转换为MB
            cpu_limit=limits.cpu_quota,
            pids_limit=limits.pids_limit,
            clone_newnet=not self.config.network.enabled,
            drop_caps=security.drop_capabilities,
            read_only=security.read_only_rootfs
        )
    
    def _create_execution_script(self, context: ExecutionContext, work_dir: str) -> str:
        """创建执行脚本"""
        script_name = "execute.py" if context.language == "python" else "execute.sh"
        script_path = os.path.join(work_dir, script_name)
        
        if context.language == "python":
            script_content = f'''#!/usr/bin/env python3
import sys
import os

os.chdir('{work_dir}')

{context.code}
'''
        else:
            script_content = f'''#!/bin/bash
cd {work_dir}
{context.code}
'''
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        return script_path
    
    def _build_environment(self) -> Dict[str, str]:
        """构建环境变量"""
        env = os.environ.copy()
        env.update(self.config.environment)
        env['SANDBOX'] = 'nsjail'
        return env
    
    def _collect_output_files(self, work_dir: str) -> Dict[str, str]:
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
    
    def __del__(self):
        """析构函数"""
        if self.config.auto_cleanup:
            try:
                self.cleanup()
            except Exception:
                pass


class UnshareExecutor(SandboxExecutor):
    """
    Unshare执行器
    使用Linux unshare命令实现基本的命名空间隔离
    """
    
    def __init__(self, config: SandboxConfig):
        super().__init__(config)
        self._temp_dir: Optional[str] = None
    
    def initialize(self) -> bool:
        """初始化"""
        self._state = SandboxState.INITIALIZING
        self._temp_dir = tempfile.mkdtemp(prefix=f"unshare_{self.config.name}_")
        self._state = SandboxState.IDLE
        return True
    
    def execute(self, context: ExecutionContext) -> ExecutionResult:
        """执行代码"""
        start_time = time.time()
        execution_id = context.execution_id
        
        errors = self.validate_context(context)
        if errors:
            return ExecutionResult.create_error(
                execution_id,
                f"Validation failed: {', '.join(errors)}",
                start_time=start_time
            )
        
        self._state = SandboxState.RUNNING
        
        try:
            work_dir = self._prepare_work_directory(context)
            script_path = self._create_execution_script(context, work_dir)
            
            # 构建unshare命令
            cmd = [
                "unshare",
                "--user", "--pid", "--mount", "--uts", "--ipc",
                "--fork", "--map-root-user",
                "--", "python3" if context.language == "python" else "/bin/bash",
                script_path
            ]
            
            timeout = context.timeout or self.config.resource_limits.timeout
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=work_dir,
                timeout=timeout
            )
            
            end_time = time.time()
            
            status = ExecutionStatus.SUCCESS if result.returncode == 0 else ExecutionStatus.FAILED
            output_files = self._collect_output_files(work_dir)
            
            self._execution_count += 1
            self._last_execution_time = end_time
            self._state = SandboxState.IDLE
            
            return ExecutionResult(
                execution_id=execution_id,
                status=status,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                output_files=output_files
            )
            
        except subprocess.TimeoutExpired:
            return ExecutionResult.create_timeout(
                execution_id, "", "", start_time,
                self.config.resource_limits.timeout
            )
        except Exception as e:
            self._state = SandboxState.ERROR
            return ExecutionResult.create_error(
                execution_id, str(e), start_time=start_time
            )
    
    def execute_batch(self, contexts: List[ExecutionContext]) -> List[ExecutionResult]:
        """批量执行"""
        return [self.execute(ctx) for ctx in contexts]
    
    def stop(self, execution_id: str) -> bool:
        """停止执行"""
        return True
    
    def cleanup(self) -> bool:
        """清理"""
        self._state = SandboxState.CLEANUP
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        self._state = SandboxState.STOPPED
        return True
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """获取资源使用"""
        return {'total_executions': self._execution_count}
    
    def health_check(self) -> bool:
        """健康检查"""
        return self._state != SandboxState.ERROR
    
    def _prepare_work_directory(self, context: ExecutionContext) -> str:
        """准备工作目录"""
        work_dir = os.path.join(self._temp_dir, context.execution_id)
        os.makedirs(work_dir, exist_ok=True)
        for filename, content in context.files.items():
            file_path = os.path.join(work_dir, filename)
            with open(file_path, 'w') as f:
                f.write(content)
        return work_dir
    
    def _create_execution_script(self, context: ExecutionContext, work_dir: str) -> str:
        """创建执行脚本"""
        script_path = os.path.join(work_dir, "execute.py")
        with open(script_path, 'w') as f:
            f.write(context.code)
        os.chmod(script_path, 0o755)
        return script_path
    
    def _collect_output_files(self, work_dir: str) -> Dict[str, str]:
        """收集输出文件"""
        output_files = {}
        output_dir = os.path.join(work_dir, "output")
        if os.path.exists(output_dir):
            for root, _, files in os.walk(output_dir):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, output_dir)
                    try:
                        with open(file_path, 'r') as f:
                            output_files[rel_path] = f.read()
                    except IOError:
                        pass
        return output_files
