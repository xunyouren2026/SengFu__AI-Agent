"""
虚拟环境执行器
Python venv隔离执行
"""

import subprocess
import os
import sys
import time
import tempfile
import shutil
import venv
import json
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from .interface import (
    SandboxExecutor, SandboxConfig, ExecutionContext, ExecutionResult,
    ExecutionStatus, SandboxState, ResourceLimits
)


class VirtualEnvManager:
    """
    Python虚拟环境管理器
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or tempfile.gettempdir()
        self._environments: Dict[str, str] = {}
    
    def create(
        self,
        name: str,
        python_version: Optional[str] = None,
        with_pip: bool = True,
        system_site_packages: bool = False,
        clear: bool = False
    ) -> str:
        """
        创建虚拟环境
        
        Args:
            name: 环境名称
            python_version: Python版本
            with_pip: 是否包含pip
            system_site_packages: 是否使用系统站点包
            clear: 是否清除已存在的环境
            
        Returns:
            环境路径
        """
        env_path = os.path.join(self.base_dir, name)
        
        # 创建虚拟环境
        builder = venv.EnvBuilder(
            with_pip=with_pip,
            system_site_packages=system_site_packages,
            clear=clear
        )
        builder.create(env_path)
        
        self._environments[name] = env_path
        return env_path
    
    def get_python_path(self, env_path: str) -> str:
        """获取虚拟环境的Python路径"""
        if os.name == 'nt':  # Windows
            return os.path.join(env_path, 'Scripts', 'python.exe')
        else:  # Unix
            return os.path.join(env_path, 'bin', 'python')
    
    def get_pip_path(self, env_path: str) -> str:
        """获取虚拟环境的pip路径"""
        if os.name == 'nt':
            return os.path.join(env_path, 'Scripts', 'pip.exe')
        else:
            return os.path.join(env_path, 'bin', 'pip')
    
    def install_packages(
        self,
        env_path: str,
        packages: List[str],
        upgrade: bool = False
    ) -> Tuple[bool, str]:
        """
        安装包
        
        Args:
            env_path: 环境路径
            packages: 包列表
            upgrade: 是否升级
            
        Returns:
            (是否成功, 输出)
        """
        pip_path = self.get_pip_path(env_path)
        
        cmd = [pip_path, 'install']
        if upgrade:
            cmd.append('--upgrade')
        cmd.extend(packages)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Installation timeout"
    
    def get_installed_packages(self, env_path: str) -> Dict[str, str]:
        """获取已安装的包"""
        pip_path = self.get_pip_path(env_path)
        
        try:
            result = subprocess.run(
                [pip_path, 'list', '--format=json'],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                packages = json.loads(result.stdout)
                return {pkg['name']: pkg['version'] for pkg in packages}
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            pass
        
        return {}
    
    def activate_env(self, env_path: str) -> Dict[str, str]:
        """
        获取激活环境后的环境变量
        
        Args:
            env_path: 环境路径
            
        Returns:
            环境变量字典
        """
        env = os.environ.copy()
        
        # 设置PATH
        if os.name == 'nt':
            bin_dir = os.path.join(env_path, 'Scripts')
        else:
            bin_dir = os.path.join(env_path, 'bin')
        
        env['PATH'] = bin_dir + os.pathsep + env.get('PATH', '')
        env['VIRTUAL_ENV'] = env_path
        
        # 移除PYTHONHOME
        env.pop('PYTHONHOME', None)
        
        return env
    
    def destroy(self, name: str) -> bool:
        """销毁虚拟环境"""
        if name in self._environments:
            env_path = self._environments[name]
            shutil.rmtree(env_path, ignore_errors=True)
            del self._environments[name]
            return True
        return False
    
    def cleanup(self) -> None:
        """清理所有环境"""
        for name in list(self._environments.keys()):
            self.destroy(name)


class VirtualEnvExecutor(SandboxExecutor):
    """
    虚拟环境执行器
    使用Python venv实现代码隔离
    """
    
    def __init__(self, config: SandboxConfig):
        super().__init__(config)
        self._env_manager = VirtualEnvManager()
        self._env_path: Optional[str] = None
        self._temp_dir: Optional[str] = None
        self._running_processes: Dict[str, subprocess.Popen] = {}
    
    def initialize(self) -> bool:
        """初始化虚拟环境"""
        self._state = SandboxState.INITIALIZING
        
        try:
            # 创建临时目录
            self._temp_dir = tempfile.mkdtemp(prefix=f"venv_sandbox_{self.config.name}_")
            
            # 创建虚拟环境
            env_name = f"sandbox_{self.config.name}"
            self._env_path = self._env_manager.create(
                name=env_name,
                with_pip=True,
                system_site_packages=False
            )
            
            # 安装配置中指定的包
            if 'requirements' in self.config.environment:
                requirements = self.config.environment['requirements']
                if isinstance(requirements, str):
                    packages = requirements.split(',')
                else:
                    packages = requirements
                self._env_manager.install_packages(self._env_path, packages)
            
            self._state = SandboxState.IDLE
            return True
            
        except Exception as e:
            self._state = SandboxState.ERROR
            return False
    
    def execute(self, context: ExecutionContext) -> ExecutionResult:
        """在虚拟环境中执行代码"""
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
            # 准备工作目录
            work_dir = self._prepare_work_directory(context)
            
            # 创建执行脚本
            script_path = self._create_execution_script(context, work_dir)
            
            # 获取Python路径和环境变量
            python_path = self._env_manager.get_python_path(self._env_path)
            env = self._env_manager.activate_env(self._env_path)
            env.update(self.config.environment)
            
            # 构建命令
            cmd = [python_path, script_path] + context.args
            
            # 执行
            timeout = context.timeout or self.config.resource_limits.timeout
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=work_dir,
                env=env,
                input=context.input_data,
                timeout=timeout
            )
            
            end_time = time.time()
            
            # 收集输出文件
            output_files = self._collect_output_files(work_dir)
            
            # 确定状态
            status = ExecutionStatus.SUCCESS if result.returncode == 0 else ExecutionStatus.FAILED
            
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
            self._state = SandboxState.IDLE
            return ExecutionResult.create_timeout(
                execution_id, "", "", start_time,
                self.config.resource_limits.timeout
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
        for proc in list(self._running_processes.values()):
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._running_processes.clear()
        
        # 清理虚拟环境
        self._env_manager.cleanup()
        
        # 清理临时目录
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
        
        self._env_path = None
        self._state = SandboxState.STOPPED
        return True
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """获取资源使用情况"""
        usage = {
            'total_executions': self._execution_count,
            'running_executions': len(self._running_processes),
            'env_path': self._env_path
        }
        
        if self._env_path:
            usage['installed_packages'] = self._env_manager.get_installed_packages(self._env_path)
        
        return usage
    
    def health_check(self) -> bool:
        """健康检查"""
        return self._env_path is not None and os.path.exists(self._env_path)
    
    def install_packages(self, packages: List[str]) -> Tuple[bool, str]:
        """
        安装额外的包
        
        Args:
            packages: 包列表
            
        Returns:
            (是否成功, 输出)
        """
        if not self._env_path:
            return False, "Environment not initialized"
        return self._env_manager.install_packages(self._env_path, packages)
    
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
        script_path = os.path.join(work_dir, "execute.py")
        
        # 添加资源限制
        limits_code = self._generate_limits_code()
        
        script_content = f'''#!/usr/bin/env python3
import sys
import os
import resource

# 设置资源限制
{limits_code}

# 设置工作目录
os.chdir('{work_dir}')

# 执行代码
{context.code}
'''
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        return script_path
    
    def _generate_limits_code(self) -> str:
        """生成资源限制代码"""
        limits = self.config.resource_limits
        code_lines = []
        
        # 内存限制
        if limits.memory_limit:
            memory_mb = limits.memory_limit // (1024 * 1024)
            code_lines.append(f'memory_limit = {limits.memory_limit}')
            code_lines.append('try:')
            code_lines.append('    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))')
            code_lines.append('except (ValueError, resource.error):')
            code_lines.append('    pass')
        
        # CPU时间限制
        if limits.timeout:
            code_lines.append(f'time_limit = {limits.timeout}')
            code_lines.append('try:')
            code_lines.append('    resource.setrlimit(resource.RLIMIT_CPU, (time_limit, time_limit))')
            code_lines.append('except (ValueError, resource.error):')
            code_lines.append('    pass')
        
        # 进程数限制
        if limits.pids_limit:
            code_lines.append(f'pids_limit = {limits.pids_limit}')
            code_lines.append('try:')
            code_lines.append('    resource.setrlimit(resource.RLIMIT_NPROC, (pids_limit, pids_limit))')
            code_lines.append('except (ValueError, resource.error):')
            code_lines.append('    pass')
        
        return '\n'.join(code_lines)
    
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


class SubprocessExecutor(SandboxExecutor):
    """
    子进程执行器
    最简单的执行器，直接在子进程中执行
    """
    
    def __init__(self, config: SandboxConfig):
        super().__init__(config)
        self._temp_dir: Optional[str] = None
    
    def initialize(self) -> bool:
        """初始化"""
        self._state = SandboxState.INITIALIZING
        self._temp_dir = tempfile.mkdtemp(prefix=f"subprocess_{self.config.name}_")
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
            
            env = os.environ.copy()
            env.update(self.config.environment)
            
            timeout = context.timeout or self.config.resource_limits.timeout
            
            result = subprocess.run(
                [sys.executable, script_path] + context.args,
                capture_output=True,
                text=True,
                cwd=work_dir,
                env=env,
                input=context.input_data,
                timeout=timeout
            )
            
            end_time = time.time()
            output_files = self._collect_output_files(work_dir)
            
            status = ExecutionStatus.SUCCESS if result.returncode == 0 else ExecutionStatus.FAILED
            
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
            self._state = SandboxState.IDLE
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
