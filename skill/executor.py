#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skill 执行引擎模块

本模块提供技能的执行功能，支持 bash/python/node 等多种脚本类型，
管理执行上下文（环境变量、工作目录、超时），捕获输出和错误，
支持并行技能执行。

作者: AGI Framework Team
版本: 1.0.0
"""

from __future__ import annotations

import os
import sys
import json
import time
import signal
import shutil
import asyncio
import threading
import subprocess
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, AsyncIterator
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from contextlib import contextmanager
from abc import ABC, abstractmethod
import tempfile
import logging
import traceback

# 配置日志
logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    """执行错误基类"""
    
    def __init__(self, message: str, exit_code: Optional[int] = None,
                 stdout: str = "", stderr: str = "", duration: float = 0.0):
        self.message = message
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.duration = duration
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        parts = [self.message]
        if self.exit_code is not None:
            parts.append(f"退出码: {self.exit_code}")
        if self.duration > 0:
            parts.append(f"耗时: {self.duration:.2f}s")
        return " | ".join(parts)


class ExecutionTimeoutError(ExecutionError):
    """执行超时错误"""
    pass


class ExecutionCancelledError(ExecutionError):
    """执行被取消错误"""
    pass


class ExecutionResourceError(ExecutionError):
    """执行资源错误"""
    pass


class ExecutionStatus(Enum):
    """执行状态枚举"""
    PENDING = auto()      # 等待执行
    RUNNING = auto()      # 执行中
    COMPLETED = auto()    # 已完成
    FAILED = auto()       # 失败
    TIMEOUT = auto()      # 超时
    CANCELLED = auto()    # 已取消


@dataclass
class ExecutionContext:
    """
    执行上下文
    
    属性:
        working_dir: 工作目录
        env: 环境变量
        timeout: 超时时间（秒）
        user: 执行用户
        group: 执行组
        umask: 文件权限掩码
        cwd: 当前工作目录（执行时设置）
    """
    working_dir: Optional[Path] = None
    env: Dict[str, str] = field(default_factory=dict)
    timeout: int = 300
    user: Optional[str] = None
    group: Optional[str] = None
    umask: Optional[int] = None
    cwd: Optional[Path] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.working_dir and not self.cwd:
            self.cwd = Path(self.working_dir).resolve()
    
    def merge_env(self, env: Dict[str, str]) -> ExecutionContext:
        """
        合并环境变量
        
        参数:
            env: 要合并的环境变量
            
        返回:
            新的执行上下文
        """
        new_env = {**self.env, **env}
        return ExecutionContext(
            working_dir=self.working_dir,
            env=new_env,
            timeout=self.timeout,
            user=self.user,
            group=self.group,
            umask=self.umask,
        )
    
    def with_timeout(self, timeout: int) -> ExecutionContext:
        """
        设置超时时间
        
        参数:
            timeout: 超时时间（秒）
            
        返回:
            新的执行上下文
        """
        return ExecutionContext(
            working_dir=self.working_dir,
            env=self.env.copy(),
            timeout=timeout,
            user=self.user,
            group=self.group,
            umask=self.umask,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'working_dir': str(self.working_dir) if self.working_dir else None,
            'env': self.env,
            'timeout': self.timeout,
            'user': self.user,
            'group': self.group,
            'umask': self.umask,
        }


@dataclass
class ExecutionResult:
    """
    执行结果
    
    属性:
        status: 执行状态
        exit_code: 退出码
        stdout: 标准输出
        stderr: 标准错误
        duration: 执行时长（秒）
        start_time: 开始时间
        end_time: 结束时间
        metadata: 额外元数据
    """
    status: ExecutionStatus
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """是否成功执行"""
        return self.status == ExecutionStatus.COMPLETED and (self.exit_code == 0 or self.exit_code is None)
    
    @property
    def failed(self) -> bool:
        """是否执行失败"""
        return self.status in (ExecutionStatus.FAILED, ExecutionStatus.TIMEOUT, ExecutionStatus.CANCELLED)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'status': self.status.name,
            'exit_code': self.exit_code,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'duration': self.duration,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'metadata': self.metadata,
            'success': self.success,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExecutionResult:
        """从字典创建"""
        return cls(
            status=ExecutionStatus[data['status']],
            exit_code=data.get('exit_code'),
            stdout=data.get('stdout', ''),
            stderr=data.get('stderr', ''),
            duration=data.get('duration', 0.0),
            start_time=datetime.fromisoformat(data['start_time']) if data.get('start_time') else None,
            end_time=datetime.fromisoformat(data['end_time']) if data.get('end_time') else None,
            metadata=data.get('metadata', {}),
        )


@dataclass
class ExecutionTask:
    """
    执行任务
    
    属性:
        id: 任务 ID
        skill_id: 技能 ID
        command: 执行命令
        script: 脚本路径
        interpreter: 解释器
        parameters: 执行参数
        context: 执行上下文
        priority: 优先级（数字越小优先级越高）
        created_at: 创建时间
    """
    id: str
    skill_id: str
    command: Optional[str] = None
    script: Optional[Path] = None
    interpreter: str = "python"
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: ExecutionContext = field(default_factory=ExecutionContext)
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'skill_id': self.skill_id,
            'command': self.command,
            'script': str(self.script) if self.script else None,
            'interpreter': self.interpreter,
            'parameters': self.parameters,
            'context': self.context.to_dict(),
            'priority': self.priority,
            'created_at': self.created_at.isoformat(),
        }


class ScriptExecutor(ABC):
    """脚本执行器抽象基类"""
    
    @abstractmethod
    def execute(self, task: ExecutionTask) -> ExecutionResult:
        """
        执行任务
        
        参数:
            task: 执行任务
            
        返回:
            执行结果
        """
        pass
    
    @abstractmethod
    async def execute_async(self, task: ExecutionTask) -> ExecutionResult:
        """
        异步执行任务
        
        参数:
            task: 执行任务
            
        返回:
            执行结果
        """
        pass
    
    def prepare_script(self, task: ExecutionTask) -> str:
        """
        准备脚本内容
        
        参数:
            task: 执行任务
            
        返回:
            脚本内容
        """
        if task.script and Path(task.script).exists():
            return Path(task.script).read_text(encoding='utf-8')
        elif task.command:
            return task.command
        else:
            raise ExecutionError("没有指定脚本或命令")
    
    def create_temp_script(self, content: str, interpreter: str) -> Path:
        """
        创建临时脚本文件
        
        参数:
            content: 脚本内容
            interpreter: 解释器类型
            
        返回:
            临时文件路径
        """
        # 根据解释器确定文件扩展名
        ext_map = {
            'python': '.py',
            'python3': '.py',
            'bash': '.sh',
            'sh': '.sh',
            'node': '.js',
            'nodejs': '.js',
            'ruby': '.rb',
            'perl': '.pl',
            'php': '.php',
        }
        ext = ext_map.get(interpreter, '.tmp')
        
        # 创建临时文件
        fd, path = tempfile.mkstemp(suffix=ext, prefix='skill_exec_')
        try:
            os.write(fd, content.encode('utf-8'))
            os.close(fd)
            
            # 设置可执行权限
            os.chmod(path, 0o755)
            
            return Path(path)
        except Exception as e:
            os.close(fd)
            if os.path.exists(path):
                os.unlink(path)
            raise ExecutionError(f"创建临时脚本失败: {e}")


class BashExecutor(ScriptExecutor):
    """Bash 脚本执行器"""
    
    def execute(self, task: ExecutionTask) -> ExecutionResult:
        """执行 Bash 脚本"""
        start_time = datetime.now()
        
        try:
            script_content = self.prepare_script(task)
            
            # 构建命令
            if task.script:
                cmd = ['bash', str(task.script)]
            else:
                cmd = ['bash', '-c', script_content]
            
            # 准备环境
            env = os.environ.copy()
            env.update(task.context.env)
            env['SKILL_ID'] = task.skill_id
            env['SKILL_TASK_ID'] = task.id
            
            # 序列化参数到环境变量
            env['SKILL_PARAMS'] = json.dumps(task.parameters, ensure_ascii=False)
            
            # 执行
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=task.context.cwd,
                env=env,
            )
            
            try:
                stdout, stderr = process.communicate(timeout=task.context.timeout)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                status = ExecutionStatus.COMPLETED if process.returncode == 0 else ExecutionStatus.FAILED
                
                return ExecutionResult(
                    status=status,
                    exit_code=process.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    duration=duration,
                    start_time=start_time,
                    end_time=end_time,
                )
                
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    exit_code=-1,
                    duration=duration,
                    start_time=start_time,
                    end_time=end_time,
                )
                
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                exit_code=-1,
                stderr=str(e),
                duration=duration,
                start_time=start_time,
                end_time=end_time,
            )
    
    async def execute_async(self, task: ExecutionTask) -> ExecutionResult:
        """异步执行 Bash 脚本"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, task)


class PythonExecutor(ScriptExecutor):
    """Python 脚本执行器"""
    
    def execute(self, task: ExecutionTask) -> ExecutionResult:
        """执行 Python 脚本"""
        start_time = datetime.now()
        
        try:
            script_content = self.prepare_script(task)
            
            # 构建命令
            python_exe = sys.executable
            
            if task.script:
                cmd = [python_exe, str(task.script)]
            else:
                # 创建临时脚本
                temp_script = self.create_temp_script(script_content, 'python')
                cmd = [python_exe, str(temp_script)]
            
            # 准备环境
            env = os.environ.copy()
            env.update(task.context.env)
            env['SKILL_ID'] = task.skill_id
            env['SKILL_TASK_ID'] = task.id
            env['SKILL_PARAMS'] = json.dumps(task.parameters, ensure_ascii=False)
            
            # 执行
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=task.context.cwd,
                env=env,
            )
            
            try:
                stdout, stderr = process.communicate(timeout=task.context.timeout)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                # 清理临时文件
                if not task.script and 'temp_script' in locals():
                    try:
                        temp_script.unlink()
                    except:
                        pass
                
                status = ExecutionStatus.COMPLETED if process.returncode == 0 else ExecutionStatus.FAILED
                
                return ExecutionResult(
                    status=status,
                    exit_code=process.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    duration=duration,
                    start_time=start_time,
                    end_time=end_time,
                )
                
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    exit_code=-1,
                    duration=duration,
                    start_time=start_time,
                    end_time=end_time,
                )
                
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                exit_code=-1,
                stderr=str(e),
                duration=duration,
                start_time=start_time,
                end_time=end_time,
            )
    
    async def execute_async(self, task: ExecutionTask) -> ExecutionResult:
        """异步执行 Python 脚本"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, task)


class NodeExecutor(ScriptExecutor):
    """Node.js 脚本执行器"""
    
    def execute(self, task: ExecutionTask) -> ExecutionResult:
        """执行 Node.js 脚本"""
        start_time = datetime.now()
        
        try:
            script_content = self.prepare_script(task)
            
            # 查找 node 可执行文件
            node_exe = shutil.which('node') or shutil.which('nodejs')
            if not node_exe:
                raise ExecutionError("未找到 Node.js 可执行文件")
            
            # 构建命令
            if task.script:
                cmd = [node_exe, str(task.script)]
            else:
                # 创建临时脚本
                temp_script = self.create_temp_script(script_content, 'node')
                cmd = [node_exe, str(temp_script)]
            
            # 准备环境
            env = os.environ.copy()
            env.update(task.context.env)
            env['SKILL_ID'] = task.skill_id
            env['SKILL_TASK_ID'] = task.id
            env['SKILL_PARAMS'] = json.dumps(task.parameters, ensure_ascii=False)
            
            # 执行
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=task.context.cwd,
                env=env,
            )
            
            try:
                stdout, stderr = process.communicate(timeout=task.context.timeout)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                # 清理临时文件
                if not task.script and 'temp_script' in locals():
                    try:
                        temp_script.unlink()
                    except:
                        pass
                
                status = ExecutionStatus.COMPLETED if process.returncode == 0 else ExecutionStatus.FAILED
                
                return ExecutionResult(
                    status=status,
                    exit_code=process.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    duration=duration,
                    start_time=start_time,
                    end_time=end_time,
                )
                
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    exit_code=-1,
                    duration=duration,
                    start_time=start_time,
                    end_time=end_time,
                )
                
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                exit_code=-1,
                stderr=str(e),
                duration=duration,
                start_time=start_time,
                end_time=end_time,
            )
    
    async def execute_async(self, task: ExecutionTask) -> ExecutionResult:
        """异步执行 Node.js 脚本"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, task)


class SkillExecutor:
    """
    技能执行引擎
    
    管理技能执行，支持多种解释器，提供并行执行能力。
    """
    
    def __init__(self, max_workers: int = 4):
        """
        初始化执行引擎
        
        参数:
            max_workers: 最大并行工作线程数
        """
        self.max_workers = max_workers
        self.executors: Dict[str, ScriptExecutor] = {
            'bash': BashExecutor(),
            'sh': BashExecutor(),
            'python': PythonExecutor(),
            'python3': PythonExecutor(),
            'node': NodeExecutor(),
            'nodejs': NodeExecutor(),
        }
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self._running_tasks: Dict[str, Future] = {}
        self._lock = threading.Lock()
    
    def register_executor(self, interpreter: str, executor: ScriptExecutor) -> None:
        """
        注册执行器
        
        参数:
            interpreter: 解释器名称
            executor: 执行器实例
        """
        self.executors[interpreter] = executor
        logger.info(f"注册执行器: {interpreter}")
    
    def execute(self, task: ExecutionTask) -> ExecutionResult:
        """
        执行任务
        
        参数:
            task: 执行任务
            
        返回:
            执行结果
        """
        executor = self._get_executor(task.interpreter)
        
        with self._lock:
            self._running_tasks[task.id] = None  # 标记为运行中
        
        try:
            result = executor.execute(task)
            return result
        finally:
            with self._lock:
                self._running_tasks.pop(task.id, None)
    
    async def execute_async(self, task: ExecutionTask) -> ExecutionResult:
        """
        异步执行任务
        
        参数:
            task: 执行任务
            
        返回:
            执行结果
        """
        executor = self._get_executor(task.interpreter)
        return await executor.execute_async(task)
    
    def execute_parallel(self, tasks: List[ExecutionTask],
                         callback: Optional[Callable[[ExecutionTask, ExecutionResult], None]] = None
                         ) -> List[ExecutionResult]:
        """
        并行执行多个任务
        
        参数:
            tasks: 任务列表
            callback: 回调函数，每个任务完成时调用
            
        返回:
            执行结果列表（按任务顺序）
        """
        results: Dict[str, ExecutionResult] = {}
        
        def execute_task(task: ExecutionTask) -> Tuple[str, ExecutionResult]:
            result = self.execute(task)
            if callback:
                callback(task, result)
            return task.id, result
        
        # 提交所有任务
        futures = {
            self.thread_pool.submit(execute_task, task): task
            for task in tasks
        }
        
        # 收集结果
        for future in as_completed(futures):
            task_id, result = future.result()
            results[task_id] = result
        
        # 按任务顺序返回结果
        return [results[task.id] for task in tasks]
    
    async def execute_parallel_async(self, tasks: List[ExecutionTask]) -> List[ExecutionResult]:
        """
        异步并行执行多个任务
        
        参数:
            tasks: 任务列表
            
        返回:
            执行结果列表
        """
        coroutines = [self.execute_async(task) for task in tasks]
        return await asyncio.gather(*coroutines)
    
    def cancel_task(self, task_id: str) -> bool:
        """
        取消任务
        
        参数:
            task_id: 任务 ID
            
        返回:
            是否成功取消
        """
        with self._lock:
            future = self._running_tasks.get(task_id)
            if future and not future.done():
                future.cancel()
                return True
        return False
    
    def cancel_all(self) -> int:
        """
        取消所有运行中的任务
        
        返回:
            取消的任务数量
        """
        count = 0
        with self._lock:
            for task_id, future in list(self._running_tasks.items()):
                if future and not future.done():
                    future.cancel()
                    count += 1
        return count
    
    def shutdown(self, wait: bool = True) -> None:
        """
        关闭执行引擎
        
        参数:
            wait: 是否等待所有任务完成
        """
        self.cancel_all()
        self.thread_pool.shutdown(wait=wait)
        logger.info("执行引擎已关闭")
    
    def _get_executor(self, interpreter: str) -> ScriptExecutor:
        """
        获取执行器
        
        参数:
            interpreter: 解释器名称
            
        返回:
            执行器实例
            
        抛出:
            ExecutionError: 找不到执行器
        """
        if interpreter not in self.executors:
            raise ExecutionError(f"不支持的解释器: {interpreter}")
        return self.executors[interpreter]
    
    def get_running_tasks(self) -> List[str]:
        """
        获取运行中的任务 ID 列表
        
        返回:
            任务 ID 列表
        """
        with self._lock:
            return list(self._running_tasks.keys())


class ExecutionMonitor:
    """执行监控器"""
    
    def __init__(self):
        self.history: List[ExecutionResult] = []
        self._lock = threading.Lock()
    
    def record(self, result: ExecutionResult) -> None:
        """
        记录执行结果
        
        参数:
            result: 执行结果
        """
        with self._lock:
            self.history.append(result)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取执行统计信息
        
        返回:
            统计信息字典
        """
        with self._lock:
            total = len(self.history)
            if total == 0:
                return {'total': 0}
            
            successful = sum(1 for r in self.history if r.success)
            failed = sum(1 for r in self.history if r.failed)
            timeouts = sum(1 for r in self.history if r.status == ExecutionStatus.TIMEOUT)
            
            durations = [r.duration for r in self.history if r.duration > 0]
            avg_duration = sum(durations) / len(durations) if durations else 0
            max_duration = max(durations) if durations else 0
            min_duration = min(durations) if durations else 0
            
            return {
                'total': total,
                'successful': successful,
                'failed': failed,
                'timeouts': timeouts,
                'success_rate': successful / total if total > 0 else 0,
                'avg_duration': avg_duration,
                'max_duration': max_duration,
                'min_duration': min_duration,
            }
    
    def clear_history(self) -> None:
        """清空历史记录"""
        with self._lock:
            self.history.clear()


class ExecutionQueue:
    """执行队列"""
    
    def __init__(self, executor: SkillExecutor, max_queue_size: int = 100):
        """
        初始化执行队列
        
        参数:
            executor: 执行引擎
            max_queue_size: 最大队列大小
        """
        self.executor = executor
        self.max_queue_size = max_queue_size
        self._queue: asyncio.Queue[ExecutionTask] = asyncio.Queue(maxsize=max_queue_size)
        self._results: Dict[str, ExecutionResult] = {}
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """启动队列处理器"""
        self._running = True
        self._worker_task = asyncio.create_task(self._process_queue())
        logger.info("执行队列已启动")
    
    async def stop(self) -> None:
        """停止队列处理器"""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("执行队列已停止")
    
    async def enqueue(self, task: ExecutionTask) -> str:
        """
        将任务加入队列
        
        参数:
            task: 执行任务
            
        返回:
            任务 ID
        """
        await self._queue.put(task)
        return task.id
    
    async def get_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[ExecutionResult]:
        """
        获取任务结果
        
        参数:
            task_id: 任务 ID
            timeout: 超时时间
            
        返回:
            执行结果，如果尚未完成则返回 None
        """
        start = time.time()
        while task_id not in self._results:
            if timeout and (time.time() - start) > timeout:
                return None
            await asyncio.sleep(0.1)
        return self._results.get(task_id)
    
    async def _process_queue(self) -> None:
        """处理队列中的任务"""
        while self._running:
            try:
                task = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                result = await self.executor.execute_async(task)
                self._results[task.id] = result
                self._queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"队列处理错误: {e}")


# 便捷函数
def execute_bash(command: str, **kwargs) -> ExecutionResult:
    """
    便捷函数：执行 Bash 命令
    
    参数:
        command: Bash 命令
        **kwargs: 额外参数
        
    返回:
        执行结果
    """
    executor = SkillExecutor()
    task = ExecutionTask(
        id=f"bash_{int(time.time())}",
        skill_id="bash",
        command=command,
        interpreter='bash',
        context=ExecutionContext(**kwargs),
    )
    return executor.execute(task)


def execute_python(script: str, **kwargs) -> ExecutionResult:
    """
    便捷函数：执行 Python 脚本
    
    参数:
        script: Python 脚本内容
        **kwargs: 额外参数
        
    返回:
        执行结果
    """
    executor = SkillExecutor()
    task = ExecutionTask(
        id=f"python_{int(time.time())}",
        skill_id="python",
        command=script,
        interpreter='python',
        context=ExecutionContext(**kwargs),
    )
    return executor.execute(task)


def execute_node(script: str, **kwargs) -> ExecutionResult:
    """
    便捷函数：执行 Node.js 脚本
    
    参数:
        script: Node.js 脚本内容
        **kwargs: 额外参数
        
    返回:
        执行结果
    """
    executor = SkillExecutor()
    task = ExecutionTask(
        id=f"node_{int(time.time())}",
        skill_id="node",
        command=script,
        interpreter='node',
        context=ExecutionContext(**kwargs),
    )
    return executor.execute(task)


# 单元测试存根
class TestSkillExecutor:
    """SkillExecutor 单元测试"""
    
    def test_execute_bash_success(self) -> None:
        """测试成功执行 Bash 命令"""
        executor = SkillExecutor()
        task = ExecutionTask(
            id="test_1",
            skill_id="test",
            command="echo 'Hello World'",
            interpreter='bash',
        )
        result = executor.execute(task)
        
        assert result.success
        assert "Hello World" in result.stdout
    
    def test_execute_bash_failure(self) -> None:
        """测试失败的 Bash 命令"""
        executor = SkillExecutor()
        task = ExecutionTask(
            id="test_2",
            skill_id="test",
            command="exit 1",
            interpreter='bash',
        )
        result = executor.execute(task)
        
        assert not result.success
        assert result.exit_code == 1
    
    def test_execute_bash_timeout(self) -> None:
        """测试 Bash 命令超时"""
        executor = SkillExecutor()
        task = ExecutionTask(
            id="test_3",
            skill_id="test",
            command="sleep 10",
            interpreter='bash',
            context=ExecutionContext(timeout=1),
        )
        result = executor.execute(task)
        
        assert result.status == ExecutionStatus.TIMEOUT
    
    def test_execute_python_success(self) -> None:
        """测试成功执行 Python 脚本"""
        executor = SkillExecutor()
        task = ExecutionTask(
            id="test_4",
            skill_id="test",
            command="print('Hello from Python')",
            interpreter='python',
        )
        result = executor.execute(task)
        
        assert result.success
        assert "Hello from Python" in result.stdout
    
    def test_execute_parallel(self) -> None:
        """测试并行执行"""
        executor = SkillExecutor(max_workers=4)
        tasks = [
            ExecutionTask(
                id=f"parallel_{i}",
                skill_id="test",
                command=f"echo 'Task {i}'",
                interpreter='bash',
            )
            for i in range(5)
        ]
        
        results = executor.execute_parallel(tasks)
        
        assert len(results) == 5
        assert all(r.success for r in results)
    
    def test_execution_context(self) -> None:
        """测试执行上下文"""
        ctx = ExecutionContext(
            working_dir="/tmp",
            env={'TEST_VAR': 'test_value'},
            timeout=60,
        )
        
        assert ctx.cwd == Path("/tmp").resolve()
        assert ctx.env['TEST_VAR'] == 'test_value'
        assert ctx.timeout == 60
        
        # 测试合并
        new_ctx = ctx.merge_env({'ANOTHER_VAR': 'another'})
        assert new_ctx.env['TEST_VAR'] == 'test_value'
        assert new_ctx.env['ANOTHER_VAR'] == 'another'
    
    def test_execution_result(self) -> None:
        """测试执行结果"""
        result = ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            exit_code=0,
            stdout="output",
            stderr="",
            duration=1.5,
        )
        
        assert result.success
        assert not result.failed
        assert result.to_dict()['success'] is True


# 全局执行引擎实例
_default_executor: Optional[SkillExecutor] = None


def get_default_executor() -> SkillExecutor:
    """获取默认执行引擎实例"""
    global _default_executor
    if _default_executor is None:
        _default_executor = SkillExecutor()
    return _default_executor
