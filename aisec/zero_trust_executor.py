"""
零信任执行器 - 零信任安全执行框架
"""
import time
import hashlib
import threading
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .command_interceptor import CommandInterceptor, InterceptResult, CommandAction


class TrustLevel(Enum):
    """信任等级"""
    UNTRUSTED = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    FULL = 4


class ExecutionMode(Enum):
    """执行模式"""
    DIRECT = "direct"
    SANDBOX = "sandbox"
    CONTAINER = "container"
    ISOLATED = "isolated"


@dataclass
class ExecutionContext:
    """执行上下文"""
    context_id: str
    trust_level: TrustLevel
    user_id: str = ""
    source_ip: str = ""
    process_id: int = 0
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionRequest:
    """执行请求"""
    request_id: str
    command: str
    context: ExecutionContext
    requested_mode: ExecutionMode = ExecutionMode.SANDBOX
    timeout: int = 60
    resources: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """执行结果"""
    success: bool
    request_id: str
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    mode_used: ExecutionMode
    security_checks: List[str]
    violations: List[str] = field(default_factory=list)


class ZeroTrustExecutor:
    """零信任执行器"""
    
    def __init__(self):
        self._command_interceptor = CommandInterceptor()
        self._trust_levels: Dict[str, TrustLevel] = {}
        self._execution_history: List[ExecutionResult] = []
        self._policies: List[Callable[[ExecutionRequest], Tuple[bool, str]]] = []
        self._lock = threading.Lock()
        self._default_timeout = 60
        self._max_concurrent = 10
        self._active_executions = 0
    
    def set_trust_level(self, entity: str, level: TrustLevel) -> None:
        """设置信任等级"""
        self._trust_levels[entity] = level
    
    def get_trust_level(self, entity: str) -> TrustLevel:
        """获取信任等级"""
        return self._trust_levels.get(entity, TrustLevel.UNTRUSTED)
    
    def add_policy(
        self,
        policy: Callable[[ExecutionRequest], Tuple[bool, str]]
    ) -> None:
        """添加执行策略"""
        self._policies.append(policy)
    
    def create_context(
        self,
        user_id: str = "",
        source_ip: str = "",
        trust_level: TrustLevel = TrustLevel.UNTRUSTED
    ) -> ExecutionContext:
        """创建执行上下文"""
        context_id = hashlib.md5(
            f"{user_id}{source_ip}{time.time()}".encode()
        ).hexdigest()[:12]
        
        return ExecutionContext(
            context_id=context_id,
            trust_level=trust_level,
            user_id=user_id,
            source_ip=source_ip
        )
    
    def create_request(
        self,
        command: str,
        context: ExecutionContext,
        mode: ExecutionMode = ExecutionMode.SANDBOX,
        timeout: int = None
    ) -> ExecutionRequest:
        """创建执行请求"""
        request_id = hashlib.md5(
            f"{command}{context.context_id}{time.time()}".encode()
        ).hexdigest()[:12]
        
        return ExecutionRequest(
            request_id=request_id,
            command=command,
            context=context,
            requested_mode=mode,
            timeout=timeout or self._default_timeout
        )
    
    def validate_request(
        self,
        request: ExecutionRequest
    ) -> Tuple[bool, List[str]]:
        """验证请求"""
        violations = []
        
        # 检查并发限制
        if self._active_executions >= self._max_concurrent:
            violations.append("超过最大并发执行数")
        
        # 检查命令
        intercept_result = self._command_interceptor.intercept(request.command)
        if intercept_result.action == CommandAction.BLOCK:
            violations.append(f"命令被阻止: {intercept_result.reason}")
        
        # 检查信任等级
        if request.context.trust_level == TrustLevel.UNTRUSTED:
            if request.requested_mode == ExecutionMode.DIRECT:
                violations.append("不信任的上下文不能直接执行")
        
        # 应用自定义策略
        for policy in self._policies:
            try:
                allowed, reason = policy(request)
                if not allowed:
                    violations.append(f"策略拒绝: {reason}")
            except Exception as e:
                violations.append(f"策略检查失败: {e}")
        
        return len(violations) == 0, violations
    
    def determine_execution_mode(
        self,
        request: ExecutionRequest
    ) -> ExecutionMode:
        """确定执行模式"""
        trust = request.context.trust_level
        requested = request.requested_mode
        
        # 根据信任等级调整执行模式
        if trust == TrustLevel.FULL:
            return requested
        
        if trust == TrustLevel.HIGH:
            if requested in [ExecutionMode.DIRECT, ExecutionMode.SANDBOX]:
                return requested
            return ExecutionMode.SANDBOX
        
        if trust == TrustLevel.MEDIUM:
            if requested == ExecutionMode.DIRECT:
                return ExecutionMode.SANDBOX
            return requested
        
        if trust == TrustLevel.LOW:
            return ExecutionMode.CONTAINER
        
        # UNTRUSTED
        return ExecutionMode.ISOLATED
    
    def execute(
        self,
        request: ExecutionRequest
    ) -> ExecutionResult:
        """执行请求"""
        start_time = time.time()
        security_checks = []
        violations = []
        
        # 验证请求
        is_valid, validation_violations = self.validate_request(request)
        violations.extend(validation_violations)
        
        if not is_valid:
            return ExecutionResult(
                success=False,
                request_id=request.request_id,
                exit_code=-1,
                stdout="",
                stderr="请求验证失败: " + "; ".join(violations),
                execution_time=time.time() - start_time,
                mode_used=ExecutionMode.ISOLATED,
                security_checks=security_checks,
                violations=violations
            )
        
        # 确定执行模式
        mode = self.determine_execution_mode(request)
        security_checks.append(f"执行模式: {mode.value}")
        
        # 拦截命令
        intercept_result = self._command_interceptor.intercept(request.command)
        security_checks.append(f"命令拦截: {intercept_result.action.value}")
        
        if intercept_result.action == CommandAction.BLOCK:
            return ExecutionResult(
                success=False,
                request_id=request.request_id,
                exit_code=-1,
                stdout="",
                stderr=f"命令被阻止: {intercept_result.reason}",
                execution_time=time.time() - start_time,
                mode_used=mode,
                security_checks=security_checks,
                violations=[intercept_result.reason]
            )
        
        # 获取最终命令
        final_command = intercept_result.final_command or request.command
        
        # 执行命令（根据模式）
        try:
            with self._lock:
                self._active_executions += 1
            
            stdout, stderr, exit_code = self._execute_internal(
                final_command,
                mode,
                request.timeout
            )
            
            success = exit_code == 0
            
        except Exception as e:
            stdout = ""
            stderr = str(e)
            exit_code = -1
            success = False
        
        finally:
            with self._lock:
                self._active_executions -= 1
        
        result = ExecutionResult(
            success=success,
            request_id=request.request_id,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            execution_time=time.time() - start_time,
            mode_used=mode,
            security_checks=security_checks,
            violations=violations
        )
        
        self._execution_history.append(result)
        return result
    
    def _execute_internal(
        self,
        command: str,
        mode: ExecutionMode,
        timeout: int
    ) -> Tuple[str, str, int]:
        """内部执行"""
        import subprocess
        
        if mode == ExecutionMode.DIRECT:
            # 直接执行
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.stdout, result.stderr, result.returncode
        
        elif mode == ExecutionMode.SANDBOX:
            # 沙箱执行（使用受限环境）
            env = {
                "PATH": "/usr/bin:/bin",
                "HOME": "/tmp",
            }
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd="/tmp"
            )
            return result.stdout, result.stderr, result.returncode
        
        elif mode == ExecutionMode.CONTAINER:
            # 容器执行（模拟）
            return self._execute_in_container(command, timeout)
        
        else:  # ISOLATED
            # 隔离执行（模拟）
            return self._execute_isolated(command, timeout)
    
    def _execute_in_container(
        self,
        command: str,
        timeout: int
    ) -> Tuple[str, str, int]:
        """容器执行"""
        # 模拟容器执行
        try:
            import subprocess
            result = subprocess.run(
                ["docker", "run", "--rm", "alpine", "sh", "-c", command],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.stdout, result.stderr, result.returncode
        except Exception as e:
            return "", f"容器执行失败: {e}", -1
    
    def _execute_isolated(
        self,
        command: str,
        timeout: int
    ) -> Tuple[str, str, int]:
        """隔离执行"""
        # 完全隔离执行（模拟）
        return "", "隔离模式：命令未执行", 0
    
    def execute_command(
        self,
        command: str,
        user_id: str = "",
        trust_level: TrustLevel = TrustLevel.UNTRUSTED
    ) -> ExecutionResult:
        """便捷执行方法"""
        context = self.create_context(
            user_id=user_id,
            trust_level=trust_level
        )
        request = self.create_request(command, context)
        return self.execute(request)
    
    def get_execution_history(
        self,
        limit: int = 100
    ) -> List[ExecutionResult]:
        """获取执行历史"""
        return self._execution_history[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self._execution_history:
            return {
                "total_executions": 0,
                "successful": 0,
                "failed": 0,
                "active": self._active_executions
            }
        
        successful = sum(1 for r in self._execution_history if r.success)
        
        return {
            "total_executions": len(self._execution_history),
            "successful": successful,
            "failed": len(self._execution_history) - successful,
            "active": self._active_executions,
            "trust_levels": {
                level.value: sum(
                    1 for entity, l in self._trust_levels.items()
                    if l == level
                )
                for level in TrustLevel
            }
        }
    
    @property
    def interceptor(self) -> CommandInterceptor:
        """获取命令拦截器"""
        return self._command_interceptor
