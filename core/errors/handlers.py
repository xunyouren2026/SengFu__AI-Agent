"""
全局异常处理器

提供统一的异常捕获、转换和响应生成机制。
"""

import logging
import traceback
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Type

from .definitions import (
    AGIError,
    ErrorCode,
    ErrorSeverity,
    NetworkError,
    SecurityError,
    get_error_category,
    get_error_metadata,
)

logger = logging.getLogger(__name__)


class ErrorContext:
    """
    错误上下文信息

    捕获异常发生时的完整上下文，包括请求信息、堆栈跟踪等。

    Attributes:
        request_id: 请求唯一标识
        timestamp: 错误发生时间
        stack_trace: 异常堆栈跟踪
        additional_info: 附加上下文信息
        error_type: 异常类型名称
        error_module: 异常所在模块
    """

    def __init__(
        self,
        request_id: Optional[str] = None,
        additional_info: Optional[Dict[str, Any]] = None,
    ):
        self.request_id = request_id or self._generate_request_id()
        self.timestamp = datetime.now(timezone.utc)
        self.stack_trace: str = ""
        self.additional_info = additional_info or {}
        self.error_type: str = ""
        self.error_module: str = ""

    @staticmethod
    def _generate_request_id() -> str:
        """生成唯一的请求ID"""
        return str(uuid.uuid4())

    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        request_id: Optional[str] = None,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> "ErrorContext":
        """
        从异常对象创建错误上下文

        Args:
            exc: 异常对象
            request_id: 请求ID
            additional_info: 附加信息

        Returns:
            填充了异常信息的ErrorContext实例
        """
        ctx = cls(request_id=request_id, additional_info=additional_info)
        ctx.stack_trace = traceback.format_exc()
        ctx.error_type = type(exc).__name__
        ctx.error_module = type(exc).__module__
        return ctx

    def to_dict(self) -> Dict[str, Any]:
        """将上下文转换为字典"""
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "error_type": self.error_type,
            "error_module": self.error_module,
            "stack_trace": self.stack_trace,
            "additional_info": self.additional_info,
        }


class ErrorResponse:
    """
    标准错误响应

    统一的错误响应格式，用于API返回和日志记录。

    Attributes:
        code: 错误码
        message: 错误消息
        details: 详细信息
        request_id: 请求ID
        timestamp: 时间戳
        severity: 严重级别
        category: 错误分类
        http_status: HTTP状态码
    """

    def __init__(
        self,
        code: int,
        message: str,
        request_id: str,
        timestamp: Optional[datetime] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: str = ErrorSeverity.ERROR,
        http_status: int = 500,
    ):
        self.code = code
        self.message = message
        self.details = details or {}
        self.request_id = request_id
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.severity = severity
        self.http_status = http_status
        self.category = get_error_category(code)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result: Dict[str, Any] = {
            "error": {
                "code": self.code,
                "message": self.message,
                "category": self.category,
                "severity": self.severity,
            },
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.details:
            result["error"]["details"] = self.details
        return result

    @classmethod
    def from_agi_error(
        cls, error: AGIError, context: Optional[ErrorContext] = None
    ) -> "ErrorResponse":
        """
        从AGIError创建错误响应

        Args:
            error: AGIError异常实例
            context: 错误上下文

        Returns:
            ErrorResponse实例
        """
        return cls(
            code=int(error.code),
            message=error.message,
            request_id=context.request_id if context else "",
            timestamp=context.timestamp if context else None,
            details=error.details,
            severity=error.severity,
            http_status=error.http_status,
        )

    def __repr__(self) -> str:
        return (
            f"ErrorResponse(code={self.code}, message={self.message!r}, "
            f"severity={self.severity})"
        )


# 自定义异常处理器类型
CustomHandler = Callable[[Exception, ErrorContext], Optional[ErrorResponse]]


class ErrorHandler:
    """
    全局异常处理器

    统一捕获、分类和处理框架中的所有异常，生成标准化的错误响应。

    Usage:
        handler = ErrorHandler()
        handler.register_handler(ValueError, my_value_handler)

        try:
            risky_operation()
        except Exception as e:
            response = handler.handle_exception(e)
    """

    def __init__(self, include_stack_trace: bool = False, log_errors: bool = True):
        """
        初始化异常处理器

        Args:
            include_stack_trace: 是否在响应中包含堆栈跟踪
            log_errors: 是否自动记录错误日志
        """
        self._handlers: OrderedDict[Type[Exception], CustomHandler] = OrderedDict()
        self._include_stack_trace = include_stack_trace
        self._log_errors = log_errors
        # 注册默认处理器
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """注册默认的异常处理器"""
        self.register_handler(AGIError, self._handle_agi_error)
        self.register_handler(KeyboardInterrupt, self._handle_interrupt)
        self.register_handler(SystemExit, self._handle_system_exit)

    def register_handler(
        self, exc_type: Type[Exception], handler: CustomHandler
    ) -> None:
        """
        注册自定义异常处理器

        Args:
            exc_type: 异常类型
            handler: 处理函数，接收 (exception, context) 参数，
                     返回 ErrorResponse 或 None
        """
        if not (isinstance(exc_type, type) and issubclass(exc_type, BaseException)):
            raise TypeError(
                f"exc_type 必须是 BaseException 的子类， got {exc_type}"
            )
        if not callable(handler):
            raise TypeError(f"handler 必须是可调用对象， got {handler}")
        self._handlers[exc_type] = handler
        logger.debug(f"已注册异常处理器: {exc_type.__name__}")

    def unregister_handler(self, exc_type: Type[Exception]) -> bool:
        """
        取消注册异常处理器

        Args:
            exc_type: 要取消的异常类型

        Returns:
            是否成功取消
        """
        if exc_type in self._handlers:
            del self._handlers[exc_type]
            logger.debug(f"已取消注册异常处理器: {exc_type.__name__}")
            return True
        return False

    def handle_exception(
        self,
        exc: Exception,
        request_id: Optional[str] = None,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> ErrorResponse:
        """
        捕获异常并转换为标准错误响应

        按照注册顺序查找匹配的处理器，找到第一个匹配的处理器后使用它
        来生成错误响应。如果没有匹配的处理器，使用默认处理逻辑。

        Args:
            exc: 捕获的异常
            request_id: 请求ID
            additional_info: 附加上下文信息

        Returns:
            标准化的ErrorResponse
        """
        context = ErrorContext.from_exception(
            exc, request_id=request_id, additional_info=additional_info
        )

        # 按注册顺序查找匹配的处理器
        for registered_type, handler in self._handlers.items():
            if isinstance(exc, registered_type):
                response = handler(exc, context)
                if response is not None:
                    self._finalize_response(response, context)
                    return response

        # 没有匹配的处理器，使用默认处理
        response = self._handle_unknown_error(exc, context)
        self._finalize_response(response, context)
        return response

    def _finalize_response(
        self, response: ErrorResponse, context: ErrorContext
    ) -> None:
        """
        最终化错误响应

        根据配置决定是否包含堆栈跟踪，并记录日志。

        Args:
            response: 错误响应
            context: 错误上下文
        """
        if self._include_stack_trace and context.stack_trace:
            response.details["stack_trace"] = context.stack_trace

        if self._log_errors:
            self._log_error(response, context)

    def _log_error(self, response: ErrorResponse, context: ErrorContext) -> None:
        """根据严重级别记录错误日志"""
        log_data = {
            "request_id": context.request_id,
            "error_code": response.code,
            "error_type": context.error_type,
            "error_msg": response.message,
        }

        if response.severity == ErrorSeverity.CRITICAL:
            logger.critical("严重错误: %s", response.message, extra=log_data)
        elif response.severity == ErrorSeverity.WARNING:
            logger.warning("警告: %s", response.message, extra=log_data)
        else:
            logger.error("错误: %s", response.message, extra=log_data)

    def _handle_agi_error(
        self, exc: Exception, context: ErrorContext
    ) -> Optional[ErrorResponse]:
        """处理AGI框架异常"""
        if isinstance(exc, AGIError):
            return ErrorResponse.from_agi_error(exc, context)
        return None

    def _handle_interrupt(
        self, exc: Exception, context: ErrorContext
    ) -> Optional[ErrorResponse]:
        """处理键盘中断"""
        return ErrorResponse(
            code=ErrorCode.WORKFLOW_EXECUTION_FAILED,
            message="操作被用户中断",
            request_id=context.request_id,
            timestamp=context.timestamp,
            severity=ErrorSeverity.WARNING,
            http_status=499,
        )

    def _handle_system_exit(
        self, exc: Exception, context: ErrorContext
    ) -> Optional[ErrorResponse]:
        """处理系统退出"""
        return ErrorResponse(
            code=ErrorCode.WORKFLOW_EXECUTION_FAILED,
            message="系统正在退出",
            request_id=context.request_id,
            timestamp=context.timestamp,
            severity=ErrorSeverity.WARNING,
            http_status=499,
        )

    def _handle_unknown_error(
        self, exc: Exception, context: ErrorContext
    ) -> ErrorResponse:
        """
        处理未知异常

        将非AGIError异常转换为标准错误响应。

        Args:
            exc: 未知异常
            context: 错误上下文

        Returns:
            标准化的ErrorResponse
        """
        error_type = type(exc).__name__
        error_msg = str(exc) if str(exc) else "发生未知错误"

        # 根据异常类型推断分类
        if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
            code = ErrorCode.NETWORK_CONNECTION_FAILED
            severity = ErrorSeverity.ERROR
            http_status = 503
        elif isinstance(exc, (ValueError, TypeError)):
            code = ErrorCode.CONFIG_INVALID_VALUE
            severity = ErrorSeverity.WARNING
            http_status = 400
        elif isinstance(exc, (PermissionError,)):
            code = ErrorCode.SECURITY_PERMISSION_DENIED
            severity = ErrorSeverity.ERROR
            http_status = 403
        elif isinstance(exc, (FileNotFoundError,)):
            code = ErrorCode.CONFIG_FILE_NOT_FOUND
            severity = ErrorSeverity.ERROR
            http_status = 404
        elif isinstance(exc, (MemoryError,)):
            code = ErrorCode.STORAGE_QUOTA_EXCEEDED
            severity = ErrorSeverity.CRITICAL
            http_status = 507
        else:
            code = ErrorCode.WORKFLOW_EXECUTION_FAILED
            severity = ErrorSeverity.ERROR
            http_status = 500

        metadata = get_error_metadata(code)

        return ErrorResponse(
            code=int(code),
            message=f"{metadata['message']}: {error_msg}",
            request_id=context.request_id,
            timestamp=context.timestamp,
            details={
                "exception_type": error_type,
                "exception_module": context.error_module,
            },
            severity=severity,
            http_status=http_status,
        )

    def get_registered_handlers(self) -> List[str]:
        """获取所有已注册的异常处理器类型名称"""
        return [t.__name__ for t in self._handlers.keys()]

    def clear_handlers(self) -> None:
        """清除所有自定义处理器，恢复默认"""
        self._handlers.clear()
        self._register_default_handlers()
