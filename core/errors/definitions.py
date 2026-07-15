"""
错误码定义与异常类层次结构

提供全局错误码枚举和结构化异常类，覆盖框架中所有可能的错误场景。
"""

from enum import IntEnum
from typing import Any, Dict, Optional


class ErrorSeverity:
    """错误严重级别常量"""

    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    _ALL = {WARNING, ERROR, CRITICAL}

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """检查严重级别是否有效"""
        return value in cls._ALL


class ErrorCode(IntEnum):
    """
    全局错误码枚举

    错误码分类:
    - 1xxx: 配置错误 (CONFIG_ERROR)
    - 2xxx: 网络错误 (NETWORK_ERROR)
    - 3xxx: 模型错误 (MODEL_ERROR)
    - 4xxx: 工具错误 (TOOL_ERROR)
    - 5xxx: 代理错误 (AGENT_ERROR)
    - 6xxx: 安全错误 (SECURITY_ERROR)
    - 7xxx: 存储错误 (STORAGE_ERROR)
    - 8xxx: 工作流错误 (WORKFLOW_ERROR)
    """

    # ==================== 配置错误 (1xxx) ====================
    CONFIG_LOAD_FAILED = 1001
    CONFIG_PARSE_ERROR = 1002
    CONFIG_MISSING_KEY = 1003
    CONFIG_INVALID_VALUE = 1004
    CONFIG_SCHEMA_MISMATCH = 1005
    CONFIG_FILE_NOT_FOUND = 1006
    CONFIG_ENV_MISSING = 1007
    CONFIG_TYPE_ERROR = 1008
    CONFIG_VALIDATION_FAILED = 1009
    CONFIG_DEFAULT_MISSING = 1010

    # ==================== 网络错误 (2xxx) ====================
    NETWORK_CONNECTION_FAILED = 2001
    NETWORK_TIMEOUT = 2002
    NETWORK_DNS_FAILED = 2003
    NETWORK_SSL_ERROR = 2004
    NETWORK_RATE_LIMITED = 2005
    NETWORK_CONNECTION_REFUSED = 2006
    NETWORK_CONNECTION_RESET = 2007
    NETWORK_TOO_MANY_REDIRECTS = 2008
    NETWORK_CHUNKED_ENCODING_ERROR = 2009
    NETWORK_PROXY_ERROR = 2010

    # ==================== 模型错误 (3xxx) ====================
    MODEL_NOT_FOUND = 3001
    MODEL_LOAD_FAILED = 3002
    MODEL_INFERENCE_ERROR = 3003
    MODEL_INPUT_INVALID = 3004
    MODEL_OUTPUT_PARSE_ERROR = 3005
    MODEL_TIMEOUT = 3006
    MODEL_CONTEXT_TOO_LONG = 3007
    MODEL_TOKEN_LIMIT_EXCEEDED = 3008
    MODEL_UNSUPPORTED_OPERATION = 3009
    MODEL_WEIGHTS_CORRUPTED = 3010
    MODEL_QUANTIZATION_ERROR = 3011
    MODEL_COMPATIBILITY_ERROR = 3012

    # ==================== 工具错误 (4xxx) ====================
    TOOL_NOT_FOUND = 4001
    TOOL_EXECUTION_FAILED = 4002
    TOOL_INPUT_INVALID = 4003
    TOOL_OUTPUT_INVALID = 4004
    TOOL_TIMEOUT = 4005
    TOOL_PERMISSION_DENIED = 4006
    TOOL_NOT_REGISTERED = 4007
    TOOL_DEPENDENCY_MISSING = 4008
    TOOL_SIGNATURE_MISMATCH = 4009
    TOOL_SANDBOX_VIOLATION = 4010

    # ==================== 代理错误 (5xxx) ====================
    AGENT_INIT_FAILED = 5001
    AGENT_EXECUTION_FAILED = 5002
    AGENT_STATE_CORRUPTED = 5003
    AGENT_COMMUNICATION_ERROR = 5004
    AGENT_PLAN_INVALID = 5005
    AGENT_MEMORY_ERROR = 5006
    AGENT_CONTEXT_OVERFLOW = 5007
    AGENT_LOOP_DETECTED = 5008
    AGENT_TASK_TIMEOUT = 5009
    AGENT_DELEGATION_FAILED = 5010
    AGENT_REASONING_FAILED = 5011

    # ==================== 安全错误 (6xxx) ====================
    SECURITY_AUTH_FAILED = 6001
    SECURITY_PERMISSION_DENIED = 6002
    SECURITY_TOKEN_EXPIRED = 6003
    SECURITY_TOKEN_INVALID = 6004
    SECURITY_INPUT_INJECTION = 6005
    SECURITY_OUTPUT_FILTERED = 6006
    SECURITY_RATE_LIMIT_EXCEEDED = 6007
    SECURITY_ENCRYPTION_ERROR = 6008
    SECURITY_SIGNATURE_INVALID = 6009
    SECURITY_SANDBOX_ESCAPE = 6010
    SECURITY_API_KEY_INVALID = 6011
    SECURITY_AUDIT_LOG_FAILED = 6012

    # ==================== 存储错误 (7xxx) ====================
    STORAGE_CONNECTION_FAILED = 7001
    STORAGE_READ_ERROR = 7002
    STORAGE_WRITE_ERROR = 7003
    STORAGE_DELETE_ERROR = 7004
    STORAGE_NOT_FOUND = 7005
    STORAGE_SERIALIZATION_ERROR = 7006
    STORAGE_DESERIALIZATION_ERROR = 7007
    STORAGE_QUOTA_EXCEEDED = 7008
    STORAGE_CORRUPTION = 7009
    STORAGE_LOCK_TIMEOUT = 7010
    STORAGE_MIGRATION_ERROR = 7011

    # ==================== 工作流错误 (8xxx) ====================
    WORKFLOW_NOT_FOUND = 8001
    WORKFLOW_EXECUTION_FAILED = 8002
    WORKFLOW_STEP_FAILED = 8003
    WORKFLOW_INVALID_STATE = 8004
    WORKFLOW_CYCLE_DETECTED = 8005
    WORKFLOW_TIMEOUT = 8006
    WORKFLOW_INPUT_INVALID = 8007
    WORKFLOW_PARALLEL_ERROR = 8008
    WORKFLOW_ROLLBACK_FAILED = 8009
    WORKFLOW_DEPENDENCY_ERROR = 8010
    WORKFLOW_BRANCH_ERROR = 8011
    WORKFLOW_COMPENSATION_FAILED = 8012


# 错误码元数据映射表: code -> (message, http_status, severity)
_ERROR_METADATA: Dict[int, Dict[str, Any]] = {
    # 配置错误
    ErrorCode.CONFIG_LOAD_FAILED: {
        "message": "配置加载失败",
        "http_status": 500,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.CONFIG_PARSE_ERROR: {
        "message": "配置解析错误",
        "http_status": 500,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.CONFIG_MISSING_KEY: {
        "message": "缺少必要的配置项",
        "http_status": 400,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.CONFIG_INVALID_VALUE: {
        "message": "配置值无效",
        "http_status": 400,
        "severity": ErrorSeverity.WARNING,
    },
    ErrorCode.CONFIG_SCHEMA_MISMATCH: {
        "message": "配置结构不匹配",
        "http_status": 400,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.CONFIG_FILE_NOT_FOUND: {
        "message": "配置文件未找到",
        "http_status": 404,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.CONFIG_ENV_MISSING: {
        "message": "缺少必要的环境变量",
        "http_status": 500,
        "severity": ErrorSeverity.CRITICAL,
    },
    ErrorCode.CONFIG_TYPE_ERROR: {
        "message": "配置类型错误",
        "http_status": 400,
        "severity": ErrorSeverity.WARNING,
    },
    ErrorCode.CONFIG_VALIDATION_FAILED: {
        "message": "配置验证失败",
        "http_status": 400,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.CONFIG_DEFAULT_MISSING: {
        "message": "缺少默认配置值",
        "http_status": 500,
        "severity": ErrorSeverity.WARNING,
    },

    # 网络错误
    ErrorCode.NETWORK_CONNECTION_FAILED: {
        "message": "网络连接失败",
        "http_status": 503,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.NETWORK_TIMEOUT: {
        "message": "网络请求超时",
        "http_status": 504,
        "severity": ErrorSeverity.WARNING,
    },
    ErrorCode.NETWORK_DNS_FAILED: {
        "message": "DNS解析失败",
        "http_status": 503,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.NETWORK_SSL_ERROR: {
        "message": "SSL/TLS错误",
        "http_status": 502,
        "severity": ErrorSeverity.CRITICAL,
    },
    ErrorCode.NETWORK_RATE_LIMITED: {
        "message": "请求频率超限",
        "http_status": 429,
        "severity": ErrorSeverity.WARNING,
    },
    ErrorCode.NETWORK_CONNECTION_REFUSED: {
        "message": "连接被拒绝",
        "http_status": 503,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.NETWORK_CONNECTION_RESET: {
        "message": "连接被重置",
        "http_status": 503,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.NETWORK_TOO_MANY_REDIRECTS: {
        "message": "重定向次数过多",
        "http_status": 502,
        "severity": ErrorSeverity.WARNING,
    },
    ErrorCode.NETWORK_CHUNKED_ENCODING_ERROR: {
        "message": "分块编码错误",
        "http_status": 502,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.NETWORK_PROXY_ERROR: {
        "message": "代理服务器错误",
        "http_status": 502,
        "severity": ErrorSeverity.ERROR,
    },

    # 模型错误
    ErrorCode.MODEL_NOT_FOUND: {
        "message": "模型未找到",
        "http_status": 404,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.MODEL_LOAD_FAILED: {
        "message": "模型加载失败",
        "http_status": 500,
        "severity": ErrorSeverity.CRITICAL,
    },
    ErrorCode.MODEL_INFERENCE_ERROR: {
        "message": "模型推理错误",
        "http_status": 500,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.MODEL_INPUT_INVALID: {
        "message": "模型输入无效",
        "http_status": 400,
        "severity": ErrorSeverity.WARNING,
    },
    ErrorCode.MODEL_OUTPUT_PARSE_ERROR: {
        "message": "模型输出解析错误",
        "http_status": 500,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.MODEL_TIMEOUT: {
        "message": "模型推理超时",
        "http_status": 504,
        "severity": ErrorSeverity.WARNING,
    },
    ErrorCode.MODEL_CONTEXT_TOO_LONG: {
        "message": "上下文长度超限",
        "http_status": 413,
        "severity": ErrorSeverity.WARNING,
    },
    ErrorCode.MODEL_TOKEN_LIMIT_EXCEEDED: {
        "message": "Token数量超限",
        "http_status": 413,
        "severity": ErrorSeverity.WARNING,
    },
    ErrorCode.MODEL_UNSUPPORTED_OPERATION: {
        "message": "不支持的操作",
        "http_status": 400,
        "severity": ErrorSeverity.WARNING,
    },
    ErrorCode.MODEL_WEIGHTS_CORRUPTED: {
        "message": "模型权重损坏",
        "http_status": 500,
        "severity": ErrorSeverity.CRITICAL,
    },
    ErrorCode.MODEL_QUANTIZATION_ERROR: {
        "message": "模型量化错误",
        "http_status": 500,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.MODEL_COMPATIBILITY_ERROR: {
        "message": "模型兼容性错误",
        "http_status": 500,
        "severity": ErrorSeverity.ERROR,
    },

    # 工具错误
    ErrorCode.TOOL_NOT_FOUND: {
        "message": "工具未找到",
        "http_status": 404,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.TOOL_EXECUTION_FAILED: {
        "message": "工具执行失败",
        "http_status": 500,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.TOOL_INPUT_INVALID: {
        "message": "工具输入无效",
        "http_status": 400,
        "severity": ErrorSeverity.WARNING,
    },
    ErrorCode.TOOL_OUTPUT_INVALID: {
        "message": "工具输出无效",
        "http_status": 500,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.TOOL_TIMEOUT: {
        "message": "工具执行超时",
        "http_status": 504,
        "severity": ErrorSeverity.WARNING,
    },
    ErrorCode.TOOL_PERMISSION_DENIED: {
        "message": "工具权限不足",
        "http_status": 403,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.TOOL_NOT_REGISTERED: {
        "message": "工具未注册",
        "http_status": 404,
        "severity": ErrorSeverity.WARNING,
    },
    ErrorCode.TOOL_DEPENDENCY_MISSING: {
        "message": "工具依赖缺失",
        "http_status": 500,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.TOOL_SIGNATURE_MISMATCH: {
        "message": "工具签名不匹配",
        "http_status": 400,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.TOOL_SANDBOX_VIOLATION: {
        "message": "沙箱违规",
        "http_status": 403,
        "severity": ErrorSeverity.CRITICAL,
    },

    # 代理错误
    ErrorCode.AGENT_INIT_FAILED: {
        "message": "代理初始化失败",
        "http_status": 500,
        "severity": ErrorSeverity.CRITICAL,
    },
    ErrorCode.AGENT_EXECUTION_FAILED: {
        "message": "代理执行失败",
        "http_status": 500,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.AGENT_STATE_CORRUPTED: {
        "message": "代理状态损坏",
        "http_status": 500,
        "severity": ErrorSeverity.CRITICAL,
    },
    ErrorCode.AGENT_COMMUNICATION_ERROR: {
        "message": "代理通信错误",
        "http_status": 503,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.AGENT_PLAN_INVALID: {
        "message": "代理计划无效",
        "http_status": 400,
        "severity": ErrorSeverity.WARNING,
    },
    ErrorCode.AGENT_MEMORY_ERROR: {
        "message": "代理记忆错误",
        "http_status": 500,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.AGENT_CONTEXT_OVERFLOW: {
        "message": "代理上下文溢出",
        "http_status": 413,
        "severity": ErrorSeverity.WARNING,
    },
    ErrorCode.AGENT_LOOP_DETECTED: {
        "message": "检测到代理循环",
        "http_status": 500,
        "severity": ErrorSeverity.WARNING,
    },
    ErrorCode.AGENT_TASK_TIMEOUT: {
        "message": "代理任务超时",
        "http_status": 504,
        "severity": ErrorSeverity.WARNING,
    },
    ErrorCode.AGENT_DELEGATION_FAILED: {
        "message": "代理委派失败",
        "http_status": 500,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.AGENT_REASONING_FAILED: {
        "message": "代理推理失败",
        "http_status": 500,
        "severity": ErrorSeverity.ERROR,
    },

    # 安全错误
    ErrorCode.SECURITY_AUTH_FAILED: {
        "message": "认证失败",
        "http_status": 401,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.SECURITY_PERMISSION_DENIED: {
        "message": "权限不足",
        "http_status": 403,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.SECURITY_TOKEN_EXPIRED: {
        "message": "令牌已过期",
        "http_status": 401,
        "severity": ErrorSeverity.WARNING,
    },
    ErrorCode.SECURITY_TOKEN_INVALID: {
        "message": "令牌无效",
        "http_status": 401,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.SECURITY_INPUT_INJECTION: {
        "message": "检测到输入注入",
        "http_status": 400,
        "severity": ErrorSeverity.CRITICAL,
    },
    ErrorCode.SECURITY_OUTPUT_FILTERED: {
        "message": "输出已被过滤",
        "http_status": 200,
        "severity": ErrorSeverity.WARNING,
    },
    ErrorCode.SECURITY_RATE_LIMIT_EXCEEDED: {
        "message": "安全速率限制超限",
        "http_status": 429,
        "severity": ErrorSeverity.WARNING,
    },
    ErrorCode.SECURITY_ENCRYPTION_ERROR: {
        "message": "加密错误",
        "http_status": 500,
        "severity": ErrorSeverity.CRITICAL,
    },
    ErrorCode.SECURITY_SIGNATURE_INVALID: {
        "message": "签名无效",
        "http_status": 401,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.SECURITY_SANDBOX_ESCAPE: {
        "message": "沙箱逃逸尝试",
        "http_status": 403,
        "severity": ErrorSeverity.CRITICAL,
    },
    ErrorCode.SECURITY_API_KEY_INVALID: {
        "message": "API密钥无效",
        "http_status": 401,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.SECURITY_AUDIT_LOG_FAILED: {
        "message": "审计日志记录失败",
        "http_status": 500,
        "severity": ErrorSeverity.ERROR,
    },

    # 存储错误
    ErrorCode.STORAGE_CONNECTION_FAILED: {
        "message": "存储连接失败",
        "http_status": 503,
        "severity": ErrorSeverity.CRITICAL,
    },
    ErrorCode.STORAGE_READ_ERROR: {
        "message": "存储读取错误",
        "http_status": 500,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.STORAGE_WRITE_ERROR: {
        "message": "存储写入错误",
        "http_status": 500,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.STORAGE_DELETE_ERROR: {
        "message": "存储删除错误",
        "http_status": 500,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.STORAGE_NOT_FOUND: {
        "message": "存储记录未找到",
        "http_status": 404,
        "severity": ErrorSeverity.WARNING,
    },
    ErrorCode.STORAGE_SERIALIZATION_ERROR: {
        "message": "序列化错误",
        "http_status": 500,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.STORAGE_DESERIALIZATION_ERROR: {
        "message": "反序列化错误",
        "http_status": 500,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.STORAGE_QUOTA_EXCEEDED: {
        "message": "存储配额超限",
        "http_status": 507,
        "severity": ErrorSeverity.WARNING,
    },
    ErrorCode.STORAGE_CORRUPTION: {
        "message": "存储数据损坏",
        "http_status": 500,
        "severity": ErrorSeverity.CRITICAL,
    },
    ErrorCode.STORAGE_LOCK_TIMEOUT: {
        "message": "存储锁超时",
        "http_status": 504,
        "severity": ErrorSeverity.WARNING,
    },
    ErrorCode.STORAGE_MIGRATION_ERROR: {
        "message": "存储迁移错误",
        "http_status": 500,
        "severity": ErrorSeverity.CRITICAL,
    },

    # 工作流错误
    ErrorCode.WORKFLOW_NOT_FOUND: {
        "message": "工作流未找到",
        "http_status": 404,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.WORKFLOW_EXECUTION_FAILED: {
        "message": "工作流执行失败",
        "http_status": 500,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.WORKFLOW_STEP_FAILED: {
        "message": "工作流步骤失败",
        "http_status": 500,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.WORKFLOW_INVALID_STATE: {
        "message": "工作流状态无效",
        "http_status": 409,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.WORKFLOW_CYCLE_DETECTED: {
        "message": "检测到工作流循环",
        "http_status": 400,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.WORKFLOW_TIMEOUT: {
        "message": "工作流超时",
        "http_status": 504,
        "severity": ErrorSeverity.WARNING,
    },
    ErrorCode.WORKFLOW_INPUT_INVALID: {
        "message": "工作流输入无效",
        "http_status": 400,
        "severity": ErrorSeverity.WARNING,
    },
    ErrorCode.WORKFLOW_PARALLEL_ERROR: {
        "message": "工作流并行执行错误",
        "http_status": 500,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.WORKFLOW_ROLLBACK_FAILED: {
        "message": "工作流回滚失败",
        "http_status": 500,
        "severity": ErrorSeverity.CRITICAL,
    },
    ErrorCode.WORKFLOW_DEPENDENCY_ERROR: {
        "message": "工作流依赖错误",
        "http_status": 400,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.WORKFLOW_BRANCH_ERROR: {
        "message": "工作流分支错误",
        "http_status": 500,
        "severity": ErrorSeverity.ERROR,
    },
    ErrorCode.WORKFLOW_COMPENSATION_FAILED: {
        "message": "工作流补偿失败",
        "http_status": 500,
        "severity": ErrorSeverity.CRITICAL,
    },
}


def get_error_metadata(code: ErrorCode) -> Dict[str, Any]:
    """
    获取错误码的元数据信息

    Args:
        code: 错误码枚举值

    Returns:
        包含 message, http_status, severity 的字典
    """
    return _ERROR_METADATA.get(code, {
        "message": "未知错误",
        "http_status": 500,
        "severity": ErrorSeverity.ERROR,
    })


def get_error_category(code: int) -> str:
    """
    根据错误码获取错误分类名称

    Args:
        code: 错误码数值

    Returns:
        错误分类名称字符串
    """
    category = code // 1000
    categories = {
        1: "CONFIG_ERROR",
        2: "NETWORK_ERROR",
        3: "MODEL_ERROR",
        4: "TOOL_ERROR",
        5: "AGENT_ERROR",
        6: "SECURITY_ERROR",
        7: "STORAGE_ERROR",
        8: "WORKFLOW_ERROR",
    }
    return categories.get(category, "UNKNOWN_ERROR")


class AGIError(Exception):
    """
    AGI框架基础异常类

    所有框架内自定义异常的基类，提供结构化错误信息。

    Attributes:
        code: 错误码枚举值
        message: 错误消息
        details: 错误详细信息字典
        cause: 原始异常
        http_status: HTTP状态码
        severity: 错误严重级别
    """

    def __init__(
        self,
        code: ErrorCode,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        metadata = get_error_metadata(code)
        self.code = code
        self.message = message or metadata["message"]
        self.details = details or {}
        self.cause = cause
        self.http_status = metadata["http_status"]
        self.severity = metadata["severity"]
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """将异常转换为字典格式"""
        result = {
            "code": int(self.code),
            "message": self.message,
            "severity": self.severity,
            "http_status": self.http_status,
            "category": get_error_category(int(self.code)),
        }
        if self.details:
            result["details"] = self.details
        if self.cause:
            result["cause"] = str(self.cause)
        return result

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(code={self.code}, "
            f"message={self.message!r}, severity={self.severity})"
        )

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class ConfigError(AGIError):
    """配置相关异常"""

    def __init__(
        self,
        code: ErrorCode = ErrorCode.CONFIG_LOAD_FAILED,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        # 确保错误码属于配置类别
        if not (1000 < code < 2000):
            code = ErrorCode.CONFIG_LOAD_FAILED
        super().__init__(code, message, details, cause)


class NetworkError(AGIError):
    """网络相关异常"""

    def __init__(
        self,
        code: ErrorCode = ErrorCode.NETWORK_CONNECTION_FAILED,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        if not (2000 < code < 3000):
            code = ErrorCode.NETWORK_CONNECTION_FAILED
        super().__init__(code, message, details, cause)


class ModelError(AGIError):
    """模型相关异常"""

    def __init__(
        self,
        code: ErrorCode = ErrorCode.MODEL_LOAD_FAILED,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        if not (3000 < code < 4000):
            code = ErrorCode.MODEL_LOAD_FAILED
        super().__init__(code, message, details, cause)


class ToolError(AGIError):
    """工具相关异常"""

    def __init__(
        self,
        code: ErrorCode = ErrorCode.TOOL_EXECUTION_FAILED,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        if not (4000 < code < 5000):
            code = ErrorCode.TOOL_EXECUTION_FAILED
        super().__init__(code, message, details, cause)


class AgentError(AGIError):
    """代理相关异常"""

    def __init__(
        self,
        code: ErrorCode = ErrorCode.AGENT_EXECUTION_FAILED,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        if not (5000 < code < 6000):
            code = ErrorCode.AGENT_EXECUTION_FAILED
        super().__init__(code, message, details, cause)


class SecurityError(AGIError):
    """安全相关异常"""

    def __init__(
        self,
        code: ErrorCode = ErrorCode.SECURITY_AUTH_FAILED,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        if not (6000 < code < 7000):
            code = ErrorCode.SECURITY_AUTH_FAILED
        super().__init__(code, message, details, cause)


class StorageError(AGIError):
    """存储相关异常"""

    def __init__(
        self,
        code: ErrorCode = ErrorCode.STORAGE_CONNECTION_FAILED,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        if not (7000 < code < 8000):
            code = ErrorCode.STORAGE_CONNECTION_FAILED
        super().__init__(code, message, details, cause)


class WorkflowError(AGIError):
    """工作流相关异常"""

    def __init__(
        self,
        code: ErrorCode = ErrorCode.WORKFLOW_EXECUTION_FAILED,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        if not (8000 < code < 9000):
            code = ErrorCode.WORKFLOW_EXECUTION_FAILED
        super().__init__(code, message, details, cause)


# 错误码到异常类的映射，用于自动选择正确的异常类
_ERROR_CLASS_MAP: Dict[int, type] = {
    1: ConfigError,
    2: NetworkError,
    3: ModelError,
    4: ToolError,
    5: AgentError,
    6: SecurityError,
    7: StorageError,
    8: WorkflowError,
}


def create_error(
    code: ErrorCode,
    message: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    cause: Optional[Exception] = None,
) -> AGIError:
    """
    根据错误码自动创建对应类型的异常

    Args:
        code: 错误码
        message: 自定义错误消息
        details: 详细信息
        cause: 原始异常

    Returns:
        对应类型的AGIError子类实例
    """
    category = int(code) // 1000
    error_class = _ERROR_CLASS_MAP.get(category, AGIError)
    return error_class(code, message, details, cause)
