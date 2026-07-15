"""
Workflow 模块

提供工作流编辑、变量管理、条件分支、循环控制、异常恢复、
公共工作流库和调试功能。
仅使用Python标准库实现。
"""

from .editor import (
    WorkflowEditor,
    Step,
    Workflow,
    StepType,
)
from .variable_manager import (
    VariableManager,
    Variable,
    VariableScope,
    VariableType,
    create_variable,
)
from .conditionals import (
    ConditionalBranch,
    Condition,
    ComplexCondition,
    ConditionEvaluator,
    Operator,
    LogicalOperator,
    create_condition,
    create_and_condition,
    create_or_condition,
    create_not_condition,
)
from .loops import (
    LoopAction,
    LoopType,
    LoopConfig,
    LoopState,
    LoopIterator,
    create_for_each_loop,
    create_while_loop,
    create_count_loop,
)
from .error_recovery import (
    ErrorRecovery,
    RecoveryAction,
    RecoveryType,
    RecoveryPolicy,
    create_retry_action,
    create_skip_action,
    create_goto_action,
    create_fallback_action,
)
from .library import (
    WorkflowLibrary,
    WorkflowTemplate,
    get_default_library,
)
from .debug import (
    WorkflowDebugger,
    DebugState,
    ScreenshotOnError,
    DebugMode,
    create_debugger,
    create_screenshot_on_error,
)

__all__ = [
    # Editor
    "WorkflowEditor",
    "Step",
    "Workflow",
    "StepType",
    # Variable Manager
    "VariableManager",
    "Variable",
    "VariableScope",
    "VariableType",
    "create_variable",
    # Conditionals
    "ConditionalBranch",
    "Condition",
    "ComplexCondition",
    "ConditionEvaluator",
    "Operator",
    "LogicalOperator",
    "create_condition",
    "create_and_condition",
    "create_or_condition",
    "create_not_condition",
    # Loops
    "LoopAction",
    "LoopType",
    "LoopConfig",
    "LoopState",
    "LoopIterator",
    "create_for_each_loop",
    "create_while_loop",
    "create_count_loop",
    # Error Recovery
    "ErrorRecovery",
    "RecoveryAction",
    "RecoveryType",
    "RecoveryPolicy",
    "create_retry_action",
    "create_skip_action",
    "create_goto_action",
    "create_fallback_action",
    # Library
    "WorkflowLibrary",
    "WorkflowTemplate",
    "get_default_library",
    # Debug
    "WorkflowDebugger",
    "DebugState",
    "ScreenshotOnError",
    "DebugMode",
    "create_debugger",
    "create_screenshot_on_error",
]
