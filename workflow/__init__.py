"""
工作流引擎 (Workflow Engine)

提供基于DAG的工作流编排和执行能力：
- DAGEngine: DAG任务图引擎
- WorkflowExecutor: 工作流执行器
- WorkflowStateManager: 状态持久化
- 各类节点: TaskNode, LLMNode, ToolNode等
- SagaCompensation: Saga补偿处理器
- WorkflowDSLParser: YAML DSL解析器
- ConditionalExecutor: 条件执行器
- ParallelExecutor: 并行执行器
- HumanApproval: 人工审批
- WorkflowVisualizer: 工作流可视化
"""

from .graph_engine import (
    DAGNode,
    DAGEdge,
    DAGEngine,
    NodeState,
)
from .executor import (
    WorkflowExecutor,
    WorkflowResult,
    NodeResult,
)
from .state_manager import (
    WorkflowStateManager,
    WorkflowState,
)
from .nodes import (
    TaskNode,
    LLMNode,
    ToolNode,
    ConditionalNode,
    LoopNode,
    ParallelNode,
    HumanApprovalNode,
    SubWorkflowNode,
    ErrorHandlerNode,
    DelayNode,
)
from .compensation import (
    SagaCompensation,
    CompensationLog,
)
from .dsl_parser import WorkflowDSLParser

# 条件执行器
from .conditional_executor import (
    ConditionalExecutor,
    ConditionEvaluator,
    LogicalOperator,
    ComparisonOperator,
    BranchExecutor,
    SwitchCaseExecutor,
    RuleEngineIntegration,
    Condition,
    CompositeCondition,
    Branch,
    Rule,
    ConditionError,
    InvalidConditionError,
    BranchExecutionError,
)

# 并行执行器
from .parallel_executor import (
    ParallelExecutor,
    DAGScheduler,
    WorkerPool,
    DependencyResolver,
    ForkJoinExecutor,
    RaceConditionHandler,
    TaskPriority,
    TaskStatus,
    SchedulingStrategy,
    TaskResult,
    DependencyNode,
    PrioritizedTask,
    ParallelExecutionError,
    DependencyError,
    RaceConditionError,
    WorkerPoolExhaustedError,
)

# 人工审批
from .human_approval import (
    HumanApproval,
    ApprovalRequest,
    ApprovalResponse,
    EscalationRule,
    NotificationDispatcher,
    TimeoutManager,
    EscalationPolicy,
    ApprovalAudit,
    ApprovalStatus,
    NotificationChannel,
    EscalationLevel,
    ApprovalError,
    ApprovalTimeoutError,
    EscalationError,
    NotificationError,
)

# 可视化（图布局）
from .visualization import (
    WorkflowVisualizer as GraphWorkflowVisualizer,
    GraphLayoutEngine,
    MermaidExporter,
    GraphvizExporter,
    ProgressRenderer,
    ASCIIVisualizer,
    HTMLGenerator,
    LayoutAlgorithm,
    ExportFormat,
    Position,
    NodeLayout,
    EdgeLayout,
    GraphLayout,
    VisualizationError,
)

# 可视化（执行流）
from .visualizer import (
    WorkflowVisualizer as ExecutionWorkflowVisualizer,
    ExecutionFlowRenderer,
    NodeStatusColorizer,
    ProgressOverlay,
    ErrorHighlighter,
    NodeSelector,
    StatsDashboard,
    NodeVisualStatus,
    NodeVisualInfo,
    EdgeVisualInfo,
    ExecutionSnapshot,
)

# 循环构造
from .loops import (
    ForEachLoop,
    WhileLoop,
    DoWhileLoop,
    LoopVariable,
    IterationTracker,
    LoopConfig,
    BreakSignal,
    ContinueSignal,
    LoopResult,
    create_foreach,
    create_while,
    create_do_while,
)

# 重试策略
from .retry_policy import (
    RetryPolicy,
    FixedDelayRetry,
    ExponentialBackoffRetry,
    JitterStrategy,
    JitterType,
    CircuitBreakerRetry,
    CircuitState,
    RetryBudget,
    ConditionRetry,
    RetryResult,
    RetryHistory,
)

# 超时包装
from .timeout_wrapper import (
    TimeoutWrapper,
    DeadlineTracker,
    HierarchicalTimeout,
    TimeoutPropagator,
    GracefulCanceller,
    TimeoutCallback,
    TimeoutConfig,
    TimeoutExpired,
    CancellationError,
)

# 工具节点
from .tool_node import (
    ToolNode as AdvancedToolNode,
    ToolDiscovery,
    ParameterValidator,
    ToolExecutor,
    ResultParser,
    ToolErrorHandler,
    ToolCache,
    ToolNodeConfig,
    ToolDefinition,
    ParameterSpec,
    ParameterType,
    ToolExecutionError,
    ToolNotFoundError,
    ParameterValidationError,
    ToolTimeoutError,
)

# LLM 节点
from .llm_node import (
    LLMNode as AdvancedLLMNode,
    PromptRenderer,
    ModelSelector,
    StreamingHandler,
    TokenCounter,
    CostTracker,
    FallbackHandler,
    OutputParser,
    LLMNodeConfig,
    ModelInfo,
    ModelPricing,
    LLMError,
    LLMTimeoutError,
    LLMRateLimitError,
    PromptRenderingError,
)

# 装饰器
from .decorators import (
    NodeType,
    RetryStrategy,
    ExecutionPhase,
    DecoratorMetadata,
    ExecutionContext,
    BaseDecorator,
    DecoratorRegistry,
    TaskDecorator,
    StepDecorator,
    ConditionDecorator,
    RetryDecorator,
    DecoratorChain,
    task,
    step,
    condition,
    retry,
)

# 审计日志
from .audit_logger import (
    AuditLevel,
    AuditCategory,
    AuditSearchOperator,
    AuditEntry,
    AuditSearchCriteria,
    AuditSearch,
    AuditReporter,
    ComplianceReport,
    WorkflowAuditLogger,
    get_audit_logger,
)

# REST API
from .engine_api import (
    WorkflowStatus,
    WorkflowPriority,
    WebhookEvent,
    WorkflowRequest,
    WorkflowResponse,
    WorkflowInfo,
    WebhookConfig,
    WebhookPayload,
    WorkflowAPI,
    WorkflowStarter,
    WorkflowPausor,
    WorkflowCanceler,
    WorkflowStatusAPI,
    get_workflow_api,
)

# 指标收集
from .metrics_collector import (
    MetricType,
    AggregationType,
    MetricPoint,
    MetricSeries,
    NodeMetrics,
    WorkflowMetrics,
    MetricsAggregator,
    BottleneckDetector,
    WorkflowMetricsCollector,
    get_metrics_collector,
)

__all__ = [
    # DAG引擎
    "DAGNode",
    "DAGEdge",
    "DAGEngine",
    "NodeState",
    # 执行器
    "WorkflowExecutor",
    "WorkflowResult",
    "NodeResult",
    # 状态管理
    "WorkflowStateManager",
    "WorkflowState",
    # 节点类型
    "TaskNode",
    "LLMNode",
    "ToolNode",
    "ConditionalNode",
    "LoopNode",
    "ParallelNode",
    "HumanApprovalNode",
    "SubWorkflowNode",
    "ErrorHandlerNode",
    "DelayNode",
    # 补偿
    "SagaCompensation",
    "CompensationLog",
    # DSL解析
    "WorkflowDSLParser",
    # 条件执行器
    "ConditionalExecutor",
    "ConditionEvaluator",
    "LogicalOperator",
    "ComparisonOperator",
    "BranchExecutor",
    "SwitchCaseExecutor",
    "RuleEngineIntegration",
    "Condition",
    "CompositeCondition",
    "Branch",
    "Rule",
    "ConditionError",
    "InvalidConditionError",
    "BranchExecutionError",
    # 并行执行器
    "ParallelExecutor",
    "DAGScheduler",
    "WorkerPool",
    "DependencyResolver",
    "ForkJoinExecutor",
    "RaceConditionHandler",
    "TaskPriority",
    "TaskStatus",
    "SchedulingStrategy",
    "TaskResult",
    "DependencyNode",
    "PrioritizedTask",
    "ParallelExecutionError",
    "DependencyError",
    "RaceConditionError",
    "WorkerPoolExhaustedError",
    # 人工审批
    "HumanApproval",
    "ApprovalRequest",
    "ApprovalResponse",
    "EscalationRule",
    "NotificationDispatcher",
    "TimeoutManager",
    "EscalationPolicy",
    "ApprovalAudit",
    "ApprovalStatus",
    "NotificationChannel",
    "EscalationLevel",
    "ApprovalError",
    "ApprovalTimeoutError",
    "EscalationError",
    "NotificationError",
    # 可视化（图布局）
    "GraphWorkflowVisualizer",
    "GraphLayoutEngine",
    "MermaidExporter",
    "GraphvizExporter",
    "ProgressRenderer",
    "ASCIIVisualizer",
    "HTMLGenerator",
    "LayoutAlgorithm",
    "ExportFormat",
    "Position",
    "NodeLayout",
    "EdgeLayout",
    "GraphLayout",
    "VisualizationError",
    # 可视化（执行流）
    "ExecutionWorkflowVisualizer",
    "ExecutionFlowRenderer",
    "NodeStatusColorizer",
    "ProgressOverlay",
    "ErrorHighlighter",
    "NodeSelector",
    "StatsDashboard",
    "NodeVisualStatus",
    "NodeVisualInfo",
    "EdgeVisualInfo",
    "ExecutionSnapshot",
    # 循环构造
    "ForEachLoop",
    "WhileLoop",
    "DoWhileLoop",
    "LoopVariable",
    "IterationTracker",
    "LoopConfig",
    "BreakSignal",
    "ContinueSignal",
    "LoopResult",
    "create_foreach",
    "create_while",
    "create_do_while",
    # 重试策略
    "RetryPolicy",
    "FixedDelayRetry",
    "ExponentialBackoffRetry",
    "JitterStrategy",
    "JitterType",
    "CircuitBreakerRetry",
    "CircuitState",
    "RetryBudget",
    "ConditionRetry",
    "RetryResult",
    "RetryHistory",
    # 超时包装
    "TimeoutWrapper",
    "DeadlineTracker",
    "HierarchicalTimeout",
    "TimeoutPropagator",
    "GracefulCanceller",
    "TimeoutCallback",
    "TimeoutConfig",
    "TimeoutExpired",
    "CancellationError",
    # 工具节点
    "AdvancedToolNode",
    "ToolDiscovery",
    "ParameterValidator",
    "ToolExecutor",
    "ResultParser",
    "ToolErrorHandler",
    "ToolCache",
    "ToolNodeConfig",
    "ToolDefinition",
    "ParameterSpec",
    "ParameterType",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ParameterValidationError",
    "ToolTimeoutError",
    # LLM 节点
    "AdvancedLLMNode",
    "PromptRenderer",
    "ModelSelector",
    "StreamingHandler",
    "TokenCounter",
    "CostTracker",
    "FallbackHandler",
    "OutputParser",
    "LLMNodeConfig",
    "ModelInfo",
    "ModelPricing",
    "LLMError",
    "LLMTimeoutError",
    "LLMRateLimitError",
    "PromptRenderingError",
    # 装饰器
    "NodeType",
    "RetryStrategy",
    "ExecutionPhase",
    "DecoratorMetadata",
    "ExecutionContext",
    "BaseDecorator",
    "DecoratorRegistry",
    "TaskDecorator",
    "StepDecorator",
    "ConditionDecorator",
    "RetryDecorator",
    "DecoratorChain",
    "task",
    "step",
    "condition",
    "retry",
    # 审计日志
    "AuditLevel",
    "AuditCategory",
    "AuditSearchOperator",
    "AuditEntry",
    "AuditSearchCriteria",
    "AuditSearch",
    "AuditReporter",
    "ComplianceReport",
    "WorkflowAuditLogger",
    "get_audit_logger",
    # REST API
    "WorkflowStatus",
    "WorkflowPriority",
    "WebhookEvent",
    "WorkflowRequest",
    "WorkflowResponse",
    "WorkflowInfo",
    "WebhookConfig",
    "WebhookPayload",
    "WorkflowAPI",
    "WorkflowStarter",
    "WorkflowPausor",
    "WorkflowCanceler",
    "WorkflowStatusAPI",
    "get_workflow_api",
    # 指标收集
    "MetricType",
    "AggregationType",
    "MetricPoint",
    "MetricSeries",
    "NodeMetrics",
    "WorkflowMetrics",
    "MetricsAggregator",
    "BottleneckDetector",
    "WorkflowMetricsCollector",
    "get_metrics_collector",
]
