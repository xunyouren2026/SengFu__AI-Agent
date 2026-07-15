"""
工作流编排引擎 - Workflow Orchestration Engine
==============================================

一个全面的工作流编排模块，包含任务节点、DAG图、调度器、执行引擎、
数据流、条件分支、事件系统、资源管理、监控和配置管理。

核心组件:
1. TaskNode - 任务节点
2. WorkflowDAG - 工作流有向无环图
3. WorkflowScheduler - 工作流调度器
4. WorkflowExecutor - 工作流执行器
5. DataFlow - 数据流管理
6. ConditionalBranch - 条件分支
7. EventSystem - 事件系统
8. ResourceManager - 资源管理器
9. Monitoring - 监控模块
10. WorkflowConfig - 配置管理
11. WorkflowEngine - 主编排引擎

Author: AGI Unified Framework
Version: 1.0.0
"""

import uuid
import time
import heapq
import threading
import queue
import json
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Callable, Any, Tuple, Union
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from abc import ABC, abstractmethod
import copy

# ============================================================================
# 配置管理 - WorkflowConfig
# ============================================================================

class WorkflowConfig:
    """工作流配置管理类"""
    
    DEFAULT_CONFIG = {
        'max_workers': 4,
        'default_timeout': 300,
        'max_retries': 3,
        'retry_delay': 1.0,
        'checkpoint_interval': 60,
        'enable_monitoring': True,
        'log_level': 'INFO',
        'resource_limits': {
            'max_cpu_percent': 80,
            'max_memory_mb': 1024,
            'max_gpu_memory_mb': 4096
        },
        'scheduling_policy': 'asap',
        'enable_data_lineage': True,
        'event_queue_size': 1000
    }
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self._config = copy.deepcopy(self.DEFAULT_CONFIG)
        if config_dict:
            self._merge_config(config_dict)
        self._setup_logging()
    
    def _merge_config(self, config_dict: Dict[str, Any]):
        """递归合并配置"""
        for key, value in config_dict.items():
            if key in self._config and isinstance(self._config[key], dict) and isinstance(value, dict):
                self._config[key].update(value)
            else:
                self._config[key] = value
    
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=getattr(logging, self._config['log_level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('WorkflowEngine')
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any):
        """设置配置项"""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        return copy.deepcopy(self._config)
    
    @classmethod
    def from_file(cls, filepath: str) -> 'WorkflowConfig':
        """从文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(config_dict)
    
    def save_to_file(self, filepath: str):
        """保存配置到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)


# ============================================================================
# 任务状态枚举
# ============================================================================

class TaskState(Enum):
    """任务状态枚举"""
    PENDING = auto()      # 等待执行
    RUNNING = auto()      # 正在执行
    COMPLETED = auto()    # 已完成
    FAILED = auto()       # 执行失败
    CANCELLED = auto()    # 已取消
    SKIPPED = auto()      # 已跳过
    TIMEOUT = auto()      # 超时


# ============================================================================
# 任务节点 - TaskNode
# ============================================================================

@dataclass
class RetryPolicy:
    """重试策略"""
    max_retries: int = 3
    delay_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_delay_seconds: float = 60.0
    retry_exceptions: Tuple[type, ...] = (Exception,)
    
    def calculate_delay(self, attempt: int) -> float:
        """计算第attempt次重试的延迟时间"""
        delay = self.delay_seconds * (self.backoff_multiplier ** attempt)
        return min(delay, self.max_delay_seconds)


@dataclass
class ResourceRequirements:
    """资源需求"""
    cpu_cores: float = 1.0
    memory_mb: int = 512
    gpu_memory_mb: int = 0
    disk_mb: int = 1024
    network_bandwidth_mbps: float = 0.0


@dataclass
class TaskNode:
    """
    工作流任务节点
    
    属性:
        task_id: 任务唯一标识
        name: 任务名称
        func: 执行函数
        inputs: 输入参数定义
        outputs: 输出参数定义
        dependencies: 上游依赖任务ID列表
        retry_policy: 重试策略
        timeout: 超时时间(秒)
        resources: 资源需求
        priority: 优先级(数字越小优先级越高)
        metadata: 元数据
    """
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "unnamed_task"
    func: Optional[Callable] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    timeout: float = 300.0
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    priority: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 运行时状态
    state: TaskState = TaskState.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    attempt_count: int = 0
    error_message: Optional[str] = None
    result: Any = None
    
    # 依赖追踪
    upstream: Set[str] = field(default_factory=set)
    downstream: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        if not self.upstream and self.dependencies:
            self.upstream = set(self.dependencies)
    
    def get_execution_time(self) -> Optional[float]:
        """获取执行时间"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    def is_ready(self, completed_tasks: Set[str]) -> bool:
        """检查任务是否就绪(所有依赖已完成)"""
        return self.upstream.issubset(completed_tasks)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'task_id': self.task_id,
            'name': self.name,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'dependencies': list(self.upstream),
            'state': self.state.name,
            'timeout': self.timeout,
            'priority': self.priority,
            'attempt_count': self.attempt_count,
            'execution_time': self.get_execution_time(),
            'error_message': self.error_message
        }


# ============================================================================
# 有向无环图 - WorkflowDAG
# ============================================================================

class WorkflowDAG:
    """
    工作流有向无环图
    
    提供DAG的构建、验证、拓扑排序和关键路径分析功能
    """
    
    def __init__(self, name: str = "workflow"):
        self.name = name
        self.nodes: Dict[str, TaskNode] = {}
        self.edges: Dict[str, Set[str]] = defaultdict(set)  # node -> downstream nodes
        self.reverse_edges: Dict[str, Set[str]] = defaultdict(set)  # node -> upstream nodes
        self._sorted_order: Optional[List[str]] = None
        self._parallel_groups: Optional[List[Set[str]]] = None
    
    def add_node(self, node: TaskNode) -> 'WorkflowDAG':
        """添加节点"""
        self.nodes[node.task_id] = node
        # 更新依赖关系
        for dep_id in node.upstream:
            self.edges[dep_id].add(node.task_id)
            self.reverse_edges[node.task_id].add(dep_id)
        self._invalidate_cache()
        return self
    
    def remove_node(self, task_id: str) -> bool:
        """移除节点"""
        if task_id not in self.nodes:
            return False
        
        # 移除相关边
        for upstream_id in self.reverse_edges[task_id]:
            self.edges[upstream_id].discard(task_id)
        for downstream_id in self.edges[task_id]:
            self.reverse_edges[downstream_id].discard(task_id)
        
        del self.edges[task_id]
        del self.reverse_edges[task_id]
        del self.nodes[task_id]
        self._invalidate_cache()
        return True
    
    def add_edge(self, from_id: str, to_id: str) -> 'WorkflowDAG':
        """添加边"""
        if from_id not in self.nodes or to_id not in self.nodes:
            raise ValueError(f"节点不存在: {from_id} -> {to_id}")
        
        self.edges[from_id].add(to_id)
        self.reverse_edges[to_id].add(from_id)
        self.nodes[to_id].upstream.add(from_id)
        self.nodes[from_id].downstream.add(to_id)
        self._invalidate_cache()
        return self
    
    def remove_edge(self, from_id: str, to_id: str) -> bool:
        """移除边"""
        if to_id in self.edges[from_id]:
            self.edges[from_id].remove(to_id)
            self.reverse_edges[to_id].remove(from_id)
            self.nodes[to_id].upstream.discard(from_id)
            self.nodes[from_id].downstream.discard(to_id)
            self._invalidate_cache()
            return True
        return False
    
    def _invalidate_cache(self):
        """使缓存失效"""
        self._sorted_order = None
        self._parallel_groups = None
    
    def detect_cycle(self) -> Optional[List[str]]:
        """
        检测图中是否存在环
        返回: 如果存在环，返回环中的节点列表；否则返回None
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {node_id: WHITE for node_id in self.nodes}
        parent = {}
        
        def dfs(node_id: str, path: List[str]) -> Optional[List[str]]:
            color[node_id] = GRAY
            path.append(node_id)
            
            for neighbor in self.edges[node_id]:
                if color[neighbor] == GRAY:
                    # 发现环
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]
                if color[neighbor] == WHITE:
                    parent[neighbor] = node_id
                    result = dfs(neighbor, path)
                    if result:
                        return result
            
            path.pop()
            color[node_id] = BLACK
            return None
        
        for node_id in self.nodes:
            if color[node_id] == WHITE:
                cycle = dfs(node_id, [])
                if cycle:
                    return cycle
        return None
    
    def topological_sort(self) -> List[str]:
        """
        拓扑排序 - Kahn算法
        返回: 按执行顺序排列的节点ID列表
        """
        if self._sorted_order is not None:
            return self._sorted_order
        
        # 计算入度
        in_degree = {node_id: len(self.reverse_edges[node_id]) for node_id in self.nodes}
        
        # 初始化队列(使用优先队列支持优先级)
        ready_queue = []
        for node_id, degree in in_degree.items():
            if degree == 0:
                node = self.nodes[node_id]
                heapq.heappush(ready_queue, (node.priority, node_id))
        
        sorted_order = []
        while ready_queue:
            _, node_id = heapq.heappop(ready_queue)
            sorted_order.append(node_id)
            
            for neighbor in self.edges[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    neighbor_node = self.nodes[neighbor]
                    heapq.heappush(ready_queue, (neighbor_node.priority, neighbor))
        
        if len(sorted_order) != len(self.nodes):
            raise ValueError("图中存在环，无法进行拓扑排序")
        
        self._sorted_order = sorted_order
        return sorted_order
    
    def get_parallel_groups(self) -> List[Set[str]]:
        """
        获取可以并行执行的节点组
        返回: 每组包含可以同时执行的节点ID集合
        """
        if self._parallel_groups is not None:
            return self._parallel_groups
        
        groups = []
        executed = set()
        remaining = set(self.nodes.keys())
        
        while remaining:
            # 找出所有依赖已满足的任务
            ready = set()
            for node_id in remaining:
                if self.reverse_edges[node_id].issubset(executed):
                    ready.add(node_id)
            
            if not ready:
                raise ValueError("图中存在环")
            
            groups.append(ready)
            executed.update(ready)
            remaining -= ready
        
        self._parallel_groups = groups
        return groups
    
    def calculate_critical_path(self) -> Tuple[List[str], float]:
        """
        计算关键路径
        使用CPM(Critical Path Method)算法
        
        返回: (关键路径上的节点ID列表, 总工期)
        """
        topo_order = self.topological_sort()
        
        # 估计每个任务的持续时间(使用timeout作为估计)
        durations = {node_id: self.nodes[node_id].timeout for node_id in self.nodes}
        
        # 前向遍历 - 计算最早开始和结束时间
        es = {node_id: 0.0 for node_id in self.nodes}  # Earliest Start
        ef = {}  # Earliest Finish
        
        for node_id in topo_order:
            ef[node_id] = es[node_id] + durations[node_id]
            for successor in self.edges[node_id]:
                es[successor] = max(es[successor], ef[node_id])
        
        # 后向遍历 - 计算最晚开始和结束时间
        project_duration = max(ef.values()) if ef else 0
        lf = {node_id: project_duration for node_id in self.nodes}  # Latest Finish
        ls = {}  # Latest Start
        
        for node_id in reversed(topo_order):
            ls[node_id] = lf[node_id] - durations[node_id]
            for predecessor in self.reverse_edges[node_id]:
                lf[predecessor] = min(lf[predecessor], ls[node_id])
        
        # 计算总浮动时间并找出关键路径
        critical_path = []
        for node_id in topo_order:
            total_float = ls[node_id] - es[node_id]
            if abs(total_float) < 0.001:  # 关键路径上的任务总浮动时间为0
                critical_path.append(node_id)
        
        return critical_path, project_duration
    
    def get_independent_nodes(self) -> Set[str]:
        """获取没有依赖的独立节点"""
        return {node_id for node_id in self.nodes if not self.reverse_edges[node_id]}
    
    def get_leaf_nodes(self) -> Set[str]:
        """获取没有后继的叶子节点"""
        return {node_id for node_id in self.nodes if not self.edges[node_id]}
    
    def get_depth(self, task_id: str) -> int:
        """获取节点在DAG中的深度"""
        if not self.reverse_edges[task_id]:
            return 0
        return 1 + max(self.get_depth(pid) for pid in self.reverse_edges[task_id])
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        return {
            'name': self.name,
            'nodes': {tid: node.to_dict() for tid, node in self.nodes.items()},
            'edges': {k: list(v) for k, v in self.edges.items()},
            'critical_path': self.calculate_critical_path()[0] if self.nodes else []
        }


# ============================================================================
# 数据流管理 - DataFlow
# ============================================================================

class DataFlow:
    """
    数据流管理
    
    支持多种数据传递方式:
    - 直接传递: 任务间直接传递数据
    - 共享内存: 使用共享存储
    - 消息队列: 异步消息传递
    - 数据血缘: 追踪数据来源
    """
    
    def __init__(self, enable_lineage: bool = True):
        self.enable_lineage = enable_lineage
        self._data_store: Dict[str, Any] = {}  # 数据存储
        self._lineage: Dict[str, List[Dict[str, Any]]] = defaultdict(list)  # 数据血缘
        self._message_queue: queue.Queue = queue.Queue()
        self._lock = threading.RLock()
    
    def store(self, task_id: str, output_name: str, data: Any, 
              metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        存储任务输出数据
        
        返回: 数据引用ID
        """
        data_ref = f"{task_id}:{output_name}"
        
        with self._lock:
            self._data_store[data_ref] = {
                'data': data,
                'task_id': task_id,
                'output_name': output_name,
                'timestamp': time.time(),
                'metadata': metadata or {}
            }
            
            if self.enable_lineage:
                self._lineage[data_ref].append({
                    'producer': task_id,
                    'timestamp': time.time(),
                    'type': 'produce'
                })
        
        return data_ref
    
    def retrieve(self, data_ref: str, consumer_task_id: Optional[str] = None) -> Any:
        """
        检索数据
        
        参数:
            data_ref: 数据引用ID
            consumer_task_id: 消费任务ID(用于血缘追踪)
        """
        with self._lock:
            if data_ref not in self._data_store:
                raise KeyError(f"数据引用不存在: {data_ref}")
            
            data_entry = self._data_store[data_ref]
            
            if self.enable_lineage and consumer_task_id:
                self._lineage[data_ref].append({
                    'consumer': consumer_task_id,
                    'timestamp': time.time(),
                    'type': 'consume'
                })
            
            return data_entry['data']
    
    def get_lineage(self, data_ref: str) -> List[Dict[str, Any]]:
        """获取数据血缘历史"""
        return self._lineage.get(data_ref, [])
    
    def publish_message(self, channel: str, message: Any, 
                       metadata: Optional[Dict[str, Any]] = None):
        """发布消息到队列"""
        self._message_queue.put({
            'channel': channel,
            'message': message,
            'timestamp': time.time(),
            'metadata': metadata or {}
        })
    
    def consume_message(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """消费消息"""
        try:
            return self._message_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def resolve_inputs(self, task: TaskNode, 
                      data_mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        解析任务输入
        
        参数:
            task: 任务节点
            data_mapping: 输入名称到数据引用的映射
        
        返回:
            解析后的输入数据字典
        """
        resolved = {}
        for input_name, data_ref in data_mapping.items():
            try:
                resolved[input_name] = self.retrieve(data_ref, task.task_id)
            except KeyError as e:
                raise KeyError(f"任务 {task.task_id} 无法解析输入 {input_name}: {e}")
        return resolved
    
    def clear(self):
        """清空数据流"""
        with self._lock:
            self._data_store.clear()
            self._lineage.clear()
            while not self._message_queue.empty():
                try:
                    self._message_queue.get_nowait()
                except queue.Empty:
                    break


# ============================================================================
# 条件分支 - ConditionalBranch
# ============================================================================

class ConditionalBranch:
    """
    条件分支控制流
    
    支持:
    - If-else 分支
    - Switch-case 分支
    - While 循环
    - For-each 循环
    - Parallel 分支 (fork-join)
    """
    
    @staticmethod
    def if_else(condition: Callable[[], bool], 
                true_branch: Callable, 
                false_branch: Optional[Callable] = None) -> Any:
        """If-else 条件分支"""
        if condition():
            return true_branch()
        elif false_branch:
            return false_branch()
        return None
    
    @staticmethod
    def switch_case(value: Any, cases: Dict[Any, Callable], 
                    default: Optional[Callable] = None) -> Any:
        """Switch-case 分支"""
        if value in cases:
            return cases[value]()
        elif default:
            return default()
        return None
    
    @staticmethod
    def while_loop(condition: Callable[[], bool], 
                   body: Callable, 
                   max_iterations: int = 1000) -> List[Any]:
        """While 循环"""
        results = []
        iteration = 0
        while condition() and iteration < max_iterations:
            result = body()
            results.append(result)
            iteration += 1
        return results
    
    @staticmethod
    def for_each(items: List[Any], 
                 body: Callable[[Any], Any],
                 parallel: bool = False,
                 max_workers: int = 4) -> List[Any]:
        """
        For-each 循环
        
        参数:
            items: 要迭代的项目列表
            body: 处理函数
            parallel: 是否并行执行
            max_workers: 并行工作线程数
        """
        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(body, item) for item in items]
                return [f.result() for f in as_completed(futures)]
        else:
            return [body(item) for item in items]
    
    @staticmethod
    def fork_join(branches: List[Callable], 
                  max_workers: int = 4) -> List[Any]:
        """
        Fork-Join 并行分支
        
        并行执行多个分支，等待所有分支完成后返回结果
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(branch) for branch in branches]
            results = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append(e)
            return results
    
    @staticmethod
    def parallel_map(func: Callable[[Any], Any], 
                     items: List[Any],
                     max_workers: int = 4) -> List[Any]:
        """并行映射"""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(func, items))


# ============================================================================
# 事件系统 - EventSystem
# ============================================================================

class EventType(Enum):
    """事件类型"""
    TASK_STARTED = auto()
    TASK_COMPLETED = auto()
    TASK_FAILED = auto()
    TASK_TIMEOUT = auto()
    WORKFLOW_STARTED = auto()
    WORKFLOW_COMPLETED = auto()
    WORKFLOW_FAILED = auto()
    TIMER = auto()
    EXTERNAL = auto()
    CUSTOM = auto()


@dataclass
class WorkflowEvent:
    """工作流事件"""
    event_type: EventType
    source: str
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


class EventListener:
    """事件监听器"""
    
    def __init__(self, event_types: List[EventType], 
                 callback: Callable[[WorkflowEvent], None],
                 filter_func: Optional[Callable[[WorkflowEvent], bool]] = None):
        self.event_types = set(event_types)
        self.callback = callback
        self.filter_func = filter_func
        self.id = str(uuid.uuid4())[:8]
    
    def should_handle(self, event: WorkflowEvent) -> bool:
        """检查是否应该处理该事件"""
        if event.event_type not in self.event_types:
            return False
        if self.filter_func and not self.filter_func(event):
            return False
        return True


class EventSystem:
    """
    事件驱动工作流系统
    
    支持:
    - 事件监听
    - 事件触发
    - 定时器触发
    - 外部事件集成
    """
    
    def __init__(self, max_queue_size: int = 1000):
        self.listeners: Dict[str, EventListener] = {}
        self.event_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._running = False
        self._dispatcher_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._timers: List[threading.Timer] = []
    
    def add_listener(self, listener: EventListener) -> str:
        """添加事件监听器"""
        with self._lock:
            self.listeners[listener.id] = listener
        return listener.id
    
    def remove_listener(self, listener_id: str) -> bool:
        """移除事件监听器"""
        with self._lock:
            if listener_id in self.listeners:
                del self.listeners[listener_id]
                return True
            return False
    
    def emit(self, event: WorkflowEvent):
        """发出事件"""
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            logging.warning(f"事件队列已满，丢弃事件: {event.event_id}")
    
    def start(self):
        """启动事件分发器"""
        if self._running:
            return
        
        self._running = True
        self._dispatcher_thread = threading.Thread(target=self._dispatch_loop)
        self._dispatcher_thread.daemon = True
        self._dispatcher_thread.start()
    
    def stop(self):
        """停止事件分发器"""
        self._running = False
        if self._dispatcher_thread:
            self._dispatcher_thread.join(timeout=5)
        
        # 取消所有定时器
        for timer in self._timers:
            timer.cancel()
        self._timers.clear()
    
    def _dispatch_loop(self):
        """事件分发循环"""
        while self._running:
            try:
                event = self.event_queue.get(timeout=0.1)
                self._dispatch_event(event)
            except queue.Empty:
                continue
    
    def _dispatch_event(self, event: WorkflowEvent):
        """分发事件到监听器"""
        with self._lock:
            listeners_copy = list(self.listeners.values())
        
        for listener in listeners_copy:
            if listener.should_handle(event):
                try:
                    listener.callback(event)
                except Exception as e:
                    logging.error(f"事件监听器处理失败: {e}")
    
    def schedule_timer(self, delay_seconds: float, 
                       callback: Callable[[], None],
                       repeat: bool = False,
                       interval: float = 0) -> threading.Timer:
        """
        调度定时器
        
        参数:
            delay_seconds: 延迟时间(秒)
            callback: 回调函数
            repeat: 是否重复
            interval: 重复间隔(秒)
        """
        def timer_callback():
            callback()
            if repeat:
                timer = threading.Timer(interval, timer_callback)
                timer.daemon = True
                timer.start()
                self._timers.append(timer)
        
        timer = threading.Timer(delay_seconds, timer_callback)
        timer.daemon = True
        timer.start()
        self._timers.append(timer)
        return timer
    
    def create_trigger(self, condition: Callable[[], bool],
                       action: Callable[[], None],
                       check_interval: float = 1.0) -> threading.Timer:
        """
        创建条件触发器
        
        当条件满足时执行动作
        """
        def check_and_trigger():
            if condition():
                action()
            else:
                # 继续检查
                timer = threading.Timer(check_interval, check_and_trigger)
                timer.daemon = True
                timer.start()
                self._timers.append(timer)
        
        timer = threading.Timer(check_interval, check_and_trigger)
        timer.daemon = True
        timer.start()
        self._timers.append(timer)
        return timer


# ============================================================================
# 资源管理器 - ResourceManager
# ============================================================================

@dataclass
class ResourceAllocation:
    """资源分配记录"""
    task_id: str
    cpu_cores: float
    memory_mb: int
    gpu_memory_mb: int
    allocated_at: float


class ResourceManager:
    """
    资源管理器
    
    管理:
    - CPU分配
    - 内存限制
    - GPU分配
    - 资源池
    """
    
    def __init__(self, 
                 total_cpu_cores: float = 8.0,
                 total_memory_mb: int = 16384,
                 total_gpu_memory_mb: int = 8192):
        self.total_cpu_cores = total_cpu_cores
        self.total_memory_mb = total_memory_mb
        self.total_gpu_memory_mb = total_gpu_memory_mb
        
        self.available_cpu = total_cpu_cores
        self.available_memory = total_memory_mb
        self.available_gpu_memory = total_gpu_memory_mb
        
        self.allocations: Dict[str, ResourceAllocation] = {}
        self._lock = threading.RLock()
        self._wait_queue: List[Tuple[str, ResourceRequirements, threading.Event]] = []
    
    def can_allocate(self, requirements: ResourceRequirements) -> bool:
        """检查是否可以分配资源"""
        with self._lock:
            return (
                self.available_cpu >= requirements.cpu_cores and
                self.available_memory >= requirements.memory_mb and
                self.available_gpu_memory >= requirements.gpu_memory_mb
            )
    
    def allocate(self, task_id: str, 
                 requirements: ResourceRequirements) -> Optional[ResourceAllocation]:
        """
        分配资源
        
        返回: 分配记录，如果资源不足返回None
        """
        with self._lock:
            if not self.can_allocate(requirements):
                return None
            
            self.available_cpu -= requirements.cpu_cores
            self.available_memory -= requirements.memory_mb
            self.available_gpu_memory -= requirements.gpu_memory_mb
            
            allocation = ResourceAllocation(
                task_id=task_id,
                cpu_cores=requirements.cpu_cores,
                memory_mb=requirements.memory_mb,
                gpu_memory_mb=requirements.gpu_memory_mb,
                allocated_at=time.time()
            )
            self.allocations[task_id] = allocation
            return allocation
    
    def release(self, task_id: str) -> bool:
        """释放资源"""
        with self._lock:
            if task_id not in self.allocations:
                return False
            
            allocation = self.allocations.pop(task_id)
            self.available_cpu += allocation.cpu_cores
            self.available_memory += allocation.memory_mb
            self.available_gpu_memory += allocation.gpu_memory_mb
            
            # 检查等待队列
            self._process_wait_queue()
            return True
    
    def _process_wait_queue(self):
        """处理等待队列"""
        satisfied = []
        for i, (task_id, requirements, event) in enumerate(self._wait_queue):
            if self.can_allocate(requirements):
                allocation = self.allocate(task_id, requirements)
                if allocation:
                    event.set()
                    satisfied.append(i)
        
        # 移除已满足的请求
        for i in reversed(satisfied):
            self._wait_queue.pop(i)
    
    def allocate_blocking(self, task_id: str, 
                         requirements: ResourceRequirements,
                         timeout: Optional[float] = None) -> Optional[ResourceAllocation]:
        """
        阻塞式资源分配
        
        如果资源不足，等待直到资源可用或超时
        """
        # 先尝试非阻塞分配
        allocation = self.allocate(task_id, requirements)
        if allocation:
            return allocation
        
        # 加入等待队列
        event = threading.Event()
        with self._lock:
            self._wait_queue.append((task_id, requirements, event))
        
        # 等待资源可用
        if event.wait(timeout=timeout):
            return self.allocations.get(task_id)
        else:
            # 超时，从等待队列移除
            with self._lock:
                self._wait_queue = [
                    (tid, req, evt) for tid, req, evt in self._wait_queue 
                    if tid != task_id
                ]
            return None
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """获取资源使用统计"""
        with self._lock:
            return {
                'cpu': {
                    'total': self.total_cpu_cores,
                    'available': self.available_cpu,
                    'used': self.total_cpu_cores - self.available_cpu,
                    'usage_percent': ((self.total_cpu_cores - self.available_cpu) / 
                                     self.total_cpu_cores * 100)
                },
                'memory': {
                    'total_mb': self.total_memory_mb,
                    'available_mb': self.available_memory,
                    'used_mb': self.total_memory_mb - self.available_memory,
                    'usage_percent': ((self.total_memory_mb - self.available_memory) / 
                                     self.total_memory_mb * 100)
                },
                'gpu_memory': {
                    'total_mb': self.total_gpu_memory_mb,
                    'available_mb': self.available_gpu_memory,
                    'used_mb': self.total_gpu_memory_mb - self.available_gpu_memory,
                    'usage_percent': ((self.total_gpu_memory_mb - self.available_gpu_memory) / 
                                     self.total_gpu_memory_mb * 100) if self.total_gpu_memory_mb > 0 else 0
                },
                'active_allocations': len(self.allocations),
                'wait_queue_length': len(self._wait_queue)
            }


# ============================================================================
# 监控模块 - Monitoring
# ============================================================================

@dataclass
class TaskMetrics:
    """任务执行指标"""
    task_id: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: float = 0.0
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    retry_count: int = 0


@dataclass
class Alert:
    """告警"""
    alert_id: str
    level: str  # INFO, WARNING, ERROR, CRITICAL
    message: str
    timestamp: float
    source: str
    data: Dict[str, Any]


class Monitoring:
    """
    工作流监控模块
    
    功能:
    - 执行指标收集
    - 延迟追踪
    - 吞吐量测量
    - 告警生成
    """
    
    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.task_metrics: Dict[str, TaskMetrics] = {}
        self.workflow_start_time: Optional[float] = None
        self.workflow_end_time: Optional[float] = None
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.alerts: List[Alert] = []
        self._lock = threading.RLock()
        self._alert_handlers: List[Callable[[Alert], None]] = []
    
    def record_task_start(self, task_id: str):
        """记录任务开始"""
        with self._lock:
            self.task_metrics[task_id] = TaskMetrics(
                task_id=task_id,
                start_time=time.time()
            )
    
    def record_task_end(self, task_id: str, success: bool = True):
        """记录任务结束"""
        with self._lock:
            if task_id in self.task_metrics:
                metrics = self.task_metrics[task_id]
                metrics.end_time = time.time()
                if metrics.start_time:
                    metrics.duration = metrics.end_time - metrics.start_time
                
                if success:
                    self.completed_tasks += 1
                else:
                    self.failed_tasks += 1
    
    def record_task_retry(self, task_id: str):
        """记录任务重试"""
        with self._lock:
            if task_id in self.task_metrics:
                self.task_metrics[task_id].retry_count += 1
    
    def start_workflow(self):
        """开始工作流监控"""
        self.workflow_start_time = time.time()
        self.completed_tasks = 0
        self.failed_tasks = 0
    
    def end_workflow(self):
        """结束工作流监控"""
        self.workflow_end_time = time.time()
    
    def get_latency_stats(self) -> Dict[str, float]:
        """获取延迟统计"""
        with self._lock:
            durations = [m.duration for m in self.task_metrics.values() 
                        if m.duration > 0]
            if not durations:
                return {}
            
            durations.sort()
            n = len(durations)
            
            return {
                'min': durations[0],
                'max': durations[-1],
                'mean': sum(durations) / n,
                'median': durations[n // 2],
                'p95': durations[int(n * 0.95)],
                'p99': durations[int(n * 0.99)]
            }
    
    def get_throughput(self) -> float:
        """获取吞吐量(任务/秒)"""
        with self._lock:
            if not self.workflow_start_time:
                return 0.0
            
            end_time = self.workflow_end_time or time.time()
            duration = end_time - self.workflow_start_time
            
            if duration <= 0:
                return 0.0
            
            total_tasks = self.completed_tasks + self.failed_tasks
            return total_tasks / duration
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """添加告警处理器"""
        self._alert_handlers.append(handler)
    
    def generate_alert(self, level: str, message: str, 
                      source: str, data: Optional[Dict[str, Any]] = None):
        """生成告警"""
        alert = Alert(
            alert_id=str(uuid.uuid4())[:8],
            level=level,
            message=message,
            timestamp=time.time(),
            source=source,
            data=data or {}
        )
        
        with self._lock:
            self.alerts.append(alert)
        
        # 调用告警处理器
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logging.error(f"告警处理器执行失败: {e}")
    
    def check_slo(self, task_id: str, max_duration: float):
        """检查服务等级目标(SLO)"""
        with self._lock:
            if task_id not in self.task_metrics:
                return
            
            metrics = self.task_metrics[task_id]
            if metrics.duration > max_duration:
                self.generate_alert(
                    level='WARNING',
                    message=f'任务 {task_id} 执行时间 {metrics.duration:.2f}s 超过SLO {max_duration:.2f}s',
                    source='slo_checker',
                    data={'task_id': task_id, 'duration': metrics.duration, 'slo': max_duration}
                )
    
    def get_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        with self._lock:
            workflow_duration = 0.0
            if self.workflow_start_time:
                end = self.workflow_end_time or time.time()
                workflow_duration = end - self.workflow_start_time
            
            return {
                'workflow_duration': workflow_duration,
                'completed_tasks': self.completed_tasks,
                'failed_tasks': self.failed_tasks,
                'success_rate': (self.completed_tasks / 
                               (self.completed_tasks + self.failed_tasks) * 100
                               if (self.completed_tasks + self.failed_tasks) > 0 else 0),
                'throughput': self.get_throughput(),
                'latency_stats': self.get_latency_stats(),
                'alert_count': len(self.alerts)
            }


# ============================================================================
# 工作流调度器 - WorkflowScheduler
# ============================================================================

class SchedulingPolicy(Enum):
    """调度策略"""
    ASAP = auto()          # As Soon As Possible
    ALAP = auto()          # As Late As Possible
    PRIORITY = auto()      # 基于优先级
    FAIR_SHARE = auto()    # 公平共享
    RESOURCE_CONSTRAINED = auto()  # 资源约束


class WorkflowScheduler:
    """
    工作流调度器
    
    实现调度算法:
    - ASAP (As Soon As Possible)
    - ALAP (As Late As Possible)
    - 资源约束调度
    - 基于优先级调度
    - 公平共享调度
    """
    
    def __init__(self, dag: WorkflowDAG, 
                 policy: SchedulingPolicy = SchedulingPolicy.ASAP,
                 resource_manager: Optional[ResourceManager] = None):
        self.dag = dag
        self.policy = policy
        self.resource_manager = resource_manager
        self._schedule: Dict[str, float] = {}  # task_id -> start_time
    
    def schedule(self) -> Dict[str, float]:
        """执行调度"""
        if self.policy == SchedulingPolicy.ASAP:
            return self._schedule_asap()
        elif self.policy == SchedulingPolicy.ALAP:
            return self._schedule_alap()
        elif self.policy == SchedulingPolicy.PRIORITY:
            return self._schedule_priority()
        elif self.policy == SchedulingPolicy.FAIR_SHARE:
            return self._schedule_fair_share()
        elif self.policy == SchedulingPolicy.RESOURCE_CONSTRAINED:
            return self._schedule_resource_constrained()
        else:
            return self._schedule_asap()
    
    def _schedule_asap(self) -> Dict[str, float]:
        """
        ASAP调度 - 尽可能早地开始任务
        """
        schedule = {}
        topo_order = self.dag.topological_sort()
        
        for task_id in topo_order:
            # 计算最早开始时间(所有前置任务完成后)
            earliest_start = 0.0
            for pred_id in self.dag.reverse_edges[task_id]:
                if pred_id in schedule:
                    pred_end = schedule[pred_id] + self.dag.nodes[pred_id].timeout
                    earliest_start = max(earliest_start, pred_end)
            
            schedule[task_id] = earliest_start
        
        self._schedule = schedule
        return schedule
    
    def _schedule_alap(self) -> Dict[str, float]:
        """
        ALAP调度 - 尽可能晚地开始任务
        """
        # 首先计算ASAP调度得到项目总工期
        asap_schedule = self._schedule_asap()
        project_duration = max(
            asap_schedule[tid] + self.dag.nodes[tid].timeout 
            for tid in asap_schedule
        ) if asap_schedule else 0
        
        schedule = {}
        topo_order = self.dag.topological_sort()
        
        # 反向遍历
        for task_id in reversed(topo_order):
            task = self.dag.nodes[task_id]
            
            if not self.dag.edges[task_id]:  # 叶子节点
                # 尽可能晚开始，但仍能在项目工期内完成
                schedule[task_id] = project_duration - task.timeout
            else:
                # 最晚开始时间取决于后继任务的最早开始时间
                latest_start = min(
                    schedule[succ_id] for succ_id in self.dag.edges[task_id]
                ) - task.timeout
                schedule[task_id] = latest_start
        
        self._schedule = schedule
        return schedule
    
    def _schedule_priority(self) -> Dict[str, float]:
        """
        基于优先级的调度
        优先级高的任务优先调度
        """
        schedule = {}
        remaining = set(self.dag.nodes.keys())
        completed = set()
        current_time = 0.0
        
        while remaining:
            # 找出所有就绪的任务
            ready = [
                tid for tid in remaining 
                if self.dag.reverse_edges[tid].issubset(completed)
            ]
            
            if not ready:
                break
            
            # 按优先级排序
            ready.sort(key=lambda tid: self.dag.nodes[tid].priority)
            
            # 调度优先级最高的任务
            task_id = ready[0]
            task = self.dag.nodes[task_id]
            
            # 计算实际开始时间(考虑依赖完成时间)
            earliest_start = current_time
            for pred_id in self.dag.reverse_edges[task_id]:
                if pred_id in schedule:
                    pred_end = schedule[pred_id] + self.dag.nodes[pred_id].timeout
                    earliest_start = max(earliest_start, pred_end)
            
            schedule[task_id] = earliest_start
            completed.add(task_id)
            remaining.remove(task_id)
            current_time = earliest_start
        
        self._schedule = schedule
        return schedule
    
    def _schedule_fair_share(self) -> Dict[str, float]:
        """
        公平共享调度
        确保每个任务获得公平的资源份额
        """
        schedule = {}
        remaining = set(self.dag.nodes.keys())
        completed = set()
        task_weights = {
            tid: 1.0 / self.dag.nodes[tid].priority 
            for tid in self.dag.nodes
        }
        total_weight = sum(task_weights.values())
        
        current_time = 0.0
        
        while remaining:
            ready = [
                tid for tid in remaining 
                if self.dag.reverse_edges[tid].issubset(completed)
            ]
            
            if not ready:
                break
            
            # 按权重比例分配时间片
            ready_weights = {tid: task_weights[tid] for tid in ready}
            total_ready_weight = sum(ready_weights.values())
            
            # 选择权重最高的任务
            task_id = max(ready, key=lambda tid: ready_weights[tid])
            task = self.dag.nodes[task_id]
            
            earliest_start = current_time
            for pred_id in self.dag.reverse_edges[task_id]:
                if pred_id in schedule:
                    pred_end = schedule[pred_id] + self.dag.nodes[pred_id].timeout
                    earliest_start = max(earliest_start, pred_end)
            
            schedule[task_id] = earliest_start
            completed.add(task_id)
            remaining.remove(task_id)
        
        self._schedule = schedule
        return schedule
    
    def _schedule_resource_constrained(self) -> Dict[str, float]:
        """
        资源约束调度
        考虑资源限制进行调度
        """
        if not self.resource_manager:
            return self._schedule_asap()
        
        schedule = {}
        remaining = set(self.dag.nodes.keys())
        completed = set()
        current_time = 0.0
        
        # 跟踪资源使用
        resource_timeline: List[Tuple[float, str, ResourceRequirements]] = []
        
        while remaining:
            # 找出所有就绪且资源可满足的任务
            ready = []
            for tid in remaining:
                if self.dag.reverse_edges[tid].issubset(completed):
                    task = self.dag.nodes[tid]
                    if self.resource_manager.can_allocate(task.resources):
                        ready.append(tid)
            
            if ready:
                # 按优先级选择任务
                ready.sort(key=lambda tid: self.dag.nodes[tid].priority)
                task_id = ready[0]
                task = self.dag.nodes[task_id]
                
                # 分配资源
                self.resource_manager.allocate(task_id, task.resources)
                
                earliest_start = current_time
                for pred_id in self.dag.reverse_edges[task_id]:
                    if pred_id in schedule:
                        pred_end = schedule[pred_id] + self.dag.nodes[pred_id].timeout
                        earliest_start = max(earliest_start, pred_end)
                
                schedule[task_id] = earliest_start
                resource_timeline.append((earliest_start, task_id, task.resources))
                completed.add(task_id)
                remaining.remove(task_id)
            else:
                # 资源不足，推进时间
                current_time += 1.0
                # 释放已完成的任务资源
                for start_time, tid, res in resource_timeline[:]:
                    task = self.dag.nodes[tid]
                    if start_time + task.timeout <= current_time:
                        self.resource_manager.release(tid)
        
        self._schedule = schedule
        return schedule
    
    def get_makespan(self) -> float:
        """获取调度总工期"""
        if not self._schedule:
            return 0.0
        
        return max(
            self._schedule[tid] + self.dag.nodes[tid].timeout 
            for tid in self._schedule
        )


# ============================================================================
# 工作流执行器 - WorkflowExecutor
# ============================================================================

class ExecutionMode(Enum):
    """执行模式"""
    SEQUENTIAL = auto()
    PARALLEL = auto()
    ASYNC = auto()


class WorkflowExecutor:
    """
    工作流执行引擎
    
    支持:
    - 顺序执行
    - 并行执行(线程池模拟)
    - 异步执行与回调
    - 错误处理和重试
    - 检查点和恢复
    """
    
    def __init__(self, 
                 dag: WorkflowDAG,
                 data_flow: DataFlow,
                 resource_manager: Optional[ResourceManager] = None,
                 monitoring: Optional[Monitoring] = None,
                 event_system: Optional[EventSystem] = None,
                 config: Optional[WorkflowConfig] = None):
        self.dag = dag
        self.data_flow = data_flow
        self.resource_manager = resource_manager
        self.monitoring = monitoring
        self.event_system = event_system
        self.config = config or WorkflowConfig()
        
        self._execution_state: Dict[str, Any] = {}
        self._checkpoints: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._stop_requested = False
    
    def execute(self, mode: ExecutionMode = ExecutionMode.PARALLEL,
                max_workers: Optional[int] = None) -> Dict[str, Any]:
        """
        执行工作流
        
        参数:
            mode: 执行模式
            max_workers: 最大工作线程数
        
        返回:
            执行结果字典
        """
        if mode == ExecutionMode.SEQUENTIAL:
            return self._execute_sequential()
        elif mode == ExecutionMode.PARALLEL:
            return self._execute_parallel(max_workers)
        elif mode == ExecutionMode.ASYNC:
            return self._execute_async(max_workers)
        else:
            return self._execute_parallel(max_workers)
    
    def _execute_task(self, task: TaskNode, 
                     input_data: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        执行单个任务
        
        返回: (是否成功, 结果或异常)
        """
        if self.monitoring:
            self.monitoring.record_task_start(task.task_id)
        
        if self.event_system:
            self.event_system.emit(WorkflowEvent(
                event_type=EventType.TASK_STARTED,
                source=task.task_id,
                timestamp=time.time(),
                data={'inputs': list(input_data.keys())}
            ))
        
        task.state = TaskState.RUNNING
        task.start_time = time.time()
        
        try:
            # 分配资源
            allocation = None
            if self.resource_manager:
                allocation = self.resource_manager.allocate_blocking(
                    task.task_id, task.resources, timeout=task.timeout
                )
                if not allocation:
                    raise TimeoutError(f"无法为任务 {task.task_id} 分配资源")
            
            # 执行函数
            if task.func:
                result = task.func(**input_data)
            else:
                result = None
            
            # 释放资源
            if self.resource_manager:
                self.resource_manager.release(task.task_id)
            
            task.state = TaskState.COMPLETED
            task.end_time = time.time()
            task.result = result
            
            if self.monitoring:
                self.monitoring.record_task_end(task.task_id, success=True)
            
            if self.event_system:
                self.event_system.emit(WorkflowEvent(
                    event_type=EventType.TASK_COMPLETED,
                    source=task.task_id,
                    timestamp=time.time(),
                    data={'duration': task.get_execution_time()}
                ))
            
            return True, result
            
        except Exception as e:
            task.state = TaskState.FAILED
            task.end_time = time.time()
            task.error_message = str(e)
            
            if self.monitoring:
                self.monitoring.record_task_end(task.task_id, success=False)
            
            if self.event_system:
                self.event_system.emit(WorkflowEvent(
                    event_type=EventType.TASK_FAILED,
                    source=task.task_id,
                    timestamp=time.time(),
                    data={'error': str(e)}
                ))
            
            return False, e
    
    def _execute_with_retry(self, task: TaskNode, 
                           input_data: Dict[str, Any]) -> Tuple[bool, Any]:
        """带重试的任务执行"""
        policy = task.retry_policy
        last_error = None
        
        for attempt in range(policy.max_retries + 1):
            task.attempt_count = attempt
            success, result = self._execute_task(task, input_data)
            
            if success:
                return True, result
            
            if not isinstance(result, tuple(policy.retry_exceptions)):
                return False, result
            
            last_error = result
            
            if attempt < policy.max_retries:
                if self.monitoring:
                    self.monitoring.record_task_retry(task.task_id)
                
                delay = policy.calculate_delay(attempt)
                time.sleep(delay)
        
        return False, last_error
    
    def _execute_sequential(self) -> Dict[str, Any]:
        """顺序执行"""
        topo_order = self.dag.topological_sort()
        results = {}
        
        for task_id in topo_order:
            if self._stop_requested:
                break
            
            task = self.dag.nodes[task_id]
            
            # 准备输入
            input_data = self._prepare_task_inputs(task, results)
            
            # 执行任务
            success, result = self._execute_with_retry(task, input_data)
            
            if success:
                results[task_id] = result
                # 存储输出到数据流
                self.data_flow.store(task_id, 'output', result)
            else:
                results[task_id] = result
                # 失败处理：继续或停止
                if not self._should_continue_on_failure(task):
                    break
        
        return results
    
    def _execute_parallel(self, max_workers: Optional[int] = None) -> Dict[str, Any]:
        """并行执行"""
        max_workers = max_workers or self.config.get('max_workers', 4)
        parallel_groups = self.dag.get_parallel_groups()
        results = {}
        
        for group in parallel_groups:
            if self._stop_requested:
                break
            
            # 并行执行当前组
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                
                for task_id in group:
                    task = self.dag.nodes[task_id]
                    input_data = self._prepare_task_inputs(task, results)
                    
                    future = executor.submit(self._execute_with_retry, task, input_data)
                    futures[future] = task_id
                
                # 收集结果
                for future in as_completed(futures):
                    task_id = futures[future]
                    try:
                        success, result = future.result()
                        if success:
                            results[task_id] = result
                            self.data_flow.store(task_id, 'output', result)
                        else:
                            results[task_id] = result
                            if not self._should_continue_on_failure(self.dag.nodes[task_id]):
                                self._stop_requested = True
                                break
                    except Exception as e:
                        results[task_id] = e
        
        return results
    
    def _execute_async(self, max_workers: Optional[int] = None) -> Dict[str, Any]:
        """异步执行(返回Future字典)"""
        max_workers = max_workers or self.config.get('max_workers', 4)
        
        executor = ThreadPoolExecutor(max_workers=max_workers)
        futures: Dict[str, Future] = {}
        results: Dict[str, Any] = {}
        completed = set()
        
        def submit_ready_tasks():
            """提交所有就绪的任务"""
            for task_id in self.dag.nodes:
                if (task_id not in futures and 
                    task_id not in completed and
                    self.dag.nodes[task_id].is_ready(completed)):
                    task = self.dag.nodes[task_id]
                    input_data = self._prepare_task_inputs(task, results)
                    future = executor.submit(self._execute_with_retry, task, input_data)
                    futures[task_id] = future
        
        def process_completed():
            """处理已完成的任务"""
            done_tasks = []
            for task_id, future in list(futures.items()):
                if future.done():
                    try:
                        success, result = future.result()
                        if success:
                            results[task_id] = result
                            self.data_flow.store(task_id, 'output', result)
                        else:
                            results[task_id] = result
                        completed.add(task_id)
                        done_tasks.append(task_id)
                    except Exception as e:
                        results[task_id] = e
                        completed.add(task_id)
                        done_tasks.append(task_id)
            
            for task_id in done_tasks:
                del futures[task_id]
        
        # 主循环
        while len(completed) < len(self.dag.nodes):
            if self._stop_requested:
                break
            
            submit_ready_tasks()
            process_completed()
            
            if len(futures) == 0 and len(completed) < len(self.dag.nodes):
                # 没有可执行的任务但还有未完成的任务，可能是死锁
                break
            
            time.sleep(0.01)
        
        executor.shutdown()
        return results
    
    def _prepare_task_inputs(self, task: TaskNode, 
                            results: Dict[str, Any]) -> Dict[str, Any]:
        """准备任务输入数据"""
        input_data = {}
        
        # 从上游任务获取输入
        for upstream_id in task.upstream:
            if upstream_id in results:
                input_data[f"from_{upstream_id}"] = results[upstream_id]
        
        # 合并任务定义的输入
        input_data.update(task.inputs)
        
        return input_data
    
    def _should_continue_on_failure(self, task: TaskNode) -> bool:
        """判断任务失败后是否继续执行"""
        # 可以根据任务配置决定是否继续
        return task.metadata.get('continue_on_failure', False)
    
    def create_checkpoint(self, name: str) -> str:
        """
        创建检查点
        
        返回: 检查点ID
        """
        checkpoint_id = str(uuid.uuid4())[:8]
        
        with self._lock:
            self._checkpoints[checkpoint_id] = {
                'name': name,
                'timestamp': time.time(),
                'node_states': {
                    tid: {
                        'state': node.state.name,
                        'result': node.result,
                        'attempt_count': node.attempt_count
                    }
                    for tid, node in self.dag.nodes.items()
                },
                'data_store': dict(self.data_flow._data_store)
            }
        
        return checkpoint_id
    
    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """从检查点恢复"""
        with self._lock:
            if checkpoint_id not in self._checkpoints:
                return False
            
            checkpoint = self._checkpoints[checkpoint_id]
            
            # 恢复节点状态
            for tid, state_data in checkpoint['node_states'].items():
                if tid in self.dag.nodes:
                    node = self.dag.nodes[tid]
                    node.state = TaskState[state_data['state']]
                    node.result = state_data['result']
                    node.attempt_count = state_data['attempt_count']
            
            # 恢复数据流
            self.data_flow._data_store = dict(checkpoint['data_store'])
            
            return True
    
    def stop(self):
        """请求停止执行"""
        self._stop_requested = True


# ============================================================================
# 主编排引擎 - WorkflowEngine
# ============================================================================

class WorkflowEngine:
    """
    工作流主编排引擎
    
    功能:
    - 工作流定义加载
    - 工作流验证
    - 工作流执行
    - 结果收集
    """
    
    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.config = config or WorkflowConfig()
        self.dag: Optional[WorkflowDAG] = None
        self.data_flow = DataFlow(enable_lineage=self.config.get('enable_data_lineage', True))
        self.resource_manager = ResourceManager()
        self.monitoring = Monitoring(self.config)
        self.event_system = EventSystem(self.config.get('event_queue_size', 1000))
        self.scheduler: Optional[WorkflowScheduler] = None
        self.executor: Optional[WorkflowExecutor] = None
        
        self._is_running = False
        self._results: Dict[str, Any] = {}
    
    def load_workflow(self, workflow_def: Union[str, Dict[str, Any], WorkflowDAG]) -> 'WorkflowEngine':
        """
        加载工作流定义
        
        支持:
        - JSON文件路径
        - 字典定义
        - WorkflowDAG对象
        """
        if isinstance(workflow_def, WorkflowDAG):
            self.dag = workflow_def
        elif isinstance(workflow_def, str):
            with open(workflow_def, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.dag = self._parse_workflow_dict(data)
        elif isinstance(workflow_def, dict):
            self.dag = self._parse_workflow_dict(workflow_def)
        else:
            raise ValueError(f"不支持的工作流定义类型: {type(workflow_def)}")
        
        return self
    
    def _parse_workflow_dict(self, data: Dict[str, Any]) -> WorkflowDAG:
        """从字典解析工作流"""
        dag = WorkflowDAG(name=data.get('name', 'workflow'))
        
        # 创建节点
        for node_data in data.get('nodes', []):
            task = TaskNode(
                task_id=node_data.get('id', str(uuid.uuid4())[:8]),
                name=node_data.get('name', 'unnamed'),
                inputs=node_data.get('inputs', {}),
                outputs=node_data.get('outputs', {}),
                dependencies=node_data.get('dependencies', []),
                timeout=node_data.get('timeout', 300),
                priority=node_data.get('priority', 10),
                metadata=node_data.get('metadata', {})
            )
            dag.add_node(task)
        
        # 添加边
        for edge in data.get('edges', []):
            dag.add_edge(edge['from'], edge['to'])
        
        return dag
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        验证工作流
        
        返回: (是否有效, 错误信息列表)
        """
        errors = []
        
        if not self.dag:
            errors.append("工作流DAG未加载")
            return False, errors
        
        # 检查环
        cycle = self.dag.detect_cycle()
        if cycle:
            errors.append(f"检测到环: {' -> '.join(cycle)}")
        
        # 检查孤立节点
        if len(self.dag.nodes) > 1:
            independent = self.dag.get_independent_nodes()
            leaves = self.dag.get_leaf_nodes()
            if len(independent) == len(self.dag.nodes):
                errors.append("警告: 所有节点都是独立的，没有依赖关系")
        
        # 检查资源需求
        for task_id, task in self.dag.nodes.items():
            if task.resources.cpu_cores > self.resource_manager.total_cpu_cores:
                errors.append(f"任务 {task_id} 请求的CPU核心数超过可用资源")
            if task.resources.memory_mb > self.resource_manager.total_memory_mb:
                errors.append(f"任务 {task_id} 请求的内存超过可用资源")
        
        return len(errors) == 0, errors
    
    def run(self, 
            mode: ExecutionMode = ExecutionMode.PARALLEL,
            scheduling_policy: SchedulingPolicy = SchedulingPolicy.ASAP,
            max_workers: Optional[int] = None) -> Dict[str, Any]:
        """
        运行工作流
        
        参数:
            mode: 执行模式
            scheduling_policy: 调度策略
            max_workers: 最大工作线程数
        
        返回:
            执行结果
        """
        # 验证
        valid, errors = self.validate()
        if not valid:
            raise ValueError(f"工作流验证失败: {errors}")
        
        self._is_running = True
        self.monitoring.start_workflow()
        
        if self.event_system:
            self.event_system.start()
            self.event_system.emit(WorkflowEvent(
                event_type=EventType.WORKFLOW_STARTED,
                source='workflow_engine',
                timestamp=time.time()
            ))
        
        try:
            # 调度
            self.scheduler = WorkflowScheduler(
                self.dag, 
                policy=scheduling_policy,
                resource_manager=self.resource_manager
            )
            schedule = self.scheduler.schedule()
            self.config.logger.info(f"调度完成，预计工期: {self.scheduler.get_makespan():.2f}s")
            
            # 执行
            self.executor = WorkflowExecutor(
                self.dag,
                self.data_flow,
                self.resource_manager,
                self.monitoring,
                self.event_system,
                self.config
            )
            
            self._results = self.executor.execute(mode, max_workers)
            
            return self._results
            
        except Exception as e:
            if self.event_system:
                self.event_system.emit(WorkflowEvent(
                    event_type=EventType.WORKFLOW_FAILED,
                    source='workflow_engine',
                    timestamp=time.time(),
                    data={'error': str(e)}
                ))
            raise
        finally:
            self._is_running = False
            self.monitoring.end_workflow()
            
            if self.event_system:
                self.event_system.emit(WorkflowEvent(
                    event_type=EventType.WORKFLOW_COMPLETED,
                    source='workflow_engine',
                    timestamp=time.time()
                ))
                self.event_system.stop()
    
    def get_results(self) -> Dict[str, Any]:
        """获取执行结果"""
        return self._results
    
    def get_task_result(self, task_id: str) -> Any:
        """获取特定任务的结果"""
        return self._results.get(task_id)
    
    def create_checkpoint(self, name: str = "manual") -> str:
        """创建工作流检查点"""
        if self.executor:
            return self.executor.create_checkpoint(name)
        return ""
    
    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """从检查点恢复"""
        if self.executor:
            return self.executor.restore_checkpoint(checkpoint_id)
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """获取工作流状态"""
        status = {
            'is_running': self._is_running,
            'monitoring': self.monitoring.get_summary() if self.monitoring else {},
            'resources': self.resource_manager.get_usage_stats() if self.resource_manager else {}
        }
        
        if self.dag:
            status['nodes'] = {
                tid: {
                    'state': node.state.name,
                    'progress': 100 if node.state == TaskState.COMPLETED else 
                               (50 if node.state == TaskState.RUNNING else 0)
                }
                for tid, node in self.dag.nodes.items()
            }
        
        return status
    
    def stop(self):
        """停止工作流执行"""
        if self.executor:
            self.executor.stop()
        self._is_running = False
    
    def export_definition(self, filepath: str):
        """导出工作流定义到文件"""
        if not self.dag:
            raise ValueError("工作流未加载")
        
        definition = self.dag.to_dict()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(definition, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def create_simple_workflow(name: str, 
                               tasks: List[Tuple[str, Callable, List[str]]]) -> WorkflowDAG:
        """
        创建简单工作流
        
        参数:
            name: 工作流名称
            tasks: [(task_name, function, dependencies), ...]
        
        返回:
            WorkflowDAG对象
        """
        dag = WorkflowDAG(name=name)
        task_map = {}
        
        # 第一遍：创建所有节点
        for task_name, func, _ in tasks:
            task = TaskNode(
                name=task_name,
                func=func
            )
            dag.add_node(task)
            task_map[task_name] = task.task_id
        
        # 第二遍：添加依赖
        for task_name, _, deps in tasks:
            task_id = task_map[task_name]
            for dep_name in deps:
                if dep_name in task_map:
                    dag.add_edge(task_map[dep_name], task_id)
        
        return dag


# ============================================================================
# 示例用法
# ============================================================================

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 示例1: 创建简单工作流
    print("=" * 50)
    print("示例1: 简单数据处理工作流")
    print("=" * 50)
    
    def extract_data():
        time.sleep(0.1)
        return {"data": [1, 2, 3, 4, 5]}
    
    def transform_data(**kwargs):
        time.sleep(0.1)
        # 获取上游任务的输出
        upstream_data = None
        for key, value in kwargs.items():
            if key.startswith("from_"):
                upstream_data = value
                break
        data = upstream_data or {"data": []}
        return {"transformed": [x * 2 for x in data.get("data", [])]}
    
    def load_data(**kwargs):
        time.sleep(0.1)
        upstream_data = None
        for key, value in kwargs.items():
            if key.startswith("from_"):
                upstream_data = value
                break
        data = upstream_data or {"transformed": []}
        return {"loaded": data.get("transformed", [])}
    
    # 创建工作流
    dag = WorkflowEngine.create_simple_workflow(
        "etl_pipeline",
        [
            ("extract", extract_data, []),
            ("transform", transform_data, ["extract"]),
            ("load", load_data, ["transform"])
        ]
    )
    
    # 运行工作流
    engine = WorkflowEngine()
    engine.load_workflow(dag)
    results = engine.run(mode=ExecutionMode.PARALLEL)
    
    print(f"执行结果: {results}")
    print(f"监控摘要: {engine.get_status()}")
    
    # 示例2: 使用DAG API
    print("\n" + "=" * 50)
    print("示例2: 使用DAG API构建复杂工作流")
    print("=" * 50)
    
    dag2 = WorkflowDAG("complex_workflow")
    
    # 创建任务节点
    task_a = TaskNode(name="TaskA", func=lambda **kwargs: "A", priority=1)
    task_b = TaskNode(name="TaskB", func=lambda **kwargs: "B", priority=2)
    task_c = TaskNode(name="TaskC", func=lambda **kwargs: "C", priority=2)
    task_d = TaskNode(name="TaskD", func=lambda **kwargs: f"D({kwargs})", priority=3)
    
    dag2.add_node(task_a)
    dag2.add_node(task_b)
    dag2.add_node(task_c)
    dag2.add_node(task_d)
    
    # 添加依赖: A -> B, A -> C, B -> D, C -> D
    dag2.add_edge(task_a.task_id, task_b.task_id)
    dag2.add_edge(task_a.task_id, task_c.task_id)
    dag2.add_edge(task_b.task_id, task_d.task_id)
    dag2.add_edge(task_c.task_id, task_d.task_id)
    
    # 拓扑排序
    print(f"拓扑排序: {dag2.topological_sort()}")
    
    # 并行组
    print(f"并行执行组: {dag2.get_parallel_groups()}")
    
    # 关键路径
    critical_path, duration = dag2.calculate_critical_path()
    print(f"关键路径: {critical_path}, 总工期: {duration:.2f}s")
    
    # 示例3: 条件分支
    print("\n" + "=" * 50)
    print("示例3: 条件分支")
    print("=" * 50)
    
    result = ConditionalBranch.if_else(
        condition=lambda: True,
        true_branch=lambda: "执行了true分支",
        false_branch=lambda: "执行了false分支"
    )
    print(f"If-else结果: {result}")
    
    # Switch-case
    result = ConditionalBranch.switch_case(
        value="b",
        cases={
            "a": lambda: "选择了A",
            "b": lambda: "选择了B",
            "c": lambda: "选择了C"
        },
        default=lambda: "默认选择"
    )
    print(f"Switch-case结果: {result}")
    
    # Fork-join
    results = ConditionalBranch.fork_join([
        lambda: "分支1结果",
        lambda: "分支2结果",
        lambda: "分支3结果"
    ])
    print(f"Fork-join结果: {results}")
    
    # 示例4: 调度器
    print("\n" + "=" * 50)
    print("示例4: 不同调度策略")
    print("=" * 50)
    
    for policy in SchedulingPolicy:
        scheduler = WorkflowScheduler(dag2, policy=policy)
        schedule = scheduler.schedule()
        print(f"{policy.name}调度: {schedule}")
    
    print("\n所有示例执行完成!")
