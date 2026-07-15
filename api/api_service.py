"""
API服务框架 - API Service Framework
提供RESTful API和gRPC服务接口，支持模型推理、训练、评估等功能
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import asyncio
import threading
import queue
import hashlib
import uuid
import os
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime
import warnings

# ==================== 请求/响应数据结构 ====================

@dataclass
class InferenceRequest:
    """推理请求"""
    request_id: str
    inputs: Union[str, List[str], np.ndarray, torch.Tensor]
    model_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class InferenceResponse:
    """推理响应"""
    request_id: str
    outputs: Any
    model_name: str
    latency_ms: float
    success: bool = True
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class TrainingRequest:
    """训练请求"""
    request_id: str
    model_name: str
    dataset_path: str
    config: Dict[str, Any]
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    callbacks: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class TrainingResponse:
    """训练响应"""
    request_id: str
    model_name: str
    status: str  # 'pending', 'running', 'completed', 'failed'
    progress: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    checkpoint_path: str = ""
    error_message: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    version: str
    framework: str
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    parameters: int = 0
    memory_mb: float = 0.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== 模型注册表 ====================

class ModelRegistry:
    """模型注册表"""
    
    def __init__(self):
        self.models: Dict[str, nn.Module] = {}
        self.model_info: Dict[str, ModelInfo] = {}
        self.model_configs: Dict[str, Dict] = {}
        self._lock = threading.Lock()
    
    def register(
        self,
        name: str,
        model: nn.Module,
        version: str = "1.0.0",
        framework: str = "pytorch",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """注册模型"""
        with self._lock:
            # 计算模型信息
            num_params = sum(p.numel() for p in model.parameters())
            memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            
            # 获取输入输出形状（如果可用）
            input_shape = None
            output_shape = None
            if hasattr(model, 'config'):
                config = model.config
                if hasattr(config, 'input_shape'):
                    input_shape = config.input_shape
                if hasattr(config, 'output_shape'):
                    output_shape = config.output_shape
            
            info = ModelInfo(
                name=name,
                version=version,
                framework=framework,
                input_shape=input_shape,
                output_shape=output_shape,
                parameters=num_params,
                memory_mb=memory_mb,
                tags=tags or [],
                metadata=metadata or {},
            )
            
            self.models[name] = model
            self.model_info[name] = info
            
            return name
    
    def unregister(self, name: str) -> bool:
        """注销模型"""
        with self._lock:
            if name in self.models:
                del self.models[name]
                del self.model_info[name]
                if name in self.model_configs:
                    del self.model_configs[name]
                return True
            return False
    
    def get(self, name: str) -> Optional[nn.Module]:
        """获取模型"""
        return self.models.get(name)
    
    def get_info(self, name: str) -> Optional[ModelInfo]:
        """获取模型信息"""
        return self.model_info.get(name)
    
    def list_models(self, tags: Optional[List[str]] = None) -> List[ModelInfo]:
        """列出模型"""
        models = list(self.model_info.values())
        
        if tags:
            models = [m for m in models if any(t in m.tags for t in tags)]
        
        return models
    
    def update_config(self, name: str, config: Dict) -> bool:
        """更新模型配置"""
        with self._lock:
            if name in self.models:
                self.model_configs[name] = config
                return True
            return False
    
    def get_config(self, name: str) -> Optional[Dict]:
        """获取模型配置"""
        return self.model_configs.get(name)


# ==================== 请求队列 ====================

class RequestQueue:
    """请求队列"""
    
    def __init__(self, max_size: int = 1000):
        self.queue = queue.Queue(maxsize=max_size)
        self.pending: Dict[str, InferenceRequest] = {}
        self.completed: Dict[str, InferenceResponse] = {}
        self._lock = threading.Lock()
    
    def submit(self, request: InferenceRequest) -> bool:
        """提交请求"""
        try:
            self.queue.put(request, block=False)
            with self._lock:
                self.pending[request.request_id] = request
            return True
        except queue.Full:
            return False
    
    def get(self, timeout: float = None) -> Optional[InferenceRequest]:
        """获取请求"""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def complete(self, response: InferenceResponse):
        """标记请求完成"""
        with self._lock:
            if response.request_id in self.pending:
                del self.pending[response.request_id]
            self.completed[response.request_id] = response
            
            # 限制已完成队列大小
            if len(self.completed) > 1000:
                # 删除最旧的
                oldest = min(self.completed.keys(), key=lambda k: self.completed[k].timestamp)
                del self.completed[oldest]
    
    def get_status(self, request_id: str) -> str:
        """获取请求状态"""
        with self._lock:
            if request_id in self.pending:
                return "pending"
            elif request_id in self.completed:
                return "completed"
            else:
                return "unknown"
    
    def get_response(self, request_id: str) -> Optional[InferenceResponse]:
        """获取响应"""
        return self.completed.get(request_id)
    
    def size(self) -> int:
        """获取队列大小"""
        return self.queue.qsize()


# ==================== 批处理器 ====================

class BatchProcessor:
    """批处理器 - 动态批处理"""
    
    def __init__(
        self,
        model: nn.Module,
        max_batch_size: int = 32,
        max_wait_ms: float = 50.0,
        device: str = "cuda",
    ):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.device = device
        
        self.request_queue = queue.Queue()
        self.response_queues: Dict[str, queue.Queue] = {}
        
        self.running = False
        self.process_thread = None
    
    def start(self):
        """启动批处理线程"""
        self.running = True
        self.process_thread = threading.Thread(target=self._process_loop)
        self.process_thread.daemon = True
        self.process_thread.start()
    
    def stop(self):
        """停止批处理线程"""
        self.running = False
        if self.process_thread:
            self.process_thread.join()
    
    def submit(self, inputs: Any, request_id: str) -> Any:
        """提交请求并等待结果"""
        response_queue = queue.Queue()
        self.response_queues[request_id] = response_queue
        
        self.request_queue.put((request_id, inputs))
        
        # 等待结果
        return response_queue.get()
    
    def _process_loop(self):
        """批处理循环"""
        while self.running:
            batch = []
            request_ids = []
            start_time = time.time()
            
            # 收集批次
            while len(batch) < self.max_batch_size:
                elapsed_ms = (time.time() - start_time) * 1000
                remaining_ms = max(0, self.max_wait_ms - elapsed_ms)
                
                try:
                    request_id, inputs = self.request_queue.get(timeout=remaining_ms / 1000)
                    batch.append(inputs)
                    request_ids.append(request_id)
                except queue.Empty:
                    break
            
            if not batch:
                continue
            
            # 执行批处理
            try:
                outputs = self._process_batch(batch)
                
                # 分发结果
                for request_id, output in zip(request_ids, outputs):
                    if request_id in self.response_queues:
                        self.response_queues[request_id].put(output)
                        del self.response_queues[request_id]
            except Exception as e:
                # 错误处理
                for request_id in request_ids:
                    if request_id in self.response_queues:
                        self.response_queues[request_id].put(Exception(str(e)))
                        del self.response_queues[request_id]
    
    def _process_batch(self, batch: List[Any]) -> List[Any]:
        """处理批次"""
        # 合并输入
        if isinstance(batch[0], torch.Tensor):
            batch_input = torch.stack(batch).to(self.device)
        elif isinstance(batch[0], np.ndarray):
            batch_input = torch.from_numpy(np.stack(batch)).to(self.device)
        else:
            # 文本输入
            batch_input = batch
        
        # 推理
        with torch.no_grad():
            outputs = self.model(batch_input)
        
        # 分离输出
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.cpu()
            return [outputs[i] for i in range(len(batch))]
        else:
            return outputs


# ==================== 推理引擎 ====================

class InferenceEngine:
    """推理引擎"""
    
    def __init__(
        self,
        model_registry: ModelRegistry,
        default_device: str = "cuda",
        enable_batching: bool = True,
        max_batch_size: int = 32,
    ):
        self.registry = model_registry
        self.default_device = default_device
        self.enable_batching = enable_batching
        self.max_batch_size = max_batch_size
        
        self.batch_processors: Dict[str, BatchProcessor] = {}
        self.request_queues: Dict[str, RequestQueue] = {}
        
        # 统计
        self.stats = defaultdict(lambda: defaultdict(float))
        self._lock = threading.Lock()
    
    def start_batching(self, model_name: str):
        """启动模型批处理"""
        model = self.registry.get(model_name)
        if model and self.enable_batching:
            processor = BatchProcessor(
                model, 
                max_batch_size=self.max_batch_size,
                device=self.default_device,
            )
            processor.start()
            self.batch_processors[model_name] = processor
    
    def stop_batching(self, model_name: str):
        """停止模型批处理"""
        if model_name in self.batch_processors:
            self.batch_processors[model_name].stop()
            del self.batch_processors[model_name]
    
    def inference(
        self,
        request: InferenceRequest,
    ) -> InferenceResponse:
        """执行推理"""
        start_time = time.time()
        
        try:
            model = self.registry.get(request.model_name)
            if model is None:
                return InferenceResponse(
                    request_id=request.request_id,
                    outputs=None,
                    model_name=request.model_name,
                    latency_ms=0,
                    success=False,
                    error_message=f"Model '{request.model_name}' not found",
                )
            
            # 准备输入
            inputs = self._prepare_inputs(request.inputs, request.parameters)
            
            # 执行推理
            if self.enable_batching and request.model_name in self.batch_processors:
                outputs = self.batch_processors[request.model_name].submit(
                    inputs, request.request_id
                )
            else:
                outputs = self._run_inference(model, inputs, request.parameters)
            
            # 处理输出
            outputs = self._process_outputs(outputs, request.parameters)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # 更新统计
            with self._lock:
                self.stats[request.model_name]['total_requests'] += 1
                self.stats[request.model_name]['total_latency_ms'] += latency_ms
            
            return InferenceResponse(
                request_id=request.request_id,
                outputs=outputs,
                model_name=request.model_name,
                latency_ms=latency_ms,
                success=True,
            )
        
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return InferenceResponse(
                request_id=request.request_id,
                outputs=None,
                model_name=request.model_name,
                latency_ms=latency_ms,
                success=False,
                error_message=str(e),
            )
    
    def _prepare_inputs(
        self,
        inputs: Any,
        parameters: Dict[str, Any],
    ) -> torch.Tensor:
        """准备输入"""
        device = parameters.get('device', self.default_device)
        
        if isinstance(inputs, torch.Tensor):
            return inputs.to(device)
        elif isinstance(inputs, np.ndarray):
            return torch.from_numpy(inputs).to(device)
        elif isinstance(inputs, str):
            # 文本输入需要tokenizer
            return inputs
        elif isinstance(inputs, list):
            if all(isinstance(x, str) for x in inputs):
                return inputs
            else:
                return torch.tensor(inputs).to(device)
        else:
            return torch.tensor(inputs).to(device)
    
    def _run_inference(
        self,
        model: nn.Module,
        inputs: Any,
        parameters: Dict[str, Any],
    ) -> Any:
        """运行推理"""
        model.eval()
        
        with torch.no_grad():
            # 检查是否有generate方法（用于语言模型）
            if hasattr(model, 'generate') and parameters.get('generate', False):
                return model.generate(
                    inputs,
                    max_length=parameters.get('max_length', 100),
                    temperature=parameters.get('temperature', 1.0),
                    top_p=parameters.get('top_p', 0.9),
                    do_sample=parameters.get('do_sample', True),
                )
            else:
                return model(inputs)
    
    def _process_outputs(
        self,
        outputs: Any,
        parameters: Dict[str, Any],
    ) -> Any:
        """处理输出"""
        if isinstance(outputs, torch.Tensor):
            if parameters.get('return_numpy', False):
                return outputs.cpu().numpy()
            elif parameters.get('return_list', False):
                return outputs.cpu().tolist()
            else:
                return outputs
        else:
            return outputs
    
    def get_stats(self, model_name: str = None) -> Dict[str, Dict[str, float]]:
        """获取统计信息"""
        with self._lock:
            if model_name:
                return {model_name: dict(self.stats[model_name])}
            else:
                return {k: dict(v) for k, v in self.stats.items()}


# ==================== 训练管理器 ====================

class TrainingManager:
    """训练管理器"""
    
    def __init__(self, model_registry: ModelRegistry):
        self.registry = model_registry
        self.training_jobs: Dict[str, TrainingResponse] = {}
        self.training_threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
    
    def submit_training(
        self,
        request: TrainingRequest,
        train_fn: Callable,
    ) -> TrainingResponse:
        """提交训练任务"""
        response = TrainingResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            status='pending',
        )
        
        with self._lock:
            self.training_jobs[request.request_id] = response
        
        # 启动训练线程
        thread = threading.Thread(
            target=self._run_training,
            args=(request, train_fn),
        )
        thread.daemon = True
        thread.start()
        
        self.training_threads[request.request_id] = thread
        
        return response
    
    def _run_training(
        self,
        request: TrainingRequest,
        train_fn: Callable,
    ):
        """运行训练"""
        with self._lock:
            self.training_jobs[request.request_id].status = 'running'
        
        try:
            # 执行训练
            result = train_fn(request)
            
            with self._lock:
                job = self.training_jobs[request.request_id]
                job.status = 'completed'
                job.progress = 1.0
                job.metrics = result.get('metrics', {})
                job.checkpoint_path = result.get('checkpoint_path', '')
        
        except Exception as e:
            with self._lock:
                job = self.training_jobs[request.request_id]
                job.status = 'failed'
                job.error_message = str(e)
    
    def get_status(self, request_id: str) -> Optional[TrainingResponse]:
        """获取训练状态"""
        with self._lock:
            return self.training_jobs.get(request_id)
    
    def cancel(self, request_id: str) -> bool:
        """取消训练"""
        with self._lock:
            if request_id in self.training_jobs:
                self.training_jobs[request_id].status = 'cancelled'
                return True
            return False
    
    def list_jobs(self, status: str = None) -> List[TrainingResponse]:
        """列出训练任务"""
        with self._lock:
            jobs = list(self.training_jobs.values())
            if status:
                jobs = [j for j in jobs if j.status == status]
            return jobs


# ==================== 缓存管理器 ====================

class CacheManager:
    """缓存管理器"""
    
    def __init__(
        self,
        max_size_mb: float = 1024,
        ttl_seconds: float = 3600,
    ):
        self.max_size_mb = max_size_mb
        self.ttl_seconds = ttl_seconds
        
        self.cache: Dict[str, Tuple[Any, float, float]] = {}  # key -> (value, size_bytes, timestamp)
        self._lock = threading.Lock()
    
    def _get_key(self, model_name: str, inputs: Any, parameters: Dict) -> str:
        """生成缓存键"""
        if isinstance(inputs, torch.Tensor):
            inputs_hash = hashlib.md5(inputs.cpu().numpy().tobytes()).hexdigest()
        elif isinstance(inputs, np.ndarray):
            inputs_hash = hashlib.md5(inputs.tobytes()).hexdigest()
        else:
            inputs_hash = hashlib.md5(str(inputs).encode()).hexdigest()
        
        params_hash = hashlib.md5(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
        
        return f"{model_name}:{inputs_hash}:{params_hash}"
    
    def get(
        self,
        model_name: str,
        inputs: Any,
        parameters: Dict,
    ) -> Optional[Any]:
        """获取缓存"""
        key = self._get_key(model_name, inputs, parameters)
        
        with self._lock:
            if key in self.cache:
                value, size_bytes, timestamp = self.cache[key]
                
                # 检查TTL
                if time.time() - timestamp < self.ttl_seconds:
                    return value
                else:
                    del self.cache[key]
        
        return None
    
    def put(
        self,
        model_name: str,
        inputs: Any,
        parameters: Dict,
        outputs: Any,
    ):
        """存入缓存"""
        key = self._get_key(model_name, inputs, parameters)
        
        # 估算大小
        if isinstance(outputs, torch.Tensor):
            size_bytes = outputs.numel() * outputs.element_size()
        elif isinstance(outputs, np.ndarray):
            size_bytes = outputs.nbytes
        else:
            size_bytes = len(str(outputs))
        
        with self._lock:
            # 检查容量
            total_size = sum(v[1] for v in self.cache.values())
            while total_size + size_bytes > self.max_size_mb * 1024 * 1024:
                # 删除最旧的
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][2])
                total_size -= self.cache[oldest_key][1]
                del self.cache[oldest_key]
            
            self.cache[key] = (outputs, size_bytes, time.time())
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self._lock:
            total_size = sum(v[1] for v in self.cache.values())
            return {
                'num_entries': len(self.cache),
                'total_size_mb': total_size / (1024 * 1024),
                'max_size_mb': self.max_size_mb,
            }


# ==================== 速率限制器 ====================

class RateLimiter:
    """速率限制器"""
    
    def __init__(
        self,
        requests_per_second: float = 100,
        tokens_per_second: float = 10000,
    ):
        self.rps_limit = requests_per_second
        self.tps_limit = tokens_per_second
        
        self.request_timestamps: deque = deque()
        self.token_counts: deque = deque()
        self._lock = threading.Lock()
    
    def acquire(self, tokens: int = 1) -> Tuple[bool, float]:
        """获取许可"""
        with self._lock:
            now = time.time()
            
            # 清理旧记录
            while self.request_timestamps and now - self.request_timestamps[0] > 1.0:
                self.request_timestamps.popleft()
            while self.token_counts and now - self.token_counts[0][0] > 1.0:
                self.token_counts.popleft()
            
            # 检查请求限制
            if len(self.request_timestamps) >= self.rps_limit:
                wait_time = 1.0 - (now - self.request_timestamps[0])
                return False, wait_time
            
            # 检查token限制
            current_tokens = sum(t for _, t in self.token_counts)
            if current_tokens + tokens > self.tps_limit:
                wait_time = 1.0 - (now - self.token_counts[0][0])
                return False, wait_time
            
            # 记录
            self.request_timestamps.append(now)
            self.token_counts.append((now, tokens))
            
            return True, 0.0
    
    def get_usage(self) -> Dict[str, float]:
        """获取使用情况"""
        with self._lock:
            now = time.time()
            
            # 清理旧记录
            while self.request_timestamps and now - self.request_timestamps[0] > 1.0:
                self.request_timestamps.popleft()
            while self.token_counts and now - self.token_counts[0][0] > 1.0:
                self.token_counts.popleft()
            
            current_tokens = sum(t for _, t in self.token_counts)
            
            return {
                'requests_per_second': len(self.request_timestamps),
                'tokens_per_second': current_tokens,
                'rps_limit': self.rps_limit,
                'tps_limit': self.tps_limit,
            }


# ==================== API服务器 ====================

class APIServer:
    """API服务器"""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        model_registry: Optional[ModelRegistry] = None,
        enable_cache: bool = True,
        enable_rate_limit: bool = True,
    ):
        self.host = host
        self.port = port
        
        self.registry = model_registry or ModelRegistry()
        self.inference_engine = InferenceEngine(self.registry)
        self.training_manager = TrainingManager(self.registry)
        
        self.cache = CacheManager() if enable_cache else None
        self.rate_limiter = RateLimiter() if enable_rate_limit else None
        
        self.request_handlers: Dict[str, Callable] = {}
        self.middleware: List[Callable] = []
        
        self.running = False
    
    def register_handler(self, endpoint: str, handler: Callable):
        """注册请求处理器"""
        self.request_handlers[endpoint] = handler
    
    def add_middleware(self, middleware: Callable):
        """添加中间件"""
        self.middleware.append(middleware)
    
    def start(self):
        """启动服务器"""
        self.running = True
        # 实际实现需要使用Flask/FastAPI等框架
        print(f"API Server started at {self.host}:{self.port}")
    
    def stop(self):
        """停止服务器"""
        self.running = False
        print("API Server stopped")
    
    def handle_inference(self, request: InferenceRequest) -> InferenceResponse:
        """处理推理请求"""
        # 速率限制
        if self.rate_limiter:
            allowed, wait_time = self.rate_limiter.acquire()
            if not allowed:
                return InferenceResponse(
                    request_id=request.request_id,
                    outputs=None,
                    model_name=request.model_name,
                    latency_ms=0,
                    success=False,
                    error_message=f"Rate limit exceeded. Retry after {wait_time:.2f}s",
                )
        
        # 缓存检查
        if self.cache:
            cached = self.cache.get(
                request.model_name,
                request.inputs,
                request.parameters,
            )
            if cached is not None:
                return InferenceResponse(
                    request_id=request.request_id,
                    outputs=cached,
                    model_name=request.model_name,
                    latency_ms=0,
                    success=True,
                    metadata={'cached': True},
                )
        
        # 执行推理
        response = self.inference_engine.inference(request)
        
        # 缓存结果
        if self.cache and response.success:
            self.cache.put(
                request.model_name,
                request.inputs,
                request.parameters,
                response.outputs,
            )
        
        return response
    
    def handle_training(self, request: TrainingRequest) -> TrainingResponse:
        """处理训练请求"""
        # 这里需要实际的训练函数
        def train_fn(req):
            # 简化实现
            return {
                'metrics': {'loss': 0.5, 'accuracy': 0.9},
                'checkpoint_path': f'/checkpoints/{req.model_name}.pt',
            }
        
        return self.training_manager.submit_training(request, train_fn)
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """获取模型信息"""
        return self.registry.get_info(model_name)
    
    def list_models(self, tags: Optional[List[str]] = None) -> List[ModelInfo]:
        """列出模型"""
        return self.registry.list_models(tags)
    
    def get_server_stats(self) -> Dict[str, Any]:
        """获取服务器统计"""
        stats = {
            'inference_stats': self.inference_engine.get_stats(),
            'training_jobs': len(self.training_manager.list_jobs()),
        }
        
        if self.cache:
            stats['cache_stats'] = self.cache.get_stats()
        
        if self.rate_limiter:
            stats['rate_limit_usage'] = self.rate_limiter.get_usage()
        
        return stats


# ==================== 异步API服务器 ====================

class AsyncAPIServer:
    """异步API服务器"""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        max_concurrent: int = 100,
    ):
        self.host = host
        self.port = port
        self.max_concurrent = max_concurrent
        
        self.registry = ModelRegistry()
        self.inference_engine = InferenceEngine(self.registry)
        
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.running = False
    
    async def start(self):
        """启动异步服务器"""
        self.running = True
        print(f"Async API Server started at {self.host}:{self.port}")
    
    async def stop(self):
        """停止异步服务器"""
        self.running = False
        print("Async API Server stopped")
    
    async def handle_inference(
        self,
        request: InferenceRequest,
    ) -> InferenceResponse:
        """异步处理推理请求"""
        async with self.semaphore:
            # 在线程池中执行推理
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self.inference_engine.inference,
                request,
            )
            return response
    
    async def handle_batch_inference(
        self,
        requests: List[InferenceRequest],
    ) -> List[InferenceResponse]:
        """异步处理批量推理请求"""
        tasks = [
            self.handle_inference(req)
            for req in requests
        ]
        return await asyncio.gather(*tasks)


# ==================== 健康检查 ====================

class HealthChecker:
    """健康检查器"""
    
    def __init__(self, server: APIServer):
        self.server = server
        self.checks: Dict[str, Callable] = {}
        self.last_check: Dict[str, Tuple[bool, str, float]] = {}
    
    def register_check(self, name: str, check_fn: Callable):
        """注册健康检查"""
        self.checks[name] = check_fn
    
    def run_checks(self) -> Dict[str, Tuple[bool, str]]:
        """运行所有健康检查"""
        results = {}
        
        for name, check_fn in self.checks.items():
            try:
                is_healthy, message = check_fn()
                results[name] = (is_healthy, message)
                self.last_check[name] = (is_healthy, message, time.time())
            except Exception as e:
                results[name] = (False, str(e))
                self.last_check[name] = (False, str(e), time.time())
        
        return results
    
    def is_healthy(self) -> bool:
        """检查是否健康"""
        results = self.run_checks()
        return all(is_healthy for is_healthy, _ in results.values())
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            'checks': {
                name: {
                    'healthy': is_healthy,
                    'message': message,
                    'timestamp': timestamp,
                }
                for name, (is_healthy, message, timestamp) in self.last_check.items()
            },
            'overall_healthy': self.is_healthy(),
        }


# ==================== 负载均衡 ====================

class LoadBalancer:
    """负载均衡器"""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.backends: List[str] = []
        self.backend_stats: Dict[str, Dict[str, float]] = {}
        self.current_index = 0
        self._lock = threading.Lock()
    
    def add_backend(self, backend: str):
        """添加后端"""
        with self._lock:
            if backend not in self.backends:
                self.backends.append(backend)
                self.backend_stats[backend] = {
                    'requests': 0,
                    'latency_ms': 0,
                    'errors': 0,
                }
    
    def remove_backend(self, backend: str):
        """移除后端"""
        with self._lock:
            if backend in self.backends:
                self.backends.remove(backend)
                del self.backend_stats[backend]
    
    def select_backend(self) -> Optional[str]:
        """选择后端"""
        with self._lock:
            if not self.backends:
                return None
            
            if self.strategy == "round_robin":
                backend = self.backends[self.current_index]
                self.current_index = (self.current_index + 1) % len(self.backends)
                return backend
            
            elif self.strategy == "least_requests":
                return min(
                    self.backends,
                    key=lambda b: self.backend_stats[b]['requests'],
                )
            
            elif self.strategy == "least_latency":
                return min(
                    self.backends,
                    key=lambda b: self.backend_stats[b]['latency_ms'],
                )
            
            else:
                return self.backends[0]
    
    def update_stats(
        self,
        backend: str,
        latency_ms: float,
        error: bool = False,
    ):
        """更新后端统计"""
        with self._lock:
            if backend in self.backend_stats:
                stats = self.backend_stats[backend]
                stats['requests'] += 1
                stats['latency_ms'] = (
                    stats['latency_ms'] * 0.9 + latency_ms * 0.1
                )
                if error:
                    stats['errors'] += 1


# ==================== 主函数 ====================

def main():
    """测试API服务框架"""
    print("API服务框架测试")
    
    # 创建模型注册表
    registry = ModelRegistry()
    
    # 创建简单模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)
        
        def forward(self, x):
            return self.linear(x)
    
    model = SimpleModel()
    registry.register("simple_model", model, tags=["test", "linear"])
    
    # 列出模型
    models = registry.list_models()
    print(f"Registered models: {[m.name for m in models]}")
    
    # 创建推理引擎
    engine = InferenceEngine(registry, default_device="cpu")
    
    # 创建推理请求
    request = InferenceRequest(
        request_id=str(uuid.uuid4()),
        inputs=np.random.randn(1, 10).astype(np.float32),
        model_name="simple_model",
    )
    
    # 执行推理
    response = engine.inference(request)
    print(f"Inference success: {response.success}")
    print(f"Latency: {response.latency_ms:.2f}ms")
    
    # 测试缓存
    cache = CacheManager()
    cache.put("simple_model", request.inputs, {}, response.outputs)
    cached = cache.get("simple_model", request.inputs, {})
    print(f"Cache hit: {cached is not None}")
    
    # 测试速率限制
    limiter = RateLimiter(requests_per_second=10)
    for i in range(15):
        allowed, wait = limiter.acquire()
        if not allowed:
            print(f"Rate limited at request {i}, wait {wait:.2f}s")
            break
    
    print("API服务框架测试完成")


if __name__ == "__main__":
    main()
