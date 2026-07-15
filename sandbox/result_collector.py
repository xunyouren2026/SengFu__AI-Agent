"""
结果收集器
收集stdout/stderr/退出码/文件输出等执行结果
"""

import os
import json
import time
import base64
import hashlib
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum


class OutputType(Enum):
    """输出类型"""
    STDOUT = "stdout"
    STDERR = "stderr"
    FILE = "file"
    METRIC = "metric"
    ARTIFACT = "artifact"


@dataclass
class OutputChunk:
    """输出块"""
    output_type: OutputType
    content: Union[str, bytes]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        content = self.content
        if isinstance(content, bytes):
            content = base64.b64encode(content).decode('ascii')
        
        return {
            'type': self.output_type.value,
            'content': content,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'is_binary': isinstance(self.content, bytes)
        }


@dataclass
class CollectedResult:
    """收集的结果"""
    execution_id: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    output_files: Dict[str, str] = field(default_factory=dict)
    output_binary_files: Dict[str, bytes] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        """执行时长"""
        return self.end_time - self.start_time
    
    @property
    def success(self) -> bool:
        """是否成功"""
        return self.exit_code == 0 and not self.errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'execution_id': self.execution_id,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'exit_code': self.exit_code,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'output_files': self.output_files,
            'metrics': self.metrics,
            'artifacts': self.artifacts,
            'metadata': self.metadata,
            'warnings': self.warnings,
            'errors': self.errors,
            'success': self.success
        }
    
    def to_json(self, indent: int = 2) -> str:
        """转换为JSON"""
        return json.dumps(self.to_dict(), indent=indent)


class ResultCollector:
    """
    结果收集器
    收集执行过程中的各种输出
    """
    
    def __init__(
        self,
        max_stdout_size: int = 10 * 1024 * 1024,  # 10MB
        max_stderr_size: int = 10 * 1024 * 1024,  # 10MB
        max_file_size: int = 100 * 1024 * 1024,   # 100MB
        collect_binary: bool = True
    ):
        """
        初始化结果收集器
        
        Args:
            max_stdout_size: 最大stdout大小
            max_stderr_size: 最大stderr大小
            max_file_size: 最大文件大小
            collect_binary: 是否收集二进制文件
        """
        self.max_stdout_size = max_stdout_size
        self.max_stderr_size = max_stderr_size
        self.max_file_size = max_file_size
        self.collect_binary = collect_binary
        
        self._current_result: Optional[CollectedResult] = None
        self._output_chunks: List[OutputChunk] = []
    
    def start_collection(self, execution_id: str) -> None:
        """
        开始收集
        
        Args:
            execution_id: 执行ID
        """
        self._current_result = CollectedResult(
            execution_id=execution_id,
            start_time=time.time()
        )
        self._output_chunks = []
    
    def collect_stdout(self, data: str) -> None:
        """
        收集stdout
        
        Args:
            data: 输出数据
        """
        if self._current_result is None:
            return
        
        # 检查大小限制
        if len(self._current_result.stdout) + len(data) > self.max_stdout_size:
            self._current_result.warnings.append("stdout truncated due to size limit")
            remaining = self.max_stdout_size - len(self._current_result.stdout)
            if remaining > 0:
                self._current_result.stdout += data[:remaining]
        else:
            self._current_result.stdout += data
        
        # 记录输出块
        self._output_chunks.append(OutputChunk(
            output_type=OutputType.STDOUT,
            content=data,
            timestamp=time.time()
        ))
    
    def collect_stderr(self, data: str) -> None:
        """
        收集stderr
        
        Args:
            data: 错误数据
        """
        if self._current_result is None:
            return
        
        # 检查大小限制
        if len(self._current_result.stderr) + len(data) > self.max_stderr_size:
            self._current_result.warnings.append("stderr truncated due to size limit")
            remaining = self.max_stderr_size - len(self._current_result.stderr)
            if remaining > 0:
                self._current_result.stderr += data[:remaining]
        else:
            self._current_result.stderr += data
        
        self._output_chunks.append(OutputChunk(
            output_type=OutputType.STDERR,
            content=data,
            timestamp=time.time()
        ))
    
    def collect_exit_code(self, code: int) -> None:
        """
        收集退出码
        
        Args:
            code: 退出码
        """
        if self._current_result:
            self._current_result.exit_code = code
    
    def collect_file(
        self,
        filepath: str,
        content: Optional[Union[str, bytes]] = None,
        relative_path: Optional[str] = None
    ) -> bool:
        """
        收集文件
        
        Args:
            filepath: 文件路径
            content: 文件内容（None则从文件读取）
            relative_path: 相对路径（用于结果中的键）
            
        Returns:
            是否成功
        """
        if self._current_result is None:
            return False
        
        try:
            # 确定相对路径
            rel_path = relative_path or os.path.basename(filepath)
            
            # 读取内容
            if content is None:
                if not os.path.exists(filepath):
                    return False
                
                file_size = os.path.getsize(filepath)
                if file_size > self.max_file_size:
                    self._current_result.warnings.append(
                        f"File {rel_path} skipped due to size limit"
                    )
                    return False
                
                with open(filepath, 'rb') as f:
                    content = f.read()
            
            # 处理内容
            if isinstance(content, bytes):
                # 尝试解码为文本
                try:
                    text_content = content.decode('utf-8')
                    self._current_result.output_files[rel_path] = text_content
                except UnicodeDecodeError:
                    if self.collect_binary:
                        self._current_result.output_binary_files[rel_path] = content
                    else:
                        # 存储base64编码
                        self._current_result.output_files[rel_path] = \
                            base64.b64encode(content).decode('ascii')
            else:
                self._current_result.output_files[rel_path] = content
            
            self._output_chunks.append(OutputChunk(
                output_type=OutputType.FILE,
                content=content,
                timestamp=time.time(),
                metadata={'path': rel_path}
            ))
            
            return True
            
        except IOError as e:
            if self._current_result:
                self._current_result.errors.append(f"Failed to collect file {filepath}: {e}")
            return False
    
    def collect_directory(
        self,
        directory: str,
        patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """
        收集目录中的文件
        
        Args:
            directory: 目录路径
            patterns: 包含模式
            exclude_patterns: 排除模式
            
        Returns:
            收集结果字典
        """
        results = {}
        directory = Path(directory)
        
        if patterns:
            file_iter = []
            for pattern in patterns:
                file_iter.extend(directory.glob(pattern))
        else:
            file_iter = directory.rglob('*')
        
        for file_path in file_iter:
            if not file_path.is_file():
                continue
            
            rel_path = str(file_path.relative_to(directory))
            
            # 检查排除模式
            if exclude_patterns:
                excluded = False
                for pattern in exclude_patterns:
                    if file_path.match(pattern):
                        excluded = True
                        break
                if excluded:
                    continue
            
            results[rel_path] = self.collect_file(
                str(file_path),
                relative_path=rel_path
            )
        
        return results
    
    def collect_metric(self, name: str, value: Any) -> None:
        """
        收集指标
        
        Args:
            name: 指标名称
            value: 指标值
        """
        if self._current_result:
            self._current_result.metrics[name] = value
            
            self._output_chunks.append(OutputChunk(
                output_type=OutputType.METRIC,
                content=json.dumps({name: value}),
                timestamp=time.time(),
                metadata={'metric_name': name}
            ))
    
    def collect_artifact(
        self,
        name: str,
        filepath: str,
        description: Optional[str] = None
    ) -> bool:
        """
        收集工件
        
        Args:
            name: 工件名称
            filepath: 文件路径
            description: 描述
            
        Returns:
            是否成功
        """
        if self._current_result is None:
            return False
        
        if not os.path.exists(filepath):
            return False
        
        # 计算文件哈希
        with open(filepath, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        self._current_result.artifacts[name] = filepath
        self._current_result.metadata[f'artifact_{name}_hash'] = file_hash
        self._current_result.metadata[f'artifact_{name}_size'] = os.path.getsize(filepath)
        
        if description:
            self._current_result.metadata[f'artifact_{name}_description'] = description
        
        return True
    
    def add_warning(self, message: str) -> None:
        """添加警告"""
        if self._current_result:
            self._current_result.warnings.append(message)
    
    def add_error(self, message: str) -> None:
        """添加错误"""
        if self._current_result:
            self._current_result.errors.append(message)
    
    def set_metadata(self, key: str, value: Any) -> None:
        """设置元数据"""
        if self._current_result:
            self._current_result.metadata[key] = value
    
    def finish_collection(self) -> CollectedResult:
        """
        完成收集
        
        Returns:
            收集的结果
        """
        if self._current_result is None:
            return CollectedResult(execution_id="unknown")
        
        self._current_result.end_time = time.time()
        result = self._current_result
        self._current_result = None
        
        return result
    
    def get_output_chunks(self) -> List[OutputChunk]:
        """获取输出块列表"""
        return list(self._output_chunks)
    
    def get_current_result(self) -> Optional[CollectedResult]:
        """获取当前结果"""
        return self._current_result


class StreamingResultCollector(ResultCollector):
    """
    流式结果收集器
    支持实时输出回调
    """
    
    def __init__(
        self,
        stdout_callback: Optional[callable] = None,
        stderr_callback: Optional[callable] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.stdout_callback = stdout_callback
        self.stderr_callback = stderr_callback
    
    def collect_stdout(self, data: str) -> None:
        """收集stdout并回调"""
        super().collect_stdout(data)
        if self.stdout_callback:
            self.stdout_callback(data)
    
    def collect_stderr(self, data: str) -> None:
        """收集stderr并回调"""
        super().collect_stderr(data)
        if self.stderr_callback:
            self.stderr_callback(data)


class ResultAggregator:
    """
    结果聚合器
    聚合多次执行的结果
    """
    
    def __init__(self):
        self._results: Dict[str, CollectedResult] = {}
        self._aggregated_metrics: Dict[str, List[Any]] = {}
    
    def add_result(self, result: CollectedResult) -> None:
        """添加结果"""
        self._results[result.execution_id] = result
        
        # 聚合指标
        for name, value in result.metrics.items():
            if name not in self._aggregated_metrics:
                self._aggregated_metrics[name] = []
            self._aggregated_metrics[name].append(value)
    
    def get_result(self, execution_id: str) -> Optional[CollectedResult]:
        """获取结果"""
        return self._results.get(execution_id)
    
    def get_all_results(self) -> Dict[str, CollectedResult]:
        """获取所有结果"""
        return dict(self._results)
    
    def get_aggregated_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        获取聚合指标
        
        Returns:
            聚合指标字典，包含min, max, avg, sum, count
        """
        aggregated = {}
        
        for name, values in self._aggregated_metrics.items():
            if not values:
                continue
            
            # 只处理数值类型
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            
            if numeric_values:
                aggregated[name] = {
                    'min': min(numeric_values),
                    'max': max(numeric_values),
                    'avg': sum(numeric_values) / len(numeric_values),
                    'sum': sum(numeric_values),
                    'count': len(numeric_values)
                }
            else:
                aggregated[name] = {
                    'values': values,
                    'count': len(values)
                }
        
        return aggregated
    
    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        total = len(self._results)
        successful = sum(1 for r in self._results.values() if r.success)
        failed = total - successful
        
        total_duration = sum(r.duration for r in self._results.values())
        avg_duration = total_duration / total if total > 0 else 0
        
        return {
            'total_executions': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total if total > 0 else 0,
            'total_duration': total_duration,
            'average_duration': avg_duration,
            'aggregated_metrics': self.get_aggregated_metrics()
        }
    
    def clear(self) -> None:
        """清空所有结果"""
        self._results.clear()
        self._aggregated_metrics.clear()
    
    def export_results(self, filepath: str) -> None:
        """导出结果到文件"""
        data = {
            'results': {k: v.to_dict() for k, v in self._results.items()},
            'summary': self.get_summary()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def import_results(self, filepath: str) -> None:
        """从文件导入结果"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for execution_id, result_data in data.get('results', {}).items():
            result = CollectedResult(
                execution_id=execution_id,
                stdout=result_data.get('stdout', ''),
                stderr=result_data.get('stderr', ''),
                exit_code=result_data.get('exit_code', 0),
                start_time=result_data.get('start_time', 0),
                end_time=result_data.get('end_time', 0),
                output_files=result_data.get('output_files', {}),
                metrics=result_data.get('metrics', {}),
                metadata=result_data.get('metadata', {}),
                warnings=result_data.get('warnings', []),
                errors=result_data.get('errors', [])
            )
            self.add_result(result)
