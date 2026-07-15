"""
Operation Audit Module

提供操作审计功能，包括操作记录、日志存储和索引管理。
"""

import json
import os
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union


class OperationType(Enum):
    """操作类型枚举"""
    CREATE = auto()
    READ = auto()
    UPDATE = auto()
    DELETE = auto()
    EXECUTE = auto()
    CONFIGURE = auto()
    AUTHENTICATE = auto()
    AUTHORIZE = auto()
    EXPORT = auto()
    IMPORT = auto()


class OperationResult(Enum):
    """操作结果枚举"""
    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"
    ERROR = "error"
    TIMEOUT = "timeout"


class OperationRecord:
    """
    操作记录类
    
    记录单次操作的完整信息，包括时间、执行者、操作类型、目标对象等。
    """
    
    def __init__(
        self,
        timestamp: Optional[datetime] = None,
        actor: str = "",
        action: str = "",
        target: str = "",
        result: OperationResult = OperationResult.SUCCESS,
        details: Optional[Dict[str, Any]] = None,
        session_id: str = "",
        ip_address: str = "",
        user_agent: str = "",
        operation_id: Optional[str] = None
    ):
        self.timestamp = timestamp or datetime.utcnow()
        self.actor = actor
        self.action = action
        self.target = target
        self.result = result
        self.details = details or {}
        self.session_id = session_id
        self.ip_address = ip_address
        self.user_agent = user_agent
        self.operation_id = operation_id or self._generate_id()
    
    def _generate_id(self) -> str:
        """生成唯一操作ID"""
        timestamp_str = self.timestamp.strftime("%Y%m%d%H%M%S%f")
        return f"OP-{timestamp_str}-{hash(self.actor + self.action) % 10000:04d}"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "operation_id": self.operation_id,
            "timestamp": self.timestamp.isoformat(),
            "actor": self.actor,
            "action": self.action,
            "target": self.target,
            "result": self.result.value if isinstance(self.result, OperationResult) else self.result,
            "details": self.details,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OperationRecord':
        """从字典创建实例"""
        result = data.get("result", "success")
        if isinstance(result, str):
            try:
                result = OperationResult(result)
            except ValueError:
                result = OperationResult.SUCCESS
        
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        return cls(
            timestamp=timestamp,
            actor=data.get("actor", ""),
            action=data.get("action", ""),
            target=data.get("target", ""),
            result=result,
            details=data.get("details", {}),
            session_id=data.get("session_id", ""),
            ip_address=data.get("ip_address", ""),
            user_agent=data.get("user_agent", ""),
            operation_id=data.get("operation_id")
        )
    
    def __repr__(self) -> str:
        return f"<OperationRecord {self.operation_id}: {self.actor} {self.action} {self.target}>"


class AuditLogStore:
    """
    审计日志存储类
    
    实现内存+文件双写机制，支持日志追加、查询、轮转和归档功能。
    """
    
    def __init__(
        self,
        log_dir: str = "/var/log/audit",
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        max_files: int = 10,
        memory_buffer_size: int = 1000
    ):
        self.log_dir = log_dir
        self.max_file_size = max_file_size
        self.max_files = max_files
        self.memory_buffer_size = memory_buffer_size
        
        self._memory_buffer: List[OperationRecord] = []
        self._lock = threading.RLock()
        self._current_file: Optional[str] = None
        
        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        self._init_current_file()
    
    def _init_current_file(self) -> None:
        """初始化当前日志文件"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self._current_file = os.path.join(self.log_dir, f"audit_{timestamp}.log")
    
    def append(self, record: OperationRecord) -> bool:
        """
        追加日志记录
        
        Args:
            record: 操作记录对象
            
        Returns:
            是否成功追加
        """
        with self._lock:
            # 写入内存缓冲区
            self._memory_buffer.append(record)
            if len(self._memory_buffer) > self.memory_buffer_size:
                self._memory_buffer.pop(0)
            
            # 写入文件
            try:
                # 检查是否需要轮转
                if self._should_rotate():
                    self.rotate()
                
                # 追加到文件
                with open(self._current_file, 'a', encoding='utf-8') as f:
                    json_line = json.dumps(record.to_dict(), ensure_ascii=False)
                    f.write(json_line + '\n')
                
                return True
            except Exception as e:
                # 记录错误但不抛出，确保内存写入成功
                print(f"Failed to write audit log to file: {e}")
                return False
    
    def _should_rotate(self) -> bool:
        """检查是否需要轮转日志文件"""
        if not os.path.exists(self._current_file):
            return False
        return os.path.getsize(self._current_file) >= self.max_file_size
    
    def rotate(self) -> None:
        """
        轮转日志文件
        
        关闭当前文件，创建新文件，并清理旧文件。
        """
        with self._lock:
            old_file = self._current_file
            self._init_current_file()
            self._cleanup_old_files()
    
    def _cleanup_old_files(self) -> None:
        """清理旧日志文件"""
        try:
            log_files = [
                f for f in os.listdir(self.log_dir)
                if f.startswith("audit_") and f.endswith(".log")
            ]
            log_files.sort(reverse=True)
            
            for old_file in log_files[self.max_files:]:
                os.remove(os.path.join(self.log_dir, old_file))
        except Exception as e:
            print(f"Failed to cleanup old log files: {e}")
    
    def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        actor: Optional[str] = None,
        action: Optional[str] = None,
        target: Optional[str] = None,
        result: Optional[OperationResult] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[OperationRecord]:
        """
        查询日志记录
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            actor: 执行者筛选
            action: 操作类型筛选
            target: 目标对象筛选
            result: 操作结果筛选
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            符合条件的操作记录列表
        """
        results = []
        
        with self._lock:
            # 从内存缓冲区查询
            for record in reversed(self._memory_buffer):
                if self._matches_filter(record, start_time, end_time, actor, action, target, result):
                    results.append(record)
            
            # 从文件查询
            try:
                log_files = [
                    f for f in os.listdir(self.log_dir)
                    if f.startswith("audit_") and f.endswith(".log")
                ]
                log_files.sort(reverse=True)
                
                for log_file in log_files:
                    if len(results) >= limit + offset:
                        break
                    
                    file_path = os.path.join(self.log_dir, log_file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if len(results) >= limit + offset:
                                break
                            
                            try:
                                data = json.loads(line.strip())
                                record = OperationRecord.from_dict(data)
                                if self._matches_filter(record, start_time, end_time, actor, action, target, result):
                                    results.append(record)
                            except (json.JSONDecodeError, KeyError):
                                continue
            except Exception as e:
                print(f"Failed to query log files: {e}")
        
        # 应用分页
        return results[offset:offset + limit]
    
    def _matches_filter(
        self,
        record: OperationRecord,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        actor: Optional[str],
        action: Optional[str],
        target: Optional[str],
        result: Optional[OperationResult]
    ) -> bool:
        """检查记录是否匹配筛选条件"""
        if start_time and record.timestamp < start_time:
            return False
        if end_time and record.timestamp > end_time:
            return False
        if actor and record.actor != actor:
            return False
        if action and record.action != action:
            return False
        if target and record.target != target:
            return False
        if result and record.result != result:
            return False
        return True
    
    def archive(self, archive_dir: str, before: Optional[datetime] = None) -> List[str]:
        """
        归档日志文件
        
        Args:
            archive_dir: 归档目录
            before: 归档此时间之前的日志
            
        Returns:
            已归档的文件列表
        """
        archived_files = []
        before = before or datetime.utcnow() - timedelta(days=30)
        
        os.makedirs(archive_dir, exist_ok=True)
        
        try:
            log_files = [
                f for f in os.listdir(self.log_dir)
                if f.startswith("audit_") and f.endswith(".log")
            ]
            
            for log_file in log_files:
                # 从文件名解析时间
                try:
                    file_time = datetime.strptime(log_file[6:21], "%Y%m%d_%H%M%S")
                except ValueError:
                    continue
                
                if file_time < before:
                    src_path = os.path.join(self.log_dir, log_file)
                    dst_path = os.path.join(archive_dir, log_file)
                    
                    # 如果当前文件正在使用，跳过
                    if src_path == self._current_file:
                        continue
                    
                    os.rename(src_path, dst_path)
                    archived_files.append(dst_path)
        except Exception as e:
            print(f"Failed to archive log files: {e}")
        
        return archived_files


class AuditLogIndexer:
    """
    审计日志索引器
    
    提供按时间、执行者、操作类型和目标对象的索引功能。
    """
    
    def __init__(self):
        self._time_index: Dict[str, List[str]] = defaultdict(list)
        self._actor_index: Dict[str, List[str]] = defaultdict(list)
        self._action_index: Dict[str, List[str]] = defaultdict(list)
        self._target_index: Dict[str, List[str]] = defaultdict(list)
        self._records: Dict[str, OperationRecord] = {}
        self._lock = threading.RLock()
    
    def add_record(self, record: OperationRecord) -> None:
        """
        添加记录到索引
        
        Args:
            record: 操作记录对象
        """
        with self._lock:
            op_id = record.operation_id
            self._records[op_id] = record
            
            # 按时间索引（按小时）
            time_key = record.timestamp.strftime("%Y-%m-%d-%H")
            self._time_index[time_key].append(op_id)
            
            # 按执行者索引
            if record.actor:
                self._actor_index[record.actor].append(op_id)
            
            # 按操作类型索引
            if record.action:
                self._action_index[record.action].append(op_id)
            
            # 按目标对象索引
            if record.target:
                self._target_index[record.target].append(op_id)
    
    def query_by_time(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[OperationRecord]:
        """
        按时间范围查询
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            符合条件的操作记录列表
        """
        results = []
        
        with self._lock:
            current = start_time.replace(minute=0, second=0, microsecond=0)
            end_hour = end_time.replace(minute=0, second=0, microsecond=0)
            
            while current <= end_hour:
                time_key = current.strftime("%Y-%m-%d-%H")
                for op_id in self._time_index.get(time_key, []):
                    record = self._records.get(op_id)
                    if record and start_time <= record.timestamp <= end_time:
                        results.append(record)
                current += timedelta(hours=1)
        
        return results
    
    def query_by_actor(self, actor: str) -> List[OperationRecord]:
        """
        按执行者查询
        
        Args:
            actor: 执行者标识
            
        Returns:
            该执行者的所有操作记录
        """
        with self._lock:
            return [
                self._records[op_id]
                for op_id in self._actor_index.get(actor, [])
                if op_id in self._records
            ]
    
    def query_by_action(self, action: str) -> List[OperationRecord]:
        """
        按操作类型查询
        
        Args:
            action: 操作类型
            
        Returns:
            该类型的所有操作记录
        """
        with self._lock:
            return [
                self._records[op_id]
                for op_id in self._action_index.get(action, [])
                if op_id in self._records
            ]
    
    def query_by_target(self, target: str) -> List[OperationRecord]:
        """
        按目标对象查询
        
        Args:
            target: 目标对象标识
            
        Returns:
            涉及该目标的所有操作记录
        """
        with self._lock:
            return [
                self._records[op_id]
                for op_id in self._target_index.get(target, [])
                if op_id in self._records
            ]
    
    def get_actor_activity_summary(self, actor: str) -> Dict[str, Any]:
        """
        获取执行者活动摘要
        
        Args:
            actor: 执行者标识
            
        Returns:
            活动统计信息
        """
        records = self.query_by_actor(actor)
        
        if not records:
            return {}
        
        action_counts = defaultdict(int)
        target_counts = defaultdict(int)
        result_counts = defaultdict(int)
        
        for record in records:
            action_counts[record.action] += 1
            target_counts[record.target] += 1
            result_counts[record.result.value if isinstance(record.result, OperationResult) else record.result] += 1
        
        return {
            "actor": actor,
            "total_operations": len(records),
            "first_operation": min(r.timestamp for r in records).isoformat(),
            "last_operation": max(r.timestamp for r in records).isoformat(),
            "action_distribution": dict(action_counts),
            "target_distribution": dict(target_counts),
            "result_distribution": dict(result_counts)
        }
    
    def clear(self) -> None:
        """清空所有索引"""
        with self._lock:
            self._time_index.clear()
            self._actor_index.clear()
            self._action_index.clear()
            self._target_index.clear()
            self._records.clear()


class OperationAuditor:
    """
    操作审计器
    
    提供统一的操作审计接口，支持多种审计场景。
    """
    
    def __init__(
        self,
        log_store: Optional[AuditLogStore] = None,
        indexer: Optional[AuditLogIndexer] = None
    ):
        self.log_store = log_store or AuditLogStore()
        self.indexer = indexer or AuditLogIndexer()
        self._lock = threading.RLock()
    
    def log_operation(
        self,
        actor: str,
        action: str,
        target: str,
        result: OperationResult = OperationResult.SUCCESS,
        details: Optional[Dict[str, Any]] = None,
        session_id: str = "",
        ip_address: str = "",
        user_agent: str = ""
    ) -> OperationRecord:
        """
        记录通用操作
        
        Args:
            actor: 执行者标识
            action: 操作类型
            target: 目标对象
            result: 操作结果
            details: 详细信息的字典
            session_id: 会话ID
            ip_address: IP地址
            user_agent: 用户代理
            
        Returns:
            创建的操作记录
        """
        record = OperationRecord(
            actor=actor,
            action=action,
            target=target,
            result=result,
            details=details,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        with self._lock:
            self.log_store.append(record)
            self.indexer.add_record(record)
        
        return record
    
    def log_data_access(
        self,
        actor: str,
        data_resource: str,
        access_type: str,
        records_accessed: int = 0,
        result: OperationResult = OperationResult.SUCCESS,
        session_id: str = "",
        ip_address: str = "",
        user_agent: str = ""
    ) -> OperationRecord:
        """
        记录数据访问操作
        
        Args:
            actor: 执行者标识
            data_resource: 数据资源标识
            access_type: 访问类型（read/write/delete）
            records_accessed: 访问的记录数量
            result: 操作结果
            session_id: 会话ID
            ip_address: IP地址
            user_agent: 用户代理
            
        Returns:
            创建的操作记录
        """
        details = {
            "access_type": access_type,
            "records_accessed": records_accessed,
            "operation_category": "data_access"
        }
        
        return self.log_operation(
            actor=actor,
            action=f"data_{access_type}",
            target=data_resource,
            result=result,
            details=details,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent
        )
    
    def log_config_change(
        self,
        actor: str,
        config_path: str,
        old_value: Any,
        new_value: Any,
        result: OperationResult = OperationResult.SUCCESS,
        session_id: str = "",
        ip_address: str = ""
    ) -> OperationRecord:
        """
        记录配置变更操作
        
        Args:
            actor: 执行者标识
            config_path: 配置项路径
            old_value: 变更前的值
            new_value: 变更后的值
            result: 操作结果
            session_id: 会话ID
            ip_address: IP地址
            
        Returns:
            创建的操作记录
        """
        details = {
            "config_path": config_path,
            "old_value": str(old_value)[:1000],  # 限制长度
            "new_value": str(new_value)[:1000],
            "operation_category": "config_change"
        }
        
        return self.log_operation(
            actor=actor,
            action="config_change",
            target=config_path,
            result=result,
            details=details,
            session_id=session_id,
            ip_address=ip_address
        )
    
    def log_auth_event(
        self,
        actor: str,
        auth_type: str,
        result: OperationResult,
        failure_reason: str = "",
        session_id: str = "",
        ip_address: str = "",
        user_agent: str = ""
    ) -> OperationRecord:
        """
        记录认证事件
        
        Args:
            actor: 执行者标识
            auth_type: 认证类型（login/logout/mfa/password_change）
            result: 认证结果
            failure_reason: 失败原因
            session_id: 会话ID
            ip_address: IP地址
            user_agent: 用户代理
            
        Returns:
            创建的操作记录
        """
        details = {
            "auth_type": auth_type,
            "failure_reason": failure_reason,
            "operation_category": "authentication"
        }
        
        return self.log_operation(
            actor=actor,
            action=f"auth_{auth_type}",
            target="authentication_service",
            result=result,
            details=details,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent
        )
    
    def log_api_call(
        self,
        actor: str,
        api_endpoint: str,
        http_method: str,
        result: OperationResult = OperationResult.SUCCESS,
        request_params: Optional[Dict[str, Any]] = None,
        response_status: int = 200,
        execution_time_ms: float = 0.0,
        session_id: str = "",
        ip_address: str = "",
        user_agent: str = ""
    ) -> OperationRecord:
        """
        记录API调用
        
        Args:
            actor: 执行者标识
            api_endpoint: API端点
            http_method: HTTP方法
            result: 调用结果
            request_params: 请求参数
            response_status: 响应状态码
            execution_time_ms: 执行时间（毫秒）
            session_id: 会话ID
            ip_address: IP地址
            user_agent: 用户代理
            
        Returns:
            创建的操作记录
        """
        details = {
            "http_method": http_method,
            "request_params": request_params or {},
            "response_status": response_status,
            "execution_time_ms": execution_time_ms,
            "operation_category": "api_call"
        }
        
        return self.log_operation(
            actor=actor,
            action=f"api_{http_method.lower()}",
            target=api_endpoint,
            result=result,
            details=details,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent
        )
    
    def query_operations(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        actor: Optional[str] = None,
        action: Optional[str] = None,
        target: Optional[str] = None,
        result: Optional[OperationResult] = None,
        limit: int = 100
    ) -> List[OperationRecord]:
        """
        查询操作记录
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            actor: 执行者筛选
            action: 操作类型筛选
            target: 目标对象筛选
            result: 操作结果筛选
            limit: 返回数量限制
            
        Returns:
            符合条件的操作记录列表
        """
        return self.log_store.query(
            start_time=start_time,
            end_time=end_time,
            actor=actor,
            action=action,
            target=target,
            result=result,
            limit=limit
        )
    
    def get_actor_history(self, actor: str) -> Dict[str, Any]:
        """
        获取执行者历史记录摘要
        
        Args:
            actor: 执行者标识
            
        Returns:
            活动摘要信息
        """
        return self.indexer.get_actor_activity_summary(actor)
