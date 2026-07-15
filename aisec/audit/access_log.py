"""
Access Log Module

提供访问日志记录、聚合和会话追踪功能。
"""

import json
import os
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple


class AccessType(Enum):
    """访问类型枚举"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"


class AccessDecision(Enum):
    """访问决策枚举"""
    GRANTED = "granted"
    DENIED = "denied"
    CONDITIONAL = "conditional"


class AccessLogEntry:
    """
    访问日志条目类
    
    记录单次访问的完整信息。
    """
    
    def __init__(
        self,
        timestamp: Optional[datetime] = None,
        user: str = "",
        resource: str = "",
        action: str = "",
        granted: bool = True,
        reason: str = "",
        location: str = "",
        device: str = "",
        session_id: str = "",
        ip_address: str = "",
        additional_context: Optional[Dict[str, Any]] = None
    ):
        self.timestamp = timestamp or datetime.utcnow()
        self.user = user
        self.resource = resource
        self.action = action
        self.granted = granted
        self.reason = reason
        self.location = location
        self.device = device
        self.session_id = session_id
        self.ip_address = ip_address
        self.additional_context = additional_context or {}
        self.entry_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """生成唯一条目ID"""
        timestamp_str = self.timestamp.strftime("%Y%m%d%H%M%S%f")
        return f"ACC-{timestamp_str}-{hash(self.user + self.resource) % 10000:04d}"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "user": self.user,
            "resource": self.resource,
            "action": self.action,
            "granted": self.granted,
            "reason": self.reason,
            "location": self.location,
            "device": self.device,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "additional_context": self.additional_context
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AccessLogEntry':
        """从字典创建实例"""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        entry = cls(
            timestamp=timestamp,
            user=data.get("user", ""),
            resource=data.get("resource", ""),
            action=data.get("action", ""),
            granted=data.get("granted", True),
            reason=data.get("reason", ""),
            location=data.get("location", ""),
            device=data.get("device", ""),
            session_id=data.get("session_id", ""),
            ip_address=data.get("ip_address", ""),
            additional_context=data.get("additional_context", {})
        )
        entry.entry_id = data.get("entry_id", entry.entry_id)
        return entry


class AccessPattern:
    """
    访问模式类
    
    描述检测到的访问行为模式。
    """
    
    def __init__(
        self,
        pattern_type: str,
        frequency: int,
        entities: List[str],
        risk_score: float,
        description: str = "",
        first_seen: Optional[datetime] = None,
        last_seen: Optional[datetime] = None
    ):
        self.pattern_type = pattern_type
        self.frequency = frequency
        self.entities = entities
        self.risk_score = risk_score
        self.description = description
        self.first_seen = first_seen or datetime.utcnow()
        self.last_seen = last_seen or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "pattern_type": self.pattern_type,
            "frequency": self.frequency,
            "entities": self.entities,
            "risk_score": self.risk_score,
            "description": self.description,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat()
        }


class AccessLogger:
    """
    访问日志记录器
    
    提供访问日志的统一记录接口。
    """
    
    def __init__(self, log_dir: str = "/var/log/access"):
        self.log_dir = log_dir
        self._entries: List[AccessLogEntry] = []
        self._lock = threading.RLock()
        
        os.makedirs(self.log_dir, exist_ok=True)
        self._current_log_file = os.path.join(
            self.log_dir,
            f"access_{datetime.utcnow().strftime('%Y%m%d')}.log"
        )
    
    def _write_to_file(self, entry: AccessLogEntry) -> None:
        """写入日志文件"""
        try:
            with open(self._current_log_file, 'a', encoding='utf-8') as f:
                json_line = json.dumps(entry.to_dict(), ensure_ascii=False)
                f.write(json_line + '\n')
        except Exception as e:
            print(f"Failed to write access log: {e}")
    
    def log_access(
        self,
        user: str,
        resource: str,
        action: str,
        granted: bool = True,
        reason: str = "",
        location: str = "",
        device: str = "",
        session_id: str = "",
        ip_address: str = "",
        additional_context: Optional[Dict[str, Any]] = None
    ) -> AccessLogEntry:
        """
        记录访问日志
        
        Args:
            user: 用户标识
            resource: 资源标识
            action: 操作类型
            granted: 是否授权通过
            reason: 授权/拒绝原因
            location: 访问位置
            device: 设备信息
            session_id: 会话ID
            ip_address: IP地址
            additional_context: 额外上下文信息
            
        Returns:
            创建的访问日志条目
        """
        entry = AccessLogEntry(
            user=user,
            resource=resource,
            action=action,
            granted=granted,
            reason=reason,
            location=location,
            device=device,
            session_id=session_id,
            ip_address=ip_address,
            additional_context=additional_context or {}
        )
        
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > 10000:  # 限制内存中的条目数
                self._entries.pop(0)
            self._write_to_file(entry)
        
        return entry
    
    def log_denied_access(
        self,
        user: str,
        resource: str,
        action: str,
        reason: str,
        location: str = "",
        device: str = "",
        session_id: str = "",
        ip_address: str = ""
    ) -> AccessLogEntry:
        """
        记录拒绝访问
        
        Args:
            user: 用户标识
            resource: 资源标识
            action: 操作类型
            reason: 拒绝原因
            location: 访问位置
            device: 设备信息
            session_id: 会话ID
            ip_address: IP地址
            
        Returns:
            创建的访问日志条目
        """
        return self.log_access(
            user=user,
            resource=resource,
            action=action,
            granted=False,
            reason=reason,
            location=location,
            device=device,
            session_id=session_id,
            ip_address=ip_address,
            additional_context={"access_type": "denied"}
        )
    
    def log_privilege_escalation(
        self,
        user: str,
        from_role: str,
        to_role: str,
        reason: str,
        authorized_by: str = "",
        session_id: str = "",
        ip_address: str = ""
    ) -> AccessLogEntry:
        """
        记录权限提升
        
        Args:
            user: 用户标识
            from_role: 原角色
            to_role: 目标角色
            reason: 提升原因
            authorized_by: 授权人
            session_id: 会话ID
            ip_address: IP地址
            
        Returns:
            创建的访问日志条目
        """
        return self.log_access(
            user=user,
            resource=f"role://{to_role}",
            action="privilege_escalation",
            granted=True,
            reason=reason,
            session_id=session_id,
            ip_address=ip_address,
            additional_context={
                "from_role": from_role,
                "to_role": to_role,
                "authorized_by": authorized_by,
                "access_type": "privilege_escalation"
            }
        )
    
    def get_entries(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user: Optional[str] = None,
        resource: Optional[str] = None,
        granted: Optional[bool] = None,
        limit: int = 100
    ) -> List[AccessLogEntry]:
        """
        获取访问日志条目
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            user: 用户筛选
            resource: 资源筛选
            granted: 授权状态筛选
            limit: 返回数量限制
            
        Returns:
            符合条件的访问日志条目列表
        """
        results = []
        
        with self._lock:
            for entry in reversed(self._entries):
                if len(results) >= limit:
                    break
                
                if start_time and entry.timestamp < start_time:
                    continue
                if end_time and entry.timestamp > end_time:
                    continue
                if user and entry.user != user:
                    continue
                if resource and entry.resource != resource:
                    continue
                if granted is not None and entry.granted != granted:
                    continue
                
                results.append(entry)
        
        return results


class AccessLogAggregator:
    """
    访问日志聚合器
    
    提供访问日志的多维度聚合分析功能。
    """
    
    def __init__(self, logger: AccessLogger):
        self.logger = logger
    
    def aggregate_by_user(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        按用户聚合访问日志
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            用户访问统计字典
        """
        entries = self.logger.get_entries(
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )
        
        user_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_accesses": 0,
            "granted": 0,
            "denied": 0,
            "unique_resources": set(),
            "actions": defaultdict(int)
        })
        
        for entry in entries:
            stats = user_stats[entry.user]
            stats["total_accesses"] += 1
            if entry.granted:
                stats["granted"] += 1
            else:
                stats["denied"] += 1
            stats["unique_resources"].add(entry.resource)
            stats["actions"][entry.action] += 1
        
        # 转换集合为列表以便序列化
        result = {}
        for user, stats in user_stats.items():
            result[user] = {
                "total_accesses": stats["total_accesses"],
                "granted": stats["granted"],
                "denied": stats["denied"],
                "unique_resources_count": len(stats["unique_resources"]),
                "actions": dict(stats["actions"])
            }
        
        return result
    
    def aggregate_by_resource(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        按资源聚合访问日志
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            资源访问统计字典
        """
        entries = self.logger.get_entries(
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )
        
        resource_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_accesses": 0,
            "granted": 0,
            "denied": 0,
            "unique_users": set(),
            "actions": defaultdict(int)
        })
        
        for entry in entries:
            stats = resource_stats[entry.resource]
            stats["total_accesses"] += 1
            if entry.granted:
                stats["granted"] += 1
            else:
                stats["denied"] += 1
            stats["unique_users"].add(entry.user)
            stats["actions"][entry.action] += 1
        
        result = {}
        for resource, stats in resource_stats.items():
            result[resource] = {
                "total_accesses": stats["total_accesses"],
                "granted": stats["granted"],
                "denied": stats["denied"],
                "unique_users_count": len(stats["unique_users"]),
                "actions": dict(stats["actions"])
            }
        
        return result
    
    def aggregate_by_time(
        self,
        interval: str = "hour",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        按时间聚合访问日志
        
        Args:
            interval: 时间间隔（hour/day/week/month）
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            时间维度访问统计字典
        """
        entries = self.logger.get_entries(
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )
        
        time_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_accesses": 0,
            "granted": 0,
            "denied": 0,
            "unique_users": set()
        })
        
        for entry in entries:
            if interval == "hour":
                key = entry.timestamp.strftime("%Y-%m-%d-%H")
            elif interval == "day":
                key = entry.timestamp.strftime("%Y-%m-%d")
            elif interval == "week":
                key = entry.timestamp.strftime("%Y-W%W")
            elif interval == "month":
                key = entry.timestamp.strftime("%Y-%m")
            else:
                key = entry.timestamp.strftime("%Y-%m-%d-%H")
            
            stats = time_stats[key]
            stats["total_accesses"] += 1
            if entry.granted:
                stats["granted"] += 1
            else:
                stats["denied"] += 1
            stats["unique_users"].add(entry.user)
        
        result = {}
        for key, stats in time_stats.items():
            result[key] = {
                "total_accesses": stats["total_accesses"],
                "granted": stats["granted"],
                "denied": stats["denied"],
                "unique_users_count": len(stats["unique_users"])
            }
        
        return result
    
    def detect_access_patterns(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[AccessPattern]:
        """
        检测访问模式
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            检测到的访问模式列表
        """
        patterns = []
        entries = self.logger.get_entries(
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )
        
        # 检测频繁访问同一资源的模式
        resource_access: Dict[str, List[AccessLogEntry]] = defaultdict(list)
        for entry in entries:
            resource_access[entry.resource].append(entry)
        
        for resource, access_list in resource_access.items():
            if len(access_list) >= 100:  # 高频访问阈值
                unique_users = set(e.user for e in access_list)
                denied_count = sum(1 for e in access_list if not e.granted)
                
                risk_score = min(100, len(access_list) / 10 + denied_count * 5)
                
                patterns.append(AccessPattern(
                    pattern_type="high_frequency_access",
                    frequency=len(access_list),
                    entities=[resource] + list(unique_users)[:10],
                    risk_score=risk_score,
                    description=f"Resource '{resource}' accessed {len(access_list)} times by {len(unique_users)} users"
                ))
        
        # 检测多次拒绝访问的模式
        user_denied: Dict[str, List[AccessLogEntry]] = defaultdict(list)
        for entry in entries:
            if not entry.granted:
                user_denied[entry.user].append(entry)
        
        for user, denied_list in user_denied.items():
            if len(denied_list) >= 10:  # 多次拒绝阈值
                risk_score = min(100, len(denied_list) * 5)
                
                patterns.append(AccessPattern(
                    pattern_type="repeated_denied_access",
                    frequency=len(denied_list),
                    entities=[user],
                    risk_score=risk_score,
                    description=f"User '{user}' had {len(denied_list)} denied access attempts"
                ))
        
        # 检测异常时间访问模式
        off_hours_access = [
            e for e in entries
            if e.timestamp.hour < 6 or e.timestamp.hour > 22
        ]
        
        if len(off_hours_access) >= 20:
            unique_users = set(e.user for e in off_hours_access)
            patterns.append(AccessPattern(
                pattern_type="off_hours_access",
                frequency=len(off_hours_access),
                entities=list(unique_users)[:10],
                risk_score=40.0,
                description=f"{len(off_hours_access)} access attempts during off-hours"
            ))
        
        return patterns


class SessionTracker:
    """
    会话追踪器
    
    提供会话生命周期管理和异常检测功能。
    """
    
    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._user_sessions: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()
    
    def track_session_start(
        self,
        session_id: str,
        user: str,
        ip_address: str,
        user_agent: str = "",
        location: str = ""
    ) -> Dict[str, Any]:
        """
        追踪会话开始
        
        Args:
            session_id: 会话ID
            user: 用户标识
            ip_address: IP地址
            user_agent: 用户代理
            location: 位置信息
            
        Returns:
            会话信息字典
        """
        session_info = {
            "session_id": session_id,
            "user": user,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "location": location,
            "start_time": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "access_count": 0,
            "denied_count": 0,
            "resources_accessed": set(),
            "active": True
        }
        
        with self._lock:
            self._sessions[session_id] = session_info
            self._user_sessions[user].add(session_id)
        
        return session_info
    
    def track_session_end(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        追踪会话结束
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话信息字典，如果会话不存在则返回None
        """
        with self._lock:
            if session_id not in self._sessions:
                return None
            
            session = self._sessions[session_id]
            session["active"] = False
            session["end_time"] = datetime.utcnow()
            session["duration_seconds"] = (
                session["end_time"] - session["start_time"]
            ).total_seconds()
            
            return session
    
    def track_session_activity(
        self,
        session_id: str,
        resource: str,
        action: str,
        granted: bool = True
    ) -> bool:
        """
        追踪会话活动
        
        Args:
            session_id: 会话ID
            resource: 访问的资源
            action: 操作类型
            granted: 是否授权通过
            
        Returns:
            是否成功记录
        """
        with self._lock:
            if session_id not in self._sessions:
                return False
            
            session = self._sessions[session_id]
            session["last_activity"] = datetime.utcnow()
            session["access_count"] += 1
            
            if not granted:
                session["denied_count"] += 1
            
            session["resources_accessed"].add(resource)
            
            return True
    
    def detect_session_anomaly(self, session_id: str) -> List[Dict[str, Any]]:
        """
        检测会话异常
        
        Args:
            session_id: 会话ID
            
        Returns:
            检测到的异常列表
        """
        anomalies = []
        
        with self._lock:
            if session_id not in self._sessions:
                return [{"type": "unknown_session", "description": "Session not found"}]
            
            session = self._sessions[session_id]
            user = session["user"]
            
            # 检查长时间会话
            duration = (datetime.utcnow() - session["start_time"]).total_seconds()
            if duration > 8 * 3600:  # 8小时
                anomalies.append({
                    "type": "long_duration_session",
                    "severity": "medium",
                    "description": f"Session has been active for {duration / 3600:.1f} hours"
                })
            
            # 检查高拒绝率
            if session["access_count"] > 0:
                denied_rate = session["denied_count"] / session["access_count"]
                if denied_rate > 0.3:  # 30%拒绝率
                    anomalies.append({
                        "type": "high_denial_rate",
                        "severity": "high",
                        "description": f"Session has {denied_rate * 100:.1f}% denial rate"
                    })
            
            # 检查异常资源访问数量
            if len(session["resources_accessed"]) > 100:
                anomalies.append({
                    "type": "excessive_resource_access",
                    "severity": "medium",
                    "description": f"Session accessed {len(session['resources_accessed'])} unique resources"
                })
            
            # 检查用户并发会话
            user_session_count = len(self._user_sessions.get(user, set()))
            if user_session_count > 5:
                anomalies.append({
                    "type": "concurrent_sessions",
                    "severity": "low",
                    "description": f"User has {user_session_count} concurrent sessions"
                })
            
            # 检查不活跃会话
            inactivity = (datetime.utcnow() - session["last_activity"]).total_seconds()
            if inactivity > 3600:  # 1小时无活动
                anomalies.append({
                    "type": "inactive_session",
                    "severity": "low",
                    "description": f"Session inactive for {inactivity / 60:.0f} minutes"
                })
        
        return anomalies
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话信息
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话信息字典
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                # 创建副本并转换集合为列表
                info = dict(session)
                info["resources_accessed"] = list(session["resources_accessed"])
                return info
            return None
    
    def get_user_sessions(self, user: str) -> List[Dict[str, Any]]:
        """
        获取用户的所有会话
        
        Args:
            user: 用户标识
            
        Returns:
            会话信息列表
        """
        with self._lock:
            session_ids = self._user_sessions.get(user, set())
            return [
                self.get_session_info(sid)
                for sid in session_ids
                if sid in self._sessions
            ]
    
    def cleanup_inactive_sessions(self, max_inactive_seconds: float = 3600) -> int:
        """
        清理不活跃会话
        
        Args:
            max_inactive_seconds: 最大不活跃时间（秒）
            
        Returns:
            清理的会话数量
        """
        cleaned = 0
        now = datetime.utcnow()
        
        with self._lock:
            to_remove = []
            for session_id, session in self._sessions.items():
                if not session["active"]:
                    to_remove.append(session_id)
                else:
                    inactive_time = (now - session["last_activity"]).total_seconds()
                    if inactive_time > max_inactive_seconds:
                        to_remove.append(session_id)
            
            for session_id in to_remove:
                session = self._sessions.pop(session_id, None)
                if session:
                    user = session["user"]
                    self._user_sessions[user].discard(session_id)
                    cleaned += 1
        
        return cleaned
