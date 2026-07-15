"""
工作流审计日志模块 (Workflow Audit Logger)

提供工作流执行的完整审计日志功能：
- 记录每个节点执行（输入/输出/时间/异常）
- 可搜索的审计跟踪
- 合规报告生成

类:
    WorkflowAuditLogger: 工作流审计日志器
    AuditEntry: 审计条目
    AuditSearch: 审计搜索
    AuditReporter: 审计报告生成器
    ComplianceReport: 合规报告
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Generator
import uuid
import threading
import pickle
import gzip
from pathlib import Path

# 配置日志
logger = logging.getLogger(__name__)

# 类型变量
T = TypeVar('T')


class AuditLevel(Enum):
    """审计级别"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditCategory(Enum):
    """审计类别"""
    WORKFLOW_START = "workflow_start"
    WORKFLOW_END = "workflow_end"
    WORKFLOW_PAUSE = "workflow_pause"
    WORKFLOW_RESUME = "workflow_resume"
    WORKFLOW_CANCEL = "workflow_cancel"
    NODE_EXECUTE = "node_execute"
    NODE_SUCCESS = "node_success"
    NODE_FAILURE = "node_failure"
    NODE_SKIP = "node_skip"
    NODE_RETRY = "node_retry"
    CONDITION_EVAL = "condition_eval"
    VARIABLE_SET = "variable_set"
    VARIABLE_GET = "variable_get"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    SECURITY_EVENT = "security_event"
    DATA_ACCESS = "data_access"
    DATA_MODIFY = "data_modify"


class AuditSearchOperator(Enum):
    """审计搜索操作符"""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    BETWEEN = "between"
    IN_LIST = "in_list"
    NOT_IN_LIST = "not_in_list"
    REGEX = "regex"


@dataclass
class AuditEntry:
    """审计条目"""
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    level: AuditLevel = AuditLevel.INFO
    category: AuditCategory = AuditCategory.SYSTEM_EVENT
    workflow_id: Optional[str] = None
    workflow_name: Optional[str] = None
    node_id: Optional[str] = None
    node_name: Optional[str] = None
    execution_id: Optional[str] = None
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    action: str = ""
    resource: Optional[str] = None
    input_data: Optional[Any] = None
    output_data: Optional[Any] = None
    error: Optional[str] = None
    error_trace: Optional[str] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "category": self.category.value,
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "node_id": self.node_id,
            "node_name": self.node_name,
            "execution_id": self.execution_id,
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "action": self.action,
            "resource": self.resource,
            "input_data": self._serialize_data(self.input_data),
            "output_data": self._serialize_data(self.output_data),
            "error": self.error,
            "error_trace": self.error_trace,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "tags": list(self.tags),
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
        }
    
    def _serialize_data(self, data: Any) -> Optional[Any]:
        """序列化数据"""
        if data is None:
            return None
        try:
            # 尝试JSON序列化
            json.dumps(data)
            return data
        except (TypeError, ValueError):
            # 对于不可序列化的对象，转换为字符串
            return str(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AuditEntry:
        """从字典创建"""
        return cls(
            entry_id=data.get("entry_id", str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            level=AuditLevel(data.get("level", "info")),
            category=AuditCategory(data.get("category", "system_event")),
            workflow_id=data.get("workflow_id"),
            workflow_name=data.get("workflow_name"),
            node_id=data.get("node_id"),
            node_name=data.get("node_name"),
            execution_id=data.get("execution_id"),
            user_id=data.get("user_id"),
            ip_address=data.get("ip_address"),
            action=data.get("action", ""),
            resource=data.get("resource"),
            input_data=data.get("input_data"),
            output_data=data.get("output_data"),
            error=data.get("error"),
            error_trace=data.get("error_trace"),
            duration_ms=data.get("duration_ms"),
            metadata=data.get("metadata", {}),
            tags=set(data.get("tags", [])),
            session_id=data.get("session_id"),
            correlation_id=data.get("correlation_id"),
        )


@dataclass
class AuditSearchCriteria:
    """审计搜索条件"""
    workflow_id: Optional[str] = None
    workflow_name: Optional[str] = None
    node_id: Optional[str] = None
    node_name: Optional[str] = None
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    categories: Optional[List[AuditCategory]] = None
    levels: Optional[List[AuditLevel]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    keywords: Optional[List[str]] = None
    tags: Optional[Set[str]] = None
    has_error: Optional[bool] = None
    duration_min_ms: Optional[float] = None
    duration_max_ms: Optional[float] = None
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "node_id": self.node_id,
            "node_name": self.node_name,
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "categories": [c.value for c in self.categories] if self.categories else None,
            "levels": [l.value for l in self.levels] if self.levels else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "keywords": self.keywords,
            "tags": list(self.tags) if self.tags else None,
            "has_error": self.has_error,
            "duration_min_ms": self.duration_min_ms,
            "duration_max_ms": self.duration_max_ms,
            "correlation_id": self.correlation_id,
            "session_id": self.session_id,
        }


class AuditSearch:
    """审计搜索"""
    
    def __init__(
        self,
        entries: Optional[List[AuditEntry]] = None,
        storage_path: Optional[str] = None,
    ):
        self.entries: List[AuditEntry] = entries or []
        self.storage_path = storage_path
        self._index: Dict[str, List[int]] = {}
        self._build_index()
    
    def _build_index(self) -> None:
        """构建索引"""
        self._index = {
            "workflow_id": {},
            "node_id": {},
            "user_id": {},
            "category": {},
            "level": {},
            "tags": {},
        }
        
        for i, entry in enumerate(self.entries):
            # 工作流ID索引
            if entry.workflow_id:
                if entry.workflow_id not in self._index["workflow_id"]:
                    self._index["workflow_id"][entry.workflow_id] = []
                self._index["workflow_id"][entry.workflow_id].append(i)
            
            # 节点ID索引
            if entry.node_id:
                if entry.node_id not in self._index["node_id"]:
                    self._index["node_id"][entry.node_id] = []
                self._index["node_id"][entry.node_id].append(i)
            
            # 用户ID索引
            if entry.user_id:
                if entry.user_id not in self._index["user_id"]:
                    self._index["user_id"][entry.user_id] = []
                self._index["user_id"][entry.user_id].append(i)
            
            # 类别索引
            if entry.category.value not in self._index["category"]:
                self._index["category"][entry.category.value] = []
            self._index["category"][entry.category.value].append(i)
            
            # 级别索引
            if entry.level.value not in self._index["level"]:
                self._index["level"][entry.level.value] = []
            self._index["level"][entry.level.value].append(i)
            
            # 标签索引
            for tag in entry.tags:
                if tag not in self._index["tags"]:
                    self._index["tags"][tag] = []
                self._index["tags"][tag].append(i)
    
    def add_entry(self, entry: AuditEntry) -> None:
        """添加条目"""
        self.entries.append(entry)
        self._update_index(len(self.entries) - 1, entry)
    
    def _update_index(self, index: int, entry: AuditEntry) -> None:
        """更新索引"""
        if entry.workflow_id:
            if entry.workflow_id not in self._index["workflow_id"]:
                self._index["workflow_id"][entry.workflow_id] = []
            self._index["workflow_id"][entry.workflow_id].append(index)
        
        if entry.node_id:
            if entry.node_id not in self._index["node_id"]:
                self._index["node_id"][entry.node_id] = []
            self._index["node_id"][entry.node_id].append(index)
        
        if entry.user_id:
            if entry.user_id not in self._index["user_id"]:
                self._index["user_id"][entry.user_id] = []
            self._index["user_id"][entry.user_id].append(index)
        
        if entry.category.value not in self._index["category"]:
            self._index["category"][entry.category.value] = []
        self._index["category"][entry.category.value].append(index)
        
        if entry.level.value not in self._index["level"]:
            self._index["level"][entry.level.value] = []
        self._index["level"][entry.level.value].append(index)
        
        for tag in entry.tags:
            if tag not in self._index["tags"]:
                self._index["tags"][tag] = []
            self._index["tags"][tag].append(index)
    
    def search(self, criteria: AuditSearchCriteria) -> List[AuditEntry]:
        """搜索条目"""
        candidates: Optional[Set[int]] = None
        
        # 基于索引的快速筛选
        if criteria.workflow_id:
            indices = set(self._index.get("workflow_id", {}).get(criteria.workflow_id, []))
            candidates = candidates & indices if candidates else indices
        
        if criteria.node_id:
            indices = set(self._index.get("node_id", {}).get(criteria.node_id, []))
            candidates = candidates & indices if candidates else indices
        
        if criteria.user_id:
            indices = set(self._index.get("user_id", {}).get(criteria.user_id, []))
            candidates = candidates & indices if candidates else indices
        
        if criteria.categories:
            cat_indices: Set[int] = set()
            for cat in criteria.categories:
                cat_indices |= set(self._index.get("category", {}).get(cat.value, []))
            candidates = candidates & cat_indices if candidates else cat_indices
        
        if criteria.levels:
            level_indices: Set[int] = set()
            for level in criteria.levels:
                level_indices |= set(self._index.get("level", {}).get(level.value, []))
            candidates = candidates & level_indices if candidates else level_indices
        
        if criteria.tags:
            tag_indices: Set[int] = set()
            for tag in criteria.tags:
                tag_indices |= set(self._index.get("tags", {}).get(tag, []))
            candidates = candidates & tag_indices if candidates else tag_indices
        
        # 获取候选条目
        if candidates is not None:
            candidate_entries = [self.entries[i] for i in candidates]
        else:
            candidate_entries = self.entries
        
        # 详细筛选
        results: List[AuditEntry] = []
        for entry in candidate_entries:
            if not self._matches_criteria(entry, criteria):
                continue
            results.append(entry)
        
        # 时间排序（最新的在前）
        results.sort(key=lambda e: e.timestamp, reverse=True)
        
        return results
    
    def _matches_criteria(self, entry: AuditEntry, criteria: AuditSearchCriteria) -> bool:
        """检查条目是否匹配条件"""
        # 时间范围
        if criteria.start_time and entry.timestamp < criteria.start_time:
            return False
        if criteria.end_time and entry.timestamp > criteria.end_time:
            return False
        
        # 关键词
        if criteria.keywords:
            entry_str = json.dumps(entry.to_dict()).lower()
            if not all(kw.lower() in entry_str for kw in criteria.keywords):
                return False
        
        # 错误过滤
        if criteria.has_error is not None:
            has_error = entry.error is not None
            if has_error != criteria.has_error:
                return False
        
        # 时长过滤
        if criteria.duration_min_ms is not None:
            if entry.duration_ms is None or entry.duration_ms < criteria.duration_min_ms:
                return False
        if criteria.duration_max_ms is not None:
            if entry.duration_ms is None or entry.duration_ms > criteria.duration_max_ms:
                return False
        
        # 相关ID
        if criteria.correlation_id and entry.correlation_id != criteria.correlation_id:
            return False
        if criteria.session_id and entry.session_id != criteria.session_id:
            return False
        
        return True
    
    def export_to_file(self, file_path: str, format: str = "json") -> None:
        """导出到文件"""
        if format == "json":
            self._export_json(file_path)
        elif format == "csv":
            self._export_csv(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_json(self, file_path: str) -> None:
        """导出为JSON"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump([e.to_dict() for e in self.entries], f, ensure_ascii=False, indent=2)
    
    def _export_csv(self, file_path: str) -> None:
        """导出为CSV"""
        if not self.entries:
            return
        
        fieldnames = list(self.entries[0].to_dict().keys())
        
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self.entries:
                row = entry.to_dict()
                # 将嵌套对象转换为字符串
                for key, value in row.items():
                    if isinstance(value, (dict, list)):
                        row[key] = json.dumps(value, ensure_ascii=False)
                writer.writerow(row)
    
    def import_from_file(self, file_path: str, format: str = "json") -> int:
        """从文件导入"""
        if format == "json":
            return self._import_json(file_path)
        elif format == "csv":
            return self._import_csv(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _import_json(self, file_path: str) -> int:
        """从JSON导入"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        count = 0
        for item in data:
            entry = AuditEntry.from_dict(item)
            self.add_entry(entry)
            count += 1
        
        return count
    
    def _import_csv(self, file_path: str) -> int:
        """从CSV导入"""
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            count = 0
            for row in reader:
                # 解析时间戳
                if "timestamp" in row:
                    row["timestamp"] = datetime.fromisoformat(row["timestamp"])
                
                # 解析枚举
                if "level" in row:
                    row["level"] = AuditLevel(row["level"])
                if "category" in row:
                    row["category"] = AuditCategory(row["category"])
                
                # 解析标签
                if "tags" in row and row["tags"]:
                    row["tags"] = set(json.loads(row["tags"]))
                
                # 解析元数据
                if "metadata" in row and row["metadata"]:
                    row["metadata"] = json.loads(row["metadata"])
                
                entry = AuditEntry(**row)
                self.add_entry(entry)
                count += 1
        
        return count


class AuditReporter:
    """审计报告生成器"""
    
    def __init__(
        self,
        audit_search: AuditSearch,
        report_path: Optional[str] = None,
    ):
        self.audit_search = audit_search
        self.report_path = report_path
    
    def generate_summary_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """生成汇总报告"""
        criteria = AuditSearchCriteria(start_time=start_time, end_time=end_time)
        entries = self.audit_search.search(criteria)
        
        # 统计信息
        total_entries = len(entries)
        workflow_count = len(set(e.workflow_id for e in entries if e.workflow_id))
        node_count = len(set(e.node_id for e in entries if e.node_id))
        user_count = len(set(e.user_id for e in entries if e.user_id))
        
        # 按类别统计
        category_counts: Dict[str, int] = {}
        for entry in entries:
            cat = entry.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # 按级别统计
        level_counts: Dict[str, int] = {}
        for entry in entries:
            level = entry.level.value
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # 错误统计
        error_entries = [e for e in entries if e.error]
        error_count = len(error_entries)
        
        # 时长统计
        durations = [e.duration_ms for e in entries if e.duration_ms is not None]
        if durations:
            avg_duration = sum(durations) / len(durations)
            min_duration = min(durations)
            max_duration = max(durations)
        else:
            avg_duration = min_duration = max_duration = 0
        
        return {
            "report_type": "summary",
            "generated_at": datetime.now().isoformat(),
            "time_range": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None,
            },
            "statistics": {
                "total_entries": total_entries,
                "unique_workflows": workflow_count,
                "unique_nodes": node_count,
                "unique_users": user_count,
                "error_count": error_count,
                "error_rate": error_count / total_entries if total_entries > 0 else 0,
            },
            "category_distribution": category_counts,
            "level_distribution": level_counts,
            "duration_stats": {
                "average_ms": avg_duration,
                "min_ms": min_duration,
                "max_ms": max_duration,
            },
        }
    
    def generate_workflow_report(
        self,
        workflow_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """生成工作流报告"""
        criteria = AuditSearchCriteria(
            workflow_id=workflow_id,
            start_time=start_time,
            end_time=end_time,
        )
        entries = self.audit_search.search(criteria)
        
        if not entries:
            return {"error": "No entries found for workflow"}
        
        workflow_name = entries[0].workflow_name
        
        # 节点执行记录
        node_executions: Dict[str, List[AuditEntry]] = {}
        for entry in entries:
            if entry.node_id:
                if entry.node_id not in node_executions:
                    node_executions[entry.node_id] = []
                node_executions[entry.node_id].append(entry)
        
        # 节点统计
        node_stats: Dict[str, Dict[str, Any]] = {}
        for node_id, node_entries in node_executions.items():
            success_entries = [e for e in node_entries if e.category == AuditCategory.NODE_SUCCESS]
            error_entries = [e for e in node_entries if e.category == AuditCategory.NODE_FAILURE]
            durations = [e.duration_ms for e in node_entries if e.duration_ms is not None]
            
            node_stats[node_id] = {
                "node_name": node_entries[0].node_name,
                "total_executions": len(node_entries),
                "success_count": len(success_entries),
                "failure_count": len(error_entries),
                "success_rate": len(success_entries) / len(node_entries) if node_entries else 0,
                "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
            }
        
        return {
            "report_type": "workflow",
            "generated_at": datetime.now().isoformat(),
            "workflow_id": workflow_id,
            "workflow_name": workflow_name,
            "time_range": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None,
            },
            "total_entries": len(entries),
            "node_statistics": node_stats,
        }
    
    def generate_compliance_report(
        self,
        compliance_rules: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """生成合规报告"""
        criteria = AuditSearchCriteria(
            start_time=start_time,
            end_time=end_time,
            categories=[
                AuditCategory.DATA_ACCESS,
                AuditCategory.DATA_MODIFY,
                AuditCategory.SECURITY_EVENT,
            ],
        )
        entries = self.audit_search.search(criteria)
        
        # 数据访问记录
        data_access_entries = [
            e for e in entries
            if e.category == AuditCategory.DATA_ACCESS
        ]
        
        # 数据修改记录
        data_modify_entries = [
            e for e in entries
            if e.category == AuditCategory.DATA_MODIFY
        ]
        
        # 安全事件
        security_entries = [
            e for e in entries
            if e.category == AuditCategory.SECURITY_EVENT
        ]
        
        # 失败操作
        failed_entries = [e for e in entries if e.error]
        
        return {
            "report_type": "compliance",
            "generated_at": datetime.now().isoformat(),
            "time_range": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None,
            },
            "compliance_checks": {
                "data_access_audit": {
                    "status": "pass" if data_access_entries else "warning",
                    "count": len(data_access_entries),
                },
                "data_modification_audit": {
                    "status": "pass" if data_modify_entries else "warning",
                    "count": len(data_modify_entries),
                },
                "security_event_monitoring": {
                    "status": "pass" if not security_entries else "alert",
                    "count": len(security_entries),
                },
                "error_tracking": {
                    "status": "fail" if failed_entries else "pass",
                    "count": len(failed_entries),
                },
            },
            "total_records": len(entries),
            "compliance_score": self._calculate_compliance_score(entries),
        }
    
    def _calculate_compliance_score(self, entries: List[AuditEntry]) -> float:
        """计算合规分数"""
        if not entries:
            return 100.0
        
        # 基础分数
        score = 100.0
        
        # 扣分项
        critical_entries = [e for e in entries if e.level == AuditLevel.CRITICAL]
        error_entries = [e for e in entries if e.error]
        
        score -= len(critical_entries) * 10
        score -= len(error_entries) * 2
        
        return max(0.0, min(100.0, score))
    
    def export_report(
        self,
        report: Dict[str, Any],
        file_path: Optional[str] = None,
        format: str = "json",
    ) -> str:
        """导出报告"""
        target_path = file_path or self.report_path
        
        if not target_path:
            raise ValueError("No report path specified")
        
        if format == "json":
            with open(target_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        elif format == "html":
            html = self._generate_html_report(report)
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(html)
        elif format == "markdown":
            markdown = self._generate_markdown_report(report)
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(markdown)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return target_path
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """生成HTML报告"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Audit Report - {report.get('report_type', 'Unknown')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f0f0f0; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Audit Report - {report.get('report_type', 'Unknown')}</h1>
    <p>Generated at: {report.get('generated_at', 'N/A')}</p>
    <pre>{json.dumps(report, indent=2, ensure_ascii=False)}</pre>
</body>
</html>
"""
        return html
    
    def _generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """生成Markdown报告"""
        md = f"""# Audit Report - {report.get('report_type', 'Unknown')}

Generated at: {report.get('generated_at', 'N/A')}

## Report Data

```
{json.dumps(report, indent=2, ensure_ascii=False)}
```
"""
        return md


class ComplianceReport:
    """合规报告"""
    
    def __init__(
        self,
        report_id: Optional[str] = None,
        report_type: str = "standard",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ):
        self.report_id = report_id or str(uuid.uuid4())
        self.report_type = report_type
        self.start_time = start_time or datetime.now() - timedelta(days=1)
        self.end_time = end_time or datetime.now()
        self.created_at = datetime.now()
        self.entries: List[AuditEntry] = []
        self.findings: List[Dict[str, Any]] = []
        self.recommendations: List[str] = []
    
    def add_entry(self, entry: AuditEntry) -> None:
        """添加条目"""
        self.entries.append(entry)
    
    def add_findings(self, finding: Dict[str, Any]) -> None:
        """添加发现"""
        self.findings.append(finding)
    
    def add_recommendation(self, recommendation: str) -> None:
        """添加建议"""
        self.recommendations.append(recommendation)
    
    def generate(self) -> Dict[str, Any]:
        """生成报告"""
        return {
            "report_id": self.report_id,
            "report_type": self.report_type,
            "time_range": {
                "start": self.start_time.isoformat(),
                "end": self.end_time.isoformat(),
            },
            "created_at": self.created_at.isoformat(),
            "summary": {
                "total_entries": len(self.entries),
                "findings_count": len(self.findings),
                "recommendations_count": len(self.recommendations),
            },
            "findings": self.findings,
            "recommendations": self.recommendations,
        }


class WorkflowAuditLogger:
    """工作流审计日志器"""
    
    _instance: Optional[WorkflowAuditLogger] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args: Any, **kwargs: Any) -> WorkflowAuditLogger:
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        max_entries: int = 100000,
        enable_compression: bool = True,
        enable_persistence: bool = True,
        retention_days: int = 90,
    ):
        if self._initialized:
            return
        
        self.storage_path = storage_path
        self.max_entries = max_entries
        self.enable_compression = enable_compression
        self.enable_persistence = enable_persistence
        self.retention_days = retention_days
        
        self._entries: List[AuditEntry] = []
        self._search = AuditSearch()
        self._reporter: Optional[AuditReporter] = None
        self._callbacks: Dict[str, List[Callable]] = {}
        self._context: Dict[str, Any] = {}
        
        self._initialized = True
        
        # 加载持久化数据
        if self.enable_persistence and self.storage_path:
            self._load_persistence()
    
    def _load_persistence(self) -> None:
        """加载持久化数据"""
        if not self.storage_path:
            return
        
        persistence_file = os.path.join(self.storage_path, "audit_log.pkl.gz")
        
        if os.path.exists(persistence_file):
            try:
                with gzip.open(persistence_file, 'rb') as f:
                    data = pickle.load(f)
                    self._entries = [AuditEntry.from_dict(e) for e in data.get("entries", [])]
                    self._search = AuditSearch(entries=self._entries)
                    logger.info(f"Loaded {len(self._entries)} audit entries from persistence")
            except Exception as e:
                logger.error(f"Failed to load audit persistence: {e}")
    
    def _save_persistence(self) -> None:
        """保存持久化数据"""
        if not self.enable_persistence or not self.storage_path:
            return
        
        os.makedirs(self.storage_path, exist_ok=True)
        persistence_file = os.path.join(self.storage_path, "audit_log.pkl.gz")
        
        try:
            with gzip.open(persistence_file, 'wb') as f:
                pickle.dump({
                    "entries": [e.to_dict() for e in self._entries[-self.max_entries:]],
                }, f)
            logger.debug("Saved audit log to persistence")
        except Exception as e:
            logger.error(f"Failed to save audit persistence: {e}")
    
    def set_context(
        self,
        workflow_id: Optional[str] = None,
        workflow_name: Optional[str] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        """设置上下文"""
        if workflow_id is not None:
            self._context["workflow_id"] = workflow_id
        if workflow_name is not None:
            self._context["workflow_name"] = workflow_name
        if user_id is not None:
            self._context["user_id"] = user_id
        if ip_address is not None:
            self._context["ip_address"] = ip_address
        if session_id is not None:
            self._context["session_id"] = session_id
        if correlation_id is not None:
            self._context["correlation_id"] = correlation_id
    
    def clear_context(self) -> None:
        """清除上下文"""
        self._context = {}
    
    def log(
        self,
        action: str,
        level: AuditLevel = AuditLevel.INFO,
        category: AuditCategory = AuditCategory.SYSTEM_EVENT,
        node_id: Optional[str] = None,
        node_name: Optional[str] = None,
        input_data: Optional[Any] = None,
        output_data: Optional[Any] = None,
        error: Optional[Exception] = None,
        duration_ms: Optional[float] = None,
        tags: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """记录审计条目"""
        entry = AuditEntry(
            timestamp=datetime.now(),
            level=level,
            category=category,
            workflow_id=self._context.get("workflow_id"),
            workflow_name=self._context.get("workflow_name"),
            node_id=node_id,
            node_name=node_name,
            user_id=self._context.get("user_id"),
            ip_address=self._context.get("ip_address"),
            action=action,
            input_data=input_data,
            output_data=output_data,
            error=str(error) if error else None,
            error_trace=traceback.format_exc() if error else None,
            duration_ms=duration_ms,
            tags=tags or set(),
            metadata=metadata or {},
            session_id=self._context.get("session_id"),
            correlation_id=self._context.get("correlation_id"),
        )
        
        self._entries.append(entry)
        self._search.add_entry(entry)
        
        # 触发回调
        self._trigger_callbacks(category, entry)
        
        # 清理过期数据
        self._cleanup_old_entries()
        
        # 定期保存
        if len(self._entries) % 100 == 0:
            self._save_persistence()
        
        return entry
    
    def log_workflow_start(
        self,
        workflow_id: str,
        workflow_name: str,
        input_data: Optional[Any] = None,
    ) -> AuditEntry:
        """记录工作流启动"""
        return self.log(
            action=f"Workflow started: {workflow_name}",
            level=AuditLevel.INFO,
            category=AuditCategory.WORKFLOW_START,
            input_data=input_data,
            tags={"workflow", "start"},
        )
    
    def log_workflow_end(
        self,
        workflow_id: str,
        workflow_name: str,
        output_data: Optional[Any] = None,
        duration_ms: Optional[float] = None,
    ) -> AuditEntry:
        """记录工作流结束"""
        return self.log(
            action=f"Workflow completed: {workflow_name}",
            level=AuditLevel.INFO,
            category=AuditCategory.WORKFLOW_END,
            output_data=output_data,
            duration_ms=duration_ms,
            tags={"workflow", "end"},
        )
    
    def log_node_execute(
        self,
        node_id: str,
        node_name: str,
        input_data: Optional[Any] = None,
    ) -> AuditEntry:
        """记录节点执行"""
        return self.log(
            action=f"Node executing: {node_name}",
            level=AuditLevel.DEBUG,
            category=AuditCategory.NODE_EXECUTE,
            node_id=node_id,
            node_name=node_name,
            input_data=input_data,
            tags={"node", "execute"},
        )
    
    def log_node_success(
        self,
        node_id: str,
        node_name: str,
        output_data: Optional[Any] = None,
        duration_ms: Optional[float] = None,
    ) -> AuditEntry:
        """记录节点成功"""
        return self.log(
            action=f"Node succeeded: {node_name}",
            level=AuditLevel.INFO,
            category=AuditCategory.NODE_SUCCESS,
            node_id=node_id,
            node_name=node_name,
            output_data=output_data,
            duration_ms=duration_ms,
            tags={"node", "success"},
        )
    
    def log_node_failure(
        self,
        node_id: str,
        node_name: str,
        error: Exception,
        duration_ms: Optional[float] = None,
    ) -> AuditEntry:
        """记录节点失败"""
        return self.log(
            action=f"Node failed: {node_name}",
            level=AuditLevel.ERROR,
            category=AuditCategory.NODE_FAILURE,
            node_id=node_id,
            node_name=node_name,
            error=error,
            duration_ms=duration_ms,
            tags={"node", "failure", "error"},
        )
    
    def log_workflow_pause(self, workflow_id: str, reason: str) -> AuditEntry:
        """记录工作流暂停"""
        return self.log(
            action=f"Workflow paused: {reason}",
            level=AuditLevel.WARNING,
            category=AuditCategory.WORKFLOW_PAUSE,
            metadata={"reason": reason},
            tags={"workflow", "pause"},
        )
    
    def log_workflow_resume(self, workflow_id: str) -> AuditEntry:
        """记录工作流恢复"""
        return self.log(
            action="Workflow resumed",
            level=AuditLevel.INFO,
            category=AuditCategory.WORKFLOW_RESUME,
            tags={"workflow", "resume"},
        )
    
    def log_workflow_cancel(self, workflow_id: str, reason: str) -> AuditEntry:
        """记录工作流取消"""
        return self.log(
            action=f"Workflow cancelled: {reason}",
            level=AuditLevel.WARNING,
            category=AuditCategory.WORKFLOW_CANCEL,
            metadata={"reason": reason},
            tags={"workflow", "cancel"},
        )
    
    def search(self, criteria: AuditSearchCriteria) -> List[AuditEntry]:
        """搜索审计条目"""
        return self._search.search(criteria)
    
    def get_reporter(self) -> AuditReporter:
        """获取报告生成器"""
        if self._reporter is None:
            self._reporter = AuditReporter(self._search, self.storage_path)
        return self._reporter
    
    def register_callback(
        self,
        category: AuditCategory,
        callback: Callable[[AuditEntry], None],
    ) -> str:
        """注册回调"""
        callback_id = str(uuid.uuid4())
        if category.value not in self._callbacks:
            self._callbacks[category.value] = []
        self._callbacks[category.value].append(callback)
        return callback_id
    
    def unregister_callback(self, callback_id: str) -> bool:
        """取消注册回调"""
        for callbacks in self._callbacks.values():
            for i, cb in enumerate(callbacks):
                if hasattr(cb, '__callback_id__') and cb.__callback_id__ == callback_id:
                    callbacks.pop(i)
                    return True
        return False
    
    def _trigger_callbacks(self, category: AuditCategory, entry: AuditEntry) -> None:
        """触发回调"""
        callbacks = self._callbacks.get(category.value, [])
        for callback in callbacks:
            try:
                callback(entry)
            except Exception as e:
                logger.error(f"Error in audit callback: {e}")
    
    def _cleanup_old_entries(self) -> None:
        """清理过期条目"""
        cutoff_time = datetime.now() - timedelta(days=self.retention_days)
        
        while self._entries and self._entries[0].timestamp < cutoff_time:
            self._entries.pop(0)
        
        while len(self._entries) > self.max_entries:
            self._entries.pop(0)
        
        # 重建索引
        self._search = AuditSearch(entries=self._entries)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_entries": len(self._entries),
            "max_entries": self.max_entries,
            "storage_path": self.storage_path,
            "retention_days": self.retention_days,
            "enable_compression": self.enable_compression,
            "enable_persistence": self.enable_persistence,
        }
    
    def export(self, file_path: str, format: str = "json") -> None:
        """导出审计日志"""
        self._search.export_to_file(file_path, format)
    
    def clear(self) -> int:
        """清除所有条目"""
        count = len(self._entries)
        self._entries.clear()
        self._search = AuditSearch()
        self._save_persistence()
        return count


# 全局审计日志器实例
_default_logger: Optional[WorkflowAuditLogger] = None


def get_audit_logger(
    storage_path: Optional[str] = None,
    **kwargs: Any
) -> WorkflowAuditLogger:
    """获取全局审计日志器"""
    global _default_logger
    
    if _default_logger is None:
        _default_logger = WorkflowAuditLogger(storage_path=storage_path, **kwargs)
    
    return _default_logger


__all__ = [
    # 枚举类型
    "AuditLevel",
    "AuditCategory",
    "AuditSearchOperator",
    # 数据类
    "AuditEntry",
    "AuditSearchCriteria",
    # 核心类
    "AuditSearch",
    "AuditReporter",
    "ComplianceReport",
    "WorkflowAuditLogger",
    # 辅助函数
    "get_audit_logger",
]
