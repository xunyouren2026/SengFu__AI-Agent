"""
数据库查询插件

提供SQL执行、结果格式化和连接管理功能。
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import threading


@dataclass
class ConnectionConfig:
    """连接配置"""
    host: str
    port: int
    database: str
    username: str
    password: str = ""
    driver: str = "postgresql"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'host': self.host,
            'port': self.port,
            'database': self.database,
            'username': self.username,
            'driver': self.driver,
        }


@dataclass
class QueryResult:
    """查询结果"""
    success: bool
    columns: List[str] = field(default_factory=list)
    rows: List[List[Any]] = field(default_factory=list)
    row_count: int = 0
    execution_time_ms: float = 0.0
    error: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'columns': self.columns,
            'rows': self.rows,
            'row_count': self.row_count,
            'execution_time_ms': self.execution_time_ms,
            'error': self.error,
        }


class DatabasePlugin:
    """数据库查询插件
    
    提供SQL执行、结果格式化和连接管理。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置字典
        """
        self._config = config or {}
        self._connections: Dict[str, ConnectionConfig] = {}
        self._lock = threading.RLock()
    
    def add_connection(self, name: str, config: ConnectionConfig) -> bool:
        """添加连接配置
        
        Args:
            name: 连接名称
            config: 连接配置
            
        Returns:
            是否成功
        """
        with self._lock:
            self._connections[name] = config
            return True
    
    def execute(self, sql: str, connection_name: str = "default") -> QueryResult:
        """执行SQL
        
        Args:
            sql: SQL语句
            connection_name: 连接名称
            
        Returns:
            查询结果
        """
        import time
        
        start = time.time()
        
        # 安全检查
        if not self._is_safe_sql(sql):
            return QueryResult(
                success=False,
                error="SQL contains unsafe operations",
            )
        
        # 模拟执行
        # 实际实现应使用数据库驱动
        
        execution_time = (time.time() - start) * 1000
        
        return QueryResult(
            success=True,
            columns=['id', 'name', 'value'],
            rows=[[1, 'test', 100]],
            row_count=1,
            execution_time_ms=execution_time,
        )
    
    def execute_batch(self, sql_statements: List[str],
                      connection_name: str = "default") -> List[QueryResult]:
        """批量执行SQL
        
        Args:
            sql_statements: SQL语句列表
            connection_name: 连接名称
            
        Returns:
            结果列表
        """
        results = []
        
        for sql in sql_statements:
            results.append(self.execute(sql, connection_name))
        
        return results
    
    def format_result(self, result: QueryResult,
                      format: str = "json") -> str:
        """格式化结果
        
        Args:
            result: 查询结果
            format: 格式 (json, csv, markdown)
            
        Returns:
            格式化字符串
        """
        if format == "json":
            return json.dumps(result.to_dict(), indent=2)
        
        elif format == "csv":
            lines = [",".join(result.columns)]
            for row in result.rows:
                lines.append(",".join(str(c) for c in row))
            return "\n".join(lines)
        
        elif format == "markdown":
            lines = ["| " + " | ".join(result.columns) + " |"]
            lines.append("| " + " | ".join(["---"] * len(result.columns)) + " |")
            for row in result.rows:
                lines.append("| " + " | ".join(str(c) for c in row) + " |")
            return "\n".join(lines)
        
        else:
            return str(result)
    
    def validate_sql(self, sql: str) -> Dict[str, Any]:
        """验证SQL语法
        
        Args:
            sql: SQL语句
            
        Returns:
            验证结果
        """
        # 简化实现
        errors = []
        
        # 检查基本语法
        if not sql.strip().endswith(';'):
            sql = sql.strip() + ';'
        
        # 检查危险操作
        dangerous = ['DROP DATABASE', 'DROP TABLE', 'DELETE FROM', 'TRUNCATE']
        for op in dangerous:
            if re.search(rf'\b{op}\b', sql, re.IGNORECASE):
                errors.append(f"Potentially dangerous operation: {op}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
        }
    
    def _is_safe_sql(self, sql: str) -> bool:
        """检查SQL是否安全"""
        # 禁止的操作
        forbidden = [
            r'DROP\s+DATABASE',
            r'DROP\s+USER',
            r'GRANT\s+ALL',
        ]
        
        for pattern in forbidden:
            if re.search(pattern, sql, re.IGNORECASE):
                return False
        
        return True
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取插件元数据"""
        return {
            'name': 'database',
            'version': '1.0.0',
            'description': 'Database query plugin with SQL execution support',
        }
