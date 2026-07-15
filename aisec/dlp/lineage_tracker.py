"""
Data Lineage Tracker Module - 数据血缘追踪器

提供完整的数据血缘追踪能力：
- 数据来源追踪
- 数据转换追踪
- 数据访问追踪
- 血缘图分析（DAG结构）
- 影响分析
"""

import json
import uuid
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from pathlib import Path
from collections import defaultdict
from datetime import datetime


class NodeType(Enum):
    """节点类型"""
    SOURCE = "source"           # 数据源
    TRANSFORMATION = "transformation"  # 转换
    SINK = "sink"               # 数据汇
    STORAGE = "storage"         # 存储
    QUERY = "query"             # 查询


@dataclass
class DataOrigin:
    """数据来源"""
    source_system: str
    source_location: str
    ingestion_time: Optional[str] = None
    owner: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.ingestion_time is None:
            self.ingestion_time = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_system": self.source_system,
            "source_location": self.source_location,
            "ingestion_time": self.ingestion_time,
            "owner": self.owner,
            "metadata": self.metadata
        }


@dataclass
class TransformationRecord:
    """转换记录"""
    operation: str
    parameters: Dict[str, Any]
    input_ids: List[str]
    output_id: str
    executor: str
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "parameters": self.parameters,
            "input_ids": self.input_ids,
            "output_id": self.output_id,
            "executor": self.executor,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class AccessRecord:
    """访问记录"""
    accessor: str
    action: str
    timestamp: str
    location: str = ""
    purpose: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "accessor": self.accessor,
            "action": self.action,
            "timestamp": self.timestamp,
            "location": self.location,
            "purpose": self.purpose,
            "metadata": self.metadata
        }


@dataclass
class LineageNode:
    """血缘节点"""
    id: str
    type: NodeType
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


@dataclass
class LineageEdge:
    """血缘边"""
    from_node: str
    to_node: str
    operation: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_node": self.from_node,
            "to_node": self.to_node,
            "operation": self.operation,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


class LineageGraph:
    """血缘图 - DAG结构"""
    
    def __init__(self):
        self.nodes: Dict[str, LineageNode] = {}
        self.edges: List[LineageEdge] = []
        self._adjacency: Dict[str, List[str]] = defaultdict(list)  # 出边
        self._reverse_adjacency: Dict[str, List[str]] = defaultdict(list)  # 入边
    
    def add_node(self, node: LineageNode) -> bool:
        """添加节点"""
        if node.id in self.nodes:
            return False
        self.nodes[node.id] = node
        return True
    
    def add_edge(self, edge: LineageEdge) -> bool:
        """添加边"""
        if edge.from_node not in self.nodes or edge.to_node not in self.nodes:
            return False
        
        # 检查是否会产生环
        if self._would_create_cycle(edge.from_node, edge.to_node):
            return False
        
        self.edges.append(edge)
        self._adjacency[edge.from_node].append(edge.to_node)
        self._reverse_adjacency[edge.to_node].append(edge.from_node)
        return True
    
    def _would_create_cycle(self, from_node: str, to_node: str) -> bool:
        """检查添加边是否会创建环"""
        # 如果to_node可以到达from_node，则添加from_node->to_node会形成环
        visited = set()
        stack = [to_node]
        
        while stack:
            current = stack.pop()
            if current == from_node:
                return True
            if current in visited:
                continue
            visited.add(current)
            stack.extend(self._adjacency[current])
        
        return False
    
    def get_upstream(self, node_id: str, depth: int = -1) -> List[LineageNode]:
        """
        获取上游节点
        
        Args:
            node_id: 节点ID
            depth: 最大深度，-1表示无限制
            
        Returns:
            上游节点列表
        """
        if node_id not in self.nodes:
            return []
        
        result = []
        visited = set()
        queue = [(node_id, 0)]
        
        while queue:
            current, current_depth = queue.pop(0)
            
            if depth >= 0 and current_depth > depth:
                continue
            
            for parent_id in self._reverse_adjacency[current]:
                if parent_id not in visited:
                    visited.add(parent_id)
                    result.append(self.nodes[parent_id])
                    queue.append((parent_id, current_depth + 1))
        
        return result
    
    def get_downstream(self, node_id: str, depth: int = -1) -> List[LineageNode]:
        """
        获取下游节点
        
        Args:
            node_id: 节点ID
            depth: 最大深度，-1表示无限制
            
        Returns:
            下游节点列表
        """
        if node_id not in self.nodes:
            return []
        
        result = []
        visited = set()
        queue = [(node_id, 0)]
        
        while queue:
            current, current_depth = queue.pop(0)
            
            if depth >= 0 and current_depth > depth:
                continue
            
            for child_id in self._adjacency[current]:
                if child_id not in visited:
                    visited.add(child_id)
                    result.append(self.nodes[child_id])
                    queue.append((child_id, current_depth + 1))
        
        return result
    
    def find_path(self, from_node: str, to_node: str) -> Optional[List[str]]:
        """
        查找从from_node到to_node的路径
        
        Returns:
            节点ID列表，如果不存在路径则返回None
        """
        if from_node not in self.nodes or to_node not in self.nodes:
            return None
        
        # BFS查找路径
        queue = [(from_node, [from_node])]
        visited = {from_node}
        
        while queue:
            current, path = queue.pop(0)
            
            if current == to_node:
                return path
            
            for neighbor in self._adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def get_roots(self) -> List[LineageNode]:
        """获取所有根节点（没有入边的节点）"""
        roots = []
        for node_id, node in self.nodes.items():
            if not self._reverse_adjacency[node_id]:
                roots.append(node)
        return roots
    
    def get_leaves(self) -> List[LineageNode]:
        """获取所有叶子节点（没有出边的节点）"""
        leaves = []
        for node_id, node in self.nodes.items():
            if not self._adjacency[node_id]:
                leaves.append(node)
        return leaves
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges]
        }
    
    def export(self, file_path: Union[str, Path]):
        """导出到文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


class DataLineageTracker:
    """数据血缘追踪器"""
    
    def __init__(self):
        self.graph = LineageGraph()
        self.origins: Dict[str, DataOrigin] = {}
        self.transformations: Dict[str, TransformationRecord] = {}
        self.access_records: Dict[str, List[AccessRecord]] = defaultdict(list)
        self._data_id_map: Dict[str, str] = {}  # 数据ID到节点ID的映射
    
    def track_origin(self, data_id: str, source: str, 
                     timestamp: Optional[str] = None,
                     owner: str = "",
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        追踪数据来源
        
        Args:
            data_id: 数据ID
            source: 数据来源系统
            timestamp: 时间戳
            owner: 数据所有者
            metadata: 元数据
            
        Returns:
            节点ID
        """
        node_id = str(uuid.uuid4())
        
        # 创建来源记录
        origin = DataOrigin(
            source_system=source,
            source_location=source,
            ingestion_time=timestamp or datetime.now().isoformat(),
            owner=owner,
            metadata=metadata or {}
        )
        self.origins[data_id] = origin
        
        # 创建节点
        node = LineageNode(
            id=node_id,
            type=NodeType.SOURCE,
            metadata={
                "data_id": data_id,
                "origin": origin.to_dict()
            }
        )
        self.graph.add_node(node)
        self._data_id_map[data_id] = node_id
        
        return node_id
    
    def track_transformation(self, data_id: str, operation: str,
                            inputs: List[str], outputs: List[str],
                            executor: str = "",
                            parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        追踪数据转换
        
        Args:
            data_id: 输出数据ID
            operation: 操作类型
            inputs: 输入数据ID列表
            outputs: 输出数据ID列表
            executor: 执行者
            parameters: 操作参数
            
        Returns:
            转换节点ID
        """
        node_id = str(uuid.uuid4())
        
        # 创建转换记录
        record = TransformationRecord(
            operation=operation,
            parameters=parameters or {},
            input_ids=inputs,
            output_id=data_id,
            executor=executor
        )
        self.transformations[data_id] = record
        
        # 创建转换节点
        node = LineageNode(
            id=node_id,
            type=NodeType.TRANSFORMATION,
            metadata={
                "data_id": data_id,
                "operation": operation,
                "executor": executor,
                "record": record.to_dict()
            }
        )
        self.graph.add_node(node)
        self._data_id_map[data_id] = node_id
        
        # 添加输入边
        for input_id in inputs:
            if input_id in self._data_id_map:
                edge = LineageEdge(
                    from_node=self._data_id_map[input_id],
                    to_node=node_id,
                    operation="input"
                )
                self.graph.add_edge(edge)
        
        # 为输出创建节点和边
        for output_id in outputs:
            if output_id not in self._data_id_map:
                output_node = LineageNode(
                    id=str(uuid.uuid4()),
                    type=NodeType.SINK,
                    metadata={"data_id": output_id}
                )
                self.graph.add_node(output_node)
                self._data_id_map[output_id] = output_node.id
            
            edge = LineageEdge(
                from_node=node_id,
                to_node=self._data_id_map[output_id],
                operation="output"
            )
            self.graph.add_edge(edge)
        
        return node_id
    
    def track_access(self, data_id: str, accessor: str, action: str,
                     timestamp: Optional[str] = None,
                     location: str = "",
                     purpose: str = "",
                     metadata: Optional[Dict[str, Any]] = None):
        """
        追踪数据访问
        
        Args:
            data_id: 数据ID
            accessor: 访问者
            action: 操作类型
            timestamp: 时间戳
            location: 访问位置
            purpose: 访问目的
            metadata: 元数据
        """
        record = AccessRecord(
            accessor=accessor,
            action=action,
            timestamp=timestamp or datetime.now().isoformat(),
            location=location,
            purpose=purpose,
            metadata=metadata or {}
        )
        self.access_records[data_id].append(record)
    
    def get_lineage(self, data_id: str) -> Dict[str, Any]:
        """
        获取完整血缘链
        
        Args:
            data_id: 数据ID
            
        Returns:
            血缘链信息
        """
        if data_id not in self._data_id_map:
            return {"error": f"Data ID {data_id} not found"}
        
        node_id = self._data_id_map[data_id]
        
        # 获取上游
        upstream = self.graph.get_upstream(node_id)
        
        # 获取下游
        downstream = self.graph.get_downstream(node_id)
        
        # 获取来源信息
        origin = self.origins.get(data_id)
        
        # 获取转换记录
        transformation = self.transformations.get(data_id)
        
        # 获取访问记录
        access = self.access_records.get(data_id, [])
        
        return {
            "data_id": data_id,
            "node_id": node_id,
            "origin": origin.to_dict() if origin else None,
            "transformation": transformation.to_dict() if transformation else None,
            "upstream": [n.to_dict() for n in upstream],
            "downstream": [n.to_dict() for n in downstream],
            "access_history": [a.to_dict() for a in access]
        }
    
    def get_impact(self, data_id: str) -> Dict[str, Any]:
        """
        获取影响分析（下游依赖）
        
        Args:
            data_id: 数据ID
            
        Returns:
            影响分析结果
        """
        if data_id not in self._data_id_map:
            return {"error": f"Data ID {data_id} not found"}
        
        node_id = self._data_id_map[data_id]
        
        # 获取所有下游节点
        downstream = self.graph.get_downstream(node_id, depth=-1)
        
        # 分类统计
        by_type = defaultdict(list)
        for node in downstream:
            by_type[node.type.value].append(node)
        
        return {
            "data_id": data_id,
            "total_downstream": len(downstream),
            "by_type": {k: [n.id for n in v] for k, v in by_type.items()},
            "downstream_nodes": [n.to_dict() for n in downstream],
            "leaf_nodes": [n.to_dict() for n in self.graph.get_leaves() 
                          if n.id in [dn.id for dn in downstream]]
        }
    
    def get_data_sources(self) -> List[Dict[str, Any]]:
        """获取所有数据源"""
        sources = []
        for node in self.graph.get_roots():
            if node.type == NodeType.SOURCE:
                data_id = node.metadata.get("data_id")
                origin = self.origins.get(data_id)
                sources.append({
                    "data_id": data_id,
                    "node": node.to_dict(),
                    "origin": origin.to_dict() if origin else None
                })
        return sources
    
    def get_transformation_history(self, data_id: str) -> List[Dict[str, Any]]:
        """获取数据的转换历史"""
        history = []
        current_id = data_id
        
        while current_id in self.transformations:
            record = self.transformations[current_id]
            history.append(record.to_dict())
            
            # 向上追溯
            if record.input_ids:
                current_id = record.input_ids[0]  # 取第一个输入
            else:
                break
        
        return history
    
    def export_lineage(self, file_path: Union[str, Path]):
        """导出血缘信息"""
        data = {
            "graph": self.graph.to_dict(),
            "origins": {k: v.to_dict() for k, v in self.origins.items()},
            "transformations": {k: v.to_dict() for k, v in self.transformations.items()},
            "access_records": {k: [a.to_dict() for a in v] 
                              for k, v in self.access_records.items()}
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_nodes": len(self.graph.nodes),
            "total_edges": len(self.graph.edges),
            "total_origins": len(self.origins),
            "total_transformations": len(self.transformations),
            "total_access_records": sum(len(v) for v in self.access_records.values()),
            "roots": len(self.graph.get_roots()),
            "leaves": len(self.graph.get_leaves()),
            "node_types": {
                node_type.value: sum(1 for n in self.graph.nodes.values() 
                                    if n.type == node_type)
                for node_type in NodeType
            }
        }


# 便捷函数
def create_lineage_tracker() -> DataLineageTracker:
    """创建血缘追踪器"""
    return DataLineageTracker()


# 示例用法
if __name__ == "__main__":
    tracker = DataLineageTracker()
    
    print("数据血缘追踪测试：")
    print("=" * 60)
    
    # 追踪数据源
    source_id = tracker.track_origin(
        data_id="customer_data_001",
        source="CRM系统",
        owner="数据团队",
        metadata={"format": "csv", "size": "10MB"}
    )
    print(f"\n1. 追踪数据源: customer_data_001")
    print(f"   节点ID: {source_id}")
    
    # 追踪数据转换
    transform_id = tracker.track_transformation(
        data_id="customer_data_cleaned",
        operation="数据清洗",
        inputs=["customer_data_001"],
        outputs=["customer_data_cleaned"],
        executor="ETL作业",
        parameters={"remove_nulls": True, "normalize": True}
    )
    print(f"\n2. 追踪转换: customer_data_cleaned")
    print(f"   节点ID: {transform_id}")
    
    # 追踪更多转换
    transform_id2 = tracker.track_transformation(
        data_id="customer_analytics",
        operation="数据分析",
        inputs=["customer_data_cleaned"],
        outputs=["customer_analytics", "customer_report"],
        executor="分析引擎",
        parameters={"metrics": ["avg_age", "total_spend"]}
    )
    print(f"\n3. 追踪转换: customer_analytics")
    print(f"   节点ID: {transform_id2}")
    
    # 追踪访问
    tracker.track_access(
        data_id="customer_analytics",
        accessor="分析师A",
        action="查询",
        location="BI系统",
        purpose="月度报告"
    )
    tracker.track_access(
        data_id="customer_analytics",
        accessor="分析师B",
        action="导出",
        location="数据仓库",
        purpose="机器学习训练"
    )
    print(f"\n4. 追踪访问记录: 2条")
    
    # 获取血缘链
    print("\n5. 获取血缘链 (customer_analytics):")
    lineage = tracker.get_lineage("customer_analytics")
    print(json.dumps(lineage, ensure_ascii=False, indent=2))
    
    # 获取影响分析
    print("\n6. 影响分析 (customer_data_001):")
    impact = tracker.get_impact("customer_data_001")
    print(f"   下游节点数: {impact['total_downstream']}")
    print(f"   按类型分布: {impact['by_type']}")
    
    # 统计信息
    print("\n7. 统计信息:")
    stats = tracker.get_statistics()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
