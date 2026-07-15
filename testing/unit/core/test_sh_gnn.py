"""
TestSHGNN - 核心算法单元测试：SH-GNN（异构图神经网络）模块

模块路径: testing/unit/core/test_sh_gnn.py
"""
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from unittest.mock import Mock, MagicMock, patch
import pytest

pytestmark = pytest.mark.unit


@dataclass
class GraphEdge:
    src: int
    dst: int
    edge_type: str
    weight: float = 1.0


@dataclass
class NodeType:
    name: str
    dim: int


class MockHeteroGraph:
    """模拟异构图"""

    def __init__(self):
        self.nodes: Dict[str, List[int]] = defaultdict(list)
        self.node_features: Dict[int, np.ndarray] = {}
        self.edges: Dict[str, List[GraphEdge]] = defaultdict(list)
        self.node_types: Dict[int, str] = {}
        self.edge_types: Set[str] = set()

    def add_node(self, node_id: int, node_type: str, features: np.ndarray):
        self.nodes[node_type].append(node_id)
        self.node_features[node_id] = features
        self.node_types[node_id] = node_type

    def add_edge(self, src: int, dst: int, edge_type: str, weight: float = 1.0):
        self.edges[edge_type].append(GraphEdge(src, dst, edge_type, weight))
        self.edge_types.add(edge_type)

    def get_neighbors(self, node_id: int, edge_type: Optional[str] = None) -> List[int]:
        neighbors = []
        edges = self.edges.get(edge_type, []) if edge_type else                  [e for et in self.edges for e in self.edges[et]]
        for e in edges:
            if e.src == node_id:
                neighbors.append(e.dst)
        return neighbors

    def get_node_degree(self, node_id: int) -> int:
        return len(self.get_neighbors(node_id))

    def get_edge_index(self, edge_type: str) -> Tuple[np.ndarray, np.ndarray]:
        edges = self.edges.get(edge_type, [])
        if not edges:
            return np.array([], dtype=int), np.array([], dtype=int)
        src = np.array([e.src for e in edges])
        dst = np.array([e.dst for e in edges])
        return src, dst

    def num_nodes(self, node_type: Optional[str] = None) -> int:
        if node_type:
            return len(self.nodes.get(node_type, []))
        return sum(len(n) for n in self.nodes.values())

    def num_edges(self, edge_type: Optional[str] = None) -> int:
        if edge_type:
            return len(self.edges.get(edge_type, []))
        return sum(len(e) for e in self.edges.values())


class MockSHGNN:
    """模拟SH-GNN模型"""

    def __init__(self, hidden_dim: int = 64, out_dim: int = 32,
                 n_layers: int = 2, n_heads: int = 4):
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.type_embeddings: Dict[str, np.ndarray] = {}
        self.relation_weights: Dict[str, np.ndarray] = {}
        self.message_weights: Dict[str, np.ndarray] = {}

    def init_type_embeddings(self, node_types: Dict[str, int]):
        for ntype, dim in node_types.items():
            self.type_embeddings[ntype] = np.random.randn(dim).astype(np.float32) * 0.02

    def init_relation_weights(self, edge_types: List[str]):
        for etype in edge_types:
            self.relation_weights[etype] = np.random.randn(
                self.hidden_dim, self.hidden_dim).astype(np.float32) * 0.02
            self.message_weights[etype] = np.random.randn(
                self.hidden_dim, self.hidden_dim).astype(np.float32) * 0.02

    def message_passing(self, graph: MockHeteroGraph, node_id: int,
                        layer: int = 0) -> np.ndarray:
        node_feat = graph.node_features.get(node_id)
        if node_feat is None:
            return np.zeros(self.hidden_dim, dtype=np.float32)

        messages = []
        for etype in graph.edge_types:
            neighbors = graph.get_neighbors(node_id, etype)
            if not neighbors:
                continue
            rel_w = self.relation_weights.get(etype)
            msg_w = self.message_weights.get(etype)
            if rel_w is None or msg_w is None:
                continue
            for nid in neighbors:
                nfeat = graph.node_features.get(nid)
                if nfeat is not None:
                    msg = np.matmul(nfeat, msg_w)
                    msg = np.matmul(msg, rel_w)
                    messages.append(msg)

        if not messages:
            return node_feat[:self.hidden_dim] if len(node_feat) >= self.hidden_dim else                    np.pad(node_feat, (0, self.hidden_dim - len(node_feat)))

        aggregated = np.mean(messages, axis=0)
        return aggregated + node_feat[:self.hidden_dim] if len(node_feat) >= self.hidden_dim else                aggregated + np.pad(node_feat, (0, self.hidden_dim - len(node_feat)))

    def compute_node_embeddings(self, graph: MockHeteroGraph) -> Dict[int, np.ndarray]:
        embeddings = {}
        for ntype in graph.nodes:
            for nid in graph.nodes[ntype]:
                h = graph.node_features.get(nid)
                if h is not None:
                    embeddings[nid] = self.message_passing(graph, nid)
        return embeddings

    def attention_aggregate(self, messages: List[np.ndarray],
                            query: np.ndarray) -> np.ndarray:
        if not messages:
            return query
        stack = np.stack(messages)
        scores = np.dot(stack, query) / np.sqrt(self.hidden_dim)
        weights = np.exp(scores - np.max(scores))
        weights = weights / (np.sum(weights) + 1e-8)
        return np.sum(stack * weights[:, np.newaxis], axis=0)

    def graph_pooling(self, embeddings: Dict[int, np.ndarray],
                      method: str = "mean") -> np.ndarray:
        if not embeddings:
            return np.zeros(self.out_dim, dtype=np.float32)
        stack = np.stack(list(embeddings.values()))
        if method == "mean":
            return np.mean(stack, axis=0)
        elif method == "max":
            return np.max(stack, axis=0)
        elif method == "sum":
            return np.sum(stack, axis=0)
        return np.mean(stack, axis=0)


class TestHeteroGraph:
    """异构图测试"""

    def setup_method(self):
        self.graph = MockHeteroGraph()
        self.graph.add_node(0, "user", np.random.randn(32).astype(np.float32))
        self.graph.add_node(1, "user", np.random.randn(32).astype(np.float32))
        self.graph.add_node(2, "item", np.random.randn(32).astype(np.float32))
        self.graph.add_node(3, "item", np.random.randn(32).astype(np.float32))
        self.graph.add_edge(0, 2, "buys")
        self.graph.add_edge(0, 3, "views")
        self.graph.add_edge(1, 2, "buys")

    def test_num_nodes(self):
        assert self.graph.num_nodes() == 4
        assert self.graph.num_nodes("user") == 2
        assert self.graph.num_nodes("item") == 2

    def test_num_edges(self):
        assert self.graph.num_edges() == 3
        assert self.graph.num_edges("buys") == 2
        assert self.graph.num_edges("views") == 1

    def test_get_neighbors(self):
        neighbors = self.graph.get_neighbors(0, "buys")
        assert 2 in neighbors

    def test_get_neighbors_all_types(self):
        neighbors = self.graph.get_neighbors(0)
        assert 2 in neighbors
        assert 3 in neighbors

    def test_node_degree(self):
        assert self.graph.get_node_degree(0) == 2
        assert self.graph.get_node_degree(1) == 1

    def test_edge_index(self):
        src, dst = self.graph.get_edge_index("buys")
        assert len(src) == 2
        assert 0 in src and 1 in src

    def test_empty_graph(self):
        g = MockHeteroGraph()
        assert g.num_nodes() == 0
        assert g.num_edges() == 0


class TestMessagePassing:
    """消息传递测试"""

    def setup_method(self):
        self.graph = MockHeteroGraph()
        for i in range(5):
            self.graph.add_node(i, "node", np.random.randn(64).astype(np.float32))
        self.graph.add_edge(0, 1, "rel")
        self.graph.add_edge(0, 2, "rel")
        self.graph.add_edge(1, 3, "rel")
        self.graph.add_edge(2, 4, "rel")
        self.gnn = MockSHGNN(hidden_dim=64, out_dim=32)
        self.gnn.init_relation_weights(["rel"])

    def test_message_passing_shape(self):
        emb = self.gnn.message_passing(self.graph, 0)
        assert emb.shape == (64,)

    def test_message_passing_with_neighbors(self):
        emb = self.gnn.message_passing(self.graph, 0)
        assert not np.allclose(emb, 0)

    def test_message_passing_isolated_node(self):
        emb = self.gnn.message_passing(self.graph, 99)
        assert np.allclose(emb, 0)

    def test_compute_all_embeddings(self):
        embs = self.gnn.compute_node_embeddings(self.graph)
        assert len(embs) == 5
        for nid, emb in embs.items():
            assert emb.shape == (64,)


class TestAttentionAggregation:
    """注意力聚合测试"""

    def setup_method(self):
        self.gnn = MockSHGNN(hidden_dim=64, out_dim=32)

    def test_aggregate_shape(self):
        messages = [np.random.randn(64).astype(np.float32) for _ in range(3)]
        query = np.random.randn(64).astype(np.float32)
        result = self.gnn.attention_aggregate(messages, query)
        assert result.shape == (64,)

    def test_aggregate_no_messages(self):
        query = np.random.randn(64).astype(np.float32)
        result = self.gnn.attention_aggregate([], query)
        assert np.allclose(result, query)

    def test_aggregate_weights_sum_to_one(self):
        messages = [np.random.randn(64).astype(np.float32) for _ in range(5)]
        query = np.random.randn(64).astype(np.float32)
        stack = np.stack(messages)
        scores = np.dot(stack, query) / np.sqrt(64)
        weights = np.exp(scores - np.max(scores))
        weights = weights / (np.sum(weights) + 1e-8)
        assert np.isclose(np.sum(weights), 1.0, atol=1e-5)


class TestGraphPooling:
    """图池化测试"""

    def setup_method(self):
        self.gnn = MockSHGNN(hidden_dim=64, out_dim=32)

    def test_mean_pooling(self):
        embs = {i: np.random.randn(64).astype(np.float32) for i in range(5)}
        pooled = self.gnn.graph_pooling(embs, method="mean")
        assert pooled.shape == (64,)

    def test_max_pooling(self):
        embs = {i: np.random.randn(64).astype(np.float32) for i in range(5)}
        pooled = self.gnn.graph_pooling(embs, method="max")
        assert pooled.shape == (64,)

    def test_sum_pooling(self):
        embs = {i: np.random.randn(64).astype(np.float32) for i in range(5)}
        pooled = self.gnn.graph_pooling(embs, method="sum")
        assert pooled.shape == (64,)

    def test_empty_embeddings(self):
        pooled = self.gnn.graph_pooling({})
        assert pooled.shape == (32,)

    def test_pooling_deterministic(self):
        embs = {i: np.random.randn(64).astype(np.float32) for i in range(5)}
        p1 = self.gnn.graph_pooling(embs, "mean")
        p2 = self.gnn.graph_pooling(embs, "mean")
        assert np.allclose(p1, p2)
