"""
AGI统一框架 - 知识图谱推理模块
实现知识图谱构建、嵌入学习、推理查询等核心功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Set
from dataclasses import dataclass, field
import math
from collections import defaultdict, deque
import random
from abc import ABC, abstractmethod
import json


# ==================== 配置类 ====================

@dataclass
class KGConfig:
    """知识图谱配置"""
    # 实体和关系配置
    num_entities: int = 10000
    num_relations: int = 100
    embedding_dim: int = 200
    
    # KGE模型配置
    model_type: str = "transE"  # transE, transR, distMult, complEx, rotatE
    
    # 训练配置
    learning_rate: float = 0.01
    batch_size: int = 1024
    num_epochs: int = 100
    negative_samples: int = 5
    
    # 推理配置
    max_path_length: int = 5
    reasoning_steps: int = 3
    
    # 正则化
    entity_norm: float = 1.0
    relation_norm: float = 1.0


# ==================== 知识图谱数据结构 ====================

class Triple:
    """三元组"""
    
    def __init__(self, head: int, relation: int, tail: int):
        self.head = head
        self.relation = relation
        self.tail = tail
        
    def __eq__(self, other):
        return (self.head == other.head and 
                self.relation == other.relation and 
                self.tail == other.tail)
    
    def __hash__(self):
        return hash((self.head, self.relation, self.tail))
    
    def __repr__(self):
        return f"({self.head}, {self.relation}, {self.tail})"
    
    def to_tuple(self) -> Tuple[int, int, int]:
        return (self.head, self.relation, self.tail)


class KnowledgeGraph:
    """知识图谱数据结构"""
    
    def __init__(self):
        # 实体和关系映射
        self.entity_to_id: Dict[str, int] = {}
        self.id_to_entity: Dict[int, str] = {}
        self.relation_to_id: Dict[str, int] = {}
        self.id_to_relation: Dict[int, str] = {}
        
        # 三元组存储
        self.triples: Set[Triple] = set()
        
        # 索引（加速查询）
        self.head_index: Dict[int, Dict[int, Set[int]]] = defaultdict(lambda: defaultdict(set))
        self.tail_index: Dict[int, Dict[int, Set[int]]] = defaultdict(lambda: defaultdict(set))
        
        # 统计
        self._next_entity_id = 0
        self._next_relation_id = 0
        
    def add_entity(self, entity: str) -> int:
        """添加实体"""
        if entity not in self.entity_to_id:
            entity_id = self._next_entity_id
            self.entity_to_id[entity] = entity_id
            self.id_to_entity[entity_id] = entity
            self._next_entity_id += 1
        return self.entity_to_id[entity]
    
    def add_relation(self, relation: str) -> int:
        """添加关系"""
        if relation not in self.relation_to_id:
            relation_id = self._next_relation_id
            self.relation_to_id[relation] = relation_id
            self.id_to_relation[relation_id] = relation
            self._next_relation_id += 1
        return self.relation_to_id[relation]
    
    def add_triple(self, head: str, relation: str, tail: str) -> Triple:
        """添加三元组"""
        head_id = self.add_entity(head)
        relation_id = self.add_relation(relation)
        tail_id = self.add_entity(tail)
        
        triple = Triple(head_id, relation_id, tail_id)
        
        if triple not in self.triples:
            self.triples.add(triple)
            # 更新索引
            self.head_index[head_id][relation_id].add(tail_id)
            self.tail_index[tail_id][relation_id].add(head_id)
        
        return triple
    
    def has_triple(self, head: int, relation: int, tail: int) -> bool:
        """检查三元组是否存在"""
        return Triple(head, relation, tail) in self.triples
    
    def get_tails(self, head: int, relation: int) -> Set[int]:
        """获取给定头实体和关系的所有尾实体"""
        return self.head_index[head][relation]
    
    def get_heads(self, tail: int, relation: int) -> Set[int]:
        """获取给定尾实体和关系的所有头实体"""
        return self.tail_index[tail][relation]
    
    def get_relations(self, head: int, tail: int) -> Set[int]:
        """获取两个实体之间的所有关系"""
        relations = set()
        for relation_id, tails in self.head_index[head].items():
            if tail in tails:
                relations.add(relation_id)
        return relations
    
    def get_neighbors(self, entity: int) -> Dict[int, Set[int]]:
        """获取实体的所有邻居"""
        neighbors = defaultdict(set)
        
        # 出边
        for relation_id, tails in self.head_index[entity].items():
            neighbors[relation_id].update(tails)
        
        # 入边
        for relation_id, heads in self.tail_index[entity].items():
            neighbors[-relation_id - 1].update(heads)  # 负数表示反向关系
        
        return dict(neighbors)
    
    def num_entities(self) -> int:
        return len(self.entity_to_id)
    
    def num_relations(self) -> int:
        return len(self.relation_to_id)
    
    def num_triples(self) -> int:
        return len(self.triples)
    
    def to_triples_array(self) -> np.ndarray:
        """转换为numpy数组"""
        return np.array([t.to_tuple() for t in self.triples])


# ==================== 知识图谱嵌入模型 ====================

class KGEModel(nn.Module):
    """知识图谱嵌入模型基类"""
    
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        
        # 实体嵌入
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        # 关系嵌入
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # 初始化
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        
    def normalize_embeddings(self):
        """归一化嵌入"""
        with torch.no_grad():
            self.entity_embeddings.weight.data = F.normalize(
                self.entity_embeddings.weight.data, p=2, dim=1
            )


class TransE(KGEModel):
    """TransE模型: h + r ≈ t"""
    
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int = 200):
        super().__init__(num_entities, num_relations, embedding_dim)
        self.norm = 1  # L1范数
        
    def forward(self, heads: torch.Tensor, relations: torch.Tensor, 
                tails: torch.Tensor) -> torch.Tensor:
        """计算三元组得分"""
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)
        
        # 得分 = ||h + r - t||
        score = torch.norm(h + r - t, p=self.norm, dim=1)
        return score
    
    def predict_tails(self, head: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        """预测尾实体"""
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        
        # t ≈ h + r
        target = h + r
        
        # 计算与所有实体的距离
        distances = torch.norm(
            target.unsqueeze(1) - self.entity_embeddings.weight.unsqueeze(0),
            p=self.norm, dim=2
        )
        return -distances  # 返回负距离（距离越小得分越高）


class TransR(KGEModel):
    """TransR模型: 在关系空间中进行平移"""
    
    def __init__(self, num_entities: int, num_relations: int, 
                 embedding_dim: int = 200, relation_dim: int = 200):
        super().__init__(num_entities, num_relations, embedding_dim)
        self.relation_dim = relation_dim
        
        # 投影矩阵
        self.projection_matrices = nn.Embedding(num_relations, embedding_dim * relation_dim)
        nn.init.xavier_uniform_(self.projection_matrices.weight)
        
    def forward(self, heads: torch.Tensor, relations: torch.Tensor,
                tails: torch.Tensor) -> torch.Tensor:
        """计算三元组得分"""
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)
        
        # 获取投影矩阵
        proj = self.projection_matrices(relations)
        proj = proj.view(-1, self.embedding_dim, self.relation_dim)
        
        # 投影
        h_proj = torch.bmm(h.unsqueeze(1), proj).squeeze(1)
        t_proj = torch.bmm(t.unsqueeze(1), proj).squeeze(1)
        
        # 得分
        score = torch.norm(h_proj + r - t_proj, p=1, dim=1)
        return score


class DistMult(KGEModel):
    """DistMult模型: 双线性形式"""
    
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int = 200):
        super().__init__(num_entities, num_relations, embedding_dim)
        
    def forward(self, heads: torch.Tensor, relations: torch.Tensor,
                tails: torch.Tensor) -> torch.Tensor:
        """计算三元组得分"""
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)
        
        # 得分 = <h, r, t> = h^T * diag(r) * t
        score = torch.sum(h * r * t, dim=1)
        return score


class ComplEx(KGEModel):
    """ComplEx模型: 复数嵌入"""
    
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int = 200):
        super().__init__(num_entities, num_relations, embedding_dim)
        
    def forward(self, heads: torch.Tensor, relations: torch.Tensor,
                tails: torch.Tensor) -> torch.Tensor:
        """计算三元组得分（复数形式）"""
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)
        
        # 分离实部和虚部
        h_re, h_im = h.chunk(2, dim=1)
        r_re, r_im = r.chunk(2, dim=1)
        t_re, t_im = t.chunk(2, dim=1)
        
        # 复数乘法: (h_re + i*h_im) * (r_re + i*r_im) * (t_re - i*t_im)
        score = (h_re * r_re * t_re + 
                h_im * r_im * t_re + 
                h_re * r_im * t_im - 
                h_im * r_re * t_im)
        
        return score.sum(dim=1)


class RotatE(KGEModel):
    """RotatE模型: 旋转嵌入"""
    
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int = 200):
        super().__init__(num_entities, num_relations, embedding_dim)
        # 关系嵌入作为相位
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
    def forward(self, heads: torch.Tensor, relations: torch.Tensor,
                tails: torch.Tensor) -> torch.Tensor:
        """计算三元组得分"""
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)
        
        # 将关系转换为旋转角度
        r_phase = torch.sigmoid(r) * 2 * math.pi
        
        # 旋转: h * e^(i*r) ≈ t
        h_re, h_im = h.chunk(2, dim=1)
        cos_r = torch.cos(r_phase[:, :h_re.size(1)])
        sin_r = torch.sin(r_phase[:, :h_re.size(1)])
        
        rotated_re = h_re * cos_r - h_im * sin_r
        rotated_im = h_re * sin_r + h_im * cos_r
        
        t_re, t_im = t.chunk(2, dim=1)
        
        # 距离
        score = torch.sqrt((rotated_re - t_re)**2 + (rotated_im - t_im)**2)
        return score.sum(dim=1)


# ==================== 知识图谱推理 ====================

class KGReasoning:
    """知识图谱推理"""
    
    def __init__(self, kg: KnowledgeGraph, model: KGEModel, 
                 config: Optional[KGConfig] = None, device: str = 'cpu'):
        self.kg = kg
        self.model = model.to(device)
        self.config = config or KGConfig()
        self.device = device
        
    def link_prediction(self, head: int, relation: int, 
                        top_k: int = 10) -> List[Tuple[int, float]]:
        """链接预测: (h, r, ?)"""
        with torch.no_grad():
            head_tensor = torch.tensor([head], device=self.device)
            relation_tensor = torch.tensor([relation], device=self.device)
            
            scores = self.model.predict_tails(head_tensor, relation_tensor)
            scores = scores.squeeze()
            
            # 获取top-k
            values, indices = torch.topk(scores, top_k)
            
            results = []
            for idx, score in zip(indices.cpu().numpy(), values.cpu().numpy()):
                results.append((int(idx), float(score)))
            
            return results
    
    def relation_prediction(self, head: int, tail: int,
                            top_k: int = 5) -> List[Tuple[int, float]]:
        """关系预测: (h, ?, t)"""
        with torch.no_grad():
            h = self.model.entity_embeddings(
                torch.tensor([head], device=self.device)
            )
            t = self.model.entity_embeddings(
                torch.tensor([tail], device=self.device)
            )
            
            # 计算与所有关系的得分
            r_embeddings = self.model.relation_embeddings.weight
            
            if isinstance(self.model, TransE):
                # TransE: r ≈ t - h
                target = t - h
                distances = torch.norm(
                    target - r_embeddings, p=1, dim=1
                )
                scores = -distances
            elif isinstance(self.model, DistMult):
                # DistMult: <h, r, t> = h * r * t
                scores = torch.sum(h * r_embeddings * t, dim=1)
            else:
                scores = torch.zeros(self.kg.num_relations())
            
            values, indices = torch.topk(scores, top_k)
            
            results = []
            for idx, score in zip(indices.cpu().numpy(), values.cpu().numpy()):
                results.append((int(idx), float(score)))
            
            return results
    
    def path_reasoning(self, start: int, end: int, 
                       max_length: int = 5) -> List[List[Tuple[int, int]]]:
        """路径推理: 寻找从start到end的路径"""
        paths = []
        queue = deque([(start, [])])
        visited = set()
        
        while queue:
            current, path = queue.popleft()
            
            if len(path) > max_length:
                continue
            
            if current == end and path:
                paths.append(path)
                continue
            
            if current in visited:
                continue
            visited.add(current)
            
            # 扩展路径
            neighbors = self.kg.get_neighbors(current)
            for relation, entities in neighbors.items():
                for entity in entities:
                    if relation >= 0:  # 正向关系
                        new_path = path + [(current, relation, entity)]
                    else:  # 反向关系
                        new_path = path + [(entity, -relation - 1, current)]
                    queue.append((entity, new_path))
        
        return paths
    
    def multi_hop_reasoning(self, query: Tuple[int, int, int],
                            num_hops: int = 2) -> float:
        """多跳推理"""
        head, relation, tail = query
        
        # 使用路径推理
        paths = self.path_reasoning(head, tail, max_length=num_hops)
        
        if not paths:
            return 0.0
        
        # 计算路径得分
        path_scores = []
        
        for path in paths:
            score = 1.0
            for h, r, t in path:
                # 使用模型计算边得分
                with torch.no_grad():
                    h_tensor = torch.tensor([h], device=self.device)
                    r_tensor = torch.tensor([r], device=self.device)
                    t_tensor = torch.tensor([t], device=self.device)
                    
                    edge_score = self.model(h_tensor, r_tensor, t_tensor)
                    score *= torch.sigmoid(-edge_score).item()
            
            path_scores.append(score)
        
        # 返回最大路径得分
        return max(path_scores) if path_scores else 0.0
    
    def rule_reasoning(self, head: int, rule: List[Tuple[int, str]]) -> Set[int]:
        """规则推理"""
        """
        rule: [(relation_id, direction), ...]
        direction: 'forward' or 'backward'
        """
        current_entities = {head}
        
        for relation, direction in rule:
            next_entities = set()
            
            for entity in current_entities:
                if direction == 'forward':
                    tails = self.kg.get_tails(entity, relation)
                    next_entities.update(tails)
                else:
                    heads = self.kg.get_heads(entity, relation)
                    next_entities.update(heads)
            
            current_entities = next_entities
            
            if not current_entities:
                break
        
        return current_entities


# ==================== 知识图谱补全 ====================

class KGCompletion:
    """知识图谱补全"""
    
    def __init__(self, kg: KnowledgeGraph, model: KGEModel,
                 config: Optional[KGConfig] = None, device: str = 'cpu'):
        self.kg = kg
        self.model = model.to(device)
        self.config = config or KGConfig()
        self.device = device
        
    def train(self, num_epochs: int = 100, 
              learning_rate: float = 0.01,
              batch_size: int = 1024,
              negative_samples: int = 5,
              callback: Optional[Callable] = None) -> Dict[str, List]:
        """训练模型"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        triples = self.kg.to_triples_array()
        num_triples = len(triples)
        
        history = {'loss': [], 'mr': [], 'hits@10': []}
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            # 打乱数据
            np.random.shuffle(triples)
            
            for i in range(0, num_triples, batch_size):
                batch = triples[i:i+batch_size]
                
                # 正样本
                heads = torch.tensor(batch[:, 0], device=self.device)
                relations = torch.tensor(batch[:, 1], device=self.device)
                tails = torch.tensor(batch[:, 2], device=self.device)
                
                # 负样本
                neg_heads, neg_tails = self._generate_negatives(
                    batch, negative_samples
                )
                
                # 计算损失
                pos_scores = self.model(heads, relations, tails)
                neg_scores = self.model(neg_heads, relations, neg_tails)
                
                # Margin ranking loss
                loss = F.margin_ranking_loss(
                    neg_scores, pos_scores,
                    torch.ones_like(pos_scores),
                    margin=1.0
                )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 归一化
                if hasattr(self.model, 'normalize_embeddings'):
                    self.model.normalize_embeddings()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / max(num_batches, 1)
            history['loss'].append(avg_loss)
            
            if callback:
                callback(epoch, {'loss': avg_loss})
        
        return history
    
    def _generate_negatives(self, batch: np.ndarray, 
                            num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成负样本"""
        batch_size = len(batch)
        
        # 随机替换头或尾
        neg_heads = batch[:, 0].copy()
        neg_tails = batch[:, 2].copy()
        
        for i in range(batch_size):
            if random.random() < 0.5:
                # 替换头
                neg_heads[i] = random.randint(0, self.kg.num_entities() - 1)
            else:
                # 替换尾
                neg_tails[i] = random.randint(0, self.kg.num_entities() - 1)
        
        return (torch.tensor(neg_heads, device=self.device),
                torch.tensor(neg_tails, device=self.device))
    
    def evaluate(self, test_triples: np.ndarray,
                 filter_triples: Optional[np.ndarray] = None) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        
        ranks = []
        hits_10 = 0
        hits_1 = 0
        
        with torch.no_grad():
            for triple in test_triples:
                h, r, t = triple
                
                # 预测尾实体
                head_tensor = torch.tensor([h], device=self.device)
                relation_tensor = torch.tensor([r], device=self.device)
                
                scores = self.model.predict_tails(head_tensor, relation_tensor).squeeze()
                
                # 排序
                sorted_indices = torch.argsort(scores, descending=True)
                rank = (sorted_indices == t).nonzero(as_tuple=True)[0][0].item() + 1
                
                ranks.append(rank)
                
                if rank <= 10:
                    hits_10 += 1
                if rank == 1:
                    hits_1 += 1
        
        n = len(test_triples)
        return {
            'MR': np.mean(ranks),
            'MRR': np.mean([1.0 / r for r in ranks]),
            'Hits@1': hits_1 / n,
            'Hits@10': hits_10 / n
        }


# ==================== 知识图谱构建 ====================

class KGBuilder:
    """知识图谱构建器"""
    
    def __init__(self):
        self.kg = KnowledgeGraph()
        
    def from_triples(self, triples: List[Tuple[str, str, str]]) -> KnowledgeGraph:
        """从三元组列表构建"""
        for head, relation, tail in triples:
            self.kg.add_triple(head, relation, tail)
        return self.kg
    
    def from_json(self, filepath: str) -> KnowledgeGraph:
        """从JSON文件构建"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for triple in data:
            self.kg.add_triple(
                triple['head'],
                triple['relation'],
                triple['tail']
            )
        
        return self.kg
    
    def from_csv(self, filepath: str, 
                 head_col: int = 0, relation_col: int = 1, tail_col: int = 2,
                 delimiter: str = '\t') -> KnowledgeGraph:
        """从CSV文件构建"""
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split(delimiter)
                if len(parts) >= 3:
                    self.kg.add_triple(
                        parts[head_col],
                        parts[relation_col],
                        parts[tail_col]
                    )
        
        return self.kg
    
    def add_inverse_relations(self) -> KnowledgeGraph:
        """添加逆关系"""
        triples_to_add = []
        
        for triple in self.kg.triples:
            head_name = self.kg.id_to_entity[triple.head]
            tail_name = self.kg.id_to_entity[triple.tail]
            relation_name = self.kg.id_to_relation[triple.relation]
            
            # 添加逆关系
            inverse_relation = f"{relation_name}_inv"
            triples_to_add.append((tail_name, inverse_relation, head_name))
        
        for head, relation, tail in triples_to_add:
            self.kg.add_triple(head, relation, tail)
        
        return self.kg


# ==================== 图神经网络推理 ====================

class RGCNLayer(nn.Module):
    """关系图卷积层"""
    
    def __init__(self, in_dim: int, out_dim: int, num_relations: int,
                 num_bases: Optional[int] = None):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_bases = num_bases
        
        if num_bases is not None:
            # 基分解
            self.bases = nn.Parameter(torch.randn(num_bases, in_dim, out_dim))
            self.relation_weights = nn.Parameter(torch.randn(num_relations, num_bases))
        else:
            # 直接参数化
            self.weights = nn.Parameter(torch.randn(num_relations, in_dim, out_dim))
        
        self.self_loop = nn.Linear(in_dim, out_dim)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        
    def forward(self, x: torch.Tensor, 
                adj_tensors: List[torch.Tensor]) -> torch.Tensor:
        """前向传播"""
        # x: (num_entities, in_dim)
        # adj_tensors: 每个关系的邻接矩阵
        
        out = self.self_loop(x)
        
        for r, adj in enumerate(adj_tensors):
            if self.num_bases is not None:
                # 基分解
                weight = torch.einsum('rb,bio->rio', 
                                     self.relation_weights, self.bases)
                weight = weight[r]
            else:
                weight = self.weights[r]
            
            # 消息传递
            message = torch.sparse.mm(adj, x)
            out = out + torch.matmul(message, weight)
        
        return F.relu(out + self.bias)


class RGCN(nn.Module):
    """关系图卷积网络"""
    
    def __init__(self, num_entities: int, num_relations: int,
                 embedding_dim: int = 200, hidden_dims: List[int] = [200, 200],
                 num_bases: Optional[int] = 30):
        super().__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        
        # 嵌入层
        self.embedding = nn.Embedding(num_entities, embedding_dim)
        
        # RGCN层
        self.layers = nn.ModuleList()
        dims = [embedding_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            self.layers.append(RGCNLayer(dims[i], dims[i+1], num_relations, num_bases))
        
        # 输出层
        self.output_dim = hidden_dims[-1]
        
    def forward(self, adj_tensors: List[torch.Tensor]) -> torch.Tensor:
        """前向传播"""
        x = self.embedding.weight
        
        for layer in self.layers:
            x = layer(x, adj_tensors)
        
        return x
    
    def score_triple(self, x: torch.Tensor, head: torch.Tensor,
                     relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        """计算三元组得分"""
        h = x[head]
        t = x[tail]
        
        # 简单的距离得分
        score = -torch.norm(h - t, p=2, dim=1)
        return score


# ==================== 工具函数 ====================

def create_kg_model(model_type: str, num_entities: int, num_relations: int,
                    embedding_dim: int = 200) -> KGEModel:
    """创建知识图谱嵌入模型"""
    if model_type.lower() == "transe":
        return TransE(num_entities, num_relations, embedding_dim)
    elif model_type.lower() == "transr":
        return TransR(num_entities, num_relations, embedding_dim)
    elif model_type.lower() == "distmult":
        return DistMult(num_entities, num_relations, embedding_dim)
    elif model_type.lower() == "complex":
        return ComplEx(num_entities, num_relations, embedding_dim)
    elif model_type.lower() == "rotate":
        return RotatE(num_entities, num_relations, embedding_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def visualize_kg_subgraph(kg: KnowledgeGraph, center_entity: int,
                          depth: int = 2) -> str:
    """可视化知识图谱子图"""
    lines = [f"Subgraph around {kg.id_to_entity[center_entity]}:"]
    
    visited = set()
    queue = deque([(center_entity, 0)])
    
    while queue:
        entity, level = queue.popleft()
        
        if entity in visited or level > depth:
            continue
        visited.add(entity)
        
        entity_name = kg.id_to_entity[entity]
        indent = "  " * level
        
        # 出边
        for relation_id, tails in kg.head_index[entity].items():
            relation_name = kg.id_to_relation[relation_id]
            for tail in tails:
                tail_name = kg.id_to_entity[tail]
                lines.append(f"{indent}{entity_name} --[{relation_name}]--> {tail_name}")
                queue.append((tail, level + 1))
        
        # 入边
        for relation_id, heads in kg.tail_index[entity].items():
            relation_name = kg.id_to_relation[relation_id]
            for head in heads:
                head_name = kg.id_to_entity[head]
                lines.append(f"{indent}{head_name} --[{relation_name}]--> {entity_name}")
                queue.append((head, level + 1))
    
    return "\n".join(lines)


def compute_kg_statistics(kg: KnowledgeGraph) -> Dict[str, Any]:
    """计算知识图谱统计信息"""
    # 度分布
    in_degrees = defaultdict(int)
    out_degrees = defaultdict(int)
    
    for triple in kg.triples:
        out_degrees[triple.head] += 1
        in_degrees[triple.tail] += 1
    
    return {
        'num_entities': kg.num_entities(),
        'num_relations': kg.num_relations(),
        'num_triples': kg.num_triples(),
        'avg_in_degree': np.mean(list(in_degrees.values())) if in_degrees else 0,
        'avg_out_degree': np.mean(list(out_degrees.values())) if out_degrees else 0,
        'max_in_degree': max(in_degrees.values()) if in_degrees else 0,
        'max_out_degree': max(out_degrees.values()) if out_degrees else 0,
        'density': kg.num_triples() / (kg.num_entities() ** 2 * kg.num_relations()) if kg.num_entities() > 0 else 0
    }
