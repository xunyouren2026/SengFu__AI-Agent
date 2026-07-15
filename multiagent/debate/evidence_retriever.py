"""
证据检索模块
为论点提供知识库支撑
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from enum import Enum
import re
from collections import defaultdict

from .protocol import Evidence, Argument


class EvidenceSourceType(Enum):
    """证据来源类型"""
    KNOWLEDGE_BASE = "knowledge_base"      # 知识库
    EXTERNAL_API = "external_api"          # 外部API
    DATABASE = "database"                  # 数据库
    DOCUMENT = "document"                  # 文档
    STATISTICAL = "statistical"            # 统计数据
    EXPERT_OPINION = "expert_opinion"      # 专家意见


class EvidenceReliability(Enum):
    """证据可靠性等级"""
    HIGH = 0.9
    MEDIUM = 0.6
    LOW = 0.3
    UNVERIFIED = 0.1


@dataclass
class KnowledgeEntry:
    """知识库条目"""
    entry_id: str
    content: str
    source: str
    source_type: EvidenceSourceType
    reliability: EvidenceReliability
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    
    def to_evidence(self) -> Evidence:
        """转换为证据对象"""
        return Evidence(
            evidence_id=self.entry_id,
            source=self.source,
            content=self.content,
            credibility=self.reliability.value,
            relevance=0.5,  # 默认相关度
            metadata={
                "source_type": self.source_type.value,
                "tags": list(self.tags),
                **self.metadata
            }
        )


@dataclass
class RetrievalQuery:
    """检索查询"""
    query_text: str
    topic: str = ""
    required_tags: Set[str] = field(default_factory=set)
    exclude_tags: Set[str] = field(default_factory=set)
    min_reliability: EvidenceReliability = EvidenceReliability.LOW
    max_results: int = 10
    source_types: Optional[Set[EvidenceSourceType]] = None


@dataclass
class RetrievalResult:
    """检索结果"""
    query: RetrievalQuery
    entries: List[KnowledgeEntry]
    scores: List[float]
    total_found: int
    retrieval_time_ms: float = 0.0
    
    def get_top_k(self, k: int = 5) -> List[Tuple[KnowledgeEntry, float]]:
        """获取前k个结果"""
        combined = list(zip(self.entries, self.scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:k]
    
    def to_evidence_list(self) -> List[Evidence]:
        """转换为证据列表"""
        return [entry.to_evidence() for entry in self.entries]


class KnowledgeBase:
    """
    知识库
    存储和管理知识条目
    """
    
    def __init__(self, name: str) -> None:
        self.name = name
        self.entries: Dict[str, KnowledgeEntry] = {}
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.source_index: Dict[str, Set[str]] = defaultdict(set)
    
    def add_entry(self, entry: KnowledgeEntry) -> None:
        """添加知识条目"""
        self.entries[entry.entry_id] = entry
        
        # 更新索引
        for tag in entry.tags:
            self.tag_index[tag].add(entry.entry_id)
        
        self.source_index[entry.source_type.value].add(entry.entry_id)
    
    def remove_entry(self, entry_id: str) -> bool:
        """移除知识条目"""
        if entry_id not in self.entries:
            return False
        
        entry = self.entries[entry_id]
        
        # 更新索引
        for tag in entry.tags:
            self.tag_index[tag].discard(entry_id)
        
        self.source_index[entry.source_type.value].discard(entry_id)
        del self.entries[entry_id]
        
        return True
    
    def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """获取知识条目"""
        return self.entries.get(entry_id)
    
    def search_by_tags(self, tags: Set[str]) -> List[KnowledgeEntry]:
        """按标签搜索"""
        if not tags:
            return list(self.entries.values())
        
        # 找到包含所有标签的条目
        result_ids = None
        for tag in tags:
            if result_ids is None:
                result_ids = self.tag_index[tag].copy()
            else:
                result_ids &= self.tag_index[tag]
        
        if result_ids is None:
            return []
        
        return [self.entries[eid] for eid in result_ids]
    
    def get_by_source_type(self, source_type: EvidenceSourceType) -> List[KnowledgeEntry]:
        """按来源类型获取"""
        entry_ids = self.source_index.get(source_type.value, set())
        return [self.entries[eid] for eid in entry_ids]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "name": self.name,
            "total_entries": len(self.entries),
            "tag_count": len(self.tag_index),
            "source_type_distribution": {
                st: len(ids) for st, ids in self.source_index.items()
            },
        }


class RelevanceScorer:
    """
    相关性评分器
    计算查询与知识条目的相关性
    """
    
    def __init__(self) -> None:
        self.term_weights: Dict[str, float] = defaultdict(lambda: 1.0)
    
    def calculate_score(
        self,
        query: RetrievalQuery,
        entry: KnowledgeEntry
    ) -> float:
        """
        计算相关性分数
        
        Args:
            query: 检索查询
            entry: 知识条目
            
        Returns:
            相关性分数 0-1
        """
        scores = []
        
        # 文本相似度
        text_score = self._text_similarity(query.query_text, entry.content)
        scores.append(text_score * 0.4)
        
        # 标签匹配
        tag_score = self._tag_match_score(query.required_tags, entry.tags)
        scores.append(tag_score * 0.2)
        
        # 可靠性分数
        reliability_score = entry.reliability.value
        scores.append(reliability_score * 0.2)
        
        # 主题相关性
        if query.topic:
            topic_score = self._topic_relevance(query.topic, entry)
            scores.append(topic_score * 0.2)
        
        return sum(scores)
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        # 简单的词重叠计算
        words1 = set(self._tokenize(text1.lower()))
        words2 = set(self._tokenize(text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        # 简化分词
        return re.findall(r'\b\w+\b', text)
    
    def _tag_match_score(self, required_tags: Set[str], entry_tags: Set[str]) -> float:
        """计算标签匹配分数"""
        if not required_tags:
            return 0.5
        
        matches = len(required_tags & entry_tags)
        return matches / len(required_tags)
    
    def _topic_relevance(self, topic: str, entry: KnowledgeEntry) -> float:
        """计算主题相关性"""
        # 检查主题词是否出现在内容或标签中
        topic_lower = topic.lower()
        content_lower = entry.content.lower()
        
        if topic_lower in content_lower:
            return 0.8
        
        if any(topic_lower in tag.lower() for tag in entry.tags):
            return 0.9
        
        # 部分匹配
        topic_words = set(self._tokenize(topic_lower))
        content_words = set(self._tokenize(content_lower))
        
        if topic_words and content_words:
            overlap = len(topic_words & content_words)
            return 0.3 + 0.5 * (overlap / len(topic_words))
        
        return 0.1


class EvidenceRetriever:
    """
    证据检索器
    为论点检索支持证据
    """
    
    def __init__(self) -> None:
        self.knowledge_bases: Dict[str, KnowledgeBase] = {}
        self.scorer = RelevanceScorer()
        self.retrieval_history: List[RetrievalResult] = []
    
    def register_knowledge_base(self, kb: KnowledgeBase) -> None:
        """注册知识库"""
        self.knowledge_bases[kb.name] = kb
    
    def retrieve(
        self,
        query: RetrievalQuery,
        knowledge_base_names: Optional[List[str]] = None
    ) -> RetrievalResult:
        """
        检索证据
        
        Args:
            query: 检索查询
            knowledge_base_names: 指定知识库名称列表，None表示全部
            
        Returns:
            检索结果
        """
        import time
        start_time = time.time()
        
        # 确定要搜索的知识库
        if knowledge_base_names:
            kbs = [
                self.knowledge_bases[name] 
                for name in knowledge_base_names 
                if name in self.knowledge_bases
            ]
        else:
            kbs = list(self.knowledge_bases.values())
        
        # 收集候选条目
        candidates: List[KnowledgeEntry] = []
        
        for kb in kbs:
            # 按标签预过滤
            if query.required_tags:
                entries = kb.search_by_tags(query.required_tags)
            else:
                entries = list(kb.entries.values())
            
            # 按来源类型过滤
            if query.source_types:
                entries = [
                    e for e in entries 
                    if e.source_type in query.source_types
                ]
            
            # 按可靠性过滤
            entries = [
                e for e in entries 
                if e.reliability.value >= query.min_reliability.value
            ]
            
            # 排除标签
            entries = [
                e for e in entries 
                if not (e.tags & query.exclude_tags)
            ]
            
            candidates.extend(entries)
        
        # 计算相关性分数
        scored_entries = [
            (entry, self.scorer.calculate_score(query, entry))
            for entry in candidates
        ]
        
        # 排序并截取
        scored_entries.sort(key=lambda x: x[1], reverse=True)
        top_entries = scored_entries[:query.max_results]
        
        entries = [e for e, _ in top_entries]
        scores = [s for _, s in top_entries]
        
        retrieval_time = (time.time() - start_time) * 1000
        
        result = RetrievalResult(
            query=query,
            entries=entries,
            scores=scores,
            total_found=len(candidates),
            retrieval_time_ms=retrieval_time
        )
        
        self.retrieval_history.append(result)
        return result
    
    def retrieve_for_argument(
        self,
        argument: Argument,
        max_evidence: int = 3
    ) -> List[Evidence]:
        """
        为论点检索支持证据
        
        Args:
            argument: 论点
            max_evidence: 最大证据数量
            
        Returns:
            证据列表
        """
        # 从论点内容构建查询
        query = RetrievalQuery(
            query_text=argument.content,
            topic=argument.content[:50],
            max_results=max_evidence,
            min_reliability=EvidenceReliability.MEDIUM
        )
        
        result = self.retrieve(query)
        
        # 转换为证据并设置相关度
        evidence_list = []
        for entry, score in result.get_top_k(max_evidence):
            evidence = entry.to_evidence()
            evidence.relevance = score
            evidence_list.append(evidence)
        
        return evidence_list
    
    def verify_claim(
        self,
        claim: str,
        expected_stance: str = "support"
    ) -> Tuple[bool, List[Evidence]]:
        """
        验证主张
        
        Args:
            claim: 要验证的主张
            expected_stance: 期望的立场 (support/refute)
            
        Returns:
            (是否验证通过, 支持证据)
        """
        query = RetrievalQuery(
            query_text=claim,
            max_results=5,
            min_reliability=EvidenceReliability.MEDIUM
        )
        
        result = self.retrieve(query)
        
        if not result.entries:
            return False, []
        
        # 根据证据可靠性判断
        avg_reliability = sum(
            entry.reliability.value for entry in result.entries
        ) / len(result.entries)
        
        verified = avg_reliability >= 0.6
        evidence_list = result.to_evidence_list()
        
        return verified, evidence_list
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """获取检索统计"""
        if not self.retrieval_history:
            return {"message": "暂无检索记录"}
        
        total_queries = len(self.retrieval_history)
        avg_results = sum(r.total_found for r in self.retrieval_history) / total_queries
        avg_time = sum(r.retrieval_time_ms for r in self.retrieval_history) / total_queries
        
        return {
            "total_queries": total_queries,
            "average_results_per_query": avg_results,
            "average_retrieval_time_ms": avg_time,
            "knowledge_bases": list(self.knowledge_bases.keys()),
        }


def create_default_knowledge_base() -> KnowledgeBase:
    """创建默认知识库"""
    kb = KnowledgeBase("default")
    
    # 添加一些示例条目
    sample_entries = [
        KnowledgeEntry(
            entry_id="kb_001",
            content="研究表明，定期运动可以显著降低心血管疾病风险",
            source="医学研究期刊2024",
            source_type=EvidenceSourceType.STATISTICAL,
            reliability=EvidenceReliability.HIGH,
            tags={"健康", "运动", "医学"}
        ),
        KnowledgeEntry(
            entry_id="kb_002",
            content="人工智能技术在医疗诊断中的应用准确率已达到95%以上",
            source="AI医疗报告",
            source_type=EvidenceSourceType.STATISTICAL,
            reliability=EvidenceReliability.MEDIUM,
            tags={"人工智能", "医疗", "技术"}
        ),
        KnowledgeEntry(
            entry_id="kb_003",
            content="教育专家普遍认为，个性化学习能提高学生的学习效率",
            source="教育研究协会",
            source_type=EvidenceSourceType.EXPERT_OPINION,
            reliability=EvidenceReliability.HIGH,
            tags={"教育", "学习", "专家意见"}
        ),
    ]
    
    for entry in sample_entries:
        kb.add_entry(entry)
    
    return kb


__all__ = [
    "EvidenceSourceType",
    "EvidenceReliability",
    "KnowledgeEntry",
    "RetrievalQuery",
    "RetrievalResult",
    "KnowledgeBase",
    "RelevanceScorer",
    "EvidenceRetriever",
    "create_default_knowledge_base",
]
