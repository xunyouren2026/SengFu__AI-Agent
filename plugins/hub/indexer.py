"""
全文搜索引擎模块

提供倒排索引、模糊搜索、标签分类和推荐算法功能。
"""

import json
import os
import re
import math
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from collections import defaultdict
from difflib import SequenceMatcher


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

@dataclass
class SearchQuery:
    """搜索查询"""
    query: str
    filters: Dict[str, Any] = field(default_factory=dict)
    sort_by: str = "relevance"
    sort_order: str = "desc"
    page: int = 1
    page_size: int = 20
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResult:
    """搜索结果"""
    items: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int
    query: str
    execution_time_ms: float = 0.0
    suggestions: List[str] = field(default_factory=list)
    facets: Dict[str, List[Tuple[str, int]]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'items': self.items,
            'total': self.total,
            'page': self.page,
            'page_size': self.page_size,
            'query': self.query,
            'execution_time_ms': self.execution_time_ms,
            'suggestions': self.suggestions,
            'facets': self.facets,
        }


@dataclass
class IndexDocument:
    """索引文档"""
    doc_id: str
    title: str = ""
    content: str = ""
    tags: List[str] = field(default_factory=list)
    category: str = ""
    author: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexDocument":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# 倒排索引
# ---------------------------------------------------------------------------

class InvertedIndex:
    """倒排索引
    
    实现基于词项的倒排索引，支持布尔查询和短语查询。
    """
    
    def __init__(self):
        # term -> {doc_id: [positions]}
        self._index: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        # doc_id -> document
        self._documents: Dict[str, IndexDocument] = {}
        # doc_id -> term frequency map
        self._doc_term_freq: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # 文档数量
        self._doc_count = 0
        # 总词项数
        self._total_terms = 0
        
        self._lock = threading.RLock()
    
    def add_document(self, doc: IndexDocument) -> None:
        """添加文档到索引"""
        with self._lock:
            # 如果文档已存在，先删除
            if doc.doc_id in self._documents:
                self.remove_document(doc.doc_id)
            
            # 分词
            tokens = self._tokenize(doc.title + " " + doc.content)
            
            # 构建位置索引
            for pos, token in enumerate(tokens):
                self._index[token][doc.doc_id].append(pos)
                self._doc_term_freq[doc.doc_id][token] += 1
            
            # 索引标签
            for tag in doc.tags:
                tag_token = f"tag:{tag.lower()}"
                self._index[tag_token][doc.doc_id].append(-1)
            
            # 索引分类
            if doc.category:
                cat_token = f"cat:{doc.category.lower()}"
                self._index[cat_token][doc.doc_id].append(-1)
            
            # 存储文档
            self._documents[doc.doc_id] = doc
            self._doc_count += 1
            self._total_terms += len(tokens)
    
    def remove_document(self, doc_id: str) -> bool:
        """从索引中移除文档"""
        with self._lock:
            if doc_id not in self._documents:
                return False
            
            doc = self._documents[doc_id]
            
            # 移除所有词项
            tokens = self._tokenize(doc.title + " " + doc.content)
            for token in set(tokens):
                if doc_id in self._index[token]:
                    del self._index[token][doc_id]
                    if not self._index[token]:
                        del self._index[token]
            
            # 移除标签索引
            for tag in doc.tags:
                tag_token = f"tag:{tag.lower()}"
                if doc_id in self._index.get(tag_token, {}):
                    del self._index[tag_token][doc_id]
            
            # 移除分类索引
            if doc.category:
                cat_token = f"cat:{doc.category.lower()}"
                if doc_id in self._index.get(cat_token, {}):
                    del self._index[cat_token][doc_id]
            
            # 清理文档数据
            del self._documents[doc_id]
            del self._doc_term_freq[doc_id]
            self._doc_count -= 1
            
            return True
    
    def search(self, query: str, 
               filters: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        """搜索文档
        
        Args:
            query: 查询字符串
            filters: 过滤器
            
        Returns:
            (doc_id, score)列表
        """
        with self._lock:
            query_tokens = self._tokenize(query)
            if not query_tokens:
                return []
            
            # 获取候选文档
            candidates: Set[str] = set()
            first = True
            
            for token in query_tokens:
                docs_with_term = set(self._index.get(token, {}).keys())
                if first:
                    candidates = docs_with_term
                    first = False
                else:
                    candidates &= docs_with_term
            
            # 计算TF-IDF分数
            scores = []
            for doc_id in candidates:
                score = self._calculate_tf_idf(doc_id, query_tokens)
                
                # 应用过滤器
                if filters and not self._apply_filters(doc_id, filters):
                    continue
                
                scores.append((doc_id, score))
            
            # 排序
            scores.sort(key=lambda x: x[1], reverse=True)
            
            return scores
    
    def search_boolean(self, must: List[str] = None,
                       should: List[str] = None,
                       must_not: List[str] = None) -> Set[str]:
        """布尔查询
        
        Args:
            must: 必须包含的词
            should: 应该包含的词
            must_not: 必须不包含的词
            
        Returns:
            匹配的文档ID集合
        """
        with self._lock:
            result = set(self._documents.keys())
            
            # must_not
            if must_not:
                for term in must_not:
                    result -= set(self._index.get(term, {}).keys())
            
            # must
            if must:
                must_docs = None
                for term in must:
                    docs = set(self._index.get(term, {}).keys())
                    if must_docs is None:
                        must_docs = docs
                    else:
                        must_docs &= docs
                if must_docs is not None:
                    result &= must_docs
            
            # should (至少匹配一个)
            if should:
                should_docs = set()
                for term in should:
                    should_docs |= set(self._index.get(term, {}).keys())
                if should_docs:
                    result &= should_docs
            
            return result
    
    def search_phrase(self, phrase: str) -> List[str]:
        """短语查询
        
        Args:
            phrase: 短语
            
        Returns:
            匹配的文档ID列表
        """
        with self._lock:
            tokens = self._tokenize(phrase)
            if not tokens:
                return []
            
            # 获取包含所有词项的文档
            candidates = set(self._index.get(tokens[0], {}).keys())
            for token in tokens[1:]:
                candidates &= set(self._index.get(token, {}).keys())
            
            # 检查位置连续性
            results = []
            for doc_id in candidates:
                positions = self._index[tokens[0]][doc_id]
                for start_pos in positions:
                    match = True
                    for i, token in enumerate(tokens[1:], 1):
                        expected_pos = start_pos + i
                        if expected_pos not in self._index[token].get(doc_id, []):
                            match = False
                            break
                    if match:
                        results.append(doc_id)
                        break
            
            return results
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        if not text:
            return []
        
        # 转换为小写
        text = text.lower()
        
        # 提取单词
        tokens = re.findall(r'\b[a-z][a-z0-9]*\b', text)
        
        # 过滤停用词
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                      'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                      'through', 'during', 'before', 'after', 'above', 'below',
                      'between', 'under', 'and', 'but', 'or', 'yet', 'so', 'if',
                      'because', 'although', 'though', 'while', 'where', 'when',
                      'that', 'which', 'who', 'whom', 'whose', 'what', 'this',
                      'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our',
                      'you', 'your', 'he', 'him', 'his', 'she', 'her', 'it',
                      'its', 'they', 'them', 'their'}
        
        return [t for t in tokens if t not in stop_words and len(t) > 1]
    
    def _calculate_tf_idf(self, doc_id: str, query_tokens: List[str]) -> float:
        """计算TF-IDF分数"""
        score = 0.0
        
        for token in query_tokens:
            # TF
            tf = self._doc_term_freq[doc_id].get(token, 0)
            
            # IDF
            doc_freq = len(self._index.get(token, {}))
            if doc_freq > 0:
                idf = math.log(self._doc_count / doc_freq)
            else:
                idf = 0
            
            score += tf * idf
        
        return score
    
    def _apply_filters(self, doc_id: str, filters: Dict[str, Any]) -> bool:
        """应用过滤器"""
        doc = self._documents.get(doc_id)
        if not doc:
            return False
        
        for key, value in filters.items():
            if key == 'category' and doc.category != value:
                return False
            if key == 'author' and doc.author != value:
                return False
            if key == 'tags' and value not in doc.tags:
                return False
            if key == 'after' and doc.timestamp < value:
                return False
            if key == 'before' and doc.timestamp > value:
                return False
        
        return True
    
    def get_document(self, doc_id: str) -> Optional[IndexDocument]:
        """获取文档"""
        with self._lock:
            return self._documents.get(doc_id)
    
    def get_term_freq(self, term: str) -> int:
        """获取词项文档频率"""
        with self._lock:
            return len(self._index.get(term, {}))
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                'document_count': self._doc_count,
                'unique_terms': len(self._index),
                'total_terms': self._total_terms,
                'avg_doc_length': self._total_terms / max(self._doc_count, 1),
            }
    
    def clear(self) -> None:
        """清空索引"""
        with self._lock:
            self._index.clear()
            self._documents.clear()
            self._doc_term_freq.clear()
            self._doc_count = 0
            self._total_terms = 0


# ---------------------------------------------------------------------------
# 模糊搜索器
# ---------------------------------------------------------------------------

class FuzzySearcher:
    """模糊搜索器
    
    支持拼写纠错和近似匹配。
    """
    
    def __init__(self, max_distance: int = 2):
        """
        Args:
            max_distance: 最大编辑距离
        """
        self._max_distance = max_distance
        self._dictionary: Set[str] = set()
        self._lock = threading.Lock()
    
    def add_terms(self, terms: List[str]) -> None:
        """添加词项到词典"""
        with self._lock:
            self._dictionary.update(t.lower() for t in terms)
    
    def search(self, query: str, limit: int = 5) -> List[Tuple[str, float]]:
        """模糊搜索
        
        Args:
            query: 查询词
            limit: 返回数量限制
            
        Returns:
            (匹配词, 相似度)列表
        """
        with self._lock:
            query = query.lower()
            matches = []
            
            for term in self._dictionary:
                distance = self._levenshtein_distance(query, term)
                if distance <= self._max_distance:
                    similarity = 1.0 - (distance / max(len(query), len(term)))
                    matches.append((term, similarity))
            
            # 按相似度排序
            matches.sort(key=lambda x: x[1], reverse=True)
            
            return matches[:limit]
    
    def suggest(self, query: str, limit: int = 5) -> List[str]:
        """获取搜索建议"""
        matches = self.search(query, limit)
        return [m[0] for m in matches]
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """计算编辑距离"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _similarity_ratio(self, s1: str, s2: str) -> float:
        """计算相似度比例"""
        return SequenceMatcher(None, s1, s2).ratio()


# ---------------------------------------------------------------------------
# 标签分类器
# ---------------------------------------------------------------------------

class TagClassifier:
    """标签分类器
    
    自动分类和标签推荐。
    """
    
    def __init__(self):
        # 类别 -> 关键词
        self._category_keywords: Dict[str, List[str]] = {
            'productivity': ['task', 'todo', 'schedule', 'calendar', 'time', 'productivity'],
            'development': ['code', 'programming', 'debug', 'git', 'api', 'developer'],
            'communication': ['chat', 'message', 'email', 'notification', 'collaboration'],
            'media': ['image', 'video', 'audio', 'media', 'player', 'stream'],
            'security': ['security', 'password', 'encrypt', 'privacy', 'protect'],
            'utility': ['tool', 'utility', 'converter', 'calculator', 'helper'],
        }
        
        # 标签共现统计
        self._tag_cooccurrence: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._tag_frequency: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
    
    def classify(self, text: str, existing_tags: List[str] = None) -> List[Tuple[str, float]]:
        """自动分类
        
        Args:
            text: 文本内容
            existing_tags: 已有标签
            
        Returns:
            (类别, 置信度)列表
        """
        text = text.lower()
        scores = []
        
        for category, keywords in self._category_keywords.items():
            score = 0
            for keyword in keywords:
                count = text.count(keyword)
                score += count
            
            # 归一化
            if keywords:
                score /= len(keywords)
            
            scores.append((category, score))
        
        # 排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回置信度大于0的
        return [(c, min(s / max(scores[0][1], 0.001), 1.0)) for c, s in scores if s > 0]
    
    def suggest_tags(self, text: str, existing_tags: List[str] = None,
                     limit: int = 10) -> List[Tuple[str, float]]:
        """推荐标签
        
        Args:
            text: 文本内容
            existing_tags: 已有标签
            limit: 返回数量
            
        Returns:
            (标签, 分数)列表
        """
        with self._lock:
            text = text.lower()
            
            # 基于内容提取候选标签
            candidates = self._extract_candidate_tags(text)
            
            # 基于共现推荐
            if existing_tags:
                for tag in existing_tags:
                    for co_tag, count in self._tag_cooccurrence.get(tag, {}).items():
                        if co_tag not in existing_tags:
                            candidates[co_tag] += count * 0.5
            
            # 排序
            suggestions = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
            
            return suggestions[:limit]
    
    def _extract_candidate_tags(self, text: str) -> Dict[str, float]:
        """从文本中提取候选标签"""
        candidates = defaultdict(float)
        
        # 提取关键词（简单实现）
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
        
        # 基于频率和长度评分
        for word, freq in word_freq.items():
            if freq >= 2 or len(word) >= 5:
                score = freq * (1 + len(word) / 10)
                candidates[word] = score
        
        return candidates
    
    def record_tag_cooccurrence(self, tags: List[str]) -> None:
        """记录标签共现"""
        with self._lock:
            for tag in tags:
                self._tag_frequency[tag] += 1
            
            for i, tag1 in enumerate(tags):
                for tag2 in tags[i+1:]:
                    self._tag_cooccurrence[tag1][tag2] += 1
                    self._tag_cooccurrence[tag2][tag1] += 1
    
    def get_related_tags(self, tag: str, limit: int = 10) -> List[Tuple[str, float]]:
        """获取相关标签"""
        with self._lock:
            related = self._tag_cooccurrence.get(tag, {})
            
            # 计算Jaccard相似度
            results = []
            for other_tag, co_count in related.items():
                tag_freq = self._tag_frequency[tag]
                other_freq = self._tag_frequency[other_tag]
                jaccard = co_count / (tag_freq + other_freq - co_count)
                results.append((other_tag, jaccard))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:limit]
    
    def get_popular_tags(self, limit: int = 20) -> List[Tuple[str, int]]:
        """获取热门标签"""
        with self._lock:
            tags = sorted(self._tag_frequency.items(), key=lambda x: x[1], reverse=True)
            return tags[:limit]


# ---------------------------------------------------------------------------
# 推荐引擎
# ---------------------------------------------------------------------------

class RecommendationEngine:
    """推荐引擎
    
    基于协同过滤和内容相似度的推荐。
    """
    
    def __init__(self):
        # 用户 -> 项目 -> 评分
        self._user_ratings: Dict[str, Dict[str, float]] = defaultdict(dict)
        # 项目 -> 用户 -> 评分
        self._item_ratings: Dict[str, Dict[str, float]] = defaultdict(dict)
        # 项目特征
        self._item_features: Dict[str, Dict[str, Any]] = {}
        
        self._lock = threading.Lock()
    
    def add_rating(self, user_id: str, item_id: str, rating: float) -> None:
        """添加评分"""
        with self._lock:
            self._user_ratings[user_id][item_id] = rating
            self._item_ratings[item_id][user_id] = rating
    
    def add_item_features(self, item_id: str, features: Dict[str, Any]) -> None:
        """添加项目特征"""
        with self._lock:
            self._item_features[item_id] = features
    
    def recommend_collaborative(self, user_id: str, limit: int = 10) -> List[Tuple[str, float]]:
        """基于协同过滤的推荐
        
        Args:
            user_id: 用户ID
            limit: 返回数量
            
        Returns:
            (项目ID, 预测评分)列表
        """
        with self._lock:
            user_ratings = self._user_ratings.get(user_id, {})
            
            if not user_ratings:
                return []
            
            # 找到相似用户
            similar_users = self._find_similar_users(user_id)
            
            # 预测评分
            predictions = {}
            
            for similar_user, similarity in similar_users:
                for item_id, rating in self._user_ratings[similar_user].items():
                    if item_id not in user_ratings:
                        if item_id not in predictions:
                            predictions[item_id] = {'sum': 0, 'weight': 0}
                        predictions[item_id]['sum'] += similarity * rating
                        predictions[item_id]['weight'] += similarity
            
            # 计算加权平均
            results = []
            for item_id, data in predictions.items():
                if data['weight'] > 0:
                    pred_rating = data['sum'] / data['weight']
                    results.append((item_id, pred_rating))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:limit]
    
    def recommend_content_based(self, item_id: str, limit: int = 10) -> List[Tuple[str, float]]:
        """基于内容的推荐
        
        Args:
            item_id: 参考项目ID
            limit: 返回数量
            
        Returns:
            (项目ID, 相似度)列表
        """
        with self._lock:
            target_features = self._item_features.get(item_id)
            if not target_features:
                return []
            
            similarities = []
            
            for other_id, other_features in self._item_features.items():
                if other_id != item_id:
                    sim = self._calculate_feature_similarity(target_features, other_features)
                    if sim > 0:
                        similarities.append((other_id, sim))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:limit]
    
    def recommend_trending(self, time_window: int = 86400,
                           limit: int = 10) -> List[Tuple[str, float]]:
        """推荐热门项目
        
        Args:
            time_window: 时间窗口（秒）
            limit: 返回数量
            
        Returns:
            (项目ID, 热度分数)列表
        """
        with self._lock:
            cutoff = time.time() - time_window
            
            # 计算近期活跃度
            item_scores = defaultdict(float)
            
            for item_id, ratings in self._item_ratings.items():
                recent_ratings = [r for r in ratings.values() if isinstance(r, dict) and r.get('timestamp', 0) > cutoff]
                if recent_ratings:
                    avg_rating = sum(r['value'] if isinstance(r, dict) else r for r in recent_ratings) / len(recent_ratings)
                    velocity = len(recent_ratings) / (time_window / 3600)  # 每小时评分数
                    item_scores[item_id] = avg_rating * math.log1p(velocity)
            
            results = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
            return results[:limit]
    
    def _find_similar_users(self, user_id: str, limit: int = 20) -> List[Tuple[str, float]]:
        """找到相似用户"""
        target_ratings = self._user_ratings.get(user_id, {})
        
        if not target_ratings:
            return []
        
        similarities = []
        
        for other_id, other_ratings in self._user_ratings.items():
            if other_id != user_id:
                sim = self._calculate_user_similarity(target_ratings, other_ratings)
                if sim > 0:
                    similarities.append((other_id, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]
    
    def _calculate_user_similarity(self, ratings1: Dict[str, float],
                                    ratings2: Dict[str, float]) -> float:
        """计算用户相似度（余弦相似度）"""
        common_items = set(ratings1.keys()) & set(ratings2.keys())
        
        if not common_items:
            return 0.0
        
        # 皮尔逊相关系数
        sum1 = sum(ratings1[i] for i in common_items)
        sum2 = sum(ratings2[i] for i in common_items)
        sum1_sq = sum(ratings1[i]**2 for i in common_items)
        sum2_sq = sum(ratings2[i]**2 for i in common_items)
        p_sum = sum(ratings1[i] * ratings2[i] for i in common_items)
        
        n = len(common_items)
        num = p_sum - (sum1 * sum2 / n)
        den = math.sqrt((sum1_sq - sum1**2 / n) * (sum2_sq - sum2**2 / n))
        
        if den == 0:
            return 0.0
        
        return num / den
    
    def _calculate_feature_similarity(self, features1: Dict[str, Any],
                                       features2: Dict[str, Any]) -> float:
        """计算特征相似度"""
        # 基于标签的Jaccard相似度
        tags1 = set(features1.get('tags', []))
        tags2 = set(features2.get('tags', []))
        
        if not tags1 or not tags2:
            return 0.0
        
        intersection = len(tags1 & tags2)
        union = len(tags1 | tags2)
        
        tag_sim = intersection / union if union > 0 else 0
        
        # 类别相同加分
        cat_sim = 1.0 if features1.get('category') == features2.get('category') else 0.0
        
        # 作者相同加分
        author_sim = 0.5 if features1.get('author') == features2.get('author') else 0.0
        
        return tag_sim * 0.6 + cat_sim * 0.3 + author_sim * 0.1


# ---------------------------------------------------------------------------
# 全文索引器
# ---------------------------------------------------------------------------

class FullTextIndexer:
    """全文索引器
    
    整合所有搜索功能的主类。
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Args:
            storage_path: 存储路径
        """
        self._storage_path = storage_path or os.path.join(
            os.path.expanduser("~"), ".clawhub", "index"
        )
        
        self._inverted_index = InvertedIndex()
        self._fuzzy_searcher = FuzzySearcher()
        self._tag_classifier = TagClassifier()
        self._recommendation = RecommendationEngine()
        
        self._lock = threading.Lock()
        
        os.makedirs(self._storage_path, exist_ok=True)
        self._load_from_disk()
    
    def index_document(self, doc: IndexDocument) -> None:
        """索引文档"""
        self._inverted_index.add_document(doc)
        
        # 更新模糊搜索词典
        tokens = self._inverted_index._tokenize(doc.title + " " + doc.content)
        self._fuzzy_searcher.add_terms(tokens)
        self._fuzzy_searcher.add_terms(doc.tags)
        
        # 记录标签共现
        self._tag_classifier.record_tag_cooccurrence(doc.tags)
        
        # 添加推荐特征
        self._recommendation.add_item_features(doc.doc_id, {
            'tags': doc.tags,
            'category': doc.category,
            'author': doc.author,
        })
    
    def remove_document(self, doc_id: str) -> bool:
        """移除文档"""
        return self._inverted_index.remove_document(doc_id)
    
    def search(self, query: SearchQuery) -> SearchResult:
        """搜索
        
        Args:
            query: 搜索查询
            
        Returns:
            搜索结果
        """
        start_time = time.time()
        
        # 执行搜索
        results = self._inverted_index.search(query.query, query.filters)
        
        # 分页
        total = len(results)
        start = (query.page - 1) * query.page_size
        end = start + query.page_size
        page_results = results[start:end]
        
        # 获取完整文档
        items = []
        for doc_id, score in page_results:
            doc = self._inverted_index.get_document(doc_id)
            if doc:
                item = doc.to_dict()
                item['_score'] = score
                items.append(item)
        
        # 获取建议
        suggestions = self._fuzzy_searcher.suggest(query.query, 3)
        
        # 计算分面
        facets = self._compute_facets([r[0] for r in results])
        
        execution_time = (time.time() - start_time) * 1000
        
        return SearchResult(
            items=items,
            total=total,
            page=query.page,
            page_size=query.page_size,
            query=query.query,
            execution_time_ms=execution_time,
            suggestions=suggestions,
            facets=facets,
        )
    
    def search_fuzzy(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """模糊搜索"""
        matches = self._fuzzy_searcher.search(query, limit)
        
        results = []
        for term, similarity in matches:
            # 查找包含该词的文档
            doc_ids = list(self._inverted_index._index.get(term, {}).keys())
            for doc_id in doc_ids[:5]:
                doc = self._inverted_index.get_document(doc_id)
                if doc:
                    item = doc.to_dict()
                    item['_match'] = term
                    item['_similarity'] = similarity
                    results.append(item)
        
        return results
    
    def get_suggestions(self, prefix: str, limit: int = 10) -> List[str]:
        """获取搜索建议"""
        return self._fuzzy_searcher.suggest(prefix, limit)
    
    def classify(self, text: str) -> List[Tuple[str, float]]:
        """自动分类"""
        return self._tag_classifier.classify(text)
    
    def suggest_tags(self, text: str, existing_tags: List[str] = None,
                     limit: int = 10) -> List[Tuple[str, float]]:
        """推荐标签"""
        return self._tag_classifier.suggest_tags(text, existing_tags, limit)
    
    def recommend_similar(self, doc_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """推荐相似文档"""
        results = self._recommendation.recommend_content_based(doc_id, limit)
        
        items = []
        for item_id, score in results:
            doc = self._inverted_index.get_document(item_id)
            if doc:
                item = doc.to_dict()
                item['_similarity'] = score
                items.append(item)
        
        return items
    
    def recommend_for_user(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """为用户推荐"""
        results = self._recommendation.recommend_collaborative(user_id, limit)
        
        items = []
        for item_id, score in results:
            doc = self._inverted_index.get_document(item_id)
            if doc:
                item = doc.to_dict()
                item['_predicted_rating'] = score
                items.append(item)
        
        return items
    
    def get_trending(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取热门"""
        results = self._recommendation.recommend_trending(limit=limit)
        
        items = []
        for item_id, score in results:
            doc = self._inverted_index.get_document(item_id)
            if doc:
                item = doc.to_dict()
                item['_trending_score'] = score
                items.append(item)
        
        return items
    
    def _compute_facets(self, doc_ids: List[str]) -> Dict[str, List[Tuple[str, int]]]:
        """计算分面统计"""
        facets = {
            'category': defaultdict(int),
            'tags': defaultdict(int),
            'author': defaultdict(int),
        }
        
        for doc_id in doc_ids:
            doc = self._inverted_index.get_document(doc_id)
            if doc:
                facets['category'][doc.category] += 1
                for tag in doc.tags:
                    facets['tags'][tag] += 1
                facets['author'][doc.author] += 1
        
        # 转换为排序列表
        result = {}
        for key, counts in facets.items():
            result[key] = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'index_stats': self._inverted_index.get_stats(),
            'popular_tags': self._tag_classifier.get_popular_tags(20),
        }
    
    def save_to_disk(self) -> None:
        """保存到磁盘"""
        try:
            data = {
                'documents': {
                    doc_id: doc.to_dict()
                    for doc_id, doc in self._inverted_index._documents.items()
                },
                'tag_frequency': dict(self._tag_classifier._tag_frequency),
                'tag_cooccurrence': {
                    k: dict(v) for k, v in self._tag_classifier._tag_cooccurrence.items()
                },
            }
            
            file_path = os.path.join(self._storage_path, 'index.json')
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
    
    def _load_from_disk(self) -> None:
        """从磁盘加载"""
        file_path = os.path.join(self._storage_path, 'index.json')
        if not os.path.exists(file_path):
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 加载文档
            for doc_id, doc_data in data.get('documents', {}).items():
                doc = IndexDocument.from_dict(doc_data)
                self._inverted_index.add_document(doc)
            
            # 加载标签统计
            self._tag_classifier._tag_frequency = defaultdict(
                int, data.get('tag_frequency', {})
            )
            
            for tag, cooccur in data.get('tag_cooccurrence', {}).items():
                self._tag_classifier._tag_cooccurrence[tag] = defaultdict(int, cooccur)
        except Exception:
            pass
