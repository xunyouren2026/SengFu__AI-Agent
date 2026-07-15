"""
市场发现API - 按能力、价格、评分搜索Agent

提供多维度的Agent服务搜索和推荐功能，支持语义匹配、价格筛选、信誉排序等。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Callable, Any, Tuple, Union
from collections import defaultdict
import heapq

from .listings import ServiceListing, ListingManager, Capability, ServiceType, PricingModel


class SortCriteria(Enum):
    """排序标准"""
    RELEVANCE = auto()           # 相关性
    PRICE_ASC = auto()           # 价格升序
    PRICE_DESC = auto()          # 价格降序
    RATING = auto()              # 评分
    COMPLETION_RATE = auto()     # 完成率
    RESPONSE_TIME = auto()       # 响应时间
    POPULARITY = auto()          #  popularity
    TRUST_SCORE = auto()         # 信任分数


class FilterOperator(Enum):
    """过滤操作符"""
    EQ = auto()      # 等于
    NE = auto()      # 不等于
    GT = auto()      # 大于
    GTE = auto()     # 大于等于
    LT = auto()      # 小于
    LTE = auto()     # 小于等于
    IN = auto()      # 包含
    CONTAINS = auto()  # 字符串包含
    REGEX = auto()   # 正则匹配


@dataclass
class FilterCondition:
    """过滤条件"""
    field: str
    operator: FilterOperator
    value: Any
    
    def evaluate(self, listing: ServiceListing) -> bool:
        """评估条件是否满足"""
        if not hasattr(listing, self.field):
            return False
        
        field_value = getattr(listing, self.field)
        
        if self.operator == FilterOperator.EQ:
            return field_value == self.value
        elif self.operator == FilterOperator.NE:
            return field_value != self.value
        elif self.operator == FilterOperator.GT:
            return field_value > self.value
        elif self.operator == FilterOperator.GTE:
            return field_value >= self.value
        elif self.operator == FilterOperator.LT:
            return field_value < self.value
        elif self.operator == FilterOperator.LTE:
            return field_value <= self.value
        elif self.operator == FilterOperator.IN:
            return field_value in self.value if isinstance(self.value, (list, set, tuple)) else self.value in field_value
        elif self.operator == FilterOperator.CONTAINS:
            return self.value in str(field_value)
        elif self.operator == FilterOperator.REGEX:
            return bool(re.search(self.value, str(field_value)))
        
        return False


@dataclass
class SearchQuery:
    """搜索查询"""
    keywords: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    service_types: Set[ServiceType] = field(default_factory=set)
    price_range: Optional[Tuple[float, float]] = None
    min_rating: float = 0.0
    min_completed_tasks: int = 0
    supported_regions: Set[str] = field(default_factory=set)
    filters: List[FilterCondition] = field(default_factory=list)
    sort_by: SortCriteria = SortCriteria.RELEVANCE
    limit: int = 20
    offset: int = 0


@dataclass
class SearchResult:
    """搜索结果"""
    listing: ServiceListing
    relevance_score: float
    matched_capabilities: List[str] = field(default_factory=list)
    matched_tags: Set[str] = field(default_factory=set)
    estimated_price: Optional[float] = None
    rank_score: float = 0.0


@dataclass
class RecommendationContext:
    """推荐上下文"""
    user_id: str
    history_listings: List[str] = field(default_factory=list)
    preferred_capabilities: Set[str] = field(default_factory=set)
    price_sensitivity: float = 0.5  # 0-1, 越高越在意价格
    quality_preference: float = 0.5  # 0-1, 越高越在意质量
    region: Optional[str] = None


class SemanticMatcher:
    """语义匹配器 - 基于关键词和描述的相似度匹配"""
    
    def __init__(self):
        self._word_vectors: Dict[str, Dict[str, float]] = {}
        self._synonyms: Dict[str, Set[str]] = defaultdict(set)
        self._build_synonym_map()
    
    def _build_synonym_map(self) -> None:
        """构建同义词映射"""
        synonym_groups = [
            {"analysis", "analytics", "analyze", "analytical"},
            {"processing", "process", "handler"},
            {"generation", "generate", "creator", "create"},
            {"classification", "classify", "categorize", "category"},
            {"translation", "translate", "translator"},
            {"summarization", "summarize", "summary"},
            {"extraction", "extract", "parser", "parse"},
            {"validation", "validate", "verification", "verify"},
            {"optimization", "optimize", "improve", "enhance"},
            {"prediction", "predict", "forecast", "forecaster"},
            {"recommendation", "recommend", "suggest"},
            {"search", "retrieval", "retrieve", "find"},
            {"monitoring", "monitor", "watch", "track"},
            {"automation", "automate", "automatic"},
            {"integration", "integrate", "connector", "connect"},
        ]
        
        for group in synonym_groups:
            for word in group:
                self._synonyms[word] = group
    
    def tokenize(self, text: str) -> List[str]:
        """分词"""
        # 简单的分词：转小写，移除非字母数字，分割
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return [w for w in text.split() if len(w) > 2]
    
    def calculate_similarity(self, query: str, text: str) -> float:
        """计算查询与文本的相似度"""
        query_tokens = set(self.tokenize(query))
        text_tokens = set(self.tokenize(text))
        
        if not query_tokens or not text_tokens:
            return 0.0
        
        # 扩展查询词集（包含同义词）
        expanded_query = set()
        for token in query_tokens:
            expanded_query.add(token)
            expanded_query.update(self._synonyms.get(token, set()))
        
        # 计算Jaccard相似度
        intersection = len(expanded_query & text_tokens)
        union = len(expanded_query | text_tokens)
        
        if union == 0:
            return 0.0
        
        jaccard = intersection / union
        
        # 额外奖励：直接匹配
        direct_matches = len(query_tokens & text_tokens)
        direct_bonus = direct_matches / len(query_tokens) if query_tokens else 0
        
        return 0.6 * jaccard + 0.4 * direct_bonus
    
    def match_capabilities(
        self,
        query: str,
        capabilities: List[Capability]
    ) -> Tuple[float, List[str]]:
        """
        匹配能力与查询
        
        Returns:
            (最高相似度, 匹配的能力名称列表)
        """
        max_score = 0.0
        matched = []
        
        for cap in capabilities:
            # 匹配能力名称
            name_score = self.calculate_similarity(query, cap.name)
            # 匹配描述
            desc_score = self.calculate_similarity(query, cap.description)
            # 匹配标签
            tag_score = max(
                [self.calculate_similarity(query, tag) for tag in cap.tags] + [0]
            )
            
            score = max(name_score, desc_score * 0.8, tag_score * 0.9)
            
            if score > 0.3:  # 阈值
                matched.append(cap.name)
                max_score = max(max_score, score)
        
        return max_score, matched


class DiscoveryEngine:
    """发现引擎 - 核心搜索和推荐功能"""
    
    def __init__(self, listing_manager: Optional[ListingManager] = None):
        self._listing_manager = listing_manager or ListingManager()
        self._semantic_matcher = SemanticMatcher()
        self._user_preferences: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._click_history: Dict[str, List[Tuple[str, float]]] = defaultdict(list)  # user_id -> [(listing_id, timestamp)]
        self._conversion_history: Dict[str, List[str]] = defaultdict(list)  # user_id -> [listing_id]
    
    def search(self, query: SearchQuery) -> Tuple[List[SearchResult], int]:
        """
        执行搜索
        
        Args:
            query: 搜索查询
            
        Returns:
            (搜索结果列表, 总匹配数)
        """
        # 获取候选列表
        candidates = self._get_candidates(query)
        
        # 应用过滤
        filtered = self._apply_filters(candidates, query)
        
        # 计算相关性分数
        scored_results = self._score_results(filtered, query)
        
        # 排序
        sorted_results = self._sort_results(scored_results, query.sort_by)
        
        # 分页
        total = len(sorted_results)
        paginated = sorted_results[query.offset:query.offset + query.limit]
        
        return paginated, total
    
    def _get_candidates(self, query: SearchQuery) -> List[ServiceListing]:
        """获取候选列表"""
        candidates: Set[str] = set()
        
        # 基于能力搜索
        if query.capabilities:
            for cap in query.capabilities:
                listings = self._listing_manager.search_by_capability(cap)
                ids = {l.listing_id for l in listings}
                if not candidates:
                    candidates = ids
                else:
                    candidates &= ids  # 交集 - 必须匹配所有能力
        
        # 基于标签搜索
        if query.tags:
            for tag in query.tags:
                listings = self._listing_manager.search_by_tag(tag)
                ids = {l.listing_id for l in listings}
                if not candidates:
                    candidates = ids
                else:
                    candidates &= ids
        
        # 基于类型搜索
        if query.service_types:
            type_ids: Set[str] = set()
            for st in query.service_types:
                listings = self._listing_manager.search_by_type(st)
                type_ids.update(l.listing_id for l in listings)
            if candidates:
                candidates &= type_ids
            else:
                candidates = type_ids
        
        # 如果没有特定条件，获取所有活跃列表
        if not candidates:
            candidates = {l.listing_id for l in self._listing_manager.get_active_listings()}
        
        return [self._listing_manager.get_listing(lid) for lid in candidates if lid]
    
    def _apply_filters(
        self,
        candidates: List[ServiceListing],
        query: SearchQuery
    ) -> List[ServiceListing]:
        """应用过滤条件"""
        filtered = []
        
        for listing in candidates:
            if not listing:
                continue
            
            # 评分过滤
            if listing.average_rating < query.min_rating:
                continue
            
            # 完成任务数过滤
            if listing.completed_tasks < query.min_completed_tasks:
                continue
            
            # 价格范围过滤
            if query.price_range and listing.pricing_tiers:
                min_price = min(t.base_price for t in listing.pricing_tiers)
                max_price = max(t.base_price for t in listing.pricing_tiers)
                if not (query.price_range[0] <= min_price <= query.price_range[1] or
                        query.price_range[0] <= max_price <= query.price_range[1]):
                    continue
            
            # 区域过滤
            if query.supported_regions:
                if not any(r in listing.supported_regions for r in query.supported_regions):
                    continue
            
            # 自定义过滤器
            passes_filters = all(f.evaluate(listing) for f in query.filters)
            if not passes_filters:
                continue
            
            filtered.append(listing)
        
        return filtered
    
    def _score_results(
        self,
        candidates: List[ServiceListing],
        query: SearchQuery
    ) -> List[SearchResult]:
        """计算结果分数"""
        results = []
        
        for listing in candidates:
            relevance = 0.0
            matched_caps = []
            matched_tags = set()
            
            # 关键词匹配
            if query.keywords:
                # 匹配描述
                desc_score = self._semantic_matcher.calculate_similarity(
                    query.keywords, listing.description
                )
                # 匹配Agent名称
                name_score = self._semantic_matcher.calculate_similarity(
                    query.keywords, listing.agent_name
                )
                # 匹配能力
                cap_score, matched_caps = self._semantic_matcher.match_capabilities(
                    query.keywords, listing.capabilities
                )
                
                relevance = max(desc_score, name_score * 0.9, cap_score)
                
                # 匹配标签
                for tag in listing.capabilities[0].tags if listing.capabilities else []:
                    tag_sim = self._semantic_matcher.calculate_similarity(query.keywords, tag)
                    if tag_sim > 0.5:
                        matched_tags.add(tag)
            
            # 计算预估价格
            estimated_price = None
            if listing.pricing_tiers:
                estimated_price = min(t.base_price for t in listing.pricing_tiers)
            
            result = SearchResult(
                listing=listing,
                relevance_score=relevance,
                matched_capabilities=matched_caps,
                matched_tags=matched_tags,
                estimated_price=estimated_price
            )
            results.append(result)
        
        return results
    
    def _sort_results(
        self,
        results: List[SearchResult],
        criteria: SortCriteria
    ) -> List[SearchResult]:
        """排序结果"""
        
        def get_sort_key(r: SearchResult) -> Tuple[float, ...]:
            listing = r.listing
            
            if criteria == SortCriteria.RELEVANCE:
                # 综合考虑相关性、评分、价格
                rating_factor = listing.average_rating / 5.0 if listing.average_rating > 0 else 0.5
                price_factor = 1.0 / (1 + (r.estimated_price or 0) / 100)
                trust_score = self._calculate_trust_score(listing)
                r.rank_score = 0.4 * r.relevance_score + 0.3 * rating_factor + 0.2 * trust_score + 0.1 * price_factor
                return (-r.rank_score, -listing.completed_tasks)
            
            elif criteria == SortCriteria.PRICE_ASC:
                return (r.estimated_price or float('inf'), -listing.average_rating)
            
            elif criteria == SortCriteria.PRICE_DESC:
                return (-(r.estimated_price or 0), -listing.average_rating)
            
            elif criteria == SortCriteria.RATING:
                return (-listing.average_rating, -listing.completed_tasks)
            
            elif criteria == SortCriteria.COMPLETION_RATE:
                # 使用完成任务数作为代理指标
                return (-listing.completed_tasks, -listing.average_rating)
            
            elif criteria == SortCriteria.RESPONSE_TIME:
                # 假设SLA中有响应时间
                sla_time = listing.sla_guarantees.get('response_time_ms', float('inf'))
                return (sla_time, -listing.average_rating)
            
            elif criteria == SortCriteria.POPULARITY:
                popularity = listing.completed_tasks + listing.total_reviews
                return (-popularity, -listing.average_rating)
            
            elif criteria == SortCriteria.TRUST_SCORE:
                trust = self._calculate_trust_score(listing)
                return (-trust, -listing.average_rating)
            
            return (0,)
        
        return sorted(results, key=get_sort_key)
    
    def _calculate_trust_score(self, listing: ServiceListing) -> float:
        """计算信任分数"""
        if listing.total_reviews == 0:
            # 新Agent的默认信任分数
            return 0.5
        
        # 基于评分的信任
        rating_score = listing.average_rating / 5.0
        
        # 基于历史表现的信任（完成的任务越多越可信）
        experience_factor = min(1.0, listing.completed_tasks / 100)
        
        # 威尔逊区间下界（处理少量评价的情况）
        n = listing.total_reviews
        p = rating_score
        z = 1.96  # 95%置信区间
        
        wilson_score = (p + z*z/(2*n) - z * math.sqrt((p*(1-p) + z*z/(4*n))/n)) / (1 + z*z/n)
        
        return 0.6 * wilson_score + 0.4 * experience_factor
    
    def recommend(
        self,
        context: RecommendationContext,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        基于用户上下文推荐Agent服务
        
        Args:
            context: 推荐上下文
            limit: 返回数量
            
        Returns:
            推荐结果列表
        """
        # 获取用户历史交互
        user_clicks = self._click_history.get(context.user_id, [])
        user_conversions = self._conversion_history.get(context.user_id, [])
        
        # 构建用户画像
        preferred_caps = set(context.preferred_capabilities)
        for listing_id, _ in user_clicks[-10:]:  # 最近10次点击
            listing = self._listing_manager.get_listing(listing_id)
            if listing:
                preferred_caps.update(c.name for c in listing.capabilities)
        
        # 构建搜索查询
        query = SearchQuery(
            capabilities=list(preferred_caps),
            min_rating=3.0 if context.quality_preference > 0.5 else 0.0,
            sort_by=SortCriteria.TRUST_SCORE,
            limit=limit * 3  # 获取更多候选用于重排
        )
        
        if context.region:
            query.supported_regions = {context.region}
        
        results, _ = self.search(query)
        
        # 个性化重排
        reranked = self._personalize_ranking(results, context, limit)
        
        return reranked
    
    def _personalize_ranking(
        self,
        results: List[SearchResult],
        context: RecommendationContext,
        limit: int
    ) -> List[SearchResult]:
        """个性化重排"""
        scored = []
        
        for result in results:
            listing = result.listing
            base_score = result.rank_score or result.relevance_score
            
            # 价格偏好调整
            if result.estimated_price:
                price_score = 1.0 / (1 + math.log1p(result.estimated_price / 10))
                if context.price_sensitivity > 0.5:
                    base_score *= (0.5 + 0.5 * price_score)
            
            # 质量偏好调整
            if context.quality_preference > 0.5:
                quality_boost = listing.average_rating / 5.0
                base_score *= (0.7 + 0.3 * quality_boost)
            
            # 多样性惩罚（避免推荐相似的）
            diversity_penalty = 0
            for _, prev_score in scored[:3]:
                similarity = self._calculate_listing_similarity(result.listing, scored[scored.index((result, base_score)) - 1][0].listing if scored else None)
                diversity_penalty += similarity * 0.1
            
            final_score = base_score - diversity_penalty
            scored.append((result, final_score))
        
        # 按分数排序
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [r for r, _ in scored[:limit]]
    
    def _calculate_listing_similarity(
        self,
        listing1: ServiceListing,
        listing2: Optional[ServiceListing]
    ) -> float:
        """计算两个上架条目的相似度"""
        if not listing2:
            return 0.0
        
        # 能力重叠
        caps1 = {c.name for c in listing1.capabilities}
        caps2 = {c.name for c in listing2.capabilities}
        
        if not caps1 or not caps2:
            return 0.0
        
        intersection = len(caps1 & caps2)
        union = len(caps1 | caps2)
        
        return intersection / union if union > 0 else 0.0
    
    def record_click(self, user_id: str, listing_id: str) -> None:
        """记录用户点击"""
        import time
        self._click_history[user_id].append((listing_id, time.time()))
        # 保持最近100条
        self._click_history[user_id] = self._click_history[user_id][-100:]
    
    def record_conversion(self, user_id: str, listing_id: str) -> None:
        """记录用户转化（下单）"""
        self._conversion_history[user_id].append(listing_id)
    
    def get_trending(
        self,
        time_window_hours: float = 24.0,
        limit: int = 10
    ) -> List[Tuple[ServiceListing, float]]:
        """
        获取热门趋势
        
        Args:
            time_window_hours: 时间窗口（小时）
            limit: 返回数量
            
        Returns:
            [(上架条目, 趋势分数), ...]
        """
        import time
        cutoff = time.time() - time_window_hours * 3600
        
        # 统计近期点击
        listing_clicks: Dict[str, int] = defaultdict(int)
        for user_id, clicks in self._click_history.items():
            for listing_id, timestamp in clicks:
                if timestamp > cutoff:
                    listing_clicks[listing_id] += 1
        
        # 计算趋势分数（考虑增长率和绝对数量）
        trending = []
        for listing_id, clicks in listing_clicks.items():
            listing = self._listing_manager.get_listing(listing_id)
            if listing and listing.status.name == "ACTIVE":
                # 趋势分数 = 点击数 * 评分因子
                score = clicks * (0.5 + 0.5 * listing.average_rating / 5.0)
                trending.append((listing, score))
        
        trending.sort(key=lambda x: x[1], reverse=True)
        return trending[:limit]
    
    def get_similar_listings(
        self,
        listing_id: str,
        limit: int = 5
    ) -> List[SearchResult]:
        """
        获取相似的上架条目
        
        Args:
            listing_id: 参考上架条目ID
            limit: 返回数量
            
        Returns:
            相似结果列表
        """
        target = self._listing_manager.get_listing(listing_id)
        if not target:
            return []
        
        # 获取同类型的所有条目
        candidates = self._listing_manager.search_by_type(target.service_type)
        
        results = []
        for listing in candidates:
            if listing.listing_id == listing_id:
                continue
            
            similarity = self._calculate_listing_similarity(target, listing)
            
            if similarity > 0.3:  # 阈值
                results.append(SearchResult(
                    listing=listing,
                    relevance_score=similarity,
                    rank_score=similarity
                ))
        
        results.sort(key=lambda x: x.rank_score, reverse=True)
        return results[:limit]
    
    def compare_listings(
        self,
        listing_ids: List[str]
    ) -> Dict[str, Any]:
        """
        比较多个上架条目
        
        Args:
            listing_ids: 要比较的上架条目ID列表
            
        Returns:
            比较结果字典
        """
        listings = []
        for lid in listing_ids:
            listing = self._listing_manager.get_listing(lid)
            if listing:
                listings.append(listing)
        
        if not listings:
            return {}
        
        comparison = {
            "listings": [l.to_dict() for l in listings],
            "price_comparison": {
                l.listing_id: min(t.base_price for t in l.pricing_tiers) if l.pricing_tiers else None
                for l in listings
            },
            "rating_comparison": {
                l.listing_id: {
                    "average": l.average_rating,
                    "total_reviews": l.total_reviews
                }
                for l in listings
            },
            "capability_overlap": self._calculate_capability_overlap(listings),
            "recommendation": self._generate_comparison_recommendation(listings)
        }
        
        return comparison
    
    def _calculate_capability_overlap(
        self,
        listings: List[ServiceListing]
    ) -> Dict[str, Any]:
        """计算能力重叠"""
        all_caps: Set[str] = set()
        listing_caps: Dict[str, Set[str]] = {}
        
        for l in listings:
            caps = {c.name for c in l.capabilities}
            listing_caps[l.listing_id] = caps
            all_caps.update(caps)
        
        common_caps = all_caps.copy()
        for caps in listing_caps.values():
            common_caps &= caps
        
        unique_caps: Dict[str, List[str]] = {}
        for lid, caps in listing_caps.items():
            unique = caps - common_caps
            for other_caps in listing_caps.values():
                if other_caps is not caps:
                    unique -= other_caps
            unique_caps[lid] = list(unique)
        
        return {
            "common": list(common_caps),
            "unique": unique_caps,
            "total_distinct": len(all_caps)
        }
    
    def _generate_comparison_recommendation(
        self,
        listings: List[ServiceListing]
    ) -> str:
        """生成比较推荐"""
        if not listings:
            return "No listings to compare"
        
        # 找到最佳性价比
        best_value = max(listings, key=lambda l: (
            l.average_rating / (min(t.base_price for t in l.pricing_tiers) + 1)
            if l.pricing_tiers and l.average_rating > 0 else 0
        ))
        
        # 找到最高评分
        best_rated = max(listings, key=lambda l: l.average_rating)
        
        # 找到最便宜
        cheapest = min(listings, key=lambda l: min(t.base_price for t in l.pricing_tiers) if l.pricing_tiers else float('inf'))
        
        if best_value.listing_id == best_rated.listing_id == cheapest.listing_id:
            return f"{best_value.agent_name} offers the best overall value with high ratings and competitive pricing."
        elif best_value.listing_id == best_rated.listing_id:
            return f"{best_value.agent_name} offers the best quality and value. {cheapest.agent_name} is the most budget-friendly option."
        else:
            return f"{best_rated.agent_name} has the highest ratings. {best_value.agent_name} offers the best value. {cheapest.agent_name} is the cheapest."
