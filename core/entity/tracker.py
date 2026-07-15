"""
Entity Tracker - 实体追踪器
跟踪对话中的实体信息，支持实体识别、消解和关系管理
"""

import re
import time
import logging
import threading
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """实体类型"""
    PERSON = "person"           # 人物
    ORGANIZATION = "organization"  # 组织
    LOCATION = "location"       # 地点
    DATE = "date"               # 日期
    TIME = "time"               # 时间
    NUMBER = "number"           # 数字
    MONEY = "money"             # 金额
    PRODUCT = "product"         # 产品
    EVENT = "event"             # 事件
    CONCEPT = "concept"         # 概念
    URL = "url"                 # 网址
    EMAIL = "email"             # 邮箱
    PHONE = "phone"             # 电话
    CUSTOM = "custom"           # 自定义


class ReferenceType(Enum):
    """引用类型"""
    EXPLICIT = "explicit"       # 显式引用（名字）
    PRONOUN = "pronoun"         # 代词引用
    DEMONSTRATIVE = "demonstrative"  # 指示代词
    DEFINITE = "definite"       # 定指描述
    INDEFINITE = "indefinite"   # 不定指描述


@dataclass
class EntityMention:
    """实体提及"""
    text: str
    start_pos: int
    end_pos: int
    mention_type: ReferenceType
    confidence: float
    sentence_id: int
    turn_id: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Entity:
    """实体"""
    id: str
    name: str
    entity_type: EntityType
    canonical_name: str
    aliases: Set[str] = field(default_factory=set)
    mentions: List[EntityMention] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    relations: Dict[str, List[str]] = field(default_factory=dict)  # relation_type -> [entity_ids]
    first_mentioned_turn: int = 0
    last_mentioned_turn: int = 0
    mention_count: int = 0
    salience: float = 0.5  # 显著性
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def add_mention(self, mention: EntityMention):
        """添加提及"""
        self.mentions.append(mention)
        self.mention_count += 1
        self.last_mentioned_turn = mention.turn_id
        self.updated_at = time.time()
        
        # 更新显著性
        self._update_salience()
    
    def _update_salience(self):
        """更新显著性"""
        # 基于提及次数和最近性
        recency_factor = 1.0 / (1 + self.last_mentioned_turn * 0.1)
        frequency_factor = min(1.0, self.mention_count * 0.2)
        self.salience = 0.5 * recency_factor + 0.5 * frequency_factor
    
    def add_alias(self, alias: str):
        """添加别名"""
        self.aliases.add(alias.lower())
    
    def add_relation(self, relation_type: str, entity_id: str):
        """添加关系"""
        if relation_type not in self.relations:
            self.relations[relation_type] = []
        if entity_id not in self.relations[relation_type]:
            self.relations[relation_type].append(entity_id)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "canonical_name": self.canonical_name,
            "aliases": list(self.aliases),
            "mention_count": self.mention_count,
            "salience": self.salience,
            "attributes": self.attributes,
            "relations": self.relations,
            "first_mentioned_turn": self.first_mentioned_turn,
            "last_mentioned_turn": self.last_mentioned_turn
        }


@dataclass
class EntityConfig:
    """实体追踪配置"""
    max_entities: int = 1000
    max_mentions_per_entity: int = 50
    salience_threshold: float = 0.1
    enable_coreference: bool = True
    enable_relation_extraction: bool = True
    custom_entity_patterns: Dict[str, str] = field(default_factory=dict)


class EntityRecognizer:
    """实体识别器"""
    
    def __init__(self, custom_patterns: Optional[Dict[str, str]] = None):
        self.patterns = self._init_patterns()
        if custom_patterns:
            for entity_type, pattern in custom_patterns.items():
                self.patterns[entity_type] = re.compile(pattern, re.IGNORECASE)
        
        self.pronouns = {
            '他', '她', '它', '他们', '她们', '它们',
            'he', 'she', 'it', 'they', 'him', 'her', 'them',
            'this', 'that', 'these', 'those', '这', '那'
        }
    
    def _init_patterns(self) -> Dict[EntityType, re.Pattern]:
        """初始化识别模式"""
        return {
            EntityType.URL: re.compile(
                r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-a-zA-Z0-9()@:%_\+.~#?&//=]*',
                re.IGNORECASE
            ),
            EntityType.EMAIL: re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                re.IGNORECASE
            ),
            EntityType.PHONE: re.compile(
                r'(?:\+?86)?[-\s]?1[3-9]\d{9}|(?:\+?\d{1,3}[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}',
                re.IGNORECASE
            ),
            EntityType.DATE: re.compile(
                r'\b(?:19|20)\d{2}[-/年](?:0?[1-9]|1[0-2])[-/月](?:0?[1-9]|[12]\d|3[01])[日]?\b'
                r'|\b(?:今天|昨天|明天|前天|后天|上周|下周|本月|上月|下月)\b',
                re.IGNORECASE
            ),
            EntityType.TIME: re.compile(
                r'\b(?:[01]?\d|2[0-3])[:点时][0-5]\d(?:分)?(?:秒)?(?:\s*[APap][Mm])?\b'
                r'|\b(?:早上|上午|中午|下午|晚上|凌晨)\b',
                re.IGNORECASE
            ),
            EntityType.MONEY: re.compile(
                r'(?:\$|￥|€|£|USD|CNY|EUR)\s*[\d,]+(?:\.\d{2})?'
                r'|[\d,]+(?:\.\d{2})?\s*(?:美元|元|人民币|欧元|英镑)',
                re.IGNORECASE
            ),
            EntityType.NUMBER: re.compile(
                r'\b\d+(?:\.\d+)?(?:%)?\b'
            ),
            EntityType.PERSON: re.compile(
                r'(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+[A-Z][a-z]+'
                r'|\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
            ),
            EntityType.ORGANIZATION: re.compile(
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Inc|Corp|Ltd|LLC|Company|Corp|集团|公司|科技|银行))\.?\b'
            ),
            EntityType.LOCATION: re.compile(
                r'\b(?:北京|上海|广州|深圳|杭州|南京|成都|武汉|西安|重庆|天津|苏州|New York|London|Paris|Tokyo)\b'
            ),
        }
    
    def recognize(self, text: str) -> List[Tuple[EntityType, str, int, int]]:
        """
        识别文本中的实体
        
        Returns:
            [(entity_type, text, start, end), ...]
        """
        entities = []
        
        for entity_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                entities.append((
                    entity_type,
                    match.group(),
                    match.start(),
                    match.end()
                ))
        
        # 按位置排序
        entities.sort(key=lambda x: x[2])
        
        # 去除重叠
        entities = self._remove_overlaps(entities)
        
        return entities
    
    def _remove_overlaps(
        self, 
        entities: List[Tuple[EntityType, str, int, int]]
    ) -> List[Tuple[EntityType, str, int, int]]:
        """去除重叠实体"""
        if not entities:
            return []
        
        result = [entities[0]]
        for entity in entities[1:]:
            # 检查是否与最后一个重叠
            last = result[-1]
            if entity[2] >= last[3]:  # 不重叠
                result.append(entity)
            elif len(entity[1]) > len(last[1]):  # 选择更长的
                result[-1] = entity
        
        return result
    
    def is_pronoun(self, text: str) -> bool:
        """检查是否为代词"""
        return text.lower() in self.pronouns


class CoreferenceResolver:
    """共指消解器"""
    
    def __init__(self):
        self.gender_pronouns = {
            'male': {'他', 'he', 'him', 'his', 'himself'},
            'female': {'她', 'her', 'hers', 'herself'},
            'neutral': {'它', 'it', 'its', 'itself', 'they', 'them', 'their'}
        }
        
        self.number_pronouns = {
            'singular': {'他', '她', '它', 'he', 'she', 'it', 'this', 'that'},
            'plural': {'他们', '她们', '它们', 'they', 'them', 'these', 'those'}
        }
    
    def resolve(
        self,
        mention_text: str,
        candidates: List[Entity],
        context: str
    ) -> Optional[Entity]:
        """
        消解代词引用
        
        Args:
            mention_text: 提及文本（可能是代词）
            candidates: 候选实体列表
            context: 上下文
        
        Returns:
            最可能的实体
        """
        if not candidates:
            return None
        
        mention_lower = mention_text.lower()
        
        # 检查性别/数量约束
        gender = self._get_gender(mention_lower)
        number = self._get_number(mention_lower)
        
        # 过滤候选
        filtered = []
        for entity in candidates:
            # 检查性别兼容性
            if gender and entity.attributes.get('gender') not in [gender, None]:
                continue
            
            # 检查数量兼容性
            if number == 'singular' and entity.attributes.get('is_plural'):
                continue
            
            filtered.append(entity)
        
        if not filtered:
            filtered = candidates
        
        # 按显著性和最近性排序
        filtered.sort(key=lambda e: (e.salience, e.last_mentioned_turn), reverse=True)
        
        return filtered[0] if filtered else None
    
    def _get_gender(self, pronoun: str) -> Optional[str]:
        """获取代词性别"""
        for gender, pronouns in self.gender_pronouns.items():
            if pronoun in pronouns:
                return gender
        return None
    
    def _get_number(self, pronoun: str) -> Optional[str]:
        """获取代词数量"""
        for number, pronouns in self.number_pronouns.items():
            if pronoun in pronouns:
                return number
        return None


class RelationExtractor:
    """关系提取器"""
    
    def __init__(self):
        self.relation_patterns = self._init_patterns()
    
    def _init_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """初始化关系模式"""
        return [
            # 中文关系
            (re.compile(r'(.+?)是(.+?)的(.+?)'), 'attribute'),
            (re.compile(r'(.+?)属于(.+?)'), 'belongs_to'),
            (re.compile(r'(.+?)位于(.+?)'), 'located_at'),
            (re.compile(r'(.+?)创建于(.+?)'), 'created_at'),
            (re.compile(r'(.+?)的(.+?)是(.+?)'), 'has_attribute'),
            (re.compile(r'(.+?)和(.+?)是(.+?)关系'), 'relation'),
            
            # 英文关系
            (re.compile(r'(.+?)\s+is\s+the\s+(.+?)\s+of\s+(.+)'), 'attribute'),
            (re.compile(r'(.+?)\s+belongs\s+to\s+(.+)'), 'belongs_to'),
            (re.compile(r'(.+?)\s+is\s+located\s+in\s+(.+)'), 'located_at'),
            (re.compile(r'(.+?)\s+was\s+created\s+by\s+(.+)'), 'created_by'),
        ]
    
    def extract(
        self, 
        text: str, 
        entities: List[Entity]
    ) -> List[Tuple[str, str, str]]:
        """
        提取实体关系
        
        Returns:
            [(entity1_id, relation_type, entity2_id), ...]
        """
        relations = []
        entity_map = {e.name.lower(): e for e in entities}
        
        for pattern, relation_type in self.relation_patterns:
            for match in pattern.finditer(text):
                groups = match.groups()
                
                # 尝试匹配实体
                matched_entities = []
                for group in groups:
                    group_lower = group.strip().lower()
                    if group_lower in entity_map:
                        matched_entities.append(entity_map[group_lower])
                
                if len(matched_entities) >= 2:
                    relations.append((
                        matched_entities[0].id,
                        relation_type,
                        matched_entities[1].id
                    ))
        
        return relations


class EntityTracker:
    """实体追踪器主类"""
    
    def __init__(
        self,
        config: Optional[EntityConfig] = None,
        ner_model: Optional[Any] = None
    ):
        self.config = config or EntityConfig()
        self.ner_model = ner_model
        
        self.recognizer = EntityRecognizer(self.config.custom_entity_patterns)
        self.coref_resolver = CoreferenceResolver()
        self.relation_extractor = RelationExtractor()
        
        # 存储
        self._entities: Dict[str, Entity] = {}
        self._name_to_ids: Dict[str, Set[str]] = defaultdict(set)
        self._turn_entities: Dict[int, List[str]] = defaultdict(list)
        
        self._entity_id_counter = 0
        self._current_turn = 0
        self._lock = threading.Lock()
    
    def _generate_entity_id(self) -> str:
        """生成实体ID"""
        self._entity_id_counter += 1
        return f"entity_{self._entity_id_counter}"
    
    def process_turn(
        self, 
        text: str, 
        turn_id: Optional[int] = None
    ) -> List[Entity]:
        """
        处理一个对话轮次
        
        Args:
            text: 输入文本
            turn_id: 轮次ID
        
        Returns:
            识别到的实体列表
        """
        turn_id = turn_id or self._current_turn
        self._current_turn = turn_id + 1
        
        # 识别实体
        recognized = self.recognizer.recognize(text)
        
        # 如果有NER模型，使用它进行更精确的识别
        if self.ner_model:
            try:
                model_entities = self._recognize_with_model(text)
                recognized = self._merge_recognitions(recognized, model_entities)
            except Exception as e:
                logger.warning(f"模型识别失败: {e}")
        
        # 创建或更新实体
        entities = []
        for entity_type, entity_text, start, end in recognized:
            entity = self._process_entity_mention(
                entity_text, 
                entity_type, 
                start, 
                end, 
                turn_id
            )
            if entity:
                entities.append(entity)
        
        # 共指消解
        if self.config.enable_coreference:
            self._resolve_coreferences(text, turn_id)
        
        # 关系提取
        if self.config.enable_relation_extraction:
            self._extract_relations(text, entities)
        
        # 记录本轮实体
        with self._lock:
            self._turn_entities[turn_id] = [e.id for e in entities]
        
        return entities
    
    def _recognize_with_model(self, text: str) -> List[Tuple[EntityType, str, int, int]]:
        """使用模型识别实体"""
        # 占位实现，实际使用时调用具体模型
        return []
    
    def _merge_recognitions(
        self,
        rule_based: List[Tuple[EntityType, str, int, int]],
        model_based: List[Tuple[EntityType, str, int, int]]
    ) -> List[Tuple[EntityType, str, int, int]]:
        """合并识别结果"""
        # 简单合并，去除重叠
        all_entities = rule_based + model_based
        all_entities.sort(key=lambda x: x[2])
        return self.recognizer._remove_overlaps(all_entities)
    
    def _process_entity_mention(
        self,
        entity_text: str,
        entity_type: EntityType,
        start: int,
        end: int,
        turn_id: int
    ) -> Optional[Entity]:
        """处理实体提及"""
        entity_text_lower = entity_text.lower()
        
        # 检查是否为代词
        if self.recognizer.is_pronoun(entity_text):
            # 共指消解
            candidates = self._get_recent_entities(turn_id)
            resolved = self.coref_resolver.resolve(entity_text, candidates, "")
            
            if resolved:
                # 添加提及
                mention = EntityMention(
                    text=entity_text,
                    start_pos=start,
                    end_pos=end,
                    mention_type=ReferenceType.PRONOUN,
                    confidence=0.8,
                    sentence_id=0,
                    turn_id=turn_id
                )
                resolved.add_mention(mention)
                return resolved
            return None
        
        # 检查是否已存在
        with self._lock:
            existing_ids = self._name_to_ids.get(entity_text_lower, set())
            
            if existing_ids:
                # 更新现有实体
                entity_id = list(existing_ids)[0]
                entity = self._entities[entity_id]
                
                mention = EntityMention(
                    text=entity_text,
                    start_pos=start,
                    end_pos=end,
                    mention_type=ReferenceType.EXPLICIT,
                    confidence=1.0,
                    sentence_id=0,
                    turn_id=turn_id
                )
                entity.add_mention(mention)
                return entity
            
            # 创建新实体
            entity = Entity(
                id=self._generate_entity_id(),
                name=entity_text,
                entity_type=entity_type,
                canonical_name=entity_text,
                first_mentioned_turn=turn_id
            )
            
            mention = EntityMention(
                text=entity_text,
                start_pos=start,
                end_pos=end,
                mention_type=ReferenceType.EXPLICIT,
                confidence=1.0,
                sentence_id=0,
                turn_id=turn_id
            )
            entity.add_mention(mention)
            
            # 存储
            self._entities[entity.id] = entity
            self._name_to_ids[entity_text_lower].add(entity.id)
            
            return entity
    
    def _get_recent_entities(self, current_turn: int, window: int = 5) -> List[Entity]:
        """获取最近的实体"""
        recent = []
        for turn_id in range(max(0, current_turn - window), current_turn):
            for entity_id in self._turn_entities.get(turn_id, []):
                if entity_id in self._entities:
                    recent.append(self._entities[entity_id])
        
        # 按显著性排序
        recent.sort(key=lambda e: e.salience, reverse=True)
        return recent
    
    def _resolve_coreferences(self, text: str, turn_id: int):
        """消解共指"""
        # 查找代词
        for pronoun in self.recognizer.pronouns:
            pattern = re.compile(r'\b' + pronoun + r'\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                candidates = self._get_recent_entities(turn_id)
                resolved = self.coref_resolver.resolve(pronoun, candidates, text)
                
                if resolved:
                    mention = EntityMention(
                        text=pronoun,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        mention_type=ReferenceType.PRONOUN,
                        confidence=0.8,
                        sentence_id=0,
                        turn_id=turn_id
                    )
                    resolved.add_mention(mention)
    
    def _extract_relations(self, text: str, entities: List[Entity]):
        """提取关系"""
        relations = self.relation_extractor.extract(text, entities)
        
        for entity1_id, relation_type, entity2_id in relations:
            if entity1_id in self._entities:
                self._entities[entity1_id].add_relation(relation_type, entity2_id)
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """获取实体"""
        return self._entities.get(entity_id)
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """按类型获取实体"""
        return [e for e in self._entities.values() if e.entity_type == entity_type]
    
    def get_entities_by_turn(self, turn_id: int) -> List[Entity]:
        """获取指定轮次的实体"""
        entity_ids = self._turn_entities.get(turn_id, [])
        return [self._entities[eid] for eid in entity_ids if eid in self._entities]
    
    def get_salient_entities(self, threshold: Optional[float] = None) -> List[Entity]:
        """获取显著实体"""
        threshold = threshold or self.config.salience_threshold
        return [e for e in self._entities.values() if e.salience >= threshold]
    
    def get_entity_relations(
        self, 
        entity_id: str
    ) -> Dict[str, List[Entity]]:
        """获取实体的所有关系"""
        entity = self._entities.get(entity_id)
        if not entity:
            return {}
        
        relations = {}
        for relation_type, related_ids in entity.relations.items():
            relations[relation_type] = [
                self._entities[rid] 
                for rid in related_ids 
                if rid in self._entities
            ]
        
        return relations
    
    def merge_entities(self, entity_id1: str, entity_id2: str) -> bool:
        """合并两个实体"""
        if entity_id1 not in self._entities or entity_id2 not in self._entities:
            return False
        
        entity1 = self._entities[entity_id1]
        entity2 = self._entities[entity_id2]
        
        # 合并别名
        entity1.aliases.update(entity2.aliases)
        entity1.add_alias(entity2.name)
        
        # 合并提及
        entity1.mentions.extend(entity2.mentions)
        entity1.mention_count += entity2.mention_count
        
        # 合并属性
        entity1.attributes.update(entity2.attributes)
        
        # 合并关系
        for relation_type, related_ids in entity2.relations.items():
            for rid in related_ids:
                entity1.add_relation(relation_type, rid)
        
        # 更新索引
        for alias in entity2.aliases:
            self._name_to_ids[alias].discard(entity_id2)
            self._name_to_ids[alias].add(entity_id1)
        
        # 删除entity2
        del self._entities[entity_id2]
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        type_counts = defaultdict(int)
        for entity in self._entities.values():
            type_counts[entity.entity_type.value] += 1
        
        return {
            "total_entities": len(self._entities),
            "total_turns": len(self._turn_entities),
            "entity_types": dict(type_counts),
            "avg_mentions": sum(e.mention_count for e in self._entities.values()) / max(len(self._entities), 1),
            "avg_salience": sum(e.salience for e in self._entities.values()) / max(len(self._entities), 1)
        }
    
    def clear(self):
        """清空所有数据"""
        with self._lock:
            self._entities.clear()
            self._name_to_ids.clear()
            self._turn_entities.clear()
            self._entity_id_counter = 0
            self._current_turn = 0


# 工厂函数
def create_entity_tracker(
    config: Optional[EntityConfig] = None,
    ner_model: Optional[Any] = None
) -> EntityTracker:
    """创建实体追踪器"""
    return EntityTracker(config, ner_model)
