"""
聊天记忆系统 - 借鉴视频生成长视频技术
======================================

本模块实现了专门用于多轮对话的记忆系统：

1. ChatMemorySystem: 多轮对话记忆
   - 对话历史管理
   - 对话摘要生成
   - 关键信息提取
   - 情感追踪
   
2. 借鉴视频镜头边界检测实现对话主题切换检测
   - 语义相似度分析
   - 关键词变化检测
   - 时间间隔分析

纯Python实现，仅使用标准库。
"""

from __future__ import annotations

import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque


# ============================================================================
# 工具函数
# ============================================================================

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """计算余弦相似度"""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def normalize_vector(v: List[float]) -> List[float]:
    """L2归一化"""
    norm = math.sqrt(sum(x * x for x in v))
    if norm == 0:
        return v[:]
    return [x / norm for x in v]


def compute_hash(content: str) -> str:
    """计算内容哈希"""
    import hashlib
    return hashlib.md5(content.encode()).hexdigest()[:16]


def extract_keywords(text: str, top_k: int = 5) -> List[str]:
    """提取关键词（简化版）"""
    # 简单的词频统计
    words = re.findall(r'\b[a-zA-Z\u4e00-\u9fff]+\b', text.lower())
    
    # 停用词
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 
                 '的', '了', '在', '是', '和', '有'}
    
    # 统计词频
    word_freq = {}
    for word in words:
        if word not in stopwords and len(word) > 1:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # 返回top-k
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:top_k]]


def analyze_sentiment(text: str) -> Dict[str, float]:
    """简单情感分析"""
    # 正面词汇
    positive_words = ['good', 'great', 'excellent', 'happy', 'love', 'like',
                      '好', '棒', '喜欢', '开心', '优秀']
    # 负面词汇
    negative_words = ['bad', 'terrible', 'hate', 'sad', 'angry', 'dislike',
                      '坏', '差', '讨厌', '难过', '生气']
    
    text_lower = text.lower()
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    total = pos_count + neg_count
    if total == 0:
        return {'positive': 0.5, 'negative': 0.5, 'neutral': 1.0}
    
    return {
        'positive': pos_count / total,
        'negative': neg_count / total,
        'neutral': max(0, 1 - total / len(text.split()))
    }


# ============================================================================
# 数据类定义
# ============================================================================

class MessageRole(Enum):
    """消息角色"""
    USER = auto()
    ASSISTANT = auto()
    SYSTEM = auto()


class TopicTransitionType(Enum):
    """话题转换类型"""
    CONTINUATION = auto()    # 继续
    SMOOTH = auto()          # 平滑过渡
    ABRUPT = auto()          # 突然切换
    RETURN = auto()          # 回到之前话题
    NEW = auto()             # 全新话题


@dataclass
class Message:
    """对话消息"""
    message_id: str
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    
    # 嵌入向量
    embedding: List[float] = field(default_factory=list)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 情感
    sentiment: Dict[str, float] = field(default_factory=dict)
    
    # 关键词
    keywords: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.sentiment:
            self.sentiment = analyze_sentiment(self.content)
        if not self.keywords:
            self.keywords = extract_keywords(self.content)


@dataclass
class DialogueTurn:
    """对话轮次（用户+助手）"""
    turn_id: str
    user_message: Message
    assistant_message: Optional[Message] = None
    timestamp: float = field(default_factory=time.time)
    
    # 话题标签
    topic_tags: List[str] = field(default_factory=list)
    
    # 重要性
    importance: float = 0.5
    
    # 轮次编号
    turn_number: int = 0


@dataclass
class TopicSegment:
    """话题片段"""
    segment_id: str
    start_turn: int
    end_turn: int
    topic_name: str
    messages: List[Message] = field(default_factory=list)
    
    # 摘要
    summary: str = ""
    
    # 关键词集合
    keywords: set = field(default_factory=set)
    
    # 情感轨迹
    sentiment_trajectory: List[Dict[str, float]] = field(default_factory=list)
    
    # 转换类型（与前一片段的关系）
    transition_from_prev: TopicTransitionType = TopicTransitionType.NEW


@dataclass
class ConversationSummary:
    """对话摘要"""
    summary_id: str
    start_time: float
    end_time: float
    
    # 总体摘要
    overall_summary: str = ""
    
    # 关键话题
    key_topics: List[str] = field(default_factory=list)
    
    # 话题片段列表
    topic_segments: List[TopicSegment] = field(default_factory=list)
    
    # 关键信息
    key_facts: Dict[str, Any] = field(default_factory=dict)
    
    # 情感趋势
    sentiment_trend: List[Tuple[float, Dict[str, float]]] = field(default_factory=list)


# ============================================================================
# 1. TopicTransitionDetector - 话题切换检测器
# ============================================================================

class TopicTransitionDetector:
    """
    话题切换检测器
    
    借鉴视频镜头边界检测技术：
    - 帧间差异 -> 消息间语义差异
    - 场景变化 -> 话题切换
    - 直方图比较 -> 关键词分布比较
    
    检测方法：
    1. 语义相似度突变
    2. 关键词集合变化
    3. 时间间隔异常
    4. 情感极性反转
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.4,
        keyword_change_threshold: float = 0.6,
        time_gap_threshold: float = 300,  # 5分钟
        window_size: int = 3
    ):
        self.similarity_threshold = similarity_threshold
        self.keyword_change_threshold = keyword_change_threshold
        self.time_gap_threshold = time_gap_threshold
        self.window_size = window_size
        
        # 检测历史
        self.detection_history: List[Dict] = []
    
    def detect_transition(
        self,
        prev_turns: List[DialogueTurn],
        current_turn: DialogueTurn
    ) -> Tuple[bool, TopicTransitionType, float]:
        """
        检测话题切换
        
        Args:
            prev_turns: 前N轮对话
            current_turn: 当前轮次
            
        Returns:
            (是否切换, 切换类型, 置信度)
        """
        if not prev_turns:
            return True, TopicTransitionType.NEW, 1.0
        
        scores = []
        
        # 1. 语义相似度检测
        semantic_score = self._compute_semantic_similarity(prev_turns, current_turn)
        scores.append(('semantic', 1 - semantic_score))
        
        # 2. 关键词变化检测
        keyword_score = self._compute_keyword_change(prev_turns, current_turn)
        scores.append(('keyword', keyword_score))
        
        # 3. 时间间隔检测
        time_score = self._compute_time_gap(prev_turns[-1], current_turn)
        scores.append(('time', time_score))
        
        # 4. 情感变化检测
        sentiment_score = self._compute_sentiment_change(prev_turns, current_turn)
        scores.append(('sentiment', sentiment_score))
        
        # 综合判断
        avg_score = sum(s[1] for s in scores) / len(scores)
        max_score = max(s[1] for s in scores)
        
        # 判断是否切换
        is_transition = avg_score > self.similarity_threshold or max_score > 0.8
        
        # 分类切换类型
        transition_type = self._classify_transition(scores, is_transition)
        
        # 记录历史
        self.detection_history.append({
            'turn_id': current_turn.turn_id,
            'scores': {k: v for k, v in scores},
            'avg_score': avg_score,
            'is_transition': is_transition,
            'transition_type': transition_type.name
        })
        
        return is_transition, transition_type, avg_score
    
    def _compute_semantic_similarity(
        self,
        prev_turns: List[DialogueTurn],
        current_turn: DialogueTurn
    ) -> float:
        """计算语义相似度"""
        # 获取之前轮次的嵌入
        prev_embeddings = []
        for turn in prev_turns[-self.window_size:]:
            if turn.user_message.embedding:
                prev_embeddings.append(turn.user_message.embedding)
            if turn.assistant_message and turn.assistant_message.embedding:
                prev_embeddings.append(turn.assistant_message.embedding)
        
        if not prev_embeddings or not current_turn.user_message.embedding:
            return 0.5
        
        # 计算平均相似度
        similarities = [
            cosine_similarity(emb, current_turn.user_message.embedding)
            for emb in prev_embeddings
        ]
        
        return sum(similarities) / len(similarities)
    
    def _compute_keyword_change(
        self,
        prev_turns: List[DialogueTurn],
        current_turn: DialogueTurn
    ) -> float:
        """计算关键词变化程度"""
        # 收集之前的关键词
        prev_keywords = set()
        for turn in prev_turns[-self.window_size:]:
            prev_keywords.update(turn.user_message.keywords)
            if turn.assistant_message:
                prev_keywords.update(turn.assistant_message.keywords)
        
        current_keywords = set(current_turn.user_message.keywords)
        
        if not prev_keywords:
            return 0.5 if current_keywords else 0.0
        
        # Jaccard距离
        intersection = len(prev_keywords & current_keywords)
        union = len(prev_keywords | current_keywords)
        
        jaccard = intersection / union if union > 0 else 0
        return 1 - jaccard  # 变化程度 = 1 - 相似度
    
    def _compute_time_gap(
        self,
        prev_turn: DialogueTurn,
        current_turn: DialogueTurn
    ) -> float:
        """计算时间间隔分数"""
        time_diff = current_turn.timestamp - prev_turn.timestamp
        
        if time_diff >= self.time_gap_threshold:
            return 1.0
        elif time_diff >= self.time_gap_threshold / 2:
            return 0.5
        else:
            return time_diff / self.time_gap_threshold * 0.3
    
    def _compute_sentiment_change(
        self,
        prev_turns: List[DialogueTurn],
        current_turn: DialogueTurn
    ) -> float:
        """计算情感变化程度"""
        # 获取之前的情感
        prev_sentiments = []
        for turn in prev_turns[-self.window_size:]:
            prev_sentiments.append(turn.user_message.sentiment)
        
        if not prev_sentiments:
            return 0.0
        
        # 平均情感
        avg_prev_pos = sum(s.get('positive', 0.5) for s in prev_sentiments) / len(prev_sentiments)
        current_pos = current_turn.user_message.sentiment.get('positive', 0.5)
        
        # 情感变化
        sentiment_change = abs(avg_prev_pos - current_pos)
        return sentiment_change
    
    def _classify_transition(
        self,
        scores: List[Tuple[str, float]],
        is_transition: bool
    ) -> TopicTransitionType:
        """分类切换类型"""
        if not is_transition:
            return TopicTransitionType.CONTINUATION
        
        score_dict = dict(scores)
        
        # 语义相似度低但关键词变化小 -> 平滑过渡
        if score_dict.get('semantic', 0) > 0.5 and score_dict.get('keyword', 0) < 0.3:
            return TopicTransitionType.SMOOTH
        
        # 时间间隔大 -> 回到之前话题或新话题
        if score_dict.get('time', 0) > 0.5:
            return TopicTransitionType.RETURN
        
        # 关键词变化大 -> 突然切换
        if score_dict.get('keyword', 0) > 0.7:
            return TopicTransitionType.ABRUPT
        
        return TopicTransitionType.NEW


# ============================================================================
# 2. DialogueSummarizer - 对话摘要器
# ============================================================================

class DialogueSummarizer:
    """
    对话摘要器
    
    生成多层次的对话摘要：
    - 单轮摘要
    - 话题片段摘要
    - 整体对话摘要
    
    借鉴视频摘要技术：
    - 关键帧提取 -> 关键信息提取
    - 场景摘要 -> 话题摘要
    """
    
    def __init__(self, summary_ratio: float = 0.3):
        self.summary_ratio = summary_ratio
        self.summary_history: List[Dict] = []
    
    def summarize_turn(self, turn: DialogueTurn) -> str:
        """摘要单轮对话"""
        user_content = turn.user_message.content
        
        # 简化摘要：提取关键句
        sentences = re.split(r'[.!?。！？]', user_content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 2:
            return user_content[:100]
        
        # 取首尾句（通常包含关键信息）
        key_sentences = [sentences[0]]
        if len(sentences) > 2:
            key_sentences.append(sentences[-1])
        
        return ' '.join(key_sentences)
    
    def summarize_topic_segment(self, segment: TopicSegment) -> str:
        """摘要话题片段"""
        if not segment.messages:
            return ""
        
        # 收集所有内容
        all_content = ' '.join(m.content for m in segment.messages)
        
        # 提取关键句
        sentences = re.split(r'[.!?。！？]', all_content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        # 取前N句作为摘要
        num_summary = max(1, int(len(sentences) * self.summary_ratio))
        summary_sentences = sentences[:num_summary]
        
        summary = ' '.join(summary_sentences)
        
        # 更新片段摘要
        segment.summary = summary
        
        return summary
    
    def summarize_conversation(
        self,
        turns: List[DialogueTurn],
        topic_segments: List[TopicSegment]
    ) -> ConversationSummary:
        """生成对话整体摘要"""
        if not turns:
            return ConversationSummary(
                summary_id=f"conv_{int(time.time())}",
                start_time=time.time(),
                end_time=time.time()
            )
        
        # 时间范围
        start_time = turns[0].timestamp
        end_time = turns[-1].timestamp
        
        # 收集关键话题
        all_keywords = set()
        for turn in turns:
            all_keywords.update(turn.user_message.keywords)
            if turn.assistant_message:
                all_keywords.update(turn.assistant_message.keywords)
        
        # 情感趋势
        sentiment_trend = []
        for turn in turns:
            sentiment_trend.append((turn.timestamp, turn.user_message.sentiment))
        
        # 生成总体摘要
        topic_summaries = [seg.summary for seg in topic_segments if seg.summary]
        overall_summary = ' '.join(topic_summaries[:3]) if topic_summaries else "对话内容"
        
        summary = ConversationSummary(
            summary_id=f"conv_{compute_hash(str(start_time))}",
            start_time=start_time,
            end_time=end_time,
            overall_summary=overall_summary,
            key_topics=list(all_keywords)[:10],
            topic_segments=topic_segments,
            sentiment_trend=sentiment_trend
        )
        
        # 记录历史
        self.summary_history.append({
            'summary_id': summary.summary_id,
            'num_turns': len(turns),
            'num_segments': len(topic_segments),
            'timestamp': time.time()
        })
        
        return summary


# ============================================================================
# 3. KeyInfoExtractor - 关键信息提取器
# ============================================================================

class KeyInfoExtractor:
    """
    关键信息提取器
    
    从对话中提取结构化信息：
    - 实体（人名、地点、组织）
    - 事实和偏好
    - 任务和待办
    - 情感态度
    """
    
    def __init__(self):
        self.extraction_patterns = {
            'preference': [
                r'我喜欢(.+?)[。.，,;；]',
                r'我喜欢(.+?)$',
                r'I like (.+?)[..,;]',
                r'I prefer (.+?)[..,;]',
                r'I love (.+?)[..,;]'
            ],
            'dislike': [
                r'我讨厌(.+?)[。.，,;；]',
                r'我不喜欢(.+?)[。.，,;；]',
                r'I hate (.+?)[..,;]',
                r'I dislike (.+?)[..,;]',
                r'I don\'t like (.+?)[..,;]'
            ],
            'fact': [               r'我是(.+?)[。.，,;；]',
                r'我在(.+?)(?:工作|学习|住)[。.，,;；]',
                r'I am (.+?)[..,;]',
                r'I work at (.+?)[..,;]',
                r'I live in (.+?)[..,;]'
            ],
            'task': [
                r'我需要(.+?)[。.，,;；]',
                r'我要(.+?)[。.，,;；]',
                r'I need to (.+?)[..,;]',
                r'I want to (.+?)[..,;]',
                r'I should (.+?)[..,;]'
            ]
        }
        self.extraction_history: List[Dict] = []
    
    def extract(self, message: Message) -> Dict[str, List[str]]:
        """从消息中提取关键信息"""
        content = message.content
        extracted = {key: [] for key in self.extraction_patterns.keys()}
        
        for info_type, patterns in self.extraction_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                extracted[info_type].extend(matches)
        
        # 记录
        self.extraction_history.append({
            'message_id': message.message_id,
            'extracted': {k: len(v) for k, v in extracted.items()},
            'timestamp': time.time()
        })
        
        return extracted
    
    def extract_from_conversation(
        self,
        turns: List[DialogueTurn]
    ) -> Dict[str, Any]:
        """从整个对话中提取关键信息"""
        all_info = {
            'preferences': [],
            'dislikes': [],
            'facts': [],
            'tasks': []
        }
        
        for turn in turns:
            # 提取用户信息
            user_info = self.extract(turn.user_message)
            all_info['preferences'].extend(user_info.get('preference', []))
            all_info['dislikes'].extend(user_info.get('dislike', []))
            all_info['facts'].extend(user_info.get('fact', []))
            all_info['tasks'].extend(user_info.get('task', []))
        
        # 去重
        for key in all_info:
            all_info[key] = list(set(all_info[key]))
        
        return all_info


# ============================================================================
# 4. EmotionTracker - 情感追踪器
# ============================================================================

class EmotionTracker:
    """
    情感追踪器
    
    追踪对话中的情感变化：
    - 情感极性（正/负/中）
    - 情感强度
    - 情感趋势
    
    借鉴视频情感分析：
    - 时序情感分析
    - 情感曲线
    """
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.emotion_history: List[Dict] = []
    
    def track_message(self, message: Message) -> Dict[str, Any]:
        """追踪单条消息的情感"""
        sentiment = message.sentiment
        
        # 计算情感极性
        if sentiment.get('positive', 0) > sentiment.get('negative', 0):
            polarity = 'positive'
            intensity = sentiment['positive']
        elif sentiment.get('negative', 0) > sentiment.get('positive', 0):
            polarity = 'negative'
            intensity = sentiment['negative']
        else:
            polarity = 'neutral'
            intensity = sentiment.get('neutral', 0.5)
        
        emotion_record = {
            'message_id': message.message_id,
            'timestamp': message.timestamp,
            'polarity': polarity,
            'intensity': intensity,
            'raw_sentiment': sentiment
        }
        
        self.emotion_history.append(emotion_record)
        
        return emotion_record
    
    def get_emotion_trend(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """获取情感趋势"""
        filtered = self.emotion_history
        
        if start_time:
            filtered = [e for e in filtered if e['timestamp'] >= start_time]
        if end_time:
            filtered = [e for e in filtered if e['timestamp'] <= end_time]
        
        return filtered
    
    def compute_emotion_statistics(self) -> Dict[str, Any]:
        """计算情感统计"""
        if not self.emotion_history:
            return {}
        
        polarities = [e['polarity'] for e in self.emotion_history]
        intensities = [e['intensity'] for e in self.emotion_history]
        
        return {
            'total_records': len(self.emotion_history),
            'positive_ratio': polarities.count('positive') / len(polarities),
            'negative_ratio': polarities.count('negative') / len(polarities),
            'neutral_ratio': polarities.count('neutral') / len(polarities),
            'avg_intensity': sum(intensities) / len(intensities),
            'max_intensity': max(intensities),
            'min_intensity': min(intensities)
        }
    
    def detect_emotion_shifts(self) -> List[Dict[str, Any]]:
        """检测情感突变点"""
        shifts = []
        
        for i in range(1, len(self.emotion_history)):
            prev = self.emotion_history[i - 1]
            curr = self.emotion_history[i]
            
            # 极性反转
            if prev['polarity'] != curr['polarity'] and prev['polarity'] != 'neutral':
                shifts.append({
                    'index': i,
                    'from': prev['polarity'],
                    'to': curr['polarity'],
                    'timestamp': curr['timestamp'],
                    'intensity_change': abs(curr['intensity'] - prev['intensity'])
                })
            # 强度突变
            elif abs(curr['intensity'] - prev['intensity']) > 0.5:
                shifts.append({
                    'index': i,
                    'type': 'intensity_shift',
                    'timestamp': curr['timestamp'],
                    'intensity_change': abs(curr['intensity'] - prev['intensity'])
                })
        
        return shifts


# ============================================================================
# 5. ChatMemorySystem - 聊天记忆系统
# ============================================================================

class ChatMemorySystem:
    """
    聊天记忆系统
    
    统一管理多轮对话的记忆：
    1. 对话历史存储
    2. 话题切换检测
    3. 对话摘要生成
    4. 关键信息提取
    5. 情感追踪
    
    借鉴视频处理流程：
    - 分镜（话题分割）
    - 场景检测（话题切换）
    - 摘要生成（视频摘要）
    """
    
    def __init__(
        self,
        max_history_turns: int = 100,
        embedding_fn: Optional[Callable[[str], List[float]]] = None
    ):
        self.max_history_turns = max_history_turns
        self.embedding_fn = embedding_fn
        
        # 组件
        self.transition_detector = TopicTransitionDetector()
        self.summarizer = DialogueSummarizer()
        self.info_extractor = KeyInfoExtractor()
        self.emotion_tracker = EmotionTracker()
        
        # 存储
        self.turns: deque = deque(maxlen=max_history_turns)
        self.topic_segments: List[TopicSegment] = []
        self.current_segment: Optional[TopicSegment] = None
        self.messages: Dict[str, Message] = {}
        
        # 当前状态
        self.current_turn_number = 0
        self.conversation_summary: Optional[ConversationSummary] = None
        
        # 统计
        self.stats = {
            'total_turns': 0,
            'total_segments': 0,
            'topic_transitions': 0
        }
    
    def add_message(
        self,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict] = None
    ) -> Message:
        """
        添加消息
        
        Args:
            role: 消息角色
            content: 消息内容
            metadata: 元数据
            
        Returns:
            创建的消息
        """
        message_id = f"msg_{compute_hash(content)}_{int(time.time() * 1000)}"
        
        # 生成嵌入
        embedding = []
        if self.embedding_fn:
            embedding = self.embedding_fn(content)
        
        message = Message(
            message_id=message_id,
            role=role,
            content=content,
            embedding=embedding,
            metadata=metadata or {}
        )
        
        self.messages[message_id] = message
        
        # 追踪情感
        self.emotion_tracker.track_message(message)
        
        return message
    
    def add_turn(
        self,
        user_content: str,
        assistant_content: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> DialogueTurn:
        """
        添加对话轮次
        
        Args:
            user_content: 用户消息
            assistant_content: 助手消息（可选）
            metadata: 元数据
            
        Returns:
            创建的对话轮次
        """
        self.current_turn_number += 1
        
        # 创建用户消息
        user_message = self.add_message(MessageRole.USER, user_content, metadata)
        
        # 创建助手消息
        assistant_message = None
        if assistant_content:
            assistant_message = self.add_message(
                MessageRole.ASSISTANT, 
                assistant_content, 
                metadata
            )
        
        # 创建轮次
        turn_id = f"turn_{self.current_turn_number}_{int(time.time())}"
        turn = DialogueTurn(
            turn_id=turn_id,
            user_message=user_message,
            assistant_message=assistant_message,
            turn_number=self.current_turn_number,
            topic_tags=user_message.keywords[:3]
        )
        
        # 检测话题切换
        prev_turns = list(self.turns)[-3:] if self.turns else []
        is_transition, transition_type, confidence = self.transition_detector.detect_transition(
            prev_turns, turn
        )
        
        # 处理话题切换
        if is_transition or not self.current_segment:
            self._finalize_current_segment()
            self._start_new_segment(turn, transition_type)
            self.stats['topic_transitions'] += 1
        else:
            # 继续当前话题
            self.current_segment.messages.append(user_message)
            if assistant_message:
                self.current_segment.messages.append(assistant_message)
            self.current_segment.end_turn = turn.turn_number
            self.current_segment.keywords.update(user_message.keywords)
        
        # 存储轮次
        self.turns.append(turn)
        self.stats['total_turns'] += 1
        
        return turn
    
    def _finalize_current_segment(self):
        """结束当前话题片段"""
        if self.current_segment and self.current_segment.messages:
            # 生成摘要
            self.summarizer.summarize_topic_segment(self.current_segment)
            self.topic_segments.append(self.current_segment)
            self.stats['total_segments'] += 1
    
    def _start_new_segment(
        self,
        turn: DialogueTurn,
        transition_type: TopicTransitionType
    ):
        """开始新的话题片段"""
        segment_id = f"seg_{len(self.topic_segments)}_{int(time.time())}"
        
        messages = [turn.user_message]
        if turn.assistant_message:
            messages.append(turn.assistant_message)
        
        self.current_segment = TopicSegment(
            segment_id=segment_id,
            start_turn=turn.turn_number,
            end_turn=turn.turn_number,
            topic_name=turn.topic_tags[0] if turn.topic_tags else "general",
            messages=messages,
            keywords=set(turn.user_message.keywords),
            transition_from_prev=transition_type
        )
    
    def get_recent_context(
        self,
        n_turns: int = 5,
        include_summary: bool = True
    ) -> str:
        """
        获取最近上下文
        
        Args:
            n_turns: 最近轮次数
            include_summary: 是否包含摘要
            
        Returns:
            格式化的上下文
        """
        recent_turns = list(self.turns)[-n_turns:]
        
        context_parts = []
        
        # 添加当前话题摘要
        if include_summary and self.current_segment and self.current_segment.summary:
            context_parts.append(f"[当前话题: {self.current_segment.topic_name}]")
            context_parts.append(self.current_segment.summary)
        
        # 添加最近对话
        for turn in recent_turns:
            context_parts.append(f"用户: {turn.user_message.content}")
            if turn.assistant_message:
                context_parts.append(f"助手: {turn.assistant_message.content}")
        
        return "\n".join(context_parts)
    
    def retrieve_relevant_history(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 3
    ) -> List[DialogueTurn]:
        """
        检索相关历史
        
        Args:
            query: 查询文本
            query_embedding: 查询嵌入
            top_k: 返回数量
            
        Returns:
            相关对话轮次
        """
        if not self.embedding_fn or not query_embedding:
            # 简单关键词匹配
            query_keywords = set(extract_keywords(query))
            scored = []
            for turn in self.turns:
                turn_keywords = set(turn.user_message.keywords)
                if turn.assistant_message:
                    turn_keywords.update(turn.assistant_message.keywords)
                overlap = len(query_keywords & turn_keywords)
                scored.append((turn, overlap))
        else:
            # 嵌入相似度
            scored = []
            for turn in self.turns:
                sim = cosine_similarity(query_embedding, turn.user_message.embedding)
                scored.append((turn, sim))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [turn for turn, _ in scored[:top_k]]
    
    def generate_summary(self) -> ConversationSummary:
        """生成对话摘要"""
        self._finalize_current_segment()
        
        all_turns = list(self.turns)
        self.conversation_summary = self.summarizer.summarize_conversation(
            all_turns, self.topic_segments
        )
        
        return self.conversation_summary
    
    def extract_key_info(self) -> Dict[str, Any]:
        """提取关键信息"""
        return self.info_extractor.extract_from_conversation(list(self.turns))
    
    def get_emotion_report(self) -> Dict[str, Any]:
        """获取情感报告"""
        return {
            'statistics': self.emotion_tracker.compute_emotion_statistics(),
            'shifts': self.emotion_tracker.detect_emotion_shifts(),
            'trend': self.emotion_tracker.get_emotion_trend()
        }
    
    def get_topic_segments(self) -> List[TopicSegment]:
        """获取所有话题片段"""
        segments = self.topic_segments.copy()
        if self.current_segment:
            segments.append(self.current_segment)
        return segments
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'current_segment_messages': len(self.current_segment.messages) if self.current_segment else 0,
            'emotion_records': len(self.emotion_tracker.emotion_history),
            'transition_detections': len(self.transition_detector.detection_history)
        }
    
    def clear(self):
        """清空所有数据"""
        self.turns.clear()
        self.topic_segments.clear()
        self.current_segment = None
        self.messages.clear()
        self.current_turn_number = 0
        self.conversation_summary = None
        self.stats = {k: 0 for k in self.stats}
        self.emotion_tracker.emotion_history.clear()
        self.transition_detector.detection_history.clear()
