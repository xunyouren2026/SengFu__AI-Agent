"""
Intent Classifier - 意图分类器
识别用户意图，支持多标签分类和槽位填充
"""

import re
import time
import logging
import threading
from typing import List, Dict, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class IntentCategory(Enum):
    """意图类别"""
    QUERY = "query"               # 查询
    COMMAND = "command"           # 命令
    QUESTION = "question"         # 提问
    CHITCHAT = "chitchat"         # 闲聊
    COMPLAINT = "complaint"       # 投诉
    FEEDBACK = "feedback"         # 反馈
    REQUEST = "request"           # 请求
    CONFIRMATION = "confirmation" # 确认
    GREETING = "greeting"         # 问候
    FAREWELL = "farewell"         # 告别
    UNKNOWN = "unknown"           # 未知


class ConfidenceLevel(Enum):
    """置信度级别"""
    HIGH = "high"       # > 0.9
    MEDIUM = "medium"   # 0.7 - 0.9
    LOW = "low"         # 0.5 - 0.7
    UNCERTAIN = "uncertain"  # < 0.5


@dataclass
class Slot:
    """槽位"""
    name: str
    value: Any
    slot_type: str
    confidence: float = 1.0
    required: bool = False
    normalized_value: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Intent:
    """意图"""
    name: str
    category: IntentCategory
    confidence: float
    slots: Dict[str, Slot] = field(default_factory=dict)
    description: str = ""
    examples: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_slot(self, name: str) -> Optional[Slot]:
        return self.slots.get(name)
    
    def get_slot_value(self, name: str, default: Any = None) -> Any:
        slot = self.slots.get(name)
        return slot.value if slot else default
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "confidence": self.confidence,
            "slots": {k: {"value": v.value, "type": v.slot_type} for k, v in self.slots.items()},
            "description": self.description
        }


@dataclass
class IntentResult:
    """意图识别结果"""
    text: str
    intents: List[Intent]
    primary_intent: Optional[Intent] = None
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.intents and not self.primary_intent:
            self.primary_intent = self.intents[0]
        
        if self.primary_intent:
            if self.primary_intent.confidence > 0.9:
                self.confidence_level = ConfidenceLevel.HIGH
            elif self.primary_intent.confidence > 0.7:
                self.confidence_level = ConfidenceLevel.MEDIUM
            elif self.primary_intent.confidence > 0.5:
                self.confidence_level = ConfidenceLevel.LOW
            else:
                self.confidence_level = ConfidenceLevel.UNCERTAIN
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "primary_intent": self.primary_intent.to_dict() if self.primary_intent else None,
            "all_intents": [i.to_dict() for i in self.intents],
            "confidence_level": self.confidence_level.value,
            "processing_time": self.processing_time
        }


@dataclass
class IntentConfig:
    """意图分类配置"""
    max_intents: int = 5
    confidence_threshold: float = 0.5
    enable_slot_filling: bool = True
    enable_multi_label: bool = True
    custom_intents: Dict[str, Any] = field(default_factory=dict)
    language: str = "zh"


class IntentPatternMatcher:
    """意图模式匹配器"""
    
    def __init__(self, language: str = "zh"):
        self.language = language
        self.patterns = self._init_patterns()
    
    def _init_patterns(self) -> Dict[str, List[Tuple[re.Pattern, float]]]:
        """初始化意图模式"""
        patterns = {
            # 问候意图
            "greeting": [
                (re.compile(r'^(你好|您好|hi|hello|hey|早上好|晚上好)', re.IGNORECASE), 0.95),
                (re.compile(r'(在吗|有人吗|请问)', re.IGNORECASE), 0.8),
            ],
            
            # 告别意图
            "farewell": [
                (re.compile(r'(再见|拜拜|bye|goodbye|下次见|走了)', re.IGNORECASE), 0.95),
                (re.compile(r'(晚安|早点休息)', re.IGNORECASE), 0.85),
            ],
            
            # 查询意图
            "query_info": [
                (re.compile(r'(查询|查一下|帮我查|请问.*是|什么是|怎么查)', re.IGNORECASE), 0.85),
                (re.compile(r'(.*多少钱|.*价格|.*费用|.*收费)', re.IGNORECASE), 0.8),
                (re.compile(r'(.*在哪|.*位置|.*地址)', re.IGNORECASE), 0.8),
            ],
            
            # 命令意图
            "command": [
                (re.compile(r'(帮我|请帮我|麻烦帮我|帮我做)', re.IGNORECASE), 0.7),
                (re.compile(r'(打开|关闭|启动|停止|设置|修改|删除|添加)', re.IGNORECASE), 0.85),
                (re.compile(r'(播放|暂停|继续|停止)', re.IGNORECASE), 0.9),
            ],
            
            # 提问意图
            "question": [
                (re.compile(r'^(什么|为什么|怎么|如何|哪|谁|多少|几)', re.IGNORECASE), 0.9),
                (re.compile(r'\?|？', re.IGNORECASE), 0.6),
                (re.compile(r'(吗|呢|吧)\s*$', re.IGNORECASE), 0.7),
            ],
            
            # 闲聊意图
            "chitchat": [
                (re.compile(r'(天气|今天|明天|周末)', re.IGNORECASE), 0.6),
                (re.compile(r'(笑话|故事|有趣)', re.IGNORECASE), 0.8),
                (re.compile(r'(无聊|陪我聊聊|聊聊天)', re.IGNORECASE), 0.9),
            ],
            
            # 投诉意图
            "complaint": [
                (re.compile(r'(投诉|举报|不满|差评|太差了|很失望)', re.IGNORECASE), 0.95),
                (re.compile(r'(问题|故障|错误|bug|不能.*用)', re.IGNORECASE), 0.7),
            ],
            
            # 反馈意图
            "feedback": [
                (re.compile(r'(建议|意见|反馈|改进|希望)', re.IGNORECASE), 0.85),
                (re.compile(r'(很好|不错|喜欢|满意|太棒了)', re.IGNORECASE), 0.8),
            ],
            
            # 确认意图
            "confirmation": [
                (re.compile(r'^(是的|对|没错|好的|ok|可以|确认)', re.IGNORECASE), 0.95),
                (re.compile(r'^(不是|不对|没有|取消|拒绝)', re.IGNORECASE), 0.95),
            ],
            
            # 请求意图
            "request": [
                (re.compile(r'(请|麻烦|劳驾|能不能|可以.*吗)', re.IGNORECASE), 0.8),
                (re.compile(r'(需要|想要|希望)', re.IGNORECASE), 0.7),
            ],
        }
        
        return patterns
    
    def match(self, text: str) -> List[Tuple[str, float]]:
        """
        匹配意图
        
        Returns:
            [(intent_name, confidence), ...]
        """
        results = []
        
        for intent_name, pattern_list in self.patterns.items():
            max_confidence = 0.0
            for pattern, base_confidence in pattern_list:
                if pattern.search(text):
                    max_confidence = max(max_confidence, base_confidence)
            
            if max_confidence > 0:
                results.append((intent_name, max_confidence))
        
        # 按置信度排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results


class SlotFiller:
    """槽位填充器"""
    
    def __init__(self):
        self.slot_patterns = self._init_slot_patterns()
    
    def _init_slot_patterns(self) -> Dict[str, Dict[str, Any]]:
        """初始化槽位模式"""
        return {
            # 时间槽位
            "time": {
                "patterns": [
                    re.compile(r'(\d{1,2}[:点时]\d{1,2}(?:分)?)'),
                    re.compile(r'(早上|上午|中午|下午|晚上|凌晨)'),
                    re.compile(r'(今天|明天|后天|大后天)'),
                ],
                "type": "time"
            },
            
            # 日期槽位
            "date": {
                "patterns": [
                    re.compile(r'(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?)'),
                    re.compile(r'(今天|明天|后天|昨天|前天)'),
                    re.compile(r'(周[一二三四五六日]|星期[一二三四五六日])'),
                ],
                "type": "date"
            },
            
            # 地点槽位
            "location": {
                "patterns": [
                    re.compile(r'(北京|上海|广州|深圳|杭州|南京|成都|武汉|西安)'),
                    re.compile(r'在(.{2,10}?)(?=的|里|中|上|下|做|吃|买|去)'),
                ],
                "type": "location"
            },
            
            # 数量槽位
            "quantity": {
                "patterns": [
                    re.compile(r'(\d+(?:\.\d+)?)\s*(个|件|本|张|次|天|小时|分钟)'),
                    re.compile(r'(一|二|三|四|五|六|七|八|九|十)\s*(个|件|本|张)'),
                ],
                "type": "number"
            },
            
            # 金额槽位
            "money": {
                "patterns": [
                    re.compile(r'(\$|￥|€)?\s*(\d+(?:\.\d{2})?)\s*(美元|元|块|人民币|欧元)?'),
                ],
                "type": "money"
            },
            
            # 人物槽位
            "person": {
                "patterns": [
                    re.compile(r'(?:叫|是|找)(.{2,4}?)(?:的|先生|女士|老师|医生)'),
                ],
                "type": "person"
            },
            
            # 电话槽位
            "phone": {
                "patterns": [
                    re.compile(r'(1[3-9]\d{9})'),
                    re.compile(r'(\d{3,4}[-\s]?\d{7,8})'),
                ],
                "type": "phone"
            },
            
            # 产品槽位
            "product": {
                "patterns": [
                    re.compile(r'(?:买|订购|要)(.{2,20}?)(?:多少钱|价格|费用)'),
                ],
                "type": "product"
            },
        }
    
    def fill_slots(
        self, 
        text: str, 
        intent_name: str
    ) -> Dict[str, Slot]:
        """
        填充槽位
        
        Args:
            text: 输入文本
            intent_name: 意图名称
        
        Returns:
            {slot_name: Slot, ...}
        """
        slots = {}
        
        # 获取该意图需要的槽位
        required_slots = self._get_required_slots(intent_name)
        
        # 提取所有可能的槽位
        for slot_name, slot_config in self.slot_patterns.items():
            for pattern in slot_config["patterns"]:
                match = pattern.search(text)
                if match:
                    value = match.group(1) if match.groups() else match.group()
                    slots[slot_name] = Slot(
                        name=slot_name,
                        value=value,
                        slot_type=slot_config["type"],
                        confidence=0.9,
                        required=slot_name in required_slots,
                        normalized_value=self._normalize_value(value, slot_config["type"])
                    )
                    break
        
        return slots
    
    def _get_required_slots(self, intent_name: str) -> Set[str]:
        """获取意图需要的槽位"""
        intent_slots = {
            "query_info": {"query", "target"},
            "command": {"action", "target"},
            "request": {"item", "quantity"},
            "booking": {"date", "time", "location"},
        }
        return intent_slots.get(intent_name, set())
    
    def _normalize_value(self, value: str, slot_type: str) -> str:
        """标准化值"""
        if slot_type == "time":
            # 标准化时间格式
            value = re.sub(r'[:点时]', ':', value)
            value = re.sub(r'分$', '', value)
        elif slot_type == "date":
            # 标准化日期格式
            value = re.sub(r'[年月]', '-', value)
            value = re.sub(r'日$', '', value)
        elif slot_type == "money":
            # 标准化金额
            value = re.sub(r'[美元元块人民币欧元]', '', value)
        
        return value.strip()


class IntentClassifier:
    """意图分类器主类"""
    
    def __init__(
        self,
        config: Optional[IntentConfig] = None,
        model: Optional[Any] = None
    ):
        self.config = config or IntentConfig()
        self.model = model
        
        self.pattern_matcher = IntentPatternMatcher(self.config.language)
        self.slot_filler = SlotFiller()
        
        # 自定义意图
        self._custom_intents: Dict[str, Intent] = {}
        self._intent_examples: Dict[str, List[str]] = defaultdict(list)
        
        # 缓存
        self._cache: Dict[str, IntentResult] = {}
        self._lock = threading.Lock()
        
        # 加载自定义意图
        if self.config.custom_intents:
            self._load_custom_intents(self.config.custom_intents)
    
    def _load_custom_intents(self, custom_intents: Dict[str, Any]):
        """加载自定义意图"""
        for intent_name, intent_data in custom_intents.items():
            intent = Intent(
                name=intent_name,
                category=IntentCategory(intent_data.get("category", "unknown")),
                confidence=1.0,
                description=intent_data.get("description", ""),
                examples=intent_data.get("examples", [])
            )
            self._custom_intents[intent_name] = intent
            self._intent_examples[intent_name] = intent_data.get("examples", [])
    
    def register_intent(
        self,
        name: str,
        category: IntentCategory,
        examples: List[str],
        description: str = ""
    ):
        """注册自定义意图"""
        intent = Intent(
            name=name,
            category=category,
            confidence=1.0,
            description=description,
            examples=examples
        )
        self._custom_intents[name] = intent
        self._intent_examples[name] = examples
    
    def classify(self, text: str) -> IntentResult:
        """
        分类意图
        
        Args:
            text: 输入文本
        
        Returns:
            IntentResult
        """
        start_time = time.time()
        
        # 检查缓存
        cache_key = text[:100]  # 简单缓存键
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        intents = []
        
        # 1. 模式匹配
        pattern_results = self.pattern_matcher.match(text)
        
        for intent_name, confidence in pattern_results[:self.config.max_intents]:
            category = self._get_category(intent_name)
            
            intent = Intent(
                name=intent_name,
                category=category,
                confidence=confidence
            )
            
            # 填充槽位
            if self.config.enable_slot_filling:
                intent.slots = self.slot_filler.fill_slots(text, intent_name)
            
            intents.append(intent)
        
        # 2. 模型分类（如果有）
        if self.model:
            try:
                model_results = self._classify_with_model(text)
                intents = self._merge_results(intents, model_results)
            except Exception as e:
                logger.warning(f"模型分类失败: {e}")
        
        # 3. 自定义意图匹配
        custom_results = self._match_custom_intents(text)
        intents.extend(custom_results)
        
        # 排序并过滤
        intents.sort(key=lambda i: i.confidence, reverse=True)
        intents = [i for i in intents if i.confidence >= self.config.confidence_threshold]
        
        # 如果没有匹配的意图，返回未知
        if not intents:
            intents.append(Intent(
                name="unknown",
                category=IntentCategory.UNKNOWN,
                confidence=0.5
            ))
        
        result = IntentResult(
            text=text,
            intents=intents[:self.config.max_intents],
            processing_time=time.time() - start_time
        )
        
        # 缓存结果
        with self._lock:
            if len(self._cache) < 1000:
                self._cache[cache_key] = result
        
        return result
    
    def _get_category(self, intent_name: str) -> IntentCategory:
        """获取意图类别"""
        category_map = {
            "greeting": IntentCategory.GREETING,
            "farewell": IntentCategory.FAREWELL,
            "query_info": IntentCategory.QUERY,
            "command": IntentCategory.COMMAND,
            "question": IntentCategory.QUESTION,
            "chitchat": IntentCategory.CHITCHAT,
            "complaint": IntentCategory.COMPLAINT,
            "feedback": IntentCategory.FEEDBACK,
            "confirmation": IntentCategory.CONFIRMATION,
            "request": IntentCategory.REQUEST,
        }
        return category_map.get(intent_name, IntentCategory.UNKNOWN)
    
    def _classify_with_model(self, text: str) -> List[Tuple[str, float]]:
        """使用模型分类"""
        # 占位实现
        return []
    
    def _merge_results(
        self,
        pattern_results: List[Intent],
        model_results: List[Tuple[str, float]]
    ) -> List[Intent]:
        """合并结果"""
        # 创建意图名称到置信度的映射
        intent_confidence = {i.name: i.confidence for i in pattern_results}
        
        # 更新置信度
        for intent_name, model_confidence in model_results:
            if intent_name in intent_confidence:
                # 融合置信度
                intent_confidence[intent_name] = (
                    intent_confidence[intent_name] * 0.4 + model_confidence * 0.6
                )
            else:
                intent_confidence[intent_name] = model_confidence
        
        # 更新意图
        for intent in pattern_results:
            if intent.name in intent_confidence:
                intent.confidence = intent_confidence[intent.name]
        
        return pattern_results
    
    def _match_custom_intents(self, text: str) -> List[Intent]:
        """匹配自定义意图"""
        results = []
        text_lower = text.lower()
        
        for intent_name, intent in self._custom_intents.items():
            # 简单关键词匹配
            for example in intent.examples:
                if example.lower() in text_lower:
                    matched_intent = Intent(
                        name=intent_name,
                        category=intent.category,
                        confidence=0.8,
                        description=intent.description
                    )
                    
                    if self.config.enable_slot_filling:
                        matched_intent.slots = self.slot_filler.fill_slots(text, intent_name)
                    
                    results.append(matched_intent)
                    break
        
        return results
    
    def batch_classify(self, texts: List[str]) -> List[IntentResult]:
        """批量分类"""
        return [self.classify(text) for text in texts]
    
    def get_intent_examples(self, intent_name: str) -> List[str]:
        """获取意图示例"""
        return self._intent_examples.get(intent_name, [])
    
    def get_supported_intents(self) -> List[str]:
        """获取支持的意图列表"""
        base_intents = list(self.pattern_matcher.patterns.keys())
        custom_intents = list(self._custom_intents.keys())
        return list(set(base_intents + custom_intents))
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_intents": len(self.get_supported_intents()),
            "custom_intents": len(self._custom_intents),
            "cache_size": len(self._cache),
            "config": {
                "confidence_threshold": self.config.confidence_threshold,
                "enable_slot_filling": self.config.enable_slot_filling,
                "language": self.config.language
            }
        }
    
    def clear_cache(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()


# 工厂函数
def create_intent_classifier(
    config: Optional[IntentConfig] = None,
    model: Optional[Any] = None
) -> IntentClassifier:
    """创建意图分类器"""
    return IntentClassifier(config, model)


# 便捷函数
def classify_intent(text: str) -> IntentResult:
    """便捷意图分类函数"""
    classifier = IntentClassifier()
    return classifier.classify(text)
