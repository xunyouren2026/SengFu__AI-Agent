"""
多语言智能路由 (Multi-Language Intelligent Router)

该模块提供多语言自动检测和智能路由功能，支持：
- 中/英/日/韩等语言的自动检测
- 语言专属模型路由
- 翻译增强选项
- 语言偏好学习

核心功能：
1. 语言自动检测：基于字符集和n-gram的语言识别
2. 模型能力匹配：根据模型的语言能力选择最佳模型
3. 翻译增强：可选的翻译服务集成
4. 语言偏好：支持用户和渠道的语言偏好设置

Author: AGI Team
Version: 1.0.0
"""

import re
import time
import threading
import logging
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict, List, Optional, Any, Set, Tuple, 
    Callable, Union, Pattern, TypeVar
)
from collections import defaultdict
from datetime import datetime

# 配置日志
logger = logging.getLogger(__name__)

T = TypeVar('T')


class Language(Enum):
    """
    支持的语言枚举
    
    使用ISO 639-1代码。
    """
    ZH = "zh"           # 中文
    EN = "en"           # 英语
    JA = "ja"           # 日语
    KO = "ko"           # 韩语
    ES = "es"           # 西班牙语
    FR = "fr"           # 法语
    DE = "de"           # 德语
    PT = "pt"           # 葡萄牙语
    RU = "ru"           # 俄语
    AR = "ar"           # 阿拉伯语
    HI = "hi"           # 印地语
    TH = "th"           # 泰语
    VI = "vi"           # 越南语
    ID = "id"           # 印尼语
    MS = "ms"           # 马来语
    TR = "tr"           # 土耳其语
    PL = "pl"           # 波兰语
    NL = "nl"           # 荷兰语
    IT = "it"           # 意大利语
    UK = "uk"           # 乌克兰语
    CS = "cs"           # 捷克语
    SV = "sv"           # 瑞典语
    DA = "da"           # 丹麦语
    FI = "fi"           # 芬兰语
    NO = "no"           # 挪威语
    HE = "he"           # 希伯来语
    UKN = "unknown"      # 未知语言
    
    @classmethod
    def from_code(cls, code: str) -> 'Language':
        """从代码获取语言枚举"""
        code = code.lower().split('-')[0]  # 处理如zh-CN的情况
        try:
            return cls(code)
        except ValueError:
            return cls.UKN
    
    @property
    def display_name(self) -> str:
        """获取显示名称"""
        names = {
            "zh": "中文", "en": "英语", "ja": "日语", "ko": "韩语",
            "es": "西班牙语", "fr": "法语", "de": "德语", "pt": "葡萄牙语",
            "ru": "俄语", "ar": "阿拉伯语", "hi": "印地语", "th": "泰语",
            "vi": "越南语", "id": "印尼语", "ms": "马来语", "tr": "土耳其语",
            "pl": "波兰语", "nl": "荷兰语", "it": "意大利语", "uk": "乌克兰语",
            "cs": "捷克语", "sv": "瑞典语", "da": "丹麦语", "fi": "芬兰语",
            "no": "挪威语", "he": "希伯来语", "unknown": "未知语言"
        }
        return names.get(self.value, self.value)


class LanguageFamily(Enum):
    """语系"""
    SINO_TIBETAN = auto()     # 汉藏语系
    INDO_EUROPEAN = auto()    # 印欧语系
    ALTAIC = auto()           # 阿尔泰语系
    JAPANIC = auto()          # 日本语系
    KOREANIC = auto()         # 朝鲜语系
    AUSTRONESIAN = auto()     # 南岛语系
    SEMITIC = auto()          # 闪米特语系
    DRAVIDIAN = auto()        # 达罗毗荼语系
    URALIC = auto()           # 乌拉尔语系
    GERMANIC = auto()         # 日耳曼语族
    ROMANCE = auto()          # 罗曼语族
    SLAVIC = auto()           # 斯拉夫语族
    UNKNOWN = auto()          # 未知


# 语言到语系的映射
LANGUAGE_FAMILY_MAP: Dict[Language, LanguageFamily] = {
    Language.ZH: LanguageFamily.SINO_TIBETAN,
    Language.EN: LanguageFamily.GERMANIC,
    Language.JA: LanguageFamily.JAPANIC,
    Language.KO: LanguageFamily.KOREANIC,
    Language.ES: LanguageFamily.ROMANCE,
    Language.FR: LanguageFamily.ROMANCE,
    Language.DE: LanguageFamily.GERMANIC,
    Language.PT: LanguageFamily.ROMANCE,
    Language.RU: LanguageFamily.SLAVIC,
    Language.AR: LanguageFamily.SEMITIC,
    Language.HI: LanguageFamily.INDO_EUROPEAN,
    Language.TH: LanguageFamily.SINO_TIBETAN,
    Language.VI: LanguageFamily.AUSTRONESIAN,
    Language.ID: LanguageFamily.AUSTRONESIAN,
    Language.MS: LanguageFamily.AUSTRONESIAN,
    Language.TR: LanguageFamily.ALTAIC,
    Language.PL: LanguageFamily.SLAVIC,
    Language.NL: LanguageFamily.GERMANIC,
    Language.IT: LanguageFamily.ROMANCE,
    Language.UK: LanguageFamily.SLAVIC,
    Language.CS: LanguageFamily.SLAVIC,
    Language.SV: LanguageFamily.GERMANIC,
    Language.DA: LanguageFamily.GERMANIC,
    Language.FI: LanguageFamily.URALIC,
    Language.NO: LanguageFamily.GERMANIC,
    Language.HE: LanguageFamily.SEMITIC,
}


@dataclass
class LanguageDetection:
    """
    语言检测结果
    
    Attributes:
        detected_language: 检测到的语言
        confidence: 置信度 (0-1)
        alternatives: 备选语言列表 [(语言, 置信度)]
        script_type: 文字类型 (如CJK, Latin, Arabic等)
        is_mixed: 是否混合语言
    """
    detected_language: Language
    confidence: float
    alternatives: List[Tuple[Language, float]] = field(default_factory=list)
    script_type: str = "latin"
    is_mixed: bool = False
    
    @property
    def language_family(self) -> LanguageFamily:
        """获取语系"""
        return LANGUAGE_FAMILY_MAP.get(self.detected_language, LanguageFamily.UNKNOWN)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "detected_language": self.detected_language.value,
            "detected_language_name": self.detected_language.display_name,
            "confidence": self.confidence,
            "alternatives": [
                {"language": lang.value, "confidence": conf}
                for lang, conf in self.alternatives
            ],
            "script_type": self.script_type,
            "is_mixed": self.is_mixed,
            "language_family": self.language_family.name,
        }


@dataclass
class LanguageConfig:
    """
    语言配置
    
    Attributes:
        language: 语言
        model_id: 首选模型ID
        fallback_model_id: 备用模型ID
        translation_enabled: 是否启用翻译
        auto_detect: 是否自动检测语言
        max_context_tokens: 最大上下文token
        metadata: 其他配置
    """
    language: Language
    model_id: Optional[str] = None
    fallback_model_id: Optional[str] = None
    translation_enabled: bool = False
    auto_detect: bool = True
    max_context_tokens: int = 4096
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TranslationOption:
    """
    翻译选项
    
    Attributes:
        enabled: 是否启用翻译
        source_lang: 源语言 (None表示自动检测)
        target_lang: 目标语言
        method: 翻译方法 (pretranslate, posttranslate, hybrid)
        quality_threshold: 质量阈值
    """
    enabled: bool = False
    source_lang: Optional[Language] = None
    target_lang: Language = Language.EN
    method: str = "pretranslate"  # pretranslate, posttranslate, hybrid
    quality_threshold: float = 0.8


# 字符范围定义
CJK_RANGES = [
    (0x4E00, 0x9FFF),    # CJK统一表意文字
    (0x3400, 0x4DBF),    # CJK扩展A
    (0x20000, 0x2A6DF),  # CJK扩展B
    (0x2A700, 0x2B73F),  # CJK扩展C
    (0x2B740, 0x2B81F),  # CJK扩展D
    (0x3000, 0x303F),    # CJK符号
    (0xFF00, 0xFFEF),    # 全角字符
]

KOREAN_RANGES = [
    (0xAC00, 0xD7AF),    # 韩文音节
    (0x1100, 0x11FF),    # 韩文字母
    (0x3130, 0x318F),    # 韩文兼容字母
]

JAPANESE_SPECIFIC = [
    (0x3040, 0x309F),    # 平假名
    (0x30A0, 0x30FF),    # 片假名
]

ARABIC_RANGES = [
    (0x0600, 0x06FF),    # 阿拉伯文
    (0x0750, 0x077F),    # 阿拉伯补充
]

CYRILLIC_RANGES = [
    (0x0400, 0x04FF),    # 西里尔文
    (0x0500, 0x052F),    # 西里尔文补充
]

THAI_RANGES = [
    (0x0E00, 0x0E7F),    # 泰文
]

DEVANAGARI_RANGES = [
    (0x0900, 0x097F),    # 天城文
]

LATIN_RANGES = [
    (0x0000, 0x024F),    # 拉丁文
    (0x1E00, 0x1EFF),    # 拉丁文扩展
]


def _char_in_range(char: str, ranges: List[Tuple[int, int]]) -> bool:
    """检查字符是否在指定范围内"""
    code = ord(char)
    return any(start <= code <= end for start, end in ranges)


def _count_chars_in_ranges(text: str, ranges: List[Tuple[int, int]]) -> int:
    """统计文本中在指定范围内的字符数"""
    return sum(1 for char in text if _char_in_range(char, ranges))


def _is_cjk(text: str) -> bool:
    """检查是否包含CJK字符"""
    return _count_chars_in_ranges(text, CJK_RANGES) > 0


def _is_korean(text: str) -> bool:
    """检查是否包含韩文"""
    return _count_chars_in_ranges(text, KOREAN_RANGES) > 0


def _is_japanese(text: str) -> bool:
    """检查是否包含日文假名"""
    return _count_chars_in_ranges(text, JAPANESE_SPECIFIC) > 0


def _is_arabic(text: str) -> bool:
    """检查是否包含阿拉伯文"""
    return _count_chars_in_ranges(text, ARABIC_RANGES) > 0


def _is_cyrillic(text: str) -> bool:
    """检查是否包含西里尔文"""
    return _count_chars_in_ranges(text, CYRILLIC_RANGES) > 0


def _is_thai(text: str) -> bool:
    """检查是否包含泰文"""
    return _count_chars_in_ranges(text, THAI_RANGES) > 0


def _is_devanagari(text: str) -> bool:
    """检查是否包含天城文"""
    return _count_chars_in_ranges(text, DEVANAGARI_RANGES) > 0


def _is_latin(text: str) -> bool:
    """检查是否包含拉丁字符"""
    return _count_chars_in_ranges(text, LATIN_RANGES) > 0


class LanguageDetector:
    """
    语言检测器
    
    基于字符范围和n-gram的语言检测。
    """
    
    # 语言特征词
    LANGUAGE_KEYWORDS: Dict[Language, List[str]] = {
        Language.ZH: ["的", "是", "在", "有", "我", "你", "他", "她", "它", "们", "了", "和", "与", "或", "但", "如果", "因为", "所以", "这个", "那个", "什么", "如何", "为什么"],
        Language.EN: ["the", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "can", "this", "that", "what", "how", "why", "when", "where", "who"],
        Language.JA: ["です", "ます", "した", "して", "する", "が", "の", "に", "を", "は", "で", "て", "と", "し", "れ", "さ", "ある", "いる", "も", "する", "看过", "です", "日本語"],
        Language.KO: ["이", "그", "저", "것", "수", "등", "들", "및", "에", "의", "를", "을", "가", "은", "는", "로", "으로", "와", "과", "에서", "에게", "한테"],
        Language.ES: ["el", "la", "los", "las", "de", "que", "es", "en", "un", "una", "por", "con", "para", "como", "pero", "este", "esta", "ese", "esa"],
        Language.FR: ["le", "la", "les", "de", "des", "que", "qui", "est", "dans", "un", "une", "pour", "par", "avec", "sur", "ce", "cette", "ces", "mais", "comme"],
        Language.DE: ["der", "die", "das", "und", "den", "von", "zu", "in", "ein", "eine", "mit", "auf", "für", "sich", "nicht", "es", "an", "werden", "aus", "er"],
        Language.RU: ["в", "на", "не", "с", "что", "он", "как", "а", "то", "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы"],
    }
    
    # 语言检测的字符数阈值
    MIN_CHARS = 5
    
    def __init__(self):
        """初始化语言检测器"""
        self._detection_cache: Dict[str, LanguageDetection] = {}
        self._cache_lock = threading.Lock()
        self._detection_count = 0
    
    def detect(self, text: str, use_cache: bool = True) -> LanguageDetection:
        """
        检测文本语言。
        
        Args:
            text: 待检测文本
            use_cache: 是否使用缓存
            
        Returns:
            语言检测结果
        """
        if not text or len(text.strip()) < self.MIN_CHARS:
            return LanguageDetection(
                detected_language=Language.UKN,
                confidence=0.0
            )
        
        # 检查缓存
        cache_key = text[:100]  # 使用前100字符作为缓存键
        if use_cache:
            with self._cache_lock:
                if cache_key in self._detection_cache:
                    return self._detection_cache[cache_key]
        
        # 执行检测
        result = self._detect_impl(text)
        
        # 更新缓存
        if use_cache:
            with self._cache_lock:
                self._detection_cache[cache_key] = result
                self._detection_count += 1
                
                # 限制缓存大小
                if len(self._detection_cache) > 10000:
                    # 清除最老的50%
                    keys_to_remove = list(self._detection_cache.keys())[:5000]
                    for key in keys_to_remove:
                        del self._detection_cache[key]
        
        return result
    
    def _detect_impl(self, text: str) -> LanguageDetection:
        """实际的语言检测实现"""
        text = text.strip()
        total_chars = len(text)
        
        if total_chars < self.MIN_CHARS:
            return LanguageDetection(
                detected_language=Language.UKN,
                confidence=0.0
            )
        
        # 首先检查Unicode范围
        script_type = self._detect_script_type(text)
        
        # 根据文字类型进行检测
        if script_type == "cjk":
            return self._detect_cjk_language(text)
        elif script_type == "korean":
            return self._detect_korean(text)
        elif script_type == "japanese":
            return self._detect_japanese(text)
        elif script_type == "arabic":
            return self._detect_language_from_keywords(text, [Language.AR])
        elif script_type == "cyrillic":
            return self._detect_language_from_keywords(text, [Language.RU])
        elif script_type == "thai":
            return self._detect_language_from_keywords(text, [Language.TH])
        elif script_type == "devanagari":
            return self._detect_language_from_keywords(text, [Language.HI])
        else:
            # 拉丁文字，使用n-gram或关键词检测
            return self._detect_latin_language(text)
    
    def _detect_script_type(self, text: str) -> str:
        """检测文字类型"""
        scores = {}
        
        if _is_cjk(text):
            scores["cjk"] = _count_chars_in_ranges(text, CJK_RANGES)
        if _is_korean(text):
            scores["korean"] = _count_chars_in_ranges(text, KOREAN_RANGES)
        if _is_japanese(text):
            scores["japanese"] = _count_chars_in_ranges(text, JAPANESE_SPECIFIC)
        if _is_arabic(text):
            scores["arabic"] = _count_chars_in_ranges(text, ARABIC_RANGES)
        if _is_cyrillic(text):
            scores["cyrillic"] = _count_chars_in_ranges(text, CYRILLIC_RANGES)
        if _is_thai(text):
            scores["thai"] = _count_chars_in_ranges(text, THAI_RANGES)
        if _is_devanagari(text):
            scores["devanagari"] = _count_chars_in_ranges(text, DEVANAGARI_RANGES)
        if _is_latin(text):
            scores["latin"] = _count_chars_in_ranges(text, LATIN_RANGES)
        
        if not scores:
            return "unknown"
        
        return max(scores, key=scores.get)
    
    def _detect_cjk_language(self, text: str) -> LanguageDetection:
        """检测CJK语言"""
        # 检测韩文
        korean_count = _count_chars_in_ranges(text, KOREAN_RANGES)
        # 检测日文假名
        japanese_count = _count_chars_in_ranges(text, JAPANESE_SPECIFIC)
        
        if korean_count > 2 and korean_count > japanese_count:
            return LanguageDetection(
                detected_language=Language.KO,
                confidence=min(0.95, 0.5 + korean_count * 0.05),
                script_type="korean"
            )
        
        if japanese_count > 2:
            return LanguageDetection(
                detected_language=Language.JA,
                confidence=min(0.95, 0.5 + japanese_count * 0.05),
                script_type="japanese"
            )
        
        # 默认检测中文关键词
        return self._detect_language_from_keywords(text, [Language.ZH])
    
    def _detect_korean(self, text: str) -> LanguageDetection:
        """检测韩语"""
        return self._detect_language_from_keywords(text, [Language.KO])
    
    def _detect_japanese(self, text: str) -> LanguageDetection:
        """检测日语"""
        return self._detect_language_from_keywords(text, [Language.JA])
    
    def _detect_latin_language(self, text: str) -> LanguageDetection:
        """检测拉丁语系语言"""
        # 使用关键词检测
        text_lower = text.lower()
        
        scores = {}
        for lang, keywords in self.LANGUAGE_KEYWORDS.items():
            if lang not in [Language.ZH, Language.JA, Language.KO]:
                score = sum(1 for kw in keywords if kw.lower() in text_lower)
                if score > 0:
                    scores[lang] = score
        
        if not scores:
            # 默认返回英语
            return LanguageDetection(
                detected_language=Language.EN,
                confidence=0.5,
                script_type="latin"
            )
        
        # 排序并返回结果
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_lang, top_score = sorted_scores[0]
        
        # 计算置信度
        confidence = min(0.95, 0.3 + top_score * 0.15)
        
        # 备选语言
        alternatives = sorted_scores[1:4]
        
        return LanguageDetection(
            detected_language=top_lang,
            confidence=confidence,
            alternatives=alternatives,
            script_type="latin"
        )
    
    def _detect_language_from_keywords(
        self,
        text: str,
        candidates: List[Language]
    ) -> LanguageDetection:
        """基于关键词检测语言"""
        text_lower = text.lower()
        
        scores = {}
        for lang in candidates:
            keywords = self.LANGUAGE_KEYWORDS.get(lang, [])
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            scores[lang] = score
        
        if not scores or max(scores.values()) == 0:
            return LanguageDetection(
                detected_language=candidates[0] if candidates else Language.UKN,
                confidence=0.3,
                script_type="mixed"
            )
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_lang, top_score = sorted_scores[0]
        
        confidence = min(0.95, 0.4 + top_score * 0.1)
        
        return LanguageDetection(
            detected_language=top_lang,
            confidence=confidence,
            alternatives=sorted_scores[1:3],
            is_mixed=len([s for s in scores.values() if s > 0]) > 1
        )
    
    def clear_cache(self) -> None:
        """清除检测缓存"""
        with self._cache_lock:
            self._detection_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取检测统计"""
        with self._cache_lock:
            return {
                "cache_size": len(self._detection_cache),
                "total_detections": self._detection_count,
            }


class LanguageRouter:
    """
    多语言智能路由器
    
    Features:
        - 自动语言检测
        - 语言专属模型路由
        - 翻译增强选项
        - 用户/渠道语言偏好
        - 语言能力缓存
    
    Example:
        ```python
        # 创建路由器
        router = LanguageRouter()
        
        # 注册语言配置
        router.register_language_config(LanguageConfig(
            language=Language.ZH,
            model_id="glm-4",
            translation_enabled=False
        ))
        
        # 配置翻译选项
        router.set_translation_option(
            user_id="user123",
            TranslationOption(
                enabled=True,
                target_lang=Language.EN
            )
        )
        
        # 路由请求
        route = router.route(
            text="你好世界",
            user_id="user123",
            channel_id="web_chat"
        )
        print(f"Detected: {route.detection.detected_language}")
        print(f"Selected model: {route.model_id}")
        ```
    """
    
    def __init__(self):
        """初始化语言路由器"""
        self._detector = LanguageDetector()
        self._language_configs: Dict[Language, LanguageConfig] = {}
        self._user_preferences: Dict[str, Language] = {}
        self._channel_preferences: Dict[str, Language] = {}
        self._translation_options: Dict[str, TranslationOption] = {}
        self._route_history: List[Dict] = []
        self._lock = threading.RLock()
        
        # 注册默认语言配置
        self._register_default_configs()
    
    def _register_default_configs(self) -> None:
        """注册默认语言配置"""
        defaults = [
            LanguageConfig(Language.ZH, model_id="glm-4", translation_enabled=False),
            LanguageConfig(Language.EN, model_id="gpt-3.5-turbo", translation_enabled=False),
            LanguageConfig(Language.JA, model_id="gpt-3.5-turbo", translation_enabled=True),
            LanguageConfig(Language.KO, model_id="gpt-3.5-turbo", translation_enabled=True),
            LanguageConfig(Language.ES, model_id="gpt-3.5-turbo", translation_enabled=True),
            LanguageConfig(Language.FR, model_id="gpt-3.5-turbo", translation_enabled=True),
            LanguageConfig(Language.DE, model_id="gpt-3.5-turbo", translation_enabled=True),
        ]
        
        for config in defaults:
            self.register_language_config(config)
    
    def register_language_config(self, config: LanguageConfig) -> None:
        """
        注册语言配置。
        
        Args:
            config: 语言配置
        """
        with self._lock:
            self._language_configs[config.language] = config
            logger.info(f"Registered language config for {config.language.value}")
    
    def get_language_config(self, language: Language) -> Optional[LanguageConfig]:
        """获取语言配置"""
        return self._language_configs.get(language)
    
    def set_user_preference(self, user_id: str, language: Language) -> None:
        """
        设置用户语言偏好。
        
        Args:
            user_id: 用户ID
            language: 偏好语言
        """
        with self._lock:
            self._user_preferences[user_id] = language
    
    def get_user_preference(self, user_id: str) -> Optional[Language]:
        """获取用户语言偏好"""
        return self._user_preferences.get(user_id)
    
    def set_channel_preference(self, channel_id: str, language: Language) -> None:
        """
        设置渠道语言偏好。
        
        Args:
            channel_id: 渠道ID
            language: 偏好语言
        """
        with self._lock:
            self._channel_preferences[channel_id] = language
    
    def get_channel_preference(self, channel_id: str) -> Optional[Language]:
        """获取渠道语言偏好"""
        return self._channel_preferences.get(channel_id)
    
    def set_translation_option(
        self,
        key: str,
        option: TranslationOption
    ) -> None:
        """
        设置翻译选项。
        
        Args:
            key: 标识键 (user_id或channel_id)
            option: 翻译选项
        """
        with self._lock:
            self._translation_options[key] = option
    
    def get_translation_option(self, key: str) -> Optional[TranslationOption]:
        """获取翻译选项"""
        return self._translation_options.get(key)
    
    def detect_language(self, text: str) -> LanguageDetection:
        """
        检测文本语言。
        
        Args:
            text: 待检测文本
            
        Returns:
            检测结果
        """
        return self._detector.detect(text)
    
    def route(
        self,
        text: str,
        user_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        force_language: Optional[Language] = None,
        prefer_same_language: bool = True
    ) -> 'RoutingResult':
        """
        路由文本请求。
        
        Args:
            text: 待处理文本
            user_id: 用户ID
            channel_id: 渠道ID
            force_language: 强制指定语言
            prefer_same_language: 是否优先使用与输入相同的语言
            
        Returns:
            路由结果
        """
        # 检测语言
        detection = self._detector.detect(text)
        
        # 确定目标语言
        target_language = force_language
        
        if target_language is None:
            # 依次检查用户偏好、渠道偏好
            if user_id and user_id in self._user_preferences:
                target_language = self._user_preferences[user_id]
            elif channel_id and channel_id in self._channel_preferences:
                target_language = self._channel_preferences[channel_id]
            else:
                # 使用检测到的语言
                target_language = detection.detected_language
        
        # 获取语言配置
        config = self._language_configs.get(target_language)
        
        # 确定使用的模型
        model_id = config.model_id if config else "gpt-3.5-turbo"
        fallback_model_id = config.fallback_model_id if config else None
        
        # 检查翻译选项
        translation = None
        needs_translation = False
        
        option_key = user_id or channel_id or "default"
        translation_option = self._translation_options.get(option_key)
        
        if translation_option and translation_option.enabled:
            # 需要翻译
            if detection.detected_language != translation_option.target_lang:
                needs_translation = True
                translation = translation_option
        
        # 检查是否需要翻译增强
        if config and config.translation_enabled and not needs_translation:
            if detection.detected_language != target_language:
                needs_translation = True
                translation = TranslationOption(
                    enabled=True,
                    source_lang=detection.detected_language,
                    target_lang=target_language,
                    method="pretranslate"
                )
        
        # 记录路由历史
        route_info = {
            "text_preview": text[:50],
            "detected_language": detection.detected_language.value,
            "target_language": target_language.value,
            "model_id": model_id,
            "needs_translation": needs_translation,
            "timestamp": datetime.now().isoformat(),
        }
        
        with self._lock:
            self._route_history.append(route_info)
            if len(self._route_history) > 1000:
                self._route_history = self._route_history[-500:]
        
        return RoutingResult(
            detection=detection,
            target_language=target_language,
            model_id=model_id,
            fallback_model_id=fallback_model_id,
            needs_translation=needs_translation,
            translation=translation,
            routing_reason=self._generate_routing_reason(
                detection, target_language, config
            )
        )
    
    def _generate_routing_reason(
        self,
        detection: LanguageDetection,
        target_language: Language,
        config: Optional[LanguageConfig]
    ) -> str:
        """生成路由原因"""
        reasons = []
        
        # 语言检测
        if detection.confidence > 0.8:
            reasons.append(f"检测到{detection.detected_language.display_name}（置信度{detection.confidence:.0%}）")
        elif detection.confidence > 0.5:
            reasons.append(f"可能为{detection.detected_language.display_name}")
        else:
            reasons.append(f"语言不确定，默认使用{target_language.display_name}")
        
        # 模型选择
        if config and config.model_id:
            reasons.append(f"使用{config.model_id}模型")
        
        # 翻译
        if config and config.translation_enabled:
            reasons.append("启用翻译增强")
        
        return "; ".join(reasons)
    
    def batch_route(
        self,
        texts: List[str],
        user_id: Optional[str] = None,
        channel_id: Optional[str] = None
    ) -> List['RoutingResult']:
        """
        批量路由文本请求。
        
        Args:
            texts: 文本列表
            user_id: 用户ID
            channel_id: 渠道ID
            
        Returns:
            路由结果列表
        """
        return [
            self.route(text, user_id, channel_id)
            for text in texts
        ]
    
    def learn_from_feedback(
        self,
        text: str,
        detected_language: Language,
        actual_language: Language,
        was_correct: bool
    ) -> None:
        """
        从反馈中学习以改进语言检测。
        
        Args:
            text: 文本
            detected_language: 检测到的语言
            actual_language: 实际语言
            was_correct: 检测是否正确
        """
        # 这里可以添加学习逻辑
        # 例如：更新关键词权重、调整字符范围阈值等
        logger.info(
            f"Language feedback: detected={detected_language.value}, "
            f"actual={actual_language.value}, correct={was_correct}"
        )
    
    def get_route_stats(self) -> Dict[str, Any]:
        """获取路由统计"""
        with self._lock:
            language_counts = defaultdict(int)
            model_counts = defaultdict(int)
            
            for route in self._route_history:
                language_counts[route["detected_language"]] += 1
                model_counts[route["model_id"]] += 1
            
            return {
                "total_routes": len(self._route_history),
                "language_distribution": dict(language_counts),
                "model_distribution": dict(model_counts),
                "detector_stats": self._detector.get_stats(),
            }
    
    def export_config(self) -> Dict[str, Any]:
        """导出配置"""
        with self._lock:
            return {
                "language_configs": {
                    lang.value: {
                        "model_id": config.model_id,
                        "fallback_model_id": config.fallback_model_id,
                        "translation_enabled": config.translation_enabled,
                        "max_context_tokens": config.max_context_tokens,
                    }
                    for lang, config in self._language_configs.items()
                },
                "user_preferences": {
                    uid: lang.value for uid, lang in self._user_preferences.items()
                },
                "channel_preferences": {
                    cid: lang.value for cid, lang in self._channel_preferences.items()
                },
            }
    
    def import_config(self, config: Dict[str, Any]) -> None:
        """导入配置"""
        with self._lock:
            # 导入语言配置
            lang_configs = config.get("language_configs", {})
            for lang_code, lang_config in lang_configs.items():
                lang = Language(lang_code)
                self._language_configs[lang] = LanguageConfig(
                    language=lang,
                    model_id=lang_config.get("model_id"),
                    fallback_model_id=lang_config.get("fallback_model_id"),
                    translation_enabled=lang_config.get("translation_enabled", False),
                    max_context_tokens=lang_config.get("max_context_tokens", 4096),
                )
            
            # 导入用户偏好
            for uid, lang_code in config.get("user_preferences", {}).items():
                self._user_preferences[uid] = Language(lang_code)
            
            # 导入渠道偏好
            for cid, lang_code in config.get("channel_preferences", {}).items():
                self._channel_preferences[cid] = Language(lang_code)


@dataclass
class RoutingResult:
    """
    路由结果
    
    Attributes:
        detection: 语言检测结果
        target_language: 目标语言
        model_id: 选择的模型ID
        fallback_model_id: 备用模型ID
        needs_translation: 是否需要翻译
        translation: 翻译选项
        routing_reason: 路由原因说明
    """
    detection: LanguageDetection
    target_language: Language
    model_id: str
    fallback_model_id: Optional[str] = None
    needs_translation: bool = False
    translation: Optional[TranslationOption] = None
    routing_reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "detection": self.detection.to_dict(),
            "target_language": self.target_language.value,
            "model_id": self.model_id,
            "fallback_model_id": self.fallback_model_id,
            "needs_translation": self.needs_translation,
            "translation": self.translation.__dict__ if self.translation else None,
            "routing_reason": self.routing_reason,
        }
