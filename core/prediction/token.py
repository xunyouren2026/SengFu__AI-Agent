"""
Token Predictor - Token预测器
预测文本的Token数量，优化上下文管理
"""

import re
import time
import logging
import threading
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


class PredictionMethod(Enum):
    """预测方法"""
    CHAR_RATIO = "char_ratio"       # 字符比例
    WORD_RATIO = "word_ratio"       # 单词比例
    TOKENIZER = "tokenizer"         # 分词器
    ML_MODEL = "ml_model"           # 机器学习模型
    HYBRID = "hybrid"               # 混合方法


@dataclass
class TokenPrediction:
    """Token预测结果"""
    text: str
    token_count: int
    method: PredictionMethod
    confidence: float
    char_count: int
    word_count: int
    processing_time: float = 0.0
    breakdown: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_count": self.token_count,
            "method": self.method.value,
            "confidence": self.confidence,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "processing_time": self.processing_time,
            "breakdown": self.breakdown
        }


@dataclass
class TokenPredictorConfig:
    """Token预测器配置"""
    # 默认方法
    default_method: PredictionMethod = PredictionMethod.HYBRID
    
    # 字符比例（针对不同语言）
    chinese_char_per_token: float = 1.5    # 中文约1.5字符=1token
    english_char_per_token: float = 4.0    # 英文约4字符=1token
    
    # 单词比例
    words_per_token: float = 0.75  # 英文约0.75单词=1token
    
    # 缓存配置
    enable_cache: bool = True
    cache_size: int = 10000
    
    # 模型配置
    model_max_length: int = 4096
    
    # 精度配置
    safety_margin: float = 1.1  # 安全边际（预测值乘以此系数）


class LanguageDetector:
    """语言检测器"""
    
    def __init__(self):
        # 中文Unicode范围
        self.chinese_ranges = [
            (0x4E00, 0x9FFF),    # CJK统一汉字
            (0x3400, 0x4DBF),    # CJK扩展A
            (0x20000, 0x2A6DF),  # CJK扩展B
            (0x2A700, 0x2B73F),  # CJK扩展C
            (0x2B740, 0x2B81F),  # CJK扩展D
        ]
        
        # 日文假名范围
        self.japanese_ranges = [
            (0x3040, 0x309F),    # 平假名
            (0x30A0, 0x30FF),    # 片假名
        ]
        
        # 韩文范围
        self.korean_ranges = [
            (0xAC00, 0xD7AF),    # 韩文音节
            (0x1100, 0x11FF),    # 韩文字母
        ]
    
    def detect(self, text: str) -> Dict[str, float]:
        """
        检测文本语言分布
        
        Returns:
            {language: ratio, ...}
        """
        if not text:
            return {"unknown": 1.0}
        
        total = len(text)
        counts = defaultdict(int)
        
        for char in text:
            code = ord(char)
            
            if self._in_ranges(code, self.chinese_ranges):
                counts["chinese"] += 1
            elif self._in_ranges(code, self.japanese_ranges):
                counts["japanese"] += 1
            elif self._in_ranges(code, self.korean_ranges):
                counts["korean"] += 1
            elif code < 128:
                if char.isalpha():
                    counts["english"] += 1
                elif char.isdigit():
                    counts["number"] += 1
                elif char.isspace():
                    counts["whitespace"] += 1
                else:
                    counts["punctuation"] += 1
            else:
                counts["other"] += 1
        
        return {lang: count / total for lang, count in counts.items()}
    
    def _in_ranges(self, code: int, ranges: List[Tuple[int, int]]) -> bool:
        """检查代码点是否在范围内"""
        for start, end in ranges:
            if start <= code <= end:
                return True
        return False
    
    def get_primary_language(self, text: str) -> str:
        """获取主要语言"""
        distribution = self.detect(text)
        if not distribution:
            return "unknown"
        return max(distribution.items(), key=lambda x: x[1])[0]


class CharRatioPredictor:
    """字符比例预测器"""
    
    def __init__(self, config: TokenPredictorConfig):
        self.config = config
        self.language_detector = LanguageDetector()
    
    def predict(self, text: str) -> TokenPrediction:
        """基于字符比例预测"""
        start_time = time.time()
        
        # 检测语言分布
        lang_dist = self.language_detector.detect(text)
        
        # 计算各语言Token数
        breakdown = {}
        total_tokens = 0
        
        char_counts = {
            "chinese": 0,
            "english": 0,
            "number": 0,
            "punctuation": 0,
            "whitespace": 0,
            "other": 0
        }
        
        # 统计字符
        for char in text:
            code = ord(char)
            if self.language_detector._in_ranges(code, self.language_detector.chinese_ranges):
                char_counts["chinese"] += 1
            elif char.isalpha() and code < 128:
                char_counts["english"] += 1
            elif char.isdigit():
                char_counts["number"] += 1
            elif char.isspace():
                char_counts["whitespace"] += 1
            elif code < 128:
                char_counts["punctuation"] += 1
            else:
                char_counts["other"] += 1
        
        # 计算Token
        breakdown["chinese_tokens"] = int(char_counts["chinese"] / self.config.chinese_char_per_token)
        breakdown["english_tokens"] = int(char_counts["english"] / self.config.english_char_per_token)
        breakdown["number_tokens"] = int(char_counts["number"] / 3)  # 数字约3字符=1token
        breakdown["punctuation_tokens"] = char_counts["punctuation"]  # 标点通常1字符=1token
        breakdown["other_tokens"] = int(char_counts["other"] / 2)
        
        total_tokens = sum(breakdown.values())
        
        # 计算置信度
        confidence = self._calculate_confidence(lang_dist)
        
        return TokenPrediction(
            text=text,
            token_count=total_tokens,
            method=PredictionMethod.CHAR_RATIO,
            confidence=confidence,
            char_count=len(text),
            word_count=len(text.split()),
            processing_time=time.time() - start_time,
            breakdown=breakdown
        )
    
    def _calculate_confidence(self, lang_dist: Dict[str, float]) -> float:
        """计算置信度"""
        # 单一语言时置信度更高
        if not lang_dist:
            return 0.5
        
        max_ratio = max(lang_dist.values())
        return min(0.95, 0.7 + max_ratio * 0.25)


class WordRatioPredictor:
    """单词比例预测器"""
    
    def __init__(self, config: TokenPredictorConfig):
        self.config = config
    
    def predict(self, text: str) -> TokenPrediction:
        """基于单词比例预测"""
        start_time = time.time()
        
        # 分词
        words = self._tokenize(text)
        word_count = len(words)
        
        # 计算Token数
        # 英文单词通常约0.75单词=1token
        # 中文需要特殊处理
        english_words = [w for w in words if self._is_english(w)]
        chinese_chars = sum(len(w) for w in words if self._is_chinese(w))
        
        english_tokens = int(len(english_words) / self.config.words_per_token)
        chinese_tokens = int(chinese_chars / self.config.chinese_char_per_token)
        
        total_tokens = english_tokens + chinese_tokens
        
        breakdown = {
            "english_words": len(english_words),
            "english_tokens": english_tokens,
            "chinese_chars": chinese_chars,
            "chinese_tokens": chinese_tokens
        }
        
        return TokenPrediction(
            text=text,
            token_count=total_tokens,
            method=PredictionMethod.WORD_RATIO,
            confidence=0.8,
            char_count=len(text),
            word_count=word_count,
            processing_time=time.time() - start_time,
            breakdown=breakdown
        )
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        # 按空格和标点分割
        tokens = re.findall(r'[\w\u4e00-\u9fff]+', text)
        return tokens
    
    def _is_english(self, word: str) -> bool:
        """检查是否为英文单词"""
        return all(ord(c) < 128 and c.isalpha() for c in word)
    
    def _is_chinese(self, word: str) -> bool:
        """检查是否为中文"""
        return all('\u4e00' <= c <= '\u9fff' for c in word)


class TokenizerPredictor:
    """分词器预测器"""
    
    def __init__(self, tokenizer: Any, config: TokenPredictorConfig):
        self.tokenizer = tokenizer
        self.config = config
    
    def predict(self, text: str) -> TokenPrediction:
        """使用分词器精确预测"""
        start_time = time.time()
        
        try:
            if hasattr(self.tokenizer, 'encode'):
                tokens = self.tokenizer.encode(text)
                token_count = len(tokens)
            elif hasattr(self.tokenizer, 'tokenize'):
                tokens = self.tokenizer.tokenize(text)
                token_count = len(tokens)
            else:
                # 回退到字符比例
                return self._fallback_predict(text)
            
            return TokenPrediction(
                text=text,
                token_count=token_count,
                method=PredictionMethod.TOKENIZER,
                confidence=1.0,
                char_count=len(text),
                word_count=len(text.split()),
                processing_time=time.time() - start_time,
                breakdown={"tokenizer_tokens": token_count}
            )
        
        except Exception as e:
            logger.warning(f"分词器预测失败: {e}")
            return self._fallback_predict(text)
    
    def _fallback_predict(self, text: str) -> TokenPrediction:
        """回退预测"""
        predictor = CharRatioPredictor(self.config)
        return predictor.predict(text)


class HybridPredictor:
    """混合预测器"""
    
    def __init__(
        self,
        config: TokenPredictorConfig,
        tokenizer: Optional[Any] = None
    ):
        self.config = config
        self.char_predictor = CharRatioPredictor(config)
        self.word_predictor = WordRatioPredictor(config)
        self.tokenizer_predictor = TokenizerPredictor(tokenizer, config) if tokenizer else None
    
    def predict(self, text: str) -> TokenPrediction:
        """混合预测"""
        start_time = time.time()
        
        predictions = []
        
        # 字符比例预测
        char_pred = self.char_predictor.predict(text)
        predictions.append((char_pred, 0.3))
        
        # 单词比例预测
        word_pred = self.word_predictor.predict(text)
        predictions.append((word_pred, 0.3))
        
        # 分词器预测（如果有）
        if self.tokenizer_predictor:
            tokenizer_pred = self.tokenizer_predictor.predict(text)
            predictions.append((tokenizer_pred, 0.4))
        
        # 加权平均
        total_weight = sum(w for _, w in predictions)
        weighted_tokens = sum(pred.token_count * weight for pred, weight in predictions)
        final_tokens = int(weighted_tokens / total_weight)
        
        # 计算置信度
        confidences = [pred.confidence * weight for pred, weight in predictions]
        final_confidence = sum(confidences) / total_weight
        
        return TokenPrediction(
            text=text,
            token_count=final_tokens,
            method=PredictionMethod.HYBRID,
            confidence=final_confidence,
            char_count=len(text),
            word_count=len(text.split()),
            processing_time=time.time() - start_time,
            breakdown={
                "char_ratio_tokens": char_pred.token_count,
                "word_ratio_tokens": word_pred.token_count,
                "final_tokens": final_tokens
            }
        )


class TokenPredictor:
    """Token预测器主类"""
    
    def __init__(
        self,
        config: Optional[TokenPredictorConfig] = None,
        tokenizer: Optional[Any] = None
    ):
        self.config = config or TokenPredictorConfig()
        self.tokenizer = tokenizer
        
        # 初始化预测器
        self.char_predictor = CharRatioPredictor(self.config)
        self.word_predictor = WordRatioPredictor(self.config)
        
        if tokenizer:
            self.tokenizer_predictor = TokenizerPredictor(tokenizer, self.config)
        else:
            self.tokenizer_predictor = None
        
        self.hybrid_predictor = HybridPredictor(self.config, tokenizer)
        
        # 缓存
        self._cache: Dict[str, TokenPrediction] = {}
        self._lock = threading.Lock()
    
    def predict(
        self,
        text: str,
        method: Optional[PredictionMethod] = None
    ) -> TokenPrediction:
        """
        预测Token数量
        
        Args:
            text: 输入文本
            method: 预测方法（可选）
        
        Returns:
            TokenPrediction
        """
        # 检查缓存
        if self.config.enable_cache:
            cache_key = f"{text[:100]}_{method.value if method else 'default'}"
            with self._lock:
                if cache_key in self._cache:
                    return self._cache[cache_key]
        
        # 选择预测方法
        method = method or self.config.default_method
        
        if method == PredictionMethod.CHAR_RATIO:
            prediction = self.char_predictor.predict(text)
        elif method == PredictionMethod.WORD_RATIO:
            prediction = self.word_predictor.predict(text)
        elif method == PredictionMethod.TOKENIZER and self.tokenizer_predictor:
            prediction = self.tokenizer_predictor.predict(text)
        elif method == PredictionMethod.HYBRID:
            prediction = self.hybrid_predictor.predict(text)
        else:
            prediction = self.hybrid_predictor.predict(text)
        
        # 应用安全边际
        prediction.token_count = int(prediction.token_count * self.config.safety_margin)
        
        # 缓存结果
        if self.config.enable_cache:
            with self._lock:
                if len(self._cache) < self.config.cache_size:
                    self._cache[cache_key] = prediction
        
        return prediction
    
    def predict_batch(
        self,
        texts: List[str],
        method: Optional[PredictionMethod] = None
    ) -> List[TokenPrediction]:
        """批量预测"""
        return [self.predict(text, method) for text in texts]
    
    def fits_in_context(
        self,
        text: str,
        max_tokens: Optional[int] = None,
        method: Optional[PredictionMethod] = None
    ) -> Tuple[bool, int]:
        """
        检查文本是否适合上下文
        
        Returns:
            (是否适合, 预测Token数)
        """
        max_tokens = max_tokens or self.config.model_max_length
        prediction = self.predict(text, method)
        return prediction.token_count <= max_tokens, prediction.token_count
    
    def truncate_to_fit(
        self,
        text: str,
        max_tokens: int,
        method: Optional[PredictionMethod] = None,
        preserve_end: bool = False
    ) -> str:
        """
        截断文本以适应Token限制
        
        Args:
            text: 输入文本
            max_tokens: 最大Token数
            method: 预测方法
            preserve_end: 是否保留结尾
        
        Returns:
            截断后的文本
        """
        prediction = self.predict(text, method)
        
        if prediction.token_count <= max_tokens:
            return text
        
        # 计算需要保留的字符比例
        ratio = max_tokens / prediction.token_count
        target_chars = int(len(text) * ratio * 0.9)  # 留一些余量
        
        if preserve_end:
            # 保留开头和结尾
            head_chars = int(target_chars * 0.7)
            tail_chars = target_chars - head_chars
            return text[:head_chars] + "\n...[省略]...\n" + text[-tail_chars:]
        else:
            return text[:target_chars]
    
    def estimate_cost(
        self,
        text: str,
        price_per_1k_tokens: float = 0.002,
        method: Optional[PredictionMethod] = None
    ) -> float:
        """
        估算成本
        
        Args:
            text: 输入文本
            price_per_1k_tokens: 每千Token价格
            method: 预测方法
        
        Returns:
            估算成本
        """
        prediction = self.predict(text, method)
        return (prediction.token_count / 1000) * price_per_1k_tokens
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "cache_size": len(self._cache),
            "config": {
                "default_method": self.config.default_method.value,
                "chinese_char_per_token": self.config.chinese_char_per_token,
                "english_char_per_token": self.config.english_char_per_token,
                "safety_margin": self.config.safety_margin,
                "model_max_length": self.config.model_max_length
            }
        }
    
    def clear_cache(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()


# 工厂函数
def create_token_predictor(
    config: Optional[TokenPredictorConfig] = None,
    tokenizer: Optional[Any] = None
) -> TokenPredictor:
    """创建Token预测器"""
    return TokenPredictor(config, tokenizer)


# 便捷函数
def count_tokens(
    text: str,
    tokenizer: Optional[Any] = None
) -> int:
    """便捷Token计数函数"""
    predictor = TokenPredictor(tokenizer=tokenizer)
    prediction = predictor.predict(text)
    return prediction.token_count
