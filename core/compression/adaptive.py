"""
Adaptive Compression - 自适应压缩系统
根据内容特征动态选择最佳压缩策略
"""

import re
import time
import logging
import threading
from typing import List, Dict, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json
import math

logger = logging.getLogger(__name__)


class CompressionLevel(Enum):
    """压缩级别"""
    NONE = 0       # 不压缩
    LIGHT = 1      # 轻度压缩
    MEDIUM = 2     # 中度压缩
    AGGRESSIVE = 3 # 激进压缩
    MAXIMUM = 4    # 最大压缩


class CompressionStrategy(Enum):
    """压缩策略"""
    TRUNCATION = "truncation"           # 截断
    SUMMARIZATION = "summarization"     # 摘要
    KEYWORD_EXTRACTION = "keyword"      # 关键词提取
    SEMANTIC_COMPRESSION = "semantic"   # 语义压缩
    STRUCTURE_PRESERVING = "structure"  # 结构保留
    HYBRID = "hybrid"                   # 混合策略


class ContentType(Enum):
    """内容类型"""
    NARRATIVE = "narrative"     # 叙述性
    TECHNICAL = "technical"     # 技术性
    CONVERSATIONAL = "conversational"  # 对话性
    STRUCTURED = "structured"   # 结构化
    CODE = "code"               # 代码
    MIXED = "mixed"             # 混合


@dataclass
class CompressionResult:
    """压缩结果"""
    original_content: str
    compressed_content: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    strategy: CompressionStrategy
    level: CompressionLevel
    preserved_elements: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    
    @property
    def tokens_saved(self) -> int:
        return self.original_tokens - self.compressed_tokens
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "compression_ratio": self.compression_ratio,
            "strategy": self.strategy.value,
            "level": self.level.value,
            "tokens_saved": self.tokens_saved,
            "processing_time": self.processing_time
        }


@dataclass
class CompressionConfig:
    """压缩配置"""
    # 目标压缩比
    target_ratio: float = 0.5
    
    # Token限制
    max_input_tokens: int = 8000
    max_output_tokens: int = 4000
    min_output_tokens: int = 100
    
    # 策略选择
    enable_truncation: bool = True
    enable_summarization: bool = True
    enable_keyword_extraction: bool = True
    enable_semantic: bool = True
    
    # 质量控制
    quality_threshold: float = 0.7
    preserve_structure: bool = True
    preserve_numbers: bool = True
    preserve_entities: bool = True
    
    # 性能配置
    max_processing_time: float = 5.0
    cache_results: bool = True
    cache_size: int = 1000


@dataclass
class ContentAnalysis:
    """内容分析结果"""
    content_type: ContentType
    token_count: int
    sentence_count: int
    paragraph_count: int
    has_code: bool
    has_numbers: bool
    has_lists: bool
    has_tables: bool
    entity_count: int
    keyword_density: float
    complexity_score: float
    recommended_strategy: CompressionStrategy
    recommended_level: CompressionLevel


class ContentAnalyzer:
    """内容分析器"""
    
    def __init__(self, tokenizer: Optional[Any] = None):
        self.tokenizer = tokenizer
        self._patterns = self._init_patterns()
    
    def _init_patterns(self) -> Dict[str, re.Pattern]:
        """初始化检测模式"""
        return {
            "code_block": re.compile(r'```[\s\S]*?```|`[^`]+`'),
            "number": re.compile(r'\b\d+(?:\.\d+)?%?\b'),
            "list_item": re.compile(r'^\s*[-*•]\s+|^\s*\d+[.)]\s+', re.MULTILINE),
            "table": re.compile(r'\|.*\|'),
            "sentence": re.compile(r'[.!?。！？]'),
            "paragraph": re.compile(r'\n\s*\n'),
            "entity": re.compile(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*'),
            "url": re.compile(r'https?://\S+'),
            "email": re.compile(r'\b[\w.-]+@[\w.-]+\.\w+\b'),
        }
    
    def analyze(self, content: str) -> ContentAnalysis:
        """分析内容"""
        # 基础统计
        token_count = self._count_tokens(content)
        sentence_count = len(self._patterns["sentence"].findall(content)) + 1
        paragraph_count = len(self._patterns["paragraph"].split(content))
        
        # 特征检测
        has_code = bool(self._patterns["code_block"].search(content))
        has_numbers = bool(self._patterns["number"].search(content))
        has_lists = bool(self._patterns["list_item"].search(content))
        has_tables = bool(self._patterns["table"].search(content))
        
        # 实体检测
        entities = self._patterns["entity"].findall(content)
        entity_count = len(entities)
        
        # 关键词密度
        words = content.lower().split()
        unique_words = set(words)
        keyword_density = len(unique_words) / len(words) if words else 0
        
        # 复杂度评分
        complexity = self._calculate_complexity(
            content, token_count, sentence_count, has_code, has_tables
        )
        
        # 内容类型判断
        content_type = self._determine_content_type(
            content, has_code, has_lists, has_tables, has_numbers
        )
        
        # 推荐策略
        strategy, level = self._recommend_compression(
            content_type, token_count, complexity
        )
        
        return ContentAnalysis(
            content_type=content_type,
            token_count=token_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            has_code=has_code,
            has_numbers=has_numbers,
            has_lists=has_lists,
            has_tables=has_tables,
            entity_count=entity_count,
            keyword_density=keyword_density,
            complexity_score=complexity,
            recommended_strategy=strategy,
            recommended_level=level
        )
    
    def _count_tokens(self, text: str) -> int:
        """计算Token数"""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        
        # 简单估算
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4) + 1
    
    def _calculate_complexity(
        self, 
        content: str, 
        token_count: int, 
        sentence_count: int,
        has_code: bool, 
        has_tables: bool
    ) -> float:
        """计算复杂度评分"""
        score = 0.5
        
        # 长度因素
        if token_count > 2000:
            score += 0.2
        elif token_count < 200:
            score -= 0.2
        
        # 句子长度因素
        avg_sentence_length = token_count / max(sentence_count, 1)
        if avg_sentence_length > 30:
            score += 0.1
        
        # 结构因素
        if has_code:
            score += 0.15
        if has_tables:
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _determine_content_type(
        self,
        content: str,
        has_code: bool,
        has_lists: bool,
        has_tables: bool,
        has_numbers: bool
    ) -> ContentType:
        """确定内容类型"""
        # 代码检测
        if has_code and content.count('```') >= 2:
            return ContentType.CODE
        
        # 结构化检测
        if has_tables or (has_lists and has_numbers):
            return ContentType.STRUCTURED
        
        # 技术性检测
        technical_keywords = ['function', 'class', 'method', 'api', 'algorithm', 
                            '函数', '类', '方法', '算法', '系统', '配置']
        technical_count = sum(1 for kw in technical_keywords if kw.lower() in content.lower())
        if technical_count > 3:
            return ContentType.TECHNICAL
        
        # 对话性检测
        dialog_indicators = ['?', '？', '我', '你', '他', '说', '问', '回答']
        dialog_count = sum(content.count(ind) for ind in dialog_indicators)
        if dialog_count > len(content) / 50:
            return ContentType.CONVERSATIONAL
        
        # 默认叙述性
        return ContentType.NARRATIVE
    
    def _recommend_compression(
        self,
        content_type: ContentType,
        token_count: int,
        complexity: float
    ) -> Tuple[CompressionStrategy, CompressionLevel]:
        """推荐压缩策略"""
        # 根据内容类型选择策略
        if content_type == ContentType.CODE:
            return CompressionStrategy.STRUCTURE_PRESERVING, CompressionLevel.LIGHT
        
        if content_type == ContentType.STRUCTURED:
            return CompressionStrategy.STRUCTURE_PRESERVING, CompressionLevel.MEDIUM
        
        if content_type == ContentType.TECHNICAL:
            return CompressionStrategy.KEYWORD_EXTRACTION, CompressionLevel.MEDIUM
        
        if content_type == ContentType.CONVERSATIONAL:
            return CompressionStrategy.SUMMARIZATION, CompressionLevel.MEDIUM
        
        # 根据Token数量选择级别
        if token_count < 500:
            level = CompressionLevel.LIGHT
        elif token_count < 2000:
            level = CompressionLevel.MEDIUM
        elif token_count < 5000:
            level = CompressionLevel.AGGRESSIVE
        else:
            level = CompressionLevel.MAXIMUM
        
        return CompressionStrategy.HYBRID, level


class TruncationCompressor:
    """截断压缩器"""
    
    def compress(
        self, 
        content: str, 
        target_tokens: int,
        preserve_end: bool = True
    ) -> str:
        """截断压缩"""
        # 按句子分割
        sentences = re.split(r'([.!?。！？]\s*)', content)
        
        result = []
        current_tokens = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            if i + 1 < len(sentences) and re.match(r'[.!?。！？]\s*', sentences[i + 1]):
                sentence += sentences[i + 1]
                i += 2
            else:
                i += 1
            
            # 简单Token估算
            sentence_tokens = len(sentence) // 4
            
            if current_tokens + sentence_tokens <= target_tokens:
                result.append(sentence)
                current_tokens += sentence_tokens
            else:
                break
        
        compressed = ''.join(result)
        
        # 如果需要保留结尾
        if preserve_end and len(compressed) < len(content) * 0.8:
            end_content = content[-int(target_tokens * 0.2 * 4):]
            compressed = compressed[:int(target_tokens * 0.8 * 4)] + "\n...[省略]...\n" + end_content
        
        return compressed


class KeywordExtractor:
    """关键词提取器"""
    
    def __init__(self):
        self._stop_words = self._init_stop_words()
    
    def _init_stop_words(self) -> Set[str]:
        """初始化停用词"""
        return {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            '的', '了', '是', '在', '有', '和', '与', '或', '等', '也',
            '这', '那', '之', '为', '以', '于', '上', '下', '中', '来'
        }
    
    def extract(
        self, 
        content: str, 
        num_keywords: int = 10
    ) -> List[Tuple[str, float]]:
        """提取关键词"""
        # 分词
        words = re.findall(r'\w+', content.lower())
        
        # 过滤停用词
        words = [w for w in words if w not in self._stop_words and len(w) > 1]
        
        # 计算词频
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
        
        # TF-IDF简化版
        total_words = len(words)
        scores = {}
        for word, freq in word_freq.items():
            tf = freq / total_words
            # 简化的IDF（实际应用中应该用更大的语料库）
            idf = math.log(total_words / (freq + 1))
            scores[word] = tf * idf
        
        # 排序
        sorted_keywords = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:num_keywords]
    
    def compress_to_keywords(
        self, 
        content: str, 
        target_tokens: int
    ) -> str:
        """压缩为关键词列表"""
        keywords = self.extract(content, num_keywords=target_tokens // 2)
        
        # 构建关键词摘要
        keyword_list = [kw for kw, score in keywords]
        
        # 提取关键句子
        sentences = re.split(r'[.!?。！？]', content)
        key_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 检查句子是否包含关键词
            sentence_lower = sentence.lower()
            if any(kw in sentence_lower for kw, _ in keywords[:5]):
                key_sentences.append(sentence)
        
        result = "关键词: " + ", ".join(keyword_list) + "\n\n"
        result += "关键内容:\n" + "\n".join(key_sentences[:5])
        
        return result


class SemanticCompressor:
    """语义压缩器"""
    
    def __init__(self, embedding_model: Optional[Any] = None):
        self.embedding_model = embedding_model
    
    def compress(
        self, 
        content: str, 
        target_tokens: int,
        preserve_important: bool = True
    ) -> str:
        """语义压缩"""
        # 分割为句子
        sentences = self._split_sentences(content)
        
        if not sentences:
            return content
        
        # 计算句子重要性
        importance_scores = self._calculate_importance(sentences, content)
        
        # 选择重要句子
        selected = self._select_sentences(
            sentences, 
            importance_scores, 
            target_tokens,
            preserve_important
        )
        
        return ' '.join(selected)
    
    def _split_sentences(self, content: str) -> List[str]:
        """分割句子"""
        # 按标点分割
        sentences = re.split(r'(?<=[.!?。！？])\s+', content)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_importance(
        self, 
        sentences: List[str], 
        full_content: str
    ) -> List[float]:
        """计算句子重要性"""
        scores = []
        
        # 提取全文关键词
        all_words = set(re.findall(r'\w+', full_content.lower()))
        
        for sentence in sentences:
            score = 0.5
            
            # 位置因素（首尾句子更重要）
            idx = sentences.index(sentence)
            if idx == 0:
                score += 0.2
            elif idx == len(sentences) - 1:
                score += 0.15
            
            # 长度因素
            if len(sentence) > 20:
                score += 0.1
            
            # 关键词覆盖
            sentence_words = set(re.findall(r'\w+', sentence.lower()))
            overlap = len(sentence_words & all_words)
            score += min(0.2, overlap * 0.02)
            
            # 数字和实体
            if re.search(r'\d+', sentence):
                score += 0.1
            
            scores.append(score)
        
        return scores
    
    def _select_sentences(
        self,
        sentences: List[str],
        scores: List[float],
        target_tokens: int,
        preserve_important: bool
    ) -> List[str]:
        """选择句子"""
        # 创建句子-分数对
        indexed = list(enumerate(sentences))
        
        # 按重要性排序
        sorted_by_score = sorted(
            indexed, 
            key=lambda x: scores[x[0]], 
            reverse=True
        )
        
        # 选择句子直到达到目标Token数
        selected_indices = set()
        current_tokens = 0
        
        for idx, sentence in sorted_by_score:
            sentence_tokens = len(sentence) // 4
            
            if current_tokens + sentence_tokens <= target_tokens:
                selected_indices.add(idx)
                current_tokens += sentence_tokens
            
            if current_tokens >= target_tokens * 0.9:
                break
        
        # 按原始顺序返回
        result = []
        for idx, sentence in indexed:
            if idx in selected_indices:
                result.append(sentence)
        
        return result


class StructurePreservingCompressor:
    """结构保留压缩器"""
    
    def compress(
        self, 
        content: str, 
        target_tokens: int
    ) -> str:
        """结构保留压缩"""
        # 检测结构元素
        elements = self._detect_structure(content)
        
        # 按结构压缩
        compressed_elements = []
        current_tokens = 0
        
        for elem_type, elem_content in elements:
            elem_tokens = len(elem_content) // 4
            
            if elem_type in ['header', 'code', 'table']:
                # 保留关键结构
                if current_tokens + elem_tokens <= target_tokens:
                    compressed_elements.append(elem_content)
                    current_tokens += elem_tokens
            else:
                # 压缩普通内容
                remaining_tokens = target_tokens - current_tokens
                if remaining_tokens > 50:
                    compressed = self._compress_paragraph(elem_content, remaining_tokens)
                    compressed_elements.append(compressed)
                    current_tokens += len(compressed) // 4
        
        return '\n\n'.join(compressed_elements)
    
    def _detect_structure(self, content: str) -> List[Tuple[str, str]]:
        """检测结构元素"""
        elements = []
        
        # 按段落分割
        paragraphs = re.split(r'\n\s*\n', content)
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # 检测元素类型
            if para.startswith('#'):
                elements.append(('header', para))
            elif para.startswith('```'):
                elements.append(('code', para))
            elif '|' in para and para.count('|') > 2:
                elements.append(('table', para))
            elif re.match(r'^\s*[-*•]\s+', para, re.MULTILINE):
                elements.append(('list', para))
            else:
                elements.append(('paragraph', para))
        
        return elements
    
    def _compress_paragraph(self, content: str, target_tokens: int) -> str:
        """压缩段落"""
        sentences = re.split(r'(?<=[.!?。！？])\s+', content)
        
        if not sentences:
            return content
        
        # 简单截断
        result = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(sentence) // 4
            if current_tokens + sentence_tokens <= target_tokens:
                result.append(sentence)
                current_tokens += sentence_tokens
            else:
                break
        
        return ' '.join(result)


class AdaptiveCompressor:
    """自适应压缩器主类"""
    
    def __init__(
        self,
        config: Optional[CompressionConfig] = None,
        embedding_model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        llm_client: Optional[Any] = None
    ):
        self.config = config or CompressionConfig()
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.llm_client = llm_client
        
        # 初始化分析器
        self.analyzer = ContentAnalyzer(tokenizer)
        
        # 初始化压缩器
        self.truncator = TruncationCompressor()
        self.keyword_extractor = KeywordExtractor()
        self.semantic_compressor = SemanticCompressor(embedding_model)
        self.structure_compressor = StructurePreservingCompressor()
        
        # 缓存
        self._cache: Dict[str, CompressionResult] = {}
        self._lock = threading.Lock()
    
    def compress(
        self,
        content: str,
        target_ratio: Optional[float] = None,
        strategy: Optional[CompressionStrategy] = None,
        level: Optional[CompressionLevel] = None
    ) -> CompressionResult:
        """
        压缩内容
        
        Args:
            content: 原始内容
            target_ratio: 目标压缩比（可选）
            strategy: 指定策略（可选）
            level: 压缩级别（可选）
        
        Returns:
            CompressionResult
        """
        start_time = time.time()
        
        # 检查缓存
        cache_key = self._get_cache_key(content, target_ratio, strategy, level)
        if self.config.cache_results and cache_key in self._cache:
            return self._cache[cache_key]
        
        # 分析内容
        analysis = self.analyzer.analyze(content)
        
        # 确定压缩参数
        target_ratio = target_ratio or self.config.target_ratio
        strategy = strategy or analysis.recommended_strategy
        level = level or analysis.recommended_level
        
        # 计算目标Token数
        target_tokens = max(
            self.config.min_output_tokens,
            min(
                int(analysis.token_count * target_ratio),
                self.config.max_output_tokens
            )
        )
        
        # 执行压缩
        compressed = self._execute_compression(
            content, 
            strategy, 
            level, 
            target_tokens,
            analysis
        )
        
        # 计算结果
        compressed_tokens = self.analyzer._count_tokens(compressed)
        actual_ratio = compressed_tokens / analysis.token_count if analysis.token_count > 0 else 1.0
        
        result = CompressionResult(
            original_content=content,
            compressed_content=compressed,
            original_tokens=analysis.token_count,
            compressed_tokens=compressed_tokens,
            compression_ratio=actual_ratio,
            strategy=strategy,
            level=level,
            metadata={
                "content_type": analysis.content_type.value,
                "complexity": analysis.complexity_score
            },
            processing_time=time.time() - start_time
        )
        
        # 缓存结果
        if self.config.cache_results:
            with self._lock:
                if len(self._cache) >= self.config.cache_size:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[cache_key] = result
        
        return result
    
    def _execute_compression(
        self,
        content: str,
        strategy: CompressionStrategy,
        level: CompressionLevel,
        target_tokens: int,
        analysis: ContentAnalysis
    ) -> str:
        """执行压缩"""
        if level == CompressionLevel.NONE:
            return content
        
        # 根据策略选择压缩方法
        if strategy == CompressionStrategy.TRUNCATION:
            return self.truncator.compress(content, target_tokens)
        
        elif strategy == CompressionStrategy.KEYWORD_EXTRACTION:
            return self.keyword_extractor.compress_to_keywords(content, target_tokens)
        
        elif strategy == CompressionStrategy.SEMANTIC_COMPRESSION:
            return self.semantic_compressor.compress(content, target_tokens)
        
        elif strategy == CompressionStrategy.STRUCTURE_PRESERVING:
            return self.structure_compressor.compress(content, target_tokens)
        
        elif strategy == CompressionStrategy.SUMMARIZATION:
            return self._summarize(content, target_tokens)
        
        elif strategy == CompressionStrategy.HYBRID:
            return self._hybrid_compress(content, target_tokens, analysis)
        
        return content
    
    def _summarize(self, content: str, target_tokens: int) -> str:
        """摘要压缩"""
        if self.llm_client:
            try:
                prompt = f"""请将以下内容压缩为{target_tokens}个token以内的摘要，保留关键信息：

{content[:3000]}"""
                
                response = self.llm_client.generate(prompt, max_tokens=target_tokens)
                return response
            except Exception as e:
                logger.warning(f"摘要生成失败: {e}")
        
        # 回退到语义压缩
        return self.semantic_compressor.compress(content, target_tokens)
    
    def _hybrid_compress(
        self, 
        content: str, 
        target_tokens: int,
        analysis: ContentAnalysis
    ) -> str:
        """混合压缩"""
        # 根据内容特征组合多种策略
        
        # 如果有代码或表格，先保留结构
        if analysis.has_code or analysis.has_tables:
            return self.structure_compressor.compress(content, target_tokens)
        
        # 如果有列表，提取关键词
        if analysis.has_lists:
            keywords = self.keyword_extractor.extract(content, num_keywords=10)
            keyword_summary = "关键词: " + ", ".join([kw for kw, _ in keywords])
            
            remaining_tokens = target_tokens - len(keyword_summary) // 4
            semantic_result = self.semantic_compressor.compress(content, remaining_tokens)
            
            return keyword_summary + "\n\n" + semantic_result
        
        # 默认：语义压缩
        return self.semantic_compressor.compress(content, target_tokens)
    
    def _get_cache_key(
        self,
        content: str,
        target_ratio: Optional[float],
        strategy: Optional[CompressionStrategy],
        level: Optional[CompressionLevel]
    ) -> str:
        """生成缓存键"""
        import hashlib
        content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
        ratio_str = f"{target_ratio:.2f}" if target_ratio else "default"
        strategy_str = strategy.value if strategy else "auto"
        level_str = level.value if level else "auto"
        return f"{content_hash}_{ratio_str}_{strategy_str}_{level_str}"
    
    def batch_compress(
        self,
        contents: List[str],
        target_ratio: Optional[float] = None
    ) -> List[CompressionResult]:
        """批量压缩"""
        return [self.compress(content, target_ratio) for content in contents]
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """获取压缩统计"""
        if not self._cache:
            return {"total_compressions": 0}
        
        results = list(self._cache.values())
        
        return {
            "total_compressions": len(results),
            "avg_compression_ratio": sum(r.compression_ratio for r in results) / len(results),
            "avg_tokens_saved": sum(r.tokens_saved for r in results) / len(results),
            "avg_processing_time": sum(r.processing_time for r in results) / len(results),
            "strategy_distribution": self._count_strategies(results)
        }
    
    def _count_strategies(self, results: List[CompressionResult]) -> Dict[str, int]:
        """统计策略使用分布"""
        counts = defaultdict(int)
        for r in results:
            counts[r.strategy.value] += 1
        return dict(counts)
    
    def clear_cache(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()


# 工厂函数
def create_compressor(
    config: Optional[CompressionConfig] = None,
    embedding_model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    llm_client: Optional[Any] = None
) -> AdaptiveCompressor:
    """创建压缩器"""
    return AdaptiveCompressor(config, embedding_model, tokenizer, llm_client)


# 便捷函数
def compress_text(
    text: str,
    target_ratio: float = 0.5,
    strategy: Optional[str] = None
) -> Tuple[str, float]:
    """
    便捷压缩函数
    
    Returns:
        (压缩后文本, 实际压缩比)
    """
    config = CompressionConfig(target_ratio=target_ratio)
    compressor = AdaptiveCompressor(config)
    
    strat = CompressionStrategy(strategy) if strategy else None
    result = compressor.compress(text, target_ratio, strat)
    
    return result.compressed_content, result.compression_ratio
