"""
文本嵌入模块

提供多种文本嵌入器实现，包括TF-IDF、词袋模型和哈希嵌入器。
仅使用Python标准库。
"""

import math
import re
import hashlib
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple


# ============================================================
# SentencePieceTokenizer: 简易分词器
# ============================================================

class SentencePieceTokenizer:
    """简易分词器，模拟SentencePiece的子词分词功能。

    支持基本的文本预处理、分词和反分词操作。
    使用基于频率的子词分割策略。
    """

    # 默认的特殊标记
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"

    def __init__(
        self,
        vocab_size: int = 8000,
        max_word_len: int = 50,
        lowercase: bool = True,
    ):
        """初始化分词器。

        Args:
            vocab_size: 词汇表大小
            max_word_len: 最大词长度
            lowercase: 是否转小写
        """
        self._vocab_size = vocab_size
        self._max_word_len = max_word_len
        self._lowercase = lowercase

        # 词汇表: token -> id
        self._token_to_id: Dict[str, int] = {}
        # 反向词汇表: id -> token
        self._id_to_token: Dict[int, str] = {}

        # 特殊标记
        self._special_tokens = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.BOS_TOKEN: 2,
            self.EOS_TOKEN: 3,
        }

        # 子词单元
        self._subword_units: List[str] = []
        # 词频统计
        self._word_counts: Counter = Counter()

        self._initialized = False

    def _pre_tokenize(self, text: str) -> List[str]:
        """预分词：将文本分割为词（基于空白和标点）。

        Args:
            text: 输入文本

        Returns:
            词列表
        """
        if self._lowercase:
            text = text.lower()

        # 在标点符号前后插入空格
        text = re.sub(r'([^\w\s])', r' \1 ', text)

        # 分割为词
        words = text.split()

        # 截断过长的词
        words = [w[:self._max_word_len] for w in words]

        return words

    def _split_word_to_subwords(self, word: str) -> List[str]:
        """将词分割为子词单元。

        使用贪心最长匹配策略。

        Args:
            word: 输入词

        Returns:
            子词列表
        """
        if not self._subword_units:
            return [word]

        subwords = []
        remaining = word

        while remaining:
            # 找到最长的匹配子词
            best_match = ""
            for unit in self._subword_units:
                if remaining.startswith(unit) and len(unit) > len(best_match):
                    best_match = unit

            if best_match:
                subwords.append(best_match)
                remaining = remaining[len(best_match):]
            else:
                # 没有匹配，保留整个剩余部分
                subwords.append(remaining)
                break

        return subwords if subwords else [word]

    def train(self, texts: List[str]) -> None:
        """训练分词器，从语料中学习子词单元。

        Args:
            texts: 训练文本列表
        """
        # 统计词频
        word_counts = Counter()
        for text in texts:
            words = self._pre_tokenize(text)
            word_counts.update(words)

        self._word_counts = word_counts

        # 初始化特殊标记
        idx = 0
        for token, tid in self._special_tokens.items():
            self._token_to_id[token] = tid
            self._id_to_token[tid] = token
            idx = max(idx, tid + 1)

        # 按频率排序词
        sorted_words = word_counts.most_common()

        # 添加高频词作为子词单元
        subword_set = set()
        for word, count in sorted_words:
            if idx >= self._vocab_size:
                break
            if word not in subword_set:
                subword_set.add(word)
                self._token_to_id[word] = idx
                self._id_to_token[idx] = word
                idx += 1

        # 添加常见前缀和后缀子词
        prefix_counts = Counter()
        suffix_counts = Counter()
        for word, count in sorted_words:
            for length in range(2, min(len(word), 8)):
                prefix_counts[word[:length]] += count
                suffix_counts[word[-length:]] += count

        for token, _ in prefix_counts.most_common(self._vocab_size // 4):
            if idx >= self._vocab_size:
                break
            if token not in subword_set:
                subword_set.add(token)
                self._token_to_id[token] = idx
                self._id_to_token[idx] = token
                idx += 1

        for token, _ in suffix_counts.most_common(self._vocab_size // 4):
            if idx >= self._vocab_size:
                break
            if token not in subword_set:
                subword_set.add(token)
                self._token_to_id[token] = idx
                self._id_to_token[idx] = token
                idx += 1

        # 按长度降序排列子词单元（用于贪心匹配）
        self._subword_units = sorted(subword_set, key=len, reverse=True)
        self._initialized = True

    def tokenize(self, text: str) -> List[str]:
        """分词。

        Args:
            text: 输入文本

        Returns:
            token列表
        """
        if not self._initialized:
            # 未训练时使用简单分词
            return self._pre_tokenize(text)

        words = self._pre_tokenize(text)
        tokens = []

        for word in words:
            subwords = self._split_word_to_subwords(word)
            tokens.extend(subwords)

        return tokens

    def tokenize_to_ids(self, text: str) -> List[int]:
        """分词并转换为ID。

        Args:
            text: 输入文本

        Returns:
            token ID列表
        """
        tokens = self.tokenize(text)
        unk_id = self._special_tokens.get(self.UNK_TOKEN, 1)

        return [self._token_to_id.get(t, unk_id) for t in tokens]

    def detokenize(self, tokens: List[str]) -> str:
        """反分词：将token列表还原为文本。

        合并相邻的子词单元，处理##前缀标记。

        Args:
            tokens: token列表

        Returns:
            还原的文本
        """
        if not tokens:
            return ""

        words = []
        current_word = ""

        for token in tokens:
            if token.startswith("##"):
                # 续接子词
                current_word += token[2:]
            else:
                if current_word:
                    words.append(current_word)
                current_word = token

        if current_word:
            words.append(current_word)

        return " ".join(words)

    def detokenize_from_ids(self, ids: List[int]) -> str:
        """从ID列表反分词。

        Args:
            ids: token ID列表

        Returns:
            还原的文本
        """
        tokens = [self._id_to_token.get(i, self.UNK_TOKEN) for i in ids]
        return self.detokenize(tokens)

    @property
    def vocab_size(self) -> int:
        """获取词汇表大小。"""
        return len(self._token_to_id)

    def get_vocab(self) -> Dict[str, int]:
        """获取词汇表。"""
        return dict(self._token_to_id)


# ============================================================
# TextEmbedder: 文本嵌入器抽象基类
# ============================================================

class TextEmbedder(ABC):
    """文本嵌入器抽象基类。"""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """嵌入单条文本。

        Args:
            text: 输入文本

        Returns:
            嵌入向量
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文本。

        Args:
            texts: 输入文本列表

        Returns:
            嵌入向量列表
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """获取嵌入维度。"""
        pass


# ============================================================
# TFIDFEmbedder: TF-IDF嵌入器
# ============================================================

class TFIDFEmbedder(TextEmbedder):
    """TF-IDF嵌入器，纯Python实现。

    支持TF（词频）和IDF（逆文档频率）计算，支持n-gram。
    """

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_df: float = 1.0,
        lowercase: bool = True,
        normalize: bool = True,
    ):
        """初始化TF-IDF嵌入器。

        Args:
            max_features: 最大特征数
            ngram_range: n-gram范围 (min_n, max_n)
            min_df: 最小文档频率
            max_df: 最大文档频率（比例）
            lowercase: 是否转小写
            normalize: 是否L2归一化
        """
        self._max_features = max_features
        self._ngram_range = ngram_range
        self._min_df = min_df
        self._max_df = max_df
        self._lowercase = lowercase
        self._normalize = normalize

        # 词汇表: term -> index
        self._vocabulary: Dict[str, int] = {}
        # IDF值: term -> idf
        self._idf: Dict[str, float] = {}
        # 文档数量
        self._n_docs = 0
        # 文档频率: term -> count
        self._df: Dict[str, int] = {}

        self._fitted = False

    def _preprocess(self, text: str) -> str:
        """文本预处理。"""
        if self._lowercase:
            text = text.lower()
        # 移除特殊字符，保留字母数字和空格
        text = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff\s]', ' ', text)
        # 合并多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _tokenize(self, text: str) -> List[str]:
        """分词。"""
        text = self._preprocess(text)
        return text.split()

    def _extract_ngrams(self, tokens: List[str]) -> List[str]:
        """提取n-gram特征。

        Args:
            tokens: 词列表

        Returns:
            n-gram列表
        """
        ngrams = []
        min_n, max_n = self._ngram_range

        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i:i + n])
                ngrams.append(ngram)

        return ngrams

    def _compute_tf(self, terms: List[str]) -> Dict[str, float]:
        """计算词频（TF）。

        使用对数归一化: tf = 1 + log(tf_raw)

        Args:
            terms: 项列表

        Returns:
            {term: tf_value}
        """
        counts = Counter(terms)
        tf = {}
        for term, count in counts.items():
            if count > 0:
                tf[term] = 1.0 + math.log(count)
            else:
                tf[term] = 0.0
        return tf

    def _compute_idf(self) -> None:
        """计算逆文档频率（IDF）。

        使用平滑IDF: idf = log((1 + n) / (1 + df)) + 1
        """
        self._idf = {}
        for term, df in self._df.items():
            self._idf[term] = math.log(
                (1.0 + self._n_docs) / (1.0 + df)
            ) + 1.0

    def fit(self, corpus: List[str]) -> "TFIDFEmbedder":
        """从语料拟合词汇表和IDF值。

        Args:
            corpus: 文档列表

        Returns:
            self
        """
        self._n_docs = len(corpus)
        self._df.clear()

        # 统计文档频率
        for doc in corpus:
            tokens = self._tokenize(doc)
            terms = self._extract_ngrams(tokens)
            unique_terms = set(terms)
            for term in unique_terms:
                self._df[term] = self._df.get(term, 0) + 1

        # 过滤低频和高频词
        min_count = self._min_df
        max_count = int(self._max_df * self._n_docs) if self._max_df <= 1.0 else self._max_df

        filtered_terms = {
            term: df for term, df in self._df.items()
            if df >= min_count and df <= max_count
        }

        # 按文档频率排序，选择top特征
        sorted_terms = sorted(filtered_terms.items(), key=lambda x: x[1], reverse=True)

        # 构建词汇表
        self._vocabulary.clear()
        for idx, (term, _) in enumerate(sorted_terms[:self._max_features]):
            self._vocabulary[term] = idx

        # 更新df为过滤后的
        self._df = {term: filtered_terms[term] for term in self._vocabulary}

        # 计算IDF
        self._compute_idf()

        self._fitted = True
        return self

    def transform(self, text: str) -> List[float]:
        """将文本转换为TF-IDF向量。

        Args:
            text: 输入文本

        Returns:
            TF-IDF向量
        """
        if not self._fitted:
            raise RuntimeError("嵌入器尚未拟合，请先调用fit()方法")

        tokens = self._tokenize(text)
        terms = self._extract_ngrams(tokens)
        tf = self._compute_tf(terms)

        # 构建TF-IDF向量
        dim = len(self._vocabulary)
        vector = [0.0] * dim

        for term, tf_val in tf.items():
            if term in self._vocabulary:
                idx = self._vocabulary[term]
                vector[idx] = tf_val * self._idf.get(term, 0.0)

        # L2归一化
        if self._normalize:
            norm = math.sqrt(sum(v * v for v in vector))
            if norm > 0:
                vector = [v / norm for v in vector]

        return vector

    def fit_transform(self, corpus: List[str]) -> List[List[float]]:
        """拟合并转换语料。

        Args:
            corpus: 文档列表

        Returns:
            TF-IDF向量列表
        """
        self.fit(corpus)
        return [self.transform(doc) for doc in corpus]

    def embed(self, text: str) -> List[float]:
        """嵌入单条文本。"""
        return self.transform(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文本。"""
        return [self.transform(text) for text in texts]

    @property
    def dimension(self) -> int:
        """获取嵌入维度。"""
        return len(self._vocabulary)

    def get_vocabulary(self) -> Dict[str, int]:
        """获取词汇表。"""
        return dict(self._vocabulary)

    def get_idf(self, term: str) -> float:
        """获取指定词的IDF值。"""
        return self._idf.get(term, 0.0)

    def get_top_terms(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """获取文本中TF-IDF最高的词。

        Args:
            text: 输入文本
            top_k: 返回数量

        Returns:
            [(term, tfidf_score), ...]
        """
        tokens = self._tokenize(text)
        terms = self._extract_ngrams(tokens)
        tf = self._compute_tf(terms)

        scores = []
        for term, tf_val in tf.items():
            if term in self._vocabulary:
                score = tf_val * self._idf.get(term, 0.0)
                scores.append((term, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ============================================================
# BagOfWordsEmbedder: 词袋嵌入器
# ============================================================

class BagOfWordsEmbedder(TextEmbedder):
    """词袋嵌入器。

    将文本表示为词频向量，忽略词序。
    """

    def __init__(
        self,
        max_features: int = 10000,
        lowercase: bool = True,
        binary: bool = False,
        normalize: bool = True,
    ):
        """初始化词袋嵌入器。

        Args:
            max_features: 最大特征数
            lowercase: 是否转小写
            binary: 是否使用二值化（0/1）
            normalize: 是否L2归一化
        """
        self._max_features = max_features
        self._lowercase = lowercase
        self._binary = binary
        self._normalize = normalize

        self._vocabulary: Dict[str, int] = {}
        self._fitted = False

    def _tokenize(self, text: str) -> List[str]:
        """分词。"""
        if self._lowercase:
            text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.split()

    def fit(self, corpus: List[str]) -> "BagOfWordsEmbedder":
        """从语料拟合词汇表。

        Args:
            corpus: 文档列表

        Returns:
            self
        """
        word_counts = Counter()
        for doc in corpus:
            tokens = self._tokenize(doc)
            word_counts.update(tokens)

        # 按频率排序
        sorted_words = word_counts.most_common(self._max_features)

        self._vocabulary.clear()
        for idx, (word, _) in enumerate(sorted_words):
            self._vocabulary[word] = idx

        self._fitted = True
        return self

    def transform(self, text: str) -> List[float]:
        """将文本转换为词袋向量。

        Args:
            text: 输入文本

        Returns:
            词袋向量
        """
        if not self._fitted:
            raise RuntimeError("嵌入器尚未拟合，请先调用fit()方法")

        tokens = self._tokenize(text)
        counts = Counter(tokens)

        dim = len(self._vocabulary)
        vector = [0.0] * dim

        for word, count in counts.items():
            if word in self._vocabulary:
                idx = self._vocabulary[word]
                vector[idx] = 1.0 if self._binary else float(count)

        if self._normalize:
            norm = math.sqrt(sum(v * v for v in vector))
            if norm > 0:
                vector = [v / norm for v in vector]

        return vector

    def fit_transform(self, corpus: List[str]) -> List[List[float]]:
        """拟合并转换。"""
        self.fit(corpus)
        return [self.transform(doc) for doc in corpus]

    def embed(self, text: str) -> List[float]:
        """嵌入单条文本。"""
        return self.transform(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文本。"""
        return [self.transform(text) for text in texts]

    @property
    def dimension(self) -> int:
        """获取嵌入维度。"""
        return len(self._vocabulary)


# ============================================================
# HashEmbedder: 哈希嵌入器（特征哈希）
# ============================================================

class HashEmbedder(TextEmbedder):
    """哈希嵌入器（特征哈希/Hashing Trick）。

    使用哈希函数将词映射到固定维度的向量空间。
    不需要预先拟合词汇表，适合流式处理。
    """

    def __init__(
        self,
        dim: int = 1024,
        ngram_range: Tuple[int, int] = (1, 2),
        lowercase: bool = True,
        normalize: bool = True,
        hash_seed: int = 42,
    ):
        """初始化哈希嵌入器。

        Args:
            dim: 输出维度
            ngram_range: n-gram范围
            lowercase: 是否转小写
            normalize: 是否L2归一化
            hash_seed: 哈希种子
        """
        self._dim = dim
        self._ngram_range = ngram_range
        self._lowercase = lowercase
        self._normalize = normalize
        self._hash_seed = hash_seed

    def _tokenize(self, text: str) -> List[str]:
        """分词。"""
        if self._lowercase:
            text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.split()

    def _extract_ngrams(self, tokens: List[str]) -> List[str]:
        """提取n-gram。"""
        ngrams = []
        min_n, max_n = self._ngram_range

        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.append(" ".join(tokens[i:i + n]))

        return ngrams

    def _hash_term(self, term: str) -> int:
        """将词哈希到指定维度。

        使用双哈希策略：一个确定索引，一个确定符号。

        Args:
            term: 输入词

        Returns:
            (index, sign) 索引和符号
        """
        # 确定索引
        h1 = hashlib.md5(
            (term + str(self._hash_seed)).encode("utf-8")
        ).hexdigest()
        index = int(h1[:8], 16) % self._dim

        # 确定符号
        h2 = hashlib.sha256(
            (term + str(self._hash_seed + 1)).encode("utf-8")
        ).hexdigest()
        sign = 1 if int(h2[:8], 16) % 2 == 0 else -1

        return index, sign

    def transform(self, text: str) -> List[float]:
        """将文本转换为哈希向量。

        Args:
            text: 输入文本

        Returns:
            哈希向量
        """
        tokens = self._tokenize(text)
        terms = self._extract_ngrams(tokens)

        vector = [0.0] * self._dim

        for term in terms:
            index, sign = self._hash_term(term)
            vector[index] += sign

        if self._normalize:
            norm = math.sqrt(sum(v * v for v in vector))
            if norm > 0:
                vector = [v / norm for v in vector]

        return vector

    def embed(self, text: str) -> List[float]:
        """嵌入单条文本。"""
        return self.transform(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文本。"""
        return [self.transform(text) for text in texts]

    @property
    def dimension(self) -> int:
        """获取嵌入维度。"""
        return self._dim
