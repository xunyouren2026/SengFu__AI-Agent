"""
水印提取 - AI内容水印提取
"""
import re
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .embedding import WatermarkType


@dataclass
class ExtractionResult:
    """提取结果"""
    success: bool
    payload: str
    confidence: float
    bits_extracted: int
    method: str
    raw_bits: str = ""


class WatermarkExtractor:
    """水印提取器"""
    
    def __init__(self, strength: float = 0.1):
        self._strength = strength
    
    def extract_text(self, text: str) -> ExtractionResult:
        """从文本提取水印"""
        zero_width_chars = {
            '\u200b': '0',
            '\u200c': '1',
        }
        
        # 提取零宽字符
        bits = []
        for char in text:
            if char in zero_width_chars:
                bits.append(zero_width_chars[char])
        
        if not bits:
            return ExtractionResult(
                success=False,
                payload="",
                confidence=0.0,
                bits_extracted=0,
                method="zero_width"
            )
        
        # 转换为字符串
        raw_bits = ''.join(bits)
        payload = self._binary_to_string(raw_bits)
        
        # 计算置信度
        confidence = min(1.0, len(bits) / 64)
        
        return ExtractionResult(
            success=True,
            payload=payload,
            confidence=confidence,
            bits_extracted=len(bits),
            method="zero_width",
            raw_bits=raw_bits
        )
    
    def extract_vector(
        self,
        vector: List[float],
        original: Optional[List[float]] = None
    ) -> ExtractionResult:
        """从向量提取水印"""
        bits = []
        strength = self._strength
        
        for i, val in enumerate(vector):
            if original and i < len(original):
                # 有原始数据时比较差异
                diff = val - original[i]
                if abs(diff) > strength * abs(original[i]) * 0.5:
                    bits.append('1' if diff > 0 else '0')
            else:
                # 无原始数据时使用统计方法
                # 假设水印在最低有效位
                int_val = int(val * 10000)
                bits.append(str(int_val & 1))
        
        raw_bits = ''.join(bits[:64])  # 限制位数
        payload = self._binary_to_string(raw_bits)
        
        confidence = self._calculate_confidence(bits)
        
        return ExtractionResult(
            success=len(bits) > 0,
            payload=payload,
            confidence=confidence,
            bits_extracted=len(bits),
            method="lsb_vector",
            raw_bits=raw_bits
        )
    
    def extract_matrix(
        self,
        matrix: List[List[float]],
        original: Optional[List[List[float]]] = None
    ) -> ExtractionResult:
        """从矩阵提取水印"""
        bits = []
        strength = self._strength
        
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                val = matrix[i][j]
                
                if original and i < len(original) and j < len(original[i]):
                    diff = val - original[i][j]
                    if abs(diff) > strength * 0.5:
                        bits.append('1' if diff > 0 else '0')
                else:
                    int_val = int(val * 100)
                    bits.append(str(int_val & 1))
        
        raw_bits = ''.join(bits[:64])
        payload = self._binary_to_string(raw_bits)
        
        confidence = self._calculate_confidence(bits)
        
        return ExtractionResult(
            success=len(bits) > 0,
            payload=payload,
            confidence=confidence,
            bits_extracted=len(bits),
            method="lsb_matrix",
            raw_bits=raw_bits
        )
    
    def extract_dct(
        self,
        matrix: List[List[float]]
    ) -> ExtractionResult:
        """使用DCT提取水印"""
        # 应用DCT
        dct_matrix = self._apply_dct(matrix)
        
        # 从中频系数提取
        bits = []
        n = len(matrix)
        mid_start = n // 4
        mid_end = n * 3 // 4
        
        threshold = self._strength * 5
        
        for i in range(mid_start, mid_end):
            for j in range(mid_start, mid_end):
                val = dct_matrix[i][j]
                if abs(val) > threshold:
                    bits.append('1' if val > 0 else '0')
        
        raw_bits = ''.join(bits[:64])
        payload = self._binary_to_string(raw_bits)
        
        confidence = self._calculate_confidence(bits)
        
        return ExtractionResult(
            success=len(bits) > 0,
            payload=payload,
            confidence=confidence,
            bits_extracted=len(bits),
            method="dct",
            raw_bits=raw_bits
        )
    
    def extract_from_model_weights(
        self,
        weights: Dict[str, List[float]],
        original_weights: Optional[Dict[str, List[float]]] = None
    ) -> ExtractionResult:
        """从模型权重提取水印"""
        bits = []
        strength = self._strength
        
        for name, weight in weights.items():
            if original_weights and name in original_weights:
                original = original_weights[name]
                
                for i in range(min(len(weight), len(original))):
                    diff = weight[i] - original[i]
                    if abs(diff) > strength * abs(original[i]) * 0.005:
                        bits.append('1' if diff > 0 else '0')
            else:
                # 使用统计方法
                for val in weight[:100]:
                    int_val = int(val * 10000)
                    bits.append(str(int_val & 1))
        
        raw_bits = ''.join(bits[:64])
        payload = self._binary_to_string(raw_bits)
        
        confidence = self._calculate_confidence(bits)
        
        return ExtractionResult(
            success=len(bits) > 0,
            payload=payload,
            confidence=confidence,
            bits_extracted=len(bits),
            method="model_weights",
            raw_bits=raw_bits
        )
    
    def detect_watermark_presence(
        self,
        data: Any,
        data_type: str = "vector"
    ) -> Tuple[bool, float]:
        """检测是否存在水印"""
        if data_type == "text":
            result = self.extract_text(data)
        elif data_type == "vector":
            result = self.extract_vector(data)
        elif data_type == "matrix":
            result = self.extract_matrix(data)
        else:
            return False, 0.0
        
        return result.success, result.confidence
    
    def _apply_dct(self, matrix: List[List[float]]) -> List[List[float]]:
        """应用DCT变换"""
        n = len(matrix)
        result = [[0.0] * n for _ in range(n)]
        
        for u in range(n):
            for v in range(n):
                sum_val = 0.0
                for i in range(n):
                    for j in range(n):
                        sum_val += matrix[i][j] * \
                            math.cos((2 * i + 1) * u * math.pi / (2 * n)) * \
                            math.cos((2 * j + 1) * v * math.pi / (2 * n))
                
                cu = 1 / math.sqrt(n) if u == 0 else math.sqrt(2 / n)
                cv = 1 / math.sqrt(n) if v == 0 else math.sqrt(2 / n)
                result[u][v] = cu * cv * sum_val
        
        return result
    
    def _binary_to_string(self, binary: str) -> str:
        """二进制转字符串"""
        result = []
        
        # 确保长度是8的倍数
        binary = binary[:len(binary) - len(binary) % 8]
        
        for i in range(0, len(binary), 8):
            byte = binary[i:i + 8]
            try:
                char = chr(int(byte, 2))
                if char.isprintable():
                    result.append(char)
            except:
                pass
        
        return ''.join(result)
    
    def _calculate_confidence(self, bits: List[str]) -> float:
        """计算置信度"""
        if not bits:
            return 0.0
        
        # 检查位的分布
        ones = bits.count('1')
        zeros = bits.count('0')
        total = len(bits)
        
        # 如果分布均匀，可能是随机噪声
        expected = total / 2
        deviation = abs(ones - expected) / expected if expected > 0 else 0
        
        # 置信度基于偏离程度和数据量
        confidence = min(1.0, (deviation + total / 64) / 2)
        
        return confidence
    
    def compare_watermarks(
        self,
        result1: ExtractionResult,
        result2: ExtractionResult
    ) -> float:
        """比较两个水印的相似度"""
        if not result1.raw_bits or not result2.raw_bits:
            return 0.0
        
        # 比较原始位
        min_len = min(len(result1.raw_bits), len(result2.raw_bits))
        if min_len == 0:
            return 0.0
        
        matches = sum(
            1 for i in range(min_len)
            if result1.raw_bits[i] == result2.raw_bits[i]
        )
        
        return matches / min_len
