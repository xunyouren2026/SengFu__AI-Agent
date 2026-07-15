"""
水印嵌入 - AI内容水印嵌入
"""
import hashlib
import struct
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class WatermarkType(Enum):
    """水印类型"""
    LSB = "lsb"               # 最低有效位
    SPREAD_SPECTRUM = "spread_spectrum"  # 扩频
    DCT = "dct"               # 离散余弦变换
    ECHO = "echo"             # 回声隐藏
    PHASE = "phase"           # 相位编码
    TEXT = "text"             # 文本水印


@dataclass
class WatermarkConfig:
    """水印配置"""
    watermark_type: WatermarkType = WatermarkType.LSB
    strength: float = 0.1
    seed: int = 42
    payload_size: int = 64  # 位


@dataclass
class WatermarkResult:
    """水印结果"""
    success: bool
    watermarked_data: Any
    payload: str
    bits_embedded: int
    method: str


class WatermarkEmbedder:
    """水印嵌入器"""
    
    def __init__(self, config: Optional[WatermarkConfig] = None):
        self._config = config or WatermarkConfig()
    
    def embed_text(
        self,
        text: str,
        payload: str
    ) -> WatermarkResult:
        """在文本中嵌入水印"""
        # 使用零宽字符嵌入
        zero_width_chars = {
            '0': '\u200b',  # 零宽空格
            '1': '\u200c',  # 零宽非连接符
        }
        
        # 将payload转换为二进制
        binary_payload = ''.join(
            format(ord(c), '08b') for c in payload
        )
        
        # 嵌入水印
        watermarked = text
        for i, bit in enumerate(binary_payload):
            if i < len(text):
                # 在字符间插入零宽字符
                pos = min(i + 1, len(watermarked))
                watermarked = (
                    watermarked[:pos] +
                    zero_width_chars[bit] +
                    watermarked[pos:]
                )
        
        return WatermarkResult(
            success=True,
            watermarked_data=watermarked,
            payload=payload,
            bits_embedded=len(binary_payload),
            method="zero_width"
        )
    
    def embed_vector(
        self,
        vector: List[float],
        payload: str
    ) -> WatermarkResult:
        """在向量中嵌入水印"""
        # 将payload转换为二进制
        binary_payload = self._string_to_binary(payload)
        
        # 扩展到向量长度
        watermarked = vector.copy()
        
        # 使用LSB方法
        strength = self._config.strength
        
        for i, bit in enumerate(binary_payload):
            if i >= len(watermarked):
                break
            
            # 在浮点数的最低有效位嵌入
            if bit == '1':
                watermarked[i] += strength * abs(watermarked[i])
            else:
                watermarked[i] -= strength * abs(watermarked[i])
        
        return WatermarkResult(
            success=True,
            watermarked_data=watermarked,
            payload=payload,
            bits_embedded=min(len(binary_payload), len(vector)),
            method="lsb_vector"
        )
    
    def embed_matrix(
        self,
        matrix: List[List[float]],
        payload: str
    ) -> WatermarkResult:
        """在矩阵中嵌入水印（如图像）"""
        binary_payload = self._string_to_binary(payload)
        
        watermarked = [row.copy() for row in matrix]
        strength = self._config.strength
        
        bit_idx = 0
        for i in range(len(watermarked)):
            for j in range(len(watermarked[i])):
                if bit_idx >= len(binary_payload):
                    break
                
                bit = binary_payload[bit_idx]
                if bit == '1':
                    watermarked[i][j] += strength
                else:
                    watermarked[i][j] -= strength
                
                bit_idx += 1
        
        return WatermarkResult(
            success=True,
            watermarked_data=watermarked,
            payload=payload,
            bits_embedded=bit_idx,
            method="lsb_matrix"
        )
    
    def embed_dct(
        self,
        matrix: List[List[float]],
        payload: str
    ) -> WatermarkResult:
        """使用DCT嵌入水印"""
        # 简化的DCT水印
        binary_payload = self._string_to_binary(payload)
        
        # 应用DCT（简化）
        dct_matrix = self._apply_dct(matrix)
        
        # 在中频系数嵌入水印
        watermarked_dct = [row.copy() for row in dct_matrix]
        strength = self._config.strength * 10
        
        mid_start = len(matrix) // 4
        mid_end = len(matrix) * 3 // 4
        
        bit_idx = 0
        for i in range(mid_start, mid_end):
            for j in range(mid_start, mid_end):
                if bit_idx >= len(binary_payload):
                    break
                
                bit = binary_payload[bit_idx]
                if bit == '1':
                    watermarked_dct[i][j] += strength
                else:
                    watermarked_dct[i][j] -= strength
                
                bit_idx += 1
        
        # 逆DCT
        watermarked = self._apply_idct(watermarked_dct)
        
        return WatermarkResult(
            success=True,
            watermarked_data=watermarked,
            payload=payload,
            bits_embedded=bit_idx,
            method="dct"
        )
    
    def _apply_dct(self, matrix: List[List[float]]) -> List[List[float]]:
        """应用DCT变换（简化）"""
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
    
    def _apply_idct(self, dct_matrix: List[List[float]]) -> List[List[float]]:
        """应用逆DCT变换"""
        n = len(dct_matrix)
        result = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                sum_val = 0.0
                for u in range(n):
                    for v in range(n):
                        cu = 1 / math.sqrt(n) if u == 0 else math.sqrt(2 / n)
                        cv = 1 / math.sqrt(n) if v == 0 else math.sqrt(2 / n)
                        sum_val += cu * cv * dct_matrix[u][v] * \
                            math.cos((2 * i + 1) * u * math.pi / (2 * n)) * \
                            math.cos((2 * j + 1) * v * math.pi / (2 * n))
                
                result[i][j] = sum_val
        
        return result
    
    def _string_to_binary(self, s: str) -> str:
        """字符串转二进制"""
        return ''.join(format(ord(c), '08b') for c in s)
    
    def generate_payload(
        self,
        owner: str,
        timestamp: float = None,
        extra: str = ""
    ) -> str:
        """生成水印载荷"""
        import time
        ts = timestamp or time.time()
        
        data = f"{owner}:{ts}:{extra}"
        hash_val = hashlib.sha256(data.encode()).hexdigest()[:16]
        
        return f"{owner[:8]}:{hash_val}"
    
    def embed_in_model_weights(
        self,
        weights: Dict[str, List[float]],
        payload: str
    ) -> Dict[str, List[float]]:
        """在模型权重中嵌入水印"""
        watermarked_weights = {}
        
        binary_payload = self._string_to_binary(payload)
        bit_idx = 0
        strength = self._config.strength
        
        for name, weight in weights.items():
            watermarked = weight.copy()
            
            for i in range(len(watermarked)):
                if bit_idx >= len(binary_payload):
                    break
                
                bit = binary_payload[bit_idx]
                if bit == '1':
                    watermarked[i] += strength * abs(watermarked[i]) * 0.01
                else:
                    watermarked[i] -= strength * abs(watermarked[i]) * 0.01
                
                bit_idx += 1
            
            watermarked_weights[name] = watermarked
        
        return watermarked_weights
