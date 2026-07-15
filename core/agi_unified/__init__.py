"""
AGI统一模块
===========

基于统一核心的AGI系统实现。

核心组件：
- agi_memory: AGI记忆系统（基于UnifiedMemoryBank）
- agi_attention: AGI注意力（基于UnifiedSlidingWindowAttention）
- agi_reasoning: AGI推理（基于UnifiedMoE）

重构原则：
- 保持原有API不变
- 内部使用统一核心
- 纯Python标准库
"""

from .agi_memory import AGIMemorySystem
from .agi_attention import AGIAttentionSystem
from .agi_reasoning import AGIReasoningSystem

__all__ = [
    'AGIMemorySystem',
    'AGIAttentionSystem',
    'AGIReasoningSystem',
]
