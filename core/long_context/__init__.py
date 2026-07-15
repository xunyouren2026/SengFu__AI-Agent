"""
长上下文/长聊天记忆系统 - 借鉴视频生成长视频技术
================================================

本模块提供了处理超长上下文和多轮对话记忆的完整解决方案。
技术灵感来源于视频生成领域处理长视频的技术。

主要组件:
-----------

1. 记忆系统 (memory_systems.py)
   - ContextMemoryBank: 外部上下文记忆库
   - LightweightContextMemory: 轻量级可训练记忆 (LoRA风格)
   - HierarchicalContextMemory: 分级记忆系统 (短期/中期/长期)
   - AdaptiveContextCompressor: 自适应上下文压缩器
   - MemoryFusionLayer: 记忆融合层

2. 注意力机制 (attention_mechanisms.py)
   - SlidingWindowContextAttention: 滑动窗口注意力+三区缓存
   - DynamicContextRouting: 动态Token路由
   - CalibContextAttention: 预计算稀疏掩码注意力

3. 上下文管理器 (context_manager.py)
   - LongContextManager: 统一管理长上下文
   - ContextChunker: 上下文分块器
   - OverlapFusion: 重叠融合器
   - BoundaryDetector: 边界检测器
   - ProgressiveLoader: 渐进式加载器

4. 聊天记忆 (chat_memory.py)
   - ChatMemorySystem: 多轮对话记忆系统
   - TopicTransitionDetector: 话题切换检测器
   - DialogueSummarizer: 对话摘要器
   - KeyInfoExtractor: 关键信息提取器
   - EmotionTracker: 情感追踪器

使用示例:
---------

    # 长上下文管理
    from agi_unified_framework.core.long_context import LongContextManager
    
    manager = LongContextManager()
    segments = manager.process(long_text)
    context = manager.retrieve(query="相关查询")
    
    # 聊天记忆
    from agi_unified_framework.core.long_context import ChatMemorySystem
    
    chat_memory = ChatMemorySystem(max_history_turns=50)
    chat_memory.add_turn("用户消息", "助手回复")
    summary = chat_memory.generate_summary()
    
    # 记忆系统
    from agi_unified_framework.core.long_context import (
        ContextMemoryBank,
        HierarchicalContextMemory
    )
    
    memory_bank = ContextMemoryBank(chunk_size=512)
    chunk_ids = memory_bank.add_context(long_content)
    results = memory_bank.retrieve(query_embedding)

纯Python实现，仅使用标准库。
"""

# ============================================================================
# 从 memory_systems.py 导出
# ============================================================================
from .memory_systems import (
    # 数据类
    ContextChunk,
    MemorySlot,
    MemoryLevel,
    
    # 记忆系统
    ContextMemoryBank,
    LightweightContextMemory,
    HierarchicalContextMemory,
    AdaptiveContextCompressor,
    MemoryFusionLayer,
    
    # 工具函数
    cosine_similarity,
    euclidean_distance,
    normalize_vector,
    random_vector,
    softmax,
    matmul,
    transpose,
    compute_hash,
)

# ============================================================================
# 从 attention_mechanisms.py 导出
# ============================================================================
from .attention_mechanisms import (
    # 数据类
    CacheZone,
    TokenInfo,
    SparseMaskPattern,
    
    # 注意力机制
    SlidingWindowContextAttention,
    DynamicContextRouting,
    CalibContextAttention,
    LongContextAttentionFactory,
    
    # 工具函数
    stable_softmax,
)

# ============================================================================
# 从 context_manager.py 导出
# ============================================================================
from .context_manager import (
    # 数据类
    BoundaryType,
    ContextSegment,
    ChunkConfig,
    ProgressiveLoadState,
    
    # 管理器
    LongContextManager,
    ContextChunker,
    OverlapFusion,
    BoundaryDetector,
    ProgressiveLoader,
)

# ============================================================================
# 从 chat_memory.py 导出
# ============================================================================
from .chat_memory import (
    # 数据类
    MessageRole,
    TopicTransitionType,
    Message,
    DialogueTurn,
    TopicSegment,
    ConversationSummary,
    
    # 聊天记忆系统
    ChatMemorySystem,
    TopicTransitionDetector,
    DialogueSummarizer,
    KeyInfoExtractor,
    EmotionTracker,
    
    # 工具函数
    extract_keywords,
    analyze_sentiment,
)

# ============================================================================
# 版本信息
# ============================================================================
__version__ = "1.0.0"
__author__ = "AGI Unified Framework"

# ============================================================================
# 模块级文档字符串
# ============================================================================
__all__ = [
    # memory_systems
    'ContextChunk',
    'MemorySlot',
    'MemoryLevel',
    'ContextMemoryBank',
    'LightweightContextMemory',
    'HierarchicalContextMemory',
    'AdaptiveContextCompressor',
    'MemoryFusionLayer',
    
    # attention_mechanisms
    'CacheZone',
    'TokenInfo',
    'SparseMaskPattern',
    'SlidingWindowContextAttention',
    'DynamicContextRouting',
    'CalibContextAttention',
    'LongContextAttentionFactory',
    
    # context_manager
    'BoundaryType',
    'ContextSegment',
    'ChunkConfig',
    'ProgressiveLoadState',
    'LongContextManager',
    'ContextChunker',
    'OverlapFusion',
    'BoundaryDetector',
    'ProgressiveLoader',
    
    # chat_memory
    'MessageRole',
    'TopicTransitionType',
    'Message',
    'DialogueTurn',
    'TopicSegment',
    'ConversationSummary',
    'ChatMemorySystem',
    'TopicTransitionDetector',
    'DialogueSummarizer',
    'KeyInfoExtractor',
    'EmotionTracker',
    
    # 工具函数
    'cosine_similarity',
    'euclidean_distance',
    'normalize_vector',
    'random_vector',
    'softmax',
    'stable_softmax',
    'matmul',
    'transpose',
    'compute_hash',
    'extract_keywords',
    'analyze_sentiment',
]
