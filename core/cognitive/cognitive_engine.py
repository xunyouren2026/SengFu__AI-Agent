"""
Cognitive System Engine
认知系统引擎

提供记忆、推理、反思、学习等认知能力
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """记忆类型"""
    EPISODIC = "episodic"  # 情景记忆
    SEMANTIC = "semantic"  # 语义记忆
    PROCEDURAL = "procedural"  # 程序记忆
    WORKING = "working"  # 工作记忆


class ReasoningType(Enum):
    """推理类型"""
    DEDUCTIVE = "deductive"  # 演绎推理
    INDUCTIVE = "inductive"  # 归纳推理
    ABDUCTIVE = "abductive"  # 溯因推理
    ANALOGICAL = "analogical"  # 类比推理
    CAUSAL = "causal"  # 因果推理


@dataclass
class Memory:
    """记忆"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    memory_type: MemoryType = MemoryType.EPISODIC
    importance: float = 0.5
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "importance": self.importance,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "metadata": self.metadata,
        }


@dataclass
class ReasoningResult:
    """推理结果"""
    success: bool
    conclusion: str = ""
    reasoning_steps: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE
    premises: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "conclusion": self.conclusion,
            "reasoning_steps": self.reasoning_steps,
            "confidence": self.confidence,
            "reasoning_type": self.reasoning_type.value,
            "premises": self.premises,
            "metadata": self.metadata,
        }


@dataclass
class ReflectionResult:
    """反思结果"""
    success: bool
    insights: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)
    self_critique: str = ""
    action_items: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "insights": self.insights,
            "improvements": self.improvements,
            "self_critique": self.self_critique,
            "action_items": self.action_items,
            "metadata": self.metadata,
        }


class MemorySystem:
    """
    记忆系统
    
    功能：
    - 多类型记忆存储
    - 记忆检索
    - 记忆巩固
    - 遗忘机制
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._memories: Dict[str, Memory] = {}
        self._type_index: Dict[MemoryType, List[str]] = {t: [] for t in MemoryType}
        self._embedding_model = None
        self._initialized = False
        
        # 配置参数
        self._max_memories = self.config.get("max_memories", 10000)
        self._forgetting_threshold = self.config.get("forgetting_threshold", 0.1)
        self._consolidation_interval = self.config.get("consolidation_interval", 3600)
        
    async def initialize(self):
        """初始化记忆系统"""
        if self._initialized:
            return
        
        # 初始化嵌入模型
        try:
            from sentence_transformers import SentenceTransformer
            model_name = self.config.get("embedding_model", "all-MiniLM-L6-v2")
            self._embedding_model = SentenceTransformer(model_name)
            logger.info(f"Embedding model loaded: {model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed, using simple similarity")
        
        self._initialized = True
        logger.info("Memory system initialized")
    
    async def store(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Memory:
        """存储记忆"""
        await self.initialize()
        
        memory = Memory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            metadata=metadata or {},
        )
        
        # 生成嵌入
        if self._embedding_model:
            memory.embedding = self._embedding_model.encode(content).tolist()
        
        self._memories[memory.id] = memory
        self._type_index[memory_type].append(memory.id)
        
        # 检查容量
        if len(self._memories) > self._max_memories:
            await self._forget()
        
        logger.debug(f"Stored memory: {memory.id}")
        
        return memory
    
    async def retrieve(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> List[Memory]:
        """检索记忆"""
        await self.initialize()
        
        # 生成查询嵌入
        query_embedding = None
        if self._embedding_model:
            query_embedding = self._embedding_model.encode(query)
        
        # 筛选候选
        candidates = list(self._memories.values())
        if memory_type:
            candidates = [m for m in candidates if m.memory_type == memory_type]
        
        # 计算相关性
        scored = []
        for memory in candidates:
            score = await self._calculate_relevance(memory, query, query_embedding)
            scored.append((memory, score))
        
        # 排序
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # 返回结果
        results = []
        for memory, score in scored[:limit]:
            memory.last_accessed = time.time()
            memory.access_count += 1
            results.append(memory)
        
        return results
    
    async def _calculate_relevance(
        self,
        memory: Memory,
        query: str,
        query_embedding: Optional[Any] = None,
    ) -> float:
        """计算相关性"""
        # 基础分数
        score = memory.importance
        
        # 时间衰减
        age = time.time() - memory.created_at
        time_decay = 1 / (1 + age / 86400)  # 一天衰减一半
        score *= time_decay
        
        # 访问频率
        access_boost = min(memory.access_count / 10, 0.5)
        score += access_boost
        
        # 语义相似度
        if query_embedding is not None and memory.embedding:
            import numpy as np
            similarity = np.dot(query_embedding, memory.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(memory.embedding)
            )
            score = score * 0.5 + similarity * 0.5
        
        return score
    
    async def _forget(self):
        """遗忘机制"""
        # 计算每个记忆的保留分数
        scored = []
        for memory in self._memories.values():
            score = await self._calculate_retention_score(memory)
            scored.append((memory.id, score))
        
        # 排序并删除低分记忆
        scored.sort(key=lambda x: x[1])
        
        to_forget = int(len(self._memories) * 0.1)  # 删除10%
        
        for memory_id, _ in scored[:to_forget]:
            memory = self._memories[memory_id]
            self._type_index[memory.memory_type].remove(memory_id)
            del self._memories[memory_id]
        
        logger.info(f"Forgot {to_forget} memories")
    
    async def _calculate_retention_score(self, memory: Memory) -> float:
        """计算保留分数"""
        score = memory.importance
        
        # 时间因素
        age = time.time() - memory.created_at
        recency = 1 / (1 + age / 86400)
        score *= recency
        
        # 访问因素
        access_score = min(memory.access_count / 5, 1.0)
        score += access_score * 0.3
        
        return score
    
    async def consolidate(self):
        """记忆巩固"""
        # 将工作记忆转化为长期记忆
        working_memories = [
            m for m in self._memories.values()
            if m.memory_type == MemoryType.WORKING
        ]
        
        for memory in working_memories:
            if memory.importance > 0.7:
                # 转化为情景记忆
                memory.memory_type = MemoryType.EPISODIC
                self._type_index[MemoryType.WORKING].remove(memory.id)
                self._type_index[MemoryType.EPISODIC].append(memory.id)
        
        logger.info(f"Consolidated {len(working_memories)} working memories")
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "total_memories": len(self._memories),
            "by_type": {
                t.value: len(ids) for t, ids in self._type_index.items()
            },
        }
        return stats


class ReasoningEngine:
    """
    推理引擎
    
    功能：
    - 多种推理类型
    - 链式推理
    - 思维树
    - 自我反思
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._llm_client = None
        self._initialized = False
        
    async def initialize(self):
        """初始化推理引擎"""
        if self._initialized:
            return
        
        # 初始化LLM客户端
        try:
            from openai import AsyncOpenAI
            
            api_key = self.config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
            if api_key:
                self._llm_client = AsyncOpenAI(api_key=api_key)
                logger.info("LLM client initialized for reasoning")
        except ImportError:
            logger.warning("openai not installed")
        
        self._initialized = True
        logger.info("Reasoning engine initialized")
    
    async def reason(
        self,
        query: str,
        reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE,
        premises: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> ReasoningResult:
        """执行推理"""
        await self.initialize()
        
        if self._llm_client:
            return await self._llm_reason(query, reasoning_type, premises, context)
        else:
            return await self._mock_reason(query, reasoning_type, premises)
    
    async def _llm_reason(
        self,
        query: str,
        reasoning_type: ReasoningType,
        premises: Optional[List[str]],
        context: Optional[str],
    ) -> ReasoningResult:
        """使用LLM进行推理"""
        # 构建提示
        type_prompts = {
            ReasoningType.DEDUCTIVE: "使用演绎推理，从一般原则推导具体结论",
            ReasoningType.INDUCTIVE: "使用归纳推理，从具体例子总结一般规律",
            ReasoningType.ABDUCTIVE: "使用溯因推理，找出最可能的解释",
            ReasoningType.ANALOGICAL: "使用类比推理，通过相似性得出结论",
            ReasoningType.CAUSAL: "使用因果推理，分析因果关系",
        }
        
        prompt = f"""{type_prompts.get(reasoning_type, "分析并推理")}

问题: {query}

"""
        
        if premises:
            prompt += "前提:\n" + "\n".join(f"- {p}" for p in premises) + "\n\n"
        
        if context:
            prompt += f"背景: {context}\n\n"
        
        prompt += """请按以下格式回答:
1. 分析步骤
2. 推理过程
3. 结论
4. 置信度(0-1)
"""
        
        try:
            response = await self._llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            
            content = response.choices[0].message.content
            
            # 解析结果
            return ReasoningResult(
                success=True,
                conclusion=content,
                reasoning_type=reasoning_type,
                premises=premises or [],
                confidence=0.8,
            )
            
        except Exception as e:
            logger.error(f"LLM reasoning failed: {e}")
            return ReasoningResult(
                success=False,
                conclusion="",
                reasoning_type=reasoning_type,
            )
    
    async def _mock_reason(
        self,
        query: str,
        reasoning_type: ReasoningType,
        premises: Optional[List[str]],
    ) -> ReasoningResult:
        """模拟推理"""
        return ReasoningResult(
            success=True,
            conclusion=f"[Mock] Based on {reasoning_type.value} reasoning: {query[:50]}...",
            reasoning_type=reasoning_type,
            premises=premises or [],
            confidence=0.5,
        )
    
    async def chain_of_thought(
        self,
        query: str,
        max_steps: int = 5,
    ) -> ReasoningResult:
        """链式推理"""
        steps = []
        
        for i in range(max_steps):
            step_result = await self.reason(
                f"Step {i+1}: {query}",
                ReasoningType.DEDUCTIVE,
            )
            steps.append(step_result.conclusion)
            
            if "conclusion" in step_result.conclusion.lower():
                break
        
        return ReasoningResult(
            success=True,
            conclusion=steps[-1] if steps else "",
            reasoning_steps=steps,
            reasoning_type=ReasoningType.DEDUCTIVE,
        )
    
    async def tree_of_thought(
        self,
        query: str,
        branches: int = 3,
        depth: int = 2,
    ) -> ReasoningResult:
        """思维树推理"""
        # TODO: 实现思维树
        return await self.reason(query)


class ReflectionEngine:
    """
    反思引擎
    
    功能：
    - 自我评估
    - 经验总结
    - 改进建议
    - 学习反馈
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._llm_client = None
        self._memory_system: Optional[MemorySystem] = None
        self._initialized = False
        
    async def initialize(self, memory_system: Optional[MemorySystem] = None):
        """初始化反思引擎"""
        if self._initialized:
            return
        
        self._memory_system = memory_system
        
        # 初始化LLM客户端
        try:
            from openai import AsyncOpenAI
            
            api_key = self.config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
            if api_key:
                self._llm_client = AsyncOpenAI(api_key=api_key)
        except ImportError:
            pass
        
        self._initialized = True
        logger.info("Reflection engine initialized")
    
    async def reflect(
        self,
        experience: str,
        context: Optional[str] = None,
    ) -> ReflectionResult:
        """执行反思"""
        await self.initialize()
        
        if self._llm_client:
            return await self._llm_reflect(experience, context)
        else:
            return await self._mock_reflect(experience)
    
    async def _llm_reflect(
        self,
        experience: str,
        context: Optional[str],
    ) -> ReflectionResult:
        """使用LLM进行反思"""
        prompt = f"""请对以下经历进行深度反思:

经历: {experience}

{f"背景: {context}" if context else ""}

请从以下角度进行反思:
1. 洞察: 从这次经历中学到了什么?
2. 改进: 有哪些可以改进的地方?
3. 自我批评: 哪些做得不够好?
4. 行动项: 下次应该怎么做?

请以JSON格式返回结果。
"""
        
        try:
            response = await self._llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            
            content = response.choices[0].message.content
            
            # 解析结果
            import json
            try:
                result = json.loads(content)
            except:
                result = {"insights": [content]}
            
            return ReflectionResult(
                success=True,
                insights=result.get("insights", []),
                improvements=result.get("improvements", []),
                self_critique=result.get("self_critique", ""),
                action_items=result.get("action_items", []),
            )
            
        except Exception as e:
            logger.error(f"LLM reflection failed: {e}")
            return ReflectionResult(success=False)
    
    async def _mock_reflect(self, experience: str) -> ReflectionResult:
        """模拟反思"""
        return ReflectionResult(
            success=True,
            insights=[f"从'{experience[:30]}...'中获得了经验"],
            improvements=["需要更多实践"],
            self_critique="这是一个模拟反思",
            action_items=["继续学习"],
        )
    
    async def learn_from_feedback(
        self,
        action: str,
        outcome: str,
        feedback: str,
    ) -> Dict[str, Any]:
        """从反馈中学习"""
        reflection = await self.reflect(
            f"行动: {action}\n结果: {outcome}\n反馈: {feedback}"
        )
        
        # 存储到记忆系统
        if self._memory_system:
            await self._memory_system.store(
                content=f"行动: {action}, 结果: {outcome}, 反馈: {feedback}",
                memory_type=MemoryType.EPISODIC,
                importance=0.8,
                metadata={"reflection": reflection.to_dict()},
            )
        
        return {
            "reflection": reflection.to_dict(),
            "stored": self._memory_system is not None,
        }


class CognitiveSystem:
    """
    认知系统
    
    整合记忆、推理、反思等认知能力
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._memory = MemorySystem(config)
        self._reasoning = ReasoningEngine(config)
        self._reflection = ReflectionEngine(config)
        self._initialized = False
        
    async def initialize(self):
        """初始化认知系统"""
        if self._initialized:
            return
        
        await self._memory.initialize()
        await self._reasoning.initialize()
        await self._reflection.initialize(self._memory)
        
        self._initialized = True
        logger.info("Cognitive system initialized")
    
    async def think(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """思考"""
        await self.initialize()
        
        # 检索相关记忆
        memories = await self._memory.retrieve(query, limit=5)
        memory_context = "\n".join([m.content for m in memories])
        
        # 推理
        reasoning_result = await self._reasoning.reason(
            query,
            context=f"{context}\n\n相关记忆:\n{memory_context}" if context else memory_context,
        )
        
        return {
            "reasoning": reasoning_result.to_dict(),
            "memories_used": [m.to_dict() for m in memories],
        }
    
    async def learn(
        self,
        experience: str,
        importance: float = 0.5,
    ) -> Memory:
        """学习"""
        await self.initialize()
        
        # 存储记忆
        memory = await self._memory.store(experience, importance=importance)
        
        # 反思
        reflection = await self._reflection.reflect(experience)
        
        # 存储反思结果
        if reflection.insights:
            await self._memory.store(
                f"反思: {'; '.join(reflection.insights)}",
                memory_type=MemoryType.SEMANTIC,
                importance=importance * 0.8,
            )
        
        return memory
    
    async def get_memory_system(self) -> MemorySystem:
        """获取记忆系统"""
        return self._memory
    
    async def get_reasoning_engine(self) -> ReasoningEngine:
        """获取推理引擎"""
        return self._reasoning
    
    async def get_reflection_engine(self) -> ReflectionEngine:
        """获取反思引擎"""
        return self._reflection


# 全局实例
_cognitive_system: Optional[CognitiveSystem] = None


def get_cognitive_system() -> CognitiveSystem:
    """获取全局认知系统"""
    global _cognitive_system
    if _cognitive_system is None:
        _cognitive_system = CognitiveSystem()
    return _cognitive_system


async def init_cognitive_system(config: Optional[Dict[str, Any]] = None):
    """初始化全局认知系统"""
    global _cognitive_system
    _cognitive_system = CognitiveSystem(config)
    await _cognitive_system.initialize()
    return _cognitive_system
