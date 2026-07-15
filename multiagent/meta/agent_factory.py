"""
动态Agent工厂 - Dynamic Agent Factory

根据任务描述动态生成专用Agent配置和实例。
实现了任务特征提取、Agent模板匹配和配置生成。
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple, TypeVar


class AgentType(Enum):
    """Agent类型"""
    RESEARCH = auto()      # 研究型
    CODE = auto()          # 代码型
    CREATIVE = auto()      # 创意型
    ANALYSIS = auto()      # 分析型
    COORDINATION = auto()  # 协调型
    SPECIALIZED = auto()   # 专用型
    HYBRID = auto()        # 混合型


class CapabilityLevel(Enum):
    """能力等级"""
    NOVICE = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4
    MASTER = 5


@dataclass
class TaskFeature:
    """任务特征"""
    domain: str                          # 领域
    complexity: float                    # 复杂度 0.0-1.0
    required_skills: Set[str]            # 所需技能
    estimated_duration: float            # 估计时长（秒）
    data_intensity: float                # 数据密集度
    creativity_required: float           # 创意需求度
    precision_required: float            # 精确度需求
    collaboration_required: float        # 协作需求度
    keywords: Set[str] = field(default_factory=set)
    
    def to_vector(self) -> List[float]:
        """转换为特征向量"""
        return [
            self.complexity,
            self.data_intensity,
            self.creativity_required,
            self.precision_required,
            self.collaboration_required
        ]


@dataclass
class AgentConfiguration:
    """Agent配置"""
    agent_id: str
    agent_type: AgentType
    name: str
    description: str
    capabilities: Dict[str, CapabilityLevel]
    knowledge_domains: Set[str]
    processing_style: str  # sequential, parallel, adaptive
    memory_config: Dict[str, Any]
    reasoning_depth: int   # 1-5
    communication_style: str  # formal, casual, technical
    specialization_score: float  # 0.0-1.0
    parent_template: Optional[str] = None
    generation_timestamp: float = field(default_factory=time.time)
    version: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.name,
            "name": self.name,
            "description": self.description,
            "capabilities": {k: v.name for k, v in self.capabilities.items()},
            "knowledge_domains": list(self.knowledge_domains),
            "processing_style": self.processing_style,
            "memory_config": self.memory_config,
            "reasoning_depth": self.reasoning_depth,
            "communication_style": self.communication_style,
            "specialization_score": self.specialization_score,
            "parent_template": self.parent_template,
            "generation_timestamp": self.generation_timestamp,
            "version": self.version
        }


@dataclass
class AgentTemplate:
    """Agent模板"""
    template_id: str
    name: str
    agent_type: AgentType
    base_capabilities: Dict[str, CapabilityLevel]
    knowledge_domains: Set[str]
    description_pattern: str
    feature_match_weights: Dict[str, float]
    configuration_generator: Callable[[TaskFeature], Dict[str, Any]]
    
    def calculate_match_score(self, features: TaskFeature) -> float:
        """计算与任务特征的匹配分数"""
        score = 0.0
        total_weight = 0.0
        
        # 技能匹配
        skill_overlap = len(
            self.base_capabilities.keys() & features.required_skills
        )
        skill_weight = self.feature_match_weights.get("skills", 0.3)
        score += (skill_overlap / max(len(features.required_skills), 1)) * skill_weight
        total_weight += skill_weight
        
        # 领域匹配
        domain_weight = self.feature_match_weights.get("domain", 0.3)
        if features.domain in self.knowledge_domains:
            score += domain_weight
        total_weight += domain_weight
        
        # 复杂度匹配
        complexity_weight = self.feature_match_weights.get("complexity", 0.2)
        # 根据Agent类型评估处理复杂度的能力
        complexity_capacity = self._estimate_complexity_capacity()
        complexity_diff = abs(complexity_capacity - features.complexity)
        score += (1.0 - complexity_diff) * complexity_weight
        total_weight += complexity_weight
        
        # 创意需求匹配
        creativity_weight = self.feature_match_weights.get("creativity", 0.2)
        if self.agent_type == AgentType.CREATIVE:
            score += features.creativity_required * creativity_weight
        elif self.agent_type == AgentType.ANALYSIS:
            score += (1.0 - features.creativity_required) * creativity_weight
        total_weight += creativity_weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _estimate_complexity_capacity(self) -> float:
        """估计处理复杂度的能力"""
        if self.agent_type == AgentType.RESEARCH:
            return 0.9
        elif self.agent_type == AgentType.ANALYSIS:
            return 0.8
        elif self.agent_type == AgentType.COORDINATION:
            return 0.7
        elif self.agent_type == AgentType.CODE:
            return 0.75
        elif self.agent_type == AgentType.CREATIVE:
            return 0.6
        return 0.5
    
    def generate_configuration(self, features: TaskFeature, 
                                agent_id: str) -> AgentConfiguration:
        """生成Agent配置"""
        custom_config = self.configuration_generator(features)
        
        # 合并基础能力和任务需求
        merged_capabilities = dict(self.base_capabilities)
        for skill in features.required_skills:
            if skill not in merged_capabilities:
                # 根据任务复杂度确定能力等级
                if features.complexity > 0.8:
                    merged_capabilities[skill] = CapabilityLevel.EXPERT
                elif features.complexity > 0.5:
                    merged_capabilities[skill] = CapabilityLevel.ADVANCED
                else:
                    merged_capabilities[skill] = CapabilityLevel.INTERMEDIATE
        
        # 生成描述
        description = self.description_pattern.format(
            domain=features.domain,
            skills=", ".join(features.required_skills)
        )
        
        return AgentConfiguration(
            agent_id=agent_id,
            agent_type=self.agent_type,
            name=custom_config.get("name", f"{self.name}_{agent_id[:8]}"),
            description=description,
            capabilities=merged_capabilities,
            knowledge_domains=self.knowledge_domains | {features.domain},
            processing_style=custom_config.get("processing_style", "adaptive"),
            memory_config=custom_config.get("memory_config", {"type": "standard"}),
            reasoning_depth=custom_config.get("reasoning_depth", 3),
            communication_style=custom_config.get("communication_style", "adaptive"),
            specialization_score=self.calculate_match_score(features),
            parent_template=self.template_id
        )


class TaskFeatureExtractor:
    """任务特征提取器"""
    
    # 领域关键词映射
    DOMAIN_KEYWORDS: Dict[str, List[str]] = {
        "software": ["code", "program", "software", "development", "api", "database"],
        "data_science": ["data", "analysis", "machine learning", "statistics", "model"],
        "design": ["design", "ui", "ux", "visual", "creative", "art"],
        "research": ["research", "study", "investigate", "explore", "analyze"],
        "writing": ["write", "content", "document", "article", "blog"],
        "coordination": ["coordinate", "manage", "organize", "schedule", "plan"],
        "mathematics": ["math", "calculation", "equation", "formula", "algorithm"],
        "business": ["business", "strategy", "market", "finance", "economy"]
    }
    
    # 技能关键词映射
    SKILL_KEYWORDS: Dict[str, List[str]] = {
        "python": ["python", "py"],
        "javascript": ["javascript", "js", "node.js", "frontend"],
        "machine_learning": ["ml", "machine learning", "deep learning", "neural"],
        "data_analysis": ["pandas", "numpy", "data analysis", "visualization"],
        "nlp": ["nlp", "natural language", "text processing", "sentiment"],
        "cv": ["computer vision", "image processing", "opencv", "vision"],
        "communication": ["communicate", "present", "explain", "document"],
        "problem_solving": ["solve", "optimize", "improve", "debug"],
        "creativity": ["create", "innovate", "design", "brainstorm"]
    }
    
    def __init__(self):
        self._cache: Dict[str, TaskFeature] = {}
        self._cache_lock = threading.RLock()
    
    def extract_features(self, task_description: str) -> TaskFeature:
        """从任务描述中提取特征"""
        # 检查缓存
        task_hash = self._hash_task(task_description)
        with self._cache_lock:
            if task_hash in self._cache:
                return self._cache[task_hash]
        
        # 提取领域
        domain = self._extract_domain(task_description)
        
        # 提取技能
        skills = self._extract_skills(task_description)
        
        # 提取关键词
        keywords = self._extract_keywords(task_description)
        
        # 评估复杂度
        complexity = self._estimate_complexity(task_description)
        
        # 评估各项需求
        data_intensity = self._estimate_data_intensity(task_description, domain)
        creativity_required = self._estimate_creativity_requirement(task_description)
        precision_required = self._estimate_precision_requirement(task_description)
        collaboration_required = self._estimate_collaboration_requirement(task_description)
        
        # 估计时长
        duration = self._estimate_duration(task_description, complexity)
        
        features = TaskFeature(
            domain=domain,
            complexity=complexity,
            required_skills=skills,
            estimated_duration=duration,
            data_intensity=data_intensity,
            creativity_required=creativity_required,
            precision_required=precision_required,
            collaboration_required=collaboration_required,
            keywords=keywords
        )
        
        # 缓存结果
        with self._cache_lock:
            self._cache[task_hash] = features
        
        return features
    
    def _hash_task(self, description: str) -> str:
        """生成任务哈希"""
        return hashlib.md5(description.lower().encode()).hexdigest()
    
    def _extract_domain(self, description: str) -> str:
        """提取领域"""
        description_lower = description.lower()
        domain_scores: Dict[str, int] = {}
        
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in description_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        return "general"
    
    def _extract_skills(self, description: str) -> Set[str]:
        """提取技能"""
        description_lower = description.lower()
        skills = set()
        
        for skill, keywords in self.SKILL_KEYWORDS.items():
            if any(kw in description_lower for kw in keywords):
                skills.add(skill)
        
        return skills
    
    def _extract_keywords(self, description: str) -> Set[str]:
        """提取关键词"""
        # 简单的关键词提取：取名词和动词
        words = re.findall(r'\b[a-zA-Z]{4,}\b', description.lower())
        
        # 过滤常见停用词
        stopwords = {"this", "that", "with", "from", "have", "been", "their", "they"}
        keywords = set(w for w in words if w not in stopwords)
        
        return keywords
    
    def _estimate_complexity(self, description: str) -> float:
        """估计任务复杂度"""
        complexity_indicators = {
            "high": ["complex", "difficult", "challenging", "advanced", "sophisticated", "integrate", "architecture"],
            "medium": ["moderate", "standard", "typical", "regular", "implement"],
            "low": ["simple", "basic", "easy", "straightforward", "minimal"]
        }
        
        description_lower = description.lower()
        
        high_count = sum(1 for w in complexity_indicators["high"] if w in description_lower)
        medium_count = sum(1 for w in complexity_indicators["medium"] if w in description_lower)
        low_count = sum(1 for w in complexity_indicators["low"] if w in description_lower)
        
        # 基于描述长度调整
        length_factor = min(len(description) / 1000, 1.0) * 0.2
        
        if high_count > 0:
            return min(0.9 + length_factor, 1.0)
        elif medium_count > 0:
            return 0.5 + length_factor
        elif low_count > 0:
            return 0.2 + length_factor
        
        # 默认中等复杂度
        return 0.5 + length_factor
    
    def _estimate_data_intensity(self, description: str, domain: str) -> float:
        """估计数据密集度"""
        data_keywords = ["data", "dataset", "database", "large", "big data", "processing", "batch"]
        description_lower = description.lower()
        
        count = sum(1 for kw in data_keywords if kw in description_lower)
        base_score = min(count * 0.2, 1.0)
        
        # 数据科学领域增加权重
        if domain == "data_science":
            base_score = min(base_score + 0.3, 1.0)
        
        return base_score
    
    def _estimate_creativity_requirement(self, description: str) -> float:
        """估计创意需求"""
        creative_keywords = ["creative", "design", "innovative", "novel", "original", "artistic", "brainstorm"]
        description_lower = description.lower()
        
        count = sum(1 for kw in creative_keywords if kw in description_lower)
        return min(count * 0.25, 1.0)
    
    def _estimate_precision_requirement(self, description: str) -> float:
        """估计精确度需求"""
        precision_keywords = ["precise", "accurate", "exact", "correct", "error-free", "verified", "tested"]
        description_lower = description.lower()
        
        count = sum(1 for kw in precision_keywords if kw in description_lower)
        return min(0.5 + count * 0.1, 1.0)
    
    def _estimate_collaboration_requirement(self, description: str) -> float:
        """估计协作需求"""
        collab_keywords = ["team", "collaborate", "coordinate", "communicate", "stakeholder", "review", "feedback"]
        description_lower = description.lower()
        
        count = sum(1 for kw in collab_keywords if kw in description_lower)
        return min(count * 0.2, 1.0)
    
    def _estimate_duration(self, description: str, complexity: float) -> float:
        """估计任务时长（秒）"""
        # 基于复杂度估计
        base_duration = complexity * 3600  # 1小时基础
        
        # 根据关键词调整
        if "quick" in description.lower() or "fast" in description.lower():
            base_duration *= 0.5
        if "extensive" in description.lower() or "comprehensive" in description.lower():
            base_duration *= 2.0
        
        return base_duration


class DynamicAgentFactory:
    """动态Agent工厂"""
    
    def __init__(self):
        self.templates: Dict[str, AgentTemplate] = {}
        self.feature_extractor = TaskFeatureExtractor()
        self.generated_agents: Dict[str, AgentConfiguration] = {}
        self._agent_counter = 0
        self._lock = threading.RLock()
        self._generation_callbacks: List[Callable[[AgentConfiguration], None]] = []
        
        # 注册默认模板
        self._register_default_templates()
    
    def _register_default_templates(self) -> None:
        """注册默认Agent模板"""
        # 研究型Agent模板
        self.register_template(AgentTemplate(
            template_id="research_agent",
            name="ResearchAgent",
            agent_type=AgentType.RESEARCH,
            base_capabilities={
                "information_gathering": CapabilityLevel.EXPERT,
                "synthesis": CapabilityLevel.ADVANCED,
                "critical_thinking": CapabilityLevel.ADVANCED,
                "writing": CapabilityLevel.INTERMEDIATE
            },
            knowledge_domains={"research", "academia", "science"},
            description_pattern="A research-oriented agent specialized in {domain} with expertise in {skills}",
            feature_match_weights={
                "skills": 0.35,
                "domain": 0.25,
                "complexity": 0.25,
                "creativity": 0.15
            },
            configuration_generator=self._generate_research_config
        ))
        
        # 代码型Agent模板
        self.register_template(AgentTemplate(
            template_id="code_agent",
            name="CodeAgent",
            agent_type=AgentType.CODE,
            base_capabilities={
                "programming": CapabilityLevel.EXPERT,
                "debugging": CapabilityLevel.ADVANCED,
                "code_review": CapabilityLevel.ADVANCED,
                "architecture": CapabilityLevel.INTERMEDIATE
            },
            knowledge_domains={"software", "engineering", "technology"},
            description_pattern="A code-focused agent for {domain} development with skills in {skills}",
            feature_match_weights={
                "skills": 0.4,
                "domain": 0.2,
                "complexity": 0.25,
                "creativity": 0.15
            },
            configuration_generator=self._generate_code_config
        ))
        
        # 创意型Agent模板
        self.register_template(AgentTemplate(
            template_id="creative_agent",
            name="CreativeAgent",
            agent_type=AgentType.CREATIVE,
            base_capabilities={
                "ideation": CapabilityLevel.EXPERT,
                "design_thinking": CapabilityLevel.EXPERT,
                "brainstorming": CapabilityLevel.ADVANCED,
                "visualization": CapabilityLevel.INTERMEDIATE
            },
            knowledge_domains={"design", "arts", "media"},
            description_pattern="A creative agent for {domain} projects focusing on {skills}",
            feature_match_weights={
                "skills": 0.3,
                "domain": 0.2,
                "complexity": 0.2,
                "creativity": 0.3
            },
            configuration_generator=self._generate_creative_config
        ))
        
        # 分析型Agent模板
        self.register_template(AgentTemplate(
            template_id="analysis_agent",
            name="AnalysisAgent",
            agent_type=AgentType.ANALYSIS,
            base_capabilities={
                "data_analysis": CapabilityLevel.EXPERT,
                "statistical_reasoning": CapabilityLevel.ADVANCED,
                "pattern_recognition": CapabilityLevel.ADVANCED,
                "visualization": CapabilityLevel.INTERMEDIATE
            },
            knowledge_domains={"data_science", "business", "mathematics"},
            description_pattern="An analytical agent for {domain} with expertise in {skills}",
            feature_match_weights={
                "skills": 0.35,
                "domain": 0.25,
                "complexity": 0.3,
                "creativity": 0.1
            },
            configuration_generator=self._generate_analysis_config
        ))
        
        # 协调型Agent模板
        self.register_template(AgentTemplate(
            template_id="coordination_agent",
            name="CoordinationAgent",
            agent_type=AgentType.COORDINATION,
            base_capabilities={
                "planning": CapabilityLevel.EXPERT,
                "communication": CapabilityLevel.EXPERT,
                "conflict_resolution": CapabilityLevel.ADVANCED,
                "resource_management": CapabilityLevel.INTERMEDIATE
            },
            knowledge_domains={"management", "coordination", "business"},
            description_pattern="A coordination agent for {domain} projects managing {skills}",
            feature_match_weights={
                "skills": 0.25,
                "domain": 0.25,
                "complexity": 0.2,
                "creativity": 0.1
            },
            configuration_generator=self._generate_coordination_config
        ))
    
    def _generate_research_config(self, features: TaskFeature) -> Dict[str, Any]:
        """生成研究型Agent配置"""
        return {
            "name": f"ResearchAgent_{features.domain}",
            "processing_style": "deep",
            "memory_config": {
                "type": "long_term",
                "retrieval_depth": "deep",
                "context_window": 8192
            },
            "reasoning_depth": 5 if features.complexity > 0.7 else 3,
            "communication_style": "academic"
        }
    
    def _generate_code_config(self, features: TaskFeature) -> Dict[str, Any]:
        """生成代码型Agent配置"""
        return {
            "name": f"CodeAgent_{features.domain}",
            "processing_style": "precise",
            "memory_config": {
                "type": "code_optimized",
                "syntax_aware": True,
                "documentation_access": True
            },
            "reasoning_depth": 4 if features.complexity > 0.6 else 3,
            "communication_style": "technical"
        }
    
    def _generate_creative_config(self, features: TaskFeature) -> Dict[str, Any]:
        """生成创意型Agent配置"""
        return {
            "name": f"CreativeAgent_{features.domain}",
            "processing_style": "exploratory",
            "memory_config": {
                "type": "associative",
                "divergent_thinking": True,
                "inspiration_pool": True
            },
            "reasoning_depth": 3,
            "communication_style": "inspirational"
        }
    
    def _generate_analysis_config(self, features: TaskFeature) -> Dict[str, Any]:
        """生成分析型Agent配置"""
        return {
            "name": f"AnalysisAgent_{features.domain}",
            "processing_style": "systematic",
            "memory_config": {
                "type": "structured",
                "data_caching": True,
                "query_optimization": True
            },
            "reasoning_depth": 4,
            "communication_style": "precise"
        }
    
    def _generate_coordination_config(self, features: TaskFeature) -> Dict[str, Any]:
        """生成协调型Agent配置"""
        return {
            "name": f"CoordinationAgent_{features.domain}",
            "processing_style": "adaptive",
            "memory_config": {
                "type": "social",
                "relationship_tracking": True,
                "priority_aware": True
            },
            "reasoning_depth": 3,
            "communication_style": "diplomatic"
        }
    
    def register_template(self, template: AgentTemplate) -> None:
        """注册Agent模板"""
        with self._lock:
            self.templates[template.template_id] = template
    
    def register_callback(self, callback: Callable[[AgentConfiguration], None]) -> None:
        """注册生成回调"""
        with self._lock:
            self._generation_callbacks.append(callback)
    
    def generate_agent(self, task_description: str, 
                       custom_id: Optional[str] = None) -> AgentConfiguration:
        """根据任务描述生成Agent"""
        # 提取任务特征
        features = self.feature_extractor.extract_features(task_description)
        
        # 选择最佳匹配的模板
        best_template = self._select_best_template(features)
        
        # 生成Agent ID
        with self._lock:
            if custom_id:
                agent_id = custom_id
            else:
                self._agent_counter += 1
                agent_id = f"agent_{int(time.time())}_{self._agent_counter}"
        
        # 生成配置
        config = best_template.generate_configuration(features, agent_id)
        
        # 保存配置
        with self._lock:
            self.generated_agents[agent_id] = config
        
        # 触发回调
        for callback in self._generation_callbacks:
            try:
                callback(config)
            except Exception:
                pass
        
        return config
    
    def _select_best_template(self, features: TaskFeature) -> AgentTemplate:
        """选择最佳匹配的模板"""
        with self._lock:
            if not self.templates:
                raise ValueError("No templates registered")
            
            best_template = None
            best_score = -1.0
            
            for template in self.templates.values():
                score = template.calculate_match_score(features)
                if score > best_score:
                    best_score = score
                    best_template = template
            
            return best_template
    
    def generate_hybrid_agent(self, task_descriptions: List[str],
                               custom_id: Optional[str] = None) -> AgentConfiguration:
        """基于多个任务描述生成混合型Agent"""
        # 提取所有特征并合并
        all_features = [self.feature_extractor.extract_features(td) for td in task_descriptions]
        
        # 合并特征
        merged_features = self._merge_features(all_features)
        
        # 选择最佳模板
        best_template = self._select_best_template(merged_features)
        
        # 生成Agent ID
        with self._lock:
            if custom_id:
                agent_id = custom_id
            else:
                self._agent_counter += 1
                agent_id = f"hybrid_agent_{int(time.time())}_{self._agent_counter}"
        
        # 生成基础配置
        config = best_template.generate_configuration(merged_features, agent_id)
        
        # 调整为混合型
        config.agent_type = AgentType.HYBRID
        config.name = f"HybridAgent_{config.name}"
        config.description = f"Hybrid agent combining capabilities for multiple domains: {config.description}"
        config.specialization_score = config.specialization_score * 0.8  # 混合型专业化程度略低
        
        # 保存配置
        with self._lock:
            self.generated_agents[agent_id] = config
        
        return config
    
    def _merge_features(self, features_list: List[TaskFeature]) -> TaskFeature:
        """合并多个任务特征"""
        if not features_list:
            raise ValueError("Empty features list")
        
        if len(features_list) == 1:
            return features_list[0]
        
        # 合并技能
        all_skills: Set[str] = set()
        for f in features_list:
            all_skills.update(f.required_skills)
        
        # 合并关键词
        all_keywords: Set[str] = set()
        for f in features_list:
            all_keywords.update(f.keywords)
        
        # 平均各项数值
        avg_complexity = sum(f.complexity for f in features_list) / len(features_list)
        avg_data = sum(f.data_intensity for f in features_list) / len(features_list)
        avg_creativity = sum(f.creativity_required for f in features_list) / len(features_list)
        avg_precision = sum(f.precision_required for f in features_list) / len(features_list)
        avg_collab = sum(f.collaboration_required for f in features_list) / len(features_list)
        avg_duration = sum(f.estimated_duration for f in features_list) / len(features_list)
        
        # 选择最常见的领域
        domain_counts: Dict[str, int] = {}
        for f in features_list:
            domain_counts[f.domain] = domain_counts.get(f.domain, 0) + 1
        most_common_domain = max(domain_counts, key=domain_counts.get)
        
        return TaskFeature(
            domain=most_common_domain,
            complexity=avg_complexity,
            required_skills=all_skills,
            estimated_duration=avg_duration,
            data_intensity=avg_data,
            creativity_required=avg_creativity,
            precision_required=avg_precision,
            collaboration_required=avg_collab,
            keywords=all_keywords
        )
    
    def get_agent_config(self, agent_id: str) -> Optional[AgentConfiguration]:
        """获取Agent配置"""
        with self._lock:
            return self.generated_agents.get(agent_id)
    
    def list_generated_agents(self) -> List[str]:
        """列出所有生成的Agent ID"""
        with self._lock:
            return list(self.generated_agents.keys())
    
    def get_agent_by_type(self, agent_type: AgentType) -> List[AgentConfiguration]:
        """按类型获取Agent配置"""
        with self._lock:
            return [
                config for config in self.generated_agents.values()
                if config.agent_type == agent_type
            ]
    
    def update_agent_config(self, agent_id: str, 
                            updates: Dict[str, Any]) -> Optional[AgentConfiguration]:
        """更新Agent配置"""
        with self._lock:
            if agent_id not in self.generated_agents:
                return None
            
            config = self.generated_agents[agent_id]
            
            # 应用更新
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            config.version += 1
            return config
    
    def clone_agent_config(self, agent_id: str, 
                           new_id: Optional[str] = None) -> Optional[AgentConfiguration]:
        """克隆Agent配置"""
        with self._lock:
            if agent_id not in self.generated_agents:
                return None
            
            original = self.generated_agents[agent_id]
            
            # 创建新ID
            if new_id is None:
                self._agent_counter += 1
                new_id = f"{agent_id}_clone_{self._agent_counter}"
            
            # 深拷贝配置
            cloned = AgentConfiguration(
                agent_id=new_id,
                agent_type=original.agent_type,
                name=f"{original.name}_clone",
                description=original.description,
                capabilities=dict(original.capabilities),
                knowledge_domains=set(original.knowledge_domains),
                processing_style=original.processing_style,
                memory_config=dict(original.memory_config),
                reasoning_depth=original.reasoning_depth,
                communication_style=original.communication_style,
                specialization_score=original.specialization_score,
                parent_template=original.parent_template,
                version=1
            )
            
            self.generated_agents[new_id] = cloned
            return cloned
    
    def get_factory_statistics(self) -> Dict[str, Any]:
        """获取工厂统计信息"""
        with self._lock:
            type_counts: Dict[str, int] = {}
            for config in self.generated_agents.values():
                type_name = config.agent_type.name
                type_counts[type_name] = type_counts.get(type_name, 0) + 1
            
            return {
                "total_templates": len(self.templates),
                "total_generated_agents": len(self.generated_agents),
                "agents_by_type": type_counts,
                "template_ids": list(self.templates.keys())
            }


# 便捷函数
def quick_create_agent(task_description: str, 
                       agent_type_hint: Optional[AgentType] = None) -> AgentConfiguration:
    """快速创建Agent的便捷函数"""
    factory = DynamicAgentFactory()
    
    if agent_type_hint:
        # 如果提供了类型提示，可以影响模板选择
        pass
    
    return factory.generate_agent(task_description)
