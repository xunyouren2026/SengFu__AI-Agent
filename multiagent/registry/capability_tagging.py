"""
能力标签体系
标准化标签如skill:python, role:architect
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class CapabilityCategory(Enum):
    """能力标签类别"""
    SKILL = "skill"           # 技能：如python, golang, ml
    ROLE = "role"             # 角色：如architect, worker, coordinator
    DOMAIN = "domain"         # 领域：如nlp, cv, robotics
    RESOURCE = "resource"     # 资源：如gpu, memory, disk
    ENVIRONMENT = "env"       # 环境：如linux, docker, k8s
    VERSION = "version"       # 版本：如v1, v2, stable
    CUSTOM = "custom"         # 自定义标签


@dataclass(frozen=True)
class CapabilityTag:
    """
    能力标签
    
    格式: category:value[:weight]
    例如: skill:python:1.0, role:worker, domain:nlp
    """
    category: str
    value: str
    weight: float = 1.0

    def __str__(self) -> str:
        if self.weight != 1.0:
            return f"{self.category}:{self.value}:{self.weight}"
        return f"{self.category}:{self.value}"

    def __hash__(self) -> int:
        return hash((self.category, self.value))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CapabilityTag):
            return False
        return self.category == other.category and self.value == other.value

    @classmethod
    def parse(cls, tag_str: str) -> CapabilityTag:
        """
        解析标签字符串
        
        Args:
            tag_str: 标签字符串，如 "skill:python:1.0"
            
        Returns:
            CapabilityTag对象
            
        Raises:
            ValueError: 格式错误
        """
        parts = tag_str.split(':')
        if len(parts) < 2:
            raise ValueError(f"Invalid tag format: {tag_str}")
        
        category = parts[0].lower()
        value = parts[1].lower()
        weight = 1.0
        
        if len(parts) >= 3:
            try:
                weight = float(parts[2])
            except ValueError:
                raise ValueError(f"Invalid weight in tag: {tag_str}")
        
        return cls(category=category, value=value, weight=weight)

    def matches(self, other: CapabilityTag) -> bool:
        """检查是否匹配（忽略weight）"""
        return self.category == other.category and self.value == other.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "value": self.value,
            "weight": self.weight
        }


class CapabilityTagSet:
    """
    能力标签集合
    
    管理一组能力标签，支持标签的增删改查和匹配
    """

    def __init__(self, tags: Optional[Set[Union[str, CapabilityTag]]] = None):
        """
        初始化标签集合
        
        Args:
            tags: 初始标签集合
        """
        self._tags: Dict[str, Set[CapabilityTag]] = {}  # category -> set of tags
        self._all_tags: Set[CapabilityTag] = set()
        
        if tags:
            for tag in tags:
                self.add(tag)

    def add(self, tag: Union[str, CapabilityTag]) -> CapabilityTag:
        """
        添加标签
        
        Args:
            tag: 标签字符串或对象
            
        Returns:
            添加的标签对象
        """
        if isinstance(tag, str):
            tag = CapabilityTag.parse(tag)
        
        # 移除同类别同值的旧标签
        self.remove(tag.category, tag.value)
        
        if tag.category not in self._tags:
            self._tags[tag.category] = set()
        
        self._tags[tag.category].add(tag)
        self._all_tags.add(tag)
        return tag

    def remove(self, category: str, value: Optional[str] = None) -> bool:
        """
        移除标签
        
        Args:
            category: 标签类别
            value: 标签值（None则移除该类别所有标签）
            
        Returns:
            是否成功移除
        """
        if category not in self._tags:
            return False
        
        if value is None:
            # 移除整个类别
            for tag in list(self._tags[category]):
                self._all_tags.discard(tag)
            del self._tags[category]
            return True
        
        # 移除特定标签
        to_remove = None
        for tag in self._tags[category]:
            if tag.value == value.lower():
                to_remove = tag
                break
        
        if to_remove:
            self._tags[category].discard(to_remove)
            self._all_tags.discard(to_remove)
            if not self._tags[category]:
                del self._tags[category]
            return True
        
        return False

    def has(self, tag: Union[str, CapabilityTag]) -> bool:
        """
        检查是否包含标签
        
        Args:
            tag: 标签字符串或对象
            
        Returns:
            是否包含
        """
        if isinstance(tag, str):
            try:
                tag = CapabilityTag.parse(tag)
            except ValueError:
                return False
        
        return tag in self._all_tags

    def has_category(self, category: str) -> bool:
        """检查是否包含某类别的标签"""
        return category in self._tags

    def get_by_category(self, category: str) -> Set[CapabilityTag]:
        """获取某类别的所有标签"""
        return self._tags.get(category, set()).copy()

    def get_all(self) -> Set[CapabilityTag]:
        """获取所有标签"""
        return self._all_tags.copy()

    def get_categories(self) -> Set[str]:
        """获取所有类别"""
        return set(self._tags.keys())

    def matches_requirements(self, required: CapabilityTagSet) -> Tuple[bool, float]:
        """
        检查是否满足要求
        
        Args:
            required: 要求的标签集合
            
        Returns:
            (是否满足, 匹配分数)
        """
        if not required._all_tags:
            return True, 1.0
        
        total_weight = 0.0
        matched_weight = 0.0
        
        for req_tag in required._all_tags:
            total_weight += req_tag.weight
            if self.has(req_tag):
                matched_weight += req_tag.weight
        
        if total_weight == 0:
            return True, 1.0
        
        score = matched_weight / total_weight
        return score >= 1.0, score

    def calculate_similarity(self, other: CapabilityTagSet) -> float:
        """
        计算与另一标签集的相似度
        
        使用Jaccard相似度：交集大小 / 并集大小
        
        Args:
            other: 另一标签集
            
        Returns:
            相似度分数 [0, 1]
        """
        intersection = self._all_tags & other._all_tags
        union = self._all_tags | other._all_tags
        
        if not union:
            return 1.0
        
        return len(intersection) / len(union)

    def to_strings(self) -> Set[str]:
        """转换为字符串集合"""
        return {str(tag) for tag in self._all_tags}

    def to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """转换为字典"""
        return {
            category: [tag.to_dict() for tag in tags]
            for category, tags in self._tags.items()
        }

    def __len__(self) -> int:
        return len(self._all_tags)

    def __iter__(self):
        return iter(self._all_tags)

    def __contains__(self, tag: Union[str, CapabilityTag]) -> bool:
        return self.has(tag)

    def __str__(self) -> str:
        return ", ".join(sorted(str(tag) for tag in self._all_tags))


class CapabilityTagRegistry:
    """
    能力标签注册表
    
    管理标准化的能力标签，提供标签验证和建议功能
    """

    # 预定义的标准标签
    STANDARD_SKILLS = {
        "python", "golang", "java", "javascript", "typescript",
        "rust", "cpp", "c", "scala", "kotlin",
        "ml", "dl", "nlp", "cv", "rl",
        "sql", "nosql", "redis", "mongodb", "postgresql"
    }

    STANDARD_ROLES = {
        "architect", "developer", "tester", "ops",
        "coordinator", "worker", "manager", "specialist"
    }

    STANDARD_DOMAINS = {
        "ai", "ml", "nlp", "cv", "robotics",
        "web", "mobile", "backend", "frontend",
        "data", "analytics", "security"
    }

    STANDARD_RESOURCES = {
        "cpu", "gpu", "tpu", "memory", "disk", "network"
    }

    STANDARD_ENVIRONMENTS = {
        "linux", "windows", "macos", "docker", "kubernetes",
        "aws", "azure", "gcp", "aliyun"
    }

    def __init__(self):
        self._standard_tags: Dict[str, Set[str]] = {
            CapabilityCategory.SKILL.value: self.STANDARD_SKILLS.copy(),
            CapabilityCategory.ROLE.value: self.STANDARD_ROLES.copy(),
            CapabilityCategory.DOMAIN.value: self.STANDARD_DOMAINS.copy(),
            CapabilityCategory.RESOURCE.value: self.STANDARD_RESOURCES.copy(),
            CapabilityCategory.ENVIRONMENT.value: self.STANDARD_ENVIRONMENTS.copy(),
        }
        
        self._custom_tags: Dict[str, Set[str]] = {}
        self._tag_descriptions: Dict[str, str] = {}

    def register_standard_tag(
        self,
        category: str,
        value: str,
        description: str = ""
    ) -> None:
        """
        注册标准标签
        
        Args:
            category: 标签类别
            value: 标签值
            description: 标签描述
        """
        if category not in self._standard_tags:
            self._standard_tags[category] = set()
        
        self._standard_tags[category].add(value.lower())
        
        if description:
            self._tag_descriptions[f"{category}:{value}"] = description

    def is_standard_tag(self, tag: Union[str, CapabilityTag]) -> bool:
        """
        检查是否为标准标签
        
        Args:
            tag: 标签字符串或对象
            
        Returns:
            是否为标准标签
        """
        if isinstance(tag, str):
            try:
                tag = CapabilityTag.parse(tag)
            except ValueError:
                return False
        
        category_tags = self._standard_tags.get(tag.category)
        if category_tags:
            return tag.value in category_tags
        return False

    def validate_tag(self, tag: Union[str, CapabilityTag]) -> Tuple[bool, Optional[str]]:
        """
        验证标签
        
        Args:
            tag: 标签字符串或对象
            
        Returns:
            (是否有效, 错误信息)
        """
        try:
            if isinstance(tag, str):
                tag = CapabilityTag.parse(tag)
        except ValueError as e:
            return False, str(e)
        
        # 检查类别是否有效
        if tag.category not in self._standard_tags and tag.category != CapabilityCategory.CUSTOM.value:
            return False, f"Unknown category: {tag.category}"
        
        # 检查值格式
        if not re.match(r'^[a-z0-9_\-\.]+$', tag.value):
            return False, f"Invalid value format: {tag.value}"
        
        # 检查权重
        if tag.weight < 0 or tag.weight > 10:
            return False, f"Weight must be between 0 and 10: {tag.weight}"
        
        return True, None

    def suggest_tags(self, prefix: str, category: Optional[str] = None) -> List[str]:
        """
        根据前缀建议标签
        
        Args:
            prefix: 前缀
            category: 限制类别
            
        Returns:
            建议的标签列表
        """
        prefix = prefix.lower()
        suggestions = []
        
        categories = [category] if category else list(self._standard_tags.keys())
        
        for cat in categories:
            if cat in self._standard_tags:
                for value in self._standard_tags[cat]:
                    if value.startswith(prefix):
                        suggestions.append(f"{cat}:{value}")
        
        return sorted(suggestions)

    def get_all_standard_tags(self) -> Dict[str, Set[str]]:
        """获取所有标准标签"""
        return {k: v.copy() for k, v in self._standard_tags.items()}

    def get_tag_description(self, category: str, value: str) -> Optional[str]:
        """获取标签描述"""
        return self._tag_descriptions.get(f"{category}:{value}")


class CapabilityMatcher:
    """
    能力匹配器
    
    提供高级的标签匹配算法
    """

    def __init__(self, registry: Optional[CapabilityTagRegistry] = None):
        self._registry = registry or CapabilityTagRegistry()

    def match(
        self,
        candidate: CapabilityTagSet,
        requirements: CapabilityTagSet,
        strategy: str = "all"
    ) -> Tuple[bool, float]:
        """
        匹配候选标签与要求
        
        Args:
            candidate: 候选标签集
            requirements: 要求的标签集
            strategy: 匹配策略 (all, any, weighted)
            
        Returns:
            (是否匹配, 匹配分数)
        """
        if strategy == "all":
            return candidate.matches_requirements(requirements)
        
        elif strategy == "any":
            for req_tag in requirements:
                if candidate.has(req_tag):
                    return True, 1.0
            return False, 0.0
        
        elif strategy == "weighted":
            return candidate.matches_requirements(requirements)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def rank_candidates(
        self,
        candidates: List[CapabilityTagSet],
        requirements: CapabilityTagSet
    ) -> List[Tuple[CapabilityTagSet, float]]:
        """
        对候选者进行排名
        
        Args:
            candidates: 候选标签集列表
            requirements: 要求的标签集
            
        Returns:
            按匹配分数排序的列表
        """
        ranked = []
        for candidate in candidates:
            _, score = candidate.matches_requirements(requirements)
            ranked.append((candidate, score))
        
        return sorted(ranked, key=lambda x: x[1], reverse=True)

    def find_best_match(
        self,
        candidates: List[CapabilityTagSet],
        requirements: CapabilityTagSet
    ) -> Optional[Tuple[CapabilityTagSet, float]]:
        """
        找到最佳匹配
        
        Args:
            candidates: 候选标签集列表
            requirements: 要求的标签集
            
        Returns:
            最佳匹配及分数，无匹配则返回None
        """
        ranked = self.rank_candidates(candidates, requirements)
        if ranked and ranked[0][1] > 0:
            return ranked[0]
        return None


def create_skill_tag(skill: str, weight: float = 1.0) -> str:
    """创建技能标签"""
    return f"skill:{skill}:{weight}"


def create_role_tag(role: str, weight: float = 1.0) -> str:
    """创建角色标签"""
    return f"role:{role}:{weight}"


def create_domain_tag(domain: str, weight: float = 1.0) -> str:
    """创建领域标签"""
    return f"domain:{domain}:{weight}"


def create_resource_tag(resource: str, weight: float = 1.0) -> str:
    """创建资源标签"""
    return f"resource:{resource}:{weight}"


def create_env_tag(env: str, weight: float = 1.0) -> str:
    """创建环境标签"""
    return f"env:{env}:{weight}"
