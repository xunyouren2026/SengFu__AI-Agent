"""
Creed.Space 客户端 - 浏览和应用共享的宪章

实现宪章市场的搜索、下载、评分、推荐和兼容性检查功能。
所有实现使用纯Python，不依赖任何外部库。
"""

import math
import random
import hashlib
import json
import time
import re
import copy
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


# ==================== 基础数据模型 ====================

class RuleSeverity(Enum):
    """规则严重度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RuleCategory(Enum):
    """规则类别"""
    SAFETY = "safety"
    PRIVACY = "privacy"
    FAIRNESS = "fairness"
    GENERAL = "general"


@dataclass
class ConstitutionalRule:
    """宪章规则"""
    rule_id: str
    description: str
    category: str = "general"
    severity: str = "medium"
    rationale: str = ""
    examples: List[str] = field(default_factory=list)
    counter_examples: List[str] = field(default_factory=list)
    enabled: bool = True

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "description": self.description,
            "category": self.category,
            "severity": self.severity,
            "rationale": self.rationale,
            "examples": self.examples,
            "counter_examples": self.counter_examples,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConstitutionalRule":
        return cls(**d)


@dataclass
class Constitution:
    """宪章 - 一组规则的集合"""
    constitution_id: str
    name: str
    description: str = ""
    version: str = "1.0.0"
    rules: List[ConstitutionalRule] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def add_rule(self, rule: ConstitutionalRule) -> None:
        self.rules.append(rule)
        self.updated_at = time.time()

    def remove_rule(self, rule_id: str) -> bool:
        for i, r in enumerate(self.rules):
            if r.rule_id == rule_id:
                self.rules.pop(i)
                self.updated_at = time.time()
                return True
        return False

    def get_rules_by_category(self, category: str) -> List[ConstitutionalRule]:
        return [r for r in self.rules if r.category == category]

    def get_enabled_rules(self) -> List[ConstitutionalRule]:
        return [r for r in self.rules if r.enabled]

    def to_dict(self) -> dict:
        return {
            "constitution_id": self.constitution_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "rules": [r.to_dict() for r in self.rules],
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Constitution":
        rules = [ConstitutionalRule.from_dict(r) for r in d.get("rules", [])]
        return cls(
            constitution_id=d["constitution_id"],
            name=d["name"],
            description=d.get("description", ""),
            version=d.get("version", "1.0.0"),
            rules=rules,
            metadata=d.get("metadata", {}),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
        )


# ==================== 市场数据模型 ====================

@dataclass
class CreedMetadata:
    """宪章元数据"""
    creed_id: str
    name: str
    author: str
    version: str
    description: str
    category: str  # safety/privacy/fairness/general
    download_count: int = 0
    rating: float = 0.0  # 0-5
    tags: List[str] = field(default_factory=list)
    rule_count: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    compatibility_version: str = "1.0"

    def to_dict(self) -> dict:
        return {
            "creed_id": self.creed_id,
            "name": self.name,
            "author": self.author,
            "version": self.version,
            "description": self.description,
            "category": self.category,
            "download_count": self.download_count,
            "rating": self.rating,
            "tags": self.tags,
            "rule_count": self.rule_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "compatibility_version": self.compatibility_version,
        }


@dataclass
class CreedSearchQuery:
    """宪章搜索查询"""
    keywords: List[str] = field(default_factory=list)
    category: Optional[str] = None
    min_rating: float = 0.0
    sort_by: str = "rating"  # rating/downloads/recent
    limit: int = 20


@dataclass
class CompatibilityReport:
    """兼容性报告"""
    compatible: bool
    conflicts: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    coverage_score: float = 0.0
    merge_suggestion: str = ""


# ==================== CreedMarketClient ====================

class CreedMarketClient:
    """Creed.Space 市场客户端 - 浏览和应用共享的宪章"""

    def __init__(self):
        self._registry: List[CreedMetadata] = self._generate_builtin_registry()
        self._cache: Dict[str, Constitution] = {}
        self._favorites: Set[str] = set()
        self._ratings: Dict[str, List[float]] = defaultdict(list)
        self._precache_builtins()

    def _precache_builtins(self) -> None:
        """预缓存内置宪章"""
        for meta in self._registry:
            if meta.creed_id not in self._cache:
                self._cache[meta.creed_id] = self._generate_builtin_constitution(meta)

    def search(self, query: CreedSearchQuery) -> List[CreedMetadata]:
        """搜索宪章 - 支持关键词匹配、类别过滤和排序"""
        results = []

        for meta in self._registry:
            # 类别过滤
            if query.category and meta.category != query.category:
                continue

            # 最低评分过滤
            if meta.rating < query.min_rating:
                continue

            # 关键词匹配(名称+描述+标签)
            if query.keywords:
                searchable_text = (
                    meta.name.lower() + " " +
                    meta.description.lower() + " " +
                    " ".join(t.lower() for t in meta.tags)
                )
                match_count = sum(
                    1 for kw in query.keywords
                    if kw.lower() in searchable_text
                )
                if match_count == 0:
                    continue
                meta._match_score = match_count / len(query.keywords)
            else:
                meta._match_score = 1.0

            results.append(meta)

        # 排序
        if query.sort_by == "rating":
            results.sort(key=lambda m: (m._match_score, m.rating), reverse=True)
        elif query.sort_by == "downloads":
            results.sort(key=lambda m: (m._match_score, m.download_count), reverse=True)
        elif query.sort_by == "recent":
            results.sort(key=lambda m: (m._match_score, m.updated_at), reverse=True)

        return results[:query.limit]

    def browse(self, category: Optional[str] = None, sort_by: str = "rating") -> List[CreedMetadata]:
        """浏览宪章"""
        query = CreedSearchQuery(category=category, sort_by=sort_by, limit=50)
        return self.search(query)

    def download(self, creed_id: str) -> Constitution:
        """下载宪章(从缓存或生成)"""
        if creed_id in self._cache:
            # 更新下载计数
            for meta in self._registry:
                if meta.creed_id == creed_id:
                    meta.download_count += 1
                    break
            return copy.deepcopy(self._cache[creed_id])

        # 查找元数据
        meta = None
        for m in self._registry:
            if m.creed_id == creed_id:
                meta = m
                break

        if meta is None:
            raise ValueError(f"Creed '{creed_id}' not found in registry")

        constitution = self._generate_builtin_constitution(meta)
        self._cache[creed_id] = constitution
        meta.download_count += 1
        return copy.deepcopy(constitution)

    def rate(self, creed_id: str, rating: float) -> None:
        """为宪章评分"""
        if not 0 <= rating <= 5:
            raise ValueError("Rating must be between 0 and 5")

        for meta in self._registry:
            if meta.creed_id == creed_id:
                self._ratings[creed_id].append(rating)
                ratings = self._ratings[creed_id]
                meta.rating = round(sum(ratings) / len(ratings), 2)
                return

        raise ValueError(f"Creed '{creed_id}' not found")

    def get_recommendations(self, user_preferences: Dict[str, float]) -> List[CreedMetadata]:
        """基于用户偏好和宪章类别的余弦相似度推荐"""
        if not user_preferences:
            return self.get_popular(10)

        # 构建用户偏好向量
        all_categories = ["safety", "privacy", "fairness", "general"]
        user_vector = [user_preferences.get(cat, 0.0) for cat in all_categories]

        scored = []
        for meta in self._registry:
            # 构建宪章类别向量
            creed_vector = [0.0] * len(all_categories)
            cat_idx = all_categories.index(meta.category) if meta.category in all_categories else 3
            creed_vector[cat_idx] = 1.0

            # 标签也可以贡献向量分量
            for tag in meta.tags:
                tag_lower = tag.lower()
                if tag_lower in all_categories:
                    idx = all_categories.index(tag_lower)
                    creed_vector[idx] += 0.3

            sim = self._cosine_similarity(user_vector, creed_vector)
            # 结合评分和相似度
            combined = 0.6 * sim + 0.4 * (meta.rating / 5.0)
            scored.append((meta, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in scored[:10]]

    def publish(self, creed: Constitution, metadata: CreedMetadata) -> str:
        """发布宪章到市场"""
        # 检查是否已存在
        for existing in self._registry:
            if existing.creed_id == metadata.creed_id:
                raise ValueError(f"Creed '{metadata.creed_id}' already exists")

        metadata.rule_count = len(creed.rules)
        self._registry.append(metadata)
        self._cache[metadata.creed_id] = copy.deepcopy(creed)
        return metadata.creed_id

    def add_favorite(self, creed_id: str) -> None:
        """收藏宪章"""
        self._favorites.add(creed_id)

    def remove_favorite(self, creed_id: str) -> None:
        """取消收藏"""
        self._favorites.discard(creed_id)

    def get_favorites(self) -> List[CreedMetadata]:
        """获取收藏列表"""
        return [m for m in self._registry if m.creed_id in self._favorites]

    def get_popular(self, top_n: int = 10) -> List[CreedMetadata]:
        """获取热门宪章"""
        sorted_meta = sorted(self._registry, key=lambda m: m.download_count, reverse=True)
        return sorted_meta[:top_n]

    def get_recent(self, top_n: int = 10) -> List[CreedMetadata]:
        """获取最近更新的宪章"""
        sorted_meta = sorted(self._registry, key=lambda m: m.updated_at, reverse=True)
        return sorted_meta[:top_n]

    def get_stats(self) -> Dict:
        """获取市场统计"""
        categories = defaultdict(int)
        total_downloads = 0
        total_rating = 0.0
        rated_count = 0

        for meta in self._registry:
            categories[meta.category] += 1
            total_downloads += meta.download_count
            if meta.rating > 0:
                total_rating += meta.rating
                rated_count += 1

        avg_rating = round(total_rating / rated_count, 2) if rated_count > 0 else 0.0

        return {
            "total_creeds": len(self._registry),
            "total_downloads": total_downloads,
            "average_rating": avg_rating,
            "category_distribution": dict(categories),
            "total_favorites": len(self._favorites),
            "cached_creeds": len(self._cache),
        }

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """计算余弦相似度"""
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _generate_builtin_registry(self) -> List[CreedMetadata]:
        """生成内置注册表(至少20条)"""
        now = time.time()
        registry = [
            CreedMetadata(
                creed_id="creed_core_safety_01",
                name="Core Safety Constitution",
                author="ConstitutionalAI Team",
                version="2.1.0",
                description="Fundamental safety rules for AI systems, covering harm prevention, self-harm protection, and dangerous content filtering.",
                category="safety",
                download_count=15420,
                rating=4.8,
                tags=["safety", "harm-prevention", "core", "essential"],
                rule_count=12,
                created_at=now - 86400 * 365,
                updated_at=now - 86400 * 5,
                compatibility_version="2.0",
            ),
            CreedMetadata(
                creed_id="creed_privacy_guard_01",
                name="Privacy Guardian",
                author="DataRights Foundation",
                version="1.5.0",
                description="Comprehensive privacy protection rules including PII handling, data minimization, and consent management.",
                category="privacy",
                download_count=11230,
                rating=4.6,
                tags=["privacy", "pii", "gdpr", "data-protection"],
                rule_count=15,
                created_at=now - 86400 * 300,
                updated_at=now - 86400 * 10,
                compatibility_version="1.5",
            ),
            CreedMetadata(
                creed_id="creed_fairness_eq_01",
                name="Fairness & Equality Shield",
                author="EthicsLab",
                version="1.3.0",
                description="Ensures equitable treatment across demographics, prevents bias in outputs and promotes inclusive language.",
                category="fairness",
                download_count=9870,
                rating=4.5,
                tags=["fairness", "bias", "equality", "inclusion"],
                rule_count=10,
                created_at=now - 86400 * 250,
                updated_at=now - 86400 * 15,
                compatibility_version="1.3",
            ),
            CreedMetadata(
                creed_id="creed_general_conduct_01",
                name="General Conduct Guidelines",
                author="AI Governance Board",
                version="3.0.0",
                description="General behavioral guidelines for AI assistants including helpfulness, honesty, and respectful communication.",
                category="general",
                download_count=18500,
                rating=4.7,
                tags=["general", "conduct", "helpful", "honest"],
                rule_count=18,
                created_at=now - 86400 * 400,
                updated_at=now - 86400 * 2,
                compatibility_version="3.0",
            ),
            CreedMetadata(
                creed_id="creed_child_safety_01",
                name="Child Safety Protector",
                author="SafeAI Alliance",
                version="1.8.0",
                description="Specialized rules for protecting minors, including content filtering and age-appropriate interaction guidelines.",
                category="safety",
                download_count=8920,
                rating=4.9,
                tags=["safety", "children", "minor-protection", "coppa"],
                rule_count=14,
                created_at=now - 86400 * 200,
                updated_at=now - 86400 * 7,
                compatibility_version="1.8",
            ),
            CreedMetadata(
                creed_id="creed_medical_ethics_01",
                name="Medical Ethics Framework",
                author="HealthAI Consortium",
                version="1.2.0",
                description="Ethical guidelines for AI in medical contexts, ensuring responsible health information sharing.",
                category="safety",
                download_count=6540,
                rating=4.7,
                tags=["safety", "medical", "health", "ethics"],
                rule_count=11,
                created_at=now - 86400 * 180,
                updated_at=now - 86400 * 20,
                compatibility_version="1.2",
            ),
            CreedMetadata(
                creed_id="creed_financial_fairness_01",
                name="Financial Fairness Rules",
                author="FinEthics Institute",
                version="1.1.0",
                description="Prevents discriminatory practices in financial AI applications and ensures fair lending recommendations.",
                category="fairness",
                download_count=5230,
                rating=4.3,
                tags=["fairness", "financial", "lending", "discrimination"],
                rule_count=9,
                created_at=now - 86400 * 150,
                updated_at=now - 86400 * 30,
                compatibility_version="1.1",
            ),
            CreedMetadata(
                creed_id="creed_data_minimization_01",
                name="Data Minimization Creed",
                author="PrivacyFirst Collective",
                version="1.4.0",
                description="Enforces data minimization principles, ensuring AI systems collect and retain only necessary data.",
                category="privacy",
                download_count=7650,
                rating=4.4,
                tags=["privacy", "data-minimization", "retention", "collection"],
                rule_count=8,
                created_at=now - 86400 * 220,
                updated_at=now - 86400 * 12,
                compatibility_version="1.4",
            ),
            CreedMetadata(
                creed_id="creed_transparency_01",
                name="Transparency & Explainability",
                author="OpenAI Ethics",
                version="2.0.0",
                description="Rules requiring AI systems to be transparent about their capabilities, limitations, and decision processes.",
                category="general",
                download_count=13400,
                rating=4.6,
                tags=["general", "transparency", "explainability", "honesty"],
                rule_count=13,
                created_at=now - 86400 * 280,
                updated_at=now - 86400 * 8,
                compatibility_version="2.0",
            ),
            CreedMetadata(
                creed_id="creed_cybersecurity_01",
                name="Cybersecurity Shield",
                author="CyberDefense Labs",
                version="1.6.0",
                description="Protects against prompt injection, adversarial attacks, and ensures secure AI operation.",
                category="safety",
                download_count=10200,
                rating=4.5,
                tags=["safety", "security", "injection", "adversarial"],
                rule_count=16,
                created_at=now - 86400 * 190,
                updated_at=now - 86400 * 4,
                compatibility_version="1.6",
            ),
            CreedMetadata(
                creed_id="creed_consent_mgmt_01",
                name="Consent Management Framework",
                author="UserRights.org",
                version="1.3.0",
                description="Manages user consent for data processing, model training, and personalized AI interactions.",
                category="privacy",
                download_count=6100,
                rating=4.2,
                tags=["privacy", "consent", "user-rights", "opt-out"],
                rule_count=10,
                created_at=now - 86400 * 160,
                updated_at=now - 86400 * 25,
                compatibility_version="1.3",
            ),
            CreedMetadata(
                creed_id="creed_employment_fairness_01",
                name="Employment Fairness Creed",
                author="Workplace Equity Lab",
                version="1.0.0",
                description="Ensures AI hiring and HR tools do not discriminate based on protected characteristics.",
                category="fairness",
                download_count=4800,
                rating=4.4,
                tags=["fairness", "employment", "hiring", "hr"],
                rule_count=12,
                created_at=now - 86400 * 120,
                updated_at=now - 86400 * 35,
                compatibility_version="1.0",
            ),
            CreedMetadata(
                creed_id="creed_content_moderation_01",
                name="Content Moderation Standard",
                author="TrustAndSafety Co",
                version="2.2.0",
                description="Standard content moderation rules for detecting and handling harmful, offensive, or misleading content.",
                category="safety",
                download_count=16700,
                rating=4.6,
                tags=["safety", "moderation", "content", "toxicity"],
                rule_count=20,
                created_at=now - 86400 * 350,
                updated_at=now - 86400 * 3,
                compatibility_version="2.2",
            ),
            CreedMetadata(
                creed_id="creed_anonymization_01",
                name="Anonymization Best Practices",
                author="AnonTech Research",
                version="1.2.0",
                description="Rules for proper data anonymization and de-identification in AI data pipelines.",
                category="privacy",
                download_count=5900,
                rating=4.3,
                tags=["privacy", "anonymization", "de-identification", "k-anonymity"],
                rule_count=9,
                created_at=now - 86400 * 140,
                updated_at=now - 86400 * 40,
                compatibility_version="1.2",
            ),
            CreedMetadata(
                creed_id="creed_education_fairness_01",
                name="Educational Equity Creed",
                author="EdTech Ethics Board",
                version="1.1.0",
                description="Ensures AI educational tools provide equitable learning opportunities regardless of student background.",
                category="fairness",
                download_count=4200,
                rating=4.5,
                tags=["fairness", "education", "equity", "learning"],
                rule_count=8,
                created_at=now - 86400 * 100,
                updated_at=now - 86400 * 45,
                compatibility_version="1.1",
            ),
            CreedMetadata(
                creed_id="creed_respectful_ai_01",
                name="Respectful AI Interaction",
                author="HumanCentered AI",
                version="1.7.0",
                description="Guidelines for respectful, empathetic, and culturally sensitive AI interactions.",
                category="general",
                download_count=11800,
                rating=4.8,
                tags=["general", "respect", "empathy", "cultural-sensitivity"],
                rule_count=11,
                created_at=now - 86400 * 310,
                updated_at=now - 86400 * 6,
                compatibility_version="1.7",
            ),
            CreedMetadata(
                creed_id="creed_self_harm_01",
                name="Self-Harm Prevention Protocol",
                author="CrisisAI Network",
                version="2.0.0",
                description="Critical rules for detecting and responding to self-harm and suicide-related content.",
                category="safety",
                download_count=9300,
                rating=4.9,
                tags=["safety", "self-harm", "crisis", "prevention"],
                rule_count=7,
                created_at=now - 86400 * 270,
                updated_at=now - 86400 * 9,
                compatibility_version="2.0",
            ),
            CreedMetadata(
                creed_id="creed_surveillance_01",
                name="Anti-Surveillance Creed",
                author="Digital Rights Watch",
                version="1.0.0",
                description="Prevents AI systems from being used for unauthorized surveillance or tracking of individuals.",
                category="privacy",
                download_count=7100,
                rating=4.6,
                tags=["privacy", "surveillance", "tracking", "digital-rights"],
                rule_count=10,
                created_at=now - 86400 * 130,
                updated_at=now - 86400 * 18,
                compatibility_version="1.0",
            ),
            CreedMetadata(
                creed_id="creed_algorithmic_fairness_01",
                name="Algorithmic Fairness Audit",
                author="FairML Initiative",
                version="1.5.0",
                description="Comprehensive algorithmic fairness rules covering disparate impact, equal opportunity, and calibration.",
                category="fairness",
                download_count=8200,
                rating=4.7,
                tags=["fairness", "algorithmic", "disparate-impact", "audit"],
                rule_count=14,
                created_at=now - 86400 * 230,
                updated_at=now - 86400 * 11,
                compatibility_version="1.5",
            ),
            CreedMetadata(
                creed_id="creed_collaboration_01",
                name="Collaborative AI Ethics",
                author="MultiAgent Ethics Lab",
                version="1.2.0",
                description="Ethical guidelines for multi-agent AI systems, ensuring fair collaboration and conflict resolution.",
                category="general",
                download_count=5500,
                rating=4.3,
                tags=["general", "multi-agent", "collaboration", "negotiation"],
                rule_count=9,
                created_at=now - 86400 * 90,
                updated_at=now - 86400 * 22,
                compatibility_version="1.2",
            ),
            CreedMetadata(
                creed_id="creed_environmental_safety_01",
                name="Environmental Safety Creed",
                author="GreenAI Foundation",
                version="1.0.0",
                description="Rules ensuring AI systems do not provide instructions harmful to the environment.",
                category="safety",
                download_count=3800,
                rating=4.1,
                tags=["safety", "environment", "ecology", "green"],
                rule_count=6,
                created_at=now - 86400 * 60,
                updated_at=now - 86400 * 50,
                compatibility_version="1.0",
            ),
            CreedMetadata(
                creed_id="creed_accessibility_01",
                name="Accessibility Inclusion Creed",
                author="A11yAI Project",
                version="1.3.0",
                description="Ensures AI outputs are accessible to users with disabilities and promotes inclusive design.",
                category="fairness",
                download_count=4600,
                rating=4.4,
                tags=["fairness", "accessibility", "disability", "inclusion"],
                rule_count=11,
                created_at=now - 86400 * 110,
                updated_at=now - 86400 * 28,
                compatibility_version="1.3",
            ),
        ]
        return registry

    def _generate_builtin_constitution(self, meta: CreedMetadata) -> Constitution:
        """根据元数据生成内置宪章"""
        cid = meta.creed_id
        name = meta.name
        rules = self._generate_rules_for_creeds(cid)
        return Constitution(
            constitution_id=cid,
            name=name,
            description=meta.description,
            version=meta.version,
            rules=rules,
            metadata={"author": meta.author, "category": meta.category},
        )

    def _generate_rules_for_creeds(self, creed_id: str) -> List[ConstitutionalRule]:
        """根据宪章ID生成对应的规则集"""
        rule_templates = {
            "creed_core_safety_01": [
                ("Do not generate content that encourages or provides instructions for violent acts.", "safety", "critical"),
                ("Refuse to assist with creating weapons, explosives, or harmful chemical agents.", "safety", "critical"),
                ("Do not produce content that promotes terrorism or extremist ideologies.", "safety", "critical"),
                ("Warn users about potentially dangerous activities and suggest safer alternatives.", "safety", "high"),
                ("Do not generate content that sexualizes minors or promotes child exploitation.", "safety", "critical"),
                ("Refuse requests to generate non-consensual intimate content of any person.", "safety", "critical"),
                ("Do not provide instructions for illegal activities that could cause significant harm.", "safety", "high"),
                ("Flag and refuse content that promotes self-harm or suicide.", "safety", "critical"),
                ("Do not assist in creating malware, ransomware, or tools for cyberattacks.", "safety", "high"),
                ("Refuse to generate content designed to harass, bully, or intimidate individuals.", "safety", "high"),
                ("Do not provide instructions for creating dangerous drugs or unregulated substances.", "safety", "high"),
                ("Ensure safety-critical information is accurate and includes appropriate disclaimers.", "safety", "medium"),
            ],
            "creed_privacy_guard_01": [
                ("Never share, store, or process personally identifiable information without explicit consent.", "privacy", "critical"),
                ("Anonymize or pseudonymize all user data before processing or storage.", "privacy", "high"),
                ("Inform users when their data is being collected and for what purpose.", "privacy", "high"),
                ("Do not infer or reveal sensitive personal attributes without user permission.", "privacy", "high"),
                ("Implement data minimization: collect only data necessary for the stated purpose.", "privacy", "medium"),
                ("Provide users with the ability to access, correct, and delete their data.", "privacy", "high"),
                ("Do not share user data with third parties without explicit user consent.", "privacy", "critical"),
                ("Apply purpose limitation: use collected data only for stated purposes.", "privacy", "medium"),
                ("Ensure data retention policies are followed and expired data is deleted.", "privacy", "medium"),
                ("Protect against re-identification of anonymized data.", "privacy", "high"),
                ("Do not use personal data for profiling without informed consent.", "privacy", "high"),
                ("Respect Do Not Track signals and user privacy preferences.", "privacy", "medium"),
                ("Implement appropriate access controls for sensitive personal data.", "privacy", "high"),
                ("Audit data processing activities for privacy compliance regularly.", "privacy", "medium"),
                ("Notify users promptly in case of data breaches involving their information.", "privacy", "critical"),
            ],
            "creed_fairness_eq_01": [
                ("Do not make assumptions or generalizations based on race, ethnicity, or national origin.", "fairness", "critical"),
                ("Ensure recommendations and decisions do not systematically disadvantage any demographic group.", "fairness", "high"),
                ("Use inclusive language that respects all gender identities and expressions.", "fairness", "high"),
                ("Do not reinforce harmful stereotypes about any protected group.", "fairness", "critical"),
                ("Provide equitable quality of service regardless of user characteristics.", "fairness", "high"),
                ("Avoid using features that serve as proxies for protected characteristics.", "fairness", "medium"),
                ("Test outputs for disparate impact across demographic groups.", "fairness", "medium"),
                ("Acknowledge and mitigate historical biases present in training data.", "fairness", "high"),
                ("Ensure accessibility features are available for users with disabilities.", "fairness", "high"),
                ("Do not prioritize one group's interests over another without ethical justification.", "fairness", "medium"),
            ],
            "creed_general_conduct_01": [
                ("Be helpful, truthful, and accurate in all responses.", "general", "high"),
                ("Clearly communicate limitations and uncertainties in your knowledge.", "general", "high"),
                ("Do not pretend to be human or have human-like consciousness.", "general", "medium"),
                ("Respect user autonomy and do not manipulate or coerce decisions.", "general", "high"),
                ("Provide balanced perspectives on controversial topics.", "general", "medium"),
                ("Acknowledge mistakes and correct errors when identified.", "general", "high"),
                ("Do not generate deceptive content or misinformation.", "general", "critical"),
                ("Maintain consistent behavior and do not arbitrarily change personality.", "general", "medium"),
                ("Protect user intellectual property and do not plagiarize content.", "general", "high"),
                ("Support multiple languages and cultural contexts appropriately.", "general", "medium"),
                ("Do not engage in or encourage addictive behaviors.", "general", "high"),
                ("Respect copyright and provide proper attribution for sourced content.", "general", "medium"),
                ("Encourage critical thinking rather than providing definitive answers on complex topics.", "general", "medium"),
                ("Maintain appropriate boundaries in emotional or personal conversations.", "general", "high"),
                ("Do not substitute for professional medical, legal, or financial advice.", "general", "high"),
                ("Report potential misuse or harmful use patterns to appropriate channels.", "general", "medium"),
                ("Support users in making informed decisions by providing relevant information.", "general", "medium"),
                ("Handle ambiguous requests by seeking clarification before acting.", "general", "low"),
            ],
        }

        # 为其他宪章生成通用规则
        default_rules = {
            "safety": [
                ("Identify and mitigate potential safety risks in user requests.", "safety", "high"),
                ("Escalate ambiguous safety concerns rather than making autonomous decisions.", "safety", "medium"),
                ("Maintain a safety-first approach in all interactions.", "safety", "high"),
            ],
            "privacy": [
                ("Apply privacy-by-design principles in all data processing.", "privacy", "high"),
                ("Minimize data exposure and implement need-to-know access.", "privacy", "medium"),
                ("Regularly review and update privacy protection measures.", "privacy", "medium"),
            ],
            "fairness": [
                ("Actively audit for bias in outputs and decision-making processes.", "fairness", "high"),
                ("Ensure equal access and opportunity through all AI-mediated services.", "fairness", "medium"),
                ("Document and address fairness concerns raised by users or stakeholders.", "fairness", "medium"),
            ],
            "general": [
                ("Operate with transparency about AI capabilities and limitations.", "general", "medium"),
                ("Prioritize user well-being in all interactions.", "general", "high"),
                ("Continuously improve based on user feedback and ethical guidelines.", "general", "medium"),
            ],
        }

        rules = []
        if creed_id in rule_templates:
            for desc, cat, sev in rule_templates[creed_id]:
                rid = hashlib.md5(f"{creed_id}:{desc}".encode()).hexdigest()[:12]
                rules.append(ConstitutionalRule(
                    rule_id=rid,
                    description=desc,
                    category=cat,
                    severity=sev,
                    rationale=f"Rule from {creed_id}",
                ))
        else:
            # 从元数据中获取类别
            category = "general"
            for meta in self._registry:
                if meta.creed_id == creed_id:
                    category = meta.category
                    break

            templates = default_rules.get(category, default_rules["general"])
            for desc, cat, sev in templates:
                rid = hashlib.md5(f"{creed_id}:{desc}".encode()).hexdigest()[:12]
                rules.append(ConstitutionalRule(
                    rule_id=rid,
                    description=desc,
                    category=cat,
                    severity=sev,
                    rationale=f"Rule from {creed_id}",
                ))

        return rules


# ==================== CreedCompatibilityChecker ====================

def _get_rule_id(rule) -> str:
    """兼容不同来源的规则ID字段"""
    return getattr(rule, 'rule_id', None) or getattr(rule, 'id', '')

class CreedCompatibilityChecker:
    """宪章兼容性检查器"""

    SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3, "critical": 4}

    def check(self, creed_a: Constitution, creed_b: Constitution) -> CompatibilityReport:
        """检查两个宪章的兼容性"""
        conflicts = []
        warnings = []
        rules_a = getattr(creed_a, 'get_enabled_rules', lambda: getattr(creed_a, 'rules', []))()
        rules_b = getattr(creed_b, 'get_enabled_rules', lambda: getattr(creed_b, 'rules', []))()

        # 1. 规则冲突检测
        conflicts.extend(self._detect_rule_conflicts(rules_a, rules_b))

        # 2. 严重度不一致检测
        warnings.extend(self._detect_severity_inconsistencies(rules_a, rules_b))

        # 3. 类别覆盖互补性分析
        coverage_score = self._analyze_coverage_complementarity(creed_a, creed_b)

        # 4. 重复规则检测
        warnings.extend(self._detect_duplicate_rules(rules_a, rules_b))

        compatible = len([c for c in conflicts if c.get("severity") == "critical"]) == 0

        merge_suggestion = self._generate_merge_suggestion(conflicts, warnings, coverage_score)

        return CompatibilityReport(
            compatible=compatible,
            conflicts=conflicts,
            warnings=warnings,
            coverage_score=coverage_score,
            merge_suggestion=merge_suggestion,
        )

    def _detect_rule_conflicts(
        self, rules_a: List[ConstitutionalRule], rules_b: List[ConstitutionalRule]
    ) -> List[dict]:
        """检测规则冲突"""
        conflicts = []

        # 定义冲突模式对
        conflict_patterns = [
            ("share", "never share"),
            ("provide", "refuse to provide"),
            ("allow", "do not allow"),
            ("generate", "do not generate"),
            ("assist", "refuse to assist"),
            ("disclose", "do not disclose"),
            ("recommend", "do not recommend"),
            ("encourage", "do not encourage"),
        ]

        for rule_a in rules_a:
            for rule_b in rules_b:
                desc_a = rule_a.description.lower()
                desc_b = rule_b.description.lower()

                # 检查是否讨论相似主题但有相反指令
                topic_overlap = self._calculate_topic_overlap(desc_a, desc_b)

                if topic_overlap > 0.5:
                    # 检查是否有相反的动作词
                    has_opposition = False
                    for positive, negative in conflict_patterns:
                        if (positive in desc_a and negative in desc_b) or \
                           (negative in desc_a and positive in desc_b):
                            has_opposition = True
                            break

                    if has_opposition:
                        severity = "high"
                        if rule_a.severity == "critical" or rule_b.severity == "critical":
                            severity = "critical"
                        conflicts.append({
                            "rule_a": _get_rule_id(rule_a),
                            "rule_b": _get_rule_id(rule_b),
                            "type": "direct_conflict",
                            "severity": severity,
                            "description": (
                                f"Rule '{rule_a.description[:60]}...' conflicts with "
                                f"'{rule_b.description[:60]}...'"
                            ),
                        })

        return conflicts

    def _detect_severity_inconsistencies(
        self, rules_a: List[ConstitutionalRule], rules_b: List[ConstitutionalRule]
    ) -> List[str]:
        """检测严重度不一致"""
        warnings = []

        for rule_a in rules_a:
            for rule_b in rules_b:
                topic_overlap = self._calculate_topic_overlap(
                    rule_a.description.lower(), rule_b.description.lower()
                )

                if topic_overlap > 0.6:
                    sev_a = self.SEVERITY_ORDER.get(rule_a.severity, 2)
                    sev_b = self.SEVERITY_ORDER.get(rule_b.severity, 2)

                    if abs(sev_a - sev_b) >= 2:
                        warnings.append(
                            f"Severity inconsistency: '{rule_a.description[:50]}...' "
                            f"({rule_a.severity}) vs '{rule_b.description[:50]}...' "
                            f"({rule_b.severity})"
                        )

        return warnings

    def _analyze_coverage_complementarity(
        self, creed_a: Constitution, creed_b: Constitution
    ) -> float:
        """分析类别覆盖互补性"""
        rules_a = getattr(creed_a, 'get_enabled_rules', lambda: getattr(creed_a, 'rules', []))()
        rules_b = getattr(creed_b, 'get_enabled_rules', lambda: getattr(creed_b, 'rules', []))()
        categories_a = set(r.category for r in rules_a)
        categories_b = set(r.category for r in rules_b)

        all_categories = categories_a | categories_b
        if not all_categories:
            return 0.0

        overlap = categories_a & categories_b
        union = categories_a | categories_b

        # Jaccard相似度衡量重叠，1-重叠衡量互补性
        jaccard = len(overlap) / len(union) if union else 0.0
        complementarity = 1.0 - jaccard

        # 覆盖度: 两个宪章合起来覆盖了多少类别
        coverage = len(union) / 4.0  # 4个主要类别
        coverage = min(coverage, 1.0)

        return round(0.4 * complementarity + 0.6 * coverage, 3)

    def _detect_duplicate_rules(
        self, rules_a: List[ConstitutionalRule], rules_b: List[ConstitutionalRule]
    ) -> List[str]:
        """检测重复规则"""
        warnings = []

        for rule_a in rules_a:
            for rule_b in rules_b:
                if _get_rule_id(rule_a) == _get_rule_id(rule_b):
                    continue

                similarity = self._calculate_text_similarity(
                    rule_a.description, rule_b.description
                )

                if similarity > 0.85:
                    warnings.append(
                        f"Near-duplicate rule: '{rule_a.description[:50]}...' "
                        f"is very similar to '{rule_b.description[:50]}...'"
                    )

        return warnings

    def _calculate_topic_overlap(self, text_a: str, text_b: str) -> float:
        """计算两个文本的主题重叠度"""
        words_a = set(re.findall(r'\b\w+\b', text_a))
        words_b = set(re.findall(r'\b\w+\b', text_b))

        # 移除停用词
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through", "during",
            "before", "after", "above", "below", "between", "out", "off", "over",
            "under", "again", "further", "then", "once", "and", "but", "or",
            "nor", "not", "so", "yet", "both", "either", "neither", "each",
            "every", "all", "any", "few", "more", "most", "other", "some",
            "such", "no", "only", "own", "same", "than", "too", "very",
            "that", "this", "these", "those", "it", "its",
        }

        words_a -= stop_words
        words_b -= stop_words

        if not words_a or not words_b:
            return 0.0

        intersection = words_a & words_b
        union = words_a | words_b

        return len(intersection) / len(union) if union else 0.0

    def _calculate_text_similarity(self, text_a: str, text_b: str) -> float:
        """计算文本相似度(基于词袋Jaccard)"""
        words_a = set(re.findall(r'\b\w+\b', text_a.lower()))
        words_b = set(re.findall(r'\b\w+\b', text_b.lower()))

        if not words_a or not words_b:
            return 0.0

        intersection = words_a & words_b
        union = words_a | words_b

        return len(intersection) / len(union) if union else 0.0

    def _generate_merge_suggestion(
        self, conflicts: List[dict], warnings: List[str], coverage_score: float
    ) -> str:
        """生成合并建议"""
        critical_conflicts = [c for c in conflicts if c.get("severity") == "critical"]
        high_conflicts = [c for c in conflicts if c.get("severity") == "high"]

        if critical_conflicts:
            suggestion = (
                f"MERGE NOT RECOMMENDED: Found {len(critical_conflicts)} critical conflict(s). "
                f"Resolve critical rule conflicts before merging. "
                f"Consider keeping rules from the higher-authority constitution."
            )
        elif high_conflicts:
            suggestion = (
                f"MERGE WITH CAUTION: Found {len(high_conflicts)} high-severity conflict(s). "
                f"Review and resolve conflicts manually. "
                f"Coverage complementarity score: {coverage_score:.2f}."
            )
        elif warnings:
            suggestion = (
                f"MERGE GENERALLY SAFE: Found {len(warnings)} warning(s) but no critical conflicts. "
                f"Review warnings for potential improvements. "
                f"Coverage complementarity score: {coverage_score:.2f}."
            )
        else:
            suggestion = (
                f"MERGE RECOMMENDED: No conflicts detected. "
                f"The constitutions are highly compatible. "
                f"Coverage complementarity score: {coverage_score:.2f}."
            )

        return suggestion

    def suggest_merge(self, creed_a: Constitution, creed_b: Constitution) -> Constitution:
        """建议合并方案"""
        report = self.check(creed_a, creed_b)

        _get_rules = lambda c: getattr(c, 'get_enabled_rules', lambda: getattr(c, 'rules', []))()
        merged_rules = list(_get_rules(creed_a))

        for rule_b in _get_rules(creed_b):
            # 检查是否与已有规则冲突
            is_conflicting = False
            for conflict in report.conflicts:
                if conflict["rule_b"] == _get_rule_id(rule_b):
                    is_conflicting = True
                    break

            if not is_conflicting:
                # 检查是否重复
                is_duplicate = False
                for rule_a in merged_rules:
                    sim = self._calculate_text_similarity(
                        rule_a.description, rule_b.description
                    )
                    if sim > 0.85:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    merged_rules.append(rule_b)

        merged_id = hashlib.md5(
            f"{creed_a.constitution_id}+{creed_b.constitution_id}".encode()
        ).hexdigest()[:16]

        return Constitution(
            constitution_id=f"merged_{merged_id}",
            name=f"Merged: {creed_a.name} + {creed_b.name}",
            description=(
                f"Merged constitution from '{creed_a.name}' and '{creed_b.name}'. "
                f"{report.merge_suggestion}"
            ),
            version="1.0.0",
            rules=merged_rules,
            metadata={
                "source_a": creed_a.constitution_id,
                "source_b": creed_b.constitution_id,
                "compatibility_report": {
                    "compatible": report.compatible,
                    "conflicts_count": len(report.conflicts),
                    "warnings_count": len(report.warnings),
                    "coverage_score": report.coverage_score,
                },
            },
        )
