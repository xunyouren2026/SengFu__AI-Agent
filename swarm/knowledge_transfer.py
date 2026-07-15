"""
知识迁移系统 - Knowledge Transfer System

提供知识蒸馏、技能图谱、联邦知识蒸馏和学习路径规划功能。
仅使用Python标准库。
"""

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any
from collections import defaultdict, deque


# ============================================================
# 数据模型
# ============================================================

@dataclass
class KnowledgeEntry:
    """知识条目"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    topic: str = ""
    content: str = ""
    confidence: float = 0.0
    source_agent: str = ""
    timestamp: float = field(default_factory=time.time)
    quality_score: float = 0.0
    tags: List[str] = field(default_factory=list)


@dataclass
class Experience:
    """经验记录"""
    agent_id: str = ""
    task_type: str = ""
    outcome: float = 0.0  # [0, 1]
    lessons: str = ""
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillNode:
    """技能节点"""
    name: str
    level: float = 0.0  # [0, 1]
    prerequisites: List[str] = field(default_factory=list)
    description: str = ""


# ============================================================
# 知识迁移系统
# ============================================================

class KnowledgeTransfer:
    """
    知识迁移系统。

    支持知识蒸馏、经验共享和知识查询。
    """

    def __init__(self):
        # Agent知识库: {agent_id: List[KnowledgeEntry]}
        self._knowledge_bases: Dict[str, List[KnowledgeEntry]] = defaultdict(list)
        # 经验库: {agent_id: List[Experience]}
        self._experience_db: Dict[str, List[Experience]] = defaultdict(list)
        # 蒸馏记录
        self._distillation_log: List[Dict[str, Any]] = []

    def distill_knowledge(
        self,
        teacher_id: str,
        student_id: str,
        topic_filter: Optional[str] = None,
        min_confidence: float = 0.5,
        transfer_ratio: float = 0.7,
    ) -> Dict[str, Any]:
        """
        知识蒸馏。

        从教师Agent蒸馏知识到学生Agent。
        蒸馏过程：
        1. 筛选教师的高置信度知识
        2. 根据主题过滤
        3. 以transfer_ratio比例传递给学生
        4. 学生获得的知识置信度 = 教师置信度 * transfer_ratio

        Args:
            teacher_id: 教师Agent ID
            student_id: 学生Agent ID
            topic_filter: 主题过滤（可选）
            min_confidence: 最低置信度阈值
            transfer_ratio: 知识传递比例

        Returns:
            蒸馏结果
        """
        teacher_knowledge = self._knowledge_bases.get(teacher_id, [])

        # 筛选符合条件的知识
        eligible = []
        for entry in teacher_knowledge:
            if entry.confidence < min_confidence:
                continue
            if topic_filter and topic_filter.lower() not in entry.topic.lower():
                continue
            eligible.append(entry)

        if not eligible:
            return {
                "success": False,
                "reason": "no_eligible_knowledge",
                "teacher_id": teacher_id,
                "student_id": student_id,
                "transferred_count": 0,
            }

        # 蒸馏知识到学生
        transferred = []
        for entry in eligible:
            distilled = KnowledgeEntry(
                topic=entry.topic,
                content=entry.content,
                confidence=round(entry.confidence * transfer_ratio, 4),
                source_agent=teacher_id,
                quality_score=round(entry.quality_score * transfer_ratio, 4),
                tags=list(entry.tags),
            )
            self._knowledge_bases[student_id].append(distilled)
            transferred.append({
                "topic": entry.topic,
                "original_confidence": entry.confidence,
                "distilled_confidence": distilled.confidence,
            })

        # 记录蒸馏日志
        log_entry = {
            "teacher_id": teacher_id,
            "student_id": student_id,
            "transferred_count": len(transferred),
            "transfer_ratio": transfer_ratio,
            "timestamp": time.time(),
        }
        self._distillation_log.append(log_entry)

        return {
            "success": True,
            "teacher_id": teacher_id,
            "student_id": student_id,
            "transferred_count": len(transferred),
            "details": transferred,
        }

    def share_experience(self, agent_id: str, experience: Experience) -> str:
        """
        共享经验。

        将经验记录添加到Agent的经验库中。
        经验可以被其他Agent查询和学习。

        Args:
            agent_id: Agent ID
            experience: 经验记录

        Returns:
            经验ID
        """
        experience.agent_id = agent_id
        self._experience_db[agent_id].append(experience)
        return experience.id if hasattr(experience, 'id') else str(uuid.uuid4())

    def query_knowledge(
        self,
        agent_id: str,
        query: str,
        top_k: int = 5,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        查询知识库。

        基于关键词匹配和置信度排序返回最相关的知识条目。

        Args:
            agent_id: Agent ID
            query: 查询字符串
            top_k: 返回前K个结果
            min_confidence: 最低置信度

        Returns:
            匹配的知识条目列表
        """
        knowledge = self._knowledge_bases.get(agent_id, [])
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        scored = []
        for entry in knowledge:
            if entry.confidence < min_confidence:
                continue

            # 计算相关性分数
            relevance = self._compute_relevance(query_terms, entry)

            if relevance > 0:
                scored.append({
                    "topic": entry.topic,
                    "content": entry.content,
                    "confidence": entry.confidence,
                    "relevance": round(relevance, 4),
                    "source": entry.source_agent,
                    "tags": entry.tags,
                })

        # 按相关性 * 置信度排序
        scored.sort(key=lambda x: x["relevance"] * x["confidence"], reverse=True)
        return scored[:top_k]

    def query_experiences(
        self,
        agent_id: str,
        task_type: Optional[str] = None,
        min_outcome: float = 0.0,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """查询经验库"""
        experiences = self._experience_db.get(agent_id, [])
        results = []

        for exp in experiences:
            if task_type and exp.task_type != task_type:
                continue
            if exp.outcome < min_outcome:
                continue
            results.append({
                "task_type": exp.task_type,
                "outcome": exp.outcome,
                "lessons": exp.lessons,
                "timestamp": exp.timestamp,
                "context": exp.context,
            })

        results.sort(key=lambda x: x["outcome"], reverse=True)
        return results[:limit]

    @staticmethod
    def _compute_relevance(query_terms: Set[str], entry: KnowledgeEntry) -> float:
        """计算查询与知识条目的相关性"""
        # 在topic、content和tags中搜索
        text = (entry.topic + " " + entry.content + " " + " ".join(entry.tags)).lower()
        text_terms = set(text.split())

        # Jaccard相似度
        intersection = query_terms & text_terms
        union = query_terms | text_terms

        if not union:
            return 0.0

        jaccard = len(intersection) / len(union)

        # 额外加分：完全匹配的term
        exact_match_bonus = 0.0
        for term in query_terms:
            if term in entry.topic.lower():
                exact_match_bonus += 0.2
            if term in " ".join(entry.tags).lower():
                exact_match_bonus += 0.1

        return min(jaccard + exact_match_bonus, 1.0)

    def add_knowledge(
        self,
        agent_id: str,
        topic: str,
        content: str,
        confidence: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> KnowledgeEntry:
        """添加知识条目"""
        entry = KnowledgeEntry(
            topic=topic,
            content=content,
            confidence=confidence,
            source_agent=agent_id,
            quality_score=confidence,
            tags=tags or [],
        )
        self._knowledge_bases[agent_id].append(entry)
        return entry

    def get_knowledge_stats(self, agent_id: str) -> Dict[str, Any]:
        """获取知识库统计"""
        knowledge = self._knowledge_bases.get(agent_id, [])
        experiences = self._experience_db.get(agent_id, [])

        if knowledge:
            avg_confidence = sum(e.confidence for e in knowledge) / len(knowledge)
            topics = set(e.topic for e in knowledge)
        else:
            avg_confidence = 0.0
            topics = set()

        return {
            "agent_id": agent_id,
            "knowledge_count": len(knowledge),
            "experience_count": len(experiences),
            "avg_confidence": round(avg_confidence, 4),
            "unique_topics": len(topics),
        }


# ============================================================
# 技能图谱
# ============================================================

class SkillGraph:
    """
    技能图谱。

    使用有向无环图（DAG）表示技能依赖关系。
    支持技能差距分析和学习路径推荐。
    """

    def __init__(self):
        # 技能节点: {skill_name: SkillNode}
        self._skills: Dict[str, SkillNode] = {}
        # Agent技能: {agent_id: {skill_name: level}}
        self._agent_skills: Dict[str, Dict[str, float]] = defaultdict(dict)
        # 邻接表（依赖关系）: {skill: [dependent_skills]}
        self._dependencies: Dict[str, List[str]] = defaultdict(list)
        # 反向邻接表: {skill: [prerequisite_skills]}
        self._dependents: Dict[str, List[str]] = defaultdict(list)

    def add_skill(
        self,
        name: str,
        prerequisites: Optional[List[str]] = None,
        description: str = "",
    ) -> None:
        """
        添加技能到图谱。

        Args:
            name: 技能名称
            prerequisites: 前置技能列表
            description: 技能描述
        """
        if name not in self._skills:
            self._skills[name] = SkillNode(
                name=name,
                description=description,
            )

        prerequisites = prerequisites or []
        self._skills[name].prerequisites = prerequisites

        # 建立依赖关系
        for prereq in prerequisites:
            if prereq not in self._skills:
                self._skills[prereq] = SkillNode(name=prereq)
            if name not in self._dependencies[prereq]:
                self._dependencies[prereq].append(name)
            if prereq not in self._dependents[name]:
                self._dependents[name].append(prereq)

        # 检查是否有环
        if self._has_cycle():
            # 回滚
            for prereq in prerequisites:
                if name in self._dependencies[prereq]:
                    self._dependencies[prereq].remove(name)
                if prereq in self._dependents[name]:
                    self._dependents[name].remove(prereq)
            self._skills[name].prerequisites = []
            raise ValueError(f"添加技能 {name} 会导致循环依赖")

    def set_agent_skill_level(self, agent_id: str, skill: str, level: float) -> None:
        """设置Agent的技能等级"""
        level = max(0.0, min(1.0, level))
        if skill not in self._skills:
            self.add_skill(skill)
        self._agent_skills[agent_id][skill] = level

    def get_skill_gap(
        self,
        target_skills: List[str],
        agent_id: str,
        min_level: float = 0.7,
    ) -> Dict[str, Any]:
        """
        技能差距分析。

        分析Agent与目标技能要求的差距。

        Args:
            target_skills: 目标技能列表
            agent_id: Agent ID
            min_level: 最低要求等级

        Returns:
            差距分析结果
        """
        gaps = []
        met_skills = []

        for skill_name in target_skills:
            current_level = self._agent_skills.get(agent_id, {}).get(skill_name, 0.0)
            gap = max(0.0, min_level - current_level)

            if gap > 0:
                gaps.append({
                    "skill": skill_name,
                    "current_level": round(current_level, 4),
                    "required_level": min_level,
                    "gap": round(gap, 4),
                    "prerequisites": self._skills.get(skill_name, SkillNode("")).prerequisites,
                })
            else:
                met_skills.append({
                    "skill": skill_name,
                    "current_level": round(current_level, 4),
                })

        # 按差距大小排序
        gaps.sort(key=lambda x: x["gap"], reverse=True)

        return {
            "agent_id": agent_id,
            "total_target_skills": len(target_skills),
            "skills_met": len(met_skills),
            "skills_gap": len(gaps),
            "met": met_skills,
            "gaps": gaps,
            "readiness": round(
                len(met_skills) / max(len(target_skills), 1), 4
            ),
        }

    def recommend_learning_path(
        self,
        agent_id: str,
        target: str,
    ) -> List[str]:
        """
        推荐学习路径。

        使用拓扑排序确定学习顺序，考虑Agent当前技能水平。
        跳过已达到足够水平的技能。

        Args:
            agent_id: Agent ID
            target: 目标技能

        Returns:
            推荐的学习路径（技能名称列表）
        """
        if target not in self._skills:
            return []

        # 收集所有需要学习的技能（目标 + 所有前置技能）
        required_skills = set()
        queue = deque([target])

        while queue:
            skill = queue.popleft()
            if skill in required_skills:
                continue
            required_skills.add(skill)
            for prereq in self._dependents.get(skill, []):
                queue.append(prereq)

        # 过滤掉已掌握的技能
        agent_skills = self._agent_skills.get(agent_id, {})
        to_learn = []
        for skill in required_skills:
            current_level = agent_skills.get(skill, 0.0)
            if current_level < 0.7:  # 未达到熟练水平
                to_learn.append(skill)

        # 拓扑排序确定学习顺序
        learning_path = self._topological_sort(set(to_learn))

        return learning_path

    def _topological_sort(self, skills: Set[str]) -> List[str]:
        """
        拓扑排序（Kahn算法）。

        确定技能的学习顺序，保证先学前置技能。
        """
        # 只考虑给定技能集合内的依赖关系
        in_degree = {skill: 0 for skill in skills}
        adj = {skill: [] for skill in skills}

        for skill in skills:
            for prereq in self._dependents.get(skill, []):
                if prereq in skills:
                    adj[prereq].append(skill)
                    in_degree[skill] += 1

        # Kahn算法
        queue = deque()
        for skill in skills:
            if in_degree[skill] == 0:
                queue.append(skill)

        result = []
        while queue:
            # 优先选择被更多技能依赖的（关键路径优先）
            queue_list = sorted(queue, key=lambda s: len(adj[s]), reverse=True)
            queue = deque(queue_list)

            node = queue.popleft()
            result.append(node)

            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # 如果有环，添加剩余节点
        remaining = skills - set(result)
        result.extend(sorted(remaining))

        return result

    def _has_cycle(self) -> bool:
        """检测图中是否有环（DFS）"""
        visited = set()
        rec_stack = set()

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in self._dependencies.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for skill in self._skills:
            if skill not in visited:
                if dfs(skill):
                    return True

        return False

    def get_skill_dependencies(self, skill: str) -> Dict[str, Any]:
        """获取技能的依赖信息"""
        if skill not in self._skills:
            return {"skill": skill, "exists": False}

        # BFS获取所有前置技能
        all_prereqs = set()
        queue = deque(self._dependents.get(skill, []))
        while queue:
            s = queue.popleft()
            if s not in all_prereqs:
                all_prereqs.add(s)
                for p in self._dependents.get(s, []):
                    queue.append(p)

        return {
            "skill": skill,
            "exists": True,
            "direct_prerequisites": self._dependents.get(skill, []),
            "all_prerequisites": list(all_prereqs),
            "dependents": self._dependencies.get(skill, []),
            "description": self._skills[skill].description,
        }


# ============================================================
# 联邦知识蒸馏
# ============================================================

class FederatedDistillation:
    """
    联邦知识蒸馏。

    聚合多个Agent的知识贡献，计算贡献权重。
    """

    def __init__(self):
        self._aggregation_history: List[Dict[str, Any]] = []

    def aggregate_knowledge(
        self,
        contributions: Dict[str, List[KnowledgeEntry]],
        min_contributors: int = 2,
    ) -> Dict[str, Any]:
        """
        聚合多方知识。

        聚合策略：
        1. 按主题分组知识
        2. 对每个主题，使用加权平均合并知识
        3. 权重基于贡献者的信誉和知识质量

        Args:
            contributions: {agent_id: [KnowledgeEntry]} 各Agent的知识贡献
            min_contributors: 最少贡献者数量

        Returns:
            聚合结果
        """
        if len(contributions) < min_contributors:
            return {
                "success": False,
                "reason": f"需要至少 {min_contributors} 个贡献者",
            }

        # 计算每个贡献者的权重
        weights = self.compute_contribution_weight(contributions)

        # 按主题分组
        topic_knowledge: Dict[str, List[Tuple[str, KnowledgeEntry]]] = defaultdict(list)
        for agent_id, entries in contributions.items():
            for entry in entries:
                topic_knowledge[entry.topic].append((agent_id, entry))

        # 聚合每个主题的知识
        aggregated = {}
        for topic, entries in topic_knowledge.items():
            # 加权平均置信度和质量
            total_weight = 0.0
            weighted_confidence = 0.0
            weighted_quality = 0.0
            contents = []

            for agent_id, entry in entries:
                w = weights.get(agent_id, 1.0 / len(contributions))
                weighted_confidence += w * entry.confidence
                weighted_quality += w * entry.quality_score
                total_weight += w
                contents.append(entry.content)

            if total_weight > 0:
                avg_confidence = weighted_confidence / total_weight
                avg_quality = weighted_quality / total_weight
            else:
                avg_confidence = 0.0
                avg_quality = 0.0

            # 合并内容（去重）
            unique_contents = list(dict.fromkeys(contents))

            aggregated[topic] = {
                "content": unique_contents,
                "confidence": round(avg_confidence, 4),
                "quality": round(avg_quality, 4),
                "contributor_count": len(set(a for a, _ in entries)),
            }

        result = {
            "success": True,
            "num_contributors": len(contributions),
            "num_topics": len(aggregated),
            "aggregated_knowledge": aggregated,
            "contributor_weights": {
                k: round(v, 4) for k, v in weights.items()
            },
        }

        self._aggregation_history.append({
            "timestamp": time.time(),
            "num_contributors": len(contributions),
            "num_topics": len(aggregated),
        })

        return result

    def compute_contribution_weight(
        self,
        contributions: Dict[str, List[KnowledgeEntry]],
    ) -> Dict[str, float]:
        """
        计算贡献权重。

        权重计算基于：
        1. 贡献数量 (30%)
        2. 平均质量 (40%)
        3. 知识多样性 (30%)

        使用归一化后加权求和。
        """
        if not contributions:
            return {}

        # 1. 贡献数量
        counts = {aid: len(entries) for aid, entries in contributions.items()}
        max_count = max(counts.values()) if counts else 1
        norm_counts = {aid: c / max_count for aid, c in counts.items()}

        # 2. 平均质量
        avg_qualities = {}
        for aid, entries in contributions.items():
            if entries:
                avg_qualities[aid] = sum(e.quality_score for e in entries) / len(entries)
            else:
                avg_qualities[aid] = 0.0
        max_quality = max(avg_qualities.values()) if avg_qualities else 1
        norm_qualities = {
            aid: q / max_quality for aid, q in avg_qualities.items()
        }

        # 3. 知识多样性（唯一主题数 / 总贡献数）
        diversities = {}
        for aid, entries in contributions.items():
            unique_topics = set(e.topic for e in entries)
            diversities[aid] = len(unique_topics) / max(len(entries), 1)
        max_div = max(diversities.values()) if diversities else 1
        norm_diversities = {
            aid: d / max_div for aid, d in diversities.items()
        }

        # 加权求和
        raw_weights = {}
        for aid in contributions:
            raw_weights[aid] = (
                0.3 * norm_counts[aid]
                + 0.4 * norm_qualities[aid]
                + 0.3 * norm_diversities[aid]
            )

        # 归一化使权重之和为1
        total = sum(raw_weights.values())
        if total > 0:
            weights = {aid: w / total for aid, w in raw_weights.items()}
        else:
            n = len(contributions)
            weights = {aid: 1.0 / n for aid in contributions}

        return weights


# ============================================================
# 学习路径规划器
# ============================================================

class LearningPathPlanner:
    """
    学习路径规划器。

    基于技能图谱规划最优学习路径。
    """

    def __init__(self, skill_graph: Optional[SkillGraph] = None):
        self._graph = skill_graph or SkillGraph()
        self._plans: Dict[str, Dict[str, Any]] = {}

    def plan(
        self,
        agent_id: str,
        goal: str,
        current_skills: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        规划学习路径。

        Args:
            agent_id: Agent ID
            goal: 目标技能
            current_skills: 当前技能水平 {skill: level}

        Returns:
            学习路径规划
        """
        # 设置当前技能
        if current_skills:
            for skill, level in current_skills.items():
                self._graph.set_agent_skill_level(agent_id, skill, level)

        # 获取推荐学习路径
        path = self._graph.recommend_learning_path(agent_id, goal)

        if not path:
            return {
                "agent_id": agent_id,
                "goal": goal,
                "path": [],
                "status": "no_path_found",
                "estimated_steps": 0,
            }

        # 计算每个步骤的详细信息
        agent_skills = self._graph._agent_skills.get(agent_id, {})
        steps = []
        for i, skill in enumerate(path):
            current_level = agent_skills.get(skill, 0.0)
            skill_info = self._graph.get_skill_dependencies(skill)

            # 估算学习难度
            difficulty = self._estimate_difficulty(skill, agent_skills)

            steps.append({
                "order": i + 1,
                "skill": skill,
                "current_level": round(current_level, 4),
                "target_level": 0.7,
                "difficulty": round(difficulty, 4),
                "description": skill_info.get("description", ""),
            })

        plan = {
            "agent_id": agent_id,
            "goal": goal,
            "path": path,
            "steps": steps,
            "status": "planned",
            "total_steps": len(path),
            "estimated_difficulty": round(
                sum(s["difficulty"] for s in steps) / max(len(steps), 1), 4
            ),
        }

        self._plans[agent_id] = plan
        return plan

    def recommend_next_skill(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        推荐下一个应学习的技能。

        基于当前计划和学习进度。
        """
        plan = self._plans.get(agent_id)
        if not plan or not plan["path"]:
            return None

        agent_skills = self._graph._agent_skills.get(agent_id, {})

        # 找到第一个未完成的技能
        for skill in plan["path"]:
            current_level = agent_skills.get(skill, 0.0)
            if current_level < 0.7:
                difficulty = self._estimate_difficulty(skill, agent_skills)
                return {
                    "skill": skill,
                    "current_level": round(current_level, 4),
                    "target_level": 0.7,
                    "difficulty": round(difficulty, 4),
                    "reason": "next_in_learning_path",
                }

        return {
            "skill": plan["path"][-1] if plan["path"] else None,
            "status": "all_skills_completed",
            "message": "学习路径已完成",
        }

    def _estimate_difficulty(
        self,
        skill: str,
        agent_skills: Dict[str, float],
    ) -> float:
        """
        估算技能学习难度。

        难度基于：
        1. 前置技能数量
        2. 未掌握的前置技能比例
        3. 技能深度（依赖链长度）
        """
        # 前置技能分析
        prereqs = self._graph._dependents.get(skill, [])
        if not prereqs:
            return 0.2  # 基础技能，难度低

        # 未掌握的前置技能比例
        unmet_prereqs = sum(
            1 for p in prereqs if agent_skills.get(p, 0.0) < 0.5
        )
        prereq_ratio = unmet_prereqs / len(prereqs)

        # 依赖链深度
        depth = self._get_dependency_depth(skill)

        # 综合难度
        difficulty = 0.3 * prereq_ratio + 0.4 * (depth / 5.0) + 0.3 * 0.5
        return min(max(difficulty, 0.1), 1.0)

    def _get_dependency_depth(self, skill: str) -> int:
        """获取技能的依赖链深度"""
        visited = set()
        max_depth = 0

        def dfs(node, depth):
            nonlocal max_depth
            if node in visited:
                return
            visited.add(node)
            max_depth = max(max_depth, depth)
            for prereq in self._graph._dependents.get(node, []):
                dfs(prereq, depth + 1)

        dfs(skill, 0)
        return max_depth

    def get_plan(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """获取Agent的学习计划"""
        return self._plans.get(agent_id)

    def update_progress(
        self,
        agent_id: str,
        skill: str,
        new_level: float,
    ) -> Dict[str, Any]:
        """更新学习进度"""
        self._graph.set_agent_skill_level(agent_id, skill, new_level)

        plan = self._plans.get(agent_id)
        if not plan:
            return {"status": "no_plan"}

        # 计算完成进度
        agent_skills = self._graph._agent_skills.get(agent_id, {})
        completed = sum(
            1 for s in plan["path"]
            if agent_skills.get(s, 0.0) >= 0.7
        )
        progress = completed / max(len(plan["path"]), 1)

        return {
            "status": "updated",
            "skill": skill,
            "new_level": round(new_level, 4),
            "overall_progress": round(progress, 4),
            "completed_steps": completed,
            "total_steps": len(plan["path"]),
        }
