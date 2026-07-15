"""
Agent课程推荐系统 - 根据能力短板推荐训练任务

实现自适应课程学习(Adaptive Curriculum Learning)，根据Agent的
当前能力和知识缺口，动态推荐合适难度的训练任务。
"""

from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import heapq
import random


@dataclass
class Task:
    """任务定义"""
    task_id: str
    name: str
    difficulty: float  # 0.0 - 1.0
    required_skills: Set[str]
    skill_rewards: Dict[str, float] = field(default_factory=dict)
    prerequisites: Set[str] = field(default_factory=set)
    estimated_time: float = 1.0  # 估计完成时间
    success_rate: float = 0.5  # 历史成功率
    
    def __hash__(self):
        return hash(self.task_id)
    
    def __eq__(self, other):
        if isinstance(other, Task):
            return self.task_id == other.task_id
        return False


@dataclass
class SkillAssessment:
    """技能评估"""
    skill_name: str
    current_level: float  # 0.0 - 1.0
    confidence: float  # 评估置信度
    last_practiced: Optional[float] = None
    practice_count: int = 0
    success_history: List[bool] = field(default_factory=list)
    
    @property
    def estimated_mastery(self) -> float:
        """估计掌握程度"""
        if not self.success_history:
            return 0.0
        recent_success = sum(self.success_history[-5:]) / min(5, len(self.success_history))
        return 0.7 * self.current_level + 0.3 * recent_success


@dataclass
class LearningObjective:
    """学习目标"""
    target_skills: Dict[str, float]  # 技能 -> 目标水平
    deadline: Optional[float] = None
    priority: float = 1.0
    description: str = ""


class AgentCapabilityProfile:
    """Agent能力画像"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.skills: Dict[str, SkillAssessment] = {}
        self.learning_history: List[Dict[str, Any]] = []
        self.preferred_difficulty: float = 0.3  # 偏好的任务难度
        self.learning_rate: float = 0.1
        
    def assess_skill(self, skill_name: str, performance: float):
        """评估技能水平"""
        if skill_name not in self.skills:
            self.skills[skill_name] = SkillAssessment(
                skill_name=skill_name,
                current_level=0.0,
                confidence=0.0
            )
        
        assessment = self.skills[skill_name]
        
        # 更新技能水平
        if assessment.practice_count == 0:
            assessment.current_level = performance
        else:
            # 指数移动平均
            assessment.current_level = (
                0.7 * assessment.current_level + 0.3 * performance
            )
        
        assessment.practice_count += 1
        assessment.success_history.append(performance > 0.6)
        
        # 更新置信度
        assessment.confidence = min(1.0, assessment.confidence + 0.1)
        
        # 记录学习历史
        self.learning_history.append({
            'skill': skill_name,
            'performance': performance,
            'new_level': assessment.current_level
        })
    
    def get_skill_gap(self, required_skill: str, target_level: float = 0.8) -> float:
        """获取技能缺口"""
        if required_skill not in self.skills:
            return target_level
        return max(0, target_level - self.skills[required_skill].current_level)
    
    def get_overall_capability(self) -> float:
        """获取整体能力水平"""
        if not self.skills:
            return 0.0
        return sum(s.current_level for s in self.skills.values()) / len(self.skills)
    
    def get_weak_skills(self, threshold: float = 0.5) -> List[str]:
        """获取薄弱技能列表"""
        weak = []
        for skill_name, assessment in self.skills.items():
            if assessment.current_level < threshold:
                weak.append(skill_name)
        return weak


class CurriculumGenerator:
    """课程生成器"""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.skill_graph: Dict[str, Set[str]] = defaultdict(set)  # 技能依赖图
        self.difficulty_history: List[float] = []
        
    def register_task(self, task: Task):
        """注册任务"""
        self.tasks[task.task_id] = task
        
        # 更新技能图
        for skill in task.required_skills:
            for prereq in task.prerequisites:
                self.skill_graph[skill].add(prereq)
    
    def generate_curriculum(self, agent_profile: AgentCapabilityProfile,
                           objective: LearningObjective,
                           max_tasks: int = 10) -> List[Task]:
        """
        生成个性化课程
        
        基于Agent当前能力和学习目标，生成最优的任务序列
        """
        # 计算每个目标技能的缺口
        skill_gaps = {}
        for skill, target in objective.target_skills.items():
            gap = agent_profile.get_skill_gap(skill, target)
            if gap > 0.01:  # 只考虑有缺口的技能
                skill_gaps[skill] = gap
        
        if not skill_gaps:
            return []  # 已经达到所有目标
        
        # 找到能覆盖这些技能缺口的任务
        candidate_tasks = []
        for task in self.tasks.values():
            relevance = self._compute_task_relevance(task, skill_gaps, agent_profile)
            if relevance > 0:
                candidate_tasks.append((relevance, task))
        
        # 按相关性和难度排序
        candidate_tasks.sort(key=lambda x: (-x[0], x[1].difficulty))
        
        # 选择任务并确保前置条件满足
        curriculum = []
        completed_tasks: Set[str] = set()
        
        for relevance, task in candidate_tasks:
            if len(curriculum) >= max_tasks:
                break
            
            # 检查前置条件
            if task.prerequisites.issubset(completed_tasks):
                curriculum.append(task)
                completed_tasks.add(task.task_id)
        
        # 拓扑排序确保依赖关系
        curriculum = self._topological_sort(curriculum)
        
        return curriculum
    
    def _compute_task_relevance(self, task: Task, skill_gaps: Dict[str, float],
                                profile: AgentCapabilityProfile) -> float:
        """计算任务相关性分数"""
        relevance = 0.0
        
        for skill in task.required_skills:
            if skill in skill_gaps:
                # 任务能弥补的技能缺口
                gap_coverage = min(task.skill_rewards.get(skill, 0.1), skill_gaps[skill])
                relevance += gap_coverage
        
        # 考虑难度匹配
        agent_capability = profile.get_overall_capability()
        difficulty_match = 1.0 - abs(task.difficulty - agent_capability)
        
        # 考虑历史成功率
        success_bonus = task.success_rate
        
        return relevance * difficulty_match * (0.5 + 0.5 * success_bonus)
    
    def _topological_sort(self, tasks: List[Task]) -> List[Task]:
        """拓扑排序任务"""
        task_ids = {t.task_id for t in tasks}
        in_degree = {t.task_id: 0 for t in tasks}
        
        for task in tasks:
            for prereq in task.prerequisites:
                if prereq in task_ids:
                    in_degree[task.task_id] += 1
        
        # Kahn算法
        queue = [t for t in tasks if in_degree[t.task_id] == 0]
        result = []
        
        while queue:
            # 按难度排序
            queue.sort(key=lambda t: t.difficulty)
            task = queue.pop(0)
            result.append(task)
            
            for other in tasks:
                if task.task_id in other.prerequisites:
                    in_degree[other.task_id] -= 1
                    if in_degree[other.task_id] == 0:
                        queue.append(other)
        
        return result
    
    def adapt_difficulty(self, agent_profile: AgentCapabilityProfile,
                        recent_performance: List[float]) -> float:
        """
        自适应调整难度
        
        根据最近表现调整推荐任务的难度
        """
        if not recent_performance:
            return agent_profile.preferred_difficulty
        
        avg_performance = sum(recent_performance) / len(recent_performance)
        
        # 如果表现好，增加难度
        if avg_performance > 0.8:
            new_difficulty = min(1.0, agent_profile.preferred_difficulty + 0.1)
        # 如果表现差，降低难度
        elif avg_performance < 0.5:
            new_difficulty = max(0.0, agent_profile.preferred_difficulty - 0.1)
        else:
            new_difficulty = agent_profile.preferred_difficulty
        
        agent_profile.preferred_difficulty = new_difficulty
        self.difficulty_history.append(new_difficulty)
        
        return new_difficulty


class CurriculumRecommender:
    """课程推荐器"""
    
    def __init__(self):
        self.generator = CurriculumGenerator()
        self.agent_profiles: Dict[str, AgentCapabilityProfile] = {}
        self.active_curricula: Dict[str, List[Task]] = {}
        self.task_completions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
    def register_agent(self, agent_id: str):
        """注册Agent"""
        if agent_id not in self.agent_profiles:
            self.agent_profiles[agent_id] = AgentCapabilityProfile(agent_id)
    
    def recommend_next_task(self, agent_id: str) -> Optional[Task]:
        """推荐下一个任务"""
        if agent_id not in self.active_curricula:
            return None
        
        curriculum = self.active_curricula[agent_id]
        if not curriculum:
            return None
        
        # 返回第一个未完成的任务
        return curriculum[0]
    
    def start_learning_path(self, agent_id: str, objective: LearningObjective):
        """开始学习路径"""
        self.register_agent(agent_id)
        profile = self.agent_profiles[agent_id]
        
        curriculum = self.generator.generate_curriculum(profile, objective)
        self.active_curricula[agent_id] = curriculum
        
        return {
            'agent_id': agent_id,
            'curriculum_length': len(curriculum),
            'first_task': curriculum[0] if curriculum else None,
            'target_skills': list(objective.target_skills.keys())
        }
    
    def report_task_completion(self, agent_id: str, task_id: str, 
                               success: bool, performance: float):
        """报告任务完成情况"""
        if agent_id not in self.agent_profiles:
            return
        
        profile = self.agent_profiles[agent_id]
        task = self.generator.tasks.get(task_id)
        
        if task:
            # 更新技能评估
            for skill in task.required_skills:
                profile.assess_skill(skill, performance)
            
            # 记录完成
            self.task_completions[agent_id].append({
                'task_id': task_id,
                'success': success,
                'performance': performance,
                'skills_practiced': list(task.required_skills)
            })
            
            # 从活跃课程中移除
            if agent_id in self.active_curricula:
                self.active_curricula[agent_id] = [
                    t for t in self.active_curricula[agent_id] 
                    if t.task_id != task_id
                ]
    
    def get_learning_report(self, agent_id: str) -> Dict[str, Any]:
        """获取学习报告"""
        if agent_id not in self.agent_profiles:
            return {}
        
        profile = self.agent_profiles[agent_id]
        completions = self.task_completions[agent_id]
        
        return {
            'agent_id': agent_id,
            'overall_capability': profile.get_overall_capability(),
            'skills': {
                name: {
                    'level': assessment.current_level,
                    'confidence': assessment.confidence,
                    'practice_count': assessment.practice_count
                }
                for name, assessment in profile.skills.items()
            },
            'tasks_completed': len(completions),
            'success_rate': (
                sum(1 for c in completions if c['success']) / len(completions)
                if completions else 0.0
            ),
            'weak_skills': profile.get_weak_skills(),
            'remaining_curriculum': len(self.active_curricula.get(agent_id, []))
        }
    
    def recommend_focus_areas(self, agent_id: str) -> List[Dict[str, Any]]:
        """推荐重点学习领域"""
        if agent_id not in self.agent_profiles:
            return []
        
        profile = self.agent_profiles[agent_id]
        weak_skills = profile.get_weak_skills(threshold=0.6)
        
        recommendations = []
        for skill in weak_skills:
            assessment = profile.skills[skill]
            
            # 找到针对该技能的任务
            relevant_tasks = [
                task for task in self.generator.tasks.values()
                if skill in task.required_skills
            ]
            
            if relevant_tasks:
                # 选择最适合当前水平的任务
                best_task = min(relevant_tasks, 
                               key=lambda t: abs(t.difficulty - assessment.current_level))
                
                recommendations.append({
                    'skill': skill,
                    'current_level': assessment.current_level,
                    'recommended_task': best_task.task_id,
                    'task_difficulty': best_task.difficulty,
                    'estimated_improvement': best_task.skill_rewards.get(skill, 0.1)
                })
        
        return sorted(recommendations, key=lambda x: x['current_level'])


class SpacedRepetitionScheduler:
    """间隔重复调度器"""
    
    def __init__(self):
        self.review_schedule: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
        self.base_interval = 1.0  # 基础间隔(天)
        
    def schedule_review(self, agent_id: str, skill: str, 
                       proficiency: float) -> float:
        """
        安排复习时间
        
        基于掌握程度计算下次复习时间
        """
        # 艾宾浩斯遗忘曲线启发
        if proficiency < 0.3:
            interval = self.base_interval
        elif proficiency < 0.6:
            interval = self.base_interval * 3
        elif proficiency < 0.8:
            interval = self.base_interval * 7
        else:
            interval = self.base_interval * 14
        
        # 添加一些随机性
        interval *= random.uniform(0.9, 1.1)
        
        return interval
    
    def get_due_reviews(self, agent_id: str, current_time: float) -> List[str]:
        """获取到期的复习项"""
        due = []
        for review_time, skill in self.review_schedule.get(agent_id, []):
            if review_time <= current_time:
                due.append(skill)
        return due
