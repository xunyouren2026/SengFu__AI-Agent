"""
技能涌现分析模块 - Skill Emergence Analysis

观察和记录Agent协作过程中新技能的产生，
分析技能演化规律，预测潜在的新技能组合。
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum
import math
import time


class SkillCategory(Enum):
    """技能类别"""
    COGNITION = "cognition"        # 认知类
    COMMUNICATION = "communication"  # 通信类
    COORDINATION = "coordination"   # 协调类
    LEARNING = "learning"          # 学习类
    REASONING = "reasoning"        # 推理类
    CREATIVE = "creative"          # 创造类
    ADAPTATION = "adaptation"      # 适应类


class EmergenceStage(Enum):
    """涌现阶段"""
    LATENT = "latent"           # 潜伏期：技能尚未显现
    NASCENT = "nascent"         # 萌芽期：技能开始出现
    DEVELOPING = "developing"   # 发展期：技能逐渐成熟
    MATURE = "mature"           # 成熟期：技能稳定
    DECLINING = "declining"     # 衰退期：技能使用减少


@dataclass
class Skill:
    """技能定义"""
    skill_id: str
    name: str
    category: SkillCategory
    description: str = ""
    prerequisites: Set[str] = field(default_factory=set)
    complexity: float = 1.0  # 1-10
    utility: float = 0.0     # 效用值
    created_at: float = field(default_factory=time.time)
    parent_skills: Set[str] = field(default_factory=set)  # 组合来源
    
    def __hash__(self):
        return hash(self.skill_id)
    
    def __eq__(self, other):
        if isinstance(other, Skill):
            return self.skill_id == other.skill_id
        return False


@dataclass
class SkillObservation:
    """技能观察记录"""
    skill_id: str
    agent_id: str
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    performance: float = 0.0  # 执行表现
    success: bool = True
    duration: float = 0.0  # 执行时长


@dataclass
class EmergenceEvent:
    """涌现事件"""
    event_id: str
    timestamp: float
    emergent_skill: Skill
    contributing_skills: Set[str]
    trigger_context: Dict[str, Any]
    confidence: float  # 涌现置信度
    validated: bool = False


class SkillGraph:
    """
    技能图谱
    
    维护技能之间的关系网络
    """
    
    def __init__(self):
        self.skills: Dict[str, Skill] = {}
        self.edges: Dict[str, Set[str]] = defaultdict(set)  # 依赖关系
        self.composition_edges: Dict[str, Set[str]] = defaultdict(set)  # 组合关系
        self.usage_counts: Dict[str, int] = defaultdict(int)
    
    def add_skill(self, skill: Skill) -> None:
        """添加技能"""
        self.skills[skill.skill_id] = skill
        
        # 添加依赖边
        for prereq in skill.prerequisites:
            self.edges[prereq].add(skill.skill_id)
        
        # 添加组合边
        for parent in skill.parent_skills:
            self.composition_edges[parent].add(skill.skill_id)
    
    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """获取技能"""
        return self.skills.get(skill_id)
    
    def get_dependencies(self, skill_id: str) -> Set[str]:
        """获取依赖的技能"""
        skill = self.skills.get(skill_id)
        if skill:
            return skill.prerequisites.copy()
        return set()
    
    def get_dependents(self, skill_id: str) -> Set[str]:
        """获取依赖此技能的技能"""
        return self.edges.get(skill_id, set()).copy()
    
    def get_composition_parents(self, skill_id: str) -> Set[str]:
        """获取组合来源技能"""
        skill = self.skills.get(skill_id)
        if skill:
            return skill.parent_skills.copy()
        return set()
    
    def get_composition_children(self, skill_id: str) -> Set[str]:
        """获取此技能参与组合产生的新技能"""
        return self.composition_edges.get(skill_id, set()).copy()
    
    def record_usage(self, skill_id: str) -> None:
        """记录技能使用"""
        self.usage_counts[skill_id] += 1
    
    def get_usage_count(self, skill_id: str) -> int:
        """获取使用次数"""
        return self.usage_counts.get(skill_id, 0)
    
    def find_path(self, from_skill: str, to_skill: str) -> Optional[List[str]]:
        """查找技能路径（BFS）"""
        if from_skill not in self.skills or to_skill not in self.skills:
            return None
        
        if from_skill == to_skill:
            return [from_skill]
        
        visited = {from_skill}
        queue = [(from_skill, [from_skill])]
        
        while queue:
            current, path = queue.pop(0)
            
            # 检查依赖和组合关系
            neighbors = self.edges.get(current, set()) | self.composition_edges.get(current, set())
            
            for neighbor in neighbors:
                if neighbor == to_skill:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def get_skill_cluster(self, skill_id: str, depth: int = 2) -> Set[str]:
        """获取技能簇（相关技能）"""
        cluster = {skill_id}
        frontier = {skill_id}
        
        for _ in range(depth):
            new_frontier = set()
            for s in frontier:
                # 添加依赖和被依赖
                new_frontier.update(self.get_dependencies(s))
                new_frontier.update(self.get_dependents(s))
                new_frontier.update(self.get_composition_parents(s))
                new_frontier.update(self.get_composition_children(s))
            
            new_frontier -= cluster
            cluster.update(new_frontier)
            frontier = new_frontier
            
            if not frontier:
                break
        
        return cluster


class SkillEmergenceDetector:
    """
    技能涌现检测器
    
    检测新技能的涌现事件
    """
    
    def __init__(
        self,
        emergence_threshold: float = 0.7,
        min_observations: int = 5
    ):
        self.emergence_threshold = emergence_threshold
        self.min_observations = min_observations
        
        # 观察历史
        self.observations: List[SkillObservation] = []
        
        # 涌现事件
        self.emergence_events: List[EmergenceEvent] = []
        
        # 技能组合模式
        self.combination_patterns: Dict[Tuple[str, ...], int] = defaultdict(int)
        
        # 技能共现统计
        self.cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)
    
    def observe(
        self,
        skill_id: str,
        agent_id: str,
        performance: float = 0.0,
        success: bool = True,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """记录技能观察"""
        observation = SkillObservation(
            skill_id=skill_id,
            agent_id=agent_id,
            timestamp=time.time(),
            context=context or {},
            performance=performance,
            success=success
        )
        self.observations.append(observation)
    
    def observe_skill_sequence(
        self,
        agent_id: str,
        skill_sequence: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """观察技能序列"""
        for i, skill_id in enumerate(skill_sequence):
            self.observe(
                skill_id=skill_id,
                agent_id=agent_id,
                context={'sequence_position': i, 'sequence_length': len(skill_sequence)}
            )
        
        # 记录组合模式
        if len(skill_sequence) >= 2:
            pattern = tuple(skill_sequence)
            self.combination_patterns[pattern] += 1
            
            # 记录共现
            for i in range(len(skill_sequence)):
                for j in range(i + 1, len(skill_sequence)):
                    pair = (skill_sequence[i], skill_sequence[j])
                    self.cooccurrence[pair] += 1
    
    def detect_emergence(
        self,
        skill_graph: SkillGraph,
        time_window: float = 3600.0  # 1小时窗口
    ) -> List[EmergenceEvent]:
        """
        检测涌现事件
        
        分析最近的观察，检测新技能的涌现
        """
        current_time = time.time()
        recent_observations = [
            o for o in self.observations
            if current_time - o.timestamp <= time_window
        ]
        
        if len(recent_observations) < self.min_observations:
            return []
        
        # 按技能分组
        skill_obs = defaultdict(list)
        for obs in recent_observations:
            skill_obs[obs.skill_id].append(obs)
        
        new_events = []
        
        # 检测每个技能的涌现
        for skill_id, observations in skill_obs.items():
            skill = skill_graph.get_skill(skill_id)
            if skill is None:
                continue
            
            # 检查是否为新涌现的技能
            emergence_score = self._compute_emergence_score(
                skill_id, observations, skill_graph
            )
            
            if emergence_score >= self.emergence_threshold:
                # 创建涌现事件
                event = EmergenceEvent(
                    event_id=f"emerge_{skill_id}_{int(current_time)}",
                    timestamp=current_time,
                    emergent_skill=skill,
                    contributing_skills=self._find_contributing_skills(
                        skill_id, recent_observations
                    ),
                    trigger_context=self._extract_trigger_context(observations),
                    confidence=emergence_score
                )
                
                self.emergence_events.append(event)
                new_events.append(event)
        
        return new_events
    
    def _compute_emergence_score(
        self,
        skill_id: str,
        observations: List[SkillObservation],
        skill_graph: SkillGraph
    ) -> float:
        """计算涌现分数"""
        if not observations:
            return 0.0
        
        # 因素1: 使用频率增长
        usage_score = min(1.0, len(observations) / 20)
        
        # 因素2: 成功率
        success_rate = sum(1 for o in observations if o.success) / len(observations)
        
        # 因素3: 性能提升趋势
        if len(observations) >= 3:
            early_perf = sum(o.performance for o in observations[:len(observations)//3])
            late_perf = sum(o.performance for o in observations[-len(observations)//3:])
            perf_trend = min(1.0, max(0.0, (late_perf - early_perf) / max(0.1, early_perf + 1)))
        else:
            perf_trend = 0.5
        
        # 因素4: 技能组合新颖性
        skill = skill_graph.get_skill(skill_id)
        if skill and skill.parent_skills:
            novelty = 0.8  # 组合技能通常更具新颖性
        else:
            novelty = 0.5
        
        # 综合分数
        score = (
            0.3 * usage_score +
            0.3 * success_rate +
            0.2 * perf_trend +
            0.2 * novelty
        )
        
        return score
    
    def _find_contributing_skills(
        self,
        skill_id: str,
        observations: List[SkillObservation]
    ) -> Set[str]:
        """找出促成涌现的技能"""
        contributing = set()
        
        # 查找共现频率高的技能
        for (s1, s2), count in self.cooccurrence.items():
            if s1 == skill_id and count >= 3:
                contributing.add(s2)
            elif s2 == skill_id and count >= 3:
                contributing.add(s1)
        
        return contributing
    
    def _extract_trigger_context(
        self,
        observations: List[SkillObservation]
    ) -> Dict[str, Any]:
        """提取触发上下文"""
        contexts = [o.context for o in observations if o.context]
        
        if not contexts:
            return {}
        
        # 合并上下文
        merged = {}
        for ctx in contexts:
            for key, value in ctx.items():
                if key not in merged:
                    merged[key] = []
                merged[key].append(value)
        
        # 统计最常见的值
        result = {}
        for key, values in merged.items():
            counter = Counter(values)
            result[key] = counter.most_common(1)[0][0]
        
        return result
    
    def predict_potential_emergence(
        self,
        skill_graph: SkillGraph,
        top_k: int = 5
    ) -> List[Tuple[Set[str], float]]:
        """
        预测潜在的技能涌现
        
        基于共现模式预测可能涌现的新技能组合
        """
        # 找出高频共现但尚未形成新技能的组合
        potential = []
        
        for (s1, s2), count in self.cooccurrence.items():
            # 检查是否已存在组合技能
            skill1 = skill_graph.get_skill(s1)
            skill2 = skill_graph.get_skill(s2)
            
            if skill1 and skill2:
                # 检查是否有技能以这两个为父技能
                existing = False
                for skill in skill_graph.skills.values():
                    if s1 in skill.parent_skills and s2 in skill.parent_skills:
                        existing = True
                        break
                
                if not existing and count >= 5:
                    # 计算涌现概率
                    emergence_prob = min(1.0, count / 20)
                    potential.append(({s1, s2}, emergence_prob))
        
        # 排序返回top_k
        potential.sort(key=lambda x: x[1], reverse=True)
        return potential[:top_k]


class SkillEmergenceAnalyzer:
    """
    技能涌现分析器
    
    综合分析技能涌现现象
    """
    
    def __init__(
        self,
        emergence_threshold: float = 0.7,
        min_observations: int = 5
    ):
        self.skill_graph = SkillGraph()
        self.detector = SkillEmergenceDetector(
            emergence_threshold=emergence_threshold,
            min_observations=min_observations
        )
        
        # 技能发展阶段跟踪
        self.skill_stages: Dict[str, EmergenceStage] = {}
        
        # 分析历史
        self.analysis_history: List[Dict[str, Any]] = []
    
    def register_skill(
        self,
        skill_id: str,
        name: str,
        category: SkillCategory,
        description: str = "",
        prerequisites: Optional[Set[str]] = None,
        complexity: float = 1.0,
        parent_skills: Optional[Set[str]] = None
    ) -> Skill:
        """注册技能"""
        skill = Skill(
            skill_id=skill_id,
            name=name,
            category=category,
            description=description,
            prerequisites=prerequisites or set(),
            complexity=complexity,
            parent_skills=parent_skills or set()
        )
        
        self.skill_graph.add_skill(skill)
        self.skill_stages[skill_id] = EmergenceStage.LATENT
        
        return skill
    
    def observe_skill_usage(
        self,
        skill_id: str,
        agent_id: str,
        performance: float = 0.0,
        success: bool = True,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """观察技能使用"""
        self.detector.observe(skill_id, agent_id, performance, success, context)
        self.skill_graph.record_usage(skill_id)
        
        # 更新技能阶段
        self._update_skill_stage(skill_id)
    
    def _update_skill_stage(self, skill_id: str) -> None:
        """更新技能发展阶段"""
        usage_count = self.skill_graph.get_usage_count(skill_id)
        current_stage = self.skill_stages.get(skill_id, EmergenceStage.LATENT)
        
        # 根据使用次数更新阶段
        if usage_count < 3:
            new_stage = EmergenceStage.LATENT
        elif usage_count < 10:
            new_stage = EmergenceStage.NASCENT
        elif usage_count < 30:
            new_stage = EmergenceStage.DEVELOPING
        elif usage_count < 100:
            new_stage = EmergenceStage.MATURE
        else:
            new_stage = EmergenceStage.MATURE
        
        # 检查是否衰退
        if current_stage == EmergenceStage.MATURE:
            # 检查最近使用
            recent_observations = [
                o for o in self.detector.observations
                if o.skill_id == skill_id and 
                   time.time() - o.timestamp < 86400  # 24小时内
            ]
            if len(recent_observations) < 2:
                new_stage = EmergenceStage.DECLINING
        
        self.skill_stages[skill_id] = new_stage
    
    def analyze_emergence(
        self,
        time_window: float = 3600.0
    ) -> Dict[str, Any]:
        """分析涌现现象"""
        # 检测涌现事件
        new_events = self.detector.detect_emergence(
            self.skill_graph, time_window
        )
        
        # 预测潜在涌现
        potential = self.detector.predict_potential_emergence(self.skill_graph)
        
        # 统计各阶段技能数量
        stage_counts = defaultdict(int)
        for stage in self.skill_stages.values():
            stage_counts[stage] += 1
        
        # 统计各类别技能
        category_counts = defaultdict(int)
        for skill in self.skill_graph.skills.values():
            category_counts[skill.category] += 1
        
        analysis_result = {
            'timestamp': time.time(),
            'new_emergence_events': len(new_events),
            'emergence_events': [
                {
                    'skill_id': e.emergent_skill.skill_id,
                    'skill_name': e.emergent_skill.name,
                    'confidence': e.confidence,
                    'contributing_skills': list(e.contributing_skills)
                }
                for e in new_events
            ],
            'potential_emergence': [
                {
                    'skills': list(skills),
                    'probability': prob
                }
                for skills, prob in potential
            ],
            'stage_distribution': {
                stage.value: count for stage, count in stage_counts.items()
            },
            'category_distribution': {
                cat.value: count for cat, count in category_counts.items()
            },
            'total_skills': len(self.skill_graph.skills),
            'total_observations': len(self.detector.observations)
        }
        
        self.analysis_history.append(analysis_result)
        
        return analysis_result
    
    def get_skill_evolution_history(
        self,
        skill_id: str
    ) -> Dict[str, Any]:
        """获取技能演化历史"""
        skill = self.skill_graph.get_skill(skill_id)
        if skill is None:
            return {}
        
        observations = [
            o for o in self.detector.observations
            if o.skill_id == skill_id
        ]
        
        if not observations:
            return {
                'skill_id': skill_id,
                'name': skill.name,
                'stage': self.skill_stages.get(skill_id, EmergenceStage.LATENT).value,
                'usage_count': self.skill_graph.get_usage_count(skill_id)
            }
        
        # 按时间排序
        observations.sort(key=lambda o: o.timestamp)
        
        # 计算性能趋势
        performances = [o.performance for o in observations]
        
        return {
            'skill_id': skill_id,
            'name': skill.name,
            'category': skill.category.value,
            'stage': self.skill_stages.get(skill_id, EmergenceStage.LATENT).value,
            'usage_count': self.skill_graph.get_usage_count(skill_id),
            'first_observed': observations[0].timestamp,
            'last_observed': observations[-1].timestamp,
            'total_observations': len(observations),
            'success_rate': sum(1 for o in observations if o.success) / len(observations),
            'avg_performance': sum(performances) / len(performances),
            'performance_trend': self._compute_trend(performances),
            'related_skills': list(self.skill_graph.get_skill_cluster(skill_id, depth=1) - {skill_id})
        }
    
    def _compute_trend(self, values: List[float]) -> str:
        """计算趋势"""
        if len(values) < 3:
            return 'stable'
        
        # 简单线性趋势
        n = len(values)
        early_avg = sum(values[:n//3]) / (n//3)
        late_avg = sum(values[-n//3:]) / (n//3)
        
        if late_avg > early_avg * 1.1:
            return 'improving'
        elif late_avg < early_avg * 0.9:
            return 'declining'
        else:
            return 'stable'
    
    def discover_skill_compositions(
        self,
        min_cooccurrence: int = 5
    ) -> List[Tuple[Set[str], str, float]]:
        """
        发现技能组合
        
        找出高频共现的技能组合，建议新技能
        """
        compositions = []
        
        for (s1, s2), count in self.detector.cooccurrence.items():
            if count >= min_cooccurrence:
                skill1 = self.skill_graph.get_skill(s1)
                skill2 = self.skill_graph.get_skill(s2)
                
                if skill1 and skill2:
                    # 生成组合名称
                    comp_name = f"{skill1.name}+{skill2.name}"
                    
                    # 计算组合效用
                    utility = (
                        skill1.utility * 0.4 +
                        skill2.utility * 0.4 +
                        count / 100 * 0.2
                    )
                    
                    compositions.append(({s1, s2}, comp_name, utility))
        
        # 按效用排序
        compositions.sort(key=lambda x: x[2], reverse=True)
        return compositions
    
    def get_emergence_report(self) -> Dict[str, Any]:
        """获取涌现报告"""
        # 最近的分析结果
        recent_analysis = self.analysis_history[-1] if self.analysis_history else {}
        
        # 涌现事件统计
        total_events = len(self.detector.emergence_events)
        validated_events = sum(
            1 for e in self.detector.emergence_events if e.validated
        )
        
        # 技能统计
        skill_stats = {
            'total': len(self.skill_graph.skills),
            'by_stage': defaultdict(int),
            'by_category': defaultdict(int)
        }
        
        for skill_id, stage in self.skill_stages.items():
            skill_stats['by_stage'][stage.value] += 1
        
        for skill in self.skill_graph.skills.values():
            skill_stats['by_category'][skill.category.value] += 1
        
        # 活跃技能（最近使用）
        active_skills = sorted(
            [
                (skill_id, self.skill_graph.get_usage_count(skill_id))
                for skill_id in self.skill_graph.skills
            ],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            'timestamp': time.time(),
            'total_emergence_events': total_events,
            'validated_emergence_events': validated_events,
            'recent_analysis': recent_analysis,
            'skill_statistics': {
                'total': skill_stats['total'],
                'by_stage': dict(skill_stats['by_stage']),
                'by_category': dict(skill_stats['by_category'])
            },
            'most_active_skills': [
                {'skill_id': s, 'usage_count': c}
                for s, c in active_skills
            ],
            'potential_emergence_count': len(
                self.detector.predict_potential_emergence(self.skill_graph)
            )
        }
    
    def validate_emergence(
        self,
        event_id: str,
        validation_result: bool
    ) -> bool:
        """验证涌现事件"""
        for event in self.detector.emergence_events:
            if event.event_id == event_id:
                event.validated = validation_result
                return True
        return False
    
    def export_skill_graph(self) -> Dict[str, Any]:
        """导出技能图谱"""
        nodes = []
        for skill in self.skill_graph.skills.values():
            nodes.append({
                'id': skill.skill_id,
                'name': skill.name,
                'category': skill.category.value,
                'complexity': skill.complexity,
                'utility': skill.utility,
                'stage': self.skill_stages.get(skill.skill_id, EmergenceStage.LATENT).value,
                'usage_count': self.skill_graph.get_usage_count(skill.skill_id)
            })
        
        edges = []
        # 依赖边
        for from_skill, to_skills in self.skill_graph.edges.items():
            for to_skill in to_skills:
                edges.append({
                    'source': from_skill,
                    'target': to_skill,
                    'type': 'dependency'
                })
        
        # 组合边
        for from_skill, to_skills in self.skill_graph.composition_edges.items():
            for to_skill in to_skills:
                edges.append({
                    'source': from_skill,
                    'target': to_skill,
                    'type': 'composition'
                })
        
        return {
            'nodes': nodes,
            'edges': edges
        }
