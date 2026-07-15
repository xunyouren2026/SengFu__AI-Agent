"""
技能关系图 - 构建技能前置依赖树，规划学习路径

实现技能图谱系统，用于建模技能之间的前置依赖关系，
并基于图谱规划最优学习路径。
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import heapq
import json


@dataclass
class SkillNode:
    """技能节点"""
    skill_id: str
    name: str
    description: str = ""
    difficulty: float = 0.5  # 0.0 - 1.0
    category: str = "general"
    estimated_learning_time: float = 1.0  # 小时
    
    # 依赖关系
    prerequisites: Set[str] = field(default_factory=set)  # 前置技能
    unlocks: Set[str] = field(default_factory=set)  # 解锁的技能
    
    # 相关技能
    related_skills: Set[str] = field(default_factory=set)  # 相关技能
    complementary_skills: Set[str] = field(default_factory=set)  # 互补技能
    
    # 元数据
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.skill_id)
    
    def __eq__(self, other):
        if isinstance(other, SkillNode):
            return self.skill_id == other.skill_id
        return False


@dataclass
class LearningPath:
    """学习路径"""
    path_id: str
    skill_sequence: List[str]
    total_difficulty: float = 0.0
    estimated_time: float = 0.0
    prerequisite_satisfaction: float = 1.0
    
    def __len__(self) -> int:
        return len(self.skill_sequence)


class SkillGraph:
    """技能关系图"""
    
    def __init__(self):
        self.nodes: Dict[str, SkillNode] = {}
        self.adjacency_list: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)
        
    def add_skill(self, skill: SkillNode) -> bool:
        """添加技能节点"""
        if skill.skill_id in self.nodes:
            return False
        
        self.nodes[skill.skill_id] = skill
        
        # 建立依赖关系
        for prereq in skill.prerequisites:
            self.adjacency_list[prereq].add(skill.skill_id)
            self.reverse_adjacency[skill.skill_id].add(prereq)
        
        return True
    
    def remove_skill(self, skill_id: str) -> bool:
        """移除技能节点"""
        if skill_id not in self.nodes:
            return False
        
        skill = self.nodes[skill_id]
        
        # 移除依赖关系
        for prereq in skill.prerequisites:
            self.adjacency_list[prereq].discard(skill_id)
            self.reverse_adjacency[skill_id].discard(prereq)
        
        # 移除解锁关系
        for unlocked in skill.unlocks:
            if unlocked in self.nodes:
                self.nodes[unlocked].prerequisites.discard(skill_id)
            self.adjacency_list[skill_id].discard(unlocked)
        
        del self.nodes[skill_id]
        return True
    
    def add_dependency(self, from_skill: str, to_skill: str) -> bool:
        """添加依赖关系: from_skill 是 to_skill 的前置条件"""
        if from_skill not in self.nodes or to_skill not in self.nodes:
            return False
        
        # 检查是否会形成环
        if self._would_create_cycle(from_skill, to_skill):
            return False
        
        self.nodes[to_skill].prerequisites.add(from_skill)
        self.nodes[from_skill].unlocks.add(to_skill)
        self.adjacency_list[from_skill].add(to_skill)
        self.reverse_adjacency[to_skill].add(from_skill)
        
        return True
    
    def remove_dependency(self, from_skill: str, to_skill: str) -> bool:
        """移除依赖关系"""
        if from_skill not in self.nodes or to_skill not in self.nodes:
            return False
        
        self.nodes[to_skill].prerequisites.discard(from_skill)
        self.nodes[from_skill].unlocks.discard(to_skill)
        self.adjacency_list[from_skill].discard(to_skill)
        self.reverse_adjacency[to_skill].discard(from_skill)
        
        return True
    
    def _would_create_cycle(self, from_skill: str, to_skill: str) -> bool:
        """检查添加依赖是否会形成环"""
        # 如果 to_skill 可以到达 from_skill，则添加边会形成环
        return self._is_reachable(to_skill, from_skill)
    
    def _is_reachable(self, start: str, target: str, visited: Optional[Set[str]] = None) -> bool:
        """检查从start是否可以到达target"""
        if visited is None:
            visited = set()
        
        if start == target:
            return True
        
        if start in visited:
            return False
        
        visited.add(start)
        
        for neighbor in self.adjacency_list.get(start, set()):
            if self._is_reachable(neighbor, target, visited):
                return True
        
        return False
    
    def get_prerequisites(self, skill_id: str, include_indirect: bool = True) -> Set[str]:
        """获取技能的所有前置条件"""
        if skill_id not in self.nodes:
            return set()
        
        if not include_indirect:
            return set(self.nodes[skill_id].prerequisites)
        
        # BFS获取所有间接前置条件
        prerequisites = set()
        queue = deque(self.nodes[skill_id].prerequisites)
        
        while queue:
            prereq = queue.popleft()
            if prereq not in prerequisites:
                prerequisites.add(prereq)
                if prereq in self.nodes:
                    queue.extend(self.nodes[prereq].prerequisites)
        
        return prerequisites
    
    def get_unlocked_skills(self, skill_id: str, include_indirect: bool = True) -> Set[str]:
        """获取技能解锁的所有技能"""
        if skill_id not in self.nodes:
            return set()
        
        if not include_indirect:
            return set(self.nodes[skill_id].unlocks)
        
        # BFS获取所有间接解锁技能
        unlocked = set()
        queue = deque(self.nodes[skill_id].unlocks)
        
        while queue:
            skill = queue.popleft()
            if skill not in unlocked:
                unlocked.add(skill)
                if skill in self.nodes:
                    queue.extend(self.nodes[skill].unlocks)
        
        return unlocked
    
    def topological_sort(self) -> List[str]:
        """拓扑排序所有技能"""
        in_degree = {skill_id: 0 for skill_id in self.nodes}
        
        for skill_id, skill in self.nodes.items():
            for prereq in skill.prerequisites:
                if prereq in self.nodes:
                    in_degree[skill_id] += 1
        
        # Kahn算法
        queue = deque([s for s, d in in_degree.items() if d == 0])
        result = []
        
        while queue:
            skill_id = queue.popleft()
            result.append(skill_id)
            
            for neighbor in self.adjacency_list.get(skill_id, set()):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result
    
    def find_learning_paths(self, target_skill: str, 
                           known_skills: Optional[Set[str]] = None) -> List[LearningPath]:
        """
        找到学习目标技能的所有可能路径
        
        使用动态规划找到所有满足依赖关系的学习路径
        """
        if target_skill not in self.nodes:
            return []
        
        known_skills = known_skills or set()
        
        # 获取所有需要学习的技能
        required_skills = self.get_prerequisites(target_skill)
        required_skills.add(target_skill)
        
        # 移除已知的技能
        skills_to_learn = required_skills - known_skills
        
        if not skills_to_learn:
            return [LearningPath(
                path_id=f"path_{target_skill}",
                skill_sequence=[target_skill],
                total_difficulty=0.0,
                estimated_time=0.0
            )]
        
        # 构建子图
        sub_graph = self._build_subgraph(skills_to_learn)
        
        # 找到所有拓扑排序
        paths = self._find_all_topological_sorts(sub_graph)
        
        # 构建LearningPath对象
        learning_paths = []
        for i, path in enumerate(paths[:10]):  # 限制路径数量
            total_difficulty = sum(self.nodes[s].difficulty for s in path)
            total_time = sum(self.nodes[s].estimated_learning_time for s in path)
            
            learning_paths.append(LearningPath(
                path_id=f"path_{target_skill}_{i}",
                skill_sequence=path,
                total_difficulty=total_difficulty,
                estimated_time=total_time
            ))
        
        return learning_paths
    
    def _build_subgraph(self, skill_ids: Set[str]) -> Dict[str, Set[str]]:
        """构建子图的邻接表"""
        subgraph = {}
        for skill_id in skill_ids:
            if skill_id in self.nodes:
                # 只包含也在子图中的依赖
                deps = {d for d in self.nodes[skill_id].prerequisites if d in skill_ids}
                subgraph[skill_id] = deps
        return subgraph
    
    def _find_all_topological_sorts(self, graph: Dict[str, Set[str]], 
                                     max_paths: int = 10) -> List[List[str]]:
        """找到所有拓扑排序"""
        in_degree = {s: 0 for s in graph}
        for deps in graph.values():
            for d in deps:
                in_degree[d] = in_degree.get(d, 0) + 1
        
        paths = []
        
        def backtrack(current_path: List[str], current_in_degree: Dict[str, int], 
                     remaining: Set[str]):
            if len(paths) >= max_paths:
                return
            
            if not remaining:
                paths.append(current_path.copy())
                return
            
            # 找到所有入度为0的节点
            available = [s for s in remaining if current_in_degree.get(s, 0) == 0]
            
            for skill in available:
                # 选择这个技能
                current_path.append(skill)
                remaining.remove(skill)
                
                # 更新入度
                new_in_degree = current_in_degree.copy()
                for dep in graph.get(skill, set()):
                    if dep in new_in_degree:
                        new_in_degree[dep] -= 1
                
                backtrack(current_path, new_in_degree, remaining)
                
                # 回溯
                current_path.pop()
                remaining.add(skill)
        
        backtrack([], in_degree, set(graph.keys()))
        return paths
    
    def find_optimal_path(self, target_skills: List[str], 
                         known_skills: Optional[Set[str]] = None,
                         optimization: str = 'time') -> LearningPath:
        """
        找到最优学习路径
        
        Args:
            target_skills: 目标技能列表
            known_skills: 已掌握的技能
            optimization: 优化目标 ('time', 'difficulty', 'balanced')
        """
        known_skills = known_skills or set()
        
        # 获取所有需要的技能
        all_required = set()
        for target in target_skills:
            all_required.update(self.get_prerequisites(target))
            all_required.add(target)
        
        skills_to_learn = all_required - known_skills
        
        if not skills_to_learn:
            return LearningPath(
                path_id="optimal_path",
                skill_sequence=[],
                total_difficulty=0.0,
                estimated_time=0.0
            )
        
        # 使用A*算法找到最优路径
        return self._astar_pathfinding(skills_to_learn, target_skills, optimization)
    
    def _astar_pathfinding(self, skills_to_learn: Set[str], 
                          target_skills: List[str],
                          optimization: str) -> LearningPath:
        """使用A*算法找最优路径"""
        
        def heuristic(learned: Set[str]) -> float:
            """启发函数: 估计剩余成本"""
            remaining = skills_to_learn - learned
            if optimization == 'time':
                return sum(self.nodes[s].estimated_learning_time for s in remaining)
            elif optimization == 'difficulty':
                return sum(self.nodes[s].difficulty for s in remaining)
            else:  # balanced
                return sum(self.nodes[s].difficulty * self.nodes[s].estimated_learning_time 
                          for s in remaining)
        
        def cost(skill: str) -> float:
            """单个技能的学习成本"""
            if optimization == 'time':
                return self.nodes[skill].estimated_learning_time
            elif optimization == 'difficulty':
                return self.nodes[skill].difficulty
            else:
                return self.nodes[skill].difficulty * self.nodes[skill].estimated_learning_time
        
        # A*搜索
        initial_state = frozenset()
        goal_state = frozenset(skills_to_learn)
        
        # 优先队列: (f_score, g_score, state, path)
        open_set = [(heuristic(initial_state), 0.0, initial_state, [])]
        visited = set()
        
        while open_set:
            f_score, g_score, current, path = heapq.heappop(open_set)
            
            if current == goal_state:
                total_difficulty = sum(self.nodes[s].difficulty for s in path)
                total_time = sum(self.nodes[s].estimated_learning_time for s in path)
                return LearningPath(
                    path_id="optimal_path",
                    skill_sequence=path,
                    total_difficulty=total_difficulty,
                    estimated_time=total_time
                )
            
            if current in visited:
                continue
            visited.add(current)
            
            # 找到可以学习的技能(所有前置条件已满足)
            available = skills_to_learn - current
            for skill in available:
                prereqs = self.nodes[skill].prerequisites
                if prereqs.issubset(current):
                    new_state = frozenset(current | {skill})
                    new_g = g_score + cost(skill)
                    new_f = new_g + heuristic(new_state)
                    heapq.heappush(open_set, (new_f, new_g, new_state, path + [skill]))
        
        # 如果没有找到路径，返回空路径
        return LearningPath(
            path_id="optimal_path",
            skill_sequence=[],
            total_difficulty=float('inf'),
            estimated_time=float('inf')
        )
    
    def get_skill_clusters(self) -> Dict[str, List[str]]:
        """获取技能聚类(基于类别)"""
        clusters = defaultdict(list)
        for skill_id, skill in self.nodes.items():
            clusters[skill.category].append(skill_id)
        return dict(clusters)
    
    def find_skill_gaps(self, current_skills: Set[str], 
                       target_skills: Set[str]) -> Set[str]:
        """找到从当前技能到目标技能需要学习的技能"""
        required = set()
        for target in target_skills:
            if target not in current_skills:
                required.update(self.get_prerequisites(target))
                required.add(target)
        return required - current_skills
    
    def compute_skill_depth(self, skill_id: str) -> int:
        """计算技能深度(最长依赖链长度)"""
        if skill_id not in self.nodes:
            return -1
        
        memo = {}
        
        def depth(s: str) -> int:
            if s in memo:
                return memo[s]
            
            if s not in self.nodes or not self.nodes[s].prerequisites:
                memo[s] = 0
                return 0
            
            max_prereq_depth = max(depth(p) for p in self.nodes[s].prerequisites if p in self.nodes)
            memo[s] = max_prereq_depth + 1
            return memo[s]
        
        return depth(skill_id)
    
    def export_to_dict(self) -> Dict[str, Any]:
        """导出图为字典"""
        return {
            'skills': {
                skill_id: {
                    'name': skill.name,
                    'description': skill.description,
                    'difficulty': skill.difficulty,
                    'category': skill.category,
                    'estimated_learning_time': skill.estimated_learning_time,
                    'prerequisites': list(skill.prerequisites),
                    'unlocks': list(skill.unlocks),
                    'related_skills': list(skill.related_skills),
                    'complementary_skills': list(skill.complementary_skills),
                    'tags': list(skill.tags)
                }
                for skill_id, skill in self.nodes.items()
            }
        }
    
    def import_from_dict(self, data: Dict[str, Any]):
        """从字典导入图"""
        self.nodes.clear()
        self.adjacency_list.clear()
        self.reverse_adjacency.clear()
        
        for skill_id, skill_data in data.get('skills', {}).items():
            skill = SkillNode(
                skill_id=skill_id,
                name=skill_data['name'],
                description=skill_data.get('description', ''),
                difficulty=skill_data.get('difficulty', 0.5),
                category=skill_data.get('category', 'general'),
                estimated_learning_time=skill_data.get('estimated_learning_time', 1.0),
                prerequisites=set(skill_data.get('prerequisites', [])),
                unlocks=set(skill_data.get('unlocks', [])),
                related_skills=set(skill_data.get('related_skills', [])),
                complementary_skills=set(skill_data.get('complementary_skills', [])),
                tags=set(skill_data.get('tags', []))
            )
            self.add_skill(skill)


class SkillPathPlanner:
    """技能路径规划器"""
    
    def __init__(self, skill_graph: SkillGraph):
        self.graph = skill_graph
        self.user_progress: Dict[str, Set[str]] = {}  # user_id -> learned_skills
        
    def register_user(self, user_id: str, known_skills: Optional[Set[str]] = None):
        """注册用户"""
        self.user_progress[user_id] = known_skills or set()
    
    def update_progress(self, user_id: str, learned_skill: str):
        """更新学习进度"""
        if user_id in self.user_progress:
            self.user_progress[user_id].add(learned_skill)
    
    def get_next_recommendations(self, user_id: str, 
                                  target_skill: Optional[str] = None,
                                  num_recommendations: int = 3) -> List[Dict[str, Any]]:
        """获取下一步学习推荐"""
        if user_id not in self.user_progress:
            return []
        
        known = self.user_progress[user_id]
        
        # 找到所有可以学习的技能
        available = []
        for skill_id, skill in self.graph.nodes.items():
            if skill_id not in known:
                # 检查前置条件是否满足
                if skill.prerequisites.issubset(known):
                    available.append(skill_id)
        
        # 如果有目标技能，优先推荐通向目标的路径
        if target_skill and target_skill in self.graph.nodes:
            # 计算每个可用技能对目标的贡献
            scored_available = []
            for skill_id in available:
                # 检查是否是目标的前置条件
                is_prereq = skill_id in self.graph.get_prerequisites(target_skill)
                
                # 计算解锁的技能数量
                unlocks_count = len(self.graph.get_unlocked_skills(skill_id))
                
                score = (2.0 if is_prereq else 0.0) + unlocks_count * 0.1
                scored_available.append((score, skill_id))
            
            scored_available.sort(reverse=True)
            available = [s for _, s in scored_available]
        
        # 构建推荐
        recommendations = []
        for skill_id in available[:num_recommendations]:
            skill = self.graph.nodes[skill_id]
            recommendations.append({
                'skill_id': skill_id,
                'name': skill.name,
                'difficulty': skill.difficulty,
                'estimated_time': skill.estimated_learning_time,
                'unlocks': list(skill.unlocks),
                'reason': self._generate_recommendation_reason(skill_id, target_skill, known)
            })
        
        return recommendations
    
    def _generate_recommendation_reason(self, skill_id: str, 
                                        target_skill: Optional[str],
                                        known_skills: Set[str]) -> str:
        """生成推荐理由"""
        if target_skill and skill_id in self.graph.get_prerequisites(target_skill):
            return f"Required for {self.graph.nodes[target_skill].name}"
        
        skill = self.graph.nodes[skill_id]
        if skill.unlocks:
            return f"Unlocks {len(skill.unlocks)} new skills"
        
        return "Foundation skill"
    
    def estimate_completion_time(self, user_id: str, 
                                 target_skill: str) -> Dict[str, Any]:
        """估计完成目标技能所需时间"""
        if user_id not in self.user_progress:
            return {'error': 'User not found'}
        
        known = self.user_progress[user_id]
        paths = self.graph.find_learning_paths(target_skill, known)
        
        if not paths:
            return {
                'target': target_skill,
                'already_known': target_skill in known,
                'estimated_time': 0.0,
                'skills_to_learn': 0
            }
        
        # 找到最快路径
        fastest = min(paths, key=lambda p: p.estimated_time)
        
        return {
            'target': target_skill,
            'target_name': self.graph.nodes[target_skill].name,
            'estimated_time': fastest.estimated_time,
            'skills_to_learn': len(fastest),
            'skill_sequence': fastest.skill_sequence,
            'total_difficulty': fastest.total_difficulty
        }
    
    def get_learning_milestones(self, user_id: str, 
                                target_skill: str) -> List[Dict[str, Any]]:
        """获取学习里程碑"""
        if user_id not in self.user_progress:
            return []
        
        known = self.user_progress[user_id]
        paths = self.graph.find_learning_paths(target_skill, known)
        
        if not paths:
            return []
        
        # 使用最优路径
        path = min(paths, key=lambda p: p.estimated_time)
        
        milestones = []
        cumulative_time = 0.0
        
        for i, skill_id in enumerate(path.skill_sequence):
            skill = self.graph.nodes[skill_id]
            cumulative_time += skill.estimated_learning_time
            
            milestones.append({
                'order': i + 1,
                'skill_id': skill_id,
                'name': skill.name,
                'difficulty': skill.difficulty,
                'cumulative_time': cumulative_time,
                'unlocks_immediate': list(skill.unlocks)
            })
        
        return milestones
