"""
共享经验回放 - 联盟成员共享成功与失败经验

实现多Agent间的经验共享机制，允许联盟成员共享和检索
成功与失败的经验，加速集体学习。
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import heapq
import time
import hashlib
import json


@dataclass
class Experience:
    """经验数据类"""
    experience_id: str
    agent_id: str
    
    # 状态-动作-奖励
    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool
    
    # 元数据
    timestamp: float = field(default_factory=time.time)
    task_id: str = ""
    episode_id: str = ""
    step_number: int = 0
    
    # 质量评估
    success: bool = True
    importance: float = 1.0  # 重要性采样权重
    
    # 标签
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.experience_id)
    
    def __eq__(self, other):
        if isinstance(other, Experience):
            return self.experience_id == other.experience_id
        return False


@dataclass
class ExperienceSummary:
    """经验摘要"""
    experience_id: str
    agent_id: str
    timestamp: float
    success: bool
    reward: float
    tags: Set[str]
    task_id: str


class ExperienceBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 10000, 
                 priority_alpha: float = 0.6,
                 priority_beta: float = 0.4):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.priorities: Dict[str, float] = {}
        
        # 优先级采样参数
        self.priority_alpha = priority_alpha  # 优先级指数
        self.priority_beta = priority_beta    # 重要性采样指数
        self.priority_beta_increment = 0.001
        
        # 统计
        self.total_added = 0
        self.total_sampled = 0
        
    def add(self, experience: Experience, priority: Optional[float] = None):
        """添加经验"""
        # 生成唯一ID
        if not experience.experience_id:
            experience.experience_id = self._generate_id(experience)
        
        self.buffer.append(experience)
        
        # 设置优先级 (默认最大优先级)
        if priority is None:
            priority = max(self.priorities.values()) if self.priorities else 1.0
        self.priorities[experience.experience_id] = priority ** self.priority_alpha
        
        self.total_added += 1
    
    def _generate_id(self, experience: Experience) -> str:
        """生成经验ID"""
        data = f"{experience.agent_id}_{experience.timestamp}_{experience.task_id}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    def sample(self, batch_size: int, 
               agent_filter: Optional[Set[str]] = None) -> List[Tuple[Experience, float]]:
        """
        优先级采样
        
        返回: (经验, 重要性采样权重) 列表
        """
        if len(self.buffer) == 0:
            return []
        
        # 过滤经验
        available = list(self.buffer)
        if agent_filter:
            available = [e for e in available if e.agent_id in agent_filter]
        
        if not available:
            return []
        
        # 计算采样概率
        total_priority = sum(self.priorities.get(e.experience_id, 1.0) 
                            for e in available)
        
        probabilities = []
        for exp in available:
            p = self.priorities.get(exp.experience_id, 1.0)
            probabilities.append(p / total_priority)
        
        # 采样
        batch_size = min(batch_size, len(available))
        sampled_indices = self._weighted_sample(len(available), probabilities, batch_size)
        
        # 计算重要性采样权重
        samples = []
        for idx in sampled_indices:
            exp = available[idx]
            prob = probabilities[idx]
            # 重要性采样权重 w = (N * P(i))^(-beta)
            weight = (len(available) * prob) ** (-self.priority_beta)
            samples.append((exp, weight))
        
        # 归一化权重
        max_weight = max(w for _, w in samples) if samples else 1.0
        samples = [(exp, w / max_weight) for exp, w in samples]
        
        self.total_sampled += batch_size
        
        # 逐渐增加beta
        self.priority_beta = min(1.0, self.priority_beta + self.priority_beta_increment)
        
        return samples
    
    def _weighted_sample(self, n: int, probabilities: List[float], 
                        k: int) -> List[int]:
        """加权随机采样"""
        if n == 0 or k == 0:
            return []
        
        indices = list(range(n))
        sampled = []
        
        for _ in range(k):
            if not indices:
                break
            # 根据概率选择
            probs = [probabilities[i] for i in indices]
            total = sum(probs)
            probs = [p / total for p in probs]
            
            r = random.random()
            cumsum = 0.0
            for i, p in zip(indices, probs):
                cumsum += p
                if r <= cumsum:
                    sampled.append(i)
                    indices.remove(i)
                    break
        
        return sampled
    
    def update_priority(self, experience_id: str, priority: float):
        """更新经验优先级"""
        self.priorities[experience_id] = priority ** self.priority_alpha
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.buffer:
            return {'size': 0}
        
        rewards = [e.reward for e in self.buffer]
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'total_added': self.total_added,
            'total_sampled': self.total_sampled,
            'avg_reward': sum(rewards) / len(rewards),
            'max_reward': max(rewards),
            'min_reward': min(rewards),
            'success_rate': sum(1 for e in self.buffer if e.success) / len(self.buffer)
        }
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
        self.priorities.clear()
        self.total_added = 0
        self.total_sampled = 0


class SharedExperiencePool:
    """共享经验池 - 联盟成员间共享经验"""
    
    def __init__(self):
        self.buffers: Dict[str, ExperienceBuffer] = {}  # 按任务分类
        self.agent_contributions: Dict[str, List[str]] = defaultdict(list)
        self.experience_index: Dict[str, ExperienceSummary] = {}
        
        # 标签索引
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        
        # 成功/失败分离
        self.success_experiences: Set[str] = set()
        self.failure_experiences: Set[str] = set()
        
    def register_buffer(self, task_id: str, capacity: int = 10000):
        """注册任务缓冲区"""
        if task_id not in self.buffers:
            self.buffers[task_id] = ExperienceBuffer(capacity=capacity)
    
    def share_experience(self, experience: Experience) -> bool:
        """分享经验到共享池"""
        task_id = experience.task_id or 'general'
        
        if task_id not in self.buffers:
            self.register_buffer(task_id)
        
        # 添加经验
        self.buffers[task_id].add(experience)
        
        # 记录贡献
        self.agent_contributions[experience.agent_id].append(experience.experience_id)
        
        # 创建摘要
        summary = ExperienceSummary(
            experience_id=experience.experience_id,
            agent_id=experience.agent_id,
            timestamp=experience.timestamp,
            success=experience.success,
            reward=experience.reward,
            tags=experience.tags,
            task_id=task_id
        )
        self.experience_index[experience.experience_id] = summary
        
        # 更新标签索引
        for tag in experience.tags:
            self.tag_index[tag].add(experience.experience_id)
        
        # 分类
        if experience.success:
            self.success_experiences.add(experience.experience_id)
        else:
            self.failure_experiences.add(experience.experience_id)
        
        return True
    
    def query_experiences(self, 
                         task_id: Optional[str] = None,
                         tags: Optional[Set[str]] = None,
                         success_only: bool = False,
                         failure_only: bool = False,
                         min_reward: Optional[float] = None,
                         max_reward: Optional[float] = None,
                         agent_id: Optional[str] = None,
                         limit: int = 100) -> List[ExperienceSummary]:
        """查询经验"""
        candidates = set(self.experience_index.keys())
        
        # 按任务过滤
        if task_id:
            task_exps = set()
            for exp_id, summary in self.experience_index.items():
                if summary.task_id == task_id:
                    task_exps.add(exp_id)
            candidates &= task_exps
        
        # 按标签过滤
        if tags:
            tag_exps = set()
            for tag in tags:
                tag_exps.update(self.tag_index.get(tag, set()))
            candidates &= tag_exps
        
        # 按成功/失败过滤
        if success_only:
            candidates &= self.success_experiences
        if failure_only:
            candidates &= self.failure_experiences
        
        # 过滤并排序
        results = []
        for exp_id in candidates:
            summary = self.experience_index[exp_id]
            
            # 按奖励过滤
            if min_reward is not None and summary.reward < min_reward:
                continue
            if max_reward is not None and summary.reward > max_reward:
                continue
            
            # 按Agent过滤
            if agent_id is not None and summary.agent_id != agent_id:
                continue
            
            results.append(summary)
        
        # 按奖励排序
        results.sort(key=lambda x: x.reward, reverse=True)
        
        return results[:limit]
    
    def retrieve_experience(self, experience_id: str) -> Optional[Experience]:
        """检索完整经验"""
        summary = self.experience_index.get(experience_id)
        if not summary:
            return None
        
        task_id = summary.task_id
        if task_id not in self.buffers:
            return None
        
        # 在缓冲区中查找
        for exp in self.buffers[task_id].buffer:
            if exp.experience_id == experience_id:
                return exp
        
        return None
    
    def sample_for_training(self, task_id: str, batch_size: int,
                           exclude_agents: Optional[Set[str]] = None) -> List[Tuple[Experience, float]]:
        """为训练采样经验"""
        if task_id not in self.buffers:
            return []
        
        buffer = self.buffers[task_id]
        
        # 确定允许的Agent
        all_agents = set(self.agent_contributions.keys())
        if exclude_agents:
            allowed_agents = all_agents - exclude_agents
        else:
            allowed_agents = all_agents
        
        return buffer.sample(batch_size, allowed_agents)
    
    def get_success_patterns(self, task_id: str, 
                            min_occurrences: int = 3) -> List[Dict[str, Any]]:
        """提取成功模式"""
        if task_id not in self.buffers:
            return []
        
        # 分析成功经验的共同特征
        success_exps = [
            e for e in self.buffers[task_id].buffer
            if e.success and e.experience_id in self.success_experiences
        ]
        
        if not success_exps:
            return []
        
        # 按动作分组统计
        action_counts = defaultdict(int)
        action_rewards = defaultdict(list)
        
        for exp in success_exps:
            action_key = self._action_to_key(exp.action)
            action_counts[action_key] += 1
            action_rewards[action_key].append(exp.reward)
        
        # 找出频繁模式
        patterns = []
        for action_key, count in action_counts.items():
            if count >= min_occurrences:
                avg_reward = sum(action_rewards[action_key]) / len(action_rewards[action_key])
                patterns.append({
                    'action_pattern': action_key,
                    'occurrences': count,
                    'avg_reward': avg_reward,
                    'confidence': count / len(success_exps)
                })
        
        patterns.sort(key=lambda x: x['confidence'], reverse=True)
        return patterns
    
    def _action_to_key(self, action: Any) -> str:
        """将动作转换为可哈希的键"""
        if isinstance(action, (str, int, float)):
            return str(action)
        try:
            return json.dumps(action, sort_keys=True)
        except:
            return str(hash(str(action)))
    
    def get_failure_lessons(self, task_id: str) -> List[Dict[str, Any]]:
        """提取失败教训"""
        if task_id not in self.buffers:
            return []
        
        failure_exps = [
            e for e in self.buffers[task_id].buffer
            if not e.success and e.experience_id in self.failure_experiences
        ]
        
        if not failure_exps:
            return []
        
        # 分析失败原因
        lessons = []
        for exp in failure_exps:
            lesson = {
                'experience_id': exp.experience_id,
                'agent_id': exp.agent_id,
                'state_summary': self._summarize_state(exp.state),
                'action': exp.action,
                'reward': exp.reward,
                'lesson': exp.metadata.get('failure_reason', 'Unknown failure')
            }
            lessons.append(lesson)
        
        return lessons
    
    def _summarize_state(self, state: Any) -> str:
        """总结状态"""
        if isinstance(state, dict):
            return f"dict_with_{len(state)}_keys"
        elif isinstance(state, (list, tuple)):
            return f"list_of_{len(state)}"
        return str(type(state).__name__)
    
    def get_agent_contribution_stats(self, agent_id: str) -> Dict[str, Any]:
        """获取Agent贡献统计"""
        exp_ids = self.agent_contributions.get(agent_id, [])
        
        if not exp_ids:
            return {'agent_id': agent_id, 'contributions': 0}
        
        summaries = [self.experience_index[eid] for eid in exp_ids 
                    if eid in self.experience_index]
        
        task_counts = defaultdict(int)
        total_reward = 0.0
        success_count = 0
        
        for summary in summaries:
            task_counts[summary.task_id] += 1
            total_reward += summary.reward
            if summary.success:
                success_count += 1
        
        return {
            'agent_id': agent_id,
            'total_experiences': len(summaries),
            'unique_tasks': len(task_counts),
            'task_distribution': dict(task_counts),
            'avg_reward': total_reward / len(summaries) if summaries else 0.0,
            'success_rate': success_count / len(summaries) if summaries else 0.0
        }
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """获取池统计信息"""
        total_experiences = sum(len(b.buffer) for b in self.buffers.values())
        
        task_stats = {}
        for task_id, buffer in self.buffers.items():
            task_stats[task_id] = buffer.get_statistics()
        
        return {
            'total_experiences': total_experiences,
            'total_tasks': len(self.buffers),
            'total_agents': len(self.agent_contributions),
            'success_experiences': len(self.success_experiences),
            'failure_experiences': len(self.failure_experiences),
            'task_statistics': task_stats,
            'agent_contributions': {
                agent: len(exps) 
                for agent, exps in self.agent_contributions.items()
            }
        }
    
    def merge_similar_experiences(self, similarity_threshold: float = 0.9):
        """合并相似经验(去重)"""
        # 简化的去重: 基于状态和动作的哈希
        seen_hashes = set()
        duplicates = []
        
        for task_id, buffer in self.buffers.items():
            for exp in list(buffer.buffer):
                exp_hash = self._compute_experience_hash(exp)
                if exp_hash in seen_hashes:
                    duplicates.append((task_id, exp.experience_id))
                else:
                    seen_hashes.add(exp_hash)
        
        # 移除重复
        for task_id, exp_id in duplicates:
            buffer = self.buffers[task_id]
            buffer.buffer = deque(
                [e for e in buffer.buffer if e.experience_id != exp_id],
                maxlen=buffer.capacity
            )
        
        return len(duplicates)
    
    def _compute_experience_hash(self, exp: Experience) -> str:
        """计算经验哈希"""
        data = f"{exp.task_id}_{exp.action}_{exp.reward}"
        return hashlib.md5(data.encode()).hexdigest()[:16]


class ExperienceTransfer:
    """经验迁移 - 跨任务经验重用"""
    
    def __init__(self, pool: SharedExperiencePool):
        self.pool = pool
        self.task_similarities: Dict[Tuple[str, str], float] = {}
        
    def compute_task_similarity(self, task1: str, task2: str) -> float:
        """计算任务相似度"""
        if (task1, task2) in self.task_similarities:
            return self.task_similarities[(task1, task2)]
        
        # 基于经验的相似度计算
        exps1 = self.pool.query_experiences(task_id=task1, limit=1000)
        exps2 = self.pool.query_experiences(task_id=task2, limit=1000)
        
        if not exps1 or not exps2:
            return 0.0
        
        # 基于奖励分布的相似度
        rewards1 = [e.reward for e in exps1]
        rewards2 = [e.reward for e in exps2]
        
        avg1 = sum(rewards1) / len(rewards1)
        avg2 = sum(rewards2) / len(rewards2)
        
        # 简单的相似度: 奖励分布的接近程度
        similarity = 1.0 / (1.0 + abs(avg1 - avg2))
        
        self.task_similarities[(task1, task2)] = similarity
        self.task_similarities[(task2, task1)] = similarity
        
        return similarity
    
    def transfer_experiences(self, source_task: str, target_task: str,
                            max_transfer: int = 100) -> List[Experience]:
        """迁移经验从源任务到目标任务"""
        similarity = self.compute_task_similarity(source_task, target_task)
        
        if similarity < 0.3:
            return []  # 相似度太低，不迁移
        
        # 获取源任务的最佳经验
        source_exps = self.pool.query_experiences(
            task_id=source_task,
            success_only=True,
            limit=max_transfer * 2
        )
        
        # 选择高质量经验
        selected = sorted(source_exps, 
                         key=lambda x: x.reward * similarity, 
                         reverse=True)[:max_transfer]
        
        # 检索完整经验
        transferred = []
        for summary in selected:
            exp = self.pool.retrieve_experience(summary.experience_id)
            if exp:
                # 修改任务ID
                exp.task_id = target_task
                exp.tags.add(f"transferred_from_{source_task}")
                transferred.append(exp)
        
        return transferred
    
    def find_related_tasks(self, task: str, min_similarity: float = 0.5) -> List[Tuple[str, float]]:
        """找到相关任务"""
        related = []
        
        for other_task in self.pool.buffers.keys():
            if other_task != task:
                sim = self.compute_task_similarity(task, other_task)
                if sim >= min_similarity:
                    related.append((other_task, sim))
        
        related.sort(key=lambda x: x[1], reverse=True)
        return related
