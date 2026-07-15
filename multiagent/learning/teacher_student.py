"""
师生学习框架 - 专家Agent向新手进行知识蒸馏

实现基于知识蒸馏(Knowledge Distillation)的师生学习框架，
专家Agent(教师)通过软标签向新手Agent(学生)传递知识。
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import random
import math


@dataclass
class KnowledgeState:
    """知识状态表示"""
    expertise_level: float  # 0.0 - 1.0
    confidence: float  # 对知识的置信度
    experience_count: int = 0  # 经验数量
    success_rate: float = 0.0  # 成功率


@dataclass
class TeachingSession:
    """教学会话记录"""
    teacher_id: str
    student_id: str
    topic: str
    examples: List[Dict[str, Any]] = field(default_factory=list)
    feedback_scores: List[float] = field(default_factory=list)
    completed: bool = False


class ExpertAgent:
    """专家Agent(教师)"""
    
    def __init__(self, agent_id: str, expertise_domains: List[str]):
        self.agent_id = agent_id
        self.expertise_domains = set(expertise_domains)
        self.knowledge_base: Dict[str, Dict[str, Any]] = {}
        self.teaching_history: List[TeachingSession] = []
        self.temperature = 1.0  # 软标签温度参数
        
    def add_knowledge(self, domain: str, key: str, value: Any, confidence: float = 1.0):
        """添加知识到知识库"""
        if domain not in self.knowledge_base:
            self.knowledge_base[domain] = {}
        self.knowledge_base[domain][key] = {
            'value': value,
            'confidence': confidence,
            'usage_count': 0
        }
        
    def get_soft_labels(self, inputs: List[Any], domain: str) -> List[Dict[int, float]]:
        """
        生成软标签(概率分布)用于知识蒸馏
        
        使用温度参数T来软化概率分布，使学生能学到更多细微差别
        """
        soft_labels = []
        for inp in inputs:
            # 模拟专家的知识分布
            hard_logits = self._compute_expert_logits(inp, domain)
            # 应用温度缩放
            soft_probs = self._softmax_with_temperature(hard_logits, self.temperature)
            soft_labels.append(soft_probs)
        return soft_labels
    
    def _compute_expert_logits(self, input_data: Any, domain: str) -> Dict[int, float]:
        """计算专家logits(模拟)"""
        # 在实际实现中，这里会使用专家模型进行推理
        if domain in self.knowledge_base:
            knowledge = self.knowledge_base[domain]
            # 基于知识库生成logits
            logits = {}
            for i, (key, info) in enumerate(knowledge.items()):
                logits[i] = info['confidence'] * random.uniform(0.8, 1.0)
            return logits
        return {0: 1.0}
    
    def _softmax_with_temperature(self, logits: Dict[int, float], temperature: float) -> Dict[int, float]:
        """带温度的softmax"""
        exp_values = {k: math.exp(v / temperature) for k, v in logits.items()}
        total = sum(exp_values.values())
        return {k: v / total for k, v in exp_values.items()}
    
    def teach(self, student: 'NoviceAgent', topic: str, examples: List[Dict[str, Any]]) -> TeachingSession:
        """向学生教授知识"""
        session = TeachingSession(
            teacher_id=self.agent_id,
            student_id=student.agent_id,
            topic=topic
        )
        
        # 生成软标签
        inputs = [ex['input'] for ex in examples]
        soft_labels = self.get_soft_labels(inputs, topic)
        
        # 学生使用软标签学习
        for example, soft_label in zip(examples, soft_labels):
            student.learn_from_soft_label(example, soft_label, topic)
            session.examples.append({
                'input': example['input'],
                'soft_label': soft_label
            })
        
        self.teaching_history.append(session)
        return session
    
    def provide_feedback(self, student_action: Any, context: Any) -> Dict[str, Any]:
        """为学生行为提供反馈"""
        # 评估学生行为
        expert_action = self._expert_decision(context)
        similarity = self._compute_action_similarity(student_action, expert_action)
        
        return {
            'score': similarity,
            'expert_action': expert_action,
            'suggestions': self._generate_suggestions(student_action, expert_action),
            'improvement_areas': self._identify_improvements(student_action, expert_action)
        }
    
    def _expert_decision(self, context: Any) -> Any:
        """专家决策"""
        # 基于知识库做出决策
        return {'action': 'expert_action', 'confidence': 0.95}
    
    def _compute_action_similarity(self, action1: Any, action2: Any) -> float:
        """计算行为相似度"""
        if isinstance(action1, dict) and isinstance(action2, dict):
            if action1.get('action') == action2.get('action'):
                return 0.9 + random.uniform(0, 0.1)
        return random.uniform(0.3, 0.7)
    
    def _generate_suggestions(self, student_action: Any, expert_action: Any) -> List[str]:
        """生成改进建议"""
        suggestions = []
        if student_action != expert_action:
            suggestions.append("Consider the expert approach")
        return suggestions
    
    def _identify_improvements(self, student_action: Any, expert_action: Any) -> List[str]:
        """识别需要改进的领域"""
        improvements = []
        if isinstance(student_action, dict) and isinstance(expert_action, dict):
            for key in expert_action:
                if key not in student_action or student_action[key] != expert_action[key]:
                    improvements.append(key)
        return improvements


class NoviceAgent:
    """新手Agent(学生)"""
    
    def __init__(self, agent_id: str, learning_rate: float = 0.1):
        self.agent_id = agent_id
        self.learning_rate = learning_rate
        self.knowledge_state: Dict[str, KnowledgeState] = {}
        self.policy: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.experience_buffer: List[Dict[str, Any]] = []
        self.distillation_weight = 0.7  # 蒸馏损失权重
        
    def learn_from_soft_label(self, example: Dict[str, Any], 
                              soft_label: Dict[int, float], 
                              topic: str):
        """从教师的软标签学习(知识蒸馏)"""
        # 计算学生当前预测
        student_logits = self._compute_logits(example['input'], topic)
        student_probs = self._softmax(student_logits)
        
        # 蒸馏损失: KL散度
        kl_loss = self._kl_divergence(soft_label, student_probs)
        
        # 如果有硬标签，计算交叉熵损失
        hard_loss = 0.0
        if 'label' in example:
            hard_loss = self._cross_entropy(example['label'], student_probs)
        
        # 组合损失
        total_loss = (self.distillation_weight * kl_loss + 
                     (1 - self.distillation_weight) * hard_loss)
        
        # 更新策略(梯度下降)
        self._update_policy(example['input'], soft_label, topic, total_loss)
        
        # 更新知识状态
        self._update_knowledge_state(topic, success=(total_loss < 0.5))
        
        # 存储经验
        self.experience_buffer.append({
            'example': example,
            'soft_label': soft_label,
            'loss': total_loss,
            'topic': topic
        })
    
    def _compute_logits(self, input_data: Any, topic: str) -> Dict[int, float]:
        """计算学生logits"""
        # 基于当前策略计算
        if topic in self.policy:
            return dict(self.policy[topic])
        return {0: 0.0}
    
    def _softmax(self, logits: Dict[int, float]) -> Dict[int, float]:
        """Softmax函数"""
        exp_values = {k: math.exp(v) for k, v in logits.items()}
        total = sum(exp_values.values())
        return {k: v / total for k, v in exp_values.items()}
    
    def _kl_divergence(self, p: Dict[int, float], q: Dict[int, float]) -> float:
        """计算KL散度"""
        kl = 0.0
        for k in p:
            if p[k] > 0:
                q_val = q.get(k, 1e-10)
                kl += p[k] * math.log(p[k] / max(q_val, 1e-10))
        return kl
    
    def _cross_entropy(self, true_label: int, probs: Dict[int, float]) -> float:
        """计算交叉熵"""
        prob = probs.get(true_label, 1e-10)
        return -math.log(max(prob, 1e-10))
    
    def _update_policy(self, input_data: Any, target: Dict[int, float], 
                       topic: str, loss: float):
        """更新策略"""
        # 简化的策略更新
        for action, prob in target.items():
            current = self.policy[topic][str(action)]
            self.policy[topic][str(action)] = current + self.learning_rate * prob * (1 - loss)
    
    def _update_knowledge_state(self, topic: str, success: bool):
        """更新知识状态"""
        if topic not in self.knowledge_state:
            self.knowledge_state[topic] = KnowledgeState(
                expertise_level=0.0,
                confidence=0.0
            )
        
        state = self.knowledge_state[topic]
        state.experience_count += 1
        
        # 更新成功率
        if state.experience_count == 1:
            state.success_rate = 1.0 if success else 0.0
        else:
            state.success_rate = (state.success_rate * (state.experience_count - 1) + 
                                 (1.0 if success else 0.0)) / state.experience_count
        
        # 更新专业水平
        state.expertise_level = min(1.0, state.expertise_level + 
                                   (0.1 if success else 0.02))
        state.confidence = min(1.0, state.confidence + 0.05)
    
    def request_teaching(self, teacher: ExpertAgent, topic: str, 
                        examples: List[Dict[str, Any]]) -> TeachingSession:
        """请求教师进行教学"""
        return teacher.teach(self, topic, examples)
    
    def get_knowledge_gap(self, required_skills: List[str]) -> List[str]:
        """获取知识缺口"""
        gaps = []
        for skill in required_skills:
            if skill not in self.knowledge_state:
                gaps.append(skill)
            elif self.knowledge_state[skill].expertise_level < 0.5:
                gaps.append(skill)
        return gaps


class TeacherStudentFramework:
    """师生学习框架主类"""
    
    def __init__(self):
        self.teachers: Dict[str, ExpertAgent] = {}
        self.students: Dict[str, NoviceAgent] = {}
        self.teaching_sessions: List[TeachingSession] = []
        self.domain_teachers: Dict[str, List[str]] = defaultdict(list)
        
    def register_teacher(self, teacher: ExpertAgent):
        """注册教师"""
        self.teachers[teacher.agent_id] = teacher
        for domain in teacher.expertise_domains:
            self.domain_teachers[domain].append(teacher.agent_id)
    
    def register_student(self, student: NoviceAgent):
        """注册学生"""
        self.students[student.agent_id] = student
    
    def find_best_teacher(self, topic: str, student_id: str) -> Optional[ExpertAgent]:
        """为给定主题找到最佳教师"""
        if topic not in self.domain_teachers:
            return None
        
        teacher_ids = self.domain_teachers[topic]
        if not teacher_ids:
            return None
        
        # 选择教学历史最丰富且未教过该学生的教师
        best_teacher = None
        best_score = -1
        
        for tid in teacher_ids:
            teacher = self.teachers[tid]
            # 计算教学分数
            score = len(teacher.teaching_history)
            # 惩罚已教过该学生的教师
            for session in teacher.teaching_history:
                if session.student_id == student_id:
                    score -= 10
            
            if score > best_score:
                best_score = score
                best_teacher = teacher
        
        return best_teacher
    
    def conduct_lesson(self, student_id: str, topic: str, 
                      examples: List[Dict[str, Any]]) -> Optional[TeachingSession]:
        """进行教学课程"""
        if student_id not in self.students:
            raise ValueError(f"Student {student_id} not found")
        
        teacher = self.find_best_teacher(topic, student_id)
        if teacher is None:
            return None
        
        student = self.students[student_id]
        session = teacher.teach(student, topic, examples)
        self.teaching_sessions.append(session)
        
        return session
    
    def evaluate_student_progress(self, student_id: str) -> Dict[str, Any]:
        """评估学生进度"""
        if student_id not in self.students:
            return {}
        
        student = self.students[student_id]
        progress = {
            'student_id': student_id,
            'overall_expertise': 0.0,
            'topics_learned': list(student.knowledge_state.keys()),
            'topic_details': {},
            'total_experiences': len(student.experience_buffer)
        }
        
        total_expertise = 0.0
        for topic, state in student.knowledge_state.items():
            progress['topic_details'][topic] = {
                'expertise_level': state.expertise_level,
                'confidence': state.confidence,
                'experience_count': state.experience_count,
                'success_rate': state.success_rate
            }
            total_expertise += state.expertise_level
        
        if student.knowledge_state:
            progress['overall_expertise'] = total_expertise / len(student.knowledge_state)
        
        return progress
    
    def get_framework_stats(self) -> Dict[str, Any]:
        """获取框架统计信息"""
        return {
            'total_teachers': len(self.teachers),
            'total_students': len(self.students),
            'total_sessions': len(self.teaching_sessions),
            'domains_covered': list(self.domain_teachers.keys()),
            'avg_session_examples': (
                sum(len(s.examples) for s in self.teaching_sessions) / 
                max(1, len(self.teaching_sessions))
            )
        }
