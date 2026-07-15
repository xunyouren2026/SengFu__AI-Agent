"""
教练智能体系统 - Agent Coach

专门分析其他Agent失败并给出优化建议。
实现了失败模式分析、性能诊断和个性化训练计划生成。
"""

from __future__ import annotations

import re
import statistics
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class FailureType(Enum):
    """失败类型"""
    TIMEOUT = auto()              # 超时
    EXCEPTION = auto()            # 异常
    WRONG_OUTPUT = auto()         # 错误输出
    QUALITY_ISSUE = auto()        # 质量问题
    RESOURCE_EXHAUSTION = auto()  # 资源耗尽
    COMMUNICATION_ERROR = auto()  # 通信错误
    LOGIC_ERROR = auto()          # 逻辑错误
    KNOWLEDGE_GAP = auto()        # 知识缺口
    CONTEXT_LOSS = auto()         # 上下文丢失


class SeverityLevel(Enum):
    """严重程度"""
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    INFO = 0


@dataclass
class FailureRecord:
    """失败记录"""
    failure_id: str
    agent_id: str
    failure_type: FailureType
    severity: SeverityLevel
    timestamp: float = field(default_factory=time.time)
    task_description: str = ""
    error_message: str = ""
    stack_trace: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    
    def to_analysis_dict(self) -> Dict[str, Any]:
        """转换为分析字典"""
        return {
            "failure_type": self.failure_type.name,
            "severity": self.severity.name,
            "timestamp": self.timestamp,
            "task": self.task_description[:200],  # 截断
            "error": self.error_message[:500],     # 截断
            "has_stack_trace": bool(self.stack_trace),
            "recovered": self.recovery_successful
        }


@dataclass
class PerformanceSnapshot:
    """性能快照"""
    agent_id: str
    timestamp: float = field(default_factory=time.time)
    success_rate: float = 0.0
    avg_latency: float = 0.0
    throughput: float = 0.0
    error_distribution: Dict[str, int] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    
    def calculate_health_score(self) -> float:
        """计算健康分数"""
        # 基于成功率、延迟和资源使用计算
        latency_score = max(0, 1.0 - self.avg_latency / 5.0)  # 假设5秒为阈值
        
        weights = {
            "success": 0.5,
            "latency": 0.3,
            "resources": 0.2
        }
        
        resource_score = 1.0 - min(
            sum(self.resource_usage.values()) / max(len(self.resource_usage), 1),
            1.0
        )
        
        return (
            self.success_rate * weights["success"] +
            latency_score * weights["latency"] +
            resource_score * weights["resources"]
        )


@dataclass
class DiagnosisReport:
    """诊断报告"""
    agent_id: str
    generated_at: float = field(default_factory=time.time)
    
    # 分析结果
    primary_issues: List[Dict[str, Any]] = field(default_factory=list)
    failure_patterns: List[Dict[str, Any]] = field(default_factory=list)
    performance_bottlenecks: List[str] = field(default_factory=list)
    
    # 根因分析
    root_causes: List[Dict[str, Any]] = field(default_factory=list)
    
    # 建议
    immediate_actions: List[str] = field(default_factory=list)
    short_term_recommendations: List[str] = field(default_factory=list)
    long_term_improvements: List[str] = field(default_factory=list)
    
    # 训练计划
    suggested_training: List[Dict[str, Any]] = field(default_factory=list)
    
    # 置信度
    confidence: float = 0.0


@dataclass
class TrainingPlan:
    """训练计划"""
    plan_id: str
    agent_id: str
    created_at: float = field(default_factory=time.time)
    
    # 目标
    objectives: List[str] = field(default_factory=list)
    
    # 训练模块
    modules: List[Dict[str, Any]] = field(default_factory=list)
    
    # 进度
    completed_modules: int = 0
    total_modules: int = 0
    
    # 评估
    baseline_score: float = 0.0
    target_score: float = 0.0
    current_score: float = 0.0
    
    # 状态
    status: str = "pending"  # pending, active, completed, cancelled
    
    def get_progress_percentage(self) -> float:
        """获取进度百分比"""
        if self.total_modules == 0:
            return 0.0
        return (self.completed_modules / self.total_modules) * 100


class FailureAnalyzer(ABC):
    """失败分析器基类"""
    
    @abstractmethod
    def analyze(self, failures: List[FailureRecord]) -> List[Dict[str, Any]]:
        """分析失败记录"""
        pass


class PatternBasedAnalyzer(FailureAnalyzer):
    """基于模式的失败分析器"""
    
    def __init__(self):
        self._patterns: Dict[str, Callable[[FailureRecord], bool]] = {
            "timeout_pattern": self._check_timeout_pattern,
            "resource_exhaustion": self._check_resource_pattern,
            "cascading_failure": self._check_cascading_pattern,
            "knowledge_gap": self._check_knowledge_gap
        }
    
    def analyze(self, failures: List[FailureRecord]) -> List[Dict[str, Any]]:
        """分析失败模式"""
        findings = []
        
        for pattern_name, checker in self._patterns.items():
            matches = [f for f in failures if checker(f)]
            if len(matches) >= 3:  # 至少3次匹配才算模式
                findings.append({
                    "pattern": pattern_name,
                    "frequency": len(matches),
                    "severity": self._calculate_pattern_severity(matches),
                    "examples": [m.failure_id for m in matches[:3]]
                })
        
        return findings
    
    def _check_timeout_pattern(self, failure: FailureRecord) -> bool:
        """检查超时模式"""
        return failure.failure_type == FailureType.TIMEOUT
    
    def _check_resource_pattern(self, failure: FailureRecord) -> bool:
        """检查资源耗尽模式"""
        return failure.failure_type == FailureType.RESOURCE_EXHAUSTION
    
    def _check_cascading_pattern(self, failure: FailureRecord) -> bool:
        """检查级联失败模式"""
        return (
            failure.failure_type == FailureType.COMMUNICATION_ERROR and
            failure.context.get("cascading", False)
        )
    
    def _check_knowledge_gap(self, failure: FailureRecord) -> bool:
        """检查知识缺口模式"""
        return failure.failure_type == FailureType.KNOWLEDGE_GAP
    
    def _calculate_pattern_severity(self, matches: List[FailureRecord]) -> str:
        """计算模式严重程度"""
        severities = [m.severity.value for m in matches]
        avg_severity = sum(severities) / len(severities)
        
        if avg_severity >= 3.5:
            return "CRITICAL"
        elif avg_severity >= 2.5:
            return "HIGH"
        elif avg_severity >= 1.5:
            return "MEDIUM"
        return "LOW"


class RootCauseAnalyzer:
    """根因分析器"""
    
    def analyze(self, failures: List[FailureRecord],
                performance: PerformanceSnapshot) -> List[Dict[str, Any]]:
        """分析根因"""
        causes = []
        
        # 分析失败类型分布
        type_counts: Dict[FailureType, int] = {}
        for f in failures:
            type_counts[f.failure_type] = type_counts.get(f.failure_type, 0) + 1
        
        # 找出最常见的失败类型
        if type_counts:
            most_common = max(type_counts.items(), key=lambda x: x[1])
            
            if most_common[1] >= len(failures) * 0.5:
                causes.append({
                    "type": "dominant_failure",
                    "description": f"Dominant failure type: {most_common[0].name}",
                    "frequency": most_common[1],
                    "recommendation": self._get_recommendation_for_type(most_common[0])
                })
        
        # 分析性能相关性
        if performance.success_rate < 0.5:
            causes.append({
                "type": "performance_degradation",
                "description": "Overall performance degradation",
                "success_rate": performance.success_rate,
                "recommendation": "Comprehensive review needed"
            })
        
        # 分析时间模式
        time_pattern = self._analyze_time_pattern(failures)
        if time_pattern:
            causes.append(time_pattern)
        
        return causes
    
    def _get_recommendation_for_type(self, failure_type: FailureType) -> str:
        """获取针对失败类型的建议"""
        recommendations = {
            FailureType.TIMEOUT: "Consider increasing timeout thresholds or optimizing processing logic",
            FailureType.EXCEPTION: "Review error handling and add more robust exception catching",
            FailureType.RESOURCE_EXHAUSTION: "Implement resource limits and scaling mechanisms",
            FailureType.KNOWLEDGE_GAP: "Expand knowledge base or provide additional training data",
            FailureType.LOGIC_ERROR: "Review and test core logic thoroughly",
            FailureType.COMMUNICATION_ERROR: "Check network stability and retry mechanisms"
        }
        return recommendations.get(failure_type, "Review and address the specific issue")
    
    def _analyze_time_pattern(self, failures: List[FailureRecord]) -> Optional[Dict[str, Any]]:
        """分析时间模式"""
        if len(failures) < 5:
            return None
        
        # 检查是否集中在某个时间段
        timestamps = sorted([f.timestamp for f in failures])
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        if not intervals:
            return None
        
        avg_interval = sum(intervals) / len(intervals)
        
        # 如果平均间隔小于5分钟，可能是突发性问题
        if avg_interval < 300:
            return {
                "type": "burst_pattern",
                "description": "Failures occur in bursts",
                "avg_interval_seconds": avg_interval,
                "recommendation": "Check for external dependencies or resource contention"
            }
        
        return None


class RecommendationEngine:
    """建议引擎"""
    
    def generate_recommendations(self, 
                                  diagnosis: DiagnosisReport,
                                  history: List[FailureRecord]) -> Dict[str, List[str]]:
        """生成建议"""
        recommendations = {
            "immediate": [],
            "short_term": [],
            "long_term": []
        }
        
        # 立即行动
        if diagnosis.primary_issues:
            for issue in diagnosis.primary_issues[:3]:
                rec = self._generate_immediate_action(issue)
                if rec:
                    recommendations["immediate"].append(rec)
        
        # 短期建议
        recommendations["short_term"] = self._generate_short_term(diagnosis, history)
        
        # 长期建议
        recommendations["long_term"] = self._generate_long_term(diagnosis)
        
        return recommendations
    
    def _generate_immediate_action(self, issue: Dict[str, Any]) -> str:
        """生成立即行动建议"""
        issue_type = issue.get("type", "")
        
        actions = {
            "timeout": "Increase timeout settings and monitor response times",
            "resource": "Scale up resources or implement rate limiting",
            "error_rate": "Enable detailed logging and investigate errors",
            "performance": "Enable performance profiling"
        }
        
        return actions.get(issue_type, f"Investigate: {issue.get('description', 'Unknown issue')}")
    
    def _generate_short_term(self, diagnosis: DiagnosisReport,
                              history: List[FailureRecord]) -> List[str]:
        """生成短期建议"""
        suggestions = []
        
        # 基于失败模式
        for pattern in diagnosis.failure_patterns:
            pattern_name = pattern.get("pattern", "")
            if "timeout" in pattern_name:
                suggestions.append("Optimize critical path operations to reduce latency")
            elif "resource" in pattern_name:
                suggestions.append("Implement resource pooling and caching")
            elif "knowledge" in pattern_name:
                suggestions.append("Expand training data in weak areas")
        
        # 基于性能瓶颈
        for bottleneck in diagnosis.performance_bottlenecks:
            suggestions.append(f"Address bottleneck: {bottleneck}")
        
        return suggestions[:5]  # 最多5条
    
    def _generate_long_term(self, diagnosis: DiagnosisReport) -> List[str]:
        """生成长期建议"""
        suggestions = [
            "Implement comprehensive monitoring and alerting",
            "Establish regular performance review cycles",
            "Develop automated testing for critical paths",
            "Consider architectural improvements for scalability"
        ]
        
        # 根据具体问题添加
        if any("knowledge" in str(p) for p in diagnosis.failure_patterns):
            suggestions.append("Invest in continuous learning mechanisms")
        
        return suggestions


class TrainingPlanGenerator:
    """训练计划生成器"""
    
    def generate_plan(self, agent_id: str,
                      diagnosis: DiagnosisReport,
                      baseline_score: float) -> TrainingPlan:
        """生成训练计划"""
        modules = []
        
        # 根据诊断结果生成模块
        for issue in diagnosis.primary_issues:
            module = self._create_module_for_issue(issue)
            if module:
                modules.append(module)
        
        # 添加通用改进模块
        modules.extend([
            {
                "name": "Error Handling",
                "description": "Improve exception handling and recovery",
                "duration": "2 hours",
                "type": "skill"
            },
            {
                "name": "Performance Optimization",
                "description": "Learn efficient processing techniques",
                "duration": "3 hours",
                "type": "skill"
            }
        ])
        
        plan_id = f"training_{agent_id}_{int(time.time())}"
        
        return TrainingPlan(
            plan_id=plan_id,
            agent_id=agent_id,
            objectives=[
                f"Reduce failure rate by 50%",
                f"Improve success rate to {min(baseline_score + 0.3, 0.95):.0%}"
            ],
            modules=modules,
            total_modules=len(modules),
            baseline_score=baseline_score,
            target_score=min(baseline_score + 0.3, 0.95),
            current_score=baseline_score
        )
    
    def _create_module_for_issue(self, issue: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """为问题创建训练模块"""
        issue_type = issue.get("type", "")
        
        modules = {
            "timeout": {
                "name": "Timeout Management",
                "description": "Learn to handle time constraints effectively",
                "duration": "1 hour",
                "type": "skill"
            },
            "knowledge_gap": {
                "name": "Knowledge Expansion",
                "description": "Study relevant domain knowledge",
                "duration": "4 hours",
                "type": "knowledge"
            },
            "logic_error": {
                "name": "Logical Reasoning",
                "description": "Practice structured problem solving",
                "duration": "2 hours",
                "type": "skill"
            }
        }
        
        return modules.get(issue_type)


class AgentCoach:
    """Agent教练"""
    
    def __init__(self):
        self.failure_history: Dict[str, List[FailureRecord]] = {}
        self.performance_history: Dict[str, List[PerformanceSnapshot]] = {}
        self.diagnosis_reports: Dict[str, List[DiagnosisReport]] = {}
        self.training_plans: Dict[str, List[TrainingPlan]] = {}
        
        self.failure_analyzer = PatternBasedAnalyzer()
        self.root_cause_analyzer = RootCauseAnalyzer()
        self.recommendation_engine = RecommendationEngine()
        self.training_generator = TrainingPlanGenerator()
        
        self._lock = threading.RLock()
        self._improvement_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
    
    def record_failure(self, failure: FailureRecord) -> None:
        """记录失败"""
        with self._lock:
            if failure.agent_id not in self.failure_history:
                self.failure_history[failure.agent_id] = []
            
            self.failure_history[failure.agent_id].append(failure)
            
            # 限制历史记录大小
            if len(self.failure_history[failure.agent_id]) > 1000:
                self.failure_history[failure.agent_id] = \
                    self.failure_history[failure.agent_id][-500:]
    
    def record_performance(self, snapshot: PerformanceSnapshot) -> None:
        """记录性能快照"""
        with self._lock:
            if snapshot.agent_id not in self.performance_history:
                self.performance_history[snapshot.agent_id] = []
            
            self.performance_history[snapshot.agent_id].append(snapshot)
            
            # 限制历史记录大小
            if len(self.performance_history[snapshot.agent_id]) > 100:
                self.performance_history[snapshot.agent_id] = \
                    self.performance_history[snapshot.agent_id][-50:]
    
    def diagnose(self, agent_id: str) -> Optional[DiagnosisReport]:
        """诊断Agent"""
        with self._lock:
            failures = self.failure_history.get(agent_id, [])
            performances = self.performance_history.get(agent_id, [])
            
            if not failures and not performances:
                return None
            
            # 获取最新性能快照
            current_performance = performances[-1] if performances else PerformanceSnapshot(agent_id=agent_id)
            
            # 分析失败模式
            failure_patterns = self.failure_analyzer.analyze(failures)
            
            # 根因分析
            root_causes = self.root_cause_analyzer.analyze(failures, current_performance)
            
            # 识别主要问题
            primary_issues = self._identify_primary_issues(failures, current_performance)
            
            # 识别性能瓶颈
            bottlenecks = self._identify_bottlenecks(current_performance)
            
            # 生成建议
            recommendations = self.recommendation_engine.generate_recommendations(
                DiagnosisReport(agent_id=agent_id),  # 临时对象
                failures
            )
            
            # 生成训练计划建议
            suggested_training = self._suggest_training_modules(failure_patterns, root_causes)
            
            # 计算置信度
            confidence = self._calculate_confidence(failures, performances)
            
            report = DiagnosisReport(
                agent_id=agent_id,
                primary_issues=primary_issues,
                failure_patterns=failure_patterns,
                performance_bottlenecks=bottlenecks,
                root_causes=root_causes,
                immediate_actions=recommendations["immediate"],
                short_term_recommendations=recommendations["short_term"],
                long_term_improvements=recommendations["long_term"],
                suggested_training=suggested_training,
                confidence=confidence
            )
            
            # 保存报告
            if agent_id not in self.diagnosis_reports:
                self.diagnosis_reports[agent_id] = []
            self.diagnosis_reports[agent_id].append(report)
            
            return report
    
    def _identify_primary_issues(self, failures: List[FailureRecord],
                                  performance: PerformanceSnapshot) -> List[Dict[str, Any]]:
        """识别主要问题"""
        issues = []
        
        # 基于失败率
        if performance.success_rate < 0.7:
            issues.append({
                "type": "error_rate",
                "description": f"High failure rate: {(1-performance.success_rate):.1%}",
                "severity": "HIGH" if performance.success_rate < 0.5 else "MEDIUM"
            })
        
        # 基于延迟
        if performance.avg_latency > 2.0:
            issues.append({
                "type": "timeout",
                "description": f"High latency: {performance.avg_latency:.2f}s",
                "severity": "HIGH" if performance.avg_latency > 5.0 else "MEDIUM"
            })
        
        # 基于资源使用
        if performance.resource_usage:
            avg_resource = sum(performance.resource_usage.values()) / len(performance.resource_usage)
            if avg_resource > 0.8:
                issues.append({
                    "type": "resource",
                    "description": f"High resource usage: {avg_resource:.1%}",
                    "severity": "HIGH"
                })
        
        return issues
    
    def _identify_bottlenecks(self, performance: PerformanceSnapshot) -> List[str]:
        """识别性能瓶颈"""
        bottlenecks = []
        
        if performance.avg_latency > 3.0:
            bottlenecks.append("Response time")
        
        if performance.throughput < 10:
            bottlenecks.append("Low throughput")
        
        for resource, usage in performance.resource_usage.items():
            if usage > 0.9:
                bottlenecks.append(f"{resource} saturation")
        
        return bottlenecks
    
    def _suggest_training_modules(self, patterns: List[Dict[str, Any]],
                                   causes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """建议训练模块"""
        modules = []
        
        for pattern in patterns:
            pattern_name = pattern.get("pattern", "")
            if "timeout" in pattern_name:
                modules.append({
                    "type": "skill",
                    "name": "Efficient Processing",
                    "priority": "high"
                })
            elif "knowledge" in pattern_name:
                modules.append({
                    "type": "knowledge",
                    "name": "Domain Knowledge",
                    "priority": "high"
                })
        
        return modules
    
    def _calculate_confidence(self, failures: List[FailureRecord],
                               performances: List[PerformanceSnapshot]) -> float:
        """计算诊断置信度"""
        factors = []
        
        # 数据量
        factors.append(min(len(failures) / 10.0, 1.0))
        factors.append(min(len(performances) / 5.0, 1.0))
        
        # 数据新鲜度
        if failures:
            last_failure_age = time.time() - failures[-1].timestamp
            factors.append(1.0 if last_failure_age < 86400 else 0.5)
        
        return sum(factors) / len(factors) if factors else 0.5
    
    def create_training_plan(self, agent_id: str) -> Optional[TrainingPlan]:
        """创建训练计划"""
        with self._lock:
            # 先诊断
            diagnosis = self.diagnose(agent_id)
            if not diagnosis:
                return None
            
            # 获取基线分数
            performances = self.performance_history.get(agent_id, [])
            baseline = performances[-1].calculate_health_score() if performances else 0.5
            
            # 生成计划
            plan = self.training_generator.generate_plan(agent_id, diagnosis, baseline)
            
            # 保存计划
            if agent_id not in self.training_plans:
                self.training_plans[agent_id] = []
            self.training_plans[agent_id].append(plan)
            
            return plan
    
    def get_improvement_trends(self, agent_id: str) -> Dict[str, Any]:
        """获取改进趋势"""
        with self._lock:
            performances = self.performance_history.get(agent_id, [])
            
            if len(performances) < 2:
                return {"message": "Not enough data"}
            
            # 计算趋势
            scores = [p.calculate_health_score() for p in performances]
            
            # 简单线性趋势
            n = len(scores)
            x_mean = (n - 1) / 2
            y_mean = sum(scores) / n
            
            numerator = sum((i - x_mean) * (s - y_mean) for i, s in enumerate(scores))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            
            slope = numerator / denominator if denominator != 0 else 0
            
            return {
                "agent_id": agent_id,
                "data_points": n,
                "current_score": scores[-1],
                "average_score": y_mean,
                "trend": "improving" if slope > 0.01 else "declining" if slope < -0.01 else "stable",
                "trend_strength": abs(slope),
                "change_rate": slope
            }
    
    def compare_agents(self, agent_ids: List[str]) -> Dict[str, Any]:
        """比较多个Agent"""
        with self._lock:
            comparison = {
                "agents": {},
                "rankings": [],
                "best_performer": None,
                "needs_attention": []
            }
            
            scores = []
            for aid in agent_ids:
                performances = self.performance_history.get(aid, [])
                if performances:
                    score = performances[-1].calculate_health_score()
                    scores.append((aid, score))
                    
                    comparison["agents"][aid] = {
                        "current_score": score,
                        "failure_count": len(self.failure_history.get(aid, [])),
                        "has_training_plan": aid in self.training_plans
                    }
                    
                    if score < 0.5:
                        comparison["needs_attention"].append(aid)
            
            # 排序
            scores.sort(key=lambda x: x[1], reverse=True)
            comparison["rankings"] = [aid for aid, _ in scores]
            
            if scores:
                comparison["best_performer"] = scores[0][0]
            
            return comparison
    
    def get_coaching_summary(self, agent_id: str) -> Dict[str, Any]:
        """获取教练总结"""
        with self._lock:
            summary = {
                "agent_id": agent_id,
                "total_failures": len(self.failure_history.get(agent_id, [])),
                "performance_snapshots": len(self.performance_history.get(agent_id, [])),
                "diagnosis_count": len(self.diagnosis_reports.get(agent_id, [])),
                "training_plans": len(self.training_plans.get(agent_id, [])),
                "recent_failures": [],
                "active_training": None
            }
            
            # 最近失败
            failures = self.failure_history.get(agent_id, [])
            summary["recent_failures"] = [
                f.to_analysis_dict() for f in failures[-5:]
            ]
            
            # 活跃训练计划
            plans = self.training_plans.get(agent_id, [])
            for plan in plans:
                if plan.status == "active":
                    summary["active_training"] = {
                        "plan_id": plan.plan_id,
                        "progress": plan.get_progress_percentage(),
                        "modules_completed": plan.completed_modules,
                        "total_modules": plan.total_modules
                    }
                    break
            
            return summary
    
    def register_improvement_callback(self,
                                       callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """注册改进回调"""
        with self._lock:
            self._improvement_callbacks.append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            total_failures = sum(len(f) for f in self.failure_history.values())
            total_agents = len(self.failure_history)
            
            # 失败类型分布
            failure_types: Dict[str, int] = {}
            for failures in self.failure_history.values():
                for f in failures:
                    ft = f.failure_type.name
                    failure_types[ft] = failure_types.get(ft, 0) + 1
            
            # 训练计划统计
            active_plans = 0
            completed_plans = 0
            for plans in self.training_plans.values():
                for p in plans:
                    if p.status == "active":
                        active_plans += 1
                    elif p.status == "completed":
                        completed_plans += 1
            
            return {
                "total_monitored_agents": total_agents,
                "total_failures_recorded": total_failures,
                "failure_type_distribution": failure_types,
                "diagnosis_reports_generated": sum(len(d) for d in self.diagnosis_reports.values()),
                "training_plans_created": sum(len(p) for p in self.training_plans.values()),
                "active_training_plans": active_plans,
                "completed_training_plans": completed_plans
            }


# 便捷函数
def create_agent_coach() -> AgentCoach:
    """创建Agent教练实例"""
    return AgentCoach()
