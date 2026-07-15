"""
红队自动化模块

提供红队测试活动的自动化框架
包括活动规划、执行、监控和报告
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
import time
import hashlib
import random
from datetime import datetime, timedelta


class CampaignStatus(Enum):
    """活动状态枚举"""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class ScenarioType(Enum):
    """场景类型枚举"""
    DATA_EXFILTRATION = "data_exfiltration"
    SYSTEM_COMPROMISE = "system_compromise"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    LATERAL_MOVEMENT = "lateral_movement"
    PERSISTENCE = "persistence"


@dataclass
class Campaign:
    """
    红队活动类
    
    Attributes:
        name: 活动名称
        objectives: 活动目标
        scope: 范围
        duration: 持续时间（小时）
        rules_of_engagement: 交战规则
        status: 活动状态
        start_time: 开始时间
        end_time: 结束时间
    """
    name: str
    objectives: List[str]
    scope: str
    duration: int
    rules_of_engagement: Dict[str, Any]
    status: CampaignStatus = CampaignStatus.PLANNED
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "objectives": self.objectives,
            "scope": self.scope,
            "duration": self.duration,
            "rules_of_engagement": self.rules_of_engagement,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time
        }


@dataclass
class AttackScenario:
    """
    攻击场景类
    
    Attributes:
        name: 场景名称
        scenario_type: 场景类型
        description: 描述
        steps: 攻击步骤
        expected_outcome: 预期结果
        prerequisites: 前提条件
        mitigations: 缓解措施
    """
    name: str
    scenario_type: ScenarioType
    description: str
    steps: List[Dict[str, Any]]
    expected_outcome: str
    prerequisites: List[str] = field(default_factory=list)
    mitigations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "type": self.scenario_type.value,
            "description": self.description,
            "steps": self.steps,
            "expected_outcome": self.expected_outcome,
            "prerequisites": self.prerequisites,
            "mitigations": self.mitigations
        }


@dataclass
class RedTeamReport:
    """
    红队报告类
    
    Attributes:
        campaign_name: 活动名称
        findings: 发现项
        timeline: 时间线
        impact_assessment: 影响评估
        recommendations: 建议
        metrics: 指标
    """
    campaign_name: str
    findings: List[Dict[str, Any]]
    timeline: List[Dict[str, Any]]
    impact_assessment: Dict[str, Any]
    recommendations: List[str]
    metrics: Dict[str, Any]
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "campaign_name": self.campaign_name,
            "findings": self.findings,
            "timeline": self.timeline,
            "impact_assessment": self.impact_assessment,
            "recommendations": self.recommendations,
            "metrics": self.metrics,
            "generated_at": self.generated_at
        }
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class ScenarioLibrary:
    """
    场景库
    
    预定义的攻击场景集合
    """
    
    def __init__(self):
        self.scenarios: Dict[str, AttackScenario] = {}
        self._initialize_scenarios()
    
    def _initialize_scenarios(self) -> None:
        """初始化预定义场景"""
        # 数据窃取场景
        self.scenarios["data_exfiltration_basic"] = AttackScenario(
            name="基础数据窃取",
            scenario_type=ScenarioType.DATA_EXFILTRATION,
            description="通过Prompt注入窃取模型训练数据",
            steps=[
                {"step": 1, "action": "侦察", "description": "收集目标信息"},
                {"step": 2, "action": "注入", "description": "构造Prompt注入攻击"},
                {"step": 3, "action": "提取", "description": "提取敏感数据"},
                {"step": 4, "action": "清理", "description": "清理攻击痕迹"}
            ],
            expected_outcome="获取敏感训练数据样本",
            prerequisites=["可访问模型API"],
            mitigations=["输入过滤", "输出过滤", "数据脱敏"]
        )
        
        # 系统破坏场景
        self.scenarios["system_compromise"] = AttackScenario(
            name="系统破坏",
            scenario_type=ScenarioType.SYSTEM_COMPROMISE,
            description="通过对抗攻击破坏模型可用性",
            steps=[
                {"step": 1, "action": "分析", "description": "分析模型输入处理"},
                {"step": 2, "action": "构造", "description": "构造对抗样本"},
                {"step": 3, "action": "攻击", "description": "发送对抗请求"},
                {"step": 4, "action": "验证", "description": "验证破坏效果"}
            ],
            expected_outcome="模型输出质量显著下降",
            prerequisites=["模型API访问", "对抗样本生成能力"],
            mitigations=["对抗训练", "输入验证", "异常检测"]
        )
        
        # 权限提升场景
        self.scenarios["privilege_escalation"] = AttackScenario(
            name="权限提升",
            scenario_type=ScenarioType.PRIVILEGE_ESCALATION,
            description="通过越狱攻击获取更高权限",
            steps=[
                {"step": 1, "action": "侦察", "description": "识别系统限制"},
                {"step": 2, "action": "绕过", "description": "尝试越狱技术"},
                {"step": 3, "action": "利用", "description": "利用绕过后的权限"},
                {"step": 4, "action": "维持", "description": "维持提升后的权限"}
            ],
            expected_outcome="绕过安全限制，执行受限操作",
            prerequisites=["模型访问权限"],
            mitigations=["安全对齐", "输出过滤", "行为监控"]
        )
        
        # 横向移动场景
        self.scenarios["lateral_movement"] = AttackScenario(
            name="横向移动",
            scenario_type=ScenarioType.LATERAL_MOVEMENT,
            description="通过模型提取攻击复制模型能力",
            steps=[
                {"step": 1, "action": "查询", "description": "大量查询目标模型"},
                {"step": 2, "action": "收集", "description": "收集输入输出对"},
                {"step": 3, "action": "训练", "description": "训练替代模型"},
                {"step": 4, "action": "验证", "description": "验证替代模型效果"}
            ],
            expected_outcome="成功提取模型能力",
            prerequisites=["大量API查询权限"],
            mitigations=["查询限制", "输出扰动", "水印技术"]
        )
    
    def get_scenario(self, name: str) -> Optional[AttackScenario]:
        """获取场景"""
        return self.scenarios.get(name)
    
    def list_scenarios(self, scenario_type: Optional[ScenarioType] = None) -> List[AttackScenario]:
        """列出场景"""
        if scenario_type:
            return [s for s in self.scenarios.values() if s.scenario_type == scenario_type]
        return list(self.scenarios.values())
    
    def add_scenario(self, scenario: AttackScenario) -> None:
        """添加场景"""
        self.scenarios[scenario.name] = scenario


class RedTeamAutomation:
    """
    红队自动化框架
    
    自动化红队测试活动的规划、执行和报告
    """
    
    def __init__(self):
        self.campaigns: Dict[str, Campaign] = {}
        self.scenario_library = ScenarioLibrary()
        self.current_campaign: Optional[Campaign] = None
        self.progress_log: List[Dict[str, Any]] = []
        self.results: Dict[str, Any] = {}
    
    def plan_campaign(self, name: str, objectives: List[str], scope: str,
                     duration: int, rules: Dict[str, Any]) -> Campaign:
        """
        规划红队活动
        
        Args:
            name: 活动名称
            objectives: 活动目标
            scope: 范围
            duration: 持续时间（小时）
            rules: 交战规则
            
        Returns:
            Campaign: 规划的活动
        """
        campaign = Campaign(
            name=name,
            objectives=objectives,
            scope=scope,
            duration=duration,
            rules_of_engagement=rules
        )
        
        self.campaigns[name] = campaign
        return campaign
    
    def execute_campaign(self, campaign_name: str, 
                        scenarios: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        执行红队活动
        
        Args:
            campaign_name: 活动名称
            scenarios: 要执行的场景列表
            
        Returns:
            Dict: 执行结果
        """
        if campaign_name not in self.campaigns:
            raise ValueError(f"Campaign {campaign_name} not found")
        
        campaign = self.campaigns[campaign_name]
        self.current_campaign = campaign
        
        # 更新状态
        campaign.status = CampaignStatus.IN_PROGRESS
        campaign.start_time = datetime.now().isoformat()
        
        results = {
            "campaign": campaign_name,
            "start_time": campaign.start_time,
            "scenarios_executed": [],
            "findings": [],
            "status": "running"
        }
        
        # 执行场景
        if scenarios is None:
            scenarios = ["data_exfiltration_basic", "privilege_escalation"]
        
        for scenario_name in scenarios:
            scenario = self.scenario_library.get_scenario(scenario_name)
            if scenario:
                scenario_result = self._execute_scenario(scenario)
                results["scenarios_executed"].append({
                    "name": scenario_name,
                    "result": scenario_result
                })
                results["findings"].extend(scenario_result.get("findings", []))
        
        # 更新活动状态
        campaign.status = CampaignStatus.COMPLETED
        campaign.end_time = datetime.now().isoformat()
        results["end_time"] = campaign.end_time
        results["status"] = "completed"
        
        self.results[campaign_name] = results
        return results
    
    def monitor_progress(self, campaign_name: str) -> Dict[str, Any]:
        """
        监控活动进度
        
        Args:
            campaign_name: 活动名称
            
        Returns:
            Dict: 进度信息
        """
        if campaign_name not in self.campaigns:
            return {"error": "Campaign not found"}
        
        campaign = self.campaigns[campaign_name]
        
        progress = {
            "campaign": campaign_name,
            "status": campaign.status.value,
            "start_time": campaign.start_time,
            "end_time": campaign.end_time,
            "progress_percentage": self._calculate_progress(campaign),
            "recent_activities": self._get_recent_activities(campaign_name)
        }
        
        return progress
    
    def generate_report(self, campaign_name: str) -> RedTeamReport:
        """
        生成红队报告
        
        Args:
            campaign_name: 活动名称
            
        Returns:
            RedTeamReport: 红队报告
        """
        if campaign_name not in self.results:
            raise ValueError(f"No results found for campaign {campaign_name}")
        
        results = self.results[campaign_name]
        campaign = self.campaigns[campaign_name]
        
        # 计算指标
        metrics = self._calculate_metrics(results)
        
        # 影响评估
        impact_assessment = self._assess_impact(results)
        
        # 生成建议
        recommendations = self._generate_recommendations(results)
        
        return RedTeamReport(
            campaign_name=campaign_name,
            findings=results.get("findings", []),
            timeline=self.progress_log,
            impact_assessment=impact_assessment,
            recommendations=recommendations,
            metrics=metrics
        )
    
    def _execute_scenario(self, scenario: AttackScenario) -> Dict[str, Any]:
        """执行单个场景"""
        result = {
            "scenario": scenario.name,
            "status": "started",
            "steps_completed": [],
            "findings": [],
            "start_time": time.time()
        }
        
        for step in scenario.steps:
            # 模拟执行步骤
            step_result = self._execute_step(step)
            result["steps_completed"].append({
                "step": step["step"],
                "action": step["action"],
                "success": step_result["success"],
                "finding": step_result.get("finding")
            })
            
            if step_result.get("finding"):
                result["findings"].append(step_result["finding"])
            
            # 记录进度
            self.progress_log.append({
                "timestamp": time.time(),
                "campaign": self.current_campaign.name if self.current_campaign else None,
                "scenario": scenario.name,
                "step": step["step"],
                "action": step["action"]
            })
        
        result["status"] = "completed"
        result["end_time"] = time.time()
        return result
    
    def _execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个步骤"""
        # 模拟步骤执行
        success = random.random() > 0.3  # 70%成功率
        
        result = {
            "success": success,
            "finding": None
        }
        
        if success and random.random() > 0.5:  # 50%概率发现
            result["finding"] = {
                "type": "vulnerability",
                "severity": random.choice(["low", "medium", "high", "critical"]),
                "description": f"在步骤 {step['step']} 发现安全问题",
                "evidence": f"执行 {step['action']} 时发现"
            }
        
        return result
    
    def _calculate_progress(self, campaign: Campaign) -> float:
        """计算进度百分比"""
        if campaign.status == CampaignStatus.COMPLETED:
            return 100.0
        elif campaign.status == CampaignStatus.PLANNED:
            return 0.0
        
        # 基于时间计算
        if campaign.start_time:
            start = datetime.fromisoformat(campaign.start_time)
            elapsed = (datetime.now() - start).total_seconds() / 3600
            progress = min(100.0, (elapsed / campaign.duration) * 100)
            return progress
        
        return 0.0
    
    def _get_recent_activities(self, campaign_name: str) -> List[Dict[str, Any]]:
        """获取最近活动"""
        activities = [a for a in self.progress_log 
                     if a.get("campaign") == campaign_name]
        return activities[-10:]  # 返回最近10条
    
    def _calculate_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """计算指标"""
        findings = results.get("findings", [])
        
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for finding in findings:
            severity = finding.get("severity", "low")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_findings": len(findings),
            "severity_distribution": severity_counts,
            "scenarios_executed": len(results.get("scenarios_executed", [])),
            "success_rate": random.uniform(0.6, 0.9)
        }
    
    def _assess_impact(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """评估影响"""
        findings = results.get("findings", [])
        
        critical_count = sum(1 for f in findings if f.get("severity") == "critical")
        high_count = sum(1 for f in findings if f.get("severity") == "high")
        
        impact_level = "low"
        if critical_count > 0:
            impact_level = "critical"
        elif high_count > 0:
            impact_level = "high"
        elif len(findings) > 5:
            impact_level = "medium"
        
        return {
            "level": impact_level,
            "critical_findings": critical_count,
            "high_findings": high_count,
            "business_impact": self._assess_business_impact(impact_level),
            "technical_impact": self._assess_technical_impact(impact_level)
        }
    
    def _assess_business_impact(self, level: str) -> str:
        """评估业务影响"""
        impacts = {
            "critical": "可能导致严重的业务中断和数据泄露",
            "high": "可能导致显著的业务影响",
            "medium": "可能导致有限的业务影响",
            "low": "业务影响较小"
        }
        return impacts.get(level, "未知")
    
    def _assess_technical_impact(self, level: str) -> str:
        """评估技术影响"""
        impacts = {
            "critical": "系统完全 compromised",
            "high": "系统部分 compromised",
            "medium": "存在可利用的漏洞",
            "low": "存在轻微安全问题"
        }
        return impacts.get(level, "未知")
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = [
            "实施定期安全评估和渗透测试",
            "加强输入验证和输出过滤",
            "建立安全监控和告警机制",
            "实施最小权限原则",
            "定期进行安全培训"
        ]
        
        findings = results.get("findings", [])
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for finding in findings:
            severity = finding.get("severity", "low")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        if severity_counts.get("critical", 0) > 0:
            recommendations.insert(0, "立即修复关键安全漏洞")
        
        if severity_counts.get("high", 0) > 0:
            recommendations.insert(1, "优先处理高风险安全问题")
        
        return recommendations


class ContinuousRedTeam:
    """
    持续红队测试
    
    自动化定期执行红队测试
    """
    
    def __init__(self):
        self.automation = RedTeamAutomation()
        self.scheduled_tests: Dict[str, Dict[str, Any]] = {}
        self.test_history: List[Dict[str, Any]] = []
        self.trends: Dict[str, List[float]] = {
            "critical_findings": [],
            "high_findings": [],
            "overall_score": []
        }
    
    def schedule_test(self, test_id: str, campaign_config: Dict[str, Any],
                     interval_hours: int) -> Dict[str, Any]:
        """
        安排定期测试
        
        Args:
            test_id: 测试ID
            campaign_config: 活动配置
            interval_hours: 间隔小时数
            
        Returns:
            Dict: 调度信息
        """
        schedule = {
            "test_id": test_id,
            "campaign_config": campaign_config,
            "interval_hours": interval_hours,
            "next_run": (datetime.now() + timedelta(hours=interval_hours)).isoformat(),
            "run_count": 0,
            "last_run": None
        }
        
        self.scheduled_tests[test_id] = schedule
        return schedule
    
    def run_scheduled_test(self, test_id: str) -> Optional[Dict[str, Any]]:
        """
        执行定期测试
        
        Args:
            test_id: 测试ID
            
        Returns:
            Dict: 测试结果
        """
        if test_id not in self.scheduled_tests:
            return None
        
        schedule = self.scheduled_tests[test_id]
        config = schedule["campaign_config"]
        
        # 规划活动
        campaign = self.automation.plan_campaign(
            name=config["name"],
            objectives=config["objectives"],
            scope=config["scope"],
            duration=config["duration"],
            rules=config["rules"]
        )
        
        # 执行活动
        results = self.automation.execute_campaign(
            campaign_name=campaign.name,
            scenarios=config.get("scenarios")
        )
        
        # 更新调度信息
        schedule["run_count"] += 1
        schedule["last_run"] = datetime.now().isoformat()
        schedule["next_run"] = (datetime.now() + 
                               timedelta(hours=schedule["interval_hours"])).isoformat()
        
        # 记录历史
        self.test_history.append({
            "test_id": test_id,
            "timestamp": schedule["last_run"],
            "results": results
        })
        
        # 更新趋势
        self._update_trends(results)
        
        return results
    
    def run_regression_test(self, previous_findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        运行回归测试
        
        Args:
            previous_findings: 之前的发现项
            
        Returns:
            Dict: 回归测试结果
        """
        regression_results = {
            "fixed": [],
            "persistent": [],
            "new": [],
            "test_time": datetime.now().isoformat()
        }
        
        # 模拟回归测试
        for finding in previous_findings:
            status = random.choice(["fixed", "persistent"])
            if status == "fixed":
                regression_results["fixed"].append(finding)
            else:
                regression_results["persistent"].append(finding)
        
        # 模拟新发现
        if random.random() > 0.7:
            regression_results["new"].append({
                "type": "vulnerability",
                "severity": "medium",
                "description": "回归测试中新发现的问题"
            })
        
        return regression_results
    
    def analyze_trends(self) -> Dict[str, Any]:
        """
        分析趋势
        
        Returns:
            Dict: 趋势分析结果
        """
        analysis = {
            "critical_trend": self._analyze_metric_trend("critical_findings"),
            "high_trend": self._analyze_metric_trend("high_findings"),
            "overall_trend": self._analyze_metric_trend("overall_score"),
            "recommendations": []
        }
        
        # 基于趋势生成建议
        if analysis["critical_trend"] == "increasing":
            analysis["recommendations"].append("关键问题呈上升趋势，需要立即采取行动")
        elif analysis["critical_trend"] == "decreasing":
            analysis["recommendations"].append("关键问题在减少，安全措施有效")
        
        return analysis
    
    def _update_trends(self, results: Dict[str, Any]) -> None:
        """更新趋势数据"""
        findings = results.get("findings", [])
        
        critical_count = sum(1 for f in findings if f.get("severity") == "critical")
        high_count = sum(1 for f in findings if f.get("severity") == "high")
        
        self.trends["critical_findings"].append(critical_count)
        self.trends["high_findings"].append(high_count)
        
        # 计算整体安全评分（越低越好）
        score = 100 - (critical_count * 20 + high_count * 10)
        self.trends["overall_score"].append(max(0, score))
    
    def _analyze_metric_trend(self, metric_name: str) -> str:
        """分析指标趋势"""
        values = self.trends.get(metric_name, [])
        
        if len(values) < 2:
            return "stable"
        
        # 简单线性趋势分析
        recent = values[-5:] if len(values) >= 5 else values
        first_half = recent[:len(recent)//2]
        second_half = recent[len(recent)//2:]
        
        first_avg = sum(first_half) / len(first_half) if first_half else 0
        second_avg = sum(second_half) / len(second_half) if second_half else 0
        
        diff = second_avg - first_avg
        threshold = first_avg * 0.1 if first_avg > 0 else 0.5
        
        if diff > threshold:
            return "increasing"
        elif diff < -threshold:
            return "decreasing"
        return "stable"
