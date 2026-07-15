"""
DLP Monitor Module - DLP监控器

提供DLP实时监控和告警能力：
- 实时数据流监控
- 策略违规检查
- 泄露告警
- DLP报告生成
"""

import json
import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Set, Union
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict


class AlertSeverity(Enum):
    """告警严重级别"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """告警类型"""
    POLICY_VIOLATION = "policy_violation"
    SENSITIVE_DATA_DETECTED = "sensitive_data_detected"
    DATA_EXFILTRATION = "data_exfiltration"
    ANOMALOUS_ACCESS = "anomalous_access"
    UNAUTHORIZED_TRANSFER = "unauthorized_transfer"


class ActionType(Enum):
    """策略动作类型"""
    ALLOW = "allow"
    BLOCK = "block"
    QUARANTINE = "quarantine"
    ENCRYPT = "encrypt"
    ALERT = "alert"
    LOG = "log"


@dataclass
class DLPAlert:
    """DLP告警"""
    alert_id: str
    severity: AlertSeverity
    alert_type: AlertType
    description: str
    affected_data: Dict[str, Any]
    timestamp: str
    source: str = ""
    destination: str = ""
    user: str = ""
    action_taken: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.alert_id:
            self.alert_id = f"ALERT_{int(time.time() * 1000)}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "type": self.alert_type.value,
            "description": self.description,
            "affected_data": self.affected_data,
            "timestamp": self.timestamp,
            "source": self.source,
            "destination": self.destination,
            "user": self.user,
            "action_taken": self.action_taken,
            "metadata": self.metadata
        }


@dataclass
class DLPRule:
    """DLP规则"""
    name: str
    description: str
    conditions: Dict[str, Any]
    actions: List[ActionType]
    severity: AlertSeverity
    enabled: bool = True
    exceptions: List[str] = field(default_factory=list)
    
    def matches(self, data: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """检查数据是否匹配规则"""
        if not self.enabled:
            return False
        
        # 检查例外
        for exception in self.exceptions:
            if exception in str(data):
                return False
        
        # 检查条件
        for key, value in self.conditions.items():
            if key == "pii_types":
                data_pii = data.get("pii_types", [])
                if not any(pii in data_pii for pii in value):
                    return False
            elif key == "classification":
                if data.get("classification") != value:
                    return False
            elif key == "destination":
                if data.get("destination") in value:  # 禁止的目的地
                    return True
                return False
            elif key == "size_threshold":
                if data.get("size", 0) < value:
                    return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "conditions": self.conditions,
            "actions": [a.value for a in self.actions],
            "severity": self.severity.value,
            "enabled": self.enabled
        }


@dataclass
class DLPPolicy:
    """DLP策略"""
    name: str
    description: str
    rules: List[DLPRule]
    severity_threshold: AlertSeverity
    exceptions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def evaluate(self, data: Dict[str, Any], context: Dict[str, Any]) -> List[DLPRule]:
        """评估数据是否符合策略"""
        matched_rules = []
        
        for rule in self.rules:
            if rule.matches(data, context):
                matched_rules.append(rule)
        
        return matched_rules
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "rules": [r.to_dict() for r in self.rules],
            "severity_threshold": self.severity_threshold.value,
            "exceptions": self.exceptions
        }


class PolicyEngine:
    """策略引擎"""
    
    def __init__(self):
        self.policies: List[DLPPolicy] = []
        self.rule_cache: Dict[str, DLPRule] = {}
    
    def add_policy(self, policy: DLPPolicy):
        """添加策略"""
        self.policies.append(policy)
        for rule in policy.rules:
            self.rule_cache[rule.name] = rule
    
    def remove_policy(self, name: str) -> bool:
        """移除策略"""
        for i, policy in enumerate(self.policies):
            if policy.name == name:
                for rule in policy.rules:
                    if rule.name in self.rule_cache:
                        del self.rule_cache[rule.name]
                del self.policies[i]
                return True
        return False
    
    def evaluate(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估数据
        
        Returns:
            评估结果，包含匹配的规则和建议动作
        """
        all_matched_rules = []
        max_severity = AlertSeverity.INFO
        all_actions: Set[ActionType] = set()
        
        for policy in self.policies:
            matched = policy.evaluate(data, context)
            all_matched_rules.extend(matched)
            
            for rule in matched:
                # 更新最高严重级别
                severity_order = [AlertSeverity.INFO, AlertSeverity.LOW, 
                                 AlertSeverity.MEDIUM, AlertSeverity.HIGH, AlertSeverity.CRITICAL]
                if severity_order.index(rule.severity) > severity_order.index(max_severity):
                    max_severity = rule.severity
                
                # 收集所有动作
                all_actions.update(rule.actions)
        
        # 确定最终动作（优先级：BLOCK > QUARANTINE > ENCRYPT > ALERT > LOG > ALLOW）
        final_action = self._resolve_actions(all_actions)
        
        return {
            "matched_rules": all_matched_rules,
            "severity": max_severity,
            "actions": list(all_actions),
            "final_action": final_action,
            "violation": len(all_matched_rules) > 0
        }
    
    def _resolve_actions(self, actions: Set[ActionType]) -> ActionType:
        """解析最终动作"""
        priority = [
            ActionType.BLOCK,
            ActionType.QUARANTINE,
            ActionType.ENCRYPT,
            ActionType.ALERT,
            ActionType.LOG,
            ActionType.ALLOW
        ]
        
        for action in priority:
            if action in actions:
                return action
        
        return ActionType.ALLOW


@dataclass
class DLPReport:
    """DLP报告"""
    report_id: str
    start_time: str
    end_time: str
    summary: Dict[str, Any]
    violations: List[Dict[str, Any]]
    trends: Dict[str, Any]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "summary": self.summary,
            "violations": self.violations,
            "trends": self.trends,
            "recommendations": self.recommendations
        }
    
    def export(self, file_path: Union[str, Path]):
        """导出报告"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


class DLPMonitor:
    """DLP监控器"""
    
    def __init__(self):
        self.policy_engine = PolicyEngine()
        self.alerts: List[DLPAlert] = []
        self.alert_handlers: List[Callable[[DLPAlert], None]] = []
        self.monitoring = False
        self.violation_log: List[Dict[str, Any]] = []
        self._init_default_policies()
    
    def _init_default_policies(self):
        """初始化默认策略"""
        # 高敏感度数据外发策略
        high_sensitivity_policy = DLPPolicy(
            name="高敏感度数据保护",
            description="阻止高敏感度数据外发到未授权目的地",
            rules=[
                DLPRule(
                    name="阻止机密数据外发",
                    description="阻止机密级别数据发送到外部邮箱",
                    conditions={
                        "classification": "confidential",
                        "destination": ["external_email", "public_cloud"]
                    },
                    actions=[ActionType.BLOCK, ActionType.ALERT],
                    severity=AlertSeverity.HIGH
                ),
                DLPRule(
                    name="PII数据加密",
                    description="包含PII的数据必须加密",
                    conditions={
                        "pii_types": ["china_id_card", "credit_card", "us_ssn"]
                    },
                    actions=[ActionType.ENCRYPT, ActionType.LOG],
                    severity=AlertSeverity.MEDIUM
                )
            ],
            severity_threshold=AlertSeverity.MEDIUM
        )
        
        # 大文件传输策略
        large_file_policy = DLPPolicy(
            name="大文件传输监控",
            description="监控大文件传输",
            rules=[
                DLPRule(
                    name="大文件告警",
                    description="超过100MB的文件传输需要审批",
                    conditions={
                        "size_threshold": 100 * 1024 * 1024  # 100MB
                    },
                    actions=[ActionType.ALERT, ActionType.QUARANTINE],
                    severity=AlertSeverity.MEDIUM
                )
            ],
            severity_threshold=AlertSeverity.LOW
        )
        
        self.policy_engine.add_policy(high_sensitivity_policy)
        self.policy_engine.add_policy(large_file_policy)
    
    def add_policy(self, policy: DLPPolicy):
        """添加策略"""
        self.policy_engine.add_policy(policy)
    
    def remove_policy(self, name: str) -> bool:
        """移除策略"""
        return self.policy_engine.remove_policy(name)
    
    def add_alert_handler(self, handler: Callable[[DLPAlert], None]):
        """添加告警处理器"""
        self.alert_handlers.append(handler)
    
    def monitor_stream(self, stream_data: Dict[str, Any], 
                       context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        实时监控数据流
        
        Args:
            stream_data: 数据流信息
            context: 上下文信息
            
        Returns:
            监控结果
        """
        context = context or {}
        
        # 评估策略
        evaluation = self.policy_engine.evaluate(stream_data, context)
        
        result = {
            "allowed": evaluation["final_action"] != ActionType.BLOCK,
            "action": evaluation["final_action"].value,
            "violations": evaluation["matched_rules"],
            "severity": evaluation["severity"].value if evaluation["violation"] else None
        }
        
        # 如果存在违规，生成告警
        if evaluation["violation"]:
            alert = self._create_alert(stream_data, evaluation, context)
            self._trigger_alert(alert)
            result["alert_id"] = alert.alert_id
        
        # 记录违规
        if evaluation["matched_rules"]:
            self.violation_log.append({
                "timestamp": datetime.now().isoformat(),
                "data": stream_data,
                "evaluation": evaluation,
                "context": context
            })
        
        return result
    
    def check_policy_violation(self, data: Dict[str, Any], 
                                policy_name: Optional[str] = None) -> Dict[str, Any]:
        """
        检查策略违规
        
        Args:
            data: 要检查的数据
            policy_name: 指定策略名称，None表示检查所有
            
        Returns:
            检查结果
        """
        context = {"check_time": datetime.now().isoformat()}
        
        if policy_name:
            # 只检查指定策略
            policy = None
            for p in self.policy_engine.policies:
                if p.name == policy_name:
                    policy = p
                    break
            
            if not policy:
                return {"error": f"Policy {policy_name} not found"}
            
            matched_rules = policy.evaluate(data, context)
            return {
                "policy": policy_name,
                "violation": len(matched_rules) > 0,
                "matched_rules": [r.name for r in matched_rules]
            }
        else:
            # 检查所有策略
            return self.policy_engine.evaluate(data, context)
    
    def _create_alert(self, data: Dict[str, Any], 
                     evaluation: Dict[str, Any],
                     context: Dict[str, Any]) -> DLPAlert:
        """创建告警"""
        severity = evaluation["severity"]
        matched_rules = evaluation["matched_rules"]
        
        # 确定告警类型
        if any("外发" in r.name or "transfer" in r.name.lower() for r in matched_rules):
            alert_type = AlertType.UNAUTHORIZED_TRANSFER
        elif any("PII" in r.name for r in matched_rules):
            alert_type = AlertType.SENSITIVE_DATA_DETECTED
        else:
            alert_type = AlertType.POLICY_VIOLATION
        
        # 构建描述
        rule_names = [r.name for r in matched_rules]
        description = f"检测到策略违规: {', '.join(rule_names)}"
        
        alert = DLPAlert(
            alert_id=f"ALERT_{int(time.time() * 1000)}",
            severity=severity,
            alert_type=alert_type,
            description=description,
            affected_data={
                "data_type": data.get("type"),
                "size": data.get("size"),
                "classification": data.get("classification"),
                "pii_detected": data.get("pii_types", [])
            },
            timestamp=datetime.now().isoformat(),
            source=context.get("source", ""),
            destination=context.get("destination", ""),
            user=context.get("user", ""),
            action_taken=evaluation["final_action"].value,
            metadata={
                "matched_rules": rule_names,
                "evaluation": evaluation
            }
        )
        
        self.alerts.append(alert)
        return alert
    
    def _trigger_alert(self, alert: DLPAlert):
        """触发告警"""
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                print(f"Alert handler error: {e}")
    
    def alert_on_leak(self, leak_info: Dict[str, Any]) -> DLPAlert:
        """
        泄露告警
        
        Args:
            leak_info: 泄露信息
            
        Returns:
            告警对象
        """
        alert = DLPAlert(
            alert_id=f"LEAK_{int(time.time() * 1000)}",
            severity=AlertSeverity.CRITICAL,
            alert_type=AlertType.DATA_EXFILTRATION,
            description=f"检测到数据泄露: {leak_info.get('description', '')}",
            affected_data=leak_info.get("affected_data", {}),
            timestamp=datetime.now().isoformat(),
            source=leak_info.get("source", ""),
            destination=leak_info.get("destination", ""),
            user=leak_info.get("user", ""),
            action_taken="ALERT",
            metadata=leak_info.get("metadata", {})
        )
        
        self.alerts.append(alert)
        self._trigger_alert(alert)
        
        return alert
    
    def generate_report(self, start_time: Optional[str] = None,
                       end_time: Optional[str] = None) -> DLPReport:
        """
        生成DLP报告
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            DLP报告
        """
        now = datetime.now()
        
        if not end_time:
            end_time = now.isoformat()
        if not start_time:
            start_time = (now - timedelta(days=7)).isoformat()
        
        # 筛选时间范围内的告警
        filtered_alerts = []
        for alert in self.alerts:
            alert_time = datetime.fromisoformat(alert.timestamp)
            start = datetime.fromisoformat(start_time)
            end = datetime.fromisoformat(end_time)
            if start <= alert_time <= end:
                filtered_alerts.append(alert)
        
        # 生成摘要
        summary = {
            "total_alerts": len(filtered_alerts),
            "by_severity": defaultdict(int),
            "by_type": defaultdict(int),
            "blocked_transfers": 0,
            "quarantined_items": 0
        }
        
        for alert in filtered_alerts:
            summary["by_severity"][alert.severity.value] += 1
            summary["by_type"][alert.alert_type.value] += 1
            
            if alert.action_taken == ActionType.BLOCK.value:
                summary["blocked_transfers"] += 1
            elif alert.action_taken == ActionType.QUARANTINE.value:
                summary["quarantined_items"] += 1
        
        # 转换defaultdict为普通dict
        summary["by_severity"] = dict(summary["by_severity"])
        summary["by_type"] = dict(summary["by_type"])
        
        # 生成趋势（按天统计）
        trends = self._calculate_trends(filtered_alerts)
        
        # 生成建议
        recommendations = self._generate_recommendations(summary, filtered_alerts)
        
        report = DLPReport(
            report_id=f"RPT_{int(time.time())}",
            start_time=start_time,
            end_time=end_time,
            summary=summary,
            violations=[a.to_dict() for a in filtered_alerts],
            trends=trends,
            recommendations=recommendations
        )
        
        return report
    
    def _calculate_trends(self, alerts: List[DLPAlert]) -> Dict[str, Any]:
        """计算趋势"""
        daily_counts = defaultdict(lambda: defaultdict(int))
        
        for alert in alerts:
            date = alert.timestamp[:10]  # YYYY-MM-DD
            daily_counts[date]["total"] += 1
            daily_counts[date][alert.severity.value] += 1
            daily_counts[date][alert.alert_type.value] += 1
        
        return {
            "daily": dict(daily_counts),
            "total_alerts": len(alerts)
        }
    
    def _generate_recommendations(self, summary: Dict[str, Any], 
                                   alerts: List[DLPAlert]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于严重级别生成建议
        critical_count = summary["by_severity"].get("critical", 0)
        high_count = summary["by_severity"].get("high", 0)
        
        if critical_count > 0:
            recommendations.append(
                f"检测到{critical_count}个严重告警，建议立即审查安全策略"
            )
        
        if high_count > 5:
            recommendations.append(
                f"高优先级告警较多({high_count})，建议加强员工安全意识培训"
            )
        
        # 基于告警类型生成建议
        exfiltration_count = summary["by_type"].get("data_exfiltration", 0)
        if exfiltration_count > 0:
            recommendations.append(
                f"检测到{exfiltration_count}起数据外泄事件，建议审查数据传输通道"
            )
        
        # 基于阻断统计生成建议
        blocked = summary.get("blocked_transfers", 0)
        if blocked > 10:
            recommendations.append(
                f"本周阻断了{blocked}次传输，建议审查业务需求与策略的匹配度"
            )
        
        if not recommendations:
            recommendations.append("当前安全状况良好，继续保持现有策略")
        
        return recommendations
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[DLPAlert]:
        """获取活动告警"""
        if severity:
            return [a for a in self.alerts if a.severity == severity]
        return self.alerts
    
    def clear_alerts(self):
        """清除所有告警"""
        self.alerts.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取监控统计"""
        return {
            "total_policies": len(self.policy_engine.policies),
            "total_rules": len(self.policy_engine.rule_cache),
            "total_alerts": len(self.alerts),
            "total_violations": len(self.violation_log),
            "alert_severity_distribution": {
                severity.value: sum(1 for a in self.alerts if a.severity == severity)
                for severity in AlertSeverity
            }
        }


# 便捷函数
def create_monitor() -> DLPMonitor:
    """创建DLP监控器"""
    return DLPMonitor()


# 示例用法
if __name__ == "__main__":
    monitor = DLPMonitor()
    
    print("DLP监控测试：")
    print("=" * 60)
    
    # 添加告警处理器
    def print_alert(alert: DLPAlert):
        print(f"\n[告警] {alert.severity.value.upper()}: {alert.description}")
        print(f"  动作: {alert.action_taken}")
    
    monitor.add_alert_handler(print_alert)
    
    # 测试数据流监控
    test_cases = [
        {
            "type": "email",
            "classification": "confidential",
            "destination": "external_email",
            "size": 1024,
            "pii_types": ["email"]
        },
        {
            "type": "file_transfer",
            "classification": "internal",
            "destination": "internal_server",
            "size": 200 * 1024 * 1024,  # 200MB
            "pii_types": []
        },
        {
            "type": "api_call",
            "classification": "public",
            "destination": "public_cloud",
            "size": 512,
            "pii_types": ["china_id_card"]
        }
    ]
    
    print("\n1. 测试数据流监控：")
    for i, data in enumerate(test_cases):
        context = {"user": f"user_{i}", "source": "internal_system"}
        result = monitor.monitor_stream(data, context)
        print(f"\n  测试 {i+1}:")
        print(f"    数据: {data}")
        print(f"    结果: 允许={result['allowed']}, 动作={result['action']}")
        if result.get('alert_id'):
            print(f"    告警ID: {result['alert_id']}")
    
    # 生成报告
    print("\n2. 生成DLP报告：")
    report = monitor.generate_report()
    print(f"   报告ID: {report.report_id}")
    print(f"   告警总数: {report.summary['total_alerts']}")
    print(f"   按严重级别: {report.summary['by_severity']}")
    print(f"   建议: {report.recommendations}")
    
    # 统计信息
    print("\n3. 统计信息：")
    stats = monitor.get_statistics()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
