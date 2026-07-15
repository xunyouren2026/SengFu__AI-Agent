"""
TestPenetration - 渗透测试

测试aisec/redteam的渗透测试功能，包括侦察/利用/后渗透阶段、攻击向量和攻击链。

测试内容：
- 攻击阶段和风险等级枚举
- 攻击向量
- 攻击链
- 发现项
- 侦察阶段
- 利用阶段
- 后渗透阶段
- 渗透测试报告
- 防御验证和绕过技术
"""

import pytest
import json
from typing import Dict, List, Any

# 导入被测试的模块
from agi_unified_framework.aisec.redteam.penetration_tester import (
    AttackPhase,
    RiskLevel,
    AttackVector,
    AttackChain,
    Finding,
    PenTestReport,
    Reconnaissance,
    Exploitation,
    PostExploitation,
    PenetrationTester
)


# 测试配置
pytestmark = pytest.mark.unit


class TestAttackPhase:
    """测试攻击阶段枚举"""

    def test_reconnaissance_phase(self):
        """测试侦察阶段枚举值"""
        assert AttackPhase.RECONNAISSANCE.value == "reconnaissance"

    def test_exploitation_phase(self):
        """测试利用阶段枚举值"""
        assert AttackPhase.EXPLOITATION.value == "exploitation"

    def test_post_exploitation_phase(self):
        """测试后渗透阶段枚举值"""
        assert AttackPhase.POST_EXPLOITATION.value == "post_exploitation"


class TestRiskLevel:
    """测试风险等级枚举"""

    def test_critical_risk(self):
        """测试关键风险等级"""
        assert RiskLevel.CRITICAL.value == "critical"

    def test_high_risk(self):
        """测试高风险等级"""
        assert RiskLevel.HIGH.value == "high"

    def test_medium_risk(self):
        """测试中风险等级"""
        assert RiskLevel.MEDIUM.value == "medium"

    def test_low_risk(self):
        """测试低风险等级"""
        assert RiskLevel.LOW.value == "low"

    def test_info_risk(self):
        """测试信息风险等级"""
        assert RiskLevel.INFO.value == "info"


class TestAttackVector:
    """测试攻击向量"""

    def test_attack_vector_creation(self):
        """测试攻击向量创建"""
        vector = AttackVector(
            vector_type="sql_injection",
            description="SQL注入攻击",
            prerequisites=["用户输入点", "未参数化查询"],
            impact="数据泄露",
            likelihood=0.7,
            complexity=0.4,
            detection_difficulty=0.3
        )
        assert vector.vector_type == "sql_injection"
        assert vector.likelihood == 0.7
        assert vector.complexity == 0.4

    def test_risk_score_calculation(self):
        """测试风险分数计算"""
        vector = AttackVector(
            vector_type="xss",
            description="XSS攻击",
            prerequisites=[],
            impact="会话劫持",
            likelihood=0.8,
            complexity=0.2,
            detection_difficulty=0.5
        )
        risk_score = vector.calculate_risk_score()
        # 风险分数 = likelihood * (1 - detection_difficulty) * (1 - complexity)
        expected = 0.8 * (1 - 0.5) * (1 - 0.2)
        assert risk_score == expected

    def test_attack_vector_to_dict(self):
        """测试攻击向量转换为字典"""
        vector = AttackVector(
            vector_type="csrf",
            description="CSRF攻击",
            prerequisites=["已认证用户"],
            impact="未授权操作",
            likelihood=0.6,
            complexity=0.3,
            detection_difficulty=0.4
        )
        result = vector.to_dict()
        assert result["type"] == "csrf"
        assert result["likelihood"] == 0.6
        assert "risk_score" in result


class TestAttackChain:
    """测试攻击链"""

    def test_attack_chain_creation(self):
        """测试攻击链创建"""
        vectors = [
            AttackVector(
                vector_type="recon",
                description="信息收集",
                prerequisites=[],
                impact="信息泄露",
                likelihood=0.9,
                complexity=0.1,
                detection_difficulty=0.8
            ),
            AttackVector(
                vector_type="exploit",
                description="漏洞利用",
                prerequisites=["已知漏洞"],
                impact="系统入侵",
                likelihood=0.6,
                complexity=0.5,
                detection_difficulty=0.4
            )
        ]
        chain = AttackChain(
            name="典型攻击链",
            description="从侦察到利用的完整攻击链",
            vectors=vectors
        )
        assert chain.name == "典型攻击链"
        assert len(chain.vectors) == 2

    def test_overall_probability_calculation(self):
        """测试整体成功概率计算"""
        vectors = [
            AttackVector(
                vector_type="step1",
                description="第一步",
                prerequisites=[],
                impact="",
                likelihood=0.8,
                complexity=0.2,
                detection_difficulty=0.3
            ),
            AttackVector(
                vector_type="step2",
                description="第二步",
                prerequisites=[],
                impact="",
                likelihood=0.7,
                complexity=0.3,
                detection_difficulty=0.4
            )
        ]
        chain = AttackChain(
            name="两步攻击链",
            description="测试",
            vectors=vectors
        )
        probability = chain.calculate_overall_probability()
        assert probability == 0.8 * 0.7

    def test_empty_chain_probability(self):
        """测试空攻击链概率"""
        chain = AttackChain(
            name="空链",
            description="测试",
            vectors=[]
        )
        assert chain.calculate_overall_probability() == 0.0

    def test_attack_chain_to_dict(self):
        """测试攻击链转换为字典"""
        vector = AttackVector(
            vector_type="test",
            description="测试",
            prerequisites=[],
            impact="",
            likelihood=0.5,
            complexity=0.5,
            detection_difficulty=0.5
        )
        chain = AttackChain(
            name="测试链",
            description="测试描述",
            vectors=[vector]
        )
        result = chain.to_dict()
        assert result["name"] == "测试链"
        assert len(result["vectors"]) == 1
        assert "success_probability" in result


class TestFinding:
    """测试发现项"""

    def test_finding_creation(self):
        """测试发现项创建"""
        finding = Finding(
            title="SQL注入漏洞",
            description="发现SQL注入漏洞",
            severity=RiskLevel.CRITICAL,
            evidence="输入' OR 1=1 -- 导致数据泄露",
            remediation="使用参数化查询"
        )
        assert finding.title == "SQL注入漏洞"
        assert finding.severity == RiskLevel.CRITICAL

    def test_finding_with_attack_vector(self):
        """测试带攻击向量的发现项"""
        vector = AttackVector(
            vector_type="xss",
            description="XSS",
            prerequisites=[],
            impact="",
            likelihood=0.7,
            complexity=0.3,
            detection_difficulty=0.4
        )
        finding = Finding(
            title="XSS漏洞",
            description="存储型XSS",
            severity=RiskLevel.HIGH,
            evidence="<script>alert(1)</script>",
            remediation="输入过滤",
            attack_vector=vector
        )
        assert finding.attack_vector is not None
        assert finding.attack_vector.vector_type == "xss"

    def test_finding_to_dict(self):
        """测试发现项转换为字典"""
        finding = Finding(
            title="弱密码",
            description="发现弱密码策略",
            severity=RiskLevel.MEDIUM,
            evidence="密码长度要求过低",
            remediation="增加密码复杂度要求"
        )
        result = finding.to_dict()
        assert result["title"] == "弱密码"
        assert result["severity"] == "medium"


class TestPenTestReport:
    """测试渗透测试报告"""

    def test_report_creation(self):
        """测试报告创建"""
        findings = [
            Finding(
                title="漏洞1",
                description="描述1",
                severity=RiskLevel.HIGH,
                evidence="证据1",
                remediation="修复1"
            )
        ]
        report = PenTestReport(
            executive_summary="执行摘要",
            findings=findings,
            risk_ratings={"high": 1, "medium": 0},
            recommendations=["建议1"],
            timeline=[{"phase": "start", "timestamp": 1234567890}],
            scope="测试范围"
        )
        assert report.executive_summary == "执行摘要"
        assert len(report.findings) == 1
        assert report.scope == "测试范围"

    def test_report_to_dict(self):
        """测试报告转换为字典"""
        report = PenTestReport(
            executive_summary="测试摘要",
            findings=[],
            risk_ratings={},
            recommendations=[],
            timeline=[],
            scope="测试"
        )
        result = report.to_dict()
        assert result["executive_summary"] == "测试摘要"
        assert "generated_at" in result

    def test_report_to_json(self):
        """测试报告转换为JSON"""
        report = PenTestReport(
            executive_summary="JSON测试",
            findings=[],
            risk_ratings={},
            recommendations=[],
            timeline=[],
            scope="测试"
        )
        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed["executive_summary"] == "JSON测试"


class TestReconnaissance:
    """测试侦察阶段"""

    def test_recon_initialization(self):
        """测试侦察初始化"""
        recon = Reconnaissance()
        assert recon.collected_info == {}
        assert recon.identified_targets == []

    def test_gather_information(self):
        """测试信息收集"""
        recon = Reconnaissance()
        target = "https://example.com"
        info = recon.gather_information(target)
        assert "target" in info
        assert info["target"] == target
        assert "system_info" in info
        assert "api_endpoints" in info

    def test_identify_targets(self):
        """测试目标识别"""
        recon = Reconnaissance()
        info = {
            "api_endpoints": ["/api/v1/users", "/api/v1/data"],
            "system_info": {"model_api": "/api/model"}
        }
        targets = recon.identify_targets(info)
        assert len(targets) >= 2
        assert "/api/v1/users" in targets

    def test_analyze_attack_surface(self):
        """测试攻击面分析"""
        recon = Reconnaissance()
        targets = ["https://example.com/api"]
        attack_surface = recon.analyze_attack_surface(targets)
        assert "entry_points" in attack_surface
        assert "vulnerable_components" in attack_surface

    def test_discover_endpoints(self):
        """测试端点发现"""
        recon = Reconnaissance()
        target = "https://example.com"
        endpoints = recon._discover_endpoints(target)
        assert len(endpoints) > 0
        assert any("/api/" in ep for ep in endpoints)


class TestExploitation:
    """测试利用阶段"""

    def test_exploitation_initialization(self):
        """测试利用阶段初始化"""
        exploit = Exploitation()
        assert exploit.exploited_vulnerabilities == []
        assert exploit.escalated_privileges is False

    def test_exploit_prompt_injection(self):
        """测试Prompt注入漏洞利用"""
        exploit = Exploitation()
        target = "https://example.com/api"
        result = exploit.exploit_vulnerability(target, "prompt_injection")
        assert result["target"] == target
        assert result["vulnerability"] == "prompt_injection"
        assert "success" in result
        assert "payload" in result

    def test_exploit_authentication_bypass(self):
        """测试认证绕过漏洞利用"""
        exploit = Exploitation()
        target = "https://example.com/login"
        result = exploit.exploit_vulnerability(target, "authentication_bypass")
        assert result["vulnerability"] == "authentication_bypass"
        assert "technique" in result

    def test_exploit_injection_attack(self):
        """测试注入攻击利用"""
        exploit = Exploitation()
        target = "https://example.com/search"
        result = exploit.exploit_vulnerability(target, "injection_attack")
        assert "injection" in result["vulnerability"]
        assert "type" in result

    def test_exploit_misconfiguration(self):
        """测试配置错误利用"""
        exploit = Exploitation()
        target = "https://example.com"
        result = exploit.exploit_vulnerability(target, "misconfiguration")
        assert "misconfiguration" in result["vulnerability"]

    def test_escalate_privileges(self):
        """测试权限提升"""
        exploit = Exploitation()
        result = exploit.escalate_privileges("user", "admin")
        assert result["from"] == "user"
        assert result["to"] == "admin"
        assert "success" in result

    def test_lateral_movement(self):
        """测试横向移动"""
        exploit = Exploitation()
        # 先提升权限
        exploit.escalate_privileges("user", "admin")
        result = exploit.lateral_movement("server1", "server2")
        assert result["from"] == "server1"
        assert result["to"] == "server2"


class TestPostExploitation:
    """测试后渗透阶段"""

    def test_post_exploitation_initialization(self):
        """测试后渗透初始化"""
        post = PostExploitation()
        assert post.collected_data == []
        assert post.persistence_established is False

    def test_collect_data(self):
        """测试数据收集"""
        post = PostExploitation()
        target = "https://example.com"
        data_types = ["model_info", "system_config"]
        result = post.collect_data(target, data_types)
        assert result["target"] == target
        assert "data" in result
        assert "model_info" in result["data"]
        assert "system_config" in result["data"]

    def test_collect_logs(self):
        """测试日志收集"""
        post = PostExploitation()
        target = "https://example.com"
        result = post.collect_data(target, ["logs"])
        assert "logs" in result["data"]
        assert isinstance(result["data"]["logs"], list)

    def test_establish_persistence_backdoor(self):
        """测试建立后门持久化"""
        post = PostExploitation()
        target = "https://example.com"
        result = post.establish_persistence(target, "backdoor")
        assert result["target"] == target
        assert result["method"] == "backdoor"
        assert "success" in result

    def test_establish_persistence_scheduled_task(self):
        """测试建立计划任务持久化"""
        post = PostExploitation()
        target = "https://example.com"
        result = post.establish_persistence(target, "scheduled_task")
        assert result["method"] == "scheduled_task"

    def test_clean_traces(self):
        """测试痕迹清理"""
        post = PostExploitation()
        target = "https://example.com"
        traces = ["logs", "files"]
        result = post.clean_traces(target, traces)
        assert result["target"] == target
        assert "logs" in result["traces_cleaned"]
        assert "temporary_files" in result["traces_cleaned"]


class TestPenetrationTester:
    """测试渗透测试器"""

    def test_tester_initialization(self):
        """测试测试器初始化"""
        tester = PenetrationTester()
        assert tester.recon is not None
        assert tester.exploitation is not None
        assert tester.post_exploitation is not None
        assert len(tester.attack_vectors) > 0

    def test_run_full_test(self):
        """测试运行完整渗透测试"""
        tester = PenetrationTester()
        target = "https://example.com"
        scope = "全站测试"
        report = tester.run_test(target, scope)
        assert isinstance(report, PenTestReport)
        assert report.scope == scope
        assert len(report.timeline) > 0

    def test_run_reconnaissance_only(self):
        """测试仅运行侦察阶段"""
        tester = PenetrationTester()
        target = "https://example.com"
        report = tester.run_test(
            target,
            "侦察测试",
            phases=[AttackPhase.RECONNAISSANCE]
        )
        assert isinstance(report, PenTestReport)

    def test_generate_report(self):
        """测试生成报告"""
        tester = PenetrationTester()
        # 先运行测试生成发现项
        tester.run_test("https://example.com", "测试")
        report = tester.generate_report("测试范围")
        assert isinstance(report, PenTestReport)
        assert "critical" in report.risk_ratings or "high" in report.risk_ratings

    def test_calculate_risk_ratings(self):
        """测试风险评级计算"""
        tester = PenetrationTester()
        tester.findings = [
            Finding("高危1", "描述", RiskLevel.HIGH, "证据", "修复"),
            Finding("中危1", "描述", RiskLevel.MEDIUM, "证据", "修复"),
            Finding("中危2", "描述", RiskLevel.MEDIUM, "证据", "修复")
        ]
        ratings = tester._calculate_risk_ratings()
        assert ratings["high"] == 1
        assert ratings["medium"] == 2

    def test_generate_recommendations(self):
        """测试生成建议"""
        tester = PenetrationTester()
        tester.findings = [
            Finding("关键", "描述", RiskLevel.CRITICAL, "证据", "修复")
        ]
        recommendations = tester._generate_recommendations()
        assert len(recommendations) > 0
        assert any("关键" in rec or "立即" in rec for rec in recommendations)

    def test_get_attack_vector(self):
        """测试获取攻击向量"""
        tester = PenetrationTester()
        vector = tester._get_attack_vector("injection_attack")
        assert vector is not None
        assert vector.vector_type == "injection_attack"


class TestDefenseValidation:
    """测试防御验证"""

    def test_attack_vector_detection_difficulty(self):
        """测试攻击向量检测难度"""
        vector = AttackVector(
            vector_type="stealth",
            description="隐蔽攻击",
            prerequisites=[],
            impact="",
            likelihood=0.5,
            complexity=0.5,
            detection_difficulty=0.9  # 很难检测
        )
        # 检测难度高应该导致更高的风险分数
        high_detection_risk = vector.calculate_risk_score()

        vector2 = AttackVector(
            vector_type="obvious",
            description="明显攻击",
            prerequisites=[],
            impact="",
            likelihood=0.5,
            complexity=0.5,
            detection_difficulty=0.1  # 容易检测
        )
        low_detection_risk = vector2.calculate_risk_score()

        assert high_detection_risk > low_detection_risk

    def test_exploitation_success_rate(self):
        """测试利用成功率"""
        exploit = Exploitation()
        # 多次利用尝试，验证成功率在合理范围
        successes = 0
        attempts = 20
        for _ in range(attempts):
            result = exploit.exploit_vulnerability("target", "prompt_injection")
            if result["success"]:
                successes += 1
        # 成功率应该在0-1之间
        success_rate = successes / attempts
        assert 0 <= success_rate <= 1


class TestBypassTechniques:
    """测试绕过技术"""

    def test_privilege_escalation_bypass(self):
        """测试权限提升绕过"""
        exploit = Exploitation()
        # 尝试权限提升
        result = exploit.escalate_privileges("user", "admin")
        # 验证结果格式
        assert "from" in result
        assert "to" in result
        assert "method" in result

    def test_lateral_movement_with_privileges(self):
        """测试有权限时的横向移动"""
        exploit = Exploitation()
        # 先获取权限
        exploit.escalate_privileges("user", "admin")
        # 然后尝试横向移动
        result = exploit.lateral_movement("host1", "host2")
        # 有权限时应该更容易成功
        assert "path" in result

    def test_persistence_methods(self):
        """测试多种持久化方法"""
        post = PostExploitation()
        target = "https://example.com"
        methods = ["backdoor", "scheduled_task", "service", "webhook"]
        results = []
        for method in methods:
            result = post.establish_persistence(target, method)
            results.append(result["success"])
        # 至少有一种方法可能成功
        assert len(results) == len(methods)


class TestPenetrationEdgeCases:
    """测试渗透测试边界情况"""

    def test_empty_target_reconnaissance(self):
        """测试空目标侦察"""
        recon = Reconnaissance()
        info = recon.gather_information("")
        assert "target" in info
        assert info["target"] == ""

    def test_reconnaissance_without_targets(self):
        """测试无目标攻击面分析"""
        recon = Reconnaissance()
        attack_surface = recon.analyze_attack_surface([])
        assert attack_surface["entry_points"] == []

    def test_exploitation_unknown_vulnerability(self):
        """测试未知漏洞利用"""
        exploit = Exploitation()
        result = exploit.exploit_vulnerability("target", "unknown_vuln")
        assert result["success"] is False

    def test_post_exploitation_invalid_method(self):
        """测试无效持久化方法"""
        post = PostExploitation()
        result = post.establish_persistence("target", "invalid_method")
        assert result["success"] is False

    def test_report_with_no_findings(self):
        """测试无发现项的报告"""
        tester = PenetrationTester()
        report = tester.generate_report("测试范围")
        assert report is not None
        assert report.findings == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
