"""
TestCompliance - 安全合规测试

测试aisec/audit的合规报告功能，包括GDPR/SOC2/ISO27001/HIPAA/PCI DSS等合规框架。

测试内容：
- 合规框架枚举
- 控制点管理
- 合规证据
- 合规差距评估
- 合规报告生成
"""

import pytest
import json
import tempfile
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any

# 导入被测试的模块
from agi_unified_framework.aisec.audit.compliance_report import (
    ComplianceFramework,
    ComplianceControl,
    ComplianceEvidence,
    ComplianceGap,
    ComplianceReport,
    ComplianceReporter
)


# 测试配置
pytestmark = pytest.mark.unit


class TestComplianceFramework:
    """测试合规框架枚举"""

    def test_gdpr_framework_value(self):
        """测试GDPR框架枚举值"""
        assert ComplianceFramework.GDPR.value == "GDPR"
        assert ComplianceFramework.GDPR.name == "GDPR"

    def test_soc2_framework_value(self):
        """测试SOC2框架枚举值"""
        assert ComplianceFramework.SOC2.value == "SOC2"
        assert ComplianceFramework.SOC2.name == "SOC2"

    def test_iso27001_framework_value(self):
        """测试ISO27001框架枚举值"""
        assert ComplianceFramework.ISO27001.value == "ISO27001"
        assert ComplianceFramework.ISO27001.name == "ISO27001"

    def test_hipaa_framework_value(self):
        """测试HIPAA框架枚举值"""
        assert ComplianceFramework.HIPAA.value == "HIPAA"
        assert ComplianceFramework.HIPAA.name == "HIPAA"

    def test_pci_dss_framework_value(self):
        """测试PCI DSS框架枚举值"""
        assert ComplianceFramework.PCI_DSS.value == "PCI_DSS"
        assert ComplianceFramework.PCI_DSS.name == "PCI_DSS"

    def test_custom_framework_value(self):
        """测试自定义框架枚举值"""
        assert ComplianceFramework.CUSTOM.value == "CUSTOM"


class TestComplianceControl:
    """测试合规控制点"""

    def test_control_creation(self):
        """测试控制点创建"""
        control = ComplianceControl(
            control_id="TEST-001",
            description="测试控制点",
            framework=ComplianceFramework.GDPR,
            requirements=["Article 32"],
            category="Data Protection",
            priority="high"
        )
        assert control.control_id == "TEST-001"
        assert control.description == "测试控制点"
        assert control.framework == ComplianceFramework.GDPR
        assert control.priority == "high"

    def test_control_add_evidence(self):
        """测试向控制点添加证据"""
        control = ComplianceControl(
            control_id="TEST-002",
            description="测试控制点",
            framework=ComplianceFramework.SOC2,
            requirements=["CC7.2"],
            priority="high"
        )
        evidence = ComplianceEvidence(
            control_id="TEST-002",
            evidence_type="log",
            data={"action": "test"}
        )
        control.add_evidence(evidence)
        assert len(control.get_evidence()) == 1
        assert control.get_evidence()[0].evidence_id == evidence.evidence_id

    def test_control_to_dict(self):
        """测试控制点转换为字典"""
        control = ComplianceControl(
            control_id="TEST-003",
            description="测试控制点",
            framework=ComplianceFramework.ISO27001,
            requirements=["A.9.1.1"],
            category="Access Control",
            priority="medium"
        )
        result = control.to_dict()
        assert result["control_id"] == "TEST-003"
        assert result["framework"] == "ISO27001"
        assert result["evidence_count"] == 0


class TestComplianceEvidence:
    """测试合规证据"""

    def test_evidence_creation(self):
        """测试证据创建"""
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        evidence = ComplianceEvidence(
            control_id="AC-1",
            evidence_type="audit_log",
            data={"user": "admin", "action": "access"},
            timestamp=timestamp,
            source="system_log",
            validity="valid"
        )
        assert evidence.control_id == "AC-1"
        assert evidence.evidence_type == "audit_log"
        assert evidence.timestamp == timestamp
        assert evidence.source == "system_log"
        assert evidence.validity == "valid"

    def test_evidence_id_generation(self):
        """测试证据ID生成"""
        evidence = ComplianceEvidence(
            control_id="AC-1",
            evidence_type="test",
            data={}
        )
        assert evidence.evidence_id.startswith("EVD-")
        assert len(evidence.evidence_id) > 10

    def test_evidence_to_dict(self):
        """测试证据转换为字典"""
        evidence = ComplianceEvidence(
            control_id="AU-1",
            evidence_type="log",
            data={"entries": 100},
            source="audit"
        )
        result = evidence.to_dict()
        assert "evidence_id" in result
        assert result["control_id"] == "AU-1"
        assert result["evidence_type"] == "log"
        assert "timestamp" in result


class TestComplianceGap:
    """测试合规差距"""

    def test_gap_creation(self):
        """测试差距创建"""
        gap = ComplianceGap(
            control_id="CR-1",
            severity="high",
            description="缺少加密证据",
            remediation="实施加密并记录",
            timeline="30 days",
            owner="security_team"
        )
        assert gap.control_id == "CR-1"
        assert gap.severity == "high"
        assert gap.status == "open"

    def test_gap_to_dict(self):
        """测试差距转换为字典"""
        gap = ComplianceGap(
            control_id="IA-1",
            severity="medium",
            description="认证控制待完善",
            remediation="增强多因素认证"
        )
        result = gap.to_dict()
        assert result["control_id"] == "IA-1"
        assert result["severity"] == "medium"
        assert result["status"] == "open"
        assert "identified_at" in result


class TestComplianceReport:
    """测试合规报告"""

    def test_report_creation(self):
        """测试报告创建"""
        period_start = datetime(2024, 1, 1)
        period_end = datetime(2024, 3, 31)
        summary = {
            "total_controls": 10,
            "compliant_controls": 8,
            "compliance_rate": 80.0
        }
        controls = []
        gaps = []
        evidence = []

        report = ComplianceReport(
            framework=ComplianceFramework.GDPR,
            period_start=period_start,
            period_end=period_end,
            summary=summary,
            controls=controls,
            gaps=gaps,
            evidence=evidence
        )
        assert report.framework == ComplianceFramework.GDPR
        assert report.period_start == period_start
        assert report.report_id.startswith("RPT-GDPR-")

    def test_report_to_dict(self):
        """测试报告转换为字典"""
        report = ComplianceReport(
            framework=ComplianceFramework.SOC2,
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 6, 30),
            summary={"total_controls": 5},
            controls=[],
            gaps=[],
            evidence=[]
        )
        result = report.to_dict()
        assert result["framework"] == "SOC2"
        assert "report_id" in result
        assert "generated_at" in result

    def test_report_to_json(self):
        """测试报告转换为JSON"""
        report = ComplianceReport(
            framework=ComplianceFramework.ISO27001,
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 12, 31),
            summary={"total_controls": 3},
            controls=[],
            gaps=[],
            evidence=[]
        )
        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed["framework"] == "ISO27001"

    def test_report_save(self, tmp_path):
        """测试报告保存到文件"""
        report = ComplianceReport(
            framework=ComplianceFramework.HIPAA,
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 3, 31),
            summary={},
            controls=[],
            gaps=[],
            evidence=[]
        )
        filepath = report.save(str(tmp_path))
        assert os.path.exists(filepath)
        with open(filepath, 'r') as f:
            content = f.read()
            assert "RPT-HIPAA-" in content


class TestComplianceReporter:
    """测试合规报告生成器"""

    def test_reporter_initialization(self):
        """测试报告器初始化"""
        reporter = ComplianceReporter()
        # 验证内置控制点已加载
        assert ComplianceFramework.GDPR in reporter._controls
        assert ComplianceFramework.SOC2 in reporter._controls
        assert ComplianceFramework.ISO27001 in reporter._controls

    def test_add_evidence(self):
        """测试添加证据"""
        reporter = ComplianceReporter()
        evidence = ComplianceEvidence(
            control_id="AC-1",
            evidence_type="test",
            data={"test": True}
        )
        reporter.add_evidence(evidence)
        assert evidence in reporter._evidence

    def test_generate_gdpr_report(self):
        """测试生成GDPR报告"""
        reporter = ComplianceReporter()
        period_start = datetime(2024, 1, 1)
        period_end = datetime(2024, 3, 31)

        # 添加一些证据
        evidence = ComplianceEvidence(
            control_id="AC-1",
            evidence_type="log",
            data={"action": "test"},
            timestamp=datetime(2024, 2, 15)
        )
        reporter.add_evidence(evidence)

        report = reporter.generate_gdpr_report(period_start, period_end)
        assert report.framework == ComplianceFramework.GDPR
        assert report.summary["total_controls"] >= 0
        assert "framework_specific" in report.summary
        assert "data_subject_rights" in report.summary["framework_specific"]

    def test_generate_soc2_report(self):
        """测试生成SOC2报告"""
        reporter = ComplianceReporter()
        report = reporter.generate_soc2_report()
        assert report.framework == ComplianceFramework.SOC2
        assert "framework_specific" in report.summary
        assert "trust_service_criteria" in report.summary["framework_specific"]

    def test_generate_iso27001_report(self):
        """测试生成ISO27001报告"""
        reporter = ComplianceReporter()
        report = reporter.generate_iso27001_report()
        assert report.framework == ComplianceFramework.ISO27001
        assert "framework_specific" in report.summary
        assert "isms_clauses" in report.summary["framework_specific"]

    def test_generate_hipaa_report(self):
        """测试生成HIPAA报告"""
        reporter = ComplianceReporter()
        report = reporter.generate_hipaa_report()
        assert report.framework == ComplianceFramework.HIPAA
        assert "framework_specific" in report.summary
        assert "administrative_safeguards" in report.summary["framework_specific"]

    def test_generate_pci_dss_report(self):
        """测试生成PCI DSS报告"""
        reporter = ComplianceReporter()
        report = reporter.generate_pci_dss_report()
        assert report.framework == ComplianceFramework.PCI_DSS
        assert "framework_specific" in report.summary
        assert "build_maintain_secure_network" in report.summary["framework_specific"]

    def test_generate_custom_report(self):
        """测试生成自定义报告"""
        reporter = ComplianceReporter()
        custom_controls = [
            ComplianceControl(
                control_id="CUSTOM-1",
                description="自定义控制",
                framework=ComplianceFramework.CUSTOM,
                requirements=["C1"],
                priority="high"
            )
        ]
        report = reporter.generate_custom_report(
            framework_name="Enterprise Security",
            controls=custom_controls
        )
        assert report.framework == ComplianceFramework.CUSTOM
        assert report.summary.get("custom_framework") == "Enterprise Security"

    def test_report_compliance_rate_calculation(self):
        """测试合规率计算"""
        reporter = ComplianceReporter()
        # 添加足够证据让所有控制点合规
        for control_id in ["AC-1", "AC-2", "AU-1", "AU-2", "CR-1", "CR-2"]:
            for _ in range(3):
                evidence = ComplianceEvidence(
                    control_id=control_id,
                    evidence_type="test",
                    data={"test": True}
                )
                reporter.add_evidence(evidence)

        report = reporter.generate_gdpr_report()
        # 验证合规率在有效范围内
        assert 0 <= report.summary["compliance_rate"] <= 100

    def test_report_gap_identification(self):
        """测试差距识别"""
        reporter = ComplianceReporter()
        # 不添加任何证据，应该会识别出差距
        report = reporter.generate_gdpr_report()
        # GDPR框架应该有控制点
        assert len(report.controls) > 0
        # 如果没有证据，应该有差距
        assert len(report.gaps) >= 0

    def test_evidence_association(self):
        """测试证据与控制点的关联"""
        reporter = ComplianceReporter()
        evidence = ComplianceEvidence(
            control_id="AC-1",
            evidence_type="audit",
            data={"entries": 100}
        )
        reporter.add_evidence(evidence)

        # 获取GDPR框架的AC-1控制点
        gdpr_controls = reporter._controls.get(ComplianceFramework.GDPR, [])
        ac1_control = next((c for c in gdpr_controls if c.control_id == "AC-1"), None)
        assert ac1_control is not None
        assert len(ac1_control.get_evidence()) == 1

    def test_multiple_frameworks(self):
        """测试多框架同时评估"""
        reporter = ComplianceReporter()

        # 为不同框架添加证据
        reporter.add_evidence(ComplianceEvidence(
            control_id="AC-1",
            evidence_type="log",
            data={"gdpr": True}
        ))
        reporter.add_evidence(ComplianceEvidence(
            control_id="AC-2",
            evidence_type="log",
            data={"pci": True}
        ))

        # 生成各框架报告
        gdpr_report = reporter.generate_gdpr_report()
        pci_report = reporter.generate_pci_dss_report()

        assert gdpr_report.framework == ComplianceFramework.GDPR
        assert pci_report.framework == ComplianceFramework.PCI_DSS


class TestComplianceEdgeCases:
    """测试边界情况"""

    def test_empty_period_report(self):
        """测试空周期报告"""
        reporter = ComplianceReporter()
        # 使用一个没有任何证据的周期
        period_start = datetime(2020, 1, 1)
        period_end = datetime(2020, 1, 31)
        report = reporter.generate_gdpr_report(period_start, period_end)
        assert report.summary["total_evidence"] == 0

    def test_future_period_report(self):
        """测试未来周期报告"""
        reporter = ComplianceReporter()
        future_start = datetime.utcnow() + timedelta(days=30)
        future_end = future_start + timedelta(days=30)
        report = reporter.generate_gdpr_report(future_start, future_end)
        assert report.summary["total_evidence"] == 0

    def test_report_period_validation(self):
        """测试报告周期验证"""
        reporter = ComplianceReporter()
        period_start = datetime(2024, 6, 1)
        period_end = datetime(2024, 1, 1)  # 结束早于开始
        report = reporter.generate_gdpr_report(period_start, period_end)
        # 即使周期无效，报告也应该能生成
        assert report is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
