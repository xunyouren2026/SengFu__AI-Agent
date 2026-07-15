"""
Compliance Report Module

提供合规报告生成功能，支持多种合规框架。
"""

import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set


class ComplianceFramework(Enum):
    """合规框架枚举"""
    GDPR = "GDPR"  # 通用数据保护条例
    SOC2 = "SOC2"  # 服务组织控制2
    ISO27001 = "ISO27001"  # 信息安全管理体系
    HIPAA = "HIPAA"  # 健康保险可携性和责任法案
    PCI_DSS = "PCI_DSS"  # 支付卡行业数据安全标准
    CUSTOM = "CUSTOM"  # 自定义框架


class ComplianceControl:
    """
    合规控制点类
    
    表示合规框架中的一个具体控制要求。
    """
    
    def __init__(
        self,
        control_id: str,
        description: str,
        framework: ComplianceFramework,
        requirements: List[str],
        category: str = "",
        priority: str = "medium"
    ):
        self.control_id = control_id
        self.description = description
        self.framework = framework
        self.requirements = requirements
        self.category = category
        self.priority = priority
        self._evidence: List['ComplianceEvidence'] = []
    
    def add_evidence(self, evidence: 'ComplianceEvidence') -> None:
        """添加合规证据"""
        self._evidence.append(evidence)
    
    def get_evidence(self) -> List['ComplianceEvidence']:
        """获取所有证据"""
        return list(self._evidence)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "control_id": self.control_id,
            "description": self.description,
            "framework": self.framework.value,
            "requirements": self.requirements,
            "category": self.category,
            "priority": self.priority,
            "evidence_count": len(self._evidence)
        }


class ComplianceEvidence:
    """
    合规证据类
    
    记录支持合规控制点的证据数据。
    """
    
    def __init__(
        self,
        control_id: str,
        evidence_type: str,
        data: Dict[str, Any],
        timestamp: Optional[datetime] = None,
        source: str = "",
        validity: str = "valid"
    ):
        self.control_id = control_id
        self.evidence_type = evidence_type
        self.data = data
        self.timestamp = timestamp or datetime.utcnow()
        self.source = source
        self.validity = validity
        self.evidence_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """生成唯一证据ID"""
        timestamp_str = self.timestamp.strftime("%Y%m%d%H%M%S%f")
        return f"EVD-{timestamp_str}-{hash(self.control_id) % 10000:04d}"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "evidence_id": self.evidence_id,
            "control_id": self.control_id,
            "evidence_type": self.evidence_type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "validity": self.validity
        }


class ComplianceGap:
    """
    合规差距类
    
    表示与合规要求的差距。
    """
    
    def __init__(
        self,
        control_id: str,
        severity: str,
        description: str,
        remediation: str,
        timeline: str = "",
        owner: str = ""
    ):
        self.control_id = control_id
        self.severity = severity
        self.description = description
        self.remediation = remediation
        self.timeline = timeline
        self.owner = owner
        self.identified_at = datetime.utcnow()
        self.status = "open"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "control_id": self.control_id,
            "severity": self.severity,
            "description": self.description,
            "remediation": self.remediation,
            "timeline": self.timeline,
            "owner": self.owner,
            "identified_at": self.identified_at.isoformat(),
            "status": self.status
        }


class ComplianceReport:
    """
    合规报告类
    
    包含完整的合规评估结果。
    """
    
    def __init__(
        self,
        framework: ComplianceFramework,
        period_start: datetime,
        period_end: datetime,
        summary: Dict[str, Any],
        controls: List[ComplianceControl],
        gaps: List[ComplianceGap],
        evidence: List[ComplianceEvidence]
    ):
        self.framework = framework
        self.period_start = period_start
        self.period_end = period_end
        self.summary = summary
        self.controls = controls
        self.gaps = gaps
        self.evidence = evidence
        self.generated_at = datetime.utcnow()
        self.report_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """生成唯一报告ID"""
        timestamp_str = self.generated_at.strftime("%Y%m%d%H%M%S")
        return f"RPT-{self.framework.value}-{timestamp_str}"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "report_id": self.report_id,
            "framework": self.framework.value,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "generated_at": self.generated_at.isoformat(),
            "summary": self.summary,
            "controls": [c.to_dict() for c in self.controls],
            "gaps": [g.to_dict() for g in self.gaps],
            "evidence_count": len(self.evidence)
        }
    
    def to_json(self, indent: int = 2) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def save(self, output_dir: str) -> str:
        """
        保存报告到文件
        
        Args:
            output_dir: 输出目录
            
        Returns:
            保存的文件路径
        """
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{self.report_id}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
        
        return filepath


class ComplianceReporter:
    """
    合规报告生成器
    
    支持多种合规框架的报告生成。
    """
    
    def __init__(self):
        self._controls: Dict[ComplianceFramework, List[ComplianceControl]] = {}
        self._evidence: List[ComplianceEvidence] = []
        self._init_builtin_controls()
    
    def _init_builtin_controls(self) -> None:
        """初始化内置控制点映射"""
        # 数据访问控制
        self._add_control(ComplianceControl(
            control_id="AC-1",
            description="数据访问控制 - 确保只有授权用户可以访问敏感数据",
            framework=ComplianceFramework.ISO27001,
            requirements=["A.9.1.1", "A.9.1.2"],
            category="Access Control",
            priority="high"
        ))
        
        self._add_control(ComplianceControl(
            control_id="AC-2",
            description="Data Access Control - Limit access to cardholder data",
            framework=ComplianceFramework.PCI_DSS,
            requirements=["Requirement 7", "Requirement 8"],
            category="Access Control",
            priority="high"
        ))
        
        # 审计日志完整性
        self._add_control(ComplianceControl(
            control_id="AU-1",
            description="审计日志完整性 - 确保审计日志的完整性和不可篡改性",
            framework=ComplianceFramework.SOC2,
            requirements=["CC7.2", "CC7.3"],
            category="Audit Logging",
            priority="high"
        ))
        
        self._add_control(ComplianceControl(
            control_id="AU-2",
            description="Audit Log Integrity - Maintain audit trails for system access",
            framework=ComplianceFramework.HIPAA,
            requirements=["164.312(b)"],
            category="Audit Controls",
            priority="high"
        ))
        
        # 数据加密
        self._add_control(ComplianceControl(
            control_id="CR-1",
            description="数据加密 - 对存储和传输的敏感数据进行加密保护",
            framework=ComplianceFramework.GDPR,
            requirements=["Article 32"],
            category="Data Protection",
            priority="high"
        ))
        
        self._add_control(ComplianceControl(
            control_id="CR-2",
            description="Encryption - Protect stored cardholder data with encryption",
            framework=ComplianceFramework.PCI_DSS,
            requirements=["Requirement 3", "Requirement 4"],
            category="Data Protection",
            priority="high"
        ))
        
        # 用户认证
        self._add_control(ComplianceControl(
            control_id="IA-1",
            description="用户认证 - 实施强身份验证机制",
            framework=ComplianceFramework.ISO27001,
            requirements=["A.9.2.1", "A.9.2.4", "A.9.4.2"],
            category="Identity and Access",
            priority="high"
        ))
        
        self._add_control(ComplianceControl(
            control_id="IA-2",
            description="User Authentication - Unique user identification and authentication",
            framework=ComplianceFramework.HIPAA,
            requirements=["164.312(d)"],
            category="Person or Entity Authentication",
            priority="high"
        ))
        
        # 数据保留策略
        self._add_control(ComplianceControl(
            control_id="DM-1",
            description="数据保留策略 - 定义并执行数据保留和删除策略",
            framework=ComplianceFramework.GDPR,
            requirements=["Article 5", "Article 17"],
            category="Data Management",
            priority="medium"
        ))
        
        self._add_control(ComplianceControl(
            control_id="DM-2",
            description="Data Retention - Limit data storage amount and retention time",
            framework=ComplianceFramework.PCI_DSS,
            requirements=["Requirement 3.1"],
            category="Data Retention",
            priority="medium"
        ))
    
    def _add_control(self, control: ComplianceControl) -> None:
        """添加控制点到对应框架"""
        if control.framework not in self._controls:
            self._controls[control.framework] = []
        self._controls[control.framework].append(control)
    
    def add_evidence(self, evidence: ComplianceEvidence) -> None:
        """添加合规证据"""
        self._evidence.append(evidence)
        
        # 将证据关联到对应的控制点
        for framework_controls in self._controls.values():
            for control in framework_controls:
                if control.control_id == evidence.control_id:
                    control.add_evidence(evidence)
    
    def _generate_report_base(
        self,
        framework: ComplianceFramework,
        period_start: datetime,
        period_end: datetime
    ) -> ComplianceReport:
        """生成报告基础结构"""
        controls = self._controls.get(framework, [])
        
        # 收集相关证据
        relevant_evidence = [
            e for e in self._evidence
            if period_start <= e.timestamp <= period_end
        ]
        
        # 评估差距
        gaps = self._evaluate_gaps(controls, relevant_evidence)
        
        # 生成摘要
        summary = self._generate_summary(controls, gaps, relevant_evidence)
        
        return ComplianceReport(
            framework=framework,
            period_start=period_start,
            period_end=period_end,
            summary=summary,
            controls=controls,
            gaps=gaps,
            evidence=relevant_evidence
        )
    
    def _evaluate_gaps(
        self,
        controls: List[ComplianceControl],
        evidence: List[ComplianceEvidence]
    ) -> List[ComplianceGap]:
        """评估合规差距"""
        gaps = []
        
        for control in controls:
            control_evidence = [e for e in evidence if e.control_id == control.control_id]
            
            if not control_evidence:
                gaps.append(ComplianceGap(
                    control_id=control.control_id,
                    severity="high" if control.priority == "high" else "medium",
                    description=f"缺少控制点 {control.control_id} 的合规证据",
                    remediation=f"收集并提交 {control.description} 的相关证据",
                    timeline="30 days"
                ))
            elif len(control_evidence) < 3:
                gaps.append(ComplianceGap(
                    control_id=control.control_id,
                    severity="low",
                    description=f"控制点 {control.control_id} 的证据不足",
                    remediation="收集更多证据以支持合规性声明",
                    timeline="60 days"
                ))
        
        return gaps
    
    def _generate_summary(
        self,
        controls: List[ComplianceControl],
        gaps: List[ComplianceGap],
        evidence: List[ComplianceEvidence]
    ) -> Dict[str, Any]:
        """生成报告摘要"""
        total_controls = len(controls)
        compliant_controls = total_controls - len([g for g in gaps if g.severity == "high"])
        
        severity_counts = defaultdict(int)
        for gap in gaps:
            severity_counts[gap.severity] += 1
        
        return {
            "total_controls": total_controls,
            "compliant_controls": compliant_controls,
            "compliance_rate": round(compliant_controls / total_controls * 100, 2) if total_controls > 0 else 0,
            "total_gaps": len(gaps),
            "gap_severity_distribution": dict(severity_counts),
            "total_evidence": len(evidence),
            "assessment_date": datetime.utcnow().isoformat()
        }
    
    def generate_gdpr_report(
        self,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> ComplianceReport:
        """
        生成GDPR合规报告
        
        Args:
            period_start: 报告期开始时间
            period_end: 报告期结束时间
            
        Returns:
            GDPR合规报告
        """
        period_end = period_end or datetime.utcnow()
        period_start = period_start or (period_end - timedelta(days=90))
        
        report = self._generate_report_base(
            ComplianceFramework.GDPR,
            period_start,
            period_end
        )
        
        # 添加GDPR特定分析
        report.summary["framework_specific"] = {
            "data_subject_rights": self._check_data_subject_rights(),
            "lawful_basis": self._check_lawful_basis(),
            "data_protection_measures": self._check_data_protection_measures()
        }
        
        return report
    
    def generate_soc2_report(
        self,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> ComplianceReport:
        """
        生成SOC2合规报告
        
        Args:
            period_start: 报告期开始时间
            period_end: 报告期结束时间
            
        Returns:
            SOC2合规报告
        """
        period_end = period_end or datetime.utcnow()
        period_start = period_start or (period_end - timedelta(days=180))
        
        report = self._generate_report_base(
            ComplianceFramework.SOC2,
            period_start,
            period_end
        )
        
        # 添加SOC2特定分析
        report.summary["framework_specific"] = {
            "trust_service_criteria": {
                "security": self._check_security_criteria(),
                "availability": self._check_availability_criteria(),
                "processing_integrity": self._check_processing_integrity(),
                "confidentiality": self._check_confidentiality(),
                "privacy": self._check_privacy_criteria()
            }
        }
        
        return report
    
    def generate_iso27001_report(
        self,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> ComplianceReport:
        """
        生成ISO27001合规报告
        
        Args:
            period_start: 报告期开始时间
            period_end: 报告期结束时间
            
        Returns:
            ISO27001合规报告
        """
        period_end = period_end or datetime.utcnow()
        period_start = period_start or (period_end - timedelta(days=365))
        
        report = self._generate_report_base(
            ComplianceFramework.ISO27001,
            period_start,
            period_end
        )
        
        # 添加ISO27001特定分析
        report.summary["framework_specific"] = {
            "isms_clauses": self._check_isms_clauses(),
            "annex_a_controls": self._check_annex_a_controls()
        }
        
        return report
    
    def generate_hipaa_report(
        self,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> ComplianceReport:
        """
        生成HIPAA合规报告
        
        Args:
            period_start: 报告期开始时间
            period_end: 报告期结束时间
            
        Returns:
            HIPAA合规报告
        """
        period_end = period_end or datetime.utcnow()
        period_start = period_start or (period_end - timedelta(days=90))
        
        report = self._generate_report_base(
            ComplianceFramework.HIPAA,
            period_start,
            period_end
        )
        
        # 添加HIPAA特定分析
        report.summary["framework_specific"] = {
            "administrative_safeguards": self._check_administrative_safeguards(),
            "physical_safeguards": self._check_physical_safeguards(),
            "technical_safeguards": self._check_technical_safeguards()
        }
        
        return report
    
    def generate_pci_dss_report(
        self,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> ComplianceReport:
        """
        生成PCI DSS合规报告
        
        Args:
            period_start: 报告期开始时间
            period_end: 报告期结束时间
            
        Returns:
            PCI DSS合规报告
        """
        period_end = period_end or datetime.utcnow()
        period_start = period_start or (period_end - timedelta(days=90))
        
        report = self._generate_report_base(
            ComplianceFramework.PCI_DSS,
            period_start,
            period_end
        )
        
        # 添加PCI DSS特定分析
        report.summary["framework_specific"] = {
            "build_maintain_secure_network": self._check_pci_requirement_1_2(),
            "protect_cardholder_data": self._check_pci_requirement_3_4(),
            "maintain_vulnerability_program": self._check_pci_requirement_5_6(),
            "implement_access_control": self._check_pci_requirement_7_8(),
            "monitor_test_networks": self._check_pci_requirement_10_11(),
            "maintain_info_security_policy": self._check_pci_requirement_12()
        }
        
        return report
    
    def generate_custom_report(
        self,
        framework_name: str,
        controls: List[ComplianceControl],
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> ComplianceReport:
        """
        生成自定义合规报告
        
        Args:
            framework_name: 自定义框架名称
            controls: 控制点列表
            period_start: 报告期开始时间
            period_end: 报告期结束时间
            
        Returns:
            自定义合规报告
        """
        period_end = period_end or datetime.utcnow()
        period_start = period_start or (period_end - timedelta(days=90))
        
        # 收集相关证据
        relevant_evidence = [
            e for e in self._evidence
            if period_start <= e.timestamp <= period_end
        ]
        
        # 评估差距
        gaps = self._evaluate_gaps(controls, relevant_evidence)
        
        # 生成摘要
        summary = self._generate_summary(controls, gaps, relevant_evidence)
        summary["custom_framework"] = framework_name
        
        return ComplianceReport(
            framework=ComplianceFramework.CUSTOM,
            period_start=period_start,
            period_end=period_end,
            summary=summary,
            controls=controls,
            gaps=gaps,
            evidence=relevant_evidence
        )
    
    # 框架特定的检查方法（占位实现）
    def _check_data_subject_rights(self) -> Dict[str, Any]:
        return {"status": "partial", "details": "Manual review required"}
    
    def _check_lawful_basis(self) -> Dict[str, Any]:
        return {"status": "compliant", "details": "Lawful basis documented"}
    
    def _check_data_protection_measures(self) -> Dict[str, Any]:
        return {"status": "compliant", "details": "Encryption and access controls in place"}
    
    def _check_security_criteria(self) -> Dict[str, Any]:
        return {"status": "compliant", "details": "Security controls implemented"}
    
    def _check_availability_criteria(self) -> Dict[str, Any]:
        return {"status": "compliant", "details": "Availability monitoring active"}
    
    def _check_processing_integrity(self) -> Dict[str, Any]:
        return {"status": "partial", "details": "Some processes need review"}
    
    def _check_confidentiality(self) -> Dict[str, Any]:
        return {"status": "compliant", "details": "Confidentiality controls effective"}
    
    def _check_privacy_criteria(self) -> Dict[str, Any]:
        return {"status": "partial", "details": "Privacy impact assessment pending"}
    
    def _check_isms_clauses(self) -> Dict[str, Any]:
        return {"status": "compliant", "details": "ISMS clauses addressed"}
    
    def _check_annex_a_controls(self) -> Dict[str, Any]:
        return {"status": "partial", "details": "Some Annex A controls need evidence"}
    
    def _check_administrative_safeguards(self) -> Dict[str, Any]:
        return {"status": "compliant", "details": "Administrative safeguards in place"}
    
    def _check_physical_safeguards(self) -> Dict[str, Any]:
        return {"status": "not_applicable", "details": "Cloud environment"}
    
    def _check_technical_safeguards(self) -> Dict[str, Any]:
        return {"status": "compliant", "details": "Technical safeguards implemented"}
    
    def _check_pci_requirement_1_2(self) -> Dict[str, Any]:
        return {"status": "compliant", "details": "Firewall configuration maintained"}
    
    def _check_pci_requirement_3_4(self) -> Dict[str, Any]:
        return {"status": "compliant", "details": "Cardholder data encrypted"}
    
    def _check_pci_requirement_5_6(self) -> Dict[str, Any]:
        return {"status": "partial", "details": "Vulnerability scans pending"}
    
    def _check_pci_requirement_7_8(self) -> Dict[str, Any]:
        return {"status": "compliant", "details": "Access controls implemented"}
    
    def _check_pci_requirement_10_11(self) -> Dict[str, Any]:
        return {"status": "compliant", "details": "Monitoring and testing active"}
    
    def _check_pci_requirement_12(self) -> Dict[str, Any]:
        return {"status": "partial", "details": "Policy review scheduled"}
