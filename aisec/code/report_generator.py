"""
代码安全报告生成器
"""
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .static_scanner import CodeIssue, ScanResult, Severity
from .dependency_scanner import Vulnerability, ScanResult as DepScanResult
from .dangerous_imports import DangerousImport


@dataclass
class SecurityReport:
    """安全报告"""
    project_name: str
    scan_time: str
    summary: Dict[str, Any]
    issues: List[Dict[str, Any]]
    vulnerabilities: List[Dict[str, Any]]
    dangerous_imports: List[Dict[str, Any]]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self):
        self._report_format = "detailed"
    
    def generate(
        self,
        project_name: str,
        scan_results: Optional[List[ScanResult]] = None,
        dep_results: Optional[DepScanResult] = None,
        import_results: Optional[Dict[str, List[DangerousImport]]] = None,
        supply_chain_risks: Optional[List] = None
    ) -> SecurityReport:
        """生成安全报告"""
        scan_time = datetime.now().isoformat()
        
        # 收集所有问题
        all_issues = []
        if scan_results:
            for result in scan_results:
                for issue in result.issues:
                    all_issues.append(self._format_issue(issue))
        
        # 收集漏洞
        all_vulnerabilities = []
        if dep_results and dep_results.vulnerabilities:
            for vuln in dep_results.vulnerabilities:
                all_vulnerabilities.append(self._format_vulnerability(vuln))
        
        # 收集危险导入
        all_imports = []
        if import_results:
            for file_path, imports in import_results.items():
                for imp in imports:
                    all_imports.append(self._format_import(imp, file_path))
        
        # 收集供应链风险
        all_supply_risks = []
        if supply_chain_risks:
            for risk in supply_chain_risks:
                all_supply_risks.append({
                    "type": risk.risk_type.value,
                    "package": risk.package_name,
                    "severity": risk.severity,
                    "description": risk.description,
                    "recommendation": risk.recommendation
                })
        
        # 生成摘要
        summary = self._generate_summary(
            all_issues,
            all_vulnerabilities,
            all_imports,
            all_supply_risks
        )
        
        # 生成建议
        recommendations = self._generate_recommendations(
            all_issues,
            all_vulnerabilities,
            all_imports,
            all_supply_risks
        )
        
        return SecurityReport(
            project_name=project_name,
            scan_time=scan_time,
            summary=summary,
            issues=all_issues,
            vulnerabilities=all_vulnerabilities,
            dangerous_imports=all_imports,
            recommendations=recommendations,
            metadata={
                "supply_chain_risks": all_supply_risks
            }
        )
    
    def _format_issue(self, issue: CodeIssue) -> Dict[str, Any]:
        """格式化代码问题"""
        return {
            "type": issue.issue_type.value,
            "severity": issue.severity.value,
            "message": issue.message,
            "file": issue.file_path,
            "line": issue.line_number,
            "column": issue.column,
            "code_snippet": issue.code_snippet,
            "confidence": issue.confidence,
            "cwe": issue.cwe_id,
            "owasp": issue.owasp_category
        }
    
    def _format_vulnerability(self, vuln: Vulnerability) -> Dict[str, Any]:
        """格式化漏洞"""
        return {
            "cve_id": vuln.cve_id,
            "package": vuln.package_name,
            "severity": vuln.severity.value,
            "description": vuln.description,
            "affected_versions": vuln.affected_versions,
            "fixed_versions": vuln.fixed_versions,
            "cvss_score": vuln.cvss_score,
            "exploit_available": vuln.exploit_available,
            "references": vuln.references
        }
    
    def _format_import(self, imp: DangerousImport, file_path: str) -> Dict[str, Any]:
        """格式化危险导入"""
        return {
            "module": imp.module,
            "alias": imp.alias,
            "file": file_path,
            "line": imp.line_number,
            "risk_level": imp.risk_level.value,
            "reason": imp.reason,
            "suggestion": imp.suggestion,
            "code_snippet": imp.code_snippet
        }
    
    def _generate_summary(
        self,
        issues: List[Dict],
        vulnerabilities: List[Dict],
        imports: List[Dict],
        supply_risks: List[Dict]
    ) -> Dict[str, Any]:
        """生成摘要"""
        # 按严重程度统计
        issue_severity = {}
        for issue in issues:
            sev = issue.get("severity", "unknown")
            issue_severity[sev] = issue_severity.get(sev, 0) + 1
        
        vuln_severity = {}
        for vuln in vulnerabilities:
            sev = vuln.get("severity", "unknown")
            vuln_severity[sev] = vuln_severity.get(sev, 0) + 1
        
        import_risk = {}
        for imp in imports:
            risk = imp.get("risk_level", "unknown")
            import_risk[risk] = import_risk.get(risk, 0) + 1
        
        supply_severity = {}
        for risk in supply_risks:
            sev = risk.get("severity", "unknown")
            supply_severity[sev] = supply_severity.get(sev, 0) + 1
        
        return {
            "total_issues": len(issues),
            "total_vulnerabilities": len(vulnerabilities),
            "total_dangerous_imports": len(imports),
            "total_supply_chain_risks": len(supply_risks),
            "issues_by_severity": issue_severity,
            "vulnerabilities_by_severity": vuln_severity,
            "imports_by_risk": import_risk,
            "supply_chain_by_severity": supply_severity,
            "critical_total": (
                issue_severity.get("critical", 0) +
                vuln_severity.get("critical", 0) +
                import_risk.get("critical", 0) +
                supply_severity.get("critical", 0)
            ),
            "high_total": (
                issue_severity.get("high", 0) +
                vuln_severity.get("high", 0) +
                import_risk.get("high", 0) +
                supply_severity.get("high", 0)
            )
        }
    
    def _generate_recommendations(
        self,
        issues: List[Dict],
        vulnerabilities: List[Dict],
        imports: List[Dict],
        supply_risks: List[Dict]
    ) -> List[str]:
        """生成修复建议"""
        recommendations = []
        
        # 基于问题类型生成建议
        issue_types = set(issue.get("type") for issue in issues)
        
        if "sql_injection" in issue_types:
            recommendations.append("使用参数化查询替代字符串拼接，防止SQL注入")
        
        if "command_injection" in issue_types:
            recommendations.append("避免使用shell=True，验证并转义用户输入")
        
        if "code_injection" in issue_types:
            recommendations.append("移除eval/exec调用，使用安全的替代方案")
        
        if "hardcoded_secret" in issue_types:
            recommendations.append("使用环境变量或密钥管理服务存储敏感信息")
        
        if "insecure_deserialize" in issue_types:
            recommendations.append("使用json替代pickle，或限制反序列化的对象类型")
        
        if "path_traversal" in issue_types:
            recommendations.append("验证并规范化文件路径，使用白名单限制访问范围")
        
        # 基于漏洞生成建议
        if vulnerabilities:
            critical_vulns = [v for v in vulnerabilities if v.get("severity") == "critical"]
            if critical_vulns:
                packages = set(v.get("package") for v in critical_vulns)
                recommendations.append(f"立即更新以下包以修复严重漏洞: {', '.join(packages)}")
        
        # 基于危险导入生成建议
        if imports:
            critical_imports = [i for i in imports if i.get("risk_level") == "critical"]
            if critical_imports:
                modules = set(i.get("module") for i in critical_imports)
                recommendations.append(f"审查并移除危险导入: {', '.join(modules)}")
        
        # 基于供应链风险生成建议
        if supply_risks:
            typo_risks = [r for r in supply_risks if r.get("type") == "typosquatting"]
            if typo_risks:
                recommendations.append("验证包名拼写，确保使用正确的官方包")
            
            malicious = [r for r in supply_risks if r.get("type") == "malicious_package"]
            if malicious:
                recommendations.append("立即移除已识别的恶意包")
        
        # 通用建议
        if not recommendations:
            recommendations.append("继续保持安全编码实践，定期进行安全扫描")
        
        return recommendations
    
    def to_json(self, report: SecurityReport, indent: int = 2) -> str:
        """转换为JSON"""
        return json.dumps({
            "project_name": report.project_name,
            "scan_time": report.scan_time,
            "summary": report.summary,
            "issues": report.issues,
            "vulnerabilities": report.vulnerabilities,
            "dangerous_imports": report.dangerous_imports,
            "recommendations": report.recommendations,
            "metadata": report.metadata
        }, indent=indent, ensure_ascii=False)
    
    def to_markdown(self, report: SecurityReport) -> str:
        """转换为Markdown"""
        lines = [
            f"# 安全扫描报告: {report.project_name}",
            "",
            f"**扫描时间**: {report.scan_time}",
            "",
            "## 摘要",
            "",
            f"- 总问题数: {report.summary['total_issues']}",
            f"- 总漏洞数: {report.summary['total_vulnerabilities']}",
            f"- 危险导入数: {report.summary['total_dangerous_imports']}",
            f"- 供应链风险数: {report.summary['total_supply_chain_risks']}",
            f"- 严重问题: {report.summary['critical_total']}",
            f"- 高危问题: {report.summary['high_total']}",
            "",
        ]
        
        if report.issues:
            lines.extend([
                "## 代码安全问题",
                "",
                "| 类型 | 严重程度 | 文件 | 行号 | 描述 |",
                "|------|----------|------|------|------|"
            ])
            for issue in report.issues[:50]:  # 限制显示数量
                lines.append(
                    f"| {issue['type']} | {issue['severity']} | "
                    f"{issue['file'].split('/')[-1]} | {issue['line']} | {issue['message'][:50]} |"
                )
            lines.append("")
        
        if report.vulnerabilities:
            lines.extend([
                "## 依赖漏洞",
                "",
                "| CVE | 包名 | 严重程度 | 描述 |",
                "|-----|------|----------|------|"
            ])
            for vuln in report.vulnerabilities[:50]:
                lines.append(
                    f"| {vuln['cve_id']} | {vuln['package']} | "
                    f"{vuln['severity']} | {vuln['description'][:50]} |"
                )
            lines.append("")
        
        if report.dangerous_imports:
            lines.extend([
                "## 危险导入",
                "",
                "| 模块 | 风险等级 | 原因 |",
                "|------|----------|------|"
            ])
            for imp in report.dangerous_imports[:50]:
                lines.append(
                    f"| {imp['module']} | {imp['risk_level']} | {imp['reason'][:50]} |"
                )
            lines.append("")
        
        if report.recommendations:
            lines.extend([
                "## 修复建议",
                ""
            ])
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")
        
        return "\n".join(lines)
    
    def to_html(self, report: SecurityReport) -> str:
        """转换为HTML"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>安全扫描报告 - {report.project_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 1px solid #ccc; padding-bottom: 5px; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
        .critical {{ color: #d32f2f; font-weight: bold; }}
        .high {{ color: #f57c00; font-weight: bold; }}
        .medium {{ color: #fbc02d; }}
        .low {{ color: #388e3c; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        .recommendation {{ background: #e8f5e9; padding: 10px; margin: 5px 0; border-radius: 3px; }}
    </style>
</head>
<body>
    <h1>安全扫描报告: {report.project_name}</h1>
    <p>扫描时间: {report.scan_time}</p>
    
    <div class="summary">
        <h2>摘要</h2>
        <p>总问题数: {report.summary['total_issues']}</p>
        <p>总漏洞数: {report.summary['total_vulnerabilities']}</p>
        <p>危险导入数: {report.summary['total_dangerous_imports']}</p>
        <p class="critical">严重问题: {report.summary['critical_total']}</p>
        <p class="high">高危问题: {report.summary['high_total']}</p>
    </div>
"""
        
        if report.issues:
            html += "<h2>代码安全问题</h2><table><tr><th>类型</th><th>严重程度</th><th>文件</th><th>行号</th><th>描述</th></tr>"
            for issue in report.issues[:100]:
                html += f'<tr><td>{issue["type"]}</td><td class="{issue["severity"]}">{issue["severity"]}</td><td>{issue["file"]}</td><td>{issue["line"]}</td><td>{issue["message"]}</td></tr>'
            html += "</table>"
        
        if report.vulnerabilities:
            html += "<h2>依赖漏洞</h2><table><tr><th>CVE</th><th>包名</th><th>严重程度</th><th>描述</th></tr>"
            for vuln in report.vulnerabilities[:100]:
                html += f'<tr><td>{vuln["cve_id"]}</td><td>{vuln["package"]}</td><td class="{vuln["severity"]}">{vuln["severity"]}</td><td>{vuln["description"]}</td></tr>'
            html += "</table>"
        
        if report.recommendations:
            html += "<h2>修复建议</h2>"
            for rec in report.recommendations:
                html += f'<div class="recommendation">{rec}</div>'
        
        html += "</body></html>"
        return html
