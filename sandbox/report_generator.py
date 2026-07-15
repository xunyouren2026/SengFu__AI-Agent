"""
报告生成器
生成执行报告，支持多种格式
"""

import os
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum


class ReportFormat(Enum):
    """报告格式"""
    JSON = "json"
    TEXT = "text"
    HTML = "html"
    MARKDOWN = "markdown"
    CSV = "csv"


@dataclass
class ReportSection:
    """报告章节"""
    title: str
    content: Any
    order: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'content': self.content,
            'order': self.order
        }


@dataclass
class ExecutionReport:
    """执行报告"""
    report_id: str
    title: str
    timestamp: float
    execution_id: str
    status: str
    duration: float
    sections: List[ReportSection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_section(self, title: str, content: Any, order: int = 0) -> None:
        """添加章节"""
        self.sections.append(ReportSection(title=title, content=content, order=order))
        self.sections.sort(key=lambda s: s.order)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'report_id': self.report_id,
            'title': self.title,
            'timestamp': self.timestamp,
            'timestamp_iso': datetime.fromtimestamp(self.timestamp).isoformat(),
            'execution_id': self.execution_id,
            'status': self.status,
            'duration': self.duration,
            'sections': [s.to_dict() for s in self.sections],
            'metadata': self.metadata
        }


class ReportGenerator:
    """
    报告生成器
    生成各种格式的执行报告
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or tempfile.gettempdir()
        self._templates: Dict[str, str] = {}
    
    def generate(
        self,
        execution_result: Any,
        format: ReportFormat = ReportFormat.JSON,
        title: Optional[str] = None,
        include_sections: Optional[List[str]] = None
    ) -> str:
        """
        生成报告
        
        Args:
            execution_result: 执行结果
            format: 报告格式
            title: 报告标题
            include_sections: 包含的章节
            
        Returns:
            报告内容
        """
        # 创建报告对象
        report = self._create_report(execution_result, title)
        
        # 添加章节
        self._add_sections(report, execution_result, include_sections)
        
        # 根据格式生成
        if format == ReportFormat.JSON:
            return self._to_json(report)
        elif format == ReportFormat.TEXT:
            return self._to_text(report)
        elif format == ReportFormat.HTML:
            return self._to_html(report)
        elif format == ReportFormat.MARKDOWN:
            return self._to_markdown(report)
        elif format == ReportFormat.CSV:
            return self._to_csv(report)
        else:
            return self._to_json(report)
    
    def save(
        self,
        execution_result: Any,
        filepath: str,
        format: Optional[ReportFormat] = None,
        **kwargs
    ) -> bool:
        """
        保存报告到文件
        
        Args:
            execution_result: 执行结果
            filepath: 文件路径
            format: 报告格式（None则根据扩展名推断）
            
        Returns:
            是否成功
        """
        # 推断格式
        if format is None:
            ext = Path(filepath).suffix.lower()
            format_map = {
                '.json': ReportFormat.JSON,
                '.txt': ReportFormat.TEXT,
                '.html': ReportFormat.HTML,
                '.md': ReportFormat.MARKDOWN,
                '.csv': ReportFormat.CSV
            }
            format = format_map.get(ext, ReportFormat.JSON)
        
        content = self.generate(execution_result, format, **kwargs)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except IOError:
            return False
    
    def _create_report(self, result: Any, title: Optional[str]) -> ExecutionReport:
        """创建报告对象"""
        import uuid
        
        # 从结果提取基本信息
        if hasattr(result, 'execution_id'):
            execution_id = result.execution_id
        else:
            execution_id = 'unknown'
        
        if hasattr(result, 'status'):
            status = result.status.value if hasattr(result.status, 'value') else str(result.status)
        else:
            status = 'unknown'
        
        if hasattr(result, 'duration'):
            duration = result.duration
        else:
            duration = 0.0
        
        return ExecutionReport(
            report_id=str(uuid.uuid4()),
            title=title or f"Execution Report: {execution_id}",
            timestamp=time.time(),
            execution_id=execution_id,
            status=status,
            duration=duration
        )
    
    def _add_sections(
        self,
        report: ExecutionReport,
        result: Any,
        include_sections: Optional[List[str]]
    ) -> None:
        """添加章节"""
        sections = include_sections or ['summary', 'output', 'resources', 'files']
        
        if 'summary' in sections:
            report.add_section(
                title="Summary",
                content=self._get_summary_content(result),
                order=1
            )
        
        if 'output' in sections:
            report.add_section(
                title="Output",
                content=self._get_output_content(result),
                order=2
            )
        
        if 'resources' in sections:
            report.add_section(
                title="Resource Usage",
                content=self._get_resource_content(result),
                order=3
            )
        
        if 'files' in sections:
            report.add_section(
                title="Output Files",
                content=self._get_files_content(result),
                order=4
            )
        
        if 'errors' in sections:
            report.add_section(
                title="Errors and Warnings",
                content=self._get_errors_content(result),
                order=5
            )
    
    def _get_summary_content(self, result: Any) -> Dict[str, Any]:
        """获取摘要内容"""
        content = {
            'execution_id': getattr(result, 'execution_id', 'unknown'),
            'status': getattr(result, 'status', 'unknown'),
            'exit_code': getattr(result, 'exit_code', 0),
            'duration': getattr(result, 'duration', 0),
            'start_time': getattr(result, 'start_time', 0),
            'end_time': getattr(result, 'end_time', 0)
        }
        
        if content['start_time']:
            content['start_time_iso'] = datetime.fromtimestamp(content['start_time']).isoformat()
        if content['end_time']:
            content['end_time_iso'] = datetime.fromtimestamp(content['end_time']).isoformat()
        
        return content
    
    def _get_output_content(self, result: Any) -> Dict[str, Any]:
        """获取输出内容"""
        stdout = getattr(result, 'stdout', '')
        stderr = getattr(result, 'stderr', '')
        
        return {
            'stdout': stdout,
            'stdout_lines': len(stdout.split('\n')) if stdout else 0,
            'stderr': stderr,
            'stderr_lines': len(stderr.split('\n')) if stderr else 0
        }
    
    def _get_resource_content(self, result: Any) -> Dict[str, Any]:
        """获取资源使用内容"""
        return getattr(result, 'resource_usage', {}) or {}
    
    def _get_files_content(self, result: Any) -> Dict[str, Any]:
        """获取输出文件内容"""
        output_files = getattr(result, 'output_files', {}) or {}
        
        return {
            'count': len(output_files),
            'files': list(output_files.keys()),
            'sizes': {
                name: len(content) if isinstance(content, str) else 0
                for name, content in output_files.items()
            }
        }
    
    def _get_errors_content(self, result: Any) -> Dict[str, Any]:
        """获取错误和警告内容"""
        return {
            'error_message': getattr(result, 'error_message', None),
            'warnings': getattr(result, 'warnings', []) or []
        }
    
    def _to_json(self, report: ExecutionReport) -> str:
        """转换为JSON"""
        return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
    
    def _to_text(self, report: ExecutionReport) -> str:
        """转换为文本"""
        lines = []
        
        # 标题
        lines.append("=" * 70)
        lines.append(report.title)
        lines.append("=" * 70)
        lines.append("")
        
        # 基本信息
        lines.append(f"Report ID: {report.report_id}")
        lines.append(f"Execution ID: {report.execution_id}")
        lines.append(f"Status: {report.status}")
        lines.append(f"Duration: {report.duration:.3f}s")
        lines.append(f"Timestamp: {datetime.fromtimestamp(report.timestamp).isoformat()}")
        lines.append("")
        
        # 章节
        for section in report.sections:
            lines.append("-" * 70)
            lines.append(section.title)
            lines.append("-" * 70)
            
            content = section.content
            if isinstance(content, dict):
                for key, value in content.items():
                    if isinstance(value, str) and len(value) > 200:
                        value = value[:200] + "..."
                    lines.append(f"{key}: {value}")
            elif isinstance(content, list):
                for item in content:
                    lines.append(f"  - {item}")
            else:
                lines.append(str(content))
            
            lines.append("")
        
        return '\n'.join(lines)
    
    def _to_html(self, report: ExecutionReport) -> str:
        """转换为HTML"""
        html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{report.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }}
        h2 {{ color: #666; margin-top: 30px; }}
        .info {{ background: #f5f5f5; padding: 10px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        pre {{ background: #f8f8f8; padding: 10px; overflow-x: auto; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        .status-success {{ color: green; }}
        .status-failed {{ color: red; }}
        .status-timeout {{ color: orange; }}
    </style>
</head>
<body>
    <h1>{report.title}</h1>
    
    <div class="info">
        <p><strong>Report ID:</strong> {report.report_id}</p>
        <p><strong>Execution ID:</strong> {report.execution_id}</p>
        <p><strong>Status:</strong> <span class="status-{report.status}">{report.status}</span></p>
        <p><strong>Duration:</strong> {report.duration:.3f}s</p>
        <p><strong>Timestamp:</strong> {datetime.fromtimestamp(report.timestamp).isoformat()}</p>
    </div>
'''
        
        for section in report.sections:
            html += f'    <div class="section">\n'
            html += f'        <h2>{section.title}</h2>\n'
            
            content = section.content
            if isinstance(content, dict):
                html += '        <table>\n'
                for key, value in content.items():
                    if isinstance(value, str) and len(value) > 500:
                        value = value[:500] + "..."
                    if isinstance(value, str) and '\n' in value:
                        value = f'<pre>{value}</pre>'
                    html += f'            <tr><th>{key}</th><td>{value}</td></tr>\n'
                html += '        </table>\n'
            elif isinstance(content, list):
                html += '        <ul>\n'
                for item in content:
                    html += f'            <li>{item}</li>\n'
                html += '        </ul>\n'
            else:
                html += f'        <pre>{content}</pre>\n'
            
            html += '    </div>\n'
        
        html += '''</body>
</html>'''
        
        return html
    
    def _to_markdown(self, report: ExecutionReport) -> str:
        """转换为Markdown"""
        lines = []
        
        lines.append(f"# {report.title}")
        lines.append("")
        
        lines.append("## Basic Information")
        lines.append("")
        lines.append(f"- **Report ID**: {report.report_id}")
        lines.append(f"- **Execution ID**: {report.execution_id}")
        lines.append(f"- **Status**: {report.status}")
        lines.append(f"- **Duration**: {report.duration:.3f}s")
        lines.append(f"- **Timestamp**: {datetime.fromtimestamp(report.timestamp).isoformat()}")
        lines.append("")
        
        for section in report.sections:
            lines.append(f"## {section.title}")
            lines.append("")
            
            content = section.content
            if isinstance(content, dict):
                lines.append("| Key | Value |")
                lines.append("|-----|-------|")
                for key, value in content.items():
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    value = str(value).replace('\n', ' ').replace('|', '\\|')
                    lines.append(f"| {key} | {value} |")
            elif isinstance(content, list):
                for item in content:
                    lines.append(f"- {item}")
            else:
                lines.append(f"```\n{content}\n```")
            
            lines.append("")
        
        return '\n'.join(lines)
    
    def _to_csv(self, report: ExecutionReport) -> str:
        """转换为CSV"""
        lines = []
        
        # 标题行
        lines.append("section,key,value")
        
        # 基本信息
        lines.append(f"info,report_id,{report.report_id}")
        lines.append(f"info,execution_id,{report.execution_id}")
        lines.append(f"info,status,{report.status}")
        lines.append(f"info,duration,{report.duration}")
        lines.append(f"info,timestamp,{report.timestamp}")
        
        # 章节内容
        for section in report.sections:
            content = section.content
            if isinstance(content, dict):
                for key, value in content.items():
                    if isinstance(value, str):
                        value = value.replace('"', '""')
                        lines.append(f'"{section.title}","{key}","{value}"')
                    else:
                        lines.append(f'"{section.title}","{key}",{value}')
        
        return '\n'.join(lines)


class BatchReportGenerator:
    """
    批量报告生成器
    生成聚合报告
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or tempfile.gettempdir()
        self._generator = ReportGenerator(output_dir)
    
    def generate_batch_report(
        self,
        results: List[Any],
        format: ReportFormat = ReportFormat.JSON,
        title: Optional[str] = None
    ) -> str:
        """
        生成批量报告
        
        Args:
            results: 结果列表
            format: 报告格式
            title: 报告标题
            
        Returns:
            报告内容
        """
        import uuid
        
        # 计算聚合统计
        total = len(results)
        successful = sum(1 for r in results if getattr(r, 'status', None) and 
                        (hasattr(r.status, 'value') and r.status.value == 'success' or str(r.status) == 'success'))
        failed = total - successful
        
        total_duration = sum(getattr(r, 'duration', 0) for r in results)
        avg_duration = total_duration / total if total > 0 else 0
        
        report_data = {
            'report_id': str(uuid.uuid4()),
            'title': title or "Batch Execution Report",
            'timestamp': time.time(),
            'timestamp_iso': datetime.now().isoformat(),
            'summary': {
                'total': total,
                'successful': successful,
                'failed': failed,
                'success_rate': successful / total if total > 0 else 0,
                'total_duration': total_duration,
                'average_duration': avg_duration
            },
            'results': []
        }
        
        # 添加每个结果
        for result in results:
            if hasattr(result, 'to_dict'):
                report_data['results'].append(result.to_dict())
            else:
                report_data['results'].append({
                    'execution_id': getattr(result, 'execution_id', 'unknown'),
                    'status': str(getattr(result, 'status', 'unknown')),
                    'duration': getattr(result, 'duration', 0)
                })
        
        if format == ReportFormat.JSON:
            return json.dumps(report_data, indent=2, ensure_ascii=False)
        elif format == ReportFormat.MARKDOWN:
            return self._batch_to_markdown(report_data)
        else:
            return json.dumps(report_data, indent=2)
    
    def _batch_to_markdown(self, data: Dict[str, Any]) -> str:
        """批量报告转Markdown"""
        lines = []
        
        lines.append(f"# {data['title']}")
        lines.append("")
        lines.append(f"Generated at: {data['timestamp_iso']}")
        lines.append("")
        
        summary = data['summary']
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- Total executions: {summary['total']}")
        lines.append(f"- Successful: {summary['successful']}")
        lines.append(f"- Failed: {summary['failed']}")
        lines.append(f"- Success rate: {summary['success_rate']:.2%}")
        lines.append(f"- Total duration: {summary['total_duration']:.3f}s")
        lines.append(f"- Average duration: {summary['average_duration']:.3f}s")
        lines.append("")
        
        lines.append("## Results")
        lines.append("")
        lines.append("| Execution ID | Status | Duration |")
        lines.append("|--------------|--------|----------|")
        
        for result in data['results']:
            lines.append(f"| {result.get('execution_id', 'unknown')} | {result.get('status', 'unknown')} | {result.get('duration', 0):.3f}s |")
        
        return '\n'.join(lines)


import tempfile
