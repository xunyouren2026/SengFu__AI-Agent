"""
辩论记录器模块
结构化存储全过程用于审计和学习
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, TextIO
from enum import Enum
from uuid import uuid4
import json
import csv
from collections import defaultdict
from pathlib import Path

from .protocol import (
    Argument, Rebuttal, Revision, Verdict, Evidence,
    DebateState, DebatePhase, Stance, ArgumentType
)


class LogFormat(Enum):
    """日志格式枚举"""
    JSON = "json"
    CSV = "csv"
    TEXT = "text"
    STRUCTURED = "structured"


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class LogEntry:
    """日志条目"""
    entry_id: str = field(default_factory=lambda: str(uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    level: LogLevel = LogLevel.INFO
    event_type: str = ""
    debate_id: str = ""
    phase: str = ""
    participant_id: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "event_type": self.event_type,
            "debate_id": self.debate_id,
            "phase": self.phase,
            "participant_id": self.participant_id,
            "content": self.content,
            "metadata": self.metadata,
        }


@dataclass
class DebateSummary:
    """辩论摘要"""
    debate_id: str
    topic: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: float
    total_participants: int
    total_arguments: int
    total_rebuttals: int
    total_revisions: int
    final_phase: str
    verdict: Optional[str]
    consensus_reached: bool
    
    # 统计信息
    stance_distribution: Dict[str, float] = field(default_factory=dict)
    argument_type_distribution: Dict[str, int] = field(default_factory=dict)
    participant_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "debate_id": self.debate_id,
            "topic": self.topic,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "total_participants": self.total_participants,
            "total_arguments": self.total_arguments,
            "total_rebuttals": self.total_rebuttals,
            "total_revisions": self.total_revisions,
            "final_phase": self.final_phase,
            "verdict": self.verdict,
            "consensus_reached": self.consensus_reached,
            "stance_distribution": self.stance_distribution,
            "argument_type_distribution": self.argument_type_distribution,
            "participant_stats": self.participant_stats,
        }


@dataclass
class ArgumentRecord:
    """论点记录"""
    argument_id: str
    speaker_id: str
    content: str
    stance: str
    argument_type: str
    timestamp: datetime
    phase: str
    evidence_count: int
    confidence: float
    target_argument_id: Optional[str]
    rebuttal_count: int = 0
    quality_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "argument_id": self.argument_id,
            "speaker_id": self.speaker_id,
            "content": self.content,
            "stance": self.stance,
            "argument_type": self.argument_type,
            "timestamp": self.timestamp.isoformat(),
            "phase": self.phase,
            "evidence_count": self.evidence_count,
            "confidence": self.confidence,
            "target_argument_id": self.target_argument_id,
            "rebuttal_count": self.rebuttal_count,
            "quality_score": self.quality_score,
        }


class EventLogger:
    """
    事件记录器
    记录辩论过程中的各类事件
    """
    
    def __init__(self, max_entries: int = 10000) -> None:
        self.max_entries = max_entries
        self.entries: List[LogEntry] = []
        self.entry_index: Dict[str, LogEntry] = {}
    
    def log(
        self,
        event_type: str,
        debate_id: str,
        phase: str = "",
        participant_id: str = "",
        level: LogLevel = LogLevel.INFO,
        content: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LogEntry:
        """
        记录事件
        
        Args:
            event_type: 事件类型
            debate_id: 辩论ID
            phase: 当前阶段
            participant_id: 参与者ID
            level: 日志级别
            content: 事件内容
            metadata: 元数据
            
        Returns:
            创建的日志条目
        """
        entry = LogEntry(
            level=level,
            event_type=event_type,
            debate_id=debate_id,
            phase=phase,
            participant_id=participant_id,
            content=content or {},
            metadata=metadata or {}
        )
        
        # 添加到列表
        self.entries.append(entry)
        self.entry_index[entry.entry_id] = entry
        
        # 限制最大条目数
        if len(self.entries) > self.max_entries:
            removed = self.entries.pop(0)
            self.entry_index.pop(removed.entry_id, None)
        
        return entry
    
    def log_argument(self, argument: Argument, debate_id: str) -> LogEntry:
        """记录论点事件"""
        return self.log(
            event_type="argument_submitted",
            debate_id=debate_id,
            phase=argument.phase.name,
            participant_id=argument.speaker_id,
            content={
                "argument_id": argument.argument_id,
                "content": argument.content,
                "stance": argument.stance.name,
                "argument_type": argument.argument_type.name,
                "confidence": argument.confidence,
                "evidence_count": len(argument.evidence_list),
            }
        )
    
    def log_rebuttal(self, rebuttal: Rebuttal, debate_id: str) -> LogEntry:
        """记录反驳事件"""
        return self.log(
            event_type="rebuttal_submitted",
            debate_id=debate_id,
            phase="REBUTTAL",
            participant_id=rebuttal.speaker_id,
            content={
                "rebuttal_id": rebuttal.rebuttal_id,
                "target_argument_id": rebuttal.target_argument_id,
                "content": rebuttal.content,
                "rebuttal_type": rebuttal.rebuttal_type,
                "effectiveness": rebuttal.effectiveness_score,
            }
        )
    
    def log_revision(self, revision: Revision, debate_id: str) -> LogEntry:
        """记录修正事件"""
        return self.log(
            event_type="argument_revised",
            debate_id=debate_id,
            phase="REVISION",
            participant_id=revision.speaker_id,
            content={
                "revision_id": revision.revision_id,
                "original_argument_id": revision.original_argument_id,
                "revised_content": revision.revised_content,
                "reason": revision.reason,
                "accepted": revision.accepted,
            }
        )
    
    def log_verdict(self, verdict: Verdict, debate_id: str) -> LogEntry:
        """记录裁决事件"""
        return self.log(
            event_type="verdict_issued",
            debate_id=debate_id,
            phase="VERDICT",
            participant_id=verdict.arbitrator_id,
            content={
                "verdict_id": verdict.verdict_id,
                "winning_stance": verdict.winning_stance.name if verdict.winning_stance else None,
                "reasoning": verdict.reasoning,
                "confidence": verdict.confidence,
            }
        )
    
    def log_phase_transition(
        self,
        debate_id: str,
        from_phase: str,
        to_phase: str
    ) -> LogEntry:
        """记录阶段转换"""
        return self.log(
            event_type="phase_transition",
            debate_id=debate_id,
            phase=to_phase,
            content={
                "from_phase": from_phase,
                "to_phase": to_phase,
            }
        )
    
    def get_entries_by_debate(self, debate_id: str) -> List[LogEntry]:
        """获取特定辩论的所有条目"""
        return [e for e in self.entries if e.debate_id == debate_id]
    
    def get_entries_by_type(self, event_type: str) -> List[LogEntry]:
        """获取特定类型的所有条目"""
        return [e for e in self.entries if e.event_type == event_type]
    
    def get_entries_by_participant(
        self,
        participant_id: str
    ) -> List[LogEntry]:
        """获取特定参与者的所有条目"""
        return [e for e in self.entries if e.participant_id == participant_id]


class DebateTranscript:
    """
    辩论记录
    存储单场辩论的完整记录
    """
    
    def __init__(self, debate_id: str, topic: str = "") -> None:
        self.debate_id = debate_id
        self.topic = topic
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        
        self.argument_records: Dict[str, ArgumentRecord] = {}
        self.rebuttal_records: Dict[str, Dict[str, Any]] = {}
        self.revision_records: Dict[str, Dict[str, Any]] = {}
        self.evidence_records: Dict[str, Dict[str, Any]] = {}
        
        self.phase_history: List[Tuple[str, datetime]] = []
        self.participant_activity: Dict[str, List[str]] = defaultdict(list)
        
        self.verdict_record: Optional[Dict[str, Any]] = None
        self.quality_scores: Dict[str, float] = {}
    
    def record_argument(
        self,
        argument: Argument,
        quality_score: float = 0.0
    ) -> ArgumentRecord:
        """记录论点"""
        record = ArgumentRecord(
            argument_id=argument.argument_id,
            speaker_id=argument.speaker_id,
            content=argument.content,
            stance=argument.stance.name,
            argument_type=argument.argument_type.name,
            timestamp=argument.timestamp,
            phase=argument.phase.name,
            evidence_count=len(argument.evidence_list),
            confidence=argument.confidence,
            target_argument_id=argument.target_argument_id,
            quality_score=quality_score
        )
        
        self.argument_records[argument.argument_id] = record
        self.participant_activity[argument.speaker_id].append(
            f"argument:{argument.argument_id}"
        )
        
        # 记录证据
        for evidence in argument.evidence_list:
            self.record_evidence(evidence, argument.argument_id)
        
        return record
    
    def record_rebuttal(self, rebuttal: Rebuttal) -> None:
        """记录反驳"""
        record = {
            "rebuttal_id": rebuttal.rebuttal_id,
            "speaker_id": rebuttal.speaker_id,
            "target_argument_id": rebuttal.target_argument_id,
            "content": rebuttal.content,
            "rebuttal_type": rebuttal.rebuttal_type,
            "timestamp": rebuttal.timestamp.isoformat(),
            "effectiveness_score": rebuttal.effectiveness_score,
            "fallacies_detected": rebuttal.fallacies_detected,
        }
        
        self.rebuttal_records[rebuttal.rebuttal_id] = record
        self.participant_activity[rebuttal.speaker_id].append(
            f"rebuttal:{rebuttal.rebuttal_id}"
        )
        
        # 更新目标论点的反驳计数
        target_id = rebuttal.target_argument_id
        if target_id in self.argument_records:
            self.argument_records[target_id].rebuttal_count += 1
    
    def record_revision(self, revision: Revision) -> None:
        """记录修正"""
        record = {
            "revision_id": revision.revision_id,
            "speaker_id": revision.speaker_id,
            "original_argument_id": revision.original_argument_id,
            "revised_content": revision.revised_content,
            "reason": revision.reason,
            "timestamp": revision.timestamp.isoformat(),
            "accepted": revision.accepted,
        }
        
        self.revision_records[revision.revision_id] = record
        self.participant_activity[revision.speaker_id].append(
            f"revision:{revision.revision_id}"
        )
    
    def record_evidence(
        self,
        evidence: Evidence,
        argument_id: str
    ) -> None:
        """记录证据"""
        record = {
            "evidence_id": evidence.evidence_id,
            "source": evidence.source,
            "content": evidence.content,
            "credibility": evidence.credibility,
            "relevance": evidence.relevance,
            "timestamp": evidence.timestamp.isoformat(),
            "associated_argument_id": argument_id,
        }
        
        self.evidence_records[evidence.evidence_id] = record
    
    def record_verdict(self, verdict: Verdict) -> None:
        """记录裁决"""
        self.verdict_record = {
            "verdict_id": verdict.verdict_id,
            "arbitrator_id": verdict.arbitrator_id,
            "topic": verdict.topic,
            "winning_stance": verdict.winning_stance.name if verdict.winning_stance else None,
            "reasoning": verdict.reasoning,
            "confidence": verdict.confidence,
            "argument_scores": verdict.argument_scores,
            "timestamp": verdict.timestamp.isoformat(),
            "recommendations": verdict.recommendations,
        }
    
    def record_phase_transition(
        self,
        from_phase: str,
        to_phase: str
    ) -> None:
        """记录阶段转换"""
        self.phase_history.append((to_phase, datetime.now()))
    
    def set_quality_score(
        self,
        argument_id: str,
        score: float
    ) -> None:
        """设置论点质量分数"""
        self.quality_scores[argument_id] = score
        if argument_id in self.argument_records:
            self.argument_records[argument_id].quality_score = score
    
    def finalize(self) -> None:
        """完成记录"""
        self.end_time = datetime.now()
    
    def get_summary(self) -> DebateSummary:
        """获取辩论摘要"""
        # 计算时长
        if self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
        else:
            duration = (datetime.now() - self.start_time).total_seconds()
        
        # 立场分布
        stance_counts: Dict[str, int] = defaultdict(int)
        for record in self.argument_records.values():
            stance_counts[record.stance] += 1
        
        total_args = len(self.argument_records)
        stance_distribution = {
            stance: count / total_args if total_args > 0 else 0.0
            for stance, count in stance_counts.items()
        }
        
        # 论点类型分布
        type_counts: Dict[str, int] = defaultdict(int)
        for record in self.argument_records.values():
            type_counts[record.argument_type] += 1
        
        # 参与者统计
        participant_stats: Dict[str, Dict[str, Any]] = {}
        for p_id, activities in self.participant_activity.items():
            arg_count = sum(1 for a in activities if a.startswith("argument:"))
            reb_count = sum(1 for a in activities if a.startswith("rebuttal:"))
            participant_stats[p_id] = {
                "total_activities": len(activities),
                "arguments": arg_count,
                "rebuttals": reb_count,
            }
        
        return DebateSummary(
            debate_id=self.debate_id,
            topic=self.topic,
            start_time=self.start_time,
            end_time=self.end_time,
            duration_seconds=duration,
            total_participants=len(self.participant_activity),
            total_arguments=len(self.argument_records),
            total_rebuttals=len(self.rebuttal_records),
            total_revisions=len(self.revision_records),
            final_phase=self.phase_history[-1][0] if self.phase_history else "CLAIM",
            verdict=self.verdict_record.get("winning_stance") if self.verdict_record else None,
            consensus_reached=self.verdict_record is not None,
            stance_distribution=stance_distribution,
            argument_type_distribution=dict(type_counts),
            participant_stats=participant_stats,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "debate_id": self.debate_id,
            "topic": self.topic,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "arguments": {
                aid: record.to_dict()
                for aid, record in self.argument_records.items()
            },
            "rebuttals": self.rebuttal_records,
            "revisions": self.revision_records,
            "evidence": self.evidence_records,
            "phase_history": [
                {"phase": phase, "timestamp": ts.isoformat()}
                for phase, ts in self.phase_history
            ],
            "verdict": self.verdict_record,
            "quality_scores": self.quality_scores,
        }


class TranscriptExporter:
    """
    记录导出器
    支持多种格式导出
    """
    
    @staticmethod
    def to_json(
        transcript: DebateTranscript,
        indent: int = 2
    ) -> str:
        """导出为JSON格式"""
        return json.dumps(
            transcript.to_dict(),
            indent=indent,
            ensure_ascii=False
        )
    
    @staticmethod
    def to_json_file(
        transcript: DebateTranscript,
        filepath: str
    ) -> None:
        """导出到JSON文件"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(TranscriptExporter.to_json(transcript))
    
    @staticmethod
    def to_csv(
        transcript: DebateTranscript,
        record_type: str = "arguments"
    ) -> str:
        """
        导出为CSV格式
        
        Args:
            transcript: 辩论记录
            record_type: 记录类型 ("arguments", "rebuttals", "revisions")
        """
        import io
        
        output = io.StringIO()
        
        if record_type == "arguments":
            fieldnames = [
                "argument_id", "speaker_id", "content", "stance",
                "argument_type", "timestamp", "phase", "evidence_count",
                "confidence", "target_argument_id", "rebuttal_count", "quality_score"
            ]
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for record in transcript.argument_records.values():
                writer.writerow({
                    "argument_id": record.argument_id,
                    "speaker_id": record.speaker_id,
                    "content": record.content,
                    "stance": record.stance,
                    "argument_type": record.argument_type,
                    "timestamp": record.timestamp.isoformat(),
                    "phase": record.phase,
                    "evidence_count": record.evidence_count,
                    "confidence": record.confidence,
                    "target_argument_id": record.target_argument_id or "",
                    "rebuttal_count": record.rebuttal_count,
                    "quality_score": record.quality_score,
                })
        
        elif record_type == "rebuttals":
            fieldnames = [
                "rebuttal_id", "speaker_id", "target_argument_id",
                "content", "rebuttal_type", "timestamp", "effectiveness_score"
            ]
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for record in transcript.rebuttal_records.values():
                writer.writerow({
                    "rebuttal_id": record["rebuttal_id"],
                    "speaker_id": record["speaker_id"],
                    "target_argument_id": record["target_argument_id"],
                    "content": record["content"],
                    "rebuttal_type": record["rebuttal_type"],
                    "timestamp": record["timestamp"],
                    "effectiveness_score": record["effectiveness_score"],
                })
        
        return output.getvalue()
    
    @staticmethod
    def to_text(transcript: DebateTranscript) -> str:
        """导出为文本格式"""
        lines: List[str] = []
        
        # 标题
        lines.append("=" * 60)
        lines.append(f"辩论记录: {transcript.topic}")
        lines.append(f"辩论ID: {transcript.debate_id}")
        lines.append(f"开始时间: {transcript.start_time}")
        lines.append("=" * 60)
        lines.append("")
        
        # 阶段历史
        if transcript.phase_history:
            lines.append("【阶段流程】")
            for phase, ts in transcript.phase_history:
                lines.append(f"  {ts.strftime('%H:%M:%S')} - {phase}")
            lines.append("")
        
        # 论点
        lines.append("【论点记录】")
        for record in transcript.argument_records.values():
            lines.append(f"  [{record.argument_id}] {record.speaker_id}")
            lines.append(f"    立场: {record.stance} | 类型: {record.argument_type}")
            lines.append(f"    内容: {record.content[:100]}...")
            lines.append(f"    证据数: {record.evidence_count} | 置信度: {record.confidence:.2f}")
            lines.append("")
        
        # 反驳
        if transcript.rebuttal_records:
            lines.append("【反驳记录】")
            for record in transcript.rebuttal_records.values():
                lines.append(f"  [{record['rebuttal_id']}] {record['speaker_id']}")
                lines.append(f"    针对: {record['target_argument_id']}")
                lines.append(f"    内容: {record['content'][:100]}...")
                lines.append("")
        
        # 裁决
        if transcript.verdict_record:
            lines.append("【最终裁决】")
            lines.append(f"  胜方立场: {transcript.verdict_record['winning_stance']}")
            lines.append(f"  置信度: {transcript.verdict_record['confidence']:.2f}")
            lines.append(f"  理由: {transcript.verdict_record['reasoning'][:200]}...")
            lines.append("")
        
        # 摘要
        summary = transcript.get_summary()
        lines.append("【统计摘要】")
        lines.append(f"  总论点数: {summary.total_arguments}")
        lines.append(f"  总反驳数: {summary.total_rebuttals}")
        lines.append(f"  参与者数: {summary.total_participants}")
        lines.append(f"  持续时间: {summary.duration_seconds:.1f}秒")
        
        return "\n".join(lines)


class TranscriptLogger:
    """
    辩论记录器
    主类，协调所有记录功能
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        auto_export: bool = False
    ) -> None:
        """
        初始化记录器
        
        Args:
            output_dir: 输出目录
            auto_export: 是否自动导出
        """
        self.output_dir = output_dir
        self.auto_export = auto_export
        
        self.event_logger = EventLogger()
        self.active_transcripts: Dict[str, DebateTranscript] = {}
        self.completed_transcripts: Dict[str, DebateTranscript] = {}
    
    def start_debate(
        self,
        debate_id: str,
        topic: str
    ) -> DebateTranscript:
        """
        开始记录新辩论
        
        Args:
            debate_id: 辩论ID
            topic: 辩论主题
            
        Returns:
            新的辩论记录
        """
        transcript = DebateTranscript(debate_id=debate_id, topic=topic)
        self.active_transcripts[debate_id] = transcript
        
        self.event_logger.log(
            event_type="debate_started",
            debate_id=debate_id,
            content={"topic": topic}
        )
        
        return transcript
    
    def get_transcript(self, debate_id: str) -> Optional[DebateTranscript]:
        """获取辩论记录"""
        return self.active_transcripts.get(debate_id) or \
               self.completed_transcripts.get(debate_id)
    
    def log_argument(
        self,
        debate_id: str,
        argument: Argument,
        quality_score: float = 0.0
    ) -> Optional[ArgumentRecord]:
        """记录论点"""
        transcript = self.active_transcripts.get(debate_id)
        if not transcript:
            return None
        
        record = transcript.record_argument(argument, quality_score)
        self.event_logger.log_argument(argument, debate_id)
        
        return record
    
    def log_rebuttal(
        self,
        debate_id: str,
        rebuttal: Rebuttal
    ) -> None:
        """记录反驳"""
        transcript = self.active_transcripts.get(debate_id)
        if not transcript:
            return
        
        transcript.record_rebuttal(rebuttal)
        self.event_logger.log_rebuttal(rebuttal, debate_id)
    
    def log_revision(
        self,
        debate_id: str,
        revision: Revision
    ) -> None:
        """记录修正"""
        transcript = self.active_transcripts.get(debate_id)
        if not transcript:
            return
        
        transcript.record_revision(revision)
        self.event_logger.log_revision(revision, debate_id)
    
    def log_verdict(
        self,
        debate_id: str,
        verdict: Verdict
    ) -> None:
        """记录裁决"""
        transcript = self.active_transcripts.get(debate_id)
        if not transcript:
            return
        
        transcript.record_verdict(verdict)
        self.event_logger.log_verdict(verdict, debate_id)
    
    def log_phase_transition(
        self,
        debate_id: str,
        from_phase: str,
        to_phase: str
    ) -> None:
        """记录阶段转换"""
        transcript = self.active_transcripts.get(debate_id)
        if not transcript:
            return
        
        transcript.record_phase_transition(from_phase, to_phase)
        self.event_logger.log_phase_transition(debate_id, from_phase, to_phase)
    
    def end_debate(self, debate_id: str) -> Optional[DebateTranscript]:
        """
        结束辩论记录
        
        Args:
            debate_id: 辩论ID
            
        Returns:
            完成的辩论记录
        """
        transcript = self.active_transcripts.pop(debate_id, None)
        if not transcript:
            return None
        
        transcript.finalize()
        self.completed_transcripts[debate_id] = transcript
        
        self.event_logger.log(
            event_type="debate_ended",
            debate_id=debate_id,
            content={"duration": transcript.get_summary().duration_seconds}
        )
        
        # 自动导出
        if self.auto_export and self.output_dir:
            self.export_transcript(transcript)
        
        return transcript
    
    def export_transcript(
        self,
        transcript: DebateTranscript,
        format: LogFormat = LogFormat.JSON,
        filepath: Optional[str] = None
    ) -> str:
        """
        导出辩论记录
        
        Args:
            transcript: 辩论记录
            format: 导出格式
            filepath: 文件路径（可选）
            
        Returns:
            导出内容或文件路径
        """
        if filepath is None and self.output_dir:
            base_name = f"debate_{transcript.debate_id}"
            ext = ".json" if format == LogFormat.JSON else \
                  ".csv" if format == LogFormat.CSV else ".txt"
            filepath = str(Path(self.output_dir) / f"{base_name}{ext}")
        
        if format == LogFormat.JSON:
            if filepath:
                TranscriptExporter.to_json_file(transcript, filepath)
                return filepath
            return TranscriptExporter.to_json(transcript)
        
        elif format == LogFormat.CSV:
            if filepath:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(TranscriptExporter.to_csv(transcript))
                return filepath
            return TranscriptExporter.to_csv(transcript)
        
        elif format == LogFormat.TEXT:
            content = TranscriptExporter.to_text(transcript)
            if filepath:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                return filepath
            return content
        
        return ""
    
    def get_debate_summary(
        self,
        debate_id: str
    ) -> Optional[DebateSummary]:
        """获取辩论摘要"""
        transcript = self.get_transcript(debate_id)
        if transcript:
            return transcript.get_summary()
        return None
    
    def get_participant_history(
        self,
        participant_id: str
    ) -> List[Dict[str, Any]]:
        """
        获取参与者的历史记录
        
        Args:
            participant_id: 参与者ID
            
        Returns:
            参与者参与的辩论列表
        """
        history = []
        
        for transcript in list(self.completed_transcripts.values()) + \
                         list(self.active_transcripts.values()):
            if participant_id in transcript.participant_activity:
                summary = transcript.get_summary()
                stats = summary.participant_stats.get(participant_id, {})
                history.append({
                    "debate_id": transcript.debate_id,
                    "topic": transcript.topic,
                    "stats": stats,
                    "date": transcript.start_time.isoformat(),
                })
        
        return history
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取记录器统计信息"""
        return {
            "active_debates": len(self.active_transcripts),
            "completed_debates": len(self.completed_transcripts),
            "total_events": len(self.event_logger.entries),
            "total_arguments": sum(
                len(t.argument_records)
                for t in list(self.completed_transcripts.values()) +
                         list(self.active_transcripts.values())
            ),
            "total_rebuttals": sum(
                len(t.rebuttal_records)
                for t in list(self.completed_transcripts.values()) +
                         list(self.active_transcripts.values())
            ),
        }
    
    def search_events(
        self,
        event_type: Optional[str] = None,
        debate_id: Optional[str] = None,
        participant_id: Optional[str] = None,
        level: Optional[LogLevel] = None
    ) -> List[LogEntry]:
        """
        搜索事件
        
        Args:
            event_type: 事件类型
            debate_id: 辩论ID
            participant_id: 参与者ID
            level: 日志级别
            
        Returns:
            匹配的事件列表
        """
        results = self.event_logger.entries
        
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        if debate_id:
            results = [e for e in results if e.debate_id == debate_id]
        if participant_id:
            results = [e for e in results if e.participant_id == participant_id]
        if level:
            results = [e for e in results if e.level == level]
        
        return results


__all__ = [
    "LogFormat",
    "LogLevel",
    "LogEntry",
    "DebateSummary",
    "ArgumentRecord",
    "EventLogger",
    "DebateTranscript",
    "TranscriptExporter",
    "TranscriptLogger",
]
