"""
蜜罐取证 - 蜜罐攻击取证分析
"""
import time
import json
import hashlib
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class AttackPhase(Enum):
    """攻击阶段"""
    RECONNAISSANCE = "reconnaissance"
    INITIAL_ACCESS = "initial_access"
    EXECUTION = "execution"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DEFENSE_EVASION = "defense_evasion"
    CREDENTIAL_ACCESS = "credential_access"
    DISCOVERY = "discovery"
    LATERAL_MOVEMENT = "lateral_movement"
    COLLECTION = "collection"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"


@dataclass
class ForensicEvidence:
    """取证证据"""
    evidence_id: str
    evidence_type: str
    timestamp: float
    source: str
    data: Any
    hash: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.hash:
            self.hash = hashlib.sha256(str(self.data).encode()).hexdigest()[:16]


@dataclass
class AttackSession:
    """攻击会话"""
    session_id: str
    source_ip: str
    start_time: float
    end_time: Optional[float] = None
    attack_phases: List[AttackPhase] = field(default_factory=list)
    evidence: List[ForensicEvidence] = field(default_factory=list)
    commands_executed: List[str] = field(default_factory=list)
    files_accessed: List[str] = field(default_factory=list)
    network_connections: List[Dict[str, Any]] = field(default_factory=list)
    techniques: List[str] = field(default_factory=list)


class ForensicsAnalyzer:
    """取证分析器"""
    
    def __init__(self):
        self._sessions: Dict[str, AttackSession] = {}
        self._evidence_store: Dict[str, ForensicEvidence] = {}
        self._attack_patterns = self._load_attack_patterns()
    
    def _load_attack_patterns(self) -> Dict[str, List[str]]:
        """加载攻击模式"""
        return {
            "reconnaissance": [
                "whoami", "id", "uname", "hostname", "ifconfig", "ip addr",
                "netstat", "ps aux", "ls -la", "cat /etc/passwd"
            ],
            "privilege_escalation": [
                "sudo", "su", "chmod 777", "chown root",
                "/etc/sudoers", "SUID", "GTFOBins"
            ],
            "persistence": [
                "crontab", "systemctl enable", "/etc/rc.local",
                ".bashrc", "ssh key", "backdoor"
            ],
            "exfiltration": [
                "curl", "wget", "nc", "scp", "rsync",
                "base64", "xxd", "tar"
            ]
        }
    
    def create_session(self, source_ip: str) -> AttackSession:
        """创建攻击会话"""
        session_id = hashlib.md5(f"{source_ip}{time.time()}".encode()).hexdigest()[:12]
        
        session = AttackSession(
            session_id=session_id,
            source_ip=source_ip,
            start_time=time.time()
        )
        
        self._sessions[session_id] = session
        return session
    
    def record_command(
        self,
        session_id: str,
        command: str,
        output: str = ""
    ) -> ForensicEvidence:
        """记录执行的命令"""
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"会话不存在: {session_id}")
        
        # 记录命令
        session.commands_executed.append(command)
        
        # 分析攻击阶段
        phase = self._analyze_command(command)
        if phase and phase not in session.attack_phases:
            session.attack_phases.append(phase)
        
        # 创建证据
        evidence = ForensicEvidence(
            evidence_id=hashlib.md5(f"{command}{time.time()}".encode()).hexdigest()[:12],
            evidence_type="command",
            timestamp=time.time(),
            source=session.source_ip,
            data={"command": command, "output": output[:1000]},
            tags=[phase.value] if phase else []
        )
        
        session.evidence.append(evidence)
        self._evidence_store[evidence.evidence_id] = evidence
        
        return evidence
    
    def record_file_access(
        self,
        session_id: str,
        file_path: str,
        action: str,
        content_hash: str = ""
    ) -> ForensicEvidence:
        """记录文件访问"""
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"会话不存在: {session_id}")
        
        session.files_accessed.append(f"{action}:{file_path}")
        
        evidence = ForensicEvidence(
            evidence_id=hashlib.md5(f"{file_path}{time.time()}".encode()).hexdigest()[:12],
            evidence_type="file_access",
            timestamp=time.time(),
            source=session.source_ip,
            data={"path": file_path, "action": action, "content_hash": content_hash},
            tags=["sensitive"] if self._is_sensitive_file(file_path) else []
        )
        
        session.evidence.append(evidence)
        self._evidence_store[evidence.evidence_id] = evidence
        
        return evidence
    
    def record_network_connection(
        self,
        session_id: str,
        dest_ip: str,
        dest_port: int,
        protocol: str
    ) -> ForensicEvidence:
        """记录网络连接"""
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"会话不存在: {session_id}")
        
        conn = {"dest_ip": dest_ip, "dest_port": dest_port, "protocol": protocol}
        session.network_connections.append(conn)
        
        evidence = ForensicEvidence(
            evidence_id=hashlib.md5(f"{dest_ip}{dest_port}{time.time()}".encode()).hexdigest()[:12],
            evidence_type="network",
            timestamp=time.time(),
            source=session.source_ip,
            data=conn,
            tags=["exfiltration"] if self._is_exfiltration(dest_port) else []
        )
        
        session.evidence.append(evidence)
        self._evidence_store[evidence.evidence_id] = evidence
        
        return evidence
    
    def _analyze_command(self, command: str) -> Optional[AttackPhase]:
        """分析命令识别攻击阶段"""
        command_lower = command.lower()
        
        for phase_name, patterns in self._attack_patterns.items():
            for pattern in patterns:
                if pattern.lower() in command_lower:
                    return AttackPhase(phase_name)
        
        return None
    
    def _is_sensitive_file(self, file_path: str) -> bool:
        """检查是否为敏感文件"""
        sensitive_paths = [
            "/etc/passwd", "/etc/shadow", "/etc/sudoers",
            ".ssh/", ".gnupg/", "id_rsa", "credentials",
            "password", "secret", "config"
        ]
        
        return any(s in file_path.lower() for s in sensitive_paths)
    
    def _is_exfiltration(self, port: int) -> bool:
        """检查是否为数据渗出"""
        exfil_ports = {4444, 5555, 6666, 31337, 1234}
        return port in exfil_ports or port > 49152
    
    def close_session(self, session_id: str) -> AttackSession:
        """关闭会话"""
        session = self._sessions.get(session_id)
        if session:
            session.end_time = time.time()
        return session
    
    def get_session(self, session_id: str) -> Optional[AttackSession]:
        """获取会话"""
        return self._sessions.get(session_id)
    
    def get_all_sessions(self) -> List[AttackSession]:
        """获取所有会话"""
        return list(self._sessions.values())
    
    def get_active_sessions(self) -> List[AttackSession]:
        """获取活动会话"""
        return [s for s in self._sessions.values() if s.end_time is None]
    
    def generate_report(self, session_id: str) -> Dict[str, Any]:
        """生成取证报告"""
        session = self._sessions.get(session_id)
        if not session:
            return {}
        
        return {
            "session_id": session.session_id,
            "source_ip": session.source_ip,
            "duration": (session.end_time or time.time()) - session.start_time,
            "attack_phases": [p.value for p in session.attack_phases],
            "commands_count": len(session.commands_executed),
            "files_accessed_count": len(session.files_accessed),
            "network_connections_count": len(session.network_connections),
            "evidence_count": len(session.evidence),
            "timeline": [
                {
                    "timestamp": e.timestamp,
                    "type": e.evidence_type,
                    "data": e.data
                }
                for e in sorted(session.evidence, key=lambda x: x.timestamp)
            ]
        }
    
    def export_evidence(self, session_id: str) -> str:
        """导出证据"""
        session = self._sessions.get(session_id)
        if not session:
            return ""
        
        return json.dumps({
            "session": {
                "session_id": session.session_id,
                "source_ip": session.source_ip,
                "start_time": session.start_time,
                "end_time": session.end_time
            },
            "evidence": [
                {
                    "evidence_id": e.evidence_id,
                    "type": e.evidence_type,
                    "timestamp": e.timestamp,
                    "data": e.data,
                    "hash": e.hash
                }
                for e in session.evidence
            ]
        }, indent=2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_sessions": len(self._sessions),
            "active_sessions": len(self.get_active_sessions()),
            "total_evidence": len(self._evidence_store),
            "attack_phases": {
                phase.value: sum(
                    1 for s in self._sessions.values()
                    if phase in s.attack_phases
                )
                for phase in AttackPhase
            }
        }
