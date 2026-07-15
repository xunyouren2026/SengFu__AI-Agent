"""
Audit Analyzer Module

提供审计分析功能，包括用户行为分析、威胁检测和风险评分。
"""

import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple


class BehaviorProfile:
    """
    行为画像类
    
    描述用户的正常行为模式和检测到的异常。
    """
    
    def __init__(
        self,
        user_id: str,
        normal_patterns: Optional[Dict[str, Any]] = None,
        anomalies: Optional[List[Dict[str, Any]]] = None,
        risk_score: float = 0.0
    ):
        self.user_id = user_id
        self.normal_patterns = normal_patterns or {}
        self.anomalies = anomalies or []
        self.risk_score = risk_score
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.baseline_established = False
    
    def add_anomaly(self, anomaly_type: str, description: str, severity: str) -> None:
        """添加异常记录"""
        self.anomalies.append({
            "type": anomaly_type,
            "description": description,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat()
        })
        self.updated_at = datetime.utcnow()
    
    def update_risk_score(self, score: float) -> None:
        """更新风险评分"""
        self.risk_score = max(0.0, min(100.0, score))
        self.updated_at = datetime.utcnow()
    
    def establish_baseline(self, patterns: Dict[str, Any]) -> None:
        """建立行为基线"""
        self.normal_patterns = patterns
        self.baseline_established = True
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "user_id": self.user_id,
            "normal_patterns": self.normal_patterns,
            "anomalies": self.anomalies,
            "risk_score": self.risk_score,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "baseline_established": self.baseline_established
        }


class ThreatIndicator:
    """
    威胁指标类
    
    表示检测到的潜在威胁信号。
    """
    
    def __init__(
        self,
        indicator_type: str,
        description: str,
        confidence: float,
        related_events: List[str],
        severity: str = "medium",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.indicator_type = indicator_type
        self.description = description
        self.confidence = max(0.0, min(1.0, confidence))
        self.related_events = related_events
        self.severity = severity
        self.metadata = metadata or {}
        self.detected_at = datetime.utcnow()
        self.indicator_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """生成唯一指标ID"""
        timestamp_str = self.detected_at.strftime("%Y%m%d%H%M%S%f")
        return f"THREAT-{timestamp_str}-{hash(self.indicator_type) % 10000:04d}"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "indicator_id": self.indicator_id,
            "indicator_type": self.indicator_type,
            "description": self.description,
            "confidence": self.confidence,
            "related_events": self.related_events,
            "severity": self.severity,
            "metadata": self.metadata,
            "detected_at": self.detected_at.isoformat()
        }


class RiskScorer:
    """
    风险评分器
    
    计算用户、资源和会话的风险评分。
    """
    
    # 风险权重配置
    WEIGHTS = {
        "failed_login": 10,
        "privilege_escalation": 25,
        "off_hours_access": 5,
        "unusual_resource_access": 15,
        "data_exfiltration_pattern": 30,
        "session_anomaly": 8,
        "geographic_anomaly": 12
    }
    
    def __init__(self):
        self._user_risk_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self._resource_risk_scores: Dict[str, float] = {}
    
    def calculate_user_risk(
        self,
        user_id: str,
        events: List[Dict[str, Any]],
        time_window: Optional[timedelta] = None
    ) -> float:
        """
        计算用户风险评分
        
        Args:
            user_id: 用户标识
            events: 用户相关事件列表
            time_window: 评估时间窗口
            
        Returns:
            风险评分（0-100）
        """
        if time_window is None:
            time_window = timedelta(days=7)
        
        cutoff_time = datetime.utcnow() - time_window
        recent_events = [
            e for e in events
            if e.get("timestamp", datetime.utcnow()) > cutoff_time
        ]
        
        risk_score = 0.0
        
        # 计算各类风险因子
        failed_logins = sum(1 for e in recent_events if e.get("event_type") == "failed_login")
        risk_score += min(failed_logins * self.WEIGHTS["failed_login"], 50)
        
        privilege_changes = sum(1 for e in recent_events if e.get("event_type") == "privilege_change")
        risk_score += privilege_changes * self.WEIGHTS["privilege_escalation"]
        
        off_hours_access = sum(1 for e in recent_events if e.get("off_hours", False))
        risk_score += min(off_hours_access * self.WEIGHTS["off_hours_access"], 30)
        
        unusual_access = sum(1 for e in recent_events if e.get("unusual_resource", False))
        risk_score += min(unusual_access * self.WEIGHTS["unusual_resource_access"], 40)
        
        # 检查数据泄露模式
        if self._detect_data_exfiltration_pattern(recent_events):
            risk_score += self.WEIGHTS["data_exfiltration_pattern"]
        
        # 归一化到0-100
        final_score = min(100.0, risk_score)
        
        # 记录历史
        self._user_risk_history[user_id].append((datetime.utcnow(), final_score))
        
        return final_score
    
    def calculate_resource_risk(
        self,
        resource_id: str,
        access_events: List[Dict[str, Any]]
    ) -> float:
        """
        计算资源风险评分
        
        Args:
            resource_id: 资源标识
            access_events: 资源访问事件列表
            
        Returns:
            风险评分（0-100）
        """
        if not access_events:
            return 0.0
        
        risk_score = 0.0
        
        # 基于访问频率的风险
        access_count = len(access_events)
        unique_users = len(set(e.get("user") for e in access_events))
        denied_count = sum(1 for e in access_events if not e.get("granted", True))
        
        # 高频访问增加风险
        if access_count > 1000:
            risk_score += 20
        elif access_count > 100:
            risk_score += 10
        
        # 大量不同用户访问增加风险
        if unique_users > 50:
            risk_score += 15
        
        # 拒绝访问比例高增加风险
        if access_count > 0:
            denied_ratio = denied_count / access_count
            if denied_ratio > 0.3:
                risk_score += 25
            elif denied_ratio > 0.1:
                risk_score += 10
        
        # 敏感数据访问模式
        sensitive_access = sum(1 for e in access_events if e.get("sensitive", False))
        if sensitive_access > 100:
            risk_score += 20
        
        final_score = min(100.0, risk_score)
        self._resource_risk_scores[resource_id] = final_score
        
        return final_score
    
    def calculate_session_risk(
        self,
        session_id: str,
        session_events: List[Dict[str, Any]],
        session_metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        计算会话风险评分
        
        Args:
            session_id: 会话标识
            session_events: 会话事件列表
            session_metadata: 会话元数据
            
        Returns:
            风险评分（0-100）
        """
        risk_score = 0.0
        
        if session_metadata:
            # 检查会话持续时间
            duration = session_metadata.get("duration_seconds", 0)
            if duration > 8 * 3600:  # 超过8小时
                risk_score += 15
            
            # 检查资源访问数量
            resource_count = session_metadata.get("resources_accessed", 0)
            if resource_count > 100:
                risk_score += 20
            
            # 检查拒绝率
            denied_count = session_metadata.get("denied_count", 0)
            access_count = session_metadata.get("access_count", 1)
            if access_count > 0:
                denied_rate = denied_count / access_count
                if denied_rate > 0.3:
                    risk_score += 25
        
        # 分析会话事件
        for event in session_events:
            if event.get("event_type") == "privilege_escalation":
                risk_score += 20
            if event.get("event_type") == "sensitive_data_access":
                risk_score += 10
            if event.get("geographic_anomaly", False):
                risk_score += self.WEIGHTS["geographic_anomaly"]
        
        return min(100.0, risk_score)
    
    def _detect_data_exfiltration_pattern(self, events: List[Dict[str, Any]]) -> bool:
        """检测数据泄露模式"""
        # 简化实现：检查大量数据导出事件
        export_events = [e for e in events if e.get("event_type") == "data_export"]
        if len(export_events) >= 3:
            total_records = sum(e.get("records_count", 0) for e in export_events)
            if total_records > 10000:
                return True
        return False
    
    def get_user_risk_trend(self, user_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """获取用户风险趋势"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        history = [
            {"timestamp": ts.isoformat(), "score": score}
            for ts, score in self._user_risk_history.get(user_id, [])
            if ts > cutoff
        ]
        return history


class AuditAnalyzer:
    """
    审计分析器
    
    提供综合的审计数据分析功能。
    """
    
    def __init__(self, risk_scorer: Optional[RiskScorer] = None):
        self.risk_scorer = risk_scorer or RiskScorer()
        self._behavior_profiles: Dict[str, BehaviorProfile] = {}
        self._threat_indicators: List[ThreatIndicator] = []
    
    def analyze_user_behavior(
        self,
        user_id: str,
        events: List[Dict[str, Any]],
        establish_baseline: bool = False
    ) -> BehaviorProfile:
        """
        分析用户行为
        
        Args:
            user_id: 用户标识
            events: 用户事件列表
            establish_baseline: 是否建立行为基线
            
        Returns:
            用户行为画像
        """
        # 获取或创建行为画像
        if user_id not in self._behavior_profiles:
            self._behavior_profiles[user_id] = BehaviorProfile(user_id=user_id)
        
        profile = self._behavior_profiles[user_id]
        
        # 分析正常行为模式
        if events:
            # 计算常用操作
            action_counts = defaultdict(int)
            resource_counts = defaultdict(int)
            hourly_distribution = defaultdict(int)
            
            for event in events:
                action = event.get("action", "unknown")
                resource = event.get("resource", "unknown")
                timestamp = event.get("timestamp")
                
                action_counts[action] += 1
                resource_counts[resource] += 1
                
                if isinstance(timestamp, datetime):
                    hourly_distribution[timestamp.hour] += 1
            
            patterns = {
                "common_actions": dict(sorted(action_counts.items(), key=lambda x: -x[1])[:5]),
                "common_resources": dict(sorted(resource_counts.items(), key=lambda x: -x[1])[:5]),
                "active_hours": dict(hourly_distribution),
                "total_events": len(events)
            }
            
            if establish_baseline or not profile.baseline_established:
                profile.establish_baseline(patterns)
        
        # 检测异常
        self._detect_user_anomalies(profile, events)
        
        # 更新风险评分
        risk_score = self.risk_scorer.calculate_user_risk(user_id, events)
        profile.update_risk_score(risk_score)
        
        return profile
    
    def _detect_user_anomalies(
        self,
        profile: BehaviorProfile,
        events: List[Dict[str, Any]]
    ) -> None:
        """检测用户行为异常"""
        if not profile.baseline_established or not events:
            return
        
        # 检测异常时间访问
        off_hours_events = [
            e for e in events
            if isinstance(e.get("timestamp"), datetime)
            and (e["timestamp"].hour < 6 or e["timestamp"].hour > 22)
        ]
        
        if len(off_hours_events) > 5:
            profile.add_anomaly(
                anomaly_type="off_hours_access",
                description=f"User accessed system {len(off_hours_events)} times during off-hours",
                severity="medium"
            )
        
        # 检测异常资源访问
        if profile.normal_patterns.get("common_resources"):
            normal_resources = set(profile.normal_patterns["common_resources"].keys())
            recent_resources = set(e.get("resource", "") for e in events[-50:])
            unusual_resources = recent_resources - normal_resources
            
            if len(unusual_resources) > 10:
                profile.add_anomaly(
                    anomaly_type="unusual_resource_access",
                    description=f"User accessed {len(unusual_resources)} unusual resources",
                    severity="high"
                )
        
        # 检测操作频率异常
        if len(events) > 1000:  # 短时间内大量操作
            profile.add_anomaly(
                anomaly_type="high_frequency_activity",
                description=f"Unusually high activity: {len(events)} events",
                severity="medium"
            )
    
    def detect_insider_threat(
        self,
        user_id: str,
        events: List[Dict[str, Any]]
    ) -> List[ThreatIndicator]:
        """
        检测内部威胁
        
        Args:
            user_id: 用户标识
            events: 用户事件列表
            
        Returns:
            威胁指标列表
        """
        indicators = []
        
        # 检测数据访问模式异常
        data_access_events = [e for e in events if e.get("event_type") == "data_access"]
        if len(data_access_events) > 100:
            # 检查是否访问了平时不访问的数据
            indicators.append(ThreatIndicator(
                indicator_type="unusual_data_access",
                description=f"User accessed {len(data_access_events)} data records in short period",
                confidence=0.7,
                related_events=[e.get("event_id", "") for e in data_access_events[:10]],
                severity="high"
            ))
        
        # 检测权限滥用
        privilege_events = [e for e in events if e.get("event_type") == "privilege_use"]
        admin_actions = [e for e in privilege_events if e.get("admin_action", False)]
        if len(admin_actions) > 10:
            indicators.append(ThreatIndicator(
                indicator_type="privilege_abuse",
                description=f"User performed {len(admin_actions)} administrative actions",
                confidence=0.6,
                related_events=[e.get("event_id", "") for e in admin_actions[:10]],
                severity="medium"
            ))
        
        # 检测数据导出异常
        export_events = [e for e in events if e.get("event_type") == "data_export"]
        total_exported = sum(e.get("records_count", 0) for e in export_events)
        if total_exported > 10000:
            indicators.append(ThreatIndicator(
                indicator_type="potential_data_exfiltration",
                description=f"User exported {total_exported} records",
                confidence=0.8,
                related_events=[e.get("event_id", "") for e in export_events[:10]],
                severity="critical"
            ))
        
        # 检测访问时间异常
        if self._is_access_pattern_suspicious(events):
            indicators.append(ThreatIndicator(
                indicator_type="suspicious_access_pattern",
                description="User access pattern deviates significantly from baseline",
                confidence=0.5,
                related_events=[e.get("event_id", "") for e in events[-10:]],
                severity="medium"
            ))
        
        self._threat_indicators.extend(indicators)
        return indicators
    
    def _is_access_pattern_suspicious(self, events: List[Dict[str, Any]]) -> bool:
        """检查访问模式是否可疑"""
        if len(events) < 10:
            return False
        
        # 检查是否在短时间内有大量操作
        timestamps = [
            e.get("timestamp") for e in events
            if isinstance(e.get("timestamp"), datetime)
        ]
        
        if len(timestamps) < 2:
            return False
        
        timestamps.sort()
        time_span = (timestamps[-1] - timestamps[0]).total_seconds()
        
        if time_span > 0:
            rate = len(events) / time_span * 60  # 每分钟事件数
            if rate > 100:  # 每分钟超过100个事件
                return True
        
        return False
    
    def detect_compromised_account(
        self,
        user_id: str,
        events: List[Dict[str, Any]]
    ) -> List[ThreatIndicator]:
        """
        检测账户泄露
        
        Args:
            user_id: 用户标识
            events: 用户事件列表
            
        Returns:
            威胁指标列表
        """
        indicators = []
        
        # 检测多次失败登录后成功
        failed_logins = [e for e in events if e.get("event_type") == "failed_login"]
        successful_login = next(
            (e for e in events if e.get("event_type") == "successful_login"),
            None
        )
        
        if len(failed_logins) >= 5 and successful_login:
            # 检查时间顺序
            failed_times = [e.get("timestamp") for e in failed_logins if e.get("timestamp")]
            success_time = successful_login.get("timestamp")
            
            if failed_times and success_time:
                last_failed = max(t for t in failed_times if isinstance(t, datetime))
                if isinstance(success_time, datetime) and success_time > last_failed:
                    indicators.append(ThreatIndicator(
                        indicator_type="brute_force_success",
                        description=f"Account accessed after {len(failed_logins)} failed attempts",
                        confidence=0.75,
                        related_events=[e.get("event_id", "") for e in failed_logins[-5:]] + [successful_login.get("event_id", "")],
                        severity="critical"
                    ))
        
        # 检测地理位置异常
        locations = set(e.get("location", "") for e in events if e.get("location"))
        if len(locations) > 3:
            indicators.append(ThreatIndicator(
                indicator_type="impossible_travel",
                description=f"Account accessed from {len(locations)} different locations",
                confidence=0.6,
                related_events=[e.get("event_id", "") for e in events if e.get("location")][:10],
                severity="high"
            ))
        
        # 检测设备变更
        devices = set(e.get("device", "") for e in events if e.get("device"))
        if len(devices) > 2:
            indicators.append(ThreatIndicator(
                indicator_type="multiple_devices",
                description=f"Account accessed from {len(devices)} different devices",
                confidence=0.4,
                related_events=[e.get("event_id", "") for e in events if e.get("device")][:10],
                severity="low"
            ))
        
        self._threat_indicators.extend(indicators)
        return indicators
    
    def generate_risk_score(
        self,
        entity_type: str,
        entity_id: str,
        events: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        生成综合风险评分
        
        Args:
            entity_type: 实体类型（user/resource/session）
            entity_id: 实体标识
            events: 相关事件列表
            metadata: 额外元数据
            
        Returns:
            风险评分详情
        """
        if entity_type == "user":
            score = self.risk_scorer.calculate_user_risk(entity_id, events)
        elif entity_type == "resource":
            score = self.risk_scorer.calculate_resource_risk(entity_id, events)
        elif entity_type == "session":
            score = self.risk_scorer.calculate_session_risk(entity_id, events, metadata)
        else:
            score = 0.0
        
        # 确定风险级别
        if score >= 80:
            level = "critical"
        elif score >= 60:
            level = "high"
        elif score >= 40:
            level = "medium"
        elif score >= 20:
            level = "low"
        else:
            level = "minimal"
        
        return {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "risk_score": round(score, 2),
            "risk_level": level,
            "assessed_at": datetime.utcnow().isoformat(),
            "factors": self._identify_risk_factors(events, metadata)
        }
    
    def _identify_risk_factors(
        self,
        events: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """识别风险因子"""
        factors = []
        
        failed_count = sum(1 for e in events if e.get("event_type") == "failed_login")
        if failed_count > 0:
            factors.append({
                "factor": "failed_authentication",
                "count": failed_count,
                "contribution": min(failed_count * 10, 50)
            })
        
        off_hours_count = sum(1 for e in events if e.get("off_hours", False))
        if off_hours_count > 0:
            factors.append({
                "factor": "off_hours_activity",
                "count": off_hours_count,
                "contribution": min(off_hours_count * 5, 30)
            })
        
        if metadata:
            if metadata.get("denied_count", 0) > 0:
                factors.append({
                    "factor": "access_denials",
                    "count": metadata["denied_count"],
                    "contribution": min(metadata["denied_count"] * 8, 40)
                })
        
        return factors
    
    def get_behavior_profile(self, user_id: str) -> Optional[BehaviorProfile]:
        """获取用户行为画像"""
        return self._behavior_profiles.get(user_id)
    
    def get_threat_indicators(
        self,
        indicator_type: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> List[ThreatIndicator]:
        """
        获取威胁指标
        
        Args:
            indicator_type: 按类型筛选
            min_confidence: 最小置信度
            
        Returns:
            威胁指标列表
        """
        indicators = self._threat_indicators
        
        if indicator_type:
            indicators = [i for i in indicators if i.indicator_type == indicator_type]
        
        indicators = [i for i in indicators if i.confidence >= min_confidence]
        
        return sorted(indicators, key=lambda x: x.confidence, reverse=True)
    
    def export_analysis_report(self, output_path: str) -> str:
        """
        导出分析报告
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            输出文件路径
        """
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "behavior_profiles": {
                uid: profile.to_dict()
                for uid, profile in self._behavior_profiles.items()
            },
            "threat_indicators": [
                indicator.to_dict()
                for indicator in self._threat_indicators
            ],
            "summary": {
                "total_profiles": len(self._behavior_profiles),
                "total_threats": len(self._threat_indicators),
                "high_risk_users": sum(
                    1 for p in self._behavior_profiles.values()
                    if p.risk_score >= 60
                )
            }
        }
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return output_path
