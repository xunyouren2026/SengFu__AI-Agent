"""
Content Detector Module - 内容检测器

提供敏感内容检测和数据外泄检测能力：
- 敏感内容检测
- 数据外泄检测（基于流量模式、访问频率、数据量）
- 异常访问检测（统计基线、行为模式分析）
"""

import json
import math
import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Set, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, deque


class DetectionType(Enum):
    """检测类型"""
    SENSITIVE_CONTENT = "sensitive_content"
    DATA_EXFILTRATION = "data_exfiltration"
    ANOMALOUS_ACCESS = "anomalous_access"
    VOLUME_ANOMALY = "volume_anomaly"
    FREQUENCY_ANOMALY = "frequency_anomaly"


class RiskLevel(Enum):
    """风险级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DetectionResult:
    """检测结果"""
    detection_type: DetectionType
    risk_level: RiskLevel
    confidence: float
    description: str
    affected_data: Dict[str, Any]
    timestamp: str
    source: str = ""
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "detection_type": self.detection_type.value,
            "risk_level": self.risk_level.value,
            "confidence": self.confidence,
            "description": self.description,
            "affected_data": self.affected_data,
            "timestamp": self.timestamp,
            "source": self.source,
            "recommendations": self.recommendations,
            "metadata": self.metadata
        }


@dataclass
class AccessPattern:
    """访问模式"""
    user: str
    resource: str
    access_count: int
    first_access: str
    last_access: str
    avg_data_size: float
    peak_hour: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContentDetector:
    """内容检测器"""
    
    def __init__(self):
        self.sensitive_keywords: Set[str] = set()
        self.sensitive_patterns: List[str] = []
        self.detection_history: List[DetectionResult] = []
        self._init_default_keywords()
    
    def _init_default_keywords(self):
        """初始化默认敏感关键词"""
        # 财务敏感词
        self.sensitive_keywords.update([
            "财务报表", "revenue", "profit", "loss", "balance sheet",
            "income statement", "cash flow", "预算", "budge"
        ])
        
        # 个人敏感词
        self.sensitive_keywords.update([
            "身份证号", "护照", "银行卡", "社保号", "ssn",
            "id card", "passport", "credit card"
        ])
        
        # 商业机密
        self.sensitive_keywords.update([
            "机密", "confidential", "proprietary", "trade secret",
            "专利", "patent pending", "算法细节"
        ])
        
        # 法律相关
        self.sensitive_keywords.update([
            "合同", "contract", "诉讼", "lawsuit", "法律意见",
            "attorney", "legal advice"
        ])
    
    def add_sensitive_keyword(self, keyword: str):
        """添加敏感关键词"""
        self.sensitive_keywords.add(keyword.lower())
    
    def remove_sensitive_keyword(self, keyword: str):
        """移除敏感关键词"""
        self.sensitive_keywords.discard(keyword.lower())
    
    def detect_sensitive_content(self, content: str, 
                                  context: Optional[Dict[str, Any]] = None) -> DetectionResult:
        """
        检测敏感内容
        
        Args:
            content: 要检测的内容
            context: 上下文信息
            
        Returns:
            检测结果
        """
        context = context or {}
        content_lower = content.lower()
        
        # 关键词匹配
        matched_keywords = []
        for keyword in self.sensitive_keywords:
            if keyword in content_lower:
                matched_keywords.append(keyword)
        
        # 计算敏感度得分
        sensitivity_score = len(matched_keywords) / max(len(self.sensitive_keywords), 1)
        
        # 确定风险级别
        if sensitivity_score > 0.1:
            risk_level = RiskLevel.CRITICAL
        elif sensitivity_score > 0.05:
            risk_level = RiskLevel.HIGH
        elif sensitivity_score > 0.02:
            risk_level = RiskLevel.MEDIUM
        elif matched_keywords:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.LOW
        
        # 生成建议
        recommendations = []
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recommendations.append("建议立即审查内容并限制访问")
            recommendations.append("考虑对内容进行加密存储")
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.append("建议标记内容并实施访问控制")
        
        result = DetectionResult(
            detection_type=DetectionType.SENSITIVE_CONTENT,
            risk_level=risk_level,
            confidence=min(sensitivity_score * 10, 1.0),
            description=f"检测到{len(matched_keywords)}个敏感关键词" if matched_keywords else "未检测到敏感内容",
            affected_data={
                "content_preview": content[:100] + "..." if len(content) > 100 else content,
                "matched_keywords": matched_keywords,
                "sensitivity_score": sensitivity_score
            },
            timestamp=datetime.now().isoformat(),
            source=context.get("source", ""),
            recommendations=recommendations,
            metadata={
                "total_keywords_checked": len(self.sensitive_keywords),
                "context": context
            }
        )
        
        self.detection_history.append(result)
        return result
    
    def detect_data_exfiltration(self, transfer_info: Dict[str, Any]) -> DetectionResult:
        """
        检测数据外泄
        
        Args:
            transfer_info: 传输信息
            
        Returns:
            检测结果
        """
        risk_indicators = []
        risk_score = 0.0
        
        # 检查目的地
        destination = transfer_info.get("destination", "")
        unauthorized_destinations = ["personal_email", "public_cloud", "usb_drive", "external_drive"]
        if any(dest in destination.lower() for dest in unauthorized_destinations):
            risk_indicators.append("未授权目的地")
            risk_score += 0.4
        
        # 检查数据量
        data_size = transfer_info.get("size", 0)
        if data_size > 100 * 1024 * 1024:  # 100MB
            risk_indicators.append("大数据量传输")
            risk_score += 0.3
        elif data_size > 10 * 1024 * 1024:  # 10MB
            risk_indicators.append("中等数据量传输")
            risk_score += 0.15
        
        # 检查数据敏感度
        classification = transfer_info.get("classification", "")
        if classification in ["secret", "restricted", "confidential"]:
            risk_indicators.append(f"高敏感度数据({classification})")
            risk_score += 0.3
        
        # 检查PII
        pii_types = transfer_info.get("pii_types", [])
        if pii_types:
            risk_indicators.append(f"包含PII({len(pii_types)}种)")
            risk_score += 0.2
        
        # 检查时间异常
        hour = datetime.now().hour
        if hour < 6 or hour > 22:  # 非工作时间
            risk_indicators.append("非工作时间传输")
            risk_score += 0.1
        
        # 确定风险级别
        if risk_score >= 0.8:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 0.4:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        # 生成建议
        recommendations = []
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recommendations.append("立即阻断传输并通知安全团队")
            recommendations.append("审查用户访问权限")
            recommendations.append("检查数据是否已泄露")
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.append("增加监控频率")
            recommendations.append("记录传输详情供审计")
        
        result = DetectionResult(
            detection_type=DetectionType.DATA_EXFILTRATION,
            risk_level=risk_level,
            confidence=min(risk_score, 1.0),
            description=f"检测到潜在数据外泄风险: {', '.join(risk_indicators)}" if risk_indicators else "未检测到明显外泄风险",
            affected_data={
                "destination": destination,
                "size": data_size,
                "classification": classification,
                "pii_types": pii_types,
                "risk_indicators": risk_indicators
            },
            timestamp=datetime.now().isoformat(),
            source=transfer_info.get("source", ""),
            recommendations=recommendations,
            metadata={
                "user": transfer_info.get("user", ""),
                "transfer_time": transfer_info.get("timestamp", "")
            }
        )
        
        self.detection_history.append(result)
        return result
    
    def detect_anomalous_access(self, access_info: Dict[str, Any],
                                 baseline: Optional[Dict[str, Any]] = None) -> DetectionResult:
        """
        检测异常访问
        
        Args:
            access_info: 访问信息
            baseline: 行为基线
            
        Returns:
            检测结果
        """
        anomalies = []
        anomaly_score = 0.0
        
        user = access_info.get("user", "")
        resource = access_info.get("resource", "")
        action = access_info.get("action", "")
        
        # 检查异常时间访问
        hour = datetime.now().hour
        if hour < 6 or hour > 22:
            anomalies.append("非工作时间访问")
            anomaly_score += 0.2
        
        # 检查异常位置
        location = access_info.get("location", "")
        if baseline and "usual_locations" in baseline:
            if location not in baseline["usual_locations"]:
                anomalies.append("异常位置访问")
                anomaly_score += 0.3
        
        # 检查异常频率
        if baseline and "avg_daily_access" in baseline:
            current_count = access_info.get("daily_count", 1)
            avg_count = baseline["avg_daily_access"]
            if current_count > avg_count * 3:
                anomalies.append("访问频率异常高")
                anomaly_score += 0.25
        
        # 检查异常数据量
        data_size = access_info.get("data_size", 0)
        if baseline and "avg_data_size" in baseline:
            avg_size = baseline["avg_data_size"]
            if data_size > avg_size * 5:
                anomalies.append("访问数据量异常大")
                anomaly_score += 0.25
        
        # 检查敏感资源
        sensitive_resources = baseline.get("sensitive_resources", []) if baseline else []
        if resource in sensitive_resources:
            anomalies.append("访问敏感资源")
            anomaly_score += 0.15
        
        # 确定风险级别
        if anomaly_score >= 0.7:
            risk_level = RiskLevel.CRITICAL
        elif anomaly_score >= 0.5:
            risk_level = RiskLevel.HIGH
        elif anomaly_score >= 0.3:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        # 生成建议
        recommendations = []
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recommendations.append("立即验证用户身份")
            recommendations.append("暂时限制用户访问权限")
            recommendations.append("审查用户近期活动")
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.append("增加该用户的监控频率")
            recommendations.append("记录访问详情")
        
        result = DetectionResult(
            detection_type=DetectionType.ANOMALOUS_ACCESS,
            risk_level=risk_level,
            confidence=min(anomaly_score, 1.0),
            description=f"检测到异常访问: {', '.join(anomalies)}" if anomalies else "访问模式正常",
            affected_data={
                "user": user,
                "resource": resource,
                "action": action,
                "location": location,
                "anomalies": anomalies
            },
            timestamp=datetime.now().isoformat(),
            source=resource,
            recommendations=recommendations,
            metadata={
                "baseline_used": baseline is not None,
                "anomaly_score": anomaly_score
            }
        )
        
        self.detection_history.append(result)
        return result
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """获取检测摘要"""
        summary = {
            "total_detections": len(self.detection_history),
            "by_type": defaultdict(int),
            "by_risk_level": defaultdict(int),
            "high_risk_count": 0
        }
        
        for result in self.detection_history:
            summary["by_type"][result.detection_type.value] += 1
            summary["by_risk_level"][result.risk_level.value] += 1
            if result.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                summary["high_risk_count"] += 1
        
        summary["by_type"] = dict(summary["by_type"])
        summary["by_risk_level"] = dict(summary["by_risk_level"])
        
        return summary


class ExfiltrationDetector:
    """外泄检测器"""
    
    def __init__(self, window_size: int = 24):
        self.window_size = window_size  # 小时
        self.transfer_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.volume_threshold = 100 * 1024 * 1024  # 100MB
        self.frequency_threshold = 10  # 每小时10次
    
    def record_transfer(self, user: str, size: int, destination: str, 
                       timestamp: Optional[str] = None):
        """记录传输"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        self.transfer_history[user].append({
            "timestamp": timestamp,
            "size": size,
            "destination": destination
        })
    
    def detect_by_traffic_pattern(self, user: str) -> DetectionResult:
        """基于流量模式检测"""
        transfers = list(self.transfer_history[user])
        
        if not transfers:
            return DetectionResult(
                detection_type=DetectionType.DATA_EXFILTRATION,
                risk_level=RiskLevel.LOW,
                confidence=0.0,
                description="无传输记录",
                affected_data={"user": user},
                timestamp=datetime.now().isoformat()
            )
        
        # 计算统计信息
        total_size = sum(t["size"] for t in transfers)
        unique_destinations = set(t["destination"] for t in transfers)
        
        risk_score = 0.0
        indicators = []
        
        # 检查总数据量
        if total_size > self.volume_threshold * 5:
            risk_score += 0.4
            indicators.append("超大传输总量")
        elif total_size > self.volume_threshold:
            risk_score += 0.2
            indicators.append("大传输总量")
        
        # 检查目的地多样性
        if len(unique_destinations) > 5:
            risk_score += 0.3
            indicators.append("多目的地传输")
        
        # 确定风险级别
        if risk_score >= 0.6:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 0.3:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        return DetectionResult(
            detection_type=DetectionType.DATA_EXFILTRATION,
            risk_level=risk_level,
            confidence=min(risk_score, 1.0),
            description=f"流量模式分析: {', '.join(indicators)}" if indicators else "流量模式正常",
            affected_data={
                "user": user,
                "total_size": total_size,
                "transfer_count": len(transfers),
                "unique_destinations": len(unique_destinations)
            },
            timestamp=datetime.now().isoformat(),
            recommendations=["监控用户传输活动"] if risk_level != RiskLevel.LOW else []
        )
    
    def detect_by_frequency(self, user: str, time_window: int = 1) -> DetectionResult:
        """基于访问频率检测"""
        transfers = list(self.transfer_history[user])
        
        if not transfers:
            return DetectionResult(
                detection_type=DetectionType.FREQUENCY_ANOMALY,
                risk_level=RiskLevel.LOW,
                confidence=0.0,
                description="无传输记录",
                affected_data={"user": user},
                timestamp=datetime.now().isoformat()
            )
        
        # 计算每小时频率
        now = datetime.now()
        recent_transfers = [
            t for t in transfers
            if (now - datetime.fromisoformat(t["timestamp"])).total_seconds() < 3600 * time_window
        ]
        
        frequency = len(recent_transfers) / time_window
        
        if frequency > self.frequency_threshold * 2:
            risk_level = RiskLevel.HIGH
            confidence = min(frequency / (self.frequency_threshold * 3), 1.0)
        elif frequency > self.frequency_threshold:
            risk_level = RiskLevel.MEDIUM
            confidence = 0.6
        else:
            risk_level = RiskLevel.LOW
            confidence = 0.2
        
        return DetectionResult(
            detection_type=DetectionType.FREQUENCY_ANOMALY,
            risk_level=risk_level,
            confidence=confidence,
            description=f"传输频率: {frequency:.1f}次/小时",
            affected_data={
                "user": user,
                "frequency": frequency,
                "threshold": self.frequency_threshold
            },
            timestamp=datetime.now().isoformat(),
            recommendations=["审查用户传输行为"] if risk_level != RiskLevel.LOW else []
        )
    
    def detect_by_volume(self, user: str, time_window: int = 1) -> DetectionResult:
        """基于数据量检测"""
        transfers = list(self.transfer_history[user])
        
        if not transfers:
            return DetectionResult(
                detection_type=DetectionType.VOLUME_ANOMALY,
                risk_level=RiskLevel.LOW,
                confidence=0.0,
                description="无传输记录",
                affected_data={"user": user},
                timestamp=datetime.now().isoformat()
            )
        
        # 计算每小时数据量
        now = datetime.now()
        recent_transfers = [
            t for t in transfers
            if (now - datetime.fromisoformat(t["timestamp"])).total_seconds() < 3600 * time_window
        ]
        
        total_volume = sum(t["size"] for t in recent_transfers)
        volume_per_hour = total_volume / time_window
        
        if volume_per_hour > self.volume_threshold * 3:
            risk_level = RiskLevel.HIGH
            confidence = 0.9
        elif volume_per_hour > self.volume_threshold:
            risk_level = RiskLevel.MEDIUM
            confidence = 0.6
        else:
            risk_level = RiskLevel.LOW
            confidence = 0.2
        
        return DetectionResult(
            detection_type=DetectionType.VOLUME_ANOMALY,
            risk_level=risk_level,
            confidence=confidence,
            description=f"传输数据量: {volume_per_hour / 1024 / 1024:.1f}MB/小时",
            affected_data={
                "user": user,
                "volume_per_hour": volume_per_hour,
                "threshold": self.volume_threshold
            },
            timestamp=datetime.now().isoformat(),
            recommendations=["审查数据传输目的"] if risk_level != RiskLevel.LOW else []
        )


class AnomalyDetector:
    """异常检测器"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.baselines: Dict[str, Dict[str, Any]] = {}
        self.access_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
    
    def build_baseline(self, user: str, historical_data: List[Dict[str, Any]]):
        """构建用户行为基线"""
        if not historical_data:
            return
        
        # 计算统计基线
        hours = [datetime.fromisoformat(d["timestamp"]).hour for d in historical_data]
        sizes = [d.get("data_size", 0) for d in historical_data]
        locations = list(set(d.get("location", "") for d in historical_data))
        
        self.baselines[user] = {
            "usual_hours": list(set(hours)),
            "avg_data_size": sum(sizes) / len(sizes) if sizes else 0,
            "std_data_size": self._calculate_std(sizes) if sizes else 0,
            "usual_locations": locations,
            "avg_daily_access": len(historical_data) / 30,  # 假设30天数据
            "peak_hour": max(set(hours), key=hours.count) if hours else 9
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
    
    def detect_statistical_anomaly(self, user: str, 
                                    access_info: Dict[str, Any]) -> DetectionResult:
        """统计基线方法检测异常"""
        baseline = self.baselines.get(user)
        
        if not baseline:
            return DetectionResult(
                detection_type=DetectionType.ANOMALOUS_ACCESS,
                risk_level=RiskLevel.LOW,
                confidence=0.0,
                description="无行为基线",
                affected_data={"user": user},
                timestamp=datetime.now().isoformat()
            )
        
        anomalies = []
        anomaly_score = 0.0
        
        # 检查时间异常
        hour = datetime.now().hour
        if hour not in baseline["usual_hours"]:
            anomalies.append("异常时间访问")
            anomaly_score += 0.2
        
        # 检查数据量异常（使用3-sigma规则）
        data_size = access_info.get("data_size", 0)
        avg_size = baseline["avg_data_size"]
        std_size = baseline["std_data_size"]
        
        if std_size > 0:
            z_score = abs(data_size - avg_size) / std_size
            if z_score > 3:
                anomalies.append("数据量严重异常")
                anomaly_score += 0.4
            elif z_score > 2:
                anomalies.append("数据量异常")
                anomaly_score += 0.2
        
        # 检查位置异常
        location = access_info.get("location", "")
        if location and location not in baseline["usual_locations"]:
            anomalies.append("异常位置")
            anomaly_score += 0.3
        
        # 确定风险级别
        if anomaly_score >= 0.6:
            risk_level = RiskLevel.HIGH
        elif anomaly_score >= 0.3:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        return DetectionResult(
            detection_type=DetectionType.ANOMALOUS_ACCESS,
            risk_level=risk_level,
            confidence=min(anomaly_score, 1.0),
            description=f"统计异常检测: {', '.join(anomalies)}" if anomalies else "访问正常",
            affected_data={
                "user": user,
                "anomalies": anomalies,
                "baseline_available": True
            },
            timestamp=datetime.now().isoformat(),
            recommendations=["审查访问行为"] if risk_level != RiskLevel.LOW else []
        )
    
    def detect_behavior_pattern(self, user: str, 
                                 recent_access: List[Dict[str, Any]]) -> DetectionResult:
        """行为模式分析"""
        if len(recent_access) < 5:
            return DetectionResult(
                detection_type=DetectionType.ANOMALOUS_ACCESS,
                risk_level=RiskLevel.LOW,
                confidence=0.0,
                description="数据不足",
                affected_data={"user": user},
                timestamp=datetime.now().isoformat()
            )
        
        # 分析访问模式
        resources = [a.get("resource", "") for a in recent_access]
        actions = [a.get("action", "") for a in recent_access]
        
        # 检查是否访问了异常多的资源
        unique_resources = len(set(resources))
        total_access = len(recent_access)
        
        # 检查是否有异常行为序列
        unusual_patterns = []
        pattern_score = 0.0
        
        if unique_resources > total_access * 0.8:
            unusual_patterns.append("资源访问过于分散")
            pattern_score += 0.3
        
        if "download" in actions and "delete" in actions:
            unusual_patterns.append("下载后删除模式")
            pattern_score += 0.4
        
        # 确定风险级别
        if pattern_score >= 0.5:
            risk_level = RiskLevel.HIGH
        elif pattern_score >= 0.3:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        return DetectionResult(
            detection_type=DetectionType.ANOMALOUS_ACCESS,
            risk_level=risk_level,
            confidence=min(pattern_score, 1.0),
            description=f"行为模式分析: {', '.join(unusual_patterns)}" if unusual_patterns else "行为模式正常",
            affected_data={
                "user": user,
                "unique_resources": unique_resources,
                "total_access": total_access,
                "patterns": unusual_patterns
            },
            timestamp=datetime.now().isoformat(),
            recommendations=["进一步审查用户行为"] if risk_level != RiskLevel.LOW else []
        )


# 便捷函数
def detect_sensitive_content(content: str) -> DetectionResult:
    """便捷函数：检测敏感内容"""
    detector = ContentDetector()
    return detector.detect_sensitive_content(content)


def detect_data_exfiltration(transfer_info: Dict[str, Any]) -> DetectionResult:
    """便捷函数：检测数据外泄"""
    detector = ContentDetector()
    return detector.detect_data_exfiltration(transfer_info)


# 示例用法
if __name__ == "__main__":
    print("内容检测器测试：")
    print("=" * 60)
    
    # 内容检测器测试
    content_detector = ContentDetector()
    
    print("\n1. 敏感内容检测：")
    test_contents = [
        "这是一份普通的会议记录",
        "财务报表显示本季度收入1000万，利润200万",
        "客户信息：身份证号110101199001011234，银行卡6222021234567890123"
    ]
    
    for content in test_contents:
        result = content_detector.detect_sensitive_content(content)
        print(f"\n  内容: {content[:30]}...")
        print(f"  风险级别: {result.risk_level.value}")
        print(f"  置信度: {result.confidence:.2f}")
        print(f"  描述: {result.description}")
    
    # 外泄检测器测试
    print("\n2. 外泄检测测试：")
    exfil_detector = ExfiltrationDetector()
    
    # 模拟传输记录
    for i in range(15):
        exfil_detector.record_transfer(
            user="user_001",
            size=50 * 1024 * 1024,  # 50MB
            destination="external_email"
        )
    
    result = exfil_detector.detect_by_traffic_pattern("user_001")
    print(f"\n  流量模式检测:")
    print(f"    风险级别: {result.risk_level.value}")
    print(f"    描述: {result.description}")
    
    result = exfil_detector.detect_by_frequency("user_001")
    print(f"\n  频率检测:")
    print(f"    风险级别: {result.risk_level.value}")
    print(f"    描述: {result.description}")
    
    # 异常检测器测试
    print("\n3. 异常检测测试：")
    anomaly_detector = AnomalyDetector()
    
    # 构建基线
    historical = [
        {"timestamp": datetime.now().isoformat(), "data_size": 1000, "location": "office"}
        for _ in range(100)
    ]
    anomaly_detector.build_baseline("user_002", historical)
    
    # 检测异常访问
    access_info = {
        "data_size": 50000,  # 异常大的数据量
        "location": "remote"  # 异常位置
    }
    result = anomaly_detector.detect_statistical_anomaly("user_002", access_info)
    print(f"\n  统计异常检测:")
    print(f"    风险级别: {result.risk_level.value}")
    print(f"    描述: {result.description}")
    
    # 检测摘要
    print("\n4. 检测摘要：")
    summary = content_detector.get_detection_summary()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
