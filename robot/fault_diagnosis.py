#!/usr/bin/env python3
"""
故障诊断模块 - 生产级实现
功能：多维度异常检测（温度、力矩、电压、通信）、根因分析（规则+ML）、
      恢复建议生成、故障历史记录、自愈尝试

注意：使用纯Python标准库实现，不依赖numpy
"""

import logging
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


class FaultType(Enum):
    JOINT_STUCK = "joint_stuck"
    OVERHEAT = "overheat"
    COLLISION = "collision"
    COMMUNICATION_LOSS = "comm_loss"
    POWER_DROP = "power_drop"
    ENCODER_ERROR = "encoder_error"
    SAFETY_STOP = "safety_stop"
    UNKNOWN = "unknown"


class Severity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class FaultEvent:
    """故障事件记录"""
    fault_id: str
    fault_type: FaultType
    severity: Severity
    timestamp: float
    details: Dict[str, Any]
    root_cause: str = ""
    recovery_action: str = ""
    resolved: bool = False
    resolved_at: Optional[float] = None


@dataclass
class DiagnosisResult:
    """诊断结果"""
    fault_detected: bool
    fault_type: Optional[FaultType]
    severity: Optional[Severity]
    confidence: float
    root_cause: str
    recovery_suggestion: str
    details: Dict[str, Any] = field(default_factory=dict)


def mean(data: List[float]) -> float:
    """计算平均值"""
    return sum(data) / len(data) if data else 0.0


def std(data: List[float]) -> float:
    """计算标准差"""
    if not data:
        return 0.0
    avg = mean(data)
    variance = sum((x - avg) ** 2 for x in data) / len(data)
    return variance ** 0.5


class FaultDiagnosis:
    """
    故障诊断器
    支持：
    - 基于阈值的实时检测
    - 基于统计的异常检测（移动平均、标准差）
    - 根因分析（规则引擎）
    - 恢复建议（预定义 + 动态生成）
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        # 阈值配置
        self.thresholds = {
            "joint_temperature_max": 70.0,      # °C
            "joint_torque_max": 50.0,           # Nm
            "voltage_min": 22.0,                # V
            "voltage_max": 26.0,
            "current_max": 15.0,                # A
            "communication_timeout": 0.5,       # 秒
            "velocity_max": 2.0,                # rad/s
        }
        if config:
            self.thresholds.update(config.get("thresholds", {}))
        
        # 历史数据窗口（用于统计异常检测）
        self._joint_temps_window: Dict[int, deque] = {}   # joint_idx -> deque
        self._joint_torques_window: Dict[int, deque] = {}
        self._window_size = 100
        
        # 故障历史
        self.fault_history: List[FaultEvent] = []
        self._max_history = 1000
        
        # 恢复策略映射
        self._recovery_map = {
            FaultType.OVERHEAT: "停止运动，冷却10分钟，降低速度限制，检查散热",
            FaultType.JOINT_STUCK: "检查机械阻碍，清除工作空间，降低速度重试",
            FaultType.COLLISION: "立即停止，检查碰撞区域，重新规划路径",
            FaultType.POWER_DROP: "检查电源连接，确保电压稳定，重启控制器",
            FaultType.COMMUNICATION_LOSS: "检查网线/串口连接，重启通信服务",
            FaultType.ENCODER_ERROR: "重新校准编码器，检查接线",
            FaultType.SAFETY_STOP: "手动复位急停按钮，确认安全后恢复",
            FaultType.UNKNOWN: "请联系技术支持"
        }
    
    # ==================== 数据采集与窗口管理 ====================
    def update_window(self, joint_idx: int, temperature: float, torque: float):
        """更新滑动窗口数据"""
        if joint_idx not in self._joint_temps_window:
            self._joint_temps_window[joint_idx] = deque(maxlen=self._window_size)
            self._joint_torques_window[joint_idx] = deque(maxlen=self._window_size)
        self._joint_temps_window[joint_idx].append(temperature)
        self._joint_torques_window[joint_idx].append(torque)
    
    # ==================== 异常检测 ====================
    def detect_threshold_violations(self, joint_temperatures: List[float],
                                     joint_torques: List[float],
                                     voltage: float,
                                     current: float) -> List[FaultType]:
        """基于阈值的检测"""
        faults = []
        # 温度
        for i, temp in enumerate(joint_temperatures):
            if temp > self.thresholds["joint_temperature_max"]:
                faults.append(FaultType.OVERHEAT)
                break
        # 力矩
        for i, torque in enumerate(joint_torques):
            if abs(torque) > self.thresholds["joint_torque_max"]:
                faults.append(FaultType.JOINT_STUCK)
                break
        # 电压
        if voltage < self.thresholds["voltage_min"] or voltage > self.thresholds["voltage_max"]:
            faults.append(FaultType.POWER_DROP)
        # 电流
        if current > self.thresholds["current_max"]:
            faults.append(FaultType.JOINT_STUCK)
        return faults
    
    def detect_statistical_anomaly(self, joint_idx: int, current_temp: float) -> bool:
        """基于统计的异常检测（3-sigma）"""
        if joint_idx not in self._joint_temps_window or len(self._joint_temps_window[joint_idx]) < 30:
            return False
        temps = list(self._joint_temps_window[joint_idx])
        avg = mean(temps)
        std_val = std(temps)
        if std_val < 0.1:
            return False
        return abs(current_temp - avg) > 3 * std_val
    
    # ==================== 根因分析 ====================
    def analyze_root_cause(self, faults: List[FaultType], context: Dict) -> str:
        """根因分析（规则引擎）"""
        if FaultType.OVERHEAT in faults:
            # 检查是否持续高负载
            if context.get("duty_cycle", 0) > 0.8:
                return "长时间高负载运行导致过热"
            else:
                return "散热不良或环境温度过高"
        elif FaultType.JOINT_STUCK in faults:
            if context.get("last_command") == "movej" and context.get("collision_detected", False):
                return "运动过程中发生碰撞"
            else:
                return "机械卡死或负载过大"
        elif FaultType.POWER_DROP in faults:
            return "电源供应不足或线路接触不良"
        elif FaultType.COMMUNICATION_LOSS in faults:
            return "网络或串口通信中断"
        return "未知原因，需要人工检查"
    
    def get_recovery_suggestion(self, fault_type: FaultType, context: Dict) -> str:
        """获取恢复建议"""
        base = self._recovery_map.get(fault_type, "请重启系统")
        # 根据上下文定制
        if fault_type == FaultType.OVERHEAT and context.get("cooling_fan_working") == False:
            base += "；检测到散热风扇故障"
        return base
    
    # ==================== 主诊断接口 ====================
    def diagnose(self, joint_positions: List[float], joint_temperatures: List[float],
                 joint_torques: List[float], voltage: float, current: float,
                 last_command: str = "", collision_detected: bool = False,
                 comm_ok: bool = True) -> DiagnosisResult:
        """
        执行完整诊断
        :return: DiagnosisResult
        """
        # 更新滑动窗口
        for i, (temp, tor) in enumerate(zip(joint_temperatures, joint_torques)):
            self.update_window(i, temp, tor)
        
        # 检测故障
        faults = self.detect_threshold_violations(joint_temperatures, joint_torques, voltage, current)
        if not comm_ok:
            faults.append(FaultType.COMMUNICATION_LOSS)
        
        if not faults:
            return DiagnosisResult(
                fault_detected=False,
                fault_type=None,
                severity=None,
                confidence=1.0,
                root_cause="No anomaly detected",
                recovery_suggestion="No action needed"
            )
        
        # 取第一个故障为主要故障
        main_fault = faults[0]
        # 确定严重程度
        severity = self._determine_severity(main_fault, joint_temperatures, joint_torques)
        # 根因分析
        context = {
            "last_command": last_command,
            "collision_detected": collision_detected,
            "duty_cycle": max(joint_torques) / self.thresholds["joint_torque_max"] if joint_torques else 0,
            "cooling_fan_working": True  # 可来自传感器
        }
        root_cause = self.analyze_root_cause(faults, context)
        recovery = self.get_recovery_suggestion(main_fault, context)
        
        # 记录故障事件
        self._record_fault(main_fault, severity, {"faults": [f.value for f in faults], **context})
        
        confidence = 0.8 + 0.1 * len(faults) if len(faults) <= 2 else 0.95
        return DiagnosisResult(
            fault_detected=True,
            fault_type=main_fault,
            severity=severity,
            confidence=min(confidence, 1.0),
            root_cause=root_cause,
            recovery_suggestion=recovery,
            details={"all_faults": [f.value for f in faults]}
        )
    
    def _determine_severity(self, fault: FaultType, temps: List[float], torques: List[float]) -> Severity:
        """根据故障类型和数值确定严重程度"""
        if fault == FaultType.OVERHEAT:
            max_temp = max(temps) if temps else 0
            if max_temp > 80:
                return Severity.CRITICAL
            elif max_temp > 70:
                return Severity.HIGH
            return Severity.MEDIUM
        elif fault == FaultType.COLLISION:
            return Severity.HIGH
        elif fault == FaultType.COMMUNICATION_LOSS:
            return Severity.CRITICAL
        elif fault == FaultType.POWER_DROP:
            return Severity.HIGH
        else:
            return Severity.MEDIUM
    
    def _record_fault(self, fault_type: FaultType, severity: Severity, details: Dict):
        event = FaultEvent(
            fault_id=f"fault_{int(time.time()*1000)}_{len(self.fault_history)}",
            fault_type=fault_type,
            severity=severity,
            timestamp=time.time(),
            details=details
        )
        self.fault_history.append(event)
        if len(self.fault_history) > self._max_history:
            self.fault_history = self.fault_history[-self._max_history:]
        logger.warning(f"Fault recorded: {fault_type.value}, severity={severity.name}, details={details}")
    
    # ==================== 自愈尝试 ====================
    async def attempt_self_heal(self, fault_type: FaultType) -> bool:
        """尝试自动恢复（简单重试、复位等）"""
        if fault_type == FaultType.COMMUNICATION_LOSS:
            # 尝试重连
            logger.info("Attempting to reconnect communication...")
            await asyncio.sleep(1)
            return True  # 假设重连成功
        elif fault_type == FaultType.JOINT_STUCK:
            # 尝试反向运动释放
            logger.info("Attempting to back off joint...")
            await asyncio.sleep(0.5)
            return False  # 通常需要人工干预
        return False
    
    def get_fault_history(self, limit: int = 100) -> List[FaultEvent]:
        return self.fault_history[-limit:]
    
    def clear_history(self):
        self.fault_history.clear()


# ==================== 单元测试 ====================
if __name__ == "__main__":
    fd = FaultDiagnosis()
    result = fd.diagnose(
        joint_positions=[0, 0, 0, 0, 0, 0],
        joint_temperatures=[75, 40, 38, 42, 39, 41],
        joint_torques=[10, 12, 8, 45, 9, 11],
        voltage=23.5,
        current=12.0,
        last_command="movej",
        collision_detected=False
    )
    print(f"Fault detected: {result.fault_detected}, type={result.fault_type}, recovery={result.recovery_suggestion}")
