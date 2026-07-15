#!/usr/bin/env python3
"""
能源管理模块 - 生产级实现
功能：实时功耗采集、节能策略（动态调速、空闲休眠）、成本统计、峰值功率限制
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class PowerSample:
    """功耗采样点"""
    timestamp: float
    voltage: float
    current: float
    power: float
    joint_torques: List[float] = field(default_factory=list)


def mean(data: List[float]) -> float:
    """计算列表平均值（标准库实现）"""
    if not data:
        return 0.0
    return sum(data) / len(data)


class EnergyManager:
    """
    能源管理器
    支持：
    - 实时功耗监控（电压、电流、功率）
    - 节能策略：动态调整运动速度、空闲休眠、功率限制
    - 成本统计（基于电价）
    - 峰值功率管理
    """

    def __init__(self, robot, config: Dict = None):
        """
        :param robot: 机器人控制器（需支持获取电压/电流，或通过传感器）
        :param config: 配置字典
        """
        self.robot = robot
        self.config = config or {}
        self.electricity_rate_per_kwh = self.config.get("rate_per_kwh", 0.12)  # USD
        self.idle_power_threshold = self.config.get("idle_power_threshold", 50.0)  # W
        self.sleep_after_idle_seconds = self.config.get("sleep_after_idle_seconds", 300)
        self.max_power_limit = self.config.get("max_power_limit", 500.0)  # W
        self.speed_reduction_factor = self.config.get("speed_reduction_factor", 0.7)

        self._samples: deque = deque(maxlen=10000)
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._idle_start_time: Optional[float] = None
        self._total_energy_kwh = 0.0
        self._last_sample_time = time.time()

    # ==================== 功耗采集 ====================
    async def sample_power(self):
        """采集当前功耗（从机器人或估算）"""
        try:
            # 尝试从机器人获取电压/电流
            if hasattr(self.robot, 'get_voltage') and hasattr(self.robot, 'get_current'):
                voltage = self.robot.get_voltage()
                current = self.robot.get_current()
            else:
                # 估算：基于关节力矩（简单模型）
                torques = self.robot.get_joint_torques() if hasattr(self.robot, 'get_joint_torques') else []
                total_torque = sum(abs(t) for t in torques)
                voltage = 24.0  # 假设
                current = 0.5 + total_torque / 50.0  # 粗略估算
            power = voltage * current
            sample = PowerSample(
                timestamp=time.time(),
                voltage=voltage,
                current=current,
                power=power,
                joint_torques=self.robot.get_joint_torques() if hasattr(self.robot, 'get_joint_torques') else []
            )
            self._samples.append(sample)

            # 累计能耗
            now = time.time()
            dt = now - self._last_sample_time
            self._total_energy_kwh += power * dt / 3600000.0  # 瓦秒 → 千瓦时
            self._last_sample_time = now
            return sample
        except Exception as e:
            logger.error(f"Power sampling error: {e}")
            return None

    async def start_monitoring(self, interval: float = 0.5):
        """启动功耗监控任务"""
        if self._running:
            return
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval))
        logger.info("Energy monitoring started")

    async def stop_monitoring(self):
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Energy monitoring stopped")

    async def _monitor_loop(self, interval: float):
        while self._running:
            await self.sample_power()
            await self._apply_energy_strategies()
            await asyncio.sleep(interval)

    # ==================== 节能策略 ====================
    async def _apply_energy_strategies(self):
        """应用节能策略（动态调速、休眠）"""
        if len(self._samples) < 10:
            return
        recent_powers = [s.power for s in self._samples][-10:]
        avg_power = sum(recent_powers) / len(recent_powers)

        # 1. 峰值功率限制：如果超过限制，降低速度
        if avg_power > self.max_power_limit:
            logger.warning(f"Power limit exceeded: {avg_power:.1f}W > {self.max_power_limit}W, reducing speed")
            await self._reduce_speed()

        # 2. 空闲检测：功率低于阈值且无运动
        if avg_power < self.idle_power_threshold:
            if self._idle_start_time is None:
                self._idle_start_time = time.time()
            elif time.time() - self._idle_start_time > self.sleep_after_idle_seconds:
                await self._enter_sleep_mode()
        else:
            self._idle_start_time = None

    async def _reduce_speed(self):
        """动态降低运动速度"""
        # 假设机器人有 set_speed_scale 方法
        if hasattr(self.robot, 'set_speed_scale'):
            current_scale = getattr(self.robot, 'speed_scale', 1.0)
            new_scale = current_scale * self.speed_reduction_factor
            self.robot.set_speed_scale(max(0.2, new_scale))
            logger.info(f"Speed scale reduced to {new_scale:.2f}")

    async def _enter_sleep_mode(self):
        """进入低功耗睡眠模式"""
        logger.info("Entering sleep mode due to long idle")
        if hasattr(self.robot, 'sleep'):
            self.robot.sleep()
        # 停止运动
        self.robot.stop()
        # 等待唤醒信号（这里简化，实际需外部触发）
        await asyncio.sleep(10)
        if hasattr(self.robot, 'wakeup'):
            self.robot.wakeup()
        logger.info("Waking up from sleep mode")
        self._idle_start_time = None

    # ==================== 速度优化（基于能耗） ====================
    async def optimize_motion_velocity(self, target_velocity: float) -> float:
        """
        根据当前能耗推荐最佳速度
        :return: 调整后的速度
        """
        if len(self._samples) < 5:
            return target_velocity
        avg_power = mean([s.power for s in self._samples][-5:])
        if avg_power > 300:
            return target_velocity * 0.7
        elif avg_power < 100:
            return min(target_velocity * 1.2, 1.0)
        return target_velocity

    # ==================== 统计与报表 ====================
    def get_power_statistics(self) -> Dict:
        """获取功耗统计"""
        if not self._samples:
            return {}
        powers = [s.power for s in self._samples]
        return {
            "avg_power_w": mean(powers),
            "max_power_w": max(powers),
            "min_power_w": min(powers),
            "total_energy_kwh": self._total_energy_kwh,
            "estimated_cost_usd": self._total_energy_kwh * self.electricity_rate_per_kwh,
            "sample_count": len(self._samples)
        }

    def get_power_curve(self, duration_seconds: float) -> List[Tuple[float, float]]:
        """获取最近一段时间内的功率曲线 (timestamp, power)"""
        now = time.time()
        cutoff = now - duration_seconds
        return [(s.timestamp, s.power) for s in self._samples if s.timestamp > cutoff]

    def reset_energy_counter(self):
        self._total_energy_kwh = 0.0
        self._last_sample_time = time.time()


# ==================== 单元测试 ====================
if __name__ == "__main__":
    from .controller_base import MockRobotController
    robot = MockRobotController()
    em = EnergyManager(robot)
    async def test():
        await em.start_monitoring(interval=0.2)
        await asyncio.sleep(5)
        await em.stop_monitoring()
        print(em.get_power_statistics())
    asyncio.run(test())
