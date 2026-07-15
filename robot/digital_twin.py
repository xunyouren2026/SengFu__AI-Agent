#!/usr/bin/env python3
"""
数字孪生模块 - 生产级实现
功能：实时同步真实机器人 → 虚拟模型，偏差计算与校正，轨迹录制与回放
支持：Gazebo、PyBullet、自定义虚拟控制器
"""

import asyncio
import logging
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from collections import deque
import math

from .controller_base import RobotControllerBase

logger = logging.getLogger(__name__)


def vec_norm(v: List[float]) -> float:
    """计算向量欧氏范数"""
    return math.sqrt(sum(x * x for x in v))


def vec_sub(a: List[float], b: List[float]) -> List[float]:
    """向量减法"""
    return [a[i] - b[i] for i in range(len(a))]


def vec_mean(vectors: List[List[float]]) -> List[float]:
    """计算向量平均值"""
    if not vectors:
        return [0.0] * len(vectors[0]) if vectors else []
    n = len(vectors)
    dim = len(vectors[0])
    result = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            result[i] += v[i]
    return [x / n for x in result]


@dataclass
class TwinSnapshot:
    """孪生数据快照"""
    timestamp: float
    real_joints: List[float]
    real_pose: Tuple[float, ...]
    virtual_joints: List[float]
    virtual_pose: Tuple[float, ...]
    deviation: float  # 位置偏差（欧氏距离）


class DigitalTwin:
    """
    数字孪生同步器
    支持：
    - 实时同步真实机器人与虚拟模型
    - 偏差计算（关节空间、笛卡尔空间）
    - 自动偏差校正（基于历史数据）
    - 轨迹录制与回放
    """

    def __init__(self, real_robot: RobotControllerBase, virtual_robot: RobotControllerBase,
                 sync_interval: float = 0.05, deviation_threshold: float = 0.01):
        """
        :param real_robot: 真实机器人控制器
        :param virtual_robot: 虚拟机器人控制器（仿真）
        :param sync_interval: 同步周期（秒）
        :param deviation_threshold: 偏差阈值（米），超过则触发校正
        """
        self.real = real_robot
        self.virtual = virtual_robot
        self.sync_interval = sync_interval
        self.deviation_threshold = deviation_threshold
        self._running = False
        self._sync_task: Optional[asyncio.Task] = None
        self._history: deque = deque(maxlen=10000)  # 历史快照
        self._bias: Optional[List[float]] = None     # 位置偏差（平移向量）
        self._joint_bias: Optional[List[float]] = None  # 关节偏差
        self._recording = False
        self._recorded_trajectory: List[Tuple[float, List[float]]] = []  # (timestamp, joint_positions)

    # ==================== 同步控制 ====================
    async def start_sync(self):
        """启动连续同步任务"""
        if self._running:
            return
        self._running = True
        self._sync_task = asyncio.create_task(self._sync_loop())
        logger.info("Digital twin sync started")

    async def stop_sync(self):
        """停止同步"""
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        logger.info("Digital twin sync stopped")

    async def sync_once(self):
        """单次同步：读取真实状态，设置虚拟状态"""
        try:
            real_joints = self.real.get_joint_positions()
            real_pose = self.real.get_tcp_pose()
            self.virtual.move_joint(real_joints)
            self.virtual.move_cartesian(real_pose)

            # 获取虚拟状态（用于偏差计算）
            virtual_joints = self.virtual.get_joint_positions()
            virtual_pose = self.virtual.get_tcp_pose()

            # 计算偏差
            deviation = self._compute_deviation(real_pose, virtual_pose)

            # 记录快照
            snapshot = TwinSnapshot(
                timestamp=time.time(),
                real_joints=real_joints.copy(),
                real_pose=real_pose,
                virtual_joints=virtual_joints.copy(),
                virtual_pose=virtual_pose,
                deviation=deviation
            )
            self._history.append(snapshot)

            # 如果偏差超限，触发校正
            if deviation > self.deviation_threshold and self._bias is not None:
                await self.correct()

        except Exception as e:
            logger.exception(f"Sync error: {e}")

    async def _sync_loop(self):
        """连续同步循环"""
        while self._running:
            start = time.time()
            await self.sync_once()
            elapsed = time.time() - start
            if elapsed < self.sync_interval:
                await asyncio.sleep(self.sync_interval - elapsed)

    # ==================== 偏差计算与校正 ====================
    def _compute_deviation(self, real_pose: Tuple, virtual_pose: Tuple) -> float:
        """计算笛卡尔空间位置偏差（欧氏距离）"""
        real_pos = list(real_pose[:3])
        virtual_pos = list(virtual_pose[:3])
        return vec_norm(vec_sub(real_pos, virtual_pos))

    def compute_bias(self, num_samples: int = 100) -> Tuple[List[float], List[float]]:
        """
        计算真实与虚拟之间的平均偏差（需在静态或运动后调用）
        返回：(位置偏差向量, 关节偏差向量)
        """
        real_positions = []
        virtual_positions = []
        real_joints_list = []
        virtual_joints_list = []

        for _ in range(num_samples):
            real_positions.append(list(self.real.get_tcp_pose()[:3]))
            virtual_positions.append(list(self.virtual.get_tcp_pose()[:3]))
            real_joints_list.append(list(self.real.get_joint_positions()))
            virtual_joints_list.append(list(self.virtual.get_joint_positions()))
            time.sleep(0.02)

        self._bias = vec_sub(vec_mean(real_positions), vec_mean(virtual_positions))
        self._joint_bias = vec_sub(vec_mean(real_joints_list), vec_mean(virtual_joints_list))
        logger.info(f"Bias computed: pos_bias={self._bias}, joint_bias={self._joint_bias}")
        return self._bias, self._joint_bias

    async def correct(self, method: str = "position"):
        """
        校正虚拟模型，使其对齐真实世界
        :param method: "position" (直接加偏差), "joint" (关节偏差), "adaptive" (逐步)
        """
        if self._bias is None:
            logger.warning("Bias not computed, call compute_bias() first")
            return

        if method == "position":
            # 获取当前虚拟位姿，加上偏差后移动
            virtual_pose = list(self.virtual.get_tcp_pose())
            corrected_pose = (
                virtual_pose[0] + self._bias[0],
                virtual_pose[1] + self._bias[1],
                virtual_pose[2] + self._bias[2],
                virtual_pose[3], virtual_pose[4], virtual_pose[5]
            )
            await self.virtual.move_cartesian(corrected_pose)
            logger.info(f"Corrected virtual model position with bias {self._bias}")
        elif method == "joint" and self._joint_bias is not None:
            virtual_joints = list(self.virtual.get_joint_positions())
            corrected_joints = [virtual_joints[i] + self._joint_bias[i] for i in range(len(virtual_joints))]
            await self.virtual.move_joint(corrected_joints)
            logger.info(f"Corrected virtual model joints with bias {self._joint_bias}")
        elif method == "adaptive":
            # 逐步校正，每次移动一小部分
            step = 0.1
            for _ in range(10):
                virtual_pose = list(self.virtual.get_tcp_pose())
                corrected_pose = (
                    virtual_pose[0] + self._bias[0] * step,
                    virtual_pose[1] + self._bias[1] * step,
                    virtual_pose[2] + self._bias[2] * step,
                    virtual_pose[3], virtual_pose[4], virtual_pose[5]
                )
                await self.virtual.move_cartesian(corrected_pose)
                await asyncio.sleep(0.1)

    # ==================== 轨迹录制与回放 ====================
    def start_recording(self):
        """开始录制真实机器人轨迹"""
        self._recording = True
        self._recorded_trajectory = []
        logger.info("Trajectory recording started")

    def stop_recording(self) -> List[Tuple[float, List[float]]]:
        """停止录制，返回录制的轨迹 (timestamp, joint_positions)"""
        self._recording = False
        logger.info(f"Trajectory recording stopped, {len(self._recorded_trajectory)} points recorded")
        return self._recorded_trajectory

    async def record_trajectory(self, duration: float, interval: float = 0.05):
        """录制指定时长的轨迹"""
        self.start_recording()
        start_time = time.time()
        while time.time() - start_time < duration:
            joints = self.real.get_joint_positions()
            self._recorded_trajectory.append((time.time(), joints.copy()))
            await asyncio.sleep(interval)
        self.stop_recording()

    async def replay_trajectory(self, trajectory: List[Tuple[float, List[float]]],
                                speed: float = 1.0, on_virtual: bool = True):
        """
        回放轨迹到虚拟机器人（或真实机器人）
        :param trajectory: 录制的时间戳+关节位置列表
        :param speed: 回放速度倍数
        :param on_virtual: True回放到虚拟机器人，False回放到真实机器人
        """
        if not trajectory:
            return
        target = self.virtual if on_virtual else self.real
        start_time = time.time()
        first_ts = trajectory[0][0]
        for ts, joints in trajectory:
            elapsed_sim = (ts - first_ts) / speed
            now = time.time()
            wait = start_time + elapsed_sim - now
            if wait > 0:
                await asyncio.sleep(wait)
            target.move_joint(joints)
        logger.info("Trajectory replay finished")

    # ==================== 偏差报告与导出 ====================
    def get_deviation_history(self) -> List[float]:
        """获取历史偏差序列"""
        return [s.deviation for s in self._history]

    def export_snapshots(self, filepath: str):
        """导出快照数据为 CSV"""
        import csv
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "real_joints", "real_pose", "virtual_joints", "virtual_pose", "deviation"])
            for s in self._history:
                writer.writerow([
                    s.timestamp,
                    s.real_joints,
                    s.real_pose,
                    s.virtual_joints,
                    s.virtual_pose,
                    s.deviation
                ])
        logger.info(f"Snapshots exported to {filepath}")


# ==================== 单元测试建议 ====================
if __name__ == "__main__":
    # 需要 MockRobotController 等
    from .controller_base import MockRobotController
    real = MockRobotController("real")
    virtual = MockRobotController("virtual")
    twin = DigitalTwin(real, virtual)
    asyncio.run(twin.compute_bias(10))
    asyncio.run(twin.start_sync())
    # 运行一段时间后停止
    # asyncio.sleep(5)
    # asyncio.run(twin.stop_sync())
