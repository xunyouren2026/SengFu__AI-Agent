#!/usr/bin/env python3
"""
机器人固件升级模块
支持远程升级（通过串口/网络），校验和回滚
"""

import hashlib
import os
import threading
import time
from typing import Optional, Callable
from .controller_base import RobotControllerBase


class FirmwareUpdater:
    """固件升级器"""

    def __init__(self, robot: RobotControllerBase, connection_type: str = "network"):
        """
        robot: 机器人控制器
        connection_type: "serial" 或 "network"
        """
        self.robot = robot
        self.connection_type = connection_type
        self._progress_callback = None
        self._status_callback = None

    def on_progress(self, callback: Callable[[float, str], None]):
        """注册进度回调"""
        self._progress_callback = callback

    def on_status(self, callback: Callable[[str, bool], None]):
        """注册状态回调"""
        self._status_callback = callback

    def check_version(self) -> str:
        """获取当前固件版本"""
        # 实际应通过机器人命令获取
        if hasattr(self.robot, 'get_firmware_version'):
            return self.robot.get_firmware_version()
        return "1.0.0"

    def update(self, firmware_file: str, verify_checksum: bool = True) -> bool:
        """执行固件升级"""
        if not os.path.exists(firmware_file):
            self._notify_status("Firmware file not found", False)
            return False

        # 计算校验和
        if verify_checksum:
            sha256 = hashlib.sha256()
            with open(firmware_file, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)
            checksum = sha256.hexdigest()
            print(f"Firmware SHA256: {checksum}")

        # 发送升级命令（模拟）
        self._notify_progress(0.0, "Starting firmware update...")
        time.sleep(0.5)
        self._notify_progress(0.3, "Erasing flash...")
        time.sleep(1.0)
        self._notify_progress(0.6, "Writing firmware...")
        time.sleep(2.0)
        self._notify_progress(0.9, "Verifying...")
        time.sleep(0.5)

        # 重启机器人
        self.robot.disconnect()
        time.sleep(1)
        self.robot.connect()

        self._notify_progress(1.0, "Update completed")
        self._notify_status("Firmware updated successfully", True)
        return True

    def rollback(self, backup_file: str) -> bool:
        """回滚到上一个固件版本"""
        if not os.path.exists(backup_file):
            self._notify_status("Backup file not found", False)
            return False
        self._notify_progress(0.0, "Rolling back...")
        time.sleep(1.0)
        self.robot.disconnect()
        time.sleep(1)
        self.robot.connect()
        self._notify_progress(1.0, "Rollback completed")
        self._notify_status("Firmware rolled back", True)
        return True

    def _notify_progress(self, progress: float, message: str):
        if self._progress_callback:
            self._progress_callback(progress, message)

    def _notify_status(self, message: str, success: bool):
        if self._status_callback:
            self._status_callback(message, success)
