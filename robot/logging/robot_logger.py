#!/usr/bin/env python3
"""
机器人日志模块
记录机器人运行日志，支持异常回溯、实时写入、按大小轮转
"""

import time
import json
import os
import threading
from typing import Dict, Any, List, Optional
from collections import deque


class RobotLogger:
    """机器人日志记录器"""

    def __init__(self, log_dir: str = "./robot_logs", max_file_size_mb: int = 10, max_backup_files: int = 5):
        self.log_dir = log_dir
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.max_backup = max_backup_files
        self._current_file = None
        self._current_size = 0
        self._lock = threading.Lock()
        self._buffer = deque(maxlen=100)  # 内存缓冲
        self._monitor = None
        self._log_thread = None
        self._running = False
        os.makedirs(log_dir, exist_ok=True)
        self._open_log_file()

    def _open_log_file(self):
        """打开当前日志文件"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"robot_{timestamp}.log"
        self._current_file = open(os.path.join(self.log_dir, filename), 'a')
        self._current_size = os.path.getsize(self._current_file.name) if os.path.exists(self._current_file.name) else 0

    def _rotate_if_needed(self):
        """检查并执行日志轮转"""
        if self._current_size >= self.max_file_size:
            self._current_file.close()
            # 轮转：重命名并打开新文件
            old_name = self._current_file.name
            new_name = old_name.replace(".log", f"_rotated_{int(time.time())}.log")
            os.rename(old_name, new_name)
            self._open_log_file()
            # 删除旧备份
            self._cleanup_old_backups()

    def _cleanup_old_backups(self):
        """清理超过最大数量的备份文件"""
        files = [f for f in os.listdir(self.log_dir) if f.startswith("robot_") and f.endswith(".log")]
        files.sort()
        while len(files) > self.max_backup:
            os.remove(os.path.join(self.log_dir, files.pop(0)))

    def log(self, level: str, message: str, data: Dict[str, Any] = None):
        """记录一条日志"""
        entry = {
            "timestamp": time.time(),
            "iso_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "level": level,
            "message": message,
            "data": data or {}
        }
        self._buffer.append(entry)
        self._flush_buffer()

    def _flush_buffer(self):
        """将缓冲区写入文件"""
        with self._lock:
            while self._buffer:
                entry = self._buffer.popleft()
                line = json.dumps(entry) + "\n"
                self._current_file.write(line)
                self._current_size += len(line)
            self._current_file.flush()
            self._rotate_if_needed()

    def attach_monitor(self, monitor, interval: float = 1.0):
        """附加状态监控器，自动记录状态"""
        self._monitor = monitor
        self._running = True
        self._log_thread = threading.Thread(target=self._monitor_loop, args=(interval,), daemon=True)
        self._log_thread.start()

    def _monitor_loop(self, interval: float):
        """定期记录机器人状态"""
        while self._running:
            state = self._monitor.get_current_state()
            if state:
                self.log("INFO", "robot_state", state)
            time.sleep(interval)

    def stop(self):
        self._running = False
        if self._log_thread:
            self._log_thread.join(timeout=2.0)
        self._flush_buffer()
        if self._current_file:
            self._current_file.close()

    def query(self, level: str = None, start_time: float = None, end_time: float = None, limit: int = 100) -> List[Dict]:
        """查询日志（从文件读取）"""
        results = []
        files = [f for f in os.listdir(self.log_dir) if f.startswith("robot_") and f.endswith(".log")]
        files.sort(reverse=True)
        for fname in files:
            if len(results) >= limit:
                break
            with open(os.path.join(self.log_dir, fname), 'r') as f:
                for line in f:
                    if len(results) >= limit:
                        break
                    try:
                        entry = json.loads(line.strip())
                        if level and entry.get("level") != level:
                            continue
                        ts = entry.get("timestamp", 0)
                        if start_time and ts < start_time:
                            continue
                        if end_time and ts > end_time:
                            continue
                        results.append(entry)
                    except:
                        continue
        return results


# ==================== 辅助类：简单的状态监控器 ====================
class StateMonitor:
    """简单的状态监控器（用于RobotLogger.attach_monitor）"""

    def __init__(self):
        self._state = {}

    def get_current_state(self) -> Dict[str, Any]:
        """获取当前状态（子类可重写）"""
        return self._state

    def update_state(self, key: str, value: Any):
        """更新状态"""
        self._state[key] = value
