"""
配置热更新监控器模块

通过轮询文件修改时间来监控配置文件变化，
当检测到变化时触发回调通知。
"""

import copy
import json
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from .loader import ConfigLoader


class ConfigChangeEvent:
    """配置变更事件。

    当监控的配置文件发生变化时创建此事件对象。

    Args:
        file_path: 发生变化的文件路径
        old_config: 变更前的配置
        new_config: 变更后的配置
        timestamp: 变更时间戳
    """

    def __init__(
        self,
        file_path: str,
        old_config: Dict[str, Any],
        new_config: Dict[str, Any],
        timestamp: float,
    ):
        self.file_path = file_path
        self.old_config = old_config
        self.new_config = new_config
        self.timestamp = timestamp

    def get_changes(self) -> Dict[str, Any]:
        """获取新旧配置之间的差异。

        Returns:
            包含 added、removed、changed 三个键的字典
        """
        added = {}
        removed = {}
        changed = {}

        old_keys = set(self.old_config.keys())
        new_keys = set(self.new_config.keys())

        for key in new_keys - old_keys:
            added[key] = self.new_config[key]

        for key in old_keys - new_keys:
            removed[key] = self.old_config[key]

        for key in old_keys & new_keys:
            old_val = self.old_config[key]
            new_val = self.new_config[key]
            if old_val != new_val:
                changed[key] = {
                    "old": old_val,
                    "new": new_val,
                }

        return {
            "added": added,
            "removed": removed,
            "changed": changed,
        }

    def __repr__(self) -> str:
        return (
            f"ConfigChangeEvent(file_path={self.file_path!r}, "
            f"timestamp={self.timestamp})"
        )


class ConfigWatcher:
    """配置热更新监控器。

    使用 os.stat() 轮询文件修改时间来检测配置文件变化。

    Args:
        poll_interval: 轮询间隔（秒），默认为 2.0
        loader: 可选的 ConfigLoader 实例，用于自动加载变更后的配置
    """

    def __init__(
        self,
        poll_interval: float = 2.0,
        loader: Optional[ConfigLoader] = None,
    ):
        if poll_interval <= 0:
            raise ValueError(f"轮询间隔必须大于0，got: {poll_interval}")

        self.poll_interval = poll_interval
        self.loader = loader or ConfigLoader()

        # 监控的文件列表：{file_path: {"mtime": float, "config": dict}}
        self._watched_files: Dict[str, Dict[str, Any]] = {}

        # 变更回调列表
        self._callbacks: List[Callable[[ConfigChangeEvent], None]] = []

        # 后台线程
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()

        # 线程锁
        self._lock = threading.Lock()

    def watch_file(self, file_path: str) -> None:
        """注册文件监控。

        Args:
            file_path: 要监控的配置文件路径

        Raises:
            FileNotFoundError: 文件不存在
        """
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"文件不存在: {abs_path}")

        stat_info = os.stat(abs_path)
        mtime = stat_info.st_mtime

        # 尝试加载当前配置
        current_config: Dict[str, Any] = {}
        try:
            if abs_path.endswith(".json"):
                current_config = self.loader.load_from_json(abs_path)
            elif abs_path.endswith((".yaml", ".yml")):
                current_config = self.loader.load_from_yaml(abs_path)
            else:
                # 尝试JSON
                try:
                    current_config = self.loader.load_from_json(abs_path)
                except (json.JSONDecodeError, ValueError):
                    current_config = self.loader.load_from_yaml(abs_path)
        except Exception:
            current_config = {}

        with self._lock:
            self._watched_files[abs_path] = {
                "mtime": mtime,
                "config": copy.deepcopy(current_config),
            }

    def unwatch_file(self, file_path: str) -> None:
        """取消文件监控。

        Args:
            file_path: 要取消监控的文件路径
        """
        abs_path = os.path.abspath(file_path)
        with self._lock:
            if abs_path in self._watched_files:
                del self._watched_files[abs_path]

    def on_change(
        self,
        callback: Callable[[ConfigChangeEvent], None],
    ) -> None:
        """注册变更回调函数。

        当监控的配置文件发生变化时，将调用此回调。

        Args:
            callback: 回调函数，接收 ConfigChangeEvent 参数
        """
        if not callable(callback):
            raise TypeError("回调必须是可调用对象")
        self._callbacks.append(callback)

    def start_watching(self) -> None:
        """启动后台监控线程。

        如果已经在运行，则不做任何操作。
        """
        if self._running:
            return

        if not self._watched_files:
            raise RuntimeError("没有注册任何监控文件，请先调用 watch_file()")

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._watch_loop,
            name="ConfigWatcher",
            daemon=True,
        )
        self._thread.start()

    def stop_watching(self) -> None:
        """停止后台监控线程。"""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=self.poll_interval * 3)
            self._thread = None

    def check_once(self) -> List[ConfigChangeEvent]:
        """执行一次文件变化检查（非阻塞，不启动后台线程）。

        Returns:
            检测到的变更事件列表
        """
        events: List[ConfigChangeEvent] = []

        with self._lock:
            watched_copy = copy.deepcopy(self._watched_files)

        for file_path, info in watched_copy.items():
            try:
                stat_info = os.stat(file_path)
                current_mtime = stat_info.st_mtime

                if current_mtime != info["mtime"]:
                    # 文件已修改，重新加载
                    new_config: Dict[str, Any] = {}
                    try:
                        if file_path.endswith(".json"):
                            new_config = self.loader.load_from_json(file_path)
                        elif file_path.endswith((".yaml", ".yml")):
                            new_config = self.loader.load_from_yaml(file_path)
                        else:
                            try:
                                new_config = self.loader.load_from_json(
                                    file_path
                                )
                            except (json.JSONDecodeError, ValueError):
                                new_config = self.loader.load_from_yaml(
                                    file_path
                                )
                    except Exception:
                        # 加载失败，跳过此次变更
                        continue

                    old_config = info["config"]
                    event = ConfigChangeEvent(
                        file_path=file_path,
                        old_config=old_config,
                        new_config=new_config,
                        timestamp=time.time(),
                    )
                    events.append(event)

                    # 更新监控状态
                    with self._lock:
                        if file_path in self._watched_files:
                            self._watched_files[file_path] = {
                                "mtime": current_mtime,
                                "config": copy.deepcopy(new_config),
                            }

            except OSError:
                # 文件可能被删除，跳过
                continue

        return events

    def get_watched_files(self) -> List[str]:
        """获取当前监控的文件列表。"""
        with self._lock:
            return list(self._watched_files.keys())

    def is_watching(self) -> bool:
        """是否正在监控。"""
        return self._running

    def _watch_loop(self) -> None:
        """后台监控循环。"""
        while not self._stop_event.is_set():
            events = self.check_once()

            for event in events:
                self._notify_callbacks(event)

            # 等待下一次轮询
            self._stop_event.wait(self.poll_interval)

    def _notify_callbacks(self, event: ConfigChangeEvent) -> None:
        """通知所有注册的回调。"""
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception:
                # 回调异常不应中断监控
                pass

    def __repr__(self) -> str:
        return (
            f"ConfigWatcher("
            f"poll_interval={self.poll_interval}, "
            f"watched_files={len(self._watched_files)}, "
            f"callbacks={len(self._callbacks)}, "
            f"running={self._running}"
            f")"
        )

    def __enter__(self) -> "ConfigWatcher":
        """上下文管理器入口。"""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """上下文管理器退出，自动停止监控。"""
        self.stop_watching()
