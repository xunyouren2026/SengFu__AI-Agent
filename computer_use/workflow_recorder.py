"""
工作流录制与回放模块

提供操作录制、工作流回放、自然语言生成和DSL导出功能。
仅使用Python标准库。
"""

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# RecordedAction: 录制的操作
# ============================================================

class ActionType(Enum):
    """操作类型枚举。"""
    MOUSE_MOVE = "mouse_move"
    MOUSE_CLICK = "mouse_click"
    MOUSE_DOUBLE_CLICK = "mouse_double_click"
    MOUSE_RIGHT_CLICK = "mouse_right_click"
    MOUSE_DRAG = "mouse_drag"
    MOUSE_SCROLL = "mouse_scroll"
    KEY_PRESS = "key_press"
    KEY_TYPE = "key_type"
    KEY_COMBINATION = "key_combination"
    CLIPBOARD_COPY = "clipboard_copy"
    CLIPBOARD_PASTE = "clipboard_paste"
    WINDOW_FOCUS = "window_focus"
    WINDOW_RESIZE = "window_resize"
    WINDOW_MOVE = "window_move"
    WINDOW_MINIMIZE = "window_minimize"
    WINDOW_MAXIMIZE = "window_maximize"
    WINDOW_CLOSE = "window_close"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    CUSTOM = "custom"


@dataclass
class RecordedAction:
    """录制的操作。"""

    type: str
    params: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    screenshot: Optional[str] = None  # 截图描述/路径
    window_info: Optional[Dict[str, Any]] = None  # 操作时的窗口信息
    description: str = ""  # 操作描述

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        result = {
            "type": self.type,
            "params": self.params,
            "timestamp": self.timestamp,
        }
        if self.screenshot:
            result["screenshot"] = self.screenshot
        if self.window_info:
            result["window_info"] = self.window_info
        if self.description:
            result["description"] = self.description
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecordedAction":
        """从字典创建。"""
        return cls(
            type=data["type"],
            params=data.get("params", {}),
            timestamp=data.get("timestamp", time.time()),
            screenshot=data.get("screenshot"),
            window_info=data.get("window_info"),
            description=data.get("description", ""),
        )

    @property
    def delay_from_previous(self) -> float:
        """获取与前一个操作的延迟时间（需要在列表中使用）。"""
        return 0.0


# ============================================================
# ActionRecorder: 操作录制器
# ============================================================

class ActionRecorder:
    """操作录制器。

    录制用户操作并保存为操作序列。
    """

    def __init__(self):
        """初始化操作录制器。"""
        self._actions: List[RecordedAction] = []
        self._is_recording = False
        self._start_time: Optional[float] = None
        self._last_action_time: Optional[float] = None

    @property
    def is_recording(self) -> bool:
        """是否正在录制。"""
        return self._is_recording

    @property
    def action_count(self) -> int:
        """已录制的操作数量。"""
        return len(self._actions)

    def start_recording(self) -> None:
        """开始录制。"""
        self._actions.clear()
        self._is_recording = True
        self._start_time = time.time()
        self._last_action_time = self._start_time

    def stop_recording(self) -> List[RecordedAction]:
        """停止录制。

        Returns:
            录制的操作列表
        """
        self._is_recording = False
        return list(self._actions)

    def record_action(
        self,
        action_type: str,
        params: Optional[Dict[str, Any]] = None,
        screenshot: Optional[str] = None,
        window_info: Optional[Dict[str, Any]] = None,
        description: str = "",
    ) -> RecordedAction:
        """记录一个操作。

        Args:
            action_type: 操作类型
            params: 操作参数
            screenshot: 截图描述
            window_info: 窗口信息
            description: 操作描述

        Returns:
            录制的操作
        """
        if not self._is_recording:
            raise RuntimeError("录制器未启动，请先调用start_recording()")

        current_time = time.time()

        action = RecordedAction(
            type=action_type,
            params=params or {},
            timestamp=current_time,
            screenshot=screenshot,
            window_info=window_info,
            description=description,
        )

        self._actions.append(action)
        self._last_action_time = current_time

        return action

    def record_mouse_move(self, x: int, y: int) -> RecordedAction:
        """记录鼠标移动。"""
        return self.record_action(
            ActionType.MOUSE_MOVE.value,
            params={"x": x, "y": y},
            description=f"移动鼠标到 ({x}, {y})",
        )

    def record_mouse_click(self, x: int, y: int, button: str = "left") -> RecordedAction:
        """记录鼠标点击。"""
        return self.record_action(
            ActionType.MOUSE_CLICK.value,
            params={"x": x, "y": y, "button": button},
            description=f"在 ({x}, {y}) {button}点击",
        )

    def record_mouse_drag(self, start_x: int, start_y: int, end_x: int, end_y: int) -> RecordedAction:
        """记录鼠标拖拽。"""
        return self.record_action(
            ActionType.MOUSE_DRAG.value,
            params={"start_x": start_x, "start_y": start_y, "end_x": end_x, "end_y": end_y},
            description=f"从 ({start_x}, {start_y}) 拖拽到 ({end_x}, {end_y})",
        )

    def record_key_press(self, key: str) -> RecordedAction:
        """记录按键。"""
        return self.record_action(
            ActionType.KEY_PRESS.value,
            params={"key": key},
            description=f"按下 {key}",
        )

    def record_key_type(self, text: str) -> RecordedAction:
        """记录文本输入。"""
        return self.record_action(
            ActionType.KEY_TYPE.value,
            params={"text": text},
            description=f"输入文本: {text[:50]}{'...' if len(text) > 50 else ''}",
        )

    def record_key_combination(self, keys: List[str]) -> RecordedAction:
        """记录组合键。"""
        combo = "+".join(keys)
        return self.record_action(
            ActionType.KEY_COMBINATION.value,
            params={"keys": keys},
            description=f"组合键: {combo}",
        )

    def record_wait(self, duration: float) -> RecordedAction:
        """记录等待。"""
        return self.record_action(
            ActionType.WAIT.value,
            params={"duration": duration},
            description=f"等待 {duration:.2f} 秒",
        )

    def get_actions(self) -> List[RecordedAction]:
        """获取所有录制的操作。"""
        return list(self._actions)

    def get_duration(self) -> float:
        """获取录制总时长。"""
        if not self._actions:
            return 0.0

        return self._actions[-1].timestamp - self._actions[0].timestamp

    def get_action_delays(self) -> List[float]:
        """获取操作间的时间延迟列表。"""
        if len(self._actions) < 2:
            return []

        delays = []
        for i in range(1, len(self._actions)):
            delay = self._actions[i].timestamp - self._actions[i - 1].timestamp
            delays.append(delay)

        return delays

    def clear(self) -> None:
        """清空录制。"""
        self._actions.clear()
        self._is_recording = False
        self._start_time = None
        self._last_action_time = None


# ============================================================
# WorkflowPlayer: 工作流回放器
# ============================================================

class WorkflowPlayer:
    """工作流回放器。

    加载录制的操作序列并回放。
    支持暂停、恢复和速度控制。
    """

    def __init__(self):
        """初始化工作流回放器。"""
        self._actions: List[RecordedAction] = []
        self._is_playing = False
        self._is_paused = False
        self._current_index = 0
        self._speed = 1.0  # 回放速度倍率
        self._playback_log: List[Dict[str, Any]] = []

        # 回放回调
        self._on_action_callback = None

    @property
    def is_playing(self) -> bool:
        """是否正在回放。"""
        return self._is_playing

    @property
    def is_paused(self) -> bool:
        """是否暂停。"""
        return self._is_paused

    @property
    def current_index(self) -> int:
        """当前回放位置。"""
        return self._current_index

    @property
    def speed(self) -> float:
        """回放速度。"""
        return self._speed

    def set_speed(self, speed: float) -> None:
        """设置回放速度。

        Args:
            speed: 速度倍率（0.5=半速, 1.0=正常, 2.0=两倍速）
        """
        self._speed = max(0.1, min(speed, 10.0))

    def set_on_action(self, callback) -> None:
        """设置操作回调函数。

        Args:
            callback: 回调函数，接受RecordedAction参数
        """
        self._on_action_callback = callback

    def load_workflow(self, actions: List[RecordedAction]) -> None:
        """加载工作流。

        Args:
            actions: 操作列表
        """
        self._actions = list(actions)
        self._current_index = 0
        self._playback_log.clear()

    def load_from_dicts(self, action_dicts: List[Dict[str, Any]]) -> None:
        """从字典列表加载工作流。

        Args:
            action_dicts: 操作字典列表
        """
        self._actions = [RecordedAction.from_dict(d) for d in action_dicts]
        self._current_index = 0
        self._playback_log.clear()

    def play(self) -> List[Dict[str, Any]]:
        """开始/继续回放。

        模拟回放所有操作，记录日志。

        Returns:
            回放日志列表
        """
        if not self._actions:
            return []

        self._is_playing = True
        self._is_paused = False
        self._playback_log.clear()

        while self._current_index < len(self._actions):
            if not self._is_playing:
                break

            if self._is_paused:
                time.sleep(0.01)
                continue

            action = self._actions[self._current_index]

            # 计算延迟
            if self._current_index > 0:
                prev_action = self._actions[self._current_index - 1]
                delay = (action.timestamp - prev_action.timestamp) / self._speed
                delay = min(delay, 5.0)  # 最大延迟5秒
                if delay > 0:
                    time.sleep(delay)

            # 执行操作回调
            if self._on_action_callback:
                self._on_action_callback(action)

            # 记录日志
            self._playback_log.append({
                "index": self._current_index,
                "action": action.to_dict(),
                "playback_time": time.time(),
            })

            self._current_index += 1

        self._is_playing = False
        return list(self._playback_log)

    def pause(self) -> None:
        """暂停回放。"""
        self._is_paused = True

    def resume(self) -> None:
        """恢复回放。"""
        self._is_paused = False

    def stop(self) -> None:
        """停止回放。"""
        self._is_playing = False
        self._is_paused = False

    def step_forward(self) -> Optional[RecordedAction]:
        """单步前进。

        Returns:
            当前操作
        """
        if self._current_index < len(self._actions):
            action = self._actions[self._current_index]
            self._current_index += 1
            return action
        return None

    def step_backward(self) -> Optional[RecordedAction]:
        """单步后退。

        Returns:
            前一个操作
        """
        if self._current_index > 0:
            self._current_index -= 1
            return self._actions[self._current_index]
        return None

    def jump_to(self, index: int) -> None:
        """跳转到指定位置。

        Args:
            index: 操作索引
        """
        self._current_index = max(0, min(index, len(self._actions) - 1))

    def get_playback_log(self) -> List[Dict[str, Any]]:
        """获取回放日志。"""
        return list(self._playback_log)

    def get_progress(self) -> float:
        """获取回放进度。

        Returns:
            进度百分比 (0.0 - 1.0)
        """
        if not self._actions:
            return 0.0
        return self._current_index / len(self._actions)


# ============================================================
# WorkflowGenerator: 自然语言工作流生成
# ============================================================

class WorkflowGenerator:
    """自然语言工作流生成器。

    将操作序列转换为自然语言描述，以及将自然语言转换为操作序列。
    """

    # 操作类型到自然语言的映射
    ACTION_TEMPLATES = {
        ActionType.MOUSE_MOVE.value: "移动鼠标到坐标 ({x}, {y})",
        ActionType.MOUSE_CLICK.value: "在坐标 ({x}, {y}) 点击鼠标{button}键",
        ActionType.MOUSE_DOUBLE_CLICK.value: "在坐标 ({x}, {y}) 双击鼠标",
        ActionType.MOUSE_RIGHT_CLICK.value: "在坐标 ({x}, {y}) 右键点击",
        ActionType.MOUSE_DRAG.value: "从坐标 ({start_x}, {start_y}) 拖拽到 ({end_x}, {end_y})",
        ActionType.MOUSE_SCROLL.value: "在坐标 ({x}, {y}) 滚动鼠标滚轮{amount}格",
        ActionType.KEY_PRESS.value: "按下按键 {key}",
        ActionType.KEY_TYPE.value: "输入文本 \"{text}\"",
        ActionType.KEY_COMBINATION.value: "按下组合键 {keys_str}",
        ActionType.CLIPBOARD_COPY.value: "复制内容到剪贴板",
        ActionType.CLIPBOARD_PASTE.value: "从剪贴板粘贴",
        ActionType.WINDOW_FOCUS.value: "聚焦窗口 \"{title}\"",
        ActionType.WINDOW_RESIZE.value: "调整窗口大小为 {width}x{height}",
        ActionType.WINDOW_MOVE.value: "移动窗口到 ({x}, {y})",
        ActionType.WINDOW_MINIMIZE.value: "最小化窗口",
        ActionType.WINDOW_MAXIMIZE.value: "最大化窗口",
        ActionType.WINDOW_CLOSE.value: "关闭窗口",
        ActionType.WAIT.value: "等待 {duration} 秒",
        ActionType.SCREENSHOT.value: "截取屏幕截图",
    }

    # 自然语言到操作的映射模式
    NL_PATTERNS = [
        (r"移动鼠标到\s*\(?(\d+)\s*,\s*(\d+)\)?", ActionType.MOUSE_MOVE.value, ["x", "y"]),
        (r"点击\s*\(?(\d+)\s*,\s*(\d+)\)?", ActionType.MOUSE_CLICK.value, ["x", "y"]),
        (r"双击\s*\(?(\d+)\s*,\s*(\d+)\)?", ActionType.MOUSE_DOUBLE_CLICK.value, ["x", "y"]),
        (r"右键点击?\s*\(?(\d+)\s*,\s*(\d+)\)?", ActionType.MOUSE_RIGHT_CLICK.value, ["x", "y"]),
        (r"拖拽?\s*从\s*\(?(\d+)\s*,\s*(\d+)\)?\s*到\s*\(?(\d+)\s*,\s*(\d+)\)?",
         ActionType.MOUSE_DRAG.value, ["start_x", "start_y", "end_x", "end_y"]),
        (r"按下\s*(\w+)", ActionType.KEY_PRESS.value, ["key"]),
        (r"输入\s*[\"'](.+?)[\"']", ActionType.KEY_TYPE.value, ["text"]),
        (r"等待\s*(\d+\.?\d*)\s*秒", ActionType.WAIT.value, ["duration"]),
        (r"聚焦窗口\s*[\"'](.+?)[\"']", ActionType.WINDOW_FOCUS.value, ["title"]),
        (r"最大化窗口", ActionType.WINDOW_MAXIMIZE.value, []),
        (r"最小化窗口", ActionType.WINDOW_MINIMIZE.value, []),
        (r"关闭窗口", ActionType.WINDOW_CLOSE.value, []),
    ]

    def actions_to_natural_language(
        self,
        actions: List[RecordedAction],
        include_timestamps: bool = False,
    ) -> List[str]:
        """将操作序列转换为自然语言描述。

        Args:
            actions: 操作列表
            include_timestamps: 是否包含时间戳

        Returns:
            自然语言描述列表
        """
        descriptions = []

        for i, action in enumerate(actions):
            template = self.ACTION_TEMPLATES.get(action.type, "执行操作: {type}")

            params = dict(action.params)
            # 处理组合键
            if "keys" in params and isinstance(params["keys"], list):
                params["keys_str"] = "+".join(params["keys"])

            try:
                desc = template.format(**params)
            except KeyError:
                desc = f"执行操作: {action.type}"

            if include_timestamps:
                timestamp_str = time.strftime(
                    "%H:%M:%S", time.localtime(action.timestamp)
                )
                desc = f"[{timestamp_str}] {desc}"

            descriptions.append(desc)

        return descriptions

    def natural_language_to_actions(self, text: str) -> List[RecordedAction]:
        """将自然语言描述转换为操作序列。

        Args:
            text: 自然语言文本（多行，每行一个操作）

        Returns:
            操作列表
        """
        lines = text.strip().split("\n")
        actions = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 去除时间戳前缀
            line = re.sub(r'^\[\d{2}:\d{2}:\d{2}\]\s*', '', line)

            action = self._parse_single_action(line)
            if action:
                actions.append(action)

        return actions

    def _parse_single_action(self, text: str) -> Optional[RecordedAction]:
        """解析单条自然语言为操作。

        Args:
            text: 自然语言文本

        Returns:
            操作对象
        """
        for pattern, action_type, param_names in self.NL_PATTERNS:
            match = re.search(pattern, text)
            if match:
                params = {}
                for j, name in enumerate(param_names):
                    value = match.group(j + 1)
                    # 尝试转换为数字
                    try:
                        if "." in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass
                    params[name] = value

                return RecordedAction(
                    type=action_type,
                    params=params,
                    description=text,
                )

        return None

    def generate_summary(self, actions: List[RecordedAction]) -> str:
        """生成操作摘要。

        Args:
            actions: 操作列表

        Returns:
            摘要文本
        """
        if not actions:
            return "无操作记录。"

        # 统计操作类型
        type_counts = {}
        for action in actions:
            type_counts[action.type] = type_counts.get(action.type, 0) + 1

        # 计算时长
        duration = 0.0
        if len(actions) > 1:
            duration = actions[-1].timestamp - actions[0].timestamp

        lines = [
            f"操作摘要: 共 {len(actions)} 个操作, 耗时 {duration:.1f} 秒",
            "",
            "操作统计:",
        ]

        for action_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  - {action_type}: {count} 次")

        return "\n".join(lines)


# ============================================================
# WorkflowDSL: 工作流DSL
# ============================================================

class WorkflowDSL:
    """工作流DSL（领域特定语言）。

    支持将工作流导出为DSL格式，以及从DSL导入。
    支持条件分支、循环和参数化。
    """

    def __init__(self):
        """初始化工作流DSL。"""
        self._statements: List[Dict[str, Any]] = []
        self._variables: Dict[str, Any] = {}

    def to_dsl(self, actions: List[RecordedAction]) -> str:
        """将操作序列导出为DSL。

        Args:
            actions: 操作列表

        Returns:
            DSL字符串
        """
        lines = ["# Workflow DSL", f"# Generated at {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]

        for i, action in enumerate(actions):
            params_str = json.dumps(action.params, ensure_ascii=False)
            lines.append(f"action {action.type} {params_str}")

            if action.description:
                lines.append(f"  # {action.description}")

        return "\n".join(lines)

    def from_dsl(self, dsl_text: str) -> List[RecordedAction]:
        """从DSL导入操作序列。

        Args:
            dsl_text: DSL文本

        Returns:
            操作列表
        """
        actions = []
        lines = dsl_text.strip().split("\n")

        for line in lines:
            line = line.strip()

            # 跳过注释和空行
            if not line or line.startswith("#"):
                continue

            action = self._parse_dsl_line(line)
            if action:
                actions.append(action)

        return actions

    def _parse_dsl_line(self, line: str) -> Optional[RecordedAction]:
        """解析DSL行。

        Args:
            line: DSL行

        Returns:
            操作对象
        """
        # action <type> <params_json>
        match = re.match(r'^action\s+(\S+)\s+(.+)$', line)
        if match:
            action_type = match.group(1)
            params_json = match.group(2)

            try:
                params = json.loads(params_json)
            except json.JSONDecodeError:
                params = {}

            return RecordedAction(
                type=action_type,
                params=params,
            )

        # 条件分支: if <condition>
        match = re.match(r'^if\s+(.+)$', line)
        if match:
            return RecordedAction(
                type="control_if",
                params={"condition": match.group(1)},
            )

        # 循环: loop <count>
        match = re.match(r'^loop\s+(\d+)$', line)
        if match:
            return RecordedAction(
                type="control_loop",
                params={"count": int(match.group(1))},
            )

        # 变量: set <name> = <value>
        match = re.match(r'^set\s+(\S+)\s*=\s*(.+)$', line)
        if match:
            return RecordedAction(
                type="control_set",
                params={"name": match.group(1), "value": match.group(2)},
            )

        # 等待: wait <duration>
        match = re.match(r'^wait\s+(\d+\.?\d*)$', line)
        if match:
            return RecordedAction(
                type=ActionType.WAIT.value,
                params={"duration": float(match.group(1))},
            )

        return None

    def generate_conditional_workflow(
        self,
        actions: List[RecordedAction],
        condition: str,
        else_actions: Optional[List[RecordedAction]] = None,
    ) -> str:
        """生成带条件分支的工作流DSL。

        Args:
            actions: 条件为真时的操作
            condition: 条件表达式
            else_actions: 条件为假时的操作

        Returns:
            DSL字符串
        """
        lines = [
            f"if {condition}",
            "  then:",
        ]

        for action in actions:
            params_str = json.dumps(action.params, ensure_ascii=False)
            lines.append(f"    action {action.type} {params_str}")

        if else_actions:
            lines.append("  else:")
            for action in else_actions:
                params_str = json.dumps(action.params, ensure_ascii=False)
                lines.append(f"      action {action.type} {params_str}")

        lines.append("  end")

        return "\n".join(lines)

    def generate_loop_workflow(
        self,
        actions: List[RecordedAction],
        count: int,
    ) -> str:
        """生成带循环的工作流DSL。

        Args:
            actions: 循环体操作
            count: 循环次数

        Returns:
            DSL字符串
        """
        lines = [
            f"loop {count}",
            "  do:",
        ]

        for action in actions:
            params_str = json.dumps(action.params, ensure_ascii=False)
            lines.append(f"    action {action.type} {params_str}")

        lines.append("  end")

        return "\n".join(lines)

    def generate_parameterized_workflow(
        self,
        template: str,
        parameters: Dict[str, Any],
    ) -> str:
        """生成参数化的工作流DSL。

        Args:
            template: DSL模板（使用 {param_name} 占位符）
            parameters: 参数字典

        Returns:
            DSL字符串
        """
        return template.format(**parameters)

    def set_variable(self, name: str, value: Any) -> None:
        """设置变量。"""
        self._variables[name] = value

    def get_variable(self, name: str, default: Any = None) -> Any:
        """获取变量。"""
        return self._variables.get(name, default)

    def evaluate_condition(self, condition: str) -> bool:
        """评估条件表达式。

        支持简单的比较表达式: variable == value, variable > value 等。

        Args:
            condition: 条件表达式

        Returns:
            条件是否为真
        """
        # 替换变量
        for var_name, var_value in self._variables.items():
            condition = condition.replace(f"${{{var_name}}}", str(var_value))
            condition = condition.replace(f"${var_name}", str(var_value))

        try:
            # 安全评估简单的比较表达式
            if "==" in condition:
                parts = condition.split("==", 1)
                return parts[0].strip() == parts[1].strip()
            elif "!=" in condition:
                parts = condition.split("!=", 1)
                return parts[0].strip() != parts[1].strip()
            elif ">" in condition:
                parts = condition.split(">", 1)
                try:
                    return float(parts[0].strip()) > float(parts[1].strip())
                except ValueError:
                    return False
            elif "<" in condition:
                parts = condition.split("<", 1)
                try:
                    return float(parts[0].strip()) < float(parts[1].strip())
                except ValueError:
                    return False
            else:
                return bool(condition.strip())
        except Exception:
            return False
