"""
Workflow Playback Engine Module

Provides workflow recording playback:
- JSON workflow loading
- Timestamp-based replay
- Speed control (pause, resume, faster, slower)
- Step skipping
- Error handling and recovery
- Playback state management

Pure Python standard library only.
"""

from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Optional, Any, Callable


class PlaybackState(Enum):
    """Playback engine states."""
    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    PLAYING = "playing"
    PAUSED = "paused"
    RESUMING = "resuming"
    STEPPING = "stepping"
    SKIPPING = "skipping"
    ERROR = "error"
    COMPLETED = "completed"
    STOPPED = "stopped"


class StepStatus(Enum):
    """Status of a workflow step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    ERROR = "error"


@dataclass
class WorkflowStep:
    """A single step in a workflow."""
    step_id: str
    step_type: str
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    duration: float = 0.0
    delay_before: float = 0.0
    delay_after: float = 0.0
    status: StepStatus = StepStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 0
    skip_on_failure: bool = False
    screenshot_before: Optional[str] = None
    screenshot_after: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_type": self.step_type,
            "action": self.action,
            "parameters": self.parameters,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "delay_before": self.delay_before,
            "delay_after": self.delay_after,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "skip_on_failure": self.skip_on_failure,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkflowStep:
        return cls(
            step_id=data["step_id"],
            step_type=data.get("step_type", "action"),
            action=data.get("action", ""),
            parameters=data.get("parameters", {}),
            timestamp=data.get("timestamp", 0.0),
            duration=data.get("duration", 0.0),
            delay_before=data.get("delay_before", 0.0),
            delay_after=data.get("delay_after", 0.0),
            max_retries=data.get("max_retries", 0),
            skip_on_failure=data.get("skip_on_failure", False),
            metadata=data.get("metadata", {}),
        )


@dataclass
class WorkflowData:
    """Complete workflow data structure."""
    workflow_id: str
    name: str
    version: str = "1.0"
    description: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    tags: List[str] = field(default_factory=list)
    steps: List[WorkflowStep] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tags": self.tags,
            "steps": [s.to_dict() for s in self.steps],
            "variables": self.variables,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkflowData:
        steps = [WorkflowStep.from_dict(s) for s in data.get("steps", [])]
        return cls(
            workflow_id=data["workflow_id"],
            name=data.get("name", ""),
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
            created_at=data.get("created_at", 0.0),
            updated_at=data.get("updated_at", 0.0),
            tags=data.get("tags", []),
            steps=steps,
            variables=data.get("variables", {}),
            metadata=data.get("metadata", {}),
        )


class SpeedController:
    """
    Controls playback speed.

    Supports speed multipliers, pause/resume, and timing adjustments.
    """

    def __init__(self, speed: float = 1.0, min_speed: float = 0.1,
                 max_speed: float = 10.0) -> None:
        self._speed = speed
        self.min_speed = min_speed
        self.max_speed = max_speed
        self._paused_duration = 0.0
        self._pause_start: Optional[float] = None
        self._base_time: float = time.monotonic()

    @property
    def speed(self) -> float:
        return self._speed

    def set_speed(self, speed: float) -> None:
        """Set the playback speed multiplier."""
        self._speed = max(self.min_speed, min(self.max_speed, speed))

    def speed_up(self, factor: float = 1.5) -> float:
        """Increase speed."""
        self._speed = min(self.max_speed, self._speed * factor)
        return self._speed

    def slow_down(self, factor: float = 0.67) -> float:
        """Decrease speed."""
        self._speed = max(self.min_speed, self._speed * factor)
        return self._speed

    def reset_speed(self) -> float:
        """Reset speed to 1x."""
        self._speed = 1.0
        return self._speed

    def adjust_delay(self, delay: float) -> float:
        """Adjust a delay based on the current speed."""
        return delay / self._speed

    def pause(self) -> None:
        """Start pause timing."""
        self._pause_start = time.monotonic()

    def resume(self) -> None:
        """End pause timing."""
        if self._pause_start is not None:
            self._paused_duration += time.monotonic() - self._pause_start
            self._pause_start = None

    def get_elapsed(self) -> float:
        """Get elapsed wall-clock time since start."""
        elapsed = time.monotonic() - self._base_time - self._paused_duration
        if self._pause_start is not None:
            elapsed -= (time.monotonic() - self._pause_start)
        return max(0.0, elapsed)

    def is_paused(self) -> bool:
        return self._pause_start is not None


class StepExecutor:
    """
    Executes individual workflow steps.

    Maps step types to action handlers and manages step execution lifecycle.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, Callable[[WorkflowStep], Dict[str, Any]]] = {}
        self._pre_hooks: List[Callable[[WorkflowStep], None]] = []
        self._post_hooks: List[Callable[[WorkflowStep, Dict[str, Any]], None]] = []
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default step handlers."""
        self._handlers["click"] = self._handle_click
        self._handlers["type"] = self._handle_type
        self._handlers["scroll"] = self._handle_scroll
        self._handlers["wait"] = self._handle_wait
        self._handlers["navigate"] = self._handle_navigate
        self._handlers["screenshot"] = self._handle_screenshot
        self._handlers["assert"] = self._handle_assert
        self._handlers["extract"] = self._handle_extract
        self._handlers["condition"] = self._handle_condition
        self._handlers["loop"] = self._handle_loop

    def register_handler(self, step_type: str,
                         handler: Callable[[WorkflowStep], Dict[str, Any]]) -> None:
        """Register a custom step handler."""
        self._handlers[step_type] = handler

    def add_pre_hook(self, hook: Callable[[WorkflowStep], None]) -> None:
        """Add a pre-execution hook."""
        self._pre_hooks.append(hook)

    def add_post_hook(self, hook: Callable[[WorkflowStep, Dict[str, Any]], None]) -> None:
        """Add a post-execution hook."""
        self._post_hooks.append(hook)

    def execute(self, step: WorkflowStep) -> Dict[str, Any]:
        """Execute a workflow step."""
        # Pre-hooks
        for hook in self._pre_hooks:
            try:
                hook(step)
            except Exception:
                pass

        # Execute
        handler = self._handlers.get(step.step_type)
        if handler:
            result = handler(step)
        else:
            result = {"status": "skipped", "reason": f"No handler for {step.step_type}"}

        # Post-hooks
        for hook in self._post_hooks:
            try:
                hook(step, result)
            except Exception:
                pass

        return result

    def _handle_click(self, step: WorkflowStep) -> Dict[str, Any]:
        return {"status": "ok", "action": "click", "target": step.parameters.get("target", "")}

    def _handle_type(self, step: WorkflowStep) -> Dict[str, Any]:
        text = step.parameters.get("text", "")
        target = step.parameters.get("target", "")
        return {"status": "ok", "action": "type", "text": text, "target": target}

    def _handle_scroll(self, step: WorkflowStep) -> Dict[str, Any]:
        amount = step.parameters.get("amount", 0)
        direction = step.parameters.get("direction", "down")
        return {"status": "ok", "action": "scroll", "amount": amount, "direction": direction}

    def _handle_wait(self, step: WorkflowStep) -> Dict[str, Any]:
        duration = step.parameters.get("duration", 1.0)
        time.sleep(duration)
        return {"status": "ok", "action": "wait", "duration": duration}

    def _handle_navigate(self, step: WorkflowStep) -> Dict[str, Any]:
        url = step.parameters.get("url", "")
        return {"status": "ok", "action": "navigate", "url": url}

    def _handle_screenshot(self, step: WorkflowStep) -> Dict[str, Any]:
        return {"status": "ok", "action": "screenshot", "path": step.parameters.get("path", "")}

    def _handle_assert(self, step: WorkflowStep) -> Dict[str, Any]:
        condition = step.parameters.get("condition", "")
        expected = step.parameters.get("expected", True)
        return {"status": "ok", "action": "assert", "condition": condition, "expected": expected}

    def _handle_extract(self, step: WorkflowStep) -> Dict[str, Any]:
        selector = step.parameters.get("selector", "")
        attribute = step.parameters.get("attribute", "text")
        return {"status": "ok", "action": "extract", "selector": selector, "attribute": attribute}

    def _handle_condition(self, step: WorkflowStep) -> Dict[str, Any]:
        condition = step.parameters.get("condition", "")
        return {"status": "ok", "action": "condition", "condition": condition}

    def _handle_loop(self, step: WorkflowStep) -> Dict[str, Any]:
        count = step.parameters.get("count", 1)
        return {"status": "ok", "action": "loop", "count": count}


class ErrorResumeStrategy:
    """
    Error handling and resume strategies for workflow playback.

    Defines how to handle errors during step execution.
    """

    def __init__(self, max_retries: int = 3,
                 retry_delay: float = 1.0,
                 on_error: str = "stop") -> None:
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.on_error = on_error  # "stop", "skip", "retry", "ask"

    def should_retry(self, step: WorkflowStep) -> bool:
        """Check if a step should be retried."""
        if step.max_retries > 0:
            return step.retry_count < step.max_retries
        return step.retry_count < self.max_retries

    def get_retry_delay(self, step: WorkflowStep) -> float:
        """Get the delay before retrying."""
        return self.retry_delay * (2 ** step.retry_count)

    def should_skip(self, step: WorkflowStep) -> bool:
        """Check if a failed step should be skipped."""
        if step.skip_on_failure:
            return True
        return self.on_error == "skip"

    def handle_error(self, step: WorkflowStep, error: Exception) -> Dict[str, Any]:
        """Handle an error during step execution."""
        step.error = str(error)
        step.retry_count += 1

        if self.should_retry(step):
            delay = self.get_retry_delay(step)
            return {
                "action": "retry",
                "retry_count": step.retry_count,
                "delay": delay,
                "error": str(error),
            }
        elif self.should_skip(step):
            return {
                "action": "skip",
                "error": str(error),
            }
        else:
            return {
                "action": "stop",
                "error": str(error),
            }


class PlaybackEngine:
    """
    Core playback engine that orchestrates workflow execution.

    Manages the playback lifecycle, step execution, and state transitions.
    """

    def __init__(self, workflow: WorkflowData,
                 speed: float = 1.0,
                 error_strategy: Optional[ErrorResumeStrategy] = None) -> None:
        self.workflow = workflow
        self.state = PlaybackState.IDLE
        self.speed_controller = SpeedController(speed)
        self.step_executor = StepExecutor()
        self.error_strategy = error_strategy or ErrorResumeStrategy()
        self._current_step_idx: int = 0
        self._execution_log: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop_requested = False
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._start_time: Optional[float] = None
        self._callbacks: Dict[str, List[Callable]] = {
            "on_step_start": [],
            "on_step_complete": [],
            "on_step_error": [],
            "on_step_skip": [],
            "on_playback_complete": [],
            "on_playback_error": [],
            "on_state_change": [],
        }

    def register_callback(self, event: str,
                          callback: Callable[..., None]) -> None:
        """Register a playback event callback."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _emit(self, event: str, **kwargs: Any) -> None:
        """Emit a playback event."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(**kwargs)
            except Exception:
                pass

    def _set_state(self, new_state: PlaybackState) -> None:
        """Set the playback state."""
        old_state = self.state
        self.state = new_state
        self._emit("on_state_change", old_state=old_state, new_state=new_state)

    def load(self) -> bool:
        """Load and validate the workflow."""
        self._set_state(PlaybackState.LOADING)
        if not self.workflow.steps:
            self._set_state(PlaybackState.ERROR)
            return False
        self._current_step_idx = 0
        self._set_state(PlaybackState.READY)
        return True

    def play(self) -> bool:
        """Start or resume playback."""
        if self.state == PlaybackState.PAUSED:
            self._set_state(PlaybackState.RESUMING)
            self.speed_controller.resume()
            self._pause_event.set()
            self._set_state(PlaybackState.PLAYING)
            return True
        elif self.state in (PlaybackState.READY, PlaybackState.IDLE):
            self._set_state(PlaybackState.PLAYING)
            self._start_time = time.monotonic()
            self.speed_controller = SpeedController(self.speed_controller.speed)
            return True
        return False

    def pause(self) -> bool:
        """Pause playback."""
        if self.state == PlaybackState.PLAYING:
            self._set_state(PlaybackState.PAUSED)
            self.speed_controller.pause()
            self._pause_event.clear()
            return True
        return False

    def stop(self) -> bool:
        """Stop playback."""
        self._stop_requested = True
        self._pause_event.set()
        self._set_state(PlaybackState.STOPPED)
        return True

    def step_forward(self) -> bool:
        """Execute the next step and pause."""
        if self.state in (PlaybackState.READY, PlaybackState.PAUSED):
            self._set_state(PlaybackState.STEPPING)
            if self._current_step_idx < len(self.workflow.steps):
                step = self.workflow.steps[self._current_step_idx]
                self._execute_step(step)
                self._current_step_idx += 1
            self._set_state(PlaybackState.PAUSED)
            return True
        return False

    def skip_step(self) -> bool:
        """Skip the current step."""
        if self.state in (PlaybackState.PLAYING, PlaybackState.PAUSED):
            if self._current_step_idx < len(self.workflow.steps):
                step = self.workflow.steps[self._current_step_idx]
                step.status = StepStatus.SKIPPED
                self._emit("on_step_skip", step=step, index=self._current_step_idx)
                self._current_step_idx += 1
                return True
        return False

    def jump_to_step(self, step_index: int) -> bool:
        """Jump to a specific step."""
        if 0 <= step_index < len(self.workflow.steps):
            self._current_step_idx = step_index
            return True
        return False

    def _execute_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Execute a single step with error handling."""
        step.status = StepStatus.RUNNING
        self._emit("on_step_start", step=step, index=self._current_step_idx)

        # Wait for delay_before
        if step.delay_before > 0:
            adjusted_delay = self.speed_controller.adjust_delay(step.delay_before)
            self._wait_with_pause(adjusted_delay)

        # Execute with retries
        max_attempts = max(1, step.max_retries + 1)
        result: Optional[Dict[str, Any]] = None

        for attempt in range(max_attempts):
            try:
                result = self.step_executor.execute(step)
                step.status = StepStatus.COMPLETED
                step.result = result
                self._emit("on_step_complete", step=step, index=self._current_step_idx, result=result)
                break
            except Exception as e:
                error_result = self.error_strategy.handle_error(step, e)
                if error_result["action"] == "retry":
                    delay = error_result.get("delay", 1.0)
                    self._wait_with_pause(delay)
                    continue
                elif error_result["action"] == "skip":
                    step.status = StepStatus.SKIPPED
                    step.error = str(e)
                    self._emit("on_step_skip", step=step, index=self._current_step_idx)
                    break
                else:
                    step.status = StepStatus.FAILED
                    step.error = str(e)
                    self._emit("on_step_error", step=step, index=self._current_step_idx, error=e)
                    self._set_state(PlaybackState.ERROR)
                    self._emit("on_playback_error", step=step, error=e)
                    break

        # Wait for delay_after
        if step.delay_after > 0 and not self._stop_requested:
            adjusted_delay = self.speed_controller.adjust_delay(step.delay_after)
            self._wait_with_pause(adjusted_delay)

        # Log execution
        self._execution_log.append({
            "step_id": step.step_id,
            "step_type": step.step_type,
            "status": step.status.value,
            "timestamp": time.time(),
            "duration": step.duration,
            "result": result,
            "error": step.error,
        })

        return result or {}

    def _wait_with_pause(self, duration: float) -> None:
        """Wait for a duration, respecting pause and stop."""
        if duration <= 0:
            return
        end_time = time.monotonic() + duration
        while time.monotonic() < end_time:
            if self._stop_requested:
                return
            self._pause_event.wait(timeout=0.05)

    def run(self) -> Dict[str, Any]:
        """Run the full workflow playback (blocking)."""
        if not self.load():
            return {"status": "error", "message": "Failed to load workflow"}

        self.play()
        total_steps = len(self.workflow.steps)

        while (self._current_step_idx < total_steps and
               not self._stop_requested and
               self.state not in (PlaybackState.ERROR, PlaybackState.STOPPED)):

            step = self.workflow.steps[self._current_step_idx]
            self._execute_step(step)
            self._current_step_idx += 1

        if self._stop_requested:
            self._set_state(PlaybackState.STOPPED)
        elif self.state != PlaybackState.ERROR:
            self._set_state(PlaybackState.COMPLETED)
            self._emit("on_playback_complete")

        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        completed = sum(1 for s in self.workflow.steps if s.status == StepStatus.COMPLETED)
        failed = sum(1 for s in self.workflow.steps if s.status == StepStatus.FAILED)
        skipped = sum(1 for s in self.workflow.steps if s.status == StepStatus.SKIPPED)

        return {
            "status": self.state.value,
            "total_steps": total_steps,
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "elapsed_time": elapsed,
            "execution_log": self._execution_log,
        }

    def get_progress(self) -> Dict[str, Any]:
        """Get current playback progress."""
        total = len(self.workflow.steps)
        if total == 0:
            return {"progress": 0.0, "current_step": 0, "total_steps": 0}
        return {
            "progress": self._current_step_idx / total,
            "current_step": self._current_step_idx,
            "total_steps": total,
            "state": self.state.value,
            "speed": self.speed_controller.speed,
            "elapsed": self.speed_controller.get_elapsed(),
        }

    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Get the execution log."""
        return list(self._execution_log)


class WorkflowPlayer:
    """
    High-level workflow player API.

    Provides a simple interface for loading and playing workflows.
    """

    def __init__(self, speed: float = 1.0,
                 max_retries: int = 3,
                 retry_delay: float = 1.0) -> None:
        self.speed = speed
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._engine: Optional[PlaybackEngine] = None
        self._workflow: Optional[WorkflowData] = None

    def load_from_file(self, file_path: str) -> bool:
        """Load a workflow from a JSON file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._workflow = WorkflowData.from_dict(data)
            return True
        except (OSError, json.JSONDecodeError, KeyError):
            return False

    def load_from_dict(self, data: Dict[str, Any]) -> bool:
        """Load a workflow from a dictionary."""
        try:
            self._workflow = WorkflowData.from_dict(data)
            return True
        except (KeyError, TypeError):
            return False

    def load_from_json(self, json_str: str) -> bool:
        """Load a workflow from a JSON string."""
        try:
            data = json.loads(json_str)
            return self.load_from_dict(data)
        except (json.JSONDecodeError, TypeError):
            return False

    def play(self) -> Dict[str, Any]:
        """Play the loaded workflow."""
        if self._workflow is None:
            return {"status": "error", "message": "No workflow loaded"}

        error_strategy = ErrorResumeStrategy(
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
        )
        self._engine = PlaybackEngine(
            workflow=self._workflow,
            speed=self.speed,
            error_strategy=error_strategy,
        )
        return self._engine.run()

    def play_async(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> threading.Thread:
        """Play the workflow in a background thread."""
        def _run() -> None:
            result = self.play()
            if callback:
                callback(result)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return thread

    def pause(self) -> bool:
        """Pause playback."""
        return self._engine.pause() if self._engine else False

    def resume(self) -> bool:
        """Resume playback."""
        return self._engine.play() if self._engine else False

    def stop(self) -> bool:
        """Stop playback."""
        return self._engine.stop() if self._engine else False

    def set_speed(self, speed: float) -> None:
        """Set playback speed."""
        self.speed = speed
        if self._engine:
            self._engine.speed_controller.set_speed(speed)

    def get_progress(self) -> Dict[str, Any]:
        """Get playback progress."""
        return self._engine.get_progress() if self._engine else {}

    def get_workflow(self) -> Optional[WorkflowData]:
        """Get the loaded workflow."""
        return self._workflow

    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Get the execution log."""
        return self._engine.get_execution_log() if self._engine else []
