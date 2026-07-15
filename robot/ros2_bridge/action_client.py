"""
ROS2 Action Client Module

ROS2 Action client implementation (pure Python simulation):
- Goal sending and management
- Feedback handling
- Result retrieval
- Cancellation support
- Timeout management
- Action client lifecycle management

Pure Python standard library only.
"""

from __future__ import annotations

import time
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Optional, Any, Callable


class GoalState(Enum):
    """States of an action goal."""
    UNKNOWN = "unknown"
    ACCEPTED = "accepted"
    EXECUTING = "executing"
    CANCELING = "canceling"
    SUCCEEDED = "succeeded"
    ABORTED = "aborted"
    CANCELED = "canceled"
    REJECTED = "rejected"


class ActionClientState(Enum):
    """States of the action client."""
    UNINITIALIZED = "uninitialized"
    CONNECTING = "connecting"
    READY = "ready"
    ACTIVE = "active"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"


@dataclass
class GoalHandle:
    """Handle to an active goal."""
    goal_id: str
    goal: Dict[str, Any]
    state: GoalState = GoalState.UNKNOWN
    start_time: float = 0.0
    end_time: float = 0.0
    feedback_messages: List[Dict[str, Any]] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    acceptance_time: float = 0.0

    @property
    def is_active(self) -> bool:
        return self.state in (GoalState.ACCEPTED, GoalState.EXECUTING)

    @property
    def is_terminal(self) -> bool:
        return self.state in (GoalState.SUCCEEDED, GoalState.ABORTED,
                              GoalState.CANCELED, GoalState.REJECTED)

    @property
    def elapsed(self) -> float:
        if self.end_time > 0:
            return self.end_time - self.start_time
        return time.monotonic() - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "state": self.state.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "result": self.result,
            "feedback_count": len(self.feedback_messages),
        }


@dataclass
class ActionResult:
    """Result of an action execution."""
    goal_id: str
    success: bool
    result: Dict[str, Any] = field(default_factory=dict)
    status_code: int = 0
    status_message: str = ""
    elapsed_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "success": self.success,
            "result": self.result,
            "status_code": self.status_code,
            "status_message": self.status_message,
            "elapsed_time": self.elapsed_time,
        }


@dataclass
class FeedbackMessage:
    """Feedback message from action execution."""
    goal_id: str
    sequence: int
    progress: float = 0.0
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


class FeedbackReceiver:
    """
    Manages feedback reception and distribution.

    Buffers feedback messages and dispatches them to registered callbacks.
    """

    def __init__(self, buffer_size: int = 100) -> None:
        self.buffer_size = buffer_size
        self._callbacks: List[Callable[[FeedbackMessage], None]] = []
        self._buffer: Dict[str, List[FeedbackMessage]] = {}
        self._lock = threading.Lock()

    def register_callback(self, callback: Callable[[FeedbackMessage], None]) -> None:
        """Register a feedback callback."""
        with self._lock:
            self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[FeedbackMessage], None]) -> None:
        """Unregister a feedback callback."""
        with self._lock:
            self._callbacks = [c for c in self._callbacks if c != callback]

    def receive(self, feedback: FeedbackMessage) -> None:
        """Receive and process a feedback message."""
        with self._lock:
            if feedback.goal_id not in self._buffer:
                self._buffer[feedback.goal_id] = []
            self._buffer[feedback.goal_id].append(feedback)

            # Trim buffer
            if len(self._buffer[feedback.goal_id]) > self.buffer_size:
                self._buffer[feedback.goal_id] = \
                    self._buffer[feedback.goal_id][-self.buffer_size:]

        # Dispatch to callbacks
        for callback in self._callbacks:
            try:
                callback(feedback)
            except Exception:
                pass

    def get_feedback(self, goal_id: str) -> List[FeedbackMessage]:
        """Get all feedback for a goal."""
        with self._lock:
            return list(self._buffer.get(goal_id, []))

    def get_latest(self, goal_id: str) -> Optional[FeedbackMessage]:
        """Get the latest feedback for a goal."""
        with self._lock:
            messages = self._buffer.get(goal_id, [])
            return messages[-1] if messages else None

    def clear(self, goal_id: Optional[str] = None) -> None:
        """Clear feedback buffer."""
        with self._lock:
            if goal_id:
                self._buffer.pop(goal_id, None)
            else:
                self._buffer.clear()


class ActionClient:
    """
    ROS2-style Action client (simulated).

    Manages goal lifecycle, feedback, and result retrieval.
    """

    def __init__(self, action_name: str, action_type: str = "generic",
                 node_name: str = "action_client_node") -> None:
        self.action_name = action_name
        self.action_type = action_type
        self.node_name = node_name
        self.state = ActionClientState.UNINITIALIZED
        self.feedback_receiver = FeedbackReceiver()
        self._goal_handles: Dict[str, GoalHandle] = {}
        self._result_callbacks: Dict[str, Callable[[ActionResult], None]] = {}
        self._active_goals: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._goal_counter = 0
        self._server_simulator: Optional[_ActionServerSimulator] = None
        self._goal_timeout = 30.0

    def wait_for_server(self, timeout: float = 5.0) -> bool:
        """Wait for the action server to become available."""
        self.state = ActionClientState.CONNECTING
        time.sleep(0.1)  # Simulate connection delay
        self.state = ActionClientState.READY
        return True

    def is_server_ready(self) -> bool:
        """Check if the action server is ready."""
        return self.state in (ActionClientState.READY, ActionClientState.ACTIVE)

    def send_goal(self, goal: Dict[str, Any],
                  feedback_callback: Optional[Callable[[FeedbackMessage], None]] = None,
                  result_callback: Optional[Callable[[ActionResult], None]] = None) -> GoalHandle:
        """Send a goal to the action server."""
        with self._lock:
            self._goal_counter += 1
            goal_id = f"{self.action_name}/{self._goal_counter}/{uuid.uuid4().hex[:8]}"

        handle = GoalHandle(
            goal_id=goal_id,
            goal=goal,
            state=GoalState.UNKNOWN,
            start_time=time.monotonic(),
        )

        with self._lock:
            self._goal_handles[goal_id] = handle
            if result_callback:
                self._result_callbacks[goal_id] = result_callback
            self._active_goals[goal_id] = threading.Event()

        if feedback_callback:
            self.feedback_receiver.register_callback(feedback_callback)

        self.state = ActionClientState.ACTIVE

        # Simulate goal acceptance
        handle.state = GoalState.ACCEPTED
        handle.acceptance_time = time.monotonic()

        return handle

    def send_goal_async(self, goal: Dict[str, Any],
                        feedback_callback: Optional[Callable[[FeedbackMessage], None]] = None,
                        result_callback: Optional[Callable[[ActionResult], None]] = None) -> threading.Thread:
        """Send a goal asynchronously."""
        def _execute() -> None:
            handle = self.send_goal(goal, feedback_callback, result_callback)
            self._simulate_execution(handle)

        thread = threading.Thread(target=_execute, daemon=True)
        thread.start()
        return thread

    def cancel_goal(self, goal_handle: GoalHandle) -> bool:
        """Cancel an active goal."""
        if goal_handle.goal_id not in self._goal_handles:
            return False
        if goal_handle.is_terminal:
            return False

        goal_handle.state = GoalState.CANCELING
        time.sleep(0.1)  # Simulate cancellation delay
        goal_handle.state = GoalState.CANCELED
        goal_handle.end_time = time.monotonic()

        event = self._active_goals.get(goal_handle.goal_id)
        if event:
            event.set()

        return True

    def cancel_all_goals(self) -> int:
        """Cancel all active goals."""
        canceled = 0
        with self._lock:
            for handle in self._goal_handles.values():
                if handle.is_active:
                    handle.state = GoalState.CANCELING
                    time.sleep(0.05)
                    handle.state = GoalState.CANCELED
                    handle.end_time = time.monotonic()
                    canceled += 1
                    event = self._active_goals.get(handle.goal_id)
                    if event:
                        event.set()
        return canceled

    def get_result(self, goal_handle: GoalHandle,
                   timeout: float = 30.0) -> Optional[ActionResult]:
        """Wait for and retrieve the result of a goal."""
        event = self._active_goals.get(goal_handle.goal_id)
        if event:
            event.wait(timeout=timeout)

        if goal_handle.is_terminal:
            return ActionResult(
                goal_id=goal_handle.goal_id,
                success=(goal_handle.state == GoalState.SUCCEEDED),
                result=goal_handle.result or {},
                status_code=0 if goal_handle.state == GoalState.SUCCEEDED else 1,
                status_message=goal_handle.state.value,
                elapsed_time=goal_handle.elapsed,
            )
        return None

    def get_goal_status(self, goal_id: str) -> Optional[GoalState]:
        """Get the status of a goal."""
        handle = self._goal_handles.get(goal_id)
        return handle.state if handle else None

    def get_all_goals(self) -> List[GoalHandle]:
        """Get all goal handles."""
        return list(self._goal_handles.values())

    def get_active_goals(self) -> List[GoalHandle]:
        """Get all active (non-terminal) goals."""
        return [h for h in self._goal_handles.values() if h.is_active]

    def _simulate_execution(self, handle: GoalHandle) -> None:
        """Simulate goal execution with feedback."""
        handle.state = GoalState.EXECUTING
        total_steps = 10
        duration = handle.goal.get("duration", 2.0)
        step_duration = duration / total_steps

        for i in range(total_steps):
            if handle.state == GoalState.CANCELING:
                break

            time.sleep(step_duration)
            progress = (i + 1) / total_steps

            feedback = FeedbackMessage(
                goal_id=handle.goal_id,
                sequence=i + 1,
                progress=progress,
                message=f"Step {i + 1}/{total_steps} complete",
                data={"progress": progress, "step": i + 1},
                timestamp=time.time(),
            )
            self.feedback_receiver.receive(feedback)
            handle.feedback_messages.append(feedback.to_dict() if hasattr(feedback, 'to_dict') else {"progress": progress})

        if handle.state != GoalState.CANCELED:
            handle.state = GoalState.SUCCEEDED
            handle.result = {
                "success": True,
                "progress": 1.0,
                "message": "Goal completed successfully",
            }

        handle.end_time = time.monotonic()

        event = self._active_goals.get(handle.goal_id)
        if event:
            event.set()

        # Call result callback
        callback = self._result_callbacks.get(handle.goal_id)
        if callback:
            result = ActionResult(
                goal_id=handle.goal_id,
                success=(handle.state == GoalState.SUCCEEDED),
                result=handle.result or {},
                elapsed_time=handle.elapsed,
            )
            try:
                callback(result)
            except Exception:
                pass

    def destroy(self) -> None:
        """Clean up the action client."""
        self.cancel_all_goals()
        self.state = ActionClientState.SHUTTING_DOWN
        time.sleep(0.1)
        self.state = ActionClientState.SHUTDOWN
        self._goal_handles.clear()
        self._result_callbacks.clear()
        self._active_goals.clear()
        self.feedback_receiver.clear()


class ActionClientManager:
    """
    Manages multiple action clients.

    Provides centralized creation, tracking, and cleanup of action clients.
    """

    def __init__(self) -> None:
        self._clients: Dict[str, ActionClient] = {}
        self._lock = threading.Lock()

    def create_client(self, action_name: str,
                      action_type: str = "generic") -> ActionClient:
        """Create a new action client."""
        with self._lock:
            if action_name in self._clients:
                return self._clients[action_name]
            client = ActionClient(action_name, action_type)
            self._clients[action_name] = client
            return client

    def get_client(self, action_name: str) -> Optional[ActionClient]:
        """Get an existing action client."""
        return self._clients.get(action_name)

    def remove_client(self, action_name: str) -> bool:
        """Remove and destroy an action client."""
        with self._lock:
            client = self._clients.pop(action_name, None)
            if client:
                client.destroy()
                return True
            return False

    def destroy_all(self) -> None:
        """Destroy all action clients."""
        with self._lock:
            for client in self._clients.values():
                client.destroy()
            self._clients.clear()

    def get_all_client_names(self) -> List[str]:
        """Get all registered action client names."""
        return list(self._clients.keys())

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all action clients."""
        status: Dict[str, Dict[str, Any]] = {}
        for name, client in self._clients.items():
            status[name] = {
                "state": client.state.value,
                "active_goals": len(client.get_active_goals()),
                "total_goals": len(client.get_all_goals()),
            }
        return status


class _ActionServerSimulator:
    """Simulates an action server for testing."""

    def __init__(self) -> None:
        self._handlers: Dict[str, Callable] = {}
        self._is_running = False

    def register_handler(self, action_type: str,
                         handler: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        """Register a goal handler."""
        self._handlers[action_type] = handler

    def simulate_goal(self, goal: Dict[str, Any],
                      action_type: str) -> Dict[str, Any]:
        """Simulate goal execution."""
        handler = self._handlers.get(action_type)
        if handler:
            return handler(goal)
        return {"success": True, "message": "Simulated completion"}
