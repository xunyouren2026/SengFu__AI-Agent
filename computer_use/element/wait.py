"""
Smart Element Waiting Module

Provides intelligent waiting strategies for UI elements:
- Wait for appearance, disappearance, text content
- Wait for visibility, clickability, staleness
- Polling strategies (fixed interval, exponential backoff)
- Timeout management
- Condition composition (AND, OR, NOT)
- Staleness detection

Pure Python standard library only.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Optional, Any, Callable, Set


class WaitConditionType(Enum):
    """Types of wait conditions."""
    PRESENCE = "presence"
    VISIBILITY = "visibility"
    CLICKABILITY = "clickability"
    TEXT_CONTAINS = "text_contains"
    TEXT_EQUALS = "text_equals"
    ATTRIBUTE_EQUALS = "attribute_equals"
    ATTRIBUTE_CONTAINS = "attribute_contains"
    STALENESS = "staleness"
    DISAPPEARANCE = "disappearance"
    INVISIBILITY = "invisibility"
    COUNT_EQUALS = "count_equals"
    COUNT_GREATER_THAN = "count_greater_than"
    COUNT_LESS_THAN = "count_less_than"
    CUSTOM = "custom"
    AND = "and"
    OR = "or"
    NOT = "not"


class PollingStrategyType(Enum):
    """Types of polling strategies."""
    FIXED_INTERVAL = "fixed_interval"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_INCREASE = "linear_increase"
    ADAPTIVE = "adaptive"


@dataclass
class WaitResult:
    """Result of a wait operation."""
    success: bool
    condition_type: WaitConditionType
    elapsed_time: float
    attempts: int
    error_message: str = ""
    element_data: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "TIMEOUT"
        return (f"WaitResult({status}, type={self.condition_type.value}, "
                f"elapsed={self.elapsed_time:.3f}s, attempts={self.attempts})")


@dataclass
class ElementData:
    """Simulated element data for testing."""
    element_id: str
    tag: str = "div"
    text: str = ""
    is_displayed: bool = True
    is_enabled: bool = True
    is_selected: bool = False
    attributes: Dict[str, str] = field(default_factory=dict)
    children: List[ElementData] = field(default_factory=list)
    rect: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0, "w": 100, "h": 20})
    version: int = 0

    def clone(self) -> ElementData:
        """Create a deep copy of this element."""
        return ElementData(
            element_id=self.element_id,
            tag=self.tag,
            text=self.text,
            is_displayed=self.is_displayed,
            is_enabled=self.is_enabled,
            is_selected=self.is_selected,
            attributes=dict(self.attributes),
            children=[c.clone() for c in self.children],
            rect=dict(self.rect),
            version=self.version,
        )

    def update_version(self) -> None:
        """Increment the version counter."""
        self.version += 1


class WaitCondition:
    """
    A single wait condition.

    Evaluates to True when the condition is satisfied.
    """

    def __init__(self, condition_type: WaitConditionType,
                 locator: Optional[str] = None,
                 value: Any = None,
                 timeout: float = 10.0,
                 poll_interval: float = 0.5,
                 message: str = "") -> None:
        self.condition_type = condition_type
        self.locator = locator
        self.value = value
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.message = message

    def evaluate(self, element_getter: Callable[[], Optional[ElementData]],
                 context: Optional[Dict[str, Any]] = None) -> bool:
        """Evaluate this condition against the current element state."""
        element = element_getter()

        if self.condition_type == WaitConditionType.PRESENCE:
            return element is not None

        elif self.condition_type == WaitConditionType.DISAPPEARANCE:
            return element is None

        elif self.condition_type == WaitConditionType.VISIBILITY:
            return element is not None and element.is_displayed

        elif self.condition_type == WaitConditionType.INVISIBILITY:
            return element is None or not element.is_displayed

        elif self.condition_type == WaitConditionType.CLICKABILITY:
            return (element is not None and element.is_displayed
                    and element.is_enabled)

        elif self.condition_type == WaitConditionType.TEXT_CONTAINS:
            if element is None:
                return False
            return self.value in element.text if self.value else False

        elif self.condition_type == WaitConditionType.TEXT_EQUALS:
            if element is None:
                return False
            return element.text == self.value

        elif self.condition_type == WaitConditionType.ATTRIBUTE_EQUALS:
            if element is None or not isinstance(self.value, tuple) or len(self.value) != 2:
                return False
            attr_name, attr_val = self.value
            return element.attributes.get(attr_name) == attr_val

        elif self.condition_type == WaitConditionType.ATTRIBUTE_CONTAINS:
            if element is None or not isinstance(self.value, tuple) or len(self.value) != 2:
                return False
            attr_name, attr_val = self.value
            actual = element.attributes.get(attr_name, "")
            return attr_val in actual

        elif self.condition_type == WaitConditionType.COUNT_EQUALS:
            if context and "elements" in context:
                return len(context["elements"]) == self.value
            return False

        elif self.condition_type == WaitConditionType.COUNT_GREATER_THAN:
            if context and "elements" in context:
                return len(context["elements"]) > self.value
            return False

        elif self.condition_type == WaitConditionType.COUNT_LESS_THAN:
            if context and "elements" in context:
                return len(context["elements"]) < self.value
            return False

        elif self.condition_type == WaitConditionType.STALENESS:
            if context and "previous_version" in context:
                return element is not None and element.version != context["previous_version"]
            return False

        elif self.condition_type == WaitConditionType.CUSTOM:
            if callable(self.value):
                return bool(self.value(element))
            return False

        return False

    def get_description(self) -> str:
        """Get a human-readable description of this condition."""
        descriptions = {
            WaitConditionType.PRESENCE: f"element to be present",
            WaitConditionType.DISAPPEARANCE: f"element to disappear",
            WaitConditionType.VISIBILITY: f"element to be visible",
            WaitConditionType.INVISIBILITY: f"element to be invisible",
            WaitConditionType.CLICKABILITY: f"element to be clickable",
            WaitConditionType.TEXT_CONTAINS: f"element text to contain '{self.value}'",
            WaitConditionType.TEXT_EQUALS: f"element text to equal '{self.value}'",
            WaitConditionType.STALENESS: f"element to become stale",
        }
        return descriptions.get(self.condition_type, self.message or str(self.condition_type))


class CompositeCondition:
    """
    Composable wait conditions using AND, OR, NOT logic.

    Allows combining multiple conditions into complex wait criteria.
    """

    def __init__(self, operator: str = "AND",
                 conditions: Optional[List[WaitCondition]] = None) -> None:
        self.operator = operator.upper()
        self.conditions: List[WaitCondition] = conditions or []

    def add(self, condition: WaitCondition) -> CompositeCondition:
        """Add a condition."""
        self.conditions.append(condition)
        return self

    def and_(self, condition: WaitCondition) -> CompositeCondition:
        """Add an AND condition."""
        self.operator = "AND"
        self.conditions.append(condition)
        return self

    def or_(self, condition: WaitCondition) -> CompositeCondition:
        """Add an OR condition."""
        if self.operator != "OR":
            # Wrap existing in OR
            self.operator = "OR"
        self.conditions.append(condition)
        return self

    def evaluate(self, element_getter: Callable[[], Optional[ElementData]],
                 context: Optional[Dict[str, Any]] = None) -> bool:
        """Evaluate the composite condition."""
        if not self.conditions:
            return True

        if self.operator == "AND":
            return all(c.evaluate(element_getter, context) for c in self.conditions)
        elif self.operator == "OR":
            return any(c.evaluate(element_getter, context) for c in self.conditions)
        elif self.operator == "NOT":
            return not self.conditions[0].evaluate(element_getter, context)
        elif self.operator == "NAND":
            return not all(c.evaluate(element_getter, context) for c in self.conditions)
        elif self.operator == "NOR":
            return not any(c.evaluate(element_getter, context) for c in self.conditions)
        elif self.operator == "XOR":
            results = [c.evaluate(element_getter, context) for c in self.conditions]
            return sum(results) == 1
        return False

    def get_description(self) -> str:
        """Get a human-readable description."""
        parts = [c.get_description() for c in self.conditions]
        return f" {self.operator} ".join(parts)


class PollingStrategy:
    """
    Polling strategies for wait operations.

    Determines the interval between condition checks.
    """

    def __init__(self, strategy_type: PollingStrategyType = PollingStrategyType.FIXED_INTERVAL,
                 base_interval: float = 0.5,
                 max_interval: float = 5.0,
                 multiplier: float = 2.0) -> None:
        self.strategy_type = strategy_type
        self.base_interval = base_interval
        self.max_interval = max_interval
        self.multiplier = multiplier
        self._attempt = 0
        self._last_interval = base_interval

    def reset(self) -> None:
        """Reset the polling state."""
        self._attempt = 0
        self._last_interval = self.base_interval

    def get_interval(self) -> float:
        """Get the next polling interval."""
        interval = self._compute_interval()
        self._attempt += 1
        self._last_interval = interval
        return interval

    def _compute_interval(self) -> float:
        """Compute the interval based on the strategy."""
        if self.strategy_type == PollingStrategyType.FIXED_INTERVAL:
            return self.base_interval

        elif self.strategy_type == PollingStrategyType.EXPONENTIAL_BACKOFF:
            interval = self.base_interval * (self.multiplier ** self._attempt)
            return min(interval, self.max_interval)

        elif self.strategy_type == PollingStrategyType.LINEAR_INCREASE:
            interval = self.base_interval + self._attempt * 0.1
            return min(interval, self.max_interval)

        elif self.strategy_type == PollingStrategyType.ADAPTIVE:
            # Start fast, slow down as attempts increase
            if self._attempt < 5:
                return self.base_interval * 0.5
            elif self._attempt < 10:
                return self.base_interval
            else:
                return min(self.base_interval * 1.5, self.max_interval)

        return self.base_interval


class TimeoutManager:
    """
    Manages timeout tracking for wait operations.

    Supports wall-clock timeouts and provides remaining time calculations.
    """

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.is_started = False

    def start(self) -> None:
        """Start the timeout clock."""
        self.start_time = time.monotonic()
        self.end_time = self.start_time + self.timeout
        self.is_started = True

    def remaining(self) -> float:
        """Get remaining time before timeout."""
        if not self.is_started or self.end_time is None:
            return self.timeout
        remaining = self.end_time - time.monotonic()
        return max(0.0, remaining)

    def elapsed(self) -> float:
        """Get elapsed time since start."""
        if not self.is_started or self.start_time is None:
            return 0.0
        return time.monotonic() - self.start_time

    def is_expired(self) -> bool:
        """Check if the timeout has been exceeded."""
        return self.remaining() <= 0

    def check(self) -> None:
        """Raise TimeoutError if expired."""
        if self.is_expired():
            raise TimeoutError(f"Timeout after {self.timeout:.1f} seconds")

    def reset(self, new_timeout: Optional[float] = None) -> None:
        """Reset the timeout."""
        if new_timeout is not None:
            self.timeout = new_timeout
        self.start()


class StalenessDetector:
    """
    Detects when a DOM element becomes stale.

    An element is considered stale when it is no longer attached to the DOM
    or has been replaced (version change).
    """

    def __init__(self) -> None:
        self._tracked: Dict[str, Tuple[int, float]] = {}

    def track(self, element_id: str, version: int) -> None:
        """Start tracking an element."""
        self._tracked[element_id] = (version, time.monotonic())

    def untrack(self, element_id: str) -> None:
        """Stop tracking an element."""
        self._tracked.pop(element_id, None)

    def is_stale(self, element: Optional[ElementData]) -> bool:
        """Check if an element has become stale."""
        if element is None:
            return True
        tracked = self._tracked.get(element.element_id)
        if tracked is None:
            return False
        original_version, _ = tracked
        return element.version != original_version

    def get_tracked_count(self) -> int:
        """Get the number of tracked elements."""
        return len(self._tracked)

    def clear(self) -> None:
        """Clear all tracked elements."""
        self._tracked.clear()


class ElementWait:
    """
    Smart element waiting with configurable strategies.

    Provides a high-level API for waiting on UI element conditions.
    """

    def __init__(self, timeout: float = 10.0,
                 poll_interval: float = 0.5,
                 polling_type: PollingStrategyType = PollingStrategyType.FIXED_INTERVAL,
                 ignored_exceptions: Optional[List[type]] = None) -> None:
        self.default_timeout = timeout
        self.default_poll_interval = poll_interval
        self.polling_type = polling_type
        self.ignored_exceptions = ignored_exceptions or []
        self.staleness_detector = StalenessDetector()
        self._last_result: Optional[WaitResult] = None

    def wait_for_presence(self, element_getter: Callable[[], Optional[ElementData]],
                          timeout: Optional[float] = None) -> WaitResult:
        """Wait for an element to be present in the DOM."""
        condition = WaitCondition(WaitConditionType.PRESENCE, timeout=timeout or self.default_timeout)
        return self._wait(condition, element_getter)

    def wait_for_visible(self, element_getter: Callable[[], Optional[ElementData]],
                         timeout: Optional[float] = None) -> WaitResult:
        """Wait for an element to be visible."""
        condition = WaitCondition(WaitConditionType.VISIBILITY, timeout=timeout or self.default_timeout)
        return self._wait(condition, element_getter)

    def wait_for_invisible(self, element_getter: Callable[[], Optional[ElementData]],
                           timeout: Optional[float] = None) -> WaitResult:
        """Wait for an element to become invisible."""
        condition = WaitCondition(WaitConditionType.INVISIBILITY, timeout=timeout or self.default_timeout)
        return self._wait(condition, element_getter)

    def wait_for_clickable(self, element_getter: Callable[[], Optional[ElementData]],
                           timeout: Optional[float] = None) -> WaitResult:
        """Wait for an element to be clickable (visible and enabled)."""
        condition = WaitCondition(WaitConditionType.CLICKABILITY, timeout=timeout or self.default_timeout)
        return self._wait(condition, element_getter)

    def wait_for_disappearance(self, element_getter: Callable[[], Optional[ElementData]],
                               timeout: Optional[float] = None) -> WaitResult:
        """Wait for an element to disappear from the DOM."""
        condition = WaitCondition(WaitConditionType.DISAPPEARANCE, timeout=timeout or self.default_timeout)
        return self._wait(condition, element_getter)

    def wait_for_text(self, element_getter: Callable[[], Optional[ElementData]],
                      expected_text: str, exact: bool = False,
                      timeout: Optional[float] = None) -> WaitResult:
        """Wait for an element to have specific text content."""
        ct = WaitConditionType.TEXT_EQUALS if exact else WaitConditionType.TEXT_CONTAINS
        condition = WaitCondition(ct, value=expected_text, timeout=timeout or self.default_timeout)
        return self._wait(condition, element_getter)

    def wait_for_attribute(self, element_getter: Callable[[], Optional[ElementData]],
                           attr_name: str, attr_value: str,
                           exact: bool = True,
                           timeout: Optional[float] = None) -> WaitResult:
        """Wait for an element to have a specific attribute value."""
        ct = WaitConditionType.ATTRIBUTE_EQUALS if exact else WaitConditionType.ATTRIBUTE_CONTAINS
        condition = WaitCondition(ct, value=(attr_name, attr_value),
                                 timeout=timeout or self.default_timeout)
        return self._wait(condition, element_getter)

    def wait_for_staleness(self, element_getter: Callable[[], Optional[ElementData]],
                           timeout: Optional[float] = None) -> WaitResult:
        """Wait for an element to become stale."""
        element = element_getter()
        if element:
            self.staleness_detector.track(element.element_id, element.version)

        condition = WaitCondition(WaitConditionType.STALENESS,
                                 timeout=timeout or self.default_timeout)

        def getter() -> Optional[ElementData]:
            el = element_getter()
            if el and el.element_id in self.staleness_detector._tracked:
                return el
            return None

        context = {"previous_version": element.version if element else 0}
        return self._wait(condition, getter, context)

    def wait_for_count(self, elements_getter: Callable[[], List[ElementData]],
                       expected_count: int,
                       comparison: str = "equals",
                       timeout: Optional[float] = None) -> WaitResult:
        """Wait for a specific number of elements."""
        if comparison == "equals":
            ct = WaitConditionType.COUNT_EQUALS
        elif comparison == "greater_than":
            ct = WaitConditionType.COUNT_GREATER_THAN
        else:
            ct = WaitConditionType.COUNT_LESS_THAN

        condition = WaitCondition(ct, value=expected_count,
                                 timeout=timeout or self.default_timeout)

        def single_getter() -> Optional[ElementData]:
            elements = elements_getter()
            return elements[0] if elements else None

        context = {"elements": elements_getter()}
        return self._wait(condition, single_getter, context)

    def wait_for_custom(self, element_getter: Callable[[], Optional[ElementData]],
                        predicate: Callable[[Optional[ElementData]], bool],
                        timeout: Optional[float] = None) -> WaitResult:
        """Wait for a custom condition."""
        condition = WaitCondition(WaitConditionType.CUSTOM, value=predicate,
                                 timeout=timeout or self.default_timeout)
        return self._wait(condition, element_getter)

    def wait_for_composite(self, element_getter: Callable[[], Optional[ElementData]],
                           composite: CompositeCondition,
                           timeout: Optional[float] = None) -> WaitResult:
        """Wait for a composite condition."""
        polling = PollingStrategy(self.polling_type, self.default_poll_interval)
        timeout_mgr = TimeoutManager(timeout or self.default_timeout)
        timeout_mgr.start()
        polling.reset()
        attempts = 0

        while not timeout_mgr.is_expired():
            if composite.evaluate(element_getter):
                return WaitResult(
                    success=True,
                    condition_type=WaitConditionType.CUSTOM,
                    elapsed_time=timeout_mgr.elapsed(),
                    attempts=attempts,
                )
            attempts += 1
            interval = polling.get_interval()
            if timeout_mgr.remaining() < interval:
                break
            time.sleep(interval)

        return WaitResult(
            success=False,
            condition_type=WaitConditionType.CUSTOM,
            elapsed_time=timeout_mgr.elapsed(),
            attempts=attempts,
            error_message=f"Composite condition not met: {composite.get_description()}",
        )

    def _wait(self, condition: WaitCondition,
              element_getter: Callable[[], Optional[ElementData]],
              context: Optional[Dict[str, Any]] = None) -> WaitResult:
        """Internal wait implementation."""
        polling = PollingStrategy(self.polling_type, self.default_poll_interval)
        timeout_mgr = TimeoutManager(condition.timeout)
        timeout_mgr.start()
        polling.reset()
        attempts = 0
        last_error = ""

        while not timeout_mgr.is_expired():
            try:
                if condition.evaluate(element_getter, context):
                    element = element_getter()
                    result = WaitResult(
                        success=True,
                        condition_type=condition.condition_type,
                        elapsed_time=timeout_mgr.elapsed(),
                        attempts=attempts,
                        element_data=self._element_to_dict(element),
                    )
                    self._last_result = result
                    return result
            except Exception as e:
                if not any(isinstance(e, exc) for exc in self.ignored_exceptions):
                    last_error = str(e)

            attempts += 1
            interval = polling.get_interval()
            remaining = timeout_mgr.remaining()
            if remaining < interval:
                break
            time.sleep(interval)

        result = WaitResult(
            success=False,
            condition_type=condition.condition_type,
            elapsed_time=timeout_mgr.elapsed(),
            attempts=attempts,
            error_message=f"Timed out waiting for {condition.get_description()}. {last_error}",
        )
        self._last_result = result
        return result

    def _element_to_dict(self, element: Optional[ElementData]) -> Optional[Dict[str, Any]]:
        """Convert element data to a dictionary."""
        if element is None:
            return None
        return {
            "element_id": element.element_id,
            "tag": element.tag,
            "text": element.text,
            "is_displayed": element.is_displayed,
            "is_enabled": element.is_enabled,
            "attributes": dict(element.attributes),
            "version": element.version,
        }

    def get_last_result(self) -> Optional[WaitResult]:
        """Get the result of the last wait operation."""
        return self._last_result
