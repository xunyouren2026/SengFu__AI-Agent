"""
Browser Action Recorder Module

Records and replays browser actions for automation and testing.
Supports click, input, navigation, scroll, and other common actions.
"""

from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import json
import time
import base64
import logging

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of browser actions that can be recorded."""
    CLICK = "click"
    INPUT = "input"
    NAVIGATE = "navigate"
    SCROLL = "scroll"
    SELECT = "select"
    HOVER = "hover"
    KEYPRESS = "keypress"
    SUBMIT = "submit"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    CUSTOM = "custom"


@dataclass
class RecordedBrowserAction:
    """
    Represents a recorded browser action.
    
    Attributes:
        type: The type of action performed
        selector: CSS selector of the target element (if applicable)
        value: Value associated with the action (text input, URL, etc.)
        timestamp: Unix timestamp when the action was recorded
        screenshot: Optional base64-encoded screenshot
        metadata: Additional metadata about the action
    """
    type: ActionType
    selector: Optional[str] = None
    value: Any = None
    timestamp: float = field(default_factory=time.time)
    screenshot: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary representation."""
        return {
            "type": self.type.value if isinstance(self.type, ActionType) else self.type,
            "selector": self.selector,
            "value": self.value,
            "timestamp": self.timestamp,
            "screenshot": self.screenshot,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RecordedBrowserAction':
        """Create action from dictionary representation."""
        action_type = data.get("type")
        if isinstance(action_type, str):
            action_type = ActionType(action_type)
        
        return cls(
            type=action_type,
            selector=data.get("selector"),
            value=data.get("value"),
            timestamp=data.get("timestamp", time.time()),
            screenshot=data.get("screenshot"),
            metadata=data.get("metadata", {}),
        )


class BrowserActionRecorder:
    """
    Records browser actions for later replay.
    
    Attaches event listeners to a page and captures user interactions
    including clicks, form inputs, navigation, and scrolling.
    
    Example:
        recorder = BrowserActionRecorder()
        await recorder.start_recording(page)
        # ... user performs actions ...
        await recorder.stop_recording()
        actions = recorder.get_actions()
    """
    
    def __init__(self, capture_screenshots: bool = False):
        """
        Initialize the action recorder.
        
        Args:
            capture_screenshots: Whether to capture screenshots with each action
        """
        self._actions: List[RecordedBrowserAction] = []
        self._page: Any = None
        self._is_recording: bool = False
        self._capture_screenshots = capture_screenshots
        self._handlers: Dict[str, Callable] = {}
        logger.info("BrowserActionRecorder initialized")
    
    async def start_recording(self, page: Any) -> None:
        """
        Start recording actions on a page.
        
        Args:
            page: The Playwright page to record actions from
        """
        if self._is_recording:
            logger.warning("Already recording, stopping previous session")
            await self.stop_recording()
        
        self._page = page
        self._actions = []
        self._is_recording = True
        
        # Set up event handlers
        await self._setup_handlers()
        
        logger.info("Started recording browser actions")
    
    async def _setup_handlers(self) -> None:
        """Set up event handlers for recording."""
        # Inject script to capture user interactions
        await self._page.evaluate("""
            () => {
                window.__actionRecorder = {
                    actions: [],
                    recordAction: function(type, selector, value, metadata) {
                        const action = {
                            type: type,
                            selector: selector,
                            value: value,
                            timestamp: Date.now() / 1000,
                            metadata: metadata || {}
                        };
                        window.__actionRecorder.actions.push(action);
                    }
                };
                
                // Click handler
                document.addEventListener('click', function(e) {
                    const selector = window.__actionRecorder.getSelector(e.target);
                    window.__actionRecorder.recordAction('click', selector, null, {
                        tagName: e.target.tagName,
                        text: e.target.textContent?.substring(0, 100)
                    });
                }, true);
                
                // Input handler
                document.addEventListener('input', function(e) {
                    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                        const selector = window.__actionRecorder.getSelector(e.target);
                        window.__actionRecorder.recordAction('input', selector, e.target.value, {
                            inputType: e.target.type,
                            tagName: e.target.tagName
                        });
                    }
                }, true);
                
                // Change handler for selects
                document.addEventListener('change', function(e) {
                    if (e.target.tagName === 'SELECT') {
                        const selector = window.__actionRecorder.getSelector(e.target);
                        const selected = Array.from(e.target.selectedOptions).map(o => o.value);
                        window.__actionRecorder.recordAction('select', selector, selected, {
                            tagName: e.target.tagName
                        });
                    }
                }, true);
                
                // Helper to get CSS selector
                window.__actionRecorder.getSelector = function(el) {
                    if (el.id) return '#' + el.id;
                    if (el.name) return el.tagName.toLowerCase() + '[name="' + el.name + '"]';
                    if (el.className) {
                        const classes = el.className.split(' ').filter(c => c).join('.');
                        if (classes) return el.tagName.toLowerCase() + '.' + classes;
                    }
                    // Fallback to path
                    let path = [];
                    let current = el;
                    while (current && current.tagName !== 'HTML') {
                        let selector = current.tagName.toLowerCase();
                        if (current.id) {
                            selector += '#' + current.id;
                            path.unshift(selector);
                            break;
                        }
                        const siblings = Array.from(current.parentNode?.children || []);
                        const sameTag = siblings.filter(s => s.tagName === current.tagName);
                        if (sameTag.length > 1) {
                            const index = sameTag.indexOf(current) + 1;
                            selector += ':nth-of-type(' + index + ')';
                        }
                        path.unshift(selector);
                        current = current.parentNode;
                    }
                    return path.join(' > ');
                };
            }
        """)
    
    async def _capture_screenshot(self) -> Optional[str]:
        """Capture a screenshot and return as base64 string."""
        if not self._capture_screenshots or not self._page:
            return None
        try:
            screenshot_bytes = await self._page.screenshot()
            return base64.b64encode(screenshot_bytes).decode('utf-8')
        except Exception as e:
            logger.warning(f"Failed to capture screenshot: {e}")
            return None
    
    async def record_action(self, action_type: ActionType, selector: Optional[str] = None,
                          value: Any = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Manually record an action.
        
        Args:
            action_type: Type of action
            selector: CSS selector of target element
            value: Associated value
            metadata: Additional metadata
        """
        if not self._is_recording:
            logger.warning("Not recording, action will not be saved")
            return
        
        screenshot = await self._capture_screenshot()
        
        action = RecordedBrowserAction(
            type=action_type,
            selector=selector,
            value=value,
            timestamp=time.time(),
            screenshot=screenshot,
            metadata=metadata or {}
        )
        
        self._actions.append(action)
        logger.debug(f"Recorded action: {action_type.value}")
    
    async def stop_recording(self) -> List[RecordedBrowserAction]:
        """
        Stop recording and return recorded actions.
        
        Returns:
            List of recorded actions
        """
        if not self._is_recording:
            logger.warning("Not currently recording")
            return self._actions
        
        # Retrieve any actions recorded by the page script
        try:
            page_actions = await self._page.evaluate("window.__actionRecorder.actions")
            for action_data in page_actions:
                action = RecordedBrowserAction.from_dict({
                    **action_data,
                    "type": action_data.get("type", "custom")
                })
                self._actions.append(action)
        except Exception as e:
            logger.warning(f"Failed to retrieve page actions: {e}")
        
        self._is_recording = False
        self._page = None
        self._handlers = {}
        
        logger.info(f"Stopped recording, captured {len(self._actions)} actions")
        return self._actions
    
    def get_actions(self) -> List[RecordedBrowserAction]:
        """Get all recorded actions."""
        return self._actions.copy()
    
    def clear_actions(self) -> None:
        """Clear all recorded actions."""
        self._actions = []
        logger.info("Cleared all recorded actions")
    
    def export_to_json(self, file_path: Optional[str] = None) -> str:
        """
        Export recorded actions to JSON.
        
        Args:
            file_path: Optional file path to save JSON
            
        Returns:
            JSON string of recorded actions
        """
        data = {
            "recorded_at": datetime.now().isoformat(),
            "action_count": len(self._actions),
            "actions": [action.to_dict() for action in self._actions]
        }
        
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
            logger.info(f"Exported actions to {file_path}")
        
        return json_str
    
    @classmethod
    def import_from_json(cls, json_str: str) -> List[RecordedBrowserAction]:
        """
        Import actions from JSON string.
        
        Args:
            json_str: JSON string containing recorded actions
            
        Returns:
            List of RecordedBrowserAction objects
        """
        data = json.loads(json_str)
        actions = [RecordedBrowserAction.from_dict(a) for a in data.get("actions", [])]
        logger.info(f"Imported {len(actions)} actions from JSON")
        return actions


class ActionReplayer:
    """
    Replays recorded browser actions.
    
    Takes a list of RecordedBrowserAction objects and replays them
    on a Playwright page with configurable speed and error handling.
    
    Example:
        replayer = ActionReplayer()
        await replayer.replay(page, recorded_actions)
    """
    
    def __init__(self, delay_between_actions: float = 0.5, 
                 continue_on_error: bool = True):
        """
        Initialize the action replayer.
        
        Args:
            delay_between_actions: Delay between actions in seconds
            continue_on_error: Whether to continue on action failure
        """
        self._delay = delay_between_actions
        self._continue_on_error = continue_on_error
        self._current_index: int = 0
        self._actions: List[RecordedBrowserAction] = []
        self._page: Any = None
        self._results: List[Dict[str, Any]] = []
        logger.info("ActionReplayer initialized")
    
    async def replay(self, page: Any, actions: List[RecordedBrowserAction],
                     start_index: int = 0) -> List[Dict[str, Any]]:
        """
        Replay a sequence of actions on a page.
        
        Args:
            page: The Playwright page to replay actions on
            actions: List of actions to replay
            start_index: Index to start replaying from
            
        Returns:
            List of replay results
        """
        self._page = page
        self._actions = actions
        self._current_index = start_index
        self._results = []
        
        logger.info(f"Starting replay of {len(actions)} actions from index {start_index}")
        
        while self._current_index < len(self._actions):
            result = await self.replay_step()
            self._results.append(result)
            
            if not result.get("success") and not self._continue_on_error:
                logger.error(f"Action failed, stopping replay: {result.get('error')}")
                break
            
            if self._delay > 0 and self._current_index < len(self._actions):
                await self._page.wait_for_timeout(int(self._delay * 1000))
        
        logger.info(f"Replay completed: {self._current_index}/{len(self._actions)} actions")
        return self._results
    
    async def replay_step(self) -> Dict[str, Any]:
        """
        Replay a single action step.
        
        Returns:
            Result dictionary with success status and details
        """
        if self._current_index >= len(self._actions):
            return {"success": False, "error": "No more actions to replay"}
        
        action = self._actions[self._current_index]
        self._current_index += 1
        
        result = {
            "index": self._current_index - 1,
            "action": action.to_dict(),
            "success": False,
            "error": None
        }
        
        try:
            await self._execute_action(action)
            result["success"] = True
            logger.debug(f"Replayed action {action.type.value}")
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Failed to replay action {action.type.value}: {e}")
        
        return result
    
    async def _execute_action(self, action: RecordedBrowserAction) -> None:
        """Execute a single action on the page."""
        action_type = action.type
        if isinstance(action_type, str):
            action_type = ActionType(action_type)
        
        selector = action.selector
        value = action.value
        
        if action_type == ActionType.CLICK:
            if selector:
                await self._page.click(selector)
            else:
                raise ValueError("Click action requires a selector")
        
        elif action_type == ActionType.INPUT:
            if selector and value is not None:
                await self._page.fill(selector, str(value))
            else:
                raise ValueError("Input action requires selector and value")
        
        elif action_type == ActionType.NAVIGATE:
            if value:
                await self._page.goto(str(value))
            else:
                raise ValueError("Navigate action requires a URL value")
        
        elif action_type == ActionType.SCROLL:
            if isinstance(value, dict):
                x = value.get("x", 0)
                y = value.get("y", 0)
                await self._page.evaluate(f"window.scrollTo({x}, {y})")
            elif selector:
                element = await self._page.query_selector(selector)
                if element:
                    await element.scroll_into_view_if_needed()
            else:
                raise ValueError("Scroll action requires selector or x/y values")
        
        elif action_type == ActionType.SELECT:
            if selector and value is not None:
                await self._page.select_option(selector, value)
            else:
                raise ValueError("Select action requires selector and value")
        
        elif action_type == ActionType.HOVER:
            if selector:
                await self._page.hover(selector)
            else:
                raise ValueError("Hover action requires a selector")
        
        elif action_type == ActionType.KEYPRESS:
            if value:
                await self._page.keyboard.press(str(value))
            else:
                raise ValueError("Keypress action requires a key value")
        
        elif action_type == ActionType.SUBMIT:
            if selector:
                await self._page.evaluate(f"document.querySelector('{selector}').submit()")
            else:
                raise ValueError("Submit action requires a selector")
        
        elif action_type == ActionType.WAIT:
            delay = float(value) if value else 1.0
            await self._page.wait_for_timeout(int(delay * 1000))
        
        elif action_type == ActionType.SCREENSHOT:
            await self._page.screenshot(path=value if value else None)
        
        elif action_type == ActionType.CUSTOM:
            # Custom actions handled via metadata
            script = action.metadata.get("script")
            if script:
                await self._page.evaluate(script)
        
        else:
            raise ValueError(f"Unknown action type: {action_type}")
    
    def skip_step(self, count: int = 1) -> None:
        """
        Skip one or more steps in the replay sequence.
        
        Args:
            count: Number of steps to skip
        """
        self._current_index = min(self._current_index + count, len(self._actions))
        logger.info(f"Skipped {count} steps, now at index {self._current_index}")
    
    def get_current_index(self) -> int:
        """Get the current replay index."""
        return self._current_index
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get all replay results."""
        return self._results.copy()
    
    def reset(self) -> None:
        """Reset the replayer to the beginning."""
        self._current_index = 0
        self._results = []
        logger.info("ActionReplayer reset")
