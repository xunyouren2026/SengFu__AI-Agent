"""
macOS Quartz API Simulation Module

Simulates macOS window management APIs for cross-platform use:
- NSWindow management
- AXUIElement accessibility
- CGWindowListCopyWindowInfo
- Workspace management
- Notification posting

Pure Python standard library only.
"""

from __future__ import annotations

import enum
import time
import threading
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Callable


class NSWindowStyle(enum.IntFlag):
    """NSWindow style masks."""
    BORDERLESS = 0
    TITLED = 1 << 0
    CLOSABLE = 1 << 1
    MINIATURIZABLE = 1 << 2
    RESIZABLE = 1 << 3
    TEXTURED_BACKGROUND = 1 << 8
    UNIFIED_TITLE_AND_TOOLBAR = 1 << 12
    FULLSCREEN = 1 << 14
    UTILITY_WINDOW = 1 << 4
    DOC_MODAL_WINDOW = 1 << 6
    NONACTIVATING_PANEL = 1 << 7
    HUD_WINDOW = 1 << 13


class NSWindowLevel(enum.IntEnum):
    """NSWindow stacking levels."""
    NORMAL = 0
    FLOATING = 3
    SUBMENU = 3
    TORN_OFF_MENU = 3
    MAIN_MENU = 24
    STATUS = 25
    MODAL_PANEL = 8
    POPUP_MENU = 101
    SCREEN_SAVER = 1000


class NSBackingStoreType(enum.IntEnum):
    """NSBackingStore types."""
    RETAINED = 0
    NONRETAINED = 1
    BUFFERED = 2


class AXRole(enum.Enum):
    """Accessibility roles (AXRole)."""
    WINDOW = "AXWindow"
    BUTTON = "AXButton"
    TEXT_FIELD = "AXTextField"
    STATIC_TEXT = "AXStaticText"
    TEXT_AREA = "AXTextArea"
    SCROLL_AREA = "AXScrollArea"
    SCROLL_BAR = "AXScrollBar"
    CHECKBOX = "AXCheckBox"
    RADIO_BUTTON = "AXRadioButton"
    POPUP_BUTTON = "AXPopUpButton"
    MENU_BUTTON = "AXMenuButton"
    TABLE = "AXTable"
    TABLE_ROW = "AXTableRow"
    COLUMN = "AXColumn"
    LIST = "AXList"
    OUTLINE = "AXOutline"
    TAB_GROUP = "AXTabGroup"
    TOOLBAR = "AXToolbar"
    PROGRESS_INDICATOR = "AXProgressIndicator"
    SLIDER = "AXSlider"
    STEPPER = "AXStepper"
    IMAGE = "AXImage"
    DISCLOSURE_TRIANGLE = "AXDisclosureTriangle"
    SPLIT_GROUP = "AXSplitGroup"
    SPLITTER = "AXSplitter"
    METER = "AXMeter"
    RATING_INDICATOR = "AXRatingIndicator"
    VALUE_INDICATOR = "AXValueIndicator"
    COLOR_WELL = "AXColorWell"
    HELP_TAG = "AXHelpTag"
    PICKER = "AXPicker"
    TOGGLE = "AXToggle"
    LINK = "AXLink"
    GROUP = "AXGroup"
    RADIO_GROUP = "AXRadioGroup"
    UNKNOWN = "AXUnknown"


class AXNotification(enum.Enum):
    """Accessibility notification types."""
    TITLE_CHANGED = "AXTitleChanged"
    VALUE_CHANGED = "AXValueChanged"
    SELECTED_CHILDREN_CHANGED = "AXSelectedChildrenChanged"
    RESIZED = "AXResized"
    MOVED = "AXMoved"
    CREATED = "AXCreated"
    DESTROYED = "AXDestroyed"
    FOCUSED_UI_ELEMENT_CHANGED = "AXFocusedUIElementChanged"
    APPLICATION_DEACTIVATED = "AXApplicationDeactivated"
    APPLICATION_ACTIVATED = "AXApplicationActivated"
    WINDOW_CREATED = "AXWindowCreated"
    WINDOW_MOVED = "AXWindowMoved"
    WINDOW_RESIZED = "AXWindowResized"
    WINDOW_MINIMIZED = "AXWindowMiniaturized"
    WINDOW_DEMINIMIZED = "AXWindowDeminiaturized"
    MAIN_WINDOW_CHANGED = "AXMainWindowChanged"
    SHEET_CREATED = "AXSheetCreated"
    UI_ELEMENT_DESTROYED = "AXUIElementDestroyed"


@dataclass
class NSRect:
    """NSRect: origin and size."""
    x: float = 0.0
    y: float = 0.0
    width: float = 100.0
    height: float = 100.0

    @property
    def origin(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @property
    def size(self) -> Tuple[float, float]:
        return (self.width, self.height)

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x + self.width and \
               self.y <= py <= self.y + self.height

    def intersects(self, other: NSRect) -> bool:
        return not (self.x + self.width < other.x or other.x + other.width < self.x or
                    self.y + self.height < other.y or other.y + other.height < self.y)

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass
class CGPoint:
    """CGPoint: 2D point."""
    x: float = 0.0
    y: float = 0.0


@dataclass
class CGSize:
    """CGSize: 2D size."""
    width: float = 0.0
    height: float = 0.0


@dataclass
class WindowInfo:
    """CGWindow information."""
    window_id: int
    window_name: str = ""
    owner_name: str = ""
    owner_pid: int = 0
    bounds: NSRect = field(default_factory=NSRect)
    layer: int = 0
    is_on_screen: bool = True
    alpha: float = 1.0
    memory_usage: int = 0
    sharing_state: int = 0
    is_visible: bool = True
    window_number: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kCGWindowNumber": self.window_number,
            "kCGWindowName": self.window_name,
            "kCGWindowOwnerName": self.owner_name,
            "kCGWindowOwnerPID": self.owner_pid,
            "kCGWindowBounds": self.bounds.to_dict(),
            "kCGWindowLayer": self.layer,
            "kCGWindowIsOnscreen": self.is_on_screen,
            "kCGWindowAlpha": self.alpha,
        }


class NSWindowManager:
    """
    Simulated NSWindow management.

    Creates, manages, and manipulates macOS-style windows.
    """

    _next_window_number: int = 1
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._windows: Dict[int, Dict[str, Any]] = {}
        self._z_order: List[int] = []
        self._main_window: Optional[int] = None
        self._key_window: Optional[int] = None
        self._frontmost_app_pid: int = 0

    def create_window(self, rect: NSRect = None,
                      style: NSWindowStyle = NSWindowStyle.TITLED | NSWindowStyle.CLOSABLE | NSWindowStyle.RESIZABLE,
                      backing: NSBackingStoreType = NSBackingStoreType.BUFFERED,
                      title: str = "Untitled",
                      defer: bool = False) -> int:
        """Create a new window and return its window number."""
        with NSWindowManager._lock:
            win_num = NSWindowManager._next_window_number
            NSWindowManager._next_window_number += 1

        if rect is None:
            rect = NSRect(x=100, y=100, width=400, height=300)

        self._windows[win_num] = {
            "window_number": win_num,
            "frame": rect,
            "style": style,
            "backing": backing,
            "title": title,
            "is_visible": False,
            "is_miniaturized": False,
            "is_zoomed": False,
            "level": NSWindowLevel.NORMAL,
            "alpha": 1.0,
            "min_size": NSRect(x=0, y=0, width=100, height=100),
            "max_size": NSRect(x=0, y=0, width=10000, height=10000),
            "owner_pid": self._frontmost_app_pid,
            "creation_time": time.time(),
        }

        self._z_order.append(win_num)

        if self._main_window is None:
            self._main_window = win_num
        if self._key_window is None:
            self._key_window = win_num

        return win_num

    def close_window(self, window_number: int) -> bool:
        """Close a window."""
        if window_number not in self._windows:
            return False

        del self._windows[window_number]
        if window_number in self._z_order:
            self._z_order.remove(window_number)

        if self._main_window == window_number:
            self._main_window = self._z_order[-1] if self._z_order else None
        if self._key_window == window_number:
            self._key_window = self._z_order[-1] if self._z_order else None

        return True

    def show_window(self, window_number: int) -> bool:
        """Order a window to the front and make it visible."""
        win = self._windows.get(window_number)
        if win is None:
            return False
        win["is_visible"] = True
        win["is_miniaturized"] = False
        if window_number in self._z_order:
            self._z_order.remove(window_number)
        self._z_order.append(window_number)
        self._key_window = window_number
        return True

    def hide_window(self, window_number: int) -> bool:
        """Hide a window."""
        win = self._windows.get(window_number)
        if win is None:
            return False
        win["is_visible"] = False
        return True

    def miniaturize_window(self, window_number: int) -> bool:
        """Miniaturize (minimize) a window."""
        win = self._windows.get(window_number)
        if win is None:
            return False
        win["is_miniaturized"] = True
        win["is_visible"] = False
        return True

    def deminiaturize_window(self, window_number: int) -> bool:
        """Deminiaturize (restore) a window."""
        win = self._windows.get(window_number)
        if win is None:
            return False
        win["is_miniaturized"] = False
        win["is_visible"] = True
        return True

    def zoom_window(self, window_number: int) -> bool:
        """Zoom (maximize) a window."""
        win = self._windows.get(window_number)
        if win is None:
            return False
        win["is_zoomed"] = not win["is_zoomed"]
        if win["is_zoomed"]:
            win["frame"] = NSRect(x=0, y=0,
                                  width=1920, height=1080)
        return True

    def set_frame(self, window_number: int, frame: NSRect) -> bool:
        """Set the frame (position and size) of a window."""
        win = self._windows.get(window_number)
        if win is None:
            return False
        win["frame"] = frame
        return True

    def get_frame(self, window_number: int) -> Optional[NSRect]:
        """Get the frame of a window."""
        win = self._windows.get(window_number)
        if win is None:
            return None
        return win["frame"]

    def set_title(self, window_number: int, title: str) -> bool:
        """Set the title of a window."""
        win = self._windows.get(window_number)
        if win is None:
            return False
        win["title"] = title
        return True

    def get_title(self, window_number: int) -> str:
        """Get the title of a window."""
        win = self._windows.get(window_number)
        return win["title"] if win else ""

    def set_level(self, window_number: int, level: NSWindowLevel) -> bool:
        """Set the window level."""
        win = self._windows.get(window_number)
        if win is None:
            return False
        win["level"] = level
        return True

    def set_alpha(self, window_number: int, alpha: float) -> bool:
        """Set the window opacity."""
        win = self._windows.get(window_number)
        if win is None:
            return False
        win["alpha"] = max(0.0, min(1.0, alpha))
        return True

    def get_main_window(self) -> Optional[int]:
        """Get the main window number."""
        return self._main_window

    def get_key_window(self) -> Optional[int]:
        """Get the key (focused) window number."""
        return self._key_window

    def get_all_windows(self) -> List[int]:
        """Get all window numbers."""
        return list(self._z_order)

    def get_visible_windows(self) -> List[int]:
        """Get visible window numbers."""
        return [wn for wn in self._z_order
                if self._windows.get(wn, {}).get("is_visible", False)]


class AccessibilityElement:
    """
    Simulated AXUIElement for accessibility access.

    Represents an accessible UI element with role, value, and children.
    """

    _next_id: int = 1

    def __init__(self, role: AXRole = AXRole.UNKNOWN,
                 label: str = "", value: str = "",
                 parent: Optional[AccessibilityElement] = None,
                 element_id: Optional[int] = None) -> None:
        if element_id is not None:
            self.element_id = element_id
        else:
            self.element_id = AccessibilityElement._next_id
            AccessibilityElement._next_id += 1

        self.role = role
        self.label = label
        self.value = value
        self.parent = parent
        self.children: List[AccessibilityElement] = []
        self.attributes: Dict[str, Any] = {}
        self.actions: List[str] = []
        self.is_enabled: bool = True
        self.is_focused: bool = False
        self.position: Optional[CGPoint] = None
        self.size: Optional[CGSize] = None
        self._notification_handlers: Dict[str, List[Callable]] = {}

        if parent:
            parent.children.append(self)

    def get_attribute(self, name: str) -> Any:
        """Get an accessibility attribute."""
        attr_map = {
            "AXRole": self.role.value,
            "AXRoleDescription": self._get_role_description(),
            "AXTitle": self.label,
            "AXValue": self.value,
            "AXDescription": self.label,
            "AXEnabled": self.is_enabled,
            "AXFocused": self.is_focused,
            "AXPosition": {"x": self.position.x, "y": self.position.y} if self.position else None,
            "AXSize": {"width": self.size.width, "height": self.size.height} if self.size else None,
            "AXChildren": [c.element_id for c in self.children],
            "AXParent": self.parent.element_id if self.parent else None,
            "AXWindow": None,
            "AXTopLevelUIElement": None,
            "AXHelp": self.attributes.get("AXHelp", ""),
            "AXSelectedText": self.value,
        }
        return attr_map.get(name, self.attributes.get(name))

    def set_attribute(self, name: str, value: Any) -> bool:
        """Set an accessibility attribute."""
        if name == "AXValue":
            self.value = str(value)
        elif name == "AXFocused":
            self.is_focused = bool(value)
        elif name == "AXEnabled":
            self.is_enabled = bool(value)
        elif name == "AXPosition" and isinstance(value, dict):
            self.position = CGPoint(x=value.get("x", 0), y=value.get("y", 0))
        elif name == "AXSize" and isinstance(value, dict):
            self.size = CGSize(width=value.get("width", 0), height=value.get("height", 0))
        else:
            self.attributes[name] = value
        return True

    def perform_action(self, action: str) -> bool:
        """Perform an accessibility action."""
        if action in ("AXPress", "AXClick"):
            return True
        elif action == "AXShowMenu":
            return True
        elif action == "AXPick":
            return True
        elif action == "AXIncrement":
            try:
                self.value = str(float(self.value) + 1)
            except (ValueError, TypeError):
                pass
            return True
        elif action == "AXDecrement":
            try:
                self.value = str(float(self.value) - 1)
            except (ValueError, TypeError):
                pass
            return True
        return action in self.actions

    def get_action_names(self) -> List[str]:
        """Get available actions for this element."""
        base_actions = []
        if self.role == AXRole.BUTTON:
            base_actions = ["AXPress"]
        elif self.role in (AXRole.CHECKBOX, AXRole.TOGGLE):
            base_actions = ["AXPress"]
        elif self.role == AXRole.TEXT_FIELD:
            base_actions = ["AXSetFocused"]
        elif self.role == AXRole.SLIDER:
            base_actions = ["AXIncrement", "AXDecrement"]
        return base_actions + self.actions

    def register_notification(self, notification: str,
                              handler: Callable[[str, Any], None]) -> None:
        """Register a notification handler."""
        if notification not in self._notification_handlers:
            self._notification_handlers[notification] = []
        self._notification_handlers[notification].append(handler)

    def post_notification(self, notification: str, info: Any = None) -> None:
        """Post a notification to handlers."""
        handlers = self._notification_handlers.get(notification, [])
        for handler in handlers:
            try:
                handler(notification, info)
            except Exception:
                pass

    def _get_role_description(self) -> str:
        """Get a human-readable role description."""
        descriptions = {
            AXRole.WINDOW: "window",
            AXRole.BUTTON: "button",
            AXRole.TEXT_FIELD: "text field",
            AXRole.STATIC_TEXT: "text",
            AXRole.TEXT_AREA: "text area",
            AXRole.CHECKBOX: "checkbox",
            AXRole.RADIO_BUTTON: "radio button",
            AXRole.TABLE: "table",
            AXRole.LIST: "list",
            AXRole.IMAGE: "image",
            AXRole.SLIDER: "slider",
            AXRole.PROGRESS_INDICATOR: "progress indicator",
            AXRole.LINK: "link",
            AXRole.GROUP: "group",
        }
        return descriptions.get(self.role, "unknown")

    def find_element_by_role(self, role: AXRole) -> List[AccessibilityElement]:
        """Find all descendants with a given role."""
        results: List[AccessibilityElement] = []
        if self.role == role:
            results.append(self)
        for child in self.children:
            results.extend(child.find_element_by_role(role))
        return results

    def find_element_by_label(self, label: str) -> Optional[AccessibilityElement]:
        """Find a descendant with a given label."""
        if self.label == label:
            return self
        for child in self.children:
            found = child.find_element_by_label(label)
            if found:
                return found
        return None


class WorkspaceManager:
    """
    Simulated macOS workspace (Mission Control) management.

    Manages virtual desktops and application spaces.
    """

    def __init__(self) -> None:
        self._spaces: List[Dict[str, Any]] = [
            {"id": 1, "name": "Desktop 1", "is_visible": True},
        ]
        self._current_space_id: int = 1
        self._next_space_id: int = 2

    def get_current_space(self) -> Dict[str, Any]:
        """Get the current space info."""
        for space in self._spaces:
            if space["id"] == self._current_space_id:
                return space
        return self._spaces[0] if self._spaces else {}

    def get_all_spaces(self) -> List[Dict[str, Any]]:
        """Get all spaces."""
        return list(self._spaces)

    def create_space(self, name: str = "") -> int:
        """Create a new space."""
        space_id = self._next_space_id
        self._next_space_id += 1
        if not name:
            name = f"Desktop {space_id}"
        self._spaces.append({"id": space_id, "name": name, "is_visible": False})
        return space_id

    def switch_to_space(self, space_id: int) -> bool:
        """Switch to a specific space."""
        for space in self._spaces:
            space["is_visible"] = (space["id"] == space_id)
        self._current_space_id = space_id
        return True

    def remove_space(self, space_id: int) -> bool:
        """Remove a space."""
        if len(self._spaces) <= 1:
            return False
        self._spaces = [s for s in self._spaces if s["id"] != space_id]
        if self._current_space_id == space_id:
            self._current_space_id = self._spaces[0]["id"]
        return True

    def move_window_to_space(self, window_number: int,
                             space_id: int) -> bool:
        """Move a window to a specific space."""
        return True

    def get_space_count(self) -> int:
        """Get the number of spaces."""
        return len(self._spaces)


class NotificationPoster:
    """
    Simulated macOS notification posting (NSUserNotificationCenter).

    Posts and manages user notifications.
    """

    def __init__(self) -> None:
        self._notifications: List[Dict[str, Any]] = []
        self._delivered: List[Dict[str, Any]] = []
        self._handlers: Dict[str, List[Callable]] = {}

    def post_notification(self, title: str, subtitle: str = "",
                          body: str = "", identifier: str = "",
                          sound_name: str = "") -> Dict[str, Any]:
        """Post a user notification."""
        notification = {
            "identifier": identifier or f"notif-{len(self._notifications)}",
            "title": title,
            "subtitle": subtitle,
            "body": body,
            "sound_name": sound_name,
            "presented": True,
            "delivered_at": time.time(),
        }
        self._notifications.append(notification)
        self._delivered.append(notification)

        # Trigger handlers
        for handler in self._handlers.get("*", []):
            try:
                handler(notification)
            except Exception:
                pass
        for handler in self._handlers.get(identifier, []):
            try:
                handler(notification)
            except Exception:
                pass

        return notification

    def remove_delivered_notification(self, identifier: str) -> bool:
        """Remove a delivered notification."""
        self._delivered = [n for n in self._delivered if n["identifier"] != identifier]
        return True

    def get_delivered_notifications(self) -> List[Dict[str, Any]]:
        """Get all delivered notifications."""
        return list(self._delivered)

    def register_handler(self, identifier: str,
                         handler: Callable[[Dict[str, Any]], None]) -> None:
        """Register a notification handler."""
        if identifier not in self._handlers:
            self._handlers[identifier] = []
        self._handlers[identifier].append(handler)

    def post_distributed_notification(self, name: str,
                                       user_info: Optional[Dict[str, Any]] = None) -> None:
        """Post a distributed notification (NSDistributedNotificationCenter)."""
        for handler in self._handlers.get(name, []):
            try:
                handler(user_info)
            except Exception:
                pass


class MacOSAPI:
    """
    High-level macOS API simulation.

    Provides a unified interface to window management, accessibility,
    workspace management, and notifications.
    """

    def __init__(self) -> None:
        self.window_manager = NSWindowManager()
        self.workspace = WorkspaceManager()
        self.notifications = NotificationPoster()
        self._accessibility_root: Optional[AccessibilityElement] = None
        self._init_accessibility_tree()

    def _init_accessibility_tree(self) -> None:
        """Initialize a basic accessibility tree."""
        self._accessibility_root = AccessibilityElement(
            role=AXRole.WINDOW, label="Desktop"
        )

    def create_window(self, title: str = "Untitled",
                      x: float = 100, y: float = 100,
                      width: float = 400, height: float = 300) -> int:
        """Create a new window."""
        rect = NSRect(x=x, y=y, width=width, height=height)
        return self.window_manager.create_window(rect=rect, title=title)

    def get_window_list(self, option: int = 0) -> List[WindowInfo]:
        """
        Get window list (simulates CGWindowListCopyWindowInfo).

        Options:
        0 - All windows
        1 - On-screen only
        2 - Exclude desktop elements
        """
        windows: List[WindowInfo] = []
        for wn in self.window_manager.get_all_windows():
            win = self.window_manager._windows.get(wn, {})
            frame = win.get("frame", NSRect())
            info = WindowInfo(
                window_id=wn,
                window_name=win.get("title", ""),
                owner_name="SimulatedApp",
                owner_pid=win.get("owner_pid", 0),
                bounds=frame,
                layer=win.get("level", 0),
                is_on_screen=win.get("is_visible", False),
                alpha=win.get("alpha", 1.0),
                is_visible=win.get("is_visible", False),
                window_number=wn,
            )
            if option == 1 and not info.is_on_screen:
                continue
            windows.append(info)
        return windows

    def get_frontmost_window(self) -> Optional[WindowInfo]:
        """Get the frontmost window."""
        wn = self.window_manager.get_key_window()
        if wn is None:
            return None
        win = self.window_manager._windows.get(wn, {})
        frame = win.get("frame", NSRect())
        return WindowInfo(
            window_id=wn,
            window_name=win.get("title", ""),
            owner_name="SimulatedApp",
            owner_pid=win.get("owner_pid", 0),
            bounds=frame,
            window_number=wn,
        )

    def get_accessibility_element(self, pid: int = 0) -> AccessibilityElement:
        """Get the accessibility root element."""
        if self._accessibility_root is None:
            self._init_accessibility_tree()
        return self._accessibility_root

    def get_screen_size(self) -> Tuple[float, float]:
        """Get the main screen size."""
        return (1920.0, 1080.0)

    def get_mouse_position(self) -> CGPoint:
        """Get the current mouse position."""
        return CGPoint(x=0, y=0)

    def set_mouse_position(self, x: float, y: float) -> None:
        """Set the mouse position."""
        pass

    def get_system_uptime(self) -> float:
        """Get system uptime in seconds."""
        return time.time()
