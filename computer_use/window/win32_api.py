"""
Windows API Simulation Module

Simulates Win32 API behavior for cross-platform use:
- Window enumeration and handle management
- Class name retrieval
- Message sending (WM_CLOSE, WM_SETTEXT, etc.)
- Win32 error codes
- Window properties and styles

Pure Python standard library only.
"""

from __future__ import annotations

import enum
import time
import threading
import weakref
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Callable


class Win32Error(enum.IntEnum):
    """Win32 error codes."""
    SUCCESS = 0
    INVALID_FUNCTION = 1
    FILE_NOT_FOUND = 2
    PATH_NOT_FOUND = 3
    ACCESS_DENIED = 5
    INVALID_HANDLE = 6
    NOT_ENOUGH_MEMORY = 8
    INVALID_PARAMETER = 87
    OPERATION_ABORTED = 995
    WINDOW_NOT_FOUND = 1400
    INVALID_WINDOW_HANDLE = 1400
    INVALID_MENU_HANDLE = 1401
    INVALID_CURSOR_HANDLE = 1402
    INVALID_ACCEL_HANDLE = 1403
    INVALID_HOOK_HANDLE = 1404
    INVALID_DWP_HANDLE = 1405
    INVALID_WINDOW_STYLE = 2002
    CLASS_ALREADY_EXISTS = 1410
    CLASS_DOES_NOT_EXIST = 1411
    WINDOW_HAS_DIFFERENT_PARENT = 1440
    MESSAGE_NOT_FOUND = 1418


@dataclass
class Rect:
    """Rectangle structure (RECT)."""
    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def x(self) -> int:
        return self.left

    @property
    def y(self) -> int:
        return self.top

    def contains(self, x: int, y: int) -> bool:
        return self.left <= x <= self.right and self.top <= y <= self.bottom

    def intersects(self, other: Rect) -> bool:
        return not (self.right < other.left or self.left > other.right or
                    self.bottom < other.top or self.top > other.bottom)

    def union(self, other: Rect) -> Rect:
        return Rect(
            left=min(self.left, other.left),
            top=min(self.top, other.top),
            right=max(self.right, other.right),
            bottom=max(self.bottom, other.bottom),
        )

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)


@dataclass
class Point:
    """Point structure (POINT)."""
    x: int = 0
    y: int = 0

    def to_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)


@dataclass
class WindowStyle:
    """Window style flags."""
    ws_overlapped: bool = True
    ws_popup: bool = False
    ws_child: bool = False
    ws_minimize: bool = False
    ws_visible: bool = True
    ws_disabled: bool = False
    ws_maximize: bool = False
    ws_caption: bool = True
    ws_border: bool = True
    ws_sysmenu: bool = True
    ws_thickframe: bool = False
    ws_group: bool = False
    ws_tabstop: bool = False

    def to_dword(self) -> int:
        """Convert to Win32 style DWORD value."""
        value = 0
        if self.ws_overlapped:
            value |= 0x00000000
        if self.ws_popup:
            value |= 0x80000000
        if self.ws_child:
            value |= 0x40000000
        if self.ws_minimize:
            value |= 0x20000000
        if self.ws_visible:
            value |= 0x10000000
        if self.ws_disabled:
            value |= 0x08000000
        if self.ws_maximize:
            value |= 0x01000000
        if self.ws_caption:
            value |= 0x00C00000
        if self.ws_border:
            value |= 0x00800000
        if self.ws_sysmenu:
            value |= 0x00080000
        if self.ws_thickframe:
            value |= 0x00040000
        if self.ws_group:
            value |= 0x00020000
        if self.ws_tabstop:
            value |= 0x00010000
        return value

    @classmethod
    def from_dword(cls, value: int) -> WindowStyle:
        """Parse from Win32 style DWORD value."""
        return cls(
            ws_popup=bool(value & 0x80000000),
            ws_child=bool(value & 0x40000000),
            ws_minimize=bool(value & 0x20000000),
            ws_visible=bool(value & 0x10000000),
            ws_disabled=bool(value & 0x08000000),
            ws_maximize=bool(value & 0x01000000),
            ws_caption=bool(value & 0x00C00000),
            ws_border=bool(value & 0x00800000),
            ws_sysmenu=bool(value & 0x00080000),
            ws_thickframe=bool(value & 0x00040000),
            ws_group=bool(value & 0x00020000),
            ws_tabstop=bool(value & 0x00010000),
        )


@dataclass
class WindowClass:
    """Window class registration info."""
    class_name: str
    style: int = 0
    window_proc: Optional[Callable] = None
    icon: int = 0
    cursor: int = 0
    background: int = 0
    menu_name: str = ""
    instance_handle: int = 0


class Win32Constants:
    """Common Win32 constants."""

    # Window messages
    WM_NULL = 0x0000
    WM_CREATE = 0x0001
    WM_DESTROY = 0x0002
    WM_MOVE = 0x0003
    WM_SIZE = 0x0005
    WM_ACTIVATE = 0x0006
    WM_SETFOCUS = 0x0007
    WM_KILLFOCUS = 0x0008
    WM_ENABLE = 0x000A
    WM_SETREDRAW = 0x000B
    WM_SETTEXT = 0x000C
    WM_GETTEXT = 0x000D
    WM_GETTEXTLENGTH = 0x000E
    WM_PAINT = 0x000F
    WM_CLOSE = 0x0010
    WM_QUERYENDSESSION = 0x0011
    WM_QUIT = 0x0012
    WM_QUERYOPEN = 0x0013
    WM_ERASEBKGND = 0x0014
    WM_SYSCOLORCHANGE = 0x0015
    WM_SHOWWINDOW = 0x0018
    WM_WININICHANGE = 0x001A
    WM_DEVMODECHANGE = 0x001B
    WM_ACTIVATEAPP = 0x001C
    WM_FONTCHANGE = 0x001D
    WM_TIMECHANGE = 0x001E
    WM_CANCELMODE = 0x001F
    WM_SETCURSOR = 0x0020
    WM_MOUSEACTIVATE = 0x0021
    WM_CHILDACTIVATE = 0x0022
    WM_QUEUESYNC = 0x0023
    WM_GETMINMAXINFO = 0x0024
    WM_PAINTICON = 0x0026
    WM_ICONERASEBKGND = 0x0027
    WM_NEXTDLGCTL = 0x0028
    WM_SPOOLERSTATUS = 0x002A
    WM_DRAWITEM = 0x002B
    WM_MEASUREITEM = 0x002C
    WM_DELETEITEM = 0x002D
    WM_VKEYTOITEM = 0x002E
    WM_CHARTOITEM = 0x002F
    WM_SETFONT = 0x0030
    WM_GETFONT = 0x0031
    WM_SETHOTKEY = 0x0032
    WM_GETHOTKEY = 0x0033
    WM_QUERYDRAGICON = 0x0037
    WM_COMPAREITEM = 0x0039
    WM_GETOBJECT = 0x003D
    WM_COMPACTING = 0x0041
    WM_COMMNOTIFY = 0x0044
    WM_WINDOWPOSCHANGING = 0x0046
    WM_WINDOWPOSCHANGED = 0x0047
    WM_POWER = 0x0048
    WM_COPYDATA = 0x004A
    WM_CANCELJOURNAL = 0x004B
    WM_NOTIFY = 0x004E
    WM_INPUTLANGCHANGEREQUEST = 0x0050
    WM_INPUTLANGCHANGE = 0x0051
    WM_TCARD = 0x0052
    WM_HELP = 0x0053
    WM_USERCHANGED = 0x0054
    WM_NOTIFYFORMAT = 0x0055
    WM_CONTEXTMENU = 0x007B
    WM_STYLECHANGING = 0x007C
    WM_STYLECHANGED = 0x007D
    WM_DISPLAYCHANGE = 0x007E
    WM_GETICON = 0x007F
    WM_SETICON = 0x0080
    WM_NCCREATE = 0x0081
    WM_NCDESTROY = 0x0082
    WM_NCCALCSIZE = 0x0083
    WM_NCHITTEST = 0x0084
    WM_NCPAINT = 0x0085
    WM_NCACTIVATE = 0x0086
    WM_GETDLGCODE = 0x0087
    WM_NCMOUSEMOVE = 0x00A0
    WM_NCLBUTTONDOWN = 0x00A1
    WM_NCLBUTTONUP = 0x00A2
    WM_NCLBUTTONDBLCLK = 0x00A3
    WM_NCRBUTTONDOWN = 0x00A4
    WM_NCRBUTTONUP = 0x00A5
    WM_KEYDOWN = 0x0100
    WM_KEYUP = 0x0101
    WM_CHAR = 0x0102
    WM_DEADCHAR = 0x0103
    WM_SYSKEYDOWN = 0x0104
    WM_SYSKEYUP = 0x0105
    WM_SYSCHAR = 0x0106
    WM_SYSDEADCHAR = 0x0107
    WM_MOUSEMOVE = 0x0200
    WM_LBUTTONDOWN = 0x0201
    WM_LBUTTONUP = 0x0202
    WM_LBUTTONDBLCLK = 0x0203
    WM_RBUTTONDOWN = 0x0204
    WM_RBUTTONUP = 0x0205
    WM_RBUTTONDBLCLK = 0x0206
    WM_MBUTTONDOWN = 0x0207
    WM_MBUTTONUP = 0x0208
    WM_MBUTTONDBLCLK = 0x0209
    WM_MOUSEWHEEL = 0x020A
    WM_XBUTTONDOWN = 0x020B
    WM_XBUTTONUP = 0x020C
    WM_XBUTTONDBLCLK = 0x020D

    # Window styles
    WS_OVERLAPPED = 0x00000000
    WS_POPUP = 0x80000000
    WS_CHILD = 0x40000000
    WS_MINIMIZE = 0x20000000
    WS_VISIBLE = 0x10000000
    WS_DISABLED = 0x08000000
    WS_MAXIMIZE = 0x01000000
    WS_CAPTION = 0x00C00000
    WS_BORDER = 0x00800000
    WS_DLGFRAME = 0x00400000
    WS_VSCROLL = 0x00200000
    WS_HSCROLL = 0x00100000
    WS_SYSMENU = 0x00080000
    WS_THICKFRAME = 0x00040000
    WS_GROUP = 0x00020000
    WS_TABSTOP = 0x00010000

    # Extended window styles
    WS_EX_DLGMODALFRAME = 0x00000001
    WS_EX_NOPARENTNOTIFY = 0x00000004
    WS_EX_TOPMOST = 0x00000008
    WS_EX_ACCEPTFILES = 0x00000010
    WS_EX_TRANSPARENT = 0x00000020
    WS_EX_MDICHILD = 0x00000040
    WS_EX_TOOLWINDOW = 0x00000080
    WS_EX_WINDOWEDGE = 0x00000100
    WS_EX_CLIENTEDGE = 0x00000200
    WS_EX_CONTEXTHELP = 0x00000400
    WS_EX_RIGHT = 0x00001000
    WS_EX_LEFT = 0x00000000
    WS_EX_RTLREADING = 0x00002000
    WS_EX_LTRREADING = 0x00000000
    WS_EX_LEFTSCROLLBAR = 0x00004000
    WS_EX_CONTROLPARENT = 0x00010000
    WS_EX_STATICEDGE = 0x00020000
    WS_EX_APPWINDOW = 0x00040000
    WS_EX_LAYERED = 0x00080000

    # Show window commands
    SW_HIDE = 0
    SW_SHOWNORMAL = 1
    SW_SHOWMINIMIZED = 2
    SW_SHOWMAXIMIZED = 3
    SW_SHOWNOACTIVATE = 4
    SW_SHOW = 5
    SW_MINIMIZE = 6
    SW_SHOWMINNOACTIVE = 7
    SW_SHOWNA = 8
    SW_RESTORE = 9
    SW_SHOWDEFAULT = 10

    # Special window handles
    HWND_BROADCAST = 0xFFFF
    HWND_BOTTOM = 0x0001
    HWND_TOP = 0x0000
    HWND_TOPMOST = -1
    HWND_NOTOPMOST = -2

    # Callback message IDs
    WM_TIMER = 0x0113
    WM_HOTKEY = 0x0312

    # Clipboard messages
    WM_CUT = 0x0300
    WM_COPY = 0x0301
    WM_PASTE = 0x0302
    WM_CLEAR = 0x0303
    WM_UNDO = 0x0304


class WindowHandle:
    """
    Simulated Win32 window handle (HWND).

    Represents a window with properties, styles, and message handling.
    """

    _next_id: int = 1
    _lock = threading.Lock()

    def __init__(self, class_name: str, title: str = "",
                 rect: Optional[Rect] = None,
                 parent: Optional[WindowHandle] = None,
                 style: Optional[WindowStyle] = None,
                 ex_style: int = 0,
                 window_id: Optional[int] = None) -> None:
        with WindowHandle._lock:
            if window_id is not None:
                self._handle_id = window_id
            else:
                self._handle_id = WindowHandle._next_id
                WindowHandle._next_id += 1

        self.class_name = class_name
        self.title = title
        self.rect = rect or Rect(100, 100, 500, 400)
        self.parent = parent
        self.style = style or WindowStyle()
        self.ex_style = ex_style
        self.children: List[WindowHandle] = []
        self.properties: Dict[str, Any] = {}
        self.user_data: Any = None
        self.is_destroyed = False
        self.is_visible = self.style.ws_visible
        self.is_enabled = not self.style.ws_disabled
        self.creation_time = time.time()
        self.last_message_time = 0.0
        self._message_handlers: Dict[int, List[Callable]] = {}
        self._message_log: List[Tuple[int, Any, Any, int]] = []

        if parent:
            parent.children.append(self)

    @property
    def handle_id(self) -> int:
        return self._handle_id

    @property
    def hwnd(self) -> int:
        """Get the HWND value (simulated)."""
        return self._handle_id

    def send_message(self, msg: int, wparam: Any = 0, lparam: Any = 0) -> int:
        """Send a message to this window and wait for processing."""
        self.last_message_time = time.time()
        self._message_log.append((msg, wparam, lparam, int(time.time() * 1000)))

        handlers = self._message_handlers.get(msg, [])
        for handler in handlers:
            result = handler(msg, wparam, lparam)
            if result is not None:
                return result

        return self._default_message_handler(msg, wparam, lparam)

    def post_message(self, msg: int, wparam: Any = 0, lparam: Any = 0) -> bool:
        """Post a message to this window's message queue (async)."""
        self._message_log.append((msg, wparam, lparam, int(time.time() * 1000)))
        return True

    def register_handler(self, msg: int, handler: Callable) -> None:
        """Register a message handler."""
        if msg not in self._message_handlers:
            self._message_handlers[msg] = []
        self._message_handlers[msg].append(handler)

    def _default_message_handler(self, msg: int, wparam: Any, lparam: Any) -> int:
        """Default message handling."""
        if msg == Win32Constants.WM_SETTEXT:
            if isinstance(lparam, str):
                self.title = lparam
            elif isinstance(wparam, str):
                self.title = wparam
            return 1
        elif msg == Win32Constants.WM_GETTEXT:
            return len(self.title)
        elif msg == Win32Constants.WM_GETTEXTLENGTH:
            return len(self.title)
        elif msg == Win32Constants.WM_CLOSE:
            self.is_visible = False
            return 0
        elif msg == Win32Constants.WM_DESTROY:
            self.is_destroyed = True
            return 0
        elif msg == Win32Constants.WM_SHOWWINDOW:
            self.is_visible = bool(wparam)
            return 0
        elif msg == Win32Constants.WM_ENABLE:
            self.is_enabled = bool(wparam)
            return 0
        elif msg == Win32Constants.WM_SIZE:
            if isinstance(lparam, tuple) and len(lparam) == 2:
                self.rect.right = self.rect.left + lparam[0]
                self.rect.bottom = self.rect.top + lparam[1]
            return 0
        elif msg == Win32Constants.WM_MOVE:
            if isinstance(lparam, tuple) and len(lparam) == 2:
                w = self.rect.width
                h = self.rect.height
                self.rect.left = lparam[0]
                self.rect.top = lparam[1]
                self.rect.right = self.rect.left + w
                self.rect.bottom = self.rect.top + h
            return 0
        return 0

    def get_message_log(self) -> List[Tuple[int, Any, Any, int]]:
        """Get the message log."""
        return list(self._message_log)

    def __repr__(self) -> str:
        return (f"WindowHandle(hwnd=0x{self.hwnd:04X}, "
                f"class='{self.class_name}', title='{self.title}')")

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, WindowHandle):
            return self.hwnd == other.hwnd
        return False

    def __hash__(self) -> int:
        return hash(self.hwnd)


class WindowManager:
    """
    Simulated Win32 window manager.

    Manages window creation, enumeration, and lifecycle.
    """

    def __init__(self) -> None:
        self._windows: Dict[int, WindowHandle] = {}
        self._classes: Dict[str, WindowClass] = {}
        self._top_level: List[int] = []
        self._z_order: List[int] = []
        self._foreground: Optional[int] = None
        self._lock = threading.Lock()

    def register_class(self, wc: WindowClass) -> bool:
        """Register a window class."""
        with self._lock:
            if wc.class_name in self._classes:
                return False
            self._classes[wc.class_name] = wc
            return True

    def unregister_class(self, class_name: str) -> bool:
        """Unregister a window class."""
        with self._lock:
            if class_name not in self._classes:
                return False
            # Check if any windows use this class
            for hwnd, window in self._windows.items():
                if window.class_name == class_name and not window.is_destroyed:
                    return False
            del self._classes[class_name]
            return True

    def create_window(self, class_name: str, title: str = "",
                      rect: Optional[Rect] = None,
                      parent: Optional[WindowHandle] = None,
                      style: Optional[WindowStyle] = None,
                      ex_style: int = 0) -> Tuple[int, Optional[WindowHandle]]:
        """
        Create a new window.

        Returns (error_code, window_handle).
        """
        with self._lock:
            if class_name not in self._classes:
                return (Win32Error.CLASS_DOES_NOT_EXIST, None)

            window = WindowHandle(
                class_name=class_name,
                title=title,
                rect=rect,
                parent=parent,
                style=style,
                ex_style=ex_style,
            )

            self._windows[window.hwnd] = window

            if parent is None:
                self._top_level.append(window.hwnd)
                self._z_order.append(window.hwnd)

            # Send WM_CREATE
            window.send_message(Win32Constants.WM_CREATE, 0, 0)

            # Send WM_SHOWWINDOW if visible
            if window.is_visible:
                window.send_message(Win32Constants.WM_SHOWWINDOW, 1, 0)

            if self._foreground is None:
                self._foreground = window.hwnd

            return (Win32Error.SUCCESS, window)

    def destroy_window(self, hwnd: int) -> int:
        """Destroy a window."""
        with self._lock:
            window = self._windows.get(hwnd)
            if window is None or window.is_destroyed:
                return Win32Error.INVALID_WINDOW_HANDLE

            # Send WM_DESTROY
            window.send_message(Win32Constants.WM_DESTROY, 0, 0)

            # Remove from parent's children
            if window.parent:
                window.parent.children = [
                    c for c in window.parent.children if c.hwnd != hwnd
                ]

            # Remove from tracking lists
            if hwnd in self._top_level:
                self._top_level.remove(hwnd)
            if hwnd in self._z_order:
                self._z_order.remove(hwnd)
            if self._foreground == hwnd:
                self._foreground = self._top_level[-1] if self._top_level else None

            # Destroy children recursively
            for child in list(window.children):
                self.destroy_window(child.hwnd)

            return Win32Error.SUCCESS

    def find_window(self, class_name: Optional[str] = None,
                    title: Optional[str] = None) -> Optional[WindowHandle]:
        """Find a window by class name and/or title."""
        with self._lock:
            for hwnd, window in self._windows.items():
                if window.is_destroyed:
                    continue
                if class_name and window.class_name != class_name:
                    continue
                if title and window.title != title:
                    continue
                return window
            return None

    def find_window_by_title_contains(self, substring: str) -> List[WindowHandle]:
        """Find windows whose title contains the given substring."""
        with self._lock:
            results: List[WindowHandle] = []
            for hwnd, window in self._windows.items():
                if not window.is_destroyed and substring in window.title:
                    results.append(window)
            return results

    def enum_windows(self) -> List[WindowHandle]:
        """Enumerate all top-level windows."""
        with self._lock:
            return [self._windows[hwnd] for hwnd in self._top_level
                    if hwnd in self._windows and not self._windows[hwnd].is_destroyed]

    def enum_child_windows(self, parent_hwnd: int) -> List[WindowHandle]:
        """Enumerate child windows of a parent."""
        with self._lock:
            window = self._windows.get(parent_hwnd)
            if window is None:
                return []
            return [c for c in window.children if not c.is_destroyed]

    def get_window(self, hwnd: int) -> Optional[WindowHandle]:
        """Get a window handle by HWND."""
        return self._windows.get(hwnd)

    def get_class_name(self, hwnd: int) -> str:
        """Get the class name of a window."""
        window = self._windows.get(hwnd)
        if window is None:
            return ""
        return window.class_name

    def get_window_text(self, hwnd: int) -> str:
        """Get the title/text of a window."""
        window = self._windows.get(hwnd)
        if window is None:
            return ""
        return window.title

    def set_window_text(self, hwnd: int, text: str) -> bool:
        """Set the title/text of a window."""
        window = self._windows.get(hwnd)
        if window is None:
            return False
        window.send_message(Win32Constants.WM_SETTEXT, 0, text)
        return True

    def get_window_rect(self, hwnd: int) -> Optional[Rect]:
        """Get the bounding rectangle of a window."""
        window = self._windows.get(hwnd)
        if window is None:
            return None
        return window.rect

    def set_window_pos(self, hwnd: int, x: int, y: int,
                       width: int, height: int) -> bool:
        """Set the position and size of a window."""
        window = self._windows.get(hwnd)
        if window is None:
            return False
        window.rect = Rect(x, y, x + width, y + height)
        window.send_message(Win32Constants.WM_SIZE, 0, (width, height))
        window.send_message(Win32Constants.WM_MOVE, 0, (x, y))
        return True

    def show_window(self, hwnd: int, cmd: int) -> bool:
        """Show, hide, minimize, or maximize a window."""
        window = self._windows.get(hwnd)
        if window is None:
            return False

        if cmd == Win32Constants.SW_HIDE:
            window.is_visible = False
        elif cmd == Win32Constants.SW_SHOW:
            window.is_visible = True
        elif cmd == Win32Constants.SW_MINIMIZE:
            window.is_visible = False
            window.style.ws_minimize = True
        elif cmd == Win32Constants.SW_SHOWMAXIMIZED:
            window.is_visible = True
            window.style.ws_maximize = True
        elif cmd == Win32Constants.SW_RESTORE:
            window.is_visible = True
            window.style.ws_maximize = False
            window.style.ws_minimize = False

        window.send_message(Win32Constants.WM_SHOWWINDOW,
                            1 if window.is_visible else 0, 0)
        return True

    def get_foreground_window(self) -> Optional[WindowHandle]:
        """Get the foreground window."""
        if self._foreground is None:
            return None
        return self._windows.get(self._foreground)

    def set_foreground_window(self, hwnd: int) -> bool:
        """Set the foreground window."""
        window = self._windows.get(hwnd)
        if window is None or window.is_destroyed:
            return False
        self._foreground = hwnd
        window.send_message(Win32Constants.WM_ACTIVATE, 1, 0)
        return True

    def is_window(self, hwnd: int) -> bool:
        """Check if a handle is a valid window."""
        window = self._windows.get(hwnd)
        return window is not None and not window.is_destroyed

    def is_window_visible(self, hwnd: int) -> bool:
        """Check if a window is visible."""
        window = self._windows.get(hwnd)
        return window is not None and not window.is_destroyed and window.is_visible

    def is_window_enabled(self, hwnd: int) -> bool:
        """Check if a window is enabled."""
        window = self._windows.get(hwnd)
        return window is not None and not window.is_destroyed and window.is_enabled

    def get_parent(self, hwnd: int) -> Optional[WindowHandle]:
        """Get the parent window."""
        window = self._windows.get(hwnd)
        if window is None:
            return None
        return window.parent

    def set_parent(self, child_hwnd: int, parent_hwnd: int) -> bool:
        """Set the parent of a window."""
        child = self._windows.get(child_hwnd)
        parent = self._windows.get(parent_hwnd)
        if child is None or parent is None:
            return False
        if child.parent:
            child.parent.children = [c for c in child.parent.children if c.hwnd != child_hwnd]
        child.parent = parent
        parent.children.append(child)
        return True

    def get_window_count(self) -> int:
        """Get the total number of active windows."""
        return sum(1 for w in self._windows.values() if not w.is_destroyed)


class MessageSender:
    """
    Send Win32 messages to windows.

    Provides convenience methods for common message types.
    """

    def __init__(self, window_manager: WindowManager) -> None:
        self.wm = window_manager
        self.constants = Win32Constants

    def close(self, hwnd: int) -> int:
        """Send WM_CLOSE to a window."""
        window = self.wm.get_window(hwnd)
        if window is None:
            return Win32Error.INVALID_WINDOW_HANDLE
        return window.send_message(self.constants.WM_CLOSE)

    def destroy(self, hwnd: int) -> int:
        """Send WM_DESTROY to a window."""
        window = self.wm.get_window(hwnd)
        if window is None:
            return Win32Error.INVALID_WINDOW_HANDLE
        return window.send_message(self.constants.WM_DESTROY)

    def set_text(self, hwnd: int, text: str) -> int:
        """Send WM_SETTEXT to a window."""
        window = self.wm.get_window(hwnd)
        if window is None:
            return Win32Error.INVALID_WINDOW_HANDLE
        return window.send_message(self.constants.WM_SETTEXT, 0, text)

    def get_text(self, hwnd: int) -> str:
        """Send WM_GETTEXT to a window."""
        window = self.wm.get_window(hwnd)
        if window is None:
            return ""
        length = window.send_message(self.constants.WM_GETTEXTLENGTH)
        if length > 0:
            window.send_message(self.constants.WM_GETTEXT, length + 1, 0)
        return window.title

    def send_key(self, hwnd: int, vk_code: int) -> int:
        """Send a key press to a window (WM_KEYDOWN + WM_KEYUP)."""
        window = self.wm.get_window(hwnd)
        if window is None:
            return Win32Error.INVALID_WINDOW_HANDLE
        window.send_message(self.constants.WM_KEYDOWN, vk_code, 0)
        window.send_message(self.constants.WM_KEYUP, vk_code, 0)
        return 0

    def send_char(self, hwnd: int, char_code: int) -> int:
        """Send a character to a window (WM_CHAR)."""
        window = self.wm.get_window(hwnd)
        if window is None:
            return Win32Error.INVALID_WINDOW_HANDLE
        return window.send_message(self.constants.WM_CHAR, char_code, 0)

    def send_click(self, hwnd: int, x: int, y: int,
                   button: str = "left") -> int:
        """Send a mouse click to a window."""
        window = self.wm.get_window(hwnd)
        if window is None:
            return Win32Error.INVALID_WINDOW_HANDLE

        if button == "left":
            window.send_message(self.constants.WM_LBUTTONDOWN, 0, (x, y))
            window.send_message(self.constants.WM_LBUTTONUP, 0, (x, y))
        elif button == "right":
            window.send_message(self.constants.WM_RBUTTONDOWN, 0, (x, y))
            window.send_message(self.constants.WM_RBUTTONUP, 0, (x, y))
        return 0

    def send_custom_message(self, hwnd: int, msg: int,
                            wparam: Any = 0, lparam: Any = 0) -> int:
        """Send a custom message to a window."""
        window = self.wm.get_window(hwnd)
        if window is None:
            return Win32Error.INVALID_WINDOW_HANDLE
        return window.send_message(msg, wparam, lparam)

    def broadcast_message(self, msg: int, wparam: Any = 0,
                          lparam: Any = 0) -> Dict[int, int]:
        """Broadcast a message to all top-level windows."""
        results: Dict[int, int] = {}
        for window in self.wm.enum_windows():
            results[window.hwnd] = window.send_message(msg, wparam, lparam)
        return results


class Win32API:
    """
    High-level Win32 API simulation.

    Provides a unified interface to window management, messaging,
    and system operations.
    """

    def __init__(self) -> None:
        self.window_manager = WindowManager()
        self.message_sender = MessageSender(self.window_manager)
        self.constants = Win32Constants
        self._error_code: int = Win32Error.SUCCESS

    @property
    def last_error(self) -> int:
        """Get the last error code."""
        return self._error_code

    def get_last_error(self) -> Win32Error:
        """Get the last error as a Win32Error enum."""
        return Win32Error(self._error_code)

    def register_class(self, class_name: str) -> bool:
        """Register a window class."""
        wc = WindowClass(class_name=class_name)
        result = self.window_manager.register_class(wc)
        if not result:
            self._error_code = Win32Error.CLASS_ALREADY_EXISTS
        else:
            self._error_code = Win32Error.SUCCESS
        return result

    def create_window(self, class_name: str, title: str = "",
                      x: int = 100, y: int = 100,
                      width: int = 400, height: int = 300) -> Optional[WindowHandle]:
        """Create a top-level window."""
        rect = Rect(x, y, x + width, y + height)
        error, window = self.window_manager.create_window(
            class_name=class_name, title=title, rect=rect
        )
        self._error_code = error
        return window

    def find_window(self, class_name: str = "", title: str = "") -> Optional[WindowHandle]:
        """Find a window by class name and/or title."""
        cn = class_name if class_name else None
        tt = title if title else None
        return self.window_manager.find_window(cn, tt)

    def get_desktop_window(self) -> WindowHandle:
        """Get a handle to the desktop window."""
        return WindowHandle(class_name="#32769", title="Desktop",
                            rect=Rect(0, 0, 1920, 1080))

    def get_shell_window(self) -> Optional[WindowHandle]:
        """Get a handle to the Shell window (explorer)."""
        return self.window_manager.find_window("CabinetWClass", "")

    def message_box(self, text: str, title: str = "",
                    buttons: int = 0) -> int:
        """Simulate a message box."""
        return 1  # IDOK

    def get_system_metrics(self, index: int) -> int:
        """Get system metrics (screen size, etc.)."""
        metrics = {
            0: 1920,  # SM_CXSCREEN
            1: 1080,  # SM_CYSCREEN
            15: 32,   # SM_CXSMICON
            16: 32,   # SM_CYSMICON
        }
        return metrics.get(index, 0)

    def get_cursor_pos(self) -> Point:
        """Get the current cursor position."""
        return Point(0, 0)

    def set_cursor_pos(self, x: int, y: int) -> bool:
        """Set the cursor position."""
        return True

    def screen_to_client(self, hwnd: int, screen_x: int,
                         screen_y: int) -> Optional[Point]:
        """Convert screen coordinates to client coordinates."""
        rect = self.window_manager.get_window_rect(hwnd)
        if rect is None:
            return None
        return Point(screen_x - rect.left, screen_y - rect.top)

    def client_to_screen(self, hwnd: int, client_x: int,
                         client_y: int) -> Optional[Point]:
        """Convert client coordinates to screen coordinates."""
        rect = self.window_manager.get_window_rect(hwnd)
        if rect is None:
            return None
        return Point(client_x + rect.left, client_y + rect.top)
