"""
X11 API Simulation Module

Simulates X11/X Window System API behavior for cross-platform use:
- Display connection management
- Window atoms and properties
- EWMH (Extended Window Manager Hints) support
- ICCCM compliance
- Property management
- Event handling

Pure Python standard library only.
"""

from __future__ import annotations

import enum
import time
import threading
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Callable, Set


class X11EventType(enum.IntEnum):
    """X11 event types."""
    KEY_PRESS = 2
    KEY_RELEASE = 3
    BUTTON_PRESS = 4
    BUTTON_RELEASE = 5
    MOTION_NOTIFY = 6
    ENTER_NOTIFY = 7
    LEAVE_NOTIFY = 8
    FOCUS_IN = 9
    FOCUS_OUT = 10
    KEYMAP_NOTIFY = 11
    EXPOSE = 12
    GRAPHICS_EXPOSURE = 13
    NO_EXPOSE = 14
    VISIBILITY_NOTIFY = 15
    CREATE_NOTIFY = 16
    DESTROY_NOTIFY = 17
    UNMAP_NOTIFY = 18
    MAP_NOTIFY = 19
    MAP_REQUEST = 20
    REPARENT_NOTIFY = 21
    CONFIGURE_NOTIFY = 22
    CONFIGURE_REQUEST = 23
    GRAVITY_NOTIFY = 24
    RESIZE_REQUEST = 25
    CIRCULATE_NOTIFY = 26
    CIRCULATE_REQUEST = 27
    PROPERTY_NOTIFY = 28
    SELECTION_CLEAR = 29
    SELECTION_REQUEST = 30
    SELECTION_NOTIFY = 31
    COLORMAP_NOTIFY = 32
    CLIENT_MESSAGE = 33
    MAPPING_NOTIFY = 34


class X11WindowClass(enum.IntEnum):
    """X11 window class types."""
    COPY_FROM_PARENT = 0
    INPUT_OUTPUT = 1
    INPUT_ONLY = 2


class X11EventMask(enum.IntFlag):
    """X11 event mask flags."""
    NO_EVENT_MASK = 0
    KEY_PRESS_MASK = 1 << 0
    KEY_RELEASE_MASK = 1 << 1
    BUTTON_PRESS_MASK = 1 << 2
    BUTTON_RELEASE_MASK = 1 << 3
    ENTER_WINDOW_MASK = 1 << 4
    LEAVE_WINDOW_MASK = 1 << 5
    POINTER_MOTION_MASK = 1 << 6
    POINTER_MOTION_HINT_MASK = 1 << 7
    BUTTON1_MOTION_MASK = 1 << 8
    BUTTON2_MOTION_MASK = 1 << 9
    BUTTON3_MOTION_MASK = 1 << 10
    BUTTON4_MOTION_MASK = 1 << 11
    BUTTON5_MOTION_MASK = 1 << 12
    BUTTON_MOTION_MASK = 1 << 13
    KEYMAP_STATE_MASK = 1 << 14
    EXPOSURE_MASK = 1 << 15
    VISIBILITY_CHANGE_MASK = 1 << 16
    STRUCTURE_NOTIFY_MASK = 1 << 17
    RESIZE_REDIRECT_MASK = 1 << 18
    SUBSTRUCTURE_NOTIFY_MASK = 1 << 19
    SUBSTRUCTURE_REDIRECT_MASK = 1 << 20
    FOCUS_CHANGE_MASK = 1 << 21
    PROPERTY_CHANGE_MASK = 1 << 22
    COLORMAP_CHANGE_MASK = 1 << 23
    OWNER_GRAB_BUTTON_MASK = 1 << 24


class X11Gravity(enum.IntEnum):
    """Window gravity for resizing."""
    BIT_GRAVITY = 1
    WIN_GRAVITY = 1
    NORTH_WEST = 1
    NORTH = 2
    NORTH_EAST = 3
    WEST = 4
    CENTER = 5
    EAST = 6
    SOUTH_WEST = 7
    SOUTH = 8
    SOUTH_EAST = 9
    STATIC = 10


@dataclass
class X11Geometry:
    """Window geometry specification."""
    x: int = 0
    y: int = 0
    width: int = 100
    height: int = 100
    border_width: int = 0
    depth: int = 24


@dataclass
class X11Event:
    """X11 event structure."""
    event_type: X11EventType
    window_id: int
    root_window: int
    time: int = 0
    x: int = 0
    y: int = 0
    x_root: int = 0
    y_root: int = 0
    detail: int = 0
    state: int = 0
    data: Any = None

    @classmethod
    def create(cls, event_type: X11EventType, window_id: int,
               **kwargs: Any) -> X11Event:
        return cls(
            event_type=event_type,
            window_id=window_id,
            root_window=0,
            time=int(time.time() * 1000),
            **kwargs,
        )


@dataclass
class X11Atom:
    """X11 atom representation."""
    name: str
    atom_id: int
    value: Any = None
    atom_type: str = "STRING"


class AtomManager:
    """
    X11 atom management.

    Handles intern atoms, predefined atoms, and atom lookups.
    """

    # Predefined X11 atoms
    PREDEFINED_ATOMS: Dict[str, int] = {
        "PRIMARY": 1,
        "SECONDARY": 2,
        "ARC": 3,
        "ATOM": 4,
        "BITMAP": 5,
        "CARDINAL": 6,
        "COLORMAP": 7,
        "CURSOR": 8,
        "CUT_BUFFER0": 9,
        "CUT_BUFFER1": 10,
        "CUT_BUFFER2": 11,
        "CUT_BUFFER3": 12,
        "CUT_BUFFER4": 13,
        "CUT_BUFFER5": 14,
        "CUT_BUFFER6": 15,
        "CUT_BUFFER7": 16,
        "DRAWABLE": 17,
        "FONT": 18,
        "INTEGER": 19,
        "PIXMAP": 20,
        "POINT": 21,
        "RECTANGLE": 22,
        "RESOURCE_MANAGER": 23,
        "RGB_COLOR_MAP": 24,
        "RGB_BEST_MAP": 25,
        "RGB_BLUE_MAP": 26,
        "RGB_DEFAULT_MAP": 27,
        "RGB_GRAY_MAP": 28,
        "RGB_GREEN_MAP": 29,
        "RGB_RED_MAP": 30,
        "STRING": 31,
        "VISUALID": 32,
        "WINDOW": 33,
        "WM_NAME": 34,
        "WM_ICON_NAME": 35,
        "WM_NORMAL_HINTS": 36,
        "WM_SIZE_HINTS": 37,
        "WM_ZOOM_HINTS": 38,
        "WM_CLASS": 39,
        "WM_TRANSIENT_FOR": 40,
        "WM_PROTOCOLS": 41,
        "WM_DELETE_WINDOW": 42,
        "WM_STATE": 43,
        "WM_CLIENT_MACHINE": 44,
        "WM_CHANGE_STATE": 45,
        "WM_COMMAND": 46,
        "WM_HINTS": 47,
        "WM_ICON_SIZE": 48,
        "WM_MOTIF_HINTS": 49,
        "WM_CLIENT_LEADER": 50,
        "WM_WINDOW_ROLE": 51,
        "WM_TAKE_FOCUS": 52,
        "WM_LOCALE_NAME": 53,
    }

    def __init__(self) -> None:
        self._atoms: Dict[int, X11Atom] = {}
        self._name_to_id: Dict[str, int] = {}
        self._next_id = 100
        self._lock = threading.Lock()

        # Initialize predefined atoms
        for name, atom_id in self.PREDEFINED_ATOMS.items():
            atom = X11Atom(name=name, atom_id=atom_id)
            self._atoms[atom_id] = atom
            self._name_to_id[name] = atom_id

    def intern_atom(self, name: str, only_if_exists: bool = False) -> int:
        """Intern an atom, returning its ID."""
        with self._lock:
            if name in self._name_to_id:
                return self._name_to_id[name]
            if only_if_exists:
                return 0
            atom_id = self._next_id
            self._next_id += 1
            atom = X11Atom(name=name, atom_id=atom_id)
            self._atoms[atom_id] = atom
            self._name_to_id[name] = atom_id
            return atom_id

    def get_atom_name(self, atom_id: int) -> str:
        """Get the name of an atom by its ID."""
        atom = self._atoms.get(atom_id)
        return atom.name if atom else ""

    def get_atom(self, atom_id: int) -> Optional[X11Atom]:
        """Get an atom by its ID."""
        return self._atoms.get(atom_id)

    def get_atom_by_name(self, name: str) -> Optional[X11Atom]:
        """Get an atom by its name."""
        atom_id = self._name_to_id.get(name)
        if atom_id is None:
            return None
        return self._atoms.get(atom_id)

    def set_atom_value(self, atom_id: int, value: Any,
                       atom_type: str = "STRING") -> bool:
        """Set the value of an atom."""
        atom = self._atoms.get(atom_id)
        if atom is None:
            return False
        atom.value = value
        atom.atom_type = atom_type
        return True

    def get_atom_value(self, atom_id: int) -> Any:
        """Get the value of an atom."""
        atom = self._atoms.get(atom_id)
        return atom.value if atom else None

    def get_all_atoms(self) -> List[X11Atom]:
        """Get all registered atoms."""
        return list(self._atoms.values())


class PropertyManager:
    """
    X11 window property management.

    Manages properties attached to windows, supporting ICCCM and EWMH.
    """

    def __init__(self, atom_manager: AtomManager) -> None:
        self.atoms = atom_manager
        self._properties: Dict[int, Dict[int, Any]] = {}
        self._property_types: Dict[int, Dict[int, int]] = {}
        self._lock = threading.Lock()

    def change_property(self, window_id: int, atom_id: int,
                        value: Any, type_atom: int = 31) -> bool:
        """Change a window property."""
        with self._lock:
            if window_id not in self._properties:
                self._properties[window_id] = {}
                self._property_types[window_id] = {}
            self._properties[window_id][atom_id] = value
            self._property_types[window_id][atom_id] = type_atom
            return True

    def get_property(self, window_id: int, atom_id: int) -> Optional[Any]:
        """Get a window property value."""
        with self._lock:
            props = self._properties.get(window_id, {})
            return props.get(atom_id)

    def delete_property(self, window_id: int, atom_id: int) -> bool:
        """Delete a window property."""
        with self._lock:
            props = self._properties.get(window_id, {})
            if atom_id in props:
                del props[atom_id]
                types = self._property_types.get(window_id, {})
                types.pop(atom_id, None)
                return True
            return False

    def list_properties(self, window_id: int) -> List[int]:
        """List all property atoms for a window."""
        with self._lock:
            return list(self._properties.get(window_id, {}).keys())

    def get_property_names(self, window_id: int) -> List[str]:
        """Get names of all properties for a window."""
        atom_ids = self.list_properties(window_id)
        return [self.atoms.get_atom_name(aid) for aid in atom_ids]


class EWMHProtocol:
    """
    Extended Window Manager Hints (EWMH) support.

    Implements the _NET_WM_* atoms and protocols for communication
    between clients and the window manager.
    """

    # EWMH supported hints
    EWMH_ATOMS: List[str] = [
        "_NET_SUPPORTED",
        "_NET_CLIENT_LIST",
        "_NET_CLIENT_LIST_STACKING",
        "_NET_NUMBER_OF_DESKTOPS",
        "_NET_DESKTOP_GEOMETRY",
        "_NET_DESKTOP_VIEWPORT",
        "_NET_CURRENT_DESKTOP",
        "_NET_DESKTOP_NAMES",
        "_NET_ACTIVE_WINDOW",
        "_NET_WORKAREA",
        "_NET_SUPPORTING_WM_CHECK",
        "_NET_VIRTUAL_ROOTS",
        "_NET_SHOWING_DESKTOP",
        "_NET_CLOSE_WINDOW",
        "_NET_MOVERESIZE_WINDOW",
        "_NET_WM_NAME",
        "_NET_WM_VISIBLE_NAME",
        "_NET_WM_ICON_NAME",
        "_NET_WM_VISIBLE_ICON_NAME",
        "_NET_WM_DESKTOP",
        "_NET_WM_WINDOW_TYPE",
        "_NET_WM_STATE",
        "_NET_WM_ALLOWED_ACTIONS",
        "_NET_WM_STRUT",
        "_NET_WM_STRUT_PARTIAL",
        "_NET_WM_ICON_GEOMETRY",
        "_NET_WM_ICON",
        "_NET_WM_PID",
        "_NET_WM_USER_TIME",
        "_NET_WM_USER_TIME_WINDOW",
        "_NET_FRAME_EXTENTS",
        "_NET_WM_OPAQUE_REGION",
        "_NET_WM_BYPASS_COMPOSITOR",
    ]

    # Window types
    WINDOW_TYPES: List[str] = [
        "_NET_WM_WINDOW_TYPE_DESKTOP",
        "_NET_WM_WINDOW_TYPE_DOCK",
        "_NET_WM_WINDOW_TYPE_TOOLBAR",
        "_NET_WM_WINDOW_TYPE_MENU",
        "_NET_WM_WINDOW_TYPE_UTILITY",
        "_NET_WM_WINDOW_TYPE_SPLASH",
        "_NET_WM_WINDOW_TYPE_DIALOG",
        "_NET_WM_WINDOW_TYPE_DROPDOWN_MENU",
        "_NET_WM_WINDOW_TYPE_POPUP_MENU",
        "_NET_WM_WINDOW_TYPE_TOOLTIP",
        "_NET_WM_WINDOW_TYPE_NOTIFICATION",
        "_NET_WM_WINDOW_TYPE_COMBO",
        "_NET_WM_WINDOW_TYPE_DND",
        "_NET_WM_WINDOW_TYPE_NORMAL",
    ]

    # Window states
    WINDOW_STATES: List[str] = [
        "_NET_WM_STATE_MODAL",
        "_NET_WM_STATE_STICKY",
        "_NET_WM_STATE_MAXIMIZED_VERT",
        "_NET_WM_STATE_MAXIMIZED_HORZ",
        "_NET_WM_STATE_SHADED",
        "_NET_WM_STATE_SKIP_TASKBAR",
        "_NET_WM_STATE_SKIP_PAGER",
        "_NET_WM_STATE_HIDDEN",
        "_NET_WM_STATE_FULLSCREEN",
        "_NET_WM_STATE_ABOVE",
        "_NET_WM_STATE_BELOW",
        "_NET_WM_STATE_DEMANDS_ATTENTION",
        "_NET_WM_STATE_FOCUSED",
    ]

    def __init__(self, atom_manager: AtomManager,
                 property_manager: PropertyManager) -> None:
        self.atoms = atom_manager
        self.properties = property_manager
        self._supported: Set[str] = set()
        self._init_atoms()

    def _init_atoms(self) -> None:
        """Initialize EWMH atoms."""
        for atom_name in self.EWMH_ATOMS + self.WINDOW_TYPES + self.WINDOW_STATES:
            self.atoms.intern_atom(atom_name)
            self._supported.add(atom_name)

    def get_supported(self) -> List[str]:
        """Get list of supported EWMH hints."""
        return sorted(self._supported)

    def set_wm_name(self, window_id: int, name: str) -> None:
        """Set _NET_WM_NAME for a window."""
        atom_id = self.atoms.intern_atom("_NET_WM_NAME")
        self.properties.change_property(window_id, atom_id, name)

    def get_wm_name(self, window_id: int) -> Optional[str]:
        """Get _NET_WM_NAME for a window."""
        atom_id = self.atoms.intern_atom("_NET_WM_NAME")
        return self.properties.get_property(window_id, atom_id)

    def set_window_type(self, window_id: int, window_type: str) -> None:
        """Set _NET_WM_WINDOW_TYPE for a window."""
        atom_id = self.atoms.intern_atom("_NET_WM_WINDOW_TYPE")
        type_atom = self.atoms.intern_atom(window_type)
        self.properties.change_property(window_id, atom_id, type_atom)

    def get_window_type(self, window_id: int) -> Optional[str]:
        """Get _NET_WM_WINDOW_TYPE for a window."""
        atom_id = self.atoms.intern_atom("_NET_WM_WINDOW_TYPE")
        type_atom = self.properties.get_property(window_id, atom_id)
        if type_atom is not None:
            return self.atoms.get_atom_name(type_atom)
        return None

    def set_window_state(self, window_id: int, states: List[str]) -> None:
        """Set _NET_WM_STATE for a window."""
        atom_id = self.atoms.intern_atom("_NET_WM_STATE")
        state_atoms = [self.atoms.intern_atom(s) for s in states]
        self.properties.change_property(window_id, atom_id, state_atoms)

    def get_window_state(self, window_id: int) -> List[str]:
        """Get _NET_WM_STATE for a window."""
        atom_id = self.atoms.intern_atom("_NET_WM_STATE")
        state_atoms = self.properties.get_property(window_id, atom_id)
        if state_atoms is None:
            return []
        if isinstance(state_atoms, list):
            return [self.atoms.get_atom_name(a) for a in state_atoms]
        return []

    def set_wm_desktop(self, window_id: int, desktop: int) -> None:
        """Set _NET_WM_DESKTOP for a window."""
        atom_id = self.atoms.intern_atom("_NET_WM_DESKTOP")
        self.properties.change_property(window_id, atom_id, desktop)

    def get_wm_desktop(self, window_id: int) -> Optional[int]:
        """Get _NET_WM_DESKTOP for a window."""
        atom_id = self.atoms.intern_atom("_NET_WM_DESKTOP")
        return self.properties.get_property(window_id, atom_id)

    def set_active_window(self, window_id: int) -> None:
        """Set _NET_ACTIVE_WINDOW."""
        atom_id = self.atoms.intern_atom("_NET_ACTIVE_WINDOW")
        self.properties.change_property(0, atom_id, window_id)

    def get_active_window(self) -> Optional[int]:
        """Get _NET_ACTIVE_WINDOW."""
        atom_id = self.atoms.intern_atom("_NET_ACTIVE_WINDOW")
        return self.properties.get_property(0, atom_id)

    def set_pid(self, window_id: int, pid: int) -> None:
        """Set _NET_WM_PID for a window."""
        atom_id = self.atoms.intern_atom("_NET_WM_PID")
        self.properties.change_property(window_id, atom_id, pid)

    def get_pid(self, window_id: int) -> Optional[int]:
        """Get _NET_WM_PID for a window."""
        atom_id = self.atoms.intern_atom("_NET_WM_PID")
        return self.properties.get_property(window_id, atom_id)


class X11Window:
    """
    Simulated X11 window.

    Represents a window with geometry, attributes, and properties.
    """

    _next_id: int = 1
    _lock = threading.Lock()

    def __init__(self, display: X11Display, parent_id: int,
                 x: int = 0, y: int = 0, width: int = 100,
                 height: int = 100, border_width: int = 0,
                 window_class: X11WindowClass = X11WindowClass.INPUT_OUTPUT,
                 depth: int = 24, visual: int = 0,
                 window_id: Optional[int] = None) -> None:
        with X11Window._lock:
            if window_id is not None:
                self.window_id = window_id
            else:
                self.window_id = X11Window._next_id
                X11Window._next_id += 1

        self.display = display
        self.parent_id = parent_id
        self.geometry = X11Geometry(x=x, y=y, width=width, height=height,
                                     border_width=border_width, depth=depth)
        self.window_class = window_class
        self.visual = visual
        self.event_mask = X11EventMask.NO_EVENT_MASK
        self.do_not_propagate_mask = X11EventMask.NO_EVENT_MASK
        self.override_redirect = False
        self.save_under = False
        self.backing_store = 0
        self.backing_planes = 0
        self.backing_pixel = 0
        self.colormap = 0
        self.cursor = 0
        self.is_mapped = False
        self.is_viewable = False
        self.input_focus = False
        self.children: List[int] = []
        self._attributes: Dict[str, Any] = {}
        self.creation_time = time.time()

    def configure(self, x: Optional[int] = None, y: Optional[int] = None,
                  width: Optional[int] = None, height: Optional[int] = None,
                  border_width: Optional[int] = None,
                  sibling: Optional[int] = None,
                  stack_mode: Optional[int] = None) -> None:
        """Configure (resize/move) the window."""
        if x is not None:
            self.geometry.x = x
        if y is not None:
            self.geometry.y = y
        if width is not None:
            self.geometry.width = max(1, width)
        if height is not None:
            self.geometry.height = max(1, height)
        if border_width is not None:
            self.geometry.border_width = border_width

    def map(self) -> None:
        """Map (show) the window."""
        self.is_mapped = True
        self.is_viewable = True

    def unmap(self) -> None:
        """Unmap (hide) the window."""
        self.is_mapped = False
        self.is_viewable = False

    def destroy(self) -> None:
        """Mark the window as destroyed."""
        self.is_mapped = False
        self.is_viewable = False

    def set_attribute(self, name: str, value: Any) -> None:
        """Set a window attribute."""
        self._attributes[name] = value

    def get_attribute(self, name: str, default: Any = None) -> Any:
        """Get a window attribute."""
        return self._attributes.get(name, default)

    def get_geometry(self) -> X11Geometry:
        """Get the window geometry."""
        return self.geometry

    def select_input(self, event_mask: X11EventMask) -> None:
        """Set the event mask for this window."""
        self.event_mask = event_mask

    def __repr__(self) -> str:
        return (f"X11Window(id={self.window_id}, "
                f"geometry=({self.geometry.x},{self.geometry.y},"
                f"{self.geometry.width}x{self.geometry.height}))")


class X11Display:
    """
    Simulated X11 display connection.

    Manages the root window, display information, and global state.
    """

    def __init__(self, display_name: str = ":0",
                 screen: int = 0) -> None:
        self.display_name = display_name
        self.screen = screen
        self.screen_count = 1
        self.root_window_id = 0
        self._windows: Dict[int, X11Window] = {}
        self._event_queue: List[X11Event] = []
        self._event_handlers: Dict[int, List[Callable[[X11Event], None]]] = {}
        self._lock = threading.Lock()
        self._connected = True
        self._error_handler: Optional[Callable] = None

        # Display info
        self.vendor = "Simulated X11"
        self.version_major = 11
        self.version_minor = 0
        self.release_number = 0
        self.image_byte_order = 0  # LSBFirst
        self.bitmap_bit_order = 0  # LSBFirst
        self.bitmap_unit = 32
        self.bitmap_pad = 32
        self.display_width = 1920
        self.display_height = 1080
        self.motion_buffer_size = 256
        self.max_request_size = 65535

        # Screen info
        self.root_depth = 24
        self.root_visual = 0
        self.default_colormap = 0
        self.white_pixel = 0xFFFFFF
        self.black_pixel = 0x000000
        self.min_maps = 1
        self.max_maps = 1
        self.backing_store = 0
        self.save_unders = False
        self.root_input_mask = X11EventMask.SUBSTRUCTURE_REDIRECT_MASK | \
                               X11EventMask.SUBSTRUCTURE_NOTIFY_MASK

        # Initialize subsystems
        self.atoms = AtomManager()
        self.properties = PropertyManager(self.atoms)
        self.ewmh = EWMHProtocol(self.atoms, self.properties)

        # Create root window
        self._create_root_window()

    def _create_root_window(self) -> None:
        """Create the root window."""
        root = X11Window(
            display=self, parent_id=0,
            x=0, y=0, width=self.display_width, height=self.display_height,
            window_class=X11WindowClass.INPUT_OUTPUT,
            window_id=0,
        )
        root.is_mapped = True
        root.is_viewable = True
        self._windows[0] = root
        self.root_window_id = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    def create_window(self, parent_id: int = 0, x: int = 0, y: int = 0,
                      width: int = 100, height: int = 100,
                      border_width: int = 0,
                      window_class: X11WindowClass = X11WindowClass.INPUT_OUTPUT,
                      visual: int = 0, depth: int = 0,
                      event_mask: X11EventMask = X11EventMask.NO_EVENT_MASK) -> X11Window:
        """Create a new window."""
        with self._lock:
            actual_depth = depth if depth > 0 else self.root_depth
            window = X11Window(
                display=self, parent_id=parent_id,
                x=x, y=y, width=width, height=height,
                border_width=border_width,
                window_class=window_class,
                depth=actual_depth, visual=visual,
            )
            window.select_input(event_mask)
            self._windows[window.window_id] = window

            # Add as child of parent
            parent = self._windows.get(parent_id)
            if parent:
                parent.children.append(window.window_id)

            # Generate CreateNotify event
            event = X11Event.create(
                X11EventType.CREATE_NOTIFY, parent_id,
                x=x, y=y,
            )
            event.data = {"new_window": window.window_id}
            self._enqueue_event(event)

            return window

    def destroy_window(self, window_id: int) -> bool:
        """Destroy a window."""
        with self._lock:
            window = self._windows.get(window_id)
            if window is None:
                return False

            # Destroy children first
            for child_id in list(window.children):
                self.destroy_window(child_id)

            window.destroy()

            # Remove from parent
            parent = self._windows.get(window.parent_id)
            if parent:
                parent.children = [c for c in parent.children if c != window_id]

            # Generate DestroyNotify event
            event = X11Event.create(X11EventType.DESTROY_NOTIFY, window.parent_id)
            event.data = {"destroyed_window": window_id}
            self._enqueue_event(event)

            del self._windows[window_id]
            return True

    def map_window(self, window_id: int) -> bool:
        """Map a window."""
        window = self._windows.get(window_id)
        if window is None:
            return False
        window.map()
        event = X11Event.create(X11EventType.MAP_NOTIFY, window.parent_id)
        event.data = {"mapped_window": window_id}
        self._enqueue_event(event)
        return True

    def unmap_window(self, window_id: int) -> bool:
        """Unmap a window."""
        window = self._windows.get(window_id)
        if window is None:
            return False
        window.unmap()
        event = X11Event.create(X11EventType.UNMAP_NOTIFY, window.parent_id)
        event.data = {"unmapped_window": window_id}
        self._enqueue_event(event)
        return True

    def configure_window(self, window_id: int, x: Optional[int] = None,
                         y: Optional[int] = None,
                         width: Optional[int] = None,
                         height: Optional[int] = None,
                         border_width: Optional[int] = None) -> bool:
        """Configure a window."""
        window = self._windows.get(window_id)
        if window is None:
            return False
        window.configure(x=x, y=y, width=width, height=height,
                         border_width=border_width)
        event = X11Event.create(X11EventType.CONFIGURE_NOTIFY, window_id,
                                x=window.geometry.x, y=window.geometry.y)
        event.data = {"width": window.geometry.width, "height": window.geometry.height}
        self._enqueue_event(event)
        return True

    def get_window(self, window_id: int) -> Optional[X11Window]:
        """Get a window by ID."""
        return self._windows.get(window_id)

    def get_root_window(self) -> X11Window:
        """Get the root window."""
        return self._windows[0]

    def query_tree(self, window_id: int) -> Optional[Tuple[int, int, List[int]]]:
        """Query the window tree (parent, root, children)."""
        window = self._windows.get(window_id)
        if window is None:
            return None
        return (window.parent_id, 0, list(window.children))

    def get_window_attributes(self, window_id: int) -> Optional[Dict[str, Any]]:
        """Get window attributes."""
        window = self._windows.get(window_id)
        if window is None:
            return None
        return {
            "backing_store": window.backing_store,
            "visual": window.visual,
            "class": window.window_class,
            "bit_gravity": X11Gravity.STATIC,
            "win_gravity": X11Gravity.NORTH_WEST,
            "backing_planes": window.backing_planes,
            "backing_pixel": window.backing_pixel,
            "save_under": window.save_under,
            "colormap": window.colormap,
            "map_installed": window.is_mapped,
            "map_state": 2 if window.is_viewable else 0,
            "all_event_masks": window.event_mask,
            "your_event_mask": window.event_mask,
            "do_not_propagate_mask": window.do_not_propagate_mask,
            "override_redirect": window.override_redirect,
        }

    def _enqueue_event(self, event: X11Event) -> None:
        """Add an event to the event queue."""
        self._event_queue.append(event)

    def next_event(self) -> Optional[X11Event]:
        """Get the next event from the queue (blocking simulation)."""
        if self._event_queue:
            return self._event_queue.pop(0)
        return None

    def peek_event(self) -> Optional[X11Event]:
        """Peek at the next event without removing it."""
        return self._event_queue[0] if self._event_queue else None

    def pending(self) -> int:
        """Get the number of pending events."""
        return len(self._event_queue)

    def flush(self) -> None:
        """Flush the output buffer (no-op in simulation)."""
        pass

    def sync(self) -> None:
        """Synchronize with the server (no-op in simulation)."""
        pass

    def close(self) -> None:
        """Close the display connection."""
        self._connected = False

    def get_screen_info(self) -> Dict[str, Any]:
        """Get screen information."""
        return {
            "width": self.display_width,
            "height": self.display_height,
            "depth": self.root_depth,
            "vendor": self.vendor,
            "version": (self.version_major, self.version_minor),
        }


class X11EventHandler:
    """
    X11 event handling and dispatch.

    Routes events to registered handlers based on event type and window.
    """

    def __init__(self, display: X11Display) -> None:
        self.display = display
        self._handlers: Dict[int, List[Callable[[X11Event], bool]]] = {}
        self._window_handlers: Dict[int, Dict[int, List[Callable[[X11Event], bool]]]] = {}
        self._running = False

    def register_handler(self, event_type: X11EventType,
                         handler: Callable[[X11Event], bool]) -> None:
        """Register a global event handler."""
        etype = int(event_type)
        if etype not in self._handlers:
            self._handlers[etype] = []
        self._handlers[etype].append(handler)

    def register_window_handler(self, window_id: int, event_type: X11EventType,
                                handler: Callable[[X11Event], bool]) -> None:
        """Register a window-specific event handler."""
        if window_id not in self._window_handlers:
            self._window_handlers[window_id] = {}
        etype = int(event_type)
        if etype not in self._window_handlers[window_id]:
            self._window_handlers[window_id][etype] = []
        self._window_handlers[window_id][etype].append(handler)

    def unregister_handler(self, event_type: X11EventType,
                           handler: Optional[Callable] = None) -> None:
        """Unregister event handler(s)."""
        etype = int(event_type)
        if handler is None:
            self._handlers.pop(etype, None)
        elif etype in self._handlers:
            self._handlers[etype] = [h for h in self._handlers[etype] if h != handler]

    def dispatch_event(self, event: X11Event) -> bool:
        """Dispatch an event to registered handlers."""
        etype = int(event.event_type)
        handled = False

        # Window-specific handlers first
        wid = event.window_id
        if wid in self._window_handlers:
            if etype in self._window_handlers[wid]:
                for handler in self._window_handlers[wid][etype]:
                    if handler(event):
                        handled = True

        # Global handlers
        if etype in self._handlers:
            for handler in self._handlers[etype]:
                if handler(event):
                    handled = True

        return handled

    def process_events(self, max_events: int = 100) -> int:
        """Process pending events."""
        count = 0
        while count < max_events:
            event = self.display.next_event()
            if event is None:
                break
            self.dispatch_event(event)
            count += 1
        return count

    def pump_events(self) -> None:
        """Pump all pending events (event loop iteration)."""
        self.process_events()

    def generate_synthetic_event(self, event_type: X11EventType,
                                window_id: int, **kwargs: Any) -> X11Event:
        """Generate and enqueue a synthetic event."""
        event = X11Event.create(event_type, window_id, **kwargs)
        self.display._enqueue_event(event)
        return event
