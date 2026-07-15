"""
Accessibility Tree Parsing Module

Provides accessibility tree construction and manipulation:
- Tree construction from UI elements
- Role detection
- State properties extraction
- Traversal (BFS/DFS)
- Diff computation between trees
- Serialization to JSON

Pure Python standard library only.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Optional, Any, Callable, Set, Iterator


class A11yRole(Enum):
    """Accessibility roles."""
    UNKNOWN = "unknown"
    WINDOW = "window"
    DIALOG = "dialog"
    ALERT_DIALOG = "alertDialog"
    PANEL = "panel"
    DOCUMENT = "document"
    ARTICLE = "article"
    SECTION = "section"
    GROUP = "group"
    TOOLBAR = "toolbar"
    STATUS_BAR = "statusBar"
    MENU_BAR = "menuBar"
    MENU = "menu"
    MENU_ITEM = "menuItem"
    TAB_LIST = "tabList"
    TAB = "tab"
    TAB_PANEL = "tabPanel"
    BUTTON = "button"
    LINK = "link"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LABEL = "label"
    TEXT_BOX = "textBox"
    SEARCH_BOX = "searchBox"
    TEXT_AREA = "textArea"
    CHECKBOX = "checkBox"
    RADIO = "radio"
    COMBO_BOX = "comboBox"
    LIST_BOX = "listBox"
    LIST_ITEM = "listItem"
    SPIN_BUTTON = "spinButton"
    SLIDER = "slider"
    PROGRESS_BAR = "progressBar"
    SWITCH = "switch"
    TABLE = "table"
    TABLE_CELL = "cell"
    TABLE_HEADER = "columnHeader"
    TABLE_ROW = "row"
    GRID = "grid"
    TREE = "tree"
    TREE_ITEM = "treeItem"
    LOG = "log"
    TIMER = "timer"
    TOOLTIP = "tooltip"
    FIGURE = "figure"
    IMG = "img"
    NAVIGATION = "navigation"
    BANNER = "banner"
    CONTENT_INFO = "contentInfo"
    FORM = "form"
    MAIN = "main"
    REGION = "region"
    SCROLLBAR = "scrollbar"


class A11yState(Enum):
    """Accessibility state properties."""
    FOCUSED = "focused"
    FOCUSABLE = "focusable"
    SELECTED = "selected"
    SELECTABLE = "selectable"
    CHECKED = "checked"
    PRESSED = "pressed"
    EXPANDED = "expanded"
    COLLAPSED = "collapsed"
    DISABLED = "disabled"
    READ_ONLY = "readOnly"
    REQUIRED = "required"
    INVALID = "invalid"
    HIDDEN = "hidden"
    VISITED = "visited"
    BUSY = "busy"
    LIVE = "live"
    ATOMIC = "atomic"
    RELEVANT = "relevant"
    DRAGGABLE = "draggable"
    DROPPABLE = "droppable"
    EDITABLE = "editable"
    MULTI_SELECTABLE = "multiSelectable"
    SELECTED_OPTION = "selectedOption"


# HTML tag to A11y role mapping
TAG_ROLE_MAP: Dict[str, A11yRole] = {
    "button": A11yRole.BUTTON,
    "a": A11yRole.LINK,
    "input": A11yRole.TEXT_BOX,
    "textarea": A11yRole.TEXT_AREA,
    "select": A11yRole.COMBO_BOX,
    "option": A11yRole.LIST_ITEM,
    "checkbox": A11yRole.CHECKBOX,
    "radio": A11yRole.RADIO,
    "h1": A11yRole.HEADING,
    "h2": A11yRole.HEADING,
    "h3": A11yRole.HEADING,
    "h4": A11yRole.HEADING,
    "h5": A11yRole.HEADING,
    "h6": A11yRole.HEADING,
    "p": A11yRole.PARAGRAPH,
    "img": A11yRole.IMG,
    "ul": A11yRole.LIST_BOX,
    "ol": A11yRole.LIST_BOX,
    "li": A11yRole.LIST_ITEM,
    "table": A11yRole.TABLE,
    "tr": A11yRole.TABLE_ROW,
    "td": A11yRole.TABLE_CELL,
    "th": A11yRole.TABLE_HEADER,
    "form": A11yRole.FORM,
    "nav": A11yRole.NAVIGATION,
    "main": A11yRole.MAIN,
    "header": A11yRole.BANNER,
    "footer": A11yRole.CONTENT_INFO,
    "aside": A11yRole.COMPLEMENTARY if hasattr(A11yRole, "COMPLEMENTARY") else A11yRole.SECTION,
    "section": A11yRole.SECTION,
    "article": A11yRole.ARTICLE,
    "dialog": A11yRole.DIALOG,
    "progress": A11yRole.PROGRESS_BAR,
    "meter": A11yRole.PROGRESS_BAR,
    "slider": A11yRole.SLIDER,
    "details": A11yRole.DISCLOSURE if hasattr(A11yRole, "DISCLOSURE") else A11yRole.SECTION,
    "summary": A11yRole.BUTTON,
    "label": A11yRole.LABEL,
    "fieldset": A11yRole.GROUP,
    "legend": A11yRole.LABEL,
    "iframe": A11yRole.DOCUMENT,
    "div": A11yRole.GENERIC if hasattr(A11yRole, "GENERIC") else A11yRole.GROUP,
    "span": A11yRole.GENERIC if hasattr(A11yRole, "GENERIC") else A11yRole.UNKNOWN,
}


@dataclass
class A11yNode:
    """A node in the accessibility tree."""
    node_id: str
    role: A11yRole = A11yRole.UNKNOWN
    name: str = ""
    description: str = ""
    value: str = ""
    states: Set[A11yState] = field(default_factory=set)
    attributes: Dict[str, str] = field(default_factory=dict)
    children: List[A11yNode] = field(default_factory=list)
    parent: Optional[A11yNode] = field(default=None, repr=False)
    bounds: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0, "w": 0, "h": 0})
    platform_node: Optional[Any] = None
    depth: int = 0
    index_in_parent: int = 0
    text_content: str = ""

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def has_text(self) -> bool:
        return bool(self.name or self.value or self.text_content)

    @property
    def is_interactive(self) -> bool:
        interactive_roles = {
            A11yRole.BUTTON, A11yRole.LINK, A11yRole.TEXT_BOX,
            A11yRole.TEXT_AREA, A11yRole.COMBO_BOX, A11yRole.CHECKBOX,
            A11yRole.RADIO, A11yRole.SLIDER, A11yRole.SWITCH,
            A11yRole.SPIN_BUTTON, A11yRole.TAB, A11yRole.MENU_ITEM,
            A11yRole.TREE_ITEM, A11yRole.LIST_ITEM,
        }
        return self.role in interactive_roles or A11yState.FOCUSABLE in self.states

    def get_all_text(self) -> str:
        """Get all text content including children."""
        parts: List[str] = []
        if self.text_content:
            parts.append(self.text_content)
        if self.name:
            parts.append(self.name)
        if self.value:
            parts.append(self.value)
        for child in self.children:
            child_text = child.get_all_text()
            if child_text:
                parts.append(child_text)
        return " ".join(parts)

    def find_by_role(self, role: A11yRole) -> List[A11yNode]:
        """Find all descendants with a given role."""
        results: List[A11yNode] = []
        if self.role == role:
            results.append(self)
        for child in self.children:
            results.extend(child.find_by_role(role))
        return results

    def find_by_name(self, name: str) -> List[A11yNode]:
        """Find all descendants with a given name."""
        results: List[A11yNode] = []
        if self.name == name:
            results.append(self)
        for child in self.children:
            results.extend(child.find_by_name(name))
        return results

    def find_by_id(self, node_id: str) -> Optional[A11yNode]:
        """Find a descendant by node ID."""
        if self.node_id == node_id:
            return self
        for child in self.children:
            found = child.find_by_id(node_id)
            if found:
                return found
        return None

    def get_path(self) -> str:
        """Get the path from root to this node."""
        parts: List[str] = []
        current: Optional[A11yNode] = self
        while current is not None:
            parts.append(f"{current.role.value}[{current.index_in_parent}]")
            current = current.parent
        parts.reverse()
        return "/".join(parts)

    def compute_hash(self) -> str:
        """Compute a content hash for diff comparison."""
        content = f"{self.role.value}:{self.name}:{self.value}:{self.text_content}"
        for state in sorted(self.states, key=lambda s: s.value):
            content += f":{state.value}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


class RoleDetector:
    """
    Detects the accessibility role of an element.

    Uses tag name, ARIA role attribute, and heuristics.
    """

    def __init__(self) -> None:
        self._aria_role_map: Dict[str, A11yRole] = {}
        self._build_aria_map()

    def _build_aria_map(self) -> None:
        """Build ARIA role to A11yRole mapping."""
        for role in A11yRole:
            self._aria_role_map[role.value] = role

    def detect(self, tag: str = "", aria_role: str = "",
               attributes: Optional[Dict[str, str]] = None,
               has_click_handler: bool = False) -> A11yRole:
        """Detect the accessibility role."""
        # ARIA role takes highest priority
        if aria_role:
            normalized = aria_role.lower().replace("-", "").replace("_", "")
            for aria_name, role in self._aria_role_map.items():
                if aria_name.lower().replace("-", "").replace("_", "") == normalized:
                    return role

        # Tag-based detection
        tag_lower = tag.lower().strip()
        if tag_lower in TAG_ROLE_MAP:
            return TAG_ROLE_MAP[tag_lower]

        # Attribute-based heuristics
        attrs = attributes or {}
        if attrs.get("role"):
            return self.detect(tag, attrs["role"], attributes)

        if attrs.get("type") in ("button", "submit", "reset"):
            return A11yRole.BUTTON
        if attrs.get("type") == "checkbox":
            return A11yRole.CHECKBOX
        if attrs.get("type") == "radio":
            return A11yRole.RADIO
        if attrs.get("type") == "range":
            return A11yRole.SLIDER
        if attrs.get("href"):
            return A11yRole.LINK
        if attrs.get("contenteditable") == "true":
            return A11yRole.TEXT_AREA
        if attrs.get("tabindex") is not None:
            if has_click_handler:
                return A11yRole.BUTTON
            return A11yRole.GROUP

        return A11yRole.UNKNOWN


class StateExtractor:
    """
    Extracts accessibility states from element attributes.

    Maps HTML/ARIA attributes to A11yState values.
    """

    # Attribute to state mapping
    ATTRIBUTE_STATE_MAP: Dict[str, Tuple[A11yState, Any]] = {
        "disabled": (A11yState.DISABLED, True),
        "readonly": (A11yState.READ_ONLY, True),
        "required": (A11yState.REQUIRED, True),
        "hidden": (A11yState.HIDDEN, True),
        "aria-disabled": (A11yState.DISABLED, "true"),
        "aria-readonly": (A11yState.READ_ONLY, "true"),
        "aria-required": (A11yState.REQUIRED, "true"),
        "aria-hidden": (A11yState.HIDDEN, "true"),
        "aria-checked": (A11yState.CHECKED, "true"),
        "aria-selected": (A11yState.SELECTED, "true"),
        "aria-pressed": (A11yState.PRESSED, "true"),
        "aria-expanded": (A11yState.EXPANDED, "true"),
        "aria-busy": (A11yState.BUSY, "true"),
        "aria-focusable": (A11yState.FOCUSABLE, "true"),
        "aria-multiselectable": (A11yState.MULTI_SELECTABLE, "true"),
        "aria-dropeffect": (A11yState.DROPPABLE, None),
        "aria-grabbed": (A11yState.DRAGGABLE, "true"),
        "aria-invalid": (A11yState.INVALID, "true"),
        "aria-live": (A11yState.LIVE, None),
        "aria-atomic": (A11yState.ATOMIC, "true"),
        "aria-relevant": (A11yState.RELEVANT, None),
        "aria-editable": (A11yState.EDITABLE, "true"),
    }

    def extract(self, attributes: Dict[str, str],
                tag: str = "") -> Set[A11yState]:
        """Extract states from element attributes."""
        states: Set[A11yState] = set()

        for attr_name, (state, expected_value) in self.ATTRIBUTE_STATE_MAP.items():
            attr_value = attributes.get(attr_name)
            if attr_value is None:
                continue

            if expected_value is None:
                # Presence-based state
                states.add(state)
            elif isinstance(expected_value, bool):
                if attr_value.lower() in ("true", ""):
                    states.add(state)
            elif isinstance(expected_value, str):
                if attr_value.lower() == expected_value.lower():
                    states.add(state)

        # Auto-detect focusable
        focusable_tags = {"a", "button", "input", "select", "textarea", "details", "summary"}
        if tag.lower() in focusable_tags or attributes.get("tabindex") is not None:
            if A11yState.DISABLED not in states:
                states.add(A11yState.FOCUSABLE)

        # Auto-detect selected for options
        if tag.lower() == "option" and attributes.get("selected"):
            states.add(A11yState.SELECTED)

        return states


class TreeTraverser:
    """
    Traverses accessibility trees using various strategies.

    Supports BFS, DFS (pre-order, in-order, post-order), and custom filters.
    """

    def dfs_preorder(self, root: A11yNode,
                     filter_fn: Optional[Callable[[A11yNode], bool]] = None) -> Iterator[A11yNode]:
        """Depth-first pre-order traversal."""
        stack: List[A11yNode] = [root]
        while stack:
            node = stack.pop()
            if filter_fn is None or filter_fn(node):
                yield node
            for child in reversed(node.children):
                stack.append(child)

    def dfs_inorder(self, root: A11yNode,
                    filter_fn: Optional[Callable[[A11yNode], bool]] = None) -> Iterator[A11yNode]:
        """Depth-first in-order traversal."""
        def _inorder(node: A11yNode) -> Iterator[A11yNode]:
            if not node.children:
                if filter_fn is None or filter_fn(node):
                    yield node
            else:
                for i, child in enumerate(node.children):
                    if i == len(node.children) // 2:
                        if filter_fn is None or filter_fn(node):
                            yield node
                    yield from _inorder(child)
                if len(node.children) == 1:
                    if filter_fn is None or filter_fn(node):
                        yield node
        yield from _inorder(root)

    def dfs_postorder(self, root: A11yNode,
                      filter_fn: Optional[Callable[[A11yNode], bool]] = None) -> Iterator[A11yNode]:
        """Depth-first post-order traversal."""
        def _postorder(node: A11yNode) -> Iterator[A11yNode]:
            for child in node.children:
                yield from _postorder(child)
            if filter_fn is None or filter_fn(node):
                yield node
        yield from _postorder(root)

    def bfs(self, root: A11yNode,
            filter_fn: Optional[Callable[[A11yNode], bool]] = None) -> Iterator[A11yNode]:
        """Breadth-first traversal."""
        queue: List[A11yNode] = [root]
        while queue:
            node = queue.pop(0)
            if filter_fn is None or filter_fn(node):
                yield node
            queue.extend(node.children)

    def find_all(self, root: A11yNode,
                 predicate: Callable[[A11yNode], bool]) -> List[A11yNode]:
        """Find all nodes matching a predicate using BFS."""
        return [node for node in self.bfs(root, predicate)]

    def find_first(self, root: A11yNode,
                   predicate: Callable[[A11yNode], bool]) -> Optional[A11yNode]:
        """Find the first node matching a predicate using BFS."""
        for node in self.bfs(root, predicate):
            return node
        return None

    def count_nodes(self, root: A11yNode) -> int:
        """Count total nodes in the tree."""
        return sum(1 for _ in self.bfs(root))

    def count_leaves(self, root: A11yNode) -> int:
        """Count leaf nodes in the tree."""
        return sum(1 for node in self.bfs(root) if node.is_leaf)

    def get_max_depth(self, root: A11yNode) -> int:
        """Get the maximum depth of the tree."""
        if not root.children:
            return 0
        return 1 + max(self.get_max_depth(child) for child in root.children)

    def flatten(self, root: A11yNode) -> List[Dict[str, Any]]:
        """Flatten the tree into a list of node summaries."""
        result: List[Dict[str, Any]] = []
        for node in self.bfs(root):
            result.append({
                "node_id": node.node_id,
                "role": node.role.value,
                "name": node.name,
                "depth": node.depth,
                "child_count": len(node.children),
                "is_interactive": node.is_interactive,
            })
        return result


class TreeDiffer:
    """
    Computes diffs between two accessibility trees.

    Detects added, removed, and modified nodes.
    """

    @dataclass
    class DiffResult:
        """Result of tree diff."""
        added_nodes: List[A11yNode] = field(default_factory=list)
        removed_nodes: List[A11yNode] = field(default_factory=list)
        modified_nodes: List[Tuple[A11yNode, A11yNode]] = field(default_factory=list)
        reordered_nodes: List[Tuple[str, List[str]]] = field(default_factory=list)

        @property
        def has_changes(self) -> bool:
            return bool(self.added_nodes or self.removed_nodes or
                        self.modified_nodes or self.reordered_nodes)

        def summary(self) -> str:
            return (f"Diff: +{len(self.added_nodes)} -{len(self.removed_nodes)} "
                    f"~{len(self.modified_nodes)} reorder:{len(self.reordered_nodes)}")

    def diff(self, old_tree: A11yNode, new_tree: A11yNode) -> DiffResult:
        """Compute the diff between two trees."""
        result = self.DiffResult()
        self._diff_nodes(old_tree, new_tree, result)
        return result

    def _diff_nodes(self, old_node: A11yNode, new_node: A11yNode,
                    result: DiffResult) -> None:
        """Recursively diff two nodes."""
        # Check if the node itself changed
        if old_node.compute_hash() != new_node.compute_hash():
            result.modified_nodes.append((old_node, new_node))

        # Check for added/removed children
        old_ids = {c.node_id for c in old_node.children}
        new_ids = {c.node_id for c in new_node.children}

        # Removed children
        for child in old_node.children:
            if child.node_id not in new_ids:
                result.removed_nodes.append(child)

        # Added children
        for child in new_node.children:
            if child.node_id not in old_ids:
                result.added_nodes.append(child)

        # Recursively diff common children
        old_map = {c.node_id: c for c in old_node.children}
        new_map = {c.node_id: c for c in new_node.children}

        for node_id in old_ids & new_ids:
            self._diff_nodes(old_map[node_id], new_map[node_id], result)

        # Check for reordering
        old_order = [c.node_id for c in old_node.children if c.node_id in new_ids]
        new_order = [c.node_id for c in new_node.children if c.node_id in old_ids]
        if old_order != new_order:
            result.reordered_nodes.append((old_node.node_id, new_order))


class TreeSerializer:
    """
    Serializes accessibility trees to various formats.

    Supports JSON, compact text, and flat representations.
    """

    def to_json(self, root: A11yNode, pretty: bool = True) -> str:
        """Serialize the tree to JSON."""
        data = self._node_to_dict(root)
        if pretty:
            return json.dumps(data, indent=2, ensure_ascii=False)
        return json.dumps(data, ensure_ascii=False)

    def _node_to_dict(self, node: A11yNode) -> Dict[str, Any]:
        """Convert a node to a dictionary."""
        result: Dict[str, Any] = {
            "nodeId": node.node_id,
            "role": node.role.value,
            "name": node.name,
            "description": node.description,
            "value": node.value,
            "states": sorted([s.value for s in node.states]),
            "attributes": dict(node.attributes),
            "bounds": dict(node.bounds),
            "depth": node.depth,
            "indexInParent": node.index_in_parent,
        }
        if node.children:
            result["children"] = [self._node_to_dict(c) for c in node.children]
        return result

    def from_json(self, json_str: str) -> A11yNode:
        """Deserialize a tree from JSON."""
        data = json.loads(json_str)
        return self._dict_to_node(data)

    def _dict_to_node(self, data: Dict[str, Any],
                      parent: Optional[A11yNode] = None) -> A11yNode:
        """Convert a dictionary to an A11yNode."""
        role_str = data.get("role", "unknown")
        try:
            role = A11yRole(role_str)
        except ValueError:
            role = A11yRole.UNKNOWN

        states: Set[A11yState] = set()
        for state_str in data.get("states", []):
            try:
                states.add(A11yState(state_str))
            except ValueError:
                pass

        node = A11yNode(
            node_id=data.get("nodeId", ""),
            role=role,
            name=data.get("name", ""),
            description=data.get("description", ""),
            value=data.get("value", ""),
            states=states,
            attributes=data.get("attributes", {}),
            parent=parent,
            bounds=data.get("bounds", {}),
            depth=data.get("depth", 0),
            index_in_parent=data.get("indexInParent", 0),
        )

        for child_data in data.get("children", []):
            child = self._dict_to_node(child_data, node)
            child.depth = node.depth + 1
            node.children.append(child)

        return node

    def to_compact_text(self, root: A11yNode, max_depth: int = -1) -> str:
        """Serialize to a compact text representation."""
        lines: List[str] = []
        self._node_to_text(root, lines, "", max_depth)
        return "\n".join(lines)

    def _node_to_text(self, node: A11yNode, lines: List[str],
                      indent: str, max_depth: int) -> None:
        """Convert a node to text lines."""
        if max_depth >= 0 and node.depth > max_depth:
            return

        name_part = f' "{node.name}"' if node.name else ""
        value_part = f' = "{node.value}"' if node.value else ""
        states_part = ""
        if node.states:
            states_part = " [" + ", ".join(s.value for s in sorted(node.states, key=lambda s: s.value)) + "]"

        line = f"{indent}{node.role.value}{name_part}{value_part}{states_part}"
        lines.append(line)

        for child in node.children:
            self._node_to_text(child, lines, indent + "  ", max_depth)

    def to_flat_list(self, root: A11yNode) -> List[Dict[str, Any]]:
        """Flatten the tree into a list of node dictionaries."""
        traverser = TreeTraverser()
        return traverser.flatten(root)


class AccessibilityTree:
    """
    High-level accessibility tree management.

    Provides tree construction, querying, diffing, and serialization.
    """

    def __init__(self) -> None:
        self.root: Optional[A11yNode] = None
        self.role_detector = RoleDetector()
        self.state_extractor = StateExtractor()
        self.traverser = TreeTraverser()
        self.differ = TreeDiffer()
        self.serializer = TreeSerializer()

    def build_from_elements(self, elements: List[Dict[str, Any]]) -> A11yNode:
        """
        Build an accessibility tree from a flat list of element dictionaries.

        Each element dict should have: id, tag, text, attributes, children_ids
        """
        # Build node map
        node_map: Dict[str, A11yNode] = {}
        for elem in elements:
            role = self.role_detector.detect(
                tag=elem.get("tag", ""),
                aria_role=elem.get("attributes", {}).get("role", ""),
                attributes=elem.get("attributes", {}),
            )
            states = self.state_extractor.extract(
                elem.get("attributes", {}),
                elem.get("tag", ""),
            )
            node = A11yNode(
                node_id=elem.get("id", ""),
                role=role,
                name=elem.get("name", "") or elem.get("text", ""),
                value=elem.get("value", ""),
                states=states,
                attributes=elem.get("attributes", {}),
                bounds=elem.get("bounds", {}),
                text_content=elem.get("text", ""),
            )
            node_map[node.node_id] = node

        # Build tree structure
        for elem in elements:
            parent_id = elem.get("parent_id")
            child_ids = elem.get("children_ids", [])
            node = node_map.get(elem.get("id", ""))
            if node is None:
                continue

            if parent_id and parent_id in node_map:
                node.parent = node_map[parent_id]
                node.parent.children.append(node)
                node.index_in_parent = len(node.parent.children) - 1

            for cid in child_ids:
                if cid in node_map:
                    child = node_map[cid]
                    child.parent = node
                    node.children.append(child)

        # Set depths
        roots = [n for n in node_map.values() if n.parent is None]
        if roots:
            self.root = roots[0]
            self._set_depths(self.root, 0)
        elif node_map:
            self.root = list(node_map.values())[0]

        return self.root

    def _set_depths(self, node: A11yNode, depth: int) -> None:
        """Recursively set node depths."""
        node.depth = depth
        for i, child in enumerate(node.children):
            child.index_in_parent = i
            self._set_depths(child, depth + 1)

    def find_by_role(self, role: A11yRole) -> List[A11yNode]:
        """Find all nodes with a given role."""
        if self.root is None:
            return []
        return self.root.find_by_role(role)

    def find_by_name(self, name: str) -> List[A11yNode]:
        """Find all nodes with a given name."""
        if self.root is None:
            return []
        return self.root.find_by_name(name)

    def find_interactive_elements(self) -> List[A11yNode]:
        """Find all interactive elements."""
        if self.root is None:
            return []
        return self.traverser.find_all(self.root, lambda n: n.is_interactive)

    def find_visible_text(self) -> str:
        """Get all visible text from the tree."""
        if self.root is None:
            return ""
        return self.root.get_all_text()

    def diff(self, other: AccessibilityTree) -> TreeDiffer.DiffResult:
        """Compute diff with another tree."""
        if self.root is None or other.root is None:
            return TreeDiffer.DiffResult()
        return self.differ.diff(self.root, other.root)

    def to_json(self, pretty: bool = True) -> str:
        """Serialize to JSON."""
        if self.root is None:
            return "{}"
        return self.serializer.to_json(self.root, pretty)

    def from_json(self, json_str: str) -> None:
        """Deserialize from JSON."""
        self.root = self.serializer.from_json(json_str)

    def to_text(self, max_depth: int = -1) -> str:
        """Serialize to compact text."""
        if self.root is None:
            return ""
        return self.serializer.to_compact_text(self.root, max_depth)

    def get_statistics(self) -> Dict[str, int]:
        """Get tree statistics."""
        if self.root is None:
            return {"total_nodes": 0, "leaves": 0, "max_depth": 0, "interactive": 0}
        return {
            "total_nodes": self.traverser.count_nodes(self.root),
            "leaves": self.traverser.count_leaves(self.root),
            "max_depth": self.traverser.get_max_depth(self.root),
            "interactive": len(self.find_interactive_elements()),
        }
