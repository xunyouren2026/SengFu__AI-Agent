"""
UI Element Module

Provides smart element waiting and accessibility tree parsing.
"""

from .wait import (
    WaitConditionType,
    PollingStrategyType,
    WaitResult,
    ElementData,
    WaitCondition,
    CompositeCondition,
    PollingStrategy,
    TimeoutManager,
    StalenessDetector,
    ElementWait,
)

from .accessibility_tree import (
    A11yRole,
    A11yState,
    A11yNode,
    RoleDetector,
    StateExtractor,
    TreeTraverser,
    TreeDiffer,
    TreeSerializer,
    AccessibilityTree,
)

__all__ = [
    "WaitConditionType",
    "PollingStrategyType",
    "WaitResult",
    "ElementData",
    "WaitCondition",
    "CompositeCondition",
    "PollingStrategy",
    "TimeoutManager",
    "StalenessDetector",
    "ElementWait",
    "A11yRole",
    "A11yState",
    "A11yNode",
    "RoleDetector",
    "StateExtractor",
    "TreeTraverser",
    "TreeDiffer",
    "TreeSerializer",
    "AccessibilityTree",
]
