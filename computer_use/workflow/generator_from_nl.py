"""
Natural Language to Workflow Generation Module

Generates executable workflows from natural language descriptions:
- Intent parsing
- Action extraction
- Parameter inference
- Workflow construction from descriptions
- Template library

Pure Python standard library only.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Optional, Any, Callable


class ActionType(Enum):
    """Types of workflow actions."""
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    TYPE = "type"
    PRESS_KEY = "press_key"
    NAVIGATE = "navigate"
    SCROLL = "scroll"
    WAIT = "wait"
    EXTRACT = "extract"
    ASSERT = "assert"
    SCREENSHOT = "screenshot"
    SELECT = "select"
    HOVER = "hover"
    DRAG = "drag"
    UPLOAD = "upload"
    DOWNLOAD = "download"
    SWITCH_TAB = "switch_tab"
    CLOSE_TAB = "close_tab"
    NEW_TAB = "new_tab"
    GO_BACK = "go_back"
    GO_FORWARD = "go_forward"
    REFRESH = "refresh"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    RESIZE = "resize"
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    IF = "if"
    LOOP = "loop"
    UNKNOWN = "unknown"


@dataclass
class ParsedIntent:
    """Parsed intent from natural language."""
    action: ActionType
    target: str = ""
    value: str = ""
    modifier: str = ""
    condition: str = ""
    confidence: float = 1.0
    raw_text: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    position: int = 0


@dataclass
class ExtractedAction:
    """An action extracted from natural language."""
    action_type: ActionType
    target: str = ""
    value: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    order: int = 0
    is_conditional: bool = False
    condition_text: str = ""
    loop_count: int = 0
    loop_variable: str = ""


class IntentParser:
    """
    Parses natural language to determine user intent.

    Uses keyword matching, pattern recognition, and context analysis
    to identify the intended action.
    """

    # Action keyword patterns
    ACTION_PATTERNS: List[Tuple[str, ActionType, float]] = [
        (r"\bclick\s+(?:on\s+)?(?:the\s+)?(.+?)(?:\s+button)?$", ActionType.CLICK, 0.95),
        (r"\bdouble[\s-]click\s+(?:on\s+)?(?:the\s+)?(.+?)(?:\s+button)?$", ActionType.DOUBLE_CLICK, 0.95),
        (r"\bright[\s-]click\s+(?:on\s+)?(?:the\s+)?(.+?)(?:\s+button)?$", ActionType.RIGHT_CLICK, 0.95),
        (r"\btype\s+[\"'](.+?)[\"']\s+(?:in(?:to)?|on)\s+(?:the\s+)?(.+?)(?:\s+field)?$",
         ActionType.TYPE, 0.9),
        (r"\benter\s+[\"'](.+?)[\"']\s+(?:in(?:to)?|on)\s+(?:the\s+)?(.+?)(?:\s+field)?$",
         ActionType.TYPE, 0.9),
        (r"\btype\s+[\"'](.+?)[\"']$", ActionType.TYPE, 0.8),
        (r"\bpress\s+(?:the\s+)?(.+?)\s+key$", ActionType.PRESS_KEY, 0.9),
        (r"\bpress\s+(.+)$", ActionType.PRESS_KEY, 0.7),
        (r"\bgo\s+to\s+(.+)$", ActionType.NAVIGATE, 0.95),
        (r"\bnavigate\s+to\s+(.+)$", ActionType.NAVIGATE, 0.95),
        (r"\bopen\s+(.+?)(?:\s+page)?$", ActionType.NAVIGATE, 0.85),
        (r"\bvisit\s+(.+)$", ActionType.NAVIGATE, 0.9),
        (r"\bscroll\s+(?:down|up)\s+(?:by\s+)?(\d+)?\s*(?:pixels|px)?$", ActionType.SCROLL, 0.9),
        (r"\bscroll\s+(.+)$", ActionType.SCROLL, 0.85),
        (r"\bwait\s+(?:for\s+)?(\d+)?\s*(seconds?|secs?|minutes?|mins?)?$", ActionType.WAIT, 0.9),
        (r"\bextract\s+(?:the\s+)?(.+?)(?:\s+from\s+(.+))?$", ActionType.EXTRACT, 0.85),
        (r"\bget\s+(?:the\s+)?(.+?)(?:\s+from\s+(.+))?$", ActionType.EXTRACT, 0.7),
        (r"\bverify\s+(?:that\s+)?(.+)$", ActionType.ASSERT, 0.85),
        (r"\bcheck\s+(?:that\s+)?(.+)$", ActionType.ASSERT, 0.7),
        (r"\bassert\s+(?:that\s+)?(.+)$", ActionType.ASSERT, 0.9),
        (r"\btake\s+(?:a\s+)?screenshot$", ActionType.SCREENSHOT, 0.95),
        (r"\bcapture\s+(?:a\s+)?screenshot$", ActionType.SCREENSHOT, 0.95),
        (r"\bscreenshot\s+(?:the\s+)?(.+)?$", ActionType.SCREENSHOT, 0.9),
        (r"\bselect\s+[\"'](.+?)[\"']\s+from\s+(?:the\s+)?(.+?)(?:\s+dropdown)?$",
         ActionType.SELECT, 0.9),
        (r"\bhover\s+(?:over|on)\s+(?:the\s+)?(.+)$", ActionType.HOVER, 0.9),
        (r"\bdrag\s+(.+?)\s+to\s+(.+)$", ActionType.DRAG, 0.9),
        (r"\bupload\s+(?:the\s+)?(?:file\s+)?(.+)$", ActionType.UPLOAD, 0.85),
        (r"\bdownload\s+(?:the\s+)?(?:file\s+)?(.+)?$", ActionType.DOWNLOAD, 0.85),
        (r"\bswitch\s+to\s+(?:tab\s+)?(\d+|.+)$", ActionType.SWITCH_TAB, 0.9),
        (r"\bclose\s+(?:the\s+)?tab$", ActionType.CLOSE_TAB, 0.9),
        (r"\bopen\s+(?:a\s+)?new\s+tab$", ActionType.NEW_TAB, 0.9),
        (r"\bgo\s+back$", ActionType.GO_BACK, 0.95),
        (r"\bgo\s+forward$", ActionType.GO_FORWARD, 0.95),
        (r"\brefresh\s+(?:the\s+)?(?:page)?$", ActionType.REFRESH, 0.95),
        (r"\breload\s+(?:the\s+)?(?:page)?$", ActionType.REFRESH, 0.9),
        (r"\bzoom\s+in$", ActionType.ZOOM_IN, 0.9),
        (r"\bzoom\s+out$", ActionType.ZOOM_OUT, 0.9),
        (r"\bmaximize\s+(?:the\s+)?(?:window)?$", ActionType.MAXIMIZE, 0.9),
        (r"\bminimize\s+(the\s+)?(?:window)?$", ActionType.MINIMIZE, 0.9),
        (r"\bif\s+(.+?)\s*,?\s*then\s+(.+)$", ActionType.IF, 0.8),
        (r"\bfor\s+(?:each\s+)?(.+?)\s+in\s+(.+)$", ActionType.LOOP, 0.8),
        (r"\brepeat\s+(\d+)\s+times?\s*,?\s*(.+)$", ActionType.LOOP, 0.85),
    ]

    def parse(self, text: str) -> ParsedIntent:
        """Parse natural language text into an intent."""
        text = text.strip()
        if not text:
            return ParsedIntent(action=ActionType.UNKNOWN, raw_text=text)

        # Try each pattern
        best_intent: Optional[ParsedIntent] = None
        best_confidence = 0.0

        for pattern, action_type, confidence in self.ACTION_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                intent = ParsedIntent(
                    action=action_type,
                    raw_text=text,
                    confidence=confidence,
                    position=match.start(),
                )

                # Extract target and value from groups
                if len(groups) >= 1 and groups[0]:
                    if action_type == ActionType.TYPE and len(groups) >= 2:
                        intent.value = groups[0]
                        intent.target = groups[1]
                    elif action_type == ActionType.SELECT and len(groups) >= 2:
                        intent.value = groups[0]
                        intent.target = groups[1]
                    elif action_type == ActionType.EXTRACT and len(groups) >= 2:
                        intent.target = groups[0]
                        intent.parameters["from"] = groups[1]
                    elif action_type == ActionType.DRAG and len(groups) >= 2:
                        intent.target = groups[0]
                        intent.parameters["destination"] = groups[1]
                    elif action_type == ActionType.IF and len(groups) >= 2:
                        intent.condition = groups[0]
                        intent.target = groups[1]
                    elif action_type == ActionType.LOOP and len(groups) >= 2:
                        intent.parameters["variable"] = groups[0]
                        intent.parameters["collection"] = groups[1]
                    elif action_type == ActionType.LOOP and len(groups) >= 1:
                        try:
                            intent.loop_count = int(groups[0])
                        except ValueError:
                            pass
                    elif action_type == ActionType.WAIT:
                        duration = self._parse_duration(groups[0], groups[1] if len(groups) > 1 else "")
                        intent.parameters["duration"] = duration
                    elif action_type == ActionType.SCROLL:
                        direction = "down" if "down" in text.lower() else "up"
                        amount = int(groups[0]) if groups[0] else 300
                        intent.parameters["direction"] = direction
                        intent.parameters["amount"] = amount
                    else:
                        intent.target = groups[0]

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_intent = intent

        if best_intent:
            return best_intent

        # Fallback: try to guess from keywords
        return self._fallback_parse(text)

    def _fallback_parse(self, text: str) -> ParsedIntent:
        """Fallback parsing using keyword analysis."""
        text_lower = text.lower()
        words = text_lower.split()

        if any(w in words for w in ("click", "press", "tap")):
            target = text
            for w in ("click", "on", "the", "button", "link"):
                target = target.replace(w, "", 1).strip()
            return ParsedIntent(action=ActionType.CLICK, target=target, raw_text=text, confidence=0.5)

        if any(w in words for w in ("type", "enter", "input")):
            return ParsedIntent(action=ActionType.TYPE, target="", value=text, raw_text=text, confidence=0.4)

        if any(w in words for w in ("open", "go", "navigate")):
            return ParsedIntent(action=ActionType.NAVIGATE, target=text, raw_text=text, confidence=0.4)

        return ParsedIntent(action=ActionType.UNKNOWN, raw_text=text, confidence=0.1)

    def _parse_duration(self, amount: str, unit: str) -> float:
        """Parse a duration string into seconds."""
        try:
            value = float(amount) if amount else 1.0
        except ValueError:
            value = 1.0

        if not unit:
            return value

        unit_lower = unit.lower().rstrip("s")
        if unit_lower in ("second", "sec"):
            return value
        elif unit_lower in ("minute", "min"):
            return value * 60
        elif unit_lower in ("hour", "hr"):
            return value * 3600
        elif unit_lower in ("millisecond", "ms"):
            return value / 1000
        return value

    def parse_multi(self, text: str) -> List[ParsedIntent]:
        """Parse multiple intents from a multi-step description."""
        # Split by sentence-ending punctuation or step separators
        sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)\s*|\bthen\b\s+|\band then\b\s+|\bafter that\b\s+',
                             text, flags=re.IGNORECASE)
        intents: List[ParsedIntent] = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                intent = self.parse(sentence)
                if intent.action != ActionType.UNKNOWN:
                    intents.append(intent)
        return intents


class ActionExtractor:
    """
    Extracts structured actions from parsed intents.

    Converts parsed intents into executable action specifications.
    """

    def extract(self, intent: ParsedIntent) -> ExtractedAction:
        """Extract a structured action from a parsed intent."""
        action = ExtractedAction(
            action_type=intent.action,
            target=intent.target,
            value=intent.value,
            parameters=dict(intent.parameters),
            is_conditional=bool(intent.condition),
            condition_text=intent.condition,
        )

        # Infer selector from target text
        if intent.target:
            selector = self._infer_selector(intent.target)
            action.parameters["selector"] = selector

        return action

    def extract_batch(self, intents: List[ParsedIntent]) -> List[ExtractedAction]:
        """Extract actions from multiple intents."""
        actions: List[ExtractedAction] = []
        for i, intent in enumerate(intents):
            action = self.extract(intent)
            action.order = i
            actions.append(action)
        return actions

    def _infer_selector(self, target: str) -> str:
        """
        Infer a CSS/XPath selector from natural language target description.

        Uses heuristics to map common descriptions to selectors.
        """
        target_lower = target.lower().strip()

        # Remove articles
        target_clean = re.sub(r'\b(the|a|an)\b', '', target_lower).strip()

        # Common patterns
        patterns = [
            (r'(.+?)\s+button', r'button:has-text("{text}")'),
            (r'(.+?)\s+link', r'a:has-text("{text}")'),
            (r'(.+?)\s+input', r'input[placeholder*="{text}"]'),
            (r'(.+?)\s+field', r'input[name*="{text}"], textarea[name*="{text}"]'),
            (r'(.+?)\s+dropdown', r'select[name*="{text}"]'),
            (r'(.+?)\s+checkbox', r'input[type="checkbox"][name*="{text}"]'),
            (r'(.+?)\s+tab', r'[role="tab"]:has-text("{text}")'),
            (r'(.+?)\s+menu', r'[role="menu"]:has-text("{text}")'),
            (r'the\s+(.+)', r'[aria-label="{text}"], [title="{text}"]'),
            (r'element with id ["\'](.+?)["\']', r'#{text}'),
            (r'element with class ["\'](.+?)["\']', r'.{text}'),
        ]

        for pattern, selector_template in patterns:
            match = re.search(pattern, target_lower)
            if match:
                text = match.group(1).strip()
                return selector_template.format(text=text)

        # Default: text-based selector
        return f'*:has-text("{target_clean}")'


class ParameterInferrer:
    """
    Infers missing parameters for actions.

    Uses context, defaults, and heuristics to fill in missing values.
    """

    # Default timeouts for different actions
    DEFAULT_TIMEOUTS: Dict[ActionType, float] = {
        ActionType.NAVIGATE: 30.0,
        ActionType.CLICK: 10.0,
        ActionType.TYPE: 10.0,
        ActionType.WAIT: 60.0,
        ActionType.EXTRACT: 10.0,
        ActionType.ASSERT: 10.0,
        ActionType.SCREENSHOT: 10.0,
        ActionType.SELECT: 10.0,
        ActionType.SCROLL: 5.0,
        ActionType.HOVER: 5.0,
    }

    def infer(self, action: ExtractedAction,
              context: Optional[Dict[str, Any]] = None) -> ExtractedAction:
        """Infer missing parameters for an action."""
        ctx = context or {}

        # Set default timeout
        if "timeout" not in action.parameters:
            action.parameters["timeout"] = self.DEFAULT_TIMEOUTS.get(
                action.action_type, 10.0
            )

        # Infer delay
        if "delay_before" not in action.parameters:
            action.parameters["delay_before"] = 0.1
        if "delay_after" not in action.parameters:
            action.parameters["delay_after"] = 0.1

        # Action-specific inference
        if action.action_type == ActionType.NAVIGATE:
            self._infer_navigate_params(action, ctx)
        elif action.action_type == ActionType.TYPE:
            self._infer_type_params(action, ctx)
        elif action.action_type == ActionType.CLICK:
            self._infer_click_params(action, ctx)
        elif action.action_type == ActionType.WAIT:
            self._infer_wait_params(action, ctx)
        elif action.action_type == ActionType.EXTRACT:
            self._infer_extract_params(action, ctx)
        elif action.action_type == ActionType.ASSERT:
            self._infer_assert_params(action, ctx)

        return action

    def _infer_navigate_params(self, action: ExtractedAction,
                                ctx: Dict[str, Any]) -> None:
        """Infer navigation parameters."""
        target = action.target or action.value
        if target and not target.startswith(("http://", "https://", "/")):
            # Might be a search query or incomplete URL
            if "." in target and " " not in target:
                action.parameters["url"] = f"https://{target}"
            else:
                action.parameters["url"] = f"https://www.google.com/search?q={target}"
        elif target:
            action.parameters["url"] = target

        action.parameters["wait_until"] = "load"

    def _infer_type_params(self, action: ExtractedAction,
                            ctx: Dict[str, Any]) -> None:
        """Infer type parameters."""
        if not action.value and not action.parameters.get("text"):
            action.parameters["text"] = action.target
        elif action.value:
            action.parameters["text"] = action.value

        action.parameters["clear_first"] = True
        action.parameters["press_enter"] = False

        # Check if should press Enter after typing
        if action.value and any(kw in action.raw_text.lower() for kw in
                                  ("submit", "enter", "confirm", "search")):
            action.parameters["press_enter"] = True

    def _infer_click_params(self, action: ExtractedAction,
                             ctx: Dict[str, Any]) -> None:
        """Infer click parameters."""
        action.parameters["button"] = "left"
        action.parameters["click_count"] = 1

    def _infer_wait_params(self, action: ExtractedAction,
                            ctx: Dict[str, Any]) -> None:
        """Infer wait parameters."""
        if "duration" not in action.parameters:
            action.parameters["duration"] = 1.0

    def _infer_extract_params(self, action: ExtractedAction,
                               ctx: Dict[str, Any]) -> None:
        """Infer extract parameters."""
        if "attribute" not in action.parameters:
            action.parameters["attribute"] = "text"

    def _infer_assert_params(self, action: ExtractedAction,
                              ctx: Dict[str, Any]) -> None:
        """Infer assert parameters."""
        if "expected" not in action.parameters:
            action.parameters["expected"] = True


class TemplateLibrary:
    """
    Library of common workflow templates.

    Provides pre-built workflow patterns for frequent operations.
    """

    def __init__(self) -> None:
        self._templates: Dict[str, Dict[str, Any]] = {}
        self._init_templates()

    def _init_templates(self) -> None:
        """Initialize built-in templates."""
        self._templates["login"] = {
            "name": "Login Workflow",
            "description": "Log into a website",
            "steps": [
                {"action": "navigate", "parameters": {"url": "{login_url}"}},
                {"action": "type", "parameters": {"text": "{username}", "selector": "#username"}},
                {"action": "type", "parameters": {"text": "{password}", "selector": "#password"}},
                {"action": "click", "parameters": {"selector": "#login-button"}},
                {"action": "wait", "parameters": {"duration": 2.0}},
                {"action": "assert", "parameters": {"selector": ".dashboard"}},
            ],
            "variables": ["login_url", "username", "password"],
        }
        self._templates["search"] = {
            "name": "Search Workflow",
            "description": "Search for something on a website",
            "steps": [
                {"action": "navigate", "parameters": {"url": "{search_url}"}},
                {"action": "click", "parameters": {"selector": "{search_box_selector}"}},
                {"action": "type", "parameters": {"text": "{query}", "press_enter": True}},
                {"action": "wait", "parameters": {"duration": 2.0}},
                {"action": "extract", "parameters": {"selector": "{results_selector}"}},
            ],
            "variables": ["search_url", "search_box_selector", "query", "results_selector"],
        }
        self._templates["form_fill"] = {
            "name": "Form Fill Workflow",
            "description": "Fill out a form",
            "steps": [
                {"action": "navigate", "parameters": {"url": "{form_url}"}},
                {"action": "type", "parameters": {"text": "{field1_value}", "selector": "{field1_selector}"}},
                {"action": "type", "parameters": {"text": "{field2_value}", "selector": "{field2_selector}"}},
                {"action": "select", "parameters": {"value": "{dropdown_value}", "selector": "{dropdown_selector}"}},
                {"action": "click", "parameters": {"selector": "{submit_selector}"}},
                {"action": "wait", "parameters": {"duration": 2.0}},
                {"action": "screenshot", "parameters": {}},
            ],
            "variables": ["form_url", "field1_selector", "field1_value",
                          "field2_selector", "field2_value",
                          "dropdown_selector", "dropdown_value", "submit_selector"],
        }
        self._templates["screenshot"] = {
            "name": "Screenshot Workflow",
            "description": "Take a screenshot of a page",
            "steps": [
                {"action": "navigate", "parameters": {"url": "{url}"}},
                {"action": "wait", "parameters": {"duration": 2.0}},
                {"action": "screenshot", "parameters": {"path": "{output_path}"}},
            ],
            "variables": ["url", "output_path"],
        }
        self._templates["data_extraction"] = {
            "name": "Data Extraction Workflow",
            "description": "Extract data from a page",
            "steps": [
                {"action": "navigate", "parameters": {"url": "{url}"}},
                {"action": "wait", "parameters": {"duration": 2.0}},
                {"action": "extract", "parameters": {"selector": "{item_selector}", "attribute": "text"}},
                {"action": "scroll", "parameters": {"direction": "down", "amount": 500}},
                {"action": "wait", "parameters": {"duration": 1.0}},
                {"action": "extract", "parameters": {"selector": "{item_selector}", "attribute": "text"}},
            ],
            "variables": ["url", "item_selector"],
        }

    def get_template(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a template by name."""
        return self._templates.get(name)

    def list_templates(self) -> List[str]:
        """List available template names."""
        return sorted(self._templates.keys())

    def search_templates(self, query: str) -> List[str]:
        """Search templates by name or description."""
        query_lower = query.lower()
        matches: List[str] = []
        for name, template in self._templates.items():
            if (query_lower in name.lower() or
                    query_lower in template.get("description", "").lower()):
                matches.append(name)
        return matches

    def add_template(self, name: str, template: Dict[str, Any]) -> None:
        """Add a custom template."""
        self._templates[name] = template

    def instantiate(self, name: str, variables: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Instantiate a template with variable values."""
        template = self.get_template(name)
        if template is None:
            return None

        import copy
        instance = copy.deepcopy(template)
        steps_json = json.dumps(instance.get("steps", []))
        for var_name, var_value in variables.items():
            steps_json = steps_json.replace(f"{{{var_name}}}", var_value)
        instance["steps"] = json.loads(steps_json)
        return instance


class WorkflowBuilder:
    """
    Builds workflow data structures from extracted actions.

    Converts a list of extracted actions into a complete workflow.
    """

    def __init__(self) -> None:
        self._step_counter = 0

    def build(self, actions: List[ExtractedAction],
              name: str = "Generated Workflow",
              description: str = "") -> Dict[str, Any]:
        """Build a workflow from a list of actions."""
        self._step_counter = 0
        steps: List[Dict[str, Any]] = []

        for action in actions:
            step = self._action_to_step(action)
            steps.append(step)

        return {
            "workflow_id": f"wf-nl-{int(time.time())}",
            "name": name,
            "version": "1.0",
            "description": description,
            "created_at": time.time(),
            "steps": steps,
            "variables": {},
        }

    def _action_to_step(self, action: ExtractedAction) -> Dict[str, Any]:
        """Convert an extracted action to a workflow step."""
        self._step_counter += 1
        return {
            "step_id": f"step-{self._step_counter}",
            "step_type": "action",
            "action": action.action_type.value,
            "name": f"{action.action_type.value} {action.target or action.value or ''}".strip(),
            "parameters": action.parameters,
            "delay_before": action.parameters.get("delay_before", 0.1),
            "delay_after": action.parameters.get("delay_after", 0.1),
            "max_retries": action.parameters.get("max_retries", 0),
            "skip_on_failure": False,
        }


class NLWorkflowGenerator:
    """
    High-level natural language to workflow generator.

    Provides the complete pipeline: parse -> extract -> infer -> build.
    """

    def __init__(self) -> None:
        self.intent_parser = IntentParser()
        self.action_extractor = ActionExtractor()
        self.parameter_inferrer = ParameterInferrer()
        self.workflow_builder = WorkflowBuilder()
        self.template_library = TemplateLibrary()

    def generate(self, description: str,
                 name: str = "Generated Workflow",
                 context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a workflow from a natural language description.

        Args:
            description: Natural language description of the workflow
            name: Name for the generated workflow
            context: Additional context for parameter inference

        Returns:
            Workflow dictionary
        """
        # Parse intents
        intents = self.intent_parser.parse_multi(description)

        if not intents:
            # Try single intent parse
            single = self.intent_parser.parse(description)
            if single.action != ActionType.UNKNOWN:
                intents = [single]
            else:
                # Try template matching
                template = self._find_matching_template(description)
                if template:
                    return template
                return self._empty_workflow(name, description)

        # Extract actions
        actions = self.action_extractor.extract_batch(intents)

        # Infer parameters
        for action in actions:
            self.parameter_inferrer.infer(action, context)

        # Build workflow
        workflow = self.workflow_builder.build(actions, name, description)
        return workflow

    def generate_from_steps(self, steps: List[str],
                            name: str = "Generated Workflow") -> Dict[str, Any]:
        """Generate a workflow from a list of step descriptions."""
        all_intents: List[ParsedIntent] = []
        for step_desc in steps:
            intents = self.intent_parser.parse_multi(step_desc)
            all_intents.extend(intents)

        if not all_intents:
            return self._empty_workflow(name)

        actions = self.action_extractor.extract_batch(all_intents)
        for action in actions:
            self.parameter_inferrer.infer(action)

        return self.workflow_builder.build(actions, name)

    def _find_matching_template(self, description: str) -> Optional[Dict[str, Any]]:
        """Find a matching template for the description."""
        desc_lower = description.lower()
        templates = self.template_library.search_templates(desc_lower)
        if templates:
            return self.template_library.get_template(templates[0])
        return None

    def _empty_workflow(self, name: str,
                        description: str = "") -> Dict[str, Any]:
        """Create an empty workflow."""
        return {
            "workflow_id": f"wf-nl-{int(time.time())}",
            "name": name,
            "version": "1.0",
            "description": description,
            "created_at": time.time(),
            "steps": [],
            "variables": {},
        }

    def get_available_templates(self) -> List[Dict[str, str]]:
        """Get available workflow templates."""
        templates = []
        for name in self.template_library.list_templates():
            template = self.template_library.get_template(name)
            if template:
                templates.append({
                    "name": name,
                    "description": template.get("description", ""),
                    "variables": template.get("variables", []),
                })
        return templates
