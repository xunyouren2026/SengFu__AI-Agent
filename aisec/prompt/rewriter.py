"""
安全改写器 - Prompt安全改写
"""
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class RewriteAction(Enum):
    """改写动作"""
    ESCAPE = "escape"           # 转义
    REMOVE = "remove"           # 移除
    REPLACE = "replace"         # 替换
    PREFIX = "prefix"           # 添加前缀
    SUFFIX = "suffix"           # 添加后缀
    WRAP = "wrap"               # 包装
    NORMALIZE = "normalize"     # 标准化


@dataclass
class RewriteRule:
    """改写规则"""
    pattern: str
    action: RewriteAction
    replacement: str = ""
    description: str = ""
    priority: int = 100
    regex: re.Pattern = field(init=False)
    
    def __post_init__(self):
        self.regex = re.compile(self.pattern, re.IGNORECASE | re.DOTALL)


@dataclass
class RewriteResult:
    """改写结果"""
    original: str
    rewritten: str
    changes: List[Dict[str, Any]]
    is_modified: bool


class PromptRewriter:
    """Prompt安全改写器"""
    
    def __init__(self):
        self._rules: List[RewriteRule] = self._load_default_rules()
        self._custom_rules: List[RewriteRule] = []
        self._escape_chars = ['\\', '"', "'", '`', '$', '{', '}', '[', ']', '<', '>']
        self._system_prefix = "[SYSTEM] "
        self._system_suffix = " [/SYSTEM]"
    
    def _load_default_rules(self) -> List[RewriteRule]:
        """加载默认改写规则"""
        rules = [
            # 转义特殊字符
            RewriteRule(
                pattern=r"\\",
                action=RewriteAction.ESCAPE,
                description="转义反斜杠",
                priority=10
            ),
            
            # 移除系统指令标记
            RewriteRule(
                pattern=r"<\|?system\|?>",
                action=RewriteAction.REMOVE,
                description="移除系统标记",
                priority=5
            ),
            RewriteRule(
                pattern=r"\[system\]",
                action=RewriteAction.REMOVE,
                description="移除系统标记",
                priority=5
            ),
            RewriteRule(
                pattern=r"###\s*system\s*:",
                action=RewriteAction.REMOVE,
                description="移除系统标记",
                priority=5
            ),
            
            # 替换危险指令
            RewriteRule(
                pattern=r"ignore\s+(all\s+)?previous\s+instructions?",
                action=RewriteAction.REPLACE,
                replacement="[该指令已被安全改写]",
                description="替换忽略指令",
                priority=1
            ),
            RewriteRule(
                pattern=r"disregard\s+(all\s+)?previous\s+instructions?",
                action=RewriteAction.REPLACE,
                replacement="[该指令已被安全改写]",
                description="替换忽略指令",
                priority=1
            ),
            
            # 标准化角色扮演
            RewriteRule(
                pattern=r"you\s+are\s+now\s+(a|an)\s+",
                action=RewriteAction.NORMALIZE,
                replacement="请作为",
                description="标准化角色扮演",
                priority=20
            ),
            
            # 移除命令替换
            RewriteRule(
                pattern=r"\$\([^)]+\)",
                action=RewriteAction.REMOVE,
                description="移除命令替换",
                priority=5
            ),
            RewriteRule(
                pattern=r"`[^`]+`",
                action=RewriteAction.REPLACE,
                replacement="[代码块已移除]",
                description="移除反引号命令",
                priority=5
            ),
            
            # 移除模板注入
            RewriteRule(
                pattern=r"\{\{[^}]+\}\}",
                action=RewriteAction.REMOVE,
                description="移除模板注入",
                priority=5
            ),
            RewriteRule(
                pattern=r"\$\{[^}]+\}",
                action=RewriteAction.REMOVE,
                description="移除变量注入",
                priority=5
            ),
        ]
        return rules
    
    def add_rule(self, rule: RewriteRule) -> None:
        """添加改写规则"""
        self._custom_rules.append(rule)
    
    def add_custom_rule(
        self,
        pattern: str,
        action: RewriteAction,
        replacement: str = "",
        description: str = "",
        priority: int = 100
    ) -> None:
        """添加自定义规则"""
        rule = RewriteRule(
            pattern=pattern,
            action=action,
            replacement=replacement,
            description=description,
            priority=priority
        )
        self._custom_rules.append(rule)
    
    def rewrite(self, prompt: str) -> RewriteResult:
        """改写Prompt"""
        rewritten = prompt
        changes = []
        
        # 合并并按优先级排序规则
        all_rules = sorted(
            self._rules + self._custom_rules,
            key=lambda r: r.priority
        )
        
        for rule in all_rules:
            matches = list(rule.regex.finditer(rewritten))
            
            if not matches:
                continue
            
            for match in reversed(matches):  # 从后往前替换
                original_text = match.group()
                new_text = self._apply_action(original_text, rule, match)
                
                if original_text != new_text:
                    changes.append({
                        "rule": rule.description,
                        "action": rule.action.value,
                        "original": original_text[:50] + "..." if len(original_text) > 50 else original_text,
                        "position": match.start()
                    })
                    
                    rewritten = rewritten[:match.start()] + new_text + rewritten[match.end():]
        
        return RewriteResult(
            original=prompt,
            rewritten=rewritten,
            changes=changes,
            is_modified=len(changes) > 0
        )
    
    def _apply_action(self, text: str, rule: RewriteRule, match: re.Match) -> str:
        """应用改写动作"""
        if rule.action == RewriteAction.ESCAPE:
            return text.replace('\\', '\\\\')
        
        elif rule.action == RewriteAction.REMOVE:
            return ""
        
        elif rule.action == RewriteAction.REPLACE:
            return rule.replacement
        
        elif rule.action == RewriteAction.PREFIX:
            return rule.replacement + text
        
        elif rule.action == RewriteAction.SUFFIX:
            return text + rule.replacement
        
        elif rule.action == RewriteAction.WRAP:
            return rule.replacement + text + rule.replacement
        
        elif rule.action == RewriteAction.NORMALIZE:
            return rule.replacement + match.group(2) if len(match.groups()) > 1 else rule.replacement
        
        return text
    
    def escape_special_chars(self, text: str) -> str:
        """转义特殊字符"""
        result = text
        for char in self._escape_chars:
            result = result.replace(char, '\\' + char)
        return result
    
    def wrap_user_input(self, text: str) -> str:
        """包装用户输入"""
        return f"[USER_INPUT_START]{text}[USER_INPUT_END]"
    
    def add_safe_boundary(self, text: str) -> str:
        """添加安全边界"""
        return f"{self._system_prefix}用户输入开始{self._system_suffix}\n{text}\n{self._system_prefix}用户输入结束{self._system_suffix}"
    
    def normalize_whitespace(self, text: str) -> str:
        """标准化空白字符"""
        # 移除多余的空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 移除行尾空白
        text = re.sub(r'[ \t]+\n', '\n', text)
        # 标准化制表符
        text = text.replace('\t', '    ')
        return text.strip()
    
    def remove_control_chars(self, text: str) -> str:
        """移除控制字符"""
        # 移除不可见控制字符（保留换行和制表符）
        return ''.join(char for char in text if char.isprintable() or char in '\n\t\r')
    
    def sanitize_for_json(self, text: str) -> str:
        """为JSON安全处理"""
        # 转义JSON特殊字符
        text = text.replace('\\', '\\\\')
        text = text.replace('"', '\\"')
        text = text.replace('\n', '\\n')
        text = text.replace('\r', '\\r')
        text = text.replace('\t', '\\t')
        return text
    
    def full_sanitize(self, prompt: str) -> RewriteResult:
        """完整安全处理"""
        # 1. 移除控制字符
        text = self.remove_control_chars(prompt)
        
        # 2. 标准化空白
        text = self.normalize_whitespace(text)
        
        # 3. 应用改写规则
        result = self.rewrite(text)
        
        # 4. 添加安全边界
        result.rewritten = self.add_safe_boundary(result.rewritten)
        
        result.changes.insert(0, {"action": "full_sanitize", "description": "完整安全处理"})
        
        return result
    
    def rewrite_batch(self, prompts: List[str]) -> List[RewriteResult]:
        """批量改写"""
        return [self.rewrite(prompt) for prompt in prompts]
    
    def get_rules(self) -> List[Dict[str, Any]]:
        """获取所有规则"""
        all_rules = sorted(
            self._rules + self._custom_rules,
            key=lambda r: r.priority
        )
        return [
            {
                "pattern": rule.pattern,
                "action": rule.action.value,
                "replacement": rule.replacement,
                "description": rule.description,
                "priority": rule.priority
            }
            for rule in all_rules
        ]
    
    def remove_rule(self, pattern: str) -> bool:
        """移除规则"""
        for i, rule in enumerate(self._custom_rules):
            if rule.pattern == pattern:
                del self._custom_rules[i]
                return True
        return False


class ContextRewriter:
    """上下文改写器"""
    
    def __init__(self):
        self._context_markers = {
            "start": "<!-- CONTEXT_START -->",
            "end": "<!-- CONTEXT_END -->",
            "separator": "<!-- CONTEXT_SEP -->"
        }
    
    def isolate_user_input(self, user_input: str, context: str = "") -> str:
        """隔离用户输入"""
        if context:
            return f"{context}\n{self._context_markers['separator']}\n{self._context_markers['start']}\n{user_input}\n{self._context_markers['end']}"
        return f"{self._context_markers['start']}\n{user_input}\n{self._context_markers['end']}"
    
    def extract_user_input(self, text: str) -> str:
        """提取用户输入"""
        pattern = re.compile(
            rf"{re.escape(self._context_markers['start'])}\n?(.*?)\n?{re.escape(self._context_markers['end'])}",
            re.DOTALL
        )
        match = pattern.search(text)
        return match.group(1).strip() if match else text
    
    def extract_context(self, text: str) -> str:
        """提取上下文"""
        sep = self._context_markers['separator']
        if sep in text:
            return text.split(sep)[0].strip()
        return ""
