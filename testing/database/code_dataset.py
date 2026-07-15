"""
CodeDataset - 测试代码数据集生成器

模块路径: testing/database/code_dataset.py

提供测试用代码数据集的生成、管理和查询功能。
"""

import os
import sys
import json
import time
import random
import hashlib
import textwrap
import ast
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

import pytest
import numpy as np


class ProgrammingLanguage(Enum):
    """编程语言"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"
    SQL = "sql"
    HTML = "html"
    CSS = "css"
    SHELL = "shell"


class CodeCategory(Enum):
    """代码类别"""
    ALGORITHM = "algorithm"
    DATA_STRUCTURE = "data_structure"
    API = "api"
    DATABASE = "database"
    UTILITY = "utility"
    CLASS_DEFINITION = "class_definition"
    FUNCTION = "function"
    TEST = "test"
    CONFIGURATION = "configuration"
    ERROR_HANDLING = "error_handling"


class CodeComplexity(Enum):
    """代码复杂度"""
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ADVANCED = "advanced"


@dataclass
class CodeSnippet:
    """代码片段"""
    snippet_id: str
    language: str
    category: str
    code: str
    description: str = ""
    complexity: str = "simple"
    line_count: int = 0
    char_count: int = 0
    checksum_md5: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snippet_id": self.snippet_id,
            "language": self.language,
            "category": self.category,
            "code": self.code,
            "description": self.description,
            "complexity": self.complexity,
            "line_count": self.line_count,
            "char_count": self.char_count,
            "checksum_md5": self.checksum_md5,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


@dataclass
class CodeTestCase:
    """代码测试用例"""
    test_id: str
    snippet_id: str
    input_data: Any
    expected_output: Any
    description: str = ""
    is_edge_case: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "snippet_id": self.snippet_id,
            "input_data": self.input_data,
            "expected_output": self.expected_output,
            "description": self.description,
            "is_edge_case": self.is_edge_case,
        }


class CodeDataset:
    """测试代码数据集生成器

    提供测试用代码数据集的生成和管理功能:
        - 生成各种语言和类别的代码片段
        - 支持多种复杂度级别
        - 自动生成配套测试用例
        - 代码质量指标计算
        - 代码搜索和过滤
        - 数据集统计和导出
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialized = False
        self._snippets: Dict[str, CodeSnippet] = {}
        self._test_cases: Dict[str, List[CodeTestCase]] = {}
        self._templates: Dict[str, List[str]] = {}
        self._seed: int = self.config.get("seed", 42)
        self._init_templates()

    def initialize(self) -> None:
        """初始化数据集生成器"""
        random.seed(self._seed)
        self._initialized = True

    def _init_templates(self) -> None:
        """初始化代码模板"""
        self._templates["python_algorithm"] = [
            textwrap.dedent("""\
            def binary_search(arr, target):
                left, right = 0, len(arr) - 1
                while left <= right:
                    mid = (left + right) // 2
                    if arr[mid] == target:
                        return mid
                    elif arr[mid] < target:
                        left = mid + 1
                    else:
                        right = mid - 1
                return -1
            """),
            textwrap.dedent("""\
            def quicksort(arr):
                if len(arr) <= 1:
                    return arr
                pivot = arr[len(arr) // 2]
                left = [x for x in arr if x < pivot]
                middle = [x for x in arr if x == pivot]
                right = [x for x in arr if x > pivot]
                return quicksort(left) + middle + quicksort(right)
            """),
            textwrap.dedent("""\
            def fibonacci(n):
                if n <= 1:
                    return n
                a, b = 0, 1
                for _ in range(2, n + 1):
                    a, b = b, a + b
                return b
            """),
            textwrap.dedent("""\
            def factorial(n):
                if n < 0:
                    raise ValueError("Factorial is not defined for negative numbers")
                result = 1
                for i in range(2, n + 1):
                    result *= i
                return result
            """),
        ]
        self._templates["python_data_structure"] = [
            textwrap.dedent("""\
            class Stack:
                def __init__(self):
                    self._items = []
                def push(self, item):
                    self._items.append(item)
                def pop(self):
                    if not self._items:
                        raise IndexError("pop from empty stack")
                    return self._items.pop()
                def peek(self):
                    if not self._items:
                        raise IndexError("peek from empty stack")
                    return self._items[-1]
                def is_empty(self):
                    return len(self._items) == 0
                def size(self):
                    return len(self._items)
            """),
            textwrap.dedent("""\
            class Queue:
                def __init__(self):
                    self._items = []
                def enqueue(self, item):
                    self._items.append(item)
                def dequeue(self):
                    if not self._items:
                        raise IndexError("dequeue from empty queue")
                    return self._items.pop(0)
                def front(self):
                    if not self._items:
                        raise IndexError("front from empty queue")
                    return self._items[0]
                def is_empty(self):
                    return len(self._items) == 0
            """),
            textwrap.dedent("""\
            class LinkedList:
                class Node:
                    def __init__(self, data):
                        self.data = data
                        self.next = None
                def __init__(self):
                    self.head = None
                def append(self, data):
                    new_node = self.Node(data)
                    if not self.head:
                        self.head = new_node
                        return
                    current = self.head
                    while current.next:
                        current = current.next
                    current.next = new_node
                def to_list(self):
                    result = []
                    current = self.head
                    while current:
                        result.append(current.data)
                        current = current.next
                    return result
            """),
        ]
        self._templates["python_function"] = [
            textwrap.dedent("""\
            def flatten(nested_list):
                result = []
                for item in nested_list:
                    if isinstance(item, list):
                        result.extend(flatten(item))
                    else:
                        result.append(item)
                return result
            """),
            textwrap.dedent("""\
            def is_palindrome(s):
                s = str(s).lower().replace(" ", "")
                return s == s[::-1]
            """),
            textwrap.dedent("""\
            def merge_dicts(*dicts):
                result = {}
                for d in dicts:
                    result.update(d)
                return result
            """),
            textwrap.dedent("""\
            def chunk_list(lst, size):
                return [lst[i:i + size] for i in range(0, len(lst), size)]
            """),
        ]
        self._templates["python_error_handling"] = [
            textwrap.dedent("""\
            def safe_divide(a, b):
                try:
                    result = a / b
                except ZeroDivisionError:
                    raise ValueError("Division by zero is not allowed")
                except TypeError:
                    raise TypeError("Both arguments must be numbers")
                return result
            """),
            textwrap.dedent("""\
            def retry_operation(func, max_retries=3, delay=1.0):
                import time
                last_error = None
                for attempt in range(max_retries):
                    try:
                        return func()
                    except Exception as e:
                        last_error = e
                        if attempt < max_retries - 1:
                            time.sleep(delay)
                raise last_error
            """),
        ]
        self._templates["javascript_function"] = [
            textwrap.dedent("""\
            function debounce(fn, delay) {
                let timer = null;
                return function(...args) {
                    clearTimeout(timer);
                    timer = setTimeout(() => fn.apply(this, args), delay);
                };
            }
            """),
            textwrap.dedent("""\
            function deepClone(obj) {
                if (obj === null || typeof obj !== 'object') return obj;
                if (Array.isArray(obj)) return obj.map(item => deepClone(item));
                const cloned = {};
                for (const key in obj) {
                    if (obj.hasOwnProperty(key)) {
                        cloned[key] = deepClone(obj[key]);
                    }
                }
                return cloned;
            }
            """),
        ]
        self._templates["sql_query"] = [
            "SELECT * FROM users WHERE active = 1 ORDER BY created_at DESC LIMIT 10;",
            "SELECT u.name, COUNT(o.id) as order_count FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id HAVING order_count > 5;",
            "INSERT INTO logs (level, message, timestamp) VALUES ('INFO', 'Operation completed', CURRENT_TIMESTAMP);",
            "UPDATE products SET price = price * 1.1 WHERE category = 'electronics' AND price < 100;",
        ]

    def generate_snippet(self, language: str = "python",
                           category: str = "algorithm",
                           complexity: str = "simple") -> CodeSnippet:
        """生成单个代码片段

        Args:
            language: 编程语言
            category: 代码类别
            complexity: 复杂度

        Returns:
            CodeSnippet代码片段
        """
        if not self._initialized:
            self.initialize()
        template_key = f"{language}_{category}"
        templates = self._templates.get(template_key, self._templates.get("python_algorithm", []))
        if not templates:
            code = f"# Generated {language} {category} snippet\npass\n"
        else:
            code = random.choice(templates)
            code = self._apply_complexity_variation(code, complexity)

        snippet_id = f"code_{hashlib.md5(f'{code}_{time.time()}'.encode()).hexdigest()[:12]}"
        lines = code.strip().split("\n")
        checksum = hashlib.md5(code.encode()).hexdigest()

        snippet = CodeSnippet(
            snippet_id=snippet_id,
            language=language,
            category=category,
            code=code,
            description=f"Generated {language} {category} code snippet",
            complexity=complexity,
            line_count=len(lines),
            char_count=len(code),
            checksum_md5=checksum,
            tags=[language, category, complexity],
            created_at=time.time(),
        )
        self._snippets[snippet_id] = snippet
        return snippet

    def _apply_complexity_variation(self, code: str, complexity: str) -> str:
        """根据复杂度级别修改代码"""
        if complexity in ("trivial", "simple"):
            return code
        elif complexity == "moderate":
            docstring = f'"""\nGenerated with moderate complexity.\nIncludes additional error handling.\n"""\n'
            return docstring + code
        elif complexity == "complex":
            header = textwrap.dedent("""\
            import logging
            from typing import Any, Optional

            logger = logging.getLogger(__name__)

            """)
            return header + code
        else:
            header = textwrap.dedent("""\
            __all__ = []
            __version__ = "1.0.0"
            __author__ = "test_generator"

            import logging
            from typing import Any, Optional, List, Dict
            from dataclasses import dataclass

            logger = logging.getLogger(__name__)

            """)
            return header + code

    def generate_test_cases(self, snippet_id: str,
                              num_cases: int = 5) -> List[CodeTestCase]:
        """为代码片段生成测试用例

        Args:
            snippet_id: 代码片段ID
            num_cases: 测试用例数量

        Returns:
            CodeTestCase列表
        """
        snippet = self._snippets.get(snippet_id)
        if not snippet:
            return []
        test_cases = []
        for i in range(num_cases):
            test_id = f"test_{hashlib.md5(f'{snippet_id}_{i}'.encode()).hexdigest()[:10]}"
            is_edge = i >= num_cases - 2
            input_data, expected_output = self._generate_test_io(snippet, is_edge)
            tc = CodeTestCase(
                test_id=test_id,
                snippet_id=snippet_id,
                input_data=input_data,
                expected_output=expected_output,
                description=f"Test case {i + 1} for {snippet_id}",
                is_edge_case=is_edge,
            )
            test_cases.append(tc)
        if snippet_id not in self._test_cases:
            self._test_cases[snippet_id] = []
        self._test_cases[snippet_id].extend(test_cases)
        return test_cases

    def _generate_test_io(self, snippet: CodeSnippet,
                           is_edge: bool) -> Tuple[Any, Any]:
        """生成测试输入和期望输出"""
        if is_edge:
            edge_cases = [
                (None, None),
                ([], []),
                ("", ""),
                (0, 0),
                (-1, -1),
                (float("inf"), float("inf")),
            ]
            return random.choice(edge_cases)
        normal_cases = [
            ([1, 2, 3, 4, 5], "depends_on_function"),
            (10, "depends_on_function"),
            ("hello world", "depends_on_function"),
            ({"key": "value"}, "depends_on_function"),
        ]
        return random.choice(normal_cases)

    def generate_batch(self, count: int, language: str = "python",
                        category: Optional[str] = None) -> List[CodeSnippet]:
        """批量生成代码片段

        Args:
            count: 生成数量
            language: 编程语言
            category: 代码类别，None表示随机

        Returns:
            代码片段列表
        """
        categories = [c.value for c in CodeCategory]
        if category is None:
            category = random.choice(categories)
        snippets = []
        complexities = [c.value for c in CodeComplexity]
        for _ in range(count):
            complexity = random.choice(complexities)
            snippet = self.generate_snippet(language, category, complexity)
            snippets.append(snippet)
        return snippets

    def get_snippet(self, snippet_id: str) -> Optional[CodeSnippet]:
        """获取代码片段

        Args:
            snippet_id: 片段ID

        Returns:
            CodeSnippet或None
        """
        return self._snippets.get(snippet_id)

    def get_test_cases(self, snippet_id: str) -> List[CodeTestCase]:
        """获取代码片段的测试用例

        Args:
            snippet_id: 片段ID

        Returns:
            测试用例列表
        """
        return self._test_cases.get(snippet_id, [])

    def search_snippets(self, keyword: str, language: Optional[str] = None,
                         category: Optional[str] = None) -> List[CodeSnippet]:
        """搜索代码片段

        Args:
            keyword: 搜索关键词
            language: 语言过滤
            category: 类别过滤

        Returns:
            匹配的代码片段列表
        """
        results = []
        for snippet in self._snippets.values():
            if language and snippet.language != language:
                continue
            if category and snippet.category != category:
                continue
            if keyword and keyword not in snippet.code and keyword not in snippet.description:
                continue
            results.append(snippet)
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息

        Returns:
            统计信息字典
        """
        if not self._snippets:
            return {"total_snippets": 0}
        lang_counts = defaultdict(int)
        cat_counts = defaultdict(int)
        comp_counts = defaultdict(int)
        total_lines = 0
        total_chars = 0
        for s in self._snippets.values():
            lang_counts[s.language] += 1
            cat_counts[s.category] += 1
            comp_counts[s.complexity] += 1
            total_lines += s.line_count
            total_chars += s.char_count
        return {
            "total_snippets": len(self._snippets),
            "total_test_cases": sum(len(tcs) for tcs in self._test_cases.values()),
            "total_lines": total_lines,
            "total_chars": total_chars,
            "by_language": dict(lang_counts),
            "by_category": dict(cat_counts),
            "by_complexity": dict(comp_counts),
            "avg_lines_per_snippet": total_lines / len(self._snippets),
        }

    def export_to_json(self, filepath: str) -> int:
        """导出为JSON文件

        Args:
            filepath: 输出文件路径

        Returns:
            导出的代码片段数量
        """
        data = {
            "statistics": self.get_statistics(),
            "snippets": {k: v.to_dict() for k, v in self._snippets.items()},
            "test_cases": {k: [tc.to_dict() for tc in tcs]
                          for k, tcs in self._test_cases.items()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        return len(self._snippets)

    def export_to_jsonl(self, filepath: str) -> int:
        """导出为JSONL格式

        Args:
            filepath: 输出文件路径

        Returns:
            导出的数量
        """
        count = 0
        with open(filepath, "w", encoding="utf-8") as f:
            for snippet in self._snippets.values():
                entry = snippet.to_dict()
                entry["test_cases"] = [tc.to_dict() for tc in self._test_cases.get(snippet.snippet_id, [])]
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
                count += 1
        return count

    def add_custom_template(self, language: str, category: str,
                             code_templates: List[str]) -> None:
        """添加自定义代码模板

        Args:
            language: 编程语言
            category: 代码类别
            code_templates: 代码模板列表
        """
        key = f"{language}_{category}"
        if key not in self._templates:
            self._templates[key] = []
        self._templates[key].extend(code_templates)

    def reset(self) -> None:
        """重置数据集"""
        self._snippets.clear()
        self._test_cases.clear()
