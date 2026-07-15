"""提示词模板注册模块。

本模块实现MCP提示词模板的注册和管理，通过MCP暴露预定义Prompts。
支持模板变量、条件渲染和模板继承。
"""

from __future__ import annotations

import json
import re
import threading
from typing import Optional, Callable, Dict, Any, List, Union
from dataclasses import dataclass, field
from enum import Enum
from string import Template
from abc import ABC, abstractmethod

from .schema import MCPPrompt, MCPPromptArgument


class PromptCategory(Enum):
    """提示词类别枚举。"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TASK = "task"
    EXAMPLE = "example"
    CUSTOM = "custom"


@dataclass
class PromptTemplate:
    """提示词模板。
    
    Attributes:
        name: 模板名称
        description: 模板描述
        template: 模板内容
        arguments: 参数定义
        category: 模板类别
        tags: 标签
        metadata: 元数据
    """
    name: str
    description: str = ""
    template: str = ""
    arguments: List[MCPPromptArgument] = field(default_factory=list)
    category: PromptCategory = PromptCategory.CUSTOM
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_mcp_prompt(self) -> MCPPrompt:
        """转换为MCP提示词定义。"""
        return MCPPrompt(
            name=self.name,
            description=self.description,
            arguments=self.arguments,
            template=self.template
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "name": self.name,
            "description": self.description,
            "template": self.template,
            "arguments": [arg.to_dict() for arg in self.arguments],
            "category": self.category.value,
            "tags": self.tags,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PromptTemplate:
        """从字典创建。"""
        arguments = [
            MCPPromptArgument.from_dict(arg)
            for arg in data.get("arguments", [])
        ]
        
        category_str = data.get("category", "custom")
        try:
            category = PromptCategory(category_str)
        except ValueError:
            category = PromptCategory.CUSTOM
        
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            template=data.get("template", ""),
            arguments=arguments,
            category=category,
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )


class TemplateEngine(ABC):
    """模板引擎抽象基类。"""
    
    @abstractmethod
    def render(
        self,
        template: str,
        variables: Dict[str, Any]
    ) -> str:
        """渲染模板。
        
        Args:
            template: 模板字符串
            variables: 变量字典
            
        Returns:
            渲染结果
        """
        pass


class SimpleTemplateEngine(TemplateEngine):
    """简单模板引擎。
    
    使用{variable}语法进行变量替换。
    """
    
    def render(
        self,
        template: str,
        variables: Dict[str, Any]
    ) -> str:
        """渲染模板。"""
        result = template
        
        for key, value in variables.items():
            # 简单变量替换
            result = result.replace(f"{{{key}}}", str(value))
        
        return result


class PythonTemplateEngine(TemplateEngine):
    """Python字符串模板引擎。
    
    使用$variable或${variable}语法。
    """
    
    def render(
        self,
        template: str,
        variables: Dict[str, Any]
    ) -> str:
        """渲染模板。"""
        try:
            t = Template(template)
            return t.safe_substitute(variables)
        except Exception:
            return template


class ConditionalTemplateEngine(TemplateEngine):
    """条件模板引擎。
    
    支持条件语句和循环。
    """
    
    def __init__(self) -> None:
        """初始化条件模板引擎。"""
        self._simple_engine = SimpleTemplateEngine()
    
    def render(
        self,
        template: str,
        variables: Dict[str, Any]
    ) -> str:
        """渲染模板。"""
        result = template
        
        # 处理条件语句 {% if condition %}...{% endif %}
        if_pattern = r'\{%\s*if\s+(\w+)\s*%\}(.*?)\{%\s*endif\s*%\}'
        
        def replace_if(match: re.Match) -> str:
            condition_var = match.group(1)
            content = match.group(2)
            
            condition_value = variables.get(condition_var, False)
            
            if condition_value:
                return content
            return ""
        
        result = re.sub(if_pattern, replace_if, result, flags=re.DOTALL)
        
        # 处理条件else语句 {% if condition %}...{% else %}...{% endif %}
        if_else_pattern = r'\{%\s*if\s+(\w+)\s*%\}(.*?)\{%\s*else\s*%\}(.*?)\{%\s*endif\s*%\}'
        
        def replace_if_else(match: re.Match) -> str:
            condition_var = match.group(1)
            true_content = match.group(2)
            false_content = match.group(3)
            
            condition_value = variables.get(condition_var, False)
            
            if condition_value:
                return true_content
            return false_content
        
        result = re.sub(if_else_pattern, replace_if_else, result, flags=re.DOTALL)
        
        # 处理循环 {% for item in items %}...{% endfor %}
        for_pattern = r'\{%\s*for\s+(\w+)\s+in\s+(\w+)\s*%\}(.*?)\{%\s*endfor\s*%\}'
        
        def replace_for(match: re.Match) -> str:
            item_var = match.group(1)
            list_var = match.group(2)
            content = match.group(3)
            
            items = variables.get(list_var, [])
            
            if not isinstance(items, (list, tuple)):
                return ""
            
            result_parts = []
            for item in items:
                item_vars = dict(variables)
                item_vars[item_var] = item
                item_vars["index"] = len(result_parts)
                rendered = self._simple_engine.render(content, item_vars)
                result_parts.append(rendered)
            
            return "".join(result_parts)
        
        result = re.sub(for_pattern, replace_for, result, flags=re.DOTALL)
        
        # 最后进行变量替换
        result = self._simple_engine.render(result, variables)
        
        return result


class PromptRegistry:
    """提示词注册表。
    
    管理提示词模板的注册、查询和渲染。
    """
    
    def __init__(
        self,
        template_engine: Optional[TemplateEngine] = None
    ):
        """初始化提示词注册表。
        
        Args:
            template_engine: 模板引擎，默认使用条件模板引擎
        """
        self._prompts: Dict[str, PromptTemplate] = {}
        self._template_engine = template_engine or ConditionalTemplateEngine()
        self._lock = threading.Lock()
        
        # 模板继承关系
        self._inheritance: Dict[str, str] = {}
    
    def register(
        self,
        name: str,
        template: str,
        description: str = "",
        arguments: Optional[List[Dict[str, Any]]] = None,
        category: PromptCategory = PromptCategory.CUSTOM,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """注册提示词模板。
        
        Args:
            name: 模板名称
            template: 模板内容
            description: 描述
            arguments: 参数定义
            category: 类别
            tags: 标签
            metadata: 元数据
        """
        args = []
        if arguments:
            for arg in arguments:
                args.append(MCPPromptArgument(
                    name=arg.get("name", ""),
                    description=arg.get("description", ""),
                    required=arg.get("required", False)
                ))
        
        prompt = PromptTemplate(
            name=name,
            description=description,
            template=template,
            arguments=args,
            category=category,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        with self._lock:
            self._prompts[name] = prompt
    
    def register_prompt(self, prompt: PromptTemplate) -> None:
        """注册提示词模板对象。
        
        Args:
            prompt: 提示词模板
        """
        with self._lock:
            self._prompts[prompt.name] = prompt
    
    def unregister(self, name: str) -> None:
        """注销提示词模板。
        
        Args:
            name: 模板名称
        """
        with self._lock:
            self._prompts.pop(name, None)
            self._inheritance.pop(name, None)
    
    def get(self, name: str) -> Optional[PromptTemplate]:
        """获取提示词模板。
        
        Args:
            name: 模板名称
            
        Returns:
            提示词模板
        """
        with self._lock:
            return self._prompts.get(name)
    
    def exists(self, name: str) -> bool:
        """检查模板是否存在。
        
        Args:
            name: 模板名称
            
        Returns:
            是否存在
        """
        with self._lock:
            return name in self._prompts
    
    def render(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None
    ) -> str:
        """渲染提示词模板。
        
        Args:
            name: 模板名称
            arguments: 参数值
            
        Returns:
            渲染结果
            
        Raises:
            ValueError: 模板不存在或缺少必需参数
        """
        with self._lock:
            prompt = self._prompts.get(name)
        
        if not prompt:
            raise ValueError(f"Prompt not found: {name}")
        
        arguments = arguments or {}
        
        # 检查必需参数
        missing = []
        for arg in prompt.arguments:
            if arg.required and arg.name not in arguments:
                missing.append(arg.name)
        
        if missing:
            raise ValueError(f"Missing required arguments: {missing}")
        
        # 处理模板继承
        template = self._resolve_inheritance(name, prompt.template)
        
        # 渲染模板
        return self._template_engine.render(template, arguments)
    
    def _resolve_inheritance(self, name: str, template: str) -> str:
        """解析模板继承。
        
        Args:
            name: 模板名称
            template: 模板内容
            
        Returns:
            解析后的模板
        """
        # 检查是否有继承声明 {% extends "parent" %}
        extends_pattern = r'\{%\s*extends\s+["\'](\w+)["\']\s*%\}'
        match = re.search(extends_pattern, template)
        
        if match:
            parent_name = match.group(1)
            
            with self._lock:
                parent = self._prompts.get(parent_name)
            
            if parent:
                # 移除继承声明
                child_template = re.sub(extends_pattern, "", template)
                
                # 获取父模板
                parent_template = self._resolve_inheritance(
                    parent_name,
                    parent.template
                )
                
                # 处理块替换 {% block name %}...{% endblock %}
                block_pattern = r'\{%\s*block\s+(\w+)\s*%\}(.*?)\{%\s*endblock\s*%\}'
                
                # 提取子模板的块
                child_blocks = {}
                for block_match in re.finditer(block_pattern, child_template, re.DOTALL):
                    block_name = block_match.group(1)
                    block_content = block_match.group(2)
                    child_blocks[block_name] = block_content
                
                # 替换父模板中的块
                def replace_block(match: re.Match) -> str:
                    block_name = match.group(1)
                    default_content = match.group(2)
                    
                    if block_name in child_blocks:
                        return child_blocks[block_name]
                    return default_content
                
                result = re.sub(block_pattern, replace_block, parent_template, flags=re.DOTALL)
                
                return result
        
        return template
    
    def set_inheritance(self, child: str, parent: str) -> None:
        """设置模板继承关系。
        
        Args:
            child: 子模板名称
            parent: 父模板名称
        """
        with self._lock:
            self._inheritance[child] = parent
    
    def list_all(self) -> List[PromptTemplate]:
        """列出所有模板。
        
        Returns:
            模板列表
        """
        with self._lock:
            return list(self._prompts.values())
    
    def list_by_category(
        self,
        category: PromptCategory
    ) -> List[PromptTemplate]:
        """按类别列出模板。
        
        Args:
            category: 类别
            
        Returns:
            模板列表
        """
        with self._lock:
            return [
                p for p in self._prompts.values()
                if p.category == category
            ]
    
    def list_by_tag(self, tag: str) -> List[PromptTemplate]:
        """按标签列出模板。
        
        Args:
            tag: 标签
            
        Returns:
            模板列表
        """
        with self._lock:
            return [
                p for p in self._prompts.values()
                if tag in p.tags
            ]
    
    def search(
        self,
        query: str,
        search_content: bool = True
    ) -> List[PromptTemplate]:
        """搜索模板。
        
        Args:
            query: 搜索关键词
            search_content: 是否搜索模板内容
            
        Returns:
            匹配的模板列表
        """
        query_lower = query.lower()
        results = []
        
        with self._lock:
            for prompt in self._prompts.values():
                # 搜索名称
                if query_lower in prompt.name.lower():
                    results.append(prompt)
                    continue
                
                # 搜索描述
                if query_lower in prompt.description.lower():
                    results.append(prompt)
                    continue
                
                # 搜索标签
                if any(query_lower in tag.lower() for tag in prompt.tags):
                    results.append(prompt)
                    continue
                
                # 搜索内容
                if search_content and query_lower in prompt.template.lower():
                    results.append(prompt)
                    continue
        
        return results
    
    def export_mcp(self) -> List[MCPPrompt]:
        """导出为MCP格式。
        
        Returns:
            MCP提示词列表
        """
        with self._lock:
            return [p.to_mcp_prompt() for p in self._prompts.values()]
    
    def import_mcp(self, prompts: List[MCPPrompt]) -> None:
        """导入MCP格式。
        
        Args:
            prompts: MCP提示词列表
        """
        with self._lock:
            for prompt in prompts:
                self._prompts[prompt.name] = PromptTemplate(
                    name=prompt.name,
                    description=prompt.description,
                    template=prompt.template,
                    arguments=prompt.arguments
                )
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。
        
        Returns:
            序列化结果
        """
        with self._lock:
            return {
                name: prompt.to_dict()
                for name, prompt in self._prompts.items()
            }
    
    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        template_engine: Optional[TemplateEngine] = None
    ) -> PromptRegistry:
        """从字典反序列化。
        
        Args:
            data: 序列化数据
            template_engine: 模板引擎
            
        Returns:
            提示词注册表
        """
        registry = cls(template_engine=template_engine)
        
        for name, prompt_data in data.items():
            prompt = PromptTemplate.from_dict(prompt_data)
            registry._prompts[name] = prompt
        
        return registry
    
    def __len__(self) -> int:
        """获取模板数量。"""
        with self._lock:
            return len(self._prompts)
    
    def __contains__(self, name: str) -> bool:
        """检查模板是否存在。"""
        return self.exists(name)


class PromptBuilder:
    """提示词构建器。
    
    提供流式API构建提示词模板。
    """
    
    def __init__(self, name: str) -> None:
        """初始化构建器。
        
        Args:
            name: 模板名称
        """
        self._name = name
        self._description = ""
        self._template = ""
        self._arguments: List[MCPPromptArgument] = []
        self._category = PromptCategory.CUSTOM
        self._tags: List[str] = []
        self._metadata: Dict[str, Any] = {}
    
    def description(self, desc: str) -> PromptBuilder:
        """设置描述。"""
        self._description = desc
        return self
    
    def template(self, content: str) -> PromptBuilder:
        """设置模板内容。"""
        self._template = content
        return self
    
    def argument(
        self,
        name: str,
        description: str = "",
        required: bool = False
    ) -> PromptBuilder:
        """添加参数。"""
        self._arguments.append(MCPPromptArgument(
            name=name,
            description=description,
            required=required
        ))
        return self
    
    def category(self, cat: PromptCategory) -> PromptBuilder:
        """设置类别。"""
        self._category = cat
        return self
    
    def tag(self, tag: str) -> PromptBuilder:
        """添加标签。"""
        self._tags.append(tag)
        return self
    
    def metadata(self, key: str, value: Any) -> PromptBuilder:
        """添加元数据。"""
        self._metadata[key] = value
        return self
    
    def build(self) -> PromptTemplate:
        """构建模板。"""
        return PromptTemplate(
            name=self._name,
            description=self._description,
            template=self._template,
            arguments=self._arguments,
            category=self._category,
            tags=self._tags,
            metadata=self._metadata
        )


def prompt(
    name: str,
    description: str = "",
    arguments: Optional[List[Dict[str, Any]]] = None
) -> Callable[[Callable], Callable]:
    """提示词装饰器。
    
    Args:
        name: 模板名称
        description: 描述
        arguments: 参数定义
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        func._prompt_name = name
        func._prompt_description = description or func.__doc__ or ""
        func._prompt_arguments = arguments or []
        return func
    
    return decorator


# 预定义的常用模板
BUILTIN_PROMPTS = {
    "system_default": PromptTemplate(
        name="system_default",
        description="默认系统提示词",
        template="You are a helpful AI assistant.",
        category=PromptCategory.SYSTEM
    ),
    
    "task_template": PromptTemplate(
        name="task_template",
        description="任务执行模板",
        template="Please complete the following task:\n\n{task}\n\nContext:\n{context}",
        arguments=[
            MCPPromptArgument(name="task", description="任务描述", required=True),
            MCPPromptArgument(name="context", description="上下文信息", required=False)
        ],
        category=PromptCategory.TASK
    ),
    
    "code_review": PromptTemplate(
        name="code_review",
        description="代码审查模板",
        template="""Please review the following code:

```{language}
{code}
```

Focus on:
- Code quality and readability
- Potential bugs or errors
- Performance considerations
- Security vulnerabilities

{additional_instructions}""",
        arguments=[
            MCPPromptArgument(name="code", description="待审查的代码", required=True),
            MCPPromptArgument(name="language", description="编程语言", required=False),
            MCPPromptArgument(name="additional_instructions", description="额外指令", required=False)
        ],
        category=PromptCategory.TASK,
        tags=["code", "review"]
    ),
    
    "few_shot": PromptTemplate(
        name="few_shot",
        description="Few-shot学习模板",
        template="""Here are some examples:

{% for example in examples %}
Example {{ index + 1 }}:
Input: {{ example.input }}
Output: {{ example.output }}

{% endfor %}
Now, please process the following:
Input: {input}
Output:""",
        arguments=[
            MCPPromptArgument(name="examples", description="示例列表", required=True),
            MCPPromptArgument(name="input", description="当前输入", required=True)
        ],
        category=PromptCategory.EXAMPLE,
        tags=["few-shot", "learning"]
    )
}


__all__ = [
    "PromptCategory",
    "PromptTemplate",
    "TemplateEngine",
    "SimpleTemplateEngine",
    "PythonTemplateEngine",
    "ConditionalTemplateEngine",
    "PromptRegistry",
    "PromptBuilder",
    "prompt",
    "BUILTIN_PROMPTS",
]
