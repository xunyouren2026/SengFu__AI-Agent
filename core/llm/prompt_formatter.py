"""
AGI Unified Framework - Prompt Formatter
Prompt格式转换器，支持多API格式的消息转换和系统提示构建
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import Message, MessageRole, ToolCall


class SystemPromptBuilder:
    """
    系统提示构建器

    提供链式API来构建结构化的系统提示，
    支持上下文、工具描述、约束条件等模块化组装。
    """

    def __init__(self, base_prompt: str = ""):
        self._sections: List[str] = []
        self._context_parts: List[str] = []
        self._tool_descriptions: List[str] = []
        self._constraints: List[str] = []
        self._examples: List[Dict[str, str]] = []
        self._output_format: str = ""
        self._persona: str = ""

        if base_prompt:
            self._sections.append(base_prompt)

    def with_persona(self, persona: str) -> "SystemPromptBuilder":
        """
        设置角色/人设

        Args:
            persona: 角色描述（如"你是一个专业的Python开发者"）
        """
        self._persona = persona
        return self

    def with_context(self, context: str) -> "SystemPromptBuilder":
        """
        添加上下文信息

        Args:
            context: 上下文描述
        """
        self._context_parts.append(context)
        return self

    def with_context_list(self, contexts: List[str]) -> "SystemPromptBuilder":
        """批量添加上下文"""
        self._context_parts.extend(contexts)
        return self

    def with_tools(self, tools: List[Dict[str, Any]]) -> "SystemPromptBuilder":
        """
        添加工具描述

        Args:
            tools: 工具定义列表，每个工具包含name和description
        """
        for tool in tools:
            func = tool.get("function", tool)
            name = func.get("name", "unknown")
            description = func.get("description", "")
            parameters = func.get("parameters", {})

            tool_desc = f"Tool: {name}\n"
            if description:
                tool_desc += f"Description: {description}\n"
            if parameters:
                tool_desc += f"Parameters: {json.dumps(parameters, ensure_ascii=False, indent=2)}\n"

            self._tool_descriptions.append(tool_desc)
        return self

    def with_tool_descriptions(self, descriptions: List[str]) -> "SystemPromptBuilder":
        """直接添加工具描述文本"""
        self._tool_descriptions.extend(descriptions)
        return self

    def with_constraints(self, *constraints: str) -> "SystemPromptBuilder":
        """
        添加约束条件

        Args:
            constraints: 约束条件列表
        """
        self._constraints.extend(constraints)
        return self

    def with_examples(self, examples: List[Dict[str, str]]) -> "SystemPromptBuilder":
        """
        添加示例

        Args:
            examples: 示例列表，每个示例包含"input"和"output"
        """
        self._examples.extend(examples)
        return self

    def with_output_format(self, format_desc: str) -> "SystemPromptBuilder":
        """
        设置输出格式要求

        Args:
            format_desc: 格式描述
        """
        self._output_format = format_desc
        return self

    def with_section(self, title: str, content: str) -> "SystemPromptBuilder":
        """
        添加自定义段落

        Args:
            title: 段落标题
            content: 段落内容
        """
        self._sections.append(f"## {title}\n{content}")
        return self

    def build(self) -> str:
        """
        构建完整的系统提示

        Returns:
            str: 组装好的系统提示文本
        """
        parts = []

        # 基础段落
        if self._sections:
            parts.extend(self._sections)

        # 角色人设
        if self._persona:
            parts.insert(0, f"# Role\n{self._persona}")

        # 上下文信息
        if self._context_parts:
            context_section = "# Context\n" + "\n".join(
                f"- {ctx}" for ctx in self._context_parts
            )
            parts.append(context_section)

        # 工具描述
        if self._tool_descriptions:
            tools_section = "# Available Tools\n" + "\n".join(self._tool_descriptions)
            parts.append(tools_section)

        # 约束条件
        if self._constraints:
            constraints_section = "# Constraints\n" + "\n".join(
                f"{i+1}. {c}" for i, c in enumerate(self._constraints)
            )
            parts.append(constraints_section)

        # 示例
        if self._examples:
            examples_section = "# Examples\n"
            for i, ex in enumerate(self._examples, 1):
                examples_section += f"\nExample {i}:\n"
                examples_section += f"Input: {ex.get('input', '')}\n"
                examples_section += f"Output: {ex.get('output', '')}\n"
            parts.append(examples_section)

        # 输出格式
        if self._output_format:
            parts.append(f"# Output Format\n{self._output_format}")

        return "\n\n".join(parts)

    def clear(self) -> "SystemPromptBuilder":
        """清空所有内容"""
        self._sections.clear()
        self._context_parts.clear()
        self._tool_descriptions.clear()
        self._constraints.clear()
        self._examples.clear()
        self._output_format = ""
        self._persona = ""
        return self

    def to_message(self) -> Message:
        """构建为Message对象"""
        return Message(role="system", content=self.build())


class PromptFormatter:
    """
    Prompt格式转换器

    支持在不同LLM API格式之间转换消息列表：
    - OpenAI格式
    - Anthropic格式（system分离）
    - Gemini格式
    - 通用聊天模板
    """

    # Anthropic支持的角色
    ANTHROPIC_ROLES = {"user", "assistant"}

    # Gemini支持的角色
    GEMINI_ROLES = {"user", "model"}

    @staticmethod
    def format_openai(messages: List[Message]) -> List[Dict[str, Any]]:
        """
        转换为OpenAI API格式

        OpenAI格式：
        [{"role": "system/user/assistant/tool", "content": "...", ...}]

        Args:
            messages: 消息列表

        Returns:
            OpenAI格式的消息列表
        """
        result = []
        for msg in messages:
            formatted = msg.to_openai_dict()
            result.append(formatted)
        return result

    @staticmethod
    def format_anthropic(messages: List[Message]) -> Dict[str, Any]:
        """
        转换为Anthropic API格式

        Anthropic格式要求system prompt分离：
        {
            "system": "...",
            "messages": [{"role": "user/assistant", "content": "..."}]
        }

        Args:
            messages: 消息列表

        Returns:
            包含system和messages的字典
        """
        system_parts = []
        conversation = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            elif msg.role == "tool":
                # Anthropic中tool结果作为user消息的tool_result content block
                tool_result = {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id or "",
                            "content": msg.content,
                        }
                    ],
                }
                conversation.append(tool_result)
            elif msg.role == "assistant" and msg.tool_calls:
                # 转换工具调用为Anthropic格式
                content_blocks = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.arguments) if tc.arguments else {}
                    except json.JSONDecodeError:
                        args = {}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.function_name,
                        "input": args,
                    })
                conversation.append({
                    "role": "assistant",
                    "content": content_blocks,
                })
            else:
                conversation.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        result: Dict[str, Any] = {"messages": conversation}
        if system_parts:
            result["system"] = "\n\n".join(system_parts)

        return result

    @staticmethod
    def format_gemini(messages: List[Message]) -> List[Dict[str, Any]]:
        """
        转换为Gemini API格式

        Gemini格式：
        [{"role": "user/model", "parts": [{"text": "..."}]}]

        Args:
            messages: 消息列表

        Returns:
            Gemini格式的消息列表
        """
        role_mapping = {
            "user": "user",
            "assistant": "model",
            "system": "user",
            "tool": "user",
        }

        result = []
        system_parts = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
                continue

            gemini_role = role_mapping.get(msg.role, "user")
            parts = [{"text": msg.content}]

            # 处理工具调用
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    parts.append({
                        "functionCall": {
                            "name": tc.function_name,
                            "args": json.loads(tc.arguments) if tc.arguments else {},
                        }
                    })

            result.append({"role": gemini_role, "parts": parts})

        # 将system消息作为第一条user消息
        if system_parts:
            system_text = "\n\n".join(system_parts)
            if result and result[0]["role"] == "user":
                result[0]["parts"].insert(0, {"text": system_text})
            else:
                result.insert(0, {"role": "user", "parts": [{"text": system_text}]})

        return result

    @staticmethod
    def apply_chat_template(
        messages: List[Message],
        template: str = "default",
        add_generation_prompt: bool = True,
    ) -> str:
        """
        应用聊天模板，将消息列表格式化为纯文本

        Args:
            messages: 消息列表
            template: 模板名称（"default", "alpaca", "chatml", "vicuna"）
            add_generation_prompt: 是否添加生成提示

        Returns:
            格式化后的纯文本
        """
        formatters = {
            "default": PromptFormatter._format_default,
            "alpaca": PromptFormatter._format_alpaca,
            "chatml": PromptFormatter._format_chatml,
            "vicuna": PromptFormatter._format_vicuna,
        }

        formatter = formatters.get(template, PromptFormatter._format_default)
        text = formatter(messages)

        if add_generation_prompt:
            generation_prompts = {
                "default": "\nAssistant:",
                "alpaca": "\n### Response:",
                "chatml": "<|im_start|>assistant\n",
                "vicuna": "\nASSISTANT:",
            }
            text += generation_prompts.get(template, "\nAssistant:")

        return text

    @staticmethod
    def _format_default(messages: List[Message]) -> str:
        """默认格式"""
        parts = []
        for msg in messages:
            role_label = msg.role.capitalize()
            parts.append(f"{role_label}: {msg.content}")
        return "\n".join(parts)

    @staticmethod
    def _format_alpaca(messages: List[Message]) -> str:
        """Alpaca格式"""
        parts = []
        for msg in messages:
            if msg.role == "system":
                parts.append(f"### Instruction:\n{msg.content}")
            elif msg.role == "user":
                parts.append(f"### Instruction:\n{msg.content}")
            elif msg.role == "assistant":
                parts.append(f"### Response:\n{msg.content}")
        return "\n\n".join(parts)

    @staticmethod
    def _format_chatml(messages: List[Message]) -> str:
        """ChatML格式"""
        parts = []
        for msg in messages:
            role = msg.role
            if role == "system":
                role = "system"
            elif role == "tool":
                role = "user"
            parts.append(f"<|im_start|>{role}\n{msg.content}<|im_end|>")
        return "\n".join(parts)

    @staticmethod
    def _format_vicuna(messages: List[Message]) -> str:
        """Vicuna格式"""
        parts = []
        for msg in messages:
            if msg.role == "system":
                parts.append(f"SYSTEM: {msg.content}")
            elif msg.role == "user":
                parts.append(f"USER: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"ASSISTANT: {msg.content}")
        return "\n".join(parts)

    @staticmethod
    def merge_messages(messages: List[Message]) -> List[Message]:
        """
        合并连续的同角色消息

        Args:
            messages: 消息列表

        Returns:
            合并后的消息列表
        """
        if not messages:
            return []

        merged = [Message(
            role=messages[0].role,
            content=messages[0].content,
            name=messages[0].name,
            tool_calls=messages[0].tool_calls,
            tool_call_id=messages[0].tool_call_id,
        )]

        for msg in messages[1:]:
            if msg.role == merged[-1].role and not msg.tool_calls:
                merged[-1] = Message(
                    role=merged[-1].role,
                    content=merged[-1].content + "\n" + msg.content,
                    name=merged[-1].name or msg.name,
                )
            else:
                merged.append(Message(
                    role=msg.role,
                    content=msg.content,
                    name=msg.name,
                    tool_calls=msg.tool_calls,
                    tool_call_id=msg.tool_call_id,
                ))

        return merged

    @staticmethod
    def truncate_messages(
        messages: List[Message],
        max_tokens: int,
        count_fn=None,
    ) -> List[Message]:
        """
        截断消息列表以适应Token限制

        保留system消息，从最早的非system消息开始截断。

        Args:
            messages: 消息列表
            max_tokens: 最大Token数
            count_fn: Token计数函数

        Returns:
            截断后的消息列表
        """
        if count_fn is None:
            count_fn = lambda text: len(text.split())

        # 分离system消息和对话消息
        system_msgs = [m for m in messages if m.role == "system"]
        conversation = [m for m in messages if m.role != "system"]

        # 计算system消息的token数
        system_tokens = sum(count_fn(m.content) for m in system_msgs)
        remaining = max_tokens - system_tokens

        if remaining <= 0:
            return system_msgs

        # 从后往前保留消息
        kept = []
        for msg in reversed(conversation):
            msg_tokens = count_fn(msg.content)
            if remaining - msg_tokens < 0:
                break
            kept.insert(0, msg)
            remaining -= msg_tokens

        return system_msgs + kept

    @staticmethod
    def extract_tool_calls_from_response(content: str) -> List[Dict[str, Any]]:
        """
        从响应文本中提取工具调用（解析JSON格式的工具调用）

        Args:
            content: 响应文本

        Returns:
            解析出的工具调用列表
        """
        tool_calls = []

        # 尝试匹配 function_call JSON块
        patterns = [
            r'\{"function_call":\s*(\{[^}]+\})\}',
            r'\{"name":\s*"([^"]+)"[^}]*"arguments":\s*(\{[^}]*\})\}',
            r'```json\s*(\{[^`]*?\})\s*```',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    if isinstance(match, tuple):
                        data = json.loads(match[0]) if match[0].startswith("{") else {}
                        if not data:
                            data = {"name": match[0], "arguments": match[1]}
                    else:
                        data = json.loads(match)

                    if "function_call" in data:
                        fc = data["function_call"]
                        tool_calls.append({
                            "name": fc.get("name", ""),
                            "arguments": json.dumps(fc.get("arguments", {})),
                        })
                    elif "name" in data:
                        tool_calls.append({
                            "name": data["name"],
                            "arguments": json.dumps(data.get("arguments", {})),
                        })
                except json.JSONDecodeError:
                    continue

        return tool_calls
