"""
ConversationDataset - 测试对话数据集生成器

模块路径: testing/database/conversation_dataset.py

提供测试用对话数据集的生成、管理和查询功能。
"""

import os
import sys
import json
import time
import random
import hashlib
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Generator
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

import pytest
import numpy as np


class ConversationRole(Enum):
    """对话角色"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"


class ConversationDomain(Enum):
    """对话领域"""
    GENERAL = "general"
    TECHNICAL = "technical"
    CUSTOMER_SERVICE = "customer_service"
    CODING = "coding"
    CREATIVE = "creative"
    EDUCATION = "education"


@dataclass
class Message:
    """对话消息"""
    role: str
    content: str
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class Conversation:
    """完整对话"""
    conversation_id: str
    messages: List[Message] = field(default_factory=list)
    domain: str = "general"
    language: str = "zh"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    tags: List[str] = field(default_factory=list)

    def add_message(self, role: str, content: str) -> Message:
        """添加消息到对话"""
        msg = Message(role=role, content=content, timestamp=time.time())
        self.messages.append(msg)
        return msg

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "messages": [m.to_dict() for m in self.messages],
            "domain": self.domain,
            "language": self.language,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "tags": self.tags,
        }


@dataclass
class ConversationStats:
    """对话统计信息"""
    total_conversations: int = 0
    total_messages: int = 0
    avg_messages_per_conv: float = 0.0
    avg_turn_length: float = 0.0
    domains: Dict[str, int] = field(default_factory=dict)
    languages: Dict[str, int] = field(default_factory=dict)


class ConversationDataset:
    """测试对话数据集生成器

    提供测试用对话数据集的生成和管理功能:
        - 生成多轮对话数据
        - 支持多种对话领域和角色
        - 支持中英文对话
        - 生成对话统计和分析数据
        - 支持对话模板和变体生成
        - 提供对话质量评估指标
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialized = False
        self._conversations: Dict[str, Conversation] = {}
        self._seed: int = self.config.get("seed", 42)
        self._templates: Dict[str, List[Dict[str, str]]] = {}
        self._vocabularies: Dict[str, List[str]] = {}
        self._init_templates()

    def initialize(self) -> None:
        """初始化数据集生成器"""
        random.seed(self._seed)
        self._initialized = True

    def _init_templates(self) -> None:
        """初始化对话模板"""
        self._templates["greeting"] = [
            {"role": "user", "content": "你好，请问你能帮我什么？"},
            {"role": "assistant", "content": "你好！我可以帮助你回答问题、提供建议和进行各种对话。请问有什么需要帮助的吗？"},
        ]
        self._templates["coding_help"] = [
            {"role": "user", "content": "请帮我写一个Python函数来计算斐波那契数列。"},
            {"role": "assistant", "content": "当然可以。以下是一个计算斐波那契数列的Python函数：\n\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b"},
            {"role": "user", "content": "能解释一下这个函数的工作原理吗？"},
            {"role": "assistant", "content": "这个函数使用迭代方法计算斐波那契数列。它从0和1开始，每次迭代将前两个数相加得到下一个数。时间复杂度是O(n)，空间复杂度是O(1)。"},
        ]
        self._templates["customer_service"] = [
            {"role": "user", "content": "我需要查询我的订单状态。"},
            {"role": "assistant", "content": "好的，请提供您的订单号，我来帮您查询。"},
            {"role": "user", "content": "订单号是ORD-20240101-001。"},
            {"role": "assistant", "content": "已查到您的订单。当前状态是：已发货，预计3-5个工作日内送达。快递单号：SF1234567890。"},
        ]
        self._templates["technical_qa"] = [
            {"role": "user", "content": "什么是RESTful API？"},
            {"role": "assistant", "content": "RESTful API是一种基于REST架构风格设计的Web API。它使用HTTP方法（GET、POST、PUT、DELETE）来操作资源，通过URL标识资源，使用JSON格式传输数据。"},
        ]
        self._vocabularies["zh_general"] = [
            "你好", "谢谢", "请问", "帮助", "问题", "回答", "理解", "解释",
            "建议", "方法", "功能", "使用", "需要", "可以", "如何", "什么",
            "为什么", "因为", "所以", "但是", "如果", "那么", "这个", "那个",
        ]
        self._vocabularies["en_general"] = [
            "hello", "thank you", "please", "help", "question", "answer",
            "understand", "explain", "suggest", "method", "function", "use",
            "need", "can", "how", "what", "why", "because", "so", "but",
            "if", "then", "this", "that",
        ]

    def generate_conversation(self, domain: str = "general",
                               num_turns: int = 4,
                               language: str = "zh") -> Conversation:
        """生成单条对话

        Args:
            domain: 对话领域
            num_turns: 对话轮数
            language: 语言

        Returns:
            Conversation对话对象
        """
        if not self._initialized:
            self.initialize()
        conv_id = f"conv_{hashlib.md5(f'{domain}_{time.time()}'.encode()).hexdigest()[:12]}"
        conv = Conversation(
            conversation_id=conv_id,
            domain=domain,
            language=language,
            created_at=time.time(),
            tags=[domain, language, f"turns_{num_turns}"],
        )

        template_key = self._select_template_key(domain)
        if template_key and template_key in self._templates:
            template = self._templates[template_key]
            for msg_template in template[:num_turns]:
                content = self._apply_variation(msg_template["content"], language)
                conv.add_message(msg_template["role"], content)
        else:
            for i in range(num_turns):
                role = "user" if i % 2 == 0 else "assistant"
                content = self._generate_random_message(role, domain, language)
                conv.add_message(role, content)

        self._conversations[conv_id] = conv
        return conv

    def _select_template_key(self, domain: str) -> Optional[str]:
        """根据领域选择模板"""
        domain_map = {
            "general": "greeting",
            "coding": "coding_help",
            "customer_service": "customer_service",
            "technical": "technical_qa",
        }
        return domain_map.get(domain)

    def _apply_variation(self, content: str, language: str) -> str:
        """对模板内容应用随机变体"""
        vocab_key = f"{language}_general"
        vocab = self._vocabularies.get(vocab_key, [])
        if not vocab:
            return content
        words = content.split()
        if len(words) > 3 and random.random() < 0.3:
            idx = random.randint(0, len(words) - 1)
            words[idx] = random.choice(vocab)
        return " ".join(words) if language == "en" else "".join(words)

    def _generate_random_message(self, role: str, domain: str,
                                  language: str) -> str:
        """生成随机消息"""
        vocab_key = f"{language}_general"
        vocab = self._vocabularies.get(vocab_key, self._vocabularies.get("zh_general", []))
        num_words = random.randint(5, 20)
        if language == "zh":
            return "".join(random.choice(vocab) for _ in range(num_words))
        return " ".join(random.choice(vocab) for _ in range(num_words))

    def generate_batch(self, count: int, domain: str = "general",
                        min_turns: int = 2, max_turns: int = 10,
                        language: str = "zh") -> List[Conversation]:
        """批量生成对话

        Args:
            count: 生成数量
            domain: 对话领域
            min_turns: 最小轮数
            max_turns: 最大轮数
            language: 语言

        Returns:
            对话列表
        """
        conversations = []
        for _ in range(count):
            num_turns = random.randint(min_turns, max_turns)
            conv = self.generate_conversation(domain, num_turns, language)
            conversations.append(conv)
        return conversations

    def generate_multi_domain_batch(self, count_per_domain: int = 10,
                                     domains: Optional[List[str]] = None) -> List[Conversation]:
        """生成多领域对话

        Args:
            count_per_domain: 每个领域的对话数量
            domains: 领域列表

        Returns:
            对话列表
        """
        if domains is None:
            domains = [d.value for d in ConversationDomain]
        conversations = []
        for domain in domains:
            batch = self.generate_batch(count_per_domain, domain=domain)
            conversations.extend(batch)
        return conversations

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """获取对话

        Args:
            conversation_id: 对话ID

        Returns:
            Conversation或None
        """
        return self._conversations.get(conversation_id)

    def get_messages(self, conversation_id: str) -> List[Message]:
        """获取对话的消息列表

        Args:
            conversation_id: 对话ID

        Returns:
            消息列表
        """
        conv = self._conversations.get(conversation_id)
        return conv.messages if conv else []

    def search_conversations(self, keyword: str,
                              domain: Optional[str] = None) -> List[str]:
        """搜索包含关键词的对话

        Args:
            keyword: 搜索关键词
            domain: 限定领域

        Returns:
            匹配的对话ID列表
        """
        results = []
        for conv_id, conv in self._conversations.items():
            if domain and conv.domain != domain:
                continue
            for msg in conv.messages:
                if keyword in msg.content:
                    results.append(conv_id)
                    break
        return results

    def get_statistics(self) -> ConversationStats:
        """获取对话数据集统计信息

        Returns:
            ConversationStats统计对象
        """
        if not self._conversations:
            return ConversationStats()
        total_msgs = sum(len(c.messages) for c in self._conversations.values())
        domains = defaultdict(int)
        languages = defaultdict(int)
        for conv in self._conversations.values():
            domains[conv.domain] += 1
            languages[conv.language] += 1
        avg_msg_len = 0.0
        all_contents = []
        for conv in self._conversations.values():
            for msg in conv.messages:
                all_contents.append(len(msg.content))
        if all_contents:
            avg_msg_len = sum(all_contents) / len(all_contents)
        return ConversationStats(
            total_conversations=len(self._conversations),
            total_messages=total_msgs,
            avg_messages_per_conv=total_msgs / len(self._conversations),
            avg_turn_length=avg_msg_len,
            domains=dict(domains),
            languages=dict(languages),
        )

    def export_to_jsonl(self, filepath: str) -> int:
        """导出为JSONL格式（每行一个对话）

        Args:
            filepath: 输出文件路径

        Returns:
            导出的对话数量
        """
        with open(filepath, "w", encoding="utf-8") as f:
            for conv in self._conversations.values():
                f.write(json.dumps(conv.to_dict(), ensure_ascii=False, default=str) + "\n")
        return len(self._conversations)

    def export_to_training_format(self, filepath: str,
                                   system_prompt: str = "") -> int:
        """导出为训练格式（适合微调的prompt-completion对）

        Args:
            filepath: 输出文件路径
            system_prompt: 系统提示

        Returns:
            导出的训练样本数量
        """
        samples = []
        for conv in self._conversations.values():
            user_msgs = [m for m in conv.messages if m.role == "user"]
            assistant_msgs = [m for m in conv.messages if m.role == "assistant"]
            for u, a in zip(user_msgs, assistant_msgs):
                samples.append({
                    "prompt": u.content,
                    "completion": a.content,
                    "system": system_prompt,
                    "domain": conv.domain,
                })
        with open(filepath, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        return len(samples)

    def add_custom_template(self, name: str, messages: List[Dict[str, str]]) -> None:
        """添加自定义对话模板

        Args:
            name: 模板名称
            messages: 消息模板列表
        """
        self._templates[name] = messages

    def reset(self) -> None:
        """重置数据集"""
        self._conversations.clear()
