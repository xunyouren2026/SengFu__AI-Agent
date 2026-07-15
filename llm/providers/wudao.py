"""
悟道 (WuDao) LLM Provider

悟道是北京智源人工智能研究院(BAAI)推出的超大规模智能模型，专注于学术研究和科学计算。

支持的模型:
- wudao-2.0: 悟道2.0版本
- wudao-chat: 对话版本

API文档: https://api.baai.ac.cn/v1

Author: AGI Team
Version: 1.0.0
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .base import (
    OpenAICompatibleProvider, LLMConfig, LLMResponse,
)

logger = logging.getLogger(__name__)


class WuDaoTaskType(Enum):
    """悟道支持的任务类型"""
    TEXT_GENERATION = "text_generation"
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CODE_GENERATION = "code_generation"
    SCIENTIFIC_COMPUTING = "scientific_computing"


@dataclass
class ResearchPaper:
    """研究论文数据结构"""
    title: str
    abstract: str
    keywords: List[str]
    content: str
    citations: List[str]


@dataclass
class ScientificQuery:
    """科学查询"""
    query: str
    domain: str  # 领域，如physics, chemistry, biology
    context: Optional[str] = None


class WuDaoProvider(OpenAICompatibleProvider):
    """
    悟道 (WuDao) LLM Provider

    悟道大模型特点:
        - 超大规模参数 (1.75万亿参数)
        - 多模态能力
        - 科学知识理解
        - 学术研究支持
        - 中英文双语

    适用场景:
        - 学术研究辅助
        - 科学文献分析
        - 知识问答
        - 多语言翻译
        - 代码生成

    Example:
        ```python
        provider = WuDaoProvider(LLMConfig(
            model_id="wudao-2.0",
            api_key="your_api_key"
        ))

        # 学术研究
        response = await provider.research_assist(
            topic="量子计算的发展",
            task_type="literature_review"
        )

        # 科学问答
        response = await provider.scientific_qa(
            query=ScientificQuery("解释量子纠缠", "physics")
        )
        ```
    """

    PROVIDER_NAME = "wudao"
    SUPPORTED_MODELS = {
        "wudao-2.0",
        "wudao-chat",
    }
    DEFAULT_MODEL = "wudao-2.0"

    # API配置
    API_BASE = "https://api.baai.ac.cn/v1"
    CHAT_ENDPOINT = "/chat/completions"
    RATE_LIMIT_INTERVAL = 0.2

    # 模型特性配置
    MODEL_CONFIGS = {
        "wudao-2.0": {
            "max_tokens": 4096,
            "context_window": 2048,
            "supports_vision": False,
            "supports_function_calling": True,
            "supports_streaming": True,
            "description": "悟道2.0，超大规模模型",
            "parameters": "1.75T",
        },
        "wudao-chat": {
            "max_tokens": 2048,
            "context_window": 4096,
            "supports_vision": False,
            "supports_function_calling": True,
            "supports_streaming": True,
            "description": "悟道对话版本",
            "parameters": "100B",
        },
    }

    # 支持的学术领域
    RESEARCH_DOMAINS = {
        "physics", "chemistry", "biology", "mathematics",
        "computer_science", "engineering", "medicine",
        "economics", "psychology", "linguistics"
    }

    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        self._research_cache: Dict[str, Any] = {}

    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        headers = super()._get_headers()
        headers["X-Research-Client"] = "python-sdk"
        return headers

    def get_stats(self) -> Dict[str, Any]:
        """获取Provider统计信息"""
        stats = super().get_stats()
        stats["research_domains"] = list(self.RESEARCH_DOMAINS)
        return stats

    # ==================== 学术研究功能 ====================

    async def research_assist(
        self,
        topic: str,
        task_type: str = "literature_review",
        **kwargs
    ) -> LLMResponse:
        """
        学术研究辅助。

        Args:
            topic: 研究主题
            task_type: 任务类型 (literature_review, hypothesis, methodology)
            **kwargs: 其他参数

        Returns:
            LLM响应对象
        """
        task_prompts = {
            "literature_review": f'请为\u201c{topic}\u201d撰写一份文献综述，包括研究背景、现状和趋势：',
            "hypothesis": f'请基于\u201c{topic}\u201d提出研究假设，并说明验证方法：',
            "methodology": f'请为\u201c{topic}\u201d研究设计实验方法：',
            "analysis": f'请分析\u201c{topic}\u201d的关键问题和挑战：',
        }

        prompt = task_prompts.get(task_type, task_prompts["literature_review"])

        messages = [
            {
                "role": "system",
                "content": "你是一位学术研究助手，擅长文献综述、研究设计和学术写作。"
            },
            {"role": "user", "content": prompt}
        ]

        config = LLMConfig(
            model_id="wudao-2.0",
            api_key=self._config.api_key if self._config else None,
            temperature=0.3,
            **kwargs
        )

        return await self._async_generate(messages, config)

    async def scientific_qa(
        self,
        query: ScientificQuery,
        **kwargs
    ) -> LLMResponse:
        """
        科学问答。

        Args:
            query: 科学查询对象
            **kwargs: 其他参数

        Returns:
            LLM响应对象
        """
        domain_prompts = {
            "physics": "你是一位物理学专家。",
            "chemistry": "你是一位化学专家。",
            "biology": "你是一位生物学专家。",
            "mathematics": "你是一位数学专家。",
            "computer_science": "你是一位计算机科学专家。",
        }

        system_prompt = domain_prompts.get(
            query.domain,
            "你是一位科学研究专家。"
        )

        user_prompt = f"问题：{query.query}"
        if query.context:
            user_prompt += f"\n\n背景信息：{query.context}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        config = LLMConfig(
            model_id="wudao-2.0",
            api_key=self._config.api_key if self._config else None,
            **kwargs
        )

        return await self._async_generate(messages, config)

    async def analyze_paper(
        self,
        paper_content: str,
        analysis_type: str = "summary",
        **kwargs
    ) -> Dict[str, Any]:
        """
        分析学术论文。

        Args:
            paper_content: 论文内容
            analysis_type: 分析类型 (summary, critique, comparison)
            **kwargs: 其他参数

        Returns:
            分析结果字典
        """
        analysis_prompts = {
            "summary": "请总结以下论文的核心贡献、方法和结论：",
            "critique": "请批判性分析以下论文的优点和不足：",
            "comparison": "请将以下论文与领域内相关工作进行比较：",
        }

        prompt = analysis_prompts.get(analysis_type, analysis_prompts["summary"])

        messages = [
            {
                "role": "system",
                "content": "你是一位学术论文评审专家，擅长论文分析和评价。"
            },
            {
                "role": "user",
                "content": f"{prompt}\n\n{paper_content[:8000]}"
            }
        ]

        config = LLMConfig(
            model_id="wudao-2.0",
            api_key=self._config.api_key if self._config else None,
            **kwargs
        )

        response = await self._async_generate(messages, config)

        return {
            "analysis": response.content,
            "analysis_type": analysis_type,
            "model": config.model_id,
            "latency_ms": response.latency_ms,
        }

    async def generate_citation(
        self,
        paper_info: Dict[str, str],
        style: str = "apa",
        **kwargs
    ) -> str:
        """
        生成引用格式。

        Args:
            paper_info: 论文信息字典
            style: 引用格式 (apa, mla, chicago, gb/t7714)
            **kwargs: 其他参数

        Returns:
            格式化引用字符串
        """
        style_prompts = {
            "apa": "请按照APA格式生成引用：",
            "mla": "请按照MLA格式生成引用：",
            "chicago": "请按照Chicago格式生成引用：",
            "gb/t7714": "请按照GB/T 7714格式生成引用：",
        }

        prompt = style_prompts.get(style, style_prompts["apa"])

        info_text = "\n".join(f"{k}: {v}" for k, v in paper_info.items())

        messages = [
            {"role": "user", "content": f"{prompt}\n\n{info_text}"}
        ]

        config = LLMConfig(
            model_id="wudao-chat",
            api_key=self._config.api_key if self._config else None,
            **kwargs
        )

        response = await self._async_generate(messages, config)
        return response.content.strip()
