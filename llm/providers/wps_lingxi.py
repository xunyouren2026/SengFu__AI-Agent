"""
WPS 灵犀 (Lingxi) AI Provider

WPS灵犀是金山办公推出的AI助手，专注于办公文档处理。

支持的模型:
- lingxi-chat: 通用对话模型
- lingxi-doc: 文档处理模型
- lingxi-ppt: PPT生成模型

API文档: https://api.wps.cn/

Author: AGI Team
Version: 1.0.0
"""

import json
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum

from .base import (
    OpenAICompatibleProvider, LLMConfig, LLMResponse,
    ModelCapability,
)

logger = logging.getLogger(__name__)


class WPSDocumentType(Enum):
    """WPS文档类型"""
    DOC = "doc"
    DOCX = "docx"
    XLS = "xls"
    XLSX = "xlsx"
    PPT = "ppt"
    PPTX = "pptx"
    PDF = "pdf"
    TXT = "txt"


@dataclass
class DocumentOperation:
    """文档操作"""
    operation_type: str
    params: Dict[str, Any]
    description: str = ""


@dataclass
class DocumentAnalysisResult:
    """文档分析结果"""
    summary: str
    key_points: List[str]
    word_count: int
    metadata: Dict[str, Any]


class WPSLingxiProvider(OpenAICompatibleProvider):
    """
    WPS 灵犀 AI Provider

    灵犀AI特点:
        - 专注于办公文档处理
        - 支持Word、Excel、PPT等格式
        - 文档智能分析和摘要
        - PPT智能生成
        - 表格数据处理

    适用场景:
        - 办公文档智能处理
        - 文档内容摘要和提取
        - PPT自动生成
        - 表格数据分析
        - 文档格式转换

    Example:
        ```python
        provider = WPSLingxiProvider(LLMConfig(
            model_id="lingxi-doc",
            api_key="your_api_key"
        ))

        # 文档分析
        result = await provider.analyze_document(
            document_path="report.docx",
            analysis_type="summary"
        )

        # PPT生成
        ppt = await provider.generate_ppt(
            topic="年度工作总结",
            slides_count=10
        )
        ```
    """

    PROVIDER_NAME = "wps_lingxi"
    SUPPORTED_MODELS = {
        "lingxi-chat",
        "lingxi-doc",
        "lingxi-ppt",
    }
    DEFAULT_MODEL = "lingxi-chat"

    # API配置
    API_BASE = "https://api.wps.cn/ai/v1"
    CHAT_ENDPOINT = "/chat/completions"
    DOC_ENDPOINT = "/documents/analyze"
    PPT_ENDPOINT = "/presentations/generate"
    RATE_LIMIT_INTERVAL = 0.2

    # 模型特性配置
    MODEL_CONFIGS = {
        "lingxi-chat": {
            "max_tokens": 4096,
            "context_window": 8192,
            "supports_vision": False,
            "supports_function_calling": True,
            "supports_streaming": True,
            "description": "通用对话模型",
        },
        "lingxi-doc": {
            "max_tokens": 4096,
            "context_window": 16384,
            "supports_vision": False,
            "supports_function_calling": True,
            "supports_streaming": True,
            "description": "文档处理专用模型",
        },
        "lingxi-ppt": {
            "max_tokens": 4096,
            "context_window": 8192,
            "supports_vision": False,
            "supports_function_calling": True,
            "supports_streaming": False,
            "description": "PPT生成专用模型",
        },
    }

    # 支持的文档格式
    SUPPORTED_DOC_FORMATS = {
        ".doc", ".docx", ".xls", ".xlsx",
        ".ppt", ".pptx", ".pdf", ".txt"
    }

    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        self._document_cache: Dict[str, Any] = {}

    def get_capabilities(self) -> Set[ModelCapability]:
        """获取当前模型支持的能力"""
        model_config = self._get_model_config()
        capabilities: Set[ModelCapability] = {
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.JSON_MODE,
        }

        if model_config.get("supports_streaming"):
            capabilities.add(ModelCapability.STREAMING)

        if model_config.get("supports_vision"):
            capabilities.add(ModelCapability.VISION)

        return capabilities

    def get_stats(self) -> Dict[str, Any]:
        """获取Provider统计信息"""
        stats = super().get_stats()
        stats["supported_formats"] = list(self.SUPPORTED_DOC_FORMATS)
        return stats

    # ==================== 文档处理功能 ====================

    async def analyze_document(
        self,
        document_content: str,
        analysis_type: str = "summary",
        **kwargs
    ) -> DocumentAnalysisResult:
        """
        分析文档内容。

        Args:
            document_content: 文档内容文本
            analysis_type: 分析类型 (summary, keywords, outline)
            **kwargs: 其他参数

        Returns:
            DocumentAnalysisResult
        """
        prompts = {
            "summary": "请对以下文档内容进行摘要，提炼出核心要点：",
            "keywords": "请从以下文档中提取关键词和关键短语：",
            "outline": "请为以下文档生成详细的大纲：",
        }

        prompt = prompts.get(analysis_type, prompts["summary"])

        messages = [
            {
                "role": "user",
                "content": f"{prompt}\n\n{document_content[:8000]}"
            }
        ]

        config = LLMConfig(
            model_id="lingxi-doc",
            api_key=self._config.api_key if self._config else None,
            **kwargs
        )

        response = await self._async_generate(messages, config)

        # 解析结果
        summary = response.content
        key_points = [line.strip()[2:] for line in summary.split("\n")
                      if line.strip().startswith("-") or line.strip().startswith("*")]

        return DocumentAnalysisResult(
            summary=summary,
            key_points=key_points if key_points else [summary[:100]],
            word_count=len(document_content.split()),
            metadata={
                "analysis_type": analysis_type,
                "model": config.model_id,
                "latency_ms": response.latency_ms,
            }
        )

    async def generate_ppt_outline(
        self,
        topic: str,
        slides_count: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成PPT大纲。

        Args:
            topic: PPT主题
            slides_count: 幻灯片数量
            **kwargs: 其他参数

        Returns:
            PPT大纲字典
        """
        prompt = f"""请为"{topic}"生成一个PPT大纲，包含{slides_count}页幻灯片。

要求：
1. 包含标题页、目录页、内容页和总结页
2. 每页包含标题和要点
3. 内容逻辑清晰，层次分明

请以JSON格式输出，格式如下：
{{
    "title": "PPT标题",
    "slides": [
        {{
            "slide_number": 1,
            "title": "幻灯片标题",
            "content": ["要点1", "要点2"],
            "layout": "title|content|two_column"
        }}
    ]
}}"""

        messages = [
            {"role": "user", "content": prompt}
        ]

        config = LLMConfig(
            model_id="lingxi-ppt",
            api_key=self._config.api_key if self._config else None,
            extra_body={"response_format": {"type": "json_object"}},
            **kwargs
        )

        response = await self._async_generate(messages, config)

        # 尝试解析JSON
        try:
            outline = json.loads(response.content)
        except json.JSONDecodeError:
            outline = {
                "title": topic,
                "slides": [{"slide_number": i + 1, "title": f"幻灯片 {i + 1}", "content": [response.content]}
                           for i in range(slides_count)]
            }

        return outline

    async def generate_ppt_content(
        self,
        outline: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        根据大纲生成PPT详细内容。

        Args:
            outline: PPT大纲
            **kwargs: 其他参数

        Returns:
            完整的PPT内容
        """
        slides = outline.get("slides", [])
        detailed_slides = []

        for slide in slides:
            slide_num = slide.get("slide_number", 0)
            title = slide.get("title", "")
            content_points = slide.get("content", [])

            prompt = f"""请为PPT第{slide_num}页"{title}"生成详细内容。

要点：
{chr(10).join(f"- {point}" for point in content_points)}

请生成适合PPT展示的详细内容，包含：
1. 该页的详细说明文字
2. 建议的视觉元素（如图表、图片等）
3. 演讲者备注"""

            messages = [
                {"role": "user", "content": prompt}
            ]

            config = LLMConfig(
                model_id="lingxi-ppt",
                api_key=self._config.api_key if self._config else None,
                **kwargs
            )

            response = await self._async_generate(messages, config)

            detailed_slides.append({
                **slide,
                "detailed_content": response.content
            })

        return {
            "title": outline.get("title", ""),
            "slides": detailed_slides
        }

    async def process_spreadsheet(
        self,
        data: List[List[Any]],
        operation: str = "analyze",
        **kwargs
    ) -> Dict[str, Any]:
        """
        处理电子表格数据。

        Args:
            data: 表格数据 (二维列表)
            operation: 操作类型 (analyze, summarize, transform)
            **kwargs: 其他参数

        Returns:
            处理结果
        """
        # 将数据转换为文本格式
        headers = data[0] if data else []
        rows = data[1:] if len(data) > 1 else []

        data_text = " | ".join(str(h) for h in headers) + "\n"
        for row in rows[:20]:  # 限制行数
            data_text += " | ".join(str(cell) for cell in row) + "\n"

        operations = {
            "analyze": "请分析以下表格数据，提供数据洞察和趋势分析：",
            "summarize": "请对以下表格数据进行汇总统计：",
            "transform": "请建议如何转换以下表格数据以更好地展示信息：",
        }

        prompt = f"{operations.get(operation, operations['analyze'])}\n\n{data_text}"

        messages = [
            {"role": "user", "content": prompt}
        ]

        config = LLMConfig(
            model_id="lingxi-doc",
            api_key=self._config.api_key if self._config else None,
            **kwargs
        )

        response = await self._async_generate(messages, config)

        return {
            "operation": operation,
            "result": response.content,
            "row_count": len(rows),
            "column_count": len(headers),
            "metadata": {
                "model": config.model_id,
                "latency_ms": response.latency_ms,
            }
        }
