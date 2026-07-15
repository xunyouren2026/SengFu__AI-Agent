"""
Google AI兼容接口模块

提供统一的Google AI服务访问接口，包括Gemini、PaLM和Vertex AI。
支持文本生成、嵌入、多模态处理和流式响应。

模块路径: compat/google/__init__.py
"""

from __future__ import annotations

import os
import logging
from typing import TYPE_CHECKING, Optional, Union

# 配置模块日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 版本信息
__version__ = "1.0.0"
__author__ = "AGI Unified Framework"

# 类型别名
ApiKeyType = Union[str, None]
ProjectIdType = Union[str, None]
LocationType = Union[str, None]


class GoogleAIError(Exception):
    """Google AI基础异常类"""
    
    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.status_code:
            return f"[HTTP {self.status_code}] {self.message}"
        return self.message


class AuthenticationError(GoogleAIError):
    """认证错误"""
    pass


class RateLimitError(GoogleAIError):
    """速率限制错误"""
    pass


class InvalidRequestError(GoogleAIError):
    """无效请求错误"""
    pass


class ModelNotFoundError(GoogleAIError):
    """模型未找到错误"""
    pass


class ServerError(GoogleAIError):
    """服务器错误"""
    pass


class ConfigurationError(GoogleAIError):
    """配置错误"""
    pass


def get_api_key(env_var: str = "GOOGLE_API_KEY") -> Optional[str]:
    """
    从环境变量获取API密钥
    
    Args:
        env_var: 环境变量名称，默认为 GOOGLE_API_KEY
        
    Returns:
        API密钥或None
    """
    return os.environ.get(env_var)


def get_project_id(env_var: str = "GOOGLE_CLOUD_PROJECT") -> Optional[str]:
    """
    从环境变量获取Google Cloud项目ID
    
    Args:
        env_var: 环境变量名称，默认为 GOOGLE_CLOUD_PROJECT
        
    Returns:
        项目ID或None
    """
    return os.environ.get(env_var)


def get_default_location() -> str:
    """
    获取默认区域
    
    Returns:
        默认区域代码
    """
    return os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")


# 延迟导入以避免循环依赖
if TYPE_CHECKING:
    from .gemini import GeminiClient, GeminiConfig
    from .palm import PalmClient, PalmConfig
    from .vertex_ai import VertexAIClient, VertexAIConfig


class GoogleAIClient:
    """
    Google AI统一客户端
    
    提供对Gemini、PaLM和Vertex AI的统一访问接口
    
    Example:
        >>> client = GoogleAIClient(api_key="your-api-key")
        >>> response = client.gemini.generate_text("Hello, world!")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        credentials_path: Optional[str] = None
    ):
        """
        初始化Google AI客户端
        
        Args:
            api_key: Google API密钥（用于Gemini和PaLM）
            project_id: Google Cloud项目ID（用于Vertex AI）
            location: Google Cloud区域
            credentials_path: 服务账号凭证文件路径
        """
        self._api_key = api_key or get_api_key()
        self._project_id = project_id or get_project_id()
        self._location = location or get_default_location()
        self._credentials_path = credentials_path
        
        self._gemini: Optional["GeminiClient"] = None
        self._palm: Optional["PalmClient"] = None
        self._vertex_ai: Optional["VertexAIClient"] = None
        
        logger.info("GoogleAIClient initialized")
    
    @property
    def gemini(self) -> "GeminiClient":
        """
        获取Gemini客户端
        
        Returns:
            GeminiClient实例
        """
        if self._gemini is None:
            from .gemini import GeminiClient
            if not self._api_key:
                raise ConfigurationError(
                    "API key is required for Gemini. "
                    "Set GOOGLE_API_KEY environment variable or pass api_key parameter."
                )
            self._gemini = GeminiClient(api_key=self._api_key)
        return self._gemini
    
    @property
    def palm(self) -> "PalmClient":
        """
        获取PaLM客户端
        
        Returns:
            PalmClient实例
        """
        if self._palm is None:
            from .palm import PalmClient
            if not self._api_key:
                raise ConfigurationError(
                    "API key is required for PaLM. "
                    "Set GOOGLE_API_KEY environment variable or pass api_key parameter."
                )
            self._palm = PalmClient(api_key=self._api_key)
        return self._palm
    
    @property
    def vertex_ai(self) -> "VertexAIClient":
        """
        获取Vertex AI客户端
        
        Returns:
            VertexAIClient实例
        """
        if self._vertex_ai is None:
            from .vertex_ai import VertexAIClient
            if not self._project_id:
                raise ConfigurationError(
                    "Project ID is required for Vertex AI. "
                    "Set GOOGLE_CLOUD_PROJECT environment variable or pass project_id parameter."
                )
            self._vertex_ai = VertexAIClient(
                project_id=self._project_id,
                location=self._location,
                credentials_path=self._credentials_path
            )
        return self._vertex_ai
    
    async def close(self) -> None:
        """关闭所有客户端连接"""
        if self._gemini:
            await self._gemini.close()
        if self._palm:
            await self._palm.close()
        if self._vertex_ai:
            await self._vertex_ai.close()
        logger.info("All Google AI clients closed")
    
    async def __aenter__(self) -> "GoogleAIClient":
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """异步上下文管理器出口"""
        await self.close()


# 便捷函数

def create_client(
    api_key: Optional[str] = None,
    project_id: Optional[str] = None,
    location: Optional[str] = None
) -> GoogleAIClient:
    """
    创建Google AI客户端
    
    Args:
        api_key: Google API密钥
        project_id: Google Cloud项目ID
        location: Google Cloud区域
        
    Returns:
        GoogleAIClient实例
    """
    return GoogleAIClient(
        api_key=api_key,
        project_id=project_id,
        location=location
    )


# 导出公共接口
__all__ = [
    # 版本信息
    "__version__",
    
    # 异常类
    "GoogleAIError",
    "AuthenticationError",
    "RateLimitError",
    "InvalidRequestError",
    "ModelNotFoundError",
    "ServerError",
    "ConfigurationError",
    
    # 类型别名
    "ApiKeyType",
    "ProjectIdType",
    "LocationType",
    
    # 工具函数
    "get_api_key",
    "get_project_id",
    "get_default_location",
    
    # 客户端类
    "GoogleAIClient",
    "create_client",
]

# 延迟加载子模块
def __getattr__(name: str):
    """延迟加载子模块"""
    if name == "GeminiClient":
        from .gemini import GeminiClient
        return GeminiClient
    elif name == "PalmClient":
        from .palm import PalmClient
        return PalmClient
    elif name == "VertexAIClient":
        from .vertex_ai import VertexAIClient
        return VertexAIClient
    elif name == "GeminiConfig":
        from .gemini import GeminiConfig
        return GeminiConfig
    elif name == "PalmConfig":
        from .palm import PalmConfig
        return PalmConfig
    elif name == "VertexAIConfig":
        from .vertex_ai import VertexAIConfig
        return VertexAIConfig
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
