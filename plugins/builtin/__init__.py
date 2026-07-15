"""
ClawHub 内置插件模块

提供常用功能的内置插件集合。
"""

from .web_search import WebSearchPlugin, SearchResult, SearchEngine
from .file_ops import FileOperationsPlugin, FileInfo, CompressionFormat
from .code_runner import CodeRunnerPlugin, ExecutionResult, Language
from .calendar import CalendarPlugin, CalendarEvent, CalendarProvider
from .image_gen import ImageGenerationPlugin, ImageResult, ImageProvider
from .document import DocumentPlugin, DocumentInfo, DocumentFormat
from .database import DatabasePlugin, QueryResult, ConnectionConfig
from .api_tester import APITesterPlugin, APIResponse, APITestCase
from .git_manager import GitManagerPlugin, GitOperation, RepositoryInfo
from .screenshot import ScreenshotPlugin, ScreenshotOptions, CaptureMode

__all__ = [
    # Web Search
    "WebSearchPlugin",
    "SearchResult",
    "SearchEngine",
    # File Operations
    "FileOperationsPlugin",
    "FileInfo",
    "CompressionFormat",
    # Code Runner
    "CodeRunnerPlugin",
    "ExecutionResult",
    "Language",
    # Calendar
    "CalendarPlugin",
    "CalendarEvent",
    "CalendarProvider",
    # Image Generation
    "ImageGenerationPlugin",
    "ImageResult",
    "ImageProvider",
    # Document
    "DocumentPlugin",
    "DocumentInfo",
    "DocumentFormat",
    # Database
    "DatabasePlugin",
    "QueryResult",
    "ConnectionConfig",
    # API Tester
    "APITesterPlugin",
    "APIResponse",
    "APITestCase",
    # Git Manager
    "GitManagerPlugin",
    "GitOperation",
    "RepositoryInfo",
    # Screenshot
    "ScreenshotPlugin",
    "ScreenshotOptions",
    "CaptureMode",
]

__version__ = "1.0.0"
