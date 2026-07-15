"""
内置工具模块
提供常用的内置工具类
"""

from .file_reader import (
    FileReader,
    TextFileReader,
    JSONFileReader,
    CSVFileReader,
    MarkdownFileReader,
    YAMLFileReader,
    BinaryFileReader,
    EncodingDetector,
    FileMetadataExtractor,
    PathValidator,
    ReadResult,
    FileType,
)

from .web_search import (
    WebSearch,
    DuckDuckGoSearch,
    BingSearch,
    SearchAggregator,
    RateLimiter,
    ResultCache,
    SearchHistory,
    SafeSearchFilter,
    SearchResult,
    SearchProvider,
    SearchResponse,
)


__all__ = [
    # 文件读取
    'FileReader',
    'TextFileReader',
    'JSONFileReader',
    'CSVFileReader',
    'MarkdownFileReader',
    'YAMLFileReader',
    'BinaryFileReader',
    'EncodingDetector',
    'FileMetadataExtractor',
    'PathValidator',
    'ReadResult',
    'FileType',

    # 网页搜索
    'WebSearch',
    'DuckDuckGoSearch',
    'BingSearch',
    'SearchAggregator',
    'RateLimiter',
    'ResultCache',
    'SearchHistory',
    'SafeSearchFilter',
    'SearchResult',
    'SearchProvider',
    'SearchResponse',
]


__version__ = '1.0.0'
__author__ = 'AGI Unified Framework'
