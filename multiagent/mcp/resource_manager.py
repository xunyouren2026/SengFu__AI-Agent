"""资源管理器模块。

本模块实现MCP资源管理，用于暴露文件、数据库连接等资源。
支持资源的注册、访问和订阅更新通知。
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from typing import Optional, Callable, Dict, Any, List, Union, Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod

from .schema import MCPResource, MCPResourceContent


class ResourceType(Enum):
    """资源类型枚举。"""
    FILE = "file"
    DIRECTORY = "directory"
    DATABASE = "database"
    HTTP = "http"
    MEMORY = "memory"
    CUSTOM = "custom"


class ResourceError(Exception):
    """资源错误基类。"""
    pass


class ResourceNotFoundError(ResourceError):
    """资源未找到错误。"""
    pass


class ResourceAccessError(ResourceError):
    """资源访问错误。"""
    pass


@dataclass
class ResourceMetadata:
    """资源元数据。
    
    Attributes:
        size: 资源大小（字节）
        created_at: 创建时间
        modified_at: 修改时间
        encoding: 编码方式
        extra: 额外元数据
    """
    size: Optional[int] = None
    created_at: Optional[float] = None
    modified_at: Optional[float] = None
    encoding: str = "utf-8"
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        result = {}
        if self.size is not None:
            result["size"] = self.size
        if self.created_at is not None:
            result["created_at"] = self.created_at
        if self.modified_at is not None:
            result["modified_at"] = self.modified_at
        result["encoding"] = self.encoding
        result.update(self.extra)
        return result


class ResourceProvider(ABC):
    """资源提供者抽象基类。"""
    
    @abstractmethod
    def list_resources(self) -> List[MCPResource]:
        """列出所有资源。
        
        Returns:
            资源列表
        """
        pass
    
    @abstractmethod
    def read_resource(self, uri: str) -> MCPResourceContent:
        """读取资源内容。
        
        Args:
            uri: 资源URI
            
        Returns:
            资源内容
        """
        pass
    
    @abstractmethod
    def exists(self, uri: str) -> bool:
        """检查资源是否存在。
        
        Args:
            uri: 资源URI
            
        Returns:
            是否存在
        """
        pass


class FileResourceProvider(ResourceProvider):
    """文件系统资源提供者。
    
    提供对本地文件系统资源的访问。
    """
    
    def __init__(
        self,
        base_path: str = ".",
        allowed_extensions: Optional[List[str]] = None,
        max_file_size: int = 10 * 1024 * 1024  # 10MB
    ):
        """初始化文件资源提供者。
        
        Args:
            base_path: 基础路径
            allowed_extensions: 允许的文件扩展名
            max_file_size: 最大文件大小（字节）
        """
        self.base_path = Path(base_path).resolve()
        self.allowed_extensions = set(allowed_extensions or [])
        self.max_file_size = max_file_size
        
        # MIME类型映射
        self._mime_types = {
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".json": "application/json",
            ".xml": "application/xml",
            ".html": "text/html",
            ".css": "text/css",
            ".js": "application/javascript",
            ".py": "text/x-python",
            ".yaml": "application/x-yaml",
            ".yml": "application/x-yaml",
            ".csv": "text/csv",
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
        }
    
    def _get_mime_type(self, path: Path) -> str:
        """获取文件的MIME类型。"""
        ext = path.suffix.lower()
        return self._mime_types.get(ext, "application/octet-stream")
    
    def _is_allowed(self, path: Path) -> bool:
        """检查文件是否允许访问。"""
        # 检查扩展名
        if self.allowed_extensions:
            if path.suffix.lower() not in self.allowed_extensions:
                return False
        
        # 检查文件大小
        try:
            if path.stat().st_size > self.max_file_size:
                return False
        except OSError:
            return False
        
        return True
    
    def _resolve_path(self, uri: str) -> Path:
        """解析URI为文件路径。"""
        # 移除file://前缀
        if uri.startswith("file://"):
            uri = uri[7:]
        
        path = Path(uri)
        
        # 如果是相对路径，相对于base_path
        if not path.is_absolute():
            path = self.base_path / path
        
        # 安全检查：确保路径在base_path内
        try:
            path = path.resolve()
            path.relative_to(self.base_path)
        except ValueError:
            raise ResourceAccessError(f"Access denied: {uri}")
        
        return path
    
    def list_resources(self) -> List[MCPResource]:
        """列出所有文件资源。"""
        resources = []
        
        if not self.base_path.exists():
            return resources
        
        for path in self.base_path.rglob("*"):
            if path.is_file() and self._is_allowed(path):
                try:
                    stat = path.stat()
                    rel_path = path.relative_to(self.base_path)
                    
                    resource = MCPResource(
                        uri=f"file://{rel_path}",
                        name=path.name,
                        description=str(rel_path),
                        mimeType=self._get_mime_type(path),
                        size=stat.st_size
                    )
                    resources.append(resource)
                except OSError:
                    continue
        
        return resources
    
    def read_resource(self, uri: str) -> MCPResourceContent:
        """读取文件内容。"""
        path = self._resolve_path(uri)
        
        if not path.exists():
            raise ResourceNotFoundError(f"File not found: {uri}")
        
        if not path.is_file():
            raise ResourceAccessError(f"Not a file: {uri}")
        
        if not self._is_allowed(path):
            raise ResourceAccessError(f"Access denied: {uri}")
        
        mime_type = self._get_mime_type(path)
        
        # 判断是否为文本文件
        text_extensions = {".txt", ".md", ".json", ".xml", ".html", ".css", 
                          ".js", ".py", ".yaml", ".yml", ".csv", ".svg"}
        
        if path.suffix.lower() in text_extensions:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return MCPResourceContent(
                uri=uri,
                mimeType=mime_type,
                text=content
            )
        else:
            with open(path, "rb") as f:
                content = f.read()
            return MCPResourceContent(
                uri=uri,
                mimeType=mime_type,
                blob=base64.b64encode(content).decode("ascii")
            )
    
    def exists(self, uri: str) -> bool:
        """检查文件是否存在。"""
        try:
            path = self._resolve_path(uri)
            return path.exists() and path.is_file()
        except ResourceAccessError:
            return False
    
    def get_metadata(self, uri: str) -> ResourceMetadata:
        """获取文件元数据。"""
        path = self._resolve_path(uri)
        
        if not path.exists():
            raise ResourceNotFoundError(f"File not found: {uri}")
        
        stat = path.stat()
        
        return ResourceMetadata(
            size=stat.st_size,
            created_at=stat.st_ctime,
            modified_at=stat.st_mtime,
            extra={
                "path": str(path),
                "name": path.name,
                "extension": path.suffix
            }
        )


class MemoryResourceProvider(ResourceProvider):
    """内存资源提供者。
    
    在内存中管理资源，适用于临时数据。
    """
    
    def __init__(self) -> None:
        """初始化内存资源提供者。"""
        self._resources: Dict[str, MCPResource] = {}
        self._contents: Dict[str, Union[str, bytes]] = {}
        self._lock = threading.Lock()
    
    def add_resource(
        self,
        uri: str,
        name: str,
        content: Union[str, bytes],
        description: str = "",
        mime_type: str = "text/plain"
    ) -> None:
        """添加资源。
        
        Args:
            uri: 资源URI
            name: 资源名称
            content: 资源内容
            description: 描述
            mime_type: MIME类型
        """
        with self._lock:
            size = len(content) if isinstance(content, bytes) else len(content.encode("utf-8"))
            
            self._resources[uri] = MCPResource(
                uri=uri,
                name=name,
                description=description,
                mimeType=mime_type,
                size=size
            )
            self._contents[uri] = content
    
    def remove_resource(self, uri: str) -> None:
        """移除资源。
        
        Args:
            uri: 资源URI
        """
        with self._lock:
            self._resources.pop(uri, None)
            self._contents.pop(uri, None)
    
    def update_resource(
        self,
        uri: str,
        content: Union[str, bytes]
    ) -> None:
        """更新资源内容。
        
        Args:
            uri: 资源URI
            content: 新内容
        """
        with self._lock:
            if uri in self._contents:
                self._contents[uri] = content
                # 更新大小
                size = len(content) if isinstance(content, bytes) else len(content.encode("utf-8"))
                resource = self._resources[uri]
                self._resources[uri] = MCPResource(
                    uri=resource.uri,
                    name=resource.name,
                    description=resource.description,
                    mimeType=resource.mimeType,
                    size=size
                )
    
    def list_resources(self) -> List[MCPResource]:
        """列出所有资源。"""
        with self._lock:
            return list(self._resources.values())
    
    def read_resource(self, uri: str) -> MCPResourceContent:
        """读取资源内容。"""
        with self._lock:
            if uri not in self._contents:
                raise ResourceNotFoundError(f"Resource not found: {uri}")
            
            resource = self._resources[uri]
            content = self._contents[uri]
            
            if isinstance(content, str):
                return MCPResourceContent(
                    uri=uri,
                    mimeType=resource.mimeType,
                    text=content
                )
            else:
                return MCPResourceContent(
                    uri=uri,
                    mimeType=resource.mimeType,
                    blob=base64.b64encode(content).decode("ascii")
                )
    
    def exists(self, uri: str) -> bool:
        """检查资源是否存在。"""
        with self._lock:
            return uri in self._contents


class DatabaseResourceProvider(ResourceProvider):
    """数据库资源提供者。
    
    提供对数据库查询结果的访问。
    """
    
    def __init__(
        self,
        connection_factory: Optional[Callable[[], Any]] = None
    ):
        """初始化数据库资源提供者。
        
        Args:
            connection_factory: 数据库连接工厂函数
        """
        self.connection_factory = connection_factory
        self._queries: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def register_query(
        self,
        uri: str,
        name: str,
        query: str,
        description: str = "",
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        """注册查询。
        
        Args:
            uri: 资源URI
            name: 查询名称
            query: SQL查询语句
            description: 描述
            params: 默认参数
        """
        with self._lock:
            self._queries[uri] = {
                "name": name,
                "query": query,
                "description": description,
                "params": params or {}
            }
    
    def unregister_query(self, uri: str) -> None:
        """注销查询。
        
        Args:
            uri: 资源URI
        """
        with self._lock:
            self._queries.pop(uri, None)
    
    def list_resources(self) -> List[MCPResource]:
        """列出所有查询资源。"""
        resources = []
        
        with self._lock:
            for uri, query_info in self._queries.items():
                resource = MCPResource(
                    uri=uri,
                    name=query_info["name"],
                    description=query_info["description"],
                    mimeType="application/json"
                )
                resources.append(resource)
        
        return resources
    
    def read_resource(self, uri: str) -> MCPResourceContent:
        """执行查询并返回结果。"""
        with self._lock:
            if uri not in self._queries:
                raise ResourceNotFoundError(f"Query not found: {uri}")
            
            query_info = self._queries[uri]
        
        if not self.connection_factory:
            raise ResourceAccessError("No database connection configured")
        
        try:
            conn = self.connection_factory()
            cursor = conn.cursor()
            
            cursor.execute(query_info["query"], query_info["params"])
            
            # 获取列名
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            
            # 获取数据
            rows = cursor.fetchall()
            
            # 转换为字典列表
            result = []
            for row in rows:
                result.append(dict(zip(columns, row)))
            
            cursor.close()
            conn.close()
            
            return MCPResourceContent(
                uri=uri,
                mimeType="application/json",
                text=json.dumps(result, ensure_ascii=False, default=str)
            )
        
        except Exception as e:
            raise ResourceAccessError(f"Query execution failed: {str(e)}")
    
    def exists(self, uri: str) -> bool:
        """检查查询是否存在。"""
        with self._lock:
            return uri in self._queries


class ResourceManager:
    """资源管理器。
    
    统一管理多种资源提供者，提供资源访问接口。
    """
    
    def __init__(self) -> None:
        """初始化资源管理器。"""
        self._providers: Dict[str, ResourceProvider] = {}
        self._subscribers: Dict[str, List[Callable[[str], None]]] = {}
        self._lock = threading.Lock()
    
    def register_provider(
        self,
        name: str,
        provider: ResourceProvider
    ) -> None:
        """注册资源提供者。
        
        Args:
            name: 提供者名称
            provider: 资源提供者
        """
        with self._lock:
            self._providers[name] = provider
    
    def unregister_provider(self, name: str) -> None:
        """注销资源提供者。
        
        Args:
            name: 提供者名称
        """
        with self._lock:
            self._providers.pop(name, None)
    
    def get_provider(self, name: str) -> Optional[ResourceProvider]:
        """获取资源提供者。
        
        Args:
            name: 提供者名称
            
        Returns:
            资源提供者
        """
        with self._lock:
            return self._providers.get(name)
    
    def list_resources(
        self,
        provider_name: Optional[str] = None
    ) -> List[MCPResource]:
        """列出资源。
        
        Args:
            provider_name: 指定提供者，为None则列出所有
            
        Returns:
            资源列表
        """
        resources = []
        
        with self._lock:
            if provider_name:
                provider = self._providers.get(provider_name)
                if provider:
                    resources.extend(provider.list_resources())
            else:
                for provider in self._providers.values():
                    try:
                        resources.extend(provider.list_resources())
                    except Exception:
                        continue
        
        return resources
    
    def read_resource(self, uri: str) -> MCPResourceContent:
        """读取资源内容。
        
        Args:
            uri: 资源URI
            
        Returns:
            资源内容
        """
        # 根据URI前缀选择提供者
        provider = self._find_provider(uri)
        
        if not provider:
            raise ResourceNotFoundError(f"No provider for URI: {uri}")
        
        return provider.read_resource(uri)
    
    def _find_provider(self, uri: str) -> Optional[ResourceProvider]:
        """根据URI查找合适的提供者。"""
        # 根据URI前缀匹配
        if uri.startswith("file://"):
            return self._providers.get("file")
        elif uri.startswith("db://"):
            return self._providers.get("database")
        elif uri.startswith("memory://"):
            return self._providers.get("memory")
        
        # 尝试所有提供者
        with self._lock:
            for provider in self._providers.values():
                if provider.exists(uri):
                    return provider
        
        return None
    
    def exists(self, uri: str) -> bool:
        """检查资源是否存在。
        
        Args:
            uri: 资源URI
            
        Returns:
            是否存在
        """
        provider = self._find_provider(uri)
        return provider is not None and provider.exists(uri)
    
    def subscribe(
        self,
        uri: str,
        callback: Callable[[str], None]
    ) -> None:
        """订阅资源更新。
        
        Args:
            uri: 资源URI
            callback: 回调函数
        """
        with self._lock:
            if uri not in self._subscribers:
                self._subscribers[uri] = []
            self._subscribers[uri].append(callback)
    
    def unsubscribe(
        self,
        uri: str,
        callback: Optional[Callable[[str], None]] = None
    ) -> None:
        """取消订阅。
        
        Args:
            uri: 资源URI
            callback: 回调函数，为None则取消所有
        """
        with self._lock:
            if callback is None:
                self._subscribers.pop(uri, None)
            elif uri in self._subscribers:
                try:
                    self._subscribers[uri].remove(callback)
                except ValueError:
                    pass
    
    def notify_update(self, uri: str) -> None:
        """通知资源更新。
        
        Args:
            uri: 资源URI
        """
        with self._lock:
            callbacks = list(self._subscribers.get(uri, []))
        
        for callback in callbacks:
            try:
                callback(uri)
            except Exception:
                pass
    
    def setup_default_providers(
        self,
        file_base_path: Optional[str] = None
    ) -> None:
        """设置默认提供者。
        
        Args:
            file_base_path: 文件提供者的基础路径
        """
        if file_base_path:
            self.register_provider("file", FileResourceProvider(file_base_path))
        
        self.register_provider("memory", MemoryResourceProvider())


__all__ = [
    "ResourceType",
    "ResourceError",
    "ResourceNotFoundError",
    "ResourceAccessError",
    "ResourceMetadata",
    "ResourceProvider",
    "FileResourceProvider",
    "MemoryResourceProvider",
    "DatabaseResourceProvider",
    "ResourceManager",
]
