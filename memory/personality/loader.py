"""
Personality Loader - 统一人格加载器

该模块提供统一的人格配置加载接口，支持多种来源：
- 本地文件加载
- 远程URL加载
- 模板实例化
- 缓存管理
- 热重载支持

使用示例:
    loader = Loader()
    
    # 从文件加载
    config = loader.load_from_file("path/to/personality.md")
    
    # 从URL加载
    config = loader.load_from_url("https://example.com/personality.md")
    
    # 从模板创建
    config = loader.create_from_template("assistant", name="My Assistant")
    
    # 热重载
    loader.enable_hot_reload()
"""

import os
import re
import hashlib
import json
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from enum import Enum
import logging
from dataclasses import dataclass, field

from . import (
    PersonalityConfig, SoulParser, Validator,
    LoadingError, ValidationError, PersonalityTrait, TraitDimension,
    CommunicationStyle, CommunicationTone, ResponseLength
)
from .validator import ValidationLevel

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """缓存条目"""
    config: PersonalityConfig
    loaded_at: datetime
    source: str
    checksum: str
    hit_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """检查是否过期"""
        age = (datetime.now() - self.loaded_at).total_seconds()
        return age > ttl_seconds


@dataclass
class LoadResult:
    """加载结果"""
    config: PersonalityConfig
    source: str
    from_cache: bool
    load_time_ms: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class TemplateInfo:
    """模板信息"""
    name: str
    path: str
    description: str
    category: str
    tags: List[str] = field(default_factory=list)


class CacheStrategy(Enum):
    """缓存策略"""
    NO_CACHE = "no_cache"           # 无缓存
    MEMORY_CACHE = "memory_cache"     # 内存缓存
    DISK_CACHE = "disk_cache"         # 磁盘缓存
    HYBRID_CACHE = "hybrid_cache"     # 混合缓存


class Loader:
    """
    统一人格加载器
    
    支持多种加载方式和缓存策略：
    - 本地文件加载
    - 远程URL加载
    - 内置模板实例化
    - 多级缓存
    - 热重载
    
    Attributes:
        cache_strategy: 缓存策略
        cache_ttl: 缓存TTL（秒）
        cache_dir: 磁盘缓存目录
        enable_hot_reload: 是否启用热重载
    """
    
    def __init__(
        self,
        cache_strategy: CacheStrategy = CacheStrategy.MEMORY_CACHE,
        cache_ttl: int = 3600,
        cache_dir: Optional[str] = None,
        enable_hot_reload: bool = False,
        hot_reload_interval: int = 60
    ):
        """
        初始化加载器
        
        Args:
            cache_strategy: 缓存策略
            cache_ttl: 缓存TTL（秒）
            cache_dir: 磁盘缓存目录
            enable_hot_reload: 是否启用热重载
            hot_reload_interval: 热重载检查间隔（秒）
        """
        self.cache_strategy = cache_strategy
        self.cache_ttl = cache_ttl
        self.cache_dir = cache_dir or self._get_default_cache_dir()
        self.enable_hot_reload = enable_hot_reload
        self.hot_reload_interval = hot_reload_interval
        
        self._parser = SoulParser()
        self._validator = Validator(level=ValidationLevel.STANDARD)
        
        # 内存缓存
        self._memory_cache: Dict[str, CacheEntry] = {}
        
        # 热重载监控
        self._watched_files: Dict[str, float] = {}
        self._hot_reload_callbacks: List[Callable] = []
        self._hot_reload_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # 确保缓存目录存在
        if self.cache_strategy in (CacheStrategy.DISK_CACHE, CacheStrategy.HYBRID_CACHE):
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # 启动热重载线程
        if enable_hot_reload:
            self._start_hot_reload_thread()
    
    def _get_default_cache_dir(self) -> str:
        """获取默认缓存目录"""
        home = os.path.expanduser("~")
        cache_dir = os.path.join(home, ".agi_framework", "personality_cache")
        return cache_dir
    
    def load_from_file(
        self,
        file_path: str,
        use_cache: bool = True,
        validate: bool = True
    ) -> PersonalityConfig:
        """
        从文件加载人格配置
        
        Args:
            file_path: 文件路径
            use_cache: 是否使用缓存
            validate: 是否验证配置
            
        Returns:
            PersonalityConfig对象
            
        Raises:
            LoadingError: 加载错误
        """
        import time
        start_time = time.time()
        
        # 规范化路径
        file_path = os.path.abspath(os.path.expanduser(file_path))
        
        # 检查缓存
        if use_cache and self.cache_strategy != CacheStrategy.NO_CACHE:
            cached = self._get_from_cache(file_path)
            if cached:
                logger.debug(f"Cache hit for {file_path}")
                return cached
        
        # 检查文件存在
        if not os.path.exists(file_path):
            raise LoadingError(f"File not found: {file_path}")
        
        # 读取文件
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            raise LoadingError(f"Error reading file: {e}")
        
        # 解析
        try:
            config = self._parser.parse(content)
            config.metadata.source = file_path
        except Exception as e:
            raise LoadingError(f"Error parsing file: {e}")
        
        # 验证
        if validate:
            result = self._validator.validate(config)
            if not result.is_valid:
                errors = [str(e) for e in result.errors]
                raise ValidationError(f"Validation failed: {'; '.join(errors)}")
        
        # 缓存
        if self.cache_strategy != CacheStrategy.NO_CACHE:
            self._save_to_cache(file_path, config)
        
        # 监控文件变化
        if self.enable_hot_reload:
            self._watch_file(file_path)
        
        logger.info(f"Loaded personality from {file_path} in {time.time() - start_time:.3f}s")
        
        return config
    
    def load_from_url(
        self,
        url: str,
        use_cache: bool = True,
        validate: bool = True,
        timeout: int = 30
    ) -> PersonalityConfig:
        """
        从URL加载人格配置
        
        Args:
            url: URL地址
            use_cache: 是否使用缓存
            validate: 是否验证配置
            timeout: 超时时间（秒）
            
        Returns:
            PersonalityConfig对象
            
        Raises:
            LoadingError: 加载错误
        """
        import time
        import urllib.request
        import urllib.error
        
        start_time = time.time()
        
        # 检查缓存
        if use_cache and self.cache_strategy != CacheStrategy.NO_CACHE:
            cached = self._get_from_cache(url)
            if cached:
                logger.debug(f"Cache hit for {url}")
                return cached
        
        # 验证URL格式
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise LoadingError(f"Invalid URL: {url}")
        
        # 下载内容
        try:
            request = urllib.request.Request(
                url,
                headers={'User-Agent': 'AGI-Personality-Loader/1.0'}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content = response.read().decode('utf-8')
        except urllib.error.HTTPError as e:
            raise LoadingError(f"HTTP error {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            raise LoadingError(f"URL error: {e.reason}")
        except Exception as e:
            raise LoadingError(f"Error downloading: {e}")
        
        # 解析
        try:
            config = self._parser.parse(content)
            config.metadata.source = url
        except Exception as e:
            raise LoadingError(f"Error parsing content: {e}")
        
        # 验证
        if validate:
            result = self._validator.validate(config)
            if not result.is_valid:
                errors = [str(e) for e in result.errors]
                raise ValidationError(f"Validation failed: {'; '.join(errors)}")
        
        # 缓存
        if self.cache_strategy != CacheStrategy.NO_CACHE:
            self._save_to_cache(url, config)
        
        logger.info(f"Loaded personality from {url} in {time.time() - start_time:.3f}s")
        
        return config
    
    def load_from_string(
        self,
        content: str,
        source: Optional[str] = None,
        validate: bool = True
    ) -> PersonalityConfig:
        """
        从字符串加载人格配置
        
        Args:
            content: SOUL.md格式的字符串
            source: 来源标识
            validate: 是否验证
            
        Returns:
            PersonalityConfig对象
        """
        try:
            config = self._parser.parse(content)
            if source:
                config.metadata.source = source
        except Exception as e:
            raise LoadingError(f"Error parsing content: {e}")
        
        if validate:
            result = self._validator.validate(config)
            if not result.is_valid:
                errors = [str(e) for e in result.errors]
                raise ValidationError(f"Validation failed: {'; '.join(errors)}")
        
        return config
    
    def load_from_dict(
        self,
        data: Dict[str, Any],
        source: Optional[str] = None,
        validate: bool = True
    ) -> PersonalityConfig:
        """
        从字典加载人格配置
        
        Args:
            data: 配置字典
            source: 来源标识
            validate: 是否验证
            
        Returns:
            PersonalityConfig对象
        """
        try:
            config = PersonalityConfig.from_dict(data)
            if source:
                config.metadata.source = source
        except Exception as e:
            raise LoadingError(f"Error creating config from dict: {e}")
        
        if validate:
            result = self._validator.validate(config)
            if not result.is_valid:
                errors = [str(e) for e in result.errors]
                raise ValidationError(f"Validation failed: {'; '.join(errors)}")
        
        return config
    
    def create_from_template(
        self,
        template_name: str,
        name: Optional[str] = None,
        **overrides
    ) -> PersonalityConfig:
        """
        从内置模板创建人格配置
        
        Args:
            template_name: 模板名称
            name: 自定义名称
            **overrides: 配置覆盖项
            
        Returns:
            PersonalityConfig对象
            
        Raises:
            LoadingError: 模板不存在
        """
        # 获取模板内容
        template_path = self._get_template_path(template_name)
        
        if not template_path or not os.path.exists(template_path):
            raise LoadingError(f"Template not found: {template_name}")
        
        # 加载模板
        config = self.load_from_file(template_path, use_cache=False)
        
        # 应用覆盖
        if name:
            config.name = name
        
        # 应用其他覆盖
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # 更新版本
        config.version = self._bump_patch_version(config.version)
        
        return config
    
    def _get_template_path(self, template_name: str) -> Optional[str]:
        """获取模板文件路径"""
        # 首先检查工作目录下的templates
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        template_dir = os.path.join(base_dir, "memory", "personality", "templates")
        
        possible_paths = [
            os.path.join(template_dir, f"{template_name}.md"),
            os.path.join(template_dir, f"{template_name}.soul.md"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def list_templates(self) -> List[TemplateInfo]:
        """
        列出所有可用的模板
        
        Returns:
            模板信息列表
        """
        templates = []
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        template_dir = os.path.join(base_dir, "memory", "personality", "templates")
        
        if not os.path.exists(template_dir):
            return templates
        
        for filename in os.listdir(template_dir):
            if filename.endswith(('.md', '.soul.md')):
                path = os.path.join(template_dir, filename)
                name = os.path.splitext(filename)[0]
                
                templates.append(TemplateInfo(
                    name=name,
                    path=path,
                    description=f"Template: {name}",
                    category="general",
                    tags=[name]
                ))
        
        return templates
    
    def _get_from_cache(self, key: str) -> Optional[PersonalityConfig]:
        """从缓存获取"""
        entry = self._memory_cache.get(key)
        
        if entry:
            if entry.is_expired(self.cache_ttl):
                del self._memory_cache[key]
            else:
                entry.hit_count += 1
                entry.last_accessed = datetime.now()
                return entry.config
        
        # 尝试磁盘缓存
        if self.cache_strategy in (CacheStrategy.DISK_CACHE, CacheStrategy.HYBRID_CACHE):
            disk_cached = self._get_from_disk_cache(key)
            if disk_cached:
                self._memory_cache[key] = CacheEntry(
                    config=disk_cached,
                    loaded_at=datetime.now(),
                    source=key,
                    checksum=""
                )
                return disk_cached
        
        return None
    
    def _save_to_cache(self, key: str, config: PersonalityConfig) -> None:
        """保存到缓存"""
        checksum = config.get_fingerprint()
        
        entry = CacheEntry(
            config=config,
            loaded_at=datetime.now(),
            source=key,
            checksum=checksum
        )
        
        self._memory_cache[key] = entry
        
        # 磁盘缓存
        if self.cache_strategy in (CacheStrategy.DISK_CACHE, CacheStrategy.HYBRID_CACHE):
            self._save_to_disk_cache(key, config)
    
    def _get_from_disk_cache(self, key: str) -> Optional[PersonalityConfig]:
        """从磁盘缓存获取"""
        cache_key = hashlib.md5(key.encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return PersonalityConfig.from_dict(data)
        except Exception as e:
            logger.warning(f"Error reading disk cache: {e}")
            return None
    
    def _save_to_disk_cache(self, key: str, config: PersonalityConfig) -> None:
        """保存到磁盘缓存"""
        cache_key = hashlib.md5(key.encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Error writing disk cache: {e}")
    
    def _watch_file(self, file_path: str) -> None:
        """监控文件变化"""
        try:
            mtime = os.path.getmtime(file_path)
            self._watched_files[file_path] = mtime
        except Exception as e:
            logger.warning(f"Error watching file {file_path}: {e}")
    
    def _start_hot_reload_thread(self) -> None:
        """启动热重载监控线程"""
        def hot_reload_loop():
            while not self._stop_event.is_set():
                try:
                    self._check_for_changes()
                except Exception as e:
                    logger.warning(f"Error in hot reload: {e}")
                
                self._stop_event.wait(self.hot_reload_interval)
        
        self._hot_reload_thread = threading.Thread(
            target=hot_reload_loop,
            daemon=True,
            name="PersonalityHotReload"
        )
        self._hot_reload_thread.start()
        logger.info("Hot reload thread started")
    
    def _check_for_changes(self) -> None:
        """检查文件变化"""
        for file_path, last_mtime in list(self._watched_files.items()):
            if not os.path.exists(file_path):
                continue
            
            try:
                current_mtime = os.path.getmtime(file_path)
                if current_mtime > last_mtime:
                    logger.info(f"Detected change in {file_path}")
                    
                    # 清除缓存
                    if file_path in self._memory_cache:
                        del self._memory_cache[file_path]
                    
                    # 触发回调
                    for callback in self._hot_reload_callbacks:
                        try:
                            callback(file_path)
                        except Exception as e:
                            logger.warning(f"Error in hot reload callback: {e}")
                    
                    # 更新mtime
                    self._watched_files[file_path] = current_mtime
                    
            except Exception as e:
                logger.warning(f"Error checking file {file_path}: {e}")
    
    def register_hot_reload_callback(self, callback: Callable[[str], None]) -> None:
        """
        注册热重载回调
        
        Args:
            callback: 回调函数，接收文件路径参数
        """
        self._hot_reload_callbacks.append(callback)
    
    def clear_cache(self, key: Optional[str] = None) -> None:
        """
        清除缓存
        
        Args:
            key: 可选的缓存键，为None则清除所有
        """
        if key:
            if key in self._memory_cache:
                del self._memory_cache[key]
        else:
            self._memory_cache.clear()
        
        # 清除磁盘缓存
        if self.cache_strategy in (CacheStrategy.DISK_CACHE, CacheStrategy.HYBRID_CACHE):
            if os.path.exists(self.cache_dir):
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.json'):
                        try:
                            os.remove(os.path.join(self.cache_dir, filename))
                        except Exception:
                            pass
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_hits = sum(e.hit_count for e in self._memory_cache.values())
        total_entries = len(self._memory_cache)
        
        return {
            "strategy": self.cache_strategy.value,
            "memory_entries": total_entries,
            "total_hits": total_hits,
            "cache_dir": self.cache_dir,
            "watched_files": len(self._watched_files),
            "hot_reload_enabled": self.enable_hot_reload
        }
    
    def enable_hot_reload(self) -> None:
        """启用热重载"""
        if not self.enable_hot_reload:
            self.enable_hot_reload = True
            self._start_hot_reload_thread()
    
    def disable_hot_reload(self) -> None:
        """禁用热重载"""
        self.enable_hot_reload = False
        self._stop_event.set()
        
        if self._hot_reload_thread:
            self._hot_reload_thread.join(timeout=5)
            self._hot_reload_thread = None
    
    def shutdown(self) -> None:
        """关闭加载器"""
        self.disable_hot_reload()
        self.clear_cache()
        logger.info("Loader shutdown complete")
    
    def _bump_patch_version(self, version: str) -> str:
        """增加补丁版本"""
        parts = version.split('.')
        if len(parts) == 3:
            parts[2] = str(int(parts[2]) + 1)
        return '.'.join(parts)


class PersonalityRegistry:
    """
    人格配置注册表
    
    提供人格配置的注册、查找和管理功能。
    """
    
    def __init__(self, loader: Optional[Loader] = None):
        """
        初始化注册表
        
        Args:
            loader: 可选的Loader实例
        """
        self._loader = loader or Loader()
        self._registry: Dict[str, PersonalityConfig] = {}
        self._aliases: Dict[str, str] = {}
    
    def register(
        self,
        config: PersonalityConfig,
        alias: Optional[str] = None
    ) -> None:
        """
        注册人格配置
        
        Args:
            config: 人格配置
            alias: 可选的别名
        """
        key = config.name.lower().replace(' ', '_')
        self._registry[key] = config
        
        if alias:
            self._aliases[alias.lower()] = key
        
        logger.debug(f"Registered personality: {config.name}")
    
    def get(self, name: str) -> Optional[PersonalityConfig]:
        """
        获取人格配置
        
        Args:
            name: 名称或别名
            
        Returns:
            PersonalityConfig或None
        """
        name_lower = name.lower().replace(' ', '_')
        
        # 直接查找
        if name_lower in self._registry:
            return self._registry[name_lower]
        
        # 查找别名
        if name_lower in self._aliases:
            key = self._aliases[name_lower]
            return self._registry.get(key)
        
        return None
    
    def list_all(self) -> List[str]:
        """列出所有注册的名称"""
        return list(self._registry.keys())
    
    def unregister(self, name: str) -> bool:
        """
        注销人格配置
        
        Args:
            name: 名称
            
        Returns:
            是否成功
        """
        name_lower = name.lower().replace(' ', '_')
        
        if name_lower in self._registry:
            del self._registry[name_lower]
            
            # 清理别名
            to_remove = [k for k, v in self._aliases.items() if v == name_lower]
            for k in to_remove:
                del self._aliases[k]
            
            return True
        
        return False


def create_loader(
    cache_strategy: CacheStrategy = CacheStrategy.MEMORY_CACHE,
    enable_hot_reload: bool = False
) -> Loader:
    """
    工厂函数：创建加载器
    
    Args:
        cache_strategy: 缓存策略
        enable_hot_reload: 是否启用热重载
        
    Returns:
        Loader实例
    """
    return Loader(
        cache_strategy=cache_strategy,
        enable_hot_reload=enable_hot_reload
    )
