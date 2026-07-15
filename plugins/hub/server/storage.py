"""
插件存储模块

提供文件存储、CDN分发、版本管理和备份策略功能。
"""

import hashlib
import json
import os
import shutil
import tarfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, BinaryIO, Callable
from dataclasses import dataclass, field, asdict
import threading


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

@dataclass
class StoredFile:
    """存储的文件"""
    file_id: str
    filename: str
    path: str
    size: int
    checksum: str
    content_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VersionInfo:
    """版本信息"""
    version: str
    file_id: str
    changelog: str = ""
    release_date: float = field(default_factory=lambda: datetime.now().timestamp())
    min_platform_version: str = ""
    deprecated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 文件存储
# ---------------------------------------------------------------------------

class FileStorage:
    """文件存储"""
    
    def __init__(self, base_path: str):
        """
        Args:
            base_path: 基础存储路径
        """
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._files: Dict[str, StoredFile] = {}
        self._lock = threading.RLock()
        
        # 加载现有文件索引
        self._load_index()
    
    def store(self, file_id: str, data: bytes,
              filename: str = "",
              content_type: str = "",
              metadata: Optional[Dict[str, Any]] = None) -> StoredFile:
        """存储文件
        
        Args:
            file_id: 文件ID
            data: 文件数据
            filename: 原始文件名
            content_type: 内容类型
            metadata: 元数据
            
        Returns:
            存储的文件信息
        """
        with self._lock:
            # 计算校验和
            checksum = hashlib.sha256(data).hexdigest()
            
            # 确定存储路径
            subdir = file_id[:2]
            storage_dir = self._base_path / subdir
            storage_dir.mkdir(exist_ok=True)
            
            file_path = storage_dir / file_id
            
            # 写入文件
            with open(file_path, 'wb') as f:
                f.write(data)
            
            # 创建记录
            stored = StoredFile(
                file_id=file_id,
                filename=filename or file_id,
                path=str(file_path),
                size=len(data),
                checksum=checksum,
                content_type=content_type,
                metadata=metadata or {},
            )
            
            self._files[file_id] = stored
            self._save_index()
            
            return stored
    
    def store_file(self, file_id: str, source_path: str,
                   metadata: Optional[Dict[str, Any]] = None) -> StoredFile:
        """从文件路径存储
        
        Args:
            file_id: 文件ID
            source_path: 源文件路径
            metadata: 元数据
            
        Returns:
            存储的文件信息
        """
        with open(source_path, 'rb') as f:
            data = f.read()
        
        filename = os.path.basename(source_path)
        
        # 猜测内容类型
        content_type = self._guess_content_type(filename)
        
        return self.store(file_id, data, filename, content_type, metadata)
    
    def retrieve(self, file_id: str) -> Optional[bytes]:
        """检索文件
        
        Args:
            file_id: 文件ID
            
        Returns:
            文件数据，不存在返回None
        """
        with self._lock:
            stored = self._files.get(file_id)
            if not stored:
                return None
            
            try:
                with open(stored.path, 'rb') as f:
                    return f.read()
            except FileNotFoundError:
                return None
    
    def retrieve_to_file(self, file_id: str, dest_path: str) -> bool:
        """检索文件到指定路径
        
        Args:
            file_id: 文件ID
            dest_path: 目标路径
            
        Returns:
            是否成功
        """
        data = self.retrieve(file_id)
        if data is None:
            return False
        
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        with open(dest_path, 'wb') as f:
            f.write(data)
        
        return True
    
    def delete(self, file_id: str) -> bool:
        """删除文件
        
        Args:
            file_id: 文件ID
            
        Returns:
            是否成功
        """
        with self._lock:
            stored = self._files.get(file_id)
            if not stored:
                return False
            
            try:
                os.remove(stored.path)
            except FileNotFoundError:
                pass
            
            del self._files[file_id]
            self._save_index()
            
            return True
    
    def exists(self, file_id: str) -> bool:
        """检查文件是否存在"""
        with self._lock:
            return file_id in self._files
    
    def get_info(self, file_id: str) -> Optional[StoredFile]:
        """获取文件信息"""
        with self._lock:
            return self._files.get(file_id)
    
    def verify_checksum(self, file_id: str) -> bool:
        """验证文件校验和"""
        with self._lock:
            stored = self._files.get(file_id)
            if not stored:
                return False
            
            try:
                with open(stored.path, 'rb') as f:
                    data = f.read()
                
                actual_checksum = hashlib.sha256(data).hexdigest()
                return hmac.compare_digest(actual_checksum, stored.checksum)
            except FileNotFoundError:
                return False
    
    def list_files(self, prefix: Optional[str] = None) -> List[StoredFile]:
        """列出文件"""
        with self._lock:
            files = list(self._files.values())
            
            if prefix:
                files = [f for f in files if f.file_id.startswith(prefix)]
            
            return files
    
    def get_total_size(self) -> int:
        """获取总存储大小"""
        with self._lock:
            return sum(f.size for f in self._files.values())
    
    def _guess_content_type(self, filename: str) -> str:
        """猜测内容类型"""
        ext = os.path.splitext(filename)[1].lower()
        
        content_types = {
            '.zip': 'application/zip',
            '.tar': 'application/x-tar',
            '.gz': 'application/gzip',
            '.json': 'application/json',
            '.py': 'text/x-python',
            '.txt': 'text/plain',
            '.md': 'text/markdown',
        }
        
        return content_types.get(ext, 'application/octet-stream')
    
    def _load_index(self) -> None:
        """加载索引"""
        index_path = self._base_path / '.index.json'
        
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    data = json.load(f)
                
                for file_id, file_data in data.items():
                    self._files[file_id] = StoredFile(**file_data)
            except Exception:
                pass
    
    def _save_index(self) -> None:
        """保存索引"""
        index_path = self._base_path / '.index.json'
        
        data = {fid: f.to_dict() for fid, f in self._files.items()}
        
        with open(index_path, 'w') as f:
            json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# CDN存储
# ---------------------------------------------------------------------------

class CDNStorage:
    """CDN存储"""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Args:
            base_url: CDN基础URL
            api_key: API密钥
        """
        self._base_url = base_url.rstrip('/')
        self._api_key = api_key
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def get_url(self, file_id: str, expires_in: Optional[int] = None) -> str:
        """获取文件URL
        
        Args:
            file_id: 文件ID
            expires_in: 过期时间（秒）
            
        Returns:
            文件URL
        """
        url = f"{self._base_url}/{file_id}"
        
        if expires_in and self._api_key:
            # 生成签名URL
            expires = int(datetime.now().timestamp()) + expires_in
            signature = self._generate_signature(file_id, expires)
            url += f"?expires={expires}&signature={signature}"
        
        return url
    
    def invalidate(self, file_id: str) -> bool:
        """使缓存失效
        
        Args:
            file_id: 文件ID
            
        Returns:
            是否成功
        """
        # 实际实现应调用CDN API
        with self._lock:
            if file_id in self._cache:
                del self._cache[file_id]
        
        return True
    
    def invalidate_prefix(self, prefix: str) -> int:
        """使前缀缓存失效
        
        Args:
            prefix: 前缀
            
        Returns:
            失效数量
        """
        with self._lock:
            to_remove = [
                fid for fid in self._cache
                if fid.startswith(prefix)
            ]
            
            for fid in to_remove:
                del self._cache[fid]
            
            return len(to_remove)
    
    def _generate_signature(self, file_id: str, expires: int) -> str:
        """生成签名"""
        if not self._api_key:
            return ""
        
        data = f"{file_id}:{expires}"
        return hmac.new(
            self._api_key.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 版本存储
# ---------------------------------------------------------------------------

class VersionStorage:
    """版本存储"""
    
    def __init__(self, file_storage: FileStorage):
        """
        Args:
            file_storage: 文件存储实例
        """
        self._file_storage = file_storage
        self._versions: Dict[str, Dict[str, VersionInfo]] = {}
        self._lock = threading.Lock()
    
    def store_version(self, plugin_id: str, version: str,
                      file_data: bytes,
                      changelog: str = "",
                      min_platform_version: str = "") -> VersionInfo:
        """存储版本
        
        Args:
            plugin_id: 插件ID
            version: 版本号
            file_data: 文件数据
            changelog: 更新日志
            min_platform_version: 最低平台版本
            
        Returns:
            版本信息
        """
        with self._lock:
            file_id = f"{plugin_id}@{version}"
            
            # 存储文件
            stored = self._file_storage.store(
                file_id=file_id,
                data=file_data,
                filename=f"{plugin_id}-{version}.zip",
                content_type='application/zip',
                metadata={
                    'plugin_id': plugin_id,
                    'version': version,
                }
            )
            
            # 创建版本信息
            version_info = VersionInfo(
                version=version,
                file_id=file_id,
                changelog=changelog,
                min_platform_version=min_platform_version,
            )
            
            if plugin_id not in self._versions:
                self._versions[plugin_id] = {}
            
            self._versions[plugin_id][version] = version_info
            
            return version_info
    
    def get_version(self, plugin_id: str, version: str) -> Optional[VersionInfo]:
        """获取版本信息"""
        with self._lock:
            return self._versions.get(plugin_id, {}).get(version)
    
    def get_version_data(self, plugin_id: str, version: str) -> Optional[bytes]:
        """获取版本数据"""
        with self._lock:
            version_info = self._versions.get(plugin_id, {}).get(version)
            if not version_info:
                return None
            
            return self._file_storage.retrieve(version_info.file_id)
    
    def list_versions(self, plugin_id: str) -> List[VersionInfo]:
        """列出所有版本"""
        with self._lock:
            versions = self._versions.get(plugin_id, {})
            return list(versions.values())
    
    def delete_version(self, plugin_id: str, version: str) -> bool:
        """删除版本"""
        with self._lock:
            if plugin_id not in self._versions:
                return False
            
            version_info = self._versions[plugin_id].get(version)
            if not version_info:
                return False
            
            # 删除文件
            self._file_storage.delete(version_info.file_id)
            
            # 删除版本记录
            del self._versions[plugin_id][version]
            
            return True
    
    def deprecate_version(self, plugin_id: str, version: str) -> bool:
        """弃用版本"""
        with self._lock:
            version_info = self._versions.get(plugin_id, {}).get(version)
            if not version_info:
                return False
            
            version_info.deprecated = True
            return True


# ---------------------------------------------------------------------------
# 备份管理器
# ---------------------------------------------------------------------------

class BackupManager:
    """备份管理器"""
    
    def __init__(self, storage_path: str, max_backups: int = 10):
        """
        Args:
            storage_path: 备份存储路径
            max_backups: 最大备份数量
        """
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._max_backups = max_backups
        self._backups: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
        self._load_backup_index()
    
    def create_backup(self, name: str, source_paths: List[str],
                      metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """创建备份
        
        Args:
            name: 备份名称
            source_paths: 源路径列表
            metadata: 元数据
            
        Returns:
            备份信息
        """
        with self._lock:
            backup_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_path = self._storage_path / f"{backup_id}.tar.gz"
            
            # 创建压缩包
            with tarfile.open(backup_path, 'w:gz') as tar:
                for source_path in source_paths:
                    if os.path.exists(source_path):
                        tar.add(source_path, arcname=os.path.basename(source_path))
            
            # 创建备份记录
            backup_info = {
                'id': backup_id,
                'name': name,
                'path': str(backup_path),
                'size': backup_path.stat().st_size,
                'created_at': datetime.now().isoformat(),
                'sources': source_paths,
                'metadata': metadata or {},
            }
            
            self._backups.append(backup_info)
            
            # 清理旧备份
            self._cleanup_old_backups()
            
            self._save_backup_index()
            
            return backup_info
    
    def restore_backup(self, backup_id: str, dest_path: str) -> bool:
        """恢复备份
        
        Args:
            backup_id: 备份ID
            dest_path: 目标路径
            
        Returns:
            是否成功
        """
        with self._lock:
            backup = None
            for b in self._backups:
                if b['id'] == backup_id:
                    backup = b
                    break
            
            if not backup:
                return False
            
            backup_path = Path(backup['path'])
            if not backup_path.exists():
                return False
            
            try:
                os.makedirs(dest_path, exist_ok=True)
                
                with tarfile.open(backup_path, 'r:gz') as tar:
                    tar.extractall(dest_path)
                
                return True
            except Exception:
                return False
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """列出备份"""
        with self._lock:
            return self._backups.copy()
    
    def delete_backup(self, backup_id: str) -> bool:
        """删除备份"""
        with self._lock:
            for i, backup in enumerate(self._backups):
                if backup['id'] == backup_id:
                    try:
                        os.remove(backup['path'])
                    except FileNotFoundError:
                        pass
                    
                    self._backups.pop(i)
                    self._save_backup_index()
                    
                    return True
            
            return False
    
    def _cleanup_old_backups(self) -> None:
        """清理旧备份"""
        while len(self._backups) > self._max_backups:
            oldest = min(self._backups, key=lambda b: b['created_at'])
            self.delete_backup(oldest['id'])
    
    def _load_backup_index(self) -> None:
        """加载备份索引"""
        index_path = self._storage_path / '.backups.json'
        
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    self._backups = json.load(f)
            except Exception:
                pass
    
    def _save_backup_index(self) -> None:
        """保存备份索引"""
        index_path = self._storage_path / '.backups.json'
        
        with open(index_path, 'w') as f:
            json.dump(self._backups, f, indent=2)


# ---------------------------------------------------------------------------
# 插件存储
# ---------------------------------------------------------------------------

class PluginStorage:
    """插件存储
    
    整合所有存储功能的主类。
    """
    
    def __init__(self, base_path: str, cdn_url: Optional[str] = None):
        """
        Args:
            base_path: 基础存储路径
            cdn_url: CDN URL（可选）
        """
        self._base_path = base_path
        
        self._file_storage = FileStorage(os.path.join(base_path, 'files'))
        self._version_storage = VersionStorage(self._file_storage)
        self._backup_manager = BackupManager(os.path.join(base_path, 'backups'))
        
        if cdn_url:
            self._cdn = CDNStorage(cdn_url)
        else:
            self._cdn = None
    
    def store_plugin(self, plugin_id: str, version: str,
                     file_data: bytes,
                     changelog: str = "") -> VersionInfo:
        """存储插件"""
        return self._version_storage.store_version(
            plugin_id=plugin_id,
            version=version,
            file_data=file_data,
            changelog=changelog,
        )
    
    def get_plugin(self, plugin_id: str, version: str) -> Optional[bytes]:
        """获取插件"""
        return self._version_storage.get_version_data(plugin_id, version)
    
    def get_download_url(self, plugin_id: str, version: str,
                         expires_in: Optional[int] = None) -> Optional[str]:
        """获取下载URL"""
        version_info = self._version_storage.get_version(plugin_id, version)
        if not version_info:
            return None
        
        if self._cdn:
            return self._cdn.get_url(version_info.file_id, expires_in)
        else:
            return f"file://{self._file_storage.get_info(version_info.file_id).path}"
    
    def create_backup(self, name: str) -> Dict[str, Any]:
        """创建备份"""
        return self._backup_manager.create_backup(
            name=name,
            source_paths=[self._base_path],
        )
    
    @property
    def files(self) -> FileStorage:
        """获取文件存储"""
        return self._file_storage
    
    @property
    def versions(self) -> VersionStorage:
        """获取版本存储"""
        return self._version_storage
    
    @property
    def backups(self) -> BackupManager:
        """获取备份管理器"""
        return self._backup_manager
    
    @property
    def cdn(self) -> Optional[CDNStorage]:
        """获取CDN存储"""
        return self._cdn
