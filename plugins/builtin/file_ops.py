"""
文件操作插件

提供文件读写、压缩解压和格式转换功能。
"""

import os
import shutil
import zipfile
import tarfile
import gzip
import json
import hashlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, BinaryIO
import threading


class CompressionFormat(Enum):
    """压缩格式"""
    ZIP = "zip"
    TAR = "tar"
    TAR_GZ = "tar.gz"
    GZIP = "gz"


@dataclass
class FileInfo:
    """文件信息"""
    path: str
    name: str
    size: int
    is_dir: bool
    modified: datetime
    checksum: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'path': self.path,
            'name': self.name,
            'size': self.size,
            'is_dir': self.is_dir,
            'modified': self.modified.isoformat(),
            'checksum': self.checksum,
        }


class FileOperationsPlugin:
    """文件操作插件
    
    提供文件读写、压缩解压和格式转换功能。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置字典
        """
        self._config = config or {}
        self._base_path = self._config.get('base_path', '/tmp')
        self._lock = threading.RLock()
    
    def read_file(self, path: str, encoding: str = 'utf-8') -> str:
        """读取文本文件
        
        Args:
            path: 文件路径
            encoding: 编码
            
        Returns:
            文件内容
        """
        full_path = self._resolve_path(path)
        
        with open(full_path, 'r', encoding=encoding) as f:
            return f.read()
    
    def read_binary(self, path: str) -> bytes:
        """读取二进制文件
        
        Args:
            path: 文件路径
            
        Returns:
            文件内容
        """
        full_path = self._resolve_path(path)
        
        with open(full_path, 'rb') as f:
            return f.read()
    
    def write_file(self, path: str, content: str,
                   encoding: str = 'utf-8') -> bool:
        """写入文本文件
        
        Args:
            path: 文件路径
            content: 内容
            encoding: 编码
            
        Returns:
            是否成功
        """
        full_path = self._resolve_path(path)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, 'w', encoding=encoding) as f:
            f.write(content)
        
        return True
    
    def write_binary(self, path: str, content: bytes) -> bool:
        """写入二进制文件
        
        Args:
            path: 文件路径
            content: 内容
            
        Returns:
            是否成功
        """
        full_path = self._resolve_path(path)
        
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, 'wb') as f:
            f.write(content)
        
        return True
    
    def delete(self, path: str) -> bool:
        """删除文件或目录
        
        Args:
            path: 路径
            
        Returns:
            是否成功
        """
        full_path = self._resolve_path(path)
        
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
        else:
            os.remove(full_path)
        
        return True
    
    def list_dir(self, path: str = '.') -> List[FileInfo]:
        """列出目录内容
        
        Args:
            path: 目录路径
            
        Returns:
            文件信息列表
        """
        full_path = self._resolve_path(path)
        
        results = []
        
        for entry in os.listdir(full_path):
            entry_path = os.path.join(full_path, entry)
            stat = os.stat(entry_path)
            
            info = FileInfo(
                path=entry_path,
                name=entry,
                size=stat.st_size,
                is_dir=os.path.isdir(entry_path),
                modified=datetime.fromtimestamp(stat.st_mtime),
            )
            
            results.append(info)
        
        return results
    
    def get_info(self, path: str) -> FileInfo:
        """获取文件信息
        
        Args:
            path: 路径
            
        Returns:
            文件信息
        """
        full_path = self._resolve_path(path)
        stat = os.stat(full_path)
        
        return FileInfo(
            path=full_path,
            name=os.path.basename(full_path),
            size=stat.st_size,
            is_dir=os.path.isdir(full_path),
            modified=datetime.fromtimestamp(stat.st_mtime),
        )
    
    def compress(self, source: str, dest: str,
                 format: CompressionFormat = CompressionFormat.ZIP) -> bool:
        """压缩文件或目录
        
        Args:
            source: 源路径
            dest: 目标路径
            format: 压缩格式
            
        Returns:
            是否成功
        """
        source_path = self._resolve_path(source)
        dest_path = self._resolve_path(dest)
        
        if format == CompressionFormat.ZIP:
            return self._compress_zip(source_path, dest_path)
        elif format == CompressionFormat.TAR:
            return self._compress_tar(source_path, dest_path)
        elif format == CompressionFormat.TAR_GZ:
            return self._compress_tar(source_path, dest_path, gzip=True)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def decompress(self, source: str, dest: str) -> bool:
        """解压文件
        
        Args:
            source: 源路径
            dest: 目标路径
            
        Returns:
            是否成功
        """
        source_path = self._resolve_path(source)
        dest_path = self._resolve_path(dest)
        
        if source_path.endswith('.zip'):
            return self._decompress_zip(source_path, dest_path)
        elif source_path.endswith('.tar'):
            return self._decompress_tar(source_path, dest_path)
        elif source_path.endswith('.tar.gz') or source_path.endswith('.tgz'):
            return self._decompress_tar(source_path, dest_path, gzip=True)
        else:
            raise ValueError(f"Unknown archive format: {source}")
    
    def _compress_zip(self, source: str, dest: str) -> bool:
        """ZIP压缩"""
        with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as zf:
            if os.path.isdir(source):
                for root, _, files in os.walk(source):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, source)
                        zf.write(file_path, arcname)
            else:
                zf.write(source, os.path.basename(source))
        
        return True
    
    def _compress_tar(self, source: str, dest: str,
                      gzip: bool = False) -> bool:
        """TAR压缩"""
        mode = 'w:gz' if gzip else 'w'
        
        with tarfile.open(dest, mode) as tf:
            tf.add(source, arcname=os.path.basename(source))
        
        return True
    
    def _decompress_zip(self, source: str, dest: str) -> bool:
        """ZIP解压"""
        os.makedirs(dest, exist_ok=True)
        
        with zipfile.ZipFile(source, 'r') as zf:
            zf.extractall(dest)
        
        return True
    
    def _decompress_tar(self, source: str, dest: str,
                        gzip: bool = False) -> bool:
        """TAR解压"""
        os.makedirs(dest, exist_ok=True)
        
        mode = 'r:gz' if gzip else 'r'
        
        with tarfile.open(source, mode) as tf:
            tf.extractall(dest)
        
        return True
    
    def copy(self, source: str, dest: str) -> bool:
        """复制文件或目录
        
        Args:
            source: 源路径
            dest: 目标路径
            
        Returns:
            是否成功
        """
        source_path = self._resolve_path(source)
        dest_path = self._resolve_path(dest)
        
        if os.path.isdir(source_path):
            shutil.copytree(source_path, dest_path)
        else:
            shutil.copy2(source_path, dest_path)
        
        return True
    
    def move(self, source: str, dest: str) -> bool:
        """移动文件或目录
        
        Args:
            source: 源路径
            dest: 目标路径
            
        Returns:
            是否成功
        """
        source_path = self._resolve_path(source)
        dest_path = self._resolve_path(dest)
        
        shutil.move(source_path, dest_path)
        
        return True
    
    def calculate_checksum(self, path: str,
                           algorithm: str = 'sha256') -> str:
        """计算文件校验和
        
        Args:
            path: 文件路径
            algorithm: 算法 (md5, sha1, sha256)
            
        Returns:
            校验和
        """
        full_path = self._resolve_path(path)
        
        hash_obj = hashlib.new(algorithm)
        
        with open(full_path, 'rb') as f:
            while chunk := f.read(8192):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    def _resolve_path(self, path: str) -> str:
        """解析路径"""
        if os.path.isabs(path):
            return path
        return os.path.join(self._base_path, path)
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取插件元数据"""
        return {
            'name': 'file_ops',
            'version': '1.0.0',
            'description': 'File operations plugin with compression support',
            'formats': [f.value for f in CompressionFormat],
        }
