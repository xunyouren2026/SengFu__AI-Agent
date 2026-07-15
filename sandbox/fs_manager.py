"""
文件系统管理器
临时目录创建、权限控制、文件系统隔离
"""

import os
import tempfile
import shutil
import stat
import pwd
import grp
import json
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import time


class FileSystemType(Enum):
    """文件系统类型"""
    TMPFS = "tmpfs"
    PROC = "proc"
    DEVPTS = "devpts"
    SYSFS = "sysfs"
    BIND = "bind"
    OVERLAY = "overlay"


@dataclass
class MountPoint:
    """挂载点配置"""
    source: str                           # 源路径
    target: str                           # 目标路径
    fs_type: FileSystemType = FileSystemType.BIND  # 文件系统类型
    options: List[str] = field(default_factory=list)  # 挂载选项
    read_only: bool = False               # 只读
    private: bool = True                  # 私有挂载
    
    def to_mount_command(self) -> List[str]:
        """转换为mount命令参数"""
        opts = []
        if self.read_only:
            opts.append("ro")
        if self.private:
            opts.append("private")
        opts.extend(self.options)
        
        return [
            "mount",
            "-t", self.fs_type.value,
            "-o", ",".join(opts) if opts else "defaults",
            self.source,
            self.target
        ]


@dataclass
class FilePermission:
    """文件权限"""
    mode: int = 0o755                     # 权限模式
    uid: Optional[int] = None             # 用户ID
    gid: Optional[int] = None             # 组ID
    
    @classmethod
    def from_mode(cls, mode: Union[int, str]) -> 'FilePermission':
        """从模式创建"""
        if isinstance(mode, str):
            mode = int(mode, 8)
        return cls(mode=mode)
    
    @classmethod
    def readonly(cls) -> 'FilePermission':
        """只读权限"""
        return cls(mode=0o444)
    
    @classmethod
    def readwrite(cls) -> 'FilePermission':
        """读写权限"""
        return cls(mode=0o644)
    
    @classmethod
    def executable(cls) -> 'FilePermission':
        """可执行权限"""
        return cls(mode=0o755)
    
    @classmethod
    def private(cls) -> 'FilePermission':
        """私有权限"""
        return cls(mode=0o700)


class FileSystemManager:
    """
    文件系统管理器
    管理临时目录、权限控制、文件系统隔离
    """
    
    def __init__(
        self,
        base_dir: Optional[str] = None,
        prefix: str = "sandbox_fs_"
    ):
        """
        初始化文件系统管理器
        
        Args:
            base_dir: 基础目录
            prefix: 临时目录前缀
        """
        self.base_dir = base_dir or tempfile.gettempdir()
        self.prefix = prefix
        self._managed_dirs: Dict[str, str] = {}
        self._mount_points: Dict[str, List[MountPoint]] = {}
    
    def create_temp_directory(
        self,
        name: Optional[str] = None,
        permission: Optional[FilePermission] = None,
        cleanup: bool = True
    ) -> str:
        """
        创建临时目录
        
        Args:
            name: 目录名称
            permission: 权限配置
            cleanup: 是否自动清理
            
        Returns:
            目录路径
        """
        if name:
            path = os.path.join(self.base_dir, f"{self.prefix}{name}")
            os.makedirs(path, exist_ok=True)
        else:
            path = tempfile.mkdtemp(prefix=self.prefix, dir=self.base_dir)
        
        # 设置权限
        if permission:
            self._apply_permission(path, permission)
        
        # 记录以便清理
        if cleanup:
            self._managed_dirs[path] = path
        
        return path
    
    def create_sandbox_root(
        self,
        sandbox_id: str,
        directories: Optional[List[str]] = None
    ) -> str:
        """
        创建沙箱根目录结构
        
        Args:
            sandbox_id: 沙箱ID
            directories: 要创建的子目录列表
            
        Returns:
            根目录路径
        """
        root = self.create_temp_directory(name=sandbox_id)
        
        # 默认目录结构
        default_dirs = directories or [
            "tmp",
            "input",
            "output",
            "work",
            "home",
            "var/tmp",
            "var/run"
        ]
        
        for dir_name in default_dirs:
            dir_path = os.path.join(root, dir_name)
            os.makedirs(dir_path, exist_ok=True)
        
        # 设置权限
        os.chmod(root, 0o750)
        
        return root
    
    def create_isolated_environment(
        self,
        sandbox_id: str,
        read_only_paths: Optional[List[str]] = None,
        writable_paths: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        创建隔离文件系统环境
        
        Args:
            sandbox_id: 沙箱ID
            read_only_paths: 只读路径列表
            writable_paths: 可写路径列表
            
        Returns:
            路径映射字典
        """
        root = self.create_sandbox_root(sandbox_id)
        paths = {'root': root}
        
        # 创建只读绑定挂载
        if read_only_paths:
            for src_path in read_only_paths:
                if os.path.exists(src_path):
                    name = os.path.basename(src_path)
                    target = os.path.join(root, name)
                    os.makedirs(target, exist_ok=True)
                    self._bind_mount(src_path, target, read_only=True)
                    paths[name] = target
        
        # 创建可写绑定挂载
        if writable_paths:
            for src_path in writable_paths:
                if os.path.exists(src_path):
                    name = os.path.basename(src_path)
                    target = os.path.join(root, f"rw_{name}")
                    os.makedirs(target, exist_ok=True)
                    self._bind_mount(src_path, target, read_only=False)
                    paths[f"rw_{name}"] = target
        
        return paths
    
    def copy_files(
        self,
        files: Dict[str, str],
        target_dir: str,
        permission: Optional[FilePermission] = None
    ) -> Dict[str, str]:
        """
        复制文件到目标目录
        
        Args:
            files: 文件字典 {目标路径: 内容}
            target_dir: 目标目录
            permission: 权限配置
            
        Returns:
            实际写入的文件路径字典
        """
        written_files = {}
        
        for rel_path, content in files.items():
            file_path = os.path.join(target_dir, rel_path)
            dir_path = os.path.dirname(file_path)
            
            # 创建目录
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            # 写入文件
            if isinstance(content, bytes):
                with open(file_path, 'wb') as f:
                    f.write(content)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            # 设置权限
            if permission:
                self._apply_permission(file_path, permission)
            
            written_files[rel_path] = file_path
        
        return written_files
    
    def read_files(
        self,
        directory: str,
        patterns: Optional[List[str]] = None,
        binary: bool = False
    ) -> Dict[str, Union[str, bytes]]:
        """
        读取目录中的文件
        
        Args:
            directory: 目录路径
            patterns: 文件模式列表
            binary: 是否以二进制模式读取
            
        Returns:
            文件内容字典
        """
        files = {}
        directory = Path(directory)
        
        if patterns:
            for pattern in patterns:
                for file_path in directory.glob(pattern):
                    if file_path.is_file():
                        rel_path = str(file_path.relative_to(directory))
                        files[rel_path] = self._read_file(str(file_path), binary)
        else:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    rel_path = str(file_path.relative_to(directory))
                    files[rel_path] = self._read_file(str(file_path), binary)
        
        return files
    
    def _read_file(self, path: str, binary: bool = False) -> Union[str, bytes]:
        """读取文件"""
        try:
            if binary:
                with open(path, 'rb') as f:
                    return f.read()
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
        except (IOError, UnicodeDecodeError):
            return ""
    
    def set_permission(
        self,
        path: str,
        permission: FilePermission
    ) -> bool:
        """
        设置路径权限
        
        Args:
            path: 路径
            permission: 权限配置
            
        Returns:
            是否成功
        """
        return self._apply_permission(path, permission)
    
    def _apply_permission(self, path: str, permission: FilePermission) -> bool:
        """应用权限"""
        try:
            # 设置模式
            os.chmod(path, permission.mode)
            
            # 设置所有者
            if permission.uid is not None or permission.gid is not None:
                uid = permission.uid or -1
                gid = permission.gid or -1
                os.chown(path, uid, gid)
            
            return True
        except OSError:
            return False
    
    def _bind_mount(
        self,
        source: str,
        target: str,
        read_only: bool = False
    ) -> bool:
        """
        执行绑定挂载
        
        Args:
            source: 源路径
            target: 目标路径
            read_only: 是否只读
            
        Returns:
            是否成功
        """
        try:
            import subprocess
            cmd = ["mount", "--bind", source, target]
            subprocess.run(cmd, check=True, capture_output=True)
            
            if read_only:
                # 重新挂载为只读
                cmd = ["mount", "-o", "remount,ro", target]
                subprocess.run(cmd, check=True, capture_output=True)
            
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def unmount(self, path: str) -> bool:
        """
        卸载挂载点
        
        Args:
            path: 挂载点路径
            
        Returns:
            是否成功
        """
        try:
            import subprocess
            cmd = ["umount", path]
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def get_disk_usage(self, path: str) -> Dict[str, int]:
        """
        获取磁盘使用情况
        
        Args:
            path: 路径
            
        Returns:
            使用情况字典
        """
        total_size = 0
        file_count = 0
        dir_count = 0
        
        for root, dirs, files in os.walk(path):
            dir_count += len(dirs)
            file_count += len(files)
            for filename in files:
                file_path = os.path.join(root, filename)
                try:
                    total_size += os.path.getsize(file_path)
                except OSError:
                    pass
        
        return {
            'total_size': total_size,
            'file_count': file_count,
            'dir_count': dir_count
        }
    
    def cleanup(self, path: Optional[str] = None) -> None:
        """
        清理临时目录
        
        Args:
            path: 要清理的路径，None表示清理所有
        """
        if path:
            if path in self._managed_dirs:
                shutil.rmtree(path, ignore_errors=True)
                del self._managed_dirs[path]
        else:
            for managed_path in list(self._managed_dirs.keys()):
                shutil.rmtree(managed_path, ignore_errors=True)
            self._managed_dirs.clear()
    
    def __enter__(self) -> 'FileSystemManager':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()


class OverlayFS:
    """
    OverlayFS管理器
    用于创建叠加文件系统
    """
    
    @staticmethod
    def create(
        lower_dir: Union[str, List[str]],
        upper_dir: str,
        work_dir: str,
        mount_point: str
    ) -> bool:
        """
        创建OverlayFS
        
        Args:
            lower_dir: 下层目录（可以是列表）
            upper_dir: 上层目录（可写层）
            work_dir: 工作目录
            mount_point: 挂载点
            
        Returns:
            是否成功
        """
        try:
            import subprocess
            
            # 处理下层目录
            if isinstance(lower_dir, list):
                lower_str = ":".join(lower_dir)
            else:
                lower_str = lower_dir
            
            # 创建必要的目录
            os.makedirs(upper_dir, exist_ok=True)
            os.makedirs(work_dir, exist_ok=True)
            os.makedirs(mount_point, exist_ok=True)
            
            # 挂载OverlayFS
            cmd = [
                "mount", "-t", "overlay", "overlay",
                "-o", f"lowerdir={lower_str},upperdir={upper_dir},workdir={work_dir}",
                mount_point
            ]
            
            result = subprocess.run(cmd, capture_output=True)
            return result.returncode == 0
            
        except (subprocess.CalledProcessError, OSError):
            return False
    
    @staticmethod
    def destroy(mount_point: str) -> bool:
        """
        销毁OverlayFS
        
        Args:
            mount_point: 挂载点
            
        Returns:
            是否成功
        """
        try:
            import subprocess
            cmd = ["umount", mount_point]
            result = subprocess.run(cmd, capture_output=True)
            return result.returncode == 0
        except subprocess.CalledProcessError:
            return False


class QuotaManager:
    """
    磁盘配额管理器
    """
    
    def __init__(self):
        self._quotas: Dict[str, int] = {}
    
    def set_quota(
        self,
        path: str,
        max_bytes: int,
        max_files: Optional[int] = None
    ) -> bool:
        """
        设置磁盘配额
        
        Args:
            path: 路径
            max_bytes: 最大字节数
            max_files: 最大文件数
            
        Returns:
            是否成功
        """
        # 使用项目配额或组配额
        # 这里简化实现，记录配额信息
        self._quotas[path] = max_bytes
        
        # 实际实现需要使用setquota命令或xfs_quota
        try:
            import subprocess
            
            # 获取设备和配额信息
            # 这里只是示例，实际需要更复杂的逻辑
            return True
        except Exception:
            return False
    
    def get_quota_usage(self, path: str) -> Dict[str, int]:
        """
        获取配额使用情况
        
        Args:
            path: 路径
            
        Returns:
            使用情况字典
        """
        usage = {'bytes': 0, 'files': 0}
        
        for root, _, files in os.walk(path):
            usage['files'] += len(files)
            for filename in files:
                file_path = os.path.join(root, filename)
                try:
                    usage['bytes'] += os.path.getsize(file_path)
                except OSError:
                    pass
        
        return usage
    
    def check_quota(self, path: str, additional_bytes: int = 0) -> bool:
        """
        检查是否超出配额
        
        Args:
            path: 路径
            additional_bytes: 额外需要的字节数
            
        Returns:
            是否在配额内
        """
        if path not in self._quotas:
            return True
        
        usage = self.get_quota_usage(path)
        return usage['bytes'] + additional_bytes <= self._quotas[path]
