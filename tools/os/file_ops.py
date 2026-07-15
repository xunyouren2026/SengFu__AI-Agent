"""
安全文件操作模块
提供带权限检查的读写删改、批量操作功能
"""

import os
import shutil
import stat
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Callable, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """文件信息数据类"""
    path: str
    size: int
    is_dir: bool
    is_file: bool
    is_symlink: bool
    permissions: str
    owner_uid: int
    owner_gid: int
    modified_time: datetime
    accessed_time: datetime
    created_time: Optional[datetime]
    checksum: Optional[str] = None


@dataclass
class OperationResult:
    """操作结果数据类"""
    success: bool
    path: str
    operation: str
    message: str
    error: Optional[str] = None
    data: Any = None


class PermissionChecker:
    """权限检查器"""
    
    def __init__(self, allowed_paths: Optional[List[str]] = None,
                 denied_paths: Optional[List[str]] = None,
                 max_file_size: int = 100 * 1024 * 1024):  # 100MB默认
        """
        初始化权限检查器
        
        Args:
            allowed_paths: 允许访问的路径列表
            denied_paths: 禁止访问的路径列表
            max_file_size: 最大文件大小限制
        """
        self.allowed_paths = [os.path.abspath(p) for p in (allowed_paths or [])]
        self.denied_paths = [os.path.abspath(p) for p in (denied_paths or [])]
        self.max_file_size = max_file_size
        self._dangerous_paths = self._get_dangerous_paths()
    
    def _get_dangerous_paths(self) -> List[str]:
        """获取危险系统路径"""
        dangerous = []
        # 常见的危险路径
        system_paths = [
            '/etc/passwd', '/etc/shadow', '/etc/sudoers',
            '/root', '/var/log',
            'C:\\Windows\\System32', 'C:\\Windows\\System',
        ]
        for path in system_paths:
            if os.path.exists(path):
                dangerous.append(os.path.abspath(path))
        return dangerous
    
    def is_path_allowed(self, path: str) -> Tuple[bool, str]:
        """
        检查路径是否允许访问
        
        Args:
            path: 要检查的路径
            
        Returns:
            (是否允许, 原因消息)
        """
        abs_path = os.path.abspath(path)
        
        # 检查是否在禁止列表
        for denied in self.denied_paths:
            if abs_path.startswith(denied):
                return False, f"路径在禁止列表中: {denied}"
        
        # 检查是否在危险路径
        for dangerous in self._dangerous_paths:
            if abs_path.startswith(dangerous):
                return False, f"路径是系统危险路径: {dangerous}"
        
        # 如果有允许列表，检查是否在其中
        if self.allowed_paths:
            in_allowed = False
            for allowed in self.allowed_paths:
                if abs_path.startswith(allowed):
                    in_allowed = True
                    break
            if not in_allowed:
                return False, "路径不在允许列表中"
        
        return True, "路径访问允许"
    
    def check_file_size(self, path: str) -> Tuple[bool, str]:
        """
        检查文件大小是否在限制内
        
        Args:
            path: 文件路径
            
        Returns:
            (是否允许, 原因消息)
        """
        try:
            if os.path.isfile(path):
                size = os.path.getsize(path)
                if size > self.max_file_size:
                    return False, f"文件大小({size})超过限制({self.max_file_size})"
            return True, "文件大小检查通过"
        except OSError as e:
            return False, f"无法检查文件大小: {e}"
    
    def check_permission(self, path: str, mode: str = 'r') -> Tuple[bool, str]:
        """
        检查文件权限
        
        Args:
            path: 文件路径
            mode: 访问模式 ('r' 读, 'w' 写, 'x' 执行)
            
        Returns:
            (是否允许, 原因消息)
        """
        try:
            if mode == 'r':
                if os.access(path, os.R_OK):
                    return True, "读权限检查通过"
                return False, "没有读权限"
            elif mode == 'w':
                if os.access(path, os.W_OK):
                    return True, "写权限检查通过"
                return False, "没有写权限"
            elif mode == 'x':
                if os.access(path, os.X_OK):
                    return True, "执行权限检查通过"
                return False, "没有执行权限"
            else:
                return False, f"未知权限模式: {mode}"
        except OSError as e:
            return False, f"权限检查失败: {e}"


class FileOperations:
    """安全文件操作类"""
    
    def __init__(self, permission_checker: Optional[PermissionChecker] = None,
                 backup_enabled: bool = True,
                 backup_dir: Optional[str] = None):
        """
        初始化文件操作器
        
        Args:
            permission_checker: 权限检查器实例
            backup_enabled: 是否启用备份
            backup_dir: 备份目录
        """
        self.permission_checker = permission_checker or PermissionChecker()
        self.backup_enabled = backup_enabled
        self.backup_dir = backup_dir or tempfile.gettempdir()
        self._operation_history: List[OperationResult] = []
    
    def _record_operation(self, result: OperationResult) -> None:
        """记录操作历史"""
        self._operation_history.append(result)
        if result.success:
            logger.info(f"操作成功: {result.operation} - {result.path}")
        else:
            logger.error(f"操作失败: {result.operation} - {result.path}: {result.error}")
    
    def _create_backup(self, path: str) -> Optional[str]:
        """创建文件备份"""
        if not self.backup_enabled or not os.path.exists(path):
            return None
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.basename(path)
            backup_path = os.path.join(
                self.backup_dir, 
                f"{filename}.backup_{timestamp}"
            )
            shutil.copy2(path, backup_path)
            return backup_path
        except Exception as e:
            logger.warning(f"创建备份失败: {e}")
            return None
    
    def get_file_info(self, path: str, compute_checksum: bool = False) -> OperationResult:
        """
        获取文件详细信息
        
        Args:
            path: 文件路径
            compute_checksum: 是否计算校验和
            
        Returns:
            操作结果
        """
        # 权限检查
        allowed, msg = self.permission_checker.is_path_allowed(path)
        if not allowed:
            result = OperationResult(False, path, "get_info", msg, msg)
            self._record_operation(result)
            return result
        
        try:
            stat_info = os.stat(path)
            
            checksum = None
            if compute_checksum and os.path.isfile(path):
                checksum = self._compute_checksum(path)
            
            file_info = FileInfo(
                path=os.path.abspath(path),
                size=stat_info.st_size,
                is_dir=os.path.isdir(path),
                is_file=os.path.isfile(path),
                is_symlink=os.path.islink(path),
                permissions=stat.filemode(stat_info.st_mode),
                owner_uid=stat_info.st_uid,
                owner_gid=stat_info.st_gid,
                modified_time=datetime.fromtimestamp(stat_info.st_mtime),
                accessed_time=datetime.fromtimestamp(stat_info.st_atime),
                created_time=datetime.fromtimestamp(stat_info.st_ctime) if hasattr(stat_info, 'st_ctime') else None,
                checksum=checksum
            )
            
            result = OperationResult(True, path, "get_info", "获取文件信息成功", data=file_info)
            self._record_operation(result)
            return result
            
        except Exception as e:
            result = OperationResult(False, path, "get_info", f"获取文件信息失败: {e}", str(e))
            self._record_operation(result)
            return result
    
    def _compute_checksum(self, path: str, algorithm: str = 'sha256') -> str:
        """计算文件校验和"""
        hash_func = hashlib.new(algorithm)
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    
    def read_file(self, path: str, encoding: str = 'utf-8',
                  binary: bool = False, start_line: int = 0,
                  max_lines: Optional[int] = None) -> OperationResult:
        """
        读取文件内容
        
        Args:
            path: 文件路径
            encoding: 文本编码
            binary: 是否以二进制模式读取
            start_line: 起始行号
            max_lines: 最大读取行数
            
        Returns:
            操作结果
        """
        # 权限检查
        allowed, msg = self.permission_checker.is_path_allowed(path)
        if not allowed:
            result = OperationResult(False, path, "read", msg, msg)
            self._record_operation(result)
            return result
        
        size_ok, size_msg = self.permission_checker.check_file_size(path)
        if not size_ok:
            result = OperationResult(False, path, "read", size_msg, size_msg)
            self._record_operation(result)
            return result
        
        perm_ok, perm_msg = self.permission_checker.check_permission(path, 'r')
        if not perm_ok:
            result = OperationResult(False, path, "read", perm_msg, perm_msg)
            self._record_operation(result)
            return result
        
        try:
            if binary:
                with open(path, 'rb') as f:
                    content = f.read()
            else:
                with open(path, 'r', encoding=encoding) as f:
                    if start_line > 0 or max_lines is not None:
                        lines = f.readlines()
                        end_line = start_line + max_lines if max_lines else len(lines)
                        content = ''.join(lines[start_line:end_line])
                    else:
                        content = f.read()
            
            result = OperationResult(True, path, "read", "读取文件成功", data=content)
            self._record_operation(result)
            return result
            
        except Exception as e:
            result = OperationResult(False, path, "read", f"读取文件失败: {e}", str(e))
            self._record_operation(result)
            return result
    
    def write_file(self, path: str, content: Union[str, bytes],
                   encoding: str = 'utf-8', binary: bool = False,
                   append: bool = False, create_dirs: bool = True) -> OperationResult:
        """
        写入文件
        
        Args:
            path: 文件路径
            content: 文件内容
            encoding: 文本编码
            binary: 是否以二进制模式写入
            append: 是否追加模式
            create_dirs: 是否自动创建目录
            
        Returns:
            操作结果
        """
        # 权限检查
        allowed, msg = self.permission_checker.is_path_allowed(path)
        if not allowed:
            result = OperationResult(False, path, "write", msg, msg)
            self._record_operation(result)
            return result
        
        # 创建备份
        if os.path.exists(path):
            self._create_backup(path)
        
        try:
            # 创建目录
            if create_dirs:
                dir_path = os.path.dirname(path)
                if dir_path and not os.path.exists(dir_path):
                    os.makedirs(dir_path)
            
            mode = 'ab' if append and binary else ('wb' if binary else ('a' if append else 'w'))
            
            if binary:
                with open(path, mode) as f:
                    f.write(content if isinstance(content, bytes) else content.encode(encoding))
            else:
                with open(path, mode, encoding=encoding) as f:
                    f.write(content)
            
            result = OperationResult(True, path, "write", "写入文件成功")
            self._record_operation(result)
            return result
            
        except Exception as e:
            result = OperationResult(False, path, "write", f"写入文件失败: {e}", str(e))
            self._record_operation(result)
            return result
    
    def delete_file(self, path: str, force: bool = False) -> OperationResult:
        """
        删除文件
        
        Args:
            path: 文件路径
            force: 是否强制删除（包括只读文件）
            
        Returns:
            操作结果
        """
        # 权限检查
        allowed, msg = self.permission_checker.is_path_allowed(path)
        if not allowed:
            result = OperationResult(False, path, "delete", msg, msg)
            self._record_operation(result)
            return result
        
        if not os.path.exists(path):
            result = OperationResult(False, path, "delete", "文件不存在", "File not found")
            self._record_operation(result)
            return result
        
        try:
            # 创建备份
            self._create_backup(path)
            
            if os.path.isfile(path) or os.path.islink(path):
                if force and not os.access(path, os.W_OK):
                    os.chmod(path, stat.S_IWUSR | stat.S_IRUSR)
                os.remove(path)
            elif os.path.isdir(path):
                if force:
                    shutil.rmtree(path)
                else:
                    os.rmdir(path)
            
            result = OperationResult(True, path, "delete", "删除成功")
            self._record_operation(result)
            return result
            
        except Exception as e:
            result = OperationResult(False, path, "delete", f"删除失败: {e}", str(e))
            self._record_operation(result)
            return result
    
    def copy_file(self, src: str, dst: str, overwrite: bool = False,
                  preserve_metadata: bool = True) -> OperationResult:
        """
        复制文件
        
        Args:
            src: 源文件路径
            dst: 目标路径
            overwrite: 是否覆盖已存在文件
            preserve_metadata: 是否保留元数据
            
        Returns:
            操作结果
        """
        # 权限检查
        for path in [src, dst]:
            allowed, msg = self.permission_checker.is_path_allowed(path)
            if not allowed:
                result = OperationResult(False, path, "copy", msg, msg)
                self._record_operation(result)
                return result
        
        if not os.path.exists(src):
            result = OperationResult(False, src, "copy", "源文件不存在", "Source not found")
            self._record_operation(result)
            return result
        
        if os.path.exists(dst) and not overwrite:
            result = OperationResult(False, dst, "copy", "目标文件已存在", "Destination exists")
            self._record_operation(result)
            return result
        
        try:
            if preserve_metadata:
                shutil.copy2(src, dst)
            else:
                shutil.copy(src, dst)
            
            result = OperationResult(True, f"{src} -> {dst}", "copy", "复制成功")
            self._record_operation(result)
            return result
            
        except Exception as e:
            result = OperationResult(False, f"{src} -> {dst}", "copy", f"复制失败: {e}", str(e))
            self._record_operation(result)
            return result
    
    def move_file(self, src: str, dst: str, overwrite: bool = False) -> OperationResult:
        """
        移动文件
        
        Args:
            src: 源文件路径
            dst: 目标路径
            overwrite: 是否覆盖已存在文件
            
        Returns:
            操作结果
        """
        # 权限检查
        for path in [src, dst]:
            allowed, msg = self.permission_checker.is_path_allowed(path)
            if not allowed:
                result = OperationResult(False, path, "move", msg, msg)
                self._record_operation(result)
                return result
        
        if not os.path.exists(src):
            result = OperationResult(False, src, "move", "源文件不存在", "Source not found")
            self._record_operation(result)
            return result
        
        if os.path.exists(dst) and not overwrite:
            result = OperationResult(False, dst, "move", "目标文件已存在", "Destination exists")
            self._record_operation(result)
            return result
        
        try:
            # 创建备份
            self._create_backup(src)
            
            shutil.move(src, dst)
            
            result = OperationResult(True, f"{src} -> {dst}", "move", "移动成功")
            self._record_operation(result)
            return result
            
        except Exception as e:
            result = OperationResult(False, f"{src} -> {dst}", "move", f"移动失败: {e}", str(e))
            self._record_operation(result)
            return result
    
    def rename_file(self, path: str, new_name: str) -> OperationResult:
        """
        重命名文件
        
        Args:
            path: 文件路径
            new_name: 新文件名
            
        Returns:
            操作结果
        """
        dir_path = os.path.dirname(path)
        new_path = os.path.join(dir_path, new_name)
        return self.move_file(path, new_path)
    
    def create_directory(self, path: str, mode: int = 0o755,
                         parents: bool = True) -> OperationResult:
        """
        创建目录
        
        Args:
            path: 目录路径
            mode: 权限模式
            parents: 是否创建父目录
            
        Returns:
            操作结果
        """
        # 权限检查
        allowed, msg = self.permission_checker.is_path_allowed(path)
        if not allowed:
            result = OperationResult(False, path, "mkdir", msg, msg)
            self._record_operation(result)
            return result
        
        try:
            if parents:
                os.makedirs(path, mode=mode, exist_ok=True)
            else:
                os.mkdir(path, mode=mode)
            
            result = OperationResult(True, path, "mkdir", "创建目录成功")
            self._record_operation(result)
            return result
            
        except Exception as e:
            result = OperationResult(False, path, "mkdir", f"创建目录失败: {e}", str(e))
            self._record_operation(result)
            return result
    
    def list_directory(self, path: str, pattern: str = '*',
                       recursive: bool = False,
                       include_hidden: bool = False) -> OperationResult:
        """
        列出目录内容
        
        Args:
            path: 目录路径
            pattern: 文件匹配模式
            recursive: 是否递归
            include_hidden: 是否包含隐藏文件
            
        Returns:
            操作结果
        """
        # 权限检查
        allowed, msg = self.permission_checker.is_path_allowed(path)
        if not allowed:
            result = OperationResult(False, path, "list", msg, msg)
            self._record_operation(result)
            return result
        
        try:
            path_obj = Path(path)
            
            if recursive:
                items = list(path_obj.rglob(pattern))
            else:
                items = list(path_obj.glob(pattern))
            
            result_list = []
            for item in items:
                if not include_hidden and item.name.startswith('.'):
                    continue
                result_list.append(str(item))
            
            result = OperationResult(True, path, "list", f"列出目录成功，共{len(result_list)}项", data=result_list)
            self._record_operation(result)
            return result
            
        except Exception as e:
            result = OperationResult(False, path, "list", f"列出目录失败: {e}", str(e))
            self._record_operation(result)
            return result
    
    def change_permissions(self, path: str, mode: Union[int, str],
                           recursive: bool = False) -> OperationResult:
        """
        修改文件权限
        
        Args:
            path: 文件路径
            mode: 权限模式（八进制数或字符串如'755'）
            recursive: 是否递归
            
        Returns:
            操作结果
        """
        # 权限检查
        allowed, msg = self.permission_checker.is_path_allowed(path)
        if not allowed:
            result = OperationResult(False, path, "chmod", msg, msg)
            self._record_operation(result)
            return result
        
        try:
            if isinstance(mode, str):
                mode = int(mode, 8)
            
            if recursive and os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    for d in dirs:
                        os.chmod(os.path.join(root, d), mode)
                    for f in files:
                        os.chmod(os.path.join(root, f), mode)
            else:
                os.chmod(path, mode)
            
            result = OperationResult(True, path, "chmod", "修改权限成功")
            self._record_operation(result)
            return result
            
        except Exception as e:
            result = OperationResult(False, path, "chmod", f"修改权限失败: {e}", str(e))
            self._record_operation(result)
            return result
    
    def search_in_file(self, path: str, pattern: str,
                       encoding: str = 'utf-8',
                       ignore_case: bool = False) -> OperationResult:
        """
        在文件中搜索内容
        
        Args:
            path: 文件路径
            pattern: 搜索模式
            encoding: 文件编码
            ignore_case: 是否忽略大小写
            
        Returns:
            操作结果
        """
        import re
        
        read_result = self.read_file(path, encoding=encoding)
        if not read_result.success:
            return read_result
        
        try:
            flags = re.IGNORECASE if ignore_case else 0
            regex = re.compile(pattern, flags)
            
            matches = []
            content = read_result.data
            for i, line in enumerate(content.splitlines(), 1):
                for match in regex.finditer(line):
                    matches.append({
                        'line': i,
                        'content': line,
                        'match': match.group(),
                        'start': match.start(),
                        'end': match.end()
                    })
            
            result = OperationResult(True, path, "search", f"搜索完成，找到{len(matches)}处匹配", data=matches)
            self._record_operation(result)
            return result
            
        except Exception as e:
            result = OperationResult(False, path, "search", f"搜索失败: {e}", str(e))
            self._record_operation(result)
            return result


class BatchOperations:
    """批量文件操作类"""
    
    def __init__(self, file_ops: Optional[FileOperations] = None):
        """
        初始化批量操作器
        
        Args:
            file_ops: 文件操作实例
        """
        self.file_ops = file_ops or FileOperations()
    
    def batch_read(self, paths: List[str], **kwargs) -> Dict[str, OperationResult]:
        """
        批量读取文件
        
        Args:
            paths: 文件路径列表
            **kwargs: 传递给read_file的参数
            
        Returns:
            路径到结果的映射
        """
        results = {}
        for path in paths:
            results[path] = self.file_ops.read_file(path, **kwargs)
        return results
    
    def batch_write(self, file_contents: Dict[str, Union[str, bytes]],
                    **kwargs) -> Dict[str, OperationResult]:
        """
        批量写入文件
        
        Args:
            file_contents: 路径到内容的映射
            **kwargs: 传递给write_file的参数
            
        Returns:
            路径到结果的映射
        """
        results = {}
        for path, content in file_contents.items():
            results[path] = self.file_ops.write_file(path, content, **kwargs)
        return results
    
    def batch_copy(self, copy_pairs: List[Tuple[str, str]],
                   **kwargs) -> Dict[str, OperationResult]:
        """
        批量复制文件
        
        Args:
            copy_pairs: (源路径, 目标路径) 元组列表
            **kwargs: 传递给copy_file的参数
            
        Returns:
            源路径到结果的映射
        """
        results = {}
        for src, dst in copy_pairs:
            results[src] = self.file_ops.copy_file(src, dst, **kwargs)
        return results
    
    def batch_delete(self, paths: List[str],
                     **kwargs) -> Dict[str, OperationResult]:
        """
        批量删除文件
        
        Args:
            paths: 文件路径列表
            **kwargs: 传递给delete_file的参数
            
        Returns:
            路径到结果的映射
        """
        results = {}
        for path in paths:
            results[path] = self.file_ops.delete_file(path, **kwargs)
        return results
    
    def batch_get_info(self, paths: List[str],
                       **kwargs) -> Dict[str, OperationResult]:
        """
        批量获取文件信息
        
        Args:
            paths: 文件路径列表
            **kwargs: 传递给get_file_info的参数
            
        Returns:
            路径到结果的映射
        """
        results = {}
        for path in paths:
            results[path] = self.file_ops.get_file_info(path, **kwargs)
        return results
    
    def batch_operation(self, operations: List[Dict[str, Any]]) -> List[OperationResult]:
        """
        执行批量操作
        
        Args:
            operations: 操作列表，每个操作是包含'operation'和参数的字典
            
        Returns:
            操作结果列表
        """
        results = []
        for op in operations:
            operation = op.pop('operation', None)
            if operation is None:
                results.append(OperationResult(False, '', 'unknown', "未指定操作类型"))
                continue
            
            method = getattr(self.file_ops, operation, None)
            if method is None:
                results.append(OperationResult(False, '', operation, f"未知操作: {operation}"))
                continue
            
            try:
                result = method(**op)
                results.append(result)
            except Exception as e:
                results.append(OperationResult(False, op.get('path', ''), operation, f"操作失败: {e}", str(e)))
        
        return results
    
    def sync_directories(self, src_dir: str, dst_dir: str,
                         delete_extra: bool = False,
                         ignore_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        同步两个目录
        
        Args:
            src_dir: 源目录
            dst_dir: 目标目录
            delete_extra: 是否删除目标目录中多余的文件
            ignore_patterns: 忽略的文件模式列表
            
        Returns:
            同步结果统计
        """
        stats = {
            'copied': 0,
            'updated': 0,
            'deleted': 0,
            'skipped': 0,
            'errors': []
        }
        
        src_path = Path(src_dir)
        dst_path = Path(dst_dir)
        
        # 确保目标目录存在
        dst_path.mkdir(parents=True, exist_ok=True)
        
        # 复制/更新文件
        for src_file in src_path.rglob('*'):
            if src_file.is_dir():
                continue
            
            # 检查忽略模式
            if ignore_patterns:
                if any(src_file.match(p) for p in ignore_patterns):
                    stats['skipped'] += 1
                    continue
            
            rel_path = src_file.relative_to(src_path)
            dst_file = dst_path / rel_path
            
            if dst_file.exists():
                # 检查是否需要更新
                src_mtime = src_file.stat().st_mtime
                dst_mtime = dst_file.stat().st_mtime
                if src_mtime > dst_mtime:
                    result = self.file_ops.copy_file(str(src_file), str(dst_file), overwrite=True)
                    if result.success:
                        stats['updated'] += 1
                    else:
                        stats['errors'].append(str(dst_file))
                else:
                    stats['skipped'] += 1
            else:
                result = self.file_ops.copy_file(str(src_file), str(dst_file))
                if result.success:
                    stats['copied'] += 1
                else:
                    stats['errors'].append(str(dst_file))
        
        # 删除多余文件
        if delete_extra:
            for dst_file in dst_path.rglob('*'):
                if dst_file.is_dir():
                    continue
                
                rel_path = dst_file.relative_to(dst_path)
                src_file = src_path / rel_path
                
                if not src_file.exists():
                    result = self.file_ops.delete_file(str(dst_file))
                    if result.success:
                        stats['deleted'] += 1
                    else:
                        stats['errors'].append(str(dst_file))
        
        return stats


@contextmanager
def safe_open(path: str, mode: str = 'r', 
              permission_checker: Optional[PermissionChecker] = None,
              **kwargs):
    """
    安全文件打开上下文管理器
    
    Args:
        path: 文件路径
        mode: 打开模式
        permission_checker: 权限检查器
        **kwargs: 传递给open的参数
        
    Yields:
        文件对象
    """
    checker = permission_checker or PermissionChecker()
    
    allowed, msg = checker.is_path_allowed(path)
    if not allowed:
        raise PermissionError(msg)
    
    if 'r' in mode or '+' in mode:
        size_ok, size_msg = checker.check_file_size(path)
        if not size_ok:
            raise IOError(size_msg)
    
    try:
        f = open(path, mode, **kwargs)
        yield f
    finally:
        f.close()


def atomic_write(path: str, content: Union[str, bytes],
                 encoding: str = 'utf-8') -> bool:
    """
    原子写入文件
    
    Args:
        path: 目标路径
        content: 内容
        encoding: 编码
        
    Returns:
        是否成功
    """
    temp_path = f"{path}.tmp"
    try:
        if isinstance(content, bytes):
            with open(temp_path, 'wb') as f:
                f.write(content)
        else:
            with open(temp_path, 'w', encoding=encoding) as f:
                f.write(content)
        
        os.replace(temp_path, path)
        return True
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False
