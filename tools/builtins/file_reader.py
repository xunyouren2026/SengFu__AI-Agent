"""
内置文件读取工具模块
支持文本/JSON/CSV/Markdown/YAML/二进制文件读取，路径验证，编码检测，流式读取，文件元数据提取
"""

import os
import re
import json
import csv
import struct
import hashlib
import logging
from pathlib import Path
from typing import (
    Optional, Union, List, Dict, Any, Callable, Iterator, Tuple,
    BinaryIO, TextIO, Generator
)
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from io import StringIO, BytesIO, UnsupportedOperation
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class FileType(Enum):
    """文件类型枚举"""
    TEXT = "text"
    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"
    YAML = "yaml"
    BINARY = "binary"
    UNKNOWN = "unknown"


@dataclass
class ReadResult:
    """读取结果数据类"""
    success: bool
    path: str
    file_type: FileType
    content: Any  # Union[str, dict, list, bytes]
    encoding: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    lines_read: int = 0
    bytes_read: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FileMetadata:
    """文件元数据类"""
    path: str
    size: int
    created_time: Optional[datetime]
    modified_time: datetime
    accessed_time: datetime
    is_readable: bool
    is_writable: bool
    is_executable: bool
    is_symlink: bool
    owner_uid: Optional[int] = None
    owner_gid: Optional[int] = None
    permissions: Optional[str] = None
    checksum: Optional[str] = None
    mime_type: Optional[str] = None
    encoding: Optional[str] = None
    line_count: Optional[int] = None
    word_count: Optional[int] = None
    char_count: Optional[int] = None


class PathValidator:
    """路径验证器 - 沙箱强制执行"""

    # 危险系统路径
    DANGEROUS_PATHS = {
        '/etc/passwd', '/etc/shadow', '/etc/sudoers',
        '/etc/ssh', '/etc/ssl', '/etc/certs',
        '/root', '/home/root',
        '/var/log', '/var/run', '/var/lock',
        '/tmp', '/var/tmp',  # 限制临时目录访问
        'C:\\Windows', 'C:\\Windows\\System32',
        'C:\\Program Files', 'C:\\Program Files (x86)',
        'C:\\Users\\Public', 'C:\\$Recycle.Bin',
    }

    # 允许的文件扩展名
    ALLOWED_EXTENSIONS: Dict[FileType, set] = {
        FileType.TEXT: {'.txt', '.log', '.cfg', '.conf', '.ini', '.xml'},
        FileType.JSON: {'.json', '.jsonl'},
        FileType.CSV: {'.csv', '.tsv'},
        FileType.MARKDOWN: {'.md', '.markdown', '.mdown'},
        FileType.YAML: {'.yaml', '.yml'},
        FileType.BINARY: {
            '.pdf', '.zip', '.tar', '.gz', '.bz2',
            '.png', '.jpg', '.jpeg', '.gif', '.bmp',
            '.mp3', '.mp4', '.avi', '.mov',
            '.doc', '.docx', '.xls', '.xlsx',
        },
    }

    def __init__(
        self,
        allowed_dirs: Optional[List[str]] = None,
        denied_dirs: Optional[List[str]] = None,
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        max_line_length: int = 10000,
        follow_symlinks: bool = False,
        strict_mode: bool = True,
    ):
        """
        初始化路径验证器

        Args:
            allowed_dirs: 允许访问的目录列表
            denied_dirs: 禁止访问的目录列表
            max_file_size: 最大文件大小
            max_line_length: 最大行长度
            follow_symlinks: 是否跟随符号链接
            strict_mode: 严格模式 - 拒绝所有未知路径
        """
        self.allowed_dirs = [os.path.abspath(d) for d in (allowed_dirs or [])]
        self.denied_dirs = [os.path.abspath(d) for d in (denied_dirs or [])]
        self.max_file_size = max_file_size
        self.max_line_length = max_line_length
        self.follow_symlinks = follow_symlinks
        self.strict_mode = strict_mode

    def validate(self, path: str, for_writing: bool = False) -> Tuple[bool, str]:
        """
        验证路径是否可访问

        Args:
            path: 文件路径
            for_writing: 是否为写入操作

        Returns:
            (是否有效, 错误消息)
        """
        # 基础路径解析
        try:
            abs_path = os.path.abspath(path)
        except Exception as e:
            return False, f"无效路径: {e}"

        # 检查符号链接
        if os.path.islink(abs_path) and not self.follow_symlinks:
            return False, "不允许访问符号链接"

        # 检查危险路径
        for dangerous in self.DANGEROUS_PATHS:
            if abs_path.startswith(dangerous):
                return False, f"禁止访问系统路径: {dangerous}"

        # 检查禁止目录
        for denied in self.denied_dirs:
            if abs_path.startswith(denied):
                return False, f"路径在禁止目录中: {denied}"

        # 严格模式下检查允许目录
        if self.strict_mode and self.allowed_dirs:
            in_allowed = any(abs_path.startswith(allowed) for allowed in self.allowed_dirs)
            if not in_allowed:
                return False, "路径不在允许目录列表中"

        # 检查路径遍历攻击
        if '..' in Path(path).parts:
            return False, "禁止路径遍历"

        # 检查文件大小
        if os.path.isfile(abs_path):
            try:
                size = os.path.getsize(abs_path)
                if size > self.max_file_size:
                    return False, f"文件大小({size})超过限制({self.max_file_size})"
            except OSError as e:
                return False, f"无法获取文件大小: {e}"

        return True, "路径有效"

    def get_allowed_extensions(self, file_type: FileType) -> set:
        """获取允许的文件扩展名"""
        return self.ALLOWED_EXTENSIONS.get(file_type, set())

    def is_extension_allowed(self, path: str, file_type: FileType) -> bool:
        """检查文件扩展名是否允许"""
        ext = os.path.splitext(path)[1].lower()
        return ext in self.get_allowed_extensions(file_type)


class EncodingDetector:
    """编码检测器"""

    # BOM标记
    BOM_MARKERS: Dict[bytes, str] = {
        b'\xef\xbb\xbf': 'utf-8-sig',
        b'\xff\xfe': 'utf-16-le',
        b'\xfe\xff': 'utf-16-be',
        b'\xff\xfe\x00\x00': 'utf-32-le',
        b'\x00\x00\xfe\xff': 'utf-32-be',
    }

    # 常见编码的置信度权重
    ENCODING_WEIGHTS: Dict[str, float] = {
        'utf-8': 1.0,
        'utf-8-sig': 1.2,
        'latin-1': 0.7,
        'cp1252': 0.6,
        'iso-8859-1': 0.6,
        'gbk': 0.5,
        'gb2312': 0.5,
        'shift-jis': 0.5,
        'euc-kr': 0.5,
    }

    def __init__(self, sample_size: int = 8192):
        """
        初始化编码检测器

        Args:
            sample_size: 采样大小
        """
        self.sample_size = sample_size

    def detect(self, file_path: str, raw_bytes: Optional[bytes] = None) -> Tuple[str, float]:
        """
        检测文件编码

        Args:
            file_path: 文件路径
            raw_bytes: 原始字节数据(可选)

        Returns:
            (编码名称, 置信度)
        """
        # 检查BOM
        if raw_bytes:
            return self._detect_from_bytes(raw_bytes)

        try:
            with open(file_path, 'rb') as f:
                sample = f.read(self.sample_size)
            return self._detect_from_bytes(sample)
        except Exception as e:
            logger.warning(f"编码检测失败: {e}")
            return 'utf-8', 0.5

    def _detect_from_bytes(self, data: bytes) -> Tuple[str, float]:
        """从字节数据检测编码"""
        # 检查BOM
        for bom, encoding in self.BOM_MARKERS.items():
            if data.startswith(bom):
                return encoding, 1.0

        # UTF-8有效性检查
        utf8_valid = self._is_valid_utf8(data)
        if utf8_valid:
            # 检查是否包含非ASCII字符
            if any(b > 127 for b in data):
                return 'utf-8', 0.9
            else:
                return 'utf-8', 0.8

        # 尝试常见编码
        best_encoding = 'utf-8'
        best_confidence = 0.5

        for encoding in ['latin-1', 'cp1252', 'gbk', 'shift-jis', 'euc-kr']:
            try:
                decoded = data.decode(encoding)
                # 检查字符分布
                confidence = self._calculate_confidence(decoded, encoding)
                if confidence > best_confidence:
                    best_encoding = encoding
                    best_confidence = confidence
            except (UnicodeDecodeError, LookupError):
                continue

        return best_encoding, best_confidence

    def _is_valid_utf8(self, data: bytes) -> bool:
        """检查是否为有效UTF-8"""
        try:
            data.decode('utf-8')
            return True
        except UnicodeDecodeError:
            return False

    def _calculate_confidence(self, text: str, encoding: str) -> float:
        """计算编码置信度"""
        base_weight = self.ENCODING_WEIGHTS.get(encoding, 0.5)

        # 检查控制字符
        control_chars = sum(1 for c in text if c in '\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f')
        if control_chars > len(text) * 0.1:
            return base_weight * 0.5

        # 检查空字符
        null_ratio = text.count('\x00') / max(len(text), 1)
        if null_ratio > 0.1:
            return base_weight * 0.3

        return base_weight


class FileMetadataExtractor:
    """文件元数据提取器"""

    def __init__(self, compute_checksum: bool = True, checksum_algorithm: str = 'sha256'):
        """
        初始化元数据提取器

        Args:
            compute_checksum: 是否计算校验和
            checksum_algorithm: 校验和算法
        """
        self.compute_checksum = compute_checksum
        self.checksum_algorithm = checksum_algorithm

    def extract(self, path: str, include_content_stats: bool = True) -> FileMetadata:
        """
        提取文件元数据

        Args:
            path: 文件路径
            include_content_stats: 是否包含内容统计

        Returns:
            文件元数据对象
        """
        abs_path = os.path.abspath(path)

        try:
            stat_info = os.stat(abs_path)
        except OSError as e:
            raise ValueError(f"无法访问文件: {e}")

        metadata = FileMetadata(
            path=abs_path,
            size=stat_info.st_size,
            created_time=datetime.fromtimestamp(stat_info.st_ctime) if hasattr(stat_info, 'st_ctime') else None,
            modified_time=datetime.fromtimestamp(stat_info.st_mtime),
            accessed_time=datetime.fromtimestamp(stat_info.st_atime),
            is_readable=os.access(abs_path, os.R_OK),
            is_writable=os.access(abs_path, os.W_OK),
            is_executable=os.access(abs_path, os.X_OK),
            is_symlink=os.path.islink(abs_path),
            owner_uid=stat_info.st_uid if hasattr(stat_info, 'st_uid') else None,
            owner_gid=stat_info.st_gid if hasattr(stat_info, 'st_gid') else None,
        )

        # 提取权限
        if hasattr(stat_info, 'st_mode'):
            import stat as stat_module
            metadata.permissions = stat_module.filemode(stat_info.st_mode)

        # 计算校验和
        if self.compute_checksum and os.path.isfile(abs_path):
            metadata.checksum = self._compute_checksum(abs_path)

        # 内容统计
        if include_content_stats and os.path.isfile(abs_path):
            self._extract_content_stats(abs_path, metadata)

        return metadata

    def _compute_checksum(self, path: str) -> str:
        """计算文件校验和"""
        hash_func = hashlib.new(self.checksum_algorithm)
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hash_func.update(chunk)
        return hash_func.hexdigest()

    def _extract_content_stats(self, path: str, metadata: FileMetadata) -> None:
        """提取内容统计"""
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                metadata.line_count = content.count('\n') + 1
                metadata.word_count = len(content.split())
                metadata.char_count = len(content)
        except Exception:
            # 二进制文件或编码问题
            metadata.line_count = None
            metadata.word_count = None
            metadata.char_count = None


class StreamReader:
    """流式读取器基类"""

    def __init__(self, chunk_size: int = 8192):
        """
        初始化流式读取器

        Args:
            chunk_size: 块大小
        """
        self.chunk_size = chunk_size

    def read_chunks(self, file_path: str) -> Generator[bytes, None, None]:
        """分块读取文件"""
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                yield chunk

    def read_lines(self, file_path: str, encoding: str = 'utf-8') -> Generator[str, None, None]:
        """逐行读取文件"""
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            for line in f:
                yield line

    def read_stream(self, file_path: str, start: int = 0, end: Optional[int] = None) -> Iterator[bytes]:
        """流式读取文件片段"""
        with open(file_path, 'rb') as f:
            if start > 0:
                f.seek(start)
            remaining = end - start if end else None
            while True:
                chunk_size = min(self.chunk_size, remaining) if remaining else self.chunk_size
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
                if remaining:
                    remaining -= len(chunk)
                    if remaining <= 0:
                        break


class TextFileReader:
    """文本文件读取器"""

    def __init__(
        self,
        path_validator: Optional[PathValidator] = None,
        encoding_detector: Optional[EncodingDetector] = None,
    ):
        """
        初始化文本文件读取器

        Args:
            path_validator: 路径验证器
            encoding_detector: 编码检测器
        """
        self.path_validator = path_validator or PathValidator()
        self.encoding_detector = encoding_detector or EncodingDetector()
        self.stream_reader = StreamReader()

    def read(
        self,
        path: str,
        encoding: Optional[str] = None,
        start_line: int = 0,
        max_lines: Optional[int] = None,
        strip_whitespace: bool = True,
    ) -> ReadResult:
        """
        读取文本文件

        Args:
            path: 文件路径
            encoding: 编码(自动检测)
            start_line: 起始行
            max_lines: 最大行数
            strip_whitespace: 去除空白

        Returns:
            读取结果
        """
        # 验证路径
        valid, msg = self.path_validator.validate(path)
        if not valid:
            return ReadResult(False, path, FileType.TEXT, None, error=msg)

        try:
            # 自动检测编码
            if encoding is None:
                encoding, _ = self.encoding_detector.detect(path)

            # 读取文件
            lines = []
            total_bytes = 0

            for i, line in enumerate(self.stream_reader.read_lines(path, encoding)):
                if i < start_line:
                    continue
                if max_lines is not None and i >= start_line + max_lines:
                    break

                line_content = line if not strip_whitespace else line.strip()
                if line_content or not strip_whitespace:
                    lines.append(line_content)
                total_bytes += len(line.encode(encoding, errors='replace'))

            content = '\n'.join(lines)
            metadata = {'line_count': len(lines), 'bytes_read': total_bytes}

            return ReadResult(
                success=True,
                path=path,
                file_type=FileType.TEXT,
                content=content,
                encoding=encoding,
                metadata=metadata,
                lines_read=len(lines),
                bytes_read=total_bytes,
            )

        except Exception as e:
            return ReadResult(False, path, FileType.TEXT, None, error=str(e))


class JSONFileReader:
    """JSON文件读取器"""

    def __init__(self, path_validator: Optional[PathValidator] = None):
        """
        初始化JSON文件读取器

        Args:
            path_validator: 路径验证器
        """
        self.path_validator = path_validator or PathValidator()

    def read(
        self,
        path: str,
        encoding: str = 'utf-8',
        parse_jsonlines: bool = False,
    ) -> ReadResult:
        """
        读取JSON文件

        Args:
            path: 文件路径
            encoding: 编码
            parse_jsonlines: 是否解析JSON Lines格式

        Returns:
            读取结果
        """
        valid, msg = self.path_validator.validate(path)
        if not valid:
            return ReadResult(False, path, FileType.JSON, None, error=msg)

        try:
            with open(path, 'r', encoding=encoding, errors='ignore') as f:
                content = f.read()

            if parse_jsonlines:
                # JSON Lines格式
                data = []
                lines = content.splitlines()
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"第{i+1}行JSON解析失败: {e}")
                metadata = {'line_count': len(data), 'type': 'jsonlines'}
            else:
                data = json.loads(content)
                metadata = {'type': 'json'}

            return ReadResult(
                success=True,
                path=path,
                file_type=FileType.JSON,
                content=data,
                encoding=encoding,
                metadata=metadata,
                bytes_read=len(content.encode(encoding, errors='replace')),
            )

        except json.JSONDecodeError as e:
            return ReadResult(False, path, FileType.JSON, None, error=f"JSON解析错误: {e}")
        except Exception as e:
            return ReadResult(False, path, FileType.JSON, None, error=str(e))


class CSVFileReader:
    """CSV文件读取器"""

    def __init__(
        self,
        path_validator: Optional[PathValidator] = None,
        encoding_detector: Optional[EncodingDetector] = None,
    ):
        """
        初始化CSV文件读取器

        Args:
            path_validator: 路径验证器
            encoding_detector: 编码检测器
        """
        self.path_validator = path_validator or PathValidator()
        self.encoding_detector = encoding_detector or EncodingDetector()

    def read(
        self,
        path: str,
        encoding: Optional[str] = None,
        delimiter: Optional[str] = None,
        has_header: bool = True,
        max_rows: Optional[int] = None,
    ) -> ReadResult:
        """
        读取CSV文件

        Args:
            path: 文件路径
            encoding: 编码(自动检测)
            delimiter: 分隔符(自动检测)
            has_header: 是否有表头
            max_rows: 最大行数

        Returns:
            读取结果
        """
        valid, msg = self.path_validator.validate(path)
        if not valid:
            return ReadResult(False, path, FileType.CSV, None, error=msg)

        try:
            # 自动检测编码
            if encoding is None:
                encoding, _ = self.encoding_detector.detect(path)

            # 自动检测分隔符
            if delimiter is None:
                delimiter = self._detect_delimiter(path, encoding)

            # 读取CSV
            rows = []
            with open(path, 'r', encoding=encoding, errors='ignore', newline='') as f:
                reader = csv.reader(f, delimiter=delimiter)
                for i, row in enumerate(reader):
                    if max_rows is not None and i >= max_rows:
                        break
                    rows.append(row)

            metadata = {
                'row_count': len(rows),
                'column_count': len(rows[0]) if rows else 0,
                'delimiter': delimiter,
                'has_header': has_header,
            }

            if has_header and rows:
                headers = rows[0]
                data = {'headers': headers, 'rows': rows[1:]}
            else:
                data = {'rows': rows}

            return ReadResult(
                success=True,
                path=path,
                file_type=FileType.CSV,
                content=data,
                encoding=encoding,
                metadata=metadata,
                lines_read=len(rows),
            )

        except Exception as e:
            return ReadResult(False, path, FileType.CSV, None, error=str(e))

    def _detect_delimiter(self, path: str, encoding: str) -> str:
        """检测CSV分隔符"""
        try:
            with open(path, 'r', encoding=encoding, errors='ignore') as f:
                sample = f.readline()

            # 常见分隔符
            delimiters = [',', '\t', ';', '|', ':']
            best_delimiter = ','
            best_count = 0

            for d in delimiters:
                count = sample.count(d)
                if count > best_count:
                    best_count = count
                    best_delimiter = d

            return best_delimiter
        except Exception:
            return ','


class MarkdownFileReader:
    """Markdown文件读取器"""

    def __init__(self, path_validator: Optional[PathValidator] = None):
        """
        初始化Markdown文件读取器

        Args:
            path_validator: 路径验证器
        """
        self.path_validator = path_validator or PathValidator()

    def read(
        self,
        path: str,
        encoding: str = 'utf-8',
        extract_metadata: bool = True,
    ) -> ReadResult:
        """
        读取Markdown文件

        Args:
            path: 文件路径
            encoding: 编码
            extract_metadata: 是否提取元数据

        Returns:
            读取结果
        """
        valid, msg = self.path_validator.validate(path)
        if not valid:
            return ReadResult(False, path, FileType.MARKDOWN, None, error=msg)

        try:
            with open(path, 'r', encoding=encoding, errors='ignore') as f:
                content = f.read()

            metadata = {'type': 'markdown'}

            if extract_metadata:
                # 提取YAML front matter
                front_matter, body = self._extract_front_matter(content)
                if front_matter:
                    metadata['front_matter'] = front_matter
                    content = body

                # 提取标题
                headings = self._extract_headings(content)
                metadata['headings'] = headings

                # 统计
                metadata['word_count'] = len(content.split())
                metadata['char_count'] = len(content)

            return ReadResult(
                success=True,
                path=path,
                file_type=FileType.MARKDOWN,
                content=content,
                encoding=encoding,
                metadata=metadata,
                bytes_read=len(content.encode(encoding, errors='replace')),
            )

        except Exception as e:
            return ReadResult(False, path, FileType.MARKDOWN, None, error=str(e))

    def _extract_front_matter(self, content: str) -> Tuple[Optional[Dict], str]:
        """提取YAML front matter"""
        pattern = r'^---\s*\n(.*?)\n---\s*\n'
        match = re.match(pattern, content, re.DOTALL)
        if match:
            import yaml
            try:
                front_matter = yaml.safe_load(match.group(1))
                body = content[match.end():]
                return front_matter, body
            except Exception:
                pass
        return None, content

    def _extract_headings(self, content: str) -> List[Dict[str, Any]]:
        """提取所有标题"""
        headings = []
        for line in content.splitlines():
            match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if match:
                headings.append({
                    'level': len(match.group(1)),
                    'text': match.group(2).strip(),
                })
        return headings


class YAMLFileReader:
    """YAML文件读取器"""

    def __init__(self, path_validator: Optional[PathValidator] = None):
        """
        初始化YAML文件读取器

        Args:
            path_validator: 路径验证器
        """
        self.path_validator = path_validator or PathValidator()

    def read(self, path: str, encoding: str = 'utf-8') -> ReadResult:
        """
        读取YAML文件

        Args:
            path: 文件路径
            encoding: 编码

        Returns:
            读取结果
        """
        valid, msg = self.path_validator.validate(path)
        if not valid:
            return ReadResult(False, path, FileType.YAML, None, error=msg)

        try:
            import yaml

            with open(path, 'r', encoding=encoding, errors='ignore') as f:
                data = yaml.safe_load(f)

            return ReadResult(
                success=True,
                path=path,
                file_type=FileType.YAML,
                content=data,
                encoding=encoding,
                metadata={'type': 'yaml'},
                bytes_read=os.path.getsize(path),
            )

        except ImportError:
            # 手动解析简单YAML
            return self._read_yaml_manual(path, encoding)
        except yaml.YAMLError as e:
            return ReadResult(False, path, FileType.YAML, None, error=f"YAML解析错误: {e}")
        except Exception as e:
            return ReadResult(False, path, FileType.YAML, None, error=str(e))

    def _read_yaml_manual(self, path: str, encoding: str) -> ReadResult:
        """手动解析简单YAML"""
        try:
            with open(path, 'r', encoding=encoding, errors='ignore') as f:
                content = f.read()

            data = {}
            current_key = None
            current_list = None

            for line in content.splitlines():
                line = line.rstrip()
                if not line or line.startswith('#'):
                    continue

                # 列表项
                if line.startswith('- '):
                    if current_list is None:
                        current_list = []
                        data[current_key] = current_list
                    current_list.append(line[2:].strip())
                    continue

                # 键值对
                match = re.match(r'^([^:]+):\s*(.*)$', line)
                if match:
                    key = match.group(1).strip()
                    value = match.group(2).strip()

                    current_key = key
                    current_list = None

                    if value:
                        data[key] = value
                    else:
                        data[key] = None

            return ReadResult(
                success=True,
                path=path,
                file_type=FileType.YAML,
                content=data,
                encoding=encoding,
                metadata={'type': 'yaml', 'manual_parse': True},
            )
        except Exception as e:
            return ReadResult(False, path, FileType.YAML, None, error=str(e))


class BinaryFileReader:
    """二进制文件读取器"""

    def __init__(self, path_validator: Optional[PathValidator] = None):
        """
        初始化二进制文件读取器

        Args:
            path_validator: 路径验证器
        """
        self.path_validator = path_validator or PathValidator()
        self.stream_reader = StreamReader()

    def read(
        self,
        path: str,
        offset: int = 0,
        size: Optional[int] = None,
        as_hex: bool = False,
    ) -> ReadResult:
        """
        读取二进制文件

        Args:
            path: 文件路径
            offset: 起始偏移
            size: 读取大小
            as_hex: 是否以十六进制返回

        Returns:
            读取结果
        """
        valid, msg = self.path_validator.validate(path)
        if not valid:
            return ReadResult(False, path, FileType.BINARY, None, error=msg)

        try:
            file_size = os.path.getsize(path)
            read_size = min(size, file_size - offset) if size else file_size - offset

            if read_size <= 0:
                return ReadResult(False, path, FileType.BINARY, None, error="无效的偏移或大小")

            with open(path, 'rb') as f:
                f.seek(offset)
                data = f.read(read_size if size else -1)

            content = data.hex() if as_hex else data
            metadata = {
                'size': file_size,
                'offset': offset,
                'bytes_read': len(data),
                'mime_type': self._detect_mime(data),
            }

            return ReadResult(
                success=True,
                path=path,
                file_type=FileType.BINARY,
                content=content,
                metadata=metadata,
                bytes_read=len(data),
            )

        except Exception as e:
            return ReadResult(False, path, FileType.BINARY, None, error=str(e))

    def _detect_mime(self, data: bytes) -> str:
        """检测MIME类型"""
        if len(data) < 4:
            return 'application/octet-stream'

        # PNG
        if data[:4] == b'\x89PNG':
            return 'image/png'

        # JPEG
        if data[:2] == b'\xff\xd8':
            return 'image/jpeg'

        # GIF
        if data[:3] == b'GIF':
            return 'image/gif'

        # PDF
        if data[:4] == b'%PDF':
            return 'application/pdf'

        # ZIP
        if data[:4] == b'PK\x03\x04':
            return 'application/zip'

        # GZIP
        if data[:2] == b'\x1f\x8b':
            return 'application/gzip'

        return 'application/octet-stream'

    def read_chunks(self, path: str, chunk_size: int = 65536) -> Generator[bytes, None, None]:
        """分块读取"""
        valid, msg = self.path_validator.validate(path)
        if not valid:
            raise ValueError(msg)
        yield from self.stream_reader.read_chunks(path)


class FileReader:
    """统一文件读取器"""

    # 文件类型到读取器的映射
    READERS: Dict[FileType, Any] = {
        FileType.TEXT: TextFileReader,
        FileType.JSON: JSONFileReader,
        FileType.CSV: CSVFileReader,
        FileType.MARKDOWN: MarkdownFileReader,
        FileType.YAML: YAMLFileReader,
        FileType.BINARY: BinaryFileReader,
    }

    # 扩展名到文件类型的映射
    EXTENSION_MAP: Dict[str, FileType] = {
        '.txt': FileType.TEXT,
        '.log': FileType.TEXT,
        '.cfg': FileType.TEXT,
        '.conf': FileType.TEXT,
        '.ini': FileType.TEXT,
        '.json': FileType.JSON,
        '.jsonl': FileType.JSON,
        '.csv': FileType.CSV,
        '.tsv': FileType.CSV,
        '.md': FileType.MARKDOWN,
        '.markdown': FileType.MARKDOWN,
        '.mdown': FileType.MARKDOWN,
        '.yaml': FileType.YAML,
        '.yml': FileType.YAML,
    }

    def __init__(
        self,
        path_validator: Optional[PathValidator] = None,
        encoding_detector: Optional[EncodingDetector] = None,
        auto_detect_type: bool = True,
    ):
        """
        初始化文件读取器

        Args:
            path_validator: 路径验证器
            encoding_detector: 编码检测器
            auto_detect_type: 自动检测文件类型
        """
        self.path_validator = path_validator or PathValidator()
        self.encoding_detector = encoding_detector or EncodingDetector()
        self.auto_detect_type = auto_detect_type

        # 初始化各类型读取器
        self._readers: Dict[FileType, Any] = {}
        for file_type, reader_class in self.READERS.items():
            if file_type == FileType.TEXT:
                self._readers[file_type] = reader_class(
                    path_validator=self.path_validator,
                    encoding_detector=self.encoding_detector,
                )
            else:
                self._readers[file_type] = reader_class(
                    path_validator=self.path_validator
                )

    def detect_file_type(self, path: str) -> FileType:
        """检测文件类型"""
        ext = os.path.splitext(path)[1].lower()

        # 根据扩展名判断
        if ext in self.EXTENSION_MAP:
            return self.EXTENSION_MAP[ext]

        # 根据内容判断
        try:
            with open(path, 'rb') as f:
                header = f.read(1024)

            # JSON
            try:
                header.decode('utf-8').strip()
                # 检查是否是JSON格式
                content_str = header.decode('utf-8', errors='ignore').strip()
                if content_str.startswith('{') or content_str.startswith('['):
                    return FileType.JSON
            except UnicodeDecodeError:
                pass

            # 检查二进制特征
            if self._is_binary_content(header):
                # 可能是二进制或CSV
                if b',' in header or b'\t' in header:
                    return FileType.CSV
                return FileType.BINARY

        except Exception:
            pass

        return FileType.UNKNOWN

    def _is_binary_content(self, data: bytes) -> bool:
        """判断是否为二进制内容"""
        # 检查空字符
        if b'\x00' in data[:100]:
            return True

        # 检查非文本字符比例
        text_chars = bytes(range(32, 127)) + b'\n\r\t'
        text_ratio = sum(1 for b in data[:512] if b in text_chars) / max(len(data[:512]), 1)
        return text_ratio < 0.7

    def read(
        self,
        path: str,
        file_type: Optional[FileType] = None,
        **kwargs,
    ) -> ReadResult:
        """
        读取文件

        Args:
            path: 文件路径
            file_type: 文件类型(自动检测)
            **kwargs: 传递给具体读取器的参数

        Returns:
            读取结果
        """
        # 检测文件类型
        if file_type is None or self.auto_detect_type:
            file_type = self.detect_file_type(path)

        # 获取读取器
        reader = self._readers.get(file_type)
        if reader is None:
            return ReadResult(
                False, path, file_type or FileType.UNKNOWN, None,
                error=f"不支持的文件类型: {file_type}"
            )

        return reader.read(path, **kwargs)

    def read_streaming(
        self,
        path: str,
        chunk_size: int = 8192,
    ) -> Generator[Union[str, bytes], None, None]:
        """流式读取文件"""
        valid, msg = self.path_validator.validate(path)
        if not valid:
            raise ValueError(msg)

        file_type = self.detect_file_type(path)

        if file_type in (FileType.TEXT, FileType.MARKDOWN, FileType.CSV, FileType.JSON):
            # 文本模式
            encoding, _ = self.encoding_detector.detect(path)
            with open(path, 'r', encoding=encoding, errors='ignore') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        else:
            # 二进制模式
            with open(path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
