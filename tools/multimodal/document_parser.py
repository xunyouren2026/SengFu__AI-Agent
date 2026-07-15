"""
Document Parser Tool - 文档理解工具
解析PDF、Word、Excel等文档
"""

import json
import re
import zipfile
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any, Union, Tuple, BinaryIO
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
import logging

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """文档类型枚举"""
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    TXT = "txt"
    HTML = "html"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


class ContentType(Enum):
    """内容类型枚举"""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    HEADING = "heading"
    LIST = "list"
    CODE = "code"
    FOOTNOTE = "footnote"


@dataclass
class DocumentSection:
    """文档节"""
    section_id: int
    content_type: ContentType
    content: str
    level: int = 0  # 标题级别
    page_number: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "content_type": self.content_type.value,
            "content": self.content,
            "level": self.level,
            "page_number": self.page_number,
            "metadata": self.metadata
        }


@dataclass
class TableData:
    """表格数据"""
    headers: List[str]
    rows: List[List[str]]
    caption: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "headers": self.headers,
            "rows": self.rows,
            "caption": self.caption
        }
    
    def to_dataframe(self) -> Dict[str, List[Any]]:
        """转换为数据框格式"""
        result = {h: [] for h in self.headers}
        for row in self.rows:
            for i, h in enumerate(self.headers):
                result[h].append(row[i] if i < len(row) else "")
        return result


@dataclass
class DocumentMetadata:
    """文档元数据"""
    title: str = ""
    author: str = ""
    subject: str = ""
    keywords: List[str] = field(default_factory=list)
    creator: str = ""
    producer: str = ""
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    page_count: int = 0
    word_count: int = 0
    char_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "author": self.author,
            "subject": self.subject,
            "keywords": self.keywords,
            "creator": self.creator,
            "producer": self.producer,
            "creation_date": self.creation_date,
            "modification_date": self.modification_date,
            "page_count": self.page_count,
            "word_count": self.word_count,
            "char_count": self.char_count
        }


@dataclass
class ParsedDocument:
    """解析后的文档"""
    document_type: DocumentType
    metadata: DocumentMetadata
    sections: List[DocumentSection]
    tables: List[TableData]
    full_text: str
    outline: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_type": self.document_type.value,
            "metadata": self.metadata.to_dict(),
            "sections": [s.to_dict() for s in self.sections],
            "tables": [t.to_dict() for t in self.tables],
            "full_text": self.full_text,
            "outline": self.outline,
            "images": self.images,
            "errors": self.errors
        }
    
    def get_text_by_type(self, content_type: ContentType) -> List[str]:
        """按类型获取文本"""
        return [s.content for s in self.sections if s.content_type == content_type]
    
    def search(self, pattern: str, case_sensitive: bool = False) -> List[Tuple[int, str]]:
        """搜索文本"""
        flags = 0 if case_sensitive else re.IGNORECASE
        results = []
        
        try:
            regex = re.compile(pattern, flags)
        except re.error:
            regex = re.compile(re.escape(pattern), flags)
        
        for section in self.sections:
            if regex.search(section.content):
                results.append((section.section_id, section.content))
        
        return results


class DocumentParser:
    """文档解析工具"""
    
    def __init__(self):
        self._parsers = {
            DocumentType.TXT: self._parse_txt,
            DocumentType.DOCX: self._parse_docx,
            DocumentType.XLSX: self._parse_xlsx,
            DocumentType.HTML: self._parse_html,
            DocumentType.MARKDOWN: self._parse_markdown,
        }
    
    def parse(self, file_path: str,
              document_type: Optional[DocumentType] = None) -> ParsedDocument:
        """解析文档"""
        # 检测文档类型
        if document_type is None:
            document_type = self._detect_type(file_path)
        
        # 读取文件
        with open(file_path, 'rb') as f:
            content = f.read()
        
        return self.parse_bytes(content, document_type)
    
    def parse_bytes(self, content: bytes,
                    document_type: DocumentType) -> ParsedDocument:
        """解析字节数据"""
        parser = self._parsers.get(document_type, self._parse_unknown)
        
        try:
            return parser(content)
        except Exception as e:
            logger.error(f"Failed to parse document: {e}")
            return ParsedDocument(
                document_type=document_type,
                metadata=DocumentMetadata(),
                sections=[],
                tables=[],
                full_text="",
                errors=[str(e)]
            )
    
    def parse_text(self, text: str,
                   document_type: DocumentType = DocumentType.TXT) -> ParsedDocument:
        """解析文本"""
        content = text.encode('utf-8')
        return self.parse_bytes(content, document_type)
    
    def _detect_type(self, file_path: str) -> DocumentType:
        """检测文档类型"""
        ext = file_path.lower().split('.')[-1] if '.' in file_path else ''
        
        type_map = {
            'pdf': DocumentType.PDF,
            'docx': DocumentType.DOCX,
            'doc': DocumentType.DOCX,
            'xlsx': DocumentType.XLSX,
            'xls': DocumentType.XLSX,
            'txt': DocumentType.TXT,
            'html': DocumentType.HTML,
            'htm': DocumentType.HTML,
            'md': DocumentType.MARKDOWN,
        }
        
        return type_map.get(ext, DocumentType.UNKNOWN)
    
    def _parse_txt(self, content: bytes) -> ParsedDocument:
        """解析纯文本"""
        text = content.decode('utf-8', errors='ignore')
        
        sections = [DocumentSection(
            section_id=0,
            content_type=ContentType.TEXT,
            content=text,
            page_number=1
        )]
        
        metadata = DocumentMetadata(
            word_count=len(text.split()),
            char_count=len(text)
        )
        
        return ParsedDocument(
            document_type=DocumentType.TXT,
            metadata=metadata,
            sections=sections,
            tables=[],
            full_text=text
        )
    
    def _parse_docx(self, content: bytes) -> ParsedDocument:
        """解析DOCX文件"""
        sections = []
        tables = []
        full_text_parts = []
        section_id = 0
        
        try:
            with zipfile.ZipFile(BytesIO(content)) as zf:
                # 读取document.xml
                doc_xml = zf.read('word/document.xml')
                root = ET.fromstring(doc_xml)
                
                # 定义命名空间
                ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
                
                # 解析段落
                for para in root.findall('.//w:p', ns):
                    text_parts = []
                    for text_elem in para.findall('.//w:t', ns):
                        if text_elem.text:
                            text_parts.append(text_elem.text)
                    
                    if text_parts:
                        text = ''.join(text_parts)
                        full_text_parts.append(text)
                        
                        # 检测标题样式
                        style = para.find('.//w:pStyle', ns)
                        is_heading = False
                        heading_level = 0
                        
                        if style is not None:
                            style_val = style.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', '')
                            if style_val.startswith('Heading'):
                                is_heading = True
                                heading_level = int(style_val.replace('Heading', '')) if style_val.replace('Heading', '').isdigit() else 1
                        
                        sections.append(DocumentSection(
                            section_id=section_id,
                            content_type=ContentType.HEADING if is_heading else ContentType.TEXT,
                            content=text,
                            level=heading_level
                        ))
                        section_id += 1
                
                # 解析表格
                for table in root.findall('.//w:tbl', ns):
                    rows = []
                    for row in table.findall('.//w:tr', ns):
                        cells = []
                        for cell in row.findall('.//w:tc', ns):
                            cell_text = []
                            for text_elem in cell.findall('.//w:t', ns):
                                if text_elem.text:
                                    cell_text.append(text_elem.text)
                            cells.append(' '.join(cell_text))
                        rows.append(cells)
                    
                    if rows:
                        headers = rows[0] if rows else []
                        data_rows = rows[1:] if len(rows) > 1 else []
                        tables.append(TableData(headers=headers, rows=data_rows))
                
                # 读取核心属性
                metadata = DocumentMetadata()
                try:
                    core_xml = zf.read('docProps/core.xml')
                    core_root = ET.fromstring(core_xml)
                    dc_ns = {'dc': 'http://purl.org/dc/elements/1.1/'}
                    
                    title = core_root.find('.//dc:title', dc_ns)
                    if title is not None and title.text:
                        metadata.title = title.text
                    
                    author = core_root.find('.//dc:creator', dc_ns)
                    if author is not None and author.text:
                        metadata.author = author.text
                except Exception:
                    pass
        
        except Exception as e:
            logger.error(f"Failed to parse DOCX: {e}")
        
        full_text = '\n'.join(full_text_parts)
        metadata.word_count = len(full_text.split())
        metadata.char_count = len(full_text)
        
        return ParsedDocument(
            document_type=DocumentType.DOCX,
            metadata=metadata,
            sections=sections,
            tables=tables,
            full_text=full_text
        )
    
    def _parse_xlsx(self, content: bytes) -> ParsedDocument:
        """解析XLSX文件"""
        tables = []
        sections = []
        full_text_parts = []
        section_id = 0
        
        try:
            with zipfile.ZipFile(BytesIO(content)) as zf:
                # 读取共享字符串
                shared_strings = []
                try:
                    ss_xml = zf.read('xl/sharedStrings.xml')
                    ss_root = ET.fromstring(ss_xml)
                    for si in ss_root.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t'):
                        if si.text:
                            shared_strings.append(si.text)
                except Exception:
                    pass
                
                # 读取工作表
                sheet_files = [f for f in zf.namelist() if f.startswith('xl/worksheets/') and f.endswith('.xml')]
                
                for sheet_file in sheet_files:
                    sheet_xml = zf.read(sheet_file)
                    sheet_root = ET.fromstring(sheet_xml)
                    
                    rows = []
                    ns = {'': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
                    
                    for row in sheet_root.findall('.//row'):
                        cells = []
                        for cell in row.findall('c'):
                            cell_type = cell.get('t', '')
                            value_elem = cell.find('v')
                            
                            if value_elem is not None and value_elem.text:
                                if cell_type == 's':
                                    # 共享字符串引用
                                    idx = int(value_elem.text)
                                    if idx < len(shared_strings):
                                        cells.append(shared_strings[idx])
                                    else:
                                        cells.append(value_elem.text)
                                else:
                                    cells.append(value_elem.text)
                            else:
                                cells.append('')
                        
                        if cells:
                            rows.append(cells)
                    
                    if rows:
                        headers = rows[0]
                        data_rows = rows[1:]
                        tables.append(TableData(headers=headers, rows=data_rows))
                        
                        # 添加为文本节
                        sheet_text = f"Sheet: {sheet_file}\n"
                        sheet_text += ' | '.join(headers) + '\n'
                        for row in data_rows[:10]:  # 只取前10行预览
                            sheet_text += ' | '.join(row) + '\n'
                        
                        full_text_parts.append(sheet_text)
                        sections.append(DocumentSection(
                            section_id=section_id,
                            content_type=ContentType.TABLE,
                            content=sheet_text
                        ))
                        section_id += 1
        
        except Exception as e:
            logger.error(f"Failed to parse XLSX: {e}")
        
        full_text = '\n'.join(full_text_parts)
        metadata = DocumentMetadata(
            word_count=len(full_text.split()),
            char_count=len(full_text)
        )
        
        return ParsedDocument(
            document_type=DocumentType.XLSX,
            metadata=metadata,
            sections=sections,
            tables=tables,
            full_text=full_text
        )
    
    def _parse_html(self, content: bytes) -> ParsedDocument:
        """解析HTML文件"""
        text = content.decode('utf-8', errors='ignore')
        
        # 移除HTML标签
        clean_text = re.sub(r'<[^>]+>', ' ', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        sections = [DocumentSection(
            section_id=0,
            content_type=ContentType.TEXT,
            content=clean_text,
            page_number=1
        )]
        
        metadata = DocumentMetadata(
            word_count=len(clean_text.split()),
            char_count=len(clean_text)
        )
        
        return ParsedDocument(
            document_type=DocumentType.HTML,
            metadata=metadata,
            sections=sections,
            tables=[],
            full_text=clean_text
        )
    
    def _parse_markdown(self, content: bytes) -> ParsedDocument:
        """解析Markdown文件"""
        text = content.decode('utf-8', errors='ignore')
        lines = text.split('\n')
        
        sections = []
        section_id = 0
        outline = []
        
        for line in lines:
            stripped = line.strip()
            
            # 检测标题
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2)
                
                sections.append(DocumentSection(
                    section_id=section_id,
                    content_type=ContentType.HEADING,
                    content=title,
                    level=level
                ))
                
                outline.append({
                    "level": level,
                    "title": title,
                    "section_id": section_id
                })
                
                section_id += 1
            
            # 检测代码块
            elif stripped.startswith('```'):
                sections.append(DocumentSection(
                    section_id=section_id,
                    content_type=ContentType.CODE,
                    content=stripped
                ))
                section_id += 1
            
            # 检测列表
            elif stripped.startswith(('- ', '* ', '1. ', '2. ')):
                sections.append(DocumentSection(
                    section_id=section_id,
                    content_type=ContentType.LIST,
                    content=stripped
                ))
                section_id += 1
            
            # 普通文本
            elif stripped:
                sections.append(DocumentSection(
                    section_id=section_id,
                    content_type=ContentType.TEXT,
                    content=stripped
                ))
                section_id += 1
        
        metadata = DocumentMetadata(
            word_count=len(text.split()),
            char_count=len(text)
        )
        
        return ParsedDocument(
            document_type=DocumentType.MARKDOWN,
            metadata=metadata,
            sections=sections,
            tables=[],
            full_text=text,
            outline=outline
        )
    
    def _parse_unknown(self, content: bytes) -> ParsedDocument:
        """解析未知类型"""
        text = content.decode('utf-8', errors='ignore')
        
        return ParsedDocument(
            document_type=DocumentType.UNKNOWN,
            metadata=DocumentMetadata(),
            sections=[DocumentSection(
                section_id=0,
                content_type=ContentType.TEXT,
                content=text[:10000]  # 限制长度
            )],
            tables=[],
            full_text=text,
            errors=["Unknown document type"]
        )
    
    def get_supported_types(self) -> List[str]:
        """获取支持的文档类型"""
        return [t.value for t in DocumentType]
    
    def extract_text(self, file_path: str) -> str:
        """提取纯文本"""
        result = self.parse(file_path)
        return result.full_text
    
    def extract_tables(self, file_path: str) -> List[TableData]:
        """提取表格"""
        result = self.parse(file_path)
        return result.tables
    
    def get_outline(self, file_path: str) -> List[Dict[str, Any]]:
        """获取文档大纲"""
        result = self.parse(file_path)
        
        if result.outline:
            return result.outline
        
        # 从标题节生成大纲
        outline = []
        for section in result.sections:
            if section.content_type == ContentType.HEADING:
                outline.append({
                    "level": section.level,
                    "title": section.content,
                    "section_id": section.section_id
                })
        
        return outline
