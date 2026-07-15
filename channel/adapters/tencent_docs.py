"""
腾讯文档 (Tencent Docs) 集成适配器模块

基于腾讯文档开放平台的文档管理适配器，
支持文档/表格/幻灯片的创建、读写、权限管理和版本控制。

官方文档: https://docs.qq.com/open/wiki/

Author: AGI Framework Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import aiohttp

from ...base import (
    ChannelAdapter,
    ChannelCapability,
    ChannelConfig,
    ConnectionState,
    MessagePriority,
    ReceiveResult,
    SendResult,
)
from ...universal_message import UniversalMessage

logger = logging.getLogger(__name__)


# ============================================================
# 异常定义
# ============================================================

class TencentDocsError(Exception):
    """腾讯文档异常基类"""

    def __init__(
        self,
        code: int,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(f"[TencentDocsError {code}] {message}")

    @classmethod
    def from_response(cls, body: Dict[str, Any]) -> TencentDocsError:
        """从 API 响应构造异常"""
        return cls(
            code=body.get("ret", body.get("code", -1)),
            message=body.get("msg", body.get("msgZh", "未知错误")),
            details=body,
        )

    @property
    def is_auth_error(self) -> bool:
        """是否为鉴权错误"""
        return self.code in (1001, 1002, 1003, 1004, 1005, 1006, 1007)

    @property
    def is_rate_limited(self) -> bool:
        """是否被限流"""
        return self.code == 2004

    @property
    def is_retryable(self) -> bool:
        """是否可重试"""
        return self.is_rate_limited or self.code in (2001, 2002, 2003)


# ============================================================
# 配置
# ============================================================

@dataclass
class TencentDocsConfig(ChannelConfig):
    """腾讯文档适配器配置

    Attributes:
        app_id: 腾讯文档开放平台应用 ID
        app_secret: 腾讯文档开放平台应用密钥
        redirect_uri: OAuth2 回调地址
        api_base: API 基础地址
        token_url: Token 获取地址
    """

    app_id: str = ""
    app_secret: str = ""
    redirect_uri: str = ""
    api_base: str = "https://docs.qq.com/openapi"
    token_url: str = "https://docs.qq.com/oauth2/token"


# ============================================================
# 数据模型
# ============================================================

class DocType(str, Enum):
    """文档类型"""
    DOC = "doc"          # 在线文档
    SHEET = "sheet"      # 在线表格
    SLIDE = "slide"      # 在线幻灯片
    FOLDER = "folder"    # 文件夹


class PermissionRole(str, Enum):
    """权限角色"""
    OWNER = "owner"          # 拥有者
    ADMIN = "admin"          # 管理员
    EDITOR = "editor"        # 可编辑
    COMMENTER = "commenter"  # 可评论
    VIEWER = "viewer"        # 可阅读


@dataclass
class TencentDocument:
    """腾讯文档信息"""
    doc_id: str
    title: str
    doc_type: DocType = DocType.DOC
    creator: str = ""
    created_at: str = ""
    updated_at: str = ""
    url: str = ""
    version: int = 0
    folder_id: str = ""
    permissions: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> TencentDocument:
        """从 API 响应构造文档对象"""
        doc_type_str = data.get("docType", data.get("type", "doc"))
        try:
            doc_type = DocType(doc_type_str)
        except ValueError:
            doc_type = DocType.DOC

        return cls(
            doc_id=data.get("docId", data.get("id", "")),
            title=data.get("title", data.get("name", "")),
            doc_type=doc_type,
            creator=data.get("creator", data.get("createUser", "")),
            created_at=data.get("createTime", data.get("createdAt", "")),
            updated_at=data.get("updateTime", data.get("updatedAt", "")),
            url=data.get("url", ""),
            version=data.get("version", 0),
            folder_id=data.get("folderId", data.get("parentId", "")),
        )


@dataclass
class SheetInfo:
    """表格信息"""
    sheet_id: str
    title: str
    doc_id: str = ""
    row_count: int = 0
    col_count: int = 0
    frozen_row_count: int = 0
    frozen_col_count: int = 0
    index: int = 0

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> SheetInfo:
        return cls(
            sheet_id=data.get("sheetId", data.get("id", "")),
            title=data.get("title", data.get("name", "")),
            doc_id=data.get("docId", ""),
            row_count=data.get("rowCount", 0),
            col_count=data.get("colCount", 0),
            frozen_row_count=data.get("frozenRowCount", 0),
            frozen_col_count=data.get("frozenColCount", 0),
            index=data.get("index", 0),
        )


@dataclass
class CellData:
    """单元格数据"""
    row: int
    col: int
    value: str = ""
    formatted_value: str = ""
    formula: str = ""
    background_color: str = ""
    font_color: str = ""
    bold: bool = False
    italic: bool = False

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> CellData:
        return cls(
            row=data.get("row", 0),
            col=data.get("col", data.get("column", 0)),
            value=str(data.get("value", "")),
            formatted_value=str(data.get("formattedValue", data.get("text", ""))),
            formula=data.get("formula", ""),
            background_color=data.get("backgroundColor", ""),
            font_color=data.get("fontColor", ""),
            bold=data.get("bold", False),
            italic=data.get("italic", False),
        )


@dataclass
class Comment:
    """文档评论"""
    comment_id: str
    doc_id: str
    content: str
    author: str = ""
    author_name: str = ""
    created_at: str = ""
    resolved: bool = False
    reply_to: str = ""

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> Comment:
        return cls(
            comment_id=data.get("commentId", data.get("id", "")),
            doc_id=data.get("docId", ""),
            content=data.get("content", data.get("text", "")),
            author=data.get("author", data.get("userId", "")),
            author_name=data.get("authorName", data.get("userName", "")),
            created_at=data.get("createTime", ""),
            resolved=data.get("resolved", False),
            reply_to=data.get("replyTo", data.get("parentId", "")),
        )


@dataclass
class VersionInfo:
    """文档版本信息"""
    version_id: str
    doc_id: str
    version_num: int = 0
    title: str = ""
    description: str = ""
    author: str = ""
    created_at: str = ""
    size: int = 0

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> VersionInfo:
        return cls(
            version_id=data.get("versionId", data.get("id", "")),
            doc_id=data.get("docId", ""),
            version_num=data.get("versionNum", data.get("number", 0)),
            title=data.get("title", ""),
            description=data.get("description", ""),
            author=data.get("author", data.get("userId", "")),
            created_at=data.get("createTime", ""),
            size=data.get("size", 0),
        )


@dataclass
class SearchResult:
    """搜索结果"""
    doc_id: str
    title: str
    doc_type: DocType = DocType.DOC
    snippet: str = ""
    url: str = ""
    updated_at: str = ""
    creator: str = ""

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> SearchResult:
        doc_type_str = data.get("docType", "doc")
        try:
            doc_type = DocType(doc_type_str)
        except ValueError:
            doc_type = DocType.DOC

        return cls(
            doc_id=data.get("docId", data.get("id", "")),
            title=data.get("title", ""),
            doc_type=doc_type,
            snippet=data.get("snippet", data.get("highlight", "")),
            url=data.get("url", ""),
            updated_at=data.get("updateTime", ""),
            creator=data.get("creator", ""),
        )


# ============================================================
# 主适配器
# ============================================================

class TencentDocsAdapter(ChannelAdapter):
    """腾讯文档集成适配器

    功能:
        - OAuth2 授权流程
        - 文档操作: 创建/读取/更新/删除文档、表格、幻灯片
        - 权限管理: 设置文档权限
        - 表格操作: 读写单元格数据、获取表格信息
        - 评论操作: 添加/列出/删除评论
        - 版本历史: 列出版本、恢复版本
        - 搜索: 按关键词搜索文档
        - Webhook: 文档变更通知

    Example:
        config = TencentDocsConfig(
            channel_id="tencent_docs",
            app_id="xxx",
            app_secret="yyy",
        )
        adapter = TencentDocsAdapter(config)
        await adapter.connect()
        doc = await adapter.create_document("测试文档", DocType.DOC)
    """

    # API 路径常量
    _DOC_CREATE = "/v2/doc/create"
    _DOC_INFO = "/v2/doc/info"
    _DOC_UPDATE = "/v2/doc/update"
    _DOC_DELETE = "/v2/doc/delete"
    _DOC_READ = "/v2/doc/read"
    _SHEET_CREATE = "/v2/sheet/create"
    _SHEET_INFO = "/v2/sheet/info"
    _SHEET_READ = "/v2/sheet/read"
    _SHEET_WRITE = "/v2/sheet/write"
    _SLIDE_CREATE = "/v2/slide/create"
    _PERMISSION_SET = "/v2/permission/set"
    _PERMISSION_LIST = "/v2/permission/list"
    _COMMENT_ADD = "/v2/comment/add"
    _COMMENT_LIST = "/v2/comment/list"
    _COMMENT_DELETE = "/v2/comment/delete"
    _VERSION_LIST = "/v2/doc/versions"
    _VERSION_RESTORE = "/v2/doc/versions/restore"
    _SEARCH = "/v2/search"

    def __init__(self, config: TencentDocsConfig) -> None:
        super().__init__(config)
        self._cfg = config
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._session: Optional[aiohttp.ClientSession] = None
        self._webhook_secret: Optional[str] = None
        self._webhook_handlers: List[Callable[[Dict[str, Any]], Any]] = []

    # ----------------------------------------------------------
    # 能力声明
    # ----------------------------------------------------------

    def _initialize_capabilities(self) -> None:
        self._capabilities = {
            ChannelCapability.WEBHOOK_MODE,
            ChannelCapability.CHANNEL_INFO,
            ChannelCapability.USER_INFO,
            ChannelCapability.RATE_LIMITING,
            ChannelCapability.FILE_ATTACHMENTS,
        }

    # ----------------------------------------------------------
    # 连接生命周期
    # ----------------------------------------------------------

    async def _connect_impl(self) -> bool:
        """连接腾讯文档平台，获取 access_token"""
        try:
            await self._fetch_token()
            self._logger.info("成功连接腾讯文档平台, token 前缀: %s...", self._access_token[:16] if self._access_token else "")
            return True
        except Exception as exc:
            self._logger.error("连接腾讯文档平台失败: %s", exc)
            return False

    async def _disconnect_impl(self) -> None:
        """断开连接"""
        self._access_token = None
        self._token_expires_at = 0.0
        self._refresh_token = None
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        self._logger.info("已断开腾讯文档平台连接")

    async def _health_check_impl(self) -> bool:
        """健康检查"""
        try:
            if not self._access_token or time.time() >= self._token_expires_at:
                await self._fetch_token()
            return self._access_token is not None
        except Exception:
            return False

    # ----------------------------------------------------------
    # Token 管理 (OAuth2 Client Credentials)
    # ----------------------------------------------------------

    async def _fetch_token(self, force: bool = False) -> str:
        """获取或刷新 access_token

        使用 OAuth2 Client Credentials 模式获取 token。
        """
        now = time.time()
        if not force and self._access_token and now < self._token_expires_at - 300:
            return self._access_token

        url = self._cfg.token_url
        data = {
            "grant_type": "client_credentials",
            "client_id": self._cfg.app_id,
            "client_secret": self._cfg.app_secret,
        }

        async with self._get_session() as session:
            async with session.post(url, data=data) as resp:
                result = await resp.json()

        if "access_token" not in result:
            raise TencentDocsError(
                code=result.get("ret", -1),
                message=result.get("msg", "获取 Token 失败"),
                details=result,
            )

        self._access_token = result["access_token"]
        self._refresh_token = result.get("refresh_token")
        expire_in = result.get("expires_in", 7200)
        self._token_expires_at = now + expire_in - 300

        return self._access_token

    def get_authorization_url(self, state: str = "", scope: str = "doc") -> str:
        """生成 OAuth2 授权链接（用于用户授权场景）

        Args:
            state: 防 CSRF 状态参数
            scope: 授权范围

        Returns:
            授权跳转 URL
        """
        params = {
            "response_type": "code",
            "client_id": self._cfg.app_id,
            "redirect_uri": self._cfg.redirect_uri,
            "scope": scope,
        }
        if state:
            params["state"] = state

        return "https://docs.qq.com/oauth2/authorize?" + urllib.parse.urlencode(params)

    async def exchange_code_for_token(self, code: str) -> Dict[str, Any]:
        """用授权码换取 access_token

        Args:
            code: OAuth2 授权码

        Returns:
            Token 信息字典
        """
        url = self._cfg.token_url
        data = {
            "grant_type": "authorization_code",
            "client_id": self._cfg.app_id,
            "client_secret": self._cfg.app_secret,
            "code": code,
            "redirect_uri": self._cfg.redirect_uri,
        }

        async with self._get_session() as session:
            async with session.post(url, data=data) as resp:
                result = await resp.json()

        if "access_token" not in result:
            raise TencentDocsError.from_response(result)

        self._access_token = result["access_token"]
        self._refresh_token = result.get("refresh_token")
        self._token_expires_at = time.time() + result.get("expires_in", 7200) - 300

        return result

    # ----------------------------------------------------------
    # HTTP 请求封装
    # ----------------------------------------------------------

    def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 aiohttp 会话"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._config.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """发送 API 请求

        自动附加 access_token，处理 token 过期自动刷新。
        """
        await self._fetch_token()

        url = self._cfg.api_base + path
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

        if params:
            url += "?" + urllib.parse.urlencode(params)

        async with self._get_session() as session:
            if method.upper() == "GET":
                async with session.get(url, headers=headers) as resp:
                    result = await resp.json()
            elif method.upper() == "POST":
                async with session.post(url, headers=headers, json=body) as resp:
                    result = await resp.json()
            elif method.upper() == "PUT":
                async with session.put(url, headers=headers, json=body) as resp:
                    result = await resp.json()
            elif method.upper() == "DELETE":
                async with session.delete(url, headers=headers) as resp:
                    result = await resp.json()
            else:
                raise ValueError(f"不支持的 HTTP 方法: {method}")

        # 检查错误
        ret_code = result.get("ret", 0)
        if ret_code != 0:
            # token 过期自动刷新重试
            if ret_code in (1001, 1002):
                self._logger.warning("Token 过期，正在刷新重试...")
                await self._fetch_token(force=True)
                return await self._request(method, path, body, params)
            raise TencentDocsError.from_response(result)

        return result

    # ----------------------------------------------------------
    # 文档操作
    # ----------------------------------------------------------

    async def create_document(
        self,
        title: str,
        doc_type: DocType = DocType.DOC,
        folder_id: Optional[str] = None,
    ) -> TencentDocument:
        """创建文档

        Args:
            title: 文档标题
            doc_type: 文档类型
            folder_id: 目标文件夹 ID

        Returns:
            创建的文档信息
        """
        body: Dict[str, Any] = {"title": title, "type": doc_type.value}
        if folder_id:
            body["folderId"] = folder_id

        if doc_type == DocType.SHEET:
            result = await self._request("POST", self._SHEET_CREATE, body=body)
        elif doc_type == DocType.SLIDE:
            result = await self._request("POST", self._SLIDE_CREATE, body=body)
        else:
            result = await self._request("POST", self._DOC_CREATE, body=body)

        doc_data = result.get("data", result.get("result", {}))
        return TencentDocument.from_api(doc_data)

    async def get_document_info(self, doc_id: str) -> TencentDocument:
        """获取文档信息

        Args:
            doc_id: 文档 ID

        Returns:
            文档信息
        """
        result = await self._request("GET", self._DOC_INFO, params={"docId": doc_id})
        doc_data = result.get("data", result.get("result", {}))
        return TencentDocument.from_api(doc_data)

    async def read_document(self, doc_id: str) -> Dict[str, Any]:
        """读取文档内容

        Args:
            doc_id: 文档 ID

        Returns:
            文档内容字典
        """
        result = await self._request("GET", self._DOC_READ, params={"docId": doc_id})
        return result.get("data", result.get("result", {}))

    async def update_document(
        self,
        doc_id: str,
        title: Optional[str] = None,
        content: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """更新文档

        Args:
            doc_id: 文档 ID
            title: 新标题（可选）
            content: 文档内容（可选）

        Returns:
            是否成功
        """
        body: Dict[str, Any] = {"docId": doc_id}
        if title:
            body["title"] = title
        if content:
            body["content"] = content

        result = await self._request("POST", self._DOC_UPDATE, body=body)
        return result.get("ret", -1) == 0

    async def delete_document(self, doc_id: str) -> bool:
        """删除文档

        Args:
            doc_id: 文档 ID

        Returns:
            是否成功
        """
        result = await self._request("DELETE", self._DOC_DELETE, params={"docId": doc_id})
        return result.get("ret", -1) == 0

    # ----------------------------------------------------------
    # 表格操作
    # ----------------------------------------------------------

    async def get_sheet_info(self, doc_id: str) -> List[SheetInfo]:
        """获取表格工作表列表

        Args:
            doc_id: 表格文档 ID

        Returns:
            工作表信息列表
        """
        result = await self._request("GET", self._SHEET_INFO, params={"docId": doc_id})
        sheets = result.get("data", result.get("result", {}).get("sheets", []))
        return [SheetInfo.from_api(s) for s in sheets]

    async def read_sheet_cells(
        self,
        doc_id: str,
        sheet_id: str,
        range_ref: str = "",
        row_start: int = 0,
        row_end: int = 100,
        col_start: int = 0,
        col_end: int = 26,
    ) -> List[List[CellData]]:
        """读取表格单元格数据

        Args:
            doc_id: 表格文档 ID
            sheet_id: 工作表 ID
            range_ref: 范围引用（如 "A1:C10"），优先于行列参数
            row_start: 起始行
            row_end: 结束行
            col_start: 起始列
            col_end: 结束列

        Returns:
            二维单元格数据数组
        """
        params: Dict[str, Any] = {"docId": doc_id, "sheetId": sheet_id}
        if range_ref:
            params["range"] = range_ref
        else:
            params["rowStart"] = row_start
            params["rowEnd"] = row_end
            params["colStart"] = col_start
            params["colEnd"] = col_end

        result = await self._request("GET", self._SHEET_READ, params=params)
        rows_data = result.get("data", result.get("result", {}).get("values", []))

        cells: List[List[CellData]] = []
        for row_idx, row in enumerate(rows_data):
            cell_row: List[CellData] = []
            for col_idx, cell_val in enumerate(row):
                if isinstance(cell_val, dict):
                    cell_row.append(CellData.from_api(cell_val))
                else:
                    cell_row.append(CellData(
                        row=row_start + row_idx,
                        col=col_start + col_idx,
                        value=str(cell_val),
                    ))
            cells.append(cell_row)

        return cells

    async def write_sheet_cells(
        self,
        doc_id: str,
        sheet_id: str,
        cells: List[Dict[str, Any]],
    ) -> bool:
        """写入表格单元格数据

        Args:
            doc_id: 表格文档 ID
            sheet_id: 工作表 ID
            cells: 单元格数据列表，每条格式:
                   {"row": 0, "col": 0, "value": "hello"}

        Returns:
            是否成功
        """
        body = {
            "docId": doc_id,
            "sheetId": sheet_id,
            "values": cells,
        }
        result = await self._request("POST", self._SHEET_WRITE, body=body)
        return result.get("ret", -1) == 0

    # ----------------------------------------------------------
    # 权限管理
    # ----------------------------------------------------------

    async def set_document_permission(
        self,
        doc_id: str,
        user_id: str,
        role: PermissionRole,
    ) -> bool:
        """设置文档权限

        Args:
            doc_id: 文档 ID
            user_id: 用户 ID
            role: 权限角色

        Returns:
            是否成功
        """
        body = {
            "docId": doc_id,
            "userId": user_id,
            "role": role.value,
        }
        result = await self._request("POST", self._PERMISSION_SET, body=body)
        return result.get("ret", -1) == 0

    async def list_permissions(self, doc_id: str) -> List[Dict[str, Any]]:
        """获取文档权限列表

        Args:
            doc_id: 文档 ID

        Returns:
            权限列表
        """
        result = await self._request("GET", self._PERMISSION_LIST, params={"docId": doc_id})
        return result.get("data", result.get("result", {}).get("permissions", []))

    # ----------------------------------------------------------
    # 评论操作
    # ----------------------------------------------------------

    async def add_comment(
        self,
        doc_id: str,
        content: str,
        position: Optional[Dict[str, Any]] = None,
        reply_to: Optional[str] = None,
    ) -> Comment:
        """添加评论

        Args:
            doc_id: 文档 ID
            content: 评论内容
            position: 评论位置（可选）
            reply_to: 回复的评论 ID（可选）

        Returns:
            创建的评论
        """
        body: Dict[str, Any] = {"docId": doc_id, "content": content}
        if position:
            body["position"] = position
        if reply_to:
            body["replyTo"] = reply_to

        result = await self._request("POST", self._COMMENT_ADD, body=body)
        comment_data = result.get("data", result.get("result", {}))
        return Comment.from_api(comment_data)

    async def list_comments(self, doc_id: str, page_size: int = 50) -> List[Comment]:
        """列出文档评论

        Args:
            doc_id: 文档 ID
            page_size: 每页数量

        Returns:
            评论列表
        """
        result = await self._request("GET", self._COMMENT_LIST, params={"docId": doc_id, "pageSize": page_size})
        comments = result.get("data", result.get("result", {}).get("comments", []))
        return [Comment.from_api(c) for c in comments]

    async def delete_comment(self, doc_id: str, comment_id: str) -> bool:
        """删除评论

        Args:
            doc_id: 文档 ID
            comment_id: 评论 ID

        Returns:
            是否成功
        """
        body = {"docId": doc_id, "commentId": comment_id}
        result = await self._request("POST", self._COMMENT_DELETE, body=body)
        return result.get("ret", -1) == 0

    # ----------------------------------------------------------
    # 版本历史
    # ----------------------------------------------------------

    async def list_versions(self, doc_id: str) -> List[VersionInfo]:
        """获取文档版本历史

        Args:
            doc_id: 文档 ID

        Returns:
            版本列表
        """
        result = await self._request("GET", self._VERSION_LIST, params={"docId": doc_id})
        versions = result.get("data", result.get("result", {}).get("versions", []))
        return [VersionInfo.from_api(v) for v in versions]

    async def restore_version(self, doc_id: str, version_id: str) -> bool:
        """恢复文档到指定版本

        Args:
            doc_id: 文档 ID
            version_id: 版本 ID

        Returns:
            是否成功
        """
        body = {"docId": doc_id, "versionId": version_id}
        result = await self._request("POST", self._VERSION_RESTORE, body=body)
        return result.get("ret", -1) == 0

    # ----------------------------------------------------------
    # 搜索
    # ----------------------------------------------------------

    async def search_documents(
        self,
        keyword: str,
        doc_type: Optional[DocType] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Tuple[List[SearchResult], int]:
        """搜索文档

        Args:
            keyword: 搜索关键词
            doc_type: 文档类型过滤（可选）
            page: 页码
            page_size: 每页数量

        Returns:
            (搜索结果列表, 总数)
        """
        params: Dict[str, Any] = {
            "keyword": keyword,
            "page": page,
            "pageSize": page_size,
        }
        if doc_type:
            params["type"] = doc_type.value

        result = await self._request("GET", self._SEARCH, params=params)
        data = result.get("data", result.get("result", {}))
        items = data.get("items", data.get("list", []))
        total = data.get("total", 0)

        return [SearchResult.from_api(item) for item in items], total

    # ----------------------------------------------------------
    # Webhook 通知
    # ----------------------------------------------------------

    def set_webhook_secret(self, secret: str) -> None:
        """设置 Webhook 签名密钥

        Args:
            secret: 签名密钥
        """
        self._webhook_secret = secret

    def register_webhook_handler(
        self,
        handler: Callable[[Dict[str, Any]], Any],
    ) -> None:
        """注册 Webhook 事件处理器

        Args:
            handler: 事件处理回调
        """
        self._webhook_handlers.append(handler)

    def unregister_webhook_handler(
        self,
        handler: Callable[[Dict[str, Any]], Any],
    ) -> None:
        """注销 Webhook 事件处理器"""
        if handler in self._webhook_handlers:
            self._webhook_handlers.remove(handler)

    def verify_webhook_signature(self, headers: Dict[str, str], body: bytes) -> bool:
        """验证 Webhook 请求签名

        Args:
            headers: 请求头
            body: 请求体

        Returns:
            签名是否有效
        """
        if not self._webhook_secret:
            return True

        signature = headers.get("X-Docs-Signature", headers.get("x-docs-signature", ""))
        if not signature:
            return False

        expected = hmac.new(
            self._webhook_secret.encode("utf-8"),
            body,
            digestmod=hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(signature, expected)

    async def process_webhook_event(self, event: Dict[str, Any]) -> Optional[UniversalMessage]:
        """处理 Webhook 事件

        Args:
            event: 事件数据

        Returns:
            转换后的 UniversalMessage（如果有）
        """
        event_type = event.get("type", event.get("eventType", ""))
        self._logger.info("收到 Webhook 事件: %s", event_type)

        # 分发到注册的处理器
        for handler in self._webhook_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as exc:
                self._logger.error("Webhook 处理器执行失败: %s", exc)

        # 转换为 UniversalMessage
        doc_id = event.get("docId", "")
        if doc_id:
            from ...universal_message import (
                MessageContent,
                MessageMetadata,
                MessageDirection,
                MessageType,
            )
            content = MessageContent(
                text=f"[文档变更] {event_type}: {doc_id}",
            )
            metadata = MessageMetadata(
                message_id=f"docs_{int(time.time())}",
                channel_id="tencent_docs",
                timestamp=time.time(),
                direction=MessageDirection.INBOUND,
                message_type=MessageType.SYSTEM,
                raw_event=event,
            )
            message = UniversalMessage(content=content, metadata=metadata)
            message.set_context("doc_id", doc_id)
            message.set_context("event_type", event_type)
            return message

        return None

    # ----------------------------------------------------------
    # ChannelAdapter 抽象方法实现
    # ----------------------------------------------------------

    async def _send_impl(self, message: UniversalMessage, priority: MessagePriority) -> SendResult:
        """发送消息（文档适配器将消息解析为文档操作）"""
        try:
            text = message.content.get_primary_text() if message.content else ""
            doc_id = message.get_context("doc_id", "")

            if not doc_id:
                return SendResult(success=False, error="缺少 doc_id 上下文", error_code="MISSING_TARGET")

            # 根据上下文判断操作类型
            operation = message.get_context("operation", "comment")

            if operation == "comment":
                comment = await self.add_comment(doc_id, text)
                return SendResult(
                    success=True,
                    message_id=comment.comment_id,
                    timestamp=time.time(),
                )
            elif operation == "update":
                success = await self.update_document(doc_id, content={"text": text})
                return SendResult(success=success, message_id=f"upd_{int(time.time())}", timestamp=time.time())
            else:
                return SendResult(success=False, error=f"不支持的操作类型: {operation}", error_code="UNSUPPORTED_OP")

        except TencentDocsError as exc:
            return SendResult(success=False, error=exc.message, error_code=str(exc.code))
        except Exception as exc:
            return SendResult(success=False, error=str(exc), error_code=type(exc).__name__)

    async def _receive_impl(self, payload: Optional[Dict] = None) -> ReceiveResult:
        """接收文档事件"""
        if not payload:
            return ReceiveResult(success=False, error="无有效载荷")

        message = await self.process_webhook_event(payload)
        if message:
            return ReceiveResult(success=True, messages=[message], raw_payload=payload)
        return ReceiveResult(success=False, error="无法解析事件")

    async def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取用户信息"""
        try:
            result = await self._request("GET", "/v2/user/info", params={"userId": user_id})
            return result.get("data", result.get("result", {}))
        except TencentDocsError as exc:
            self._logger.warning("获取用户信息失败 (user=%s): %s", user_id, exc)
            return None

    async def get_channel_info(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """获取通道信息"""
        return {
            "channel_id": channel_id,
            "name": "腾讯文档",
            "api_base": self._cfg.api_base,
        }

    # ----------------------------------------------------------
    # 辅助方法
    # ----------------------------------------------------------

    def get_config(self) -> TencentDocsConfig:
        """获取当前配置"""
        return self._cfg

    async def test_connection(self) -> Dict[str, Any]:
        """测试连接"""
        try:
            token = await self._fetch_token()
            return {
                "status": "ok",
                "token_prefix": token[:16] + "...",
                "token_expires_in": int(self._token_expires_at - time.time()),
            }
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def __repr__(self) -> str:
        return (
            f"TencentDocsAdapter("
            f"channel_id={self._config.channel_id!r}, "
            f"api_base={self._cfg.api_base!r}, "
            f"state={self._state.name})"
        )
