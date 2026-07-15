"""
百度网盘适配器模块

该模块提供百度网盘(Baidu Netdisk)的通道适配器实现，支持：
- OAuth2认证流程
- 文件列表获取
- 文件上传/下载（支持断点续传）
- 分享管理（创建、查询、取消）
- 离线下载
- 回收站操作
- 速率限制处理

API文档: https://pan.baidu.com/union/doc
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import mimetypes
import os
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, AsyncIterator, BinaryIO, Dict, List, Optional, Tuple, Union

import aiohttp
from aiohttp import ClientTimeout, FormData

from ...base import (
    ChannelAdapter,
    ChannelCapability,
    ChannelConfig,
    ConnectionState,
    MessagePriority,
    ReceiveResult,
    RetryConfig,
    SendResult,
)
from ...universal_message import (
    Attachment,
    AttachmentType,
    ChannelIdentity,
    MessageContent,
    MessageDirection,
    MessageMetadata,
    MessageType,
    UniversalMessage,
    UserIdentity,
)

logger = logging.getLogger(__name__)


class FileCategory(Enum):
    """文件分类"""
    ALL = 0         # 全部
    VIDEO = 1       # 视频
    AUDIO = 2       # 音频
    IMAGE = 3       # 图片
    DOCUMENT = 4    # 文档
    APPLICATION = 5 # 应用
    OTHER = 6       # 其他
    SEED = 7        # 种子


class ShareType(Enum):
    """分享类型"""
    PUBLIC = 0      # 公开分享
    PRIVATE = 1     # 私密分享


@dataclass
class BaiduPanConfig(ChannelConfig):
    """百度网盘配置类
    
    Attributes:
        app_key: 应用AppKey
        app_secret: 应用AppSecret
        access_token: 访问令牌
        refresh_token: 刷新令牌
        expires_at: 令牌过期时间
        base_url: API基础URL
        connect_timeout: 连接超时时间（秒）
        read_timeout: 读取超时时间（秒）
        max_retry: 最大重试次数
        chunk_size: 上传分片大小（字节）
        rate_limit_delay: 速率限制延迟（秒）
    """
    app_key: str = ""
    app_secret: str = ""
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_at: float = 0.0
    base_url: str = "https://pan.baidu.com/rest/2.0/xpan"
    connect_timeout: int = 30
    read_timeout: int = 60
    max_retry: int = 3
    chunk_size: int = 4 * 1024 * 1024  # 4MB
    rate_limit_delay: float = 1.0
    
    def __post_init__(self):
        if not self.app_key or not self.app_secret:
            raise BaiduPanConfigError("AppKey和AppSecret不能为空")


class BaiduPanConfigError(Exception):
    """百度网盘配置错误"""
    pass


class BaiduPanAPIError(Exception):
    """百度网盘API错误
    
    Attributes:
        code: 错误代码
        message: 错误信息
    """
    
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")
    
    @property
    def is_retryable(self) -> bool:
        """判断错误是否可重试"""
        # 速率限制错误
        if self.code == 4 or self.code == 31034:
            return True
        # 服务端错误
        return 500 <= self.code < 600
    
    @property
    def is_rate_limit(self) -> bool:
        """判断是否为速率限制错误"""
        return self.code == 4 or self.code == 31034


@dataclass
class PanFile:
    """网盘文件信息
    
    Attributes:
        fs_id: 文件ID
        path: 文件路径
        name: 文件名
        size: 文件大小
        is_dir: 是否为目录
        modify_time: 修改时间
        create_time: 创建时间
        md5: 文件MD5
        dlink: 下载链接
        category: 文件分类
    """
    fs_id: int
    path: str
    name: str
    size: int
    is_dir: bool
    modify_time: Optional[str] = None
    create_time: Optional[str] = None
    md5: Optional[str] = None
    dlink: Optional[str] = None
    category: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "fs_id": self.fs_id,
            "path": self.path,
            "name": self.name,
            "size": self.size,
            "is_dir": self.is_dir,
            "modify_time": self.modify_time,
            "create_time": self.create_time,
            "md5": self.md5,
            "dlink": self.dlink,
            "category": self.category,
        }
    
    @property
    def human_size(self) -> str:
        """获取人类可读的文件大小"""
        size = self.size
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} PB"


@dataclass
class ShareInfo:
    """分享信息
    
    Attributes:
        share_id: 分享ID
        share_uk: 分享用户ID
        short_url: 短链接
        link: 完整链接
        password: 提取码
        expiration: 过期时间
        file_count: 文件数量
    """
    share_id: str
    share_uk: str
    short_url: str
    link: str
    password: Optional[str] = None
    expiration: Optional[str] = None
    file_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "share_id": self.share_id,
            "share_uk": self.share_uk,
            "short_url": self.short_url,
            "link": self.link,
            "password": self.password,
            "expiration": self.expiration,
            "file_count": self.file_count,
        }


@dataclass
class OfflineDownloadTask:
    """离线下载任务
    
    Attributes:
        task_id: 任务ID
        url: 下载URL
        save_path: 保存路径
        status: 任务状态
        progress: 下载进度
    """
    task_id: str
    url: str
    save_path: str
    status: str = "pending"
    progress: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "url": self.url,
            "save_path": self.save_path,
            "status": self.status,
            "progress": self.progress,
        }


class BaiduPanAdapter(ChannelAdapter):
    """百度网盘适配器
    
    提供百度网盘的统一接口，支持OAuth2认证、文件管理、分享管理、
    离线下载等功能。
    
    Example:
        config = BaiduPanConfig(
            app_key="your-app-key",
            app_secret="your-app-secret",
            access_token="your-access-token",
            refresh_token="your-refresh-token",
        )
        adapter = BaiduPanAdapter(config)
        await adapter.connect()
        
        # 获取文件列表
        files = await adapter.list_files("/")
        
        # 上传文件
        await adapter.upload_file("/local/file.txt", "/remote/file.txt")
        
        # 创建分享
        share = await adapter.create_share(["/remote/file.txt"])
    """
    
    OAUTH_URL = "https://openapi.baidu.com/oauth/2.0"
    API_URL = "https://pan.baidu.com/rest/2.0/xpan"
    PCS_URL = "https://d.pcs.baidu.com/rest/2.0/pcs"
    
    def __init__(self, config: BaiduPanConfig) -> None:
        super().__init__(config)
        self._cfg = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_request_time: float = 0.0
    
    def _initialize_capabilities(self) -> None:
        """初始化适配器能力"""
        self._capabilities = {
            ChannelCapability.FILE_ATTACHMENTS,
            ChannelCapability.MEDIA_MESSAGES,
            ChannelCapability.CHANNEL_INFO,
        }
    
    async def _check_rate_limit(self):
        """检查并处理速率限制"""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._cfg.rate_limit_delay:
            await asyncio.sleep(self._cfg.rate_limit_delay - elapsed)
        self._last_request_time = time.time()
    
    async def _refresh_access_token(self) -> bool:
        """刷新访问令牌
        
        Returns:
            是否刷新成功
        """
        if not self._cfg.refresh_token:
            return False
        
        url = f"{self.OAUTH_URL}/token"
        params = {
            "grant_type": "refresh_token",
            "refresh_token": self._cfg.refresh_token,
            "client_id": self._cfg.app_key,
            "client_secret": self._cfg.app_secret,
        }
        
        await self._check_rate_limit()
        
        async with self._session.get(url, params=params) as response:
            result = await response.json()
        
        if "access_token" in result:
            self._cfg.access_token = result["access_token"]
            self._cfg.refresh_token = result.get("refresh_token", self._cfg.refresh_token)
            self._cfg.expires_at = time.time() + result.get("expires_in", 2592000)
            return True
        
        return False
    
    async def _ensure_token_valid(self):
        """确保访问令牌有效"""
        if time.time() >= self._cfg.expires_at - 300:  # 提前5分钟刷新
            if not await self._refresh_access_token():
                raise BaiduPanAPIError(-1, "Access token已过期且无法刷新")
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        base_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """发送API请求
        
        Args:
            method: HTTP方法
            endpoint: API端点
            params: URL参数
            data: 请求数据
            base_url: 基础URL
            
        Returns:
            API响应
        """
        if not self._session:
            raise BaiduPanAPIError(-1, "适配器未连接")
        
        await self._ensure_token_valid()
        await self._check_rate_limit()
        
        base = base_url or self._cfg.base_url
        url = f"{base}{endpoint}"
        
        if params is None:
            params = {}
        params["access_token"] = self._cfg.access_token
        
        timeout = ClientTimeout(
            connect=self._cfg.connect_timeout,
            total=self._cfg.read_timeout,
        )
        
        try:
            if method.upper() == "GET":
                async with self._session.get(
                    url=url,
                    params=params,
                    timeout=timeout,
                ) as response:
                    result = await response.json()
            else:
                async with self._session.post(
                    url=url,
                    params=params,
                    data=data,
                    timeout=timeout,
                ) as response:
                    result = await response.json()
            
            # 检查错误
            errno = result.get("errno", 0)
            if errno != 0:
                error = BaiduPanAPIError(errno, result.get("errmsg", "Unknown error"))
                if error.is_rate_limit:
                    # 速率限制，等待后重试
                    await asyncio.sleep(self._cfg.rate_limit_delay * 2)
                    return await self._make_request(method, endpoint, params, data, base_url)
                raise error
            
            return result
            
        except aiohttp.ClientError as e:
            raise BaiduPanAPIError(-1, f"请求错误: {e}")
    
    async def _connect_impl(self) -> bool:
        """实现连接逻辑"""
        try:
            timeout = ClientTimeout(
                connect=self._cfg.connect_timeout,
                total=self._cfg.read_timeout,
            )
            self._session = aiohttp.ClientSession(timeout=timeout)
            
            # 测试连接：获取用户信息
            await self.get_user_info()
            
            self._logger.info("已连接到百度网盘")
            return True
        except Exception as e:
            self._logger.error(f"连接百度网盘失败: {e}")
            return False
    
    async def _disconnect_impl(self) -> None:
        """实现断开连接逻辑"""
        if self._session:
            await self._session.close()
            self._session = None
        self._logger.info("已断开与百度网盘的连接")
    
    async def _health_check_impl(self) -> bool:
        """实现健康检查逻辑"""
        try:
            await self.get_user_info()
            return True
        except Exception:
            return False
    
    async def _send_impl(self, message: UniversalMessage, priority: MessagePriority) -> SendResult:
        """实现发送逻辑（用于上传文件）"""
        try:
            operation = message.get_context("operation", "upload")
            
            if operation == "upload":
                local_path = message.get_context("local_path")
                remote_path = message.get_context("remote_path")
                
                if not local_path or not remote_path:
                    return SendResult(
                        success=False,
                        error="缺少local_path或remote_path参数",
                        error_code="MISSING_PARAMS",
                    )
                
                result = await self.upload_file(local_path, remote_path)
                return SendResult(
                    success=True,
                    message_id=str(result.get("fs_id", "")),
                    timestamp=time.time(),
                )
            
            elif operation == "create_share":
                paths = message.get_context("paths", [])
                if not paths:
                    return SendResult(
                        success=False,
                        error="缺少paths参数",
                        error_code="MISSING_PARAMS",
                    )
                
                result = await self.create_share(paths)
                return SendResult(
                    success=True,
                    message_id=result.share_id,
                    timestamp=time.time(),
                )
            
            else:
                return SendResult(
                    success=False,
                    error=f"不支持的操作: {operation}",
                    error_code="UNSUPPORTED_OPERATION",
                )
                
        except BaiduPanAPIError as e:
            return SendResult(
                success=False,
                error=e.message,
                error_code=str(e.code),
            )
        except Exception as e:
            return SendResult(
                success=False,
                error=str(e),
                error_code="UNKNOWN_ERROR",
            )
    
    async def _receive_impl(self, payload: Optional[Dict]) -> ReceiveResult:
        """实现接收逻辑（用于下载文件）"""
        try:
            operation = payload.get("operation", "list") if payload else "list"
            messages = []
            
            if operation == "list":
                path = payload.get("path", "/") if payload else "/"
                files = await self.list_files(path)
                for file in files:
                    content = MessageContent(text=f"[{file.name}]")
                    metadata = MessageMetadata(
                        message_id=str(file.fs_id),
                        channel_id="baidu_pan",
                        timestamp=time.time(),
                        direction=MessageDirection.INBOUND,
                        message_type=MessageType.FILE if not file.is_dir else MessageType.DATA,
                    )
                    msg = UniversalMessage(content=content, metadata=metadata)
                    msg.set_context("file", file.to_dict())
                    messages.append(msg)
            
            elif operation == "download":
                remote_path = payload.get("remote_path") if payload else None
                local_path = payload.get("local_path") if payload else None
                
                if remote_path and local_path:
                    await self.download_file(remote_path, local_path)
                    content = MessageContent(text=f"[Downloaded: {remote_path}]")
                    metadata = MessageMetadata(
                        message_id=remote_path,
                        channel_id="baidu_pan",
                        timestamp=time.time(),
                        direction=MessageDirection.INBOUND,
                        message_type=MessageType.FILE,
                    )
                    msg = UniversalMessage(content=content, metadata=metadata)
                    msg.set_context("local_path", local_path)
                    messages.append(msg)
            
            return ReceiveResult(success=True, messages=messages)
        except Exception as e:
            return ReceiveResult(success=False, error=str(e))
    
    # ========== OAuth2认证 ==========
    
    @classmethod
    def get_authorization_url(
        cls,
        app_key: str,
        redirect_uri: str,
        scope: str = "basic,netdisk",
        display: str = "page",
    ) -> str:
        """获取授权URL
        
        Args:
            app_key: 应用AppKey
            redirect_uri: 回调地址
            scope: 权限范围
            display: 显示方式
            
        Returns:
            授权URL
        """
        params = {
            "response_type": "code",
            "client_id": app_key,
            "redirect_uri": redirect_uri,
            "scope": scope,
            "display": display,
        }
        query = urllib.parse.urlencode(params)
        return f"{cls.OAUTH_URL}/authorize?{query}"
    
    @classmethod
    async def exchange_code_for_token(
        cls,
        app_key: str,
        app_secret: str,
        code: str,
        redirect_uri: str,
    ) -> Dict[str, Any]:
        """用授权码换取令牌
        
        Args:
            app_key: 应用AppKey
            app_secret: 应用AppSecret
            code: 授权码
            redirect_uri: 回调地址
            
        Returns:
            令牌信息
        """
        url = f"{cls.OAUTH_URL}/token"
        params = {
            "grant_type": "authorization_code",
            "code": code,
            "client_id": app_key,
            "client_secret": app_secret,
            "redirect_uri": redirect_uri,
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                result = await response.json()
        
        return result
    
    # ========== 用户信息 ==========
    
    async def get_user_info(self) -> Dict[str, Any]:
        """获取用户信息
        
        Returns:
            用户信息
        """
        result = await self._make_request("GET", "/quota")
        return {
            "total": result.get("total", 0),
            "used": result.get("used", 0),
            "free": result.get("free", 0),
        }
    
    async def get_quota(self) -> Dict[str, Any]:
        """获取网盘容量信息
        
        Returns:
            容量信息
        """
        return await self.get_user_info()
    
    # ========== 文件操作 ==========
    
    async def list_files(
        self,
        path: str = "/",
        order: str = "time",
        desc: int = 1,
        page: int = 1,
        page_size: int = 100,
        folder_only: bool = False,
    ) -> List[PanFile]:
        """获取文件列表
        
        Args:
            path: 目录路径
            order: 排序方式
            desc: 是否降序
            page: 页码
            page_size: 每页数量
            folder_only: 仅显示文件夹
            
        Returns:
            文件列表
        """
        params = {
            "dir": path,
            "order": order,
            "desc": desc,
            "page": page,
            "num": page_size,
            "folder": 1 if folder_only else 0,
        }
        
        result = await self._make_request("GET", "/file", params=params)
        
        files = []
        for item in result.get("list", []):
            file = PanFile(
                fs_id=item.get("fs_id", 0),
                path=item.get("path", ""),
                name=item.get("server_filename", ""),
                size=item.get("size", 0),
                is_dir=item.get("isdir", 0) == 1,
                modify_time=item.get("server_mtime"),
                create_time=item.get("server_ctime"),
                md5=item.get("md5"),
                category=item.get("category", 0),
            )
            files.append(file)
        
        return files
    
    async def get_file_info(self, path: str) -> Optional[PanFile]:
        """获取文件信息
        
        Args:
            path: 文件路径
            
        Returns:
            文件信息
        """
        params = {
            "path": path,
            "dlink": 1,
        }
        
        result = await self._make_request("GET", "/multimedia", params=params)
        
        if not result.get("list"):
            return None
        
        item = result["list"][0]
        return PanFile(
            fs_id=item.get("fs_id", 0),
            path=item.get("path", ""),
            name=item.get("server_filename", ""),
            size=item.get("size", 0),
            is_dir=item.get("isdir", 0) == 1,
            modify_time=item.get("server_mtime"),
            create_time=item.get("server_ctime"),
            md5=item.get("md5"),
            dlink=item.get("dlink"),
            category=item.get("category", 0),
        )
    
    async def upload_file(
        self,
        local_path: str,
        remote_path: str,
        ondup: str = "overwrite",
    ) -> Dict[str, Any]:
        """上传文件
        
        Args:
            local_path: 本地文件路径
            remote_path: 远程文件路径
            ondup: 重名处理方式
            
        Returns:
            上传结果
        """
        file_size = os.path.getsize(local_path)
        
        # 小文件直接上传
        if file_size <= self._cfg.chunk_size:
            return await self._upload_small_file(local_path, remote_path, ondup)
        
        # 大文件分片上传
        return await self._upload_large_file(local_path, remote_path, ondup)
    
    async def _upload_small_file(
        self,
        local_path: str,
        remote_path: str,
        ondup: str,
    ) -> Dict[str, Any]:
        """上传小文件"""
        # 获取上传URL
        params = {
            "path": remote_path,
            "uploadid": "",
            "partseq": "0",
        }
        
        result = await self._make_request(
            "GET",
            "/pcs/superfile2",
            params=params,
            base_url=self.PCS_URL,
        )
        
        upload_url = result.get("server")
        
        # 上传文件
        with open(local_path, "rb") as f:
            data = f.read()
        
        form = FormData()
        form.add_field("file", data, filename=os.path.basename(local_path))
        
        await self._check_rate_limit()
        
        async with self._session.post(upload_url, data=form) as response:
            upload_result = await response.json()
        
        return {
            "success": True,
            "fs_id": upload_result.get("fs_id"),
            "md5": upload_result.get("md5"),
            "path": remote_path,
        }
    
    async def _upload_large_file(
        self,
        local_path: str,
        remote_path: str,
        ondup: str,
    ) -> Dict[str, Any]:
        """上传大文件（分片上传）"""
        # 预上传，获取uploadid
        params = {
            "path": remote_path,
            "size": os.path.getsize(local_path),
            "isdir": 0,
            "rtype": 3 if ondup == "overwrite" else 1,
        }
        
        result = await self._make_request("GET", "/precreate", params=params)
        upload_id = result.get("uploadid")
        block_list = result.get("block_list", [])
        
        # 分片上传
        md5_list = []
        with open(local_path, "rb") as f:
            for seq in block_list:
                chunk = f.read(self._cfg.chunk_size)
                chunk_md5 = hashlib.md5(chunk).hexdigest()
                md5_list.append(chunk_md5)
                
                # 上传分片
                upload_params = {
                    "path": remote_path,
                    "uploadid": upload_id,
                    "partseq": seq,
                }
                
                await self._check_rate_limit()
                
                form = FormData()
                form.add_field("file", chunk, filename="chunk")
                
                async with self._session.post(
                    f"{self.PCS_URL}/superfile2",
                    params={**upload_params, "access_token": self._cfg.access_token},
                    data=form,
                ) as response:
                    await response.json()
        
        # 创建文件
        create_params = {
            "path": remote_path,
            "size": os.path.getsize(local_path),
            "isdir": 0,
            "uploadid": upload_id,
            "block_list": json.dumps(md5_list),
            "rtype": 3 if ondup == "overwrite" else 1,
        }
        
        result = await self._make_request("POST", "/create", params=create_params)
        
        return {
            "success": True,
            "fs_id": result.get("fs_id"),
            "path": remote_path,
        }
    
    async def download_file(
        self,
        remote_path: str,
        local_path: str,
    ) -> Dict[str, Any]:
        """下载文件
        
        Args:
            remote_path: 远程文件路径
            local_path: 本地保存路径
            
        Returns:
            下载结果
        """
        # 获取下载链接
        file_info = await self.get_file_info(remote_path)
        if not file_info or not file_info.dlink:
            raise BaiduPanAPIError(-1, "无法获取下载链接")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # 下载文件
        await self._check_rate_limit()
        
        async with self._session.get(file_info.dlink) as response:
            with open(local_path, "wb") as f:
                async for chunk in response.content.iter_chunked(8192):
                    f.write(chunk)
        
        return {
            "success": True,
            "path": remote_path,
            "local_path": local_path,
            "size": file_info.size,
        }
    
    async def delete_file(self, path: str) -> bool:
        """删除文件
        
        Args:
            path: 文件路径
            
        Returns:
            是否成功
        """
        params = {
            "filelist": json.dumps([path]),
            "async": 0,
        }
        
        await self._make_request("POST", "/file", params=params)
        return True
    
    async def move_file(self, src_path: str, dst_path: str) -> bool:
        """移动文件
        
        Args:
            src_path: 源路径
            dst_path: 目标路径
            
        Returns:
            是否成功
        """
        params = {
            "filelist": json.dumps([{"path": src_path, "dest": dst_path, "newname": os.path.basename(dst_path)}]),
            "async": 0,
        }
        
        await self._make_request("POST", "/file", params=params)
        return True
    
    async def copy_file(self, src_path: str, dst_path: str) -> bool:
        """复制文件
        
        Args:
            src_path: 源路径
            dst_path: 目标路径
            
        Returns:
            是否成功
        """
        params = {
            "filelist": json.dumps([{"path": src_path, "dest": dst_path, "newname": os.path.basename(dst_path)}]),
            "async": 0,
        }
        
        await self._make_request("POST", "/file", params=params)
        return True
    
    async def create_folder(self, path: str) -> bool:
        """创建文件夹
        
        Args:
            path: 文件夹路径
            
        Returns:
            是否成功
        """
        params = {
            "path": path,
            "size": 0,
            "isdir": 1,
            "rtype": 1,
        }
        
        await self._make_request("POST", "/create", params=params)
        return True
    
    # ========== 分享管理 ==========
    
    async def create_share(
        self,
        paths: List[str],
        share_type: ShareType = ShareType.PRIVATE,
        password: Optional[str] = None,
        expiration: int = 0,
        description: str = "",
    ) -> ShareInfo:
        """创建分享
        
        Args:
            paths: 文件路径列表
            share_type: 分享类型
            password: 提取码（私密分享时）
            expiration: 有效期（秒，0表示永久）
            description: 描述
            
        Returns:
            分享信息
        """
        params = {
            "fid_list": json.dumps([await self._get_fsid(path) for path in paths]),
            "sch_id": 0,
            "description": description,
            "period": expiration,
            "channel": "weixin",
        }
        
        if share_type == ShareType.PRIVATE and password:
            params["pwd"] = password
        
        result = await self._make_request("POST", "/share/set", params=params)
        
        return ShareInfo(
            share_id=str(result.get("shareid", "")),
            share_uk=str(result.get("uk", "")),
            short_url=result.get("shortlink", ""),
            link=result.get("link", ""),
            password=result.get("pwd"),
            file_count=len(paths),
        )
    
    async def _get_fsid(self, path: str) -> int:
        """获取文件的fs_id"""
        file_info = await self.get_file_info(path)
        if not file_info:
            raise BaiduPanAPIError(-1, f"文件不存在: {path}")
        return file_info.fs_id
    
    async def list_shares(self, page: int = 1, page_size: int = 100) -> List[ShareInfo]:
        """获取分享列表
        
        Args:
            page: 页码
            page_size: 每页数量
            
        Returns:
            分享列表
        """
        params = {
            "page": page,
            "num": page_size,
        }
        
        result = await self._make_request("GET", "/share/list", params=params)
        
        shares = []
        for item in result.get("list", []):
            share = ShareInfo(
                share_id=str(item.get("shareId", "")),
                share_uk=str(item.get("uk", "")),
                short_url=item.get("shortlink", ""),
                link=item.get("link", ""),
                password=item.get("pwd"),
                expiration=item.get("expiredType"),
                file_count=item.get("fileCount", 0),
            )
            shares.append(share)
        
        return shares
    
    async def cancel_share(self, share_id: str) -> bool:
        """取消分享
        
        Args:
            share_id: 分享ID
            
        Returns:
            是否成功
        """
        params = {
            "shareid_list": json.dumps([share_id]),
        }
        
        await self._make_request("POST", "/share/cancel", params=params)
        return True
    
    # ========== 离线下载 ==========
    
    async def add_offline_task(
        self,
        url: str,
        save_path: str,
        filename: Optional[str] = None,
    ) -> OfflineDownloadTask:
        """添加离线下载任务
        
        Args:
            url: 下载URL
            save_path: 保存路径
            filename: 文件名
            
        Returns:
            下载任务
        """
        params = {
            "save_path": save_path,
            "source_url": url,
            "type": 0,  # 普通下载
        }
        
        if filename:
            params["file_name"] = filename
        
        result = await self._make_request("POST", "/services/cloud_dl", params=params)
        
        return OfflineDownloadTask(
            task_id=str(result.get("task_id", "")),
            url=url,
            save_path=save_path,
            status="pending",
        )
    
    async def list_offline_tasks(
        self,
        page: int = 1,
        page_size: int = 100,
    ) -> List[OfflineDownloadTask]:
        """获取离线下载任务列表
        
        Args:
            page: 页码
            page_size: 每页数量
            
        Returns:
            任务列表
        """
        params = {
            "start": (page - 1) * page_size,
            "limit": page_size,
        }
        
        result = await self._make_request("GET", "/services/cloud_dl", params=params)
        
        tasks = []
        for item in result.get("task_info", []):
            task = OfflineDownloadTask(
                task_id=str(item.get("task_id", "")),
                url=item.get("source_url", ""),
                save_path=item.get("save_path", ""),
                status=item.get("status", "unknown"),
                progress=item.get("progress", 0.0),
            )
            tasks.append(task)
        
        return tasks
    
    async def cancel_offline_task(self, task_id: str) -> bool:
        """取消离线下载任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否成功
        """
        params = {
            "task_id": task_id,
        }
        
        await self._make_request("POST", "/services/cloud_dl/cancel", params=params)
        return True
    
    # ========== 回收站操作 ==========
    
    async def list_recycle_files(self, page: int = 1, page_size: int = 100) -> List[PanFile]:
        """获取回收站文件列表
        
        Args:
            page: 页码
            page_size: 每页数量
            
        Returns:
            文件列表
        """
        params = {
            "start": (page - 1) * page_size,
            "limit": page_size,
        }
        
        result = await self._make_request("GET", "/recycle/list", params=params)
        
        files = []
        for item in result.get("list", []):
            file = PanFile(
                fs_id=item.get("fs_id", 0),
                path=item.get("path", ""),
                name=item.get("server_filename", ""),
                size=item.get("size", 0),
                is_dir=item.get("isdir", 0) == 1,
                modify_time=item.get("server_mtime"),
                create_time=item.get("server_ctime"),
                category=item.get("category", 0),
            )
            files.append(file)
        
        return files
    
    async def restore_recycle_file(self, fs_id: int) -> bool:
        """恢复回收站文件
        
        Args:
            fs_id: 文件ID
            
        Returns:
            是否成功
        """
        params = {
            "fidlist": json.dumps([fs_id]),
        }
        
        await self._make_request("POST", "/recycle/restore", params=params)
        return True
    
    async def delete_recycle_file(self, fs_id: int) -> bool:
        """彻底删除回收站文件
        
        Args:
            fs_id: 文件ID
            
        Returns:
            是否成功
        """
        params = {
            "fidlist": json.dumps([fs_id]),
        }
        
        await self._make_request("POST", "/recycle/delete", params=params)
        return True
    
    async def clear_recycle(self) -> bool:
        """清空回收站
        
        Returns:
            是否成功
        """
        await self._make_request("POST", "/recycle/clear")
        return True
    
    # ========== 工具方法 ==========
    
    def get_capabilities(self):
        """获取适配器能力"""
        return self._capabilities
    
    def supports_capability(self, capability) -> bool:
        """检查是否支持特定能力"""
        return capability in self._capabilities
    
    def __repr__(self) -> str:
        return f"BaiduPanAdapter(app_key={self._cfg.app_key[:8]}...)"


# ========== CLI测试代码 ==========

async def test_baidu_pan():
    """测试百度网盘适配器"""
    import os
    
    # 从环境变量获取配置
    app_key = os.environ.get("BAIDU_PAN_APP_KEY", "test-app-key")
    app_secret = os.environ.get("BAIDU_PAN_APP_SECRET", "test-app-secret")
    access_token = os.environ.get("BAIDU_PAN_ACCESS_TOKEN", "test-access-token")
    refresh_token = os.environ.get("BAIDU_PAN_REFRESH_TOKEN", "test-refresh-token")
    
    config = BaiduPanConfig(
        app_key=app_key,
        app_secret=app_secret,
        access_token=access_token,
        refresh_token=refresh_token,
    )
    
    adapter = BaiduPanAdapter(config)
    
    try:
        # 连接
        print("正在连接百度网盘...")
        connected = await adapter.connect()
        print(f"连接结果: {connected}")
        
        if not connected:
            print("连接失败，请检查配置")
            return
        
        # 健康检查
        healthy = await adapter.health_check()
        print(f"健康检查: {healthy}")
        
        # 获取容量信息
        print("\n获取容量信息:")
        quota = await adapter.get_quota()
        print(f"  总容量: {quota.get('total', 0) / (1024**3):.2f} GB")
        print(f"  已使用: {quota.get('used', 0) / (1024**3):.2f} GB")
        print(f"  可用: {quota.get('free', 0) / (1024**3):.2f} GB")
        
        # 获取文件列表
        print("\n获取根目录文件列表:")
        files = await adapter.list_files("/", page_size=10)
        for file in files[:5]:
            type_str = "[DIR]" if file.is_dir else "[FILE]"
            print(f"  {type_str} {file.name} ({file.human_size})")
        
        print("\n测试完成!")
        
    except Exception as e:
        print(f"测试出错: {e}")
    finally:
        await adapter.disconnect()


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # 运行测试
    asyncio.run(test_baidu_pan())
