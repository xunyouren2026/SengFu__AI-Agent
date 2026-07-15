"""
腾讯云COS适配器模块

该模块提供腾讯云对象存储服务(COS)的通道适配器实现，支持：
- Bucket列表与管理
- 文件上传/下载（支持流式上传）
- 分片上传（Multipart Upload）
- 预签名URL生成
- CDN集成
- 生命周期管理
- 跨域配置

API文档: https://cloud.tencent.com/document/product/436/7751
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import mimetypes
import os
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, BinaryIO, Dict, List, Optional, Tuple, Union
from xml.etree import ElementTree as ET

import aiohttp
from aiohttp import ClientTimeout

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


@dataclass
class COSConfig(ChannelConfig):
    """腾讯云COS配置类
    
    Attributes:
        secret_id: 腾讯云SecretId
        secret_key: 腾讯云SecretKey
        endpoint: COS服务端点，如 cos.ap-guangzhou.myqcloud.com
        bucket_name: 默认Bucket名称
        region: 地域ID，如 ap-guangzhou
        use_https: 是否使用HTTPS
        connect_timeout: 连接超时时间（秒）
        read_timeout: 读取超时时间（秒）
        max_retry: 最大重试次数
        enable_crc: 是否启用CRC校验
        max_part_size: 分片上传的最大分片大小（字节）
        default_expire: 预签名URL默认过期时间（秒）
        cdn_domain: CDN加速域名（可选）
    """
    secret_id: str = ""
    secret_key: str = ""
    endpoint: str = "cos.ap-guangzhou.myqcloud.com"
    bucket_name: Optional[str] = None
    region: str = "ap-guangzhou"
    use_https: bool = True
    connect_timeout: int = 30
    read_timeout: int = 60
    max_retry: int = 3
    enable_crc: bool = True
    max_part_size: int = 100 * 1024 * 1024  # 100MB
    default_expire: int = 3600  # 1小时
    cdn_domain: Optional[str] = None
    
    def __post_init__(self):
        if not self.secret_id or not self.secret_key:
            raise COSConfigError("SecretId和SecretKey不能为空")


class COSConfigError(Exception):
    """COS配置错误"""
    pass


class COSAPIError(Exception):
    """COS API错误
    
    Attributes:
        code: 错误代码
        message: 错误信息
        request_id: 请求ID
        status_code: HTTP状态码
    """
    
    def __init__(
        self,
        code: str,
        message: str,
        request_id: Optional[str] = None,
        status_code: int = 0,
    ):
        self.code = code
        self.message = message
        self.request_id = request_id
        self.status_code = status_code
        super().__init__(f"[{code}] {message} (RequestId: {request_id})")
    
    @property
    def is_retryable(self) -> bool:
        """判断错误是否可重试"""
        retryable_codes = [
            "RequestTimeout",
            "InternalError",
            "ServiceUnavailable",
            "SlowDown",
        ]
        return self.code in retryable_codes or self.status_code in [500, 502, 503, 504]


class COSObject:
    """COS对象信息类
    
    Attributes:
        key: 对象键名
        size: 对象大小（字节）
        last_modified: 最后修改时间
        etag: ETag标识
        storage_class: 存储类型
        owner: 所有者信息
    """
    
    def __init__(self, data: Dict[str, Any]):
        self.key = data.get("Key", "")
        self.size = int(data.get("Size", 0))
        self.last_modified = data.get("LastModified", "")
        self.etag = data.get("ETag", "").strip('"')
        self.storage_class = data.get("StorageClass", "STANDARD")
        self.owner = data.get("Owner", {})
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "key": self.key,
            "size": self.size,
            "last_modified": self.last_modified,
            "etag": self.etag,
            "storage_class": self.storage_class,
            "owner": self.owner,
        }


class COSBucket:
    """COS Bucket信息类
    
    Attributes:
        name: Bucket名称
        creation_date: 创建时间
        location: 地域
    """
    
    def __init__(self, data: Dict[str, Any]):
        self.name = data.get("Name", "")
        self.creation_date = data.get("CreationDate", "")
        self.location = data.get("Location", "")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "creation_date": self.creation_date,
            "location": self.location,
        }


class COSMultipartUpload:
    """COS分片上传任务类
    
    Attributes:
        upload_id: 上传任务ID
        bucket: Bucket名称
        key: 对象键名
        initiated: 初始化时间
        parts: 已上传的分片列表
    """
    
    def __init__(self, upload_id: str, bucket: str, key: str, initiated: str = ""):
        self.upload_id = upload_id
        self.bucket = bucket
        self.key = key
        self.initiated = initiated
        self.parts: List[Dict[str, Any]] = []
    
    def add_part(self, part_number: int, etag: str, size: int = 0):
        """添加已上传的分片信息"""
        self.parts.append({
            "PartNumber": part_number,
            "ETag": etag,
            "Size": size,
        })
    
    def to_xml(self) -> str:
        """生成分片列表XML"""
        parts_xml = ""
        for part in sorted(self.parts, key=lambda x: x["PartNumber"]):
            parts_xml += f"""<Part>
<PartNumber>{part['PartNumber']}</PartNumber>
<ETag>{part['ETag']}</ETag>
</Part>"""
        return f"<?xml version=\"1.0\" encoding=\"UTF-8\"?><CompleteMultipartUpload>{parts_xml}</CompleteMultipartUpload>"


class TencentCOSAdapter(ChannelAdapter):
    """腾讯云COS适配器
    
    提供腾讯云对象存储服务的统一接口，支持文件上传下载、分片上传、
    预签名URL生成、CDN集成等功能。
    
    Example:
        config = COSConfig(
            secret_id="your-secret-id",
            secret_key="your-secret-key",
            endpoint="cos.ap-guangzhou.myqcloud.com",
            bucket_name="my-bucket",
        )
        adapter = TencentCOSAdapter(config)
        await adapter.connect()
        
        # 上传文件
        await adapter.upload_file("local/file.txt", "remote/file.txt")
        
        # 生成预签名URL
        url = await adapter.generate_presigned_url("remote/file.txt", expire=3600)
    """
    
    DEFAULT_ENDPOINT = "cos.ap-guangzhou.myqcloud.com"
    CHUNK_SIZE = 64 * 1024  # 64KB
    
    def __init__(self, config: COSConfig) -> None:
        super().__init__(config)
        self._cfg = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._bucket_cache: Dict[str, Any] = {}
    
    def _initialize_capabilities(self) -> None:
        """初始化适配器能力"""
        self._capabilities = {
            ChannelCapability.FILE_ATTACHMENTS,
            ChannelCapability.MEDIA_MESSAGES,
            ChannelCapability.CHANNEL_INFO,
        }
    
    def _get_base_url(self, bucket: Optional[str] = None) -> str:
        """获取基础URL
        
        Args:
            bucket: Bucket名称，为None则使用默认Bucket
            
        Returns:
            基础URL字符串
        """
        protocol = "https" if self._cfg.use_https else "http"
        bucket_name = bucket or self._cfg.bucket_name
        
        if bucket_name:
            return f"{protocol}://{bucket_name}.{self._cfg.endpoint}"
        return f"{protocol}://{self._cfg.endpoint}"
    
    def _get_cdn_url(self, object_key: str) -> Optional[str]:
        """获取CDN URL
        
        Args:
            object_key: 对象键名
            
        Returns:
            CDN URL或None
        """
        if not self._cfg.cdn_domain:
            return None
        return f"https://{self._cfg.cdn_domain}/{object_key}"
    
    def _sign_request(
        self,
        method: str,
        url_path: str,
        headers: Dict[str, str],
        query_params: Optional[Dict[str, str]] = None,
        expire: int = 0,
    ) -> str:
        """生成请求签名（COS签名算法）
        
        Args:
            method: HTTP方法
            url_path: URL路径
            headers: 请求头
            query_params: 查询参数
            expire: 过期时间
            
        Returns:
            签名字符串
        """
        # 获取当前时间戳
        now = int(time.time())
        
        # 签名有效期
        if expire == 0:
            expire = now + 3600
        
        # 构建签名密钥
        sign_time = f"{now};{expire}"
        sign_key = hmac.new(
            self._cfg.secret_key.encode("utf-8"),
            sign_time.encode("utf-8"),
            hashlib.sha1,
        ).hexdigest()
        
        # 构建HTTP字符串
        http_method = method.upper()
        http_uri = urllib.parse.quote(url_path, safe="/")
        
        # 处理查询参数
        http_parameters = ""
        if query_params:
            sorted_params = sorted(query_params.items())
            http_parameters = "&".join(f"{k}={urllib.parse.quote(str(v), safe='')}" for k, v in sorted_params)
        
        # 处理Headers
        http_headers = ""
        header_list = []
        if headers:
            sorted_headers = sorted(headers.items())
            header_parts = []
            for k, v in sorted_headers:
                k_lower = k.lower()
                if k_lower in ["content-type", "content-md5", "host"] or k_lower.startswith("x-cos-"):
                    header_parts.append(f"{k_lower}={urllib.parse.quote(str(v).strip(), safe='')}")
                    header_list.append(k_lower)
            http_headers = "&".join(header_parts)
        
        http_string = f"{http_method}\n{http_uri}\n{http_parameters}\n{http_headers}\n"
        
        # 构建待签名字符串
        string_to_sign = f"sha1\n{sign_time}\n{hashlib.sha1(http_string.encode('utf-8')).hexdigest()}\n"
        
        # 计算签名
        signature = hmac.new(
            sign_key.encode("utf-8"),
            string_to_sign.encode("utf-8"),
            hashlib.sha1,
        ).hexdigest()
        
        return signature
    
    def _get_auth_headers(
        self,
        method: str,
        url_path: str,
        content_type: str = "",
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """获取认证头
        
        Args:
            method: HTTP方法
            url_path: URL路径
            content_type: 内容类型
            extra_headers: 额外头信息
            
        Returns:
            包含认证信息的请求头
        """
        now = int(time.time())
        expire = now + 3600
        
        headers = {
            "Host": f"{self._cfg.bucket_name}.{self._cfg.endpoint}" if self._cfg.bucket_name else self._cfg.endpoint,
        }
        
        if content_type:
            headers["Content-Type"] = content_type
        
        if extra_headers:
            headers.update(extra_headers)
        
        # 生成签名
        signature = self._sign_request(method, url_path, headers)
        
        # 构建Authorization头
        sign_time = f"{now};{expire}"
        header_list = ";".join(sorted(k.lower() for k in headers.keys() if k.lower().startswith("x-cos-") or k.lower() in ["content-type", "content-md5", "host"]))
        
        headers["Authorization"] = f"q-sign-algorithm=sha1&q-ak={self._cfg.secret_id}&q-sign-time={sign_time}&q-key-time={sign_time}&q-header-list={header_list}&q-url-param-list=&q-signature={signature}"
        
        return headers
    
    async def _make_request(
        self,
        method: str,
        url_path: str,
        bucket: Optional[str] = None,
        data: Optional[Union[bytes, BinaryIO]] = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
        stream: bool = False,
    ) -> Tuple[int, Dict[str, str], Union[bytes, AsyncIterator[bytes]]]:
        """发送HTTP请求
        
        Args:
            method: HTTP方法
            url_path: URL路径
            bucket: Bucket名称
            data: 请求数据
            headers: 请求头
            params: URL参数
            stream: 是否流式响应
            
        Returns:
            (状态码, 响应头, 响应体)元组
        """
        if not self._session:
            raise COSAPIError("NotConnected", "适配器未连接")
        
        base_url = self._get_base_url(bucket)
        url = f"{base_url}{url_path}"
        
        if params:
            query_string = urllib.parse.urlencode(params)
            url = f"{url}?{query_string}"
        
        auth_headers = self._get_auth_headers(method, url_path)
        if headers:
            auth_headers.update(headers)
        
        timeout = ClientTimeout(
            connect=self._cfg.connect_timeout,
            total=self._cfg.read_timeout,
        )
        
        try:
            async with self._session.request(
                method=method,
                url=url,
                headers=auth_headers,
                data=data,
                timeout=timeout,
            ) as response:
                if stream:
                    return response.status, dict(response.headers), response.content.iter_chunked(self.CHUNK_SIZE)
                
                body = await response.read()
                
                if response.status >= 400:
                    error_info = self._parse_error_response(body)
                    raise COSAPIError(
                        error_info.get("Code", "UnknownError"),
                        error_info.get("Message", "Unknown error"),
                        error_info.get("RequestId"),
                        response.status,
                    )
                
                return response.status, dict(response.headers), body
                
        except aiohttp.ClientError as e:
            raise COSAPIError("RequestError", str(e))
    
    def _parse_error_response(self, body: bytes) -> Dict[str, str]:
        """解析错误响应XML"""
        try:
            root = ET.fromstring(body)
            result = {}
            for child in root:
                result[child.tag] = child.text or ""
            return result
        except ET.ParseError:
            return {"Code": "ParseError", "Message": body.decode("utf-8", errors="ignore")}
    
    def _parse_list_buckets(self, body: bytes) -> List[COSBucket]:
        """解析Bucket列表响应"""
        try:
            root = ET.fromstring(body)
            buckets = []
            for bucket_elem in root.findall(".//Bucket"):
                bucket_data = {}
                for child in bucket_elem:
                    bucket_data[child.tag] = child.text or ""
                buckets.append(COSBucket(bucket_data))
            return buckets
        except ET.ParseError as e:
            logger.error(f"解析Bucket列表失败: {e}")
            return []
    
    def _parse_list_objects(self, body: bytes) -> Tuple[List[COSObject], bool, Optional[str]]:
        """解析对象列表响应
        
        Returns:
            (对象列表, 是否截断, 下一个Marker)
        """
        try:
            root = ET.fromstring(body)
            objects = []
            
            for obj_elem in root.findall(".//Contents"):
                obj_data = {}
                for child in obj_elem:
                    obj_data[child.tag] = child.text or ""
                objects.append(COSObject(obj_data))
            
            is_truncated_elem = root.find(".//IsTruncated")
            truncated = is_truncated_elem.text == "true" if is_truncated_elem is not None else False
            
            next_marker_elem = root.find(".//NextMarker")
            marker = next_marker_elem.text if next_marker_elem is not None else None
            
            return objects, truncated, marker
        except ET.ParseError as e:
            logger.error(f"解析对象列表失败: {e}")
            return [], False, None
    
    async def _connect_impl(self) -> bool:
        """实现连接逻辑"""
        try:
            timeout = ClientTimeout(
                connect=self._cfg.connect_timeout,
                total=self._cfg.read_timeout,
            )
            self._session = aiohttp.ClientSession(timeout=timeout)
            
            # 测试连接：获取Bucket列表
            await self.list_buckets()
            
            self._logger.info(f"已连接到腾讯云COS: {self._cfg.endpoint}")
            return True
        except Exception as e:
            self._logger.error(f"连接腾讯云COS失败: {e}")
            return False
    
    async def _disconnect_impl(self) -> None:
        """实现断开连接逻辑"""
        if self._session:
            await self._session.close()
            self._session = None
        self._bucket_cache.clear()
        self._logger.info("已断开与腾讯云COS的连接")
    
    async def _health_check_impl(self) -> bool:
        """实现健康检查逻辑"""
        try:
            await self.list_buckets()
            return True
        except Exception:
            return False
    
    async def _send_impl(self, message: UniversalMessage, priority: MessagePriority) -> SendResult:
        """实现发送逻辑（用于上传文件）"""
        try:
            bucket = message.get_context("bucket") or self._cfg.bucket_name
            object_key = message.get_context("object_key")
            file_path = message.get_context("file_path")
            
            if not bucket or not object_key:
                return SendResult(
                    success=False,
                    error="缺少bucket或object_key参数",
                    error_code="MISSING_PARAMS",
                )
            
            # 如果有附件，上传附件
            if message.content.attachments:
                for attachment in message.content.attachments:
                    if attachment.data:
                        await self.upload_bytes(attachment.data, object_key, bucket)
                    elif attachment.url:
                        await self.upload_file(attachment.url, object_key, bucket)
            elif file_path:
                await self.upload_file(file_path, object_key, bucket)
            else:
                # 上传文本内容
                content = message.content.get_primary_text().encode("utf-8")
                await self.upload_bytes(content, object_key, bucket)
            
            return SendResult(
                success=True,
                message_id=f"{bucket}/{object_key}",
                timestamp=time.time(),
            )
        except COSAPIError as e:
            return SendResult(
                success=False,
                error=e.message,
                error_code=e.code,
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
            bucket = payload.get("bucket") if payload else self._cfg.bucket_name
            object_key = payload.get("object_key") if payload else None
            prefix = payload.get("prefix") if payload else None
            
            if not bucket:
                return ReceiveResult(success=False, error="缺少bucket参数")
            
            messages = []
            
            if object_key:
                # 下载单个对象
                data = await self.download_bytes(object_key, bucket)
                content = MessageContent(text=f"[COS Object: {object_key}]")
                metadata = MessageMetadata(
                    message_id=f"{bucket}/{object_key}",
                    channel_id="tencent_cos",
                    timestamp=time.time(),
                    direction=MessageDirection.INBOUND,
                    message_type=MessageType.FILE,
                )
                msg = UniversalMessage(content=content, metadata=metadata)
                msg.set_context("bucket", bucket)
                msg.set_context("object_key", object_key)
                msg.set_context("data", data)
                messages.append(msg)
            elif prefix is not None:
                # 列出对象
                objects = await self.list_objects(bucket, prefix=prefix)
                for obj in objects:
                    content = MessageContent(text=f"[COS Object: {obj.key}]")
                    metadata = MessageMetadata(
                        message_id=f"{bucket}/{obj.key}",
                        channel_id="tencent_cos",
                        timestamp=time.time(),
                        direction=MessageDirection.INBOUND,
                        message_type=MessageType.FILE,
                    )
                    msg = UniversalMessage(content=content, metadata=metadata)
                    msg.set_context("bucket", bucket)
                    msg.set_context("object_key", obj.key)
                    msg.set_context("object_info", obj.to_dict())
                    messages.append(msg)
            
            return ReceiveResult(success=True, messages=messages)
        except Exception as e:
            return ReceiveResult(success=False, error=str(e))
    
    # ========== Bucket操作 ==========
    
    async def list_buckets(self) -> List[COSBucket]:
        """获取Bucket列表
        
        Returns:
            Bucket列表
        """
        status, headers, body = await self._make_request("GET", "/", bucket=None)
        return self._parse_list_buckets(body)
    
    async def create_bucket(
        self,
        bucket_name: str,
        acl: str = "private",
    ) -> bool:
        """创建Bucket
        
        Args:
            bucket_name: Bucket名称
            acl: 访问权限 (private/public-read/public-read-write)
            
        Returns:
            是否创建成功
        """
        headers = {
            "x-cos-acl": acl,
        }
        
        try:
            await self._make_request(
                "PUT",
                "/",
                bucket=bucket_name,
                headers=headers,
            )
            return True
        except COSAPIError as e:
            if e.code == "BucketAlreadyExists":
                return True
            raise
    
    async def delete_bucket(self, bucket_name: str) -> bool:
        """删除Bucket
        
        Args:
            bucket_name: Bucket名称
            
        Returns:
            是否删除成功
        """
        try:
            await self._make_request("DELETE", "/", bucket=bucket_name)
            return True
        except COSAPIError as e:
            if e.code == "NoSuchBucket":
                return True
            raise
    
    async def get_bucket_info(self, bucket_name: str) -> Dict[str, Any]:
        """获取Bucket信息
        
        Args:
            bucket_name: Bucket名称
            
        Returns:
            Bucket信息字典
        """
        status, headers, body = await self._make_request(
            "GET",
            "/",
            bucket=bucket_name,
            params={"bucketInfo": ""},
        )
        # 解析XML响应
        try:
            root = ET.fromstring(body)
            info = {}
            for elem in root.iter():
                if elem.text:
                    info[elem.tag] = elem.text
            return info
        except ET.ParseError:
            return {}
    
    # ========== 对象操作 ==========
    
    async def list_objects(
        self,
        bucket: Optional[str] = None,
        prefix: str = "",
        marker: str = "",
        max_keys: int = 1000,
        delimiter: str = "",
    ) -> List[COSObject]:
        """列出对象
        
        Args:
            bucket: Bucket名称
            prefix: 前缀过滤
            marker: 分页标记
            max_keys: 最大返回数量
            delimiter: 分隔符
            
        Returns:
            对象列表
        """
        bucket_name = bucket or self._cfg.bucket_name
        if not bucket_name:
            raise COSAPIError("MissingBucket", "未指定Bucket名称")
        
        params = {
            "prefix": prefix,
            "max-keys": str(max_keys),
        }
        if marker:
            params["marker"] = marker
        if delimiter:
            params["delimiter"] = delimiter
        
        status, headers, body = await self._make_request(
            "GET",
            "/",
            bucket=bucket_name,
            params=params,
        )
        
        objects, truncated, next_marker = self._parse_list_objects(body)
        return objects
    
    async def upload_file(
        self,
        local_path: str,
        object_key: str,
        bucket: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """上传文件
        
        Args:
            local_path: 本地文件路径
            object_key: COS对象键名
            bucket: Bucket名称
            metadata: 自定义元数据
            content_type: 内容类型
            
        Returns:
            上传结果信息
        """
        bucket_name = bucket or self._cfg.bucket_name
        if not bucket_name:
            raise COSAPIError("MissingBucket", "未指定Bucket名称")
        
        if not content_type:
            content_type, _ = mimetypes.guess_type(local_path)
            content_type = content_type or "application/octet-stream"
        
        file_size = os.path.getsize(local_path)
        
        # 大文件使用分片上传
        if file_size > self._cfg.max_part_size:
            return await self.multipart_upload(local_path, object_key, bucket_name, metadata)
        
        # 小文件直接上传
        headers = {"Content-Type": content_type}
        if metadata:
            for key, value in metadata.items():
                headers[f"x-cos-meta-{key}"] = value
        
        with open(local_path, "rb") as f:
            data = f.read()
        
        status, resp_headers, body = await self._make_request(
            "PUT",
            f"/{urllib.parse.quote(object_key, safe='/')}",
            bucket=bucket_name,
            data=data,
            headers=headers,
        )
        
        return {
            "success": True,
            "bucket": bucket_name,
            "key": object_key,
            "etag": resp_headers.get("ETag", "").strip('"'),
            "size": file_size,
        }
    
    async def upload_bytes(
        self,
        data: bytes,
        object_key: str,
        bucket: Optional[str] = None,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """上传字节数据
        
        Args:
            data: 字节数据
            object_key: COS对象键名
            bucket: Bucket名称
            content_type: 内容类型
            metadata: 自定义元数据
            
        Returns:
            上传结果信息
        """
        bucket_name = bucket or self._cfg.bucket_name
        if not bucket_name:
            raise COSAPIError("MissingBucket", "未指定Bucket名称")
        
        headers = {"Content-Type": content_type}
        if metadata:
            for key, value in metadata.items():
                headers[f"x-cos-meta-{key}"] = value
        
        status, resp_headers, body = await self._make_request(
            "PUT",
            f"/{urllib.parse.quote(object_key, safe='/')}",
            bucket=bucket_name,
            data=data,
            headers=headers,
        )
        
        return {
            "success": True,
            "bucket": bucket_name,
            "key": object_key,
            "etag": resp_headers.get("ETag", "").strip('"'),
            "size": len(data),
        }
    
    async def upload_stream(
        self,
        stream: BinaryIO,
        object_key: str,
        bucket: Optional[str] = None,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """流式上传
        
        Args:
            stream: 文件流
            object_key: COS对象键名
            bucket: Bucket名称
            content_type: 内容类型
            metadata: 自定义元数据
            
        Returns:
            上传结果信息
        """
        data = stream.read()
        return await self.upload_bytes(data, object_key, bucket, content_type, metadata)
    
    async def download_file(
        self,
        object_key: str,
        local_path: str,
        bucket: Optional[str] = None,
    ) -> Dict[str, Any]:
        """下载文件到本地
        
        Args:
            object_key: COS对象键名
            local_path: 本地保存路径
            bucket: Bucket名称
            
        Returns:
            下载结果信息
        """
        bucket_name = bucket or self._cfg.bucket_name
        if not bucket_name:
            raise COSAPIError("MissingBucket", "未指定Bucket名称")
        
        status, headers, body = await self._make_request(
            "GET",
            f"/{urllib.parse.quote(object_key, safe='/')}",
            bucket=bucket_name,
        )
        
        # 确保目录存在
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        with open(local_path, "wb") as f:
            f.write(body)
        
        return {
            "success": True,
            "bucket": bucket_name,
            "key": object_key,
            "local_path": local_path,
            "size": len(body),
            "content_type": headers.get("Content-Type", "application/octet-stream"),
        }
    
    async def download_bytes(
        self,
        object_key: str,
        bucket: Optional[str] = None,
    ) -> bytes:
        """下载对象为字节
        
        Args:
            object_key: COS对象键名
            bucket: Bucket名称
            
        Returns:
            对象字节数据
        """
        bucket_name = bucket or self._cfg.bucket_name
        if not bucket_name:
            raise COSAPIError("MissingBucket", "未指定Bucket名称")
        
        status, headers, body = await self._make_request(
            "GET",
            f"/{urllib.parse.quote(object_key, safe='/')}",
            bucket=bucket_name,
        )
        
        return body
    
    async def download_stream(
        self,
        object_key: str,
        bucket: Optional[str] = None,
    ) -> AsyncIterator[bytes]:
        """流式下载对象
        
        Args:
            object_key: COS对象键名
            bucket: Bucket名称
            
        Yields:
            数据块
        """
        bucket_name = bucket or self._cfg.bucket_name
        if not bucket_name:
            raise COSAPIError("MissingBucket", "未指定Bucket名称")
        
        status, headers, stream = await self._make_request(
            "GET",
            f"/{urllib.parse.quote(object_key, safe='/')}",
            bucket=bucket_name,
            stream=True,
        )
        
        async for chunk in stream:
            yield chunk
    
    async def delete_object(self, object_key: str, bucket: Optional[str] = None) -> bool:
        """删除对象
        
        Args:
            object_key: COS对象键名
            bucket: Bucket名称
            
        Returns:
            是否删除成功
        """
        bucket_name = bucket or self._cfg.bucket_name
        if not bucket_name:
            raise COSAPIError("MissingBucket", "未指定Bucket名称")
        
        try:
            await self._make_request(
                "DELETE",
                f"/{urllib.parse.quote(object_key, safe='/')}",
                bucket=bucket_name,
            )
            return True
        except COSAPIError as e:
            if e.code == "NoSuchKey":
                return True
            raise
    
    async def get_object_info(
        self,
        object_key: str,
        bucket: Optional[str] = None,
    ) -> Dict[str, Any]:
        """获取对象元信息
        
        Args:
            object_key: COS对象键名
            bucket: Bucket名称
            
        Returns:
            对象元信息
        """
        bucket_name = bucket or self._cfg.bucket_name
        if not bucket_name:
            raise COSAPIError("MissingBucket", "未指定Bucket名称")
        
        status, headers, body = await self._make_request(
            "HEAD",
            f"/{urllib.parse.quote(object_key, safe='/')}",
            bucket=bucket_name,
        )
        
        return {
            "bucket": bucket_name,
            "key": object_key,
            "size": int(headers.get("Content-Length", 0)),
            "content_type": headers.get("Content-Type", "application/octet-stream"),
            "etag": headers.get("ETag", "").strip('"'),
            "last_modified": headers.get("Last-Modified", ""),
        }
    
    async def copy_object(
        self,
        source_key: str,
        dest_key: str,
        source_bucket: Optional[str] = None,
        dest_bucket: Optional[str] = None,
    ) -> Dict[str, Any]:
        """复制对象
        
        Args:
            source_key: 源对象键名
            dest_key: 目标对象键名
            source_bucket: 源Bucket
            dest_bucket: 目标Bucket
            
        Returns:
            复制结果
        """
        src_bucket = source_bucket or self._cfg.bucket_name
        dst_bucket = dest_bucket or self._cfg.bucket_name
        
        if not src_bucket or not dst_bucket:
            raise COSAPIError("MissingBucket", "未指定Bucket名称")
        
        source = f"{src_bucket}.cos.{self._cfg.region}.myqcloud.com/{urllib.parse.quote(source_key, safe='/')}"
        headers = {"x-cos-copy-source": source}
        
        status, resp_headers, body = await self._make_request(
            "PUT",
            f"/{urllib.parse.quote(dest_key, safe='/')}",
            bucket=dst_bucket,
            headers=headers,
        )
        
        return {
            "success": True,
            "source": f"{src_bucket}/{source_key}",
            "destination": f"{dst_bucket}/{dest_key}",
        }
    
    # ========== 分片上传 ==========
    
    async def init_multipart_upload(
        self,
        object_key: str,
        bucket: Optional[str] = None,
        content_type: str = "application/octet-stream",
    ) -> COSMultipartUpload:
        """初始化分片上传
        
        Args:
            object_key: COS对象键名
            bucket: Bucket名称
            content_type: 内容类型
            
        Returns:
            分片上传任务对象
        """
        bucket_name = bucket or self._cfg.bucket_name
        if not bucket_name:
            raise COSAPIError("MissingBucket", "未指定Bucket名称")
        
        headers = {"Content-Type": content_type}
        
        status, resp_headers, body = await self._make_request(
            "POST",
            f"/{urllib.parse.quote(object_key, safe='/')}",
            bucket=bucket_name,
            params={"uploads": ""},
            headers=headers,
        )
        
        # 解析响应获取UploadId
        try:
            root = ET.fromstring(body)
            upload_id = root.find(".//UploadId")
            upload_id_text = upload_id.text if upload_id is not None else ""
            return COSMultipartUpload(upload_id_text, bucket_name, object_key)
        except ET.ParseError as e:
            raise COSAPIError("ParseError", f"解析响应失败: {e}")
    
    async def upload_part(
        self,
        upload: COSMultipartUpload,
        part_number: int,
        data: bytes,
    ) -> str:
        """上传分片
        
        Args:
            upload: 分片上传任务
            part_number: 分片序号（1-10000）
            data: 分片数据
            
        Returns:
            分片ETag
        """
        params = {
            "partNumber": str(part_number),
            "uploadId": upload.upload_id,
        }
        
        status, headers, body = await self._make_request(
            "PUT",
            f"/{urllib.parse.quote(upload.key, safe='/')}",
            bucket=upload.bucket,
            params=params,
            data=data,
        )
        
        etag = headers.get("ETag", "").strip('"')
        upload.add_part(part_number, etag, len(data))
        return etag
    
    async def complete_multipart_upload(
        self,
        upload: COSMultipartUpload,
    ) -> Dict[str, Any]:
        """完成分片上传
        
        Args:
            upload: 分片上传任务
            
        Returns:
            完成结果
        """
        params = {"uploadId": upload.upload_id}
        xml_body = upload.to_xml()
        
        headers = {"Content-Type": "application/xml"}
        
        status, resp_headers, body = await self._make_request(
            "POST",
            f"/{urllib.parse.quote(upload.key, safe='/')}",
            bucket=upload.bucket,
            params=params,
            data=xml_body.encode("utf-8"),
            headers=headers,
        )
        
        return {
            "success": True,
            "bucket": upload.bucket,
            "key": upload.key,
            "etag": resp_headers.get("ETag", "").strip('"'),
        }
    
    async def abort_multipart_upload(self, upload: COSMultipartUpload) -> bool:
        """取消分片上传
        
        Args:
            upload: 分片上传任务
            
        Returns:
            是否取消成功
        """
        params = {"uploadId": upload.upload_id}
        
        await self._make_request(
            "DELETE",
            f"/{urllib.parse.quote(upload.key, safe='/')}",
            bucket=upload.bucket,
            params=params,
        )
        
        return True
    
    async def multipart_upload(
        self,
        local_path: str,
        object_key: str,
        bucket: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        part_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """执行完整的分片上传流程
        
        Args:
            local_path: 本地文件路径
            object_key: COS对象键名
            bucket: Bucket名称
            metadata: 自定义元数据
            part_size: 分片大小
            
        Returns:
            上传结果
        """
        part_size = part_size or self._cfg.max_part_size
        
        # 初始化分片上传
        upload = await self.init_multipart_upload(object_key, bucket)
        
        try:
            # 上传分片
            with open(local_path, "rb") as f:
                part_number = 1
                while True:
                    data = f.read(part_size)
                    if not data:
                        break
                    await self.upload_part(upload, part_number, data)
                    part_number += 1
            
            # 完成上传
            result = await self.complete_multipart_upload(upload)
            return result
            
        except Exception as e:
            # 出错时取消上传
            await self.abort_multipart_upload(upload)
            raise
    
    # ========== 预签名URL ==========
    
    def generate_presigned_url(
        self,
        object_key: str,
        bucket: Optional[str] = None,
        expire: int = 3600,
        method: str = "GET",
    ) -> str:
        """生成预签名URL
        
        Args:
            object_key: COS对象键名
            bucket: Bucket名称
            expire: 过期时间（秒）
            method: HTTP方法
            
        Returns:
            预签名URL
        """
        bucket_name = bucket or self._cfg.bucket_name
        if not bucket_name:
            raise COSAPIError("MissingBucket", "未指定Bucket名称")
        
        protocol = "https" if self._cfg.use_https else "http"
        host = f"{bucket_name}.{self._cfg.endpoint}"
        url_path = f"/{object_key}"
        
        # 计算过期时间
        now = int(time.time())
        expires = now + expire
        
        # 构建查询参数
        params = {}
        
        # 生成签名
        headers = {"Host": host}
        signature = self._sign_request(method, url_path, headers, params, expires)
        
        # 构建URL
        sign_time = f"{now};{expires}"
        query_string = urllib.parse.urlencode({
            "q-sign-algorithm": "sha1",
            "q-ak": self._cfg.secret_id,
            "q-sign-time": sign_time,
            "q-key-time": sign_time,
            "q-header-list": "host",
            "q-url-param-list": "",
            "q-signature": signature,
        })
        
        url = f"{protocol}://{host}{url_path}?{query_string}"
        
        return url
    
    # ========== CDN集成 ==========
    
    async def refresh_cdn_cache(
        self,
        object_keys: List[str],
        bucket: Optional[str] = None,
    ) -> Dict[str, Any]:
        """刷新CDN缓存
        
        Args:
            object_keys: 对象键名列表
            bucket: Bucket名称
            
        Returns:
            刷新结果
        """
        if not self._cfg.cdn_domain:
            raise COSAPIError("NoCDN", "未配置CDN域名")
        
        # 构建刷新URL列表
        urls = []
        for key in object_keys:
            url = self._get_cdn_url(key)
            if url:
                urls.append(url)
        
        # 这里应该调用腾讯云CDN API进行刷新
        # 简化处理，仅返回URL列表
        return {
            "success": True,
            "urls": urls,
            "message": "请使用腾讯云CDN API进行实际刷新操作",
        }
    
    def get_cdn_url(self, object_key: str) -> Optional[str]:
        """获取CDN访问URL
        
        Args:
            object_key: 对象键名
            
        Returns:
            CDN URL或None
        """
        return self._get_cdn_url(object_key)
    
    # ========== 跨域配置 ==========
    
    async def get_cors_config(self, bucket: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取跨域配置
        
        Args:
            bucket: Bucket名称
            
        Returns:
            CORS规则列表
        """
        bucket_name = bucket or self._cfg.bucket_name
        if not bucket_name:
            raise COSAPIError("MissingBucket", "未指定Bucket名称")
        
        try:
            status, headers, body = await self._make_request(
                "GET",
                "/",
                bucket=bucket_name,
                params={"cors": ""},
            )
            
            # 解析XML
            root = ET.fromstring(body)
            rules = []
            for rule_elem in root.findall(".//CORSRule"):
                rule = {}
                for child in rule_elem:
                    tag = child.tag
                    if tag in ["AllowedOrigin", "AllowedMethod", "AllowedHeader"]:
                        if tag not in rule:
                            rule[tag] = []
                        rule[tag].append(child.text)
                    else:
                        rule[tag] = child.text
                rules.append(rule)
            return rules
        except COSAPIError as e:
            if e.code == "NoSuchCORSConfiguration":
                return []
            raise
    
    async def set_cors_config(
        self,
        rules: List[Dict[str, Any]],
        bucket: Optional[str] = None,
    ) -> bool:
        """设置跨域配置
        
        Args:
            rules: CORS规则列表
            bucket: Bucket名称
            
        Returns:
            是否设置成功
        """
        bucket_name = bucket or self._cfg.bucket_name
        if not bucket_name:
            raise COSAPIError("MissingBucket", "未指定Bucket名称")
        
        # 构建XML
        rules_xml = ""
        for rule in rules:
            rule_xml = "<CORSRule>"
            for origin in rule.get("AllowedOrigin", []):
                rule_xml += f"<AllowedOrigin>{origin}</AllowedOrigin>"
            for method in rule.get("AllowedMethod", []):
                rule_xml += f"<AllowedMethod>{method}</AllowedMethod>"
            for header in rule.get("AllowedHeader", []):
                rule_xml += f"<AllowedHeader>{header}</AllowedHeader>"
            if "MaxAgeSeconds" in rule:
                rule_xml += f"<MaxAgeSeconds>{rule['MaxAgeSeconds']}</MaxAgeSeconds>"
            rule_xml += "</CORSRule>"
            rules_xml += rule_xml
        
        xml_body = f"""<?xml version="1.0" encoding="UTF-8"?>
<CORSConfiguration>
{rules_xml}
</CORSConfiguration>"""
        
        headers = {"Content-Type": "application/xml"}
        
        await self._make_request(
            "PUT",
            "/",
            bucket=bucket_name,
            params={"cors": ""},
            data=xml_body.encode("utf-8"),
            headers=headers,
        )
        
        return True
    
    # ========== 工具方法 ==========
    
    def get_capabilities(self):
        """获取适配器能力"""
        return self._capabilities
    
    def supports_capability(self, capability) -> bool:
        """检查是否支持特定能力"""
        return capability in self._capabilities
    
    def __repr__(self) -> str:
        return f"TencentCOSAdapter(endpoint={self._cfg.endpoint}, bucket={self._cfg.bucket_name})"


# ========== CLI测试代码 ==========

async def test_tencent_cos():
    """测试腾讯云COS适配器"""
    import os
    
    # 从环境变量获取配置
    secret_id = os.environ.get("TENCENT_SECRET_ID", "test-secret-id")
    secret_key = os.environ.get("TENCENT_SECRET_KEY", "test-secret-key")
    endpoint = os.environ.get("TENCENT_COS_ENDPOINT", "cos.ap-guangzhou.myqcloud.com")
    bucket_name = os.environ.get("TENCENT_COS_BUCKET", "test-bucket")
    
    config = COSConfig(
        secret_id=secret_id,
        secret_key=secret_key,
        endpoint=endpoint,
        bucket_name=bucket_name,
    )
    
    adapter = TencentCOSAdapter(config)
    
    try:
        # 连接
        print("正在连接腾讯云COS...")
        connected = await adapter.connect()
        print(f"连接结果: {connected}")
        
        if not connected:
            print("连接失败，请检查配置")
            return
        
        # 健康检查
        healthy = await adapter.health_check()
        print(f"健康检查: {healthy}")
        
        # 列出Bucket
        print("\n列出所有Bucket:")
        buckets = await adapter.list_buckets()
        for bucket in buckets:
            print(f"  - {bucket.name} ({bucket.location})")
        
        # 生成预签名URL示例
        print("\n生成预签名URL示例:")
        presigned_url = adapter.generate_presigned_url("test/file.txt", expire=3600)
        print(f"  URL: {presigned_url[:100]}...")
        
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
    asyncio.run(test_tencent_cos())
