"""
AGI Unified Framework - WeChat Work (企业微信) Adapter Module

本模块提供企业微信官方 API 的完整适配器实现，支持消息收发、通讯录管理、
媒体文件上传下载、自定义菜单、标签管理、OAuth2 授权、回调验证以及
AI 机器人消息等全部核心功能。

官方 API 基址: https://qyapi.weixin.qq.com

功能特性:
- 消息发送: 文本、Markdown、图片、文件、视频、语音、图文、交互卡片、小程序
- 消息接收: 解析回调 XML 中所有消息类型
- 用户管理: 获取用户信息、部门用户列表、批量获取用户
- 部门管理: 部门列表、部门详情
- 媒体管理: 上传/下载图片、语音、视频、文件
- 菜单管理: 创建/查询/删除自定义菜单
- 标签管理: 创建/删除标签、标签成员增删
- 回调验证: EncodingAESKey 签名校验
- OAuth2: 通过授权码获取用户身份
- AI 机器人: 以 AI 员工身份发送消息 (2025 新特性)

Author: AGI Team
License: Apache 2.0
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import struct
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from ..base import (
    ChannelAdapter,
    ChannelCapability,
    ChannelConfig,
    ConnectionState,
    MessagePriority,
    ReceiveResult,
    RetryConfig,
    SendResult,
)
from ..universal_message import (
    Attachment,
    AttachmentType,
    ChannelIdentity,
    MessageContent,
    MessageDirection,
    MessageMetadata,
    MessageStatus,
    MessageType,
    UniversalMessage,
    UserIdentity,
)

logger = logging.getLogger(__name__)


# ============================================================
# 配置数据类
# ============================================================


@dataclass
class WeComConfig(ChannelConfig):
    """
    企业微信适配器配置

    Attributes:
        corpid: 企业 ID
        corp_secret: 应用 Secret（自建应用）
        agent_id: 应用 AgentId
        contact_secret: 通讯录同步 Secret（可选，用于通讯录管理）
        media_dir: 媒体文件本地缓存目录
        encoding_aes_key: 回调消息加解密密钥 (EncodingAESKey)
        token: 回调消息验证 Token
        receive_id: 接收消息的企业微信 userid（可选）
    """
    corpid: str = ""
    corp_secret: str = ""
    agent_id: str = ""
    contact_secret: str = ""
    media_dir: str = "/tmp/wecom_media"
    encoding_aes_key: str = ""
    token: str = ""
    receive_id: str = ""


# ============================================================
# 异常类
# ============================================================


class WeComError(Exception):
    """
    企业微信 API 错误

    包含错误码、错误消息以及原始响应，支持判断是否可重试或鉴权错误。
    """

    def __init__(
        self,
        code: int,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(f"[WeCom {code}] {message}")

    @classmethod
    def from_response(cls, response: Dict[str, Any]) -> "WeComError":
        """从 API 响应字典构造异常"""
        return cls(
            code=response.get("errcode", -1),
            message=response.get("errmsg", "未知错误"),
            details=response,
        )

    @property
    def is_retryable(self) -> bool:
        """判断是否为可重试错误（Token 过期、频率限制等）"""
        return self.code in (40014, 42001, 45009, 44001, 200002)

    @property
    def is_auth_error(self) -> bool:
        """判断是否为鉴权错误"""
        return self.code in (40014, 40001, 41001, 40056, 60104)

    @property
    def is_rate_limited(self) -> bool:
        """判断是否为频率限制"""
        return self.code == 45009


# ============================================================
# 回调消息加解密 (基于企业微信官方 PKCS7 算法)
# ============================================================


class WeComCrypto:
    """
    企业微信回调消息加解密工具

    实现 EncodingAESKey 的 AES-256-CBC 加解密，以及签名验证。
    算法参考企业微信官方文档。
    """

    def __init__(self, encoding_aes_key: str, token: str, corpid: str) -> None:
        """
        初始化加解密工具

        Args:
            encoding_aes_key: 43 位 EncodingAESKey
            token: 回调验证 Token
            corpid: 企业 ID
        """
        self._aes_key = base64.b64decode(encoding_aes_key + "=")
        self._token = token
        self._corpid = corpid

    def verify_signature(self, msg_signature: str, timestamp: str, nonce: str, encrypt: str) -> bool:
        """
        验证回调签名

        Args:
            msg_signature: 企业微信推送的签名
            timestamp: 时间戳
            nonce: 随机字符串
            encrypt: 加密消息体

        Returns:
            签名是否有效
        """
        parts = sorted([self._token, timestamp, nonce, encrypt])
        sha1 = hashlib.sha1("".join(parts).encode("utf-8")).hexdigest()
        return sha1 == msg_signature

    def generate_signature(self, timestamp: str, nonce: str, encrypt: str) -> str:
        """生成回调签名"""
        parts = sorted([self._token, timestamp, nonce, encrypt])
        return hashlib.sha1("".join(parts).encode("utf-8")).hexdigest()

    def decrypt(self, encrypt: str) -> Tuple[str, str]:
        """
        解密消息

        Args:
            encrypt: Base64 编码的加密消息

        Returns:
            (corpid, xml_content) 元组
        """
        try:
            from Crypto.Cipher import AES
        except ImportError:
            try:
                from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
                from cryptography.hazmat.backends import default_backend
                _use_cryptography = True
            except ImportError:
                raise ImportError(
                    "需要安装 pycryptodome 或 cryptography 库来支持消息加解密: "
                    "pip install pycryptodome"
                )

        cipher_text = base64.b64decode(encrypt)

        try:
            # 优先使用 pycryptodome
            from Crypto.Cipher import AES
            from Crypto.Util.Padding import unpad

            iv = cipher_text[:16]
            cipher = AES.new(self._aes_key, AES.MODE_CBC, iv)
            decrypted = unpad(cipher.decrypt(cipher_text[16:]), 32)
        except ImportError:
            # 回退到 cryptography
            iv = cipher_text[:16]
            cipher = Cipher(
                algorithms.AES(self._aes_key),
                modes.CBC(iv),
                backend=default_backend(),
            )
            decryptor = cipher.decryptor()
            decrypted = decryptor.update(cipher_text[16:]) + decryptor.finalize()
            # PKCS7 去除填充
            pad_len = decrypted[-1]
            decrypted = decrypted[:-pad_len]

        # 解密后格式: 16字节随机串 + 4字节消息长度(网络序) + 消息内容 + corpid
        content_len = struct.unpack("!I", decrypted[16:20])[0]
        content = decrypted[20 : 20 + content_len].decode("utf-8")
        corpid_from_msg = decrypted[20 + content_len :].decode("utf-8")

        return corpid_from_msg, content

    def encrypt(self, reply_msg: str) -> str:
        """
        加密回复消息

        Args:
            reply_msg: 待加密的 XML 字符串

        Returns:
            Base64 编码的加密字符串
        """
        import random

        corpid_bytes = self._corpid.encode("utf-8")
        msg_bytes = reply_msg.encode("utf-8")
        # 16 字节随机填充
        random_str = bytes([random.randint(0, 255) for _ in range(16)])

        buf = random_str + struct.pack("!I", len(msg_bytes)) + msg_bytes + corpid_bytes
        # PKCS7 填充到 32 字节对齐
        block_size = 32
        pad_len = block_size - (len(buf) % block_size)
        buf += bytes([pad_len] * pad_len)

        try:
            from Crypto.Cipher import AES
            from Crypto.Util.Padding import pad

            iv = self._aes_key[:16]
            cipher = AES.new(self._aes_key, AES.MODE_CBC, iv)
            encrypted = cipher.encrypt(pad(buf, 32))
        except ImportError:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.backends import default_backend

            iv = self._aes_key[:16]
            cipher = Cipher(
                algorithms.AES(self._aes_key),
                modes.CBC(iv),
                backend=default_backend(),
            )
            encryptor = cipher.encryptor()
            encrypted = encryptor.update(buf) + encryptor.finalize()

        return base64.b64encode(encrypted).decode("utf-8")

    def wrap_encrypted_reply(self, reply_msg: str, nonce: str, timestamp: Optional[str] = None) -> str:
        """
        生成加密回复 XML

        Args:
            reply_msg: 原始回复 XML
            nonce: 随机字符串
            timestamp: 时间戳（可选，默认当前时间）

        Returns:
            加密后的 XML 字符串
        """
        if timestamp is None:
            timestamp = str(int(time.time()))
        encrypt = self.encrypt(reply_msg)
        signature = self.generate_signature(timestamp, nonce, encrypt)
        return (
            f"<xml>"
            f"<Encrypt><![CDATA[{encrypt}]]></Encrypt>"
            f"<MsgSignature><![CDATA[{signature}]]></MsgSignature>"
            f"<TimeStamp>{timestamp}</TimeStamp>"
            f"<Nonce><![CDATA[{nonce}]]></Nonce>"
            f"</xml>"
        )


# ============================================================
# 辅助工具类: 交互卡片构建
# ============================================================


class WeComCard:
    """企业微信交互卡片构建工具"""

    @staticmethod
    def text_card(
        title: str,
        description: str,
        url: str,
        btn_txt: str = "详情",
    ) -> Dict[str, Any]:
        """
        文本卡片消息

        Args:
            title: 标题，不超过 128 字节
            description: 描述，不超过 512 字节
            url: 点击跳转链接
            btn_txt: 按钮文字
        """
        return {
            "title": title,
            "description": description,
            "url": url,
            "btntxt": btn_txt,
        }

    @staticmethod
    def news_article(
        title: str,
        description: str,
        url: str,
        pic_url: str = "",
    ) -> Dict[str, Any]:
        """
        图文消息中的单条图文

        Args:
            title: 标题
            description: 描述
            url: 跳转链接
            pic_url: 图文封面
        """
        article: Dict[str, Any] = {
            "title": title,
            "description": description,
            "url": url,
        }
        if pic_url:
            article["picurl"] = pic_url
        return article

    @staticmethod
    def news_articles(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """多条图文消息"""
        return {"articles": articles}

    @staticmethod
    def template_card_text(
        text: str,
        title: str = "",
        button_text: str = "详情",
        button_url: str = "",
    ) -> Dict[str, Any]:
        """
        模板卡片 - 文本通知型

        Args:
            text: 文本内容，支持 markdown
            title: 标题
            button_text: 按钮文字
            button_url: 按钮跳转链接
        """
        card: Dict[str, Any] = {
            "card_type": "text_notice",
            "main_title": {"content": title} if title else {},
            "text_content": {"content": text},
        }
        if button_text:
            card["card_action"] = {
                "type": 1,
                "url": button_url,
                "title": button_text,
            }
        return card

    @staticmethod
    def template_card_button(
        title: str,
        description: str,
        button_list: List[Dict[str, str]],
        horizontal_button_list: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        模板卡片 - 按钮交互型

        Args:
            title: 标题
            description: 描述
            button_list: 竖向按钮列表 [{"text": "xxx", "style": 1, "key": "btn1"}]
            horizontal_button_list: 横向按钮列表
        """
        card: Dict[str, Any] = {
            "card_type": "button_interaction",
            "main_title": {"content": title},
            "sub_title_text": {"content": description},
            "button_list": button_list,
        }
        if horizontal_button_list:
            card["horizontal_button_list"] = horizontal_button_list
        return card

    @staticmethod
    def mini_program_card(
        title: str,
        appid: str,
        pagepath: str,
        pic_media_id: str = "",
        description: str = "",
    ) -> Dict[str, Any]:
        """
        小程序通知卡片

        Args:
            title: 标题
            appid: 小程序 appid
            pagepath: 小程序页面路径
            pic_media_id: 封面图片 media_id
            description: 描述
        """
        card: Dict[str, Any] = {
            "card_type": "mini_program_notice",
            "main_title": {"content": title},
            "source": {"desc": description} if description else {},
            "mini_program": {
                "appid": appid,
                "pagepath": pagepath,
            },
        }
        if pic_media_id:
            card["image_key"] = pic_media_id
        return card


# ============================================================
# 核心适配器
# ============================================================


class WeComAdapter(ChannelAdapter):
    """
    企业微信 (WeCom) 官方 API 适配器

    基于 https://qyapi.weixin.qq.com 提供完整的企业微信 API 集成，
    继承自 ChannelAdapter 基类，实现所有抽象方法。

    使用示例:
        ```python
        config = WeComConfig(
            channel_id="wecom_main",
            corpid="ww1234567890",
            corp_secret="your_app_secret",
            agent_id="1000002",
            encoding_aes_key="your_encoding_aes_key",
            token="your_token",
        )
        adapter = WeComAdapter(config)
        await adapter.connect()
        result = await adapter.send_text("zhangsan", "你好，世界！")
        ```
    """

    BASE_API = "https://qyapi.weixin.qq.com/cgi-bin"

    def __init__(self, config: WeComConfig) -> None:
        """
        初始化企业微信适配器

        Args:
            config: 企业微信配置
        """
        super().__init__(config)
        self._cfg = config

        # Token 管理
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._contact_access_token: Optional[str] = None
        self._contact_token_expires_at: float = 0.0

        # 加解密工具
        self._crypto: Optional[WeComCrypto] = None
        if config.encoding_aes_key and config.token:
            self._crypto = WeComCrypto(
                encoding_aes_key=config.encoding_aes_key,
                token=config.token,
                corpid=config.corpid,
            )

        # 确保媒体目录存在
        os.makedirs(config.media_dir, exist_ok=True)

    def _initialize_capabilities(self) -> None:
        """初始化企业微信支持的能力集"""
        self._capabilities = {
            ChannelCapability.TEXT_MESSAGES,
            ChannelCapability.MARKDOWN_MESSAGES,
            ChannelCapability.MEDIA_MESSAGES,
            ChannelCapability.FILE_ATTACHMENTS,
            ChannelCapability.DIRECT_MESSAGES,
            ChannelCapability.GROUPS,
            ChannelCapability.WEBHOOK_MODE,
            ChannelCapability.BUTTONS,
            ChannelCapability.INTERACTIVE_MESSAGES,
            ChannelCapability.CHANNEL_INFO,
            ChannelCapability.USER_INFO,
            ChannelCapability.RATE_LIMITING,
            ChannelCapability.PUSH_NOTIFICATIONS,
        }

    # ============================================================
    # Token 管理
    # ============================================================

    async def _get_access_token(self, force: bool = False) -> str:
        """
        获取自建应用的 access_token，支持自动缓存与过期刷新

        Args:
            force: 是否强制刷新

        Returns:
            access_token 字符串
        """
        now = time.time()
        if not force and self._access_token and now < self._token_expires_at - 120:
            return self._access_token

        url = f"{self.BASE_API}/gettoken"
        params = {
            "corpid": self._cfg.corpid,
            "corpsecret": self._cfg.corp_secret,
        }
        result = await self._make_request("GET", url, params=params)

        errcode = result.get("errcode", 0)
        if errcode != 0:
            raise WeComError(errcode, result.get("errmsg", "获取 access_token 失败"), result)

        self._access_token = result["access_token"]
        self._token_expires_at = now + result.get("expires_in", 7200)
        self._logger.info("企业微信 access_token 已刷新，有效期 %d 秒", result.get("expires_in", 7200))
        return self._access_token

    async def _get_contact_access_token(self, force: bool = False) -> str:
        """
        获取通讯录管理的 access_token（使用通讯录 Secret）

        Args:
            force: 是否强制刷新

        Returns:
            通讯录 access_token 字符串
        """
        if not self._cfg.contact_secret:
            # 无通讯录 Secret 时回退到应用 Token
            return await self._get_access_token(force)

        now = time.time()
        if not force and self._contact_access_token and now < self._contact_token_expires_at - 120:
            return self._contact_access_token

        url = f"{self.BASE_API}/gettoken"
        params = {
            "corpid": self._cfg.corpid,
            "corpsecret": self._cfg.contact_secret,
        }
        result = await self._make_request("GET", url, params=params)

        errcode = result.get("errcode", 0)
        if errcode != 0:
            raise WeComError(errcode, result.get("errmsg", "获取通讯录 access_token 失败"), result)

        self._contact_access_token = result["access_token"]
        self._contact_token_expires_at = now + result.get("expires_in", 7200)
        return self._contact_access_token

    # ============================================================
    # HTTP 请求封装
    # ============================================================

    async def _make_request(
        self,
        method: str,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        raw_body: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        发送 HTTP 请求到企业微信 API

        Args:
            method: HTTP 方法 (GET/POST)
            url: 完整 URL
            data: JSON 请求体
            params: URL 查询参数
            files: 上传文件 {"field_name": {"file_path": ..., "content_type": ...}}
            raw_body: 原始请求体（用于文件上传等场景）
            headers: 额外请求头

        Returns:
            API 响应字典
        """
        import aiohttp

        req_headers = {"Content-Type": "application/json; charset=utf-8"}
        if headers:
            req_headers.update(headers)

        timeout = aiohttp.ClientTimeout(total=self._config.timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            if params:
                query = urllib.parse.urlencode(params)
                url = url + ("&" if "?" in url else "?") + query

            if files:
                # 文件上传使用 multipart/form-data
                form = aiohttp.FormData()
                for field_name, file_info in files.items():
                    file_path = file_info.get("file_path", "")
                    content_type = file_info.get("content_type", "application/octet-stream")
                    if file_path and os.path.exists(file_path):
                        form.add_field(
                            field_name,
                            open(file_path, "rb"),
                            filename=os.path.basename(file_path),
                            content_type=content_type,
                        )
                    elif "content" in file_info:
                        form.add_field(
                            field_name,
                            file_info["content"],
                            filename=file_info.get("filename", "file"),
                            content_type=content_type,
                        )
                async with session.post(url, data=form) as response:
                    result = await response.json()
            elif raw_body:
                req_headers["Content-Type"] = headers.get("Content-Type", "application/octet-stream") if headers else "application/octet-stream"
                async with session.request(method, url, data=raw_body, headers=req_headers) as response:
                    result = await response.json()
            elif method.upper() == "GET":
                async with session.get(url, headers=req_headers) as response:
                    result = await response.json()
            else:
                async with session.request(method, url, json=data, headers=req_headers) as response:
                    result = await response.json()

        return result

    # ============================================================
    # 连接管理
    # ============================================================

    async def _connect_impl(self) -> bool:
        """建立连接：通过获取 access_token 验证配置有效性"""
        try:
            token = await self._get_access_token()
            self._logger.info("企业微信连接成功，token: %s...", token[:20])
            return True
        except WeComError as e:
            self._logger.error("企业微信连接失败: %s", e)
            return False
        except Exception as e:
            self._logger.error("企业微信连接异常: %s", e)
            return False

    async def _disconnect_impl(self) -> None:
        """断开连接：清除缓存的 Token"""
        self._access_token = None
        self._token_expires_at = 0.0
        self._contact_access_token = None
        self._contact_token_expires_at = 0.0

    async def _health_check_impl(self) -> bool:
        """健康检查：尝试获取 Token"""
        try:
            await self._get_access_token()
            return True
        except Exception:
            return False

    # ============================================================
    # 消息发送
    # ============================================================

    async def send_message(
        self,
        user_id: str,
        msg_type: str,
        content: Dict[str, Any],
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        发送应用消息

        Args:
            user_id: 接收者 userid，多个用 | 分隔
            msg_type: 消息类型 (text/image/voice/video/file/news/mpnews/textcard/template_card)
            content: 消息内容字典
            agent_id: 应用 AgentId（可选，默认使用配置值）

        Returns:
            API 响应
        """
        await self._get_access_token()
        url = f"{self.BASE_API}/message/send"
        payload: Dict[str, Any] = {
            "touser": user_id,
            "msgtype": msg_type,
            "agentid": int(agent_id or self._cfg.agent_id),
            msg_type: content,
        }
        result = await self._make_request("POST", url, data=payload)
        errcode = result.get("errcode", 0)
        if errcode != 0:
            raise WeComError(errcode, result.get("errmsg", "发送消息失败"), result)
        return result

    async def send_text(
        self,
        user_id: str,
        text: str,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """发送文本消息"""
        return await self.send_message(
            user_id=user_id,
            msg_type="text",
            content={"content": text},
            agent_id=agent_id,
        )

    async def send_markdown(
        self,
        user_id: str,
        content: str,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """发送 Markdown 消息"""
        return await self.send_message(
            user_id=user_id,
            msg_type="markdown",
            content={"content": content},
            agent_id=agent_id,
        )

    async def send_image(
        self,
        user_id: str,
        media_id: str,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """发送图片消息"""
        return await self.send_message(
            user_id=user_id,
            msg_type="image",
            content={"media_id": media_id},
            agent_id=agent_id,
        )

    async def send_voice(
        self,
        user_id: str,
        media_id: str,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """发送语音消息"""
        return await self.send_message(
            user_id=user_id,
            msg_type="voice",
            content={"media_id": media_id},
            agent_id=agent_id,
        )

    async def send_video(
        self,
        user_id: str,
        media_id: str,
        description: str = "",
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """发送视频消息"""
        content: Dict[str, Any] = {"media_id": media_id}
        if description:
            content["description"] = description
        return await self.send_message(
            user_id=user_id,
            msg_type="video",
            content=content,
            agent_id=agent_id,
        )

    async def send_file(
        self,
        user_id: str,
        media_id: str,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """发送文件消息"""
        return await self.send_message(
            user_id=user_id,
            msg_type="file",
            content={"media_id": media_id},
            agent_id=agent_id,
        )

    async def send_news(
        self,
        user_id: str,
        articles: List[Dict[str, str]],
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        发送图文消息（mpnews 类型）

        Args:
            user_id: 接收者
            articles: 图文列表 [{"title": "", "thumb_media_id": "", "content": "", "author": "", ...}]
            agent_id: 应用 ID
        """
        return await self.send_message(
            user_id=user_id,
            msg_type="mpnews",
            content={"articles": articles},
            agent_id=agent_id,
        )

    async def send_text_card(
        self,
        user_id: str,
        title: str,
        description: str,
        url: str,
        btn_txt: str = "详情",
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """发送文本卡片消息"""
        return await self.send_message(
            user_id=user_id,
            msg_type="textcard",
            content={
                "title": title,
                "description": description,
                "url": url,
                "btntxt": btn_txt,
            },
            agent_id=agent_id,
        )

    async def send_template_card(
        self,
        user_id: str,
        card: Dict[str, Any],
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        发送模板卡片消息（交互卡片）

        Args:
            user_id: 接收者
            card: 卡片内容，使用 WeComCard 构建
            agent_id: 应用 ID
        """
        return await self.send_message(
            user_id=user_id,
            msg_type="template_card",
            content=card,
            agent_id=agent_id,
        )

    async def send_mini_program(
        self,
        user_id: str,
        title: str,
        appid: str,
        page: str,
        pic_media_id: str = "",
        description: str = "",
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        发送小程序通知消息

        Args:
            user_id: 接收者
            title: 标题
            appid: 小程序 appid
            page: 小程序页面路径
            pic_media_id: 封面图 media_id
            description: 描述
            agent_id: 应用 ID
        """
        card = WeComCard.mini_program_card(
            title=title,
            appid=appid,
            pagepath=page,
            pic_media_id=pic_media_id,
            description=description,
        )
        return await self.send_message(
            user_id=user_id,
            msg_type="template_card",
            content=card,
            agent_id=agent_id,
        )

    async def send_to_party(
        self,
        party_id: str,
        msg_type: str,
        content: Dict[str, Any],
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """按部门发送消息"""
        await self._get_access_token()
        url = f"{self.BASE_API}/message/send"
        payload: Dict[str, Any] = {
            "toparty": party_id,
            "msgtype": msg_type,
            "agentid": int(agent_id or self._cfg.agent_id),
            msg_type: content,
        }
        result = await self._make_request("POST", url, data=payload)
        errcode = result.get("errcode", 0)
        if errcode != 0:
            raise WeComError(errcode, result.get("errmsg", "按部门发送消息失败"), result)
        return result

    async def send_to_tag(
        self,
        tag_id: str,
        msg_type: str,
        content: Dict[str, Any],
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """按标签发送消息"""
        await self._get_access_token()
        url = f"{self.BASE_API}/message/send"
        payload: Dict[str, Any] = {
            "totag": tag_id,
            "msgtype": msg_type,
            "agentid": int(agent_id or self._cfg.agent_id),
            msg_type: content,
        }
        result = await self._make_request("POST", url, data=payload)
        errcode = result.get("errcode", 0)
        if errcode != 0:
            raise WeComError(errcode, result.get("errmsg", "按标签发送消息失败"), result)
        return result

    # ============================================================
    # AI 机器人消息 (2025 新特性)
    # ============================================================

    async def send_as_ai_employee(
        self,
        user_id: str,
        msg_type: str,
        content: Dict[str, Any],
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        以 AI 员工身份发送消息（2025 新特性）

        企业微信支持将应用配置为 AI 员工，发送消息时携带 AI 身份标识，
        用户可在聊天中直接与 AI 进行交互。

        Args:
            user_id: 接收者 userid
            msg_type: 消息类型
            content: 消息内容
            agent_id: 应用 AgentId

        Returns:
            API 响应
        """
        await self._get_access_token()
        url = f"{self.BASE_API}/message/send"
        payload: Dict[str, Any] = {
            "touser": user_id,
            "msgtype": msg_type,
            "agentid": int(agent_id or self._cfg.agent_id),
            msg_type: content,
            "msgtype_extra": {"ai_employee": True},
        }
        result = await self._make_request("POST", url, data=payload)
        errcode = result.get("errcode", 0)
        if errcode != 0:
            raise WeComError(errcode, result.get("errmsg", "AI 员工发送消息失败"), result)
        return result

    # ============================================================
    # 消息接收与回调解析
    # ============================================================

    def parse_callback_xml(self, xml_body: str) -> Dict[str, Any]:
        """
        解析企业微信回调 XML 消息

        Args:
            xml_body: 原始 XML 字符串

        Returns:
            解析后的消息字典
        """
        root = ET.fromstring(xml_body)
        result: Dict[str, Any] = {}

        # 提取所有子元素
        for child in root:
            result[child.tag] = child.text or ""

        # 消息类型映射
        msg_type = result.get("MsgType", "")
        result["_msg_type"] = msg_type

        # 解析特定类型的嵌套内容
        if msg_type == "image":
            result["image"] = {
                "media_id": result.get("MediaId", ""),
                "pic_url": result.get("PicUrl", ""),
            }
        elif msg_type == "voice":
            result["voice"] = {
                "media_id": result.get("MediaId", ""),
                "format": result.get("Format", ""),
                "recognition": result.get("Recognition", ""),
            }
        elif msg_type == "video" or msg_type == "shortvideo":
            result["video"] = {
                "media_id": result.get("MediaId", ""),
                "thumb_media_id": result.get("ThumbMediaId", ""),
            }
        elif msg_type == "location":
            result["location"] = {
                "latitude": result.get("Location_X", ""),
                "longitude": result.get("Location_Y", ""),
                "scale": result.get("Scale", ""),
                "label": result.get("Label", ""),
            }
        elif msg_type == "link":
            result["link"] = {
                "title": result.get("Title", ""),
                "description": result.get("Description", ""),
                "url": result.get("Url", ""),
            }
        elif msg_type == "event":
            result["event"] = {
                "event_type": result.get("Event", ""),
                "event_key": result.get("EventKey", ""),
                "ticket": result.get("Ticket", ""),
            }

        return result

    def decrypt_callback(self, xml_body: str, msg_signature: str, timestamp: str, nonce: str) -> Dict[str, Any]:
        """
        解密回调消息并解析

        Args:
            xml_body: 加密的 XML 消息体
            msg_signature: 消息签名
            timestamp: 时间戳
            nonce: 随机字符串

        Returns:
            解密后的消息字典

        Raises:
            WeComError: 签名验证失败或解密失败
        """
        if not self._crypto:
            raise WeComError(-1, "未配置 EncodingAESKey 和 Token，无法解密回调消息")

        root = ET.fromstring(xml_body)
        encrypt_node = root.find("Encrypt")
        if encrypt_node is None:
            raise WeComError(-1, "回调 XML 中缺少 Encrypt 节点")

        encrypt_content = encrypt_node.text or ""

        # 验证签名
        if not self._crypto.verify_signature(msg_signature, timestamp, nonce, encrypt_content):
            raise WeComError(-2, "回调消息签名验证失败")

        # 解密
        corpid, decrypted_xml = self._crypto.decrypt(encrypt_content)
        if corpid != self._cfg.corpid:
            raise WeComError(-3, f"解密后 corpid 不匹配: {corpid} != {self._cfg.corpid}")

        return self.parse_callback_xml(decrypted_xml)

    def verify_callback_signature(
        self,
        msg_signature: str,
        timestamp: str,
        nonce: str,
        encrypt: str,
    ) -> bool:
        """
        验证回调签名

        Args:
            msg_signature: 企业微信推送的签名
            timestamp: 时间戳
            nonce: 随机字符串
            encrypt: 加密消息体

        Returns:
            签名是否有效
        """
        if not self._crypto:
            return False
        return self._crypto.verify_signature(msg_signature, timestamp, nonce, encrypt)

    def build_reply_xml(
        self,
        to_user: str,
        from_user: str,
        msg_type: str,
        content: str,
        agent_id: Optional[str] = None,
    ) -> str:
        """
        构建被动回复 XML

        Args:
            to_user: 接收者 userid
            from_user: 企业微信应用的 CorpID
            msg_type: 消息类型
            content: 消息内容
            agent_id: 应用 AgentId

        Returns:
            XML 字符串
        """
        create_time = str(int(time.time()))
        if msg_type == "text":
            return (
                f"<xml>"
                f"<ToUserName><![CDATA[{to_user}]]></ToUserName>"
                f"<FromUserName><![CDATA[{from_user}]]></FromUserName>"
                f"<CreateTime>{create_time}</CreateTime>"
                f"<MsgType><![CDATA[text]]></MsgType>"
                f"<Content><![CDATA[{content}]]></Content>"
                f"</xml>"
            )
        elif msg_type == "image":
            return (
                f"<xml>"
                f"<ToUserName><![CDATA[{to_user}]]></ToUserName>"
                f"<FromUserName><![CDATA[{from_user}]]></FromUserName>"
                f"<CreateTime>{create_time}</CreateTime>"
                f"<MsgType><![CDATA[image]]></MsgType>"
                f"<Image><MediaId><![CDATA[{content}]]></MediaId></Image>"
                f"</xml>"
            )
        elif msg_type == "voice":
            return (
                f"<xml>"
                f"<ToUserName><![CDATA[{to_user}]]></ToUserName>"
                f"<FromUserName><![CDATA[{from_user}]]></FromUserName>"
                f"<CreateTime>{create_time}</CreateTime>"
                f"<MsgType><![CDATA[voice]]></MsgType>"
                f"<Voice><MediaId><![CDATA[{content}]]></MediaId></Voice>"
                f"</xml>"
            )
        elif msg_type == "video":
            # content 格式: media_id|title|description
            parts = content.split("|", 2)
            media_id = parts[0]
            title = parts[1] if len(parts) > 1 else ""
            desc = parts[2] if len(parts) > 2 else ""
            return (
                f"<xml>"
                f"<ToUserName><![CDATA[{to_user}]]></ToUserName>"
                f"<FromUserName><![CDATA[{from_user}]]></FromUserName>"
                f"<CreateTime>{create_time}</CreateTime>"
                f"<MsgType><![CDATA[video]]></MsgType>"
                f"<Video>"
                f"<MediaId><![CDATA[{media_id}]]></MediaId>"
                f"<Title><![CDATA[{title}]]></Title>"
                f"<Description><![CDATA[{desc}]]></Description>"
                f"</Video>"
                f"</xml>"
            )
        elif msg_type == "news":
            return (
                f"<xml>"
                f"<ToUserName><![CDATA[{to_user}]]></ToUserName>"
                f"<FromUserName><![CDATA[{from_user}]]></FromUserName>"
                f"<CreateTime>{create_time}</CreateTime>"
                f"<MsgType><![CDATA[news]]></MsgType>"
                f"<ArticleCount>1</ArticleCount>"
                f"<Articles>"
                f"<item><Title><![CDATA[{content}]]></Title></item>"
                f"</Articles>"
                f"</xml>"
            )
        else:
            return ""

    def build_encrypted_reply(
        self,
        to_user: str,
        from_user: str,
        msg_type: str,
        content: str,
        nonce: str,
        agent_id: Optional[str] = None,
    ) -> str:
        """
        构建加密的被动回复 XML

        Args:
            to_user: 接收者
            from_user: 发送者 (CorpID)
            msg_type: 消息类型
            content: 消息内容
            nonce: 随机字符串
            agent_id: 应用 ID

        Returns:
            加密后的 XML 字符串
        """
        reply_xml = self.build_reply_xml(to_user, from_user, msg_type, content, agent_id)
        if not self._crypto:
            return reply_xml
        return self._crypto.wrap_encrypted_reply(reply_xml, nonce)

    # ============================================================
    # 内部消息转换
    # ============================================================

    def _transform_incoming_message(self, msg_data: Dict[str, Any]) -> UniversalMessage:
        """
        将企业微信回调消息转换为 UniversalMessage

        Args:
            msg_data: 解析后的回调消息字典

        Returns:
            UniversalMessage 实例
        """
        msg_type = msg_data.get("_msg_type", msg_data.get("MsgType", "text"))
        text = msg_data.get("Content", "")

        # 确定消息内容
        if msg_type == "text":
            content = MessageContent(text=text)
        elif msg_type == "image":
            pic_url = msg_data.get("PicUrl", "")
            content = MessageContent(
                text="[图片]",
                raw_content={"media_id": msg_data.get("MediaId", ""), "pic_url": pic_url},
            )
        elif msg_type == "voice":
            recognition = msg_data.get("Recognition", "")
            content = MessageContent(
                text=recognition if recognition else "[语音消息]",
                raw_content={
                    "media_id": msg_data.get("MediaId", ""),
                    "format": msg_data.get("Format", ""),
                    "recognition": recognition,
                },
            )
        elif msg_type in ("video", "shortvideo"):
            content = MessageContent(
                text="[视频消息]",
                raw_content={
                    "media_id": msg_data.get("MediaId", ""),
                    "thumb_media_id": msg_data.get("ThumbMediaId", ""),
                },
            )
        elif msg_type == "location":
            content = MessageContent(
                text=f"[位置] {msg_data.get('Label', '')}",
                raw_content={
                    "latitude": msg_data.get("Location_X", ""),
                    "longitude": msg_data.get("Location_Y", ""),
                    "scale": msg_data.get("Scale", ""),
                    "label": msg_data.get("Label", ""),
                },
            )
        elif msg_type == "link":
            content = MessageContent(
                text=f"[链接] {msg_data.get('Title', '')}",
                raw_content={
                    "title": msg_data.get("Title", ""),
                    "description": msg_data.get("Description", ""),
                    "url": msg_data.get("Url", ""),
                },
            )
        elif msg_type == "event":
            event_type = msg_data.get("Event", "")
            event_key = msg_data.get("EventKey", "")
            content = MessageContent(
                text=f"[事件] {event_type}: {event_key}",
                raw_content={"event_type": event_type, "event_key": event_key},
            )
        else:
            content = MessageContent(text=text or f"[{msg_type}消息]")

        # 构建发送者信息
        sender_id = msg_data.get("FromUserName", "")
        sender = UserIdentity(
            user_id=sender_id,
            display_name=sender_id,
        )

        # 构建消息元数据
        msg_id = msg_data.get("MsgId", msg_data.get("MsgID", str(int(time.time() * 1000))))
        create_time = float(msg_data.get("CreateTime", time.time()))

        # 映射消息类型
        type_mapping = {
            "text": MessageType.TEXT,
            "image": MessageType.IMAGE,
            "voice": MessageType.AUDIO,
            "video": MessageType.VIDEO,
            "shortvideo": MessageType.VIDEO,
            "location": MessageType.LOCATION,
            "link": MessageType.TEXT,
            "event": MessageType.SYSTEM,
        }
        mapped_type = type_mapping.get(msg_type, MessageType.TEXT)

        metadata = MessageMetadata(
            message_id=str(msg_id),
            channel_id="wecom",
            timestamp=create_time,
            direction=MessageDirection.INBOUND,
            message_type=mapped_type,
            raw_event=msg_data,
            sender=sender,
            channel=ChannelIdentity(
                channel_id=msg_data.get("ToUserName", ""),
                channel_type="wecom",
                name="企业微信",
            ),
        )

        message = UniversalMessage(content=content, metadata=metadata)
        message.set_context("user_id", sender_id)
        message.set_context("agent_id", msg_data.get("AgentID", ""))
        message.set_context("msg_type", msg_type)
        if msg_data.get("MediaId"):
            message.set_context("media_id", msg_data["MediaId"])

        return message

    # ============================================================
    # ChannelAdapter 抽象方法实现
    # ============================================================

    async def _send_impl(
        self,
        message: UniversalMessage,
        priority: MessagePriority,
    ) -> SendResult:
        """
        发送消息的内部实现

        根据 UniversalMessage 的内容自动选择消息类型并发送。
        """
        try:
            user_id = message.get_context("user_id") or self._cfg.receive_id
            if not user_id:
                return SendResult(
                    success=False,
                    error="未指定接收者 user_id",
                    error_code="MISSING_TARGET",
                )

            # 根据消息内容判断发送类型
            msg_type_ctx = message.get_context("msg_type")
            text = message.content.get_primary_text()

            if msg_type_ctx == "markdown" or message.content.markdown:
                result = await self.send_markdown(user_id, message.content.markdown or text)
            elif msg_type_ctx == "image":
                media_id = message.get_context("media_id", "")
                if not media_id:
                    return SendResult(success=False, error="缺少图片 media_id", error_code="MISSING_MEDIA")
                result = await self.send_image(user_id, media_id)
            elif msg_type_ctx == "voice":
                media_id = message.get_context("media_id", "")
                if not media_id:
                    return SendResult(success=False, error="缺少语音 media_id", error_code="MISSING_MEDIA")
                result = await self.send_voice(user_id, media_id)
            elif msg_type_ctx == "video":
                media_id = message.get_context("media_id", "")
                if not media_id:
                    return SendResult(success=False, error="缺少视频 media_id", error_code="MISSING_MEDIA")
                result = await self.send_video(user_id, media_id)
            elif msg_type_ctx == "file":
                media_id = message.get_context("media_id", "")
                if not media_id:
                    return SendResult(success=False, error="缺少文件 media_id", error_code="MISSING_MEDIA")
                result = await self.send_file(user_id, media_id)
            elif msg_type_ctx == "textcard":
                card_data = message.get_context("card", {})
                result = await self.send_text_card(
                    user_id,
                    card_data.get("title", text),
                    card_data.get("description", ""),
                    card_data.get("url", ""),
                    card_data.get("btn_txt", "详情"),
                )
            elif msg_type_ctx == "template_card":
                card_data = message.get_context("card", {})
                result = await self.send_template_card(user_id, card_data)
            elif msg_type_ctx == "mpnews":
                articles = message.get_context("articles", [])
                result = await self.send_news(user_id, articles)
            elif msg_type_ctx == "miniprogram":
                mp_data = message.get_context("miniprogram", {})
                result = await self.send_mini_program(
                    user_id,
                    mp_data.get("title", text),
                    mp_data.get("appid", ""),
                    mp_data.get("page", ""),
                    mp_data.get("pic_media_id", ""),
                    mp_data.get("description", ""),
                )
            else:
                # 默认发送文本
                result = await self.send_text(user_id, text)

            return SendResult(
                success=True,
                message_id=str(result.get("msgid", "")),
                timestamp=time.time(),
            )
        except WeComError as e:
            return SendResult(success=False, error=e.message, error_code=str(e.code))
        except Exception as e:
            return SendResult(success=False, error=str(e), error_code=type(e).__name__)

    async def _receive_impl(
        self,
        payload: Optional[Dict[str, Any]],
    ) -> ReceiveResult:
        """
        接收消息的内部实现

        Args:
            payload: 回调消息字典（已解析的 XML 数据）
        """
        if not payload:
            return ReceiveResult(success=False, error="未提供消息负载")

        try:
            message = self._transform_incoming_message(payload)
            return ReceiveResult(
                success=True,
                messages=[message],
                raw_payload=payload,
            )
        except Exception as e:
            self._logger.error("解析企业微信消息失败: %s", e)
            return ReceiveResult(success=False, error=str(e))

    async def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        获取用户详细信息

        Args:
            user_id: 企业微信 userid

        Returns:
            用户信息字典
        """
        try:
            token = await self._get_contact_access_token()
            url = f"{self.BASE_API}/user/get"
            params = {"access_token": token, "userid": user_id}
            result = await self._make_request("GET", url, params=params)

            errcode = result.get("errcode", 0)
            if errcode != 0:
                self._logger.warning("获取用户信息失败 [%d]: %s", errcode, result.get("errmsg", ""))
                return {"user_id": user_id, "error": result.get("errmsg", "")}

            return {
                "user_id": result.get("userid", user_id),
                "name": result.get("name", ""),
                "english_name": result.get("english_name", ""),
                "mobile": result.get("mobile", ""),
                "department": result.get("department", []),
                "position": result.get("position", ""),
                "gender": result.get("gender", 0),
                "email": result.get("email", ""),
                "avatar": result.get("avatar", ""),
                "telephone": result.get("telephone", ""),
                "is_leader": result.get("is_leader_in_dept", []),
                "enable": result.get("enable", 1),
                "alias": result.get("alias", ""),
                "status": result.get("status", 1),
            }
        except Exception as e:
            self._logger.warning("获取用户信息异常: %s", e)
            return {"user_id": user_id, "error": str(e)}

    async def get_channel_info(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """
        获取频道（群聊）信息

        Args:
            channel_id: 频道 ID

        Returns:
            频道信息字典
        """
        try:
            token = await self._get_access_token()
            url = f"{self.BASE_API}/appchat/get"
            params = {"access_token": token, "chatid": channel_id}
            result = await self._make_request("GET", url, params=params)

            errcode = result.get("errcode", 0)
            if errcode != 0:
                self._logger.warning("获取群聊信息失败 [%d]: %s", errcode, result.get("errmsg", ""))
                return {"channel_id": channel_id, "error": result.get("errmsg", "")}

            return {
                "channel_id": result.get("chatid", channel_id),
                "name": result.get("name", ""),
                "owner": result.get("owner", ""),
                "member_count": len(result.get("member_list", [])),
            }
        except Exception as e:
            self._logger.warning("获取频道信息异常: %s", e)
            return {"channel_id": channel_id, "error": str(e)}

    # ============================================================
    # 用户管理
    # ============================================================

    async def list_department_users(
        self,
        department_id: int,
        fetch_child: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        获取部门成员列表（简要信息）

        Args:
            department_id: 部门 ID
            fetch_child: 是否递归获取子部门成员

        Returns:
            用户信息列表
        """
        token = await self._get_contact_access_token()
        url = f"{self.BASE_API}/user/simplelist"
        params = {
            "access_token": token,
            "department_id": department_id,
            "fetch_child": 1 if fetch_child else 0,
        }
        result = await self._make_request("GET", url, params=params)
        errcode = result.get("errcode", 0)
        if errcode != 0:
            raise WeComError(errcode, result.get("errmsg", "获取部门用户列表失败"), result)
        return result.get("userlist", [])

    async def list_department_users_detail(
        self,
        department_id: int,
        fetch_child: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        获取部门成员详情列表

        Args:
            department_id: 部门 ID
            fetch_child: 是否递归获取子部门成员

        Returns:
            用户详细信息列表
        """
        token = await self._get_contact_access_token()
        url = f"{self.BASE_API}/user/list"
        params = {
            "access_token": token,
            "department_id": department_id,
            "fetch_child": 1 if fetch_child else 0,
        }
        result = await self._make_request("GET", url, params=params)
        errcode = result.get("errcode", 0)
        if errcode != 0:
            raise WeComError(errcode, result.get("errmsg", "获取部门用户详情失败"), result)
        return result.get("userlist", [])

    async def batch_get_users(
        self,
        user_id_list: List[str],
    ) -> List[Dict[str, Any]]:
        """
        批量获取用户信息

        Args:
            user_id_list: userid 列表（最多 200 个）

        Returns:
            用户信息列表
        """
        token = await self._get_contact_access_token()
        url = f"{self.BASE_API}/user/batchget"
        payload = {
            "useridlist": user_id_list[:200],
        }
        params = {"access_token": token}
        result = await self._make_request("POST", url, data=payload, params=params)
        errcode = result.get("errcode", 0)
        if errcode != 0:
            raise WeComError(errcode, result.get("errmsg", "批量获取用户失败"), result)
        return result.get("userlist", [])

    async def get_user_id_by_phone(self, mobile: str) -> Optional[str]:
        """
        通过手机号获取 userid

        Args:
            mobile: 手机号

        Returns:
            userid 或 None
        """
        token = await self._get_contact_access_token()
        url = f"{self.BASE_API}/user/getuserid"
        payload = {"mobile": mobile}
        params = {"access_token": token}
        result = await self._make_request("POST", url, data=payload, params=params)
        errcode = result.get("errcode", 0)
        if errcode != 0:
            self._logger.warning("手机号查 userid 失败 [%d]: %s", errcode, result.get("errmsg", ""))
            return None
        return result.get("userid")

    # ============================================================
    # 部门管理
    # ============================================================

    async def list_departments(self, department_id: int = 0) -> List[Dict[str, Any]]:
        """
        获取部门列表

        Args:
            department_id: 父部门 ID（0 表示根部门）

        Returns:
            部门信息列表
        """
        token = await self._get_contact_access_token()
        url = f"{self.BASE_API}/department/list"
        params = {
            "access_token": token,
            "id": department_id,
        }
        result = await self._make_request("GET", url, params=params)
        errcode = result.get("errcode", 0)
        if errcode != 0:
            raise WeComError(errcode, result.get("errmsg", "获取部门列表失败"), result)
        return result.get("department", [])

    async def get_department_info(self, department_id: int) -> Dict[str, Any]:
        """
        获取单个部门详情

        Args:
            department_id: 部门 ID

        Returns:
            部门信息字典
        """
        token = await self._get_contact_access_token()
        url = f"{self.BASE_API}/department/get"
        params = {
            "access_token": token,
            "id": department_id,
        }
        result = await self._make_request("GET", url, params=params)
        errcode = result.get("errcode", 0)
        if errcode != 0:
            raise WeComError(errcode, result.get("errmsg", "获取部门信息失败"), result)
        return {
            "id": result.get("id"),
            "name": result.get("name", ""),
            "name_en": result.get("name_en", ""),
            "parentid": result.get("parentid"),
            "order": result.get("order", 0),
            "department_leader": result.get("department_leader", []),
        }

    # ============================================================
    # 媒体文件管理
    # ============================================================

    async def upload_media(
        self,
        file_path: str,
        media_type: str = "file",
    ) -> Optional[str]:
        """
        上传临时媒体文件

        Args:
            file_path: 本地文件路径
            media_type: 媒体类型 (image/voice/video/file)

        Returns:
            media_id 或 None
        """
        await self._get_access_token()
        url = f"{self.BASE_API}/media/upload"
        params = {"access_token": self._access_token, "type": media_type}

        # 根据媒体类型设置 Content-Type
        content_types = {
            "image": "image/jpeg",
            "voice": "audio/amr",
            "video": "video/mp4",
            "file": "application/octet-stream",
        }
        content_type = content_types.get(media_type, "application/octet-stream")

        result = await self._make_request(
            "POST",
            url,
            params=params,
            files={"media": {"file_path": file_path, "content_type": content_type}},
        )

        errcode = result.get("errcode", 0)
        if errcode != 0:
            self._logger.error("上传媒体文件失败 [%d]: %s", errcode, result.get("errmsg", ""))
            return None

        media_id = result.get("media_id")
        self._logger.info("媒体文件上传成功，media_id: %s", media_id)
        return media_id

    async def upload_image(self, file_path: str) -> Optional[str]:
        """上传图片"""
        return await self.upload_media(file_path, "image")

    async def upload_voice(self, file_path: str) -> Optional[str]:
        """上传语音（amr 格式）"""
        return await self.upload_media(file_path, "voice")

    async def upload_video(self, file_path: str) -> Optional[str]:
        """上传视频（mp4 格式）"""
        return await self.upload_media(file_path, "video")

    async def upload_file(self, file_path: str) -> Optional[str]:
        """上传文件"""
        return await self.upload_media(file_path, "file")

    async def download_media(
        self,
        media_id: str,
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        下载临时媒体文件

        Args:
            media_id: 媒体文件 ID
            save_path: 保存路径（可选，默认保存到 media_dir）

        Returns:
            保存后的本地文件路径，或 None
        """
        import aiohttp

        await self._get_access_token()
        url = f"{self.BASE_API}/media/get"
        params = {"access_token": self._access_token, "media_id": media_id}

        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as response:
                content_type = response.headers.get("Content-Type", "")

                # 如果返回的是 JSON，说明出错了
                if "application/json" in content_type:
                    result = await response.json()
                    errcode = result.get("errcode", -1)
                    self._logger.error("下载媒体失败 [%d]: %s", errcode, result.get("errmsg", ""))
                    return None

                # 从 Content-Disposition 中提取文件名
                disposition = response.headers.get("Content-Disposition", "")
                filename = "media_file"
                if "filename=" in disposition:
                    filename = disposition.split("filename=")[-1].strip('"')

                if not save_path:
                    save_path = os.path.join(self._cfg.media_dir, filename)

                # 确保目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                # 写入文件
                with open(save_path, "wb") as f:
                    f.write(await response.read())

                self._logger.info("媒体文件已下载: %s", save_path)
                return save_path

    async def upload_image_by_url(
        self,
        image_url: str,
        media_type: str = "image",
    ) -> Optional[str]:
        """
        通过 URL 上传图片

        Args:
            image_url: 图片 URL
            media_type: 媒体类型

        Returns:
            media_id 或 None
        """
        import aiohttp

        # 先下载图片到临时文件
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(image_url) as response:
                if response.status != 200:
                    self._logger.error("下载图片失败: HTTP %d", response.status)
                    return None
                content = await response.read()

        # 保存到临时文件
        ext = ".jpg"
        if "png" in image_url:
            ext = ".png"
        elif "gif" in image_url:
            ext = ".gif"
        temp_path = os.path.join(self._cfg.media_dir, f"temp_{int(time.time())}{ext}")
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(content)

        # 上传
        try:
            return await self.upload_media(temp_path, media_type)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # ============================================================
    # 自定义菜单管理
    # ============================================================

    async def create_menu(
        self,
        menu_data: Dict[str, Any],
        agent_id: Optional[str] = None,
    ) -> bool:
        """
        创建自定义菜单

        Args:
            menu_data: 菜单配置（参考企业微信官方文档）
            agent_id: 应用 AgentId

        Returns:
            是否成功
        """
        await self._get_access_token()
        aid = agent_id or self._cfg.agent_id
        url = f"{self.BASE_API}/menu/create"
        params = {"access_token": self._access_token, "agentid": aid}
        result = await self._make_request("POST", url, data=menu_data, params=params)
        errcode = result.get("errcode", 0)
        if errcode != 0:
            raise WeComError(errcode, result.get("errmsg", "创建菜单失败"), result)
        return True

    async def get_menu(
        self,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        查询自定义菜单

        Args:
            agent_id: 应用 AgentId

        Returns:
            菜单配置字典
        """
        await self._get_access_token()
        aid = agent_id or self._cfg.agent_id
        url = f"{self.BASE_API}/menu/get"
        params = {"access_token": self._access_token, "agentid": aid}
        result = await self._make_request("GET", url, params=params)
        errcode = result.get("errcode", 0)
        if errcode != 0:
            raise WeComError(errcode, result.get("errmsg", "获取菜单失败"), result)
        return result.get("menu", {})

    async def delete_menu(
        self,
        agent_id: Optional[str] = None,
    ) -> bool:
        """
        删除自定义菜单

        Args:
            agent_id: 应用 AgentId

        Returns:
            是否成功
        """
        await self._get_access_token()
        aid = agent_id or self._cfg.agent_id
        url = f"{self.BASE_API}/menu/delete"
        params = {"access_token": self._access_token, "agentid": aid}
        result = await self._make_request("GET", url, params=params)
        errcode = result.get("errcode", 0)
        if errcode != 0:
            raise WeComError(errcode, result.get("errmsg", "删除菜单失败"), result)
        return True

    # ============================================================
    # 标签管理
    # ============================================================

    async def create_tag(self, tag_name: str, tag_id: Optional[int] = None) -> Dict[str, Any]:
        """
        创建标签

        Args:
            tag_name: 标签名称（不超过 32 个字符）
            tag_id: 标签 ID（可选，不指定则自动生成）

        Returns:
            创建结果，包含 tagid
        """
        token = await self._get_contact_access_token()
        url = f"{self.BASE_API}/tag/create"
        payload: Dict[str, Any] = {"tagname": tag_name}
        if tag_id is not None:
            payload["tagid"] = tag_id
        params = {"access_token": token}
        result = await self._make_request("POST", url, data=payload, params=params)
        errcode = result.get("errcode", 0)
        if errcode != 0:
            raise WeComError(errcode, result.get("errmsg", "创建标签失败"), result)
        return {"tagid": result.get("tagid"), "tagname": tag_name}

    async def delete_tag(self, tag_id: int) -> bool:
        """
        删除标签

        Args:
            tag_id: 标签 ID

        Returns:
            是否成功
        """
        token = await self._get_contact_access_token()
        url = f"{self.BASE_API}/tag/delete"
        params = {"access_token": token, "tagid": tag_id}
        result = await self._make_request("GET", url, params=params)
        errcode = result.get("errcode", 0)
        if errcode != 0:
            raise WeComError(errcode, result.get("errmsg", "删除标签失败"), result)
        return True

    async def update_tag(self, tag_id: int, tag_name: str) -> bool:
        """
        更新标签名称

        Args:
            tag_id: 标签 ID
            tag_name: 新标签名称

        Returns:
            是否成功
        """
        token = await self._get_contact_access_token()
        url = f"{self.BASE_API}/tag/update"
        payload = {"tagid": tag_id, "tagname": tag_name}
        params = {"access_token": token}
        result = await self._make_request("POST", url, data=payload, params=params)
        errcode = result.get("errcode", 0)
        if errcode != 0:
            raise WeComError(errcode, result.get("errmsg", "更新标签失败"), result)
        return True

    async def get_tag_users(self, tag_id: int) -> List[Dict[str, Any]]:
        """
        获取标签下的成员列表

        Args:
            tag_id: 标签 ID

        Returns:
            用户列表
        """
        token = await self._get_contact_access_token()
        url = f"{self.BASE_API}/tag/get"
        params = {"access_token": token, "tagid": tag_id}
        result = await self._make_request("GET", url, params=params)
        errcode = result.get("errcode", 0)
        if errcode != 0:
            raise WeComError(errcode, result.get("errmsg", "获取标签成员失败"), result)
        return result.get("userlist", [])

    async def add_tag_users(
        self,
        tag_id: int,
        user_list: Optional[List[str]] = None,
        party_list: Optional[List[int]] = None,
    ) -> bool:
        """
        为标签添加成员

        Args:
            tag_id: 标签 ID
            user_list: 用户 userid 列表
            party_list: 部门 ID 列表

        Returns:
            是否成功
        """
        token = await self._get_contact_access_token()
        url = f"{self.BASE_API}/tag/addtagusers"
        payload: Dict[str, Any] = {"tagid": tag_id}
        if user_list:
            payload["userlist"] = user_list
        if party_list:
            payload["partylist"] = party_list
        params = {"access_token": token}
        result = await self._make_request("POST", url, data=payload, params=params)
        errcode = result.get("errcode", 0)
        if errcode != 0:
            raise WeComError(errcode, result.get("errmsg", "添加标签成员失败"), result)
        return True

    async def remove_tag_users(
        self,
        tag_id: int,
        user_list: Optional[List[str]] = None,
        party_list: Optional[List[int]] = None,
    ) -> bool:
        """
        从标签中移除成员

        Args:
            tag_id: 标签 ID
            user_list: 用户 userid 列表
            party_list: 部门 ID 列表

        Returns:
            是否成功
        """
        token = await self._get_contact_access_token()
        url = f"{self.BASE_API}/tag/deltagusers"
        payload: Dict[str, Any] = {"tagid": tag_id}
        if user_list:
            payload["userlist"] = user_list
        if party_list:
            payload["partylist"] = party_list
        params = {"access_token": token}
        result = await self._make_request("POST", url, data=payload, params=params)
        errcode = result.get("errcode", 0)
        if errcode != 0:
            raise WeComError(errcode, result.get("errmsg", "移除标签成员失败"), result)
        return True

    async def list_tags(self) -> List[Dict[str, Any]]:
        """
        获取所有标签列表

        Returns:
            标签列表
        """
        token = await self._get_contact_access_token()
        url = f"{self.BASE_API}/tag/list"
        params = {"access_token": token}
        result = await self._make_request("GET", url, params=params)
        errcode = result.get("errcode", 0)
        if errcode != 0:
            raise WeComError(errcode, result.get("errmsg", "获取标签列表失败"), result)
        return result.get("taglist", [])

    # ============================================================
    # OAuth2 授权
    # ============================================================

    def get_oauth_url(
        self,
        redirect_uri: str,
        state: str = "",
        scope: str = "snsapi_base",
    ) -> str:
        """
        构造 OAuth2 授权链接

        Args:
            redirect_uri: 授权后重定向的回调链接
            state: 用于保持请求和回调的状态参数
            scope: 应用授权作用域 (snsapi_base 或 snsapi_privateinfo)

        Returns:
            授权 URL
        """
        params = {
            "appid": self._cfg.corpid,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": scope,
            "state": state,
        }
        return f"https://open.weixin.qq.com/connect/oauth2/authorize?{urllib.parse.urlencode(params)}#wechat_redirect"

    async def get_user_info_by_code(
        self,
        code: str,
    ) -> Dict[str, Any]:
        """
        通过 OAuth2 授权码获取用户信息

        Args:
            code: OAuth2 授权码

        Returns:
            用户信息字典（userid, name, department, ...）
        """
        await self._get_access_token()
        url = f"{self.BASE_API}/user/getuserinfo"
        params = {
            "access_token": self._access_token,
            "code": code,
        }
        result = await self._make_request("GET", url, params=params)
        errcode = result.get("errcode", 0)
        if errcode != 0:
            raise WeComError(errcode, result.get("errmsg", "OAuth2 获取用户信息失败"), result)

        user_id = result.get("userid", "")
        info: Dict[str, Any] = {
            "user_id": user_id,
            "device_id": result.get("device_id", ""),
            "user_ticket": result.get("user_ticket", ""),
        }

        # 如果有 user_ticket，可以获取更详细的信息
        if user_id:
            detailed = await self.get_user_info(user_id)
            if detailed and "error" not in detailed:
                info.update(detailed)

        return info

    async def get_user_info_with_ticket(
        self,
        user_ticket: str,
    ) -> Dict[str, Any]:
        """
        使用 user_ticket 获取敏感信息（如手机号）

        Args:
            user_ticket: 用户票据

        Returns:
            用户敏感信息
        """
        await self._get_access_token()
        url = f"{self.BASE_API}/user/getuserdetail"
        payload = {"user_ticket": user_ticket}
        params = {"access_token": self._access_token}
        result = await self._make_request("POST", url, data=payload, params=params)
        errcode = result.get("errcode", 0)
        if errcode != 0:
            raise WeComError(errcode, result.get("errmsg", "获取用户敏感信息失败"), result)
        return result

    # ============================================================
    # 群聊管理
    # ============================================================

    async def create_app_chat(
        self,
        name: str,
        owner: str,
        user_list: List[str],
        chat_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        创建应用群聊

        Args:
            name: 群聊名称
            owner: 群主 userid
            user_list: 成员 userid 列表（至少包含群主）
            chat_id: 群聊 ID（可选）

        Returns:
            chat_id 或 None
        """
        await self._get_access_token()
        url = f"{self.BASE_API}/appchat/create"
        payload: Dict[str, Any] = {
            "name": name,
            "owner": owner,
            "userlist": user_list,
        }
        if chat_id:
            payload["chatid"] = chat_id
        params = {"access_token": self._access_token}
        result = await self._make_request("POST", url, data=payload, params=params)
        errcode = result.get("errcode", 0)
        if errcode != 0:
            raise WeComError(errcode, result.get("errmsg", "创建群聊失败"), result)
        return result.get("chatid")

    async def update_app_chat(
        self,
        chat_id: str,
        name: Optional[str] = None,
        owner: Optional[str] = None,
        add_user_list: Optional[List[str]] = None,
        del_user_list: Optional[List[str]] = None,
    ) -> bool:
        """
        修改群聊信息

        Args:
            chat_id: 群聊 ID
            name: 新群名
            owner: 新群主
            add_user_list: 新增成员列表
            del_user_list: 移除成员列表

        Returns:
            是否成功
        """
        await self._get_access_token()
        url = f"{self.BASE_API}/appchat/update"
        payload: Dict[str, Any] = {"chatid": chat_id}
        if name is not None:
            payload["name"] = name
        if owner is not None:
            payload["owner"] = owner
        if add_user_list:
            payload["add_user_list"] = add_user_list
        if del_user_list:
            payload["del_user_list"] = del_user_list
        params = {"access_token": self._access_token}
        result = await self._make_request("POST", url, data=payload, params=params)
        errcode = result.get("errcode", 0)
        if errcode != 0:
            raise WeComError(errcode, result.get("errmsg", "修改群聊失败"), result)
        return True

    async def send_app_chat_message(
        self,
        chat_id: str,
        msg_type: str,
        content: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        发送应用群聊消息

        Args:
            chat_id: 群聊 ID
            msg_type: 消息类型
            content: 消息内容

        Returns:
            API 响应
        """
        await self._get_access_token()
        url = f"{self.BASE_API}/appchat/send"
        payload = {
            "chatid": chat_id,
            "msgtype": msg_type,
            msg_type: content,
        }
        params = {"access_token": self._access_token}
        result = await self._make_request("POST", url, data=payload, params=params)
        errcode = result.get("errcode", 0)
        if errcode != 0:
            raise WeComError(errcode, result.get("errmsg", "发送群聊消息失败"), result)
        return result

    # ============================================================
    # 外部联系人（可选功能）
    # ============================================================

    async def get_external_contact_list(
        self,
        user_id: str,
    ) -> List[Dict[str, Any]]:
        """
        获取外部联系人列表

        Args:
            user_id: 企业成员 userid

        Returns:
            外部联系人列表
        """
        await self._get_access_token()
        url = f"{self.BASE_API}/externalcontact/list"
        params = {"access_token": self._access_token, "userid": user_id}
        result = await self._make_request("GET", url, params=params)
        errcode = result.get("errcode", 0)
        if errcode != 0:
            raise WeComError(errcode, result.get("errmsg", "获取外部联系人失败"), result)
        return result.get("external_userid", [])

    # ============================================================
    # 工具方法
    # ============================================================

    def get_capabilities(self) -> set:
        """获取支持的能力集"""
        return self._capabilities

    def supports_capability(self, capability: ChannelCapability) -> bool:
        """检查是否支持指定能力"""
        return capability in self._capabilities

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """获取频率限制信息"""
        return {
            "type": "企业微信 API",
            "default_limit": 20000,
            "window_seconds": 86400,
            "message": "企业微信接口频率限制请参考官方文档: https://developer.work.weixin.qq.com/document/path/90670",
        }

    def get_config(self) -> WeComConfig:
        """获取当前配置"""
        return self._cfg

    def set_debug_mode(self, enabled: bool) -> None:
        """设置调试模式"""
        level = logging.DEBUG if enabled else logging.INFO
        logging.getLogger("agi_unified_framework.channel.adapters.wechat_work").setLevel(level)
        self._logger.setLevel(level)

    async def test_connection(self) -> Dict[str, Any]:
        """测试连接"""
        try:
            token = await self._get_access_token(force=True)
            return {
                "status": "ok",
                "token_prefix": token[:10] + "...",
                "expires_in": int(self._token_expires_at - time.time()),
                "corpid": self._cfg.corpid[:6] + "...",
                "agent_id": self._cfg.agent_id,
            }
        except WeComError as e:
            return {"status": "error", "code": e.code, "message": e.message}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def process_callback(
        self,
        event_type: str,
        event_data: Dict[str, Any],
    ) -> Optional[UniversalMessage]:
        """
        处理回调事件

        Args:
            event_type: 事件类型
            event_data: 事件数据

        Returns:
            UniversalMessage 或 None
        """
        if event_type in ("message", "im.message.receive_v1"):
            return self._transform_incoming_message(event_data)
        elif event_type == "enter_agent":
            # 用户进入应用
            return self._create_system_message(event_data, "用户进入应用")
        elif event_type == "subscribe":
            # 关注事件
            return self._create_system_message(event_data, "用户关注")
        elif event_type == "unsubscribe":
            # 取消关注
            return self._create_system_message(event_data, "用户取消关注")
        return None

    def _create_system_message(
        self,
        event_data: Dict[str, Any],
        text: str,
    ) -> UniversalMessage:
        """创建系统消息"""
        content = MessageContent(text=f"[系统] {text}")
        metadata = MessageMetadata(
            message_id="sys_" + str(int(time.time())),
            channel_id="wecom",
            timestamp=time.time(),
            direction=MessageDirection.INBOUND,
            message_type=MessageType.SYSTEM,
            raw_event=event_data,
        )
        return UniversalMessage(content=content, metadata=metadata)

    async def validate_webhook_signature(
        self,
        headers: Dict[str, str],
        body: bytes,
    ) -> bool:
        """
        验证 Webhook 签名

        Args:
            headers: HTTP 请求头
            body: 请求体

        Returns:
            签名是否有效
        """
        msg_signature = headers.get("msg_signature", headers.get("X-WeCom-Signature", ""))
        timestamp = headers.get("timestamp", headers.get("X-WeCom-Timestamp", ""))
        nonce = headers.get("nonce", headers.get("X-WeCom-Nonce", ""))

        if not msg_signature or not timestamp or not nonce:
            return False

        try:
            root = ET.fromstring(body.decode("utf-8"))
            encrypt_node = root.find("Encrypt")
            if encrypt_node is None:
                return False
            encrypt_content = encrypt_node.text or ""
            return self.verify_callback_signature(msg_signature, timestamp, nonce, encrypt_content)
        except Exception:
            return False

    # ============================================================
    # 类方法: 快捷构造
    # ============================================================

    @classmethod
    def from_app_config(
        cls,
        corpid: str,
        corp_secret: str,
        agent_id: str,
        contact_secret: str = "",
        encoding_aes_key: str = "",
        token: str = "",
    ) -> "WeComAdapter":
        """
        通过应用配置快速创建适配器

        Args:
            corpid: 企业 ID
            corp_secret: 应用 Secret
            agent_id: 应用 AgentId
            contact_secret: 通讯录 Secret
            encoding_aes_key: 回调加解密密钥
            token: 回调验证 Token

        Returns:
            WeComAdapter 实例
        """
        config = WeComConfig(
            channel_id="wecom",
            corpid=corpid,
            corp_secret=corp_secret,
            agent_id=agent_id,
            contact_secret=contact_secret,
            encoding_aes_key=encoding_aes_key,
            token=token,
        )
        return cls(config)

    def __repr__(self) -> str:
        return (
            f"WeComAdapter("
            f"corpid={self._cfg.corpid[:6]}..., "
            f"agent_id={self._cfg.agent_id}, "
            f"connected={self._state == ConnectionState.CONNECTED})"
        )
