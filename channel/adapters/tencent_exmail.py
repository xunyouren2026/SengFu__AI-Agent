"""
腾讯企业邮适配器模块

该模块提供腾讯企业邮(Exmail)的通道适配器实现，支持：
- SMTP发送邮件
- IMAP接收邮件
- 文件夹管理
- 联系人同步
- 邮件搜索
- 附件处理

API文档: https://exmail.qq.com/qy_mng_logic/doc
"""

from __future__ import annotations

import asyncio
import base64
import email
import email.encoders
import email.header
import email.mime.base
import email.mime.image
import email.mime.multipart
import email.mime.text
import imaplib
import logging
import quopri
import re
import smtplib
import ssl
import time
from dataclasses import dataclass, field
from email.message import Message
from email.parser import Parser
from email.policy import default
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import unquote

import aioimaplib
import aiosmtplib

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
class ExmailConfig(ChannelConfig):
    """腾讯企业邮配置类
    
    Attributes:
        # SMTP设置
        smtp_host: SMTP服务器地址
        smtp_port: SMTP端口
        smtp_username: SMTP用户名
        smtp_password: SMTP密码
        smtp_use_tls: 是否使用TLS
        smtp_use_ssl: 是否使用SSL
        smtp_timeout: SMTP超时时间
        
        # IMAP设置
        imap_host: IMAP服务器地址
        imap_port: IMAP端口
        imap_username: IMAP用户名
        imap_password: IMAP密码
        imap_use_ssl: 是否使用SSL
        imap_timeout: IMAP超时时间
        imap_mailbox: 默认邮箱文件夹
        
        # 邮件默认设置
        default_from_email: 默认发件人邮箱
        default_from_name: 默认发件人名称
        max_attachment_size_mb: 最大附件大小（MB）
        email_signature: 邮件签名
        
        # 轮询设置
        polling_interval: 轮询间隔（秒）
        mark_as_read: 是否标记为已读
        
        # 安全设置
        verify_ssl: 是否验证SSL证书
    """
    # SMTP Settings
    smtp_host: str = "smtp.exmail.qq.com"
    smtp_port: int = 465
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    smtp_use_ssl: bool = True
    smtp_timeout: int = 30
    
    # IMAP Settings
    imap_host: str = "imap.exmail.qq.com"
    imap_port: int = 993
    imap_username: str = ""
    imap_password: str = ""
    imap_use_ssl: bool = True
    imap_timeout: int = 30
    imap_mailbox: str = "INBOX"
    
    # Email Defaults
    default_from_email: str = ""
    default_from_name: str = ""
    max_attachment_size_mb: int = 50
    email_signature: str = ""
    
    # Polling Settings
    polling_interval: int = 60
    mark_as_read: bool = False
    
    # Security
    verify_ssl: bool = True


class ExmailError(Exception):
    """腾讯企业邮错误"""
    pass


@dataclass
class ExmailContact:
    """腾讯企业邮联系人
    
    Attributes:
        email: 邮箱地址
        name: 姓名
        department: 部门
        phone: 电话
    """
    email: str
    name: str = ""
    department: str = ""
    phone: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "email": self.email,
            "name": self.name,
            "department": self.department,
            "phone": self.phone,
        }


@dataclass
class ExmailFolder:
    """邮箱文件夹
    
    Attributes:
        name: 文件夹名称
        path: 文件夹路径
        message_count: 邮件数量
        unread_count: 未读数量
    """
    name: str
    path: str
    message_count: int = 0
    unread_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "path": self.path,
            "message_count": self.message_count,
            "unread_count": self.unread_count,
        }


class TencentExmailAdapter(ChannelAdapter):
    """腾讯企业邮适配器
    
    提供腾讯企业邮的统一接口，支持SMTP发送邮件、IMAP接收邮件、
    文件夹管理、联系人同步等功能。
    
    Example:
        config = ExmailConfig(
            smtp_username="user@company.com",
            smtp_password="password",
            imap_username="user@company.com",
            imap_password="password",
        )
        adapter = TencentExmailAdapter(config)
        await adapter.connect()
        
        # 发送邮件
        await adapter.send_email(
            to="recipient@example.com",
            subject="测试邮件",
            body="这是一封测试邮件",
        )
        
        # 接收邮件
        messages = await adapter.receive_emails()
    """
    
    DEFAULT_IMAP_PORT = 993
    DEFAULT_SMTP_PORT = 465
    
    def __init__(self, config: ExmailConfig) -> None:
        super().__init__(config)
        self._cfg = config
        self._smtp_connection: Optional[aiosmtplib.SMTP] = None
        self._imap_connection: Optional[aioimaplib.IMAP4_SSL] = None
        self._running = False
    
    def _initialize_capabilities(self) -> None:
        """初始化适配器能力"""
        self._capabilities = {
            ChannelCapability.TEXT_MESSAGES,
            ChannelCapability.HTML_MESSAGES,
            ChannelCapability.FILE_ATTACHMENTS,
            ChannelCapability.DIRECT_MESSAGES,
            ChannelCapability.CHANNEL_INFO,
        }
    
    def _create_ssl_context(self) -> ssl.SSLContext:
        """创建SSL上下文"""
        context = ssl.create_default_context()
        if not self._cfg.verify_ssl:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        return context
    
    async def _connect_smtp(self) -> None:
        """连接SMTP服务器"""
        try:
            context = self._create_ssl_context()
            
            if self._cfg.smtp_use_ssl:
                self._smtp_connection = aiosmtplib.SMTP(
                    hostname=self._cfg.smtp_host,
                    port=self._cfg.smtp_port or self.DEFAULT_SMTP_PORT,
                    use_tls=True,
                    tls_context=context,
                )
            else:
                self._smtp_connection = aiosmtplib.SMTP(
                    hostname=self._cfg.smtp_host,
                    port=self._cfg.smtp_port or 587,
                    use_tls=False,
                )
            
            await self._smtp_connection.connect()
            
            if not self._cfg.smtp_use_ssl and self._cfg.smtp_use_tls:
                await self._smtp_connection.starttls(context=context)
            
            if self._cfg.smtp_username and self._cfg.smtp_password:
                await self._smtp_connection.login(
                    self._cfg.smtp_username,
                    self._cfg.smtp_password,
                )
            
            self._logger.info(f"已连接到SMTP服务器 {self._cfg.smtp_host}")
        except Exception as e:
            self._logger.error(f"SMTP连接失败: {e}")
            raise
    
    async def _disconnect_smtp(self) -> None:
        """断开SMTP连接"""
        if self._smtp_connection:
            try:
                await self._smtp_connection.quit()
            except Exception as e:
                self._logger.warning(f"断开SMTP连接时出错: {e}")
            finally:
                self._smtp_connection = None
    
    async def _connect_imap(self) -> None:
        """连接IMAP服务器"""
        try:
            context = self._create_ssl_context()
            
            self._imap_connection = aioimaplib.IMAP4_SSL(
                host=self._cfg.imap_host,
                port=self._cfg.imap_port or self.DEFAULT_IMAP_PORT,
                ssl_context=context,
            )
            
            await self._imap_connection.wait_hello_from_server()
            
            if self._cfg.imap_username and self._cfg.imap_password:
                await self._imap_connection.login(
                    self._cfg.imap_username,
                    self._cfg.imap_password,
                )
            
            self._logger.info(f"已连接到IMAP服务器 {self._cfg.imap_host}")
        except Exception as e:
            self._logger.error(f"IMAP连接失败: {e}")
            raise
    
    async def _disconnect_imap(self) -> None:
        """断开IMAP连接"""
        if self._imap_connection:
            try:
                await self._imap_connection.logout()
            except Exception as e:
                self._logger.warning(f"断开IMAP连接时出错: {e}")
            finally:
                self._imap_connection = None
    
    async def _connect_impl(self) -> bool:
        """实现连接逻辑"""
        try:
            if self._cfg.smtp_username:
                await self._connect_smtp()
            if self._cfg.imap_username:
                await self._connect_imap()
            return True
        except Exception as e:
            self._logger.error(f"连接失败: {e}")
            return False
    
    async def _disconnect_impl(self) -> None:
        """实现断开连接逻辑"""
        self._running = False
        await self._disconnect_smtp()
        await self._disconnect_imap()
    
    async def _health_check_impl(self) -> bool:
        """实现健康检查逻辑"""
        try:
            if self._smtp_connection:
                # 发送NOOP命令检查连接
                await self._smtp_connection.noop()
            if self._imap_connection:
                # 发送NOOP命令检查连接
                await self._imap_connection.noop()
            return True
        except Exception:
            return False
    
    async def _send_impl(self, message: UniversalMessage, priority: MessagePriority) -> SendResult:
        """实现发送逻辑"""
        try:
            to_email = message.get_context("to_email")
            if not to_email:
                return SendResult(
                    success=False,
                    error="缺少收件人地址",
                    error_code="MISSING_RECIPIENT",
                )
            
            to_emails = to_email if isinstance(to_email, list) else [to_email]
            cc = message.get_context("cc")
            bcc = message.get_context("bcc")
            subject = message.get_context("subject") or "无主题"
            
            # 构建邮件
            if message.content.html:
                msg = email.mime.multipart.MIMEMultipart("mixed")
                msg["Subject"] = subject
                msg["From"] = self._format_address(
                    message.get_context("from_email") or self._cfg.default_from_email,
                    message.get_context("from_name") or self._cfg.default_from_name,
                )
                msg["To"] = ", ".join(to_emails)
                
                if cc:
                    msg["Cc"] = ", ".join(cc) if isinstance(cc, list) else cc
                
                # 添加HTML内容
                html_part = email.mime.text.MIMEText(
                    message.content.html,
                    "html",
                    "utf-8",
                )
                msg.attach(html_part)
                
                # 添加纯文本内容
                if message.content.get_primary_text():
                    text_part = email.mime.text.MIMEText(
                        message.content.get_primary_text(),
                        "plain",
                        "utf-8",
                    )
                    msg.attach(text_part)
            else:
                msg = email.mime.text.MIMEText(
                    message.content.get_primary_text() or "",
                    "plain",
                    "utf-8",
                )
                msg["Subject"] = subject
                msg["From"] = self._format_address(
                    message.get_context("from_email") or self._cfg.default_from_email,
                    message.get_context("from_name") or self._cfg.default_from_name,
                )
                msg["To"] = ", ".join(to_emails)
                
                if cc:
                    msg["Cc"] = ", ".join(cc) if isinstance(cc, list) else cc
            
            # 添加附件
            for attachment in message.content.attachments:
                if attachment.data:
                    part = email.mime.base.MIMEBase("application", "octet-stream")
                    part.set_payload(attachment.data)
                    email.encoders.encode_base64(part)
                    part.add_header(
                        "Content-Disposition",
                        f"attachment; filename={attachment.name}",
                    )
                    msg.attach(part)
            
            # 发送邮件
            if not self._smtp_connection:
                await self._connect_smtp()
            
            await self._smtp_connection.send_message(msg)
            
            return SendResult(
                success=True,
                message_id=str(time.time()),
                timestamp=time.time(),
            )
            
        except Exception as e:
            self._logger.error(f"发送邮件失败: {e}")
            return SendResult(
                success=False,
                error=str(e),
                error_code="SEND_ERROR",
            )
    
    async def _receive_impl(self, payload: Optional[Dict]) -> ReceiveResult:
        """实现接收逻辑"""
        try:
            if not self._imap_connection:
                return ReceiveResult(success=False, error="IMAP未连接")
            
            mailbox = payload.get("mailbox") if payload else self._cfg.imap_mailbox
            limit = payload.get("limit", 10) if payload else 10
            
            await self._imap_connection.select(mailbox)
            
            # 搜索未读邮件
            status, message_ids = await self._imap_connection.search("UNSEEN")
            
            if status != "OK":
                return ReceiveResult(success=False, error=f"搜索失败: {status}")
            
            messages = []
            msg_ids = message_ids[0].split()[-limit:]
            
            for msg_id in msg_ids:
                try:
                    status, msg_data = await self._imap_connection.fetch(
                        msg_id,
                        "(RFC822)",
                    )
                    if status == "OK" and msg_data:
                        raw_email = msg_data[0][1] if isinstance(msg_data[0], tuple) else msg_data[0]
                        email_message = email.message_from_bytes(raw_email)
                        universal_msg = self._transform_incoming_message(email_message)
                        messages.append(universal_msg)
                        
                        if self._cfg.mark_as_read:
                            await self._imap_connection.store(msg_id, "+FLAGS", "\\Seen")
                except Exception as e:
                    self._logger.warning(f"获取邮件 {msg_id} 失败: {e}")
            
            return ReceiveResult(success=True, messages=messages)
            
        except Exception as e:
            self._logger.error(f"接收邮件失败: {e}")
            return ReceiveResult(success=False, error=str(e))
    
    def _transform_incoming_message(self, email_msg: email.message.Message) -> UniversalMessage:
        """转换收到的邮件为通用消息"""
        # 解析主题
        subject_header = email_msg.get("Subject", "")
        decoded_subject = email.header.decode_header(subject_header)
        subject = ""
        for part, charset in decoded_subject:
            if isinstance(part, bytes):
                subject += part.decode(charset or "utf-8", errors="replace")
            else:
                subject += part
        
        # 解析发件人
        from_header = email_msg.get("From", "")
        from_email = self._extract_email(from_header)
        from_name = self._extract_name(from_header)
        
        # 解析正文
        body_text = ""
        body_html = ""
        attachments = []
        
        if email_msg.is_multipart():
            for part in email_msg.walk():
                content_type = part.get_content_type()
                content_disposition = part.get_content_disposition()
                
                if content_disposition == "attachment":
                    attachment = self._parse_attachment(part)
                    if attachment:
                        attachments.append(attachment)
                elif content_type == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        body_text = payload.decode(charset, errors="replace")
                elif content_type == "text/html":
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        body_html = payload.decode(charset, errors="replace")
        else:
            payload = email_msg.get_payload(decode=True)
            if payload:
                charset = email_msg.get_content_charset() or "utf-8"
                content_type = email_msg.get_content_type()
                if content_type == "text/plain":
                    body_text = payload.decode(charset, errors="replace")
                elif content_type == "text/html":
                    body_html = payload.decode(charset, errors="replace")
        
        # 创建通用消息
        content = MessageContent(
            text=body_text,
            html=body_html,
            attachments=attachments,
        )
        
        metadata = MessageMetadata(
            message_id=email_msg.get("Message-ID", str(time.time())),
            channel_id="tencent_exmail",
            timestamp=time.time(),
            direction=MessageDirection.INBOUND,
            message_type=MessageType.EMAIL,
            sender=UserIdentity(
                user_id=from_email,
                username=from_email,
                email=from_email,
                display_name=from_name or from_email,
            ),
        )
        
        message = UniversalMessage(content=content, metadata=metadata)
        message.set_context("subject", subject)
        message.set_context("from_email", from_email)
        
        return message
    
    def _parse_attachment(self, part: email.message.Message) -> Optional[Attachment]:
        """解析附件"""
        filename = part.get_filename()
        if not filename:
            return None
        
        # 解码文件名
        decoded_name = email.header.decode_header(filename)
        filename = ""
        for part_name, charset in decoded_name:
            if isinstance(part_name, bytes):
                filename += part_name.decode(charset or "utf-8", errors="replace")
            else:
                filename += part_name
        
        payload = part.get_payload(decode=True)
        if not payload:
            return None
        
        content_type = part.get_content_type()
        
        attachment_type = AttachmentType.FILE
        if "image" in content_type:
            attachment_type = AttachmentType.IMAGE
        elif "audio" in content_type:
            attachment_type = AttachmentType.AUDIO
        elif "video" in content_type:
            attachment_type = AttachmentType.VIDEO
        
        return Attachment(
            name=filename,
            type=attachment_type,
            data=payload,
            content_type=content_type,
            size=len(payload),
        )
    
    def _format_address(self, email_address: str, name: str = "") -> str:
        """格式化邮箱地址"""
        if name:
            return f"{name} <{email_address}>"
        return email_address
    
    def _extract_email(self, header: str) -> str:
        """从头部提取邮箱地址"""
        match = re.search(r"<([^>]+)>", header)
        if match:
            return match.group(1)
        return header.strip()
    
    def _extract_name(self, header: str) -> str:
        """从头部提取名称"""
        if "<" in header and ">" in header:
            name = header.split("<")[0].strip().strip('"').strip()
            return name
        return ""
    
    # ========== 邮件操作 ==========
    
    async def send_email(
        self,
        to: Union[str, List[str]],
        subject: str,
        body: str,
        html_body: Optional[str] = None,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        attachments: Optional[List[str]] = None,
    ) -> SendResult:
        """发送邮件
        
        Args:
            to: 收件人地址
            subject: 主题
            body: 正文
            html_body: HTML正文
            cc: 抄送
            bcc: 密送
            attachments: 附件路径列表
            
        Returns:
            发送结果
        """
        msg = UniversalMessage(
            content=MessageContent(text=body, html=html_body),
        )
        msg.set_context("to_email", to)
        msg.set_context("subject", subject)
        
        if cc:
            msg.set_context("cc", cc)
        if bcc:
            msg.set_context("bcc", bcc)
        
        return await self.send(msg)
    
    async def receive_emails(
        self,
        mailbox: Optional[str] = None,
        limit: int = 10,
    ) -> List[UniversalMessage]:
        """接收邮件
        
        Args:
            mailbox: 邮箱文件夹
            limit: 数量限制
            
        Returns:
            邮件列表
        """
        payload = {
            "mailbox": mailbox or self._cfg.imap_mailbox,
            "limit": limit,
        }
        
        result = await self.receive(payload)
        return result.messages if result.success else []
    
    # ========== 文件夹管理 ==========
    
    async def list_folders(self) -> List[ExmailFolder]:
        """列出邮箱文件夹
        
        Returns:
            文件夹列表
        """
        if not self._imap_connection:
            raise ExmailError("IMAP未连接")
        
        status, folders = await self._imap_connection.list()
        
        if status != "OK":
            raise ExmailError(f"获取文件夹列表失败: {status}")
        
        result = []
        for folder_data in folders:
            # 解析文件夹信息
            parts = folder_data.decode().split(' "/" ')
            if len(parts) >= 2:
                flags = parts[0]
                name = parts[1].strip('"')
                
                folder = ExmailFolder(
                    name=name,
                    path=name,
                )
                result.append(folder)
        
        return result
    
    async def create_folder(self, name: str) -> bool:
        """创建文件夹
        
        Args:
            name: 文件夹名称
            
        Returns:
            是否成功
        """
        if not self._imap_connection:
            raise ExmailError("IMAP未连接")
        
        status, _ = await self._imap_connection.create(name)
        return status == "OK"
    
    async def delete_folder(self, name: str) -> bool:
        """删除文件夹
        
        Args:
            name: 文件夹名称
            
        Returns:
            是否成功
        """
        if not self._imap_connection:
            raise ExmailError("IMAP未连接")
        
        status, _ = await self._imap_connection.delete(name)
        return status == "OK"
    
    async def move_email(
        self,
        message_id: str,
        destination: str,
    ) -> bool:
        """移动邮件
        
        Args:
            message_id: 邮件ID
            destination: 目标文件夹
            
        Returns:
            是否成功
        """
        if not self._imap_connection:
            raise ExmailError("IMAP未连接")
        
        await self._imap_connection.select(self._cfg.imap_mailbox)
        status, _ = await self._imap_connection.copy(message_id, destination)
        
        if status == "OK":
            await self._imap_connection.store(message_id, "+FLAGS", "\\Deleted")
            await self._imap_connection.expunge()
            return True
        
        return False
    
    # ========== 工具方法 ==========
    
    def get_capabilities(self):
        """获取适配器能力"""
        return self._capabilities
    
    def supports_capability(self, capability) -> bool:
        """检查是否支持特定能力"""
        return capability in self._capabilities
    
    def __repr__(self) -> str:
        return f"TencentExmailAdapter(smtp={self._cfg.smtp_host}, imap={self._cfg.imap_host})"


# ========== CLI测试代码 ==========

async def test_tencent_exmail():
    """测试腾讯企业邮适配器"""
    import os
    
    # 从环境变量获取配置
    smtp_username = os.environ.get("EXMAIL_SMTP_USERNAME", "user@company.com")
    smtp_password = os.environ.get("EXMAIL_SMTP_PASSWORD", "password")
    imap_username = os.environ.get("EXMAIL_IMAP_USERNAME", "user@company.com")
    imap_password = os.environ.get("EXMAIL_IMAP_PASSWORD", "password")
    
    config = ExmailConfig(
        smtp_username=smtp_username,
        smtp_password=smtp_password,
        imap_username=imap_username,
        imap_password=imap_password,
    )
    
    adapter = TencentExmailAdapter(config)
    
    try:
        # 连接
        print("正在连接腾讯企业邮...")
        connected = await adapter.connect()
        print(f"连接结果: {connected}")
        
        if not connected:
            print("连接失败，请检查配置")
            return
        
        # 健康检查
        healthy = await adapter.health_check()
        print(f"健康检查: {healthy}")
        
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
    asyncio.run(test_tencent_exmail())
