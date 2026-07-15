"""Email Adapter Module for AGI Unified Framework

This module provides a comprehensive email channel adapter supporting both SMTP sending
and IMAP receiving capabilities. It integrates with the unified messaging architecture
to provide seamless email communication across the platform.

Key Features:
- SMTP/SMTPS support for sending emails
- IMAP/IMAPS support for receiving emails
- HTML and plain text email support
- File attachment handling
- Email thread management
- Message filtering and search
- Email queue management for async sending
"""

from __future__ import annotations
import asyncio
import base64
import email
import email.encoders
import email.header
import email.mime.audio
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
    MessageDirection,
    MessageMetadata,
    MessageType,
    UniversalMessage,
    UserIdentity,
    ChannelIdentity,
)

logger = logging.getLogger(__name__)


@dataclass
class EmailConfig(ChannelConfig):
    """Email channel configuration."""
    
    # SMTP Settings
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    smtp_use_ssl: bool = False
    smtp_timeout: int = 30
    
    # IMAP Settings
    imap_host: str = "imap.gmail.com"
    imap_port: int = 993
    imap_username: str = ""
    imap_password: str = ""
    imap_use_ssl: bool = True
    imap_timeout: int = 30
    imap_mailbox: str = "INBOX"
    
    # Email Defaults
    default_from_email: str = ""
    default_from_name: str = ""
    max_attachment_size_mb: int = 25
    email_signature: str = ""
    
    # Polling Settings
    polling_interval: int = 60
    mark_as_read: bool = False
    move_processed_to: Optional[str] = None
    
    # Threading Settings
    thread_by_subject: bool = True
    thread_reply_tracking: bool = True
    
    # Security
    verify_ssl: bool = True
    allow_less_secure_apps: bool = False


class EmailMessageBuilder:
    """Helper class to build email messages with various content types."""
    
    @staticmethod
    def create_text_email(
        subject: str,
        body: str,
        from_email: str,
        to_emails: List[str],
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        reply_to: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> email.mime.multipart.MIMEMultipart:
        msg = email.mime.multipart.MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = EmailMessageBuilder._format_address(from_email)
        msg["To"] = ", ".join(to_emails)
        if cc:
            msg["Cc"] = ", ".join(cc)
        if bcc:
            msg["Bcc"] = ", ".join(bcc)
        if reply_to:
            msg["Reply-To"] = reply_to
        if headers:
            for key, value in headers.items():
                msg[key] = value
        
        plain_part = email.mime.text.MIMEText(body, "plain", "utf-8")
        msg.attach(plain_part)
        return msg
    
    @staticmethod
    def create_html_email(
        subject: str,
        html_body: str,
        plain_body: Optional[str] = None,
        from_email: str = "",
        to_emails: Optional[List[str]] = None,
        cc: Optional[List[str]] = None,
        reply_to: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> email.mime.multipart.MIMEMultipart:
        msg = email.mime.multipart.MIMEMultipart("mixed")
        msg["Subject"] = subject
        msg["From"] = EmailMessageBuilder._format_address(from_email)
        msg["To"] = ", ".join(to_emails) if to_emails else ""
        if cc:
            msg["Cc"] = ", ".join(cc)
        if reply_to:
            msg["Reply-To"] = reply_to
        if headers:
            for key, value in headers.items():
                msg[key] = value
        
        if plain_body:
            plain_part = email.mime.text.MIMEText(plain_body, "plain", "utf-8")
            msg.attach(plain_part)
        
        html_part = email.mime.text.MIMEText(html_body, "html", "utf-8")
        msg.attach(html_part)
        return msg
    
    @staticmethod
    def add_attachment(
        msg: email.mime.multipart.MIMEMultipart,
        file_path: str,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
        encoding: str = "base64",
    ) -> bool:
        try:
            with open(file_path, "rb") as f:
                part = email.mime.base.MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            
            if encoding == "base64":
                email.encoders.encode_base64(part)
            elif encoding == "quoted-printable":
                email.encoders.encode_quopri(part)
            
            filename = filename or os.path.basename(file_path)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename*=UTF-8''{EmailMessageBuilder._encode_filename(filename)}",
            )
            
            if content_type:
                part.add_header("Content-Type", content_type)
            
            msg.attach(part)
            return True
        except Exception as e:
            logger.error(f"Failed to add attachment {file_path}: {e}")
            return False
    
    @staticmethod
    def add_inline_image(
        msg: email.mime.multipart.MIMEMultipart,
        file_path: str,
        content_id: Optional[str] = None,
    ) -> Optional[str]:
        try:
            with open(file_path, "rb") as f:
                image_data = f.read()
            
            content_id = content_id or f"image_{hashlib.md5(image_data).hexdigest()[:8]}@domain"
            image_type = "png" if file_path.endswith(".png") else "jpeg"
            
            part = email.mime.image.MIMEImage(image_data, image_type)
            part.add_header("Content-ID", f"<{content_id}>")
            part.add_header("Content-Disposition", "inline")
            msg.attach(part)
            return content_id
        except Exception as e:
            logger.error(f"Failed to add inline image {file_path}: {e}")
            return None
    
    @staticmethod
    def _format_address(email_address: str, name: Optional[str] = None) -> str:
        if not name:
            return email_address
        return f"{name} <{email_address}>"
    
    @staticmethod
    def _encode_filename(filename: str) -> str:
        encoded = email.header.encode_header(filename, "utf-8")
        return str(email.header.make_header(encoded))


class EmailThreadManager:
    """Manages email threading and conversation tracking."""
    
    THREAD_HEADER_RE = re.compile(r"(<[^>]+>)")
    SUBJECT_CLEANUP_RE = re.compile(r"^(Re:|Fwd?:|AW:|RE:|FW:)\s*", re.IGNORECASE)
    
    @staticmethod
    def get_thread_id(message: email.message.Message) -> Optional[str]:
        references = message.get("References", "")
        in_reply_to = message.get("In-Reply-To", "")
        
        if references:
            refs = references.split()
            return refs[-1].strip("<>") if refs else None
        elif in_reply_to:
            return in_reply_to.strip("<>")
        return None
    
    @staticmethod
    def get_parent_thread_id(message: email.message.Message) -> Optional[str]:
        references = message.get("References", "")
        if not references:
            return None
        refs = references.split()
        return refs[-1].strip("<>") if len(refs) > 1 else None
    
    @staticmethod
    def create_reply_thread_header(original_message: email.message.Message) -> Tuple[str, str]:
        subject = original_message.get("Subject", "")
        clean_subject = EmailThreadManager.normalize_subject(subject)
        thread_id = EmailThreadManager.get_thread_id(original_message) or EmailThreadManager.generate_thread_id()
        
        references = original_message.get("References", "")
        if references:
            new_references = f"{references} {thread_id}"
        else:
            in_reply_to = original_message.get("In-Reply-To", "")
            new_references = f"{in_reply_to} {thread_id}".strip()
        
        return thread_id, new_references
    
    @staticmethod
    def normalize_subject(subject: str) -> str:
        return EmailThreadManager.SUBJECT_CLEANUP_RE.sub("", subject).strip()
    
    @staticmethod
    def generate_thread_id() -> str:
        import uuid
        return f"<{uuid.uuid4().hex}@email.thread>"


class EmailAdapter(ChannelAdapter):
    """Email channel adapter supporting SMTP sending and IMAP receiving.
    
    This adapter provides a unified interface for email communication,
    supporting both synchronous and asynchronous operations.
    
    Example:
        config = EmailConfig(
            smtp_host="smtp.gmail.com",
            smtp_username="user@gmail.com",
            smtp_password="app_password",
            imap_host="imap.gmail.com",
            imap_username="user@gmail.com",
            imap_password="app_password",
        )
        adapter = EmailAdapter(config)
        await adapter.connect()
        
        # Send email
        message = UniversalMessage(...)
        result = await adapter.send(message)
        
        # Receive emails
        messages = await adapter.receive()
    """
    
    DEFAULT_IMAP_PORT = 993
    DEFAULT_SMTP_PORT = 587
    DEFAULT_SMTP_SSL_PORT = 465
    
    def __init__(self, config: EmailConfig) -> None:
        super().__init__(config)
        self._cfg = config
        self._smtp_connection: Optional[smtplib.SMTP] = None
        self._imap_connection: Optional[imaplib.IMAP4_SSL] = None
        self._async_imap: Optional[aioimaplib.IMAP4_SSL] = None
        self._email_queue: asyncio.Queue = asyncio.Queue()
        self._polling_task: Optional[asyncio.Task] = None
        self._running = False
    
    def _initialize_capabilities(self) -> None:
        self._capabilities = {
            ChannelCapability.TEXT_MESSAGES,
            ChannelCapability.HTML_MESSAGES,
            ChannelCapability.FILE_ATTACHMENTS,
            ChannelCapability.INLINE_IMAGES,
            ChannelCapability.DIRECT_MESSAGES,
            ChannelCapability.THREADING,
            ChannelCapability.SEARCH,
            ChannelCapability.EMAIL_THREADS,
            ChannelCapability.FORWARDING,
            ChannelCapability.SIGNATURES,
            ChannelCapability.CHANNEL_INFO,
        }
    
    async def _connect_impl(self) -> bool:
        try:
            if self._cfg.smtp_host and self._cfg.smtp_username:
                await self._connect_smtp()
            if self._cfg.imap_host and self._cfg.imap_username:
                await self._connect_imap()
            return True
        except Exception as e:
            self._logger.error(f"Failed to connect to email server: {e}")
            return False
    
    async def _disconnect_impl(self) -> None:
        self._running = False
        if self._polling_task:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
        await self._disconnect_smtp()
        await self._disconnect_imap()
    
    async def _connect_smtp(self) -> None:
        try:
            if self._cfg.smtp_use_ssl:
                context = self._create_ssl_context()
                self._smtp_connection = aiosmtplib.SMTP(
                    hostname=self._cfg.smtp_host,
                    port=self._cfg.smtp_port or self.DEFAULT_SMTP_SSL_PORT,
                    use_tls=False,
                    tls_context=context,
                )
            else:
                self._smtp_connection = aiosmtplib.SMTP(
                    hostname=self._cfg.smtp_host,
                    port=self._cfg.smtp_port or self.DEFAULT_SMTP_PORT,
                    use_tls=False,
                )
            
            await self._smtp_connection.connect()
            
            if not self._cfg.smtp_use_ssl and self._cfg.smtp_use_tls:
                await self._smtp_connection.starttls(context=self._create_ssl_context())
            
            if self._cfg.smtp_username and self._cfg.smtp_password:
                await self._smtp_connection.login(self._cfg.smtp_username, self._cfg.smtp_password)
            
            self._logger.info(f"Connected to SMTP server {self._cfg.smtp_host}:{self._cfg.smtp_port}")
        except Exception as e:
            self._logger.error(f"SMTP connection failed: {e}")
            raise
    
    async def _disconnect_smtp(self) -> None:
        if self._smtp_connection and self._smtp_connection.is_connected:
            try:
                await self._smtp_connection.quit()
            except Exception as e:
                self._logger.warning(f"Error disconnecting SMTP: {e}")
            finally:
                self._smtp_connection = None
    
    async def _connect_imap(self) -> None:
        try:
            context = self._create_ssl_context() if self._cfg.verify_ssl else None
            self._async_imap = aioimaplib.IMAP4_SSL(
                host=self._cfg.imap_host,
                port=self._cfg.imap_port or self.DEFAULT_IMAP_PORT,
                ssl_context=context,
            )
            
            await self._async_imap.starttls(ssl_context=context)
            
            if self._cfg.imap_username and self._cfg.imap_password:
                await self._async_imap.login(self._cfg.imap_username, self._cfg.imap_password)
            
            self._logger.info(f"Connected to IMAP server {self._cfg.imap_host}:{self._cfg.imap_port}")
        except Exception as e:
            self._logger.error(f"IMAP connection failed: {e}")
            raise
    
    async def _disconnect_imap(self) -> None:
        if self._async_imap and self._async_imap.state != "disconnected":
            try:
                await self._async_imap.logout()
            except Exception as e:
                self._logger.warning(f"Error disconnecting IMAP: {e}")
            finally:
                self._async_imap = None
    
    def _create_ssl_context(self) -> ssl.SSLContext:
        context = ssl.create_default_context()
        if not self._cfg.verify_ssl:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        return context
    
    async def _send_impl(self, message: UniversalMessage, priority: MessagePriority) -> SendResult:
        try:
            to_email = message.get_context("to_email")
            if not to_email:
                return SendResult(success=False, error="No recipient email address", error_code="MISSING_RECIPIENT")
            
            to_emails = to_email if isinstance(to_email, list) else [to_email]
            cc = message.get_context("cc")
            bcc = message.get_context("bcc")
            subject = message.get_context("subject") or "No Subject"
            
            if message.content.html:
                msg = EmailMessageBuilder.create_html_email(
                    subject=subject,
                    html_body=message.content.html,
                    plain_body=message.content.get_primary_text(),
                    from_email=message.get_context("from_email") or self._cfg.default_from_email,
                    to_emails=to_emails,
                    cc=cc.split(",") if cc else None,
                    reply_to=message.get_context("reply_to"),
                )
            else:
                msg = EmailMessageBuilder.create_text_email(
                    subject=subject,
                    body=message.content.get_primary_text(),
                    from_email=message.get_context("from_email") or self._cfg.default_from_email,
                    to_emails=to_emails,
                    cc=cc.split(",") if cc else None,
                    bcc=bcc.split(",") if bcc else None,
                    reply_to=message.get_context("reply_to"),
                )
            
            for attachment in message.content.attachments:
                if attachment.url:
                    EmailMessageBuilder.add_attachment(msg, attachment.url, attachment.name)
                elif attachment.data:
                    EmailMessageBuilder.add_inline_image(msg, attachment.data) if attachment.type == AttachmentType.IMAGE else EmailMessageBuilder.add_attachment(msg, attachment.data, attachment.name)
            
            if self._cfg.email_signature and message.content.get_primary_text():
                pass
            
            if priority == MessagePriority.HIGH:
                await self._send_email_immediate(msg, to_emails)
            else:
                await self._email_queue.put((msg, to_emails))
            
            return SendResult(
                success=True,
                message_id=message.metadata.message_id if message.metadata else str(time.time()),
                timestamp=time.time(),
            )
        except Exception as e:
            self._logger.error(f"Failed to send email: {e}")
            return SendResult(success=False, error=str(e), error_code=type(e).__name__)
    
    async def _send_email_immediate(self, msg: email.mime.multipart.MIMEMultipart, to_emails: List[str]) -> None:
        if not self._smtp_connection or not self._smtp_connection.is_connected:
            await self._connect_smtp()
        await self._smtp_connection.send_message(msg, to_addrs=to_emails)
    
    async def _process_email_queue(self) -> None:
        while self._running:
            try:
                msg, to_emails = await asyncio.wait_for(self._email_queue.get(), timeout=5.0)
                await self._send_email_immediate(msg, to_emails)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self._logger.error(f"Error processing email queue: {e}")
    
    async def _receive_impl(self, payload: Optional[Dict]) -> ReceiveResult:
        try:
            if not self._async_imap:
                return ReceiveResult(success=False, error="IMAP not connected")
            
            search_criteria = payload.get("search_criteria") if payload else "UNSEEN"
            mailbox = payload.get("mailbox") if payload else self._cfg.imap_mailbox
            
            await self._async_imap.select(mailbox)
            status, message_ids = await self._async_imap.search(search_criteria)
            
            if status != "OK":
                return ReceiveResult(success=False, error=f"Search failed: {status}")
            
            messages = []
            for msg_id in message_ids[0].split():
                try:
                    status, msg_data = await self._async_imap.fetch(msg_id, "(RFC822)")
                    if status == "OK" and msg_data:
                        raw_email = msg_data[0][1] if isinstance(msg_data[0], tuple) else msg_data[0]
                        email_message = email.message_from_bytes(raw_email)
                        universal_msg = self._transform_incoming_message(email_message)
                        messages.append(universal_msg)
                        
                        if not self._cfg.mark_as_read:
                            pass
                except Exception as e:
                    self._logger.warning(f"Failed to fetch message {msg_id}: {e}")
            
            return ReceiveResult(success=True, messages=messages)
        except Exception as e:
            self._logger.error(f"Failed to receive emails: {e}")
            return ReceiveResult(success=False, error=str(e))
    
    def _transform_incoming_message(self, email_msg: email.message.Message) -> UniversalMessage:
        subject = email.header.decode_header(email_msg.get("Subject", ""))[0]
        subject_text = subject[0]
        if isinstance(subject_text, bytes):
            subject_text = subject_text.decode(subject[1] or "utf-8", errors="replace")
        
        from_header = email_msg.get("From", "")
        from_email = EmailParser.extract_email(from_header)
        from_name = EmailParser.extract_name(from_header)
        
        to_header = email_msg.get("To", "")
        to_email = EmailParser.extract_email(to_header)
        
        message_id = email_msg.get("Message-ID", "").strip("<>")
        in_reply_to = email_msg.get("In-Reply-To", "").strip("<>")
        references = email_msg.get("References", "")
        
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
                elif content_disposition == "inline":
                    pass
                elif content_type == "text/plain":
                    payload = part.get_payload(decode=True)
                    charset = part.get_content_charset() or "utf-8"
                    body_text = payload.decode(charset, errors="replace")
                elif content_type == "text/html":
                    payload = part.get_payload(decode=True)
                    charset = part.get_content_charset() or "utf-8"
                    body_html = payload.decode(charset, errors="replace")
        else:
            payload = email_msg.get_payload(decode=True)
            charset = email_msg.get_content_charset() or "utf-8"
            content_type = email_msg.get_content_type()
            if content_type == "text/plain":
                body_text = payload.decode(charset, errors="replace")
            elif content_type == "text/html":
                body_html = payload.decode(charset, errors="replace")
            else:
                body_text = payload.decode(charset, errors="replace")
        
        from ...universal_message import MessageContent
        content = MessageContent(text=body_text, html=body_html, attachments=attachments)
        
        timestamp = EmailParser.parse_date(email_msg.get("Date", ""))
        
        metadata = MessageMetadata(
            message_id=message_id or str(time.time()),
            channel_id="email",
            channel_specific_id=message_id,
            timestamp=timestamp,
            direction=MessageDirection.INBOUND,
            message_type=MessageType.EMAIL,
            raw_event=dict(email_msg),
            sender=UserIdentity(user_id=from_email, username=from_email, email=from_email, display_name=from_name),
            channel=ChannelIdentity(channel_id=to_email, channel_type="email", email=to_email),
        )
        
        message = UniversalMessage(content=content, metadata=metadata)
        message.set_context("subject", subject_text)
        message.set_context("from_email", from_email)
        message.set_context("to_email", to_email)
        message.set_context("in_reply_to", in_reply_to)
        message.set_context("references", references)
        message.set_context("thread_id", EmailThreadManager.get_thread_id(email_msg))
        
        return message
    
    def _parse_attachment(self, part: email.message.Message) -> Optional[Attachment]:
        filename = part.get_filename()
        if not filename:
            return None
        
        filename = email.header.decode_header(filename)[0]
        if isinstance(filename, bytes):
            filename = filename.decode("utf-8", errors="replace")
        
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
    
    async def _health_check_impl(self) -> bool:
        try:
            if self._smtp_connection and self._smtp_connection.is_connected:
                pass
            if self._async_imap and self._async_imap.state == "authenticated":
                return True
            return False
        except Exception:
            return False
    
    async def send_email(
        self,
        to: Union[str, List[str]],
        subject: str,
        body: str,
        html_body: Optional[str] = None,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        attachments: Optional[List[str]] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> SendResult:
        to_emails = to if isinstance(to, list) else [to]
        msg = UniversalMessage(content=MessageContent(text=body, html=html_body))
        msg.set_context("to_email", to_emails)
        msg.set_context("subject", subject)
        if cc:
            msg.set_context("cc", ",".join(cc))
        if bcc:
            msg.set_context("bcc", ",".join(bcc))
        return await self.send(msg, priority=priority)
    
    async def reply_to_email(
        self,
        original_message: UniversalMessage,
        body: str,
        html_body: Optional[str] = None,
        attachments: Optional[List[str]] = None,
    ) -> SendResult:
        to_email = original_message.get_context("from_email")
        if not to_email:
            return SendResult(success=False, error="Cannot determine reply recipient")
        
        subject = original_message.get_context("subject", "No Subject")
        if not subject.lower().startswith("re:"):
            subject = f"Re: {subject}"
        
        return await self.send_email(
            to=to_email,
            subject=subject,
            body=body,
            html_body=html_body,
            attachments=attachments,
            priority=MessagePriority.HIGH,
        )
    
    async def forward_email(
        self,
        original_message: UniversalMessage,
        to: Union[str, List[str]],
        body: Optional[str] = None,
    ) -> SendResult:
        forward_text = body or "---------- Forwarded message ----------\n"
        forward_text += f"From: {original_message.get_context('from_email')}\n"
        forward_text += f"Date: {original_message.metadata.timestamp}\n"
        forward_text += f"Subject: {original_message.get_context('subject')}\n"
        forward_text += f"To: {original_message.get_context('to_email')}\n\n"
        forward_text += original_message.content.get_primary_text()
        
        subject = original_message.get_context("subject", "No Subject")
        if not subject.lower().startswith("fwd:"):
            subject = f"Fwd: {subject}"
        
        return await self.send_email(to=to, subject=subject, body=forward_text, attachments=[a.url for a in original_message.content.attachments if a.url])
    
    async def search_emails(
        self,
        criteria: str,
        mailbox: Optional[str] = None,
        limit: int = 50,
    ) -> List[UniversalMessage]:
        try:
            if not self._async_imap:
                return []
            
            mailbox = mailbox or self._cfg.imap_mailbox
            await self._async_imap.select(mailbox)
            
            status, message_ids = await self._async_imap.search(None, criteria)
            if status != "OK":
                return []
            
            messages = []
            msg_ids = message_ids[0].split()[-limit:]
            
            for msg_id in msg_ids:
                try:
                    status, msg_data = await self._async_imap.fetch(msg_id, "(RFC822)")
                    if status == "OK" and msg_data:
                        raw_email = msg_data[0][1] if isinstance(msg_data[0], tuple) else msg_data[0]
                        email_message = email.message_from_bytes(raw_email)
                        messages.append(self._transform_incoming_message(email_message))
                except Exception as e:
                    self._logger.warning(f"Failed to fetch message {msg_id}: {e}")
            
            return messages
        except Exception as e:
            self._logger.error(f"Failed to search emails: {e}")
            return []
    
    async def move_email(self, message_id: str, destination: str, source: Optional[str] = None) -> bool:
        try:
            if not self._async_imap:
                return False
            source = source or self._cfg.imap_mailbox
            await self._async_imap.select(source)
            await self._async_imap.copy(message_id, destination)
            await self._async_imap.store(message_id, "+FLAGS", "\\Deleted")
            await self._async_imap.expunge()
            return True
        except Exception as e:
            self._logger.error(f"Failed to move email: {e}")
            return False
    
    async def mark_as_read(self, message_id: str, mailbox: Optional[str] = None) -> bool:
        try:
            if not self._async_imap:
                return False
            mailbox = mailbox or self._cfg.imap_mailbox
            await self._async_imap.select(mailbox)
            await self._async_imap.store(message_id, "+FLAGS", "\\Seen")
            return True
        except Exception as e:
            self._logger.error(f"Failed to mark email as read: {e}")
            return False
    
    def start_polling(self) -> None:
        if self._polling_task:
            return
        self._running = True
        self._polling_task = asyncio.create_task(self._poll_loop())
    
    async def _poll_loop(self) -> None:
        while self._running:
            try:
                result = await self.receive()
                if result.success and result.messages:
                    for message in result.messages:
                        await self._handle_incoming_message(message)
                await asyncio.sleep(self._cfg.polling_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Polling error: {e}")
                await asyncio.sleep(self._cfg.polling_interval)
    
    async def _handle_incoming_message(self, message: UniversalMessage) -> None:
        for handler in self._message_handlers:
            try:
                await handler(message)
            except Exception as e:
                self._logger.error(f"Message handler error: {e}")
    
    def get_capabilities(self) -> Set[ChannelCapability]:
        return self._capabilities
    
    def supports_capability(self, capability: ChannelCapability) -> bool:
        return capability in self._capabilities
    
    def __repr__(self) -> str:
        return f"EmailAdapter(smtp={self._cfg.smtp_host}, imap={self._cfg.imap_host}, connected={self._connection_state == ConnectionState.CONNECTED})"


class EmailParser:
    """Utility class for parsing email addresses and headers."""
    
    EMAIL_RE = re.compile(r"<([^>]+)>")
    
    @staticmethod
    def extract_email(header: str) -> str:
        if "@" in header:
            match = EmailParser.EMAIL_RE.search(header)
            if match:
                return match.group(1)
            return header.strip().strip("<>").split()[0] if header.strip() else ""
        return ""
    
    @staticmethod
    def extract_name(header: str) -> Optional[str]:
        if "<" in header and ">" in header:
            name = header.split("<")[0].strip().strip('"').strip()
            return name if name else None
        return None
    
    @staticmethod
    def parse_date(date_str: str) -> float:
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(date_str)
            return dt.timestamp()
        except Exception:
            return time.time()


import hashlib
import os
