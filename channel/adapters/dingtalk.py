"""DingTalk Adapter Module"""
from __future__ import annotations
import asyncio
import hashlib
import hmac
import logging
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from ...base import ChannelAdapter, ChannelCapability, ChannelConfig, MessagePriority, ReceiveResult, SendResult
from ...universal_message import UniversalMessage

logger = logging.getLogger(__name__)


@dataclass
class DingTalkConfig(ChannelConfig):
    client_id: str = ""
    client_secret: str = ""
    agent_id: str = ""
    is_group_robot: bool = False
    secret: str = ""
    access_token: Optional[str] = None
    token_expires_at: float = 0.0


class DingTalkCard:
    @staticmethod
    def simple_card(title: str, content: str, button_text: Optional[str] = None, button_url: Optional[str] = None) -> Dict[str, Any]:
        card = {"config": {"auto_layout": True, "card_style": True}, "header": {"title": {"tag": "plain_text", "content": title}, "template": "blue"}, "body": {"content": content}}
        if button_text and button_url:
            card["action"] = {"configs": [{"action_text": button_text, "action_type": 1, "click_info": {"type": "open_url", "url": button_url}}]}
        return card
    
    @staticmethod
    def rich_card(title: str, form_items: Optional[List[Dict[str, str]]] = None, button_text: Optional[str] = None, button_url: Optional[str] = None) -> Dict[str, Any]:
        body = []
        if form_items:
            for item in form_items:
                label = item.get("label", "")
                value = item.get("value", "")
                body.append({"tag": "column", "span": 2, "fields": [{"tag": "markdown", "content": "**" + label + "**"}, {"tag": "markdown", "content": value}]})
        card = {"config": {"auto_layout": True}, "header": {"title": {"tag": "plain_text", "content": title}, "template": "blue"}, "body": {"fields": body if body else [{"tag": "markdown", "content": ""}]}}
        if button_text and button_url:
            card["action"] = {"configs": [{"action_text": button_text, "action_type": 1, "click_info": {"type": "open_url", "url": button_url}}]}
        return card
    
    @staticmethod
    def feed_card(links: List[Dict[str, str]]) -> Dict[str, Any]:
        return {"config": {"auto_layout": True}, "feeds": [{"icon": l.get("icon", ""), "title": l.get("title", ""), "url": l.get("url", "")} for l in links]}


class DingTalkMarkdown:
    @staticmethod
    def section(title: str, content: str) -> str:
        return "**" + title + "**\n\n" + content + "\n\n"
    
    @staticmethod
    def ordered_list(items: List[str]) -> str:
        return "\n".join(str(i+1) + ". " + item for i, item in enumerate(items)) + "\n\n"
    
    @staticmethod
    def unordered_list(items: List[str]) -> str:
        return "\n".join("- " + item for item in items) + "\n\n"
    
    @staticmethod
    def code_block(content: str, language: str = "") -> str:
        return "```" + language + "\n" + content + "\n```\n\n"
    
    @staticmethod
    def link(text: str, url: str) -> str:
        return "[" + text + "](" + url + ")"


class DingTalkAdapter(ChannelAdapter):
    BASE_API = "https://oapi.dingtalk.com"
    
    def __init__(self, config: DingTalkConfig) -> None:
        super().__init__(config)
        self._cfg = config
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0
    
    def _initialize_capabilities(self) -> None:
        self._capabilities = {ChannelCapability.TEXT_MESSAGES, ChannelCapability.MARKDOWN_MESSAGES, ChannelCapability.MEDIA_MESSAGES, ChannelCapability.FILE_ATTACHMENTS, ChannelCapability.DIRECT_MESSAGES, ChannelCapability.GROUPS, ChannelCapability.WEBHOOK_MODE, ChannelCapability.BUTTONS, ChannelCapability.INTERACTIVE_MESSAGES, ChannelCapability.CHANNEL_INFO, ChannelCapability.USER_INFO, ChannelCapability.RATE_LIMITING}
    
    async def _get_access_token(self, force: bool = False) -> str:
        now = time.time()
        if not force and self._access_token and now < self._token_expires_at - 60:
            return self._access_token
        result = await self._make_request("GET", "/gettoken", {"appkey": self._cfg.client_id, "appsecret": self._cfg.client_secret})
        self._access_token = result.get("access_token")
        self._token_expires_at = now + result.get("expires_in", 7200)
        return self._access_token
    
    async def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, params: Optional[Dict] = None) -> Dict:
        import aiohttp
        url = endpoint if endpoint.startswith("https://") else self.BASE_API + "/" + endpoint.lstrip("/")
        headers = {"Content-Type": "application/json"}
        timeout = aiohttp.ClientTimeout(total=self._config.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            if params:
                query = urllib.parse.urlencode(params)
                url = url + "?" + query if "?" not in url else url + "&" + query
            async with session.request(method, url, json=data, headers=headers) as response:
                result = await response.json()
        if result.get("errcode", 0) != 0:
            raise DingTalkError(result.get("errcode", 1), result.get("errmsg", "Unknown error"))
        return result
    
    async def _connect_impl(self) -> bool:
        try:
            if self._cfg.is_group_robot:
                self._logger.info("Connected to DingTalk (Group Robot mode)")
                return True
            token = await self._get_access_token()
            self._logger.info("Connected to DingTalk, token: " + token[:20] + "...")
            return True
        except Exception as e:
            self._logger.error("Failed to connect to DingTalk: " + str(e))
            return False
    
    async def _disconnect_impl(self) -> None:
        self._access_token = None
        self._token_expires_at = 0.0
    
    async def _health_check_impl(self) -> bool:
        try:
            if self._cfg.is_group_robot:
                return True
            await self._get_access_token()
            return True
        except Exception:
            return False
    
    def _generate_signature(self, timestamp: str) -> str:
        secret = self._cfg.secret
        secret_enc = secret.encode("utf-8")
        string_to_sign = timestamp + "\n" + secret
        hmac_code = hmac.new(secret_enc, string_to_sign.encode("utf-8"), digestmod=hashlib.sha256).digest()
        return urllib.parse.quote_plus(hmac_code.hexdigest())
    
    async def _send_webhook_message(self, webhook_url: str, msg_type: str, content: Dict) -> Dict:
        import aiohttp
        payload = {"msgtype": msg_type, msg_type: content}
        if self._cfg.secret:
            timestamp = str(int(time.time() * 1000))
            sign = self._generate_signature(timestamp)
            separator = "&" if "?" in webhook_url else "?"
            webhook_url = webhook_url + separator + "timestamp=" + timestamp + "&sign=" + sign
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                result = await response.json()
        if result.get("errcode", 0) != 0:
            raise DingTalkError(result.get("errcode", 1), result.get("errmsg", "Unknown error"))
        return result
    
    async def _send_impl(self, message: UniversalMessage, priority: MessagePriority) -> SendResult:
        try:
            if self._cfg.is_group_robot:
                webhook_url = message.get_context("webhook_url") or message.get_context("chat_id")
                if not webhook_url:
                    return SendResult(success=False, error="No webhook URL", error_code="MISSING_TARGET")
                msg_type, content = self._build_message(message)
                result = await self._send_webhook_message(webhook_url, msg_type, content)
                return SendResult(success=True, message_id=str(result.get("msg_id", "")), timestamp=time.time())
            else:
                await self._get_access_token()
                user_id = message.get_context("user_id") or (message.metadata.sender.user_id if message.metadata and message.metadata.sender else None)
                if not user_id:
                    return SendResult(success=False, error="No user_id", error_code="MISSING_TARGET")
                msg_type, content = self._build_message(message)
                result = await self.send_message(user_id=user_id, msg_type=msg_type, content=content)
                return SendResult(success=True, message_id=str(result.get("msg_id", "")), timestamp=time.time())
        except DingTalkError as e:
            return SendResult(success=False, error=e.message, error_code=str(e.code))
        except Exception as e:
            return SendResult(success=False, error=str(e), error_code=type(e).__name__)
    
    def _build_message(self, message: UniversalMessage) -> Tuple[str, Dict]:
        text = message.content.get_primary_text()
        if message.content.markdown:
            return ("markdown", {"text": message.content.markdown})
        return ("text", {"content": text})
    
    async def _receive_impl(self, payload: Optional[Dict]) -> ReceiveResult:
        if payload:
            message = self._transform_incoming_message(payload)
            return ReceiveResult(success=True, messages=[message], raw_payload=payload)
        return ReceiveResult(success=False, error="No payload provided")

    def _transform_incoming_message(self, event: Dict) -> UniversalMessage:
        from ...universal_message import UniversalMessage, MessageContent, MessageMetadata, MessageType, MessageDirection, UserIdentity, ChannelIdentity
        msg_type = event.get("msgtype", "text")
        content_data = event.get(msg_type, {})
        text = content_data.get("content", "")
        sender_info = event.get("sender", {})
        sender_nick = sender_info.get("nick", sender_info.get("staffId", "Unknown"))
        sender_id = sender_info.get("staffId", "")
        conversation_info = event.get("conversationInfo", event.get("conversation", {}))
        chat_id = conversation_info.get("chatId", event.get("chatId", ""))
        msg_id = event.get("msgId", str(time.time()))
        content = MessageContent(text=text)
        if msg_type == "markdown":
            content = MessageContent(markdown=text)
        elif msg_type == "image":
            content = MessageContent(text="[Image]", media_url=content_data.get("picUrl", ""))
        elif msg_type == "file":
            content = MessageContent(text="[File: " + content_data.get("fileName", "unknown") + "]")
        elif msg_type == "voice":
            content = MessageContent(text="[Voice message]")
        elif msg_type == "video":
            content = MessageContent(text="[Video message]")
        metadata = MessageMetadata(
            message_id=msg_id,
            channel_id="dingtalk",
            channel_specific_id=msg_id,
            timestamp=event.get("createAt", time.time()),
            direction=MessageDirection.INBOUND,
            message_type=MessageType.from_string(msg_type),
            raw_event=event,
            sender=UserIdentity(user_id=sender_id, username=sender_nick, display_name=sender_nick),
            channel=ChannelIdentity(channel_id=chat_id, channel_type="dingtalk", name=conversation_info.get("title", "DingTalk Chat"))
        )
        message = UniversalMessage(content=content, metadata=metadata)
        if chat_id:
            message.set_context("chat_id", chat_id)
        if sender_id:
            message.set_context("user_id", sender_id)
        if event.get("robotCode"):
            message.set_context("robot_code", event["robotCode"])
        if content_data.get("media_id"):
            message.set_context("media_id", content_data["media_id"])
        return message

    async def get_user_info(self, user_id: str) -> Dict[str, Any]:
        try:
            await self._get_access_token()
            result = await self._make_request("GET", "/user/get", {"access_token": self._access_token, "userid": user_id})
            return {"user_id": result.get("userid", user_id), "name": result.get("name", ""), "avatar": result.get("avatar", ""), "email": result.get("email", ""), "mobile": result.get("mobile", "")}
        except Exception as e:
            self._logger.warning("Failed to get user info for " + user_id + ": " + str(e))
            return {"user_id": user_id, "name": "Unknown", "error": str(e)}

    async def get_channel_info(self, chat_id: str) -> Dict[str, Any]:
        try:
            await self._get_access_token()
            result = await self._make_request("GET", "/chat/get", {"access_token": self._access_token, "chatid": chat_id})
            return {"chat_id": result.get("chatid", chat_id), "name": result.get("name", ""), "owner": result.get("owner", ""), "member_count": result.get("membersCount", 0)}
        except Exception as e:
            self._logger.warning("Failed to get channel info for " + chat_id + ": " + str(e))
            return {"chat_id": chat_id, "name": "Unknown", "error": str(e)}

    async def send_message(self, user_id: str, msg_type: str, content: Dict, agent_id: Optional[str] = None) -> Dict[str, Any]:
        await self._get_access_token()
        payload = {"touser": user_id, "agentid": agent_id or self._cfg.agent_id, "msgtype": msg_type, msg_type: content}
        return await self._make_request("POST", "/message/send", payload, {"access_token": self._access_token})

    async def send_group_message(self, chat_id: str, msg_type: str, content: Dict) -> Dict[str, Any]:
        await self._get_access_token()
        payload = {"chatid": chat_id, "msgtype": msg_type, msg_type: content}
        return await self._make_request("POST", "/chat/send", payload, {"access_token": self._access_token})

    async def create_card_message(self, template: Dict, recipient: Dict) -> Dict[str, Any]:
        await self._get_access_token()
        payload = {"agent_id": self._cfg.agent_id, "user_ids": [recipient.get("user_id", "")], "template_id": template.get("template_id", ""), "template_data": recipient.get("data", {})}
        return await self._make_request("POST", "/card/create", payload, {"access_token": self._access_token})

    async def update_card_message(self, card_id: str, template: Dict, recipient: Dict) -> Dict[str, Any]:
        await self._get_access_token()
        payload = {"card_id": card_id, "template_id": template.get("template_id", ""), "template_data": recipient.get("data", {})}
        return await self._make_request("POST", "/card/update", payload, {"access_token": self._access_token})

    async def upload_media(self, file_path: str, media_type: str = "file") -> Optional[str]:
        import aiohttp
        await self._get_access_token()
        url = self.BASE_API + "/media/upload?access_token=" + self._access_token + "&type=" + media_type
        form = aiohttp.FormData()
        form.add_field("media", open(file_path, "rb"), filename=os.path.basename(file_path), content_type="application/octet-stream")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=form) as response:
                result = await response.json()
        if result.get("errcode", 0) == 0:
            return result.get("media_id")
        self._logger.error("Failed to upload media: " + str(result))
        return None

    async def get_conversation_list(self, offset: int = 0, limit: int = 20) -> List[Dict[str, Any]]:
        await self._get_access_token()
        payload = {"offset": offset, "limit": limit}
        result = await self._make_request("POST", "/chat/listid", payload, {"access_token": self._access_token})
        return result.get("conversations", [])

    async def get_group_member_list(self, chat_id: str, offset: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        await self._get_access_token()
        payload = {"chatId": chat_id, "offset": offset, "limit": limit}
        result = await self._make_request("POST", "/chat/getMemberIds", payload, {"access_token": self._access_token})
        return [{"user_id": uid} for uid in result.get("memberIds", [])]

    async def create_group(self, name: str, owner_user_id: str, member_ids: List[str]) -> Optional[str]:
        await self._get_access_token()
        payload = {"name": name, "owner": owner_user_id, "useridlist": member_ids}
        result = await self._make_request("POST", "/chat/create", payload, {"access_token": self._access_token})
        return result.get("chatid")

    async def update_group(self, chat_id: str, name: Optional[str] = None, owner_user_id: Optional[str] = None) -> bool:
        await self._get_access_token()
        payload = {"chatid": chat_id}
        if name:
            payload["name"] = name
        if owner_user_id:
            payload["owner"] = owner_user_id
        result = await self._make_request("POST", "/chat/update", payload, {"access_token": self._access_token})
        return result.get("errcode", 0) == 0

    async def _handle_callback_event(self, event_type: str, event_data: Dict) -> Optional[UniversalMessage]:
        if event_type in ("im_message_received", "chat robot message received"):
            return self._transform_incoming_message(event_data)
        elif event_type == "im_message_revoked":
            return self._create_recalled_message(event_data)
        elif event_type in ("user_add", "user_leave"):
            return self._create_member_change_message(event_type, event_data)
        elif event_type == "group_update":
            return self._create_group_update_message(event_data)
        return None

    def _create_recalled_message(self, event_data: Dict) -> UniversalMessage:
        from ...universal_message import UniversalMessage, MessageContent, MessageMetadata, MessageType, MessageDirection, UserIdentity, ChannelIdentity
        content = MessageContent(text="[Message recalled]")
        metadata = MessageMetadata(
            message_id=event_data.get("msgId", str(time.time())),
            channel_id="dingtalk",
            timestamp=event_data.get("createAt", time.time()),
            direction=MessageDirection.INBOUND,
            message_type=MessageType.SYSTEM,
            raw_event=event_data
        )
        return UniversalMessage(content=content, metadata=metadata)

    def _create_member_change_message(self, event_type: str, event_data: Dict) -> UniversalMessage:
        from ...universal_message import UniversalMessage, MessageContent, MessageMetadata, MessageType, MessageDirection, UserIdentity
        action = "joined" if "add" in event_type else "left"
        content = MessageContent(text="[" + event_data.get("operator", "Someone") + " " + action + " the group]")
        metadata = MessageMetadata(
            message_id="sys_" + str(time.time()),
            channel_id="dingtalk",
            timestamp=event_data.get("createAt", time.time()),
            direction=MessageDirection.INBOUND,
            message_type=MessageType.SYSTEM,
            raw_event=event_data
        )
        return UniversalMessage(content=content, metadata=metadata)

    def _create_group_update_message(self, event_data: Dict) -> UniversalMessage:
        from ...universal_message import UniversalMessage, MessageContent, MessageMetadata, MessageType, MessageDirection
        content = MessageContent(text="[Group settings updated]")
        metadata = MessageMetadata(
            message_id="sys_" + str(time.time()),
            channel_id="dingtalk",
            timestamp=event_data.get("createAt", time.time()),
            direction=MessageDirection.INBOUND,
            message_type=MessageType.SYSTEM,
            raw_event=event_data
        )
        return UniversalMessage(content=content, metadata=metadata)

    @classmethod
    def from_webhook_mode(cls, webhook_url: str, secret: Optional[str] = None) -> DingTalkAdapter:
        config = DingTalkConfig(name="dingtalk", is_group_robot=True, secret=secret or "", webhook_url=webhook_url)
        adapter = cls(config)
        return adapter

    @classmethod
    def from_app_mode(cls, client_id: str, client_secret: str, agent_id: str) -> DingTalkAdapter:
        config = DingTalkConfig(name="dingtalk", client_id=client_id, client_secret=client_secret, agent_id=agent_id)
        return cls(config)

    def get_capabilities(self) -> Set[ChannelCapability]:
        return self._capabilities

    def supports_capability(self, capability: ChannelCapability) -> bool:
        return capability in self._capabilities

    def get_rate_limit_info(self) -> Dict[str, Any]:
        return {"type": "DingTalk API", "default_limit": 1000, "window_seconds": 3600, "message": "Consult DingTalk API documentation for current limits"}

    async def validate_webhook_signature(self, headers: Dict, body: bytes) -> bool:
        if not self._cfg.secret:
            return True
        timestamp = headers.get("timestamp", "")
        sign_str = headers.get("sign", "")
        if not timestamp or not sign_str:
            return False
        computed_sign = self._generate_signature(timestamp)
        return computed_sign == sign_str

    async def process_callback(self, event_type: str, event_data: Dict) -> Optional[UniversalMessage]:
        return await self._handle_callback_event(event_type, event_data)

    def get_config(self) -> DingTalkConfig:
        return self._cfg

    def set_debug_mode(self, enabled: bool) -> None:
        if enabled:
            logging.getLogger("dingtalk").setLevel(logging.DEBUG)
            self._logger.setLevel(logging.DEBUG)
        else:
            logging.getLogger("dingtalk").setLevel(logging.INFO)
            self._logger.setLevel(logging.INFO)

    async def test_connection(self) -> Dict[str, Any]:
        try:
            if self._cfg.is_group_robot:
                return {"status": "ok", "mode": "webhook", "message": "Webhook robot mode configured"}
            token = await self._get_access_token()
            return {"status": "ok", "mode": "app", "token_prefix": token[:10] + "...", "expires_in": int(self._token_expires_at - time.time())}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _validate_message_content(self, message: UniversalMessage) -> Tuple[bool, Optional[str]]:
        if not message.content.get_primary_text() and not message.content.markdown and not message.content.attachments:
            return False, "Message content is empty"
        return True, None

    def __repr__(self) -> str:
        mode = "webhook" if self._cfg.is_group_robot else "app"
        return f"DingTalkAdapter(name={self._config.name}, mode={mode}, connected={self._connection_state == ConnectionState.CONNECTED})"


class DingTalkError(Exception):
    def __init__(self, code: int, message: str, details: Optional[Dict] = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(f"[{code}] {message}")

    @classmethod
    def from_response(cls, response: Dict) -> DingTalkError:
        return cls(code=response.get("errcode", 1), message=response.get("errmsg", "Unknown error"), details=response)

    @property
    def is_retryable(self) -> bool:
        return self.code in (40001, 40014, 42001)

    @property
    def is_auth_error(self) -> bool:
        return self.code in (40001, 40014, 40078, 40102)


import os
