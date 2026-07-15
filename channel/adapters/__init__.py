"""Channel Adapters Module for AGI Unified Framework

This module provides platform-specific channel adapters that implement
the unified ChannelAdapter interface for various instant messaging platforms.

Supported Platforms:
- Telegram: Bot API with webhook and polling support
- Discord: Discord API with gateway connection and slash commands
- Slack: Slack Web API with Block Kit and interactive components
- Feishu (飞书): Feishu Open Platform with card messages
- DingTalk (钉钉): DingTalk API with enterprise features
- Email: SMTP/IMAP email support
- WeChat Work (企业微信): WeCom official API with full enterprise features
- Tuya IoT (涂鸦智能): IoT device management
- Tencent Docs (腾讯文档): Document collaboration
- Aliyun OSS (阿里云OSS): Object storage
- Tencent COS (腾讯云COS): Object storage
- Tencent Vector DB (腾讯云向量数据库): Vector database for RAG
- DingTalk Yida (钉钉宜搭): Low-code platform
- Tencent WeDa (腾讯云微搭): Low-code platform
- Baidu Pan (百度网盘): Cloud storage
- Tencent Exmail (腾讯企业邮): Enterprise email
- Mingdao (明道云): No-code platform
- Qingflow (轻流): Business process automation
- TiDB Cloud: Distributed SQL database

Usage Example:
    from agi_unified_framework.channel.adapters import (
        TelegramAdapter,
        DiscordAdapter,
        SlackAdapter,
        FeishuAdapter,
        DingTalkAdapter,
        EmailAdapter,
    )

    # Initialize adapters
    telegram = TelegramAdapter.from_bot_token("BOT_TOKEN")
    discord = DiscordAdapter.from_bot_token("DISCORD_BOT_TOKEN")
    slack = SlackAdapter.from_oauth_token("SLACK_OAUTH_TOKEN")

    # Use with ChannelGateway
    from agi_unified_framework.channel import ChannelGateway
    gateway = ChannelGateway()
    gateway.register_channel("telegram", telegram)
    gateway.register_channel("discord", discord)
"""

from __future__ import annotations

# Core adapter base classes
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

# Universal message format
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

# Channel adapters
from .telegram import (
    TelegramAdapter,
    TelegramConfig,
    TelegramBot,
    TelegramError,
)

from .discord import (
    DiscordAdapter,
    DiscordConfig,
    DiscordClient,
    DiscordError,
)

from .slack import (
    SlackAdapter,
    SlackConfig,
    SlackBoltApp,
    SlackError,
)

from .feishu import (
    FeishuAdapter,
    FeishuConfig,
    FeishuClient,
    FeishuError,
)

from .dingtalk import (
    DingTalkAdapter,
    DingTalkConfig,
    DingTalkCard,
    DingTalkMarkdown,
    DingTalkError,
)

from .email import (
    EmailAdapter,
    EmailConfig,
    EmailMessageBuilder,
    EmailThreadManager,
    EmailParser,
)

from .wechat_work import (
    WeComAdapter,
    WeComConfig,
    WeComCard,
    WeComCrypto,
    WeComError,
)

from .tuya_iot import (
    TuyaIoTAdapter,
    TuyaConfig,
    TuyaError,
)

from .tencent_docs import (
    TencentDocsAdapter,
    TencentDocsConfig,
    TencentDocsError,
)

# 新增Channel适配器
from .aliyun_oss import (
    AliyunOSSAdapter,
    AliyunOSSConfig,
    AliyunOSSError,
)

from .tencent_cos import (
    TencentCOSAdapter,
    TencentCOSConfig,
    TencentCOSError,
)

from .tencent_vector_db import (
    TencentVectorDBAdapter,
    TencentVectorDBConfig,
    TencentVectorDBError,
)

from .dingtalk_yida import (
    DingTalkYidaAdapter,
    DingTalkYidaConfig,
    DingTalkYidaError,
)

from .tencent_weda import (
    TencentWeDaAdapter,
    TencentWeDaConfig,
    TencentWeDaError,
)

from .baidu_pan import (
    BaiduPanAdapter,
    BaiduPanConfig,
    BaiduPanError,
)

from .tencent_exmail import (
    TencentExmailAdapter,
    TencentExmailConfig,
    TencentExmailError,
)

from .mingdao import (
    MingdaoAdapter,
    MingdaoConfig,
    MingdaoError,
)

from .qingflow import (
    QingflowAdapter,
    QingflowConfig,
    QingflowError,
)

from .tidb_cloud import (
    TiDBCloudAdapter,
    TiDBCloudConfig,
    TiDBCloudError,
)

# Re-export adapter factory functions
_ADAPTER_FACTORIES = {
    "telegram": TelegramAdapter.from_bot_token,
    "discord": DiscordAdapter.from_bot_token,
    "slack": SlackAdapter.from_oauth_token,
    "feishu": FeishuAdapter.from_app_config,
    "dingtalk": DingTalkAdapter.from_webhook_mode,
    "email": EmailAdapter,
    "wechat_work": WeComAdapter.from_app_config,
    "tuya_iot": TuyaIoTAdapter.from_app_config,
    "tencent_docs": TencentDocsAdapter.from_app_config,
    # 新增Channel适配器
    "aliyun_oss": AliyunOSSAdapter.from_app_config,
    "tencent_cos": TencentCOSAdapter.from_app_config,
    "tencent_vector_db": TencentVectorDBAdapter.from_app_config,
    "dingtalk_yida": DingTalkYidaAdapter.from_app_config,
    "tencent_weda": TencentWeDaAdapter.from_app_config,
    "baidu_pan": BaiduPanAdapter.from_app_config,
    "tencent_exmail": TencentExmailAdapter.from_app_config,
    "mingdao": MingdaoAdapter.from_app_config,
    "qingflow": QingflowAdapter.from_app_config,
    "tidb_cloud": TiDBCloudAdapter.from_app_config,
}


def get_adapter_factory(channel_type: str):
    """Get the adapter factory function for a given channel type.
    
    Args:
        channel_type: The channel type (telegram, discord, slack, etc.)
        
    Returns:
        The adapter factory function or None if not found.
    """
    return _ADAPTER_FACTORIES.get(channel_type.lower())


def list_supported_channels() -> list:
    """Get a list of supported channel types.
    
    Returns:
        List of supported channel type names.
    """
    return list(_ADAPTER_FACTORIES.keys())


def is_channel_supported(channel_type: str) -> bool:
    """Check if a channel type is supported.
    
    Args:
        channel_type: The channel type to check.
        
    Returns:
        True if the channel type is supported, False otherwise.
    """
    return channel_type.lower() in _ADAPTER_FACTORIES


__all__ = [
    # Base classes
    "ChannelAdapter",
    "ChannelCapability",
    "ChannelConfig",
    "ConnectionState",
    "MessagePriority",
    "ReceiveResult",
    "RetryConfig",
    "SendResult",
    # Message types
    "Attachment",
    "AttachmentType",
    "ChannelIdentity",
    "MessageContent",
    "MessageDirection",
    "MessageMetadata",
    "MessageStatus",
    "MessageType",
    "UniversalMessage",
    "UserIdentity",
    # Telegram
    "TelegramAdapter",
    "TelegramConfig",
    "TelegramBot",
    "TelegramError",
    # Discord
    "DiscordAdapter",
    "DiscordConfig",
    "DiscordClient",
    "DiscordError",
    # Slack
    "SlackAdapter",
    "SlackConfig",
    "SlackBoltApp",
    "SlackError",
    # Feishu
    "FeishuAdapter",
    "FeishuConfig",
    "FeishuClient",
    "FeishuError",
    # DingTalk
    "DingTalkAdapter",
    "DingTalkConfig",
    "DingTalkCard",
    "DingTalkMarkdown",
    "DingTalkError",
    # Email
    "EmailAdapter",
    "EmailConfig",
    "EmailMessageBuilder",
    "EmailThreadManager",
    "EmailParser",
    # WeChat Work
    "WeComAdapter",
    "WeComConfig",
    "WeComCard",
    "WeComCrypto",
    "WeComError",
    # Tuya IoT
    "TuyaIoTAdapter",
    "TuyaConfig",
    "TuyaError",
    # Tencent Docs
    "TencentDocsAdapter",
    "TencentDocsConfig",
    "TencentDocsError",
    # 新增Channel适配器
    "AliyunOSSAdapter",
    "AliyunOSSConfig",
    "AliyunOSSError",
    "TencentCOSAdapter",
    "TencentCOSConfig",
    "TencentCOSError",
    "TencentVectorDBAdapter",
    "TencentVectorDBConfig",
    "TencentVectorDBError",
    "DingTalkYidaAdapter",
    "DingTalkYidaConfig",
    "DingTalkYidaError",
    "TencentWeDaAdapter",
    "TencentWeDaConfig",
    "TencentWeDaError",
    "BaiduPanAdapter",
    "BaiduPanConfig",
    "BaiduPanError",
    "TencentExmailAdapter",
    "TencentExmailConfig",
    "TencentExmailError",
    "MingdaoAdapter",
    "MingdaoConfig",
    "MingdaoError",
    "QingflowAdapter",
    "QingflowConfig",
    "QingflowError",
    "TiDBCloudAdapter",
    "TiDBCloudConfig",
    "TiDBCloudError",
    # Utilities
    "get_adapter_factory",
    "list_supported_channels",
    "is_channel_supported",
]
