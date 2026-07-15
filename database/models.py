"""
AGI Unified Framework - 数据库 ORM 模型定义

本模块定义了系统中所有 SQLAlchemy ORM 模型，涵盖以下子系统：
- 用户系统 (User, UserSettings)
- 对话系统 (Conversation, Message)
- 模型管理 (Model, ModelLoadBalance)
- 训练系统 (TrainingJob, Checkpoint)
- 生成系统 (GeneratedContent)
- 工作流系统 (Workflow, WorkflowExecution)
- 智能体系统 (Agent, Alliance)
- 插件系统 (Plugin)
- 渠道系统 (Channel)
- 数据集系统 (Dataset)
- 人格系统 (Personality)
- 安全审计 (AuditLog)
- 系统设置 (SystemSetting)

设计原则:
    1. 所有模型使用 UTC 时区的时间戳
    2. 使用 JSON 字段存储灵活的配置数据
    3. 完整的关系定义和级联操作
    4. 详细的索引优化查询性能
    5. 类型注解和文档字符串

依赖:
    - sqlalchemy >= 1.4 (可选，优雅降级)
"""

import os
import sys
import enum
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# 尝试导入 SQLAlchemy
try:
    from sqlalchemy import (
        create_engine,
        Column,
        Integer,
        String,
        Text,
        Boolean,
        Float,
        DateTime,
        Enum,
        ForeignKey,
        Index,
        UniqueConstraint,
        CheckConstraint,
        JSON,
        LargeBinary,
        Numeric,
        BigInteger,
        SmallInteger,
        func,
        select,
        or_,
        and_,
        event,
        DDL,
    )
    from sqlalchemy.orm import (
        relationship,
        declarative_base,
        Session,
        joinedload,
        lazyload,
        validates,
    )
    from sqlalchemy.sql import expression

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logger.warning(
        "SQLAlchemy 未安装。ORM 模型将不可用。"
        "请运行: pip install sqlalchemy"
    )


# ============================================================
# 枚举类型定义
# ============================================================

class UserRole(enum.Enum):
    """用户角色枚举"""
    SUPER_ADMIN = "super_admin"      # 超级管理员
    ADMIN = "admin"                  # 管理员
    USER = "user"                    # 普通用户
    GUEST = "guest"                  # 访客
    DEVELOPER = "developer"          # 开发者
    API_USER = "api_user"            # API 用户


class ModelProvider(enum.Enum):
    """模型提供商枚举"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    CUSTOM = "custom"
    ZHIPUAI = "zhipuai"
    BAIDU = "baidu"
    ALIYUN = "aliyun"
    ALIBABA = "alibaba"
    TENCENT = "tencent"
    MOONSHOT = "moonshot"
    DEEPSEEK = "deepseek"
    MINIMAX = "minimax"
    SPARK = "spark"
    YI = "yi"
    WENXIN = "wenxin"
    DASHSCOPE = "dashscope"
    AZURE_OPENAI = "azure_openai"
    COHERE = "cohere"
    BAICHUAN = "baichuan"
    ZHIPU = "zhipu"
    QWEN = "qwen"
    OLLAMA = "ollama"
    VLLM = "vllm"


class ModelType(enum.Enum):
    """模型类型枚举"""
    LLM = "llm"                      # 大语言模型
    CHAT = "chat"                    # 对话模型
    COMPLETION = "completion"        # 补全模型
    EMBEDDING = "embedding"          # 嵌入模型
    IMAGE = "image"                  # 图像生成模型
    AUDIO = "audio"                  # 音频模型
    VIDEO = "video"                  # 视频模型
    CODE = "code"                    # 代码模型
    RERANK = "rerank"                # 重排序模型
    TTS = "tts"                      # 语音合成模型
    STT = "stt"                      # 语音识别模型
    MODERATION = "moderation"        # 内容审核模型
    MULTIMODAL = "multimodal"        # 多模态模型
    CUSTOM = "custom"                # 自定义模型


class ModelStatus(enum.Enum):
    """模型状态枚举"""
    ACTIVE = "active"                # 活跃可用
    INACTIVE = "inactive"            # 未激活
    TESTING = "testing"              # 测试中
    ERROR = "error"                  # 错误状态
    MAINTENANCE = "maintenance"      # 维护中
    DEPRECATED = "deprecated"        # 已弃用


class TrainingStatus(enum.Enum):
    """训练任务状态枚举"""
    PENDING = "pending"              # 等待中
    QUEUED = "queued"                # 已排队
    RUNNING = "running"              # 运行中
    PAUSED = "paused"                # 已暂停
    COMPLETED = "completed"          # 已完成
    FAILED = "failed"                # 已失败
    CANCELLED = "cancelled"          # 已取消


class ContentType(enum.Enum):
    """生成内容类型枚举"""
    IMAGE = "image"                  # 图像
    VIDEO = "video"                  # 视频
    AUDIO = "audio"                  # 音频
    THREE_D = "3d"                   # 3D 模型
    TEXT = "text"                    # 文本
    CODE = "code"                    # 代码
    MUSIC = "music"                  # 音乐


class WorkflowStatus(enum.Enum):
    """工作流状态枚举"""
    DRAFT = "draft"                  # 草稿
    ACTIVE = "active"                # 活跃
    PUBLISHED = "published"          # 已发布
    PAUSED = "paused"                # 已暂停
    ARCHIVED = "archived"            # 已归档
    DEPRECATED = "deprecated"        # 已弃用
    ERROR = "error"                  # 错误


class ExecutionStatus(enum.Enum):
    """工作流执行状态枚举"""
    PENDING = "pending"              # 等待执行
    RUNNING = "running"              # 执行中
    COMPLETED = "completed"          # 已完成
    FAILED = "failed"                # 已失败
    CANCELLED = "cancelled"          # 已取消
    TIMEOUT = "timeout"              # 超时


class AgentStatus(enum.Enum):
    """智能体状态枚举"""
    ACTIVE = "active"                # 活跃
    INACTIVE = "inactive"            # 未激活
    BUSY = "busy"                    # 忙碌中
    ERROR = "error"                  # 错误
    MAINTENANCE = "maintenance"      # 维护中


class PluginStatus(enum.Enum):
    """插件状态枚举"""
    INSTALLED = "installed"          # 已安装
    ENABLED = "enabled"              # 已启用
    DISABLED = "disabled"            # 已禁用
    ERROR = "error"                  # 错误
    UPDATING = "updating"            # 更新中


class ChannelType(enum.Enum):
    """渠道类型枚举"""
    DISCORD = "discord"              # Discord
    TELEGRAM = "telegram"            # Telegram
    SLACK = "slack"                  # Slack
    EMAIL = "email"                  # 邮件
    FEISHU = "feishu"                # 飞书
    DINGTALK = "dingtalk"            # 钉钉
    WECHAT = "wechat"                # 微信
    WEBHOOK = "webhook"              # Webhook
    API = "api"                      # API
    CUSTOM = "custom"                # 自定义


class ChannelStatus(enum.Enum):
    """渠道状态枚举"""
    ACTIVE = "active"                # 活跃
    INACTIVE = "inactive"            # 未激活
    ERROR = "error"                  # 错误
    MAINTENANCE = "maintenance"      # 维护中


class DatasetType(enum.Enum):
    """数据集类型枚举"""
    TEXT = "text"                    # 文本数据集
    IMAGE = "image"                  # 图像数据集
    AUDIO = "audio"                  # 音频数据集
    VIDEO = "video"                  # 视频数据集
    MULTIMODAL = "multimodal"        # 多模态数据集
    CONVERSATION = "conversation"    # 对话数据集
    CODE = "code"                    # 代码数据集
    MIXED = "mixed"                  # 混合数据集
    CUSTOM = "custom"                # 自定义


class AuditAction(enum.Enum):
    """审计操作类型枚举"""
    LOGIN = "login"                  # 登录
    LOGOUT = "logout"                # 登出
    CREATE = "create"                # 创建
    READ = "read"                    # 读取
    UPDATE = "update"                # 更新
    DELETE = "delete"                # 删除
    EXECUTE = "execute"              # 执行
    EXPORT = "export"                # 导出
    IMPORT = "import"                # 导入
    CONFIGURE = "configure"          # 配置
    DEPLOY = "deploy"                # 部署
    PERMISSION_CHANGE = "permission_change"  # 权限变更


class ResourceType(enum.Enum):
    """资源类型枚举"""
    USER = "user"                    # 用户
    CONVERSATION = "conversation"    # 对话
    MESSAGE = "message"              # 消息
    MODEL = "model"                  # 模型
    TRAINING_JOB = "training_job"    # 训练任务
    WORKFLOW = "workflow"            # 工作流
    AGENT = "agent"                  # 智能体
    PLUGIN = "plugin"                # 插件
    CHANNEL = "channel"              # 渠道
    DATASET = "dataset"              # 数据集
    PERSONALITY = "personality"      # 人格
    SYSTEM = "system"                # 系统
    API_KEY = "api_key"              # API 密钥
    ALLIANCE = "alliance"            # 联盟


class LoadBalanceStrategy(enum.Enum):
    """负载均衡策略枚举"""
    ROUND_ROBIN = "round_robin"      # 轮询
    WEIGHTED = "weighted"            # 加权
    LEAST_CONNECTIONS = "least_connections"  # 最少连接
    RANDOM = "random"                # 随机
    PRIORITY = "priority"            # 优先级
    COST_BASED = "cost_based"        # 基于成本
    LATENCY_BASED = "latency_based"  # 基于延迟
    HEALTH_BASED = "health_based"    # 基于健康状态


# ============================================================
# 声明式基类
# ============================================================

if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()
    Base.metadata.naming_convention = {
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
else:
    # 优雅降级：创建一个占位基类
    class Base:  # type: ignore
        """SQLAlchemy 不可用时的占位基类"""
        metadata = None
        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)


# ============================================================
# 辅助函数
# ============================================================

def get_utc_now() -> datetime:
    """获取当前 UTC 时间"""
    return datetime.now(timezone.utc)


def default_utc_now() -> datetime:
    """SQLAlchemy 默认值函数，返回当前 UTC 时间"""
    return get_utc_now()


# ============================================================
# 用户系统模型
# ============================================================

class User(Base):  # type: ignore
    """
    用户模型

    存储系统用户的基本信息，包括认证信息和个人资料。

    属性:
        id: 用户唯一标识符
        username: 用户名（唯一）
        email: 电子邮件地址（唯一）
        password_hash: bcrypt 哈希后的密码
        role: 用户角色
        avatar: 头像 URL
        is_active: 是否激活
        is_verified: 是否已验证邮箱
        last_login: 最后登录时间
        login_count: 登录次数
        created_at: 创建时间
        updated_at: 更新时间

    关系:
        settings: 用户设置（一对一）
        conversations: 用户的对话列表
        training_jobs: 用户的训练任务
        generated_contents: 用户生成的内容
        workflows: 用户的工作流
        agents: 用户的智能体
        personalities: 用户的人格
        audit_logs: 用户的审计日志
    """
    __tablename__ = "users"

    # 基本字段
    id = Column(Integer, primary_key=True, autoincrement=True, comment="用户ID")
    username = Column(
        String(64), unique=True, nullable=False,
        comment="用户名"
    )
    email = Column(
        String(255), unique=True, nullable=False,
        comment="电子邮件"
    )
    password_hash = Column(
        String(255), nullable=False,
        comment="密码哈希（bcrypt）"
    )
    role = Column(
        Enum(UserRole), nullable=False, default=UserRole.USER,
        comment="用户角色"
    )

    # 个人资料
    avatar = Column(String(512), nullable=True, comment="头像URL")
    display_name = Column(String(128), nullable=True, comment="显示名称")
    bio = Column(Text, nullable=True, comment="个人简介")
    phone = Column(String(32), nullable=True, comment="手机号码")
    location = Column(String(255), nullable=True, comment="位置")
    website = Column(String(512), nullable=True, comment="个人网站")

    # 状态字段
    is_active = Column(
        Boolean, nullable=False, default=True,
        comment="是否激活"
    )
    is_verified = Column(
        Boolean, nullable=False, default=False,
        comment="是否已验证邮箱"
    )
    is_locked = Column(
        Boolean, nullable=False, default=False,
        comment="是否锁定"
    )
    locked_until = Column(
        DateTime(timezone=True), nullable=True,
        comment="锁定截止时间"
    )
    failed_login_attempts = Column(
        Integer, nullable=False, default=0,
        comment="连续登录失败次数"
    )

    # 登录信息
    last_login = Column(
        DateTime(timezone=True), nullable=True,
        comment="最后登录时间"
    )
    last_login_ip = Column(String(45), nullable=True, comment="最后登录IP")
    login_count = Column(Integer, nullable=False, default=0, comment="登录次数")
    current_session_token = Column(String(512), nullable=True, comment="当前会话令牌")

    # 安全字段
    two_factor_enabled = Column(
        Boolean, nullable=False, default=False,
        comment="是否启用两步验证"
    )
    two_factor_secret = Column(String(255), nullable=True, comment="两步验证密钥")
    password_changed_at = Column(
        DateTime(timezone=True), nullable=True,
        comment="密码修改时间"
    )
    password_reset_token = Column(String(255), nullable=True, comment="密码重置令牌")
    password_reset_expires = Column(
        DateTime(timezone=True), nullable=True,
        comment="密码重置令牌过期时间"
    )
    email_verification_token = Column(String(255), nullable=True, comment="邮箱验证令牌")
    email_verification_expires = Column(
        DateTime(timezone=True), nullable=True,
        comment="邮箱验证令牌过期时间"
    )

    # 配额字段
    api_quota_daily = Column(Integer, nullable=False, default=1000, comment="每日API调用配额")
    api_quota_used = Column(Integer, nullable=False, default=0, comment="已使用API配额")
    storage_quota_mb = Column(Integer, nullable=False, default=1024, comment="存储配额(MB)")
    storage_used_mb = Column(Float, nullable=False, default=0.0, comment="已用存储(MB)")

    # 时间戳
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        comment="创建时间"
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        onupdate=get_utc_now,
        comment="更新时间"
    )

    # 关系定义
    settings = relationship(
        "UserSettings", back_populates="user", uselist=False,
        cascade="all, delete-orphan", lazy="select"
    )
    conversations = relationship(
        "Conversation", back_populates="user",
        cascade="all, delete-orphan", lazy="select"
    )
    training_jobs = relationship(
        "TrainingJob", back_populates="user",
        cascade="all, delete-orphan", lazy="select"
    )
    generated_contents = relationship(
        "GeneratedContent", back_populates="user",
        cascade="all, delete-orphan", lazy="select"
    )
    workflows = relationship(
        "Workflow", back_populates="user",
        cascade="all, delete-orphan", lazy="select"
    )
    agents = relationship(
        "Agent", back_populates="user",
        cascade="all, delete-orphan", lazy="select"
    )
    personalities = relationship(
        "Personality", back_populates="user",
        cascade="all, delete-orphan", lazy="select"
    )
    audit_logs = relationship(
        "AuditLog", back_populates="user",
        cascade="all, delete-orphan", lazy="select"
    )

    # 索引
    __table_args__ = (
        Index("ix_users_role", "role"),
        Index("ix_users_is_active", "is_active"),
        Index("ix_users_created_at", "created_at"),
        Index("ix_users_username_email", "username", "email"),
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}', role={self.role.value})>"

    def to_dict(self, exclude_sensitive: bool = True) -> Dict[str, Any]:
        """
        将用户对象转换为字典

        参数:
            exclude_sensitive: 是否排除敏感字段

        返回:
            用户信息字典
        """
        data = {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role.value if self.role else None,
            "avatar": self.avatar,
            "display_name": self.display_name,
            "bio": self.bio,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "login_count": self.login_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if not exclude_sensitive:
            data["password_hash"] = self.password_hash
            data["two_factor_secret"] = self.two_factor_secret
        return data


class UserSettings(Base):  # type: ignore
    """
    用户设置模型

    存储用户的个性化配置，包括界面主题、语言偏好、通知设置等。

    属性:
        id: 设置唯一标识符
        user_id: 关联的用户 ID（外键）
        theme: 界面主题（light/dark/auto）
        language: 界面语言
        sidebar_collapsed: 侧边栏是否折叠
        notifications_enabled: 是否启用通知
        notification_types: 通知类型配置（JSON）
        font_size: 字体大小
        code_font: 代码字体
        auto_save: 是否自动保存
        auto_save_interval: 自动保存间隔（秒）

    关系:
        user: 关联的用户（一对一）
    """
    __tablename__ = "user_settings"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="设置ID")
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"),
        unique=True, nullable=False,
        comment="用户ID"
    )

    # 界面设置
    theme = Column(String(32), nullable=False, default="auto", comment="界面主题")
    language = Column(String(16), nullable=False, default="zh-CN", comment="界面语言")
    sidebar_collapsed = Column(
        Boolean, nullable=False, default=False,
        comment="侧边栏是否折叠"
    )
    font_size = Column(Integer, nullable=False, default=14, comment="字体大小")
    code_font = Column(
        String(64), nullable=False, default="JetBrains Mono",
        comment="代码字体"
    )
    code_font_size = Column(Integer, nullable=False, default=13, comment="代码字体大小")
    editor_mode = Column(String(32), nullable=False, default="vscode", comment="编辑器模式")
    color_scheme = Column(String(64), nullable=True, comment="配色方案")

    # 通知设置
    notifications_enabled = Column(
        Boolean, nullable=False, default=True,
        comment="是否启用通知"
    )
    notification_types = Column(
        JSON, nullable=True,
        comment="通知类型配置"
    )
    notification_sound = Column(
        Boolean, nullable=False, default=True,
        comment="通知声音"
    )
    notification_desktop = Column(
        Boolean, nullable=False, default=True,
        comment="桌面通知"
    )
    notification_email = Column(
        Boolean, nullable=False, default=False,
        comment="邮件通知"
    )

    # 自动保存设置
    auto_save = Column(
        Boolean, nullable=False, default=True,
        comment="是否自动保存"
    )
    auto_save_interval = Column(
        Integer, nullable=False, default=30,
        comment="自动保存间隔（秒）"
    )

    # 对话设置
    default_model = Column(String(128), nullable=True, comment="默认模型")
    default_system_prompt = Column(Text, nullable=True, comment="默认系统提示词")
    max_tokens = Column(Integer, nullable=False, default=4096, comment="最大token数")
    temperature = Column(Float, nullable=False, default=0.7, comment="默认温度")
    top_p = Column(Float, nullable=False, default=0.9, comment="默认top_p")
    stream_response = Column(
        Boolean, nullable=False, default=True,
        comment="是否流式响应"
    )
    show_token_count = Column(
        Boolean, nullable=False, default=True,
        comment="是否显示token计数"
    )

    # 隐私设置
    profile_visible = Column(
        Boolean, nullable=False, default=True,
        comment="个人资料是否可见"
    )
    activity_visible = Column(
        Boolean, nullable=False, default=False,
        comment="活动状态是否可见"
    )
    data_collection = Column(
        Boolean, nullable=False, default=True,
        comment="是否允许数据收集"
    )

    # 高级设置
    config_json = Column(JSON, nullable=True, comment="额外配置（JSON）")

    # 时间戳
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        comment="创建时间"
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        onupdate=get_utc_now,
        comment="更新时间"
    )

    # 关系定义
    user = relationship("User", back_populates="settings", lazy="select")

    # 索引
    __table_args__ = (
        Index("ix_user_settings_theme", "theme"),
        Index("ix_user_settings_language", "language"),
    )

    def __repr__(self) -> str:
        return f"<UserSettings(id={self.id}, user_id={self.user_id})>"

    def to_dict(self) -> Dict[str, Any]:
        """将用户设置转换为字典"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "theme": self.theme,
            "language": self.language,
            "sidebar_collapsed": self.sidebar_collapsed,
            "font_size": self.font_size,
            "code_font": self.code_font,
            "code_font_size": self.code_font_size,
            "editor_mode": self.editor_mode,
            "color_scheme": self.color_scheme,
            "notifications_enabled": self.notifications_enabled,
            "notification_types": self.notification_types,
            "notification_sound": self.notification_sound,
            "notification_desktop": self.notification_desktop,
            "notification_email": self.notification_email,
            "auto_save": self.auto_save,
            "auto_save_interval": self.auto_save_interval,
            "default_model": self.default_model,
            "default_system_prompt": self.default_system_prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream_response": self.stream_response,
            "show_token_count": self.show_token_count,
            "profile_visible": self.profile_visible,
            "activity_visible": self.activity_visible,
            "data_collection": self.data_collection,
            "config_json": self.config_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ============================================================
# 对话系统模型
# ============================================================

class Conversation(Base):  # type: ignore
    """
    对话模型

    存储用户与 AI 模型之间的对话会话信息。

    属性:
        id: 对话唯一标识符
        user_id: 所属用户 ID
        title: 对话标题
        model_name: 使用的模型名称
        system_prompt: 系统提示词
        is_archived: 是否已归档
        is_pinned: 是否置顶
        tags: 标签（JSON 数组）
        summary: 对话摘要
        message_count: 消息数量
        total_tokens: 总 token 数

    关系:
        user: 所属用户
        messages: 对话中的消息列表
    """
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="对话ID")
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        comment="用户ID"
    )

    # 对话信息
    title = Column(String(255), nullable=False, default="新对话", comment="对话标题")
    model_name = Column(String(128), nullable=True, comment="使用的模型名称")
    system_prompt = Column(Text, nullable=True, comment="系统提示词")
    description = Column(Text, nullable=True, comment="对话描述")

    # 状态字段
    is_archived = Column(
        Boolean, nullable=False, default=False,
        comment="是否已归档"
    )
    is_pinned = Column(
        Boolean, nullable=False, default=False,
        comment="是否置顶"
    )
    is_bookmarked = Column(
        Boolean, nullable=False, default=False,
        comment="是否收藏"
    )

    # 标签和分类
    tags = Column(JSON, nullable=True, comment="标签（JSON数组）")
    category = Column(String(64), nullable=True, comment="分类")

    # 统计信息
    summary = Column(Text, nullable=True, comment="对话摘要")
    message_count = Column(Integer, nullable=False, default=0, comment="消息数量")
    total_tokens = Column(Integer, nullable=False, default=0, comment="总token数")
    total_cost = Column(Float, nullable=False, default=0.0, comment="总费用")

    # 配置
    config_json = Column(JSON, nullable=True, comment="对话配置（JSON）")

    # 时间戳
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        comment="创建时间"
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        onupdate=get_utc_now,
        comment="更新时间"
    )
    last_message_at = Column(
        DateTime(timezone=True), nullable=True,
        comment="最后消息时间"
    )

    # 关系定义
    user = relationship("User", back_populates="conversations", lazy="select")
    messages = relationship(
        "Message", back_populates="conversation",
        cascade="all, delete-orphan", lazy="select",
        order_by="Message.created_at"
    )

    # 索引
    __table_args__ = (
        Index("ix_conversations_user_id", "user_id"),
        Index("ix_conversations_model_name", "model_name"),
        Index("ix_conversations_created_at", "created_at"),
        Index("ix_conversations_updated_at", "updated_at"),
        Index("ix_conversations_is_archived", "is_archived"),
        Index("ix_conversations_is_pinned", "is_pinned"),
    )

    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, title='{self.title}', user_id={self.user_id})>"

    def to_dict(self, include_messages: bool = False) -> Dict[str, Any]:
        """将对话转换为字典"""
        data = {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "model_name": self.model_name,
            "system_prompt": self.system_prompt,
            "description": self.description,
            "is_archived": self.is_archived,
            "is_pinned": self.is_pinned,
            "is_bookmarked": self.is_bookmarked,
            "tags": self.tags,
            "category": self.category,
            "summary": self.summary,
            "message_count": self.message_count,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "config_json": self.config_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_message_at": self.last_message_at.isoformat() if self.last_message_at else None,
        }
        if include_messages:
            data["messages"] = [m.to_dict() for m in self.messages]
        return data


class Message(Base):  # type: ignore
    """
    消息模型

    存储对话中的每条消息，包括用户消息和 AI 回复。

    属性:
        id: 消息唯一标识符
        conversation_id: 所属对话 ID
        role: 消息角色（user/assistant/system/tool）
        content: 消息内容
        tokens: 消息 token 数
        model_name: 生成模型名称
        parent_id: 父消息 ID（用于编辑历史）
        is_edited: 是否已编辑
        is_deleted: 是否已删除
        metadata: 消息元数据（JSON）

    关系:
        conversation: 所属对话
    """
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="消息ID")
    conversation_id = Column(
        Integer, ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        comment="对话ID"
    )

    # 消息内容
    role = Column(String(32), nullable=False, comment="消息角色")
    content = Column(Text, nullable=False, comment="消息内容")

    # Token 统计
    prompt_tokens = Column(Integer, nullable=True, comment="提示token数")
    completion_tokens = Column(Integer, nullable=True, comment="补全token数")
    total_tokens = Column(Integer, nullable=True, comment="总token数")

    # 模型信息
    model_name = Column(String(128), nullable=True, comment="生成模型名称")
    model_provider = Column(String(64), nullable=True, comment="模型提供商")

    # 消息层级
    parent_id = Column(
        Integer, ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True, comment="父消息ID"
    )

    # 状态字段
    is_edited = Column(Boolean, nullable=False, default=False, comment="是否已编辑")
    is_deleted = Column(Boolean, nullable=False, default=False, comment="是否已删除")
    is_pinned = Column(Boolean, nullable=False, default=False, comment="是否置顶")
    is_flagged = Column(Boolean, nullable=False, default=False, comment="是否标记")
    flag_reason = Column(String(255), nullable=True, comment="标记原因")

    # 附件和媒体
    attachments = Column(JSON, nullable=True, comment="附件列表（JSON）")
    images = Column(JSON, nullable=True, comment="图片列表（JSON）")

    # 工具调用
    tool_calls = Column(JSON, nullable=True, comment="工具调用（JSON）")
    tool_results = Column(JSON, nullable=True, comment="工具结果（JSON）")

    # 评分和反馈
    rating = Column(Integer, nullable=True, comment="用户评分（1-5）")
    feedback = Column(Text, nullable=True, comment="用户反馈")

    # 费用
    cost = Column(Float, nullable=True, comment="消息费用")
    latency_ms = Column(Integer, nullable=True, comment="响应延迟（毫秒）")

    # 元数据
    metadata_json = Column(JSON, nullable=True, comment="消息元数据（JSON）")

    # 时间戳
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        comment="创建时间"
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        onupdate=get_utc_now,
        comment="更新时间"
    )

    # 关系定义
    conversation = relationship("Conversation", back_populates="messages", lazy="select")
    parent = relationship("Message", remote_side=[id], lazy="select")

    # 索引
    __table_args__ = (
        Index("ix_messages_conversation_id", "conversation_id"),
        Index("ix_messages_role", "role"),
        Index("ix_messages_created_at", "created_at"),
        Index("ix_messages_parent_id", "parent_id"),
        Index("ix_messages_is_deleted", "is_deleted"),
    )

    def __repr__(self) -> str:
        content_preview = self.content[:50] if self.content else ""
        return f"<Message(id={self.id}, role='{self.role}', content='{content_preview}...')>"

    def to_dict(self) -> Dict[str, Any]:
        """将消息转换为字典"""
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model_name": self.model_name,
            "model_provider": self.model_provider,
            "parent_id": self.parent_id,
            "is_edited": self.is_edited,
            "is_deleted": self.is_deleted,
            "is_pinned": self.is_pinned,
            "is_flagged": self.is_flagged,
            "flag_reason": self.flag_reason,
            "attachments": self.attachments,
            "images": self.images,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "rating": self.rating,
            "feedback": self.feedback,
            "cost": self.cost,
            "latency_ms": self.latency_ms,
            "metadata_json": self.metadata_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ============================================================
# 模型管理模型
# ============================================================

class Model(Base):  # type: ignore
    """
    AI 模型配置模型

    存储系统中可用的 AI 模型配置信息，包括 API 端点、认证信息和模型参数。

    属性:
        id: 模型唯一标识符
        name: 模型显示名称
        provider: 模型提供商
        model_type: 模型类型
        api_endpoint: API 端点 URL
        api_key_encrypted: 加密的 API 密钥
        config_json: 模型配置（JSON）
        status: 模型状态
        max_tokens: 最大 token 数
        cost_per_1k_tokens: 每 1000 token 费用

    关系:
        load_balances: 负载均衡配置
        training_jobs: 关联的训练任务
    """
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="模型ID")
    name = Column(String(128), nullable=False, unique=True, comment="模型名称")
    display_name = Column(String(255), nullable=True, comment="显示名称")
    description = Column(Text, nullable=True, comment="模型描述")

    # 提供商信息
    provider = Column(
        Enum(ModelProvider), nullable=False, default=ModelProvider.LOCAL,
        comment="模型提供商"
    )
    model_type = Column(
        Enum(ModelType), nullable=False, default=ModelType.LLM,
        comment="模型类型"
    )
    version = Column(String(64), nullable=True, comment="模型版本")

    # API 配置
    api_endpoint = Column(String(512), nullable=True, comment="API端点URL")
    api_key_encrypted = Column(Text, nullable=True, comment="加密的API密钥")
    api_key_name = Column(String(128), nullable=True, comment="API密钥名称")
    organization_id = Column(String(128), nullable=True, comment="组织ID")
    project_id = Column(String(128), nullable=True, comment="项目ID")

    # 模型参数
    max_tokens = Column(Integer, nullable=True, comment="最大token数")
    max_context_length = Column(Integer, nullable=True, comment="最大上下文长度")
    temperature_range = Column(JSON, nullable=True, comment="温度范围")
    top_p_range = Column(JSON, nullable=True, comment="top_p范围")
    supports_streaming = Column(
        Boolean, nullable=False, default=True,
        comment="是否支持流式响应"
    )
    supports_function_calling = Column(
        Boolean, nullable=False, default=False,
        comment="是否支持函数调用"
    )
    supports_vision = Column(
        Boolean, nullable=False, default=False,
        comment="是否支持视觉"
    )
    supports_json_mode = Column(
        Boolean, nullable=False, default=False,
        comment="是否支持JSON模式"
    )

    # 费用信息
    cost_per_1k_input_tokens = Column(
        Float, nullable=False, default=0.0,
        comment="每千输入token费用"
    )
    cost_per_1k_output_tokens = Column(
        Float, nullable=False, default=0.0,
        comment="每千输出token费用"
    )
    currency = Column(String(8), nullable=False, default="USD", comment="货币单位")

    # 性能指标
    avg_latency_ms = Column(Float, nullable=True, comment="平均延迟（毫秒）")
    success_rate = Column(Float, nullable=True, comment="成功率")
    total_requests = Column(BigInteger, nullable=False, default=0, comment="总请求数")
    total_tokens_used = Column(BigInteger, nullable=False, default=0, comment="总使用token数")
    total_errors = Column(Integer, nullable=False, default=0, comment="总错误数")

    # 状态
    status = Column(
        Enum(ModelStatus), nullable=False, default=ModelStatus.ACTIVE,
        comment="模型状态"
    )
    is_default = Column(
        Boolean, nullable=False, default=False,
        comment="是否为默认模型"
    )
    priority = Column(Integer, nullable=False, default=0, comment="优先级")

    # 配置
    config_json = Column(JSON, nullable=True, comment="模型配置（JSON）")
    capabilities = Column(JSON, nullable=True, comment="模型能力列表（JSON）")

    # 标签
    tags = Column(JSON, nullable=True, comment="标签（JSON数组）")

    # 时间戳
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        comment="创建时间"
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        onupdate=get_utc_now,
        comment="更新时间"
    )
    last_used_at = Column(
        DateTime(timezone=True), nullable=True,
        comment="最后使用时间"
    )
    last_health_check = Column(
        DateTime(timezone=True), nullable=True,
        comment="最后健康检查时间"
    )

    # 关系定义
    load_balances = relationship(
        "ModelLoadBalance", back_populates="model",
        cascade="all, delete-orphan", lazy="select"
    )

    # 索引
    __table_args__ = (
        Index("ix_models_provider", "provider"),
        Index("ix_models_model_type", "model_type"),
        Index("ix_models_status", "status"),
        Index("ix_models_is_default", "is_default"),
        Index("ix_models_priority", "priority"),
        Index("ix_models_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Model(id={self.id}, name='{self.name}', provider={self.provider.value})>"

    def to_dict(self, exclude_secrets: bool = True) -> Dict[str, Any]:
        """将模型配置转换为字典"""
        data = {
            "id": self.id,
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "provider": self.provider.value if self.provider else None,
            "model_type": self.model_type.value if self.model_type else None,
            "version": self.version,
            "api_endpoint": self.api_endpoint,
            "api_key_name": self.api_key_name,
            "organization_id": self.organization_id,
            "project_id": self.project_id,
            "max_tokens": self.max_tokens,
            "max_context_length": self.max_context_length,
            "temperature_range": self.temperature_range,
            "top_p_range": self.top_p_range,
            "supports_streaming": self.supports_streaming,
            "supports_function_calling": self.supports_function_calling,
            "supports_vision": self.supports_vision,
            "supports_json_mode": self.supports_json_mode,
            "cost_per_1k_input_tokens": self.cost_per_1k_input_tokens,
            "cost_per_1k_output_tokens": self.cost_per_1k_output_tokens,
            "currency": self.currency,
            "avg_latency_ms": self.avg_latency_ms,
            "success_rate": self.success_rate,
            "total_requests": self.total_requests,
            "total_tokens_used": self.total_tokens_used,
            "total_errors": self.total_errors,
            "status": self.status.value if self.status else None,
            "is_default": self.is_default,
            "priority": self.priority,
            "config_json": self.config_json,
            "capabilities": self.capabilities,
            "tags": self.tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
        }
        if not exclude_secrets:
            data["api_key_encrypted"] = self.api_key_encrypted
        return data


class ModelLoadBalance(Base):  # type: ignore
    """
    模型负载均衡配置模型

    存储模型的负载均衡策略和权重配置。

    属性:
        id: 配置唯一标识符
        model_id: 关联模型 ID
        strategy: 负载均衡策略
        weight: 权重
        health_score: 健康分数
        max_requests_per_minute: 每分钟最大请求数
        max_concurrent_requests: 最大并发请求数

    关系:
        model: 关联的模型
    """
    __tablename__ = "model_load_balances"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="配置ID")
    model_id = Column(
        Integer, ForeignKey("models.id", ondelete="CASCADE"),
        nullable=False, unique=True,
        comment="模型ID"
    )

    # 负载均衡配置
    strategy = Column(
        Enum(LoadBalanceStrategy), nullable=False,
        default=LoadBalanceStrategy.ROUND_ROBIN,
        comment="负载均衡策略"
    )
    weight = Column(Integer, nullable=False, default=100, comment="权重")
    health_score = Column(Float, nullable=False, default=100.0, comment="健康分数")
    health_check_interval = Column(
        Integer, nullable=False, default=60,
        comment="健康检查间隔（秒）"
    )

    # 限流配置
    max_requests_per_minute = Column(
        Integer, nullable=False, default=60,
        comment="每分钟最大请求数"
    )
    max_concurrent_requests = Column(
        Integer, nullable=False, default=10,
        comment="最大并发请求数"
    )
    max_tokens_per_minute = Column(
        Integer, nullable=True,
        comment="每分钟最大token数"
    )

    # 熔断配置
    circuit_breaker_enabled = Column(
        Boolean, nullable=False, default=True,
        comment="是否启用熔断器"
    )
    circuit_breaker_threshold = Column(
        Integer, nullable=False, default=5,
        comment="熔断阈值（连续错误次数）"
    )
    circuit_breaker_timeout = Column(
        Integer, nullable=False, default=60,
        comment="熔断恢复超时（秒）"
    )

    # 重试配置
    retry_enabled = Column(
        Boolean, nullable=False, default=True,
        comment="是否启用重试"
    )
    max_retries = Column(Integer, nullable=False, default=3, comment="最大重试次数")
    retry_delay = Column(Float, nullable=False, default=1.0, comment="重试延迟（秒）")
    retry_backoff_multiplier = Column(
        Float, nullable=False, default=2.0,
        comment="重试退避倍数"
    )

    # 超时配置
    request_timeout = Column(
        Integer, nullable=False, default=30,
        comment="请求超时（秒）"
    )
    connect_timeout = Column(
        Integer, nullable=False, default=10,
        comment="连接超时（秒）"
    )

    # 统计
    current_requests = Column(
        Integer, nullable=False, default=0,
        comment="当前请求数"
    )
    requests_last_minute = Column(
        Integer, nullable=False, default=0,
        comment="最近一分钟请求数"
    )
    errors_last_minute = Column(
        Integer, nullable=False, default=0,
        comment="最近一分钟错误数"
    )

    # 配置
    config_json = Column(JSON, nullable=True, comment="额外配置（JSON）")

    # 时间戳
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        comment="创建时间"
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        onupdate=get_utc_now,
        comment="更新时间"
    )

    # 关系定义
    model = relationship("Model", back_populates="load_balances", lazy="select")

    # 索引
    __table_args__ = (
        Index("ix_model_lb_strategy", "strategy"),
        Index("ix_model_lb_health_score", "health_score"),
    )

    def __repr__(self) -> str:
        return (
            f"<ModelLoadBalance(id={self.id}, model_id={self.model_id}, "
            f"strategy={self.strategy.value})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        """将负载均衡配置转换为字典"""
        return {
            "id": self.id,
            "model_id": self.model_id,
            "strategy": self.strategy.value if self.strategy else None,
            "weight": self.weight,
            "health_score": self.health_score,
            "health_check_interval": self.health_check_interval,
            "max_requests_per_minute": self.max_requests_per_minute,
            "max_concurrent_requests": self.max_concurrent_requests,
            "max_tokens_per_minute": self.max_tokens_per_minute,
            "circuit_breaker_enabled": self.circuit_breaker_enabled,
            "circuit_breaker_threshold": self.circuit_breaker_threshold,
            "circuit_breaker_timeout": self.circuit_breaker_timeout,
            "retry_enabled": self.retry_enabled,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "retry_backoff_multiplier": self.retry_backoff_multiplier,
            "request_timeout": self.request_timeout,
            "connect_timeout": self.connect_timeout,
            "current_requests": self.current_requests,
            "requests_last_minute": self.requests_last_minute,
            "errors_last_minute": self.errors_last_minute,
            "config_json": self.config_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ============================================================
# 训练系统模型
# ============================================================

class TrainingJob(Base):  # type: ignore
    """
    训练任务模型

    存储模型训练任务的配置、状态和结果。

    属性:
        id: 任务唯一标识符
        name: 任务名称
        user_id: 创建者用户 ID
        model_id: 关联模型 ID
        status: 任务状态
        config_json: 训练配置（JSON）
        metrics_json: 训练指标（JSON）
        progress: 训练进度（0-100）

    关系:
        user: 创建者
        model: 关联模型
        checkpoints: 训练检查点列表
    """
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="训练任务ID")
    name = Column(String(255), nullable=False, comment="任务名称")
    description = Column(Text, nullable=True, comment="任务描述")
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment="创建者用户ID"
    )
    model_id = Column(
        Integer, ForeignKey("models.id", ondelete="SET NULL"),
        nullable=True,
        comment="关联模型ID"
    )

    # 任务状态
    status = Column(
        Enum(TrainingStatus), nullable=False, default=TrainingStatus.PENDING,
        comment="任务状态"
    )
    progress = Column(Float, nullable=False, default=0.0, comment="训练进度（0-100）")
    priority = Column(Integer, nullable=False, default=0, comment="优先级")

    # 训练配置
    config_json = Column(JSON, nullable=True, comment="训练配置（JSON）")
    hyperparameters = Column(JSON, nullable=True, comment="超参数（JSON）")

    # 数据集配置
    dataset_id = Column(Integer, nullable=True, comment="数据集ID")
    dataset_config = Column(JSON, nullable=True, comment="数据集配置（JSON）")
    train_split = Column(Float, nullable=False, default=0.8, comment="训练集比例")
    val_split = Column(Float, nullable=False, default=0.1, comment="验证集比例")
    test_split = Column(Float, nullable=False, default=0.1, comment="测试集比例")

    # 训练指标
    metrics_json = Column(JSON, nullable=True, comment="训练指标（JSON）")
    best_metrics = Column(JSON, nullable=True, comment="最佳指标（JSON）")
    final_metrics = Column(JSON, nullable=True, comment="最终指标（JSON）")

    # 资源使用
    gpu_type = Column(String(64), nullable=True, comment="GPU类型")
    gpu_count = Column(Integer, nullable=False, default=1, comment="GPU数量")
    total_epochs = Column(Integer, nullable=True, comment="总轮次")
    current_epoch = Column(Integer, nullable=False, default=0, comment="当前轮次")
    current_step = Column(Integer, nullable=False, default=0, comment="当前步数")
    total_steps = Column(Integer, nullable=True, comment="总步数")
    estimated_time_remaining = Column(Integer, nullable=True, comment="预计剩余时间（秒）")

    # 输出
    output_dir = Column(String(512), nullable=True, comment="输出目录")
    output_model_path = Column(String(512), nullable=True, comment="输出模型路径")
    log_path = Column(String(512), nullable=True, comment="日志路径")
    tensorboard_path = Column(String(512), nullable=True, comment="TensorBoard路径")

    # 错误信息
    error_message = Column(Text, nullable=True, comment="错误信息")
    error_traceback = Column(Text, nullable=True, comment="错误堆栈")

    # 费用
    estimated_cost = Column(Float, nullable=True, comment="预估费用")
    actual_cost = Column(Float, nullable=True, comment="实际费用")

    # 时间戳
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        comment="创建时间"
    )
    started_at = Column(
        DateTime(timezone=True), nullable=True,
        comment="开始时间"
    )
    completed_at = Column(
        DateTime(timezone=True), nullable=True,
        comment="完成时间"
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        onupdate=get_utc_now,
        comment="更新时间"
    )

    # 关系定义
    user = relationship("User", back_populates="training_jobs", lazy="select")
    checkpoints = relationship(
        "Checkpoint", back_populates="training_job",
        cascade="all, delete-orphan", lazy="select",
        order_by="Checkpoint.created_at"
    )

    # 索引
    __table_args__ = (
        Index("ix_training_jobs_status", "status"),
        Index("ix_training_jobs_user_id", "user_id"),
        Index("ix_training_jobs_model_id", "model_id"),
        Index("ix_training_jobs_priority", "priority"),
        Index("ix_training_jobs_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<TrainingJob(id={self.id}, name='{self.name}', "
            f"status={self.status.value})>"
        )

    def to_dict(self, include_checkpoints: bool = False) -> Dict[str, Any]:
        """将训练任务转换为字典"""
        data = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "user_id": self.user_id,
            "model_id": self.model_id,
            "status": self.status.value if self.status else None,
            "progress": self.progress,
            "priority": self.priority,
            "config_json": self.config_json,
            "hyperparameters": self.hyperparameters,
            "dataset_id": self.dataset_id,
            "dataset_config": self.dataset_config,
            "train_split": self.train_split,
            "val_split": self.val_split,
            "test_split": self.test_split,
            "metrics_json": self.metrics_json,
            "best_metrics": self.best_metrics,
            "final_metrics": self.final_metrics,
            "gpu_type": self.gpu_type,
            "gpu_count": self.gpu_count,
            "total_epochs": self.total_epochs,
            "current_epoch": self.current_epoch,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "estimated_time_remaining": self.estimated_time_remaining,
            "output_dir": self.output_dir,
            "output_model_path": self.output_model_path,
            "log_path": self.log_path,
            "tensorboard_path": self.tensorboard_path,
            "error_message": self.error_message,
            "estimated_cost": self.estimated_cost,
            "actual_cost": self.actual_cost,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_checkpoints:
            data["checkpoints"] = [cp.to_dict() for cp in self.checkpoints]
        return data


class Checkpoint(Base):  # type: ignore
    """
    训练检查点模型

    存储训练过程中的检查点信息。

    属性:
        id: 检查点唯一标识符
        training_job_id: 关联训练任务 ID
        epoch: 轮次
        step: 步数
        metrics_json: 检查点指标（JSON）
        file_path: 检查点文件路径
        file_size: 文件大小（字节）
        is_best: 是否为最佳检查点

    关系:
        training_job: 关联的训练任务
    """
    __tablename__ = "checkpoints"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="检查点ID")
    training_job_id = Column(
        Integer, ForeignKey("training_jobs.id", ondelete="CASCADE"),
        nullable=False,
        comment="训练任务ID"
    )

    # 检查点信息
    epoch = Column(Integer, nullable=False, comment="轮次")
    step = Column(Integer, nullable=False, default=0, comment="步数")
    metrics_json = Column(JSON, nullable=True, comment="检查点指标（JSON）")
    loss = Column(Float, nullable=True, comment="损失值")

    # 文件信息
    file_path = Column(String(512), nullable=True, comment="检查点文件路径")
    file_size = Column(BigInteger, nullable=True, comment="文件大小（字节）")
    file_hash = Column(String(128), nullable=True, comment="文件哈希")

    # 状态
    is_best = Column(Boolean, nullable=False, default=False, comment="是否为最佳检查点")
    is_valid = Column(Boolean, nullable=False, default=True, comment="是否有效")

    # 备注
    notes = Column(Text, nullable=True, comment="备注")

    # 时间戳
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        comment="创建时间"
    )

    # 关系定义
    training_job = relationship("TrainingJob", back_populates="checkpoints", lazy="select")

    # 索引
    __table_args__ = (
        Index("ix_checkpoints_training_job_id", "training_job_id"),
        Index("ix_checkpoints_epoch", "epoch"),
        Index("ix_checkpoints_is_best", "is_best"),
        Index("ix_checkpoints_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<Checkpoint(id={self.id}, training_job_id={self.training_job_id}, "
            f"epoch={self.epoch}, step={self.step})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        """将检查点转换为字典"""
        return {
            "id": self.id,
            "training_job_id": self.training_job_id,
            "epoch": self.epoch,
            "step": self.step,
            "metrics_json": self.metrics_json,
            "loss": self.loss,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "file_hash": self.file_hash,
            "is_best": self.is_best,
            "is_valid": self.is_valid,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ============================================================
# 生成系统模型
# ============================================================

class GeneratedContent(Base):  # type: ignore
    """
    生成内容模型

    存储用户通过 AI 生成的各类内容（图像、视频、音频、3D 等）。

    属性:
        id: 内容唯一标识符
        user_id: 创建者用户 ID
        type: 内容类型
        prompt: 生成提示词
        params_json: 生成参数（JSON）
        result_path: 结果文件路径
        thumbnail_path: 缩略图路径

    关系:
        user: 创建者
    """
    __tablename__ = "generated_contents"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="内容ID")
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment="用户ID"
    )

    # 内容信息
    type = Column(
        Enum(ContentType), nullable=False,
        comment="内容类型"
    )
    title = Column(String(255), nullable=True, comment="标题")
    prompt = Column(Text, nullable=False, comment="生成提示词")
    negative_prompt = Column(Text, nullable=True, comment="反向提示词")

    # 模型信息
    model_name = Column(String(128), nullable=True, comment="使用的模型名称")
    model_provider = Column(String(64), nullable=True, comment="模型提供商")

    # 生成参数
    params_json = Column(JSON, nullable=True, comment="生成参数（JSON）")
    seed = Column(BigInteger, nullable=True, comment="随机种子")
    width = Column(Integer, nullable=True, comment="宽度")
    height = Column(Integer, nullable=True, comment="高度")
    duration = Column(Float, nullable=True, comment="时长（秒）")
    fps = Column(Integer, nullable=True, comment="帧率")
    quality = Column(String(32), nullable=True, comment="质量")

    # 结果文件
    result_path = Column(String(512), nullable=True, comment="结果文件路径")
    result_url = Column(String(512), nullable=True, comment="结果文件URL")
    thumbnail_path = Column(String(512), nullable=True, comment="缩略图路径")
    file_size = Column(BigInteger, nullable=True, comment="文件大小（字节）")
    file_hash = Column(String(128), nullable=True, comment="文件哈希")
    mime_type = Column(String(64), nullable=True, comment="MIME类型")

    # 元数据
    metadata_json = Column(JSON, nullable=True, comment="元数据（JSON）")
    tags = Column(JSON, nullable=True, comment="标签（JSON数组）")

    # 费用
    cost = Column(Float, nullable=True, comment="生成费用")
    tokens_used = Column(Integer, nullable=True, comment="使用token数")
    generation_time_ms = Column(Integer, nullable=True, comment="生成时间（毫秒）")

    # 状态
    is_public = Column(Boolean, nullable=False, default=False, comment="是否公开")
    is_favorite = Column(Boolean, nullable=False, default=False, comment="是否收藏")

    # 时间戳
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        comment="创建时间"
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        onupdate=get_utc_now,
        comment="更新时间"
    )

    # 关系定义
    user = relationship("User", back_populates="generated_contents", lazy="select")

    # 索引
    __table_args__ = (
        Index("ix_generated_contents_user_id", "user_id"),
        Index("ix_generated_contents_type", "type"),
        Index("ix_generated_contents_created_at", "created_at"),
        Index("ix_generated_contents_is_public", "is_public"),
        Index("ix_generated_contents_is_favorite", "is_favorite"),
    )

    def __repr__(self) -> str:
        return (
            f"<GeneratedContent(id={self.id}, type={self.type.value}, "
            f"user_id={self.user_id})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        """将生成内容转换为字典"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "type": self.type.value if self.type else None,
            "title": self.title,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "model_name": self.model_name,
            "model_provider": self.model_provider,
            "params_json": self.params_json,
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "duration": self.duration,
            "fps": self.fps,
            "quality": self.quality,
            "result_path": self.result_path,
            "result_url": self.result_url,
            "thumbnail_path": self.thumbnail_path,
            "file_size": self.file_size,
            "file_hash": self.file_hash,
            "mime_type": self.mime_type,
            "metadata_json": self.metadata_json,
            "tags": self.tags,
            "cost": self.cost,
            "tokens_used": self.tokens_used,
            "generation_time_ms": self.generation_time_ms,
            "is_public": self.is_public,
            "is_favorite": self.is_favorite,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ============================================================
# 工作流系统模型
# ============================================================

class Workflow(Base):  # type: ignore
    """
    工作流模型

    存储自动化工作流的定义和配置。

    属性:
        id: 工作流唯一标识符
        name: 工作流名称
        user_id: 创建者用户 ID
        config_json: 工作流配置（JSON）
        status: 工作流状态
        description: 工作流描述
        version: 工作流版本

    关系:
        user: 创建者
        executions: 执行记录列表
    """
    __tablename__ = "workflows"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="工作流ID")
    name = Column(String(255), nullable=False, comment="工作流名称")
    description = Column(Text, nullable=True, comment="工作流描述")
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment="创建者用户ID"
    )

    # 工作流配置
    config_json = Column(JSON, nullable=True, comment="工作流配置（JSON）")
    graph_json = Column(JSON, nullable=True, comment="工作流图定义（JSON）")
    variables = Column(JSON, nullable=True, comment="工作流变量（JSON）")
    triggers = Column(JSON, nullable=True, comment="触发器配置（JSON）")

    # 状态
    status = Column(
        Enum(WorkflowStatus), nullable=False, default=WorkflowStatus.DRAFT,
        comment="工作流状态"
    )
    version = Column(Integer, nullable=False, default=1, comment="版本号")
    is_template = Column(
        Boolean, nullable=False, default=False,
        comment="是否为模板"
    )
    category = Column(String(64), nullable=True, comment="分类")
    tags = Column(JSON, nullable=True, comment="标签（JSON数组）")

    # 执行统计
    total_executions = Column(
        Integer, nullable=False, default=0,
        comment="总执行次数"
    )
    successful_executions = Column(
        Integer, nullable=False, default=0,
        comment="成功执行次数"
    )
    failed_executions = Column(
        Integer, nullable=False, default=0,
        comment="失败执行次数"
    )
    avg_execution_time_ms = Column(
        Float, nullable=True,
        comment="平均执行时间（毫秒）"
    )

    # 调度
    schedule_cron = Column(String(128), nullable=True, comment="Cron 调度表达式")
    schedule_enabled = Column(
        Boolean, nullable=False, default=False,
        comment="是否启用调度"
    )
    last_execution_at = Column(
        DateTime(timezone=True), nullable=True,
        comment="最后执行时间"
    )
    next_execution_at = Column(
        DateTime(timezone=True), nullable=True,
        comment="下次执行时间"
    )

    # 时间戳
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        comment="创建时间"
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        onupdate=get_utc_now,
        comment="更新时间"
    )

    # 关系定义
    user = relationship("User", back_populates="workflows", lazy="select")
    executions = relationship(
        "WorkflowExecution", back_populates="workflow",
        cascade="all, delete-orphan", lazy="select",
        order_by="WorkflowExecution.started_at"
    )

    # 索引
    __table_args__ = (
        Index("ix_workflows_user_id", "user_id"),
        Index("ix_workflows_status", "status"),
        Index("ix_workflows_category", "category"),
        Index("ix_workflows_is_template", "is_template"),
        Index("ix_workflows_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Workflow(id={self.id}, name='{self.name}', status={self.status.value})>"

    def to_dict(self, include_executions: bool = False) -> Dict[str, Any]:
        """将工作流转换为字典"""
        data = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "user_id": self.user_id,
            "config_json": self.config_json,
            "graph_json": self.graph_json,
            "variables": self.variables,
            "triggers": self.triggers,
            "status": self.status.value if self.status else None,
            "version": self.version,
            "is_template": self.is_template,
            "category": self.category,
            "tags": self.tags,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "avg_execution_time_ms": self.avg_execution_time_ms,
            "schedule_cron": self.schedule_cron,
            "schedule_enabled": self.schedule_enabled,
            "last_execution_at": self.last_execution_at.isoformat() if self.last_execution_at else None,
            "next_execution_at": self.next_execution_at.isoformat() if self.next_execution_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_executions:
            data["executions"] = [e.to_dict() for e in self.executions]
        return data


class WorkflowExecution(Base):  # type: ignore
    """
    工作流执行记录模型

    存储工作流的每次执行记录和结果。

    属性:
        id: 执行记录唯一标识符
        workflow_id: 关联工作流 ID
        status: 执行状态
        result_json: 执行结果（JSON）
        error_message: 错误信息
        started_at: 开始时间
        completed_at: 完成时间
        duration_ms: 执行时长（毫秒）

    关系:
        workflow: 关联的工作流
    """
    __tablename__ = "workflow_executions"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="执行记录ID")
    workflow_id = Column(
        Integer, ForeignKey("workflows.id", ondelete="CASCADE"),
        nullable=False,
        comment="工作流ID"
    )

    # 执行信息
    status = Column(
        Enum(ExecutionStatus), nullable=False, default=ExecutionStatus.PENDING,
        comment="执行状态"
    )
    trigger_type = Column(String(64), nullable=True, comment="触发类型")
    trigger_data = Column(JSON, nullable=True, comment="触发数据（JSON）")

    # 输入输出
    input_json = Column(JSON, nullable=True, comment="输入数据（JSON）")
    output_json = Column(JSON, nullable=True, comment="输出数据（JSON）")
    result_json = Column(JSON, nullable=True, comment="执行结果（JSON）")

    # 执行详情
    steps_json = Column(JSON, nullable=True, comment="执行步骤详情（JSON）")
    error_message = Column(Text, nullable=True, comment="错误信息")
    error_traceback = Column(Text, nullable=True, comment="错误堆栈")
    error_step = Column(String(128), nullable=True, comment="错误步骤")

    # 执行者
    executed_by = Column(Integer, nullable=True, comment="执行者用户ID")
    executed_by_system = Column(
        Boolean, nullable=False, default=False,
        comment="是否系统执行"
    )

    # 资源使用
    duration_ms = Column(Integer, nullable=True, comment="执行时长（毫秒）")
    memory_peak_mb = Column(Float, nullable=True, comment="峰值内存（MB）")
    cpu_time_ms = Column(Integer, nullable=True, comment="CPU时间（毫秒）")

    # 重试信息
    retry_count = Column(Integer, nullable=False, default=0, comment="重试次数")
    parent_execution_id = Column(
        Integer, ForeignKey("workflow_executions.id", ondelete="SET NULL"),
        nullable=True, comment="父执行记录ID（重试时）"
    )

    # 时间戳
    started_at = Column(
        DateTime(timezone=True), nullable=True,
        comment="开始时间"
    )
    completed_at = Column(
        DateTime(timezone=True), nullable=True,
        comment="完成时间"
    )
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        comment="创建时间"
    )

    # 关系定义
    workflow = relationship("Workflow", back_populates="executions", lazy="select")
    parent_execution = relationship(
        "WorkflowExecution", remote_side=[id], lazy="select"
    )

    # 索引
    __table_args__ = (
        Index("ix_wf_executions_workflow_id", "workflow_id"),
        Index("ix_wf_executions_status", "status"),
        Index("ix_wf_executions_started_at", "started_at"),
        Index("ix_wf_executions_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<WorkflowExecution(id={self.id}, workflow_id={self.workflow_id}, "
            f"status={self.status.value})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        """将执行记录转换为字典"""
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "status": self.status.value if self.status else None,
            "trigger_type": self.trigger_type,
            "trigger_data": self.trigger_data,
            "input_json": self.input_json,
            "output_json": self.output_json,
            "result_json": self.result_json,
            "steps_json": self.steps_json,
            "error_message": self.error_message,
            "error_traceback": self.error_traceback,
            "error_step": self.error_step,
            "executed_by": self.executed_by,
            "executed_by_system": self.executed_by_system,
            "duration_ms": self.duration_ms,
            "memory_peak_mb": self.memory_peak_mb,
            "cpu_time_ms": self.cpu_time_ms,
            "retry_count": self.retry_count,
            "parent_execution_id": self.parent_execution_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ============================================================
# 智能体系统模型
# ============================================================

class Agent(Base):  # type: ignore
    """
    智能体模型

    存储系统中 AI 智能体的配置和能力定义。

    属性:
        id: 智能体唯一标识符
        name: 智能体名称
        user_id: 创建者用户 ID
        config_json: 智能体配置（JSON）
        capabilities: 能力列表（JSON）
        status: 智能体状态
        description: 智能体描述

    关系:
        user: 创建者
    """
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="智能体ID")
    name = Column(String(255), nullable=False, comment="智能体名称")
    description = Column(Text, nullable=True, comment="智能体描述")
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment="创建者用户ID"
    )

    # 智能体配置
    config_json = Column(JSON, nullable=True, comment="智能体配置（JSON）")
    system_prompt = Column(Text, nullable=True, comment="系统提示词")
    model_name = Column(String(128), nullable=True, comment="使用的模型名称")
    model_config = Column(JSON, nullable=True, comment="模型配置（JSON）")

    # 能力定义
    capabilities = Column(JSON, nullable=True, comment="能力列表（JSON）")
    tools = Column(JSON, nullable=True, comment="工具列表（JSON）")
    knowledge_bases = Column(JSON, nullable=True, comment="知识库列表（JSON）")

    # 行为配置
    personality_id = Column(Integer, nullable=True, comment="关联人格ID")
    memory_config = Column(JSON, nullable=True, comment="记忆配置（JSON）")
    reasoning_config = Column(JSON, nullable=True, comment="推理配置（JSON）")
    planning_config = Column(JSON, nullable=True, comment="规划配置（JSON）")

    # 状态
    status = Column(
        Enum(AgentStatus), nullable=False, default=AgentStatus.ACTIVE,
        comment="智能体状态"
    )
    is_public = Column(Boolean, nullable=False, default=False, comment="是否公开")
    is_template = Column(Boolean, nullable=False, default=False, comment="是否为模板")
    version = Column(Integer, nullable=False, default=1, comment="版本号")
    category = Column(String(64), nullable=True, comment="分类")
    tags = Column(JSON, nullable=True, comment="标签（JSON数组）")
    icon = Column(String(512), nullable=True, comment="图标URL")

    # 统计
    total_interactions = Column(
        Integer, nullable=False, default=0,
        comment="总交互次数"
    )
    avg_response_time_ms = Column(
        Float, nullable=True,
        comment="平均响应时间（毫秒）"
    )
    user_satisfaction_score = Column(
        Float, nullable=True,
        comment="用户满意度评分"
    )

    # 时间戳
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        comment="创建时间"
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        onupdate=get_utc_now,
        comment="更新时间"
    )
    last_active_at = Column(
        DateTime(timezone=True), nullable=True,
        comment="最后活跃时间"
    )

    # 关系定义
    user = relationship("User", back_populates="agents", lazy="select")

    # 索引
    __table_args__ = (
        Index("ix_agents_user_id", "user_id"),
        Index("ix_agents_status", "status"),
        Index("ix_agents_category", "category"),
        Index("ix_agents_is_public", "is_public"),
        Index("ix_agents_is_template", "is_template"),
        Index("ix_agents_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Agent(id={self.id}, name='{self.name}', status={self.status.value})>"

    def to_dict(self) -> Dict[str, Any]:
        """将智能体转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "user_id": self.user_id,
            "config_json": self.config_json,
            "system_prompt": self.system_prompt,
            "model_name": self.model_name,
            "model_config": self.model_config,
            "capabilities": self.capabilities,
            "tools": self.tools,
            "knowledge_bases": self.knowledge_bases,
            "personality_id": self.personality_id,
            "memory_config": self.memory_config,
            "reasoning_config": self.reasoning_config,
            "planning_config": self.planning_config,
            "status": self.status.value if self.status else None,
            "is_public": self.is_public,
            "is_template": self.is_template,
            "version": self.version,
            "category": self.category,
            "tags": self.tags,
            "icon": self.icon,
            "total_interactions": self.total_interactions,
            "avg_response_time_ms": self.avg_response_time_ms,
            "user_satisfaction_score": self.user_satisfaction_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_active_at": self.last_active_at.isoformat() if self.last_active_at else None,
        }


class Alliance(Base):  # type: ignore
    """
    智能体联盟模型

    存储多个智能体组成的联盟配置和协作规则。

    属性:
        id: 联盟唯一标识符
        name: 联盟名称
        members_json: 成员列表（JSON）
        config_json: 联盟配置（JSON）
        description: 联盟描述
    """
    __tablename__ = "alliances"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="联盟ID")
    name = Column(String(255), nullable=False, comment="联盟名称")
    description = Column(Text, nullable=True, comment="联盟描述")

    # 联盟配置
    members_json = Column(JSON, nullable=True, comment="成员列表（JSON）")
    config_json = Column(JSON, nullable=True, comment="联盟配置（JSON）")
    collaboration_rules = Column(
        JSON, nullable=True,
        comment="协作规则（JSON）"
    )
    communication_config = Column(
        JSON, nullable=True,
        comment="通信配置（JSON）"
    )
    consensus_config = Column(
        JSON, nullable=True,
        comment="共识配置（JSON）"
    )

    # 状态
    status = Column(
        Enum(AgentStatus), nullable=False, default=AgentStatus.ACTIVE,
        comment="联盟状态"
    )
    is_active = Column(Boolean, nullable=False, default=True, comment="是否活跃")
    max_members = Column(Integer, nullable=False, default=10, comment="最大成员数")
    current_member_count = Column(
        Integer, nullable=False, default=0,
        comment="当前成员数"
    )

    # 统计
    total_collaborations = Column(
        Integer, nullable=False, default=0,
        comment="总协作次数"
    )
    avg_collaboration_time_ms = Column(
        Float, nullable=True,
        comment="平均协作时间（毫秒）"
    )

    # 标签
    category = Column(String(64), nullable=True, comment="分类")
    tags = Column(JSON, nullable=True, comment="标签（JSON数组）")

    # 时间戳
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        comment="创建时间"
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        onupdate=get_utc_now,
        comment="更新时间"
    )
    last_active_at = Column(
        DateTime(timezone=True), nullable=True,
        comment="最后活跃时间"
    )

    # 索引
    __table_args__ = (
        Index("ix_alliances_status", "status"),
        Index("ix_alliances_is_active", "is_active"),
        Index("ix_alliances_category", "category"),
        Index("ix_alliances_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Alliance(id={self.id}, name='{self.name}', status={self.status.value})>"

    def to_dict(self) -> Dict[str, Any]:
        """将联盟转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "members_json": self.members_json,
            "config_json": self.config_json,
            "collaboration_rules": self.collaboration_rules,
            "communication_config": self.communication_config,
            "consensus_config": self.consensus_config,
            "status": self.status.value if self.status else None,
            "is_active": self.is_active,
            "max_members": self.max_members,
            "current_member_count": self.current_member_count,
            "total_collaborations": self.total_collaborations,
            "avg_collaboration_time_ms": self.avg_collaboration_time_ms,
            "category": self.category,
            "tags": self.tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_active_at": self.last_active_at.isoformat() if self.last_active_at else None,
        }


# ============================================================
# 插件系统模型
# ============================================================

class Plugin(Base):  # type: ignore
    """
    插件模型

    存储系统中安装的插件信息。

    属性:
        id: 插件唯一标识符
        name: 插件名称
        version: 插件版本
        author: 插件作者
        description: 插件描述
        config_json: 插件配置（JSON）
        status: 插件状态
        installed_at: 安装时间
    """
    __tablename__ = "plugins"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="插件ID")
    name = Column(String(128), nullable=False, unique=True, comment="插件名称")
    display_name = Column(String(255), nullable=True, comment="显示名称")
    description = Column(Text, nullable=True, comment="插件描述")
    author = Column(String(128), nullable=True, comment="作者")
    homepage = Column(String(512), nullable=True, comment="主页URL")
    repository = Column(String(512), nullable=True, comment="代码仓库URL")
    license = Column(String(64), nullable=True, comment="许可证")
    version = Column(String(32), nullable=False, default="1.0.0", comment="版本号")

    # 插件配置
    config_json = Column(JSON, nullable=True, comment="插件配置（JSON）")
    settings_schema = Column(JSON, nullable=True, comment="设置模式（JSON）")
    permissions = Column(JSON, nullable=True, comment="权限列表（JSON）")

    # 依赖
    dependencies = Column(JSON, nullable=True, comment="依赖列表（JSON）")
    compatible_version = Column(String(64), nullable=True, comment="兼容版本范围")

    # 状态
    status = Column(
        Enum(PluginStatus), nullable=False, default=PluginStatus.INSTALLED,
        comment="插件状态"
    )

    # 文件信息
    install_path = Column(String(512), nullable=True, comment="安装路径")
    entry_point = Column(String(255), nullable=True, comment="入口点")
    icon = Column(String(512), nullable=True, comment="图标URL")

    # 统计
    total_uses = Column(Integer, nullable=False, default=0, comment="总使用次数")
    avg_execution_time_ms = Column(Float, nullable=True, comment="平均执行时间（毫秒）")

    # 标签
    category = Column(String(64), nullable=True, comment="分类")
    tags = Column(JSON, nullable=True, comment="标签（JSON数组）")

    # 时间戳
    installed_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        comment="安装时间"
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        onupdate=get_utc_now,
        comment="更新时间"
    )
    last_used_at = Column(
        DateTime(timezone=True), nullable=True,
        comment="最后使用时间"
    )

    # 索引
    __table_args__ = (
        Index("ix_plugins_status", "status"),
        Index("ix_plugins_category", "category"),
        Index("ix_plugins_author", "author"),
        Index("ix_plugins_installed_at", "installed_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<Plugin(id={self.id}, name='{self.name}', "
            f"version='{self.version}', status={self.status.value})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        """将插件转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "author": self.author,
            "homepage": self.homepage,
            "repository": self.repository,
            "license": self.license,
            "version": self.version,
            "config_json": self.config_json,
            "settings_schema": self.settings_schema,
            "permissions": self.permissions,
            "dependencies": self.dependencies,
            "compatible_version": self.compatible_version,
            "status": self.status.value if self.status else None,
            "install_path": self.install_path,
            "entry_point": self.entry_point,
            "icon": self.icon,
            "total_uses": self.total_uses,
            "avg_execution_time_ms": self.avg_execution_time_ms,
            "category": self.category,
            "tags": self.tags,
            "installed_at": self.installed_at.isoformat() if self.installed_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
        }


# ============================================================
# 渠道系统模型
# ============================================================

class Channel(Base):  # type: ignore
    """
    渠道模型

    存储外部通信渠道的配置信息（Discord、Telegram、Slack 等）。

    属性:
        id: 渠道唯一标识符
        name: 渠道名称
        channel_type: 渠道类型
        config_json: 渠道配置（JSON）
        status: 渠道状态
    """
    __tablename__ = "channels"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="渠道ID")
    name = Column(String(255), nullable=False, comment="渠道名称")
    description = Column(Text, nullable=True, comment="渠道描述")

    # 渠道类型
    channel_type = Column(
        Enum(ChannelType), nullable=False,
        comment="渠道类型"
    )

    # 渠道配置
    config_json = Column(JSON, nullable=True, comment="渠道配置（JSON）")
    webhook_url = Column(String(512), nullable=True, comment="Webhook URL")
    api_token_encrypted = Column(Text, nullable=True, comment="加密的API令牌")
    bot_id = Column(String(128), nullable=True, comment="机器人ID")

    # 状态
    status = Column(
        Enum(ChannelStatus), nullable=False, default=ChannelStatus.INACTIVE,
        comment="渠道状态"
    )
    is_default = Column(
        Boolean, nullable=False, default=False,
        comment="是否为默认渠道"
    )

    # 功能配置
    features = Column(JSON, nullable=True, comment="功能配置（JSON）")
    allowed_commands = Column(JSON, nullable=True, comment="允许的命令（JSON）")
    blocked_users = Column(JSON, nullable=True, comment="黑名单用户（JSON）")
    allowed_users = Column(JSON, nullable=True, comment="白名单用户（JSON）")

    # 限流
    rate_limit_per_minute = Column(
        Integer, nullable=False, default=60,
        comment="每分钟消息限制"
    )
    rate_limit_per_day = Column(
        Integer, nullable=False, default=1000,
        comment="每日消息限制"
    )

    # 统计
    total_messages_sent = Column(
        Integer, nullable=False, default=0,
        comment="总发送消息数"
    )
    total_messages_received = Column(
        Integer, nullable=False, default=0,
        comment="总接收消息数"
    )
    total_errors = Column(
        Integer, nullable=False, default=0,
        comment="总错误数"
    )
    last_message_at = Column(
        DateTime(timezone=True), nullable=True,
        comment="最后消息时间"
    )

    # 标签
    tags = Column(JSON, nullable=True, comment="标签（JSON数组）")

    # 时间戳
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        comment="创建时间"
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        onupdate=get_utc_now,
        comment="更新时间"
    )
    last_health_check = Column(
        DateTime(timezone=True), nullable=True,
        comment="最后健康检查时间"
    )

    # 索引
    __table_args__ = (
        Index("ix_channels_channel_type", "channel_type"),
        Index("ix_channels_status", "status"),
        Index("ix_channels_is_default", "is_default"),
        Index("ix_channels_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<Channel(id={self.id}, name='{self.name}', "
            f"type={self.channel_type.value})>"
        )

    def to_dict(self, exclude_secrets: bool = True) -> Dict[str, Any]:
        """将渠道转换为字典"""
        data = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "channel_type": self.channel_type.value if self.channel_type else None,
            "config_json": self.config_json,
            "webhook_url": self.webhook_url,
            "bot_id": self.bot_id,
            "status": self.status.value if self.status else None,
            "is_default": self.is_default,
            "features": self.features,
            "allowed_commands": self.allowed_commands,
            "blocked_users": self.blocked_users,
            "allowed_users": self.allowed_users,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "rate_limit_per_day": self.rate_limit_per_day,
            "total_messages_sent": self.total_messages_sent,
            "total_messages_received": self.total_messages_received,
            "total_errors": self.total_errors,
            "last_message_at": self.last_message_at.isoformat() if self.last_message_at else None,
            "tags": self.tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
        }
        if not exclude_secrets:
            data["api_token_encrypted"] = self.api_token_encrypted
        return data


# ============================================================
# 数据集系统模型
# ============================================================

class Dataset(Base):  # type: ignore
    """
    数据集模型

    存储系统中的数据集信息。

    属性:
        id: 数据集唯一标识符
        name: 数据集名称
        type: 数据集类型
        size: 数据集大小（字节）
        file_path: 数据集文件路径
        config_json: 数据集配置（JSON）
    """
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="数据集ID")
    name = Column(String(255), nullable=False, comment="数据集名称")
    description = Column(Text, nullable=True, comment="数据集描述")
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment="创建者用户ID"
    )

    # 数据集信息
    type = Column(
        Enum(DatasetType), nullable=False, default=DatasetType.TEXT,
        comment="数据集类型"
    )
    format = Column(String(64), nullable=True, comment="数据格式")
    version = Column(String(32), nullable=False, default="1.0.0", comment="版本号")

    # 文件信息
    file_path = Column(String(512), nullable=True, comment="数据集文件路径")
    file_size = Column(BigInteger, nullable=True, comment="文件大小（字节）")
    file_hash = Column(String(128), nullable=True, comment="文件哈希")
    compression = Column(String(32), nullable=True, comment="压缩格式")

    # 数据集统计
    sample_count = Column(BigInteger, nullable=True, comment="样本数量")
    feature_count = Column(Integer, nullable=True, comment="特征数量")
    class_count = Column(Integer, nullable=True, comment="类别数量")
    language = Column(String(32), nullable=True, comment="数据语言")
    avg_sample_length = Column(Float, nullable=True, comment="平均样本长度")

    # 配置
    config_json = Column(JSON, nullable=True, comment="数据集配置（JSON）")
    preprocessing_config = Column(
        JSON, nullable=True,
        comment="预处理配置（JSON）"
    )
    split_config = Column(JSON, nullable=True, comment="数据划分配置（JSON）")

    # 状态
    is_public = Column(Boolean, nullable=False, default=False, comment="是否公开")
    is_verified = Column(Boolean, nullable=False, default=False, comment="是否已验证")
    quality_score = Column(Float, nullable=True, comment="质量评分")

    # 标签
    category = Column(String(64), nullable=True, comment="分类")
    tags = Column(JSON, nullable=True, comment="标签（JSON数组）")
    license = Column(String(64), nullable=True, comment="数据许可证")
    source = Column(String(255), nullable=True, comment="数据来源")

    # 使用统计
    total_training_uses = Column(
        Integer, nullable=False, default=0,
        comment="训练使用次数"
    )

    # 时间戳
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        comment="创建时间"
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        onupdate=get_utc_now,
        comment="更新时间"
    )
    last_used_at = Column(
        DateTime(timezone=True), nullable=True,
        comment="最后使用时间"
    )

    # 索引
    __table_args__ = (
        Index("ix_datasets_user_id", "user_id"),
        Index("ix_datasets_type", "type"),
        Index("ix_datasets_category", "category"),
        Index("ix_datasets_is_public", "is_public"),
        Index("ix_datasets_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<Dataset(id={self.id}, name='{self.name}', type={self.type.value})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        """将数据集转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "user_id": self.user_id,
            "type": self.type.value if self.type else None,
            "format": self.format,
            "version": self.version,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "file_hash": self.file_hash,
            "compression": self.compression,
            "sample_count": self.sample_count,
            "feature_count": self.feature_count,
            "class_count": self.class_count,
            "language": self.language,
            "avg_sample_length": self.avg_sample_length,
            "config_json": self.config_json,
            "preprocessing_config": self.preprocessing_config,
            "split_config": self.split_config,
            "is_public": self.is_public,
            "is_verified": self.is_verified,
            "quality_score": self.quality_score,
            "category": self.category,
            "tags": self.tags,
            "license": self.license,
            "source": self.source,
            "total_training_uses": self.total_training_uses,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
        }


# ============================================================
# 人格系统模型
# ============================================================

class Personality(Base):  # type: ignore
    """
    人格模型

    存储智能体的人格定义，包括灵魂 Markdown 和行为配置。

    属性:
        id: 人格唯一标识符
        name: 人格名称
        user_id: 创建者用户 ID
        soul_md: 灵魂 Markdown 内容
        config_json: 人格配置（JSON）
        version: 人格版本
    """
    __tablename__ = "personalities"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="人格ID")
    name = Column(String(255), nullable=False, comment="人格名称")
    description = Column(Text, nullable=True, comment="人格描述")
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment="创建者用户ID"
    )

    # 人格定义
    soul_md = Column(Text, nullable=True, comment="灵魂Markdown内容")
    system_prompt = Column(Text, nullable=True, comment="系统提示词")
    greeting = Column(Text, nullable=True, comment="问候语")

    # 人格特征
    personality_traits = Column(
        JSON, nullable=True,
        comment="人格特征（JSON）"
    )
    speaking_style = Column(
        JSON, nullable=True,
        comment="说话风格（JSON）"
    )
    knowledge_areas = Column(
        JSON, nullable=True,
        comment="知识领域（JSON）"
    )
    emotional_profile = Column(
        JSON, nullable=True,
        comment="情感画像（JSON）"
    )

    # 配置
    config_json = Column(JSON, nullable=True, comment="人格配置（JSON）")
    behavior_rules = Column(JSON, nullable=True, comment="行为规则（JSON）")
    response_templates = Column(
        JSON, nullable=True,
        comment="响应模板（JSON）"
    )

    # 版本管理
    version = Column(Integer, nullable=False, default=1, comment="版本号")
    parent_id = Column(
        Integer, ForeignKey("personalities.id", ondelete="SET NULL"),
        nullable=True, comment="父人格ID"
    )
    changelog = Column(Text, nullable=True, comment="变更日志")

    # 状态
    is_public = Column(Boolean, nullable=False, default=False, comment="是否公开")
    is_template = Column(Boolean, nullable=False, default=False, comment="是否为模板")
    is_active = Column(Boolean, nullable=False, default=True, comment="是否活跃")
    category = Column(String(64), nullable=True, comment="分类")
    tags = Column(JSON, nullable=True, comment="标签（JSON数组）")
    avatar = Column(String(512), nullable=True, comment="头像URL")
    icon = Column(String(512), nullable=True, comment="图标URL")

    # 统计
    total_uses = Column(Integer, nullable=False, default=0, comment="总使用次数")
    avg_rating = Column(Float, nullable=True, comment="平均评分")

    # 时间戳
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        comment="创建时间"
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        onupdate=get_utc_now,
        comment="更新时间"
    )

    # 关系定义
    user = relationship("User", back_populates="personalities", lazy="select")
    parent = relationship("Personality", remote_side=[id], lazy="select")

    # 索引
    __table_args__ = (
        Index("ix_personalities_user_id", "user_id"),
        Index("ix_personalities_category", "category"),
        Index("ix_personalities_is_public", "is_public"),
        Index("ix_personalities_is_template", "is_template"),
        Index("ix_personalities_version", "version"),
        Index("ix_personalities_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<Personality(id={self.id}, name='{self.name}', "
            f"version={self.version})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        """将人格转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "user_id": self.user_id,
            "soul_md": self.soul_md,
            "system_prompt": self.system_prompt,
            "greeting": self.greeting,
            "personality_traits": self.personality_traits,
            "speaking_style": self.speaking_style,
            "knowledge_areas": self.knowledge_areas,
            "emotional_profile": self.emotional_profile,
            "config_json": self.config_json,
            "behavior_rules": self.behavior_rules,
            "response_templates": self.response_templates,
            "version": self.version,
            "parent_id": self.parent_id,
            "changelog": self.changelog,
            "is_public": self.is_public,
            "is_template": self.is_template,
            "is_active": self.is_active,
            "category": self.category,
            "tags": self.tags,
            "avatar": self.avatar,
            "icon": self.icon,
            "total_uses": self.total_uses,
            "avg_rating": self.avg_rating,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ============================================================
# 安全审计模型
# ============================================================

class AuditLog(Base):  # type: ignore
    """
    审计日志模型

    记录系统中所有重要的操作和事件，用于安全审计和追踪。

    属性:
        id: 日志唯一标识符
        user_id: 操作用户 ID
        action: 操作类型
        resource_type: 资源类型
        resource_id: 资源 ID
        details_json: 操作详情（JSON）
        ip_address: 操作者 IP 地址
        user_agent: 用户代理
        created_at: 操作时间
    """
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="审计日志ID")
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment="操作用户ID"
    )

    # 操作信息
    action = Column(
        Enum(AuditAction), nullable=False,
        comment="操作类型"
    )
    resource_type = Column(
        Enum(ResourceType), nullable=False,
        comment="资源类型"
    )
    resource_id = Column(Integer, nullable=True, comment="资源ID")
    resource_name = Column(String(255), nullable=True, comment="资源名称")

    # 操作详情
    details_json = Column(JSON, nullable=True, comment="操作详情（JSON）")
    old_values = Column(JSON, nullable=True, comment="变更前的值（JSON）")
    new_values = Column(JSON, nullable=True, comment="变更后的值（JSON）")

    # 请求信息
    ip_address = Column(String(45), nullable=True, comment="IP地址")
    ip_location = Column(String(255), nullable=True, comment="IP地理位置")
    user_agent = Column(String(512), nullable=True, comment="用户代理")
    request_method = Column(String(16), nullable=True, comment="请求方法")
    request_path = Column(String(512), nullable=True, comment="请求路径")
    request_id = Column(String(128), nullable=True, comment="请求ID")

    # 结果
    status = Column(String(32), nullable=False, default="success", comment="操作状态")
    error_message = Column(Text, nullable=True, comment="错误信息")
    duration_ms = Column(Integer, nullable=True, comment="操作耗时（毫秒）")

    # 会话信息
    session_id = Column(String(128), nullable=True, comment="会话ID")
    device_fingerprint = Column(String(128), nullable=True, comment="设备指纹")

    # 时间戳
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        comment="创建时间"
    )

    # 关系定义
    user = relationship("User", back_populates="audit_logs", lazy="select")

    # 索引
    __table_args__ = (
        Index("ix_audit_logs_user_id", "user_id"),
        Index("ix_audit_logs_action", "action"),
        Index("ix_audit_logs_resource_type", "resource_type"),
        Index("ix_audit_logs_resource_id", "resource_id"),
        Index("ix_audit_logs_status", "status"),
        Index("ix_audit_logs_ip_address", "ip_address"),
        Index("ix_audit_logs_created_at", "created_at"),
        Index("ix_audit_logs_request_id", "request_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<AuditLog(id={self.id}, action={self.action.value}, "
            f"resource_type={self.resource_type.value})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        """将审计日志转换为字典"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "action": self.action.value if self.action else None,
            "resource_type": self.resource_type.value if self.resource_type else None,
            "resource_id": self.resource_id,
            "resource_name": self.resource_name,
            "details_json": self.details_json,
            "old_values": self.old_values,
            "new_values": self.new_values,
            "ip_address": self.ip_address,
            "ip_location": self.ip_location,
            "user_agent": self.user_agent,
            "request_method": self.request_method,
            "request_path": self.request_path,
            "request_id": self.request_id,
            "status": self.status,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
            "session_id": self.session_id,
            "device_fingerprint": self.device_fingerprint,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ============================================================
# 系统设置模型
# ============================================================

class SystemSetting(Base):  # type: ignore
    """
    系统设置模型

    存储系统级别的配置项，使用 key-value 结构。

    属性:
        id: 设置唯一标识符
        key: 设置键名（唯一）
        value: 设置值（JSON 兼容）
        description: 设置描述
        category: 设置分类
        value_type: 值类型
        is_public: 是否公开（API 可读）
        is_editable: 是否可编辑
    """
    __tablename__ = "system_settings"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="设置ID")
    key = Column(
        String(128), unique=True, nullable=False,
        comment="设置键名"
    )
    value = Column(Text, nullable=False, comment="设置值")
    description = Column(Text, nullable=True, comment="设置描述")

    # 分类和类型
    category = Column(String(64), nullable=True, comment="设置分类")
    value_type = Column(
        String(32), nullable=False, default="string",
        comment="值类型（string/number/boolean/json）"
    )
    default_value = Column(Text, nullable=True, comment="默认值")

    # 访问控制
    is_public = Column(
        Boolean, nullable=False, default=False,
        comment="是否公开"
    )
    is_editable = Column(
        Boolean, nullable=False, default=True,
        comment="是否可编辑"
    )
    requires_restart = Column(
        Boolean, nullable=False, default=False,
        comment="修改后是否需要重启"
    )

    # 验证
    validation_rules = Column(
        JSON, nullable=True,
        comment="验证规则（JSON）"
    )
    allowed_values = Column(
        JSON, nullable=True,
        comment="允许的值列表（JSON）"
    )
    min_value = Column(Float, nullable=True, comment="最小值")
    max_value = Column(Float, nullable=True, comment="最大值")

    # 元数据
    metadata_json = Column(JSON, nullable=True, comment="元数据（JSON）")

    # 时间戳
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        comment="创建时间"
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        onupdate=get_utc_now,
        comment="更新时间"
    )

    # 索引
    __table_args__ = (
        Index("ix_system_settings_key", "key"),
        Index("ix_system_settings_category", "category"),
        Index("ix_system_settings_is_public", "is_public"),
    )

    def __repr__(self) -> str:
        return f"<SystemSetting(id={self.id}, key='{self.key}', value='{self.value}')>"

    def get_typed_value(self) -> Any:
        """
        获取类型转换后的值

        根据 value_type 将字符串值转换为对应的 Python 类型。

        返回:
            转换后的值
        """
        if self.value_type == "number":
            try:
                if "." in self.value:
                    return float(self.value)
                return int(self.value)
            except (ValueError, TypeError):
                return self.value
        elif self.value_type == "boolean":
            return self.value.lower() in ("true", "1", "yes", "on")
        elif self.value_type == "json":
            import json
            try:
                return json.loads(self.value)
            except (json.JSONDecodeError, TypeError):
                return self.value
        return self.value

    def to_dict(self) -> Dict[str, Any]:
        """将系统设置转换为字典"""
        return {
            "id": self.id,
            "key": self.key,
            "value": self.value,
            "typed_value": self.get_typed_value(),
            "description": self.description,
            "category": self.category,
            "value_type": self.value_type,
            "default_value": self.default_value,
            "is_public": self.is_public,
            "is_editable": self.is_editable,
            "requires_restart": self.requires_restart,
            "validation_rules": self.validation_rules,
            "allowed_values": self.allowed_values,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "metadata_json": self.metadata_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ============================================================
# 模型注册表（方便查找所有模型）
# ============================================================

# ============================================================
# 认知系统模型 (Cognitive)
# ============================================================

class Reflection(Base):  # type: ignore
    """
    反思记录模型
    
    存储AI系统的反思记录，用于自我改进。
    """
    __tablename__ = "reflections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String(64), nullable=False, index=True)
    reflection_type = Column(String(32), nullable=False, default="general")  # general, error, improvement
    content = Column(Text, nullable=False)
    insights = Column(JSON, nullable=True)  # 提取的洞察
    related_memories = Column(JSON, nullable=True)  # 相关记忆ID列表
    trigger_event = Column(Text, nullable=True)  # 触发反思的事件
    effectiveness_score = Column(Float, nullable=True)  # 有效性评分
    created_at = Column(DateTime(timezone=True), nullable=False, default=get_utc_now)

    __table_args__ = (
        Index("ix_reflections_agent_id", "agent_id"),
        Index("ix_reflections_type", "reflection_type"),
        Index("ix_reflections_created_at", "created_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "reflection_type": self.reflection_type,
            "content": self.content,
            "insights": self.insights,
            "related_memories": self.related_memories,
            "trigger_event": self.trigger_event,
            "effectiveness_score": self.effectiveness_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Memory(Base):  # type: ignore
    """
    记忆模型
    
    存储AI系统的长期记忆。
    """
    __tablename__ = "memories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String(64), nullable=False, index=True)
    memory_type = Column(String(32), nullable=False, default="episodic")  # episodic, semantic, procedural
    content = Column(Text, nullable=False)
    summary = Column(String(500), nullable=True)
    importance_score = Column(Float, nullable=False, default=0.5)
    access_count = Column(Integer, nullable=False, default=0)
    last_accessed_at = Column(DateTime(timezone=True), nullable=True)
    tags = Column(JSON, nullable=True)
    source = Column(String(128), nullable=True)
    related_entities = Column(JSON, nullable=True)
    embedding_vector = Column(LargeBinary, nullable=True)  # 向量嵌入
    created_at = Column(DateTime(timezone=True), nullable=False, default=get_utc_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=get_utc_now, onupdate=get_utc_now)

    __table_args__ = (
        Index("ix_memories_agent_id", "agent_id"),
        Index("ix_memories_type", "memory_type"),
        Index("ix_memories_importance", "importance_score"),
        Index("ix_memories_created_at", "created_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "memory_type": self.memory_type,
            "content": self.content,
            "summary": self.summary,
            "importance_score": self.importance_score,
            "access_count": self.access_count,
            "last_accessed_at": self.last_accessed_at.isoformat() if self.last_accessed_at else None,
            "tags": self.tags,
            "source": self.source,
            "related_entities": self.related_entities,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Goal(Base):  # type: ignore
    """
    目标模型
    
    存储AI系统的目标规划。
    """
    __tablename__ = "goals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String(64), nullable=False, index=True)
    parent_goal_id = Column(Integer, ForeignKey("goals.id"), nullable=True)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    goal_type = Column(String(32), nullable=False, default="task")  # task, project, learning
    priority = Column(Integer, nullable=False, default=3)  # 1-5
    status = Column(String(32), nullable=False, default="pending")  # pending, active, completed, failed, cancelled
    progress_percent = Column(Integer, nullable=False, default=0)
    deadline = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    sub_goals = Column(JSON, nullable=True)  # 子目标ID列表
    required_resources = Column(JSON, nullable=True)
    dependencies = Column(JSON, nullable=True)  # 依赖的其他目标ID
    created_at = Column(DateTime(timezone=True), nullable=False, default=get_utc_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=get_utc_now, onupdate=get_utc_now)

    __table_args__ = (
        Index("ix_goals_agent_id", "agent_id"),
        Index("ix_goals_status", "status"),
        Index("ix_goals_priority", "priority"),
        Index("ix_goals_deadline", "deadline"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "parent_goal_id": self.parent_goal_id,
            "title": self.title,
            "description": self.description,
            "goal_type": self.goal_type,
            "priority": self.priority,
            "status": self.status,
            "progress_percent": self.progress_percent,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "sub_goals": self.sub_goals,
            "required_resources": self.required_resources,
            "dependencies": self.dependencies,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ============================================================
# 编排系统模型 (Orchestration)
# ============================================================

class Strategy(Base):  # type: ignore
    """
    策略模型
    
    存储编排策略配置。
    """
    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    strategy_type = Column(String(32), nullable=False, default="routing")  # routing, load_balance, circuit_breaker
    config = Column(JSON, nullable=False)  # 策略配置
    rules = Column(JSON, nullable=True)  # 规则列表
    is_active = Column(Boolean, nullable=False, default=True)
    priority = Column(Integer, nullable=False, default=100)
    created_at = Column(DateTime(timezone=True), nullable=False, default=get_utc_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=get_utc_now, onupdate=get_utc_now)

    __table_args__ = (
        Index("ix_strategies_type", "strategy_type"),
        Index("ix_strategies_active", "is_active"),
        Index("ix_strategies_priority", "priority"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "strategy_type": self.strategy_type,
            "config": self.config,
            "rules": self.rules,
            "is_active": self.is_active,
            "priority": self.priority,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class RoutingRule(Base):  # type: ignore
    """
    路由规则模型
    
    存储请求路由规则。
    """
    __tablename__ = "routing_rules"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    priority = Column(Integer, nullable=False, default=100)
    condition_type = Column(String(32), nullable=False, default="path")  # path, header, query, body
    condition_value = Column(String(500), nullable=False)  # 条件值，如正则表达式
    target_service = Column(String(100), nullable=False)  # 目标服务
    target_endpoint = Column(String(200), nullable=True)  # 目标端点
    transform_config = Column(JSON, nullable=True)  # 请求转换配置
    is_active = Column(Boolean, nullable=False, default=True)
    match_count = Column(Integer, nullable=False, default=0)
    last_matched_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=get_utc_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=get_utc_now, onupdate=get_utc_now)

    __table_args__ = (
        Index("ix_routing_rules_priority", "priority"),
        Index("ix_routing_rules_active", "is_active"),
        Index("ix_routing_rules_condition", "condition_type"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "priority": self.priority,
            "condition_type": self.condition_type,
            "condition_value": self.condition_value,
            "target_service": self.target_service,
            "target_endpoint": self.target_endpoint,
            "transform_config": self.transform_config,
            "is_active": self.is_active,
            "match_count": self.match_count,
            "last_matched_at": self.last_matched_at.isoformat() if self.last_matched_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class LoadBalancer(Base):  # type: ignore
    """
    负载均衡器模型
    
    存储负载均衡配置。
    """
    __tablename__ = "load_balancers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    algorithm = Column(String(32), nullable=False, default="round_robin")  # round_robin, least_conn, ip_hash, weighted
    backend_services = Column(JSON, nullable=False)  # 后端服务列表
    health_check_config = Column(JSON, nullable=True)  # 健康检查配置
    is_active = Column(Boolean, nullable=False, default=True)
    total_requests = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=get_utc_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=get_utc_now, onupdate=get_utc_now)

    __table_args__ = (
        Index("ix_load_balancers_active", "is_active"),
        Index("ix_load_balancers_algorithm", "algorithm"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "algorithm": self.algorithm,
            "backend_services": self.backend_services,
            "health_check_config": self.health_check_config,
            "is_active": self.is_active,
            "total_requests": self.total_requests,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class CircuitBreaker(Base):  # type: ignore
    """
    熔断器模型
    
    存储熔断器配置和状态。
    """
    __tablename__ = "circuit_breakers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    target_service = Column(String(100), nullable=False)
    failure_threshold = Column(Integer, nullable=False, default=5)
    recovery_timeout = Column(Integer, nullable=False, default=60)  # 秒
    half_open_max_calls = Column(Integer, nullable=False, default=3)
    state = Column(String(32), nullable=False, default="closed")  # closed, open, half_open
    failure_count = Column(Integer, nullable=False, default=0)
    last_failure_at = Column(DateTime(timezone=True), nullable=True)
    last_state_change_at = Column(DateTime(timezone=True), nullable=True)
    total_failures = Column(Integer, nullable=False, default=0)
    total_successes = Column(Integer, nullable=False, default=0)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=get_utc_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=get_utc_now, onupdate=get_utc_now)

    __table_args__ = (
        Index("ix_circuit_breakers_service", "target_service"),
        Index("ix_circuit_breakers_state", "state"),
        Index("ix_circuit_breakers_active", "is_active"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "target_service": self.target_service,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "half_open_max_calls": self.half_open_max_calls,
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_at": self.last_failure_at.isoformat() if self.last_failure_at else None,
            "last_state_change_at": self.last_state_change_at.isoformat() if self.last_state_change_at else None,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ============================================================
# 智能体日志和任务模型 (Agent)
# ============================================================

class AgentLog(Base):  # type: ignore
    """
    智能体日志模型
    """
    __tablename__ = "agent_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String(64), nullable=False, index=True)
    log_level = Column(String(16), nullable=False, default="info")  # debug, info, warning, error, critical
    message = Column(Text, nullable=False)
    source = Column(String(128), nullable=True)  # 日志来源模块
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=get_utc_now)

    __table_args__ = (
        Index("ix_agent_logs_agent_id", "agent_id"),
        Index("ix_agent_logs_level", "log_level"),
        Index("ix_agent_logs_created_at", "created_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "log_level": self.log_level,
            "message": self.message,
            "source": self.source,
            "metadata": self.metadata_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AgentTask(Base):  # type: ignore
    """
    智能体任务模型
    """
    __tablename__ = "agent_tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String(64), nullable=False, index=True)
    task_type = Column(String(64), nullable=False)
    task_input = Column(JSON, nullable=False)
    task_output = Column(JSON, nullable=True)
    status = Column(String(32), nullable=False, default="pending")  # pending, running, completed, failed, cancelled
    priority = Column(String(16), nullable=False, default="medium")  # low, medium, high, critical
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_ms = Column(Integer, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=get_utc_now)

    __table_args__ = (
        Index("ix_agent_tasks_agent_id", "agent_id"),
        Index("ix_agent_tasks_status", "status"),
        Index("ix_agent_tasks_priority", "priority"),
        Index("ix_agent_tasks_created_at", "created_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "task_type": self.task_type,
            "task_input": self.task_input,
            "task_output": self.task_output,
            "status": self.status,
            "priority": self.priority,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "tokens_used": self.tokens_used,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Debate(Base):  # type: ignore
    """
    辩论模型
    """
    __tablename__ = "debates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    topic = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    debate_type = Column(String(32), nullable=False, default="proposition")  # proposition, comparison, brainstorming
    status = Column(String(32), nullable=False, default="pending")  # pending, active, paused, completed, cancelled
    participant_ids = Column(JSON, nullable=False)  # 参与者Agent ID列表
    rounds = Column(Integer, nullable=False, default=3)
    current_round = Column(Integer, nullable=False, default=0)
    arguments = Column(JSON, nullable=True)  # 论点列表
    votes = Column(JSON, nullable=True)  # 投票结果
    winner_id = Column(String(64), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=get_utc_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=get_utc_now, onupdate=get_utc_now)

    __table_args__ = (
        Index("ix_debates_status", "status"),
        Index("ix_debates_type", "debate_type"),
        Index("ix_debates_created_at", "created_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "description": self.description,
            "debate_type": self.debate_type,
            "status": self.status,
            "participant_ids": self.participant_ids,
            "rounds": self.rounds,
            "current_round": self.current_round,
            "arguments": self.arguments,
            "votes": self.votes,
            "winner_id": self.winner_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Marketplace(Base):  # type: ignore
    """
    市场模型
    
    存储智能体和插件市场条目。
    """
    __tablename__ = "marketplace"

    id = Column(Integer, primary_key=True, autoincrement=True)
    item_type = Column(String(32), nullable=False, default="agent")  # agent, plugin, skill, model
    item_id = Column(String(64), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    author_id = Column(String(64), nullable=False)
    author_name = Column(String(100), nullable=True)
    version = Column(String(32), nullable=False, default="1.0.0")
    tags = Column(JSON, nullable=True)
    capabilities = Column(JSON, nullable=True)
    rating = Column(Float, nullable=False, default=0.0)
    rating_count = Column(Integer, nullable=False, default=0)
    download_count = Column(Integer, nullable=False, default=0)
    price = Column(Float, nullable=False, default=0.0)  # 0 = free
    is_public = Column(Boolean, nullable=False, default=True)
    is_verified = Column(Boolean, nullable=False, default=False)
    config = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=get_utc_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=get_utc_now, onupdate=get_utc_now)

    __table_args__ = (
        Index("ix_marketplace_type", "item_type"),
        Index("ix_marketplace_item_id", "item_id"),
        Index("ix_marketplace_author", "author_id"),
        Index("ix_marketplace_rating", "rating"),
        Index("ix_marketplace_public", "is_public"),
        Index("ix_marketplace_verified", "is_verified"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "item_type": self.item_type,
            "item_id": self.item_id,
            "name": self.name,
            "description": self.description,
            "author_id": self.author_id,
            "author_name": self.author_name,
            "version": self.version,
            "tags": self.tags,
            "capabilities": self.capabilities,
            "rating": self.rating,
            "rating_count": self.rating_count,
            "download_count": self.download_count,
            "price": self.price,
            "is_public": self.is_public,
            "is_verified": self.is_verified,
            "config": self.config,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ============================================================
# 训练日志和超参数搜索模型 (Training)
# ============================================================

class TrainingLog(Base):  # type: ignore
    """
    训练日志模型
    """
    __tablename__ = "training_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    training_job_id = Column(String(64), nullable=False, index=True)
    log_level = Column(String(16), nullable=False, default="info")
    message = Column(Text, nullable=False)
    step = Column(Integer, nullable=True)
    epoch = Column(Integer, nullable=True)
    metrics = Column(JSON, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=get_utc_now)

    __table_args__ = (
        Index("ix_training_logs_job_id", "training_job_id"),
        Index("ix_training_logs_level", "log_level"),
        Index("ix_training_logs_step", "step"),
        Index("ix_training_logs_created_at", "created_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "training_job_id": self.training_job_id,
            "log_level": self.log_level,
            "message": self.message,
            "step": self.step,
            "epoch": self.epoch,
            "metrics": self.metrics,
            "metadata": self.metadata_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class HPSearch(Base):  # type: ignore
    """
    超参数搜索模型
    """
    __tablename__ = "hp_searches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    training_job_id = Column(String(64), nullable=False, index=True)
    search_algorithm = Column(String(32), nullable=False, default="grid")  # grid, random, bayesian
    search_space = Column(JSON, nullable=False)  # 搜索空间定义
    max_trials = Column(Integer, nullable=False, default=10)
    completed_trials = Column(Integer, nullable=False, default=0)
    best_trial_id = Column(Integer, nullable=True)
    best_params = Column(JSON, nullable=True)
    best_score = Column(Float, nullable=True)
    status = Column(String(32), nullable=False, default="pending")  # pending, running, completed, failed
    trials = Column(JSON, nullable=True)  # 所有trial结果
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=get_utc_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=get_utc_now, onupdate=get_utc_now)

    __table_args__ = (
        Index("ix_hp_searches_job_id", "training_job_id"),
        Index("ix_hp_searches_status", "status"),
        Index("ix_hp_searches_algorithm", "search_algorithm"),
        Index("ix_hp_searches_created_at", "created_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "training_job_id": self.training_job_id,
            "search_algorithm": self.search_algorithm,
            "search_space": self.search_space,
            "max_trials": self.max_trials,
            "completed_trials": self.completed_trials,
            "best_trial_id": self.best_trial_id,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "status": self.status,
            "trials": self.trials,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ============================================================
# 角色权限管理模型
# ============================================================

class Role(Base):  # type: ignore
    """
    角色模型

    存储系统角色信息，用于权限管理。

    属性:
        id: 角色唯一标识符
        name: 角色名称（唯一）
        description: 角色描述
        permissions: 权限列表（JSON）
        created_at: 创建时间
        updated_at: 更新时间
    """
    __tablename__ = "roles"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="角色ID")
    name = Column(
        String(64), unique=True, nullable=False,
        comment="角色名称"
    )
    description = Column(Text, nullable=True, comment="角色描述")
    permissions = Column(
        JSON, nullable=False, default=list,
        comment="权限列表（JSON数组）"
    )

    # 时间戳
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        comment="创建时间"
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        onupdate=get_utc_now,
        comment="更新时间"
    )

    __table_args__ = (
        Index("ix_roles_name", "name"),
        Index("ix_roles_created_at", "created_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """将角色转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "permissions": self.permissions,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Permission(Base):  # type: ignore
    """
    权限模型

    存储系统权限定义。

    属性:
        id: 权限唯一标识符
        name: 权限名称（唯一）
        code: 权限代码（唯一）
        description: 权限描述
        created_at: 创建时间
    """
    __tablename__ = "permissions"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="权限ID")
    name = Column(
        String(64), unique=True, nullable=False,
        comment="权限名称"
    )
    code = Column(
        String(64), unique=True, nullable=False,
        comment="权限代码"
    )
    description = Column(Text, nullable=True, comment="权限描述")

    # 时间戳
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        comment="创建时间"
    )

    __table_args__ = (
        Index("ix_permissions_name", "name"),
        Index("ix_permissions_code", "code"),
        Index("ix_permissions_created_at", "created_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """将权限转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "code": self.code,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ============================================================
# API密钥管理模型
# ============================================================

class APIKey(Base):  # type: ignore
    """
    API密钥模型

    存储用户API密钥信息。

    属性:
        id: 密钥唯一标识符
        user_id: 所属用户ID
        name: 密钥名称
        key_hash: 密钥哈希值
        permissions: 权限列表（JSON）
        is_active: 是否激活
        last_used_at: 最后使用时间
        created_at: 创建时间
    """
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="密钥ID")
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, comment="所属用户ID"
    )
    name = Column(
        String(128), nullable=False,
        comment="密钥名称"
    )
    key_hash = Column(
        String(255), nullable=False,
        comment="密钥哈希值"
    )
    permissions = Column(
        JSON, nullable=False, default=list,
        comment="权限列表（JSON数组）"
    )

    # 状态
    is_active = Column(
        Boolean, nullable=False, default=True,
        comment="是否激活"
    )

    # 使用信息
    last_used_at = Column(
        DateTime(timezone=True), nullable=True,
        comment="最后使用时间"
    )
    usage_count = Column(
        Integer, nullable=False, default=0,
        comment="使用次数"
    )

    # 时间戳
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        comment="创建时间"
    )

    # 关系
    user = relationship("User", backref="api_keys")

    __table_args__ = (
        Index("ix_api_keys_user_id", "user_id"),
        Index("ix_api_keys_key_hash", "key_hash"),
        Index("ix_api_keys_is_active", "is_active"),
        Index("ix_api_keys_created_at", "created_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """将API密钥转换为字典"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "permissions": self.permissions,
            "is_active": self.is_active,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "usage_count": self.usage_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ============================================================
# 备份管理模型
# ============================================================

class Backup(Base):  # type: ignore
    """
    备份记录模型

    存储系统备份记录。

    属性:
        id: 备份唯一标识符
        name: 备份名称
        size_bytes: 备份大小（字节）
        status: 备份状态
        path: 备份文件路径
        created_at: 创建时间
        completed_at: 完成时间
    """
    __tablename__ = "backups"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="备份ID")
    name = Column(
        String(128), nullable=False,
        comment="备份名称"
    )
    size_bytes = Column(
        BigInteger, nullable=False, default=0,
        comment="备份大小（字节）"
    )
    status = Column(
        String(32), nullable=False, default="pending",
        comment="备份状态：pending, in_progress, completed, failed"
    )
    path = Column(
        String(512), nullable=True,
        comment="备份文件路径"
    )
    description = Column(Text, nullable=True, comment="备份描述")

    # 时间戳
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        comment="创建时间"
    )
    completed_at = Column(
        DateTime(timezone=True), nullable=True,
        comment="完成时间"
    )

    __table_args__ = (
        Index("ix_backups_name", "name"),
        Index("ix_backups_status", "status"),
        Index("ix_backups_created_at", "created_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """将备份记录转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "size_bytes": self.size_bytes,
            "status": self.status,
            "path": self.path,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


# ============================================================
# 仪表盘管理模型
# ============================================================

class Dashboard(Base):  # type: ignore
    """
    仪表盘配置模型

    存储用户仪表盘配置。

    属性:
        id: 仪表盘唯一标识符
        user_id: 所属用户ID
        name: 仪表盘名称
        layout: 布局配置（JSON）
        widgets: 组件配置（JSON）
        is_default: 是否为默认仪表盘
        created_at: 创建时间
    """
    __tablename__ = "dashboards"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="仪表盘ID")
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, comment="所属用户ID"
    )
    name = Column(
        String(128), nullable=False,
        comment="仪表盘名称"
    )
    layout = Column(
        JSON, nullable=False, default=dict,
        comment="布局配置（JSON）"
    )
    widgets = Column(
        JSON, nullable=False, default=list,
        comment="组件配置（JSON数组）"
    )

    # 状态
    is_default = Column(
        Boolean, nullable=False, default=False,
        comment="是否为默认仪表盘"
    )

    # 时间戳
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        comment="创建时间"
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        onupdate=get_utc_now,
        comment="更新时间"
    )

    # 关系
    user = relationship("User", backref="dashboards")

    __table_args__ = (
        Index("ix_dashboards_user_id", "user_id"),
        Index("ix_dashboards_is_default", "is_default"),
        Index("ix_dashboards_created_at", "created_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """将仪表盘转换为字典"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "layout": self.layout,
            "widgets": self.widgets,
            "is_default": self.is_default,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ============================================================
# 告警管理模型
# ============================================================

class Alert(Base):  # type: ignore
    """
    告警记录模型

    存储系统告警记录。

    属性:
        id: 告警唯一标识符
        severity: 严重级别
        title: 告警标题
        message: 告警消息
        source: 告警来源
        status: 告警状态
        acknowledged_by: 确认人用户ID
        created_at: 创建时间
    """
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="告警ID")
    severity = Column(
        String(32), nullable=False,
        comment="严重级别：critical, high, medium, low, info"
    )
    title = Column(
        String(255), nullable=False,
        comment="告警标题"
    )
    message = Column(Text, nullable=False, comment="告警消息")
    source = Column(
        String(128), nullable=False,
        comment="告警来源"
    )
    status = Column(
        String(32), nullable=False, default="active",
        comment="告警状态：active, acknowledged, resolved"
    )

    # 确认信息
    acknowledged_by = Column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True, comment="确认人用户ID"
    )
    acknowledged_at = Column(
        DateTime(timezone=True), nullable=True,
        comment="确认时间"
    )

    # 元数据
    metadata_json = Column(
        JSON, nullable=True,
        comment="告警元数据（JSON）"
    )

    # 时间戳
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        comment="创建时间"
    )

    # 关系
    acknowledged_user = relationship("User", foreign_keys=[acknowledged_by])

    __table_args__ = (
        Index("ix_alerts_severity", "severity"),
        Index("ix_alerts_status", "status"),
        Index("ix_alerts_source", "source"),
        Index("ix_alerts_acknowledged_by", "acknowledged_by"),
        Index("ix_alerts_created_at", "created_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """将告警转换为字典"""
        return {
            "id": self.id,
            "severity": self.severity,
            "title": self.title,
            "message": self.message,
            "source": self.source,
            "status": self.status,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "metadata": self.metadata_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ============================================================
# 许可证管理模型
# ============================================================

class License(Base):  # type: ignore
    """
    许可证信息模型

    存储系统许可证信息。

    属性:
        id: 许可证唯一标识符
        key: 许可证密钥
        type: 许可证类型
        expires_at: 过期时间
        features: 功能列表（JSON）
        is_active: 是否激活
        created_at: 创建时间
    """
    __tablename__ = "licenses"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="许可证ID")
    key = Column(
        String(255), unique=True, nullable=False,
        comment="许可证密钥"
    )
    type = Column(
        String(64), nullable=False,
        comment="许可证类型"
    )
    expires_at = Column(
        DateTime(timezone=True), nullable=True,
        comment="过期时间"
    )
    features = Column(
        JSON, nullable=False, default=list,
        comment="功能列表（JSON数组）"
    )

    # 状态
    is_active = Column(
        Boolean, nullable=False, default=True,
        comment="是否激活"
    )

    # 元数据
    metadata_json = Column(
        JSON, nullable=True,
        comment="许可证元数据（JSON）"
    )

    # 时间戳
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        comment="创建时间"
    )

    __table_args__ = (
        Index("ix_licenses_key", "key"),
        Index("ix_licenses_type", "type"),
        Index("ix_licenses_is_active", "is_active"),
        Index("ix_licenses_expires_at", "expires_at"),
        Index("ix_licenses_created_at", "created_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """将许可证转换为字典"""
        return {
            "id": self.id,
            "key": self.key,
            "type": self.type,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "features": self.features,
            "is_active": self.is_active,
            "metadata": self.metadata_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ============================================================
# 帮助文档管理模型
# ============================================================

class HelpDoc(Base):  # type: ignore
    """
    帮助文档模型

    存储系统帮助文档。

    属性:
        id: 文档唯一标识符
        title: 文档标题
        content: 文档内容
        category: 文档分类
        tags: 标签列表（JSON）
        view_count: 查看次数
        created_at: 创建时间
    """
    __tablename__ = "help_docs"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="文档ID")
    title = Column(
        String(255), nullable=False,
        comment="文档标题"
    )
    content = Column(Text, nullable=False, comment="文档内容")
    category = Column(
        String(64), nullable=True,
        comment="文档分类"
    )
    tags = Column(
        JSON, nullable=False, default=list,
        comment="标签列表（JSON数组）"
    )

    # 统计
    view_count = Column(
        Integer, nullable=False, default=0,
        comment="查看次数"
    )

    # 时间戳
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        comment="创建时间"
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        onupdate=get_utc_now,
        comment="更新时间"
    )

    __table_args__ = (
        Index("ix_help_docs_title", "title"),
        Index("ix_help_docs_category", "category"),
        Index("ix_help_docs_created_at", "created_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """将帮助文档转换为字典"""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "category": self.category,
            "tags": self.tags,
            "view_count": self.view_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ============================================================
# FAQ管理模型
# ============================================================

class FAQ(Base):  # type: ignore
    """
    FAQ模型

    存储常见问题解答。

    属性:
        id: FAQ唯一标识符
        question: 问题
        answer: 答案
        category: 分类
        helpful_count: 有帮助计数
        created_at: 创建时间
    """
    __tablename__ = "faqs"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="FAQ ID")
    question = Column(
        String(512), nullable=False,
        comment="问题"
    )
    answer = Column(Text, nullable=False, comment="答案")
    category = Column(
        String(64), nullable=True,
        comment="分类"
    )

    # 统计
    helpful_count = Column(
        Integer, nullable=False, default=0,
        comment="有帮助计数"
    )
    not_helpful_count = Column(
        Integer, nullable=False, default=0,
        comment="无帮助计数"
    )

    # 时间戳
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        comment="创建时间"
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=get_utc_now,
        onupdate=get_utc_now,
        comment="更新时间"
    )

    __table_args__ = (
        Index("ix_faqs_question", "question"),
        Index("ix_faqs_category", "category"),
        Index("ix_faqs_created_at", "created_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """将FAQ转换为字典"""
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "category": self.category,
            "helpful_count": self.helpful_count,
            "not_helpful_count": self.not_helpful_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ============================================================
# 模型注册表（方便查找所有模型）
# ============================================================

ALL_MODELS: List[Any] = [
    User,
    UserSettings,
    Conversation,
    Message,
    Model,
    ModelLoadBalance,
    TrainingJob,
    Checkpoint,
    GeneratedContent,
    Workflow,
    WorkflowExecution,
    Agent,
    Alliance,
    Plugin,
    Channel,
    Dataset,
    Personality,
    AuditLog,
    SystemSetting,
    # 新增模型
    Reflection,
    Memory,
    Goal,
    Strategy,
    RoutingRule,
    LoadBalancer,
    CircuitBreaker,
    AgentLog,
    AgentTask,
    Debate,
    Marketplace,
    TrainingLog,
    HPSearch,
    # 系统管理模型
    Role,
    Permission,
    APIKey,
    Backup,
    Dashboard,
    Alert,
    License,
    HelpDoc,
    FAQ,
]

"""所有 ORM 模型列表，用于批量操作"""
