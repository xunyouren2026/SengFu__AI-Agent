"""
AGI Unified Framework - 种子数据模块

本模块提供数据库预填充数据，包括：
- 默认系统设置（20+ 项）
- 默认模型配置（AGI-Ultra, AGI-Sage 等）
- 默认插件列表
- 示例工作流模板
- 示例人格模板
- 示例 FAQ 数据

设计原则:
    1. 所有种子数据都是幂等的（重复插入不会出错）
    2. 使用 upsert 模式（存在则跳过，不存在则插入）
    3. 数据分类管理，支持选择性填充
    4. 提供便捷的种子数据管理器

依赖:
    - sqlalchemy >= 1.4 (可选，优雅降级)
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# 尝试导入 SQLAlchemy
try:
    from sqlalchemy import or_
    from sqlalchemy.orm import Session
    from sqlalchemy.exc import SQLAlchemyError, IntegrityError

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logger.warning("SQLAlchemy 未安装。种子数据功能将不可用。")

# 导入模型
try:
    from database.models import (
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
        UserRole,
        ModelProvider,
        ModelType,
        ModelStatus,
        ContentType,
        WorkflowStatus,
        AgentStatus,
        PluginStatus,
        ChannelType,
        ChannelStatus,
        DatasetType,
        LoadBalanceStrategy,
    )
    from database.utils import hash_password
except ImportError:
    pass


# ============================================================
# 默认系统设置数据
# ============================================================

DEFAULT_SYSTEM_SETTINGS: List[Dict[str, Any]] = [
    # 基本设置
    {
        "key": "site_name",
        "value": "AGI Unified Framework",
        "description": "站点名称",
        "category": "general",
        "value_type": "string",
        "is_public": True,
        "is_editable": True,
    },
    {
        "key": "site_description",
        "value": "通用人工智能统一框架 - 融合对话、训练、生成、智能体等能力",
        "description": "站点描述",
        "category": "general",
        "value_type": "string",
        "is_public": True,
        "is_editable": True,
    },
    {
        "key": "site_url",
        "value": "http://localhost:8000",
        "description": "站点 URL",
        "category": "general",
        "value_type": "string",
        "is_public": True,
        "is_editable": True,
    },
    {
        "key": "site_version",
        "value": "1.0.0",
        "description": "系统版本号",
        "category": "general",
        "value_type": "string",
        "is_public": True,
        "is_editable": False,
    },
    {
        "key": "maintenance_mode",
        "value": "false",
        "description": "维护模式（启用后仅管理员可访问）",
        "category": "general",
        "value_type": "boolean",
        "is_public": True,
        "is_editable": True,
        "requires_restart": False,
    },
    # 安全设置
    {
        "key": "max_login_attempts",
        "value": "5",
        "description": "最大登录尝试次数",
        "category": "security",
        "value_type": "number",
        "is_public": False,
        "is_editable": True,
        "min_value": 1,
        "max_value": 20,
    },
    {
        "key": "lockout_duration",
        "value": "30",
        "description": "账户锁定时长（分钟）",
        "category": "security",
        "value_type": "number",
        "is_public": False,
        "is_editable": True,
        "min_value": 1,
        "max_value": 1440,
    },
    {
        "key": "session_timeout",
        "value": "3600",
        "description": "会话超时时间（秒）",
        "category": "security",
        "value_type": "number",
        "is_public": False,
        "is_editable": True,
        "min_value": 300,
        "max_value": 86400,
    },
    {
        "key": "api_key_expiry_days",
        "value": "365",
        "description": "API 密钥有效期（天）",
        "category": "security",
        "value_type": "number",
        "is_public": False,
        "is_editable": True,
    },
    {
        "key": "password_min_length",
        "value": "8",
        "description": "密码最小长度",
        "category": "security",
        "value_type": "number",
        "is_public": False,
        "is_editable": True,
        "min_value": 6,
        "max_value": 128,
    },
    {
        "key": "enable_registration",
        "value": "true",
        "description": "是否允许用户注册",
        "category": "security",
        "value_type": "boolean",
        "is_public": True,
        "is_editable": True,
    },
    {
        "key": "enable_captcha",
        "value": "false",
        "description": "是否启用验证码",
        "category": "security",
        "value_type": "boolean",
        "is_public": False,
        "is_editable": True,
    },
    # 对话设置
    {
        "key": "default_max_tokens",
        "value": "4096",
        "description": "默认最大 token 数",
        "category": "conversation",
        "value_type": "number",
        "is_public": True,
        "is_editable": True,
        "min_value": 64,
        "max_value": 128000,
    },
    {
        "key": "default_temperature",
        "value": "0.7",
        "description": "默认温度参数",
        "category": "conversation",
        "value_type": "number",
        "is_public": True,
        "is_editable": True,
        "min_value": 0.0,
        "max_value": 2.0,
    },
    {
        "key": "default_top_p",
        "value": "0.9",
        "description": "默认 top_p 参数",
        "category": "conversation",
        "value_type": "number",
        "is_public": True,
        "is_editable": True,
        "min_value": 0.0,
        "max_value": 1.0,
    },
    {
        "key": "enable_streaming",
        "value": "true",
        "description": "是否默认启用流式响应",
        "category": "conversation",
        "value_type": "boolean",
        "is_public": True,
        "is_editable": True,
    },
    {
        "key": "max_conversation_history",
        "value": "50",
        "description": "最大对话历史消息数",
        "category": "conversation",
        "value_type": "number",
        "is_public": False,
        "is_editable": True,
        "min_value": 1,
        "max_value": 200,
    },
    {
        "key": "enable_context_compression",
        "value": "true",
        "description": "是否启用上下文压缩",
        "category": "conversation",
        "value_type": "boolean",
        "is_public": False,
        "is_editable": True,
    },
    # 生成设置
    {
        "key": "default_image_width",
        "value": "1024",
        "description": "默认图像宽度",
        "category": "generation",
        "value_type": "number",
        "is_public": True,
        "is_editable": True,
    },
    {
        "key": "default_image_height",
        "value": "1024",
        "description": "默认图像高度",
        "category": "generation",
        "value_type": "number",
        "is_public": True,
        "is_editable": True,
    },
    {
        "key": "max_image_size",
        "value": "2048",
        "description": "最大图像尺寸",
        "category": "generation",
        "value_type": "number",
        "is_public": True,
        "is_editable": True,
    },
    {
        "key": "default_video_fps",
        "value": "24",
        "description": "默认视频帧率",
        "category": "generation",
        "value_type": "number",
        "is_public": True,
        "is_editable": True,
    },
    {
        "key": "max_video_duration",
        "value": "60",
        "description": "最大视频时长（秒）",
        "category": "generation",
        "value_type": "number",
        "is_public": True,
        "is_editable": True,
    },
    # 训练设置
    {
        "key": "max_training_jobs_per_user",
        "value": "3",
        "description": "每用户最大并行训练任务数",
        "category": "training",
        "value_type": "number",
        "is_public": True,
        "is_editable": True,
    },
    {
        "key": "default_training_epochs",
        "value": "10",
        "description": "默认训练轮次",
        "category": "training",
        "value_type": "number",
        "is_public": True,
        "is_editable": True,
    },
    {
        "key": "default_batch_size",
        "value": "32",
        "description": "默认批次大小",
        "category": "training",
        "value_type": "number",
        "is_public": True,
        "is_editable": True,
    },
    {
        "key": "default_learning_rate",
        "value": "0.001",
        "description": "默认学习率",
        "category": "training",
        "value_type": "number",
        "is_public": True,
        "is_editable": True,
    },
    {
        "key": "checkpoint_save_interval",
        "value": "5",
        "description": "检查点保存间隔（轮次）",
        "category": "training",
        "value_type": "number",
        "is_public": False,
        "is_editable": True,
    },
    # 存储设置
    {
        "key": "storage_backend",
        "value": "local",
        "description": "存储后端（local/s3/oss）",
        "category": "storage",
        "value_type": "string",
        "is_public": False,
        "is_editable": True,
        "allowed_values": ["local", "s3", "oss", "gcs"],
    },
    {
        "key": "storage_path",
        "value": "./data/storage",
        "description": "本地存储路径",
        "category": "storage",
        "value_type": "string",
        "is_public": False,
        "is_editable": True,
    },
    {
        "key": "max_file_size_mb",
        "value": "100",
        "description": "最大文件大小（MB）",
        "category": "storage",
        "value_type": "number",
        "is_public": True,
        "is_editable": True,
    },
    {
        "key": "max_storage_per_user_mb",
        "value": "1024",
        "description": "每用户最大存储空间（MB）",
        "category": "storage",
        "value_type": "number",
        "is_public": True,
        "is_editable": True,
    },
    # 通知设置
    {
        "key": "enable_email_notifications",
        "value": "false",
        "description": "是否启用邮件通知",
        "category": "notification",
        "value_type": "boolean",
        "is_public": False,
        "is_editable": True,
    },
    {
        "key": "smtp_host",
        "value": "",
        "description": "SMTP 服务器地址",
        "category": "notification",
        "value_type": "string",
        "is_public": False,
        "is_editable": True,
    },
    {
        "key": "smtp_port",
        "value": "587",
        "description": "SMTP 服务器端口",
        "category": "notification",
        "value_type": "number",
        "is_public": False,
        "is_editable": True,
    },
    {
        "key": "notification_email_from",
        "value": "noreply@agi-framework.local",
        "description": "通知发件人邮箱",
        "category": "notification",
        "value_type": "string",
        "is_public": False,
        "is_editable": True,
    },
]


# ============================================================
# 默认模型配置数据
# ============================================================

DEFAULT_MODELS: List[Dict[str, Any]] = [
    {
        "name": "agi-ultra",
        "display_name": "AGI-Ultra",
        "description": "AGI 框架旗舰模型，具备最强推理和创作能力",
        "provider": ModelProvider.LOCAL,
        "model_type": ModelType.LLM,
        "version": "1.0",
        "max_tokens": 128000,
        "max_context_length": 128000,
        "supports_streaming": True,
        "supports_function_calling": True,
        "supports_vision": True,
        "supports_json_mode": True,
        "cost_per_1k_input_tokens": 0.0,
        "cost_per_1k_output_tokens": 0.0,
        "status": ModelStatus.ACTIVE,
        "is_default": True,
        "priority": 100,
        "capabilities": ["reasoning", "coding", "creative_writing", "analysis", "math"],
        "tags": ["flagship", "recommended", "multimodal"],
    },
    {
        "name": "agi-sage",
        "display_name": "AGI-Sage",
        "description": "AGI 框架智能助手模型，平衡性能与效率",
        "provider": ModelProvider.LOCAL,
        "model_type": ModelType.LLM,
        "version": "1.0",
        "max_tokens": 65536,
        "max_context_length": 65536,
        "supports_streaming": True,
        "supports_function_calling": True,
        "supports_vision": False,
        "supports_json_mode": True,
        "cost_per_1k_input_tokens": 0.0,
        "cost_per_1k_output_tokens": 0.0,
        "status": ModelStatus.ACTIVE,
        "is_default": False,
        "priority": 80,
        "capabilities": ["reasoning", "coding", "analysis", "summarization"],
        "tags": ["balanced", "efficient"],
    },
    {
        "name": "agi-swift",
        "display_name": "AGI-Swift",
        "description": "AGI 框架快速响应模型，适合实时对话场景",
        "provider": ModelProvider.LOCAL,
        "model_type": ModelType.CHAT,
        "version": "1.0",
        "max_tokens": 16384,
        "max_context_length": 16384,
        "supports_streaming": True,
        "supports_function_calling": False,
        "supports_vision": False,
        "supports_json_mode": False,
        "cost_per_1k_input_tokens": 0.0,
        "cost_per_1k_output_tokens": 0.0,
        "status": ModelStatus.ACTIVE,
        "is_default": False,
        "priority": 60,
        "capabilities": ["chat", "quick_response", "translation"],
        "tags": ["fast", "lightweight"],
    },
    {
        "name": "agi-code",
        "display_name": "AGI-Code",
        "description": "AGI 框架代码专家模型，专注于代码生成和审查",
        "provider": ModelProvider.LOCAL,
        "model_type": ModelType.CODE,
        "version": "1.0",
        "max_tokens": 65536,
        "max_context_length": 65536,
        "supports_streaming": True,
        "supports_function_calling": True,
        "supports_vision": True,
        "supports_json_mode": True,
        "cost_per_1k_input_tokens": 0.0,
        "cost_per_1k_output_tokens": 0.0,
        "status": ModelStatus.ACTIVE,
        "is_default": False,
        "priority": 70,
        "capabilities": ["coding", "code_review", "debugging", "refactoring", "testing"],
        "tags": ["code", "developer"],
    },
    {
        "name": "agi-vision",
        "display_name": "AGI-Vision",
        "description": "AGI 框架视觉理解模型，支持图像分析和描述",
        "provider": ModelProvider.LOCAL,
        "model_type": ModelType.MULTIMODAL,
        "version": "1.0",
        "max_tokens": 32768,
        "max_context_length": 32768,
        "supports_streaming": True,
        "supports_function_calling": False,
        "supports_vision": True,
        "supports_json_mode": True,
        "cost_per_1k_input_tokens": 0.0,
        "cost_per_1k_output_tokens": 0.0,
        "status": ModelStatus.ACTIVE,
        "is_default": False,
        "priority": 50,
        "capabilities": ["image_understanding", "ocr", "image_description", "visual_qa"],
        "tags": ["vision", "multimodal"],
    },
    {
        "name": "agi-embed",
        "display_name": "AGI-Embed",
        "description": "AGI 框架嵌入模型，用于文本向量化和语义搜索",
        "provider": ModelProvider.LOCAL,
        "model_type": ModelType.EMBEDDING,
        "version": "1.0",
        "max_tokens": 8192,
        "supports_streaming": False,
        "supports_function_calling": False,
        "supports_vision": False,
        "supports_json_mode": False,
        "cost_per_1k_input_tokens": 0.0,
        "cost_per_1k_output_tokens": 0.0,
        "status": ModelStatus.ACTIVE,
        "is_default": False,
        "priority": 90,
        "capabilities": ["embedding", "semantic_search", "clustering", "classification"],
        "tags": ["embedding", "vector"],
    },
    {
        "name": "gpt-4o",
        "display_name": "GPT-4o",
        "description": "OpenAI GPT-4o 多模态模型",
        "provider": ModelProvider.OPENAI,
        "model_type": ModelType.MULTIMODAL,
        "version": "2024-05",
        "max_tokens": 128000,
        "max_context_length": 128000,
        "supports_streaming": True,
        "supports_function_calling": True,
        "supports_vision": True,
        "supports_json_mode": True,
        "cost_per_1k_input_tokens": 0.005,
        "cost_per_1k_output_tokens": 0.015,
        "status": ModelStatus.INACTIVE,
        "is_default": False,
        "priority": 40,
        "capabilities": ["reasoning", "coding", "vision", "function_calling"],
        "tags": ["openai", "multimodal", "external"],
    },
    {
        "name": "claude-3-opus",
        "display_name": "Claude 3 Opus",
        "description": "Anthropic Claude 3 Opus 模型",
        "provider": ModelProvider.ANTHROPIC,
        "model_type": ModelType.LLM,
        "version": "2024-02",
        "max_tokens": 200000,
        "max_context_length": 200000,
        "supports_streaming": True,
        "supports_function_calling": True,
        "supports_vision": True,
        "supports_json_mode": True,
        "cost_per_1k_input_tokens": 0.015,
        "cost_per_1k_output_tokens": 0.075,
        "status": ModelStatus.INACTIVE,
        "is_default": False,
        "priority": 35,
        "capabilities": ["reasoning", "coding", "analysis", "creative_writing"],
        "tags": ["anthropic", "external", "long_context"],
    },
]


# ============================================================
# 默认插件数据
# ============================================================

DEFAULT_PLUGINS: List[Dict[str, Any]] = [
    {
        "name": "web-search",
        "display_name": "网络搜索",
        "description": "集成网络搜索功能，支持实时信息检索",
        "author": "AGI Framework Team",
        "version": "1.0.0",
        "category": "tool",
        "status": PluginStatus.ENABLED,
        "permissions": ["network_access"],
        "tags": ["search", "web", "tool"],
    },
    {
        "name": "code-executor",
        "display_name": "代码执行器",
        "description": "安全的沙箱代码执行环境，支持 Python、JavaScript 等语言",
        "author": "AGI Framework Team",
        "version": "1.0.0",
        "category": "tool",
        "status": PluginStatus.ENABLED,
        "permissions": ["sandbox_execute"],
        "tags": ["code", "execution", "sandbox"],
    },
    {
        "name": "file-manager",
        "display_name": "文件管理器",
        "description": "文件上传、下载和管理功能",
        "author": "AGI Framework Team",
        "version": "1.0.0",
        "category": "tool",
        "status": PluginStatus.ENABLED,
        "permissions": ["file_read", "file_write"],
        "tags": ["file", "storage", "management"],
    },
    {
        "name": "image-generator",
        "display_name": "图像生成器",
        "description": "基于 Stable Diffusion 的图像生成插件",
        "author": "AGI Framework Team",
        "version": "1.0.0",
        "category": "generation",
        "status": PluginStatus.ENABLED,
        "permissions": ["gpu_access"],
        "tags": ["image", "generation", "ai"],
    },
    {
        "name": "knowledge-base",
        "display_name": "知识库",
        "description": "RAG 知识库管理，支持文档上传和语义检索",
        "author": "AGI Framework Team",
        "version": "1.0.0",
        "category": "knowledge",
        "status": PluginStatus.ENABLED,
        "permissions": ["vector_store_access"],
        "tags": ["knowledge", "rag", "retrieval"],
    },
    {
        "name": "scheduler",
        "display_name": "任务调度器",
        "description": "定时任务和周期性任务调度管理",
        "author": "AGI Framework Team",
        "version": "1.0.0",
        "category": "system",
        "status": PluginStatus.ENABLED,
        "permissions": ["schedule_access"],
        "tags": ["scheduler", "cron", "automation"],
    },
    {
        "name": "email-sender",
        "display_name": "邮件发送器",
        "description": "邮件发送和模板管理功能",
        "author": "AGI Framework Team",
        "version": "1.0.0",
        "category": "communication",
        "status": PluginStatus.INSTALLED,
        "permissions": ["smtp_access"],
        "tags": ["email", "notification", "communication"],
    },
    {
        "name": "data-visualizer",
        "display_name": "数据可视化",
        "description": "图表生成和数据可视化工具",
        "author": "AGI Framework Team",
        "version": "1.0.0",
        "category": "tool",
        "status": PluginStatus.INSTALLED,
        "permissions": [],
        "tags": ["chart", "visualization", "data"],
    },
    {
        "name": "translator",
        "display_name": "翻译器",
        "description": "多语言翻译插件，支持 100+ 种语言",
        "author": "AGI Framework Team",
        "version": "1.0.0",
        "category": "tool",
        "status": PluginStatus.ENABLED,
        "permissions": [],
        "tags": ["translation", "language", "i18n"],
    },
    {
        "name": "security-scanner",
        "display_name": "安全扫描器",
        "description": "代码安全扫描和漏洞检测",
        "author": "AGI Framework Team",
        "version": "1.0.0",
        "category": "security",
        "status": PluginStatus.INSTALLED,
        "permissions": ["code_read"],
        "tags": ["security", "scan", "vulnerability"],
    },
]


# ============================================================
# 示例工作流模板
# ============================================================

DEFAULT_WORKFLOW_TEMPLATES: List[Dict[str, Any]] = [
    {
        "name": "智能客服工作流",
        "description": "自动化客服流程：意图识别 -> 知识检索 -> 回答生成 -> 质量检查",
        "category": "customer_service",
        "is_template": True,
        "status": WorkflowStatus.ACTIVE,
        "tags": ["客服", "自动化", "RAG"],
        "config_json": {
            "nodes": [
                {"id": "input", "type": "input", "name": "用户输入"},
                {"id": "intent", "type": "llm", "name": "意图识别",
                 "config": {"model": "agi-swift", "task": "intent_classification"}},
                {"id": "retrieve", "type": "rag", "name": "知识检索",
                 "config": {"top_k": 5, "similarity_threshold": 0.7}},
                {"id": "generate", "type": "llm", "name": "回答生成",
                 "config": {"model": "agi-sage", "task": "answer_generation"}},
                {"id": "quality", "type": "llm", "name": "质量检查",
                 "config": {"model": "agi-swift", "task": "quality_check"}},
                {"id": "output", "type": "output", "name": "输出回答"},
            ],
            "edges": [
                {"from": "input", "to": "intent"},
                {"from": "intent", "to": "retrieve"},
                {"from": "retrieve", "to": "generate"},
                {"from": "generate", "to": "quality"},
                {"from": "quality", "to": "output"},
            ],
        },
    },
    {
        "name": "代码审查工作流",
        "description": "自动化代码审查：代码解析 -> 安全扫描 -> 质量评估 -> 生成报告",
        "category": "development",
        "is_template": True,
        "status": WorkflowStatus.ACTIVE,
        "tags": ["代码", "审查", "安全"],
        "config_json": {
            "nodes": [
                {"id": "input", "type": "input", "name": "代码输入"},
                {"id": "parse", "type": "code_parser", "name": "代码解析"},
                {"id": "security", "type": "security_scan", "name": "安全扫描"},
                {"id": "quality", "type": "llm", "name": "质量评估",
                 "config": {"model": "agi-code", "task": "code_review"}},
                {"id": "report", "type": "report_generator", "name": "生成报告"},
                {"id": "output", "type": "output", "name": "输出报告"},
            ],
            "edges": [
                {"from": "input", "to": "parse"},
                {"from": "parse", "to": "security"},
                {"from": "parse", "to": "quality"},
                {"from": "security", "to": "report"},
                {"from": "quality", "to": "report"},
                {"from": "report", "to": "output"},
            ],
        },
    },
    {
        "name": "内容创作工作流",
        "description": "多步骤内容创作：主题分析 -> 大纲生成 -> 内容撰写 -> SEO优化 -> 校对",
        "category": "content",
        "is_template": True,
        "status": WorkflowStatus.ACTIVE,
        "tags": ["内容", "创作", "写作"],
        "config_json": {
            "nodes": [
                {"id": "input", "type": "input", "name": "创作需求"},
                {"id": "analyze", "type": "llm", "name": "主题分析",
                 "config": {"model": "agi-sage", "task": "topic_analysis"}},
                {"id": "outline", "type": "llm", "name": "大纲生成",
                 "config": {"model": "agi-ultra", "task": "outline_generation"}},
                {"id": "write", "type": "llm", "name": "内容撰写",
                 "config": {"model": "agi-ultra", "task": "content_writing"}},
                {"id": "seo", "type": "llm", "name": "SEO优化",
                 "config": {"model": "agi-sage", "task": "seo_optimization"}},
                {"id": "proofread", "type": "llm", "name": "校对润色",
                 "config": {"model": "agi-sage", "task": "proofreading"}},
                {"id": "output", "type": "output", "name": "输出内容"},
            ],
            "edges": [
                {"from": "input", "to": "analyze"},
                {"from": "analyze", "to": "outline"},
                {"from": "outline", "to": "write"},
                {"from": "write", "to": "seo"},
                {"from": "seo", "to": "proofread"},
                {"from": "proofread", "to": "output"},
            ],
        },
    },
    {
        "name": "数据分析工作流",
        "description": "自动化数据分析：数据加载 -> 清洗 -> 分析 -> 可视化 -> 报告",
        "category": "analytics",
        "is_template": True,
        "status": WorkflowStatus.ACTIVE,
        "tags": ["数据", "分析", "可视化"],
        "config_json": {
            "nodes": [
                {"id": "input", "type": "input", "name": "数据源"},
                {"id": "load", "type": "data_loader", "name": "数据加载"},
                {"id": "clean", "type": "data_processor", "name": "数据清洗"},
                {"id": "analyze", "type": "code_executor", "name": "数据分析"},
                {"id": "visualize", "type": "chart_generator", "name": "可视化"},
                {"id": "report", "type": "llm", "name": "生成报告",
                 "config": {"model": "agi-sage", "task": "data_report"}},
                {"id": "output", "type": "output", "name": "输出报告"},
            ],
            "edges": [
                {"from": "input", "to": "load"},
                {"from": "load", "to": "clean"},
                {"from": "clean", "to": "analyze"},
                {"from": "analyze", "to": "visualize"},
                {"from": "analyze", "to": "report"},
                {"from": "visualize", "to": "output"},
                {"from": "report", "to": "output"},
            ],
        },
    },
    {
        "name": "多智能体协作工作流",
        "description": "多智能体协作解决问题：任务分解 -> 分配 -> 并行执行 -> 结果整合",
        "category": "multiagent",
        "is_template": True,
        "status": WorkflowStatus.DRAFT,
        "tags": ["智能体", "协作", "并行"],
        "config_json": {
            "nodes": [
                {"id": "input", "type": "input", "name": "任务输入"},
                {"id": "decompose", "type": "llm", "name": "任务分解",
                 "config": {"model": "agi-ultra", "task": "task_decomposition"}},
                {"id": "allocate", "type": "task_allocator", "name": "任务分配"},
                {"id": "agent1", "type": "agent", "name": "研究智能体"},
                {"id": "agent2", "type": "agent", "name": "分析智能体"},
                {"id": "agent3", "type": "agent", "name": "创作智能体"},
                {"id": "integrate", "type": "llm", "name": "结果整合",
                 "config": {"model": "agi-ultra", "task": "result_integration"}},
                {"id": "output", "type": "output", "name": "最终输出"},
            ],
            "edges": [
                {"from": "input", "to": "decompose"},
                {"from": "decompose", "to": "allocate"},
                {"from": "allocate", "to": "agent1"},
                {"from": "allocate", "to": "agent2"},
                {"from": "allocate", "to": "agent3"},
                {"from": "agent1", "to": "integrate"},
                {"from": "agent2", "to": "integrate"},
                {"from": "agent3", "to": "integrate"},
                {"from": "integrate", "to": "output"},
            ],
        },
    },
]


# ============================================================
# 示例人格模板
# ============================================================

DEFAULT_PERSONALITY_TEMPLATES: List[Dict[str, Any]] = [
    {
        "name": "专业助手",
        "description": "专业、准确、高效的 AI 助手人格",
        "is_template": True,
        "is_public": True,
        "is_active": True,
        "category": "assistant",
        "tags": ["专业", "助手", "通用"],
        "system_prompt": (
            "你是一位专业的 AI 助手。你的回答应该准确、简洁、有条理。"
            "你善于分析问题并提供实用的解决方案。"
            "当你不确定答案时，会诚实地说明。"
            "你使用正式但友好的语气。"
        ),
        "greeting": "你好！我是你的专业 AI 助手。有什么我可以帮助你的吗？",
        "personality_traits": {
            "professionalism": 0.9,
            "friendliness": 0.7,
            "creativity": 0.5,
            "humor": 0.3,
            "formality": 0.8,
        },
        "speaking_style": {
            "tone": "professional",
            "verbosity": "concise",
            "emoji_usage": "minimal",
            "language_style": "formal",
        },
    },
    {
        "name": "创意伙伴",
        "description": "富有创造力和想象力的 AI 创作伙伴",
        "is_template": True,
        "is_public": True,
        "is_active": True,
        "category": "creative",
        "tags": ["创意", "写作", "头脑风暴"],
        "system_prompt": (
            "你是一位富有创造力的 AI 创作伙伴。你善于头脑风暴、"
            "提供新颖的视角和创意方案。你的回答充满想象力，"
            "同时保持逻辑性。你喜欢用生动的语言和比喻来表达想法。"
            "你鼓励用户发挥创造力，并提供有建设性的反馈。"
        ),
        "greeting": "嗨！我是你的创意伙伴。让我们一起探索无限可能吧！",
        "personality_traits": {
            "professionalism": 0.5,
            "friendliness": 0.9,
            "creativity": 0.95,
            "humor": 0.7,
            "formality": 0.3,
        },
        "speaking_style": {
            "tone": "enthusiastic",
            "verbosity": "expressive",
            "emoji_usage": "moderate",
            "language_style": "casual",
        },
    },
    {
        "name": "编程导师",
        "description": "耐心细致的编程教学助手",
        "is_template": True,
        "is_public": True,
        "is_active": True,
        "category": "education",
        "tags": ["编程", "教学", "技术"],
        "system_prompt": (
            "你是一位经验丰富的编程导师。你善于用简单易懂的方式解释"
            "复杂的编程概念。你总是先理解学生的问题，然后提供循序渐进的指导。"
            "你会给出代码示例并详细解释每一行的作用。"
            "你鼓励学生独立思考，而不是直接给出答案。"
        ),
        "greeting": "你好！我是你的编程导师。无论你是初学者还是经验丰富的开发者，我都可以帮助你提升编程技能。",
        "personality_traits": {
            "professionalism": 0.8,
            "friendliness": 0.8,
            "creativity": 0.6,
            "humor": 0.4,
            "formality": 0.5,
            "patience": 0.95,
        },
        "speaking_style": {
            "tone": "patient",
            "verbosity": "detailed",
            "emoji_usage": "minimal",
            "language_style": "educational",
        },
    },
    {
        "name": "数据分析师",
        "description": "擅长数据分析和可视化的 AI 专家",
        "is_template": True,
        "is_public": True,
        "is_active": True,
        "category": "analytics",
        "tags": ["数据", "分析", "统计"],
        "system_prompt": (
            "你是一位专业的数据分析师。你善于从数据中发现洞察和趋势。"
            "你的分析严谨、有逻辑，并善于用清晰的方式呈现结果。"
            "你精通统计学、机器学习和数据可视化。"
            "你会主动指出数据中的异常和潜在问题。"
        ),
        "greeting": "你好！我是你的数据分析师。让我帮你从数据中发现有价值的洞察。",
        "personality_traits": {
            "professionalism": 0.95,
            "friendliness": 0.6,
            "creativity": 0.5,
            "humor": 0.2,
            "formality": 0.8,
            "analytical": 0.95,
        },
        "speaking_style": {
            "tone": "analytical",
            "verbosity": "data_driven",
            "emoji_usage": "minimal",
            "language_style": "technical",
        },
    },
    {
        "name": "心理咨询师",
        "description": "温暖、共情的 AI 心理支持助手",
        "is_template": True,
        "is_public": True,
        "is_active": True,
        "category": "wellness",
        "tags": ["心理", "情感", "支持"],
        "system_prompt": (
            "你是一位温暖、有共情能力的 AI 心理支持助手。"
            "你善于倾听和理解他人的感受，提供情感支持。"
            "你不会评判他人，而是以接纳和尊重的态度对待每一个人。"
            "你会在适当的时候引导用户进行自我反思，"
            "但不会替代专业的心理咨询。"
            "如果用户表现出严重的心理问题，你会建议寻求专业帮助。"
        ),
        "greeting": "你好，我在这里倾听你。无论你正在经历什么，都可以和我分享。",
        "personality_traits": {
            "professionalism": 0.7,
            "friendliness": 0.95,
            "creativity": 0.4,
            "humor": 0.3,
            "formality": 0.3,
            "empathy": 0.95,
            "patience": 0.95,
        },
        "speaking_style": {
            "tone": "warm",
            "verbosity": "supportive",
            "emoji_usage": "minimal",
            "language_style": "gentle",
        },
    },
    {
        "name": "科研助手",
        "description": "严谨的学术研究辅助工具",
        "is_template": True,
        "is_public": True,
        "is_active": True,
        "category": "research",
        "tags": ["科研", "学术", "论文"],
        "system_prompt": (
            "你是一位严谨的学术研究助手。你熟悉各学科的研究方法和前沿动态。"
            "你善于文献综述、研究设计和数据分析。"
            "你的回答基于科学证据，会标注引用来源。"
            "你注重学术诚信，会指出研究中的局限性。"
        ),
        "greeting": "你好！我是你的科研助手。让我帮你进行文献检索、研究设计和学术写作。",
        "personality_traits": {
            "professionalism": 0.95,
            "friendliness": 0.5,
            "creativity": 0.6,
            "humor": 0.1,
            "formality": 0.9,
            "rigor": 0.95,
        },
        "speaking_style": {
            "tone": "academic",
            "verbosity": "thorough",
            "emoji_usage": "none",
            "language_style": "scholarly",
        },
    },
]


# ============================================================
# 示例 FAQ 数据
# ============================================================

DEFAULT_FAQ_DATA: List[Dict[str, str]] = [
    {
        "question": "如何开始使用 AGI Unified Framework？",
        "answer": (
            "1. 安装依赖：pip install -r requirements.txt\n"
            "2. 初始化数据库：python -m database.seed_data\n"
            "3. 启动服务：python -m api.main\n"
            "4. 访问 Web UI：http://localhost:8000\n"
            "详细文档请参考 docs/ 目录。"
        ),
    },
    {
        "question": "支持哪些 AI 模型？",
        "answer": (
            "AGI Unified Framework 支持多种 AI 模型：\n"
            "- 本地模型：AGI-Ultra, AGI-Sage, AGI-Swift, AGI-Code, AGI-Vision\n"
            "- OpenAI：GPT-4o, GPT-4, GPT-3.5\n"
            "- Anthropic：Claude 3 Opus, Claude 3 Sonnet\n"
            "- 国产模型：智谱 GLM, 百度文心, 阿里通义, 腾讯混元\n"
            "您可以在模型管理页面添加和配置自定义模型。"
        ),
    },
    {
        "question": "如何训练自己的模型？",
        "answer": (
            "1. 准备训练数据集（支持多种格式）\n"
            "2. 在训练页面创建新的训练任务\n"
            "3. 配置超参数（学习率、批次大小、轮次等）\n"
            "4. 选择基础模型和训练策略\n"
            "5. 启动训练并监控进度\n"
            "6. 训练完成后可以下载模型或直接部署"
        ),
    },
    {
        "question": "如何创建智能体？",
        "answer": (
            "1. 进入智能体管理页面\n"
            "2. 点击「创建智能体」\n"
            "3. 选择模板或从零开始\n"
            "4. 配置智能体名称、描述、系统提示词\n"
            "5. 选择底层模型和能力\n"
            "6. 配置工具和知识库\n"
            "7. 测试并发布智能体"
        ),
    },
    {
        "question": "如何设置自动化工作流？",
        "answer": (
            "1. 进入工作流编辑器\n"
            "2. 选择模板或创建空白工作流\n"
            "3. 拖拽添加节点（LLM、工具、条件等）\n"
            "4. 连接节点定义执行流程\n"
            "5. 配置每个节点的参数\n"
            "6. 设置触发条件（手动/定时/事件）\n"
            "7. 测试并启用工作流"
        ),
    },
    {
        "question": "数据安全如何保障？",
        "answer": (
            "AGI Unified Framework 采取多重安全措施：\n"
            "- 所有密码使用 bcrypt 哈希存储\n"
            "- API 密钥使用 Fernet 对称加密\n"
            "- 支持 HTTPS 和 TLS 加密通信\n"
            "- 完善的权限控制和审计日志\n"
            "- 数据防泄漏（DLP）和内容安全检测\n"
            "- 沙箱隔离的代码执行环境"
        ),
    },
]


# ============================================================
# 种子数据管理器
# ============================================================

class SeedDataManager:
    """
    种子数据管理器

    管理数据库种子数据的插入和更新。

    属性:
        db: 数据库会话

    示例:
        >>> manager = SeedDataManager(db_session)
        >>> manager.seed_all()
        >>> manager.seed_system_settings()
    """

    def __init__(self, db: "Session") -> None:
        """
        初始化种子数据管理器

        参数:
            db: 数据库会话
        """
        self.db = db

    def seed_all(
        self,
        include_settings: bool = True,
        include_models: bool = True,
        include_plugins: bool = True,
        include_workflows: bool = True,
        include_personalities: bool = True,
        include_admin_user: bool = True,
    ) -> Dict[str, int]:
        """
        填充所有种子数据

        参数:
            include_settings: 是否填充系统设置
            include_models: 是否填充模型配置
            include_plugins: 是否填充插件列表
            include_workflows: 是否填充工作流模板
            include_personalities: 是否填充人格模板
            include_admin_user: 是否创建管理员用户

        返回:
            各类数据的填充数量统计
        """
        stats: Dict[str, int] = {}

        if include_settings:
            stats["settings"] = self.seed_system_settings()
        if include_models:
            stats["models"] = self.seed_default_models()
        if include_plugins:
            stats["plugins"] = self.seed_default_plugins()
        if include_workflows:
            stats["workflows"] = self.seed_workflow_templates()
        if include_personalities:
            stats["personalities"] = self.seed_personality_templates()
        if include_admin_user:
            stats["admin_user"] = self.seed_admin_user()

        logger.info(f"种子数据填充完成: {stats}")
        return stats

    def seed_system_settings(self) -> int:
        """
        填充默认系统设置

        返回:
            插入的设置数量
        """
        count = 0
        for setting_data in DEFAULT_SYSTEM_SETTINGS:
            try:
                existing = (
                    self.db.query(SystemSetting)
                    .filter(SystemSetting.key == setting_data["key"])
                    .first()
                )
                if not existing:
                    setting = SystemSetting(**setting_data)
                    self.db.add(setting)
                    count += 1
            except SQLAlchemyError as e:
                logger.warning(f"插入系统设置 {setting_data['key']} 失败: {e}")

        if count > 0:
            self.db.commit()
            logger.info(f"系统设置填充完成: {count} 项")
        return count

    def seed_default_models(self) -> int:
        """
        填充默认模型配置

        返回:
            插入的模型数量
        """
        count = 0
        for model_data in DEFAULT_MODELS:
            try:
                existing = (
                    self.db.query(Model)
                    .filter(Model.name == model_data["name"])
                    .first()
                )
                if not existing:
                    model = Model(**model_data)
                    self.db.add(model)
                    count += 1
            except SQLAlchemyError as e:
                logger.warning(f"插入模型 {model_data['name']} 失败: {e}")

        if count > 0:
            self.db.commit()
            logger.info(f"默认模型填充完成: {count} 个")
        return count

    def seed_default_plugins(self) -> int:
        """
        填充默认插件列表

        返回:
            插入的插件数量
        """
        count = 0
        for plugin_data in DEFAULT_PLUGINS:
            try:
                existing = (
                    self.db.query(Plugin)
                    .filter(Plugin.name == plugin_data["name"])
                    .first()
                )
                if not existing:
                    plugin = Plugin(**plugin_data)
                    self.db.add(plugin)
                    count += 1
            except SQLAlchemyError as e:
                logger.warning(f"插入插件 {plugin_data['name']} 失败: {e}")

        if count > 0:
            self.db.commit()
            logger.info(f"默认插件填充完成: {count} 个")
        return count

    def seed_workflow_templates(self) -> int:
        """
        填充工作流模板

        返回:
            插入的工作流数量
        """
        count = 0
        for workflow_data in DEFAULT_WORKFLOW_TEMPLATES:
            try:
                existing = (
                    self.db.query(Workflow)
                    .filter(
                        Workflow.name == workflow_data["name"],
                        Workflow.is_template == True,  # noqa: E712
                    )
                    .first()
                )
                if not existing:
                    workflow = Workflow(**workflow_data)
                    self.db.add(workflow)
                    count += 1
            except SQLAlchemyError as e:
                logger.warning(f"插入工作流 {workflow_data['name']} 失败: {e}")

        if count > 0:
            self.db.commit()
            logger.info(f"工作流模板填充完成: {count} 个")
        return count

    def seed_personality_templates(self) -> int:
        """
        填充人格模板

        返回:
            插入的人格数量
        """
        count = 0
        for personality_data in DEFAULT_PERSONALITY_TEMPLATES:
            try:
                existing = (
                    self.db.query(Personality)
                    .filter(
                        Personality.name == personality_data["name"],
                        Personality.is_template == True,  # noqa: E712
                    )
                    .first()
                )
                if not existing:
                    personality = Personality(**personality_data)
                    self.db.add(personality)
                    count += 1
            except SQLAlchemyError as e:
                logger.warning(f"插入人格 {personality_data['name']} 失败: {e}")

        if count > 0:
            self.db.commit()
            logger.info(f"人格模板填充完成: {count} 个")
        return count

    def seed_admin_user(self) -> int:
        """
        创建默认管理员用户

        默认用户名: admin
        默认密码: admin123
        创建后会自动创建关联的用户设置。

        返回:
            创建的用户数量
        """
        try:
            existing = (
                self.db.query(User)
                .filter(User.username == "admin")
                .first()
            )
            if existing:
                logger.info("管理员用户已存在，跳过创建")
                return 0

            admin = User(
                username="admin",
                email="admin@agi-framework.local",
                password_hash=hash_password("admin123"),
                role=UserRole.SUPER_ADMIN,
                display_name="系统管理员",
                is_active=True,
                is_verified=True,
            )
            self.db.add(admin)
            self.db.flush()

            # 创建管理员设置
            settings = UserSettings(user_id=admin.id)
            self.db.add(settings)

            self.db.commit()
            logger.info("默认管理员用户创建成功 (用户名: admin, 密码: admin123)")
            logger.warning("请立即修改默认管理员密码！")
            return 1
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"创建管理员用户失败: {e}")
            return 0

    def seed_sample_conversations(self, user_id: int = 1) -> int:
        """
        填充示例对话数据

        参数:
            user_id: 关联的用户 ID

        返回:
            创建的对话数量
        """
        sample_conversations = [
            {
                "title": "AGI 框架介绍",
                "model_name": "agi-ultra",
                "messages": [
                    {"role": "user", "content": "请介绍一下 AGI Unified Framework 的主要功能。"},
                    {"role": "assistant", "content": (
                        "AGI Unified Framework 是一个功能丰富的通用人工智能框架，主要功能包括：\n\n"
                        "1. **多模型对话** - 支持多种 AI 模型的对话交互\n"
                        "2. **模型训练** - 提供完整的模型训练和微调能力\n"
                        "3. **内容生成** - 支持图像、视频、音频、3D 等多模态生成\n"
                        "4. **智能体系统** - 创建和管理自定义 AI 智能体\n"
                        "5. **工作流自动化** - 可视化工作流编辑和执行\n"
                        "6. **插件生态** - 可扩展的插件系统\n"
                        "7. **多渠道集成** - 支持 Discord、Telegram、Slack 等\n"
                        "8. **联邦学习** - 分布式模型训练和协作"
                    )},
                ],
            },
            {
                "title": "Python 编程帮助",
                "model_name": "agi-code",
                "messages": [
                    {"role": "user", "content": "请用 Python 实现一个简单的 LRU 缓存。"},
                    {"role": "assistant", "content": (
                        "以下是一个使用 Python 实现的 LRU 缓存：\n\n"
                        "```python\n"
                        "from collections import OrderedDict\n\n"
                        "class LRUCache:\n"
                        "    def __init__(self, capacity: int):\n"
                        "        self.cache = OrderedDict()\n"
                        "        self.capacity = capacity\n\n"
                        "    def get(self, key: int) -> int:\n"
                        "        if key not in self.cache:\n"
                        "            return -1\n"
                        "        self.cache.move_to_end(key)\n"
                        "        return self.cache[key]\n\n"
                        "    def put(self, key: int, value: int) -> None:\n"
                        "        if key in self.cache:\n"
                        "            self.cache.move_to_end(key)\n"
                        "        self.cache[key] = value\n"
                        "        if len(self.cache) > self.capacity:\n"
                        "            self.cache.popitem(last=False)\n"
                        "```\n\n"
                        "这个实现使用了 OrderedDict，get 和 put 操作的时间复杂度都是 O(1)。"
                    )},
                ],
            },
        ]

        count = 0
        for conv_data in sample_conversations:
            try:
                conversation = Conversation(
                    user_id=user_id,
                    title=conv_data["title"],
                    model_name=conv_data["model_name"],
                )
                self.db.add(conversation)
                self.db.flush()

                for msg_data in conv_data["messages"]:
                    message = Message(
                        conversation_id=conversation.id,
                        role=msg_data["role"],
                        content=msg_data["content"],
                    )
                    self.db.add(message)

                conversation.message_count = len(conv_data["messages"])
                count += 1
            except SQLAlchemyError as e:
                logger.warning(f"创建示例对话失败: {e}")

        if count > 0:
            self.db.commit()
            logger.info(f"示例对话填充完成: {count} 个")
        return count


# ============================================================
# 便捷函数
# ============================================================

def seed_all(db: "Session") -> Dict[str, int]:
    """
    填充所有种子数据

    参数:
        db: 数据库会话

    返回:
        填充统计

    示例:
        >>> from database.connection import DatabaseManager
        >>> from database.models import Base
        >>> from database.seed_data import seed_all
        >>> manager = DatabaseManager("sqlite:///./data/app.db")
        >>> manager.initialize()
        >>> manager.create_tables(Base)
        >>> with manager.session_scope() as session:
        ...     stats = seed_all(session)
    """
    manager = SeedDataManager(db)
    return manager.seed_all()


def seed_system_settings(db: "Session") -> int:
    """填充系统设置"""
    manager = SeedDataManager(db)
    return manager.seed_system_settings()


def seed_default_models(db: "Session") -> int:
    """填充默认模型"""
    manager = SeedDataManager(db)
    return manager.seed_default_models()


def seed_default_plugins(db: "Session") -> int:
    """填充默认插件"""
    manager = SeedDataManager(db)
    return manager.seed_default_plugins()


def seed_workflow_templates(db: "Session") -> int:
    """填充工作流模板"""
    manager = SeedDataManager(db)
    return manager.seed_workflow_templates()


def seed_personality_templates(db: "Session") -> int:
    """填充人格模板"""
    manager = SeedDataManager(db)
    return manager.seed_personality_templates()


def seed_sample_data(db: "Session") -> Dict[str, int]:
    """填充示例数据（对话、FAQ 等）"""
    manager = SeedDataManager(db)
    stats: Dict[str, int] = {}
    stats["conversations"] = manager.seed_sample_conversations()
    return stats
