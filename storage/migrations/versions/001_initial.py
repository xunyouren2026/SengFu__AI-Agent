"""
初始迁移脚本

创建所有数据表

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """升级数据库"""
    
    # 创建人格表
    op.create_table(
        'personalities',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False, index=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('avatar_url', sa.String(512), nullable=True),
        sa.Column('soul_md_content', sa.Text(), nullable=False),
        sa.Column('soul_md_version', sa.String(50), nullable=False, default='1.0.0'),
        sa.Column('soul_md_hash', sa.String(64), nullable=False, index=True),
        sa.Column('personality_type', sa.Enum('DEFAULT', 'CUSTOM', 'TEMPLATE', 'CLONE', name='personalitytype'), nullable=False),
        sa.Column('status', sa.Enum('DRAFT', 'ACTIVE', 'ARCHIVED', 'DEPRECATED', name='personalitystatus'), nullable=False, index=True),
        sa.Column('is_template', sa.Boolean(), default=False, nullable=False),
        sa.Column('template_id', sa.String(36), sa.ForeignKey('personalities.id', ondelete='SET NULL'), nullable=True, index=True),
        sa.Column('current_version', sa.String(50), nullable=False, default='1.0.0'),
        sa.Column('version_count', sa.Integer(), default=1, nullable=False),
        sa.Column('created_by', sa.String(255), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('metadata', sa.JSON(), default=dict, nullable=False),
        sa.Column('tags', postgresql.ARRAY(sa.String()), default=list, nullable=False),
    )
    
    # 创建人格版本表
    op.create_table(
        'personality_versions',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('personality_id', sa.String(36), sa.ForeignKey('personalities.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('soul_md_content', sa.Text(), nullable=False),
        sa.Column('soul_md_hash', sa.String(64), nullable=False),
        sa.Column('change_summary', sa.String(500), nullable=True),
        sa.Column('change_details', sa.JSON(), default=dict, nullable=False),
        sa.Column('created_by', sa.String(255), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint('personality_id', 'version', name='uq_personality_version'),
    )
    
    # 创建人格模板表
    op.create_table(
        'personality_templates',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False, unique=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('category', sa.String(100), nullable=False, index=True),
        sa.Column('soul_md_template', sa.Text(), nullable=False),
        sa.Column('default_traits', sa.JSON(), default=dict, nullable=False),
        sa.Column('popularity', sa.Integer(), default=0, nullable=False),
        sa.Column('is_official', sa.Boolean(), default=False, nullable=False),
        sa.Column('is_active', sa.Boolean(), default=True, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )
    
    # 创建渠道表
    op.create_table(
        'channels',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False, index=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('channel_type', sa.Enum('WECHAT', 'WECHAT_WORK', 'DINGTALK', 'SLACK', 'DISCORD', 'TELEGRAM', 'WEBHOOK', 'API', 'EMAIL', 'SMS', 'FEISHU', 'LARK', 'TEAMS', name='channeltype'), nullable=False, index=True),
        sa.Column('status', sa.Enum('ACTIVE', 'INACTIVE', 'PENDING', 'ERROR', 'SUSPENDED', 'DELETED', name='channelstatus'), nullable=False, index=True),
        sa.Column('webhook_url', sa.String(1024), nullable=True),
        sa.Column('callback_url', sa.String(1024), nullable=True),
        sa.Column('api_endpoint', sa.String(1024), nullable=True),
        sa.Column('credential_id', sa.String(36), nullable=True),
        sa.Column('config_id', sa.String(36), nullable=True),
        sa.Column('rate_limit_per_minute', sa.Integer(), default=60, nullable=False),
        sa.Column('rate_limit_per_hour', sa.Integer(), default=1000, nullable=False),
        sa.Column('retry_max_attempts', sa.Integer(), default=3, nullable=False),
        sa.Column('retry_delay_seconds', sa.Integer(), default=5, nullable=False),
        sa.Column('total_messages_sent', sa.BigInteger(), default=0, nullable=False),
        sa.Column('total_messages_received', sa.BigInteger(), default=0, nullable=False),
        sa.Column('last_message_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error_count', sa.Integer(), default=0, nullable=False),
        sa.Column('last_error_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_error_message', sa.Text(), nullable=True),
        sa.Column('created_by', sa.String(255), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('metadata', sa.JSON(), default=dict, nullable=False),
        sa.Column('tags', sa.JSON(), default=list, nullable=False),
    )
    
    # 创建渠道配置表
    op.create_table(
        'channel_configs',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('channel_id', sa.String(36), sa.ForeignKey('channels.id', ondelete='CASCADE'), nullable=True, unique=True, index=True),
        sa.Column('config_json', sa.JSON(), default=dict, nullable=False),
        sa.Column('default_personality_id', sa.String(36), nullable=True),
        sa.Column('welcome_message', sa.Text(), nullable=True),
        sa.Column('auto_reply_enabled', sa.Boolean(), default=True, nullable=False),
        sa.Column('moderation_enabled', sa.Boolean(), default=False, nullable=False),
        sa.Column('max_message_length', sa.Integer(), default=2000, nullable=False),
        sa.Column('message_timeout_seconds', sa.Integer(), default=30, nullable=False),
        sa.Column('typing_indicator_enabled', sa.Boolean(), default=True, nullable=False),
        sa.Column('custom_headers', sa.JSON(), default=dict, nullable=False),
        sa.Column('timeout_config', sa.JSON(), default=dict, nullable=False),
        sa.Column('proxy_config', sa.JSON(), default=dict, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )
    
    # 创建渠道认证信息表
    op.create_table(
        'channel_credentials',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('channel_id', sa.String(36), sa.ForeignKey('channels.id', ondelete='CASCADE'), nullable=True, unique=True, index=True),
        sa.Column('api_key_encrypted', sa.LargeBinary(), nullable=True),
        sa.Column('api_secret_encrypted', sa.LargeBinary(), nullable=True),
        sa.Column('access_token_encrypted', sa.LargeBinary(), nullable=True),
        sa.Column('refresh_token_encrypted', sa.LargeBinary(), nullable=True),
        sa.Column('app_id', sa.String(255), nullable=True),
        sa.Column('client_id', sa.String(255), nullable=True),
        sa.Column('token_expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('token_scope', sa.String(500), nullable=True),
        sa.Column('encryption_version', sa.String(20), default='v1', nullable=False),
        sa.Column('encryption_key_id', sa.String(100), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )
    
    # 创建消息表
    op.create_table(
        'messages',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('message_id', sa.String(255), nullable=True, index=True),
        sa.Column('session_id', sa.String(36), sa.ForeignKey('sessions.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('channel_id', sa.String(36), sa.ForeignKey('channels.id', ondelete='SET NULL'), nullable=True, index=True),
        sa.Column('direction', sa.Enum('INBOUND', 'OUTBOUND', name='messagedirection'), nullable=False, index=True),
        sa.Column('message_type', sa.Enum('TEXT', 'IMAGE', 'AUDIO', 'VIDEO', 'FILE', 'LOCATION', 'LINK', 'CARD', 'TEMPLATE', 'SYSTEM', 'EVENT', 'COMMAND', 'RICH_TEXT', 'MARKDOWN', name='messagetype'), nullable=False),
        sa.Column('status', sa.Enum('PENDING', 'SENDING', 'SENT', 'DELIVERED', 'READ', 'FAILED', 'CANCELLED', 'DELETED', 'RECALLED', name='messagestatus'), nullable=False, index=True),
        sa.Column('sender_id', sa.String(255), nullable=False, index=True),
        sa.Column('sender_name', sa.String(255), nullable=True),
        sa.Column('sender_avatar', sa.String(512), nullable=True),
        sa.Column('receiver_id', sa.String(255), nullable=True, index=True),
        sa.Column('receiver_name', sa.String(255), nullable=True),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('content_html', sa.Text(), nullable=True),
        sa.Column('content_metadata', sa.JSON(), default=dict, nullable=False),
        sa.Column('parent_message_id', sa.String(36), sa.ForeignKey('messages.id', ondelete='SET NULL'), nullable=True, index=True),
        sa.Column('reply_to_message_id', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('sent_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('delivered_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('read_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('processed_by', sa.String(255), nullable=True),
        sa.Column('processing_time_ms', sa.Integer(), nullable=True),
        sa.Column('error_code', sa.String(100), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('metadata', sa.JSON(), default=dict, nullable=False),
        sa.Column('tags', postgresql.ARRAY(sa.String()), default=list, nullable=False),
    )
    
    # 创建会话表
    op.create_table(
        'sessions',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('session_id', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('user_id', sa.String(255), nullable=False, index=True),
        sa.Column('channel_id', sa.String(36), sa.ForeignKey('channels.id', ondelete='SET NULL'), nullable=True, index=True),
        sa.Column('personality_id', sa.String(36), sa.ForeignKey('personalities.id', ondelete='SET NULL'), nullable=True, index=True),
        sa.Column('status', sa.Enum('ACTIVE', 'IDLE', 'PAUSED', 'CLOSED', 'EXPIRED', 'ARCHIVED', name='sessionstatus'), nullable=False, index=True),
        sa.Column('priority', sa.Enum('LOW', 'NORMAL', 'HIGH', 'URGENT', name='sessionpriority'), nullable=False),
        sa.Column('title', sa.String(500), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('message_count', sa.Integer(), default=0, nullable=False),
        sa.Column('unread_count', sa.Integer(), default=0, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('last_message_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('closed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration_seconds', sa.Integer(), nullable=True),
        sa.Column('source_ip', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('locale', sa.String(10), nullable=True),
        sa.Column('metadata', sa.JSON(), default=dict, nullable=False),
        sa.Column('tags', sa.JSON(), default=list, nullable=False),
        sa.Column('custom_fields', sa.JSON(), default=dict, nullable=False),
    )
    
    # 创建索引
    op.create_index('ix_personalities_status_type', 'personalities', ['status', 'personality_type'])
    op.create_index('ix_channels_status_type', 'channels', ['status', 'channel_type'])
    op.create_index('ix_messages_session_created', 'messages', ['session_id', 'created_at'])
    op.create_index('ix_messages_channel_created', 'messages', ['channel_id', 'created_at'])


def downgrade() -> None:
    """降级数据库"""
    
    # 删除表（反向顺序）
    op.drop_table('messages')
    op.drop_table('channel_credentials')
    op.drop_table('channel_configs')
    op.drop_table('channels')
    op.drop_table('personality_versions')
    op.drop_table('personality_templates')
    op.drop_table('personalities')
    op.drop_table('sessions')
    
    # 删除枚举类型
    op.execute('DROP TYPE IF EXISTS personalitytype')
    op.execute('DROP TYPE IF EXISTS personalitystatus')
    op.execute('DROP TYPE IF EXISTS channeltype')
    op.execute('DROP TYPE IF EXISTS channelstatus')
    op.execute('DROP TYPE IF EXISTS messagetype')
    op.execute('DROP TYPE IF EXISTS messagestatus')
    op.execute('DROP TYPE IF EXISTS messagedirection')
    op.execute('DROP TYPE IF EXISTS sessionstatus')
    op.execute('DROP TYPE IF EXISTS sessionpriority')
